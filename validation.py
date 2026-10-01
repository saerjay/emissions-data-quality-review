"""
Validation core: schema rules, intensity ratios, synthetic errors and the
EmissionsDataValidator. Kept free of Streamlit so the evaluation harness, the
pipeline and the tests can import it without starting a UI.
"""
from datetime import date

import numpy as np
import pandas as pd
import pandera.pandas as pa
import shap
from sklearn.ensemble import IsolationForest

from transforms import FITTED, PARAMETER_FREE, apply_transform, fit_transform_params

FALLBACK_TRANSFORM = 'signed_log'
ID_COLS = ['country', 'year', 'iso_code']

# ---------------------------------------------------------------------------
# 1. Schema definition
# ---------------------------------------------------------------------------
# Every rule carries a plain-English reason so each flagged value can be explained.
# Features are looked up here by name because data_prep.py selects them dynamically.
FEATURE_RULES = {
    'population': [
        (pa.Check.gt(0, error='population > 0'),
         "A country cannot have zero or negative population."),
    ],
    'gdp': [
        (pa.Check.gt(0, error='gdp > 0'),
         "GDP is an absolute output measure and must be positive."),
    ],
    'co2': [
        (pa.Check.ge(0, error='emissions >= 0'),
         "Fossil CO2 emissions (Mt) cannot be negative."),
    ],
    'methane': [
        (pa.Check.ge(0, error='emissions >= 0'),
         "Methane emissions (Mt CO2e) cannot be negative."),
    ],
    'nitrous_oxide': [
        (pa.Check.ge(0, error='emissions >= 0'),
         "Nitrous oxide emissions (Mt CO2e) cannot be negative."),
    ],
    'primary_energy_consumption': [
        (pa.Check.ge(0, error='energy >= 0'),
         "Energy consumption (TWh) cannot be negative."),
    ],
    'ghg_per_capita': [
        # Net GHG includes land-use change, so small negatives are real (carbon-sink
        # countries, e.g. Samoa 2008 = -0.018). The observed range is about -0.02 to 106 t.
        (pa.Check.in_range(-50, 150, error='-50 <= ghg_per_capita <= 150'),
         "Net GHG per person outside [-50, 150] t CO2e; the real-world maximum is about 106 t."),
    ],
}
DEFAULT_RULE = (pa.Check.ge(0, error='value >= 0'),
                "Unrecognised numeric feature: assumed to be a quantity that cannot be negative.")

STRUCTURAL_EXPLANATIONS = {
    'not_nullable': "Required value is missing.",
    'valid_iso_code': "ISO code must be 3 uppercase letters (or an OWID_ code).",
    'valid_year_range': "Year must fall between 2000 and the current year.",
    'multiple_fields_uniqueness': "Duplicate record: each country-year must be reported once.",
    'plausible_co2_per_capita': "Implied CO2 per person (co2 / population) exceeds 100 t; the real maximum "
                                "is about 68 t (Qatar). Usually a unit error in co2 or population.",
    'coerce_dtype': "Value could not be converted to the expected data type.",
    'column_in_dataframe': "Required column is missing from the file.",
}


def build_schema(df):
    """Builds a pandera schema for whichever SHAP-selected features are present in `df`."""
    columns = {
        'country': pa.Column(str, nullable=False),
        'iso_code': pa.Column(str, pa.Check.str_matches(r'^([A-Z]{3}|OWID_[A-Z]+)$', error='valid_iso_code'), nullable=False),
        'year': pa.Column(int, pa.Check.in_range(2000, date.today().year, error='valid_year_range'),
                          nullable=False),
    }
    for col in df.select_dtypes(include='number').columns:
        if col == 'year':
            continue
        rules = FEATURE_RULES.get(col, [DEFAULT_RULE])
        columns[col] = pa.Column(float, [check for check, _ in rules], nullable=False)

    frame_checks = []
    if {'co2', 'population'}.issubset(df.columns):
        # Cross-field check: catches unit errors that look fine column by column
        frame_checks.append(pa.Check(
            # Missing inputs are left to the not_nullable rule rather than double-reported
            lambda d: (d['co2'] * 1e6 / d['population'] <= 100) | d['co2'].isna() | d['population'].isna(),
            name='plausible_co2_per_capita'
        ))

    return pa.DataFrameSchema(
        columns, checks=frame_checks, coerce=True, strict=False,
        unique=['country', 'year'] if {'country', 'year'}.issubset(df.columns) else None,
    )


def rule_explanation(column, check):
    """Maps a pandera check name back to its plain-English reason."""
    check = str(check)
    for rule, reason in FEATURE_RULES.get(column, [DEFAULT_RULE]):
        if rule.error == check:
            return reason
    for key, reason in STRUCTURAL_EXPLANATIONS.items():
        if check.startswith(key) or key in check:
            return reason
    return check


# Intensity ratios: name -> (numerator, denominator, multiplier, unit).
# An Isolation Forest splits one column at a time, so it cannot see that two columns
# disagree (e.g. GDP entered x1000 while population and CO2 are normal). Ratios make
# those cross-column inconsistencies visible as a single extreme value.
# Units of the OWID base columns, used in plain-English explanations
BASE_UNITS = {
    'co2': 'Mt CO2', 'methane': 'Mt CO2e', 'nitrous_oxide': 'Mt CO2e', 'population': 'people',
    'gdp': 'int-$', 'primary_energy_consumption': 'TWh', 'ghg_per_capita': 't CO2e per person',
}

INTENSITY_RATIOS = {
    'co2_per_capita': ('co2', 'population', 1e6, 't CO2'),
    'methane_per_capita': ('methane', 'population', 1e6, 't CO2e'),
    'nitrous_oxide_per_capita': ('nitrous_oxide', 'population', 1e6, 't CO2e'),
    'energy_per_capita': ('primary_energy_consumption', 'population', 1e9, 'kWh'),
    'co2_per_gdp': ('co2', 'gdp', 1e9, 'kg CO2 per $'),
    'gdp_per_capita': ('gdp', 'population', 1, '$'),
}


def add_intensity_ratios(frame):
    """Adds each ratio whose inputs are present. Returns the frame and the names added."""
    frame = frame.copy()
    added = []
    for name, (num, den, mult, _) in INTENSITY_RATIOS.items():
        if {num, den}.issubset(frame.columns) and name not in frame.columns:
            frame[name] = frame[num] * mult / frame[den]
            added.append(name)
    return frame, added


def attach_review_flags(scored, review):
    """
    Adds `eda_review`: the temporal events data_prep.py flagged for this country-year
    (e.g. "co2: spike_and_revert"). An anomaly that is also a within-country spike is the
    strongest data-error lead, since it is odd both across countries and in its own history.
    """
    scored = scored.copy()
    if review is None or review.empty:
        scored['eda_review'] = ''
        return scored
    labels = (review.assign(label=review['feature'] + ': ' + review['pattern'])
                    .groupby(['country', 'year'])['label'].agg('; '.join))
    keys = pd.MultiIndex.from_frame(scored[['country', 'year']])
    scored['eda_review'] = labels.reindex(keys).fillna('').values
    return scored


# ---------------------------------------------------------------------------
# 2. Synthetic error injection (demonstrates each check firing)
# ---------------------------------------------------------------------------
def inject_synthetic_errors(df, seed=7):
    """
    Corrupts a copy of the data with realistic QA failures and returns an answer key.
    The last error is a x1000 unit mistake, which passes the schema by design and
    should be caught by the statistical layer instead.
    """
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    features = [c for c in df.select_dtypes(include='number').columns if c != 'year']
    rows = rng.choice(len(df), size=6, replace=False)
    key = []

    def record(i, column, error, expected):
        key.append({'row': int(i), 'country': df.at[i, 'country'], 'year': df.at[i, 'year'],
                    'column': column, 'injected_error': error, 'expected_to_be_caught_by': expected})

    emission_col = next((c for c in ['co2', 'methane', 'nitrous_oxide'] if c in features), features[0])
    df[emission_col] = df[emission_col].astype(float)
    df.at[rows[0], emission_col] = -abs(df.at[rows[0], emission_col]) - 1
    record(rows[0], emission_col, "Negative emissions value", "Schema")

    df.at[rows[1], 'iso_code'] = 'XX1'
    record(rows[1], 'iso_code', "Malformed ISO code", "Schema")

    null_col = features[-1]
    df.at[rows[2], null_col] = np.nan
    record(rows[2], null_col, "Missing value", "Schema")

    df.at[rows[3], 'year'] = 1850
    record(rows[3], 'year', "Out-of-scope year", "Schema")

    duplicate = df.iloc[[rows[4]]]
    df = pd.concat([df, duplicate], ignore_index=True)
    key.append({'row': len(df) - 1, 'country': duplicate['country'].iloc[0], 'year': duplicate['year'].iloc[0],
                'column': 'country, year', 'injected_error': "Duplicate country-year record",
                'expected_to_be_caught_by': "Schema"})

    # Pick a unit-error column the schema cannot catch on its own (no cross-field bound)
    unit_col = next((c for c in ['gdp', 'methane', 'nitrous_oxide', 'primary_energy_consumption']
                     if c in features), features[0])
    df[unit_col] = df[unit_col].astype(float)
    df.at[rows[5], unit_col] = df.at[rows[5], unit_col] * 1000
    record(rows[5], unit_col, "Unit error (value x1000)", "Anomaly detection")

    return df, pd.DataFrame(key)


# ---------------------------------------------------------------------------
# 3. Validator
# ---------------------------------------------------------------------------
# Data quality dimension of each rule (the standard DQ framework used in QA reporting)
DIMENSIONS = {
    'not_nullable': 'completeness',
    'multiple_fields_uniqueness': 'uniqueness',
    'plausible_co2_per_capita': 'plausibility',
    'column_in_dataframe': 'completeness',
}


def quality_dimension(rule):
    """Completeness, uniqueness and plausibility are named explicitly; every other rule is validity."""
    return next((dim for key, dim in DIMENSIONS.items() if str(rule).startswith(key)), 'validity')


def completeness_summary(df, features, group='country'):
    """
    Per feature: share of values present, and how many countries report it fully, partly or
    never. Missing data is reported as a quality finding rather than silently dropped.
    """
    rows = []
    for col in features:
        present = df[col].notna().groupby(df[group]).mean()
        rows.append({
            'feature': col,
            'pct_values_present': round(df[col].notna().mean() * 100, 1),
            'countries_complete': int((present == 1).sum()),
            'countries_partial': int(((present > 0) & (present < 1)).sum()),
            'countries_missing': int((present == 0).sum()),
        })
    return pd.DataFrame(rows)


class EmissionsDataValidator:
    """
    Two-layer QA for emissions data:
      1. Deterministic schema rules (pandera): hard logical errors.
      2. Isolation Forest: multivariate statistical outliers, explained per row with SHAP.
      3. Range check: robust z-score against the training distribution. An Isolation Forest
         cannot extrapolate (its splits lie inside the training range, so a value 1000x beyond
         the largest training value scores like that largest value); this layer catches values
         far outside everything seen in training. `range_z=None` disables it.

    When the data has a `split` column, the transform, the model and its threshold are all
    learned from the `fit_split` rows only (train), and every row is then scored. Without
    one, the model is fitted on all valid rows (e.g. an uploaded file).
    """
    def __init__(self, df, contamination=0.02, transform=FALLBACK_TRANSFORM, use_ratios=True, random_state=42,
                 explain=True, fit_split='train', range_z=4.0):
        if transform not in PARAMETER_FREE + FITTED:
            raise ValueError(f"Unknown transform '{transform}'. Choose from {PARAMETER_FREE + FITTED}")
        self.df = df.reset_index(drop=True).copy()
        self.contamination = contamination
        self.transform = transform
        self.X_model = None
        self.use_ratios = use_ratios
        self.random_state = random_state
        self.base_features = [c for c in self.df.select_dtypes(include='number').columns if c != 'year']
        self.feature_cols = list(self.base_features)
        self.schema_failures = None
        self.invalid_idx = set()
        self.model = None
        self.scored = None
        self.shap_values = None
        self.explain = explain
        self.range_z = range_z
        self.range_scores = None
        self.fit_split = fit_split
        self.n_fit = 0
        self.transform_params = {}

    def validate_schema(self):
        """Runs every rule lazily (collects all failures instead of stopping at the first)."""
        schema = build_schema(self.df)
        try:
            schema.validate(self.df, lazy=True)
            failures = pd.DataFrame(columns=['row', 'country', 'year', 'column', 'rule', 'dimension',
                                             'failing_value', 'why'])
        except pa.errors.SchemaErrors as err:
            fc = err.failure_cases.copy()
            fc['row'] = pd.to_numeric(fc['index'], errors='coerce').astype('Int64')
            fc['column'] = fc['column'].fillna('(whole row)')
            fc['rule'] = fc['check'].astype(str)
            fc['why'] = [rule_explanation(c, r) for c, r in zip(fc['column'], fc['rule'])]
            fc['failing_value'] = [
                (str(int(v)) if c == 'year' else f"{v:,.4g}")
                if isinstance(v, (int, float)) and not pd.isna(v) else str(v)
                for c, v in zip(fc['column'], fc['failure_case'])]
            lookup = self.df.reindex(fc['row'].fillna(-1).astype(int))
            fc['country'] = lookup['country'].values if 'country' in self.df else None
            fc['year'] = lookup['year'].values if 'year' in self.df else None
            failures = fc[['row', 'country', 'year', 'column', 'rule', 'failing_value', 'why']]
            # Cross-field failures report the computed boolean; show the inputs instead
            mask = failures['rule'].eq('plausible_co2_per_capita')
            if mask.any():
                rows = failures.loc[mask, 'row'].astype(int)
                per_cap = self.df.loc[rows, 'co2'] * 1e6 / self.df.loc[rows, 'population']
                failures.loc[mask, 'column'] = 'co2 / population'
                failures.loc[mask, 'failing_value'] = [f"{v:,.1f} t per person" for v in per_cap]
            # Uniqueness is reported once per key column; collapse to one line per duplicated row
            dup = failures['rule'].eq('multiple_fields_uniqueness')
            failures.loc[dup, 'column'] = 'country, year'
            failures.loc[dup, 'failing_value'] = (failures.loc[dup, 'country'].astype(str) + ' '
                                                  + failures.loc[dup, 'year'].astype(str))
            failures = failures.drop_duplicates().reset_index(drop=True)
            failures.insert(5, 'dimension', failures['rule'].map(quality_dimension))

        self.invalid_idx = set(failures['row'].dropna().astype(int))
        self.schema_failures = failures
        print(f"INFO: Schema validation complete. {len(failures)} failures across {len(self.invalid_idx)} rows.")
        return failures

    def _model_matrix(self, frame):
        X = frame[self.feature_cols].astype(float)
        return apply_transform(X, self.transform, params=self.transform_params)

    def detect_anomalies(self):
        """
        Fits the Isolation Forest only on schema-valid, complete rows: a model cannot take NaN,
        and rows already known to be wrong would distort what "normal" looks like.
        """
        eligible = self.df.drop(index=list(self.invalid_idx), errors='ignore')
        eligible = eligible.dropna(subset=self.base_features)
        if self.use_ratios:
            eligible, ratios = add_intensity_ratios(eligible)
            self.feature_cols = self.base_features + ratios
        fit_rows = eligible[eligible['split'] == self.fit_split] if 'split' in eligible else eligible
        if fit_rows.empty:
            raise ValueError(f"No valid rows in the '{self.fit_split}' split to fit the model on.")
        self.n_fit = len(fit_rows)

        # Transform parameters (fitted lambdas) are learned from the fit rows only, then the
        # whole matrix is transformed once and reused for scoring and SHAP explanations
        self.transform_params = fit_transform_params(fit_rows[self.feature_cols].astype(float), self.transform)
        X = self._model_matrix(eligible)
        self.X_model = X

        # The threshold (contamination) is set on the fit rows, so it transfers to unseen rows
        self.model = IsolationForest(contamination=self.contamination, n_estimators=200,
                                     random_state=self.random_state)
        self.model.fit(X.loc[fit_rows.index])

        scored = eligible.copy()
        scored['anomaly_score'] = self.model.decision_function(X)  # < 0 means anomalous
        scored['model_anomaly'] = self.model.predict(X) == -1

        # Range layer: robust z (median / MAD) of every feature against the fit rows.
        # Unit errors are multiplicative (x1000 is the same mistake at 0.01 or 10,000), so
        # strictly positive features are measured on log10, where x1000 is always a shift of 3.
        # Features that can be negative (net GHG) use the model's transformed scale.
        R = self._range_matrix(eligible, X, fit_rows)
        reference = R.loc[fit_rows.index]
        center = reference.median()
        spread = (reference - center).abs().median() / 0.6745
        spread = spread.where(spread > 0, reference.std()).replace(0, np.nan)
        self.range_scores = ((R - center) / spread).abs().fillna(0)
        if self.range_z is None:
            beyond = self.range_scores < 0      # all False
        else:
            beyond = self.range_scores > self.range_z
        scored['out_of_range'] = beyond.any(axis=1)
        scored['range_features'] = [', '.join(beyond.columns[flags]) for flags in beyond.to_numpy()]
        scored['is_anomaly'] = scored['model_anomaly'] | scored['out_of_range']

        # Unusualness (0-100): the share of reference (fit) records that look LESS unusual
        # than this one. From the pattern score for every row; when the range check fires, from
        # its distance too, whichever is higher. An empirical percentile, not a probability of error.
        fit_scores = np.sort(scored.loc[fit_rows.index, 'anomaly_score'].to_numpy())
        model_pct = (len(fit_scores) - np.searchsorted(fit_scores, scored['anomaly_score'].to_numpy(),
                                                       side='right')) / len(fit_scores) * 100
        max_z = self.range_scores.max(axis=1)
        fit_z = np.sort(max_z.loc[fit_rows.index].to_numpy())
        range_pct = np.searchsorted(fit_z, max_z.to_numpy(), side='left') / len(fit_z) * 100
        scored['unusualness'] = np.where(scored['out_of_range'], np.maximum(model_pct, range_pct),
                                                model_pct).round(1)
        scored['detected_by'] = np.select(
            [scored['model_anomaly'] & scored['out_of_range'], scored['out_of_range'], scored['model_anomaly']],
            ['model + range', 'range', 'model'], '')
        # Percentile of each value against the reference (fit) rows, for plain-English explanations
        for col in self.feature_cols:
            reference = np.sort(fit_rows[col].to_numpy(dtype=float))
            scored[f'{col}_pctile'] = np.searchsorted(reference, eligible[col].to_numpy(dtype=float),
                                                      side='right') / len(reference) * 100
        self.scored = scored
        print(f"INFO: Anomaly detection complete. Fitted on {self.n_fit} rows; "
              f"{scored['is_anomaly'].sum()} of {len(scored)} rows flagged.")
        return scored

    def explain_anomalies(self):
        """
        SHAP values per flagged row. For an Isolation Forest, SHAP explains the expected path
        length: a negative contribution means the feature made the row *easier* to isolate,
        so we flip the sign and report `push_toward_anomaly` (positive = more suspicious).
        """
        flagged = self.scored[self.scored['is_anomaly']]
        if flagged.empty:
            self.shap_values = pd.DataFrame()
            return self.shap_values

        explainer = shap.TreeExplainer(self.model)
        values = explainer.shap_values(self.X_model.loc[flagged.index])
        contributions = pd.DataFrame(-values, columns=self.feature_cols, index=flagged.index)
        self.shap_values = contributions

        # Out-of-range rows are explained by their most extreme feature; the rest by the top SHAP driver
        drivers = contributions.idxmax(axis=1)
        far = flagged.index[flagged['out_of_range']]
        drivers[far] = self.range_scores.loc[far].idxmax(axis=1)
        self.scored.loc[flagged.index, 'top_driver'] = drivers
        self.scored.loc[flagged.index, 'explanation'] = [
            self._describe_range(flagged.loc[i], drivers[i]) if i in far else self._describe(flagged.loc[i], drivers[i])
            for i in flagged.index
        ]
        return contributions

    @staticmethod
    def _fmt(value, feature):
        """Human-readable number with its unit: 6,020 Mt CO2 rather than 6.02e+03."""
        unit = INTENSITY_RATIOS[feature][3] if feature in INTENSITY_RATIOS else BASE_UNITS.get(feature, '')
        v = float(value)
        if abs(v) >= 1e9:
            text = f"{v / 1e9:,.3g} billion"
        elif abs(v) >= 1e6:
            text = f"{v / 1e6:,.3g} million"
        elif abs(v) >= 1000:
            text = f"{v:,.0f}"
        else:
            text = f"{v:.3g}"
        return f"{text} {unit}".strip()

    def _describe(self, row, driver):
        pct = row[f'{driver}_pctile']
        position = "unusually high" if pct >= 50 else "unusually low"
        return f"{driver} = {self._fmt(row[driver], driver)} is {position} ({pct:.1f}th percentile)"

    def _range_matrix(self, eligible, X, fit_rows):
        R = X.copy()
        for col in self.feature_cols:
            if (fit_rows[col] > 0).all():
                values = eligible[col].astype(float)
                # A non-positive value where training was strictly positive is infinitely far out
                R[col] = np.where(values > 0, np.log10(values.where(values > 0)), np.inf)
        return R

    def _describe_range(self, row, driver):
        z = self.range_scores.at[row.name, driver]
        return (f"{driver} = {self._fmt(row[driver], driver)} is far outside the reference range "
                f"(robust z = {z:.1f}; possible unit or entry error)")

    def run(self):
        """Executes the full pipeline and returns headline metrics."""
        self.validate_schema()
        self.detect_anomalies()
        if self.explain:
            self.explain_anomalies()
        else:
            self.shap_values = pd.DataFrame()
        n = len(self.df)
        n_anom = int(self.scored['is_anomaly'].sum())
        return {
            'rows': n,
            'schema_rows': len(self.invalid_idx),
            'schema_failures': len(self.schema_failures),
            'anomalies': n_anom,
            'pass_rate': (n - len(self.invalid_idx) - n_anom) / n if n else 0.0,
        }
