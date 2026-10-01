import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import shap
from sklearn.ensemble import IsolationForest

from design import MAP_RAMP, TOKENS
from outliers import outlier_profile, temporal_spikes
from splitting import country_strata, split_summary, stratified_group_split
from transforms import apply_transform, compare_transforms, lambda_test, select_transform
from validation import completeness_summary

# Sequential single-hue ramp (light -> dark blue) used for all missingness heatmaps
MISSINGNESS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "missingness", [TOKENS['card']] + MAP_RAMP
)
# Colorblind-safe categorical pair (blue / orange) and chart ink
# Same theme as the app (design.py): teal for normal series, rust for flagged / raw-scale series
BLUE, ORANGE = TOKENS['teal'], TOKENS['rust']
INK, INK_MUTED, GRID, PAPER, AXIS = TOKENS['ink'], TOKENS['ink_muted'], TOKENS['grid'], TOKENS['card'], TOKENS['axis']


def _short_number(value, _pos=None):
    """Axis labels people can read at a glance: 1.5B, 250M, 12k, 0.4."""
    for size, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'k')):
        if abs(value) >= size:
            return f"{value / size:g}{suffix}"
    return f"{value:g}"


class EmissionsDataProcessor:
    """
    Handles the extraction, transformation, and feature selection
    of global emissions data for anomaly detection modeling.

    Every cleaning decision is recorded in `self.cleaning_log` and exported to
    `reports/cleaning_log.csv` so the final dataset can be fully explained.
    """
    ID_COLS = ['country', 'year', 'iso_code']
    CANDIDATE_FEATURES = [
        'population', 'gdp', 'co2', 'primary_energy_consumption',
        'methane', 'nitrous_oxide', 'ghg_per_capita'
    ]

    def __init__(self, url, start_year=2000, min_year_coverage=0.50,
                 report_dir='reports', viz_dir='visualizations'):
        self.url = url
        self.start_year = start_year
        self.min_year_coverage = min_year_coverage
        self.report_dir = report_dir
        self.viz_dir = viz_dir
        self.raw_data = None
        self.scoped_data = None
        self.clean_data = None
        self.cleaning_log = []
        self.transform = None   # set by select_transformation(); None means untransformed
        self.country_split = None  # set by assign_splits(): country -> train / validation / test
        self._pass = 1             # 1 = selection pass on all candidates, 2 = rebuild on chosen features
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

    def _log(self, step, rule, detail, rows_before, rows_after):
        """Records a cleaning decision so the pipeline is auditable end to end."""
        self.cleaning_log.append({
            'pass': self._pass, 'step': step, 'rule': rule, 'detail': detail,
            'rows_before': rows_before, 'rows_after': rows_after,
            'rows_removed': rows_before - rows_after
        })

    def ingest(self):
        """Fetches the raw dataset directly from the source."""
        print("INFO: Initiating download of OWID CO2 dataset...")
        self.raw_data = pd.read_csv(self.url)
        print(f"INFO: Ingestion complete. Dataset shape: {self.raw_data.shape}")
        return self.raw_data

    def snapshot_raw(self, snapshot_dir='data/raw', source_url=None):
        """
        Saves the exact raw file used for this run with a SHA-256 manifest. The OWID file is
        updated in place upstream, so without a snapshot a rerun could silently use new data.
        """
        os.makedirs(snapshot_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        path = os.path.join(snapshot_dir, f"owid-co2-data_{stamp}.csv")
        self.raw_data.to_csv(path, index=False)
        manifest = {
            'file': os.path.basename(path),
            'sha256': self._sha256(path),
            'rows': len(self.raw_data),
            'columns': self.raw_data.shape[1],
            'source_url': source_url or self.url,
            'downloaded_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }
        with open(path + '.manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f"INFO: Raw snapshot saved to {path} (sha256 {manifest['sha256'][:12]}...)")
        return path

    def ingest_snapshot(self, path):
        """Loads a saved snapshot after verifying its checksum against the manifest."""
        with open(path + '.manifest.json', encoding='utf-8') as f:
            manifest = json.load(f)
        if self._sha256(path) != manifest['sha256']:
            raise ValueError(f"Snapshot checksum mismatch for {path}: the file changed since it was saved.")
        self.raw_data = pd.read_csv(path)
        self.url = manifest.get('source_url')
        print(f"INFO: Loaded verified snapshot {path} ({len(self.raw_data)} rows)")
        return self.raw_data

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def scope(self):
        """Restricts to modern years and real countries (aggregate regions have no ISO code)."""
        df = self.raw_data[self.ID_COLS + self.CANDIDATE_FEATURES]
        before = len(df)
        df = df[df['year'] >= self.start_year]
        df = df.dropna(subset=['iso_code'])
        self._log('1. Scope', f"year >= {self.start_year} and iso_code present",
                  "Keeps modern data and drops aggregate regions (World, Africa, High-income...)",
                  before, len(df))
        self.scoped_data = df
        print(f"INFO: Scoped dataset shape: {df.shape}")
        return df

    def assign_splits(self, fractions=None, seed=42, n_bins=4, stratify_on='co2'):
        """
        Assigns every country to train / validation / test BEFORE any modelling decision.
        Grouped by country (a country's years are near-duplicates, so splitting rows would
        leak) and stratified by size (quartiles of median CO2), so every split holds the same
        mix of small and large emitters. Transform choice, feature selection and model fitting
        then use training countries only; validation picks settings; test is scored once.
        """
        fractions = fractions or {'train': 0.6, 'validation': 0.2, 'test': 0.2}
        strata = country_strata(self.scoped_data, stratify_on, n_bins=n_bins)
        self.country_split = stratified_group_split(strata, fractions, seed=seed)
        self.country_strata = strata
        split_summary(self.scoped_data, self.country_split, strata) \
            .to_csv(f"{self.report_dir}/split_summary.csv", index=False)
        pd.DataFrame({'country': self.country_split.index, 'split': self.country_split.values,
                      'stratum': strata[self.country_split.index].values}) \
            .to_csv(f"{self.report_dir}/split_assignment.csv", index=False)
        counts = self.country_split.value_counts().to_dict()
        rows = len(self.scoped_data)
        self._log('1b. Train/validation/test split',
                  f"by country, stratified on {stratify_on} size quartiles, seed {seed}",
                  f"Countries per split {counts}. See split_summary.csv", rows, rows)
        print(f"INFO: Countries per split: {counts}")
        return self.country_split

    def training_view(self):
        """clean_data restricted to training countries (all rows when no split is assigned)."""
        if self.country_split is None:
            return self.clean_data
        train = self.country_split[self.country_split == 'train'].index
        return self.clean_data[self.clean_data['country'].isin(train)]

    def profile_missingness(self):
        """
        Investigates *where* values are missing before deciding *how* to treat them.

        Missing data is only safe to impute when it is scattered randomly. If it is
        concentrated in specific years (source reporting lag) or specific countries
        (a source never covers them), it is structural and imputation would fabricate
        values. This method measures both patterns for every candidate feature.
        """
        print("\nINFO: Profiling missingness across years and countries...")
        df = self.scoped_data
        features = self.CANDIDATE_FEATURES
        is_missing = df[features].isna()

        # % missing per year (rows = year, cols = feature)
        by_year = is_missing.groupby(df['year']).mean()
        # % missing per country (rows = country, cols = feature)
        by_country = is_missing.groupby(df['country']).mean()

        # Years where a feature is entirely unreported, and countries a feature never covers
        empty_years = by_year.eq(1)
        empty_countries = by_country.eq(1)

        diagnosis = []
        for col in features:
            n_missing = is_missing[col].sum()
            from_empty_years = is_missing[col] & df['year'].isin(empty_years.index[empty_years[col]])
            from_empty_countries = is_missing[col] & df['country'].isin(empty_countries.index[empty_countries[col]])
            structural = from_empty_years | from_empty_countries
            share_structural = structural.sum() / n_missing if n_missing else np.nan

            if n_missing == 0:
                pattern = 'COMPLETE'
            elif share_structural >= 0.90:
                pattern = 'STRUCTURAL'
            else:
                pattern = 'SCATTERED'

            diagnosis.append({
                'feature': col,
                'pct_missing': round(n_missing / len(df) * 100, 1),
                'years_fully_missing': ', '.join(map(str, empty_years.index[empty_years[col]])) or '-',
                'countries_fully_missing': int(empty_countries[col].sum()),
                'pct_of_gaps_from_empty_years': round(from_empty_years.sum() / n_missing * 100, 1) if n_missing else 0.0,
                'pct_of_gaps_from_empty_countries': round(from_empty_countries.sum() / n_missing * 100, 1) if n_missing else 0.0,
                'pattern': pattern
            })
        diagnosis = pd.DataFrame(diagnosis)

        print("\nMissingness Diagnosis (STRUCTURAL = >=90% of gaps come from empty years or empty countries):")
        print(diagnosis.to_string(index=False))

        # Exports for the app and for anyone auditing the pipeline
        (by_year * 100).round(1).to_csv(f"{self.report_dir}/missingness_by_year.csv")
        (by_country * 100).round(1).to_csv(f"{self.report_dir}/missingness_by_country.csv")
        diagnosis.to_csv(f"{self.report_dir}/missingness_diagnosis.csv", index=False)

        self._plot_missingness_heatmap(
            by_year.T, "Missing values by year (% of countries)",
            f"{self.viz_dir}/missingness_by_year.png", xlabel="Year"
        )
        gappy_countries = by_country[by_country.gt(0).any(axis=1)]
        self._plot_missingness_heatmap(
            gappy_countries, f"Countries with any missing values ({len(gappy_countries)} of {len(by_country)}) - % of years missing",
            f"{self.viz_dir}/missingness_by_country.png", xlabel="Feature"
        )
        print(f"INFO: Missingness reports saved to {self.report_dir}/ and heatmaps to {self.viz_dir}/")

        self.missingness_by_year = by_year
        self.missingness_diagnosis = diagnosis
        return diagnosis

    def _plot_missingness_heatmap(self, matrix, title, path, xlabel):
        """Draws a single-hue heatmap: white = complete, dark blue = fully missing."""
        height = max(3, 0.18 * len(matrix) + 1.5)
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * matrix.shape[1] + 3), height))
        im = ax.imshow(matrix.values * 100, aspect='auto', cmap=MISSINGNESS_CMAP, vmin=0, vmax=100)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=9, color=INK_MUTED)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=9, color=INK_MUTED)
        ax.set_xlabel(xlabel, color=INK_MUTED)
        ax.set_title(title, loc='left', fontsize=12, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('% missing', color=INK_MUTED)
        cbar.outline.set_visible(False)
        fig.savefig(path, bbox_inches='tight', dpi=120, facecolor=PAPER)
        plt.close(fig)

    def clean(self):
        """
        Pass 1: treats structural missingness first (by removing the years/countries a source
        does not cover), then applies the tiered strategy to whatever gaps remain. Judged on
        every candidate feature, because features have not been chosen yet.
        """
        print("\nINFO: Commencing data cleaning (pass 1, all candidate features)...")
        self.clean_data = self._clean_frame(self.scoped_data.copy(), self.CANDIDATE_FEATURES,
                                            steps=('2', '3', '4'), dropped_file='dropped_countries.csv')
        print(f"\nINFO: Cleaning complete. Cleaned dataset shape: {self.clean_data.shape}\n")
        return self.clean_data

    def _clean_frame(self, df, features, steps, dropped_file):
        """Reporting-lag trim, uncovered-country removal, then the tiered gap strategy."""
        lag_step, cover_step, gap_step = steps

        # Trim trailing years that sources have not reported yet.
        # A year is "reported" if every feature covers at least `min_year_coverage` of countries.
        year_coverage = df[features].notna().groupby(df['year']).mean()
        reported = year_coverage.min(axis=1) >= self.min_year_coverage
        # Walk back from the latest year and stop at the first fully reported one (only trims the tail)
        last_reported_year = reported[reported].index.max()
        lagging = [int(y) for y in reported.index if y > last_reported_year]
        before = len(df)
        df = df[df['year'] <= last_reported_year]
        lagging_features = {
            col for y in lagging for col in features
            if year_coverage.loc[y, col] < self.min_year_coverage
        }
        self._log(f'{lag_step}. Reporting lag', f"keep year <= {last_reported_year}",
                  f"Years {lagging} dropped: {sorted(lagging_features)} not yet published for most countries",
                  before, len(df))
        print(f"  - Reporting lag: keeping {self.start_year}-{last_reported_year}, dropped years {lagging}")

        # Drop countries a source never covers (100% missing for any feature in the window).
        # Imputing a whole series for a country would be fabrication, not estimation.
        country_missing = df[features].isna().groupby(df['country']).mean()
        uncovered = country_missing[country_missing.eq(1).any(axis=1)]
        co2_total = df['co2'].sum()
        before = len(df)
        df = df[~df['country'].isin(uncovered.index)]
        co2_retained = df['co2'].sum() / co2_total

        uncovered_report = pd.DataFrame({
            'country': uncovered.index,
            'features_never_reported': [
                ', '.join(uncovered.columns[row.eq(1)]) for _, row in uncovered.iterrows()
            ]
        })
        uncovered_report.to_csv(f"{self.report_dir}/{dropped_file}", index=False)
        self._log(f'{cover_step}. Uncovered countries', "drop countries with any feature 100% missing",
                  f"{len(uncovered)} countries dropped (mostly microstates/territories); "
                  f"{co2_retained:.1%} of CO2 emissions retained. See {dropped_file}",
                  before, len(df))
        print(f"  - Uncovered countries: dropped {len(uncovered)}, retaining {co2_retained:.1%} of CO2 emissions")

        # Tiered strategy for the remaining (non-structural) gaps
        print("\nINFO: Executing tiered missing value strategy on remaining gaps...")
        total_rows = len(df)
        cols_to_drop, cols_to_interpolate, cols_to_dropna = [], [], []

        for col in features:
            missing_pct = df[col].isna().sum() / total_rows
            if missing_pct > 0.20:
                cols_to_drop.append(col)
                action = "DROP FEATURE"
            elif missing_pct >= 0.05:
                cols_to_interpolate.append(col)
                action = "WITHIN-COUNTRY INTERPOLATION"
            elif missing_pct > 0:
                cols_to_dropna.append(col)
                action = "DROP NA ROWS"
            else:
                action = "KEEP"
            print(f"  - Feature '{col}' missing {missing_pct:.1%} | Action: {action}")

        if cols_to_drop:
            before = len(df)
            df = df.drop(columns=cols_to_drop)
            self._log(f'{gap_step}a. Sparse features', "drop feature if > 20% missing",
                      f"Dropped {cols_to_drop}", before, len(df))

        if cols_to_interpolate:
            # Linear interpolation over time within each country, interior gaps only:
            # uses the country's own trend instead of a global median, and never extrapolates.
            df = df.sort_values(['country', 'year'])
            filled_before = df[cols_to_interpolate].isna().sum().sum()
            df[cols_to_interpolate] = (
                df.groupby('country')[cols_to_interpolate]
                  .transform(lambda s: s.interpolate(method='linear', limit_area='inside'))
            )
            filled = filled_before - df[cols_to_interpolate].isna().sum().sum()
            self._log(f'{gap_step}b. Moderate gaps', "5-20% missing: within-country linear interpolation (interior only)",
                      f"Filled {filled} values in {cols_to_interpolate}", len(df), len(df))
            # Anything interpolation could not reach (leading/trailing gaps) is dropped
            cols_to_dropna += cols_to_interpolate

        if cols_to_dropna:
            before = len(df)
            gap_rows = df[df[cols_to_dropna].isna().any(axis=1)]
            df = df.dropna(subset=cols_to_dropna)
            affected = ', '.join(sorted(gap_rows['country'].unique())) or 'none'
            self._log(f'{gap_step}c. Residual gaps', "< 5% missing (or not interpolable): drop rows",
                      f"Rows removed from: {affected}", before, len(df))

        return df.reset_index(drop=True)

    def profile_outliers(self, spike_z_threshold=10):
        """
        Tests for outliers two ways and records how each kind is handled.

        1. Cross-sectional (vs other countries): IQR fences and robust z-scores on the raw
           and log scales. Emissions, GDP and population are heavily right-skewed, so on the
           raw scale these tests flag the largest economies simply for being large.
           Handling: KEEP. They are real, and the log transform used by the model
           (see outlier_profile.csv: skew and % flagged before vs after) absorbs the skew.
           Clipping or deleting China would be falsifying the data.

        2. Temporal (vs the country's own history): robust z-score of year-over-year log
           changes. A one-year spike that reverts is the classic signature of either a real
           shock (war, COVID) or a data-entry error, and code cannot tell which.
           Handling: FLAG FOR REVIEW in outlier_review.csv. Rows are not removed, because
           the downstream anomaly detector exists to surface exactly these cases, and deleting
           them first would hide them from the QA process.
        """
        print("INFO: Testing for outliers (cross-sectional and temporal)...")
        df = self.clean_data
        features = [c for c in self.CANDIDATE_FEATURES if c in df.columns]

        profile = outlier_profile(df, features)
        review = pd.concat(
            [temporal_spikes(df, col, z_threshold=spike_z_threshold) for col in features],
            ignore_index=True
        )
        review = review.sort_values('robust_z', key=abs, ascending=False).reset_index(drop=True)

        print("\nCross-sectional outliers (% of rows flagged, raw vs log scale):")
        print(profile.to_string(index=False))
        print(f"\nTemporal events flagged for review ({len(review)}):")
        print(review.head(15).to_string(index=False) if len(review) else "  none")

        profile.to_csv(f"{self.report_dir}/outlier_profile.csv", index=False)
        review.to_csv(f"{self.report_dir}/outlier_review.csv", index=False)
        self._plot_outlier_scale_comparison(profile)
        self._plot_spike_examples(df, review)

        patterns = review['pattern'].value_counts().to_dict() if len(review) else {}
        rows = len(df)
        self._log('4d. Outlier review', "flag, do not remove",
                  f"Cross-sectional: raw-scale IQR flags {profile['pct_iqr_raw'].mean():.1f}% of values on average, "
                  f"log-scale {profile['pct_iqr_log'].mean():.1f}%, so extremes are skew and handled by the log transform. "
                  f"Temporal: {len(review)} events flagged for review {patterns}. See outlier_review.csv",
                  rows, rows)

        self.outlier_profile = profile
        self.outlier_review = review
        return profile

    def _style_axes(self, ax):
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

    def _plot_outlier_scale_comparison(self, profile):
        """Grouped bars: % of values flagged by IQR on the raw scale vs the log scale."""
        profile = profile.sort_values('pct_iqr_raw')
        y = np.arange(len(profile))
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(profile) + 1.5))
        bar_h = 0.36
        for offset, col, color, label in [(bar_h / 2 + 0.02, 'pct_iqr_raw', ORANGE, 'Raw scale'),
                                          (-bar_h / 2 - 0.02, 'pct_iqr_log', BLUE, 'Log scale')]:
            ax.barh(y + offset, profile[col], height=bar_h, color=color, label=label)
            for yi, v in zip(y + offset, profile[col]):
                ax.text(v + 0.2, yi, f"{v:.1f}%", va='center', fontsize=7, color=INK_MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(profile['feature'])
        ax.set_xlabel('% of values outside Tukey fences (1.5 x IQR)', color=INK_MUTED)
        ax.set_title("Most raw-scale 'outliers' are skew: they disappear on the log scale",
                     loc='left', fontsize=11, color=INK)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc='lower right', fontsize=8)
        self._style_axes(ax)
        fig.savefig(f"{self.viz_dir}/outlier_scale_comparison.png", bbox_inches='tight', dpi=120,
                    facecolor=PAPER)
        plt.close(fig)

    def _plot_spike_examples(self, df, review, max_panels=6):
        """Small multiples of the strongest temporal events, each in its country's own history."""
        # One panel per country-feature series, strongest first, marking every flagged year
        series_keys = review[['country', 'feature']].drop_duplicates().head(max_panels)
        events = [review[(review['country'] == c) & (review['feature'] == f)] for c, f in series_keys.values]
        n = max(len(events), 1)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.0 * rows), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        if not events:
            ax = axes.flat[0]
            ax.set_visible(True)
            ax.text(0.5, 0.5, "No temporal spikes detected", ha='center', va='center', color=INK_MUTED)
            ax.axis('off')
        for ax, ev in zip(axes.flat, events):
            ax.set_visible(True)
            country, feature = ev['country'].iloc[0], ev['feature'].iloc[0]
            series = df[df['country'] == country].sort_values('year')
            ax.plot(series['year'], series[feature], color=BLUE, linewidth=2)
            ax.scatter(ev['year'], ev['value'], marker='^', s=70, color=ORANGE, zorder=3,
                       edgecolor=PAPER, linewidth=1.5)
            # Short titles: the strongest years in time order, the rest summarised as a count
            years = sorted(int(y) for y in ev['year'])
            flagged = ', '.join(map(str, years[:3])) + (f" +{len(years) - 3} more" if len(years) > 3 else "")
            ax.set_title(f"{country}: {feature}\nflagged years: {flagged}", loc='left', fontsize=10, color=INK)
            ax.yaxis.set_major_formatter(FuncFormatter(_short_number))   # 150B, not 1.5 with a 1e11 offset
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            self._style_axes(ax)
        fig.suptitle("Strongest year-over-year events flagged for review", x=0.01, ha='left',
                     fontsize=11, color=INK)
        fig.tight_layout()
        fig.savefig(f"{self.viz_dir}/outlier_spikes.png", bbox_inches='tight', dpi=120, facecolor=PAPER)
        plt.close(fig)

    def select_transformation(self, tolerance=0.01):
        """
        Chooses the feature transformation with statistical tests (see transforms.py):
          - Shapiro-Wilk W for every candidate transform on every feature
          - likelihood-ratio test of H0: lambda = 0 ("is log the optimal power?")
        The choice is applied to the SHAP baseline model here and recommended to the app,
        so feature selection and anomaly detection see the data the same way.
        """
        print("\nINFO: Testing candidate transformations...")
        train = self.training_view()
        features = [c for c in self.CANDIDATE_FEATURES if c in train.columns]
        comparison = compare_transforms(train, features)
        lambda_tests = pd.DataFrame([{'feature': col, **lambda_test(train[col])} for col in features])
        choice, reason, summary = select_transform(comparison, tolerance=tolerance)
        summary['chosen'] = summary['transform'].eq(choice)
        summary['reason'] = np.where(summary['chosen'], reason, '')

        wide = comparison.pivot(index='feature', columns='transform', values='shapiro_w')
        print("\nShapiro-Wilk W by feature (1.0 = perfectly normal):")
        print(wide.to_string())
        print("\nLikelihood-ratio test, H0: lambda = 0 (log is the optimal power transform):")
        print(lambda_tests.to_string(index=False))
        print(f"\nINFO: Selected transform: {choice}. {reason}")

        comparison.to_csv(f"{self.report_dir}/transform_comparison.csv", index=False)
        lambda_tests.to_csv(f"{self.report_dir}/transform_lambda_tests.csv", index=False)
        summary.to_csv(f"{self.report_dir}/transform_selection.csv", index=False)

        n_log_ok = int(lambda_tests['log_supported'].sum())
        rows = len(self.clean_data)
        self._log('4e. Transform selection', "highest mean Shapiro-Wilk W; prefer parameter-free within tolerance",
                  f"Selected {choice}. {reason} Log (lambda = 0) not rejected for {n_log_ok} of "
                  f"{len(features)} features. See transform_*.csv", rows, rows)

        self.transform = choice
        self.transform_summary = summary
        self.lambda_tests = lambda_tests
        return choice

    def _shap_ranking(self, frame, numeric_cols, transform, max_rows=None, seed=42):
        """Fits a baseline Isolation Forest and ranks features by mean |SHAP|."""
        if max_rows and len(frame) > max_rows:
            frame = frame.sample(max_rows, random_state=seed)
        numeric_df = apply_transform(frame[numeric_cols], transform)
        model = IsolationForest(contamination=0.02, random_state=seed)
        model.fit(numeric_df)
        shap_values = shap.TreeExplainer(model).shap_values(numeric_df)
        ranking = pd.DataFrame({
            'Feature': numeric_cols,
            'SHAP_Importance': np.abs(shap_values).mean(axis=0),
            'Transform': transform
        }).sort_values(by='SHAP_Importance', ascending=False).reset_index(drop=True)
        return ranking, shap_values, numeric_df

    @staticmethod
    def _pick(ranking, top_n, required):
        """Required features fill the budget first; the SHAP ranking fills the rest."""
        actual_top_n = min(max(top_n, len(required)), len(ranking))
        by_shap = [f for f in ranking['Feature'] if f not in required][:actual_top_n - len(required)]
        return [f for f in ranking['Feature'] if f in required or f in by_shap]

    def select_features_with_shap(self, top_n=4, required=None):
        """
        Uses SHAP values to mathematically identify the most important features, fitted on
        training countries only.

        `required` features are always kept, then the remaining slots go to the highest-SHAP
        features. SHAP measures how useful a feature is for *isolating outliers*, not how
        important it is to the business: a CO2 QA tool must validate CO2 even if another
        column happens to separate outliers more easily.
        """
        required = list(required or [])
        missing = [c for c in required if c not in self.clean_data.columns]
        if missing:
            raise ValueError(f"Required feature(s) {missing} did not survive cleaning; "
                             f"check the missingness diagnosis before relaxing the requirement.")
        transform = self.transform or 'none'
        print(f"INFO: Running SHAP Feature Selection via Baseline Isolation Forest (transform: {transform})...")

        # The model sees the same transformed features that the app's model will
        numeric_cols = [col for col in self.clean_data.columns if col not in self.ID_COLS + ['split']]
        shap_df, shap_values, numeric_df = self._shap_ranking(self.training_view(), numeric_cols, transform)

        plt.figure(figsize=(10, 6), facecolor=PAPER)
        shap.summary_plot(shap_values, numeric_df, plot_type="bar", show=False, color=BLUE)
        plt.gca().set_facecolor(PAPER)
        plt.title("SHAP global feature importance for outlier detection (training countries)",
                  loc='left', fontsize=12, color=INK)
        plot_path = f'{self.viz_dir}/shap_feature_importance.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=120, facecolor=PAPER)
        plt.close()
        print(f"INFO: SHAP feature importance plot saved to {plot_path}")

        best_features = self._pick(shap_df, top_n, required)
        shap_df['Selected'] = shap_df['Feature'].isin(best_features)
        shap_df['Selection_Reason'] = np.select(
            [shap_df['Feature'].isin(required), shap_df['Selected']],
            ['required (core metric)', 'top SHAP'], 'not selected')
        shap_df.to_csv(f"{self.report_dir}/shap_feature_importance.csv", index=False)

        print("\nFeature Importance Ranking:")
        print(shap_df.to_string(index=False))
        print(f"\nINFO: Selected {len(best_features)} features: {best_features} (required: {required})")

        before = len(self.clean_data)
        self.pass1_data = self.clean_data          # kept whole for the stability check
        self.clean_data = self.clean_data[self.ID_COLS + best_features]
        self.selected_features = best_features
        dropped = [c for c in numeric_cols if c not in best_features]
        self._log('5. SHAP selection',
                  f"keep required features, then top features by mean |SHAP| ({len(best_features)} total), "
                  f"fitted on training countries",
                  f"Selected {best_features} (required: {required}); discarded {dropped}",
                  before, len(self.clean_data))
        return best_features

    def stability_check(self, n_boot=30, top_n=4, required=None, seed=42, max_rows=800):
        """
        Is the SHAP selection a stable signal or luck of this particular sample? Resamples
        training COUNTRIES with replacement (a bootstrap that respects the country grouping),
        reruns the selection on each resample and reports how often each feature is chosen.
        Near 100% = robust choice; near 50% = the ranking is a coin flip between features.
        """
        required = list(required or [])
        rng = np.random.default_rng(seed)
        pass1 = getattr(self, 'pass1_data', self.clean_data)
        train = pass1 if self.country_split is None else             pass1[pass1['country'].map(self.country_split).eq('train')]
        numeric_cols = [c for c in self.CANDIDATE_FEATURES if c in train.columns]
        countries = train['country'].unique()
        by_country = {c: g for c, g in train.groupby('country')}
        transform = self.transform or 'none'
        picks, ranks = [], []
        print(f"INFO: Stability check: {n_boot} bootstrap resamples of {len(countries)} training countries...")
        for b in range(n_boot):
            sample = pd.concat([by_country[c] for c in rng.choice(countries, size=len(countries), replace=True)])
            ranking, _, _ = self._shap_ranking(sample, numeric_cols, transform, max_rows=max_rows, seed=b)
            picks.append(self._pick(ranking, top_n, required))
            ranks.append(ranking.reset_index().set_index('Feature')['index'] + 1)
        ranks = pd.concat(ranks, axis=1)
        stability = pd.DataFrame({
            'feature': numeric_cols,
            'selection_frequency': [np.mean([f in p for p in picks]) for f in numeric_cols],
            'mean_rank': [ranks.loc[f].mean() for f in numeric_cols],
            'rank_sd': [ranks.loc[f].std() for f in numeric_cols],
            'required': [f in required for f in numeric_cols],
        }).sort_values(['selection_frequency', 'mean_rank'], ascending=[False, True]).round(3)
        stability.to_csv(f"{self.report_dir}/feature_stability.csv", index=False)
        print(stability.to_string(index=False))

        chosen = getattr(self, 'selected_features', [])
        weakest = stability[stability['feature'].isin(chosen) & ~stability['required']]
        rows = len(self.clean_data)
        self._log('5b. Selection stability', f"{n_boot} country-level bootstrap resamples of the training split",
                  "Selection frequency of chosen features: "
                  + ', '.join(f"{r.feature} {r.selection_frequency:.0%}" for r in weakest.itertuples())
                  + ". See feature_stability.csv", rows, rows)
        self.feature_stability = stability
        return stability

    def rebuild(self, features):
        """
        Pass 2: rebuilds the dataset from the scoped data using ONLY the selected features.
        Pass 1 had to judge coverage on every candidate, so a country or year missing a feature
        that was later discarded (e.g. GDP) was dropped for nothing. Every decision here is the
        same rule as pass 1, applied to the columns the model actually uses.
        """
        print(f"\nINFO: Rebuilding dataset (pass 2) on selected features {features}...")
        self._pass = 2
        df = self.scoped_data[self.ID_COLS + list(features)].copy()
        df = self._clean_frame(df, list(features), steps=('6', '7', '8'),
                               dropped_file='dropped_countries_final.csv')
        if self.country_split is not None:
            df['split'] = df['country'].map(self.country_split)
        self.clean_data = df
        self.selected_features = list(features)
        print(f"INFO: Rebuilt dataset: {len(df)} rows, {df['country'].nunique()} countries, "
              f"{df['year'].min()}-{df['year'].max()}")
        return df

    def report_completeness(self):
        """
        Reports missing data as a quality finding (completeness is a core data quality
        dimension): per feature, and per country with its split, over the final year window.
        Nothing is removed here; it documents what the model cannot check.
        """
        features = [c for c in self.clean_data.columns if c not in self.ID_COLS + ['split']]
        window = self.scoped_data[self.scoped_data['year'] <= self.clean_data['year'].max()]
        summary = completeness_summary(window, features)
        summary.to_csv(f"{self.report_dir}/completeness.csv", index=False)

        present = window[features].notna().groupby(window['country']).mean()
        status = np.select([present.eq(1).all(axis=1), present.eq(0).any(axis=1)],
                           ['complete', 'missing'], 'partial')
        by_country = pd.DataFrame({
            'country': present.index,
            'status': status,
            'features_missing_entirely': [', '.join(present.columns[row.eq(0)]) for _, row in present.iterrows()],
            'pct_values_present': (present.mean(axis=1) * 100).round(1).values,
            'in_final_dataset': present.index.isin(self.clean_data['country']),
        })
        if self.country_split is not None:
            by_country.insert(1, 'split', by_country['country'].map(self.country_split).values)
        by_country.sort_values(['status', 'country']).to_csv(
            f"{self.report_dir}/completeness_by_country.csv", index=False)

        counts = pd.Series(status).value_counts().to_dict()
        print("\nCompleteness of selected features:")
        print(summary.to_string(index=False))
        rows = len(self.clean_data)
        self._log('9. Completeness report', "report, do not remove",
                  f"Countries by completeness of {features}: {counts}. See completeness*.csv", rows, rows)
        return summary

    def export(self, filepath):
        """Saves the cleaned and optimized dataset plus the audit trail of how it was built."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.clean_data.to_csv(filepath, index=False)
        pd.DataFrame(self.cleaning_log).to_csv(f"{self.report_dir}/cleaning_log.csv", index=False)
        print(f"INFO: Exported optimized dataset to {filepath}")
        print(f"INFO: Exported cleaning audit trail to {self.report_dir}/cleaning_log.csv")
        print("\nCleaning Log:")
        print(pd.DataFrame(self.cleaning_log)[['step', 'rule', 'rows_before', 'rows_after']].to_string(index=False))


if __name__ == "__main__":
    import sys

    URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    processor = EmissionsDataProcessor(url=URL)

    # Reproducible by default: `python data_prep.py <snapshot.csv>` reruns on a saved, verified file
    if len(sys.argv) > 1:
        processor.ingest_snapshot(sys.argv[1])
    else:
        processor.ingest()
        processor.snapshot_raw("data/raw")

    processor.scope()
    processor.assign_splits({'train': 0.6, 'validation': 0.2, 'test': 0.2}, seed=42)
    # Pass 1: exploration and selection (selection steps see training countries only)
    processor.profile_missingness()
    processor.clean()
    processor.profile_outliers()
    processor.select_transformation()
    selected = processor.select_features_with_shap(top_n=4, required=['co2'])
    processor.stability_check(n_boot=30, top_n=4, required=['co2'])
    # Pass 2: rebuild every country on the chosen features only
    processor.rebuild(selected)
    processor.report_completeness()
    processor.export("data/shap_selected_emissions_data.csv")
