"""
Tests for the application layer: schema rules, error injection, the validator, and the
link between exploratory outlier review and model anomalies.
"""
import numpy as np
import pandas as pd
import pytest

import app
from validation import (EmissionsDataValidator, add_intensity_ratios, attach_review_flags,
                        build_schema, inject_synthetic_errors)


def failures_for(df):
    v = EmissionsDataValidator(df)
    return v.validate_schema()


# --- Schema: every rule fires on its own error, and only on it -----------------
def test_clean_panel_passes_every_rule(panel):
    assert failures_for(panel).empty


@pytest.mark.parametrize("column, bad_value, expected_rule", [
    ('co2', -1.0, 'emissions >= 0'),
    ('population', 0.0, 'population > 0'),
    ('gdp', -5.0, 'gdp > 0'),
    ('ghg_per_capita', 400.0, '-50 <= ghg_per_capita <= 150'),
    ('iso_code', 'xx1', 'valid_iso_code'),
    ('year', 1850, 'valid_year_range'),
    ('co2', np.nan, 'not_nullable'),
])
def test_each_rule_catches_its_error(panel, column, bad_value, expected_rule):
    panel = panel.copy()
    panel[column] = panel[column].astype(object) if isinstance(bad_value, str) else panel[column]
    panel.loc[5, column] = bad_value
    failures = failures_for(panel)
    assert expected_rule in failures['rule'].values
    assert set(failures['row']) == {5}


def test_small_negative_net_ghg_is_legal_carbon_sink(panel):
    panel = panel.copy()
    panel.loc[3, 'ghg_per_capita'] = -0.018
    assert failures_for(panel).empty


def test_duplicate_country_year_reported_once_per_row(panel):
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    failures = failures_for(dup)
    assert (failures['rule'] == 'multiple_fields_uniqueness').sum() == 2    # original + copy
    assert failures['column'].eq('country, year').all()


def test_cross_field_rule_catches_population_unit_error(panel):
    panel = panel.copy()
    panel.loc[7, 'population'] /= 1000          # population entered in thousands
    failures = failures_for(panel)
    assert 'plausible_co2_per_capita' in failures['rule'].values


def test_every_failure_has_a_plain_english_reason(panel):
    bad, _ = inject_synthetic_errors(panel)
    failures = failures_for(bad)
    assert not failures.empty
    assert (failures['why'] != failures['rule']).all()


def test_unknown_numeric_feature_gets_default_rule(panel):
    panel = panel.assign(water_use=1.0)
    schema = build_schema(panel)
    assert 'water_use' in schema.columns
    panel.loc[0, 'water_use'] = -1
    assert 'value >= 0' in failures_for(panel)['rule'].values


# --- Synthetic error injection -------------------------------------------------
def test_injection_answer_key_covers_both_layers(panel):
    bad, key = inject_synthetic_errors(panel)
    assert len(bad) == len(panel) + 1                    # one duplicated row
    assert set(key['expected_to_be_caught_by']) == {'Schema', 'Anomaly detection'}
    assert (key['expected_to_be_caught_by'] == 'Schema').sum() == 5


def test_injection_does_not_modify_the_input(panel):
    original = panel.copy()
    inject_synthetic_errors(panel)
    pd.testing.assert_frame_equal(panel, original)


# --- Validator -----------------------------------------------------------------
def test_intensity_ratios_have_correct_units(panel):
    out, added = add_intensity_ratios(panel.head(1))
    row = out.iloc[0]
    assert set(added) == {'co2_per_capita', 'co2_per_gdp', 'gdp_per_capita'}
    assert row['co2_per_capita'] == pytest.approx(row['co2'] * 1e6 / row['population'])
    assert row['co2_per_gdp'] == pytest.approx(row['co2'] * 1e9 / row['gdp'])


def test_ratios_skipped_when_inputs_missing(panel):
    _, added = add_intensity_ratios(panel.drop(columns=['gdp']))
    assert added == ['co2_per_capita']


def test_invalid_rows_are_excluded_from_model_fit(panel):
    bad, key = inject_synthetic_errors(panel)
    v = EmissionsDataValidator(bad)
    v.run()
    assert v.invalid_idx.isdisjoint(v.scored.index)


def test_contamination_controls_flag_rate(panel):
    v = EmissionsDataValidator(panel, contamination=0.05)
    v.run()
    assert v.scored['is_anomaly'].mean() == pytest.approx(0.05, abs=0.01)


def test_every_anomaly_has_an_explanation_and_shap_row(panel):
    v = EmissionsDataValidator(panel)
    v.run()
    flagged = v.scored[v.scored['is_anomaly']]
    assert flagged['explanation'].notna().all()
    assert list(v.shap_values.index) == list(flagged.index)


def test_shap_sign_convention_positive_means_more_anomalous(panel):
    v = EmissionsDataValidator(panel)
    v.run()
    # Flagged rows should on average be pushed toward anomaly
    assert v.shap_values.sum(axis=1).mean() > 0


def test_ratios_catch_the_unit_error_the_schema_cannot(panel):
    bad, key = inject_synthetic_errors(panel)
    unit_row = key.loc[key['expected_to_be_caught_by'] == 'Anomaly detection', 'row'].iloc[0]
    v = EmissionsDataValidator(bad, use_ratios=True)
    v.run()
    assert unit_row not in v.invalid_idx
    assert v.scored.loc[unit_row, 'is_anomaly']
    assert 'per_capita' in v.scored.loc[unit_row, 'top_driver'] or 'per_gdp' in v.scored.loc[unit_row, 'top_driver']


# --- New: link model anomalies to the exploratory outlier review --------------
def test_review_flags_attach_to_matching_country_year():
    scored = pd.DataFrame({'country': ['A', 'A', 'B'], 'year': [2010, 2011, 2010],
                           'is_anomaly': [True, False, True]})
    review = pd.DataFrame({'country': ['A', 'A'], 'year': [2010, 2010], 'feature': ['co2', 'gdp'],
                           'pattern': ['spike_and_revert', 'level_shift']})
    out = attach_review_flags(scored, review)
    assert out.loc[0, 'eda_review'] == 'co2: spike_and_revert; gdp: level_shift'
    assert out.loc[1, 'eda_review'] == ''
    assert out.loc[2, 'eda_review'] == ''
    assert list(out.index) == list(scored.index)


def test_review_flags_tolerate_missing_review_file():
    scored = pd.DataFrame({'country': ['A'], 'year': [2010], 'is_anomaly': [True]})
    assert attach_review_flags(scored, None)['eda_review'].tolist() == ['']


# --- Accessibility --------------------------------------------------------------
def test_charts_use_the_theme_roles():
    from design import TOKENS
    # flagged = rust, normal = teal, Medium = periwinkle (see design.py for the accessibility rules)
    assert {app.ANOMALY_COLOR, app.TOWARD_ANOMALY} == {TOKENS['rust']}
    assert {app.SERIES_BLUE, app.TOWARD_NORMAL} == {TOKENS['teal']}
    assert app.MEDIUM_COLOR == TOKENS['periwinkle']
    assert not hasattr(app, 'ANOMALY_RED')


# --- Transform selection in the app (new: written before implementation) -----
from app import recommended_transform


@pytest.mark.parametrize('name', ['none', 'signed_log', 'asinh', 'cbrt', 'yeo_johnson'])
def test_validator_runs_with_every_transform(panel, name):
    v = EmissionsDataValidator(panel, transform=name)
    summary = v.run()
    assert summary['anomalies'] > 0
    assert v.transform == name


def test_fitted_transform_is_fit_once_on_all_rows(panel):
    v = EmissionsDataValidator(panel, transform='yeo_johnson')
    v.run()
    # The model matrix covers every scored row; explanations reuse it rather than re-fitting
    assert list(v.X_model.index) == list(v.scored.index)


def test_unknown_transform_is_rejected_up_front(panel):
    with pytest.raises(ValueError, match='Unknown transform'):
        EmissionsDataValidator(panel, transform='log2')


def test_recommended_transform_reads_the_report(tmp_path):
    pd.DataFrame({'transform': ['asinh', 'signed_log'], 'chosen': [True, False],
                  'reason': ['because', '']}).to_csv(tmp_path / 'transform_selection.csv', index=False)
    assert recommended_transform(str(tmp_path)) == ('asinh', 'because')


def test_recommended_transform_falls_back_without_report(tmp_path):
    name, reason = recommended_transform(str(tmp_path))
    assert name == 'signed_log'
    assert 'data_prep.py' in reason


# --- Per-capita ratios for every emissions quantity (new: written first) ------
def test_every_extensive_quantity_gets_a_per_capita_ratio(panel):
    df = panel.assign(methane=panel['co2'] * 0.3, nitrous_oxide=panel['co2'] * 0.05,
                      primary_energy_consumption=panel['co2'] * 4)
    out, added = add_intensity_ratios(df.head(1))
    assert {'methane_per_capita', 'nitrous_oxide_per_capita', 'energy_per_capita'}.issubset(added)
    row = out.iloc[0]
    assert row['methane_per_capita'] == pytest.approx(row['methane'] * 1e6 / row['population'])


def test_methane_unit_error_is_caught_via_per_capita_ratio(panel):
    df = panel.assign(methane=panel['co2'] * 0.3).drop(columns=['gdp'])
    bad, key = inject_synthetic_errors(df)
    unit = key[key['expected_to_be_caught_by'] == 'Anomaly detection'].iloc[0]
    assert unit['column'] == 'methane'
    v = EmissionsDataValidator(bad, transform='asinh')
    v.run()
    assert v.scored.loc[unit['row'], 'is_anomaly']


# --- Defaults chosen by the evaluation harness (new: written first) -----------
import json

from app import recommended_config


def test_recommended_config_reads_the_harness_choice(tmp_path):
    (tmp_path / 'model_config.json').write_text(json.dumps({
        'transform': 'cbrt', 'use_ratios': False, 'contamination': 0.03, 'range_z': 6.0,
        'reason': 'best recall', 'test': {'recall_statistical': 0.8}}))
    config, reason = recommended_config(str(tmp_path))
    assert config == {'transform': 'cbrt', 'use_ratios': False, 'contamination': 0.03, 'range_z': 6.0}
    assert reason == 'best recall'


def test_recommended_config_falls_back_to_statistical_transform_choice(tmp_path):
    pd.DataFrame({'transform': ['asinh'], 'chosen': [True], 'reason': ['normality']}) \
        .to_csv(tmp_path / 'transform_selection.csv', index=False)
    config, reason = recommended_config(str(tmp_path))
    assert config['transform'] == 'asinh'
    assert 'evaluation.py' in reason
