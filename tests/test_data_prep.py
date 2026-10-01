"""
Tests for the ETL pipeline, run against the synthetic `raw_owid` fixture (no download).
Each test pins down one documented cleaning decision.
"""
import os

import numpy as np
import pandas as pd
import pytest

from data_prep import EmissionsDataProcessor


@pytest.fixture
def processor(tmp_path, raw_owid):
    p = EmissionsDataProcessor(url=None, report_dir=str(tmp_path / 'reports'), viz_dir=str(tmp_path / 'viz'))
    p.raw_data = raw_owid
    return p


@pytest.fixture
def cleaned(processor):
    processor.scope()
    processor.profile_missingness()
    processor.clean()
    return processor


# --- Step 1: scope ------------------------------------------------------------
def test_scope_keeps_modern_years_and_real_countries_only(processor):
    df = processor.scope()
    assert df['year'].min() == 2000
    assert 'World' not in df['country'].values          # aggregate region, no ISO code
    assert 'extra_column' not in df.columns
    assert list(df.columns) == processor.ID_COLS + processor.CANDIDATE_FEATURES


# --- Missingness investigation ------------------------------------------------
def test_missingness_diagnosis_identifies_structural_gdp_gaps(processor):
    processor.scope()
    diag = processor.profile_missingness().set_index('feature')
    assert diag.loc['gdp', 'pattern'] == 'STRUCTURAL'
    assert diag.loc['gdp', 'years_fully_missing'] == '2023, 2024'
    assert diag.loc['gdp', 'countries_fully_missing'] == 1        # Tiny Isle
    assert diag.loc['population', 'pattern'] == 'COMPLETE'


def test_missingness_reports_are_written(processor):
    processor.scope()
    processor.profile_missingness()
    for name in ['missingness_by_year.csv', 'missingness_by_country.csv', 'missingness_diagnosis.csv']:
        assert os.path.exists(os.path.join(processor.report_dir, name))
    assert os.path.exists(os.path.join(processor.viz_dir, 'missingness_by_year.png'))


# --- Steps 2-4: cleaning ------------------------------------------------------
def test_unpublished_trailing_years_are_trimmed(cleaned):
    assert cleaned.clean_data['year'].max() == 2022


def test_uncovered_country_is_dropped_and_documented(cleaned):
    assert 'Tiny Isle' not in cleaned.clean_data['country'].values
    dropped = pd.read_csv(os.path.join(cleaned.report_dir, 'dropped_countries.csv'))
    assert dropped.set_index('country').loc['Tiny Isle', 'features_never_reported'] == 'gdp'


def test_interior_gaps_use_the_countrys_own_trend(cleaned):
    # Gapland methane is linear (20 + 0.1 per year since 1998) with interior gaps
    gap = cleaned.clean_data.set_index(['country', 'year']).loc['Gapland', 'methane']
    assert gap.loc[2004] == pytest.approx(20.6)
    assert gap.loc[2011] == pytest.approx(21.3)


def test_leading_gaps_are_dropped_not_extrapolated(cleaned):
    newstate_years = cleaned.clean_data.loc[cleaned.clean_data['country'] == 'Newstate', 'year']
    assert newstate_years.min() == 2003


def test_no_missing_values_remain(cleaned):
    assert cleaned.clean_data.isna().sum().sum() == 0


def test_cleaning_log_is_a_consistent_audit_trail(cleaned):
    log = pd.DataFrame(cleaned.cleaning_log)
    assert log['step'].iloc[0].startswith('1.')
    # Each step starts where the previous one ended
    assert (log['rows_before'].iloc[1:].values == log['rows_after'].iloc[:-1].values).all()
    assert (log['rows_removed'] == log['rows_before'] - log['rows_after']).all()


# --- Outlier investigation (new: written before implementation) ---------------
@pytest.fixture
def with_spike(processor):
    """Alpha's CO2 is entered 20x too high in 2010 only."""
    raw = processor.raw_data.copy()
    mask = (raw['country'] == 'Alpha') & (raw['year'] == 2010)
    raw.loc[mask, 'co2'] *= 20
    processor.raw_data = raw
    processor.scope()
    processor.profile_missingness()
    processor.clean()
    return processor


def test_profile_outliers_returns_raw_vs_log_comparison(with_spike):
    profile = with_spike.profile_outliers()
    assert set(profile['feature']) == set(with_spike.CANDIDATE_FEATURES)
    assert {'skew_raw', 'skew_log', 'pct_iqr_raw', 'pct_iqr_log'}.issubset(profile.columns)


def test_profile_outliers_flags_the_injected_spike_for_review(with_spike):
    with_spike.profile_outliers()
    review = with_spike.outlier_review
    hit = review[(review['country'] == 'Alpha') & (review['year'] == 2010) & (review['feature'] == 'co2')]
    assert len(hit) == 1
    assert hit['pattern'].iloc[0] == 'spike_and_revert'


def test_outliers_are_flagged_never_removed(with_spike):
    before = with_spike.clean_data.copy()
    with_spike.profile_outliers()
    pd.testing.assert_frame_equal(with_spike.clean_data, before)
    step = pd.DataFrame(with_spike.cleaning_log).iloc[-1]
    assert step['step'].startswith('4d')
    assert step['rows_removed'] == 0


def test_outlier_reports_and_charts_are_written(with_spike):
    with_spike.profile_outliers()
    for name in ['outlier_profile.csv', 'outlier_review.csv']:
        assert os.path.exists(os.path.join(with_spike.report_dir, name))
    for name in ['outlier_scale_comparison.png', 'outlier_spikes.png']:
        assert os.path.exists(os.path.join(with_spike.viz_dir, name))


# --- Step 5: SHAP selection and export ----------------------------------------
def test_shap_selection_keeps_ids_plus_top_n(cleaned):
    cleaned.select_features_with_shap(top_n=3)
    assert list(cleaned.clean_data.columns[:3]) == cleaned.ID_COLS
    assert cleaned.clean_data.shape[1] == 6
    ranking = pd.read_csv(os.path.join(cleaned.report_dir, 'shap_feature_importance.csv'))
    assert ranking['Selected'].sum() == 3
    assert ranking['SHAP_Importance'].is_monotonic_decreasing


def test_shap_top_n_is_capped_at_available_features(cleaned):
    cleaned.select_features_with_shap(top_n=50)
    assert cleaned.clean_data.shape[1] == len(cleaned.ID_COLS) + len(cleaned.CANDIDATE_FEATURES)


def test_export_writes_dataset_and_audit_trail(cleaned, tmp_path):
    cleaned.select_features_with_shap(top_n=4)
    out = tmp_path / 'data' / 'out.csv'
    cleaned.export(str(out))
    assert pd.read_csv(out).shape == cleaned.clean_data.shape
    assert os.path.exists(os.path.join(cleaned.report_dir, 'cleaning_log.csv'))


# --- Transform selection (new: written before implementation) ----------------
@pytest.fixture
def transformed(cleaned):
    cleaned.select_transformation()
    return cleaned


def test_transformation_is_chosen_from_candidates(transformed):
    from transforms import PARAMETER_FREE, FITTED
    assert transformed.transform in PARAMETER_FREE + FITTED
    assert transformed.transform != 'none'           # synthetic features span orders of magnitude


def test_transform_reports_are_written(transformed):
    for name in ['transform_comparison.csv', 'transform_lambda_tests.csv', 'transform_selection.csv']:
        assert os.path.exists(os.path.join(transformed.report_dir, name))
    lam = pd.read_csv(os.path.join(transformed.report_dir, 'transform_lambda_tests.csv'))
    assert set(lam['feature']) == set(transformed.CANDIDATE_FEATURES)
    sel = pd.read_csv(os.path.join(transformed.report_dir, 'transform_selection.csv'))
    assert sel.loc[sel['chosen'], 'transform'].tolist() == [transformed.transform]


def test_transform_selection_is_logged_without_changing_rows(transformed):
    step = pd.DataFrame(transformed.cleaning_log).iloc[-1]
    assert step['step'].startswith('4e')
    assert step['rows_removed'] == 0
    assert transformed.transform in step['detail']


def test_shap_baseline_uses_the_selected_transform(transformed):
    transformed.select_features_with_shap(top_n=3)
    ranking = pd.read_csv(os.path.join(transformed.report_dir, 'shap_feature_importance.csv'))
    assert (ranking['Transform'] == transformed.transform).all()


# --- Required domain features (new: written before implementation) -----------
def test_required_feature_is_kept_even_when_shap_ranks_it_last(transformed, monkeypatch):
    # Force co2 to the bottom of the SHAP ranking
    import data_prep
    real = data_prep.shap.TreeExplainer

    class LowCo2Explainer:
        def __init__(self, model):
            self.inner = real(model)

        def shap_values(self, X):
            values = self.inner.shap_values(X)
            values[:, list(X.columns).index('co2')] = 0.0
            return values

    monkeypatch.setattr(data_prep.shap, 'TreeExplainer', LowCo2Explainer)
    transformed.select_features_with_shap(top_n=4, required=['co2'])
    kept = list(transformed.clean_data.columns[len(transformed.ID_COLS):])
    assert 'co2' in kept
    assert len(kept) == 4


def test_required_features_fill_the_budget_first(transformed):
    transformed.select_features_with_shap(top_n=4, required=['co2'])
    ranking = pd.read_csv(os.path.join(transformed.report_dir, 'shap_feature_importance.csv'))
    selected = ranking[ranking['Selected']]
    assert len(selected) == 4
    assert selected.set_index('Feature').loc['co2', 'Selection_Reason'] == 'required (core metric)'
    assert (selected['Selection_Reason'] == 'top SHAP').sum() == 3


def test_missing_required_feature_is_an_error(transformed):
    transformed.clean_data = transformed.clean_data.drop(columns=['co2'])
    with pytest.raises(ValueError, match='co2'):
        transformed.select_features_with_shap(top_n=4, required=['co2'])


def test_selection_decision_is_logged(transformed):
    transformed.select_features_with_shap(top_n=4, required=['co2'])
    step = pd.DataFrame(transformed.cleaning_log).iloc[-1]
    assert 'required' in step['detail'] and 'co2' in step['detail']


# =============================================================================
# Reproducibility, splitting, two-pass cleaning, stability, completeness
# (new: written before implementation)
# =============================================================================
import json

SPLIT = {'train': 0.5, 'validation': 0.25, 'test': 0.25}


# --- raw snapshot + checksum ---------------------------------------------------
def test_snapshot_writes_csv_and_checksum_manifest(processor, tmp_path):
    path = processor.snapshot_raw(str(tmp_path / 'raw'), source_url='https://example.org/owid.csv')
    manifest = json.load(open(str(path) + '.manifest.json'))
    assert manifest['source_url'] == 'https://example.org/owid.csv'
    assert manifest['rows'] == len(processor.raw_data)
    assert len(manifest['sha256']) == 64


def test_snapshot_round_trips_and_verifies(processor, tmp_path):
    path = processor.snapshot_raw(str(tmp_path / 'raw'))
    fresh = EmissionsDataProcessor(url=None, report_dir=processor.report_dir, viz_dir=processor.viz_dir)
    fresh.ingest_snapshot(path)
    assert fresh.raw_data.shape == processor.raw_data.shape


def test_tampered_snapshot_is_rejected(processor, tmp_path):
    path = processor.snapshot_raw(str(tmp_path / 'raw'))
    with open(path, 'a', encoding='utf-8') as f:
        f.write('tampered\n')
    fresh = EmissionsDataProcessor(url=None, report_dir=processor.report_dir, viz_dir=processor.viz_dir)
    with pytest.raises(ValueError, match='checksum'):
        fresh.ingest_snapshot(path)


# --- split ------------------------------------------------------------------------
@pytest.fixture
def split_processor(processor):
    processor.scope()
    processor.assign_splits(SPLIT, seed=0)
    return processor


def test_every_scoped_country_gets_one_split(split_processor):
    split = split_processor.country_split
    assert set(split.index) == set(split_processor.scoped_data['country'])
    assert set(split) <= set(SPLIT)
    assert os.path.exists(os.path.join(split_processor.report_dir, 'split_summary.csv'))


def test_training_view_contains_only_training_countries(split_processor):
    split_processor.profile_missingness()
    split_processor.clean()
    train = split_processor.training_view()
    train_countries = set(split_processor.country_split[lambda s: s == 'train'].index)
    # A subset: training countries removed by cleaning (e.g. Tiny Isle, no GDP) are absent
    assert set(train['country']) <= train_countries
    assert set(train['country']) == train_countries & set(split_processor.clean_data['country'])


def test_selection_steps_use_training_rows_only(split_processor, monkeypatch):
    split_processor.profile_missingness()
    split_processor.clean()
    seen = []
    import data_prep
    real = data_prep.compare_transforms
    monkeypatch.setattr(data_prep, 'compare_transforms', lambda df, f: seen.append(set(df['country'])) or real(df, f))
    split_processor.select_transformation()
    train_countries = set(split_processor.country_split[lambda s: s == 'train'].index)
    assert seen and seen[0] <= train_countries


# --- stability ---------------------------------------------------------------------
def test_stability_reports_selection_frequency(split_processor):
    split_processor.profile_missingness()
    split_processor.clean()
    split_processor.select_transformation()
    stab = split_processor.stability_check(n_boot=4, top_n=3, required=['co2'], seed=0)
    assert set(stab['feature']) == set(split_processor.CANDIDATE_FEATURES)
    assert stab['selection_frequency'].between(0, 1).all()
    assert stab.set_index('feature').loc['co2', 'selection_frequency'] == 1.0
    assert os.path.exists(os.path.join(split_processor.report_dir, 'feature_stability.csv'))


# --- pass 2: rebuild on the selected features --------------------------------------
@pytest.fixture
def rebuilt(split_processor):
    p = split_processor
    p.profile_missingness()
    p.clean()
    p.select_transformation()
    p.rebuild(['co2', 'methane', 'population'])
    return p


def test_rebuild_recovers_countries_lost_only_to_unused_features(rebuilt):
    # Tiny Isle was dropped in pass 1 for never reporting gdp; gdp is not selected
    assert 'Tiny Isle' in set(rebuilt.clean_data['country'])


def test_rebuild_keeps_years_lagging_only_in_unused_features(rebuilt):
    # 2023-2024 were trimmed because of gdp alone
    assert rebuilt.clean_data['year'].max() == 2024


def test_rebuilt_data_has_selected_features_and_split(rebuilt):
    assert list(rebuilt.clean_data.columns) == rebuilt.ID_COLS + ['co2', 'methane', 'population', 'split']
    assert rebuilt.clean_data.isna().sum().sum() == 0
    assert rebuilt.clean_data['split'].isin(SPLIT).all()


def test_rebuild_is_logged_as_a_second_pass(rebuilt):
    log = pd.DataFrame(rebuilt.cleaning_log)
    second = log[log['pass'] == 2]
    assert len(second) >= 2
    assert second.iloc[0]['rows_before'] == len(rebuilt.scoped_data)
    assert (second['rows_before'].iloc[1:].values == second['rows_after'].iloc[:-1].values).all()


# --- completeness as a quality finding ---------------------------------------------
def test_completeness_report_covers_selected_features(rebuilt):
    report = rebuilt.report_completeness()
    assert set(report['feature']) == {'co2', 'methane', 'population'}
    by_country = pd.read_csv(os.path.join(rebuilt.report_dir, 'completeness_by_country.csv'))
    assert {'country', 'split', 'status'}.issubset(by_country.columns)
    assert set(by_country['status']) <= {'complete', 'partial', 'missing'}
