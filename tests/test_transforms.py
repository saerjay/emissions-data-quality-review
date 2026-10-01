"""
Transformation selection (written before transforms.py existed).

Question under test: which transformation makes each feature closest to symmetric/normal,
and is a plain log statistically adequate or does the data want a different power?
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from transforms import (PARAMETER_FREE, FITTED, apply_transform, compare_transforms,
                        lambda_test, select_transform)


@pytest.fixture
def lognormal():
    return pd.Series(np.random.default_rng(1).lognormal(mean=3, sigma=1.5, size=2000))


@pytest.fixture
def normal():
    return pd.Series(np.random.default_rng(2).normal(loc=100, scale=10, size=2000))


# --- apply_transform ---------------------------------------------------------
def test_signed_log_is_log1p_for_positive_values():
    x = pd.Series([0.0, 1.0, 50.0])
    assert np.allclose(apply_transform(x, 'signed_log'), np.log1p(x))


@pytest.mark.parametrize('name', ['signed_log', 'asinh', 'cbrt', 'yeo_johnson'])
def test_transforms_accept_zero_and_negative_values(name):
    # Net GHG per capita can be negative (carbon sinks); co2 can be exactly zero
    out = apply_transform(pd.Series([-2.0, 0.0, 0.5, 10.0, 1000.0]), name)
    assert np.isfinite(out).all()
    assert out.is_monotonic_increasing                     # order is preserved


def test_box_cox_refuses_non_positive_data():
    with pytest.raises(ValueError, match='positive'):
        apply_transform(pd.Series([0.0, 1.0, 2.0]), 'box_cox')


def test_none_is_identity():
    x = pd.Series([3.0, 1.0, 2.0])
    pd.testing.assert_series_equal(apply_transform(x, 'none'), x)


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match='Unknown transform'):
        apply_transform(pd.Series([1.0]), 'log2')


def test_transform_works_column_wise_on_dataframes():
    df = pd.DataFrame({'a': [1.0, 10.0, 100.0], 'b': [2.0, 20.0, 200.0]})
    out = apply_transform(df, 'yeo_johnson')
    assert list(out.columns) == ['a', 'b']
    assert out.shape == df.shape


def test_log_base_is_irrelevant_so_it_is_not_a_candidate(lognormal):
    # log10(x) = ln(x) / ln(10): a rescale, so every shape statistic is identical
    w_ln = stats.shapiro(np.log(lognormal)).statistic
    w_10 = stats.shapiro(np.log10(lognormal)).statistic
    assert w_ln == pytest.approx(w_10)
    assert 'log10' not in PARAMETER_FREE + FITTED


# --- compare_transforms --------------------------------------------------------
def test_comparison_scores_every_candidate_per_feature(lognormal, normal):
    df = pd.DataFrame({'skewed': lognormal, 'symmetric': normal})
    table = compare_transforms(df, ['skewed', 'symmetric'])
    assert {'feature', 'transform', 'shapiro_w', 'skew'}.issubset(table.columns)
    assert set(table['transform']) == set(PARAMETER_FREE + FITTED)
    assert len(table) == 2 * len(PARAMETER_FREE + FITTED)


def test_log_family_beats_raw_on_skewed_data(lognormal):
    table = compare_transforms(pd.DataFrame({'x': lognormal}), ['x']).set_index('transform')
    assert table.loc['none', 'shapiro_w'] < 0.8
    # Fitted power transforms recover the exact log for lognormal data...
    for name in ['box_cox', 'yeo_johnson']:
        assert table.loc[name, 'shapiro_w'] > 0.995
    # ...while log1p/asinh are close but bent by their offset on values below 1
    for name in ['signed_log', 'asinh']:
        assert 0.97 < table.loc[name, 'shapiro_w'] < table.loc['box_cox', 'shapiro_w']


def test_box_cox_is_skipped_not_crashed_for_non_positive_features():
    x = pd.Series(np.r_[np.random.default_rng(3).lognormal(size=500), [0.0]])
    table = compare_transforms(pd.DataFrame({'x': x}), ['x']).set_index('transform')
    assert np.isnan(table.loc['box_cox', 'shapiro_w'])


# --- lambda_test: likelihood-ratio test of H0: lambda = 0 (log is optimal) ------
def test_log_is_not_rejected_for_lognormal_data(lognormal):
    result = lambda_test(lognormal)
    assert result['family'] == 'box_cox'
    assert result['lambda_hat'] == pytest.approx(0, abs=0.05)
    assert result['ci_low'] < 0 < result['ci_high']
    assert result['p_value'] > 0.05
    assert result['log_supported']


def test_log_is_rejected_for_symmetric_data(normal):
    result = lambda_test(normal)
    assert result['p_value'] < 0.001
    assert not result['log_supported']
    assert not (result['ci_low'] < 0 < result['ci_high'])


def test_lambda_test_falls_back_to_yeo_johnson_with_non_positive_values(lognormal):
    result = lambda_test(pd.concat([lognormal, pd.Series([-0.5, 0.0])]))
    assert result['family'] == 'yeo_johnson'
    assert np.isfinite(result['lr_stat'])


# --- select_transform ----------------------------------------------------------
def _comparison(scores):
    return pd.DataFrame([{'feature': f, 'transform': t, 'shapiro_w': w, 'skew': 0.0}
                         for f, per in scores.items() for t, w in per.items()])


def test_prefers_parameter_free_when_fitted_gain_is_negligible():
    comp = _comparison({'a': {'none': 0.3, 'signed_log': 0.980, 'asinh': 0.985, 'cbrt': 0.8,
                              'box_cox': 0.990, 'yeo_johnson': 0.990}})
    choice, reason, summary = select_transform(comp, tolerance=0.01)
    assert choice == 'asinh'
    assert 'parameter-free' in reason
    assert summary.iloc[0]['transform'] in ('box_cox', 'yeo_johnson')   # ranked by mean W


def test_chooses_fitted_transform_when_gain_is_material():
    comp = _comparison({'a': {'none': 0.3, 'signed_log': 0.90, 'asinh': 0.91, 'cbrt': 0.8,
                              'box_cox': 0.99, 'yeo_johnson': 0.98}})
    choice, reason, _ = select_transform(comp, tolerance=0.01)
    assert choice == 'box_cox'


def test_transform_unavailable_for_any_feature_is_not_eligible():
    comp = _comparison({'a': {'none': 0.3, 'signed_log': 0.90, 'asinh': 0.91, 'cbrt': 0.8,
                              'box_cox': 0.999, 'yeo_johnson': 0.95},
                        'b': {'none': 0.3, 'signed_log': 0.90, 'asinh': 0.91, 'cbrt': 0.8,
                              'box_cox': np.nan, 'yeo_johnson': 0.95}})
    choice, _, _ = select_transform(comp, tolerance=0.01)
    assert choice == 'yeo_johnson'


# --- Fit on training data, apply unchanged elsewhere (new: written first) -----
from transforms import fit_transform_params


def test_parameter_free_transforms_have_no_params():
    df = pd.DataFrame({'a': [1.0, 10.0, 100.0]})
    assert fit_transform_params(df, 'asinh') == {}


def test_fitted_params_are_learned_per_column():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({'a': rng.lognormal(size=300), 'b': rng.normal(50, 5, size=300)})
    params = fit_transform_params(df, 'yeo_johnson')
    assert set(params) == {'a', 'b'}
    assert params['a'] != pytest.approx(params['b'])


def test_applying_stored_params_does_not_refit():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({'a': rng.lognormal(size=300)})
    other = pd.DataFrame({'a': rng.lognormal(mean=2, size=50)})   # different distribution
    params = fit_transform_params(train, 'yeo_johnson')
    applied = apply_transform(other, 'yeo_johnson', params=params)
    expected = stats.yeojohnson(other['a'].values, lmbda=params['a'])
    assert np.allclose(applied['a'], expected)
    # A refit on `other` would give a different result
    assert not np.allclose(applied['a'], apply_transform(other, 'yeo_johnson')['a'])


def test_box_cox_params_round_trip():
    rng = np.random.default_rng(6)
    train = pd.DataFrame({'a': rng.lognormal(size=200)})
    params = fit_transform_params(train, 'box_cox')
    out = apply_transform(train, 'box_cox', params=params)
    assert np.allclose(out['a'], stats.boxcox(train['a'].values, lmbda=params['a']))
