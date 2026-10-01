"""
Choosing a variance-stabilising transformation with statistical tests.

Why transform at all? sklearn's IsolationForest draws each split uniformly between a
feature's min and max. On a right-skewed feature (co2 spans 0.04 to 12,000 Mt) nearly
every split lands in the empty upper tail, so the model can only "see" the biggest
countries. A transform that makes the distribution symmetric spreads the splits over the
range where the data actually lives.

Candidates
  parameter-free: none, signed_log (sign * log1p|x|), asinh, cbrt
  fitted:         box_cox (x > 0 only), yeo_johnson (any sign), lambda estimated by MLE
  (log base is not a candidate: log10 x = ln x / ln 10 is a rescale with identical shape.)

Tests
  1. Shapiro-Wilk W per feature and transform. With thousands of rows every p-value
     rejects exact normality, so W itself (1.0 = perfectly normal) is used as the effect size.
  2. Likelihood-ratio test of H0: lambda = 0 in the Box-Cox (or Yeo-Johnson) family,
     i.e. "is a log the optimal power transform?", with a 95% profile-likelihood CI.

Decision rule
  Use the parameter-free transform with the highest mean W, unless the best fitted
  transform beats it by more than `tolerance` W. Parameter-free transforms are preferred
  because they are interpretable, need no stored lambda, and apply identically to new data.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

PARAMETER_FREE = ['none', 'signed_log', 'asinh', 'cbrt']
FITTED = ['box_cox', 'yeo_johnson']
SHAPIRO_MAX_N = 5000     # Shapiro-Wilk p-values are unreliable above this; W is sampled


def signed_log(x):
    """log1p that preserves sign, so small negative values (carbon sinks) are allowed."""
    return np.sign(x) * np.log1p(np.abs(x))


def fit_transform_params(df, name):
    """
    Learns a fitted transform's lambda per column (on training data only).
    Parameter-free transforms return {}. Pass the result to apply_transform(params=...)
    so validation and test data are transformed with the training lambda, never refitted.
    """
    if name == 'box_cox':
        return {col: float(stats.boxcox(df[col].dropna().astype(float).values)[1]) for col in df.columns}
    if name == 'yeo_johnson':
        return {col: float(stats.yeojohnson(df[col].dropna().astype(float).values)[1]) for col in df.columns}
    return {}


def _transform_series(s, name, lmbda=None):
    s = s.astype(float)
    if name == 'none':
        return s
    if name == 'signed_log':
        return signed_log(s)
    if name == 'asinh':
        return np.arcsinh(s)
    if name == 'cbrt':
        return np.cbrt(s)
    if name == 'box_cox':
        if (s <= 0).any():
            raise ValueError("box_cox requires strictly positive values")
        out = stats.boxcox(s.values, lmbda=lmbda) if lmbda is not None else stats.boxcox(s.values)[0]
        return pd.Series(out, index=s.index, name=s.name)
    if name == 'yeo_johnson':
        out = stats.yeojohnson(s.values, lmbda=lmbda) if lmbda is not None else stats.yeojohnson(s.values)[0]
        return pd.Series(out, index=s.index, name=s.name)
    raise ValueError(f"Unknown transform '{name}'. Choose from {PARAMETER_FREE + FITTED}")


def apply_transform(x, name, params=None):
    """
    Applies a named transform to a Series, or column by column to a DataFrame.
    For fitted transforms, `params` (from fit_transform_params) fixes lambda per column;
    without it lambda is fitted on `x` itself.
    """
    params = params or {}
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: _transform_series(col, name, params.get(col.name)))
    return _transform_series(x, name, params.get(x.name))


def _shapiro_w(values, seed=0):
    values = np.asarray(values)
    if len(values) > SHAPIRO_MAX_N:
        values = np.random.default_rng(seed).choice(values, SHAPIRO_MAX_N, replace=False)
    return stats.shapiro(values).statistic


def compare_transforms(df, features):
    """Shapiro-Wilk W and skewness for every (feature, transform) pair; NaN when not applicable."""
    rows = []
    for col in features:
        x = df[col].dropna().astype(float)
        for name in PARAMETER_FREE + FITTED:
            try:
                y = apply_transform(x, name)
                w, skew = _shapiro_w(y), stats.skew(y)
            except ValueError:
                w, skew = np.nan, np.nan
            rows.append({'feature': col, 'transform': name,
                         'kind': 'fitted' if name in FITTED else 'parameter-free',
                         'shapiro_w': round(float(w), 4), 'skew': round(float(skew), 3)})
    return pd.DataFrame(rows)


def lambda_test(x, alpha=0.05):
    """
    Likelihood-ratio test of H0: lambda = 0 (log is the optimal power transform).
    Box-Cox when every value is positive, otherwise Yeo-Johnson (whose lambda = 0 is log1p
    for non-negative values).
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if (x > 0).all():
        family, llf, lam_hat = 'box_cox', (lambda l: stats.boxcox_llf(l, x)), stats.boxcox_normmax(x, method='mle')
    else:
        family, llf, lam_hat = 'yeo_johnson', (lambda l: stats.yeojohnson_llf(l, x)), stats.yeojohnson_normmax(x)
    lam_hat = float(lam_hat)
    ll_max = llf(lam_hat)

    lr_stat = max(2 * (ll_max - llf(0.0)), 0.0)
    p_value = float(stats.chi2.sf(lr_stat, df=1))

    # Profile-likelihood CI: every lambda whose LR statistic stays below the chi-square cut-off
    cutoff = stats.chi2.ppf(1 - alpha, df=1) / 2
    gap = lambda l: ll_max - llf(l) - cutoff

    def bound(direction):
        step = 0.05
        for _ in range(60):
            edge = lam_hat + direction * step
            if gap(edge) > 0:
                lo, hi = sorted([lam_hat, edge])
                return optimize.brentq(gap, lo, hi)
            step *= 1.5
        return direction * np.inf

    return {
        'family': family,
        'lambda_hat': round(lam_hat, 4),
        'ci_low': round(bound(-1), 4),
        'ci_high': round(bound(+1), 4),
        'lr_stat': round(lr_stat, 2),
        'p_value': p_value,
        'log_supported': p_value >= alpha,
    }


def select_transform(comparison, tolerance=0.01):
    """
    Returns (chosen transform, plain-English reason, summary ranked by mean W).
    Only transforms that apply to every feature are eligible.
    """
    pivot = comparison.pivot_table(index='transform', columns='feature', values='shapiro_w', dropna=False)
    skew = comparison.assign(abs_skew=comparison['skew'].abs()).groupby('transform')['abs_skew'].mean()
    summary = pd.DataFrame({
        'mean_shapiro_w': pivot.mean(axis=1, skipna=False),
        'min_shapiro_w': pivot.min(axis=1, skipna=False),
        'mean_abs_skew': skew,
    })
    summary['kind'] = ['fitted' if t in FITTED else 'parameter-free' for t in summary.index]
    summary['eligible'] = summary['mean_shapiro_w'].notna()
    summary = summary.sort_values('mean_shapiro_w', ascending=False, na_position='last')
    summary = summary.rename_axis('transform').reset_index()

    eligible = summary[summary['eligible']]
    best_fixed = eligible[eligible['kind'] == 'parameter-free'].iloc[0]
    fitted = eligible[eligible['kind'] == 'fitted']
    gain = fitted.iloc[0]['mean_shapiro_w'] - best_fixed['mean_shapiro_w'] if len(fitted) else -np.inf

    if gain > tolerance:
        best = fitted.iloc[0]
        choice = best['transform']
        reason = (f"{choice} (fitted lambda) improves mean Shapiro-Wilk W by {gain:.4f} over the best "
                  f"parameter-free option ({best_fixed['transform']}), more than the {tolerance} tolerance.")
    else:
        choice = best_fixed['transform']
        detail = (f"the best fitted option ({fitted.iloc[0]['transform']}) gains only {gain:+.4f} W"
                  if len(fitted) else "no fitted option applies to every feature")
        reason = (f"{choice} has the highest mean Shapiro-Wilk W ({best_fixed['mean_shapiro_w']:.4f}) of the "
                  f"parameter-free transforms; {detail}, within the {tolerance} tolerance, so the simpler, "
                  f"interpretable transform is preferred.")
    return choice, reason, summary
