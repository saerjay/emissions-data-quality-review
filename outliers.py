"""
Outlier tests for exploratory analysis.

Two different questions, two different tools:
  1. Cross-sectional: is this value extreme compared with *other countries*?
     -> IQR fences and robust z-scores, compared on the raw and log scales.
  2. Temporal: is this value extreme compared with *the same country's own history*?
     -> robust z-score of year-over-year log changes, split into
        spike_and_revert / rebound / level_shift patterns.

Handling policy (see EmissionsDataProcessor.profile_outliers):
  - Cross-sectional extremes in emissions data are mostly genuine heavy tails (China,
    the US), so they are kept and handled with a log transform, not removed or clipped.
  - Temporal spikes may be real events (war, COVID) or data errors, and that cannot be
    decided automatically, so they are flagged for human review and never deleted.
"""
import numpy as np
import pandas as pd

from transforms import signed_log

MAD_TO_SD = 0.6745            # MAD of a normal distribution = 0.6745 standard deviations
MEAN_AD_TO_SD = 1.253314      # fallback constant when MAD is zero (Iglewicz & Hoaglin)


def iqr_outliers(series, k=1.5):
    """Tukey fences: True where a value is below Q1 - k*IQR or above Q3 + k*IQR."""
    q1, q3 = series.quantile([0.25, 0.75])
    spread = q3 - q1
    return (series < q1 - k * spread) | (series > q3 + k * spread)


def robust_zscore(series):
    """
    z-score built from the median and median absolute deviation (MAD).
    Unlike mean/std, the outliers being hunted cannot inflate the yardstick.
    """
    median = series.median()
    deviation = series - median
    mad = deviation.abs().median()
    if mad > 0:
        return MAD_TO_SD * deviation / mad
    mean_ad = deviation.abs().mean()
    if mean_ad > 0:
        return deviation / (MEAN_AD_TO_SD * mean_ad)
    return pd.Series(0.0, index=series.index)


def outlier_profile(df, features, k=1.5, z_threshold=3.5):
    """
    Per-feature summary showing how many values each test flags on the raw and log
    scales. A large drop from raw to log means the "outliers" were really skew.
    """
    rows = []
    for col in features:
        raw = df[col].dropna()
        logged = signed_log(raw)
        rows.append({
            'feature': col,
            'skew_raw': round(raw.skew(), 2),
            'skew_log': round(logged.skew(), 2),
            'pct_iqr_raw': round(iqr_outliers(raw, k).mean() * 100, 2),
            'pct_iqr_log': round(iqr_outliers(logged, k).mean() * 100, 2),
            'pct_robust_z_log': round((robust_zscore(logged).abs() > z_threshold).mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def temporal_spikes(df, feature, z_threshold=10, revert_ratio=0.5, group='country', time='year'):
    """
    Flags year-over-year changes that are extreme relative to the typical change across
    the whole panel (robust z of the log change > z_threshold).

    Patterns:
      spike_and_revert - the next year moves back by at least `revert_ratio` of the jump
                         (the classic signature of a one-off event or a data-entry error)
      rebound          - the recovery year that follows a spike (same event, not a new one)
      level_shift      - the jump persists (rebasing, methodology change, or structural change)
    """
    d = df[[group, time, feature]].dropna().sort_values([group, time]).reset_index(drop=True)
    g = d.groupby(group)[feature]
    change = signed_log(d[feature]).groupby(d[group]).diff()
    next_change = change.groupby(d[group]).shift(-1)

    z = pd.Series(np.nan, index=d.index)
    valid = change.notna()
    z[valid] = robust_zscore(change[valid])
    flagged = z.abs() > z_threshold

    reverts = (np.sign(next_change) == -np.sign(change)) & (next_change.abs() >= revert_ratio * change.abs())
    is_spike = flagged & reverts
    is_rebound = flagged & is_spike.groupby(d[group]).shift(1, fill_value=False).astype(bool)

    out = d.assign(
        feature=feature,
        value=d[feature],
        previous_value=g.shift(1),
        pct_change=d[feature] / g.shift(1) - 1,
        robust_z=z.round(1),
        pattern=np.select([is_rebound, is_spike], ['rebound', 'spike_and_revert'], 'level_shift'),
    )[flagged]
    return out[[group, time, 'feature', 'value', 'previous_value', 'pct_change', 'robust_z', 'pattern']] \
        .reset_index(drop=True)
