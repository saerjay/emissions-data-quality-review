"""
Stratified, grouped train / validation / test split.

Unit of splitting = the country. A country's consecutive years are near-duplicates, so if
France 2010 were in training and France 2011 in test, the test score would measure memory,
not generalisation. Every country therefore lives in exactly one split.

Stratification = emissions size. Countries are binned into quantiles of their median CO2
and each bin is divided in the requested proportions, so every split holds the same mix of
microstates and major emitters. Without it, a 40-country test split could by chance contain
none of the largest emitters and give a misleading score.
"""
import numpy as np
import pandas as pd


def country_strata(df, value_col='co2', n_bins=4, group='country'):
    """Labels each country by the quantile of its median `value_col` (size_q1 = smallest)."""
    medians = df.groupby(group)[value_col].median()
    strata = pd.Series('no_data', index=medians.index, name='stratum', dtype=object)
    known = medians.dropna()
    # Rank first so ties cannot collapse bins; quantiles of ranks are equal-sized
    bins = pd.qcut(known.rank(method='first'), q=n_bins, labels=[f'size_q{i + 1}' for i in range(n_bins)])
    strata[known.index] = bins.astype(str)
    return strata


def _allocate(n, fractions):
    """Largest-remainder rounding: integer counts per split that sum to n."""
    names = list(fractions)
    exact = np.array([n * fractions[k] for k in names])
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts))[: n - counts.sum()]:
        counts[i] += 1
    return dict(zip(names, counts))


def stratified_group_split(strata, fractions, seed=42):
    """Returns a Series country -> split name, dividing every stratum in `fractions`."""
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {sum(fractions.values()):.3f}")
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=strata.index, dtype=object, name='split')
    for _, members in strata.groupby(strata):
        countries = rng.permutation(members.index.to_numpy())
        start = 0
        for name, count in _allocate(len(countries), fractions).items():
            assignment[countries[start:start + count]] = name
            start += count
    return assignment


def split_summary(df, split, strata, group='country'):
    """Countries and rows per split and stratum, for the audit trail."""
    rows = df[group].map(split).rename('split').to_frame().assign(stratum=df[group].map(strata))
    row_counts = rows.groupby(['split', 'stratum']).size().rename('rows')
    country_counts = pd.DataFrame({'split': split, 'stratum': strata}).groupby(['split', 'stratum']).size().rename('countries')
    return pd.concat([country_counts, row_counts], axis=1).fillna(0).astype(int).reset_index()
