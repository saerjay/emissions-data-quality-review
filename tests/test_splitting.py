"""
Train / validation / test splitting (written before splitting.py existed).

Two requirements:
  - GROUPED by country: a country's years are near-duplicates of each other, so a country
    lives in exactly one split (otherwise test scores are inflated by leakage).
  - STRATIFIED by emissions size: every split gets the same mix of small and large
    emitters, so a split is not, by chance, all microstates or all G20 economies.
"""
import numpy as np
import pandas as pd
import pytest

from splitting import country_strata, split_summary, stratified_group_split

FRACTIONS = {'train': 0.6, 'validation': 0.2, 'test': 0.2}


@pytest.fixture
def countries():
    """80 countries x 5 years with emissions spanning 5 orders of magnitude, plus one with no data."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        level = 10 ** rng.uniform(-1, 4)
        rows += [{'country': f'C{i:02d}', 'year': y, 'co2': level * (1 + 0.01 * (y - 2000))}
                 for y in range(2000, 2005)]
    rows += [{'country': 'Nodata', 'year': y, 'co2': np.nan} for y in range(2000, 2005)]
    return pd.DataFrame(rows)


# --- strata -----------------------------------------------------------------
def test_strata_are_size_quartiles_of_each_countrys_median(countries):
    strata = country_strata(countries, 'co2', n_bins=4)
    assert strata.index.is_unique
    counts = strata[strata != 'no_data'].value_counts()
    assert sorted(counts.index) == ['size_q1', 'size_q2', 'size_q3', 'size_q4']
    assert counts.max() - counts.min() <= 1                       # equal-sized quartiles
    medians = countries.groupby('country')['co2'].median()
    assert medians[strata == 'size_q4'].min() > medians[strata == 'size_q1'].max()


def test_country_without_data_gets_its_own_stratum(countries):
    assert country_strata(countries, 'co2')['Nodata'] == 'no_data'


# --- split ------------------------------------------------------------------
def test_every_country_is_assigned_exactly_once(countries):
    strata = country_strata(countries, 'co2')
    split = stratified_group_split(strata, FRACTIONS, seed=1)
    assert split.index.equals(strata.index)
    assert set(split) <= set(FRACTIONS)


def test_no_country_appears_in_two_splits(countries):
    split = stratified_group_split(country_strata(countries, 'co2'), FRACTIONS, seed=1)
    rows = countries.assign(split=countries['country'].map(split))
    assert (rows.groupby('country')['split'].nunique() == 1).all()


def test_each_stratum_is_divided_in_the_requested_proportions(countries):
    strata = country_strata(countries, 'co2')
    split = stratified_group_split(strata, FRACTIONS, seed=1)
    for stratum, members in strata.groupby(strata):
        n = len(members)
        got = split[members.index].value_counts()
        for name, frac in FRACTIONS.items():
            assert abs(got.get(name, 0) - n * frac) <= 1, (stratum, name)


def test_split_is_reproducible_with_a_seed(countries):
    strata = country_strata(countries, 'co2')
    a = stratified_group_split(strata, FRACTIONS, seed=3)
    b = stratified_group_split(strata, FRACTIONS, seed=3)
    c = stratified_group_split(strata, FRACTIONS, seed=4)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(c)


def test_fractions_must_sum_to_one(countries):
    with pytest.raises(ValueError, match='sum to 1'):
        stratified_group_split(country_strata(countries, 'co2'), {'train': 0.7, 'test': 0.2})


def test_summary_counts_countries_and_rows_per_split(countries):
    strata = country_strata(countries, 'co2')
    split = stratified_group_split(strata, FRACTIONS, seed=1)
    summary = split_summary(countries, split, strata)
    assert summary['countries'].sum() == countries['country'].nunique()
    assert summary['rows'].sum() == len(countries)
    assert {'split', 'stratum', 'countries', 'rows'}.issubset(summary.columns)
