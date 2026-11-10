"""Check conserved population allocation of BEA combination-area GDP."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from county_gdp import allocate_combined_gdp, read_county_source


def test_reviewed_crosswalk_covers_each_virginia_member_once():
    """Keep the checked BEA groups and Census member identifiers unambiguous."""
    path = Path(__file__).resolve().parents[1] / 'data/workflow_assets/county_integration/virginia_gdp_crosswalk.csv'
    crosswalk = pd.read_csv(path, dtype=str)
    assert len(crosswalk) == crosswalk.county_geoid.nunique() == 51
    assert crosswalk.bea_geofips.nunique() == 23
    assert set(crosswalk.loc[crosswalk.bea_geofips == '51919', 'county_geoid']) == {
        '51059', '51600', '51610'}


@pytest.fixture
def gdp_inputs(tmp_path):
    """Provide three jurisdictions in one combined area plus a direct county."""
    crosswalk = pd.DataFrame({
        "bea_geofips": ["51919"] * 3,
        "county_geoid": ["51059", "51600", "51610"],
        "bea_name": ["Fairfax, Fairfax City + Falls Church"] * 3,
        "county_name": ["Fairfax County", "Fairfax city", "Falls Church city"],
        "source_url": ["https://www.bea.gov/"] * 3,
    })
    path = tmp_path / "crosswalk.csv"
    crosswalk.to_csv(path, index=False)
    gdp = pd.Series([1000., 500., np.nan], index=["51919", "51013", "01001"], name="GDP")
    population = pd.Series([60., 30., 10., 80.], index=["51059", "51600", "51610", "51013"])
    return gdp, population, crosswalk, path


def test_population_split_preserves_combined_total_and_direct_counties(gdp_inputs):
    """The combined value is distributed once with no change to direct GDP."""
    gdp, population, _, path = gdp_inputs
    expanded, allocations, checks = allocate_combined_gdp(gdp, population, path)
    assert "51919" not in expanded.index
    assert expanded.loc[["51059", "51600", "51610"]].tolist() == [600., 300., 100.]
    assert expanded.loc["51013"] == 500.
    assert pd.isna(expanded.loc["01001"])
    assert math.fsum(expanded.dropna()) == math.fsum(gdp.dropna())
    assert allocations.allocation_weight.tolist() == [.6, .3, .1]
    assert allocations.allocation_method.eq("population_2024").all()
    assert checks.member_count.tolist() == [3]
    assert checks.difference_gdp_2024.tolist() == [0.]
    assert checks.weight_sum.tolist() == [1.]


def test_populations_and_source_indexes_remain_unchanged(gdp_inputs):
    """A caller can subset outputs without changing the allocation denominator."""
    gdp, population, _, path = gdp_inputs
    original_gdp, original_population = gdp.copy(), population.copy()
    expanded, _, _ = allocate_combined_gdp(gdp, population, path)
    assert expanded.reindex(["51600"]).iloc[0] == 300.
    pd.testing.assert_series_equal(gdp, original_gdp)
    pd.testing.assert_series_equal(population, original_population)


def test_fractional_dollar_allocations_conserve_large_source_total(gdp_inputs):
    """Rounding is limited to floating-point residue, not whole-dollar shares."""
    gdp, population, _, path = gdp_inputs
    gdp.loc["51919"] = 177534419000.
    population.loc[["51059", "51600", "51610"]] = [1159806., 26038., 15046.]
    expanded, allocations, checks = allocate_combined_gdp(gdp, population, path)
    assert math.fsum(expanded.loc[["51059", "51600", "51610"]]) == gdp.loc["51919"]
    assert (allocations.allocated_gdp_2024 % 1 != 0).any()
    assert np.allclose(
        allocations.allocated_gdp_2024,
        allocations.population_2024 / allocations.group_population_2024 * gdp.loc["51919"],
        rtol=1e-15,
    )
    assert abs(checks.difference_gdp_2024.iloc[0]) < 0.0001


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf, "bad"])
def test_invalid_member_population_is_rejected(gdp_inputs, value):
    """Every constituent needs a finite, positive population."""
    gdp, population, _, path = gdp_inputs
    population = population.astype(object)
    population.loc["51610"] = value
    with pytest.raises(ValueError):
        allocate_combined_gdp(gdp, population, path)


def test_missing_member_population_is_rejected(gdp_inputs):
    """Missing population must not silently give a member zero GDP."""
    gdp, population, _, path = gdp_inputs
    with pytest.raises(ValueError, match="population"):
        allocate_combined_gdp(gdp, population.drop("51610"), path)


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_combined_gdp_is_rejected(gdp_inputs, value):
    """A combination area needs a finite, positive GDP source value."""
    gdp, population, _, path = gdp_inputs
    gdp.loc["51919"] = value
    with pytest.raises(ValueError):
        allocate_combined_gdp(gdp, population, path)


def test_missing_and_unmapped_groups_are_rejected(gdp_inputs):
    """Every present combined area is mapped and every mapped area is present."""
    gdp, population, _, path = gdp_inputs
    with pytest.raises(ValueError, match="missing"):
        allocate_combined_gdp(gdp.drop("51919"), population, path)
    with pytest.raises(ValueError, match="unmapped"):
        allocate_combined_gdp(pd.concat([gdp, pd.Series({"51901": 20.})]), population, path)


def test_direct_member_gdp_collision_is_rejected(gdp_inputs):
    """Do not overwrite direct county GDP with a portion of a combined value."""
    gdp, population, _, path = gdp_inputs
    for value in (23., np.nan):
        with pytest.raises(ValueError, match="already exist"):
            allocate_combined_gdp(pd.concat([gdp, pd.Series({"51600": value})]), population, path)


@pytest.mark.parametrize("column,value", [
    ("bea_geofips", "01001"), ("county_geoid", "5160"),
    ("county_geoid", "51919"), ("county_name", ""), ("source_url", ""),
])
def test_invalid_crosswalk_fields_are_rejected(gdp_inputs, column, value):
    """Identifiers, required names, and source provenance must be explicit."""
    gdp, population, crosswalk, path = gdp_inputs
    crosswalk.loc[0, column] = value
    crosswalk.to_csv(path, index=False)
    with pytest.raises(ValueError):
        allocate_combined_gdp(gdp, population, path)


def test_duplicate_members_and_inconsistent_group_names_are_rejected(gdp_inputs):
    """One county cannot get two allocations or inconsistent provenance."""
    gdp, population, crosswalk, path = gdp_inputs
    for broken in (
        pd.concat([crosswalk, crosswalk.iloc[[0]]], ignore_index=True),
        crosswalk.assign(county_geoid=["51059", "51059", "51610"]),
        crosswalk.assign(bea_name=["wrong", "Fairfax", "Fairfax"]),
    ):
        broken.to_csv(path, index=False)
        with pytest.raises(ValueError):
            allocate_combined_gdp(gdp, population, path)


def test_source_indexes_require_unique_normalized_keys(gdp_inputs):
    """Allocation does not reinterpret raw or duplicate source identifiers."""
    gdp, population, _, path = gdp_inputs
    for bad_gdp in (
        pd.concat([gdp, gdp.iloc[[0]]]),
        gdp.rename(index={"51919": 51919}),
        gdp.rename(index={"51919": "51919.0"}),
    ):
        with pytest.raises(ValueError, match="unique normalized"):
            allocate_combined_gdp(bad_gdp, population, path)


def test_source_reader_filters_normalizes_scales_and_preserves_missing(tmp_path):
    """Read only the requested year/line and preserve absent GDP values."""
    path = tmp_path / "gdp.csv"
    pd.DataFrame({
        "GeoFIPS": ["01000", " 1001.0 ", "01003", "01005", "footer"],
        "LineCode": ["3", "3", "3", "3", "other"],
        "2024": ["10", "1.25", "(D)", "-2", "ignored"],
    }).to_csv(path, index=False)
    spec = {"key": "GeoFIPS", "field": "2024", "where": {"LineCode": "3"},
            "missing_tokens": ["(D)"], "scale": 1000., "exclude_state_totals": True}
    values, count = read_county_source(path, spec)
    assert count == 5
    assert values.index.tolist() == ["01001", "01003", "01005"]
    assert values.loc["01001"] == 1250.
    assert pd.isna(values.loc["01003"])
    assert values.loc["01005"] == -2000.


@pytest.mark.parametrize("keys,values", [
    (["01001", "1001.0"], [1., 2.]),
    (["01001", "bad"], [1., 2.]),
    (["01001", "123456"], [1., 2.]),
    (["01001", "01003"], [1., "invalid"]),
    (["01001", "01003"], [1., np.inf]),
])
def test_source_reader_rejects_ambiguous_keys_and_invalid_numbers(tmp_path, keys, values):
    """Malformed source rows fail instead of silently losing GDP."""
    path = tmp_path / "gdp.csv"
    pd.DataFrame({"key": keys, "value": values}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_county_source(path, {"key": "key", "field": "value"})
