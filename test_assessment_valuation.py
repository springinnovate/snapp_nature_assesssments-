"""Verify shared harmonization and publication on small zone fixtures."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import MultiPolygon, box
import yaml

import assessment_valuation as valuation
from assessment_zone_geometry import ZONE_PRODUCTS


@pytest.mark.parametrize("source_name", ["county_area", "natural_areas"])
def test_missing_reporting_area_input_explains_preparation(monkeypatch, source_name):
    """Show the required preparation command for either reporting-area artifact."""
    data = Path(valuation.__file__).parent / "data"
    config = data / "workflow_assets/county_integration"
    rules = yaml.safe_load((config / "derived_metrics.yaml").read_text(encoding="utf-8"))
    missing = (data / rules["sources"][source_name]["path"]).resolve()
    monkeypatch.setattr(Path, "is_file", lambda path: path != missing)
    with pytest.raises(FileNotFoundError, match="prepare_county_reporting_areas.py.*--runner"):
        valuation.load_valuation_config(data, config)


@pytest.mark.parametrize("county_total_area", [200.0, 0.0, np.nan])
def test_adjustment_and_recreation_are_applied_once(monkeypatch, tmp_path, county_total_area):
    """Check all product keys, land allocation, totals, factors and the output pair."""
    config = Path(valuation.__file__).parent / "data/workflow_assets/county_integration"
    factors = pd.read_csv(config / "adjustment_factors.csv").set_index("service")
    rules = valuation.load_export_rules(config / "export_rules.yaml")
    derived = yaml.safe_load((config / "derived_metrics.yaml").read_text(encoding="utf-8"))
    air_path = tmp_path / "air_quality.csv"
    pd.DataFrame({"FIPS": ["46113"], "land": ["forest"], "pollutant": ["PM2.5"],
                  "dollars_annual": [20.0]}).to_csv(air_path, index=False)
    paths = {"air_quality": air_path, "air_quality_county_aliases": config / "air_quality_county_aliases.csv"}
    real_compute_task = valuation.compute_task
    monkeypatch.setattr(valuation, "load_valuation_config", lambda *_: (paths, factors, rules, derived))
    monkeypatch.setattr(valuation, "validate_crosswalk", lambda *_: None)
    task_values = {
        "grazing": {"grazing": 40},
        "timber": {"timber": 0},
        "air_quality": {"avoided_air_quality_heath_costs": 20},
        "urban_heat": {"urban_heat": 5},
        "physical_health": {"physical_health": 10},
        "jobs": {"jobs_cultural": 1, "jobs_harvesting": 2, "jobs_raw_materials": 3,
                 "jobs_processing": 4, "jobs": 10},
        "recreation": {"recreation": 400},
    }

    def county_task(task, paths, base, crosswalk, workers):
        """Simulate raw whole-county values from the tested service-specific joins."""
        if task == "air_quality":
            return real_compute_task(task, paths, base, crosswalk, workers)
        values = {name: pd.Series(number, index=base.index) for name, number in task_values[task].items()}
        return {
            "originals": values,
            "metadata": [{"source": task}],
            "statuses": [pd.DataFrame({"GEOID": base.index, "status": "observed"})],
            "summary_fields": pd.DataFrame({"sum_dollars_annual": 20}, index=base.index),
        }

    def vector_values(columns, zones):
        """Provide raw spatial measurements independently from land shares."""
        return {product: pd.DataFrame(columns, index=frame.index) for product, frame in zones.items()}

    def county_metrics(frame, data_root, config_path):
        """Compute the existing county total for the reconciliation assertion."""
        total = frame[derived["es_total"]["components"]].fillna(0).sum(axis=1)
        return pd.DataFrame({"es_total": total, "Pop_2024": 10,
                             "area_ha_county_excl_ocean_great_lakes": county_total_area}), {}

    monkeypatch.setattr(valuation, "compute_task", county_task)
    monkeypatch.setattr(valuation, "value_fisheries", lambda zones, *_: vector_values(
        {"marine_fisheries": 4, "inland_fisheries": 8}, zones))
    monkeypatch.setattr(valuation, "value_coastal", lambda zones, *_, **__: vector_values(
        {"corals": 5, "mangroves": np.nan}, zones))
    monkeypatch.setattr(valuation, "calculate_county_metrics", county_metrics)

    zones, raster_sums = {}, {}
    for product in ZONE_PRODUCTS:
        fields = {"GEOID": ["46102"], "NAME": ["NA"]}
        if product == "counties":
            fields.update(area_ha_county_land=[100.0], area_ha_public_lands=[50.0], pct_public_lands=[50.0],
                          area_ha_conservation_lands=[25.0], pct_conservation_lands=[25.0],
                          area_ha_bbb_lands=[0.0], pct_bbb_lands=[0.0])
        else:
            fields["county_land_share"] = [0.25]
        if product.endswith("_and_manager_type"):
            fields = {"GEOID": ["46102", "46102"], "Mang_Type": ["FED", "LOC"],
                      "county_land_share": [0.15, 0.1]}
        if product == "bbb_lands_by_county":
            fields = {"GEOID": pd.Series(dtype="str"), "county_land_share": pd.Series(dtype="float64")}
        size = len(fields["GEOID"])
        zones[product] = gpd.GeoDataFrame(fields, geometry=[MultiPolygon([box(0, 0, 100, 100)])] * size,
                                           crs="EPSG:6933")
        keys = ["GEOID", "Mang_Type"] if "Mang_Type" in fields else ["GEOID"]
        sums = zones[product][keys].copy()
        for source in valuation.BASE_FIELDS.values():
            sums[source] = [100 * (index + 1) for index in range(size)]
        if product == "counties":
            sums["sum_totalC_tCO2e_ha_2020"] = [27.0]
            sums["sum_totalC_tCO2e_pixel_2020"] = [2.0]  # Retired input must not affect dollars.
        raster_sums[product] = sums.iloc[::-1].reset_index(drop=True)

    outputs, audits, qa = valuation.value_zones(
        zones, raster_sums, tmp_path, config, coral_reference_counties=zones["counties"],
        workers=2, geometry_workers=1,
    )
    assert set(outputs) == set(ZONE_PRODUCTS)
    county = outputs["counties"].iloc[0]
    ratio_fields = [f"ratio_{name}_lands_to_county_area_excl_ocean_great_lakes"
                    for name in ("public", "conservation", "bbb")]
    expected_ratios = [0.25, 0.125, 0.0] if county_total_area > 0 else [np.nan] * 3
    np.testing.assert_allclose(county[ratio_fields].to_numpy(dtype=float), expected_ratios)
    legacy_pct_fields = {"pct_public_lands", "pct_conservation_lands", "pct_bbb_lands"}
    assert legacy_pct_fields.isdisjoint(county.index)
    # Saved zone inputs retain their attributes; only the new exports drop them.
    assert legacy_pct_fields.issubset(zones["counties"].columns)
    assert county["recreation_adj"] == pytest.approx(400)
    assert county["air_quality_avoided_mortality_annual_value_adj"] == pytest.approx(20 * 1.135)
    assert county["total_ecosystem_services_w_rec"] - county["total_ecosystem_services"] == pytest.approx(400)
    assert county["sum_pollination_attributed_annual_crop_yield_value_adj"] == pytest.approx(9)
    assert county["pollination_adj_factor"] == pytest.approx(0.09)
    carbon = "sum_CO2e_mitigation_dollar_value_per_pixel_adj"
    assert county[carbon] == pytest.approx(2.43 * 4.32612)
    assert county["carbon_pixel_adj_factor"] == pytest.approx(4.32612)
    assert "sum_totalC_tCO2e_pixel_2020" not in county
    assert carbon in qa["total_components"]
    assert county["sum_flood_marginal_npv_masked_to_wetlands_adj"] == pytest.approx(9)
    assert "sum_flood_marginal_npv_masked_to_wetlands_adj" not in qa["total_components"]
    assert "es_total" not in county
    assert pd.isna(county["sum_mangrove_coastal_flood_protection_value_adj"])
    for product, frame in outputs.items():
        retained = zones[product].drop(columns=list(legacy_pct_fields), errors="ignore")
        pd.testing.assert_frame_equal(frame[retained.columns], retained)
        assert legacy_pct_fields.isdisjoint(frame.columns)
        assert not any(name.endswith("_to_total_county_area") for name in frame.columns)
        assert {"sum_totalC_tCO2e_ha_2020", "sum_totalC_tCO2e_ha_2020_adj",
                "carbon_ha_adj_factor"}.isdisjoint(frame.columns)
        assert "sum_pollination_attributed_annual_crop_yield_value" not in frame
        if product == "counties":
            continue
        assert set(ratio_fields).isdisjoint(frame.columns)
        assert {"recreation_adj", "recreation_adj_factor", "total_ecosystem_services_w_rec"}.isdisjoint(frame)
        assert np.allclose(frame["grazing_adj"], 40 * frame["county_land_share"])
        assert np.allclose(frame["air_quality_avoided_mortality_annual_value_adj"],
                           20 * 1.135 * frame["county_land_share"])
        assert np.allclose(frame["marine_fisheries_adj"], 4 * 1.027)
        if "Mang_Type" in frame:
            assert list(frame["sum_pollination_attributed_annual_crop_yield_value_adj"]) == pytest.approx([9, 18])
            assert list(frame[carbon]) == pytest.approx([9 * 4.32612, 18 * 4.32612])
    assert "integration_county_table_status" in audits
    assert qa["blank_component_rows"]["counties"]["sum_mangrove_coastal_flood_protection_value_adj"] == 1

    for product in ("counties", "public_lands_by_county_and_manager_type", "bbb_lands_by_county"):
        path = tmp_path / product / f"{product}_test.gpkg"
        csv_path, gpkg = valuation.write_zone_pair(outputs[product], path, product, audits if product == "counties" else {})
        assert csv_path.is_file() and gpkg.is_file()
        published = pyogrio.read_dataframe(gpkg, layer=product)
        assert len(published) == len(outputs[product])
        if product == "counties":
            for exported in (pd.read_csv(csv_path), published):
                np.testing.assert_allclose(exported[ratio_fields].iloc[0], expected_ratios)
                assert legacy_pct_fields.isdisjoint(exported.columns)
        with pytest.raises(FileExistsError):
            valuation.write_zone_pair(outputs[product], path, product, {})
