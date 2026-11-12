"""Verify shared harmonization and publication on small zone fixtures."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import MultiPolygon, box
import yaml

from assessment import REPO_ROOT
import assessment.assessment_valuation as valuation
from assessment.assessment_zone_geometry import ZONE_PRODUCTS


@pytest.mark.parametrize("source_name", ["county_area", "county_area_inclusive", "natural_areas"])
def test_missing_reporting_area_input_explains_preparation(monkeypatch, source_name):
    """Show the preparation command for each missing reporting-area artifact."""
    data = REPO_ROOT / "data"
    config = data / "workflow_assets/county_integration"
    rules = yaml.safe_load((config / "derived_metrics.yaml").read_text(encoding="utf-8"))
    missing = (data / rules["sources"][source_name]["path"]).resolve()
    monkeypatch.setattr(Path, "is_file", lambda path: path != missing)
    command = r"python -m assessment\.workflows\.prepare_county_reporting_areas --runner"
    with pytest.raises(FileNotFoundError, match=command):
        valuation.load_valuation_config(data, config)


@pytest.mark.parametrize(("county_area", "inclusive_area", "reuse_zones", "bbb_area"), [
    (200.0, 400.0, False, 5.0), (200.0, 400.0, True, 0.0),
    (0.0, 400.0, True, 0.0), (200.0, np.nan, True, 0.0),
])
def test_adjustment_and_recreation_are_applied_once(
    monkeypatch, tmp_path, county_area, inclusive_area, reuse_zones, bbb_area,
):
    """Check all product keys, land allocation, totals, factors and the output pair."""
    config = REPO_ROOT / "data/workflow_assets/county_integration"
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
        return pd.DataFrame({"es_total": total, "Pop_2024": 10, "area_ha_county": county_area,
                             "area_ha_county_incl_ocean_great_lakes": inclusive_area}), {}

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
            fields.update(area_ha_county_land=[100.0], area_ha_public_lands=[50.0],
                          area_ha_conservation_lands=[25.0], area_ha_bbb_lands=[bbb_area])
            if reuse_zones:
                fields.update(pct_public_lands=[50.0], pct_conservation_lands=[25.0], pct_bbb_lands=[0.0])
        else:
            fields["county_land_share"] = [0.25]
            fields["area_ha_land"] = [20.0 if product == "public_lands_by_county" else 10.0]
        if product.endswith("_and_manager_type"):
            fields = {"GEOID": ["46102", "46102"], "Mang_Type": ["FED", "LOC"],
                      "county_land_share": [0.15, 0.1], "area_ha_land": [15.0, 10.0]}
        if product == "bbb_lands_by_county" and bbb_area:
            fields["area_ha_land"] = [bbb_area]
        elif product == "bbb_lands_by_county":
            fields = {"GEOID": pd.Series(dtype="str"), "county_land_share": pd.Series(dtype="float64"),
                      "area_ha_land": pd.Series(dtype="float64")}
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
    service_ratios = derived["county_ecosystem_service_ratios"]
    assert service_ratios == {
        "ratio_public_lands_es_value_to_county_es_value": "public_lands_by_county",
        "ratio_conservation_lands_es_value_to_county_es_value": "conservation_lands_by_county",
        "ratio_bbb_lands_es_value_to_county_es_value": "bbb_lands_by_county",
    }
    per_hectare = derived["county_land_service_value_per_ha"]
    assert per_hectare == {
        "public_lands_counties_total_es_value_per_land_ha": "public_lands_by_county",
        "conservation_lands_counties_total_es_value_per_land_ha": "conservation_lands_by_county",
        "bbb_lands_counties_total_es_value_per_land_ha": "bbb_lands_by_county",
    }
    footprint_per_hectare = derived["footprint_land_service_value_per_ha"]
    assert footprint_per_hectare == {
        "total_public_es_value_per_public_land_ha": "public_lands_by_county",
        "total_conservation_es_value_per_conservation_land_ha": "conservation_lands_by_county",
        "total_bbb_es_value_per_bbb_land_ha": "bbb_lands_by_county",
    }
    footprint_fields = {product: field for field, product in footprint_per_hectare.items()}
    retired_service_fields = {f"ratio_{name}_lands_to_county_ecosystem_services"
                              for name in ("public", "conservation", "bbb")}
    retired_service_fields.update(f"{name}_lands_total_es_value_per_land_ha"
                                  for name in ("public", "conservation", "bbb"))
    expected_service_ratios = [
        outputs[product]["total_ecosystem_services"].sum() / county["total_ecosystem_services"]
        for product in service_ratios.values()
    ]
    np.testing.assert_allclose(county[list(service_ratios)].to_numpy(dtype=float), expected_service_ratios)
    expected_per_hectare = [
        outputs["public_lands_by_county"]["total_ecosystem_services"].iloc[0] / 20.0,
        outputs["conservation_lands_by_county"]["total_ecosystem_services"].iloc[0] / 10.0,
        outputs["bbb_lands_by_county"]["total_ecosystem_services"].iloc[0] / bbb_area if bbb_area else np.nan,
    ]
    expected_footprint_values = dict(zip(per_hectare.values(), expected_per_hectare, strict=True))
    np.testing.assert_allclose(county[list(per_hectare)].to_numpy(dtype=float), expected_per_hectare,
                               equal_nan=True)
    service_fields = [*service_ratios, *per_hectare]
    assert outputs["counties"].columns[-6:].tolist() == service_fields
    real_land_metrics = valuation.calculate_land_service_metrics

    def previous_land_metrics(products, ratios, per_hectare_rules, footprint_rules):
        """Retain the real pre-existing shares while omitting only per-hectare outputs."""
        metrics, _ = real_land_metrics(products, ratios, per_hectare_rules, footprint_rules)
        return metrics[list(ratios)], {}

    # All other published attributes and geometries are independent of the per-hectare fields.
    with monkeypatch.context() as original:
        original.setattr(valuation, "calculate_land_service_metrics", previous_land_metrics)
        previous, _, _ = valuation.value_zones(
            zones, raster_sums, tmp_path, config, coral_reference_counties=zones["counties"],
            workers=2, geometry_workers=1,
        )
    for product, frame in outputs.items():
        assert retired_service_fields.isdisjoint(frame.columns)
        if product == "counties":
            pd.testing.assert_frame_equal(frame.drop(columns=list(per_hectare)), previous[product])
            assert set(footprint_per_hectare).isdisjoint(frame.columns)
        elif product in footprint_fields:
            field = footprint_fields[product]
            pd.testing.assert_frame_equal(frame.drop(columns=field), previous[product])
            assert set(frame).intersection(footprint_per_hectare) == {field}
            assert set(service_fields).isdisjoint(frame.columns)
            np.testing.assert_allclose(
                frame[field], np.full(len(frame), expected_footprint_values[product]), equal_nan=True,
            )
        else:
            pd.testing.assert_frame_equal(frame, previous[product])
            assert set(service_fields).isdisjoint(frame.columns)
            assert set(footprint_per_hectare).isdisjoint(frame.columns)
    ratio_fields = [f"ratio_{name}_lands_to_county_area{suffix}"
                    for name in ("public", "conservation", "bbb")
                    for suffix in ("", "_incl_ocean_great_lakes")]
    expected_ratios = [hectares / area if np.isfinite(area) and area > 0 else np.nan
                       for hectares in (50.0, 25.0, bbb_area) for area in (county_area, inclusive_area)]
    np.testing.assert_allclose(county[ratio_fields].to_numpy(dtype=float), expected_ratios)
    ratio_start = county.index.get_loc(ratio_fields[0])
    assert county.index[ratio_start:ratio_start + len(ratio_fields)].tolist() == ratio_fields
    legacy_pct_fields = {"pct_public_lands", "pct_conservation_lands", "pct_bbb_lands"}
    assert legacy_pct_fields.isdisjoint(county.index)
    # Saved zone inputs retain their attributes; only the new exports drop them.
    assert legacy_pct_fields.intersection(zones["counties"].columns) == (legacy_pct_fields if reuse_zones else set())
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
        assert not any(name.startswith("ratio_") and name.endswith(("_to_total_county_area", "_excl_ocean_great_lakes"))
                       for name in frame.columns)
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

    for product in ZONE_PRODUCTS:
        path = tmp_path / product / f"{product}_test.gpkg"
        csv_path, gpkg = valuation.write_zone_pair(outputs[product], path, product, audits if product == "counties" else {})
        assert csv_path.is_file() and gpkg.is_file()
        published = pyogrio.read_dataframe(gpkg, layer=product)
        assert len(published) == len(outputs[product])
        for exported in (pd.read_csv(csv_path, dtype={"GEOID": str}), published):
            assert retired_service_fields.isdisjoint(exported.columns)
            if product == "counties":
                np.testing.assert_allclose(exported[list(service_ratios)].iloc[0], expected_service_ratios)
                np.testing.assert_allclose(exported[list(per_hectare)].iloc[0], expected_per_hectare,
                                           equal_nan=True)
                assert exported.drop(columns="geometry", errors="ignore").columns[-6:].tolist() == service_fields
                np.testing.assert_allclose(exported[ratio_fields].iloc[0], expected_ratios)
                assert legacy_pct_fields.isdisjoint(exported.columns)
                start = exported.columns.get_loc(ratio_fields[0])
                assert exported.columns[start:start + len(ratio_fields)].tolist() == ratio_fields
            elif product in footprint_fields:
                field = footprint_fields[product]
                assert set(exported).intersection(footprint_per_hectare) == {field}
                np.testing.assert_allclose(
                    exported[field].to_numpy(dtype=float), np.full(len(exported), expected_footprint_values[product]),
                    equal_nan=True,
                )
            else:
                assert set(footprint_per_hectare).isdisjoint(exported.columns)
        with pytest.raises(FileExistsError):
            valuation.write_zone_pair(outputs[product], path, product, {})


def test_land_service_metrics_match_geoids_and_preserve_missing_totals(tmp_path):
    """Align each category's totals and land area without recreation or managers."""
    config = REPO_ROOT / "data/workflow_assets/county_integration/derived_metrics.yaml"
    derived = yaml.safe_load(config.read_text())
    ratios = derived["county_ecosystem_service_ratios"]
    per_hectare = derived["county_land_service_value_per_ha"]
    footprint_per_hectare = derived["footprint_land_service_value_per_ha"]
    counties = gpd.GeoDataFrame({
        "GEOID": ["01003", "01001", "01005", "01007", "01009", "01011", "01013"],
        "total_ecosystem_services": [100.0, 200.0, 0.0, np.nan, -10.0, np.inf, 50.0],
        "total_ecosystem_services_w_rec": [10000.0] * 7,
    }, geometry=[MultiPolygon([box(0, 0, 1, 1)])] * 7, crs="EPSG:6933", index=[7, 3, 9, 1, 6, 8, 4])
    products = {
        "counties": counties,
        "public_lands_by_county": pd.DataFrame({
            "GEOID": ["01001", "01003", "01013", "01005", "01009", "01007", "01011"],
            "total_ecosystem_services": [50.0, 125.0, np.nan, 12.0, -6.0, 24.0, 0.0],
            "total_ecosystem_services_w_rec": [1000000.0] * 7,
            "area_ha_land": [10.0, 5.0, 2.0, 3.0, 2.0, 6.0, 5.0],
        }, index=[20, 10, 30, 40, 60, 50, 70]),
        "conservation_lands_by_county": pd.DataFrame({
            "GEOID": ["01001"], "total_ecosystem_services": [60.0],
            "total_ecosystem_services_w_rec": [2000000.0], "area_ha_land": [20.0],
        }, index=[80]),
        "bbb_lands_by_county": pd.DataFrame({
            "GEOID": pd.Series(dtype="string"), "total_ecosystem_services": pd.Series(dtype=float),
            "area_ha_land": pd.Series(dtype=float),
        }),
        "public_lands_by_county_and_manager_type": pd.DataFrame({
            "GEOID": ["01001", "01001"], "total_ecosystem_services": [1000000.0, 1000000.0],
            "area_ha_land": [0.01, 0.01],
        }),
        "conservation_lands_by_county_and_manager_type": pd.DataFrame({
            "GEOID": ["01001", "01001"], "total_ecosystem_services": [2000000.0, 2000000.0],
            "area_ha_land": [0.02, 0.02],
        }),
    }
    originals = {product: frame.copy(deep=True) for product, frame in products.items()}
    metrics, densities = valuation.calculate_land_service_metrics(products, ratios, per_hectare, footprint_per_hectare)
    expected = pd.DataFrame({
        list(ratios)[0]: [1.25, 0.25, np.nan, np.nan, np.nan, np.nan, np.nan],
        list(ratios)[1]: [0.0, 0.3, np.nan, np.nan, np.nan, np.nan, 0.0],
        list(ratios)[2]: [0.0, 0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
        list(per_hectare)[0]: [25.0, 5.0, 4.0, 4.0, -3.0, 0.0, np.nan],
        list(per_hectare)[1]: [np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        list(per_hectare)[2]: [np.nan] * 7,
    }, index=counties.index)
    pd.testing.assert_frame_equal(metrics, expected)
    expected_densities = {
        "public_lands_by_county": [5.0, 25.0, np.nan, 4.0, -3.0, 4.0, 0.0],
        "conservation_lands_by_county": [3.0],
        "bbb_lands_by_county": [],
    }
    assert set(densities) == set(expected_densities)
    for field, product in footprint_per_hectare.items():
        pd.testing.assert_series_equal(densities[product], pd.Series(
            expected_densities[product],
            index=pd.Index(products[product]["GEOID"].astype("string"), name="GEOID"),
            dtype=float,
            name=field,
        ))
    for product, original in originals.items():
        pd.testing.assert_frame_equal(products[product], original)
    # Blank and zero metrics survive both formats with leading-zero GEOIDs.
    published = counties.replace([np.inf], np.nan).join(metrics).reset_index(drop=True)
    csv_path, gpkg_path = valuation.write_zone_pair(published, tmp_path / "counties_metrics.gpkg", "counties", {})
    for readback in (pd.read_csv(csv_path, dtype={"GEOID": str}), pyogrio.read_dataframe(gpkg_path)):
        assert readback.GEOID.tolist() == counties.GEOID.tolist()
        assert readback.drop(columns="geometry", errors="ignore").columns[-6:].tolist() == list(expected.columns)
        np.testing.assert_allclose(readback[list(expected.columns)], expected, equal_nan=True)


@pytest.mark.parametrize("category", ["public_lands_by_county", "conservation_lands_by_county", "bbb_lands_by_county"])
@pytest.mark.parametrize(("presence", "numerator", "area", "county_total", "expected"), [
    pytest.param("present", 50.0, 10.0, 100.0, 5.0, id="positive-land-area"),
    pytest.param("present", 0.0, 10.0, 100.0, 0.0, id="zero-category-total"),
    pytest.param("present", -50.0, 10.0, 100.0, -5.0, id="negative-category-total"),
    pytest.param("present", 50.0, 10.0, 0.0, 5.0, id="zero-county-total"),
    pytest.param("present", 50.0, 10.0, np.nan, 5.0, id="missing-county-total"),
    pytest.param("present", 50.0, 0.0, 100.0, np.nan, id="zero-land-area"),
    pytest.param("present", 50.0, -10.0, 100.0, np.nan, id="negative-land-area"),
    pytest.param("present", 50.0, np.nan, 100.0, np.nan, id="missing-land-area"),
    pytest.param("present", 50.0, np.inf, 100.0, np.nan, id="infinite-land-area"),
    pytest.param("present", 50.0, -np.inf, 100.0, np.nan, id="negative-infinite-land-area"),
    pytest.param("present", np.nan, 10.0, 100.0, np.nan, id="missing-category-total"),
    pytest.param("present", np.inf, 10.0, 100.0, np.nan, id="infinite-category-total"),
    pytest.param("present", -np.inf, 10.0, 100.0, np.nan, id="negative-infinite-category-total"),
    pytest.param("absent", 50.0, 10.0, 100.0, np.nan, id="absent-category-row"),
    pytest.param("empty", 50.0, 10.0, 100.0, np.nan, id="empty-category-product"),
])
def test_land_service_value_per_hectare_handles_category_edge_cases(
    category, presence, numerator, area, county_total, expected,
):
    """Use each category's finite total and positive land area independently."""
    config = REPO_ROOT / "data/workflow_assets/county_integration/derived_metrics.yaml"
    derived = yaml.safe_load(config.read_text())
    ratios = derived["county_ecosystem_service_ratios"]
    per_hectare = derived["county_land_service_value_per_ha"]
    footprint_per_hectare = derived["footprint_land_service_value_per_ha"]
    products = {"counties": pd.DataFrame({
        "GEOID": ["01001", "01003"], "total_ecosystem_services": [county_total, 100.0],
    }, index=[9, 4])}
    for product in per_hectare.values():
        products[product] = pd.DataFrame({
            "GEOID": ["01003", "01001"], "total_ecosystem_services": [10.0, numerator],
            "area_ha_land": [2.0, area], "total_ecosystem_services_w_rec": [1000000.0] * 2,
        }, index=[30, 10])
    if presence == "absent":
        products[category] = products[category].iloc[:1].copy()
    elif presence == "empty":
        products[category] = products[category].iloc[:0].copy()

    metrics, densities = valuation.calculate_land_service_metrics(products, ratios, per_hectare, footprint_per_hectare)

    assert metrics.index.tolist() == [9, 4]
    assert metrics.columns.tolist() == [*ratios, *per_hectare]
    footprint_fields = {product: field for field, product in footprint_per_hectare.items()}
    for field, product in per_hectare.items():
        expected_values = [expected, np.nan if presence == "empty" and product == category else 5.0]
        if presence in {"absent", "empty"} and product != category:
            expected_values[0] = numerator / area
        np.testing.assert_allclose(metrics[field], expected_values, equal_nan=True)
        density = densities[product]
        assert density.name == footprint_fields[product]
        assert density.index.tolist() == products[product]["GEOID"].tolist()
        np.testing.assert_allclose(density, list(reversed(expected_values))[:len(density)], equal_nan=True)
    if presence in {"absent", "empty"}:
        ratio_field = next(field for field, product in ratios.items() if product == category)
        assert metrics.loc[9, ratio_field] == 0.0
    elif not np.isfinite(county_total) or county_total <= 0:
        assert metrics.loc[9, list(ratios)].isna().all()


@pytest.mark.parametrize("problem", [
    "duplicate_county", "duplicate_category", "unknown_county", "missing_geoid", "malformed_geoid",
    "manager_ratio_product", "manager_per_hectare_product", "manager_footprint_product",
    "ratio_field_collision", "per_hectare_field_collision", "footprint_field_collision",
    "cross_mapping_field_collision",
])
def test_land_service_metrics_reject_ambiguous_inputs(problem):
    """Reject malformed joins and manager products instead of guessing or overwriting."""
    config = REPO_ROOT / "data/workflow_assets/county_integration/derived_metrics.yaml"
    derived = yaml.safe_load(config.read_text())
    ratios = derived["county_ecosystem_service_ratios"]
    per_hectare = derived["county_land_service_value_per_ha"]
    footprint_per_hectare = derived["footprint_land_service_value_per_ha"]
    products = {name: pd.DataFrame({
        "GEOID": ["01001"], "total_ecosystem_services": [10.0], "area_ha_land": [2.0],
    })
                for name in ("counties", *ratios.values())}
    category = "public_lands_by_county"
    if problem.startswith("duplicate"):
        name = "counties" if problem == "duplicate_county" else category
        products[name] = pd.concat([products[name], products[name]], ignore_index=True)
    elif problem == "unknown_county":
        products[category]["GEOID"] = ["01003"]
    elif problem in {"missing_geoid", "malformed_geoid"}:
        products[category]["GEOID"] = [None if problem == "missing_geoid" else "1001"]
    elif problem == "manager_ratio_product":
        ratios[next(iter(ratios))] = "public_lands_by_county_and_manager_type"
    elif problem == "manager_per_hectare_product":
        per_hectare[next(iter(per_hectare))] = "public_lands_by_county_and_manager_type"
    elif problem == "manager_footprint_product":
        footprint_per_hectare[next(iter(footprint_per_hectare))] = "public_lands_by_county_and_manager_type"
    elif problem == "ratio_field_collision":
        products["counties"][next(iter(ratios))] = 0.0
    elif problem == "per_hectare_field_collision":
        products["counties"][next(iter(per_hectare))] = 0.0
    elif problem == "footprint_field_collision":
        products[category][next(field for field, product in footprint_per_hectare.items() if product == category)] = 0.0
    else:
        per_hectare[next(iter(ratios))] = per_hectare.pop(next(iter(per_hectare)))
    with pytest.raises(ValueError):
        valuation.calculate_land_service_metrics(products, ratios, per_hectare, footprint_per_hectare)
