"""Regression tests for study-unit mangroves and nonnegative timber values."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from build_county_ecosystem_service_valuations_2026 import (
    BASE_FIELDS,
    CONFIG,
    adjusted_county_fields,
    compute_task,
    load_checkpoint,
    load_export_rules,
    plan_county_export,
    read_attributes,
    save_checkpoint,
    sha256,
    write_output,
)


MANGROVE_OUTPUT = "sum_mangrove_coastal_flood_protection_value_adj"


class MangroveTimberIntegrationTests(unittest.TestCase):
    """Exercise source ingestion, audit persistence, and published field policy."""

    def setUp(self):
        """Create an empty historical-county crosswalk for direct joins."""
        self.crosswalk = pd.DataFrame({
            "source_geoid": pd.Series(dtype=str),
            "target_geoid": pd.Series(dtype=str),
            "weight": pd.Series(dtype=float),
        })

    def test_timber_clamps_source_values_and_preserves_missing_policy(self):
        """Negative direct and unmatched values become zero with a raw audit."""
        ids = pd.Index(["01001", "01003", "01005", "01007", "01009"], name="GEOID")
        base = pd.DataFrame({"STATEFP": "01"}, index=ids)
        source = pd.DataFrame({
            "fips_padded": ["01001", "01003", "01005", "01007", "99999", "99997"],
            "total_annual_rent": ["$-25", "$120", "0", None, "5", "-8"],
            "coverage_status": ["valued"] * 6,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timber.csv"
            source.to_csv(path, index=False)
            original_hash = sha256(path)
            result = compute_task("timber", {"timber": path}, base, self.crosswalk, 1, None)
            self.assertEqual(sha256(path), original_hash)

        self.assertEqual(result["originals"]["timber"].to_list(), [0, 120, 0, 0, 0])
        status = pd.concat(result["statuses"]).set_index("GEOID").status
        self.assertEqual(status["01001"], "negative_source_clamped_zero")
        self.assertEqual(status["01003"], "source")
        self.assertEqual(status["01007"], "missing_value_zero")
        self.assertEqual(status["01009"], "missing_source_zero")
        metadata = result["metadata"][0]
        self.assertEqual(metadata["raw_input_known_total"], 92)
        self.assertEqual(metadata["input_known_total"], 125)
        self.assertEqual(metadata["negative_source_count"], 2)
        self.assertEqual(metadata["negative_source_total"], -33)
        self.assertEqual(metadata["clamp_increase_total"], 33)
        self.assertEqual(metadata["unmatched_total"], 5)
        self.assertEqual(metadata["negative_counties"], 0)
        clamped = {row["source_key"]: row["value"] for row in result["issues"]
                   if row["reason"] == "negative_source_clamped_zero"}
        self.assertEqual(clamped, {"01001": -25, "99997": -8})
        self.assertEqual(metadata["original_total"] + metadata["unmatched_total"],
                         metadata["input_known_total"])

    def test_timber_clamps_before_connecticut_crosswalk_aggregation(self):
        """A negative old county cannot cancel a positive county contribution."""
        ids = pd.Index(["09110", "09120", "01001"], name="GEOID")
        base = pd.DataFrame({"STATEFP": ["09", "09", "01"]}, index=ids)
        crosswalk = pd.DataFrame({
            "source_geoid": ["09001", "09001", "09003"],
            "target_geoid": ["09110", "09120", "09120"],
            "weight": [0.25, 0.75, 1.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timber.csv"
            pd.DataFrame({
                "fips_padded": ["09001", "09003", "01001"],
                "total_annual_rent": [-100, 200, 50],
                "coverage_status": ["valued"] * 3,
            }).to_csv(path, index=False)
            result = compute_task("timber", {"timber": path}, base, crosswalk, 1, None)

        self.assertEqual(result["originals"]["timber"].to_list(), [0, 200, 50])
        metadata = result["metadata"][0]
        self.assertEqual(metadata["raw_input_known_total"], 150)
        self.assertEqual(metadata["input_known_total"], 250)
        self.assertEqual(metadata["original_total"], 250)
        self.assertEqual(metadata["negative_source_total"], -100)
        self.assertEqual(metadata["clamp_increase_total"], 100)

    def test_mangrove_task_reads_actual_fids_and_preserves_unknown_values(self):
        """GeoPackage IDs survive allocation; null benefits do not become zeros."""
        county_ids = ["01001", "01003", "01005", "01007"]
        counties = gpd.GeoDataFrame(
            {"GEOID": county_ids, "STATEFP": ["01"] * 4},
            geometry=[box(0, 0, 3, 1), box(3, 0, 4, 1),
                      box(20, 0, 21, 1), box(30, 0, 31, 1)],
            crs=6933,
        )
        sources = gpd.GeoDataFrame(
            {"fid": [7, 19, 41, 53, 61], "ISO3": ["USA"] * 5,
             "Country": ["United States"] * 5,
             "Ben_Stock_2020": [1000.0, 200.0, np.nan, 40.0, 80.0]},
            geometry=[box(0, 0, 10, 1), box(0, 0, 1, 1),
                      box(20, 0, 21, 1), box(20, 0, 21, 1), box(40, 0, 41, 1)],
            crs=6933,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = {"base": Path(directory) / "counties.gpkg",
                     "mangroves": Path(directory) / "mangroves.gpkg"}
            counties.to_file(paths["base"], layer="counties", driver="GPKG")
            sources.to_file(paths["mangroves"], layer="UCSC_CWON_studyunits", driver="GPKG")
            hashes = {key: sha256(path) for key, path in paths.items()}
            base = read_attributes(paths["base"])[0].set_index("GEOID")
            result = compute_task("mangroves", paths, base, self.crosswalk, 2, None)
            self.assertEqual({key: sha256(path) for key, path in paths.items()}, hashes)
            np.testing.assert_allclose(result["originals"]["mangroves"],
                                       [950, 250, np.nan, 0], equal_nan=True)
            self.assertEqual(set(result["mangrove_audit"].source_key), {"7", "19", "41", "53", "61"})
            self.assertEqual(len(result["mangrove_audit"]), 5)
            audit = result["mangrove_audit"].set_index("source_key")
            self.assertEqual(audit.loc["7", "intersecting_counties"], 2)
            self.assertEqual(audit.loc["7", "allocated_value"], 1000)
            self.assertEqual(audit.loc["19", "allocated_value"], 200)
            self.assertEqual(audit.loc["61", "unallocated_value"], 80)
            self.assertTrue(pd.isna(audit.loc["41", "input_value"]))
            status = pd.concat(result["statuses"]).set_index("GEOID").status
            self.assertEqual(status["01005"], "incomplete_source")
            self.assertEqual(status["01007"], "no_intersection")
            metadata = result["metadata"][0]
            self.assertEqual(metadata["source_field"], "Ben_Stock_2020")
            self.assertEqual(metadata["input_known_total"], 1320)
            self.assertEqual(metadata["allocated_known_total"], 1240)
            self.assertEqual(metadata["unmatched_total"], 80)
            self.assertEqual(metadata["withheld_partial_total"], 40)
            self.assertEqual(result["originals"]["mangroves"].sum(), 1200)
            self.assertEqual(metadata["missing_source_values"], 1)
            self.assertEqual(metadata["unintersected_source_units"], 1)

            checkpoint = Path(directory) / "mangroves.sqlite"
            save_checkpoint(checkpoint, "mangrove-fixture", result)
            restored = load_checkpoint(checkpoint, "mangrove-fixture")
            pd.testing.assert_series_equal(
                restored["originals"]["mangroves"], result["originals"]["mangroves"],
                check_names=False,
            )
            pd.testing.assert_frame_equal(restored["mangrove_audit"], result["mangrove_audit"],
                                          check_dtype=False)
            pd.testing.assert_frame_equal(pd.concat(restored["statuses"], ignore_index=True),
                                          pd.concat(result["statuses"], ignore_index=True))
            self.assertEqual(restored["metadata"], result["metadata"])
            self.assertEqual(restored["issues"], result["issues"])

    def test_configuration_selects_study_units_and_excludes_classification_sums(self):
        """Reviewed source and export policies identify the replacement valuation."""
        self.assertNotIn("mangroves", BASE_FIELDS)
        sources = pd.read_csv(CONFIG / "sources.csv").set_index("source")
        self.assertTrue(str(sources.loc["mangroves", "path"]).endswith(
            "mangroves/ucsc_cwon_studyunits.gpkg"))
        factors = pd.read_csv(CONFIG / "adjustment_factors.csv").set_index("service")
        self.assertEqual(factors.loc["mangroves", "factor"], 1.182)
        rules = load_export_rules(CONFIG / "export_rules.yaml")
        self.assertEqual(rules["rename_fields"]["mangroves_adj"], MANGROVE_OUTPUT)
        self.assertTrue({"sum_mangrove_CONUS", "sum_mangrove_CONUS_adj"}
                        <= set(rules["optional_exclude_fields"]))

    def test_mangrove_country_filter_uses_iso3_before_allocation(self):
        """Foreign and unknown-country polygons cannot add or invalidate totals."""
        counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003"], "STATEFP": ["01", "01"]},
            geometry=[box(0, 0, 1, 1), box(10, 0, 11, 1)], crs=6933,
        )
        sources = gpd.GeoDataFrame(
            {"fid": [10, 20, 30, 40, 50, 60],
             "ISO3": ["USA", "PRI", "VIR", "MEX", "MEX", None],
             # Deliberately inconsistent labels ensure that ISO3 alone decides.
             "Country": ["Mexico", "United States", "Puerto Rico",
                         "United States", "United States", "United States"],
             "Ben_Stock_2020": [100.0, 200.0, 300.0, 900.0, np.nan, 800.0]},
            geometry=[box(0, 0, 2, 1)] * 6, crs=6933,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = {"base": Path(directory) / "counties.gpkg",
                     "mangroves": Path(directory) / "mangroves.gpkg"}
            counties.to_file(paths["base"], layer="counties", driver="GPKG")
            sources.to_file(paths["mangroves"], layer="UCSC_CWON_studyunits", driver="GPKG")
            original_hash = sha256(paths["mangroves"])
            base = read_attributes(paths["base"])[0].set_index("GEOID")
            result = compute_task("mangroves", paths, base, self.crosswalk, 1, None)
            self.assertEqual(sha256(paths["mangroves"]), original_hash)

        self.assertEqual(result["originals"]["mangroves"].to_list(), [600, 0])
        self.assertEqual(set(result["mangrove_audit"].source_key), {"10", "20", "30"})
        self.assertEqual(set(result["mangrove_audit"].ISO3), {"USA", "PRI", "VIR"})
        self.assertEqual(result["issues"], [])
        metadata = result["metadata"][0]
        self.assertEqual(metadata["source_country_filter"], "USA,PRI,VIR")
        self.assertEqual(metadata["source_unit_count"], 3)
        self.assertEqual(metadata["excluded_country_source_units"], 3)
        self.assertEqual(metadata["input_known_total"], 600)
        self.assertEqual(metadata["allocated_known_total"], 600)
        self.assertEqual(metadata["original_total"], 600)
        self.assertEqual(metadata["unmatched_total"], 0)
        self.assertEqual(metadata["withheld_partial_total"], 0)
        self.assertEqual(metadata["missing_source_values"], 0)

    def test_paired_exports_apply_factor_once_and_keep_allocation_audit(self):
        """Published county files replace categorical sums with adjusted benefits."""
        counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003"], "STATEFP": ["01", "01"],
             "ALAND": [3, 1], "sum_mangrove_CONUS": [-128.0, -256.0],
             "sum_mangrove_CONUS_adj": [-151.296, -302.592]},
            geometry=[box(0, 0, 3, 1), box(3, 0, 4, 1)], crs=6933,
        )
        sources = gpd.GeoDataFrame(
            {"fid": [17], "ISO3": ["USA"], "Country": ["United States"],
             "Ben_Stock_2020": [1000.0]}, geometry=[box(0, 0, 10, 1)], crs=6933,
        )
        factors = pd.read_csv(CONFIG / "adjustment_factors.csv").set_index("service").loc[
            ["timber", "mangroves"]]
        rules = load_export_rules(CONFIG / "export_rules.yaml")
        rules["exclude_fields"] = []
        rules["rename_fields"] = {"mangroves_adj": rules["rename_fields"]["mangroves_adj"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"base": root / "counties.gpkg", "mangroves": root / "mangroves.gpkg",
                     "timber": root / "timber.csv"}
            counties.to_file(paths["base"], layer="counties", driver="GPKG")
            sources.to_file(paths["mangroves"], layer="UCSC_CWON_studyunits", driver="GPKG")
            pd.DataFrame({"fips_padded": ["01001", "01003"],
                          "total_annual_rent": [-2.0, 10.0],
                          "coverage_status": ["valued", "valued"]}).to_csv(paths["timber"], index=False)
            hashes = {key: sha256(path) for key, path in paths.items()}
            attributes, layer, geom = read_attributes(paths["base"])
            base = attributes.set_index("GEOID")
            results = {task: compute_task(task, paths, base, self.crosswalk, 1, None)
                       for task in ("mangroves", "timber")}
            originals = {service: series for result in results.values()
                         for service, series in result["originals"].items()}
            plan = plan_county_export(attributes.columns, factors.index, rules, geom)
            fields, mapping = adjusted_county_fields(
                base, originals, pd.DataFrame(index=base.index), factors, plan)
            self.assertEqual(mapping["mangroves"], MANGROVE_OUTPUT)
            np.testing.assert_allclose(fields[MANGROVE_OUTPUT], [886.5, 295.5])
            self.assertEqual(fields.mangroves_adj_factor.to_list(), [1.182, 1.182])
            self.assertEqual(fields.timber_adj.to_list(), [0, 10])
            self.assertLess(list(fields).index(MANGROVE_OUTPUT), list(fields).index("timber_adj"))
            self.assertNotIn("mangroves_adj", fields)
            self.assertFalse(any("mangrove_CONUS" in field for field in fields))
            tables = {
                "integration_mangrove_allocation": results["mangroves"]["mangrove_audit"],
                "integration_services": pd.DataFrame([
                    metadata for result in results.values() for metadata in result["metadata"]]),
                "integration_source_issues": pd.DataFrame(results["timber"]["issues"]),
                "integration_export_fields": pd.DataFrame(plan),
            }
            removed = [row["source_field"] for row in plan
                       if row["source_field"] and row["group"] != "base_attributes"]
            output = root / "snapp_ecosystem_services_counties_test.gpkg"
            write_output(paths["base"], output, layer, geom, fields, tables, removed_fields=removed)
            saved = read_attributes(output)[0]
            csv = pd.read_csv(output.with_suffix(".csv"), dtype={"GEOID": str, "STATEFP": str},
                              float_precision="round_trip")
            pd.testing.assert_frame_equal(saved, csv, check_dtype=False)
            self.assertEqual(list(saved), ["fid", "GEOID", "STATEFP", "ALAND"] + list(fields))
            self.assertFalse(any("mangrove_CONUS" in field for field in saved))
            self.assertTrue(gpd.read_file(output, layer=layer).geometry.equals(counties.geometry))
            self.assertEqual({key: sha256(path) for key, path in paths.items()}, hashes)
            with closing(sqlite3.connect(output)) as connection:
                audit = pd.read_sql_query("SELECT * FROM integration_mangrove_allocation", connection)
                pd.testing.assert_frame_equal(audit.drop(columns="fid"),
                                              results["mangroves"]["mangrove_audit"], check_dtype=False)
                timber = pd.read_sql_query(
                    "SELECT * FROM integration_services WHERE service = 'timber'", connection).iloc[0]
                self.assertEqual(timber.negative_source_total, -2)
                self.assertEqual(timber.clamp_increase_total, 2)


if __name__ == "__main__":
    unittest.main()
