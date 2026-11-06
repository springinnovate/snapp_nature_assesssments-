"""Regression tests for whole-point coral allocation and county exports."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from coral_county_allocation import allocate_coral_values
from build_county_ecosystem_service_valuations_2026 import (
    BASE_FIELDS, CONFIG, adjusted_county_fields, compute_task, load_checkpoint,
    load_export_rules, plan_county_export, read_attributes, save_checkpoint,
    sha256, write_output,
)


CORAL_OUTPUT = "sum_aeb_coral_BC_USD11_adj"


class CoralCountyAllocationTests(unittest.TestCase):
    """Test nearest polygon assignment, conservation, and reproducible outputs."""

    def setUp(self):
        """Create three counties and contained, offshore, and zero-valued points."""
        self.counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003", "01005"], "STATEFP": ["01"] * 3,
             "ALAND": [100, 200, 300]},
            geometry=[box(-81, 25, -80, 26), box(-79, 25, -78, 26),
                      box(-77, 25, -76, 26)], crs=4326,
        )
        self.points = gpd.GeoDataFrame(
            {"OBJECTID": [206, 207, 208, 209], "BC_USD11": [100.0, 20.0, 50.0, 0.0]},
            geometry=[Point(-80.5, 25.5), Point(-79.9, 25.5),
                      Point(-78.5, 25.5), Point(-78.25, 25.5)], crs=4326,
        )

    def test_whole_values_sum_once_with_zero_for_unassigned_counties(self):
        """Containment and offshore points conserve every value without rasterization."""
        original_counties = self.counties.copy()
        original_points = self.points.copy()
        totals, statuses, audit = allocate_coral_values(self.counties, self.points)
        np.testing.assert_allclose(totals, [120, 50, 0])
        self.assertEqual(totals.sum(), self.points.BC_USD11.sum())
        self.assertEqual(statuses.tolist(), ["nearest_county", "nearest_county", "no_assigned_points"])
        self.assertEqual(audit.source_key.tolist(), ["206", "207", "208", "209"])
        self.assertEqual(audit.GEOID.tolist(), ["01001", "01001", "01003", "01003"])
        self.assertGreater(audit.distance_m.iloc[1], 0)
        self.assertEqual(audit.distance_m.iloc[0], 0)
        pd.testing.assert_frame_equal(self.counties, original_counties)
        pd.testing.assert_frame_equal(self.points, original_points)

    def test_nearest_boundary_not_nearest_centroid(self):
        """An elongated polygon wins over a nearer centroid in another county."""
        counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003"]},
            geometry=[box(-90, 25, -80, 26), box(-79, 25, -78.9, 26)], crs=4326,
        )
        points = self.points.iloc[[1]]
        totals, _, audit = allocate_coral_values(counties, points)
        self.assertEqual(audit.GEOID.tolist(), ["01001"])
        self.assertEqual(totals.tolist(), [20, 0])

    def test_metric_distance_is_not_unprojected_degrees(self):
        """At high latitude, one longitude degree is nearer than 0.6 latitude."""
        counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003"]},
            geometry=[box(1, 59.9, 1.1, 60.1), box(-0.1, 60.6, 0.1, 60.7)], crs=4326,
        )
        points = gpd.GeoDataFrame({"OBJECTID": [1], "BC_USD11": [10]},
                                 geometry=[Point(0, 60)], crs=4326)
        _, _, audit = allocate_coral_values(counties, points)
        self.assertEqual(audit.GEOID.tolist(), ["01001"])
        self.assertTrue(55000 < audit.distance_m.iloc[0] < 56000)

    def test_ties_are_deterministic_and_never_duplicate_values(self):
        """Shared boundaries and equidistant counties select the lowest GEOID."""
        for bounds in ([(-1, -1, 0, 1), (0, -1, 1, 1)],
                       [(-2, -1, -1, 1), (1, -1, 2, 1)]):
            with self.subTest(bounds=bounds):
                counties = gpd.GeoDataFrame({"GEOID": ["01003", "01001"]},
                                           geometry=[box(*b) for b in bounds], crs=4326)
                points = gpd.GeoDataFrame({"OBJECTID": [1], "BC_USD11": [10]},
                                         geometry=[Point(0, 0)], crs=4326)
                totals, _, audit = allocate_coral_values(counties, points)
                self.assertEqual(totals.tolist(), [0, 10])
                self.assertEqual(audit.GEOID.iloc[0], "01001")
                self.assertEqual(audit.tie_count.iloc[0], 2)
                self.assertEqual(audit.tied_counties.iloc[0], "01001;01003")
                shuffled, _, _ = allocate_coral_values(counties.iloc[::-1], points)
                pd.testing.assert_series_equal(totals, shuffled.reindex(totals.index))

    def test_missing_value_propagates_only_to_its_assigned_county(self):
        """Unknown contributions are not silently replaced by zero or partial sums."""
        self.points.loc[1, "BC_USD11"] = np.nan
        totals, status, audit = allocate_coral_values(self.counties, self.points)
        self.assertTrue(np.isnan(totals.loc["01001"]))
        self.assertEqual(totals.loc["01003"], 50)
        self.assertEqual(status.loc["01001"], "incomplete_source")
        self.assertEqual(audit.input_value.isna().sum(), 1)

    def test_declared_crs_is_used_and_inputs_may_have_different_crs(self):
        """Projected inputs yield the same assignments as their WGS84 originals."""
        totals, _, audit = allocate_coral_values(self.counties.to_crs(5070), self.points.to_crs(32617))
        np.testing.assert_allclose(totals, [120, 50, 0])
        self.assertEqual(audit.GEOID.tolist(), ["01001", "01001", "01003", "01003"])

    def test_bad_identifiers_values_and_geometry_fail_clearly(self):
        """Malformed source rows cannot silently vanish from the assigned totals."""
        for column, value in [("OBJECTID", 206), ("OBJECTID", None),
                              ("BC_USD11", np.inf), ("BC_USD11", "not-a-number"),
                              ("geometry", None), ("geometry", Point()),
                              ("geometry", box(0, 0, 1, 1))]:
            with self.subTest(column=column, value=value):
                points = self.points.copy()
                if column == "BC_USD11":
                    points[column] = points[column].astype(object)
                points.loc[1, column] = value
                with self.assertRaises(ValueError):
                    allocate_coral_values(self.counties, points)
        for column, value in [("GEOID", "01001"), ("GEOID", None),
                              ("GEOID", "123"), ("geometry", None)]:
            with self.subTest(column=column, value=value):
                counties = self.counties.copy()
                counties.loc[1, column] = value
                with self.assertRaises(ValueError):
                    allocate_coral_values(counties, self.points)
        points = self.points.copy()
        points.crs = None
        with self.assertRaisesRegex(ValueError, "declare a CRS"):
            allocate_coral_values(self.counties, points)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            allocate_coral_values(self.counties, self.points.iloc[:0])

    def test_tracked_config_replaces_legacy_fields_even_when_base_has_them(self):
        """New and historical base schemas both expose only the adjusted point sum."""
        self.assertNotIn("corals", BASE_FIELDS)
        sources = pd.read_csv(CONFIG / "sources.csv").set_index("source")
        self.assertEqual(sources.loc["corals", "path"], "analysis_inputs/ecosystem_services/aeb_coral.gpkg")
        factors = pd.read_csv(CONFIG / "adjustment_factors.csv").set_index("service")
        self.assertEqual(factors.loc["corals", "factor"], 1.368)
        rules = load_export_rules(CONFIG / "export_rules.yaml")
        for legacy in ([], ["sum_Coral_Reefs_2024adj_CPI", "sum_Coral_Reefs_2024adj_CPI_adj",
                            "mean_Coral_Reefs_2024adj_CPI"]):
            with self.subTest(legacy=legacy):
                columns = ["GEOID", "ALAND", *BASE_FIELDS.values(), *rules["exclude_fields"], *legacy]
                plan = plan_county_export(columns, factors.index, rules)
                kept = [row["output_field"] for row in plan if row["action"] != "exclude"]
                self.assertIn(CORAL_OUTPUT, kept)
                self.assertIn("corals_adj_factor", kept)
                self.assertFalse(any("Coral_Reefs" in field for field in kept))

    def test_checkpoint_and_paired_exports_preserve_point_audit_and_factor(self):
        """The real ingestion/writer path applies 1.368 once and preserves geometry."""
        self.counties["sum_Coral_Reefs_2024adj_CPI"] = [9000, 8000, 7000]
        self.counties["sum_Coral_Reefs_2024adj_CPI_adj"] = [9000, 8000, 7000]
        rules = load_export_rules(CONFIG / "export_rules.yaml")
        rules["exclude_fields"] = []
        rules["rename_fields"] = {"corals_adj": rules["rename_fields"]["corals_adj"]}
        factors = pd.read_csv(CONFIG / "adjustment_factors.csv").set_index("service").loc[["corals"]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"base": root / "counties.gpkg", "corals": root / "aeb_coral.gpkg"}
            self.counties.to_file(paths["base"], layer="counties", driver="GPKG")
            self.points.to_file(paths["corals"], layer="aeb_coral", driver="GPKG")
            hashes = {key: sha256(path) for key, path in paths.items()}
            attributes, layer, geom = read_attributes(paths["base"])
            base = attributes.set_index("GEOID")
            result = compute_task("corals", paths, base, pd.DataFrame(), 1, None)
            checkpoint = root / "corals.sqlite"
            save_checkpoint(checkpoint, "coral-fixture", result)
            restored = load_checkpoint(checkpoint, "coral-fixture")
            pd.testing.assert_frame_equal(restored["coral_audit"], result["coral_audit"], check_dtype=False)
            np.testing.assert_allclose(restored["originals"]["corals"], [120, 50, 0])
            self.assertEqual(restored["metadata"], result["metadata"])
            meta = result["metadata"][0]
            self.assertEqual(meta["input_known_total"], 170)
            self.assertEqual(meta["original_total"], 170)
            self.assertEqual(meta["source_point_count"], 4)
            self.assertEqual(meta["assigned_point_count"], 4)
            plan = plan_county_export(attributes.columns, factors.index, rules, geom)
            fields, mapping = adjusted_county_fields(base, restored["originals"], pd.DataFrame(), factors, plan)
            self.assertEqual(mapping["corals"], CORAL_OUTPUT)
            np.testing.assert_allclose(fields[CORAL_OUTPUT], [164.16, 68.4, 0])
            self.assertEqual(fields.corals_adj_factor.tolist(), [1.368] * 3)
            self.assertEqual(list(fields), [CORAL_OUTPUT, "corals_adj_factor"])
            tables = {"integration_coral_allocation": restored["coral_audit"],
                      "integration_services": pd.DataFrame(restored["metadata"])}
            removed = [row["source_field"] for row in plan
                       if row["source_field"] and row["group"] != "base_attributes"]
            output = root / "combined.gpkg"
            write_output(paths["base"], output, layer, geom, fields, tables, removed_fields=removed)
            saved = read_attributes(output)[0]
            csv = pd.read_csv(output.with_suffix(".csv"), dtype={"GEOID": str, "STATEFP": str},
                              float_precision="round_trip")
            pd.testing.assert_frame_equal(saved, csv, check_dtype=False)
            self.assertEqual(list(saved), ["fid", "GEOID", "STATEFP", "ALAND", CORAL_OUTPUT, "corals_adj_factor"])
            self.assertTrue(gpd.read_file(output, layer=layer).geometry.equals(self.counties.geometry))
            self.assertEqual({key: sha256(path) for key, path in paths.items()}, hashes)
            with closing(sqlite3.connect(output)) as connection:
                audit = pd.read_sql_query("SELECT * FROM integration_coral_allocation", connection)
            pd.testing.assert_frame_equal(audit.drop(columns="fid"), result["coral_audit"], check_dtype=False)


if __name__ == "__main__":
    unittest.main()
