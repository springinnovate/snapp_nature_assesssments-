"""Conservation, missing coverage, and source-preservation tests for issue #58."""
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from integrate_county_valuations_2026 import (
    DROP_FIELDS, county_values, fips, numbers, polygon_values,
    read_attributes, sha256, state_values, validate_crosswalk, write_output,
    load_checkpoint, save_checkpoint,
)


class CountyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.ids = pd.Index(["09110", "09120", "01001"], name="GEOID")
        self.crosswalk = pd.DataFrame({
            "source_geoid": ["09001", "09001", "09003"],
            "target_geoid": ["09110", "09120", "09120"],
            "weight": [0.25, 0.75, 1.0],
        })

    def test_identifier_and_currency_parsing(self):
        self.assertEqual(fips(1001), "01001")
        self.assertEqual(fips("1001.0"), "01001")
        values = numbers(pd.Series([" $1,234.50 ", "", None, "0", "-2", "$ -"]))
        self.assertEqual(values.iloc[0], 1234.5)
        self.assertTrue(values.iloc[1:3].isna().all())
        self.assertEqual(values.iloc[3], 0)
        self.assertEqual(values.iloc[5], 0)
        for invalid in ["C1018", "1001.5", "123456"]:
            with self.assertRaises(ValueError):
                fips(invalid)
        with self.assertRaises(ValueError):
            numbers(pd.Series(["#REF!"]))

    def test_county_split_conserves_totals_and_reports_unmatched(self):
        source = pd.DataFrame({"fips": [9001, 9003, 1001, 99999], "value": [100, 200, 0, 5]})
        values, status, issues, audit = county_values(source, "fips", "value", self.ids, self.crosswalk)
        self.assertEqual(values.to_list(), [25, 275, 0])
        self.assertEqual(status["09120"], "ct_land_area_estimate")
        self.assertEqual(audit["input_known_total"], 305)
        self.assertEqual(audit["unmatched_total"], 5)
        self.assertEqual(issues[0]["source_key"], "99999")

    def test_missing_old_county_does_not_look_like_complete_estimate(self):
        source = pd.DataFrame({"fips": [9001], "value": [100]})
        values, status, issues, audit = county_values(source, "fips", "value", self.ids, self.crosswalk)
        self.assertEqual(values["09110"], 25)
        self.assertTrue(np.isnan(values["09120"]))
        self.assertEqual(status["09120"], "incomplete_source")
        self.assertEqual(audit["withheld_partial_total"], 75)
        self.assertTrue(np.isnan(values["01001"]))

    def test_timber_zero_policy_and_detail_summation(self):
        source = pd.DataFrame({"fips": [9001, 9003], "value": [100, None]})
        values, status, _, _ = county_values(source, "fips", "value", self.ids, self.crosswalk, zero_missing=True)
        self.assertEqual(values.to_list(), [25, 75, 0])
        self.assertEqual(status["01001"], "missing_source_zero")
        detail = pd.DataFrame({"fips": [1001, 1001], "value": [10, 20]})
        with self.assertRaises(ValueError):
            county_values(detail, "fips", "value", self.ids, self.crosswalk)
        values, _, _, _ = county_values(detail, "fips", "value", self.ids, self.crosswalk, aggregate=True)
        self.assertEqual(values["01001"], 30)
        detail.loc[1, "value"] = np.nan
        values, _, _, audit = county_values(detail, "fips", "value", self.ids, self.crosswalk, aggregate=True)
        self.assertTrue(np.isnan(values["01001"]))
        self.assertEqual(audit["withheld_partial_total"], 10)

    def test_reject_mixed_ct_geographies_and_invalid_weights(self):
        mixed = pd.DataFrame({"fips": [9001, 9110], "value": [100, 25]})
        with self.assertRaises(ValueError):
            county_values(mixed, "fips", "value", self.ids, self.crosswalk)
        validate_crosswalk(self.crosswalk, self.ids)
        invalid = self.crosswalk.copy()
        invalid.loc[0, "weight"] = 0.5
        with self.assertRaises(ValueError):
            validate_crosswalk(invalid, self.ids)

    def test_state_allocation_excludes_total_row_and_preserves_absence(self):
        states = pd.Series(["01", "01", "01", "72"], index=["01001", "01003", "01005", "72001"])
        weights = pd.Series([1.0, 3.0, 0.0, 0.0], index=states.index)
        frame = pd.DataFrame({"State": ["Alabama", "Total", None], "value": [80, 80, None]})
        values, status, checks = state_values(frame, "value", states, weights)
        self.assertEqual(values.iloc[:3].to_list(), [20, 60, 0])
        self.assertTrue(np.isnan(values["72001"]))
        self.assertEqual(status["01005"], "zero_weight")
        self.assertEqual(checks[0]["allocated_value"], 80)
        with self.assertRaises(ValueError):
            state_values(frame, "value", states, weights * 0)

    def test_polygon_allocation_keeps_uncovered_fraction(self):
        counties = gpd.GeoDataFrame({"GEOID": ["01001", "01003"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 3, 1)], crs=6933)
        polygons = gpd.GeoDataFrame({"siteid": ["a", "b"], "value": [40.0, 10.0]}, geometry=[box(0, 0, 4, 1), box(5, 0, 6, 1)], crs=6933)
        values, audit = polygon_values(counties, polygons, "value", "siteid", chunk_size=1)
        self.assertEqual(values.to_list(), [10, 20])
        self.assertEqual(audit.unallocated_value.to_list(), [10, 10])
        overlap = gpd.GeoDataFrame({"GEOID": ["01001", "01003"]}, geometry=[box(0, 0, 4, 1)] * 2, crs=6933)
        with self.assertRaises(ValueError):
            polygon_values(overlap, polygons, "value", "siteid")

    def test_gpkg_keeps_geometry_source_and_attributes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base.gpkg"
            output = Path(temp) / "output.gpkg"
            frame = gpd.GeoDataFrame({"GEOID": ["01001", "01003"], "keep": [-12.0, 0.0],
                                     **{field: [1.0, 2.0] for field in DROP_FIELDS}},
                                    geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
            frame.to_file(base, layer="counties", driver="GPKG")
            before = sha256(base)
            additions = pd.DataFrame({"test_orig": [12.0, np.nan], "test_adj_factor": [1.5, 1.5], "test_adj": [18.0, np.nan]}, index=frame.GEOID)
            table = pd.DataFrame({"service": ["test"], "factor": [1.5]})
            write_output(base, output, "counties", "geom", additions, {"integration_services": table})
            self.assertEqual(sha256(base), before)
            attrs, _, _ = read_attributes(output)
            self.assertEqual(attrs.keep.to_list(), [-12, 0])
            self.assertFalse(set(DROP_FIELDS) & set(attrs.columns))
            with closing(sqlite3.connect(output)) as con:
                self.assertEqual(con.execute("SELECT data_type FROM gpkg_contents WHERE table_name='integration_services'").fetchone()[0], "attributes")
            self.assertTrue(gpd.read_file(output, layer="counties").geometry.equals(frame.geometry))
            with self.assertRaises(FileExistsError):
                write_output(base, output, "counties", "geom", additions, {})
            failed_output = Path(temp) / "failed.gpkg"
            invalid = additions.copy()
            invalid.index = ["99999", "01003"]
            with self.assertRaises(KeyError):
                write_output(base, failed_output, "counties", "geom", invalid, {})
            self.assertFalse(failed_output.exists())
            self.assertEqual(sha256(base), before)

    def test_service_checkpoint_round_trip_and_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "service.sqlite"
            result = {
                "originals": {"test": pd.Series([12.0, np.nan], index=["01001", "01003"])},
                "statuses": [pd.DataFrame({"GEOID": ["01001", "01003"], "service": ["test"] * 2, "status": ["source", "missing"]})],
                "metadata": [{"service": "test"}], "issues": [], "checks": [],
                "recreation_audit": pd.DataFrame(),
            }
            save_checkpoint(path, "abc", result)
            loaded = load_checkpoint(path, "abc")
            pd.testing.assert_series_equal(result["originals"]["test"], loaded["originals"]["test"], check_names=False)
            self.assertEqual(loaded["metadata"], result["metadata"])
            with self.assertRaises(ValueError):
                load_checkpoint(path, "changed-input")

    def test_recreation_recovers_completed_batches_after_failure(self):
        counties = gpd.GeoDataFrame({"GEOID": ["01001"]}, geometry=[box(0, 0, 4, 1)], crs=6933)
        polygons = gpd.GeoDataFrame({"siteid": ["a", "b", "c"], "value": [10.0, 20.0, 30.0]},
                                   geometry=[box(i, 0, i + 1, 1) for i in range(3)], crs=6933)
        original_intersection = shapely.intersection
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interrupted batch")
            return original_intersection(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            with patch("shapely.intersection", side_effect=fail_second):
                with self.assertRaises(RuntimeError):
                    polygon_values(counties, polygons, "value", "siteid", chunk_size=1, checkpoint_dir=cache)
            self.assertTrue((cache / "000000000.npz").exists())
            with patch("shapely.intersection", wraps=original_intersection) as resumed:
                values, audit = polygon_values(counties, polygons, "value", "siteid", chunk_size=1, workers=2, checkpoint_dir=cache)
                self.assertLess(resumed.call_count, 3)
            self.assertEqual(values["01001"], 60)
            self.assertEqual(audit.unallocated_value.sum(), 0)


if __name__ == "__main__":
    unittest.main()
