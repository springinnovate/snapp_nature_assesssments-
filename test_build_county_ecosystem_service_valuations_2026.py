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
from shapely.geometry import Polygon, box

from build_county_ecosystem_service_valuations_2026 import (
    DROP_FIELDS, adjusted_county_fields, air_quality_summary, county_values, fips, numbers, polygon_values,
    read_attributes, sha256, state_values, validate_crosswalk, write_output,
    load_checkpoint, run, save_checkpoint,
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

    def test_air_quality_summaries_keep_raw_units_and_missing_counties(self):
        """Statistics describe category values, not pixels or adjusted dollars."""
        source = pd.DataFrame({"FIPS": [1001, 1001, 1001, 99999],
                               "land": ["forest", "grass", "shrub", "forest"],
                               "pollutant": ["PM2.5"] * 4,
                               "dollars_annual": [10.0, 20.0, 30.0, 5.0]})
        totals, _, _, audit = county_values(source, "FIPS", "dollars_annual", self.ids,
                                          self.crosswalk, aggregate=True)
        summary = air_quality_summary(source, self.ids, self.crosswalk, totals)
        row = summary.loc["01001"]
        self.assertEqual(row.sum_dollars_annual, 60)
        self.assertEqual(row.mean_dollars_annual, 20)
        self.assertEqual(row.median_dollars_annual, 20)
        self.assertEqual(row.min_dollars_annual, 10)
        self.assertEqual(row.max_dollars_annual, 30)
        self.assertAlmostEqual(row.stdev_dollars_annual, np.std([10, 20, 30], ddof=0))
        self.assertEqual(row.valid_count_dollars_annual, 3)
        self.assertEqual(row.total_count_dollars_annual, 3)
        self.assertTrue(np.isnan(summary.loc["09110", "sum_dollars_annual"]))
        self.assertEqual(summary.loc["09110", "total_count_dollars_annual"], 0)
        self.assertEqual(audit["unmatched_total"], 5)

    def test_air_quality_ct_recombines_categories_before_statistics(self):
        """CT means count target categories once, not once per old-county piece."""
        source = pd.DataFrame({"FIPS": [9001, 9001, 9003, 9003],
                               "land": ["forest", "grass"] * 2,
                               "pollutant": ["PM2.5"] * 4,
                               "dollars_annual": [10.0, 30.0, 40.0, 80.0]})
        totals = county_values(source, "FIPS", "dollars_annual", self.ids,
                               self.crosswalk, aggregate=True)[0]
        summary = air_quality_summary(source, self.ids, self.crosswalk, totals)
        self.assertEqual(summary.loc["09110", "sum_dollars_annual"], 10)
        self.assertEqual(summary.loc["09110", "mean_dollars_annual"], 5)
        self.assertEqual(summary.loc["09120", "sum_dollars_annual"], 150)
        self.assertEqual(summary.loc["09120", "mean_dollars_annual"], 75)
        self.assertEqual(summary.loc["09120", "stdev_dollars_annual"], 27.5)
        self.assertEqual(summary.loc["09120", "valid_count_dollars_annual"], 2)
        partial = source.iloc[:2]
        totals = county_values(partial, "FIPS", "dollars_annual", self.ids,
                               self.crosswalk, aggregate=True)[0]
        summary = air_quality_summary(partial, self.ids, self.crosswalk, totals)
        self.assertTrue(np.isnan(summary.loc["09120", "mean_dollars_annual"]))
        self.assertEqual(summary.loc["09120", "valid_count_dollars_annual"], 0)
        self.assertEqual(summary.loc["09120", "total_count_dollars_annual"], 2)

    def test_air_quality_incomplete_values_and_normalized_duplicate_keys(self):
        """Missing contributions cannot become complete statistics or zeros."""
        source = pd.DataFrame({"FIPS": [1001, "01001"], "land": ["forest", "grass"],
                               "pollutant": ["PM2.5"] * 2, "dollars_annual": [10.0, None]})
        totals = county_values(source, "FIPS", "dollars_annual", self.ids,
                               self.crosswalk, aggregate=True)[0]
        summary = air_quality_summary(source, self.ids, self.crosswalk, totals)
        self.assertTrue(np.isnan(summary.loc["01001", "sum_dollars_annual"]))
        self.assertTrue(np.isnan(summary.loc["01001", "mean_dollars_annual"]))
        self.assertEqual(summary.loc["01001", "valid_count_dollars_annual"], 1)
        self.assertEqual(summary.loc["01001", "total_count_dollars_annual"], 2)
        source.loc[1, "land"] = "forest"
        with self.assertRaisesRegex(ValueError, "Duplicate air-quality"):
            air_quality_summary(source, self.ids, self.crosswalk, totals)

    def test_adjusted_fields_use_statistic_names_without_raw_or_duplicate_values(self):
        """Value statistics get _adj; counts/areas stay raw; factors stay visible."""
        stem = "pollination_attributed_annual_crop_yield_value"
        base = pd.DataFrame({f"{stat}_{stem}": [100.0, np.nan]
                             for stat in ("sum", "mean", "median", "min", "max", "stdev", "std", "p50")},
                            index=self.ids[:2])
        for stat in ("valid_count", "total_count", "area_valid", "proportion_valid"):
            base[f"{stat}_{stem}"] = [20.0, 30.0]
        summaries = pd.DataFrame({f"{stat}_dollars_annual": [10.0, np.nan]
                                  for stat in ("sum", "mean", "median", "min", "max", "stdev")},
                                 index=base.index)
        summaries["valid_count_dollars_annual"] = [9, 0]
        summaries["total_count_dollars_annual"] = [9, 0]
        originals = {"pollination": base[f"sum_{stem}"],
                     "avoided_air_quality_heath_costs": summaries.sum_dollars_annual.copy(),
                     "timber": pd.Series([20.0, -10.0], index=base.index)}
        factors = pd.DataFrame({"factor": [0.09, 1.135, 1.0]}, index=list(originals))
        fields, mapping = adjusted_county_fields(base, originals, summaries, factors)
        for stat in ("sum", "mean", "median", "min", "max", "stdev", "std", "p50"):
            np.testing.assert_allclose(fields[f"{stat}_{stem}_adj"], [9, np.nan])
            self.assertNotIn(f"{stat}_{stem}", fields)
        for stat in ("valid_count", "total_count", "area_valid", "proportion_valid"):
            self.assertNotIn(f"{stat}_{stem}_adj", fields)
        for stat in ("sum", "mean", "median", "min", "max", "stdev"):
            np.testing.assert_allclose(fields[f"{stat}_dollars_annual_adj"], [11.35, np.nan])
            self.assertNotIn(f"{stat}_dollars_annual", fields)
        self.assertEqual(fields.valid_count_dollars_annual.to_list(), [9, 0])
        self.assertEqual(fields.total_count_dollars_annual.to_list(), [9, 0])
        self.assertEqual(fields.pollination_adj_factor.to_list(), [0.09, 0.09])
        self.assertEqual(fields.avoided_air_quality_heath_costs_adj_factor.to_list(), [1.135, 1.135])
        self.assertEqual(fields.timber_adj.to_list(), [20, -10])
        self.assertEqual(mapping["pollination"], f"sum_{stem}_adj")
        self.assertEqual(mapping["avoided_air_quality_heath_costs"], "sum_dollars_annual_adj")
        self.assertFalse(any(field.endswith("_orig") for field in fields))
        self.assertNotIn("pollination_adj", fields)
        self.assertNotIn("avoided_air_quality_heath_costs_adj", fields)
        summaries.loc[base.index[0], "sum_dollars_annual"] = 99
        with self.assertRaisesRegex(ValueError, "canonical sum"):
            adjusted_county_fields(base, originals, summaries, factors)

    def test_builder_rejects_scaled_pollination_instead_of_applying_point09_twice(self):
        """An already-per-pixel VRT aggregate must not silently be scaled again."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.mkdir()
            gpd.GeoDataFrame(
                {"GEOID": ["01001"], "STATEFP": ["01"],
                 "sum_pollination_attributed_annual_crop_yield_value_usd_per_pixel": [9.0]},
                geometry=[box(0, 0, 1, 1)], crs=4326,
            ).to_file(root / "base.gpkg", layer="counties", driver="GPKG")
            (config / "sources.csv").write_text("source,path\nbase,base.gpkg\n", encoding="utf-8")
            (config / "adjustment_factors.csv").write_text(
                "service,factor,reason,source_units\npollination,0.09,conversion,USD\n", encoding="utf-8")
            self.crosswalk.to_csv(config / "ct_county_crosswalk.csv", index=False)
            with self.assertRaisesRegex(ValueError, "must not receive.*second time"):
                run(root, config, workers=1, geometry_workers=1)
            with closing(sqlite3.connect(root / "base.gpkg")) as con:
                con.execute("CREATE TABLE integration_run (created_utc TEXT)")
                con.commit()
            with self.assertRaisesRegex(ValueError, "already an integrated"):
                run(root, config, workers=1, geometry_workers=1)

    def test_polygon_allocation_keeps_uncovered_fraction(self):
        counties = gpd.GeoDataFrame({"GEOID": ["01001", "01003"]}, geometry=[box(0, 0, 1, 1), box(1, 0, 3, 1)], crs=6933)
        polygons = gpd.GeoDataFrame({"siteid": ["a", "b"], "value": [40.0, 10.0]}, geometry=[box(0, 0, 4, 1), box(5, 0, 6, 1)], crs=6933)
        values, audit = polygon_values(counties, polygons, "value", "siteid", chunk_size=1)
        self.assertEqual(values.to_list(), [10, 20])
        self.assertEqual(audit.unallocated_value.to_list(), [10, 10])
        overlap = gpd.GeoDataFrame({"GEOID": ["01001", "01003"]}, geometry=[box(0, 0, 4, 1)] * 2, crs=6933)
        with self.assertRaises(ValueError):
            polygon_values(overlap, polygons, "value", "siteid")

    def test_hexagon_crossing_county_boundary_distributes_value_by_area(self):
        """A symmetric boundary-crossing hexagon assigns half its value per county."""
        counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003"]},
            geometry=[box(-2, -2, 0, 2), box(0, -2, 2, 2)], crs=6933)
        hexagon = Polygon([(-1, 0), (-0.5, 1), (0.5, 1), (1, 0), (0.5, -1), (-0.5, -1)])
        polygons = gpd.GeoDataFrame({"siteid": ["hex"], "value": [120.0]},
                                   geometry=[hexagon], crs=6933)
        values, audit = polygon_values(counties, polygons, "value", "siteid", workers=2)
        np.testing.assert_allclose(values, [60, 60])
        self.assertEqual(audit.allocated_fraction.iloc[0], 1)
        self.assertEqual(audit.unallocated_value.iloc[0], 0)

    def test_small_coverage_excess_is_capped_with_auditable_correction(self):
        """Tiny excess preserves county shares and uncovered source value."""
        counties = gpd.GeoDataFrame(
            {"GEOID": ["01001", "01003"]},
            geometry=[box(0, 0, 1.000004, 1), box(1, 0, 2, 1)], crs=6933)
        polygons = gpd.GeoDataFrame(
            {"siteid": ["covered", "partial"], "value": [100.0, 80.0]},
            geometry=[box(0, 0, 2, 1), box(0, 2, 4, 3)], crs=6933)
        with tempfile.TemporaryDirectory() as temp:
            values, audit = polygon_values(counties, polygons, "value", "siteid",
                                            chunk_size=1, workers=2, checkpoint_dir=Path(temp))
            self.assertAlmostEqual(values.sum(), 100)
            self.assertAlmostEqual(values.iloc[0] / values.iloc[1], 1.000004)
            self.assertAlmostEqual(audit.raw_allocated_fraction.iloc[0], 1.000002)
            self.assertEqual(audit.allocated_fraction.iloc[0], 1)
            self.assertAlmostEqual(audit.coverage_correction_value.iloc[0], 0.0002)
            self.assertEqual(audit.unallocated_value.to_list(), [0, 80])
            with patch("shapely.intersection", side_effect=AssertionError("should resume")):
                resumed_values, resumed_audit = polygon_values(
                    counties, polygons, "value", "siteid", chunk_size=1, checkpoint_dir=Path(temp))
            pd.testing.assert_series_equal(values, resumed_values)
            pd.testing.assert_frame_equal(audit, resumed_audit)
        counties.loc[0, "geometry"] = box(0, 0, 1.001, 1)
        with self.assertRaisesRegex(ValueError, "covered.*1.000500000"):
            polygon_values(counties, polygons, "value", "siteid")

    def test_gpkg_keeps_geometry_source_and_attributes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base.gpkg"
            output = Path(temp) / "output.gpkg"
            frame = gpd.GeoDataFrame({"GEOID": ["01001", "01003"], "keep": [-12.0, 0.0],
                                     "sum_pollination_attributed_annual_crop_yield_value": [100.0, 200.0],
                                     **{field: [1.0, 2.0] for field in DROP_FIELDS}},
                                    geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
            frame.to_file(base, layer="counties", driver="GPKG")
            before = sha256(base)
            additions = pd.DataFrame({"test_adj_factor": [1.5, 1.5], "test_adj": [18.0, np.nan]}, index=frame.GEOID)
            additions["pollination_adj_factor"] = 0.09
            raw_field = "sum_pollination_attributed_annual_crop_yield_value"
            additions[raw_field + "_adj"] = [9.0, 18.0]
            additions["sum_dollars_annual_adj"] = [12.0, np.nan]
            additions["mean_dollars_annual_adj"] = [4.0, np.nan]
            renames = {raw_field: raw_field + "_adj"}
            table = pd.DataFrame({"service": ["test"], "factor": [1.5]})
            write_output(base, output, "counties", "geom", additions, {"integration_services": table}, renamed_fields=renames)
            self.assertEqual(sha256(base), before)
            attrs, _, _ = read_attributes(output)
            self.assertEqual(attrs.keep.to_list(), [-12, 0])
            self.assertFalse(set(DROP_FIELDS) & set(attrs.columns))
            self.assertNotIn(raw_field, attrs)
            self.assertNotIn("pollination_adj", attrs)
            self.assertFalse(any(field.endswith("_orig") for field in attrs))
            self.assertEqual(attrs[raw_field + "_adj"].to_list(), [9, 18])
            self.assertEqual(attrs.pollination_adj_factor.to_list(), [0.09, 0.09])
            self.assertEqual(read_attributes(base)[0][raw_field].to_list(), [100, 200])
            csv = pd.read_csv(output.with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip")
            self.assertNotIn("geometry", csv.columns)
            pd.testing.assert_frame_equal(attrs, csv, check_dtype=False)
            self.assertTrue(np.isnan(csv.mean_dollars_annual_adj.iloc[1]))
            with closing(sqlite3.connect(output)) as con:
                self.assertEqual(con.execute("SELECT data_type FROM gpkg_contents WHERE table_name='integration_services'").fetchone()[0], "attributes")
            self.assertTrue(gpd.read_file(output, layer="counties").geometry.equals(frame.geometry))
            with self.assertRaises(FileExistsError):
                write_output(base, output, "counties", "geom", additions, {})
            failed_output = Path(temp) / "failed.gpkg"
            invalid = additions.copy()
            invalid.index = ["99999", "01003"]
            with self.assertRaises(KeyError):
                write_output(base, failed_output, "counties", "geom", invalid, {}, renamed_fields=renames)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_output.with_suffix(".csv").exists())
            self.assertEqual(sha256(base), before)
            with self.assertRaisesRegex(ValueError, "raw-to-_adj"):
                write_output(base, failed_output, "counties", "geom", additions, {},
                             renamed_fields={"GEOID": "GEOID_adj"})
            overlap = additions.copy()
            overlap["keep"] = 1.0
            with self.assertRaisesRegex(ValueError, "raw-to-_adj"):
                write_output(base, failed_output, "counties", "geom", overlap, {})
            occupied = Path(temp) / "occupied.gpkg"
            occupied.with_suffix(".csv").touch()
            with self.assertRaises(FileExistsError):
                write_output(base, occupied, "counties", "geom", additions, {})
            self.assertFalse(occupied.exists())
            original_link = Path.hardlink_to

            def fail_csv_publish(destination, target):
                """Simulate failure publishing the second member of the pair."""
                if destination.suffix == ".csv":
                    raise PermissionError("simulated CSV publication failure")
                return original_link(destination, target)

            with patch.object(Path, "hardlink_to", new=fail_csv_publish):
                with self.assertRaises(PermissionError):
                    write_output(base, failed_output, "counties", "geom", additions, {}, renamed_fields=renames)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_output.with_suffix(".csv").exists())

    def test_service_checkpoint_round_trip_and_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "service.sqlite"
            result = {
                "originals": {"test": pd.Series([12.0, np.nan], index=["01001", "01003"])},
                "statuses": [pd.DataFrame({"GEOID": ["01001", "01003"], "service": ["test"] * 2, "status": ["source", "missing"]})],
                "metadata": [{"service": "test"}], "issues": [], "checks": [],
                "recreation_audit": pd.DataFrame(),
                "summary_fields": pd.DataFrame({"sum_dollars_annual": [12.0, np.nan]},
                                               index=pd.Index(["01001", "01003"], name="GEOID")),
            }
            save_checkpoint(path, "abc", result)
            loaded = load_checkpoint(path, "abc")
            pd.testing.assert_series_equal(result["originals"]["test"], loaded["originals"]["test"], check_names=False)
            self.assertEqual(loaded["metadata"], result["metadata"])
            pd.testing.assert_frame_equal(result["summary_fields"], loaded["summary_fields"])
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
