"""Conservation, missing coverage, and source-preservation tests for issue #58."""
from pathlib import Path
from contextlib import closing
from copy import deepcopy
import sqlite3
import json
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import box
from osgeo import ogr

from build_county_ecosystem_service_valuations_2026 import (
    BASE_FIELDS, NON_SUM_FIELD, adjusted_county_fields, load_export_rules, plan_county_export,
    air_quality_summary, compute_task, county_values, fips, numbers,
    read_attributes, sha256, state_values, validate_crosswalk, write_output,
    load_checkpoint, run, save_checkpoint,
)

RULES_PATH = Path(__file__).parent / "data/workflow_assets/county_integration/export_rules.yaml"
AIR_QUALITY_OUTPUT = "air_quality_avoided_mortality_annual_value"
LEGACY_CROP_FIELD = "sum_national_attributed_annual_crop_yield_value_zstd"


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

    def test_grazing_uses_subtotal_before_eight_month_multiplier(self):
        """Use the annual forage value once, including for split counties."""
        base = pd.DataFrame(index=self.ids)
        source = pd.DataFrame({
            "CountyCode": [9001, 9003, 1001],
            "CountyGrazingValueMonth": [80, 160, 24],
            "CountyGrazingValueYear": [640, 1280, 192],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grazing.xlsx"
            source.to_excel(path, sheet_name="SNAPPGrazingCalculations", index=False, engine="openpyxl")
            original_hash = sha256(path)
            result = compute_task("grazing", {"grazing": path}, base, self.crosswalk, 1)
            self.assertEqual(sha256(path), original_hash)

        self.assertEqual(result["originals"]["grazing"].to_list(), [20, 220, 24])
        metadata = result["metadata"][0]
        self.assertEqual(metadata["source_field"], "CountyGrazingValueMonth")
        self.assertIn("eight-month multiplier excluded", metadata["method"])
        self.assertEqual(metadata["original_total"], 264)

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

    def test_air_quality_alias_restores_value_before_both_aggregations(self):
        """Restore renamed counties while preserving source files and CT weights."""
        ids = pd.Index(["46102", "46003", "09110", "09120", "01001"], name="GEOID")
        source = pd.DataFrame({
            "FIPS": ["46113", "46113", "46003", "09001", "09003"],
            "land": ["forest", "grass", "forest", "forest", "forest"],
            "pollutant": ["PM2.5"] * 5, "dollars_annual": [10., 20., 5., 8., 12.],
        })
        alias_path = RULES_PATH.with_name("air_quality_county_aliases.csv")
        aliases = pd.read_csv(alias_path, dtype=str)
        expected = [30., 5., 2., 18., np.nan]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "air.csv"
            source.to_csv(path, index=False)
            digest = sha256(path)
            paths = {"air_quality": path, "air_quality_county_aliases": alias_path}
            result = compute_task("air_quality", paths, pd.DataFrame(index=ids), self.crosswalk, 1)
            self.assertEqual(sha256(path), digest)
            source.assign(FIPS=source.FIPS.replace({"46113": "46102"})).to_csv(path, index=False)
            repeated = compute_task("air_quality", paths, pd.DataFrame(index=ids), self.crosswalk, 1)
            np.testing.assert_allclose(repeated["originals"]["avoided_air_quality_heath_costs"], expected)
            self.assertEqual(json.loads(repeated["metadata"][0]["county_code_aliases"]), [])
            duplicate = pd.concat([source, source.iloc[[0]].assign(FIPS="46102")], ignore_index=True)
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate air-quality"):
                compute_task("air_quality", paths, pd.DataFrame(index=ids), self.crosswalk, 1)
        np.testing.assert_allclose(result["originals"]["avoided_air_quality_heath_costs"], expected)
        np.testing.assert_allclose(result["summary_fields"].sum_dollars_annual, expected)
        self.assertEqual(result["metadata"][0]["unmatched_total"], 0.)
        self.assertEqual(json.loads(result["metadata"][0]["county_code_aliases"]), [
            dict(aliases.iloc[0], source_rows=2, raw_known_total=30.)])

    def test_air_quality_rejects_duplicate_alias_sources(self):
        """Require one correction per source county instead of choosing silently."""
        source = pd.DataFrame(columns=["FIPS", "land", "pollutant", "dollars_annual"])
        aliases = pd.read_csv(RULES_PATH.with_name("air_quality_county_aliases.csv"), dtype=str)
        with tempfile.TemporaryDirectory() as directory:
            paths = {"air_quality": Path(directory) / "air.csv",
                     "air_quality_county_aliases": Path(directory) / "aliases.csv"}
            source.to_csv(paths["air_quality"], index=False)
            pd.concat([aliases, aliases.assign(target_geoid="46115")]).to_csv(
                paths["air_quality_county_aliases"], index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate source GEOIDs"):
                compute_task("air_quality", paths, pd.DataFrame(index=self.ids), self.crosswalk, 1)

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
        """Export only sums/totals and factors, with descriptive air-quality names."""
        stem = "pollination_attributed_annual_crop_yield_value"
        base = pd.DataFrame({f"{stat}_{stem}": [100.0, np.nan]
                             for stat in ("sum", "mean", "median", "min", "max", "stdev", "std", "p50")},
                            index=self.ids[:2])
        for stat in ("valid_count", "total_count", "area_valid", "proportion_valid"):
            base[f"{stat}_{stem}"] = [20.0, 30.0]
        base["sum_unconfigured_quantity"] = [7.0, np.nan]
        summaries = pd.DataFrame({f"{stat}_dollars_annual": [10.0, np.nan]
                                  for stat in ("sum", "mean", "median", "min", "max", "stdev")},
                                 index=base.index)
        summaries["valid_count_dollars_annual"] = [9, 0]
        summaries["total_count_dollars_annual"] = [9, 0]
        originals = {"pollination": base[f"sum_{stem}"],
                     "avoided_air_quality_heath_costs": summaries.sum_dollars_annual.copy(),
                     "timber": pd.Series([20.0, 0.0], index=base.index)}
        factors = pd.DataFrame({"factor": [0.09, 1.135, 1.0]}, index=list(originals))
        rules = load_export_rules(RULES_PATH)
        rules["exclude_fields"] = []
        rules["rename_fields"] = {key: value for key, value in rules["rename_fields"].items()
                                  if key.startswith("avoided_air_quality_")}
        plan = plan_county_export(base.columns, factors.index, rules)
        fields, mapping = adjusted_county_fields(base, originals, summaries, factors, plan)
        self.assertEqual(list(fields), [
            f"sum_{stem}_adj", "sum_unconfigured_quantity", AIR_QUALITY_OUTPUT + "_adj",
            "timber_adj", "pollination_adj_factor", AIR_QUALITY_OUTPUT + "_adj_factor", "timber_adj_factor",
        ])
        np.testing.assert_allclose(fields[f"sum_{stem}_adj"], [9, np.nan])
        np.testing.assert_allclose(fields.sum_unconfigured_quantity, [7, np.nan])
        np.testing.assert_allclose(fields[AIR_QUALITY_OUTPUT + "_adj"], [11.35, np.nan])
        self.assertEqual(fields.pollination_adj_factor.to_list(), [0.09, 0.09])
        self.assertEqual(fields[AIR_QUALITY_OUTPUT + "_adj_factor"].to_list(), [1.135, 1.135])
        self.assertEqual(fields.timber_adj.to_list(), [20, 0])
        self.assertEqual(mapping["pollination"], f"sum_{stem}_adj")
        self.assertEqual(mapping["avoided_air_quality_heath_costs"], AIR_QUALITY_OUTPUT + "_adj")
        self.assertFalse(any(field.endswith("_orig") for field in fields))
        self.assertNotIn("pollination_adj", fields)
        self.assertNotIn("avoided_air_quality_heath_costs_adj", fields)
        summaries.loc[base.index[0], "sum_dollars_annual"] = 99
        with self.assertRaisesRegex(ValueError, "canonical sum"):
            adjusted_county_fields(base, originals, summaries, factors, plan)

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
            (config / "export_rules.yaml").write_bytes(RULES_PATH.read_bytes())
            derived = yaml.safe_load(RULES_PATH.with_name("derived_metrics.yaml").read_text(encoding="utf-8"))
            derived["sources"]["gdp"].pop("allocation_crosswalk", None)
            (config / "derived_metrics.yaml").write_text(yaml.safe_dump(derived), encoding="utf-8")
            self.crosswalk.to_csv(config / "ct_county_crosswalk.csv", index=False)
            with self.assertRaisesRegex(ValueError, "must not receive.*second time"):
                run(root, config, workers=1, geometry_workers=1)
            with closing(sqlite3.connect(root / "base.gpkg")) as con:
                con.execute("CREATE TABLE integration_run (created_utc TEXT)")
                con.commit()
            with self.assertRaisesRegex(ValueError, "already an integrated"):
                run(root, config, workers=1, geometry_workers=1)

    def test_recreation_csv_joins_current_geoids_and_reports_unmatched(self):
        """Use county dollar totals without allocating historical CT values."""
        source = pd.DataFrame({
            "GEOID": ["01001", "09001", "09110", "01003"],
            "county_cs_2024usd": [0.0, 400.0, 100.0, np.nan],
            "county_visits_2024": [9999, 9999, 9999, 9999],
        })
        ids = self.ids.append(pd.Index(["01003"], name="GEOID"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recreation.csv"
            source.to_csv(path, index=False)
            result = compute_task("recreation", {"recreation": path}, pd.DataFrame(index=ids), self.crosswalk, 1)
        self.assertEqual(set(result["originals"]), {"recreation"})
        values = result["originals"]["recreation"]
        self.assertEqual(values.index.to_list(), ids.to_list())
        np.testing.assert_allclose(values, [100.0, np.nan, 0.0, np.nan])
        self.assertIn(
            {"service": "recreation", "source_key": "09001", "reason": "unmatched_county", "value": 400.0},
            result["issues"],
        )
        self.assertEqual(result["metadata"][0]["source_field"], "county_cs_2024usd")
        self.assertEqual(result["metadata"][0]["input_known_total"], 500.0)
        self.assertEqual(result["metadata"][0]["unmatched_total"], 400.0)

    def test_recreation_csv_rejects_duplicate_county_totals(self):
        """Duplicate county rows must not be silently summed or dropped."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recreation.csv"
            pd.DataFrame({"GEOID": ["01001", "01001"], "county_cs_2024usd": [100.0, 200.0],
                          "county_visits_2024": [10, 20]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                compute_task("recreation", {"recreation": path},
                             pd.DataFrame(index=self.ids), self.crosswalk, 1)

    def test_gpkg_keeps_geometry_source_and_attributes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "base.gpkg"
            output = Path(temp) / "output.gpkg"
            frame = gpd.GeoDataFrame({"GEOID": ["01001", "01003"], "keep": [-12.0, 0.0],
                                     "sum_pollination_attributed_annual_crop_yield_value": [100.0, 200.0],
                                     "ALAND": [1000, 2000], "AWATER": [10, 20],
                                     "sum_unconfigured_quantity": [7.0, 8.0],
                                     **{f"{stat}_quantity": [1.0, 2.0] for stat in (
                                         "mean", "median", "min", "max", "std", "stdev", "p5", "p50", "p95",
                                         "valid_count", "total_count", "area_ha_valid", "area_ha_total",
                                         "proportion_valid_nonzero")},
                                     LEGACY_CROP_FIELD: [1.0, 2.0]},
                                    geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
            frame.to_file(base, layer="counties", driver="GPKG")
            before = sha256(base)
            raw_field = "sum_pollination_attributed_annual_crop_yield_value"
            additions = pd.DataFrame({
                raw_field + "_adj": [9.0, 18.0], "sum_unconfigured_quantity": [7.0, 8.0],
                AIR_QUALITY_OUTPUT + "_adj": [12.0, np.nan], "test_adj": [18.0, np.nan],
                "pollination_adj_factor": [0.09, 0.09], AIR_QUALITY_OUTPUT + "_adj_factor": [1.135, 1.135],
                "test_adj_factor": [1.5, 1.5],
            }, index=frame.GEOID)
            removed = [field for field in frame if field.startswith("sum_") or NON_SUM_FIELD.match(field)]
            table = pd.DataFrame({"service": ["test"], "factor": [1.5]})
            write_output(base, output, "counties", "geom", additions, {"integration_services": table}, removed_fields=removed)
            self.assertEqual(sha256(base), before)
            attrs, _, _ = read_attributes(output)
            self.assertEqual(list(attrs), ["fid", "GEOID", "keep", "ALAND", "AWATER"] + list(additions))
            self.assertEqual(attrs.keep.to_list(), [-12, 0])
            self.assertNotIn(LEGACY_CROP_FIELD, attrs)
            self.assertNotIn(raw_field, attrs)
            self.assertNotIn("pollination_adj", attrs)
            self.assertFalse(any(field.endswith("_orig") for field in attrs))
            self.assertEqual(attrs[raw_field + "_adj"].to_list(), [9, 18])
            self.assertEqual(attrs.pollination_adj_factor.to_list(), [0.09, 0.09])
            self.assertEqual(read_attributes(base)[0][raw_field].to_list(), [100, 200])
            csv = pd.read_csv(output.with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip")
            self.assertNotIn("geometry", csv.columns)
            pd.testing.assert_frame_equal(attrs, csv, check_dtype=False)
            self.assertTrue(np.isnan(csv[AIR_QUALITY_OUTPUT + "_adj"].iloc[1]))
            with closing(sqlite3.connect(output)) as con:
                self.assertEqual(con.execute("SELECT data_type FROM gpkg_contents WHERE table_name='integration_services'").fetchone()[0], "attributes")
            self.assertTrue(gpd.read_file(output, layer="counties").geometry.equals(frame.geometry))
            with self.assertRaises(FileExistsError):
                write_output(base, output, "counties", "geom", additions, {})
            failed_output = Path(temp) / "failed.gpkg"
            invalid = additions.copy()
            invalid.index = ["99999", "01003"]
            with self.assertRaises(KeyError):
                write_output(base, failed_output, "counties", "geom", invalid, {}, removed_fields=removed)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_output.with_suffix(".csv").exists())
            self.assertEqual(sha256(base), before)
            with self.assertRaisesRegex(ValueError, "Only generated statistics"):
                write_output(base, failed_output, "counties", "geom", additions, {},
                             removed_fields=[*removed, "GEOID"])
            overlap = additions.copy()
            overlap["keep"] = 1.0
            with self.assertRaisesRegex(ValueError, "Only generated statistics"):
                write_output(base, failed_output, "counties", "geom", overlap, {}, removed_fields=removed)
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
                    write_output(base, failed_output, "counties", "geom", additions, {}, removed_fields=removed)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_output.with_suffix(".csv").exists())

    def test_service_checkpoint_round_trip_and_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "service.sqlite"
            result = {
                "originals": {"test": pd.Series([12.0, np.nan], index=["01001", "01003"])},
                "statuses": [pd.DataFrame({"GEOID": ["01001", "01003"], "service": ["test"] * 2, "status": ["source", "missing"]})],
                "metadata": [{"service": "test"}], "issues": [], "checks": [],
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

class CountyExportRuleTests(unittest.TestCase):
    """Configured field policy preserves values, attribution, and run provenance."""

    def setUp(self):
        """Build a small county input containing every required rename/exclusion."""
        self.rules = load_export_rules(RULES_PATH)
        # This presentation fixture exercises the pre-existing service subset.
        # Mangrove source allocation and its rename have their own integration tests.
        self.rules["rename_fields"].pop("mangroves_adj")
        self.rules["rename_fields"].pop("corals_adj")
        self.ids = pd.Index(["01001", "01003"], name="GEOID")
        self.base = pd.DataFrame({
            "GEOID": self.ids, "ALAND": [1000, 2000], "AWATER": [10, 20],
            "sum_annual_value_masked_to_wetlands": [10.0, np.nan],
            "sum_marginal_npv_masked_to_wetlands": [200.0, np.nan],
            "sum_totalC_tCO2e_ha_2020": [27.0, np.nan],
            "sum_totalC_tCO2e_pixel_2020": [100.0, np.nan],
            **{f"sum_Wsw_{kind}": [3.0, 4.0] for kind in ("irr", "pow", "pub", "total")},
            "sum_Wval_sw_2020usd_total": [7.0, np.nan],
            "mean_annual_value_masked_to_wetlands": [2.0, np.nan],
            "area_ha_valid_totalC_tCO2e_pixel_2020": [0.09, 0.0],
        }, index=self.ids)
        self.originals = {
            "flood_annual": self.base.sum_annual_value_masked_to_wetlands,
            "flood_npv": self.base.sum_marginal_npv_masked_to_wetlands,
            "carbon_ha": self.base.sum_totalC_tCO2e_ha_2020,
            "carbon_pixel": pd.Series([2.43, np.nan], index=self.ids),
            "avoided_air_quality_heath_costs": pd.Series([50.0, np.nan], index=self.ids),
            "timber": pd.Series([30.0, np.nan], index=self.ids),
        }
        self.factors = pd.read_csv(RULES_PATH.with_name("adjustment_factors.csv"), index_col="service").loc[list(self.originals)]
        self.summaries = pd.DataFrame({
            "sum_dollars_annual": self.originals["avoided_air_quality_heath_costs"],
            "mean_dollars_annual": [50.0 / 9, np.nan],
            "valid_count_dollars_annual": [9, 0], "total_count_dollars_annual": [9, 0],
        }, index=self.ids)

    def test_tracked_rules_name_flood_and_carbon_values_and_remove_only_requested_water_sums(self):
        """Dollar aliases retain actual adjusted totals and distinguish Wsw from Wval."""
        plan = plan_county_export(self.base.columns, self.factors.index, self.rules)
        fields, mapping = adjusted_county_fields(self.base, self.originals, self.summaries, self.factors, plan)
        self.assertEqual(list(fields), [
            "sum_flood_annual_value_masked_to_wetlands_adj",
            "sum_flood_marginal_npv_masked_to_wetlands_adj",
            "sum_Wval_sw_2020usd_total",
            "sum_CO2e_mitigation_dollar_value_per_pixel_adj",
            AIR_QUALITY_OUTPUT + "_adj", "timber_adj",
            "flood_annual_adj_factor", "flood_npv_adj_factor", "carbon_pixel_adj_factor",
            AIR_QUALITY_OUTPUT + "_adj_factor", "timber_adj_factor",
        ])
        self.assertAlmostEqual(fields.loc["01001", mapping["carbon_pixel"]], 27 * 0.09 * 3.66 * 1.182)
        self.assertEqual(fields.loc["01001", "carbon_pixel_adj_factor"], 4.32612)
        self.assertNotIn("sum_totalC_tCO2e_ha_2020_adj", fields)
        self.assertNotIn("carbon_ha_adj_factor", fields)
        self.assertNotIn("carbon_ha", mapping)
        self.assertTrue(all(row["action"] == "exclude" for row in plan if row["service"] == "carbon_ha"))
        self.assertIn("per tonne CO2e price", self.factors.loc["carbon_pixel", "reason"])
        self.assertAlmostEqual(fields.loc["01001", mapping["flood_annual"]], 0.9)
        self.assertAlmostEqual(fields.loc["01001", mapping["flood_npv"]], 18)
        for service in ("flood_annual", "flood_npv"):
            self.assertEqual(fields[service + "_adj_factor"].to_list(), [0.09, 0.09])
            self.assertIn("per hectare", self.factors.loc[service, "source_units"])
            self.assertIn("no inflation adjustment", self.factors.loc[service, "reason"])
        self.assertAlmostEqual(fields.loc["01001", mapping["avoided_air_quality_heath_costs"]], 56.75)
        self.assertEqual(fields.loc["01001", "sum_Wval_sw_2020usd_total"], 7)
        self.assertFalse(any(field.startswith("sum_Wsw_") for field in fields))
        self.assertFalse(any(NON_SUM_FIELD.match(field) for field in fields))
        self.assertTrue(fields.loc["01003", [field for field in fields if not field.endswith("_adj_factor")]].isna().all())
        self.assertEqual(mapping["carbon_pixel"], "sum_CO2e_mitigation_dollar_value_per_pixel_adj")
        self.assertEqual(mapping["avoided_air_quality_heath_costs"], AIR_QUALITY_OUTPUT + "_adj")

    def test_base_task_integrates_density_without_reading_old_pixel_sums(self):
        """Convert density to fractional tonnes even if an old pixel sum is present."""
        raw = pd.DataFrame(0.0, index=self.ids, columns=list(BASE_FIELDS.values()))
        raw["sum_totalC_tCO2e_ha_2020"] = [27.0, np.nan]
        for frame in (raw, raw.assign(sum_totalC_tCO2e_pixel_2020="not used")):
            result = compute_task("base", {}, frame, pd.DataFrame(), 1)
            tonnes = result["originals"]["carbon_pixel"]
            self.assertAlmostEqual(tonnes.loc["01001"], 2.43)
            self.assertTrue(pd.isna(tonnes.loc["01003"]))
            metadata = next(row for row in result["metadata"] if row["service"] == "carbon_pixel")
            self.assertEqual(metadata["source_field"], "sum_totalC_tCO2e_ha_2020")
            self.assertAlmostEqual(metadata["original_total"], 2.43)

    def test_yaml_only_rename_and_group_order_preserve_semantic_sum_group(self):
        """A custom alias stays with its source group regardless of its prefix."""
        self.rules["rename_fields"]["carbon_pixel_adj"] = "carbon_dollar_value_adj"
        self.rules["column_groups"] = ["base_attributes", "adjustment_factors", "sum_values", "service_totals"]
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "export_rules.yaml"
            custom.write_text(yaml.safe_dump(self.rules), encoding="utf-8")
            loaded = load_export_rules(custom)
        plan = plan_county_export(self.base.columns, self.factors.index, loaded)
        fields, mapping = adjusted_county_fields(self.base, self.originals, self.summaries, self.factors, plan)
        self.assertTrue(all(field.endswith("_adj_factor") for field in list(fields)[:5]))
        self.assertLess(list(fields).index("carbon_dollar_value_adj"), list(fields).index(AIR_QUALITY_OUTPUT + "_adj"))
        self.assertEqual(mapping["carbon_pixel"], "carbon_dollar_value_adj")
        self.assertAlmostEqual(fields.loc["01001", "carbon_dollar_value_adj"], 2.43 * 4.32612)
        self.assertEqual(next(row["group"] for row in plan if row["output_field"] == "carbon_dollar_value_adj"), "sum_values")
        self.assertEqual(plan, plan_county_export(self.base.columns, self.factors.index, loaded))

    def test_rules_reject_stale_names_and_protect_base_attributes(self):
        """Misspelled exact rules and county attribute edits fail before calculation."""
        for setting, value, message in (
            ("exclude_fields", ["sum_Wsw_typo"], "Unknown required exclusion"),
            ("rename_fields", {"sum_typo_adj": "better_name"}, "Unknown rename source"),
            ("exclude_fields", ["ALAND"], "cannot remove or rename county attribute"),
            ("rename_fields", {"GEOID": "county_id"}, "cannot remove or rename county attribute"),
            ("exclude_patterns", ["A*"], "cannot remove or rename county attribute"),
        ):
            with self.subTest(setting=setting, value=value):
                rules = deepcopy(self.rules)
                rules[setting] = value
                with self.assertRaisesRegex(ValueError, message):
                    plan_county_export(self.base.columns, self.factors.index, rules)
        with self.assertRaisesRegex(ValueError, "Missing source sums"):
            plan_county_export(self.base.drop(columns="sum_totalC_tCO2e_ha_2020").columns,
                              self.factors.index, self.rules)

    def test_rules_reject_collisions_and_overlapping_actions(self):
        """SQLite case-insensitive names and simultaneous policy conflicts are checked."""
        source = "carbon_pixel_adj"
        for target in ("geoid", "GEOM", "SUM_FLOOD_ANNUAL_VALUE_MASKED_TO_WETLANDS_ADJ"):
            with self.subTest(target=target):
                rules = deepcopy(self.rules)
                rules["rename_fields"][source] = target
                with self.assertRaisesRegex(ValueError, "names collide"):
                    plan_county_export(self.base.columns, self.factors.index, rules)
        rules = deepcopy(self.rules)
        rules["exclude_fields"].append(source)
        with self.assertRaisesRegex(ValueError, "both exclude and rename"):
            plan_county_export(self.base.columns, self.factors.index, rules)

    def test_excluding_service_total_also_excludes_its_factor(self):
        """Exported service values retain factors and omitted totals leave no orphan factor."""
        rules = deepcopy(self.rules)
        rules["exclude_fields"].append("timber_adj")
        plan = plan_county_export(self.base.columns, self.factors.index, rules)
        fields, mapping = adjusted_county_fields(self.base, self.originals, self.summaries, self.factors, plan)
        self.assertNotIn("timber_adj", fields)
        self.assertNotIn("timber_adj_factor", fields)
        self.assertNotIn("timber", mapping)
        self.assertTrue(all(row["action"] == "exclude" for row in plan if row["service"] == "timber"))
        rules = deepcopy(self.rules)
        rules["exclude_fields"].append("timber_adj_factor")
        with self.assertRaisesRegex(ValueError, "Keep the factor"):
            plan_county_export(self.base.columns, self.factors.index, rules)

    def test_excluding_every_value_fails_before_output_writing(self):
        """A county-attributes-only policy fails before producing empty SQL updates."""
        rules = deepcopy(self.rules)
        rules["rename_fields"] = {}
        rules["exclude_patterns"].extend(["sum_*", "*_adj"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            plan_county_export(self.base.columns, self.factors.index, rules)

    def test_yaml_rejects_duplicate_unknown_invalid_and_escaping_settings(self):
        """Malformed presentation settings cannot silently affect a published result."""
        cases = [
            ("duplicate top-level key", RULES_PATH.read_text(encoding="utf-8") + "\nschema_version: 1\n"),
            ("duplicate nested key", RULES_PATH.read_text(encoding="utf-8") + "\n  carbon_pixel_adj: duplicate\n"),
        ]
        for label, mutate in (
            ("unknown setting", lambda rules: rules.update({"typo": True})),
            ("invalid schema", lambda rules: rules.update({"schema_version": 2})),
            ("unknown output setting", lambda rules: rules["output"].update({"typo": "x"})),
            ("duplicate groups", lambda rules: rules["column_groups"].append("sum_values")),
            ("base not first", lambda rules: rules.update({"column_groups": list(reversed(rules["column_groups"]))})),
            ("invalid identifier", lambda rules: rules["rename_fields"].update({"timber_adj": "two words"})),
            ("parent directory", lambda rules: rules["output"].update({"directory": "../elsewhere"})),
            ("absolute directory", lambda rules: rules["output"].update({"directory": "C:/elsewhere"})),
            ("rooted forward slash", lambda rules: rules["output"].update({"directory": "/escape"})),
            ("rooted backslash", lambda rules: rules["output"].update({"directory": "\\escape"})),
            ("escaping filename", lambda rules: rules["output"].update({"filename_prefix": "../elsewhere"})),
            ("invalid layer", lambda rules: rules["output"].update({"layer_name": "two words"})),
            ("empty layer", lambda rules: rules["output"].update({"layer_name": ""})),
            ("null layer", lambda rules: rules["output"].update({"layer_name": None})),
            ("reserved layer", lambda rules: rules["output"].update({"layer_name": "gpkg_contents"})),
            ("audit layer", lambda rules: rules["output"].update({"layer_name": "integration_run"})),
        ):
            rules = deepcopy(self.rules)
            mutate(rules)
            cases.append((label, yaml.safe_dump(rules)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export_rules.yaml"
            for label, text in cases:
                with self.subTest(label=label):
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_export_rules(path)

    def test_policy_round_trip_keeps_geometry_csv_order_and_embedded_rules(self):
        """A published pair contains the new names and reproducible presentation policy."""
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.gpkg"
            output = Path(directory) / "output.gpkg"
            source = gpd.GeoDataFrame(self.base.reset_index(drop=True),
                                      geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
            source.to_file(base_path, layer="counties", driver="GPKG")
            source_hash = sha256(base_path)
            attributes, layer, geom = read_attributes(base_path)
            attributes = attributes.set_index("GEOID", drop=False)
            plan = plan_county_export(attributes.columns, self.factors.index, self.rules, geom)
            fields, _ = adjusted_county_fields(attributes, self.originals, self.summaries, self.factors, plan)
            audit = pd.DataFrame(plan)
            snapshot = pd.DataFrame([{"path": "export_rules.yaml", "sha256": sha256(RULES_PATH),
                                      "yaml_text": RULES_PATH.read_text(encoding="utf-8")}])
            removed = [row["source_field"] for row in plan if row["source_field"] and row["group"] != "base_attributes"]
            write_output(base_path, output, layer, geom, fields,
                         {"integration_export_rules": snapshot, "integration_export_fields": audit},
                         removed_fields=removed, output_layer=self.rules["output"]["layer_name"])
            saved = read_attributes(output)[0]
            csv = pd.read_csv(output.with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip")
            self.assertEqual(list(saved), ["fid", "GEOID", "ALAND", "AWATER"] + list(fields))
            pd.testing.assert_frame_equal(saved, csv, check_dtype=False)
            for field, expected in (
                ("sum_flood_annual_value_masked_to_wetlands_adj", 0.9),
                ("sum_flood_marginal_npv_masked_to_wetlands_adj", 18),
                ("flood_annual_adj_factor", 0.09),
                ("flood_npv_adj_factor", 0.09),
            ):
                self.assertAlmostEqual(saved.loc[0, field], expected)
            self.assertEqual(sha256(base_path), source_hash)
            self.assertEqual(read_attributes(output)[1], self.rules["output"]["layer_name"])
            self.assertTrue(gpd.read_file(output, layer=self.rules["output"]["layer_name"]).geometry.equals(source.geometry))
            with closing(sqlite3.connect(output)) as con:
                persisted = pd.read_sql_query("SELECT path,sha256,yaml_text FROM integration_export_rules", con)
                pd.testing.assert_frame_equal(persisted, snapshot)
                mapping = pd.read_sql_query("SELECT source_field,output_field,action FROM integration_export_fields", con)
                row = mapping.loc[mapping.source_field == "sum_totalC_tCO2e_pixel_2020"].iloc[0]
                self.assertTrue(pd.isna(row.output_field))
                self.assertEqual(row.action, "exclude")
                self.assertAlmostEqual(saved.loc[0, "sum_CO2e_mitigation_dollar_value_per_pixel_adj"], 2.43 * 4.32612)
                omitted = mapping[mapping.source_field.str.startswith("sum_Wsw_")]
                self.assertEqual(len(omitted), 4)
                self.assertTrue(omitted.output_field.isna().all())
                self.assertTrue((omitted.action == "exclude").all())


class CountyLayerNameTests(unittest.TestCase):
    """Renaming a feature layer preserves data, coverage and GeoPackage behavior."""

    def test_legacy_layer_renamed_without_changing_csv_geometry_or_spatial_index(self):
        """The 50_states label changes neither territories nor any feature value."""
        old = "tl_2024_us_county_60_states"
        new = load_export_rules(RULES_PATH)["output"]["layer_name"]
        self.assertEqual(new, "tl_2024_us_county_50_states")
        source = gpd.GeoDataFrame(
            {"GEOID": ["01001", "11001", "72001", "78010"],
             "STATEFP": ["01", "11", "72", "78"]},
            geometry=[box(i * 2, 0, i * 2 + 1, 1) for i in range(4)], crs=6933)
        fields = pd.DataFrame({"timber_adj": [0.0, 10.0, 20.0, np.nan]},
                              index=pd.Index(source.GEOID, name="GEOID"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, legacy, renamed = root / "base.gpkg", root / "legacy.gpkg", root / "renamed.gpkg"
            source.to_file(base, layer=old, driver="GPKG")
            original_hash = sha256(base)
            _, _, geom = read_attributes(base)
            write_output(base, legacy, old, geom, fields, {})
            write_output(base, renamed, old, geom, fields, {}, output_layer=new)
            self.assertEqual(sha256(base), original_hash)
            self.assertEqual(legacy.with_suffix(".csv").read_bytes(), renamed.with_suffix(".csv").read_bytes())
            attributes, layer, saved_geom = read_attributes(renamed)
            self.assertEqual((layer, saved_geom), (new, geom))
            self.assertEqual(attributes.GEOID.to_list(), source.GEOID.to_list())
            self.assertTrue(gpd.read_file(renamed, layer=new).geometry.equals(source.geometry))
            with closing(sqlite3.connect(base)) as original, closing(sqlite3.connect(renamed)) as saved:
                self.assertEqual(saved.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(saved.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertEqual(saved.execute(
                    "SELECT table_name,identifier FROM gpkg_contents WHERE data_type='features'"
                ).fetchall(), [(new, new)])
                self.assertEqual(saved.execute(
                    "SELECT table_name FROM gpkg_extensions WHERE extension_name='gpkg_rtree_index'"
                ).fetchall(), [(new,)])
                self.assertEqual(original.execute(f'SELECT fid,"{geom}" FROM "{old}" ORDER BY fid').fetchall(),
                                 saved.execute(f'SELECT fid,"{geom}" FROM "{new}" ORDER BY fid').fetchall())
                self.assertEqual(saved.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE ?", (f"%{old}%",)).fetchall(), [])
                self.assertEqual(saved.execute(f'SELECT COUNT(*) FROM "rtree_{new}_{geom}"').fetchone()[0], 4)
            # Check that renamed spatial-index triggers still work, not just that
            # the copied index has the right row count. Edit this temporary test
            # deliverable only, after checking exact CSV/GeoPackage parity.
            dataset = ogr.Open(str(renamed), update=1)
            try:
                layer = dataset.GetLayerByName(new)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("GEOID", "01003")
                feature.SetField("STATEFP", "01")
                feature.SetGeometry(ogr.CreateGeometryFromWkt(box(20, 0, 21, 1).wkt))
                self.assertEqual(layer.CreateFeature(feature), 0)
                feature = None
                layer.SetSpatialFilterRect(19, -1, 22, 2)
                self.assertEqual(layer.GetFeatureCount(), 1)
                layer = None
            finally:
                dataset = None
            with closing(sqlite3.connect(renamed)) as saved:
                self.assertEqual(saved.execute(f'SELECT COUNT(*) FROM "rtree_{new}_{geom}"').fetchone()[0], 5)

    def test_already_correct_layer_and_legacy_config_are_supported(self):
        """Correct input names need no rename; older YAML can omit layer_name."""
        rules = load_export_rules(RULES_PATH)
        new = rules["output"]["layer_name"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, base, output = root / "export_rules.yaml", root / "base.gpkg", root / "output.gpkg"
            rules["output"].pop("layer_name")
            config.write_text(yaml.safe_dump(rules), encoding="utf-8")
            self.assertNotIn("layer_name", load_export_rules(config)["output"])
            gpd.GeoDataFrame({"GEOID": ["72001"]}, geometry=[box(0, 0, 1, 1)], crs=6933).to_file(
                base, layer=new, driver="GPKG")
            fields = pd.DataFrame({"timber_adj": [1.0]}, index=pd.Index(["72001"], name="GEOID"))
            _, _, geom = read_attributes(base)
            with patch("build_county_ecosystem_service_valuations_2026.ogr.Open",
                       side_effect=AssertionError("No rename is necessary")):
                write_output(base, output, new, geom, fields, {}, output_layer=new)
            self.assertEqual(read_attributes(output)[1], new)

    def test_layer_collision_does_not_publish_or_modify_source(self):
        """An occupied target name fails safely before publishing either file."""
        old, new = "tl_2024_us_county_60_states", "tl_2024_us_county_50_states"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, output = root / "base.gpkg", root / "output.gpkg"
            source = gpd.GeoDataFrame({"GEOID": ["01001"]}, geometry=[box(0, 0, 1, 1)], crs=6933)
            source.to_file(base, layer=old, driver="GPKG")
            source.to_file(base, layer=new, driver="GPKG")
            original_hash = sha256(base)
            fields = pd.DataFrame({"timber_adj": [1.0]}, index=pd.Index(["01001"], name="GEOID"))
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_output(base, output, old, "geom", fields, {}, output_layer=new)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".csv").exists())
            self.assertEqual(sha256(base), original_hash)


if __name__ == "__main__":
    unittest.main()
