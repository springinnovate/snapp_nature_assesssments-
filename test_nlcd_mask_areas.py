"""Regression tests for mask hectares, NoData, provenance and unchanged services."""

from contextlib import closing
import configparser
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from build_county_ecosystem_service_valuations_2026 import read_attributes, read_db, sha256
from nlcd_mask_areas import CLASSES, FIELD_DEFINITIONS, OPERATIONS, derive_nlcd_mask_areas
from prepare_tif_zonal_stats_config import prepare_config
from update_nlcd_mask_areas import DEFAULT_CONFIG, export_mask_areas, run_mask_update, validate_mask_definition


def mask_statistics() -> pd.DataFrame:
    """Build three counties: partial NoData, all NoData, and outside the raster.

    Returns:
        Raw toolkit-style binary mask statistics with a 0.09 hectare pixel.
    """
    data = {"GEOID": ["01001", "01003", "01005"]}
    for name in CLASSES:
        for op, values in {
            "sum": [1, 0, 0], "valid_count": [6, 0, 0], "total_count": [7, 7, 0],
            "area_ha_valid": [0.54, 0, 0], "area_ha_total": [0.63, 0.63, 0],
            "min": [0, np.nan, np.nan], "max": [1, np.nan, np.nan],
        }.items():
            data[f"{op}_reclassified_NLCD2023_{name}_2026_05_18_12_57_08"] = values
    return pd.DataFrame(data)


def write_fixture(root: Path) -> tuple[Path, Path]:
    """Write tiny co-registered rasters and a tracked-style mask profile.

    Args:
        root: Temporary test directory.

    Returns:
        Reviewed profile path and an existing county valuation GeoPackage.
    """
    config = configparser.ConfigParser(interpolation=None)
    config.read(DEFAULT_CONFIG, encoding="utf-8")
    config["project"].update({"raster_dir": "masks", "zonal_units_dir": "zones",
                              "results_dir": "results", "global_work_dir": "work",
                              "derived_raster_dir": "derived", "max_simplify_tolerance_meters": "0"})
    for entry in json.loads(config["project"]["raster_inventory"]):
        path = root / "masks" / entry["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((2, 7), dtype=np.uint8)
        pixels[0, CLASSES.index(entry["label"])] = 1
        pixels[:, 6] = 255
        pixels[1, :] = 255
        with rasterio.open(path, "w", driver="GTiff", width=7, height=2, count=1,
                           dtype="uint8", crs="EPSG:5070", transform=from_origin(0, 60, 30, 30), nodata=255) as dst:
            dst.write(pixels, 1)
    vector = root / "zones/counties/tl_2024_us_county_50_states.gpkg"
    vector.parent.mkdir(parents=True)
    counties = gpd.GeoDataFrame({"GEOID": ["01001", "01003", "01005"], "NAME": ["A", "B", "C"]},
                                geometry=[box(0, 30, 210, 60), box(0, 0, 210, 30), box(300, 0, 510, 30)], crs="EPSG:5070")
    counties.to_file(vector, layer="tl_2024_us_county_60_states", driver="GPKG")
    base = root / "existing_services.gpkg"
    counties["sum_example_service_adj"] = [123.456, -12.5, np.nan]
    counties["example_adj_factor"] = 1.368
    counties.to_file(base, layer="tl_2024_us_county_50_states", driver="GPKG")
    profile = root / "profile.yaml"
    with profile.open("w", encoding="utf-8") as stream:
        config.write(stream)
    return profile, base


class MaskAreaTests(unittest.TestCase):
    """Exercise conversion and mask-only export safety."""

    def test_nodata_and_zero_are_not_class_area(self):
        """Count class pixels directly, not valid coverage times old proportion."""
        raw = mask_statistics()
        raw["sum_service_adj"] = [42, 50, 0]
        result = derive_nlcd_mask_areas(raw)
        self.assertEqual(list(result), ["GEOID", "sum_service_adj"] + list(FIELD_DEFINITIONS))
        self.assertAlmostEqual(result.area_ha_valid_nlcd[0], .54)
        self.assertAlmostEqual(result.area_ha_nlcd_forests[0], .09)
        self.assertAlmostEqual(result.proportion_nlcd_forests_of_valid[0], 1 / 6)
        self.assertAlmostEqual(result.area_ha_all_natural[0], .45)
        self.assertNotAlmostEqual(.54 * (1 / 7), .09)
        for row in (1, 2):
            self.assertEqual(result.area_ha_valid_nlcd[row], 0)
            self.assertEqual(result.area_ha_all_natural[row], 0)
            self.assertTrue(pd.isna(result.proportion_nlcd_forests_of_valid[row]))
        pd.testing.assert_series_equal(result.sum_service_adj, raw.sum_service_adj)

    def test_empty_sum_and_empty_input(self):
        """Handle null empty sums and empty frames without fabricating coverage."""
        raw = mask_statistics()
        for c in raw:
            if c.startswith("sum_"):
                raw.loc[1:, c] = np.nan
        result = derive_nlcd_mask_areas(raw)
        self.assertEqual(result.area_ha_all_natural[1], 0)
        self.assertTrue(derive_nlcd_mask_areas(raw.iloc[:0]).empty)
        unchanged = pd.DataFrame({"GEOID": ["01001"], "sum_service_adj": [1.]})
        pd.testing.assert_frame_equal(derive_nlcd_mask_areas(unchanged), unchanged)

    def test_full_class_coverage_and_no_class_pixels(self):
        """A fully forested footprint has share one; other class shares are zero."""
        raw = mask_statistics()
        for name in CLASSES:
            for op in ("sum", "min", "max"):
                raw.loc[0, f"{op}_reclassified_NLCD2023_{name}_2026_05_18_12_57_08"] = (
                    (6 if op == "sum" else 1) if name == "forests" else 0)
        result = derive_nlcd_mask_areas(raw)
        self.assertAlmostEqual(result.area_ha_all_natural[0], .54)
        self.assertEqual(result.proportion_nlcd_forests_of_valid[0], 1)
        self.assertEqual(result.proportion_nlcd_wetlands_of_valid[0], 0)

    def test_reject_legacy_incomplete_ambiguous_and_invalid_statistics(self):
        """Stop on unsafe inputs instead of silently preserving incorrect areas."""
        raw = mask_statistics()
        with self.assertRaisesRegex(ValueError, "Rerun masks only"):
            derive_nlcd_mask_areas(raw[[c for c in raw if c.startswith("area_ha_valid_")]])
        with self.assertRaisesRegex(ValueError, "Multiple versions"):
            duplicate = raw.copy()
            duplicate["sum_reclassified_NLCD2023_forests"] = 1
            derive_nlcd_mask_areas(duplicate)
        for prefix, value in (("valid_count", 7), ("sum", 1.5), ("max", 255), ("area_ha_valid", np.nan)):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                changed = raw.copy()
                column = f"{prefix}_reclassified_NLCD2023_forests_2026_05_18_12_57_08"
                changed[column] = changed[column].astype(float)
                changed.loc[0, column] = value
                derive_nlcd_mask_areas(changed)
        overlap = raw.copy()
        for name in CLASSES:
            overlap.loc[0, f"sum_reclassified_NLCD2023_{name}_2026_05_18_12_57_08"] = 2
        with self.assertRaisesRegex(ValueError, "exceed valid coverage"):
            derive_nlcd_mask_areas(overlap)

    def test_profile_grid_validation_and_frozen_mask_only_config(self):
        """Freeze only the five masks and reject a misaligned raster."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, _ = write_fixture(root)
            metadata = validate_mask_definition(profile)
            self.assertEqual([r["pixel_area_ha"] for r in metadata], [.09] * 5)
            frozen = prepare_config(profile, output_path=root / "run.yaml")
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read(frozen, encoding="utf-8")
            self.assertEqual([s for s in cfg if s.startswith("job:")], ["job:counties_masks"])
            self.assertEqual(set(cfg["job:counties_masks"]["operations"].replace(" ", "").split(",")), set(OPERATIONS))
            self.assertNotIn("ecosystem_services", cfg["job:counties_masks"]["base_raster_pattern"])
            self.assertIn("snapp_source_config_sha256", cfg["project"])
            with rasterio.open(metadata[0]["path"], "r+") as raster:
                raster.transform = from_origin(1, 60, 30, 30)
            with self.assertRaisesRegex(ValueError, "identical CRS"):
                validate_mask_definition(profile)

    def test_append_preserves_values_geometry_and_csv_parity(self):
        """Join by GEOID, preserve inputs, and publish only matching new pairs."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, base = write_fixture(root)
            frozen = prepare_config(profile, output_path=root / "run.yaml")
            raw = root / "results/counties/counties_masks_20261026_000000.gpkg"
            raw.parent.mkdir(parents=True)
            geometries = gpd.read_file(base).set_index("GEOID").geometry
            stats = mask_statistics().set_index("GEOID")
            # Deliberately permute raw county row order; joins must use GEOID.
            raw_frame = gpd.GeoDataFrame(stats, geometry=geometries).reset_index().iloc[::-1]
            raw_frame.to_file(raw, layer="tl_2024_us_county_60_states", driver="GPKG")
            before = sha256(base)
            output = root / "updated.gpkg"
            export_mask_areas(raw, output, frozen, base, validate_mask_definition(profile), derived_config=None)
            self.assertEqual(sha256(base), before)
            attrs, layer, _ = read_attributes(output)
            self.assertEqual(layer, "tl_2024_us_county_50_states")
            csv = pd.read_csv(output.with_suffix(".csv"), dtype={"GEOID": str, "NAME": str}, float_precision="round_trip")
            pd.testing.assert_frame_equal(attrs, csv, check_dtype=False)
            self.assertEqual(attrs.sum_example_service_adj.iloc[:2].tolist(), [123.456, -12.5])
            self.assertTrue(gpd.read_file(base).geometry.geom_equals(gpd.read_file(output, layer=layer).geometry).all())
            with closing(read_db(output)) as con:
                self.assertEqual(con.execute("SELECT count(*) FROM nlcd_area_fields").fetchone()[0], 12)
                self.assertEqual(con.execute("SELECT count(*) FROM nlcd_area_sources").fetchone()[0], 5)
            with self.assertRaises(FileExistsError):
                export_mask_areas(raw, output, frozen, base, derived_config=None)
            with self.assertRaisesRegex(ValueError, "already contains"):
                export_mask_areas(raw, root / "second.gpkg", frozen, output)

    def test_reject_mismatched_county_boundaries(self):
        """Matching GEOIDs alone do not make different county boundaries safe."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, base = write_fixture(root)
            frozen = prepare_config(profile, output_path=root / "run.yaml")
            raw = root / "results/counties/counties_masks_20261026_000000.gpkg"
            raw.parent.mkdir(parents=True)
            frame = gpd.GeoDataFrame(mask_statistics(), geometry=[box(0, 0, 1, 1)] * 3, crs="EPSG:5070")
            frame.to_file(raw, layer="counties", driver="GPKG")
            with self.assertRaisesRegex(ValueError, "boundaries differ"):
                export_mask_areas(raw, root / "wrong.gpkg", frozen, base)
            self.assertFalse((root / "wrong.gpkg").exists())

    def test_runner_failure_cannot_publish_final_outputs(self):
        """Keep the frozen config on failure but never publish incomplete results."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, base = write_fixture(root)
            with patch("update_nlcd_mask_areas.ROOT", root), patch(
                "update_nlcd_mask_areas.subprocess.run", side_effect=subprocess.CalledProcessError(1, "runner")
            ) as launch, self.assertRaises(subprocess.CalledProcessError):
                run_mask_update(Path(__file__), profile, base, derived_config=None)
            self.assertIn("--job-workers", launch.call_args.args[0])
            self.assertEqual(len(list((root / "runs").glob("*.yaml"))), 1)
            self.assertFalse((root / "data/analysis_results/combined").exists())

    @unittest.skipUnless(os.environ.get("SNAPP_TEST_ZONAL_RUNNER"), "Set SNAPP_TEST_ZONAL_RUNNER for real toolkit integration")
    def test_real_toolkit_mask_only_workflow(self):
        """Run five tiny TIFFs through the real toolkit and both output exports."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile, base = write_fixture(root)
            before = sha256(base)
            with patch("update_nlcd_mask_areas.ROOT", root):
                outputs = run_mask_update(Path(os.environ["SNAPP_TEST_ZONAL_RUNNER"]), profile, base,
                                          raster_workers=1, derived_config=None)
            self.assertEqual(len(outputs), 2)
            self.assertEqual(sha256(base), before)
            for output in outputs:
                attrs, _, _ = read_attributes(output)
                result = attrs.set_index("GEOID")
                self.assertAlmostEqual(result.loc["01001", "area_ha_all_natural"], .45)
                self.assertAlmostEqual(result.loc["01001", "area_ha_valid_nlcd"], .54)
                self.assertEqual(result.loc["01003", "area_ha_all_natural"], 0)
                self.assertTrue(output.with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
