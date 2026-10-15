"""Focused tests for explicit TIFF zonal-statistics config preparation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import configparser
import hashlib
import json
import subprocess
import unittest

import fiona
import numpy as np
from osgeo import gdal, osr

from prepare_tif_zonal_stats_config import (
    DEFAULT_CONFIG,
    RasterInput,
    discover_zone_jobs,
    load_and_prepare_rasters,
    load_run_config,
    prepare_config,
    write_runner_config,
)


class TifZonalStatsConfigTest(unittest.TestCase):
    """Verify scaling, newest-vector discovery, and config rendering."""

    def test_scaled_raster_is_a_lazy_vrt(self) -> None:
        """Verify an included per-hectare raster is scaled without TIFF copying."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raster_dir = root / "rasters"
            raster_dir.mkdir()
            source_path = raster_dir / "pollination.tif"
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(str(source_path), 2, 2, 1, gdal.GDT_Float32)
            dataset.SetGeoTransform((0, 30, 0, 60, 0, -30))
            spatial_reference = osr.SpatialReference()
            spatial_reference.ImportFromEPSG(5070)
            dataset.SetProjection(spatial_reference.ExportToWkt())
            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(-1)
            band.WriteArray(np.array([[1, 2], [-1, 4]], dtype=np.float32))
            dataset = None

            manifest_path = root / "run.yaml"
            manifest_path.write_text(
                """
[project]
max_simplify_tolerance_meters = 15
raster_inventory = [{"label": "Pollination", "filename": "pollination.tif", "units": "USD per ha", "include": true, "scale": 0.09, "derived_filename": "pollination_per_pixel.vrt"}, {"label": "Missing pending raster", "filename": "pending.tif", "units": "USD", "include": false, "reason": "not delivered"}]
[job:example]
""".strip(),
                encoding="utf-8",
            )

            selected, blocked = load_and_prepare_rasters(
                manifest_path,
                raster_dir,
                root / "derived",
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].path.suffix, ".vrt")
            self.assertIn("not delivered", blocked[0])
            scaled_dataset = gdal.Open(str(selected[0].path), gdal.GA_ReadOnly)
            scaled = scaled_dataset.GetRasterBand(1).ReadAsArray()
            self.assertAlmostEqual(float(scaled[0, 0]), 0.09, places=6)
            self.assertAlmostEqual(float(scaled[1, 1]), 0.36, places=6)
            self.assertEqual(float(scaled[1, 0]), -1.0)
            scaled_dataset = None
            with self.assertRaises(FileExistsError):
                load_and_prepare_rasters(manifest_path, raster_dir, root / "derived")

    def test_newest_county_cut_vector_is_rendered(self) -> None:
        """Verify timestamp discovery chooses the newest requested land product."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            zonal_units_dir = root / "zonal_units"
            all_lands_dir = zonal_units_dir / "padus_all_lands_by_county"
            all_lands_dir.mkdir(parents=True)
            layer = "padus_all_lands_clipped_by_county"
            schema = {"geometry": "Point", "properties": {"GEOID": "str:5"}}
            for timestamp in ("2026_08_07_01_00_00", "2026_08_09_01_00_00"):
                path = (
                    all_lands_dir
                    / f"padus_all_lands_clipped_by_county_{timestamp}.gpkg"
                )
                with fiona.open(
                    path,
                    mode="w",
                    driver="GPKG",
                    layer=layer,
                    schema=schema,
                    crs="EPSG:5070",
                ) as destination:
                    destination.write(
                        {
                            "geometry": {"type": "Point", "coordinates": (0, 0)},
                            "properties": {"GEOID": "00001"},
                        }
                    )

            jobs, unavailable = discover_zone_jobs(
                zonal_units_dir,
                selected_keys={"all"},
            )
            self.assertFalse(unavailable)
            self.assertEqual(len(jobs), 1)
            self.assertTrue(
                jobs[0].vector_path.name.endswith("2026_08_09_01_00_00.gpkg")
            )

            raster_path = root / "example.tif"
            raster_path.touch()
            config_path = root / "configs" / "test_config.yaml"
            write_runner_config(
                config_path,
                root / "work",
                root / "results",
                [RasterInput("Example", raster_path, "USD per pixel", raster_path, 1)],
                jobs,
            )
            config_text = config_path.read_text(encoding="utf-8")
            self.assertIn("2026_08_09_01_00_00.gpkg", config_text)
            self.assertNotIn("2026_08_07_01_00_00.gpkg", config_text)
            self.assertIn("[job:pad_ecosystem_services]", config_text)
            self.assertIn("max_simplify_tolerance_meters = 15", config_text)

    def test_run_snapshots_preserve_policy_and_conversion_history(self) -> None:
        """Changing future scale/metrics cannot mutate an earlier prepared run."""
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rasters = root / "rasters"
            rasters.mkdir()
            source = rasters / "pollination.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(str(source), 1, 1, 1, gdal.GDT_Float32)
            dataset.SetGeoTransform((0, 30, 0, 30, 0, -30))
            spatial_reference = osr.SpatialReference()
            spatial_reference.ImportFromEPSG(5070)
            dataset.SetProjection(spatial_reference.ExportToWkt())
            dataset.GetRasterBand(1).WriteArray(np.array([[100]], dtype=np.float32))
            dataset = None
            zones = root / "zones" / "counties"
            zones.mkdir(parents=True)
            with fiona.open(zones / "tl_2024_us_county_50_states.gpkg", mode="w",
                            driver="GPKG", layer="tl_2024_us_county_60_states",
                            schema={"geometry": "Point", "properties": {"GEOID": "str:5"}},
                            crs="EPSG:5070") as destination:
                destination.write({"geometry": {"type": "Point", "coordinates": (0, 0)},
                                   "properties": {"GEOID": "00001"}})
            definition = load_run_config(DEFAULT_CONFIG)
            project = definition["project"]
            for option, path in (("raster_dir", rasters), ("zonal_units_dir", root / "zones"),
                                 ("derived_raster_dir", root / "derived"),
                                 ("results_dir", root / "results"), ("global_work_dir", root / "work")):
                project[option] = str(path)
            inventory = [
                {"label": "Pollination", "filename": source.name, "units": "USD per ha",
                 "include": True, "scale": 0.09, "derived_filename": "pollination_per_pixel.vrt"},
                {"label": "Pending", "filename": "pending.tif", "units": "USD",
                 "include": False, "reason": "not delivered"},
            ]
            project["raster_inventory"] = json.dumps(inventory)
            config_path = root / "definition.yaml"
            with config_path.open("w", encoding="utf-8") as stream:
                definition.write(stream)
            original_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            first = prepare_config(config_path)
            first_bytes = first.read_bytes()
            frozen = configparser.ConfigParser(interpolation=None)
            frozen.read(first, encoding="utf-8")
            provenance = frozen["project"]
            self.assertEqual(provenance["name"], first.stem)
            self.assertEqual(provenance["snapp_source_config_sha256"], original_hash)
            recorded = json.loads(provenance["snapp_source_definition"])
            self.assertEqual(recorded["project"]["raster_inventory"], inventory)
            self.assertEqual(len(json.loads(provenance["snapp_unavailable_zones"])), 5)
            first_raster = (first.parent / json.loads(provenance["snapp_rasters"])[0]["analysis_path"]).resolve()
            self.assertAlmostEqual(float(gdal.Open(str(first_raster)).ReadAsArray()[0, 0]), 9)
            self.assertFalse(any("*" in section.get("agg_vector", "") for section in frozen.values()))

            inventory[0]["scale"] = 0.18
            project["raster_inventory"] = json.dumps(inventory)
            project["max_simplify_tolerance_meters"] = "7.5"
            definition["job:counties_ecosystem_services"]["operations"] = "sum, valid_count"
            with config_path.open("w", encoding="utf-8") as stream:
                definition.write(stream)
            second = prepare_config(config_path, selected_keys={"counties"})
            frozen.read(second, encoding="utf-8")
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), first_bytes)
            self.assertEqual(frozen["project"]["max_simplify_tolerance_meters"], "7.5")
            self.assertEqual(frozen["job:counties_ecosystem_services"]["operations"], "sum, valid_count")
            second_raster = (second.parent / json.loads(frozen["project"]["snapp_rasters"])[0]["analysis_path"]).resolve()
            self.assertNotEqual(first_raster, second_raster)
            self.assertAlmostEqual(float(gdal.Open(str(first_raster)).ReadAsArray()[0, 0]), 9)
            self.assertAlmostEqual(float(gdal.Open(str(second_raster)).ReadAsArray()[0, 0]), 18)
            with self.assertRaises(FileExistsError):
                prepare_config(config_path, output_path=first)
            with self.assertRaises(FileExistsError):
                prepare_config(config_path, output_path=config_path)
            self.assertEqual(first.read_bytes(), first_bytes)

    def test_run_definitions_and_history_are_not_git_ignored(self) -> None:
        """Both the reviewed policy and timestamped run history can be committed."""
        root = Path(__file__).resolve().parent
        paths = ["data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml",
                 "data/workflow_assets/zonal_stats/runs/example_20261015_120000.yaml"]
        result = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=root,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
