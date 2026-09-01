"""Focused tests for explicit TIFF zonal-statistics config preparation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import fiona
import numpy as np
from osgeo import gdal, osr

from prepare_tif_zonal_stats_config import (
    RasterInput,
    discover_zone_jobs,
    load_and_prepare_rasters,
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

            manifest_path = root / "rasters.toml"
            manifest_path.write_text(
                """
[[rasters]]
label = "Pollination"
filename = "pollination.tif"
units = "USD per ha"
include = true
scale = 0.09
derived_filename = "pollination_per_pixel.vrt"

[[rasters]]
label = "Missing pending raster"
filename = "pending.tif"
units = "USD"
include = false
reason = "not delivered"
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
                [RasterInput("Example", raster_path, "USD per pixel")],
                jobs,
            )
            config_text = config_path.read_text(encoding="utf-8")
            self.assertIn("2026_08_09_01_00_00.gpkg", config_text)
            self.assertNotIn("2026_08_07_01_00_00.gpkg", config_text)
            self.assertIn("[job:pad_ecosystem_services]", config_text)
            self.assertIn("max_simplify_tolerance_meters = 15", config_text)


if __name__ == "__main__":
    unittest.main()
