"""Tests for PAD-US population-center proximity filtering."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Point, box
from shapely.ops import transform

from filter_padus_by_population_centers import (
    CDP_LAYER,
    INCORPORATED_LAYER,
    METERS_PER_MILE,
    _buffer_geometry_locally,
    _cdp_centroid,
    filter_padus_by_population_centers,
)


class PopulationCenterBufferTest(unittest.TestCase):
    """Verify place-type-specific buffer construction and validation."""

    def test_municipal_buffer_measures_from_boundary(self) -> None:
        """Verify municipality distance is measured from its boundary."""
        crs = CRS.from_epsg(32611)
        municipality = box(500_000, 4_000_000, 530_000, 4_030_000)

        zone = _buffer_geometry_locally(
            municipality,
            crs,
            crs,
            5 * METERS_PER_MILE,
            boundary_only=True,
        )

        self.assertFalse(zone.covers(Point(515_000, 4_015_000)))
        self.assertTrue(zone.covers(Point(499_000, 4_015_000)))

    def test_cdp_centroid_uses_census_fields(self) -> None:
        """Verify a CDP center uses Census centroid fields, not geometry."""
        row = gpd.GeoDataFrame(
            {"CENTLON": [-115.25], "CENTLAT": [36.25]},
            geometry=[box(-120, 40, -119, 41)],
            crs="EPSG:4326",
        ).iloc[0]

        center = _cdp_centroid(row, CDP_LAYER)

        self.assertEqual((center.x, center.y), (-115.25, 36.25))

    def test_cdp_centroid_fields_are_required(self) -> None:
        """Verify missing Census centroid fields produce a clear error."""
        row = gpd.GeoDataFrame(
            {"GEOID": ["test"]},
            geometry=[box(-115, 36, -114, 37)],
            crs="EPSG:4326",
        ).iloc[0]

        with self.assertRaisesRegex(ValueError, "CENTLAT, CENTLON"):
            _cdp_centroid(row, CDP_LAYER)


class PopulationCenterFilterIntegrationTest(unittest.TestCase):
    """Exercise the complete GeoPackage filtering workflow."""

    def test_end_to_end_geopackage_filter(self) -> None:
        """Verify matching features are clipped, attributed, and written."""
        with TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            centers_path = temp / "centers.gpkg"
            input_path = temp / "public_lands.gpkg"
            output_path = temp / "filtered.gpkg"

            incorporated = gpd.GeoDataFrame(
                {
                    "GEOID": ["municipality"],
                    "CENTLON": [-115.0],
                    "CENTLAT": [36.0],
                },
                geometry=[box(-115.01, 35.99, -114.99, 36.01)],
                crs="EPSG:4326",
            )
            incorporated.to_file(
                centers_path,
                layer=INCORPORATED_LAYER,
                driver="GPKG",
            )
            cdps = gpd.GeoDataFrame(
                {
                    "GEOID": ["cdp"],
                    "CENTLON": [-114.5],
                    "CENTLAT": [36.0],
                },
                geometry=[box(-114.51, 35.99, -114.49, 36.01)],
                crs="EPSG:4326",
            )
            cdps.to_file(
                centers_path,
                layer=CDP_LAYER,
                driver="GPKG",
                mode="a",
            )

            project_to_utm = Transformer.from_crs(4326, 32611, always_xy=True)
            public_lands = gpd.GeoDataFrame(
                {"source_id": [1, 2, 3]},
                geometry=[
                    transform(
                        project_to_utm.transform,
                        box(-115.005, 35.995, -114.995, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-116.01, 35.99, -115.99, 36.01),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-114.505, 35.995, -114.495, 36.005),
                    ),
                ],
                crs="EPSG:32611",
            )
            public_lands.to_file(
                input_path,
                layer="public_lands",
                driver="GPKG",
            )

            result = filter_padus_by_population_centers(
                input_path=input_path,
                input_layer_name="public_lands",
                population_centers_path=centers_path,
                output_path=output_path,
            )

            output = gpd.read_file(output_path)
            self.assertEqual(result.input_features, 3)
            self.assertEqual(result.retained_features, 2)
            self.assertEqual(result.incorporated_places, 1)
            self.assertEqual(result.census_designated_places, 1)
            self.assertEqual(set(output["source_id"]), {1, 3})
            self.assertEqual(output.crs, public_lands.crs)
            self.assertTrue(output.geometry.is_valid.all())


if __name__ == "__main__":
    unittest.main()
