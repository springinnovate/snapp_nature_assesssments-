"""Tests for the bill-specific PAD-US candidate-land filter."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Point, box
from shapely.ops import transform

from filter_bbb_padus_by_population_centers import (
    BBB_CANDIDATE_LAND_TYPE,
    CDP_LAYER,
    DEFAULT_OUTPUT_LAYER,
    DEFAULT_UNPROTECTED_OUTPUT_LAYER,
    FEDERALLY_PROTECTED_DESIGNATION_TYPES,
    INCORPORATED_LAYER,
    METERS_PER_MILE,
    UNPROTECTED_LAND_TYPE,
    _buffer_geometry_locally,
    _cdp_centroid,
    _open_input_layer,
    filter_bbb_padus_by_population_centers,
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


class BbbPopulationCenterFilterIntegrationTest(unittest.TestCase):
    """Exercise the complete bill-specific GeoPackage screening workflow."""

    def test_statutory_designation_codes_are_protected(self) -> None:
        """Verify PAD-US codes cover each directly mappable bill category."""
        self.assertEqual(
            set(FEDERALLY_PROTECTED_DESIGNATION_TYPES),
            {"NCA", "NM", "NP", "NRA", "NT", "NWR", "WA", "WSR"},
        )

    def test_missing_bbb_fields_are_rejected(self) -> None:
        """Verify PAD-US inputs must contain all bill-filter attributes."""
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "missing_fields.gpkg"
            gpd.GeoDataFrame(
                {"source_id": [1]},
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:32611",
            ).to_file(input_path, layer="missing_fields", driver="GPKG")

            with self.assertRaisesRegex(ValueError, "BBB filter field"):
                _open_input_layer(input_path, "missing_fields")

    def test_intermediate_and_final_outputs_must_be_distinct(self) -> None:
        """Verify the two named products cannot overwrite one another."""
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "same.gpkg"
            with self.assertRaisesRegex(ValueError, "must be different files"):
                filter_bbb_padus_by_population_centers(
                    input_path=Path("missing-input.gpkg"),
                    population_centers_path=Path("missing-centers.gpkg"),
                    unprotected_output_path=output_path,
                    output_path=output_path,
                )

    def test_end_to_end_geopackage_filter(self) -> None:
        """Verify all mappable bill filters are applied before output."""
        with TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            centers_path = temp / "centers.gpkg"
            input_path = temp / "padus_lands.gpkg"
            unprotected_output_path = temp / "unprotected_blm.gpkg"
            output_path = temp / "filtered.gpkg"

            incorporated = gpd.GeoDataFrame(
                {
                    "GEOID": ["municipality", "montana_municipality"],
                    "STATE": [4, 30],
                    "CENTLON": [-115.0, -110.0],
                    "CENTLAT": [36.0, 46.0],
                },
                geometry=[
                    box(-115.01, 35.99, -114.99, 36.01),
                    box(-110.01, 45.99, -109.99, 46.01),
                ],
                crs="EPSG:4326",
            )
            incorporated.to_file(
                centers_path,
                layer=INCORPORATED_LAYER,
                driver="GPKG",
            )
            cdps = gpd.GeoDataFrame(
                {
                    "GEOID": ["cdp", "montana_cdp"],
                    "STATE": [6, 30],
                    "CENTLON": [-114.5, -110.5],
                    "CENTLAT": [36.0, 46.0],
                },
                geometry=[
                    box(-114.51, 35.99, -114.49, 36.01),
                    box(-110.51, 45.99, -110.49, 46.01),
                ],
                crs="EPSG:4326",
            )
            cdps.to_file(
                centers_path,
                layer=CDP_LAYER,
                driver="GPKG",
                mode="a",
            )

            project_to_utm = Transformer.from_crs(4326, 32611, always_xy=True)
            padus_lands = gpd.GeoDataFrame(
                {
                    "source_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                    "land_type": ["all"] * 11,
                    "FeatClass": [
                        "Fee",
                        "Fee",
                        "Fee",
                        "Fee",
                        "Fee",
                        "Fee",
                        "Designation",
                        "Designation",
                        "Proclamation",
                        "Designation",
                        "Proclamation",
                    ],
                    "Own_Type": [
                        "FED",
                        "FED",
                        "FED",
                        "FED",
                        "FED",
                        "STAT",
                        "FED",
                        "DESG",
                        "DESG",
                        "DESG",
                        "DESG",
                    ],
                    "Mang_Name": [
                        "BLM",
                        "BLM",
                        "BLM",
                        "BLM",
                        "NPS",
                        "BLM",
                        "BLM",
                        "BLM",
                        "NPS",
                        "BLM",
                        "FWS",
                    ],
                    "State_Nm": [
                        "AZ",
                        "NV",
                        "CA",
                        "MT",
                        "AZ",
                        "AZ",
                        "AZ",
                        "AZ",
                        "CA",
                        "AZ",
                        "AZ",
                    ],
                    "Des_Tp": [
                        "PUB",
                        "PUB",
                        "PUB",
                        "PUB",
                        "FOTH",
                        "PUB",
                        "WSA",
                        "NM",
                        "PROC",
                        "WSA",
                        "PROC",
                    ],
                },
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
                    transform(
                        project_to_utm.transform,
                        box(-115.005, 35.995, -114.995, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-115.005, 35.995, -114.995, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-115.005, 35.995, -114.995, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-115.005, 35.995, -114.995, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-115.005, 35.995, -115.0, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-114.505, 35.995, -114.495, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-115.0, 35.995, -114.995, 36.005),
                    ),
                    transform(
                        project_to_utm.transform,
                        box(-113.01, 35.99, -112.99, 36.01),
                    ),
                ],
                crs="EPSG:32611",
            )
            padus_lands.to_file(
                input_path,
                layer="padus_lands",
                driver="GPKG",
            )

            result = filter_bbb_padus_by_population_centers(
                input_path=input_path,
                input_layer_name="padus_lands",
                population_centers_path=centers_path,
                unprotected_output_path=unprotected_output_path,
                output_path=output_path,
            )

            unprotected_output = gpd.read_file(unprotected_output_path)
            output = gpd.read_file(output_path)
            self.assertEqual(
                gpd.list_layers(unprotected_output_path)["name"].tolist(),
                [DEFAULT_UNPROTECTED_OUTPUT_LAYER],
            )
            self.assertEqual(
                gpd.list_layers(output_path)["name"].tolist(),
                [DEFAULT_OUTPUT_LAYER],
            )
            self.assertEqual(result.input_features, 11)
            self.assertEqual(result.bbb_candidate_features, 3)
            self.assertEqual(result.unprotected_features, 2)
            self.assertEqual(result.retained_features, 1)
            self.assertEqual(result.incorporated_places, 1)
            self.assertEqual(result.census_designated_places, 1)
            self.assertEqual(result.federally_protected_features, 3)
            self.assertEqual(set(unprotected_output["source_id"]), {1, 2})
            self.assertEqual(
                set(unprotected_output["land_type"]),
                {UNPROTECTED_LAND_TYPE},
            )
            self.assertEqual(set(output["source_id"]), {1})
            self.assertEqual(
                set(output["land_type"]),
                {BBB_CANDIDATE_LAND_TYPE},
            )
            self.assertEqual(result.stats["bbb_attribute_skipped"], 8)
            self.assertEqual(result.stats["outside_proximity_skipped"], 1)
            self.assertEqual(result.stats["federally_protected_clipped"], 1)
            self.assertEqual(
                result.stats["federally_protected_fully_excluded"],
                1,
            )
            self.assertEqual(output.crs, padus_lands.crs)
            self.assertEqual(unprotected_output.crs, padus_lands.crs)
            self.assertTrue(unprotected_output.geometry.is_valid.all())
            self.assertTrue(output.geometry.is_valid.all())
            self.assertLess(
                output.geometry.area.iloc[0],
                padus_lands.geometry.iloc[0].area,
            )


if __name__ == "__main__":
    unittest.main()
