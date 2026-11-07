"""Test state-rate reuse and Great Lakes exclusions for footprint fisheries."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon, box

from assessment_fisheries import value_fisheries
from assessment_zone_geometry import ZONE_PRODUCTS


class FootprintFisheriesTests(TestCase):
    """Compare full-county and smaller footprints in a two-county state."""

    def setUp(self):
        """Write real spatial-indexed fixture sources in a temporary directory."""
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.paths = {
            name: Path(tmp.name) / (name + extension)
            for name, extension in (
                ("coastline_lines", ".gpkg"),
                ("freshwater_polygons", ".gpkg"),
                ("great_lakes_features", ".csv"),
                ("marine_fisheries", ".csv"),
                ("inland_fisheries", ".csv"),
            )
        }
        gpd.GeoDataFrame(
            {"NAME": ["Atlantic", "Atlantic", " gREat Lakes ", "Atlantic"]},
            geometry=[
                LineString([(10, 20), (90, 20)]),
                LineString([(110, 20), (190, 20)]),
                LineString([(10, 80), (90, 80)]),
                LineString([(210, 20), (290, 20)]),
            ],
            crs=6933,
        ).to_file(self.paths["coastline_lines"], index=False)
        lake = box(0, 40, 100, 100)
        gpd.GeoDataFrame(
            {
                "source_layer": ["NHDWaterbody", "NHDWaterbody", "NHDArea", "NHDWaterbody"],
                "source_fid": [11, 12, 11, 13],
            },
            # A second copy of the lake has an unlisted FID. Removing just the
            # listed feature rather than excluding its footprint would miss it.
            geometry=[lake, lake, box(0, 0, 100, 60), box(100, 0, 200, 40)],
            crs=6933,
        ).to_file(self.paths["freshwater_polygons"], index=False)
        pd.DataFrame({"source_fid": [11]}).to_csv(self.paths["great_lakes_features"], index=False)
        pd.DataFrame(
            {"State": [" Michigan ", "Ohio", "Total"], "ncp_value": [1600, np.nan, 1600]},
        ).to_csv(self.paths["marine_fisheries"], index=False)
        pd.DataFrame(
            {"State": ["MICHIGAN"], "Consumptive use value (USD)_Intermediate assumption": [8000]},
        ).to_csv(self.paths["inland_fisheries"], index=False)
        self.zones = dict(zip(ZONE_PRODUCTS, (
            self.zone(["26001", "26003", "39001"], [box(x, 0, x + 100, 100) for x in (0, 100, 200)]),
            self.zone(["26001", "26003"], [box(0, 0, 50, 40), box(100, 0, 150, 40)]),
            self.zone(["26003", "26001", "26001"], [box(100, 0, 150, 40), box(0, 0, 30, 40), box(30, 0, 50, 40)]),
            self.zone(["26001"], [box(0, 60, 100, 100)]),
            self.zone(["26001"], [box(0, 60, 100, 100)]),
            self.zone(["39001"], [box(200, 0, 250, 40)]),
        )))

    @staticmethod
    def zone(ids, geometry):
        """Build a simple zone frame for the given identifiers and polygons."""
        return gpd.GeoDataFrame({"GEOID": ids}, geometry=geometry, crs=6933)

    def test_uses_one_state_denominator_and_excludes_great_lakes(self):
        """Conserve state totals only across whole counties, not each footprint."""
        results = value_fisheries(self.zones, self.paths, geometry_workers=2)
        self.assertEqual(set(results), set(ZONE_PRODUCTS))
        expected = {
            "counties": [[800, 4000], [800, 4000], [np.nan, np.nan]],
            "public_lands_by_county": [[400, 2000], [400, 2000]],
            "public_lands_by_county_and_manager_type": [[400, 2000], [200, 1200], [200, 800]],
            "conservation_lands_by_county": [[0, 0]],
            "conservation_lands_by_county_and_manager_type": [[0, 0]],
            "bbb_lands_by_county": [[np.nan, np.nan]],
        }
        for product, values in expected.items():
            with self.subTest(product=product):
                self.assertTrue(results[product].index.equals(self.zones[product].index))
                np.testing.assert_allclose(
                    results[product][["marine_fisheries", "inland_fisheries"]], values, equal_nan=True,
                )

    def test_fails_when_configured_lake_not_in_source(self):
        """Reject stale lake FIDs instead of treating the water as eligible."""
        pd.DataFrame({"source_fid": [11, 999]}).to_csv(self.paths["great_lakes_features"], index=False)
        with self.assertRaisesRegex(ValueError, r"Missing Great Lakes features.*999"):
            value_fisheries(self.zones, self.paths, geometry_workers=1)

    def test_repairs_zone_geometry_for_overlay_only(self):
        """Fix invalid projected polygons without changing input geometries."""
        product = "conservation_lands_by_county"
        self.zones[product] = self.zone(
            ["26001"], [Polygon([(0, 60), (100, 100), (0, 100), (100, 60)])],
        )
        result = value_fisheries(self.zones, self.paths, geometry_workers=1)
        np.testing.assert_array_equal(result[product].iloc[0], [0.0, 0.0])
        self.assertFalse(self.zones[product].geometry.iloc[0].is_valid)
