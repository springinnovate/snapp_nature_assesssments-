"""Test state-rate reuse and lake exclusions for footprint fisheries."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon, box

from assessment.assessment_fisheries import read_marine_shoreline, value_fisheries
from assessment.assessment_zone_geometry import ZONE_PRODUCTS


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
                ("inland_fisheries_exclusions", ".csv"),
                ("marine_fisheries", ".csv"),
                ("inland_fisheries", ".csv"),
            )
        }
        gpd.GeoDataFrame(
            {"NAME": ["Atlantic", "Atlantic", " gREat Lakes ", "Atlantic", "Atlantic", "Atlantic"]},
            geometry=[
                LineString([(10, 20), (90, 20)]),
                LineString([(110, 20), (190, 20)]),
                LineString([(10, 80), (90, 80)]),
                LineString([(210, 20), (290, 20)]),
                LineString([(310, 80), (390, 80)]),
                LineString([(410, 20), (490, 20)]),
            ],
            crs=6933,
        ).to_file(self.paths["coastline_lines"], index=False)
        lake = box(0, 40, 100, 100)
        salt_lake = box(300, 20, 400, 100)
        gpd.GeoDataFrame(
            {
                "source_layer": ["NHDWaterbody", "NHDWaterbody", "NHDArea", "NHDWaterbody",
                                 "NHDWaterbody", "NHDWaterbody", "NHDArea", "NHDWaterbody"],
                "source_fid": [11, 12, 11, 13, 14, 15, 14, 16],
            },
            # A second copy of the lake has an unlisted FID. Removing just the
            # listed feature rather than excluding its footprint would miss it.
            geometry=[lake, lake, box(0, 0, 100, 60), box(100, 0, 200, 40),
                      salt_lake, salt_lake, box(300, 0, 400, 40), box(400, 0, 500, 40)],
            crs=6933,
        ).to_file(self.paths["freshwater_polygons"], index=False)
        pd.DataFrame({"source_fid": [11]}).to_csv(self.paths["great_lakes_features"], index=False)
        pd.DataFrame({"source_fid": [14]}).to_csv(
            self.paths["inland_fisheries_exclusions"], index=False,
        )
        pd.DataFrame(
            {
                "State": [" Michigan ", "Ohio", "Alaska", "Hawaii", "Guam", "Total"],
                "ncp_value": [1600, np.nan, 1000, 1200, 800, 4600],
            },
        ).to_csv(self.paths["marine_fisheries"], index=False)
        pd.DataFrame(
            {
                "State": ["MICHIGAN", "Puerto Rico", "American Samoa"],
                "Consumptive use value (USD)_Intermediate assumption": [8000, 1000, 200],
            },
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
        """Conserve CONUS state totals across whole counties, not each footprint."""
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

    def test_requires_support_for_each_supplied_conus_state(self):
        """Exclude non-CONUS values but fail when a CONUS total lacks support."""
        pd.DataFrame(
            {"State": ["Michigan", "Texas", "Alaska"], "ncp_value": [1600, 1000, 1000]},
        ).to_csv(self.paths["marine_fisheries"], index=False)
        with self.assertRaisesRegex(ValueError, "No allocation support for marine_fisheries states:.*48"):
            value_fisheries(self.zones, self.paths, geometry_workers=1)

    def test_redistributes_utah_inland_only_across_all_products(self):
        """Conserve Utah's total using eligible water and leave marine values alone."""
        national_before = value_fisheries(self.zones, self.paths, geometry_workers=1)
        for service, field, value in (
            ("inland_fisheries", "Consumptive use value (USD)_Intermediate assumption", 6000),
            ("marine_fisheries", "ncp_value", 1200),
        ):
            path = self.paths[service]
            totals = pd.read_csv(path)
            totals.loc[len(totals)] = {"State": "Utah", field: value}
            totals.to_csv(path, index=False)

        lake_only = self.zone(["49011"], [box(300, 60, 400, 100)])
        utah = dict(zip(ZONE_PRODUCTS, (
            self.zone(["49011", "49057"], [box(300, 0, 400, 100), box(400, 0, 500, 100)]),
            self.zone(["49011"], [box(300, 0, 350, 20)]),
            self.zone(["49011", "49011"], [box(300, 0, 325, 20), box(325, 0, 350, 20)]),
            lake_only,
            lake_only,
            lake_only,
        )))
        zones = {product: pd.concat([frame, utah[product]], ignore_index=True)
                 for product, frame in self.zones.items()}
        expected = (
            [[600, 2000], [600, 4000]],
            [[0, 1000]],
            [[0, 500], [0, 500]],
            [[600, 0]],
            [[600, 0]],
            [[600, 0]],
        )
        results = value_fisheries(zones, self.paths, geometry_workers=2)
        for product, utah_values in zip(ZONE_PRODUCTS, expected):
            with self.subTest(product=product):
                offset = len(self.zones[product])
                pd.testing.assert_frame_equal(results[product].iloc[:offset], national_before[product])
                np.testing.assert_allclose(results[product].iloc[offset:], utah_values)
        np.testing.assert_allclose(
            results["counties"].iloc[-2:]["inland_fisheries"].sum(), 6000,
        )

    def test_rejects_non_conus_county_geometries(self):
        """Do not compute state rates from a mixed-domain county layer."""
        self.zones["counties"].loc[0, "GEOID"] = "02001"
        with self.assertRaisesRegex(ValueError, "out-of-scope county state FIPS.*02"):
            value_fisheries(self.zones, self.paths, geometry_workers=1)

    def test_marine_source_is_clipped_before_projection(self):
        """Drop remote multipart pieces but retain shoreline crossing the envelope."""
        path = self.paths["coastline_lines"].with_name("geographic_coastline.gpkg")
        local = LineString([(-123, 31), (-120, 39)])
        gpd.GeoDataFrame(
            {"NAME": ["Pacific", "Pacific", "Great Lakes"]},
            geometry=[
                MultiLineString([local, LineString([(179, 52), (-179, 52)])]),
                MultiLineString([LineString([(-126, 35), (-124, 35)])]),
                MultiLineString([LineString([(-123, 36), (-120, 36)])]),
            ],
            crs=4269,
        ).to_file(path, index=False)
        counties = gpd.GeoDataFrame(
            {"GEOID": ["06001"]}, geometry=[box(-125, 30, -118, 42)], crs=4269,
        )
        tree = read_marine_shoreline(path, counties)
        expected = gpd.GeoSeries(
            [local, LineString([(-125, 35), (-124, 35)])], crs=4269,
        ).to_crs(6933).length.to_numpy()
        np.testing.assert_allclose(
            sorted(geometry.length for geometry in tree.geometries), sorted(expected),
        )

    def test_fails_when_configured_lake_not_in_source(self):
        """Reject stale lake FIDs instead of treating the water as eligible."""
        pd.DataFrame({"source_fid": [11, 999]}).to_csv(self.paths["great_lakes_features"], index=False)
        with self.assertRaisesRegex(ValueError, r"Missing Great Lakes features.*999"):
            value_fisheries(self.zones, self.paths, geometry_workers=1)

    def test_fails_when_configured_great_salt_lake_not_in_source(self):
        """Reject stale Utah identifiers without silently restoring the lake weight."""
        pd.DataFrame({"source_fid": [14, 999]}).to_csv(
            self.paths["inland_fisheries_exclusions"], index=False,
        )
        with self.assertRaisesRegex(ValueError, r"Missing Great Salt Lake features.*999"):
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
