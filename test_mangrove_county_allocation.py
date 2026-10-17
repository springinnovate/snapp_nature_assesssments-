"""Tests for county-covered-area mangrove allocation and source conservation."""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box

from mangrove_county_allocation import allocate_mangrove_values


class MangroveAllocationTests(unittest.TestCase):
    """Exercise allocation, incomplete data, geometry, and input validation."""

    def setUp(self):
        """Prepare metre-based counties and a 30/10/60 county/ocean study unit."""
        self.counties = gpd.GeoDataFrame({"GEOID": ["01001", "01003", "01005"]},
                                       geometry=[box(0, 0, 3, 10), box(3, 0, 4, 10), box(20, 0, 30, 10)], crs=6933)
        self.sources = gpd.GeoDataFrame({"source_id": ["shared"], "Ben_Stock_2020": [1000.0],
                                        "ISO3": ["USA"], "Country": ["United States"]},
                                       geometry=[box(0, 0, 10, 10)], crs=6933)

    def test_ocean_excluded_from_denominator_and_audit_conserves(self):
        """Thirty and ten square metres split a total 75/25, not 30/10."""
        values, status, audit = allocate_mangrove_values(self.counties, self.sources)
        self.assertEqual(values.tolist(), [750, 250, 0])
        self.assertEqual(status.tolist(), ["allocated", "allocated", "no_intersection"])
        row = audit.iloc[0]
        self.assertEqual(row.source_area_m2, 100)
        self.assertEqual(row.intersected_area_m2, 40)
        self.assertEqual(row.intersecting_counties, 2)
        self.assertEqual(row.allocated_fraction, 1)
        self.assertEqual(row.allocated_value, 1000)
        self.assertEqual(row.unallocated_value, 0)
        self.assertEqual(row.ISO3, "USA")
        self.assertEqual(row.Country, "United States")
        self.assertEqual(row.allocated_value + row.unallocated_value, row.input_value)

    def test_unique_partial_intersection_receives_whole_value_and_sources_add(self):
        """Exclusive and shared contributions sum without losing uncovered area."""
        sources = gpd.GeoDataFrame({"source_id": ["shared", "solo", "other"],
                                   "Ben_Stock_2020": [1000.0, 200.0, 50.0]},
                                  geometry=[box(0, 0, 10, 10), box(-10, 0, 2, 10), box(0, 0, 1, 1)], crs=6933)
        values, _, audit = allocate_mangrove_values(self.counties, sources)
        self.assertEqual(values.tolist(), [1000, 250, 0])
        self.assertEqual(audit.allocated_value.sum(), sources.Ben_Stock_2020.sum())
        self.assertEqual(audit.iloc[1].source_area_m2, 120)
        self.assertEqual(audit.iloc[1].intersected_area_m2, 20)
        self.assertEqual(audit.iloc[1].allocated_value, 200)

    def test_missing_values_withhold_only_intersecting_counties(self):
        """Unknown intersecting sources cannot silently produce partial totals."""
        sources = gpd.GeoDataFrame({"source_id": ["known", "unknown", "offshore"],
                                   "Ben_Stock_2020": [1000.0, np.nan, np.nan]},
                                  geometry=[box(0, 0, 10, 10), box(0, 0, 2, 2), box(100, 0, 101, 1)], crs=6933)
        values, status, audit = allocate_mangrove_values(self.counties, sources)
        self.assertTrue(np.isnan(values.iloc[0]))
        self.assertEqual(values.iloc[1:].tolist(), [250, 0])
        self.assertEqual(status.tolist(), ["incomplete_source", "allocated", "no_intersection"])
        self.assertEqual(audit.status.tolist(), ["allocated", "missing_value", "missing_value"])
        self.assertEqual(audit.allocated_fraction.tolist(), [1, 1, 0])
        self.assertTrue(audit.iloc[1:].allocated_value.isna().all())
        self.assertEqual(audit.iloc[0].allocated_value, 1000)

    def test_no_intersection_boundary_touch_and_empty_geometry_are_audited(self):
        """Zero-area touches and unusable polygons allocate no monetary value."""
        sources = gpd.GeoDataFrame({"source_id": ["offshore", "touch", "empty"],
                                   "Ben_Stock_2020": [80.0, 20.0, 10.0]},
                                  geometry=[box(100, 0, 101, 1), box(4, 0, 5, 10), None], crs=6933)
        values, _, audit = allocate_mangrove_values(self.counties, sources)
        self.assertEqual(values.tolist(), [0, 0, 0])
        self.assertEqual(audit.allocated_value.sum(), 0)
        self.assertEqual(audit.unallocated_value.sum(), 110)
        self.assertTrue(audit.intersecting_counties.eq(0).all())
        self.assertEqual(audit.reason.tolist(), ["no_positive_county_intersection",
                                               "no_positive_county_intersection", "invalid_or_empty_geometry"])

    def test_empty_sources_and_empty_counties(self):
        """An empty input produces a correctly indexed result and a usable audit."""
        values, status, audit = allocate_mangrove_values(self.counties, self.sources.iloc[:0])
        self.assertEqual(values.tolist(), [0, 0, 0])
        self.assertTrue(status.eq("no_intersection").all())
        self.assertTrue(audit.empty)
        self.assertIn("source_key", audit)
        values, status, audit = allocate_mangrove_values(self.counties.iloc[:0], self.sources)
        self.assertTrue(values.empty)
        self.assertTrue(status.empty)
        self.assertEqual(audit.iloc[0].unallocated_value, 1000)

    def test_invalid_geometry_repaired_and_geographic_crs_supported(self):
        """Self-intersections are repaired and coordinates are projected before area."""
        invalid = self.sources.copy()
        invalid.geometry = [Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])]
        values, _, audit = allocate_mangrove_values(self.counties, invalid)
        self.assertEqual(values.tolist(), [1000, 0, 0])
        self.assertEqual(audit.iloc[0].source_area_m2, 2)
        projected = allocate_mangrove_values(self.counties.to_crs(4326), self.sources.to_crs(4326))[0]
        np.testing.assert_allclose(projected, [750, 250, 0], atol=0.001)

    def test_worker_count_and_chunk_size_do_not_change_result(self):
        """Thread scheduling and batching preserve contribution and audit order."""
        sources = gpd.GeoDataFrame({"source_id": list(range(11)), "Ben_Stock_2020": [1e9, 0.1, 0.2, 5, 0, 17, 29, 41, 1e-6, 3, 7]},
                                  geometry=[box(0, 0, 10, 10)] * 11, crs=6933)
        sequential = allocate_mangrove_values(self.counties, sources, workers=1, chunk_size=4)
        concurrent = allocate_mangrove_values(self.counties, sources, workers=3, chunk_size=2)
        for left, right in zip(sequential[:2], concurrent[:2]):
            pd.testing.assert_series_equal(left, right)
        pd.testing.assert_frame_equal(sequential[2], concurrent[2])
        self.assertAlmostEqual(sequential[0].sum(), sources.Ben_Stock_2020.sum(), places=6)

    def test_identifiers_values_crs_and_parallel_options_are_validated(self):
        """Ambiguous identifiers and malformed inputs fail rather than drop data."""
        for frame_name, field in (("counties", "GEOID"), ("sources", "source_id")):
            original = getattr(self, frame_name)
            duplicate = pd.concat([original, original.iloc[:1]], ignore_index=True)
            missing = original.copy()
            missing.loc[missing.index[0], field] = None
            blank = original.copy()
            blank.loc[blank.index[0], field] = " "
            for invalid in (duplicate, missing, blank, original.drop(columns=field)):
                with self.subTest(frame=frame_name), self.assertRaises(ValueError):
                    allocate_mangrove_values(invalid if frame_name == "counties" else self.counties,
                                             invalid if frame_name == "sources" else self.sources)
        for frame_name in ("counties", "sources"):
            invalid = getattr(self, frame_name).copy()
            invalid.crs = None
            with self.assertRaisesRegex(ValueError, "CRS"):
                allocate_mangrove_values(invalid if frame_name == "counties" else self.counties,
                                         invalid if frame_name == "sources" else self.sources)
        for value in (np.inf, -np.inf, "not a value"):
            source = self.sources.copy()
            source["Ben_Stock_2020"] = [value]
            with self.assertRaises(ValueError):
                allocate_mangrove_values(self.counties, source)
        for setting in ("workers", "chunk_size"):
            for value in (0, -1, 1.5, True):
                with self.subTest(setting=setting, value=value), self.assertRaises(ValueError):
                    allocate_mangrove_values(self.counties, self.sources, **{setting: value})
        with self.assertRaisesRegex(ValueError, "value field"):
            allocate_mangrove_values(self.counties, self.sources.drop(columns="Ben_Stock_2020"))
        invalid_counties = self.counties.copy()
        invalid_counties.loc[0, "geometry"] = None
        with self.assertRaisesRegex(ValueError, "positive-area geometry"):
            allocate_mangrove_values(invalid_counties, self.sources)


if __name__ == "__main__":
    unittest.main()
