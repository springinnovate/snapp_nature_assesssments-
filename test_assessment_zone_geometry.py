"""Focused tests for six-zone geometry and county land-area accounting."""

from unittest import TestCase

import shapely
from shapely.geometry import box

from assessment_zone_geometry import ZONE_PRODUCTS, build_county_zones


class CountyZoneGeometryTests(TestCase):
    """Exercise group dissolves, land masks and sparse county outputs."""

    def test_grouped_geometry_and_land_areas(self):
        """Count each piece once per footprint and exclude water from ratios."""
        county = box(0, 0, 200, 100)
        land = [box(-50, -50, 80, 150), box(120, 0, 200, 100)]
        padus = [
            (box(-10, 0, 80, 100), "FED", True, True),
            (box(30, 0, 100, 100), "FED", True, False),
            (box(100, 0, 170, 100), "STAT", True, False),
            (box(170, 0, 205, 100), "PVT", False, True),
            (box(200, 0, 205, 100), "UNK", True, True),
        ]
        bbb = [box(60, 0, 130, 100), box(70, 0, 100, 100)]
        result = build_county_zones("01001", county, padus, land, bbb)
        self.assertEqual(set(result), set(ZONE_PRODUCTS))
        county_row, = result["counties"]
        self.assertEqual(county_row["GEOID"], "01001")
        self.assertAlmostEqual(county_row["area_ha_county_land"], 1.6)
        for name, area, share, managers in (
            ("public", 1.3, 81.25, ["FED", "STAT"]),
            ("conservation", 1.1, 68.75, ["FED", "PVT"]),
        ):
            with self.subTest(footprint=name):
                footprint, = result[f"{name}_lands_by_county"]
                groups = result[f"{name}_lands_by_county_and_manager_type"]
                self.assertEqual([row["Mang_Type"] for row in groups], managers)
                self.assertTrue(footprint["geometry"].equals(shapely.union_all(
                    [row["geometry"] for row in groups]
                )))
                self.assertAlmostEqual(footprint["area_ha_land"], area)
                self.assertAlmostEqual(footprint["county_land_share"], share / 100)
                self.assertAlmostEqual(county_row[f"area_ha_{name}_lands"], area)
                self.assertAlmostEqual(county_row[f"pct_{name}_lands"], share)
                self.assertEqual(county_row[f"area_ha_{name}_manager_overlap"], 0.0)
        public, = result["public_lands_by_county"]
        self.assertEqual(public["geometry"].area, 17_000)
        self.assertAlmostEqual(county_row["area_ha_bbb_lands"], 0.3)
        self.assertAlmostEqual(county_row["pct_bbb_lands"], 18.75)

    def test_manager_overlap_audited_without_assignment(self):
        """Keep both manager labels and report overlap in land and water."""
        county = box(0, 0, 100, 100)
        land = [box(0, 0, 40, 100), box(50, 0, 100, 100)]
        padus = [
            (box(0, 0, 70, 100), "FED", True, True),
            (box(40, 0, 100, 100), "STAT", True, True),
        ]
        result = build_county_zones("01001", county, padus, land, [])
        row, = result["counties"]
        for name in ("public", "conservation"):
            self.assertAlmostEqual(row[f"area_ha_{name}_manager_overlap"], 0.3)
            self.assertAlmostEqual(row[f"area_ha_{name}_manager_land_overlap"], 0.2)
            self.assertAlmostEqual(row[f"pct_{name}_lands"], 100)
        self.assertFalse(result["bbb_lands_by_county"])
        self.assertEqual(row["pct_bbb_lands"], 0.0)

    def test_water_only_footprint_has_zero_share(self):
        """Keep water in the zone geometry but not in its land measure."""
        county = box(0, 0, 100, 100)
        land = [box(0, 0, 50, 100)]
        water = box(50, 0, 100, 100)
        result = build_county_zones("01001", county, [], land, [water])
        row, = result["bbb_lands_by_county"]
        self.assertTrue(row["geometry"].equals(water))
        self.assertEqual(row["area_ha_land"], 0.0)
        self.assertEqual(row["county_land_share"], 0.0)

    def test_missing_land_mask_coverage_fails(self):
        """Reject an undefined denominator instead of emitting zero or NaN."""
        county = box(0, 0, 100, 100)
        with self.assertRaisesRegex(ValueError, "01001.*no positive land-mask coverage"):
            build_county_zones("01001", county, [], [box(100, 0, 200, 100)], [])
