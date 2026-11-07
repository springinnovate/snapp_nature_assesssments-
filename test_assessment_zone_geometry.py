"""Focused tests for six-zone geometry and county land-area accounting."""

from unittest import TestCase

import shapely
from shapely.geometry import box

from assessment_zone_geometry import ZONE_PRODUCTS, build_county_zones


class CountyZoneGeometryTests(TestCase):
    """Exercise group dissolves, ALAND ratios and sparse county outputs."""

    def test_grouped_geometry_and_land_areas(self):
        """Use unioned geometry for each numerator and ALAND for each denominator."""
        county = box(0, 0, 200, 100)
        aland = 18_000
        padus = [
            (box(-10, 0, 80, 100), "FED", True, True),
            (box(30, 0, 100, 100), "FED", True, False),
            (box(100, 0, 170, 100), "STAT", True, False),
            (box(170, 0, 205, 100), "PVT", False, True),
            (box(200, 0, 205, 100), "UNK", True, True),
        ]
        bbb = [box(60, 0, 130, 100), box(70, 0, 100, 100)]
        result = build_county_zones("01001", county, aland, padus, bbb)
        self.assertEqual(set(result), set(ZONE_PRODUCTS))
        county_row, = result["counties"]
        self.assertEqual(county_row["GEOID"], "01001")
        self.assertAlmostEqual(county_row["area_ha_county_land"], 1.8)
        self.assertTrue(county_row["geometry"].equals(county))
        for name, area, managers in (
            ("public", 1.7, ["FED", "STAT"]),
            ("conservation", 1.1, ["FED", "PVT"]),
        ):
            with self.subTest(footprint=name):
                footprint, = result[f"{name}_lands_by_county"]
                groups = result[f"{name}_lands_by_county_and_manager_type"]
                self.assertEqual([row["Mang_Type"] for row in groups], managers)
                self.assertTrue(footprint["geometry"].equals(shapely.union_all(
                    [row["geometry"] for row in groups]
                )))
                self.assertAlmostEqual(footprint["area_ha_land"], area)
                self.assertAlmostEqual(footprint["county_land_share"], area / 1.8)
                self.assertAlmostEqual(county_row[f"area_ha_{name}_lands"], area)
                self.assertAlmostEqual(county_row[f"pct_{name}_lands"], 100 * area / 1.8)
                self.assertEqual(county_row[f"area_ha_{name}_manager_overlap"], 0.0)
        public, = result["public_lands_by_county"]
        self.assertEqual(public["geometry"].area, 17_000)
        self.assertAlmostEqual(county_row["area_ha_bbb_lands"], 0.7)
        self.assertAlmostEqual(county_row["pct_bbb_lands"], 100 * 0.7 / 1.8)

    def test_manager_overlap_audited_without_assignment(self):
        """Keep both manager labels and report overlap without assigning it."""
        county = box(0, 0, 100, 100)
        padus = [
            (box(0, 0, 70, 100), "FED", True, True),
            (box(40, 0, 100, 100), "STAT", True, True),
        ]
        result = build_county_zones("01001", county, 10_000, padus, [])
        row, = result["counties"]
        for name in ("public", "conservation"):
            self.assertAlmostEqual(row[f"area_ha_{name}_manager_overlap"], 0.3)
            self.assertAlmostEqual(row[f"pct_{name}_lands"], 100)
        self.assertFalse(result["bbb_lands_by_county"])
        self.assertEqual(row["pct_bbb_lands"], 0.0)

    def test_small_rounding_excess_bounds_only_the_share(self):
        """Retain measured area while rounding a very small excess to 100%."""
        county = box(0, 0, 100, 100)
        result = build_county_zones("01001", county, 9_999.5, [], [county])
        row, = result["bbb_lands_by_county"]
        self.assertTrue(row["geometry"].equals(county))
        self.assertEqual(row["area_ha_land"], 1.0)
        self.assertEqual(row["county_land_share"], 1.0)
        self.assertEqual(result["counties"][0]["pct_bbb_lands"], 100.0)

    def test_union_materially_exceeding_aland_fails(self):
        """Check the dissolved footprint, even when each manager fits."""
        county = box(0, 0, 100, 100)
        padus = [
            (box(0, 0, 40, 100), "FED", True, False),
            (box(40, 0, 100, 100), "STAT", True, False),
        ]
        with self.assertRaisesRegex(ValueError, "01001.*public_lands_by_county.*exceeds ALAND"):
            build_county_zones("01001", county, 9_000, padus, [])

    def test_missing_footprints_are_sparse(self):
        """A county without a footprint keeps a county row and zero shares."""
        county = box(0, 0, 100, 100)
        result = build_county_zones("01001", county, 8_000, [], [])
        row, = result["counties"]
        for product in ZONE_PRODUCTS[1:]:
            self.assertEqual(result[product], [])
        for name in ("public", "conservation", "bbb"):
            self.assertEqual(row[f"area_ha_{name}_lands"], 0.0)
            self.assertEqual(row[f"pct_{name}_lands"], 0.0)

    def test_invalid_aland_fails(self):
        """Reject an undefined denominator instead of emitting zero or NaN."""
        county = box(0, 0, 100, 100)
        for aland in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(aland=aland):
                with self.assertRaisesRegex(ValueError, "01001.*finite positive ALAND"):
                    build_county_zones("01001", county, aland, [], [])
