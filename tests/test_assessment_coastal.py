"""Test coastal values on county zones and independently overlapping masks."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point, box

from assessment.assessment_coastal import CoastalPlan, value_coastal
from assessment.assessment_zone_geometry import ZONE_PRODUCTS


class AssessmentCoastalTests(unittest.TestCase):
    """Exercise both coastal sources on small projected polygon fixtures."""

    def setUp(self) -> None:
        """Write source fixtures and define all six prepared products."""
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.paths = {name: root / f"{name}.gpkg" for name in ("corals", "mangroves")}
        crs = "EPSG:6933"
        points = gpd.GeoDataFrame(
            {"OBJECTID": [1, 2, 3, 4], "BC_USD11": [5.0, 7.0, 11.0, np.nan]},
            geometry=[Point(250, 200), Point(-200, 200), Point(1300, 200), Point(3250, 200)],
            crs=crs,
        )
        study_units = gpd.GeoDataFrame(
            {
                "ISO3": ["USA", "USA", "MEX", "PRI", "VIR"],
                "Ben_Stock_2020": [100.0, np.nan, 999.0, 999.0, 999.0],
            },
            geometry=[box(-1000, 0, 2000, 1000), box(750, 0, 1000, 1000)]
            + [box(0, 0, 1000, 1000)] * 3,
            crs=crs,
        )
        pyogrio.write_dataframe(points, self.paths["corals"], layer="aeb_coral")
        pyogrio.write_dataframe(study_units, self.paths["mangroves"], layer="UCSC_CWON_studyunits")

        self.zones = {
            "counties": gpd.GeoDataFrame(
                {"GEOID": ["01001", "01003", "01005"]},
                geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000), box(3000, 0, 4000, 1000)],
                crs=crs,
            ),
            "public_lands_by_county": gpd.GeoDataFrame(
                {"GEOID": ["01001"]}, geometry=[box(0, 0, 750, 1000)], crs=crs,
            ),
            "public_lands_by_county_and_manager_type": gpd.GeoDataFrame(
                {"GEOID": ["01001", "01001"], "Mang_Type": ["FED", "STAT"]},
                geometry=[box(0, 0, 500, 1000), box(250, 0, 750, 1000)], crs=crs,
                index=pd.RangeIndex(9, 11),
            ),
            "conservation_lands_by_county": gpd.GeoDataFrame(
                {"GEOID": ["01003"]}, geometry=[box(1000, 0, 1500, 1000)], crs=crs,
            ),
            "conservation_lands_by_county_and_manager_type": gpd.GeoDataFrame(
                {"GEOID": ["01003"], "Mang_Type": ["FED"]},
                geometry=[box(1000, 0, 1500, 1000)], crs=crs,
            ),
            "bbb_lands_by_county": gpd.GeoDataFrame(
                {"GEOID": ["01001", "01005"]},
                geometry=[box(750, 0, 1000, 1000), box(3100, 0, 3400, 1000)], crs=crs,
            ),
        }

    def test_fixed_mangrove_denominator(self) -> None:
        """Neither outside-county area nor mask sizes reset source support."""
        values = value_coastal(self.zones, self.paths, geometry_workers=2,
                               coral_reference_counties=self.zones["counties"])
        # 100 over 2,000,000 m² of county support; 1,000,000 m² outside
        # counties and study units outside USA do not contribute.
        np.testing.assert_allclose(values["counties"].mangroves, [np.nan, 50.0, 0.0])
        np.testing.assert_allclose(values["public_lands_by_county"].mangroves, [37.5])
        managers = values["public_lands_by_county_and_manager_type"]
        self.assertTrue(managers.index.equals(pd.RangeIndex(9, 11)))
        np.testing.assert_allclose(managers.mangroves, [25.0, 25.0])
        np.testing.assert_allclose(values["conservation_lands_by_county"].mangroves, [25.0])
        np.testing.assert_allclose(values["conservation_lands_by_county_and_manager_type"].mangroves, [25.0])
        # The null study unit only touches the public footprint but overlaps BBB.
        np.testing.assert_allclose(values["bbb_lands_by_county"].mangroves, [np.nan, 0.0])

    def test_offshore_corals_are_not_reassigned_to_footprints(self) -> None:
        """Whole counties retain offshore points; masks require point coverage."""
        values = value_coastal(self.zones, self.paths, geometry_workers=1,
                               coral_reference_counties=self.zones["counties"])
        np.testing.assert_allclose(values["counties"].corals, [12.0, 11.0, np.nan])
        np.testing.assert_allclose(values["public_lands_by_county"].corals, [5.0])
        # A point on the second manager's boundary counts in both masks.
        np.testing.assert_allclose(values["public_lands_by_county_and_manager_type"].corals, [5.0, 5.0])
        np.testing.assert_allclose(values["conservation_lands_by_county"].corals, [11.0])
        np.testing.assert_allclose(values["bbb_lands_by_county"].corals, [0.0, np.nan])

    def test_boundary_point_only_counts_for_its_assigned_county(self) -> None:
        """A point touching two county footprints follows the county tie-break."""
        point_path = self.paths["corals"].with_name("boundary_coral.gpkg")
        points = gpd.GeoDataFrame(
            {"OBJECTID": [1], "BC_USD11": [13.0]},
            geometry=[Point(1000, 0)], crs="EPSG:6933",
        )
        pyogrio.write_dataframe(points, point_path, layer="aeb_coral")
        self.zones["public_lands_by_county"] = self.zones["counties"].iloc[:2]
        values = value_coastal(self.zones, self.paths | {"corals": point_path}, geometry_workers=1,
                               coral_reference_counties=self.zones["counties"])
        np.testing.assert_allclose(values["counties"].corals, [13.0, 0.0, 0.0])
        np.testing.assert_allclose(values["public_lands_by_county"].corals, [13.0, 0.0])

    def test_non_conus_corals_are_excluded_without_reassignment(self) -> None:
        """Keep an offshore CONUS point but do not redistribute excluded points."""
        point_path = self.paths["corals"].with_name("scoped_coral.gpkg")
        excluded = gpd.GeoDataFrame(
            {"GEOID": ["02020", "15003", "72001", "78010"]},
            geometry=[box(x, 0, x + 1000, 1000) for x in (5000, 6000, 7000, 8000)],
            crs="EPSG:6933",
        )
        points = gpd.GeoDataFrame(
            {"OBJECTID": [1, 2, 3, 4, 5], "BC_USD11": [7.0, 20.0, 30.0, 40.0, 50.0]},
            geometry=[Point(-200, 200)] + [Point(x, 200) for x in (5250, 6250, 7250, 8250)],
            crs="EPSG:6933",
        )
        pyogrio.write_dataframe(points, point_path, layer="aeb_coral")
        reference = pd.concat([self.zones["counties"], excluded], ignore_index=True)
        values = value_coastal(self.zones, self.paths | {"corals": point_path}, geometry_workers=1,
                               coral_reference_counties=reference)
        np.testing.assert_allclose(values["counties"].corals, [7.0, 0.0, 0.0])
        for product in set(ZONE_PRODUCTS) - {"counties"}:
            self.assertTrue(values[product].corals.eq(0.0).all())

    def test_empty_footprint_returns_both_columns(self) -> None:
        """Empty products preserve their schema and index without fabricating rows."""
        product = "bbb_lands_by_county"
        self.zones[product] = self.zones[product].iloc[:0]
        values = value_coastal(self.zones, self.paths, geometry_workers=1,
                               coral_reference_counties=self.zones["counties"])
        self.assertEqual(set(values), set(ZONE_PRODUCTS))
        self.assertEqual(list(values[product].columns), ["corals", "mangroves"])
        self.assertTrue(values[product].index.equals(self.zones[product].index))

    def test_spawned_flat_tasks_match_serial_values(self) -> None:
        """Process jobs preserve support, nulls, exclusions and manager indexes."""
        # An excluded county must retain its coral assignment rather than having
        # that point redistributed to CONUS. The source fixture also has MEX,
        # PRI and VIR mangroves that must not contribute.
        excluded = gpd.GeoDataFrame({"GEOID": ["02020"]},
                                    geometry=[box(5000, 0, 6000, 1000)], crs="EPSG:6933")
        reference = pd.concat([self.zones["counties"], excluded], ignore_index=True)
        points = pyogrio.read_dataframe(self.paths["corals"], layer="aeb_coral")
        extra = gpd.GeoDataFrame({"OBJECTID": [5], "BC_USD11": [999.0]},
                                 geometry=[Point(5500, 200)], crs=points.crs)
        pyogrio.write_dataframe(pd.concat([points, extra], ignore_index=True),
                                self.paths["corals"], layer="aeb_coral")
        expected = value_coastal(self.zones, self.paths, geometry_workers=1,
                                 coral_reference_counties=reference)
        with patch("assessment.assessment_coastal.MANGROVE_CHUNK_SIZE", 1), \
                patch("assessment.assessment_coastal.FOOTPRINT_CHUNK_SIZE", 1):
            plan = CoastalPlan(self.zones, self.paths, coral_reference_counties=reference)
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                pending = {pool.submit(function, *args): key
                           for key, function, args in plan.tasks()}
                source_jobs = len(pending)
                while pending:
                    future = next(as_completed(pending))
                    key = pending.pop(future)
                    for next_key, function, args in plan.accept(key, future.result()):
                        pending[pool.submit(function, *args)] = next_key
        self.assertEqual(source_jobs, 3)  # One coral task, two USA study units.
        actual = plan.finish()
        for product in ZONE_PRODUCTS:
            pd.testing.assert_frame_equal(actual[product], expected[product])
        np.testing.assert_allclose(actual["counties"].corals, [12.0, 11.0, np.nan])
        np.testing.assert_allclose(actual["counties"].mangroves, [np.nan, 50.0, 0.0])

    def test_incomplete_plan_cannot_publish(self) -> None:
        """A source failure cannot yield a zero-filled coastal result."""
        plan = CoastalPlan(self.zones, self.paths, coral_reference_counties=self.zones["counties"])
        with self.assertRaisesRegex(ValueError, "unfinished"):
            plan.finish()

    def test_no_eligible_mangroves_produces_zeros(self) -> None:
        """Empty source selections and empty products retain their schema."""
        units = pyogrio.read_dataframe(self.paths["mangroves"], layer="UCSC_CWON_studyunits")
        units.ISO3 = "MEX"
        pyogrio.write_dataframe(units, self.paths["mangroves"], layer="UCSC_CWON_studyunits")
        self.zones["bbb_lands_by_county"] = self.zones["bbb_lands_by_county"].iloc[:0]
        values = value_coastal(self.zones, self.paths, geometry_workers=1,
                               coral_reference_counties=self.zones["counties"])
        for product in ZONE_PRODUCTS:
            self.assertTrue(values[product].mangroves.eq(0).all())
            self.assertTrue(values[product].index.equals(self.zones[product].index))


if __name__ == "__main__":
    unittest.main()
