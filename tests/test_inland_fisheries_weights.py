"""Check configured lake exclusions and their effect on inland fishery allocation."""

from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from assessment.build_county_ecosystem_service_valuations_2026 import compute_task, inland_freshwater_weights, state_values


class InlandFisheriesWeightsTest(unittest.TestCase):
    """Use a mixed state with lake-only, mixed, and unaffected counties."""

    def setUp(self):
        """Write a prepared freshwater fixture with duplicate lake polygons."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.freshwater = Path(self.temporary.name) / "freshwater.gpkg"
        self.ids = pd.Index(["26003", "26001", "26005", "26007"], name="GEOID")
        self.counties = gpd.GeoDataFrame(
            {"GEOID": self.ids},
            geometry=[
                box(0, 0, 100, 100),
                box(100, 0, 200, 100),
                box(200, 0, 300, 100),
                box(50, 100, 100, 200),
            ],
            crs=6933,
        )
        lake = box(0, 0, 150, 100)
        gpd.GeoDataFrame(
            {
                "source_layer": ["NHDWaterbody"] * 3 + ["NHDArea"] + ["NHDWaterbody"] * 4,
                "source_fid": [11, 12, 13, 11, 14, 21, 22, 23],
            },
            geometry=[
                lake,
                lake,
                box(125, 0, 175, 100),
                box(180, 0, 190, 100),
                box(200, 0, 250, 100),
                box(10000, 0, 10140, 120),
                box(10130, 0, 10150, 100),
                box(10125, 0, 10175, 100),
            ],
            crs=6933,
        ).to_file(self.freshwater, layer="nhd_freshwater_clipped_to_usa", driver="GPKG")
        # The unaffected baselines intentionally differ from fresh measurements.
        self.baseline = pd.Series([2.0, 1.6, 1.05, 0.35], index=self.ids).sort_index()

    def test_excludes_only_great_lake_footprint(self):
        """Remove all lake copies, trim overlap, and keep unaffected weights."""
        weights = inland_freshwater_weights(self.counties, self.freshwater, [11, 12], self.baseline)
        self.assertTrue(weights.index.is_unique)
        self.assertEqual(set(weights.index), set(self.ids))
        # In the mixed county retain 0.25 ha of the unrelated named polygon
        # outside the lake and 0.10 ha of NHDArea with the same source FID.
        np.testing.assert_allclose(weights.reindex(self.ids), [0.0, 0.35, 1.05, 0.35])
        np.testing.assert_allclose(self.baseline.reindex(self.ids), [2.0, 1.6, 1.05, 0.35])

    def test_preserves_entire_state_value(self):
        """Reallocate only by inland freshwater without dropping dollars."""
        weights = inland_freshwater_weights(self.counties, self.freshwater, [11, 12], self.baseline)
        source = pd.DataFrame({"State": ["MICHIGAN"], "value": [1000.0]})
        values, status, checks = state_values(source, "value", pd.Series("26", index=self.ids), weights)
        np.testing.assert_allclose(values, [0.0, 200.0, 600.0, 200.0])
        self.assertEqual(status.loc["26003"], "zero_weight")
        self.assertAlmostEqual(checks[0]["allocated_value"], checks[0]["input_value"])

    def test_builder_reallocates_utah_and_preserves_great_lakes_correction(self):
        """Read both configured selections and redistribute only within each state."""
        root = Path(self.temporary.name)
        extra_ids = pd.Index(["49001", "49003", "49005", "16001", "16003"], name="GEOID")
        extra_counties = gpd.GeoDataFrame(
            {"GEOID": extra_ids}, geometry=[
                box(10000, 0, 10100, 100), box(10100, 0, 10200, 100), box(10200, 0, 10300, 100),
                box(10000, 100, 10100, 200), box(10200, 100, 10300, 200),
            ], crs=6933,
        )
        counties = pd.concat([self.counties, extra_counties], ignore_index=True)
        baseline = pd.concat([self.baseline, pd.Series([4.0, 4.0, 1.0, 1.5, 4.5], index=extra_ids)])
        paths = {
            "base": root / "counties.gpkg", "freshwater": root / "baseline.csv", "freshwater_polygons": self.freshwater,
            "great_lakes_features": root / "great_lakes.csv", "inland_fisheries_exclusions": root / "salt_lake.csv",
            "inland_fisheries": root / "harvest.csv",
        }
        counties.to_file(paths["base"], layer="counties", driver="GPKG")
        baseline.rename("intersect_area_ha_water").to_csv(paths["freshwater"])
        pd.DataFrame({"source_fid": [11, 12]}).to_csv(paths["great_lakes_features"], index=False)
        pd.DataFrame({"source_fid": [21, 22]}).to_csv(paths["inland_fisheries_exclusions"], index=False)
        pd.DataFrame({
            "State": ["MICHIGAN", "UTAH", "IDAHO"],
            "Consumptive use value (USD)_Intermediate assumption": [1000.0, 800.0, 600.0],
        }).to_csv(paths["inland_fisheries"], index=False)
        ids = self.ids.append(extra_ids)
        base = pd.DataFrame({"STATEFP": ids.str[:2]}, index=ids)

        result = compute_task("inland_fisheries", paths, base, pd.DataFrame(), 1)

        # The Idaho county touches the test lake across the state line but the
        # Utah exclusion does not change Idaho's supplied weights.
        np.testing.assert_allclose(result["originals"]["inland_fisheries"],
                                   [0.0, 200.0, 600.0, 200.0, 0.0, 160.0, 640.0, 150.0, 450.0])
        for check in result["checks"]:
            self.assertAlmostEqual(check["allocated_value"], check["input_value"])
        self.assertEqual(len(result["checks"]), 3)
        self.assertIn("Great Salt Lake", result["metadata"][0]["method"])

    def test_rejects_unknown_configured_lake_fid(self):
        """Do not publish an allocation from an incomplete selection."""
        with self.assertRaisesRegex(ValueError, "Missing configured inland fisheries exclusions.*999"):
            inland_freshwater_weights(self.counties, self.freshwater, [11, 12], self.baseline,
                                      great_salt_lake_fids=[999])


if __name__ == "__main__":
    unittest.main()
