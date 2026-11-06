"""Check Great Lakes exclusion and its effect on inland fishery allocation."""

from pathlib import Path
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from build_county_ecosystem_service_valuations_2026 import inland_freshwater_weights, state_values


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
                "source_layer": ["NHDWaterbody"] * 3 + ["NHDArea", "NHDWaterbody"],
                "source_fid": [11, 12, 13, 11, 14],
                "fcode": [39010, 39010, 39010, 46000, 39010],
                "name": ["Lake Michigan", "Lake Michigan", "Lake Michigan", None, "Nearby lake"],
            },
            geometry=[
                lake,
                lake,
                box(125, 0, 175, 100),
                box(180, 0, 190, 100),
                box(200, 0, 250, 100),
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


if __name__ == "__main__":
    unittest.main()
