"""Check county marine weights and the state allocation they drive."""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from assessment.build_county_ecosystem_service_valuations_2026 import (
    marine_coastline_weights,
    state_values,
)


class MarineFisheriesWeightsTest(unittest.TestCase):
    """Use one state with lake-only, marine-only, mixed, and point-touch counties."""

    def setUp(self):
        """Build county and shoreline fixtures with known lengths in metres."""
        self.ids = pd.Index(["36003", "36001", "36007", "36005"], name="GEOID")
        self.counties = gpd.GeoDataFrame(
            {"GEOID": self.ids},
            geometry=[
                box(0, 0, 1000, 1000),
                box(1000, 0, 2000, 1000),
                box(2000, 0, 4000, 1000),
                box(4000, 500, 5000, 1500),
            ],
            crs=6933,
        )
        self.coastline = gpd.GeoDataFrame(
            {"NAME": ["Great Lakes", "Atlantic", " great LAKES ", "Atlantic"]},
            geometry=[
                LineString([(100, 100), (900, 100)]),
                LineString([(1100, 100), (1900, 100)]),
                LineString([(2100, 750), (4000, 750)]),
                LineString([(2600, 500), (3800, 500)]),
            ],
            crs=6933,
        )
        # Deliberately differ from the measured marine-only and point-touch
        # lengths to detect an unintended recalculation of their weights.
        self.baseline = pd.Series([0.8, 0.6, 3.1, 0.6], index=self.ids).sort_index()

    def test_replaces_only_weights_with_positive_lake_overlap(self):
        """Keep baseline weights for marine-only and lake point-touch counties."""
        weights = marine_coastline_weights(self.counties, self.coastline, self.baseline)
        self.assertTrue(weights.index.is_unique)
        self.assertEqual(set(weights.index), set(self.ids))
        np.testing.assert_allclose(weights.reindex(self.ids), [0.0, 0.6, 1.2, 0.6])
        np.testing.assert_allclose(self.baseline.reindex(self.ids), [0.8, 0.6, 3.1, 0.6])

    def test_state_value_is_allocated_only_to_marine_coast(self):
        """Redistribute the entire state total without giving lake counties value."""
        weights = marine_coastline_weights(self.counties, self.coastline, self.baseline)
        source = pd.DataFrame({"State": ["NEW YORK"], "ncp_value": [1000.0]})
        states = pd.Series("36", index=self.ids)
        values, status, checks = state_values(source, "ncp_value", states, weights)
        np.testing.assert_allclose(values, [0.0, 250.0, 500.0, 250.0])
        self.assertEqual(status.tolist(), ["zero_weight", "state_weighted", "state_weighted", "state_weighted"])
        self.assertAlmostEqual(checks[0]["allocated_value"], checks[0]["input_value"])


if __name__ == "__main__":
    unittest.main()
