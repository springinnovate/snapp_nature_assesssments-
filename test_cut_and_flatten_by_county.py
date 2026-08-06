"""Focused routing tests for county-flattened PAD-US products."""

from pathlib import Path
import unittest

from cut_and_flatten_by_county import (
    PADUS_PUBLIC_ACCESS_LANDS_OUT_DIR,
    _derive_output_names,
)


class CountyOutputRoutingTest(unittest.TestCase):
    """Verify public-access inputs use their dedicated output directory."""

    def test_public_access_output_routing(self) -> None:
        layer_name, output_path = _derive_output_names(
            Path(
                "data/processing_outputs/padus_clipped_to_usa/"
                "public_access_lands/"
                "padus_public_access_lands_clipped_to_usa_"
                "2026_08_06_12_00_00.gpkg"
            )
        )

        self.assertEqual(
            layer_name,
            "padus_public_access_lands_clipped_by_county",
        )
        self.assertEqual(output_path.parent, PADUS_PUBLIC_ACCESS_LANDS_OUT_DIR)
        self.assertTrue(
            output_path.name.startswith(
                "padus_public_access_lands_clipped_by_county_"
            )
        )


if __name__ == "__main__":
    unittest.main()
