"""Focused routing tests for county-flattened PAD-US products."""

from pathlib import Path
import unittest

from cut_and_flatten_by_county import (
    BBB_CANDIDATE_BLM_LANDS_OUT_DIR,
    BLM_UNPROTECTED_LANDS_OUT_DIR,
    PADUS_PUBLIC_ACCESS_LANDS_OUT_DIR,
    _derive_output_names,
)


class CountyOutputRoutingTest(unittest.TestCase):
    """Verify named land products use dedicated output directories."""

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

    def test_unprotected_blm_output_routing(self) -> None:
        """Verify the intermediate BBB product gets a dedicated destination."""
        layer_name, output_path = _derive_output_names(
            Path(
                "data/processing_outputs/"
                "blm_lands_excluding_federally_protected_areas/"
                "blm_lands_excluding_federally_protected_areas_"
                "2026_08_08_19_38_52.gpkg"
            )
        )

        self.assertEqual(
            layer_name,
            "blm_lands_excluding_federally_protected_areas_by_county",
        )
        self.assertEqual(output_path.parent, BLM_UNPROTECTED_LANDS_OUT_DIR)

    def test_final_bbb_output_routing(self) -> None:
        """Verify the final BBB product gets a dedicated destination."""
        layer_name, output_path = _derive_output_names(
            Path(
                "data/processing_outputs/bbb_candidate_blm_lands/"
                "bbb_candidate_blm_lands_within_5_miles_of_"
                "population_centers_2026_08_08_19_38_52.gpkg"
            )
        )

        self.assertEqual(
            layer_name,
            "bbb_candidate_blm_lands_within_5_miles_of_"
            "population_centers_by_county",
        )
        self.assertEqual(output_path.parent, BBB_CANDIDATE_BLM_LANDS_OUT_DIR)


if __name__ == "__main__":
    unittest.main()
