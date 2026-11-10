"""Exercise PAD-US product selection against a domain-less vector source."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import box

from padus_land_rules import RuleConfigError
import prepare_padus_all_and_public_lands as prepare


class PreparePadusProductsTests(unittest.TestCase):
    """Validate and clip a small VectorAnalysis-like source without owner fields."""

    def setUp(self):
        """Write a projected source and boundary to a temporary test directory."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "Vector Analysis.gpkg"
        self.layer = "vector_analysis"
        self.boundary = self.root / "boundary.gpkg"
        self.output = self.root / "products"
        self.rules = Path(__file__).resolve().parents[1] / "config" / "padus_land_rules.rules"
        gpd.GeoDataFrame(
            {
                "Unit_Nm": ["federal", "unknown_park", "ngo", "unlisted", "outside"],
                "Mang_Type": ["FED", "UNK", "NGO", "UNLISTED", "LOC"],
                "Mang_Name": ["BLM", "UNK", "NGO", "UNK", "UNK"],
                "Des_Tp": ["CONE", "SP", "SP", "SP", "LP"],
                "Pub_Access": ["OA", "OA", "OA", "OA", "OA"],
            },
            geometry=[box(x, 10, x + 80, 90) for x in (10, 150, 300, 450, 900)],
            crs=5070,
        ).to_file(self.source, layer=self.layer, driver="GPKG")
        gpd.GeoDataFrame(
            {"name": ["USA"]}, geometry=[box(0, 0, 500, 100)], crs=5070,
        ).to_file(self.boundary, layer="boundary", driver="GPKG")

    def run_products(self, *products, validate_only=False):
        """Invoke the CLI with real files and one in-process worker.

        Args:
            *products: Product names to select.
            validate_only: Stop after input validation when true.
        """
        argv = [
            "prepare_padus_all_and_public_lands.py",
            "--padus-path", str(self.source),
            "--padus-layer", self.layer,
            "--products", *products,
            "--rules-config", str(self.rules),
            "--output-dir", str(self.output),
        ]
        if validate_only:
            argv.append("--validate-only")
        with (
            patch.object(sys, "argv", argv),
            patch.object(prepare, "USA_BOUNDARY_PATH", self.boundary),
            patch.object(prepare, "ProcessPoolExecutor", ThreadPoolExecutor),
            patch.object(prepare, "N_WORKERS", 1),
            patch.object(prepare, "WORKER", {}),
        ):
            prepare.main()

    def test_validate_only_checks_active_rules_without_writing(self):
        """No owner fields, coded domains, boundary load, or outputs are needed."""
        with patch.object(prepare, "_read_usa_boundary", side_effect=AssertionError("boundary opened")):
            self.run_products("all", "public", validate_only=True)
        self.assertFalse(self.output.exists())

    def test_all_only_requires_no_classification_fields(self):
        """An all-land run does not need fields used only by subset rules."""
        self.source = self.root / "no_rule_fields.gpkg"
        gpd.GeoDataFrame(
            {"Unit_Nm": ["unnamed management"]},
            geometry=[box(10, 10, 90, 90)],
            crs=5070,
        ).to_file(self.source, layer=self.layer, driver="GPKG")
        self.run_products("all")
        all_path, = (self.output / "all_lands").glob("*.gpkg")
        self.assertEqual(gpd.read_file(all_path).land_type.to_list(), ["all"])
        self.assertFalse((self.output / "public_lands").exists())
        self.assertFalse((self.output / "public_access_lands").exists())

    def test_public_access_requires_owner_and_local_manager_fields(self):
        """Do not silently evaluate the public-access rule with absent fields."""
        with self.assertRaisesRegex(RuleConfigError, r"Own_Type|Loc_Mang"):
            self.run_products("public-access", validate_only=True)
        self.assertFalse(self.output.exists())

    def test_domain_fallback_rejects_unrecognized_rule_code(self):
        """Domain-less input still validates configured PAD-US code values."""
        self.rules = self.root / "invalid.rules"
        self.rules.write_text(
            "public_land = Mang_Type in {UNLISTED}\n"
            "public_access = public_land AND Pub_Access in {OA}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuleConfigError, r"UNLISTED"):
            self.run_products("all", "public", validate_only=True)
        self.assertFalse(self.output.exists())

    def test_all_public_clips_and_writes_only_requested_products(self):
        """Keep public as a subset and do not create a public-access product."""
        self.run_products("all", "public")
        all_path, = (self.output / "all_lands").glob("*.gpkg")
        public_path, = (self.output / "public_lands").glob("*.gpkg")
        self.assertFalse((self.output / "public_access_lands").exists())
        all_lands = gpd.read_file(all_path).set_index("Unit_Nm")
        public = gpd.read_file(public_path).set_index("Unit_Nm")
        self.assertEqual(set(all_lands.index), {"federal", "unknown_park", "ngo", "unlisted"})
        self.assertEqual(set(public.index), {"federal", "unknown_park"})
        self.assertEqual(set(all_lands.land_type), {"all"})
        self.assertEqual(set(public.land_type), {"public"})
        self.assertTrue(all_lands.loc[public.index].geometry.equals(public.geometry))
        self.assertAlmostEqual(all_lands.loc["unlisted"].geometry.bounds[2], 500.0)


if __name__ == "__main__":
    unittest.main()
