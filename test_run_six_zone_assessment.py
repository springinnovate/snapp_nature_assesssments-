"""Exercise the six-zone wrapper without running production analysis."""

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import MultiPolygon, box

import run_six_zone_assessment as wrapper


class SixZoneAssessmentTests(unittest.TestCase):
    """Check run handoffs and output isolation with one county per product."""

    def setUp(self) -> None:
        """Create a temporary data root with the wrapper's required inputs."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "data"
        self.run_id = "test-six-zones"
        self.results = self.data / "analysis_results/assessment_runs" / self.run_id
        self.work = self.data / "processing_outputs/assessment_runs" / self.run_id
        self.zone_root = self.work / "zonal_units"
        self.zonal_config = self.root / wrapper.DEFAULT_CONFIG.relative_to(wrapper.ROOT)
        self.frozen = self.zonal_config.parent / "runs" / f"snapp_six_zone_{self.run_id}.yaml"
        self.runner = self.root / "runner.py"
        self.bbb = self.root / "bbb_geometry.gpkg"
        self.county = gpd.GeoDataFrame(
            {"GEOID": ["01001"], "ALAND": [100]},
            geometry=[MultiPolygon([box(0, 0, 10, 10)])], crs="EPSG:6933",
        )
        counties, padus = self.root / "counties.gpkg", self.root / "padus.gpkg"
        pyogrio.write_dataframe(self.county, counties, layer="counties_input")
        reference = self.root / wrapper.COUNTY_PATH
        reference.parent.mkdir(parents=True)
        reference_frame = gpd.GeoDataFrame(
            {"GEOID": ["01001", "15001"]},
            geometry=[MultiPolygon([box(0, 0, 10, 10)]), MultiPolygon([box(100, 100, 110, 110)])],
            crs=self.county.crs,
        )
        pyogrio.write_dataframe(reference_frame, reference, layer="coral_reference")
        pyogrio.write_dataframe(self.county[["GEOID", "geometry"]], self.bbb, layer="bbb_input")
        pyogrio.write_dataframe(
            self.county.assign(Mang_Type="FED", Des_Tp="NP", GAP_Sts="1"),
            padus, layer="padus_input",
        )
        config_dir = self.data / "workflow_assets/county_integration"
        for path in (
            self.runner, self.zonal_config, self.root / "config/padus_land_rules.rules",
            *(config_dir / name for name in wrapper.CONFIG_FILES),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        self.args = [
            "--runner", str(self.runner), "--bbb-by-county", str(self.bbb),
            "--padus-path", str(padus), "--padus-layer", "padus_input",
            "--counties", str(counties), "--run-id", self.run_id,
        ]
        self.calls = []
        self.fail_runner = False
        self.read_rasters = Mock(return_value={})
        self.valuation = Mock(side_effect=self.fake_valuation)

    def fake_subprocess(self, command, *, cwd, check, **_) -> subprocess.CompletedProcess:
        """Produce the prepared zone files or simulate the toolkit process."""
        script = Path(command[2])
        self.assertEqual(cwd, self.root)
        self.assertTrue(check)
        self.calls.append(script.name)
        if script.name == "prepare_assessment_zones.py":
            self.assertEqual(command[command.index("--bbb-by-county") + 1], str(self.bbb))
            self.assertEqual(command[command.index("--bbb-layer") + 1], "bbb_input")
            self.assertEqual(Path(command[command.index("--output-root") + 1]), self.work.parent)
            outputs = {}
            for product in wrapper.ZONE_PRODUCTS:
                path = self.zone_root / product / f"{product}_{self.run_id}.gpkg"
                path.parent.mkdir(parents=True)
                frame = self.county.copy()
                if product.endswith("_and_manager_type"):
                    frame["Mang_Type"] = "FED"
                pyogrio.write_dataframe(frame, path, layer=product)
                outputs[product] = {"path": str(path), "layer": product}
            (self.zone_root / "zone_manifest.json").write_text(
                json.dumps({
                    "run_id": self.run_id,
                    "geographic_scope": "CONUS",
                    "outputs": outputs,
                    "counties_with_manager_overlap": {
                        "area_ha_public_manager_overlap": 0,
                        "area_ha_conservation_manager_overlap": 0,
                    },
                }),
                encoding="utf-8",
            )
        elif script == self.runner:
            self.assertEqual(Path(command[3]), self.frozen)
            self.assertTrue(self.frozen.is_file())
            if self.fail_runner:
                raise subprocess.CalledProcessError(17, command)
        else:
            self.fail(f"Unexpected script: {script}")
        return subprocess.CompletedProcess(command, 0)

    def fake_freeze(self, zones, config, frozen, work, raw, **_) -> SimpleNamespace:
        """Check the six fresh zone paths and write a stand-in runner config."""
        self.assertEqual(set(zones), set(wrapper.ZONE_PRODUCTS))
        self.assertEqual(config, self.zonal_config)
        self.assertEqual(frozen, self.frozen)
        self.assertEqual(work, self.work / "zonal_stats_work")
        self.assertEqual(raw, self.work / "raw_zonals")
        for product, (path, layer) in zones.items():
            self.assertEqual(layer, product)
            self.assertEqual(path, self.zone_root / product / f"{product}_{self.run_id}.gpkg")
        frozen.parent.mkdir(parents=True)
        frozen.write_text("fixture\n", encoding="utf-8")
        return SimpleNamespace(jobs=("mock-job",), skipped_products=(), frozen_config=frozen)

    def fake_valuation(self, zones, *_args, coral_reference_counties, **_kwargs):
        """Return values with the expected county-only recreation schema."""
        self.assertEqual(set(zones), set(wrapper.ZONE_PRODUCTS))
        self.assertEqual(coral_reference_counties.GEOID.tolist(), ["01001", "15001"])
        self.assertEqual(zones["counties"].GEOID.tolist(), ["01001"])
        valued = {product: frame.assign(es_total=10.0) for product, frame in zones.items()}
        valued["counties"]["recreation_adj"] = 5.0
        return valued, {}, {}

    def run_wrapper(self, *extra: str) -> Path | None:
        """Run real orchestration and publication with heavy stages stubbed."""
        rasters = [
            SimpleNamespace(path=Path(f"{field.removeprefix('sum_')}.tif"))
            for field in wrapper.BASE_FIELDS.values()
        ]
        args = wrapper.parse_args([*self.args, *extra])
        with (
            patch.object(wrapper.subprocess, "run", side_effect=self.fake_subprocess),
            patch.object(wrapper, "load_valuation_config", return_value=({}, None, None, {"sources": {}})),
            patch.object(wrapper, "validate_zone_raster_inputs", return_value=(rasters, [])),
            patch.object(wrapper, "freeze_zone_zonals", side_effect=self.fake_freeze),
            patch.object(wrapper, "read_zone_raster_sums", self.read_rasters),
            patch.object(wrapper, "value_zones", self.valuation),
        ):
            return wrapper.run_assessment(args, root=self.root)

    def test_complete_run_publishes_all_six_pairs(self) -> None:
        """Publish exact pairs and record geometry inputs and stage completion."""
        self.assertFalse(self.bbb.with_suffix(".csv").exists())
        self.assertEqual(self.run_wrapper(), self.results)
        manifest = json.loads((self.results / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["geographic_scope"], "CONUS")
        self.assertEqual(manifest["zone_inputs"]["bbb_by_county"], [str(self.bbb), "bbb_input"])
        self.assertEqual(self.calls, ["prepare_assessment_zones.py", "runner.py"])
        self.assertEqual(list(manifest["stages"].items()), [
            ("Prepare six assessment zones", "complete"),
            ("Freeze raw raster zonal jobs", "complete"),
            ("Run and verify raster zonals", "complete"),
            ("Value all six zone products", "complete"),
            ("Publish six CSV and GeoPackage pairs", "complete"),
        ])
        self.assertEqual(set(manifest["outputs"]), set(wrapper.ZONE_PRODUCTS))
        for product, record in manifest["outputs"].items():
            with self.subTest(product=product):
                gpkg = self.results / product / f"{product}_{self.run_id}.gpkg"
                csv_path = gpkg.with_suffix(".csv")
                self.assertEqual(record, {"csv": str(csv_path), "gpkg": str(gpkg), "rows": 1})
                self.assertCountEqual(gpkg.parent.iterdir(), [csv_path, gpkg])
                attributes = pd.read_csv(csv_path, dtype={"GEOID": str})
                self.assertEqual(attributes.GEOID.tolist(), ["01001"])
                self.assertEqual("recreation_adj" in attributes, product == "counties")

    def test_runner_failure_stops_before_valuation_and_publication(self) -> None:
        """Record the runner failure without publishing any final pair."""
        self.fail_runner = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_wrapper()
        manifest = json.loads((self.results / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"]["Run and verify raster zonals"], "failed")
        self.assertEqual(manifest["outputs"], {})
        self.assertFalse(any(self.results.rglob("*.gpkg")))
        self.read_rasters.assert_not_called()
        self.valuation.assert_not_called()

    def test_validate_only_writes_no_run_artifacts(self) -> None:
        """Check input metadata without launching any stage or writing results."""
        self.assertIsNone(self.run_wrapper("--validate-only"))
        self.assertEqual(self.calls, [])
        for path in (self.results, self.work, self.frozen):
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
