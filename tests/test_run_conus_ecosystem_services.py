"""Exercise the CONUS assessment wrapper without running production analysis."""

from configparser import ConfigParser
from contextlib import redirect_stderr
from importlib.util import find_spec
from io import StringIO
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
from shapely.geometry import MultiPolygon, Point, box

import run_conus_ecosystem_services as wrapper

if find_spec("matplotlib") is not None:
    from docs.render_readme_maps import read_assessment_products
else:
    read_assessment_products = None


class ConusEcosystemServicesTests(unittest.TestCase):
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
        self.land_mask = self.root / "county_land.gpkg"
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
        pyogrio.write_dataframe(self.county[["GEOID", "geometry"]], self.land_mask, layer="county_land_mask")
        pyogrio.write_dataframe(
            self.county.assign(Mang_Type="FED", Des_Tp="NP", GAP_Sts="1"),
            padus, layer="padus_input",
        )
        config_dir = self.data / "workflow_assets/county_integration"
        self.air_aliases = config_dir / "air_quality_county_aliases.csv"
        for path in (
            self.runner, self.zonal_config, self.root / "config/padus_land_rules.rules",
            self.air_aliases,
            config_dir / "virginia_gdp_crosswalk.csv",
            *(config_dir / name for name in wrapper.CONFIG_FILES),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        self.args = [
            "--runner", str(self.runner), "--bbb-by-county", str(self.bbb),
            "--land-mask", str(self.land_mask),
            "--padus-path", str(padus), "--padus-layer", "padus_input",
            "--counties", str(counties), "--run-id", self.run_id,
        ]
        self.calls = []
        self.toolkit_batches = []
        self.zone_geometries = {}
        self.fail_runner = False
        self.read_rasters = Mock(return_value={})
        self.valuation = Mock(side_effect=self.fake_valuation)

    def fake_subprocess(self, command, *, cwd, check, **_) -> subprocess.CompletedProcess:
        """Produce the prepared zone files or simulate the toolkit process."""
        self.assertEqual(cwd, self.root)
        self.assertTrue(check)
        self.assertEqual(command[1], "-B")
        if command[2:4] == ["-m", "assessment.prepare_assessment_zones"]:
            self.calls.append(command[3])
            self.assertEqual(command[command.index("--bbb-by-county") + 1], str(self.bbb))
            self.assertEqual(command[command.index("--bbb-layer") + 1], "bbb_input")
            self.assertEqual(command[command.index("--land-mask") + 1], str(self.land_mask))
            self.assertEqual(command[command.index("--land-mask-layer") + 1], "county_land_mask")
            self.assertEqual(Path(command[command.index("--output-root") + 1]), self.work.parent)
            run_id = command[command.index("--run-id") + 1]
            zone_root = self.work.with_name(run_id) / "zonal_units"
            outputs = {}
            for product in wrapper.ZONE_PRODUCTS:
                path = zone_root / product / f"{product}_{run_id}.gpkg"
                path.parent.mkdir(parents=True)
                frame = self.county.copy()
                if product in self.zone_geometries:
                    frame = frame.set_geometry([self.zone_geometries[product]])
                if product.endswith("_and_manager_type"):
                    frame["Mang_Type"] = "FED"
                pyogrio.write_dataframe(frame, path, layer=product)
                outputs[product] = {"path": str(path), "layer": product}
            (zone_root / "zone_manifest.json").write_text(
                json.dumps({
                    "run_id": run_id,
                    "geographic_scope": "CONUS",
                    "outputs": outputs,
                    "counties_with_manager_overlap": {
                        "area_ha_public_manager_overlap": 0,
                        "area_ha_conservation_manager_overlap": 0,
                    },
                }),
                encoding="utf-8",
            )
        elif Path(command[2]) == self.runner:
            self.calls.append(self.runner.name)
            self.assertTrue(Path(command[3]).is_file())
            if self.fail_runner:
                raise subprocess.CalledProcessError(17, command)
        else:
            self.fail(f"Unexpected command: {command}")
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
            patch.object(wrapper, "load_valuation_config", return_value=({"air_quality_county_aliases": self.air_aliases}, None, None, {"sources": {
                "gdp": {"path": "gdp.csv", "allocation_crosswalk":
                        "workflow_assets/county_integration/virginia_gdp_crosswalk.csv"}}})),
            patch.object(wrapper, "validate_zone_raster_inputs", return_value=(rasters, [])),
            patch.object(wrapper, "freeze_zone_zonals", side_effect=self.fake_freeze),
            patch.object(wrapper, "read_zone_raster_sums", self.read_rasters),
            patch.object(wrapper, "value_zones", self.valuation),
        ):
            return wrapper.run_assessment(args, root=self.root)

    def write_raster_inventory(self) -> None:
        """Provide real runner configuration metadata for the orchestration tests."""
        raster_dir = self.root / "rasters"
        raster_dir.mkdir()
        inventory = []
        for column in wrapper.BASE_FIELDS.values():
            path = raster_dir / f"{column.removeprefix('sum_')}.tif"
            path.touch()
            inventory.append({"label": column, "filename": path.name, "units": "test units", "include": True})
        config = ConfigParser(interpolation=None)
        config["project"] = {
            "raster_dir": str(raster_dir), "max_simplify_tolerance_meters": "3",
            "raster_inventory": json.dumps(inventory),
        }
        config["job:template"] = {"zone_key": "unused"}
        with self.zonal_config.open("w", encoding="utf-8") as stream:
            config.write(stream)

    def simulate_toolkit(self, command, **kwargs):
        """Write paired sums for the jobs actually present in each frozen profile."""
        result = self.fake_subprocess(command, **kwargs)
        if Path(command[2]) != self.runner:
            return result
        frozen = Path(command[3])
        config = ConfigParser(interpolation=None)
        config.read(frozen, encoding="utf-8")
        jobs = [config[name] for name in config.sections() if name.startswith("job:")]
        self.toolkit_batches.append({job["agg_layer"] for job in jobs})
        stamp, raster_sum = ("20261106_000001", 1.0) if frozen == self.frozen else ("20261106_000002", 2.0)
        for job in jobs:
            columns = job["agg_field"].split(", ")
            frame = pyogrio.read_dataframe(
                frozen.parent / job["agg_vector"], layer=job["agg_layer"], columns=columns, read_geometry=False,
            )
            for column in wrapper.BASE_FIELDS.values():
                frame[column] = raster_sum
            output = frozen.parent / job["output_csv"]
            stamped = output.with_name(f"{output.stem}_{stamp}.csv")
            stamped.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(stamped, index=False)
            stamped.with_suffix(".gpkg").write_bytes(b"simulated toolkit output")
        return result

    def run_with_toolkit_outputs(self, *argv: str) -> Path | None:
        """Use the real frozen plan and result readers with simulated toolkit sums."""
        args = wrapper.parse_args(list(argv))
        with (
            patch.object(wrapper.subprocess, "run", side_effect=self.simulate_toolkit),
            patch.object(wrapper, "load_valuation_config", return_value=(
                {"air_quality_county_aliases": self.air_aliases}, None, None, {"sources": {"gdp": {"path": "gdp.csv"}}},
            )),
            patch.object(wrapper, "value_zones", self.valuation),
        ):
            return wrapper.run_assessment(args, root=self.root)

    def conservation_refresh_args(self, run_id: str) -> list[str]:
        """Use the documented flags for a selective refresh of the fixture run."""
        return [
            "--reuse-zonals-from", self.run_id, "--refresh-conservation-zonals",
            "--runner", str(self.runner), "--run-id", run_id,
        ]

    def test_complete_run_publishes_all_six_pairs(self) -> None:
        """Publish exact pairs and record geometry inputs and stage completion."""
        self.assertFalse(self.bbb.with_suffix(".csv").exists())
        self.assertEqual(self.run_wrapper(), self.results)
        manifest = json.loads((self.results / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["geographic_scope"], "CONUS")
        self.assertEqual(manifest["valuation_execution"], {
            "executor": "shared_process_pool", "workers": 4, "native_threads_per_worker": 1,
        })
        self.assertEqual(self.valuation.call_args.kwargs["workers"], 4)
        self.assertNotIn("geometry_workers", self.valuation.call_args.kwargs)
        self.assertEqual(manifest["configuration_sha256"][str(self.air_aliases)],
                         wrapper.sha256(self.air_aliases))
        self.assertEqual(manifest["valuation_inputs"]["air_quality_county_aliases"], str(self.air_aliases))
        self.assertEqual(manifest["zone_inputs"]["bbb_by_county"], [str(self.bbb), "bbb_input"])
        self.assertEqual(manifest["zone_inputs"]["land_mask"], [str(self.land_mask), "county_land_mask"])
        self.assertEqual(self.calls, ["assessment.prepare_assessment_zones", "runner.py"])
        self.assertEqual(list(manifest["stages"].items()), [
            ("Prepare six assessment zones", "complete"),
            ("Freeze raw raster zonal jobs", "complete"),
            ("Run and verify raster zonals", "complete"),
            ("Value all six zone products", "complete"),
            ("Publish six CSV and GeoPackage pairs", "complete"),
        ])
        self.assertEqual(set(manifest["outputs"]), set(wrapper.ZONE_PRODUCTS))
        self.assertEqual(
            {path.name for path in self.results.iterdir()},
            {"run_manifest.json"} | {
                f"{product}_{self.run_id}.{extension}"
                for product in wrapper.ZONE_PRODUCTS for extension in ("csv", "gpkg")
            },
        )
        self.assertTrue(all(path.is_file() for path in self.results.iterdir()))
        for product, record in manifest["outputs"].items():
            with self.subTest(product=product):
                gpkg = self.results / f"{product}_{self.run_id}.gpkg"
                csv_path = gpkg.with_suffix(".csv")
                self.assertEqual(record, {"csv": str(csv_path), "gpkg": str(gpkg), "rows": 1})
                attributes = pd.read_csv(csv_path, dtype={"GEOID": str})
                self.assertEqual(attributes.GEOID.tolist(), ["01001"])
                self.assertEqual("recreation_adj" in attributes, product == "counties")

    def test_legacy_geometry_limit_overrides_one_valuation_pool(self) -> None:
        """Keep preparation and valuation limits without nested worker counts."""
        self.run_wrapper("--workers", "2", "--geometry-workers", "3")
        manifest = json.loads((self.results / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["valuation_execution"]["workers"], 3)
        self.assertEqual(self.valuation.call_args.kwargs["workers"], 3)
        self.assertNotIn("geometry_workers", self.valuation.call_args.kwargs)

    def test_workers_controls_shared_pool_without_legacy_override(self) -> None:
        """Use the requested worker count for all non-zonal task families."""
        self.run_wrapper("--workers", "2")
        self.assertEqual(self.valuation.call_args.kwargs["workers"], 2)

    def test_reuse_verified_run_without_preparing_zones_or_running_toolkit(self) -> None:
        """Revalue saved sums in a new directory without changing the source run."""
        self.assertEqual(self.run_wrapper(), self.results)
        source_manifest = self.results / "run_manifest.json"
        frozen_bytes = self.frozen.read_bytes()
        existing_plan = SimpleNamespace(skipped_products=())
        raw = self.work / "raw_zonals"
        fake_sums = {"saved": object()}
        for layout in ("flat", "nested"):
            with self.subTest(source_layout=layout):
                if layout == "nested":
                    original = json.loads(source_manifest.read_text(encoding="utf-8"))
                    for product, record in original["outputs"].items():
                        (self.results / product).mkdir()
                        for extension in ("csv", "gpkg"):
                            path = Path(record[extension])
                            nested_path = self.results / product / path.name
                            path.rename(nested_path)
                            record[extension] = str(nested_path)
                    source_manifest.write_text(json.dumps(original), encoding="utf-8")
                source_files = {
                    path: path.read_bytes() for path in self.results.rglob("*") if path.is_file()
                }
                reused = self.results.with_name(f"revalued-{layout}")
                args = wrapper.parse_args(["--reuse-zonals-from", self.run_id, "--run-id", reused.name])
                with (
                    patch.object(wrapper.subprocess, "run") as subprocess_run,
                    patch.object(wrapper, "validate_zone_raster_inputs") as inspect_tiffs,
                    patch.object(wrapper, "freeze_zone_zonals") as freeze,
                    patch.object(wrapper, "load_valuation_config", return_value=({"air_quality_county_aliases": self.air_aliases}, None, None, {
                        "sources": {"gdp": {"path": "gdp.csv"}}})),
                    patch.object(wrapper, "read_frozen_zone_raster_plan", return_value=existing_plan) as reconstruct,
                    patch.object(wrapper, "read_zone_raster_sums", return_value=fake_sums) as read_sums,
                    patch.object(wrapper, "value_zones", side_effect=self.fake_valuation) as value,
                ):
                    self.assertEqual(wrapper.run_assessment(args, root=self.root), reused)
                subprocess_run.assert_not_called()
                inspect_tiffs.assert_not_called()
                freeze.assert_not_called()
                read_sums.assert_called_once_with(existing_plan)
                self.assertIs(value.call_args.args[1], fake_sums)
                zones, frozen, work_dir, raw_dir = reconstruct.call_args.args
                self.assertEqual((frozen, work_dir, raw_dir), (self.frozen, self.work / "zonal_stats_work", raw))
                self.assertEqual(zones["counties"], (self.zone_root / "counties" / f"counties_{self.run_id}.gpkg", "counties"))
                self.assertEqual(
                    {path: path.read_bytes() for path in self.results.rglob("*") if path.is_file()},
                    source_files,
                )
                self.assertEqual(self.frozen.read_bytes(), frozen_bytes)
                self.assertFalse(self.work.with_name(reused.name).exists())
                manifest = json.loads((reused / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["reused_zonals_from"], {"run_id": self.run_id, "manifest": str(source_manifest)})
                self.assertEqual(manifest["configuration_sha256"][str(self.air_aliases)],
                                 wrapper.sha256(self.air_aliases))
                self.assertEqual(manifest["stages"]["Verify reused raster zonals"], "complete")
                self.assertNotIn("Run and verify raster zonals", manifest["stages"])
                self.assertEqual(set(manifest["outputs"]), set(wrapper.ZONE_PRODUCTS))
                self.assertEqual(
                    {path.name for path in reused.iterdir()},
                    {"run_manifest.json"} | {
                        f"{product}_{reused.name}.{extension}"
                        for product in wrapper.ZONE_PRODUCTS for extension in ("csv", "gpkg")
                    },
                )
                self.assertTrue(all(path.is_file() for path in reused.iterdir()))
                for product, record in manifest["outputs"].items():
                    gpkg = reused / f"{product}_{reused.name}.gpkg"
                    self.assertEqual(record, {"csv": str(gpkg.with_suffix(".csv")), "gpkg": str(gpkg), "rows": 1})

    def test_refresh_cli_requires_source_and_runner_and_rejects_zone_overrides(self) -> None:
        """Require saved inputs for the selective workflow and a runner for new zonals."""
        for argv, error in (
            (["--refresh-conservation-zonals", "--runner", str(self.runner)], "requires --reuse-zonals-from"),
            (["--refresh-conservation-zonals", "--reuse-zonals-from", self.run_id], "requires --runner"),
            ([*self.conservation_refresh_args("refresh"), "--land-mask", str(self.land_mask)], "omit zone overrides"),
            (["--reuse-zonals-from", self.run_id, "--runner", str(self.runner)], "--runner is only needed"),
        ):
            with self.subTest(argv=argv), redirect_stderr(StringIO()) as stderr, self.assertRaises(SystemExit):
                wrapper.parse_args(argv)
            self.assertIn(error, stderr.getvalue())

    def test_refresh_validate_only_reads_retained_sums_without_writing(self) -> None:
        """Check the saved raster pairs and original inputs without launching stages."""
        self.write_raster_inventory()
        self.run_with_toolkit_outputs(*self.args)
        self.calls.clear()
        self.valuation.reset_mock()
        self.assertIsNone(self.run_with_toolkit_outputs(*self.conservation_refresh_args("preflight"), "--validate-only"))
        self.assertEqual(self.calls, [])
        self.valuation.assert_not_called()
        for path in (self.results.with_name("preflight"), self.work.with_name("preflight"),
                     self.frozen.with_name("snapp_six_zone_preflight.yaml")):
            self.assertFalse(path.exists())

    def test_refresh_runs_only_conservation_jobs_and_supports_later_revaluation(self) -> None:
        """Keep the original four sums, replace both conservation sums, and revalue the mixed run."""
        conservation = {"conservation_lands_by_county", "conservation_lands_by_county_and_manager_type"}
        self.write_raster_inventory()
        self.zone_geometries.update({product: MultiPolygon([box(0, 0, 5, 10)]) for product in conservation})
        self.run_with_toolkit_outputs(*self.args)
        self.zone_geometries.clear()
        self.calls.clear()
        source_manifest = self.results / "run_manifest.json"
        original_artifacts = {path: path.read_bytes() for path in (source_manifest, self.frozen)}
        refreshed = self.run_with_toolkit_outputs(*self.conservation_refresh_args("gap4"))
        self.assertEqual(refreshed, self.results.with_name("gap4"))
        self.assertEqual(self.toolkit_batches, [set(wrapper.ZONE_PRODUCTS), conservation])
        self.assertEqual(self.calls, ["assessment.prepare_assessment_zones", "runner.py"])
        for path, original in original_artifacts.items():
            self.assertEqual(path.read_bytes(), original)
        manifest = json.loads((refreshed / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(set(manifest["outputs"]), set(wrapper.ZONE_PRODUCTS))
        self.assertEqual(set(manifest["refreshed_raster_products"]), conservation)
        self.assertEqual(manifest["reused_zonals_from"]["run_id"], self.run_id)
        self.assertEqual(set(manifest["reused_zonals_from"]["products"]), set(wrapper.ZONE_PRODUCTS) - conservation)
        refreshed_zones, refreshed_sums = self.valuation.call_args.args[:2]
        self.assertEqual(set(refreshed_sums), set(wrapper.ZONE_PRODUCTS))
        raster_column = next(iter(wrapper.BASE_FIELDS.values()))
        for product, frame in refreshed_sums.items():
            self.assertEqual(frame[raster_column].tolist(), [2.0 if product in conservation else 1.0])
            self.assertEqual(refreshed_zones[product].geometry.area.tolist(), [100.0])

        self.calls.clear()
        self.assertEqual(
            self.run_with_toolkit_outputs("--reuse-zonals-from", "gap4", "--run-id", "gap4-revalued"),
            self.results.with_name("gap4-revalued"),
        )
        self.assertEqual(self.calls, [])
        revalued_zones, revalued_sums = self.valuation.call_args.args[:2]
        for product in wrapper.ZONE_PRODUCTS:
            pd.testing.assert_frame_equal(revalued_sums[product], refreshed_sums[product])
            self.assertEqual(revalued_zones[product].geometry.area.tolist(), [100.0])

    def test_refresh_rejects_changed_retained_geometry_before_toolkit(self) -> None:
        """Require matching retained geometry before planning or running new raster jobs."""
        self.write_raster_inventory()
        self.run_with_toolkit_outputs(*self.args)
        self.calls.clear()
        self.valuation.reset_mock()
        self.zone_geometries["bbb_lands_by_county"] = MultiPolygon([box(0, 0, 5, 10)])
        with self.assertRaisesRegex(ValueError, "bbb_lands_by_county: .*run fresh zonals"):
            self.run_with_toolkit_outputs(*self.conservation_refresh_args("mismatch"))
        self.assertEqual(self.calls, ["assessment.prepare_assessment_zones"])
        self.valuation.assert_not_called()
        self.assertFalse(self.frozen.with_name("snapp_six_zone_mismatch.yaml").exists())
        manifest_path = self.results.with_name("mismatch") / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["outputs"], {})

    def test_reuse_refuses_a_run_with_unverified_raster_sums(self) -> None:
        """Stop before creating outputs when the source zonals did not finish."""
        self.fail_runner = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_wrapper()
        args = wrapper.parse_args(["--reuse-zonals-from", self.run_id, "--run-id", "revalued"])
        with (
            patch.object(wrapper, "load_valuation_config", return_value=({"air_quality_county_aliases": self.air_aliases}, None, None, {
                "sources": {"gdp": {"path": "gdp.csv"}}})),
            patch.object(wrapper, "read_frozen_zone_raster_plan") as reconstruct,
            self.assertRaisesRegex(ValueError, "no completed, verified CONUS raster zonals"),
        ):
            wrapper.run_assessment(args, root=self.root)
        reconstruct.assert_not_called()
        self.assertFalse(self.results.with_name("revalued").exists())

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

    def test_preflight_rejects_nonpolygon_land_mask(self) -> None:
        """Stop before creating run artifacts if the selected mask layer is not polygonal."""
        invalid = gpd.GeoDataFrame({"GEOID": ["01001"]}, geometry=[Point(5, 5)], crs=self.county.crs)
        pyogrio.write_dataframe(invalid, self.land_mask, layer="not_a_mask")
        with self.assertRaisesRegex(ValueError, "must be a polygon land-mask layer"):
            self.run_wrapper("--land-mask-layer", "not_a_mask")
        self.assertEqual(self.calls, [])
        for path in (self.results, self.work, self.frozen):
            self.assertFalse(path.exists())


@unittest.skipIf(read_assessment_products is None, "README map checks require optional Matplotlib")
class ReadmeMapRunTests(unittest.TestCase):
    """Check that maps use only the explicitly selected run's recorded files."""

    def setUp(self) -> None:
        """Create an explicit run manifest with the five map source products."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name).resolve() / "mapped-run"
        self.run_dir.mkdir()
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.names = (
            "counties", "public_lands_by_county", "public_lands_by_county_and_manager_type",
            "conservation_lands_by_county", "bbb_lands_by_county",
        )
        self.manifest = {
            "run_id": self.run_dir.name, "geographic_scope": "CONUS", "status": "complete",
            "outputs": {},
        }
        for name in self.names:
            path = self.run_dir / f"{name}_{self.run_dir.name}.gpkg"
            path.touch()
            self.manifest["outputs"][name] = {"gpkg": str(path)}

    def test_manifest_resolves_flat_and_legacy_nested_outputs(self) -> None:
        """Read absolute or run-relative paths without guessing from other files."""
        (self.run_dir / "unrecorded.gpkg").touch()
        for layout in ("flat", "nested"):
            for relative in (False, True):
                with self.subTest(layout=layout, relative=relative):
                    expected = {}
                    for name in self.names:
                        directory = self.run_dir / name if layout == "nested" else self.run_dir
                        directory.mkdir(exist_ok=True)
                        path = directory / f"{name}_{self.run_dir.name}.gpkg"
                        path.touch()
                        (directory / "unrecorded.gpkg").touch()
                        self.manifest["outputs"][name]["gpkg"] = str(
                            path.relative_to(self.run_dir) if relative else path
                        )
                        expected[name] = path
                    self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
                    self.assertEqual(read_assessment_products(self.run_dir), expected)

    def test_manifest_rejects_unexpected_or_missing_product_paths(self) -> None:
        """Reject unrelated products, another run, traversal, and missing files."""
        filename = f"counties_{self.run_dir.name}.gpkg"
        outside = self.run_dir.parent / filename
        outside.touch()
        wrong_name = self.run_dir / "counties_other-run.gpkg"
        wrong_name.touch()
        wrong_product = Path(self.manifest["outputs"]["public_lands_by_county"]["gpkg"])
        for recorded in (
            str(outside), f"../{filename}", str(wrong_name), str(wrong_product),
            str(self.run_dir / "counties" / filename),
        ):
            with self.subTest(recorded=recorded):
                self.manifest["outputs"]["counties"]["gpkg"] = recorded
                self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Unexpected GeoPackage path for counties"):
                    read_assessment_products(self.run_dir)
        for record in ({}, {"gpkg": ""}, {"gpkg": None}):
            with self.subTest(record=record):
                self.manifest["outputs"]["counties"] = record
                self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Missing manifest GeoPackage path for counties"):
                    read_assessment_products(self.run_dir)

    def test_manifest_requires_selected_completed_conus_run(self) -> None:
        """Reject an unfinished run, unrelated run ID, or other geography."""
        for key, value in (("status", "failed"), ("run_id", "other-run"), ("geographic_scope", "US")):
            with self.subTest(key=key):
                invalid = self.manifest | {key: value}
                self.manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Choose a completed CONUS assessment run"):
                    read_assessment_products(self.run_dir)


if __name__ == "__main__":
    unittest.main()
