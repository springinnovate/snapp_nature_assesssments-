"""Exercise the assessment wrapper without running production analysis."""

import configparser
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_snapp_assessment as wrapper


class AssessmentWrapperTests(unittest.TestCase):
    """Check explicit stage handoffs, output isolation, and failure handling."""

    def setUp(self):
        """Create the configured inputs in an isolated repository stand-in."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "data"
        self.run_id = "test-assessment"
        self.results = self.data / "analysis_results" / "assessment_runs" / self.run_id
        self.work = self.data / "processing_outputs" / "assessment_runs" / self.run_id
        self.zonal_assets = self.data / "workflow_assets" / "zonal_stats"
        self.snapshot = self.zonal_assets / "runs" / f"snapp_assessment_{self.run_id}.yaml"
        self.manifest = self.results / "run_manifest.json"
        self.padus = self.data / "analysis_inputs" / "test_padus.gdb"
        self.padus.mkdir(parents=True)
        self.runner = self.root / "toolkit" / "runner.py"
        self.bbb = self.data / "analysis_results" / "bbb_candidate_ecosystem_services_20260901_214726.gpkg"
        self.stage_calls = []
        self.cuts = {}
        self.fail_at = None
        self.wrong_config_cut = False
        self.mismatch_runner_csv = False
        for path in (
            self.runner,
            self.root / "prepare_padus_all_and_public_lands.py",
            self.root / "cut_and_flatten_by_county.py",
            self.root / "prepare_tif_zonal_stats_config.py",
            self.root / "build_county_ecosystem_service_valuations_2026.py",
            self.data / "analysis_inputs" / "boundaries" / "usa_boundary" / "usa_vector.gpkg",
            self.data / "analysis_inputs" / "zonal_units" / "counties" / "tl_2024_us_county_50_states.gpkg",
            self.data / "analysis_inputs" / "ecosystem_services" / "enabled.tif",
            self.data / "analysis_results" / "county_zonals.gpkg",
            self.data / "analysis_results" / "county_masks.gpkg",
            self.data / "analysis_inputs" / "recreation.csv",
        ):
            self.write_file(path)
        self.write_file(self.bbb, "reused BBB geometry")
        self.write_file(self.bbb.with_suffix(".csv"), "reused BBB values")
        self.write_file(self.root / "config" / "padus_land_rules.rules")
        county_assets = self.data / "workflow_assets" / "county_integration"
        self.write_file(
            county_assets / "sources.csv",
            "source,path\nbase,analysis_results/county_zonals.gpkg\n"
            "recreation,analysis_inputs/recreation.csv\n",
        )
        self.write_file(
            county_assets / "derived_metrics.yaml",
            "sources:\n  natural_areas:\n    path: analysis_results/county_masks.gpkg\n",
        )
        for filename in ("adjustment_factors.csv", "ct_county_crosswalk.csv", "export_rules.yaml"):
            self.write_file(county_assets / filename)
        self.write_file(
            self.zonal_assets / "snapp_assessment_tif_zonal_stats.yaml",
            "[project]\nraster_dir = ../../analysis_inputs/ecosystem_services\n"
            "raster_inventory = "
            '[{"filename": "enabled.tif", "include": true}, '
            '{"filename": "disabled.tif", "include": false}]\n'
            "[job:padus_all_lands]\nzone_key = all\n"
            "vector_pattern = padus_all_lands_by_county/padus_all_lands_clipped_by_county_*.gpkg\n"
            "results_subdir = padus_all_lands\n"
            "[job:padus_public_lands]\nzone_key = public\n"
            "vector_pattern = padus_public_lands_by_county/padus_public_lands_clipped_by_county_*.gpkg\n"
            "results_subdir = padus_public_lands\n",
        )
        self.args = [
            "--runner", str(self.runner), "--bbb-results", str(self.bbb),
            "--run-id", self.run_id, "--padus-path", str(self.padus),
            "--padus-layer", "test_vector_analysis", "--job-workers", "3",
            "--raster-workers", "4", "--workers", "5", "--geometry-workers", "6",
        ]

    @staticmethod
    def write_file(path, content="fixture"):
        """Write a nonempty fixture file under the temporary root."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def fake_subprocess(self, command, *, cwd, check, **_):
        """Check CLI handoffs and simulate only the selected stage outputs."""
        command = [str(arg) for arg in command]
        script = Path(command[1]).name
        self.assertEqual(Path(cwd), self.root)
        self.assertTrue(check)
        self.stage_calls.append((script, command))

        def option(flag):
            """Read the value of a passed CLI option."""
            return command[command.index(flag) + 1]

        if script == self.fail_at:
            raise subprocess.CalledProcessError(17, command)
        if script == "prepare_padus_all_and_public_lands.py":
            self.assertEqual(Path(option("--padus-path")), self.padus)
            self.assertEqual(option("--padus-layer"), "test_vector_analysis")
            product_idx = command.index("--products")
            self.assertEqual(command[product_idx + 1:product_idx + 3], ["all", "public"])
            if "--validate-only" not in command:
                dest = Path(option("--output-dir"))
                self.assertEqual(dest, self.work / "padus_clipped_to_usa")
                for product in ("all", "public"):
                    self.write_file(
                        dest / f"{product}_lands" / f"padus_{product}_lands_clipped_to_usa_2026_11_06_00_00_00.gpkg"
                    )
        elif script == "cut_and_flatten_by_county.py":
            source = next(Path(arg) for arg in command[2:] if arg.endswith(".gpkg"))
            self.assertTrue(source.is_file())
            self.assertTrue(source.is_relative_to(self.work / "padus_clipped_to_usa"))
            dest = Path(option("--output-dir"))
            self.assertEqual(dest.parent, self.work / "zonal_units")
            self.cuts[dest.name] = dest / source.name.replace("_clipped_to_usa_", "_clipped_by_county_")
            self.write_file(self.cuts[dest.name])
        elif script == "prepare_tif_zonal_stats_config.py":
            self.assertEqual(
                [command[i + 1] for i, arg in enumerate(command) if arg == "--zone"],
                ["all", "public"],
            )
            self.assertEqual(Path(option("--zonal-units-dir")), self.work / "zonal_units")
            self.assertEqual(Path(option("--results-dir")), self.results)
            self.assertEqual(Path(option("--output")), self.snapshot)
            config = configparser.ConfigParser()
            for product in ("all", "public"):
                job = f"padus_{product}_lands"
                destination = self.results / job / job
                vector = self.cuts[f"{job}_by_county"]
                if self.wrong_config_cut and product == "public":
                    vector = self.cuts["padus_all_lands_by_county"]
                config[f"job:{job}"] = {
                    "agg_vector": str(vector),
                    "output_gpkg": str(destination.with_suffix(".gpkg")),
                    "output_csv": str(destination.with_suffix(".csv")),
                }
            self.snapshot.parent.mkdir(parents=True, exist_ok=True)
            with self.snapshot.open("w", encoding="utf-8") as handle:
                config.write(handle)
        elif script == "runner.py":
            self.assertIn(str(self.snapshot), command)
            self.assertEqual(option("--job-workers"), "3")
            self.assertEqual(option("--raster-workers"), "4")
            config = configparser.ConfigParser()
            config.read(self.snapshot, encoding="utf-8")
            for section in config.sections():
                for key in ("output_gpkg", "output_csv"):
                    path = Path(config[section][key])
                    stamp = "20261105_000000" if self.mismatch_runner_csv and key == "output_csv" else "20261106_000000"
                    self.write_file(path.with_name(f"{path.stem}_{stamp}{path.suffix}"))
        elif script == "build_county_ecosystem_service_valuations_2026.py":
            output = Path(option("--output"))
            self.assertEqual(output.parent, self.results / "counties")
            self.assertEqual(option("--workers"), "5")
            self.assertEqual(option("--geometry-workers"), "6")
            self.write_file(output)
            self.write_file(output.with_suffix(".csv"))
        else:
            self.fail(f"Unexpected script: {script}")
        return subprocess.CompletedProcess(command, 0)

    def run_wrapper(self, *extra_args):
        """Invoke the wrapper with a subprocess stub and explicit fixture paths."""
        args = wrapper.parse_args([*self.args, *extra_args])
        with patch.object(wrapper.subprocess, "run", side_effect=self.fake_subprocess):
            return wrapper.run_assessment(args, root=self.root)

    def test_complete_run_keeps_four_matched_pairs_separate(self):
        """Pass only new county cuts to the toolkit and copy the supplied BBB pair."""
        self.assertEqual(self.run_wrapper(), self.results)
        self.assertEqual(
            [script for script, _ in self.stage_calls],
            [
                "prepare_padus_all_and_public_lands.py",
                "prepare_padus_all_and_public_lands.py",
                "cut_and_flatten_by_county.py",
                "cut_and_flatten_by_county.py",
                "prepare_tif_zonal_stats_config.py",
                "runner.py",
                "build_county_ecosystem_service_valuations_2026.py",
            ],
        )
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        stamps = {
            "padus_all_lands": "20261106_000000",
            "padus_public_lands": "20261106_000000",
            "counties": self.run_id,
            "bbb_candidate_blm_lands": "20260901_214726",
        }
        self.assertEqual(set(manifest["outputs"]), set(stamps))
        for product, stamp in stamps.items():
            with self.subTest(product=product):
                pair = [self.results / product / f"{product}_{stamp}.{ext}" for ext in ("csv", "gpkg")]
                self.assertCountEqual(pair, (self.results / product).iterdir())
                self.assertEqual(manifest["outputs"][product], list(map(str, pair)))
        bbb_source_pair = [self.bbb.with_suffix(".csv"), self.bbb]
        self.assertEqual(manifest["reused_bbb"], list(map(str, bbb_source_pair)))
        for source, copy in zip(bbb_source_pair, manifest["outputs"]["bbb_candidate_blm_lands"]):
            self.assertEqual(source.read_bytes(), Path(copy).read_bytes())

    def test_validate_only_does_not_create_run_outputs(self):
        """Check the source rules without creating directories or running jobs."""
        self.assertIsNone(self.run_wrapper("--validate-only"))
        self.assertEqual(len(self.stage_calls), 1)
        self.assertIn("--validate-only", self.stage_calls[0][1])
        self.assertFalse(self.results.exists())
        self.assertFalse(self.work.exists())
        self.assertFalse(self.snapshot.exists())

    def test_missing_bbb_csv_fails_preflight(self):
        """Reject an unpaired BBB input before launching a stage."""
        self.bbb.with_suffix(".csv").unlink()
        with self.assertRaises((FileNotFoundError, ValueError)) as raised:
            self.run_wrapper()
        self.assertIn(self.bbb.with_suffix(".csv").name, str(raised.exception))
        self.assertEqual(self.stage_calls, [])
        self.assertFalse(self.results.exists())

    def test_untimestamped_bbb_pair_fails_preflight(self):
        """Require a BBB source timestamp instead of inventing a new one."""
        untimestamped = self.bbb.with_name("bbb.gpkg")
        self.write_file(untimestamped)
        self.write_file(untimestamped.with_suffix(".csv"))
        with self.assertRaisesRegex(ValueError, "_YYYYMMDD_HHMMSS timestamp"):
            self.run_wrapper("--bbb-results", str(untimestamped))
        self.assertEqual(self.stage_calls, [])
        self.assertFalse(self.results.exists())

    def test_failed_toolkit_does_not_run_county_builder(self):
        """Leave a failure record without continuing or copying BBB."""
        self.fail_at = "runner.py"
        with self.assertRaises((subprocess.CalledProcessError, RuntimeError)):
            self.run_wrapper()
        self.assertEqual(self.stage_calls[-1][0], "runner.py")
        self.assertFalse((self.results / "counties").exists())
        self.assertFalse((self.results / "bbb_candidate_blm_lands").exists())
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8"))["status"], "failed")

    def test_toolkit_outputs_need_matching_timestamps(self):
        """Do not combine files from different jobs or runs even on exit zero."""
        self.mismatch_runner_csv = True
        with self.assertRaises((FileNotFoundError, RuntimeError, ValueError)):
            self.run_wrapper()
        self.assertEqual(self.stage_calls[-1][0], "runner.py")
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8"))["status"], "failed")

    def test_existing_run_is_not_overwritten(self):
        """Keep previous results even when the caller chooses the same run ID."""
        self.write_file(self.manifest, "prior run")
        with self.assertRaises((FileExistsError, ValueError)):
            self.run_wrapper()
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "prior run")
        self.assertEqual(self.stage_calls, [])

    def test_frozen_config_rejects_wrong_cut(self):
        """Do not run public zonals with the all-lands vector."""
        self.wrong_config_cut = True
        with self.assertRaisesRegex(ValueError, "Frozen config paths for public"):
            self.run_wrapper()
        self.assertEqual(self.stage_calls[-1][0], "prepare_tif_zonal_stats_config.py")
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
