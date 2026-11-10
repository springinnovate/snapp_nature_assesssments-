"""Run the all/public PAD-US and whole-county refresh with a reused BBB pair."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from assessment import REPO_ROOT as ROOT
from assessment.assessment_geography import PADUS_ANALYSIS_LAYER, PADUS_ANALYSIS_PATH


PADUS_PRODUCTS = {
    "all": "padus_all_lands",
    "public": "padus_public_lands",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the run inputs and worker limits.

    Args:
        argv: Argument strings, or None to read the command line.

    Returns:
        The selected inputs and options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path, help="Path to zonal_stats_toolkit/runner.py")
    parser.add_argument(
        "--bbb-results", required=True, type=Path,
        help="Existing BBB GPKG and matching CSV ending in _YYYYMMDD_HHMMSS",
    )
    parser.add_argument("--padus-path", type=Path, default=PADUS_ANALYSIS_PATH)
    parser.add_argument("--padus-layer", default=PADUS_ANALYSIS_LAYER)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--job-workers", type=int, default=2)
    parser.add_argument("--raster-workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16, help="County valuation service processes")
    parser.add_argument("--geometry-workers", type=int, default=16, help="County mangrove intersection threads")
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without writing or running analysis")
    return parser.parse_args(argv)


def require_pair(gpkg: Path) -> tuple[Path, Path]:
    """Require a nonempty CSV/GeoPackage pair with matching stems.

    Args:
        gpkg: The explicitly selected GeoPackage.

    Returns:
        CSV and GeoPackage paths, in that order.

    Raises:
        ValueError: The selected path does not end in .gpkg.
        FileNotFoundError: Either file is missing or empty.
    """
    if gpkg.suffix.lower() != ".gpkg":
        raise ValueError(f"Expected a GeoPackage path: {gpkg}")
    pair = (gpkg.with_suffix(".csv"), gpkg)
    for path in pair:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty file in output pair: {path}")
    return pair


def only_output(directory: Path, pattern: str) -> Path:
    """Find a single nonempty output in the current run's directory.

    Args:
        directory: Isolated directory for this stage's outputs.
        pattern: The expected filename pattern.

    Returns:
        The matching output path.

    Raises:
        RuntimeError: The stage did not produce exactly one nonempty match.
    """
    paths = list(directory.glob(pattern))
    if len(paths) != 1 or not paths[0].is_file() or paths[0].stat().st_size == 0:
        raise RuntimeError(f"Expected one nonempty {pattern} in {directory}; found {len(paths)} match(es)")
    return paths[0]


def run_assessment(args: argparse.Namespace, root: Path = ROOT) -> Path | None:
    """Validate and run the assessment stages, keeping each footprint separate.

    Args:
        args: Inputs and settings from parse_args.
        root: Repository root containing scripts and the data stack.

    Returns:
        The completed results directory, or None for input validation only.

    Raises:
        ValueError: A run option or generated config does not match the workflow.
        FileNotFoundError: A configured input or expected output is unavailable.
        RuntimeError: A stage did not produce an unambiguous output.
        subprocess.CalledProcessError: A child command failed.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens, starting with a letter or digit")
    if min(args.job_workers, args.raster_workers, args.workers, args.geometry_workers) < 1:
        raise ValueError("Worker limits must be positive")

    root = root.resolve()
    data = root / "data"
    runner = (root / args.runner).resolve()
    padus = (root / args.padus_path).resolve()
    bbb_pair = require_pair((root / args.bbb_results).resolve())
    bbb_timestamp = re.search(r"_(\d{8}_\d{6})$", bbb_pair[1].stem)
    if bbb_timestamp is None:
        raise ValueError(f"BBB results must have a _YYYYMMDD_HHMMSS timestamp: {bbb_pair[1]}")
    county_config = data / "workflow_assets/county_integration"
    zonal_config = data / "workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml"
    with (county_config / "sources.csv").open(encoding="utf-8-sig", newline="") as stream:
        county_sources = {row["source"]: data / row["path"] for row in csv.DictReader(stream)}
    metrics = yaml.safe_load((county_config / "derived_metrics.yaml").read_text(encoding="utf-8"))
    template = ConfigParser(interpolation=None)
    with zonal_config.open(encoding="utf-8") as stream:
        template.read_file(stream)
    project = template["project"]
    raster_dir = zonal_config.parent / project["raster_dir"]
    required_files = [
        runner,
        data / "analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg",
        data / "analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg",
        *county_sources.values(),
        *(data / source["path"] for source in metrics["sources"].values()),
        *(county_config / name for name in ("ct_county_crosswalk.csv", "adjustment_factors.csv", "export_rules.yaml")),
        *(raster_dir / raster["filename"] for raster in json.loads(project["raster_inventory"]) if raster["include"]),
    ]
    missing = [path.resolve() for path in required_files if not path.is_file()]
    if not padus.exists():
        missing.append(padus)
    if missing:
        raise FileNotFoundError("Missing assessment inputs:\n" + "\n".join(str(path) for path in missing))

    results = data / "analysis_results/assessment_runs" / args.run_id
    work = data / "processing_outputs/assessment_runs" / args.run_id
    frozen_config = zonal_config.parent / "runs" / f"snapp_assessment_{args.run_id}.yaml"
    if not args.validate_only:
        for path in (results, work, frozen_config):
            if path.exists():
                raise FileExistsError(f"Run path already exists; choose a different --run-id: {path}")

    prep_command = ["-m", "assessment.workflows.prepare_padus_all_and_public_lands"]
    prep_args = [
        "--padus-path", str(padus), "--padus-layer", args.padus_layer, "--products", *PADUS_PRODUCTS,
    ]
    print("Checking the PAD-US layer and selected rules.", flush=True)
    subprocess.run([sys.executable, *prep_command, *prep_args, "--validate-only"], cwd=root, check=True)
    if args.validate_only:
        print("Assessment inputs validated; no run files written.", flush=True)
        return None

    results.mkdir(parents=True)
    work.mkdir(parents=True)
    manifest_path = results / "run_manifest.json"
    manifest = {
        "run_id": args.run_id,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "padus_source": str(padus),
        "padus_layer": args.padus_layer,
        "county_sources_config": str(county_config / "sources.csv"),
        "county_base": str(county_sources["base"].resolve()),
        "reused_bbb": [str(path) for path in bbb_pair],
        "frozen_zonal_config": str(frozen_config),
        "stages": [],
        "outputs": {},
    }

    def run_stage(name: str, command: list[str], *arguments: str) -> None:
        """Run one Python entry point and record its arguments and result."""
        argv = [sys.executable, *command, *arguments]
        stage = {"name": name, "argv": argv, "status": "running"}
        manifest["stages"].append(stage)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"\n{name}", flush=True)
        try:
            subprocess.run(argv, cwd=root, check=True)
        except (Exception, KeyboardInterrupt):
            stage["status"] = "failed"
            raise
        stage["status"] = "complete"

    clipped = work / "padus_clipped_to_usa"
    zones = work / "zonal_units"
    try:
        run_stage("Prepare all/public PAD-US", prep_command, *prep_args, "--output-dir", str(clipped))
        cuts = {}
        for product, folder in PADUS_PRODUCTS.items():
            source = only_output(clipped / f"{product}_lands", f"{folder}_clipped_to_usa_*.gpkg")
            destination = zones / f"{folder}_by_county"
            run_stage(f"Cut {product} lands by county", ["-m", "assessment.workflows.cut_and_flatten_by_county"],
                      str(source), "--output-dir", str(destination))
            cuts[product] = only_output(destination, f"{folder}_clipped_by_county_*.gpkg")

        run_stage(
            "Freeze all/public zonal config", ["-m", "assessment.prepare_tif_zonal_stats_config"],
            "--config", str(zonal_config),
            "--zonal-units-dir", str(zones), "--output", str(frozen_config),
            "--results-dir", str(results), "--work-dir", str(work / "zonal_stats_work"),
            "--derived-raster-dir", str(work / "zonal_stats_derived_rasters"),
            "--zone", "all", "--zone", "public",
        )
        frozen = ConfigParser(interpolation=None)
        with frozen_config.open(encoding="utf-8") as stream:
            frozen.read_file(stream)
        if {section for section in frozen.sections() if section.startswith("job:")} != {
            f"job:{folder}" for folder in PADUS_PRODUCTS.values()
        }:
            raise ValueError("Frozen config must contain exactly the all-land and public-land jobs")
        for product, folder in PADUS_PRODUCTS.items():
            job = frozen[f"job:{folder}"]
            expected = {"agg_vector": cuts[product], "output_csv": results / folder / f"{folder}.csv",
                        "output_gpkg": results / folder / f"{folder}.gpkg"}
            if any((frozen_config.parent / job[key]).resolve() != path.resolve()
                   for key, path in expected.items()):
                raise ValueError(f"Frozen config paths for {product} do not match this run")

        run_stage(
            "Run all/public zonals", [str(runner)], str(frozen_config),
            "--job-workers", str(args.job_workers), "--raster-workers", str(args.raster_workers),
        )
        for folder in PADUS_PRODUCTS.values():
            pair = require_pair(only_output(results / folder, f"{folder}_*.gpkg"))
            manifest["outputs"][folder] = [str(path) for path in pair]

        counties = results / "counties" / f"counties_{args.run_id}.gpkg"
        run_stage(
            "Build whole-county valuations", ["-m", "assessment.build_county_ecosystem_service_valuations_2026"],
            "--data-root", str(data),
            "--config-dir", str(county_config), "--output", str(counties),
            "--workers", str(args.workers), "--geometry-workers", str(args.geometry_workers),
        )
        manifest["outputs"]["counties"] = [str(path) for path in require_pair(counties)]

        bbb_dir = results / "bbb_candidate_blm_lands"
        bbb_dir.mkdir()
        bbb_output = bbb_dir / f"{bbb_dir.name}_{bbb_timestamp[1]}.gpkg"
        for source, destination in zip(bbb_pair, (bbb_output.with_suffix(".csv"), bbb_output)):
            shutil.copy2(source, destination)
        manifest["outputs"][bbb_dir.name] = [
            str(path) for path in require_pair(bbb_output)
        ]
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error) or type(error).__name__
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Completed assessment: {results}", flush=True)
    print(f"Run manifest: {manifest_path}", flush=True)
    return results


if __name__ == "__main__":
    run_assessment(parse_args())
