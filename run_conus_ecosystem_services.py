"""Run the CONUS Ecosystem Services Assessment and publish its output layers."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
import pyogrio
from tqdm import tqdm

from assessment import REPO_ROOT as ROOT
from assessment.assessment_geography import PADUS_ANALYSIS_LAYER, PADUS_ANALYSIS_PATH
from assessment.assessment_raster_zonals import (
    completed_zone_raster_jobs,
    freeze_zone_zonals,
    read_frozen_zone_raster_plan,
    read_zone_raster_sums,
    validate_additional_raster_inputs,
    validate_zone_raster_inputs,
)
from assessment.assessment_valuation import CONFIG_FILES, load_valuation_config, value_zones, write_zone_pair
from assessment.assessment_zone_geometry import ZONE_PRODUCTS
from assessment.build_county_ecosystem_service_valuations_2026 import BASE_FIELDS, sha256
from assessment.prepare_tif_zonal_stats_config import DEFAULT_CONFIG
from assessment.prepare_all_pad_zones import (
    ALL_PAD_PRODUCT, PADUS_FULL_LAYER, PADUS_FULL_PATH, prepare_all_pad_zones,
)


COUNTY_PATH = Path("data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_PREPARE = "Prepare six assessment zones"
_FREEZE = "Freeze raw raster zonal jobs"
_RUN_RASTERS = "Run and verify raster zonals"
_REUSE_RASTERS = "Verify reused raster zonals"
_PREPARE_PAD = "Prepare individual PAD-US county features"
_FREEZE_PAD = "Freeze individual PAD-US raster jobs"
_RUN_PAD = "Run and verify individual PAD-US raster zonals"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the assessment inputs and worker limits.

    Args:
        argv: Argument strings, or None to use the command line.

    Returns:
        Selected inputs and execution options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, help="Path to zonal_stats_toolkit/runner.py for new raster jobs")
    parser.add_argument("--bbb-by-county", type=Path, help="Prepared BBB county geometry GeoPackage for a fresh run")
    parser.add_argument("--bbb-layer")
    parser.add_argument("--padus-path", type=Path, default=PADUS_ANALYSIS_PATH)
    parser.add_argument("--padus-layer", default=PADUS_ANALYSIS_LAYER)
    parser.add_argument("--counties", type=Path, default=COUNTY_PATH)
    parser.add_argument("--counties-layer")
    parser.add_argument("--land-mask", type=Path, help="Census county land polygons keyed by GEOID for a fresh run")
    parser.add_argument("--land-mask-layer", help="Layer in the land-mask file")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-run", metavar="RUN_ID",
                        help="Resume a failed all-PAD raster stage using its verified prepared features")
    parser.add_argument("--job-workers", type=int, default=None,
                        help="Legacy toolkit TaskGraph workers (default: 2)")
    parser.add_argument("--raster-workers", type=int, default=None,
                        help="Legacy raster workers per group (default: 2)")
    parser.add_argument("--zonal-workers", type=int,
                        help="Total compute processes shared by all feature groups and rasters")
    parser.add_argument("--workers", type=int, default=4,
                        help="Zone-preparation workers and shared non-zonal valuation processes (default: 4)")
    parser.add_argument("--geometry-workers", type=int, default=None,
                        help="Compatibility override for the entire shared valuation pool, not a nested pool")
    parser.add_argument(
        "--reuse-zonals-from", metavar="RUN_ID",
        help="Reuse a prior run's prepared zones and verified raster sums",
    )
    parser.add_argument("--include-all-pad", action="store_true",
                        help="Add original PAD-US features cut by county; run only their new zonals when reusing")
    parser.add_argument("--all-pad-path", type=Path, default=PADUS_FULL_PATH,
                        help="Full original PAD-US geodatabase for --include-all-pad")
    parser.add_argument("--all-pad-layer", default=PADUS_FULL_LAYER,
                        help="Original combined PAD-US source layer, preserving all its attributes")
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without creating run files")
    args = parser.parse_args(argv)
    if args.resume_run:
        if (args.run_id is not None or args.reuse_zonals_from or args.include_all_pad
                or any(value is not None for value in (args.bbb_by_county, args.bbb_layer,
                    args.land_mask, args.land_mask_layer, args.counties_layer))
                or (args.padus_path, args.padus_layer, args.counties, args.all_pad_path, args.all_pad_layer)
                != (PADUS_ANALYSIS_PATH, PADUS_ANALYSIS_LAYER, COUNTY_PATH, PADUS_FULL_PATH, PADUS_FULL_LAYER)):
            parser.error("--resume-run uses the saved run ID and inputs; omit run/source/zone overrides")
        if args.runner is None:
            parser.error("--resume-run requires --runner")
        args.run_id = args.resume_run
    elif args.run_id is None:
        args.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.zonal_workers is not None:
        if args.zonal_workers < 1:
            parser.error("--zonal-workers must be positive")
        if args.job_workers is not None or args.raster_workers is not None:
            parser.error("--zonal-workers replaces --job-workers and --raster-workers; omit those flags")
        if args.reuse_zonals_from and not args.include_all_pad:
            parser.error("--zonal-workers requires new raster jobs; omit it for valuation-only reuse")
    args.job_workers = 2 if args.job_workers is None else args.job_workers
    args.raster_workers = 2 if args.raster_workers is None else args.raster_workers
    if args.reuse_zonals_from is None and not args.resume_run:
        if any(value is None for value in (args.runner, args.bbb_by_county, args.land_mask)):
            parser.error("a fresh run requires --runner, --bbb-by-county, and --land-mask")
    elif (any(value is not None for value in (args.bbb_by_county, args.bbb_layer,
                                              args.land_mask, args.land_mask_layer, args.counties_layer))
          or (args.padus_path, args.padus_layer, args.counties) != (PADUS_ANALYSIS_PATH, PADUS_ANALYSIS_LAYER, COUNTY_PATH)):
        parser.error("--reuse-zonals-from uses the original geometry inputs; omit zone overrides")
    if args.include_all_pad and args.runner is None:
        parser.error("--include-all-pad requires --runner for the new feature zonals")
    if args.reuse_zonals_from and args.runner is not None and not args.include_all_pad:
        parser.error("--runner with --reuse-zonals-from requires --include-all-pad")
    if not args.include_all_pad and (args.all_pad_path, args.all_pad_layer) != (PADUS_FULL_PATH, PADUS_FULL_LAYER):
        parser.error("--all-pad-path and --all-pad-layer require --include-all-pad")
    return args


def zonal_worker_options(args: argparse.Namespace) -> list[str]:
    """Select one toolkit execution mode without combining worker budgets."""
    if args.zonal_workers is not None:
        return ["--total-workers", str(args.zonal_workers)]
    return ["--job-workers", str(args.job_workers), "--raster-workers", str(args.raster_workers)]


def validate_shared_zonal_runner(runner: Path, root: Path) -> None:
    """Require advertised shared-pool support before preparing new geometries."""
    try:
        result = subprocess.run(
            [sys.executable, "-B", str(runner), "--help"], cwd=root,
            check=True, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"Cannot check shared-pool support in {runner}; update your zonal_stats_toolkit checkout") from error
    if not re.search(r"(?<![\w-])--total-workers(?![\w-])", result.stdout):
        raise ValueError(
            f"Toolkit runner {runner} does not support --total-workers; "
            "update your zonal_stats_toolkit checkout before using --zonal-workers"
        )


def read_prepared_zones(zone_root: Path, run_id: str) -> tuple[dict[str, tuple[Path, str]], set[str]]:
    """Resolve the exact six outputs of a CONUS zone-preparation run.

    Args:
        zone_root: The original run's zonal_units directory.
        run_id: The original geometry run identifier.

    Returns:
        Product paths and layers, plus the manager products requiring shards.

    Raises:
        ValueError: The manifest or recorded output paths do not match this run.
    """
    manifest = json.loads((zone_root / "zone_manifest.json").read_text(encoding="utf-8"))
    if (manifest["run_id"] != run_id or manifest.get("geographic_scope") != "CONUS"
            or set(manifest["outputs"]) != set(ZONE_PRODUCTS)):
        raise ValueError("Zone manifest does not identify this run's six CONUS products")
    zones = {}
    for product in ZONE_PRODUCTS:
        path = zone_root / product / f"{product}_{run_id}.gpkg"
        output = manifest["outputs"][product]
        if Path(output["path"]).resolve() != path.resolve() or output["layer"] != product or not path.is_file():
            raise ValueError(f"Unexpected zone path or layer for {product}")
        zones[product] = (path, product)
    overlap = manifest["counties_with_manager_overlap"]
    return zones, {
        f"{name}_lands_by_county_and_manager_type" for name in ("public", "conservation")
        if overlap[f"area_ha_{name}_manager_overlap"] > 0
    }


def read_vector_inputs(args: argparse.Namespace, root: Path) -> dict[str, tuple[str, str]]:
    """Resolve and validate the vector inputs used by a fresh assessment.

    Args:
        args: Fresh-run vector arguments.
        root: Repository root for relative paths.

    Returns:
        Absolute paths and selected layer names keyed by source.
    """
    vector_inputs = {
        "padus": (args.padus_path, args.padus_layer, {"Mang_Type", "Des_Tp", "GAP_Sts"}),
        "counties": (args.counties, args.counties_layer, {"GEOID", "ALAND"}),
        "land_mask": (args.land_mask, args.land_mask_layer, {"GEOID"}),
        "bbb_by_county": (args.bbb_by_county, args.bbb_layer, {"GEOID"}),
        # Keep the original national reference even if --counties is already
        # a CONUS subset; overseas coral points must not move to the mainland.
        "coral_reference_counties": (COUNTY_PATH, None, {"GEOID"}),
    }
    input_paths = {}
    for key, (path, layer, fields) in vector_inputs.items():
        path = (root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{key} input not found: {path}")
        if layer is None:
            spatial_layers = [name for name, geom_type in pyogrio.list_layers(path) if geom_type is not None]
            if len(spatial_layers) != 1:
                raise ValueError(f"Select a layer for {key}: {spatial_layers}")
            layer = spatial_layers[0]
        info = pyogrio.read_info(path, layer=layer)
        if fields.difference(info["fields"]) or not info["crs"]:
            raise ValueError(f"{path} / {layer} must have a CRS and the fields {sorted(fields)}")
        if key == "land_mask" and info["geometry_type"] not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"{path} / {layer} must be a polygon land-mask layer")
        input_paths[key] = (str(path), str(layer))
    return input_paths


@dataclass
class SavedZonals:
    """Validated original and optional feature-level raster batches."""

    manifest_path: Path
    manifest: dict
    base_run_id: str
    base_manifest: dict
    zones: dict[str, tuple[Path, str]]
    sums: dict[str, pd.DataFrame]
    overlap: set[str]
    empty_products: tuple[str, ...]
    frozen_configs: tuple[Path, ...]
    all_pad: dict | None = None


def all_pad_paths(root: Path, run_id: str) -> tuple[Path, Path, Path, Path]:
    """Return isolated paths for one run's additional feature raster batch."""
    work = root / "data/processing_outputs/assessment_runs" / run_id
    zone = work / "zonal_units" / ALL_PAD_PRODUCT / f"{ALL_PAD_PRODUCT}_{run_id}.gpkg"
    frozen = root / DEFAULT_CONFIG.relative_to(ROOT).parent / "runs" / f"snapp_all_pad_{run_id}.yaml"
    return zone, frozen, work / "all_pad_zonal_stats_work", work / "all_pad_raw_zonals"


def read_saved_zonals(
    root: Path, run_id: str, seen: frozenset[str] = frozenset(), *, include_all_pad: bool = True,
) -> SavedZonals:
    """Follow saved provenance and verify each raster batch once, without rerunning it."""
    if not _RUN_ID.fullmatch(run_id) or run_id.casefold() in seen:
        raise ValueError("Invalid or cyclic saved raster provenance")
    seen = seen | {run_id.casefold()}
    manifest_path = root / "data/analysis_results/assessment_runs" / run_id / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    if (manifest.get("run_id") != run_id or manifest.get("geographic_scope") != "CONUS"
            or manifest.get("status") not in {"complete", "failed"}):
        raise ValueError(f"Source run {run_id} has no completed, verified CONUS raster zonals")
    parent = manifest.get("reused_zonals_from")
    if parent is not None:
        if stages.get(_REUSE_RASTERS) != "complete":
            raise ValueError(f"Source run {run_id} has no completed, verified CONUS raster zonals")
        parent_id = parent["run_id"]
        expected = root / "data/analysis_results/assessment_runs" / parent_id / "run_manifest.json"
        if Path(parent.get("manifest", "")).resolve() != expected.resolve():
            raise ValueError("Unexpected reused manifest path")
        saved = read_saved_zonals(root, parent_id, seen)
        saved.manifest_path, saved.manifest = manifest_path, manifest
        for field in ("zone_manifest", "frozen_zonal_config", "raw_zonal_directory"):
            if Path(manifest.get(field, "")).resolve() != Path(saved.base_manifest[field]).resolve():
                raise ValueError(f"Source run {run_id} records an unexpected {field} path")
    else:
        if any(stages.get(stage) != "complete" for stage in (_PREPARE, _FREEZE, _RUN_RASTERS)):
            raise ValueError(f"Source run {run_id} has no completed, verified CONUS raster zonals")
        work = root / "data/processing_outputs/assessment_runs" / run_id
        zone_root = work / "zonal_units"
        frozen = root / DEFAULT_CONFIG.relative_to(ROOT).parent / "runs" / f"snapp_six_zone_{run_id}.yaml"
        raw = work / "raw_zonals"
        for field, expected in {"zone_manifest": zone_root / "zone_manifest.json",
                                "frozen_zonal_config": frozen, "raw_zonal_directory": raw}.items():
            if Path(manifest.get(field, "")).resolve() != expected.resolve():
                raise ValueError(f"Source run {run_id} records an unexpected {field} path")
        zones, overlap = read_prepared_zones(zone_root, run_id)
        plan = read_frozen_zone_raster_plan(
            zones, frozen, work / "zonal_stats_work", raw,
            expected_raster_columns=tuple(BASE_FIELDS.values()), overlapping_manager_products=overlap,
        )
        saved = SavedZonals(manifest_path, manifest, run_id, manifest, zones,
                            read_zone_raster_sums(plan), overlap, tuple(plan.skipped_products), (frozen,))
    if not include_all_pad:
        return saved
    additional = manifest.get("all_pad_zonals")
    if additional != saved.all_pad:
        if saved.all_pad is not None or additional is None:
            raise ValueError("Saved run changes or drops inherited PAD-US raster provenance")
        if (additional.get("run_id") != run_id
                or any(stages.get(stage) != "complete" for stage in (_PREPARE_PAD, _FREEZE_PAD, _RUN_PAD))):
            raise ValueError("PAD-US feature raster batch has not completed")
        zone, frozen, work, raw = all_pad_paths(root, run_id)
        for field, expected in {"zone": zone, "frozen_config": frozen,
                                "work_directory": work, "raw_directory": raw}.items():
            if Path(additional.get(field, "")).resolve() != expected.resolve():
                raise ValueError(f"Unexpected PAD-US {field} path")
        for field, path in (("zone_sha256", zone), ("frozen_sha256", frozen)):
            if additional.get(field) != sha256(path):
                raise ValueError(f"Saved PAD-US {field} differs from its verified raster batch")
        extra_zones = {ALL_PAD_PRODUCT: (zone, ALL_PAD_PRODUCT)}
        plan = read_frozen_zone_raster_plan(
            extra_zones, frozen, work, raw, expected_raster_columns=tuple(BASE_FIELDS.values()),
            products=(ALL_PAD_PRODUCT,), overlapping_manager_products=(),
        )
        if additional.get("resume_batches"):
            validate_pad_job_batches(plan, additional)
        saved.zones = saved.zones | extra_zones
        saved.sums = saved.sums | read_zone_raster_sums(
            plan, **({"allow_mixed_timestamps": True} if additional.get("resume_batches") else {}),
        )
        saved.empty_products += tuple(plan.skipped_products)
        saved.frozen_configs += (frozen,)
        saved.all_pad = additional
    return saved


def verify_pad_raster_sources(plan, additional: dict, *, legacy: bool = False) -> dict[str, str]:
    """Hash frozen TIFF sources, retaining a verified baseline for resumed jobs."""
    config = ConfigParser(interpolation=None)
    config.read(plan.frozen_config, encoding="utf-8")
    project = config["project"]
    paths = sorted({(plan.frozen_config.parent / entry["analysis_path"]).resolve()
                    for entry in json.loads(project["snapp_rasters"])})
    previous = additional.get("source_raster_sha256")
    if previous is not None and set(previous) != {str(path) for path in paths}:
        raise ValueError("Recorded source raster paths differ from the frozen plan")
    cutoff = datetime.fromisoformat(project["snapp_created_utc"]).timestamp()
    current = {}
    for path in tqdm(paths, desc="Verify all-PAD raster inputs", unit="raster"):
        before = path.stat()
        if legacy and previous is None and before.st_mtime > cutoff:
            raise ValueError(f"Raster was modified after the original plan was frozen: {path}")
        current[str(path)] = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Raster changed while verifying its content: {path}")
    if previous is not None and previous != current:
        raise ValueError("Source raster content changed since the recorded all-PAD batch")
    return current


def validate_pad_job_batches(plan, additional: dict, *, capture: bool = False):
    """Require each retained result timestamp to belong to its recorded job batch."""
    observed = {}
    for job in plan.jobs:
        paths = list(job.output_base.parent.glob(f"{job.output_base.name}_*.csv"))
        if len(paths) == 1:
            observed[job.output_base.name] = paths[0].stem.removeprefix(job.output_base.name + "_")
    batches = additional.get("resume_batches", [])
    if not batches:
        return completed_zone_raster_jobs(plan)
    if capture:
        stamps = {}
        for job in plan.jobs:
            if job.output_base.name not in batches[-1]["jobs"]:
                continue
            try:
                valid = completed_zone_raster_jobs(
                    replace(plan, jobs=(job,)), check_unexpected_outputs=False,
                )
            except Exception:
                continue  # The full-plan validation below still reports this invalid pair.
            if valid:
                stamps[job.output_base.name] = observed[job.output_base.name]
        if len(set(stamps.values())) > 1:
            raise ValueError("Completed jobs from one resume batch have different timestamps")
        batches[-1]["job_timestamps"] = stamps
    expected = dict(additional["retained_job_timestamps"])
    job_names = {job.output_base.name for job in plan.jobs}
    for batch in batches:
        selected = set(batch["jobs"])
        stamps = batch["job_timestamps"]
        if (not selected.issubset(job_names) or not set(stamps).issubset(selected)
                or set(stamps).intersection(expected) or len(set(stamps.values())) > 1):
            raise ValueError("Invalid job/timestamp provenance for resumed toolkit batch")
        expected.update(stamps)
    complete = completed_zone_raster_jobs(plan, expected_timestamps=expected)
    return complete


def read_resumable_all_pad(root: Path, run_id: str):
    """Verify a failed raster batch and identify jobs without a completed pair."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid resume run ID")
    results = root / "data/analysis_results/assessment_runs" / run_id
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    if (manifest.get("run_id") != run_id or manifest.get("status") != "failed"
            or manifest.get("geographic_scope") != "CONUS"
            or any(stages.get(stage) != "complete" for stage in (_PREPARE_PAD, _FREEZE_PAD))
            or stages.get(_RUN_PAD) != "failed"):
        raise ValueError("Resume requires a failed all-PAD raster stage with completed preparation and freezing")
    if manifest.get("outputs") or any(
        path.suffix.lower() in {".csv", ".gpkg"} for path in results.rglob("*")
    ):
        raise FileExistsError("Resume cannot replace existing final assessment outputs")
    hashes = manifest.get("configuration_sha256", {})
    if not hashes or any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Assessment configuration changed since the failed run")
    saved = read_saved_zonals(root, run_id, include_all_pad=False)
    if saved.all_pad is not None:
        raise ValueError("Cannot resume an all-PAD batch inherited from another run")
    additional = manifest.get("all_pad_zonals", {})
    zone, frozen, work, raw = all_pad_paths(root, run_id)
    if additional.get("run_id") != run_id:
        raise ValueError("All-PAD provenance does not identify this run")
    for field, path in {"zone": zone, "frozen_config": frozen,
                        "work_directory": work, "raw_directory": raw}.items():
        if Path(additional.get(field, "")).resolve() != path.resolve():
            raise ValueError(f"Unexpected saved all-PAD {field} path")
    for field, path in (("zone_sha256", zone), ("frozen_sha256", frozen)):
        if additional.get(field) != sha256(path):
            raise ValueError(f"Saved all-PAD {field} changed since preparation")
    plan = read_frozen_zone_raster_plan(
        {ALL_PAD_PRODUCT: (zone, ALL_PAD_PRODUCT)}, frozen, work, raw,
        expected_raster_columns=tuple(BASE_FIELDS.values()),
        products=(ALL_PAD_PRODUCT,), overlapping_manager_products=(),
    )
    if additional.get("raster_jobs") != len(plan.jobs):
        raise ValueError("Saved all-PAD job count differs from its frozen plan")
    additional["source_raster_hash_origin"] = additional.get(
        "source_raster_hash_origin", "resume_preflight" if "source_raster_sha256" not in additional
        else "recorded_batch",
    )
    additional["source_raster_sha256"] = verify_pad_raster_sources(plan, additional, legacy=True)
    complete = validate_pad_job_batches(plan, additional)
    if not additional.get("resume_batches"):
        additional["retained_job_timestamps"] = {
            job.output_base.name: next(job.output_base.parent.glob(
                f"{job.output_base.name}_*.csv")).stem.removeprefix(job.output_base.name + "_")
            for job in complete
        }
    missing = tuple(job for job in plan.jobs if job not in complete)
    return manifest, saved, plan, missing


@contextmanager
def _resume_lock(path: Path):
    """Reject another active resume; kernel locks release if the process exits."""
    with path.open("a+b") as lock:
        if path.stat().st_size == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another process is already resuming this assessment") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def write_resume_config(plan, missing):
    """Write a new execution plan beside the unchanged frozen source config."""
    config = ConfigParser(interpolation=None)
    config.read(plan.frozen_config, encoding="utf-8")
    selected = {f"job:{job.output_base.name}" for job in missing}
    for section in config.sections():
        if section != "project" and section not in selected:
            config.remove_section(section)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = plan.frozen_config.with_name(f"{plan.frozen_config.stem}_resume_{stamp}.yaml")
    config["project"]["name"] = path.stem
    with path.open("x", encoding="utf-8") as stream:
        config.write(stream)
    return path


def run_assessment(args: argparse.Namespace, root: Path = ROOT) -> Path | None:
    """Value and publish a fresh assessment or reuse completed raster zonals.

    Args:
        args: Parsed inputs and worker limits.
        root: Repository directory containing code, configuration and data.

    Returns:
        The final results directory, or None for read-only preflight.

    Raises:
        ValueError: Inputs or stage handoffs do not meet the run contract.
        FileExistsError: This run ID is already in use.
        subprocess.CalledProcessError: An analysis stage failed.
    """
    if not _RUN_ID.fullmatch(args.run_id):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens")
    resuming = bool(args.resume_run)
    source_id = args.run_id if resuming else args.reuse_zonals_from
    if source_id is not None and (not _RUN_ID.fullmatch(source_id)
                                  or (not resuming and source_id.casefold() == args.run_id.casefold())):
        raise ValueError("Reuse requires a valid source run ID distinct from the new run ID")
    valuation_workers = args.workers if args.geometry_workers is None else args.geometry_workers
    if min(args.job_workers, args.raster_workers, args.workers, valuation_workers) < 1:
        raise ValueError("Worker limits must be positive")
    if args.zonal_workers is not None and args.zonal_workers < 1:
        raise ValueError("--zonal-workers must be positive")
    root = root.resolve()
    resume_manifest_digest = sha256(
        root / "data/analysis_results/assessment_runs" / args.run_id / "run_manifest.json"
    ) if resuming else None
    resume_state = read_resumable_all_pad(root, args.run_id) if resuming else None
    data = root / "data"
    config_dir = data / "workflow_assets/county_integration"
    zonal_config = root / DEFAULT_CONFIG.relative_to(ROOT)
    paths, _, _, derived = load_valuation_config(data, config_dir)
    results = data / "analysis_results/assessment_runs" / args.run_id
    new_work = data / "processing_outputs/assessment_runs" / args.run_id
    new_frozen = zonal_config.parent / "runs" / f"snapp_six_zone_{args.run_id}.yaml"
    pad_zone, pad_frozen, pad_work, pad_raw = all_pad_paths(root, args.run_id)
    if not args.validate_only and not resuming:
        for path in (results, new_work, new_frozen, pad_frozen):
            if path.exists():
                raise FileExistsError(f"Run path already exists; use a new --run-id: {path}")
    geometry_run = args.run_id if source_id is None else source_id
    work = new_work.with_name(geometry_run)
    frozen = new_frozen.with_name(f"snapp_six_zone_{geometry_run}.yaml")
    zone_root = work / "zonal_units"
    raw_root = work / "raw_zonals"
    work_dir = work / "zonal_stats_work"
    config_files = [config_dir / filename for filename in CONFIG_FILES]
    config_files.append(paths["air_quality_county_aliases"])
    gdp_crosswalk = derived["sources"]["gdp"].get("allocation_crosswalk")
    if gdp_crosswalk:
        config_files.append((data / gdp_crosswalk).resolve())
    if source_id is None or args.include_all_pad or resuming:
        runner = (root / args.runner).resolve()
        if not runner.is_file():
            raise FileNotFoundError(f"Toolkit runner not found: {runner}")
        if args.zonal_workers is not None:
            validate_shared_zonal_runner(runner, root)
        rasters, blocked = validate_zone_raster_inputs(zonal_config)
        raster_fields = {f"sum_{raster.path.stem}" for raster in rasters}
        expected_fields = set(BASE_FIELDS.values())
        if raster_fields != expected_fields:
            raise ValueError(
                f"TIFF inventory does not match the configured valuation services: "
                f"missing {sorted(expected_fields - raster_fields)}, extra {sorted(raster_fields - expected_fields)}"
            )
    if source_id is None:
        input_paths = read_vector_inputs(args, root)
        config_files.extend((zonal_config, root / "config/padus_land_rules.rules"))
    else:
        saved = resume_state[1] if resuming else read_saved_zonals(root, source_id)
        if args.include_all_pad and saved.all_pad is not None:
            raise ValueError("Source run already contains all-PAD zonals; omit --include-all-pad and --runner to reuse them")
        source_manifest_path = saved.manifest_path
        source_manifest = saved.base_manifest
        geometry_run = saved.base_run_id
        work = new_work.with_name(geometry_run)
        zone_root, work_dir, raw_root = work / "zonal_units", work / "zonal_stats_work", work / "raw_zonals"
        frozen = saved.frozen_configs[0]
        zones, raster_sums, overlapping_products = saved.zones, saved.sums, saved.overlap
        input_paths = source_manifest["zone_inputs"]
        coral_path, coral_layer = input_paths["coral_reference_counties"]
        if Path(coral_path).resolve() != (root / COUNTY_PATH).resolve():
            raise ValueError(f"Source run {source_id} records an unexpected coral county reference")
        info = pyogrio.read_info(coral_path, layer=coral_layer)
        if not info["crs"] or "GEOID" not in info["fields"]:
            raise ValueError("Coral county reference must have a CRS and GEOID")
        blocked = source_manifest["blocked_rasters"]
        config_files.extend(saved.frozen_configs)
    if args.include_all_pad:
        full_pad_path = (root / args.all_pad_path).resolve()
        if not full_pad_path.exists():
            raise FileNotFoundError(f"Full PAD-US source not found: {full_pad_path}")
        info = pyogrio.read_info(full_pad_path, layer=args.all_pad_layer)
        if (not info["crs"] or "Mang_Type" not in info["fields"]
                or info["geometry_type"] not in {"Polygon", "MultiPolygon"}):
            raise ValueError("Full PAD-US source must contain polygon geometry, a CRS, and Mang_Type")
        mask_path, mask_layer = input_paths["land_mask"]
        info = pyogrio.read_info(mask_path, layer=mask_layer)
        if not info["crs"] or "GEOID" not in info["fields"]:
            raise ValueError("Original county land mask must contain a CRS and GEOID")
        if source_id is not None:
            validate_additional_raster_inputs(zonal_config, frozen)
        config_files.append(zonal_config)
    if resuming:
        config_files.append(zonal_config)
        validate_additional_raster_inputs(zonal_config, frozen)
    hashes = {str(path): sha256(path) for path in config_files}
    if resuming:
        recorded = resume_state[0]["configuration_sha256"]
        if any(recorded.get(path) != digest for path, digest in hashes.items()):
            raise ValueError("Current assessment configuration differs from the failed run")
        hashes = {**recorded, **hashes, str(pad_frozen): sha256(pad_frozen)}
    if args.validate_only:
        detail = (f"prepared all-PAD features and {len(resume_state[2].jobs)} shards; "
                  f"{len(resume_state[2].jobs) - len(resume_state[3])} completed raster jobs retained, "
                  f"{len(resume_state[3])} raster jobs remaining"
                  if resuming else "reused zones and raster sums" if source_id is not None else "inputs")
        print(f"CONUS Ecosystem Services Assessment: {detail} validated; no run files written.", flush=True)
        return None

    guard = _resume_lock(new_work / ".resume.lock") if resuming else nullcontext()
    with guard:
        if resuming:
            current = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
            if (current.get("status") != "failed"
                    or sha256(results / "run_manifest.json") != resume_manifest_digest):
                raise RuntimeError("Run state changed during resume preflight; validate it again")
            if current.get("outputs") or any(
                path.suffix.lower() in {".csv", ".gpkg"} for path in results.rglob("*")
            ):
                raise FileExistsError("Resume cannot replace existing final assessment outputs")
        results.mkdir(parents=True, exist_ok=resuming)
        manifest_path = results / "run_manifest.json"
        manifest = {
            "run_id": args.run_id, "status": "running",
            "geographic_scope": "CONUS",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "zone_inputs": input_paths,
            "valuation_inputs": {name: str(path) for name, path in paths.items()
                                 if name not in {"base", "coastline", "freshwater"}},
            "county_denominator_inputs": {name: str((data / spec["path"]).resolve())
                                          for name, spec in derived["sources"].items()},
            "configuration_sha256": hashes,
            "frozen_zonal_config": str(frozen),
            "zone_manifest": str(zone_root / "zone_manifest.json"),
            "county_area_qa": str(zone_root / "county_area_qa.csv"),
            "raw_zonal_directory": str(raw_root),
            "blocked_rasters": blocked,
            "valuation_execution": {"executor": "shared_process_pool", "workers": valuation_workers,
                                    "native_threads_per_worker": 1},
            "zonal_execution": (
                {"executor": "reuse", "new_raster_jobs": False}
                if source_id is not None and not args.include_all_pad and not resuming else
                {"executor": "shared_process_pool", "total_workers": args.zonal_workers,
                 "task_unit": "feature_group_and_raster", "native_threads_per_worker": 1}
                if args.zonal_workers is not None else
                {"executor": "sequential_groups", "job_workers": args.job_workers,
                 "raster_workers": args.raster_workers}
            ),
            "stages": {}, "outputs": {},
        }
        if source_id is not None:
            manifest["reused_zonals_from"] = {"run_id": source_id, "manifest": str(source_manifest_path)}
            if saved.all_pad is not None:
                manifest["all_pad_zonals"] = saved.all_pad
        if resuming:
            current_execution = {key: manifest[key] for key in ("valuation_execution", "zonal_execution")}
            manifest = resume_state[0]
            manifest.setdefault("resume_attempts", []).append({
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "previous_error": manifest.get("error"),
                "previous_finished_utc": manifest.get("finished_utc"),
            })
            manifest.update(status="running", configuration_sha256=hashes, **current_execution)
            manifest.pop("error", None)
            manifest.pop("finished_utc", None)

        def record(stage: str, status: str) -> None:
            """Persist stage status and the currently verified outputs."""
            manifest["stages"][stage] = status
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"{stage}: {status}", flush=True)

        env = os.environ.copy()
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        stage = _PREPARE if source_id is None else _REUSE_RASTERS
        try:
            if source_id is None:
                record(stage, "running")
                subprocess.run([
                    sys.executable, "-B", "-m", "assessment.prepare_assessment_zones",
                    "--padus-path", input_paths["padus"][0], "--padus-layer", input_paths["padus"][1],
                    "--counties", input_paths["counties"][0], "--counties-layer", input_paths["counties"][1],
                    "--land-mask", input_paths["land_mask"][0], "--land-mask-layer", input_paths["land_mask"][1],
                    "--bbb-by-county", input_paths["bbb_by_county"][0], "--bbb-layer", input_paths["bbb_by_county"][1],
                    "--rules", str(root / "config/padus_land_rules.rules"),
                    "--run-id", args.run_id, "--output-root", str(work.parent), "--workers", str(args.workers),
                ], cwd=root, env=env, check=True)
                zones, overlapping_products = read_prepared_zones(zone_root, geometry_run)
                manifest["manager_overlap_products"] = sorted(overlapping_products)
                record(stage, "complete")

                stage = _FREEZE
                record(stage, "running")
                plan = freeze_zone_zonals(zones, zonal_config, frozen, work_dir, raw_root,
                                         overlapping_manager_products=overlapping_products)
                manifest["empty_zone_products"] = list(plan.skipped_products)
                record(stage, "complete")

                stage = _RUN_RASTERS
                record(stage, "running")
                if plan.jobs:
                    subprocess.run([
                        sys.executable, "-B", str(runner), str(plan.frozen_config),
                        *zonal_worker_options(args),
                    ], cwd=root, env=env, check=True)
                raster_sums = read_zone_raster_sums(plan)
                record(stage, "complete")
            else:
                manifest["manager_overlap_products"] = sorted(overlapping_products)
                manifest["empty_zone_products"] = list(saved.empty_products)
                record(stage, "complete")

            if args.include_all_pad or resuming:
                if resuming:
                    pad_plan, missing_jobs = resume_state[2:]
                    extra_zones = {ALL_PAD_PRODUCT: (pad_zone, ALL_PAD_PRODUCT)}
                else:
                    stage = _PREPARE_PAD
                    record(stage, "running")
                    provenance = prepare_all_pad_zones(
                        full_pad_path, args.all_pad_layer, *zones["counties"],
                        Path(input_paths["land_mask"][0]), input_paths["land_mask"][1], pad_zone,
                        workers=args.workers,
                    )
                    extra_zones = {ALL_PAD_PRODUCT: (pad_zone, ALL_PAD_PRODUCT)}
                    manifest["all_pad_zonals"] = {
                        "run_id": args.run_id, "zone": str(pad_zone), "zone_sha256": sha256(pad_zone),
                        "frozen_config": str(pad_frozen), "work_directory": str(pad_work),
                        "raw_directory": str(pad_raw), "preparation": provenance,
                    }
                    record(stage, "complete")
                    stage = _FREEZE_PAD
                    record(stage, "running")
                    pad_plan = freeze_zone_zonals(
                        extra_zones, zonal_config, pad_frozen, pad_work, pad_raw,
                        products=(ALL_PAD_PRODUCT,), overlapping_manager_products=(),
                    )
                    manifest["all_pad_zonals"]["frozen_sha256"] = sha256(pad_frozen)
                    manifest["all_pad_zonals"]["raster_jobs"] = len(pad_plan.jobs)
                    record(stage, "complete")
                stage = _RUN_PAD
                record(stage, "running")
                if not resuming:
                    manifest["all_pad_zonals"]["source_raster_sha256"] = verify_pad_raster_sources(
                        pad_plan, manifest["all_pad_zonals"],
                    )
                    manifest["all_pad_zonals"]["source_raster_hash_origin"] = "before_raster_jobs"
                    record(stage, "running")
                execution_config = pad_plan.frozen_config
                if resuming:
                    execution_config = write_resume_config(pad_plan, missing_jobs) if missing_jobs else None
                    batch = {"started_utc": datetime.now(timezone.utc).isoformat(),
                             "jobs": [job.output_base.name for job in missing_jobs],
                             "retained_jobs": len(pad_plan.jobs) - len(missing_jobs),
                             "job_timestamps": {}}
                    if execution_config is not None:
                        batch.update(config=str(execution_config), sha256=sha256(execution_config))
                        hashes[str(execution_config)] = batch["sha256"]
                    manifest["all_pad_zonals"].setdefault("resume_batches", []).append(batch)
                    record(stage, "running")
                if pad_plan.jobs and execution_config is not None:
                    try:
                        subprocess.run([
                            sys.executable, "-B", str(runner), str(execution_config),
                            *zonal_worker_options(args),
                        ], cwd=root, env=env, check=True)
                    except (Exception, KeyboardInterrupt) as error:
                        if resuming:
                            try:
                                validate_pad_job_batches(pad_plan, manifest["all_pad_zonals"], capture=True)
                            except Exception as validation_error:
                                error.add_note(f"Retained-output validation: {validation_error}")
                                batch["output_validation_error"] = str(validation_error)
                        raise
                    else:
                        if resuming:
                            validate_pad_job_batches(pad_plan, manifest["all_pad_zonals"], capture=True)
                            record(stage, "running")
                verify_pad_raster_sources(pad_plan, manifest["all_pad_zonals"])
                raster_sums = raster_sums | read_zone_raster_sums(
                    pad_plan, **({"allow_mixed_timestamps": True} if resuming else {}),
                )
                zones = zones | extra_zones
                manifest["empty_zone_products"].extend(pad_plan.skipped_products)
                record(stage, "complete")

            stage = "Value all seven zone products" if ALL_PAD_PRODUCT in zones else "Value all six zone products"
            record(stage, "running")
            if any(sha256(path) != digest for path, digest in hashes.items()):
                raise ValueError("Assessment configuration changed during the run")
            zone_frames = {
                product: pyogrio.read_dataframe(path, layer=layer)
                for product, (path, layer) in tqdm(zones.items(), desc="Read prepared zones", unit="product")
            }
            coral_path, coral_layer = input_paths["coral_reference_counties"]
            coral_reference = pyogrio.read_dataframe(coral_path, layer=coral_layer, columns=["GEOID"])
            valued, county_audits, coverage = value_zones(
                zone_frames, raster_sums, data, config_dir,
                coral_reference_counties=coral_reference,
                workers=valuation_workers,
            )
            if any(sha256(path) != digest for path, digest in hashes.items()):
                raise ValueError("Assessment configuration changed during valuation")
            manifest["valuation_qa"] = coverage
            record(stage, "complete")

            stage = "Publish seven CSV and GeoPackage pairs" if ALL_PAD_PRODUCT in zones else "Publish six CSV and GeoPackage pairs"
            record(stage, "running")
            for product in tqdm(zones, desc="Publish assessment pairs", unit="product"):
                output = results / f"{product}_{args.run_id}.gpkg"
                csv_path, gpkg_path = write_zone_pair(
                    valued[product], output, product, county_audits if product == "counties" else {},
                )
                manifest["outputs"][product] = {"csv": str(csv_path), "gpkg": str(gpkg_path), "rows": len(valued[product])}
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            manifest["status"] = "complete"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            record(stage, "complete")
        except (Exception, KeyboardInterrupt) as error:
            manifest["status"] = "failed"
            manifest["error"] = str(error) or type(error).__name__
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            record(stage, "failed")
            raise
        print(f"Completed CONUS Ecosystem Services Assessment: {results}", flush=True)
        return results


if __name__ == "__main__":
    run_assessment(parse_args())
