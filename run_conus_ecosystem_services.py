"""Run the CONUS Ecosystem Services Assessment and publish its output layers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--job-workers", type=int, default=2)
    parser.add_argument("--raster-workers", type=int, default=2)
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
    if args.reuse_zonals_from is None:
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


def read_saved_zonals(root: Path, run_id: str, seen: frozenset[str] = frozenset()) -> SavedZonals:
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
        saved.zones = saved.zones | extra_zones
        saved.sums = saved.sums | read_zone_raster_sums(plan)
        saved.empty_products += tuple(plan.skipped_products)
        saved.frozen_configs += (frozen,)
        saved.all_pad = additional
    return saved


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
    source_id = args.reuse_zonals_from
    if source_id is not None and (not _RUN_ID.fullmatch(source_id) or source_id.casefold() == args.run_id.casefold()):
        raise ValueError("Reuse requires a valid source run ID distinct from the new run ID")
    valuation_workers = args.workers if args.geometry_workers is None else args.geometry_workers
    if min(args.job_workers, args.raster_workers, args.workers, valuation_workers) < 1:
        raise ValueError("Worker limits must be positive")
    root = root.resolve()
    data = root / "data"
    config_dir = data / "workflow_assets/county_integration"
    zonal_config = root / DEFAULT_CONFIG.relative_to(ROOT)
    paths, _, _, derived = load_valuation_config(data, config_dir)
    results = data / "analysis_results/assessment_runs" / args.run_id
    new_work = data / "processing_outputs/assessment_runs" / args.run_id
    new_frozen = zonal_config.parent / "runs" / f"snapp_six_zone_{args.run_id}.yaml"
    pad_zone, pad_frozen, pad_work, pad_raw = all_pad_paths(root, args.run_id)
    if not args.validate_only:
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
    if source_id is None or args.include_all_pad:
        runner = (root / args.runner).resolve()
        if not runner.is_file():
            raise FileNotFoundError(f"Toolkit runner not found: {runner}")
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
        saved = read_saved_zonals(root, source_id)
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
    hashes = {str(path): sha256(path) for path in config_files}
    if args.validate_only:
        detail = "reused zones and raster sums" if source_id is not None else "inputs"
        print(f"CONUS Ecosystem Services Assessment: {detail} validated; no run files written.", flush=True)
        return None

    results.mkdir(parents=True)
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
        "stages": {}, "outputs": {},
    }
    if source_id is not None:
        manifest["reused_zonals_from"] = {"run_id": source_id, "manifest": str(source_manifest_path)}
        if saved.all_pad is not None:
            manifest["all_pad_zonals"] = saved.all_pad

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
                    "--job-workers", str(args.job_workers), "--raster-workers", str(args.raster_workers),
                ], cwd=root, env=env, check=True)
            raster_sums = read_zone_raster_sums(plan)
            record(stage, "complete")
        else:
            manifest["manager_overlap_products"] = sorted(overlapping_products)
            manifest["empty_zone_products"] = list(saved.empty_products)
            record(stage, "complete")

        if args.include_all_pad:
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
            if pad_plan.jobs:
                subprocess.run([
                    sys.executable, "-B", str(runner), str(pad_plan.frozen_config),
                    "--job-workers", str(args.job_workers), "--raster-workers", str(args.raster_workers),
                ], cwd=root, env=env, check=True)
            raster_sums = raster_sums | read_zone_raster_sums(pad_plan)
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
