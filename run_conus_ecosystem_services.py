"""Run the CONUS Ecosystem Services Assessment and publish its output layers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pyogrio
from tqdm import tqdm

from assessment import REPO_ROOT as ROOT
from assessment.assessment_geography import PADUS_ANALYSIS_LAYER, PADUS_ANALYSIS_PATH
from assessment.assessment_raster_zonals import (
    freeze_zone_zonals,
    read_frozen_zone_raster_plan,
    read_zone_raster_sums,
    validate_zone_raster_inputs,
)
from assessment.assessment_valuation import CONFIG_FILES, load_valuation_config, value_zones, write_zone_pair
from assessment.assessment_zone_geometry import ZONE_PRODUCTS
from assessment.build_county_ecosystem_service_valuations_2026 import BASE_FIELDS, sha256
from assessment.prepare_tif_zonal_stats_config import DEFAULT_CONFIG


COUNTY_PATH = Path("data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_PREPARE = "Prepare six assessment zones"
_FREEZE = "Freeze raw raster zonal jobs"
_RUN_RASTERS = "Run and verify raster zonals"
_REUSE_RASTERS = "Verify reused raster zonals"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the assessment inputs and worker limits.

    Args:
        argv: Argument strings, or None to use the command line.

    Returns:
        Selected inputs and execution options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, help="Path to zonal_stats_toolkit/runner.py for a fresh run")
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
    parser.add_argument("--workers", type=int, default=4, help="Concurrent county geometry and tabular workers")
    parser.add_argument("--geometry-workers", type=int, default=4, help="Concurrent fisheries and coastal measurement workers")
    parser.add_argument(
        "--reuse-zonals-from", metavar="RUN_ID",
        help="Reuse a prior run's prepared zones and verified raster sums; run valuation and publication only",
    )
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without creating run files")
    args = parser.parse_args(argv)
    if args.reuse_zonals_from is None:
        if any(value is None for value in (args.runner, args.bbb_by_county, args.land_mask)):
            parser.error("a fresh run requires --runner, --bbb-by-county, and --land-mask")
    elif (any(value is not None for value in (args.runner, args.bbb_by_county, args.bbb_layer,
                                              args.land_mask, args.land_mask_layer, args.counties_layer))
          or (args.padus_path, args.padus_layer, args.counties) != (PADUS_ANALYSIS_PATH, PADUS_ANALYSIS_LAYER, COUNTY_PATH)):
        parser.error("--reuse-zonals-from uses the original geometry inputs; omit runner and zone overrides")
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
    if min(args.job_workers, args.raster_workers, args.workers, args.geometry_workers) < 1:
        raise ValueError("Worker limits must be positive")
    root = root.resolve()
    data = root / "data"
    config_dir = data / "workflow_assets/county_integration"
    zonal_config = root / DEFAULT_CONFIG.relative_to(ROOT)
    paths, _, _, derived = load_valuation_config(data, config_dir)
    results = data / "analysis_results/assessment_runs" / args.run_id
    new_work = data / "processing_outputs/assessment_runs" / args.run_id
    new_frozen = zonal_config.parent / "runs" / f"snapp_six_zone_{args.run_id}.yaml"
    if not args.validate_only:
        for path in (results, new_work, new_frozen):
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
    if source_id is None:
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
        input_paths = read_vector_inputs(args, root)
        config_files.extend((zonal_config, root / "config/padus_land_rules.rules"))
    else:
        source_manifest_path = results.with_name(source_id) / "run_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if (source_manifest.get("run_id") != source_id or source_manifest.get("geographic_scope") != "CONUS"
                or source_manifest.get("status") not in {"complete", "failed"}
                or any(source_manifest.get("stages", {}).get(stage) != "complete"
                       for stage in (_PREPARE, _FREEZE, _RUN_RASTERS))):
            raise ValueError(f"Source run {source_id} has no completed, verified CONUS raster zonals")
        for key, expected in {
            "zone_manifest": zone_root / "zone_manifest.json", "frozen_zonal_config": frozen,
            "raw_zonal_directory": raw_root,
        }.items():
            if Path(source_manifest.get(key, "")).resolve() != expected.resolve():
                raise ValueError(f"Source run {source_id} records an unexpected {key} path")
        input_paths = source_manifest["zone_inputs"]
        coral_path, coral_layer = input_paths["coral_reference_counties"]
        if Path(coral_path).resolve() != (root / COUNTY_PATH).resolve():
            raise ValueError(f"Source run {source_id} records an unexpected coral county reference")
        info = pyogrio.read_info(coral_path, layer=coral_layer)
        if not info["crs"] or "GEOID" not in info["fields"]:
            raise ValueError("Coral county reference must have a CRS and GEOID")
        blocked = source_manifest["blocked_rasters"]
        zones, overlapping_products = read_prepared_zones(zone_root, geometry_run)
        config_files.append(frozen)
    hashes = {str(path): sha256(path) for path in config_files}
    if source_id is not None:
        plan = read_frozen_zone_raster_plan(
            zones, frozen, work_dir, raw_root, expected_raster_columns=tuple(BASE_FIELDS.values()),
            overlapping_manager_products=overlapping_products,
        )
        raster_sums = read_zone_raster_sums(plan)
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
        "stages": {}, "outputs": {},
    }
    if source_id is not None:
        manifest["reused_zonals_from"] = {"run_id": source_id, "manifest": str(source_manifest_path)}

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
            manifest["empty_zone_products"] = list(plan.skipped_products)
            record(stage, "complete")

        stage = "Value all six zone products"
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
            workers=args.workers, geometry_workers=args.geometry_workers,
        )
        if any(sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError("Assessment configuration changed during valuation")
        manifest["valuation_qa"] = coverage
        record(stage, "complete")

        stage = "Publish six CSV and GeoPackage pairs"
        record(stage, "running")
        for product in tqdm(ZONE_PRODUCTS, desc="Publish assessment pairs", unit="product"):
            output = results / product / f"{product}_{args.run_id}.gpkg"
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
