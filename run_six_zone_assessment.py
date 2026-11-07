"""Prepare, value, and publish the six CONUS SNAPP assessment zone products."""

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

from assessment_raster_zonals import (
    freeze_zone_zonals,
    read_zone_raster_sums,
    validate_zone_raster_inputs,
)
from assessment_valuation import CONFIG_FILES, load_valuation_config, value_zones, write_zone_pair
from assessment_zone_geometry import ZONE_PRODUCTS
from build_county_ecosystem_service_valuations_2026 import BASE_FIELDS, sha256
from prepare_tif_zonal_stats_config import DEFAULT_CONFIG
from run_snapp_assessment import PADUS_ANALYSIS_LAYER, PADUS_ANALYSIS_PATH, ROOT


COUNTY_PATH = Path("data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the six-zone assessment inputs and worker limits.

    Args:
        argv: Argument strings, or None to use the command line.

    Returns:
        Selected inputs and execution options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True, help="Path to zonal_stats_toolkit/runner.py")
    parser.add_argument("--bbb-by-county", type=Path, required=True, help="Prepared BBB county geometry GeoPackage")
    parser.add_argument("--bbb-layer")
    parser.add_argument("--padus-path", type=Path, default=PADUS_ANALYSIS_PATH)
    parser.add_argument("--padus-layer", default=PADUS_ANALYSIS_LAYER)
    parser.add_argument("--counties", type=Path, default=COUNTY_PATH)
    parser.add_argument("--counties-layer")
    parser.add_argument("--land-mask", type=Path, required=True, help="Census county land polygons keyed by GEOID")
    parser.add_argument("--land-mask-layer", help="Layer in the land-mask file")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--job-workers", type=int, default=2)
    parser.add_argument("--raster-workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent county geometry and tabular workers")
    parser.add_argument("--geometry-workers", type=int, default=4, help="Concurrent fisheries and coastal measurement workers")
    parser.add_argument("--validate-only", action="store_true", help="Check inputs without creating run files")
    return parser.parse_args(argv)


def run_assessment(args: argparse.Namespace, root: Path = ROOT) -> Path | None:
    """Execute zone preparation, raw raster zonals and final valuation.

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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens")
    if min(args.job_workers, args.raster_workers, args.workers, args.geometry_workers) < 1:
        raise ValueError("Worker limits must be positive")
    root = root.resolve()
    data = root / "data"
    runner = (root / args.runner).resolve()
    config_dir = data / "workflow_assets/county_integration"
    zonal_config = root / DEFAULT_CONFIG.relative_to(ROOT)
    if not runner.is_file():
        raise FileNotFoundError(f"Toolkit runner not found: {runner}")
    paths, _, _, derived = load_valuation_config(data, config_dir)
    rasters, blocked = validate_zone_raster_inputs(zonal_config)
    raster_fields = {f"sum_{raster.path.stem}" for raster in rasters}
    expected_fields = set(BASE_FIELDS.values())
    if raster_fields != expected_fields:
        raise ValueError(
            f"TIFF inventory does not match the configured valuation services: "
            f"missing {sorted(expected_fields - raster_fields)}, extra {sorted(raster_fields - expected_fields)}"
        )
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
    if args.validate_only:
        print("Six-zone CONUS assessment inputs validated; no run files written.", flush=True)
        return None

    results = data / "analysis_results/assessment_runs" / args.run_id
    work = data / "processing_outputs/assessment_runs" / args.run_id
    frozen = zonal_config.parent / "runs" / f"snapp_six_zone_{args.run_id}.yaml"
    for path in (results, work, frozen):
        if path.exists():
            raise FileExistsError(f"Run path already exists; use a new --run-id: {path}")
    results.mkdir(parents=True)
    config_files = [*(config_dir / filename for filename in CONFIG_FILES), zonal_config, root / "config/padus_land_rules.rules"]
    hashes = {str(path): sha256(path) for path in config_files}
    zone_root = work / "zonal_units"
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
        "raw_zonal_directory": str(work / "raw_zonals"),
        "blocked_rasters": blocked,
        "stages": {}, "outputs": {},
    }

    def record(stage: str, status: str) -> None:
        """Persist stage status and the currently verified outputs."""
        manifest["stages"][stage] = status
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{stage}: {status}", flush=True)

    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    stage = "Prepare six assessment zones"
    try:
        record(stage, "running")
        subprocess.run([
            sys.executable, "-B", str(root / "prepare_assessment_zones.py"),
            "--padus-path", input_paths["padus"][0], "--padus-layer", input_paths["padus"][1],
            "--counties", input_paths["counties"][0], "--counties-layer", input_paths["counties"][1],
            "--land-mask", input_paths["land_mask"][0], "--land-mask-layer", input_paths["land_mask"][1],
            "--bbb-by-county", input_paths["bbb_by_county"][0], "--bbb-layer", input_paths["bbb_by_county"][1],
            "--rules", str(root / "config/padus_land_rules.rules"),
            "--run-id", args.run_id, "--output-root", str(work.parent), "--workers", str(args.workers),
        ], cwd=root, env=env, check=True)
        zone_manifest = json.loads((zone_root / "zone_manifest.json").read_text(encoding="utf-8"))
        if (zone_manifest["run_id"] != args.run_id or zone_manifest.get("geographic_scope") != "CONUS"
                or set(zone_manifest["outputs"]) != set(ZONE_PRODUCTS)):
            raise ValueError("Zone manifest does not identify this run's six CONUS products")
        zones = {}
        for product in ZONE_PRODUCTS:
            path = zone_root / product / f"{product}_{args.run_id}.gpkg"
            output = zone_manifest["outputs"][product]
            if Path(output["path"]).resolve() != path.resolve() or output["layer"] != product or not path.is_file():
                raise ValueError(f"Unexpected zone path or layer for {product}")
            zones[product] = (path, product)
        overlap = zone_manifest["counties_with_manager_overlap"]
        overlapping_products = {
            f"{name}_lands_by_county_and_manager_type" for name in ("public", "conservation")
            if overlap[f"area_ha_{name}_manager_overlap"] > 0
        }
        manifest["manager_overlap_products"] = sorted(overlapping_products)
        record(stage, "complete")

        stage = "Freeze raw raster zonal jobs"
        record(stage, "running")
        plan = freeze_zone_zonals(zones, zonal_config, frozen, work / "zonal_stats_work", work / "raw_zonals",
                                 overlapping_manager_products=overlapping_products)
        manifest["empty_zone_products"] = list(plan.skipped_products)
        record(stage, "complete")

        stage = "Run and verify raster zonals"
        record(stage, "running")
        if plan.jobs:
            subprocess.run([
                sys.executable, "-B", str(runner), str(plan.frozen_config),
                "--job-workers", str(args.job_workers), "--raster-workers", str(args.raster_workers),
            ], cwd=root, env=env, check=True)
        raster_sums = read_zone_raster_sums(plan)
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
    print(f"Completed six-zone assessment: {results}", flush=True)
    return results


if __name__ == "__main__":
    run_assessment(parse_args())
