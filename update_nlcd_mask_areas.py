"""Rerun county NLCD masks only and optionally append areas to existing valuations.

No ecosystem-service raster is sampled and no monetary adjustment is applied.
Every run freezes its reviewed definition and uses an isolated raw-results/cache
directory. Existing results are never overwritten or chosen by a 'latest' glob.
"""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm

from build_county_ecosystem_service_valuations_2026 import (
    read_attributes, sha256, write_output,
)
from nlcd_mask_areas import CLASSES, FIELD_DEFINITIONS, OPERATIONS, RAW_FIELD, derive_nlcd_mask_areas
from county_derived_metrics import DEFAULT_CONFIG as DEFAULT_DERIVED_CONFIG, calculate_county_metrics, load_derived_rules
from prepare_tif_zonal_stats_config import load_run_config, prepare_config


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "data/workflow_assets/zonal_stats/snapp_assessment_nlcd_masks.yaml"


def validate_mask_definition(config_path: Path) -> list[dict]:
    """Check that this workflow cannot select ecosystem-service rasters.

    Args:
        config_path: Reviewed mask-only definition, with explicit source paths.

    Returns:
        Source metadata for the provenance tables (not a full raster scan).

    Raises:
        ValueError: Inventory, binary datatype, NoData, grid, or jobs differ
            from the five co-registered NLCD masks and a single county job.
    """
    config = load_run_config(config_path)
    inventory = json.loads(config["project"]["raster_inventory"])
    jobs = [s for s in config.sections() if s.startswith("job:")]
    if (jobs != ["job:counties_masks"] or config[jobs[0]]["zone_key"] != "counties"
            or config[jobs[0]]["agg_field"] != "GEOID"
            or config[jobs[0]]["results_subdir"] != "counties"
            or set(config[jobs[0]]["operations"].replace(" ", "").split(",")) != set(OPERATIONS)):
        raise ValueError("Expected one county mask job with the required count/area operations.")
    if len(inventory) != len(CLASSES) or {r["label"] for r in inventory} != set(CLASSES):
        raise ValueError("Expected exactly five named NLCD masks.")
    records, grid = [], None
    for entry in tqdm(inventory, desc="Validate NLCD mask grids", unit="raster"):
        path = (config_path.parent / config["project"]["raster_dir"] / entry["filename"]).resolve()
        match = RAW_FIELD.fullmatch("sum_" + path.stem)
        if (not match or match["class"] != entry["label"] or not entry["include"]
                or entry.get("scale", 1) != 1):
            raise ValueError(f"Not an unscaled NLCD mask: {path}")
        with rasterio.open(path) as source:
            current = (source.crs, source.transform, source.width, source.height)
            if (source.count != 1 or source.dtypes != ("uint8",) or source.nodata != 255
                    or source.crs is None or not source.crs.is_projected
                    or source.crs.linear_units_factor[1] != 1
                    or source.transform.b != 0 or source.transform.d != 0
                    or source.transform.a <= 0 or source.transform.e >= 0):
                raise ValueError(f"Expected a north-up projected metre-grid byte mask with NoData 255: {path}")
            if grid is not None and current != grid:
                raise ValueError("All NLCD masks must have identical CRS, extent and pixel grid.")
            grid = current
            records.append({"class": entry["label"], "path": str(path),
                            "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                            "crs": source.crs.to_wkt(), "transform": json.dumps(tuple(source.transform)),
                            "width": source.width, "height": source.height,
                            "pixel_area_ha": abs(source.transform.a * source.transform.e) / 10000,
                            "nodata": source.nodata})
    return records


def export_mask_areas(mask_results: Path, output: Path, run_config: Path,
                      county_results: Path | None = None, raster_metadata: list[dict] | None = None,
                      *, derived_config: Path | None = DEFAULT_DERIVED_CONFIG,
                      data_root: Path | None = None) -> Path:
    """Publish corrected areas, optionally alongside unchanged service values.

    Args:
        mask_results: One explicit raw county mask GeoPackage from the new run.
        output: New GeoPackage path; a matching CSV is also written.
        run_config: Frozen config used for this exact mask run.
        county_results: Existing final county GeoPackage to augment, or None to
            export mask-only attributes and geometry from ``mask_results``.
        raster_metadata: Validated raster grid and source stat records.
        derived_config: Rules for totals/ratios on augmented county outputs.
            None disables that stage for programmatic mask-only testing.
        data_root: Denominator source root; defaults to this repository's data.

    Returns:
        New GeoPackage path, after CSV parity and preservation checks.

    Raises:
        ValueError: County keys differ, mask stats fail QA, or the base already
            has these area fields. A corrected output is not an input for this
            append-only operation; reuse its original pre-mask base instead.
        FileExistsError: An output exists.
    """
    base_path = county_results or mask_results
    frozen = configparser.ConfigParser(interpolation=None)
    frozen.read(run_config, encoding="utf-8")
    jobs = [s for s in frozen.sections() if s.startswith("job:")]
    if jobs != ["job:counties_masks"] or "snapp_source_definition" not in frozen["project"]:
        raise ValueError("Supply the frozen county mask-only config for these raw results.")
    expected = (run_config.parent / frozen[jobs[0]]["output_gpkg"]).resolve()
    if (mask_results.resolve().parent != expected.parent
            or not mask_results.stem.startswith(expected.stem + "_")):
        raise ValueError("Mask results path does not match this frozen run's output location.")
    inputs = {"mask_results": mask_results, "base_results": base_path, "run_config": run_config,
              "area_calculator": Path(__file__).with_name("nlcd_mask_areas.py"), "workflow": Path(__file__)}
    fingerprints = {name: sha256(path) for name, path in tqdm(inputs.items(), desc="Hash mask workflow inputs", unit="file")}
    masks, _, _ = read_attributes(mask_results)
    base, layer, geom = read_attributes(base_path)
    for label, frame in (("masks", masks), ("base", base)):
        if ("GEOID" not in frame or frame.GEOID.isna().any() or frame.GEOID.duplicated().any()
                or not frame.GEOID.map(lambda v: isinstance(v, str) and len(v) == 5 and v.isdigit()).all()):
            raise ValueError(f"{label} must have unique five-character county GEOIDs.")
    if set(base.GEOID) != set(masks.GEOID):
        raise ValueError("County GEOIDs differ between mask and base results; no silent partial joins.")
    if set(base.columns) & set(FIELD_DEFINITIONS):
        raise ValueError("Base already contains NLCD area fields; select the original pre-mask county results.")
    if county_results:
        with tqdm(total=3, desc="Verify county boundary compatibility", unit="step") as progress:
            mask_geometry = gpd.read_file(mask_results, columns=["GEOID"]).set_index("GEOID").geometry
            progress.update(1)
            base_geometry = gpd.read_file(base_path, layer=layer, columns=["GEOID"]).set_index("GEOID").geometry
            progress.update(1)
            mask_geometry = mask_geometry.reindex(base_geometry.index)
            exact = base_geometry.geom_equals_exact(mask_geometry, tolerance=0)
            if (mask_geometry.crs != base_geometry.crs
                    or not base_geometry[~exact].geom_equals(mask_geometry[~exact]).all()):
                raise ValueError("County boundaries differ; calculate masks against the same county geography.")
            progress.update(1)
    computed = derive_nlcd_mask_areas(masks.set_index("GEOID"))
    additions = computed[list(FIELD_DEFINITIONS)].reindex(base.GEOID)
    # Input rasters are byte masks; min/max checks reject sampled nonbinary values.
    # Compare count-derived areas to metadata as an independent unit check.
    if raster_metadata:
        expected_pixel_area = raster_metadata[0]["pixel_area_ha"]
        for name in CLASSES:
            count_col = next(c for c in masks if (match := RAW_FIELD.fullmatch(c))
                             and match["operation"] == "sum" and match["class"] == name)
            expected = pd.to_numeric(masks.set_index("GEOID")[count_col]).fillna(0) * expected_pixel_area
            pd.testing.assert_series_equal(additions[f"area_ha_nlcd_{name}"], expected.reindex(additions.index),
                                           check_names=False, check_dtype=False, rtol=1e-9, atol=1e-8)
    removed = [c for c in base if RAW_FIELD.fullmatch(c)]
    created = datetime.now(timezone.utc).isoformat()
    tables = {
        "nlcd_area_sources": pd.DataFrame([
            {"source": key, "path": str(path.resolve()), "sha256": fingerprints[key]}
            for key, path in inputs.items()]),
        "nlcd_area_fields": pd.DataFrame([{"field": key, "definition": value} for key, value in FIELD_DEFINITIONS.items()]),
        "nlcd_area_run": pd.DataFrame([{
            "created_utc": created, "frozen_config": run_config.read_text(encoding="utf-8"),
            "method": "class hectares = binary sum * (total area / total count); class share = sum / valid count",
            "no_coverage": "zero hectares, null class proportions; missing rows/statistics are errors",
            "service_values": "preserved unchanged; no ecosystem-service raster or monetary conversion executed",
        }]),
    }
    if raster_metadata:
        tables["nlcd_area_rasters"] = pd.DataFrame(raster_metadata)
    if county_results and derived_config is not None:
        # Use THIS run's newly corrected masks, never the older configured release.
        derived, derived_tables = calculate_county_metrics(
            base.set_index("GEOID"), data_root or ROOT / "data", derived_config,
            natural_areas=additions, natural_source=mask_results)
        additions = pd.concat([additions, derived.drop(columns=list(FIELD_DEFINITIONS))], axis=1)
        tables.update(derived_tables)
    for key, path in inputs.items():
        if sha256(path) != fingerprints[key]:
            raise ValueError(f"Input changed during mask export: {path}")
    write_output(base_path, output, layer, geom, additions, tables, removed_fields=removed,
                 output_layer="tl_2024_us_county_50_states")
    tqdm.write(f"Verified: {output}\nVerified: {output.with_suffix('.csv')}")
    return output


def run_mask_update(runner: Path, config_path: Path = DEFAULT_CONFIG,
                    county_results: Path | None = None, raster_workers: int = 2,
                    *, derived_config: Path | None = DEFAULT_DERIVED_CONFIG,
                    data_root: Path | None = None) -> list[Path]:
    """Prepare and run only masks, then publish new paired result files.

    Args:
        runner: Local zonal_stats_toolkit/runner.py, run with this Python environment.
        config_path: Tracked county mask-only profile to freeze for this run.
        county_results: Optional explicit existing final county GeoPackage.
        raster_workers: Parallel mask rasters; no service-raster jobs are selected.
        derived_config: Rules for county totals/ratios, or None for isolated tests.
        data_root: Denominator source root; defaults to this repository's data.

    Returns:
        Corrected mask GeoPackage and, if requested, augmented county GeoPackage.

    Raises:
        ValueError: Inputs fail validation or the runner produces ambiguous results.
        subprocess.CalledProcessError: The mask-only runner fails; no final export
            is attempted, and the frozen config and raw work remain for diagnosis.
    """
    runner, config_path = runner.resolve(), config_path.resolve()
    if not runner.is_file() or raster_workers < 1:
        raise ValueError("Supply an existing runner.py and a positive raster worker count.")
    if county_results:
        county_results = county_results.resolve()
        base, _, _ = read_attributes(county_results)
        if set(base) & set(FIELD_DEFINITIONS):
            raise ValueError("Select the original pre-mask county results, not an earlier augmented output.")
        if derived_config is not None:
            missing = set(load_derived_rules(derived_config)["es_total"]["components"]) - set(base)
            if missing:
                raise ValueError(f"County input lacks required adjusted es_total components: {sorted(missing)}")
    metadata = validate_mask_definition(config_path)
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    run_root = ROOT / "data/analysis_results/nlcd_mask_areas" / stamp
    frozen_path = config_path.parent / "runs" / f"snapp_assessment_nlcd_masks_{stamp}.yaml"
    frozen = prepare_config(config_path=config_path, output_path=frozen_path,
                            results_dir=run_root / "raw", work_dir=ROOT / "data/processing_outputs/nlcd_mask_areas_work" / stamp,
                            selected_keys={"counties"})
    command = [sys.executable, str(runner), str(frozen), "--job-workers", "1", "--raster-workers", str(raster_workers)]
    tqdm.write("Running five NLCD masks only: " + subprocess.list2cmdline(command))
    subprocess.run(command, check=True)
    results = list((run_root / "raw/counties").glob("counties_masks_*.gpkg"))
    if len(results) != 1 or not results[0].with_suffix(".csv").is_file():
        raise ValueError(f"Expected exactly one complete raw mask CSV/GeoPackage pair in {run_root}.")
    for record in metadata:
        stat = Path(record["path"]).stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Mask changed during computation: {record['path']}")
    raw = results[0]
    # Ensure both raw formats came from the same completed run.
    attrs, _, _ = read_attributes(raw)
    csv = pd.read_csv(raw.with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip")
    common = [c for c in attrs if RAW_FIELD.fullmatch(c)]
    pd.testing.assert_frame_equal(attrs.set_index("GEOID")[common].sort_index(),
                                  csv.set_index("GEOID")[common].sort_index(), check_dtype=False)
    outputs = [export_mask_areas(raw, run_root / f"counties_nlcd_mask_areas_{stamp}.gpkg", frozen,
                                raster_metadata=metadata)]
    if county_results:
        outputs.append(export_mask_areas(raw, ROOT / "data/analysis_results/combined" /
                                        f"snapp_ecosystem_services_counties_{stamp}.gpkg", frozen,
                                        county_results=county_results, raster_metadata=metadata,
                                        derived_config=derived_config, data_root=data_root))
    return outputs


def main() -> None:
    """Run the mask-only workflow or append one explicitly selected completed run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, help="Local zonal_stats_toolkit/runner.py; required unless reusing --mask-results")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--county-results", type=Path, help="Existing final county GPKG; service values are copied unchanged")
    parser.add_argument("--derived-config", type=Path, default=DEFAULT_DERIVED_CONFIG)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="Denominator source root")
    parser.add_argument("--raster-workers", type=int, default=2)
    parser.add_argument("--mask-results", type=Path, help="Reuse this exact NEW raw mask GPKG instead of running any zonals")
    parser.add_argument("--run-config", type=Path, help="Frozen config corresponding to --mask-results")
    parser.add_argument("--output", type=Path, help="New output GPKG for --mask-results reuse mode; also writes CSV")
    args = parser.parse_args()
    if args.mask_results:
        if not args.run_config or not args.output or args.runner:
            parser.error("--mask-results requires --run-config and --output, and cannot be combined with --runner")
        export_mask_areas(args.mask_results, args.output, args.run_config, args.county_results,
                          derived_config=args.derived_config, data_root=args.data_root)
    else:
        if not args.runner or args.run_config or args.output:
            parser.error("Supply --runner, or use --mask-results with --run-config and --output")
        run_mask_update(args.runner, args.config, args.county_results, args.raster_workers,
                        derived_config=args.derived_config, data_root=args.data_root)


if __name__ == "__main__":
    main()
