"""Prepare an explicit zonal-statistics config for SNAPP valuation rasters.

The script reads the reviewed raster manifest, creates any lightweight derived
VRTs requested by that manifest, discovers the newest county-cut land products,
and writes a configuration consumable by ``zonal_stats_toolkit/runner.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import tomllib

import fiona
from osgeo import gdal
from tqdm import tqdm


DEFAULT_MANIFEST = Path("config/ecosystem_service_rasters.toml")
DEFAULT_RASTER_DIR = Path("data/analysis_inputs/ecosystem_services")
DEFAULT_DERIVED_RASTER_DIR = Path(
    "data/processing_outputs/zonal_stats_derived_rasters"
)
DEFAULT_ZONAL_UNITS_DIR = Path("data/analysis_inputs/zonal_units")
DEFAULT_OUTPUT_CONFIG = Path(
    "data/processing_outputs/zonal_stats_config/"
    "snapp_assessment_tif_zonal_stats.yaml"
)
DEFAULT_RESULTS_DIR = Path("data/analysis_results/zonal_statistics")
DEFAULT_WORK_DIR = Path("data/processing_outputs/zonal_stats_work")
MAX_SIMPLIFY_TOLERANCE_METERS = 15
TIMESTAMP_SUFFIX = re.compile(
    r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.gpkg$"
)


@dataclass(frozen=True)
class RasterInput:
    """One validated source or derived raster selected for analysis.

    Attributes:
        label: Human-readable ecosystem-service name.
        path: Raster or VRT path passed to the zonal-statistics runner.
        units: Units supplied with the handoff.
    """

    label: str
    path: Path
    units: str


@dataclass(frozen=True)
class ZoneJob:
    """One county-keyed vector product used as an aggregation layer.

    Attributes:
        key: Stable command-line key for selecting this zone.
        label: Human-readable zonal dataset name.
        vector_path: Concrete GeoPackage selected for the run.
        layer: GeoPackage layer containing one feature per county.
        job_stem: Output filename stem expected by the combination script.
        results_subdir: Subdirectory under the zonal-statistics results root.
    """

    key: str
    label: str
    vector_path: Path
    layer: str
    job_stem: str
    results_subdir: str


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for config generation.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate the TIFF handoff and generate a raster-only "
            "zonal_stats_toolkit configuration using the newest county-cut "
            "PAD-US/BBB GeoPackages."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raster-dir", type=Path, default=DEFAULT_RASTER_DIR)
    parser.add_argument(
        "--derived-raster-dir",
        type=Path,
        default=DEFAULT_DERIVED_RASTER_DIR,
    )
    parser.add_argument(
        "--zonal-units-dir",
        type=Path,
        default=DEFAULT_ZONAL_UNITS_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--zone",
        action="append",
        choices=(
            "counties",
            "all",
            "public",
            "public-access",
            "blm-unprotected",
            "bbb-candidates",
        ),
        help=(
            "Generate only this zonal dataset. Repeat for multiple datasets; "
            "the default is every dataset currently available."
        ),
    )
    return parser.parse_args()


def load_and_prepare_rasters(
    manifest_path: Path,
    raster_dir: Path,
    derived_raster_dir: Path,
) -> tuple[list[RasterInput], list[str]]:
    """Validate included rasters and create scaled virtual rasters when needed.

    A VRT applies a scale lazily, so the pollination per-hectare to per-pixel
    conversion does not require writing a second multi-gigabyte TIFF.

    Args:
        manifest_path: TOML file describing included and blocked rasters.
        raster_dir: Directory containing downloaded source TIFFs.
        derived_raster_dir: Directory for generated scaled VRT files.

    Returns:
        Included raster inputs and human-readable messages for blocked entries.

    Raises:
        FileNotFoundError: If the manifest or an included source raster is absent.
        ValueError: If the manifest is malformed or a derived VRT cannot be made.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Raster manifest not found: {manifest_path}")

    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    entries = manifest.get("rasters")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{manifest_path} must contain one or more [[rasters]] entries.")

    selected: list[RasterInput] = []
    blocked: list[str] = []
    derived_raster_dir.mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()

    for entry in tqdm(entries, desc="Validate TIFF manifest", unit="raster"):
        try:
            label = str(entry["label"]).strip()
            filename = str(entry["filename"]).strip()
            units = str(entry["units"]).strip()
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Every [[rasters]] entry in {manifest_path} needs label, "
                "filename, and units."
            ) from error
        if not label or not filename or not units:
            raise ValueError(
                f"Raster manifest entry has a blank label, filename, or units: {entry}"
            )

        if not bool(entry.get("include", False)):
            reason = str(entry.get("reason", "not approved for this run")).strip()
            blocked.append(f"{label}: {reason}")
            continue

        source_path = (raster_dir / filename).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Included raster '{label}' is missing: {source_path}"
            )

        scale = float(entry.get("scale", 1.0))
        selected_path = source_path
        if scale != 1.0:
            derived_filename = str(entry.get("derived_filename", "")).strip()
            if not derived_filename.lower().endswith(".vrt"):
                raise ValueError(
                    f"Scaled raster '{label}' needs a .vrt derived_filename."
                )
            selected_path = (derived_raster_dir / derived_filename).resolve()
            source_dataset = gdal.Open(str(source_path), gdal.GA_ReadOnly)
            if source_dataset is None:
                raise ValueError(f"GDAL could not open included raster: {source_path}")
            source_band = source_dataset.GetRasterBand(1)
            translate_options = {
                "format": "VRT",
                "outputType": gdal.GDT_Float32,
                "scaleParams": [[0.0, 1.0, 0.0, scale]],
            }
            source_nodata = source_band.GetNoDataValue()
            if source_nodata is not None:
                translate_options["noData"] = source_nodata
            derived_dataset = gdal.Translate(
                str(selected_path),
                source_dataset,
                options=gdal.TranslateOptions(**translate_options),
            )
            source_dataset = None
            if derived_dataset is None:
                raise ValueError(
                    f"GDAL could not create scaled VRT for '{label}': {selected_path}"
                )
            derived_dataset.SetMetadataItem("SNAPP_SOURCE_UNITS", units)
            derived_dataset.SetMetadataItem("SNAPP_SCALE_FACTOR", str(scale))
            derived_dataset.FlushCache()
            derived_dataset = None

        selected.append(RasterInput(label=label, path=selected_path, units=units))

    if not selected:
        raise ValueError(f"No rasters are enabled in {manifest_path}.")
    if len({item.path.resolve() for item in selected}) != len(selected):
        raise ValueError(f"The raster manifest selects the same output path more than once.")
    return selected, blocked


def discover_zone_jobs(
    zonal_units_dir: Path,
    selected_keys: set[str] | None = None,
) -> tuple[list[ZoneJob], list[str]]:
    """Discover the newest concrete GeoPackage for each supported zonal product.

    Args:
        zonal_units_dir: Root directory containing county-keyed zonal products.
        selected_keys: Optional subset of stable zone keys requested by the user.

    Returns:
        Validated zone jobs and messages for unavailable optional products.

    Raises:
        FileNotFoundError: If an explicitly requested zone is unavailable.
        ValueError: If a selected GeoPackage lacks its expected layer or GEOID.
    """
    definitions = (
        (
            "counties",
            "Counties",
            "counties/tl_2024_us_county_50_states.gpkg",
            "tl_2024_us_county_60_states",
            "counties_ecosystem_services",
            "counties",
            False,
        ),
        (
            "all",
            "PAD-US all lands by county",
            "padus_all_lands_by_county/padus_all_lands_clipped_by_county_*.gpkg",
            "padus_all_lands_clipped_by_county",
            "pad_ecosystem_services",
            "padus_all_lands",
            True,
        ),
        (
            "public",
            "PAD-US public lands by county",
            "padus_public_lands_by_county/padus_public_lands_clipped_by_county_*.gpkg",
            "padus_public_lands_clipped_by_county",
            "public_ecosystem_services",
            "padus_public_lands",
            True,
        ),
        (
            "public-access",
            "PAD-US public-access lands by county",
            "padus_public_access_lands_by_county/"
            "padus_public_access_lands_clipped_by_county_*.gpkg",
            "padus_public_access_lands_clipped_by_county",
            "public_access_ecosystem_services",
            "padus_public_access_lands",
            True,
        ),
        (
            "blm-unprotected",
            "BLM lands excluding mapped federal protections by county",
            "blm_lands_excluding_federally_protected_areas_by_county/"
            "blm_lands_excluding_federally_protected_areas_by_county_*.gpkg",
            "blm_lands_excluding_federally_protected_areas_by_county",
            "blm_unprotected_ecosystem_services",
            "blm_lands_excluding_federally_protected_areas",
            True,
        ),
        (
            "bbb-candidates",
            "BBB candidate BLM lands by county",
            "bbb_candidate_blm_lands_by_county/"
            "bbb_candidate_blm_lands_within_5_miles_of_"
            "population_centers_by_county_*.gpkg",
            "bbb_candidate_blm_lands_within_5_miles_of_"
            "population_centers_by_county",
            "bbb_candidate_ecosystem_services",
            "bbb_candidate_blm_lands",
            True,
        ),
    )

    jobs: list[ZoneJob] = []
    unavailable: list[str] = []
    for definition in tqdm(definitions, desc="Discover county polygons", unit="zone"):
        key, label, pattern, layer, job_stem, results_subdir, timestamped = definition
        if selected_keys is not None and key not in selected_keys:
            continue

        if timestamped:
            candidates = list(zonal_units_dir.glob(pattern))
            if candidates:
                def timestamp_key(path: Path) -> tuple[str, float]:
                    """Return sortable time information for one candidate path.

                    Args:
                        path: Timestamped GeoPackage candidate.

                    Returns:
                        Embedded filename timestamp and filesystem mtime.
                    """
                    match = TIMESTAMP_SUFFIX.search(path.name)
                    return (match.group(1) if match else "", path.stat().st_mtime)

                vector_path = max(candidates, key=timestamp_key)
            else:
                vector_path = None
        else:
            candidate = zonal_units_dir / pattern
            vector_path = candidate if candidate.is_file() else None

        if vector_path is None:
            message = f"{key}: no GeoPackage matches {zonal_units_dir / pattern}"
            if selected_keys is not None:
                raise FileNotFoundError(message)
            unavailable.append(message)
            continue

        layers = fiona.listlayers(vector_path)
        if layer not in layers:
            raise ValueError(
                f"Expected layer '{layer}' in {vector_path}; available layers: {layers}"
            )
        with fiona.open(vector_path, layer=layer) as source:
            if "GEOID" not in source.schema.get("properties", {}):
                raise ValueError(f"Layer '{layer}' in {vector_path} has no GEOID field.")

        jobs.append(
            ZoneJob(
                key=key,
                label=label,
                vector_path=vector_path,
                layer=layer,
                job_stem=job_stem,
                results_subdir=results_subdir,
            )
        )

    if not jobs:
        raise ValueError("No usable zonal datasets were discovered.")
    return jobs, unavailable


def write_runner_config(
    output_path: Path,
    work_dir: Path,
    results_dir: Path,
    rasters: list[RasterInput],
    zone_jobs: list[ZoneJob],
) -> None:
    """Write a runner configuration with explicit rasters and vector versions.

    Args:
        output_path: Destination configuration path.
        work_dir: Toolkit work/cache directory.
        results_dir: Root directory for timestamped result projects.
        rasters: Reviewed raster paths to list explicitly in every job.
        zone_jobs: Concrete county-keyed aggregation layers for the run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_dir = output_path.parent.resolve()

    def relative_path(path: Path) -> str:
        """Return a path relative to the generated configuration.

        Args:
            path: Input or output path to represent in the config.

        Returns:
            Portable forward-slash relative path.
        """
        return Path(os.path.relpath(path.resolve(), config_dir)).as_posix()

    lines = [
        "# Generated by prepare_tif_zonal_stats_config.py; do not edit by hand.",
        "# Rasters are explicit so unrelated TIFFs in the source folder are ignored.",
        "# Included raster inventory:",
    ]
    lines.extend(
        f"# - {raster.label}: {raster.path.name} [{raster.units}]"
        for raster in rasters
    )
    lines.extend(
        [
            "",
            "[project]",
            f"name = {output_path.stem}",
            f"global_work_dir = {relative_path(work_dir)}",
            "log_level = INFO",
            f"max_simplify_tolerance_meters = {MAX_SIMPLIFY_TOLERANCE_METERS}",
            "",
        ]
    )
    raster_lines = [relative_path(raster.path) for raster in rasters]

    for zone in zone_jobs:
        output_base = results_dir / zone.results_subdir / zone.job_stem
        lines.extend(
            [
                f"# {zone.label}",
                f"[job:{zone.job_stem}]",
                f"agg_vector = {relative_path(zone.vector_path)}",
                f"agg_layer = {zone.layer}",
                "agg_field = GEOID",
                "operations = sum, mean, area_ha_valid, proportion_valid_nonzero",
                "base_raster_pattern =",
            ]
        )
        for index, raster_path in enumerate(raster_lines):
            comma = "," if index < len(raster_lines) - 1 else ""
            lines.append(f"    {raster_path}{comma}")
        lines.extend(
            [
                f"output_csv = {relative_path(output_base.with_suffix('.csv'))}",
                f"output_gpkg = {relative_path(output_base.with_suffix('.gpkg'))}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_config(
    manifest_path: Path = DEFAULT_MANIFEST,
    raster_dir: Path = DEFAULT_RASTER_DIR,
    derived_raster_dir: Path = DEFAULT_DERIVED_RASTER_DIR,
    zonal_units_dir: Path = DEFAULT_ZONAL_UNITS_DIR,
    output_path: Path = DEFAULT_OUTPUT_CONFIG,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    work_dir: Path = DEFAULT_WORK_DIR,
    selected_keys: set[str] | None = None,
) -> Path:
    """Prepare all inputs and write one validated raster-only runner config.

    Args:
        manifest_path: Reviewed raster inventory.
        raster_dir: Directory containing source TIFFs.
        derived_raster_dir: Directory for lazily scaled VRTs.
        zonal_units_dir: Directory containing county-keyed aggregation layers.
        output_path: Destination runner configuration.
        results_dir: Root directory for zonal-stat outputs.
        work_dir: Toolkit work/cache directory.
        selected_keys: Optional subset of zonal dataset keys.

    Returns:
        Path to the generated runner configuration.
    """
    rasters, blocked_rasters = load_and_prepare_rasters(
        manifest_path,
        raster_dir,
        derived_raster_dir,
    )
    zone_jobs, unavailable_zones = discover_zone_jobs(zonal_units_dir, selected_keys)
    write_runner_config(output_path, work_dir, results_dir, rasters, zone_jobs)

    tqdm.write(f"Generated {output_path} with {len(rasters)} rasters and {len(zone_jobs)} zones.")
    tqdm.write("Selected zones:")
    for zone in zone_jobs:
        tqdm.write(f"  {zone.key}: {zone.vector_path}")
    if blocked_rasters:
        tqdm.write("Blocked/not-yet-supplied spreadsheet rasters:")
        for message in blocked_rasters:
            tqdm.write(f"  {message}")
    if unavailable_zones:
        tqdm.write("Unavailable optional zonal products:")
        for message in unavailable_zones:
            tqdm.write(f"  {message}")
    return output_path


def main() -> None:
    """Run config preparation from command-line arguments."""
    args = _parse_args()
    prepare_config(
        manifest_path=args.manifest,
        raster_dir=args.raster_dir,
        derived_raster_dir=args.derived_raster_dir,
        zonal_units_dir=args.zonal_units_dir,
        output_path=args.output,
        results_dir=args.results_dir,
        work_dir=args.work_dir,
        selected_keys=set(args.zone) if args.zone else None,
    )


if __name__ == "__main__":
    main()
