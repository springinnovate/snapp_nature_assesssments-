"""Freeze a reviewed SNAPP zonal .yaml into a timestamped runnable .yaml.

The toolkit's .yaml files use INI sections. The reviewed definition holds raster
selection, conversions, metrics, zones, and output policy together. Preparation
validates inputs, resolves vector versions, and records provenance in a new run
config without overwriting earlier configs or their derived VRTs.
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

import fiona
from osgeo import gdal
from tqdm import tqdm


DEFAULT_CONFIG = Path(__file__).resolve().parent / (
    "data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml"
)
TIMESTAMP_SUFFIX = re.compile(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.gpkg$")


@dataclass(frozen=True)
class RasterInput:
    """One reviewed raster or derived VRT.

    Attributes:
        label: Ecosystem-service name.
        path: Raster path passed to the runner.
        units: Reviewed source and analysis units.
        source_path: Original TIFF used to produce the analysis raster.
        scale: Multiplicative conversion from source to analysis pixels.
    """

    label: str
    path: Path
    units: str
    source_path: Path
    scale: float


@dataclass(frozen=True)
class ZoneJob:
    """A concrete county-keyed aggregation job.

    Attributes:
        key: Zone selector from the reviewed config.
        label: Human-readable zonal dataset name.
        vector_path: Concrete GeoPackage selected for this run.
        layer: GeoPackage feature layer.
        job_stem: Job and output filename stem.
        results_subdir: Directory under the configured results root.
        agg_field: County identifier field.
        operations: Comma-separated toolkit operations.
    """

    key: str
    label: str
    vector_path: Path
    layer: str
    job_stem: str
    results_subdir: str
    agg_field: str
    operations: str


def load_run_config(config_path: Path) -> configparser.ConfigParser:
    """Read the reviewed definition using the toolkit's INI syntax.

    Args:
        config_path: Version-controlled .yaml run definition.

    Returns:
        Parsed definition, including its inline JSON raster inventory.

    Raises:
        FileNotFoundError: The definition is absent.
        ValueError: Project metadata, inventory, or sections are malformed.
    """
    config = configparser.ConfigParser(interpolation=None)
    with config_path.open(encoding="utf-8") as stream:
        config.read_file(stream)
    if "project" not in config or "raster_inventory" not in config["project"]:
        raise ValueError(f"{config_path} requires [project].raster_inventory.")
    entries = json.loads(config["project"]["raster_inventory"])
    if not isinstance(entries, list) or not entries:
        raise ValueError("raster_inventory must be a nonempty JSON list.")
    if any(s != "project" and not s.startswith("job:") for s in config.sections()):
        raise ValueError("Only [project] and [job:...] sections are supported.")
    if not any(s.startswith("job:") for s in config.sections()):
        raise ValueError("The run definition requires at least one [job:...] section.")
    tolerance = config["project"].getfloat("max_simplify_tolerance_meters")
    if tolerance is None or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_simplify_tolerance_meters must be finite and nonnegative.")
    return config


def load_and_prepare_rasters(
    config_path: Path,
    raster_dir: Path,
    derived_raster_dir: Path,
) -> tuple[list[RasterInput], list[str]]:
    """Validate the inline inventory and create run-specific scaled VRTs.

    Args:
        config_path: Reviewed .yaml with [project].raster_inventory.
        raster_dir: Downloaded source TIFF directory.
        derived_raster_dir: New run-specific directory for scaled VRTs.

    Returns:
        Selected rasters and reasons excluded inventory entries are blocked.

    Raises:
        FileNotFoundError: An enabled TIFF is absent.
        FileExistsError: A derived VRT would overwrite an earlier VRT.
        ValueError: An inventory entry or GDAL conversion is invalid.
    """
    config = load_run_config(config_path)
    entries = json.loads(config["project"]["raster_inventory"])
    selected: list[RasterInput] = []
    blocked: list[str] = []
    gdal.UseExceptions()
    for entry in tqdm(entries, desc="Validate TIFF inventory", unit="raster"):
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            for key in ("label", "filename", "units")
        ) or not isinstance(entry.get("include"), bool):
            raise ValueError("Each raster needs label, filename, units, and a boolean include.")
        label, filename, units = (entry[key].strip() for key in ("label", "filename", "units"))
        if not entry["include"]:
            blocked.append(f"{label}: {entry.get('reason', 'not approved for this run')}")
            continue
        source_path = (raster_dir / filename).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Included raster '{label}' is missing: {source_path}")
        scale = float(entry.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Raster '{label}' requires a finite positive scale.")
        selected_path = source_path
        if scale != 1.0:
            derived_filename = str(entry.get("derived_filename", "")).strip()
            if not derived_filename.lower().endswith(".vrt"):
                raise ValueError(f"Scaled raster '{label}' needs a .vrt derived_filename.")
            selected_path = (derived_raster_dir / derived_filename).resolve()
            if selected_path.exists():
                raise FileExistsError(selected_path)
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            source = gdal.Open(str(source_path), gdal.GA_ReadOnly)
            options = {"format": "VRT", "outputType": gdal.GDT_Float32,
                       "scaleParams": [[0.0, 1.0, 0.0, scale]]}
            nodata = source.GetRasterBand(1).GetNoDataValue()
            if nodata is not None:
                options["noData"] = nodata
            derived = gdal.Translate(str(selected_path), source, options=gdal.TranslateOptions(**options))
            if derived is None:
                raise ValueError(f"GDAL could not create scaled VRT: {selected_path}")
            derived.SetMetadataItem("SNAPP_SOURCE_UNITS", units)
            derived.SetMetadataItem("SNAPP_SCALE_FACTOR", str(scale))
            derived.FlushCache()
            derived = source = None
        selected.append(RasterInput(label, selected_path, units, source_path, scale))
    if not selected or len({r.path for r in selected}) != len(selected):
        raise ValueError("The inventory must select at least one raster and no duplicate paths.")
    return selected, blocked


def discover_zone_jobs(
    zonal_units_dir: Path,
    selected_keys: set[str] | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[list[ZoneJob], list[str]]:
    """Resolve the reviewed zone patterns to concrete, newest GeoPackages.

    Args:
        zonal_units_dir: Root containing county-keyed aggregation products.
        selected_keys: Optional subset of zone keys requested by the user.
        config_path: Reviewed .yaml defining all zone patterns and metrics.

    Returns:
        Selected concrete jobs and reasons unavailable zones were skipped.

    Raises:
        FileNotFoundError: An explicitly selected zone is unavailable.
        ValueError: A zone key, layer, schema, or job definition is invalid.
    """
    config = load_run_config(config_path)
    sections = [s for s in config.sections() if s.startswith("job:")]
    keys = [config[s]["zone_key"] for s in sections]
    if len(set(keys)) != len(keys):
        raise ValueError("Zone keys must be unique.")
    if selected_keys is not None and not selected_keys <= set(keys):
        raise ValueError(f"Unknown zones: {sorted(selected_keys - set(keys))}")
    jobs: list[ZoneJob] = []
    unavailable: list[str] = []
    for section in tqdm(sections, desc="Discover county polygons", unit="zone"):
        job = config[section]
        key = job["zone_key"]
        if selected_keys is not None and key not in selected_keys:
            continue
        pattern = job["vector_pattern"]
        candidates = list(zonal_units_dir.glob(pattern))
        if not candidates:
            message = f"{key}: no GeoPackage matches {zonal_units_dir / pattern}"
            if selected_keys is not None:
                raise FileNotFoundError(message)
            unavailable.append(message)
            continue
        candidates.sort(key=lambda p: (
            TIMESTAMP_SUFFIX.search(p.name).group(1) if TIMESTAMP_SUFFIX.search(p.name) else "",
            p.stat().st_mtime, p.name,
        ))
        vector_path = candidates[-1]
        layer = job["agg_layer"]
        layers = fiona.listlayers(vector_path)
        if layer not in layers:
            raise ValueError(f"Expected layer '{layer}' in {vector_path}; available layers: {layers}")
        with fiona.open(vector_path, layer=layer) as source:
            if job["agg_field"] not in source.schema.get("properties", {}):
                raise ValueError(f"Layer '{layer}' in {vector_path} has no {job['agg_field']} field.")
        jobs.append(ZoneJob(key, job["label"], vector_path, layer, section.split(":", 1)[1],
                            job["results_subdir"], job["agg_field"], job["operations"]))
    if not jobs:
        raise ValueError("No usable zonal datasets were discovered.")
    return jobs, unavailable


def write_runner_config(
    output_path: Path,
    work_dir: Path,
    results_dir: Path,
    rasters: list[RasterInput],
    zone_jobs: list[ZoneJob],
    config_path: Path = DEFAULT_CONFIG,
    unavailable_zones: list[str] | None = None,
) -> None:
    """Freeze a runnable config with selections, conversions, and provenance.

    Args:
        output_path: New run config path, which must not already exist.
        work_dir: Toolkit cache directory.
        results_dir: Timestamped toolkit results root.
        rasters: Validated source/analysis raster selections.
        zone_jobs: Resolved county aggregation jobs and metrics.
        config_path: Reviewed source definition to identify and snapshot.
        unavailable_zones: Reasons optional zone jobs were skipped.

    Raises:
        FileExistsError: The requested run config already exists.
    """
    config = load_run_config(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_dir = output_path.parent.resolve()

    def relative_path(path: Path) -> str:
        """Represent a path relative to this frozen run config.

        Args:
            path: Concrete filesystem path.

        Returns:
            Relative path, or an absolute path for a different Windows drive.
        """
        try:
            return Path(os.path.relpath(path.resolve(), config_dir)).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    # The runner accepts extra project metadata and only project/job sections.
    frozen = configparser.ConfigParser(interpolation=None)
    reviewed = {section: dict(config[section]) for section in config.sections()}
    reviewed["project"]["raster_inventory"] = json.loads(config["project"]["raster_inventory"])
    frozen["project"] = {
        "name": output_path.stem,
        "global_work_dir": relative_path(work_dir),
        "log_level": config["project"].get("log_level", "INFO"),
        "max_simplify_tolerance_meters": config["project"]["max_simplify_tolerance_meters"],
        "snapp_created_utc": datetime.now(timezone.utc).isoformat(),
        "snapp_source_config": relative_path(config_path),
        "snapp_source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "snapp_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapp_source_definition": json.dumps(reviewed, indent=2),
        "snapp_rasters": json.dumps([
            {"label": r.label, "source_path": relative_path(r.source_path),
             "analysis_path": relative_path(r.path), "units": r.units, "scale": r.scale}
            for r in rasters
        ], indent=2),
        "snapp_unavailable_zones": json.dumps(unavailable_zones or []),
    }
    raster_paths = ",\n".join(relative_path(r.path) for r in rasters)
    for zone in zone_jobs:
        output_base = results_dir / zone.results_subdir / zone.job_stem
        frozen[f"job:{zone.job_stem}"] = {
            "agg_vector": relative_path(zone.vector_path), "agg_layer": zone.layer,
            "agg_field": zone.agg_field, "operations": zone.operations,
            "base_raster_pattern": raster_paths,
            "output_csv": relative_path(output_base.with_suffix(".csv")),
            "output_gpkg": relative_path(output_base.with_suffix(".gpkg")),
        }
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write("# Frozen SNAPP run config; preserve with its results. Do not edit.\n")
        frozen.write(stream)


def prepare_config(
    config_path: Path = DEFAULT_CONFIG,
    raster_dir: Path | None = None,
    derived_raster_dir: Path | None = None,
    zonal_units_dir: Path | None = None,
    output_path: Path | None = None,
    results_dir: Path | None = None,
    work_dir: Path | None = None,
    selected_keys: set[str] | None = None,
) -> Path:
    """Validate a reviewed definition and create a new runnable run snapshot.

    Args:
        config_path: Version-controlled zonal definition.
        raster_dir: Optional source TIFF root override.
        derived_raster_dir: Optional derived VRT root override.
        zonal_units_dir: Optional aggregation products root override.
        output_path: Optional new config path; defaults to a timestamped file.
        results_dir: Optional toolkit result root override.
        work_dir: Optional toolkit cache root override.
        selected_keys: Optional subset of zone keys.

    Returns:
        Path to the new, self-describing runnable .yaml snapshot.

    Raises:
        FileExistsError: An output snapshot already exists.
        ValueError: The reviewed definition changed during preparation.
    """
    config_path = config_path.resolve()
    source_bytes = config_path.read_bytes()
    config = load_run_config(config_path)
    project = config["project"]
    config_dir = config_path.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    if output_path is None:
        output_path = config_dir / "runs" / f"{project['name']}_{stamp}.yaml"
    if output_path.exists() or output_path.resolve() == config_path:
        raise FileExistsError(output_path)
    raster_dir = raster_dir or (config_dir / project["raster_dir"])
    derived_root = derived_raster_dir or (config_dir / project["derived_raster_dir"])
    zonal_units_dir = zonal_units_dir or (config_dir / project["zonal_units_dir"])
    results_dir = results_dir or (config_dir / project["results_dir"])
    work_dir = work_dir or (config_dir / project["global_work_dir"])
    zone_jobs, unavailable = discover_zone_jobs(zonal_units_dir, selected_keys, config_path)
    rasters, blocked = load_and_prepare_rasters(config_path, raster_dir, derived_root / output_path.stem)
    if config_path.read_bytes() != source_bytes:
        raise ValueError("Run definition changed during preparation; rerun.")
    write_runner_config(output_path, work_dir, results_dir, rasters, zone_jobs, config_path, unavailable)
    tqdm.write(f"Prepared {output_path} with {len(rasters)} rasters and {len(zone_jobs)} zones.")
    for zone in zone_jobs:
        tqdm.write(f"  {zone.key}: {zone.vector_path}")
    for heading, messages in (("Blocked/not-yet-supplied spreadsheet rasters", blocked),
                              ("Unavailable optional zonal products", unavailable)):
        if messages:
            tqdm.write(heading + ":")
            for message in messages:
                tqdm.write("  " + message)
    tqdm.write("Run with zonal_stats_toolkit/runner.py using this exact timestamped config.")
    return output_path


def main() -> None:
    """Prepare a timestamped run config using CLI overrides when supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Reviewed zonal .yaml definition (replaces the separate TOML manifest)")
    for option in ("raster-dir", "derived-raster-dir", "zonal-units-dir", "results-dir", "work-dir"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--output", type=Path, help="New snapshot path; existing files are never overwritten")
    parser.add_argument("--zone", action="append", help="Zone key from the reviewed config; repeat to select several")
    args = parser.parse_args()
    prepare_config(config_path=args.config, raster_dir=args.raster_dir,
                   derived_raster_dir=args.derived_raster_dir, zonal_units_dir=args.zonal_units_dir,
                   output_path=args.output, results_dir=args.results_dir, work_dir=args.work_dir,
                   selected_keys=set(args.zone) if args.zone else None)


if __name__ == "__main__":
    main()
