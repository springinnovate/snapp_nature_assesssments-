"""Freeze and collect fresh raster sums for the six assessment footprints."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from configparser import ConfigParser
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyogrio
from tqdm import tqdm

from assessment_zone_geometry import ZONE_PRODUCTS
from prepare_tif_zonal_stats_config import (
    RasterInput,
    ZoneJob,
    load_and_prepare_rasters,
    load_run_config,
    write_runner_config,
)


_KEY_FIELDS = {
    product: ("GEOID", "Mang_Type") if product.endswith("_and_manager_type") else ("GEOID",)
    for product in ZONE_PRODUCTS
}
_MANAGER_PRODUCTS = frozenset(product for product, fields in _KEY_FIELDS.items() if len(fields) == 2)
_RESULT_STAMP = re.compile(r"\d{8}_\d{6}")


@dataclass(frozen=True)
class ZoneRasterJob:
    """One expected pair of timestamped toolkit outputs.

    Attributes:
        product: Assessment product to which this job contributes.
        output_base: Untimestamped toolkit output path, without an extension.
        key_fields: Output key columns, in their expected order.
        expected_keys: Exact key tuples from the original job geometries.
    """

    product: str
    output_base: Path
    key_fields: tuple[str, ...]
    expected_keys: frozenset[tuple[str, ...]]


@dataclass(frozen=True)
class ZoneRasterPlan:
    """Frozen raster run and the expected output schema for each product.

    Attributes:
        frozen_config: Exact toolkit configuration file to run.
        raster_columns: Sum column names corresponding to the unscaled TIFFs.
        jobs: Nonempty toolkit jobs and their expected result keys.
        blocked_rasters: Reviewed reasons for inventory exclusions.
        skipped_products: Empty products that need no toolkit job.
    """

    frozen_config: Path
    raster_columns: tuple[str, ...]
    jobs: tuple[ZoneRasterJob, ...]
    blocked_rasters: tuple[str, ...]
    skipped_products: tuple[str, ...]


def _zone_keys(frame: pd.DataFrame, fields: tuple[str, ...], label: str) -> pd.DataFrame:
    """Validate and retain original text keys without losing leading zeros.

    Args:
        frame: Prepared zone rows or one saved manager shard.
        fields: Exact zone-key columns.
        label: Product or shard name for diagnostics.

    Returns:
        The text key columns, with the original row indexes.

    Raises:
        ValueError: One or more key fields or values are invalid.
    """
    if not set(fields) <= set(frame.columns):
        raise ValueError(f"{label} requires key fields {', '.join(fields)}")
    keys = frame[list(fields)].astype("string")
    if (
        not keys["GEOID"].str.fullmatch(r"[0-9]{5}").fillna(False).all()
        or ("Mang_Type" in fields and not keys["Mang_Type"].str.strip().ne("").fillna(False).all())
        or keys.duplicated().any()
    ):
        raise ValueError(f"{label} has blank, malformed, or duplicate zone keys")
    return keys


def _job_key_groups(product: str, keys: pd.DataFrame, split: bool):
    """Assign the same deterministic stems when freezing or reusing jobs.

    Args:
        product: One of the six assessment zone products.
        keys: Validated, nonempty product keys.
        split: Separate the product by manager type.

    Yields:
        The toolkit job stem and corresponding subset of zone keys.
    """
    if split:
        for number, (_, group) in enumerate(keys.groupby("Mang_Type", sort=True), start=1):
            yield f"{product}__manager_{number:02d}", group
    else:
        yield product, keys


def validate_zone_raster_inputs(config_path: Path) -> tuple[list[RasterInput], list[str]]:
    """Check the reviewed TIFF inventory without writing or scaling inputs.

    Args:
        config_path: Reviewed zonal config with an inline raster inventory.

    Returns:
        Unscaled raster inputs and reasons for reviewed exclusions.

    Raises:
        FileNotFoundError: An enabled raster is not available.
        ValueError: The inventory is invalid or produces ambiguous field names.
    """
    config_path = config_path.resolve()
    project = load_run_config(config_path)["project"]
    rasters, blocked = load_and_prepare_rasters(
        config_path, config_path.parent / project["raster_dir"], config_path.parent, apply_scale=False,
    )
    if len({raster.path.stem.casefold() for raster in rasters}) != len(rasters):
        raise ValueError("Enabled TIFFs must have unique filename stems for zonal output fields")
    for raster in rasters:
        if any(character in str(raster.path) for character in ",*?["):
            raise ValueError(f"Toolkit raster paths cannot contain commas or glob characters: {raster.path}")
    return rasters, blocked


def freeze_zone_zonals(
    zones: Mapping[str, tuple[Path, str]],
    config_path: Path,
    frozen_config: Path,
    work_dir: Path,
    raw_results_dir: Path,
    *,
    overlapping_manager_products: Collection[str] | None = None,
) -> ZoneRasterPlan:
    """Freeze fresh source-pixel sums for the exact prepared zone layers.

    A toolkit rasterization keeps one feature ID per pixel. Run overlapping
    manager types separately so each row receives its own footprint's sum.
    Empty products keep their schema but do not invoke the toolkit.

    Args:
        zones: The exact GeoPackage path and layer name for each zone product.
        config_path: Reviewed inventory and toolkit settings.
        frozen_config: Destination for the immutable runnable config.
        work_dir: Isolated directory for toolkit work and manager-zone shards.
        raw_results_dir: Isolated destination for the toolkit result pairs.
        overlapping_manager_products: Products flagged by the zone manifest.
            If omitted, conservatively split both manager products.

    Returns:
        The frozen config and exact expected toolkit output plan.

    Raises:
        FileExistsError: The config or a work-layer shard already exists.
        ValueError: Zones, their identifying keys, or the raster inventory are invalid.
    """
    config_path, frozen_config = config_path.resolve(), frozen_config.resolve()
    work_dir, raw_results_dir = work_dir.resolve(), raw_results_dir.resolve()
    if frozen_config.exists():
        raise FileExistsError(frozen_config)
    if set(zones) != set(ZONE_PRODUCTS):
        raise ValueError("Fresh raster zonals require exactly the six prepared zone products")
    split_products = _MANAGER_PRODUCTS if overlapping_manager_products is None else set(overlapping_manager_products)
    if not split_products <= _MANAGER_PRODUCTS:
        raise ValueError("Overlap flags must refer to manager-type zone products")
    source_bytes = config_path.read_bytes()
    rasters, blocked = validate_zone_raster_inputs(config_path)
    zone_jobs: list[ZoneJob] = []
    expected_jobs: list[ZoneRasterJob] = []
    skipped: list[str] = []
    for product in tqdm(ZONE_PRODUCTS, desc="Plan raster zonals", unit="zone"):
        source, layer = zones[product]
        source = source.resolve()
        key_fields = _KEY_FIELDS[product]
        split = product in split_products
        frame = pyogrio.read_dataframe(source, layer=layer, columns=list(key_fields), read_geometry=split)
        key_values = _zone_keys(frame, key_fields, product)
        if frame.empty:
            if product == "counties":
                raise ValueError("The counties zone must not be empty")
            skipped.append(product)
            continue

        for stem, keys in _job_key_groups(product, key_values, split):
            vector = source
            if split:
                vector = work_dir / "manager_zone_shards" / f"{stem}.gpkg"
                if vector.exists():
                    raise FileExistsError(vector)
                vector.parent.mkdir(parents=True, exist_ok=True)
                pyogrio.write_dataframe(frame.loc[keys.index], vector, layer=layer, promote_to_multi=True)
            zone_jobs.append(ZoneJob(
                key=stem, label=product, vector_path=vector, layer=layer,
                job_stem=stem, results_subdir=product, agg_field=", ".join(key_fields), operations="sum",
            ))
            expected_jobs.append(ZoneRasterJob(
                product, raw_results_dir / product / stem, key_fields,
                frozenset(keys.itertuples(index=False, name=None)),
            ))

    if config_path.read_bytes() != source_bytes:
        raise ValueError("Reviewed raster config changed during preparation; rerun")
    write_runner_config(
        frozen_config, work_dir, raw_results_dir, rasters, zone_jobs, config_path,
        [f"{product}: empty zone layer" for product in skipped],
    )
    return ZoneRasterPlan(
        frozen_config, tuple(f"sum_{raster.path.stem}" for raster in rasters),
        tuple(expected_jobs), tuple(blocked), tuple(skipped),
    )


def read_frozen_zone_raster_plan(
    zones: Mapping[str, tuple[Path, str]],
    frozen_config: Path,
    work_dir: Path,
    raw_results_dir: Path,
    *,
    expected_raster_columns: Collection[str],
    overlapping_manager_products: Collection[str] | None = None,
) -> ZoneRasterPlan:
    """Reconstruct a completed run's raster plan without touching the TIFFs.

    Args:
        zones: The saved GeoPackage path and layer name for each product.
        frozen_config: Original toolkit configuration from the completed run.
        work_dir: Original work directory containing any manager shards.
        raw_results_dir: Original directory for the raw toolkit result pairs.
        expected_raster_columns: Unscaled sum columns needed by valuation.
        overlapping_manager_products: Original manager-overlap flags, if
            available. Otherwise, infer splitting from the saved job names.

    Returns:
        A validated plan for read_zone_raster_sums.

    Raises:
        ValueError: The saved config is incompatible with the requested zones,
            unscaled sum schema, or original run paths.
    """
    if set(zones) != set(ZONE_PRODUCTS):
        raise ValueError("Reusing raster zonals requires exactly the six prepared zone products")
    frozen_config, work_dir, raw_results_dir = (path.resolve() for path in (frozen_config, work_dir, raw_results_dir))
    config = ConfigParser(interpolation=None)
    with frozen_config.open(encoding="utf-8") as stream:
        config.read_file(stream)
    if "project" not in config or config.defaults():
        raise ValueError(f"Invalid frozen raster config: {frozen_config}")
    project = config["project"]
    if (project.get("name") != frozen_config.stem
            or (frozen_config.parent / project.get("global_work_dir", "")).resolve() != work_dir):
        raise ValueError("Frozen raster config does not reference this run's work directory")

    try:
        raster_entries = json.loads(project["snapp_rasters"])
        inventory = json.loads(project["snapp_source_definition"])["project"]["raster_inventory"]
        unavailable = json.loads(project["snapp_unavailable_zones"])
        blocked = tuple(
            f"{entry['label'].strip()}: {entry.get('reason', 'not approved for this run')}"
            for entry in inventory if not entry["include"]
        )
        raster_paths = tuple(entry["analysis_path"] for entry in raster_entries)
        if not raster_paths or any(
            type(entry["scale"]) not in (int, float) or entry["scale"] != 1
            or entry["analysis_path"] != entry["source_path"]
            or Path(entry["analysis_path"]).suffix.lower() not in {".tif", ".tiff"}
            for entry in raster_entries
        ):
            raise ValueError("Expected the original, unscaled TIFF paths")
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid frozen raster metadata in {frozen_config}: {error}") from error
    raster_columns = tuple(f"sum_{Path(path).stem}" for path in raster_paths)
    if (len({name.casefold() for name in raster_columns}) != len(raster_columns)
            or set(raster_columns) != set(expected_raster_columns)):
        raise ValueError("Frozen raster sum columns do not match the valuation services")

    if overlapping_manager_products is None:
        split_products = {
            product for product in _MANAGER_PRODUCTS
            if any(section.startswith(f"job:{product}__manager_") for section in config.sections())
        }
    else:
        split_products = set(overlapping_manager_products)
    if not split_products <= _MANAGER_PRODUCTS:
        raise ValueError("Overlap flags must refer to manager-type zone products")

    jobs: list[ZoneRasterJob] = []
    skipped: list[str] = []
    expected_sections = {"project"}
    for product in tqdm(ZONE_PRODUCTS, desc="Validate frozen raster plan", unit="zone"):
        source, layer = zones[product]
        fields = _KEY_FIELDS[product]
        frame = pyogrio.read_dataframe(source, layer=layer, columns=list(fields), read_geometry=False)
        keys = _zone_keys(frame, fields, product)
        if keys.empty:
            if product == "counties":
                raise ValueError("The counties zone must not be empty")
            skipped.append(product)
            continue
        split = product in split_products
        for stem, group in _job_key_groups(product, keys, split):
            section = f"job:{stem}"
            expected_sections.add(section)
            if section not in config:
                raise ValueError(f"Frozen raster config is missing required [{section}]")
            job = config[section]
            vector = work_dir / "manager_zone_shards" / f"{stem}.gpkg" if split else source.resolve()
            output_base = raw_results_dir / product / stem
            expected = {"agg_vector": vector, "output_csv": output_base.with_suffix(".csv"),
                        "output_gpkg": output_base.with_suffix(".gpkg")}
            if any(not job.get(key) or (frozen_config.parent / job[key]).resolve() != path
                   for key, path in expected.items()):
                raise ValueError(f"Frozen [{section}] does not use this run's vector and result paths")
            if (job.get("agg_layer") != layer
                    or tuple(field.strip() for field in job.get("agg_field", "").split(",")) != fields
                    or job.get("operations") != "sum"
                    or tuple(path.strip() for path in job.get("base_raster_pattern", "").split(",")) != raster_paths
                    or set(job) != {*expected, "agg_layer", "agg_field", "operations", "base_raster_pattern"}):
                raise ValueError(f"Frozen [{section}] has an unexpected zone or raster sum schema")
            expected_keys = frozenset(group.itertuples(index=False, name=None))
            if split:
                shard = pyogrio.read_dataframe(vector, layer=layer, columns=list(fields), read_geometry=False)
                shard_keys = _zone_keys(shard, fields, stem)
                if frozenset(shard_keys.itertuples(index=False, name=None)) != expected_keys:
                    raise ValueError(f"Saved manager shard keys do not match the prepared zone: {vector}")
            jobs.append(ZoneRasterJob(product, output_base, fields, expected_keys))
    if set(config.sections()) != expected_sections:
        raise ValueError("Frozen raster config has unexpected jobs or sections")
    if unavailable != [f"{product}: empty zone layer" for product in skipped]:
        raise ValueError("Frozen raster config's empty zone list does not match prepared zones")
    return ZoneRasterPlan(frozen_config, raster_columns, tuple(jobs), blocked, tuple(skipped))


def read_zone_raster_sums(plan: ZoneRasterPlan) -> dict[str, pd.DataFrame]:
    """Read paired toolkit outputs and reconcile every result with its zone key.

    Args:
        plan: Output contract recorded before invoking the zonal toolkit.

    Returns:
        One frame per product, including empty products, with text keys and
        unscaled numeric raster sum columns.

    Raises:
        ValueError: Outputs are missing, ambiguous, stale, or malformed.
    """
    frames: dict[str, list[pd.DataFrame]] = {product: [] for product in ZONE_PRODUCTS}
    run_stamp = None
    for job in tqdm(plan.jobs, desc="Read raster sum tables", unit="job"):
        prefix = f"{job.output_base.name}_"
        candidates: dict[str, list[Path]] = {".csv": [], ".gpkg": []}
        for path in job.output_base.parent.glob(f"{prefix}*"):
            if path.suffix in candidates and _RESULT_STAMP.fullmatch(path.stem.removeprefix(prefix)):
                candidates[path.suffix].append(path)
        csvs, vectors = candidates[".csv"], candidates[".gpkg"]
        if (
            len(csvs) != 1 or len(vectors) != 1 or csvs[0].stem != vectors[0].stem
            or any(not path.is_file() or path.stat().st_size == 0 for path in (csvs[0], vectors[0]))
        ):
            raise ValueError(f"Expected one nonempty matched toolkit CSV/GPKG pair for {job.output_base}")
        stamp = csvs[0].stem.removeprefix(prefix)
        if run_stamp is not None and stamp != run_stamp:
            raise ValueError(f"Toolkit result for {job.output_base} has a different run timestamp: {stamp}")
        run_stamp = stamp

        columns = [*job.key_fields, *plan.raster_columns]
        frame = pd.read_csv(csvs[0], dtype={field: "string" for field in job.key_fields}, keep_default_na=False)
        if set(frame.columns) != set(columns):
            raise ValueError(f"Unexpected key or raster sum columns in {csvs[0]}")
        keys = list(frame[list(job.key_fields)].itertuples(index=False, name=None))
        if len(keys) != len(job.expected_keys) or set(keys) != job.expected_keys:
            raise ValueError(f"Toolkit output keys do not match the prepared zone: {csvs[0]}")
        sums = frame[list(plan.raster_columns)].apply(pd.to_numeric, errors="coerce").astype("float64")
        if not np.isfinite(sums.to_numpy()).all():
            raise ValueError(f"Toolkit output contains missing or non-finite raster sums: {csvs[0]}")
        frames[job.product].append(pd.concat([frame[list(job.key_fields)], sums], axis=1))

    return {
        product: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({
            **{key: pd.Series(dtype="string") for key in _KEY_FIELDS[product]},
            **{column: pd.Series(dtype="float64") for column in plan.raster_columns},
        })
        for product, parts in frames.items()
    }
