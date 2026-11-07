"""Harmonize raw ecosystem-service measures for the six assessment products."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from typing import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from tqdm import tqdm
import yaml

from assessment_coastal import value_coastal
from assessment_fisheries import value_fisheries
from assessment_zone_geometry import ZONE_PRODUCTS
from build_county_ecosystem_service_valuations_2026 import (
    BASE_FIELDS,
    adjusted_county_fields,
    compute_task,
    load_export_rules,
    numbers,
    plan_county_export,
    read_db,
    validate_crosswalk,
)
from county_derived_metrics import calculate_county_metrics


COUNTY_TABLES = ("grazing", "timber", "air_quality", "urban_heat", "physical_health", "jobs", "recreation")
VALUE_SOURCES = (
    *COUNTY_TABLES, "marine_fisheries", "inland_fisheries", "coastline_lines",
    "freshwater_polygons", "great_lakes_features", "corals", "mangroves",
)
CONFIG_FILES = ("sources.csv", "adjustment_factors.csv", "ct_county_crosswalk.csv",
                "export_rules.yaml", "derived_metrics.yaml")


def load_valuation_config(data_root: Path, config_dir: Path) -> tuple[dict, pd.DataFrame, dict, dict]:
    """Validate reviewed service sources, factors, and export policies.

    Args:
        data_root: Root containing local analysis data.
        config_dir: Directory containing tracked valuation settings.

    Returns:
        Source paths, factors indexed by service, export rules, and derived rules.

    Raises:
        FileNotFoundError: A required source is unavailable.
        ValueError: Source names or factors are invalid.
    """
    sources = pd.read_csv(config_dir / "sources.csv")
    factors = pd.read_csv(config_dir / "adjustment_factors.csv").set_index("service")
    if sources.source.duplicated().any() or factors.index.duplicated().any():
        raise ValueError("Duplicate source or adjustment-factor definitions")
    if not np.isfinite(factors.factor).all() or not factors.factor.gt(0).all():
        raise ValueError("Every service needs a finite positive adjustment factor")
    paths = {row.source: (data_root / row.path).resolve() for row in sources.itertuples()}
    missing = set(VALUE_SOURCES) - set(paths)
    if missing:
        raise ValueError(f"Missing service sources in sources.csv: {sorted(missing)}")
    rules = load_export_rules(config_dir / "export_rules.yaml")
    derived = yaml.safe_load((config_dir / "derived_metrics.yaml").read_text(encoding="utf-8"))
    for path in [*(paths[name] for name in VALUE_SOURCES),
                 *(config_dir / name for name in CONFIG_FILES),
                 *((data_root / source["path"]).resolve() for source in derived["sources"].values())]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing valuation input: {path}")
    return paths, factors, rules, derived


def value_zones(
    zones: Mapping[str, gpd.GeoDataFrame],
    raster_sums: Mapping[str, pd.DataFrame],
    data_root: Path,
    config_dir: Path,
    *,
    workers: int,
    geometry_workers: int,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, pd.DataFrame], dict]:
    """Calculate adjusted values, totals, and county ratios on prepared zones.

    Args:
        zones: Six prepared geometry frames with stable county/manager keys.
        raster_sums: Raw toolkit sums and key columns for each product.
        data_root: Root containing the local input data.
        config_dir: Tracked sources, factors, and output rules.
        workers: Concurrent county table tasks.
        geometry_workers: Concurrent spatial measurement workers.

    Returns:
        Valued geometry frames, county audit tables, and run-level coverage QA.

    Raises:
        ValueError: Keys, units, expected services, or reconciliations fail.
    """
    if set(zones) != set(ZONE_PRODUCTS) or set(raster_sums) != set(zones):
        raise ValueError("Valuation requires the same six geometry and raster products")
    paths, factors, rules, derived = load_valuation_config(data_root, config_dir)
    counties = zones["counties"]
    county_base = pd.DataFrame(counties.drop(columns=counties.geometry.name)).set_index("GEOID")
    crosswalk = pd.read_csv(config_dir / "ct_county_crosswalk.csv",
                            dtype={"source_geoid": str, "target_geoid": str})
    validate_crosswalk(crosswalk, county_base.index)
    # Reuse the county builder's source-specific joins, CT crosswalk and
    # timber clamp. Read each source once regardless of the number of products.
    originals = {}
    air_summary = None
    task_metadata = []
    status_frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(compute_task, task, paths, county_base, crosswalk, geometry_workers)
                   for task in COUNTY_TABLES]
        for task, future in tqdm(zip(COUNTY_TABLES, pending), total=len(pending),
                                 desc="Join county service tables", unit="service"):
            result = future.result()
            originals.update(result["originals"])
            task_metadata.extend(result["metadata"])
            status_frames.extend(result["statuses"])
            if task == "air_quality":
                air_summary = result["summary_fields"]
    tabular = pd.DataFrame(originals, index=county_base.index)
    fisheries = value_fisheries(zones, paths, geometry_workers)
    coastal = value_coastal(zones, paths, geometry_workers)
    result_frames = {}
    county_audits = {}
    coverage = {}
    components = derived["es_total"]["components"]
    recreation_field = rules["rename_fields"].get("recreation_adj", "recreation_adj")
    if components.count(recreation_field) != 1 or len(set(components)) != len(components):
        raise ValueError("The total component list must include recreation exactly once and no duplicates")
    non_recreation_components = [field for field in components if field != recreation_field]
    blank_value = derived["es_total"]["blank_value"]
    if blank_value != 0:
        raise ValueError("Blank service values must contribute zero to assessment totals")

    for product, zone in tqdm(zones.items(), desc="Adjust and total zone values", unit="product"):
        is_county = product == "counties"
        keys = ["GEOID", "Mang_Type"] if product.endswith("_and_manager_type") else ["GEOID"]
        zone_keys = pd.MultiIndex.from_frame(zone[keys])
        raw_sums = raster_sums[product]
        raw_keys = pd.MultiIndex.from_frame(raw_sums[keys])
        if not zone_keys.is_unique or not raw_keys.is_unique or set(raw_keys) != set(zone_keys):
            raise ValueError(f"{product}: zonal keys differ from prepared geometry keys")
        if not set(BASE_FIELDS.values()) <= set(raw_sums):
            raise ValueError(f"{product}: missing configured raster sums: {sorted(set(BASE_FIELDS.values()) - set(raw_sums))}")
        raw_sums = raw_sums.set_axis(raw_keys).reindex(zone_keys).set_axis(zone.index)
        base = pd.concat([
            pd.DataFrame(zone.drop(columns=zone.geometry.name)), raw_sums[list(BASE_FIELDS.values())],
        ], axis=1)
        if not base.columns.is_unique:
            raise ValueError(f"{product}: prepared attributes collide with raster sums")
        shares = pd.Series(1.0, index=zone.index) if is_county else zone["county_land_share"]
        if not np.isfinite(shares).all() or not shares.between(0, 1).all():
            raise ValueError(f"{product}: county land shares must be finite and in [0, 1]")
        values = tabular.reindex(zone.GEOID).set_axis(zone.index)
        if not is_county:
            values = values.drop(columns="recreation")
        values = values.mul(shares, axis=0)
        summary = air_summary.reindex(zone.GEOID).set_axis(zone.index).mul(shares, axis=0)
        spatial = [fisheries[product], coastal[product]]
        if any(not frame.index.equals(zone.index) for frame in spatial):
            raise ValueError(f"{product}: spatial value rows do not match zone rows")
        raw_values = pd.concat([
            values, *spatial,
            pd.DataFrame({service: numbers(raw_sums[field]) for service, field in BASE_FIELDS.items()}),
        ], axis=1)
        expected = set(factors.index) - (set() if is_county else {"recreation"})
        if not raw_values.columns.is_unique or set(raw_values) != expected:
            raise ValueError(f"{product}: services do not match the adjustment table")
        if np.isinf(raw_values.astype(float)).any().any():
            raise ValueError(f"{product}: nonfinite raw service value")
        active_factors = factors.loc[[service for service in factors.index if service in expected]]
        plan = plan_county_export(base.columns, active_factors.index, rules)
        adjusted, _ = adjusted_county_fields(base, raw_values, summary, active_factors, plan)
        if not set(non_recreation_components) <= set(adjusted):
            raise ValueError(f"{product}: an adjusted total component is missing")
        selected = adjusted[non_recreation_components]
        adjusted["total_ecosystem_services"] = selected.fillna(blank_value).sum(axis=1)
        coverage[product] = {field: int(selected[field].isna().sum()) for field in selected}
        if is_county:
            adjusted["total_ecosystem_services_w_rec"] = (
                adjusted["total_ecosystem_services"] + adjusted[recreation_field].fillna(blank_value)
            )
            coverage[product][recreation_field] = int(adjusted[recreation_field].isna().sum())
            denominators, county_audits = calculate_county_metrics(
                adjusted.set_axis(county_base.index), data_root, config_dir / "derived_metrics.yaml"
            )
            if not np.allclose(denominators["es_total"], adjusted["total_ecosystem_services_w_rec"],
                               rtol=1e-12, atol=0.001):
                raise ValueError("County total with recreation differs from the configured legacy component sum")
            adjusted = pd.concat([adjusted, denominators.drop(columns="es_total").set_axis(zone.index)], axis=1)
            county_audits.update({
                "integration_county_table_methods": pd.DataFrame(task_metadata),
                "integration_county_table_status": pd.concat(status_frames, ignore_index=True),
            })
        if not adjusted.columns.is_unique or set(adjusted).intersection(zone.columns):
            raise ValueError(f"{product}: exported valuation fields collide with zone attributes")
        result_frames[product] = gpd.GeoDataFrame(pd.concat([zone, adjusted], axis=1),
                                                 geometry=zone.geometry.name, crs=zone.crs)
    return result_frames, county_audits, {
        "total_components": non_recreation_components,
        "blank_component_rows": coverage,
        "county_only_component": recreation_field,
    }


def write_zone_pair(
    frame: gpd.GeoDataFrame, destination: Path, product: str, tables: Mapping[str, pd.DataFrame]
) -> tuple[Path, Path]:
    """Stage, verify, and publish one geometry/attribute pair without overwrite.

    Args:
        frame: Prepared geometry and adjusted attributes for a product.
        destination: New GeoPackage path under its product directory.
        product: GeoPackage feature layer name.
        tables: Optional provenance tables to embed alongside the feature layer.

    Returns:
        Matching CSV and GeoPackage paths.

    Raises:
        FileExistsError: An output already exists.
        ValueError: Integrity, attributes or geometry verification fails.
    """
    pair = (destination.with_suffix(".csv"), destination)
    for path in pair:
        if path.exists():
            raise FileExistsError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="publish-", dir=destination.parent) as temporary:
        staged = (Path(temporary) / pair[0].name, Path(temporary) / destination.name)
        pyogrio.write_dataframe(frame, staged[1], layer=product, geometry_type="MultiPolygon")
        if tables:
            with closing(sqlite3.connect(staged[1])) as db:
                for name, table in tables.items():
                    table.to_sql(name, db, index=False, if_exists="fail")
        with closing(read_db(staged[1])) as db:
            if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise ValueError(f"{product}: GeoPackage integrity check failed")
        saved = pyogrio.read_dataframe(staged[1], layer=product)
        if not saved.empty and (saved.crs != frame.crs
                                or not shapely.equals_exact(saved.geometry, frame.geometry, 0).all()):
            raise ValueError(f"{product}: saved geometry differs from prepared zones")
        attributes = pd.DataFrame(saved.drop(columns=saved.geometry.name))
        expected = pd.DataFrame(frame.drop(columns=frame.geometry.name))
        pd.testing.assert_frame_equal(attributes, expected, check_dtype=False, check_exact=False, rtol=1e-12)
        attributes.to_csv(staged[0], index=False)
        text_fields = {name: str for name in attributes if not pd.api.types.is_numeric_dtype(attributes[name])}
        readback = pd.read_csv(staged[0], dtype=text_fields, keep_default_na=False, na_values=[""],
                               float_precision="round_trip")
        pd.testing.assert_frame_equal(attributes, readback, check_dtype=False, check_exact=False, rtol=1e-12)
        published = []
        try:
            for path, temporary_path in zip(pair, staged):
                path.hardlink_to(temporary_path)
                published.append(path)
        except Exception:
            for path in published:
                path.unlink()
            raise
    return pair
