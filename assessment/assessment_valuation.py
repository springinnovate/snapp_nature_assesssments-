"""Harmonize raw ecosystem-service measures for the six assessment products."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, closing, contextmanager
from multiprocessing import get_context
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from typing import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import network
import shapely
from tqdm import tqdm
import yaml

from assessment.assessment_coastal import CoastalPlan
from assessment.assessment_fisheries import FisheriesPlan
from assessment.assessment_geography import CONUS_STATE_FIPS
from assessment.assessment_zone_geometry import ZONE_PRODUCTS
from assessment.build_county_ecosystem_service_valuations_2026 import (
    BASE_FIELDS,
    adjusted_county_fields,
    compute_task,
    load_export_rules,
    plan_county_export,
    raster_service_sums,
    read_db,
    validate_crosswalk,
)
from assessment.county_derived_metrics import calculate_county_metrics


COUNTY_TABLES = ("grazing", "timber", "air_quality", "urban_heat", "physical_health", "jobs", "recreation")
VALUE_SOURCES = (
    *COUNTY_TABLES, "air_quality_county_aliases", "marine_fisheries", "inland_fisheries", "coastline_lines",
    "freshwater_polygons", "great_lakes_features", "inland_fisheries_exclusions",
    "corals", "mangroves",
)
CONFIG_FILES = ("sources.csv", "adjustment_factors.csv", "ct_county_crosswalk.csv",
                "export_rules.yaml", "derived_metrics.yaml", "great_lakes_nhd_features.csv",
                "inland_fisheries_exclusions.csv", "county_area_exclusions.yaml")


@contextmanager
def _valuation_pool(workers):
    """Own one spawned pool; limit native threads before child imports NumPy."""
    if workers < 1:
        raise ValueError("Valuation workers must be positive")
    limits = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    settings = dict.fromkeys(limits, "1")
    settings["PROJ_NETWORK"] = "ON" if network.is_network_enabled() else "OFF"
    previous = {name: os.environ.get(name) for name in settings}
    try:
        os.environ.update(settings)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            yield pool
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_source_tasks(pool, plans, table_tasks, workers):
    """Interleave service families and dependent batches in one bounded queue.

    Only this parent submits work. Plans supply (key, function, args) tasks;
    accepting a result may release dependent tasks. Result keys, rather than
    completion order, determine output row and summation order.
    """
    ready = deque([("tables", iter(table_tasks)),
                   *((name, iter(plan.tasks())) for name, plan in plans.items())])
    pending, tables = {}, {}
    with ExitStack() as stack:
        progress = {
            name: stack.enter_context(tqdm(total=total, desc=label, unit=unit, position=position))
            for position, (name, total, label, unit) in enumerate((
                ("tables", len(COUNTY_TABLES), "Join county service tables", "service"),
                ("fisheries", plans["fisheries"].task_count, "Measure fisheries zones", "county"),
                ("coastal", None, "Value coastal batches", "batch"),
            ))
        }
        try:
            while ready or pending:
                # Keep large geometry payloads bounded, including on Windows.
                while ready and len(pending) < 2 * workers:
                    family, tasks = ready.popleft()
                    task = next(tasks, None)
                    if task is None:
                        continue
                    key, function, args = task
                    pending[pool.submit(function, *args)] = family, key
                    ready.append((family, tasks))
                if not pending:
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    family, key = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        error.add_note(f"Valuation task: {family} {key}")
                        raise
                    if family == "tables":
                        tables[key] = result
                    else:
                        released = plans[family].accept(key, result)
                        if released is not None:
                            ready.appendleft((family, iter(released)))
                    progress[family].update(1)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return tables, plans["fisheries"].finish(), plans["coastal"].finish()


def _collect_service_values(zones, paths, county_base, crosswalk, coral_reference_counties, workers,
                            *, fisheries_cache_dir=None, cache_report=None):
    """Prepare bounded spatial jobs and share workers with county-table joins."""
    plans = {
        "fisheries": FisheriesPlan(zones, paths, cache_dir=fisheries_cache_dir),
        "coastal": CoastalPlan(zones, paths, coral_reference_counties=coral_reference_counties),
    }
    table_tasks = ((task, compute_task, (task, paths, county_base, crosswalk, 1))
                   for task in COUNTY_TABLES)
    print(f"Value services with one shared pool: {workers} processes (1 native thread each)", flush=True)
    with _valuation_pool(workers) as pool:
        results = _run_source_tasks(pool, plans, table_tasks, workers)
    if cache_report is not None:
        cache_report.update(plans["fisheries"].cache_info)
    return results


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
    allocation = derived["sources"]["gdp"].get("allocation_crosswalk")
    denominator_paths = [(data_root / source["path"]).resolve() for source in derived["sources"].values()]
    reporting_area_paths = {
        (data_root / derived["sources"][name]["path"]).resolve()
        for name in ("county_area", "county_area_inclusive", "natural_areas")
    }
    if allocation:
        denominator_paths.append((data_root / allocation).resolve())
    for path in [*(paths[name] for name in VALUE_SOURCES),
                 *(config_dir / name for name in CONFIG_FILES),
                 *denominator_paths]:
        if not path.is_file():
            if path in reporting_area_paths:
                raise FileNotFoundError(
                    f"Missing county reporting-area input: {path}. Run "
                    'python -m assessment.workflows.prepare_county_reporting_areas '
                    '--runner "<path to runner.py>" first.')
            raise FileNotFoundError(f"Missing valuation input: {path}")
    return paths, factors, rules, derived


def calculate_county_land_service_metrics(
    products: Mapping[str, pd.DataFrame], ratios: Mapping[str, str], per_hectare: Mapping[str, str]
) -> pd.DataFrame:
    """Calculate land-category value shares and value per land hectare.

    Args:
        products: Valued products containing GEOID and total_ecosystem_services.
            Land-category products also contain area_ha_land.
        ratios: Output field names mapped to by-county land products.
        per_hectare: Per-hectare output names mapped to by-county land products.

    Returns:
        Columns in county row order, using totals excluding recreation. An
        absent category contributes zero to its county value share, but has
        no per-hectare value. Missing totals and missing, nonpositive, or
        nonfinite denominators produce NaN.

    Raises:
        ValueError: Products, county keys, or output names are invalid.
    """
    expected = {"public_lands_by_county", "conservation_lands_by_county", "bbb_lands_by_county"}
    if any(len(mapping) != len(expected) or set(mapping.values()) != expected
           for mapping in (ratios, per_hectare)):
        raise ValueError("County land-service metrics require the three by-county land products")
    counties = products["counties"]
    if (set(ratios).intersection(per_hectare)
            or (set(ratios) | set(per_hectare)).intersection(counties.columns)):
        raise ValueError("County land-service fields collide with existing attributes")
    totals, areas = {}, {}
    for product in ("counties", *ratios.values()):
        frame = products[product]
        ids = pd.Index(frame["GEOID"].astype("string"), name="GEOID")
        if not ids.is_unique or not ids.str.fullmatch(r"\d{5}").fillna(False).all():
            raise ValueError(f"{product}: land-service metrics require unique five-character GEOIDs")
        values = pd.to_numeric(frame["total_ecosystem_services"], errors="raise")
        totals[product] = pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=ids)
        if product != "counties":
            land = pd.to_numeric(frame["area_ha_land"], errors="raise")
            areas[product] = pd.Series(land.to_numpy(dtype=float, na_value=np.nan), index=ids)
    county_total = totals["counties"]
    denominator = county_total.to_numpy()
    usable = np.isfinite(denominator) & (denominator > 0)
    metrics = pd.DataFrame(index=counties.index)
    for output, product in ratios.items():
        if not totals[product].index.isin(county_total.index).all():
            raise ValueError(f"{product}: land-service metrics contain GEOIDs absent from counties")
        # Only absent category rows become zero; observed missing totals stay missing.
        numerator = totals[product].reindex(county_total.index, fill_value=0.0).to_numpy()
        metrics[output] = np.divide(
            numerator, denominator, out=np.full(len(counties), np.nan),
            where=usable & np.isfinite(numerator),
        )
    for output, product in per_hectare.items():
        numerator = totals[product].reindex(county_total.index).to_numpy()
        denominator = areas[product].reindex(county_total.index).to_numpy()
        metrics[output] = np.divide(
            numerator, denominator, out=np.full(len(counties), np.nan),
            where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0),
        )
    return metrics


def value_zones(
    zones: Mapping[str, gpd.GeoDataFrame],
    raster_sums: Mapping[str, pd.DataFrame],
    data_root: Path,
    config_dir: Path,
    *,
    coral_reference_counties: gpd.GeoDataFrame,
    workers: int,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, pd.DataFrame], dict]:
    """Calculate adjusted values, totals, and county ratios on prepared zones.

    Args:
        zones: Six prepared geometry frames with stable county/manager keys.
        raster_sums: Raw toolkit sums and key columns for each product.
        data_root: Root containing the local input data.
        config_dir: Tracked sources, factors, and output rules.
        coral_reference_counties: Original county reference used to assign
            coral points before retaining only CONUS values.
        workers: Processes shared by county tables, fisheries and coastal batches.

    Returns:
        Valued geometry frames, county audit tables, and run-level coverage QA.

    Raises:
        ValueError: Keys, units, expected services, or reconciliations fail.
    """
    if set(zones) != set(ZONE_PRODUCTS) or set(raster_sums) != set(zones):
        raise ValueError("Valuation requires the same six geometry and raster products")
    paths, factors, rules, derived = load_valuation_config(data_root, config_dir)
    counties = zones["counties"].drop(
        columns=["pct_public_lands", "pct_conservation_lands", "pct_bbb_lands"], errors="ignore")
    # Reused zone files may still carry the replaced land-only percentages.
    zones = dict(zones, counties=counties)
    if not counties.GEOID.astype("string").str[:2].isin(CONUS_STATE_FIPS).all():
        raise ValueError("County inputs must be within CONUS")
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
    fisheries_cache = {}
    tables, fisheries, coastal = _collect_service_values(
        zones, paths, county_base, crosswalk, coral_reference_counties, workers,
        fisheries_cache_dir=data_root / "processing_outputs" / "fisheries_weights",
        cache_report=fisheries_cache,
    )
    # Completion order must not affect exported field or provenance row order.
    for task in COUNTY_TABLES:
        result = tables[task]
        originals.update(result["originals"])
        task_metadata.extend(result["metadata"])
        status_frames.extend(result["statuses"])
        if task == "air_quality":
            air_summary = result["summary_fields"]
    tabular = pd.DataFrame(originals, index=county_base.index)
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
            pd.DataFrame(raster_service_sums(raw_sums), index=zone.index),
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
            denominators = denominators.drop(columns="es_total").set_axis(zone.index)
            # Report each footprint against both county area definitions.
            # Allocation shares still use land-only areas and ALAND.
            footprint_rules = derived["county_footprint_ratios"]
            divisors = {}
            for suffix, field in footprint_rules["denominators"].items():
                county_area = denominators[field]
                divisors[suffix] = county_area.where(np.isfinite(county_area) & county_area.gt(0))
            footprint_ratios = pd.DataFrame({
                f"{output}{suffix}": zone[numerator] / divisor
                for output, numerator in footprint_rules["numerators"].items()
                for suffix, divisor in divisors.items()
            }, index=zone.index)
            adjusted = pd.concat([adjusted, denominators, footprint_ratios], axis=1)
            county_audits.update({
                "integration_county_table_methods": pd.DataFrame(task_metadata),
                "integration_county_table_status": pd.concat(status_frames, ignore_index=True),
            })
        if not adjusted.columns.is_unique or set(adjusted).intersection(zone.columns):
            raise ValueError(f"{product}: exported valuation fields collide with zone attributes")
        result_frames[product] = gpd.GeoDataFrame(pd.concat([zone, adjusted], axis=1),
                                                 geometry=zone.geometry.name, crs=zone.crs)
    county_land_metrics = calculate_county_land_service_metrics(
        result_frames, derived["county_ecosystem_service_ratios"], derived["county_land_service_value_per_ha"],
    )
    result_frames["counties"] = pd.concat([result_frames["counties"], county_land_metrics], axis=1)
    return result_frames, county_audits, {
        "fisheries_weights_cache": fisheries_cache,
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
        destination: New GeoPackage path in the selected output directory.
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
