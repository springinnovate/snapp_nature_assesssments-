"""Build paired CSV/GeoPackage county ecosystem-service valuation exports.

Join county estimates, allocate state fisheries, mangrove polygons and coral
points to counties, and apply the recorded valuation adjustments.
The output preserves county boundaries and identifiers, groups county sums and
other adjusted totals before their factors, and omits non-sum statistics.

Run: python build_county_ecosystem_service_valuations_2026.py
Inputs: data/workflow_assets/county_integration/sources.csv (paths under data/).
Export policy: export_rules.yaml in the same configuration directory.
Default outputs: data/analysis_results/combined/snapp_ecosystem_services_counties_<timestamp>.{csv,gpkg}.
Source files are never modified. See docs/county_valuation_integration_2026.md
for input releases, allocation methods, units, and interpretation limits.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml
from osgeo import gdal, ogr
from tqdm import tqdm

from assessment import REPO_ROOT as ROOT
from assessment.coral_county_allocation import allocate_coral_values
from assessment.mangrove_county_allocation import allocate_mangrove_values
from assessment.county_derived_metrics import calculate_county_metrics

CONFIG = ROOT / "data/workflow_assets/county_integration"
AREA_CRS = "EPSG:6933"  # Equal area, including Alaska, Hawaii and US territories.
# Assessment scope: do not assign neighboring countries' study-unit totals
# merely because their offshore geometry also intersects a U.S. county.
MANGROVE_COUNTRIES = ("USA", "PRI", "VIR")
# Recognize generated statistics so YAML export rules cannot remove county attributes.
NON_SUM_FIELD = re.compile(
    r"^(?:mean|median|min|max|stdev|std|variance|var|p\d+(?:\.\d+)?|"
    r"count|valid_count|total_count|nodata_count|nonzero_count|"
    r"area_ha_total|area_ha_valid|area_total|area_valid|proportion_[a-z_]+)_"
)
CARBON_PIXEL_AREA_HA = 0.09  # The carbon density source has 30 m × 30 m pixels.
BASE_FIELDS = {
    "pollination": "sum_pollination_attributed_annual_crop_yield_value",
    "carbon_ha": "sum_totalC_tCO2e_ha_2020",
    "dredging": "sum_avoided_dredging_costs_raster",
    "sdwa": "sum_avoided_sdwa_health_costs_raster",
    "water_treatment": "sum_avoided_treatment_costs_raster",
    "provisioning": "sum_Wval_sw_2020usd_total",
    "provisioning_irr": "sum_Wval_sw_2020usd_irr",
    "provisioning_pow": "sum_Wval_sw_2020usd_pow",
    "provisioning_pub": "sum_Wval_sw_2020usd_pub",
    "flood_npv": "sum_marginal_npv_masked_to_wetlands",
    "flood_annual": "sum_annual_value_masked_to_wetlands",
    "mental_health": "sum_mental_health_national_existing_greenness_cost_90m",
}
JOB_FIELDS = {
    "jobs_cultural": "Cultural Activities Total Wages",
    "jobs_raw_materials": "Raw Materials Total Wages",
    "jobs_processing": "Processing Total Wages",
    "jobs_harvesting": "Harvesting Total Wages",
}
STATE_NAMES = dict(zip(
    "ALABAMA|ALASKA|ARIZONA|ARKANSAS|CALIFORNIA|COLORADO|CONNECTICUT|DELAWARE|DISTRICT OF COLUMBIA|FLORIDA|GEORGIA|HAWAII|IDAHO|ILLINOIS|INDIANA|IOWA|KANSAS|KENTUCKY|LOUISIANA|MAINE|MARYLAND|MASSACHUSETTS|MICHIGAN|MINNESOTA|MISSISSIPPI|MISSOURI|MONTANA|NEBRASKA|NEVADA|NEW HAMPSHIRE|NEW JERSEY|NEW MEXICO|NEW YORK|NORTH CAROLINA|NORTH DAKOTA|OHIO|OKLAHOMA|OREGON|PENNSYLVANIA|RHODE ISLAND|SOUTH CAROLINA|SOUTH DAKOTA|TENNESSEE|TEXAS|UTAH|VERMONT|VIRGINIA|WASHINGTON|WEST VIRGINIA|WISCONSIN|WYOMING|PUERTO RICO|VIRGIN ISLANDS".split("|"),
    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56 72 78".split(),
))


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def read_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def read_attributes(path):
    with closing(read_db(path)) as con:
        layers = con.execute("SELECT table_name,column_name FROM gpkg_geometry_columns").fetchall()
        if len(layers) != 1:
            raise ValueError(f"Expected one feature layer: {path}")
        layer, geom = layers[0]
        fields = [row[1] for row in con.execute(f"PRAGMA table_info({quote(layer)})") if row[1] != geom]
        frame = pd.read_sql_query(f"SELECT {','.join(map(quote, fields))} FROM {quote(layer)}", con)
    return frame, layer, geom


def fips(value):
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,5}(?:\.0)?", text):
        raise ValueError(f"Invalid county FIPS: {value!r}")
    return text.removesuffix(".0").zfill(5)


def numbers(series):
    """Parse currency without silently turning nonempty invalid values into zero."""
    text = series.astype("string").str.strip()
    # The heat CSV uses the Excel accounting-format "$ -" representation of zero.
    text = text.mask(text.str.fullmatch(r"\$\s*-", na=False), "0")
    cleaned = text.str.replace(r"[$,]", "", regex=True).str.strip()
    cleaned = cleaned.replace("", pd.NA)
    values = pd.to_numeric(cleaned, errors="raise").astype(float)
    if np.isinf(values).any():
        raise ValueError(f"Non-finite numeric values in {series.name}")
    return values


def raster_service_sums(frame):
    """Read raster aggregates and integrate the carbon density over pixel area.

    Args:
        frame: Unscaled zonal sums, including the 30 m carbon density sum.

    Returns:
        Unadjusted service series. The integrated carbon series is in tonnes
        CO2e, before the monetary factor is applied.
    """
    values = {service: numbers(frame[field]) for service, field in BASE_FIELDS.items()}
    values["carbon_pixel"] = values["carbon_ha"] * CARBON_PIXEL_AREA_HA
    return values


def close(actual, expected, label):
    if not np.isclose(actual, expected, rtol=1e-9, atol=0.01):
        raise ValueError(f"{label}: {actual} != {expected}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_crosswalk(crosswalk, county_ids):
    if crosswalk.duplicated(["source_geoid", "target_geoid"]).any():
        raise ValueError("Duplicate crosswalk pairs")
    if not set(crosswalk.target_geoid) <= set(county_ids):
        raise ValueError("Crosswalk targets absent from base counties")
    if crosswalk.weight.isna().any() or not np.isfinite(crosswalk.weight).all() or (crosswalk.weight <= 0).any():
        raise ValueError("Crosswalk weights must be finite and positive")
    for key, total in crosswalk.groupby("source_geoid").weight.sum().items():
        close(total, 1, f"Crosswalk weights for {key}")


def county_values(frame, key, field, ids, crosswalk, *, aggregate=False, zero_missing=False):
    """Join unique county totals or sum detail, preserving incomplete coverage.

    CT historical totals use land-area shares. A target receiving any unknown
    contribution stays null (except the explicitly requested timber-zero policy).
    Unmatched source amounts and known partial amounts are reconciled separately.
    """
    data = pd.DataFrame({"key": frame[key].map(fips), "value": numbers(frame[field])})
    if not aggregate and data.key.duplicated().any():
        raise ValueError(f"Duplicate county totals in {field}")
    grouped = data.groupby("key").value
    totals = grouped.sum(min_count=1)
    totals[grouped.count() < grouped.size()] = np.nan
    original_known_total = float(data.value.sum())
    ct = set(crosswalk.source_geoid)
    # Missing old counties cannot be inferred as zero for health/heat values.
    if set(totals.index) & ct:
        if set(totals.index) & set(crosswalk.target_geoid):
            raise ValueError("Source mixes historical Connecticut counties and planning regions")
        totals = totals.reindex(totals.index.union(pd.Index(sorted(ct))))
    lookup = {k: list(zip(g.target_geoid, g.weight)) for k, g in crosswalk.groupby("source_geoid")}
    result = pd.Series(np.nan, index=ids, dtype=float)
    status = pd.Series("missing_source", index=ids, dtype=object)
    contributions = {k: [] for k in ids}
    issues = []
    unmatched = 0.0
    for source_id, value in totals.items():
        targets = [(source_id, 1.0)] if source_id in contributions else lookup.get(source_id, [])
        if not targets:
            issues.append({"source_key": source_id, "reason": "unmatched_county", "value": None if pd.isna(value) else float(value)})
            if pd.notna(value):
                unmatched += value
            continue
        for target, weight in targets:
            contributions[target].append((value * weight, source_id != target, pd.isna(value)))
    for target, parts in contributions.items():
        if not parts:
            if zero_missing:
                result[target] = 0.0
                status[target] = "missing_source_zero"
            continue
        missing = any(p[2] for p in parts)
        known = sum(p[0] for p in parts if not p[2])
        if missing and not zero_missing:
            status[target] = "incomplete_source"
            if known:
                issues.append({"source_key": target, "reason": "known_partial_withheld", "value": float(known)})
        else:
            result[target] = known
            status[target] = "missing_value_zero" if missing else ("ct_land_area_estimate" if any(p[1] for p in parts) else "source")
    # For detail groups with null components, retain known partial source values in audit.
    partial_input = float(data.loc[data.key.isin(totals[totals.isna()].index), "value"].sum())
    withheld = sum(x["value"] for x in issues if x["reason"] == "known_partial_withheld")
    close(float(result.sum()) + unmatched + withheld + partial_input, original_known_total, field)
    audit = {"input_known_total": original_known_total, "unmatched_total": float(unmatched),
             "withheld_partial_total": float(withheld + partial_input)}
    return result, status, issues, audit


def air_quality_summary(frame, ids, crosswalk, county_totals):
    """Summarize unadjusted annual dollars across target-county categories.

    Historical Connecticut records are allocated by the existing land-area
    crosswalk, then recombined by land cover and pollutant before summarizing.
    A category with any unknown contribution stays unknown. Monetary summaries
    stay null for an incomplete county; count fields describe known categories.

    Args:
        frame: Source records with FIPS, land, pollutant, and dollars_annual.
        ids: Target county GEOID index.
        crosswalk: Historical source-to-target county land-area weights.
        county_totals: Reconciled totals returned by county_values.

    Returns:
        DataFrame indexed by GEOID with sum, mean, median, min, max, population
        stdev, valid_count, and total_count fields suffixed dollars_annual.

    Raises:
        ValueError: Categories are invalid/duplicated or totals do not reconcile.
    """
    data = frame[["FIPS", "land", "pollutant"]].copy()
    data["FIPS"] = data.FIPS.map(fips)
    data["value"] = numbers(frame.dollars_annual)
    if data[["land", "pollutant"]].isna().any().any():
        raise ValueError("Air-quality land and pollutant categories must be non-null")
    if data.duplicated(["FIPS", "land", "pollutant"]).any():
        raise ValueError("Duplicate air-quality county/land/pollutant rows")
    if set(data.FIPS) & set(crosswalk.source_geoid):
        # Match county_values' incomplete-CT policy when an old county is absent.
        missing = sorted(set(crosswalk.source_geoid) - set(data.FIPS))
        if missing:
            categories = data[["land", "pollutant"]].drop_duplicates()
            unknown = pd.DataFrame({"FIPS": missing}).merge(categories, how="cross")
            unknown["value"] = np.nan
            data = pd.concat([data, unknown], ignore_index=True)
    direct = data.loc[data.FIPS.isin(ids)].assign(GEOID=lambda d: d.FIPS)
    historical = data.loc[~data.FIPS.isin(ids)].merge(
        crosswalk, left_on="FIPS", right_on="source_geoid", how="inner")
    historical["value"] = historical.value * historical.weight
    historical["GEOID"] = historical.target_geoid
    allocated = pd.concat([direct, historical], ignore_index=True)
    categories = allocated.groupby(["GEOID", "land", "pollutant"]).value
    category_values = categories.sum(min_count=1)
    category_values[categories.count() < categories.size()] = np.nan
    grouped = category_values.groupby(level="GEOID")
    summary = grouped.agg(["mean", "median", "min", "max", "count", "size"])
    summary["stdev"] = grouped.std(ddof=0)
    sums = grouped.sum(min_count=1)
    sums[grouped.count() < grouped.size()] = np.nan
    if not np.allclose(sums.reindex(ids), county_totals.reindex(ids),
                       rtol=1e-12, atol=0.001, equal_nan=True):
        raise ValueError("Air-quality category summaries disagree with reconciled county totals")
    summary["sum"] = county_totals
    summary = summary.reindex(ids)
    for field in ("sum", "mean", "median", "min", "max", "stdev"):
        summary.loc[county_totals.isna(), field] = np.nan
    summary[["count", "size"]] = summary[["count", "size"]].fillna(0).astype("int64")
    summary = summary.rename(columns={"count": "valid_count", "size": "total_count"})
    fields = ["sum", "mean", "median", "min", "max", "stdev", "valid_count", "total_count"]
    return summary[fields].add_suffix("_dollars_annual")


def state_values(frame, field, county_states, weights):
    """Allocate each supplied state total; absent states remain unknown."""
    weights = weights.reindex(county_states.index)
    if weights.isna().any() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Allocation weights must cover every county and be finite/nonnegative")
    source = frame.copy()
    names = source.State.fillna("").str.strip().str.upper()
    source = source.loc[~names.isin(["", "TOTAL"])].copy()
    source["statefp"] = names.loc[source.index].map(STATE_NAMES)
    if source.statefp.isna().any() or source.statefp.duplicated().any():
        raise ValueError("Unrecognized or duplicate state in fisheries source")
    source["value"] = numbers(source[field])
    result = pd.Series(np.nan, index=county_states.index)
    status = pd.Series("missing_state_source", index=county_states.index, dtype=object)
    checks = []
    for row in source.itertuples():
        mask = county_states == row.statefp
        denominator = float(weights[mask].sum())
        if not mask.any() or denominator <= 0:
            raise ValueError(f"No allocation support for state {row.statefp}")
        if pd.isna(row.value):
            continue
        result[mask] = row.value * weights[mask] / denominator
        status[mask] = "state_weighted"
        status[mask & (weights == 0)] = "zero_weight"
        allocated = float(result[mask].sum())
        close(allocated, row.value, f"State {row.statefp}")
        checks.append({"source_key": row.statefp, "input_value": float(row.value), "allocated_value": allocated})
    return result, status, checks


def input_weights(path, ids, prefix):
    frame = read_attributes(path)[0] if path.suffix == ".gpkg" else pd.read_csv(path, dtype={"GEOID": str})
    frame["GEOID"] = frame.GEOID.map(fips)
    fields = [c for c in frame if c.startswith(prefix)]
    if len(fields) != 1 or frame.GEOID.duplicated().any() or set(frame.GEOID) != set(ids):
        raise ValueError(f"Unexpected county weight schema/coverage: {path}")
    return numbers(frame.set_index("GEOID")[fields[0]]).reindex(ids)


def marine_coastline_weights(counties, coastline, baseline):
    """Replace weights that include Great Lakes shoreline with marine-only lengths.

    Args:
        counties: County polygons with one row per five-digit GEOID.
        coastline: TIGER coastline lines with waterbody labels in NAME.
        baseline: Existing county coastline lengths in kilometres, indexed by GEOID.

    Returns:
        Kilometres per GEOID. Only counties with positive Great Lakes overlap
        are remeasured; all other supplied lengths are preserved.
    """
    ids = pd.Index(counties.GEOID, name="GEOID")
    is_lake = coastline["NAME"].astype("string").str.strip().str.casefold().eq("great lakes")
    weights = baseline.reindex(ids).copy()
    if not is_lake.any():
        return weights

    # Match the nationwide planar measurement used by the vector zonal job.
    lines = coastline[["geometry"]].assign(great_lakes=is_lake.fillna(False))
    intersections = gpd.overlay(
        lines.to_crs(AREA_CRS).explode(index_parts=False),
        counties[["GEOID", "geometry"]].to_crs(AREA_CRS),
        how="intersection",
        keep_geom_type=False,
    )
    lengths = intersections.length / 1000
    lake = intersections.great_lakes
    affected = intersections.loc[lake & lengths.gt(0), "GEOID"].unique()
    marine_lengths = lengths.loc[~lake].groupby(intersections.GEOID).sum()
    weights.loc[affected] = marine_lengths.reindex(affected, fill_value=0.0)
    return weights


def inland_freshwater_weights(counties, freshwater_path, lake_fids, baseline, *, great_salt_lake_fids=None):
    """Remeasure inland-fishery area outside configured lake footprints.

    Args:
        counties: County polygons with one row per five-digit GEOID.
        freshwater_path: Prepared NHD polygons with source_layer and source_fid.
        lake_fids: NHDWaterbody source identifiers for the Laurentian Great Lakes.
        baseline: Existing county freshwater hectares indexed by GEOID.
        great_salt_lake_fids: NHDWaterbody source identifiers excluded in Utah.

    Returns:
        Hectares per GEOID, with supplied weights preserved outside the lakes.

    Raises:
        ValueError: Exclusion identifiers are invalid or absent from the prepared layer.
    """
    weights = baseline.reindex(pd.Index(counties.GEOID, name="GEOID")).copy()
    great_lakes = list(lake_fids)
    great_salt_lake = [] if great_salt_lake_fids is None else list(great_salt_lake_fids)
    ids = pd.Index([*great_lakes, *great_salt_lake])
    if (not great_lakes or (great_salt_lake_fids is not None and not great_salt_lake)
            or ids.hasnans or not pd.api.types.is_integer_dtype(ids.dtype)
            or ids.has_duplicates or (ids < 0).any()):
        raise ValueError("Inland fisheries exclusion source FIDs must be distinct nonnegative integers")
    lake_ids = set(ids)
    fid_list = ", ".join(str(fid) for fid in sorted(lake_ids))
    with tqdm(total=3, desc="Correct inland freshwater weights", unit="step") as progress:
        progress.set_postfix_str("Load lake exclusion polygons")
        lakes = gpd.read_file(
            freshwater_path, columns=["source_fid"],
            where=f"source_layer = 'NHDWaterbody' AND source_fid IN ({fid_list})",
        )
        missing_ids = lake_ids - set(lakes.source_fid)
        if missing_ids:
            raise ValueError(f"Missing configured inland fisheries exclusions in {freshwater_path}: {sorted(missing_ids)}")
        lake_shapes = lakes.to_crs(AREA_CRS)
        great_lakes_footprint = lake_shapes.loc[lake_shapes.source_fid.isin(great_lakes)].geometry.union_all()
        progress.update(1)

        progress.set_postfix_str("Find counties intersecting excluded lakes")
        projected = counties[["GEOID", "geometry"]].to_crs(AREA_CRS)
        scopes = [(projected, great_lakes_footprint)]
        if great_salt_lake:
            utah = projected.GEOID.str.startswith("49")
            salt_lake_footprint = lake_shapes.loc[lake_shapes.source_fid.isin(great_salt_lake)].geometry.union_all()
            scopes = [(projected.loc[~utah], great_lakes_footprint), (projected.loc[utah], salt_lake_footprint)]
        affected_parts = []
        for scope, footprint in scopes:
            candidates = scope.iloc[scope.sindex.query(footprint)].copy()
            candidates.geometry = candidates.geometry.make_valid()
            affected = candidates.loc[candidates.geometry.intersection(footprint).area.gt(0)].copy()
            affected.geometry = affected.geometry.difference(footprint)
            affected_parts.append(affected)
        affected = pd.concat(affected_parts)
        progress.update(1)
        if affected.empty:
            progress.update(1)
            return weights

        progress.set_postfix_str("Measure remaining freshwater")
        # Clip the county zones, not the shared water layer. Read each affected
        # state separately so a single spatial filter does not span both the
        # Laurentian Great Lakes and Utah. Keep per-feature area sums.
        for _, group in affected.groupby(affected.GEOID.str[:2], sort=False):
            remaining = group.loc[~group.geometry.is_empty]
            weights.loc[group.GEOID] = 0.0
            if remaining.empty:
                continue
            mask = counties.loc[remaining.index].to_crs(lakes.crs).geometry.union_all()
            water = gpd.read_file(freshwater_path, columns=[], mask=mask)
            intersections = gpd.overlay(
                water.to_crs(AREA_CRS).explode(index_parts=False), remaining,
                how="intersection", keep_geom_type=False,
            )
            hectares = (intersections.area / 10_000).groupby(intersections.GEOID).sum()
            weights.loc[remaining.GEOID] = hectares.reindex(remaining.GEOID, fill_value=0.0)
        progress.update(1)
    return weights


def load_export_rules(path):
    """Load and validate the tracked YAML presentation policy.

    Args:
        path: Export-rules YAML file under the selected configuration directory.

    Returns:
        Validated rule dictionary; source/factor CSVs remain authoritative.

    Raises:
        ValueError: YAML has duplicate keys, unknown settings, or invalid values.
    """
    class UniqueKeysLoader(yaml.SafeLoader):
        """Reject duplicate YAML keys instead of silently discarding a rule."""

        def construct_mapping(self, node, deep=False):
            """Construct a mapping with unique keys.

            Args:
                node: YAML mapping node.
                deep: Whether to construct nested values immediately.

            Returns:
                Parsed mapping.

            Raises:
                ValueError: A key appears more than once.
            """
            keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
            if len(keys) != len(set(keys)):
                raise ValueError("Duplicate key in export rules")
            return super().construct_mapping(node, deep=deep)

    try:
        rules = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeysLoader)
    except (yaml.YAMLError, TypeError) as error:
        raise ValueError(f"Invalid export rules in {path}: {error}") from error
    expected = {"schema_version", "output", "column_groups", "exclude_fields",
                "optional_exclude_fields", "exclude_patterns", "rename_fields"}
    if not isinstance(rules, dict) or set(rules) != expected or rules["schema_version"] != 1:
        raise ValueError("Export rules need schema_version 1 and exactly the documented settings")
    for name in ("column_groups", "exclude_fields", "optional_exclude_fields", "exclude_patterns"):
        value = rules[name]
        if (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)
                or len(value) != len(set(value))):
            raise ValueError(f"Export {name} must be a list of unique nonempty strings")
    groups = rules["column_groups"]
    if (set(groups) != {"base_attributes", "sum_values", "service_totals", "adjustment_factors"}
            or groups[0] != "base_attributes"):
        raise ValueError("column_groups must list each documented group once, with base_attributes first")
    renames = rules["rename_fields"]
    if (not isinstance(renames, dict) or any(not isinstance(key, str) or not key
            or not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)
            for key, value in renames.items())):
        raise ValueError("rename_fields must map existing names to valid column identifiers")
    output = rules["output"]
    if (not isinstance(output, dict) or not {"directory", "filename_prefix"} <= set(output)
            or set(output) - {"directory", "filename_prefix", "layer_name"}):
        raise ValueError("output requires directory and filename_prefix; layer_name is optional")
    if (not isinstance(output["directory"], str) or not output["directory"]
            or Path(output["directory"]).is_absolute() or Path(output["directory"]).drive
            or Path(output["directory"]).root
            or ".." in Path(output["directory"]).parts):
        raise ValueError("output.directory must stay relative to the data root")
    if (not isinstance(output["filename_prefix"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", output["filename_prefix"])):
        raise ValueError("output.filename_prefix must be a simple filename prefix")
    if "layer_name" in output and (
            not isinstance(output["layer_name"], str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", output["layer_name"])
            or output["layer_name"].lower().startswith(("gpkg_", "sqlite_", "rtree_", "integration_"))):
        raise ValueError("output.layer_name must be a non-reserved simple layer identifier")
    return rules


def plan_county_export(base_columns, services, rules, geometry_field="geom"):
    """Resolve names, omissions, and ordering before computing service values.

    Args:
        base_columns: Original attributes, including county identifiers.
        services: Configured service keys from adjustment_factors.csv.
        rules: Validated export policy from load_export_rules.
        geometry_field: Protected geometry-column name.

    Returns:
        Ordered records describing every source/generated field and its action.

    Raises:
        ValueError: A rule is stale, conflicts, removes a county attribute, drops
            all values, or produces colliding names (case-insensitive, as in SQLite).
    """
    source_services = {BASE_FIELDS[s]: s for s in services if s in BASE_FIELDS}
    missing = set(source_services) - set(base_columns)
    if missing:
        raise ValueError(f"Missing source sums: {sorted(missing)}")
    records = []
    for source in base_columns:
        service = source_services.get(source, "")
        statistic = source.startswith("sum_") or NON_SUM_FIELD.match(source)
        records.append({"source_field": source, "default_field": source + "_adj" if service else source,
                        "service": service, "group": "sum_values" if source.startswith("sum_")
                        else "service_totals" if statistic else "base_attributes"})
    for service in services:
        if service not in BASE_FIELDS:
            records.append({"source_field": "", "default_field": service + "_adj",
                            "service": service,
                            "group": "sum_values" if service in {"carbon_pixel", "mangroves", "corals"} else "service_totals"})
        records.append({"source_field": "", "default_field": service + "_adj_factor",
                        "service": service, "group": "adjustment_factors"})
    defaults = [row["default_field"] for row in records]
    if len(defaults) != len({field.casefold() for field in defaults}):
        raise ValueError("Default county output names collide; use the raw base input")
    available = set(defaults) | set(base_columns)
    missing = set(rules["exclude_fields"]) - available
    if missing:
        raise ValueError(f"Unknown required exclusion fields: {sorted(missing)}")
    missing = set(rules["rename_fields"]) - set(defaults)
    if missing:
        raise ValueError(f"Unknown rename source fields: {sorted(missing)}")
    excludes = set(rules["exclude_fields"]) | set(rules["optional_exclude_fields"])
    for row in records:
        names = [row["default_field"], row["source_field"]]
        row["action"] = "exclude" if (excludes.intersection(names) or any(
            fnmatchcase(name, pattern) for name in names if name for pattern in rules["exclude_patterns"])) else "keep"
    excluded_services = {row["service"] for row in records if row["service"]
                         and row["group"] != "adjustment_factors" and row["action"] == "exclude"}
    seen = {geometry_field.casefold()}
    for row in records:
        default = row["default_field"]
        if row["group"] == "adjustment_factors" and row["service"] in excluded_services:
            row["action"] = "exclude"
        elif row["group"] == "adjustment_factors" and row["action"] == "exclude":
            raise ValueError(f"Keep the factor for an exported service total: {default}")
        target = rules["rename_fields"].get(default, default)
        if row["group"] == "base_attributes" and (row["action"] == "exclude" or target != default):
            raise ValueError(f"Export rules cannot remove or rename county attribute {default}")
        if row["action"] == "exclude":
            if default in rules["rename_fields"]:
                raise ValueError(f"Cannot both exclude and rename {default}")
            row["output_field"] = None
            continue
        if target.casefold() in seen:
            raise ValueError(f"Output field names collide: {target}")
        seen.add(target.casefold())
        row["output_field"] = target
        if target != row["source_field"]:
            row["action"] = "rename" if row["source_field"] else "create"
    if not any(row["action"] != "exclude" and row["group"] in {"sum_values", "service_totals"}
               for row in records):
        raise ValueError("Export rules must retain at least one service value")
    return sorted(records, key=lambda row: rules["column_groups"].index(row["group"]))


def adjusted_county_fields(base, originals, summaries, factors, export_plan):
    """Build ordered county sums, other adjusted totals, and visible factors.

    Configured raster sums receive an _adj suffix before YAML renaming. Retained
    unconfigured sums keep their values. The default rules omit non-sum statistics
    and coverage diagnostics; the writer preserves original attributes first.

    Args:
        base: Original county attributes indexed by GEOID.
        originals: Raw per-service county Series from the integration tasks.
        summaries: Raw air-quality statistics indexed by GEOID.
        factors: Reviewed adjustment configuration indexed by service.
        export_plan: Resolved records from plan_county_export.

    Returns:
        Ordered value/factor fields and a mapping from each service to its total
        field. No raw-value copies or duplicate service totals are created.

    Raises:
        ValueError: A field is assigned twice or air-quality totals disagree.
    """
    values_by_field = {}
    output_fields = {}
    for service, values in tqdm(originals.items(), desc="Apply value adjustments", unit="service"):
        factor = float(factors.loc[service, "factor"])
        values_by_field[f"{service}_adj_factor"] = factor
        if service in BASE_FIELDS:
            total_field = BASE_FIELDS[service]
        elif service == "avoided_air_quality_heath_costs":
            total_field = service
            if not np.allclose(summaries["sum_dollars_annual"].reindex(base.index), values.reindex(base.index),
                               rtol=1e-12, atol=0.001, equal_nan=True):
                raise ValueError("Air-quality canonical sum differs from the service total")
        else:
            total_field = service
        field = total_field + "_adj"
        if field in values_by_field:
            raise ValueError(f"Duplicate adjusted county field: {field}")
        values_by_field[field] = values.reindex(base.index).astype(float) * factor
    fields = pd.DataFrame(index=base.index)
    for row in export_plan:
        if row["action"] == "exclude" or row["group"] == "base_attributes":
            continue
        default, target = row["default_field"], row["output_field"]
        fields[target] = (values_by_field[default] if default in values_by_field
                          else numbers(base[row["source_field"]]))
        if row["service"] and row["group"] != "adjustment_factors":
            output_fields[row["service"]] = target
    return fields, output_fields


def write_output(base_path, output, layer, geom, additions, tables, removed_fields=(), *, output_layer=None):
    """Validate and publish a GeoPackage and matching geometry-free CSV.

    Args:
        base_path: Original county GeoPackage to preserve.
        output: New .gpkg path; its .csv sibling is written in the same run.
        layer: Original county feature layer name.
        geom: Original geometry column name, excluded only from the CSV.
        additions: County-indexed sums, other totals, then factors in export order.
        tables: Supporting provenance and reconciliation tables.
        removed_fields: Source statistics to remove before appending ordered values.
            May include sums that are reinserted, with adjusted names if configured.
        output_layer: Optional feature-layer name for the new GeoPackage only.
            Omission preserves the input name. Geometry and values are unchanged.

    Raises:
        FileExistsError: Either destination already exists.
        ValueError: The extension or preservation/export checks fail.
    """
    if output.suffix.lower() != ".gpkg":
        raise ValueError("--output must be a .gpkg path; its .csv sibling is automatic")
    csv_output = output.with_suffix(".csv")
    for destination in (output, csv_output):
        if destination.exists():
            raise FileExistsError(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_layer = layer
    with tempfile.TemporaryDirectory(prefix="county-integration-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        csv_stage = stage.with_suffix(".csv")
        with closing(read_db(base_path)) as original, closing(sqlite3.connect(stage)) as con:
            original.backup(con)
        if output_layer is not None and output_layer != source_layer:
            # Use the GeoPackage driver, not bare SQLite ALTER TABLE: it also
            # renames the spatial index, triggers, and GeoPackage registrations.
            # https://gdal.org/en/stable/drivers/vector/gpkg.html#sql
            with tqdm(total=1, desc="Rename output county layer", unit="layer") as progress:
                dataset = ogr.Open(str(stage), update=1)
                if dataset is None:
                    raise ValueError(f"Cannot open staged GeoPackage for layer rename: {stage}")
                try:
                    if dataset.GetLayerByName(output_layer) is not None:
                        raise ValueError(f"Output layer already exists: {output_layer}")
                    gdal.ErrorReset()
                    result = dataset.ExecuteSQL(
                        f"ALTER TABLE {quote(source_layer)} RENAME TO {quote(output_layer)}")
                    if result is not None:
                        dataset.ReleaseResultSet(result)
                    if gdal.GetLastErrorType() >= gdal.CE_Failure or dataset.GetLayerByName(output_layer) is None:
                        raise ValueError(f"Cannot rename county layer: {gdal.GetLastErrorMsg()}")
                    layer = output_layer
                finally:
                    dataset = None
                progress.update(1)
        with closing(read_db(base_path)) as original, closing(sqlite3.connect(stage)) as con:
            # GDAL's FID-update triggers reference these functions even when their
            # WHEN condition is false. Any actual invocation is a programming error:
            # this integration must never modify FIDs or geometries.
            def unexpected_geometry_edit(*args):
                raise RuntimeError("Unexpected geometry edit")
            for name in ("ST_IsEmpty", "ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
                con.create_function(name, 1, unexpected_geometry_edit)
            original_fields = [r[1] for r in con.execute(f"PRAGMA table_info({quote(layer)})")]
            removed = set(removed_fields)
            if (not removed <= set(original_fields)
                    or removed & {"GEOID", "fid", geom}
                    or any(not (field.startswith("sum_") or NON_SUM_FIELD.match(field)) for field in removed)
                    or set(additions.columns) & (set(original_fields) - removed)):
                raise ValueError("Only generated statistics may be removed or replaced")
            for field in tqdm([c for c in original_fields if c in removed],
                              desc="Remove old statistics", unit="field"):
                con.execute(f"ALTER TABLE {quote(layer)} DROP COLUMN {quote(field)}")
            for field in tqdm(additions.columns, desc="Prepare value fields", unit="field"):
                sqltype = "INTEGER" if pd.api.types.is_integer_dtype(additions[field]) else "REAL"
                con.execute(f"ALTER TABLE {quote(layer)} ADD COLUMN {quote(field)} {sqltype}")
            # Use the existing integer primary key, avoiding a full geometry-table
            # scan for each GEOID (GEOID is not indexed in the supplied base).
            fid_by_geoid = dict(con.execute(f"SELECT GEOID,fid FROM {quote(layer)}"))
            sql = f"UPDATE {quote(layer)} SET " + ",".join(f"{quote(c)}=?" for c in additions) + ' WHERE "fid"=?'
            rows = [tuple(None if pd.isna(v) else float(v) for v in values) + (fid_by_geoid[key],)
                    for key, values in zip(additions.index, additions.to_numpy())]
            for start in tqdm(range(0, len(rows), 250), desc="Write counties", unit="batch"):
                con.executemany(sql, rows[start:start + 250])
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            con.execute("UPDATE gpkg_contents SET last_change=? WHERE table_name=?", (now, layer))
            if output_layer is not None:
                con.execute("UPDATE gpkg_contents SET identifier=? WHERE table_name=?", (layer, layer))
            con.commit()
            for name, frame in tqdm(tables.items(), desc="Write audit tables", unit="table"):
                definitions = []
                for column in frame:
                    dtype = frame[column].dtype
                    sqltype = "INTEGER" if pd.api.types.is_integer_dtype(dtype) else ("REAL" if pd.api.types.is_numeric_dtype(dtype) else "TEXT")
                    definitions.append(f"{quote(column)} {sqltype}")
                con.execute(f"CREATE TABLE {quote(name)} (fid INTEGER PRIMARY KEY AUTOINCREMENT, {','.join(definitions)})")
                frame.to_sql(name, con, index=False, if_exists="append")
                con.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,description,last_change) VALUES (?, 'attributes', ?, ?, ?)",
                            (name, name, "October 2026 county valuation integration audit", now))
            con.commit()
            keep = [c for c in original_fields if c not in removed]
            query = f"SELECT {','.join(map(quote, keep))} FROM {quote(layer)} ORDER BY fid"
            source_query = f"SELECT {','.join(map(quote, keep))} FROM {quote(source_layer)} ORDER BY fid"
            before, after = original.execute(source_query), con.execute(query)
            with tqdm(total=len(additions), desc="Verify preserved counties", unit="county") as progress:
                while True:
                    old_rows, new_rows = before.fetchmany(100), after.fetchmany(100)
                    if old_rows != new_rows:
                        raise ValueError("Original geometry, IDs or retained attributes changed")
                    if not old_rows:
                        break
                    progress.update(len(old_rows))
            saved = pd.read_sql_query(f"SELECT GEOID,{','.join(map(quote, additions.columns))} FROM {quote(layer)}", con).set_index("GEOID")
            pd.testing.assert_frame_equal(saved.reindex(additions.index), additions, check_names=False, check_dtype=False)
            if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("GeoPackage integrity check failed")
            if con.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("GeoPackage foreign-key check failed")
            export_fields = [r[1] for r in con.execute(f"PRAGMA table_info({quote(layer)})") if r[1] != geom]
            if export_fields != [c for c in keep if c != geom] + list(additions.columns):
                raise ValueError("County field order differs from the planned export")
            attributes = pd.read_sql_query(
                f"SELECT {','.join(map(quote, export_fields))} FROM {quote(layer)} ORDER BY fid", con)
            with csv_stage.open("x", encoding="utf-8", newline="") as stream:
                for start in tqdm(range(0, len(attributes), 250), desc="Write county CSV", unit="batch"):
                    attributes.iloc[start:start + 250].to_csv(stream, index=False, header=start == 0)
            text_columns = {c: str for c in attributes if pd.api.types.is_string_dtype(attributes[c].dtype)
                            or pd.api.types.is_object_dtype(attributes[c].dtype)}
            csv_attributes = pd.read_csv(csv_stage, dtype=text_columns, float_precision="round_trip")
            pd.testing.assert_frame_equal(attributes, csv_attributes, check_dtype=False)
        # Publish only validated files, refuse overwrites, and roll back our own
        # first link if publication of the second file fails.
        published = []
        try:
            for destination, staged in ((output, stage), (csv_output, csv_stage)):
                destination.hardlink_to(staged)
                published.append(destination)
        except Exception:
            for destination in published:
                destination.unlink()
            raise


def compute_task(task, paths, base, crosswalk, geometry_workers):
    """Compute one independent service family in a worker process.

    Args:
        task: Service family to compute.
        paths: Source keys mapped to resolved local input paths.
        base: Raw county attributes indexed by GEOID.
        crosswalk: Historical Connecticut county allocation weights.
        geometry_workers: Threads for mangrove intersections.

    Returns:
        Unadjusted county values, statuses, source metadata and allocation audits.
    """
    ids = base.index
    mangrove_audit = pd.DataFrame()
    coral_audit = pd.DataFrame()
    summary_fields = pd.DataFrame(index=ids)
    originals, statuses, metadata, issues, checks = {}, [], [], [], []

    def add(service, values, status, source, source_field, method, audit=None):
        """Record a service's county values and reconciliation metadata.

        Args:
            service: Adjustment-factor service key.
            values: Unadjusted county estimates indexed by GEOID.
            status: County coverage or allocation status.
            source: Input key in sources.csv.
            source_field: Original valuation column.
            method: Description of the integration method.
            audit: Optional additional reconciliation metrics.
        """
        values = values.reindex(ids).astype(float)
        if np.isinf(values).any():
            raise ValueError(f"Non-finite aggregate: {service}")
        originals[service] = values
        statuses.append(pd.DataFrame({"service": service, "GEOID": ids, "status": status.reindex(ids).to_numpy()}))
        row = {"service": service, "source": source, "source_field": source_field, "method": method,
               "original_total": float(values.sum()),
               "populated_counties": int(values.notna().sum()), "missing_counties": int(values.isna().sum()),
               "negative_counties": int((values < 0).sum())}
        row.update(audit or {})
        metadata.append(row)
        logging.info("%s: %s populated counties; original total %.3f", service, row["populated_counties"], row["original_total"])

    def county_add(service, frame, key, field, source, *, use_crosswalk=True, method_note=None, **options):
        """Join tabular county values with service-specific source policies.

        Args:
            service: Adjustment-factor service key.
            frame: Source records; never modified in place.
            key: Source county identifier column.
            field: Source valuation column.
            source: Input key in sources.csv.
            use_crosswalk: Allocate historical Connecticut rows when present.
            method_note: Optional service-specific detail to record in provenance.
            **options: Aggregation and missing-value options for county_values.
        """
        method = "FIPS join; historical CT allocated by land area" if use_crosswalk else "direct GEOID join"
        if method_note:
            method += f"; {method_note}"
        if service == "timber":
            # Clamp individual source county rents BEFORE the CT crosswalk:
            # negative source rents must not cancel another county's positive rent.
            raw_rent = numbers(frame[field])
            negative = raw_rent < 0
            frame = frame.copy()
            frame[field] = raw_rent.clip(lower=0)
            negative_ids = set(frame.loc[negative, key].map(fips))
            issues.extend({"service": service, "source_key": fips(county_id),
                           "reason": "negative_source_clamped_zero", "value": float(value)}
                          for county_id, value in zip(frame.loc[negative, key], raw_rent[negative]))
            method += "; negative source rents clamped to zero before allocation"
        values, status, problems, audit = county_values(
            frame, key, field, ids, crosswalk if use_crosswalk else crosswalk.iloc[:0], **options,
        )
        if service == "timber":
            audit.update({"raw_input_known_total": float(raw_rent.sum()),
                          "negative_source_count": int(negative.sum()),
                          "negative_source_total": float(raw_rent[negative].sum()),
                          "clamp_increase_total": float(-raw_rent[negative].sum())})
            close(audit["input_known_total"], audit["raw_input_known_total"] + audit["clamp_increase_total"],
                  "Timber source clamp")
            timber_status = frame.assign(_key=frame[key].map(fips)).set_index("_key").coverage_status
            for county_id in ids.intersection(timber_status.index):
                if county_id in negative_ids:
                    status[county_id] = "negative_source_clamped_zero"
                elif timber_status[county_id] != "valued":
                    status[county_id] = str(timber_status[county_id]) + "_zero"
        issues.extend(dict(service=service, **p) for p in problems)
        add(service, values, status, source, field, method, audit)

    if task == "base":
        for service, values in raster_service_sums(base).items():
            status = pd.Series(np.where(values.isna(), "missing_source_value", "existing_base"), index=ids)
            if service == "carbon_pixel":
                add(service, values, status, "base", BASE_FIELDS["carbon_ha"],
                    f"Density sum multiplied by {CARBON_PIXEL_AREA_HA} hectares per 30 m pixel")
            else:
                add(service, values, status, "base", BASE_FIELDS[service], "existing county aggregate preserved")
    if task == "grazing":
        grazing = pd.read_excel(paths["grazing"], sheet_name="SNAPPGrazingCalculations", engine="openpyxl")
        # Annual usable forage / 780 lb per animal-unit month already gives the
        # annual animal-unit months. Price them once using the workbook subtotal.
        county_add(
            "grazing", grazing, "CountyCode", "CountyGrazingValueMonth", "grazing",
            method_note="annual forage priced per animal-unit month; workbook's final eight-month multiplier excluded",
        )
    if task == "timber":
        timber = pd.read_csv(paths["timber"], dtype={"fips_padded": str})
        county_add("timber", timber, "fips_padded", "total_annual_rent", "timber", zero_missing=True)
    if task == "air_quality":
        air = pd.read_csv(paths["air_quality"], dtype={"FIPS": str})
        aliases = pd.read_csv(paths["air_quality_county_aliases"], dtype=str)
        if aliases.source_geoid.duplicated().any():
            raise ValueError("Duplicate source GEOIDs in air-quality county aliases")
        original_codes = air.FIPS.map(fips)
        air["FIPS"] = original_codes.replace(
            aliases.set_index("source_geoid").target_geoid)
        alias_audit = []
        for alias in aliases.itertuples(index=False):
            selected = original_codes.eq(alias.source_geoid)
            if selected.any():
                alias_audit.append(dict(
                    alias._asdict(), source_rows=int(selected.sum()),
                    raw_known_total=float(numbers(air.loc[selected, "dollars_annual"]).sum())))
        county_add("avoided_air_quality_heath_costs", air, "FIPS", "dollars_annual", "air_quality", aggregate=True)
        metadata[-1]["county_code_aliases"] = json.dumps(alias_audit, sort_keys=True)
        # The existing category check also rejects collisions caused by an alias.
        summary_fields = air_quality_summary(
            air, ids, crosswalk, originals["avoided_air_quality_heath_costs"])
    if task == "urban_heat":
        heat = pd.read_csv(paths["urban_heat"], dtype={"COUNTY_GEOID": str}).dropna(how="all")
        county_add("urban_heat", heat, "COUNTY_GEOID", "Sum of total_dollars_saved", "urban_heat", aggregate=True)
    if task == "physical_health":
        physical = read_attributes(paths["physical_health"])[0]
        county_add("physical_health", physical, "GEOID", "total_value_usd", "physical_health")
    if task == "jobs":
        jobs = pd.read_excel(paths["jobs"], sheet_name="totals by county", header=1, dtype={"County FIPS": str}, engine="openpyxl")
        job_keys = jobs["County FIPS"].fillna("").str.strip()
        keep = job_keys.str.fullmatch(r"\d{5}") & ~job_keys.str.endswith(("000", "999"))
        for key in job_keys[~keep]:
            issues.append({"service": "jobs", "source_key": key, "reason": "excluded_noncounty_summary", "value": None})
        jobs = jobs.loc[keep].copy()
        for service, field in JOB_FIELDS.items():
            county_add(service, jobs, "County FIPS", field, "jobs")
        job_values = pd.DataFrame({s: originals[s] for s in JOB_FIELDS}).sum(axis=1, min_count=4)
        add("jobs", job_values, pd.Series(np.where(job_values.isna(), "missing_source", "sum_four_categories"), index=ids),
            "jobs", "; ".join(JOB_FIELDS.values()), "sum four raw wage categories")
    if task in {"marine_fisheries", "inland_fisheries"}:
        if task == "marine_fisheries":
            logging.info("Correct county coastline weights for Great Lakes overlap")
            counties = gpd.read_file(paths["base"], columns=["GEOID"])
            counties.GEOID = counties.GEOID.map(fips)
            coastline = gpd.read_file(paths["coastline_lines"], columns=["NAME"])
            baseline = input_weights(paths["coastline"], ids, "intersect_length_km_")
            weights = marine_coastline_weights(counties, coastline, baseline)
            field = "ncp_value"
            method = ("State total allocated by county coastline share; counties intersecting Great Lakes "
                      "shoreline remeasured using marine lines only in EPSG:6933")
        else:
            counties = gpd.read_file(paths["base"], columns=["GEOID"])
            counties.GEOID = counties.GEOID.map(fips)
            baseline = input_weights(paths["freshwater"], ids, "intersect_area_ha_")
            lake_fids = {}
            for key in ("great_lakes_features", "inland_fisheries_exclusions"):
                selected = pd.read_csv(paths[key], dtype={"source_fid": "int64"})["source_fid"]
                if selected.empty:
                    raise ValueError(f"Inland fisheries exclusion table is empty: {paths[key]}")
                lake_fids[key] = selected
            weights = inland_freshwater_weights(
                counties, paths["freshwater_polygons"], lake_fids["great_lakes_features"], baseline,
                great_salt_lake_fids=lake_fids["inland_fisheries_exclusions"],
            )
            field = "Consumptive use value (USD)_Intermediate assumption"
            method = ("State total allocated by county freshwater share; affected counties measured "
                      "outside the five Laurentian Great Lakes and Great Salt Lake in EPSG:6933")
        frame = pd.read_csv(paths[task])
        values, status, state_checks = state_values(frame, field, base.STATEFP, weights)
        checks.extend(dict(service=task, **c) for c in state_checks)
        add(task, values, status, task, field, method)
    if task == "corals":
        with tqdm(total=2, desc="Load coral allocation inputs", unit="layer") as progress:
            counties = gpd.read_file(paths["base"], columns=["GEOID"])
            counties.GEOID = counties.GEOID.map(fips)
            progress.update(1)
            points = gpd.read_file(paths["corals"], layer="aeb_coral", columns=["OBJECTID", "BC_USD11"])
            progress.update(1)
        values, status, coral_audit = allocate_coral_values(counties, points)
        input_total = float(coral_audit.input_value.sum())
        withheld = float(coral_audit.loc[coral_audit.GEOID.isin(values.index[values.isna()]), "input_value"].sum())
        close(float(values.sum()) + withheld, input_total, "Coral source allocation")
        add("corals", values, status, "corals", "BC_USD11",
            "Entire point total assigned once to nearest county polygon in point-centered WGS84 AEQD metres; "
            "ties within 1e-6 m choose lowest GEOID; null contributions make county value unknown",
            {"input_known_total": input_total, "allocated_known_total": input_total,
             "withheld_partial_total": withheld, "unmatched_total": 0.0,
             "source_point_count": len(points), "assigned_point_count": len(coral_audit),
             "missing_source_values": int(coral_audit.input_value.isna().sum()),
             "tied_source_points": int((coral_audit.tie_count > 1).sum())})
        issues.extend({"service": "corals", "source_key": row.source_key,
                       "reason": "missing_source_value", "value": None}
                      for row in coral_audit.itertuples() if pd.isna(row.input_value))
    if task == "mangroves":
        with tqdm(total=2, desc="Load mangrove allocation inputs", unit="layer") as progress:
            counties = gpd.read_file(paths["base"], columns=["GEOID"])
            counties.GEOID = counties.GEOID.map(fips)
            progress.update(1)
            polygons = gpd.read_file(
                paths["mangroves"], layer="UCSC_CWON_studyunits", engine="pyogrio",
                columns=["ISO3", "Country", "Ben_Stock_2020"], fid_as_index=True,
            ).rename_axis("source_id").reset_index()
            progress.update(1)
        source_unit_count = len(polygons)
        polygons = polygons.loc[polygons.ISO3.isin(MANGROVE_COUNTRIES)].copy()
        logging.info("Mangrove country filter %s: retained %s of %s study units",
                     ",".join(MANGROVE_COUNTRIES), len(polygons), source_unit_count)
        values, status, mangrove_audit = allocate_mangrove_values(
            counties, polygons, workers=geometry_workers)
        known_allocated = float(mangrove_audit.allocated_value.sum())
        unallocated = float(mangrove_audit.unallocated_value.sum())
        input_total = float(mangrove_audit.input_value.sum())
        withheld = known_allocated - float(values.sum())
        close(known_allocated + unallocated, input_total, "Mangrove source allocation")
        add("mangroves", values, status, "mangroves", "Ben_Stock_2020",
            "Study-unit total allocated by EPSG:6933 county intersection area / sum of positive county intersection areas; "
            "uncovered ocean excluded from denominator; null contributions make county value unknown",
            {"input_known_total": input_total, "allocated_known_total": known_allocated,
             "unmatched_total": unallocated, "withheld_partial_total": withheld,
             "source_unit_count": len(polygons), "missing_source_values": int(mangrove_audit.input_value.isna().sum()),
             "source_country_filter": ",".join(MANGROVE_COUNTRIES),
             "excluded_country_source_units": source_unit_count - len(polygons),
             "unintersected_source_units": int((mangrove_audit.intersecting_counties == 0).sum())})
        for row in mangrove_audit.itertuples():
            if pd.isna(row.input_value) or row.intersecting_counties == 0:
                issues.append({"service": "mangroves", "source_key": str(row.source_key),
                               "reason": "missing_source_value" if pd.isna(row.input_value) else "no_county_intersection",
                               "value": None if pd.isna(row.input_value) else float(row.input_value)})
    if task == "recreation":
        logging.info("Join county recreation values by GEOID")
        recreation = pd.read_csv(
            paths["recreation"], usecols=["GEOID", "county_cs_2024usd"], dtype={"GEOID": str},
        )
        county_add("recreation", recreation, "GEOID", "county_cs_2024usd", "recreation", use_crosswalk=False)
    return {"originals": originals, "statuses": statuses, "metadata": metadata,
            "issues": issues, "checks": checks,
            "mangrove_audit": mangrove_audit, "coral_audit": coral_audit,
            "summary_fields": summary_fields}


def save_checkpoint(path, signature, result):
    """Publish a complete SQLite checkpoint atomically; never load Python pickle.

    Args:
        path: Destination checkpoint file.
        signature: Hash identifying code, inputs and allocation dependencies.
        result: Completed service result, including any spatial allocation audit.
    """
    with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=path.parent) as temporary:
        stage = Path(temporary) / path.name
        with closing(sqlite3.connect(stage)) as con:
            pd.DataFrame(result["originals"]).to_sql("valuations", con, index_label="GEOID")
            if len(result.get("summary_fields", pd.DataFrame()).columns):
                result["summary_fields"].to_sql("summary_fields", con, index_label="GEOID")
            pd.concat(result["statuses"], ignore_index=True).to_sql("statuses", con, index=False)
            if len(result.get("mangrove_audit", pd.DataFrame()).columns):
                result["mangrove_audit"].to_sql("mangroves", con, index=False)
            if len(result.get("coral_audit", pd.DataFrame()).columns):
                result["coral_audit"].to_sql("corals", con, index=False)
            payload = {k: result[k] for k in ("metadata", "issues", "checks")}
            con.execute("CREATE TABLE completed (signature TEXT, payload TEXT)")
            con.execute("INSERT INTO completed VALUES (?, ?)", (signature, json.dumps(payload, allow_nan=False)))
            con.commit()
        stage.replace(path)


def load_checkpoint(path, signature):
    """Restore a verified service result and its allocation audits.

    Args:
        path: Existing SQLite checkpoint file.
        signature: Required code/input signature.

    Returns:
        Completed service result in the same format as compute_task.

    Raises:
        ValueError: SQLite integrity or the stored signature is invalid.
    """
    with closing(read_db(path)) as con:
        if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"Invalid checkpoint: {path}; rerun with --no-resume")
        saved_signature, payload = con.execute("SELECT signature,payload FROM completed").fetchone()
        if saved_signature != signature:
            raise ValueError(f"Checkpoint signature mismatch: {path}")
        result = json.loads(payload)
        values = pd.read_sql_query("SELECT * FROM valuations", con).set_index("GEOID")
        result["originals"] = {c: values[c] for c in values}
        result["statuses"] = [pd.read_sql_query("SELECT * FROM statuses", con)]
        has_mangroves = con.execute("SELECT 1 FROM sqlite_master WHERE name='mangroves'").fetchone()
        result["mangrove_audit"] = pd.read_sql_query("SELECT * FROM mangroves", con) if has_mangroves else pd.DataFrame()
        has_corals = con.execute("SELECT 1 FROM sqlite_master WHERE name='corals'").fetchone()
        result["coral_audit"] = pd.read_sql_query("SELECT * FROM corals", con) if has_corals else pd.DataFrame()
        has_summary = con.execute("SELECT 1 FROM sqlite_master WHERE name='summary_fields'").fetchone()
        result["summary_fields"] = (pd.read_sql_query("SELECT * FROM summary_fields", con).set_index("GEOID")
                                    if has_summary else pd.DataFrame(index=values.index))
    return result


def run(data_root, config_dir, output=None, workers=8, geometry_workers=8, resume=True):
    """Build and verify paired county exports using reviewed CSV/YAML configs.

    Args:
        data_root: Root of the source and analysis data stack.
        config_dir: Source/factor/crosswalk CSVs, export_rules.yaml and derived_metrics.yaml.
        output: Optional new GeoPackage path; CSV uses the same basename.
        workers: Independent ecosystem-service worker processes.
        geometry_workers: Intersection threads for the mangrove task.
        resume: Reuse matching validated checkpoints when available.

    Returns:
        Completed GeoPackage path, with a matching CSV alongside it.

    Raises:
        ValueError: Configuration, required raw pollination input, or QA fails.
        FileExistsError: Either requested output file already exists.
    """
    if workers < 1 or geometry_workers < 1:
        raise ValueError("Worker counts must be positive")
    if output is not None:
        if output.suffix.lower() != ".gpkg":
            raise ValueError("--output must have a .gpkg extension")
        for destination in (output, output.with_suffix(".csv")):
            if destination.exists():
                raise FileExistsError(destination)
    script_hash = sha256(__file__)
    allocator_path = Path(__file__).with_name("mangrove_county_allocation.py")
    allocator_hash = sha256(allocator_path)
    coral_allocator_path = Path(__file__).with_name("coral_county_allocation.py")
    coral_allocator_hash = sha256(coral_allocator_path)
    geometry_utils_path = Path(__file__).with_name("geometry_utils.py")
    geometry_utils_hash = sha256(geometry_utils_path)
    config_names = ("sources.csv", "adjustment_factors.csv", "ct_county_crosswalk.csv",
                    "export_rules.yaml", "derived_metrics.yaml")
    config_paths = {name: config_dir / name for name in config_names}
    derived_rules = yaml.safe_load((config_dir / "derived_metrics.yaml").read_text(encoding="utf-8"))
    gdp_crosswalk = derived_rules["sources"]["gdp"].get("allocation_crosswalk")
    if gdp_crosswalk:
        config_paths["gdp_allocation_crosswalk"] = (data_root / gdp_crosswalk).resolve()
    config_hashes = {name: sha256(path) for name, path in config_paths.items()}
    rules = load_export_rules(config_dir / "export_rules.yaml")
    rules_text = (config_dir / "export_rules.yaml").read_text(encoding="utf-8")
    sources = pd.read_csv(config_dir / "sources.csv")
    factors = pd.read_csv(config_dir / "adjustment_factors.csv").set_index("service")
    if sources.source.duplicated().any() or factors.index.duplicated().any():
        raise ValueError("Duplicate source or adjustment-factor definitions")
    if factors.factor.isna().any() or not np.isfinite(factors.factor).all() or (factors.factor <= 0).any():
        raise ValueError("Every service needs a finite positive adjustment factor")
    paths = {r.source: (data_root / r.path).resolve() for r in sources.itertuples()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with closing(read_db(paths["base"])) as con:
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='integration_run'").fetchone():
            raise ValueError("The base is already an integrated output; use raw county zonals to avoid applying factors twice")
    fingerprints = {name: sha256(path) for name, path in tqdm(paths.items(), desc="Hash sources", unit="file")}
    if output is not None and output.exists():
        raise FileExistsError(output)
    base, layer, geom = read_attributes(paths["base"])
    base_columns = list(base.columns)
    base["GEOID"] = base.GEOID.map(fips)
    if base.GEOID.duplicated().any():
        raise ValueError("Duplicate base county FIPS")
    base = base.set_index("GEOID")
    ids = base.index
    if BASE_FIELDS["pollination"] not in base:
        raise ValueError(
            "The county base must contain the unscaled "
            "sum_pollination_attributed_annual_crop_yield_value field. Use the "
            "raw county zonals configured in sources.csv; a per-pixel VRT aggregate "
            "must not receive the pollination factor 0.09 a second time."
        )
    if not (base.STATEFP == ids.str[:2]).all():
        raise ValueError("Base state and county identifiers disagree")
    export_plan = plan_county_export(base_columns, factors.index, rules, geometry_field=geom)
    derived_rules = yaml.safe_load((config_dir / "derived_metrics.yaml").read_text(encoding="utf-8"))
    missing_components = set(derived_rules["es_total"]["components"]) - {
        row["output_field"] for row in export_plan if row["output_field"]}
    if missing_components:
        raise ValueError(f"Export cannot supply required es_total components: {sorted(missing_components)}. "
                         "Check the county base, export_rules.yaml and derived_metrics.yaml.")
    for spec in derived_rules["sources"].values():
        path = (data_root / spec["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required derived-metric source is missing: {path}")
    crosswalk = pd.read_csv(config_dir / "ct_county_crosswalk.csv", dtype={"source_geoid": str, "target_geoid": str})
    validate_crosswalk(crosswalk, ids)
    cache_root = data_root / "processing_outputs/county_integration_2026"
    cache_root.mkdir(parents=True, exist_ok=True)
    dependencies = {
        "base": ["base"], "grazing": ["base", "grazing"], "timber": ["base", "timber"],
        "air_quality": ["base", "air_quality", "air_quality_county_aliases"], "urban_heat": ["base", "urban_heat"],
        "physical_health": ["base", "physical_health"], "jobs": ["base", "jobs"],
        "marine_fisheries": ["base", "marine_fisheries", "coastline", "coastline_lines"],
        "inland_fisheries": ["base", "inland_fisheries", "freshwater", "freshwater_polygons",
                            "great_lakes_features", "inland_fisheries_exclusions"],
        "recreation": ["base", "recreation"],
        "mangroves": ["base", "mangroves"],
        "corals": ["base", "corals"],
    }
    crosswalk_hash = config_hashes["ct_county_crosswalk.csv"]
    results = {}
    signatures = {}
    pending = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, names in dependencies.items():
            signature = hashlib.sha256(json.dumps({"script": script_hash, "crosswalk": crosswalk_hash,
                "mangrove_allocator": allocator_hash if task == "mangroves" else None,
                "coral_allocator": coral_allocator_hash if task == "corals" else None,
                "geometry_utils": geometry_utils_hash if task in {"mangroves", "corals"} else None,
                "versions": [pd.__version__, np.__version__, shapely.__version__, gpd.__version__],
                "inputs": {n: fingerprints[n] for n in names}}, sort_keys=True).encode()).hexdigest()
            signatures[task] = signature
            cache = cache_root / f"{task}_{signature}.sqlite"
            if resume and cache.exists():
                results[task] = load_checkpoint(cache, signature)
                logging.info("Resumed completed service: %s", task)
            else:
                future = pool.submit(compute_task, task, paths, base, crosswalk, geometry_workers)
                pending[future] = (task, signature, cache)
        failures = []
        for future in tqdm(as_completed(pending), total=len(dependencies), initial=len(results), desc="Ecosystem service groups", unit="group"):
            task, signature, cache = pending[future]
            try:
                result = future.result()
                save_checkpoint(cache, signature, result)
                results[task] = result
            except Exception as error:
                logging.exception("Service failed: %s; other services will finish and checkpoint", task)
                failures.append(f"{task}: {error}")
        if failures:
            raise RuntimeError("Integration incomplete; rerun to resume. " + "; ".join(failures))
    originals, statuses, metadata, issues, checks = {}, [], [], [], []
    for task in dependencies:
        result = results[task]
        originals.update(result["originals"])
        statuses.extend(result["statuses"])
        metadata.extend(result["metadata"])
        issues.extend(result["issues"])
        checks.extend(result["checks"])
    if set(originals) != set(factors.index):
        raise ValueError("Adjustment factor configuration does not exactly match integrated services")
    summary_frames = []
    for result in results.values():
        summary_fields = result.get("summary_fields", pd.DataFrame())
        if len(summary_fields.columns):
            summary_frames.append(summary_fields)
    summaries = pd.concat(summary_frames, axis=1) if summary_frames else pd.DataFrame(index=ids)
    additions, output_fields = adjusted_county_fields(base, originals, summaries, factors, export_plan)
    derived, derived_tables = calculate_county_metrics(
        additions, data_root, config_dir / "derived_metrics.yaml")
    additions = pd.concat([additions, derived], axis=1)
    renamed_fields = {row["source_field"]: row["output_field"] for row in export_plan
                      if row["source_field"] and row["action"] == "rename"}
    source_statistics = [row["source_field"] for row in export_plan
                         if row["source_field"] and row["group"] != "base_attributes"]
    # Source hashes also make interrupted/rerun inputs identifiable.
    for name, path in tqdm(paths.items(), desc="Verify source hashes", unit="file"):
        if sha256(path) != fingerprints[name]:
            raise ValueError(f"Source changed during integration: {path}")
    if (sha256(__file__) != script_hash or sha256(allocator_path) != allocator_hash
            or sha256(coral_allocator_path) != coral_allocator_hash
            or sha256(geometry_utils_path) != geometry_utils_hash
            or any(sha256(config_paths[name]) != digest for name, digest in config_hashes.items())):
        raise ValueError("Script or configuration changed during integration; rerun")
    source_audit = sources.copy()
    source_audit["sha256"] = source_audit.source.map(fingerprints)
    source_audit["bytes"] = source_audit.source.map({n: p.stat().st_size for n, p in paths.items()})
    source_audit = pd.concat([source_audit, pd.DataFrame([
        {"source": name, "path": str(path), "sha256": config_hashes[name], "bytes": path.stat().st_size}
        for name, path in config_paths.items()
    ])], ignore_index=True)
    meta = pd.DataFrame(metadata)
    meta["output_field"] = meta.service.map(output_fields)
    meta["factor_field"] = meta.service.map({row["service"]: row["output_field"] for row in export_plan
                                            if row["group"] == "adjustment_factors"})
    for field in ("factor", "reason", "source_units"):
        meta["factor_reason" if field == "reason" else field] = meta.service.map(factors[field])
    meta["adjusted_total"] = meta.original_total * meta.factor
    tables = {
        "integration_services": meta,
        "integration_sources": source_audit,
        "integration_county_status": pd.concat(statuses, ignore_index=True),
        "integration_source_issues": pd.DataFrame(issues, columns=["service", "source_key", "reason", "value"]),
        "integration_state_checks": pd.DataFrame(checks),
        "integration_mangrove_allocation": results["mangroves"]["mangrove_audit"],
        "integration_coral_allocation": results["corals"]["coral_audit"],
        "integration_ct_crosswalk": crosswalk,
        "integration_checkpoints": pd.DataFrame({"task": list(signatures), "signature": list(signatures.values())}),
        "integration_export_rules": pd.DataFrame([{
            "path": "export_rules.yaml", "sha256": config_hashes["export_rules.yaml"], "yaml_text": rules_text,
        }]),
        "integration_export_fields": pd.DataFrame(export_plan),
        "integration_summary_fields": pd.DataFrame([
            {"field": output_fields["avoided_air_quality_heath_costs"], "source": "air_quality", "source_field": "dollars_annual",
             "units": "USD annual; configured air-quality factor applied",
             "method": "County sum across land/pollutant categories after CT crosswalk; null if incomplete"}
        ] if "avoided_air_quality_heath_costs" in output_fields else [],
            columns=["field", "source", "source_field", "units", "method"]),
        "integration_run": pd.DataFrame([{
            "created_utc": datetime.now(timezone.utc).isoformat(), "script": Path(__file__).name,
            "script_sha256": sha256(__file__), "area_crs": AREA_CRS,
            "mangrove_allocator_sha256": allocator_hash,
            "coral_allocator_sha256": coral_allocator_hash,
            "geometry_utils_sha256": geometry_utils_hash,
            "base_counties": len(ids),
            "source_layer": layer, "output_layer": rules["output"].get("layer_name", layer),
            "removed_fields": json.dumps([field for field in source_statistics
                                           if field not in additions and field not in renamed_fields]),
            "renamed_value_fields": json.dumps(renamed_fields, sort_keys=True),
            "notes": "Carbon factor is supplied USD 3.66 per tonne CO2e times 1.182 inflation; no mass conversion. "
                     "Per-hectare carbon sum is not an area-integrated county total. "
                     "es_total uses only the embedded derived_metrics.yaml component list with mixed time bases. "
                     "Mangrove ISO3 limited to USA,PRI,VIR; Ben_Stock_2020 treated as study-unit USD 2020 total; "
                     "allocated over county intersections only, excluding uncovered ocean. "
                     "Negative timber source rents clamped to zero before county allocation. "
                     "Coral BC_USD11 point totals assigned once to nearest county polygons in point-centered WGS84 AEQD metres. "
                     "Configured sums/totals adjusted once; one visible factor per service; no duplicate raw values. "
                     "Export names, exclusions, order, and default output naming follow the embedded export_rules.yaml. "
                     "Geometry and geographic attributes preserved. "
                     "CSV and GeoPackage county attributes verified equal; CSV excludes geometry only.",
        }]),
    }
    tables.update(derived_tables)
    if output is None:
        output = data_root / rules["output"]["directory"] / (
            f"{rules['output']['filename_prefix']}_{datetime.now():%Y_%m_%d_%H_%M_%S}.gpkg")
    logging.info("Writing %s", output)
    write_output(paths["base"], output, layer, geom, additions, tables,
                 removed_fields=source_statistics, output_layer=rules["output"].get("layer_name"))
    logging.info("Verified %s counties and %s value/factor fields; geometry and county attributes preserved", len(ids), len(additions.columns))
    return output


def main():
    """Build matching timestamped county CSV/GeoPackage exports from CLI options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--config-dir", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, help="New .gpkg path; matching .csv is automatic; neither may already exist")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Independent service processes (default: up to 8)")
    parser.add_argument("--geometry-workers", type=int, default=min(8, os.cpu_count() or 1), help="Mangrove intersection threads (default: up to 8)")
    parser.add_argument("--no-resume", action="store_true", help="Recompute services instead of reusing checkpoints")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = run(args.data_root.resolve(), args.config_dir.resolve(), args.output.resolve() if args.output else None,
                 args.workers, args.geometry_workers, not args.no_resume)
    print(output)
    print(output.with_suffix(".csv"))


if __name__ == "__main__":
    main()
