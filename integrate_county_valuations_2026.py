"""Reproduce the October 2026 SNAPP county valuation integration (issue #58).

Run from any directory; defaults are relative to this script. Source files are
never modified. See docs/county_valuation_integration_2026.md for assumptions.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "data/workflow_assets/county_integration"
AREA_CRS = "EPSG:6933"  # Equal area, including Alaska, Hawaii and US territories.
DROP_FIELDS = (
    "sum_national_attributed_annual_crop_yield_value_zstd",
    "sum_pollination_attributed_annual_crop_yield_value",
)
BASE_FIELDS = {
    "carbon_ha": "sum_totalC_tCO2e_ha_2020",
    "carbon_pixel": "sum_totalC_tCO2e_pixel_2020",
    "dredging": "sum_avoided_dredging_costs_raster",
    "sdwa": "sum_avoided_sdwa_health_costs_raster",
    "water_treatment": "sum_avoided_treatment_costs_raster",
    "provisioning": "sum_Wval_sw_2020usd_total",
    "provisioning_irr": "sum_Wval_sw_2020usd_irr",
    "provisioning_pow": "sum_Wval_sw_2020usd_pow",
    "provisioning_pub": "sum_Wval_sw_2020usd_pub",
    "flood_npv": "sum_marginal_npv_masked_to_wetlands",
    "flood_annual": "sum_annual_value_masked_to_wetlands",
    "corals": "sum_Coral_Reefs_2024adj_CPI",
    "mangroves": "sum_mangrove_CONUS",
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


def polygon_values(counties, polygons, value_field, id_field, chunk_size=2000,
                   workers=1, checkpoint_dir=None):
    """Area-weight source polygons without renormalizing away uncovered area."""
    if counties.crs is None or polygons.crs is None:
        raise ValueError("Polygon allocation requires defined coordinate systems")
    if polygons[id_field].isna().any() or polygons[id_field].duplicated().any():
        raise ValueError("Recreation source IDs must be non-null and unique")
    values = numbers(polygons[value_field]).to_numpy()
    if np.isnan(values).any():
        raise ValueError("Missing recreation polygon value")
    county_geom = shapely.make_valid(counties.to_crs(AREA_CRS).geometry.to_numpy())
    source_geom = shapely.make_valid(polygons.to_crs(AREA_CRS).geometry.to_numpy())
    areas = shapely.area(source_geom)
    if (areas <= 0).any() or not np.isfinite(areas).all():
        raise ValueError("Empty or invalid recreation polygon area")
    tree = shapely.STRtree(county_geom)
    totals = np.zeros(len(counties))
    fractions = np.zeros(len(polygons))
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def batch(start):
        cache = checkpoint_dir / f"{start:09d}.npz" if checkpoint_dir else None
        size = min(chunk_size, len(polygons) - start)
        if cache and cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                partial_totals, partial_fractions = saved["totals"], saved["fractions"]
            if partial_totals.shape != totals.shape or partial_fractions.shape != (size,):
                raise ValueError(f"Invalid recreation checkpoint: {cache}")
            if not np.isfinite(partial_totals).all() or not np.isfinite(partial_fractions).all():
                raise ValueError(f"Non-finite recreation checkpoint: {cache}")
            return start, partial_totals, partial_fractions
        partial_totals = np.zeros(len(counties))
        partial_fractions = np.zeros(size)
        subset = source_geom[start:start + chunk_size]
        si, ci = tree.query(subset, predicate="intersects")
        if len(si):
            intersection_area = shapely.area(shapely.intersection(subset[si], county_geom[ci]))
            fraction = intersection_area / areas[start + si]
            np.add.at(partial_fractions, si, fraction)
            np.add.at(partial_totals, ci, values[start + si] * fraction)
        if cache:
            with tempfile.NamedTemporaryFile(dir=checkpoint_dir, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, totals=partial_totals, fractions=partial_fractions)
            temporary.replace(cache)
        return start, partial_totals, partial_fractions

    starts = range(0, len(polygons), chunk_size)
    batch_totals = np.zeros((len(starts), len(counties)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(batch, start) for start in starts]
        for future in tqdm(as_completed(futures), total=len(starts), desc="Recreation batches", unit="batch"):
            start, partial_totals, partial_fractions = future.result()
            batch_totals[start // chunk_size] = partial_totals
            fractions[start:start + len(partial_fractions)] = partial_fractions
    # Progress follows completed batches, but totals reduce in stable source order.
    totals = batch_totals.sum(axis=0)
    if (fractions > 1 + 1e-6).any():
        raise ValueError("County overlaps allocate a recreation polygon more than once")
    close(float(totals.sum()), float(np.dot(values, fractions)), "Recreation allocation")
    audit = pd.DataFrame({"source_key": polygons[id_field].astype(str).to_numpy(),
                          "input_value": values, "allocated_fraction": fractions,
                          "allocated_value": values * fractions,
                          "unallocated_value": values * (1 - fractions)})
    return pd.Series(totals, index=counties.GEOID), audit


def input_weights(path, ids, prefix):
    frame = read_attributes(path)[0] if path.suffix == ".gpkg" else pd.read_csv(path, dtype={"GEOID": str})
    frame["GEOID"] = frame.GEOID.map(fips)
    fields = [c for c in frame if c.startswith(prefix)]
    if len(fields) != 1 or frame.GEOID.duplicated().any() or set(frame.GEOID) != set(ids):
        raise ValueError(f"Unexpected county weight schema/coverage: {path}")
    return numbers(frame.set_index("GEOID")[fields[0]]).reindex(ids)


def write_output(base_path, output, layer, geom, additions, tables):
    """Back up the whole GeoPackage and change attributes only, then validate."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="county-integration-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        with closing(read_db(base_path)) as original, closing(sqlite3.connect(stage)) as con:
            original.backup(con)
            # GDAL's FID-update triggers reference these functions even when their
            # WHEN condition is false. Any actual invocation is a programming error:
            # this integration must never modify FIDs or geometries.
            def unexpected_geometry_edit(*args):
                raise RuntimeError("Unexpected geometry edit")
            for name in ("ST_IsEmpty", "ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
                con.create_function(name, 1, unexpected_geometry_edit)
            original_fields = [r[1] for r in con.execute(f"PRAGMA table_info({quote(layer)})")]
            if set(additions.columns) & set(original_fields):
                raise ValueError("Output fields already present; use the original base input")
            for field in DROP_FIELDS:
                con.execute(f"ALTER TABLE {quote(layer)} DROP COLUMN {quote(field)}")
            for field in tqdm(additions.columns, desc="Create valuation fields", unit="field"):
                con.execute(f"ALTER TABLE {quote(layer)} ADD COLUMN {quote(field)} REAL")
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
            keep = [c for c in original_fields if c not in DROP_FIELDS]
            query = f"SELECT {','.join(map(quote, keep))} FROM {quote(layer)} ORDER BY fid"
            before, after = original.execute(query), con.execute(query)
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
        # Hard-link publication is atomic and refuses to replace existing files.
        output.hardlink_to(stage)


def compute_task(task, paths, base, crosswalk, geometry_workers, chunk_dir):
    """Compute one independent service family in a worker process."""
    ids = base.index
    recreation_audit = pd.DataFrame()
    originals, statuses, metadata, issues, checks = {}, [], [], [], []

    def add(service, values, status, source, source_field, method, audit=None):
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

    def county_add(service, frame, key, field, source, **options):
        values, status, problems, audit = county_values(frame, key, field, ids, crosswalk, **options)
        if service == "timber":
            timber_status = frame.assign(_key=frame[key].map(fips)).set_index("_key").coverage_status
            for county_id in ids.intersection(timber_status.index):
                if timber_status[county_id] != "valued":
                    status[county_id] = str(timber_status[county_id]) + "_zero"
        issues.extend(dict(service=service, **p) for p in problems)
        add(service, values, status, source, field, "FIPS join; historical CT allocated by land area", audit)

    for service, field in (BASE_FIELDS.items() if task == "base" else []):
        values = numbers(base[field])
        status = pd.Series(np.where(values.isna(), "missing_source_value", "existing_base"), index=ids)
        add(service, values, status, "base", field, "existing county aggregate preserved")
    if task == "grazing":
        grazing = pd.read_excel(paths["grazing"], sheet_name="SNAPPGrazingCalculations", engine="openpyxl")
        county_add("grazing", grazing, "CountyCode", "CountyGrazingValueYear", "grazing")
    if task == "timber":
        timber = pd.read_csv(paths["timber"], dtype={"fips_padded": str})
        county_add("timber", timber, "fips_padded", "total_annual_rent", "timber", zero_missing=True)
    if task == "air_quality":
        air = pd.read_csv(paths["air_quality"], dtype={"FIPS": str})
        if air.duplicated(["FIPS", "land", "pollutant"]).any():
            raise ValueError("Duplicate air-quality county/land/pollutant rows")
        county_add("avoided_air_quality_heath_costs", air, "FIPS", "dollars_annual", "air_quality", aggregate=True)
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
    for service, weight_source, prefix, field in (
        ("marine_fisheries", "coastline", "intersect_length_km_", "ncp_value"),
        ("inland_fisheries", "freshwater", "intersect_area_ha_", "Consumptive use value (USD)_Intermediate assumption"),
    ):
        if task != service:
            continue
        weights = input_weights(paths[weight_source], ids, prefix)
        frame = pd.read_csv(paths[service])
        values, status, state_checks = state_values(frame, field, base.STATEFP, weights)
        checks.extend(dict(service=service, **c) for c in state_checks)
        add(service, values, status, service, field, f"state total allocated by county {weight_source} share")
    if task == "recreation":
        logging.info("Reading county and recreation geometry")
        counties = gpd.read_file(paths["base"], columns=["GEOID"])
        counties.GEOID = counties.GEOID.map(fips)
        polygons = gpd.read_file(paths["recreation"], columns=["siteid", "val_2024"])
        recreation, recreation_audit = polygon_values(counties, polygons, "val_2024", "siteid", workers=geometry_workers, checkpoint_dir=chunk_dir)
        add("recreation", recreation, pd.Series("polygon_area_allocation", index=ids), "recreation", "val_2024",
            "EPSG:6933 intersection area / full source polygon area; uncovered value retained in audit",
            {"input_known_total": float(recreation_audit.input_value.sum()),
             "unmatched_total": float(recreation_audit.unallocated_value.sum())})
    return {"originals": originals, "statuses": statuses, "metadata": metadata,
            "issues": issues, "checks": checks, "recreation_audit": recreation_audit}


def save_checkpoint(path, signature, result):
    """Publish a complete SQLite checkpoint atomically; never load Python pickle."""
    with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=path.parent) as temporary:
        stage = Path(temporary) / path.name
        with closing(sqlite3.connect(stage)) as con:
            pd.DataFrame(result["originals"]).to_sql("valuations", con, index_label="GEOID")
            pd.concat(result["statuses"], ignore_index=True).to_sql("statuses", con, index=False)
            if len(result["recreation_audit"].columns):
                result["recreation_audit"].to_sql("recreation", con, index=False)
            payload = {k: result[k] for k in ("metadata", "issues", "checks")}
            con.execute("CREATE TABLE completed (signature TEXT, payload TEXT)")
            con.execute("INSERT INTO completed VALUES (?, ?)", (signature, json.dumps(payload, allow_nan=False)))
            con.commit()
        stage.replace(path)


def load_checkpoint(path, signature):
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
        has_recreation = con.execute("SELECT 1 FROM sqlite_master WHERE name='recreation'").fetchone()
        result["recreation_audit"] = pd.read_sql_query("SELECT * FROM recreation", con) if has_recreation else pd.DataFrame()
    return result


def run(data_root, config_dir, output=None, workers=8, geometry_workers=8, resume=True):
    if workers < 1 or geometry_workers < 1:
        raise ValueError("Worker counts must be positive")
    script_hash = sha256(__file__)
    config_hashes = {name: sha256(config_dir / name) for name in
                     ("sources.csv", "adjustment_factors.csv", "ct_county_crosswalk.csv")}
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
    fingerprints = {name: sha256(path) for name, path in tqdm(paths.items(), desc="Hash sources", unit="file")}
    if output is not None and output.exists():
        raise FileExistsError(output)
    base, layer, geom = read_attributes(paths["base"])
    base["GEOID"] = base.GEOID.map(fips)
    if base.GEOID.duplicated().any():
        raise ValueError("Duplicate base county FIPS")
    base = base.set_index("GEOID")
    ids = base.index
    if not (base.STATEFP == ids.str[:2]).all():
        raise ValueError("Base state and county identifiers disagree")
    crosswalk = pd.read_csv(config_dir / "ct_county_crosswalk.csv", dtype={"source_geoid": str, "target_geoid": str})
    validate_crosswalk(crosswalk, ids)
    cache_root = data_root / "processing_outputs/county_integration_2026"
    cache_root.mkdir(parents=True, exist_ok=True)
    dependencies = {
        "base": ["base"], "grazing": ["base", "grazing"], "timber": ["base", "timber"],
        "air_quality": ["base", "air_quality"], "urban_heat": ["base", "urban_heat"],
        "physical_health": ["base", "physical_health"], "jobs": ["base", "jobs"],
        "marine_fisheries": ["base", "marine_fisheries", "coastline"],
        "inland_fisheries": ["base", "inland_fisheries", "freshwater"],
        "recreation": ["base", "recreation"],
    }
    crosswalk_hash = config_hashes["ct_county_crosswalk.csv"]
    results = {}
    signatures = {}
    pending = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, names in dependencies.items():
            signature = hashlib.sha256(json.dumps({"script": script_hash, "crosswalk": crosswalk_hash,
                "versions": [pd.__version__, np.__version__, shapely.__version__, gpd.__version__],
                "inputs": {n: fingerprints[n] for n in names}}, sort_keys=True).encode()).hexdigest()
            signatures[task] = signature
            cache = cache_root / f"{task}_{signature}.sqlite"
            if resume and cache.exists():
                results[task] = load_checkpoint(cache, signature)
                logging.info("Resumed completed service: %s", task)
            else:
                chunk_dir = cache_root / f"recreation_chunks_{signature}" if task == "recreation" and resume else None
                future = pool.submit(compute_task, task, paths, base, crosswalk, geometry_workers, chunk_dir)
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
    recreation_audit = results["recreation"]["recreation_audit"]
    if set(originals) != set(factors.index):
        raise ValueError("Adjustment factor configuration does not exactly match integrated services")
    additions = pd.DataFrame(index=ids)
    for service, values in originals.items():
        factor = float(factors.loc[service, "factor"])
        additions[f"{service}_orig"] = values
        additions[f"{service}_adj_factor"] = factor
        additions[f"{service}_adj"] = values * factor
    # Source hashes also make interrupted/rerun inputs identifiable.
    for name, path in tqdm(paths.items(), desc="Verify source hashes", unit="file"):
        if sha256(path) != fingerprints[name]:
            raise ValueError(f"Source changed during integration: {path}")
    if sha256(__file__) != script_hash or any(sha256(config_dir / name) != digest for name, digest in config_hashes.items()):
        raise ValueError("Script or configuration changed during integration; rerun")
    source_audit = sources.copy()
    source_audit["sha256"] = source_audit.source.map(fingerprints)
    source_audit["bytes"] = source_audit.source.map({n: p.stat().st_size for n, p in paths.items()})
    source_audit = pd.concat([source_audit, pd.DataFrame([
        {"source": name, "path": name, "sha256": sha256(config_dir / name), "bytes": (config_dir / name).stat().st_size}
        for name in ("sources.csv", "adjustment_factors.csv", "ct_county_crosswalk.csv")
    ])], ignore_index=True)
    meta = pd.DataFrame(metadata)
    for field in ("factor", "reason", "source_units"):
        meta["factor_reason" if field == "reason" else field] = meta.service.map(factors[field])
    meta["adjusted_total"] = meta.original_total * meta.factor
    tables = {
        "integration_services": meta,
        "integration_sources": source_audit,
        "integration_county_status": pd.concat(statuses, ignore_index=True),
        "integration_source_issues": pd.DataFrame(issues, columns=["service", "source_key", "reason", "value"]),
        "integration_state_checks": pd.DataFrame(checks),
        "integration_recreation_allocation": recreation_audit,
        "integration_ct_crosswalk": crosswalk,
        "integration_checkpoints": pd.DataFrame({"task": list(signatures), "signature": list(signatures.values())}),
        "integration_run": pd.DataFrame([{
            "created_utc": datetime.now(timezone.utc).isoformat(), "script": Path(__file__).name,
            "script_sha256": sha256(__file__), "area_crs": AREA_CRS,
            "base_counties": len(ids), "removed_fields": json.dumps(DROP_FIELDS),
            "notes": "Carbon units not reinterpreted as dollars; mangrove signs retained; no cross-service grand total. "
                     "Unclassified base fields and non-sum crop/pollination diagnostics retained unchanged.",
        }]),
    }
    if output is None:
        output = data_root / "analysis_results/combined" / f"counties_ecosystem_services_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
    logging.info("Writing %s", output)
    write_output(paths["base"], output, layer, geom, additions, tables)
    logging.info("Verified %s counties, %s valuation fields, geometry and original attributes preserved", len(ids), len(additions.columns))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--config-dir", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, help="New GeoPackage path; existing files are never overwritten")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Independent service processes (default: up to 8)")
    parser.add_argument("--geometry-workers", type=int, default=min(8, os.cpu_count() or 1), help="Recreation intersection threads (default: up to 8)")
    parser.add_argument("--no-resume", action="store_true", help="Recompute services and recreation batches instead of reusing checkpoints")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(args.data_root.resolve(), args.config_dir.resolve(), args.output.resolve() if args.output else None,
              args.workers, args.geometry_workers, not args.no_resume))


if __name__ == "__main__":
    main()
