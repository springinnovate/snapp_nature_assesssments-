"""Read county denominators and allocate combined Virginia GDP without duplication."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def read_county_source(path: Path, spec: dict) -> tuple[pd.Series, int]:
    """Read, select, normalize, and scale a county-keyed numeric source.

    Args:
        path: CSV source path.
        spec: Source configuration with key, field, encoding, optional filters,
            missing-value tokens, scale, and state-total exclusion settings.

    Returns:
        Numeric values indexed by five-character geographic keys and the
        number of rows in the original CSV before selection.

    Raises:
        ValueError: Keys are malformed or duplicated after selection, selected
            values are nonnumeric or infinite, or the scale is invalid.
    """
    key = spec["key"]
    filters = spec.get("where", {})
    table = pd.read_csv(
        path,
        encoding=spec.get("encoding", "utf-8-sig"),
        dtype={column: str for column in (key, *filters)},
        na_values=spec.get("missing_tokens", []),
    )
    original_count = len(table)
    for column, required in filters.items():
        table = table.loc[table[column].str.strip().eq(str(required))].copy()
    keys = table[key].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    valid_keys = keys.str.fullmatch(r"[0-9]{1,5}").fillna(False)
    if not valid_keys.all():
        raise ValueError(f"Invalid county keys in {path}: {table.loc[~valid_keys, key].tolist()[:5]}")
    table[key] = keys.str.zfill(5)
    if spec.get("exclude_state_totals", False):
        table = table.loc[~table[key].str.endswith("000")].copy()
    if table[key].duplicated().any():
        raise ValueError(f"Duplicate county keys in {path}: {table.loc[table[key].duplicated(), key].tolist()[:5]}")
    scale = spec.get("scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid numeric scale for {path}: {scale!r}")
    values = pd.to_numeric(table.set_index(key)[spec["field"]], errors="raise").astype(float) * scale
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f"Infinite county values in {path}")
    return values, original_count


def allocate_combined_gdp(
    gdp: pd.Series, population: pd.Series, crosswalk_path: Path
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split combined Virginia GDP using full-group 2024 population shares.

    Args:
        gdp: GDP in dollars indexed by normalized geographic identifiers.
        population: County populations indexed by normalized county identifiers.
            Include every crosswalk member, even for a subset output.
        crosswalk_path: CSV mapping each BEA combination area to its constituent
            counties and independent cities, with names and source URLs.

    Returns:
        GDP with combination rows replaced by allocated county rows, the
        member-level allocation audit, and one conservation check per group.

    Raises:
        ValueError: The crosswalk or source indexes are invalid, required groups
            or populations are unavailable, GDP is already present for a member,
            or any allocation fails to conserve its source total.
    """
    for label, source in (("GDP", gdp), ("population", population)):
        keys = pd.Index(source.index.astype(str))
        if (source.index.has_duplicates or not all(isinstance(key, str) for key in source.index)
                or not keys.str.fullmatch(r"[0-9]{5}").all()):
            raise ValueError(f"{label} requires unique normalized five-character keys")
        if np.isinf(pd.to_numeric(source, errors="raise").to_numpy(dtype=float)).any():
            raise ValueError(f"{label} contains infinite values")
    columns = ["bea_geofips", "county_geoid", "bea_name", "county_name", "source_url"]
    crosswalk = pd.read_csv(crosswalk_path, dtype=str, keep_default_na=False)
    if set(crosswalk.columns) != set(columns) or crosswalk.empty:
        raise ValueError(f"GDP crosswalk must have exactly these columns and at least one row: {columns}")
    crosswalk = crosswalk[columns].apply(lambda column: column.str.strip())
    if (crosswalk.eq("").any().any()
            or not crosswalk.bea_geofips.str.fullmatch(r"519[0-9]{2}").all()
            or not crosswalk.county_geoid.str.fullmatch(r"51[0-9]{3}").all()
            or crosswalk.county_geoid.str.startswith("519").any()
            or crosswalk.duplicated(["bea_geofips", "county_geoid"]).any()
            or crosswalk.county_geoid.duplicated().any()):
        raise ValueError("GDP crosswalk has blank, invalid, duplicated, or multiply assigned county keys")
    group_codes = set(crosswalk.bea_geofips)
    source_groups = {key for key in gdp.index if str(key).startswith("519")}
    if source_groups != group_codes:
        raise ValueError(
            "Combined Virginia GDP groups do not match the crosswalk: "
            f"unmapped={sorted(source_groups - group_codes)}, missing={sorted(group_codes - source_groups)}"
        )
    collisions = set(crosswalk.county_geoid) & set(gdp.index)
    if collisions:
        raise ValueError(f"Direct GDP values already exist for allocated county keys: {sorted(collisions)}")
    allocations = []
    checks = []
    for group_code, group in crosswalk.groupby("bea_geofips", sort=False):
        if group.bea_name.nunique() != 1 or group.source_url.nunique() != 1 or len(group) < 2:
            raise ValueError(f"GDP crosswalk needs consistent names/sources and at least two members for {group_code}")
        source_gdp = float(gdp.loc[group_code])
        if not math.isfinite(source_gdp) or source_gdp <= 0:
            raise ValueError(f"Missing or nonpositive combined GDP for {group_code}")
        member_population = pd.to_numeric(population.reindex(group.county_geoid), errors="raise").to_numpy(dtype=float)
        if not np.isfinite(member_population).all() or not (member_population > 0).all():
            raise ValueError(f"Missing or nonpositive population for members of {group_code}")
        group_population = math.fsum(member_population)
        if not math.isfinite(group_population) or group_population <= 0:
            raise ValueError(f"Invalid combined population for {group_code}")
        weights = member_population / group_population
        member_gdp = weights * source_gdp
        # Assign only floating-point residue to the largest member so the source
        # total is conserved; do not round each allocation to whole dollars.
        member_gdp[int(np.argmax(weights))] += source_gdp - math.fsum(member_gdp)
        allocated = math.fsum(member_gdp)
        if not np.isfinite(member_gdp).all() or not math.isclose(allocated, source_gdp, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"Combined GDP allocation does not conserve {group_code}")
        for (_, row), pop, weight, value in zip(group.iterrows(), member_population, weights, member_gdp):
            allocations.append({
                **row.to_dict(),
                "population_2024": float(pop),
                "group_population_2024": group_population,
                "allocation_weight": float(weight),
                "source_gdp_2024": source_gdp,
                "allocated_gdp_2024": float(value),
                "allocation_method": "population_2024",
            })
        checks.append({
            "bea_geofips": group_code,
            "bea_name": group.bea_name.iloc[0],
            "source_gdp_2024": source_gdp,
            "allocated_gdp_2024": allocated,
            "difference_gdp_2024": allocated - source_gdp,
            "member_count": len(group),
            "group_population_2024": group_population,
            "weight_sum": math.fsum(weights),
            "allocation_method": "population_2024",
            "source_url": group.source_url.iloc[0],
        })
    allocation_table = pd.DataFrame(allocations)
    expanded = pd.concat([
        gdp.drop(index=list(group_codes)).astype(float),
        allocation_table.set_index("county_geoid")["allocated_gdp_2024"],
    ])
    expanded.index.name = gdp.index.name
    expanded.name = gdp.name
    if not expanded.index.is_unique or not math.isclose(
        math.fsum(expanded.dropna()), math.fsum(gdp.dropna()), rel_tol=1e-12, abs_tol=1e-6
    ):
        raise ValueError("Expanded GDP does not conserve the full source total")
    return expanded, allocation_table, pd.DataFrame(checks)
