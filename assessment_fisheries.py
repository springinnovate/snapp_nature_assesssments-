"""Allocate state fishery values over county and land-footprint geometries."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyogrio import read_dataframe, read_info
from pyproj import Transformer
import shapely
from tqdm import tqdm

from assessment_geography import CONUS_STATE_FIPS, county_bounds_in_crs
from build_county_ecosystem_service_valuations_2026 import AREA_CRS, STATE_NAMES, close, numbers
from geometry_utils import repair_polygonal_geometry

_FIELDS = {
    "marine_fisheries": "ncp_value",
    "inland_fisheries": "Consumptive use value (USD)_Intermediate assumption",
}
_UTAH_FIPS = "49"
# Recognize non-CONUS names that may be supplied by the source tables, even
# when absent from the legacy county builder's state lookup.
_STATE_NAMES = {
    **STATE_NAMES,
    "AMERICAN SAMOA": "60",
    "GUAM": "66",
    "NORTHERN MARIANA ISLANDS": "69",
    "COMMONWEALTH OF THE NORTHERN MARIANA ISLANDS": "69",
    "U.S. MINOR OUTLYING ISLANDS": "74",
    "UNITED STATES MINOR OUTLYING ISLANDS": "74",
    "U.S. VIRGIN ISLANDS": "78",
    "UNITED STATES VIRGIN ISLANDS": "78",
}


def _read_state_totals(path: Path, field: str) -> pd.Series:
    """Load CONUS state values, preserving blanks and excluding the total row.

    Args:
        path: Fisheries CSV with a State column.
        field: Monetary valuation field to use.

    Returns:
        Values indexed by two-character CONUS state FIPS code.

    Raises:
        ValueError: A source state is unknown or an in-scope state occurs
            more than once.
    """
    frame = pd.read_csv(path)
    states = frame["State"].fillna("").str.strip().str.upper()
    include = ~states.isin(["", "TOTAL"])
    ids = states.loc[include].map(_STATE_NAMES)
    if ids.isna().any():
        raise ValueError(f"Unrecognized or duplicate state in fisheries source: {path}")
    ids = ids.loc[ids.isin(CONUS_STATE_FIPS)]
    if ids.duplicated().any():
        raise ValueError(f"Unrecognized or duplicate state in fisheries source: {path}")
    return pd.Series(numbers(frame.loc[ids.index, field]).to_numpy(), index=ids.to_numpy())


def _read_marine_shoreline(path: Path, counties: gpd.GeoDataFrame) -> shapely.STRtree:
    """Index marine shoreline after scoping it in the source CRS.

    Args:
        path: Coastline source with waterbody labels in NAME.
        counties: CONUS county geometries defining the spatial envelope.

    Returns:
        Spatial index of shoreline projected into the measurement CRS.

    Raises:
        ValueError: The coastline source has no CRS.
    """
    info = read_info(path)
    if not info["crs"]:
        raise ValueError(f"Coastline input must define a CRS: {path}")
    bounds = county_bounds_in_crs(counties, info["crs"])
    coast = read_dataframe(path, columns=["NAME"], bbox=bounds)
    great_lakes = coast["NAME"].astype("string").str.strip().str.casefold().eq("great lakes")
    # Clip the candidate lines before projection, not after: distant parts
    # of a returned multipart feature should not project across this domain.
    marine = coast.loc[~great_lakes.fillna(False)].geometry
    clipped = shapely.intersection(marine.to_numpy(), shapely.box(*bounds))
    clipped = clipped[shapely.length(clipped) > 0]
    return shapely.STRtree(gpd.GeoSeries(clipped, crs=coast.crs).to_crs(AREA_CRS).to_numpy())


def _read_excluded_lakes(path: Path, features: Path, label: str):
    """Read and dissolve NHD footprints to subtract from inland weights.

    Args:
        path: Prepared NHD GeoPackage containing the freshwater geometries.
        features: Tracked CSV of NHDWaterbody source identifiers to exclude.
        label: Lake or lake group name for validation messages.

    Returns:
        The combined exclusion geometry in the measurement CRS.

    Raises:
        ValueError: The identifier list is invalid or not found in the source.
    """
    ids = pd.read_csv(features, dtype={"source_fid": "int64"})["source_fid"]
    if ids.empty or (ids < 0).any() or ids.duplicated().any():
        raise ValueError(f"Expected distinct nonnegative {label} source FIDs in {features}")
    id_list = ", ".join(map(str, ids))
    lakes = read_dataframe(
        path, columns=["source_fid"],
        where=f"source_layer = 'NHDWaterbody' AND source_fid IN ({id_list})",
    )
    missing = set(ids) - set(lakes["source_fid"])
    if missing:
        raise ValueError(f"Missing {label} features in {path}: {sorted(missing)}")
    projected = lakes.to_crs(AREA_CRS).geometry.to_numpy()
    merged = repair_polygonal_geometry(shapely.union_all(shapely.make_valid(projected)))
    if merged is None or not np.isfinite(merged.area) or merged.area <= 0:
        raise ValueError(f"No valid {label} polygon area in {path}")
    shapely.prepare(merged)
    return merged


def _measure(zone, tree: shapely.STRtree, *, length: bool) -> float:
    """Sum eligible source-feature intersections with one zone.

    Args:
        zone: Projected geometry to measure inside.
        tree: Spatial index of source lines or polygons in the same CRS.
        length: Measure kilometres when true, hectares otherwise.

    Returns:
        Sum of feature lengths or areas in the zone, or zero for no overlap.
    """
    ids = tree.query(zone, predicate="intersects")
    intersections = shapely.intersection(tree.geometries[ids], zone)
    measures = shapely.length(intersections) if length else shapely.area(intersections)
    return float(measures.sum()) / (1000 if length else 10_000)


def value_fisheries(
    zones: Mapping[str, gpd.GeoDataFrame],
    paths: Mapping[str, Path],
    geometry_workers: int,
) -> dict[str, pd.DataFrame]:
    """Measure fishery weights and apply one rate per service and state.

    CONUS county measurements set the fixed state denominators. Each footprint,
    including each manager row, gets the same rate times its own weight. The
    Great Lakes are excluded from marine shoreline and inland water area.
    Great Salt Lake is excluded only from Utah's inland water area. The
    state totals, source data and zone geometries are not changed.

    Args:
        zones: Product frames keyed by name, with GEOID and geometry. The
            counties product must contain one row per county in the contiguous
            United States and DC; the other frames can repeat GEOID for manager
            groups.
        paths: Input paths keyed as in the county integration sources table.
        geometry_workers: Maximum concurrent county measurement tasks.

    Returns:
        A table per product, indexed like its zone frame, with unadjusted
        marine_fisheries and inland_fisheries values. A measured zero in a
        valued state is zero; missing state values remain null.

    Raises:
        ValueError: Zone keys, spatial measurements, or state allocations
            are inconsistent with the full-county baseline.
    """
    if geometry_workers < 1:
        raise ValueError("Fisheries geometry_workers must be positive")
    if "counties" not in zones or zones["counties"].empty:
        raise ValueError("A complete counties layer is required for fisheries denominators")

    zone_ids = {product: frame["GEOID"].astype("string") for product, frame in zones.items()}
    county_ids = zone_ids["counties"]
    if county_ids.duplicated().any() or not county_ids.str.fullmatch(r"[0-9]{5}", na=False).all():
        raise ValueError("Fisheries county GEOIDs must be unique five-digit strings")
    non_conus = sorted(set(county_ids.str[:2]) - CONUS_STATE_FIPS)
    if non_conus:
        raise ValueError(f"CONUS fisheries received out-of-scope county state FIPS codes: {non_conus}")
    projected = {}
    by_county = {}
    for product, frame in zones.items():
        ids = zone_ids[product]
        if not ids.isin(county_ids).all():
            raise ValueError(f"{product} fisheries zones contain unknown county GEOIDs")
        geometries = frame.to_crs(AREA_CRS).geometry.to_numpy(copy=True)
        for position in np.flatnonzero(~shapely.is_valid(geometries)):
            repaired = repair_polygonal_geometry(shapely.make_valid(geometries[position]))
            if repaired is None:
                raise ValueError(f"Invalid {product} fisheries geometry for GEOID {ids.iloc[position]}")
            geometries[position] = repaired
        projected[product] = geometries
        by_county[product] = ids.groupby(ids, sort=False).indices

    totals = {service: _read_state_totals(paths[service], field) for service, field in _FIELDS.items()}
    marine_tree = _read_marine_shoreline(paths["coastline_lines"], zones["counties"])
    water_path = paths["freshwater_polygons"]
    great_lakes = _read_excluded_lakes(water_path, paths["great_lakes_features"], "Great Lakes")
    great_salt_lake = _read_excluded_lakes(
        water_path, paths["inland_fisheries_exclusions"], "Great Salt Lake",
    )
    water_info = read_info(water_path)
    if not water_info["crs"] or not water_info["capabilities"]["fast_spatial_filter"]:
        raise ValueError(f"Freshwater input must have a CRS and a spatial index: {water_path}")
    to_native = Transformer.from_crs(AREA_CRS, water_info["crs"], always_xy=True)

    def county_weights(task: tuple[int, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Read local water once and measure all zone rows for a county.

        Args:
            task: Position of the county row and its GEOID.

        Returns:
            Products represented in this county, each with its row positions
            and an aligned array of marine kilometres and inland hectares.
        """
        county_pos, geoid = task
        state = geoid[:2]
        county = projected["counties"][county_pos]
        measure_marine = state in totals["marine_fisheries"].index
        measure_inland = state in totals["inland_fisheries"].index
        if not measure_marine and not measure_inland:
            return {}
        inland_tree = None
        if measure_inland:
            # The NHD layer has millions of features. The native-CRS spatial
            # filter uses its R-tree before reading any candidate geometries.
            bbox = to_native.transform_bounds(*county.bounds, densify_pts=41)
            water = read_dataframe(water_path, columns=[], bbox=bbox).to_crs(AREA_CRS)
            candidates = water.geometry.to_numpy(copy=True)
            invalid = ~shapely.is_valid(candidates)
            if invalid.any():
                candidates[invalid] = shapely.make_valid(candidates[invalid])
            inland_tree = shapely.STRtree(candidates)
        excluded_lakes = great_salt_lake if state == _UTAH_FIPS else great_lakes
        county_lakes = (
            shapely.intersection(excluded_lakes, county)
            if measure_inland and shapely.intersects(excluded_lakes, county) else None
        )
        measured = {}
        for product, groups in by_county.items():
            positions = groups.get(geoid)
            if positions is None:
                continue
            values = np.zeros((len(positions), len(_FIELDS)), dtype=float)
            for row, position in enumerate(positions):
                geometry = projected[product][position]
                if measure_marine:
                    values[row, 0] = _measure(geometry, marine_tree, length=True)
                if inland_tree is not None:
                    inland_zone = shapely.difference(geometry, county_lakes) if county_lakes is not None else geometry
                    values[row, 1] = _measure(inland_zone, inland_tree, length=False)
            measured[product] = positions, values
        return measured

    weights = {product: np.zeros((len(frame), len(_FIELDS))) for product, frame in zones.items()}
    with ThreadPoolExecutor(max_workers=geometry_workers) as pool:
        for measured in tqdm(
            pool.map(county_weights, enumerate(county_ids)), total=len(county_ids),
            desc="Measure fisheries zones", unit="county",
        ):
            for product, (positions, values) in measured.items():
                weights[product][positions] = values

    county_states = county_ids.str[:2].to_numpy()
    baseline = pd.DataFrame(weights["counties"], index=county_ids, columns=_FIELDS)
    state_weights = baseline.groupby(county_states).sum()
    rate_by_state = {}
    for service, state_totals in totals.items():
        support = state_weights[service].reindex(state_totals.index)
        unsupported = support.isna() | support.le(0)
        if unsupported.any():
            raise ValueError(f"No allocation support for {service} states: {list(support.index[unsupported])}")
        rate_by_state[service] = state_totals / support

    results = {}
    for product, frame in zones.items():
        values = weights[product]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Nonfinite or negative fisheries weights in {product}")
        if product != "counties":
            county_values = baseline.reindex(zone_ids[product]).to_numpy()
            tolerance = np.maximum(1e-7, county_values * 1e-9)
            if (values > county_values + tolerance).any():
                raise ValueError(f"Fisheries weight exceeds county baseline in {product}")
            # Only suppress sub-tolerance numerical overlay excess.
            values = np.minimum(values, county_values)
        table = pd.DataFrame(index=frame.index)
        for col, service in enumerate(_FIELDS):
            rates = zone_ids[product].str[:2].map(rate_by_state[service]).to_numpy(dtype=float)
            table[service] = values[:, col] * rates
        results[product] = table

    for service, state_totals in totals.items():
        allocated = results["counties"][service].groupby(county_states).sum(min_count=1)
        for state, value in state_totals.dropna().items():
            close(allocated.loc[state], value, f"{service} state {state} allocation")
    return results
