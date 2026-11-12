"""Allocate state fishery values over county and land-footprint geometries."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyogrio import read_dataframe, read_info
from pyproj import Transformer
import shapely
from tqdm import tqdm

from assessment.assessment_geography import CONUS_STATE_FIPS, county_bounds_in_crs
from assessment.build_county_ecosystem_service_valuations_2026 import AREA_CRS, STATE_NAMES, close, numbers
from assessment.geometry_utils import repair_polygonal_geometry

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


def read_marine_shoreline(path: Path, counties: gpd.GeoDataFrame) -> shapely.STRtree:
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
    return _read_marine_shoreline_bounds(path, bounds)


def _read_marine_shoreline_bounds(path, bounds):
    """Load the marine index within an already validated source-CRS envelope."""
    coast = read_dataframe(path, columns=["NAME"], bbox=bounds)
    great_lakes = coast["NAME"].astype("string").str.strip().str.casefold().eq("great lakes")
    # Clip the candidate lines before projection, not after: distant parts
    # of a returned multipart feature should not project across this domain.
    marine = coast.loc[~great_lakes.fillna(False)].geometry
    clipped = shapely.intersection(marine.to_numpy(), shapely.box(*bounds))
    clipped = clipped[shapely.length(clipped) > 0]
    return shapely.STRtree(gpd.GeoSeries(clipped, crs=coast.crs).to_crs(AREA_CRS).to_numpy())


def read_excluded_lakes(path: Path, features: Path, label: str):
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


@lru_cache(maxsize=1)
def _fisheries_context(sources, shoreline_bounds, water_crs):
    """Cache read-only source indexes once per worker, never the national NHD.

    Source signatures contain path, size and modification time so a new plan
    cannot reuse cached geometries after an input changes in the same process.
    No open GDAL dataset handles are shared between processes.
    """
    paths = {name: Path(path) for name, path, _, _ in sources}
    water_path = paths["freshwater_polygons"]
    return (
        _read_marine_shoreline_bounds(paths["coastline_lines"], shoreline_bounds),
        read_excluded_lakes(water_path, paths["great_lakes_features"], "Great Lakes"),
        read_excluded_lakes(water_path, paths["inland_fisheries_exclusions"], "Great Salt Lake"),
        Transformer.from_crs(AREA_CRS, water_crs, always_xy=True),
    )


def _county_weights(context, geoid, county, local_zones, measure_marine, measure_inland):
    """Measure every product's local rows using one indexed county NHD read.

    The module-level worker is spawn-picklable. Its payload contains only this
    county's geometries and output positions, not the six national datasets.
    """
    if not measure_marine and not measure_inland:
        return {}
    marine_tree, great_lakes, great_salt_lake, to_native = _fisheries_context(*context)
    inland_tree = None
    if measure_inland:
        water_path = next(path for name, path, _, _ in context[0] if name == "freshwater_polygons")
        bbox = to_native.transform_bounds(*county.bounds, densify_pts=41)
        water = read_dataframe(water_path, columns=[], bbox=bbox).to_crs(AREA_CRS)
        candidates = water.geometry.to_numpy(copy=True)
        invalid = ~shapely.is_valid(candidates)
        if invalid.any():
            candidates[invalid] = shapely.make_valid(candidates[invalid])
        inland_tree = shapely.STRtree(candidates)
    excluded_lakes = great_salt_lake if geoid[:2] == _UTAH_FIPS else great_lakes
    county_lakes = (
        shapely.intersection(excluded_lakes, county)
        if measure_inland and shapely.intersects(excluded_lakes, county) else None
    )
    measured = {}
    for product, (positions, geometries) in local_zones.items():
        values = np.zeros((len(positions), len(_FIELDS)), dtype=float)
        for row, geometry in enumerate(geometries):
            if measure_marine:
                values[row, 0] = _measure(geometry, marine_tree, length=True)
            if inland_tree is not None:
                inland_zone = shapely.difference(geometry, county_lakes) if county_lakes is not None else geometry
                values[row, 1] = _measure(inland_zone, inland_tree, length=False)
        measured[product] = positions, values
    return measured


class FisheriesPlan:
    """Prepare county jobs for a caller-owned pool and reconcile their weights."""

    def __init__(self, zones: Mapping[str, gpd.GeoDataFrame], paths: Mapping[str, Path]):
        """Validate sources and retain projected geometry only in the coordinator."""
        if "counties" not in zones or zones["counties"].empty:
            raise ValueError("A complete counties layer is required for fisheries denominators")
        self.zones = zones
        self.zone_ids = {product: frame["GEOID"].astype("string") for product, frame in zones.items()}
        county_ids = self.zone_ids["counties"]
        if county_ids.duplicated().any() or not county_ids.str.fullmatch(r"[0-9]{5}", na=False).all():
            raise ValueError("Fisheries county GEOIDs must be unique five-digit strings")
        non_conus = sorted(set(county_ids.str[:2]) - CONUS_STATE_FIPS)
        if non_conus:
            raise ValueError(f"CONUS fisheries received out-of-scope county state FIPS codes: {non_conus}")
        self.projected, self.by_county = {}, {}
        for product, frame in zones.items():
            ids = self.zone_ids[product]
            if not ids.isin(county_ids).all():
                raise ValueError(f"{product} fisheries zones contain unknown county GEOIDs")
            geometries = frame.to_crs(AREA_CRS).geometry.to_numpy(copy=True)
            for position in np.flatnonzero(~shapely.is_valid(geometries)):
                repaired = repair_polygonal_geometry(shapely.make_valid(geometries[position]))
                if repaired is None:
                    raise ValueError(f"Invalid {product} fisheries geometry for GEOID {ids.iloc[position]}")
                geometries[position] = repaired
            self.projected[product] = geometries
            self.by_county[product] = ids.groupby(ids, sort=False).indices
        self.totals = {service: _read_state_totals(paths[service], field) for service, field in _FIELDS.items()}
        coast_info = read_info(paths["coastline_lines"])
        if not coast_info["crs"]:
            raise ValueError(f"Coastline input must define a CRS: {paths['coastline_lines']}")
        # Compute from actual county vertices, not a transformed bounding box.
        bounds = county_bounds_in_crs(zones["counties"], coast_info["crs"])
        water_path = paths["freshwater_polygons"]
        water_info = read_info(water_path)
        if not water_info["crs"] or not water_info["capabilities"]["fast_spatial_filter"]:
            raise ValueError(f"Freshwater input must have a CRS and a spatial index: {water_path}")
        sources = []
        for name in ("coastline_lines", "freshwater_polygons", "great_lakes_features", "inland_fisheries_exclusions"):
            path = Path(paths[name]).resolve()
            stat = path.stat()
            sources.append((name, str(path), stat.st_size, stat.st_mtime_ns))
        self.context = (tuple(sources), bounds, water_info["crs"])
        # Validate source geometries before submitting jobs, also warming the
        # cache for the standalone thread-based compatibility entry point.
        _fisheries_context(*self.context)
        self.weights = {product: np.zeros((len(frame), len(_FIELDS))) for product, frame in zones.items()}
        self.remaining = set(county_ids)
        self.task_count = len(county_ids)

    def tasks(self):
        """Yield county-local payloads lazily so a bounded pool queue limits copies."""
        for county_pos, geoid in enumerate(self.zone_ids["counties"]):
            local = {}
            for product, groups in self.by_county.items():
                positions = groups.get(geoid)
                if positions is not None:
                    local[product] = positions, self.projected[product][positions]
            yield geoid, _county_weights, (
                self.context, geoid, self.projected["counties"][county_pos], local,
                geoid[:2] in self.totals["marine_fisheries"].index,
                geoid[:2] in self.totals["inland_fisheries"].index,
            )

    def accept(self, key, result):
        """Place a completed county in fixed row positions, independent of order."""
        if key not in self.remaining:
            raise ValueError(f"Duplicate or unknown fisheries county result: {key}")
        for product, (positions, values) in result.items():
            self.weights[product][positions] = values
        self.remaining.remove(key)

    def finish(self):
        """Apply fixed state rates and retain the county and footprint checks."""
        if self.remaining:
            raise ValueError(f"Missing fisheries county results: {sorted(self.remaining)}")
        county_ids = self.zone_ids["counties"]
        county_states = county_ids.str[:2].to_numpy()
        baseline = pd.DataFrame(self.weights["counties"], index=county_ids, columns=_FIELDS)
        state_weights = baseline.groupby(county_states).sum()
        rate_by_state = {}
        for service, state_totals in self.totals.items():
            support = state_weights[service].reindex(state_totals.index)
            unsupported = support.isna() | support.le(0)
            if unsupported.any():
                raise ValueError(f"No allocation support for {service} states: {list(support.index[unsupported])}")
            rate_by_state[service] = state_totals / support

        results = {}
        for product, frame in self.zones.items():
            values = self.weights[product]
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Nonfinite or negative fisheries weights in {product}")
            if product != "counties":
                county_values = baseline.reindex(self.zone_ids[product]).to_numpy()
                tolerance = np.maximum(1e-7, county_values * 1e-9)
                if (values > county_values + tolerance).any():
                    raise ValueError(f"Fisheries weight exceeds county baseline in {product}")
                values = np.minimum(values, county_values)
            table = pd.DataFrame(index=frame.index)
            for col, service in enumerate(_FIELDS):
                rates = self.zone_ids[product].str[:2].map(rate_by_state[service]).to_numpy(dtype=float)
                table[service] = values[:, col] * rates
            results[product] = table
        for service, state_totals in self.totals.items():
            allocated = results["counties"][service].groupby(county_states).sum(min_count=1)
            for state, value in state_totals.dropna().items():
                close(allocated.loc[state], value, f"{service} state {state} allocation")
        return results


def value_fisheries(zones, paths, geometry_workers):
    """Standalone fisheries calculation; the assessment uses its shared pool.

    Full counties define state denominators for every footprint and manager
    row. Great Lakes water and shoreline are excluded; Great Salt Lake water
    is excluded in Utah. Missing state values remain null.
    """
    if geometry_workers < 1:
        raise ValueError("Fisheries geometry_workers must be positive")
    plan = FisheriesPlan(zones, paths)
    with ThreadPoolExecutor(max_workers=geometry_workers) as pool:
        futures = {pool.submit(function, *args): key for key, function, args in plan.tasks()}
        for future in tqdm(as_completed(futures), total=plan.task_count,
                           desc="Measure fisheries zones", unit="county"):
            plan.accept(futures[future], future.result())
    return plan.finish()
