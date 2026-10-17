"""Allocate study-unit mangrove benefits over county-covered area only."""

from concurrent.futures import ThreadPoolExecutor
from math import fsum
from numbers import Integral

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

from geometry_utils import polygonal_multipolygon


AUDIT_COLUMNS = [
    "source_key", "input_value", "source_area_m2", "intersected_area_m2",
    "intersecting_counties", "allocated_fraction", "allocated_value",
    "unallocated_value", "status", "reason",
]


def _project_polygons(frame, label, chunk_size, require_area=False):
    """Repair polygonal geometry and project bounded batches to equal-area metres.

    Args:
        frame: GeoDataFrame with a declared CRS.
        label: Dataset label shown in progress updates and validation errors.
        chunk_size: Maximum number of features projected in one batch.
        require_area: Whether unusable geometry raises instead of returning None.

    Returns:
        List of valid positive-area EPSG:6933 geometries, with None for unusable
        source geometry when require_area is false.

    Raises:
        ValueError: A required county geometry cannot provide a finite area.
    """
    result = []
    with tqdm(total=len(frame), desc=f"Repair/project {label}", unit="feature") as progress:
        for start in range(0, len(frame), chunk_size):
            normalized = []
            for geometry in frame.geometry.iloc[start:start + chunk_size]:
                if geometry is not None and not geometry.is_empty:
                    try:
                        if not np.isfinite(shapely.get_coordinates(geometry)).all():
                            geometry = None
                        else:
                            if not geometry.is_valid:
                                geometry = shapely.make_valid(geometry)
                            geometry = polygonal_multipolygon(geometry)
                    except shapely.GEOSException:
                        geometry = None
                else:
                    geometry = None
                normalized.append(geometry)
            projected = gpd.GeoSeries(normalized, crs=frame.crs).to_crs("EPSG:6933")
            for offset, geometry in enumerate(projected):
                try:
                    if geometry is not None and not geometry.is_empty:
                        if not np.isfinite(shapely.get_coordinates(geometry)).all():
                            geometry = None
                        else:
                            if not geometry.is_valid:
                                geometry = shapely.make_valid(geometry)
                            geometry = polygonal_multipolygon(geometry)
                            if geometry is not None and (
                                    not np.isfinite(geometry.area) or geometry.area <= 0):
                                geometry = None
                except shapely.GEOSException:
                    geometry = None
                if geometry is None and require_area:
                    identifier = frame["GEOID"].iloc[start + offset]
                    raise ValueError(f"{label} {identifier!r} has no valid finite positive-area geometry")
                result.append(geometry)
            progress.update(len(projected))
    return result


def _allocate_chunk(start, stop, source_geometry, source_values, source_keys,
                    county_geometry, county_tree, metadata):
    """Intersect one ordered source batch and retain allocations and audit rows.

    Args:
        start: Inclusive first source position in the batch.
        stop: Exclusive final source position in the batch.
        source_geometry: Repaired EPSG:6933 study-unit geometries.
        source_values: Numeric benefit values, including NaN for unknown values.
        source_keys: Unique canonical source identifiers.
        county_geometry: Repaired EPSG:6933 county geometries.
        county_tree: Spatial index containing county_geometry in positional order.
        metadata: Optional ISO3 and Country columns indexed by source position.

    Returns:
        Tuple of ordered source audit rows, known (county position, value)
        contributions, and county positions affected by unknown source values.

    Raises:
        ValueError: An intersection produces a nonfinite area or allocation.
    """
    rows, contributions, missing_counties = [], [], set()
    for position in range(start, stop):
        geometry = source_geometry[position]
        value = source_values[position]
        row = {
            "source_key": source_keys[position], "input_value": value,
            "source_area_m2": 0.0 if geometry is None else geometry.area,
            "intersected_area_m2": 0.0, "intersecting_counties": 0,
            "allocated_fraction": 0.0, "allocated_value": 0.0,
            "unallocated_value": value, "status": "unallocated",
            "reason": "invalid_or_empty_geometry" if geometry is None else "no_positive_county_intersection",
        }
        for field in metadata:
            row[field] = metadata[field][position]
        if geometry is not None:
            candidates = np.sort(county_tree.query(geometry, predicate="intersects"))
            if len(candidates):
                intersections = shapely.intersection(geometry, county_geometry[candidates])
                areas = np.asarray(shapely.area(intersections), dtype=float)
                if not np.isfinite(areas).all():
                    raise ValueError(f"Nonfinite intersection area for source {row['source_key']!r}")
                positive = areas > 0
                candidates, areas = candidates[positive], areas[positive]
                if len(candidates):
                    denominator = fsum(areas)
                    if not np.isfinite(denominator):
                        raise ValueError(f"Nonfinite total intersection area for source {row['source_key']!r}")
                    row.update(intersected_area_m2=denominator,
                               intersecting_counties=len(candidates), allocated_fraction=1.0)
                    if np.isnan(value):
                        missing_counties.update(candidates.tolist())
                    else:
                        allocations = value * (areas / denominator)
                        if not np.isfinite(allocations).all():
                            raise ValueError(f"Nonfinite allocation for source {row['source_key']!r}")
                        # Put floating-point residual in the largest county share, so
                        # the entire source value is distributed, not an ocean fraction.
                        allocations[np.argmax(areas)] += value - fsum(allocations)
                        contributions.extend(zip(candidates.tolist(), allocations.tolist()))
                        row.update(allocated_value=fsum(allocations), unallocated_value=0.0,
                                   status="allocated", reason="normalized_county_intersection_area")
        if np.isnan(value):
            row.update(allocated_value=np.nan, unallocated_value=np.nan,
                       status="missing_value", reason="unknown_source_value")
        rows.append(row)
    return rows, contributions, missing_counties


def allocate_mangrove_values(counties, polygons, value_field="Ben_Stock_2020",
                             id_field="source_id", workers=1, chunk_size=2000):
    """Distribute each benefit total among its positive-area county intersections.

    A source's full value is divided in proportion to its intersections with
    counties; ocean and any other area outside all counties are excluded from
    the denominator. County values sum the shares of every intersecting source.
    A null source value makes all positively intersecting counties unknown,
    rather than publishing their partial known totals. Counties with no source
    intersections receive zero. Negative finite inputs are preserved here;
    this function does not apply a valuation adjustment or clipping policy.

    Args:
        counties: GeoDataFrame with unique, nonmissing GEOID values and polygon
            geometry. Output Series retain these GEOID values and their order.
        polygons: GeoDataFrame containing study-unit geometry, identifiers, and
            unadjusted monetary values. Both frames must declare their CRS.
        value_field: Source field containing each study unit's total valuation.
        id_field: Unique source identifier retained as source_key in the audit.
        workers: Positive number of threads for independent source batches.
        chunk_size: Positive maximum number of source features in each batch.

    Returns:
        Tuple of county values Series, county coverage-status Series, and a
        per-source audit DataFrame. Both Series are indexed by GEOID. Status is
        allocated, incomplete_source, or no_intersection. The audit includes
        all sources, including unknown or unallocated values; allocated_fraction
        is the fraction of a source's value assigned geographically (one or zero),
        not its fraction of land area covered by counties. ISO3 and Country are
        included when available. Known source allocations remain in the audit
        even when a county total is withheld because another source is unknown.

    Raises:
        ValueError: Identifiers, required fields, CRS, worker/chunk settings,
            county geometry, numeric values, or computed areas are invalid.
    """
    for name, value in (("workers", workers), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for label, frame, identifier in (("counties", counties, "GEOID"),
                                      ("study units", polygons, id_field)):
        if not isinstance(frame, gpd.GeoDataFrame) or frame.crs is None:
            raise ValueError(f"{label} must be a GeoDataFrame with a declared CRS")
        if identifier not in frame.columns:
            raise ValueError(f"{label} is missing identifier field {identifier!r}")
        identifiers = frame[identifier]
        if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
            raise ValueError(f"{label} contains missing identifiers in {identifier!r}")
        if identifiers.astype(str).str.strip().duplicated().any():
            raise ValueError(f"{label} contains duplicate identifiers in {identifier!r}")
    if value_field not in polygons:
        raise ValueError(f"study units is missing value field {value_field!r}")
    try:
        source_values = pd.to_numeric(polygons[value_field], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_field!r} must contain numeric values or nulls") from error
    if np.isinf(source_values).any():
        raise ValueError(f"{value_field!r} contains nonfinite values other than nulls")
    county_index = pd.Index(counties["GEOID"], name="GEOID")
    source_keys = polygons[id_field].astype(str).str.strip().tolist()
    metadata = {field: polygons[field].tolist() for field in ("ISO3", "Country") if field in polygons}
    if polygons.empty:
        return (pd.Series(0.0, index=county_index, name="mangroves"),
                pd.Series("no_intersection", index=county_index, name="mangroves_status"),
                pd.DataFrame(columns=AUDIT_COLUMNS + list(metadata)))
    county_geometry = np.asarray(_project_polygons(counties, "counties", chunk_size, require_area=True), dtype=object)
    source_geometry = _project_polygons(polygons, "mangrove study units", chunk_size)
    county_tree = shapely.STRtree(county_geometry)
    chunks = [(start, min(start + chunk_size, len(polygons))) for start in range(0, len(polygons), chunk_size)]
    arguments = [(start, stop, source_geometry, source_values, source_keys,
                  county_geometry, county_tree, metadata) for start, stop in chunks]
    by_county = [[] for _ in county_index]
    missing_counties, audit_rows = set(), []
    # Consume futures in source order, so thread count cannot change summation
    # order or audit row order. Shapely releases the GIL during intersections.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_allocate_chunk, *argument) for argument in arguments]
        with tqdm(total=len(polygons), desc="Allocate mangrove study units", unit="feature") as progress:
            for future in futures:
                rows, contributions, unknown = future.result()
                audit_rows.extend(rows)
                for county_position, contribution in contributions:
                    by_county[county_position].append(contribution)
                missing_counties.update(unknown)
                progress.update(len(rows))
    values = pd.Series(0.0, index=county_index, name="mangroves")
    status = pd.Series("no_intersection", index=county_index, name="mangroves_status")
    for position, contributions in enumerate(tqdm(by_county, desc="Sum mangrove county shares", unit="county")):
        if position in missing_counties:
            values.iloc[position], status.iloc[position] = np.nan, "incomplete_source"
        elif contributions:
            values.iloc[position], status.iloc[position] = fsum(contributions), "allocated"
    return values, status, pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS + list(metadata))
