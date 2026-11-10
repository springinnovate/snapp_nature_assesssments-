"""Restrict county-based coral and mangrove allocations to CONUS zones."""

from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from tqdm import tqdm

from assessment.assessment_geography import CONUS_STATE_FIPS
from assessment.assessment_zone_geometry import ZONE_PRODUCTS
from assessment.coral_county_allocation import allocate_coral_values
from assessment.mangrove_county_allocation import _project_polygons, allocate_mangrove_values


AREA_CRS = "EPSG:6933"


def _footprint_corals(
    shapes: np.ndarray,
    geoids: np.ndarray,
    points: np.ndarray,
    assignments: pd.DataFrame,
) -> np.ndarray:
    """Sum points covered by footprints within their assigned counties.

    Args:
        shapes: County-clipped footprint geometries in the point CRS.
        geoids: County identifiers aligned to the footprint geometries.
        points: Coral point geometries in assignment order.
        assignments: Coral audit rows with assigned GEOID and raw input_value.

    Returns:
        Raw values in footprint order; an included null point makes its
        footprint unknown, and footprints with no included points receive zero.
    """
    totals = np.zeros(len(shapes), dtype=float)
    for point, assignment in zip(points, assignments.itertuples(index=False)):
        positions = np.flatnonzero(geoids == assignment.GEOID)
        included = positions[shapely.intersects(shapes[positions], point)]
        totals[included] += assignment.input_value
    return totals


def _footprint_mangroves(
    shapes: np.ndarray,
    tree: shapely.STRtree,
    source_areas: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Value footprint intersections using fixed whole-county source support.

    Args:
        shapes: County-clipped footprint polygons in EPSG:6933.
        tree: Index of repaired study units with positive whole-county support.
        source_areas: Whole-county intersection area sums aligned to the tree.
        source_values: Raw study-unit values aligned to the tree.

    Returns:
        Raw values in footprint order, with zero for no overlap and null for
        any positive-area overlap with an unknown source.

    Raises:
        ValueError: An intersection produces nonfinite area or allocation.
    """
    footprint_ids, source_ids = tree.query(shapes, predicate="intersects")
    areas = shapely.area(shapely.intersection(shapes[footprint_ids], tree.geometries[source_ids]))
    if not np.isfinite(areas).all():
        raise ValueError("Nonfinite mangrove footprint intersection area")
    positive = areas > 0
    source_ids = source_ids[positive]
    contributions = source_values[source_ids] * (areas[positive] / source_areas[source_ids])
    totals = np.bincount(footprint_ids[positive], weights=contributions, minlength=len(shapes))
    if np.isinf(totals).any():
        raise ValueError("Nonfinite mangrove footprint allocation")
    return totals.astype(float, copy=False)


def value_coastal(
    zones: Mapping[str, gpd.GeoDataFrame],
    paths: Mapping[str, Path],
    geometry_workers: int,
    *,
    coral_reference_counties: gpd.GeoDataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate unadjusted coastal values for the six CONUS zone products.

    Assign coral points to the original, unfiltered counties, then retain
    assignments to assessment counties. This keeps offshore CONUS points
    without redistributing points assigned to out-of-scope counties. A
    footprint receives only assigned points it intersects. For mangroves,
    use USA study units and a denominator fixed to their CONUS county
    intersections. Products and manager rows can overlap and their values
    are not additive.

    Args:
        zones: The six prepared products. County GEOIDs are unique in the
            counties product; all other rows have valid county-clipped polygons
            and may repeat GEOID for different managers.
        paths: Configured source files keyed by corals and mangroves.
        geometry_workers: Worker count for the whole-county mangrove allocation.
        coral_reference_counties: Original, unfiltered county polygons with
            unique GEOIDs, including counties outside the assessment scope.

    Returns:
        One DataFrame per product, in that product's input index order, with
        pre-factor corals and mangroves columns. Zero indicates no contributing
        source; a contributing source with a missing value propagates null.

    Raises:
        ValueError: A zone product or county identifier is inconsistent.
    """
    if set(zones) != set(ZONE_PRODUCTS):
        raise ValueError("Coastal valuation requires all six prepared zone products")
    counties = zones["counties"]
    county_ids = counties.GEOID.astype("string")
    if (not county_ids.str.fullmatch("[0-9]{5}").fillna(False).all()
            or county_ids.duplicated().any()
            or not county_ids.str[:2].isin(CONUS_STATE_FIPS).all()):
        raise ValueError("Coastal valuation requires unique CONUS county GEOIDs")
    if ("GEOID" not in coral_reference_counties
            or not county_ids.isin(coral_reference_counties.GEOID.astype("string")).all()):
        raise ValueError("The coral reference must contain every assessment county GEOID")
    with tqdm(total=2, desc="Read coastal valuation inputs", unit="layer") as progress:
        points = pyogrio.read_dataframe(
            paths["corals"], layer="aeb_coral", columns=["OBJECTID", "BC_USD11"]
        )
        progress.update(1)
        mangrove_layer = "UCSC_CWON_studyunits"
        mangrove_crs = pyogrio.read_info(paths["mangroves"], layer=mangrove_layer)["crs"]
        if mangrove_crs is None:
            raise ValueError("Mangrove study units must declare a CRS")
        bounds = tuple(counties.geometry.to_crs(mangrove_crs).total_bounds)
        # The bounding box retains offshore parts of candidate study units.
        # The county allocator checks their actual intersection support.
        study_units = pyogrio.read_dataframe(
            paths["mangroves"], layer=mangrove_layer, bbox=bounds, where="ISO3 = 'USA'",
            columns=["ISO3", "Ben_Stock_2020"], fid_as_index=True,
        ).rename_axis("source_id").reset_index()
        progress.update(1)

    coral_counties, _, coral_audit = allocate_coral_values(coral_reference_counties, points)
    coral_audit = coral_audit.loc[coral_audit.GEOID.isin(county_ids)]
    mangrove_counties, _, mangrove_audit = allocate_mangrove_values(
        counties, study_units, workers=geometry_workers
    )
    values = {
        "counties": pd.DataFrame({
            "corals": coral_counties.reindex(county_ids).to_numpy(),
            "mangroves": mangrove_counties.reindex(counties.GEOID).to_numpy(),
        }, index=counties.index),
    }

    point_positions = pd.Index(points.OBJECTID.astype("string").str.strip()).get_indexer(
        coral_audit.source_key
    )
    point_shapes = points.geometry.iloc[point_positions].to_crs(AREA_CRS).to_numpy()
    support = mangrove_audit.set_index("source_key").reindex(study_units.source_id.astype(str))
    supported = support.intersected_area_m2.gt(0).to_numpy()
    # Use the county allocator's projection/repair for the numerators too.
    study_shapes = _project_polygons(study_units.loc[supported], "supported mangroves", 2000)
    study_tree = shapely.STRtree(study_shapes)
    areas = support.intersected_area_m2.to_numpy(dtype=float)[supported]
    totals = support.input_value.to_numpy(dtype=float)[supported]

    for product in tqdm(ZONE_PRODUCTS[1:], desc="Value coastal footprints", unit="product"):
        zone = zones[product]
        geoids = zone.GEOID.astype("string")
        if not geoids.isin(county_ids).all():
            raise ValueError(f"{product} contains an unknown county GEOID")
        shapes = zone.geometry.to_crs(AREA_CRS).to_numpy()
        values[product] = pd.DataFrame({
            "corals": _footprint_corals(shapes, geoids.to_numpy(), point_shapes, coral_audit),
            "mangroves": _footprint_mangroves(shapes, study_tree, areas, totals),
        }, index=zone.index)
    return values
