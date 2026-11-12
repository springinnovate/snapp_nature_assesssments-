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
from assessment.mangrove_county_allocation import (
    _allocate_chunk, _finish_allocation, _prepare_allocation,
)


AREA_CRS = "EPSG:6933"
MANGROVE_CHUNK_SIZE = 32
FOOTPRINT_CHUNK_SIZE = 128


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


def _coral_source_values(reference, points):
    """Keep worker output quiet while the parent reports completed jobs."""
    return allocate_coral_values(reference, points, show_progress=False)


def _mangrove_source_chunk(source_shapes, source_values, source_keys,
                           county_shapes, county_positions, metadata):
    """Allocate a bounded source batch using only its candidate counties."""
    rows, contributions, unknown = _allocate_chunk(
        0, len(source_shapes), source_shapes, source_values, source_keys,
        county_shapes, shapely.STRtree(county_shapes), metadata,
    )
    return (rows, [(int(county_positions[position]), value) for position, value in contributions],
            {int(county_positions[position]) for position in unknown})


def _coastal_footprint_chunk(shapes, geoids, points, coral_audit, sources, areas, values):
    """Measure a bounded footprint batch without creating another executor."""
    return np.column_stack((
        _footprint_corals(shapes, geoids, points, coral_audit),
        _footprint_mangroves(shapes, shapely.STRtree(sources), areas, values),
    ))


class CoastalPlan:
    """Schedule source allocations and dependent footprints on the caller's pool.

    The parent prepares spatial indexes and sends only candidate geometries in
    each bounded task. Source results are assembled in input order even when
    workers complete out of order. All task functions are module-level and no
    task creates an executor.
    """

    def __init__(self, zones, paths, *, coral_reference_counties):
        if set(zones) != set(ZONE_PRODUCTS):
            raise ValueError("Coastal valuation requires all six prepared zone products")
        self.zones = zones
        counties = zones["counties"]
        county_ids = counties.GEOID.astype("string")
        if (not county_ids.str.fullmatch("[0-9]{5}").fillna(False).all()
                or county_ids.duplicated().any()
                or not county_ids.str[:2].isin(CONUS_STATE_FIPS).all()):
            raise ValueError("Coastal valuation requires unique CONUS county GEOIDs")
        if ("GEOID" not in coral_reference_counties
                or not county_ids.isin(coral_reference_counties.GEOID.astype("string")).all()):
            raise ValueError("The coral reference must contain every assessment county GEOID")
        for product, zone in zones.items():
            if not zone.GEOID.astype("string").isin(county_ids).all():
                raise ValueError(f"{product} contains an unknown county GEOID")
        self.reference = coral_reference_counties[["GEOID", coral_reference_counties.geometry.name]]
        with tqdm(total=2, desc="Read coastal valuation inputs", unit="layer") as progress:
            self.points = pyogrio.read_dataframe(
                paths["corals"], layer="aeb_coral", columns=["OBJECTID", "BC_USD11"])
            progress.update(1)
            layer = "UCSC_CWON_studyunits"
            source_crs = pyogrio.read_info(paths["mangroves"], layer=layer)["crs"]
            if source_crs is None:
                raise ValueError("Mangrove study units must declare a CRS")
            bounds = tuple(counties.geometry.to_crs(source_crs).total_bounds)
            # Preserve offshore source parts; county intersections set support.
            study_units = pyogrio.read_dataframe(
                paths["mangroves"], layer=layer, bbox=bounds, where="ISO3 = 'USA'",
                columns=["ISO3", "Ben_Stock_2020"], fid_as_index=True,
            ).rename_axis("source_id").reset_index()
            progress.update(1)
        self.prepared = _prepare_allocation(counties, study_units)
        self.source_starts = list(range(0, len(study_units), MANGROVE_CHUNK_SIZE))
        self.source_results = {}
        self.coral_result = None
        self.values = {product: pd.DataFrame(0.0, index=zone.index, columns=["corals", "mangroves"])
                       for product, zone in zones.items()}
        self.footprints_started = False
        self.footprints_generated = False
        self.remaining_footprints = set()

    def tasks(self):
        """Yield independent coral and mangrove source-allocation jobs."""
        yield ("corals", 0), _coral_source_values, (self.reference, self.points)
        _, shapes, values, keys, county_shapes, metadata = self.prepared
        tree = shapely.STRtree(county_shapes)
        for start in self.source_starts:
            stop = start + MANGROVE_CHUNK_SIZE
            subset = shapes[start:stop]
            candidates = np.unique(tree.query(subset, predicate="intersects")[1])
            yield ("mangroves", start), _mangrove_source_chunk, (
                subset, values[start:stop], keys[start:stop], county_shapes[candidates],
                candidates, {field: items[start:stop] for field, items in metadata.items()},
            )

    def accept(self, key, result):
        """Store a completed job and release footprints when sources are ready."""
        kind, position = key
        if kind == "corals":
            if self.coral_result is not None:
                raise ValueError(f"Duplicate coastal task result: {key}")
            self.coral_result = result
        elif kind == "mangroves":
            if position not in self.source_starts or position in self.source_results:
                raise ValueError(f"Unexpected or duplicate coastal task result: {key}")
            self.source_results[position] = result
        else:
            if key not in self.remaining_footprints:
                raise ValueError(f"Unexpected or duplicate coastal task result: {key}")
            self.values[kind].iloc[position:position + len(result)] = result
            self.remaining_footprints.remove(key)
        if (not self.footprints_started and self.coral_result is not None
                and len(self.source_results) == len(self.source_starts)):
            self.footprints_started = True
            return self._footprint_tasks()
        return ()

    def _footprint_tasks(self):
        """Use fixed county support to schedule bounded land-footprint batches."""
        county_index, source_shapes, _, source_keys, _, metadata = self.prepared
        mangrove_counties, _, audit = _finish_allocation(
            county_index, metadata, [self.source_results[start] for start in self.source_starts])
        coral_counties, _, coral_audit = self.coral_result
        counties = self.zones["counties"]
        self.values["counties"]["corals"] = coral_counties.reindex(counties.GEOID).to_numpy()
        self.values["counties"]["mangroves"] = mangrove_counties.reindex(counties.GEOID).to_numpy()
        coral_audit = coral_audit.loc[coral_audit.GEOID.isin(counties.GEOID)]
        point_positions = pd.Index(self.points.OBJECTID.astype("string").str.strip()).get_indexer(
            coral_audit.source_key)
        points = self.points.geometry.iloc[point_positions].to_crs(AREA_CRS).to_numpy()
        support = audit.set_index("source_key").reindex(source_keys)
        supported = support.intersected_area_m2.gt(0).to_numpy()
        source_shapes = source_shapes[supported]
        areas = support.intersected_area_m2.to_numpy(dtype=float)[supported]
        values = support.input_value.to_numpy(dtype=float)[supported]
        tree = shapely.STRtree(source_shapes)
        for product in ZONE_PRODUCTS[1:]:
            zone = self.zones[product]
            for start in range(0, len(zone), FOOTPRINT_CHUNK_SIZE):
                chunk = zone.iloc[start:start + FOOTPRINT_CHUNK_SIZE]
                shapes = chunk.geometry.to_crs(AREA_CRS).to_numpy()
                candidates = np.unique(tree.query(shapes, predicate="intersects")[1])
                key = (product, start)
                self.remaining_footprints.add(key)
                yield key, _coastal_footprint_chunk, (
                    shapes, chunk.GEOID.astype("string").to_numpy(), points, coral_audit,
                    source_shapes[candidates], areas[candidates], values[candidates],
                )
        self.footprints_generated = True

    def finish(self):
        """Return values only after all source and footprint jobs complete."""
        if not self.footprints_generated or self.remaining_footprints:
            raise ValueError("Coastal valuation has unfinished tasks")
        return self.values


def value_coastal(
    zones: Mapping[str, gpd.GeoDataFrame], paths: Mapping[str, Path], geometry_workers: int,
    *, coral_reference_counties: gpd.GeoDataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate coastal values serially; the assessment schedules CoastalPlan.

    Keep the direct-call API for standalone use. Assessment workers execute the
    same tasks through its shared process pool. Values use whole-county source
    support and preserve input row indexes, overlapping products and nulls.
    """
    if geometry_workers < 1:
        raise ValueError("Coastal geometry_workers must be positive")
    plan = CoastalPlan(zones, paths, coral_reference_counties=coral_reference_counties)
    pending = list(plan.tasks())
    while pending:
        key, function, arguments = pending.pop(0)
        pending.extend(plan.accept(key, function(*arguments)))
    return plan.finish()
