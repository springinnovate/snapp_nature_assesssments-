"""Build county and footprint zones from equal-area polygon geometries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import fsum, isfinite

import shapely
from shapely import Geometry

from geometry_utils import repair_polygonal_geometry

ZONE_PRODUCTS = (
    "counties",
    "public_lands_by_county",
    "public_lands_by_county_and_manager_type",
    "conservation_lands_by_county",
    "conservation_lands_by_county_and_manager_type",
    "bbb_lands_by_county",
)
_M2_PER_HA = 10_000.0
_ABS_AREA_TOL_M2 = 0.01
_REL_AREA_TOL = 1e-10
# Census ATOTAL and ALAND are published integer areas. The projected geometry
# can differ slightly because it represents each segment as a straight line.
MAX_LAND_MASK_ERROR_PCT = 0.01


def _clip_union(candidates: Iterable[Geometry], mask: Geometry) -> Geometry | None:
    """Clip candidates, then dissolve their positive-area polygonal pieces.

    Args:
        candidates: Polygonal geometries in the same CRS as the mask.
        mask: Geometry to intersect with each candidate.

    Returns:
        A repaired multipolygon, or None when only empty/zero-area pieces
        intersect the mask.

    Raises:
        ValueError: A positive-area dissolve cannot be repaired.
    """
    parts = shapely.intersection(list(candidates), mask)
    polygons = parts[shapely.area(parts) > 0]
    if len(polygons) == 0:
        return None
    dissolved = repair_polygonal_geometry(shapely.union_all(polygons))
    if dissolved is None:
        raise ValueError("A positive-area dissolve did not yield a valid polygon")
    return dissolved


def build_county_zones(
    geoid: str,
    county_geom: Geometry,
    county_land_area_m2: float,
    padus_candidates: Iterable[tuple[Geometry, str, bool, bool]],
    bbb_candidates: Iterable[Geometry],
    *,
    county_land_mask: Geometry,
) -> dict[str, list[dict[str, object]]]:
    """Build all six zone products and county area checks for one county.

    County and footprint geometries retain water for spatial analysis. Only
    land hectares and shares use the intersection with Census land faces.
    Shares use the county's ALAND, not its full polygon area.
    Per-manager zones retain their original labels; overlap is measured as
    the excess of summed areas over the union area, not assigned by precedence.

    Args:
        geoid: County identifier.
        county_geom: Valid county polygon in EPSG:6933.
        county_land_area_m2: Census ALAND, in square metres.
        padus_candidates: Candidate (geometry, manager type, public flag,
            conservation flag) records; normalized manager labels must be
            nonempty, including any explicit unknown category.
        bbb_candidates: Candidate BBB polygons for this county.
        county_land_mask: Dissolved Census land faces for this GEOID in EPSG:6933.

    Returns:
        Rows keyed by product name. The counties product has exactly one row.
        A footprint product without positive county overlap has no rows.

    Raises:
        ValueError: County ALAND or its land mask is invalid, a participating
            manager label is missing, or a geometry area is inconsistent.
    """
    county_area = float(county_geom.area)
    if not isfinite(county_area) or county_area <= 0:
        raise ValueError(f"County {geoid} does not have positive polygon area")
    if not isfinite(county_land_area_m2) or county_land_area_m2 <= 0:
        raise ValueError(f"County {geoid} must have finite positive ALAND: {county_land_area_m2}")
    county_tol = max(_ABS_AREA_TOL_M2, county_area * _REL_AREA_TOL)
    mask_area = float(county_land_mask.area)
    if not isfinite(mask_area) or mask_area <= 0 or not county_land_mask.is_valid:
        raise ValueError(f"County {geoid} must have a valid positive-area land mask")
    mask_tol = max(_ABS_AREA_TOL_M2, mask_area * _REL_AREA_TOL)
    if not shapely.covers(county_geom, county_land_mask):
        outside = shapely.difference(county_land_mask, county_geom).area
        if outside > mask_tol:
            raise ValueError(f"County {geoid} land mask extends outside its county by {outside:.6f} m²")
    if abs(mask_area - county_land_area_m2) > county_land_area_m2 * MAX_LAND_MASK_ERROR_PCT / 100:
        raise ValueError(
            f"County {geoid} land mask area {mask_area:.6f} m² differs from ALAND "
            f"{county_land_area_m2:.6f} m² by more than {MAX_LAND_MASK_ERROR_PCT}%"
        )
    shapely.prepare(county_land_mask)

    county_row = {
        "GEOID": geoid,
        "geometry": county_geom,
        "area_ha_county_land": county_land_area_m2 / _M2_PER_HA,
    }
    results = {product: [] for product in ZONE_PRODUCTS}
    results["counties"].append(county_row)
    pieces = {"public_lands": defaultdict(list), "conservation_lands": defaultdict(list)}
    candidates = [
        (geom, manager, public, conservation)
        for geom, manager, public, conservation in padus_candidates
        if public or conservation
    ]
    # Clip once even when one PAD-US feature contributes to both products.
    clipped = shapely.intersection([row[0] for row in candidates], county_geom)
    for geom, (_, manager, public, conservation) in zip(clipped, candidates):
        if geom.area <= 0:
            continue
        if not isinstance(manager, str) or not manager.strip():
            raise ValueError(f"County {geoid} has a PAD-US feature without a normalized manager type")
        if public:
            pieces["public_lands"][manager].append(geom)
        if conservation:
            pieces["conservation_lands"][manager].append(geom)

    def footprint_row(geometry: Geometry, product: str) -> dict[str, object]:
        """Compute a land-only measure and share for a retained footprint.

        Args:
            geometry: County-clipped zone geometry, including any water.
            product: Zone product label for diagnostic errors.

        Returns:
            Attributes and geometry for the output zone.

        Raises:
            ValueError: A footprint's total or land area is inconsistent.
        """
        area = float(geometry.area)
        if not isfinite(area) or not 0 < area <= county_area + county_tol:
            raise ValueError(f"County {geoid} has inconsistent {product} footprint area: {area}")
        land_area = area if shapely.covers(county_land_mask, geometry) else geometry.intersection(county_land_mask).area
        if not isfinite(land_area) or not 0 <= land_area <= min(area + county_tol, mask_area + mask_tol):
            raise ValueError(f"County {geoid} has inconsistent {product} land area: {land_area}")
        # Only the share is bounded, and only after checking the entire mask's
        # agreement with ALAND. Preserve the measured hectares for auditing.
        return {
            "GEOID": geoid,
            "geometry": geometry,
            "area_ha_land": land_area / _M2_PER_HA,
            "county_land_share": min(1.0, land_area / county_land_area_m2),
        }

    for footprint, managers in pieces.items():
        county_product = f"{footprint}_by_county"
        manager_product = f"{county_product}_and_manager_type"
        manager_geometries = []
        for manager, group in sorted(managers.items()):
            geometry = repair_polygonal_geometry(shapely.union_all(group))
            if geometry is None:
                raise ValueError(f"County {geoid} could not dissolve {footprint} manager {manager}")
            manager_geometries.append(geometry)
            row = footprint_row(geometry, manager_product)
            row["Mang_Type"] = manager
            results[manager_product].append(row)

        zone_area_ha = 0.0
        share = 0.0
        overlap_m2 = 0.0
        if manager_geometries:
            geometry = repair_polygonal_geometry(shapely.union_all(manager_geometries))
            if geometry is None:
                raise ValueError(f"County {geoid} could not dissolve {footprint}")
            row = footprint_row(geometry, county_product)
            results[county_product].append(row)
            zone_area_ha = row["area_ha_land"]
            share = row["county_land_share"]
            overlap_m2 = fsum(float(geom.area) for geom in manager_geometries) - float(geometry.area)
            overlap_tol = max(_ABS_AREA_TOL_M2, geometry.area * _REL_AREA_TOL)
            if overlap_m2 < -overlap_tol:
                raise ValueError(f"County {geoid} {footprint} manager areas do not cover the union")
            if abs(overlap_m2) <= overlap_tol:
                overlap_m2 = 0.0

        label = footprint.removesuffix("_lands")
        county_row[f"area_ha_{footprint}"] = zone_area_ha
        county_row[f"pct_{footprint}"] = 100.0 * share
        county_row[f"area_ha_{label}_manager_overlap"] = overlap_m2 / _M2_PER_HA

    bbb_geometry = _clip_union(bbb_candidates, county_geom)
    if bbb_geometry is not None:
        results["bbb_lands_by_county"].append(footprint_row(bbb_geometry, "BBB lands"))
    bbb_rows = results["bbb_lands_by_county"]
    county_row["area_ha_bbb_lands"] = bbb_rows[0]["area_ha_land"] if bbb_rows else 0.0
    county_row["pct_bbb_lands"] = 100.0 * bbb_rows[0]["county_land_share"] if bbb_rows else 0.0
    return results
