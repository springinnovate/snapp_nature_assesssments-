"""Shared geometry normalization and repair helpers."""

from __future__ import annotations

import shapely
from shapely.geometry import MultiPolygon


def polygonal_multipolygon(geom):
    """Extract polygonal content as a multipolygon.

    Args:
        geom: Shapely geometry to normalize.

    Returns:
        A MultiPolygon, or None if no polygonal geometry remains.
    """
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        parts = [geom]
    elif geom.geom_type == "MultiPolygon":
        parts = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            normalized = polygonal_multipolygon(part)
            if normalized is not None:
                parts.extend(normalized.geoms)
    else:
        return None

    parts = [part for part in parts if not part.is_empty and part.area > 0]
    if not parts:
        return None

    return MultiPolygon(parts)


def repair_polygonal_geometry(geom):
    """Repair and normalize polygon geometry.

    Args:
        geom: Shapely geometry produced by clipping or overlay.

    Returns:
        A valid MultiPolygon, or None if repair cannot produce polygonal output.
    """
    geom = polygonal_multipolygon(geom)
    if geom is None:
        return None

    if geom.is_valid:
        return geom

    repaired = polygonal_multipolygon(shapely.make_valid(geom))
    if repaired is not None and repaired.is_valid:
        return repaired

    if repaired is not None:
        repaired = polygonal_multipolygon(repaired.buffer(0))
        if repaired is not None and repaired.is_valid:
            return repaired

    repaired = polygonal_multipolygon(geom.buffer(0))
    if repaired is not None and repaired.is_valid:
        return repaired

    return None
