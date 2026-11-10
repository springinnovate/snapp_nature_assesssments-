"""Geographic scope and shared geographic input defaults."""

from __future__ import annotations

from math import inf, isfinite, nextafter
from pathlib import Path

import geopandas as gpd


PADUS_ANALYSIS_PATH = Path(
    "data/analysis_inputs/padus/PADUS4_1VectorAnalysis_PADUS_Only (1)"
    "/PADUS4_1VectorAnalysis_PADUS_Only.gdb"
)
PADUS_ANALYSIS_LAYER = "PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP"


# The contiguous 48 states and Washington, DC. Use an allowlist so Alaska,
# Hawaiʻi, and every territory are excluded even when input releases change.
CONUS_STATE_FIPS = frozenset(
    "01 04 05 06 08 09 10 11 12 13 16 17 18 19 20 21 22 23 24 25 26 27 28 29 "
    "30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split()
)


def county_bounds_in_crs(
    counties: gpd.GeoDataFrame, crs
) -> tuple[float, float, float, float]:
    """Return a source-space envelope around the actual selected county polygons.

    Args:
        counties: CONUS county polygons, including their in-scope waters.
        crs: Native CRS of the input layer to spatially subset.

    Returns:
        Inclusive min x, min y, max x, max y in the input's CRS.

    Raises:
        ValueError: The input lacks a CRS or a finite positive-area envelope.
    """
    if crs is None or counties.crs is None or counties.empty:
        raise ValueError("Source-space clipping requires county geometry and both CRSs")
    minx, miny, maxx, maxy = map(float, counties.to_crs(crs).total_bounds)
    if not all(map(isfinite, (minx, miny, maxx, maxy))) or minx >= maxx or miny >= maxy:
        raise ValueError("CONUS counties have an invalid source-space extent")
    return (nextafter(minx, -inf), nextafter(miny, -inf),
            nextafter(maxx, inf), nextafter(maxy, inf))
