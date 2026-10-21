"""Assign coral point valuations once to the nearest county polygon."""

from math import fsum

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from tqdm import tqdm

from geometry_utils import repair_polygonal_geometry


def allocate_coral_values(counties, points):
    """Sum whole BC_USD11 point values in their nearest counties.

    Distance is measured to unsimplified county polygons, not centroids, in a
    WGS84 azimuthal-equidistant projection centered on each point. A point inside
    a county has distance zero. Distances within one micrometre are ties; choose
    the lowest GEOID and record all tied counties, never duplicating a value.

    Args:
        counties: GeoDataFrame with unique five-digit GEOID values and polygons.
        points: GeoDataFrame with unique OBJECTID values, BC_USD11, and Points.
            Both inputs must declare their coordinate reference systems.

    Returns:
        Tuple of raw county totals, county statuses, and one audit row per point.
        Counties without assigned points receive zero. A null source value makes
        its assigned county unknown, even if other assigned values are known.

    Raises:
        ValueError: Required identifiers, CRS, geometry, numeric values, or
            allocation conservation checks are invalid.
    """
    if counties.crs is None or points.crs is None:
        raise ValueError("Coral points and counties must declare a CRS")
    if len(counties) == 0 or len(points) == 0:
        raise ValueError("Coral allocation requires nonempty counties and points")
    if "GEOID" not in counties or not {"OBJECTID", "BC_USD11"} <= set(points):
        raise ValueError("Coral allocation requires GEOID, OBJECTID, and BC_USD11")
    ids = pd.Index(counties.GEOID.astype("string"), name="GEOID")
    keys = points.OBJECTID.astype("string").str.strip()
    if ids.isna().any() or not ids.str.fullmatch(r"\d{5}").all() or ids.has_duplicates:
        raise ValueError("County GEOID must be unique five-digit identifiers")
    if keys.isna().any() or keys.eq("").any() or keys.duplicated().any():
        raise ValueError("Coral OBJECTID must be nonempty and unique")
    values = pd.to_numeric(points.BC_USD11, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("Coral BC_USD11 contains infinite values")

    repaired = []
    for identifier, geometry in tqdm(zip(ids, counties.geometry), total=len(counties),
                                     desc="Validate coral county geometry", unit="county"):
        if (geometry is None or geometry.is_empty
                or geometry.geom_type not in {"Polygon", "MultiPolygon"}
                or not np.isfinite(shapely.get_coordinates(geometry)).all()):
            raise ValueError(f"Invalid county polygon: {identifier}")
        geometry = repair_polygonal_geometry(geometry)
        if geometry is None or geometry.area <= 0:
            raise ValueError(f"Invalid county polygon: {identifier}")
        repaired.append(geometry)
    with tqdm(total=2, desc="Project coral allocation inputs to WGS84", unit="layer") as progress:
        county_geometry = gpd.GeoSeries(repaired, crs=counties.crs).to_crs(4326)
        progress.update(1)
        point_geometry = points.geometry.to_crs(4326)
        progress.update(1)
    if not np.isfinite(shapely.get_coordinates(county_geometry.array)).all():
        raise ValueError("County projection produced nonfinite coordinates")

    contributions = {identifier: [] for identifier in ids}
    unknown = set()
    audit = []
    for key, value, point in tqdm(zip(keys, values, point_geometry), total=len(points),
                                  desc="Assign coral points to nearest counties", unit="point"):
        if (point is None or point.is_empty or point.geom_type != "Point"
                or not point.is_valid or not np.isfinite(shapely.get_coordinates(point)).all()
                or not -180 <= point.x <= 180 or not -90 <= point.y <= 90):
            raise ValueError(f"Invalid coral point geometry: {key}")
        local_crs = (f"+proj=aeqd +lat_0={point.y:.15g} +lon_0={point.x:.15g} "
                     "+datum=WGS84 +units=m +no_defs")
        projected = county_geometry.to_crs(local_crs)
        if not np.isfinite(shapely.get_coordinates(projected.array)).all():
            raise ValueError(f"Nonfinite county projection for coral point {key}")
        distances = shapely.distance(projected.array, Point(0, 0))
        if not np.isfinite(distances).all():
            raise ValueError(f"Nonfinite county distances for coral point {key}")
        nearest = float(distances.min())
        tied = sorted(ids[np.isclose(distances, nearest, rtol=0, atol=1e-6)].tolist())
        assigned = tied[0]
        if np.isnan(value):
            unknown.add(assigned)
        else:
            contributions[assigned].append(float(value))
        audit.append({"source_key": str(key), "input_value": value, "GEOID": assigned,
                      "distance_m": float(distances[ids.get_loc(assigned)]),
                      "tie_count": len(tied), "tied_counties": ";".join(tied),
                      "assignment_method": "point_centered_WGS84_AEQD_nearest_polygon"})

    audit = pd.DataFrame(audit)
    totals = pd.Series({identifier: fsum(parts) for identifier, parts in contributions.items()},
                       dtype=float).reindex(ids)
    known_total = fsum(values[np.isfinite(values)])
    if (audit.source_key.duplicated().any() or len(audit) != len(points)
            or not np.isclose(fsum(totals), known_total, rtol=1e-12, atol=1e-6)):
        raise ValueError("Coral nearest-county allocation failed source conservation")
    status = pd.Series("no_assigned_points", index=ids)
    status.loc[audit.GEOID.unique()] = "nearest_county"
    for identifier in unknown:
        totals.loc[identifier] = np.nan
        status.loc[identifier] = "incomplete_source"
    return totals, status, audit
