"""Keep individual PAD-US features and their attributes when cutting by county."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from tqdm import tqdm

from assessment.assessment_geography import CONUS_STATE_FIPS, county_bounds_in_crs
from assessment.assessment_zone_geometry import build_county_zones
from assessment.geometry_utils import repair_polygonal_geometry
from assessment.prepare_assessment_zones import AREA_CRS, project_polygons, read_polygons


ALL_PAD_PRODUCT = "all_PAD_lands_by_county_and_manager_type"
_GENERATED_FIELDS = ("feature_id", "GEOID", "area_ha", "area_ha_land", "county_land_share")


def _county_ids(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Validate county keys and retain the contiguous states and DC."""
    ids = frame["GEOID"].astype("string")
    if not ids.str.fullmatch(r"[0-9]{5}").fillna(False).all() or ids.duplicated().any():
        raise ValueError(f"{label} must have unique five-digit GEOID strings")
    return frame.assign(GEOID=ids).loc[ids.str[:2].isin(CONUS_STATE_FIPS)].sort_values("GEOID")


def _read_padus(
    path: Path, layer: str, counties: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Read all source fields and stable source IDs, clipping remote overhangs."""
    info = pyogrio.read_info(path, layer=layer)
    if not info["crs"] or info["geometry_type"] not in {"Polygon", "MultiPolygon"}:
        raise ValueError("The all-PAD source must contain polygons with a defined CRS")
    fields = list(info["fields"])
    if "Mang_Type" not in fields:
        raise ValueError("The all-PAD source must contain Mang_Type")
    collisions = {field for field in fields if field.casefold() in {
        generated.casefold() for generated in _GENERATED_FIELDS
    }}
    if collisions:
        raise ValueError(f"PAD-US source fields collide with generated fields: {sorted(collisions)}")
    if len({field.casefold() for field in fields}) != len(fields):
        raise ValueError("PAD-US source contains case-insensitive duplicate field names")
    bbox = county_bounds_in_crs(counties, info["crs"])
    print("Read all-PAD source geometry and attributes", flush=True)
    frame = pyogrio.read_dataframe(path, layer=layer, bbox=bbox, fid_as_index=True)
    print(f"Read {len(frame):,} PAD-US features in the county extent", flush=True)
    # FileGDB exposes OBJECTID as the feature ID, outside the ordinary fields.
    # Also accept a previously copied source with an explicit OBJECTID column.
    objectid_columns = [field for field in fields if field.casefold() == "objectid"]
    if objectid_columns and objectid_columns != ["OBJECTID"]:
        raise ValueError("The PAD-US source identifier must use the original OBJECTID field name")
    if "OBJECTID" not in frame:
        frame.insert(0, "OBJECTID", frame.index.to_numpy())
    objectids = pd.to_numeric(frame["OBJECTID"], errors="coerce")
    if (objectids.isna().any() or not np.isfinite(objectids).all()
            or (objectids % 1 != 0).any() or objectids.duplicated().any()):
        raise ValueError("PAD-US OBJECTID values must be unique, finite integers")
    frame = frame.reset_index(drop=True)
    rows_read = len(frame)
    if not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all() or frame.geometry.is_empty.any():
        raise ValueError("PAD-US source contains null, empty, or nonpolygonal geometry")
    if not frame.empty:
        # National features can also contain Alaska or overseas polygons. Clip
        # these in source coordinates so projection cannot wrap them into CONUS.
        bounds = shapely.bounds(frame.geometry)
        overhang = ((bounds[:, :2] < bbox[:2]) | (bounds[:, 2:] > bbox[2:])).any(axis=1)
        if overhang.any():
            shapes = frame.geometry.loc[overhang].to_numpy(copy=True)
            invalid = ~shapely.is_valid(shapes)
            shapes[invalid] = shapely.make_valid(shapes[invalid])
            clipped = shapely.intersection(shapes, shapely.box(*bbox))
            area = shapely.area(clipped)
            if not np.isfinite(area).all():
                raise ValueError("PAD-US source produced nonfinite clipped areas")
            clipped_rows = frame.index[overhang]
            frame = frame.drop(index=clipped_rows[area <= 0]).copy()
            repaired = [repair_polygonal_geometry(geom) for geom in clipped[area > 0]]
            if any(geom is None for geom in repaired):
                raise ValueError("PAD-US source contains unrecoverable clipped polygons")
            frame.loc[clipped_rows[area > 0], frame.geometry.name] = gpd.GeoSeries(
                repaired, index=clipped_rows[area > 0], crs=frame.crs,
            )
    print("Project and validate all-PAD source geometry", flush=True)
    return project_polygons(frame, "all-PAD source").sort_values("OBJECTID").reset_index(drop=True), {
        "path": str(path.resolve()), "layer": layer, "rows_before_scope": info["features"],
        "rows_read_in_county_extent": rows_read, "source_fields": fields,
        "source_feature_id": info["fid_column"],
    }


def prepare_all_pad_zones(
    padus_path: Path, padus_layer: str,
    counties_path: Path, counties_layer: str,
    land_mask_path: Path, land_mask_layer: str,
    output_path: Path, *, workers: int = 4,
) -> dict:
    """Write a feature-preserving all-PAD county layer and return its provenance.

    Every source feature contributes at most one multipart record per county.
    Source attributes, including original area fields and manager labels, are
    copied unchanged. New area_ha measures the county-cut geometry including
    water; area_ha_land and county_land_share follow existing valuation rules.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    counties, counties_layer = read_polygons(
        counties_path, counties_layer, {"GEOID", "ALAND"},
    )
    counties = _county_ids(counties, "Counties")
    if counties.empty:
        raise ValueError("No CONUS counties in the supplied county layer")
    masks, land_mask_layer = read_polygons(land_mask_path, land_mask_layer, {"GEOID"})
    masks = _county_ids(masks, "Land masks")
    if set(masks["GEOID"]) != set(counties["GEOID"]):
        raise ValueError("County land mask GEOIDs must match the supplied CONUS counties")
    source, source_info = _read_padus(padus_path, padus_layer, counties)
    counties = project_polygons(counties, "Counties").reset_index(drop=True)
    masks = project_polygons(masks, "Land masks").set_index("GEOID")
    aland = pd.to_numeric(counties["ALAND"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    source_index = source.sindex

    def process_county(index: int) -> gpd.GeoDataFrame:
        """Cut each candidate independently and copy its source attributes."""
        county = counties.iloc[index]
        geoid = str(county["GEOID"])
        county_shape = county.geometry
        mask = masks.loc[geoid].geometry
        # Reuse the same mask-vs-ALAND and containment checks as existing zones.
        build_county_zones(geoid, county_shape, aland[index], [], [], county_land_mask=mask)
        candidates = np.sort(source_index.query(county_shape, predicate="intersects"))
        result = source.iloc[candidates].copy()
        clipped = shapely.intersection(result.geometry.to_numpy(), county_shape)
        positive = shapely.area(clipped) > 0
        result = result.loc[positive].copy()
        geometry = [repair_polygonal_geometry(geom) for geom in clipped[positive]]
        if any(geom is None for geom in geometry):
            raise ValueError(f"County {geoid} produced an unrecoverable PAD-US intersection")
        result.geometry = gpd.GeoSeries(geometry, index=result.index, crs=AREA_CRS)
        area = result.geometry.area.to_numpy()
        # build_county_zones prepared this mask. Most inland features are
        # wholly on land and need no expensive overlay with the coastal mask.
        shapes = result.geometry.to_numpy()
        crossing = ~shapely.covers(mask, shapes)
        land_area = area.copy()
        land_area[crossing] = shapely.area(shapely.intersection(shapes[crossing], mask))
        tolerance = max(0.01, county_shape.area * 1e-10)
        if (not np.isfinite(area).all() or not np.isfinite(land_area).all()
                or (land_area < 0).any() or (land_area > area + tolerance).any()
                or (land_area > mask.area + tolerance).any()):
            raise ValueError(f"County {geoid} produced inconsistent PAD-US feature areas")
        result["feature_id"] = [f"{geoid}_{int(value)}" for value in result["OBJECTID"]]
        result["GEOID"] = geoid
        result["area_ha"] = area / 10_000
        result["area_ha_land"] = land_area / 10_000
        result["county_land_share"] = np.minimum(1.0, land_area / aland[index])
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pieces = list(tqdm(
            pool.map(process_county, range(len(counties))), total=len(counties),
            desc="Cut PAD-US source features by county", unit="county",
        ))
    output = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), geometry=source.geometry.name, crs=AREA_CRS)
    if output["feature_id"].duplicated().any():
        raise ValueError("PAD-US county intersections have duplicate feature identifiers")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(output, output_path, layer=ALL_PAD_PRODUCT, geometry_type="MultiPolygon")
    return {
        "geographic_scope": "CONUS", "area_crs": AREA_CRS,
        "inputs": {
            "padus": source_info,
            "counties": {"path": str(counties_path.resolve()), "layer": counties_layer},
            "land_mask": {"path": str(land_mask_path.resolve()), "layer": land_mask_layer},
        },
        "selection": "All source PAD-US features with positive county intersection; no GAP or manager filter",
        "row_definition": "One source OBJECTID and county GEOID; separate source features are never dissolved",
        "feature_id": "GEOID + '_' + source OBJECTID",
        "area_method": {
            "area_ha": "County-cut feature area, including water, in EPSG:6933 / 10000",
            "area_ha_land": "County-cut feature intersected with Census county land mask / 10000",
            "county_land_share": "Land intersection area / Census ALAND, bounded to 1 after mask validation",
        },
        "output": {"path": str(output_path), "layer": ALL_PAD_PRODUCT, "rows": len(output)},
    }
