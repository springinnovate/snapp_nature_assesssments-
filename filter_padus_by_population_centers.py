"""Clip a PAD-US GeoPackage to qualifying population-center proximity zones.

This implements only Section 50301(f)(3)(D)'s spatial condition:

* five statute miles from an incorporated municipality's boundary; or
* five statute miles from a Census-designated place's Census centroid.

It does not evaluate the section's other exclusions or disposal requirements.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from osgeo import gdal, ogr, osr
from pyproj import CRS, Transformer
import shapely
from shapely import wkb
from shapely.geometry import Point
from shapely.ops import transform
from shapely.strtree import STRtree
from tqdm import tqdm

from geometry_utils import repair_polygonal_geometry


gdal.UseExceptions()
ogr.UseExceptions()

DEFAULT_POPULATION_CENTERS = Path(
    "data/analysis_inputs/census_population_centers_2020.gpkg"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processing_outputs/padus_population_center_proximity"
)
INCORPORATED_LAYER = "incorporated_places_pop1000"
CDP_LAYER = "census_designated_places_pop1000"
DEFAULT_OUTPUT_LAYER = "padus_within_population_center_proximity"
DEFAULT_DISTANCE_MILES = 5.0
METERS_PER_MILE = 1609.344
TRANSACTION_SIZE = 10_000


@dataclass(frozen=True)
class FilterResult:
    """Summary returned by :func:`filter_padus_by_population_centers`."""

    output_path: Path
    input_features: int
    retained_features: int
    incorporated_places: int
    census_designated_places: int
    stats: Counter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clip a PAD-US GeoPackage to land within five miles of a "
            "qualifying incorporated-place boundary or CDP centroid."
        )
    )
    parser.add_argument("input_gpkg", type=Path, help="Input PAD-US GeoPackage.")
    parser.add_argument(
        "--input-layer",
        help="Input layer name. Defaults to the only layer in the GeoPackage.",
    )
    parser.add_argument(
        "--population-centers-gpkg",
        type=Path,
        default=DEFAULT_POPULATION_CENTERS,
        help=f"Population-center GeoPackage (default: {DEFAULT_POPULATION_CENTERS}).",
    )
    parser.add_argument(
        "--incorporated-layer",
        default=INCORPORATED_LAYER,
        help=f"Incorporated-place layer (default: {INCORPORATED_LAYER}).",
    )
    parser.add_argument(
        "--cdp-layer",
        default=CDP_LAYER,
        help=f"Census-designated-place layer (default: {CDP_LAYER}).",
    )
    parser.add_argument(
        "--distance-miles",
        type=float,
        default=DEFAULT_DISTANCE_MILES,
        help=f"Buffer distance in statute miles (default: {DEFAULT_DISTANCE_MILES}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GeoPackage. Defaults to a timestamped processing output.",
    )
    parser.add_argument(
        "--output-layer",
        default=DEFAULT_OUTPUT_LAYER,
        help=f"Output layer name (default: {DEFAULT_OUTPUT_LAYER}).",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return DEFAULT_OUTPUT_DIR / (
        f"{input_path.stem}_within_5_miles_population_centers_{timestamp}.gpkg"
    )


def _open_input_layer(
    input_path: Path,
    requested_layer: str | None,
) -> tuple[gdal.Dataset, ogr.Layer, str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input GeoPackage does not exist: {input_path}")

    dataset = gdal.OpenEx(str(input_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    if dataset is None:
        raise RuntimeError(f"Could not open input GeoPackage: {input_path}")

    if requested_layer:
        layer = dataset.GetLayerByName(requested_layer)
        if layer is None:
            raise ValueError(
                f"Input layer {requested_layer!r} does not exist in {input_path}."
            )
        layer_name = requested_layer
    else:
        if dataset.GetLayerCount() != 1:
            raise ValueError(
                f"{input_path} contains {dataset.GetLayerCount()} layers; "
                "provide --input-layer."
            )
        layer = dataset.GetLayer(0)
        layer_name = layer.GetName()

    if layer.GetSpatialRef() is None:
        raise ValueError(f"Input layer {layer_name!r} has no CRS.")
    return dataset, layer, layer_name


def _read_population_layer(path: Path, layer_name: str) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Population-center GeoPackage does not exist: {path}")
    try:
        frame = gpd.read_file(path, layer=layer_name)
    except Exception as error:
        message = (
            f"Could not read population-center layer {layer_name!r} "
            f"from {path}: {error}"
        )
        raise ValueError(
            message
        ) from error
    if frame.crs is None:
        raise ValueError(f"Population-center layer {layer_name!r} has no CRS.")
    if frame.empty:
        raise ValueError(f"Population-center layer {layer_name!r} is empty.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(
            f"Population-center layer {layer_name!r} contains empty geometry."
        )
    return frame


@lru_cache(maxsize=256)
def _coordinate_transformer(source_crs: CRS, target_crs: CRS) -> Transformer:
    """Cache the small set of source/UTM/output transformations in one run."""
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _transform_geometry(geometry, source_crs: CRS, target_crs: CRS):
    if source_crs == target_crs:
        return geometry
    transformer = _coordinate_transformer(source_crs, target_crs)
    return transform(transformer.transform, geometry)


def _local_utm_crs(geometry, source_crs: CRS) -> CRS:
    representative_point = geometry.representative_point()
    lon_lat = _transform_geometry(
        representative_point,
        source_crs,
        CRS.from_epsg(4326),
    )
    longitude = lon_lat.x
    latitude = lon_lat.y
    if not (-80 <= latitude <= 84):
        raise ValueError(
            f"Population-center latitude {latitude} is outside UTM coverage."
        )
    zone = max(1, min(60, int((longitude + 180) // 6) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _buffer_geometry_locally(
    geometry,
    source_crs: CRS,
    target_crs: CRS,
    distance_meters: float,
    *,
    boundary_only: bool,
):
    """Buffer one geometry in its local UTM CRS and return it in target CRS."""
    local_crs = _local_utm_crs(geometry, source_crs)
    local_geometry = _transform_geometry(geometry, source_crs, local_crs)
    buffer_source = local_geometry.boundary if boundary_only else local_geometry
    buffered = buffer_source.buffer(distance_meters)
    buffered = _transform_geometry(buffered, local_crs, target_crs)
    buffered = repair_polygonal_geometry(buffered)
    if buffered is None:
        raise ValueError("A population-center buffer produced no polygon geometry.")
    return buffered


def _cdp_centroid(row, layer_name: str) -> Point:
    missing = {"CENTLON", "CENTLAT"} - set(row.index)
    if missing:
        raise ValueError(
            f"Population-center layer {layer_name!r} is missing centroid field(s): "
            + ", ".join(sorted(missing))
        )
    try:
        return Point(float(row["CENTLON"]), float(row["CENTLAT"]))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Population-center layer {layer_name!r} has invalid CENTLON/CENTLAT "
            f"for GEOID {row.get('GEOID', '<unknown>')!r}."
        ) from error


def build_proximity_zone(
    population_centers_path: Path,
    incorporated_layer: str,
    cdp_layer: str,
    target_crs: CRS,
    distance_miles: float,
) -> tuple[list, int, int]:
    """Build non-overlapping proximity-zone parts in the input layer CRS."""
    if distance_miles <= 0:
        raise ValueError("--distance-miles must be greater than zero.")

    incorporated = _read_population_layer(population_centers_path, incorporated_layer)
    cdps = _read_population_layer(population_centers_path, cdp_layer)
    incorporated_crs = CRS.from_user_input(incorporated.crs)
    cdp_crs = CRS.from_user_input(cdps.crs)
    distance_meters = distance_miles * METERS_PER_MILE

    buffers = []
    for geometry in incorporated.geometry:
        buffers.append(
            _buffer_geometry_locally(
                geometry,
                incorporated_crs,
                target_crs,
                distance_meters,
                boundary_only=True,
            )
        )

    for _, row in cdps.iterrows():
        center = _cdp_centroid(row, cdp_layer)
        buffers.append(
            _buffer_geometry_locally(
                center,
                cdp_crs,
                target_crs,
                distance_meters,
                boundary_only=False,
            )
        )

    merged = repair_polygonal_geometry(shapely.union_all(buffers))
    if merged is None:
        raise RuntimeError("Population-center buffers produced no proximity zone.")
    return list(merged.geoms), len(incorporated), len(cdps)


def _create_output(
    output_path: Path,
    output_layer_name: str,
    input_layer: ogr.Layer,
) -> tuple[ogr.DataSource, ogr.Layer]:
    if output_path.exists():
        raise FileExistsError(f"Output GeoPackage already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(output_path))
    if dataset is None:
        raise RuntimeError(f"Could not create output GeoPackage: {output_path}")
    layer = dataset.CreateLayer(
        output_layer_name,
        input_layer.GetSpatialRef(),
        ogr.wkbMultiPolygon,
        options=["SPATIAL_INDEX=YES"],
    )
    if layer is None:
        raise RuntimeError(f"Could not create output layer {output_layer_name!r}.")

    input_definition = input_layer.GetLayerDefn()
    for field_index in range(input_definition.GetFieldCount()):
        source = input_definition.GetFieldDefn(field_index)
        destination = ogr.FieldDefn(source.GetName(), source.GetType())
        destination.SetSubType(source.GetSubType())
        destination.SetWidth(source.GetWidth())
        destination.SetPrecision(source.GetPrecision())
        layer.CreateField(destination)
    return dataset, layer


def _write_feature(
    output_layer: ogr.Layer,
    input_feature: ogr.Feature,
    geometry,
) -> None:
    output_feature = ogr.Feature(output_layer.GetLayerDefn())
    input_definition = input_feature.GetDefnRef()
    for field_index in range(input_definition.GetFieldCount()):
        value = input_feature.GetField(field_index)
        if value is not None:
            output_feature.SetField(field_index, value)
    output_feature.SetGeometry(ogr.CreateGeometryFromWkb(wkb.dumps(geometry)))
    output_layer.CreateFeature(output_feature)


def filter_padus_by_population_centers(
    input_path: Path,
    population_centers_path: Path,
    output_path: Path,
    *,
    input_layer_name: str | None = None,
    output_layer_name: str = DEFAULT_OUTPUT_LAYER,
    incorporated_layer: str = INCORPORATED_LAYER,
    cdp_layer: str = CDP_LAYER,
    distance_miles: float = DEFAULT_DISTANCE_MILES,
) -> FilterResult:
    """Clip input features to the union of qualifying proximity zones."""
    input_dataset, input_layer, resolved_input_layer = _open_input_layer(
        input_path,
        input_layer_name,
    )
    input_count = input_layer.GetFeatureCount()
    target_crs = CRS.from_wkt(input_layer.GetSpatialRef().ExportToWkt())
    zone_parts, incorporated_count, cdp_count = build_proximity_zone(
        population_centers_path,
        incorporated_layer,
        cdp_layer,
        target_crs,
        distance_miles,
    )
    zone_index = STRtree(zone_parts)

    output_dataset, output_layer = _create_output(
        output_path,
        output_layer_name,
        input_layer,
    )
    stats = Counter()
    output_layer.StartTransaction()

    for input_feature in tqdm(
        input_layer,
        total=input_count,
        desc=f"Filter {resolved_input_layer}",
        unit="feature",
    ):
        stats["scanned"] += 1
        ogr_geometry = input_feature.GetGeometryRef()
        if ogr_geometry is None or ogr_geometry.IsEmpty():
            stats["empty_geometry_skipped"] += 1
            continue

        geometry = wkb.loads(bytes(ogr_geometry.GetLinearGeometry().ExportToWkb()))
        if not geometry.is_valid:
            geometry = repair_polygonal_geometry(geometry)
            if geometry is None:
                stats["invalid_geometry_skipped"] += 1
                continue

        candidate_indexes = zone_index.query(geometry, predicate="intersects")
        if len(candidate_indexes) == 0:
            stats["outside_proximity_skipped"] += 1
            continue

        candidate_zone = shapely.union_all(
            [zone_parts[index] for index in candidate_indexes]
        )
        clipped = repair_polygonal_geometry(
            shapely.intersection(geometry, candidate_zone)
        )
        if clipped is None or clipped.area <= 0:
            stats["zero_area_intersection_skipped"] += 1
            continue

        _write_feature(output_layer, input_feature, clipped)
        stats["retained"] += 1
        if stats["retained"] % TRANSACTION_SIZE == 0:
            output_layer.CommitTransaction()
            output_layer.StartTransaction()

    output_layer.CommitTransaction()
    output_dataset.FlushCache()
    output_layer = None
    output_dataset = None
    input_layer = None
    input_dataset = None

    return FilterResult(
        output_path=output_path,
        input_features=input_count,
        retained_features=stats["retained"],
        incorporated_places=incorporated_count,
        census_designated_places=cdp_count,
        stats=stats,
    )


def main() -> None:
    args = _parse_args()
    output_path = args.output or _default_output_path(args.input_gpkg)
    result = filter_padus_by_population_centers(
        input_path=args.input_gpkg,
        input_layer_name=args.input_layer,
        population_centers_path=args.population_centers_gpkg,
        incorporated_layer=args.incorporated_layer,
        cdp_layer=args.cdp_layer,
        distance_miles=args.distance_miles,
        output_path=output_path,
        output_layer_name=args.output_layer,
    )
    print(
        "Population centers: "
        f"incorporated={result.incorporated_places:,} "
        f"cdp={result.census_designated_places:,}"
    )
    print(
        "Feature stats: "
        + " ".join(f"{key}={value:,}" for key, value in result.stats.items())
    )
    print(
        f"Wrote {result.retained_features:,} of {result.input_features:,} "
        f"input feature(s): {result.output_path}"
    )


if __name__ == "__main__":
    main()
