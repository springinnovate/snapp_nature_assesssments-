"""Calculate per-feature polygon area in hectares for a vector layer.

The script reads a vector dataset supported by GDAL/OGR, writes a timestamped
GeoPackage copy with an ``area_ha`` field, and writes a matching CSV table.

For projected layers, area is computed in the layer's projected units and
converted to hectares. For geographic layers, polygon area is computed
geodesically on the CRS ellipsoid so degree-based coordinates are handled
correctly.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

from osgeo import gdal, ogr, osr
import pyproj
from shapely import wkb
from shapely.geometry.base import BaseGeometry

gdal.UseExceptions()

AREA_FIELD = "area_ha"
FID_FIELD = "source_fid"
HECTARES_PER_SQUARE_METER = 1.0 / 10000.0


def _make_area_calculator(srs: osr.SpatialReference):
    """Create a function that returns area in hectares for an OGR geometry."""
    if srs is None:
        raise RuntimeError(
            "Input layer has no CRS. Area cannot be calculated safely because "
            "the coordinate units are unknown."
        )

    if srs.IsProjected():
        meters_per_unit = srs.GetLinearUnits()
        if not meters_per_unit:
            raise RuntimeError("Projected CRS does not report linear units.")
        sq_meters_per_sq_unit = meters_per_unit * meters_per_unit

        def projected_area_ha(geom: ogr.Geometry) -> float:
            return abs(geom.GetArea()) * sq_meters_per_sq_unit * HECTARES_PER_SQUARE_METER

        return projected_area_ha

    if srs.IsGeographic():
        crs = pyproj.CRS.from_wkt(srs.ExportToWkt())
        geod = crs.get_geod()
        if geod is None:
            ellipsoid = crs.ellipsoid
            if ellipsoid.semi_major_metre and ellipsoid.inverse_flattening:
                geod = pyproj.Geod(
                    a=ellipsoid.semi_major_metre,
                    rf=ellipsoid.inverse_flattening,
                )
            else:
                geod = pyproj.Geod(ellps="WGS84")

        def geographic_area_ha(geom: ogr.Geometry) -> float:
            shapely_geom: BaseGeometry = wkb.loads(bytes(geom.ExportToWkb()))
            area_m2, _ = geod.geometry_area_perimeter(shapely_geom)
            return abs(area_m2) * HECTARES_PER_SQUARE_METER

        return geographic_area_ha

    raise RuntimeError(
        "Input layer CRS is neither projected nor geographic. "
        "Please reproject the layer to an equal-area CRS first."
    )


def _copy_schema_with_area_field(src_layer: ogr.Layer, dst_layer: ogr.Layer) -> None:
    """Copy source fields and add ``area_ha`` when it is absent."""
    src_defn = src_layer.GetLayerDefn()
    has_area_field = False

    for i in range(src_defn.GetFieldCount()):
        field_defn = src_defn.GetFieldDefn(i)
        if field_defn.GetName() == AREA_FIELD:
            has_area_field = True
        dst_layer.CreateField(field_defn)

    if not has_area_field:
        area_defn = ogr.FieldDefn(AREA_FIELD, ogr.OFTReal)
        area_defn.SetWidth(32)
        area_defn.SetPrecision(10)
        dst_layer.CreateField(area_defn)


def _csv_fieldnames(src_layer: ogr.Layer) -> list[str]:
    """Build CSV field order with area first, then FID and source fields."""
    src_defn = src_layer.GetLayerDefn()
    fieldnames = [AREA_FIELD, FID_FIELD]
    for i in range(src_defn.GetFieldCount()):
        name = src_defn.GetFieldDefn(i).GetName()
        if name == AREA_FIELD:
            continue
        if name == FID_FIELD:
            continue
        fieldnames.append(name)
    return fieldnames


def _output_geometry_type(src_layer: ogr.Layer) -> int:
    """Return a permissive output geometry type for mixed polygon datasets."""
    geom_type = src_layer.GetGeomType()
    flat_geom_type = ogr.GT_Flatten(geom_type)

    if flat_geom_type == ogr.wkbPolygon:
        return ogr.wkbMultiPolygon
    if flat_geom_type == ogr.wkbLineString:
        return ogr.wkbMultiLineString
    if flat_geom_type == ogr.wkbPoint:
        return ogr.wkbMultiPoint
    return geom_type


def _promote_geometry_if_needed(geom: ogr.Geometry, dst_geom_type: int) -> ogr.Geometry:
    """Promote single-part geometries when the output layer is multi-part."""
    flat_src_type = ogr.GT_Flatten(geom.GetGeometryType())
    flat_dst_type = ogr.GT_Flatten(dst_geom_type)

    if flat_src_type == ogr.wkbPolygon and flat_dst_type == ogr.wkbMultiPolygon:
        multi_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        multi_geom.AddGeometry(geom)
        return multi_geom
    if flat_src_type == ogr.wkbLineString and flat_dst_type == ogr.wkbMultiLineString:
        multi_geom = ogr.Geometry(ogr.wkbMultiLineString)
        multi_geom.AddGeometry(geom)
        return multi_geom
    if flat_src_type == ogr.wkbPoint and flat_dst_type == ogr.wkbMultiPoint:
        multi_geom = ogr.Geometry(ogr.wkbMultiPoint)
        multi_geom.AddGeometry(geom)
        return multi_geom
    return geom


def _safe_text(value: str) -> str:
    """Replace undecodable legacy text with valid UTF-8 replacement chars."""
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _safe_field_value(value):
    """Make OGR field values safe for UTF-8 CSV and GeoPackage output."""
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, list):
        return [_safe_field_value(item) for item in value]
    return value


def calculate_vector_area_ha(
    vector_path: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path, int]:
    """Write a GeoPackage and CSV with per-feature area in hectares."""
    vector_path = vector_path.resolve()
    if output_dir is not None:
        output_dir = output_dir.resolve()

    src_ds = gdal.OpenEx(str(vector_path), gdal.OF_VECTOR)
    src_layer = src_ds.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    area_ha = _make_area_calculator(src_srs)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_stem = f"{vector_path.stem}_area_ha_{timestamp}"
    output_directory = output_dir if output_dir is not None else vector_path.parent
    output_gpkg_path = output_directory / f"{output_stem}.gpkg"
    output_csv_path = output_directory / f"{output_stem}.csv"
    output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    gpkg_driver = ogr.GetDriverByName("GPKG")
    if output_gpkg_path.exists():
        gpkg_driver.DeleteDataSource(str(output_gpkg_path))

    dst_ds = gpkg_driver.CreateDataSource(str(output_gpkg_path))
    if dst_ds is None:
        raise RuntimeError(f"Could not create output GeoPackage: {output_gpkg_path}")

    out_layer_name = output_gpkg_path.stem
    dst_geom_type = _output_geometry_type(src_layer)
    dst_layer = dst_ds.CreateLayer(
        out_layer_name,
        src_srs,
        dst_geom_type,
        options=["SPATIAL_INDEX=YES"],
    )
    if dst_layer is None:
        raise RuntimeError(f"Could not create output layer: {out_layer_name}")

    _copy_schema_with_area_field(src_layer, dst_layer)
    dst_defn = dst_layer.GetLayerDefn()
    csv_fieldnames = _csv_fieldnames(src_layer)
    src_defn = src_layer.GetLayerDefn()
    src_field_names = [
        src_defn.GetFieldDefn(i).GetName() for i in range(src_defn.GetFieldCount())
    ]

    feature_count = 0
    with output_csv_path.open(
        "w", newline="", encoding="utf-8", errors="replace"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()

        src_layer.ResetReading()
        for src_feature in src_layer:
            geom = src_feature.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                feature_area_ha = 0.0
                out_geom = None
            else:
                geom_clone = geom.Clone()
                feature_area_ha = area_ha(geom_clone)
                out_geom = _promote_geometry_if_needed(geom_clone, dst_geom_type)

            dst_feature = ogr.Feature(dst_defn)
            for field_name in src_field_names:
                field_value = src_feature.GetField(field_name)
                if field_value is not None:
                    dst_feature.SetField(field_name, _safe_field_value(field_value))
            dst_feature.SetField(AREA_FIELD, float(feature_area_ha))
            if out_geom is not None:
                dst_feature.SetGeometry(out_geom)
            dst_layer.CreateFeature(dst_feature)

            csv_row = {
                AREA_FIELD: feature_area_ha,
                FID_FIELD: src_feature.GetFID(),
            }
            for field_name in csv_fieldnames[2:]:
                csv_row[field_name] = _safe_field_value(src_feature.GetField(field_name))
            writer.writerow(csv_row)

            dst_feature = None
            feature_count += 1

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None
    return output_gpkg_path, output_csv_path, feature_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-feature area in hectares for a vector layer and "
            "write timestamped GeoPackage and CSV outputs."
        )
    )
    parser.add_argument(
        "vector_path",
        type=Path,
        help="Path to an input vector dataset, such as a .shp or .gpkg.",
    )
    parser.add_argument(
        "--layer",
        help="Layer name to read. Defaults to the first layer.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Zero-based layer index to read when --layer is not set. Default: 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for outputs. Defaults to the input vector directory.",
    )
    args = parser.parse_args(argv)

    output_gpkg_path, output_csv_path, feature_count = calculate_vector_area_ha(
        args.vector_path,
        output_dir=args.output_dir,
        layer_name=args.layer,
        layer_index=args.layer_index,
    )
    print(f"Processed {feature_count} feature(s).")
    print(f"Wrote vector: {output_gpkg_path}")
    print(f"Wrote CSV: {output_csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
