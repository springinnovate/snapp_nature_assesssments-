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
    """Create an area calculator for the input layer CRS.

    Args:
        srs: Spatial reference for the input layer.

    Returns:
        A callable that accepts an OGR geometry and returns area in hectares.

    Raises:
        RuntimeError: If the spatial reference is missing or does not describe
            projected or geographic coordinates.
    """
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


def _csv_fieldnames(src_layer: ogr.Layer) -> list[str]:
    """Build the CSV field order for the output table.

    Args:
        src_layer: Input vector layer whose attribute fields should be copied.

    Returns:
        CSV field names with ``area_ha`` first, followed by ``source_fid`` and
        the source attribute fields.
    """
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


def _csv_safe_field_value(value):
    """Make an OGR field value safe for UTF-8 CSV output.

    Args:
        value: OGR field value to write to the CSV table.

    Returns:
        The original value, with strings converted to valid UTF-8 and list
        values sanitized item by item.
    """
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_csv_safe_field_value(item) for item in value]
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
    csv_fieldnames = _csv_fieldnames(src_layer)

    src_layer.ResetReading()
    source_fids = [src_feature.GetFID() for src_feature in src_layer]
    src_layer.ResetReading()

    gpkg_driver = ogr.GetDriverByName("GPKG")
    if output_gpkg_path.exists():
        gpkg_driver.DeleteDataSource(str(output_gpkg_path))

    dst_ds = gpkg_driver.CreateDataSource(str(output_gpkg_path))
    if dst_ds is None:
        raise RuntimeError(f"Could not create output GeoPackage: {output_gpkg_path}")

    dst_layer = dst_ds.CopyLayer(
        src_layer,
        output_gpkg_path.stem,
        options=["SPATIAL_INDEX=YES"],
    )
    if dst_layer is None:
        raise RuntimeError(f"Could not create output layer: {output_gpkg_path.stem}")

    if dst_layer.GetLayerDefn().GetFieldIndex(AREA_FIELD) == -1:
        area_defn = ogr.FieldDefn(AREA_FIELD, ogr.OFTReal)
        area_defn.SetWidth(32)
        area_defn.SetPrecision(10)
        dst_layer.CreateField(area_defn)

    feature_count = 0
    with output_csv_path.open(
        "w", newline="", encoding="utf-8", errors="replace"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()

        dst_layer.ResetReading()
        for dst_feature, source_fid in zip(dst_layer, source_fids):
            geom = dst_feature.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                feature_area_ha = 0.0
            else:
                feature_area_ha = area_ha(geom)

            dst_feature.SetField(AREA_FIELD, float(feature_area_ha))
            dst_layer.SetFeature(dst_feature)

            csv_row = {
                AREA_FIELD: feature_area_ha,
                FID_FIELD: source_fid,
            }
            for field_name in csv_fieldnames[2:]:
                csv_row[field_name] = _csv_safe_field_value(
                    dst_feature.GetField(field_name)
                )
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
