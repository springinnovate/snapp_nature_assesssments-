"""Join HUC-level valuation data to watershed polygons and rasterize to a wetlands mask grid.

This script performs the following workflow:

1. Reads a CSV table containing HUC12 identifiers and associated
   annual value and marginal NPV fields.
2. Joins these tabular values to a national HUC vector dataset and
   writes the result to a GeoPackage.
3. Reprojects the joined GeoPackage to match the spatial reference
   system (SRS) of a wetlands mask raster.
4. Simplifies watershed geometries using a tolerance derived from
   the mask pixel size (in pixel units).
5. Rasterizes the annual value and marginal NPV attributes to new
   single-band GeoTIFF rasters aligned to the wetlands mask grid.
6. Applies a binary wetlands mask so that output raster values are
   set to NoData outside wetland pixels.

Outputs include:
- A joined GeoPackage with valuation attributes.
- A reprojected and simplified GeoPackage aligned to the mask SRS.
- Rasterized annual value and marginal NPV GeoTIFFs masked to wetlands.

Configuration is controlled via module-level constants defining input
paths, field names, NoData value, simplification tolerance (in pixels),
and topology preservation behavior.

This script requires GDAL/OGR with GeoPackage support and processes
features and rasters in a streaming/block-wise manner to limit memory use.
"""

import argparse

from osgeo import gdal, ogr, osr
from tqdm import tqdm
import numpy as np
import pandas as pd

gdal.UseExceptions()

WETLANDS_MASK_RASTER_PATH = r"D:\repositories\zonal_stats_toolkit\snappdata\reclassified_Annual_NLCD_wetlands_reclass.tif"
HUC_TABLE_PATH = (
    r"D:\repositories\zonal_stats_toolkit\snappdata\huc_summary_forSNAPP_v2.csv"
)
HUC_VECTOR_PATH = r"D:\repositories\zonal_stats_toolkit\snappdata\WBD_National_GDB\WBD_National_GDB.gdb"

ANNUAL_VALUE_FIELD = "marginal_annualvalue_2020"
MARGINAL_NPV_FIELD = "marginal_npv_2020"
HUC_FIELD = "huc12"
WETLAND_AREA_HA_FIELD = "wetland_area_ha_2023"

OUT_GPKG_PATH = r"hucs_joined_with_value_fields.gpkg"
OUT_GPKG_REPROJECTED_PATH = r"hucs_joined_with_value_fields_reprojected.gpkg"
OUT_GPKG_SIMPLIFIED_PATH = r"hucs_joined_with_value_fields_reprojected_simplified.gpkg"

OUT_ANNUAL_RASTER_PATH = r"annual_value_masked_to_wetlands.tif"
OUT_NPV_RASTER_PATH = r"marginal_npv_masked_to_wetlands.tif"

NODATA_VALUE = -9999.0
SIMPLIFY_TOLERANCE_PIXELS = 0.5
SIMPLIFY_PRESERVE_TOPOLOGY = True


def _create_like_mask(mask_ds, out_path, nodata, gdal_type):
    """Create a single-band GeoTIFF matching the spatial metadata of a mask dataset.

    The output dataset copies the geotransform, projection, and raster
    dimensions from ``mask_ds``. If a file already exists at ``out_path``,
    it is deleted before creation. The output is created with tiling,
    LZW compression, and BigTIFF enabled if safer.

    Args:
        mask_ds (gdal.Dataset): Source dataset whose spatial metadata
            (dimensions, geotransform, projection) will be copied.
        out_path (str): Path to the output GeoTIFF file.
        nodata (float | int): NoData value to assign to the output band.
        gdal_type (int): GDAL data type (e.g., gdal.GDT_Float32) for
            the output raster band.

    Returns:
        gdal.Dataset: The created GDAL dataset opened in write mode.

    Raises:
        RuntimeError: If the dataset cannot be created by the GDAL driver.
    """
    drv = gdal.GetDriverByName("GTiff")
    try:
        drv.Delete(out_path)
    except Exception:
        pass

    out_ds = drv.Create(
        out_path,
        mask_ds.RasterXSize,
        mask_ds.RasterYSize,
        1,
        gdal_type,
        [
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
        ],
    )
    out_ds.SetGeoTransform(mask_ds.GetGeoTransform())
    out_ds.SetProjection(mask_ds.GetProjection())
    b = out_ds.GetRasterBand(1)
    b.SetNoDataValue(float(nodata))
    b.FlushCache()
    return out_ds


def _apply_binary_mask(raster_path, mask_ds, nodata):
    """Apply a binary mask to a raster in place.

    Opens the raster at ``raster_path`` in update mode and sets all pixel
    values to ``nodata`` where the corresponding mask pixel is not equal
    to 1. Processing is performed in blocks based on the raster band’s
    native block size (or 256x256 if undefined) to limit memory usage.

    Args:
        raster_path (str): Path to the raster file to modify in place.
        mask_ds (gdal.Dataset): GDAL dataset containing a single-band
            binary mask aligned with the target raster.
        nodata (float | int): NoData value to assign to masked pixels.

    Returns:
        None
    """
    ds = gdal.Open(raster_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    mask_band = mask_ds.GetRasterBand(1)

    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    block_x, block_y = band.GetBlockSize()
    if block_x == 0 or block_y == 0:
        block_x, block_y = 256, 256

    for yoff in tqdm(range(0, ysize, block_y)):
        ywin = min(block_y, ysize - yoff)
        for xoff in range(0, xsize, block_x):
            xwin = min(block_x, xsize - xoff)

            m = mask_band.ReadAsArray(xoff, yoff, xwin, ywin)
            v = band.ReadAsArray(xoff, yoff, xwin, ywin)

            v[m != 1] = float(nodata)
            band.WriteArray(v, xoff, yoff)

    band.FlushCache()
    ds = None


def _get_mask_pixel_size_and_srs(mask_path):
    """Return pixel size and spatial reference for a raster mask.

    Opens the dataset at ``mask_path`` in read-only mode, extracts the
    geotransform and projection, and returns the absolute pixel width,
    absolute pixel height, and spatial reference object.

    Args:
        mask_path (str): Path to the raster mask file.

    Returns:
        tuple[float, float, osr.SpatialReference]: A tuple containing:
            - Pixel width (absolute value of geotransform[1]).
            - Pixel height (absolute value of geotransform[5]).
            - Spatial reference parsed from the dataset projection WKT.
    """
    ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    ds = None
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return abs(gt[1]), abs(gt[5]), srs


def _clone_layer_schema(src_lyr, dst_lyr):
    """Copy all field definitions from one OGR layer to another.

    Iterates over the source layer definition and creates matching
    fields on the destination layer. Only attribute schema (field
    definitions) is copied; features and geometry are not transferred.

    Args:
        src_lyr (ogr.Layer): Source layer whose field definitions
            will be read.
        dst_lyr (ogr.Layer): Destination layer where fields will
            be created.

    Returns:
        None
    """
    defn = src_lyr.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        dst_lyr.CreateField(defn.GetFieldDefn(i))


def _reproject_gpkg_to_srs(in_gpkg_path, out_gpkg_path, target_srs):
    """Reproject all features from a GeoPackage to a target spatial reference.

    Opens the input GeoPackage, reads the first layer, and writes a new
    GeoPackage at ``out_gpkg_path`` with all geometries transformed to
    ``target_srs``. The output layer preserves the original layer name,
    geometry type, and attribute schema. An existing output file is
    deleted if present. A spatial index is created on the output layer.

    Args:
        in_gpkg_path (str): Path to the input GeoPackage.
        out_gpkg_path (str): Path where the reprojected GeoPackage
            will be written.
        target_srs (osr.SpatialReference): Target spatial reference
            system for reprojection.

    Returns:
        None

    Raises:
        RuntimeError: If the input dataset cannot be opened or the
            output dataset cannot be created.
    """
    in_ds = ogr.Open(in_gpkg_path, 0)
    in_lyr = in_ds.GetLayer(0)
    layer_name = in_lyr.GetName()

    src_srs = in_lyr.GetSpatialRef()
    ct = osr.CoordinateTransformation(src_srs, target_srs)

    drv = ogr.GetDriverByName("GPKG")
    try:
        drv.DeleteDataSource(out_gpkg_path)
    except Exception:
        pass
    out_ds = drv.CreateDataSource(out_gpkg_path)

    out_lyr = out_ds.CreateLayer(
        layer_name,
        target_srs,
        in_lyr.GetGeomType(),
        options=["SPATIAL_INDEX=YES"],
    )
    _clone_layer_schema(in_lyr, out_lyr)
    out_defn = out_lyr.GetLayerDefn()

    in_lyr.ResetReading()
    for feat in tqdm(in_lyr, desc="reproject"):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        g = geom.Clone()
        g.Transform(ct)

        out_feat = ogr.Feature(out_defn)
        out_feat.SetFrom(feat)
        out_feat.SetGeometry(g)
        out_lyr.CreateFeature(out_feat)
        out_feat = None

    out_ds = None
    in_ds = None


def _simplify_gpkg(in_gpkg_path, out_gpkg_path, tol):
    """Simplify geometries in a GeoPackage and write to a new file.

    Opens the input GeoPackage, reads the first layer, and writes a new
    GeoPackage at ``out_gpkg_path`` with simplified geometries. The
    output layer preserves the original layer name, spatial reference,
    geometry type, and attribute schema. An existing output file is
    deleted if present. A spatial index is created on the output layer.

    Geometry simplification is performed using either
    ``SimplifyPreserveTopology`` or ``Simplify`` depending on the value
    of ``SIMPLIFY_PRESERVE_TOPOLOGY``.

    Args:
        in_gpkg_path (str): Path to the input GeoPackage.
        out_gpkg_path (str): Path where the simplified GeoPackage
            will be written.
        tol (float): Tolerance distance for geometry simplification,
            in the units of the layer's spatial reference.

    Returns:
        None

    Raises:
        RuntimeError: If the input dataset cannot be opened or the
            output dataset cannot be created.
    """
    in_ds = ogr.Open(in_gpkg_path, 0)
    in_lyr = in_ds.GetLayer(0)
    layer_name = in_lyr.GetName()

    drv = ogr.GetDriverByName("GPKG")
    try:
        drv.DeleteDataSource(out_gpkg_path)
    except Exception:
        pass
    out_ds = drv.CreateDataSource(out_gpkg_path)

    out_lyr = out_ds.CreateLayer(
        layer_name,
        in_lyr.GetSpatialRef(),
        in_lyr.GetGeomType(),
        options=["SPATIAL_INDEX=YES"],
    )
    _clone_layer_schema(in_lyr, out_lyr)
    out_defn = out_lyr.GetLayerDefn()

    in_lyr.ResetReading()
    for feat in tqdm(in_lyr, desc="simplify"):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        g = geom.Clone()
        if SIMPLIFY_PRESERVE_TOPOLOGY:
            g2 = g.SimplifyPreserveTopology(float(tol))
        else:
            g2 = g.Simplify(float(tol))

        out_feat = ogr.Feature(out_defn)
        out_feat.SetFrom(feat)
        out_feat.SetGeometry(g2)
        out_lyr.CreateFeature(out_feat)
        out_feat = None

    out_ds = None
    in_ds = None


def _calculate_area_weighted_sum(raster_path):
    """Compute the area-weighted sum of raster values in hectares.

    Opens a single-band raster, removes NoData and non-finite values,
    and computes the sum of remaining pixel values multiplied by the
    pixel area in hectares. Pixel area is derived from the raster
    geotransform assuming projected units in meters.

    Args:
        raster_path (str): Path to a single-band raster file.

    Returns:
        float: Area-weighted sum equal to:

            sum(valid_pixel_values) * pixel_area_ha

        where ``pixel_area_ha`` is the per-pixel area converted from
        square meters to hectares.
    """
    px_w, px_h, _ = _get_mask_pixel_size_and_srs(raster_path)
    pixel_area_ha = abs(px_w * px_h) / 10_000
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    if nodata is not None:
        array = array[array != nodata]
    array = array[np.isfinite(array)]

    area_weighted_sum = np.sum(array) * pixel_area_ha
    return area_weighted_sum


def main():
    parser = argparse.ArgumentParser(
        description="Run HUC valuation rasterization workflow."
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help=(
            "If set, calculate the sum of annual_value_masked_to_wetlands and "
            "marginal_npv_masked_to_wetlands multiplied by pixel area and "
            "compare to table-based calculations "
            "(ANNUAL_VALUE_FIELD * WETLAND_AREA_HA_FIELD and "
            "MARGINAL_NPV_FIELD * WETLAND_AREA_HA_FIELD)."
        ),
    )
    args = parser.parse_args()

    huc_table = pd.read_csv(HUC_TABLE_PATH)
    if args.analysis:
        print("Running analysis pathway")
        table_annual_value = sum(
            huc_table[ANNUAL_VALUE_FIELD] * huc_table[WETLAND_AREA_HA_FIELD]
        )
        table_marginal_value = sum(
            huc_table[MARGINAL_NPV_FIELD] * huc_table[WETLAND_AREA_HA_FIELD]
        )
        raster_annual_value = _calculate_area_weighted_sum(OUT_ANNUAL_RASTER_PATH)
        raster_npv_value = _calculate_area_weighted_sum(OUT_NPV_RASTER_PATH)

        diff_annual = raster_annual_value - table_annual_value
        diff_marginal = raster_npv_value - table_marginal_value

        pct_annual = (
            diff_annual / table_annual_value * 100
            if table_annual_value != 0
            else float("nan")
        )
        pct_marginal = (
            diff_marginal / table_marginal_value * 100
            if table_marginal_value != 0
            else float("nan")
        )

        print("\n===== ANALYSIS COMPARISON =====\n")

        print("ANNUAL VALUE ($)")
        print(f"  Table total : ${table_annual_value:,.2f}")
        print(f"  Raster total: ${raster_annual_value:,.2f}")
        print(f"  Difference  : ${diff_annual:,.2f} ({pct_annual:,.6f}%)\n")

        print("MARGINAL NPV ($)")
        print(f"  Table total : ${table_marginal_value:,.2f}")
        print(f"  Raster total: ${raster_npv_value:,.2f}")
        print(f"  Difference  : ${diff_marginal:,.2f} ({pct_marginal:,.6f}%)\n")

        return

    table_huc_int = (
        pd.to_numeric(huc_table[HUC_FIELD], errors="coerce").dropna().astype("int64")
    )
    table_av = pd.to_numeric(huc_table[ANNUAL_VALUE_FIELD], errors="coerce")
    table_npv = pd.to_numeric(huc_table[MARGINAL_NPV_FIELD], errors="coerce")
    table_map = dict(zip(table_huc_int, zip(table_av, table_npv)))
    huc_table = pd.read_csv(HUC_TABLE_PATH)
    print(huc_table)
    return

    table_huc_int = (
        pd.to_numeric(huc_table[HUC_FIELD], errors="coerce").dropna().astype("int64")
    )
    table_av = pd.to_numeric(huc_table[ANNUAL_VALUE_FIELD], errors="coerce")
    table_npv = pd.to_numeric(huc_table[MARGINAL_NPV_FIELD], errors="coerce")
    table_map = dict(zip(table_huc_int, zip(table_av, table_npv)))

    gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS", "SKIP")

    huc_vector = gdal.OpenEx(HUC_VECTOR_PATH, gdal.OF_VECTOR)
    huc_layer = huc_vector.GetLayer()

    gpkg_drv = ogr.GetDriverByName("GPKG")
    try:
        gpkg_drv.DeleteDataSource(OUT_GPKG_PATH)
    except Exception:
        pass

    out_ds = gpkg_drv.CreateDataSource(OUT_GPKG_PATH)
    out_layer = out_ds.CreateLayer(
        huc_layer.GetName(),
        huc_layer.GetSpatialRef(),
        huc_layer.GetGeomType(),
        options=["SPATIAL_INDEX=YES"],
    )

    in_defn = huc_layer.GetLayerDefn()
    for i in range(in_defn.GetFieldCount()):
        out_layer.CreateField(in_defn.GetFieldDefn(i))

    out_layer.CreateField(ogr.FieldDefn(ANNUAL_VALUE_FIELD, ogr.OFTReal))
    out_layer.CreateField(ogr.FieldDefn(MARGINAL_NPV_FIELD, ogr.OFTReal))

    out_defn = out_layer.GetLayerDefn()

    huc_layer.ResetReading()
    for feat in tqdm(huc_layer, desc="join"):
        huc_int = int(str(feat.GetField(HUC_FIELD)))
        vals = table_map.get(huc_int)
        if vals is None:
            continue

        out_feat = ogr.Feature(out_defn)
        out_feat.SetFrom(feat)
        out_feat.SetField(ANNUAL_VALUE_FIELD, float(vals[0]))
        out_feat.SetField(MARGINAL_NPV_FIELD, float(vals[1]))
        out_layer.CreateFeature(out_feat)
        out_feat = None

    out_ds = None
    huc_vector = None

    px_w, px_h, mask_srs = _get_mask_pixel_size_and_srs(WETLANDS_MASK_RASTER_PATH)
    tol = SIMPLIFY_TOLERANCE_PIXELS * max(px_w, px_h)

    _reproject_gpkg_to_srs(OUT_GPKG_PATH, OUT_GPKG_REPROJECTED_PATH, mask_srs)
    _simplify_gpkg(OUT_GPKG_REPROJECTED_PATH, OUT_GPKG_SIMPLIFIED_PATH, tol)

    mask_ds = gdal.Open(WETLANDS_MASK_RASTER_PATH, gdal.GA_ReadOnly)

    print(f"creating {OUT_ANNUAL_RASTER_PATH}")
    annual_ds = _create_like_mask(
        mask_ds, OUT_ANNUAL_RASTER_PATH, NODATA_VALUE, gdal.GDT_Float32
    )
    print(f"creating {OUT_NPV_RASTER_PATH}")
    npv_ds = _create_like_mask(
        mask_ds, OUT_NPV_RASTER_PATH, NODATA_VALUE, gdal.GDT_Float32
    )

    vec_ds = gdal.OpenEx(OUT_GPKG_SIMPLIFIED_PATH, gdal.OF_VECTOR)
    vec_lyr = vec_ds.GetLayer(0)

    print(f"rasterizing {OUT_ANNUAL_RASTER_PATH}")
    gdal.RasterizeLayer(
        annual_ds, [1], vec_lyr, options=[f"ATTRIBUTE={ANNUAL_VALUE_FIELD}"]
    )
    print(f"rasterizing {OUT_NPV_RASTER_PATH}")
    gdal.RasterizeLayer(
        npv_ds, [1], vec_lyr, options=[f"ATTRIBUTE={MARGINAL_NPV_FIELD}"]
    )

    annual_ds = None
    npv_ds = None
    vec_ds = None

    # _apply_binary_mask(OUT_ANNUAL_RASTER_PATH, mask_ds, NODATA_VALUE)
    _apply_binary_mask(OUT_NPV_RASTER_PATH, mask_ds, NODATA_VALUE)
    mask_ds = None


if __name__ == "__main__":
    main()
