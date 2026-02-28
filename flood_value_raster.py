from osgeo import gdal, ogr, osr
import pandas as pd
from tqdm import tqdm

gdal.UseExceptions()

WETLANDS_MASK_RASTER_PATH = r"D:\repositories\zonal_stats_toolkit\snappdata\reclassified_Annual_NLCD_wetlands_reclass.tif"
HUC_TABLE_PATH = (
    r"D:\repositories\zonal_stats_toolkit\snappdata\huc_summary_forSNAPP_v2.csv"
)
HUC_VECTOR_PATH = r"D:\repositories\zonal_stats_toolkit\snappdata\WBD_National_GDB\WBD_National_GDB.gdb"

ANNUAL_VALUE_FIELD = "marginal_annualvalue_2020"
MARGINAL_NPV_FIELD = "marginal_npv_2020"
HUC_FIELD = "huc12"

OUT_GPKG_PATH = r"hucs_joined_with_value_fields.gpkg"
OUT_GPKG_REPROJECTED_PATH = r"hucs_joined_with_value_fields_reprojected.gpkg"
OUT_GPKG_SIMPLIFIED_PATH = r"hucs_joined_with_value_fields_reprojected_simplified.gpkg"

OUT_ANNUAL_RASTER_PATH = r"annual_value_masked_to_wetlands.tif"
OUT_NPV_RASTER_PATH = r"marginal_npv_masked_to_wetlands.tif"

NODATA_VALUE = -9999.0
SIMPLIFY_TOLERANCE_PIXELS = 0.5
SIMPLIFY_PRESERVE_TOPOLOGY = True


def _create_like_mask(mask_ds, out_path, nodata, gdal_type):
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


def _mask_pixel_size_and_srs(mask_path):
    ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    ds = None
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return abs(gt[1]), abs(gt[5]), srs


def _clone_layer_schema(src_lyr, dst_lyr):
    defn = src_lyr.GetLayerDefn()
    for i in range(defn.GetFieldCount()):
        dst_lyr.CreateField(defn.GetFieldDefn(i))


def _reproject_gpkg_to_srs(in_gpkg_path, out_gpkg_path, target_srs):
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


def main():
    huc_table = pd.read_csv(HUC_TABLE_PATH)

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

    px_w, px_h, mask_srs = _mask_pixel_size_and_srs(WETLANDS_MASK_RASTER_PATH)
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
