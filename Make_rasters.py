import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
from pathlib import Path

# ==============================================
# RASTER EXPORT FOR AIR QUALITY LAYERS (CONUS)
# ==============================================
# - Uses county polygons + deposition_intensity_per_km2_by_env.csv
# - Filters to CONUS counties only:
#   excludes AK (02), HI (15), PR (72), VI (78), GU (66), MP (69), AS (60)
# - Creates one GeoTIFF per "intensity_*" column
# - CRS: EPSG:5070 (USA_Contiguous_Albers_Equal_Area_Conic)
# - Resolution: ~1 km
# ==============================================

# ---- BASE PATHS (you said this is correct) ----
BASE_DIR  = Path("/Users/jahnelle/Desktop/Deposition_Maps")
DATA_CSV  = BASE_DIR / "deposition_intensity_per_km2_by_env.csv"
GEO_PATH  = BASE_DIR / "gz_2010_us_050_00_5m.json"  # county boundaries
OUT_DIR   = BASE_DIR / "rasters"
OUT_DIR.mkdir(exist_ok=True)

# State FIPS codes to exclude (non-CONUS)
EXCLUDE_STATE_FIPS = {"02", "15", "72", "78", "66", "69", "60"}  # AK, HI, PR, VI, GU, MP, AS

# Target CRS and pixel size
TARGET_CRS = "EPSG:5070"
RESOLUTION = 1000  # meters ~ 1 km


def prepare_fips(geo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a 'fips' column (5-digit) from common county ID fields."""
    cols_lower = {c.lower(): c for c in geo.columns}

    if "geoid" in cols_lower:
        geo["fips"] = geo[cols_lower["geoid"]].astype(str).str.zfill(5)
    elif "geo_id" in cols_lower:
        geo["fips"] = geo[cols_lower["geo_id"]].astype(str).str[-5:]
    elif "fips" in cols_lower:
        geo["fips"] = geo[cols_lower["fips"]].astype(str).str.zfill(5)
    elif "statefp" in cols_lower and "countyfp" in cols_lower:
        s = geo[cols_lower["statefp"]].astype(str).str.zfill(2)
        c = geo[cols_lower["countyfp"]].astype(str).str.zfill(3)
        geo["fips"] = s + c
    else:
        raise ValueError("Could not find GEOID/FIPS fields in county geometry.")

    return geo

def build_grid_bounds(gdf: gpd.GeoDataFrame, resolution: float):
    """Compute raster grid shape + transform from GeoDataFrame bounds."""
    minx, miny, maxx, maxy = gdf.total_bounds
    width  = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    transform = from_origin(minx, maxy, resolution, resolution)
    return width, height, transform


def rasterize_column(gdf: gpd.GeoDataFrame, colname: str,
                     width: int, height: int, transform, out_path: Path):
    """Rasterize one attribute column to GeoTIFF."""
    print(f"Rasterizing {colname}...")

    # Prepare (geometry, value) tuples, skip NaNs
    shapes = []
    values = gdf[colname].astype(float)
    for geom, val in zip(gdf.geometry, values):
        if np.isnan(val):
            continue
        shapes.append((geom, float(val)))

    if not shapes:
        print(f"  ⚠ No valid data for {colname}, skipping.")
        return

    # Rasterize with NaN as nodata
    arr = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype="float32"
    )

    # Write GeoTIFF
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=TARGET_CRS,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(arr, 1)

 print(f"  ✅ Saved: {out_path.name}")


def main():
    print("Loading CSV and county geometry...")

    # Load attribute data
    df = pd.read_csv(DATA_CSV, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)

    # Load geometry and set CRS
    g = gpd.read_file(GEO_PATH, encoding="latin-1")
    g = prepare_fips(g)
    g = g.to_crs(TARGET_CRS)

    # Filter to CONUS counties only by state FIPS
    g["state_fips"] = g["fips"].str[:2]
    g_conus = g[~g["state_fips"].isin(EXCLUDE_STATE_FIPS)].copy()

    # Join attributes to CONUS counties
    mg = g_conus.merge(df, on="fips", how="left")
    

    print(f"  Counties in CONUS grid: {len(mg)}")

 print(f"  Counties in CONUS grid: {len(mg)}")

    # Build common raster grid over CONUS
    width, height, transform = build_grid_bounds(mg, RESOLUTION)
    print(f"  Raster grid: {width} x {height} pixels, {RESOLUTION} m resolution")

    # Select air-quality layers to rasterize
    cols_to_raster = [c for c in df.columns if c.startswith("intensity_")]
    print("  Columns to rasterize:")
    for c in cols_to_raster:
        print("   -", c)

    # Rasterize each column
    for col in cols_to_raster:
        out_path = OUT_DIR / f"{col}.tif"
        rasterize_column(mg, col, width, height, transform, out_path)

    print("\\nAll rasters generated in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

