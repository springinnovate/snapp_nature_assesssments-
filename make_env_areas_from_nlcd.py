import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import mask
import numpy as np
from pathlib import Path

# ==============================================
# BUILD COUNTY-LEVEL ENV AREA TABLE FROM NLCD2020
# ==============================================
# Requirements:
#   - geopandas
#   - rasterio
#
# Outputs:
#   nlcd_2020_envarea_by_county.csv
#
# Steps:
#   1. Load county boundaries (same as used for rasters)
#   2. Load NLCD 2020 land cover raster
#   3. For each county polygon:
#       - mask NLCD to that county
#       - count pixels of each NLCD class
#       - convert to km² and aggregate to:
#           * forest (41, 42, 43)
#           * shrubland (52)
#           * grassland/herbaceous (71)
# ==============================================

# ---- EDIT THESE PATHS IF NEEDED ----
BASE_DIR   = Path("/Users/jahnelle/Desktop/Deposition_Maps")
COUNTIES   = BASE_DIR / "gz_2010_us_050_00_5m.json"  # county boundaries you already have
NLCD_PATH  = BASE_DIR / "Annual_NLCD_LndCov_2020_CU_C1V0.tif"  # <-- EDIT if your file name is different
OUT_CSV    = BASE_DIR / "nlcd_2020_envarea_by_county.csv"

# Non-CONUS states to exclude if you want strict CONUS
EXCLUDE_STATE_FIPS = {"02", "15", "72", "78", "66", "69", "60"}  # AK, HI, PR, VI, GU, MP, AS

# NLCD code groupings for env types (adjust if needed)
FOREST_CODES    = [41, 42, 43]
SHRUB_CODES     = [52]
GRASSLAND_CODES = [71]   # you can add 81/82 here if you want to treat some ag as grassland


def prepare_fips(geo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create a 'fips' column (5-digit) from common fields."""
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


def main():
    print("Loading county boundaries...")
    g = gpd.read_file(COUNTIES, encoding="latin-1")
    g = prepare_fips(g)

    # Filter to CONUS if desired
    g["state_fips"] = g["fips"].str[:2]
    g = g[~g["state_fips"].isin(EXCLUDE_STATE_FIPS)].copy()
    g = g.drop(columns=["state_fips"])

    print(f"  Counties in CONUS: {len(g)}")

    print("Loading NLCD raster...")
    with rasterio.open(NLCD_PATH) as src:
        nlcd_crs = src.crs
        nlcd_transform = src.transform
        nlcd_nodata = src.nodata
        # Pixel size (assumes square pixels)
        pixel_width, pixel_height = src.res
        pixel_area_m2 = abs(pixel_width * pixel_height)

        print(f"  NLCD CRS: {nlcd_crs}")
        print(f"  Pixel size: {pixel_width} x {pixel_height} m (area {pixel_area_m2} m²)")

        # Reproject counties to NLCD CRS
        g = g.to_crs(nlcd_crs)

        # Prepare result lists
        rows = []

        # Loop over counties
        for idx, row in g.iterrows():
            fips = row["fips"]
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            try:
                # Mask NLCD to this county geometry
                out_image, out_transform = mask.mask(src, [geom], crop=True)
            except Exception as e:
                print(f"  ⚠ Error masking FIPS {fips}: {e}")
                continue

            data = out_image[0]  # first (and only) band

            # Mask out nodata
            if nlcd_nodata is not None:
                data = np.where(data == nlcd_nodata, np.nan, data)

            # Flatten and drop NaNs
            flat = data.flatten()
            flat = flat[~np.isnan(flat)]

            if flat.size == 0:
                forest_area_km2 = shrub_area_km2 = grass_area_km2 = 0.0
            else:
                vals, counts = np.unique(flat.astype(int), return_counts=True)
                # Build a dict mapping NLCD code -> pixel count
                count_dict = {int(v): int(c) for v, c in zip(vals, counts)}

                # Sum relevant codes for each env type
                forest_pixels = sum(count_dict.get(code, 0) for code in FOREST_CODES)
                shrub_pixels  = sum(count_dict.get(code, 0) for code in SHRUB_CODES)
                grass_pixels  = sum(count_dict.get(code, 0) for code in GRASSLAND_CODES)

                # Convert pixel counts to area in km²
                forest_area_km2 = forest_pixels * pixel_area_m2 / 1_000_000.0
                shrub_area_km2  = shrub_pixels  * pixel_area_m2 / 1_000_000.0
                grass_area_km2  = grass_pixels  * pixel_area_m2 / 1_000_000.0

            rows.append({
                "fips": fips,
                "forest_area_km2": forest_area_km2,
                "shrubland_area_km2": shrub_area_km2,
                "grassland_area_km2": grass_area_km2
            })

            if len(rows) % 200 == 0:
                print(f"  Processed {len(rows)} counties...")

    # Build DataFrame and save
    df = pd.DataFrame(rows)
    df.sort_values("fips", inplace=True)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved county env area table to:")
    print(OUT_CSV.resolve())


if __name__ == "__main__":
    main()
