#!/usr/bin/env python3
"""
make_env_areas_from_nlcd_padus4.py

Computes county-level NLCD 2020 areas (km²) for:
  - forest (NLCD 41/42/43)
  - shrubland (NLCD 52)
  - grassland (NLCD 71)   # NOTE: if you want pasture/hay too, add 81 below

Splits each class into PUBLIC vs PRIVATE using PAD-US 4 polygons.

Output:
  nlcd_2020_envarea_by_county_with_ownership.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, transform as win_transform


# =========================
# EDIT THESE PATHS
# =========================
WORKDIR = Path("/Users/jahnelle/Desktop/Deposition_Maps")

COUNTIES_PATH = Path("/Users/jahnelle/Desktop/Deposition_Maps/cb_2018_us_county_5m.shp")   # your counties
NLCD_TIF     = WORKDIR / "Annual_NLCD_LndCov_2020_CU_C1V0.tif"          # <-- rename to your exact NLCD tif
PADUS_GDB   = Path("/Users/jahnelle/Desktop/Deposition_Maps/PADUS4_1Geodatabase.gdb")
PADUS_LAYER = "PADUS4_1Fee"  
pad = gpd.read_file(PADUS_GDB, layer=PADUS_LAYER)


OUT_CSV      = WORKDIR / "nlcd_2020_envarea_by_county_with_ownership.csv"


# =========================
# OPTIONAL PAD-US FILTER
# =========================
# If your PAD-US file already only includes "public" lands, leave FILTER_EXPR = None.
# If you loaded a broader PAD-US layer and need to filter, set a pandas query string.
# Example (ONLY if those columns exist in your PAD-US file):
#   FILTER_EXPR = "Own_Type == 'Federal' or Own_Type == 'State' or Own_Type == 'Local'"
FILTER_EXPR = None


# =========================
# NLCD CLASS DEFINITIONS
# =========================
FOREST = {41, 42, 43}
SHRUB  = {52}
GRASS  = {71}  # consider adding 81 if you want pasture/hay: {71, 81}


# CONUS filter (optional): remove AK, HI, territories by STATEFP
EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}  # AK, HI, AS, GU, MP, PR, VI


def main():
    print("Loading county boundaries...")
    counties = gpd.read_file(COUNTIES_PATH, engine="fiona")

    # Ensure county GEOID -> fips (required for joining + output)
    if "GEOID" in counties.columns:
        counties["fips"] = counties["GEOID"].astype(str).str.zfill(5)
    elif "STATEFP" in counties.columns and "COUNTYFP" in counties.columns:
        counties["fips"] = (
            counties["STATEFP"].astype(str).str.zfill(2) +
            counties["COUNTYFP"].astype(str).str.zfill(3)
        )
    else:
        raise ValueError("County file must have GEOID or STATEFP+COUNTYFP.")


       # --- Build 5-digit county FIPS robustly across common Census schemas ---
    cols = set(counties.columns)

    if "STATEFP" in cols and "COUNTYFP" in cols:
        counties["fips"] = (counties["STATEFP"].astype(str).str.zfill(2) +
                            counties["COUNTYFP"].astype(str).str.zfill(3))
        counties = counties[~counties["STATEFP"].astype(str).str.zfill(2).isin(EXCLUDE_STATEFP)].copy()

    elif "GEOID" in cols:
        counties["fips"] = counties["GEOID"].astype(str).str.zfill(5)
        counties = counties[~counties["fips"].str[:2].isin(EXCLUDE_STATEFP)].copy()

    elif "GEO_ID" in cols:
        # Often like: 0500000US01001  -> take last 5 digits
        counties["fips"] = counties["GEO_ID"].astype(str).str.extract(r"(\\d{5})$")[0]
        counties["fips"] = counties["fips"].astype(str).str.zfill(5)
        counties = counties[~counties["fips"].str[:2].isin(EXCLUDE_STATEFP)].copy()

    elif "STATE" in cols and "COUNTY" in cols:
        # gz_2010_us_050_00_5m.shp style
        counties["fips"] = (counties["STATE"].astype(int).astype(str).str.zfill(2) +
                            counties["COUNTY"].astype(int).astype(str).str.zfill(3))
        counties = counties[~counties["fips"].str[:2].isin(EXCLUDE_STATEFP)].copy()

    else:
        print("County columns are:", list(counties.columns))
        raise ValueError("County file must have STATEFP+COUNTYFP, GEOID, GEO_ID, or STATE+COUNTY.")


    print(f"  Counties in CONUS: {len(counties):,}")

    print("Loading PAD-US 4 polygons...")
    pad = gpd.read_file(PADUS_GDB, layer=PADUS_LAYER)
    if FILTER_EXPR:
        try:
            pad = pad.query(FILTER_EXPR).copy()
            print(f"  PAD-US filtered with: {FILTER_EXPR}")
            print(f"  Remaining PAD-US features: {len(pad):,}")
        except Exception as e:
            raise RuntimeError(f"PAD-US FILTER_EXPR failed: {e}")

    # Build spatial index (for speed)
    pad_sindex = pad.sindex

    print("Loading NLCD raster...")
    with rasterio.open(NLCD_TIF) as src:
        print(f"  NLCD CRS: {src.crs}")
        px_area_km2 = abs(src.res[0] * src.res[1]) / 1e6
        print(f"  Pixel size: {src.res[0]} x {src.res[1]} m (area {px_area_km2*1e6:.1f} m²)")

        # Reproject vectors to NLCD CRS
        counties = counties.to_crs(src.crs)
        pad = pad.to_crs(src.crs)

        rows = []
        for idx, row in enumerate(counties.itertuples(index=False), start=1):
            geom = row.geometry
            fips = row.fips

            # Window for this county
            b = geom.bounds
            win = from_bounds(*b, transform=src.transform)
            win = win.round_offsets().round_lengths()

            nlcd = src.read(1, window=win, masked=True)
            if nlcd.size == 0:
                continue

            w_transform = win_transform(win, src.transform)

            # County mask in window grid
            county_mask = rasterize(
                [(geom, 1)],
                out_shape=nlcd.shape,
                transform=w_transform,
                fill=0,
                dtype="uint8",
                all_touched=False
            ).astype(bool)

            if not county_mask.any():
                continue

            # Find PAD-US features intersecting county bbox (fast)
            cand_idx = list(pad_sindex.intersection(geom.bounds))
            if cand_idx:
                pad_clip = pad.iloc[cand_idx]
                pad_clip = pad_clip[pad_clip.intersects(geom)]
            else:
                pad_clip = pad.iloc[0:0]

            # Public mask in same window grid
            if len(pad_clip) > 0:
                pub_mask = rasterize(
                    [(g, 1) for g in pad_clip.geometry],
                    out_shape=nlcd.shape,
                    transform=w_transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False
                ).astype(bool)
            else:
                pub_mask = np.zeros(nlcd.shape, dtype=bool)

            # Define private mask inside county
            priv_mask = county_mask & (~pub_mask)

            # Helper: area by class set and mask
            def area_km2(classes, mask):
                vals = nlcd[mask]
                if hasattr(vals, "filled"):
                    vals = vals.filled(0)
                return float(np.isin(vals, list(classes)).sum() * px_area_km2)

            forest_all = area_km2(FOREST, county_mask)
            shrub_all  = area_km2(SHRUB,  county_mask)
            grass_all  = area_km2(GRASS,  county_mask)

            forest_pub = area_km2(FOREST, county_mask & pub_mask)
            shrub_pub  = area_km2(SHRUB,  county_mask & pub_mask)
            grass_pub  = area_km2(GRASS,  county_mask & pub_mask)

            forest_prv = area_km2(FOREST, priv_mask)
            shrub_prv  = area_km2(SHRUB,  priv_mask)
            grass_prv  = area_km2(GRASS,  priv_mask)

            rows.append({
                "fips": fips,
                "forest_area_km2": forest_all,
                "shrubland_area_km2": shrub_all,
                "grassland_area_km2": grass_all,
                "forest_area_public_km2": forest_pub,
                "shrubland_area_public_km2": shrub_pub,
                "grassland_area_public_km2": grass_pub,
                "forest_area_private_km2": forest_prv,
                "shrubland_area_private_km2": shrub_prv,
                "grassland_area_private_km2": grass_prv,
            })

            if idx % 200 == 0:
                print(f"  Processed {idx} counties...")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
