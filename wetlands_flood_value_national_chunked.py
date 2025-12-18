#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wetlands Flood Mitigation Value (NLCD wetlands mask) + Public/Private split (PAD-US)
-------------------------------------------------------------------------------
This script:
  1) Reads a HUC12-level flood value table (CSV) and WBD HUC12 polygons.
  2) Rasterizes flood value onto the NLCD grid, but ONLY where NLCD classes are wetlands (90, 95).
  3) Uses PAD-US to attribute wetland area AND HUC12 flood value into PUBLIC vs PRIVATE.

Key design choices to keep this from crashing:
  - Process one state at a time (HUC12 subsets).
  - Process each state's NLCD window in tiles (default 2048).
  - DO NOT reproject the entire PAD-US layer (too big). Instead:
      * For each tile: read PAD-US features by bbox in PAD-US CRS (fast), then reproject that subset to NLCD CRS.

Outputs:
  - A single national raster: Wetlands_FloodMitigation_Value_US.tif (wetlands-only; float32; nodata=-9999)
  - Terminal logs:
      * per-state public/private wetland area fraction (pixel-based)
      * per-state public/private attributed flood value (HUC-weighted by wetland ownership fraction)
      * national totals at end

Notes:
  - This produces ONE raster (total flood value on wetlands). Public/private totals are printed (and can be saved to CSV).
  - Public/private attribution uses PAD-US Own_Type (or fallback). Adjust regex if needed.

Run:
  conda activate floodmap-311
  python /path/to/this_script.py
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window
from rasterio.errors import WindowError
from shapely.geometry import box
from shapely.errors import GEOSException
from pyproj import CRS, Transformer

# ---------------- USER CONFIG ----------------
TILE = 2048  # lower if you hit memory limits (1024); higher if you have lots of RAM (4096)

WBD_GDB = "/Users/jahnelle/Desktop/Flood_Maps/WBD_National_GDB/WBD_National.gdb"
WBD_LAYER = "WBDHU12"

PADUS_PATH = "/Users/jahnelle/Desktop/Flood_Maps/PADUS4_1Geodatabase.gdb"
PADUS_LAYER = "PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement"

# Ownership fields in your PAD-US layer (you printed these already)
PADUS_OWN_FIELD_PRIMARY = "Own_Type"
PADUS_OWN_FIELD_FALLBACK = "Own_Name"

# Robust classifier: treat anything containing these tokens as public
# (covers FED/FEDERAL, STATE, LOCAL, TRIBAL, etc.)
PUBLIC_REGEX = r"FED|STATE|LOCAL|TRIB|COUNTY|CITY|MUNIC|BOROUGH|PARISH"

CSV_PATH = "/Users/jahnelle/Desktop/Flood_Maps/huc_summary_forSNAPP_v2.csv"
CSV_HUC12_FIELD = "huc12"
CSV_VALUE_FIELD = "marginal_annualvalue_2020"  # or "marginal_npv_2020"

NLCD_PATH = "/Users/jahnelle/Desktop/Flood_Maps/Annual_NLCD_LndCov_2020_CU_C1V0/Annual_NLCD_LndCov_2020_CU_C1V0.tif"
WETLAND_CLASSES = [90, 95]

OUTPUT_TIF = "/Users/jahnelle/Desktop/Flood_Maps/Wetlands_FloodMitigation_Value_US.tif"
NODATA_OUT = -9999.0

# Set to a smaller list for testing (e.g. ["AL","GA","NC"]) then expand
STATE_CODES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY",
    "LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY",
    "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY","DC","PR"
]

# Optional: write a CSV summary of state totals
WRITE_STATE_SUMMARY_CSV = True
STATE_SUMMARY_OUT = "/Users/jahnelle/Desktop/Flood_Maps/flood_public_private_state_summary.csv"


# ---------------- HELPERS ----------------
def robust_read_vector(path, layer=None, bbox=None):
    """
    Read a vector layer robustly. bbox can be:
      - (minx, miny, maxx, maxy) in the layer CRS
    """
    try:
        return gpd.read_file(path, layer=layer, bbox=bbox)
    except TypeError:
        # some engines don't accept bbox kwarg
        return gpd.read_file(path, layer=layer)
    except Exception as e1:
        print("pyogrio failed, trying fiona:", e1, file=sys.stderr)
        return gpd.read_file(path, layer=layer, engine="fiona", bbox=bbox)


def init_output_like(src_path, out_path, nodata_val, tile=2048):
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        prof.update(dtype="float32", count=1, nodata=nodata_val, compress="lzw")
        width, height = src.width, src.height
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with rasterio.open(out_path, "w", **prof) as dst:
            for row_off in range(0, height, tile):
                h = min(tile, height - row_off)
                for col_off in range(0, width, tile):
                    w = min(tile, width - col_off)
                    win = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                    dst.write(np.full((h, w), nodata_val, dtype=np.float32), 1, window=win)


def normalize_states_column(wbd):
    for cand in ("STUSPS","ST","STATE","STATEABBR","ST_ABBREV"):
        if cand in wbd.columns:
            return wbd, cand, False
    for cand in ("States","STATES","states"):
        if cand in wbd.columns:
            s = wbd[cand].astype(str).str.upper().str.replace(" ", "", regex=False)
            wbd = wbd.copy()
            wbd["__states_list__"] = s.apply(lambda x: x.split(",") if x not in ("", "NONE", "NAN") else [])
            return wbd, "__states_list__", True
    raise ValueError("Could not find state/territory field (e.g., STUSPS or 'States') in WBDHU12.")


def fix_invalid_geoms(gdf):
    if gdf.empty:
        return gdf
    bad = ~gdf.geometry.is_valid
    if bad.any():
        gdf = gdf.copy()
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)
    return gdf


def classify_public(series):
    s = series.astype(str).str.upper()
    return s.str.contains(PUBLIC_REGEX, regex=True).to_numpy()


def bounds_in_crs(bounds, src_crs, dst_crs):
    """
    Transform (minx, miny, maxx, maxy) bounds from src_crs -> dst_crs.
    """
    transformer = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)
    minx, miny, maxx, maxy = bounds
    xs = [minx, minx, maxx, maxx]
    ys = [miny, maxy, miny, maxy]
    tx, ty = transformer.transform(xs, ys)
    return (min(tx), min(ty), max(tx), max(ty))


# ---------------- MAIN ----------------
def main():
    print(f"Opening NLCD (grid + wetlands classes): {NLCD_PATH}")
    with rasterio.open(NLCD_PATH) as src_nlcd:
        nlcd_crs = src_nlcd.crs
        nlcd_transform = src_nlcd.transform
        nlcd_width, nlcd_height = src_nlcd.width, src_nlcd.height

    print(f"Creating output raster initialized to NODATA at: {OUTPUT_TIF}")
    init_output_like(NLCD_PATH, OUTPUT_TIF, NODATA_OUT, tile=2048)

    print(f"Loading WBD layer: {WBD_GDB} (layer={WBD_LAYER})")
    wbd = robust_read_vector(WBD_GDB, layer=WBD_LAYER)
    print(f"  Loaded {len(wbd):,} HUC12 polygons.")

    # Normalize HUC12 name
    if "HUC12" in wbd.columns and CSV_HUC12_FIELD != "HUC12":
        wbd = wbd.rename(columns={"HUC12": CSV_HUC12_FIELD})
    if CSV_HUC12_FIELD not in wbd.columns:
        raise ValueError(f"HUC12 key '{CSV_HUC12_FIELD}' not found in WBD columns.")

    # Zero-pad HUC12 to 12 chars in WBD
    wbd[CSV_HUC12_FIELD] = wbd[CSV_HUC12_FIELD].astype(str).str.zfill(12)

    # Detect state field
    wbd, st_field, uses_list = normalize_states_column(wbd)
    print(f"  Using state field: {st_field} (list={uses_list})")

    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, dtype={CSV_HUC12_FIELD: str})
    needed = {CSV_HUC12_FIELD, CSV_VALUE_FIELD}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV missing columns: {miss}. Found: {list(df.columns)}")
    df[CSV_HUC12_FIELD] = df[CSV_HUC12_FIELD].astype(str).str.zfill(12)

    # CRS align WBD to NLCD
    if wbd.crs != nlcd_crs:
        print(f"Reprojecting WBD from {wbd.crs} → {nlcd_crs}")
        wbd = wbd.to_crs(nlcd_crs)

    # We'll read PAD-US in its native CRS, but subset by bbox and reproject per-tile.
    # Read just metadata (CRS) by reading 1 row.
    print(f"Checking PAD-US layer CRS: {PADUS_PATH} (layer={PADUS_LAYER})")
    padus_meta = robust_read_vector(PADUS_PATH, layer=PADUS_LAYER, bbox=None)
    if padus_meta.empty:
        raise ValueError("PAD-US read returned empty. Check PADUS_PATH and PADUS_LAYER.")
    padus_crs = padus_meta.crs
    if padus_crs is None:
        raise ValueError("PAD-US CRS is None. Please define/repair CRS for PAD-US layer.")
    del padus_meta

    # national accumulators
    national_pub_px = 0
    national_priv_px = 0
    national_pub_value = 0.0
    national_priv_value = 0.0

    state_rows = []

    print("Beginning state-by-state processing...")
    for st in STATE_CODES:
        if uses_list:
            mask = wbd[st_field].apply(lambda L: isinstance(L, (list, tuple)) and (st in L))
            sub = wbd[mask]
        else:
            sub = wbd[wbd[st_field].astype(str).str.upper() == st]

        if sub.empty:
            print(f"  {st}: no HUC polygons; skipping.")
            continue

        # Join CSV values onto HUC polygons
        sub = sub.merge(df[[CSV_HUC12_FIELD, CSV_VALUE_FIELD]], on=CSV_HUC12_FIELD, how="left")
        n_has_vals = int(sub[CSV_VALUE_FIELD].notna().sum())
        print(f"  {st}: {len(sub):,} HUC12 polygons (joined; {n_has_vals:,} with values).")

        # Bounds for this state subset in NLCD CRS
        minx, miny, maxx, maxy = sub.total_bounds

        # Determine raster window for the state bounds
        with rasterio.open(NLCD_PATH) as src_nlcd:
            try:
                win = from_bounds(minx, miny, maxx, maxy, transform=src_nlcd.transform)
                win = win.intersection(Window(0, 0, src_nlcd.width, src_nlcd.height))
            except WindowError:
                print(f"    {st}: outside NLCD raster; skipping.")
                continue

            if int(win.width) <= 0 or int(win.height) <= 0:
                print(f"    {st}: empty NLCD window; skipping.")
                continue

            win_h = int(win.height)
            win_w = int(win.width)

        # Clip HUCs to bbox for speed
        bbox_geom = box(minx, miny, maxx, maxy)
        try:
            sub_clip = gpd.clip(sub, bbox_geom)
        except GEOSException:
            sub = fix_invalid_geoms(sub)
            sub_clip = gpd.clip(sub, bbox_geom)

        sub_clip = sub_clip[sub_clip.geometry.notna()].copy()
        sub_clip = fix_invalid_geoms(sub_clip)

        # Factorize HUC ids to compact integer indices for bincount
        sub_clip = sub_clip[sub_clip[CSV_HUC12_FIELD].notna()].copy()
        sub_clip["__huc_idx__"], _ = pd.factorize(sub_clip[CSV_HUC12_FIELD], sort=False)

        n_hucs = int(sub_clip["__huc_idx__"].max()) + 1 if len(sub_clip) else 0
        if n_hucs == 0:
            print(f"    {st}: no valid HUC indices after clip; skipping.")
            continue

        # Map HUC idx -> total HUC value (from CSV join)
        idx_to_value = (
            sub_clip.drop_duplicates("__huc_idx__")[["__huc_idx__", CSV_VALUE_FIELD]]
            .set_index("__huc_idx__")[CSV_VALUE_FIELD]
            .fillna(0.0)
            .astype(float)
            .reindex(range(n_hucs), fill_value=0.0)
            .values
        )

        # Geometry->idx shapes (for tile rasterization)
        huc_shapes = []
        value_shapes = []
        for geom, idx, val in zip(sub_clip.geometry, sub_clip["__huc_idx__"], sub_clip[CSV_VALUE_FIELD]):
            if geom is None or geom.is_empty:
                continue
            huc_shapes.append((geom, int(idx)))
            v = float(val) if pd.notnull(val) else 0.0
            value_shapes.append((geom, v))

        # State-level bincounts for ownership fraction
        tot_counts_state = np.zeros(n_hucs, dtype=np.float64)
        pub_counts_state = np.zeros(n_hucs, dtype=np.float64)

        # State-level area totals (wetland pixels)
        st_pub_px = 0
        st_priv_px = 0

        # Process state window in tiles
        with rasterio.open(NLCD_PATH) as src_nlcd:
            for r0 in range(0, win_h, TILE):
                rh = min(TILE, win_h - r0)
                for c0 in range(0, win_w, TILE):
                    cw = min(TILE, win_w - c0)

                    tile_win = Window(
                        col_off=win.col_off + c0,
                        row_off=win.row_off + r0,
                        width=cw,
                        height=rh,
                    )
                    tile_transform = src_nlcd.window_transform(tile_win)

                    wetlands_tile = src_nlcd.read(1, window=tile_win)
                    wet_mask = np.isin(wetlands_tile, WETLAND_CLASSES)
                    if not np.any(wet_mask):
                        continue

                    # Tile bounds in NLCD CRS -> transform to PAD-US CRS for bbox read
                    tile_bounds_nlcd = rasterio.windows.bounds(tile_win, nlcd_transform)
                    tile_bounds_padus = bounds_in_crs(tile_bounds_nlcd, nlcd_crs, padus_crs)

                    # Read PAD-US subset by bbox (fast), then reproject subset to NLCD CRS
                    padus_sub = robust_read_vector(PADUS_PATH, layer=PADUS_LAYER, bbox=tile_bounds_padus)
                    if not padus_sub.empty:
                        padus_sub = padus_sub[padus_sub.geometry.notna()].copy()
                        padus_sub = fix_invalid_geoms(padus_sub)
                        if padus_sub.crs != nlcd_crs:
                            padus_sub = padus_sub.to_crs(nlcd_crs)

                    # Build public raster mask (1=public, 0=not public)
                    if padus_sub.empty:
                        public_ras = np.zeros((rh, cw), dtype=np.uint8)
                    else:
                        if PADUS_OWN_FIELD_PRIMARY in padus_sub.columns and padus_sub[PADUS_OWN_FIELD_PRIMARY].notna().any():
                            own_series = padus_sub[PADUS_OWN_FIELD_PRIMARY]
                        else:
                            own_series = padus_sub.get(PADUS_OWN_FIELD_FALLBACK, pd.Series([], dtype=object))

                        is_public = classify_public(own_series) if len(own_series) else np.array([], dtype=bool)

                        public_shapes = []
                        for geom, flag in zip(padus_sub.geometry, is_public):
                            if flag and geom is not None and (not geom.is_empty):
                                public_shapes.append((geom, 1))

                        if not public_shapes:
                            public_ras = np.zeros((rh, cw), dtype=np.uint8)
                        else:
                            public_ras = rasterize(
                                public_shapes,
                                out_shape=(rh, cw),
                                transform=tile_transform,
                                fill=0,
                                dtype="uint8",
                            )

                    pub_wet = (public_ras == 1) & wet_mask
                    priv_wet = (public_ras == 0) & wet_mask

                    # Pixel (area) totals
                    ppx = int(np.count_nonzero(pub_wet))
                    qpx = int(np.count_nonzero(priv_wet))
                    st_pub_px += ppx
                    st_priv_px += qpx

                    # Rasterize HUC indices for this tile and update per-HUC counts
                    huc_idx_tile = rasterize(
                        huc_shapes,
                        out_shape=(rh, cw),
                        transform=tile_transform,
                        fill=-1,
                        dtype="int32",
                    )

                    valid = wet_mask & (huc_idx_tile >= 0)
                    if np.any(valid):
                        counts = np.bincount(huc_idx_tile[valid].ravel(), minlength=n_hucs).astype(np.float64)
                        tot_counts_state[:len(counts)] += counts

                        valid_pub = pub_wet & (huc_idx_tile >= 0)
                        if np.any(valid_pub):
                            counts_pub = np.bincount(huc_idx_tile[valid_pub].ravel(), minlength=n_hucs).astype(np.float64)
                            pub_counts_state[:len(counts_pub)] += counts_pub

                    # Rasterize total flood value for this tile and write wetlands-only to output raster
                    ras_tile = rasterize(
                        value_shapes,
                        out_shape=(rh, cw),
                        transform=tile_transform,
                        fill=NODATA_OUT,
                        dtype="float32",
                    ).astype(np.float32)

                    ras_masked = np.where(wet_mask, ras_tile, NODATA_OUT).astype(np.float32)

                    with rasterio.open(OUTPUT_TIF, "r+") as dst_out:
                        existing = dst_out.read(1, window=tile_win)
                        update = np.where(
                            (existing == NODATA_OUT) & (ras_masked != NODATA_OUT),
                            ras_masked,
                            existing.astype(np.float32),
                        ).astype(np.float32)
                        dst_out.write(update, 1, window=tile_win)

        # State: compute $ split via per-HUC public fraction
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_pub = np.where(tot_counts_state > 0, pub_counts_state / tot_counts_state, 0.0)

        st_pub_value = float(np.sum(idx_to_value * frac_pub))
        st_priv_value = float(np.sum(idx_to_value * (1.0 - frac_pub)))

        national_pub_px += st_pub_px
        national_priv_px += st_priv_px
        national_pub_value += st_pub_value
        national_priv_value += st_priv_value

        tot_px = st_pub_px + st_priv_px
        if tot_px > 0:
            print(f"    {st}: wetland ownership (area) → public {st_pub_px/tot_px:.2%}, private {st_priv_px/tot_px:.2%}")
        else:
            print(f"    {st}: wetland ownership (area) → no wetlands counted")

        print(f"    {st}: flood value attributed → public ${st_pub_value:,.0f}, private ${st_priv_value:,.0f}")

        state_rows.append({
            "state": st,
            "wetland_pub_px": st_pub_px,
            "wetland_priv_px": st_priv_px,
            "wetland_pub_frac": (st_pub_px / tot_px) if tot_px > 0 else np.nan,
            "wetland_priv_frac": (st_priv_px / tot_px) if tot_px > 0 else np.nan,
            "flood_value_public": st_pub_value,
            "flood_value_private": st_priv_value,
            "n_hucs": n_hucs,
            "n_hucs_with_values": int(np.count_nonzero(idx_to_value)),
        })

    # National totals
    tot_px = national_pub_px + national_priv_px
    print("\n===== NATIONAL TOTALS =====")
    if tot_px > 0:
        print(f"Wetland ownership (area): public {national_pub_px/tot_px:.2%}, private {national_priv_px/tot_px:.2%}")
    else:
        print("Wetland ownership (area): no wetlands counted.")

    print(f"Flood value attributed: public ${national_pub_value:,.0f}, private ${national_priv_value:,.0f}")
    print(f"✅ Finished. Output written to: {OUTPUT_TIF}")

    if WRITE_STATE_SUMMARY_CSV and state_rows:
        out = pd.DataFrame(state_rows)
        out.to_csv(STATE_SUMMARY_OUT, index=False)
        print(f"✅ State summary CSV written to: {STATE_SUMMARY_OUT}")


if __name__ == "__main__":
    main()
