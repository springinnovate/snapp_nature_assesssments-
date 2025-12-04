import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import numpy as np

# ======================================================
# CONUS maps (log-scale YlOrRd) + corrected national totals
# ------------------------------------------------------
# CHANGE: National totals now use a DIRECT SUM over counties,
#         with NO area weighting (fixes "too many deaths" issue
#         caused by double-applying area to already-integrated values).
# ------------------------------------------------------
# What stays the same:
#   - Only PNGs
#   - Log scale
#   - YlOrRd palette
#   - Titles unchanged from your requested wording
#   - CONUS only (excludes AK, HI, PR, VI, GU, MP, AS)
#   - Headline totals use only roll-up columns beginning with
#     'intensity_landscape_all_' (to avoid double counting)
# ======================================================

INCLUDE_TERRITORIES = False

PALETTE = "YlOrRd"
FIGSIZE = (14, 9)
FONT_SIZE = 11
COUNTY_LW = 0.1
PCT_CLIP = (5, 99)

# <<<---- EDIT THESE PATHS IF NEEDED
DATA_CSV = Path("/Users/jahnelle/Desktop/Deposition_Maps/deposition_intensity_per_km2_by_env.csv")
GEO_PATH = Path("/Users/jahnelle/Desktop/Deposition_Maps/gz_2010_us_050_00_5m.json")
OUT_DIR  = Path("/Users/jahnelle/Desktop/Deposition_Maps/map_outputs")
# ----->>>

OUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------
# Helpers
# ---------------------
def _prepare_fips(geo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
        raise ValueError("Could not find GEOID/fips fields in county geometry.")
    return geo


def _draw_one(mg: gpd.GeoDataFrame, col: str, title: str, out_png: Path, cmap: str = PALETTE):
    plt.rcParams.update({"font.size": FONT_SIZE})

    series = mg[col].astype(float)
    series = series.where(series > 0, np.nan)
    if series.dropna().empty:
        print(f"⚠️  Skipping {col}: no positive values")
        return False

    vmin, vmax = np.nanpercentile(series, PCT_CLIP[0]), np.nanpercentile(series, PCT_CLIP[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0 or vmin >= vmax:
        print(f"⚠️  Skipping {col}: invalid range (vmin={vmin}, vmax={vmax})")
        return False

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    mg.plot(column=col, ax=ax, cmap=cmap, norm=norm,
            linewidth=COUNTY_LW, edgecolor="black", legend=True,
            missing_kwds={"color": "lightgrey", "alpha": 0.4})
    ax.set_axis_off()
    ax.set_title(title, fontsize=FONT_SIZE + 1)

    # Zoom to bounds
    minx, miny, maxx, maxy = mg.total_bounds
    pad_x = 0.02 * (maxx - minx)
    pad_y = 0.02 * (maxy - miny)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    fig.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {out_png.name}")
    return True


def _direct_sum_over_counties(mg: gpd.GeoDataFrame, df: pd.DataFrame, colnames):
    """Sum county values directly (NO area weighting)."""
    if not colnames:
        return 0.0

joined = mg[["fips"]].merge(df[["fips"] + colnames], on="fips", how="left")
    total = 0.0
    for c in colnames:
        total += np.nansum(joined[c].astype(float))
    return float(total)


# ---------------------
# Main
# ---------------------
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_CSV, dtype={"fips": str})
    g = gpd.read_file(GEO_PATH, encoding="latin-1")
    g = _prepare_fips(g)

    if not INCLUDE_TERRITORIES:
        exclude = {"02","15","72","78","66","69","60"}  # AK, HI, PR, VI, GU, MP, AS
        g = g[~g["fips"].str[:2].isin(exclude)].copy()

    # Join geometry to attributes
    mg = g.merge(df, on="fips", how="left")

    # ---------------- Render maps (unchanged titles) ----------------
    metric_cols = [c for c in df.columns if c.startswith("intensity_")]
    for col in metric_cols:
        name = col.lower()
        if ("valuation" in name) and ("usd" in name):
            title = "Nature’s contribution to avoided mortality cost (USD/km²)"
        elif "total_deaths_per_km2" in name:
            title = "Total avoided deaths per km² due to nature-based air pollutant removal (PM₂.₅/NO₂/SO₂)"
        elif ("_pm25_" in name):
            title = "Avoided deaths per km² due to nature-based air pollutant removal (PM₂.₅)"
        elif ("_no2_" in name) or ("_nox_" in name):
            title = "Avoided deaths per km² due to nature-based air pollutant removal (NO₂)"
        elif ("_so2_" in name) or ("_sox_" in name):
            title = "Avoided deaths per km² due to nature-based air pollutant removal (SO₂)"
        else:
            title = col.replace("_", " ")
        out_png = OUT_DIR / f"{col}.log.clean.png"
        _ = _draw_one(mg, col, title, out_png, cmap=PALETTE)

    # --------------- Headline totals (roll-up only, DIRECT SUM) ---------------
    rollup_prefix = "intensity_landscape_all_"
    # deaths: prefer explicit total_deaths_per_km2
    rollup_total_deaths_cols = [c for c in df.columns
                                if c.startswith(rollup_prefix) and "total_deaths_per_km2" in c]
    # if no explicit total, sum per-pollutant once
    rollup_pm_cols = [c for c in df.columns
                      if c.startswith(rollup_prefix) and "_pm25_" in c and c.endswith("deaths_per_km2")]
    rollup_nx_cols = [c for c in df.columns
                      if c.startswith(rollup_prefix) and (("_no2_" in c) or ("_nox_" in c)) and c.endswith("deaths_per_km2")]
    rollup_sx_cols = [c for c in df.columns
                      if c.startswith(rollup_prefix) and (("_so2_" in c) or ("_sox_" in c)) and c.endswith("deaths_per_km2")]
    # valuation USD (any year tag like 2006usd)
    rollup_val_cols = [c for c in df.columns
                       if c.startswith(rollup_prefix) and ("valuation" in c.lower()) and ("usd" in c.lower())]

    if rollup_total_deaths_cols:
total_avoided_deaths_all = _direct_sum_over_counties(mg, df, rollup_total_deaths_cols)
    else:
        total_avoided_deaths_all = _direct_sum_over_counties(mg, df, rollup_pm_cols + rollup_nx_cols + rollup_sx_cols)

    total_avoided_usd_all = _direct_sum_over_counties(mg, df, rollup_val_cols)

    # Save totals to CSV
    summary_path = OUT_DIR.parent / "summary_totals.csv"
    pd.DataFrame({
        "metric": ["total_avoided_deaths_CONUS_rollup_direct_sum",
                   "total_avoided_mortality_cost_CONUS_usd_rollup_direct_sum"],
        "value": [total_avoided_deaths_all, total_avoided_usd_all]
    }).to_csv(summary_path, index=False)

    print("--------------------------------------------------")
    print(f"🌿 Total avoided deaths (CONUS, roll-up, direct sum): {total_avoided_deaths_all:,.2f}")
    print(f"💰 Total avoided mortality cost (CONUS, USD roll-up, direct sum): ${total_avoided_usd_all:,.2f}")
    print(f"All PNGs saved in: {OUT_DIR.resolve()}")
    print(f"Summary CSV saved to: {summary_path.resolve()}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()

