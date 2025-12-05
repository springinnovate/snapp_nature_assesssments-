# Compute county-level *per km²* deposition-only health impacts (no land-cover area provided).
# This produces intensity metrics that can be multiplied by area later when available.
import pandas as pd
import numpy as np
from pathlib import Path
from caas_jupyter_tools import display_dataframe_to_user

# Inputs
DEP_LANDSCAPE = Path("/mnt/data/Landscape_air_pollutant_removal_ranges.csv")
DEP_SHRUB     = Path("/mnt/data/Shrubland_Landscape_air_pollutant_removal_ranges.csv")
DEP_GRASS     = Path("/mnt/data/Grassland_Landscape_air_pollutant_removal_ranges.csv")
RCM_PATH      = Path("/mnt/data/rcm-gemm.csv")
OUT_DIR = Path("/mnt/data/outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def detect_pollutant_cols(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    pm_candidates = [c for c in cols if c.lower().replace("_","").replace(".","").startswith("pm25")] + \
                    [c for c in cols if "pm2.5" in c.lower() or "pm2_5" in c.lower()]
    pm_col = next((c for c in pm_candidates if "mean" in c.lower()), None) or (pm_candidates[0] if pm_candidates else None)
    no2_candidates = [c for c in cols if c.lower().startswith("no2")]
    no2_col = next((c for c in no2_candidates if "mean" in c.lower()), None) or (no2_candidates[0] if no2_candidates else None)
    so2_candidates = [c for c in cols if c.lower().startswith("so2")]
    so2_col = next((c for c in so2_candidates if "mean" in c.lower()), None) or (so2_candidates[0] if so2_candidates else None)
    return {"pm25": pm_col, "nox": no2_col, "sox": so2_col}

def load_dep_table(path: Path, env_label: str):
    df = pd.read_csv(path)
    # Normalize FIPS from either FIPS or fips
    if "FIPS" in df.columns:
        df["fips"] = df["FIPS"].astype(str).str.zfill(5)
    elif "fips" in df.columns:
        df["fips"] = df["fips"].astype(str).str.zfill(5)
    else:
        raise ValueError(f"{path.name} is missing a FIPS/fips column.")
    # Ensure state/county name columns exist (fill if missing)
    for name_col in ["PrimaryPartitionName","SecondaryPartitionName"]:
        if name_col not in df.columns:
            df[name_col] = np.nan
    df["env_type"] = env_label
    polcols = detect_pollutant_cols(df)
    df = df.rename(columns={polcols["pm25"]:"PM25_val",
                            polcols["nox"] :"NO2_val",
                            polcols["sox"] :"SO2_val"})
    df["pm25_col_used"] = polcols["pm25"]
    df["nox_col_used"]  = polcols["nox"]
    df["sox_col_used"]  = polcols["sox"]
    return df[["fips","PrimaryPartitionName","SecondaryPartitionName","env_type",
               "PM25_val","NO2_val","SO2_val","pm25_col_used","nox_col_used","sox_col_used"]]

# Load deposition data
dep_land = load_dep_table(DEP_LANDSCAPE, "landscape_all")
dep_shrb = load_dep_table(DEP_SHRUB, "shrubland")
dep_gras = load_dep_table(DEP_GRASS, "grassland")
dep_all = pd.concat([dep_land, dep_shrb, dep_gras], ignore_index=True)

# Load RCM
rcm = pd.read_csv(RCM_PATH)
rcm = rcm.rename(columns={c: c.lower() for c in rcm.columns})
rcm["fips"] = rcm["fips"].astype(str).str.zfill(5)

def norm_pol(x: str):
    xl = str(x).lower()
    if "pm" in xl and "2.5" in xl or "pm2.5" in xl or xl == "pm25":
        return "pm25"
    if "no2" in xl or "nox" in xl:
        return "nox"
    if "so2" in xl or "sox" in xl:
        return "sox"
    return xl

rcm["pol_norm"] = rcm["pollutant"].map(norm_pol)
model_col = "model" if "model" in rcm.columns else None
crf_col = "crf" if "crf" in rcm.columns else None
if model_col and crf_col:
    mode_pair = rcm.groupby([model_col, crf_col]).size().sort_values(ascending=False).index[0]
    sel_model, sel_crf = mode_pair[0], mode_pair[1]
    rcm_sel = rcm[(rcm[model_col] == sel_model) & (rcm[crf_col] == sel_crf)].copy()
else:
    sel_model = rcm[model_col].iloc[0] if model_col else "unknown"
    sel_crf = rcm[crf_col].iloc[0] if crf_col else "unknown"
    rcm_sel = rcm.copy()

rcm_keep = rcm_sel[["fips","pol_norm","mortality"]].copy()
rcm_wide = rcm_keep.pivot_table(index="fips", columns="pol_norm", values="mortality", aggfunc="mean").reset_index()
for pol in ["pm25","nox","sox"]:
    if pol not in rcm_wide.columns:
        rcm_wide[pol] = np.nan

# Build a *per km²* impact intensity table (no area scaling)
# intensity_per_km2 = removal_rate(g/m2) * 1000 * mortality
# (mirrors your code's unit factor; if you prefer explicit km²->m², we can adjust to 1e6 * kg conversion)
def intensities_for_env(env_df: pd.DataFrame, env_label: str) -> pd.DataFrame:
    sub = env_df[["fips","PrimaryPartitionName","SecondaryPartitionName","PM25_val","NO2_val","SO2_val"]].copy()
    sub = sub.merge(rcm_wide, on="fips", how="left", suffixes=("","_mort"))
    # Compute per-pol intensities
    sub[f"intensity_{env_label}_pm25_deaths_per_km2"] = sub["PM25_val"] * 1000.0 * sub["pm25"]
    sub[f"intensity_{env_label}_nox_deaths_per_km2"]  = sub["NO2_val"]  * 1000.0 * sub["nox"]
    sub[f"intensity_{env_label}_sox_deaths_per_km2"]  = sub["SO2_val"]  * 1000.0 * sub["sox"]
    # totals per env
    totcols = [f"intensity_{env_label}_pm25_deaths_per_km2",
               f"intensity_{env_label}_nox_deaths_per_km2",
               f"intensity_{env_label}_sox_deaths_per_km2"]
    sub[f"intensity_{env_label}_total_deaths_per_km2"] = sub[totcols].sum(axis=1, skipna=True)
    # valuation per km²
    VSL_2006_USD = 7_400_000
    sub[f"intensity_{env_label}_valuation_2006usd_per_km2"] = sub[f"intensity_{env_label}_total_deaths_per_km2"] * VSL_2006_USD
    # keep core id columns once
    return sub

int_land = intensities_for_env(dep_land, "landscape_all")
int_shrb = intensities_for_env(dep_shrb, "shrubland")
int_gras = intensities_for_env(dep_gras, "grassland")

# Merge the three env intensity tables on FIPS
cols_keep = ["fips","PrimaryPartitionName","SecondaryPartitionName"]
int_all = int_land.merge(int_shrb.drop(columns=cols_keep), left_index=False, right_index=False, how="outer", on=None)
# ensure we keep fips/state/county from landscape table; fallback to shrub/grass if missing
int_all = int_land[cols_keep].merge(int_all, left_index=False, right_index=False, how="right", on=cols_keep)

# Save outputs
out_csv = OUT_DIR / "deposition_intensity_per_km2_by_env.csv"
int_all.to_csv(out_csv, index=False)

# Display a preview
display_dataframe_to_user("Per-km² deposition-only impact intensities (by environment)", int_all.head(20))

str(out_csv)
