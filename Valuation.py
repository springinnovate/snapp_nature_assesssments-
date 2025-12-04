import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# Air Quality Totals Breakdown (CONUS, Direct Sum)
# ------------------------------------------------------
# Reads the deposition_intensity_per_km2_by_env.csv and prints:
#   Total avoided deaths (all pollutants, CONUS)
#   Total avoided mortality cost (2006 USD, VSL=7.4M)
#      PM2.5 deaths  -> $...
#      NOx   deaths  -> $...
#      SOx   deaths  -> $...
#
# Notes
# - Uses *direct sums over counties* (NO area weighting).
# - Filters to contiguous US by FIPS state code (excludes AK, HI, PR, VI, GU, MP, AS).
# - Uses only the roll-up layer: columns starting with 'intensity_landscape_all_'.
# - If no explicit roll-up 'total_deaths_per_km2' exists, sums PM+NOx+SOx once.
# - Costs computed via VSL = 7.4e6 (2006 USD) to match workplan language.
# - Also saves a CSV 'aq_breakdown_totals.csv' with the same numbers.
# ======================================================

# ---- USER PATHS (edit if needed) ----
DATA_CSV = Path("/Users/jahnelle/Desktop/Deposition_Maps/deposition_intensity_per_km2_by_env.csv")
OUT_CSV  = DATA_CSV.parent / "aq_breakdown_totals.csv"

# ---- Constants ----
VSL_2006 = 7_400_000.0
ROLLUP_PREFIX = "intensity_landscape_all_"
EXCLUDE_STATE_FIPS = {"02","15","72","78","66","69","60"}  # AK, HI, PR, VI, GU, MP, AS


def _direct_sum(df: pd.DataFrame, cols):
    if not cols:
        return 0.0
    s = 0.0
    for c in cols:
        if c in df.columns:
            s += pd.to_numeric(df[c], errors="coerce").sum(skipna=True)
    return float(s)


def main():
    # Load
    df = pd.read_csv(DATA_CSV, dtype={"fips": str})

    # Filter to CONUS by FIPS
    df["fips"] = df["fips"].str.zfill(5)
    df = df[~df["fips"].str[:2].isin(EXCLUDE_STATE_FIPS)].copy()

    # Identify columns
    cols = df.columns.tolist()
    roll_cols = [c for c in cols if c.startswith(ROLLUP_PREFIX)]

    # Per-pollutant deaths in roll-up
    pm_cols  = [c for c in roll_cols if ("_pm25_" in c) and c.endswith("deaths_per_km2")]
    nx_cols  = [c for c in roll_cols if (("_no2_" in c) or ("_nox_" in c)) and c.endswith("deaths_per_km2")]
    sx_cols  = [c for c in roll_cols if (("_so2_" in c) or ("_sox_" in c)) and c.endswith("deaths_per_km2")]

    # Explicit total deaths column (if present)
    total_cols = [c for c in roll_cols if "total_deaths_per_km2" in c]

    # Sum per pollutant (direct, no area weighting)
     pm_deaths  = _direct_sum(df, pm_cols)
    nox_deaths = _direct_sum(df, nx_cols)
    sox_deaths = _direct_sum(df, sx_cols)

    if total_cols:
        all_deaths = _direct_sum(df, total_cols)
    else:
        # If no explicit total, sum PM+NOx+SOx once
        all_deaths = pm_deaths + nox_deaths + sox_deaths

    # Costs from VSL (requested convention)
    pm_cost  = pm_deaths  * VSL_2006
    nox_cost = nox_deaths * VSL_2006
    sox_cost = sox_deaths * VSL_2006
    all_cost = all_deaths * VSL_2006

    # Print nicely
    def fmt_num(x):  return f"{x:,.2f}"
    def fmt_usd(x):  return f"${x:,.2f}"

    print(f"Total avoided deaths (all pollutants, CONUS): {fmt_num(all_deaths)}\n")
    print(f"Total avoided mortality cost (2006 USD, VSL=7.4M): {fmt_usd(all_cost)}")
    print(f"   PM2.5 deaths: {fmt_num(pm_deaths)} → {fmt_usd(pm_cost)}")
    print(f"   NOx   deaths: {fmt_num(nox_deaths)} → {fmt_usd(nox_cost)}")
    print(f"   SOx   deaths: {fmt_num(sox_deaths)} → {fmt_usd(sox_cost)}")

    # Save to CSV for records
    out = pd.DataFrame({
        "metric": [
            "total_avoided_deaths_CONUS_rollup_direct_sum",
            "total_avoided_mortality_cost_CONUS_2006USD_VSL7.4M",
            "pm25_deaths_CONUS_rollup_direct_sum",
            "pm25_cost_CONUS_2006USD_VSL7.4M",
            "nox_deaths_CONUS_rollup_direct_sum",
            "nox_cost_CONUS_2006USD_VSL7.4M",
            "sox_deaths_CONUS_rollup_direct_sum",
            "sox_cost_CONUS_2006USD_VSL7.4M",
        ],
        "value": [
            all_deaths, all_cost,
            pm_deaths, pm_cost,
            nox_deaths, nox_cost,
            sox_deaths, sox_cost,
        ]
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved breakdown to: {OUT_CSV}")

if __name__ == "__main__":
    main()
