"""
AQ deposition roll-up (SNAPP)

Compute avoided mortality (deaths) and avoided mortality cost (USD) from
nature-based air pollutant removal via deposition.

Inputs
------
1) deposition_intensity_per_km2_by_env.csv
   - per-km² avoided deaths intensity fields (by environment + pollutant)
2) nlcd_2020_envarea_by_county.csv (or similar)
   - county areas (km²) for forest/shrubland/grassland

Safeguards
----------
- Prevents DOUBLE COUNTING: does not add pollutant-specific values on top of TOTAL.
- Supports "benchmark mode" to mimic Sumil ES&T forests-only deposition check:
  total_deaths = Σ (forest_area_km² × forest_total_deaths_per_km²)

Outputs
-------
- Prints national totals + by-env + by-pollutant summaries
- Writes tidy CSV:
  aq_deposition_avoided_deaths_and_valuation_summary.csv
"""

import argparse
import pandas as pd
from pathlib import Path

EXCLUDE_STATE_FIPS_DEFAULT = {"02","15","60","66","69","72","78"}  # AK, HI, AS, GU, MP, PR, VI
VSL_2006_DEFAULT = 7_400_000.0
OWNERSHIPS = ["all", "public", "private"]

# Map "environment" names in the intensity file -> area columns in the NLCD/PADUS area file
ENV_TO_AREA = {
    "all": {
        "landscape_all": "forest_area_km2",
        "shrubland":     "shrubland_area_km2",
        "grassland":     "grassland_area_km2",
    },
    "public": {
        "landscape_all": "forest_area_public_km2",
        "shrubland":     "shrubland_area_public_km2",
        "grassland":     "grassland_area_public_km2",
    },
    "private": {
        "landscape_all": "forest_area_private_km2",
        "shrubland":     "shrubland_area_private_km2",
        "grassland":     "grassland_area_private_km2",
    }
}


# Map env names in intensity file -> area columns in NLCD area file
DEFAULT_ENV_TO_AREA = {
    "landscape_all": "forest_area_km2",
    "shrubland":     "shrubland_area_km2",
    "grassland":     "grassland_area_km2",
}

AREA_SETS = {
    "total": DEFAULT_ENV_TO_AREA,

    "public": {
        "landscape_all": "forest_area_public_km2",
        "shrubland":     "shrubland_area_public_km2",
        "grassland":     "grassland_area_public_km2",
    },

    "private": {
        "landscape_all": "forest_area_private_km2",
        "shrubland":     "shrubland_area_private_km2",
        "grassland":     "grassland_area_private_km2",
    }
}


def _zfill_fips(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)

def _detect_envs(intensity_cols):
    envs = set()
    for c in intensity_cols:
        if not (c.startswith("intensity_") and c.endswith("_deaths_per_km2")):
            continue
        parts = c.split("_")
        try:
            deaths_idx = parts.index("deaths")
        except ValueError:
            continue
        token_before_deaths = parts[deaths_idx - 1]  # "total" or pollutant token
        if token_before_deaths == "total":
            env = "_".join(parts[1:deaths_idx-1])
        else:
            env = "_".join(parts[1:deaths_idx-2])
        if env:
            envs.add(env)
    return sorted(envs)

def _sum_series(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intensity", required=True, help="Path to deposition_intensity_per_km2_by_env.csv")
    ap.add_argument("--areas", required=True, help="Path to nlcd_*_envarea_by_county.csv")
    ap.add_argument("--out", default=None, help="Output CSV path (default: alongside intensity file)")
    ap.add_argument("--vsl2006", type=float, default=VSL_2006_DEFAULT, help="VSL in 2006 USD (default 7.4M)")
    ap.add_argument("--conus-only", action="store_true", help="Exclude AK/HI/territories via state FIPS")
    ap.add_argument("--benchmark-forest-only", action="store_true",
                    help="Benchmark mode: forests-only (landscape_all total deaths per km2 × forest_area_km2)")
    args = ap.parse_args()

    intensity_path = Path(args.intensity)
    areas_path = Path(args.areas)
    out_path = Path(args.out) if args.out else intensity_path.parent / "aq_deposition_avoided_deaths_and_valuation_summary.csv"

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity", required=True)
    parser.add_argument("--areas", required=True)
    parser.add_argument("--out", default="aq_deposition_avoided_deaths_and_valuation_summary.csv")
    parser.add_argument("--vsl2006", type=float, default=7_400_000)
    parser.add_argument("--conus-only", action="store_true")
    parser.add_argument("--benchmark-forest-only", action="store_true")
    return parser

 # Normalize FIPS
    i = normalize_fips(i)
    a = normalize_fips(a)
    
def main():
    parser = build_parser()
    args = parser.parse_args()   # 🔑 THIS LINE WAS MISSING

    out_path = (
    Path(args.out)
    if args.out
    else Path(args.intensity).parent / "aq_deposition_avoided_deaths_and_valuation_summary.csv"
)

    print("Loading intensity and area tables...")
    intensity_path = args.intensity
    areas_path = args.areas

    i = pd.read_csv(intensity_path, dtype=str)
    a = pd.read_csv(areas_path, dtype=str)


    if args.conus_only:
        i = i[~i["fips"].str[:2].isin(EXCLUDE_STATE_FIPS_DEFAULT)].copy()
        a = a[~a["fips"].str[:2].isin(EXCLUDE_STATE_FIPS_DEFAULT)].copy()

    df = i.merge(a, on="fips", how="inner")
    print(f"  Merged rows (counties with both intensity + area): {len(df):,}")

    # ===============================
    # BENCHMARK CHECK (Sumil test)
    # ===============================
    if args.benchmark_forest_only:
        env = "landscape_all"
        deaths_col = f"intensity_{env}_total_deaths_per_km2"
        area_col = DEFAULT_ENV_TO_AREA[env]

        deaths = (
            pd.to_numeric(df[deaths_col], errors="coerce").fillna(0.0)
            * pd.to_numeric(df[area_col], errors="coerce").fillna(0.0)
        )

        total_deaths = deaths.sum()
        total_cost = total_deaths * args.vsl2006

        print("\n=== ES&T-STYLE BENCHMARK (FORESTS ONLY) ===")
        print(f"Total avoided deaths: {total_deaths:,.2f}")
        print(f"Total avoided mortality cost (2006 USD): ${total_cost:,.2f}")

        return  # ← LEGAL because we are inside main()

    # ===============================
    # FULL ROLL-UP (ALL / PUBLIC / PRIVATE)
    # ===============================
    for ownership in OWNERSHIPS:
        print(f"\n===== {ownership.upper()} LANDS =====")

        total_deaths = 0.0
        total_cost = 0.0

        for env, area_col in ENV_TO_AREA[ownership].items():
            intensity_col = f"intensity_{env}_total_deaths_per_km2"
    
            if intensity_col not in df.columns or area_col not in df.columns:
                print(f"  {env:12s}: skipped (missing data)")
                continue

            area = pd.to_numeric(df[area_col], errors="coerce").fillna(0.0)
            intensity = pd.to_numeric(df[intensity_col], errors="coerce").fillna(0.0)

            deaths = (intensity * area).sum()
            cost = deaths * args.vsl2006

            total_deaths += deaths
            total_cost += cost

            print(f"  {env:12s}: {deaths:,.2f} deaths, ${cost:,.2f}")

        print("\n================= NATIONAL TOTALS =================")
        print(f"Total avoided deaths: {total_deaths:,.2f}")
        print(f"Total avoided mortality cost (2006 USD): ${total_cost:,.2f}")



if __name__ == "__main__":
    main()

