import pandas as pd
from pathlib import Path

# ===============================
# COMBINE INTENSITY + AREA
# TO GET REAL AVOIDED DEATHS & $
# ===============================

BASE_DIR = Path("/Users/jahnelle/Desktop/Deposition_Maps")

INTENSITY_CSV = BASE_DIR / "deposition_intensity_per_km2_by_env.csv"
AREA_CSV      = BASE_DIR / "nlcd_2020_envarea_by_county.csv"

VSL_2006_USD = 7_400_000  # Value of Statistical Life

def main():
    print("Loading intensity and area tables...")
    inten = pd.read_csv(INTENSITY_CSV, dtype={"fips": str})
    area  = pd.read_csv(AREA_CSV, dtype={"fips": str})

    inten["fips"] = inten["fips"].str.zfill(5)
    area["fips"]  = area["fips"].str.zfill(5)

    # Merge on FIPS
    df = inten.merge(area, on="fips", how="inner")

    # Map env labels to area columns
    env_to_area = {
        "landscape_all": "forest_area_km2",     # forest + trees (LandScape dep table)
        "shrubland":     "shrubland_area_km2",
        "grassland":     "grassland_area_km2",
    }

    pollutants = ["pm25", "nox", "sox"]

    # Containers for totals
    totals = {
        "env_pol_deaths": {},   # (env, pol) -> deaths
        "pol_deaths": {pol: 0.0 for pol in pollutants},
        "env_deaths": {env: 0.0 for env in env_to_area.keys()},
        "all_deaths": 0.0
    }

    # Compute deaths per env & pollutant
    for env, area_col in env_to_area.items():
        if area_col not in df.columns:
            print(f"⚠ Area column {area_col} missing, skipping env {env}")
            continue

        for pol in pollutants:
            inten_col = f"intensity_{env}_{pol}_deaths_per_km2"
            if inten_col not in df.columns:
                print(f"⚠ Intensity column {inten_col} missing, skipping.")
                continue

            deaths_col = f"deaths_{env}_{pol}"
            df[deaths_col] = df[inten_col] * df[area_col]

            total_deaths_env_pol = df[deaths_col].sum()
            totals["env_pol_deaths"][(env, pol)] = total_deaths_env_pol
            totals["pol_deaths"][pol] += total_deaths_env_pol
            totals["env_deaths"][env] += total_deaths_env_pol
            totals["all_deaths"] += total_deaths_env_pol

    # Compute valuations
    total_val_usd = totals["all_deaths"] * VSL_2006_USD
    pol_vals = {pol: totals["pol_deaths"][pol] * VSL_2006_USD for pol in pollutants}
    env_vals = {env: totals["env_deaths"][env] * VSL_2006_USD for env in env_to_area.keys()}

    # Pretty print results
    print("\n================= AVOIDED MORTALITY (DEATHS) =================")
    print(f"Total avoided deaths (all envs, all pollutants): {totals['all_deaths']:,.2f}\n")

    print("By pollutant:")
    for pol in pollutants:
        print(f"  {pol.upper():<5}: {totals['pol_deaths'][pol]:,.2f} deaths")

    print("\nBy environment:")
    for env in env_to_area.keys():
        print(f"  {env:<13}: {totals['env_deaths'][env]:,.2f} deaths")

    print("\n================= AVOIDED MORTALITY COST (2006 USD) =================")
    print(f"Total avoided mortality cost: ${total_val_usd:,.2f}\n")

    print("By pollutant:")
    for pol in pollutants:
        print(f"  {pol.upper():<5}: ${pol_vals[pol]:,.2f}")

    print("\nBy environment:")
    for env in env_to_area.keys():
        print(f"  {env:<13}: ${env_vals[env]:,.2f}")

    # Optionally, save a summary CSV
    summary_rows = []

    for (env, pol), deaths in totals["env_pol_deaths"].items():
        summary_rows.append({
            "env": env,
            "pollutant": pol,
            "avoided_deaths": deaths,
            "avoided_cost_2006usd": deaths * VSL_2006_USD
        })

    summary_df = pd.DataFrame(summary_rows)
    out_path = BASE_DIR / "deposition_avoided_deaths_and_valuation_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print("\nSaved detailed env × pollutant summary to:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
