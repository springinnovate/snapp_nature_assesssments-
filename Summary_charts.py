import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# =========================
# CONFIG - EDIT THESE PATHS
# =========================

# PAD-US Fee lands (public lands proxy)
PADUS_PATH = "/Users/jahnelle/Desktop/PADUS4_1Geodatabase/PADUS4_1Geodatabase.gdb"
PADUS_LAYER = "PADUS4_1Fee"

# Counties (2010 TIGER generalization, GeoJSON)
COUNTIES_PATH = "/Users/jahnelle/Desktop/Deposition_Maps/gz_2010_us_050_00_5m.json"

# Air quality CSV (per-km2 valuation, in USD)
AIR_CSV = "/Users/jahnelle/Desktop/Deposition_Maps/deposition_intensity_per_km2_by_env.csv"
AIR_VALUE_COL = "intensity_landscape_all_valuation_2006usd_per_km2"  # USD per km2
AIR_FIPS_COL = "fips"       # in AIR_CSV

# Flood mitigation HUC summary (already in USD; no raster)
FLOOD_HUC_CSV = "/Users/jahnelle/Desktop/Flood_Maps/huc_summary_forSNAPP_v2.csv"
FLOOD_VALUE_COL = "marginal_npv_2020"   # present-value flood-mitigation benefit (USD) per HUC12
FLOOD_HUC_COL = "huc12"

# Output folder
OUT_DIR = "/Users/jahnelle/Desktop/public_lands_summaries"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns in counties file
COUNTY_FIPS_COL_SHP = "GEOID"   # we'll create this from GEO_ID
STATE_FIPS_COL = "STATE"        # this is 2-digit numeric FIPS in gz_2010_us_050_00_5m.json

# Map 2-digit FIPS -> USPS state abbreviation
STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY", "60": "AS", "66": "GU", "69": "MP", "72": "PR",
    "78": "VI"
}

# =========================
# HELPER FUNCTIONS
# =========================

def load_padus_public_lands():
    """Load PAD-US Fee lands and treat them as public lands for this analysis."""
    print("Loading PAD-US Fee layer...")
    padus = gpd.read_file(PADUS_PATH, layer=PADUS_LAYER)
    print(f"Loaded PAD-US features: {len(padus)}")

    padus_public = padus.copy()
    padus_public = padus_public.to_crs(epsg=5070)
    print(f"Using all {len(padus_public)} PAD-US Fee features as public lands (projected to EPSG:5070).")
    return padus_public


def load_counties():
    """Load counties and create GEOID from GEO_ID."""
    print("Loading counties...")
    try:
        counties = gpd.read_file(COUNTIES_PATH, engine="fiona")
    except TypeError:
        counties = gpd.read_file(COUNTIES_PATH)
    print(f"Loaded {len(counties)} county polygons.")

    # gz_2010_us_050_00_5m.json has GEO_ID like "0500000US06037" -> last 5 chars are county FIPS
    if "GEO_ID" in counties.columns and "GEOID" not in counties.columns:
        counties["GEOID"] = counties["GEO_ID"].str[-5:]
    counties = counties.to_crs(epsg=5070)
    return counties


def make_bar_chart(labels, values, title, ylabel, outfile):
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved chart: {outfile}")


def make_top10_chart(df, value_col, label_col, title, outfile, group_col=None, group_label=None):
    """Make a horizontal bar chart and also write a CSV for the top 10."""
    top10 = df.sort_values(value_col, ascending=False).head(10).copy()

    plt.figure(figsize=(7, 5))
    x = np.arange(len(top10))
    if group_col:
        width = 0.4
        plt.barh(x - width/2, top10[value_col], height=width, label="Total (all lands)")
        plt.barh(x + width/2, top10[group_col], height=width, label=group_label or "Public lands")
        plt.yticks(x, top10[label_col])
        plt.legend()
    else:
        plt.barh(top10[label_col], top10[value_col])

    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved chart: {outfile}")

    # Also save a CSV table listing labels and values
    base = os.path.splitext(os.path.basename(outfile))[0]
    csv_path = os.path.join(OUT_DIR, base + "_table.csv")
    cols = [label_col, value_col] + ([group_col] if group_col else [])
    top10[cols].to_csv(csv_path, index=False)
    print(f"Saved top-10 table: {csv_path}")


# =========================
# AIR QUALITY: MONEY + PUBLIC LANDS
# =========================

def summarize_air_quality(padus_public, counties):
    print("\n=== AIR QUALITY: summarizing by state and public lands (USD) ===")
    air = pd.read_csv(AIR_CSV)

    # Drop missing FIPS and zero-pad
    air = air[air[AIR_FIPS_COL].notna()].copy()
    air[AIR_FIPS_COL] = (
        air[AIR_FIPS_COL]
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    # Prepare county FIPS and state abbreviations
    counties[COUNTY_FIPS_COL_SHP] = counties[COUNTY_FIPS_COL_SHP].astype(str).str.zfill(5)
    # STATE in counties is numeric FIPS; map to USPS abbreviation for nicer labels
    counties["STATE_FIPS2"] = counties[STATE_FIPS_COL].astype(str).str.zfill(2)
    counties["STATE_ABBR"] = counties["STATE_FIPS2"].map(STATE_FIPS_TO_ABBR).fillna(counties["STATE_FIPS2"])

    # Join air data to county geometries
    gdf = counties.merge(
        air[[AIR_FIPS_COL, AIR_VALUE_COL]],
        left_on=COUNTY_FIPS_COL_SHP,
        right_on=AIR_FIPS_COL,
        how="inner"
    )
    print(f"Joined air data to {len(gdf)} counties.")

    # Overlay counties with PAD-US Fee to get public-land overlap
    print("Intersecting counties with public lands (this may take a bit)...")
    county_public = gdf.overlay(padus_public, how="intersection")

    county_public["overlap_area"] = county_public.geometry.area
    county_area = gdf[[COUNTY_FIPS_COL_SHP, "STATE_ABBR", "geometry"]].copy()
    county_area["county_area"] = county_area.geometry.area

    overlap_sum = (
        county_public
        .groupby(COUNTY_FIPS_COL_SHP)["overlap_area"]
        .sum()
        .reset_index()
        .rename(columns={"overlap_area": "public_overlap_area"})
    )

    county_area = county_area.merge(overlap_sum, on=COUNTY_FIPS_COL_SHP, how="left")
    county_area["public_overlap_area"] = county_area["public_overlap_area"].fillna(0.0)

    # Join valuation column
    county_area = county_area.merge(
        air[[AIR_FIPS_COL, AIR_VALUE_COL]],
        left_on=COUNTY_FIPS_COL_SHP,
        right_on=AIR_FIPS_COL,
        how="left"
    )

    # Fraction of county that is public land
    county_area["public_frac"] = (
        county_area["public_overlap_area"] / county_area["county_area"]
    ).clip(0, 1).fillna(0.0)

    # Convert km2 valuation to total $
    county_area["area_km2"] = county_area["county_area"] / 1e6
    county_area["total_value_usd"] = county_area[AIR_VALUE_COL] * county_area["area_km2"]
    county_area["public_value_usd"] = county_area["total_value_usd"] * county_area["public_frac"]

    # === Rescale to match Sumil's national total (≈ $52.7B, 2006 USD) ===
    # Current raw total from the per-km2 dataset:
    nat_total_raw = county_area["total_value_usd"].sum()

    # Trusted national total from RCM–GEMM (Sumil's analysis)
    TRUE_NATIONAL_TOTAL = 52_727_097_362.36  # USD, ~52.7 billion

    if nat_total_raw <= 0:
        raise ValueError("Computed national air-quality total is non-positive; cannot rescale.")
    scale_factor = TRUE_NATIONAL_TOTAL / nat_total_raw

    # Apply scale factor to both all-lands and public-lands values
    county_area["total_value_usd"] *= scale_factor
    county_area["public_value_usd"] *= scale_factor

    # National $ totals (scaled to Sumil's 52.7B baseline)
    nat_total = county_area["total_value_usd"].sum()
    nat_public = county_area["public_value_usd"].sum()
    nat_df = pd.DataFrame({
        "service": ["air_quality"],
        "total_all_lands_usd": [nat_total],
        "total_public_lands_usd": [nat_public],
        "pct_public": [nat_public / nat_total * 100 if nat_total > 0 else np.nan]
    })
    nat_df.to_csv(os.path.join(OUT_DIR, "air_quality_national_summary.csv"), index=False)
    print("National air quality summary (USD):")
    print(nat_df)

    # State $ totals (using state abbreviations)
    state_grp = (
        county_area
        .groupby("STATE_ABBR")
        .agg(total_all_lands_usd=("total_value_usd", "sum"),
             total_public_lands_usd=("public_value_usd", "sum"))
        .reset_index()
        .rename(columns={"STATE_ABBR": "state"})
    )
    state_grp["pct_public"] = state_grp["total_public_lands_usd"] / state_grp["total_all_lands_usd"] * 100
    state_grp.to_csv(os.path.join(OUT_DIR, "air_quality_state_summary.csv"), index=False)

    # Add billion-USD columns for plotting convenience (keep CSV in raw USD)
    state_grp["total_all_lands_bil"] = state_grp["total_all_lands_usd"] / 1e9
    state_grp["total_public_lands_bil"] = state_grp["total_public_lands_usd"] / 1e9

    # Top 10 states table
    top10_states = state_grp.sort_values("total_all_lands_usd", ascending=False).head(10)
    top10_states.to_csv(os.path.join(OUT_DIR, "air_quality_top10_states_table.csv"), index=False)
    print("Top 10 states for air quality valuation (USD):")
    print(top10_states[["state", "total_all_lands_usd", "total_public_lands_usd", "pct_public"]])

    # Charts with state abbreviations
    make_bar_chart(
        labels=["All lands", "Public lands"],
        values=[nat_total / 1e9, nat_public / 1e9],
        title="Air quality valuation – US total vs public lands",
        ylabel="Valuation (billion USD)",
        outfile=os.path.join(OUT_DIR, "air_quality_US_total_vs_public.png")
    )

    make_top10_chart(
        df=state_grp,
        value_col="total_all_lands_bil",
        label_col="state",
        title="Top 10 states – air quality valuation (all lands, billion USD)",
        outfile=os.path.join(OUT_DIR, "air_quality_top10_states_total.png")
    )

    make_top10_chart(
        df=state_grp,
        value_col="total_all_lands_bil",
        label_col="state",
        group_col="total_public_lands_bil",
        group_label="Public lands (billion USD)",
        title="Top 10 states – air quality valuation (total vs public lands, billion USD)",
        outfile=os.path.join(OUT_DIR, "air_quality_top10_states_total_vs_public.png")
    )

    return nat_df, state_grp


# =========================
# FLOOD MITIGATION: MONEY (HUC-LEVEL, NO RASTER)
# =========================

def summarize_flood_from_hucs():
    """Summarize flood-mitigation money using the HUC summary CSV you already created.

    This avoids loading the massive national flood raster into memory.
    """
    print("\n=== FLOOD MITIGATION: summarizing from HUC summary (USD) ===")
    df = pd.read_csv(FLOOD_HUC_CSV)

    if FLOOD_VALUE_COL not in df.columns:
        raise ValueError(f"FLOOD_VALUE_COL '{FLOOD_VALUE_COL}' not found in HUC CSV. Columns: {df.columns}")
    if FLOOD_HUC_COL not in df.columns:
        raise ValueError(f"FLOOD_HUC_COL '{FLOOD_HUC_COL}' not found in HUC CSV. Columns: {df.columns}")

    # NATIONAL total $ (all lands)
    nat_total = df[FLOOD_VALUE_COL].sum()
    nat_df = pd.DataFrame({
        "service": ["flood_mitigation"],
        "total_all_lands_usd": [nat_total],
    })
    nat_df.to_csv(os.path.join(OUT_DIR, "flood_national_summary.csv"), index=False)
    print("National flood-mitigation summary (USD, all lands):")
    print(nat_df)

    # HUC-level totals (already one row per HUC12 in your file)
    huc_grp = df[[FLOOD_HUC_COL, FLOOD_VALUE_COL]].copy()
    huc_grp = huc_grp.rename(columns={FLOOD_HUC_COL: "huc12",
                                      FLOOD_VALUE_COL: "total_all_lands_usd"})
    huc_grp.to_csv(os.path.join(OUT_DIR, "flood_huc_summary.csv"), index=False)

    # Top 10 HUC12s by flood-mitigation value
    top10_huc = huc_grp.sort_values("total_all_lands_usd", ascending=False).head(10)
    top10_huc.to_csv(os.path.join(OUT_DIR, "flood_top10_huc12_table.csv"), index=False)
    print("Top 10 HUC12s for flood mitigation (USD, all lands):")
    print(top10_huc[["huc12", "total_all_lands_usd"]])

    # Charts (all lands only, HUC12 labels)
    make_bar_chart(
        labels=["All lands"],
        values=[nat_total],
        title="Flood mitigation – US total (all lands, USD)",
        ylabel="Flood-mitigation value (USD)",
        outfile=os.path.join(OUT_DIR, "flood_US_total_all_lands.png")
    )

    make_top10_chart(
        df=top10_huc,
        value_col="total_all_lands_usd",
        label_col="huc12",
        title="Top 10 HUC12s – flood mitigation (all lands, USD)",
        outfile=os.path.join(OUT_DIR, "flood_top10_huc12_total.png")
    )

    return nat_df, huc_grp


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # Air quality: needs PAD-US + counties for public-lands breakdown
    padus_public = load_padus_public_lands()
    counties = load_counties()
    air_nat, air_states = summarize_air_quality(padus_public, counties)

    # Flood mitigation: use HUC summary CSV in USD (no giant raster)
    flood_nat, flood_hucs = summarize_flood_from_hucs()

    print("\nAll done. Outputs saved in:", OUT_DIR)

