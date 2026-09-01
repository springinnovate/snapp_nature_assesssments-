"""Build a GIS-ready GeoPackage of qualifying 2020 Census population centers.

This intentionally avoids the TIGERweb ArcGIS query endpoint, which some
browsers and network gateways reject. Geometry comes from ordinary Census
TIGER/Line ZIP downloads and attributes come from Census's static TIGERweb
tables. Both sources use January 1, 2020 place geography and 2020 Census counts.
"""

from __future__ import annotations

import argparse
import os
import sys
from io import StringIO
from pathlib import Path


# Help GDAL/PROJ locate their data when this repository's Conda environment is
# invoked directly rather than activated first.
conda_prefix = Path(sys.prefix)
os.environ.setdefault("GDAL_DATA", str(conda_prefix / "Library" / "share" / "gdal"))
os.environ.setdefault("PROJ_LIB", str(conda_prefix / "Library" / "share" / "proj"))

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm


TIGER_PLACE_ROOT = "https://www2.census.gov/geo/tiger/TIGER2020/PLACE"
TIGER_TABLE_ROOT = "https://tigerweb.geo.census.gov/tigerwebmain/Files/acs24"

# States listed as eligible in section 50301(a)(3) of the cited legislative
# text. Keys are state FIPS codes; values are lowercase postal abbreviations.
ELIGIBLE_STATES = {
    "02": "ak",
    "04": "az",
    "06": "ca",
    "08": "co",
    "16": "id",
    "32": "nv",
    "35": "nm",
    "41": "or",
    "49": "ut",
    "53": "wa",
    "56": "wy",
}

HTTP_HEADERS = {"User-Agent": "SNAPP-Nature-Assessments/1.0"}


def download(session: requests.Session, url: str, target: Path) -> Path:
    """Download and cache one Census source file.

    Args:
        session: Reusable HTTP session.
        url: Census source URL.
        target: Local cache destination.

    Returns:
        Path to the existing or newly downloaded cache file.

    Raises:
        requests.HTTPError: If the Census request fails.
    """
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, headers=HTTP_HEADERS, timeout=180)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def read_place_geometry(
    session: requests.Session, cache_dir: Path
) -> gpd.GeoDataFrame:
    """Download and combine 2020 TIGER/Line place geometry.

    Args:
        session: Reusable HTTP session.
        cache_dir: Directory for downloaded state archives.

    Returns:
        Place polygons for all eligible states using normalized field names.
    """
    state_frames: list[gpd.GeoDataFrame] = []
    for state_fips in tqdm(
        ELIGIBLE_STATES,
        desc="Load Census place geometry",
        unit="state",
    ):
        filename = f"tl_2020_{state_fips}_place.zip"
        archive = download(
            session,
            f"{TIGER_PLACE_ROOT}/{filename}",
            cache_dir / filename,
        )
        state_frames.append(gpd.read_file(archive, engine="pyogrio"))

    places = gpd.GeoDataFrame(
        pd.concat(state_frames, ignore_index=True),
        geometry="geometry",
        crs=state_frames[0].crs,
    )
    return places.rename(
        columns={
            "STATEFP": "STATE",
            "PLACEFP": "PLACE",
            "PLACENS": "PLACENS",
            "GEOID": "GEOID",
            "NAME": "BASENAME",
            "NAMELSAD": "NAME",
            "LSAD": "LSADC",
            "CLASSFP": "PLACECC",
            "FUNCSTAT": "FUNCSTAT",
            "ALAND": "AREALAND",
            "AWATER": "AREAWATER",
            "INTPTLAT": "INTPTLAT",
            "INTPTLON": "INTPTLON",
        }
    )


def read_population_tables(
    session: requests.Session, cache_dir: Path, place_type: str
) -> pd.DataFrame:
    """Download and combine incorporated-place or CDP population tables.

    Args:
        session: Reusable HTTP session.
        cache_dir: Directory for downloaded Census tables.
        place_type: Census table type, either ``incplace`` or ``cdp``.

    Returns:
        Combined table with normalized GEOIDs and integer populations.

    Raises:
        RuntimeError: If a downloaded file contains no HTML table.
    """
    tables: list[pd.DataFrame] = []
    for state_abbreviation in tqdm(
        ELIGIBLE_STATES.values(),
        desc=f"Load {place_type} population tables",
        unit="state",
    ):
        filename = (
            f"tigerweb_acs24_{place_type}_2020_tab20_{state_abbreviation}.html"
        )
        table_file = download(
            session,
            f"{TIGER_TABLE_ROOT}/{filename}",
            cache_dir / filename,
        )
        parsed = pd.read_html(StringIO(table_file.read_text(encoding="utf-8")))
        if not parsed:
            raise RuntimeError(f"No tabular data found in {table_file}")
        table = parsed[0]
        table.columns = [str(column).strip() for column in table.columns]
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    combined["GEOID"] = combined["GEOID"].astype(str).str.zfill(7)
    combined["POP100"] = pd.to_numeric(combined["POP100"], errors="coerce")
    # Some state/type combinations include a blank placeholder row.
    combined = combined.dropna(subset=["POP100"]).copy()
    combined["POP100"] = combined["POP100"].astype("int64")
    return combined


def join_geometry_and_attributes(
    places: gpd.GeoDataFrame,
    attributes: pd.DataFrame,
    minimum_population: int,
) -> gpd.GeoDataFrame:
    """Join place polygons to Census attributes and filter by population.

    Args:
        places: TIGER/Line place geometry.
        attributes: Incorporated-place or CDP Census table.
        minimum_population: Inclusive population threshold.

    Returns:
        Qualifying places with Census attributes and provenance fields.
    """
    qualifying = attributes.loc[attributes["POP100"] >= minimum_population].copy()

    # Prefer the Census static-table values where a field occurs in both files.
    geometry_only = places[["GEOID", "geometry"]].copy()
    joined = geometry_only.merge(qualifying, on="GEOID", how="inner", validate="one_to_one")
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=places.crs)
    joined["population_year"] = 2020
    joined["boundary_vintage"] = "2020-01-01"
    joined["source"] = "U.S. Census Bureau TIGER/Line and Census 2020 TIGERweb tables"
    return joined


def build_cdp_centers(cdp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert Census CDP centroid attributes into point geometry.

    Args:
        cdp: Qualifying CDPs containing ``CENTLON`` and ``CENTLAT``.

    Returns:
        CDP attributes with Census centroid point geometry in EPSG:4326.
    """
    longitude = pd.to_numeric(cdp["CENTLON"], errors="raise")
    latitude = pd.to_numeric(cdp["CENTLAT"], errors="raise")
    attributes = cdp.drop(columns="geometry").copy()
    return gpd.GeoDataFrame(
        attributes,
        geometry=gpd.points_from_xy(longitude, latitude, crs="EPSG:4326"),
        crs="EPSG:4326",
    )


def main() -> None:
    """Build the population-center GeoPackage used by the BBB filter."""
    parser = argparse.ArgumentParser(
        description=(
            "Build incorporated-place and Census-designated-place layers for "
            "eligible states using the 2020 Census population threshold."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis_inputs/census_population_centers_2020.gpkg"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data_unorganized/census_population_centers_2020"),
    )
    parser.add_argument("--minimum-population", type=int, default=1000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        places = read_place_geometry(session, args.cache_dir)
        incorporated_attributes = read_population_tables(
            session, args.cache_dir, "incplace"
        )
        cdp_attributes = read_population_tables(session, args.cache_dir, "cdp")

    incorporated = join_geometry_and_attributes(
        places, incorporated_attributes, args.minimum_population
    )
    cdp = join_geometry_and_attributes(places, cdp_attributes, args.minimum_population)
    cdp_centers = build_cdp_centers(cdp)

    frames = {
        "incorporated_places_pop1000": incorporated,
        "census_designated_places_pop1000": cdp,
        "cdp_centers_pop1000": cdp_centers,
    }

    # Recreate the output so repeat runs cannot leave stale layers.
    if args.output.exists():
        args.output.unlink()

    for layer_name, frame in tqdm(
        frames.items(),
        total=len(frames),
        desc="Write population-center GeoPackage",
        unit="layer",
    ):
        frame.to_file(args.output, layer=layer_name, driver="GPKG", engine="pyogrio")
        print(f"{layer_name}: {len(frame):,} features")

    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
