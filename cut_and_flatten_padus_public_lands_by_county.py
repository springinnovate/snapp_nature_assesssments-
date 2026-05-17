"""Cut cleaned PAD-US public lands by county and flatten each county."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from os import cpu_count
from pathlib import Path

import geopandas as gpd
import shapely
from tqdm import tqdm

from geometry_utils import polygonal_multipolygon, repair_polygonal_geometry

COUNTY_VECTOR_PATH = Path("data/tl_2024_us_county_50_states.gpkg")
OUT_DIR = Path("output")
OUT_LAYER_NAME = "padus_public_lands_clipped_by_county"
OUT_STEM = "padus_public_lands_clipped_by_county"
N_WORKERS = cpu_count() or 1


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cut a cleaned PAD-US public lands layer by county, union each "
            "county's public lands into one feature, and copy county fields."
        )
    )
    parser.add_argument(
        "padus_public_lands_path",
        type=Path,
        help="Cleaned PAD-US public lands vector to cut by county.",
    )
    return parser.parse_args()


def _require_matching_crs(
    padus_public_lands: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
) -> None:
    """Require the PAD-US and county layers to use the same CRS.

    Args:
        padus_public_lands: Cleaned PAD-US public lands features.
        counties: County boundary features.

    Raises:
        ValueError: If either layer lacks a CRS or the CRS values differ.
    """
    if padus_public_lands.crs is None:
        raise ValueError("PAD-US public lands input does not define a CRS.")
    if counties.crs is None:
        raise ValueError(f"County vector does not define a CRS: {COUNTY_VECTOR_PATH}")
    if padus_public_lands.crs != counties.crs:
        raise ValueError(
            "PAD-US public lands and county CRS values differ. "
            "Reproject one input before running this script. "
            f"PAD-US CRS={padus_public_lands.crs}; county CRS={counties.crs}"
        )


def _process_county(
    county_number: int,
    county_fields: dict,
    county_geom_wkb: bytes,
    candidate_wkbs: list[bytes],
) -> tuple[int, dict | None, Counter]:
    """Clip and flatten all candidate public-land pieces for one county.

    Args:
        county_number: Original county row index, used to preserve output order.
        county_fields: Non-geometry county attributes to copy to the output.
        county_geom_wkb: County geometry WKB.
        candidate_wkbs: Public-land geometry WKB values whose envelopes
            intersect the county.

    Returns:
        County row index, output feature fields plus geometry when the county
        has positive-area public-land overlap, and processing counters.
    """
    stats = Counter()
    county_geom = shapely.from_wkb(county_geom_wkb)
    county_geom = repair_polygonal_geometry(county_geom)
    if county_geom is None:
        stats["invalid_county_geometry"] += 1
        return county_number, None, stats

    if not candidate_wkbs:
        stats["no_candidate_features"] += 1
        return county_number, None, stats

    pieces = []
    for candidate_wkb in candidate_wkbs:
        public_land_geom = shapely.from_wkb(candidate_wkb)
        intersection = shapely.intersection(public_land_geom, county_geom)
        intersection = polygonal_multipolygon(intersection)
        if intersection is None:
            stats["zero_area_intersections"] += 1
            continue
        pieces.extend(intersection.geoms)

    if not pieces:
        stats["empty_county_intersection"] += 1
        return county_number, None, stats

    flattened = shapely.union_all(pieces)
    flattened = repair_polygonal_geometry(flattened)
    if flattened is None or flattened.area <= 0:
        stats["invalid_flattened_geometry"] += 1
        return county_number, None, stats

    output_fields = dict(county_fields)
    output_fields["geometry"] = shapely.to_wkb(flattened)
    stats["kept_counties"] += 1
    stats["intersected_pieces"] += len(pieces)
    return county_number, output_fields, stats


def main() -> None:
    """Run the county clipping and flattening workflow."""
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_path = OUT_DIR / f"{OUT_STEM}_{timestamp}.gpkg"

    step_bar = tqdm(total=6, desc="Cut PAD-US by county", unit="step")

    step_bar.set_description("Read PAD-US public lands")
    padus_public_lands = gpd.read_file(args.padus_public_lands_path)
    if padus_public_lands.empty:
        raise ValueError(
            f"PAD-US public lands input is empty: {args.padus_public_lands_path}"
        )
    step_bar.update()

    step_bar.set_description("Read counties")
    counties = gpd.read_file(COUNTY_VECTOR_PATH)
    if counties.empty:
        raise ValueError(f"County vector is empty: {COUNTY_VECTOR_PATH}")
    step_bar.update()

    step_bar.set_description("Validate CRS")
    _require_matching_crs(padus_public_lands, counties)
    step_bar.update()

    step_bar.set_description("Build county jobs")
    county_geometry_column = counties.geometry.name
    county_field_names = [
        field_name for field_name in counties.columns if field_name != county_geometry_column
    ]
    public_land_wkbs = padus_public_lands.geometry.to_wkb()
    public_land_index = padus_public_lands.sindex
    jobs = []
    for county_number, county_row in tqdm(
        counties.iterrows(),
        total=len(counties),
        desc="Build county jobs",
        unit="county",
    ):
        county_geom = county_row[county_geometry_column]
        candidate_indexes = public_land_index.query(county_geom, predicate="intersects")
        candidate_wkbs = public_land_wkbs.iloc[candidate_indexes].tolist()
        jobs.append(
            (
                county_number,
                county_row[county_field_names].to_dict(),
                shapely.to_wkb(county_geom),
                candidate_wkbs,
            )
        )
    step_bar.update()

    print(
        f"Workers={N_WORKERS:,} counties={len(jobs):,} "
        f"public_land_features={len(padus_public_lands):,}",
        flush=True,
    )
    print(f"Output: {out_path}", flush=True)

    step_bar.set_description("Process county jobs")
    all_stats = Counter()
    output_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_process_county, *job) for job in jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Cut and flatten counties",
            unit="county",
        ):
            county_number, output_row, stats = future.result()
            all_stats.update(stats)
            if output_row is not None:
                output_rows.append((county_number, output_row))
    step_bar.update()

    step_bar.set_description("Write output")
    output_rows.sort(key=lambda row: row[0])
    output_feature_rows = []
    for _, output_row in output_rows:
        output_row = dict(output_row)
        output_row["geometry"] = shapely.from_wkb(output_row["geometry"])
        output_feature_rows.append(output_row)

    output_gdf = gpd.GeoDataFrame(
        output_feature_rows,
        columns=[*county_field_names, "geometry"],
        geometry="geometry",
        crs=padus_public_lands.crs,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_gdf.to_file(out_path, layer=OUT_LAYER_NAME, driver="GPKG", index=False)
    step_bar.update()
    step_bar.close()

    print(
        "Processing stats: "
        + " ".join(f"{key}={value:,}" for key, value in all_stats.items()),
        flush=True,
    )
    print(f"Wrote {len(output_gdf):,} county feature(s): {out_path}", flush=True)


if __name__ == "__main__":
    main()
