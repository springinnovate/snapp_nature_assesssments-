"""Cut polygon features by county and flatten each county."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from os import cpu_count
from pathlib import Path
import re

import geopandas as gpd
import shapely
from tqdm import tqdm

from geometry_utils import polygonal_multipolygon, repair_polygonal_geometry

COUNTY_VECTOR_PATH = Path(
    "data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg"
)
DEFAULT_OUT_DIR = Path("data/processing_outputs/cut_by_county")
PADUS_ALL_LANDS_OUT_DIR = Path(
    "data/analysis_inputs/zonal_units/padus_all_lands_by_county"
)
PADUS_PUBLIC_LANDS_OUT_DIR = Path(
    "data/analysis_inputs/zonal_units/padus_public_lands_by_county"
)
N_WORKERS = cpu_count() or 1
TIMESTAMP_SUFFIX = re.compile(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cut a polygon layer by county, union each county's pieces into "
            "one feature, and copy county fields."
        )
    )
    parser.add_argument(
        "input_vector_path",
        type=Path,
        help="Polygon vector to cut by county.",
    )
    return parser.parse_args()


def _prepare_counties_crs(
    input_features: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Transform counties to the input CRS when needed.

    Args:
        input_features: Input polygon features.
        counties: County boundary features.

    Returns:
        County boundary features in the input CRS.

    Raises:
        ValueError: If either layer lacks a CRS.
    """
    if input_features.crs is None:
        raise ValueError("Input vector does not define a CRS.")
    if counties.crs is None:
        raise ValueError(f"County vector does not define a CRS: {COUNTY_VECTOR_PATH}")
    if input_features.crs != counties.crs:
        return counties.to_crs(input_features.crs)
    return counties


def _derive_output_names(input_path: Path) -> tuple[str, Path]:
    """Derive the timestamp-free output layer name and output path.

    Args:
        input_path: Input vector path.

    Returns:
        Output layer name and timestamped output path.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    input_stem = TIMESTAMP_SUFFIX.sub("", input_path.stem)
    if "_clipped_to_usa" in input_stem:
        out_stem = input_stem.replace("_clipped_to_usa", "_clipped_by_county")
    else:
        out_stem = f"{input_stem}_clipped_by_county"

    if out_stem.startswith("padus_all_lands_"):
        out_dir = PADUS_ALL_LANDS_OUT_DIR
    elif out_stem.startswith("padus_public_lands_"):
        out_dir = PADUS_PUBLIC_LANDS_OUT_DIR
    else:
        out_dir = DEFAULT_OUT_DIR
    return out_stem, out_dir / f"{out_stem}_{timestamp}.gpkg"


def _process_county(
    county_number: int,
    county_fields: dict,
    county_geom_wkb: bytes,
    candidate_inputs: list[tuple[bytes, dict]],
) -> tuple[int, dict | None, Counter]:
    """Clip and flatten all candidate polygon pieces for one county.

    Args:
        county_number: Original county row index, used to preserve output order.
        county_fields: Non-geometry county attributes to copy to the output.
        county_geom_wkb: County geometry WKB.
        candidate_inputs: Input geometry WKB values and non-geometry fields
            whose envelopes intersect the county.

    Returns:
        County row index, output feature fields plus geometry when the county
        has positive-area input overlap, and processing counters.
    """
    stats = Counter()
    county_geom = shapely.from_wkb(county_geom_wkb)
    county_geom = repair_polygonal_geometry(county_geom)
    if county_geom is None:
        stats["invalid_county_geometry"] += 1
        return county_number, None, stats

    if not candidate_inputs:
        stats["no_candidate_features"] += 1
        return county_number, None, stats

    pieces = []
    input_fields = None
    for candidate_wkb, candidate_fields in candidate_inputs:
        input_geom = shapely.from_wkb(candidate_wkb)
        intersection = shapely.intersection(input_geom, county_geom)
        intersection = polygonal_multipolygon(intersection)
        if intersection is None:
            stats["zero_area_intersections"] += 1
            continue
        if input_fields is None:
            input_fields = dict(candidate_fields)
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
    output_fields.update(input_fields)
    output_fields["geometry"] = shapely.to_wkb(flattened)
    stats["kept_counties"] += 1
    stats["intersected_pieces"] += len(pieces)
    return county_number, output_fields, stats


def main() -> None:
    """Run the county clipping and flattening workflow."""
    args = _parse_args()
    out_layer_name, out_path = _derive_output_names(args.input_vector_path)

    step_bar = tqdm(total=6, desc="Cut input by county", unit="step")

    step_bar.set_description("Read input features")
    input_features = gpd.read_file(args.input_vector_path)
    if input_features.empty:
        raise ValueError(f"Input vector is empty: {args.input_vector_path}")
    step_bar.update()

    step_bar.set_description("Read counties")
    counties = gpd.read_file(COUNTY_VECTOR_PATH)
    if counties.empty:
        raise ValueError(f"County vector is empty: {COUNTY_VECTOR_PATH}")
    step_bar.update()

    step_bar.set_description("Prepare county CRS")
    counties = _prepare_counties_crs(input_features, counties)
    step_bar.update()

    step_bar.set_description("Build county jobs")
    county_geometry_column = counties.geometry.name
    county_field_names = [
        field_name for field_name in counties.columns if field_name != county_geometry_column
    ]
    input_geometry_column = input_features.geometry.name
    input_field_names = [
        field_name
        for field_name in input_features.columns
        if field_name != input_geometry_column
    ]
    input_wkbs = input_features.geometry.to_wkb()
    input_field_rows = input_features[input_field_names].to_dict(orient="records")
    input_index = input_features.sindex
    jobs = []
    for county_number, county_row in tqdm(
        counties.iterrows(),
        total=len(counties),
        desc="Build county jobs",
        unit="county",
    ):
        county_geom = county_row[county_geometry_column]
        candidate_indexes = input_index.query(county_geom, predicate="intersects")
        candidate_inputs = [
            (input_wkbs.iloc[candidate_index], input_field_rows[candidate_index])
            for candidate_index in candidate_indexes
        ]
        jobs.append(
            (
                county_number,
                county_row[county_field_names].to_dict(),
                shapely.to_wkb(county_geom),
                candidate_inputs,
            )
        )
    step_bar.update()

    print(
        f"Workers={N_WORKERS:,} counties={len(jobs):,} "
        f"input_features={len(input_features):,}",
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
        columns=[*county_field_names, *input_field_names, "geometry"],
        geometry="geometry",
        crs=input_features.crs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_gdf.to_file(out_path, layer=out_layer_name, driver="GPKG", index=False)
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
