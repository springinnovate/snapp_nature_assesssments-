"""Allocate 2024 recreation value to counties by proportional feature area."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from math import isfinite
from os import cpu_count
from pathlib import Path

import geopandas as gpd
import shapely
from tqdm import tqdm

from geometry_utils import polygonal_multipolygon, repair_polygonal_geometry

RECREATION_VECTOR_PATH = Path(
    "data/analysis_inputs/recreation/usa_nature_assessment_recreation.gpkg"
)
COUNTY_VECTOR_PATH = Path(
    "data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg"
)
OUT_DIR = Path("data/analysis_inputs/zonal_units/recreation_by_county")
OUT_STEM = "recreation_value_by_county"
AREA_CRS = "EPSG:5070"
GEOID_FIELD = "GEOID"
SOURCE_VALUE_FIELD = "val_2024"
OUT_VALUE_FIELD = "proportional_recreation_val_2024"
N_WORKERS = cpu_count() or 1


def _timestamped_output_path() -> Path:
    """Build the timestamped output GeoPackage path.

    Returns:
        Output path for the county recreation value layer.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return OUT_DIR / f"{OUT_STEM}_{timestamp}.gpkg"


def _require_fields(gdf: gpd.GeoDataFrame, fields: set[str], path: Path) -> None:
    """Verify that a GeoDataFrame contains required fields.

    Args:
        gdf: GeoDataFrame to inspect.
        fields: Required field names.
        path: Source path, used in error messages.

    Raises:
        ValueError: If any required field is missing.
    """
    missing_fields = fields - set(gdf.columns)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Missing required field(s) in {path}: {missing}")


def _project_to_area_crs(gdf: gpd.GeoDataFrame, path: Path) -> gpd.GeoDataFrame:
    """Project a vector layer to the equal-area CRS used for allocation.

    Args:
        gdf: GeoDataFrame to project.
        path: Source path, used in error messages.

    Returns:
        GeoDataFrame in EPSG:5070.

    Raises:
        ValueError: If the input layer does not define a CRS.
    """
    if gdf.crs is None:
        raise ValueError(f"Vector does not define a CRS: {path}")
    if gdf.crs.to_epsg() == 5070:
        return gdf
    return gdf.to_crs(AREA_CRS)


def _coerce_nonzero_value(value) -> float | None:
    """Convert a source recreation value to a finite nonzero float.

    Args:
        value: Raw `val_2024` field value.

    Returns:
        Numeric value, or None when the value is null, zero, non-finite, or
        non-numeric.
    """
    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric_value) or numeric_value == 0:
        return None
    return numeric_value


def _prepare_recreation_features(
    recreation: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, Counter]:
    """Filter and normalize recreation features before county allocation.

    Args:
        recreation: Recreation features in EPSG:5070.

    Returns:
        Prepared recreation features and processing counters. Prepared features
        contain only geometry, `val_2024`, and full feature area.
    """
    stats = Counter()
    prepared_rows = []
    geometry_column = recreation.geometry.name

    for _, row in tqdm(
        recreation.iterrows(),
        total=len(recreation),
        desc="Prepare recreation features",
        unit="feature",
    ):
        value = _coerce_nonzero_value(row[SOURCE_VALUE_FIELD])
        if value is None:
            stats["zero_null_or_invalid_values"] += 1
            continue

        geom = repair_polygonal_geometry(row[geometry_column])
        if geom is None:
            stats["invalid_recreation_geometry"] += 1
            continue

        full_area = geom.area
        if full_area <= 0:
            stats["zero_area_recreation_geometry"] += 1
            continue

        prepared_rows.append(
            {
                SOURCE_VALUE_FIELD: value,
                "full_area": full_area,
                "geometry": geom,
            }
        )

    prepared = gpd.GeoDataFrame(
        prepared_rows,
        columns=[SOURCE_VALUE_FIELD, "full_area", "geometry"],
        geometry="geometry",
        crs=AREA_CRS,
    )
    stats["prepared_recreation_features"] = len(prepared)
    return prepared, stats


def _build_county_jobs(
    counties: gpd.GeoDataFrame,
    recreation: gpd.GeoDataFrame,
) -> list[tuple[int, str, bytes, list[tuple[bytes, float, float]]]]:
    """Build per-county allocation jobs using a recreation spatial index.

    Args:
        counties: County geometries in EPSG:5070.
        recreation: Prepared recreation features in EPSG:5070.

    Returns:
        Job arguments for county allocation workers.
    """
    recreation_index = recreation.sindex
    recreation_wkbs = recreation.geometry.to_wkb()
    recreation_values = recreation[SOURCE_VALUE_FIELD].to_numpy()
    recreation_areas = recreation["full_area"].to_numpy()
    geometry_column = counties.geometry.name
    jobs = []

    for county_number, county_row in tqdm(
        counties.iterrows(),
        total=len(counties),
        desc="Build county jobs",
        unit="county",
    ):
        county_geom = county_row[geometry_column]
        candidate_indexes = recreation_index.query(county_geom, predicate="intersects")
        candidate_recreation = [
            (
                recreation_wkbs.iloc[candidate_index],
                float(recreation_values[candidate_index]),
                float(recreation_areas[candidate_index]),
            )
            for candidate_index in candidate_indexes
        ]
        jobs.append(
            (
                county_number,
                str(county_row[GEOID_FIELD]),
                shapely.to_wkb(county_geom),
                candidate_recreation,
            )
        )

    return jobs


def _process_county(
    county_number: int,
    geoid: str,
    county_geom_wkb: bytes,
    candidate_recreation: list[tuple[bytes, float, float]],
) -> tuple[int, dict, Counter]:
    """Allocate candidate recreation values to one county.

    Args:
        county_number: County row number, used to preserve output order.
        geoid: County GEOID.
        county_geom_wkb: County geometry WKB in EPSG:5070.
        candidate_recreation: Candidate recreation geometry WKB, value, and
            full area records whose envelopes intersect the county.

    Returns:
        County row number, output row fields, and processing counters.

    Raises:
        ValueError: If the county geometry cannot be repaired into polygonal
            geometry.
    """
    stats = Counter()
    county_geom = repair_polygonal_geometry(shapely.from_wkb(county_geom_wkb))
    if county_geom is None:
        raise ValueError(f"County geometry cannot be repaired: GEOID={geoid}")

    proportional_value = 0.0
    if not candidate_recreation:
        stats["counties_without_candidates"] += 1
    for recreation_wkb, recreation_value, recreation_area in candidate_recreation:
        if recreation_area <= 0:
            stats["zero_area_recreation_candidates"] += 1
            continue

        recreation_geom = repair_polygonal_geometry(shapely.from_wkb(recreation_wkb))
        if recreation_geom is None:
            stats["invalid_recreation_candidates"] += 1
            continue

        intersection = shapely.intersection(recreation_geom, county_geom)
        intersection = polygonal_multipolygon(intersection)
        if intersection is None:
            stats["zero_area_intersections"] += 1
            continue

        intersection_area = intersection.area
        if intersection_area <= 0:
            stats["zero_area_intersections"] += 1
            continue

        proportional_value += recreation_value * (intersection_area / recreation_area)
        stats["positive_area_intersections"] += 1

    if proportional_value == 0:
        stats["zero_value_counties"] += 1
    else:
        stats["valued_counties"] += 1

    output_row = {
        GEOID_FIELD: geoid,
        OUT_VALUE_FIELD: proportional_value,
        "geometry": shapely.to_wkb(county_geom),
    }
    return county_number, output_row, stats


def main() -> None:
    """Run the recreation value by county preparation workflow."""
    out_path = _timestamped_output_path()
    step_bar = tqdm(total=7, desc="Prepare recreation by county", unit="step")

    step_bar.set_description("Read recreation")
    recreation = gpd.read_file(RECREATION_VECTOR_PATH)
    if recreation.empty:
        raise ValueError(f"Recreation vector is empty: {RECREATION_VECTOR_PATH}")
    _require_fields(recreation, {SOURCE_VALUE_FIELD}, RECREATION_VECTOR_PATH)
    step_bar.update()

    step_bar.set_description("Read counties")
    counties = gpd.read_file(COUNTY_VECTOR_PATH)
    if counties.empty:
        raise ValueError(f"County vector is empty: {COUNTY_VECTOR_PATH}")
    _require_fields(counties, {GEOID_FIELD}, COUNTY_VECTOR_PATH)
    step_bar.update()

    step_bar.set_description("Project to EPSG:5070")
    recreation = _project_to_area_crs(recreation, RECREATION_VECTOR_PATH)
    counties = _project_to_area_crs(counties, COUNTY_VECTOR_PATH)
    step_bar.update()

    step_bar.set_description("Prepare recreation index")
    recreation, all_stats = _prepare_recreation_features(recreation)
    if recreation.empty:
        raise ValueError("No nonzero recreation features remain after filtering.")
    step_bar.update()

    step_bar.set_description("Build county jobs")
    jobs = _build_county_jobs(counties, recreation)
    step_bar.update()

    print(
        f"Workers={N_WORKERS:,} counties={len(jobs):,} "
        f"recreation_features={len(recreation):,}",
        flush=True,
    )
    print(f"Output: {out_path}", flush=True)

    step_bar.set_description("Allocate values")
    output_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_process_county, *job) for job in jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Allocate recreation values",
            unit="county",
        ):
            county_number, output_row, stats = future.result()
            all_stats.update(stats)
            output_rows.append((county_number, output_row))
    step_bar.update()

    output_rows.sort(key=lambda row: row[0])
    output_feature_rows = []
    for _, output_row in output_rows:
        output_row = dict(output_row)
        output_row["geometry"] = shapely.from_wkb(output_row["geometry"])
        output_feature_rows.append(output_row)

    output_gdf = gpd.GeoDataFrame(
        output_feature_rows,
        columns=[GEOID_FIELD, OUT_VALUE_FIELD, "geometry"],
        geometry="geometry",
        crs=AREA_CRS,
    )

    step_bar.set_description("Write recreation by county")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_gdf.to_file(out_path, layer=OUT_STEM, driver="GPKG", index=False)
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
