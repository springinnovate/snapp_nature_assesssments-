from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from os import cpu_count
from threading import Lock
import time

import geopandas as gpd
import numpy as np
import shapely
from tqdm import tqdm

PADUS_PATH = r"data\padus_50_states_export.gpkg"
COUNTY_PATH = r"data\tl_2024_us_county_50_states.gpkg"
OUT_PATH = r"data\padus_50_states_cut_by_county.gpkg"
OUT_LAYER = "padus_50_states_cut_by_county"

N_WORKERS = min(4, max(1, cpu_count() - 1))
SLOW_COUNTY_SECONDS = 60
HEARTBEAT_SECONDS = 5

MANG_TYPE_COLUMN = "Mang_Type"

PRIORITY_MANG_TYPES = {
    "Federal",
    "State",
    "Local Government",
    "Regional Agency Special District",
    "Joint",
    "Territorial",
}

COUNTY_STATUS = {}
COUNTY_STATUS_LOCK = Lock()


def format_stats(stats):
    return " ".join(
        f"{key}={value:,}" if isinstance(value, int) else f"{key}={value}"
        for key, value in stats.items()
    )


def log_line(message):
    with COUNTY_STATUS_LOCK:
        print(message, flush=True)


def set_county_status(geoid, stage, **stats):
    now = time.time()

    with COUNTY_STATUS_LOCK:
        started = COUNTY_STATUS.get(geoid, {}).get("started", now)
        COUNTY_STATUS[geoid] = {
            "stage": stage,
            "started": started,
            "updated": now,
            **stats,
        }

        elapsed = now - started
        fields = format_stats(stats)
        if elapsed > SLOW_COUNTY_SECONDS:
            print(
                f"STAGE GEOID={geoid} "
                f"elapsed={elapsed:.1f}s "
                f"stage={stage} "
                f"{fields}",
                flush=True,
            )


def clear_county_status(geoid):
    with COUNTY_STATUS_LOCK:
        COUNTY_STATUS.pop(geoid, None)


def print_county_status():
    now = time.time()

    with COUNTY_STATUS_LOCK:
        statuses = list(COUNTY_STATUS.items())

    for geoid, status in sorted(statuses):
        elapsed = now - status["started"]
        idle = now - status["updated"]
        fields = format_stats(
            {
                key: value
                for key, value in status.items()
                if key not in ("stage", "started", "updated")
            }
        )

        log_line(
            f"RUNNING GEOID={geoid} "
            f"elapsed={elapsed:.1f}s "
            f"idle={idle:.1f}s "
            f'stage={status["stage"]} '
            f"{fields}"
        )


def has_priority_mang_type(value):
    if value is None or value is np.nan:
        return False

    return str(value).strip() in PRIORITY_MANG_TYPES


def polygonal_only(geom):
    if geom is None or geom.is_empty:
        return None

    if not geom.is_valid:
        geom = shapely.make_valid(geom)

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        parts = [
            part
            for part in geom.geoms
            if part.geom_type in ("Polygon", "MultiPolygon") and not part.is_empty
        ]

        if parts:
            return shapely.union_all(parts)

    return None


def has_polygonal_overlap(left, right):
    if left is None or right is None or left.is_empty or right.is_empty:
        return False

    if not shapely.intersects(left, right):
        return False

    try:
        overlap = shapely.intersection(left, right)
    except shapely.errors.GEOSException:
        overlap = shapely.intersection(
            shapely.make_valid(left),
            shapely.make_valid(right),
        )

    overlap = polygonal_only(overlap)

    return overlap is not None and not overlap.is_empty and overlap.area > 0


def resolve_county_overlaps(county_padus, geometry_name, geoid, candidate_count):
    pass_number = 0

    while len(county_padus) > 1:
        pass_number += 1

        set_county_status(
            geoid,
            "finding same-county overlaps",
            candidates=candidate_count,
            clipped=len(county_padus),
            pass_number=pass_number,
        )

        pairs = county_padus.sindex.query(
            county_padus[geometry_name],
            predicate="intersects",
        )

        i, j = pairs
        mask = i < j
        i = i[mask]
        j = j[mask]

        if len(i) == 0:
            break

        set_county_status(
            geoid,
            "removing same-county overlaps",
            candidates=candidate_count,
            clipped=len(county_padus),
            pairs=len(i),
            pass_number=pass_number,
        )

        changed = False
        geometry_idx = county_padus.columns.get_loc(geometry_name)
        mang_type_idx = county_padus.columns.get_loc(MANG_TYPE_COLUMN)

        for pair_number, (left, right) in enumerate(zip(i, j), start=1):
            left_geom = county_padus.iat[left, geometry_idx]
            right_geom = county_padus.iat[right, geometry_idx]

            if not has_polygonal_overlap(left_geom, right_geom):
                continue

            left_priority = has_priority_mang_type(
                county_padus.iat[left, mang_type_idx]
            )
            right_priority = has_priority_mang_type(
                county_padus.iat[right, mang_type_idx]
            )

            if left_priority:
                keep_idx = left
                cut_idx = right
            elif right_priority:
                keep_idx = right
                cut_idx = left
            else:
                keep_idx = right
                cut_idx = left

            keep_geom = county_padus.iat[keep_idx, geometry_idx]
            cut_geom = county_padus.iat[cut_idx, geometry_idx]

            try:
                new_geom = shapely.difference(cut_geom, keep_geom)
            except shapely.errors.GEOSException:
                new_geom = shapely.difference(
                    shapely.make_valid(cut_geom),
                    shapely.make_valid(keep_geom),
                )

            county_padus.iat[cut_idx, geometry_idx] = polygonal_only(new_geom)
            changed = True

            if pair_number % 1_000 == 0:
                set_county_status(
                    geoid,
                    "removing same-county overlaps",
                    candidates=candidate_count,
                    clipped=len(county_padus),
                    pairs=len(i),
                    pair=pair_number,
                    pass_number=pass_number,
                )

        county_padus = county_padus[county_padus[geometry_name].notna()].copy()
        county_padus = county_padus[~county_padus[geometry_name].is_empty].copy()
        county_padus = county_padus.reset_index(drop=True)

        if not changed:
            break

    return county_padus


def process_county(args):
    geoid, county_geom, padus, padus_sindex, geometry_name = args
    start_time = time.time()

    set_county_status(geoid, "querying PAD-US index")

    candidate_idx = padus_sindex.query(county_geom, predicate="intersects")
    county_padus = padus.iloc[candidate_idx].copy()

    set_county_status(geoid, f"{len(candidate_idx)} features intersects {geoid} county")

    if len(county_padus) == 0:
        clear_county_status(geoid)
        return [], None

    set_county_status(
        geoid,
        "clipping candidates to county",
        candidates=len(candidate_idx),
    )

    county_padus[geometry_name] = shapely.intersection(
        county_padus[geometry_name].values,
        county_geom,
    )

    set_county_status(
        geoid,
        "filtering county-clipped polygonal results",
        candidates=len(candidate_idx),
    )

    county_padus[geometry_name] = county_padus[geometry_name].map(polygonal_only)
    county_padus = county_padus[county_padus[geometry_name].notna()].copy()
    county_padus = county_padus[~county_padus[geometry_name].is_empty].copy()
    county_padus = county_padus.reset_index(drop=True)

    if len(county_padus) == 0:
        clear_county_status(geoid)
        return [], None

    clipped_count = len(county_padus)

    county_padus = resolve_county_overlaps(
        county_padus,
        geometry_name,
        geoid,
        len(candidate_idx),
    )

    if len(county_padus) == 0:
        clear_county_status(geoid)
        return [], None

    set_county_status(
        geoid,
        "building output rows",
        candidates=len(candidate_idx),
        clipped=clipped_count,
        remaining=len(county_padus),
    )

    rows = []

    for row_number, row in enumerate(county_padus.to_dict("records"), start=1):
        row["GEOID"] = geoid
        row["component"] = row_number

        geom = polygonal_only(row[geometry_name])

        if geom is not None and not geom.is_empty:
            row[geometry_name] = geom
            rows.append(row)

    elapsed = time.time() - start_time
    slow_county = None

    if elapsed >= SLOW_COUNTY_SECONDS:
        slow_county = {
            "GEOID": geoid,
            "seconds": elapsed,
            "candidate_count": len(candidate_idx),
            "clipped_count": clipped_count,
            "feature_count": len(rows),
        }

    clear_county_status(geoid)
    return rows, slow_county


def main():
    with tqdm(total=8) as pbar:
        pbar.set_description("Reading PAD-US")
        padus = gpd.read_file(PADUS_PATH)
        pbar.update()

        pbar.set_description("Filtering PAD-US geometries")
        geometry_name = padus.geometry.name
        null_count = padus[geometry_name].isna().sum()
        empty_count = padus[geometry_name].is_empty.sum()

        print(f"Null geometries: {null_count:,}", flush=True)
        print(f"Empty geometries: {empty_count:,}", flush=True)

        padus = padus[padus[geometry_name].notna()].copy()
        padus = padus[~padus[geometry_name].is_empty].copy()
        padus = padus.reset_index(drop=True)

        pbar.update()

        tolerance = 15

        pbar.set_description(f"Simplifying to {tolerance} meters")
        print(
            f"invalid before simplify: {(~padus.geometry.is_valid).sum():,}",
            flush=True,
        )

        simplified = padus.geometry.simplify(tolerance, preserve_topology=False)

        print(
            f"invalid after simplify: {(~simplified.is_valid).sum():,}",
            flush=True,
        )

        simplified = simplified.make_valid()

        print(
            f"invalid after repair: {(~simplified.is_valid).sum():,}",
            flush=True,
        )

        padus.geometry = simplified
        pbar.update()

        pbar.set_description("Reading counties")
        counties = gpd.read_file(COUNTY_PATH)
        pbar.update()

        pbar.set_description("Preparing data")
        counties = counties[["GEOID", "geometry"]].to_crs(padus.crs)
        geometry_name = padus.geometry.name

        padus[geometry_name] = padus[geometry_name].map(polygonal_only)
        padus = padus[padus[geometry_name].notna()].copy()
        padus = padus[~padus[geometry_name].is_empty].copy()
        padus = padus.reset_index(drop=True)

        pbar.update()

        print(f"PAD-US features: {len(padus):,}", flush=True)
        print(f"County features: {len(counties):,}", flush=True)
        print(f"Workers: {N_WORKERS:,}", flush=True)
        print(f"CRS: {padus.crs}", flush=True)

        pbar.set_description("Building PAD-US spatial index")
        padus_sindex = padus.sindex
        pbar.update()

        pbar.set_description("Clipping by county")

        jobs = [
            (
                county.GEOID,
                county.geometry,
                padus,
                padus_sindex,
                geometry_name,
            )
            for county in counties.itertuples(index=False)
        ]

        rows = []
        slow_counties = []

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_county, job): job[0] for job in jobs}

            pending = set(futures)
            last_heartbeat = time.time()

            with tqdm(total=len(futures), desc="Counties") as county_pbar:
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=1,
                        return_when=FIRST_COMPLETED,
                    )

                    now = time.time()

                    if now - last_heartbeat >= HEARTBEAT_SECONDS:
                        print_county_status()
                        last_heartbeat = now

                    for future in done:
                        geoid = futures[future]
                        clear_county_status(geoid)

                        county_rows, slow_county = future.result()
                        rows.extend(county_rows)
                        county_pbar.update()

                        if slow_county is not None:
                            slow_counties.append(slow_county)
                            log_line(
                                f"Slow county GEOID={slow_county['GEOID']} "
                                f"seconds={slow_county['seconds']:.2f} "
                                f"candidates={slow_county['candidate_count']:,} "
                                f"clipped={slow_county['clipped_count']:,} "
                                f"features={slow_county['feature_count']:,}"
                            )

        pbar.update()

        if slow_counties:
            print("\nSlowest counties:", flush=True)
            for county in sorted(
                slow_counties,
                key=lambda item: item["seconds"],
                reverse=True,
            )[:25]:
                print(
                    f"  GEOID={county['GEOID']} "
                    f"seconds={county['seconds']:.2f} "
                    f"candidates={county['candidate_count']:,} "
                    f"clipped={county['clipped_count']:,} "
                    f"features={county['feature_count']:,}",
                    flush=True,
                )

        pbar.set_description("Writing")
        out = gpd.GeoDataFrame(rows, geometry=geometry_name, crs=padus.crs)
        out = out[out.geometry.notna()].copy()
        out = out[~out.geometry.is_empty].copy()
        out = out.reset_index(drop=True)
        out.to_file(OUT_PATH, layer=OUT_LAYER, driver="GPKG")
        pbar.update()


if __name__ == "__main__":
    main()
