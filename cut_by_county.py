from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from os import cpu_count
from threading import Lock
import time

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

PADUS_PATH = r'data\padus_50_states_export.gpkg'
COUNTY_PATH = r'data\tl_2024_us_county_50_states.gpkg'
OUT_PATH = r'data\county_dissolved_public_lands.gpkg'
OUT_LAYER = 'county_dissolved_public_lands'

N_WORKERS = min(4, max(1, cpu_count() - 1))
SLOW_COUNTY_SECONDS = 10
HEARTBEAT_SECONDS = 15

COUNTY_STATUS = {}
COUNTY_STATUS_LOCK = Lock()


def set_county_status(geoid, stage, **stats):
    now = time.time()

    with COUNTY_STATUS_LOCK:
        started = COUNTY_STATUS.get(geoid, {}).get('started', now)
        COUNTY_STATUS[geoid] = {
            'stage': stage,
            'started': started,
            'updated': now,
            **stats,
        }


def clear_county_status(geoid):
    with COUNTY_STATUS_LOCK:
        COUNTY_STATUS.pop(geoid, None)


def print_county_status():
    now = time.time()

    with COUNTY_STATUS_LOCK:
        statuses = list(COUNTY_STATUS.items())

    for geoid, status in sorted(statuses):
        elapsed = now - status['started']
        idle = now - status['updated']
        fields = ' '.join(
            f'{key}={value:,}' if isinstance(value, int) else f'{key}={value}'
            for key, value in status.items()
            if key not in ('stage', 'started', 'updated')
        )

        tqdm.write(
            f'RUNNING GEOID={geoid} '
            f'elapsed={elapsed:.1f}s '
            f'idle={idle:.1f}s '
            f'stage={status["stage"]} '
            f'{fields}'
        )


def merge_field(values):
    nonblank = values[~values.isna()]
    nonblank = nonblank[nonblank.astype(str).str.strip() != '']
    unique = nonblank.unique()

    if len(unique) == 1:
        return unique[0]

    return ''


def polygonal_only(geom):
    if geom is None or geom.is_empty:
        return None

    if not geom.is_valid:
        geom = shapely.make_valid(geom)

    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom

    if geom.geom_type == 'GeometryCollection':
        parts = [
            part
            for part in geom.geoms
            if part.geom_type in ('Polygon', 'MultiPolygon') and not part.is_empty
        ]

        if parts:
            return shapely.union_all(parts)

    return None


def dissolve_county(args):
    geoid, county_geom, padus, padus_sindex, columns, geometry_name = args
    start_time = time.time()

    set_county_status(geoid, 'querying PAD-US index')

    candidate_idx = padus_sindex.query(county_geom, predicate='intersects')
    county_padus = padus.iloc[candidate_idx].copy()

    if len(county_padus) == 0:
        clear_county_status(geoid)
        return [], None

    set_county_status(
        geoid,
        'clipping candidates',
        candidates=len(candidate_idx),
    )

    county_padus[geometry_name] = shapely.intersection(
        county_padus[geometry_name].values,
        county_geom,
    )

    set_county_status(
        geoid,
        'filtering polygonal results',
        candidates=len(candidate_idx),
    )

    county_padus[geometry_name] = county_padus[geometry_name].map(polygonal_only)
    county_padus = county_padus[county_padus[geometry_name].notna()].copy()
    county_padus = county_padus[~county_padus[geometry_name].is_empty].copy()
    county_padus = county_padus.reset_index(drop=True)

    if len(county_padus) == 0:
        clear_county_status(geoid)
        return [], None

    set_county_status(
        geoid,
        'finding intersecting clipped polygons',
        candidates=len(candidate_idx),
        clipped=len(county_padus),
    )

    pairs = county_padus.sindex.query(
        county_padus[geometry_name],
        predicate='intersects',
    )

    i, j = pairs
    mask = i < j
    i = i[mask]
    j = j[mask]

    n = len(county_padus)

    set_county_status(
        geoid,
        'building connected components',
        candidates=len(candidate_idx),
        clipped=n,
        pairs=len(i),
    )

    if len(i) == 0:
        labels = np.arange(n)
    else:
        graph = coo_matrix(
            (np.ones(len(i) * 2, dtype=bool), (np.r_[i, j], np.r_[j, i])),
            shape=(n, n),
        )

        _, labels = connected_components(graph, directed=False)

    county_padus['component'] = labels

    groups = list(county_padus.groupby('component', sort=False).indices.items())

    set_county_status(
        geoid,
        'dissolving components',
        candidates=len(candidate_idx),
        clipped=n,
        pairs=len(i),
        components=len(groups),
    )

    rows = []

    for group_number, (component, idx) in enumerate(groups, start=1):
        set_county_status(
            geoid,
            'dissolving components',
            candidates=len(candidate_idx),
            clipped=n,
            pairs=len(i),
            components=len(groups),
            component=group_number,
        )

        group = county_padus.iloc[idx]

        row = {column: merge_field(group[column]) for column in columns}

        row['GEOID'] = geoid
        row['component'] = component

        if len(group) == 1:
            geom = group[geometry_name].iloc[0]
        else:
            try:
                geom = shapely.union_all(group[geometry_name].values)
            except shapely.errors.GEOSException:
                geoms = shapely.make_valid(group[geometry_name].values)

                try:
                    geom = shapely.union_all(geoms)
                except shapely.errors.GEOSException:
                    geom = shapely.union_all(shapely.buffer(geoms, 0))

        row[geometry_name] = polygonal_only(geom)

        if row[geometry_name] is not None and not row[geometry_name].is_empty:
            rows.append(row)

    elapsed = time.time() - start_time
    slow_county = None

    if elapsed >= SLOW_COUNTY_SECONDS:
        slow_county = {
            'GEOID': geoid,
            'seconds': elapsed,
            'candidate_count': len(candidate_idx),
            'clipped_count': len(county_padus),
            'component_count': len(groups),
        }

    clear_county_status(geoid)
    return rows, slow_county


def main():
    with tqdm(total=6) as pbar:
        pbar.set_description('Reading PAD-US')
        padus = gpd.read_file(PADUS_PATH)
        pbar.update()

        pbar.set_description('Reading counties')
        counties = gpd.read_file(COUNTY_PATH)
        pbar.update()

        pbar.set_description('Preparing data')
        counties = counties[['GEOID', 'geometry']].to_crs(padus.crs)
        geometry_name = padus.geometry.name

        padus = padus[padus[geometry_name].notna()].copy()
        padus = padus[~padus[geometry_name].is_empty].copy()
        padus = padus.reset_index(drop=True)

        columns = [
            column
            for column in padus.columns
            if column not in (geometry_name, 'GEOID', 'component')
        ]

        pbar.update()

        print(f'PAD-US features: {len(padus):,}')
        print(f'County features: {len(counties):,}')
        print(f'Workers: {N_WORKERS:,}')
        print(f'CRS: {padus.crs}')

        pbar.set_description('Building PAD-US spatial index')
        padus_sindex = padus.sindex
        pbar.update()

        pbar.set_description('Dissolving by county')

        jobs = [
            (
                county.GEOID,
                county.geometry,
                padus,
                padus_sindex,
                columns,
                geometry_name,
            )
            for county in counties.itertuples(index=False)
        ]

        rows = []
        slow_counties = []

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(dissolve_county, job): job[0]
                for job in jobs
            }

            pending = set(futures)
            last_heartbeat = time.time()

            with tqdm(total=len(futures), desc='Counties') as county_pbar:
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
                            tqdm.write(
                                f"Slow county GEOID={slow_county['GEOID']} "
                                f"seconds={slow_county['seconds']:.2f} "
                                f"candidates={slow_county['candidate_count']:,} "
                                f"clipped={slow_county['clipped_count']:,} "
                                f"components={slow_county['component_count']:,}"
                            )

        pbar.update()

        if slow_counties:
            print('\nSlowest counties:')
            for county in sorted(
                slow_counties,
                key=lambda item: item['seconds'],
                reverse=True,
            )[:25]:
                print(
                    f"  GEOID={county['GEOID']} "
                    f"seconds={county['seconds']:.2f} "
                    f"candidates={county['candidate_count']:,} "
                    f"clipped={county['clipped_count']:,} "
                    f"components={county['component_count']:,}"
                )

        pbar.set_description('Writing')
        out = gpd.GeoDataFrame(rows, geometry=geometry_name, crs=padus.crs)
        out = out[out.geometry.notna()].copy()
        out = out[~out.geometry.is_empty].copy()
        out = out.reset_index(drop=True)
        out.to_file(OUT_PATH, layer=OUT_LAYER, driver='GPKG')
        pbar.update()


if __name__ == '__main__':
    main()
