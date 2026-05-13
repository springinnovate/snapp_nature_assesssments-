from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

PATH = r'data\PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001\PADUS4_1Geodatabase.gdb'
LAYER_NAME = 'PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement'
OUT_PATH = r'data\dissolved.gpkg'
OUT_LAYER = 'dissolved'
CHUNK_SIZE = 10_000
N_WORKERS = max(1, cpu_count() - 1)


def merge_field(values):
    nonblank = values[~values.isna()]
    nonblank = nonblank[nonblank.astype(str).str.strip() != '']
    unique = nonblank.unique()

    if len(unique) == 1:
        return unique[0]

    return ''


def dissolve_component(item):
    component, group, geometry_name, crs = item

    attrs = {
        column: merge_field(group[column])
        for column in group.columns
        if column not in (geometry_name, 'component')
    }

    attrs['component'] = component
    attrs[geometry_name] = shapely.union_all(group[geometry_name].values)

    return attrs


def main():
    with tqdm(total=8) as pbar:
        pbar.set_description('Reading layer')
        gdf = gpd.read_file(PATH, layer=LAYER_NAME)
        pbar.update()

        print(f'CRS: {gdf.crs}')
        print(f'Features: {len(gdf):,}')
        print(f'Workers: {N_WORKERS:,}')

        pbar.set_description('Building spatial index')
        sindex = gdf.sindex
        pbar.update()

        def query_chunk(start):
            stop = min(start + CHUNK_SIZE, len(gdf))
            pairs = sindex.query(gdf.geometry.iloc[start:stop], predicate='intersects')

            left = pairs[0] + start
            right = pairs[1]

            mask = left < right
            return left[mask], right[mask]

        pbar.set_description('Querying intersections')
        starts = list(range(0, len(gdf), CHUNK_SIZE))
        left_parts = []
        right_parts = []

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(query_chunk, start) for start in starts]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Intersection chunks',
            ):
                left, right = future.result()
                left_parts.append(left)
                right_parts.append(right)

        i = np.concatenate(left_parts)
        j = np.concatenate(right_parts)
        pbar.update()

        print(f'Intersecting pairs: {len(i):,}')
        print(f'Average pairs per feature: {len(i) / len(gdf):.2f}')

        pbar.set_description('Building graph')
        n = len(gdf)
        graph = coo_matrix(
            (np.ones(len(i) * 2, dtype=bool), (np.r_[i, j], np.r_[j, i])),
            shape=(n, n),
        )
        pbar.update()

        pbar.set_description('Finding components')
        _, labels = connected_components(graph, directed=False)
        gdf['component'] = labels
        pbar.update()

        print(f'Components: {gdf["component"].nunique():,}')

        pbar.set_description('Preparing dissolve jobs')
        geometry_name = gdf.geometry.name
        crs = gdf.crs
        jobs = [
            (component, group.copy(), geometry_name, crs)
            for component, group in gdf.groupby('component', sort=False)
        ]
        pbar.update()

        pbar.set_description('Dissolving')
        rows = []

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(dissolve_component, job) for job in jobs]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Dissolve components',
            ):
                rows.append(future.result())

        out = gpd.GeoDataFrame(rows, geometry=geometry_name, crs=crs)
        out = out.sort_values('component').reset_index(drop=True)
        pbar.update()

        pbar.set_description('Writing')
        out.to_file(OUT_PATH, layer=OUT_LAYER, driver='GPKG')
        pbar.update()


if __name__ == '__main__':
    main()
