from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count
import time

import geopandas as gpd
import shapely
from tqdm import tqdm

INPUT_PATH = r'data\PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001\PADUS4_1Geodatabase.gdb'
INPUT_LAYER = 'PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement'
OUT_PATH = r'data\padus_no_holes.gpkg'
OUT_LAYER = 'padus_no_holes'

N_WORKERS = min(4, max(1, cpu_count() - 1))
CHUNK_SIZE = 1_000
SLOW_FEATURE_SECONDS = 5


def polygonal_parts(geom):
    if geom is None or geom.is_empty:
        return []

    if not geom.is_valid:
        geom = shapely.make_valid(geom)

    if geom.geom_type == 'Polygon':
        return [geom]

    if geom.geom_type == 'MultiPolygon':
        return list(geom.geoms)

    if geom.geom_type == 'GeometryCollection':
        parts = []

        for part in geom.geoms:
            if part.geom_type == 'Polygon':
                parts.append(part)
            elif part.geom_type == 'MultiPolygon':
                parts.extend(part.geoms)

        return parts

    return []


def remove_holes(geom):
    parts = [
        shapely.Polygon(part.exterior)
        for part in polygonal_parts(geom)
    ]

    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    return shapely.MultiPolygon(parts)


def process_chunk(items):
    results = []
    slow_features = []

    for idx, geom in items:
        start = time.time()
        cleaned = remove_holes(geom)
        elapsed = time.time() - start

        if elapsed >= SLOW_FEATURE_SECONDS:
            slow_features.append((
                idx,
                elapsed,
                geom.geom_type if geom is not None else None,
                geom.is_valid if geom is not None else None,
                len(geom.geoms) if geom is not None and hasattr(geom, 'geoms') else None,
            ))

        results.append((idx, cleaned))

    return results, slow_features


def main():
    with tqdm(total=4) as pbar:
        pbar.set_description('Reading layer')
        gdf = gpd.read_file(INPUT_PATH, layer=INPUT_LAYER)
        pbar.update()

        print(f'Input features: {len(gdf):,}')
        print(f'CRS: {gdf.crs}')
        print(f'Workers: {N_WORKERS:,}')
        print(f'Chunk size: {CHUNK_SIZE:,}')

        pbar.set_description('Removing holes')

        chunks = [
            list(zip(
                range(start, min(start + CHUNK_SIZE, len(gdf))),
                gdf.geometry.iloc[start:start + CHUNK_SIZE].values,
            ))
            for start in range(0, len(gdf), CHUNK_SIZE)
        ]

        cleaned = [None] * len(gdf)
        slow_features = []

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [
                executor.submit(process_chunk, chunk)
                for chunk in chunks
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc='Geometry chunks',
            ):
                chunk_results, chunk_slow_features = future.result()

                for idx, geom in chunk_results:
                    cleaned[idx] = geom

                slow_features.extend(chunk_slow_features)

        gdf['geometry'] = cleaned
        pbar.update()

        if slow_features:
            print('\nSlow features:')
            for idx, elapsed, geom_type, is_valid, part_count in sorted(
                slow_features,
                key=lambda item: item[1],
                reverse=True,
            ):
                print(
                    f'  index={idx:,} '
                    f'seconds={elapsed:.2f} '
                    f'type={geom_type} '
                    f'is_valid={is_valid} '
                    f'parts={part_count}'
                )

        pbar.set_description('Dropping empty geometries')
        before = len(gdf)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        pbar.update()

        print(f'\nDropped features: {before - len(gdf):,}')
        print(f'Output features: {len(gdf):,}')

        pbar.set_description('Writing')
        gdf.to_file(OUT_PATH, layer=OUT_LAYER, driver='GPKG')
        pbar.update()


if __name__ == '__main__':
    main()
