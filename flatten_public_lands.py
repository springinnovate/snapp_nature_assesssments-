from concurrent.futures import ThreadPoolExecutor, as_completed
from os import cpu_count

import geopandas as gpd
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

PATH = r"data\PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001\PADUS4_1Geodatabase.gdb"
LAYER_NAME = "PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement"
OUT_PATH = # commenting out so it crashes we don't seem to use this file anymore r"data\dissolved.gpkg"
OUT_LAYER = # commenting out so it crashes we don't seem to use this file anymore "dissolved"
CHUNK_SIZE = 100
N_WORKERS = max(1, cpu_count() - 1)


def merge_field(values):
    nonblank = values[~values.isna()]
    nonblank = nonblank[nonblank.astype(str).str.strip() != ""]
    unique = nonblank.unique()

    if len(unique) == 1:
        return unique[0]

    return ""


def polygonal_only(geom):
    geom = shapely.make_valid(geom)

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        parts = [
            part for part in geom.geoms if part.geom_type in ("Polygon", "MultiPolygon")
        ]

        if parts:
            return shapely.union_all(parts)

    return None


def merge_component(component, idx, gdf, columns, geometry_name):
    group = gdf.iloc[idx]

    row = {column: merge_field(group[column]) for column in columns}

    row["component"] = component

    if len(group) == 1:
        geom = group[geometry_name].iloc[0]
        row[geometry_name] = (
            shapely.make_valid(geom) if not shapely.is_valid(geom) else geom
        )
        row[geometry_name] = polygonal_only(row[geometry_name])
        return row

    geoms = group[geometry_name].values

    try:
        row[geometry_name] = shapely.union_all(geoms)
        row[geometry_name] = polygonal_only(row[geometry_name])
        return row
    except shapely.errors.GEOSException as exc:
        print(
            f"Union failed for component {component} with {len(group):,} features; retrying make_valid. {exc}"
        )

    geoms = shapely.make_valid(geoms)

    try:
        row[geometry_name] = shapely.union_all(geoms)
        row[geometry_name] = polygonal_only(row[geometry_name])
        return row
    except shapely.errors.GEOSException as exc:
        print(
            f"make_valid union failed for component {component}; retrying buffer(0). {exc}"
        )

    row[geometry_name] = shapely.union_all(shapely.buffer(geoms, 0))
    row[geometry_name] = polygonal_only(row[geometry_name])
    return row


def main():
    with tqdm(total=9) as pbar:
        pbar.set_description("Reading layer")
        gdf = gpd.read_file(PATH, layer=LAYER_NAME)
        pbar.update()

        print(f"CRS: {gdf.crs}")
        print(f"Features: {len(gdf):,}")
        print(f"Workers: {N_WORKERS:,}")

        pbar.set_description("Building spatial index")
        sindex = gdf.sindex
        pbar.update()

        def query_chunk(start):
            stop = min(start + CHUNK_SIZE, len(gdf))
            pairs = sindex.query(gdf.geometry.iloc[start:stop], predicate="intersects")

            left = pairs[0] + start
            right = pairs[1]

            mask = left < right
            return left[mask], right[mask]

        pbar.set_description("Querying intersections")
        starts = list(range(0, len(gdf), CHUNK_SIZE))
        left_parts = []
        right_parts = []

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(query_chunk, start) for start in starts]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Intersection chunks",
            ):
                left, right = future.result()
                left_parts.append(left)
                right_parts.append(right)

        i = np.concatenate(left_parts)
        j = np.concatenate(right_parts)
        pbar.update()

        print(f"Intersecting pairs: {len(i):,}")
        print(f"Average pairs per feature: {len(i) / len(gdf):.2f}")

        pbar.set_description("Building graph")
        n = len(gdf)
        graph = coo_matrix(
            (np.ones(len(i) * 2, dtype=bool), (np.r_[i, j], np.r_[j, i])),
            shape=(n, n),
        )
        pbar.update()

        pbar.set_description("Finding components")
        _, labels = connected_components(graph, directed=False)
        gdf["component"] = labels
        pbar.update()

        print(f'Components: {gdf["component"].nunique():,}')

        pbar.set_description("Preparing dissolve groups")
        geometry_name = gdf.geometry.name
        crs = gdf.crs
        columns = [
            column
            for column in gdf.columns
            if column not in (geometry_name, "component")
        ]

        groups = gdf.groupby("component", sort=False).indices
        sizes = np.array([len(idx) for idx in groups.values()])

        print(f"Singleton components: {(sizes == 1).sum():,}")
        print(f"Multi-feature components: {(sizes > 1).sum():,}")
        print(f"Largest component: {sizes.max():,}")

        singletons = [
            (component, idx) for component, idx in groups.items() if len(idx) == 1
        ]

        multis = [(component, idx) for component, idx in groups.items() if len(idx) > 1]

        pbar.update()

        rows = []

        pbar.set_description("Copying singletons")
        for component, idx in tqdm(singletons, desc="Singleton components"):
            rows.append(
                merge_component(
                    component,
                    idx,
                    gdf,
                    columns,
                    geometry_name,
                )
            )
        pbar.update()

        pbar.set_description("Dissolving multi-feature components")
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [
                executor.submit(
                    merge_component,
                    component,
                    idx,
                    gdf,
                    columns,
                    geometry_name,
                )
                for component, idx in multis
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Dissolve multi-components",
            ):
                rows.append(future.result())

        out = gpd.GeoDataFrame(rows, geometry=geometry_name, crs=crs)
        out = out[out.geometry.notna()].copy()
        out = out.sort_values("component").reset_index(drop=True)
        pbar.update()

        pbar.set_description("Writing")
        out.to_file(OUT_PATH, layer=OUT_LAYER, driver="GPKG")
        pbar.update()


if __name__ == "__main__":
    main()
