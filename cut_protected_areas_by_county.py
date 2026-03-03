from pathlib import Path
import logging

import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
from shapely import make_valid
from ecoshard import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(filename)s:%(lineno)d %(message)s",
)

tqdm.pandas()
OUT_DIR = "./output"
INTERMEDIATE_DIR = "./intermediate"

PUBLIC_LANDS_VECTOR_PATH = "./data/public_lands_only_fixed.gpkg"
COUNTIES_VECTOR_PATH = "./data/gz_2010_us_050_00_5m/gz_2010_us_050_00_5m.shp"
CUT_PUBLIC_LANDS_VECTOR_PATH = (
    f"{OUT_DIR}/public_lands_cut_by_gz_2010_us_050_00_5m.gpkg"
)

SIMPLIFY_TOLERANCE_IN_TARGET_UNITS = 50


def _layer_from_path(path: str) -> str:
    return Path(path).stem


def _read(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def _write(gdf: gpd.GeoDataFrame, path: str, layer: str | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".gpkg":
        gdf.to_file(path, driver="GPKG", layer=layer or p.stem, index=False)
    else:
        gdf.to_file(path, index=False)


def fix_geometries(in_path: str, out_path: str, layer: str) -> None:
    gdf = _read(in_path)
    gdf.geometry = make_valid(gdf.geometry)
    _write(gdf, out_path, layer=layer)


def reproject(in_path: str, out_path: str, target_crs, layer: str) -> None:
    gdf = _read(in_path)
    gdf = gdf.to_crs(target_crs)
    _write(gdf, out_path, layer=layer)


def simplify(
    in_path: str, out_path: str, tolerance: float, layer: str | None = None
) -> None:
    gdf = _read(in_path)
    gdf.geometry = gdf.geometry.simplify(tolerance, preserve_topology=True)
    _write(gdf, out_path, layer=layer)


def drop_zero_area(in_path: str, out_path: str, layer: str | None = None) -> None:
    gdf = _read(in_path)
    gdf = gdf[gdf.geometry.notna() & (gdf.geometry.area > 0)].copy()
    _write(gdf, out_path, layer=layer)


def cut_public_lands_by_counties(
    public_lands_path: str,
    counties_path: str,
    out_path: str,
    out_layer: str = "public_lands_cut",
    county_keep_cols: list[str] | None = None,
) -> None:
    public_lands = _read(public_lands_path).reset_index(drop=True)
    counties = _read(counties_path).reset_index(drop=True)

    public_lands["_pl_id"] = public_lands.index

    keep_cols = county_keep_cols or ["geometry"]
    keep_cols = [c for c in keep_cols if c in counties.columns]
    if "geometry" not in keep_cols:
        keep_cols = ["geometry"] + keep_cols

    hits = gpd.sjoin(
        public_lands[["geometry", "_pl_id"]],
        counties[["geometry"]],
        how="inner",
        predicate="intersects",
    )

    pl_idx = hits.index.unique()
    cty_idx = hits["index_right"].unique()

    public_lands_sub = public_lands.loc[pl_idx].copy()
    counties_sub = counties.loc[cty_idx, keep_cols].copy()

    cut = gpd.overlay(
        public_lands_sub, counties_sub, how="intersection", keep_geom_type=False
    )
    cut = cut[cut.geometry.notna() & (cut.geometry.area > 0)].copy()

    for col in cut.columns:
        if col != "geometry" and pd.api.types.is_object_dtype(cut[col]):
            cut[col] = cut[col].astype(str)

    _write(cut, out_path, layer=out_layer)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = Path(INTERMEDIATE_DIR)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    task_graph = taskgraph.TaskGraph(str(intermediate_dir), 2, 15.0)

    pl_fixed = str(intermediate_dir / "public_lands_fixed.gpkg")
    cty_fixed = str(intermediate_dir / "counties_fixed.gpkg")

    fix_public_lands_task = task_graph.add_task(
        func=fix_geometries,
        args=(PUBLIC_LANDS_VECTOR_PATH, pl_fixed, "public_lands_fixed"),
        target_path_list=[pl_fixed],
        task_name="fix public lands",
    )
    fix_counties_task = task_graph.add_task(
        func=fix_geometries(COUNTIES_VECTOR_PATH, cty_fixed, "counties_fixed"),
        target_path_list=[cty_fixed],
        task_name="fix counties",
    )

    fix_public_lands_task.join()
    work_crs = _read(pl_fixed).crs

    cty_reproj = str(intermediate_dir / "counties_reprojected_to_public_lands.gpkg")
    reproject_counties_task = task_graph.add_task(
        func=reproject,
        args=(cty_fixed, cty_reproj, work_crs, "counties_reprojected"),
        target_path_list=[cty_reproj],
        dependent_task_list=[fix_counties_task],
        task_name="reproject fixed counties",
    )

    pl_simpl = str(intermediate_dir / "public_lands_simplified.gpkg")
    public_lands_simplified_task = task_graph.add_task(
        func=simplify,
        args=(
            pl_fixed,
            pl_simpl,
            SIMPLIFY_TOLERANCE_IN_TARGET_UNITS,
            "public_lands_simplified",
        ),
        target_path_list=[pl_simpl],
        dependent_task_list=[fix_public_lands_task],
        task_name="simplify public lands",
    )

    pl_clean = str(intermediate_dir / "public_lands_nonzero_area.gpkg")
    cty_clean = str(intermediate_dir / "counties_nonzero_area.gpkg")
    clean_public_lands_task = task_graph.add_task(
        func=drop_zero_area,
        args=(pl_simpl, pl_clean, "public_lands_nonzero_area"),
        target_path_list=[pl_clean],
        dependent_task_list=[public_lands_simplified_task],
        task_name="drop zero area public_lands_nonzero_area",
    )
    clean_counties_task = task_graph.add_task(
        func=drop_zero_area,
        args=(cty_reproj, cty_clean, "counties_nonzero_area"),
        target_path_list=[cty_clean],
        dependent_task_list=[reproject_counties_task],
        task_name="drop zero area counties_nonzero_area",
    )

    task_graph.add_task(
        func=cut_public_lands_by_counties,
        args=(
            pl_clean,
            cty_clean,
            CUT_PUBLIC_LANDS_VECTOR_PATH,
            "public_lands_cut",
            ["geometry"],
        ),
        target_path_list=[CUT_PUBLIC_LANDS_VECTOR_PATH],
        dependent_task_list=[clean_public_lands_task, clean_counties_task],
        task_name="cut public lands by county",
    )


if __name__ == "__main__":
    main()
