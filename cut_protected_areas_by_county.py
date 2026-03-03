"""Cut public lands polygons by county boundaries.

This script prepares two vector layers (public lands and counties) and produces a
new vector layer where public lands geometries are intersected with county
polygons. Public lands features fully contained by a county remain unchanged,
while features that cross county boundaries are split into multiple features.

Processing is organized as a task graph with intermediate artifacts written to
`INTERMEDIATE_DIR`. The final cut dataset is written to `OUT_DIR`.

Typical pipeline:
  1. Fix invalid geometries in both layers.
  2. Reproject counties to the public lands CRS (work CRS).
  3. Simplify public lands geometries.
  4. Drop null and zero-area geometries.
  5. Spatially prefilter intersecting features (sjoin) and cut via overlay
     intersection.

Outputs are written as GeoPackage layers when the destination path ends in
`.gpkg`; otherwise, the driver is inferred by GeoPandas.
"""

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
    """Derive a default layer name from a filesystem path.

    For a path like `/some/dir/foo.gpkg`, this returns `foo`.

    Args:
        path: Path to a dataset file.

    Returns:
        The filename stem (final path component without extension).
    """
    return Path(path).stem


def _read(path: str) -> gpd.GeoDataFrame:
    """Read a vector dataset into a GeoDataFrame.

    Args:
        path: Path to a vector dataset readable by GeoPandas.

    Returns:
        A GeoDataFrame containing the dataset's features and geometry column.
    """
    return gpd.read_file(path)


def _write(gdf: gpd.GeoDataFrame, path: str, layer: str | None = None) -> None:
    """Write a GeoDataFrame to disk.

    If `path` ends with `.gpkg`, the output is written as a GeoPackage and the
    layer name is set to `layer` or (if unset) the stem of `path`. Otherwise,
    GeoPandas infers the driver from the file extension.

    Args:
        gdf: GeoDataFrame to write.
        path: Destination path.
        layer: Optional layer name (used for GeoPackage outputs only).

    Returns:
        None
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".gpkg":
        gdf.to_file(path, driver="GPKG", layer=layer or p.stem, index=False)
    else:
        gdf.to_file(path, index=False)


def fix_geometries(in_path: str, out_path: str, layer: str) -> None:
    """Fix invalid geometries and write the result to a new dataset.

    This uses `shapely.make_valid` to repair invalid geometries (e.g. self-
    intersections). The operation is applied to every feature geometry and the
    result is written to `out_path`.

    Args:
        in_path: Input vector dataset path.
        out_path: Output vector dataset path.
        layer: Output layer name (for GeoPackage outputs).

    Returns:
        None
    """
    gdf = _read(in_path)
    gdf.geometry = make_valid(gdf.geometry)
    _write(gdf, out_path, layer=layer)


def reproject(in_path: str, out_path: str, target_crs, layer: str) -> None:
    """Reproject a vector dataset to a target CRS and write the result.

    Args:
        in_path: Input vector dataset path.
        out_path: Output vector dataset path.
        target_crs: CRS passed to GeoPandas `to_crs` (e.g. EPSG int/string or
            CRS object).
        layer: Output layer name (for GeoPackage outputs).

    Returns:
        None
    """
    gdf = _read(in_path)
    gdf = gdf.to_crs(target_crs)
    _write(gdf, out_path, layer=layer)


def simplify(
    in_path: str, out_path: str, tolerance: float, layer: str | None = None
) -> None:
    """Simplify geometries and write the result.

    Uses Shapely geometry simplification with `preserve_topology=True`.

    Args:
        in_path: Input vector dataset path.
        out_path: Output vector dataset path.
        tolerance: Simplification tolerance in the dataset's coordinate units.
        layer: Optional output layer name (for GeoPackage outputs).

    Returns:
        None
    """
    gdf = _read(in_path)
    gdf.geometry = gdf.geometry.simplify(tolerance, preserve_topology=True)
    _write(gdf, out_path, layer=layer)


def drop_zero_area(in_path: str, out_path: str, layer: str | None = None) -> None:
    """Drop null and zero-area geometries and write the result.

    Note that `.area` is computed in the layer CRS; for meaningful area filtering
    this should be a projected CRS.

    Args:
        in_path: Input vector dataset path.
        out_path: Output vector dataset path.
        layer: Optional output layer name (for GeoPackage outputs).

    Returns:
        None
    """
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
    """Cut public lands polygons by county boundaries.

    The cut operation is implemented as an overlay intersection between public
    lands and county polygons. This yields:
      - One output feature for each public-lands portion within a county.
      - Public-lands features fully contained by a county remain geometrically
        unchanged (aside from possible vertex ordering).
      - Public-lands features that cross county boundaries are split into
        multiple output features, one per intersected county.

    Prior to overlay, a spatial join is used to restrict processing to
    intersecting features only.

    Args:
        public_lands_path: Path to the prepared public lands dataset.
        counties_path: Path to the prepared counties dataset (must be in the same
            CRS as `public_lands_path`).
        out_path: Destination path for the cut output dataset.
        out_layer: Layer name for GeoPackage outputs.
        county_keep_cols: Optional list of county columns to retain in the output
            (in addition to geometry). Columns not present in `counties` are
            ignored. If omitted, only county geometry is used.

    Returns:
        None
    """
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
    """Run the end-to-end cutting workflow using a TaskGraph pipeline.

    This function:
      - Creates output and intermediate directories.
      - Creates a TaskGraph to manage intermediate steps and dependencies.
      - Fixes geometries for both layers.
      - Reprojects counties to the public lands CRS.
      - Simplifies public lands geometries.
      - Drops null/zero-area geometries for both layers.
      - Cuts public lands by counties via overlay intersection and writes the
        final GeoPackage.

    Returns:
        None
    """
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
