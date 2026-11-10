"""Render the README's maps from an explicit completed assessment run.

Run from the repository root with the assessment's geospatial Python environment.
Inputs are opened read-only. Only PNG figures and their provenance JSON are written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PROJ_NETWORK", "OFF")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
runtime_share = Path(sys.prefix) / "Library/share"
for variable, directory in (("GDAL_DATA", "gdal"), ("PROJ_DATA", "proj")):
    if (runtime_share / directory).is_dir():
        os.environ.setdefault(variable, str(runtime_share / directory))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import numpy as np
from osgeo import gdal, osr
from pyproj import Transformer
import pyogrio
import shapely
from tqdm import tqdm

gdal.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
CRS = "EPSG:5070"
COLORS = {
    "paper": "#fbfaf7", "land": "#e9e9df", "water": "#438fb0",
    "context_water": "#dbe9ee", "public": "#397e5b",
    "conservation": "#786a9f", "bbb": "#c77837",
    "ink": "#263e43", "muted": "#566568", "boundary": "#a6b1aa",
    "excluded": "#ede2cc", "exclusion_edge": "#9a7844",
}
UTAH = (-113.05, 39.8, -111.15, 41.75)
MICHIGAN = (-90.5, 41.2, -83.7, 46.5)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
    "savefig.facecolor": COLORS["paper"], "figure.facecolor": COLORS["paper"],
    "hatch.linewidth": 0.45, "path.simplify": False,
})


def read_window(path, layer, bounds, columns=None):
    """Read a spatially indexed source window into the map projection.

    Args:
        path: GeoPackage path.
        layer: Source layer name.
        bounds: Bounds in EPSG:5070, or None for the full source.
        columns: Attribute columns to retain.

    Returns:
        A projected GeoDataFrame.
    """
    info = pyogrio.read_info(path, layer=layer)
    bbox = None if bounds is None else Transformer.from_crs(
        CRS, info["crs"], always_xy=True,
    ).transform_bounds(*bounds, densify_pts=41)
    return pyogrio.read_dataframe(path, layer=layer, bbox=bbox, columns=columns).to_crs(CRS)


def raster_mask(path, layer_name, bounds, width=2000, where=None):
    """Rasterize real source polygons at figure resolution without editing them.

    Args:
        path: Read-only vector source.
        layer_name: Polygon layer name.
        bounds: Map extent in EPSG:5070.
        width: Rendering width in pixels.
        where: Optional attribute filter.

    Returns:
        A boolean rendering mask, not an analytical raster.
    """
    minx, miny, maxx, maxy = bounds
    height = max(1, round(width * (maxy - miny) / (maxx - minx)))
    target = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
    target.SetGeoTransform((minx, (maxx - minx) / width, 0, maxy, 0, -(maxy - miny) / height))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(5070)
    target.SetProjection(srs.ExportToWkt())
    source = gdal.OpenEx(str(path), gdal.OF_VECTOR | gdal.OF_READONLY)
    layer = source.GetLayerByName(layer_name)
    native_bounds = Transformer.from_crs(
        CRS, layer.GetSpatialRef().ExportToWkt(), always_xy=True,
    ).transform_bounds(*bounds, densify_pts=41)
    layer.SetSpatialFilterRect(*native_bounds)
    if where is not None:
        layer.SetAttributeFilter(where)
    gdal.RasterizeLayer(target, [1], layer, burn_values=[1], options=["ALL_TOUCHED=FALSE"])
    result = target.ReadAsArray().astype(bool)
    source = target = None
    return result


def add_mask(ax, mask, bounds, color, zorder=2):
    """Draw a rendering mask with a transparent background.

    Args:
        ax: Matplotlib map axes.
        mask: Boolean image.
        bounds: Extent in EPSG:5070.
        color: Polygon fill color.
        zorder: Drawing order.
    """
    minx, miny, maxx, maxy = bounds
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([color]),
              extent=(minx, maxx, miny, maxy), interpolation="nearest", zorder=zorder)


def base_map(ax, counties, land, lakes, bounds, *, regional=False):
    """Draw geographic context using local county, U.S., and lake geometries.

    Args:
        ax: Matplotlib map axes.
        counties: Projected county geometries and names.
        land: Projected U.S. boundary geometry.
        lakes: Dissolved mapped exclusion geometry.
        bounds: Map extent in EPSG:5070.
        regional: Whether to use finer display boundaries.
    """
    window = shapely.box(*bounds)
    ax.set_facecolor(COLORS["paper"])
    local_land = shapely.intersection(land, window)
    gpd.GeoSeries([local_land], crs=CRS).plot(ax=ax, facecolor=COLORS["land"], edgecolor="none", zorder=0)
    local_lakes = shapely.intersection(lakes, window)
    if not local_lakes.is_empty:
        gpd.GeoSeries([local_lakes], crs=CRS).plot(ax=ax, facecolor=COLORS["context_water"], edgecolor="none", zorder=1)
    selected = counties.loc[counties.intersects(window)]
    lines = shapely.get_parts(shapely.boundary(shapely.simplify(selected.geometry.to_numpy(), 60 if regional else 250)))
    segments = [shapely.get_coordinates(line) for line in lines if not line.is_empty]
    ax.add_collection(LineCollection(segments, colors=COLORS["boundary"], linewidths=0.45 if regional else 0.18, zorder=4))
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()


def scale_bar(ax, bounds, kilometres):
    """Draw a scale bar in the projected map coordinates.

    Args:
        ax: Matplotlib map axes.
        bounds: EPSG:5070 map bounds.
        kilometres: Bar length in kilometres.
    """
    x = bounds[0] + (bounds[2] - bounds[0]) * 0.07
    y = bounds[1] + (bounds[3] - bounds[1]) * 0.05
    ax.plot([x, x + kilometres * 1000], [y, y], color=COLORS["ink"], linewidth=2, zorder=10)
    ax.text(x, y + (bounds[3] - bounds[1]) * 0.018, f"{kilometres:g} km", fontsize=10.5,
            va="bottom", zorder=11, path_effects=[pe.withStroke(linewidth=3, foreground=COLORS["paper"])])


def save_figure(fig, path):
    """Save a compact PNG and close the plotting figure.

    Args:
        fig: Completed Matplotlib figure.
        path: PNG destination.
    """
    fig.savefig(path, dpi=160, pil_kwargs={"optimize": True}, metadata={
        "Description": "Map rendered from local assessment data; see map_sources.json.",
    })
    plt.close(fig)


def main():
    """Render three explanatory maps and record exact local source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-run", type=Path, required=True,
                        help="Directory containing the completed six-product run_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/images")
    args = parser.parse_args()
    run_dir = args.assessment_run.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Choose a completed assessment run")
    products = {}
    for name in ("counties", "public_lands_by_county", "conservation_lands_by_county", "bbb_lands_by_county"):
        matches = list((run_dir / name).glob("*.gpkg"))
        if len(matches) != 1:
            raise ValueError(f"Expected one GeoPackage for {name}: {matches}")
        products[name] = matches[0]
    usa_path = ROOT / "data/analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg"
    water_path = ROOT / "data/analysis_inputs/hydrography/nhdfreshwater/nhd_freshwater_clipped_to_usa_2026_05_18_01_44_14.gpkg"
    water_layer = "nhd_freshwater_clipped_to_usa"
    rules_dir = ROOT / "data/workflow_assets/county_integration"
    rule_paths = [rules_dir / "great_lakes_nhd_features.csv", rules_dir / "inland_fisheries_exclusions.csv"]
    sources = [manifest_path, *products.values(), usa_path, water_path, *rule_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Read current county and reference geometries", flush=True)
    counties = read_window(products["counties"], "counties", None, ["GEOID", "STATEFP", "NAME"])
    if len(counties) != 3109 or not counties.GEOID.is_unique:
        raise ValueError("Expected 3,109 unique CONUS county equivalents")
    if set(counties.STATEFP).intersection({"02", "15", "60", "66", "69", "72", "78"}):
        raise ValueError("County input includes non-CONUS areas")
    land = shapely.union_all(read_window(usa_path, "usa_vector", None, []).geometry.to_numpy())
    lake_names = {}
    for path in rule_paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            lake_names.update({int(row["source_fid"]): row["lake"] for row in csv.DictReader(stream)})
    ids = ",".join(map(str, sorted(lake_names)))
    lake_filter = f"source_layer='NHDWaterbody' AND source_fid IN ({ids})"
    lake_rows = pyogrio.read_dataframe(water_path, layer=water_layer, columns=["source_fid"], where=lake_filter).to_crs(CRS)
    if set(lake_rows.source_fid) != set(lake_names):
        raise ValueError("Some mapped lake exclusion features are missing")
    lakes = shapely.union_all(lake_rows.geometry.to_numpy())
    grouped_lakes = defaultdict(list)
    for row in lake_rows.itertuples():
        grouped_lakes[lake_names[row.source_fid]].append(row.geometry)
    grouped_lakes = {name: shapely.union_all(parts) for name, parts in grouped_lakes.items()}
    to_map = Transformer.from_crs(4326, CRS, always_xy=True)
    utah_bounds = to_map.transform_bounds(*UTAH, densify_pts=41)
    michigan_bounds = to_map.transform_bounds(*MICHIGAN, densify_pts=41)
    bounds = counties.total_bounds
    padding = (bounds[2:] - bounds[:2]) * 0.015
    national_bounds = (*tuple(bounds[:2] - padding), *tuple(bounds[2:] + padding))

    print("Render CONUS counties and public lands", flush=True)
    fig = plt.figure(figsize=(12.8, 8.2))
    fig.text(.05, .95, "Counties and public lands", fontsize=21, weight="semibold")
    fig.text(.05, .912, "Contiguous United States and Washington, DC", fontsize=12, color=COLORS["muted"])
    ax = fig.add_axes((.025, .105, .95, .77))
    base_map(ax, counties, land, lakes, national_bounds)
    add_mask(ax, raster_mask(products["public_lands_by_county"], "public_lands_by_county", national_bounds), national_bounds, COLORS["public"])
    scale_bar(ax, national_bounds, 500)
    fig.legend(handles=[Patch(facecolor=COLORS["public"], label="Public-land footprint"),
                        Line2D([0], [0], color=COLORS["boundary"], lw=1, label="County boundary")],
               loc="lower center", bbox_to_anchor=(.5, .02), ncol=2, frameon=False, fontsize=11.5)
    save_figure(fig, args.output_dir / "assessment_geography.png")

    print("Render northern Utah land-category comparison", flush=True)
    fig = plt.figure(figsize=(13.4, 7.1))
    fig.text(.04, .95, "Three views of northern Utah", fontsize=21, weight="semibold")
    fig.text(.04, .902, "The same county boundaries, with separate public, conservation, and BBB footprints", fontsize=11.5, color=COLORS["muted"])
    panels = [("Public lands", "public_lands_by_county", "public"),
              ("Conservation lands", "conservation_lands_by_county", "conservation"),
              ("BBB candidates", "bbb_lands_by_county", "bbb")]
    for i, (title, product, color) in enumerate(panels):
        ax = fig.add_axes((.035 + i * .325, .13, .30, .69))
        base_map(ax, counties, land, lakes, utah_bounds, regional=True)
        add_mask(ax, raster_mask(products[product], product, utah_bounds, width=950), utah_bounds, COLORS[color])
        ax.set_title(title, fontsize=14, fontweight="semibold", color=COLORS[color], pad=10)
        local = counties.loc[counties.NAME.isin(["Davis", "Weber", "Salt Lake", "Tooele", "Utah"]) & counties.STATEFP.eq("49")]
        for row in local.itertuples():
            clipped = row.geometry.intersection(shapely.box(*utah_bounds))
            if clipped.is_empty:
                continue
            point = clipped.representative_point()
            ax.text(point.x, point.y, row.NAME, ha="center", va="center", fontsize=11,
                    zorder=8, path_effects=[pe.withStroke(linewidth=2.7, foreground=COLORS["paper"])])
        scale_bar(ax, utah_bounds, 40)
    fig.text(.5, .045, "County names shown for reference. Public and conservation categories can overlap.", ha="center", fontsize=11, color=COLORS["muted"])
    save_figure(fig, args.output_dir / "land_footprints_utah.png")

    print("Render water inputs and fisheries exclusions", flush=True)
    fig = plt.figure(figsize=(13.4, 7.7))
    fig.text(.04, .95, "Water inputs and fisheries exclusions", fontsize=21, weight="semibold")
    fig.text(.04, .904, "NHD water polygons within the U.S. assessment area", fontsize=12, color=COLORS["muted"])
    for i, (title, window, bar_length) in enumerate([
        ("Northern Utah", utah_bounds, 40), ("Lake Michigan region", michigan_bounds, 150),
    ]):
        ax = fig.add_axes((.04 + i * .49, .14, .43, .68))
        base_map(ax, counties, land, lakes, window, regional=True)
        water = raster_mask(water_path, water_layer, window, width=1500)
        excluded = raster_mask(water_path, water_layer, window, width=1500, where=lake_filter)
        add_mask(ax, water & ~excluded, window, COLORS["water"])
        clipped_lakes = shapely.intersection(lakes, shapely.box(*window))
        gpd.GeoSeries([clipped_lakes], crs=CRS).plot(ax=ax, facecolor=COLORS["excluded"],
            edgecolor=COLORS["exclusion_edge"], linewidth=.55, hatch="///", zorder=3)
        label_name = "Great Salt Lake" if i == 0 else "Lake Michigan"
        label_geometry = shapely.intersection(grouped_lakes[label_name], shapely.box(*window))
        point = label_geometry.representative_point()
        ax.text(point.x, point.y, label_name.replace(" ", "\n", 1), ha="center", va="center", fontsize=12,
                color=COLORS["ink"], zorder=8, path_effects=[pe.withStroke(linewidth=3, foreground=COLORS["excluded"])])
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=10)
        ax.set_xlim(window[0], window[2])
        ax.set_ylim(window[1], window[3])
        scale_bar(ax, window, bar_length)
    fig.legend(handles=[Patch(facecolor=COLORS["water"], label="Water included in inland-fisheries weights"),
                        Patch(facecolor=COLORS["excluded"], edgecolor=COLORS["exclusion_edge"], hatch="///", label="Excluded lake footprint")],
               loc="lower center", bbox_to_anchor=(.5, .025), ncol=2, frameon=False, fontsize=11)
    save_figure(fig, args.output_dir / "inland_fisheries_water.png")

    source_records = []
    for path in tqdm(sources, desc="Hash figure sources", unit="file"):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                hasher.update(block)
        source_records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hasher.hexdigest(), "bytes": path.stat().st_size})
    provenance = {
        "assessment_run": run_dir.relative_to(ROOT).as_posix(), "county_count": len(counties),
        "projection": CRS, "sources": source_records,
        "figures": {
            "assessment_geography.png": {"bounds_epsg5070": list(national_bounds), "public_source": products["public_lands_by_county"].relative_to(ROOT).as_posix()},
            "land_footprints_utah.png": {"bounds_lonlat": list(UTAH), "bounds_epsg5070": list(utah_bounds)},
            "inland_fisheries_water.png": {"bounds_lonlat": [list(UTAH), list(MICHIGAN)], "excluded_source_fids": sorted(lake_names), "minimum_water_feature_area_filter": None},
        },
        "rendering": "Actual source polygon footprints rasterized only for PNG display. County lines simplified 250 m nationally and 60 m regionally for legibility. Original vectors and all assessment calculations are unchanged. Water figures use all source feature sizes, not the culled overview export.",
        "versions": {"gdal": gdal.VersionInfo("RELEASE_NAME"), "geopandas": gpd.__version__, "shapely": shapely.__version__, "matplotlib": matplotlib.__version__},
    }
    (args.output_dir / "map_sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote three maps and provenance to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
