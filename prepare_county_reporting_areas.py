"""Prepare county reporting areas excluding oceans and the five Great Lakes.

Other inland water remains in the reporting geometry. This workflow measures
five binary NLCD masks only; it does not rerun ecosystem-service zonals or
change assessment footprints and fisheries allocations.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from tqdm import tqdm
import yaml

from assessment_fisheries import read_excluded_lakes, read_marine_shoreline
from assessment_geography import CONUS_STATE_FIPS
from assessment_zone_geometry import MAX_LAND_MASK_ERROR_PCT
from build_county_ecosystem_service_valuations_2026 import read_attributes, sha256
from geometry_utils import repair_polygonal_geometry
from nlcd_mask_areas import FIELD_DEFINITIONS
from prepare_assessment_zones import project_polygons, read_polygons
from prepare_census_county_land_mask import publish_new_file
from update_nlcd_mask_areas import export_mask_areas, run_mask_zonals, verify_mask_rasters


ROOT = Path(__file__).resolve().parent
AREA_CRS = "EPSG:6933"
LAYER = "county_reporting_areas"
AREA_FIELD = "area_ha_county_excl_ocean_great_lakes"
DEFAULT_OUTPUT = ROOT / "data/analysis_inputs/boundaries/county_reporting_areas/tl_2024_us_county_excl_ocean_great_lakes.gpkg"
DEFAULT_RULES = ROOT / "data/workflow_assets/county_integration/county_area_exclusions.yaml"
DEFAULT_MASK_CONFIG = ROOT / "data/workflow_assets/zonal_stats/snapp_assessment_nlcd_masks.yaml"
DEFAULT_SOURCES = ROOT / "data/workflow_assets/county_integration/sources.csv"
WATER_SOURCES = ("coastline_lines", "freshwater_polygons", "great_lakes_features")
GREAT_LAKES = {"Lake Superior", "Lake Michigan", "Lake Huron", "Lake Erie", "Lake Ontario"}


def select_ocean_water(
    counties: gpd.GeoDataFrame, land: gpd.GeoDataFrame, usa, marine_tree: shapely.STRtree, workers: int,
) -> np.ndarray:
    """Select outside-outline county water connected to mapped marine shoreline.

    Args:
        counties: County polygons with GEOID and AWATER in EPSG:6933.
        land: Census land in EPSG:6933, indexed by GEOID.
        usa: National boundary geometry in EPSG:6933.
        marine_tree: Index of marine shoreline lines in EPSG:6933.
        workers: Number of concurrent county overlay workers.

    Returns:
        Offshore polygon parts, including connected parts across county lines.
    """
    shapely.prepare(usa)
    candidates = counties.loc[counties.AWATER.gt(0) & ~shapely.covers(usa, counties.geometry)]
    land_by_geoid = land.geometry.to_dict()

    def outside_water(county) -> np.ndarray:
        """Return the non-land parts of one county outside the U.S. outline.

        Args:
            county: Candidate county record with GEOID and geometry.

        Returns:
            Polygon parts in the area CRS.
        """
        water = county.geometry.difference(usa).difference(land_by_geoid[county.GEOID])
        return shapely.get_parts(repair_polygonal_geometry(water))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(tqdm(pool.map(outside_water, candidates.itertuples(index=False)), total=len(candidates),
                          desc="Find water outside the U.S. boundary", unit="county"))
    shapes = np.concatenate(parts) if parts else np.empty(0, dtype=object)
    seeds = np.unique(marine_tree.query(shapes, predicate="intersects")[0])
    selected = np.zeros(len(shapes), dtype=bool)
    selected[seeds] = True
    queue = deque(seeds.tolist())
    tree = shapely.STRtree(shapes)
    with tqdm(desc="Connect offshore water across counties", unit="polygon") as progress:
        while queue:
            part = shapes[queue.popleft()]
            neighbors = tree.query(part, predicate="intersects")
            neighbors = neighbors[~selected[neighbors]]
            # A shared edge connects waters; touching only at a point does not.
            connected = neighbors[shapely.length(shapely.intersection(part, shapes[neighbors])) > 0]
            selected[connected] = True
            queue.extend(connected.tolist())
            progress.update()
    return shapes[selected]


def build_reporting_geometries(counties: gpd.GeoDataFrame, land: gpd.GeoDataFrame,
                               excluded: gpd.GeoDataFrame, workers: int) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Subtract the union of selected water, preserving Census land polygons.

    Args:
        counties: County polygons with GEOID, ALAND and AWATER in EPSG:6933.
        land: Matching county Census land masks in EPSG:6933, indexed by GEOID.
        excluded: Globally selected water polygons in EPSG:6933.
        workers: Number of concurrent county overlay workers.

    Returns:
        Reporting geometries and per-county area QA records.

    Raises:
        ValueError: Masks or resulting reporting areas fail area bounds.
    """
    shapes = excluded.geometry.to_numpy()
    tree = shapely.STRtree(shapes)

    def measure(county) -> tuple[dict, dict]:
        """Measure a county's retained area after a nonoverlapping water union.

        Args:
            county: County row with GEOID, ALAND, AWATER and geometry.

        Returns:
            Reporting geometry attributes and the corresponding area audit.

        Raises:
            ValueError: Land masks or reporting areas fail area bounds.
        """
        mask = land.loc[county.GEOID].geometry
        whole = county.geometry
        tolerance = max(0.01, county.ALAND * MAX_LAND_MASK_ERROR_PCT / 100)
        if (abs(mask.area - county.ALAND) > tolerance
                or mask.difference(whole).area > max(0.01, mask.area * 1e-10)):
            raise ValueError(f"{county.GEOID}: Census land mask does not match county ALAND/boundary")
        matches = tree.query(whole, predicate="intersects")
        removal = shapely.union_all(shapely.intersection(shapes[matches], whole)).difference(mask)
        retained = whole if removal.area == 0 else repair_polygonal_geometry(whole.difference(removal))
        if retained is None:
            raise ValueError(f"{county.GEOID}: no retained county reporting area")
        if (removal.area > county.AWATER + tolerance or retained.area < county.ALAND - tolerance
                or retained.area > whole.area + tolerance):
            raise ValueError(f"{county.GEOID}: excluded/retained area fails county land/water bounds")
        row = {"GEOID": county.GEOID, "ALAND": int(county.ALAND), "AWATER": int(county.AWATER),
               AREA_FIELD: retained.area / 10000, "geometry": retained}
        qa = {"GEOID": county.GEOID, "area_ha_full_county": whole.area / 10000,
              "area_ha_land": county.ALAND / 10000, "area_ha_excluded_water": removal.area / 10000,
              AREA_FIELD: retained.area / 10000, "candidate_exclusions": len(matches),
              "area_balance_error_m2": abs(whole.area - retained.area - removal.area)}
        return row, qa

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(measure, counties.itertuples(index=False)), total=len(counties),
                            desc="Build county reporting areas", unit="county"))
    return (gpd.GeoDataFrame([row for row, _ in records], geometry="geometry", crs=AREA_CRS),
            pd.DataFrame([qa for _, qa in records]))


def prepare_reporting_areas(args: argparse.Namespace) -> Path:
    """Prepare immutable county area and natural-area denominators.

    Args:
        args: Local source paths, output and parallelism.

    Returns:
        Published reporting GeoPackage, or the explicitly named geometry-only file.

    Raises:
        ValueError: Inputs, classifications, output scope or area QA are invalid.
        FileExistsError: Any final output or this run's work directory exists.
        subprocess.CalledProcessError: The mask-only toolkit run fails.
    """
    output = args.output.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens")
    if output.suffix.lower() != ".gpkg" or args.workers < 1 or args.raster_workers < 1:
        raise ValueError("Use a .gpkg output and positive worker counts")
    if args.geoids and output == DEFAULT_OUTPUT.resolve():
        raise ValueError("Subset verification requires a separate --output, not the full-run default")
    if args.geometry_only:
        output = output.with_name(output.stem + "_geometry.gpkg")
    targets = [output, output.with_suffix(".csv"), output.with_name(output.stem + "_qa.csv"),
               output.with_name(output.stem + "_manifest.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Reporting output already exists; choose a new --output")
    if not args.geometry_only and (args.runner is None or not args.runner.is_file()):
        raise ValueError("Supply --runner, or --geometry-only for a nonproduction geometry check")
    rules = yaml.safe_load(args.rules.read_text(encoding="utf-8"))
    if (rules.get("schema_version") != 2 or rules.get("census_vintage") != 2024
            or not rules.get("usa_boundary")):
        raise ValueError("Expected 2024 Census reporting areas and a local U.S. boundary")
    water_paths = pd.read_csv(args.sources, dtype=str).set_index("source")["path"]
    if not water_paths.index.is_unique or not set(WATER_SOURCES) <= set(water_paths.index):
        raise ValueError(f"Expected unique source paths including {', '.join(WATER_SOURCES)} in {args.sources}")
    boundary = args.usa_boundary or (ROOT / "data" / rules["usa_boundary"])
    sources = {"counties": args.counties.resolve(), "land_mask": args.land_mask.resolve(),
               "rules": args.rules.resolve(), "workflow": Path(__file__).resolve(),
               "source_table": args.sources.resolve(), "usa_boundary": boundary.resolve(),
               **{key: (ROOT / "data" / water_paths[key]).resolve() for key in WATER_SOURCES}}
    if not args.geometry_only:
        sources["mask_definition"] = args.mask_config.resolve()
    stats = {name: (stat.st_size, stat.st_mtime_ns) for name, path in sources.items() for stat in [path.stat()]}
    fingerprints = {name: sha256(path) for name, path in tqdm(sources.items(), desc="Hash reporting inputs", unit="file")}
    counties, county_layer = read_polygons(args.counties, args.counties_layer, {"GEOID", "ALAND", "AWATER"})
    ids = counties.GEOID.astype("string")
    counties = counties.loc[ids.str[:2].isin(CONUS_STATE_FIPS)].assign(GEOID=ids)
    if counties.empty or counties.GEOID.isna().any() or counties.GEOID.duplicated().any():
        raise ValueError("Expected nonempty CONUS counties with unique GEOIDs")
    if args.geoids:
        if set(args.geoids) - set(counties.GEOID):
            raise ValueError("Requested GEOIDs are absent from the CONUS county source")
        counties = counties.loc[counties.GEOID.isin(args.geoids)].copy()
    if not counties.GEOID.str.fullmatch(r"\d{5}").all():
        raise ValueError("County GEOIDs must be five-digit strings")
    counties = project_polygons(counties, "counties").sort_values("GEOID").reset_index(drop=True)
    if (not np.isfinite(counties[["ALAND", "AWATER"]]).all().all()
            or counties.ALAND.le(0).any() or counties.AWATER.lt(0).any()):
        raise ValueError("County ALAND/AWATER must be finite with positive land and nonnegative water")
    land, land_layer = read_polygons(args.land_mask, args.land_mask_layer, {"GEOID"})
    land = project_polygons(land.loc[land.GEOID.isin(counties.GEOID)], "land mask").set_index("GEOID")
    if not land.index.is_unique or set(land.index) != set(counties.GEOID):
        raise ValueError("Land-mask county keys differ from reporting counties")
    outline, outline_layer = read_polygons(sources["usa_boundary"], args.usa_boundary_layer, set(), county_extent=counties)
    if outline.empty:
        raise ValueError("The U.S. boundary does not overlap the reporting extent")
    usa = project_polygons(outline, "U.S. boundary").geometry.union_all()
    shoreline = read_marine_shoreline(sources["coastline_lines"], counties)
    ocean = select_ocean_water(counties, land, usa, shoreline, args.workers)
    if not args.geoids and not len(ocean):
        raise ValueError("No ocean area selected; verify the local U.S. boundary and marine shoreline")
    tqdm.write("Read the five Great Lakes footprints from local NHD freshwater data")
    if not args.geoids:
        lake_names = set(pd.read_csv(sources["great_lakes_features"], usecols=["lake"]).lake)
        if lake_names != GREAT_LAKES:
            raise ValueError("The Great Lakes crosswalk must identify exactly the five Great Lakes")
    lakes = read_excluded_lakes(sources["freshwater_polygons"], sources["great_lakes_features"], "Great Lakes")
    lake_parts = shapely.get_parts(lakes)
    selected = gpd.GeoDataFrame(geometry=[*ocean, *lake_parts], crs=AREA_CRS)
    areas, qa = build_reporting_geometries(counties, land, selected, args.workers)
    water_hash = hashlib.sha256()
    for geometry in selected.geometry:
        water_hash.update(shapely.to_wkb(geometry, byte_order=1))
    selection = pd.DataFrame([
        {"selection_reason": reason, "polygon_parts": len(parts), "source_area_ha": shapely.area(parts).sum() / 10000}
        for reason, parts in (("shoreline_connected_ocean", ocean), ("great_lakes", lake_parts))
    ])
    work = ROOT / "data/processing_outputs/county_reporting_areas" / args.run_id
    work.mkdir(parents=True, exist_ok=False)
    geometry_path = work / "county_reporting_geometry.gpkg"
    pyogrio.write_dataframe(areas, geometry_path, layer=LAYER, geometry_type="MultiPolygon")
    manifest = {"status": "geometry_only" if args.geometry_only else "complete", "run_id": args.run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "subset_verification" if args.geoids else "CONUS",
                "rules": rules, "source_hashes": fingerprints, "source_paths": {key: str(value) for key, value in sources.items()},
                "source_layers": {"counties": county_layer, "land_mask": land_layer, "usa_boundary": outline_layer},
                "selected_water_sha256": water_hash.hexdigest(),
                "rows": len(areas),
                "selection_method": "Marine-shoreline-connected water outside the local U.S. boundary plus five NHD Great Lakes; protect Census land",
                "assessment_geometry_and_fisheries": "unchanged", "output": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".county-reporting-") as temporary:
        stage = Path(temporary) / output.name
        if args.geometry_only:
            shutil.copyfile(geometry_path, stage)
            areas.drop(columns="geometry").to_csv(stage.with_suffix(".csv"), index=False)
        else:
            frozen = ROOT / "data/workflow_assets/zonal_stats/runs" / f"county_reporting_masks_{args.run_id}.yaml"
            raw, frozen, raster_metadata = run_mask_zonals(
                args.runner, args.mask_config, frozen, work / "raw", work / "toolkit_cache",
                args.raster_workers, reporting_zone=(geometry_path, LAYER))
            export_mask_areas(raw, stage, frozen, raster_metadata=raster_metadata,
                              derived_config=None, output_layer=LAYER)
            verify_mask_rasters(raster_metadata)
            attrs, _, _ = read_attributes(stage)
            if not set(FIELD_DEFINITIONS) <= set(attrs):
                raise ValueError("Reporting artifact is missing corrected NLCD fields")
            if (AREA_FIELD not in attrs or attrs.GEOID.duplicated().any()
                    or set(attrs.GEOID) != set(areas.GEOID)):
                raise ValueError("Reporting artifact lost its county area field or county keys")
            pd.testing.assert_series_equal(attrs.set_index("GEOID")[AREA_FIELD].sort_index(),
                                           areas.set_index("GEOID")[AREA_FIELD].sort_index(),
                                           check_dtype=False, rtol=1e-12, atol=1e-8)
            csv = pd.read_csv(stage.with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip")
            pd.testing.assert_frame_equal(attrs.set_index("GEOID").sort_index(),
                                          csv.set_index("GEOID").sort_index(), check_dtype=False)
            manifest.update(mask_config=str(frozen), mask_config_sha256=sha256(frozen), raster_metadata=raster_metadata)
        for name, path in sources.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != stats[name]:
                raise ValueError(f"Reporting input changed during preparation: {path}")
        with closing(sqlite3.connect(stage)) as connection:
            for name, table in (("county_reporting_area_qa", qa),
                                ("county_reporting_water_selection", selection),
                                ("county_reporting_area_run", pd.DataFrame([{
                                    "run_id": args.run_id, "manifest_json": json.dumps(manifest)
                                }]))):
                table.to_sql(name, connection, index=False)
                connection.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,description) VALUES (?, 'attributes', ?, ?)",
                                   (name, name, "County reporting-area preparation provenance"))
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Reporting GeoPackage integrity check failed")
        qa.to_csv(Path(temporary) / targets[2].name, index=False)
        manifest["output_sha256"] = sha256(stage)
        (Path(temporary) / targets[3].name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for target in targets:
                publish_new_file(Path(temporary) / target.name, target)
                published.append(target)
        except (Exception, KeyboardInterrupt):
            for target in published:
                target.unlink()
            raise
    tqdm.write(f"Prepared county reporting areas: {output}")
    return output


def main(argv: list[str] | None = None) -> None:
    """Parse paths and prepare reporting areas with optional mask-only zonals.

    Args:
        argv: Explicit CLI arguments, or None to use the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counties", type=Path, default=ROOT / "data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg")
    parser.add_argument("--counties-layer")
    parser.add_argument("--land-mask", type=Path, default=ROOT / "data/analysis_inputs/boundaries/county_land/tl_2024_us_county_land_conus.gpkg")
    parser.add_argument("--land-mask-layer", default="county_land_mask")
    parser.add_argument("--usa-boundary", type=Path, help="Override the local U.S. boundary in the reporting rules")
    parser.add_argument("--usa-boundary-layer")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="Source table for shoreline and Great Lakes inputs")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--mask-config", type=Path, default=DEFAULT_MASK_CONFIG)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-only", action="store_true", help="Write a separately named geometry-only verification artifact")
    parser.add_argument("--geoids", nargs="+", help="Subset verification only; requires an explicit nondefault output")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--raster-workers", type=int, default=2)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    prepare_reporting_areas(parser.parse_args(argv))


if __name__ == "__main__":
    main()
