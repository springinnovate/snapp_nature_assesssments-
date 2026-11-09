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
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from time import sleep
from urllib.request import urlopen
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from tqdm import tqdm
import yaml

from assessment_geography import CONUS_STATE_FIPS
from assessment_zone_geometry import MAX_LAND_MASK_ERROR_PCT
from build_county_ecosystem_service_valuations_2026 import read_attributes, sha256
from geometry_utils import repair_polygonal_geometry
from nlcd_mask_areas import FIELD_DEFINITIONS, RAW_FIELD
from prepare_assessment_zones import project_polygons, read_polygons
from prepare_census_county_land_mask import publish_new_file
from prepare_tif_zonal_stats_config import (
    ZoneJob, load_and_prepare_rasters, load_run_config, write_runner_config,
)
from update_nlcd_mask_areas import export_mask_areas, validate_mask_definition


ROOT = Path(__file__).resolve().parent
AREA_CRS = "EPSG:6933"
NATIVE_CRS = "EPSG:4269"
LAYER = "county_reporting_areas"
AREA_FIELD = "area_ha_county_excl_ocean_great_lakes"
DEFAULT_OUTPUT = ROOT / "data/analysis_inputs/boundaries/county_reporting_areas/tl_2024_us_county_excl_ocean_great_lakes.gpkg"
DEFAULT_RULES = ROOT / "data/workflow_assets/county_integration/county_area_exclusions.yaml"
DEFAULT_MASK_CONFIG = ROOT / "data/workflow_assets/zonal_stats/snapp_assessment_nlcd_masks.yaml"


def read_county_water(geoid: str, archive: Path, rules: dict, download: bool,
                      expected_water_m2: float | None = None) -> tuple[gpd.GeoDataFrame, dict]:
    """Read one Census water archive, downloading only with explicit permission.

    Args:
        geoid: County identifier used in the archive name.
        archive: Cached official ZIP file.
        rules: Reviewed water classification and source URL settings.
        download: Whether missing archives may be downloaded.
        expected_water_m2: Optional county AWATER; zero requires no water file.

    Returns:
        Candidate ocean/lake/bay polygons and the archive's provenance record.

    Raises:
        FileNotFoundError: The archive is missing and downloads are disabled.
        ValueError: Archive attributes, identifiers or polygons are invalid.
    """
    fields = ["HYDROID", "FULLNAME", "MTFCC", "ALAND", "AWATER"]
    if expected_water_m2 == 0:
        frame = gpd.GeoDataFrame(columns=[*fields, "great_lake", "county_geoid", "geometry"],
                                 geometry="geometry", crs=NATIVE_CRS)
        return frame, {"GEOID": geoid, "path": None, "reason": "County AWATER is zero",
                       "total_features": 0, "candidate_features": 0}
    if not archive.is_file():
        if not download:
            raise FileNotFoundError(f"Missing {archive}; supply it or use --download-missing")
        archive.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                with tempfile.TemporaryDirectory(dir=archive.parent, prefix=".areawater-download-") as temporary:
                    staged = Path(temporary) / archive.name
                    with urlopen(rules["source_url"].rstrip("/") + "/" + archive.name, timeout=60) as source, staged.open("xb") as target:
                        shutil.copyfileobj(source, target)
                    with ZipFile(staged) as zipped:
                        members = zipped.namelist()
                        expected = [f"{archive.stem}.{extension}" for extension in ("shp", "shx", "dbf", "prj")]
                        if any(members.count(name) != 1 for name in expected) or zipped.testzip() is not None:
                            raise ValueError(f"Incomplete or corrupt Census archive: {archive.name}")
                    info = pyogrio.read_info(f"/vsizip/{staged.as_posix()}/{archive.stem}.shp")
                    if not set(fields) <= set(info["fields"]) or not info["crs"]:
                        raise ValueError(f"Invalid Census water Shapefile: {archive.name}")
                    publish_new_file(staged, archive)
                break
            except Exception:
                if attempt == 2:
                    raise
                tqdm.write(f"Retry {attempt + 1}/2 downloading {archive.name}")
                sleep(attempt + 1)
    before = archive.stat()
    path = f"/vsizip/{archive.resolve().as_posix()}/{archive.stem}.shp"
    try:
        water = pyogrio.read_dataframe(path, columns=fields)
    except Exception as error:
        raise ValueError(f"Cannot read cached Census archive {archive}; replace it with a valid official ZIP before retrying") from error
    if water.crs is None or not set(fields) <= set(water):
        raise ValueError(f"{archive}: expected Census area-water attributes and a CRS")
    if water.HYDROID.isna().any() or water.HYDROID.duplicated().any():
        raise ValueError(f"{archive}: missing or duplicate HYDROID")
    names = water.FULLNAME.fillna("").str.casefold().str.split().str.join(" ")
    lake_names = {name.casefold(): lake for lake, aliases in rules["great_lakes_names"].items() for name in aliases}
    water["great_lake"] = names.map(lake_names)
    keep = water.MTFCC.isin([rules["ocean_mtfcc"], rules["connected_bay_mtfcc"]]) | water.great_lake.notna()
    counts = {"total_features": len(water), "candidate_features": int(keep.sum())}
    water = water.loc[keep].copy()
    if not water.empty:
        if not water.geom_type.isin(["Polygon", "MultiPolygon"]).all() or water.geometry.is_empty.any():
            raise ValueError(f"{archive}: expected nonempty water polygons")
        water = water.to_crs(NATIVE_CRS)
        invalid = ~water.geometry.is_valid
        if invalid.any():
            repaired = water.loc[invalid].geometry.map(repair_polygonal_geometry)
            if repaired.isna().any():
                raise ValueError(f"{archive}: water polygons cannot be repaired")
            water.loc[invalid, water.geometry.name] = repaired
    else:
        water = water.to_crs(NATIVE_CRS)
    water["county_geoid"] = geoid
    digest = sha256(archive)
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Archive changed while reading: {archive}")
    return water, {"GEOID": geoid, "path": str(archive.resolve()), "sha256": digest,
                   "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, **counts}


def select_connected_water(water: gpd.GeoDataFrame, rules: dict, require_all_lakes: bool) -> gpd.GeoDataFrame:
    """Select ocean/Great Lakes seeds and only bays connected to those seeds.

    Args:
        water: Candidate polygons from all selected counties in native NAD83.
        rules: Reviewed ocean, bay and Great Lakes classifications.
        require_all_lakes: Require all five named Great Lakes for a full run.

    Returns:
        Selected polygons with seed or connected-bay selection reasons.

    Raises:
        ValueError: A full run lacks a Great Lakes seed or polygon CRS.
    """
    if water.crs != NATIVE_CRS:
        raise ValueError("Water connectivity must be evaluated in native Census NAD83")
    missing = set(rules["great_lakes_names"]) - set(water.great_lake.dropna())
    if require_all_lakes and missing:
        raise ValueError(f"Full reporting-area run is missing Great Lakes seeds: {sorted(missing)}")
    ocean = water.MTFCC.eq(rules["ocean_mtfcc"]).to_numpy()
    lake = water.great_lake.notna().to_numpy()
    selected = ocean | lake
    queue = deque(np.flatnonzero(selected).tolist())
    shapes = water.geometry.to_numpy()
    tree = shapely.STRtree(shapes)
    with tqdm(total=len(water), desc="Connect marine and Great Lakes bays", unit="polygon") as progress:
        while queue:
            current = queue.popleft()
            for neighbor in tree.query(shapes[current], predicate="intersects"):
                if not selected[neighbor]:
                    selected[neighbor] = True
                    queue.append(int(neighbor))
            progress.update()
    result = water.loc[selected].copy()
    result["selection_reason"] = np.where(ocean[selected], "ocean", np.where(lake[selected], "great_lake", "connected_bay"))
    return result


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
        removal = shapely.union_all(shapes[matches]).intersection(whole).difference(mask)
        retained = repair_polygonal_geometry(whole.difference(removal))
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
        args: Source paths, approved download flag, output and parallelism.

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
    if rules.get("schema_version") != 1 or rules.get("census_vintage") != 2024 or len(rules.get("great_lakes_names", {})) != 5:
        raise ValueError("Expected reviewed 2024 Census exclusions and all five Great Lakes")
    sources = {"counties": args.counties.resolve(), "land_mask": args.land_mask.resolve(),
               "rules": args.rules.resolve(), "workflow": Path(__file__).resolve()}
    if not args.geometry_only:
        sources["mask_definition"] = args.mask_config.resolve()
    fingerprints = {name: sha256(path) for name, path in tqdm(sources.items(), desc="Hash reporting inputs", unit="file")}
    stats = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in sources.items()}
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
    archives = [args.areawater_dir / f"tl_2024_{geoid}_areawater.zip" for geoid in counties.GEOID]
    with ThreadPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        pieces = list(tqdm(pool.map(read_county_water, counties.GEOID, archives,
                                   [rules] * len(archives), [args.download_missing] * len(archives), counties.AWATER),
                           total=len(archives), desc="Read Census water archives", unit="county"))
    water = gpd.GeoDataFrame(pd.concat([frame for frame, _ in pieces], ignore_index=True), crs=NATIVE_CRS)
    selected = select_connected_water(water, rules, require_all_lakes=not bool(args.geoids))
    areas, qa = build_reporting_geometries(counties, land, selected.to_crs(AREA_CRS), args.workers)
    work = ROOT / "data/processing_outputs/county_reporting_areas" / args.run_id
    work.mkdir(parents=True, exist_ok=False)
    geometry_path = work / "county_reporting_geometry.gpkg"
    pyogrio.write_dataframe(areas, geometry_path, layer=LAYER, geometry_type="MultiPolygon")
    manifest = {"status": "geometry_only" if args.geometry_only else "complete", "run_id": args.run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "subset_verification" if args.geoids else "CONUS",
                "rules": rules, "source_hashes": fingerprints, "source_paths": {key: str(value) for key, value in sources.items()},
                "source_layers": {"counties": county_layer, "land_mask": land_layer},
                "water_archives": [record for _, record in pieces], "rows": len(areas),
                "selection_method": "Ocean/Great Lakes seeds and transitively intersecting bays in native Census NAD83; union selected water and protect Census land",
                "assessment_geometry_and_fisheries": "unchanged", "output": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".county-reporting-") as temporary:
        stage = Path(temporary) / output.name
        if args.geometry_only:
            shutil.copyfile(geometry_path, stage)
            areas.drop(columns="geometry").to_csv(stage.with_suffix(".csv"), index=False)
        else:
            raster_metadata = validate_mask_definition(args.mask_config)
            profile = load_run_config(args.mask_config)
            raster_dir = args.mask_config.parent / profile["project"]["raster_dir"]
            rasters, _ = load_and_prepare_rasters(args.mask_config, raster_dir, work / "derived", apply_scale=False)
            frozen = ROOT / "data/workflow_assets/zonal_stats/runs" / f"county_reporting_masks_{args.run_id}.yaml"
            job = ZoneJob("counties", "County reporting-area NLCD masks", geometry_path, LAYER,
                          "counties_masks", "counties", "GEOID", profile["job:counties_masks"]["operations"])
            write_runner_config(frozen, work / "toolkit_cache", work / "raw", rasters, [job], args.mask_config)
            subprocess.run([sys.executable, "-B", str(args.runner.resolve()), str(frozen),
                            "--job-workers", "1", "--raster-workers", str(args.raster_workers)], check=True)
            raw = list((work / "raw/counties").glob("counties_masks_*.gpkg"))
            if len(raw) != 1 or not raw[0].with_suffix(".csv").is_file():
                raise ValueError("Expected one complete mask-only raw CSV/GeoPackage pair")
            attrs, _, _ = read_attributes(raw[0])
            csv = pd.read_csv(raw[0].with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip")
            fields = [field for field in attrs if RAW_FIELD.fullmatch(field)]
            pd.testing.assert_frame_equal(attrs.set_index("GEOID")[fields].sort_index(),
                                          csv.set_index("GEOID")[fields].sort_index(), check_dtype=False)
            export_mask_areas(raw[0], stage, frozen, raster_metadata=raster_metadata,
                              derived_config=None, output_layer=LAYER)
            for record in raster_metadata:
                stat = Path(record["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                    raise ValueError(f"NLCD mask changed during preparation: {record['path']}")
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
            if (path.stat().st_size, path.stat().st_mtime_ns) != stats[name]:
                raise ValueError(f"Reporting input changed during preparation: {path}")
        for record in manifest["water_archives"]:
            if record["path"] is None:
                continue
            stat = Path(record["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise ValueError(f"Water archive changed during preparation: {record['path']}")
        selection_columns = ["county_geoid", "HYDROID", "FULLNAME", "MTFCC", "great_lake", "selection_reason"]
        with closing(sqlite3.connect(stage)) as connection:
            for name, table in (("county_reporting_area_qa", qa),
                                ("county_reporting_water_selection", selected[selection_columns]),
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
    parser.add_argument("--areawater-dir", type=Path, default=ROOT / "data/analysis_inputs/boundaries/census_areawater/2024")
    parser.add_argument("--download-missing", action="store_true")
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
