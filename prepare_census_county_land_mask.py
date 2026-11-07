"""Build a county land mask from 2024 Census TIGER/Line topological faces."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.client import IncompleteRead
import json
from math import fsum, isfinite
import os
from pathlib import Path
import shutil
import tempfile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.errors import GEOSException
from tqdm import tqdm

from assessment_geography import CONUS_STATE_FIPS
from assessment_zone_geometry import MAX_LAND_MASK_ERROR_PCT
from geometry_utils import repair_polygonal_geometry


ROOT = Path(__file__).resolve().parent
COUNTIES = ROOT / "data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg"
OUTPUT = ROOT / "data/analysis_inputs/boundaries/county_land/tl_2024_us_county_land_conus.gpkg"
FACES_URL = "https://www2.census.gov/geo/tiger/TIGER2024/FACES/"
AREA_CRS = "EPSG:6933"
LAYER = "county_land_mask"
LAND_FLAGS = frozenset({"G", "I", "L", "S"})
WATER_FLAG = "P"
FACE_FIELDS = ["STATEFP", "COUNTYFP", "TFID", "LWFLAG", "ATOTAL"]
BATCH_COUNTIES = 64


def face_shapefile(archive: Path, geoid: str) -> str:
    """Validate the archive members needed to read one official county file.

    Args:
        archive: Path to the downloaded or locally cached county ZIP.
        geoid: Five-character current Census county identifier.

    Returns:
        GDAL virtual path to the county's Shapefile, without extracting it.

    Raises:
        ValueError: A required member is absent, duplicated, or empty.
    """
    stem = f"tl_2024_{geoid}_faces"
    with ZipFile(archive) as source:
        members = source.infolist()
        for extension in ("shp", "shx", "dbf", "prj"):
            name = f"{stem}.{extension}"
            matches = [member for member in members if member.filename == name]
            if len(matches) != 1 or matches[0].file_size == 0:
                raise ValueError(f"{archive} must contain one nonempty {name}")
    return f"/vsizip/{archive.resolve().as_posix()}/{stem}.shp"


def publish_new_file(staged: Path, destination: Path) -> None:
    """Publish a staged file without overwriting an existing file.

    Args:
        staged: Complete file in a staging directory beside the destination.
        destination: New path for the completed file.

    Raises:
        FileExistsError: The destination is already in use.
    """
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise
    except OSError:
        # Some network and removable filesystems do not support hard links.
        created = False
        try:
            with staged.open("rb") as source, destination.open("xb") as target:
                created = True
                shutil.copyfileobj(source, target)
        except (Exception, KeyboardInterrupt):
            if created:
                destination.unlink(missing_ok=True)
            raise


def download_archive(archive: Path, geoid: str) -> None:
    """Fetch one missing official archive into the resumable local cache.

    Args:
        archive: Destination ZIP path under the explicitly chosen cache directory.
        geoid: Expected current county GEOID.

    Raises:
        RuntimeError: Census could not supply a downloadable archive after retries.
    """
    if archive.is_file():
        return
    with tempfile.TemporaryDirectory(dir=archive.parent, prefix=f".{geoid}-") as staging:
        staged = Path(staging) / archive.name
        url = FACES_URL + archive.name
        for attempt in range(3):
            try:
                with urlopen(url, timeout=30) as response, staged.open("wb") as target:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
                # The archive server can return an HTML rejection page with
                # HTTP 200. Validate the container before retaining it.
                face_shapefile(staged, geoid)
                break
            except (URLError, TimeoutError, IncompleteRead, BadZipFile, ValueError) as error:
                retryable = not isinstance(error, HTTPError) or error.code in {429, 500, 502, 503, 504}
                if not retryable or attempt == 2:
                    raise RuntimeError(f"Could not download {url}: {error}") from error
                sleep(2 ** attempt)
        try:
            publish_new_file(staged, archive)
        except FileExistsError:
            # Another explicitly started cache fill finished this county first.
            face_shapefile(archive, geoid)


def integer_values(values: pd.Series, label: str, *, minimum: int = 0) -> np.ndarray:
    """Convert Census count/area fields without silently rounding invalid values.

    Args:
        values: Census attribute values, in their published units.
        label: Source and field name used in an error message.
        minimum: Smallest allowed integer, inclusive.

    Returns:
        Validated 64-bit integers.

    Raises:
        ValueError: The field contains a null, noninteger, or out-of-range value.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    if not np.all(np.isfinite(numeric) & (numeric >= minimum) & (numeric < 2**53) & (numeric == np.floor(numeric))):
        raise ValueError(f"{label} must contain integers >= {minimum} and below 2**53")
    return numeric.astype("int64")


def build_county_mask(archive: Path, county) -> dict:
    """Check one archive against the county record and dissolve its land faces.

    Args:
        archive: Official ZIP containing current 2024 county faces.
        county: Selected reference row in EPSG:6933 with GEOID, ALAND and AWATER.

    Returns:
        One feature record with the mask, provenance, and area QA values.

    Raises:
        ValueError: IDs, face classifications, measures, or geometry do not reconcile.
    """
    geoid = county.GEOID
    stat = archive.stat()
    with archive.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    shp = face_shapefile(archive, geoid)
    faces = pyogrio.read_dataframe(shp, columns=FACE_FIELDS)
    if faces.empty or faces.crs is None or not set(FACE_FIELDS) <= set(faces.columns):
        raise ValueError(f"{geoid}: expected 2024 Census faces with required fields and a CRS")
    states = faces["STATEFP"].astype("string")
    counties = faces["COUNTYFP"].astype("string")
    if not states.eq(geoid[:2]).fillna(False).all() or not counties.eq(geoid[2:]).fillna(False).all():
        raise ValueError(f"{geoid}: archive contains an incorrect or missing STATEFP/COUNTYFP")
    tfids = integer_values(faces["TFID"], f"{geoid} TFID", minimum=1)
    if len(set(tfids)) != len(tfids):
        raise ValueError(f"{geoid}: duplicate TFID in county archive")
    flags = faces["LWFLAG"].astype("string")
    allowed = LAND_FLAGS | {WATER_FLAG}
    if not flags.isin(allowed).all():
        raise ValueError(f"{geoid}: unrecognized or missing LWFLAG: {list(flags[~flags.isin(allowed)].unique())}")
    land = flags.isin(LAND_FLAGS).to_numpy(dtype=bool)
    atotal = integer_values(faces["ATOTAL"], f"{geoid} ATOTAL")
    land_atotal = sum(map(int, atotal[land]))
    water_atotal = sum(map(int, atotal[~land]))
    if (land_atotal, water_atotal) != (county.ALAND, county.AWATER):
        raise ValueError(
            f"{geoid}: ATOTAL land/water ({land_atotal:,}/{water_atotal:,} m²) differs from "
            f"ALAND/AWATER ({county.ALAND:,}/{county.AWATER:,} m²)"
        )
    geometry = faces.loc[land].geometry.to_numpy(copy=True)
    if not np.isin(shapely.get_type_id(geometry), [3, 6]).all() or shapely.is_empty(geometry).any():
        raise ValueError(f"{geoid}: land faces must be nonempty polygons")
    invalid = ~shapely.is_valid(geometry)
    if invalid.any():
        geometry[invalid] = [repair_polygonal_geometry(part) for part in geometry[invalid]]
    areas = shapely.area(geometry)
    if not np.all(np.isfinite(areas) & (areas > 0)):
        raise ValueError(f"{geoid}: land faces contain invalid or zero-area polygons")
    # TIGER faces form a topology in their source CRS. Dissolve shared edges
    # there so a projection cannot independently move the two sides of an
    # edge. Prefer the linear coverage operation for clean county partitions.
    native_sum = fsum(areas)
    native_tol = native_sum * 1e-10
    try:
        dissolved = shapely.coverage_union_all(geometry)
    except GEOSException:
        dissolved = None
    if dissolved is None or not dissolved.is_valid or abs(dissolved.area - native_sum) > native_tol:
        dissolved = shapely.union_all(geometry)
    dissolved = repair_polygonal_geometry(dissolved)
    if dissolved is None:
        raise ValueError(f"{geoid}: land faces did not dissolve to a valid mask")
    projected = gpd.GeoSeries([dissolved], crs=faces.crs).to_crs(AREA_CRS).iloc[0]
    mask = repair_polygonal_geometry(projected)
    if mask is None or not isfinite(mask.area):
        raise ValueError(f"{geoid}: land faces did not dissolve to a valid mask")
    error_m2 = mask.area - county.ALAND
    error_pct = 100.0 * error_m2 / county.ALAND
    tolerance_m2 = county.ALAND * MAX_LAND_MASK_ERROR_PCT / 100.0
    overlap_m2 = 0.0
    if native_sum - dissolved.area > native_tol:
        # Only a suspect coverage needs the more expensive projection of all
        # its component faces to quantify overlap in physical square metres.
        projected_faces = gpd.GeoSeries(geometry, crs=faces.crs).to_crs(AREA_CRS)
        overlap_m2 = max(0.0, fsum(projected_faces.area) - mask.area)
    outside_m2 = 0.0 if county.geometry.covers(mask) else float(mask.difference(county.geometry).area)
    outside_tol = max(0.01, mask.area * 1e-10)
    if abs(error_pct) > MAX_LAND_MASK_ERROR_PCT or overlap_m2 > tolerance_m2 or outside_m2 > outside_tol:
        raise ValueError(
            f"{geoid}: geometry fails {MAX_LAND_MASK_ERROR_PCT:g}% ALAND tolerance: "
            f"area error {error_pct:+.6f}%, overlap {overlap_m2:.3f} m², outside county {outside_m2:.3f} m²"
        )
    after = archive.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{geoid}: source archive changed while preparing the mask")
    return {
        "GEOID": geoid, "ALAND": int(county.ALAND), "AWATER": int(county.AWATER),
        "land_atotal_m2": land_atotal, "water_atotal_m2": water_atotal,
        "geometry_area_m2": mask.area, "area_error_m2": error_m2, "area_error_pct": error_pct,
        "outside_county_m2": outside_m2, "face_overlap_m2": overlap_m2,
        "face_count": len(faces), "land_face_count": int(land.sum()), "water_face_count": int((~land).sum()),
        "source_archive": archive.name, "source_archive_bytes": stat.st_size, "source_archive_sha256": checksum,
        "geometry": mask,
    }


def prepare_mask(args: argparse.Namespace) -> Path:
    """Build and publish a GeoPackage and QA sidecars after every county passes.

    Args:
        args: Input locations, optional download permission, and worker count.

    Returns:
        The completed mask GeoPackage path.

    Raises:
        FileExistsError: Any requested output already exists.
        FileNotFoundError: An archive is missing and downloads were not requested.
        ValueError: The inputs or calculated areas fail validation.
    """
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output = args.output.parent.resolve() / args.output.name
    if output.suffix.lower() != ".gpkg":
        raise ValueError("--output must name a .gpkg file")
    qa_path = output.with_name(f"{output.stem}_qa.csv")
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    for target in (output, qa_path, manifest_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Output already exists: {target}")
    counties_path = args.counties.resolve()
    county_stat = counties_path.stat()
    county_fields = ["GEOID", "ALAND", "AWATER"]
    layer = args.counties_layer
    if layer is None:
        spatial_layers = [name for name, geom_type in pyogrio.list_layers(counties_path) if geom_type]
        if len(spatial_layers) != 1:
            raise ValueError(f"Select --counties-layer in {counties_path}: {spatial_layers}")
        layer = spatial_layers[0]
    print(f"Read county boundaries: {counties_path}", flush=True)
    counties = pyogrio.read_dataframe(counties_path, layer=layer, columns=county_fields)
    if counties.crs is None or not set(county_fields) <= set(counties.columns):
        raise ValueError("County input must define a CRS and GEOID, ALAND, and AWATER")
    ids = counties["GEOID"].astype("string")
    if not ids.str.fullmatch(r"[0-9]{5}").fillna(False).all() or ids.duplicated().any():
        raise ValueError("County input must contain unique five-character GEOIDs")
    counties = counties.assign(GEOID=ids).loc[ids.str[:2].isin(CONUS_STATE_FIPS)].sort_values("GEOID").copy()
    if counties.empty:
        raise ValueError("County input contains no CONUS counties")
    counties["ALAND"] = integer_values(counties["ALAND"], "County ALAND", minimum=1)
    counties["AWATER"] = integer_values(counties["AWATER"], "County AWATER")

    faces_dir = args.faces_dir.resolve()
    archives = {geoid: faces_dir / f"tl_2024_{geoid}_faces.zip" for geoid in counties["GEOID"]}
    missing = [(archive, geoid) for geoid, archive in archives.items() if not archive.is_file()]
    if missing and not args.download_missing:
        examples = ", ".join(path.name for path, _ in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} 2024 faces ZIPs in {faces_dir}: {examples}. "
            "Supply the official archives or use --download-missing."
        )
    for archive in archives.values():
        if archive.exists() and (not archive.is_file() or archive.stat().st_size == 0):
            raise ValueError(f"Expected a nonempty archive: {archive}")

    counties = counties.to_crs(AREA_CRS)
    geom_name = counties.geometry.name
    if not counties.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all() or counties.geometry.is_empty.any():
        raise ValueError("County input must contain nonempty polygons")
    invalid = ~counties.geometry.is_valid
    if invalid.any():
        repaired = counties.geometry.loc[invalid].map(repair_polygonal_geometry)
        if repaired.isna().any():
            raise ValueError("County input contains unrecoverable polygons")
        counties.loc[invalid, geom_name] = repaired
    counties = counties.rename_geometry("geometry") if geom_name != "geometry" else counties

    if missing:
        faces_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(4, args.workers)) as pool:
            try:
                missing_paths, missing_geoids = zip(*missing)
                downloads = pool.map(download_archive, missing_paths, missing_geoids)
                for _ in tqdm(downloads, total=len(missing), desc="Download missing Census archives", unit="county"):
                    pass
            except (Exception, KeyboardInterrupt):
                pool.shutdown(cancel_futures=True)
                raise

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".census-land-mask-") as staging:
        staging = Path(staging)
        records = []
        rows = list(counties[[*county_fields, "geometry"]].itertuples(index=False))

        def build(row) -> dict:
            """Run the county check and add archive context to any failure."""
            archive = archives[row.GEOID]
            try:
                return build_county_mask(archive, row)
            except Exception as error:
                raise ValueError(f"Unable to build county {row.GEOID} from {archive}: {error}") from error

        with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
            total=len(rows), desc="Validate and dissolve Census land faces", unit="county",
        ) as progress:
            # Process in batches so outstanding geometry results stay bounded.
            for offset in range(0, len(rows), BATCH_COUNTIES):
                batch = []
                for feature in pool.map(build, rows[offset:offset + BATCH_COUNTIES]):
                    batch.append(feature)
                    progress.update()
                frame = gpd.GeoDataFrame(batch, geometry="geometry", crs=AREA_CRS)
                pyogrio.write_dataframe(
                    frame, staging / output.name, layer=LAYER, geometry_type="MultiPolygon", append=offset > 0,
                )
                records.extend({key: value for key, value in row.items() if key != "geometry"} for row in batch)

        qa = pd.DataFrame(records)
        qa.to_csv(staging / qa_path.name, index=False)
        with counties_path.open("rb") as source:
            county_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        county_after = counties_path.stat()
        if (county_stat.st_size, county_stat.st_mtime_ns) != (county_after.st_size, county_after.st_mtime_ns):
            raise ValueError(f"County reference changed while preparing the mask: {counties_path}")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "vintage": 2024, "geographic_scope": "CONUS",
            "source_archive_url": FACES_URL, "source_archive_directory": str(faces_dir),
            "counties": {"path": str(counties_path), "layer": layer, "sha256": county_sha256},
            "output": {"path": str(output), "layer": LAYER, "crs": AREA_CRS, "rows": len(qa)},
            "qa_csv": str(qa_path), "land_flags": sorted(LAND_FLAGS), "water_flag": WATER_FLAG,
            "dissolve_method": "Coverage union in source CRS, ordinary union fallback, then project",
            "max_area_error_pct": MAX_LAND_MASK_ERROR_PCT,
            "observed_max_abs_area_error_pct": float(qa["area_error_pct"].abs().max()),
            "observed_max_outside_county_m2": float(qa["outside_county_m2"].max()),
            "source_archives": {
                row["GEOID"]: {"filename": row["source_archive"], "bytes": row["source_archive_bytes"],
                               "sha256": row["source_archive_sha256"]}
                for row in records
            },
        }
        (staging / manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for target in (qa_path, manifest_path, output):
                publish_new_file(staging / target.name, target)
                published.append(target)
        except (Exception, KeyboardInterrupt):
            for target in published:
                target.unlink(missing_ok=True)
            raise

    print(f"Prepared {len(qa):,} Census county masks: {output}\nQA: {qa_path}\nManifest: {manifest_path}", flush=True)
    return output


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options and build the Census county land mask.

    Args:
        argv: Explicit argument strings, or None to read the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces-dir", type=Path, required=True, help="Directory of tl_2024_<GEOID>_faces.zip files")
    parser.add_argument("--download-missing", action="store_true", help="Download missing archives to --faces-dir")
    parser.add_argument("--counties", type=Path, default=COUNTIES, help="2024 TIGER county polygons with ALAND/AWATER")
    parser.add_argument("--counties-layer", help="County layer name, if the file has multiple spatial layers")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="New mask GeoPackage path")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent county workers (default: 4)")
    prepare_mask(parser.parse_args(argv))


if __name__ == "__main__":
    main()
