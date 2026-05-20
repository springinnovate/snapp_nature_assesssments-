"""Prepare all-land and public-land PAD-US layers clipped to the USA boundary.

This script creates two cleaned PAD-US products from the same source feature
scan:

- all lands: every PAD-US feature with positive-area overlap after clipping
- public lands: the subset of clipped PAD-US features that passes the
  `Mang_Type` / `Own_Type` public-land rule below

The public-land rule is intentionally kept in this file as plain constants and
branching logic. If the project definition of public land changes, start with
`KEEP_MANG_TYPES`, `KEEP_OWN_TYPES_WHEN_UNKNOWN_MANAGER`, and
`_feature_is_public`.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import csv
from os import cpu_count
from pathlib import Path

from osgeo import gdal, ogr, osr
import shapely
from shapely import wkb
from tqdm import tqdm

from geometry_utils import repair_polygonal_geometry

gdal.UseExceptions()
ogr.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")
gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS", "SKIP")

PADUS_GDB_PATH = Path(
    "data/analysis_inputs/padus/"
    "PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/"
    "PADUS4_1Geodatabase.gdb"
)
PADUS_LAYER_NAME = "PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement"
USA_BOUNDARY_PATH = Path("data/analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg")

OUT_DIR = Path("data/processing_outputs/padus_clipped_to_usa")
ALL_OUT_DIR = OUT_DIR / "all_lands"
PUBLIC_OUT_DIR = OUT_DIR / "public_lands"
PUBLIC_OUT_STEM = "padus_public_lands_clipped_to_usa"
ALL_OUT_STEM = "padus_all_lands_clipped_to_usa"
FAILURE_STEM = "padus_lands_clipped_to_usa"

SIMPLIFY_TOLERANCE_METERS = 15.0
VERTICES_PER_JOB = 10000
N_WORKERS = cpu_count()

# PAD-US stores coded values in the geodatabase even when GIS software displays
# longer descriptions. Edit these stored codes when the public-land rule changes.
#
# Manager rule:
# - Keep Federal, State, Local Government, Regional Agency Special District,
#   Joint, and Territorial managed lands in the public-land output.
KEEP_MANG_TYPES = {"FED", "STAT", "LOC", "DIST", "JNT", "TERR"}

# Owner fallback rule:
# - If Manager Type is Unknown (`UNK`), keep features only when Owner Type is
#   Local Government, Regional Agency Special District, Federal, Joint, or State.
KEEP_OWN_TYPES_WHEN_UNKNOWN_MANAGER = {"LOC", "DIST", "FED", "JNT", "STAT"}

WORKER = {}


def _set_axis_order(srs: osr.SpatialReference) -> osr.SpatialReference:
    """Use GIS-style x/y axis order for coordinate transformations.

    Args:
        srs: Spatial reference to configure.

    Returns:
        The same spatial reference object.
    """
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _feature_is_public(feature: ogr.Feature) -> bool:
    """Return whether a PAD-US feature passes the public-land rules.

    Args:
        feature: PAD-US source feature.

    Returns:
        True if the feature should be included in the public-land output.
    """
    mang_type = feature.GetField("Mang_Type")

    # First preference: use manager type. These codes are the clearest signal
    # that a feature belongs in the public-land output.
    if mang_type in KEEP_MANG_TYPES:
        return True

    # Fallback: when the manager is unknown, use owner type as a secondary
    # public-land signal. Unknown manager plus any other owner code is excluded.
    return (
        mang_type == "UNK"
        and feature.GetField("Own_Type") in KEEP_OWN_TYPES_WHEN_UNKNOWN_MANAGER
    )


def _ogr_vertex_count(geom: ogr.Geometry) -> int:
    """Count vertices in an OGR geometry.

    Args:
        geom: Geometry whose vertices should be counted.

    Returns:
        Recursive point count for the geometry.
    """
    if geom is None:
        return 0

    count = geom.GetPointCount()
    for part_index in range(geom.GetGeometryCount()):
        count += _ogr_vertex_count(geom.GetGeometryRef(part_index))
    return count


def _read_usa_boundary(process_srs: osr.SpatialReference):
    """Read, reproject, and merge the USA boundary.

    Args:
        process_srs: Spatial reference used for processing.

    Returns:
        Valid polygonal USA boundary geometry in the processing CRS.
    """
    boundary_vector = gdal.OpenEx(str(USA_BOUNDARY_PATH), gdal.OF_VECTOR)
    boundary_layer = boundary_vector.GetLayer()
    source_srs = _set_axis_order(boundary_layer.GetSpatialRef())
    process_srs = _set_axis_order(process_srs.Clone())
    transform = None
    if not source_srs.IsSame(process_srs):
        transform = osr.CoordinateTransformation(source_srs, process_srs)

    geoms = []
    for feature in boundary_layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        if transform is not None:
            geom = geom.Clone()
            geom.Transform(transform)
        geoms.append(wkb.loads(bytes(geom.ExportToWkb())))

    boundary_vector = None
    boundary_layer = None
    boundary = shapely.union_all(geoms)
    boundary = repair_polygonal_geometry(boundary)
    if boundary is None:
        raise RuntimeError("USA boundary did not contain valid polygon geometry.")
    return boundary


def _build_jobs(layer: ogr.Layer, boundary) -> tuple[list[list[int]], Counter]:
    """Scan PAD-US features and group candidate FIDs into vertex-sized jobs.

    Args:
        layer: PAD-US source layer.
        boundary: USA boundary geometry in the source layer CRS.

    Returns:
        Job FID lists and scan counters.
    """
    jobs = []
    current_job = []
    current_vertices = 0
    stats = Counter()
    boundary_minx, boundary_miny, boundary_maxx, boundary_maxy = boundary.bounds

    for feature_index, feature in enumerate(
        tqdm(
            layer,
            total=layer.GetFeatureCount(),
            desc="Scan PAD-US features",
            unit="feature",
        )
    ):
        stats["scanned"] += 1

        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            stats["empty_geometry_skipped"] += 1
            continue

        minx, maxx, miny, maxy = geom.GetEnvelope()
        if (
            maxx < boundary_minx
            or minx > boundary_maxx
            or maxy < boundary_miny
            or miny > boundary_maxy
        ):
            stats["envelope_skipped"] += 1
            continue

        feature_vertices = max(1, _ogr_vertex_count(geom))
        current_job.append(feature.GetFID())
        current_vertices += feature_vertices
        stats["candidate_features"] += 1
        stats["candidate_vertices"] += feature_vertices
        if _feature_is_public(feature):
            stats["public_candidate_features"] += 1

        if current_vertices >= VERTICES_PER_JOB:
            jobs.append(current_job)
            current_job = []
            current_vertices = 0

    if current_job:
        jobs.append(current_job)

    stats["jobs"] = len(jobs)
    return jobs, stats


def _init_worker(
    gdb_path: str,
    layer_name: str,
    source_srs_wkt: str,
    process_srs_wkt: str,
    boundary_wkb: bytes,
) -> None:
    """Initialize per-process PAD-US reader state.

    Args:
        gdb_path: path to a geodatabase.
        layer_name: layer to process in the gdb
        source_srs_wkt: Source layer CRS WKT.
        process_srs_wkt: Processing CRS WKT.
        boundary_wkb: USA boundary WKB in the processing CRS.
    """
    source_srs = _set_axis_order(osr.SpatialReference(wkt=source_srs_wkt))
    process_srs = _set_axis_order(osr.SpatialReference(wkt=process_srs_wkt))
    transform = None
    if not source_srs.IsSame(process_srs):
        transform = osr.CoordinateTransformation(source_srs, process_srs)

    padus_vector = gdal.OpenEx(gdb_path)
    padus_layer = padus_vector.GetLayer(layer_name)
    WORKER["ds"] = padus_vector
    WORKER["layer"] = padus_layer
    WORKER["transform"] = transform
    WORKER["boundary"] = wkb.loads(boundary_wkb)


def _process_job(
    fids: list[int],
) -> tuple[list[bytes], list[bytes], Counter, list[dict[str, str]]]:
    """Process a chunk of PAD-US FIDs.

    Args:
        fids: Source feature IDs to process.

    Returns:
        All-land output WKB values, public-land output WKB values, processing
        counters, and repair-failure rows.
    """
    layer = WORKER["layer"]
    transform = WORKER["transform"]
    boundary = WORKER["boundary"]
    all_out_wkbs = []
    public_out_wkbs = []
    failures = []
    stats = Counter()

    for fid in fids:
        feature = layer.GetFeature(fid)
        if feature is None:
            stats["missing_feature"] += 1
            failures.append({"source_fid": fid, "reason": "missing feature"})
            continue

        try:
            geom = feature.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                stats["empty_geometry"] += 1
                continue

            if transform is not None:
                geom = geom.Clone()
                geom.Transform(transform)
            shapely_geom = wkb.loads(bytes(geom.ExportToWkb()))
            if not shapely_geom.is_valid:
                shapely_geom = repair_polygonal_geometry(shapely_geom)
                if shapely_geom is None:
                    stats["invalid_source_geometry"] += 1
                    failures.append(
                        {"source_fid": fid, "reason": "invalid source geometry"}
                    )
                    continue

            simplified = shapely.simplify(
                shapely_geom,
                SIMPLIFY_TOLERANCE_METERS,
                preserve_topology=True,
            )
            clipped = shapely.intersection(simplified, boundary)
            clipped = repair_polygonal_geometry(clipped)

            if clipped is None or clipped.is_empty or clipped.area <= 0:
                stats["boundary_skipped"] += 1
                continue

            clipped_wkb = wkb.dumps(clipped)
            all_out_wkbs.append(clipped_wkb)
            stats["all_kept"] += 1
            if _feature_is_public(feature):
                public_out_wkbs.append(clipped_wkb)
                stats["public_kept"] += 1
        except Exception as error:
            stats["exceptions"] += 1
            failures.append({"source_fid": fid, "reason": str(error)})

    return all_out_wkbs, public_out_wkbs, stats, failures


def _create_output_layer(
    out_path: Path,
    layer_name: str,
    process_srs: osr.SpatialReference,
) -> tuple[ogr.DataSource, ogr.Layer]:
    """Create the output GeoPackage layer.

    Args:
        out_path: Output GeoPackage path.
        layer_name: Output layer name.
        process_srs: Output spatial reference.

    Returns:
        Output GDAL datasource and layer.
    """
    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output GeoPackage: {out_path}")

    out_layer = out_ds.CreateLayer(
        layer_name,
        process_srs,
        ogr.wkbMultiPolygon,
        options=["SPATIAL_INDEX=YES"],
    )
    if out_layer is None:
        raise RuntimeError(f"Could not create output layer: {layer_name}")
    out_layer.CreateField(ogr.FieldDefn("land_type", ogr.OFTString))

    return out_ds, out_layer


def _write_geometry_feature(
    out_layer: ogr.Layer,
    out_defn: ogr.FeatureDefn,
    geom_wkb: bytes,
    land_type: str,
) -> None:
    """Write one output geometry with its land type.

    Args:
        out_layer: Destination OGR layer.
        out_defn: Destination layer definition.
        geom_wkb: Output geometry WKB.
        land_type: Output land-type label.
    """
    feature = ogr.Feature(out_defn)
    feature.SetField("land_type", land_type)
    feature.SetGeometry(ogr.CreateGeometryFromWkb(geom_wkb))
    out_layer.CreateFeature(feature)
    feature = None


def _write_failures(failure_path: Path, failures: list[dict[str, str]]) -> None:
    """Write repair failures to a CSV log.

    Args:
        failure_path: Destination CSV path.
        failures: Failure rows to write.
    """
    with failure_path.open("w", newline="", encoding="utf-8") as failure_file:
        writer = csv.DictWriter(failure_file, fieldnames=["source_fid", "reason"])
        writer.writeheader()
        writer.writerows(failures)


def _choose_process_srs(
    source_srs: osr.SpatialReference,
) -> osr.SpatialReference:
    """Choose a meter-based CRS for simplification and clipping.

    Args:
        source_srs: PAD-US layer spatial reference.

    Returns:
        Processing spatial reference.
    """
    source_srs = _set_axis_order(source_srs.Clone())
    if source_srs.IsProjected() and abs(source_srs.GetLinearUnits() - 1.0) < 1e-9:
        return source_srs

    process_srs = osr.SpatialReference()
    process_srs.ImportFromEPSG(5070)
    return _set_axis_order(process_srs)


def main() -> None:
    """Run the PAD-US preprocessing workflow."""
    # the GDB has tons of broken polygons, this says ignore it when loading
    # which will make the load faster then we're fixing it in this script
    # anyway
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    public_out_path = PUBLIC_OUT_DIR / f"{PUBLIC_OUT_STEM}_{timestamp}.gpkg"
    all_out_path = ALL_OUT_DIR / f"{ALL_OUT_STEM}_{timestamp}.gpkg"
    failure_path = OUT_DIR / f"{FAILURE_STEM}_{timestamp}_skipped.csv"
    PUBLIC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ALL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step_bar = tqdm(total=5, desc="PAD-US preprocessing", unit="step")

    step_bar.set_description("Open PAD-US")
    padus_vector = gdal.OpenEx(PADUS_GDB_PATH)
    padus_layer = padus_vector.GetLayer(PADUS_LAYER_NAME)
    source_srs = _set_axis_order(padus_layer.GetSpatialRef())
    process_srs = _choose_process_srs(source_srs)
    step_bar.update()

    step_bar.set_description("Read USA boundary")
    boundary = _read_usa_boundary(process_srs)
    step_bar.update()

    step_bar.set_description("Build jobs")
    boundary_for_scan = boundary
    if not source_srs.IsSame(process_srs):
        process_to_source = osr.CoordinateTransformation(process_srs, source_srs)
        boundary_ogr = ogr.CreateGeometryFromWkb(wkb.dumps(boundary))
        boundary_ogr.Transform(process_to_source)
        boundary_for_scan = wkb.loads(bytes(boundary_ogr.ExportToWkb()))

    jobs, scan_stats = _build_jobs(padus_layer, boundary_for_scan)
    padus_layer = None
    padus_vector = None
    step_bar.update()

    print(
        "Scan stats: "
        + " ".join(f"{key}={value:,}" for key, value in scan_stats.items()),
        flush=True,
    )
    print(
        f"Workers={N_WORKERS:,} vertices_per_job={VERTICES_PER_JOB:,} "
        f"jobs={len(jobs):,}",
        flush=True,
    )
    print(f"Public output: {public_out_path}", flush=True)
    print(f"All-lands output: {all_out_path}", flush=True)
    print(f"Failure log: {failure_path}", flush=True)

    step_bar.set_description("Create outputs")
    public_out_ds, public_out_layer = _create_output_layer(
        public_out_path,
        PUBLIC_OUT_STEM,
        process_srs,
    )
    all_out_ds, all_out_layer = _create_output_layer(
        all_out_path,
        ALL_OUT_STEM,
        process_srs,
    )
    public_out_defn = public_out_layer.GetLayerDefn()
    all_out_defn = all_out_layer.GetLayerDefn()
    step_bar.update()

    step_bar.set_description("Process jobs")
    all_stats = Counter()
    failures = []
    public_written = 0
    all_written = 0

    worker_args = (
        PADUS_GDB_PATH,
        PADUS_LAYER_NAME,
        source_srs.ExportToWkt(),
        process_srs.ExportToWkt(),
        wkb.dumps(boundary),
    )

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=worker_args,
    ) as executor:
        futures = [executor.submit(_process_job, job) for job in jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Process PAD-US jobs",
            unit="job",
        ):
            all_out_wkbs, public_out_wkbs, stats, job_failures = future.result()
            all_stats.update(stats)
            failures.extend(job_failures)

            for geom_wkb in all_out_wkbs:
                _write_geometry_feature(all_out_layer, all_out_defn, geom_wkb, "all")
                all_written += 1
            for geom_wkb in public_out_wkbs:
                _write_geometry_feature(
                    public_out_layer,
                    public_out_defn,
                    geom_wkb,
                    "public",
                )
                public_written += 1

    all_out_ds.FlushCache()
    public_out_ds.FlushCache()
    all_out_ds = None
    public_out_ds = None

    if failures:
        _write_failures(failure_path, failures)

    step_bar.update()
    step_bar.close()

    print(
        "Processing stats: "
        + " ".join(f"{key}={value:,}" for key, value in all_stats.items()),
        flush=True,
    )
    print(f"Wrote {all_written:,} all-land feature(s): {all_out_path}", flush=True)
    print(
        f"Wrote {public_written:,} public-land feature(s): {public_out_path}",
        flush=True,
    )
    if failures:
        print(f"Wrote {len(failures):,} skipped feature log row(s): {failure_path}")
    else:
        print("No unrepaired geometry failures were logged.")


if __name__ == "__main__":
    main()
