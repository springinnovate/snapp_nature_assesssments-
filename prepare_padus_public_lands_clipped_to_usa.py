"""Prepare a cleaned PAD-US public lands layer clipped to a USA boundary."""

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
    "./data/PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/PADUS4_1Geodatabase.gdb"
)
PADUS_LAYER_NAME = "PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement"
USA_BOUNDARY_PATH = Path(
    r"D:\repositories\snapp_nature_assesssments-\data\usa_vector.gpkg"
)

OUT_DIR = Path("output")
OUT_LAYER_NAME = "padus_public_lands_clipped_to_usa"
OUT_STEM = "padus_public_lands_clipped_to_usa"

SIMPLIFY_TOLERANCE_METERS = 15.0
VERTICES_PER_JOB = 10000
N_WORKERS = cpu_count()
KEEP_MANG_TYPES = {"FED", "STAT", "LOC", "DIST", "JNT", "TERR"}
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


def _feature_should_be_kept(feature: ogr.Feature) -> bool:
    """Return whether a PAD-US feature passes the owner/manager rules.

    Args:
        feature: PAD-US source feature.

    Returns:
        True if the feature should be considered for geometry processing.
    """
    mang_type = feature.GetField("Mang_Type")
    if mang_type in KEEP_MANG_TYPES:
        return True

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

        if not _feature_should_be_kept(feature):
            stats["attribute_skipped"] += 1
            continue

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
) -> tuple[list[bytes], Counter, list[dict[str, str]]]:
    """Process a chunk of PAD-US FIDs.

    Args:
        fids: Source feature IDs to process.

    Returns:
        Output geometry WKB values, processing counters, and repair-failure rows.
    """
    layer = WORKER["layer"]
    transform = WORKER["transform"]
    boundary = WORKER["boundary"]
    out_wkbs = []
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

            out_wkbs.append(wkb.dumps(clipped))
            stats["kept"] += 1
        except Exception as error:
            stats["exceptions"] += 1
            failures.append({"source_fid": fid, "reason": str(error)})

    return out_wkbs, stats, failures


def _create_output_layer(
    out_path: Path,
    process_srs: osr.SpatialReference,
) -> tuple[ogr.DataSource, ogr.Layer]:
    """Create the output GeoPackage layer.

    Args:
        out_path: Output GeoPackage path.
        process_srs: Output spatial reference.

    Returns:
        Output GDAL datasource and layer.
    """
    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(str(out_path))
    if out_ds is None:
        raise RuntimeError(f"Could not create output GeoPackage: {out_path}")

    out_layer = out_ds.CreateLayer(
        OUT_LAYER_NAME,
        process_srs,
        ogr.wkbMultiPolygon,
        options=["SPATIAL_INDEX=YES"],
    )
    if out_layer is None:
        raise RuntimeError(f"Could not create output layer: {OUT_LAYER_NAME}")

    return out_ds, out_layer


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
    out_path = OUT_DIR / f"{OUT_STEM}_{timestamp}.gpkg"
    failure_path = OUT_DIR / f"{OUT_STEM}_{timestamp}_skipped.csv"
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
    print(f"Output: {out_path}", flush=True)
    print(f"Failure log: {failure_path}", flush=True)

    step_bar.set_description("Create output")
    out_ds, out_layer = _create_output_layer(out_path, process_srs)
    out_defn = out_layer.GetLayerDefn()
    step_bar.update()

    step_bar.set_description("Process jobs")
    all_stats = Counter()
    failures = []
    written = 0

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
            out_wkbs, stats, job_failures = future.result()
            all_stats.update(stats)
            failures.extend(job_failures)

            for geom_wkb in out_wkbs:
                feature = ogr.Feature(out_defn)
                feature.SetGeometry(ogr.CreateGeometryFromWkb(geom_wkb))
                out_layer.CreateFeature(feature)
                feature = None
                written += 1

    out_ds.FlushCache()
    out_ds = None

    if failures:
        _write_failures(failure_path, failures)

    step_bar.update()
    step_bar.close()

    print(
        "Processing stats: "
        + " ".join(f"{key}={value:,}" for key, value in all_stats.items()),
        flush=True,
    )
    print(f"Wrote {written:,} output feature(s): {out_path}", flush=True)
    if failures:
        print(f"Wrote {len(failures):,} skipped feature log row(s): {failure_path}")
    else:
        print("No unrepaired geometry failures were logged.")


if __name__ == "__main__":
    main()
