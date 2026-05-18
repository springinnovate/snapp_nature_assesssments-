"""Prepare simplified NHD freshwater polygons clipped to a USA boundary."""

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

NHD_GDB_PATH = Path(
    "data/analysis_inputs/hydrography/nhdplus/"
    "NHDPlus_H_National_Release_2_GDB/NHDPlus_H_National_Release_2_GDB.gdb"
)
USA_BOUNDARY_PATH = Path(
    "data/analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg"
)
OUT_DIR = Path("data/analysis_inputs/hydrography/nhdfreshwater")
OUT_STEM = "nhd_freshwater_clipped_to_usa"
FAILURE_STEM = "nhd_freshwater_clipped_to_usa"

SIMPLIFY_TOLERANCE_METERS = 15.0
VERTICES_PER_JOB = 10000
N_WORKERS = cpu_count() or 1

FRESHWATER_FTYPES_BY_LAYER = {
    "NHDWaterbody": {
        390,  # LakePond
        436,  # Reservoir
        466,  # SwampMarsh
    },
    "NHDArea": {
        460,  # StreamRiver
        537,  # Area of Complex Channels
    },
}
FRESHWATER_FTYPE_LABELS = {
    390: "LakePond",
    436: "Reservoir",
    460: "StreamRiver",
    466: "SwampMarsh",
    537: "AreaOfComplexChannels",
}

WORKER = {}


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


def _choose_process_srs(
    source_srs: osr.SpatialReference,
) -> osr.SpatialReference:
    """Choose a meter-based CRS for simplification and clipping.

    Args:
        source_srs: NHD layer spatial reference.

    Returns:
        Processing spatial reference.
    """
    source_srs = source_srs.Clone()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if source_srs.IsProjected() and abs(source_srs.GetLinearUnits() - 1.0) < 1e-9:
        return source_srs

    process_srs = osr.SpatialReference()
    process_srs.ImportFromEPSG(5070)
    process_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return process_srs


def _read_usa_boundary(process_srs: osr.SpatialReference):
    """Read, reproject, and merge the USA boundary.

    Args:
        process_srs: Spatial reference used for processing.

    Returns:
        Valid polygonal USA boundary geometry in the processing CRS.
    """
    boundary_vector = gdal.OpenEx(str(USA_BOUNDARY_PATH), gdal.OF_VECTOR)
    if boundary_vector is None:
        raise RuntimeError(f"Could not open USA boundary: {USA_BOUNDARY_PATH}")

    boundary_layer = boundary_vector.GetLayer()
    source_srs = boundary_layer.GetSpatialRef()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    process_srs = process_srs.Clone()
    process_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
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
        geoms.append(shapely.force_2d(wkb.loads(bytes(geom.ExportToWkb()))))

    boundary_vector = None
    boundary_layer = None
    boundary = shapely.union_all(geoms)
    boundary = repair_polygonal_geometry(boundary)
    if boundary is None:
        raise RuntimeError("USA boundary did not contain valid polygon geometry.")
    return boundary


def _build_layer_jobs(
    layer: ogr.Layer,
    layer_name: str,
    boundary,
) -> tuple[list[tuple[str, list[int]]], Counter]:
    """Scan one NHD layer and group freshwater candidate FIDs into jobs.

    Args:
        layer: NHD source layer.
        layer_name: Source layer name.
        boundary: USA boundary geometry in the source layer CRS.

    Returns:
        Job tuples and scan counters.
    """
    ftypes = FRESHWATER_FTYPES_BY_LAYER[layer_name]
    layer.SetAttributeFilter(
        "ftype IN ({})".format(",".join(str(ftype) for ftype in sorted(ftypes)))
    )

    jobs = []
    current_job = []
    current_vertices = 0
    stats = Counter()
    boundary_minx, boundary_miny, boundary_maxx, boundary_maxy = boundary.bounds

    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc=f"Scan {layer_name}",
        unit="feature",
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

        if current_vertices >= VERTICES_PER_JOB:
            jobs.append((layer_name, current_job))
            current_job = []
            current_vertices = 0

    if current_job:
        jobs.append((layer_name, current_job))

    layer.SetAttributeFilter(None)
    stats["jobs"] = len(jobs)
    return jobs, stats


def _boundary_in_source_srs(
    boundary,
    process_srs: osr.SpatialReference,
    source_srs: osr.SpatialReference,
):
    """Transform the processing boundary back to the source CRS for scanning.

    Args:
        boundary: USA boundary geometry in the processing CRS.
        process_srs: Processing spatial reference.
        source_srs: Source layer spatial reference.

    Returns:
        USA boundary geometry in the source CRS.
    """
    if source_srs.IsSame(process_srs):
        return boundary

    transform = osr.CoordinateTransformation(process_srs, source_srs)
    boundary_ogr = ogr.CreateGeometryFromWkb(wkb.dumps(boundary))
    boundary_ogr.Transform(transform)
    return shapely.force_2d(wkb.loads(bytes(boundary_ogr.ExportToWkb())))


def _init_worker(
    gdb_path: str,
    source_srs_wkt: str,
    process_srs_wkt: str,
    boundary_wkb: bytes,
) -> None:
    """Initialize per-process NHD reader state.

    Args:
        gdb_path: Path to the NHD geodatabase.
        source_srs_wkt: Source layer CRS WKT.
        process_srs_wkt: Processing CRS WKT.
        boundary_wkb: USA boundary WKB in the processing CRS.
    """
    source_srs = osr.SpatialReference(wkt=source_srs_wkt)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    process_srs = osr.SpatialReference(wkt=process_srs_wkt)
    process_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = None
    if not source_srs.IsSame(process_srs):
        transform = osr.CoordinateTransformation(source_srs, process_srs)

    nhd_vector = gdal.OpenEx(gdb_path)
    if nhd_vector is None:
        raise RuntimeError(f"Could not open NHD geodatabase: {gdb_path}")

    WORKER["ds"] = nhd_vector
    WORKER["layers"] = {
        layer_name: nhd_vector.GetLayerByName(layer_name)
        for layer_name in FRESHWATER_FTYPES_BY_LAYER
    }
    WORKER["transform"] = transform
    WORKER["boundary"] = wkb.loads(boundary_wkb)


def _process_job(
    layer_name: str,
    fids: list[int],
) -> tuple[list[dict], Counter, list[dict[str, str]]]:
    """Process one chunk of NHD feature IDs.

    Args:
        layer_name: Source layer name.
        fids: Source feature IDs to process.

    Returns:
        Output rows, processing counters, and failure rows.
    """
    layer = WORKER["layers"][layer_name]
    transform = WORKER["transform"]
    boundary = WORKER["boundary"]
    out_rows = []
    failures = []
    stats = Counter()

    for fid in fids:
        feature = layer.GetFeature(fid)
        if feature is None:
            stats["missing_feature"] += 1
            failures.append(
                {"source_layer": layer_name, "source_fid": fid, "reason": "missing"}
            )
            continue

        try:
            geom = feature.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                stats["empty_geometry"] += 1
                continue

            if transform is not None:
                geom = geom.Clone()
                geom.Transform(transform)

            shapely_geom = shapely.force_2d(wkb.loads(bytes(geom.ExportToWkb())))
            if not shapely_geom.is_valid:
                shapely_geom = repair_polygonal_geometry(shapely_geom)
                if shapely_geom is None:
                    stats["invalid_source_geometry"] += 1
                    failures.append(
                        {
                            "source_layer": layer_name,
                            "source_fid": fid,
                            "reason": "invalid source geometry",
                        }
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

            out_rows.append(
                {
                    "source_layer": layer_name,
                    "source_fid": fid,
                    "ftype": feature.GetField("ftype"),
                    "fcode": feature.GetField("fcode"),
                    "feature_type": FRESHWATER_FTYPE_LABELS[
                        feature.GetField("ftype")
                    ],
                    "geom_wkb": wkb.dumps(clipped),
                }
            )
            stats["kept"] += 1
        except Exception as error:
            stats["exceptions"] += 1
            failures.append(
                {
                    "source_layer": layer_name,
                    "source_fid": fid,
                    "reason": str(error),
                }
            )

    return out_rows, stats, failures


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
        OUT_STEM,
        process_srs,
        ogr.wkbMultiPolygon,
        options=["SPATIAL_INDEX=YES"],
    )
    if out_layer is None:
        raise RuntimeError(f"Could not create output layer: {OUT_STEM}")

    fields = [
        ("source_layer", ogr.OFTString),
        ("source_fid", ogr.OFTInteger64),
        ("ftype", ogr.OFTInteger),
        ("fcode", ogr.OFTInteger),
        ("feature_type", ogr.OFTString),
    ]
    for field_name, field_type in fields:
        out_layer.CreateField(ogr.FieldDefn(field_name, field_type))

    return out_ds, out_layer


def _write_output_feature(
    out_layer: ogr.Layer,
    out_defn: ogr.FeatureDefn,
    out_row: dict,
) -> None:
    """Write one prepared freshwater feature.

    Args:
        out_layer: Destination OGR layer.
        out_defn: Destination layer definition.
        out_row: Output row with source fields and geometry WKB.
    """
    feature = ogr.Feature(out_defn)
    feature.SetField("source_layer", out_row["source_layer"])
    feature.SetField("source_fid", out_row["source_fid"])
    feature.SetField("ftype", out_row["ftype"])
    feature.SetField("fcode", out_row["fcode"])
    feature.SetField("feature_type", out_row["feature_type"])
    feature.SetGeometry(ogr.CreateGeometryFromWkb(out_row["geom_wkb"]))
    out_layer.CreateFeature(feature)
    feature = None


def _write_failures(failure_path: Path, failures: list[dict[str, str]]) -> None:
    """Write repair and processing failures to a CSV log.

    Args:
        failure_path: Destination CSV path.
        failures: Failure rows to write.
    """
    with failure_path.open("w", newline="", encoding="utf-8") as failure_file:
        writer = csv.DictWriter(
            failure_file,
            fieldnames=["source_layer", "source_fid", "reason"],
        )
        writer.writeheader()
        writer.writerows(failures)


def main() -> None:
    """Run the NHD freshwater preprocessing workflow."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_path = OUT_DIR / f"{OUT_STEM}_{timestamp}.gpkg"
    failure_path = OUT_DIR / f"{FAILURE_STEM}_{timestamp}_skipped.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step_bar = tqdm(total=5, desc="NHD freshwater preprocessing", unit="step")

    step_bar.set_description("Open NHD")
    nhd_vector = gdal.OpenEx(str(NHD_GDB_PATH))
    if nhd_vector is None:
        raise RuntimeError(f"Could not open NHD geodatabase: {NHD_GDB_PATH}")

    source_layer = nhd_vector.GetLayerByName(next(iter(FRESHWATER_FTYPES_BY_LAYER)))
    source_srs = source_layer.GetSpatialRef()
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    process_srs = _choose_process_srs(source_srs)
    step_bar.update()

    step_bar.set_description("Read USA boundary")
    boundary = _read_usa_boundary(process_srs)
    boundary_for_scan = _boundary_in_source_srs(boundary, process_srs, source_srs)
    step_bar.update()

    step_bar.set_description("Build jobs")
    jobs = []
    scan_stats = Counter()
    for layer_name in FRESHWATER_FTYPES_BY_LAYER:
        layer = nhd_vector.GetLayerByName(layer_name)
        if layer is None:
            raise RuntimeError(f"NHD layer does not exist: {layer_name}")
        layer_jobs, layer_stats = _build_layer_jobs(
            layer,
            layer_name,
            boundary_for_scan,
        )
        jobs.extend(layer_jobs)
        scan_stats.update(
            {f"{layer_name}.{key}": value for key, value in layer_stats.items()}
        )

    nhd_vector = None
    source_layer = None
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
        str(NHD_GDB_PATH),
        source_srs.ExportToWkt(),
        process_srs.ExportToWkt(),
        wkb.dumps(boundary),
    )

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=worker_args,
    ) as executor:
        futures = [executor.submit(_process_job, *job) for job in jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Process NHD freshwater jobs",
            unit="job",
        ):
            out_rows, stats, job_failures = future.result()
            all_stats.update(stats)
            failures.extend(job_failures)
            for out_row in out_rows:
                _write_output_feature(out_layer, out_defn, out_row)
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
    print(f"Wrote {written:,} freshwater feature(s): {out_path}", flush=True)
    if failures:
        print(f"Wrote {len(failures):,} skipped feature log row(s): {failure_path}")
    else:
        print("No unrepaired geometry failures were logged.")


if __name__ == "__main__":
    main()
