"""Prepare all-land, public-land, and public-access PAD-US layers.

This script creates three cleaned PAD-US products from the same source feature
scan:

- all lands: every PAD-US feature with positive-area overlap after clipping
- public lands: features passing the configured ownership/management rule
- public access: public-land features passing the configured access rule

Classification policy is stored in `config/padus_land_rules.rules` and parsed
by `padus_land_rules` without executing Python or SQL.
"""

from __future__ import annotations

import argparse
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

from assessment.geometry_utils import repair_polygonal_geometry
from assessment.padus_land_rules import (
    PADUS_RULE_FIELDS,
    VIRTUAL_MANAGER_VALUES,
    RuleConfigError,
    RuleSet,
    build_rule_context,
    load_rule_config,
)

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
PRODUCTS = {
    "all": ("all_lands", "padus_all_lands_clipped_to_usa", None),
    "public": ("public_lands", "padus_public_lands_clipped_to_usa", "public_land"),
    "public-access": ("public_access_lands", "padus_public_access_lands_clipped_to_usa", "public_access"),
}
# PAD-US 4.1 Agency Type and Designation Type coded domains. The public-land
# fields can carry these codes even when a derived vector omits the domains.
PUBLIC_LAND_DOMAINS = {
    "Mang_Type": frozenset("DESG DIST FED JNT LOC NGO PVT STAT TERR TRIB UNK".split()),
    "Des_Tp": frozenset(
        ("ACC ACEC AGRE CONE FACY FORE FOTH HCA HCAE IRA LCA LHCA LOTH LP LREC LRMA "
         "MIL MIT MPA NCA ND NF NG NLS NM NP NRA NSBV NT NWR OCS OTHE PAGR PCON PFOR "
         "PHCA POTH PPRK PRAN PREC PROC PUB RANE REA REC RECE RMA RNA SCA SDA SHCA "
         "SOTH SP SREC SRMA SW TRIBL UNK UNKE WA WPA WSA WSR").split()
    ),
}
FAILURE_STEM = "padus_lands_clipped_to_usa"
SOURCE_OBJECTID_FIELD = "OBJECTID"
DEFAULT_RULE_CONFIG_PATH = Path("config/padus_land_rules.rules")

SIMPLIFY_TOLERANCE_METERS = 15.0
VERTICES_PER_JOB = 10000
N_WORKERS = cpu_count()

WORKER = {}
FieldSpec = tuple[str, int, int, int, int]
OutputRecord = tuple[bytes, dict[str, object]]


def _set_axis_order(srs: osr.SpatialReference) -> osr.SpatialReference:
    """Use GIS-style x/y axis order for coordinate transformations.

    Args:
        srs: Spatial reference to configure.

    Returns:
        The same spatial reference object.
    """
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all-land, public-land, and public-access PAD-US layers."
        )
    )
    parser.add_argument("--padus-path", type=Path, default=PADUS_GDB_PATH, help="PAD-US vector dataset")
    parser.add_argument("--padus-layer", default=PADUS_LAYER_NAME, help="Layer inside the PAD-US dataset")
    parser.add_argument(
        "--products", nargs="+", choices=tuple(PRODUCTS), default=tuple(PRODUCTS),
        help="Land products to write (default: all public public-access)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Root for the land-product folders")
    parser.add_argument("--validate-only", action="store_true", help="Check input and rules without writing outputs")
    parser.add_argument(
        "--rules-config",
        type=Path,
        default=DEFAULT_RULE_CONFIG_PATH,
        help="Human-readable PAD-US land-classification rule file.",
    )
    return parser.parse_args()


def _feature_classifications(
    feature: ogr.Feature,
    rules: RuleSet,
    source_fields: tuple[str, ...],
) -> dict[str, bool]:
    """Evaluate configured classifications for one PAD-US feature.

    Args:
        feature: PAD-US source feature.
        rules: Rules for the requested products.
        source_fields: Physical source fields used by those rules.

    Returns:
        Boolean results keyed by configured rule name.
    """
    attributes = {field_name: feature.GetField(field_name) for field_name in source_fields}
    classifications = rules.evaluate(build_rule_context(attributes))
    if classifications.get("public_access") and not classifications.get("public_land"):
        raise RuleConfigError(
            "The configured public_access rule selected a feature that the "
            "public_land rule did not select."
        )
    return classifications


def _rule_domain_values(
    dataset: gdal.Dataset,
    layer: ogr.Layer,
    fields: frozenset[str],
) -> dict[str, set[str] | frozenset[str]]:
    """Read coded domains or use the PAD-US 4.1 public-land code lists.

    Args:
        dataset: Open PAD-US source dataset.
        layer: Layer whose domains take precedence over the fallback.
        fields: Fields referenced by the active rules.

    Returns:
        Allowed codes keyed by rule field.
    """
    layer_defn = layer.GetLayerDefn()
    allowed_values: dict[str, set[str] | frozenset[str]] = {}
    for field_name in PADUS_RULE_FIELDS:
        if field_name not in fields:
            continue
        field_index = layer_defn.GetFieldIndex(field_name)
        if field_index < 0:
            raise RuleConfigError(
                f"PAD-US layer does not contain required rule field {field_name!r}."
            )
        field_defn = layer_defn.GetFieldDefn(field_index)
        domain_name = field_defn.GetDomainName()
        domain = dataset.GetFieldDomain(domain_name) if domain_name else None
        enumeration = domain.GetEnumeration() if domain is not None else None
        if enumeration:
            allowed_values[field_name] = set(enumeration)
        elif field_name in PUBLIC_LAND_DOMAINS:
            allowed_values[field_name] = PUBLIC_LAND_DOMAINS[field_name]
        else:
            raise RuleConfigError(
                f"PAD-US field {field_name!r} does not expose a coded domain."
            )
    allowed_values["manager"] = VIRTUAL_MANAGER_VALUES
    return allowed_values


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


def _ogr_geometry_to_shapely(geom: ogr.Geometry):
    """Convert an OGR geometry to a Shapely geometry.

    PAD-US FileGDB features can be exposed as OGR `MULTISURFACE` geometries.
    Shapely cannot read those nonlinear wrapper types from WKB, even when the
    contents are polygonal. Linearizing with OGR first converts them to simple
    polygonal geometry types such as `MULTIPOLYGON`.

    Args:
        geom: OGR geometry to convert.

    Returns:
        Equivalent Shapely geometry with linear OGR geometry types.
    """
    linear_geom = geom.GetLinearGeometry()
    return wkb.loads(bytes(linear_geom.ExportToWkb()))


def _source_field_specs(layer_defn: ogr.FeatureDefn) -> list[FieldSpec]:
    """Return serializable field definitions from the PAD-US source layer.

    Args:
        layer_defn: PAD-US source layer definition.

    Returns:
        Field name, type, subtype, width, and precision for each source field.
    """
    field_specs = []
    for field_index in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(field_index)
        field_specs.append(
            (
                field_defn.GetName(),
                field_defn.GetType(),
                field_defn.GetSubType(),
                field_defn.GetWidth(),
                field_defn.GetPrecision(),
            )
        )
    return field_specs


def _feature_source_fields(feature: ogr.Feature, field_names: list[str]) -> dict:
    """Read source PAD-US attributes for an output feature.

    Args:
        feature: PAD-US source feature.
        field_names: Normal source fields to copy.

    Returns:
        Field values keyed by output field name, including the source FID as
        `OBJECTID`.
    """
    fields = {SOURCE_OBJECTID_FIELD: feature.GetFID()}
    for field_name in field_names:
        fields[field_name] = feature.GetField(field_name)
    return fields


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
        geoms.append(_ogr_geometry_to_shapely(geom))

    boundary_vector = None
    boundary_layer = None
    boundary = shapely.union_all(geoms)
    boundary = repair_polygonal_geometry(boundary)
    if boundary is None:
        raise RuntimeError("USA boundary did not contain valid polygon geometry.")
    return boundary


def _build_jobs(
    layer: ogr.Layer,
    boundary,
    rules: RuleSet,
    rule_fields: tuple[str, ...],
) -> tuple[list[list[int]], Counter]:
    """Scan PAD-US features and group candidate FIDs into vertex-sized jobs.

    Args:
        layer: PAD-US source layer.
        boundary: USA boundary geometry in the source layer CRS.
        rules: Classifications selected for this run.
        rule_fields: Physical source fields used by those rules.

    Returns:
        Job FID lists and scan counters.
    """
    jobs = []
    current_job = []
    current_vertices = 0
    stats = Counter()
    boundary_minx, boundary_miny, boundary_maxx, boundary_maxy = boundary.bounds

    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Scan PAD-US features",
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
        classifications = _feature_classifications(feature, rules, rule_fields)
        if classifications.get("public_land"):
            stats["public_candidate_features"] += 1
        if classifications.get("public_access"):
            stats["public_access_candidate_features"] += 1

        if current_vertices >= VERTICES_PER_JOB:
            jobs.append(current_job)
            current_job = []
            current_vertices = 0

    if current_job:
        jobs.append(current_job)

    stats["jobs"] = len(jobs)
    return jobs, stats


def _init_worker(
    padus_path: str,
    layer_name: str,
    source_field_names: list[str],
    rules: RuleSet,
    rule_fields: tuple[str, ...],
    products: tuple[str, ...],
    source_srs_wkt: str,
    process_srs_wkt: str,
    boundary_wkb: bytes,
) -> None:
    """Initialize per-process PAD-US reader state.

    Args:
        padus_path: Path to the selected PAD-US dataset.
        layer_name: Layer to process in the dataset.
        source_field_names: Source fields to copy into each output row.
        rules: Validated rules for the requested products.
        rule_fields: Physical source fields used by those rules.
        products: Land-product keys to write.
        source_srs_wkt: Source layer CRS WKT.
        process_srs_wkt: Processing CRS WKT.
        boundary_wkb: USA boundary WKB in the processing CRS.
    """
    source_srs = _set_axis_order(osr.SpatialReference(wkt=source_srs_wkt))
    process_srs = _set_axis_order(osr.SpatialReference(wkt=process_srs_wkt))
    transform = None
    if not source_srs.IsSame(process_srs):
        transform = osr.CoordinateTransformation(source_srs, process_srs)

    padus_vector = gdal.OpenEx(padus_path, gdal.OF_VECTOR | gdal.OF_READONLY)
    padus_layer = padus_vector.GetLayerByName(layer_name)
    WORKER["ds"] = padus_vector
    WORKER["layer"] = padus_layer
    WORKER["source_field_names"] = source_field_names
    WORKER["rules"] = rules
    WORKER["rule_fields"] = rule_fields
    WORKER["products"] = products
    WORKER["transform"] = transform
    WORKER["boundary"] = wkb.loads(boundary_wkb)


def _process_job(
    fids: list[int],
) -> tuple[
    dict[str, list[OutputRecord]],
    Counter,
    list[dict[str, str]],
]:
    """Process a chunk of PAD-US FIDs.

    Args:
        fids: Source feature IDs to process.

    Returns:
        Output records keyed by product, processing counters, and failure rows.
    """
    layer = WORKER["layer"]
    source_field_names = WORKER["source_field_names"]
    rules = WORKER["rules"]
    rule_fields = WORKER["rule_fields"]
    transform = WORKER["transform"]
    boundary = WORKER["boundary"]
    records = {product: [] for product in WORKER["products"]}
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
            shapely_geom = _ogr_geometry_to_shapely(geom)
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
            source_fields = _feature_source_fields(feature, source_field_names)
            classifications = _feature_classifications(feature, rules, rule_fields)
            for product, product_records in records.items():
                rule = PRODUCTS[product][2]
                if rule is None or classifications[rule]:
                    product_records.append((clipped_wkb, source_fields))
                    stats[f"{product.replace('-', '_')}_kept"] += 1
        except Exception as error:
            stats["exceptions"] += 1
            failures.append({"source_fid": fid, "reason": str(error)})

    return records, stats, failures


def _create_output_layer(
    out_path: Path,
    layer_name: str,
    process_srs: osr.SpatialReference,
    source_field_specs: list[FieldSpec],
) -> tuple[ogr.DataSource, ogr.Layer]:
    """Create the output GeoPackage layer.

    Args:
        out_path: Output GeoPackage path.
        layer_name: Output layer name.
        process_srs: Output spatial reference.
        source_field_specs: PAD-US source field definitions to copy.

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
    out_layer.CreateField(ogr.FieldDefn(SOURCE_OBJECTID_FIELD, ogr.OFTInteger64))
    for field_name, field_type, field_subtype, field_width, field_precision in (
        source_field_specs
    ):
        field_defn = ogr.FieldDefn(field_name, field_type)
        field_defn.SetSubType(field_subtype)
        field_defn.SetWidth(field_width)
        field_defn.SetPrecision(field_precision)
        out_layer.CreateField(field_defn)

    return out_ds, out_layer


def _write_geometry_feature(
    out_layer: ogr.Layer,
    out_defn: ogr.FeatureDefn,
    geom_wkb: bytes,
    land_type: str,
    source_fields: dict[str, object],
) -> None:
    """Write one output geometry with its land type.

    Args:
        out_layer: Destination OGR layer.
        out_defn: Destination layer definition.
        geom_wkb: Output geometry WKB.
        land_type: Output land-type label.
        source_fields: PAD-US source attributes to copy.
    """
    feature = ogr.Feature(out_defn)
    feature.SetField("land_type", land_type)
    for field_name, value in source_fields.items():
        if value is not None:
            feature.SetField(field_name, value)
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
    """Prepare the selected PAD-US land products."""
    args = _parse_args()
    products = tuple(dict.fromkeys(args.products))
    required = set()
    if "public" in products or "public-access" in products:
        required.add("public_land")
    if "public-access" in products:
        required.add("public_access")
    required_rules = frozenset(required)
    if required_rules:
        _, config_rules = load_rule_config(args.rules_config)
        rules = config_rules.select(required_rules)
    else:
        rules = RuleSet({})

    padus_path = args.padus_path.resolve()
    padus_vector = gdal.OpenEx(str(padus_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    padus_layer = padus_vector.GetLayerByName(args.padus_layer)
    if padus_layer is None:
        raise ValueError(f"Layer {args.padus_layer!r} not found in {padus_path}")
    source_field_specs = _source_field_specs(padus_layer.GetLayerDefn())
    source_field_names = [field_spec[0] for field_spec in source_field_specs]
    referenced = rules.referenced_fields()
    physical_fields = set(referenced.intersection(PADUS_RULE_FIELDS))
    if "manager" in referenced:
        physical_fields.update(("Mang_Name", "Loc_Mang"))
    missing_fields = physical_fields.difference(source_field_names)
    if missing_fields:
        raise RuleConfigError(
            f"PAD-US layer {args.padus_layer!r} is missing fields for "
            f"{', '.join(products)}: {', '.join(sorted(missing_fields))}"
        )
    rule_fields = tuple(sorted(physical_fields))
    rules.validate(
        _rule_domain_values(padus_vector, padus_layer, referenced),
        required_rules=required_rules,
    )
    source_srs = padus_layer.GetSpatialRef()
    if source_srs is None:
        raise ValueError(f"PAD-US layer {args.padus_layer!r} has no CRS")
    source_srs = _set_axis_order(source_srs)
    process_srs = _choose_process_srs(source_srs)

    print(f"PAD-US source: {padus_path}", flush=True)
    print(f"Layer: {args.padus_layer}", flush=True)
    print(f"Products: {', '.join(products)}", flush=True)
    if args.validate_only:
        print("PAD-US input and rule validation passed; no outputs written.", flush=True)
        return

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    paths = {
        product: args.output_dir / PRODUCTS[product][0] / f"{PRODUCTS[product][1]}_{timestamp}.gpkg"
        for product in products
    }
    failure_path = args.output_dir / f"{FAILURE_STEM}_{timestamp}_skipped.csv"
    for path in (*paths.values(), failure_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    step_bar = tqdm(total=5, initial=1, desc="PAD-US preprocessing", unit="step")
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

    jobs, scan_stats = _build_jobs(padus_layer, boundary_for_scan, rules, rule_fields)
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
    if required_rules:
        print(f"Rules: {args.rules_config}", flush=True)
    print(f"Failure log: {failure_path}", flush=True)

    step_bar.set_description("Create outputs")
    outputs = {}
    for product, path in paths.items():
        print(f"{product} output: {path}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        outputs[product] = _create_output_layer(
            path, PRODUCTS[product][1], process_srs, source_field_specs,
        )
    step_bar.update()

    step_bar.set_description("Process jobs")
    all_stats = Counter()
    failures = []
    written = Counter()
    worker_args = (
        str(padus_path),
        args.padus_layer,
        source_field_names,
        rules,
        rule_fields,
        products,
        source_srs.ExportToWkt(),
        process_srs.ExportToWkt(),
        wkb.dumps(boundary),
    )

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=worker_args,
    ) as executor:
        futures = {executor.submit(_process_job, job) for job in jobs}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Process PAD-US jobs",
            unit="job",
        ):
            records, stats, job_failures = future.result()
            # Release completed batches rather than retaining their geometries.
            futures.remove(future)
            all_stats.update(stats)
            failures.extend(job_failures)
            for product, product_records in records.items():
                out_layer = outputs[product][1]
                out_defn = out_layer.GetLayerDefn()
                for geom_wkb, source_fields in product_records:
                    _write_geometry_feature(
                        out_layer, out_defn, geom_wkb, product.replace("-", "_"), source_fields,
                    )
                written[product] += len(product_records)

    for output_ds, _ in outputs.values():
        output_ds.FlushCache()
    outputs.clear()
    output_ds = None

    if failures:
        _write_failures(failure_path, failures)

    step_bar.update()
    step_bar.close()
    print(
        "Processing stats: "
        + " ".join(f"{key}={value:,}" for key, value in all_stats.items()),
        flush=True,
    )
    for product, path in paths.items():
        print(f"Wrote {written[product]:,} {product} feature(s): {path}", flush=True)
    if failures:
        print(f"Wrote {len(failures):,} skipped feature log row(s): {failure_path}")
    else:
        print("No unrepaired geometry failures were logged.")


if __name__ == "__main__":
    main()
