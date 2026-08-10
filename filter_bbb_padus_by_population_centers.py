"""Screen PAD-US land against mappable Section 50301 sale criteria.

This applies the bill's mappable Bureau of Land Management, eligible-state,
and population-center conditions:

* federally owned fee land managed by the Bureau of Land Management;
* land in one of the 11 eligible states;
* five statute miles from an incorporated municipality's boundary; or
* five statute miles from a Census-designated place's Census centroid; and
* land outside federally protected areas represented in PAD-US.

The workflow writes an intermediate layer of BLM land after protected areas are
subtracted and a final layer that also applies the population-center rule.
Neither result determines that a tract will be sold. PAD-US does not establish
grazing permits, incompatible existing rights, residential suitability, or
tract selection. Fish and Wildlife Service approved/proclamation boundaries
are used as a conservative proxy for units of the National Wildlife Refuge and
National Fish Hatchery Systems because PAD-US does not identify every system
boundary with a distinct designation code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from osgeo import gdal, ogr
from pyproj import CRS, Transformer
import shapely
from shapely import wkb
from shapely.geometry import Point
from shapely.ops import transform
from shapely.strtree import STRtree
from tqdm import tqdm

from geometry_utils import repair_polygonal_geometry


gdal.UseExceptions()
ogr.UseExceptions()

DEFAULT_POPULATION_CENTERS = Path(
    "data/analysis_inputs/census_population_centers_2020.gpkg"
)
DEFAULT_UNPROTECTED_OUTPUT_DIR = Path(
    "data/processing_outputs/blm_lands_excluding_federally_protected_areas"
)
DEFAULT_OUTPUT_DIR = Path("data/processing_outputs/bbb_candidate_blm_lands")
INCORPORATED_LAYER = "incorporated_places_pop1000"
CDP_LAYER = "census_designated_places_pop1000"
DEFAULT_UNPROTECTED_OUTPUT_LAYER = (
    "blm_lands_excluding_federally_protected_areas"
)
DEFAULT_OUTPUT_LAYER = "bbb_candidate_blm_lands"
UNPROTECTED_LAND_TYPE = "blm_excluding_federally_protected"
BBB_CANDIDATE_LAND_TYPE = "bbb_candidate_blm"
DEFAULT_DISTANCE_MILES = 5.0
METERS_PER_MILE = 1609.344
TRANSACTION_SIZE = 10_000
ELIGIBLE_STATE_CODES = (
    "AK",
    "AZ",
    "CA",
    "CO",
    "ID",
    "NV",
    "NM",
    "OR",
    "UT",
    "WA",
    "WY",
)
ELIGIBLE_STATE_FIPS = frozenset((2, 4, 6, 8, 16, 32, 35, 41, 49, 53, 56))
REQUIRED_PADUS_FIELDS = frozenset(
    ("FeatClass", "Own_Type", "Mang_Name", "State_Nm", "Des_Tp")
)
FEDERALLY_PROTECTED_DESIGNATION_TYPES = (
    "NCA",
    "NM",
    "NP",
    "NRA",
    "NT",
    "NWR",
    "WA",
    "WSR",
)
BBB_ATTRIBUTE_FILTER = (
    "FeatClass = 'Fee' AND Own_Type = 'FED' AND Mang_Name = 'BLM' "
    "AND State_Nm IN ("
    + ", ".join(f"'{state}'" for state in ELIGIBLE_STATE_CODES)
    + ")"
)
FEDERALLY_PROTECTED_ATTRIBUTE_FILTER = (
    "Des_Tp IN ("
    + ", ".join(
        f"'{designation_type}'"
        for designation_type in FEDERALLY_PROTECTED_DESIGNATION_TYPES
    )
    + ") OR (FeatClass = 'Proclamation' AND Mang_Name IN ('NPS', 'FWS'))"
)


@dataclass(frozen=True)
class BbbFilterResult:
    """Summary returned by the bill-specific PAD-US screening workflow."""

    unprotected_output_path: Path
    output_path: Path
    input_features: int
    bbb_candidate_features: int
    unprotected_features: int
    retained_features: int
    incorporated_places: int
    census_designated_places: int
    federally_protected_features: int
    stats: Counter


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the bill-specific screening workflow.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Select federally owned BLM-managed fee land in an eligible state "
            "and clip it to qualifying population-center proximity zones after "
            "subtracting federally protected PAD-US areas."
        )
    )
    parser.add_argument("input_gpkg", type=Path, help="Input PAD-US GeoPackage.")
    parser.add_argument(
        "--input-layer",
        help="Input layer name. Defaults to the only layer in the GeoPackage.",
    )
    parser.add_argument(
        "--population-centers-gpkg",
        type=Path,
        default=DEFAULT_POPULATION_CENTERS,
        help=f"Population-center GeoPackage (default: {DEFAULT_POPULATION_CENTERS}).",
    )
    parser.add_argument(
        "--incorporated-layer",
        default=INCORPORATED_LAYER,
        help=f"Incorporated-place layer (default: {INCORPORATED_LAYER}).",
    )
    parser.add_argument(
        "--cdp-layer",
        default=CDP_LAYER,
        help=f"Census-designated-place layer (default: {CDP_LAYER}).",
    )
    parser.add_argument(
        "--distance-miles",
        type=float,
        default=DEFAULT_DISTANCE_MILES,
        help=f"Buffer distance in statute miles (default: {DEFAULT_DISTANCE_MILES}).",
    )
    parser.add_argument(
        "--unprotected-output",
        type=Path,
        help=(
            "Intermediate GeoPackage containing eligible-state BLM fee land "
            "after federally protected areas are subtracted. Defaults to a "
            "timestamped processing output."
        ),
    )
    parser.add_argument(
        "--unprotected-output-layer",
        default=DEFAULT_UNPROTECTED_OUTPUT_LAYER,
        help=(
            "Intermediate layer name "
            f"(default: {DEFAULT_UNPROTECTED_OUTPUT_LAYER})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Final BBB candidate GeoPackage. Defaults to a timestamped "
            "processing output."
        ),
    )
    parser.add_argument(
        "--output-layer",
        default=DEFAULT_OUTPUT_LAYER,
        help=f"Output layer name (default: {DEFAULT_OUTPUT_LAYER}).",
    )
    return parser.parse_args()


def _open_input_layer(
    input_path: Path,
    requested_layer: str | None,
) -> tuple[gdal.Dataset, ogr.Layer, str]:
    """Open and validate the PAD-US input layer.

    Args:
        input_path: GeoPackage containing PAD-US features.
        requested_layer: Explicit layer name, or ``None`` to require a
            single-layer GeoPackage.

    Returns:
        Open dataset, selected layer, and resolved layer name.

    Raises:
        FileNotFoundError: If the input GeoPackage does not exist.
        RuntimeError: If GDAL cannot open the GeoPackage.
        ValueError: If the requested layer is unavailable, the GeoPackage has
            multiple layers without an override, the layer lacks a CRS, or a
            field required by the bill-specific filter is missing.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input GeoPackage does not exist: {input_path}")

    dataset = gdal.OpenEx(str(input_path), gdal.OF_VECTOR | gdal.OF_READONLY)
    if dataset is None:
        raise RuntimeError(f"Could not open input GeoPackage: {input_path}")

    if requested_layer:
        layer = dataset.GetLayerByName(requested_layer)
        if layer is None:
            raise ValueError(
                f"Input layer {requested_layer!r} does not exist in {input_path}."
            )
        layer_name = requested_layer
    else:
        if dataset.GetLayerCount() != 1:
            raise ValueError(
                f"{input_path} contains {dataset.GetLayerCount()} layers; "
                "provide --input-layer."
            )
        layer = dataset.GetLayer(0)
        layer_name = layer.GetName()

    if layer.GetSpatialRef() is None:
        raise ValueError(f"Input layer {layer_name!r} has no CRS.")

    definition = layer.GetLayerDefn()
    available_fields = {
        definition.GetFieldDefn(index).GetName()
        for index in range(definition.GetFieldCount())
    }
    missing_fields = REQUIRED_PADUS_FIELDS - available_fields
    if missing_fields:
        raise ValueError(
            f"Input layer {layer_name!r} is missing BBB filter field(s): "
            + ", ".join(sorted(missing_fields))
        )
    return dataset, layer, layer_name


def _read_population_layer(path: Path, layer_name: str) -> gpd.GeoDataFrame:
    """Read and validate one population-center layer.

    Args:
        path: Population-center GeoPackage.
        layer_name: Layer to load from the GeoPackage.

    Returns:
        Non-empty GeoDataFrame with a defined CRS and complete geometry.

    Raises:
        FileNotFoundError: If the population-center GeoPackage does not exist.
        ValueError: If the layer cannot be read or contains unusable data.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Population-center GeoPackage does not exist: {path}")
    try:
        frame = gpd.read_file(path, layer=layer_name)
    except Exception as error:
        message = (
            f"Could not read population-center layer {layer_name!r} "
            f"from {path}: {error}"
        )
        raise ValueError(message) from error
    if frame.crs is None:
        raise ValueError(f"Population-center layer {layer_name!r} has no CRS.")
    if frame.empty:
        raise ValueError(f"Population-center layer {layer_name!r} is empty.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(
            f"Population-center layer {layer_name!r} contains empty geometry."
        )
    return frame


@lru_cache(maxsize=256)
def _coordinate_transformer(source_crs: CRS, target_crs: CRS) -> Transformer:
    """Return a cached coordinate transformer.

    Args:
        source_crs: Geometry's current coordinate reference system.
        target_crs: Desired coordinate reference system.

    Returns:
        Reusable always-x/y pyproj transformer.
    """
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _transform_geometry(geometry, source_crs: CRS, target_crs: CRS):
    """Transform a Shapely geometry between coordinate systems.

    Args:
        geometry: Shapely geometry to transform.
        source_crs: Geometry's current coordinate reference system.
        target_crs: Desired coordinate reference system.

    Returns:
        Geometry in ``target_crs``, or the original geometry when both CRSs
        are equal.
    """
    if source_crs == target_crs:
        return geometry
    transformer = _coordinate_transformer(source_crs, target_crs)
    return transform(transformer.transform, geometry)


def _local_utm_crs(geometry, source_crs: CRS) -> CRS:
    """Choose the local UTM CRS containing a geometry's representative point.

    Args:
        geometry: Shapely geometry used to choose a UTM zone.
        source_crs: Geometry's coordinate reference system.

    Returns:
        Northern- or southern-hemisphere UTM CRS for the geometry.

    Raises:
        ValueError: If the representative point is outside UTM coverage.
    """
    representative_point = geometry.representative_point()
    lon_lat = _transform_geometry(
        representative_point,
        source_crs,
        CRS.from_epsg(4326),
    )
    longitude = lon_lat.x
    latitude = lon_lat.y
    if not (-80 <= latitude <= 84):
        raise ValueError(
            f"Population-center latitude {latitude} is outside UTM coverage."
        )
    zone = max(1, min(60, int((longitude + 180) // 6) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _buffer_geometry_locally(
    geometry,
    source_crs: CRS,
    target_crs: CRS,
    distance_meters: float,
    *,
    boundary_only: bool,
):
    """Buffer one geometry accurately in its local UTM CRS.

    Args:
        geometry: Shapely geometry to buffer.
        source_crs: Geometry's coordinate reference system.
        target_crs: CRS required by the PAD-US input layer.
        distance_meters: Buffer distance in meters.
        boundary_only: Whether to buffer only the geometry's boundary instead
            of its complete area.

    Returns:
        Valid MultiPolygon buffer transformed into ``target_crs``.

    Raises:
        ValueError: If buffering does not produce polygon geometry.
    """
    local_crs = _local_utm_crs(geometry, source_crs)
    local_geometry = _transform_geometry(geometry, source_crs, local_crs)
    buffer_source = local_geometry.boundary if boundary_only else local_geometry
    buffered = buffer_source.buffer(distance_meters)
    buffered = _transform_geometry(buffered, local_crs, target_crs)
    buffered = repair_polygonal_geometry(buffered)
    if buffered is None:
        raise ValueError("A population-center buffer produced no polygon geometry.")
    return buffered


def _cdp_centroid(row, layer_name: str) -> Point:
    """Create a CDP center point from Census centroid attributes.

    Args:
        row: Population-center feature attributes.
        layer_name: Source layer name used in validation messages.

    Returns:
        Point constructed from ``CENTLON`` and ``CENTLAT``.

    Raises:
        ValueError: If either centroid field is missing or nonnumeric.
    """
    missing = {"CENTLON", "CENTLAT"} - set(row.index)
    if missing:
        raise ValueError(
            f"Population-center layer {layer_name!r} is missing centroid field(s): "
            + ", ".join(sorted(missing))
        )
    try:
        return Point(float(row["CENTLON"]), float(row["CENTLAT"]))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Population-center layer {layer_name!r} has invalid CENTLON/CENTLAT "
            f"for GEOID {row.get('GEOID', '<unknown>')!r}."
        ) from error


def build_proximity_zone(
    population_centers_path: Path,
    incorporated_layer: str,
    cdp_layer: str,
    target_crs: CRS,
    distance_miles: float,
) -> tuple[list, int, int]:
    """Build non-overlapping proximity-zone parts in the PAD-US CRS.

    Progress bars report population-layer loads, local buffering for both
    place types, and the final union operation.

    Args:
        population_centers_path: GeoPackage containing qualifying places.
        incorporated_layer: Incorporated-municipality layer name.
        cdp_layer: Census-designated-place layer name.
        target_crs: PAD-US CRS for the returned zones.
        distance_miles: Buffer distance in statute miles.

    Returns:
        Non-overlapping zone polygons, incorporated-place count, and CDP
        count.

    Raises:
        ValueError: If the distance is nonpositive or source data is invalid.
        RuntimeError: If the buffers cannot produce a polygonal union.
    """
    if distance_miles <= 0:
        raise ValueError("--distance-miles must be greater than zero.")

    with tqdm(
        total=2,
        desc="Load population-center layers",
        unit="layer",
    ) as load_bar:
        load_bar.set_postfix_str(incorporated_layer)
        incorporated = _read_population_layer(
            population_centers_path,
            incorporated_layer,
        )
        load_bar.update()
        load_bar.set_postfix_str(cdp_layer)
        cdps = _read_population_layer(population_centers_path, cdp_layer)
        load_bar.update()

    for frame, layer_name in (
        (incorporated, incorporated_layer),
        (cdps, cdp_layer),
    ):
        if "STATE" not in frame.columns:
            raise ValueError(
                f"Population-center layer {layer_name!r} is missing STATE."
            )
    incorporated = incorporated[
        incorporated["STATE"].isin(ELIGIBLE_STATE_FIPS)
    ].copy()
    cdps = cdps[cdps["STATE"].isin(ELIGIBLE_STATE_FIPS)].copy()
    if incorporated.empty or cdps.empty:
        raise ValueError(
            "Population-center layers contain no places in the eligible states."
        )

    incorporated_crs = CRS.from_user_input(incorporated.crs)
    cdp_crs = CRS.from_user_input(cdps.crs)
    distance_meters = distance_miles * METERS_PER_MILE

    buffers = []
    for geometry in tqdm(
        incorporated.geometry,
        total=len(incorporated),
        desc="Buffer municipality boundaries",
        unit="place",
    ):
        buffers.append(
            _buffer_geometry_locally(
                geometry,
                incorporated_crs,
                target_crs,
                distance_meters,
                boundary_only=True,
            )
        )

    for _, row in tqdm(
        cdps.iterrows(),
        total=len(cdps),
        desc="Buffer CDP centroids",
        unit="place",
    ):
        center = _cdp_centroid(row, cdp_layer)
        buffers.append(
            _buffer_geometry_locally(
                center,
                cdp_crs,
                target_crs,
                distance_meters,
                boundary_only=False,
            )
        )

    with tqdm(
        total=1,
        desc="Union proximity buffers",
        unit="operation",
    ) as union_bar:
        merged = repair_polygonal_geometry(shapely.union_all(buffers))
        union_bar.update()
    if merged is None:
        raise RuntimeError("Population-center buffers produced no proximity zone.")
    return list(merged.geoms), len(incorporated), len(cdps)


def load_federally_protected_geometries(
    input_layer: ogr.Layer,
) -> tuple[list, Counter]:
    """Load federally protected area geometries represented in PAD-US.

    The designation codes cover National Monuments, National Recreation Areas,
    Wilderness Areas, Wild and Scenic Rivers, National Trails, National
    Conservation Areas, National Wildlife Refuges, and National Parks. NPS and
    FWS approved/proclamation boundaries supplement those codes to represent
    units of the National Park, National Wildlife Refuge, and National Fish
    Hatchery Systems. The FWS boundary selection is intentionally conservative
    because PAD-US does not expose a distinct system-membership field.

    Args:
        input_layer: PAD-US layer containing both ownership and protected-area
            records.

    Returns:
        Valid polygonal geometries and counters describing source geometry
        handling.

    Raises:
        RuntimeError: If GDAL cannot apply the protected-area filter.
    """
    if input_layer.SetAttributeFilter(FEDERALLY_PROTECTED_ATTRIBUTE_FILTER) != 0:
        raise RuntimeError(
            "GDAL could not apply the federally protected PAD-US filter."
        )
    source_count = input_layer.GetFeatureCount()
    input_layer.ResetReading()
    stats = Counter({"federally_protected_source_features": source_count})
    geometries = []
    try:
        for feature in tqdm(
            input_layer,
            total=source_count,
            desc="Load federally protected areas",
            unit="feature",
        ):
            ogr_geometry = feature.GetGeometryRef()
            if ogr_geometry is None or ogr_geometry.IsEmpty():
                stats["federally_protected_empty_skipped"] += 1
                continue
            geometry = repair_polygonal_geometry(
                wkb.loads(bytes(ogr_geometry.GetLinearGeometry().ExportToWkb()))
            )
            if geometry is None:
                stats["federally_protected_invalid_skipped"] += 1
                continue
            geometries.extend(geometry.geoms)
    finally:
        input_layer.SetAttributeFilter(None)
        input_layer.ResetReading()
    stats["federally_protected_geometry_parts"] = len(geometries)
    return geometries, stats


def _create_output(
    output_path: Path,
    output_layer_name: str,
    input_layer: ogr.Layer,
) -> tuple[ogr.DataSource, ogr.Layer]:
    """Create an output GeoPackage matching the PAD-US source schema.

    Args:
        output_path: New GeoPackage to create.
        output_layer_name: Name for the clipped output layer.
        input_layer: PAD-US layer whose CRS and fields are copied.

    Returns:
        Writable output dataset and layer.

    Raises:
        FileExistsError: If the requested output already exists.
        RuntimeError: If GDAL cannot create the output dataset or layer.
    """
    if output_path.exists():
        raise FileExistsError(f"Output GeoPackage already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(output_path))
    if dataset is None:
        raise RuntimeError(f"Could not create output GeoPackage: {output_path}")
    layer = dataset.CreateLayer(
        output_layer_name,
        input_layer.GetSpatialRef(),
        ogr.wkbMultiPolygon,
        options=["SPATIAL_INDEX=YES"],
    )
    if layer is None:
        raise RuntimeError(f"Could not create output layer {output_layer_name!r}.")

    input_definition = input_layer.GetLayerDefn()
    for field_index in tqdm(
        range(input_definition.GetFieldCount()),
        desc="Copy output schema",
        unit="field",
    ):
        source = input_definition.GetFieldDefn(field_index)
        destination = ogr.FieldDefn(source.GetName(), source.GetType())
        destination.SetSubType(source.GetSubType())
        destination.SetWidth(source.GetWidth())
        destination.SetPrecision(source.GetPrecision())
        layer.CreateField(destination)
    return dataset, layer


def filter_bbb_padus_by_population_centers(
    input_path: Path,
    population_centers_path: Path,
    output_path: Path,
    *,
    unprotected_output_path: Path,
    input_layer_name: str | None = None,
    unprotected_output_layer_name: str = DEFAULT_UNPROTECTED_OUTPUT_LAYER,
    output_layer_name: str = DEFAULT_OUTPUT_LAYER,
    incorporated_layer: str = INCORPORATED_LAYER,
    cdp_layer: str = CDP_LAYER,
    distance_miles: float = DEFAULT_DISTANCE_MILES,
) -> BbbFilterResult:
    """Screen PAD-US land against the bill's mappable sale conditions.

    Separate progress bars cover setup stages, source-feature scanning, output
    writes, and GeoPackage finalization.

    Args:
        input_path: PAD-US GeoPackage to screen, preferably the all-land output.
        population_centers_path: GeoPackage containing qualifying places.
        output_path: New final BBB candidate GeoPackage to create.
        unprotected_output_path: New intermediate GeoPackage for eligible-state
            BLM fee land after federally protected areas are subtracted.
        input_layer_name: Optional PAD-US layer override.
        unprotected_output_layer_name: Layer name for the intermediate
            unprotected BLM features.
        output_layer_name: Layer name for the final BBB candidate features.
        incorporated_layer: Incorporated-municipality layer name.
        cdp_layer: Census-designated-place layer name.
        distance_miles: Proximity distance in statute miles.

    Returns:
        Intermediate and final output paths, source counts, stage counts, and
        processing counters.

    Raises:
        FileNotFoundError: If an input GeoPackage does not exist.
        FileExistsError: If the output GeoPackage already exists.
        ValueError: If output paths are identical or an input layer, CRS,
            distance, or centroid is invalid.
        RuntimeError: If GDAL or geometry processing cannot create the result.
    """
    if unprotected_output_path.resolve() == output_path.resolve():
        raise ValueError(
            "Intermediate unprotected and final BBB outputs must be different files."
        )
    for output_description, candidate_path in (
        ("Intermediate unprotected", unprotected_output_path),
        ("Final BBB", output_path),
    ):
        if candidate_path.exists():
            raise FileExistsError(
                f"{output_description} GeoPackage already exists: {candidate_path}"
            )

    with tqdm(total=9, desc="Set up BBB land filter", unit="stage") as setup_bar:
        setup_bar.set_postfix_str("Open PAD-US input")
        input_dataset, input_layer, resolved_input_layer = _open_input_layer(
            input_path, input_layer_name
        )
        input_count = input_layer.GetFeatureCount()
        target_crs = CRS.from_wkt(input_layer.GetSpatialRef().ExportToWkt())
        setup_bar.update()

        setup_bar.set_postfix_str("Load federally protected areas")
        protected_parts, protected_stats = load_federally_protected_geometries(
            input_layer
        )
        stats = Counter(protected_stats)
        setup_bar.update()

        setup_bar.set_postfix_str("Build protected-area spatial index")
        protected_index = STRtree(protected_parts)
        setup_bar.update()

        setup_bar.set_postfix_str("Select BLM fee land in eligible states")
        if input_layer.SetAttributeFilter(BBB_ATTRIBUTE_FILTER) != 0:
            raise RuntimeError("GDAL could not apply the BBB PAD-US attribute filter.")
        bbb_candidate_count = input_layer.GetFeatureCount()
        input_layer.ResetReading()
        stats.update(
            {
                "input_features": input_count,
                "bbb_attribute_candidates": bbb_candidate_count,
                "bbb_attribute_skipped": input_count - bbb_candidate_count,
            }
        )
        setup_bar.update()

        setup_bar.set_postfix_str("Build population proximity zones")
        zone_parts, incorporated_count, cdp_count = build_proximity_zone(
            population_centers_path,
            incorporated_layer,
            cdp_layer,
            target_crs,
            distance_miles,
        )
        setup_bar.update()

        setup_bar.set_postfix_str("Build proximity spatial index")
        zone_index = STRtree(zone_parts)
        setup_bar.update()

        setup_bar.set_postfix_str("Create unprotected BLM GeoPackage")
        unprotected_dataset, unprotected_layer = _create_output(
            unprotected_output_path,
            unprotected_output_layer_name,
            input_layer,
        )
        setup_bar.update()

        setup_bar.set_postfix_str("Create final BBB GeoPackage")
        output_dataset, output_layer = _create_output(
            output_path,
            output_layer_name,
            input_layer,
        )
        setup_bar.update()

        setup_bar.set_postfix_str("Start output transactions")
        unprotected_layer.StartTransaction()
        output_layer.StartTransaction()
        setup_bar.update()

    with (
        tqdm(
            input_layer,
            total=bbb_candidate_count,
            desc=f"Process BBB candidates from {resolved_input_layer}",
            unit="feature",
        ) as scan_bar,
        tqdm(
            desc="Write BLM land excluding protected areas",
            unit="feature",
        ) as unprotected_write_bar,
        tqdm(desc="Write final BBB candidates", unit="feature") as write_bar,
    ):
        for input_feature in scan_bar:
            stats["scanned"] += 1
            ogr_geometry = input_feature.GetGeometryRef()
            if ogr_geometry is None or ogr_geometry.IsEmpty():
                stats["empty_geometry_skipped"] += 1
                continue

            geometry = repair_polygonal_geometry(
                wkb.loads(bytes(ogr_geometry.GetLinearGeometry().ExportToWkb()))
            )
            if geometry is None:
                stats["invalid_geometry_skipped"] += 1
                continue

            protected_indexes = protected_index.query(
                geometry,
                predicate="intersects",
            )
            unprotected = geometry
            if len(protected_indexes) > 0:
                protected_zone = shapely.union_all(
                    [protected_parts[index] for index in protected_indexes]
                )
                unprotected = repair_polygonal_geometry(
                    shapely.difference(geometry, protected_zone)
                )
                if unprotected is None or unprotected.area <= 0:
                    stats["federally_protected_fully_excluded"] += 1
                    continue
                if unprotected.area < geometry.area:
                    stats["federally_protected_clipped"] += 1

            unprotected_feature = ogr.Feature(unprotected_layer.GetLayerDefn())
            input_definition = input_feature.GetDefnRef()
            for field_index in range(input_definition.GetFieldCount()):
                value = input_feature.GetField(field_index)
                if value is not None:
                    unprotected_feature.SetField(field_index, value)
            if unprotected_feature.GetFieldIndex("land_type") >= 0:
                unprotected_feature.SetField(
                    "land_type",
                    UNPROTECTED_LAND_TYPE,
                )
            unprotected_feature.SetGeometry(
                ogr.CreateGeometryFromWkb(wkb.dumps(unprotected))
            )
            unprotected_layer.CreateFeature(unprotected_feature)
            unprotected_feature = None
            stats["unprotected_retained"] += 1
            unprotected_write_bar.update()
            if stats["unprotected_retained"] % TRANSACTION_SIZE == 0:
                unprotected_layer.CommitTransaction()
                unprotected_layer.StartTransaction()

            candidate_indexes = zone_index.query(
                unprotected,
                predicate="intersects",
            )
            if len(candidate_indexes) == 0:
                stats["outside_proximity_skipped"] += 1
                continue

            candidate_zone = shapely.union_all(
                [zone_parts[index] for index in candidate_indexes]
            )
            clipped = repair_polygonal_geometry(
                shapely.intersection(unprotected, candidate_zone)
            )
            if clipped is None or clipped.area <= 0:
                stats["zero_area_intersection_skipped"] += 1
                continue

            output_feature = ogr.Feature(output_layer.GetLayerDefn())
            for field_index in range(input_definition.GetFieldCount()):
                value = input_feature.GetField(field_index)
                if value is not None:
                    output_feature.SetField(field_index, value)
            if output_feature.GetFieldIndex("land_type") >= 0:
                output_feature.SetField("land_type", BBB_CANDIDATE_LAND_TYPE)
            output_feature.SetGeometry(ogr.CreateGeometryFromWkb(wkb.dumps(clipped)))
            output_layer.CreateFeature(output_feature)
            output_feature = None
            stats["retained"] += 1
            write_bar.update()
            scan_bar.set_postfix(
                final=stats["retained"],
                unprotected=stats["unprotected_retained"],
                distance_skipped=stats["outside_proximity_skipped"],
                protected_clipped=stats["federally_protected_clipped"],
                protected_skipped=stats[
                    "federally_protected_fully_excluded"
                ],
                refresh=False,
            )
            if stats["retained"] % TRANSACTION_SIZE == 0:
                output_layer.CommitTransaction()
                output_layer.StartTransaction()
        scan_bar.set_postfix(
            final=stats["retained"],
            unprotected=stats["unprotected_retained"],
            distance_skipped=stats["outside_proximity_skipped"],
            protected_clipped=stats["federally_protected_clipped"],
            protected_skipped=stats["federally_protected_fully_excluded"],
            refresh=True,
        )

    with tqdm(total=4, desc="Finalize output GeoPackages", unit="step") as final_bar:
        final_bar.set_postfix_str("Commit unprotected BLM features")
        unprotected_layer.CommitTransaction()
        final_bar.update()
        final_bar.set_postfix_str("Commit final BBB features")
        output_layer.CommitTransaction()
        final_bar.update()
        final_bar.set_postfix_str("Flush unprotected BLM dataset")
        unprotected_dataset.FlushCache()
        final_bar.update()
        final_bar.set_postfix_str("Flush final BBB dataset")
        output_dataset.FlushCache()
        final_bar.update()
    unprotected_layer = None
    unprotected_dataset = None
    output_layer = None
    output_dataset = None
    input_layer = None
    input_dataset = None

    return BbbFilterResult(
        unprotected_output_path=unprotected_output_path,
        output_path=output_path,
        input_features=input_count,
        bbb_candidate_features=bbb_candidate_count,
        unprotected_features=stats["unprotected_retained"],
        retained_features=stats["retained"],
        incorporated_places=incorporated_count,
        census_designated_places=cdp_count,
        federally_protected_features=stats[
            "federally_protected_source_features"
        ],
        stats=stats,
    )


def main() -> None:
    """Run the bill-specific candidate-land filter and print its summary."""
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    distance_label = f"{args.distance_miles:g}".replace(".", "p")
    unprotected_output_path = args.unprotected_output or (
        DEFAULT_UNPROTECTED_OUTPUT_DIR
        / (
            "blm_lands_excluding_federally_protected_areas_"
            f"{timestamp}.gpkg"
        )
    )
    output_path = args.output or (
        DEFAULT_OUTPUT_DIR
        / (
            "bbb_candidate_blm_lands_within_"
            f"{distance_label}_miles_of_population_centers_"
            f"{timestamp}.gpkg"
        )
    )
    result = filter_bbb_padus_by_population_centers(
        input_path=args.input_gpkg,
        input_layer_name=args.input_layer,
        population_centers_path=args.population_centers_gpkg,
        incorporated_layer=args.incorporated_layer,
        cdp_layer=args.cdp_layer,
        distance_miles=args.distance_miles,
        unprotected_output_path=unprotected_output_path,
        unprotected_output_layer_name=args.unprotected_output_layer,
        output_path=output_path,
        output_layer_name=args.output_layer,
    )
    print(
        "Population centers: "
        f"incorporated={result.incorporated_places:,} "
        f"cdp={result.census_designated_places:,}"
    )
    print(
        "BBB attribute filter: "
        f"candidates={result.bbb_candidate_features:,} "
        f"excluded={result.input_features - result.bbb_candidate_features:,}"
    )
    print(
        "Federally protected PAD-US records: "
        f"source={result.federally_protected_features:,} "
        f"clipped={result.stats['federally_protected_clipped']:,} "
        "fully_excluded="
        f"{result.stats['federally_protected_fully_excluded']:,}"
    )
    print(
        "Processing stats: "
        + " ".join(f"{key}={value:,}" for key, value in result.stats.items())
    )
    print(
        f"Wrote {result.unprotected_features:,} BLM feature(s) after "
        "federally protected areas were subtracted: "
        f"{result.unprotected_output_path}"
    )
    print(
        f"Wrote {result.retained_features:,} of "
        f"{result.unprotected_features:,} unprotected BLM feature(s) inside "
        "the population-center proximity zones: "
        f"{result.output_path}"
    )


if __name__ == "__main__":
    main()
