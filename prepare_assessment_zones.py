"""Prepare six assessment zone GeoPackages and validate county land areas."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from tqdm import tqdm

from assessment_zone_geometry import ZONE_PRODUCTS, build_county_zones
from geometry_utils import repair_polygonal_geometry
from padus_land_rules import load_rule_config


ROOT = Path(__file__).resolve().parent
AREA_CRS = "EPSG:6933"
CONSERVATION_GAP_CODES = (1, 2, 3)
COUNTY_AREA_FIELDS = (
    "area_ha_county_land",
    "area_ha_public_lands",
    "pct_public_lands",
    "area_ha_conservation_lands",
    "pct_conservation_lands",
    "area_ha_bbb_lands",
    "pct_bbb_lands",
    "area_ha_public_manager_overlap",
    "area_ha_conservation_manager_overlap",
)


def read_polygons(
    path: Path, layer: str | None, required: set[str], *, attributes: bool = False
) -> tuple[gpd.GeoDataFrame, str]:
    """Read one unambiguous polygon layer and check its required fields.

    Args:
        path: Source vector dataset.
        layer: Explicit layer, or None for a single spatial layer.
        required: Required attribute names.
        attributes: Whether to retain all source attributes.

    Returns:
        The source frame and resolved layer name.

    Raises:
        ValueError: A layer is ambiguous, missing data, or not polygonal.
    """
    polygon_types = ("Polygon", "MultiPolygon")
    layers = {
        name: geom_type for name, geom_type in pyogrio.list_layers(path)
        if geom_type is not None
    }
    if layer is None and len(layers) == 1:
        layer = next(iter(layers))
    if layer is None or layer not in layers:
        raise ValueError(f"Select a spatial layer in {path}; choices: {', '.join(layers)}")
    info = pyogrio.read_info(path, layer=layer)
    missing = required.difference(info["fields"])
    if missing:
        raise ValueError(f"{path} / {layer} is missing fields: {', '.join(sorted(missing))}")
    frame = pyogrio.read_dataframe(path, layer=layer, columns=None if attributes else sorted(required))
    if frame.empty or frame.crs is None:
        raise ValueError(f"{path} / {layer} must contain polygons with a defined CRS")
    if not frame.geom_type.isin(polygon_types).all() or frame.geometry.is_empty.any():
        raise ValueError(f"{path} / {layer} contains null, empty, or nonpolygonal geometry")
    return frame, layer


def project_polygons(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Project polygons for equal-area overlay and repair invalid features.

    Args:
        frame: Nonempty or classified-empty polygon frame with a CRS.
        label: Dataset label used in validation messages.

    Returns:
        Polygons projected to the common area CRS.

    Raises:
        ValueError: Some features cannot be repaired after projection.
    """
    frame = frame.to_crs(AREA_CRS)
    invalid = ~frame.geometry.is_valid | (frame.geometry.area <= 0)
    if invalid.any():
        repaired = frame.geometry.loc[invalid].map(repair_polygonal_geometry)
        if repaired.isna().any():
            raise ValueError(f"{label} contains {repaired.isna().sum()} unrecoverable polygons")
        frame.loc[invalid, frame.geometry.name] = repaired
    return frame


def prepare_zones(args: argparse.Namespace) -> Path:
    """Build, check, and write the zone products for a new run.

    Args:
        args: Input paths, optional layers, run ID, and worker limits.

    Returns:
        The directory containing zone GeoPackages and area QA.

    Raises:
        ValueError: Inputs or calculated areas fail validation.
        FileExistsError: This run's zone directory already exists.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("Run ID must be 1–64 letters, digits, underscores or hyphens")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output = args.output_root.resolve() / args.run_id / "zonal_units"
    if output.exists():
        raise FileExistsError(f"Zone output already exists: {output}")

    rules_text, rules = load_rule_config(args.rules)
    public_rule = rules.select(frozenset({"public_land"}))
    rule_fields = public_rule.referenced_fields()
    if "manager" in rule_fields:
        raise ValueError("The public_land rule for zone preparation must use physical PAD-US fields")
    sources = {
        "counties": (args.counties, args.counties_layer, {"GEOID", "ALAND"}),
        "padus": (args.padus_path, args.padus_layer, {"Mang_Type", "GAP_Sts", *rule_fields}),
        "bbb_by_county": (args.bbb_by_county, args.bbb_layer, {"GEOID"}),
    }
    frames = {}
    provenance = {}
    for label, (path, layer, required) in tqdm(sources.items(), desc="Read zone inputs", unit="layer"):
        frame, resolved_layer = read_polygons(path, layer, required, attributes=label == "counties")
        provenance[label] = {"path": str(path.resolve()), "layer": resolved_layer, "rows": len(frame)}
        frames[label] = frame

    for label in ("counties", "bbb_by_county"):
        ids = frames[label]["GEOID"].astype("string")
        if not ids.str.fullmatch(r"[0-9]{5}").fillna(False).all() or ids.duplicated().any():
            raise ValueError(f"{label} must have unique five-digit GEOID strings")
        frames[label]["GEOID"] = ids

    county_input = frames["counties"]
    aland = pd.to_numeric(county_input["ALAND"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(aland).all() or (aland <= 0).any():
        raise ValueError("Counties must have finite positive Census ALAND values in square metres")
    unknown_bbb = set(frames["bbb_by_county"]["GEOID"]).difference(county_input["GEOID"])
    if unknown_bbb:
        raise ValueError(f"BBB includes GEOIDs not found in counties: {', '.join(sorted(unknown_bbb)[:10])}")
    if set(COUNTY_AREA_FIELDS).intersection(county_input.columns):
        raise ValueError("Use the original county boundary layer, not previously prepared assessment zones")

    padus = frames["padus"]
    # The public rule only depends on attributes: evaluate distinct rule-field
    # combinations, then map results back rather than parsing a million rows.
    rule_columns = sorted(rule_fields)
    attributes = padus[rule_columns].fillna("")
    combinations = {
        values: public_rule.evaluate(dict(zip(rule_columns, values)))["public_land"]
        for values in attributes.drop_duplicates().itertuples(index=False, name=None)
    }
    is_public = np.fromiter(
        (combinations[values] for values in attributes.itertuples(index=False, name=None)),
        dtype=bool, count=len(padus),
    )
    is_conservation = pd.to_numeric(padus["GAP_Sts"], errors="coerce").isin(CONSERVATION_GAP_CODES).to_numpy()
    selected = is_public | is_conservation
    frames["padus"] = padus.loc[selected, ["Mang_Type", padus.geometry.name]].copy()
    manager = frames["padus"]["Mang_Type"].astype("string").str.strip().replace("", pd.NA).fillna("UNSPECIFIED")
    managers = manager.to_numpy()
    public_flags = is_public[selected]
    conservation_flags = is_conservation[selected]
    del padus, attributes

    for label in tqdm(frames, desc="Project zone inputs", unit="layer"):
        frames[label] = project_polygons(frames[label], label)
    counties = frames["counties"]
    padus_shapes = frames["padus"].geometry.to_numpy()
    padus_index = frames["padus"].sindex
    bbb_shapes = dict(zip(frames["bbb_by_county"]["GEOID"], frames["bbb_by_county"].geometry))
    county_shapes = counties.geometry.to_numpy()
    county_ids = counties["GEOID"].to_numpy()
    county_attributes = counties.drop(columns=counties.geometry.name).to_dict("records")

    def process_county(index: int) -> dict:
        """Select candidates and build zone rows for the indexed county."""
        county = county_shapes[index]
        geoid = county_ids[index]
        candidates = (
            (padus_shapes[i], managers[i], public_flags[i], conservation_flags[i])
            for i in padus_index.query(county)
        )
        return build_county_zones(
            geoid, county, aland[index], candidates,
            [bbb_shapes[geoid]] if geoid in bbb_shapes else [],
        )

    rows = {product: [] for product in ZONE_PRODUCTS}
    area_qa = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, zones in enumerate(tqdm(
            pool.map(process_county, range(len(counties))),
            total=len(counties), desc="Build and check county zones", unit="county",
        )):
            county = zones["counties"][0]
            area_qa.append({
                "GEOID": county["GEOID"],
                "area_ha_county_polygon": county_shapes[index].area / 10_000,
                **{field: county[field] for field in COUNTY_AREA_FIELDS},
            })
            zones["counties"][0] = county_attributes[index] | county
            for product, features in zones.items():
                rows[product].extend(features)

    output.mkdir(parents=True)
    manifest = {
        "run_id": args.run_id,
        "area_crs": AREA_CRS,
        "inputs": provenance,
        "public_rule": {
            "path": str(args.rules.resolve()),
            "text": rules_text,
            "text_sha256": hashlib.sha256(rules_text.encode("utf-8")).hexdigest(),
        },
        "conservation_gap_codes": CONSERVATION_GAP_CODES,
        "land_area_method": {
            "denominator": "Census ALAND (square metres)",
            "numerator": "Unioned footprint area clipped to county in EPSG:6933",
            "footprints_assumed_land_only": True,
        },
        "county_area_qa": str(output / "county_area_qa.csv"),
        "outputs": {},
    }
    for product, features in tqdm(rows.items(), desc="Write zone GeoPackages", unit="layer"):
        path = output / product / f"{product}_{args.run_id}.gpkg"
        path.parent.mkdir()
        if features:
            layer = gpd.GeoDataFrame(features, geometry="geometry", crs=AREA_CRS)
        else:
            columns = {"GEOID": pd.Series(dtype="str")}
            if product.endswith("_and_manager_type"):
                columns["Mang_Type"] = pd.Series(dtype="str")
            columns.update(area_ha_land=pd.Series(dtype="float64"), county_land_share=pd.Series(dtype="float64"))
            layer = gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries([], crs=AREA_CRS), crs=AREA_CRS)
        pyogrio.write_dataframe(layer, path, layer=product, geometry_type="MultiPolygon")
        manifest["outputs"][product] = {"path": str(path), "layer": product, "rows": len(layer)}

    pd.DataFrame(area_qa).to_csv(output / "county_area_qa.csv", index=False)
    overlap_fields = [field for field in COUNTY_AREA_FIELDS if field.endswith("_overlap")]
    overlap_counts = {field: sum(row[field] > 0 for row in area_qa) for field in overlap_fields}
    manifest["counties_with_manager_overlap"] = overlap_counts
    (output / "zone_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any(overlap_counts.values()):
        print("Manager zones overlap in some counties; do not sum them without reviewing county_area_qa.csv.")
    print(f"Prepared six zone layers and area QA: {output}")
    return output


def main(argv: list[str] | None = None) -> None:
    """Parse command-line inputs and prepare the six zone layers.

    Args:
        argv: Arguments to parse, or None to use the command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padus-path", type=Path, required=True)
    parser.add_argument("--padus-layer")
    parser.add_argument("--counties", type=Path, default=(
        ROOT / "data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg"
    ))
    parser.add_argument("--counties-layer")
    parser.add_argument("--bbb-by-county", type=Path, required=True, help="Existing BBB polygon zones with unique GEOIDs")
    parser.add_argument("--bbb-layer")
    parser.add_argument("--rules", type=Path, default=ROOT / "config/padus_land_rules.rules")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processing_outputs/assessment_runs")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent county geometry workers (default: 4)")
    prepare_zones(parser.parse_args(argv))


if __name__ == "__main__":
    main()
