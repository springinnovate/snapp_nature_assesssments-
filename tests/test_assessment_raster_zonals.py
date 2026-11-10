"""Check the six-zone raster plan without running a production analysis."""

import configparser
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import box

from assessment.assessment_raster_zonals import (
    freeze_zone_zonals,
    read_frozen_zone_raster_plan,
    read_zone_raster_sums,
    validate_zone_raster_inputs,
)
from assessment.assessment_zone_geometry import ZONE_PRODUCTS


PUBLIC_MANAGERS = "public_lands_by_county_and_manager_type"
CONSERVATION_MANAGERS = "conservation_lands_by_county_and_manager_type"
RESULT_STAMP = "20261106_123456"


@pytest.fixture
def raster_inputs(tmp_path):
    """Create an inventory with one configured scale and six synthetic zones.

    Args:
        tmp_path: Fresh test directory supplied by pytest.

    Returns:
        The concrete zone paths and the arguments needed to freeze the plan.
    """
    raster_dir = tmp_path / "source rasters"
    raster_dir.mkdir()
    for name in ("pollination", "flood"):
        (raster_dir / f"{name}.tif").write_bytes(b"source placeholder")
    reviewed = tmp_path / "reviewed.yaml"
    inventory = [
        {"label": "Pollination", "filename": "pollination.tif", "units": "USD/ha", "include": True,
         "scale": 0.09, "derived_filename": "pollination_usd_per_pixel.vrt"},
        {"label": "Flood", "filename": "flood.tif", "units": "USD/ha", "include": True},
        {"label": "Unavailable", "filename": "absent.tif", "units": "USD", "include": False,
         "reason": "Not a valuation TIFF"},
    ]
    config = configparser.ConfigParser(interpolation=None)
    config["project"] = {"raster_dir": raster_dir.name, "max_simplify_tolerance_meters": "3",
                         "raster_inventory": json.dumps(inventory)}
    config["job:old_template"] = {"zone_key": "unused"}
    with reviewed.open("w", encoding="utf-8") as stream:
        config.write(stream)
    zones = {}
    for product in ZONE_PRODUCTS:
        path = tmp_path / f"{product}.gpkg"
        if product in (PUBLIC_MANAGERS, CONSERVATION_MANAGERS):
            fields = {"GEOID": ["01001", "01001"], "Mang_Type": ["FED", "NA"]}
            geometries = [box(0, 0, 100, 100), box(50, 0, 150, 100)] if product == PUBLIC_MANAGERS else [
                box(0, 0, 100, 100), box(100, 0, 150, 100),
            ]
        elif product == "bbb_lands_by_county":
            fields, geometries = {"GEOID": pd.Series(dtype="string")}, []
        else:
            fields, geometries = {"GEOID": ["01001"]}, [box(0, 0, 150, 100)]
        frame = gpd.GeoDataFrame(fields, geometry=geometries, crs="EPSG:6933")
        pyogrio.write_dataframe(frame, path, layer=product, geometry_type="MultiPolygon", promote_to_multi=True)
        zones[product] = (path, product)
    return zones, reviewed, tmp_path / "frozen" / "raster_run.yaml", tmp_path / "work", tmp_path / "raw"


def write_pairs(plan):
    """Simulate paired, timestamped output files from the toolkit.

    Args:
        plan: A frozen run plan with known keys and columns.

    Returns:
        Mapping from untimestamped job output paths to the written CSV paths.
    """
    outputs = {}
    for job in plan.jobs:
        base = job.output_base.with_name(f"{job.output_base.name}_{RESULT_STAMP}")
        base.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(sorted(job.expected_keys), columns=job.key_fields)
        for index, column in enumerate(plan.raster_columns, start=1):
            frame[column] = float(index * 100)
        csv_path = base.with_suffix(".csv")
        frame.to_csv(csv_path, index=False)
        base.with_suffix(".gpkg").write_bytes(b"toolkit vector output placeholder")
        outputs[job.output_base] = csv_path
    return outputs


def test_freeze_source_pixels_and_split_only_overlapping_product(raster_inputs):
    """Preserve inventory provenance and keep overlapping managers independent."""
    zones, reviewed, frozen, work, raw = raster_inputs
    source_bytes = reviewed.read_bytes()
    inputs, excluded = validate_zone_raster_inputs(reviewed)
    assert all(raster.path == raster.source_path and raster.scale == 1.0 for raster in inputs)
    assert not frozen.exists() and not work.exists()
    plan = freeze_zone_zonals(*raster_inputs, overlapping_manager_products={PUBLIC_MANAGERS})
    assert plan.frozen_config == frozen
    assert plan.raster_columns == ("sum_pollination", "sum_flood")
    assert plan.blocked_rasters == tuple(excluded) == ("Unavailable: Not a valuation TIFF",)
    assert plan.skipped_products == ("bbb_lands_by_county",)
    assert len(plan.jobs) == 6
    assert reviewed.read_bytes() == source_bytes
    assert not list(reviewed.parent.rglob("*.vrt"))

    config = configparser.ConfigParser(interpolation=None)
    config.read(frozen, encoding="utf-8")
    assert config["project"]["snapp_source_config_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    inventory = json.loads(config["project"]["snapp_source_definition"])["project"]["raster_inventory"]
    assert inventory[0]["scale"] == 0.09
    assert inventory[2]["reason"] == "Not a valuation TIFF"
    assert config["project"]["max_simplify_tolerance_meters"] == "3"
    assert json.loads(config["project"]["snapp_unavailable_zones"]) == ["bbb_lands_by_county: empty zone layer"]
    for item in json.loads(config["project"]["snapp_rasters"]):
        assert item["scale"] == 1.0
        assert item["source_path"] == item["analysis_path"]

    for job in plan.jobs:
        section = config[f"job:{job.output_base.name}"]
        assert section["agg_field"] == ", ".join(job.key_fields)
        assert section["operations"] == "sum"
        assert (frozen.parent / section["output_csv"]).resolve() == job.output_base.with_suffix(".csv")
        assert job.output_base.is_relative_to(raw)
        assert {Path(path.strip()).name for path in section["base_raster_pattern"].split(",")} == {
            "flood.tif", "pollination.tif",
        }
        vector = (frozen.parent / section["agg_vector"]).resolve()
        if job.product != PUBLIC_MANAGERS:
            assert vector == zones[job.product][0]
        else:
            assert vector.is_relative_to(work)
            frame = gpd.read_file(vector, layer=section["agg_layer"])
            assert len(frame) == 1
            assert {tuple(row) for row in frame[["GEOID", "Mang_Type"]].to_numpy()} == job.expected_keys
            assert frame.geometry.iloc[0].area == 10_000


def test_collect_same_run_sums_and_preserve_composite_text_keys(raster_inputs):
    """Combine manager shards without dropping leading zeros or literal NA codes."""
    plan = freeze_zone_zonals(*raster_inputs)
    assert len(plan.jobs) == 7  # Both manager products are split by default.
    write_pairs(plan)
    sums = read_zone_raster_sums(plan)
    assert set(sums) == set(ZONE_PRODUCTS)
    assert sums["counties"].to_dict("records") == [{"GEOID": "01001", "sum_pollination": 100, "sum_flood": 200}]
    assert sums["bbb_lands_by_county"].empty
    assert list(sums["bbb_lands_by_county"].columns) == ["GEOID", *plan.raster_columns]
    assert sums[PUBLIC_MANAGERS][["GEOID", "Mang_Type"]].to_records(index=False).tolist() == [
        ("01001", "FED"), ("01001", "NA"),
    ]


@pytest.mark.parametrize("defect", ["missing_pair", "different_stamp", "wrong_keys", "nonfinite_sum"])
def test_collect_rejects_incomplete_or_malformed_output(raster_inputs, defect):
    """Require a fresh matched pair and finite sums for every expected key."""
    plan = freeze_zone_zonals(*raster_inputs, overlapping_manager_products=set())
    outputs = write_pairs(plan)
    csv = outputs[plan.jobs[1].output_base]
    if defect == "missing_pair":
        csv.with_suffix(".gpkg").unlink()
        message = "matched toolkit CSV/GPKG pair"
    elif defect == "different_stamp":
        for old in (csv, csv.with_suffix(".gpkg")):
            old.rename(old.with_name(old.name.replace(RESULT_STAMP, "20261106_123457")))
        message = "different run timestamp"
    elif defect == "wrong_keys":
        csv.write_text(csv.read_text(encoding="utf-8").replace("01001", "01003"), encoding="utf-8")
        message = "output keys do not match"
    else:
        csv.write_text(csv.read_text(encoding="utf-8").replace("100.0", "inf"), encoding="utf-8")
        message = "non-finite raster sums"
    with pytest.raises(ValueError, match=message):
        read_zone_raster_sums(plan)


def test_missing_enabled_source_stops_before_writing(raster_inputs):
    """Missing approved TIFFs fail early, while reviewed exclusions are allowed."""
    _, reviewed, frozen, work, _ = raster_inputs
    (reviewed.parent / "source rasters" / "pollination.tif").unlink()
    with pytest.raises(FileNotFoundError, match="Included raster 'Pollination' is missing"):
        freeze_zone_zonals(*raster_inputs)
    assert not frozen.exists() and not work.exists()


def test_reuse_frozen_plan_without_current_raster_inputs(raster_inputs):
    """Read a mixed split/unsplit run even when the reviewed config and TIFFs are gone."""
    zones, reviewed, frozen, work, raw = raster_inputs
    old = freeze_zone_zonals(*raster_inputs, overlapping_manager_products={PUBLIC_MANAGERS})
    write_pairs(old)
    original_sums = read_zone_raster_sums(old)
    reviewed.unlink()
    for raster in (reviewed.parent / "source rasters").glob("*.tif"):
        raster.unlink()
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in reviewed.parent.rglob("*") if p.is_file()}

    for overlap in (None, {PUBLIC_MANAGERS}):
        plan = read_frozen_zone_raster_plan(
            zones, frozen, work, raw, expected_raster_columns=set(old.raster_columns),
            overlapping_manager_products=overlap,
        )
        assert plan == old
        reused_sums = read_zone_raster_sums(plan)
        for product in ZONE_PRODUCTS:
            pd.testing.assert_frame_equal(reused_sums[product], original_sums[product])
    assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in reviewed.parent.rglob("*") if p.is_file()}


def test_reuse_frozen_carbon_pixel_column_is_schema_only(raster_inputs):
    """Reuse the active sums from an older run without inspecting retired values."""
    zones, reviewed, frozen, work, raw = raster_inputs
    config = configparser.ConfigParser(interpolation=None)
    config.read(reviewed, encoding="utf-8")
    inventory = json.loads(config["project"]["raster_inventory"])
    retired = "sum_totalC_tCO2e_pixel_2020"
    retired_tiff = reviewed.parent / "source rasters" / f"{retired.removeprefix('sum_')}.tif"
    retired_tiff.write_bytes(b"source placeholder")
    inventory.append({"label": "Carbon per pixel", "filename": retired_tiff.name,
                      "units": "t CO2e per pixel", "include": True})
    config["project"]["raster_inventory"] = json.dumps(inventory)
    with reviewed.open("w", encoding="utf-8") as stream:
        config.write(stream)

    old_plan = freeze_zone_zonals(*raster_inputs, overlapping_manager_products={PUBLIC_MANAGERS})
    csvs = write_pairs(old_plan)
    original = read_zone_raster_sums(old_plan)
    required = set(old_plan.raster_columns) - {retired}
    retired_tiff.unlink()
    county_csv = csvs[raw / "counties" / "counties"]
    saved = pd.read_csv(county_csv, dtype={"GEOID": "string"}).assign(**{retired: "not a value"})
    saved.to_csv(county_csv, index=False)

    plan = read_frozen_zone_raster_plan(zones, frozen, work, raw, expected_raster_columns=required)
    assert set(plan.raster_columns) == required
    assert plan.retired_raster_columns == (retired,)
    reused = read_zone_raster_sums(plan)
    for product in ZONE_PRODUCTS:
        pd.testing.assert_frame_equal(reused[product], original[product].drop(columns=retired))

    with pytest.raises(ValueError, match="sum columns do not match"):
        read_frozen_zone_raster_plan(zones, frozen, work, raw, expected_raster_columns={"sum_flood"})
    saved.drop(columns=retired).to_csv(county_csv, index=False)
    with pytest.raises(ValueError, match="Unexpected key or raster sum columns"):
        read_zone_raster_sums(plan)


@pytest.mark.parametrize(("defect", "message"), [
    ("wrong_output", "does not use this run's vector and result paths"),
    ("missing_shard", "missing required"),
    ("wrong_manager_keys", "Saved manager shard keys do not match"),
    ("scaled_raster", "original, unscaled TIFF paths"),
    ("unreviewed_service", "sum columns do not match"),
])
def test_reuse_rejects_incompatible_saved_plan(raster_inputs, defect, message):
    """Prevent path substitution, incomplete manager coverage, and double scaling."""
    zones, _, frozen, work, raw = raster_inputs
    old = freeze_zone_zonals(*raster_inputs, overlapping_manager_products={PUBLIC_MANAGERS})
    expected = set(old.raster_columns)
    config = configparser.ConfigParser(interpolation=None)
    config.read(frozen, encoding="utf-8")
    shard = f"{PUBLIC_MANAGERS}__manager_01"
    if defect == "wrong_output":
        config["job:counties"]["output_csv"] = "some_other_run/counties.csv"
    elif defect == "missing_shard":
        config.remove_section(f"job:{shard}")
    elif defect == "wrong_manager_keys":
        vector = work / "manager_zone_shards" / f"{shard}.gpkg"
        frame = pyogrio.read_dataframe(vector, layer=PUBLIC_MANAGERS).assign(Mang_Type="wrong")
        vector.unlink()
        pyogrio.write_dataframe(frame, vector, layer=PUBLIC_MANAGERS)
    elif defect == "scaled_raster":
        entries = json.loads(config["project"]["snapp_rasters"])
        entries[0]["scale"] = 0.09
        config["project"]["snapp_rasters"] = json.dumps(entries)
    else:
        expected.add("sum_unreviewed")
    with frozen.open("w", encoding="utf-8") as stream:
        config.write(stream)
    with pytest.raises(ValueError, match=message):
        read_frozen_zone_raster_plan(
            zones, frozen, work, raw, expected_raster_columns=expected,
            overlapping_manager_products={PUBLIC_MANAGERS},
        )
