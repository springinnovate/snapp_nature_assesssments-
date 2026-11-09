"""Test reporting-water selection and mask-only artifact publication."""

import argparse
import configparser
import io
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import box
import yaml

import prepare_county_reporting_areas as builder
from nlcd_mask_areas import CLASSES, FIELD_DEFINITIONS, OPERATIONS
from prepare_tif_zonal_stats_config import RasterInput


@pytest.fixture
def rules():
    """Provide reviewed selection rules without depending on production data."""
    return {"schema_version": 1, "census_vintage": 2024, "source_url": "https://example.invalid/",
            "ocean_mtfcc": "H2053", "connected_bay_mtfcc": "H2051",
            "great_lakes_names": {name: [f"Lk {name}", f"Lake {name}"]
                                  for name in ("Superior", "Michigan", "Huron", "Erie", "Ontario")}}


def test_bays_connect_across_counties_but_inland_bays_remain(rules):
    """Follow a bay chain globally, not merely within each county file."""
    water = gpd.GeoDataFrame({
        "MTFCC": ["H2053", "H2051", "H2051", "H2051", "H2030"],
        "great_lake": [None, None, None, None, "Superior"],
        "county_geoid": ["01001", "01001", "01003", "01003", "26083"],
    }, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1),
                 box(10, 0, 11, 1), box(20, 0, 21, 1)], crs=builder.NATIVE_CRS)
    selected = builder.select_connected_water(water, rules, require_all_lakes=False)
    assert selected.index.tolist() == [0, 1, 2, 4]
    assert selected.selection_reason.tolist() == ["ocean", "connected_bay", "connected_bay", "great_lake"]
    with pytest.raises(ValueError, match="missing Great Lakes"):
        builder.select_connected_water(water, rules, require_all_lakes=True)


def test_exclusion_union_counts_overlap_once_and_preserves_land():
    """Use unioned water area, leaving protected Census land and inland counties."""
    counties = gpd.GeoDataFrame({"GEOID": ["12099", "49011", "06061"], "ALAND": [5000] * 3,
                                "AWATER": [5000] * 3},
                               geometry=[box(x, 0, x + 100, 100) for x in (0, 200, 400)], crs=builder.AREA_CRS)
    land = gpd.GeoDataFrame({"GEOID": counties.GEOID},
                           geometry=[box(x, 0, x + 50, 100) for x in (0, 200, 400)], crs=builder.AREA_CRS).set_index("GEOID")
    excluded = gpd.GeoDataFrame(geometry=[box(40, 0, 80, 100), box(60, 0, 90, 100)], crs=builder.AREA_CRS)
    result, qa = builder.build_reporting_geometries(counties, land, excluded, workers=2)
    np.testing.assert_allclose(result[builder.AREA_FIELD], [0.6, 1., 1.])
    np.testing.assert_allclose(qa.area_ha_excluded_water, [0.4, 0., 0.])
    assert (qa.area_balance_error_m2 < 1e-8).all()
    for row in result.itertuples():
        assert row.geometry.covers(land.loc[row.GEOID].geometry)


@pytest.fixture
def preparation_inputs(tmp_path, monkeypatch, rules):
    """Create two inland counties and cached water files, with no network use."""
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    shapes = [box(-112, 40, -111.99, 40.01), box(-120, 39, -119.99, 39.01)]
    land_shapes = [box(-112, 40, -111.995, 40.01), box(-120, 39, -119.995, 39.01)]
    frame = gpd.GeoDataFrame({"GEOID": ["49011", "06061"]}, geometry=shapes, crs=builder.NATIVE_CRS)
    land = gpd.GeoDataFrame({"GEOID": frame.GEOID}, geometry=land_shapes, crs=builder.NATIVE_CRS)
    projected = frame.to_crs(builder.AREA_CRS)
    frame["ALAND"] = np.round(land.to_crs(builder.AREA_CRS).area).astype("int64")
    frame["AWATER"] = np.round(projected.area - frame.ALAND).astype("int64")
    counties_path, land_path = tmp_path / "counties.gpkg", tmp_path / "land.gpkg"
    pyogrio.write_dataframe(frame, counties_path, layer="counties")
    pyogrio.write_dataframe(land, land_path, layer="county_land_mask")
    archive_dir = tmp_path / "water"
    archive_dir.mkdir()
    for geoid, shape, name in zip(frame.GEOID, shapes, ["Great Salt Lk", "Lk Tahoe"]):
        stem = f"tl_2024_{geoid}_areawater"
        shape_dir = tmp_path / stem
        shape_dir.mkdir()
        water = gpd.GeoDataFrame({"HYDROID": [geoid], "FULLNAME": [name], "MTFCC": ["H2030"],
                                 "ALAND": [0], "AWATER": [1]}, geometry=[shape], crs=builder.NATIVE_CRS)
        pyogrio.write_dataframe(water, shape_dir / f"{stem}.shp")
        with ZipFile(archive_dir / f"{stem}.zip", "w") as archive:
            for file in shape_dir.iterdir():
                archive.write(file, file.name)
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(yaml.safe_dump(rules), encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("# test runner\n", encoding="utf-8")
    args = argparse.Namespace(counties=counties_path, counties_layer="counties", land_mask=land_path,
                              land_mask_layer="county_land_mask", areawater_dir=archive_dir, download_missing=False,
                              rules=rules_path, mask_config=builder.DEFAULT_MASK_CONFIG, runner=runner,
                              output=tmp_path / "reporting.gpkg", geometry_only=True,
                              geoids=["49011", "06061"], workers=2, raster_workers=2, run_id="test-reporting")
    return args


def test_geometry_only_preserves_salt_lake_and_tahoe(preparation_inputs):
    """Keep both non-Great-Lakes water bodies and label partial geometry-only outputs."""
    args = preparation_inputs
    path = builder.prepare_reporting_areas(args)
    assert path.name == "reporting_geometry.gpkg"
    frame = pyogrio.read_dataframe(path, layer=builder.LAYER).set_index("GEOID")
    assert set(FIELD_DEFINITIONS).isdisjoint(frame.columns)
    qa = pd.read_csv(path.with_name(path.stem + "_qa.csv"), dtype={"GEOID": str}).set_index("GEOID")
    assert (qa.area_ha_excluded_water == 0).all()
    np.testing.assert_allclose(frame[builder.AREA_FIELD].sort_index(), qa.area_ha_full_county.sort_index())
    manifest = json.loads(path.with_name(path.stem + "_manifest.json").read_text())
    assert manifest["scope"] == "subset_verification"
    assert manifest["status"] == "geometry_only"
    assert len(manifest["water_archives"]) == 2
    assert all(len(row["sha256"]) == 64 for row in manifest["water_archives"])
    with pytest.raises(FileExistsError):
        builder.prepare_reporting_areas(args)


def test_complete_artifact_runs_only_mask_job(preparation_inputs, monkeypatch):
    """Publish twelve corrected mask fields with immutable geometry and provenance."""
    args = preparation_inputs
    args.geometry_only = False
    args.run_id = "test-complete"
    metadata, rasters = [], []
    for name in CLASSES:
        path = args.output.parent / f"reclassified_NLCD2023_{name}.tif"
        path.write_bytes(b"fixture")
        metadata.append({"class": name, "path": str(path), "bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns, "pixel_area_ha": .09})
        rasters.append(RasterInput(name, path, "binary", path, 1.))
    monkeypatch.setattr(builder, "validate_mask_definition", lambda _: metadata)
    monkeypatch.setattr(builder, "load_and_prepare_rasters", lambda *_, **__: (rasters, []))
    calls = []

    def fake_runner(command, check):
        """Write small, internally consistent binary-mask stats for the real exporter."""
        calls.append(command)
        assert check
        config_path = Path(command[3])
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path, encoding="utf-8")
        assert config.sections() == ["project", "job:counties_masks"]
        job = config["job:counties_masks"]
        vector = (config_path.parent / job["agg_vector"]).resolve()
        frame = pyogrio.read_dataframe(vector, layer=builder.LAYER)
        raw_path = (config_path.parent / job["output_gpkg"]).resolve()
        raw_path = raw_path.with_name(raw_path.stem + "_test.gpkg")
        raw_path.parent.mkdir(parents=True)
        values = {"sum": 1., "valid_count": 10., "total_count": 10.,
                  "area_ha_valid": .9, "area_ha_total": .9, "min": 0., "max": 1.}
        for name in CLASSES:
            for operation in OPERATIONS:
                frame[f"{operation}_reclassified_NLCD2023_{name}"] = values[operation]
        pyogrio.write_dataframe(frame, raw_path, layer=builder.LAYER)
        frame.drop(columns="geometry").to_csv(raw_path.with_suffix(".csv"), index=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(builder.subprocess, "run", fake_runner)
    path = builder.prepare_reporting_areas(args)
    assert len(calls) == 1
    frame = pyogrio.read_dataframe(path, layer=builder.LAYER)
    assert set(FIELD_DEFINITIONS) <= set(frame)
    assert builder.AREA_FIELD in frame
    np.testing.assert_allclose(frame[builder.AREA_FIELD], frame.geometry.area / 10000)
    attrs = builder.read_attributes(path)[0].set_index("GEOID").sort_index()
    csv = pd.read_csv(path.with_suffix(".csv"), dtype={"GEOID": str}, float_precision="round_trip").set_index("GEOID").sort_index()
    pd.testing.assert_frame_equal(attrs, csv, check_dtype=False)
    assert not any(column.startswith("sum_reclassified") for column in frame)
    np.testing.assert_allclose(frame.area_ha_all_natural, .45)
    np.testing.assert_allclose(frame.area_ha_valid_nlcd, .9)
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM county_reporting_area_qa").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM county_reporting_water_selection").fetchone()[0] == 0
        embedded = json.loads(connection.execute(
            "SELECT manifest_json FROM county_reporting_area_run").fetchone()[0])
        assert embedded["rules"]["census_vintage"] == 2024
        assert embedded["source_hashes"]["rules"] == builder.sha256(args.rules)
    manifest = json.loads(path.with_name(path.stem + "_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["mask_config_sha256"] == builder.sha256(Path(manifest["mask_config"]))


def test_missing_archive_does_not_download_without_flag(tmp_path, rules):
    """Fail closed when a required Census archive was not supplied."""
    with pytest.raises(FileNotFoundError, match="download-missing"):
        builder.read_county_water("12099", tmp_path / "missing.zip", rules, False)


def test_subset_cannot_replace_full_default(preparation_inputs):
    """Prevent a small QA subset from becoming the configured nationwide artifact."""
    preparation_inputs.output = builder.DEFAULT_OUTPUT
    with pytest.raises(ValueError, match="Subset verification"):
        builder.prepare_reporting_areas(preparation_inputs)


def test_download_retries_rejected_and_wrong_archives_before_caching(preparation_inputs, monkeypatch):
    """Never preserve an HTTP-200 rejection page or an unrelated valid ZIP."""
    args = preparation_inputs
    original = args.areawater_dir / "tl_2024_49011_areawater.zip"
    target = args.output.parent / "downloaded" / original.name
    wrong = io.BytesIO()
    with ZipFile(wrong, "w") as zipped:
        zipped.writestr("wrong.txt", "not a water Shapefile")
    responses = iter([b"<html>The requested URL was rejected</html>", wrong.getvalue(), original.read_bytes()])
    calls = []

    def response(url, timeout):
        """Serve successive fake network responses without touching cached files."""
        assert not target.exists()
        calls.append(url)
        return io.BytesIO(next(responses))

    monkeypatch.setattr(builder, "urlopen", response)
    monkeypatch.setattr(builder, "sleep", lambda _: None)
    rules = yaml.safe_load(args.rules.read_text())
    _, record = builder.read_county_water("49011", target, rules, True)
    assert len(calls) == 3
    assert target.read_bytes() == original.read_bytes()
    assert record["sha256"] == builder.sha256(original)


def test_rejected_download_never_leaves_a_cached_archive(tmp_path, rules, monkeypatch):
    """Abort after a bounded number of bad responses without poisoning the cache."""
    target = tmp_path / "water" / "tl_2024_12099_areawater.zip"
    calls = []

    def response(url, timeout):
        """Return an HTTP-200 HTML error body for every attempted download."""
        calls.append(url)
        return io.BytesIO(b"<html>not a ZIP</html>")

    monkeypatch.setattr(builder, "urlopen", response)
    monkeypatch.setattr(builder, "sleep", lambda _: None)
    with pytest.raises(Exception, match="not a zip"):
        builder.read_county_water("12099", target, rules, True)
    assert len(calls) == 3
    assert not target.exists()


def test_zero_awater_county_requires_no_archive(tmp_path, rules, monkeypatch):
    """Retain no-water counties without requiring a nonexistent water download."""
    monkeypatch.setattr(builder, "urlopen", lambda *_args, **_kwargs: pytest.fail("Unexpected download"))
    frame, record = builder.read_county_water("20071", tmp_path / "missing.zip", rules, True, expected_water_m2=0)
    assert frame.empty
    assert record["reason"] == "County AWATER is zero"
    assert record["path"] is None
