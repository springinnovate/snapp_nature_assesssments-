"""Test reporting-water selection and mask-only artifact publication."""

import argparse
import configparser
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, box

import prepare_county_reporting_areas as builder
import update_nlcd_mask_areas as masks
from nlcd_mask_areas import CLASSES, FIELD_DEFINITIONS, OPERATIONS
from prepare_tif_zonal_stats_config import RasterInput


def test_ocean_follows_shoreline_connected_water_across_counties():
    """Keep disconnected inland border water; a single point does not connect it."""
    counties = gpd.GeoDataFrame({
        "GEOID": ["01001", "01003", "27077", "12011"],
        "AWATER": [6_000_000] * 4,
    }, geometry=[*[box(0, y, 2500, y + 3000) for y in (0, 3000, 10000)],
                 box(2500, 6000, 5000, 9000)], crs=builder.AREA_CRS)
    land = gpd.GeoDataFrame({"GEOID": counties.GEOID},
                           geometry=[*[box(0, y, 500, y + 3000) for y in (0, 3000, 10000)],
                                     box(4500, 6000, 5000, 9000)],
                           crs=builder.AREA_CRS).set_index("GEOID")
    usa = box(0, 0, 300, 13000)
    coast = shapely.STRtree([LineString([(500, 0), (500, 1000)])])
    selected = builder.select_ocean_water(counties, land, usa, coast, workers=2)
    assert shapely.union_all(selected).equals(box(500, 0, 2500, 6000))


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
def preparation_inputs(tmp_path, monkeypatch):
    """Create inland, Great Lakes and coastal counties with local vector sources."""
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    locations = [(0, 0), (10000, 0), (20000, 0), (30000, 0)]
    shapes = [box(x, y, x + 100, y + 100) for x, y in locations]
    land_shapes = [box(x, y, x + 50, y + 100) for x, y in locations]
    frame = gpd.GeoDataFrame({"GEOID": ["49011", "06061", "26083", "12099"], "ALAND": [5000] * 4,
                              "AWATER": [5000] * 4}, geometry=shapes, crs=builder.AREA_CRS)
    land = gpd.GeoDataFrame({"GEOID": frame.GEOID}, geometry=land_shapes, crs=builder.AREA_CRS)
    counties_path, land_path = tmp_path / "counties.gpkg", tmp_path / "land.gpkg"
    pyogrio.write_dataframe(frame, counties_path, layer="counties")
    pyogrio.write_dataframe(land, land_path, layer="county_land_mask")
    data = tmp_path / "data"
    data.mkdir()
    usa, coast = data / "usa.gpkg", data / "coast.gpkg"
    water, crosswalk = data / "water.gpkg", data / "lakes.csv"
    pyogrio.write_dataframe(gpd.GeoDataFrame(
        geometry=[MultiPolygon([box(-1000, -1000, 40, 1000), box(1000, -1000, 30060, 1000)])],
        crs=builder.AREA_CRS), usa, layer="usa")
    pyogrio.write_dataframe(gpd.GeoDataFrame(
        {"NAME": ["Pacific", "Great Lakes"]},
        geometry=[LineString([(30070, 0), (30070, 100)]), LineString([(80, 0), (80, 100)])],
        crs=builder.AREA_CRS), coast, layer="coast")
    pyogrio.write_dataframe(gpd.GeoDataFrame(
        {"source_layer": ["NHDWaterbody"] * 7, "source_fid": range(1, 8)},
        geometry=[box(20040, 0, 20090, 100), *[box(x, 0, x + 25, 100) for x in range(22000, 24000, 500)],
                  box(50, 0, 100, 100), box(10050, 0, 10100, 100)],
        crs=builder.AREA_CRS), water, layer="water")
    pd.DataFrame({"lake": [f"Lake {name}" for name in ("Superior", "Michigan", "Huron", "Erie", "Ontario")],
                  "source_fid": range(1, 6)}).to_csv(crosswalk, index=False)
    sources = data / "sources.csv"
    pd.DataFrame({"source": ["coastline_lines", "freshwater_polygons", "great_lakes_features"],
                  "path": [coast.name, water.name, crosswalk]}).to_csv(sources, index=False)
    runner = tmp_path / "runner.py"
    runner.write_text("# test runner\n", encoding="utf-8")
    args = argparse.Namespace(counties=counties_path, counties_layer="counties", land_mask=land_path,
                              land_mask_layer="county_land_mask", usa_boundary=usa, usa_boundary_layer=None,
                              sources=sources,
                              rules=builder.DEFAULT_RULES, mask_config=builder.DEFAULT_MASK_CONFIG,
                              runner=runner,
                              output=tmp_path / "reporting.gpkg", geometry_only=True,
                              geoids=["49011", "06061", "26083", "12099"], workers=2, raster_workers=2,
                              run_id="test-reporting")
    return args


def test_geometry_only_keeps_inland_water_but_excludes_ocean_and_great_lakes(preparation_inputs):
    """Protect land and subtract only the union of selected water polygons."""
    args = preparation_inputs
    path = builder.prepare_reporting_areas(args)
    assert path.name == "reporting_geometry.gpkg"
    frame = pyogrio.read_dataframe(path, layer=builder.LAYER).set_index("GEOID")
    assert set(FIELD_DEFINITIONS).isdisjoint(frame.columns)
    qa = pd.read_csv(path.with_name(path.stem + "_qa.csv"), dtype={"GEOID": str}).set_index("GEOID")
    np.testing.assert_allclose(qa.loc[args.geoids, "area_ha_excluded_water"], [0, 0, .4, .4])
    np.testing.assert_allclose(frame.loc[args.geoids, builder.AREA_FIELD], [1, 1, .6, .6])
    manifest = json.loads(path.with_name(path.stem + "_manifest.json").read_text())
    assert manifest["scope"] == "subset_verification"
    assert manifest["status"] == "geometry_only"
    assert "water_archives" not in manifest
    assert Path(manifest["source_paths"]["usa_boundary"]) == args.usa_boundary
    assert len(manifest["selected_water_sha256"]) == 64
    for name in ("rules", "usa_boundary", "coastline_lines", "freshwater_polygons", "great_lakes_features"):
        assert manifest["source_hashes"][name] == builder.sha256(Path(manifest["source_paths"][name]))
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
    monkeypatch.setattr(masks, "validate_mask_definition", lambda _: metadata)
    monkeypatch.setattr(masks, "load_and_prepare_rasters", lambda *_, **__: (rasters, []))
    calls = []

    def fake_runner(command, check):
        """Write small, internally consistent binary-mask stats for the real exporter."""
        calls.append(command)
        assert check
        assert command[1:3] == ["-B", str(args.runner.resolve())]
        assert command[4:] == ["--job-workers", "1", "--raster-workers", str(args.raster_workers)]
        config_path = Path(command[3])
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path, encoding="utf-8")
        assert config.sections() == ["project", "job:counties_masks"]
        job = config["job:counties_masks"]
        assert job["agg_layer"] == builder.LAYER
        assert job["agg_field"] == "GEOID"
        assert set(job["operations"].replace(" ", "").split(",")) == set(OPERATIONS)
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

    monkeypatch.setattr(masks.subprocess, "run", fake_runner)
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
        assert connection.execute("SELECT COUNT(*) FROM county_reporting_area_qa").fetchone()[0] == 4
        selection = connection.execute("SELECT selection_reason FROM county_reporting_water_selection").fetchall()
        assert sorted(selection) == [("great_lakes",), ("shoreline_connected_ocean",)]
        embedded = json.loads(connection.execute(
            "SELECT manifest_json FROM county_reporting_area_run").fetchone()[0])
        assert Path(embedded["source_paths"]["usa_boundary"]) == args.usa_boundary
    manifest = json.loads(path.with_name(path.stem + "_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["raster_metadata"] == metadata
    assert manifest["mask_config_sha256"] == builder.sha256(Path(manifest["mask_config"]))


def test_subset_cannot_replace_full_default(preparation_inputs):
    """Prevent a small QA subset from becoming the configured nationwide artifact."""
    preparation_inputs.output = builder.DEFAULT_OUTPUT
    with pytest.raises(ValueError, match="Subset verification"):
        builder.prepare_reporting_areas(preparation_inputs)


def test_missing_local_lake_feature_fails_without_publishing(preparation_inputs):
    """Do not silently produce reporting areas when a listed lake feature is absent."""
    args = preparation_inputs
    crosswalk = args.sources.parent / "lakes.csv"
    lakes = pd.read_csv(crosswalk)
    lakes.loc[0, "source_fid"] = 999
    lakes.to_csv(crosswalk, index=False)
    with pytest.raises(ValueError, match="Missing Great Lakes features"):
        builder.prepare_reporting_areas(args)
    assert not args.output.with_name("reporting_geometry.gpkg").exists()
