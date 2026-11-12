"""Verify fisheries measurement reuse and content-based invalidation."""

import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

import assessment.assessment_fisheries as fisheries


@pytest.fixture
def inputs(tmp_path):
    """Use real indexed GeoPackages and distinct manager keys in a tiny state."""
    from tests.test_assessment_fisheries import FootprintFisheriesTests as _FisheriesFixture

    fixture = _FisheriesFixture()
    fixture.setUp()
    for product, frame in fixture.zones.items():
        if product.endswith("_and_manager_type"):
            frame["Mang_Type"] = ["FED", "STAT", "LOC"][:len(frame)]
    try:
        yield fixture.zones, fixture.paths, tmp_path / "fisheries-cache"
    finally:
        fixture.doCleanups()


def _complete(plan):
    """Execute the same bounded county jobs inline, then validate allocations."""
    for key, function, args in plan.tasks():
        plan.accept(key, function(*args))
    return plan.finish()


def _assert_frames_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for product in expected:
        pd.testing.assert_frame_equal(actual[product], expected[product], check_exact=True)


def test_warm_cache_skips_spatial_reads_and_preserves_all_six_products(inputs, monkeypatch):
    """A hit reads saved weights rather than projecting or intersecting again."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    expected = _complete(cold)
    assert Path(cold.cache_info["path"]).is_file()

    def unexpected_spatial_work(*_, **__):
        pytest.fail("A fisheries weight cache hit must not perform spatial work")

    monkeypatch.setattr(fisheries, "_fisheries_context", unexpected_spatial_work)
    monkeypatch.setattr(fisheries, "read_dataframe", unexpected_spatial_work)
    monkeypatch.setattr(fisheries, "_county_weights", unexpected_spatial_work)
    monkeypatch.setattr(type(zones["counties"]), "to_crs", unexpected_spatial_work)
    warm = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert warm.cache_info["status"] == "hit"
    assert warm.cache_info["key"] == cold.cache_info["key"]
    assert warm.task_count == 0
    assert list(warm.tasks()) == []
    _assert_frames_equal(warm.finish(), expected)


def test_changed_monetary_totals_reuse_weights_but_recalculate_values(inputs):
    """Only spatial weights are cached; current state dollars set current rates."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    before = _complete(cold)
    for service, field in fisheries._FIELDS.items():
        frame = pd.read_csv(paths[service])
        frame[field] *= 2
        frame.to_csv(paths[service], index=False)
    warm = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert warm.cache_info["status"] == "hit"
    assert warm.cache_info["key"] == cold.cache_info["key"]
    after = warm.finish()
    for product in zones:
        pd.testing.assert_frame_equal(after[product], before[product] * 2, check_exact=True)
    np.testing.assert_allclose(after["counties"].marine_fisheries.sum(), 3200)
    np.testing.assert_allclose(after["counties"].inland_fisheries.sum(), 16000)


def test_output_attributes_and_dataframe_index_do_not_change_measurement_key(inputs):
    """Renamed output fields and pandas row labels do not change spatial weights."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    expected = _complete(cold)
    changed = {}
    for product, frame in zones.items():
        changed[product] = frame.assign(renamed_output_field=123.0)
        changed[product].index = pd.Index([f"row-{i}" for i in range(len(frame))], name="new_index")
    warm = fisheries.FisheriesPlan(changed, paths, cache_dir=directory)
    assert warm.cache_info["status"] == "hit"
    actual = warm.finish()
    for product in zones:
        expected[product].index = changed[product].index
    _assert_frames_equal(actual, expected)


@pytest.mark.parametrize("change", ["geometry", "manager", "county_key", "crs", "row_order"])
def test_zone_geometry_keys_crs_and_row_order_cannot_reuse_misaligned_weights(inputs, change):
    """Every measurement dependency participates in the cache identity."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    _complete(cold)
    changed = {product: frame.copy() for product, frame in zones.items()}
    product = "public_lands_by_county"
    if change == "geometry":
        changed[product].loc[0, "geometry"] = box(0, 0, 40, 40)
    elif change == "manager":
        changed["public_lands_by_county_and_manager_type"].loc[0, "Mang_Type"] = "DIST"
    elif change == "county_key":
        for frame in changed.values():
            frame.GEOID = frame.GEOID.replace({"26001": "26005"})
    elif change == "crs":
        changed[product] = changed[product].set_crs(3857, allow_override=True)
    else:
        changed[product] = changed[product].iloc[::-1].reset_index(drop=True)
    fresh = fisheries.FisheriesPlan(changed, paths, cache_dir=directory)
    assert fresh.cache_info["status"] != "hit"
    assert fresh.cache_info["key"] != cold.cache_info["key"]
    assert fresh.task_count == len(zones["counties"])


@pytest.mark.parametrize("source", ["coastline_lines", "freshwater_polygons", "great_lakes_features",
                                    "inland_fisheries_exclusions"])
def test_source_content_invalidates_even_with_same_size_and_mtime(inputs, source):
    """Metadata-only fingerprints cannot hide changed spatial inputs or rules."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    _complete(cold)
    path = paths[source]
    previous = path.stat()
    if path.suffix == ".csv":
        before, after = ("11", "12") if source == "great_lakes_features" else ("14", "15")
        content = path.read_bytes()
        path.write_bytes(content.replace(before.encode(), after.encode()))
    else:
        with sqlite3.connect(path) as connection:
            # Metadata is still source content. This updates existing SQLite
            # pages without requiring GDAL's geometry-trigger SQL functions.
            connection.execute("UPDATE gpkg_contents SET description='changed source'")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] != "hit"
    assert fresh.cache_info["key"] != cold.cache_info["key"]
    assert fresh.task_count == len(zones["counties"])


def test_new_state_support_requires_measurement_instead_of_reusing_zero_weights(inputs):
    """Adding a state to a valuation table cannot reuse its unmeasured zeros."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    _complete(cold)
    inland = pd.read_csv(paths["inland_fisheries"])
    inland.loc[len(inland)] = {"State": "Ohio", fisheries._FIELDS["inland_fisheries"]: 100}
    inland.to_csv(paths["inland_fisheries"], index=False)
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] != "hit"
    assert fresh.cache_info["key"] != cold.cache_info["key"]


@pytest.mark.parametrize("source", ["coastline_lines", "freshwater_polygons"])
def test_changed_spatial_source_geometry_recomputes_the_allocation(inputs, source):
    """Changing source geometry changes both the cache and the resulting shares."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    before = _complete(cold)
    frame = fisheries.read_dataframe(paths[source])
    if source == "coastline_lines":
        frame.loc[0, "geometry"] = LineString([(10, 20), (50, 20)])
        service = "marine_fisheries"
    else:
        frame.loc[3, "geometry"] = box(100, 0, 200, 20)
        service = "inland_fisheries"
    frame.to_file(paths[source], index=False)
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] == "miss"
    assert fresh.cache_info["key"] != cold.cache_info["key"]
    after = _complete(fresh)
    assert after["counties"][service].iloc[0] != before["counties"][service].iloc[0]
    np.testing.assert_allclose(after["counties"][service].sum(), before["counties"][service].sum())


@pytest.mark.parametrize("dependency", ["schema", "runtime", "geometry_code"])
def test_measurement_implementation_identity_invalidates_cache(inputs, monkeypatch, dependency):
    """Measurement code and numerical-runtime changes cannot reuse stale weights."""
    from assessment import fisheries_weights_cache as cache

    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    _complete(cold)
    if dependency == "schema":
        monkeypatch.setattr(cache, "_SCHEMA", cache._SCHEMA + 1)
    elif dependency == "runtime":
        monkeypatch.setattr(cache.pyproj, "proj_version_str", "test-changed-proj")
    else:
        original_hash = cache.sha256

        def changed_code(path):
            return "0" * 64 if Path(path).name == "geometry_utils.py" else original_hash(path)

        monkeypatch.setattr(cache, "sha256", changed_code)
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] == "miss"
    assert fresh.cache_info["key"] != cold.cache_info["key"]


def test_incomplete_and_failed_allocations_do_not_publish_cache(inputs):
    """A cache represents a fully measured, validated set of products only."""
    zones, paths, directory = inputs
    incomplete = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    with pytest.raises(ValueError, match="Missing fisheries county results"):
        incomplete.finish()
    assert not Path(incomplete.cache_info["path"]).exists()
    totals = pd.read_csv(paths["marine_fisheries"])
    totals.loc[len(totals)] = {"State": "Texas", "ncp_value": 100}
    totals.to_csv(paths["marine_fisheries"], index=False)
    unsupported = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    with pytest.raises(ValueError, match="No allocation support"):
        _complete(unsupported)
    assert not Path(unsupported.cache_info["path"]).exists()


def test_truncated_cache_is_not_accepted_as_valid_measurements(inputs):
    """A partial artifact is fully recalculated rather than trusted."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    expected = _complete(cold)
    path = Path(cold.cache_info["path"])
    path.write_bytes(path.read_bytes()[:64])
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] == "miss"
    assert fresh.cache_info["reason"] == "invalid"
    assert fresh.task_count == len(zones["counties"])
    _assert_frames_equal(_complete(fresh), expected)


def test_source_changed_during_measurement_cannot_publish_cache(inputs):
    """Recheck full source content before recording completed spatial weights."""
    zones, paths, directory = inputs
    plan = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    for key, function, args in plan.tasks():
        plan.accept(key, function(*args))
    exclusion = paths["great_lakes_features"]
    original = exclusion.stat()
    exclusion.write_bytes(exclusion.read_bytes().replace(b"11", b"12"))
    os.utime(exclusion, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError):
        plan.finish()
    assert not Path(plan.cache_info["path"]).exists()


@pytest.mark.parametrize("damage", ["changed_finite_weight", "negative_weight", "nan_weight", "wrong_shape"])
def test_cache_payload_validation_rejects_modified_arrays(inputs, damage):
    """Schema and payload checks reject readable but corrupted measurements."""
    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    expected = _complete(cold)
    path = Path(cold.cache_info["path"])
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    candidates = [name for name, value in payload.items()
                  if value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == 2
                  and np.issubdtype(value.dtype, np.number)]
    assert candidates, "The checkpoint must store numeric spatial weights"
    name = candidates[0]
    if damage == "wrong_shape":
        payload[name] = payload[name][:-1]
    else:
        payload[name][0, 0] = {
            "changed_finite_weight": payload[name][0, 0] + 1,
            "negative_weight": -1,
            "nan_weight": np.nan,
        }[damage]
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] == "miss"
    assert fresh.cache_info["reason"] == "invalid"
    _assert_frames_equal(_complete(fresh), expected)


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
@pytest.mark.parametrize("phase", ["lookup", "save"])
def test_active_geopackage_sidecars_block_cache_without_modifying_them(inputs, suffix, phase):
    """Uncheckpointed source edits cannot be represented by the main-file hash."""
    zones, paths, directory = inputs
    plan = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    if phase == "lookup":
        _complete(plan)
    else:
        for key, function, args in plan.tasks():
            plan.accept(key, function(*args))
    sidecar = Path(str(paths["freshwater_polygons"]) + suffix)
    content = b"test active SQLite transaction; preserve this sidecar"
    sidecar.write_bytes(content)
    with pytest.raises(ValueError, match="active SQLite sidecar"):
        if phase == "lookup":
            fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
        else:
            plan.finish()
    assert sidecar.read_bytes() == content
    if phase == "save":
        assert not Path(plan.cache_info["path"]).exists()


@pytest.mark.parametrize("change", ["add_grid", "change_database"])
@pytest.mark.parametrize("phase", ["lookup", "save"])
def test_local_projection_resource_changes_invalidate_cache(inputs, monkeypatch, tmp_path, change, phase):
    """Local projection files are dependencies even when version strings match."""
    from assessment import fisheries_weights_cache as cache

    zones, paths, directory = inputs
    database = str(tmp_path / "fixture-proj.db")
    resources = {database: "1" * 64}
    monkeypatch.setattr(cache, "_projection_resources", lambda: resources.copy())
    plan = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    if phase == "lookup":
        _complete(plan)
    else:
        for key, function, args in plan.tasks():
            plan.accept(key, function(*args))
    if change == "add_grid":
        resources[str(tmp_path / "fixture-grid.tif")] = "2" * 64
    else:
        resources[database] = "2" * 64
    if phase == "lookup":
        fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
        assert fresh.cache_info["status"] == "miss"
        assert fresh.cache_info["key"] != plan.cache_info["key"]
    else:
        with pytest.raises(ValueError, match="projection resources changed"):
            plan.finish()
        assert not Path(plan.cache_info["path"]).exists()


def test_online_proj_disables_persistent_cache_without_disabling_measurement(inputs, monkeypatch):
    """Online grids cannot promise stable reuse, but fresh allocation still works."""
    zones, paths, directory = inputs
    expected = _complete(fisheries.FisheriesPlan(zones, paths))
    # Change the reported mode only; never enable actual network downloads.
    monkeypatch.setattr(fisheries.network, "is_network_enabled", lambda: True)
    plan = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert plan.cache_info == {"status": "disabled", "reason": "proj_network_enabled"}
    assert plan.task_count == len(zones["counties"])
    _assert_frames_equal(_complete(plan), expected)
    assert not directory.exists()


def test_broken_compressed_cache_is_recomputed(inputs, monkeypatch):
    """A decompression error is a safe miss rather than a reuse of bad weights."""
    from assessment import fisheries_weights_cache as cache

    zones, paths, directory = inputs
    cold = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    expected = _complete(cold)

    def broken_load(*_, **__):
        raise zlib.error("fixture compressed data corruption")

    monkeypatch.setattr(cache.np, "load", broken_load)
    fresh = fisheries.FisheriesPlan(zones, paths, cache_dir=directory)
    assert fresh.cache_info["status"] == "miss"
    assert fresh.cache_info["reason"] == "invalid"
    _assert_frames_equal(_complete(fresh), expected)
