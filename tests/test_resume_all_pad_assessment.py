"""Resume a failed all-PAD batch using small real files and a stubbed toolkit."""

from configparser import ConfigParser
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from osgeo import gdal, osr
from shapely.geometry import box

import run_conus_ecosystem_services as wrapper
from tests import test_run_conus_ecosystem_services as existing


def write_pair(plan, job, stamp="20261112_123456"):
    """Create an actual matching CSV/GeoPackage pair, without raster analysis."""
    base = job.output_base.with_name(f"{job.output_base.name}_{stamp}")
    base.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(sorted(job.expected_keys), columns=job.key_fields)
    for number, column in enumerate(plan.raster_columns, 1):
        frame[column] = float(number)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    vector = gpd.GeoDataFrame(frame, geometry=[box(0, 0, 10, 10)] * len(frame), crs=6933)
    pyogrio.write_dataframe(vector, base.with_suffix(".gpkg"), layer=job.product)
    return base.with_suffix(".csv"), base.with_suffix(".gpkg")


def snapshot(root):
    """Record file contents to detect writes in read-only preflight."""
    return {str(path.relative_to(root)): wrapper.sha256(path)
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def failed_run():
    """Use real frozen plans, hashes and pairs; replace only expensive execution."""
    fixture = existing.ConusEcosystemServicesTests(methodName="runTest")
    fixture.setUp()
    try:
        fixture.run_wrapper()
        raster_dir = fixture.root / "rasters"
        raster_dir.mkdir()
        inventory = []
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(6933)
        for field in dict.fromkeys(wrapper.BASE_FIELDS.values()):
            name = field.removeprefix("sum_") + ".tif"
            dataset = gdal.GetDriverByName("GTiff").Create(str(raster_dir / name), 2, 2, 1, gdal.GDT_Float32)
            dataset.SetProjection(spatial_ref.ExportToWkt())
            dataset.SetGeoTransform((0, 10, 0, 20, 0, -10))
            dataset.GetRasterBand(1).Fill(1)
            dataset = None
            inventory.append({"label": name, "filename": name, "units": "USD/pixel", "include": True})
        reviewed = ConfigParser(interpolation=None)
        reviewed["project"] = {"raster_dir": str(raster_dir), "max_simplify_tolerance_meters": "15",
                               "raster_inventory": json.dumps(inventory)}
        reviewed["job:template"] = {"zone_key": "unused"}
        with fixture.zonal_config.open("w", encoding="utf-8") as stream:
            reviewed.write(stream)
        zones, overlap = wrapper.read_prepared_zones(fixture.zone_root, fixture.run_id)
        fixture.frozen.unlink()  # Replace the fixture's placeholder, not an analysis file.
        base_plan = wrapper.freeze_zone_zonals(
            zones, fixture.zonal_config, fixture.frozen, fixture.work / "zonal_stats_work",
            fixture.work / "raw_zonals", overlapping_manager_products=overlap,
        )
        for job in base_plan.jobs:
            write_pair(base_plan, job)
        base_manifest_path = fixture.results / "run_manifest.json"
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        base_manifest["configuration_sha256"] = {
            name: wrapper.sha256(name) for name in base_manifest["configuration_sha256"]
        }
        base_manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")

        run_id = "failed-pad"
        zone, frozen, work, raw = wrapper.all_pad_paths(fixture.root, run_id)
        zone.parent.mkdir(parents=True)
        pad = gpd.GeoDataFrame({
            "GEOID": ["01001", "01001"], "feature_id": ["01001_1", "01001_2"],
            "Mang_Type": ["FED", "FED"], "area_ha_land": [.01, .01],
        }, geometry=[box(0, 0, 10, 10), box(0, 0, 10, 10)], crs=6933)
        pyogrio.write_dataframe(pad, zone, layer=wrapper.ALL_PAD_PRODUCT, promote_to_multi=True)
        plan = wrapper.freeze_zone_zonals(
            {wrapper.ALL_PAD_PRODUCT: (zone, wrapper.ALL_PAD_PRODUCT)}, fixture.zonal_config,
            frozen, work, raw, products=(wrapper.ALL_PAD_PRODUCT,), overlapping_manager_products=(),
        )
        assert len(plan.jobs) == 2
        results = fixture.results.parent / run_id
        results.mkdir()
        manifest_path = results / "run_manifest.json"
        manifest = copy.deepcopy(base_manifest)
        manifest.update(run_id=run_id, status="failed", outputs={}, error="simulated toolkit failure",
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        reused_zonals_from={"run_id": fixture.run_id, "manifest": str(base_manifest_path)})
        manifest["stages"] = {
            wrapper._REUSE_RASTERS: "complete", wrapper._PREPARE_PAD: "complete",
            wrapper._FREEZE_PAD: "complete", wrapper._RUN_PAD: "failed",
        }
        manifest["configuration_sha256"][str(fixture.frozen)] = wrapper.sha256(fixture.frozen)
        manifest["all_pad_zonals"] = {
            "run_id": run_id, "zone": str(zone), "zone_sha256": wrapper.sha256(zone),
            "frozen_config": str(frozen), "frozen_sha256": wrapper.sha256(frozen),
            "work_directory": str(work), "raw_directory": str(raw), "raster_jobs": len(plan.jobs),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        state = SimpleNamespace(f=fixture, run_id=run_id, results=results, manifest_path=manifest_path,
                                plan=plan, zone=zone, frozen=frozen, work=work, raw=raw,
                                source=next(raster_dir.glob("*.tif")), calls=[], fail_after=None,
                                mutate_after_toolkit=None, partial_failure=False)
        yield state
    finally:
        fixture.doCleanups()


@contextmanager
def execution(state):
    """Run real preflight, resume and publication without launching analysis."""
    def toolkit(command, **kwargs):
        assert Path(command[2]) == state.f.runner
        if command[3:] == ["--help"]:
            return subprocess.CompletedProcess(command, 0, stdout="usage: runner.py --total-workers N")
        state.calls.append(command)
        assert command[4:] == ["--total-workers", "2"]
        config = ConfigParser(interpolation=None)
        config.read(command[3], encoding="utf-8")
        selected = {name.removeprefix("job:") for name in config.sections() if name.startswith("job:")}
        for number, job in enumerate((job for job in state.plan.jobs if job.output_base.name in selected), 1):
            _, gpkg = write_pair(state.plan, job, "20261113_123456")
            if number == state.fail_after:
                if state.partial_failure:
                    gpkg.unlink()
                raise subprocess.CalledProcessError(17, command)
        if state.mutate_after_toolkit is not None:
            state.mutate_after_toolkit()
        return subprocess.CompletedProcess(command, 0)

    def valuation(zones, sums, *_args, **_kwargs):
        assert set(zones) == set(wrapper.ZONE_PRODUCTS) | {wrapper.ALL_PAD_PRODUCT}
        assert set(sums) == set(zones)
        assert len(sums[wrapper.ALL_PAD_PRODUCT]) == 2
        valued = {name: frame.assign(es_total=10.0) for name, frame in zones.items()}
        valued["counties"]["recreation_adj"] = 5.0
        return valued, {}, {}

    with (
        patch.object(wrapper.subprocess, "run", side_effect=toolkit),
        patch.object(wrapper, "load_valuation_config", return_value=(
            {"air_quality_county_aliases": state.f.air_aliases}, None, None,
            {"sources": {"gdp": {"path": "gdp.csv", "allocation_crosswalk":
                                      "workflow_assets/county_integration/virginia_gdp_crosswalk.csv"}}},
        )),
        patch.object(wrapper, "prepare_all_pad_zones", side_effect=AssertionError("must reuse prepared PAD-US")),
        patch.object(wrapper, "freeze_zone_zonals", side_effect=AssertionError("must reuse frozen plan")),
        patch.object(wrapper, "value_zones", side_effect=valuation),
    ):
        yield


def resume(state, *extra):
    args = wrapper.parse_args(["--resume-run", state.run_id, "--runner", str(state.f.runner),
                               "--zonal-workers", "2", *extra])
    with execution(state):
        return wrapper.run_assessment(args, root=state.f.root)


@pytest.mark.parametrize("retained", [0, 1, 2])
def test_resume_only_missing_jobs_and_publish_seven_pairs(failed_run, retained):
    state = failed_run
    retained_paths = [path for job in state.plan.jobs[:retained] for path in write_pair(state.plan, job)]
    protected = [state.zone, state.frozen, *state.work.rglob("*.gpkg"), *retained_paths]
    before = {path: wrapper.sha256(path) for path in protected}
    parent_before = snapshot(state.f.results)
    assert resume(state) == state.results
    manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["stages"][wrapper._RUN_PAD] == "complete"
    assert len(manifest["resume_attempts"]) == 1
    assert manifest["reused_zonals_from"]["run_id"] == state.f.run_id
    assert set(manifest["outputs"]) == set(wrapper.ZONE_PRODUCTS) | {wrapper.ALL_PAD_PRODUCT}
    assert len(list(state.results.glob("*.csv"))) == len(list(state.results.glob("*.gpkg"))) == 7
    assert {path: wrapper.sha256(path) for path in protected} == before
    assert snapshot(state.f.results) == parent_before
    batch = manifest["all_pad_zonals"]["resume_batches"][-1]
    assert batch["retained_jobs"] == retained
    assert batch["jobs"] == [job.output_base.name for job in state.plan.jobs[retained:]]
    assert len(state.calls) == (retained < 2)
    if state.calls:
        assert Path(state.calls[0][3]) != state.frozen
        assert wrapper.sha256(state.calls[0][3]) == batch["sha256"]
    # A resumed run remains a valid source, including mixed retained/new timestamps.
    saved = wrapper.read_saved_zonals(state.f.root, state.run_id)
    assert len(saved.sums[wrapper.ALL_PAD_PRODUCT]) == 2


def test_validate_only_is_read_only(failed_run):
    before = snapshot(failed_run.f.root)
    assert resume(failed_run, "--validate-only") is None
    assert snapshot(failed_run.f.root) == before
    assert failed_run.calls == []


def test_first_resume_rejects_unrecorded_mixed_batch_timestamps(failed_run):
    state = failed_run
    write_pair(state.plan, state.plan.jobs[0], "20261112_123456")
    write_pair(state.plan, state.plan.jobs[1], "20261112_234567")
    before = snapshot(state.f.root)
    with pytest.raises(ValueError, match="timestamp|batch"):
        resume(state)
    assert snapshot(state.f.root) == before


def change_source_pixels(path):
    """Change a real TIFF's pixels while preserving its size and mtime."""
    original = path.stat()
    dataset = gdal.Open(str(path), gdal.GA_Update)
    dataset.GetRasterBand(1).Fill(7)
    dataset = None
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert path.stat().st_size == original.st_size


def test_legacy_resume_rejects_source_newer_than_frozen_plan(failed_run):
    state = failed_run
    stat = state.source.stat()
    os.utime(state.source, ns=(stat.st_atime_ns, state.frozen.stat().st_mtime_ns + 60_000_000_000))
    before = snapshot(state.f.root)
    with pytest.raises(ValueError, match="(?i)raster|tiff|source"):
        resume(state)
    assert snapshot(state.f.root) == before
    assert state.calls == []


def test_retry_rejects_changed_source_content_even_with_same_size_and_mtime(failed_run):
    state = failed_run
    state.fail_after = 1
    with pytest.raises(subprocess.CalledProcessError):
        resume(state)
    manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert manifest["all_pad_zonals"]["source_raster_sha256"][str(state.source)] == wrapper.sha256(state.source)
    change_source_pixels(state.source)
    before = snapshot(state.f.root)
    with pytest.raises(ValueError, match="(?i)raster|tiff|source"):
        resume(state)
    assert snapshot(state.f.root) == before
    assert len(state.calls) == 1


def test_source_change_during_toolkit_prevents_final_publication(failed_run):
    state = failed_run
    state.mutate_after_toolkit = lambda: change_source_pixels(state.source)
    with pytest.raises(ValueError, match="(?i)raster|tiff|source"):
        resume(state)
    manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["outputs"] == {}
    assert not list(state.results.glob("*.csv"))
    assert not list(state.results.glob("*.gpkg"))


@pytest.mark.parametrize("change", ["configuration", "zone", "frozen", "shard"])
def test_resume_rejects_changed_inputs_before_writes(failed_run, change):
    state = failed_run
    target = {"configuration": state.f.air_aliases, "zone": state.zone, "frozen": state.frozen,
              "shard": next(state.work.rglob("*.gpkg"))}[change]
    with target.open("ab") as stream:
        stream.write(b"changed")
    before = snapshot(state.f.root)
    with pytest.raises(ValueError):
        resume(state)
    assert snapshot(state.f.root) == before
    assert state.calls == []


@pytest.mark.parametrize("problem", ["half", "duplicate", "different_values", "different_keys", "unexpected"])
def test_resume_rejects_untrustworthy_completed_pairs(failed_run, problem):
    state = failed_run
    csv, gpkg = write_pair(state.plan, state.plan.jobs[0])
    if problem == "half":
        gpkg.unlink()
    elif problem == "duplicate":
        write_pair(state.plan, state.plan.jobs[0], "20261112_234567")
    elif problem in {"different_values", "different_keys"}:
        frame = pd.read_csv(csv, dtype={"GEOID": str, "feature_id": str})
        frame.loc[0, "feature_id" if problem == "different_keys" else state.plan.raster_columns[0]] = (
            "wrong" if problem == "different_keys" else 900.0)
        frame.to_csv(csv, index=False)
    else:
        (csv.parent / "unknown.csv").write_text("unexpected", encoding="utf-8")
    before = snapshot(state.f.root)
    with pytest.raises((ValueError, FileNotFoundError)):
        resume(state)
    assert snapshot(state.f.root) == before


@pytest.mark.parametrize("problem", ["complete", "running", "prepare", "freeze", "raster", "output_file", "output_record"])
def test_resume_rejects_wrong_stage_or_published_results(failed_run, problem):
    state = failed_run
    manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    if problem in {"complete", "running"}:
        manifest["status"] = problem
    elif problem in {"prepare", "freeze", "raster"}:
        stage = {"prepare": wrapper._PREPARE_PAD, "freeze": wrapper._FREEZE_PAD, "raster": wrapper._RUN_PAD}[problem]
        manifest["stages"][stage] = "running"
    elif problem == "output_file":
        nested = state.results / "nested"
        nested.mkdir()
        (nested / "partial.csv").write_text("partial", encoding="utf-8")
    else:
        manifest["outputs"] = {"counties": {"csv": "missing.csv"}}
    state.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = snapshot(state.f.root)
    with pytest.raises((ValueError, FileExistsError)):
        resume(state)
    assert snapshot(state.f.root) == before


def test_second_failure_retains_completed_jobs_for_next_resume(failed_run):
    state = failed_run
    state.fail_after = 1
    with pytest.raises(subprocess.CalledProcessError):
        resume(state)
    failed = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["stages"][wrapper._RUN_PAD] == "failed"
    retained = {path: wrapper.sha256(path) for path in state.raw.rglob("*") if path.is_file()}
    assert len(retained) == 2
    state.fail_after = None
    resume(state)
    complete = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert len(complete["resume_attempts"]) == len(complete["all_pad_zonals"]["resume_batches"]) == 2
    assert complete["all_pad_zonals"]["resume_batches"][-1]["retained_jobs"] == 1
    assert {path: wrapper.sha256(path) for path in retained} == retained


def test_partial_output_preserves_toolkit_error_and_completed_sibling(failed_run):
    state = failed_run
    state.fail_after = 2
    state.partial_failure = True
    with pytest.raises(subprocess.CalledProcessError) as caught:
        resume(state)
    assert caught.value.returncode == 17
    failed = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["stages"][wrapper._RUN_PAD] == "failed"
    complete_job, partial_job = state.plan.jobs
    retained = {path: wrapper.sha256(path) for path in complete_job.output_base.parent.glob(
        complete_job.output_base.name + "_*")}
    assert len(retained) == 2
    partial = list(partial_job.output_base.parent.glob(partial_job.output_base.name + "_*.csv"))
    assert len(partial) == 1
    partial[0].unlink()  # Simulate the operator removing only the incomplete synthetic pair.
    state.fail_after = None
    resume(state)
    complete = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert complete["all_pad_zonals"]["resume_batches"][-1]["jobs"] == [partial_job.output_base.name]
    assert {path: wrapper.sha256(path) for path in retained} == retained


@pytest.mark.parametrize("change", ["unknown_job", "forged_timestamp"])
def test_retry_rejects_unrecorded_job_or_timestamp(failed_run, change):
    state = failed_run
    state.fail_after = 1
    with pytest.raises(subprocess.CalledProcessError):
        resume(state)
    manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    batch = manifest["all_pad_zonals"]["resume_batches"][-1]
    if change == "unknown_job":
        batch["jobs"].append("unknown-job")
    else:
        name = next(iter(batch["job_timestamps"]))
        batch["job_timestamps"][name] = "20261101_000000"
    state.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = snapshot(state.f.root)
    with pytest.raises(ValueError, match="timestamp|batch|provenance"):
        resume(state)
    assert snapshot(state.f.root) == before
    assert len(state.calls) == 1


@pytest.mark.parametrize("override", [
    ["--run-id", "other"], ["--reuse-zonals-from", "other"], ["--include-all-pad"],
    ["--bbb-by-county", "other.gpkg"], ["--land-mask", "other.gpkg"],
    ["--counties", "other.gpkg"], ["--all-pad-layer", "other"],
])
def test_resume_cli_rejects_input_and_run_overrides(override):
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--resume-run", "failed-pad", "--runner", "runner.py", *override])


def test_resume_requires_runner():
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--resume-run", "failed-pad"])


def test_resume_lock_is_exclusive_and_released(tmp_path):
    lock = tmp_path / "resume.lock"
    with wrapper._resume_lock(lock):
        with pytest.raises(RuntimeError, match="already resuming"):
            with wrapper._resume_lock(lock):
                pytest.fail("second resume acquired the lock")
    with wrapper._resume_lock(lock):
        pass


@pytest.mark.parametrize("change", ["manifest", "final_output"])
def test_resume_rechecks_state_after_preflight(failed_run, change):
    state = failed_run
    original_lock = wrapper._resume_lock

    @contextmanager
    def racing_lock(path):
        if change == "manifest":
            manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
            manifest["error"] = "concurrent change"
            state.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            (state.results / "concurrent.csv").write_text("concurrent", encoding="utf-8")
        with original_lock(path):
            yield

    with patch.object(wrapper, "_resume_lock", side_effect=racing_lock):
        with pytest.raises((RuntimeError, FileExistsError)):
            resume(state)
    assert state.calls == []
    assert json.loads(state.manifest_path.read_text(encoding="utf-8"))["status"] == "failed"
