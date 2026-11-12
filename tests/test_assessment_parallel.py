"""Check the bounded shared valuation scheduler and Windows-spawn workers."""

from concurrent.futures import FIRST_COMPLETED, Future
from multiprocessing import get_start_method
import os

import pandas as pd
import pytest

import assessment.assessment_valuation as valuation


NATIVE_LIMITS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _identity(value):
    """Return a small picklable task result."""
    return value


def _worker_environment():
    """Inspect the actual spawned process, after its numerical imports."""
    return os.getpid(), get_start_method(), {name: os.environ.get(name) for name in NATIVE_LIMITS}


def _fail():
    """Raise a recognizable task error."""
    raise ValueError("fixture geometry failed")


def _county_table_fixture(task, paths, county_base, crosswalk, geometry_workers):
    """Provide importable county tasks alongside real spawned spatial workers."""
    assert geometry_workers == 1
    return pd.DataFrame({task: range(1, len(county_base) + 1)}, index=county_base.index)


class DeferredPool:
    """Hold tasks until a test-controlled wait completes or fails them."""

    def __init__(self):
        self.tasks = []
        self.max_pending = 0

    def submit(self, function, *args):
        future = Future()
        self.tasks.append((future, function, args))
        self.max_pending = max(self.max_pending, sum(not item[0].done() for item in self.tasks))
        return future

    def complete(self, future):
        _, function, args = next(item for item in self.tasks if item[0] is future)
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)


class FixturePlan:
    """Release a dependent coastal batch when its preparation result arrives."""

    def __init__(self, family, count=1):
        self.family = family
        self.task_count = count
        self.accepted = []
        self.generated = []

    def tasks(self):
        for key in range(self.task_count):
            self.generated.append(key)
            yield key, _identity, ((self.family, key),)

    def accept(self, key, result):
        assert result == (self.family, key)
        self.accepted.append(key)
        if self.family == "coastal" and key == 0:
            return [("footprints", _identity, ((self.family, "footprints"),))]
        return ()

    def finish(self):
        expected = set(range(self.task_count))
        if self.family == "coastal":
            expected.add("footprints")
        assert set(self.accepted) == expected
        return self.family


def _table_tasks():
    return ((name, _identity, (("tables", name),)) for name in valuation.COUNTY_TABLES)


def test_queue_is_bounded_interleaved_and_collects_reversed_completions(monkeypatch):
    """Large fisheries streams neither fill memory nor delay ready coastal work."""
    pool = DeferredPool()
    plans = {"fisheries": FixturePlan("fisheries", 80), "coastal": FixturePlan("coastal")}
    waits = []
    completions = []

    def reverse_completion(pending, *, return_when):
        assert return_when == FIRST_COMPLETED
        waits.append((len(pending), len(plans["fisheries"].generated)))
        done = list(reversed(pending))
        for future in done:
            pool.complete(future)
            completions.append(future.result())
        return done, set()

    monkeypatch.setattr(valuation, "wait", reverse_completion)
    tables, fisheries, coastal = valuation._run_source_tasks(pool, plans, _table_tasks(), workers=2)
    assert pool.max_pending == 4
    assert all(size <= 4 for size, _ in waits)
    assert waits[0][1] == 1  # The remaining county payloads have not been produced yet.
    submitted = [args[0] for _, _, args in pool.tasks]
    assert [family for family, _ in submitted[:3]] == ["tables", "fisheries", "coastal"]
    assert completions[:4] == list(reversed(submitted[:4]))
    assert completions.index(("coastal", "footprints")) < completions.index(("fisheries", 79))
    assert tables == {name: ("tables", name) for name in valuation.COUNTY_TABLES}
    assert (fisheries, coastal) == ("fisheries", "coastal")


def test_task_failure_keeps_context_and_cancels_queued_work(monkeypatch):
    """A bad geometry stops dispatch rather than proceeding to publication."""
    pool = DeferredPool()
    plans = {"fisheries": FixturePlan("fisheries", 20), "coastal": FixturePlan("coastal")}

    def fail_first(pending, *, return_when):
        assert return_when == FIRST_COMPLETED
        future = next(iter(pending))
        pool.complete(future)
        return {future}, set(pending) - {future}

    monkeypatch.setattr(valuation, "wait", fail_first)
    with pytest.raises(ValueError, match="fixture geometry failed") as error:
        valuation._run_source_tasks(
            pool, plans, [("bad-table", _fail, ())], workers=2,
        )
    assert "Valuation task: tables bad-table" in error.value.__notes__
    assert len(pool.tasks) == 4
    assert all(future.cancelled() for future, _, _ in pool.tasks[1:])
    assert not plans["fisheries"].accepted
    assert not plans["coastal"].accepted


def test_collection_uses_one_executor_for_all_families(monkeypatch):
    """The parent owns the only pool; county tasks receive no subworker budget."""
    pool = DeferredPool()
    constructed = []
    plans = {"fisheries": FixturePlan("fisheries", 4), "coastal": FixturePlan("coastal")}

    class SinglePool:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def __enter__(self):
            return pool

        def __exit__(self, *_):
            return False

    def complete_all(pending, *, return_when):
        assert return_when == FIRST_COMPLETED
        for future in pending:
            pool.complete(future)
        return set(pending), set()

    def county_table(name, paths, county_base, crosswalk, geometry_workers):
        assert (paths, county_base, crosswalk) == ("paths", "counties", "crosswalk")
        assert geometry_workers == 1
        return ("tables", name)

    monkeypatch.setattr(valuation, "FisheriesPlan", lambda *_: plans["fisheries"])
    monkeypatch.setattr(valuation, "CoastalPlan", lambda *_, **__: plans["coastal"])
    monkeypatch.setattr(valuation, "ProcessPoolExecutor", SinglePool)
    monkeypatch.setattr(valuation, "compute_task", county_table)
    monkeypatch.setattr(valuation, "wait", complete_all)
    tables, fisheries, coastal = valuation._collect_service_values(
        "zones", "paths", "counties", "crosswalk", "coral counties", workers=2,
    )
    assert len(constructed) == 1
    assert constructed[0]["max_workers"] == 2
    assert constructed[0]["mp_context"].get_start_method() == "spawn"
    assert len(tables) == len(valuation.COUNTY_TABLES)
    assert (fisheries, coastal) == ("fisheries", "coastal")


def test_spawned_worker_native_limits_and_parent_environment_restoration(monkeypatch):
    """Limits precede spawned numerical imports and do not leak into the caller."""
    for name in NATIVE_LIMITS[:2]:
        monkeypatch.setenv(name, "7")
    for name in NATIVE_LIMITS[2:]:
        monkeypatch.delenv(name, raising=False)
    before = {name: os.environ.get(name) for name in NATIVE_LIMITS}
    with valuation._valuation_pool(1) as pool:
        worker_pid, start_method, limits = pool.submit(_worker_environment).result(timeout=60)
        assert worker_pid != os.getpid()
        assert start_method == "spawn"
        assert limits == dict.fromkeys(NATIVE_LIMITS, "1")
    assert {name: os.environ.get(name) for name in NATIVE_LIMITS} == before


def test_parent_environment_is_restored_on_pool_failure(monkeypatch):
    """Construction errors restore both existing and originally unset variables."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "6")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    before = {name: os.environ.get(name) for name in NATIVE_LIMITS}

    def broken_pool(**_):
        assert {name: os.environ.get(name) for name in NATIVE_LIMITS} == dict.fromkeys(NATIVE_LIMITS, "1")
        raise RuntimeError("pool could not start")

    monkeypatch.setattr(valuation, "ProcessPoolExecutor", broken_pool)
    with pytest.raises(RuntimeError, match="pool could not start"), valuation._valuation_pool(2):
        pytest.fail("Failed pool must not yield")
    assert {name: os.environ.get(name) for name in NATIVE_LIMITS} == before


def test_pool_rejects_nonpositive_workers():
    """Reject an invalid worker budget before touching the environment."""
    with pytest.raises(ValueError, match="positive"), valuation._valuation_pool(0):
        pytest.fail("Invalid worker budget must not yield")


def test_real_spatial_plans_share_spawned_pool_and_match_one_worker(monkeypatch):
    """Run real GeoPackage tasks and dependent coastal batches through one pool."""
    # Reuse the fixture builders without exposing their TestCase classes to
    # pytest collection in this module.
    from tests.test_assessment_coastal import AssessmentCoastalTests as _CoastalFixture
    from tests.test_assessment_fisheries import FootprintFisheriesTests as _FisheriesFixture
    from assessment.assessment_coastal import value_coastal
    from assessment.assessment_fisheries import value_fisheries

    fisheries_fixture = _FisheriesFixture()
    coastal_fixture = _CoastalFixture()
    try:
        fisheries_fixture.setUp()
        coastal_fixture.setUp()
        zones = fisheries_fixture.zones
        paths = fisheries_fixture.paths | coastal_fixture.paths
        for position, frame in enumerate(zones.values()):
            frame.index = pd.Index(range(10 * position, 10 * position + len(frame)), name="zone_row")
        counties = zones["counties"]
        county_base = pd.DataFrame(counties.drop(columns=counties.geometry.name)).set_index("GEOID")
        crosswalk = pd.DataFrame(columns=["source_geoid", "target_geoid", "weight"])
        expected_fisheries = value_fisheries(zones, paths, geometry_workers=1)
        expected_coastal = value_coastal(
            zones, paths, geometry_workers=1, coral_reference_counties=counties,
        )
        monkeypatch.setattr(valuation, "compute_task", _county_table_fixture)
        results = [valuation._collect_service_values(
            zones, paths, county_base, crosswalk, counties, workers=workers,
        ) for workers in (1, 2)]
        for tables, fisheries, coastal in results:
            assert set(tables) == set(valuation.COUNTY_TABLES)
            assert set(fisheries) == set(coastal) == set(zones)
            for task in valuation.COUNTY_TABLES:
                pd.testing.assert_frame_equal(
                    tables[task], _county_table_fixture(task, paths, county_base, crosswalk, 1),
                    check_exact=True,
                )
            for product in zones:
                pd.testing.assert_frame_equal(fisheries[product], expected_fisheries[product], check_exact=True)
                pd.testing.assert_frame_equal(coastal[product], expected_coastal[product], check_exact=True)
                assert fisheries[product].index.equals(zones[product].index)
                assert coastal[product].index.equals(zones[product].index)
    finally:
        fisheries_fixture.doCleanups()
        coastal_fixture.doCleanups()
