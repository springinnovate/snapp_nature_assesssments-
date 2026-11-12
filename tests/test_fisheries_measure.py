"""Check the contained-water shortcut against exact constructive intersections."""

import numpy as np
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box

from assessment.assessment_fisheries import _measure


def _exact_measure(zone, tree, *, length=False):
    candidates = tree.geometries[tree.query(zone, predicate="intersects")]
    clipped = shapely.intersection(candidates, zone)
    measures = shapely.length(clipped) if length else shapely.area(clipped)
    return float(measures.sum()) / (1000 if length else 10_000)


def test_only_boundary_crossing_water_requires_intersection(monkeypatch):
    """Keep complete areas, including shared boundaries, without clipping them."""
    zone = box(0, 0, 10, 10)
    sources = [box(1, 1, 2, 2), box(0, 0, 1, 1), box(9, 9, 11, 11), box(20, 20, 21, 21)]
    tree = shapely.STRtree(sources)
    before = [shape.wkb for shape in [zone, *sources]]
    expected = _exact_measure(zone, tree)
    real_intersection = shapely.intersection
    clipped = []

    def record_intersection(candidates, geometry):
        clipped.extend(candidates)
        return real_intersection(candidates, geometry)

    monkeypatch.setattr(shapely, "intersection", record_intersection)
    assert _measure(zone, tree, length=False) == expected == 3 / 10_000
    assert len(clipped) == 1
    assert clipped[0].equals(sources[2])
    assert [shape.wkb for shape in [zone, *sources]] == before


@pytest.mark.parametrize("zone", [
    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]]),
    MultiPolygon([box(0, 0, 3, 10), box(7, 0, 10, 10)]),
    shapely.difference(box(0, 0, 10, 10), box(4, 0, 6, 10)),
])
def test_holes_multipart_and_exclusion_footprints_match_exact_clip(zone):
    """A bounding-box containment test would incorrectly count holes/exclusions."""
    sources = [box(1, 1, 2, 2), box(4.2, 4.2, 5.8, 5.8), box(3, 3, 5, 5),
               box(9, 9, 11, 11), box(10, 1, 11, 2), zone,
               MultiPolygon([box(0, 0, 1, 1), box(8, 8, 9, 9)]),
               GeometryCollection([box(1, 3, 2, 4), LineString([(1, 4), (1, 5)])])]
    tree = shapely.STRtree(sources)
    np.testing.assert_allclose(_measure(zone, tree, length=False), _exact_measure(zone, tree),
                               rtol=1e-14, atol=1e-14)


def test_repaired_water_and_empty_matches():
    """Handle the repaired candidates supplied by the fisheries worker."""
    bowtie = Polygon([(1, 1), (4, 4), (1, 4), (4, 1), (1, 1)])
    tree = shapely.STRtree([shapely.make_valid(bowtie), box(5, 5, 7, 7)])
    zone = box(0, 0, 6, 6)
    assert _measure(zone, tree, length=False) == _exact_measure(zone, tree)
    assert _measure(box(20, 20, 21, 21), tree, length=False) == 0.0
    assert _measure(zone, shapely.STRtree([]), length=False) == 0.0


def test_contained_only_water_never_calls_constructive_overlay(monkeypatch):
    """All-contained queries can return their full source areas directly."""
    def unexpected_intersection(*_):
        pytest.fail("Contained water must not require an intersection")

    monkeypatch.setattr(shapely, "intersection", unexpected_intersection)
    tree = shapely.STRtree([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    assert _measure(box(0, 0, 10, 10), tree, length=False) == 2 / 10_000


def test_marine_lengths_keep_exact_intersection_path():
    """The freshwater optimization does not alter marine shoreline measurement."""
    zone = box(0, 0, 10, 10)
    tree = shapely.STRtree([LineString([(1, 1), (2, 2)]), LineString([(9, 5), (11, 5)])])
    assert _measure(zone, tree, length=True) == _exact_measure(zone, tree, length=True)
