"""Exercise assessment-zone CLI outputs using known equal-area geometry."""

import csv
import json

import geopandas as gpd
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from prepare_assessment_zones import main


CRS = "EPSG:6933"
RUN_ID = "synthetic-assessment"


def write_layer(path, fields, geometries):
    """Write one projected fixture layer in the same format as the CLI inputs.

    Args:
        path: Destination GeoPackage path.
        fields: Attribute columns to include in the feature layer.
        geometries: Polygon geometries in metres in the fixture CRS.
    """
    gpd.GeoDataFrame(fields, geometry=geometries, crs=CRS).to_file(
        path, layer=path.stem, driver="GPKG", index=False
    )


def read_product(zones, name):
    """Read a named output layer through its public on-disk contract.

    Args:
        zones: The run's zonal-units directory.
        name: Product directory and layer name.

    Returns:
        The exported features projected into the equal-area fixture CRS.
    """
    return gpd.read_file(zones / name / f"{name}_{RUN_ID}.gpkg", layer=name).to_crs(CRS)


def assert_footprint(geometry, expected):
    """Compare a published polygon with its independently defined footprint.

    Args:
        geometry: A polygon read from an output layer.
        expected: The analytically expected polygon or polygon union.
    """
    assert geometry.is_valid and not geometry.is_empty
    assert geometry.symmetric_difference(expected).area == pytest.approx(0, abs=0.05)


@pytest.fixture
def zone_inputs(tmp_path):
    """Build land-only footprints in counties that are larger than their ALAND.

    Args:
        tmp_path: A fresh test directory supplied by pytest.

    Returns:
        CLI arguments, output paths, and fixture paths for the test to modify.
    """
    counties = tmp_path / "counties.gpkg"
    padus = tmp_path / "padus.gpkg"
    bbb = tmp_path / "bbb.gpkg"
    output_root = tmp_path / "assessment runs"
    county_boxes = [box(x, 0, x + 1000, 1000) for x in range(0, 3000, 1000)]
    ids = ["01001", "01003", "01005"]
    write_layer(counties, {"GEOID": ids, "ALAND": [800000] * 3, "NAME": ["One", "Two", "Three"]}, county_boxes)
    write_layer(
        padus,
        {
            "Mang_Type": ["FED", "STAT", "UNK", "NGO", "PVT", "UNK", "LOC", "FED"],
            "Des_Tp": ["ND", "ND", "ACC", "ND", "ND", "ND", "ND", "ND"],
            "GAP_Sts": ["1", "2", "3", "2", "1", "4", "4", "1"],
        },
        [
            box(0, 0, 600, 750), box(400, 0, 1200, 750), box(1200, 0, 1400, 750),
            box(1300, 0, 1700, 750), box(1600, 0, 1800, 750), box(1800, 0, 1900, 750),
            box(1900, 500, 2000, 750), box(200, 0, 400, 750),
        ],
    )
    write_layer(
        bbb,
        {"GEOID": ["01001", "01003"]},
        [box(800, 500, 1200, 750), box(900, 0, 1400, 750)],
    )
    args = [
        "--counties", str(counties), "--padus-path", str(padus),
        "--bbb-by-county", str(bbb), "--output-root", str(output_root), "--run-id", RUN_ID,
        "--workers", "1",
    ]
    return args, output_root / RUN_ID / "zonal_units", counties, padus, county_boxes


def test_six_products_use_county_clipped_footprints_and_aland(zone_inputs):
    """Verify land numerators, Census denominators, and independent land rules."""
    args, zones, _, _, county_boxes = zone_inputs
    main(args)

    expected = {
        "public_lands_by_county": {
            "01001": (box(0, 0, 1000, 750), 75),
            "01003": (unary_union([box(1000, 0, 1400, 750), box(1900, 500, 2000, 750)]), 32.5),
        },
        "conservation_lands_by_county": {
            "01001": (box(0, 0, 1000, 750), 75),
            "01003": (box(1000, 0, 1800, 750), 60),
        },
        "bbb_lands_by_county": {
            "01001": (box(800, 500, 1000, 750), 5),
            "01003": (box(1000, 0, 1400, 750), 30),
        },
    }
    manager_common = {
        ("01001", "FED"): (box(0, 0, 600, 750), 45),
        ("01001", "STAT"): (box(400, 0, 1000, 750), 45),
        ("01003", "STAT"): (box(1000, 0, 1200, 750), 15),
        ("01003", "UNK"): (box(1200, 0, 1400, 750), 15),
    }
    manager_expected = {
        "public_lands_by_county_and_manager_type": {
            **manager_common,
            ("01003", "LOC"): (box(1900, 500, 2000, 750), 2.5),
        },
        "conservation_lands_by_county_and_manager_type": {
            **manager_common,
            ("01003", "NGO"): (box(1300, 0, 1700, 750), 30),
            ("01003", "PVT"): (box(1600, 0, 1800, 750), 15),
        },
    }
    assert {path.name for path in zones.iterdir() if path.is_dir()} == {
        "counties", *expected, *manager_expected,
    }

    counties = read_product(zones, "counties").set_index("GEOID")
    ids = ["01001", "01003", "01005"]
    assert counties.index.is_unique
    assert set(counties.index) == set(ids)
    assert list(counties.loc[ids, "ALAND"]) == [800000] * 3
    assert list(counties.loc[ids, "NAME"]) == ["One", "Two", "Three"]
    for geoid, geometry in zip(ids, county_boxes):
        assert_footprint(counties.loc[geoid, "geometry"], geometry)
        assert counties.loc[geoid, "area_ha_county_land"] == pytest.approx(80)

    for name, footprints in {**expected, **manager_expected}.items():
        fields = ["GEOID", "Mang_Type"] if name in manager_expected else "GEOID"
        result = read_product(zones, name).set_index(fields)
        assert result.index.is_unique
        assert set(result.index) == set(footprints)
        for key, (geometry, hectares) in footprints.items():
            assert_footprint(result.loc[key, "geometry"], geometry)
            assert result.loc[key, "area_ha_land"] == pytest.approx(hectares)
            assert result.loc[key, "county_land_share"] == pytest.approx(hectares / 80)

    for product, footprints in expected.items():
        land_class = product.removesuffix("_by_county")
        for geoid in ids:
            hectares = footprints.get(geoid, (None, 0))[1]
            assert counties.loc[geoid, f"area_ha_{land_class}"] == pytest.approx(hectares)
            assert counties.loc[geoid, f"pct_{land_class}"] == pytest.approx(hectares / 80 * 100)

    with (zones / "county_area_qa.csv").open(encoding="utf-8-sig", newline="") as qa_file:
        qa_rows = {row["GEOID"]: row for row in csv.DictReader(qa_file)}
    assert set(qa_rows) == set(ids)
    for geoid, row in qa_rows.items():
        assert float(row["area_ha_county_land"]) == pytest.approx(80)
        assert float(row["area_ha_county_polygon"]) == pytest.approx(100)
        assert float(row["area_ha_public_manager_overlap"]) == pytest.approx(15 if geoid == "01001" else 0)
        assert float(row["area_ha_conservation_manager_overlap"]) == pytest.approx(0 if geoid == "01005" else 15)
        assert not any(
            field.startswith("land_area_error_") or field.endswith("_manager_land_overlap") for field in row
        )
    manifest = json.loads((zones / "zone_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID
    assert set(manifest["inputs"]) == {"counties", "padus", "bbb_by_county"}
    assert manifest["land_area_method"]["footprints_assumed_land_only"] is True
    assert set(manifest["outputs"]) == {"counties", *expected, *manager_expected}
    assert manifest["counties_with_manager_overlap"] == {
        "area_ha_public_manager_overlap": 1,
        "area_ha_conservation_manager_overlap": 2,
    }


def test_existing_run_is_not_overwritten(zone_inputs):
    """Reject a previously used run ID without changing the existing outputs."""
    args, zones, _, _, _ = zone_inputs
    zones.mkdir(parents=True)
    prior = zones / "zone_manifest.json"
    prior.write_text("previous run", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Zone output already exists"):
        main(args)
    assert prior.read_text(encoding="utf-8") == "previous run"
    assert list(zones.iterdir()) == [prior]


def test_empty_public_layer_and_missing_conservation_manager(zone_inputs):
    """Write empty products with a usable schema and keep unassigned conservation land."""
    args, zones, _, padus, _ = zone_inputs
    padus.unlink()
    write_layer(padus, {"Mang_Type": [None], "Des_Tp": ["ACC"], "GAP_Sts": [3]},
                [box(0, 0, 400, 750)])
    main(args)
    public = read_product(zones, "public_lands_by_county_and_manager_type")
    assert public.empty
    assert set(public.columns) == {"GEOID", "Mang_Type", "area_ha_land", "county_land_share", "geometry"}
    conservation = read_product(zones, "conservation_lands_by_county_and_manager_type")
    assert list(conservation["Mang_Type"]) == ["UNSPECIFIED"]
    assert conservation.loc[0, "area_ha_land"] == pytest.approx(30)


@pytest.mark.parametrize("aland", [0, None])
def test_invalid_aland_does_not_publish(zone_inputs, aland):
    """Reject a nonpositive or missing Census land-area denominator."""
    args, zones, counties, _, county_boxes = zone_inputs
    counties.unlink()
    write_layer(
        counties,
        {"GEOID": ["01001", "01003", "01005"], "ALAND": [800000, 800000, aland]},
        county_boxes,
    )
    with pytest.raises(ValueError, match="finite positive Census ALAND"):
        main(args)
    assert not zones.exists()


def test_union_materially_larger_than_aland_does_not_publish(zone_inputs):
    """Reject a county union larger than ALAND even if each manager fits."""
    args, zones, counties, _, county_boxes = zone_inputs
    counties.unlink()
    write_layer(
        counties,
        {"GEOID": ["01001", "01003", "01005"], "ALAND": [600000, 800000, 800000]},
        county_boxes,
    )
    with pytest.raises(ValueError, match="01001.*ALAND"):
        main(args)
    assert not zones.exists()
