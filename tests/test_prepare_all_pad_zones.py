"""Verify original PAD-US records survive county clipping and area allocation."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import MultiPolygon, box

from assessment.prepare_all_pad_zones import ALL_PAD_PRODUCT, prepare_all_pad_zones


CRS = "EPSG:6933"


def write_layer(path: Path, fields: dict, geometry: list) -> None:
    """Create a small source layer with genuine driver feature identifiers."""
    pyogrio.write_dataframe(gpd.GeoDataFrame(fields, geometry=geometry, crs=CRS), path, layer=path.stem)


@pytest.fixture
def inputs(tmp_path):
    counties = tmp_path / "counties.gpkg"
    masks = tmp_path / "land.gpkg"
    padus = tmp_path / "padus.gpkg"
    output = tmp_path / "output" / "all_pad.gpkg"
    write_layer(counties, {"GEOID": ["01001", "01003", "02001"], "ALAND": [8000] * 3}, [
        box(0, 0, 100, 100), box(100, 0, 200, 100), box(200, 0, 300, 100),
    ])
    write_layer(masks, {"GEOID": ["01001", "01003", "02001"]}, [
        box(0, 0, 100, 80), box(100, 0, 200, 80), box(200, 0, 300, 80),
    ])
    write_layer(padus, {
        "Mang_Type": ["FED", "TRIB", "TRIB", "PVT", None],
        "GAP_Sts": ["1", "4", None, "4", "9"],
        "Own_Name": ["Outside", "Nation", None, "Alaska", "Water"],
        "Unit_Nm": ["A", "Cross county", "Overlapping record", "D", "Water record"],
        "GIS_Acres": [999.0, 345.6, 12.3, 987.6, 123.4],
        "SHAPE_Area": [8.0, 7.0, 6.0, 5.0, 4.0],
    }, [
        box(400, 0, 450, 50), box(50, 0, 150, 100), box(60, 20, 90, 40),
        box(200, 0, 220, 40), box(10, 90, 20, 100),
    ])
    return (padus, "padus", counties, "counties", masks, "land", output)


def test_source_records_fields_and_ids_survive_county_cut(inputs):
    """Overlapping same-manager records and every GAP code remain independent."""
    provenance = prepare_all_pad_zones(*inputs, workers=2)
    result = pyogrio.read_dataframe(inputs[-1], layer=ALL_PAD_PRODUCT).set_index("feature_id")
    assert set(result.index) == {"01001_2", "01003_2", "01001_3", "01001_5"}
    assert result.index.is_unique
    assert result.loc["01001_2", "Mang_Type"] == result.loc["01001_3", "Mang_Type"] == "TRIB"
    assert result.loc["01001_2", "GAP_Sts"] == "4"
    assert pd.isna(result.loc["01001_3", "GAP_Sts"])
    assert pd.isna(result.loc["01001_5", "Mang_Type"])
    assert result.loc["01001_5", "GAP_Sts"] == "9"
    original = pyogrio.read_dataframe(inputs[0], layer="padus", fid_as_index=True)
    for feature in result.itertuples():
        source = original.loc[feature.OBJECTID]
        for field in original.columns.drop("geometry"):
            value = getattr(feature, field)
            assert (pd.isna(value) and pd.isna(source[field])) or value == source[field]
    assert result.loc["01001_2", "area_ha"] == pytest.approx(0.5)
    assert result.loc["01003_2", "area_ha"] == pytest.approx(0.5)
    assert result.loc["01001_2", "area_ha_land"] == pytest.approx(0.4)
    assert result.loc["01003_2", "county_land_share"] == pytest.approx(0.5)
    assert result.loc["01001_3", "area_ha"] == pytest.approx(0.06)
    assert result.loc["01001_5", "area_ha"] == pytest.approx(0.01)
    assert result.loc["01001_5", "area_ha_land"] == 0
    assert result.loc["01001_5", "county_land_share"] == 0
    assert result.loc["01001_2"].geometry.symmetric_difference(box(50, 0, 100, 100)).area == 0
    assert result.loc["01003_2"].geometry.symmetric_difference(box(100, 0, 150, 100)).area == 0
    assert provenance["output"]["rows"] == 4
    assert provenance["inputs"]["padus"]["source_feature_id"] == "fid"
    assert "Own_Name" in provenance["inputs"]["padus"]["source_fields"]


def test_explicit_original_objectid_and_multipart_survive(inputs):
    """Prepared sources retain original IDs and multipart records within a county."""
    padus = inputs[0]
    padus.unlink()
    write_layer(padus, {"OBJECTID": [909], "Mang_Type": ["TRIB"], "Own_Name": ["Nation"]}, [
        MultiPolygon([box(10, 10, 20, 20), box(30, 30, 40, 40), box(120, 10, 130, 20)]),
    ])
    prepare_all_pad_zones(*inputs)
    result = pyogrio.read_dataframe(inputs[-1], layer=ALL_PAD_PRODUCT).set_index("feature_id")
    assert set(result.index) == {"01001_909", "01003_909"}
    assert result.loc["01001_909", "OBJECTID"] == 909
    assert len(result.loc["01001_909"].geometry.geoms) == 2
    assert result.loc["01001_909", "area_ha"] == pytest.approx(0.02)


@pytest.mark.parametrize("field", ["feature_id", "GEOID", "AREA_HA", "county_land_share"])
def test_source_field_collisions_fail_without_output(inputs, field):
    """Never overwrite an original attribute with a calculated field."""
    padus = inputs[0]
    padus.unlink()
    write_layer(padus, {"Mang_Type": ["TRIB"], field: ["source value"]}, [box(10, 10, 20, 20)])
    with pytest.raises(ValueError, match="collide"):
        prepare_all_pad_zones(*inputs)
    assert not inputs[-1].exists()


def test_empty_conus_intersection_keeps_original_schema(inputs):
    """An all-outside source still produces a usable empty product schema."""
    padus = inputs[0]
    padus.unlink()
    write_layer(padus, {"Mang_Type": ["TRIB"], "Own_Name": ["Outside"]}, [box(400, 10, 450, 20)])
    provenance = prepare_all_pad_zones(*inputs)
    result = pyogrio.read_dataframe(inputs[-1], layer=ALL_PAD_PRODUCT)
    assert result.empty
    assert set(result.columns) == {
        "Mang_Type", "Own_Name", "OBJECTID", "feature_id", "GEOID", "area_ha", "area_ha_land",
        "county_land_share", "geometry",
    }
    assert provenance["output"]["rows"] == 0


def test_land_mask_must_reconcile_with_census_aland(inputs):
    """Bad land masks cannot silently distort allocation shares."""
    masks = inputs[4]
    masks.unlink()
    write_layer(masks, {"GEOID": ["01001", "01003"]}, [box(0, 0, 100, 75), box(100, 0, 200, 80)])
    with pytest.raises(ValueError, match="differs from ALAND"):
        prepare_all_pad_zones(*inputs)
    assert not inputs[-1].exists()


def test_contained_crossing_and_lake_features_measure_land_separately(inputs):
    """Lake holes affect allocation area without cutting water from the feature."""
    padus, _, counties, _, masks, _, output = inputs
    for path in (padus, counties, masks):
        path.unlink()
    county = box(0, 0, 100, 100)
    lake = box(40, 40, 60, 60)
    write_layer(counties, {"GEOID": ["01001"], "ALAND": [9600]}, [county])
    write_layer(masks, {"GEOID": ["01001"]}, [county.difference(lake)])
    write_layer(padus, {"Mang_Type": ["TRIB"] * 3, "Unit_Nm": ["Land", "Crossing", "Lake"]}, [
        box(10, 10, 20, 20), box(30, 30, 70, 70), box(45, 45, 55, 55),
    ])
    prepare_all_pad_zones(*inputs)
    result = pyogrio.read_dataframe(output, layer=ALL_PAD_PRODUCT).set_index("Unit_Nm")
    assert result.loc["Land", "area_ha_land"] == pytest.approx(0.01)
    assert result.loc["Crossing", "area_ha_land"] == pytest.approx(0.12)
    assert result.loc["Crossing", "area_ha"] == pytest.approx(0.16)
    assert result.loc["Lake", "area_ha_land"] == 0
    assert result.loc["Lake", "area_ha"] == pytest.approx(0.01)
    assert result.loc["Crossing", "county_land_share"] == pytest.approx(0.125)
    assert result.loc["Crossing"].geometry.area == pytest.approx(1600)


def test_missing_county_mask_fails_before_writing(inputs):
    masks = inputs[4]
    masks.unlink()
    write_layer(masks, {"GEOID": ["01001"]}, [box(0, 0, 100, 80)])
    with pytest.raises(ValueError, match="GEOIDs must match"):
        prepare_all_pad_zones(*inputs)
    assert not inputs[-1].exists()


def test_output_is_never_overwritten(inputs):
    output = inputs[-1]
    output.parent.mkdir()
    output.write_bytes(b"existing output")
    with pytest.raises(FileExistsError):
        prepare_all_pad_zones(*inputs)
    assert output.read_bytes() == b"existing output"


def test_invalid_source_ids_fail(inputs):
    padus = inputs[0]
    padus.unlink()
    write_layer(padus, {"OBJECTID": [1, 1], "Mang_Type": ["TRIB", "TRIB"]}, [
        box(10, 10, 20, 20), box(30, 30, 40, 40),
    ])
    with pytest.raises(ValueError, match="OBJECTID values must be unique"):
        prepare_all_pad_zones(*inputs)
    assert not inputs[-1].exists()
