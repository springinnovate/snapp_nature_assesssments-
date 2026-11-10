"""Validate the Census mask builder with small, local Shapefile archives."""

from io import BytesIO
import hashlib
import json
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import box

import assessment.workflows.prepare_census_county_land_mask as builder


def write_faces(directory, geoid, shapes, *, flags=None, areas=None, ids=None, state=None):
    """Write a small ZIP with the same core file and field names as Census.

    Args:
        directory: Fixture directory for the zipped Shapefile.
        geoid: The county identifier used in the archive name.
        shapes: Polygon geometries in square-metre units.
        flags: Optional land/water classification for each face.
        areas: Optional ATOTAL values to replace the known geometric areas.
        ids: Optional TFIDs to replace the generated distinct identifiers.
        state: Optional state code to replace the county's actual state code.

    Returns:
        Path to the synthetic archive.
    """
    stem = f"tl_2024_{geoid}_faces"
    shp_dir = directory / stem
    shp_dir.mkdir(exist_ok=True)
    frame = gpd.GeoDataFrame({
        "STATEFP": [state or geoid[:2]] * len(shapes), "COUNTYFP": [geoid[2:]] * len(shapes),
        "TFID": list(range(100, 100 + len(shapes))) if ids is None else ids,
        "LWFLAG": ["L"] * len(shapes) if flags is None else flags,
        "ATOTAL": [int(shape.area) for shape in shapes] if areas is None else areas,
    }, geometry=shapes, crs=builder.AREA_CRS)
    pyogrio.write_dataframe(frame, shp_dir / f"{stem}.shp")
    archive = directory / f"{stem}.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as target:
        for member in sorted(shp_dir.glob(f"{stem}.*")):
            target.write(member, arcname=member.name)
    return archive


@pytest.fixture
def inputs(tmp_path):
    """Create a two-county CONUS mask input plus an out-of-scope county.

    Args:
        tmp_path: Fresh test directory supplied by pytest.

    Returns:
        CLI arguments, output path, and the editable archive test fixture.
    """
    county_file = tmp_path / "counties.gpkg"
    counties = gpd.GeoDataFrame({
        "GEOID": ["11001", "02020", "01001"],
        "ALAND": [1_000_000, 1_000_000, 1_200_000],
        "AWATER": [0, 0, 800_000],
    }, geometry=[box(2000, 0, 3000, 1000), box(3000, 0, 4000, 1000), box(0, 0, 2000, 1000)],
        crs=builder.AREA_CRS)
    pyogrio.write_dataframe(counties, county_file, layer="counties")
    faces_dir = tmp_path / "faces cache"
    faces_dir.mkdir()
    shapes = [box(x, 0, x + 300, 1000) for x in range(0, 1200, 300)] + [box(1200, 0, 2000, 1000)]
    write_faces(faces_dir, "01001", shapes, flags=["L", "I", "G", "S", "P"])
    write_faces(faces_dir, "11001", [box(2000, 0, 3000, 1000)])
    output = tmp_path / "publish here" / "mask.gpkg"
    args = ["--faces-dir", str(faces_dir), "--counties", str(county_file), "--output", str(output), "--workers", "2"]
    return args, output, faces_dir, shapes


def test_build_uses_census_land_flags_and_records_source_hashes(inputs, monkeypatch):
    """Reconcile official area attributes and preserve the nonperennial land faces."""
    args, output, faces_dir, _ = inputs
    monkeypatch.setattr(builder, "BATCH_COUNTIES", 1)
    builder.main(args)
    mask = pyogrio.read_dataframe(output, layer="county_land_mask").set_index("GEOID")
    assert list(mask.index) == ["01001", "11001"]
    assert mask.crs.to_epsg() == 6933
    assert mask.geometry.is_valid.all() and mask.geom_type.eq("MultiPolygon").all()
    assert mask.loc["01001", "geometry"].equals(box(0, 0, 1200, 1000))
    assert mask.loc["01001", "geometry_area_m2"] == 1_200_000
    assert mask.loc["01001", "land_face_count"] == 4
    assert mask.loc["01001", "water_face_count"] == 1
    assert list(mask["land_atotal_m2"]) == list(mask["ALAND"])
    assert list(mask["water_atotal_m2"]) == list(mask["AWATER"])
    assert list(mask["area_error_pct"]) == [0, 0]
    assert list(mask["outside_county_m2"]) == [0, 0]

    qa = pd.read_csv(output.with_name("mask_qa.csv"), dtype={"GEOID": "string"}).set_index("GEOID")
    pd.testing.assert_frame_equal(pd.DataFrame(mask.drop(columns="geometry")), qa, check_dtype=False,
                                  check_index_type=False)
    manifest = json.loads(output.with_name("mask_manifest.json").read_text(encoding="utf-8"))
    assert manifest["geographic_scope"] == "CONUS" and manifest["vintage"] == 2024
    assert manifest["land_flags"] == ["G", "I", "L", "S"] and manifest["water_flag"] == "P"
    assert manifest["output"]["rows"] == 2
    assert manifest["observed_max_abs_area_error_pct"] == 0
    for geoid in mask.index:
        archive = faces_dir / f"tl_2024_{geoid}_faces.zip"
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert mask.loc[geoid, "source_archive_sha256"] == checksum
        assert manifest["source_archives"][geoid] == {"filename": archive.name, "sha256": checksum,
                                                       "bytes": archive.stat().st_size}


@pytest.mark.parametrize(("defect", "message"), [
    ("wrong_county", "incorrect or missing STATEFP/COUNTYFP"),
    ("duplicate_tfid", "duplicate TFID"),
    ("bad_flag", "unrecognized or missing LWFLAG"),
    ("land_area", "ATOTAL land/water.*differs from ALAND/AWATER"),
    ("water_area", "ATOTAL land/water.*differs from ALAND/AWATER"),
    ("geom_area", "geometry fails.*ALAND tolerance"),
    ("overlap", "overlap 100000"),
    ("outside", "outside county 300000"),
])
def test_rejects_wrong_or_inconsistent_archive_without_publishing(inputs, defect, message):
    """Fail before publishing any file if a source county fails validation."""
    args, output, faces_dir, shapes = inputs
    options = {"flags": ["L", "I", "G", "S", "P"]}
    if defect == "wrong_county":
        options["state"] = "02"
    elif defect == "duplicate_tfid":
        options["ids"] = [1, 2, 3, 3, 5]
    elif defect == "bad_flag":
        options["flags"][2] = "X"
    else:
        options["areas"] = [int(shape.area) for shape in shapes]
        if defect == "land_area":
            options["areas"][0] += 1
        elif defect == "water_area":
            options["areas"][4] += 1
        elif defect == "geom_area":
            shapes[0] = box(0, 0, 200, 1000)
        elif defect == "overlap":
            shapes[0] = box(100, 0, 400, 1000)
        else:
            shapes[0] = box(0, -1000, 300, 0)
    write_faces(faces_dir, "01001", shapes, **options)
    with pytest.raises(ValueError, match=message):
        builder.main(args)
    assert not list(output.parent.iterdir())


def test_missing_archive_requires_opt_in_and_download_is_reusable(inputs, monkeypatch):
    """Download only on request, then reuse the cached ZIP without networking."""
    args, output, faces_dir, _ = inputs
    archive = faces_dir / "tl_2024_11001_faces.zip"
    response = archive.read_bytes()
    archive.unlink()
    calls = []
    delays = []

    def fake_download(url, *, timeout):
        """Return the known local ZIP as the response for one allowed URL."""
        calls.append(url)
        assert timeout > 0
        return BytesIO(b"<html>temporary proxy rejection</html>" if len(calls) == 1 else response)

    monkeypatch.setattr(builder, "urlopen", fake_download)
    monkeypatch.setattr(builder, "sleep", delays.append)
    with pytest.raises(FileNotFoundError, match="Missing 1 2024 faces ZIPs"):
        builder.main(args)
    assert calls == [] and not output.parent.exists()

    builder.main([*args, "--download-missing"])
    assert calls == [builder.FACES_URL + archive.name] * 2
    assert delays == [1]
    assert archive.read_bytes() == response
    builder.main([*args, "--output", str(output.with_name("second_mask.gpkg")), "--download-missing"])
    assert len(calls) == 2


def test_refuses_to_overwrite_existing_sidecar(inputs):
    """Protect all three destination paths even if no GeoPackage exists yet."""
    args, output, _, _ = inputs
    output.parent.mkdir()
    sidecar = output.with_name("mask_qa.csv")
    sidecar.write_text("prior file", encoding="utf-8")
    with pytest.raises(FileExistsError, match="mask_qa.csv"):
        builder.main(args)
    assert list(output.parent.iterdir()) == [sidecar]
    assert sidecar.read_text(encoding="utf-8") == "prior file"


def test_shared_download_validates_zip_and_schema_before_caching(inputs, monkeypatch):
    """Reuse the downloader for an alternate product with stronger validation."""
    _, _, faces_dir, _ = inputs
    source = faces_dir / "tl_2024_11001_faces.zip"
    target = faces_dir / "other cache" / source.name
    payload = source.read_bytes()
    calls, delays = [], []
    read_info = builder.pyogrio.read_info
    testzip = ZipFile.testzip

    def download(url, *, timeout):
        """Serve the local ZIP without publishing it before validation."""
        assert not target.exists()
        calls.append((url, timeout))
        return BytesIO(payload)

    def verify_members(archive):
        """Simulate one corrupt download, then perform real CRC checks."""
        return source.stem + ".dbf" if len(calls) == 1 else testzip(archive)

    def inspect_schema(path):
        """Simulate one mismatched schema before returning real metadata."""
        info = read_info(path)
        if len(calls) == 2:
            info["fields"] = []
        return info

    monkeypatch.setattr(builder, "urlopen", download)
    monkeypatch.setattr(builder, "sleep", delays.append)
    monkeypatch.setattr(ZipFile, "testzip", verify_members)
    monkeypatch.setattr(builder.pyogrio, "read_info", inspect_schema)
    builder.download_archive(target, "11001", source_url="https://example.invalid/other", timeout=60,
                             validate=lambda path: builder.county_shapefile(path, source.stem, fields=["TFID"]))

    assert calls == [("https://example.invalid/other/" + source.name, 60)] * 3
    assert delays == [1, 2]
    assert target.read_bytes() == payload
