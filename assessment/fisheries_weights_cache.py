"""Content-addressed, validated spatial measurements for fisheries allocations."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
import zlib

import geopandas as gpd
import numpy as np
import pyogrio
import pyproj
import shapely
from tqdm import tqdm

from assessment.build_county_ecosystem_service_valuations_2026 import AREA_CRS, sha256


SPATIAL_SOURCES = ("coastline_lines", "freshwater_polygons", "great_lakes_features",
                   "inland_fisheries_exclusions")
_SCHEMA = 1
_COLUMNS = ["marine_shoreline_km", "inland_water_area_ha"]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _geometry_signature(frame):
    """Hash geometry, CRS and ordered row keys, ignoring report-only attributes."""
    keys = [field for field in ("GEOID", "Mang_Type") if field in frame]
    digest = hashlib.sha256(_json({
        "crs": frame.crs.to_wkt() if frame.crs else None,
        "keys": frame[keys].astype("string").to_numpy(dtype=object, na_value=None).tolist(),
    }).encode())
    # Stream geometries rather than materializing another national WKB copy.
    for geometry in frame.geometry:
        wkb = shapely.to_wkb(geometry, byte_order=1)
        if wkb is None:
            digest.update(b"missing")
        else:
            digest.update(len(wkb).to_bytes(8, "little"))
            digest.update(wkb)
    return {"rows": len(frame), "sha256": digest.hexdigest()}


def _projection_resources():
    """Fingerprint local coordinate databases/grids, including user-installed ones."""
    directories = {*pyproj.datadir.get_data_dir().split(os.pathsep),
                   pyproj.datadir.get_user_data_dir(create=False)}
    return {str(path): sha256(path) for directory in sorted(directories)
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


class FisheriesWeightsCache:
    """Reuse measurements only when their complete spatial identity matches.

    A cache stores raw kilometres/hectares, never state values or adjusted
    dollars. Source files are hashed on every lookup. Monetary amounts and
    report columns are deliberately absent from the identity; supplied state
    keys are included because they determine which counties are measured.
    """

    def __init__(self, directory, zones, paths, totals):
        self.paths = {name: Path(paths[name]) for name in SPATIAL_SOURCES}
        self.stamps = self._stamps()
        sources = {name: sha256(path) for name, path in tqdm(
            self.paths.items(), desc="Hash fisheries spatial sources", unit="file")}
        self._assert_unchanged()
        package = Path(__file__).parent
        self.identity = {
            "schema": _SCHEMA, "columns": _COLUMNS, "area_crs": AREA_CRS,
            "geometry": {product: _geometry_signature(frame) for product, frame in tqdm(
                zones.items(), desc="Fingerprint fisheries zones", unit="product")},
            "sources_sha256": sources,
            "measured_states": {service: sorted(values.index.tolist()) for service, values in totals.items()},
            "code_sha256": {name: sha256(package / name) for name in (
                "assessment_fisheries.py", "assessment_geography.py", "geometry_utils.py",
                "fisheries_weights_cache.py")},
            "projection_sha256": _projection_resources(),
            "runtime": {
                "shapely": shapely.__version__, "geos": shapely.geos_version_string,
                "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str,
                "proj_network": pyproj.network.is_network_enabled(),
                "proj_data": pyproj.datadir.get_data_dir(),
                "pyogrio": pyogrio.__version__, "gdal": list(pyogrio.__gdal_version__),
                "geopandas": gpd.__version__, "numpy": np.__version__,
            },
        }
        self.key = hashlib.sha256(_json(self.identity).encode()).hexdigest()
        self.path = Path(directory) / f"{self.key}.npz"
        self.report = {"status": "miss", "key": self.key, "path": str(self.path),
                       "reason": "not_found", "sources_sha256": sources}

    def _stamps(self):
        # GDAL can see committed edits in a WAL that are absent from the main
        # GeoPackage bytes. Require stable, checkpointed inputs for hashing.
        for path in self.paths.values():
            if path.suffix.casefold() == ".gpkg":
                for suffix in ("-wal", "-journal"):
                    sidecar = Path(str(path) + suffix)
                    if sidecar.is_file() and sidecar.stat().st_size:
                        raise ValueError(
                            f"Fisheries source has an active SQLite sidecar: {sidecar}. "
                            "Close its writer and checkpoint the GeoPackage before running.")
        return {name: (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                for name, path in self.paths.items() for stat in (path.stat(),)}

    def _assert_unchanged(self):
        if self._stamps() != self.stamps:
            raise ValueError("Fisheries spatial sources changed during measurement/cache validation")

    def load(self):
        """Return validated arrays or a cache miss; never deserialize pickle."""
        if not self.path.is_file():
            return None
        try:
            with np.load(self.path, allow_pickle=False) as saved:
                products = self.identity["geometry"]
                if set(saved.files) != {*products, "metadata"}:
                    raise ValueError("Cache product set differs from the prepared zones")
                metadata = json.loads(str(saved["metadata"].item()))
                if metadata["key"] != self.key or metadata["identity"] != self.identity:
                    raise ValueError("Cache spatial identity does not match")
                weights = {}
                for product, signature in products.items():
                    array = saved[product]
                    if (array.dtype != np.dtype("float64") or array.shape != (signature["rows"], 2)
                            or not np.isfinite(array).all() or (array < 0).any()
                            or hashlib.sha256(array.tobytes()).hexdigest() != metadata["weights_sha256"][product]):
                        raise ValueError(f"Invalid cached fisheries weights for {product}")
                    weights[product] = array.copy()
        except (ValueError, KeyError, TypeError, OSError, EOFError, zipfile.BadZipFile, zlib.error):
            self.report["reason"] = "invalid"
            print(f"Ignoring invalid fisheries weights cache: {self.path}", flush=True)
            return None
        self._assert_unchanged()
        self.report.update(status="hit", reason="matching_spatial_inputs")
        print(f"Reuse fisheries weights: {self.path.name}", flush=True)
        return weights

    def save(self, weights):
        """Publish one complete artifact atomically after allocation QA passes."""
        self._assert_unchanged()
        if (_projection_resources() != self.identity["projection_sha256"]
                or any(sha256(Path(__file__).parent / name) != digest
                       for name, digest in self.identity["code_sha256"].items())):
            raise ValueError("Fisheries measurement code or projection resources changed during the run")
        if self.report["status"] == "hit":
            return
        # Check actual bytes again before publishing a potentially long run.
        if any(sha256(path) != self.identity["sources_sha256"][name]
               for name, path in self.paths.items()):
            raise ValueError("Fisheries spatial sources changed during measurement")
        self._assert_unchanged()
        metadata = {"key": self.key, "identity": self.identity, "weights_sha256": {
            product: hashlib.sha256(values.tobytes()).hexdigest() for product, values in weights.items()}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="weights-", dir=self.path.parent) as temporary:
            stage = Path(temporary) / self.path.name
            np.savez_compressed(stage, metadata=np.asarray(_json(metadata)), **weights)
            stage.replace(self.path)
        self.report["written"] = True
        print(f"Saved fisheries weights: {self.path.name}", flush=True)
