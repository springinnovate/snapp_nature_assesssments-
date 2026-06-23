"""Combine individual zonal-stat outputs into final assessment datasets.

The zonal statistics workflow writes one CSV and/or GeoPackage per metric
family. Those outputs may be grouped into subdirectories, with each subdirectory
treated as its own final-output project. This final step joins those
metric-family outputs into three deliverable datasets:

- counties
- PAD-US all lands cut by county
- PAD-US public lands cut by county

Each combined dataset is keyed by `GEOID`. Shared fields such as county names or
state codes are kept once. If the same field appears in multiple inputs with
different values for the same `GEOID`, the script raises an error instead of
silently choosing one.

The prepared recreation value by county is joined after the zonal-stat outputs.
It is a county-level metric, so PAD-US final outputs receive the value for the
matching county `GEOID`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

DEFAULT_RESULTS_DIR = Path("data/analysis_results/zonal_statistics")
DEFAULT_OUTPUT_DIR = Path("data/analysis_results/combined")
DEFAULT_RECREATION_DIR = Path("data/analysis_inputs/zonal_units/recreation_by_county")
JOIN_FIELD = "GEOID"
RECREATION_VALUE_STEM = "recreation_value_by_county"
RECREATION_VALUE_FIELD = "proportional_recreation_val_2024"
NLCD_VALID_AREA_PREFIX = "area_ha_valid_reclassified_NLCD2023_"
NLCD_PROPORTION_PREFIX = "proportion_valid_nonzero_reclassified_NLCD2023_"
NLCD_CLASS_AREA_FIELDS = (
    ("forests", "area_ha_nlcd_forests"),
    ("grasslands", "area_ha_nlcd_grasslands"),
    ("shrubland", "area_ha_nlcd_shrubland"),
    ("water_snow", "area_ha_nlcd_water_snow"),
    ("wetlands", "area_ha_nlcd_wetlands"),
)
INPUT_TIMESTAMP_FORMATS = ("%Y%m%d_%H%M%S", "%Y_%m_%d_%H_%M_%S")
OUTPUT_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S"
TIMESTAMP_SUFFIX = re.compile(
    r"_(?:\d{8}_\d{6}|\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$"
)


@dataclass(frozen=True)
class ResultGroup:
    """Expected final-output group and its source zonal-stat job stems.

    Attributes:
        output_stem: Stem used for final CSV/GPKG outputs and GPKG layer names.
        job_stems: Expected zonal-stat output stems to combine.
    """

    output_stem: str
    job_stems: tuple[str, ...]


@dataclass(frozen=True)
class ResultProject:
    """Directory-based final-output project.

    Attributes:
        output_stem: Stem used for final CSV/GPKG outputs and GPKG layer names.
        job_prefixes: Filename prefixes that belong to the project directory.
    """

    output_stem: str
    job_prefixes: tuple[str, ...]


RESULT_GROUPS = {
    "counties": ResultGroup(
        output_stem="counties_combined",
        job_stems=(
            "counties_ecosystem_services",
            "counties_masks",
            "counties_area",
            "counties_freshwater_area",
            "counties_coastline_length",
        ),
    ),
    "padus_all_lands": ResultGroup(
        output_stem="padus_all_lands_combined",
        job_stems=(
            "pad_ecosystem_services",
            "pad_masks",
            "pad_area",
            "pad_freshwater_area",
            "pad_coastline_length",
        ),
    ),
    "padus_public_lands": ResultGroup(
        output_stem="padus_public_lands_combined",
        job_stems=(
            "public_ecosystem_services",
            "public_masks",
            "public_area",
            "public_freshwater_area",
            "public_coastline_length",
        ),
    ),
}

RESULT_PROJECTS = {
    "counties": ResultProject(
        output_stem="counties_combined",
        job_prefixes=("counties_",),
    ),
    "padus_all_lands": ResultProject(
        output_stem="padus_all_lands_combined",
        job_prefixes=("pad_",),
    ),
    "padus_public_lands": ResultProject(
        output_stem="padus_public_lands_combined",
        job_prefixes=("public_",),
    ),
}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Combine latest zonal-stat CSV/GPKG outputs into final county, "
            "PAD-US all-land, and PAD-US public-land deliverables."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing individual zonal_stats_toolkit outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where final combined outputs should be written.",
    )
    parser.add_argument(
        "--recreation-dir",
        type=Path,
        default=DEFAULT_RECREATION_DIR,
        help=(
            "Directory containing prepared recreation value by county "
            "GeoPackages."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny CSV combine and conflict-detection smoke test.",
    )
    parser.add_argument(
        "--check-gpkg-geometry",
        action="store_true",
        help=(
            "Compare repeated GPKG geometries before joining. This is slow "
            "for large layers and is disabled by default."
        ),
    )
    return parser.parse_args()


def _input_timestamp(path: Path, job_stem: str) -> tuple[datetime, float]:
    """Return sortable timestamp information for a job output path.

    Args:
        path: Candidate output path.
        job_stem: Expected job output stem before any timestamp suffix.

    Returns:
        Parsed embedded timestamp plus file mtime. Untimestamped files sort
        before timestamped files, with mtime as a final tie breaker.
    """
    suffix = path.stem.removeprefix(job_stem).lstrip("_")
    for timestamp_format in INPUT_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(suffix, timestamp_format), path.stat().st_mtime
        except ValueError:
            pass

    if not suffix:
        return datetime.min, path.stat().st_mtime
    return datetime.min, path.stat().st_mtime


def _latest_job_output(
    results_dir: Path,
    job_stem: str,
    suffix: str,
) -> Path | None:
    """Find the latest timestamped output for one job and file type.

    Args:
        results_dir: Project directory containing zonal-stat outputs.
        job_stem: Expected job output stem.
        suffix: File suffix such as `.csv` or `.gpkg`.

    Returns:
        Latest matching path, or None if no matching path exists.
    """
    candidates = [
        path
        for path in results_dir.glob(f"{job_stem}*{suffix}")
        if path.stem == job_stem or path.stem.startswith(f"{job_stem}_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _input_timestamp(path, job_stem))


def _latest_recreation_value_path(recreation_dir: Path) -> Path:
    """Find the latest prepared recreation value by county GeoPackage.

    Args:
        recreation_dir: Directory containing recreation value outputs.

    Returns:
        Latest matching recreation value GeoPackage.

    Raises:
        FileNotFoundError: If no matching recreation value output exists.
    """
    path = _latest_job_output(recreation_dir, RECREATION_VALUE_STEM, ".gpkg")
    if path is None:
        raise FileNotFoundError(
            f"No {RECREATION_VALUE_STEM} GeoPackage found in {recreation_dir}."
        )
    return path


def _read_recreation_value(path: Path) -> pd.DataFrame:
    """Read the prepared recreation value fields used by final outputs.

    Args:
        path: Prepared recreation value GeoPackage.

    Returns:
        DataFrame containing `GEOID` and the recreation value field.
    """
    recreation = gpd.read_file(path, ignore_geometry=True)
    _validate_join_field(recreation, path)
    if RECREATION_VALUE_FIELD not in recreation.columns:
        raise ValueError(f"{path} does not contain required field {RECREATION_VALUE_FIELD}.")

    recreation = recreation[[JOIN_FIELD, RECREATION_VALUE_FIELD]].copy()
    recreation[JOIN_FIELD] = recreation[JOIN_FIELD].astype(str)
    return recreation


def _latest_group_outputs(
    results_dir: Path,
    group: ResultGroup,
    suffix: str,
) -> list[Path] | None:
    """Find latest outputs for every expected job in a group.

    Args:
        results_dir: Directory containing zonal-stat outputs.
        group: Result group to resolve.
        suffix: File suffix such as `.csv` or `.gpkg`.

    Returns:
        Latest paths in expected job order, or None when no jobs have outputs
        for this suffix.

    Raises:
        FileNotFoundError: If only some expected jobs have outputs.
    """
    paths_by_job = {
        job_stem: _latest_job_output(results_dir, job_stem, suffix)
        for job_stem in group.job_stems
    }
    found = {job_stem: path for job_stem, path in paths_by_job.items() if path}
    if not found:
        return None
    if len(found) != len(group.job_stems):
        missing = sorted(set(group.job_stems) - set(found))
        raise FileNotFoundError(
            f"Missing {suffix} output(s) for {group.output_stem}: "
            + ", ".join(missing)
        )
    return [paths_by_job[job_stem] for job_stem in group.job_stems]


def _job_stem_from_output(path: Path) -> str:
    """Remove a runner timestamp suffix from an output filename stem.

    Args:
        path: Zonal-stat output path.

    Returns:
        Job stem without a trailing timestamp.
    """
    return TIMESTAMP_SUFFIX.sub("", path.stem)


def _latest_project_outputs(
    project_dir: Path,
    project: ResultProject,
    suffix: str,
) -> list[Path] | None:
    """Find the latest output for every job present in one project directory.

    Args:
        project_dir: Directory containing one project's zonal-stat outputs.
        project: Project naming rule.
        suffix: File suffix such as `.csv` or `.gpkg`.

    Returns:
        Latest paths sorted by job stem, or None when the project has no files
        for this suffix.
    """
    paths_by_job_stem = {}
    for path in project_dir.glob(f"*{suffix}"):
        if not path.stem.startswith(project.job_prefixes):
            continue
        job_stem = _job_stem_from_output(path)
        paths_by_job_stem.setdefault(job_stem, []).append(path)

    if not paths_by_job_stem:
        return None

    latest_paths = []
    for job_stem, paths in sorted(paths_by_job_stem.items()):
        latest_paths.append(max(paths, key=lambda path: _input_timestamp(path, job_stem)))
    return latest_paths


def _group_results_dir(results_dir: Path, group_name: str) -> Path:
    """Return the directory that contains one result group's job outputs.

    Args:
        results_dir: Root zonal-stat output directory.
        group_name: Result group key such as `counties`.

    Returns:
        Child project directory when present, otherwise the root results
        directory for backwards compatibility with flat output layouts.
    """
    project_dir = results_dir / group_name
    if project_dir.is_dir():
        return project_dir
    return results_dir


def _has_project_directories(results_dir: Path) -> bool:
    """Return whether the result root contains known project directories.

    Args:
        results_dir: Root zonal-stat output directory.

    Returns:
        True when at least one known project directory exists.
    """
    return any((results_dir / project_name).is_dir() for project_name in RESULT_PROJECTS)


def _validate_join_field(frame: pd.DataFrame, path: Path) -> None:
    """Validate that an input contains unique GEOID values.

    Args:
        frame: Input tabular data.
        path: Source path used in error messages.

    Raises:
        ValueError: If `GEOID` is missing or duplicated.
    """
    if JOIN_FIELD not in frame.columns:
        raise ValueError(f"{path} does not contain required field {JOIN_FIELD}.")
    duplicated = frame[frame[JOIN_FIELD].duplicated()][JOIN_FIELD].astype(str)
    if not duplicated.empty:
        examples = ", ".join(duplicated.head(10))
        raise ValueError(f"{path} contains duplicate {JOIN_FIELD} values: {examples}")


def _aligned_series_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    """Compare two aligned series while treating paired nulls as equal.

    Args:
        left: Existing combined values.
        right: Incoming values aligned to `left`.

    Returns:
        Boolean series indicating row-wise equality.
    """
    return left.eq(right) | (left.isna() & right.isna())


def _conflicting_fields(
    combined: pd.DataFrame,
    incoming: pd.DataFrame,
    incoming_path: Path,
    ignore_fields: set[str] | None = None,
) -> list[str]:
    """Find duplicate non-geometry fields with conflicting values.

    Args:
        combined: Existing combined frame indexed by `GEOID`.
        incoming: Incoming frame indexed by `GEOID`.
        incoming_path: Source path used in error messages.
        ignore_fields: Shared fields to ignore during comparison.

    Returns:
        Shared field names whose values conflict.

    Raises:
        ValueError: If the incoming frame contains a different `GEOID` set than
            the combined frame.
    """
    ignore_fields = ignore_fields or set()
    extra_keys = incoming.index.difference(combined.index)
    if not extra_keys.empty:
        examples = ", ".join(extra_keys.astype(str).tolist()[:10])
        raise ValueError(f"{incoming_path} contains unexpected {JOIN_FIELD}: {examples}")

    missing_keys = combined.index.difference(incoming.index)
    if not missing_keys.empty:
        examples = ", ".join(missing_keys.astype(str).tolist()[:10])
        raise ValueError(f"{incoming_path} is missing {JOIN_FIELD}: {examples}")

    incoming = incoming.reindex(combined.index)
    conflicts = []
    shared_fields = (
        set(combined.columns).intersection(incoming.columns) - {JOIN_FIELD} - ignore_fields
    )
    for field_name in sorted(shared_fields):
        if not _aligned_series_equal(combined[field_name], incoming[field_name]).all():
            conflicts.append(field_name)
    return conflicts


def _join_metric_frame(
    combined: pd.DataFrame,
    incoming: pd.DataFrame,
    incoming_path: Path,
    ignore_fields: set[str] | None = None,
) -> pd.DataFrame:
    """Join new metric columns after checking duplicate-field conflicts.

    Args:
        combined: Existing combined frame indexed by `GEOID`.
        incoming: Incoming frame indexed by `GEOID`.
        incoming_path: Source path used in error messages.
        ignore_fields: Shared fields to ignore during duplicate comparison.

    Returns:
        Combined frame with only new incoming columns appended.

    Raises:
        ValueError: If shared fields conflict.
    """
    conflicts = _conflicting_fields(combined, incoming, incoming_path, ignore_fields)
    if conflicts:
        raise ValueError(
            f"{incoming_path} has conflicting duplicate field(s): "
            + ", ".join(conflicts)
        )

    incoming = incoming.reindex(combined.index)
    new_columns = [column for column in incoming.columns if column not in combined.columns]
    return combined.join(incoming[new_columns])


def _join_recreation_value(
    combined: pd.DataFrame,
    recreation: pd.DataFrame,
    recreation_path: Path,
    require_all_keys: bool,
) -> pd.DataFrame:
    """Join prepared county recreation value onto a combined output frame.

    Args:
        combined: Existing combined frame indexed by `GEOID`.
        recreation: Prepared recreation value frame indexed by `GEOID`.
        recreation_path: Recreation source path, used in error messages.
        require_all_keys: Whether every combined `GEOID` must be present in the
            recreation value frame.

    Returns:
        Combined frame with `proportional_recreation_val_2024` appended.

    Raises:
        ValueError: If the recreation metric is missing required keys, or if an
            existing recreation value field has conflicting values.
    """
    missing_keys = combined.index.difference(recreation.index)
    if require_all_keys and not missing_keys.empty:
        examples = ", ".join(missing_keys.astype(str).tolist()[:10])
        raise ValueError(
            f"{recreation_path} is missing {JOIN_FIELD} needed for recreation "
            f"value join: {examples}"
        )

    joined_values = (
        recreation[RECREATION_VALUE_FIELD].reindex(combined.index).fillna(0)
    )
    if RECREATION_VALUE_FIELD in combined.columns:
        if not _aligned_series_equal(
            combined[RECREATION_VALUE_FIELD], joined_values
        ).all():
            raise ValueError(
                f"{recreation_path} has conflicting duplicate field: "
                f"{RECREATION_VALUE_FIELD}"
            )
        return combined

    combined = combined.copy()
    combined[RECREATION_VALUE_FIELD] = joined_values
    return combined


def _add_recreation_value_to_output(
    combined: pd.DataFrame,
    project_name: str,
    recreation: pd.DataFrame,
    recreation_path: Path,
) -> pd.DataFrame:
    """Add prepared recreation value to one final output table.

    Args:
        combined: Combined CSV or GeoPackage frame.
        project_name: Final output project name.
        recreation: Prepared recreation value frame indexed by `GEOID`.
        recreation_path: Recreation source path, used in error messages.

    Returns:
        Combined frame with recreation value joined.
    """
    combined = combined.copy()
    combined[JOIN_FIELD] = combined[JOIN_FIELD].astype(str)
    combined = combined.set_index(JOIN_FIELD, drop=False)
    combined = _join_recreation_value(
        combined,
        recreation,
        recreation_path,
        require_all_keys=project_name == "counties",
    )
    return combined.reset_index(drop=True)


def _find_nlcd_mask_column(
    frame: pd.DataFrame,
    prefix: str,
    class_name: str,
) -> str | None:
    """Find one timestamped NLCD mask metric column.

    Args:
        frame: Combined output frame.
        prefix: Metric column prefix.
        class_name: NLCD class group name such as `forests`.

    Returns:
        Matching column name, or None when the class metric is absent.

    Raises:
        ValueError: If multiple matching columns are present.
    """
    column_prefix = f"{prefix}{class_name}"
    matches = [column for column in frame.columns if column.startswith(column_prefix)]
    if len(matches) > 1:
        raise ValueError(
            f"Expected one NLCD mask column matching {column_prefix}, "
            f"found {len(matches)}: {', '.join(matches)}"
        )
    if not matches:
        return None
    return matches[0]


def _derive_nlcd_class_area_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive stable NLCD class area fields and drop valid-area artifacts.

    Args:
        frame: Combined CSV or GeoPackage output frame.

    Returns:
        Output frame with `area_ha_nlcd_*` fields added and
        `area_ha_valid_reclassified_NLCD2023_*` fields removed.

    Raises:
        ValueError: If a land-cover class has only one of the area/proportion
            input fields.
    """
    frame = frame.copy()
    valid_area_columns_to_drop = []
    for class_name, output_field in NLCD_CLASS_AREA_FIELDS:
        valid_area_column = _find_nlcd_mask_column(
            frame,
            NLCD_VALID_AREA_PREFIX,
            class_name,
        )
        proportion_column = _find_nlcd_mask_column(
            frame,
            NLCD_PROPORTION_PREFIX,
            class_name,
        )
        if valid_area_column is None and proportion_column is None:
            continue
        if valid_area_column is None or proportion_column is None:
            raise ValueError(
                f"Cannot derive {output_field}; expected both "
                f"{NLCD_VALID_AREA_PREFIX}{class_name}* and "
                f"{NLCD_PROPORTION_PREFIX}{class_name}*."
            )

        frame[output_field] = frame[valid_area_column] * frame[proportion_column]
        valid_area_columns_to_drop.append(valid_area_column)

    if valid_area_columns_to_drop:
        frame = frame.drop(columns=valid_area_columns_to_drop)
    return frame


def _combine_csvs(paths: list[Path], progress_label: str) -> pd.DataFrame:
    """Combine one result family's CSV outputs.

    Args:
        paths: Source CSV paths in expected join order.
        progress_label: Label to show in progress output.

    Returns:
        Combined CSV dataframe.
    """
    tqdm.write(f"Reading CSV base for {progress_label}: {paths[0].name}")
    combined = pd.read_csv(paths[0], dtype={JOIN_FIELD: str})
    _validate_join_field(combined, paths[0])
    combined = combined.set_index(JOIN_FIELD, drop=False)

    for path in tqdm(
        paths[1:],
        desc=f"Join CSV {progress_label}",
        unit="file",
        leave=False,
    ):
        incoming = pd.read_csv(path, dtype={JOIN_FIELD: str})
        _validate_join_field(incoming, path)
        incoming = incoming.set_index(JOIN_FIELD, drop=False)
        combined = _join_metric_frame(combined, incoming, path)

    return combined.reset_index(drop=True)


def _geometry_conflicts(
    combined: gpd.GeoDataFrame,
    incoming: gpd.GeoDataFrame,
    incoming_path: Path,
) -> list[str]:
    """Find GEOID values whose geometries differ between two GeoDataFrames.

    Args:
        combined: Existing combined GeoDataFrame indexed by `GEOID`.
        incoming: Incoming GeoDataFrame indexed by `GEOID`.
        incoming_path: Source path used in error messages.

    Returns:
        GEOID examples with conflicting geometry.
    """
    incoming = incoming.reindex(combined.index)
    equal = combined.geometry.geom_equals(incoming.geometry)
    conflicts = combined.index[~equal.fillna(False)]
    if conflicts.empty:
        return []
    return conflicts.astype(str).tolist()[:10]


def _combine_gpkgs(
    paths: list[Path],
    progress_label: str,
    check_geometry: bool = False,
) -> gpd.GeoDataFrame:
    """Combine one result family's GeoPackage outputs.

    Args:
        paths: Source GeoPackage paths in expected join order.
        progress_label: Label to show in progress output.
        check_geometry: Whether to run expensive geometry equality checks
            against each incoming GeoPackage.

    Returns:
        Combined GeoDataFrame.
    """
    tqdm.write(f"Reading GPKG base for {progress_label}: {paths[0].name}")
    combined = gpd.read_file(paths[0])
    _validate_join_field(combined, paths[0])
    combined[JOIN_FIELD] = combined[JOIN_FIELD].astype(str)
    combined = combined.set_index(JOIN_FIELD, drop=False)

    for path in tqdm(
        paths[1:],
        desc=f"Join GPKG {progress_label}",
        unit="file",
        leave=False,
    ):
        if check_geometry:
            incoming = gpd.read_file(path)
        else:
            incoming = gpd.read_file(path, ignore_geometry=True)
        _validate_join_field(incoming, path)
        incoming[JOIN_FIELD] = incoming[JOIN_FIELD].astype(str)
        incoming = incoming.set_index(JOIN_FIELD, drop=False)

        if check_geometry:
            conflicts = _geometry_conflicts(combined, incoming, path)
            if conflicts:
                raise ValueError(
                    f"{path} has geometry conflicts for {JOIN_FIELD}: "
                    + ", ".join(conflicts)
                )
            incoming = incoming.drop(columns=incoming.geometry.name)

        combined = _join_metric_frame(
            combined,
            pd.DataFrame(incoming),
            path,
            ignore_fields={combined.geometry.name},
        )

    return combined.reset_index(drop=True)


def combine_outputs(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    recreation_dir: Path = DEFAULT_RECREATION_DIR,
    check_gpkg_geometry: bool = False,
) -> list[Path]:
    """Combine latest zonal-stat outputs into final CSV and GPKG datasets.

    Args:
        results_dir: Directory containing individual zonal-stat outputs.
        output_dir: Directory where final combined outputs should be written.
        recreation_dir: Directory containing prepared recreation value output.
        check_gpkg_geometry: Whether to run expensive geometry equality checks
            against repeated GPKG geometry columns.

    Returns:
        Written output paths.

    Raises:
        RuntimeError: If no complete CSV or GPKG result groups are available.
    """
    timestamp = datetime.now().strftime(OUTPUT_TIMESTAMP_FORMAT)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    recreation_path = _latest_recreation_value_path(recreation_dir)
    recreation_value = _read_recreation_value(recreation_path)
    recreation_value = recreation_value.set_index(JOIN_FIELD, drop=False)
    tqdm.write(f"Using recreation value: {recreation_path}")

    if _has_project_directories(results_dir):
        project_items = [
            (project_name, project)
            for project_name, project in RESULT_PROJECTS.items()
            if (results_dir / project_name).is_dir()
        ]
        for project_name, project in tqdm(
            project_items,
            desc="Combine result projects",
            unit="project",
        ):
            project_dir = results_dir / project_name

            csv_paths = _latest_project_outputs(project_dir, project, ".csv")
            if csv_paths is not None:
                output_path = output_dir / f"{project.output_stem}_{timestamp}.csv"
                combined_csv = _combine_csvs(csv_paths, project.output_stem)
                combined_csv = _add_recreation_value_to_output(
                    combined_csv,
                    project_name,
                    recreation_value,
                    recreation_path,
                )
                combined_csv = _derive_nlcd_class_area_fields(combined_csv)
                tqdm.write(f"Writing CSV {output_path}")
                combined_csv.to_csv(output_path, index=False)
                written_paths.append(output_path)

            gpkg_paths = _latest_project_outputs(project_dir, project, ".gpkg")
            if gpkg_paths is not None:
                output_path = output_dir / f"{project.output_stem}_{timestamp}.gpkg"
                combined_gpkg = _combine_gpkgs(
                    gpkg_paths,
                    project.output_stem,
                    check_geometry=check_gpkg_geometry,
                )
                combined_gpkg = _add_recreation_value_to_output(
                    combined_gpkg,
                    project_name,
                    recreation_value,
                    recreation_path,
                )
                combined_gpkg = _derive_nlcd_class_area_fields(combined_gpkg)
                tqdm.write(f"Writing GPKG {output_path}")
                combined_gpkg.to_file(
                    output_path,
                    layer=project.output_stem,
                    driver="GPKG",
                    index=False,
                )
                written_paths.append(output_path)
    else:
        for group_name, group in tqdm(
            RESULT_GROUPS.items(),
            desc="Combine result groups",
            unit="group",
        ):
            group_results_dir = _group_results_dir(results_dir, group_name)

            csv_paths = _latest_group_outputs(group_results_dir, group, ".csv")
            if csv_paths is not None:
                output_path = output_dir / f"{group.output_stem}_{timestamp}.csv"
                combined_csv = _combine_csvs(csv_paths, group.output_stem)
                combined_csv = _add_recreation_value_to_output(
                    combined_csv,
                    group_name,
                    recreation_value,
                    recreation_path,
                )
                combined_csv = _derive_nlcd_class_area_fields(combined_csv)
                tqdm.write(f"Writing CSV {output_path}")
                combined_csv.to_csv(output_path, index=False)
                written_paths.append(output_path)

            gpkg_paths = _latest_group_outputs(group_results_dir, group, ".gpkg")
            if gpkg_paths is not None:
                output_path = output_dir / f"{group.output_stem}_{timestamp}.gpkg"
                combined_gpkg = _combine_gpkgs(
                    gpkg_paths,
                    group.output_stem,
                    check_geometry=check_gpkg_geometry,
                )
                combined_gpkg = _add_recreation_value_to_output(
                    combined_gpkg,
                    group_name,
                    recreation_value,
                    recreation_path,
                )
                combined_gpkg = _derive_nlcd_class_area_fields(combined_gpkg)
                tqdm.write(f"Writing GPKG {output_path}")
                combined_gpkg.to_file(
                    output_path,
                    layer=group.output_stem,
                    driver="GPKG",
                    index=False,
                )
                written_paths.append(output_path)

    if not written_paths:
        raise RuntimeError(f"No complete zonal-stat result groups found in {results_dir}.")
    return written_paths


def _write_smoke_test_inputs(results_dir: Path) -> None:
    """Write tiny CSV inputs used by `--smoke-test`.

    Args:
        results_dir: Temporary result directory.
    """
    for group_name, group in RESULT_GROUPS.items():
        project_dir = results_dir / group_name
        project_dir.mkdir()
        for job_number, job_stem in enumerate(group.job_stems):
            rows = {
                JOIN_FIELD: ["001", "002"],
                "county_name": ["A", "B"],
                f"metric_{job_number}": [job_number, job_number + 1],
            }
            if job_stem.endswith("_masks"):
                for class_index, (class_name, _) in enumerate(NLCD_CLASS_AREA_FIELDS):
                    rows[
                        f"{NLCD_VALID_AREA_PREFIX}{class_name}_2026_01_01_00_00_00"
                    ] = [100.0, 200.0]
                    rows[
                        f"{NLCD_PROPORTION_PREFIX}{class_name}_2026_01_01_00_00_00"
                    ] = [0.1 * (class_index + 1), 0.2 * (class_index + 1)]
            frame = pd.DataFrame(rows)
            frame.to_csv(project_dir / f"{job_stem}_20260101_000000.csv", index=False)


def _write_smoke_test_recreation_input(recreation_dir: Path) -> None:
    """Write a tiny prepared recreation value GeoPackage for `--smoke-test`.

    Args:
        recreation_dir: Temporary recreation value directory.
    """
    recreation_dir.mkdir()
    recreation = gpd.GeoDataFrame(
        {
            JOIN_FIELD: ["001", "002", "003"],
            RECREATION_VALUE_FIELD: [10.0, 20.0, 30.0],
        },
        geometry=gpd.points_from_xy([0, 1, 2], [0, 1, 2]),
        crs="EPSG:5070",
    )
    recreation.to_file(
        recreation_dir / f"{RECREATION_VALUE_STEM}_2026_01_01_00_00_00.gpkg",
        layer=RECREATION_VALUE_STEM,
        driver="GPKG",
        index=False,
    )


def _validate_smoke_test_nlcd_fields(csv_outputs: list[Path]) -> None:
    """Validate derived NLCD class area fields written by the smoke test.

    Args:
        csv_outputs: Smoke-test CSV output paths.

    Raises:
        AssertionError: If the derived NLCD fields are missing or wrong, or if
            dropped valid-area artifact fields remain.
    """
    for csv_path in csv_outputs:
        frame = pd.read_csv(csv_path, dtype={JOIN_FIELD: str})
        artifact_columns = [
            column
            for column in frame.columns
            if column.startswith(NLCD_VALID_AREA_PREFIX)
        ]
        if artifact_columns:
            raise AssertionError(
                f"Smoke test retained NLCD valid-area artifact fields: "
                + ", ".join(artifact_columns)
            )

        for class_index, (_, output_field) in enumerate(NLCD_CLASS_AREA_FIELDS):
            if output_field not in frame.columns:
                raise AssertionError(f"Smoke test missing derived field {output_field}.")
            expected = [10.0 * (class_index + 1), 40.0 * (class_index + 1)]
            actual = frame[output_field].tolist()
            if any(abs(left - right) > 1e-9 for left, right in zip(actual, expected)):
                raise AssertionError(
                    f"Smoke test wrote unexpected values for {output_field}: "
                    f"{actual}"
                )


def _validate_smoke_test_recreation_values(csv_outputs: list[Path]) -> None:
    """Validate recreation values written by the smoke test.

    Args:
        csv_outputs: Smoke-test CSV output paths.

    Raises:
        AssertionError: If an expected recreation value is missing or wrong.
    """
    expected_values = {
        "counties_combined": {"001": 10.0, "002": 20.0},
        "padus_all_lands_combined": {"001": 10.0, "002": 20.0},
        "padus_public_lands_combined": {"001": 10.0, "002": 20.0},
    }
    for csv_path in csv_outputs:
        output_stem = TIMESTAMP_SUFFIX.sub("", csv_path.stem)
        expected = expected_values[output_stem]
        frame = pd.read_csv(csv_path, dtype={JOIN_FIELD: str})
        actual = frame.set_index(JOIN_FIELD)[RECREATION_VALUE_FIELD].to_dict()
        if actual != expected:
            raise AssertionError(
                f"Unexpected recreation values for {output_stem}: {actual}"
            )


def _run_smoke_test() -> None:
    """Run a lightweight CSV combine and conflict-detection smoke test."""
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        results_dir = tmpdir_path / "results"
        output_dir = tmpdir_path / "combined"
        recreation_dir = tmpdir_path / "recreation"
        results_dir.mkdir()
        _write_smoke_test_inputs(results_dir)
        _write_smoke_test_recreation_input(recreation_dir)

        written = combine_outputs(results_dir, output_dir, recreation_dir=recreation_dir)
        csv_outputs = [path for path in written if path.suffix == ".csv"]
        if len(csv_outputs) != len(RESULT_GROUPS):
            raise AssertionError("Smoke test did not write one CSV per result group.")
        _validate_smoke_test_nlcd_fields(csv_outputs)
        _validate_smoke_test_recreation_values(csv_outputs)

        conflict_path = (
            results_dir / "counties" / "counties_masks_20260102_000000.csv"
        )
        pd.DataFrame(
            {
                JOIN_FIELD: ["001", "002"],
                "county_name": ["changed", "B"],
                "metric_conflict": [1, 2],
            }
        ).to_csv(conflict_path, index=False)

        try:
            combine_outputs(results_dir, output_dir, recreation_dir=recreation_dir)
        except ValueError as error:
            if "conflicting duplicate field" not in str(error):
                raise
        else:
            raise AssertionError("Smoke test did not catch duplicate-field conflict.")


def main() -> None:
    """Run the final zonal-stat combine workflow."""
    args = _parse_args()
    if getattr(args, "smoke_test", False):
        _run_smoke_test()
        print("Smoke test passed.")
        return

    written_paths = combine_outputs(
        args.results_dir,
        args.output_dir,
        recreation_dir=args.recreation_dir,
        check_gpkg_geometry=args.check_gpkg_geometry,
    )
    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
