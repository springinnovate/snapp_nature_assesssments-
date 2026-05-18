"""Generate NLCD land-cover mask rasters from reclass CSV tables."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from multiprocessing import RLock
from os import cpu_count
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

DEFAULT_NLCD_RASTER_PATH = Path(
    "data/analysis_inputs/nlcd/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
)
DEFAULT_RECLASS_TABLE_DIR = Path("data/workflow_assets/landcover_reclass")
DEFAULT_OUTPUT_DIR = Path("data/analysis_inputs/masks")
OUTPUT_NODATA = 255
WINDOWS_PER_PROGRESS_UPDATE = 100


@dataclass(frozen=True)
class MaskJob:
    """A single mask raster generation job.

    Args:
        table_path: Reclass CSV path.
        output_path: Timestamped output raster path.
        progress_position: Fixed tqdm terminal row for this job.
    """

    table_path: Path
    output_path: Path
    progress_position: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate 0/1/nodata byte masks from every NLCD reclass CSV table."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count() or 1,
        help="Number of mask rasters to generate in parallel. Default: CPU count.",
    )
    return parser.parse_args()


def _read_reclass_table(table_path: Path) -> dict[int, int]:
    """Read a two-column NLCD reclass table.

    Args:
        table_path: CSV file with ``id`` and ``reclass`` columns.

    Returns:
        Mapping from source NLCD integer class to output mask value.

    Raises:
        ValueError: If the table is empty, missing required columns, contains
            duplicate source IDs, or maps to values other than 0 and 1.
    """
    table = pd.read_csv(table_path, usecols=["id", "reclass"])
    if table.empty:
        raise ValueError(f"Reclass table has no rows: {table_path}")
    if table["id"].duplicated().any():
        duplicated_ids = table.loc[table["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Table {table_path} repeats NLCD classes: {duplicated_ids}")
    invalid_values = set(table["reclass"]) - {0, 1}
    if invalid_values:
        raise ValueError(
            f"Table {table_path} maps to invalid mask values: {sorted(invalid_values)}"
        )
    return dict(zip(table["id"].astype(int), table["reclass"].astype(int)))


def _chunked(iterable, chunk_size: int):
    """Yield fixed-size chunks from an iterable.

    Args:
        iterable: Iterable to chunk.
        chunk_size: Maximum number of items in each yielded chunk.

    Yields:
        Lists containing up to ``chunk_size`` items.

    Raises:
        ValueError: If ``chunk_size`` is less than one.
    """
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1.")

    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _build_output_profile(source_profile: dict) -> dict:
    """Build the output raster profile from the NLCD source profile.

    Args:
        source_profile: Rasterio profile from the source NLCD raster.

    Returns:
        Rasterio profile for byte 0/1/nodata mask output.
    """
    output_profile = dict(source_profile)
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=OUTPUT_NODATA,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    return output_profile


def _reclassify_array(source_array: np.ma.MaskedArray, mapping: dict[int, int]) -> np.ndarray:
    """Convert a source NLCD block to a 0/1/nodata byte mask block.

    Args:
        source_array: Masked source NLCD block.
        mapping: Source NLCD class to output mask value.

    Returns:
        A uint8 array with values 0, 1, or ``OUTPUT_NODATA``.
    """
    output = np.full(source_array.shape, OUTPUT_NODATA, dtype=np.uint8)
    data = np.ma.getdata(source_array)
    source_mask = np.ma.getmaskarray(source_array)
    for source_value, mask_value in mapping.items():
        output[(data == source_value) & ~source_mask] = mask_value
    return output


def _worker_initializer(lock: RLock) -> None:
    """Initialize tqdm locking inside a worker process.

    Args:
        lock: Multiprocessing lock shared by tqdm progress bars.
    """
    tqdm.set_lock(lock)


def _generate_mask(job: MaskJob) -> Path:
    """Generate one mask raster from one reclass table.

    Args:
        job: Mask generation job metadata.

    Returns:
        Output raster path.

    Raises:
        FileExistsError: If the timestamped output path already exists.
        ValueError: If the source raster has more than one band.
    """
    mapping = _read_reclass_table(job.table_path)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if job.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {job.output_path}")

    with rasterio.open(DEFAULT_NLCD_RASTER_PATH) as source:
        if source.count != 1:
            raise ValueError(
                f"Expected a single-band NLCD raster: {DEFAULT_NLCD_RASTER_PATH}"
            )

        output_profile = _build_output_profile(source.profile)
        block_windows = list(source.block_windows(1))
        with rasterio.open(job.output_path, "w", **output_profile) as destination:
            with tqdm(
                total=len(block_windows),
                desc=job.table_path.stem,
                unit="block",
                position=job.progress_position,
                leave=True,
                dynamic_ncols=True,
            ) as progress_bar:
                for window_batch in _chunked(
                    block_windows, WINDOWS_PER_PROGRESS_UPDATE
                ):
                    for _, window in window_batch:
                        source_array = source.read(1, window=window, masked=True)
                        output_array = _reclassify_array(source_array, mapping)
                        destination.write(output_array, 1, window=window)
                    progress_bar.update(len(window_batch))

    return job.output_path


def _discover_jobs(
    timestamp: str,
) -> list[MaskJob]:
    """Build mask generation jobs from all CSV tables in a directory.

    Args:
        timestamp: Timestamp string to add to each output raster stem.

    Returns:
        One job per CSV table, sorted by table name.

    Raises:
        FileNotFoundError: If the table directory does not exist.
        ValueError: If no CSV tables are found.
    """
    if not DEFAULT_RECLASS_TABLE_DIR.exists():
        raise FileNotFoundError(
            f"Reclass table directory not found: {DEFAULT_RECLASS_TABLE_DIR}"
        )

    table_paths = sorted(DEFAULT_RECLASS_TABLE_DIR.glob("*.csv"))
    if not table_paths:
        raise ValueError(f"No CSV reclass tables found in: {DEFAULT_RECLASS_TABLE_DIR}")

    jobs = []
    for position, table_path in enumerate(table_paths, start=1):
        table_stem = table_path.stem
        output_path = (
            DEFAULT_OUTPUT_DIR
            / table_stem
            / f"reclassified_NLCD2023_{table_stem}_{timestamp}.tif"
        )
        jobs.append(
            MaskJob(
                table_path=table_path,
                output_path=output_path,
                progress_position=position,
            )
        )
    return jobs


def generate_nlcd_reclass_masks(
    workers: int,
) -> list[Path]:
    """Generate all NLCD reclass masks in parallel.

    Args:
        workers: Maximum number of parallel worker processes.

    Returns:
        Generated output raster paths.

    Raises:
        FileNotFoundError: If the source raster does not exist.
        ValueError: If worker count is invalid.
    """
    if not DEFAULT_NLCD_RASTER_PATH.exists():
        raise FileNotFoundError(f"NLCD raster not found: {DEFAULT_NLCD_RASTER_PATH}")
    if workers < 1:
        raise ValueError("Workers must be at least 1.")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    jobs = _discover_jobs(timestamp)
    worker_count = min(workers, len(jobs))

    print(f"NLCD raster: {DEFAULT_NLCD_RASTER_PATH}", flush=True)
    print(f"Reclass tables: {DEFAULT_RECLASS_TABLE_DIR}", flush=True)
    print(f"Output root: {DEFAULT_OUTPUT_DIR}", flush=True)
    print(f"Workers: {worker_count:,}", flush=True)

    lock = RLock()
    outputs = []
    with tqdm(
        total=len(jobs),
        desc="Completed masks",
        unit="mask",
        position=0,
        dynamic_ncols=True,
    ) as overall_bar:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_initializer,
            initargs=(lock,),
        ) as executor:
            futures = [
                executor.submit(
                    _generate_mask,
                    job,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                outputs.append(future.result())
                overall_bar.update()

    return sorted(outputs)


def main() -> None:
    """Run the NLCD mask generation workflow."""
    args = _parse_args()
    outputs = generate_nlcd_reclass_masks(
        workers=args.workers,
    )
    print("Generated masks:", flush=True)
    for output in outputs:
        print(f"  {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
