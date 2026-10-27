"""Append county ES totals and ratios without rerunning zonals or valuations.

The input is an explicit completed county GeoPackage, never an inferred latest
CSV. Both output formats retain its attributes, FIDs, geometry and audit tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from build_county_ecosystem_service_valuations_2026 import (
    load_export_rules, read_attributes, sha256, write_output,
)
from county_derived_metrics import DEFAULT_CONFIG, DERIVED_FIELDS, calculate_county_metrics
from nlcd_mask_areas import FIELD_DEFINITIONS

ROOT = Path(__file__).resolve().parent


def finalize_counties(county_results: Path, data_root: Path = ROOT / 'data',
                      config_path: Path = DEFAULT_CONFIG, output: Path | None = None) -> Path:
    """Publish a new CSV/GeoPackage pair with totals and denominator ratios.

    Args:
        county_results: Completed, already adjusted county GeoPackage. It may
            already include the twelve corrected NLCD fields, but not ES ratios.
        data_root: Root resolving denominator paths in the rules.
        config_path: Reviewed derived_metrics.yaml; its sibling export_rules.yaml
            controls the default output directory, prefix and layer name.
        output: Optional new GeoPackage path; neither it nor its CSV may exist.

    Returns:
        Verified output GeoPackage path, with a matching CSV alongside it.

    Raises:
        ValueError: Required values/areas are invalid, partial or already derived.
        FileExistsError: Either requested output already exists.
    """
    county_results, data_root, config_path = county_results.resolve(), data_root.resolve(), config_path.resolve()
    if county_results.suffix.lower() != '.gpkg':
        raise ValueError('--county-results must be a completed county GeoPackage')
    export_path = config_path.with_name('export_rules.yaml')
    export = load_export_rules(export_path)['output']
    if output is None:
        output = data_root / export['directory'] / (
            f"{export['filename_prefix']}_{datetime.now():%Y_%m_%d_%H_%M_%S_%f}.gpkg")
    output = output.resolve()
    if output.suffix.lower() != '.gpkg':
        raise ValueError('--output must have a .gpkg extension')
    for path in (output, output.with_suffix('.csv')):
        if path.exists():
            raise FileExistsError(path)
    inputs = {'county_results': county_results, 'finalizer': Path(__file__),
              'export_rules': export_path,
              'output_writer': ROOT / 'build_county_ecosystem_service_valuations_2026.py'}
    hashes = {name: sha256(path) for name, path in tqdm(inputs.items(), desc='Hash finalization inputs', unit='file')}
    with tqdm(total=1, desc='Read existing county values', unit='table') as progress:
        frame, layer, geom = read_attributes(county_results)
        frame = frame.set_index('GEOID')
        progress.update(1)
    if set(frame) & set(DERIVED_FIELDS):
        raise ValueError('Input already has derived metrics; use its original pre-derived county results')
    existing_masks = set(frame) & set(FIELD_DEFINITIONS)
    if existing_masks and existing_masks != set(FIELD_DEFINITIONS):
        raise ValueError('Input contains only some corrected NLCD fields; supply a complete mask-area export')
    metrics, tables = calculate_county_metrics(
        frame, data_root, config_path,
        natural_areas=frame[list(FIELD_DEFINITIONS)] if existing_masks else None,
        natural_source=county_results if existing_masks else None)
    # Keep existing areas, service values and factors; append the new columns.
    additions = metrics[[c for c in metrics if c not in frame]]
    for name, path in inputs.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f'Input changed during finalization: {path}')
    tables['derived_metric_publication'] = pd.DataFrame([
        {'source': name, 'path': str(path.resolve()), 'sha256': hashes[name],
         'bytes': path.stat().st_size, 'yaml_text': export_path.read_text(encoding='utf-8') if name == 'export_rules' else None}
        for name, path in inputs.items()])
    write_output(county_results, output, layer, geom, additions, tables,
                 output_layer=export.get('layer_name', layer))
    tqdm.write(f'Verified: {output}\nVerified: {output.with_suffix(".csv")}')
    return output


def main() -> None:
    """Finalize one explicit county release without any raster/valuation jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--county-results', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, help='New GPKG; matching CSV is automatic')
    args = parser.parse_args()
    finalize_counties(args.county_results, args.data_root, args.config, args.output)


if __name__ == '__main__':
    main()
