"""Add county service totals and ratios to the pipeline's adjusted values.

The caller supplies numeric county values and corrected NLCD areas on the same
county geography. The tracked YAML selects the components and denominator
tables. CSV ingestion handles source formats; the calculation uses those
values directly. Each result carries the rules and source-file fingerprints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from nlcd_mask_areas import FIELD_DEFINITIONS


DEFAULT_CONFIG = Path(__file__).resolve().parent / 'data/workflow_assets/county_integration/derived_metrics.yaml'
DERIVED_FIELDS = ('area_ha_county', 'GDP_2024', 'Pop_2024', 'es_total',
                  'es_per_ha', 'es_per_ha_natural', 'es_per_gdp', 'es_per_capita')


def calculate_county_metrics(frame: pd.DataFrame, data_root: Path, config_path: Path = DEFAULT_CONFIG,
                             *, natural_areas: pd.DataFrame | None = None,
                             natural_source: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Join denominators, sum adjusted services, and calculate county ratios.

    Args:
        frame: Adjusted numeric service values with a unique five-character
            GEOID index. Missing observations are represented by NaN.
        data_root: Root for source paths in the tracked YAML.
        config_path: Pipeline-owned component list, source definitions and ratios.
        natural_areas: Corrected NLCD fields indexed by GEOID on the same county
            geography as frame. If omitted, read the configured mask release.
        natural_source: Source file for supplied natural_areas; required when
            passing current mask results so the output records their provenance.

    Returns:
        County-indexed denominator, area, total and ratio columns, plus source
        and calculation records for the output GeoPackage.
    """
    config_path, data_root = config_path.resolve(), data_root.resolve()
    config_text = config_path.read_text(encoding='utf-8')
    rules = yaml.safe_load(config_text)
    ids = frame.index.rename('GEOID')
    components = rules['es_total']['components']
    component_values = frame[components]

    paths = {name: (data_root / spec['path']).resolve() for name, spec in rules['sources'].items()}
    if natural_areas is not None:
        paths['natural_areas'] = natural_source.resolve()
    paths.update({'derived_rules': config_path, 'derived_calculator': Path(__file__).resolve()})
    hashes = {}
    for name, path in tqdm(paths.items(), desc='Record derived-metric sources', unit='file'):
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()

    # Read the external tables in their source formats, then align county keys.
    values = pd.DataFrame(index=ids)
    status = pd.DataFrame(index=ids)
    issues = []
    input_counts = {}
    for name in tqdm(('county_area', 'gdp', 'population'), desc='Join county denominators', unit='table'):
        spec = rules['sources'][name]
        filters = spec.get('where', {})
        table = pd.read_csv(
            paths[name], encoding=spec.get('encoding', 'utf-8-sig'),
            dtype={column: str for column in (spec['key'], *filters)},
            na_values=spec.get('missing_tokens', []))
        input_counts[name] = len(table)
        for column, required in filters.items():
            table = table.loc[table[column].str.strip().eq(str(required))].copy()
        table[spec['key']] = table[spec['key']].str.strip().str.replace(r'\.0$', '', regex=True).str.zfill(5)
        if spec.get('exclude_state_totals', False):
            table = table.loc[~table[spec['key']].str.endswith('000')]

        series = pd.to_numeric(table.set_index(spec['key'])[spec['field']]) * spec['scale']
        aligned = series.loc[ids] if spec['require_all_counties'] else series.reindex(ids)
        values[spec['output']] = aligned
        status[spec['output'] + '_status'] = np.select(
            [~ids.isin(series.index), aligned.isna(), aligned.le(0)],
            ['unmatched_county', 'missing_value', 'nonpositive'], default='available')
        for key, value in series.loc[~series.index.isin(ids)].items():
            issues.append({'source': name, 'source_key': key, 'reason': 'unmatched_source_not_allocated',
                           'value': value})

    # These areas have already been calculated and checked by the mask workflow.
    if natural_areas is None:
        spec = rules['sources']['natural_areas']
        with tqdm(total=1, desc='Read corrected NLCD areas', unit='table') as progress:
            natural_areas = gpd.read_file(
                paths['natural_areas'], layer=spec['layer'],
                columns=[spec['key'], *FIELD_DEFINITIONS], read_geometry=False).set_index(spec['key'])
            progress.update(1)
    masks = natural_areas.loc[ids, list(FIELD_DEFINITIONS)]
    values = values.join(masks)
    status['area_ha_all_natural_status'] = np.where(
        masks.area_ha_valid_nlcd > 0,
        np.where(masks.area_ha_all_natural > 0, 'available', 'zero'), 'no_nlcd_coverage')

    # The same total is the numerator for every ratio.
    values['es_total'] = component_values.fillna(rules['es_total']['blank_value']).sum(axis=1)
    status['es_total_blank_components_as_zero'] = component_values.isna().sum(axis=1)
    for output, denominator in rules['ratios'].items():
        divisor = values[denominator].where(values[denominator] > 0)
        values[output] = values.es_total / divisor
        status[output + '_status'] = np.where(divisor.notna(), 'available', 'undefined_denominator')

    tables = {
        'derived_metric_sources': pd.DataFrame([{'source': name, 'path': str(path), 'sha256': hashes[name],
            'bytes': path.stat().st_size, 'input_rows': input_counts.get(name),
            'selection': 'explicit_override' if name == 'natural_areas' and natural_source is not None else 'configured'}
            for name, path in paths.items()]),
        'derived_metric_rules': pd.DataFrame([{'path': str(config_path), 'sha256': hashes['derived_rules'], 'yaml_text': config_text}]),
        'derived_metric_county_status': status.reset_index(),
        'derived_metric_source_issues': pd.DataFrame(issues, columns=['source', 'source_key', 'reason', 'value']),
        'derived_metric_components': pd.DataFrame([{'position': i, 'field': name,
            'blank_count': int(component_values[name].isna().sum()),
            'total_included': float(component_values[name].fillna(rules['es_total']['blank_value']).sum())}
            for i, name in enumerate(components, 1)]),
        'derived_metric_run': pd.DataFrame([{'created_utc': datetime.now(timezone.utc).isoformat(),
            'interpretation': rules['es_total']['interpretation'], 'counties': len(ids),
            'component_count': len(components), 'ratios': json.dumps(rules['ratios']),
            'join_policy': 'Exact five-character county FIPS matches',
            'missing_policy': 'Blank service values contribute zero; unavailable/nonpositive denominators give null ratios'}]),
    }
    return values, tables
