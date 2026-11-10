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

from assessment import REPO_ROOT
from assessment.county_gdp import allocate_combined_gdp, read_county_source
from assessment.nlcd_mask_areas import FIELD_DEFINITIONS


DEFAULT_CONFIG = REPO_ROOT / 'data/workflow_assets/county_integration/derived_metrics.yaml'
DERIVED_FIELDS = ('area_ha_county', 'area_ha_county_incl_ocean_great_lakes',
                  'area_ha_all_natural_incl_ocean_great_lakes', 'GDP_2024', 'GDP_2024_is_allocated', 'Pop_2024', 'es_total',
                  'es_per_ha', 'es_per_ha_incl_ocean_great_lakes', 'es_per_ha_natural',
                  'es_per_ha_natural_incl_ocean_great_lakes', 'es_per_gdp', 'es_per_capita')


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
    crosswalk = rules['sources']['gdp'].get('allocation_crosswalk')
    if crosswalk:
        paths['gdp_allocation_crosswalk'] = (data_root / crosswalk).resolve()
    if natural_areas is not None:
        paths['natural_areas'] = natural_source.resolve()
    paths.update({'derived_rules': config_path, 'derived_calculator': Path(__file__).resolve(),
                  'gdp_calculator': Path(__file__).with_name('county_gdp.py').resolve()})
    hashes = {}
    for name, path in tqdm(paths.items(), desc='Record derived-metric sources', unit='file'):
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()

    # Read the external tables in their source formats, then align county keys.
    values = pd.DataFrame(index=ids)
    status = pd.DataFrame(index=ids)
    issues = []
    input_counts = {}
    source_values = {}
    tabular_sources = [name for name in ('county_area', 'county_area_inclusive', 'gdp', 'population')
                       if name in rules['sources']]
    for name in tqdm(tabular_sources, desc='Join county denominators', unit='table'):
        spec = rules['sources'][name]
        series, input_counts[name] = read_county_source(paths[name], spec)
        source_values[name] = series
    allocations = pd.DataFrame()
    group_checks = pd.DataFrame()
    if crosswalk:
        source_values['gdp'], allocations, group_checks = allocate_combined_gdp(
            source_values['gdp'], source_values['population'], paths['gdp_allocation_crosswalk'])
        input_counts['gdp_allocation_crosswalk'] = len(allocations)
    allocated_ids = set(allocations['county_geoid']) if len(allocations) else set()
    for name, series in source_values.items():
        spec = rules['sources'][name]
        aligned = series.loc[ids] if spec['require_all_counties'] else series.reindex(ids)
        values[spec['output']] = aligned
        status[spec['output'] + '_status'] = np.select(
            [~ids.isin(series.index), aligned.isna(), aligned.le(0)],
            ['unmatched_county', 'missing_value', 'nonpositive'], default='available')
        for key, value in series.loc[~series.index.isin(ids)].items():
            issues.append({'source': name, 'source_key': key, 'reason': 'unmatched_source_not_allocated',
                           'value': value})
    values['GDP_2024_is_allocated'] = ids.isin(allocated_ids).astype('int64')
    status.loc[ids.isin(allocated_ids), 'GDP_2024_status'] = 'population_allocated'

    if 'county_area_inclusive' in source_values:
        retained = values[rules['sources']['county_area']['output']]
        whole = values[rules['sources']['county_area_inclusive']['output']]
        if ((retained > whole) & ~np.isclose(retained, whole, rtol=1e-10, atol=1e-6)).any():
            raise ValueError('County area excluding ocean/Great Lakes exceeds the full county area')

    # These areas have already been calculated and checked by the mask workflow.
    if natural_areas is None:
        spec = rules['sources']['natural_areas']
        area_field = spec.get('county_area_field')
        with tqdm(total=1, desc='Read corrected NLCD areas', unit='table') as progress:
            natural_areas = gpd.read_file(
                paths['natural_areas'], layer=spec['layer'],
                columns=[spec['key'], *FIELD_DEFINITIONS, *([area_field] if area_field else [])],
                read_geometry=False).set_index(spec['key'])
            progress.update(1)
    area_field = rules['sources']['natural_areas'].get('county_area_field')
    if area_field:
        if area_field not in natural_areas:
            raise ValueError('NLCD areas must be recalculated on the ocean/Great-Lakes-excluded county footprint')
        county_area = values[rules['sources']['county_area']['output']]
        if not np.allclose(natural_areas.loc[ids, area_field], county_area, rtol=1e-10, atol=1e-6):
            raise ValueError('County-area and NLCD sources describe different reporting footprints')
    masks = natural_areas.loc[ids, list(FIELD_DEFINITIONS)].copy()
    if 'natural_areas_inclusive' in paths:
        spec = rules['sources']['natural_areas_inclusive']
        with tqdm(total=1, desc='Read full-county NLCD areas', unit='table') as progress:
            full = gpd.read_file(paths['natural_areas_inclusive'], layer=spec['layer'],
                                 columns=[spec['key'], spec['field'], 'area_ha_valid_nlcd'],
                                 read_geometry=False).set_index(spec['key'])
            progress.update(1)
        if not full.index.is_unique or not ids.isin(full.index).all():
            raise ValueError('Full-county NLCD areas require exactly one matching row for each county')
        input_counts['natural_areas_inclusive'] = len(full)
        inclusive = full.loc[ids, spec['field']]
        if not np.isfinite(inclusive).all() or inclusive.lt(0).any():
            raise ValueError('Full-county natural areas must be finite and nonnegative')
        masks.insert(masks.columns.get_loc('area_ha_all_natural') + 1, spec['output'], inclusive)
        status[spec['output'] + '_status'] = np.where(
            full.loc[ids, 'area_ha_valid_nlcd'] > 0,
            np.where(inclusive > 0, 'available', 'zero'), 'no_nlcd_coverage')
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
            'join_policy': ('Exact five-character county FIPS; configured Virginia combinations '
                            'allocated by complete-group 2024 population' if crosswalk else
                            'Exact five-character county FIPS matches'),
            'missing_policy': 'Blank service values contribute zero; unavailable/nonpositive denominators give null ratios'}]),
    }
    if crosswalk:
        tables['derived_metric_gdp_allocations'] = allocations
        tables['derived_metric_gdp_group_checks'] = group_checks
    return values, tables
