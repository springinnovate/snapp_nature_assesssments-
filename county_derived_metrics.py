"""Calculate auditable county service totals and denominator-based ratios.

This stage consumes already adjusted county values and corrected NLCD areas.
It neither samples rasters nor applies ecosystem-service adjustment factors.
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from nlcd_mask_areas import CLASSES, FIELD_DEFINITIONS


DEFAULT_CONFIG = Path(__file__).resolve().parent / 'data/workflow_assets/county_integration/derived_metrics.yaml'
RATIOS = {'es_per_ha': 'area_ha_county', 'es_per_ha_natural': 'area_ha_all_natural',
          'es_per_gdp': 'GDP_2024', 'es_per_capita': 'Pop_2024'}
DERIVED_FIELDS = ('area_ha_county', 'GDP_2024', 'Pop_2024', 'es_total', *RATIOS)


class _UniqueRulesLoader(yaml.SafeLoader):
    """Reject duplicated policy keys instead of silently using the final value."""

    def construct_mapping(self, node, deep=False):
        """Construct one YAML mapping with unambiguous keys.

        Args:
            node: YAML mapping node.
            deep: Whether nested values should be constructed immediately.

        Returns:
            Parsed mapping with unique keys.

        Raises:
            ValueError: The configuration contains a duplicate key.
        """
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate key in derived-metric rules')
        return super().construct_mapping(node, deep=deep)


def load_derived_rules(path: Path) -> dict:
    """Read and validate the explicit post-calculation policy.

    Args:
        path: Tracked standard-YAML definition.

    Returns:
        Validated component allowlist, source schemas and ratio mapping.

    Raises:
        ValueError: A rule is malformed or would silently change units/policies.
    """
    try:
        rules = yaml.load(path.read_text(encoding='utf-8'), Loader=_UniqueRulesLoader)
    except (yaml.YAMLError, TypeError) as error:
        raise ValueError(f'Invalid derived rules in {path}: {error}') from error
    if (not isinstance(rules, dict) or rules.get('schema_version') != 1
            or set(rules) != {'schema_version', 'es_total', 'sources', 'ratios'}):
        raise ValueError('derived_metrics.yaml requires schema_version: 1')
    total = rules.get('es_total', {})
    if not isinstance(total, dict) or set(total) != {'components', 'blank_value', 'interpretation'}:
        raise ValueError('es_total requires components, blank_value and interpretation')
    components = total.get('components', [])
    if (not isinstance(components, list) or not components
            or any(not isinstance(c, str) or not c.endswith('_adj') for c in components)
            or len(components) != len(set(components)) or total.get('blank_value') != 0
            or not isinstance(total.get('interpretation'), str) or not total['interpretation'].strip()):
        raise ValueError('es_total requires unique adjusted component names, blank_value: 0, and an interpretation')
    if rules.get('ratios') != RATIOS:
        raise ValueError('Use the four documented ratio names and denominators')
    sources = rules.get('sources', {})
    if not isinstance(sources, dict) or set(sources) != {'county_area', 'natural_areas', 'gdp', 'population'}:
        raise ValueError('Derived metrics require county_area, natural_areas, gdp and population sources')
    for name, spec in sources.items():
        if not isinstance(spec, dict) or not all(isinstance(spec.get(k), str) and spec[k] for k in ('path', 'key')):
            raise ValueError(f'Invalid derived source: {name}')
        if name == 'natural_areas':
            if (set(spec) != {'path', 'layer', 'key'} or not isinstance(spec.get('layer'), str)
                    or not spec['layer'] or Path(spec['path']).suffix.lower() != '.gpkg'):
                raise ValueError('Natural areas require an explicit corrected mask GeoPackage/layer')
            continue
        if set(spec) - {'path', 'key', 'field', 'output', 'scale', 'require_all_counties', 'encoding',
                        'where', 'expected_units', 'missing_tokens', 'exclude_state_totals'}:
            raise ValueError(f'Unknown source setting for {name}')
        expected = {'county_area': 'area_ha_county', 'gdp': 'GDP_2024', 'population': 'Pop_2024'}[name]
        if (not spec.get('field') or spec.get('output') != expected
                or Path(spec['path']).suffix.lower() != '.csv'
                or not isinstance(spec.get('require_all_counties'), bool)
                or not isinstance(spec.get('scale'), (int, float))
                or not np.isfinite(spec['scale']) or spec['scale'] <= 0):
            raise ValueError(f'Invalid output, scale, or join policy for {name}')
    return rules


def county_keys(values: pd.Series, label: str) -> pd.Series:
    """Normalize county identifiers without losing leading zeros or duplicates.

    Args:
        values: County identifiers from a table or frame index.
        label: Source description for validation errors.

    Returns:
        Five-digit string keys, preserving the input row index.

    Raises:
        ValueError: A key is blank, malformed or duplicated after normalization.
    """
    text = values.astype('string').str.strip()
    if not text.str.fullmatch(r'\d{1,5}(?:\.0)?').fillna(False).all():
        raise ValueError(f'{label}: expected nonblank county FIPS keys')
    text = text.str.replace(r'\.0$', '', regex=True).str.zfill(5)
    if text.duplicated().any():
        raise ValueError(f'{label}: duplicate county keys after normalization')
    return text


def numeric_values(values: pd.Series, label: str, missing_tokens=()) -> pd.Series:
    """Parse numbers, recognizing only explicitly permitted missing tokens.

    Args:
        values: Numeric or textual values.
        label: Source/field description for errors.
        missing_tokens: Additional source-defined null markers, such as (NA).

    Returns:
        Float values; blanks and permitted missing markers become NaN.

    Raises:
        ValueError: A nonblank value is nonnumeric or infinite.
    """
    text = values.astype('string').str.strip()
    text = text.mask(text.isin(['', *missing_tokens]))
    try:
        parsed = pd.to_numeric(text, errors='raise').astype(float)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{label}: invalid numeric value') from error
    if np.isinf(parsed.to_numpy()).any():
        raise ValueError(f'{label}: infinite values are not supported')
    return parsed


def calculate_county_metrics(frame: pd.DataFrame, data_root: Path, config_path: Path = DEFAULT_CONFIG,
                             *, natural_areas: pd.DataFrame | None = None,
                             natural_source: Path | None = None,
                             county_vector: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Calculate totals/ratios from current adjusted values and reviewed inputs.

    Args:
        frame: Current adjusted service attributes indexed by county GEOID.
        data_root: Root resolving source paths in the reviewed YAML.
        config_path: Tracked post-calculation rules.
        natural_areas: Optional current mask results, indexed by GEOID. Overrides
            the configured mask release; all twelve corrected fields are required.
        natural_source: Exact source of the override for provenance.
        county_vector: County GeoPackage whose geography must match a configured
            mask source. Overrides are already on the caller's county geography.

    Returns:
        Derived area/denominator/total/ratio fields, plus GeoPackage audit tables.

    Raises:
        ValueError: Required columns, keys, units or areas are invalid. Required
            fields cannot be silently substituted with the legacy coral raster.
    """
    config_path, data_root = config_path.resolve(), data_root.resolve()
    config_text = config_path.read_text(encoding='utf-8')
    rules = load_derived_rules(config_path)
    ids = pd.Index(county_keys(pd.Series(frame.index), 'county values'), name='GEOID')
    frame = frame.copy()
    frame.index = ids
    components = rules['es_total']['components']
    missing = set(components) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required adjusted es_total components: {sorted(missing)}. '
                         'Missing columns are not blanks; the corrected coral field requires PR #69.')
    if set(frame) & set(DERIVED_FIELDS):
        raise ValueError('Input already has derived metrics; use its original pre-derived county results')
    paths = {name: (data_root / spec['path']).resolve() for name, spec in rules['sources'].items()}
    if natural_areas is not None:
        if natural_source is None:
            raise ValueError('Current natural-area overrides require their exact source path')
        paths['natural_areas'] = natural_source.resolve()
    elif natural_source is not None:
        raise ValueError('natural_source requires natural_areas')
    paths.update({'derived_rules': config_path, 'derived_calculator': Path(__file__).resolve()})
    hashes = {}
    for name, path in tqdm(paths.items(), desc='Hash derived-metric inputs', unit='file'):
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    values = pd.DataFrame(index=ids)
    status = pd.DataFrame(index=ids)
    issues = []
    input_counts = {}
    for name in tqdm(('county_area', 'gdp', 'population'), desc='Join county denominators', unit='table'):
        spec = rules['sources'][name]
        table = pd.read_csv(paths[name], dtype=str, encoding=spec.get('encoding', 'utf-8-sig'),
                            keep_default_na=False)
        input_counts[name] = len(table)
        needed = {spec['key'], spec['field'], *spec.get('where', {})}
        if spec.get('expected_units'):
            needed.add(spec['expected_units']['column'])
        if not needed <= set(table):
            raise ValueError(f'{name}: missing configured source columns {sorted(needed - set(table))}')
        for column, required in spec.get('where', {}).items():
            table = table.loc[table[column].str.strip().eq(str(required))].copy()
        if table.empty:
            raise ValueError(f'{name}: no source rows match the configured selection')
        table[spec['key']] = county_keys(table[spec['key']], name)
        if spec.get('exclude_state_totals', False):
            table = table.loc[~table[spec['key']].str.endswith('000')].copy()
        unit_rule = spec.get('expected_units')
        if unit_rule and not table[unit_rule['column']].eq(unit_rule['value']).all():
            raise ValueError(f'{name}: unexpected units; refusing to apply scale {spec["scale"]}')
        series = numeric_values(table[spec['field']], name, spec.get('missing_tokens', ())) * spec['scale']
        series.index = table[spec['key']]
        if (series.dropna() < 0).any():
            raise ValueError(f'{name}: negative denominators are invalid')
        if name == 'population' and not np.array_equal(series.dropna(), np.round(series.dropna())):
            raise ValueError('Population estimates must be whole persons')
        aligned = series.reindex(ids)
        absent = ~ids.isin(series.index)
        if spec['require_all_counties'] and (absent.any() or aligned.isna().any()):
            raise ValueError(f'{name}: required county denominator is missing')
        values[spec['output']] = aligned
        status[spec['output'] + '_status'] = np.select(
            [absent, aligned.isna(), aligned.eq(0)], ['unmatched_county', 'missing_value', 'zero'], default='available')
        for key, value in series.loc[~series.index.isin(ids)].items():
            issues.append({'source': name, 'source_key': key, 'reason': 'unmatched_source_not_allocated',
                           'value': value})

    if natural_areas is None:
        spec = rules['sources']['natural_areas']
        with closing(sqlite3.connect(paths['natural_areas'].as_uri() + '?mode=ro', uri=True)) as con:
            layer = spec['layer'].replace('"', '""')
            actual = con.execute('SELECT table_name FROM gpkg_geometry_columns').fetchall()
            if actual != [(spec['layer'],)]:
                raise ValueError('Configured mask feature layer is missing or ambiguous')
            # Select attributes only; the explicit layer and schema are checked.
            wanted = [spec['key'], *FIELD_DEFINITIONS]
            available = {row[1] for row in con.execute(f'PRAGMA table_info("{layer}")')}
            if not set(wanted) <= available:
                raise ValueError('Configured mask layer lacks the twelve corrected mask fields')
            query = 'SELECT ' + ','.join('"' + c.replace('"', '""') + '"' for c in wanted) + ' FROM "' + layer + '"'
            natural_areas = pd.read_sql_query(query, con).set_index(spec['key'])
        if county_vector is not None:
            with tqdm(total=3, desc='Verify natural-area county geography', unit='step') as progress:
                mask_geometry = gpd.read_file(paths['natural_areas'], layer=spec['layer'], columns=[spec['key']]).set_index(spec['key']).geometry
                progress.update(1)
                with closing(sqlite3.connect(county_vector.resolve().as_uri() + '?mode=ro', uri=True)) as con:
                    county_layers = con.execute('SELECT table_name FROM gpkg_geometry_columns').fetchall()
                if len(county_layers) != 1:
                    raise ValueError('Expected one county feature layer, excluding auxiliary audit tables')
                county_geometry = gpd.read_file(county_vector, layer=county_layers[0][0], columns=['GEOID']).set_index('GEOID').geometry
                progress.update(1)
                if set(mask_geometry.index) != set(county_geometry.index) or mask_geometry.crs != county_geometry.crs:
                    raise ValueError('Natural-area and county geometries have different keys or CRS')
                mask_geometry = mask_geometry.reindex(county_geometry.index)
                exact = county_geometry.geom_equals_exact(mask_geometry, tolerance=0)
                if not county_geometry[~exact].geom_equals(mask_geometry[~exact]).all():
                    raise ValueError('Natural-area county boundaries differ')
                progress.update(1)
    masks = natural_areas.copy()
    masks.index = pd.Index(county_keys(pd.Series(masks.index), 'corrected masks'), name='GEOID')
    if set(masks.index) != set(ids) or not set(FIELD_DEFINITIONS) <= set(masks):
        raise ValueError('Corrected natural areas must contain all county keys and twelve explicit mask fields')
    masks = masks.reindex(ids)[list(FIELD_DEFINITIONS)].apply(pd.to_numeric, errors='raise').astype(float)
    area_fields = ['area_ha_nlcd_' + c for c in CLASSES]
    hectares = masks[['area_ha_valid_nlcd', *area_fields, 'area_ha_all_natural']]
    if not np.isfinite(hectares.to_numpy()).all() or (hectares < 0).any().any():
        raise ValueError('Corrected mask hectares must be finite and nonnegative')
    natural = masks[area_fields].sum(axis=1)
    if (not np.allclose(natural, masks.area_ha_all_natural, rtol=0, atol=1e-7)
            or (natural > masks.area_ha_valid_nlcd + 1e-7).any()):
        raise ValueError('Natural area must be the sum of disjoint classes, no greater than valid coverage')
    covered = masks.area_ha_valid_nlcd > 0
    for name in CLASSES:
        expected = masks['area_ha_nlcd_' + name].div(masks.area_ha_valid_nlcd.where(covered))
        if not np.allclose(masks['proportion_nlcd_' + name + '_of_valid'], expected,
                           rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f'Incorrect {name} proportion or zero-coverage handling')
    for field in FIELD_DEFINITIONS:
        if field in frame and not np.allclose(numeric_values(frame[field], field), masks[field],
                                              rtol=1e-12, atol=1e-7, equal_nan=True):
            raise ValueError(f'Existing {field} differs from selected corrected mask data')
    for field in FIELD_DEFINITIONS:
        values[field] = masks[field]
    status['area_ha_all_natural_status'] = np.where(covered, np.where(natural > 0, 'available', 'zero'), 'no_nlcd_coverage')
    component_values = pd.DataFrame({name: numeric_values(frame[name], name) for name in components}, index=ids)
    status['es_total_blank_components_as_zero'] = component_values.isna().sum(axis=1)
    values['es_total'] = component_values.fillna(0.0).sum(axis=1)
    if not np.isfinite(values.es_total).all():
        raise ValueError('es_total overflowed; input totals must be finite')
    for output, denominator in rules['ratios'].items():
        divisor = values[denominator].where(values[denominator] > 0)
        values[output] = values.es_total / divisor
        if np.isinf(values[output]).any():
            raise ValueError(f'{output} overflowed')
        status[output + '_status'] = np.where(divisor.notna(), 'available', 'undefined_denominator')
    for name, path in paths.items():
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != hashes[name]:
                raise ValueError(f'Derived-metric input changed during calculation: {path}')
    tables = {
        'derived_metric_sources': pd.DataFrame([{'source': name, 'path': str(path), 'sha256': hashes[name],
            'bytes': path.stat().st_size, 'input_rows': input_counts.get(name),
            'selection': 'explicit_override' if name == 'natural_areas' and natural_source is not None else 'configured'}
            for name, path in paths.items()]),
        'derived_metric_rules': pd.DataFrame([{'path': str(config_path), 'sha256': hashes['derived_rules'], 'yaml_text': config_text}]),
        'derived_metric_county_status': status.reset_index(),
        'derived_metric_source_issues': pd.DataFrame(issues, columns=['source', 'source_key', 'reason', 'value']),
        'derived_metric_components': pd.DataFrame([{'position': i, 'field': name,
            'blank_count': int(component_values[name].isna().sum()), 'total_included': float(component_values[name].fillna(0).sum())}
            for i, name in enumerate(components, 1)]),
        'derived_metric_run': pd.DataFrame([{'created_utc': datetime.now(timezone.utc).isoformat(),
            'interpretation': rules['es_total']['interpretation'], 'counties': len(ids),
            'component_count': len(components), 'ratios': json.dumps(rules['ratios']),
            'join_policy': 'Exact normalized county keys only; combined GDP regions are not allocated',
            'missing_policy': 'Blank service values contribute zero; unavailable/nonpositive ratio denominators give null'}]),
    }
    return values, tables
