"""Totals, units, joins, NoData and paired-export regressions for issue #72."""

from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
import yaml

from build_county_ecosystem_service_valuations_2026 import read_attributes, read_db, sha256
from county_derived_metrics import DEFAULT_CONFIG, DERIVED_FIELDS, calculate_county_metrics, county_keys, load_derived_rules
from finalize_county_ecosystem_services import finalize_counties
from nlcd_mask_areas import CLASSES, FIELD_DEFINITIONS
from prepare_tif_zonal_stats_config import prepare_config
from test_nlcd_mask_areas import mask_statistics, write_fixture
from update_nlcd_mask_areas import export_mask_areas


class DerivedMetricsTests(unittest.TestCase):
    """Verify arithmetic and integration independently of production data."""

    def setUp(self):
        """Create explicit small source releases with different county row orders."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = pd.Index(['01001', '01003', '01005'], name='GEOID')
        self.frame = pd.DataFrame({'first_adj': [10., np.nan, 5.], 'second_adj': [2., 4., np.nan],
                                   'sum_do_not_double_count_adj': [1000.] * 3,
                                   'sum_flood_marginal_npv_masked_to_wetlands_adj': [1e12, 2e12, np.nan]},
                                  index=self.ids)
        self.masks = pd.DataFrame(index=self.ids)
        self.masks['area_ha_valid_nlcd'] = [2., 2., 0.]
        for name in CLASSES:
            self.masks['area_ha_nlcd_' + name] = ([.5, .5, 0.] if name == 'forests'
                                                else [.5, 0., 0.] if name == 'wetlands' else 0.)
        self.masks['area_ha_all_natural'] = [1., .5, 0.]
        for name in CLASSES:
            self.masks['proportion_nlcd_' + name + '_of_valid'] = (
                self.masks['area_ha_nlcd_' + name] / self.masks.area_ha_valid_nlcd.replace(0, np.nan))
        self.geometry = gpd.GeoSeries([box(i, 0, i + 1, 1) for i in range(3)], index=self.ids, crs=5070)
        self.mask_path = self.root / 'masks.gpkg'
        gpd.GeoDataFrame(self.masks, geometry=self.geometry).reset_index().iloc[::-1].to_file(
            self.mask_path, layer='counties', driver='GPKG')
        self.base = self.root / 'base.gpkg'
        gpd.GeoDataFrame(self.frame, geometry=self.geometry).reset_index().to_file(self.base, layer='counties', driver='GPKG')
        pd.DataFrame({'GEOID': [1005, 1001, 1003], 'hectares': [5., 2., 4.]}).to_csv(self.root / 'area.csv', index=False)
        pd.DataFrame({'GeoFIPS': ['01000', '01001', '01003', '01005', '51901'],
                      'LineCode': ['3'] * 5, 'Unit': ['Thousands of dollars'] * 5,
                      '2024': ['100', '1', '0', '(NA)', '90']}).to_csv(self.root / 'gdp.csv', index=False)
        pd.DataFrame({'GEOFIPS': ['01000', '01003', '01001'], 'SUMLEV': ['040', '050', '050'],
                      'POPESTIMATE2024': [100, 2, 3], 'CTYNAME': ['State', 'Peña', 'Autauga']}).to_csv(
            self.root / 'population.csv', index=False, encoding='cp1252')
        self.config = self.root / 'derived_metrics.yaml'
        self.rules = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        self.rules['es_total']['components'] = ['first_adj', 'second_adj']
        for name, path in [('county_area', 'area.csv'), ('natural_areas', 'masks.gpkg'),
                           ('gdp', 'gdp.csv'), ('population', 'population.csv')]:
            self.rules['sources'][name]['path'] = path
        self.rules['sources']['county_area']['field'] = 'hectares'
        self.rules['sources']['natural_areas']['layer'] = 'counties'
        self.config.write_text(yaml.safe_dump(self.rules), encoding='utf-8')
        self.config.with_name('export_rules.yaml').write_bytes(DEFAULT_CONFIG.with_name('export_rules.yaml').read_bytes())

    def test_exact_allowlist_blanks_units_and_null_ratios(self):
        """Sum only listed adjusted values and use true denominators, not coverage."""
        result, tables = calculate_county_metrics(self.frame, self.root, self.config, county_vector=self.base)
        np.testing.assert_allclose(result.es_total, [12., 4., 5.])
        np.testing.assert_allclose(result.es_per_ha, [6., 1., 1.])
        np.testing.assert_allclose(result.es_per_ha_natural, [12., 8., np.nan], equal_nan=True)
        np.testing.assert_allclose(result.GDP_2024, [1000., 0., np.nan], equal_nan=True)
        np.testing.assert_allclose(result.es_per_gdp, [.012, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(result.es_per_capita, [4., 2., np.nan], equal_nan=True)
        status = tables['derived_metric_county_status'].set_index('GEOID')
        self.assertEqual(status.es_total_blank_components_as_zero.to_list(), [0, 1, 1])
        self.assertEqual(status.loc['01005', 'Pop_2024_status'], 'unmatched_county')
        self.assertEqual(status.loc['01005', 'area_ha_all_natural_status'], 'no_nlcd_coverage')
        issues = tables['derived_metric_source_issues']
        self.assertEqual(issues.source_key.to_list(), ['51901'])
        self.assertEqual(issues.value.to_list(), [90000.])
        sources = tables['derived_metric_sources'].set_index('source')
        self.assertEqual(sources.loc['gdp', 'sha256'], sha256(self.root / 'gdp.csv'))
        self.assertEqual(yaml.safe_load(tables['derived_metric_rules'].yaml_text.iloc[0]), self.rules)

    def test_invalid_required_components_and_numbers_fail(self):
        """Missing columns differ from permitted blanks; malformed values fail."""
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            calculate_county_metrics(self.frame.drop(columns='first_adj'), self.root, self.config)
        for value in ('oops', np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                changed = self.frame.astype(object)
                changed.loc['01001', 'first_adj'] = value
                calculate_county_metrics(changed, self.root, self.config)
        for ids in (['1001', '01001'], ['x'], [None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                county_keys(pd.Series(ids), 'fixture')

    def test_default_policy_excludes_npv_from_total_and_all_ratios(self):
        """Keep annual flood benefits while NPV neither contributes nor is required."""
        components = load_derived_rules(DEFAULT_CONFIG)['es_total']['components']
        npv = 'sum_flood_marginal_npv_masked_to_wetlands_adj'
        self.assertEqual(len(components), 19)
        self.assertFalse(any('npv' in name.lower() for name in components))
        self.assertIn('sum_flood_annual_value_masked_to_wetlands_adj', components)
        self.rules['es_total']['components'] = components
        self.config.write_text(yaml.safe_dump(self.rules), encoding='utf-8')
        frame = pd.DataFrame(1., index=self.ids, columns=components)
        frame[npv] = [1e12, 2e12, np.nan]
        result, tables = calculate_county_metrics(frame, self.root, self.config)
        without_npv, _ = calculate_county_metrics(frame.drop(columns=npv), self.root, self.config)
        pd.testing.assert_frame_equal(result, without_npv)
        np.testing.assert_allclose(result.es_total, [19.] * 3)
        np.testing.assert_allclose(result.es_per_ha, [9.5, 4.75, 3.8])
        np.testing.assert_allclose(result.es_per_ha_natural, [19., 38., np.nan], equal_nan=True)
        np.testing.assert_allclose(result.es_per_gdp, [.019, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(result.es_per_capita, [19. / 3, 9.5, np.nan], equal_nan=True)
        self.assertNotIn(npv, tables['derived_metric_components'].field.to_list())

    def test_rule_typos_duplicates_and_already_derived_inputs_fail(self):
        """Never ignore ambiguous YAML or recompute against a stale derived export."""
        text = self.config.read_text(encoding='utf-8')
        for content in (text + '\nschema_version: 1\n', text + '\nunknown: true\n',
                         text.replace('scale: 1000.0', 'scalle: 1000.0')):
            with self.subTest(content=content[-40:]), self.assertRaises(ValueError):
                self.config.write_text(content, encoding='utf-8')
                load_derived_rules(self.config)
        self.config.write_text(text, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'already has derived metrics'):
            calculate_county_metrics(self.frame.assign(es_total=0), self.root, self.config)
        self.rules['es_total']['components'] = ['first_adj', 'first_adj']
        self.config.write_text(yaml.safe_dump(self.rules), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unique adjusted'):
            load_derived_rules(self.config)

    def test_invalid_sources_or_units_cannot_silently_publish(self):
        """Reject wrong currency units, duplicate keys and invalid denominators."""
        original = (self.root / 'gdp.csv').read_bytes()
        for field, value in [('Unit', 'Dollars'), ('GeoFIPS', '01003'), ('2024', '-3'), ('2024', 'garbage')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                table = pd.read_csv(self.root / 'gdp.csv', dtype=str)
                table.loc[1, field] = value
                table.to_csv(self.root / 'gdp.csv', index=False)
                try:
                    calculate_county_metrics(self.frame, self.root, self.config)
                finally:
                    (self.root / 'gdp.csv').write_bytes(original)
        (self.root / 'area.csv').write_text('GEOID,hectares\n01001,2\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'required county denominator'):
            calculate_county_metrics(self.frame, self.root, self.config)

    def test_current_mask_override_is_used_and_audited(self):
        """A mask update must use its fresh areas, not a prior configured release."""
        current = self.masks.copy()
        current.loc['01001', 'area_ha_nlcd_forests'] = 1.
        current.loc['01001', 'area_ha_all_natural'] = 1.5
        current.loc['01001', 'proportion_nlcd_forests_of_valid'] = .5
        result, tables = calculate_county_metrics(self.frame, self.root, self.config,
                                                  natural_areas=current, natural_source=self.base)
        self.assertEqual(result.loc['01001', 'es_per_ha_natural'], 8.)
        source = tables['derived_metric_sources'].set_index('source').loc['natural_areas']
        self.assertEqual(source.selection, 'explicit_override')
        self.assertEqual(source.path, str(self.base))
        self.assertEqual(source.sha256, sha256(self.base))
        for field, value in [('area_ha_all_natural', 5.), ('area_ha_nlcd_forests', -1.),
                              ('proportion_nlcd_forests_of_valid', .9)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                invalid = self.masks.copy()
                invalid.loc['01001', field] = value
                calculate_county_metrics(self.frame, self.root, self.config,
                                          natural_areas=invalid, natural_source=self.base)

    def test_finalizer_preserves_base_and_writes_matching_formats(self):
        """Preserve attributes, factors, geometry, FIDs, audits and source hashes."""
        masked = pd.concat([self.frame, self.masks], axis=1)
        masked['first_adj_factor'] = .09
        source = self.root / 'already_masked.gpkg'
        gpd.GeoDataFrame(masked, geometry=self.geometry).reset_index().to_file(source, layer='counties', driver='GPKG')
        with closing(sqlite3.connect(source)) as con:
            con.execute('CREATE TABLE existing_audit (value TEXT)')
            con.execute("INSERT INTO existing_audit VALUES ('keep me')")
            con.commit()
        before = sha256(source)
        output = finalize_counties(source, self.root, self.config)
        self.assertEqual(sha256(source), before)
        attrs, layer, geom = read_attributes(output)
        csv = pd.read_csv(output.with_suffix('.csv'), dtype={'GEOID': str}, float_precision='round_trip')
        pd.testing.assert_frame_equal(attrs, csv, check_dtype=False)
        original, _, _ = read_attributes(source)
        pd.testing.assert_frame_equal(original, attrs[list(original)], check_dtype=False)
        self.assertEqual(set(attrs) - set(original), set(DERIVED_FIELDS))
        with closing(read_db(source)) as left, closing(read_db(output)) as right:
            self.assertEqual(left.execute('SELECT fid,geom FROM counties ORDER BY fid').fetchall(),
                             right.execute(f'SELECT fid,"{geom}" FROM "{layer}" ORDER BY fid').fetchall())
            self.assertEqual(right.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(right.execute('SELECT value FROM existing_audit').fetchone()[0], 'keep me')
            self.assertEqual(right.execute("SELECT sha256 FROM derived_metric_publication WHERE source='county_results'").fetchone()[0], before)
        with self.assertRaises(FileExistsError):
            finalize_counties(source, self.root, self.config, output)
        with self.assertRaisesRegex(ValueError, 'already has derived metrics'):
            finalize_counties(output, self.root, self.config, self.root / 'again.gpkg')

    def test_finalizer_can_append_configured_masks_to_unmasked_input(self):
        """Completed valuations can use the explicit corrected mask release."""
        output = finalize_counties(self.base, self.root, self.config, self.root / 'with_masks.gpkg')
        attrs, _, _ = read_attributes(output)
        self.assertTrue(set(FIELD_DEFINITIONS) <= set(attrs))
        self.assertEqual(attrs.es_total.to_list(), [12., 4., 5.])

    def test_mask_export_also_calculates_metrics_from_current_run(self):
        """The mask workflow wires new areas into ratios without monetary reruns."""
        profile, base = write_fixture(self.root)
        rules = self.rules.copy()
        rules['es_total'] = {'components': ['sum_example_service_adj'], 'blank_value': 0.,
                             'interpretation': 'Synthetic fixture'}
        self.config.write_text(yaml.safe_dump(rules), encoding='utf-8')
        frozen = prepare_config(profile, output_path=self.root / 'run.yaml')
        raw = self.root / 'results/counties/counties_masks_20261026_000000.gpkg'
        raw.parent.mkdir(parents=True)
        geometry = gpd.read_file(base).set_index('GEOID').geometry
        gpd.GeoDataFrame(mask_statistics().set_index('GEOID'), geometry=geometry).reset_index().to_file(
            raw, layer='counties', driver='GPKG')
        output = self.root / 'mask_and_totals.gpkg'
        export_mask_areas(raw, output, frozen, base, derived_config=self.config, data_root=self.root)
        attrs, _, _ = read_attributes(output)
        np.testing.assert_allclose(attrs.es_total, [123.456, -12.5, 0.])
        self.assertAlmostEqual(attrs.es_per_ha_natural.iloc[0], 123.456 / .45)
        self.assertTrue(attrs.es_per_ha_natural.iloc[1:].isna().all())

    def test_primary_builder_uses_post_adjustment_values(self):
        """Exercise the real build/export path with only service worker I/O stubbed."""
        import build_county_ecosystem_service_valuations_2026 as builder

        raw_field = builder.BASE_FIELDS['pollination']
        raw = pd.DataFrame({'STATEFP': '01', raw_field: [10., 4., 5.]}, index=self.ids)
        source = self.root / 'raw_counties.gpkg'
        gpd.GeoDataFrame(raw, geometry=self.geometry).reset_index().to_file(source, layer='counties', driver='GPKG')
        names = ['base', 'grazing', 'timber', 'air_quality', 'urban_heat', 'physical_health', 'jobs',
                 'marine_fisheries', 'coastline', 'inland_fisheries', 'freshwater', 'recreation', 'mangroves', 'corals']
        pd.DataFrame({'source': names, 'path': source.name}).to_csv(self.root / 'sources.csv', index=False)
        pd.DataFrame({'service': ['pollination'], 'factor': [.09], 'reason': ['hectares to pixels'],
                      'source_units': ['USD/ha']}).to_csv(self.root / 'adjustment_factors.csv', index=False)
        pd.DataFrame({'source_geoid': ['01001'], 'target_geoid': ['01001'], 'weight': [1.]}).to_csv(
            self.root / 'ct_county_crosswalk.csv', index=False)
        export = yaml.safe_load(self.config.with_name('export_rules.yaml').read_text(encoding='utf-8'))
        export.update({'exclude_fields': [], 'optional_exclude_fields': [], 'exclude_patterns': [], 'rename_fields': {}})
        self.config.with_name('export_rules.yaml').write_text(yaml.safe_dump(export), encoding='utf-8')
        self.rules['es_total']['components'] = [raw_field + '_adj']
        self.config.write_text(yaml.safe_dump(self.rules), encoding='utf-8')

        def fixture_task(task, paths, base, crosswalk, geometry_workers, chunk_dir):
            """Return service-task fixtures while exercising orchestration and export.

            Args:
                task: Named service family.
                paths: Unused fixture paths.
                base: Raw county frame.
                crosswalk: Unused fixture crosswalk.
                geometry_workers: Unused worker count.
                chunk_dir: Unused checkpoint directory.

            Returns:
                The normal worker result schema with raw, unadjusted pollination.
            """
            return {'originals': {'pollination': base[raw_field]} if task == 'base' else {},
                    'statuses': [pd.DataFrame({'service': [task], 'status': ['fixture']})],
                    'metadata': [{'service': 'pollination', 'original_total': 19.}] if task == 'base' else [],
                    'issues': [], 'checks': [{'service': task, 'status': 'fixture'}],
                    'recreation_audit': pd.DataFrame(columns=['siteid']),
                    'mangrove_audit': pd.DataFrame(columns=['unit'])}

        with patch.object(builder, 'ProcessPoolExecutor', ThreadPoolExecutor), patch.object(
            builder, 'compute_task', side_effect=fixture_task
        ), patch.object(builder, 'save_checkpoint'):
            output = builder.run(self.root, self.root, self.root / 'primary.gpkg', workers=1, geometry_workers=1)
        attrs, _, _ = read_attributes(output)
        np.testing.assert_allclose(attrs.es_total, [.9, .36, .45])
        np.testing.assert_allclose(attrs.es_per_ha, [.45, .09, .09])
        self.assertEqual(attrs.pollination_adj_factor.to_list(), [.09] * 3)
        with closing(read_db(output)) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM derived_metric_components').fetchone()[0], 1)

    def test_invalid_mask_geography_fails_before_export(self):
        """Equal county keys do not permit using a different county footprint."""
        wrong = self.root / 'wrong_counties.gpkg'
        gpd.GeoDataFrame(self.frame, geometry=self.geometry.translate(xoff=5)).reset_index().to_file(
            wrong, layer='counties', driver='GPKG')
        with self.assertRaisesRegex(ValueError, 'boundaries differ'):
            finalize_counties(wrong, self.root, self.config, self.root / 'invalid.gpkg')
        self.assertFalse((self.root / 'invalid.gpkg').exists())


if __name__ == '__main__':
    unittest.main()
