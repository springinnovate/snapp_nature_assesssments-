# County ecosystem-service valuations for the October 2026 SNAPP assessment

## Purpose and scope

This workflow builds a county-level dataset of ecosystem-service estimates for
the October 2026 SNAPP nature assessment. It starts with the September 10, 2026
county results and adds estimates of grazing, air-quality benefits, wages,
timber, fisheries, urban heat, recreation, and physical health. It also applies
the assessment's recorded adjustment factors to selected existing services.

The result is a matching CSV and GeoPackage, with one record per county or county equivalent,
including Puerto Rico and the U.S. Virgin Islands. It can be used to map service
values, examine geographic coverage, and compare estimates where their units and
valuation methods are compatible. The original county boundaries are retained.

The script is
[`build_county_ecosystem_service_valuations_2026.py`](../build_county_ecosystem_service_valuations_2026.py)
in the repository's top-level folder. This is a record of a specific assessment
update; using different source releases requires reviewing the input paths,
fields, adjustment factors, and geographic coverage. The general zonal-statistics
and PAD-US workflows are described separately in the [README](../README.md).
The development record is [issue #58](https://github.com/springinnovate/snapp_nature_assesssments-/issues/58).

## How to run it

1. Obtain the source releases listed below and place them at the specified paths
   under the repository's `data/` folder. The large data files are not included
   in Git. The three small configuration CSVs and export-rules YAML are included.
2. Activate the geospatial Python environment described in the
   [README](../README.md#environment). For a new setup, create it with
   `conda env create -f environment.yml`, then run `conda activate geo`.
   The environment includes `openpyxl`, which reads the Excel inputs, and
   `PyYAML`, which reads the export rules. Existing environments without PyYAML
   can install it with `python -m pip install pyyaml`.
3. From the repository folder, run:

   ```powershell
   python build_county_ecosystem_service_valuations_2026.py
   ```

The script displays progress and prints both completed output paths. Source files
are preserved, and an existing output is never overwritten. If a run fails,
correct the reported problem and rerun the same command; completed work can be
reused when its calculation inputs and code are unchanged. Changes limited to
export rules reuse raw calculation checkpoints and produce a new output pair.

If your `data/` folder is elsewhere, supply its location:

```powershell
python build_county_ecosystem_service_valuations_2026.py --data-root "D:\SNAPP\data"
```

For the workstation used for this assessment, run:

```powershell
conda activate zonal-stats-toolkit
python build_county_ecosystem_service_valuations_2026.py --workers 16 --geometry-workers 16
```

This reuses the existing September county raster aggregates; it does not rerun
raster zonal statistics. Check `sources.csv` before running and retain its pinned
raw September county input. `adjustment_factors.csv` contains the `pollination`
row with factor `0.09`. The converted dollar total is
`sum_pollination_attributed_annual_crop_yield_value_adj`. The visible
`pollination_adj_factor` records `0.09`. Raw-value columns and the duplicate
`pollination_adj` alias are not exported. This 30 m pixel-area
conversion must not also be applied upstream. The builder rejects a base that
only contains the already-scaled per-pixel VRT aggregate, or an already-integrated
output, to prevent applying adjustments twice. Source files are never changed.

To select a basename, pass `--output <new-name>.gpkg`; its `.csv` sibling is
automatic. Neither destination may already exist. A reduced or reprojected GIS
export is not needed to obtain the requested fields.

Use a local output disk, such as NTFS on Windows. By default, the result is saved
under `data/analysis_results/combined/` on that disk. See `--help` for additional
path options.

## Input data

Paths below are relative to `data/`. The exact input list is also recorded in
[`sources.csv`](../data/workflow_assets/county_integration/sources.csv). If a
filename or location changes, update that file before running. The source fields
and allocation methods are described in the next section.

| Dataset | File within `data/` |
| --- | --- |
| Starting county results | `analysis_results/zonal_statistics/counties/counties_ecosystem_services_20260910_130816.gpkg` |
| County coastline lengths | `analysis_results/zonal_statistics/counties/counties_coastline_length_20260910_130816.gpkg` |
| County freshwater areas | `analysis_results/zonal_statistics/counties/counties_freshwater_area_20260910_130816.csv` |
| Grazing | `analysis_inputs/ecosystem_services/grazing/SNAPPGrazingDataset_final_062326.xlsx` |
| Nature-related wages | `analysis_inputs/ecosystem_services/jobs/Finalized Flagged and Sorted Industry and County Codes.xlsx` |
| Timber | `analysis_inputs/ecosystem_services/timber/total_valuation_all_regions_fvs.csv` |
| Air quality | `analysis_inputs/ecosystem_services/air_quality/snapp_air_quality.csv` |
| Marine fisheries | `analysis_inputs/ecosystem_services/marine_fisheries/NCPvalue_landings_marine_fish_noaa_2023_state.csv` |
| Inland fisheries | `analysis_inputs/ecosystem_services/inland_fisheries/Harvest_table_with_value.csv` |
| Urban heat | `analysis_inputs/ecosystem_services/urban_heat/HEAT_all_tracts_2021.csv` |
| Physical health | `analysis_inputs/ecosystem_services/physical_health/hypertension_linear_us_counties_all.gpkg` |
| Recreation | `analysis_inputs/recreation/usa_nature_assessment_recreation__snapp_estimates.gpkg` |

The additional calculation configuration files are
[`adjustment_factors.csv`](../data/workflow_assets/county_integration/adjustment_factors.csv),
which records the multipliers and source units, and
[`ct_county_crosswalk.csv`](../data/workflow_assets/county_integration/ct_county_crosswalk.csv),
which relates historical Connecticut counties to planning regions.
The tracked [`export_rules.yaml`](../data/workflow_assets/county_integration/export_rules.yaml)
controls only the exported schema and default output names, as described below.

The mangrove study-unit file `ucsc_cwon_studyunits.gpkg` is not an input to this
run. The assessment uses the mangrove values already in the starting county
GeoPackage. Likewise, the earlier standalone recreation preparation script is
not needed: this script allocates the listed recreation polygons directly.

## Export rules

Edit `data/workflow_assets/county_integration/export_rules.yaml` to change field
names, omit fields, reorder column groups, or set the default output directory
and filename prefix. This file uses YAML syntax. It does not replace
`sources.csv`, `adjustment_factors.csv`, or `ct_county_crosswalk.csv`, which remain
the authoritative calculation configuration. All four files are read from the
existing `--config-dir` directory when that option is supplied. The script's
source-field mappings and allocation methods remain calculation code; this YAML
controls how their results are exported.

| Key | Meaning |
| --- | --- |
| `rename_fields` | Map a default generated export name to its final exported name. For a configured raster sum, the default is its source name plus `_adj`; other services use the internal service key plus `_adj`. Renames are evaluated together, not chained. |
| `exclude_fields` | Exact source-statistic or default export names that must exist and must be omitted. A misspelled or absent name stops the run. |
| `optional_exclude_fields` | Exact field names to omit if present, for known differences between source versions. |
| `exclude_patterns` | Shell-style `fnmatch` patterns for excluding generated statistics and diagnostics. |
| `column_groups` | Each of `base_attributes`, `sum_values`, `service_totals`, and `adjustment_factors`, listed once. `base_attributes` must be first; the remaining groups may be reordered. |
| `output.directory` | Default output directory, relative to the selected data root. |
| `output.filename_prefix` | Prefix before the timestamp in both output filenames. |

For example, the flood annual rename is recorded explicitly as:

```yaml
rename_fields:
  sum_annual_value_masked_to_wetlands_adj: sum_flood_annual_value_masked_to_wetlands_adj
```

Edit the existing mapping rather than adding a second `rename_fields` section.
The default configuration also renames the flood NPV and pixel-carbon totals,
applies the descriptive air-quality names, excludes the four `sum_Wsw_*`
quantity fields, and omits non-sum statistics and coverage diagnostics.
The monetary `sum_Wval_sw_2020usd_*_adj` provisioning totals are retained.
Original `ALAND` and `AWATER` remain county geographic attributes.

The builder validates unknown names, conflicting rules, and final-name
collisions before expensive calculations start. Exclusions cannot remove
original county attributes. It applies the same resolved
schema to the CSV and GeoPackage. Each output GeoPackage records the complete
YAML text and hash in `integration_export_rules`, and the resolved action and
final name for each field in `integration_export_fields`. Service metadata uses
the final exported names. Other input and configuration hashes remain recorded
with the run, so the output does not depend on a mutable configuration file for
its provenance.

Rerun the county builder after changing these rules; raster zonal statistics do
not need to be rerun. An edit limited to export rules reuses the same raw
calculation checkpoints. Source, calculation-code, or crosswalk changes still
invalidate the affected cached work. The new export receives a fresh timestamp;
existing deliverables are not overwritten. `--output` overrides the YAML's
default directory and filename for a single run.

## How values are assigned to counties

County identifiers are standardized as five-digit FIPS codes, preserving leading
zeroes. County totals are joined directly; datasets with multiple observations
per county are summed as specified below. Duplicate records that would count a
county total twice are rejected.

| Output service | Source field and method | Source units/year | Adjustment factor |
| --- | --- | --- | --- |
| `grazing` | Join `CountyGrazingValueYear` by `CountyCode` from the `SNAPPGrazingCalculations` sheet | Annual USD, 2024 | 1 |
| `air_quality_avoided_mortality_annual_value` | Sum `dollars_annual` across land-cover and pollutant rows by `FIPS` | Annual USD | 1.135 |
| `jobs_cultural`, `jobs_raw_materials`, `jobs_processing`, `jobs_harvesting` | Join the four raw Total Wages columns from the `totals by county` sheet by `County FIPS` | USD wages, 2022 | 1.065 |
| `jobs` | Sum those four wage categories | USD wages, 2022 | 1.065 |
| `timber` | Join `total_annual_rent` by `fips_padded` | Annual USD, 2024 | 1 |
| `marine_fisheries` | Allocate each state's `ncp_value` by its counties' shares of the state's coastline length | USD, 2023 | 1.027 |
| `inland_fisheries` | Allocate each state's `Consumptive use value (USD)_Intermediate assumption` by its counties' shares of the state's freshwater area | USD, 2022 | 1.065 |
| `urban_heat` | Sum `Sum of total_dollars_saved` by `COUNTY_GEOID` | USD, 2021 | 1.135 |
| `physical_health` | Join `total_value_usd` by `GEOID` | USD, 2023 | 1.027 |
| `recreation` | Allocate `val_2024` by the fraction of each source polygon's area within each county, then sum by county | USD, 2024 | 1 |

### Air-quality annual total

The county CSV and feature layer contain
`air_quality_avoided_mortality_annual_value_adj`: the county sum of
`dollars_annual` multiplied by `1.135`. The multiplier is visible as
`air_quality_avoided_mortality_annual_value_adj_factor`. No air-quality category
statistics, counts, raw-value copies, or duplicate totals are exported.

The supplied CSV has three land-cover classes and three pollutants (nine records
per source county). These are category estimates, not raster pixels or per-area
values. Historical Connecticut contributions are allocated by the existing
crosswalk, then recombined within each target land-cover/pollutant category.
An unknown contribution keeps that category unknown; the county total remains
null for incomplete or absent coverage. Coverage information remains available
in the GeoPackage audit tables.

The reviewed configuration retains its existing service key
`avoided_air_quality_heath_costs`. The descriptive export name above is mapped
from that key by `integration_services.output_field`; the configuration and
multiplier are unchanged. The jobs measure is wages, not a count of jobs.
It uses the raw wage sheet rather than the population-normalized sheet; state,
national, metropolitan, and unallocated summary rows are excluded. Grazing and
timber source values are already county totals, so no additional acreage
multiplication is applied.

State fisheries allocation assumes that value is proportional to the selected
coastline or freshwater-area measure. Recreation allocation assumes that value
is uniform within each source polygon. Recreation areas are measured in global
equal-area projection EPSG:6933, which supports the states and territories in
this assessment. A polygon's full area is the denominator: value outside the
county footprint stays unallocated and is recorded, rather than redistributed
to the counties that happen to overlap it.

A hexagon crossing county boundaries is split proportionally among those
counties. For example, a $100 hexagon with 30% of its area in one county and 70%
in another contributes $30 and $70. The shares are measured against the full
source polygon area, including any portion outside the county footprint.

Very small source slivers can accumulate numerical error during intersection.
If summed county shares exceed 100% by at most `1e-5` (0.001% of the source
area), the script scales that polygon's county contributions together to total
100%. Larger excess still stops the run and identifies the affected source IDs.
The allocation table records `raw_allocated_fraction` and
`coverage_correction_value` alongside the corrected fraction and dollars.
This correction never redistributes genuinely uncovered source value.

### Existing services and assessment decisions

The following county sums are read from the starting GeoPackage, multiplied by
the configured factor, and given an `_adj` suffix before the export rules apply
any descriptive rename. Factor-1 services
still use `_adj`. Other generated statistics and coverage diagnostics are
omitted: mean, median, min, max, standard deviation, percentiles, valid/total
counts, valid-area fields, and proportions. Original geographic attributes,
including county land and water areas (`ALAND` and `AWATER`), are retained.

| Output service | Existing source field | Adjustment factor |
| --- | --- | --- |
| `pollination` | `sum_pollination_attributed_annual_crop_yield_value` | 0.09 |
| `carbon_ha`, `carbon_pixel` | `sum_totalC_tCO2e_ha_2020`, `sum_totalC_tCO2e_pixel_2020` | 4.32612 each |
| `dredging` | `sum_avoided_dredging_costs_raster` | 1 |
| `sdwa` | `sum_avoided_sdwa_health_costs_raster` | 1 |
| `water_treatment` | `sum_avoided_treatment_costs_raster` | 1 |
| `provisioning` | `sum_Wval_sw_2020usd_total` | 1.182 |
| `provisioning_irr`, `provisioning_pow`, `provisioning_pub` | Corresponding `sum_Wval_sw_2020usd_irr`, `_pow`, and `_pub` components | 1.182 |
| `flood_npv`, `flood_annual` | `sum_marginal_npv_masked_to_wetlands`, `sum_annual_value_masked_to_wetlands` | 1 |
| `corals` | `sum_Coral_Reefs_2024adj_CPI` | 1 |
| `mangroves` | `sum_mangrove_CONUS` | 1.182 |
| `mental_health` | `sum_mental_health_national_existing_greenness_cost_90m` | 1 |

The assessment supplied USD 3.66 per tonne CO2e as the carbon price, followed by
the 1.182 inflation adjustment: `3.66 × 1.182 = 4.32612`. That multiplier is
applied once to both existing carbon aggregates. It is a price conversion, not
a carbon-to-CO2 mass conversion. The pixel-carbon result is named
`sum_CO2e_mitigation_dollar_value_per_pixel_adj`: it is the county sum of
monetary per-pixel values, not a per-pixel county average. The per-hectare source
is still a sum of per-hectare values multiplied by the price; the workflow does
not multiply those values by area, so that separate field is not an
area-integrated county dollar total.

The flood annual and NPV exports are respectively
`sum_flood_annual_value_masked_to_wetlands_adj` and
`sum_flood_marginal_npv_masked_to_wetlands_adj`. These renames leave the
calculated values and factors unchanged.

Coral values already include a 2024 dollar adjustment and receive factor 1.
Existing mangrove values are retained, including their signs, with factor 1.182.
The remaining factors follow the recorded assessment choices; the script does
not estimate new inflation rates or infer unrecorded source years.

The duplicate crop-yield sum `sum_national_attributed_annual_crop_yield_value_zstd`
is removed as an assessment decision. The adjusted pollination sum is included.
The `sum_Wsw_irr`, `sum_Wsw_pow`, `sum_Wsw_pub`, and `sum_Wsw_total` quantity
fields are also excluded by the export rules. This does not remove the
`sum_Wval_sw_2020usd_*_adj` monetary provisioning values. Other retained,
unconfigured source sums, including the two unidentified
`CONUS_Coastal_2021_CCAP` bands, keep their existing names and values without
additional valuation adjustments; their other statistics and diagnostics are
omitted as well.

### Connecticut county boundaries

Several sources use Connecticut's eight historical counties, while the output
uses the nine planning regions in the starting county layer. Historical totals
are distributed by land-area shares from the Census Bureau's
[2022 county subdivision to 2020 block group relationship file](https://www2.census.gov/geo/docs/maps-data/data/rel2022/acs22_cousub22_blkgrp20_st09.txt).
The [record-layout documentation](https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2022-connecticut-record-layout.html)
defines its fields.

The crosswalk has 19 positive-area links. To reproduce it, take the first five
characters of `GEOID_BLKGRP_20` as the historical county and of `GEOID_COUSUB_22`
as the planning region. Sum `AREALAND_PART` by that pair, omit zero-area links,
and divide by the summed land area for each historical county. Weights sum to
one within each old county. The raw source file's SHA-256 checksum is
`06957a5e777dc96da4c678a79d27dce6617cfbd510c9611dbf7720c3cd16c5bd`.
The derived crosswalk is included in the repository, so the raw download is not
required to run the valuation script.

These allocations are land-area estimates, not population- or service-specific
measurements. If a planning region requires a missing historical county value,
its result remains null and known partial contributions are recorded separately.
Timber follows the zero-filling rule below. Other unmatched historical county
codes are reported without an assumed correspondence.

## Expected output

With the default export rules, each successful run creates:

```text
data/analysis_results/combined/snapp_ecosystem_services_counties_YYYY_MM_DD_HH_MM_SS.csv
data/analysis_results/combined/snapp_ecosystem_services_counties_YYYY_MM_DD_HH_MM_SS.gpkg
```

Open it in QGIS and select the county layer `tl_2024_us_county_60_states`. For the
listed inputs, it has 3,225 counties or county equivalents, the original county
boundaries, adjusted totals and 28 visible service adjustment factors. Both files
have matching county attributes, including the adjusted pollination sum. Geometry
is present only in the GeoPackage. The CSV preserves leading-zero GEOIDs and nulls; do not let
spreadsheet software interpret GEOID as a number. Publication verifies both
files and refuses to replace an existing file; a publication failure rolls back
any file published by that attempt.
Puerto Rico and the U.S. Virgin Islands remain present even when source data for
a service are unavailable.

The default export rules put county columns in this order in both files:

1. Original county identifiers and geographic attributes, in their source order,
   including original land/water areas.
2. All retained `sum_` fields, including adjusted raster sums and unconfigured
   source sums.
3. Other adjusted service totals, including
   `air_quality_avoided_mortality_annual_value_adj`.
4. All adjustment-factor columns.

Generated non-sum statistics and coverage diagnostics are absent from the county
table. The separate GeoPackage audit tables retain source and coverage details.

Each configured service has adjusted values and a visible factor, without raw-value copies:

| Suffix | Meaning | Example |
| --- | --- | --- |
| `_adj_factor` | Multiplier applied, including an explicit 1.0 where applicable | `grazing_adj_factor` |
| `_adj` | Source sum or allocated county value multiplied by the factor | `sum_pollination_attributed_annual_crop_yield_value_adj`, `grazing_adj` |

For example, a raw county sum of 100 with factor 1.135 becomes an adjusted sum
of 113.5. No `_orig` columns are emitted. Raster services default to the source
sum name plus `_adj`, with the descriptive flood and pixel-carbon names set in
the export rules. Services without existing statistic fields default to
`<service>_adj`, with the descriptive air-quality export name listed above.
`integration_services.output_field` identifies
each service's total field; that audit table also records the factor and aggregate
before/after totals. A value may have been allocated from a state or polygon
source rather than measured directly at county scale.

To read the county layer in Python, substitute the actual timestamped filename:

```python
import geopandas as gpd

path = "data/analysis_results/combined/snapp_ecosystem_services_counties_<timestamp>.gpkg"
counties = gpd.read_file(path, layer="tl_2024_us_county_60_states")
print(counties[["GEOID", "NAME", "grazing_adj_factor", "grazing_adj"]].head())
```

The GeoPackage also contains tables to help evaluate and reproduce the estimates:

| Question | Table to inspect |
| --- | --- |
| Which source field, units, method, and multiplier produced this service? | `integration_services` |
| Which exact source files and configuration versions were used? | `integration_sources` |
| Which complete export-rules YAML and hash produced this schema? | `integration_export_rules` |
| Which fields were retained, renamed, or excluded, and what are their final names? | `integration_export_fields` |
| Is this county value observed, allocated, zero-filled, or missing? | `integration_county_status` |
| Which records could not be assigned, or contributed only to an incomplete estimate? | `integration_source_issues` |
| Do the county fisheries allocations reproduce the state totals? | `integration_state_checks` |
| How much of each recreation polygon's value was allocated or left outside the county footprint? | `integration_recreation_allocation` |
| Which Connecticut county-to-region weights were used? | `integration_ct_crosswalk` |
| Which script version and execution produced the file? | `integration_run` |
| Which saved intermediate results identify this run's completed work? | `integration_checkpoints` |
| What does the exported air-quality annual total mean? | `integration_summary_fields` |

## Interpreting the values

A null value means that a complete estimate is unavailable. It should not
normally be treated as a measured zero. Timber is an explicit exception in this
assessment: missing rows and missing values are set to zero, with coverage
statuses distinguishing `no_timberland`, `no_species_coverage`, and absent data.
The heat CSV's 12 accounting-format `$ -` cells are also read as zero.

Within a state with a fisheries value, a county with zero allocation weight
receives zero. States absent from a fisheries source remain null, including
Puerto Rico. A supplied state total with no positive allocation support causes
the run to stop rather than assign the total arbitrarily.

Do not sum all adjusted columns as a single ecosystem-service total. They
include different units and time horizons, such as annual values, flood NPV,
wages, and the non-area-integrated carbon-per-hectare aggregate. Jobs and water
provisioning also appear as both components and totals. The `_adj` suffix records a multiplication; it does not
by itself establish that values are economically comparable or non-overlapping.

### Findings from the October 1, 2026 assessment run

The county allocations reproduced all 71 supplied fisheries state totals (23
marine and 48 inland; maximum absolute difference below $0.000001). All 325,791
recreation polygons were accounted for. Their source values totaled
$881,801,006,000, of which $881,637,401,419.08 was assigned to counties and
$163,604,580.92 remained outside the covered footprint after the bounded
numerical coverage correction. That correction removed $4.31 of aggregate
excess from 78,137 polygons with extremely small raw excess fractions in this
validation run. The workflow checks geometry coverage, source-total
reconciliation, and preservation of the starting county data before producing
the completed file.

The following source limitations remain in the result:

- All 142 nonzero existing mangrove aggregates are negative. Their signs are
  preserved; an inflation adjustment does not establish why they are negative.
- Timber has 350 negative county estimates. Four unmatched historical Alaska
  codes contain a net -$7,590,984.93; two other unmatched codes have null source
  values. These records are listed in `integration_source_issues`.
- Connecticut air quality, grazing, and timber cover all nine planning regions
  after allocation. Jobs already use the new regions. Physical-health inputs
  do not support a complete value for any of the nine regions. Heat covers six;
  the other three remain null, with $10,950,652.94 of known partial contributions
  recorded in the issue table.
- Physical health has 1,095 populated counties, urban heat has 1,187, and each
  jobs category has 3,216. Check the county-status table when choosing a study
  area or comparing totals across services.
