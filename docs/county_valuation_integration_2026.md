# County ecosystem-service valuations for the October 2026 SNAPP assessment

## Purpose and scope

This workflow builds a county-level dataset of ecosystem-service estimates for
the October 2026 SNAPP nature assessment. It starts with the September 10, 2026
county results and adds estimates of grazing, air-quality benefits, wages,
timber, fisheries, urban heat, recreation, and physical health. It also applies
the assessment's recorded adjustment factors to selected existing services.

The result is a GeoPackage with one map feature per county or county equivalent,
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
   in Git. The three small configuration CSVs are included.
2. Activate the geospatial Python environment described in the
   [README](../README.md#environment). For a new setup, create it with
   `conda env create -f environment.yml`, then run `conda activate geo`.
   The environment includes `openpyxl`, which reads the Excel inputs.
3. From the repository folder, run:

   ```powershell
   python build_county_ecosystem_service_valuations_2026.py
   ```

The script displays progress and prints the completed output path. Source files
are preserved, and an existing output is never overwritten. If a run fails,
correct the reported problem and rerun the same command; completed work can be
reused when its inputs and code are unchanged.

If your `data/` folder is elsewhere, supply its location:

```powershell
python build_county_ecosystem_service_valuations_2026.py --data-root "D:\SNAPP\data"
```

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

The additional configuration files are
[`adjustment_factors.csv`](../data/workflow_assets/county_integration/adjustment_factors.csv),
which records the multipliers and source units, and
[`ct_county_crosswalk.csv`](../data/workflow_assets/county_integration/ct_county_crosswalk.csv),
which relates historical Connecticut counties to planning regions.

The mangrove study-unit file `ucsc_cwon_studyunits.gpkg` is not an input to this
run. The assessment uses the mangrove values already in the starting county
GeoPackage. Likewise, the earlier standalone recreation preparation script is
not needed: this script allocates the listed recreation polygons directly.

## How values are assigned to counties

County identifiers are standardized as five-digit FIPS codes, preserving leading
zeroes. County totals are joined directly; datasets with multiple observations
per county are summed as specified below. Duplicate records that would count a
county total twice are rejected.

| Output service | Source field and method | Source units/year | Adjustment factor |
| --- | --- | --- | --- |
| `grazing` | Join `CountyGrazingValueYear` by `CountyCode` from the `SNAPPGrazingCalculations` sheet | Annual USD, 2024 | 1 |
| `avoided_air_quality_heath_costs` | Sum `dollars_annual` across land-cover and pollutant rows by `FIPS` | Annual USD | 1.135 |
| `jobs_cultural`, `jobs_raw_materials`, `jobs_processing`, `jobs_harvesting` | Join the four raw Total Wages columns from the `totals by county` sheet by `County FIPS` | USD wages, 2022 | 1.065 |
| `jobs` | Sum those four wage categories | USD wages, 2022 | 1.065 |
| `timber` | Join `total_annual_rent` by `fips_padded` | Annual USD, 2024 | 1 |
| `marine_fisheries` | Allocate each state's `ncp_value` by its counties' shares of the state's coastline length | USD, 2023 | 1.027 |
| `inland_fisheries` | Allocate each state's `Consumptive use value (USD)_Intermediate assumption` by its counties' shares of the state's freshwater area | USD, 2022 | 1.065 |
| `urban_heat` | Sum `Sum of total_dollars_saved` by `COUNTY_GEOID` | USD, 2021 | 1.135 |
| `physical_health` | Join `total_value_usd` by `GEOID` | USD, 2023 | 1.027 |
| `recreation` | Allocate `val_2024` by the fraction of each source polygon's area within each county, then sum by county | USD, 2024 | 1 |

The air-quality field intentionally retains the assessment's spelling
`avoided_air_quality_heath_costs`. The jobs measure is wages, not a count of jobs.
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

### Existing services and assessment decisions

The following county sums are taken directly from the starting GeoPackage.
Their original columns remain available alongside the new valuation fields.

| Output service | Existing source field | Adjustment factor |
| --- | --- | --- |
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

The assessment specified `3.66 × 1.182 = 4.32612` for both carbon fields. Their
source names already include `tCO2e`, and no additional dollar-per-tonne carbon
price was specified. These adjusted columns therefore must not be interpreted
as established USD valuations. The per-hectare raster sum is retained as
supplied; this workflow does not reconstruct it into a physical county total.

Coral values already include a 2024 dollar adjustment and receive factor 1.
Existing mangrove values are retained, including their signs, with factor 1.182.
The remaining factors follow the recorded assessment choices; the script does
not estimate new inflation rates or infer unrecorded source years.

The two duplicate crop/pollination sums
`sum_national_attributed_annual_crop_yield_value_zstd` and
`sum_pollination_attributed_annual_crop_yield_value` are removed as an assessment
decision. Their non-sum diagnostic columns remain. Other original fields,
including `Wsw` quantities and the two unidentified `CONUS_Coastal_2021_CCAP`
bands, are preserved without additional valuation adjustments.

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

Each successful run creates:

```text
data/analysis_results/combined/counties_ecosystem_services_YYYYMMDD_HHMMSS.gpkg
```

Open it in QGIS and select the county layer `tl_2024_us_county_60_states`. For the
listed inputs, it has 3,225 counties or county equivalents, the original county
boundaries, and 81 new fields describing 27 services or service components.
Puerto Rico and the U.S. Virgin Islands remain present even when source data for
a service are unavailable.

Each service has three fields:

| Suffix | Meaning | Example |
| --- | --- | --- |
| `_orig` | County value after joining/allocation, before adjustment | `grazing_orig` |
| `_adj_factor` | Multiplier applied, including an explicit 1.0 where applicable | `grazing_adj_factor` |
| `_adj` | Original value multiplied by the adjustment factor | `grazing_adj` |

For example, a county value of 100 with factor 1.135 becomes 113.5. The `_orig`
field is a county estimate; it may have been allocated from a state or polygon
source rather than measured directly at county scale.

To read the county layer in Python, substitute the actual timestamped filename:

```python
import geopandas as gpd

path = "data/analysis_results/combined/counties_ecosystem_services_<timestamp>.gpkg"
counties = gpd.read_file(path, layer="tl_2024_us_county_60_states")
print(counties[["GEOID", "NAME", "grazing_orig", "grazing_adj_factor", "grazing_adj"]].head())
```

The GeoPackage also contains tables to help evaluate and reproduce the estimates:

| Question | Table to inspect |
| --- | --- |
| Which source field, units, method, and multiplier produced this service? | `integration_services` |
| Which exact source files and configuration versions were used? | `integration_sources` |
| Is this county value observed, allocated, zero-filled, or missing? | `integration_county_status` |
| Which records could not be assigned, or contributed only to an incomplete estimate? | `integration_source_issues` |
| Do the county fisheries allocations reproduce the state totals? | `integration_state_checks` |
| How much of each recreation polygon's value was allocated or left outside the county footprint? | `integration_recreation_allocation` |
| Which Connecticut county-to-region weights were used? | `integration_ct_crosswalk` |
| Which script version and execution produced the file? | `integration_run` |
| Which saved intermediate results identify this run's completed work? | `integration_checkpoints` |

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
wages, and carbon quantities. Jobs and water provisioning also appear as both
components and totals. The `_adj` suffix records a multiplication; it does not
by itself establish that values are economically comparable or non-overlapping.

### Findings from the October 1, 2026 assessment run

The county allocations reproduced all 71 supplied fisheries state totals (23
marine and 48 inland; maximum absolute difference below $0.000001). All 325,791
recreation polygons were accounted for. Their source values totaled
$881,801,006,000, of which $881,637,401,423.39 was assigned to counties and
$163,604,576.61 remained outside the covered footprint, net of very small
boundary overlaps. The workflow checks geometry coverage, source-total
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
