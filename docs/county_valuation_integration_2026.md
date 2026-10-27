# County ecosystem-service valuations for the October 2026 SNAPP assessment

## Purpose and scope

This dataset brings ecosystem-service estimates onto a common county geography
for the October 2026 SNAPP nature assessment. It combines raster-derived county
sums with tabular, state-level, and polygon estimates, then applies the
assessment's recorded unit conversions and valuation adjustments.

The output has one record for each of 3,225 counties or county equivalents,
including Puerto Rico and the U.S. Virgin Islands. County boundaries and
identifiers come from the supplied September 10, 2026 county GeoPackage.
Coverage varies by service: retaining a county in the output does not imply
that every service has an estimate there.

These data support mapping and service-specific comparisons. They do not
constitute a single, additive measure of ecosystem value. Some estimates are
annual flows, others are net present values or wages, and several component
fields are also represented in a total.

The implementation is
[`build_county_ecosystem_service_valuations_2026.py`](../build_county_ecosystem_service_valuations_2026.py).
This guide describes its inputs, allocation assumptions, adjustments, and
interpretation limits. The [README](../README.md) describes the separate
raster zonal-statistics and PAD-US workflows.

County NLCD areas can be appended **after** this valuation integration without
resampling any ecosystem-service raster or reapplying a monetary factor. See
the [mask-only update command](../README.md#fix-county-mask-areas-without-rerunning-ecosystem-service-zonals).
It writes a new matching CSV/GeoPackage, preserving all existing service fields,
and appends `area_ha_valid_nlcd`, five `area_ha_nlcd_<class>` values,
`area_ha_all_natural`, and five `proportion_nlcd_<class>_of_valid` values.
These observed areas are not the full county polygon area. Source NLCD NoData
has zero observed coverage and undefined (null) class proportions.
The area fields are nonmonetary: no adjustment factor is applied. The valuation
builder now consumes the explicit corrected mask release in `derived_metrics.yaml`.
It does not discover a latest run. Mask updates use their own freshly computed
areas for the additional totals/ratios described below.

## Input data

The analysis uses the releases listed in
[`sources.csv`](../data/workflow_assets/county_integration/sources.csv).
Paths in the table are relative to the repository's `data/` directory.

| Dataset | File within `data/` |
| --- | --- |
| Raster-derived county results and county boundaries | `analysis_results/zonal_statistics/counties/counties_ecosystem_services_20260910_130816.gpkg` |
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
| Mangrove coastal flood protection | `analysis_inputs/ecosystem_services/mangroves/ucsc_cwon_studyunits.gpkg` |

Four additional files record the assessment decisions:

- [`adjustment_factors.csv`](../data/workflow_assets/county_integration/adjustment_factors.csv):
  multipliers, their stated reasons, and source units.
- [`ct_county_crosswalk.csv`](../data/workflow_assets/county_integration/ct_county_crosswalk.csv):
  land-area weights relating historical Connecticut counties to planning regions.
- [`export_rules.yaml`](../data/workflow_assets/county_integration/export_rules.yaml):
  field names, exclusions, ordering, and output naming.
- [`derived_metrics.yaml`](../data/workflow_assets/county_integration/derived_metrics.yaml):
  explicit adjusted-value components, denominator releases/columns/units and ratios.

The source files determine the estimates' spatial and temporal coverage.
Adjustment factors are assessment-supplied constants; the workflow does not
estimate inflation rates or infer undocumented source years.

## County geography and geographic harmonization

County identifiers are standardized as five-digit FIPS codes with leading
zeroes. County-level observations are joined by those codes. Multiple records
are summed only for sources whose records represent component estimates, such
as air-quality categories and urban-heat tracts. Duplicate records in sources
that provide a single county total are rejected.

### Connecticut counties and planning regions

Several sources use Connecticut's eight historical counties, while the output
uses nine planning regions. Historical county totals are allocated to planning
regions in proportion to land area.

The crosswalk is derived from the Census Bureau's
[2022 county subdivision to 2020 block group relationship file](https://www2.census.gov/geo/docs/maps-data/data/rel2022/acs22_cousub22_blkgrp20_st09.txt).
The [record-layout documentation](https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2022-connecticut-record-layout.html)
defines its fields. For each record, the first five characters of
`GEOID_BLKGRP_20` identify the historical county and the first five characters
of `GEOID_COUSUB_22` identify the planning region. `AREALAND_PART` is summed by
county–region pair; zero-area pairs are omitted. Each pair's area is divided by
the total represented land area of its historical county.

The resulting crosswalk has 19 positive-area links, with weights summing to
one within each historical county. The raw relationship file has SHA-256
checksum `06957a5e777dc96da4c678a79d27dce6617cfbd510c9611dbf7720c3cd16c5bd`.
The derived crosswalk is included with the analysis configuration.

This allocation assumes uniform value per unit land area within each historical
county. It is not a population-weighted or service-specific redistribution.
If a required historical county value is missing, the affected planning-region
estimate is null and known partial contributions are recorded separately.
Timber uses the explicit zero-filling convention described below. Other
unmatched county codes are reported without an assumed correspondence.

## Service estimation methods

### County and subcounty tabular estimates

The following measures are joined or aggregated before the listed factor is
applied. The `integration_services` table records the source field, method,
factor, and final output field for each measure.

| Service | Source field and county assignment | Source units/year | Factor |
| --- | --- | --- | --- |
| Grazing | Join `CountyGrazingValueYear` by `CountyCode` from the `SNAPPGrazingCalculations` sheet | Annual USD, 2024 | 1 |
| Air-quality avoided mortality | Sum `dollars_annual` across land-cover and pollutant rows by `FIPS` | Annual USD | 1.135 |
| Nature-related wages | Join the four Total Wages columns from the `totals by county` sheet by `County FIPS`; sum them for the combined wage total | USD wages, 2022 | 1.065 |
| Timber | Clamp negative `total_annual_rent` to zero, then join by `fips_padded` | Annual USD, 2024 | 1 |
| Urban heat | Sum `Sum of total_dollars_saved` by `COUNTY_GEOID` | USD, 2021 | 1.135 |
| Physical health | Join `total_value_usd` by `GEOID` | USD, 2023 | 1.027 |

Grazing and timber inputs are county totals, so no additional acreage
multiplication is applied. The wage fields measure money, not numbers of jobs.
They use the raw Total Wages columns for cultural activities, raw materials,
processing, and harvesting. State, national, metropolitan, and unallocated
summary rows are excluded.

Negative timber `total_annual_rent` values are set to zero at the source-county
level, before Connecticut crosswalk allocation or any county aggregation.
Thus a negative source county cannot offset a positive source county when both
contribute to a planning region. This is an assessment convention, not a change
to the source CSV. `integration_source_issues` records each original negative
value with reason `negative_source_clamped_zero`; `integration_services`
records the affected count, original negative total, increase from clamping,
and source totals before and after clamping. The existing timber missing-value
zero-filling convention and adjustment factor of 1 are unchanged.

The air-quality output,
`air_quality_avoided_mortality_annual_value_adj`, is the county sum of
`dollars_annual` multiplied by 1.135. The source release contains three
land-cover classes and three pollutants, giving nine records per source county.
These are category estimates, not raster pixels or per-area values.
Connecticut contributions are allocated using the land-area crosswalk and
recombined within each planning-region land-cover/pollutant category.

Explicitly null air-quality contributions propagate to null category and county
totals; counties with no source coverage are also null. The workflow does not
independently test whether every expected land-cover/pollutant row is present,
so source-category completeness remains a separate quality-control consideration.

### State fisheries allocation

Fisheries values are supplied at state scale and allocated using county
geographic weights:

```text
county value = state value × county weight / sum of county weights in that state
```

| Service | State source value | County allocation weight | Source units/year | Factor |
| --- | --- | --- | --- | --- |
| Marine fisheries | `ncp_value` | Coastline length | USD, 2023 | 1.027 |
| Inland fisheries | `Consumptive use value (USD)_Intermediate assumption` | Freshwater area | USD, 2022 | 1.065 |

The denominator is the sum of the supplied county weights within each state,
not an independently measured state coastline length or freshwater area.
The method assumes that state value is proportional to the selected geographic
measure. It does not locate fishing activity or beneficiaries within counties.

For a state with a numeric source value, counties with zero allocation weight
receive zero. States absent from the source or with null values remain null.
A supplied value without positive allocation support stops the analysis rather
than being distributed arbitrarily. County allocations are checked against
each supplied state total.

### Recreation polygon allocation

Recreation values come from the source polygons' `val_2024` field in 2024 USD.
A polygon's value is distributed among counties in proportion to overlap area:

```text
county contribution = polygon value × area(polygon ∩ county) / full polygon area
county recreation value = sum of its polygon contributions
```

Areas are calculated in the global equal-area projection EPSG:6933, including
Alaska, Hawaii, and the territories. Geometries are repaired for these area
calculations; the original county geometries are retained in the output.
The denominator is the full repaired source-polygon area. Value outside the
county footprint is recorded as unallocated, not redistributed among the
intersecting counties.

This method assumes uniform value within each source polygon. A $100 hexagon
with 30% of its area in one county and 70% in another contributes $30 and $70.
If only 80% of the hexagon intersects the county footprint, only $80 is allocated.

Numerical error in intersections of small slivers can produce summed county
shares slightly above one. Excess fractions of at most `1e-5` (0.001% of the
source area) are corrected by proportionally scaling that polygon's county
contributions to sum to one. Larger excesses stop the analysis and identify
the affected source polygons. The allocation table records the raw fraction,
corrected fraction, correction value, and unallocated value.

A county with no intersecting recreation polygon receives zero. This is a
consequence of the polygon coverage and aggregation method, not evidence of
measured zero recreational value.

### Mangrove study-unit allocation

Mangrove estimates use `Ben_Stock_2020` in the `UCSC_CWON_studyunits` layer of
`ucsc_cwon_studyunits.gpkg`. The assessment's
[Final datasets links spreadsheet, row 16](https://docs.google.com/spreadsheets/d/15Oqu4Vq2WWPRaleW8F0kxfZUBU3Bg6YDBbc9QZ7I9jk/edit#gid=0&range=A16:P16)
selects `ISO3`, `Country`, and `Ben_Stock_2020`, identifies the source currency
as USD 2020, and specifies factor 1.182. Its units cell still says "average
per-hectare"; for this integration, the assessment explicitly interprets
`Ben_Stock_2020` as a **total per study unit**, not a per-hectare rate. The
workflow does not independently establish an annual or net-present-value
interpretation for this field.

Only study units with `ISO3` equal to `USA`, `PRI`, or `VIR` are included.
This explicit country filter prevents offshore polygons assigned to neighboring
countries from contributing their values merely because they intersect a U.S.
county. It is applied before the spatial allocation; study-unit counts, source
totals, and the per-source allocation audit describe the selected units only.
`integration_services` records the country filter and the number of excluded
units. The source hash in `integration_sources` still identifies the complete
input GeoPackage, not a separately saved subset.

Each study unit's value is distributed over only its positive-area county
intersections, calculated in the equal-area projection EPSG:6933:

```text
county contribution = Ben_Stock_2020 × area(study unit ∩ county)
                      / sum of all positive county-intersection areas for that study unit
county mangrove value = sum of its study-unit contributions
```

Area outside all counties, including ocean, is excluded from the denominator.
For example, a $1,000 study unit intersecting 30 ha in one county and 10 ha in
another contributes $750 and $250, even if an additional 60 ha lies outside
the county footprint. If exactly one county intersects a study unit, that
county receives its entire value. Each county's contributions from exclusive
and shared study units are added together. This assumes uniform value per
unit of intersected county area; it does not locate flood-protection benefits
or beneficiaries within a study unit.

Boundary-only touches have zero area and receive no allocation. Study units
without any positive-area county intersection remain unallocated and are
reported in the audit. A county intersecting a study unit with a null value
has a null mangrove total, rather than a misleading partial sum of its known
contributions. Counties without positive-area study-unit intersections receive
zero under this allocation convention.

The county total is multiplied by 1.182 and exported as
`sum_mangrove_coastal_flood_protection_value_adj`, with
`mangroves_adj_factor` retained separately. No hectare multiplication is
applied. Unlike recreation, this method allocates the entire known study-unit
value whenever at least one positive-area county intersection exists;
recreation continues to use the full source-polygon area as its denominator.

This vector estimate replaces the categorical `mangrove_CONUS.tif` aggregate.
The historical `sum_mangrove_CONUS` field and its adjusted variant are omitted
from the county deliverables, even when present in the starting zonal results.

### Raster-derived county estimates and adjustments

The raster estimates are read from the supplied county zonal-results
GeoPackage. This integration step uses those county aggregates directly; it
does not sample rasters or change the zonal boundaries. Their spatial support
and missing-value conventions are inherited from the source analysis.

For each configured service:

```text
adjusted county estimate = source county estimate × adjustment factor
```

| Service | Source county field | Factor |
| --- | --- | --- |
| Pollination | `sum_pollination_attributed_annual_crop_yield_value` | 0.09 |
| Carbon per hectare | `sum_totalC_tCO2e_ha_2020` | 4.32612 |
| Carbon per pixel | `sum_totalC_tCO2e_pixel_2020` | 4.32612 |
| Dredging | `sum_avoided_dredging_costs_raster` | 1 |
| Safe Drinking Water Act health benefits | `sum_avoided_sdwa_health_costs_raster` | 1 |
| Water treatment | `sum_avoided_treatment_costs_raster` | 1 |
| Water provisioning total | `sum_Wval_sw_2020usd_total` | 1.182 |
| Water provisioning components | `sum_Wval_sw_2020usd_irr`, `sum_Wval_sw_2020usd_pow`, `sum_Wval_sw_2020usd_pub` | 1.182 |
| Flood mitigation NPV | `sum_marginal_npv_masked_to_wetlands` | 0.09 |
| Flood mitigation annual value | `sum_annual_value_masked_to_wetlands` | 0.09 |
| Corals | `sum_Coral_Reefs_2024adj_CPI` | 1 |
| Mental health | `sum_mental_health_national_existing_greenness_cost_90m` | 1 |

Pollination is supplied as dollars per hectare on a 30 m grid. Each pixel
represents 0.09 ha, so multiplying the raw county raster sum by 0.09 gives
`sum_pollination_attributed_annual_crop_yield_value_adj`. This area conversion
is applied once. The configured input contains the unscaled aggregate; an
already-adjusted county export is not an admissible input.

Carbon uses the assessment-supplied price of USD 3.66 per tonne CO2e and the
1.182 inflation multiplier: `3.66 × 1.182 = 4.32612`. This is a price
conversion, not a carbon-to-CO2 mass conversion.
`sum_CO2e_mitigation_dollar_value_per_pixel_adj` is the county sum of monetary
per-pixel values, despite `per_pixel` in its name. It is not a county average.
The separate carbon-per-hectare field is a sum of per-hectare values multiplied
by the price; no area integration is applied to that field. It is therefore
not a county dollar total.

The flood outputs are
`sum_flood_annual_value_masked_to_wetlands_adj` and
`sum_flood_marginal_npv_masked_to_wetlands_adj`. Annual value and net present
value have different time interpretations and are not additive.

Both flood rasters retain per-hectare values on a 30 m grid. The county builder
multiplies each raw zonal sum by `30 × 30 / 10,000 = 0.09` hectares, once, to
produce dollar totals. `flood_annual_adj_factor` and `flood_npv_adj_factor`
record this area conversion; it is not an inflation adjustment. Source TIFFs
and raw zonal outputs remain unscaled, and the zonal inventory must not apply
the same conversion upstream. To update existing combined county results,
rerun only `build_county_ecosystem_service_valuations_2026.py`; no raster
regeneration or zonal-statistics rerun is needed. Historical outputs are unchanged.

Coral inputs include a 2024 dollar adjustment and receive factor 1. A multiplier
alone does not validate source values or establish comparability with other
services.

## Output fields and interpretation

The CSV and GeoPackage contain matching county attributes. The county feature
layer is `tl_2024_us_county_50_states`; geometry is present only in the
GeoPackage. Import `GEOID` as text when using spreadsheet software so that
leading zeroes are preserved.

The layer name is an output label, not a geographic filter: all 3,225 counties
and county equivalents remain, including DC, Puerto Rico, and the U.S. Virgin
Islands. Historical inputs use the mislabeled `tl_2024_us_county_60_states`
layer. The builder reads that name as-is and renames only the new output,
including its GeoPackage registrations and spatial index. Source files and
older deliverables are not modified. `integration_run` records both the
source and output layer names.

Columns are ordered as follows:

1. Original county identifiers and geographic attributes, including `ALAND`
   and `AWATER`, in source order.
2. Retained `sum_` fields.
3. Other adjusted service totals, including the air-quality annual value.
4. Adjustment-factor fields.

An `_adj` suffix identifies a source or allocated county value multiplied by
its recorded factor. The associated `_adj_factor` field shows that multiplier,
including 1 where applicable. For example, `grazing_adj` and
`grazing_adj_factor` describe the grazing value and its factor.
`integration_services.output_field` identifies the exported value field for
each service, including those with descriptive flood, carbon, and air-quality
names.

The county table emphasizes totals and their factors. Generated means,
medians, extrema, standard deviations, percentiles, counts, valid-area
statistics, and coverage proportions are excluded. Geographic land and water
areas remain county attributes. There are no duplicate raw-value copies of
the adjusted estimates; service-level totals before and after adjustment are
recorded in the supporting tables.

The export excludes the four `sum_Wsw_irr`, `sum_Wsw_pow`, `sum_Wsw_pub`, and
`sum_Wsw_total` quantity fields, while retaining the distinct
`sum_Wval_sw_2020usd_*_adj` monetary provisioning fields. The duplicate
crop-yield field `sum_national_attributed_annual_crop_yield_value_zstd` is also
excluded. Other unconfigured sums, including the two unidentified
`CONUS_Coastal_2021_CCAP` bands, retain their source names and values. These
unidentified fields should not be interpreted as monetary valuations.

### Missing values, zeroes, and comparison limits

A null indicates that an estimate is unavailable under the source and
aggregation rules. It is not generally a measured zero. Explicitly null
components propagate to null totals in tabular aggregations, but the method
cannot detect every omitted source record.

There are specific zero conventions:

- Timber missing rows and missing values are assigned zero. Coverage statuses
  distinguish `no_timberland`, `no_species_coverage`, and absent data.
- Negative timber source-county values are clamped to zero before geographic
  allocation; the audit distinguishes this treatment from missing-value filling.
- Urban-heat accounting-format `$ -` cells are interpreted as zero.
- Fisheries counties with zero allocation weight receive zero within states
  that have numeric source values.
- Recreation counties without intersecting source polygons receive zero.
- Mangrove counties without positive-area study-unit intersections receive zero;
  intersections with null-valued study units instead propagate null totals.

Unmatched county records, including historical Alaska codes, are listed in the source
issues table. Connecticut coverage depends on whether the contributing
historical county estimates are available; a land-area crosswalk does not
supply missing service data.

Comparisons should account for coverage, allocation assumptions, units, and
time horizon. In particular, wages and water provisioning each include
component fields as well as their combined total. Adding both double-counts
those components. Annual values, flood NPV, and the non-area-integrated
carbon-per-hectare field cannot be combined as a single monetary total.
The `_adj` suffix records an adjustment, not proof of economic comparability
or independence.

## County totals and ratios

### What the summary columns mean

Each county row contains `es_total`, the sum of 19 adjusted service values,
and four ways to compare that total with the county's size, economy and
population. The [component list](../data/workflow_assets/county_integration/derived_metrics.yaml)
gives the exact fields used. Annual flood benefits are included; flood NPV is
kept separately. Where the table contains both a service total and its component
parts, the calculation uses the total once.

| Column | Calculation | Units |
| --- | --- | --- |
| `es_total` | Sum of the selected adjusted service fields | Dollars across the included valuations |
| `es_per_ha` | `es_total / area_ha_county` | Dollars per county hectare |
| `es_per_ha_natural` | `es_total / area_ha_all_natural` | Dollars per observed natural-land hectare |
| `es_per_gdp` | `es_total / GDP_2024` | Ratio; 0.02 means the total is numerically 2% of GDP |
| `es_per_capita` | `es_total / Pop_2024` | Dollars per resident |

Treat `es_total` as an assessment summary. Its components include annual
benefits, wages and carbon-stock valuation, so their time periods and economic
meanings differ. In particular, `es_per_gdp` is a size comparison, not a measure
of how much these services contribute to annual GDP. The
[interpretation notes](#interpreting-the-values) describe these limits.

A blank service value contributes zero to the sum. Its original service column
stays blank, so you can distinguish missing coverage from an estimated zero.

### Where the denominators come from

The output keeps all four denominator columns. They are joined by five-digit
county FIPS, preserving leading zeroes.

| Column | Source |
| --- | --- |
| `area_ha_county` | `intersect_area_ha_tl_2024_us_county_50_states` in `counties_area_20260910_130816.csv`, matched by `GEOID` |
| `area_ha_all_natural` | Sum of the corrected NLCD forest, grassland, shrubland, water/snow and wetland areas |
| `GDP_2024` | `2024` in `CAGDP1_LineCode3_All_States.csv`, LineCode 3, matched by `GeoFIPS`. Values are in thousands of dollars and are multiplied by 1,000 |
| `Pop_2024` | `POPESTIMATE2024` in `co-est2025-alldata.csv`, county rows (`SUMLEV=050`), matched by `GEOFIPS` |

Full source paths are in
[`derived_metrics.yaml`](../data/workflow_assets/county_integration/derived_metrics.yaml).
The GDP and population files are under `data/analysis_inputs/`; the county-area
and corrected NLCD files are earlier analysis results.

Natural-land area is the area of the five selected NLCD classes.
`area_ha_valid_nlcd` measures the larger footprint with usable NLCD observations.
The natural-land ratio uses the former. For a county outside NLCD coverage,
zero observed hectares means the data are unavailable, rather than that the
county has no natural land.

### Why a ratio may be blank

A ratio is blank when its denominator is missing or nonpositive. In the
October 26 release:

- 134 counties lack a matching GDP row: 81 territory rows and 53 Hawaii/Virginia
  counties or cities represented only by combined BEA regions. Those combined
  amounts remain in the source records rather than being assigned to each county.
- 81 territory rows lack population estimates.
- 116 counties have no observed NLCD coverage, so their natural-land ratios are blank.

To investigate a particular county, find its `GEOID` in the GeoPackage's
`derived_metric_county_status` table. This records the denominator match and the
number of service values treated as zero.

### Trace a result or run it again

Start with `derived_metric_rules` inside the result GeoPackage. It saves the
YAML used for that run, even if the repository settings have since changed.
`derived_metric_sources` gives the source paths and file hashes;
`derived_metric_components` lists the included fields, their totals across
counties, and their blank counts. Unmatched source rows are recorded in
`derived_metric_source_issues`.

For a new output from existing county values, use the
[finalization command in the README](../README.md#add-county-totals-and-ratios-without-rerunning-zonals).
It copies the original values, factors and geometry, then appends the summaries
to a new timestamped CSV/GeoPackage pair. Reuse that original input when changing
settings. The full county builder performs the same calculation after applying
service adjustments; a county mask update uses the areas it has just calculated.

## Quality checks and provenance

The workflow checks identifiers and allocation support, reconciles state
fisheries, recreation, and mangrove source totals, and verifies that the output preserves
county geometries and retained attributes. Source-file hashes are checked
before publication. CSV and GeoPackage attributes are compared for matching
fields, order, and values.

The GeoPackage includes supporting tables for evaluating the estimates:

| Question | Table |
| --- | --- |
| Which source field, units, allocation method, and factor produced a service? | `integration_services` |
| Which exact input and configuration files were used? | `integration_sources` |
| Is a county estimate observed, allocated, zero-filled, or missing? | `integration_county_status` |
| Which records were unmatched or contributed only to an incomplete estimate? | `integration_source_issues` |
| Do county fisheries allocations reproduce their state totals? | `integration_state_checks` |
| How much recreation value is allocated, corrected, or outside the county footprint? | `integration_recreation_allocation` |
| Which mangrove study units were allocated, and which had no county-intersection support? | `integration_mangrove_allocation` |
| Which countries' mangrove study units were selected, and how many were excluded? | `integration_services` |
| Which timber source values were clamped, and how much did clamping change the total? | `integration_source_issues`, `integration_services` |
| Which Connecticut allocation weights were used? | `integration_ct_crosswalk` |
| Which naming and exclusion rules produced the exported fields? | `integration_export_rules`, `integration_export_fields` |
| What does the air-quality annual total represent? | `integration_summary_fields` |
| Which script version, execution, and intermediate results identify the run? | `integration_run`, `integration_checkpoints` |

These tables distinguish the source estimates from the geographic allocation
and adjustment decisions made by this workflow. They are the run-specific
record for interpreting a particular deliverable.

## Reproducing the dataset

The large input datasets are stored at the paths listed above. The five
configuration files are included under
`data/workflow_assets/county_integration/`. Activate the geospatial environment
described in the [README](../README.md#environment), then run from the repository
directory:

```powershell
python build_county_ecosystem_service_valuations_2026.py --workers 16 --geometry-workers 16
```

The integration uses the configured county aggregates, so this step does not
require a raster zonal-statistics run. Rerunning this builder is sufficient to
replace the old categorical mangrove aggregate with the study-unit allocation
and to apply the timber clamp. The default output pair is:

```text
data/analysis_results/combined/snapp_ecosystem_services_counties_YYYY_MM_DD_HH_MM_SS.csv
data/analysis_results/combined/snapp_ecosystem_services_counties_YYYY_MM_DD_HH_MM_SS.gpkg
```

The script prints both paths. Source files and existing deliverables are
preserved. `--data-root` selects the input data root, `--config-dir` selects the
five configuration files, and `--output <name>.gpkg` specifies a new output path
with a matching CSV. See `--help` for execution options.

### Export rules

[`export_rules.yaml`](../data/workflow_assets/county_integration/export_rules.yaml)
records the presentation of the results, separately from the source paths,
adjustment factors, and county crosswalk. It contains:

| Setting | Role |
| --- | --- |
| `rename_fields` | Maps generated value and factor names to their final output names |
| `exclude_fields` | Specifies required source or generated fields omitted from the export |
| `optional_exclude_fields` | Specifies fields omitted when present |
| `exclude_patterns` | Identifies groups of generated statistics and diagnostics to omit |
| `column_groups` | Orders county attributes, sums, other totals, and factors |
| `output.directory`, `output.filename_prefix` | Define the output location relative to the data root and the filename prefix |
| `output.layer_name` | Sets the new GeoPackage feature-layer name without changing data or coverage; omission retains the source name |

For example, the annual flood field is identified by:

```yaml
rename_fields:
  sum_annual_value_masked_to_wetlands_adj: sum_flood_annual_value_masked_to_wetlands_adj
```

The rules affect presentation, not the allocation methods or multipliers.
The complete YAML text and its hash are embedded in each GeoPackage, together
with the resolved action for each field. This preserves the meaning of the
exported names and omissions as part of the scientific record.
