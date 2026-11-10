# County ecosystem-service valuations for the 2026 SNAPP assessment

## Purpose and scope

This dataset brings ecosystem-service estimates onto a common county geography
for the 2026 SNAPP nature assessment. It combines raster-derived county
sums with tabular, state-level, polygon, and point estimates, then applies the
assessment's recorded unit conversions and valuation adjustments.

The output has one record for each of 3,225 counties or county equivalents,
including Puerto Rico and the U.S. Virgin Islands. County boundaries and
identifiers come from the county zonal-results GeoPackage selected in `sources.csv`.
Coverage varies by service: retaining a county in the output does not imply
that every service has an estimate there.

These data support mapping and service-specific comparisons. They do not
constitute a single, additive measure of ecosystem value. Some estimates are
annual flows, others are net present values or wages, and several component
fields are also represented in a total.

The implementation is
[`build_county_ecosystem_service_valuations_2026.py`](../build_county_ecosystem_service_valuations_2026.py).
This guide describes its inputs, allocation assumptions, adjustments, and
interpretation limits. See the [supporting workflows](supporting_workflows.md)
for standalone raster zonal-statistics and PAD-US commands.

The valuation builder includes county NLCD areas from the mask-area GeoPackage
selected in `derived_metrics.yaml`: `area_ha_valid_nlcd`, five
`area_ha_nlcd_<class>` values, `area_ha_all_natural`, and five
`proportion_nlcd_<class>_of_valid` values.
These observed areas are not the full county polygon area. Source NLCD NoData
has zero observed coverage and undefined (null) class proportions.
The area fields are nonmonetary: no adjustment factor is applied. Use the
[mask-only command](supporting_workflows.md#calculate-county-nlcd-areas)
to calculate those areas independently of the service zonals. That command can
also append the areas, totals and ratios to completed valuations that lack them.

## Input data

The analysis uses the releases listed in
[`sources.csv`](../data/workflow_assets/county_integration/sources.csv).
Paths in the table are relative to the repository's `data/` directory.

| Dataset | File within `data/` |
| --- | --- |
| Raster-derived county results and county boundaries | `analysis_results/zonal_statistics/counties/counties_ecosystem_services_20260910_130816.gpkg` |
| County coastline lengths | `analysis_results/zonal_statistics/counties/counties_coastline_length_20260910_130816.gpkg` |
| TIGER coastline lines for marine fisheries | `analysis_inputs/linear_features/coastline/tl_2019_us_coastline_50_states.gpkg` |
| County freshwater areas | `analysis_results/zonal_statistics/counties/counties_freshwater_area_20260910_130816.csv` |
| Prepared NHD freshwater polygons | `analysis_inputs/hydrography/nhdfreshwater/nhd_freshwater_clipped_to_usa_2026_05_18_01_44_14.gpkg` |
| Great Lakes source feature IDs | `workflow_assets/county_integration/great_lakes_nhd_features.csv` |
| Great Salt Lake source feature IDs | `workflow_assets/county_integration/inland_fisheries_exclusions.csv` |
| Grazing | `analysis_inputs/ecosystem_services/grazing/SNAPPGrazingDataset_final_062326.xlsx` |
| Nature-related wages | `analysis_inputs/ecosystem_services/jobs/Finalized Flagged and Sorted Industry and County Codes.xlsx` |
| Timber | `analysis_inputs/ecosystem_services/timber/total_valuation_all_regions_fvs.csv` |
| Air quality | `analysis_inputs/ecosystem_services/air_quality/snapp_air_quality.csv` |
| Air-quality county-code aliases | `workflow_assets/county_integration/air_quality_county_aliases.csv` |
| Marine fisheries | `analysis_inputs/ecosystem_services/marine_fisheries/NCPvalue_landings_marine_fish_noaa_2023_state.csv` |
| Inland fisheries | `analysis_inputs/ecosystem_services/inland_fisheries/Harvest_table_with_value.csv` |
| Urban heat | `analysis_inputs/ecosystem_services/urban_heat/HEAT_all_tracts_2021.csv` |
| Physical health | `analysis_inputs/ecosystem_services/physical_health/hypertension_linear_us_counties_all.gpkg` |
| County recreation | `analysis_inputs/recreation/2026_11_02/usa_nature_assessment_recreation_20261102.csv` |
| Mangrove coastal flood protection | `analysis_inputs/ecosystem_services/mangroves/ucsc_cwon_studyunits.gpkg` |
| Coral coastal protection | `analysis_inputs/ecosystem_services/aeb_coral.gpkg` |

Additional files record the assessment decisions:

- [`adjustment_factors.csv`](../data/workflow_assets/county_integration/adjustment_factors.csv):
  multipliers, their stated reasons, and source units.
- [`ct_county_crosswalk.csv`](../data/workflow_assets/county_integration/ct_county_crosswalk.csv):
  land-area weights relating historical Connecticut counties to planning regions.
- [`export_rules.yaml`](../data/workflow_assets/county_integration/export_rules.yaml):
  field names, exclusions, ordering, and output naming.
- [`derived_metrics.yaml`](../data/workflow_assets/county_integration/derived_metrics.yaml):
  explicit adjusted-value components, denominator releases/columns/units and ratios.
- [`virginia_gdp_crosswalk.csv`](../data/workflow_assets/county_integration/virginia_gdp_crosswalk.csv):
  BEA combined-area codes and their constituent Virginia county/city GEOIDs.

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
| Grazing | Join `CountyGrazingValueMonth` by `CountyCode` from the `SNAPPGrazingCalculations` sheet; see [grazing valuation](#grazing-valuation) for the time basis | Annual USD, 2024 | 1 |
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
Before aggregation, the workflow applies the reviewed one-to-one county codes
in `air_quality_county_aliases.csv`. This maps source `46113` (Shannon County,
South Dakota) to `46102` (Oglala Lakota County), following the
[Census county name/code change](https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes/2015/geography-changes.html).
This is an identifier correction, not an area- or population-weighted allocation.
The supplied air-quality CSV remains unchanged. Duplicate county/land-cover/pollutant
records after normalization stop the run instead of counting the same contribution
twice. The county method metadata records applied aliases and their source row
counts and known dollar totals; the alias file is fingerprinted with the run.
Connecticut contributions are allocated using the land-area crosswalk and
recombined within each planning-region land-cover/pollutant category.

Explicitly null air-quality contributions propagate to null category and county
totals; counties with no source coverage are also null. The workflow does not
independently test whether every expected land-cover/pollutant row is present,
so source-category completeness remains a separate quality-control consideration.

#### Grazing valuation

The assessment treats the workbook's `MEAN_Biomass_AVG`, a 2016–2025
Google Earth Engine mean in pounds per acre, as mean **annual** forage
production. The original Earth Engine script and time aggregation were not
available to verify directly. The separate 2025 RAP biomass estimate in the
April workbook is on the same scale. As an independent check, the workbook's
decade mean for Cherry County, Nebraska is 1,722 lb/acre, comparable to the
[RAP year-end report](https://rangelands.app/production-report/county#fips=31031;year=2025;doy=353;reference=rolling;metric=ytd;perAcre=true)
of 1,740 lb/acre for the 2015–2024 annual mean and 1,861 lb/acre for 2025.
These are scale checks, not evidence of identical inputs or acreage masks.

The workbook multiplies biomass by grazing acres and a 25% usable-forage
share. Dividing that annual supply by
[780 lb per animal-unit month](https://www.fs.usda.gov/pnw/pubs/pnw_gtr248.pdf#page=19)
gives the total animal-unit months the year's forage can support. Pricing
those months at the workbook's monthly grazing fee gives the annual dollar
value. The assessment therefore reads the subtotal named
`CountyGrazingValueMonth` rather than `CountyGrazingValueYear`, which multiplies
that subtotal by eight. The source workbook is unchanged. The adjustment
factor remains 1, and land products receive the same county land share of the
corrected value.

### State fisheries allocation

Fisheries values are supplied at state scale and allocated using county
geographic weights:

```text
county value = state value × county weight / sum of county weights in that state
```

| Service | State source value | County allocation weight | Source units/year | Factor |
| --- | --- | --- | --- | --- |
| Marine fisheries | `ncp_value` | Marine coastline length (km) | USD, 2023 | 1.027 |
| Inland fisheries | `Consumptive use value (USD)_Intermediate assumption` | Freshwater area outside the applicable lake exclusions (ha) | USD, 2022 | 1.065 |

For marine fisheries, the county builder starts with the county lengths from
the `coastline` row in `sources.csv`. It uses the TIGER GeoPackage in the
`coastline_lines` row to measure Great Lakes lines (`NAME = 'Great Lakes'`)
and the remaining coastline separately. It intersects each group with the county
polygons and measures length in the EPSG:6933 projected CRS. A county with
positive Great Lakes length uses its measured length from the remaining lines,
or zero if none intersect. All other counties use their supplied coastline
lengths. These weights determine how the state value is distributed among its
counties; the state source value and adjustment factor do not depend on the
measured length.

For inland fisheries, the county builder starts with the county areas from
the `freshwater` row in `sources.csv`. It reads the prepared NHD polygons from
`freshwater_polygons` and selects `NHDWaterbody` features by source FID.
`great_lakes_features` selects the nine features for Lakes Superior, Michigan,
Huron, Erie, and Ontario. `inland_fisheries_exclusions` selects the 26 features
for Great Salt Lake (GNIS ID `01441459`), which is excluded in Utah. The
selection tables are tied to the configured NHD release; verify both tables
against the source if changing releases.

The builder unions each selection so overlapping polygons are only excluded
once. It remeasures only counties with a positive-area intersection with
their applicable exclusion. For those counties, it sums the remaining
per-feature freshwater intersection areas in EPSG:6933, converted to hectares.
Other counties retain their supplied freshwater areas. These exclusions affect
only inland fisheries allocation; they do not change the source state totals,
the adjustment factor, marine fisheries, or the general freshwater zonals.

The denominator is the sum of the county weights within each state,
not an independently measured state coastline length or freshwater area.
The method assumes that state value is proportional to the selected geographic
measure. It does not locate fishing activity or beneficiaries within counties.

For a state with a numeric source value, counties with zero allocation weight
receive zero. States absent from the source or with null values remain null.
A supplied value without positive allocation support stops the analysis rather
than being distributed arbitrarily. County allocations are checked against
each supplied state total.

### County recreation values

The recreation CSV contains county estimates for recreational use of protected
areas in 2024. Keep the supplied
`usa_nature_assessment_recreation_readme_20261102.txt` beside the CSV locally
and follow its use and distribution terms.

The builder joins `county_cs_2024usd` to the county layer on five-character
`GEOID`, preserving leading zeroes. The field is the estimated annual use
value of visitation in 2024 USD. A factor of 1 gives `recreation_adj`;
`recreation_adj_factor` records that factor. `county_visits_2024` counts
visitor-days and is not used in the valuation. These estimates belong to the
whole-county export; the CSV does not provide recreation values for individual
PAD-US or BBB footprints.

Counties without a matching value have a null recreation estimate. A supplied
zero remains zero. See `integration_sources` for the CSV used in a run and
`integration_county_status` for its county coverage.

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
`mangroves_adj_factor` retained separately. No hectare multiplication is applied.

The valuation export excludes `sum_mangrove_CONUS` and its adjusted variant:
they represent categorical land cover, not monetary values.

### Coral point allocation

Coral estimates use `BC_USD11` in the `aeb_coral` layer of `aeb_coral.gpkg`,
with `OBJECTID` identifying each point. The supplied release contains nine
points. Each point's entire value is assigned **once** to its nearest county
polygon, and all assigned values are summed per county:

```text
sum_aeb_coral_BC_USD11_adj = sum(assigned points' BC_USD11) × 1.368
```

The assessment-supplied multiplier is recorded in `adjustment_factors.csv`
and exported as `corals_adj_factor`. The source field is treated as a USD 2011
total per point, not a per-hectare or per-pixel rate. No rasterization or pixel
area conversion is performed. The multiplier alone does not establish the
source's benefit time horizon or comparability with other services.

Distance is measured to the full county polygon (not its centroid) in metres
using a WGS84 azimuthal-equidistant projection centered on each source point.
County polygons are repaired when necessary but are not simplified; the
original output county geometry is preserved. Points inside a polygon have
zero distance. Offshore points have no maximum-distance cutoff. If distances
tie within one micrometre, the lowest GEOID wins; no point is duplicated among
counties, even if county polygons overlap. `integration_coral_allocation`
records every source identifier, raw value, assigned GEOID, distance, method,
and tied counties. These are spatial assignments, not observations of which
county receives the economic benefit.

Counties without assigned points receive zero under this coverage convention,
not a measured absence of coral benefits. A null-valued assigned point makes
its county total null, rather than allowing a partial sum. Invalid source
identifiers, non-point geometries, or nonnumeric values stop the analysis.
Known source values are conserved before any incomplete county totals are
withheld; both quantities are recorded in `integration_services`.

Use the point GeoPackage directly in the county builder. Coral raster sums, if
present in the input county zonals, are excluded from the valuation export.

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
| Carbon stock value | `sum_totalC_tCO2e_ha_2020` × 0.09 ha per pixel | 4.32612 |
| Dredging | `sum_avoided_dredging_costs_raster` | 1 |
| Safe Drinking Water Act health benefits | `sum_avoided_sdwa_health_costs_raster` | 1 |
| Water treatment | `sum_avoided_treatment_costs_raster` | 1 |
| Water provisioning total | `sum_Wval_sw_2020usd_total` | 1.182 |
| Water provisioning components | `sum_Wval_sw_2020usd_irr`, `sum_Wval_sw_2020usd_pow`, `sum_Wval_sw_2020usd_pub` | 1.182 |
| Flood mitigation NPV | `sum_marginal_npv_masked_to_wetlands` | 0.09 |
| Flood mitigation annual value | `sum_annual_value_masked_to_wetlands` | 0.09 |
| Mental health | `sum_mental_health_national_existing_greenness_cost_90m` | 1 |

Pollination is supplied as dollars per hectare on a 30 m grid. Each pixel
represents 0.09 ha, so multiplying the raw county raster sum by 0.09 gives
`sum_pollination_attributed_annual_crop_yield_value_adj`. This area conversion
is applied once. The configured input contains the unscaled aggregate; an
already-adjusted county export is not an admissible input.

Carbon stock valuation uses the zonal sum of `totalC_tCO2e_ha_2020.tif`, a
30 m raster in tonnes CO2e per hectare. The valuation first multiplies the
sum by `0.09` hectares per pixel to obtain tonnes CO2e, preserving fractional
tonnes. It then multiplies by the assessment-supplied price and inflation
factor: `3.66 × 1.182 = 4.32612` USD per tonne CO2e. The per-pixel TIFF and
its zonal sums are not used, so saved raw zonals can be reused as long as
they contain the per-hectare sum.

The existing output name `sum_CO2e_mitigation_dollar_value_per_pixel_adj`
identifies the area-integrated carbon stock dollar total, not a per-pixel rate
or a county average. Its existing factor field, `carbon_pixel_adj_factor`,
records `4.32612`; the `0.09` area conversion is applied separately. Density
sums remain internal calculation inputs. The final CSV and GeoPackage omit
the unintegrated density sum and its associated factor so they cannot be
mistaken for an additional carbon valuation.

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
regeneration or zonal-statistics rerun is needed.

## Output fields and interpretation

The CSV and GeoPackage contain matching county attributes. The county feature
layer is `tl_2024_us_county_50_states`; geometry is present only in the
GeoPackage. Import `GEOID` as text when using spreadsheet software so that
leading zeroes are preserved.

The layer name is an output label, not a geographic filter: all 3,225 counties
and county equivalents remain, including DC, Puerto Rico, and the U.S. Virgin
Islands. `sources.csv` specifies the input feature-layer name;
`export_rules.yaml` specifies the output name. `integration_run` records both.

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
- Recreation counties without a matched value remain null; source zeros
  remain zero.
- Coral counties without assigned points receive zero; assigned null-valued
  points instead propagate null totals.
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
| `es_per_ha` | `es_total / area_ha_county_excl_ocean_great_lakes` | Dollars per county hectare, excluding ocean and Great Lakes |
| `es_per_ha_natural` | `es_total / area_ha_all_natural` | Dollars per observed natural-land hectare |
| `es_per_gdp` | `es_total / GDP_2024` | Ratio; 0.02 means the total is numerically 2% of GDP |
| `es_per_capita` | `es_total / Pop_2024` | Dollars per resident |

Treat `es_total` as an assessment summary. Its components include annual
benefits, wages and carbon-stock valuation, so their time periods and economic
meanings differ. In particular, `es_per_gdp` is a size comparison, not a measure
of how much these services contribute to annual GDP. The
[interpretation notes](#missing-values-zeroes-and-comparison-limits) describe these limits.

A blank service value contributes zero to the sum. Its original service column
stays blank, so you can distinguish missing coverage from an estimated zero.

### Where the denominators come from

The output keeps all four denominator columns. Direct matches use five-digit
county FIPS, preserving leading zeroes. The Virginia GDP allocation described
below maps combined-area codes to their constituent county/city codes.

| Column | Source |
| --- | --- |
| `area_ha_county_excl_ocean_great_lakes` | County geometry area after subtracting mapped ocean water and the five NHD Great Lakes footprints, matched by `GEOID` |
| `area_ha_all_natural` | Sum of the NLCD forest, grassland, shrubland, water/snow and wetland areas measured within the same reporting footprint |
| `GDP_2024` | `2024` in `CAGDP1_LineCode3_All_States.csv`, LineCode 3, matched by `GeoFIPS` or allocated from a mapped Virginia combined area. Values are in thousands of dollars and are multiplied by 1,000 |
| `Pop_2024` | `POPESTIMATE2024` in `co-est2025-alldata.csv`, county rows (`SUMLEV=050`), matched by `GEOFIPS` |

Full source paths are in
[`derived_metrics.yaml`](../data/workflow_assets/county_integration/derived_metrics.yaml).
The GDP and population files are under `data/analysis_inputs/`; prepare the
county-area and NLCD files together with
[`prepare_county_reporting_areas.py`](../prepare_county_reporting_areas.py).
The CSV denominator and GeoPackage natural areas must carry matching county
areas. A mismatched or older unmasked NLCD release is rejected.

The reporting footprint uses the TIGER marine shoreline to identify connected
nonland water outside the local U.S. boundary, including across county lines.
It also excludes the five selected Great Lakes footprints from the clipped
NHD layer. It preserves Census land and retains other inland water such as
Great Salt Lake and Tahoe. The generalized boundary approximates the coastal
cutoff; some tidal waters may remain. This reporting denominator does not
replace the assessment geometries or the land-only `ALAND` used to allocate
county services. Natural-land area is the area of the five selected NLCD
classes within that footprint.
`area_ha_valid_nlcd` measures the larger footprint with usable NLCD observations.
The natural-land ratio uses the former. For a county outside NLCD coverage,
zero observed hectares means the data are unavailable, rather than that the
county has no natural land.

### Virginia combined-area GDP

BEA reports some Virginia counties and independent cities as combined areas.
The [crosswalk](../data/workflow_assets/county_integration/virginia_gdp_crosswalk.csv)
maps each combined-area code to its constituent GEOIDs, with the official BEA
source recorded in the file. Select this file with
`sources.gdp.allocation_crosswalk` in `derived_metrics.yaml`.

The combined GDP is divided by 2024 population share:

```text
county GDP_2024 = combined-area GDP_2024 × county Pop_2024 / combined-area Pop_2024
```

The denominator uses all members of the combined area from the population
source, even when the output contains only a subset of them. The allocations
across the full group sum to its original GDP; directly matched county GDP
values are not altered or given an additional allocation. This preserves
the original totals without repeating a combined-area value on each row.

`GDP_2024_is_allocated` is `1` for these population-allocated estimates and
`0` otherwise. A zero flag does not imply that a missing GDP value is available.
The populated county GDP then supplies the denominator for `es_per_gdp`.
Population allocation is an estimate, not a separate BEA estimate for each
jurisdiction: population measures residents rather than where production
occurs.

The GeoPackage's `derived_metric_gdp_allocations` table records the source
group, population weights and county allocations.
`derived_metric_gdp_group_checks` reconciles each full group to its original
GDP value. These tables distinguish population allocation from direct GDP
matches.

### Why a ratio may be blank

A ratio is blank when its denominator is missing or nonpositive. Common causes
are an absent county row in the GDP or population table, or a county outside
NLCD coverage. A GDP record that has neither a direct county match nor a
configured combined-area mapping leaves `es_per_gdp` blank. Missing or invalid
population for a mapped Virginia member is an allocation error, not an assumed
zero population.

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

To refresh a completed CONUS assessment after changing the GDP source or
allocation rules, reuse the original raster run. From the repository root:

```powershell
python run_conus_ecosystem_services.py --reuse-zonals-from '<original-raster-run-id>' --run-id '<new-run-id>'
```

This reads saved geometries and raster sums, recalculates the valuations and
county denominators, and publishes a new set of output pairs without rerunning
the raster zonals. Use the original raster run ID, not a valuation-only rerun ID.

For a new output from existing county values, use the
[county finalization command](supporting_workflows.md#build-or-finalize-county-valuations).
It copies the original values, factors and geometry, then appends the summaries
to a new timestamped CSV/GeoPackage pair. Reuse that original input when changing
settings. The full county builder performs the same calculation after applying
service adjustments; a county mask update uses the areas it has just calculated.

## Quality checks and provenance

The workflow checks identifiers and allocation support, reconciles state
fisheries, mangrove, and coral source totals, and verifies that the output
preserves county geometries and retained attributes. Source-file hashes are checked
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
| Which mangrove study units were allocated, and which had no county-intersection support? | `integration_mangrove_allocation` |
| Which county received each coral point, at what distance, and were there ties? | `integration_coral_allocation` |
| Which countries' mangrove study units were selected, and how many were excluded? | `integration_services` |
| Which timber source values were clamped, and how much did clamping change the total? | `integration_source_issues`, `integration_services` |
| Which Connecticut allocation weights were used? | `integration_ct_crosswalk` |
| Which Virginia GDP values were population-allocated, and do they preserve each source total? | `derived_metric_gdp_allocations`, `derived_metric_gdp_group_checks` |
| Which naming and exclusion rules produced the exported fields? | `integration_export_rules`, `integration_export_fields` |
| What does the air-quality annual total represent? | `integration_summary_fields` |
| Which script version, execution, and intermediate results identify the run? | `integration_run`, `integration_checkpoints` |

These tables distinguish the source estimates from the geographic allocation
and adjustment decisions made by this workflow. They are the run-specific
record for interpreting a particular deliverable.

## Reproducing the dataset

The input datasets are stored at the paths listed above. The configuration
files are included under
`data/workflow_assets/county_integration/`. Activate the geospatial environment
described in the [README](../README.md#1-set-up-python), then run from the repository
directory:

```powershell
python build_county_ecosystem_service_valuations_2026.py --workers 16 --geometry-workers 16
```

The integration uses the configured county aggregates and NLCD mask areas.
It calculates fisheries weights from the configured coastline, freshwater,
and lake exclusion inputs as part of the build; no separate zonal statistics
run is needed.
The default output pair is:

```text
data/analysis_results/combined/snapp_ecosystem_services_counties_YYYY_MM_DD_HH_MM_SS.csv
data/analysis_results/combined/snapp_ecosystem_services_counties_YYYY_MM_DD_HH_MM_SS.gpkg
```

The script prints both paths. Source files and existing deliverables are
preserved. `--data-root` selects the input data root, `--config-dir` selects the
configuration directory, and `--output <name>.gpkg` specifies a new output path
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
