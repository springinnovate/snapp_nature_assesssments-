# CONUS Ecosystem Services Assessment

We estimate the monetary value of ecosystem services for counties and selected
lands in the contiguous 48 states and Washington, DC (CONUS). The assessment
produces six output layers, each as a CSV and GeoPackage pair:

1. Counties
2. Public lands by county
3. Public lands by county and manager type
4. Conservation lands by county
5. Conservation lands by county and manager type
6. Big Beautiful Bill (BBB) candidate lands by county

## How the assessment works

County boundaries provide a common geography for comparing ecosystem-service
estimates across counties and the lands within them. We combine spatial models
of services such as carbon storage and flood mitigation with county and state
estimates of services such as grazing and fisheries. Public and conservation
lands are separate, overlapping classifications. The BBB layer represents a
separate candidate land-sale scenario.

The methods follow four stages:

1. [Define analysis areas](#define-analysis-areas) from county boundaries,
   public and conservation lands, and [BBB candidate lands](#identify-bbb-candidate-lands)
   to establish the geography of each output.
2. [Measure area and shoreline](#measure-area-and-shoreline) to compare
   differently sized counties and distribute county and state service totals
   to the lands within them. [Sum raster estimates](#summarize-raster-values)
   of ecosystem services within each analysis area.
3. [Allocate service estimates](#allocate-service-values) supplied by county,
   state, or study area to the analysis areas, then [adjust monetary
   values](#adjust-monetary-values) for the source price years.
4. [Calculate totals and county ratios](#calculate-totals-and-county-ratios)
   to report ecosystem-service value per hectare, per resident, and relative
   to county GDP alongside the individual service estimates.

## Methods

### Define analysis areas

We use the Census Bureau's
[2024 TIGER/Line county boundaries](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?layergroup=Counties+%28and+equivalent%29&year=2024)
to define the whole-county analysis and divide selected lands by county.
The boundary file includes county equivalents such as independent cities and
Connecticut planning regions.

We use the USGS [PAD-US 4.1 Vector Analysis file](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-spatial-analysis-and-reports)
to identify public and conservation lands. The file maps public lands and
protected areas and records their management and conservation status.

- **Public lands.** The [public-land filter](config/padus_land_rules.rules)
  selects records with federal, state, local government, special district,
  joint, or territorial manager types. For records coded as having an unknown
  manager type, it selects the designations listed in the rule.
- **Conservation lands.** We select records with GAP status 1, 2, or 3,
  independently of the public-land filter.

We clip the selected polygons to county boundaries, then merge overlapping
polygons within each land category and within each category's manager types.
The county-level public and conservation layers each contain the union of
their selected polygons. Manager areas can overlap, so their areas and service
estimates are not necessarily additive. Public and conservation estimates
also overlap and should be compared separately.

### Measure area and shoreline

Area measurements let us compare ecosystem-service value per hectare across
counties of different sizes and allocate county service estimates to public,
conservation, and BBB lands. Shoreline length and inland water area let us
distribute statewide fisheries totals among counties and land categories.

We retain the full county-clipped polygons, including any water, for raster
summaries and spatial service allocations, and measure land-only hectares
separately. We measure vector areas and lengths in the equal-area projection
EPSG:6933.

- **Land area.** We intersect public, conservation, and BBB polygons with the
  2024 Census land mask, constructed from TIGER/Line Faces classified as land. The
  intersection area divided by Census county land area (`ALAND`) gives the
  fraction of county land in each category. County land hectares come directly
  from `ALAND`, converted from square metres.
- **County area.** We measure two areas for each county. The first excludes the
  [NHD Great Lakes polygons](data/workflow_assets/county_integration/great_lakes_nhd_features.csv)
  and water outside the U.S. boundary connected to the TIGER marine shoreline.
  The Census land mask protects land from removal; the generalized coastal
  boundary can retain some tidal water. Other inland water, including Great
  Salt Lake and Lake Tahoe, remains. The second measures the full county
  polygon, including ocean and Great Lakes water. To report the fraction of
  each county in public, conservation, or BBB lands, we divide the category's
  land-only hectares by each of these county areas.
- **Natural area.** We sum forest, grassland, shrubland, water/snow, and wetland
  pixels in five masks derived from the 2023 National Land Cover Database
  (NLCD), then multiply by 0.09 hectares per 30 m pixel. We calculate natural
  area on both county footprints described above. Classified water and snow
  contribute to natural area; NoData pixels do not.
- **Shoreline and inland water.** We intersect each analysis area with 2019
  TIGER/Line shoreline and National Hydrography Dataset (NHD) water polygons,
  applying the lake exclusions described under [fisheries](#fisheries).

### Identify BBB candidate lands

The BBB footprint represents a land-sale scenario based on
[Section 50301 of the June 2025 Senate proposal](https://www.congress.gov/amendment/119th-congress/senate-amendment/2368/text).
Within CONUS, the screen:

1. Selects federally owned fee lands managed by the Bureau of Land Management
   in Arizona, California, Colorado, Idaho, Nevada, New Mexico, Oregon, Utah,
   Washington, and Wyoming.
2. Removes federally protected areas mapped in PAD-US, including parks,
   monuments, recreation and conservation areas, wildlife refuges, wilderness,
   wild and scenic rivers, and national trails. National Park Service and Fish
   and Wildlife Service proclamation boundaries supplement these designations.
3. Keeps land within five statute miles of an incorporated municipality's
   boundary or a Census-designated place's published center point. Both types
   of places must have at least 1,000 residents in the 2020 Census.

We clip and merge candidate land by county. The screen applies the mapped
criteria only; evaluating grazing permits, existing rights, housing
suitability, and tract selection requires a separate review.
See the [BBB input workflow](docs/supporting_workflows.md#prepare-the-bbb-county-footprint)
to recreate the footprint.

### Summarize raster values

The raster sources represent carbon stock, pollination, flood mitigation,
mental-health benefits, surface-water provisioning, and avoided dredging,
drinking-water health, and water-treatment costs. We sum valid pixel values
whose centers fall within each analysis area. For raster sampling, boundary
simplification is capped at 15 m, with smaller tolerances for small or narrow
polygon parts. We sample overlapping manager areas separately. The
[raster inventory](data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml)
identifies the source files and units.

Carbon stock is supplied as tonnes of CO₂e per hectare. We multiply the sum
of those densities by 0.09 hectares per 30 m pixel to obtain total tonnes of
CO₂e, then multiply by $3.66 per tonne and a 1.182 inflation factor to obtain
the stock's monetary value. Pollination and flood-mitigation rasters also
contain per-hectare values, so we multiply their sums by 0.09 hectares per
pixel. We sum rasters already expressed as values per pixel directly.

The supplied mental-health model has gaps and sparse coverage in Connecticut,
Washington, DC, and other locations. A zero sum where the model has no valid
pixels indicates missing coverage, not an absence of mental-health benefits.

### Allocate service values

#### County-based services

We assemble county estimates of
[grazing](data/analysis_inputs/ecosystem_services/grazing/SNAPPGrazingDataset_final_062326.xlsx),
[timber](data/analysis_inputs/ecosystem_services/timber/total_valuation_all_regions_fvs.csv),
[ecosystem-related wages](data/analysis_inputs/ecosystem_services/jobs/Finalized%20Flagged%20and%20Sorted%20Industry%20and%20County%20Codes.xlsx),
[air-quality benefits](data/analysis_inputs/ecosystem_services/air_quality/snapp_air_quality.csv),
[urban-heat mitigation](data/analysis_inputs/ecosystem_services/urban_heat/HEAT_all_tracts_2021.csv),
and [physical-health benefits](data/analysis_inputs/ecosystem_services/physical_health/hypertension_linear_us_counties_all.gpkg).
These source-file links point into the downloaded
[assessment data stack](https://drive.google.com/drive/folders/141tOj6sf8Go0UttVogSc_T3Jet1wXzlu);
the data files are not hosted on GitHub. The tracked
[source list](data/workflow_assets/county_integration/sources.csv) records their paths.

Within each county, we multiply each service estimate by the fraction of
county land in the public, conservation, or BBB category. We apply the same
calculation to each public or conservation manager type. This assumes a
uniform service value per land hectare within the county. Urban-heat estimates
are first summed from Census tracts to counties; the wage estimate sums
cultural, raw-materials, processing, and harvesting wages.

We prepare the source estimates as follows:

- **Connecticut geography.** The Census county geography changed from eight
  counties to nine planning regions, with different boundaries and identifiers.
  Some service sources use the eight counties. We use the
  [Connecticut crosswalk](data/workflow_assets/county_integration/ct_county_crosswalk.csv)
  to distribute each old county's estimate among the planning regions in
  proportion to their shared land area.
- **Air-quality county codes.** The source identifies Shannon County, South
  Dakota, by its former code `46113`. We [replace it with Oglala Lakota County's code
  `46102`](data/workflow_assets/county_integration/air_quality_county_aliases.csv)
  so those records join to the current county before we sum their values.
- **Timber values.** Some source counties have negative estimated timber rents.
  We report these as zero timber value before combining counties or allocating
  their values to land categories.
- **Grazing time basis.** We value the forage available to livestock over one
  year, treating mean forage biomass as annual production. The workbook
  calculates how many animal-unit months (one animal unit fed for one month)
  the usable forage can support, then applies the fee per animal-unit month.
  This already values
  the year's forage, so we omit the workbook's additional multiplication by
  eight. See the
  [grazing calculation](docs/county_valuation_integration_2026.md#grazing-valuation).

#### Recreation

Spencer Wood's [Outdoor Recreation & Data Lab](https://outdoorrd.org/) at the
University of Washington supplied county totals of annual recreation use value
for protected areas, in 2024 dollars. The supplied estimates have no
within-county breakdown by land category, so we report recreation only in the
county output.

#### Fisheries

The fisheries sources provide state totals. We distribute marine fisheries
value in proportion to shoreline length and inland fisheries value in
proportion to NHD water area, assuming a uniform value per unit length or area
within each state. For each county and land category, we multiply the state
total by its share of the state's eligible shoreline or water area, measured
across full counties. We exclude the Great Lakes from both measurements and
Great Salt Lake from Utah's inland water area before calculating these shares.

#### Coastal protection

Coastal-protection estimates are supplied for coral locations and mangrove
study areas. We assign coral point values to the nearest county in the
national reference layer and retain assignments to CONUS counties. A public,
conservation, or BBB polygon in the assigned county receives the value if the
point falls inside or on its boundary. For mangroves, we use study units identified as U.S.
units (`ISO3 = 'USA'`) and allocate each unit's value by the analysis area's
share of that unit's total overlap with CONUS counties.

### Adjust monetary values

The source valuations use different price years. We apply the price-year
multipliers in the [adjustment table](data/workflow_assets/county_integration/adjustment_factors.csv):
1.182 for surface-water provisioning and mangroves; 1.135 for air quality and
urban heat; 1.065 for wages and inland fisheries; 1.027 for marine fisheries
and physical health; and 1.368 for corals. Each factor is applied once. The
table records the source units and service-specific multipliers.

### Calculate totals and county ratios

We sum the adjusted ecosystem-service estimates for each county and land
category using the [configured components](data/workflow_assets/county_integration/derived_metrics.yaml).
Every output includes a total excluding recreation; counties also have a total
including recreation. We include annual flood benefits and total surface-water
provisioning in these sums. We report flood net present value and provisioning
subcategories separately, avoiding their addition to totals that already
include the corresponding service. Missing estimates contribute zero to the
sum. The total combines annual benefits, wages, and stock values that may
overlap economically; it is an assessment summary with mixed time bases.

To compare county ecosystem-service value relative to county size, population,
and economic activity, we divide the total including recreation by county
area, NLCD natural area, 2024 Census population, and 2024 BEA GDP. The
per-hectare comparisons use both the county footprints excluding ocean and
Great Lakes water and the full county footprints.

BEA reports some Virginia counties and independent cities as combined areas.
We split these [combined GDP estimates](data/workflow_assets/county_integration/virginia_gdp_crosswalk.csv)
by each member's share of the group's 2024 Census population and flag the
population-based estimates in the output. Ratios with
missing or nonpositive denominators are blank.

## Output fields and units

Each row describes a county, a land category within a county, or a land
category and manager type within a county. The CSV and GeoPackage contain
the same attributes; the GeoPackage also contains the polygon geometry in
EPSG:6933.

### Identifiers and land measurements

| Field | Meaning | Units and layers |
| --- | --- | --- |
| `GEOID` | County or county-equivalent identifier; preserve leading zeroes | Five-character code; all layers |
| `Mang_Type` | PAD-US manager type; `UNSPECIFIED` denotes a missing manager label | Code; the two manager-type layers |
| `area_ha_land` | Land-only area of the public, conservation, or BBB polygon | Hectares; the five land-category layers |
| `county_land_share` | Polygon's land area divided by Census county land area; used to allocate county service estimates | Fraction from 0 to 1; the five land-category layers |
| `area_ha_county_land` | Census county land area (`ALAND / 10,000`) | Hectares; counties |
| `area_ha_public_lands` | Unduplicated public land area within the county | Hectares; counties |
| `area_ha_conservation_lands` | Unduplicated conservation land area within the county | Hectares; counties |
| `area_ha_bbb_lands` | Unduplicated BBB candidate land area within the county | Hectares; counties |
| `area_ha_public_manager_overlap`, `area_ha_conservation_manager_overlap` | Sum of manager polygon areas minus their union; measures area counted more than once across manager groups, including any water | Hectares; counties |

The county output also retains Census attributes, including `NAME` and
`NAMELSAD` (names), `STATEFP` and `COUNTYFP` (codes), `ALAND` and `AWATER`
(square metres), and `INTPTLAT` and `INTPTLON` (internal-point latitude and
longitude in degrees). These source attributes are separate from the
calculated reporting areas below.

### Ecosystem-service values

The fields below contain adjusted monetary totals for each row's analysis
area, not per-pixel or per-hectare rates. All six layers have these fields
except recreation, which is county-only. `USD/year` identifies an explicitly
annual estimate; `USD` alone does not establish a benefit period. All dollar
values include the configured adjustments.

| Field | Value represented | Units |
| --- | --- | --- |
| `sum_pollination_attributed_annual_crop_yield_value_adj` | Annual crop-yield value attributed to pollination | USD/year |
| `sum_CO2e_mitigation_dollar_value_per_pixel_adj` | Value of the mapped 2020 carbon stock, summed over the analysis area | USD, stock value |
| `sum_avoided_dredging_costs_raster_adj` | Avoided dredging costs | USD |
| `sum_avoided_sdwa_health_costs_raster_adj` | Avoided drinking-water health costs | USD |
| `sum_avoided_treatment_costs_raster_adj` | Avoided water-treatment costs | USD |
| `sum_Wval_sw_2020usd_total_adj` | Total surface-water provisioning value | USD |
| `sum_Wval_sw_2020usd_irr_adj` | Irrigation component of surface-water provisioning | USD |
| `sum_Wval_sw_2020usd_pow_adj` | Power-generation component of surface-water provisioning | USD |
| `sum_Wval_sw_2020usd_pub_adj` | Public-water-supply component of surface-water provisioning | USD |
| `sum_flood_annual_value_masked_to_wetlands_adj` | Annual wetland flood-mitigation value | USD/year |
| `sum_flood_marginal_npv_masked_to_wetlands_adj` | Net present value of wetland flood mitigation | USD, present value |
| `sum_mental_health_national_existing_greenness_cost_90m_adj` | Mental-health benefits of existing greenness | USD |
| `sum_aeb_coral_BC_USD11_adj` | Coastal-protection value from coral points assigned to the area | USD |
| `sum_mangrove_coastal_flood_protection_value_adj` | Coastal flood-protection value allocated from mangrove study units | USD |
| `air_quality_avoided_mortality_annual_value_adj` | Annual value of deaths avoided through air-quality benefits | USD/year |
| `grazing_adj` | Annual forage value | USD/year |
| `timber_adj` | Annual timber rent | USD/year |
| `jobs_cultural_adj`, `jobs_raw_materials_adj`, `jobs_processing_adj`, `jobs_harvesting_adj` | Wages in the four nature-related sectors; not numbers of jobs | USD |
| `jobs_adj` | Total wages across those four sectors | USD |
| `marine_fisheries_adj`, `inland_fisheries_adj` | Shares of the supplied state fisheries values | USD |
| `urban_heat_adj` | Urban-heat mitigation value | USD |
| `physical_health_adj` | Physical-health value from the hypertension estimates | USD |
| `recreation_adj` | Annual recreation use value for protected areas; counties only | USD/year |
| `total_ecosystem_services` | Sum of the 18 selected service components, excluding recreation | USD, mixed benefit periods |
| `total_ecosystem_services_w_rec` | The same total plus recreation; counties only | USD, mixed benefit periods |

The totals include the provisioning total and wage total once, not their
subcategories as additional benefits. Flood NPV is reported separately from
the totals. Missing service estimates contribute zero to totals; a blank
individual estimate still denotes missing data. Raster sums with no valid
pixels can also be zero, as described for mental health above.

### Adjustment factors

Each service has a `*_adj_factor` field recording the applied multiplier.
Most are price-year multipliers, but the area conversions and carbon price
have different units. These are the default values in the
[adjustment table](data/workflow_assets/county_integration/adjustment_factors.csv):

| Factor fields | Value | Meaning and units |
| --- | --- | --- |
| `pollination_adj_factor`, `flood_annual_adj_factor`, `flood_npv_adj_factor` | 0.09 | Hectares per 30 m pixel; converts the sum of per-hectare raster values to a monetary total |
| `carbon_pixel_adj_factor` | 4.32612 | Adjusted USD per tonne CO₂e (`3.66 × 1.182`), applied after the separate 0.09-hectare conversion |
| `provisioning_adj_factor`, `provisioning_irr_adj_factor`, `provisioning_pow_adj_factor`, `provisioning_pub_adj_factor`, `mangroves_adj_factor` | 1.182 | Price-year multiplier, dimensionless |
| `air_quality_avoided_mortality_annual_value_adj_factor`, `urban_heat_adj_factor` | 1.135 | Price-year multiplier, dimensionless |
| `jobs_adj_factor`, `jobs_cultural_adj_factor`, `jobs_raw_materials_adj_factor`, `jobs_processing_adj_factor`, `jobs_harvesting_adj_factor`, `inland_fisheries_adj_factor` | 1.065 | Price-year multiplier, dimensionless |
| `marine_fisheries_adj_factor`, `physical_health_adj_factor` | 1.027 | Price-year multiplier, dimensionless |
| `corals_adj_factor` | 1.368 | Price-year multiplier, dimensionless |
| `dredging_adj_factor`, `sdwa_adj_factor`, `water_treatment_adj_factor`, `grazing_adj_factor`, `timber_adj_factor`, `mental_health_adj_factor`, `recreation_adj_factor` | 1 | No additional monetary scaling; dimensionless. Recreation's factor is county-only. |

### County areas and comparisons

These fields appear only in the county output. The paired reporting-area
fields without an inclusive suffix exclude ocean and Great Lakes water but
retain other inland water. Their `_incl_ocean_great_lakes` companions use the
full county polygon. Neither version changes the land-only areas used to
allocate county service estimates.

| Field | Meaning | Units |
| --- | --- | --- |
| `area_ha_county` | County area excluding ocean and Great Lakes water | Hectares |
| `area_ha_county_incl_ocean_great_lakes` | Full county polygon area | Hectares |
| `area_ha_valid_nlcd` | Area with valid NLCD data after excluding ocean and Great Lakes water, including natural and other land-cover classes | Hectares |
| `area_ha_nlcd_forests`, `area_ha_nlcd_grasslands`, `area_ha_nlcd_shrubland`, `area_ha_nlcd_water_snow`, `area_ha_nlcd_wetlands` | Area of each NLCD class after excluding ocean and Great Lakes water | Hectares |
| `area_ha_all_natural` | Sum of those five class areas | Hectares |
| `area_ha_all_natural_incl_ocean_great_lakes` | The same five classes measured over the full county; NoData is not counted | Hectares |
| `proportion_nlcd_<class>_of_valid` | Class area divided by valid NLCD coverage; `<class>` is `forests`, `grasslands`, `shrubland`, `water_snow`, or `wetlands` | Fraction from 0 to 1; blank without valid coverage |
| `ratio_public_lands_to_county_area`, `ratio_conservation_lands_to_county_area`, `ratio_bbb_lands_to_county_area` | Category's land-only hectares divided by `area_ha_county` | Fraction, not percent |
| Each of the three ratio names above with `_incl_ocean_great_lakes` appended | Same numerator divided by full county area | Fraction, not percent |
| `GDP_2024` | 2024 county GDP, including population-based allocations where needed | USD; source thousands converted to dollars |
| `GDP_2024_is_allocated` | Whether GDP was allocated from a combined Virginia area | 1 = allocated; 0 = otherwise |
| `Pop_2024` | Estimated county population in 2024 | People |
| `es_per_ha`, `es_per_ha_incl_ocean_great_lakes` | County total including recreation divided by the corresponding county area | USD/hectare |
| `es_per_ha_natural`, `es_per_ha_natural_incl_ocean_great_lakes` | County total including recreation divided by the corresponding natural area | USD/hectare |
| `es_per_capita` | County total including recreation divided by population | USD/person |
| `es_per_gdp` | County total including recreation divided by GDP | USD/USD ratio, not percent |

Only the combined natural-area total has an inclusive companion; the five
class areas and their proportions use the footprint excluding ocean and
Great Lakes water. Ratios are blank when their denominators are missing or
nonpositive. Because the ecosystem-service total mixes benefit periods, its
per-hectare and per-person values are not annual rates, and its GDP ratio is
not a measure of annual contribution to GDP.

## Run the assessment

### 1. Set up Python

Use a shared Python environment for this repository and
[`zonal_stats_toolkit`](https://github.com/springinnovate/zonal_stats_toolkit).
The commands below run from the assessment repository root and assume a
toolkit checkout beside this repository. Adjust that path if needed. In a
CMD shell with Conda available, create and activate the environment:

```cmd
conda env create -f "..\zonal_stats_toolkit\environment-runner.yml"
conda activate zonal-stats-toolkit
conda install -c conda-forge pyogrio openpyxl
```

### 2. Place and configure the inputs

Copy the [assessment data stack](https://drive.google.com/drive/folders/141tOj6sf8Go0UttVogSc_T3Jet1wXzlu)
into `data/` at the repository root, preserving the directory structure below.
Use the tracked workflow configuration files from this checkout.

```text
data/
├── analysis_inputs/
│   ├── padus/                                  PAD-US 4.1 VectorAnalysis geodatabase
│   ├── zonal_units/
│   │   ├── counties/                           2024 TIGER/Line county GeoPackage
│   │   └── bbb_candidate_blm_lands_by_county/    BBB candidate county GeoPackage
│   ├── boundaries/county_land/                 Census county land-mask GeoPackage
│   ├── boundaries/usa_boundary/                U.S. boundary used for reporting-area ocean exclusion
│   ├── boundaries/county_reporting_areas/      County area and NLCD metrics excluding ocean/Great Lakes
│   ├── ecosystem_services/                     Service TIFFs, tables, coral and mangrove layers
│   ├── masks/                                  Five 2023 NLCD class masks
│   ├── hydrography/nhdfreshwater/                NHD water polygons for inland fisheries and Great Lakes exclusion
│   ├── linear_features/coastline/               TIGER shoreline for marine fisheries and ocean identification
│   ├── recreation/                             County recreation table
│   ├── CAGDP1_LineCode3_All_States.csv            County GDP
│   └── co-est2025-alldata.csv                    County population estimates
├── analysis_results/nlcd_mask_areas/           Prepared full-county NLCD GeoPackage
└── workflow_assets/                            Tracked input selections and valuation rules
```

The configuration files and command-line options identify the exact inputs:

| To change | Where to set it |
| --- | --- |
| PAD-US | `--padus-path` and `--padus-layer`. The default is `PADUS4_1VectorAnalysis_PADUS_Only.gdb` in `data/analysis_inputs/padus/PADUS4_1VectorAnalysis_PADUS_Only (1)/`, layer `PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP`. |
| County boundaries | `--counties` and `--counties-layer`. Counties require five-digit `GEOID` keys and `ALAND` in square metres. Keep the default national county file at `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` for coral assignments. |
| Land mask and BBB footprint | `--land-mask` and `--bbb-by-county`, with `--land-mask-layer` and `--bbb-layer` as needed. Match the county release and `GEOID` keys. The mask must contain one footprint for every CONUS county; the BBB file contains one footprint per county with candidate land. |
| Service TIFFs | The [raster inventory](data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml) names the required source TIFFs and their units. Place compatible replacements in `data/analysis_inputs/ecosystem_services/` with the expected names. |
| Valuation tables and spatial sources | [`sources.csv`](data/workflow_assets/county_integration/sources.csv) gives paths relative to `data/`. Keep the expected fields, identifiers, and units. When replacing NHD, verify the lake exclusion IDs against that release. |
| County ratio inputs | [`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml) selects both county and NLCD area denominators, GDP, and population. Keep the prepared reporting-area CSV and GeoPackage, their QA CSV, and the full-county NLCD GeoPackage. |
| Factors, fields, or public-land rules | [`adjustment_factors.csv`](data/workflow_assets/county_integration/adjustment_factors.csv), [`export_rules.yaml`](data/workflow_assets/county_integration/export_rules.yaml), and [`padus_land_rules.rules`](config/padus_land_rules.rules). |

The [supporting workflows](docs/supporting_workflows.md) describe how to
prepare the county land mask, BBB footprint, NLCD masks, and other spatial
inputs when replacing the supplied data.

If the reporting-area files are missing, generate them from local vectors
and the five NLCD masks:

```cmd
set "PROJ_NETWORK=OFF"
python prepare_county_reporting_areas.py --runner "..\zonal_stats_toolkit\runner.py"
```

Keep the CSV, GeoPackage, QA CSV, and manifest together under
`data/analysis_inputs/boundaries/county_reporting_areas/`. Keep the separate
full-county NLCD GeoPackage selected in `derived_metrics.yaml`.

### 3. Validate and launch

In CMD from the repository root, set the paths below to your input files.
Offline PROJ settings keep county reprojection consistent with the supplied
land mask. Choose a new `--run-id` for each run, or omit it to use a UTC
timestamp.

```cmd
set "PROJ_NETWORK=OFF"
set "SNAPP_RUNNER=..\zonal_stats_toolkit\runner.py"
set "SNAPP_BBB=data\analysis_inputs\zonal_units\bbb_candidate_blm_lands_by_county\bbb_candidate_blm_lands_within_5_miles_of_population_centers_by_county_2026_08_09_18_45_22.gpkg"
set "SNAPP_MASK=data\analysis_inputs\boundaries\county_land\tl_2024_us_county_land_conus.gpkg"

python run_conus_ecosystem_services.py --runner "%SNAPP_RUNNER%" --bbb-by-county "%SNAPP_BBB%" --land-mask "%SNAPP_MASK%" --validate-only
python run_conus_ecosystem_services.py --runner "%SNAPP_RUNNER%" --bbb-by-county "%SNAPP_BBB%" --land-mask "%SNAPP_MASK%" --run-id conus-assessment
```

Run the second command after validation passes. Use `--help` for worker
settings and additional input options.

### 4. Find the output files

```text
data/analysis_results/assessment_runs/<run-id>/
├── run_manifest.json
├── counties/
├── public_lands_by_county/
├── public_lands_by_county_and_manager_type/
├── conservation_lands_by_county/
├── conservation_lands_by_county_and_manager_type/
└── bbb_lands_by_county/
```

Each product folder contains `<product>_<run-id>.csv` and
`<product>_<run-id>.gpkg`. The GeoPackage stores the analysis geometry and
attributes; the CSV contains the same attributes. Check `run_manifest.json`
for `"status": "complete"` and the output paths. Keep the manifest with the
outputs as a record of the inputs and settings.

### Revalue a completed run

To update valuation factors, tables, or county ratios, reuse a completed
run's saved analysis boundaries and raster sums. Pass its run ID and a new
output ID:

```cmd
python run_conus_ecosystem_services.py --reuse-zonals-from "conus-assessment" --run-id "conus-revalued"
```

Add `--validate-only` to check reuse before launching. Changes to analysis
boundaries or service rasters require a fresh assessment run.
