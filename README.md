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
2. [Measure area and shoreline](#measure-area-and-shoreline) to establish
   allocation weights and comparison areas, and [sum raster estimates](#summarize-raster-values)
   of ecosystem services within each analysis area.
3. [Allocate service estimates](#allocate-service-values) supplied by county,
   state, or study area to the analysis areas, then [convert units and adjust
   monetary values](#convert-units-and-adjust-monetary-values).
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

We retain the full county-clipped polygons, including any water, for raster
summaries and spatial service allocations. We measure land-only hectares
separately to allocate county service estimates. County area and natural area
provide denominators for per-hectare comparisons; shoreline length and inland
water area provide fisheries allocation weights. We measure vector areas and
lengths in the equal-area projection EPSG:6933.

- **Land area.** We intersect each selected-land polygon with the 2024 Census
  land mask, constructed from TIGER/Line Faces classified as land. The
  intersection area divided by Census county land area (`ALAND`) gives the
  selected land's share of its county. County land hectares come directly
  from `ALAND`, converted from square metres.
- **County area.** We report two denominators. `area_ha_county` excludes the
  [NHD Great Lakes polygons](data/workflow_assets/county_integration/great_lakes_nhd_features.csv)
  and water outside the U.S. boundary connected to the TIGER marine shoreline.
  The Census land mask protects land from removal; the generalized coastal
  boundary can retain some tidal water. Other inland water, including Great
  Salt Lake and Lake Tahoe, remains. `area_ha_county_incl_ocean_great_lakes`
  measures the full county polygon. Public, conservation, and BBB land ratios
  use land-only hectares as the numerator with each of these denominators.
- **Natural area.** We sum forest, grassland, shrubland, water/snow, and wetland
  pixels in five masks derived from the 2023 National Land Cover Database
  (NLCD), then multiply by 0.09 hectares per 30 m pixel. We calculate natural
  area on both county footprints described above. Classified water and snow
  contribute to natural area; NoData pixels do not.
- **Shoreline and inland water.** We intersect each analysis area with 2019
  TIGER/Line shoreline and National Hydrography Dataset (NHD) water polygons,
  applying the lake exclusions described under [fisheries](#allocate-service-values).

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

The supplied mental-health model has gaps and sparse coverage in Connecticut,
Washington, DC, and other locations. A zero sum where the model has no valid
pixels indicates missing coverage, not an absence of mental-health benefits.

### Allocate service values

**County-based services.** We assemble county estimates of grazing, timber,
ecosystem-related wages (`jobs`), air-quality benefits, urban-heat mitigation,
and physical-health benefits. Each selected land receives its county's
estimate multiplied by its land share. This allocation assumes a uniform
value per land hectare within the county.

The [Connecticut crosswalk](data/workflow_assets/county_integration/ct_county_crosswalk.csv)
allocates estimates for historical counties to current planning regions by
land area. We apply [county-code corrections](data/workflow_assets/county_integration/air_quality_county_aliases.csv)
to air-quality records before aggregation and set negative timber rents to
zero. The [grazing calculation](docs/county_valuation_integration_2026.md#grazing-valuation)
treats mean forage biomass as annual production and values the resulting
animal-unit months.

**Recreation.** Spencer Wood's [Outdoor Recreation & Data Lab](https://outdoorrd.org/)
at the University of Washington supplied county estimates of annual recreation
use value for protected areas, in 2024 dollars. Recreation appears only in
the county output.

**Fisheries.** We allocate state marine fisheries values by shoreline length
and state inland fisheries values by NHD water area. Each analysis area
receives the state value multiplied by its share of the eligible shoreline
length or water area across full counties in that state. We exclude the Great
Lakes from both weights and Great Salt Lake from Utah's inland weights. These
are spatial allocations of state estimates, not measurements of local catch.

**Coastal protection.** We assign coral point values to the nearest county in
the national reference layer and retain assignments to CONUS counties. A
selected area in the assigned county receives the value if the point falls
inside or on its boundary. For mangroves, we use study units identified as U.S.
units (`ISO3 = 'USA'`) and allocate each unit's value by the analysis area's
share of that unit's total overlap with CONUS counties.

### Convert units and adjust monetary values

We integrate per-hectare raster values over pixel area. For carbon, we
multiply the sum of per-hectare stock values by
0.09 hectares per 30 m pixel to obtain tonnes of CO₂e, then value the stock at
$3.66 per tonne with a 1.182 inflation multiplier. Pollination and flood
mitigation also require the 0.09-hectare conversion. Rasters already expressed
as values per pixel are summed directly.

The [adjustment table](data/workflow_assets/county_integration/adjustment_factors.csv)
records the source units and configured factors. Price-year multipliers are
1.182 for surface-water provisioning and mangroves; 1.135 for air quality and
urban heat; 1.065 for wages and inland fisheries; 1.027 for marine fisheries
and physical health; and 1.368 for corals. We apply each factor once and
export the adjusted estimates in `*_adj` fields with their `*_adj_factor`
fields. Carbon's factor is 4.32612 (3.66 × 1.182), applied after the separate
0.09-hectare conversion.

### Calculate totals and county ratios

`total_ecosystem_services` sums the [adjusted service components](data/workflow_assets/county_integration/derived_metrics.yaml)
excluding recreation. We include annual flood benefits and total
surface-water provisioning, with flood net present value and the provisioning
subcategories reported separately. Missing estimates contribute zero to the
sum. The total combines annual benefits, wages, and stock values that may
overlap economically; it is an assessment summary with mixed time bases.

The county field `total_ecosystem_services_w_rec` adds recreation and is the
numerator for county ecosystem-service ratios. `es_per_ha` uses county area and
`es_per_ha_natural` uses NLCD natural area. Both have an
`_incl_ocean_great_lakes` counterpart. `es_per_capita` uses 2024 Census
population; `es_per_gdp` uses 2024 BEA GDP.
[Combined Virginia county and independent city GDP estimates](data/workflow_assets/county_integration/virginia_gdp_crosswalk.csv)
are split by each member's share of the group's 2024 Census population.
`GDP_2024_is_allocated` identifies those population-based estimates. Ratios with
missing or nonpositive denominators are blank.

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
