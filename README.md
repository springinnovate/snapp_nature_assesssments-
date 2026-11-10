# CONUS Ecosystem Services Assessment

This SNAPP assessment estimates ecosystem-service values for counties and
selected land footprints in the contiguous 48 states and Washington, DC. It
publishes six output layers, each as a matching CSV and GeoPackage pair:

1. Counties
2. Public lands by county
3. Public lands by county and manager type
4. Conservation lands by county
5. Conservation lands by county and manager type
6. Big Beautiful Bill (BBB) candidate lands by county

## Methods

### Zones, area, and shoreline

The assessment clips PAD-US 4.1 VectorAnalysis polygons to 2024 TIGER/Line
county boundaries. [Public lands](config/padus_land_rules.rules) include federal,
state, local, district, joint, and territorial managers. A designation list
determines inclusion when the manager is explicitly unknown. Conservation lands
are selected independently by GAP status (1–3). Within each county, polygons
are merged into a single footprint for each land category and, separately,
for each manager type. Manager footprints can overlap; use the county-level
footprint for a category total.

Vector areas and lengths are measured in EPSG:6933, an equal-area projection.

- **Land area** is the area of a footprint intersected with the 2024 Census
  land mask, which comes from TIGER/Line Faces classified as land. A
  footprint's land share is its land area divided by Census county land area
  (`ALAND`). Raster and fisheries measurements use the full zone geometry.
- **County area** is measured from the county polygon. `area_ha_county`
  excludes the [NHD Great Lakes polygons](data/workflow_assets/county_integration/great_lakes_nhd_features.csv)
  and water outside the U.S. boundary connected to the TIGER marine shoreline.
  The Census land mask protects land from removal. The generalized coastal
  boundary can leave some tidal water. Other inland water, including Great Salt
  Lake and Lake Tahoe, stays in the area. `area_ha_county_incl_ocean_great_lakes`
  measures the full county polygon. Both versions of the public, conservation,
  and BBB land ratios divide land-only hectares by the corresponding county
  area.
- **Natural area** is the sum of forest, grassland, shrubland, water/snow, and
  wetland area in the 2023 National Land Cover Database (NLCD). Count the pixels
  in each class and multiply by 0.09 hectares per 30 m pixel. The two natural
  area fields use the corresponding county footprints. Classified water/snow
  counts as natural area; NoData does not.
- **Shoreline and inland water** are measured by intersecting each zone with
  2019 TIGER/Line shoreline and National Hydrography Dataset (NHD) water
  polygons. These lengths and areas are the fisheries weights.

### BBB candidate lands

The BBB footprint represents a land-sale scenario based on
[Section 50301 of the June 2025 Senate proposal](https://www.congress.gov/amendment/119th-congress/senate-amendment/2368/text).
The screen:

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

Candidate land is clipped and merged by county. The screen does not establish
sale eligibility, account for grazing permits or existing rights, or assess
suitability for housing. See the [BBB input workflow](docs/supporting_workflows.md#prepare-the-bbb-county-footprint)
to recreate the footprint.

### Raster summaries

For each zone, sum valid raster cells whose centers fall inside the zone.
Boundary simplification is capped at 15 m, with smaller tolerances for small
or narrow polygon parts. Overlapping manager footprints are measured separately.
The [raster inventory](data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml)
lists the sources and units.

- **Carbon stock:** Multiply the per-hectare sum by 0.09 hectares per pixel to
  get tonnes of CO₂e, then apply $3.66 per tonne and a 1.182 inflation factor.
- **Pollination and flood mitigation:** Multiply per-hectare values by
  0.09 hectares per pixel.
- **Mental health:** The supplied model raster has sparse coverage in
  Connecticut and Washington, DC, including four Connecticut planning regions
  with no valid pixels. A zero zonal sum can reflect missing model coverage
  rather than an estimate of zero.

### Service allocations

**County estimates.** Grazing, timber, jobs, air quality, urban heat, and
physical health values are matched by county and allocated to footprints by
land share. The [Connecticut crosswalk](data/workflow_assets/county_integration/ct_county_crosswalk.csv)
splits values for historical counties among current planning regions by land
area. Air quality uses [county-code corrections](data/workflow_assets/county_integration/air_quality_county_aliases.csv)
before aggregation. Negative timber rents are set to zero. The
[grazing estimate](docs/county_valuation_integration_2026.md#grazing-valuation)
treats mean forage biomass as annual production and values the resulting
animal-unit months.

**Recreation.** Spencer Wood's [Outdoor Recreation & Data Lab](https://outdoorrd.org/)
at the University of Washington supplied estimates by county. Recreation
appears only in the county output.

**Fisheries.** State marine values are allocated by shoreline length; state
inland values are allocated by NHD water area. Each zone's length or area is
divided by the eligible statewide measure across full counties. The Great
Lakes are excluded from both weights, and Great Salt Lake is excluded from
Utah's inland weights. These allocations do not measure local catch.

**Coastal protection.** Coral point values go to the nearest county in the
national reference layer, with a value reported only for CONUS assignments.
A footprint receives the value when it covers the assigned point. Mangrove
study-unit values are allocated by each zone's share of the unit's total
overlap with CONUS counties.

### Totals and county ratios

`total_ecosystem_services` sums [adjusted service values](data/workflow_assets/county_integration/derived_metrics.yaml)
without recreation. The total includes annual flood benefits and total
surface-water provisioning; flood net present value is reported separately.
Missing estimates contribute zero to the sum. The components combine annual
benefits, wages, and stock values and may overlap economically.

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

Use one Python environment for this repository and
[`zonal_stats_toolkit`](https://github.com/springinnovate/zonal_stats_toolkit).
From the assessment repository root, the commands below assume the toolkit is
cloned next to this repository. Adjust the toolkit path to match your checkout.
In CMD, create the environment if needed:

```cmd
conda env create -f "..\zonal_stats_toolkit\environment-runner.yml"
conda activate zonal-stats-toolkit
conda install -c conda-forge pyogrio openpyxl
```

### 2. Place and configure the inputs

Copy the [assessment data stack](https://drive.google.com/drive/folders/141tOj6sf8Go0UttVogSc_T3Jet1wXzlu)
into `data/` at the repository root, keeping its paths. Use the workflow
configuration files from this checkout.

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

Use these files to locate the exact inputs or select replacements:

| To change | Where to set it |
| --- | --- |
| PAD-US | `--padus-path` and `--padus-layer`. The default is `PADUS4_1VectorAnalysis_PADUS_Only.gdb` in `data/analysis_inputs/padus/PADUS4_1VectorAnalysis_PADUS_Only (1)/`, layer `PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP`. |
| County boundaries, land mask, or BBB footprint | `--counties`, `--land-mask`, and `--bbb-by-county`. Match the county release and five-digit `GEOID` keys across inputs. Counties require `ALAND` in square metres; mask and BBB GeoPackages have one footprint per `GEOID` they cover. Select layers with `--counties-layer`, `--land-mask-layer`, or `--bbb-layer` as needed. Keep the default national county file at `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` for coral assignments. |
| Service TIFFs | The [raster inventory](data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml) names the required source TIFFs and their units. Place compatible replacements in `data/analysis_inputs/ecosystem_services/` with the expected names. |
| Valuation tables and spatial sources | [`sources.csv`](data/workflow_assets/county_integration/sources.csv) gives paths relative to `data/`. Keep the expected fields, identifiers, and units. When replacing NHD, verify the lake exclusion IDs against that release. |
| County ratio inputs | [`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml) selects both county and NLCD area denominators, GDP, and population. Keep the prepared reporting-area CSV and GeoPackage, their QA CSV, and the full-county NLCD GeoPackage. |
| Factors, fields, or public-land rules | [`adjustment_factors.csv`](data/workflow_assets/county_integration/adjustment_factors.csv), [`export_rules.yaml`](data/workflow_assets/county_integration/export_rules.yaml), and [`padus_land_rules.rules`](config/padus_land_rules.rules). |

See [supporting workflows](docs/supporting_workflows.md) to prepare or replace
the county land mask, BBB footprint, or other prepared data sources.

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

In CMD from the repository root, set the paths below to your actual files. Use
offline PROJ settings so the county reprojection matches the supplied land
mask. Choose a new `--run-id` for each run; omit it to use a UTC timestamp.

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

### 4. Find the final files

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
`<product>_<run-id>.gpkg`. The GeoPackage includes the mapped geometry; the CSV
contains the same attributes. Check for `"status": "complete"` and the file
paths in `run_manifest.json`.

### Revalue a completed run

To update valuations or ratios, reuse the saved zones and raster sums. Pass
the original zonal run ID and a new output ID:

```cmd
python run_conus_ecosystem_services.py --reuse-zonals-from "conus-assessment" --run-id "conus-revalued"
```

Add `--validate-only` to check reuse before launching. Use a fresh assessment
when changing zones or service rasters. Keep `run_manifest.json` with the
outputs to record the inputs and settings for each run.
