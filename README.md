# CONUS Ecosystem Services Assessment

This SNAPP assessment estimates ecosystem-service values for counties and
selected land footprints in the contiguous 48 states and Washington, DC. It
publishes six output layers, each as a matching CSV and GeoPackage pair:

1. Counties
2. Public lands by county
3. Public lands by county and manager type
4. Conservation lands by county
5. Conservation lands by county and manager type
6. BBB candidate lands by county

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
│   ├── ecosystem_services/                     Service TIFFs, tables, coral and mangrove layers
│   ├── hydrography/nhdfreshwater/                NHD water polygons for inland fisheries
│   ├── linear_features/coastline/               TIGER shoreline for marine fisheries
│   ├── recreation/                             County recreation table
│   ├── CAGDP1_LineCode3_All_States.csv            County GDP
│   └── co-est2025-alldata.csv                    County population estimates
└── workflow_assets/                            Tracked input selections and valuation rules
```

Use these files to locate the exact inputs or select replacements:

| To change | Where to set it |
| --- | --- |
| PAD-US | `--padus-path` and `--padus-layer`. The default is `PADUS4_1VectorAnalysis_PADUS_Only.gdb` in `data/analysis_inputs/padus/PADUS4_1VectorAnalysis_PADUS_Only (1)/`, layer `PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP`. |
| County boundaries, land mask, or BBB footprint | `--counties`, `--land-mask`, and `--bbb-by-county`. Match the county release and five-digit `GEOID` keys across inputs. Counties require `ALAND` in square metres; mask and BBB GeoPackages have one footprint per `GEOID` they cover. Select layers with `--counties-layer`, `--land-mask-layer`, or `--bbb-layer` as needed. Keep the default national county file at `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` for coral assignments. |
| Service TIFFs | The [raster inventory](data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml) names the required source TIFFs and their units. Place compatible replacements in `data/analysis_inputs/ecosystem_services/` with the expected names. |
| Valuation tables and spatial sources | [`sources.csv`](data/workflow_assets/county_integration/sources.csv) gives paths relative to `data/`. Keep the expected fields, identifiers, and units. When replacing NHD, verify the lake exclusion IDs against that release. |
| County ratio inputs | [`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml) selects the county area, NLCD natural area, GDP, and population files. These four inputs are required for both fresh runs and revaluation. |
| Factors, fields, or public-land rules | [`adjustment_factors.csv`](data/workflow_assets/county_integration/adjustment_factors.csv), [`export_rules.yaml`](data/workflow_assets/county_integration/export_rules.yaml), and [`padus_land_rules.rules`](config/padus_land_rules.rules). |

See [supporting workflows](docs/supporting_workflows.md) to prepare or replace
the county land mask, BBB footprint, or other prepared data sources.

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

The first command checks the input files, TIFF inventory, and vector metadata.
Run the second after validation passes. It prepares the output geometries,
calculates raster sums, and publishes the valuations. Use `--help` for worker settings and
additional input options.

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

For changes to valuation tables, allocations, or adjustment factors, reuse
the saved zones and raster sums. Use the original run ID in the same checkout
with the original projection settings and the current valuation inputs:

```cmd
python run_conus_ecosystem_services.py --reuse-zonals-from "conus-assessment" --run-id "conus-revalued"
```

The command checks the saved results and publishes six new pairs, leaving the
source run intact. Add `--validate-only` to check reuse before launching.
Use a fresh assessment when changing the zones or source rasters.

This also updates GDP allocations and ratios without rerunning zonals. Use the
original run that calculated the raster sums, not a valuation-only derivative.

## Methods for reporting

**Zones and land shares.** The workflow clips PAD-US 4.1 VectorAnalysis
polygons to CONUS counties. The public rule selects federal, state, local,
district, joint, and territorial managers; it uses a designation allowlist
for explicit unknown managers. Conservation independently selects GAP status
1–3. County/manager rows are keyed by `GEOID` and `Mang_Type`; county-level
footprints are unions within each county. The BBB product uses the supplied
county-clipped candidate BLM footprint, a spatial screen for mapped criteria.
Footprints and manager groups can overlap; use the county-level union for
product totals rather than adding manager rows.

Land hectares are the area of the intersection with the 2024 Census land
mask in EPSG:6933. The mask dissolves TIGER/Line Faces classified as land for
Census area measurement, and its area is checked against Census `ALAND`.
The land share is footprint land area divided by county `ALAND`; the county
export reports hectares (`area_ha_*_lands`) and percentages (`pct_*_lands`)
for public, conservation, and BBB lands. Raster and spatial valuations use
the full zone geometries, including water.

**Raster sums.** The toolkit reprojects zones to each raster's native CRS and
sums valid first-band pixels whose centers fall inside the zone. Geometry
simplification is capped at 15 m, with tighter limits for small or narrow parts.
Overlapping manager types are sampled separately. A raster sum of zero can
represent either zero-valued pixels or no valid pixels. Unit conversions are
applied once during valuation: pollination and flood values per hectare use
0.09 ha per 30 m pixel. Carbon stock is calculated from the per-hectare raster
sum multiplied by 0.09, then valued at $3.66 per tonne CO₂e multiplied by a 1.182
inflation adjustment.

**County tables.** Grazing, timber, jobs, air quality, urban heat, and physical
health are joined by county. A [land-area crosswalk](data/workflow_assets/county_integration/ct_county_crosswalk.csv)
maps Connecticut source IDs to the target county release.
Grazing treats mean biomass as annual production and prices the resulting
animal-unit months once, using the workbook subtotal before its eight-month
multiplier. See the [grazing method and evidence](docs/county_valuation_integration_2026.md#grazing-valuation).
Negative timber rents are set to zero before allocation. Footprints receive
these county values in proportion to their land shares. Recreation is joined
directly by county from `county_cs_2024usd` and reported for counties.

**Fisheries.** Each supplied state total is allocated by eligible shoreline
length for marine fisheries or eligible NHD water area for inland fisheries.
A footprint uses its own intersected length or area divided by the same
statewide denominator measured across full counties. The five Laurentian
Great Lakes are excluded from both weights, and Great Salt Lake is also
excluded from Utah's inland weights. The exclusions change the distribution
of values, while preserving the supplied state totals. Values are
area- or length-based allocations, not observations of local catch.

**Coastal protection.** Coral `BC_USD11` point values are assigned to the
nearest county in the national reference, then retained for CONUS
assignments. A footprint receives an assigned point's value if it covers the
point. Each U.S. mangrove study unit's `Ben_Stock_2020` is distributed by
intersected area using the unit's total overlap with CONUS counties as the
denominator for both counties and footprints.

**Totals and ratios.** Service columns end in `_adj` and their multipliers
are reported as `_adj_factor`. `total_ecosystem_services` sums the
[configured components](data/workflow_assets/county_integration/derived_metrics.yaml)
without recreation. The total uses annual flood benefits, total surface-water
provisioning, and per-pixel carbon value; flood net present value is reported
separately. County `total_ecosystem_services_w_rec` adds recreation and is the
numerator for `es_per_ha`, `es_per_ha_natural`, `es_per_gdp`, and `es_per_capita`.
The denominators are full county area, the sum of five NLCD class areas
(forest, grassland, shrubland, water/snow, and wetland), 2024 BEA GDP in dollars,
and 2024 Census population estimates. Missing service estimates
remain blank in service columns and contribute zero to totals. Ratios with
missing or nonpositive denominators are blank. The total is an assessment
summary of components with different time bases and possible economic overlap.

**GDP geography.** County GDP matches BEA geographic codes directly except for
the Virginia combined county/city areas listed in the
[GDP crosswalk](data/workflow_assets/county_integration/virginia_gdp_crosswalk.csv).
Each combined-area total is split using the members' shares of 2024 Census
population. The full group's population is used even for a subset of counties,
and allocated GDP sums back to the original total. These estimates have
`GDP_2024_is_allocated = 1`; other rows have `0`. The GeoPackage's
`derived_metric_gdp_allocations` and `derived_metric_gdp_group_checks` tables
record the source amounts, weights, and reconciliation. Population allocation
does not constitute an official BEA estimate for each member jurisdiction.

For reproducibility, report the input releases, code and toolkit revisions,
and the run ID. Keep `run_manifest.json` with the final pairs; it records the
configured input paths, configuration hashes, and valuation coverage checks.
