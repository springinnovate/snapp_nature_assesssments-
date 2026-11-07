# SNAPP Nature Assessments

This repository supports a SNAPP assessment of ecosystem services over counties,
all PAD-US lands, and public PAD-US lands in the United States. The workflow
prepares geospatial inputs, runs zonal statistics with
[`zonal_stats_toolkit`](https://github.com/springinnovate/zonal_stats_toolkit),
and builds county ecosystem-service valuations. Both workflows write CSV and
GeoPackage outputs.

The input data stack is stored in this Google Drive folder:
<https://drive.google.com/drive/folders/141tOj6sf8Go0UttVogSc_T3Jet1wXzlu>

For a local run, copy that data stack into the repository root as `data/`.
Large data files are ignored by git; only small workflow assets such as
configuration files and reclassification tables are tracked.

## Environment

The preparation scripts use the conda-compatible environment defined in
`environment.yml`. Create it from the repository root with:

```powershell
conda env create -f environment.yml
conda activate geo
```

The environment file includes the major geospatial packages used here, including
GDAL, GeoPandas, Rasterio, Shapely 2, NumPy, Pandas, PyYAML, and tqdm.

The zonal statistics step is run with `zonal_stats_toolkit`. See the toolkit
repository for its current installation instructions and supported execution
environment:
<https://github.com/springinnovate/zonal_stats_toolkit>

## Run the assessment

Run from the repository root in a Python environment that supports both this
repository's geospatial scripts and the zonal-statistics toolkit. The wrapper
prepares and cuts the PAD-US VectorAnalysis layer, runs TIFF zonals for all
lands and public lands, and builds county valuations from the existing county
zonals and NLCD areas. Check the county input paths in
[`sources.csv`](data/workflow_assets/county_integration/sources.csv) and
[`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml).

Pass the toolkit runner path and the GeoPackage from the BBB run to reuse.
The matching CSV must be beside the GeoPackage with the same filename stem,
ending in the source run's `_YYYYMMDD_HHMMSS` timestamp:

```powershell
python run_snapp_assessment.py `
  --runner 'D:\repositories\zonal_stats_toolkit\runner.py' `
  --bbb-results 'data/analysis_results/zonal_statistics/bbb_candidate_blm_lands/bbb_candidate_ecosystem_services_20260901_214726.gpkg'
```

Add `--validate-only` to check inputs without running the assessment. Use
`--run-id` to name the run, `--padus-path` and `--padus-layer` to select a
different all/public source, and `--job-workers` and `--raster-workers` to
control toolkit parallelism. Use `--workers` and `--geometry-workers` for the
county builder.

Find the matched CSV and GeoPackage pairs under
`data/analysis_results/assessment_runs/<run-id>/`. Each filename starts with
its product directory name:

- `counties/counties_<run-id>.{csv,gpkg}`
- `padus_all_lands/padus_all_lands_<zonal-timestamp>.{csv,gpkg}`
- `padus_public_lands/padus_public_lands_<zonal-timestamp>.{csv,gpkg}`
- `bbb_candidate_blm_lands/bbb_candidate_blm_lands_<source-timestamp>.{csv,gpkg}`

Keep `run_manifest.json` in that run directory with the results. Intermediates
go under `data/processing_outputs/assessment_runs/<run-id>/`; the frozen zonal
config is `data/workflow_assets/zonal_stats/runs/snapp_assessment_<run-id>.yaml`.

The wrapper reuses the BBB pair without rerunning its zonals. It does not
build a public-access product or add county recreation values to footprint
outputs. Use a fresh run ID: the wrapper does not overwrite existing data and
stops if a step fails.

## Data Layout

The `data/` directory is organized by workflow role.

| Directory | Contents |
| --- | --- |
| `data/analysis_inputs` | Inputs used directly by preprocessing or analysis jobs. This includes source rasters, source vectors, prepared masks, freshwater polygons, and county-level zonal units. |
| `data/processing_outputs` | Intermediate products that document how analysis inputs were derived. |
| `data/workflow_assets` | Small tracked configuration files, reclassification tables, and runner configuration. |
| `data/analysis_results/zonal_statistics` | Individual `zonal_stats_toolkit` outputs, grouped into one subdirectory per final zonal dataset. |
| `data/analysis_results/combined` | County valuation CSV and GeoPackage outputs. |

The main input paths are:

| Input | Path |
| --- | --- |
| USA boundary | `data/analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg` |
| Counties | `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` |
| PAD-US geodatabase | `data/analysis_inputs/padus/PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/PADUS4_1Geodatabase.gdb` |
| Ecosystem service rasters | `data/analysis_inputs/ecosystem_services` (selected in `data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml`) |
| NLCD 2023 land cover raster | `data/analysis_inputs/nlcd/Annual_NLCD_LndCov_2023_CU_C1V0.tif` |
| Land-cover reclassification tables | `data/workflow_assets/landcover_reclass/*.csv` |
| NHDPlus HR geodatabase | `data/analysis_inputs/hydrography/nhdplus/NHDPlus_H_National_Release_2_GDB/NHDPlus_H_National_Release_2_GDB.gdb` |
| Coastline | `data/analysis_inputs/linear_features/coastline/tl_2019_us_coastline_50_states.gpkg` |

## Execution Workflow

Run commands from the repository root unless otherwise noted.

### Preparation

#### 1. Prepare PAD-US Lands

`prepare_padus_all_and_public_lands.py` clips PAD-US features to the USA
boundary, simplifies them with a 15 m tolerance, and repairs geometries where
possible. With no arguments it reads the full PAD-US geodatabase listed above,
uses layer `PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement`, and
writes all three products:

- `padus_all_lands_clipped_to_usa_<timestamp>.gpkg` in
  `data/processing_outputs/padus_clipped_to_usa/all_lands`
- `padus_public_lands_clipped_to_usa_<timestamp>.gpkg` in
  `data/processing_outputs/padus_clipped_to_usa/public_lands`
- `padus_public_access_lands_clipped_to_usa_<timestamp>.gpkg` in
  `data/processing_outputs/padus_clipped_to_usa/public_access_lands`

Run:

```powershell
python prepare_padus_all_and_public_lands.py
```

Use `--padus-path` and `--padus-layer` to select a different input. Use
`--products` to select the outputs and `--output-dir <root>` to place the product
folders under a different root. Add `--validate-only` to check the selected
input and rules without writing the clipped GeoPackages.

The default classification policy is in
[`config/padus_land_rules.rules`](config/padus_land_rules.rules). Pass a
different rule file with `--rules-config <path>` when needed. The rules use the
stored PAD-US codes, not the longer descriptions displayed in GIS software.

The all-land product includes every PAD-US feature that has positive-area
overlap with the USA boundary after processing.

The public-land product selects features with federal, state, local government,
regional district, joint, or territorial manager types (`FED`, `STAT`, `LOC`,
`DIST`, `JNT`, `TERR`). Only an explicit unknown manager type (`UNK`) uses the
`Des_Tp` allowlist in the rule file. For example, a federally managed
conservation easement qualifies, but a conservation easement with an unknown
manager does not. Blank and other manager codes are excluded. Owner type does
not determine public-land status.

The public-access product starts from public land and excludes NASA-managed
features. Open Access (`OA`) records otherwise qualify directly. Restricted
Access (`RA`) and Unknown (`UK`) records also qualify if they are not privately
owned, are not managed by DOD or DOE, and do not have one of the excluded
private or military designations. Closed (`XA`) records do not qualify. The
script recognizes NASA by its exact PAD-US local manager value,
`National Aeronautics and Space Administration (NASA)`.

The clipped products contain `land_type`, source `OBJECTID`, the PAD-US
source attributes, and geometry. The rule parser rejects unsupported fields,
operators, codes, and malformed expressions before geometry processing starts.

##### Prepare all and public lands from the PAD-US analysis layer

Use the analysis layer for the `all` and `public` products. It does not carry
the owner and local-manager attributes required for the full public-access and
BBB workflows. Use the full source for those workflows, or keep their existing
county cuts and results separate from this run.

```powershell
$padusAnalysisPath = "data/analysis_inputs/padus/PADUS4_1VectorAnalysis_PADUS_Only (1)/PADUS4_1VectorAnalysis_PADUS_Only.gdb"
$padusAnalysisLayer = "PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP"
python prepare_padus_all_and_public_lands.py --padus-path $padusAnalysisPath --padus-layer $padusAnalysisLayer --products all public --validate-only
python prepare_padus_all_and_public_lands.py --padus-path $padusAnalysisPath --padus-layer $padusAnalysisLayer --products all public
```

Cut the two GeoPackages printed by preparation. Replace the placeholders with
those paths:

```powershell
python cut_and_flatten_by_county.py "<all-lands-clipped.gpkg>"
python cut_and_flatten_by_county.py "<public-lands-clipped.gpkg>"
```

Then prepare the [TIFF zonals](#ecosystem-service-tiff-zonals) for just these
products. Replace `<run>` with a unique run name in both paths to isolate the
results and temporary files:

```powershell
python prepare_tif_zonal_stats_config.py --zone all --zone public `
  --results-dir "data/analysis_results/zonal_statistics/padus_vectoranalysis_<run>" `
  --work-dir "data/processing_outputs/zonal_stats_work/padus_vectoranalysis_<run>"
```

Confirm that the printed zone paths are the two county cuts from this run,
then pass the printed config to the toolkit runner.

#### 2. Cut PAD-US Lands By County

`cut_and_flatten_by_county.py` converts a clipped PAD-US product into one
feature per county. It intersects the input with county boundaries, combines the
pieces within each county into a single non-overlapping polygon or multipolygon,
and copies the county attributes plus `land_type`.

Run the script once for each prepared PAD-US and BBB product:

```powershell
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\all_lands\padus_all_lands_clipped_to_usa_<timestamp>.gpkg
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\public_lands\padus_public_lands_clipped_to_usa_<timestamp>.gpkg
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\public_access_lands\padus_public_access_lands_clipped_to_usa_<timestamp>.gpkg
python cut_and_flatten_by_county.py .\data\processing_outputs\blm_lands_excluding_federally_protected_areas\blm_lands_excluding_federally_protected_areas_<timestamp>.gpkg
python cut_and_flatten_by_county.py .\data\processing_outputs\bbb_candidate_blm_lands\bbb_candidate_blm_lands_within_5_miles_of_population_centers_<timestamp>.gpkg
```

The resulting by-county PAD-US products are written under
`data/analysis_inputs/zonal_units`. Public-access outputs are routed to
`padus_public_access_lands_by_county`, protected-area-filtered BLM outputs to
`blm_lands_excluding_federally_protected_areas_by_county`, and final BBB
outputs to `bbb_candidate_blm_lands_by_county`. Select the products to analyze
in the zonal-statistics configuration.

#### 2a. Screen BBB PAD-US Candidate Lands

Use the USA-clipped **all-lands** GeoPackage from the full PAD-US source. The
screen writes eligible-state BLM fee land minus mapped federal protections and
a second layer restricted to land within five miles of qualifying population
centers.
The outputs are screening layers, not a list of tracts eligible for sale.

Run it with a PAD-US GeoPackage as the positional argument:

```powershell
python prepare_population_centers_2020.py

python filter_bbb_padus_by_population_centers.py `
  .\data\processing_outputs\padus_clipped_to_usa\all_lands\padus_all_lands_clipped_to_usa_<timestamp>.gpkg
```

The preparation command downloads and caches 2020 Census place geometry and
population tables, then writes the population-center input to
`data/analysis_inputs/census_population_centers_2020.gpkg`, using layers
`incorporated_places_pop1000` and `census_designated_places_pop1000`. The BBB
filter uses that file by default. Override the input, layers, output, or
distance when needed:

```powershell
python filter_bbb_padus_by_population_centers.py <padus.gpkg> `
  --population-centers-gpkg <population-centers.gpkg> `
  --input-layer <padus-layer> `
  --unprotected-output <blm-minus-protected.gpkg> `
  --output <final-bbb-candidates.gpkg> `
  --distance-miles 5
```

Without output overrides, the script writes the intermediate GeoPackage under
`data/processing_outputs/blm_lands_excluding_federally_protected_areas` and the
final GeoPackage under `data/processing_outputs/bbb_candidate_blm_lands`.

#### 3. Prepare Land-Cover Masks

`generate_nlcd_reclass_masks.py` creates 0/1/nodata byte rasters from the NLCD
2023 land cover raster and the CSV tables in
`data/workflow_assets/landcover_reclass`. Outputs are grouped by
reclassification table under `data/analysis_inputs/masks`.

The active top-level reclassification tables select the following NLCD classes.
Class names come from the MRLC NLCD land cover class legend:
<https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description>

- `forests`: `41` Deciduous Forest, `42` Evergreen Forest, and `43` Mixed Forest.
- `grasslands`: `71` Grassland/Herbaceous.
- `shrubland`: `51` Dwarf Scrub and `52` Shrub/Scrub.
- `water_snow`: `11` Open Water and `12` Perennial Ice/Snow.
- `wetlands`: `90` Woody Wetlands and `95` Emergent Herbaceous Wetlands.

Each table maps selected classes to `1` and listed non-selected NLCD classes to
`0`. Source nodata or unmapped classes are written as output nodata (`255`).
Tables under `data/workflow_assets/landcover_reclass/ignore` are exploratory
natural-land groupings and are not generated by default.

Run:

```powershell
python generate_nlcd_reclass_masks.py
```

#### 4. Prepare Freshwater Polygons

`prepare_nhd_freshwater_clipped_to_usa.py` prepares a simplified NHD freshwater
polygon layer from `NHDWaterbody` and `NHDArea`, clips it to the USA boundary,
and writes a timestamped GeoPackage under
`data/analysis_inputs/hydrography/nhdfreshwater`.

Run:

```powershell
python prepare_nhd_freshwater_clipped_to_usa.py
```

Freshwater is defined conservatively from NHD `FType` values using the USGS NHD
feature domains:
<https://www.usgs.gov/ngp-standards-and-specifications/national-hydrography-dataset-nhd-data-dictionary-feature-domains>

- `NHDWaterbody`: `390` LakePond, `436` Reservoir, and `466` SwampMarsh.
- `NHDArea`: `460` StreamRiver and `537` AreaOfComplexChannels.

Saltwater, estuarine, canal, ditch, playa, and ice-mass classes are not included
unless the rule is changed explicitly in the script.

#### Prepare the six assessment zones and check county land areas

Supply the PAD-US VectorAnalysis polygons and an existing BBB county-cut
GeoPackage. Treat both footprint inputs as land-only. Use Census `ALAND`
for the county land area; no separate land mask is required.

```powershell
python prepare_assessment_zones.py `
  --padus-path '<PAD-US-VectorAnalysis.gdb>' `
  --padus-layer 'PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP' `
  --bbb-by-county '<existing-BBB-county-cut.gpkg>' `
  --run-id 'assessment-2026'
```

Use `--counties` for an alternate county boundary with unique five-digit
`GEOID` values and Census `ALAND` in square metres. For multi-layer inputs,
select the desired layer with `--padus-layer`, `--counties-layer`, or
`--bbb-layer`. Use a fresh run ID, and set `--workers` to control concurrent
county geometry work (default: 4).

Find the prepared GeoPackages under
`data/processing_outputs/assessment_runs/<run-id>/zonal_units/`, one per
product directory: `counties`, `public_lands_by_county`,
`public_lands_by_county_and_manager_type`, `conservation_lands_by_county`,
`conservation_lands_by_county_and_manager_type`, and `bbb_lands_by_county`.
Use `zone_manifest.json` for input and output paths and `county_area_qa.csv`
for per-county area checks.

Public zones use the configured public-land rule. Conservation zones
independently include `GAP_Sts` codes 1, 2, and 3. Manager zones use
`GEOID` and `Mang_Type`; missing conservation manager codes are labeled
`UNSPECIFIED`. The county-level footprints are unions of the manager zones.
Review any reported cross-manager overlap before adding manager values.

County geometries retain their original water coverage. The area and share
calculation divides the clipped, unioned footprint area by `ALAND`, not the
full county polygon area. The QA CSV includes both county areas to make this
distinction visible. The command stops before publishing if `ALAND` is missing
or invalid, or if a footprint exceeds it by more than the larger of 1 m²
and one part per million of `ALAND`. Within that rounding tolerance, shares
are capped at 100% without changing the measured area. Passing this check
does not verify the land-only assumption. This command prepares geometry and
area checks; it does not calculate ecosystem-service valuations.

### Zonal Statistics

#### Ecosystem-service TIFF zonals

Place the raster inputs in `data/analysis_inputs/ecosystem_services`. Edit
[`snapp_assessment_tif_zonal_stats.yaml`](data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml)
to select rasters (`"include": true` in the `[project].raster_inventory` JSON
list). Paths are relative to the config file; vector patterns are relative to
`zonal_units_dir`. Use `--config <path.yaml>` for a different definition.

```powershell
python prepare_tif_zonal_stats_config.py
```

The script validates the enabled inputs, picks the newest matching county-cut
GeoPackage per zone, and prints a runnable config under
`data/workflow_assets/zonal_stats/runs/`. Check the selected paths and keep the
config with the results. Use repeatable `--zone` options to select a subset:

```powershell
python prepare_tif_zonal_stats_config.py --zone all --zone public --zone public-access --zone bbb-candidates
```

This TIFF profile applies the pollination hectare-to-pixel conversion. For the
county valuation builder, use county zonals with **unscaled** pollination sums
instead. Supply coral and mangrove vectors and the air-quality CSV to the
[county builder](#build-county-ecosystem-service-valuations-for-the-2026-assessment),
not to the TIFF runner.

Run the exact path printed by preparation. Do not use a wildcard that would
execute several runs. Replace `<zonal_stats_toolkit_repo>` with your toolkit
clone and `<UTC-timestamp>` with the printed filename:

```powershell
conda activate zonal-stats-toolkit
python "<zonal_stats_toolkit_repo>\runner.py" ".\data\workflow_assets\zonal_stats\runs\snapp_assessment_tif_zonal_stats_<UTC-timestamp>.yaml" --job-workers 4 --raster-workers 2
```

For PAD-US and BBB zonals, use the toolkit's CSV and GeoPackage with the same
job name and timestamp. Specify `--results-dir` when preparing the config to
keep each run's outputs in its own directory. The default is
`data/analysis_results/zonal_statistics/`; each footprint has a subdirectory.
For example, the all-lands pair is
`padus_all_lands/padus_all_lands_<timestamp>.{csv,gpkg}` within the
selected results directory. Keep the run config with the files.

#### Raster and vector configuration

The zonal statistics configuration is:

```text
data/workflow_assets/zonal_stats/snapp_assessment_zonal_stats.yaml
```

Before running, set the concrete county-cut and freshwater GeoPackage paths
and review the raster glob in this file. This configuration does not use the
TIFF profile's raster inventory or scaling rules.

The configuration runs the same analysis families for three zonal datasets:

- Counties, keyed by `GEOID`.
- PAD-US all lands by county, keyed by `GEOID`.
- PAD-US public lands by county, keyed by `GEOID`.

From this repository root, run the toolkit runner. Replace
`<zonal_stats_toolkit_repo>` with the path to a local clone of
[`zonal_stats_toolkit`](https://github.com/springinnovate/zonal_stats_toolkit):

```powershell
python "<zonal_stats_toolkit_repo>\runner.py" .\data\workflow_assets\zonal_stats\snapp_assessment_zonal_stats.yaml
```

This configuration runs separate raster, mask, area, freshwater, and coastline
jobs. Its configured metrics are:

| Input family | Operations |
| --- | --- |
| Ecosystem service rasters | `sum`, `mean`, `area_ha_valid`, `proportion_valid_nonzero` |
| NLCD reclassification masks | `sum`, `valid_count`, `total_count`, `area_ha_valid`, `area_ha_total`, `min`, `max` |
| Zonal unit area | `intersect_area_ha` |
| NHD freshwater polygons | `intersect_area_ha` |
| Coastline | `intersect_length_km` |

Keep each job's timestamped CSV and GeoPackage together under
`data/analysis_results/zonal_statistics/`.

### Calculate county NLCD areas without rerunning ecosystem-service zonals

Use the same Python environment as the zonal toolkit. Replace
`<zonal_stats_toolkit_repo>` with your toolkit clone and `<county-values.gpkg>`
with a completed county valuation file that does not contain NLCD area fields
or ES totals and ratios. Run from the repository:

```powershell
python update_nlcd_mask_areas.py --runner "<zonal_stats_toolkit_repo>\runner.py" --county-results "<county-values.gpkg>" --raster-workers 2
```

This **single command** freezes the tracked
[`snapp_assessment_nlcd_masks.yaml`](data/workflow_assets/zonal_stats/snapp_assessment_nlcd_masks.yaml),
runs only its five existing NLCD masks, validates the raw results, and writes:

- `data/analysis_results/nlcd_mask_areas/<timestamp>/counties_nlcd_mask_areas_<timestamp>.{csv,gpkg}`:
  county attributes, one valid-coverage area, five class areas and proportions,
  and the sum of the five class areas.
- `data/analysis_results/combined/snapp_ecosystem_services_counties_<timestamp>.{csv,gpkg}`:
  the explicitly selected existing county valuations with those same twelve
  fields appended, plus the [county totals and ratios](#add-county-totals-and-ratios-without-rerunning-zonals).
  Service values, adjustment factors, geometry, and existing
  provenance tables are preserved, not recalculated.

The script prints the exact paths and preserves existing outputs. Omit
`--county-results` to produce only the mask-area pair, for example before running
the full county valuation builder. Set `sources.natural_areas` in
[`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml)
to that mask-area GeoPackage and its feature layer before building valuations.
With `--county-results`, the command also calculates `es_total`, denominators
and ratios using the mask areas from this run. It does not regenerate the masks
or rerun ecosystem-service zonals and valuations.

`area_ha_valid_nlcd` is the observed NLCD footprint, **not** the county's full
polygon area. The five class areas cover forests, grasslands, shrubland,
water/snow, and wetlands; their sum cannot exceed that footprint. A valid zero
pixel contributes to coverage but not class area. NoData contributes to neither.
Zero coverage produces zero observed hectares and null proportions; it does not
mean a county outside NLCD coverage has no natural land. These are pixel-centre
areas with the reviewed 15 m simplification cap, not fractional boundary areas.

The reviewed profile is version-controlled. Each invocation creates a new
immutable config under `data/workflow_assets/zonal_stats/runs/` and isolated raw
results/cache directories. Keep that config with the results. GeoPackages embed
the frozen config, field definitions, source/config/code hashes, and raster grid,
size and modification-time records. Raster contents are not fully hashed.

To append the **same completed mask run to a later county valuation export**,
no zonals need to run again. Use the paths printed by the original mask run:

```powershell
python update_nlcd_mask_areas.py --mask-results "<exact raw counties_masks_TIMESTAMP.gpkg>" --run-config "<its frozen mask .yaml>" --county-results "<new pre-mask county valuations.gpkg>" --output "<new output path.gpkg>"
```

This verifies county keys/boundaries and refuses mismatched inputs. CSV/GPKG
parity and preservation of every retained base attribute and geometry are checked
before publication. The raw mask run remains in its timestamped `raw/counties/`
directory; use that raw GeoPackage, not the already renamed mask-area export.

## Build County Ecosystem-Service Valuations for the 2026 Assessment

Use the county builder to combine county zonals with the tabular and vector
service inputs. Recreation and fisheries are county valuation fields, not
PAD-US or BBB zonal outputs. The builder does not run zonal statistics.

1. Place the files listed in the [input guide](docs/county_valuation_integration_2026.md#input-data)
   under `data/`. For coral, supply `data/analysis_inputs/ecosystem_services/aeb_coral.gpkg`
   with point layer `aeb_coral` and value field `BC_USD11`. For recreation, supply
   `data/analysis_inputs/recreation/2026_11_02/usa_nature_assessment_recreation_20261102.csv`
   and keep its companion README alongside it locally.
2. Check [`sources.csv`](data/workflow_assets/county_integration/sources.csv)
   for the input paths. Use county zonals with unscaled pollination sums; the
   builder applies the hectare-to-pixel factor.
3. Check [`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml)
   for the county-area, NLCD mask-area, GDP, and population inputs. If you need
   the NLCD areas, run the [mask-only step](#calculate-county-nlcd-areas-without-rerunning-ecosystem-service-zonals)
   without `--county-results` and set `sources.natural_areas` to its output.

With the geospatial environment active, run:

```powershell
python build_county_ecosystem_service_valuations_2026.py
```

Use `--workers` and `--geometry-workers` to set parallelism. The builder
prints the paths to the matching
`data/analysis_results/combined/snapp_ecosystem_services_counties_<timestamp>.{csv,gpkg}`
files. Both include the service values, adjustment factors, NLCD areas,
`es_total` and ratios. Use the GeoPackage for county geometry and the input
and allocation records.

To change output names or columns, edit
[`export_rules.yaml`](data/workflow_assets/county_integration/export_rules.yaml).
For field definitions and interpretation, see the
[county methods guide](docs/county_valuation_integration_2026.md#output-fields-and-interpretation).

### Add county totals and ratios without rerunning zonals

#### Run the finalizer

Use this when the county service values are already calculated and you want to
add `es_total` and its four ratios. The full county builder includes these
automatically, so it does not need a separate finalization step.

Activate the geospatial Python environment. Replace `<county-values.gpkg>`
with a completed, adjusted county valuation file that does not already contain
the ES totals and ratios, then run from the repository:

```powershell
python finalize_county_ecosystem_services.py --county-results "<county-values.gpkg>"
```

The script copies the service values, factors, county boundaries and supporting
tables, then appends the summaries. It prints the paths to a CSV and GeoPackage
in `data/analysis_results/combined/`, named
`snapp_ecosystem_services_counties_<timestamp>`. It does not rerun zonals or
service valuations.

To choose the output name, add `--output "<output.gpkg>"`; the CSV gets
the same name. When changing the calculation settings, rerun using the same
input you used the first time, rather than the file that already has totals.

The settings are in
[`derived_metrics.yaml`](data/workflow_assets/county_integration/derived_metrics.yaml):
the service fields to sum, the county-area/GDP/population files, and the ratio
denominators. Source paths are relative to `data/`. The input may contain all
twelve NLCD area/share fields. If it has none, the script joins the mask-area
GeoPackage selected in this file; the masks must use the same county geography.

The mask updater with `--county-results` also includes these summaries.

For the meaning of `es_total`, the ratios, and blank values, see the
[county results guide](docs/county_valuation_integration_2026.md#county-totals-and-ratios).
