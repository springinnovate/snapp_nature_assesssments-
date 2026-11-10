# Supporting workflows

Run these commands from the repository root in the
[geospatial environment](../README.md#1-set-up-python). Use them to prepare
replacement inputs or run a standalone analysis. They are not steps to repeat
when running the [CONUS Ecosystem Services Assessment](../README.md) with its
prepared inputs.

## Prepare the Census county land mask

Use the 2024 TIGER/Line Faces ZIP archives, `tl_2024_<GEOID>_faces.zip`, for all CONUS counties. Use county boundaries from the same Census release.

```powershell
python prepare_census_county_land_mask.py --faces-dir 'data/analysis_inputs/boundaries/census_faces/2024'
```

Add `--download-missing` to download missing official archives into that directory. Use `--counties` and `--counties-layer` to select the county source. The default output is `data/analysis_inputs/boundaries/county_land/tl_2024_us_county_land_conus.gpkg`, layer `county_land_mask`. Keep its matching `_qa.csv` and `_manifest.json`. Pass the GeoPackage as `--land-mask` in a fresh assessment.

## Prepare county reporting areas and natural-area denominators

Run this when the prepared reporting-area files are missing or the county/water
sources change:

```cmd
set "PROJ_NETWORK=OFF"
python prepare_county_reporting_areas.py --runner "..\zonal_stats_toolkit\runner.py"
```

The command uses local inputs only: 2024 county boundaries and Census land
mask, `boundaries/usa_boundary/usa_vector.gpkg` under `data/analysis_inputs/`,
and the TIGER coastline, clipped NHD water polygons, and Great Lakes feature
list configured in [`sources.csv`](../data/workflow_assets/county_integration/sources.csv).
It uses the marine shoreline to identify connected water outside the U.S.
boundary, including water shared across county lines, and removes the five
selected NHD Great Lakes footprints. Overlapping exclusions are counted once
and Census land is preserved. Great Salt Lake, Tahoe, and other inland water
remain in the reporting area. The generalized boundary approximates the
coastal cutoff; some tidal waters may remain.
Use `--usa-boundary` for a different local boundary and `--sources` for a
different source table.

Only the five NLCD binary masks are sampled on this footprint. The final
`tl_2024_us_county_excl_ocean_great_lakes.csv` and matching GeoPackage under
`data/analysis_inputs/boundaries/county_reporting_areas/` contain the county
denominator and natural-area metrics. Keep their QA and manifest sidecars.
Existing final files are not overwritten; use `--output` for a separately named
replacement and update both county-area and natural-area paths in
[`derived_metrics.yaml`](../data/workflow_assets/county_integration/derived_metrics.yaml).
Do not use a `--geometry-only` result as a finished denominator source: it has
not calculated the natural-area metrics.

Then [revalue the completed assessment](../README.md#revalue-a-completed-run).
This leaves ecosystem-service raster sums, service values, fisheries weights,
and the original assessment geometries unchanged.

## Prepare the BBB county footprint

Use the full PAD-US combined source with the ownership and management
attributes, not the PAD-US VectorAnalysis layer. The first command uses the
[default full geodatabase and USA boundary paths](../prepare_padus_all_and_public_lands.py)
to create the clipped `all` product. Use the path it prints in the BBB command.
If you already have the clipped layer or Census population centers, reuse them.

```powershell
python prepare_padus_all_and_public_lands.py --products all
python prepare_population_centers_2020.py
python filter_bbb_padus_by_population_centers.py '<printed PAD-US all-lands GeoPackage>'
python cut_and_flatten_by_county.py '<printed BBB candidate GeoPackage>'
```

The Census script downloads and caches 2020 place geometry and population data and recreates `data/analysis_inputs/census_population_centers_2020.gpkg`. Use its `--output` and the filter's `--population-centers-gpkg` together to keep a separate population-center file. The filter writes an unprotected BLM product and a BBB candidate product. Cut the **candidate** GeoPackage printed under `data/processing_outputs/bbb_candidate_blm_lands/`; the county cut is written to `data/analysis_inputs/zonal_units/bbb_candidate_blm_lands_by_county/`. Use that county cut for `--bbb-by-county`, not its zonal results. It is a mapped screening footprint, not a determination that a tract can or will be sold.

## Prepare NHD water polygons

```powershell
python prepare_nhd_freshwater_clipped_to_usa.py
```

The [script](../prepare_nhd_freshwater_clipped_to_usa.py) reads the configured national NHDPlus HR geodatabase under `data/analysis_inputs/hydrography/nhdplus/` and the USA boundary under `data/analysis_inputs/boundaries/usa_boundary/`. It writes a timestamped GeoPackage to `data/analysis_inputs/hydrography/nhdfreshwater/`. To use the new file, update `freshwater_polygons` in [`sources.csv`](../data/workflow_assets/county_integration/sources.csv). If changing NHD releases, verify that the two inland fishery exclusion tables still identify the intended source features.

The water layer selects NHD feature types LakePond (`390`), Reservoir (`436`), SwampMarsh (`466`), StreamRiver (`460`), and AreaOfComplexChannels (`537`). These feature types do not guarantee fresh water. The Great Lakes and Great Salt Lake exclusions are applied to inland fisheries allocations, not removed from the shared water layer.

## Calculate county NLCD areas

If replacement masks are needed, place `Annual_NLCD_LndCov_2023_CU_C1V0.tif`
under `data/analysis_inputs/nlcd/` and check the
[reclassification tables](../data/workflow_assets/landcover_reclass/).

```powershell
python generate_nlcd_reclass_masks.py
```

The generated TIFFs go under `data/analysis_inputs/masks/`. Point the
[NLCD zonal profile](../data/workflow_assets/zonal_stats/snapp_assessment_nlcd_masks.yaml)
at the five mask TIFFs you intend to measure, then run the
[county reporting-area preparation](#prepare-county-reporting-areas-and-natural-area-denominators)
with a new `--output`. This measures the masks on the ocean/Great-Lakes-excluded
footprint and supplies matching inputs for both area-normalized ES metrics.

For a separate full-county NLCD coverage diagnostic, use:

```powershell
python update_nlcd_mask_areas.py --runner '<path to zonal_stats_toolkit/runner.py>' --raster-workers 2
```

This prints a verified county CSV/GeoPackage pair under
`data/analysis_results/nlcd_mask_areas/<timestamp>/`. It includes the entire
county geometry and is not the reporting-area input for the assessment ratios.

## Refresh PAD-US footprints

To refresh the separate all/public footprint zonals and whole-county valuations, use the all/public wrapper. It also copies a completed BBB zonal pair. Supply its GeoPackage with the same-stem CSV beside it; both source filenames must end in `_YYYYMMDD_HHMMSS` before the extension.

```powershell
python run_snapp_assessment.py `
  --runner '<path to zonal_stats_toolkit/runner.py>' `
  --bbb-results '<completed BBB zonal results.gpkg>'
```

Add `--validate-only` to check input paths. The output is four pairs in `data/analysis_results/assessment_runs/<run-id>/`; `run_manifest.json` names the files.

For geometry alone, use the PAD-US VectorAnalysis layer for `all` and `public`:

```powershell
python prepare_padus_all_and_public_lands.py `
  --padus-path 'data/analysis_inputs/padus/PADUS4_1VectorAnalysis_PADUS_Only (1)/PADUS4_1VectorAnalysis_PADUS_Only.gdb' `
  --padus-layer 'PADUS4_1VectorAnalysis_PADUS_Only_Simp_SingP' --products all public
python cut_and_flatten_by_county.py '<printed clipped GeoPackage>'
```

Run the county cut once for each desired product. It writes a unioned footprint per county under `data/analysis_inputs/zonal_units/` for recognized products. To prepare `public-access`, use the full PAD-US source, which is the default when no path or layer is passed, and request `--products public-access`.

## Run standalone zonals

To run selected ecosystem-service TIFF zonals against already prepared footprints, check the [TIFF inventory](../data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml). Select zone keys explicitly; the preparer prints the matching input GeoPackages and the frozen config to run.

```powershell
python prepare_tif_zonal_stats_config.py --zone all --zone public `
  --results-dir 'data/analysis_results/zonal_statistics/<run-id>' `
  --work-dir 'data/processing_outputs/zonal_stats_work/<run-id>'
python '<path to zonal_stats_toolkit/runner.py>' '<printed frozen config.yaml>'
```

Confirm the printed input paths before running. Keep the matching CSV and GeoPackage for each job, together with the frozen config. This TIFF profile scales pollination; do not use its scaled county pollination sums as raw inputs to the county builder.

For standalone raster and vector jobs, including county area, freshwater area, and coastline length, review the concrete paths and raster glob in [the raster/vector profile](../data/workflow_assets/zonal_stats/snapp_assessment_zonal_stats.yaml) and run it directly:

```powershell
python '<path to zonal_stats_toolkit/runner.py>' 'data/workflow_assets/zonal_stats/snapp_assessment_zonal_stats.yaml'
```

## Build or finalize county valuations

To combine whole-county zonals with the county service inputs, select the raw zonal GeoPackage in [`sources.csv`](../data/workflow_assets/county_integration/sources.csv) and check the [input guide](county_valuation_integration_2026.md#input-data). The raw pollination values must be unscaled. Set the area, NLCD, GDP, and population inputs in [`derived_metrics.yaml`](../data/workflow_assets/county_integration/derived_metrics.yaml).

```powershell
python build_county_ecosystem_service_valuations_2026.py
```

This writes a matched `snapp_ecosystem_services_counties_<timestamp>` CSV/GeoPackage pair to `data/analysis_results/combined/`, including service adjustments and the county totals and ratios. The builder does not run zonal statistics.

To add totals and ratios to an already adjusted county valuation that lacks them, run the finalizer instead:

```powershell
python finalize_county_ecosystem_services.py --county-results '<adjusted county valuations.gpkg>'
```

The finalizer copies the values and geometry unchanged and writes a new pair in the same default output directory. Add `--output '<new output.gpkg>'` to set the destination. It reads the corrected NLCD areas from the input or the configured mask-area GeoPackage. See the [county valuation methods](county_valuation_integration_2026.md) for output field definitions.
