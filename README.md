# SNAPP Nature Assessments

This repository supports a SNAPP assessment of ecosystem services over land
tenure and administrative units in the United States. The workflow prepares
geospatial inputs, runs zonal statistics with `zonal_stats_toolkit`, and joins
the metric-specific outputs into final county, PAD-US all-land, and PAD-US
public-land datasets.

The preparation scripts assume commands are run from the repository root on
Windows:

```powershell
cd D:\repositories\snapp_nature_assesssments-
conda env create -f environment.yml
conda activate geo
```

## Data Layout

Project data are intentionally organized by workflow role rather than by source
download. Large data files are ignored by git.

| Directory | Role |
| --- | --- |
| `data/analysis_inputs` | Inputs used directly by preprocessing or analysis jobs. This includes source rasters, source vectors, prepared masks, freshwater polygons, and county-level zonal units. |
| `data/processing_outputs` | Intermediate products that document how analysis inputs were derived but are not final assessment deliverables. |
| `data/workflow_assets` | Small tracked configuration files, reclassification tables, and runner configuration. |
| `data/analysis_results/zonal_statistics` | Individual `zonal_stats_toolkit` outputs, grouped into one subdirectory per final zonal dataset. |
| `data/analysis_results/combined` | Final joined CSV and GeoPackage deliverables. |

The main analysis inputs are:

| Input | Path |
| --- | --- |
| USA boundary | `data/analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg` |
| Counties | `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` |
| PAD-US geodatabase | `data/analysis_inputs/padus/PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/PADUS4_1Geodatabase.gdb` |
| Ecosystem service rasters | `data/analysis_inputs/ecosystem_services/*.tif` |
| NLCD 2023 land cover raster | `data/analysis_inputs/nlcd/Annual_NLCD_LndCov_2023_CU_C1V0.tif` |
| Land-cover reclassification tables | `data/workflow_assets/landcover_reclass/*.csv` |
| NHDPlus HR geodatabase | `data/analysis_inputs/hydrography/nhdplus/NHDPlus_H_National_Release_2_GDB/NHDPlus_H_National_Release_2_GDB.gdb` |
| Coastline | `data/analysis_inputs/linear_features/coastline/tl_2019_us_coastline_50_states.gpkg` |

## PAD-US Preparation

`prepare_padus_all_and_public_lands.py` reads the PAD-US layer
`PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement`, clips it to the
USA boundary, simplifies geometry with a 15 m tolerance, repairs invalid
polygonal geometry where possible, and writes two stripped GeoPackage products:

| Product | Output directory |
| --- | --- |
| All PAD-US lands intersecting the USA boundary | `data/processing_outputs/padus_clipped_to_usa/all_lands` |
| Public PAD-US lands intersecting the USA boundary | `data/processing_outputs/padus_clipped_to_usa/public_lands` |

Run:

```powershell
python prepare_padus_all_and_public_lands.py
```

The public-land rule is written in the script as explicit constants and in the
`_feature_is_public` decision function. PAD-US stores coded values in the
geodatabase even when GIS software displays longer descriptions, so filtering
uses stored codes.

Features are retained in the public-land output when `Mang_Type` is one of:

| Display description | Stored value |
| --- | --- |
| Federal | `FED` |
| State | `STAT` |
| Local Government | `LOC` |
| Regional Agency Special District | `DIST` |
| Joint | `JNT` |
| Territorial | `TERR` |

When `Mang_Type` is `UNK` (`Unknown`), the feature is retained only when
`Own_Type` is one of:

| Display description | Stored value |
| --- | --- |
| Local Government | `LOC` |
| Regional Agency Special District | `DIST` |
| Federal | `FED` |
| Joint | `JNT` |
| State | `STAT` |

The all-land output uses the same geometry processing, but does not apply the
public-land attribute rule. Both products keep only `land_type` and geometry.

## County Zonal Units

`cut_and_flatten_by_county.py` cuts a polygon input by county. For each county,
it finds input features with positive-area intersection, intersects them with
the county boundary, unions the pieces into one non-overlapping polygon or
multipolygon, repairs invalid geometry where possible, and copies the county
attributes plus the input `land_type` value.

Run the script once for each PAD-US clipped-to-USA product:

```powershell
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\all_lands\padus_all_lands_clipped_to_usa_YYYY_MM_DD_HH_MM_SS.gpkg
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\public_lands\padus_public_lands_clipped_to_usa_YYYY_MM_DD_HH_MM_SS.gpkg
```

Outputs are written to:

| Product | Output directory |
| --- | --- |
| PAD-US all lands by county | `data/analysis_inputs/zonal_units/padus_all_lands_by_county` |
| PAD-US public lands by county | `data/analysis_inputs/zonal_units/padus_public_lands_by_county` |

These by-county products become analysis inputs for the final zonal statistics.

## Land-Cover Masks

`generate_nlcd_reclass_masks.py` creates byte rasters from the NLCD 2023 land
cover raster and the CSV tables in `data/workflow_assets/landcover_reclass`.
Each output raster uses `1` for selected classes, `0` for unselected classes,
and `255` for nodata. Outputs are grouped by reclassification table stem under
`data/analysis_inputs/masks`.

Run:

```powershell
python generate_nlcd_reclass_masks.py
```

The script runs one process per reclassification table, reads the NLCD raster in
blocks, writes one output raster per table, and shows a progress bar for each
mask.

## Freshwater Preparation

`prepare_nhd_freshwater_clipped_to_usa.py` prepares a simplified NHD freshwater
polygon layer. It reads `NHDWaterbody` and `NHDArea`, keeps freshwater feature
types, clips them to the USA boundary, simplifies with a 15 m tolerance, repairs
invalid polygonal geometry where possible, and writes a timestamped GeoPackage
to `data/analysis_inputs/hydrography/nhdfreshwater`.

Run:

```powershell
python prepare_nhd_freshwater_clipped_to_usa.py
```

Freshwater is defined conservatively from NHD `FType` values. The included types
are based on the USGS NHD feature domains and feature-class descriptions:
<https://www.usgs.gov/ngp-standards-and-specifications/national-hydrography-dataset-nhd-data-dictionary-feature-domains>

| Source layer | FType | Description |
| --- | --- | --- |
| `NHDWaterbody` | `390` | LakePond |
| `NHDWaterbody` | `436` | Reservoir |
| `NHDWaterbody` | `466` | SwampMarsh |
| `NHDArea` | `460` | StreamRiver |
| `NHDArea` | `537` | AreaOfComplexChannels |

The preparation excludes saltwater, estuarine, canal, ditch, playa, and ice-mass
classes unless the rule is changed explicitly in the script.

## Zonal Statistics

The zonal statistics configuration is:

```text
data/workflow_assets/zonal_stats/snapp_assessment_zonal_stats.yaml
```

The runner configuration is INI-style despite the `.yaml` extension. It runs the
same analysis families for three zonal datasets:

| Zonal dataset | Aggregation vector |
| --- | --- |
| Counties | `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` |
| PAD-US all lands by county | `data/analysis_inputs/zonal_units/padus_all_lands_by_county/padus_all_lands_clipped_by_county_YYYY_MM_DD_HH_MM_SS.gpkg` |
| PAD-US public lands by county | `data/analysis_inputs/zonal_units/padus_public_lands_by_county/padus_public_lands_clipped_by_county_YYYY_MM_DD_HH_MM_SS.gpkg` |

The aggregation key for all jobs is `GEOID`.

Before running, update the concrete timestamped paths in the configuration if
new PAD-US by-county or NHD freshwater products have been generated. The toolkit
supports glob patterns for rasters, but vector inputs must be concrete paths.

Run from this repository root with the `zonal_stats_toolkit` environment, or an
equivalent environment that can import the toolkit dependencies, active:

```powershell
conda activate zonal-stats-toolkit
python D:\repositories\zonal_stats_toolkit\runner.py .\data\workflow_assets\zonal_stats\snapp_assessment_zonal_stats.yaml
```

The configured metrics are:

| Input family | Operations |
| --- | --- |
| Ecosystem service rasters | `sum`, `mean`, `stdev`, `valid_count`, `total_count`, `area_ha_valid`, `area_ha_total` |
| NLCD reclassification masks | `sum` |
| Zonal unit area | `intersect_area_ha` |
| NHD freshwater polygons | `intersect_area_ha` |
| Coastline | `intersect_length_km` |

Outputs are written under `data/analysis_results/zonal_statistics` in three
subdirectories: `counties`, `padus_all_lands`, and `padus_public_lands`.

## Final Combination

`combine_final_zonal_stats_results.py` joins the latest timestamped CSV and
GeoPackage outputs within each zonal-statistics subdirectory. CSV files are
joined to CSV outputs, and GeoPackage files are joined to GeoPackage outputs.
The script keeps shared columns once and raises an error if repeated fields have
conflicting values for the same `GEOID`.

Run:

```powershell
python combine_final_zonal_stats_results.py
```

The final deliverables are written to `data/analysis_results/combined`:

| Deliverable group | CSV and GeoPackage stem |
| --- | --- |
| Counties | `counties_combined_YYYY_MM_DD_HH_MM_SS` |
| PAD-US all lands by county | `padus_all_lands_combined_YYYY_MM_DD_HH_MM_SS` |
| PAD-US public lands by county | `padus_public_lands_combined_YYYY_MM_DD_HH_MM_SS` |

Use `--check-gpkg-geometry` only when geometry consistency needs to be verified
during the final join. It is slower and is disabled by default because the
GeoPackage geometries should already match within each result group.

## Runtime Notes

The PAD-US, NHD, and NLCD preparation steps are the expensive parts of the
workflow. They use process-based parallelism and report progress with `tqdm`.
Runtime depends on disk speed, geometry complexity, and the number of CPU cores.

The zonal statistics runner is also expected to be long-running because it
summarizes multiple rasters and vector measures over three zonal datasets. The
final combination step is comparatively light; on the local prepared outputs it
has completed in roughly 10 to 15 seconds when geometry comparison is disabled.
