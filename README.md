# SNAPP Nature Assessments

This repository supports a SNAPP assessment of ecosystem services over counties,
all PAD-US lands, and public PAD-US lands in the United States. The workflow
prepares geospatial inputs, runs zonal statistics with
[`zonal_stats_toolkit`](https://github.com/springinnovate/zonal_stats_toolkit),
and combines the metric-specific outputs into final CSV and GeoPackage
deliverables.

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
GDAL, GeoPandas, Rasterio, Shapely 2, NumPy, Pandas, and tqdm.

The zonal statistics step is run with `zonal_stats_toolkit`. See the toolkit
repository for its current installation instructions and supported execution
environment:
<https://github.com/springinnovate/zonal_stats_toolkit>

## Data Layout

The `data/` directory is organized by workflow role.

| Directory | Contents |
| --- | --- |
| `data/analysis_inputs` | Inputs used directly by preprocessing or analysis jobs. This includes source rasters, source vectors, prepared masks, freshwater polygons, and county-level zonal units. |
| `data/processing_outputs` | Intermediate products that document how analysis inputs were derived. |
| `data/workflow_assets` | Small tracked configuration files, reclassification tables, and runner configuration. |
| `data/analysis_results/zonal_statistics` | Individual `zonal_stats_toolkit` outputs, grouped into one subdirectory per final zonal dataset. |
| `data/analysis_results/combined` | Final joined CSV and GeoPackage deliverables. |

The main input paths are:

| Input | Path |
| --- | --- |
| USA boundary | `data/analysis_inputs/boundaries/usa_boundary/usa_vector.gpkg` |
| Counties | `data/analysis_inputs/zonal_units/counties/tl_2024_us_county_50_states.gpkg` |
| PAD-US geodatabase | `data/analysis_inputs/padus/PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/PADUS4_1Geodatabase.gdb` |
| Recreation value polygons | `data/analysis_inputs/recreation/usa_nature_assessment_recreation.gpkg` |
| Ecosystem service rasters | `data/analysis_inputs/ecosystem_services/*.tif` |
| NLCD 2023 land cover raster | `data/analysis_inputs/nlcd/Annual_NLCD_LndCov_2023_CU_C1V0.tif` |
| Land-cover reclassification tables | `data/workflow_assets/landcover_reclass/*.csv` |
| NHDPlus HR geodatabase | `data/analysis_inputs/hydrography/nhdplus/NHDPlus_H_National_Release_2_GDB/NHDPlus_H_National_Release_2_GDB.gdb` |
| Coastline | `data/analysis_inputs/linear_features/coastline/tl_2019_us_coastline_50_states.gpkg` |

## Execution Workflow

Run commands from the repository root unless otherwise noted.

### Preparation

#### 1. Prepare PAD-US Lands

`prepare_padus_all_and_public_lands.py` converts the PAD-US layer
`PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement` into two
USA-clipped GeoPackages: one all-land product and one public-land subset.
Geometries are simplified with a 15 m tolerance, clipped to the USA boundary,
and repaired where possible.

- `padus_all_lands_clipped_to_usa_<timestamp>.gpkg` in
  `data/processing_outputs/padus_clipped_to_usa/all_lands`
- `padus_public_lands_clipped_to_usa_<timestamp>.gpkg` in
  `data/processing_outputs/padus_clipped_to_usa/public_lands`

Run:

```powershell
python prepare_padus_all_and_public_lands.py
```

The all-land product includes every PAD-US feature that has positive-area
overlap with the USA boundary after processing.

The public-land product is a subset of the all-land product. PAD-US stores
coded values in the geodatabase even when GIS software displays longer
descriptions, so the rule uses stored codes:

- Before applying the inclusion rules, exclude features where `Pub_Access` is
  `XA` (closed access), `Own_Type` is `PVT` (private owner), `Mang_Name` or
  `Own_Name` is `DOD` or `DOE`, or `Des_Tp` is `MIL`. The `Pub_Access` value
  `UK` (unknown access) is not excluded by itself.
- Keep features where `Mang_Type` is `FED`, `STAT`, `LOC`, `DIST`, `JNT`, or
  `TERR`. These correspond to Federal, State, Local Government, Regional Agency
  Special District, Joint, and Territorial managers.
- If `Mang_Type` is `UNK`, keep the feature only when `Own_Type` is `LOC`,
  `DIST`, `FED`, `JNT`, or `STAT`. These correspond to Local Government,
  Regional Agency Special District, Federal, Joint, and State owners.
- Exclude all other features from the public-land product.

Both PAD-US products keep only `land_type` and geometry.

#### 2. Cut PAD-US Lands By County

`cut_and_flatten_by_county.py` converts a clipped PAD-US product into one
feature per county. It intersects the input with county boundaries, combines the
pieces within each county into a single non-overlapping polygon or multipolygon,
and copies the county attributes plus `land_type`.

Run the script once for all lands and once for public lands:

```powershell
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\all_lands\padus_all_lands_clipped_to_usa_<timestamp>.gpkg
python cut_and_flatten_by_county.py .\data\processing_outputs\padus_clipped_to_usa\public_lands\padus_public_lands_clipped_to_usa_<timestamp>.gpkg
```

The resulting by-county PAD-US products are written under
`data/analysis_inputs/zonal_units` and become zonal units for later analysis.

#### 2a. Prepare Recreation Value By County

`prepare_recreation_value_by_county.py` allocates the `val_2024` values from
`data/analysis_inputs/recreation/usa_nature_assessment_recreation.gpkg` to
counties. The script reprojects recreation features and counties to EPSG:5070
for equal-area intersection math, skips recreation features where `val_2024` is
zero or null, and uses a spatial index over the remaining recreation features to
find possible county overlaps.

For each positive-area county intersection, the script assigns the county the
same fraction of the recreation feature value as the county receives of that
feature's area:

```text
county_value += val_2024 * (intersection_area / recreation_feature_area)
```

All counties are kept in the output. Counties without positive-area recreation
overlap receive `0`. The output fields are `GEOID` and
`proportional_recreation_val_2024`, with county geometry written in EPSG:5070.

Run:

```powershell
python prepare_recreation_value_by_county.py
```

The output is written to
`data/analysis_inputs/zonal_units/recreation_by_county/recreation_value_by_county_<timestamp>.gpkg`.

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

### Zonal Statistics

The zonal statistics configuration is:

```text
data/workflow_assets/zonal_stats/snapp_assessment_zonal_stats.yaml
```

Before running, update the concrete timestamped vector paths in that file if new
PAD-US by-county or NHD freshwater products have been generated. The toolkit
supports glob patterns for raster inputs, but vector inputs are listed as
explicit GeoPackage paths.

The configuration runs the same analysis families for three zonal datasets:

- Counties, keyed by `GEOID`.
- PAD-US all lands by county, keyed by `GEOID`.
- PAD-US public lands by county, keyed by `GEOID`.

From this repository root, run the toolkit runner. Replace
`<zonal_stats_toolkit_repo>` with the path to a local clone of
[`zonal_stats_toolkit`](https://github.com/springinnovate/zonal_stats_toolkit):

```powershell
python <zonal_stats_toolkit_repo>\pipeline_runner.py .\data\workflow_assets\zonal_stats\snapp_assessment_zonal_stats.yaml
```

The configured metrics are:

| Input family | Operations |
| --- | --- |
| Ecosystem service rasters | `sum`, `mean`, `stdev`, `valid_count`, `total_count`, `area_ha_valid`, `area_ha_total` |
| NLCD reclassification masks | `sum` |
| Zonal unit area | `intersect_area_ha` |
| NHD freshwater polygons | `intersect_area_ha` |
| Coastline | `intersect_length_km` |

The runner writes individual timestamped outputs under
`data/analysis_results/zonal_statistics`.

### Final Combination

`combine_final_zonal_stats_results.py` joins the latest timestamped CSV and
GeoPackage outputs within each zonal-statistics subdirectory. Shared columns are
kept once, and repeated fields with conflicting values for the same `GEOID`
raise an error. It also joins the latest prepared recreation value by county
from `data/analysis_inputs/zonal_units/recreation_by_county`, carrying only
`proportional_recreation_val_2024` into the county, PAD-US all-land, and PAD-US
public-land final outputs.

During this step, timestamped NLCD mask artifact fields such as
`area_ha_valid_reclassified_NLCD2023_*` are replaced with stable derived class
area fields:

- `area_ha_nlcd_forests`
- `area_ha_nlcd_grasslands`
- `area_ha_nlcd_shrubland`
- `area_ha_nlcd_water_snow`
- `area_ha_nlcd_wetlands`

The final outputs keep the corresponding `proportion_valid_nonzero_*` fields.

Run:

```powershell
python combine_final_zonal_stats_results.py
```

The final deliverables are:

- `counties_combined_<timestamp>.csv` and
  `counties_combined_<timestamp>.gpkg`.
- `padus_all_lands_combined_<timestamp>.csv` and
  `padus_all_lands_combined_<timestamp>.gpkg`.
- `padus_public_lands_combined_<timestamp>.csv` and
  `padus_public_lands_combined_<timestamp>.gpkg`.

By default, these files are written to `data/analysis_results/combined`.
Each contains `proportional_recreation_val_2024` joined by `GEOID`.

## Runtime Notes

The PAD-US, NHD, NLCD preparation, and zonal statistics steps are the expensive
parts of the workflow. On the local NVMe workstation used for this work, with 32
logical processors and 128 GB RAM, the full preparation and zonal statistics
workflow should be expected to take a few hours.

The final combination step is comparatively light and should take about 10 to 15
seconds.
