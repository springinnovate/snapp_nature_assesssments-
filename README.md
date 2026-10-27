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
GDAL, GeoPandas, Rasterio, Shapely 2, NumPy, Pandas, PyYAML, and tqdm.

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
| Ecosystem service rasters | `data/analysis_inputs/ecosystem_services` (selected in `data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml`) |
| NLCD 2023 land cover raster | `data/analysis_inputs/nlcd/Annual_NLCD_LndCov_2023_CU_C1V0.tif` |
| Land-cover reclassification tables | `data/workflow_assets/landcover_reclass/*.csv` |
| NHDPlus HR geodatabase | `data/analysis_inputs/hydrography/nhdplus/NHDPlus_H_National_Release_2_GDB/NHDPlus_H_National_Release_2_GDB.gdb` |
| Coastline | `data/analysis_inputs/linear_features/coastline/tl_2019_us_coastline_50_states.gpkg` |

## Execution Workflow

Run commands from the repository root unless otherwise noted.

### Preparation

#### 1. Prepare PAD-US Lands

`prepare_padus_all_and_public_lands.py` converts the PAD-US layer
`PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement` into three
USA-clipped GeoPackages: an all-land product, a public-land subset, and a
public-access subset of the public-land product.
Geometries are simplified with a 15 m tolerance, clipped to the USA boundary,
and repaired where possible.

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

The default classification policy is stored in
`config/padus_land_rules.rules`. Pass a different rule file with
`--rules-config <path>` when needed. The file uses a constrained, declarative
syntax; it is parsed and validated without executing Python or SQL:

```text
public_land =
    Own_Type not in {TRIB}
    AND (
        Own_Type in {DIST, FED, JNT, LOC, STAT}
        OR (
            Own_Type in {DESG, NGO, PVT, UNK}
            AND Mang_Type in {DIST, FED, JNT, LOC, STAT}
        )
    )

public_access =
    public_land
    AND (
        Pub_Access in {OA}
        OR (
            Pub_Access in {RA, UK}
            AND Own_Type not in {PVT}
            AND manager not in {DOD, DOE, NASA}
            AND Des_Tp not in {
                MIL, PAGR, PCON, PFOR, PHCA, POTH, PPRK, PRAN, PREC
            }
        )
    )
```

The all-land product includes every PAD-US feature that has positive-area
overlap with the USA boundary after processing.

The public-land product is a subset of the all-land product. PAD-US stores
coded values in the geodatabase even when GIS software displays longer
descriptions, so the configured rule uses stored codes. Federal, Joint, Local
Government, Regional Agency Special District, and State owners are included.
Designation, Non-Governmental Organization, Private, and Unknown owners are
included only when managed by one of those public entity types. Tribal and
Territorial records are not selected.

The public-access product is a subset of public land. Open Access (`OA`) records
are included directly. Restricted Access (`RA`) and Unknown (`UK`) records are
included only when they are not privately owned, are not managed by DOD, DOE,
or NASA, and do not use one of the configured private or military designation
codes. Closed (`XA`) records are excluded. PAD-US 4.1 stores NASA under the
local manager value `National Aeronautics and Space Administration (NASA)`, so
the script normalizes that exact value to the configured `NASA` manager token.

All three clipped products contain `land_type`, source `OBJECTID`, the PAD-US
source attributes, and geometry. The rule parser rejects unsupported fields,
operators, codes, and malformed expressions before geometry processing starts.

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

`filter_bbb_padus_by_population_centers.py` screens a USA-clipped PAD-US
GeoPackage against the mappable land criteria in Section 50301. Before doing
geometry work, it selects records where:

- `FeatClass = 'Fee'`;
- `Own_Type = 'FED'`;
- `Mang_Name = 'BLM'`; and
- `State_Nm` is Alaska, Arizona, California, Colorado, Idaho, Nevada, New
  Mexico, Oregon, Utah, Washington, or Wyoming.

The script first subtracts federally protected areas represented by overlapping
PAD-US records and writes that intermediate BLM-only result. It uses designation
codes for National Monuments, National Recreation Areas, Wilderness Areas, Wild
and Scenic Rivers, National Trails, National Conservation Areas, National
Wildlife Refuges, and National Parks. It also uses NPS and FWS
approved/proclamation boundaries to cover National Park System units and,
conservatively, National Wildlife Refuge and National Fish Hatchery System
units. The FWS boundary proxy can include some additional FWS administrative
areas because PAD-US does not provide a separate system-membership field for
every boundary.

It then clips those records to the union of:

- a five-statute-mile band around the **boundary** of each incorporated
  municipality with a population of at least 1,000; and
- a five-statute-mile circle around the Census-provided `CENTLON`/`CENTLAT`
  centroid of each census-designated place with a population of at least 1,000.

Each place is buffered in its local UTM coordinate system so the five-mile
distance is not calculated in longitude/latitude or a single nationwide map
projection. Input features are clipped to the union of those zones, and all
source attributes are retained. Progress bars report PAD-US attribute
selection, protected-area loading and indexing, population-layer loading,
place buffering, zone union, output-schema creation, candidate scanning,
feature writing, and GeoPackage finalization.

The two outputs are screening layers, not determinations that a tract will be
offered or sold:

- `blm_lands_excluding_federally_protected_areas_<timestamp>.gpkg`, containing
  eligible-state BLM fee land after only the protected-area subtraction, with
  layer `blm_lands_excluding_federally_protected_areas` and `land_type` value
  `blm_excluding_federally_protected`; and
- `bbb_candidate_blm_lands_within_5_miles_of_population_centers_<timestamp>.gpkg`,
  containing the preceding land that also satisfies the bill's five-mile rule,
  with layer `bbb_candidate_blm_lands` and `land_type` value
  `bbb_candidate_blm`.

PAD-US does not establish existing grazing permits or leases, incompatible
valid existing rights, residential suitability, or tract selection.
Protected-area coverage is limited to the records present in the supplied
PAD-US GeoPackage.

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

#### 2b. Prepare Recreation Value By County

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

#### TIFF handoff workflow

For ecosystem-service TIFF analysis, place source rasters in
`data/analysis_inputs/ecosystem_services`, then review
`data/workflow_assets/zonal_stats/snapp_assessment_tif_zonal_stats.yaml`.
This version-controlled definition contains the raster inventory, units,
include/exclude decisions, scale factors, zone patterns, metrics, output roots,
and simplification tolerance.

The toolkit uses INI sections in files named `.yaml`; the inline
`[project].raster_inventory` is a JSON list inside that existing format. Only
entries with `"include": true` are analyzed. Paths in `[project]` are relative
to the definition; zone patterns are relative to `zonal_units_dir`. Use
`--config <path.yaml>` to prepare another reviewed definition.

Generate a raster-only runner configuration with:

```powershell
python prepare_tif_zonal_stats_config.py
```

The preparation step validates every enabled TIFF, automatically selects the
newest timestamped county-cut GeoPackage for each available land product, and
writes a new runnable snapshot:

```text
data/workflow_assets/zonal_stats/runs/snapp_assessment_tif_zonal_stats_<UTC-timestamp>.yaml
```

The definition and `runs/*.yaml` snapshots are Git-trackable. Commit the reviewed
definition and the snapshot actually used for an analysis. Each snapshot records
the complete source definition (including blocked raster reasons), its SHA-256,
the generator SHA-256, source and analysis raster paths, units and scale factors,
selected concrete vector versions, skipped-zone reasons, metrics, and outputs.
Keep that snapshot with any exported results so the analysis can be traced back
to its exact inputs and policy. `--output` can name a snapshot;
existing configs are never overwritten.

The reviewed `[project]` section defaults to
`max_simplify_tolerance_meters = 15`; preparation copies its configured value
into the snapshot. The zonal-statistics toolkit uses this
explicit physical tolerance when simplifying aggregation geometry; it does not
derive the tolerance from each raster's pixel size. The cap limits boundary
displacement while retaining geometry simplification for PAD-US performance.

By default the generated config covers counties, PAD-US all lands, PAD-US
public lands, PAD-US public-access lands, and BBB candidate BLM lands. It also
includes BLM lands excluding mapped federal protections when that intermediate
product has been cut by county. Use repeatable `--zone` options to run only a
subset; for example:

```powershell
python prepare_tif_zonal_stats_config.py --zone all --zone public --zone public-access --zone bbb-candidates
```

The inline inventory applies the required `0.09` pollination per-hectare to 30 m-pixel
conversion through a lightweight VRT, avoiding a duplicate multi-gigabyte
TIFF. Each run's VRT is stored in its own subdirectory under
`data/processing_outputs/zonal_stats_derived_rasters`, so later changes to scale
factors cannot overwrite an earlier run's conversion. Air quality and mental
health remain disabled in this handoff until validated value TIFFs are selected;
the county air-quality CSV is handled by the separate county valuation builder.
Mangroves remain disabled in this raster inventory because
`mangrove_CONUS.tif` contains categorical CCAP values (`-128`, `0`, `16`, and
`17`), not monetary values. The county valuation builder instead uses the
`ucsc_cwon_studyunits.gpkg` vector source described below.

The coral row also needs a metadata decision before publication: the spreadsheet
labels it USD 2010, while the local filename is `Coral_Reefs_2024adj_CPI.tif`.
It is enabled for the requested zonal run, and the discrepancy is recorded in
the inline inventory and every frozen run snapshot.

Run the exact path printed by preparation. Do not use a wildcard that would
execute several runs. Replace `<zonal_stats_toolkit_repo>` with your toolkit
clone and `<UTC-timestamp>` with the printed filename:

```powershell
conda activate zonal-stats-toolkit
python "<zonal_stats_toolkit_repo>\runner.py" ".\data\workflow_assets\zonal_stats\runs\snapp_assessment_tif_zonal_stats_<UTC-timestamp>.yaml" --job-workers 4 --raster-workers 2
```

The TIFF jobs write timestamped CSV and GeoPackage results under
`data/analysis_results/zonal_statistics/<zonal-product>`. After the run, create
the latest joined deliverables with:

```powershell
conda activate geo
python combine_final_zonal_stats_results.py
```

The resulting combined filenames identify the zonal product directly, including
`padus_public_access_lands_combined_*` and
`bbb_candidate_blm_lands_combined_*`.

#### Combined raster and vector configuration

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
python "<zonal_stats_toolkit_repo>\runner.py" .\data\workflow_assets\zonal_stats\snapp_assessment_zonal_stats.yaml
```

This configuration combines raster, mask, area, freshwater, and coastline
jobs. Its configured metrics are:

| Input family | Operations |
| --- | --- |
| Ecosystem service rasters | `sum`, `mean`, `area_ha_valid`, `proportion_valid_nonzero` |
| NLCD reclassification masks | `sum`, `valid_count`, `total_count`, `area_ha_valid`, `area_ha_total`, `min`, `max` |
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
`proportional_recreation_val_2024` into each available county-keyed final
output.

During this step, timestamped NLCD mask artifact fields such as
`area_ha_valid_reclassified_NLCD2023_*` are replaced with stable derived class
area fields:

- `area_ha_nlcd_forests`
- `area_ha_nlcd_grasslands`
- `area_ha_nlcd_shrubland`
- `area_ha_nlcd_water_snow`
- `area_ha_nlcd_wetlands`

The outputs also include one `area_ha_valid_nlcd` coverage field,
`area_ha_all_natural`, and five `proportion_nlcd_<class>_of_valid` fields.
Raw proportions and repeated coverage fields are omitted. Class area
is the number of **1-valued pixels times pixel hectares**, not valid area times
the toolkit's nonzero proportion (which includes NoData in its denominator).
The mask inputs must contain `sum`, `valid_count`, and `total_count`. If those
metrics are missing, run the mask-only command below; ecosystem-service zonals
remain reusable.

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

### Combined zonal results

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
- `padus_public_access_lands_combined_<timestamp>.csv` and
  `padus_public_access_lands_combined_<timestamp>.gpkg`, when those zonal jobs
  have been run.
- `blm_lands_excluding_federally_protected_areas_combined_<timestamp>.csv` and
  `.gpkg`, when that county-cut intermediate and its zonal jobs exist.
- `bbb_candidate_blm_lands_combined_<timestamp>.csv` and
  `bbb_candidate_blm_lands_combined_<timestamp>.gpkg`, when those zonal jobs
  have been run.

By default, these files are written to `data/analysis_results/combined`.
Each contains `proportional_recreation_val_2024` joined by `GEOID`.

## Build County Ecosystem-Service Valuations for the 2026 Assessment

[`build_county_ecosystem_service_valuations_2026.py`](build_county_ecosystem_service_valuations_2026.py)
builds a county dataset for comparing ecosystem-service values in the 2026
SNAPP assessment. It combines grazing, air quality, wages, timber, fisheries,
urban heat, recreation, mangrove coastal flood protection, and physical-health
estimates with the county zonal results selected in `sources.csv`. County
estimates are joined by FIPS; state fisheries, recreation polygons, and mangrove
study units are distributed to counties using the documented allocation rules.

Place the source files under `data/` as listed in the
[input and methods guide](docs/county_valuation_integration_2026.md#input-data).
Review the paths in `sources.csv` and `derived_metrics.yaml` under
`data/workflow_assets/county_integration/`. The latter selects the county-area,
NLCD mask-area, GDP and population inputs. If you need to calculate NLCD areas,
run the [mask-only step](#calculate-county-nlcd-areas-without-rerunning-ecosystem-service-zonals)
first. With the repository's Python environment active, run from the repository folder:

```powershell
python build_county_ecosystem_service_valuations_2026.py
```

The builder writes matching
`data/analysis_results/combined/snapp_ecosystem_services_counties_<timestamp>.csv`
and `.gpkg` files with matching county attributes. The GeoPackage also contains the
county map and tables describing sources and missing coverage. The original
files are preserved.

The county feature layer is named `tl_2024_us_county_50_states`, controlled by
`output.layer_name` in `export_rules.yaml`. The name is a label, not a geographic
filter: DC, Puerto Rico and the U.S. Virgin Islands are included. Use the actual
feature-layer name from each input file in `sources.csv`.

Both formats put original county identifiers and geographic attributes first,
including `ALAND` and `AWATER`, followed by all retained `sum_` fields, other
service totals, and adjustment-factor columns. Configured raster sums default to
their source names plus `_adj`, including
`sum_pollination_attributed_annual_crop_yield_value_adj` with
`pollination_adj_factor` of `0.09`. Retained source sums without a configured
adjustment keep their existing names and values. Mean, median, min, max, standard
deviation, percentiles and raw raster coverage diagnostics are omitted.
The post-calculation stage then appends the twelve NLCD area/share
fields and the county denominators, total and ratios described below.
There are no original-value copies or duplicate service-total aliases.

The tracked
[`export_rules.yaml`](data/workflow_assets/county_integration/export_rules.yaml)
controls export field names, exclusions, column groups, output directory, and
filename prefix. This is a YAML file, separate from the toolkit's
INI-style runner configuration. Source selection, multipliers, and Connecticut
allocation remain in `sources.csv`, `adjustment_factors.csv`, and
`ct_county_crosswalk.csv`. The rules rename the flood totals to
`sum_flood_annual_value_masked_to_wetlands_adj` and
`sum_flood_marginal_npv_masked_to_wetlands_adj`, rename pixel carbon to
`sum_CO2e_mitigation_dollar_value_per_pixel_adj`, and omit the four `sum_Wsw_*`
quantity fields. The similarly named `sum_Wval_sw_2020usd_*_adj` monetary
provisioning values remain.

Mangroves use `Ben_Stock_2020` from the `UCSC_CWON_studyunits` layer in
`data/analysis_inputs/ecosystem_services/mangroves/ucsc_cwon_studyunits.gpkg`.
Only `ISO3` values `USA`, `PRI`, and `VIR` are included; neighboring countries'
offshore study units cannot contribute merely by intersecting a U.S. county.
Each selected study-unit total is allocated in proportion to its positive-area county
intersections; ocean or other area outside all counties is excluded from the
denominator. A unit intersecting only one county contributes its entire value
there. Contributions are summed by county and multiplied by 1.182 to produce
`sum_mangrove_coastal_flood_protection_value_adj`, with `mangroves_adj_factor`
shown separately. Categorical `sum_mangrove_CONUS` fields are excluded from
the valuation export.
This denominator differs deliberately from recreation's full-polygon-area
denominator. See the [mangrove allocation method](docs/county_valuation_integration_2026.md#mangrove-study-unit-allocation)
for the source-units caveat, null handling, and unallocated study units.

Negative timber `total_annual_rent` is clamped to zero at the source-county
level, before Connecticut redistribution or aggregation. The source CSV is
unchanged; the output retains `timber_adj_factor` of 1 and records the clamp in
its audit tables. Missing timber values follow the documented
zero-filling convention.

The carbon multiplier is the assessment-supplied price of USD 3.66 per tonne CO2e
times the 1.182 inflation adjustment, or 4.32612. The pixel-carbon output is the
county sum of monetary per-pixel values. The separately retained carbon-per-hectare
sum is not an area-integrated county dollar total.

Export rules are validated before expensive processing. Each GeoPackage stores
their full YAML text, hash, and resolved field actions so its schema can be traced
to the configuration that produced it. Editing only export rules reuses completed
raw calculation checkpoints. See the
[export-rules guide](docs/county_valuation_integration_2026.md#export-rules) for
the supported keys.

Air quality exports only `air_quality_avoided_mortality_annual_value_adj`, the
county total across CSV land-cover/pollutant rows, and
`air_quality_avoided_mortality_annual_value_adj_factor` of `1.135`. See the guide
for units, Connecticut allocation, and null handling. To run county integration
with parallel workers:

```powershell
python build_county_ecosystem_service_valuations_2026.py --workers 16 --geometry-workers 16
```

Before comparing or summing values, read the guide's
[interpretation notes](docs/county_valuation_integration_2026.md#missing-values-zeroes-and-comparison-limits):
missing values differ from zero, some estimates are allocated across boundaries,
and the services include different units and time horizons.

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

#### Read the result

Each row is a county, identified by `GEOID`. The CSV and GeoPackage have the
same attributes; the GeoPackage also contains county boundaries.

| Column | What it contains |
| --- | --- |
| `es_total` | Sum of the 19 selected adjusted service values |
| `es_per_ha` | Total divided by the county's full area in hectares |
| `es_per_ha_natural` | Total divided by its observed natural-land hectares |
| `es_per_gdp` | Total divided by 2024 GDP in dollars |
| `es_per_capita` | Total divided by the 2024 population estimate |

Annual flood benefits are included in the total; flood NPV remains a separate
column. Blank service values contribute zero to the total. A blank ratio means
its denominator was missing or nonpositive. The denominator columns are kept
beside the summaries so you can inspect the calculation.

These summaries combine estimates with different time bases, including wages
and carbon-stock valuation. Use the
[results guide](docs/county_valuation_integration_2026.md#county-totals-and-ratios)
for their interpretation, denominator sources, and instructions for tracing a
particular output back to its inputs.

## Runtime Notes

The PAD-US, NHD, NLCD preparation, and zonal statistics steps are the expensive
parts of the workflow. On the local NVMe workstation used for this work, with 32
logical processors and 128 GB RAM, the full preparation and zonal statistics
workflow should be expected to take a few hours.

The final combination step is comparatively light and should take about 10 to 15
seconds.
