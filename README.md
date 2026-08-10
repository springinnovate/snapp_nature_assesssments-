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
outputs to `bbb_candidate_blm_lands_by_county`. This change does not add these
products as jobs in the zonal-statistics configuration.

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
