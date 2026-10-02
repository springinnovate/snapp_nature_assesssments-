# October 2026 county valuation integration

Issue: https://github.com/springinnovate/snapp_nature_assesssments-/issues/58

`integrate_county_valuations_2026.py` records the assessment-specific integration
of the September 10 county raster aggregates and the additional valuation
datasets supplied for the October 2026 assessment. It does not replace the
general zonal-statistics workflow or its PAD-US outputs.

## Run

Use the repository's geospatial Python environment, including `openpyxl` for
reading the two Excel workbooks. Stage the downloads using
`move_county_inputs.cmd` first; its `--dry-run` option previews the destinations.

```powershell
python integrate_county_valuations_2026.py --workers 8 --geometry-workers 8
```

Paths default to the directory containing the script, regardless of the current
working directory. Optional `--data-root`, `--config-dir`, and `--output` arguments
allow explicit alternatives. An existing output is never overwritten.

The output is `data/analysis_results/combined/counties_ecosystem_services_YYYYMMDD_HHMMSS.gpkg`.
The existing county layer name, 3,225 features, FIDs, geometry bytes, coordinate
system, spatial index, and retained source attributes are preserved. Puerto Rico
and the U.S. Virgin Islands stay in the layer even where a service has no source
coverage. The original GeoPackage is never edited.

## Parallel execution, progress, and recovery

Ten independent service groups run in separate processes. Small related
operations, such as the four jobs categories and their total, stay together to
avoid repeatedly reading the same workbook. Recreation intersections use a
separate thread pool over batches of 2,000 polygons; Shapely performs the geometry
work outside Python's GIL. The worker settings are independent. The defaults are
up to eight processes and eight geometry threads, and either can be reduced to
one on a memory-constrained machine.

`tqdm` reports source hashing, completed service groups, recreation batches,
output fields, county writes, and audit-table writes. Geometry loading and
reprojection precede the recreation batch bar and can take noticeable time.

Completed services are stored as SQLite checkpoints under
`data/processing_outputs/county_integration_2026/`. Recreation also checkpoints
individual batches as NumPy arrays. Rerunning the same command reuses completed
work. Checkpoint identities incorporate relevant source SHA-256 hashes, the
Connecticut crosswalk, script contents, and core library versions. Different
inputs or code therefore cannot reuse stale results. Factors are applied after
the cached original aggregates, so changing only a multiplier does not require
repeating geometry work.

A failed service does not discard other completed services. The program lets
other submitted services finish and checkpoint, reports the failure, and exits
without publishing an output. A forcefully terminated run can reuse all batches
and services whose checkpoints were fully published. Temporary checkpoint files
are ignored. `--no-resume` recomputes everything; it is also useful if a checkpoint
has been corrupted.

The final database is built and validated in a temporary directory on the output
filesystem. Only the completed database is published, with a hard link that
refuses to overwrite an existing filename. This requires an output filesystem
with hard-link support, such as the local NTFS disk. A failure while writing the
GeoPackage leaves the service checkpoints available for a rerun.

## Sources and configuration

The exact filenames and paths relative to `data/` are in
`data/workflow_assets/county_integration/sources.csv`. The factors and their
interpretations are in `adjustment_factors.csv` in the same directory. These
small CSVs and the derived Connecticut crosswalk are tracked by Git; data,
checkpoints, and output GeoPackages are ignored.

| Source | Destination within `data/` |
| --- | --- |
| Original county ecosystem services; county coastline lengths; county freshwater CSV | `analysis_results/zonal_statistics/counties/` |
| Grazing workbook | `analysis_inputs/ecosystem_services/grazing/` |
| Raw jobs workbook | `analysis_inputs/ecosystem_services/jobs/` |
| Timber county rents | `analysis_inputs/ecosystem_services/timber/` |
| Air-quality county/land/pollutant CSV | `analysis_inputs/ecosystem_services/air_quality/` |
| Marine and inland fisheries state CSVs | `analysis_inputs/ecosystem_services/marine_fisheries/` and `inland_fisheries/` |
| Urban heat CSV | `analysis_inputs/ecosystem_services/urban_heat/` |
| Physical health GeoPackage | `analysis_inputs/ecosystem_services/physical_health/` |
| Recreation polygons | `analysis_inputs/recreation/` |
| Mangrove study units, retained for reference only | `analysis_inputs/ecosystem_services/mangroves/` |

The recreation source retains its actual downloaded filename
`usa_nature_assessment_recreation__snapp_estimates.gpkg`. The older standalone
recreation preparation script has a different default filename; this integration
reads its own explicit source manifest and does not invoke that older script.

## Service definitions

Each integrated service has `<service>_orig`, `<service>_adj_factor`, and
`<service>_adj`. The factor is always stored, including 1.0 and counties whose
original value is missing. The calculation is `adjusted = original * factor`.
An original value here means the county aggregate before adjustment; source
polygon and state totals are retained in allocation audit tables where relevant.

| Service | Original county value | Factor |
| --- | --- | --- |
| `grazing` | `CountyGrazingValueYear`, `SNAPPGrazingCalculations` sheet, `CountyCode` join | 1 |
| `avoided_air_quality_heath_costs` | Sum `dollars_annual` across land and pollutant rows by `FIPS` | 1.135 |
| `jobs_cultural`, `jobs_raw_materials`, `jobs_processing`, `jobs_harvesting` | Four raw Total Wages columns in `totals by county`, second-row header | 1.065 |
| `jobs` | Sum the four wage categories | 1.065 |
| `timber` | `total_annual_rent`, joined by `fips_padded` | 1 |
| `marine_fisheries` | State `ncp_value` allocated by county coastline share within that state | 1.027 |
| `inland_fisheries` | State `Consumptive use value (USD)_Intermediate assumption` allocated by freshwater-area share | 1.065 |
| `urban_heat` | Sum `Sum of total_dollars_saved` by `COUNTY_GEOID` | 1.135 |
| `physical_health` | `total_value_usd`, joined by `GEOID` | 1.027 |
| `recreation` | Sum `val_2024 * county_intersection_area / full_source_polygon_area` | 1 |
| `carbon_ha`, `carbon_pixel` | Existing `sum_totalC_tCO2e_ha_2020` and `sum_totalC_tCO2e_pixel_2020` | 4.32612 each |
| `dredging`, `sdwa`, `water_treatment` | Corresponding existing avoided-cost sums | 1 |
| `provisioning` | Existing `sum_Wval_sw_2020usd_total` | 1.182 |
| `provisioning_irr`, `provisioning_pow`, `provisioning_pub` | Existing water-value component sums | 1.182 |
| `flood_npv`, `flood_annual` | Existing marginal NPV and annual wetland sums | 1 |
| `corals` | Existing `sum_Coral_Reefs_2024adj_CPI` | 1 |
| `mangroves` | Existing `sum_mangrove_CONUS` | 1.182 |
| `mental_health` | Existing mental-health sum | 1 |

Decisions confirmed by the requester:

- Preserve the spelling `avoided_air_quality_heath_costs`.
- Both carbon sums receive `3.66 * 1.182 = 4.32612`. Their names already include
  `tCO2e`; preserve that ambiguity in metadata. No additional dollar-per-tonne
  social cost was specified, so the script does not claim these are USD values.
  The per-hectare raster sum is retained as supplied, not reconstructed into a
  physical county total.
- Corals already contain the 2024 adjustment and therefore receive 1.0.
- Reuse existing mangrove values, including their negative signs, without using
  the study-unit polygons or silently changing the sign.
- Timber rent is already a county total; do not multiply it by acreage again.
- Keep both the jobs components and their total. State, national, metro, and
  unallocated summary rows are excluded from the county wage joins.
- Keep water provisioning. Remove exactly
  `sum_national_attributed_annual_crop_yield_value_zstd` and
  `sum_pollination_attributed_annual_crop_yield_value` from the output. Their
  non-sum diagnostic fields remain unchanged.

Other original fields, including `Wsw` quantities and the two unidentified
`CONUS_Coastal_2021_CCAP` bands, are retained unchanged. They are not assigned an
invented valuation factor. No grand total combines services, stocks, flows,
carbon quantities, NPV, or the jobs/provisioning components with their totals.

## Geography and missing values

FIPS codes are normalized to five-character strings; leading zeroes are restored.
Duplicate county totals are rejected. Air-quality detail and heat values are
summed explicitly. State totals and blank footer rows are excluded from inland
fisheries before allocation. The heat CSV's 12 `$ -` cells use accounting-format
zero; other nonempty nonnumeric values cause an error.

Timber missing rows and missing values become zero, per the requester. Coverage
statuses preserve `no_timberland`, `no_species_coverage`, and missing-source
distinctions. Other missing sources and null source values remain null. A county
with zero coastline/freshwater weight receives zero only when its state has a
supplied value. Missing states, including Puerto Rico in the fisheries sources,
remain unknown. A supplied state with no positive allocation support is an error.

### Connecticut

The base uses nine planning regions; several sources use the eight historical
counties. The checked-in `ct_county_crosswalk.csv` has 19 positive-area links,
derived from the Census Bureau's
[2022 county subdivision to 2020 block group relationship file](https://www2.census.gov/geo/docs/maps-data/data/rel2022/acs22_cousub22_blkgrp20_st09.txt).
The [record-layout documentation](https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/2022-connecticut-record-layout.html)
defines its geography identifiers and intersection-area fields.

Raw file location: `data/analysis_inputs/boundaries/county_crosswalks/acs22_cousub22_blkgrp20_st09.txt`.
SHA-256: `06957a5e777dc96da4c678a79d27dce6617cfbd510c9611dbf7720c3cd16c5bd`.
The raw download is not required at runtime because the small derived crosswalk
is versioned alongside the script.

Derivation: take the first five characters of `GEOID_BLKGRP_20` as the historical
county, and of `GEOID_COUSUB_22` as the planning region. Sum `AREALAND_PART` by
that pair and divide by the sum over each historical county. Drop zero-area
links. Weights sum to one per old county, preserving known source totals.
These are land-area-based estimates, not population- or service-specific
measurements. If a target region depends on any absent/null historical county,
its value stays null and known partial contributions are recorded separately.
Timber alone follows its explicit missing-as-zero policy. Sources mixing old
counties and new regions are rejected to avoid double counting.

Other unmatched historical codes are listed with their values in
`integration_source_issues`; the program does not guess a correspondence or
silently count the same source more than once.

### Recreation

Intersections use global equal-area EPSG:6933 for the 50 states and territories.
Only working geometries are repaired/reprojected. Allocation uses the full
source polygon area as denominator. It does not inflate covered portions to
absorb value outside the county footprint. Each source polygon has a coverage
fraction and unallocated value in the output audit. Fractions above
`1 + 1e-6` cause an error rather than double-counting overlapping counties.

## Audit and validation

The output contains these ordinary GeoPackage attribute tables:

- `integration_services`: source fields, methods, units, factors, totals,
  negative-value counts, and populated/missing county counts.
- `integration_sources`: exact source filenames, sizes, and SHA-256 hashes,
  including the configuration files.
- `integration_county_status`: per-service county coverage and estimation status.
- `integration_source_issues`: unmatched county values, excluded jobs summary
  identifiers, and known partial contributions withheld from incomplete totals.
- `integration_state_checks`: input and allocated totals for every supplied
  fisheries state with a value.
- `integration_recreation_allocation`: every source polygon's input value,
  allocated fraction, allocated value, and uncovered value.
- `integration_ct_crosswalk`: the exact crosswalk used in the run.
- `integration_checkpoints`: service-group checkpoint signatures.
- `integration_run`: execution time, script checksum, CRS, and interpretation notes.

The script reconciles county joins and state allocations, checks recreation
coverage, verifies that sources did not change during the run, and reads back
the output valuation columns. It compares every retained original attribute and
geometry byte against the source, then runs SQLite integrity and foreign-key
checks before publishing.

```powershell
python -m unittest test_integrate_county_valuations_2026 -v
```

Tests cover FIPS and currency parsing, split-county conservation, incomplete
crosswalk coverage, missing/zero distinctions, state totals, polygon splits and
uncovered value, overlapping counties, checkpoint signatures, recovery after a
failed geometry batch, and GeoPackage preservation/overwrite protection.

## Validation on the staged assessment data, October 1, 2026

The full run using eight service processes and eight geometry threads completed
in approximately 47 seconds on the local workstation. A full checkpoint resume
took approximately six seconds and produced identical county attributes and
geometry. The complete repository test suite passed (17 tests). A simulated
failed recreation batch resumed successfully, and a failed database write did
not publish a partial output or alter the source.

The validated output contains 3,225 counties, 27 service definitions, 81 new
valuation fields, and nine audit tables. All 71 supplied fisheries state totals
reconcile (23 marine and 48 inland; maximum absolute residual below $0.000001).
All 325,791 recreation polygons are audited. Their source value totals
$881,801,006,000; $881,637,401,423.39 was allocated, with $163,604,576.61 outside
the covered county footprint, net of sub-tolerance geometry overlap.

Data qualifications retained in this run:

- All 142 nonzero existing mangrove aggregates are negative. The integration
  preserves these signs and applies 1.182 as instructed.
- Timber has 350 negative county values after joining. Four unmatched historical
  Alaska codes contain a net -$7,590,984.93. Two other unmatched codes have null
  source values. These records remain in the issue table rather than being
  assigned a guessed geography.
- Connecticut air quality, grazing, and timber values were allocated to all
  nine planning regions. Jobs already used the new regions. Physical-health
  source values did not support a complete value for any of the nine regions;
  heat supported six, with three left null and $10,950,652.94 of known partial
  heat contributions retained in the issue table.
- Physical health has 1,095 populated counties and urban heat has 1,187.
  The four jobs categories each have 3,216 populated counties. The coverage
  table distinguishes missing sources from measured zeroes.

These qualifications are source coverage/sign issues, not failed conservation
checks. Use the service and county-status tables when selecting counties or
comparing totals.
