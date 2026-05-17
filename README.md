# SNAPP Nature Assessments

This repository supports a SNAPP project analyzing ecosystem services over
public lands in the United States.

## PAD-US Lands Preprocessing

The PAD-US geodatabase stores short coded values in fields, while GIS software
may display longer descriptions from coded-value domains. This matters for
scripts that read the geodatabase directly: filtering should use the stored
codes, not the displayed descriptions.

The preprocessing script, `prepare_padus_lands_clipped_to_usa.py`, uses:

- Source geodatabase: `data/PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/PADUS4_1Geodatabase.gdb`
- Source layer: `PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement`
- USA boundary: `data/usa_vector.gpkg`

It processes PAD-US geometries once, clips them to the USA boundary, and writes
two cleaned GeoPackage outputs:

- `output/padus_all_lands_clipped_to_usa_YYYY_MM_DD_HH_MM_SS.gpkg`
- `output/padus_public_lands_clipped_to_usa_YYYY_MM_DD_HH_MM_SS.gpkg`

Both outputs strip source PAD-US fields and keep only `land_type` plus geometry.
The all-lands output writes `land_type = all`. The public-lands output writes
`land_type = public`.

## Public-Land Rules

PAD-US `Mang_Type` has the display alias `Manager Type`. Keep a feature when
`Mang_Type` is one of these stored values:

| Display description | Stored value |
| --- | --- |
| Federal | `FED` |
| State | `STAT` |
| Local Government | `LOC` |
| Regional Agency Special District | `DIST` |
| Joint | `JNT` |
| Territorial | `TERR` |

If `Mang_Type` is `UNK` (`Unknown`), keep the feature only when `Own_Type`
(`Owner Type`) is one of these stored values:

| Display description | Stored value |
| --- | --- |
| Local Government | `LOC` |
| Regional Agency Special District | `DIST` |
| Federal | `FED` |
| Joint | `JNT` |
| State | `STAT` |

All other features are excluded from the public-lands output, but still appear
in the all-lands output when their processed geometry has positive-area overlap
with the USA boundary.

## County Flattening

The `cut_and_flatten_by_county.py` script cuts either clipped lands product by
county and flattens each county to one multipolygon feature. It copies all
non-geometry county fields and adds the input `land_type` value to each output
feature.

Output names are derived from the input product:

- `padus_public_lands_clipped_to_usa_...gpkg` becomes `padus_public_lands_clipped_by_county_YYYY_MM_DD_HH_MM_SS.gpkg`
- `padus_all_lands_clipped_to_usa_...gpkg` becomes `padus_all_lands_clipped_by_county_YYYY_MM_DD_HH_MM_SS.gpkg`
