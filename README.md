# SNAPP Nature Assessments

This repository supports a SNAPP project analyzing ecosystem services over
public lands in the United States.

## PAD-US Public Lands Preprocessing

The PAD-US geodatabase stores short coded values in fields, while GIS software
may display longer descriptions from coded-value domains. This matters for
scripts that read the geodatabase directly: filtering should use the stored
codes, not the displayed descriptions.

The preprocessing script uses:

- Source geodatabase: `data/PADUS4_1Geodatabase.gdb-20260513T025718Z-3-001/PADUS4_1Geodatabase.gdb`
- Source layer: `PADUS4_1Combined_Proclamation_Marine_Fee_Designation_Easement`
- USA boundary: `data/usa_vector.gpkg`
- Output layer name: `padus_public_lands_clipped_to_usa`
- Output path pattern: `output/padus_public_lands_clipped_to_usa_YYYY_MM_DD_HH_MM_SS.gpkg`

The output is a cleaned geometry-only GeoPackage layer. The existence of a
feature in this output means it passed the public-land filter and was clipped to
the USA boundary.

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

All other features are skipped.
