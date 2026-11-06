"""Convert binary NLCD mask statistics into coverage, class areas and shares.

The toolkit's ``proportion_valid_nonzero`` divides by ALL sampled pixels,
including NoData. Multiplying that by valid area is therefore incorrect.
Instead, a binary mask's sum counts its 1-valued pixels. Zero-valued pixels
contribute to coverage but not to class area; NoData contributes to neither.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from tqdm import tqdm


CLASSES = ("forests", "grasslands", "shrubland", "water_snow", "wetlands")
OPERATIONS = ("sum", "valid_count", "total_count", "area_ha_valid", "area_ha_total", "min", "max")
RAW_FIELD = re.compile(
    r"^(?P<operation>[a-z_]+)_reclassified_NLCD2023_"
    r"(?P<class>forests|grasslands|shrubland|water_snow|wetlands)"
    r"(?:_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})?$"
)
FIELD_DEFINITIONS = {
    "area_ha_valid_nlcd": "Shared valid NLCD coverage in hectares; includes both 0 and 1, excludes NoData.",
    **{f"area_ha_nlcd_{name}": "Hectares of 1-valued pixels in the " + name + " mask." for name in CLASSES},
    "area_ha_all_natural": "Sum of the five disjoint NLCD class areas, including water/snow; not total county area.",
    **{f"proportion_nlcd_{name}_of_valid": "Class area / valid NLCD coverage; null when coverage is zero." for name in CLASSES},
}


def derive_nlcd_mask_areas(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace raw mask metrics with explicitly named hectares and proportions.

    Args:
        frame: Zonal attributes, optionally joined with other metric families.
            All five binary masks must have the operations in ``OPERATIONS``.

    Returns:
        Copy preserving non-mask fields and adding the twelve defined fields.
        If no raw mask fields exist, returns a copy unchanged.

    Raises:
        ValueError: Statistics are incomplete, nonbinary, inconsistent across
            masks, or imply more natural area than valid coverage. Historical
            valid-area/proportion-only results require a mask-only rerun.
    """
    matched = {column: RAW_FIELD.fullmatch(column) for column in frame.columns}
    matched = {column: match for column, match in matched.items() if match}
    if not matched:
        return frame.copy()
    columns = {}
    for column, match in matched.items():
        key = (match["class"], match["operation"])
        if key in columns:
            raise ValueError(f"Multiple versions of NLCD mask statistic {key}; select one run.")
        columns[key] = column
    missing = [(name, op) for name in CLASSES for op in OPERATIONS if (name, op) not in columns]
    if missing:
        raise ValueError(f"Incomplete NLCD mask statistics: {missing}. Rerun masks only with "
                         "snapp_assessment_nlcd_masks.yaml; old valid area times proportion is not class area.")
    derived = pd.DataFrame(index=frame.index)
    shared = None
    for name in tqdm(CLASSES, desc="Calculate NLCD class areas", unit="class"):
        values = {op: pd.to_numeric(frame[columns[name, op]], errors="raise").astype(float)
                  for op in OPERATIONS}
        valid, total = values["valid_count"], values["total_count"]
        for op in ("valid_count", "total_count", "area_ha_valid", "area_ha_total"):
            if not np.isfinite(values[op]).all() or (values[op] < 0).any():
                raise ValueError(f"Invalid {name} {op}; missing coverage is not zero coverage.")
        if ((valid > total).any() or not np.array_equal(valid, np.round(valid))
                or not np.array_equal(total, np.round(total))):
            raise ValueError(f"Invalid NLCD pixel counts for {name}.")
        pixel_area = values["area_ha_total"].div(total.where(total > 0))
        sampled = total > 0
        if ((pixel_area[sampled] <= 0).any()
                or not np.allclose(pixel_area[sampled], pixel_area[sampled].iloc[0] if sampled.any() else 0)
                or not np.allclose(values["area_ha_valid"], valid * pixel_area.fillna(0), rtol=1e-9, atol=1e-8)
                or (values["area_ha_total"][~sampled] != 0).any()):
            raise ValueError(f"Inconsistent NLCD pixel area for {name}.")
        coverage = pd.DataFrame({op: values[op] for op in
                                 ("valid_count", "total_count", "area_ha_valid", "area_ha_total")})
        if shared is not None and not np.allclose(coverage, shared, rtol=1e-9, atol=1e-8):
            raise ValueError("NLCD masks must share grid and county coverage; do not mix runs.")
        shared = coverage
        selected = values["sum"]
        # An empty zonal sum may be null; a nonempty zone must never be silently filled.
        selected = selected.mask((valid == 0) & selected.isna(), 0)
        if (not np.isfinite(selected).all() or (selected < 0).any() or (selected > valid).any()
                or not np.array_equal(selected, np.round(selected))):
            raise ValueError(f"Invalid binary NLCD sum for {name}.")
        occupied = valid > 0
        expected_min = (selected[occupied] == valid[occupied]).astype(float)
        expected_max = (selected[occupied] > 0).astype(float)
        if (not np.array_equal(values["min"][occupied], expected_min)
                or not np.array_equal(values["max"][occupied], expected_max)):
            raise ValueError(f"NLCD {name} statistics are not from a 0/1 binary mask.")
        derived[f"area_ha_nlcd_{name}"] = selected * pixel_area.fillna(0)
        derived[f"proportion_nlcd_{name}_of_valid"] = selected.div(valid.where(valid > 0))
    derived["area_ha_valid_nlcd"] = shared["area_ha_valid"]
    derived["area_ha_all_natural"] = derived[[f"area_ha_nlcd_{name}" for name in CLASSES]].sum(axis=1)
    if (derived.area_ha_all_natural > derived.area_ha_valid_nlcd + 1e-7).any():
        raise ValueError("Natural class areas exceed valid coverage; check mask overlap or mixed runs.")
    # Replace only this explicit schema and recognized raw mask metrics.
    result = frame.drop(columns=list(matched) + [c for c in FIELD_DEFINITIONS if c in frame]).copy()
    for field in FIELD_DEFINITIONS:
        result[field] = derived[field]
    return result
