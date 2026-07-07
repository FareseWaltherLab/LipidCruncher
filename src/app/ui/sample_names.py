"""
Sample display-name helpers.

Original sample names (e.g. the uploaded column headers like "mouse liver #5")
survive only in the ``column_mapping`` table. These helpers turn that mapping
into a ``{internal_label -> display_name}`` dict, keep it in sync when samples
are regrouped, and format labels for the sidebar sample selectors as
``"s3 — mouse liver #5"``.

Pure logic — no Streamlit dependencies.
"""

import re
from typing import Dict, Optional

import pandas as pd

# Matches a standardized intensity column, e.g. "intensity[s3]".
_INTENSITY_LABEL_RE = re.compile(r"^intensity\[(s\d+)\]$")
# Matches a wrapped header, e.g. "MeanArea[SampleA]" -> "SampleA".
_WRAPPED_HEADER_RE = re.compile(r"^\w+\[(.+)\]$")
# Matches a standardized intensity column in a regroup rename map key/value.
_INTENSITY_ANY_RE = re.compile(r"^intensity\[(s\d+)\]$")


def _clean_header(original_name: str) -> str:
    """Strip a ``Prefix[...]`` wrapper from a column header if present.

    ``MeanArea[SampleA]`` -> ``SampleA``; a plain header is returned unchanged.
    """
    text = str(original_name).strip()
    match = _WRAPPED_HEADER_RE.match(text)
    return match.group(1).strip() if match else text


def build_names_from_mapping(
    column_mapping: Optional[pd.DataFrame],
) -> Dict[str, str]:
    """Build a ``{s-label -> display name}`` map from a column-mapping table.

    Args:
        column_mapping: DataFrame with 'standardized_name' and 'original_name'
            columns (as stored in session state).

    Returns:
        Dict mapping internal labels (s1, s2, ...) to cleaned original names.
        Names identical to their label (no meaningful header) are omitted.
    """
    if column_mapping is None or column_mapping.empty:
        return {}
    if not {'standardized_name', 'original_name'}.issubset(column_mapping.columns):
        return {}

    names: Dict[str, str] = {}
    for std, orig in zip(
        column_mapping['standardized_name'], column_mapping['original_name']
    ):
        match = _INTENSITY_LABEL_RE.match(str(std))
        if not match:
            continue
        label = match.group(1)
        cleaned = _clean_header(orig)
        if cleaned and cleaned != label:
            names[label] = cleaned
    return names


def display_label(label: str, names: Optional[Dict[str, str]]) -> str:
    """Format a single sample label as ``"s3 — name"`` when a name exists."""
    if names:
        name = names.get(label)
        if name and name != label:
            return f"{label} — {name}"
    return label


def remap_names_after_regroup(
    names: Optional[Dict[str, str]],
    old_to_new: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """Remap names when manual regrouping permutes/renames the intensity columns.

    Args:
        names: Current ``{s-label -> name}`` map.
        old_to_new: Mapping of ``intensity[s_old] -> intensity[s_new]`` produced
            by ``SampleGroupingService.regroup_samples``.
    """
    if not names:
        return names
    remapped: Dict[str, str] = {}
    for old_col, new_col in old_to_new.items():
        old_match = _INTENSITY_ANY_RE.match(str(old_col))
        new_match = _INTENSITY_ANY_RE.match(str(new_col))
        if old_match and new_match:
            old_label = old_match.group(1)
            if old_label in names:
                remapped[new_match.group(1)] = names[old_label]
    return remapped
