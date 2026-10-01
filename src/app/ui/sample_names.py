"""
Sample display-name helpers.

Original sample names (e.g. the uploaded column headers like "mouse liver #5")
survive only in the ``column_mapping`` table (LipidSearch 5.2 alignment uploads
take theirs from the Alignment Setting file instead). These helpers turn that mapping
into a ``{internal_label -> display_name}`` dict, keep it in sync when samples
are regrouped or excluded, format labels for the sidebar sample selectors as
``"s3 — mouse liver #5"``, and put the names into CSV downloads.

Pure logic — no Streamlit dependencies.
"""

import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

# Matches a standardized intensity column, e.g. "intensity[s3]".
_INTENSITY_LABEL_RE = re.compile(r"^intensity\[(s\d+)\]$")
# Matches a single wrapped header, e.g. "MeanArea[SampleA]" -> "SampleA". The
# content excludes ']' so a composite header (a merged dual-polarity token
# expression like "OriginalArea[s1-1] + OriginalArea[s1-2]") does not match and
# is left intact rather than mangled by a greedy capture.
_WRAPPED_HEADER_RE = re.compile(r"^\w+\[([^\]]+)\]$")
# Matches a standardized intensity column in a regroup rename map key/value.
_INTENSITY_ANY_RE = re.compile(r"^intensity\[(s\d+)\]$")
# Matches any per-sample column, e.g. "concentration[s3]" -> ("concentration", "s3").
_SAMPLE_COLUMN_RE = re.compile(r"^(\w+)\[(s\d+)\]$")
# Matches a bare internal sample label, e.g. "s3".
_BARE_LABEL_RE = re.compile(r"^s\d+$")


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


def remap_names_after_exclusion(
    names: Optional[Dict[str, str]],
    labels_before: List[str],
    removed: List[str],
    labels_after: List[str],
) -> Dict[str, str]:
    """Re-key names into the label space left by Quality Check's sample exclusion.

    Excluding samples renumbers the survivors (remove s2 and the old s3 becomes
    s2), so every frame produced after the exclusion needs the names re-keyed
    the same way. Mirrors ``QualityCheckService._drop_and_rename_columns``: the
    survivors, in their original order, take ``labels_after`` in turn.

    Returns the CSV display text, disambiguated in the original label space
    first, so a shared name keeps its original label (``"QC (s5)"``) in every
    CSV rather than picking up the survivor's new one.

    Args:
        names: ``{s-label -> name}`` in the pre-exclusion label space.
        labels_before: Sample labels before the exclusion, in order.
        removed: Labels that were excluded.
        labels_after: Sample labels after the exclusion, in order.
    """
    if not names:
        return {}
    names = _csv_display_names(names)
    survivors = [label for label in labels_before if label not in removed]
    return {
        new: names[old]
        for old, new in zip(survivors, labels_after)
        if old in names
    }


def _csv_display_names(names: Dict[str, str]) -> Dict[str, str]:
    """Map each named label to the text written for it in a CSV.

    Blank names and names equal to their own label are dropped (the label is
    kept). A name shared by several samples, or equal to another sample's
    label, gets its label appended (``"QC (s5)"``) so no two samples can end
    up with the same header.
    """
    cleaned = {}
    for label, name in names.items():
        text = str(name).strip() if name is not None else ''
        if text and text != label:
            cleaned[label] = text
    counts = Counter(cleaned.values())
    return {
        label: (
            text if counts[text] == 1 and not _BARE_LABEL_RE.match(text)
            else f"{text} ({label})"
        )
        for label, text in cleaned.items()
    }


def name_samples_for_csv(
    df: pd.DataFrame,
    names: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """Replace internal sample labels with display names for a CSV download.

    ``concentration[s1]`` becomes ``concentration[ID_01]`` (any ``prefix[sN]``
    column keeps its prefix), a bare ``s1`` column header becomes ``ID_01``, and
    ``s1`` values in a ``Sample`` column become ``ID_01``. Unnamed samples keep
    their label. ``names`` must be keyed in the same label space as ``df``.

    Returns:
        A renamed copy, or ``df`` itself when there is nothing to rename.
    """
    display = _csv_display_names(names or {})
    if not display:
        return df

    def _header(column):
        match = _SAMPLE_COLUMN_RE.match(str(column))
        if match and match.group(2) in display:
            return f"{match.group(1)}[{display[match.group(2)]}]"
        return display.get(column, column)

    renamed = [_header(column) for column in df.columns]
    # A name that clashes with another column's header keeps its label rather
    # than producing a duplicate header.
    counts = Counter(renamed)
    renamed = [
        new if counts[new] == 1 else old
        for old, new in zip(df.columns, renamed)
    ]

    out = df.copy()
    out.columns = renamed
    if 'Sample' in out.columns:
        out['Sample'] = out['Sample'].map(lambda value: display.get(value, value))
    return out
