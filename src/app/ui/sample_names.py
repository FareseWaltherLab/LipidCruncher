"""
Sample display-name helpers.

Original sample names (e.g. the uploaded column headers like "mouse liver #5")
survive only in the ``column_mapping`` table (LipidSearch 5.2 alignment uploads
take theirs from the Alignment Setting file instead). These helpers turn that mapping
into a ``{internal_label -> display_name}`` dict, format labels for the
sidebar sample selectors as ``"s3 — mouse liver #5"``, and put the names into
CSV downloads. ``SampleFrame.renumbered`` keeps the map in step when samples
are regrouped or excluded.

Pure logic — no Streamlit dependencies.
"""

import re
from collections import Counter
from typing import Dict, Optional

import pandas as pd

# Matches a standardized intensity column, e.g. "intensity[s3]".
_INTENSITY_LABEL_RE = re.compile(r"^intensity\[(s\d+)\]$")
# Matches a single wrapped header, e.g. "MeanArea[SampleA]" -> "SampleA". The
# content excludes ']' so a composite header (a merged dual-polarity token
# expression like "OriginalArea[s1-1] + OriginalArea[s1-2]") does not match and
# is left intact rather than mangled by a greedy capture.
_WRAPPED_HEADER_RE = re.compile(r"^\w+\[([^\]]+)\]$")
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


def names_from_editor(edited: pd.DataFrame) -> Dict[str, str]:
    """Collect ``{s-label -> name}`` from the sidebar name editor's rows.

    A cleared cell comes back as None (or NaN), which means no name rather
    than the text "None"; blank names are dropped so the label is kept.
    """
    return {
        row['sample']: str(row['name']).strip()
        for _, row in edited.iterrows()
        if not pd.isna(row['name']) and str(row['name']).strip()
    }


def sample_display_map(names: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Map each named label to the text shown for it on screen and in exports.

    The same text a CSV header uses, so a plot, a table and a download never
    disagree; unnamed labels are absent (show the label itself).
    """
    return _csv_display_names(names or {})


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
