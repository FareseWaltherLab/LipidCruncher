"""
LipidSearch 5.2 dual-polarity alignment support.

LipidSearch 5.2 can acquire each biological sample in both positive and
negative ion mode and, when files are grouped by condition, exports one column
set per raw file — the per-file tokens ``OriginalArea[s{cond}-{file}]``. Each
lipid is detected in only one polarity, so a single per-file column is mostly
empty on its own; the two polarity files of one sample must be merged.

The **Alignment Setting file** (a LipidSearch export) maps every per-file token
to its raw filename and condition/group. The raw filename encodes both the
biological sample (a shared base, e.g. ``140509_FT_01``) and the polarity marker
(``...n``/``...p``). We pair files by that shared base — deterministic, and not
reliant on column adjacency (which the acquisition operator can reorder).

Pure logic — no Streamlit dependencies.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# Section header in the Alignment Setting file that precedes the file rows.
_TARGET_JOB_HEADER = '*Target search job'

# Per-file column token of a condition-grouped dual-polarity export, e.g.
# 'OriginalArea[s1-1]' — a sample index (s1) plus a within-condition file index.
_GROUPED_PER_FILE_RE = re.compile(r'^OriginalArea\[s\d+-\d+\]$')


def has_grouped_per_file_columns(df) -> bool:
    """True when a LipidSearch frame is the condition-grouped dual-polarity
    layout, i.e. it carries per-file OriginalArea[s{cond}-{file}] columns.
    Such a frame cannot be turned into per-sample intensities without the
    Alignment Setting file to pair its positive/negative files."""
    return any(_GROUPED_PER_FILE_RE.match(str(c)) for c in df.columns)

# Trailing polarity marker on a raw filename stem (case-insensitive). Longer
# tokens are tried first so 'neg'/'pos' win over a bare 'n'/'p'.
_POLARITY_SUFFIX_RE = re.compile(
    r'[ _-]?(negative|positive|neg|pos|n|p)$', re.IGNORECASE
)


class AlignmentError(ValueError):
    """Raised when the alignment file cannot be parsed or paired."""


@dataclass
class BiologicalSample:
    """One biological sample: the per-file tokens (a positive/negative pair)
    that share a raw-filename base."""
    base: str            # filename base shared by the pair, e.g. '140509_FT_01'
    condition: str       # e.g. 'Control'
    dataids: List[str]   # per-file tokens, e.g. ['s1-1', 's1-2']


@dataclass
class AlignmentMap:
    """Parsed Alignment Setting file, grouped into biological samples."""
    samples: List[BiologicalSample]   # ordered as encountered in the file
    conditions: List[str]             # unique condition labels, ordered
    samples_per_condition: List[int]  # count per condition, aligned to conditions


def _sample_base(filename: str) -> str:
    """Return the biological-sample base of a raw filename: drop the extension
    and the trailing polarity marker. e.g. '140509_FT_01n.raw' -> '140509_FT_01'."""
    stem = re.sub(r'\.[^.]+$', '', filename.strip())
    return _POLARITY_SUFFIX_RE.sub('', stem)


def parse_alignment_file(text: str) -> AlignmentMap:
    """Parse a LipidSearch Alignment Setting file into an AlignmentMap.

    The file is tab-delimited with a ``*Target search job`` section whose rows
    are: job name, raw filename, per-file token (s1-1, ...), condition/group,
    UUID. The condition may be prefixed with ``*`` (LipidSearch marks the
    reference group); the marker is stripped from the label.

    Files are grouped into biological samples by their filename base (the name
    with the polarity marker removed). Ordering follows first appearance.

    Raises:
        AlignmentError: if no job rows are found, or a sample's files do not
            pair cleanly (e.g. two files with the same polarity marker).
    """
    rows = []
    in_jobs = False
    for line in text.splitlines():
        if line.startswith(_TARGET_JOB_HEADER):
            in_jobs = True
            continue
        if in_jobs:
            # Section ends at a blank line or the next '*'-prefixed header.
            if not line.strip() or line.startswith('*'):
                break
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            filename, dataid, condition = parts[1].strip(), parts[2].strip(), parts[3]
            condition = condition.lstrip('*').strip()
            if filename and dataid and condition:
                rows.append((dataid, filename, condition))

    if not rows:
        raise AlignmentError(
            "No '*Target search job' rows found in the alignment file. "
            "Expected a LipidSearch Alignment Setting export."
        )

    # Group by (condition, base) preserving first-seen order.
    samples: List[BiologicalSample] = []
    index = {}
    for dataid, filename, condition in rows:
        key = (condition, _sample_base(filename))
        if key not in index:
            index[key] = BiologicalSample(base=key[1], condition=condition, dataids=[])
            samples.append(index[key])
        index[key].dataids.append(dataid)

    _validate_pairing(samples, rows)

    conditions: List[str] = []
    for s in samples:
        if s.condition not in conditions:
            conditions.append(s.condition)
    samples_per_condition = [
        sum(1 for s in samples if s.condition == c) for c in conditions
    ]

    return AlignmentMap(
        samples=samples,
        conditions=conditions,
        samples_per_condition=samples_per_condition,
    )


def _validate_pairing(samples: List[BiologicalSample], rows) -> None:
    """Guard against a filename convention that groups incorrectly.

    Every biological sample must have the same number of files, and within a
    multi-file sample the polarity markers must be distinct (so we never merge
    two files of the same polarity).
    """
    sizes = {len(s.dataids) for s in samples}
    if len(sizes) != 1:
        raise AlignmentError(
            "Alignment files did not pair evenly: biological samples have "
            f"differing file counts {sorted(sizes)}. Check the raw filenames "
            "follow a consistent positive/negative naming convention."
        )

    fname_by_dataid = {dataid: filename for dataid, filename, _ in rows}
    for s in samples:
        markers = []
        for d in s.dataids:
            stem = re.sub(r'\.[^.]+$', '', fname_by_dataid[d].strip())
            m = _POLARITY_SUFFIX_RE.search(stem)
            markers.append(m.group(1).lower() if m else '')
        if len(set(markers)) != len(markers):
            raise AlignmentError(
                f"Sample '{s.base}' ({s.condition}) has files with duplicate "
                f"polarity markers {markers}; cannot pair positive/negative."
            )


def merge_dual_polarity(
    df: pd.DataFrame,
    alignment: AlignmentMap,
    area_prefix: str = 'OriginalArea',
) -> pd.DataFrame:
    """Add merged per-sample intensity columns to a LipidSearch dataframe.

    For each biological sample, the per-file ``{area_prefix}[dataid]`` columns
    are summed (missing values treated as zero, but a row stays NaN when every
    file of that sample is missing the lipid). Because a lipid is detected in
    only one polarity, this sum equals coalescing the pair. Samples are
    renumbered flat ``intensity[s1..sN]`` in alignment order.

    Args:
        df: Raw LipidSearch dataframe (already delimiter-parsed).
        alignment: Parsed AlignmentMap.
        area_prefix: Per-file area column family to merge ('OriginalArea' = raw,
            the correct source when LipidSearch normalization is off).

    Returns:
        A copy of df with ``intensity[s1..sN]`` columns added.

    Raises:
        AlignmentError: if a referenced per-file column is absent from df.
    """
    result = df.copy()
    for i, sample in enumerate(alignment.samples, start=1):
        cols = [f'{area_prefix}[{d}]' for d in sample.dataids]
        missing = [c for c in cols if c not in result.columns]
        if missing:
            raise AlignmentError(
                "Alignment references columns not present in the data file: "
                f"{', '.join(missing)}. Do the LipidMol and alignment exports "
                "come from the same job?"
            )
        block = result[cols].apply(pd.to_numeric, errors='coerce')
        all_nan = block.isna().all(axis=1)
        merged = block.fillna(0).sum(axis=1)
        merged[all_nan] = np.nan
        result[f'intensity[s{i}]'] = merged
    return result
