"""
LipidSearch 5.2 condition-grouped alignment support.

When LipidSearch 5.2 groups files by condition it exports one column set per
raw file — the per-file tokens ``OriginalArea[s{cond}-{file}]`` — rather than a
flat ``MeanArea[sN]`` per sample. Turning those into per-sample intensities
requires the **Alignment Setting file**, a LipidSearch export mapping every
per-file token to its raw filename and condition/group.

Three acquisition layouts occur in practice, and one run can mix them:

1. **Separate positive/negative files** — one biological sample acquired twice,
   the polarity marked in the filename (``..._01n.raw`` / ``..._01p.raw``).
   Each lipid is detected in only one polarity, so the pair must be merged.
2. **Polarity switching** — pos and neg are embedded in a *single* acquisition,
   so the sample has one per-file token and needs no merging. This is the
   common case for the study samples of a run.
3. **One raw file, several search jobs** — e.g. a pooled QC ("ID") run acquired
   in both modes separately, which LipidSearch stores under a single filename
   with one token per job. The jobs are complementary and must be merged.
   Such runs exist to build the peak annotations that are then applied to the
   rest of the sample set; users typically drop them before analysis.

Samples are therefore grouped by raw-filename base (the name with any polarity
marker removed), which handles all three: (1) groups by the shared base, (2) is
a group of one, (3) groups by the identical filename. Grouping by name rather
than by column adjacency is deterministic — the operator can reorder columns.

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

    The file has a ``*Target search job`` section whose rows are: job name, raw
    filename, per-file token (s1-1, ...), condition/group, UUID. It may be tab-
    or comma-delimited (LipidSearch varies by export); the delimiter is sniffed
    from the file. The condition may be prefixed with ``*`` (LipidSearch marks
    the reference group); the marker is stripped from the label.

    Files are grouped into biological samples by their filename base (the name
    with the polarity marker removed). Ordering follows first appearance.

    Raises:
        AlignmentError: if no job rows are found, or a sample's files cannot be
            grouped unambiguously (see ``_validate_pairing``).
    """
    # The file is uniformly tab- or comma-delimited; pick whichever character
    # dominates the whole file (the header rows carry no delimiter, so a
    # first-line sniff is unreliable). Default to comma on a tie.
    delimiter = '\t' if text.count('\t') > text.count(',') else ','
    rows = []
    in_jobs = False
    for line in text.splitlines():
        if line.startswith(_TARGET_JOB_HEADER):
            in_jobs = True
            continue
        if in_jobs:
            parts = line.split(delimiter)
            # Section ends at the next '*'-prefixed header or a blank line —
            # which, in a comma export, is padded with empty cells (",,,,").
            if line.startswith('*') or all(not p.strip() for p in parts):
                break
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
    """Guard against a filename convention that groups unrelated files.

    A biological sample may carry a single per-file token (single-polarity
    acquisition — most conditions in a run) or several. Sample counts need not
    be uniform across conditions: a run can mix single-polarity samples with
    dual-polarity ones.

    When a sample carries several tokens they must form an unambiguous group:
    either every token is the *same* raw filename (one raw file searched by
    multiple jobs, e.g. a positive and a negative product search that
    LipidSearch stores under one filename), or the filenames differ only by a
    distinct polarity marker (a positive/negative file pair). A sample whose
    files share a polarity marker but differ in name cannot be paired.
    """
    fname_by_dataid = {dataid: filename for dataid, filename, _ in rows}
    for s in samples:
        if len(s.dataids) < 2:
            continue
        fnames = [fname_by_dataid[d].strip() for d in s.dataids]
        if len(set(fnames)) == 1:
            continue  # one raw file, multiple search jobs — merge is unambiguous
        markers = []
        for fn in fnames:
            stem = re.sub(r'\.[^.]+$', '', fn)
            m = _POLARITY_SUFFIX_RE.search(stem)
            markers.append(m.group(1).lower() if m else '')
        if len(set(markers)) != len(markers):
            raise AlignmentError(
                f"Sample '{s.base}' ({s.condition}) groups files {fnames} that "
                f"cannot be paired: their polarity markers {markers} are not "
                "distinct. Check the raw filenames follow a consistent "
                "positive/negative naming convention."
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
