"""Guesses for the AI Studio sidebar on a file of unknown format.

Which columns hold sample values, and which condition each sample belongs
to. Every guess is only a default the user reviews and edits.
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import pandas as pd

from app.services.format_detection import GENERIC_METADATA_NAME_PATTERNS

# 'Area[KO-1]': a measurement name with the sample in brackets.
_FAMILY = re.compile(r'^(?P<family>[^\[\]]+)\[(?P<sample>[^\[\]]+)\]$')

# A trailing replicate number, with an optional separator and 'rep' or
# 'replicate' before it: 'KO-1', 'KO_rep2', 'WT 3', 'WT3'.
_REPLICATE_SUFFIX = re.compile(
    r'(?:[\s_\-.]+(?:rep(?:licate)?)?[\s_\-.]*)?\d+$', re.IGNORECASE,
)


class SampleInferenceService:
    """Static guesses about sample columns and conditions."""

    @staticmethod
    def numeric_columns(df: pd.DataFrame) -> List[str]:
        """Columns holding at least one number; only these can be samples."""
        return [
            c for c in df.columns
            if pd.to_numeric(df[c], errors='coerce').notna().any()
        ]

    @staticmethod
    def column_families(columns: Sequence[str]) -> Dict[str, List[str]]:
        """Group 'Name[sample]' columns by Name.

        'Area[KO-1]' and 'Area[KO-2]' fall under 'Area'. A name with a single
        column is not a family and is left out.
        """
        families: Dict[str, List[str]] = {}
        for c in columns:
            match = _FAMILY.match(str(c))
            if match:
                families.setdefault(match['family'], []).append(c)
        return {name: cols for name, cols in families.items() if len(cols) > 1}

    @staticmethod
    def sample_name(column: str) -> str:
        """'Area[KO-1]' → 'KO-1'; any other column name is returned as is."""
        match = _FAMILY.match(str(column))
        return match['sample'] if match else str(column)

    @staticmethod
    def looks_like_metadata(name: str) -> bool:
        """Whether a column name looks like per-lipid metadata (RT, m/z,
        score, ...) rather than a sample.

        Uses the Generic detector's name patterns, matched on words so that
        'PrecursorMZ' and 'Notes' match but 'Heart_1' does not match 'rt'.
        A pattern of four or more letters also matches a longer word that
        starts with it ('note' → 'Notes').
        """
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(name)).lower()
        words = [w for w in re.split(r'[^a-z0-9/]+', spaced) if w]
        joined = f" {' '.join(words)} "
        for pattern in GENERIC_METADATA_NAME_PATTERNS:
            if ' ' in pattern:
                if f' {pattern} ' in joined:
                    return True
            elif any(
                w == pattern or (len(pattern) >= 4 and w.startswith(pattern))
                for w in words
            ):
                return True
        return False

    @staticmethod
    def guess_conditions(sample_names: Sequence[str]) -> Optional[List[str]]:
        """Guess each sample's condition by dropping a trailing replicate
        number: 'KO-1', 'KO_rep2', 'WT3' → 'KO', 'KO', 'WT'.

        Returns None when the names show no grouping: a name that is only a
        number, fewer than two conditions, or no condition with two samples.
        """
        guesses = [
            _REPLICATE_SUFFIX.sub('', str(name)).strip() for name in sample_names
        ]
        if not all(guesses):
            return None
        counts = Counter(guesses)
        if len(counts) < 2 or max(counts.values()) < 2:
            return None
        return guesses
