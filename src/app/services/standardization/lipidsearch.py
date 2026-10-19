"""
LipidSearch format: 5.0 and sample-grouped 5.2 exports (MeanArea[...]
columns), and dual-polarity 5.2 exports merged through the Alignment
Setting file (services/lipidsearch_alignment.py).
"""

import pandas as pd

from app.services.lipidsearch_alignment import (
    AlignmentError, parse_alignment_file, merge_dual_polarity,
)
from app.services.standardization.result import StandardizationResult


def process(df: pd.DataFrame) -> StandardizationResult:
    """Validate and standardize LipidSearch data."""
    if not isinstance(df, pd.DataFrame):
        return StandardizationResult(False, "Expected a DataFrame for LipidSearch format")

    # Validate
    required_cols = [
        'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
        'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return StandardizationResult(
            False, f"Missing required columns: {', '.join(missing)}"
        )

    meanarea_cols = [c for c in df.columns if c.startswith('MeanArea[')]
    if not meanarea_cols:
        return StandardizationResult(False, "No MeanArea columns found")

    # Standardize: rename MeanArea[x] -> intensity[x]
    result_df = df.copy()
    rename_dict = {}
    for col in meanarea_cols:
        sample = col.split('[', 1)[1].rstrip(']')
        rename_dict[col] = f'intensity[{sample}]'
    result_df = result_df.rename(columns=rename_dict)

    # Build column mapping (only renamed columns)
    mapping_rows = []
    for orig, std in rename_dict.items():
        mapping_rows.append({'standardized_name': std, 'original_name': orig})
    mapping_df = pd.DataFrame(mapping_rows) if mapping_rows else None

    return StandardizationResult(
        success=True,
        message="Data successfully standardized",
        standardized_df=result_df,
        column_mapping=mapping_df,
        n_intensity_cols=len(meanarea_cols),
    )


def standardize_with_alignment(
    df: pd.DataFrame, alignment_text: str,
) -> StandardizationResult:
    """Standardize a dual-polarity LipidSearch 5.2 export via its Alignment
    Setting file.

    Parses the alignment to pair each biological sample's positive/negative
    files, merges them into flat intensity[s1..sN] columns, and returns the
    experiment layout (conditions and per-condition sample counts) and the
    sample names read from the alignment so the UI can pre-populate them.
    """
    if not isinstance(df, pd.DataFrame):
        return StandardizationResult(False, "Expected a DataFrame for LipidSearch format")

    required_cols = [
        'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
        'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey',
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return StandardizationResult(
            False, f"Missing required columns: {', '.join(missing)}"
        )

    try:
        alignment = parse_alignment_file(alignment_text)
        merged = merge_dual_polarity(df, alignment)
    except AlignmentError as e:
        return StandardizationResult(False, str(e))

    # Column mapping: each merged intensity[sN] <- its per-file tokens.
    mapping_rows = [
        {'standardized_name': f'intensity[s{i}]',
         'original_name': ' + '.join(f'OriginalArea[{d}]' for d in s.dataids)}
        for i, s in enumerate(alignment.samples, start=1)
    ]
    mapping_df = pd.DataFrame(mapping_rows) if mapping_rows else None

    return StandardizationResult(
        success=True,
        message="Data successfully standardized",
        standardized_df=merged,
        column_mapping=mapping_df,
        n_intensity_cols=len(alignment.samples),
        lipidsearch_conditions=alignment.conditions,
        lipidsearch_samples_per_condition=alignment.samples_per_condition,
        lipidsearch_sample_names={
            f's{i}': s.name for i, s in enumerate(alignment.samples, start=1)
        },
    )
