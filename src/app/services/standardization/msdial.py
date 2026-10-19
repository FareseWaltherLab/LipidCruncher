"""
MS-DIAL format: locates the header row and the raw / pre-normalized sample
columns, standardizes them, and rebuilds the frame when the user overrides
which columns are samples.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.services import lipid_names
from app.services.format_detection import FormatDetectionService
from app.services.standardization.result import StandardizationResult


@dataclass
class MSDIALFeatures:
    """Typed feature flags and column lists for MS-DIAL data.

    Replaces the untyped ``Dict`` previously used, so that all field
    accesses are validated at definition time rather than at runtime.
    """
    has_ontology: bool = False
    has_quality_score: bool = False
    has_msms_matched: bool = False
    has_rt: bool = False
    has_mz: bool = False
    has_normalized_data: bool = False
    lipid_column: str = ''
    lipid_column_index: int = 0
    raw_sample_columns: List[str] = field(default_factory=list)
    raw_sample_indices: List[int] = field(default_factory=list)
    normalized_sample_columns: List[str] = field(default_factory=list)
    normalized_sample_indices: List[int] = field(default_factory=list)
    header_row_index: int = -1
    actual_columns: List[str] = field(default_factory=list)
    column_indices: Dict[str, int] = field(default_factory=dict)


@dataclass
class MSDIALStructure:
    """Detected structure of an MS-DIAL file.

    Attributes:
        header_row_idx: Row index where the actual header is found (-1 if none).
        actual_columns: List of column names from the detected header row.
        lipid_col: Name of the lipid name column (e.g., 'Metabolite name').
        lipid_col_idx: Index of the lipid column in actual_columns.
        col_indices: Mapping of column name to first-occurrence index.
        features: Typed MS-DIAL feature flags and column lists.
    """
    header_row_idx: int
    actual_columns: List[str]
    lipid_col: str
    lipid_col_idx: int
    col_indices: Dict[str, int]
    features: MSDIALFeatures


@dataclass
class MSDIALOverrideResult:
    """Result of applying MS-DIAL sample column override.

    Attributes:
        standardized_df: Rebuilt DataFrame with only selected intensity columns.
        column_mapping: Updated column mapping DataFrame.
        n_intensity_cols: Number of intensity columns after override.
        sample_names: Updated s1->original_name mapping.
        raw_sample_columns: Updated raw sample column list.
        normalized_sample_columns: Updated normalized sample column list.
    """
    standardized_df: pd.DataFrame
    column_mapping: pd.DataFrame
    n_intensity_cols: int
    sample_names: Dict[str, str]
    raw_sample_columns: List[str]
    normalized_sample_columns: List[str]


def apply_sample_override(
    df: pd.DataFrame,
    column_mapping: pd.DataFrame,
    manual_samples: List[str],
    features: "MSDIALFeatures",
) -> MSDIALOverrideResult:
    """Rebuild DataFrame and column mapping after manual MS-DIAL sample override.

    Pure logic — no session state access.

    Args:
        df: Current standardized DataFrame with intensity[sN] columns.
        column_mapping: Current column mapping DataFrame with
            'standardized_name' and 'original_name' columns.
        manual_samples: User-selected original sample column names.
        features: Typed MS-DIAL features (needs normalized_sample_columns).

    Returns:
        MSDIALOverrideResult with rebuilt DataFrame, mapping, and updated lists.
    """
    # Update feature lists
    raw_sample_columns = list(manual_samples)
    current_norm_samples = features.normalized_sample_columns
    normalized_sample_columns = [s for s in current_norm_samples if s in manual_samples]

    # Build reverse lookup: standardized_name -> original_name
    std_to_orig = dict(zip(
        column_mapping['standardized_name'],
        column_mapping['original_name'],
    ))

    # Identify which intensity columns to keep (by original name)
    intensity_cols_to_keep = [
        col for col in df.columns
        if col.startswith('intensity[') and std_to_orig.get(col) in manual_samples
    ]

    # Build new DataFrame with metadata + selected intensity columns
    non_intensity_cols = [col for col in df.columns if not col.startswith('intensity[')]
    new_df = df[non_intensity_cols + intensity_cols_to_keep].copy()

    # Build new column mapping (metadata rows + sequential intensity rows)
    new_mapping_rows = [
        {'standardized_name': row['standardized_name'], 'original_name': row['original_name']}
        for _, row in column_mapping.iterrows()
        if not row['standardized_name'].startswith('intensity[')
    ]

    rename_map = {}
    for i, old_col in enumerate(intensity_cols_to_keep, 1):
        new_col = f'intensity[s{i}]'
        rename_map[old_col] = new_col
        new_mapping_rows.append({
            'standardized_name': new_col,
            'original_name': std_to_orig.get(old_col, old_col),
        })

    new_df = new_df.rename(columns=rename_map)

    sample_names = {
        f's{i}': orig for i, orig in enumerate(manual_samples, 1)
    }

    return MSDIALOverrideResult(
        standardized_df=new_df,
        column_mapping=pd.DataFrame(new_mapping_rows),
        n_intensity_cols=len(intensity_cols_to_keep),
        sample_names=sample_names,
        raw_sample_columns=raw_sample_columns,
        normalized_sample_columns=normalized_sample_columns,
    )


def detect_sample_columns(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[List[str], List[str], Optional[int], List[int], List[int]]:
    """Detect sample intensity columns in MS-DIAL data.

    Returns:
        (raw_sample_cols, normalized_sample_cols, lipid_is_idx,
         raw_sample_indices, normalized_sample_indices)
    """
    if columns is None:
        columns = df.columns.tolist()

    metadata_cols = FormatDetectionService.MSDIAL_METADATA_COLUMNS

    raw_cols: List[str] = []
    norm_cols: List[str] = []
    raw_indices: List[int] = []
    norm_indices: List[int] = []
    lipid_is_idx = None

    for idx, col in enumerate(columns):
        if col == 'Lipid IS':
            lipid_is_idx = idx
            break

    for idx, col in enumerate(columns):
        if col in metadata_cols or col == 'Lipid IS':
            continue
        if pd.isna(col) or (isinstance(col, str) and not col.strip()):
            continue

        try:
            col_data = df.iloc[:, idx]
            numeric_values = pd.to_numeric(col_data, errors='coerce')
            numeric_count = numeric_values.notna().sum()

            if numeric_count > len(col_data) * 0.5:
                if lipid_is_idx is not None:
                    if idx < lipid_is_idx:
                        raw_cols.append(col)
                        raw_indices.append(idx)
                    elif idx > lipid_is_idx:
                        norm_cols.append(col)
                        norm_indices.append(idx)
                else:
                    raw_cols.append(col)
                    raw_indices.append(idx)
        except (ValueError, TypeError, IndexError):
            continue

    return raw_cols, norm_cols, lipid_is_idx, raw_indices, norm_indices


def detect_structure(
    df: pd.DataFrame,
) -> MSDIALStructure:
    """Detect MS-DIAL file structure: header row, columns, and features.

    Returns:
        MSDIALStructure with header info, column indices, and features.

    Raises:
        ValueError: If no lipid column or sample columns are found.
    """
    header_row_idx = FormatDetectionService._detect_msdial_header_row(df)

    if header_row_idx >= 0:
        actual_columns = df.iloc[header_row_idx].tolist()
    else:
        actual_columns = df.columns.tolist()

    # Find lipid name column
    lipid_col = None
    lipid_col_idx = None
    for col_name in ['Metabolite name', 'Metabolite', 'Name', 'Lipid', 'LipidMolec']:
        if col_name in actual_columns:
            lipid_col = col_name
            lipid_col_idx = actual_columns.index(col_name)
            break

    if lipid_col is None:
        raise ValueError(
            "No lipid name column found. Expected 'Metabolite name' column."
        )

    # Create temp df with correct headers for sample detection
    if header_row_idx >= 0:
        temp_df = df.iloc[header_row_idx + 1:].copy()
        temp_df.columns = actual_columns
    else:
        temp_df = df

    raw_samples, norm_samples, lipid_is_idx, raw_indices, norm_indices = (
        detect_sample_columns(
            temp_df, actual_columns
        )
    )

    if len(raw_samples) + len(norm_samples) < 1:
        raise ValueError(
            "No sample intensity columns detected. "
            "Check that your export includes sample data."
        )

    # Build column index mapping (first occurrence only)
    col_indices: Dict[str, int] = {}
    for idx, col in enumerate(actual_columns):
        if col not in col_indices:
            col_indices[col] = idx

    features = MSDIALFeatures(
        has_ontology='Ontology' in actual_columns,
        has_quality_score='Total score' in actual_columns,
        has_msms_matched='MS/MS matched' in actual_columns,
        has_rt='Average Rt(min)' in actual_columns,
        has_mz='Average Mz' in actual_columns,
        has_normalized_data=len(norm_samples) > 0,
        lipid_column=lipid_col,
        lipid_column_index=lipid_col_idx,
        raw_sample_columns=raw_samples,
        raw_sample_indices=raw_indices,
        normalized_sample_columns=norm_samples,
        normalized_sample_indices=norm_indices,
        header_row_index=header_row_idx,
        actual_columns=actual_columns,
        column_indices=col_indices,
    )

    return MSDIALStructure(
        header_row_idx=header_row_idx,
        actual_columns=actual_columns,
        lipid_col=lipid_col,
        lipid_col_idx=lipid_col_idx,
        col_indices=col_indices,
        features=features,
    )


def standardize_columns(
    df: pd.DataFrame,
    header_row_idx: int,
    lipid_col_idx: int,
    col_indices: Dict[str, int],
    features: "MSDIALFeatures",
    sample_cols: List[str],
    sample_indices: List[int],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Standardize MS-DIAL columns into the common schema.

    Returns:
        (result_df, column_mapping_dict) where column_mapping_dict maps
        original sample column names to intensity[sN] names.
    """
    if header_row_idx >= 0:
        data_df = df.iloc[header_row_idx + 1:].copy()
    else:
        data_df = df.copy()
    data_df = data_df.reset_index(drop=True)

    standardized_data: Dict[str, pd.Series] = {}

    # LipidMolec + ClassKey
    standardized_data['LipidMolec'] = data_df.iloc[:, lipid_col_idx].apply(
        lipid_names.standardize_lipid_name
    )
    standardized_data['ClassKey'] = standardized_data['LipidMolec'].apply(
        lipid_names.infer_class_key
    )

    # Optional metadata columns
    if features.has_rt and 'Average Rt(min)' in col_indices:
        standardized_data['BaseRt'] = pd.to_numeric(
            data_df.iloc[:, col_indices['Average Rt(min)']], errors='coerce'
        )
    if features.has_mz and 'Average Mz' in col_indices:
        standardized_data['CalcMass'] = pd.to_numeric(
            data_df.iloc[:, col_indices['Average Mz']], errors='coerce'
        )

    # Quality columns (preserved for filtering)
    if features.has_quality_score and 'Total score' in col_indices:
        standardized_data['Total score'] = pd.to_numeric(
            data_df.iloc[:, col_indices['Total score']], errors='coerce'
        )
    if features.has_msms_matched and 'MS/MS matched' in col_indices:
        standardized_data['MS/MS matched'] = data_df.iloc[
            :, col_indices['MS/MS matched']
        ]

    # Sample intensity columns
    column_mapping_dict: Dict[str, str] = {}
    for i, (col_name, col_idx) in enumerate(
        zip(sample_cols, sample_indices), 1
    ):
        new_col = f'intensity[s{i}]'
        standardized_data[new_col] = pd.to_numeric(
            data_df.iloc[:, col_idx], errors='coerce'
        ).fillna(0)
        column_mapping_dict[col_name] = new_col

    return pd.DataFrame(standardized_data), column_mapping_dict


def build_mapping(
    lipid_col: str,
    col_indices: Dict[str, int],
    features: "MSDIALFeatures",
    column_mapping_dict: Dict[str, str],
) -> pd.DataFrame:
    """Build the column mapping DataFrame for MS-DIAL."""
    mapping_rows = [
        {'original_name': lipid_col, 'standardized_name': 'LipidMolec'}
    ]
    if features.has_rt and 'Average Rt(min)' in col_indices:
        mapping_rows.append({
            'original_name': 'Average Rt(min)',
            'standardized_name': 'BaseRt',
        })
    if features.has_mz and 'Average Mz' in col_indices:
        mapping_rows.append({
            'original_name': 'Average Mz',
            'standardized_name': 'CalcMass',
        })
    for orig, std in column_mapping_dict.items():
        mapping_rows.append({
            'original_name': orig,
            'standardized_name': std,
        })
    return pd.DataFrame(mapping_rows)[['standardized_name', 'original_name']]


def process(
    df: pd.DataFrame, use_normalized: bool = False
) -> StandardizationResult:
    """Validate and standardize MS-DIAL data."""
    if not isinstance(df, pd.DataFrame):
        return StandardizationResult(False, "Expected a DataFrame for MS-DIAL format")

    try:
        structure = detect_structure(df)

        # Choose raw vs normalized sample columns
        norm_samples = structure.features.normalized_sample_columns
        if use_normalized and len(norm_samples) > 0:
            sample_cols = norm_samples
            sample_indices = structure.features.normalized_sample_indices
        else:
            sample_cols = structure.features.raw_sample_columns
            sample_indices = structure.features.raw_sample_indices

        result_df, column_mapping_dict = (
            standardize_columns(
                df, structure.header_row_idx, structure.lipid_col_idx,
                structure.col_indices, structure.features,
                sample_cols, sample_indices,
            )
        )

        mapping_df = build_mapping(
            structure.lipid_col, structure.col_indices,
            structure.features, column_mapping_dict,
        )

        sample_names = {
            f's{i}': col for i, col in enumerate(sample_cols, 1)
        }

        return StandardizationResult(
            success=True,
            message="Data successfully standardized",
            standardized_df=result_df,
            column_mapping=mapping_df,
            n_intensity_cols=len(sample_cols),
            msdial_features=structure.features,
            msdial_sample_names=sample_names,
        )

    except ValueError as e:
        return StandardizationResult(False, str(e))
    except KeyError as e:
        return StandardizationResult(
            False, f"MS-DIAL standardization error: {e}"
        )
