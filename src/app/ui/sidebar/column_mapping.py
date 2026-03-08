"""
Column Mapping UI Components for LipidCruncher sidebar.

This module contains:
- standardize_uploaded_data: Standardize uploaded data and create column mapping
- display_column_mapping: Display column mapping table with MS-DIAL sample override
"""

import streamlit as st
import pandas as pd

from app.constants import FORMAT_DISPLAY_TO_ENUM
from app.services.data_standardization import DataStandardizationService
from app.services.format_detection import DataFormat, FormatDetectionService


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_msdial_sample_override(
    df: pd.DataFrame,
    manual_samples: list,
    features: dict
) -> pd.DataFrame:
    """Rebuild DataFrame and column mapping after manual MS-DIAL sample override.

    Updates session state with new column mapping, sample counts, and sample names.

    Args:
        df: Current standardized DataFrame
        manual_samples: User-selected sample column names
        features: MS-DIAL features dict from session state

    Returns:
        New DataFrame with only selected intensity columns, renamed sequentially.
    """
    # Update feature lists for both raw and normalized sample columns
    st.session_state.msdial_features['raw_sample_columns'] = manual_samples

    current_norm_samples = features.get('normalized_sample_columns', [])
    filtered_norm_samples = [s for s in current_norm_samples if s in manual_samples]
    st.session_state.msdial_features['normalized_sample_columns'] = filtered_norm_samples

    # Build reverse lookup: standardized_name -> original_name
    current_mapping = st.session_state.column_mapping
    std_to_orig = dict(zip(
        current_mapping['standardized_name'],
        current_mapping['original_name']
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
        for _, row in current_mapping.iterrows()
        if not row['standardized_name'].startswith('intensity[')
    ]

    rename_map = {}
    for i, old_col in enumerate(intensity_cols_to_keep, 1):
        new_col = f'intensity[s{i}]'
        rename_map[old_col] = new_col
        new_mapping_rows.append({
            'standardized_name': new_col,
            'original_name': std_to_orig.get(old_col, old_col)
        })

    new_df = new_df.rename(columns=rename_map)

    # Update session state
    st.session_state.n_intensity_cols = len(intensity_cols_to_keep)
    st.session_state.column_mapping = pd.DataFrame(new_mapping_rows)
    st.session_state.msdial_sample_names = {
        f's{i}': orig for i, orig in enumerate(manual_samples, 1)
    }

    # Save override for preservation across re-standardization (e.g., data type changes)
    st.session_state['_msdial_override_samples'] = list(manual_samples)

    return new_df


# =============================================================================
# UI Components
# =============================================================================

def standardize_uploaded_data(df: pd.DataFrame, data_format: str) -> pd.DataFrame:
    """
    Standardize uploaded data and create column mapping.

    Args:
        df: Raw uploaded DataFrame
        data_format: The selected data format string

    Returns:
        The standardized DataFrame, or None if standardization failed
    """
    # Metabolomics Workbench is already standardized during file loading
    # (it requires raw text parsing, done in load_sample_dataset/display_file_upload)
    if data_format == 'Metabolomics Workbench':
        st.session_state.format_type = data_format
        return df

    format_enum = FORMAT_DISPLAY_TO_ENUM.get(data_format, DataFormat.GENERIC)
    use_normalized = st.session_state.get('msdial_data_type_index', 0) == 1

    result = DataStandardizationService.validate_and_standardize(
        df, format_enum, msdial_use_normalized=use_normalized
    )

    if not result.success:
        st.sidebar.error(result.message)
        return None

    st.session_state.format_type = data_format

    # Store standardization outputs in session state
    if result.column_mapping is not None:
        if st.session_state.get('column_mapping') is None:
            st.session_state.column_mapping = result.column_mapping
    if st.session_state.get('n_intensity_cols') is None:
        st.session_state.n_intensity_cols = result.n_intensity_cols

    # MS-DIAL specific state
    if result.msdial_features is not None:
        st.session_state.msdial_features = result.msdial_features
    if result.msdial_sample_names is not None:
        st.session_state.msdial_sample_names = result.msdial_sample_names

    return result.standardized_df


def display_column_mapping(df: pd.DataFrame, data_format: str) -> tuple:
    """
    Display column mapping in the sidebar.
    For MS-DIAL, includes override sample detection expander.

    Args:
        df: Standardized DataFrame
        data_format: The selected data format string

    Returns:
        tuple: (success: bool, modified_df: pd.DataFrame or None)
    """
    if st.session_state.get('column_mapping') is None:
        return True, None

    st.sidebar.subheader("Column Name Standardization")

    # Display mapping table
    mapping_df = st.session_state.column_mapping.copy()
    st.sidebar.dataframe(
        mapping_df.reset_index(drop=True),
        use_container_width=True
    )

    # MS-DIAL: Optional override for sample column detection
    if data_format == 'MS-DIAL':
        with st.sidebar.expander("🔧 Override Sample Detection (Optional)", expanded=False):
            st.write("Only change this if auto-detection incorrectly classified columns.")

            features = st.session_state.get('msdial_features', {})
            detected_samples = features.get('raw_sample_columns', [])
            all_columns = features.get('actual_columns', [])

            # Deduplicate: raw and normalized columns share same names in MS-DIAL
            available_for_samples = list(dict.fromkeys(
                col for col in all_columns
                if col not in FormatDetectionService.MSDIAL_METADATA_COLUMNS
            ))

            manual_samples = st.multiselect(
                "Sample columns:",
                options=available_for_samples,
                default=detected_samples,
                key='manual_sample_override',
                help="Select all columns containing sample intensity data"
            )

            if manual_samples and manual_samples != detected_samples:
                new_df = _apply_msdial_sample_override(df, manual_samples, features)

                st.success(f"✓ Using {len(manual_samples)} manually selected samples")
                st.write("**Updated column mapping:**")
                st.dataframe(
                    st.session_state.column_mapping.reset_index(drop=True),
                    use_container_width=True
                )

                return True, new_df

    return True, None
