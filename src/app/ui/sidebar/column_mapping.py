"""
Column Mapping UI Components for LipidCruncher sidebar.

This module contains:
- standardize_uploaded_data: Standardize uploaded data and create column mapping
- display_column_mapping: Display column mapping table with MS-DIAL sample override
"""

from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd

from app.constants import FORMAT_METABOLOMICS_WORKBENCH, FORMAT_MSDIAL, resolve_format_enum
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
    """Apply MS-DIAL sample override and update session state.

    Delegates pure logic to DataStandardizationService, then stores
    results in session state.

    Args:
        df: Current standardized DataFrame
        manual_samples: User-selected sample column names
        features: MS-DIAL features dict from session state

    Returns:
        New DataFrame with only selected intensity columns, renamed sequentially.
    """
    result = DataStandardizationService.apply_msdial_sample_override(
        df=df,
        column_mapping=st.session_state.column_mapping,
        manual_samples=manual_samples,
        features=features,
    )

    # Update session state with results
    st.session_state.msdial_features.raw_sample_columns = result.raw_sample_columns
    st.session_state.msdial_features.normalized_sample_columns = result.normalized_sample_columns
    st.session_state.n_intensity_cols = result.n_intensity_cols
    st.session_state.column_mapping = result.column_mapping
    st.session_state.msdial_sample_names = result.sample_names
    st.session_state['_msdial_override_samples'] = list(manual_samples)

    return result.standardized_df


# =============================================================================
# UI Components
# =============================================================================

def standardize_uploaded_data(df: pd.DataFrame, data_format: str) -> Optional[pd.DataFrame]:
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
    if data_format == FORMAT_METABOLOMICS_WORKBENCH:
        st.session_state.format_type = data_format
        return df

    format_enum = resolve_format_enum(data_format)
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


def display_column_mapping(df: pd.DataFrame, data_format: str) -> Tuple[bool, Optional[pd.DataFrame]]:
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
    if data_format == FORMAT_MSDIAL:
        with st.sidebar.expander("🔧 Override Sample Detection (Optional)", expanded=False):
            st.write("Only change this if auto-detection incorrectly classified columns.")

            features = st.session_state.get('msdial_features', None)
            detected_samples = getattr(features, 'raw_sample_columns', [])
            all_columns = getattr(features, 'actual_columns', [])

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
