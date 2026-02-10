"""
Column Mapping UI Components for LipidCruncher sidebar.

This module contains:
- standardize_uploaded_data: Standardize uploaded data and create column mapping
- display_column_mapping: Display column mapping table with MS-DIAL sample override
"""

import streamlit as st
import pandas as pd

from lipidomics.data_format_handler import DataFormatHandler


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

    format_map = {
        'LipidSearch 5.0': 'lipidsearch',
        'MS-DIAL': 'msdial',
        'Generic Format': 'generic',
    }
    internal_format = format_map.get(data_format, 'generic')

    # Use DataFormatHandler to validate and standardize
    standardized_df, success, message = DataFormatHandler.validate_and_preprocess(
        df, internal_format
    )

    if not success:
        st.sidebar.error(message)
        return None

    st.session_state.format_type = data_format
    return standardized_df


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

            # Exclude known metadata columns
            available_for_samples = [
                col for col in all_columns
                if col not in DataFormatHandler.MSDIAL_METADATA_COLUMNS
            ]

            manual_samples = st.multiselect(
                "Sample columns:",
                options=available_for_samples,
                default=detected_samples,
                key='manual_sample_override',
                help="Select all columns containing sample intensity data"
            )

            if manual_samples and manual_samples != detected_samples:
                # Update the feature list for both raw and normalized sample columns
                # (MS-DIAL exports have same sample names before and after 'Lipid IS' column)
                st.session_state.msdial_features['raw_sample_columns'] = manual_samples

                # Filter normalized columns to match the manual selection
                current_norm_samples = features.get('normalized_sample_columns', [])
                filtered_norm_samples = [s for s in current_norm_samples if s in manual_samples]
                st.session_state.msdial_features['normalized_sample_columns'] = filtered_norm_samples

                # Get current DataFrame and rebuild intensity columns
                current_df = df.copy()
                current_mapping = st.session_state.column_mapping

                # Build reverse lookup: standardized_name -> original_name
                std_to_orig = dict(zip(
                    current_mapping['standardized_name'],
                    current_mapping['original_name']
                ))

                # Identify which intensity columns to keep (by original name)
                intensity_cols_to_keep = []
                for col in current_df.columns:
                    if col.startswith('intensity['):
                        orig_name = std_to_orig.get(col)
                        if orig_name in manual_samples:
                            intensity_cols_to_keep.append(col)

                # Get non-intensity columns (metadata)
                non_intensity_cols = [col for col in current_df.columns if not col.startswith('intensity[')]

                # Build new DataFrame with only selected intensity columns
                new_df = current_df[non_intensity_cols + intensity_cols_to_keep].copy()

                # Rename intensity columns to be sequential
                rename_map = {}
                new_mapping_rows = []

                # Keep metadata mappings
                for _, row in current_mapping.iterrows():
                    if not row['standardized_name'].startswith('intensity['):
                        new_mapping_rows.append({
                            'standardized_name': row['standardized_name'],
                            'original_name': row['original_name']
                        })

                # Create new sequential intensity column names
                for i, old_col in enumerate(intensity_cols_to_keep, 1):
                    new_col = f'intensity[s{i}]'
                    rename_map[old_col] = new_col
                    orig_name = std_to_orig.get(old_col, old_col)
                    new_mapping_rows.append({
                        'standardized_name': new_col,
                        'original_name': orig_name
                    })

                new_df = new_df.rename(columns=rename_map)

                # Update session state
                st.session_state.n_intensity_cols = len(intensity_cols_to_keep)
                st.session_state.column_mapping = pd.DataFrame(new_mapping_rows)

                # Update sample name mapping
                st.session_state.msdial_sample_names = {
                    f's{i}': orig for i, orig in enumerate(manual_samples, 1)
                }

                st.success(f"✓ Using {len(manual_samples)} manually selected samples")

                # Display updated mapping table
                st.write("**Updated column mapping:**")
                st.dataframe(
                    st.session_state.column_mapping.reset_index(drop=True),
                    use_container_width=True
                )

                return True, new_df

    return True, None
