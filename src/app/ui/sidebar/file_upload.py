"""
File Upload UI Components for LipidCruncher sidebar.

This module contains:
- display_format_selection: Format dropdown selector
- load_sample_dataset: Load sample data for the selected format
- display_file_upload: File uploader with sample data option
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.data_standardization import DataStandardizationService
from app.services.format_detection import DataFormat
from app.ui.content import get_sample_data_info


# =============================================================================
# Path Configuration
# =============================================================================

# Get the sample datasets directory relative to this file
# This file is at: src/app/ui/sidebar/file_upload.py
# Sample datasets are at: sample_datasets/
_SIDEBAR_DIR = Path(__file__).parent
_UI_DIR = _SIDEBAR_DIR.parent
_APP_DIR = _UI_DIR.parent
_SRC_DIR = _APP_DIR.parent
SAMPLE_DATA_DIR = _SRC_DIR.parent / "sample_datasets"


# =============================================================================
# Helper Functions
# =============================================================================

def _store_workbench_result(result) -> pd.DataFrame:
    """Store Metabolomics Workbench standardization result in session state.

    Args:
        result: StandardizationResult from DataStandardizationService.

    Returns:
        Standardized DataFrame, or None if standardization failed.
    """
    if result.success:
        st.session_state.standardized_df = result.standardized_df
        if result.workbench_conditions:
            st.session_state.workbench_conditions = result.workbench_conditions
        if result.workbench_samples:
            st.session_state.workbench_samples = result.workbench_samples
        return result.standardized_df
    else:
        st.sidebar.error(result.message)
        return None


# =============================================================================
# UI Components
# =============================================================================

def display_format_selection() -> str:
    """Display format selection dropdown in sidebar."""
    return st.sidebar.selectbox(
        'Select Data Format',
        ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0', 'MS-DIAL']
    )


def load_sample_dataset(data_format: str) -> pd.DataFrame:
    """Load a sample dataset for the selected format.

    Args:
        data_format: The selected data format string

    Returns:
        DataFrame with loaded sample data, or None if loading failed
    """
    info = get_sample_data_info(data_format)
    if info:
        filepath = SAMPLE_DATA_DIR / info['file']
        if filepath.exists():
            st.session_state.sample_data_file = info['file']

            if data_format == 'Metabolomics Workbench':
                # Metabolomics Workbench needs raw text for parsing special markers
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                result = DataStandardizationService.validate_and_standardize(
                    text_content, DataFormat.METABOLOMICS_WORKBENCH
                )
                return _store_workbench_result(result)
            else:
                return pd.read_csv(filepath)
    return None


def display_file_upload(data_format: str) -> pd.DataFrame:
    """Display file upload widget and sample data option.

    Args:
        data_format: The selected data format string

    Returns:
        DataFrame with uploaded/loaded data, or None if no data
    """
    # Sample data option
    with st.sidebar.expander("🧪 Try Sample Data", expanded=False):
        info = get_sample_data_info(data_format)
        if info:
            st.markdown(f"**{data_format} Example:**")
            st.markdown(info['description'])
            st.markdown("---")
        if st.button("Load Sample Data", key="load_sample"):
            sample_df = load_sample_dataset(data_format)
            if sample_df is not None:
                st.session_state.using_sample_data = True
                st.session_state.raw_df = sample_df
                st.rerun()

    # Check if using sample data
    if st.session_state.get('using_sample_data') and st.session_state.get('raw_df') is not None:
        sample_file = st.session_state.get('sample_data_file', 'sample data')
        st.sidebar.info(f"📁 Using sample: {sample_file}")
        if st.sidebar.button("Clear & Upload Your Data"):
            st.session_state.using_sample_data = False
            st.session_state.sample_data_file = None
            StreamlitAdapter.reset_data_state()
            st.rerun()
        return st.session_state.raw_df

    # File upload
    file_types = ['csv'] if data_format == 'Metabolomics Workbench' else ['csv', 'txt']
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset',
        type=file_types,
        help="Limit 800MB per file"
    )

    if uploaded_file is not None:
        try:
            if data_format == 'Metabolomics Workbench':
                # Metabolomics Workbench needs raw text for parsing special markers
                text_content = uploaded_file.getvalue().decode('utf-8')
                result = DataStandardizationService.validate_and_standardize(
                    text_content, DataFormat.METABOLOMICS_WORKBENCH
                )
                df = _store_workbench_result(result)
                if df is not None:
                    st.sidebar.success("File uploaded and processed successfully!")
                    st.session_state.raw_df = df
                return df
            else:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
                st.session_state.raw_df = df
                return df
        except (ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None

    return None
