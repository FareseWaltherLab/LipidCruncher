"""
Shared fixtures and wrapper functions for UI tests.

Wrapper functions are used with AppTest.from_function() to test UI components
without importing main_app.py (which has st.set_page_config() at module level).
"""

import pytest
from streamlit.testing.v1 import AppTest


DEFAULT_TIMEOUT = 15


# =============================================================================
# Wrapper Functions
# =============================================================================

def landing_page_script():
    """Render landing page only."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    import app.ui.landing_page as lp
    lp.PDF2IMAGE_AVAILABLE = False  # Avoid PDF conversion timeouts
    from app.ui.landing_page import display_landing_page
    display_landing_page()


def format_and_upload_script():
    """Format selectbox + file upload + standardization."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.sidebar.file_upload import display_format_selection, display_file_upload
    from app.ui.sidebar.column_mapping import standardize_uploaded_data
    data_format = display_format_selection()
    raw_df = display_file_upload(data_format)
    if raw_df is not None:
        if st.session_state.get('standardized_df') is None:
            std_df = standardize_uploaded_data(raw_df, data_format)
            st.session_state.standardized_df = std_df
        st.text(f"data_loaded:{raw_df.shape[0]}x{raw_df.shape[1]}")
    else:
        st.text("no_data")


def full_sidebar_script():
    """Format + upload + experiment + grouping + BQC + confirm."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.sidebar.file_upload import display_format_selection, display_file_upload
    from app.ui.sidebar.column_mapping import standardize_uploaded_data, display_column_mapping
    from app.ui.sidebar.sample_grouping import display_sample_grouping
    data_format = display_format_selection()
    raw_df = display_file_upload(data_format)
    if raw_df is not None:
        if st.session_state.get('standardized_df') is None:
            std_df = standardize_uploaded_data(raw_df, data_format)
            if std_df is not None:
                st.session_state.standardized_df = std_df
        std_df = st.session_state.get('standardized_df')
        if std_df is not None:
            mapping_valid, modified_df = display_column_mapping(std_df, data_format)
            if mapping_valid:
                if modified_df is not None:
                    std_df = modified_df
                    st.session_state.standardized_df = std_df
                experiment, bqc_label = display_sample_grouping(std_df, data_format)
                if experiment is not None:
                    st.text(f"confirmed:{experiment.n_conditions}c")
                else:
                    st.text("not_confirmed")
    else:
        st.text("no_data")


def app_page_script():
    """Full app page without set_page_config (for Back to Home tests)."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    import app.ui.landing_page as lp
    lp.PDF2IMAGE_AVAILABLE = False
    from app.ui.landing_page import display_logo
    from app.ui.format_requirements import display_format_requirements
    from app.ui.sidebar import display_format_selection, display_file_upload
    _, center, _ = st.columns([1, 3, 1])
    data_format = display_format_selection()
    with center:
        display_logo(centered=True)
        display_format_requirements(data_format)
    raw_df = display_file_upload(data_format)
    if raw_df is None:
        with center:
            st.info("Upload a dataset or load sample data to begin.")
            if st.button("← Back to Home"):
                st.session_state.page = 'landing'
                st.rerun()


def msdial_data_type_script():
    """MS-DIAL data type radio only."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import display_msdial_data_type_selection
    display_msdial_data_type_selection()
    use_norm = st.session_state.get('msdial_use_normalized', False)
    st.text(f"use_normalized:{use_norm}")


def override_preservation_script():
    """Data type radio + override tracking (Group 7, test 2)."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import display_msdial_data_type_selection
    display_msdial_data_type_selection()
    override = st.session_state.get('_msdial_override_samples')
    st.text(f"override:{override}")
    st.text(f"std_df_cleared:{st.session_state.get('standardized_df') is None}")


def override_reset_script():
    """Reset button + override tracking (Group 7, test 3)."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    override = st.session_state.get('_msdial_override_samples')
    st.text(f"override:{override}")
    if st.button("Reset", key='reset_btn'):
        StreamlitAdapter.reset_data_state()
        st.rerun()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def landing_app():
    """Landing page rendered and ready for assertions."""
    return AppTest.from_function(landing_page_script, default_timeout=DEFAULT_TIMEOUT).run()


@pytest.fixture
def format_upload_app():
    """Format selectbox + file upload rendered (no data loaded)."""
    return AppTest.from_function(format_and_upload_script, default_timeout=DEFAULT_TIMEOUT).run()


@pytest.fixture
def full_sidebar_app():
    """Full sidebar rendered (no data loaded)."""
    return AppTest.from_function(full_sidebar_script, default_timeout=DEFAULT_TIMEOUT).run()


@pytest.fixture
def generic_sidebar_app(full_sidebar_app):
    """Generic data loaded + 3x4=12 experiment config matching 12-sample dataset."""
    at = full_sidebar_app
    at.sidebar.button(key='load_sample').click().run()
    # Set 3 conditions x 4 samples = 12 (matches Generic dataset)
    at.sidebar.number_input[0].set_value(3).run()
    at.number_input(key='n_samples_0').set_value(4).run()
    at.number_input(key='n_samples_1').set_value(4).run()
    at.number_input(key='n_samples_2').set_value(4).run()
    return at


@pytest.fixture
def msdial_features_dict():
    """MS-DIAL feature detection results for pre-populating session state."""
    return {
        'has_normalized_data': True,
        'raw_sample_columns': ['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
        'normalized_sample_columns': ['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
        'has_quality_score': True,
        'has_msms_matched': False,
    }


@pytest.fixture
def msdial_data_type_app(msdial_features_dict):
    """MS-DIAL data type selection with pre-populated features."""
    at = AppTest.from_function(msdial_data_type_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['msdial_features'] = msdial_features_dict
    return at.run()
