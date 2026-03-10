"""
Shared fixtures and wrapper functions for UI tests.

Wrapper functions are used with AppTest.from_function() to test UI components
without importing main_app.py (which has st.set_page_config() at module level).
"""

import pytest
from streamlit.testing.v1 import AppTest

# Pre-load app modules to avoid circular import when AppTest runs scripts.
# Without this, the first AppTest.from_function().run() can trigger a circular
# import between app.constants and app.services.data_cleaning.base.
# Import format_detection first (no dependencies on constants), then constants.
import app.services.format_detection  # noqa: F401
import app.constants  # noqa: F401


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


# =============================================================================
# Module 2: Quality Check — Wrapper Functions
# =============================================================================

def qc_module_script():
    """Run QC module with test data from session state.

    Expects session state keys:
        _test_df: DataFrame with concentration[s] columns
        _test_experiment: ExperimentConfig
        _test_bqc_label: Optional BQC label string
        _test_format_type: Format string (default 'Generic Format')
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.quality_check import display_quality_check_module

    df = st.session_state['_test_df']
    experiment = st.session_state['_test_experiment']
    bqc_label = st.session_state.get('_test_bqc_label')
    format_type = st.session_state.get('_test_format_type', 'Generic Format')

    qc_df, updated_exp = display_quality_check_module(
        continuation_df=df,
        experiment=experiment,
        bqc_label=bqc_label,
        format_type=format_type,
    )
    st.text(f"qc_rows:{qc_df.shape[0]}")
    st.text(f"qc_samples:{len(updated_exp.full_samples_list)}")


# =============================================================================
# Module 2: Quality Check — Data Builders
# =============================================================================

def make_qc_dataframe(n_lipids=20, n_samples=6, with_rt=False):
    """Build a QC-ready DataFrame with concentration columns.

    Args:
        n_lipids: Number of lipid rows.
        n_samples: Number of sample columns (concentration[s1]..concentration[sN]).
        with_rt: Whether to include BaseRt and CalcMass columns (for RT tests).

    Returns:
        DataFrame with LipidMolec, ClassKey, and concentration columns.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    half = n_lipids // 2
    classes = ['PC'] * half + ['PE'] * (n_lipids - half)
    lipids = [f'{cls}({i}:0)' for i, cls in enumerate(classes)]

    data = {
        'LipidMolec': lipids,
        'ClassKey': classes,
    }
    if with_rt:
        data['BaseRt'] = np.random.uniform(1, 30, n_lipids).tolist()
        data['CalcMass'] = np.random.uniform(500, 1000, n_lipids).tolist()

    for i in range(1, n_samples + 1):
        data[f'concentration[s{i}]'] = np.random.uniform(500, 5000, n_lipids).tolist()

    return pd.DataFrame(data)


def make_qc_bqc_dataframe(n_lipids=20, high_cov_count=3):
    """Build a QC DataFrame with BQC condition and some high-CoV lipids.

    Experiment layout: Control(s1-s3), Treatment(s4-s6), BQC(s7-s8).
    First `high_cov_count` lipids have extreme values in BQC samples
    to guarantee CoV > 30%.

    Returns:
        DataFrame with 8 sample columns.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    half = n_lipids // 2
    classes = ['PC'] * half + ['PE'] * (n_lipids - half)
    lipids = [f'{cls}({i}:0)' for i, cls in enumerate(classes)]

    data = {
        'LipidMolec': lipids,
        'ClassKey': classes,
    }
    for i in range(1, 9):
        data[f'concentration[s{i}]'] = np.random.uniform(500, 5000, n_lipids).tolist()

    df = pd.DataFrame(data)

    # Make first `high_cov_count` lipids have very high CoV in BQC samples (s7, s8)
    for j in range(high_cov_count):
        df.loc[j, 'concentration[s7]'] = 10.0
        df.loc[j, 'concentration[s8]'] = 50000.0

    return df


# =============================================================================
# Module 2: Quality Check — Fixtures
# =============================================================================

# =============================================================================
# Module Navigation — Wrapper Functions
# =============================================================================

def module1_nav_script():
    """Module 1 with navigation buttons (Next + Back to Home).

    Renders navigation buttons that appear when normalized_df is in session state.
    Replicates main_app.py lines 214-226 navigation logic.

    Expects session state keys:
        normalized_df: DataFrame (optional — controls Next button visibility)
        module: str (default 'Data Cleaning, Filtering, & Normalization')
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.constants import COV_THRESHOLD_DEFAULT
    StreamlitAdapter.initialize_session_state()

    def _reset_qc_state():
        st.session_state.qc_continuation_df = None
        st.session_state.qc_bqc_plot = None
        st.session_state.qc_cov_threshold = COV_THRESHOLD_DEFAULT
        st.session_state.qc_correlation_plots = {}
        st.session_state.qc_pca_plot = None
        st.session_state.qc_samples_removed = []

    current_module = st.session_state.get('module', 'Data Cleaning, Filtering, & Normalization')
    st.text(f"module:{current_module}")

    normalized_df = st.session_state.get('normalized_df')
    if normalized_df is not None:
        st.text(f"normalized_rows:{normalized_df.shape[0]}")
        if st.button("Next: Quality Check & Analysis →", key='next_module'):
            _reset_qc_state()
            st.session_state.module = "Quality Check & Analysis"
            st.rerun()

    if st.button("← Back to Home", key='back_home_m1'):
        st.session_state.page = 'landing'
        StreamlitAdapter.reset_data_state()
        st.rerun()


def module2_nav_script():
    """Module 2 with navigation buttons (Back to Data Processing + Back to Home).

    Replicates main_app.py lines 236-266 navigation logic (without QC rendering).

    Expects session state keys:
        normalized_df: DataFrame (required — Module 2 entry gate)
        module: str = 'Quality Check & Analysis'
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.constants import COV_THRESHOLD_DEFAULT
    StreamlitAdapter.initialize_session_state()

    def _reset_qc_state():
        st.session_state.qc_continuation_df = None
        st.session_state.qc_bqc_plot = None
        st.session_state.qc_cov_threshold = COV_THRESHOLD_DEFAULT
        st.session_state.qc_correlation_plots = {}
        st.session_state.qc_pca_plot = None
        st.session_state.qc_samples_removed = []

    continuation_df = st.session_state.get('normalized_df')
    if continuation_df is None:
        st.error("No normalized data available. Please complete Module 1 first.")
        if st.button("← Back to Data Processing", key='back_m1_error'):
            st.session_state.module = "Data Cleaning, Filtering, & Normalization"
            st.rerun()
        return

    current_module = st.session_state.get('module', 'unknown')
    st.text(f"module:{current_module}")
    st.text(f"normalized_rows:{continuation_df.shape[0]}")

    # Only store QC result when actually in Module 2 (not after navigating away)
    if current_module == 'Quality Check & Analysis':
        st.session_state.qc_continuation_df = continuation_df

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data Processing", key='back_m1'):
            _reset_qc_state()
            st.session_state.module = "Data Cleaning, Filtering, & Normalization"
            st.rerun()
    with col2:
        if st.button("← Back to Home", key='back_home_m2'):
            st.session_state.page = 'landing'
            StreamlitAdapter.reset_data_state()
            st.rerun()


# =============================================================================
# Module Navigation — Fixtures
# =============================================================================

@pytest.fixture
def module1_nav_app():
    """Module 1 navigation with normalized_df pre-populated."""
    import numpy as np
    import pandas as pd

    at = AppTest.from_function(module1_nav_script, default_timeout=DEFAULT_TIMEOUT)
    # Minimal normalized DataFrame
    df = pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:1)'],
        'ClassKey': ['PC', 'PE'],
        'concentration[s1]': [100.0, 200.0],
        'concentration[s2]': [150.0, 250.0],
    })
    at.session_state['normalized_df'] = df
    return at.run()


@pytest.fixture
def module2_nav_app():
    """Module 2 navigation with normalized_df and QC state pre-populated."""
    import numpy as np
    import pandas as pd

    at = AppTest.from_function(module2_nav_script, default_timeout=DEFAULT_TIMEOUT)
    df = pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:1)'],
        'ClassKey': ['PC', 'PE'],
        'concentration[s1]': [100.0, 200.0],
        'concentration[s2]': [150.0, 250.0],
    })
    at.session_state['normalized_df'] = df
    at.session_state['module'] = 'Quality Check & Analysis'
    # Pre-populate QC state to verify it gets cleared
    at.session_state['qc_continuation_df'] = df.copy()
    at.session_state['qc_bqc_plot'] = 'fake_plot'
    at.session_state['qc_cov_threshold'] = 50
    at.session_state['qc_correlation_plots'] = {'Control': 'fake_corr'}
    at.session_state['qc_pca_plot'] = 'fake_pca'
    at.session_state['qc_samples_removed'] = ['s1']
    return at.run()


@pytest.fixture
def qc_generic_app():
    """QC module: Generic format, no BQC, 2x3=6 samples, 20 lipids."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(qc_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_qc_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_format_type'] = 'Generic Format'
    return at.run()


@pytest.fixture
def qc_bqc_app():
    """QC module: Generic format, with BQC, Control(3)+Treatment(3)+BQC(2)=8."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(qc_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_qc_bqc_dataframe(n_lipids=20, high_cov_count=3)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=3,
        conditions_list=['Control', 'Treatment', 'BQC'],
        number_of_samples_list=[3, 3, 2],
    )
    at.session_state['_test_bqc_label'] = 'BQC'
    at.session_state['_test_format_type'] = 'Generic Format'
    return at.run()


@pytest.fixture
def qc_lipidsearch_app():
    """QC module: LipidSearch format, no BQC, 2x3=6 samples, with RT columns."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(qc_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_qc_dataframe(
        n_lipids=20, n_samples=6, with_rt=True
    )
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_format_type'] = 'LipidSearch 5.0'
    return at.run()


@pytest.fixture
def qc_small_app():
    """QC module: Generic format, no BQC, 2x2=4 samples (for PCA removal tests)."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(qc_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_qc_dataframe(n_lipids=10, n_samples=4)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_format_type'] = 'Generic Format'
    return at.run()
