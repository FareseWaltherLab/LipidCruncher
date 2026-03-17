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
        st.session_state._preserved_bqc_filter_choice = 'No'
        st.session_state._preserved_rt_viewing_mode = 'Comparison Mode'
        st.session_state._preserved_pca_samples_remove = []

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
        st.session_state._preserved_bqc_filter_choice = 'No'
        st.session_state._preserved_rt_viewing_mode = 'Comparison Mode'
        st.session_state._preserved_pca_samples_remove = []

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


# =============================================================================
# Module 1: Main Content — Wrapper Functions
# =============================================================================

def zero_filtering_script():
    """Zero filtering UI with test data from session state.

    Expects session state keys:
        _test_cleaned_df: DataFrame with intensity[s] columns
        _test_experiment: ExperimentConfig
        _test_bqc_label: Optional BQC label string
        _test_data_format: Format string (default 'Generic Format')
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.zero_filtering import display_zero_filtering_config

    df = st.session_state['_test_cleaned_df']
    experiment = st.session_state['_test_experiment']
    bqc_label = st.session_state.get('_test_bqc_label')
    data_format = st.session_state.get('_test_data_format', 'Generic Format')

    result = display_zero_filtering_config(df, experiment, bqc_label, data_format)
    if result and result[0] is not None:
        st.text(f"filtered_rows:{result[0].shape[0]}")
        st.text(f"removed:{len(result[1])}")
    else:
        st.text("no_result")


def internal_standards_script():
    """Internal standards management UI with test data from session state.

    Expects session state keys:
        _test_cleaned_df: DataFrame with intensity[s] columns
        _test_auto_detected_df: Optional auto-detected standards DataFrame
        _test_experiment: Optional ExperimentConfig (for consistency plots)
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.internal_standards import display_manage_internal_standards

    cleaned_df = st.session_state['_test_cleaned_df']
    auto_detected_df = st.session_state.get('_test_auto_detected_df')

    # Set experiment in session state for consistency plots
    if '_test_experiment' in st.session_state:
        st.session_state.experiment = st.session_state['_test_experiment']

    result = display_manage_internal_standards(cleaned_df, auto_detected_df)
    if result is not None and not result.empty:
        st.text(f"standards_count:{len(result)}")
    else:
        st.text("no_standards")


def normalization_script():
    """Normalization UI with test data from session state.

    Expects session state keys:
        _test_cleaned_df: DataFrame with intensity[s] columns and ClassKey
        _test_intsta_df: Optional internal standards DataFrame
        _test_experiment: ExperimentConfig
        _test_data_format: Format string (default 'Generic Format')
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.normalization import display_normalization_ui

    cleaned_df = st.session_state['_test_cleaned_df']
    intsta_df = st.session_state.get('_test_intsta_df')
    experiment = st.session_state['_test_experiment']
    data_format = st.session_state.get('_test_data_format', 'Generic Format')

    result = display_normalization_ui(cleaned_df, intsta_df, experiment, data_format)
    if result is not None:
        st.text(f"normalized_rows:{result.shape[0]}")
    else:
        st.text("not_normalized")


def grade_filtering_script():
    """Grade filtering UI with test data from session state.

    Expects session state keys:
        _test_df: DataFrame with ClassKey and TotalGrade columns
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import display_grade_filtering_config

    df = st.session_state['_test_df']
    config = display_grade_filtering_config(df)
    if config is not None:
        st.text(f"custom_config:{len(config)}")
    else:
        st.text("default_config")


def quality_filtering_script():
    """Quality filtering UI with MS-DIAL features in session state.

    Expects session state keys:
        msdial_features: dict with 'has_quality_score' and 'has_msms_matched'
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import display_quality_filtering_config

    config = display_quality_filtering_config()
    if config is not None:
        st.text(f"threshold:{config['total_score_threshold']}")
        st.text(f"msms:{config['require_msms']}")
    else:
        st.text("no_quality_config")


def column_mapping_script():
    """Column mapping display with test data from session state.

    Expects session state keys:
        _test_df: Standardized DataFrame
        _test_data_format: Format string
        column_mapping: DataFrame with Original/Standardized columns
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.sidebar.column_mapping import display_column_mapping

    df = st.session_state['_test_df']
    data_format = st.session_state.get('_test_data_format', 'Generic Format')

    success, modified_df = display_column_mapping(df, data_format)
    st.text(f"mapping_success:{success}")
    if modified_df is not None:
        st.text(f"modified_cols:{modified_df.shape[1]}")


# =============================================================================
# Module 1: Main Content — Data Builders
# =============================================================================

def make_cleaned_dataframe(n_lipids=20, n_samples=6, classes=None):
    """Build a cleaned DataFrame with intensity columns for Module 1 tests.

    Args:
        n_lipids: Number of lipid rows.
        n_samples: Number of sample columns (intensity[s1]..intensity[sN]).
        classes: Optional list of class names (length n_lipids). Default: half PC, half PE.

    Returns:
        DataFrame with LipidMolec, ClassKey, and intensity columns.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    if classes is None:
        half = n_lipids // 2
        classes = ['PC'] * half + ['PE'] * (n_lipids - half)
    lipids = [f'{cls}({i}:0)' for i, cls in enumerate(classes)]

    data = {
        'LipidMolec': lipids,
        'ClassKey': classes,
    }
    for i in range(1, n_samples + 1):
        data[f'intensity[s{i}]'] = np.random.uniform(500, 5000, n_lipids).tolist()

    return pd.DataFrame(data)


def make_intsta_dataframe(standards=None):
    """Build an internal standards DataFrame.

    Args:
        standards: Optional list of (lipid_name, class_name) tuples.
            Default: [('PC(15:0_18:1)+D7:(s)', 'PC'), ('PE(15:0_18:1)+D7:(s)', 'PE')]

    Returns:
        DataFrame with LipidMolec, ClassKey, and intensity columns.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(99)
    if standards is None:
        standards = [
            ('PC(15:0_18:1)+D7:(s)', 'PC'),
            ('PE(15:0_18:1)+D7:(s)', 'PE'),
        ]

    data = {
        'LipidMolec': [s[0] for s in standards],
        'ClassKey': [s[1] for s in standards],
    }
    for i in range(1, 7):
        data[f'intensity[s{i}]'] = np.random.uniform(100, 1000, len(standards)).tolist()

    return pd.DataFrame(data)


def make_grade_dataframe(n_lipids=15, classes=None):
    """Build a DataFrame with ClassKey and TotalGrade columns for grade filtering tests.

    Returns:
        DataFrame with LipidMolec, ClassKey, TotalGrade, and intensity columns.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    if classes is None:
        classes = ['PC'] * 5 + ['PE'] * 5 + ['LPC'] * 3 + ['SM'] * 2
        n_lipids = len(classes)
    grades = ['A', 'B', 'C', 'D', 'A'] * (n_lipids // 5 + 1)
    grades = grades[:n_lipids]
    lipids = [f'{cls}({i}:0)' for i, cls in enumerate(classes)]

    data = {
        'LipidMolec': lipids,
        'ClassKey': classes,
        'TotalGrade': grades,
    }
    for i in range(1, 7):
        data[f'intensity[s{i}]'] = np.random.uniform(500, 5000, n_lipids).tolist()

    return pd.DataFrame(data)


# =============================================================================
# Module 1: Main Content — Fixtures
# =============================================================================

@pytest.fixture
def zero_filter_generic_app():
    """Zero filtering: Generic format, 2x3=6 samples, no BQC."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(zero_filtering_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_data_format'] = 'Generic Format'
    return at.run()


@pytest.fixture
def zero_filter_bqc_app():
    """Zero filtering: Generic format, with BQC, Control(3)+Treatment(3)+BQC(2)=8."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(zero_filtering_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=8)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=3,
        conditions_list=['Control', 'Treatment', 'BQC'],
        number_of_samples_list=[3, 3, 2],
    )
    at.session_state['_test_bqc_label'] = 'BQC'
    at.session_state['_test_data_format'] = 'Generic Format'
    return at.run()


@pytest.fixture
def zero_filter_lipidsearch_app():
    """Zero filtering: LipidSearch format, 2x3=6 samples, no BQC."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(zero_filtering_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_data_format'] = 'LipidSearch 5.0'
    return at.run()


@pytest.fixture
def intsta_with_standards_app():
    """Internal standards management: auto-detected standards present."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(internal_standards_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_auto_detected_df'] = make_intsta_dataframe()
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    return at.run()


@pytest.fixture
def intsta_no_standards_app():
    """Internal standards management: no auto-detected standards."""
    at = AppTest.from_function(internal_standards_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_auto_detected_df'] = None
    return at.run()


@pytest.fixture
def norm_with_standards_app():
    """Normalization: with internal standards, Generic format, 2x3=6 samples."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(normalization_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_intsta_df'] = make_intsta_dataframe()
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_data_format'] = 'Generic Format'
    return at.run()


@pytest.fixture
def norm_no_standards_app():
    """Normalization: no internal standards, Generic format, 2x3=6 samples."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(normalization_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_intsta_df'] = None
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_data_format'] = 'Generic Format'
    return at.run()


@pytest.fixture
def grade_filtering_app():
    """Grade filtering: LipidSearch data with ClassKey and TotalGrade."""
    at = AppTest.from_function(grade_filtering_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_grade_dataframe()
    return at.run()


@pytest.fixture
def quality_filtering_app():
    """Quality filtering: MS-DIAL with quality score available."""
    at = AppTest.from_function(quality_filtering_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['msdial_features'] = {
        'has_quality_score': True,
        'has_msms_matched': True,
        'raw_sample_columns': ['s1', 's2', 's3'],
        'normalized_sample_columns': [],
    }
    return at.run()


@pytest.fixture
def column_mapping_generic_app():
    """Column mapping: Generic format with mapping DataFrame."""
    import pandas as pd

    at = AppTest.from_function(column_mapping_script, default_timeout=DEFAULT_TIMEOUT)
    cleaned_df = make_cleaned_dataframe(n_lipids=10, n_samples=4)
    at.session_state['_test_df'] = cleaned_df
    at.session_state['_test_data_format'] = 'Generic Format'
    at.session_state['column_mapping'] = pd.DataFrame({
        'Original': ['LipidMolec', 'col1', 'col2', 'col3', 'col4'],
        'Standardized': ['LipidMolec', 'intensity[s1]', 'intensity[s2]', 'intensity[s3]', 'intensity[s4]'],
    })
    return at.run()


@pytest.fixture
def column_mapping_msdial_app():
    """Column mapping: MS-DIAL format with override available."""
    import pandas as pd

    at = AppTest.from_function(column_mapping_script, default_timeout=DEFAULT_TIMEOUT)
    cleaned_df = make_cleaned_dataframe(n_lipids=10, n_samples=4)
    at.session_state['_test_df'] = cleaned_df
    at.session_state['_test_data_format'] = 'MS-DIAL'
    at.session_state['column_mapping'] = pd.DataFrame({
        'Original': ['Metabolite name', 's1', 's2', 's3', 's4'],
        'Standardized': ['LipidMolec', 'intensity[s1]', 'intensity[s2]', 'intensity[s3]', 'intensity[s4]'],
    })
    at.session_state['msdial_features'] = {
        'has_normalized_data': False,
        'raw_sample_columns': ['s1', 's2', 's3', 's4'],
        'normalized_sample_columns': [],
        'has_quality_score': True,
        'has_msms_matched': False,
        'actual_columns': ['Metabolite name', 'Total score', 's1', 's2', 's3', 's4'],
    }
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


# =============================================================================
# Module 3: Analysis — Wrapper Functions
# =============================================================================

def analysis_module_script():
    """Run Analysis module with test data from session state.

    Expects session state keys:
        _test_df: DataFrame with concentration[s] columns, LipidMolec, ClassKey
        _test_experiment: ExperimentConfig
        _test_bqc_label: Optional BQC label string
        _test_format_type: Format string (default 'Generic Format')
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    # Initialize analysis_all_plots dict (normally done by main_app.py routing)
    if not st.session_state.get('analysis_all_plots'):
        st.session_state.analysis_all_plots = {}
    from app.ui.main_content.analysis import display_analysis_module

    df = st.session_state['_test_df']
    experiment = st.session_state['_test_experiment']
    bqc_label = st.session_state.get('_test_bqc_label')
    format_type = st.session_state.get('_test_format_type', 'Generic Format')

    display_analysis_module(
        df=df,
        experiment=experiment,
        bqc_label=bqc_label,
        format_type=format_type,
    )
    st.text("analysis_rendered:ok")


def module3_nav_script():
    """Module 3 with navigation buttons (Back to QC + Back to Home).

    Replicates main_app.py _display_module3() navigation logic (without analysis rendering).

    Expects session state keys:
        qc_continuation_df: DataFrame (optional — analysis data source)
        normalized_df: DataFrame (fallback data source)
        module: str = 'Visualize & Analyze'
    """
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    def _reset_analysis_state():
        st.session_state.analysis_selection = None
        st.session_state.analysis_bar_chart_fig = None
        st.session_state.analysis_pie_chart_figs = {}
        st.session_state.analysis_saturation_figs = {}
        st.session_state.analysis_fach_fig = None
        st.session_state.analysis_pathway_fig = None
        st.session_state.analysis_volcano_fig = None
        st.session_state.analysis_volcano_data = None
        st.session_state.analysis_heatmap_fig = None
        st.session_state.analysis_heatmap_clusters = None
        st.session_state.analysis_all_plots = {}

    analysis_df = st.session_state.get('qc_continuation_df')
    if analysis_df is None:
        analysis_df = st.session_state.get('normalized_df')
    if analysis_df is None:
        st.error("No data available. Please complete Modules 1 and 2 first.")
        if st.button("← Back to Quality Check", key="back_qc_error"):
            st.session_state.module = "Quality Check & Analysis"
            st.rerun()
        return

    current_module = st.session_state.get('module', 'unknown')
    st.text(f"module:{current_module}")
    st.text(f"analysis_rows:{analysis_df.shape[0]}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Quality Check", key="back_qc_module3"):
            _reset_analysis_state()
            st.session_state.module = "Quality Check & Analysis"
            st.rerun()
    with col2:
        if st.button("← Back to Home", key="back_home_module3"):
            st.session_state.page = 'landing'
            StreamlitAdapter.reset_data_state()
            st.rerun()


# =============================================================================
# Module 3: Analysis — Data Builders
# =============================================================================

def make_analysis_dataframe(n_lipids=20, n_samples=6, detailed_fa=False):
    """Build an analysis-ready DataFrame with concentration columns.

    Args:
        n_lipids: Number of lipid rows.
        n_samples: Number of sample columns (concentration[s1]..concentration[sN]).
        detailed_fa: Use detailed fatty acid names (e.g., PC(16:0_18:1)) instead
                     of consolidated (e.g., PC(34:1)).

    Returns:
        DataFrame with LipidMolec, ClassKey, and concentration columns.
    """
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    half = n_lipids // 2
    classes = ['PC'] * half + ['PE'] * (n_lipids - half)

    if detailed_fa:
        lipids = []
        for i, cls in enumerate(classes):
            c1 = 14 + (i % 6)  # chain1: 14-19
            c2 = 16 + (i % 5)  # chain2: 16-20
            d1 = i % 2          # double bonds: 0 or 1
            d2 = (i + 1) % 3    # double bonds: 0, 1, or 2
            lipids.append(f'{cls}({c1}:{d1}_{c2}:{d2})')
    else:
        lipids = [f'{cls}({30 + i}:{i % 3})' for i, cls in enumerate(classes)]

    data = {
        'LipidMolec': lipids,
        'ClassKey': classes,
    }

    for i in range(1, n_samples + 1):
        data[f'concentration[s{i}]'] = np.random.uniform(500, 5000, n_lipids).tolist()

    return pd.DataFrame(data)


# =============================================================================
# Module 3: Analysis — Fixtures
# =============================================================================

@pytest.fixture
def analysis_generic_app():
    """Analysis module: Generic format, 2x3=6 samples, 20 lipids, consolidated names."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_format_type'] = 'Generic Format'
    return at.run()


@pytest.fixture
def analysis_detailed_fa_app():
    """Analysis module: Generic format, 2x3=6 samples, 20 lipids, detailed FA names."""
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(
        n_lipids=20, n_samples=6, detailed_fa=True,
    )
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_format_type'] = 'Generic Format'
    return at.run()


@pytest.fixture
def module3_nav_app():
    """Module 3 navigation with analysis data pre-populated."""
    at = AppTest.from_function(module3_nav_script, default_timeout=DEFAULT_TIMEOUT)
    df = make_analysis_dataframe(n_lipids=10, n_samples=6)
    at.session_state['qc_continuation_df'] = df
    at.session_state['normalized_df'] = df.copy()
    at.session_state['module'] = 'Visualize & Analyze'
    # Pre-populate analysis state to verify it gets cleared
    at.session_state['analysis_bar_chart_fig'] = 'fake_bar'
    at.session_state['analysis_volcano_fig'] = 'fake_volcano'
    at.session_state['analysis_all_plots'] = {'bar': 'fake'}
    return at.run()
