"""
UI tests for module navigation between Module 1 and Module 2.

Tests cover:
1. "Next: Quality Check" button — sets module, resets QC state
2. "Back to Data Processing" button — resets module, resets QC state
3. "Back to Home" from Module 2 — resets page and data state
4. _reset_qc_state() — clears all 6 QC session state keys
5. State preservation — normalized_df survives round-trip navigation
6. Edge cases — no normalized_df hides Next button, Module 2 error gate

6 tests addressing Issue 6 from the code review.
"""

from app.constants import COV_THRESHOLD_DEFAULT

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    module1_nav_script,
    module2_nav_script,
    reset_with_overlapping_widget_script,
)

from streamlit.testing.v1 import AppTest


class TestNextToQualityCheck:
    """Tests for Module 1 → Module 2 navigation."""

    def test_next_button_sets_module_to_qc(self, module1_nav_app):
        """Clicking 'Next' navigates to Quality Check module."""
        at = module1_nav_app
        at.button(key='next_module').click().run()
        assert at.session_state['module'] == 'Quality Check & Analysis'

    def test_next_button_resets_qc_state(self, module1_nav_app):
        """Clicking 'Next' clears all QC session state keys including preserved widget values."""
        at = module1_nav_app
        # Pre-populate QC state with non-default values
        at.session_state['qc_continuation_df'] = 'stale'
        at.session_state['qc_bqc_plot'] = 'stale'
        at.session_state['qc_cov_threshold'] = 99
        at.session_state['qc_correlation_plots'] = {'x': 'stale'}
        at.session_state['qc_pca_plot'] = 'stale'
        at.session_state['qc_samples_removed'] = ['s1', 's2']
        at.session_state['_preserved_bqc_filter_choice'] = 'Yes'
        at.session_state['_preserved_rt_viewing_mode'] = 'Individual Mode'
        at.session_state['_preserved_pca_samples_remove'] = ['s1']

        at.button(key='next_module').click().run()

        assert at.session_state['qc_continuation_df'] is None
        assert at.session_state['qc_bqc_plot'] is None
        assert at.session_state['qc_cov_threshold'] == COV_THRESHOLD_DEFAULT
        assert at.session_state['qc_correlation_plots'] == {}
        assert at.session_state['qc_pca_plot'] is None
        assert at.session_state['qc_samples_removed'] == []
        assert at.session_state['_preserved_bqc_filter_choice'] == 'No'
        assert at.session_state['_preserved_rt_viewing_mode'] == 'Comparison Mode'
        assert at.session_state['_preserved_pca_samples_remove'] == []

    def test_next_button_hidden_without_normalized_df(self):
        """'Next' button is not rendered when normalized_df is None."""
        at = AppTest.from_function(module1_nav_script, default_timeout=DEFAULT_TIMEOUT)
        # Don't set normalized_df
        at.run()
        next_buttons = [b for b in at.button if 'next_module' == b.key]
        assert len(next_buttons) == 0


class TestBackToDataProcessing:
    """Tests for Module 2 → Module 1 navigation."""

    def test_back_to_data_processing_resets_module(self, module2_nav_app):
        """Clicking 'Back to Data Processing' returns to Module 1."""
        at = module2_nav_app
        at.button(key='back_m1').click().run()
        assert at.session_state['module'] == 'Data Cleaning, Filtering, & Normalization'

    def test_back_to_data_processing_resets_qc_state(self, module2_nav_app):
        """Clicking 'Back to Data Processing' clears all QC state."""
        at = module2_nav_app
        # Fixture pre-populated QC state with non-defaults
        at.button(key='back_m1').click().run()

        assert at.session_state['qc_continuation_df'] is None
        assert at.session_state['qc_bqc_plot'] is None
        assert at.session_state['qc_cov_threshold'] == COV_THRESHOLD_DEFAULT
        assert at.session_state['qc_correlation_plots'] == {}
        assert at.session_state['qc_pca_plot'] is None
        assert at.session_state['qc_samples_removed'] == []


class TestBackToHomeFromModule2:
    """Tests for Module 2 → Landing page navigation."""

    def test_back_to_home_resets_page(self, module2_nav_app):
        """Clicking 'Back to Home' from Module 2 sets page to 'landing'."""
        at = module2_nav_app
        at.button(key='back_home_m2').click().run()
        assert at.session_state['page'] == 'landing'

    def test_back_to_home_clears_data_state(self, module2_nav_app):
        """Clicking 'Back to Home' from Module 2 resets all data state."""
        at = module2_nav_app
        at.button(key='back_home_m2').click().run()
        # reset_data_state clears normalized_df and all QC state
        assert at.session_state['normalized_df'] is None
        assert at.session_state['qc_continuation_df'] is None
        assert at.session_state['qc_samples_removed'] == []


class TestStatePreservation:
    """Tests for state preservation across module navigation."""

    def test_normalized_df_survives_round_trip(self):
        """normalized_df persists after Module 1 → Module 2 → back to Module 1."""
        import pandas as pd

        at = AppTest.from_function(module1_nav_script, default_timeout=DEFAULT_TIMEOUT)
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:1)', 'SM(20:0)'],
            'ClassKey': ['PC', 'PE', 'SM'],
            'concentration[s1]': [100.0, 200.0, 300.0],
            'concentration[s2]': [150.0, 250.0, 350.0],
        })
        at.session_state['normalized_df'] = df
        at.run()

        # Navigate to Module 2
        at.button(key='next_module').click().run()
        assert at.session_state['module'] == 'Quality Check & Analysis'

        # normalized_df should still be present
        assert at.session_state['normalized_df'] is not None
        assert at.session_state['normalized_df'].shape[0] == 3

    def test_module2_error_gate_without_data(self):
        """Module 2 shows error and back button when normalized_df is missing."""
        at = AppTest.from_function(module2_nav_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['module'] = 'Quality Check & Analysis'
        # Don't set normalized_df
        at.run()

        # Should show error
        assert len(at.error) >= 1
        assert 'No normalized data' in at.error[0].value

        # Back button should be available
        back_buttons = [b for b in at.button if 'back_m1' in (b.key or '')]
        assert len(back_buttons) >= 1


class TestResetDataStateWidgetOverlap:
    """Regression tests for reset_data_state vs widget-instantiated keys.

    The bug: reset_data_state() iterated SessionState fields and assigned
    defaults via st.session_state[key] = .... Two SessionState fields
    (norm_method_selection, protein_input_method) double as Streamlit widget
    keys. If the user reached the normalization UI before clicking
    Back-to-Home, those widgets had been instantiated in the same script
    run, and Streamlit raised StreamlitAPIException on the assignment.

    Earlier tests didn't catch this because (a) unit tests stub
    st.session_state with a plain dict that doesn't enforce the
    widget-instantiation check, and (b) the existing nav fixtures don't
    render the normalization UI before clicking Back-to-Home.
    """

    def test_back_home_does_not_raise_when_widget_keys_overlap(self):
        """reset_data_state must not raise after widgets with overlapping
        SessionState keys (norm_method_selection, protein_input_method) have
        been instantiated in the current script run.
        """
        import pandas as pd

        at = AppTest.from_function(
            reset_with_overlapping_widget_script,
            default_timeout=DEFAULT_TIMEOUT,
        )
        at.session_state['normalized_df'] = pd.DataFrame({'A': [1, 2]})
        at.run()
        assert not at.exception
        assert at.session_state['normalized_df'] is not None

        at.button(key='back_home_reset').click().run()

        # The crux: no StreamlitAPIException raised by reset_data_state.
        assert not at.exception
        assert at.session_state['page'] == 'landing'
        assert at.session_state['normalized_df'] is None
