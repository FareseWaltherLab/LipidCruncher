"""
UI tests for Module 2: Quality Check and Anomaly Detection.

Tests use Streamlit's AppTest framework with wrapper functions from conftest.py.
The QC module is parameterized via session state (_test_df, _test_experiment, etc.)
to test different data scenarios without importing main_app.py.

23 tests across 6 groups covering:
1. Entry point — validation errors, format resolution
2. Box plots — rendering, download buttons
3. BQC assessment — skip/show, threshold, filter yes/no, lipid removal
4. Retention time — hidden for Generic, visible for LipidSearch, modes
5. Correlation — condition selectbox, sample type info
6. PCA — multiselect, sample removal, min-2 error
"""

from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    qc_module_script,
)


# =============================================================================
# Group 1: Entry Point (2 tests)
# =============================================================================

class TestQCEntryPoint:
    """Tests for QC module entry point — validation and format resolution."""

    def test_validation_errors_displayed(self):
        """Empty DataFrame triggers validation error messages."""
        import pandas as pd
        from app.models.experiment import ExperimentConfig

        at = AppTest.from_function(
            qc_module_script, default_timeout=DEFAULT_TIMEOUT
        )
        at.session_state['_test_df'] = pd.DataFrame()
        at.session_state['_test_experiment'] = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3],
        )
        at.run()
        error_values = [e.value for e in at.error]
        assert any("empty" in v.lower() for v in error_values)

    def test_module_renders_without_error(self, qc_generic_app):
        """Valid data renders the QC module without exceptions."""
        at = qc_generic_app
        assert not at.exception
        # Subheader should be rendered
        subheader_values = [s.value for s in at.subheader]
        assert any("Quality Check" in v for v in subheader_values)


# =============================================================================
# Group 2: Box Plots (3 tests)
# =============================================================================

class TestBoxPlots:
    """Tests for box plot section rendering."""

    def test_box_plots_no_exceptions(self, qc_generic_app):
        """Box plots section renders without exceptions."""
        at = qc_generic_app
        assert not at.exception

    def test_box_plots_output_rows(self, qc_generic_app):
        """QC module returns correct row count in output text."""
        at = qc_generic_app
        text_values = [t.value for t in at.text]
        assert any("qc_rows:20" in v for v in text_values)

    def test_box_plots_output_samples(self, qc_generic_app):
        """QC module returns correct sample count in output text."""
        at = qc_generic_app
        text_values = [t.value for t in at.text]
        assert any("qc_samples:6" in v for v in text_values)


# =============================================================================
# Group 3: BQC Assessment (7 tests)
# =============================================================================

class TestBQCAssessment:
    """Tests for BQC quality assessment section."""

    def test_bqc_skipped_no_label(self, qc_generic_app):
        """When bqc_label is None, BQC widgets are not rendered."""
        at = qc_generic_app
        # bqc_cov_threshold and bqc_filter_choice should NOT exist
        bqc_thresholds = [n for n in at.number_input if n.key == 'bqc_cov_threshold']
        bqc_filters = [r for r in at.radio if r.key == 'bqc_filter_choice']
        assert len(bqc_thresholds) == 0
        assert len(bqc_filters) == 0

    def test_bqc_threshold_default_30(self, qc_bqc_app):
        """BQC CoV threshold input defaults to 30."""
        at = qc_bqc_app
        threshold = at.number_input(key='bqc_cov_threshold')
        assert threshold is not None
        assert threshold.value == 30

    def test_bqc_filter_radio_default_no(self, qc_bqc_app):
        """BQC filter radio defaults to 'No'."""
        at = qc_bqc_app
        radio = at.radio(key='bqc_filter_choice')
        assert radio is not None
        assert radio.value == 'No'

    def test_bqc_no_filter_success_message(self, qc_bqc_app):
        """With filter=No, success message 'No lipids removed' appears."""
        at = qc_bqc_app
        success_values = [s.value for s in at.success]
        assert any("No lipids removed" in v for v in success_values)

    def test_bqc_threshold_stored_in_session(self, qc_bqc_app):
        """CoV threshold is stored in session state."""
        at = qc_bqc_app
        assert at.session_state['qc_cov_threshold'] == 30

    def test_bqc_filter_yes_shows_warning(self, qc_bqc_app):
        """Setting filter to 'Yes' shows removal warning for high-CoV lipids."""
        at = qc_bqc_app
        at.radio(key='bqc_filter_choice').set_value('Yes').run()
        assert not at.exception
        # Should show warning about removed lipids (3 high-CoV lipids)
        warning_values = [w.value for w in at.warning]
        assert any("Removed" in v for v in warning_values)

    def test_bqc_filter_yes_reduces_rows(self, qc_bqc_app):
        """Setting filter to 'Yes' reduces row count from original 20."""
        at = qc_bqc_app
        at.radio(key='bqc_filter_choice').set_value('Yes').run()
        text_values = [t.value for t in at.text]
        # Extract row count — should be less than original 20
        row_texts = [v for v in text_values if v.startswith("qc_rows:")]
        assert len(row_texts) == 1
        row_count = int(row_texts[0].split(":")[1])
        assert row_count < 20


# =============================================================================
# Group 4: Retention Time (4 tests)
# =============================================================================

class TestRetentionTime:
    """Tests for retention time analysis section."""

    def test_rt_hidden_generic(self, qc_generic_app):
        """Generic format does not show retention time radio."""
        at = qc_generic_app
        rt_radios = [r for r in at.radio if r.key == 'rt_viewing_mode']
        assert len(rt_radios) == 0

    def test_rt_visible_lipidsearch(self, qc_lipidsearch_app):
        """LipidSearch format shows retention time viewing mode radio."""
        at = qc_lipidsearch_app
        rt_radio = at.radio(key='rt_viewing_mode')
        assert rt_radio is not None

    def test_rt_default_comparison_mode(self, qc_lipidsearch_app):
        """Default RT viewing mode is 'Comparison Mode'."""
        at = qc_lipidsearch_app
        rt_radio = at.radio(key='rt_viewing_mode')
        assert rt_radio.value == 'Comparison Mode'

    def test_rt_class_selection_exists(self, qc_lipidsearch_app):
        """Class selection multiselect exists in comparison mode."""
        at = qc_lipidsearch_app
        class_select = at.multiselect(key='rt_class_selection')
        assert class_select is not None
        # Should have PC and PE as options (from our test data)
        assert len(class_select.value) > 0


# =============================================================================
# Group 5: Correlation (3 tests)
# =============================================================================

class TestCorrelation:
    """Tests for pairwise correlation analysis section."""

    def test_correlation_condition_selectbox(self, qc_generic_app):
        """Correlation condition selectbox exists with eligible conditions."""
        at = qc_generic_app
        selectbox = at.selectbox(key='corr_condition')
        assert selectbox is not None
        # Both conditions have 3 replicates (>1), so both eligible
        assert 'Control' in selectbox.options
        assert 'Treatment' in selectbox.options

    def test_correlation_bio_replicate_info(self, qc_generic_app):
        """Without BQC, shows biological replicates info."""
        at = qc_generic_app
        info_values = [i.value for i in at.info]
        assert any("Biological replicates" in v for v in info_values)

    def test_correlation_tech_replicate_info(self, qc_bqc_app):
        """With BQC label, shows technical replicates info."""
        at = qc_bqc_app
        info_values = [i.value for i in at.info]
        assert any("Technical replicates" in v for v in info_values)


# =============================================================================
# Group 6: PCA Analysis (4 tests)
# =============================================================================

class TestPCA:
    """Tests for PCA analysis section."""

    def test_pca_multiselect_exists(self, qc_generic_app):
        """PCA sample removal multiselect exists."""
        at = qc_generic_app
        ms = at.multiselect(key='pca_samples_remove')
        assert ms is not None
        # Options should be all samples
        assert 's1' in ms.options
        assert 's6' in ms.options

    def test_pca_stores_plot_in_session(self, qc_generic_app):
        """PCA plot is stored in session state."""
        at = qc_generic_app
        assert at.session_state['qc_pca_plot'] is not None

    def test_pca_remove_samples_updates_state(self, qc_generic_app):
        """Removing samples updates qc_samples_removed in session state."""
        at = qc_generic_app
        at.multiselect(key='pca_samples_remove').set_value(['s1']).run()
        assert not at.exception
        assert at.session_state['qc_samples_removed'] == ['s1']
        # Output should show 5 samples remaining
        text_values = [t.value for t in at.text]
        assert any("qc_samples:5" in v for v in text_values)

    def test_pca_min_2_error(self, qc_small_app):
        """Removing all but 1 sample shows min-2 error."""
        at = qc_small_app
        # 4 samples total, remove 3 → only 1 left → error
        at.multiselect(key='pca_samples_remove').set_value(
            ['s1', 's2', 's3']
        ).run()
        assert not at.exception
        error_values = [e.value for e in at.error]
        assert any("At least two samples" in v for v in error_values)
