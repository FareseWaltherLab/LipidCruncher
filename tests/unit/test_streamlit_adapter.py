"""
Unit tests for StreamlitAdapter.

Tests the adapter layer that bridges UI and services.
Focuses on testable components: SessionState dataclass, utility methods.

Note: Methods with @st.cache_data decorators and session state access
require mocking or integration tests with Streamlit.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import asdict, fields

from app.adapters.streamlit_adapter import (
    SessionState,
    StreamlitAdapter,
    _WIDGET_KEYS,
    _DYNAMIC_KEY_PREFIXES,
    _PRESERVE_ON_RESET,
)
from app.models.experiment import ExperimentConfig
from app.services.format_detection import DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig


class MockSessionState(dict):
    """
    Mock for Streamlit's session_state that supports both dict-like and attribute access.
    This mimics how st.session_state works in actual Streamlit.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'MockSessionState' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'MockSessionState' object has no attribute '{key}'")


# =============================================================================
# SessionState Dataclass Tests
# =============================================================================

class TestSessionStateDefaults:
    """Tests for SessionState default values."""

    def test_default_data_states_none(self):
        """Test that data states default to None."""
        state = SessionState()
        assert state.raw_df is None
        assert state.cleaned_df is None
        assert state.intsta_df is None
        assert state.normalized_df is None
        assert state.continuation_df is None
        assert state.original_column_order is None

    def test_default_experiment_config_none(self):
        """Test that experiment configuration defaults to None."""
        state = SessionState()
        assert state.experiment is None
        assert state.format_type is None
        assert state.bqc_label is None

    def test_default_process_control(self):
        """Test default values for process control."""
        state = SessionState()
        assert state.page == 'landing'
        assert state.module == "Data Cleaning, Filtering, & Normalization"
        assert state.confirmed is False
        assert state.grouping_complete is True

    def test_default_normalization_settings(self):
        """Test default normalization settings."""
        state = SessionState()
        assert state.normalization_method == 'None'
        assert state.normalization_inputs == {}
        assert state.selected_classes == []
        assert state.create_norm_dataset is False

    def test_default_grade_config(self):
        """Test default grade configuration."""
        state = SessionState()
        assert state.grade_config is None

    def test_default_msdial_config(self):
        """Test default MS-DIAL configuration."""
        state = SessionState()
        assert state.msdial_quality_config is None
        assert state.msdial_features is None
        assert state.msdial_use_normalized is False
        assert state.msdial_data_type_index == 0

    def test_default_standards_settings(self):
        """Test default standards settings."""
        state = SessionState()
        assert state.original_auto_intsta_df is None
        assert state.preserved_intsta_df is None
        assert state.preserved_standards_mode is None

    def test_default_qc_settings(self):
        """Test default quality check settings."""
        state = SessionState()
        assert state.qc_continuation_df is None
        assert state.qc_bqc_plot is None
        assert state.qc_cov_threshold == 30
        assert state.qc_correlation_plots == {}
        assert state.qc_pca_plot is None
        assert state.qc_samples_removed == []
        assert state._preserved_bqc_filter_choice == 'No'
        assert state._preserved_rt_viewing_mode == 'Comparison Mode'
        assert state._preserved_pca_samples_remove == []

    def test_default_analysis_none_fields(self):
        """Test that analysis fields storing single values default to None."""
        state = SessionState()
        assert state.analysis_selection is None
        assert state.analysis_bar_chart_fig is None
        assert state.analysis_fach_fig is None
        assert state.analysis_pathway_fig is None
        assert state.analysis_volcano_fig is None
        assert state.analysis_volcano_data is None
        assert state.analysis_heatmap_fig is None
        assert state.analysis_heatmap_clusters is None

    def test_default_analysis_dict_fields(self):
        """Test that analysis dict fields default to empty dicts."""
        state = SessionState()
        assert state.analysis_pie_chart_figs == {}
        assert state.analysis_saturation_figs == {}
        assert state.analysis_all_plots == {}

    def test_default_analysis_dict_fields_are_dicts(self):
        """Test that analysis dict fields are actually dict instances."""
        state = SessionState()
        assert isinstance(state.analysis_pie_chart_figs, dict)
        assert isinstance(state.analysis_saturation_figs, dict)
        assert isinstance(state.analysis_all_plots, dict)


class TestSessionStateCreation:
    """Tests for SessionState creation with custom values."""

    def test_create_with_dataframe(self):
        """Test creating SessionState with DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        state = SessionState(raw_df=df)
        assert state.raw_df is not None
        assert len(state.raw_df) == 3

    def test_create_with_experiment(self):
        """Test creating SessionState with ExperimentConfig."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        state = SessionState(experiment=experiment)
        assert state.experiment is not None
        assert state.experiment.n_conditions == 2

    def test_create_with_format_type(self):
        """Test creating SessionState with format type."""
        state = SessionState(format_type=DataFormat.LIPIDSEARCH)
        assert state.format_type == DataFormat.LIPIDSEARCH

    def test_create_with_custom_page(self):
        """Test creating SessionState with custom page."""
        state = SessionState(page='analysis')
        assert state.page == 'analysis'

    def test_create_with_normalization_inputs(self):
        """Test creating SessionState with normalization inputs."""
        inputs = {'method': 'log2', 'param1': 5}
        state = SessionState(normalization_inputs=inputs)
        assert state.normalization_inputs == inputs

    def test_create_with_grade_config(self):
        """Test creating SessionState with grade config."""
        grade_config = GradeFilterConfig(grade_config={'PC': ['A']})
        state = SessionState(grade_config=grade_config)
        assert state.grade_config is not None

    def test_create_with_all_fields(self):
        """Test creating SessionState with all fields."""
        df = pd.DataFrame({'A': [1]})
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1]
        )

        state = SessionState(
            raw_df=df,
            cleaned_df=df,
            intsta_df=df,
            experiment=experiment,
            format_type=DataFormat.GENERIC,
            page='results',
            module='Analysis',
            confirmed=True,
            normalization_method='Log2'
        )

        assert state.raw_df is not None
        assert state.experiment is not None
        assert state.page == 'results'
        assert state.confirmed is True

    def test_create_with_analysis_selection(self):
        """Test creating SessionState with analysis selection."""
        state = SessionState(analysis_selection='Abundance Bar Chart')
        assert state.analysis_selection == 'Abundance Bar Chart'

    def test_create_with_analysis_fig(self):
        """Test creating SessionState with an analysis figure (Any type)."""
        mock_fig = MagicMock()
        state = SessionState(analysis_bar_chart_fig=mock_fig)
        assert state.analysis_bar_chart_fig is mock_fig

    def test_create_with_analysis_pie_chart_figs(self):
        """Test creating SessionState with pie chart figs dict."""
        figs = {'Control': MagicMock(), 'Treatment': MagicMock()}
        state = SessionState(analysis_pie_chart_figs=figs)
        assert len(state.analysis_pie_chart_figs) == 2
        assert 'Control' in state.analysis_pie_chart_figs

    def test_create_with_analysis_heatmap_clusters(self):
        """Test creating SessionState with heatmap clusters DataFrame."""
        df = pd.DataFrame({'Cluster': [1, 2], 'Count': [5, 3]})
        state = SessionState(analysis_heatmap_clusters=df)
        assert state.analysis_heatmap_clusters is not None
        assert len(state.analysis_heatmap_clusters) == 2

    def test_create_with_analysis_all_plots(self):
        """Test creating SessionState with all_plots dict."""
        plots = {'bar_chart': MagicMock(), 'pie_charts': MagicMock()}
        state = SessionState(analysis_all_plots=plots)
        assert len(state.analysis_all_plots) == 2


class TestSessionStateDataIntegrity:
    """Tests for SessionState data integrity."""

    def test_dataframe_reference_preserved(self):
        """Test that DataFrame references are preserved."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        state = SessionState(raw_df=df)
        # Should be same reference
        assert state.raw_df is df

    def test_mutable_defaults_independent(self):
        """Test that mutable defaults are independent per instance."""
        state1 = SessionState()
        state2 = SessionState()

        state1.normalization_inputs['key'] = 'value'

        # state2 should not be affected
        assert 'key' not in state2.normalization_inputs

    def test_list_defaults_independent(self):
        """Test that list defaults are independent per instance."""
        state1 = SessionState()
        state2 = SessionState()

        state1.selected_classes.append('PC')

        # state2 should not be affected
        assert 'PC' not in state2.selected_classes

    def test_analysis_pie_chart_figs_independent(self):
        """Test that analysis_pie_chart_figs dict defaults are independent per instance."""
        state1 = SessionState()
        state2 = SessionState()

        state1.analysis_pie_chart_figs['Control'] = 'fig1'

        assert 'Control' not in state2.analysis_pie_chart_figs

    def test_analysis_saturation_figs_independent(self):
        """Test that analysis_saturation_figs dict defaults are independent per instance."""
        state1 = SessionState()
        state2 = SessionState()

        state1.analysis_saturation_figs['PC'] = 'fig1'

        assert 'PC' not in state2.analysis_saturation_figs

    def test_analysis_all_plots_independent(self):
        """Test that analysis_all_plots dict defaults are independent per instance."""
        state1 = SessionState()
        state2 = SessionState()

        state1.analysis_all_plots['bar'] = 'fig1'

        assert 'bar' not in state2.analysis_all_plots

    def test_analysis_fig_reference_preserved(self):
        """Test that analysis figure references are preserved."""
        mock_fig = MagicMock()
        state = SessionState(analysis_volcano_fig=mock_fig)
        assert state.analysis_volcano_fig is mock_fig

    def test_analysis_heatmap_clusters_reference_preserved(self):
        """Test that analysis heatmap clusters DataFrame reference is preserved."""
        df = pd.DataFrame({'Cluster': [1, 2]})
        state = SessionState(analysis_heatmap_clusters=df)
        assert state.analysis_heatmap_clusters is df


# =============================================================================
# Widget Keys Registry Tests
# =============================================================================

class TestWidgetKeysRegistry:
    """Tests for _WIDGET_KEYS completeness and structure."""

    def test_widget_keys_is_set(self):
        """Test that _WIDGET_KEYS is a set (no duplicates)."""
        assert isinstance(_WIDGET_KEYS, set)

    def test_widget_keys_all_strings(self):
        """Test that all widget keys are strings."""
        for key in _WIDGET_KEYS:
            assert isinstance(key, str), f"Widget key {key!r} is not a string"

    def test_analysis_radio_in_widget_keys(self):
        """Test that the analysis radio widget key is registered."""
        assert 'analysis_radio' in _WIDGET_KEYS

    def test_bar_chart_widget_keys_registered(self):
        """Test that bar chart widget keys are registered."""
        expected = {'bar_conditions', 'bar_classes', 'bar_scale_radio',
                    'bar_stats_mode', 'bar_detailed_stats'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_pie_chart_widget_keys_registered(self):
        """Test that pie chart widget keys are registered."""
        assert 'pie_classes' in _WIDGET_KEYS

    def test_saturation_widget_keys_registered(self):
        """Test that saturation widget keys are registered."""
        expected = {'sat_conditions', 'sat_classes', 'sat_plot_type',
                    'sat_show_significance', 'sat_detailed_stats', 'sat_stats_mode'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_fach_widget_keys_registered(self):
        """Test that FACH widget keys are registered."""
        expected = {'fach_class', 'fach_conditions'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_pathway_widget_keys_registered(self):
        """Test that pathway widget keys are registered."""
        expected = {'pathway_control', 'pathway_experimental'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_volcano_widget_keys_registered(self):
        """Test that volcano widget keys are registered."""
        expected = {'volcano_control', 'volcano_experimental', 'volcano_classes',
                    'volcano_p_threshold', 'volcano_fc_threshold',
                    'volcano_hide_nonsig', 'volcano_top_n',
                    'volcano_additional_labels', 'volcano_stats_mode',
                    'volcano_detailed_stats'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_heatmap_widget_keys_registered(self):
        """Test that heatmap widget keys are registered."""
        expected = {'heatmap_conditions', 'heatmap_classes', 'heatmap_type',
                    'heatmap_n_clusters', 'heatmap_cluster_view'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_qc_widget_keys_still_registered(self):
        """Test that existing QC widget keys are still present (no regression)."""
        expected = {'bqc_cov_threshold', 'bqc_filter_choice', 'pca_samples_remove',
                    'corr_condition', 'rt_viewing_mode'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_sidebar_widget_keys_still_registered(self):
        """Test that existing sidebar widget keys are still present (no regression)."""
        expected = {'grouping_radio', 'bqc_radio', 'confirm_checkbox'}
        assert expected.issubset(_WIDGET_KEYS)

    def test_widget_keys_session_state_overlap_is_intentional(self):
        """Test that only known widget keys overlap with SessionState field names.

        Some keys (norm_method_selection, protein_input_method) intentionally
        exist in both _WIDGET_KEYS and SessionState — they're widget keys that
        also have session state backing for preservation across reruns.
        """
        field_names = {f.name for f in fields(SessionState)}
        overlap = _WIDGET_KEYS & field_names
        expected_overlap = {'norm_method_selection', 'protein_input_method'}
        assert overlap == expected_overlap, \
            f"Unexpected overlap: {overlap - expected_overlap}"


class TestDynamicKeyPrefixes:
    """Tests for _DYNAMIC_KEY_PREFIXES completeness and structure."""

    def test_dynamic_prefixes_is_tuple(self):
        """Test that _DYNAMIC_KEY_PREFIXES is a tuple."""
        assert isinstance(_DYNAMIC_KEY_PREFIXES, tuple)

    def test_dynamic_prefixes_all_strings(self):
        """Test that all dynamic prefixes are strings."""
        for prefix in _DYNAMIC_KEY_PREFIXES:
            assert isinstance(prefix, str), f"Prefix {prefix!r} is not a string"

    def test_dynamic_prefixes_end_with_underscore(self):
        """Test that all dynamic prefixes end with underscore (convention)."""
        for prefix in _DYNAMIC_KEY_PREFIXES:
            assert prefix.endswith('_'), f"Prefix {prefix!r} doesn't end with '_'"

    def test_volcano_label_prefixes_registered(self):
        """Test that volcano label position prefixes are registered."""
        assert 'volcano_label_x_' in _DYNAMIC_KEY_PREFIXES
        assert 'volcano_label_y_' in _DYNAMIC_KEY_PREFIXES

    def test_analysis_download_prefixes_registered(self):
        """Test that analysis download button prefixes are registered."""
        assert 'analysis_svg_' in _DYNAMIC_KEY_PREFIXES
        assert 'analysis_csv_' in _DYNAMIC_KEY_PREFIXES

    def test_existing_prefixes_still_registered(self):
        """Test that existing dynamic key prefixes are still present (no regression)."""
        expected = ('protein_', 'conc_', 'standard_selection_', 'grade_select_',
                    'cond_name_', 'n_samples_', 'select_',
                    'qc_rt_svg_individual_', 'rt_csv_individual_')
        for prefix in expected:
            assert prefix in _DYNAMIC_KEY_PREFIXES, f"Missing prefix: {prefix}"

    def test_dynamic_prefixes_dont_overlap_with_widget_keys(self):
        """Test that no static widget key starts with a dynamic prefix.

        Known exceptions: 'protein_input_method' starts with 'protein_' prefix
        but is a static widget key, not a dynamic one.
        """
        known_exceptions = {'protein_input_method', 'protein_csv_upload'}
        for key in _WIDGET_KEYS - known_exceptions:
            for prefix in _DYNAMIC_KEY_PREFIXES:
                assert not key.startswith(prefix), \
                    f"Widget key {key!r} starts with dynamic prefix {prefix!r}"


class TestPreserveOnReset:
    """Tests for _PRESERVE_ON_RESET configuration."""

    def test_preserve_on_reset_contains_page(self):
        """Test that 'page' is preserved on reset."""
        assert 'page' in _PRESERVE_ON_RESET

    def test_preserve_on_reset_contains_module(self):
        """Test that 'module' is preserved on reset."""
        assert 'module' in _PRESERVE_ON_RESET

    def test_preserve_on_reset_does_not_contain_analysis_keys(self):
        """Test that analysis keys are NOT preserved on reset (they should reset)."""
        analysis_keys = {
            'analysis_selection', 'analysis_bar_chart_fig',
            'analysis_pie_chart_figs', 'analysis_saturation_figs',
            'analysis_fach_fig', 'analysis_pathway_fig',
            'analysis_volcano_fig', 'analysis_volcano_data',
            'analysis_heatmap_fig', 'analysis_heatmap_clusters',
            'analysis_all_plots',
        }
        overlap = _PRESERVE_ON_RESET & analysis_keys
        assert overlap == set(), f"Analysis keys should not be preserved on reset: {overlap}"


# =============================================================================
# StreamlitAdapter Utility Method Tests
# =============================================================================


class TestConfigHashing:
    """Tests for config object hashing (used by @st.cache_data)."""

    def test_experiment_config_hashable(self):
        """Test that ExperimentConfig is hashable."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )
        assert isinstance(hash(config), int)

    def test_equal_experiments_same_hash(self):
        """Test that equal ExperimentConfigs produce the same hash."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )
        config2 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )
        assert hash(config1) == hash(config2)

    def test_different_experiments_different_hash(self):
        """Test that different ExperimentConfigs produce different hashes."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )
        config2 = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 2, 2]
        )
        assert hash(config1) != hash(config2)

    def test_grade_config_hashable(self):
        """Test that GradeFilterConfig is hashable."""
        config = GradeFilterConfig(grade_config={'PC': ['A', 'B']})
        assert isinstance(hash(config), int)

    def test_grade_config_none_hashable(self):
        """Test that GradeFilterConfig with None is hashable."""
        config = GradeFilterConfig(grade_config=None)
        assert isinstance(hash(config), int)

    def test_equal_grade_configs_same_hash(self):
        """Test that equal GradeFilterConfigs produce the same hash."""
        config1 = GradeFilterConfig(grade_config={'PC': ['A', 'B']})
        config2 = GradeFilterConfig(grade_config={'PC': ['A', 'B']})
        assert hash(config1) == hash(config2)
        assert config1 == config2

    def test_quality_config_hashable(self):
        """Test that QualityFilterConfig is hashable."""
        config = QualityFilterConfig(total_score_threshold=70, require_msms=True)
        assert isinstance(hash(config), int)

    def test_equal_quality_configs_same_hash(self):
        """Test that equal QualityFilterConfigs produce the same hash."""
        config1 = QualityFilterConfig(total_score_threshold=70, require_msms=True)
        config2 = QualityFilterConfig(total_score_threshold=70, require_msms=True)
        assert hash(config1) == hash(config2)
        assert config1 == config2

    def test_different_quality_configs_different_hash(self):
        """Test that different QualityFilterConfigs produce different hashes."""
        config1 = QualityFilterConfig(total_score_threshold=70, require_msms=True)
        config2 = QualityFilterConfig(total_score_threshold=60, require_msms=False)
        assert hash(config1) != hash(config2)
        assert config1 != config2


# =============================================================================
# StreamlitAdapter Service Wrapper Tests (with mocking)
# =============================================================================

class TestDetectFormat:
    """Tests for detect_format wrapper."""

    def test_detect_lipidsearch_format(self):
        """Test detecting LipidSearch format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1e6],
        })

        result = StreamlitAdapter.detect_format(df)

        assert result == DataFormat.LIPIDSEARCH

    def test_detect_msdial_format(self):
        """Test detecting MS-DIAL format."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 32:1'],
            'Ontology': ['PC'],
            'Average Mz': [760.5],
            'Total score': [90.0],
            's1': [1e6],
        })

        result = StreamlitAdapter.detect_format(df)

        assert result == DataFormat.MSDIAL

    def test_detect_generic_format(self):
        """Test detecting Generic format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'Sample1': [1e6],
        })

        result = StreamlitAdapter.detect_format(df)

        assert result == DataFormat.GENERIC

    def test_detect_unknown_format(self):
        """Test detecting Unknown format."""
        df = pd.DataFrame({
            'random_column': [1, 2, 3],
        })

        result = StreamlitAdapter.detect_format(df)

        assert result == DataFormat.UNKNOWN


# =============================================================================
# SessionState Integration Tests (with mocked st.session_state)
# =============================================================================

class TestSessionStateIntegration:
    """Integration tests for session state management with mocked Streamlit."""

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_session_state(self, mock_st):
        """Test session state initialization."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.initialize_session_state()

        # Should initialize all expected keys
        assert 'raw_df' in mock_st.session_state
        assert 'cleaned_df' in mock_st.session_state
        assert 'experiment' in mock_st.session_state
        assert 'page' in mock_st.session_state
        assert mock_st.session_state['page'] == 'landing'

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_preserves_existing_values(self, mock_st):
        """Test that initialization preserves existing values."""
        mock_st.session_state = MockSessionState({'page': 'analysis'})

        StreamlitAdapter.initialize_session_state()

        # Should preserve existing value
        assert mock_st.session_state['page'] == 'analysis'

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_data_state(self, mock_st):
        """Test resetting data state."""
        mock_st.session_state = MockSessionState({
            'raw_df': pd.DataFrame({'A': [1]}),
            'cleaned_df': pd.DataFrame({'B': [2]}),
            'experiment': ExperimentConfig(
                n_conditions=1,
                conditions_list=['A'],
                number_of_samples_list=[1]
            ),
            'confirmed': True,
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['raw_df'] is None
        assert mock_st.session_state['cleaned_df'] is None
        assert mock_st.session_state['experiment'] is None
        assert mock_st.session_state['confirmed'] is False

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_creates_analysis_keys(self, mock_st):
        """Test that initialization creates all analysis session state keys."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.initialize_session_state()

        analysis_keys = [
            'analysis_selection', 'analysis_bar_chart_fig',
            'analysis_pie_chart_figs', 'analysis_saturation_figs',
            'analysis_fach_fig', 'analysis_pathway_fig',
            'analysis_volcano_fig', 'analysis_volcano_data',
            'analysis_heatmap_fig', 'analysis_heatmap_clusters',
            'analysis_all_plots',
        ]
        for key in analysis_keys:
            assert key in mock_st.session_state, f"Missing analysis key: {key}"

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_analysis_none_defaults(self, mock_st):
        """Test that analysis None-defaulting keys are initialized to None."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.initialize_session_state()

        assert mock_st.session_state['analysis_selection'] is None
        assert mock_st.session_state['analysis_bar_chart_fig'] is None
        assert mock_st.session_state['analysis_fach_fig'] is None
        assert mock_st.session_state['analysis_pathway_fig'] is None
        assert mock_st.session_state['analysis_volcano_fig'] is None
        assert mock_st.session_state['analysis_volcano_data'] is None
        assert mock_st.session_state['analysis_heatmap_fig'] is None
        assert mock_st.session_state['analysis_heatmap_clusters'] is None

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_analysis_dict_defaults(self, mock_st):
        """Test that analysis dict-defaulting keys are initialized to empty dicts."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.initialize_session_state()

        assert mock_st.session_state['analysis_pie_chart_figs'] == {}
        assert mock_st.session_state['analysis_saturation_figs'] == {}
        assert mock_st.session_state['analysis_all_plots'] == {}

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_preserves_existing_analysis_values(self, mock_st):
        """Test that initialization preserves existing analysis values."""
        mock_fig = MagicMock()
        mock_st.session_state = MockSessionState({
            'analysis_selection': 'Volcano Plot',
            'analysis_volcano_fig': mock_fig,
        })

        StreamlitAdapter.initialize_session_state()

        assert mock_st.session_state['analysis_selection'] == 'Volcano Plot'
        assert mock_st.session_state['analysis_volcano_fig'] is mock_fig

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_clears_analysis_state(self, mock_st):
        """Test that reset_data_state clears all analysis session state keys."""
        mock_fig = MagicMock()
        mock_st.session_state = MockSessionState({
            'analysis_selection': 'Bar Chart',
            'analysis_bar_chart_fig': mock_fig,
            'analysis_pie_chart_figs': {'Ctrl': mock_fig},
            'analysis_saturation_figs': {'PC': mock_fig},
            'analysis_fach_fig': mock_fig,
            'analysis_pathway_fig': mock_fig,
            'analysis_volcano_fig': mock_fig,
            'analysis_volcano_data': {'some': 'data'},
            'analysis_heatmap_fig': mock_fig,
            'analysis_heatmap_clusters': pd.DataFrame({'C': [1]}),
            'analysis_all_plots': {'bar': mock_fig},
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['analysis_selection'] is None
        assert mock_st.session_state['analysis_bar_chart_fig'] is None
        assert mock_st.session_state['analysis_pie_chart_figs'] == {}
        assert mock_st.session_state['analysis_saturation_figs'] == {}
        assert mock_st.session_state['analysis_fach_fig'] is None
        assert mock_st.session_state['analysis_pathway_fig'] is None
        assert mock_st.session_state['analysis_volcano_fig'] is None
        assert mock_st.session_state['analysis_volcano_data'] is None
        assert mock_st.session_state['analysis_heatmap_fig'] is None
        assert mock_st.session_state['analysis_heatmap_clusters'] is None
        assert mock_st.session_state['analysis_all_plots'] == {}

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_preserves_page_and_module(self, mock_st):
        """Test that reset_data_state preserves page and module."""
        mock_st.session_state = MockSessionState({
            'page': 'app',
            'module': 'Quality Check & Analysis',
            'analysis_selection': 'Bar Chart',
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['page'] == 'app'
        assert mock_st.session_state['module'] == 'Quality Check & Analysis'
        assert mock_st.session_state['analysis_selection'] is None

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_removes_analysis_widget_keys(self, mock_st):
        """Test that reset_data_state removes analysis widget keys."""
        mock_st.session_state = MockSessionState({
            'analysis_radio': 'Abundance Bar Chart',
            'bar_conditions': ['Control'],
            'volcano_p_threshold': 0.05,
            'heatmap_n_clusters': 3,
        })

        StreamlitAdapter.reset_data_state()

        assert 'analysis_radio' not in mock_st.session_state
        assert 'bar_conditions' not in mock_st.session_state
        assert 'volcano_p_threshold' not in mock_st.session_state
        assert 'heatmap_n_clusters' not in mock_st.session_state

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_removes_analysis_dynamic_keys(self, mock_st):
        """Test that reset_data_state removes analysis dynamic widget keys."""
        mock_st.session_state = MockSessionState({
            'volcano_label_x_PC(16:0_18:1)': 1.5,
            'volcano_label_y_PC(16:0_18:1)': -2.3,
            'analysis_svg_bar_chart': 'key1',
            'analysis_csv_abundance': 'key2',
        })

        StreamlitAdapter.reset_data_state()

        assert 'volcano_label_x_PC(16:0_18:1)' not in mock_st.session_state
        assert 'volcano_label_y_PC(16:0_18:1)' not in mock_st.session_state
        assert 'analysis_svg_bar_chart' not in mock_st.session_state
        assert 'analysis_csv_abundance' not in mock_st.session_state

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_removes_mixed_dynamic_keys(self, mock_st):
        """Test that reset removes both old and new dynamic keys together."""
        mock_st.session_state = MockSessionState({
            'protein_sample1': 5.0,          # existing prefix
            'conc_IS1': 100.0,               # existing prefix
            'volcano_label_x_lipid1': 1.0,   # new prefix
            'analysis_svg_volcano': 'key',   # new prefix
        })

        StreamlitAdapter.reset_data_state()

        assert 'protein_sample1' not in mock_st.session_state
        assert 'conc_IS1' not in mock_st.session_state
        assert 'volcano_label_x_lipid1' not in mock_st.session_state
        assert 'analysis_svg_volcano' not in mock_st.session_state


# =============================================================================
# Widget Preservation Tests
# =============================================================================

class TestWidgetPreservation:
    """Tests for restore_widget_value and save_widget_value."""

    @patch('app.adapters.streamlit_adapter.st')
    def test_restore_from_widget_key(self, mock_st):
        """Test restore returns widget key value when present."""
        mock_st.session_state = MockSessionState({'my_widget': 'current_value'})

        result = StreamlitAdapter.restore_widget_value('my_widget', '_preserved_my_widget', 'default')

        assert result == 'current_value'

    @patch('app.adapters.streamlit_adapter.st')
    def test_restore_from_preserved_key(self, mock_st):
        """Test restore falls back to preserved key when widget key absent."""
        mock_st.session_state = MockSessionState({'_preserved_my_widget': 'saved_value'})

        result = StreamlitAdapter.restore_widget_value('my_widget', '_preserved_my_widget', 'default')

        assert result == 'saved_value'

    @patch('app.adapters.streamlit_adapter.st')
    def test_restore_returns_default(self, mock_st):
        """Test restore returns default when neither key exists."""
        mock_st.session_state = MockSessionState()

        result = StreamlitAdapter.restore_widget_value('my_widget', '_preserved_my_widget', 'default')

        assert result == 'default'

    @patch('app.adapters.streamlit_adapter.st')
    def test_save_widget_value(self, mock_st):
        """Test save copies widget value to preserved key."""
        mock_st.session_state = MockSessionState({'my_widget': 'new_value'})

        StreamlitAdapter.save_widget_value('my_widget', '_preserved_my_widget')

        assert mock_st.session_state['_preserved_my_widget'] == 'new_value'

    @patch('app.adapters.streamlit_adapter.st')
    def test_save_widget_value_none_skips(self, mock_st):
        """Test save does not write None to preserved key."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.save_widget_value('my_widget', '_preserved_my_widget')

        assert '_preserved_my_widget' not in mock_st.session_state


# =============================================================================
# SessionState Field Completeness Tests
# =============================================================================

class TestSessionStateFieldCompleteness:
    """Tests ensuring all expected fields exist in SessionState."""

    def test_all_analysis_fields_exist(self):
        """Test that all 11 analysis fields exist in SessionState."""
        state = SessionState()
        analysis_fields = [
            'analysis_selection', 'analysis_bar_chart_fig',
            'analysis_pie_chart_figs', 'analysis_saturation_figs',
            'analysis_fach_fig', 'analysis_pathway_fig',
            'analysis_volcano_fig', 'analysis_volcano_data',
            'analysis_heatmap_fig', 'analysis_heatmap_clusters',
            'analysis_all_plots',
        ]
        for field_name in analysis_fields:
            assert hasattr(state, field_name), f"Missing analysis field: {field_name}"

    def test_analysis_field_count(self):
        """Test total number of analysis fields (guard against accidental deletion)."""
        field_names = [f.name for f in fields(SessionState)]
        analysis_fields = [f for f in field_names if f.startswith('analysis_')]
        assert len(analysis_fields) == 17

    def test_qc_field_count_unchanged(self):
        """Test that QC fields count hasn't changed (no regression)."""
        field_names = [f.name for f in fields(SessionState)]
        qc_fields = [f for f in field_names if f.startswith('qc_')]
        assert len(qc_fields) == 9

    def test_total_field_count(self):
        """Test total SessionState field count matches expected."""
        field_names = [f.name for f in fields(SessionState)]
        # 61 (pre-Module 3) + 17 (analysis) + 3 (QC plot storage) + 2 (LSI report) = 83
        assert len(field_names) == 83

    def test_asdict_includes_analysis_fields(self):
        """Test that asdict() includes all analysis fields."""
        state = SessionState()
        state_dict = asdict(state)
        assert 'analysis_selection' in state_dict
        assert 'analysis_all_plots' in state_dict
        assert 'analysis_heatmap_clusters' in state_dict

    def test_widget_keys_count_includes_analysis(self):
        """Test that _WIDGET_KEYS count includes analysis keys."""
        analysis_widget_keys = {k for k in _WIDGET_KEYS if any(
            k.startswith(p) for p in ('analysis_', 'bar_', 'pie_', 'sat_',
                                       'fach_', 'pathway_', 'volcano_', 'heatmap_')
        )}
        # 1 (analysis_radio) + 5 (bar) + 1 (pie) + 6 (sat) + 2 (fach)
        # + 8 (pathway) + 10 (volcano) + 5 (heatmap) = 38
        assert len(analysis_widget_keys) == 38

    def test_dynamic_prefix_count_includes_analysis(self):
        """Test that _DYNAMIC_KEY_PREFIXES count includes analysis prefixes."""
        # 9 (pre-Module 3) + 4 (analysis) = 13
        assert len(_DYNAMIC_KEY_PREFIXES) == 13


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for StreamlitAdapter."""

    def test_experiment_hash_single_condition(self):
        """Test hashing experiment with single condition."""
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Single'],
            number_of_samples_list=[1]
        )
        assert isinstance(hash(experiment), int)

    def test_detect_format_empty_dataframe(self):
        """Test format detection with empty DataFrame."""
        df = pd.DataFrame()
        result = StreamlitAdapter.detect_format(df)
        assert result == DataFormat.UNKNOWN

    def test_detect_format_single_row(self):
        """Test format detection with single row."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'Sample1': [1e6],
        })
        result = StreamlitAdapter.detect_format(df)
        assert result == DataFormat.GENERIC

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_with_no_dynamic_keys(self, mock_st):
        """Test reset works when no dynamic keys exist."""
        mock_st.session_state = MockSessionState({'page': 'app', 'module': 'M1'})

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['page'] == 'app'

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_with_many_dynamic_keys(self, mock_st):
        """Test reset handles many dynamic keys efficiently."""
        state = MockSessionState()
        for i in range(50):
            state[f'volcano_label_x_lipid_{i}'] = float(i)
            state[f'volcano_label_y_lipid_{i}'] = float(i)
            state[f'analysis_svg_plot_{i}'] = f'key_{i}'
            state[f'analysis_csv_data_{i}'] = f'key_{i}'
        mock_st.session_state = state

        StreamlitAdapter.reset_data_state()

        remaining_dynamic = [k for k in mock_st.session_state
                             if isinstance(k, str) and k.startswith(_DYNAMIC_KEY_PREFIXES)]
        assert remaining_dynamic == []

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_does_not_remove_non_prefixed_keys(self, mock_st):
        """Test reset doesn't remove keys that don't match any prefix or widget key."""
        mock_st.session_state = MockSessionState({
            'custom_user_key': 'keep_me',
            'unrelated_data': 42,
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state.get('custom_user_key') == 'keep_me'
        assert mock_st.session_state.get('unrelated_data') == 42

    def test_session_state_equality(self):
        """Test that two default SessionState instances are equal."""
        state1 = SessionState()
        state2 = SessionState()
        assert state1 == state2

    def test_session_state_inequality_analysis_selection(self):
        """Test that SessionState with different analysis_selection are not equal."""
        state1 = SessionState(analysis_selection='Bar Chart')
        state2 = SessionState(analysis_selection='Volcano Plot')
        assert state1 != state2

    def test_session_state_inequality_analysis_figs(self):
        """Test that SessionState with different analysis figs are not equal."""
        state1 = SessionState(analysis_pie_chart_figs={'A': 'fig1'})
        state2 = SessionState(analysis_pie_chart_figs={'B': 'fig2'})
        assert state1 != state2
