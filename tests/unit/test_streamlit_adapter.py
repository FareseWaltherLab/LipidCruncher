"""
Unit tests for StreamlitAdapter.

Tests the adapter layer that bridges UI and services.
Focuses on testable components: SessionState dataclass, utility methods.

Note: Methods with @st.cache_data decorators and session state access
require mocking or integration tests with Streamlit.
"""
import pandas as pd
from unittest.mock import MagicMock, patch
from dataclasses import asdict, fields

from app.adapters.streamlit_adapter import (
    SessionState,
    StreamlitAdapter,
    _ROUTING_FIELDS,
    _ROUTING_WIDGETS,
    _STAGES,
)
from app.constants import MODULE_DATA_PROCESSING
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
        assert state.selected_classes == []

    def test_default_grade_config(self):
        """Test default grade configuration."""
        state = SessionState()
        assert state.grade_config is None

    def test_default_msdial_config(self):
        """Test default MS-DIAL configuration."""
        state = SessionState()
        assert state.msdial_features is None
        assert state.msdial_use_normalized is False
        assert state.msdial_data_type_index == 0

    def test_default_standards_settings(self):
        """Test default standards settings."""
        state = SessionState()
        assert state.original_auto_intsta_df is None

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
        assert state.analysis_volcano_fig is None
        assert state.analysis_volcano_data is None
        assert state.analysis_heatmap_fig is None
        assert state.analysis_heatmap_clusters is None

    def test_default_analysis_dict_fields(self):
        """Test that analysis dict fields default to empty dicts."""
        state = SessionState()
        assert state.analysis_all_plots == {}

    def test_default_analysis_dict_fields_are_dicts(self):
        """Test that analysis dict fields are actually dict instances."""
        state = SessionState()
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

        state1.lsi_manual_fields['key'] = 'value'

        # state2 should not be affected
        assert 'key' not in state2.lsi_manual_fields

    def test_list_defaults_independent(self):
        """Test that list defaults are independent per instance."""
        state1 = SessionState()
        state2 = SessionState()

        state1.selected_classes.append('PC')

        # state2 should not be affected
        assert 'PC' not in state2.selected_classes


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
# Stage registry (which stage owns each session key)
# =============================================================================

def _stage_keys(stage):
    return stage.fields | stage.widgets | stage.plain


class TestStageRegistry:
    """Every session key belongs to one stage, so invalidate_from() clears
    it at the right time. This replaced four hand-kept lists that had
    drifted: 37 rendered widget keys were in none of them."""

    def test_every_field_belongs_to_exactly_one_stage(self):
        owners = {}
        for stage in _STAGES:
            for key in stage.fields | stage.widgets:
                owners.setdefault(key, []).append(stage.name)
        for name in _ROUTING_FIELDS:
            owners.setdefault(name, []).append('routing')
        field_names = {f.name for f in fields(SessionState)}
        unowned = sorted(n for n in field_names if n not in owners)
        assert unowned == [], f"SessionState fields in no stage: {unowned}"
        doubled = {k: v for k, v in owners.items() if len(v) > 1}
        assert doubled == {}

    def test_no_key_is_in_two_stages(self):
        seen = {}
        for stage in _STAGES:
            for key in _stage_keys(stage):
                assert key not in seen, f"{key} in {seen.get(key)} and {stage.name}"
                seen[key] = stage.name

    def test_stage_fields_are_session_state_fields(self):
        field_names = {f.name for f in fields(SessionState)}
        for stage in _STAGES:
            assert stage.fields <= field_names, stage.fields - field_names

    def test_fields_written_by_widgets_are_listed_as_widgets(self):
        """norm_method_selection and protein_input_method are both fields and
        widget keys: they must be popped, not assigned, on a reset."""
        field_names = {f.name for f in fields(SessionState)}
        widget_fields = {k for s in _STAGES for k in s.widgets} & field_names
        assert widget_fields == {'norm_method_selection', 'protein_input_method'}

    def test_prefixes_end_with_underscore(self):
        for stage in _STAGES:
            for prefix in stage.prefixes:
                assert prefix.endswith('_'), prefix

    def test_every_literal_widget_key_in_the_ui_is_registered(self):
        """Scans the UI for key='...' literals: a widget keyed outside the
        registry would survive every reset."""
        import ast
        from pathlib import Path

        src = Path(__file__).parents[2] / 'src'
        files = list((src / 'app' / 'ui').rglob('*.py')) + [src / 'main_app.py']
        registered = set(_ROUTING_WIDGETS)
        prefixes = ()
        for stage in _STAGES:
            registered |= _stage_keys(stage)
            prefixes += stage.prefixes
        unregistered = []
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.keyword) and node.arg == 'key'
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    key = node.value.value
                    if key not in registered and not key.startswith(prefixes):
                        unregistered.append(f"{path.name}:{key}")
        assert unregistered == []


class TestInvalidateFrom:
    """invalidate_from(stage) clears that stage and every later one."""

    @patch('app.adapters.streamlit_adapter.st')
    def test_clears_later_stages_and_keeps_earlier_ones(self, mock_st):
        mock_st.session_state = MockSessionState({
            'raw_df': pd.DataFrame({'a': [1]}),          # upload
            'normalized_df': pd.DataFrame({'b': [1]}),   # processing
            'lsi_manual_fields': {'Ion mode': 'neg'},    # report
            'qc_pca_plot': object(),                     # qc
            'pca_samples_remove': ['s1'],                # qc widget
            'qc_rt_svg_individual_PC': b'svg',           # qc prefix
            'analysis_volcano_data': object(),           # analysis
            'volcano_label_x_PC': 1.0,                   # analysis prefix
            'page': 'app',                               # routing
        })

        StreamlitAdapter.invalidate_from('qc')

        state = mock_st.session_state
        assert state['raw_df'] is not None
        assert state['normalized_df'] is not None
        assert state['lsi_manual_fields'] == {'Ion mode': 'neg'}
        assert state['qc_pca_plot'] is None
        assert state['analysis_volcano_data'] is None
        for popped in ('pca_samples_remove', 'qc_rt_svg_individual_PC',
                       'volcano_label_x_PC'):
            assert popped not in state
        assert state['page'] == 'app'

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_data_state_keeps_only_the_page(self, mock_st):
        mock_st.session_state = MockSessionState({
            'page': 'app', 'module': 'Quality Check & Analysis',
            'raw_df': pd.DataFrame({'a': [1]}), '_dataset_source_id': 'x',
            'lsi_manual_fields': {'Ion mode': 'neg'},
        })

        StreamlitAdapter.reset_data_state()

        state = mock_st.session_state
        assert state['page'] == 'app'
        assert state['module'] == MODULE_DATA_PROCESSING
        assert state['raw_df'] is None
        assert '_dataset_source_id' not in state
        assert state['lsi_manual_fields'] == {}


class TestNavigationActions:
    """Next / Back / Home live in the adapter, so the app and the UI test
    scripts run the same code (the scripts used to copy main_app's resets,
    and their copies had drifted)."""

    def _state(self):
        return MockSessionState({
            'page': 'app', 'module': MODULE_DATA_PROCESSING,
            'normalized_df': pd.DataFrame({'b': [1]}),
            'qc_input_frame': object(),
            'analysis_volcano_fig': object(),
        })

    @patch('app.adapters.streamlit_adapter.st')
    def test_next_starts_qc_over(self, mock_st):
        mock_st.session_state = self._state()
        StreamlitAdapter.go_to_quality_check()
        state = mock_st.session_state
        assert state['module'] == 'Quality Check & Analysis'
        assert state['qc_input_frame'] is None
        assert state['analysis_volcano_fig'] is None
        assert state['normalized_df'] is not None

    @patch('app.adapters.streamlit_adapter.st')
    def test_back_returns_to_module1_and_clears_qc(self, mock_st):
        mock_st.session_state = self._state()
        mock_st.session_state['module'] = 'Quality Check & Analysis'
        StreamlitAdapter.back_to_data_processing()
        state = mock_st.session_state
        assert state['module'] == MODULE_DATA_PROCESSING
        assert state['qc_input_frame'] is None
        assert state['normalized_df'] is not None

    @patch('app.adapters.streamlit_adapter.st')
    def test_home_drops_the_dataset(self, mock_st):
        mock_st.session_state = self._state()
        StreamlitAdapter.back_to_home()
        state = mock_st.session_state
        assert state['page'] == 'landing'
        assert state['normalized_df'] is None


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
    def test_reset_to_experiment_setup_keeps_upload_clears_downstream(self, mock_st):
        """Soft reset keeps the upload/standardization, clears everything else."""
        mock_st.session_state = MockSessionState({
            # Upload + standardization — must be preserved.
            'raw_df': pd.DataFrame({'A': [1]}),
            'standardized_df': pd.DataFrame({'intensity[s1]': [1]}),
            'column_mapping': pd.DataFrame({'standardized_name': ['intensity[s1]']}),
            'format_type': 'MS-DIAL',
            'using_sample_data': True,
            'lipidsearch_sample_names': {'s1': 'ID_01'},
            # Downstream state — must be cleared.
            'experiment': ExperimentConfig(
                n_conditions=1, conditions_list=['A'], number_of_samples_list=[1],
            ),
            'confirmed': True,
            'grouping_complete': True,
            'sample_names': {'s1': 'liver'},
            'cleaned_df': pd.DataFrame({'B': [2]}),
            'normalized_df': pd.DataFrame({'C': [3]}),
            'qc_continuation_df': pd.DataFrame({'D': [4]}),
            'module': 'Quality Check & Analysis',
            # Widget/dynamic keys that should be popped.
            'select_A': ['s1'],
            'confirm_checkbox': True,
        })

        StreamlitAdapter.reset_to_experiment_setup()

        # Preserved: upload + standardization.
        assert mock_st.session_state['raw_df'] is not None
        assert mock_st.session_state['standardized_df'] is not None
        assert mock_st.session_state['column_mapping'] is not None
        assert mock_st.session_state['format_type'] == 'MS-DIAL'
        assert mock_st.session_state['using_sample_data'] is True
        # Alignment-derived names survive so sample_names can be re-seeded.
        assert mock_st.session_state['lipidsearch_sample_names'] == {'s1': 'ID_01'}

        # Cleared: experiment + all downstream.
        assert mock_st.session_state['experiment'] is None
        assert mock_st.session_state['confirmed'] is False
        assert mock_st.session_state['sample_names'] is None
        assert mock_st.session_state['cleaned_df'] is None
        assert mock_st.session_state['normalized_df'] is None
        assert mock_st.session_state['qc_continuation_df'] is None
        assert 'select_A' not in mock_st.session_state
        # Confirm checkbox is forced off (not merely popped) so a pre-fill rerun
        # that re-creates the widget cannot restore its cached 'checked' value.
        assert mock_st.session_state['confirm_checkbox'] is False

        # Returned to Module 1 (Define Experiment step) and cache cleared.
        assert mock_st.session_state['module'] == MODULE_DATA_PROCESSING
        mock_st.cache_data.clear.assert_called_once()

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_to_experiment_setup_keeps_the_upload_on_the_next_run(self, mock_st):
        """Regression: the soft reset dropped _dataset_source_id, so on the
        next rerun the still-uploaded file looked like a new source and
        reset_for_new_source wiped the data the soft reset had kept."""
        standardized = pd.DataFrame({'intensity[s1]': [1]})
        mock_st.session_state = MockSessionState({
            'raw_df': pd.DataFrame({'A': [1]}),
            'standardized_df': standardized,
            '_dataset_source_id': 'upload:file-1',
        })

        StreamlitAdapter.reset_to_experiment_setup()
        wiped = StreamlitAdapter.reset_for_new_source('upload:file-1')

        assert wiped is False
        assert mock_st.session_state['standardized_df'] is standardized

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_to_experiment_setup_undoes_confirmed_regroup(self, mock_st):
        """Regression: after a confirmed regroup the soft reset kept the
        renumbered standardized_df, so re-seeded names landed on the wrong data."""
        original = pd.DataFrame({'intensity[s1]': [1.0], 'intensity[s2]': [2.0]})
        regrouped = pd.DataFrame({'intensity[s2]': [1.0], 'intensity[s1]': [2.0]})
        mock_st.session_state = MockSessionState({
            'standardized_df': regrouped,
            '_pre_regroup_df': original,
            '_pre_regroup_sample_names': {'s1': 'alpha', 's2': 'beta'},
        })

        StreamlitAdapter.reset_to_experiment_setup()

        assert mock_st.session_state['standardized_df'] is original
        assert '_pre_regroup_df' not in mock_st.session_state
        assert '_pre_regroup_sample_names' not in mock_st.session_state

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_data_state_drops_regroup_snapshot(self, mock_st):
        """Regression: a hard reset (Back to Home, Clear & Upload) kept the
        regroup snapshot, so the next upload's "Yes" grouping restored the
        previous dataset's names, and those names reach the CSV downloads."""
        mock_st.session_state = MockSessionState({
            '_pre_regroup_df': pd.DataFrame({'intensity[s1]': [1.0]}),
            '_pre_regroup_sample_names': {'s1': 'old file sample'},
        })

        StreamlitAdapter.reset_data_state()

        assert '_pre_regroup_df' not in mock_st.session_state
        assert '_pre_regroup_sample_names' not in mock_st.session_state

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_data_state_rearms_alignment_prefill(self, mock_st):
        """Regression: after Start Over the experiment pre-fill never fired
        again for the same Alignment Setting file, leaving default conditions."""
        mock_st.session_state = MockSessionState({
            '_ls_align_prefilled_for': 'alignment text',
        })

        StreamlitAdapter.reset_data_state()

        assert '_ls_align_prefilled_for' not in mock_st.session_state

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_data_state_clears_pca_group_exclusion(self, mock_st):
        """Regression: the PCA group exclusion lived outside SessionState, so
        no reset cleared it and it re-applied to the next dataset with a
        same-named condition."""
        mock_st.session_state = MockSessionState({
            '_preserved_pca_groups_remove': ['Blank'],
            '_preserved_pca_samples_remove': ['s1'],
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['_preserved_pca_groups_remove'] == []
        assert mock_st.session_state['_preserved_pca_samples_remove'] == []

    @patch('app.adapters.streamlit_adapter.st')
    def test_qc_invalidation_clears_pca_group_exclusion(self, mock_st):
        """Next / Back clear the QC stage, including the PCA group exclusion."""
        mock_st.session_state = MockSessionState({
            '_preserved_pca_groups_remove': ['Blank'],
        })

        StreamlitAdapter.invalidate_from('qc')

        assert mock_st.session_state['_preserved_pca_groups_remove'] == []

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_for_new_source_resets_loaded_dataset(self, mock_st):
        """Regression: uploading a different file replaced raw_df only, so
        every downstream frame still came from the previous dataset."""
        mock_st.session_state = MockSessionState({
            'raw_df': pd.DataFrame({'A': [1]}),
            'cleaned_df': pd.DataFrame({'B': [2]}),
            'normalized_df': pd.DataFrame({'C': [3]}),
            '_dataset_source_id': 'upload:first',
        })

        assert StreamlitAdapter.reset_for_new_source('upload:second') is True

        assert mock_st.session_state['raw_df'] is None
        assert mock_st.session_state['cleaned_df'] is None
        assert mock_st.session_state['normalized_df'] is None
        assert mock_st.session_state['_dataset_source_id'] == 'upload:second'
        mock_st.cache_data.clear.assert_called_once()

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_for_new_source_same_source_is_noop(self, mock_st):
        """The same upload survives every rerun; it must not reset anything."""
        df = pd.DataFrame({'A': [1]})
        mock_st.session_state = MockSessionState({
            'raw_df': df,
            'cleaned_df': pd.DataFrame({'B': [2]}),
            '_dataset_source_id': 'upload:first',
        })

        assert StreamlitAdapter.reset_for_new_source('upload:first') is False

        assert mock_st.session_state['raw_df'] is df
        assert mock_st.session_state['cleaned_df'] is not None
        mock_st.cache_data.clear.assert_not_called()

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_for_new_source_first_upload_only_records_source(self, mock_st):
        mock_st.session_state = MockSessionState({'raw_df': None})

        assert StreamlitAdapter.reset_for_new_source('upload:first') is False

        assert mock_st.session_state['_dataset_source_id'] == 'upload:first'
        mock_st.cache_data.clear.assert_not_called()

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_data_state_forgets_dataset_source(self, mock_st):
        mock_st.session_state = MockSessionState({'_dataset_source_id': 'upload:x'})

        StreamlitAdapter.reset_data_state()

        assert '_dataset_source_id' not in mock_st.session_state

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_creates_analysis_keys(self, mock_st):
        """Test that initialization creates all analysis session state keys."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.initialize_session_state()

        analysis_keys = [
            'analysis_selection',
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
        assert mock_st.session_state['analysis_volcano_fig'] is None
        assert mock_st.session_state['analysis_volcano_data'] is None
        assert mock_st.session_state['analysis_heatmap_fig'] is None
        assert mock_st.session_state['analysis_heatmap_clusters'] is None

    @patch('app.adapters.streamlit_adapter.st')
    def test_initialize_analysis_dict_defaults(self, mock_st):
        """Test that analysis dict-defaulting keys are initialized to empty dicts."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.initialize_session_state()

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
            'analysis_volcano_fig': mock_fig,
            'analysis_volcano_data': {'some': 'data'},
            'analysis_heatmap_fig': mock_fig,
            'analysis_heatmap_clusters': pd.DataFrame({'C': [1]}),
            'analysis_all_plots': {'bar': mock_fig},
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['analysis_selection'] is None
        assert mock_st.session_state['analysis_volcano_fig'] is None
        assert mock_st.session_state['analysis_volcano_data'] is None
        assert mock_st.session_state['analysis_heatmap_fig'] is None
        assert mock_st.session_state['analysis_heatmap_clusters'] is None
        assert mock_st.session_state['analysis_all_plots'] == {}

    @patch('app.adapters.streamlit_adapter.st')
    def test_reset_preserves_page_and_returns_to_module1(self, mock_st):
        """reset_data_state keeps the page but sends the module back to Module 1."""
        mock_st.session_state = MockSessionState({
            'page': 'app',
            'module': 'Quality Check & Analysis',
            'analysis_selection': 'Bar Chart',
        })

        StreamlitAdapter.reset_data_state()

        assert mock_st.session_state['page'] == 'app'
        assert mock_st.session_state['module'] == MODULE_DATA_PROCESSING
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
            'analysis_selection',
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
        # 17 - 1 (analysis_pathway_active_classes: nothing read or wrote it)
        # - the write-only per-page figure copies (the report reads
        #   analysis_all_plots): bar chart, pie charts, saturation, chain length,
        #   FACH, pathway
        assert len(analysis_fields) == 10

    def test_qc_field_count_unchanged(self):
        """Test that QC fields count hasn't changed (no regression)."""
        field_names = [f.name for f in fields(SessionState)]
        qc_fields = [f for f in field_names if f.startswith('qc_')]
        # 9 + qc_input_frame and qc_input_bqc_label (frozen at Module 1's
        # handoff) + qc_frame (the samples left after PCA exclusion)
        assert len(qc_fields) == 12

    def test_asdict_includes_analysis_fields(self):
        """Test that asdict() includes all analysis fields."""
        state = SessionState()
        state_dict = asdict(state)
        assert 'analysis_selection' in state_dict
        assert 'analysis_all_plots' in state_dict
        assert 'analysis_heatmap_clusters' in state_dict


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

        prefixes = ('volcano_label_x_', 'volcano_label_y_', 'analysis_svg_',
                    'analysis_csv_')
        remaining_dynamic = [k for k in mock_st.session_state
                             if isinstance(k, str) and k.startswith(prefixes)]
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
        state1 = SessionState(analysis_all_plots={'A': 'fig1'})
        state2 = SessionState(analysis_all_plots={'B': 'fig2'})
        assert state1 != state2
