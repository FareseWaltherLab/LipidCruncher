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
from dataclasses import asdict

from app.adapters.streamlit_adapter import SessionState, StreamlitAdapter
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
        assert state.msdial_features == {}
        assert state.msdial_use_normalized is False
        assert state.msdial_data_type_index == 0

    def test_default_standards_settings(self):
        """Test default standards settings."""
        state = SessionState()
        assert state.original_auto_intsta_df is None
        assert state.preserved_intsta_df is None
        assert state.preserved_standards_mode is None


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
