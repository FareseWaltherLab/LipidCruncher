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
from app.services.zero_filtering import ZeroFilterConfig


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
        assert state.grade_filter_mode_saved == 0
        assert state.grade_selections_saved == {}

    def test_default_msdial_config(self):
        """Test default MS-DIAL configuration."""
        state = SessionState()
        assert state.msdial_quality_config is None
        assert state.msdial_features == {}
        assert state.msdial_use_normalized is False
        assert state.msdial_data_type_index == 0
        assert state.msdial_quality_level_index == 1

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

class TestComputeDfHash:
    """Tests for compute_df_hash utility method."""

    def test_hash_simple_dataframe(self):
        """Test hashing a simple DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 32  # MD5 hash length

    def test_same_df_same_hash(self):
        """Test that identical DataFrames produce same hash."""
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        hash1 = StreamlitAdapter.compute_df_hash(df1)
        hash2 = StreamlitAdapter.compute_df_hash(df2)

        assert hash1 == hash2

    def test_different_df_different_hash(self):
        """Test that different DataFrames produce different hashes."""
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        df2 = pd.DataFrame({'A': [1, 2, 4]})  # Different value

        hash1 = StreamlitAdapter.compute_df_hash(df1)
        hash2 = StreamlitAdapter.compute_df_hash(df2)

        assert hash1 != hash2

    def test_different_columns_different_hash(self):
        """Test that different column names produce different hashes."""
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        df2 = pd.DataFrame({'B': [1, 2, 3]})  # Different column name

        hash1 = StreamlitAdapter.compute_df_hash(df1)
        hash2 = StreamlitAdapter.compute_df_hash(df2)

        assert hash1 != hash2

    def test_different_shape_different_hash(self):
        """Test that different shapes produce different hashes."""
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        df2 = pd.DataFrame({'A': [1, 2, 3, 4]})  # Different length

        hash1 = StreamlitAdapter.compute_df_hash(df1)
        hash2 = StreamlitAdapter.compute_df_hash(df2)

        assert hash1 != hash2

    def test_empty_dataframe_hash(self):
        """Test hashing an empty DataFrame."""
        df = pd.DataFrame()
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)

    def test_single_row_hash(self):
        """Test hashing a single-row DataFrame."""
        df = pd.DataFrame({'A': [1]})
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)

    def test_large_dataframe_hash(self):
        """Test hashing a large DataFrame is efficient."""
        df = pd.DataFrame({
            'A': range(10000),
            'B': range(10000),
            'C': range(10000),
        })
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)

    def test_hash_with_nan_values(self):
        """Test hashing DataFrame with NaN values."""
        df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)

    def test_hash_with_mixed_types(self):
        """Test hashing DataFrame with mixed types."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
        })
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)


class TestExperimentToDict:
    """Tests for experiment_to_dict utility method."""

    def test_basic_experiment_to_dict(self):
        """Test converting basic experiment to dict."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )

        result = StreamlitAdapter.experiment_to_dict(experiment)

        assert isinstance(result, dict)
        assert result['n_conditions'] == 2
        assert result['conditions_list'] == ['Control', 'Treatment']
        assert result['number_of_samples_list'] == [3, 3]

    def test_single_condition_to_dict(self):
        """Test converting single condition experiment to dict."""
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Samples'],
            number_of_samples_list=[5]
        )

        result = StreamlitAdapter.experiment_to_dict(experiment)

        assert result['n_conditions'] == 1
        assert len(result['conditions_list']) == 1

    def test_many_conditions_to_dict(self):
        """Test converting many conditions experiment to dict."""
        experiment = ExperimentConfig(
            n_conditions=5,
            conditions_list=['A', 'B', 'C', 'D', 'E'],
            number_of_samples_list=[1, 2, 3, 4, 5]
        )

        result = StreamlitAdapter.experiment_to_dict(experiment)

        assert result['n_conditions'] == 5
        assert len(result['conditions_list']) == 5
        assert result['number_of_samples_list'] == [1, 2, 3, 4, 5]

    def test_dict_can_recreate_experiment(self):
        """Test that dict can be used to recreate ExperimentConfig."""
        original = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )

        result_dict = StreamlitAdapter.experiment_to_dict(original)
        recreated = ExperimentConfig(**result_dict)

        assert recreated.n_conditions == original.n_conditions
        assert recreated.conditions_list == original.conditions_list
        assert recreated.number_of_samples_list == original.number_of_samples_list


class TestConfigToDict:
    """Tests for config_to_dict utility method."""

    def test_none_config(self):
        """Test converting None config."""
        result = StreamlitAdapter.config_to_dict(None)
        assert result is None

    def test_grade_config_to_dict(self):
        """Test converting GradeFilterConfig to dict."""
        config = GradeFilterConfig(grade_config={'PC': ['A', 'B']})
        result = StreamlitAdapter.config_to_dict(config)

        assert isinstance(result, dict)
        assert 'grade_config' in result

    def test_quality_config_to_dict(self):
        """Test converting QualityFilterConfig to dict."""
        config = QualityFilterConfig(total_score_threshold=70, require_msms=True)
        result = StreamlitAdapter.config_to_dict(config)

        assert isinstance(result, dict)
        assert result['total_score_threshold'] == 70
        assert result['require_msms'] is True

    def test_zero_config_to_dict(self):
        """Test converting ZeroFilterConfig to dict."""
        config = ZeroFilterConfig(
            detection_threshold=0.1,
            bqc_threshold=0.3,
            non_bqc_threshold=0.5
        )
        result = StreamlitAdapter.config_to_dict(config)

        assert isinstance(result, dict)
        assert result['detection_threshold'] == 0.1
        assert result['bqc_threshold'] == 0.3
        assert result['non_bqc_threshold'] == 0.5


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
    def test_reset_normalization_state(self, mock_st):
        """Test resetting normalization state."""
        mock_st.session_state = MockSessionState({
            'normalization_method': 'Log2',
            'normalization_inputs': {'key': 'value'},
            'selected_classes': ['PC', 'PE'],
            'normalized_df': pd.DataFrame({'A': [1]}),
        })

        StreamlitAdapter.reset_normalization_state()

        assert mock_st.session_state['normalization_method'] == 'None'
        assert mock_st.session_state['normalization_inputs'] == {}
        assert mock_st.session_state['selected_classes'] == []
        assert mock_st.session_state['normalized_df'] is None


class TestSessionStateAccessors:
    """Tests for session state accessor methods."""

    @patch('app.adapters.streamlit_adapter.st')
    def test_get_experiment(self, mock_st):
        """Test getting experiment from session state."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )
        mock_st.session_state = MockSessionState({'experiment': experiment})

        result = StreamlitAdapter.get_experiment()

        assert result is experiment

    @patch('app.adapters.streamlit_adapter.st')
    def test_get_experiment_none(self, mock_st):
        """Test getting experiment when not set."""
        mock_st.session_state = MockSessionState()

        result = StreamlitAdapter.get_experiment()

        assert result is None

    @patch('app.adapters.streamlit_adapter.st')
    def test_set_experiment(self, mock_st):
        """Test setting experiment in session state."""
        mock_st.session_state = MockSessionState()
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )

        StreamlitAdapter.set_experiment(experiment)

        assert mock_st.session_state['experiment'] is experiment

    @patch('app.adapters.streamlit_adapter.st')
    def test_get_format_type(self, mock_st):
        """Test getting format type."""
        mock_st.session_state = MockSessionState({'format_type': DataFormat.LIPIDSEARCH})

        result = StreamlitAdapter.get_format_type()

        assert result == DataFormat.LIPIDSEARCH

    @patch('app.adapters.streamlit_adapter.st')
    def test_set_format_type(self, mock_st):
        """Test setting format type."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.set_format_type(DataFormat.MSDIAL)

        assert mock_st.session_state['format_type'] == DataFormat.MSDIAL

    @patch('app.adapters.streamlit_adapter.st')
    def test_get_cleaned_df(self, mock_st):
        """Test getting cleaned DataFrame."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        mock_st.session_state = MockSessionState({'cleaned_df': df})

        result = StreamlitAdapter.get_cleaned_df()

        assert result is df

    @patch('app.adapters.streamlit_adapter.st')
    def test_set_cleaned_df(self, mock_st):
        """Test setting cleaned DataFrame."""
        mock_st.session_state = MockSessionState()
        df = pd.DataFrame({'A': [1, 2, 3]})

        StreamlitAdapter.set_cleaned_df(df)

        assert mock_st.session_state['cleaned_df'] is df

    @patch('app.adapters.streamlit_adapter.st')
    def test_get_bqc_label(self, mock_st):
        """Test getting BQC label."""
        mock_st.session_state = MockSessionState({'bqc_label': 'BQC'})

        result = StreamlitAdapter.get_bqc_label()

        assert result == 'BQC'

    @patch('app.adapters.streamlit_adapter.st')
    def test_set_bqc_label(self, mock_st):
        """Test setting BQC label."""
        mock_st.session_state = MockSessionState()

        StreamlitAdapter.set_bqc_label('QC')

        assert mock_st.session_state['bqc_label'] == 'QC'


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Edge case tests for StreamlitAdapter."""

    def test_compute_hash_unicode_columns(self):
        """Test hash computation with unicode column names."""
        df = pd.DataFrame({'名前': [1, 2, 3], 'データ': [4, 5, 6]})
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)

    def test_compute_hash_special_characters(self):
        """Test hash computation with special characters in data."""
        df = pd.DataFrame({
            'A': ['test!@#$%', 'hello\nworld', 'tab\there'],
        })
        hash_val = StreamlitAdapter.compute_df_hash(df)
        assert isinstance(hash_val, str)

    def test_experiment_to_dict_with_empty_lists(self):
        """Test experiment to dict edge case."""
        # This would normally fail validation, but tests the method
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Single'],
            number_of_samples_list=[1]
        )
        result = StreamlitAdapter.experiment_to_dict(experiment)
        assert result['n_conditions'] == 1

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
