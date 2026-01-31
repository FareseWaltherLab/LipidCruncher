"""
Unit tests for DataIngestionWorkflow.

Tests the complete data ingestion pipeline orchestration:
format detection → cleaning → zero filtering → standards extraction
"""
import pytest
import pandas as pd
import numpy as np

from app.workflows.data_ingestion import (
    DataIngestionWorkflow,
    IngestionConfig,
    IngestionResult
)
from app.models.experiment import ExperimentConfig
from app.services.format_detection import DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.services.zero_filtering import ZeroFilterConfig


# ==================== Fixtures ====================

@pytest.fixture
def basic_experiment():
    """Basic two-condition experiment."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3]
    )


@pytest.fixture
def simple_experiment():
    """Simple 2x2 experiment matching other test files."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


@pytest.fixture
def experiment_with_bqc():
    """Experiment with BQC condition."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['BQC', 'Control', 'Treatment'],
        number_of_samples_list=[2, 3, 3]
    )


@pytest.fixture
def lipidsearch_df(simple_experiment):
    """Sample LipidSearch 5.0 format DataFrame (standardized column names)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'TotalGrade': ['A', 'B', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 85.0, 95.0, 90.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2', 'd18:1_16:0'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def lipidsearch_df_with_standards(simple_experiment):
    """LipidSearch DataFrame with internal standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_18:1-d7)', 'PE(18:0_20:4)', 'PE(17:0_20:4-d7)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'CalcMass': [760.5, 774.6, 768.5, 782.6],
        'BaseRt': [10.5, 10.8, 12.3, 12.6],
        'TotalGrade': ['A', 'A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '15:0_18:1', '18:0_20:4', '17:0_20:4'],
        'intensity[s1]': [1e6, 5e5, 2e6, 5e5],
        'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5],
        'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5],
        'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5],
    })


@pytest.fixture
def msdial_df(simple_experiment):
    """Sample MS-DIAL format DataFrame (standardized column names)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'Total score': [90.0, 85.0, 88.0, 82.0],
        'MS/MS matched': ['TRUE', 'TRUE', 'TRUE', 'TRUE'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def generic_df(simple_experiment):
    """Sample Generic format DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def df_with_zeros(simple_experiment):
    """DataFrame with some zero values for filtering tests."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'TotalGrade': ['A', 'A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2', 'd18:1_16:0'],
        # PC: all good
        # PE: many zeros (should be filtered)
        # TG: some zeros but not enough
        # SM: all good
        'intensity[s1]': [1e6, 0, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 0, 0, 1.6e6],
        'intensity[s3]': [1.2e6, 0, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 0, 3.3e6, 1.8e6],
    })


# ==================== IngestionResult Tests ====================

class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = IngestionResult(detected_format=DataFormat.GENERIC)
        assert result.detected_format == DataFormat.GENERIC
        assert result.format_confidence == "high"
        assert result.cleaned_df is None
        assert result.internal_standards_df is None
        assert result.cleaning_messages == []
        assert result.zero_filtered is False
        assert result.is_valid is True
        assert result.validation_errors == []

    def test_species_removed_count(self):
        """Test species removed count calculation."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=100,
            species_after_filter=80
        )
        assert result.species_removed_count == 20

    def test_removal_percentage(self):
        """Test removal percentage calculation."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=100,
            species_after_filter=75
        )
        assert result.removal_percentage == 25.0

    def test_removal_percentage_zero_before(self):
        """Test removal percentage when no species before."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=0,
            species_after_filter=0
        )
        assert result.removal_percentage == 0.0


# ==================== IngestionConfig Tests ====================

class TestIngestionConfig:
    """Tests for IngestionConfig dataclass."""

    def test_minimal_config(self, simple_experiment):
        """Test creating config with minimal required fields."""
        config = IngestionConfig(experiment=simple_experiment)
        assert config.experiment == simple_experiment
        assert config.data_format is None
        assert config.apply_zero_filter is True
        assert config.use_external_standards is False

    def test_full_config(self, simple_experiment):
        """Test creating config with all fields."""
        grade_config = GradeFilterConfig()
        zero_config = ZeroFilterConfig()

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=True,
            zero_filter_config=zero_config,
            bqc_label='BQC'
        )

        assert config.data_format == DataFormat.LIPIDSEARCH
        assert config.grade_config is not None
        assert config.bqc_label == 'BQC'


# ==================== Format Detection Tests ====================

class TestDetectFormatOnly:
    """Tests for detect_format_only method."""

    def test_lipidsearch_detection_raw_format(self):
        """Test LipidSearch format detection with raw MeanArea columns."""
        # Raw LipidSearch format uses MeanArea[*] columns
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1e6],
            'MeanArea[s2]': [1.1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.LIPIDSEARCH
        assert confidence == "high"

    def test_standardized_df_detected_as_generic(self, lipidsearch_df):
        """Test that standardized intensity[] columns are detected as Generic."""
        # Our fixtures use intensity[*] columns (standardized format)
        # which lack the MeanArea[*] signature, so they detect as Generic
        detected, confidence = DataIngestionWorkflow.detect_format_only(lipidsearch_df)
        assert detected == DataFormat.GENERIC
        assert confidence == "medium"

    def test_generic_detection(self, generic_df):
        """Test Generic format detection."""
        detected, confidence = DataIngestionWorkflow.detect_format_only(generic_df)
        assert detected == DataFormat.GENERIC
        assert confidence == "medium"

    def test_unknown_format(self):
        """Test unknown format detection."""
        df = pd.DataFrame({'random': [1, 2, 3]})
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.UNKNOWN
        assert confidence == "low"


# ==================== Validation Tests ====================

class TestValidateForFormat:
    """Tests for validate_for_format method."""

    def test_valid_lipidsearch(self, lipidsearch_df):
        """Test validation for valid LipidSearch data."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            lipidsearch_df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is True
        assert errors == []

    def test_valid_generic(self, generic_df):
        """Test validation for valid Generic data."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            generic_df, DataFormat.GENERIC
        )
        assert is_valid is True
        assert errors == []

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            pd.DataFrame(), DataFormat.GENERIC
        )
        assert is_valid is False
        assert 'empty' in errors[0].lower()

    def test_missing_lipidmolec(self):
        """Test validation fails when LipidMolec is missing."""
        df = pd.DataFrame({'ClassKey': ['PC'], 'Sample1': [1e6]})
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.GENERIC
        )
        assert is_valid is False
        assert 'LipidMolec' in errors[0]

    def test_lipidsearch_missing_columns(self, generic_df):
        """Test validation fails for LipidSearch when columns missing."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            generic_df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is False
        assert 'LipidSearch' in errors[0]


# ==================== Get Sample Columns Tests ====================

class TestGetSampleColumns:
    """Tests for get_sample_columns method."""

    def test_lipidsearch_samples(self, lipidsearch_df):
        """Test getting sample columns from LipidSearch data."""
        # Note: After cleaning, columns are intensity[*], but for raw detection
        # it looks for MeanArea[*]
        samples = DataIngestionWorkflow.get_sample_columns(
            lipidsearch_df, DataFormat.LIPIDSEARCH
        )
        # Our fixture uses intensity[*] not MeanArea[*], so none match
        assert len(samples) == 0  # MeanArea columns not present

    def test_generic_samples(self, generic_df):
        """Test getting sample columns from Generic data."""
        samples = DataIngestionWorkflow.get_sample_columns(
            generic_df, DataFormat.GENERIC
        )
        # Should exclude LipidMolec and ClassKey
        assert 'LipidMolec' not in samples
        assert 'ClassKey' not in samples
        # Should include intensity columns
        assert 'intensity[s1]' in samples


# ==================== Full Workflow Tests ====================

class TestWorkflowRun:
    """Tests for the complete run() method."""

    def test_basic_workflow_lipidsearch(self, lipidsearch_df, simple_experiment):
        """Test basic workflow with LipidSearch data."""
        # Explicitly specify format since fixture uses standardized column names
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False  # Skip for simplicity
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.cleaned_df is not None
        assert len(result.validation_errors) == 0

    def test_basic_workflow_generic(self, generic_df, simple_experiment):
        """Test basic workflow with Generic data."""
        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.GENERIC
        assert result.cleaned_df is not None

    def test_workflow_with_explicit_format(self, lipidsearch_df, simple_experiment):
        """Test workflow with explicitly specified format."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.is_valid is True

    def test_workflow_extracts_standards(self, lipidsearch_df_with_standards, simple_experiment):
        """Test that workflow extracts internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        # Internal standards with (d7) should be extracted
        assert result.internal_standards_df is not None

    def test_workflow_with_grade_config(self, lipidsearch_df, simple_experiment):
        """Test workflow with grade filtering config."""
        # GradeFilterConfig takes a dict mapping class to allowed grades
        grade_config = GradeFilterConfig(grade_config={'PC': ['A'], 'PE': ['A', 'B']})
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True

    def test_workflow_unknown_format_fails(self, simple_experiment):
        """Test that unknown format causes validation failure."""
        df = pd.DataFrame({'random_col': [1, 2, 3]})
        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is False
        assert DataFormat.UNKNOWN == result.detected_format
        assert len(result.validation_errors) > 0

    def test_workflow_empty_df_fails(self, simple_experiment):
        """Test that empty DataFrame causes failure."""
        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(pd.DataFrame(), config)

        assert result.is_valid is False


# ==================== Zero Filtering Tests ====================

class TestWorkflowZeroFiltering:
    """Tests for zero filtering in the workflow."""

    def test_zero_filtering_applied(self, df_with_zeros, simple_experiment):
        """Test that zero filtering is applied."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        assert result.is_valid is True
        assert result.zero_filtered is True

    def test_zero_filtering_skipped_when_disabled(self, lipidsearch_df, simple_experiment):
        """Test that zero filtering is skipped when disabled."""
        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.zero_filtered is False
        assert result.species_before_filter == 0
        assert result.species_after_filter == 0

    def test_zero_filtering_with_custom_config(self, df_with_zeros, simple_experiment):
        """Test zero filtering with custom configuration."""
        zero_config = ZeroFilterConfig(
            detection_threshold=0,
            bqc_threshold=0.3,
            non_bqc_threshold=0.5
        )
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=zero_config
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        assert result.is_valid is True


# ==================== External Standards Tests ====================

class TestWorkflowExternalStandards:
    """Tests for external standards handling."""

    def test_external_standards_used(self, lipidsearch_df, simple_experiment):
        """Test that external standards are used when provided."""
        external_standards = pd.DataFrame({
            'LipidMolec': ['PC-IS(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5e5],
            'intensity[s2]': [5.1e5],
            'intensity[s3]': [5.2e5],
            'intensity[s4]': [5.3e5],
        })

        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=external_standards
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.internal_standards_df is not None
        assert len(result.internal_standards_df) == 1
        # Should have a message about external standards
        assert any('external' in msg.lower() for msg in result.cleaning_messages)


# ==================== Error Handling Tests ====================

class TestWorkflowErrorHandling:
    """Tests for error handling in the workflow."""

    def test_cleaning_error_captured(self, simple_experiment):
        """Test that cleaning errors are captured in result."""
        # Create DataFrame that will fail cleaning (missing intensity columns)
        df = pd.DataFrame({
            'LipidMolec': ['PC 32:0'],
            'ClassKey': ['PC'],
            # Missing intensity columns
        })

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        # Should fail during cleaning
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_invalid_zero_config_raises(self, simple_experiment):
        """Test that invalid zero filter config raises ValueError."""
        with pytest.raises(ValueError, match="detection_threshold"):
            ZeroFilterConfig(detection_threshold=-1)


# ==================== Integration Tests ====================

class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    def test_full_pipeline_lipidsearch(self, lipidsearch_df_with_standards, simple_experiment):
        """Test complete pipeline with LipidSearch data."""
        # Use default grade config (None = all grades A, B, C allowed)
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=GradeFilterConfig(),  # Default config
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.cleaned_df is not None
        assert 'LipidMolec' in result.cleaned_df.columns

    def test_result_df_usable_for_downstream(self, lipidsearch_df, simple_experiment):
        """Test that result DataFrame is usable for downstream analysis."""
        config = IngestionConfig(
            experiment=simple_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        df = result.cleaned_df

        # Should be able to group by class
        if 'ClassKey' in df.columns:
            grouped = df.groupby('ClassKey').size()
            assert len(grouped) > 0

    def test_multiple_runs_independent(self, lipidsearch_df, generic_df, simple_experiment):
        """Test that multiple workflow runs are independent."""
        config1 = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        config2 = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False
        )

        result1 = DataIngestionWorkflow.run(lipidsearch_df.copy(), config1)
        result2 = DataIngestionWorkflow.run(generic_df.copy(), config2)

        # Results should be independent
        assert result1.detected_format == DataFormat.LIPIDSEARCH
        assert result2.detected_format == DataFormat.GENERIC
        assert result1.is_valid is True
        assert result2.is_valid is True
