"""Unit tests for DataCleaningService."""
import pytest
import pandas as pd
import numpy as np
from app.services.data_cleaning import (
    DataCleaningService,
    GradeFilterConfig,
    QualityFilterConfig,
    CleaningResult,
    BaseDataCleaner,
    LipidSearchCleaner,
    MSDIALCleaner,
    GenericCleaner,
)
from app.services.format_detection import DataFormat
from app.models.experiment import ExperimentConfig

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def single_condition_experiment():
    """Single condition experiment with 3 samples."""
    return ExperimentConfig(
        n_conditions=1,
        conditions_list=['Control'],
        number_of_samples_list=[3]
    )


@pytest.fixture
def large_experiment():
    """Large experiment with 4 conditions, varying samples."""
    return ExperimentConfig(
        n_conditions=4,
        conditions_list=['Control', 'Low', 'Medium', 'High'],
        number_of_samples_list=[3, 3, 4, 4]
    )


@pytest.fixture
def lipidsearch_df():
    """Valid LipidSearch 5.0 DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'B', 'A'],
        'TotalSmpIDRate(%)': [100.0, 85.0, 95.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'intensity[s1]': [1000.0, 2000.0, 3000.0],
        'intensity[s2]': [1100.0, 2100.0, 3100.0],
        'intensity[s3]': [1200.0, 2200.0, 3200.0],
        'intensity[s4]': [1300.0, 2300.0, 3300.0],
    })


@pytest.fixture
def msdial_df():
    """Valid MS-DIAL DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'Total score': [85.0, 72.0, 90.0],
        'MS/MS matched': ['TRUE', 'FALSE', 'TRUE'],
        'intensity[s1]': [1000.0, 2000.0, 3000.0],
        'intensity[s2]': [1100.0, 2100.0, 3100.0],
        'intensity[s3]': [1200.0, 2200.0, 3200.0],
        'intensity[s4]': [1300.0, 2300.0, 3300.0],
    })


@pytest.fixture
def generic_df():
    """Valid Generic format DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [1000.0, 2000.0, 3000.0],
        'intensity[s2]': [1100.0, 2100.0, 3100.0],
        'intensity[s3]': [1200.0, 2200.0, 3200.0],
        'intensity[s4]': [1300.0, 2300.0, 3300.0],
    })


# =============================================================================
# GradeFilterConfig Tests
# =============================================================================

class TestGradeFilterConfig:
    """Tests for GradeFilterConfig."""

    def test_default_config(self):
        """Test default grade config."""
        config = GradeFilterConfig()
        assert config.is_default is True
        assert config.grade_config is None

    def test_custom_config(self):
        """Test custom grade config."""
        grade_dict = {'PC': ['A', 'B'], 'PE': ['A']}
        config = GradeFilterConfig(grade_dict)
        assert config.is_default is False
        assert config.grade_config == grade_dict

    def test_empty_dict_config(self):
        """Test empty dict is not default."""
        config = GradeFilterConfig({})
        assert config.is_default is False
        assert config.grade_config == {}

    def test_config_with_all_grades(self):
        """Test config with all grades for a class."""
        grade_dict = {'PC': ['A', 'B', 'C', 'D']}
        config = GradeFilterConfig(grade_dict)
        assert 'A' in config.grade_config['PC']
        assert 'D' in config.grade_config['PC']

    def test_config_with_single_grade(self):
        """Test config with single grade."""
        grade_dict = {'PC': ['A']}
        config = GradeFilterConfig(grade_dict)
        assert config.grade_config['PC'] == ['A']


# =============================================================================
# QualityFilterConfig Tests
# =============================================================================

class TestQualityFilterConfig:
    """Tests for QualityFilterConfig."""

    def test_default_config(self):
        """Test default quality config (no filtering)."""
        config = QualityFilterConfig()
        assert config.total_score_threshold == 0
        assert config.require_msms is False

    def test_custom_config(self):
        """Test custom quality config."""
        config = QualityFilterConfig(total_score_threshold=75, require_msms=True)
        assert config.total_score_threshold == 75
        assert config.require_msms is True

    @pytest.mark.parametrize("preset,expected_threshold,expected_msms", [
        ("strict", 80, True),
        ("moderate", 60, False),
        ("permissive", 40, False),
        ("no_filtering", 0, False),
    ])
    def test_quality_filter_presets(self, preset, expected_threshold, expected_msms):
        """Test all QualityFilterConfig factory presets."""
        config = getattr(QualityFilterConfig, preset)()
        assert config.total_score_threshold == expected_threshold
        assert config.require_msms is expected_msms

    def test_score_only_config(self):
        """Test config with only score threshold."""
        config = QualityFilterConfig(total_score_threshold=50)
        assert config.total_score_threshold == 50
        assert config.require_msms is False

    def test_msms_only_config(self):
        """Test config with only MS/MS requirement."""
        config = QualityFilterConfig(require_msms=True)
        assert config.total_score_threshold == 0
        assert config.require_msms is True

    def test_boundary_score_values(self):
        """Test boundary score values."""
        config_zero = QualityFilterConfig(total_score_threshold=0)
        config_hundred = QualityFilterConfig(total_score_threshold=100)
        assert config_zero.total_score_threshold == 0
        assert config_hundred.total_score_threshold == 100


# =============================================================================
# CleaningResult Tests
# =============================================================================

class TestCleaningResult:
    """Tests for CleaningResult."""

    def test_result_creation(self):
        """Test creating a cleaning result."""
        cleaned = pd.DataFrame({'LipidMolec': ['PC(16:0)']})
        standards = pd.DataFrame({'LipidMolec': ['PC(d7)']})
        messages = ['Removed 5 duplicates']

        result = CleaningResult(cleaned, standards, messages)

        assert len(result.cleaned_df) == 1
        assert len(result.internal_standards_df) == 1
        assert result.filter_messages == messages

    def test_result_empty_messages(self):
        """Test result with no messages."""
        result = CleaningResult(pd.DataFrame(), pd.DataFrame())
        assert result.filter_messages == []

    def test_result_empty_standards(self):
        """Test result with empty standards."""
        cleaned = pd.DataFrame({'LipidMolec': ['PC(16:0)']})
        result = CleaningResult(cleaned, pd.DataFrame())
        assert len(result.internal_standards_df) == 0

    def test_result_multiple_messages(self):
        """Test result with multiple messages."""
        messages = ['Msg 1', 'Msg 2', 'Msg 3']
        result = CleaningResult(pd.DataFrame(), pd.DataFrame(), messages)
        assert len(result.filter_messages) == 3

    def test_result_preserves_dataframe_columns(self):
        """Test that result preserves DataFrame structure."""
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000]
        })
        result = CleaningResult(cleaned, pd.DataFrame())
        assert 'LipidMolec' in result.cleaned_df.columns
        assert 'ClassKey' in result.cleaned_df.columns
        assert 'intensity[s1]' in result.cleaned_df.columns


# =============================================================================
# BaseDataCleaner Validation Tests
# =============================================================================

class TestBaseDataCleanerValidation:
    """Tests for BaseDataCleaner validation methods."""

    def test_is_effectively_empty_true_for_empty_df(self):
        """Test empty DataFrame is detected."""
        assert BaseDataCleaner.is_effectively_empty(pd.DataFrame()) is True

    def test_is_effectively_empty_true_for_all_nan_lipids(self):
        """Test DataFrame with all NaN lipids is detected as empty."""
        df = pd.DataFrame({'LipidMolec': [np.nan, np.nan]})
        assert BaseDataCleaner.is_effectively_empty(df) is True

    def test_is_effectively_empty_true_for_all_empty_strings(self):
        """Test DataFrame with all empty lipid names is detected."""
        df = pd.DataFrame({'LipidMolec': ['', '  ', '\t']})
        assert BaseDataCleaner.is_effectively_empty(df) is True

    def test_is_effectively_empty_false_for_valid_df(self, generic_df):
        """Test valid DataFrame is not empty."""
        assert BaseDataCleaner.is_effectively_empty(generic_df) is False

    def test_is_effectively_empty_false_for_single_valid_row(self):
        """Test single valid row is not empty."""
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)']})
        assert BaseDataCleaner.is_effectively_empty(df) is False

    def test_is_effectively_empty_mixed_valid_and_empty(self):
        """Test mixed valid and empty values."""
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)', '', np.nan]})
        assert BaseDataCleaner.is_effectively_empty(df) is False

    def test_is_effectively_empty_no_lipidmolec_column(self):
        """Test DataFrame without LipidMolec column."""
        df = pd.DataFrame({'Other': [1, 2, 3]})
        assert BaseDataCleaner.is_effectively_empty(df) is False

    def test_find_lipid_column_standard(self):
        """Test finding standard LipidMolec column."""
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)'], 'Other': [1]})
        assert BaseDataCleaner.find_lipid_column(df) == 'LipidMolec'

    def test_find_lipid_column_case_insensitive(self):
        """Test finding LipidMolec column case-insensitively."""
        df = pd.DataFrame({'lipidmolec': ['PC(16:0)'], 'Other': [1]})
        assert BaseDataCleaner.find_lipid_column(df) == 'lipidmolec'

    def test_find_lipid_column_uppercase(self):
        """Test finding LIPIDMOLEC column."""
        df = pd.DataFrame({'LIPIDMOLEC': ['PC(16:0)'], 'Other': [1]})
        assert BaseDataCleaner.find_lipid_column(df) == 'LIPIDMOLEC'

    def test_find_lipid_column_mixed_case(self):
        """Test finding LipidMOLEC column."""
        df = pd.DataFrame({'LipidMOLEC': ['PC(16:0)'], 'Other': [1]})
        assert BaseDataCleaner.find_lipid_column(df) == 'LipidMOLEC'

    def test_find_lipid_column_raises_if_missing(self):
        """Test error when LipidMolec column is missing."""
        df = pd.DataFrame({'Other': [1]})
        with pytest.raises(KeyError, match="LipidMolec"):
            BaseDataCleaner.find_lipid_column(df)


# =============================================================================
# BaseDataCleaner Numeric Conversion Tests
# =============================================================================

class TestBaseDataCleanerNumericConversion:
    """Tests for numeric conversion methods."""

    def test_convert_columns_to_numeric(self, simple_experiment_2x2):
        """Test converting intensity columns to numeric."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s1]': ['1000'],
            'intensity[s2]': ['2000.5'],
            'intensity[s3]': ['invalid'],
            'intensity[s4]': ['-100'],
        })
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)

        assert result['intensity[s1]'].iloc[0] == 1000.0
        assert result['intensity[s2]'].iloc[0] == 2000.5
        assert result['intensity[s3]'].iloc[0] == 0.0  # invalid -> 0
        assert result['intensity[s4]'].iloc[0] == 0.0  # negative clipped to 0

    def test_convert_handles_nan_values(self, simple_experiment_2x2):
        """Test NaN values are converted to 0."""
        df = pd.DataFrame({
            'intensity[s1]': [np.nan],
            'intensity[s2]': [None],
            'intensity[s3]': [1000],
            'intensity[s4]': [2000],
        })
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)

        assert result['intensity[s1]'].iloc[0] == 0.0
        assert result['intensity[s2]'].iloc[0] == 0.0

    def test_convert_preserves_non_intensity_columns(self, simple_experiment_2x2):
        """Test that non-intensity columns are preserved."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000],
            'intensity[s2]': [2000],
            'intensity[s3]': [3000],
            'intensity[s4]': [4000],
        })
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)

        assert result['LipidMolec'].iloc[0] == 'PC(16:0)'
        assert result['ClassKey'].iloc[0] == 'PC'

    def test_convert_handles_scientific_notation(self, simple_experiment_2x2):
        """Test scientific notation is handled."""
        df = pd.DataFrame({
            'intensity[s1]': ['1e6'],
            'intensity[s2]': ['2.5e3'],
            'intensity[s3]': [1000],
            'intensity[s4]': [2000],
        })
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)

        assert result['intensity[s1]'].iloc[0] == 1000000.0
        assert result['intensity[s2]'].iloc[0] == 2500.0

    def test_convert_handles_whitespace(self, simple_experiment_2x2):
        """Test whitespace in numeric strings."""
        df = pd.DataFrame({
            'intensity[s1]': [' 1000 '],
            'intensity[s2]': ['  2000  '],
            'intensity[s3]': [1000],
            'intensity[s4]': [2000],
        })
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)

        assert result['intensity[s1]'].iloc[0] == 1000.0
        assert result['intensity[s2]'].iloc[0] == 2000.0

    def test_convert_empty_string_to_zero(self, simple_experiment_2x2):
        """Test empty strings become zero."""
        df = pd.DataFrame({
            'intensity[s1]': [''],
            'intensity[s2]': [''],
            'intensity[s3]': [1000],
            'intensity[s4]': [2000],
        })
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)

        assert result['intensity[s1]'].iloc[0] == 0.0
        assert result['intensity[s2]'].iloc[0] == 0.0

    def test_convert_handles_missing_columns(self, simple_experiment_2x2):
        """Test handling when some intensity columns are missing."""
        df = pd.DataFrame({
            'intensity[s1]': [1000],
            'intensity[s2]': [2000],
            # s3 and s4 missing
        })
        # Should not raise, just convert existing columns
        result = BaseDataCleaner.convert_columns_to_numeric(df, simple_experiment_2x2.full_samples_list)
        assert 'intensity[s1]' in result.columns


# =============================================================================
# BaseDataCleaner Row Filtering Tests
# =============================================================================

class TestBaseDataCleanerRowFiltering:
    """Tests for row filtering methods."""

    @pytest.mark.parametrize("invalid_values,expected_valid", [
        ([''], 2),                                               # empty string
        (['   ', '\t\n'], 2),                                    # whitespace
        (['#', '###', '@@@', '%%%'], 2),                         # special chars
        (['Unknown', 'UNKNOWN', 'unknown'], 2),                  # unknown variants
        (['nan', 'null', 'none', 'NaN', 'NULL', 'None'], 2),    # nan/null/none strings
        (['12345', '999', '0'], 2),                              # numbers only
    ])
    def test_remove_invalid_lipid_rows(self, invalid_values, expected_valid):
        """Test removing rows with various invalid lipid names."""
        valid = ['PC(16:0)', 'PE(18:0)']
        all_lipids = [valid[0]] + invalid_values + [valid[1]]
        df = pd.DataFrame({
            'LipidMolec': all_lipids,
            'Value': range(len(all_lipids))
        })
        result = BaseDataCleaner.remove_invalid_lipid_rows(df)
        assert len(result) == expected_valid

    def test_remove_invalid_lipid_rows_actual_nan(self):
        """Test removing actual NaN/None values (not strings)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', np.nan, None, 'PE(18:0)'],
            'Value': [1, 2, 3, 4]
        })
        result = BaseDataCleaner.remove_invalid_lipid_rows(df)
        assert len(result) == 2

    def test_remove_invalid_keeps_lipids_with_numbers(self):
        """Test that valid lipids containing numbers are kept."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0_20:4)', 'TG12345'],
            'Value': [1, 2, 3]
        })
        result = BaseDataCleaner.remove_invalid_lipid_rows(df)
        assert len(result) == 3

    def test_remove_all_zero_rows(self):
        """Test removing rows with all zero intensities."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'intensity[s1]': [1000, 0, 0],
            'intensity[s2]': [2000, 0, 0],
        })
        result = BaseDataCleaner.remove_all_zero_rows(df)
        assert len(result) == 1
        assert result['LipidMolec'].iloc[0] == 'PC(16:0)'

    def test_remove_all_zero_rows_keeps_partial_zeros(self):
        """Test that rows with some non-zero values are kept."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'intensity[s1]': [1000, 0],
            'intensity[s2]': [0, 500],
        })
        result = BaseDataCleaner.remove_all_zero_rows(df)
        assert len(result) == 2

    def test_remove_all_zero_rows_handles_nan(self):
        """Test that all-NaN rows are also removed."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'intensity[s1]': [1000, np.nan],
            'intensity[s2]': [2000, np.nan],
        })
        result = BaseDataCleaner.remove_all_zero_rows(df)
        assert len(result) == 1

    def test_remove_all_zero_rows_no_intensity_columns(self):
        """Test handling when no intensity columns exist."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
        })
        result = BaseDataCleaner.remove_all_zero_rows(df)
        assert len(result) == 2

    def test_get_intensity_columns(self):
        """Test getting intensity column names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000],
            'intensity[s2]': [2000],
            'Other': [3000],
        })
        result = BaseDataCleaner.get_intensity_columns(df)
        assert 'intensity[s1]' in result
        assert 'intensity[s2]' in result
        assert 'Other' not in result
        assert 'LipidMolec' not in result


# =============================================================================
# BaseDataCleaner Internal Standards Tests
# =============================================================================

class TestBaseDataCleanerInternalStandards:
    """Tests for internal standards extraction."""

    def test_extract_deuterium_d5(self):
        """Test extracting (d5) standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(d5)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1
        assert 'PC(d5)' in standards['LipidMolec'].values

    def test_extract_deuterium_d7(self):
        """Test extracting (d7) standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(d7)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_deuterium_d9(self):
        """Test extracting (d9) standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(d9)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_plus_d_standards(self):
        """Test extracting +D standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)+D7', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1
        assert 'PC(16:0)+D7' in standards['LipidMolec'].values

    def test_extract_minus_d_standards(self):
        """Test extracting -D standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)-D7', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_istd_in_classkey(self):
        """Test extracting standards with ISTD in ClassKey."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'ISTD', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_internal_in_classkey(self):
        """Test extracting standards with 'Internal' in ClassKey."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'Internal Standard', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_splash_standards(self):
        """Test extracting SPLASH standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'SPLASH-PC(16:0)', 'SPLASH_PE', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 2

    def test_extract_cholesterol_d_standards(self):
        """Test extracting Ch-D cholesterol standards."""
        df = pd.DataFrame({
            'LipidMolec': ['Ch()', 'Ch-D7()', 'Ch-D9()', 'PE(18:0)'],
            'ClassKey': ['Ch', 'Ch', 'Ch', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 2

    def test_extract_colon_s_notation(self):
        """Test extracting :(s) notation standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0):(s)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_is_suffix(self):
        """Test extracting _IS suffix standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)_IS', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_parenthesis_is(self):
        """Test extracting (IS) notation standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)(IS)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1

    def test_extract_no_standards(self):
        """Test when no standards are present."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(cleaned) == 3
        assert len(standards) == 0

    def test_extract_multiple_standards(self):
        """Test extracting multiple different standards."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(d7)', 'PE(d9)', 'TG(16:0)+D5', 'SM(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE', 'TG', 'SM'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(cleaned) == 2
        assert len(standards) == 3

    def test_extract_empty_dataframe(self):
        """Test extracting from empty DataFrame."""
        df = pd.DataFrame(columns=['LipidMolec', 'ClassKey'])
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(cleaned) == 0
        assert len(standards) == 0

    def test_extract_no_classkey_column(self):
        """Test extracting when ClassKey column is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(d7)', 'PE(18:0)'],
        })
        cleaned, standards = BaseDataCleaner.extract_internal_standards(df)
        assert len(standards) == 1  # d7 still detected


# =============================================================================
# LipidSearchCleaner Tests
# =============================================================================

class TestLipidSearchCleanerBasic:
    """Basic tests for LipidSearchCleaner."""

    def test_clean_valid_data(self, lipidsearch_df, simple_experiment_2x2):
        """Test cleaning valid LipidSearch data."""
        cleaned, messages = LipidSearchCleaner.clean(lipidsearch_df, simple_experiment_2x2)

        assert not cleaned.empty
        assert 'LipidMolec' in cleaned.columns
        assert 'ClassKey' in cleaned.columns
        assert 'TotalSmpIDRate(%)' not in cleaned.columns  # Should be removed

    def test_clean_raises_on_empty_df(self, simple_experiment_2x2):
        """Test error on empty DataFrame."""
        with pytest.raises(ValueError, match="empty"):
            LipidSearchCleaner.clean(pd.DataFrame(), simple_experiment_2x2)

    def test_clean_returns_messages(self, lipidsearch_df, simple_experiment_2x2):
        """Test that cleaning returns messages."""
        cleaned, messages = LipidSearchCleaner.clean(lipidsearch_df, simple_experiment_2x2)
        assert isinstance(messages, list)

    def test_clean_preserves_required_columns(self, lipidsearch_df, simple_experiment_2x2):
        """Test that required columns are preserved."""
        cleaned, _ = LipidSearchCleaner.clean(lipidsearch_df, simple_experiment_2x2)
        assert 'LipidMolec' in cleaned.columns
        assert 'ClassKey' in cleaned.columns
        assert 'CalcMass' in cleaned.columns
        assert 'BaseRt' in cleaned.columns


class TestLipidSearchFAKeyHandling:
    """Tests for FA key handling in LipidSearch."""

    def test_remove_missing_fa_keys(self):
        """Test removing rows with missing FA keys."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'FAKey': ['16:0', np.nan, '16:0'],
        })
        result = LipidSearchCleaner._remove_missing_fa_keys(df)
        assert len(result) == 2

    def test_keep_cholesterol_without_fakey(self):
        """Test that cholesterol is kept even without FA key."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'Ch()', 'PE(18:0)'],
            'ClassKey': ['PC', 'Ch', 'PE'],
            'FAKey': ['16:0', np.nan, np.nan],
        })
        result = LipidSearchCleaner._remove_missing_fa_keys(df)
        assert len(result) == 2
        assert 'Ch' in result['ClassKey'].values

    def test_keep_ch_d_standards(self):
        """Test Ch-D standards are kept."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'Ch-D7()', 'PE(18:0)'],
            'ClassKey': ['PC', 'Ch', 'PE'],
            'FAKey': ['16:0', np.nan, np.nan],
        })
        result = LipidSearchCleaner._remove_missing_fa_keys(df)
        assert len(result) == 2
        assert 'Ch-D7()' in result['LipidMolec'].values

    def test_empty_df_returns_empty(self):
        """Test empty DataFrame returns empty."""
        result = LipidSearchCleaner._remove_missing_fa_keys(pd.DataFrame())
        assert result.empty


class TestLipidSearchGradeFiltering:
    """Tests for grade filtering in LipidSearch."""

    def test_default_grade_filter(self):
        """Test default grade filtering (A, B, C)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC1', 'PC2', 'PC3', 'PC4'],
            'ClassKey': ['PC', 'PC', 'PC', 'PC'],
            'TotalGrade': ['A', 'B', 'C', 'D'],
        })
        result = LipidSearchCleaner._apply_grade_filter(df)
        assert len(result) == 3
        assert 'D' not in result['TotalGrade'].values

    def test_custom_grade_filter(self):
        """Test custom grade filtering."""
        df = pd.DataFrame({
            'LipidMolec': ['PC1', 'PC2', 'PE1', 'PE2'],
            'ClassKey': ['PC', 'PC', 'PE', 'PE'],
            'TotalGrade': ['A', 'B', 'A', 'B'],
        })
        config = {'PC': ['A'], 'PE': ['A', 'B']}
        result = LipidSearchCleaner._apply_grade_filter(df, config)
        assert len(result) == 3  # PC1 + PE1 + PE2

    def test_grade_filter_empty_result(self):
        """Test grade filter resulting in empty DataFrame."""
        df = pd.DataFrame({
            'LipidMolec': ['PC1'],
            'ClassKey': ['PC'],
            'TotalGrade': ['D'],
        })
        result = LipidSearchCleaner._apply_grade_filter(df)
        assert len(result) == 0

    def test_custom_filter_excludes_class(self):
        """Test custom filter that excludes a class entirely."""
        df = pd.DataFrame({
            'LipidMolec': ['PC1', 'PE1'],
            'ClassKey': ['PC', 'PE'],
            'TotalGrade': ['A', 'A'],
        })
        config = {'PC': ['A']}  # PE not included
        result = LipidSearchCleaner._apply_grade_filter(df, config)
        assert len(result) == 1
        assert result['ClassKey'].iloc[0] == 'PC'

    def test_filter_accepts_grade_d_when_configured(self):
        """Test that grade D is accepted when configured."""
        df = pd.DataFrame({
            'LipidMolec': ['PC1', 'PC2'],
            'ClassKey': ['PC', 'PC'],
            'TotalGrade': ['A', 'D'],
        })
        config = {'PC': ['A', 'D']}
        result = LipidSearchCleaner._apply_grade_filter(df, config)
        assert len(result) == 2


class TestLipidSearchNameStandardization:
    """Tests for lipid name standardization."""

    def test_standardize_simple_name(self):
        """Test standardizing simple lipid name."""
        result = LipidSearchCleaner._standardize_single_name('PC', '16:0_18:1')
        assert result == 'PC 16:0_18:1'

    def test_standardize_sorts_fatty_acids(self):
        """Test that fatty acids are sorted."""
        result = LipidSearchCleaner._standardize_single_name('PC', '18:1_16:0')
        assert result == 'PC 16:0_18:1'

    def test_standardize_three_fatty_acids(self):
        """Test sorting three fatty acids."""
        result = LipidSearchCleaner._standardize_single_name('TG', '20:4_16:0_18:1')
        assert result == 'TG 16:0_18:1_20:4'

    def test_standardize_cholesterol(self):
        """Test standardizing cholesterol."""
        result = LipidSearchCleaner._standardize_single_name('Ch', '')
        assert result == 'Ch'

    def test_standardize_cholesterol_d(self):
        """Test standardizing deuterated cholesterol."""
        result = LipidSearchCleaner._standardize_single_name('Ch', 'D7')
        assert result == 'Ch-D7'

    def test_standardize_cholesterol_d9(self):
        """Test standardizing D9 cholesterol."""
        result = LipidSearchCleaner._standardize_single_name('Ch', 'D9')
        assert result == 'Ch-D9'

    def test_standardize_with_internal_standard_suffix(self):
        """Test standardizing with +D suffix."""
        result = LipidSearchCleaner._standardize_single_name('PC', '16:0_18:1+D7')
        assert '+D7' in result
        assert '16:0' in result

    def test_standardize_nan_fakey(self):
        """Test standardizing with NaN FA key."""
        result = LipidSearchCleaner._standardize_single_name('PC', np.nan)
        assert result == 'PC'

    def test_standardize_none_fakey(self):
        """Test standardizing with None FA key."""
        result = LipidSearchCleaner._standardize_single_name('PC', None)
        assert result == 'PC'

    def test_standardize_empty_fakey(self):
        """Test standardizing with empty FA key."""
        result = LipidSearchCleaner._standardize_single_name('PC', '')
        assert result == 'PC'

    def test_standardize_parentheses_only(self):
        """Test standardizing with () FA key."""
        result = LipidSearchCleaner._standardize_single_name('PC', '()')
        assert result == 'PC'

    def test_extract_internal_standard_suffix(self):
        """Test extracting internal standard suffix."""
        fa, suffix = LipidSearchCleaner._extract_internal_standard_suffix('16:0+D7')
        assert fa == '16:0'
        assert suffix == '+D7'

    def test_extract_no_suffix(self):
        """Test when no suffix present."""
        fa, suffix = LipidSearchCleaner._extract_internal_standard_suffix('16:0')
        assert fa == '16:0'
        assert suffix == ''


class TestLipidSearchAUCSelection:
    """Tests for AUC selection in LipidSearch."""

    def test_select_best_auc_by_quality(self, simple_experiment_2x2):
        """Test selecting best entry by quality rate."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)'],
            'ClassKey': ['PC', 'PC'],
            'CalcMass': [760.5, 760.5],
            'BaseRt': [10.5, 10.5],
            'TotalGrade': ['A', 'A'],
            'TotalSmpIDRate(%)': [90.0, 100.0],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1000, 2000],
            'intensity[s3]': [1000, 2000],
            'intensity[s4]': [1000, 2000],
        })
        result = LipidSearchCleaner._select_best_auc(df, simple_experiment_2x2)

        assert len(result) == 1
        assert result['intensity[s1]'].iloc[0] == 2000

    def test_select_best_auc_by_grade(self, simple_experiment_2x2):
        """Test selecting best entry by grade when quality is equal."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)'],
            'ClassKey': ['PC', 'PC'],
            'CalcMass': [760.5, 760.5],
            'BaseRt': [10.5, 10.5],
            'TotalGrade': ['B', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1000, 2000],
            'intensity[s3]': [1000, 2000],
            'intensity[s4]': [1000, 2000],
        })
        result = LipidSearchCleaner._select_best_auc(df, simple_experiment_2x2)

        assert len(result) == 1
        assert result['intensity[s1]'].iloc[0] == 2000  # Grade A entry

    def test_select_multiple_unique_lipids(self, simple_experiment_2x2):
        """Test selecting from multiple unique lipids."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'TotalGrade': ['A', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1000, 2000],
            'intensity[s3]': [1000, 2000],
            'intensity[s4]': [1000, 2000],
        })
        result = LipidSearchCleaner._select_best_auc(df, simple_experiment_2x2)
        assert len(result) == 2

    def test_select_best_row_nan_grade_deprioritized(self):
        """Test that NaN TotalGrade is deprioritized over any valid grade."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)'],
            'TotalGrade': [float('nan'), 'D'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'intensity[s1]': [1000, 2000],
        })
        result = LipidSearchCleaner._select_best_row(df)

        assert result['intensity[s1]'] == 2000  # Grade D selected, not NaN

    def test_select_best_row_all_nan_grades(self):
        """Test that all-NaN grades still selects a row without error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)'],
            'TotalGrade': [float('nan'), float('nan')],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'intensity[s1]': [1000, 2000],
        })
        result = LipidSearchCleaner._select_best_row(df)

        assert result['LipidMolec'] == 'PC(16:0)'  # Returns a valid row


# =============================================================================
# MSDIALCleaner Tests
# =============================================================================

class TestMSDIALCleanerBasic:
    """Basic tests for MSDIALCleaner."""

    def test_clean_valid_data(self, msdial_df, simple_experiment_2x2):
        """Test cleaning valid MS-DIAL data."""
        cleaned, messages = MSDIALCleaner.clean(msdial_df, simple_experiment_2x2)

        assert not cleaned.empty
        assert 'LipidMolec' in cleaned.columns
        assert 'ClassKey' in cleaned.columns
        assert 'Total score' not in cleaned.columns

    def test_clean_raises_on_empty_df(self, simple_experiment_2x2):
        """Test error on empty DataFrame."""
        with pytest.raises(ValueError, match="empty"):
            MSDIALCleaner.clean(pd.DataFrame(), simple_experiment_2x2)

    def test_clean_preserves_required_columns(self, msdial_df, simple_experiment_2x2):
        """Test required columns are preserved."""
        cleaned, _ = MSDIALCleaner.clean(msdial_df, simple_experiment_2x2)
        assert 'LipidMolec' in cleaned.columns
        assert 'ClassKey' in cleaned.columns


class TestMSDIALQualityFiltering:
    """Tests for MS-DIAL quality filtering."""

    def test_score_filter_removes_low_scores(self, msdial_df, simple_experiment_2x2):
        """Test that low scores are filtered out."""
        config = QualityFilterConfig(total_score_threshold=80)
        cleaned, messages = MSDIALCleaner.clean(msdial_df, simple_experiment_2x2, config)
        assert len(cleaned) == 2  # 85 and 90

    def test_msms_filter_requires_match(self, msdial_df, simple_experiment_2x2):
        """Test MS/MS filter removes entries without match."""
        config = QualityFilterConfig(require_msms=True)
        cleaned, messages = MSDIALCleaner.clean(msdial_df, simple_experiment_2x2, config)
        assert len(cleaned) == 2

    def test_combined_filters(self, msdial_df, simple_experiment_2x2):
        """Test combined score and MS/MS filtering."""
        config = QualityFilterConfig.strict()
        cleaned, messages = MSDIALCleaner.clean(msdial_df, simple_experiment_2x2, config)
        assert len(cleaned) == 2

    def test_no_filtering(self, msdial_df, simple_experiment_2x2):
        """Test with no quality filtering."""
        config = QualityFilterConfig.no_filtering()
        cleaned, messages = MSDIALCleaner.clean(msdial_df, simple_experiment_2x2, config)
        assert len(cleaned) == 3

    def test_score_filter_message(self, simple_experiment_2x2):
        """Test that score filtering generates message."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'Total score': [50.0, 90.0],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1100, 2100],
            'intensity[s3]': [1200, 2200],
            'intensity[s4]': [1300, 2300],
        })
        config = QualityFilterConfig(total_score_threshold=80)
        _, messages = MSDIALCleaner.clean(df, simple_experiment_2x2, config)
        assert any('Quality filter' in msg for msg in messages)

    def test_msms_filter_message(self, simple_experiment_2x2):
        """Test that MS/MS filtering generates message."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'MS/MS matched': ['FALSE', 'TRUE'],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1100, 2100],
            'intensity[s3]': [1200, 2200],
            'intensity[s4]': [1300, 2300],
        })
        config = QualityFilterConfig(require_msms=True)
        _, messages = MSDIALCleaner.clean(df, simple_experiment_2x2, config)
        assert any('MS/MS' in msg for msg in messages)

    def test_msms_accepts_various_true_values(self, simple_experiment_2x2):
        """Test MS/MS filter accepts TRUE, 1, YES."""
        df = pd.DataFrame({
            'LipidMolec': ['PC1', 'PC2', 'PC3', 'PC4'],
            'ClassKey': ['PC', 'PC', 'PC', 'PC'],
            'MS/MS matched': ['TRUE', 'true', '1', 'YES'],
            'intensity[s1]': [1000, 2000, 3000, 4000],
            'intensity[s2]': [1100, 2100, 3100, 4100],
            'intensity[s3]': [1200, 2200, 3200, 4200],
            'intensity[s4]': [1300, 2300, 3300, 4300],
        })
        config = QualityFilterConfig(require_msms=True)
        cleaned, _ = MSDIALCleaner.clean(df, simple_experiment_2x2, config)
        assert len(cleaned) == 4


class TestMSDIALDuplicateRemoval:
    """Tests for MS-DIAL duplicate removal."""

    def test_keeps_highest_score_duplicate(self, simple_experiment_2x2):
        """Test that duplicate with highest score is kept."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)'],
            'ClassKey': ['PC', 'PC'],
            'Total score': [70.0, 90.0],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1000, 2000],
            'intensity[s3]': [1000, 2000],
            'intensity[s4]': [1000, 2000],
        })
        cleaned, messages = MSDIALCleaner.clean(df, simple_experiment_2x2)

        assert len(cleaned) == 1
        assert cleaned['intensity[s1]'].iloc[0] == 2000

    def test_removes_multiple_duplicates(self, simple_experiment_2x2):
        """Test removing multiple duplicates."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)', 'PC(16:0)'],
            'ClassKey': ['PC', 'PC', 'PC'],
            'Total score': [70.0, 90.0, 80.0],
            'intensity[s1]': [1000, 2000, 1500],
            'intensity[s2]': [1000, 2000, 1500],
            'intensity[s3]': [1000, 2000, 1500],
            'intensity[s4]': [1000, 2000, 1500],
        })
        cleaned, _ = MSDIALCleaner.clean(df, simple_experiment_2x2)
        assert len(cleaned) == 1
        assert cleaned['intensity[s1]'].iloc[0] == 2000


class TestMSDIALColumnExtraction:
    """Tests for MS-DIAL column extraction."""

    def test_extracts_required_columns(self, simple_experiment_2x2):
        """Test extracting required columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'ExtraCol': ['extra'],
            'intensity[s1]': [1000],
            'intensity[s2]': [2000],
            'intensity[s3]': [3000],
            'intensity[s4]': [4000],
        })
        result = MSDIALCleaner._extract_columns(df, simple_experiment_2x2.full_samples_list)
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert 'ExtraCol' not in result.columns

    def test_extracts_optional_columns(self, simple_experiment_2x2):
        """Test extracting optional columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'BaseRt': [10.5],
            'CalcMass': [760.5],
            'intensity[s1]': [1000],
            'intensity[s2]': [2000],
            'intensity[s3]': [3000],
            'intensity[s4]': [4000],
        })
        result = MSDIALCleaner._extract_columns(df, simple_experiment_2x2.full_samples_list)
        assert 'BaseRt' in result.columns
        assert 'CalcMass' in result.columns

    def test_raises_on_missing_lipidmolec(self, simple_experiment_2x2):
        """Test error when LipidMolec is missing."""
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'intensity[s1]': [1000],
            'intensity[s2]': [2000],
            'intensity[s3]': [3000],
            'intensity[s4]': [4000],
        })
        with pytest.raises(KeyError, match="LipidMolec"):
            MSDIALCleaner._extract_columns(df, simple_experiment_2x2.full_samples_list)


# =============================================================================
# GenericCleaner Tests
# =============================================================================

class TestGenericCleanerBasic:
    """Basic tests for GenericCleaner."""

    def test_clean_valid_data(self, generic_df, simple_experiment_2x2):
        """Test cleaning valid Generic data."""
        cleaned, messages = GenericCleaner.clean(generic_df, simple_experiment_2x2)

        assert not cleaned.empty
        assert 'LipidMolec' in cleaned.columns
        assert 'ClassKey' in cleaned.columns
        assert len(cleaned) == 3

    def test_clean_raises_on_empty_df(self, simple_experiment_2x2):
        """Test error on empty DataFrame."""
        with pytest.raises(ValueError, match="empty"):
            GenericCleaner.clean(pd.DataFrame(), simple_experiment_2x2)

    def test_removes_duplicates(self, simple_experiment_2x2):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [1000, 2000, 3000],
            'intensity[s2]': [1000, 2000, 3000],
            'intensity[s3]': [1000, 2000, 3000],
            'intensity[s4]': [1000, 2000, 3000],
        })
        cleaned, messages = GenericCleaner.clean(df, simple_experiment_2x2)

        assert len(cleaned) == 2
        assert any('duplicate' in msg.lower() for msg in messages)

    def test_removes_invalid_rows(self, simple_experiment_2x2):
        """Test invalid row removal."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', '', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [1000, 2000, 3000],
            'intensity[s2]': [1000, 2000, 3000],
            'intensity[s3]': [1000, 2000, 3000],
            'intensity[s4]': [1000, 2000, 3000],
        })
        cleaned, messages = GenericCleaner.clean(df, simple_experiment_2x2)
        assert len(cleaned) == 2

    def test_removes_all_zero_rows(self, simple_experiment_2x2):
        """Test all-zero row removal."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000, 0],
            'intensity[s2]': [1000, 0],
            'intensity[s3]': [1000, 0],
            'intensity[s4]': [1000, 0],
        })
        cleaned, messages = GenericCleaner.clean(df, simple_experiment_2x2)
        assert len(cleaned) == 1


# =============================================================================
# DataCleaningService Integration Tests
# =============================================================================

class TestDataCleaningServiceDispatch:
    """Tests for DataCleaningService format dispatching."""

    def test_dispatch_lipidsearch(self, lipidsearch_df, simple_experiment_2x2):
        """Test dispatching to LipidSearch cleaner."""
        result = DataCleaningService.clean_data(
            lipidsearch_df, simple_experiment_2x2, DataFormat.LIPIDSEARCH
        )
        assert isinstance(result, CleaningResult)
        assert not result.cleaned_df.empty

    def test_dispatch_msdial(self, msdial_df, simple_experiment_2x2):
        """Test dispatching to MS-DIAL cleaner."""
        result = DataCleaningService.clean_data(
            msdial_df, simple_experiment_2x2, DataFormat.MSDIAL
        )
        assert isinstance(result, CleaningResult)
        assert not result.cleaned_df.empty

    def test_dispatch_generic(self, generic_df, simple_experiment_2x2):
        """Test dispatching to Generic cleaner."""
        result = DataCleaningService.clean_data(
            generic_df, simple_experiment_2x2, DataFormat.GENERIC
        )
        assert isinstance(result, CleaningResult)
        assert not result.cleaned_df.empty

    def test_dispatch_metabolomics_workbench(self, generic_df, simple_experiment_2x2):
        """Test dispatching Metabolomics Workbench to Generic cleaner."""
        result = DataCleaningService.clean_data(
            generic_df, simple_experiment_2x2, DataFormat.METABOLOMICS_WORKBENCH
        )
        assert isinstance(result, CleaningResult)
        assert not result.cleaned_df.empty

    def test_dispatch_unknown_uses_generic(self, generic_df, simple_experiment_2x2):
        """Test that unknown format uses Generic cleaner."""
        result = DataCleaningService.clean_data(
            generic_df, simple_experiment_2x2, DataFormat.UNKNOWN
        )
        assert isinstance(result, CleaningResult)


class TestDataCleaningServiceConvenience:
    """Tests for convenience methods."""

    def test_clean_lipidsearch_convenience(self, lipidsearch_df, simple_experiment_2x2):
        """Test clean_lipidsearch convenience method."""
        result = DataCleaningService.clean_lipidsearch(lipidsearch_df, simple_experiment_2x2)
        assert isinstance(result, CleaningResult)

    def test_clean_msdial_convenience(self, msdial_df, simple_experiment_2x2):
        """Test clean_msdial convenience method."""
        result = DataCleaningService.clean_msdial(msdial_df, simple_experiment_2x2)
        assert isinstance(result, CleaningResult)

    def test_clean_generic_convenience(self, generic_df, simple_experiment_2x2):
        """Test clean_generic convenience method."""
        result = DataCleaningService.clean_generic(generic_df, simple_experiment_2x2)
        assert isinstance(result, CleaningResult)

    def test_clean_lipidsearch_with_grade_config(self, lipidsearch_df, simple_experiment_2x2):
        """Test clean_lipidsearch with grade config."""
        config = GradeFilterConfig({'PC': ['A'], 'PE': ['A', 'B'], 'TG': ['A']})
        result = DataCleaningService.clean_lipidsearch(lipidsearch_df, simple_experiment_2x2, config)
        assert isinstance(result, CleaningResult)

    def test_clean_msdial_with_quality_config(self, msdial_df, simple_experiment_2x2):
        """Test clean_msdial with quality config."""
        config = QualityFilterConfig.moderate()
        result = DataCleaningService.clean_msdial(msdial_df, simple_experiment_2x2, config)
        assert isinstance(result, CleaningResult)


class TestDataCleaningServiceInternalStandards:
    """Tests for internal standards extraction via service."""

    def test_extracts_standards_from_lipidsearch(self, simple_experiment_2x2):
        """Test that internal standards are extracted from LipidSearch data."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(15:0_18:1)+D7', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'CalcMass': [760.5, 767.5, 768.5],
            'BaseRt': [10.5, 10.5, 12.3],
            'TotalGrade': ['A', 'A', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
            'FAKey': ['16:0', '15:0_18:1+D7', '18:0'],
            'intensity[s1]': [1000, 500, 2000],
            'intensity[s2]': [1100, 550, 2100],
            'intensity[s3]': [1200, 600, 2200],
            'intensity[s4]': [1300, 650, 2300],
        })
        result = DataCleaningService.clean_lipidsearch(df, simple_experiment_2x2)

        assert len(result.cleaned_df) == 2
        assert len(result.internal_standards_df) == 1

    def test_extracts_standards_from_msdial(self, simple_experiment_2x2):
        """Test that internal standards are extracted from MS-DIAL data."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(d7)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'Total score': [85.0, 85.0, 85.0],
            'intensity[s1]': [1000, 500, 2000],
            'intensity[s2]': [1100, 550, 2100],
            'intensity[s3]': [1200, 600, 2200],
            'intensity[s4]': [1300, 650, 2300],
        })
        result = DataCleaningService.clean_msdial(df, simple_experiment_2x2)

        assert len(result.cleaned_df) == 2
        assert len(result.internal_standards_df) == 1

    def test_extracts_standards_from_generic(self, simple_experiment_2x2):
        """Test that internal standards are extracted from Generic data."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(d7)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [1000, 500, 2000],
            'intensity[s2]': [1100, 550, 2100],
            'intensity[s3]': [1200, 600, 2200],
            'intensity[s4]': [1300, 650, 2300],
        })
        result = DataCleaningService.clean_generic(df, simple_experiment_2x2)

        assert len(result.cleaned_df) == 2
        assert len(result.internal_standards_df) == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_row_data(self, simple_experiment_2x2):
        """Test cleaning single row of data."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        result = DataCleaningService.clean_generic(df, simple_experiment_2x2)
        assert len(result.cleaned_df) == 1

    def test_all_zero_data_raises(self, simple_experiment_2x2):
        """Test that all-zero data raises error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0],
        })
        with pytest.raises(ValueError, match="non-zero"):
            DataCleaningService.clean_generic(df, simple_experiment_2x2)

    def test_all_invalid_lipids_raises(self, simple_experiment_2x2):
        """Test that all invalid lipid names raises error."""
        df = pd.DataFrame({
            'LipidMolec': ['', 'nan', 'Unknown'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'intensity[s1]': [1000.0, 2000.0, 3000.0],
            'intensity[s2]': [1100.0, 2100.0, 3100.0],
            'intensity[s3]': [1200.0, 2200.0, 3200.0],
            'intensity[s4]': [1300.0, 2300.0, 3300.0],
        })
        with pytest.raises(ValueError, match="invalid"):
            DataCleaningService.clean_generic(df, simple_experiment_2x2)

    def test_missing_required_column_raises(self, simple_experiment_2x2):
        """Test that missing required column raises error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            # Missing ClassKey
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        with pytest.raises(KeyError, match="ClassKey"):
            DataCleaningService.clean_generic(df, simple_experiment_2x2)

    def test_case_insensitive_columns(self, simple_experiment_2x2):
        """Test that column names are matched case-insensitively."""
        df = pd.DataFrame({
            'LIPIDMOLEC': ['PC(16:0)'],
            'classkey': ['PC'],
            'INTENSITY[s1]': [1000.0],
            'intensity[S2]': [1100.0],
            'Intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        result = DataCleaningService.clean_generic(df, simple_experiment_2x2)
        assert len(result.cleaned_df) == 1

    def test_large_dataset(self, simple_experiment_2x2):
        """Test cleaning large dataset."""
        n_rows = 1000
        df = pd.DataFrame({
            'LipidMolec': [f'PC({i}:0)' for i in range(n_rows)],
            'ClassKey': ['PC'] * n_rows,
            'intensity[s1]': np.random.rand(n_rows) * 10000,
            'intensity[s2]': np.random.rand(n_rows) * 10000,
            'intensity[s3]': np.random.rand(n_rows) * 10000,
            'intensity[s4]': np.random.rand(n_rows) * 10000,
        })
        result = DataCleaningService.clean_generic(df, simple_experiment_2x2)
        assert len(result.cleaned_df) == n_rows

    def test_unicode_lipid_names(self, simple_experiment_2x2):
        """Test handling unicode in lipid names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(α-18:0)', 'PE(β-20:4)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [1000, 2000, 3000],
            'intensity[s2]': [1100, 2100, 3100],
            'intensity[s3]': [1200, 2200, 3200],
            'intensity[s4]': [1300, 2300, 3300],
        })
        result = DataCleaningService.clean_generic(df, simple_experiment_2x2)
        assert len(result.cleaned_df) == 3


class TestLipidSearchEdgeCases:
    """Edge cases specific to LipidSearch."""

    def test_all_grade_d_raises(self, simple_experiment_2x2):
        """Test that all grade D data raises error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['D'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        with pytest.raises(ValueError, match="grade"):
            DataCleaningService.clean_lipidsearch(df, simple_experiment_2x2)

    def test_all_missing_fakey_raises(self, simple_experiment_2x2):
        """Test that all missing FA keys raises error (non-Ch)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'TotalGrade': ['A', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'FAKey': [np.nan, np.nan],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        with pytest.raises(ValueError, match="FA"):
            DataCleaningService.clean_lipidsearch(df, simple_experiment_2x2)

    def test_only_cholesterol_data(self, simple_experiment_2x2):
        """Test with only cholesterol data (no FA keys needed)."""
        df = pd.DataFrame({
            'LipidMolec': ['Ch()', 'Ch()'],
            'ClassKey': ['Ch', 'Ch'],
            'CalcMass': [386.6, 386.6],
            'BaseRt': [10.5, 10.6],
            'TotalGrade': ['A', 'B'],
            'TotalSmpIDRate(%)': [100.0, 90.0],
            'FAKey': [np.nan, np.nan],
            'intensity[s1]': [1000.0, 1100.0],
            'intensity[s2]': [1100.0, 1200.0],
            'intensity[s3]': [1200.0, 1300.0],
            'intensity[s4]': [1300.0, 1400.0],
        })
        result = DataCleaningService.clean_lipidsearch(df, simple_experiment_2x2)
        assert len(result.cleaned_df) == 1  # Deduplicated


class TestMSDIALEdgeCases:
    """Edge cases specific to MS-DIAL."""

    def test_all_filtered_by_quality_raises(self, simple_experiment_2x2):
        """Test that filtering all data raises error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'Total score': [30.0],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        config = QualityFilterConfig(total_score_threshold=80)
        with pytest.raises(ValueError, match="quality"):
            DataCleaningService.clean_msdial(df, simple_experiment_2x2, config)

    def test_missing_total_score_column(self, simple_experiment_2x2):
        """Test cleaning without Total score column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        result = DataCleaningService.clean_msdial(df, simple_experiment_2x2)
        assert len(result.cleaned_df) == 2

    def test_missing_msms_column(self, simple_experiment_2x2):
        """Test cleaning without MS/MS matched column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'Total score': [85.0, 90.0],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        config = QualityFilterConfig(require_msms=True)
        # Should not raise, MS/MS filter just won't apply
        result = DataCleaningService.clean_msdial(df, simple_experiment_2x2, config)
        assert len(result.cleaned_df) == 2

    def test_invalid_score_values(self, simple_experiment_2x2):
        """Test handling invalid score values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'Total score': ['invalid', 90.0],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        config = QualityFilterConfig(total_score_threshold=80)
        # 'invalid' should be coerced to NaN, so only 90.0 remains
        result = DataCleaningService.clean_msdial(df, simple_experiment_2x2, config)
        assert len(result.cleaned_df) == 1


# =============================================================================
# Generic Format ClassKey Edge Cases (Bug Fix Tests)
# =============================================================================

class TestGenericCleanerClassKeyEdgeCases:
    """
    Edge case tests for ClassKey column handling in Generic format.

    These tests verify that the ClassKey column is correctly identified and
    NOT counted as an intensity column. This was a bug fixed in commit 4b728ac.
    """

    def test_classkey_preserved_as_metadata(self, simple_experiment_2x2):
        """
        Test that ClassKey column is preserved in output and not treated as intensity.

        Bug context: ClassKey column was incorrectly counted as an intensity column
        when it appeared between LipidMolec and intensity columns.
        """
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        # ClassKey should be preserved as metadata
        assert 'ClassKey' in cleaned.columns
        assert cleaned['ClassKey'].iloc[0] == 'PC'
        assert cleaned['ClassKey'].iloc[1] == 'PE'

        # Should have exactly 4 intensity columns, not 5
        intensity_cols = [c for c in cleaned.columns if c.startswith('intensity[')]
        assert len(intensity_cols) == 4

    def test_classkey_case_insensitive(self, simple_experiment_2x2):
        """Test that ClassKey is recognized regardless of case."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'classkey': ['PC', 'PE'],  # lowercase
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        # Should be standardized to 'ClassKey'
        assert 'ClassKey' in cleaned.columns
        intensity_cols = [c for c in cleaned.columns if c.startswith('intensity[')]
        assert len(intensity_cols) == 4

    def test_classkey_uppercase(self, simple_experiment_2x2):
        """Test that CLASSKEY (uppercase) is recognized."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'CLASSKEY': ['PC', 'PE'],  # uppercase
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        assert 'ClassKey' in cleaned.columns
        intensity_cols = [c for c in cleaned.columns if c.startswith('intensity[')]
        assert len(intensity_cols) == 4

    def test_classkey_with_short_class_names(self, simple_experiment_2x2):
        """Test ClassKey with typical short class names like PC, PE, TG, DG."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'DG(14:0)'],
            'ClassKey': ['PC', 'PE', 'TG', 'DG'],  # 2-char names
            'intensity[s1]': [1000.0, 2000.0, 3000.0, 4000.0],
            'intensity[s2]': [1100.0, 2100.0, 3100.0, 4100.0],
            'intensity[s3]': [1200.0, 2200.0, 3200.0, 4200.0],
            'intensity[s4]': [1300.0, 2300.0, 3300.0, 4300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        assert 'ClassKey' in cleaned.columns
        assert set(cleaned['ClassKey'].unique()) == {'PC', 'PE', 'TG', 'DG'}

    def test_classkey_with_longer_class_names(self, simple_experiment_2x2):
        """Test ClassKey with longer class names like CerG1, SPH, SM."""
        df = pd.DataFrame({
            'LipidMolec': ['CerG1(18:1)', 'SPH(18:0)', 'SM(16:0)'],
            'ClassKey': ['CerG1', 'SPH', 'SM'],  # Longer names
            'intensity[s1]': [1000.0, 2000.0, 3000.0],
            'intensity[s2]': [1100.0, 2100.0, 3100.0],
            'intensity[s3]': [1200.0, 2200.0, 3200.0],
            'intensity[s4]': [1300.0, 2300.0, 3300.0],
        })
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[4]
        )
        cleaned, _ = GenericCleaner.clean(df, experiment)

        assert 'ClassKey' in cleaned.columns
        assert set(cleaned['ClassKey'].unique()) == {'CerG1', 'SPH', 'SM'}

    def test_classkey_not_confused_with_numeric_column(self, simple_experiment_2x2):
        """Test that a numeric second column is NOT treated as ClassKey."""
        # This tests the inverse case - if second column is numeric, it's intensity
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],  # Non-numeric
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        # ClassKey should NOT be treated as intensity
        assert 'ClassKey' in cleaned.columns
        assert cleaned['ClassKey'].dtype == object  # String type

    def test_classkey_values_preserved_after_cleaning(self, simple_experiment_2x2):
        """Test that ClassKey values are preserved through all cleaning steps."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(16:0)', 'PE(18:0)', ''],  # Includes duplicate and invalid
            'ClassKey': ['PC', 'PC', 'PE', 'Unknown'],
            'intensity[s1]': [1000.0, 2000.0, 3000.0, 0.0],
            'intensity[s2]': [1100.0, 2100.0, 3100.0, 0.0],
            'intensity[s3]': [1200.0, 2200.0, 3200.0, 0.0],
            'intensity[s4]': [1300.0, 2300.0, 3300.0, 0.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        # After cleaning: duplicate removed, invalid row removed, but ClassKey preserved
        assert 'ClassKey' in cleaned.columns
        assert len(cleaned) == 2  # PC (deduplicated) + PE
        assert set(cleaned['ClassKey'].unique()) == {'PC', 'PE'}


class TestGenericCleanerColumnExtraction:
    """Tests for column extraction logic in Generic format."""

    def test_intensity_column_count_with_classkey(self, simple_experiment_2x2):
        """Verify correct intensity column count when ClassKey is present."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        # Should have LipidMolec + ClassKey + 4 intensity columns = 6 total
        assert len(cleaned.columns) == 6

        # Exactly 4 intensity columns
        intensity_cols = [c for c in cleaned.columns if c.startswith('intensity[')]
        assert len(intensity_cols) == 4

    def test_missing_intensity_column_raises_error(self, simple_experiment_2x2):
        """Test that missing intensity column raises appropriate error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            # Missing s3 and s4
        })
        with pytest.raises(KeyError, match="intensity"):
            GenericCleaner.clean(df, simple_experiment_2x2)

    def test_extra_columns_ignored(self, simple_experiment_2x2):
        """Test that extra columns are ignored in output."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'ExtraMetadata': ['some_value'],  # Extra column
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })
        cleaned, _ = GenericCleaner.clean(df, simple_experiment_2x2)

        # ExtraMetadata should be ignored
        assert 'ExtraMetadata' not in cleaned.columns
        assert len(cleaned.columns) == 6  # LipidMolec + ClassKey + 4 intensity


# =============================================================================
# Sample Column Detection Pattern Tests
# =============================================================================

class TestSampleColumnDetection:
    """
    Tests for sample column detection using intensity[...] pattern.

    After standardization, all formats use intensity[sample_name] columns.
    Bug context: Sample detection was using wrong pattern after standardization.
    """

    def test_intensity_pattern_detection(self, simple_experiment_2x2):
        """Test that intensity[...] columns are correctly identified."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })

        # Get intensity columns using the standard pattern
        intensity_cols = BaseDataCleaner.get_intensity_columns(df)

        assert len(intensity_cols) == 4
        assert all(col.startswith('intensity[') for col in intensity_cols)
        assert 'intensity[s1]' in intensity_cols
        assert 'intensity[s4]' in intensity_cols

    def test_non_intensity_columns_excluded(self, simple_experiment_2x2):
        """Test that non-intensity columns are not detected as samples."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],  # Metadata column
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
        })

        intensity_cols = BaseDataCleaner.get_intensity_columns(df)

        assert 'LipidMolec' not in intensity_cols
        assert 'ClassKey' not in intensity_cols
        assert 'CalcMass' not in intensity_cols
        assert len(intensity_cols) == 4

    def test_sample_names_with_special_characters(self):
        """Test intensity columns with special characters in sample names."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[2, 2]
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[Sample_1]': [1000.0],
            'intensity[Sample-2]': [1100.0],
            'intensity[Sample.3]': [1200.0],
            'intensity[Sample 4]': [1300.0],  # Space in name
        })

        intensity_cols = BaseDataCleaner.get_intensity_columns(df)

        assert len(intensity_cols) == 4
        assert 'intensity[Sample_1]' in intensity_cols
        assert 'intensity[Sample 4]' in intensity_cols

    def test_empty_dataframe_no_intensity_columns(self):
        """Test that empty DataFrame returns no intensity columns."""
        df = pd.DataFrame()

        intensity_cols = BaseDataCleaner.get_intensity_columns(df)

        assert len(intensity_cols) == 0

    def test_dataframe_without_intensity_columns(self):
        """Test DataFrame with no intensity columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'OtherColumn': [1000.0],
        })

        intensity_cols = BaseDataCleaner.get_intensity_columns(df)

        assert len(intensity_cols) == 0
