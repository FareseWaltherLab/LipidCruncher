"""Unit tests for shared validation utilities in app/services/validation.py.

Covers: validate_dataframe_not_empty, get_matching_concentration_columns,
validate_concentration_columns.
"""
import pytest
import pandas as pd

from app.models.experiment import ExperimentConfig
from app.services.validation import (
    get_matching_concentration_columns,
    validate_concentration_columns,
    validate_dataframe_not_empty,
)
from app.services.exceptions import EmptyDataError, ValidationError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def experiment_2x3():
    """2 conditions x 3 samples = 6 total."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )


@pytest.fixture
def full_conc_df():
    """DataFrame with all 6 concentration columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4'],
        'ClassKey': ['PC', 'PE'],
        'concentration[s1]': [100.0, 200.0],
        'concentration[s2]': [110.0, 210.0],
        'concentration[s3]': [120.0, 220.0],
        'concentration[s4]': [130.0, 230.0],
        'concentration[s5]': [140.0, 240.0],
        'concentration[s6]': [150.0, 250.0],
    })


@pytest.fixture
def partial_conc_df():
    """DataFrame with only 4 of 6 expected concentration columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1'],
        'ClassKey': ['PC'],
        'concentration[s1]': [100.0],
        'concentration[s2]': [110.0],
        'concentration[s3]': [120.0],
        'concentration[s4]': [130.0],
    })


@pytest.fixture
def no_conc_df():
    """DataFrame with no concentration columns at all."""
    return pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1'],
        'ClassKey': ['PC'],
        'intensity[s1]': [100.0],
    })


# =============================================================================
# validate_dataframe_not_empty
# =============================================================================


class TestValidateDataframeNotEmpty:
    """Tests for validate_dataframe_not_empty()."""

    def test_valid_dataframe_passes(self, full_conc_df):
        """Non-empty DataFrame should not raise."""
        validate_dataframe_not_empty(full_conc_df)  # no exception

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise EmptyDataError."""
        with pytest.raises(EmptyDataError, match="empty"):
            validate_dataframe_not_empty(pd.DataFrame())

    def test_none_raises(self):
        """None input should raise EmptyDataError."""
        with pytest.raises(EmptyDataError, match="empty"):
            validate_dataframe_not_empty(None)

    def test_single_row_passes(self):
        """DataFrame with one row should pass."""
        df = pd.DataFrame({'A': [1]})
        validate_dataframe_not_empty(df)

    def test_empty_dataframe_is_service_error(self):
        """EmptyDataError should be catchable as ValueError (backward compat)."""
        with pytest.raises(ValueError):
            validate_dataframe_not_empty(pd.DataFrame())


# =============================================================================
# get_matching_concentration_columns
# =============================================================================


class TestGetMatchingConcentrationColumns:
    """Tests for get_matching_concentration_columns()."""

    def test_all_columns_present(self, full_conc_df, experiment_2x3):
        """All 6 samples should be returned when all columns exist."""
        result = get_matching_concentration_columns(full_conc_df, experiment_2x3)
        assert result == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_partial_columns(self, partial_conc_df, experiment_2x3):
        """Only samples with matching columns should be returned."""
        result = get_matching_concentration_columns(partial_conc_df, experiment_2x3)
        assert result == ['s1', 's2', 's3', 's4']
        assert 's5' not in result
        assert 's6' not in result

    def test_no_matching_columns(self, no_conc_df, experiment_2x3):
        """No matches should return empty list."""
        result = get_matching_concentration_columns(no_conc_df, experiment_2x3)
        assert result == []

    def test_preserves_sample_order(self, experiment_2x3):
        """Returned samples should follow experiment order, not column order."""
        df = pd.DataFrame({
            'concentration[s4]': [1.0],
            'concentration[s1]': [2.0],
            'concentration[s6]': [3.0],
        })
        result = get_matching_concentration_columns(df, experiment_2x3)
        assert result == ['s1', 's4', 's6']

    def test_extra_columns_ignored(self, experiment_2x3):
        """Columns not in the experiment should be ignored."""
        df = pd.DataFrame({
            'concentration[s1]': [1.0],
            'concentration[s99]': [2.0],
        })
        result = get_matching_concentration_columns(df, experiment_2x3)
        assert result == ['s1']

    def test_empty_dataframe_returns_empty(self, experiment_2x3):
        """Empty DataFrame returns empty list (has column names but no rows)."""
        df = pd.DataFrame(columns=['concentration[s1]', 'concentration[s2]'])
        result = get_matching_concentration_columns(df, experiment_2x3)
        assert result == ['s1', 's2']


# =============================================================================
# validate_concentration_columns
# =============================================================================


class TestValidateConcentrationColumns:
    """Tests for validate_concentration_columns()."""

    def test_all_columns_present(self, full_conc_df, experiment_2x3):
        """Should return all sample labels when all columns exist."""
        result = validate_concentration_columns(full_conc_df, experiment_2x3)
        assert len(result) == 6

    def test_partial_columns_returns_available(self, partial_conc_df, experiment_2x3):
        """Should return only available samples without raising."""
        result = validate_concentration_columns(partial_conc_df, experiment_2x3)
        assert result == ['s1', 's2', 's3', 's4']

    def test_no_columns_raises_validation_error(self, no_conc_df, experiment_2x3):
        """Should raise ValidationError when no columns match."""
        with pytest.raises(ValidationError, match="no concentration columns"):
            validate_concentration_columns(no_conc_df, experiment_2x3)

    def test_validation_error_is_service_error(self, no_conc_df, experiment_2x3):
        """ValidationError should be catchable as ValueError (backward compat)."""
        with pytest.raises(ValueError):
            validate_concentration_columns(no_conc_df, experiment_2x3)

    def test_intensity_columns_not_accepted(self, experiment_2x3):
        """Intensity columns should not satisfy concentration validation."""
        df = pd.DataFrame({
            'intensity[s1]': [1.0],
            'intensity[s2]': [2.0],
        })
        with pytest.raises(ValidationError):
            validate_concentration_columns(df, experiment_2x3)