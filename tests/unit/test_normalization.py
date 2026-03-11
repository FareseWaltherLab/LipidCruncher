"""Unit tests for NormalizationService."""
import pytest
import pandas as pd
import numpy as np
from app.services.normalization import (
    NormalizationService,
    NormalizationResult,
)
from app.models.normalization import NormalizationConfig
from app.models.experiment import ExperimentConfig

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def single_sample_experiment():
    """Experiment with single sample per condition."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['A', 'B'],
        number_of_samples_list=[1, 1]
    )



@pytest.fixture
def basic_lipid_df(simple_experiment_2x3):
    """Basic lipid DataFrame with intensity values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [1000.0, 2000.0, 3000.0],
        'intensity[s2]': [1100.0, 2100.0, 3100.0],
        'intensity[s3]': [1200.0, 2200.0, 3200.0],
        'intensity[s4]': [1300.0, 2300.0, 3300.0],
        'intensity[s5]': [1400.0, 2400.0, 3400.0],
        'intensity[s6]': [1500.0, 2500.0, 3500.0],
    })


@pytest.fixture
def multi_class_df(simple_experiment_2x3):
    """DataFrame with multiple lipids per class."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(18:0_18:1)', 'PC(16:0_20:4)',
            'PE(18:0_20:4)', 'PE(16:0_22:6)',
            'TG(16:0_18:1_18:2)',
        ],
        'ClassKey': ['PC', 'PC', 'PC', 'PE', 'PE', 'TG'],
        'intensity[s1]': [1000.0, 1100.0, 1200.0, 2000.0, 2100.0, 3000.0],
        'intensity[s2]': [1050.0, 1150.0, 1250.0, 2050.0, 2150.0, 3050.0],
        'intensity[s3]': [1100.0, 1200.0, 1300.0, 2100.0, 2200.0, 3100.0],
        'intensity[s4]': [1150.0, 1250.0, 1350.0, 2150.0, 2250.0, 3150.0],
        'intensity[s5]': [1200.0, 1300.0, 1400.0, 2200.0, 2300.0, 3200.0],
        'intensity[s6]': [1250.0, 1350.0, 1450.0, 2250.0, 2350.0, 3250.0],
    })


@pytest.fixture
def df_with_standards(simple_experiment_2x3):
    """DataFrame that includes internal standards as lipids."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(18:0_18:1)',
            'PC(15:0_15:0)_IS',
            'PE(18:0_20:4)',
            'PE(17:0_17:0)_IS',
        ],
        'ClassKey': ['PC', 'PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [1000.0, 1100.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1050.0, 1150.0, 510.0, 2050.0, 610.0],
        'intensity[s3]': [1100.0, 1200.0, 520.0, 2100.0, 620.0],
        'intensity[s4]': [1150.0, 1250.0, 530.0, 2150.0, 630.0],
        'intensity[s5]': [1200.0, 1300.0, 540.0, 2200.0, 640.0],
        'intensity[s6]': [1250.0, 1350.0, 550.0, 2250.0, 650.0],
    })


@pytest.fixture
def intsta_df(simple_experiment_2x3):
    """Internal standards DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_15:0)_IS', 'PE(17:0_17:0)_IS', 'TG(15:0_15:0_15:0)_IS'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [500.0, 600.0, 700.0],
        'intensity[s2]': [510.0, 610.0, 710.0],
        'intensity[s3]': [520.0, 620.0, 720.0],
        'intensity[s4]': [530.0, 630.0, 730.0],
        'intensity[s5]': [540.0, 640.0, 740.0],
        'intensity[s6]': [550.0, 650.0, 750.0],
    })


@pytest.fixture
def intsta_df_single_standard():
    """Internal standards DataFrame with single standard."""
    return pd.DataFrame({
        'LipidMolec': ['Universal_IS'],
        'ClassKey': ['IS'],
        'intensity[s1]': [1000.0],
        'intensity[s2]': [1000.0],
        'intensity[s3]': [1000.0],
        'intensity[s4]': [1000.0],
        'intensity[s5]': [1000.0],
        'intensity[s6]': [1000.0],
    })


@pytest.fixture
def intsta_df_with_zeros(simple_experiment_2x3):
    """Internal standards DataFrame with some zero values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_15:0)_IS', 'PE(17:0_17:0)_IS'],
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [500.0, 0.0],
        'intensity[s2]': [510.0, 610.0],
        'intensity[s3]': [520.0, 620.0],
        'intensity[s4]': [0.0, 630.0],
        'intensity[s5]': [540.0, 640.0],
        'intensity[s6]': [550.0, 650.0],
    })


@pytest.fixture
def intsta_df_all_zeros():
    """Internal standards DataFrame with all zero values for one standard."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_15:0)_IS'],
        'ClassKey': ['PC'],
        'intensity[s1]': [0.0],
        'intensity[s2]': [0.0],
        'intensity[s3]': [0.0],
        'intensity[s4]': [0.0],
        'intensity[s5]': [0.0],
        'intensity[s6]': [0.0],
    })


@pytest.fixture
def protein_concentrations():
    """Protein concentrations for samples."""
    return {
        's1': 1.5,
        's2': 1.6,
        's3': 1.4,
        's4': 1.7,
        's5': 1.5,
        's6': 1.8,
    }


@pytest.fixture
def uniform_protein_concentrations():
    """Uniform protein concentrations for easier calculation verification."""
    return {
        's1': 2.0,
        's2': 2.0,
        's3': 2.0,
        's4': 2.0,
        's5': 2.0,
        's6': 2.0,
    }


@pytest.fixture
def none_config():
    """Configuration for no normalization."""
    return NormalizationConfig(method='none')


@pytest.fixture
def none_config_with_classes():
    """Configuration for no normalization with class selection."""
    return NormalizationConfig(
        method='none',
        selected_classes=['PC', 'PE']
    )


@pytest.fixture
def internal_standard_config():
    """Configuration for internal standard normalization."""
    return NormalizationConfig(
        method='internal_standard',
        selected_classes=['PC', 'PE'],
        internal_standards={
            'PC': 'PC(15:0_15:0)_IS',
            'PE': 'PE(17:0_17:0)_IS',
        },
        intsta_concentrations={
            'PC(15:0_15:0)_IS': 10.0,
            'PE(17:0_17:0)_IS': 5.0,
        }
    )


@pytest.fixture
def internal_standard_config_all_classes():
    """Configuration for internal standard normalization with all classes."""
    return NormalizationConfig(
        method='internal_standard',
        selected_classes=['PC', 'PE', 'TG'],
        internal_standards={
            'PC': 'PC(15:0_15:0)_IS',
            'PE': 'PE(17:0_17:0)_IS',
            'TG': 'TG(15:0_15:0_15:0)_IS',
        },
        intsta_concentrations={
            'PC(15:0_15:0)_IS': 10.0,
            'PE(17:0_17:0)_IS': 5.0,
            'TG(15:0_15:0_15:0)_IS': 8.0,
        }
    )


@pytest.fixture
def protein_config(protein_concentrations):
    """Configuration for protein normalization."""
    return NormalizationConfig(
        method='protein',
        protein_concentrations=protein_concentrations
    )


@pytest.fixture
def protein_config_uniform(uniform_protein_concentrations):
    """Configuration for protein normalization with uniform concentrations."""
    return NormalizationConfig(
        method='protein',
        protein_concentrations=uniform_protein_concentrations
    )


@pytest.fixture
def both_config(protein_concentrations):
    """Configuration for combined normalization."""
    return NormalizationConfig(
        method='both',
        selected_classes=['PC', 'PE'],
        internal_standards={
            'PC': 'PC(15:0_15:0)_IS',
            'PE': 'PE(17:0_17:0)_IS',
        },
        intsta_concentrations={
            'PC(15:0_15:0)_IS': 10.0,
            'PE(17:0_17:0)_IS': 5.0,
        },
        protein_concentrations=protein_concentrations
    )


# =============================================================================
# NormalizationResult Tests
# =============================================================================

class TestNormalizationResultProperties:
    """Tests for NormalizationResult dataclass."""

    def test_result_has_normalized_df(self, basic_lipid_df):
        """Test that NormalizationResult has normalized_df field."""
        result = NormalizationResult(
            normalized_df=basic_lipid_df,
            removed_standards=[],
            method_applied='Test'
        )
        assert hasattr(result, 'normalized_df')

    def test_result_has_removed_standards(self, basic_lipid_df):
        """Test that NormalizationResult has removed_standards field."""
        result = NormalizationResult(
            normalized_df=basic_lipid_df,
            removed_standards=['IS1'],
            method_applied='Test'
        )
        assert hasattr(result, 'removed_standards')

    def test_result_has_method_applied(self, basic_lipid_df):
        """Test that NormalizationResult has method_applied field."""
        result = NormalizationResult(
            normalized_df=basic_lipid_df,
            removed_standards=[],
            method_applied='Test method'
        )
        assert hasattr(result, 'method_applied')

    def test_result_stores_dataframe_correctly(self, basic_lipid_df):
        """Test that normalized_df stores DataFrame correctly."""
        result = NormalizationResult(
            normalized_df=basic_lipid_df,
            removed_standards=[],
            method_applied='Test'
        )
        pd.testing.assert_frame_equal(result.normalized_df, basic_lipid_df)

    def test_result_stores_standards_list(self):
        """Test that removed_standards stores list correctly."""
        standards = ['IS_1', 'IS_2', 'IS_3']
        result = NormalizationResult(
            normalized_df=pd.DataFrame(),
            removed_standards=standards,
            method_applied='Test'
        )
        assert result.removed_standards == standards

    def test_result_stores_empty_standards_list(self):
        """Test that empty removed_standards list works."""
        result = NormalizationResult(
            normalized_df=pd.DataFrame(),
            removed_standards=[],
            method_applied='Test'
        )
        assert result.removed_standards == []

    def test_result_stores_method_string(self):
        """Test that method_applied stores string correctly."""
        method = "Internal standards normalization"
        result = NormalizationResult(
            normalized_df=pd.DataFrame(),
            removed_standards=[],
            method_applied=method
        )
        assert result.method_applied == method


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidationEmpty:
    """Tests for empty/None DataFrame validation."""

    def test_empty_dataframe_raises_error(self, simple_experiment_2x3, none_config):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            NormalizationService.normalize(empty_df, none_config, simple_experiment_2x3)

    def test_none_dataframe_raises_error(self, simple_experiment_2x3, none_config):
        """Test that None DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            NormalizationService.normalize(None, none_config, simple_experiment_2x3)

    def test_dataframe_with_no_rows_raises_error(self, simple_experiment_2x3, none_config):
        """Test that DataFrame with columns but no rows raises error."""
        df = pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'intensity[s1]'])
        with pytest.raises(ValueError, match="empty"):
            NormalizationService.normalize(df, none_config, simple_experiment_2x3)


class TestInputValidationColumns:
    """Tests for required column validation."""

    def test_missing_lipidmolec_column(self, simple_experiment_2x3, none_config):
        """Test that missing LipidMolec column raises ValueError."""
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0],
            'intensity[s5]': [100.0],
            'intensity[s6]': [100.0],
        })
        with pytest.raises(ValueError, match="LipidMolec"):
            NormalizationService.normalize(df, none_config, simple_experiment_2x3)

    def test_missing_classkey_column(self, simple_experiment_2x3, none_config):
        """Test that missing ClassKey column raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0],
            'intensity[s5]': [100.0],
            'intensity[s6]': [100.0],
        })
        with pytest.raises(ValueError, match="ClassKey"):
            NormalizationService.normalize(df, none_config, simple_experiment_2x3)

    def test_missing_both_required_columns(self, simple_experiment_2x3, none_config):
        """Test that missing both required columns raises ValueError."""
        df = pd.DataFrame({
            'OtherCol': ['value'],
            'intensity[s1]': [100.0],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            NormalizationService.normalize(df, none_config, simple_experiment_2x3)

    def test_missing_intensity_columns(self, simple_experiment_2x3, none_config):
        """Test that missing intensity columns raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'other_column': [100.0],
        })
        with pytest.raises(ValueError, match="intensity"):
            NormalizationService.normalize(df, none_config, simple_experiment_2x3)

    def test_valid_dataframe_passes_validation(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that valid DataFrame passes validation."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert isinstance(result, NormalizationResult)


# =============================================================================
# No Normalization Method Tests
# =============================================================================

class TestNoNormalizationBasic:
    """Basic tests for 'none' normalization method."""

    def test_none_method_returns_result(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method returns NormalizationResult."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert isinstance(result, NormalizationResult)

    def test_none_method_returns_copy(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method returns a copy of the data."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        basic_lipid_df.loc[0, 'intensity[s1]'] = 9999.0
        assert result.normalized_df.loc[0, 'concentration[s1]'] != 9999.0

    def test_none_method_preserves_row_count(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method preserves row count."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert len(result.normalized_df) == len(basic_lipid_df)

    def test_none_method_preserves_values(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method preserves original values."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 1000.0
        assert result.normalized_df.loc[1, 'concentration[s2]'] == 2100.0

    def test_none_method_empty_removed_standards(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method has empty removed_standards list."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert result.removed_standards == []

    def test_none_method_correct_method_applied(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method reports correct method_applied."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert 'None' in result.method_applied or 'none' in result.method_applied.lower()


class TestNoNormalizationColumnRenaming:
    """Tests for column renaming in 'none' method."""

    def test_none_method_renames_intensity_to_concentration(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method renames intensity to concentration."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert 'concentration[s1]' in result.normalized_df.columns
        assert 'intensity[s1]' not in result.normalized_df.columns

    def test_none_method_renames_all_intensity_columns(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that 'none' method renames all intensity columns."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        for i in range(1, 7):
            assert f'concentration[s{i}]' in result.normalized_df.columns
            assert f'intensity[s{i}]' not in result.normalized_df.columns

    def test_none_method_preserves_non_intensity_columns(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that non-intensity columns are preserved."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert 'LipidMolec' in result.normalized_df.columns
        assert 'ClassKey' in result.normalized_df.columns


class TestNoNormalizationClassFiltering:
    """Tests for class filtering in 'none' method."""

    def test_none_method_filters_by_selected_classes(self, multi_class_df, simple_experiment_2x3, none_config_with_classes):
        """Test that 'none' method filters by selected classes."""
        result = NormalizationService.normalize(multi_class_df, none_config_with_classes, simple_experiment_2x3)
        classes = result.normalized_df['ClassKey'].unique()
        assert 'PC' in classes
        assert 'PE' in classes
        assert 'TG' not in classes

    def test_none_method_returns_all_classes_when_none_selected(self, multi_class_df, simple_experiment_2x3, none_config):
        """Test that 'none' method returns all classes when none selected."""
        result = NormalizationService.normalize(multi_class_df, none_config, simple_experiment_2x3)
        classes = result.normalized_df['ClassKey'].unique()
        assert len(classes) == 3

    def test_none_method_single_class_filter(self, multi_class_df, simple_experiment_2x3):
        """Test filtering to single class."""
        config = NormalizationConfig(method='none', selected_classes=['PC'])
        result = NormalizationService.normalize(multi_class_df, config, simple_experiment_2x3)
        assert all(result.normalized_df['ClassKey'] == 'PC')

    def test_none_method_preserves_lipids_in_selected_classes(self, multi_class_df, simple_experiment_2x3, none_config_with_classes):
        """Test that lipids in selected classes are preserved."""
        result = NormalizationService.normalize(multi_class_df, none_config_with_classes, simple_experiment_2x3)
        lipids = result.normalized_df['LipidMolec'].tolist()
        assert 'PC(16:0_18:1)' in lipids
        assert 'PE(18:0_20:4)' in lipids


# =============================================================================
# Internal Standard Normalization Tests
# =============================================================================

class TestInternalStandardValidation:
    """Tests for internal standard normalization validation."""

    def test_is_normalization_requires_intsta_df(self, basic_lipid_df, simple_experiment_2x3, internal_standard_config):
        """Test that IS normalization requires intsta_df."""
        with pytest.raises(ValueError, match="Internal standards DataFrame is required"):
            NormalizationService.normalize(
                basic_lipid_df, internal_standard_config, simple_experiment_2x3, intsta_df=None
            )

    def test_is_normalization_requires_nonempty_intsta_df(self, basic_lipid_df, simple_experiment_2x3, internal_standard_config):
        """Test that IS normalization requires non-empty intsta_df."""
        with pytest.raises(ValueError, match="Internal standards DataFrame is required"):
            NormalizationService.normalize(
                basic_lipid_df, internal_standard_config, simple_experiment_2x3, intsta_df=pd.DataFrame()
            )

    def test_is_normalization_validates_intsta_df_lipidmolec(self, basic_lipid_df, simple_experiment_2x3, internal_standard_config):
        """Test that IS normalization validates intsta_df has LipidMolec."""
        bad_intsta = pd.DataFrame({
            'OtherCol': ['IS1'],
            'intensity[s1]': [100.0],
        })
        with pytest.raises(ValueError, match="LipidMolec"):
            NormalizationService.normalize(
                basic_lipid_df, internal_standard_config, simple_experiment_2x3, intsta_df=bad_intsta
            )

    def test_is_normalization_validates_intsta_df_intensity(self, basic_lipid_df, simple_experiment_2x3, internal_standard_config):
        """Test that IS normalization validates intsta_df has intensity columns."""
        bad_intsta = pd.DataFrame({
            'LipidMolec': ['PC(15:0_15:0)_IS'],
            'other_col': [100.0],
        })
        with pytest.raises(ValueError, match="intensity"):
            NormalizationService.normalize(
                basic_lipid_df, internal_standard_config, simple_experiment_2x3, intsta_df=bad_intsta
            )


class TestInternalStandardRemoval:
    """Tests for internal standard removal."""

    def test_is_normalization_removes_standards_from_output(self, df_with_standards, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization removes standards from output."""
        result = NormalizationService.normalize(
            df_with_standards, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        lipid_names = result.normalized_df['LipidMolec'].tolist()
        assert 'PC(15:0_15:0)_IS' not in lipid_names
        assert 'PE(17:0_17:0)_IS' not in lipid_names

    def test_is_normalization_reports_removed_standards(self, df_with_standards, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization reports removed standards."""
        result = NormalizationService.normalize(
            df_with_standards, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert 'PC(15:0_15:0)_IS' in result.removed_standards
        assert 'PE(17:0_17:0)_IS' in result.removed_standards

    def test_is_normalization_keeps_non_standard_lipids(self, df_with_standards, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization keeps non-standard lipids."""
        result = NormalizationService.normalize(
            df_with_standards, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        lipid_names = result.normalized_df['LipidMolec'].tolist()
        assert 'PC(16:0_18:1)' in lipid_names
        assert 'PC(18:0_18:1)' in lipid_names
        assert 'PE(18:0_20:4)' in lipid_names


class TestInternalStandardFormula:
    """Tests for internal standard normalization formula."""

    def test_is_normalization_applies_correct_formula(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization applies: (intensity/standard) * concentration."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        # PC(16:0_18:1) in s1: (1000 / 500) * 10.0 = 20.0
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        expected = (1000.0 / 500.0) * 10.0
        assert abs(pc_row['concentration[s1]'] - expected) < 0.001

    def test_is_normalization_different_samples_same_lipid(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test IS normalization across different samples for same lipid."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        # s1: (1000 / 500) * 10 = 20
        # s2: (1050 / 510) * 10 = 20.588
        assert abs(pc_row['concentration[s1]'] - 20.0) < 0.001
        assert abs(pc_row['concentration[s2]'] - (1050 / 510 * 10)) < 0.001

    def test_is_normalization_different_standards_per_class(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization uses different standards for different classes."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        pe_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PE(18:0_20:4)'].iloc[0]
        # PC s1: (1000 / 500) * 10 = 20
        # PE s1: (2000 / 600) * 5 = 16.67
        assert abs(pc_row['concentration[s1]'] - 20.0) < 0.001
        assert abs(pe_row['concentration[s1]'] - (2000 / 600 * 5)) < 0.001

    def test_is_normalization_different_concentrations(self, multi_class_df, simple_experiment_2x3, intsta_df):
        """Test IS normalization with different standard concentrations."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(15:0_15:0)_IS',
                'PE': 'PE(17:0_17:0)_IS',
            },
            intsta_concentrations={
                'PC(15:0_15:0)_IS': 1.0,  # Low concentration
                'PE(17:0_17:0)_IS': 100.0,  # High concentration
            }
        )
        result = NormalizationService.normalize(
            multi_class_df, config, simple_experiment_2x3, intsta_df=intsta_df
        )
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        pe_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PE(18:0_20:4)'].iloc[0]
        # PC s1: (1000 / 500) * 1 = 2
        # PE s1: (2000 / 600) * 100 = 333.33
        assert abs(pc_row['concentration[s1]'] - 2.0) < 0.001
        assert abs(pe_row['concentration[s1]'] - (2000 / 600 * 100)) < 0.01


class TestInternalStandardClassFiltering:
    """Tests for class filtering in IS normalization."""

    def test_is_normalization_filters_by_selected_classes(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization only includes selected classes."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        classes = result.normalized_df['ClassKey'].unique()
        assert 'PC' in classes
        assert 'PE' in classes
        assert 'TG' not in classes

    def test_is_normalization_all_classes(self, multi_class_df, simple_experiment_2x3, internal_standard_config_all_classes, intsta_df):
        """Test IS normalization with all classes selected."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config_all_classes, simple_experiment_2x3, intsta_df=intsta_df
        )
        classes = result.normalized_df['ClassKey'].unique()
        assert len(classes) == 3


class TestInternalStandardMissingMapping:
    """Tests for missing standard mappings."""

    def test_is_normalization_missing_standard_in_intsta_df(self, basic_lipid_df, simple_experiment_2x3, intsta_df):
        """Test error when mapped standard not found in intsta_df."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'NonExistentStandard'},
            intsta_concentrations={'NonExistentStandard': 10.0}
        )
        with pytest.raises(ValueError, match="not found"):
            NormalizationService.normalize(basic_lipid_df, config, simple_experiment_2x3, intsta_df=intsta_df)

    def test_is_normalization_missing_class_mapping(self, multi_class_df, simple_experiment_2x3, intsta_df):
        """Test error when class has no standard mapping."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'TG'],
            internal_standards={
                'PC': 'PC(15:0_15:0)_IS',
                'PE': 'PE(17:0_17:0)_IS',
                # TG mapping missing
            },
            intsta_concentrations={
                'PC(15:0_15:0)_IS': 10.0,
                'PE(17:0_17:0)_IS': 5.0,
            }
        )
        with pytest.raises(ValueError, match="No internal standard mapped"):
            NormalizationService.normalize(multi_class_df, config, simple_experiment_2x3, intsta_df=intsta_df)


class TestInternalStandardZeroHandling:
    """Tests for handling zero values in internal standards."""

    def test_is_normalization_zero_standard_produces_zero(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df_with_zeros):
        """Test that division by zero standard produces zero (not inf)."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df_with_zeros
        )
        assert not np.isinf(result.normalized_df['concentration[s1]']).any()

    def test_is_normalization_replaces_inf_with_zero(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df_with_zeros):
        """Test that infinity values are replaced with zero."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df_with_zeros
        )
        for col in result.normalized_df.columns:
            if col.startswith('concentration['):
                assert not np.isinf(result.normalized_df[col]).any()

    def test_is_normalization_replaces_nan_with_zero(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df_with_zeros):
        """Test that NaN values are replaced with zero."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df_with_zeros
        )
        for col in result.normalized_df.columns:
            if col.startswith('concentration['):
                assert not result.normalized_df[col].isna().any()

    def test_is_normalization_all_zeros_standard(self, multi_class_df, simple_experiment_2x3, intsta_df_all_zeros):
        """Test IS normalization when standard has all zero values."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(15:0_15:0)_IS'},
            intsta_concentrations={'PC(15:0_15:0)_IS': 10.0}
        )
        result = NormalizationService.normalize(
            multi_class_df, config, simple_experiment_2x3, intsta_df=intsta_df_all_zeros
        )
        # All values should be 0 since standard was all zeros
        for col in result.normalized_df.columns:
            if col.startswith('concentration['):
                assert (result.normalized_df[col] == 0).all()


class TestInternalStandardColumnRenaming:
    """Tests for column renaming in IS normalization."""

    def test_is_normalization_renames_columns(self, multi_class_df, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that IS normalization renames intensity to concentration."""
        result = NormalizationService.normalize(
            multi_class_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert 'concentration[s1]' in result.normalized_df.columns
        assert 'intensity[s1]' not in result.normalized_df.columns


# =============================================================================
# Protein Normalization Tests
# =============================================================================

class TestProteinNormalizationValidation:
    """Tests for protein normalization validation."""

    def test_protein_normalization_requires_concentrations(self):
        """Test that protein normalization requires protein_concentrations (Pydantic validation)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="protein_concentrations"):
            NormalizationConfig(method='protein')

    def test_protein_normalization_empty_concentrations(self):
        """Test that empty protein_concentrations raises error (Pydantic validation)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="protein_concentrations"):
            NormalizationConfig(method='protein', protein_concentrations={})


class TestProteinNormalizationFormula:
    """Tests for protein normalization formula."""

    def test_protein_normalization_applies_formula(self, basic_lipid_df, simple_experiment_2x3, protein_config):
        """Test that protein normalization applies: intensity / protein_conc."""
        result = NormalizationService.normalize(basic_lipid_df, protein_config, simple_experiment_2x3)
        # s1 has protein conc 1.5: 1000 / 1.5 = 666.67
        expected = 1000.0 / 1.5
        assert abs(result.normalized_df.loc[0, 'concentration[s1]'] - expected) < 0.01

    def test_protein_normalization_all_samples(self, basic_lipid_df, simple_experiment_2x3, protein_config):
        """Test protein normalization applies to all samples."""
        result = NormalizationService.normalize(basic_lipid_df, protein_config, simple_experiment_2x3)
        assert abs(result.normalized_df.loc[0, 'concentration[s1]'] - (1000 / 1.5)) < 0.01
        assert abs(result.normalized_df.loc[0, 'concentration[s6]'] - (1500 / 1.8)) < 0.01

    def test_protein_normalization_uniform_concentrations(self, basic_lipid_df, simple_experiment_2x3, protein_config_uniform):
        """Test protein normalization with uniform concentrations."""
        result = NormalizationService.normalize(basic_lipid_df, protein_config_uniform, simple_experiment_2x3)
        # All divided by 2.0
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 500.0
        assert result.normalized_df.loc[0, 'concentration[s2]'] == 550.0

    def test_protein_normalization_all_lipids(self, basic_lipid_df, simple_experiment_2x3, protein_config_uniform):
        """Test protein normalization applies to all lipids."""
        result = NormalizationService.normalize(basic_lipid_df, protein_config_uniform, simple_experiment_2x3)
        # All values should be halved
        for i in range(len(result.normalized_df)):
            for j in range(1, 7):
                original = basic_lipid_df.loc[i, f'intensity[s{j}]']
                normalized = result.normalized_df.loc[i, f'concentration[s{j}]']
                assert normalized == original / 2.0


class TestProteinNormalizationZeroHandling:
    """Tests for handling zero/negative protein concentrations."""

    def test_protein_normalization_rejects_zero_concentration(self):
        """Test that Pydantic validation rejects zero concentration."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="must be positive"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={
                    's1': 1.5,
                    's2': 0.0,  # Zero - rejected by Pydantic
                    's3': 1.4,
                    's4': 1.7,
                    's5': 1.5,
                    's6': 1.8,
                }
            )

    def test_protein_normalization_rejects_negative_concentration(self):
        """Test that Pydantic validation rejects negative concentration."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="must be positive"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={
                    's1': 1.5,
                    's2': -1.0,  # Negative - rejected by Pydantic
                    's3': 1.4,
                    's4': 1.7,
                    's5': 1.5,
                    's6': 1.8,
                }
            )

    def test_protein_normalization_missing_sample_skipped(self, basic_lipid_df, simple_experiment_2x3):
        """Test that samples without protein concentration are skipped."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={
                's1': 1.5,
                # s2 missing
                's3': 1.4,
                's4': 1.7,
                's5': 1.5,
                's6': 1.8,
            }
        )
        result = NormalizationService.normalize(basic_lipid_df, config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s2]'] == 1100.0


class TestProteinNormalizationClassFiltering:
    """Tests for class filtering in protein normalization."""

    def test_protein_normalization_filters_by_class(self, multi_class_df, simple_experiment_2x3, protein_concentrations):
        """Test protein normalization with class filtering."""
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE'],
            protein_concentrations=protein_concentrations
        )
        result = NormalizationService.normalize(multi_class_df, config, simple_experiment_2x3)
        classes = result.normalized_df['ClassKey'].unique()
        assert 'TG' not in classes


class TestProteinNormalizationColumnRenaming:
    """Tests for column renaming in protein normalization."""

    def test_protein_normalization_renames_columns(self, basic_lipid_df, simple_experiment_2x3, protein_config):
        """Test that protein normalization renames intensity to concentration."""
        result = NormalizationService.normalize(basic_lipid_df, protein_config, simple_experiment_2x3)
        assert 'concentration[s1]' in result.normalized_df.columns
        assert 'intensity[s1]' not in result.normalized_df.columns

    def test_protein_normalization_empty_removed_standards(self, basic_lipid_df, simple_experiment_2x3, protein_config):
        """Test that protein normalization has empty removed_standards list."""
        result = NormalizationService.normalize(basic_lipid_df, protein_config, simple_experiment_2x3)
        assert result.removed_standards == []


# =============================================================================
# Combined Normalization Tests
# =============================================================================

class TestCombinedNormalizationValidation:
    """Tests for combined normalization validation."""

    def test_both_requires_intsta_df(self, basic_lipid_df, simple_experiment_2x3, both_config):
        """Test that 'both' method requires intsta_df."""
        with pytest.raises(ValueError, match="Internal standards DataFrame is required"):
            NormalizationService.normalize(basic_lipid_df, both_config, simple_experiment_2x3, intsta_df=None)

    def test_both_requires_protein_concentrations(self):
        """Test that 'both' method requires protein_concentrations (Pydantic validation)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="protein_concentrations"):
            NormalizationConfig(
                method='both',
                selected_classes=['PC'],
                internal_standards={'PC': 'PC(15:0_15:0)_IS'},
                intsta_concentrations={'PC(15:0_15:0)_IS': 10.0},
                # Missing protein_concentrations
            )


class TestCombinedNormalizationFormula:
    """Tests for combined normalization formula."""

    def test_both_applies_is_then_protein(self, multi_class_df, simple_experiment_2x3, both_config, intsta_df):
        """Test that 'both' method applies IS first, then protein."""
        result = NormalizationService.normalize(
            multi_class_df, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        # PC(16:0_18:1) s1:
        # IS: (1000 / 500) * 10 = 20
        # Protein: 20 / 1.5 = 13.33
        expected = (1000 / 500 * 10) / 1.5
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        assert abs(pc_row['concentration[s1]'] - expected) < 0.01

    def test_both_different_samples(self, multi_class_df, simple_experiment_2x3, both_config, intsta_df):
        """Test combined normalization across different samples."""
        result = NormalizationService.normalize(
            multi_class_df, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        # s6: (1250 / 550) * 10 / 1.8 = 12.63
        expected_s6 = (1250 / 550 * 10) / 1.8
        assert abs(pc_row['concentration[s6]'] - expected_s6) < 0.01


class TestCombinedNormalizationStandards:
    """Tests for standard handling in combined normalization."""

    def test_both_removes_standards(self, df_with_standards, simple_experiment_2x3, both_config, intsta_df):
        """Test that 'both' method removes standards."""
        result = NormalizationService.normalize(
            df_with_standards, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert 'PC(15:0_15:0)_IS' not in result.normalized_df['LipidMolec'].values
        assert 'PE(17:0_17:0)_IS' not in result.normalized_df['LipidMolec'].values

    def test_both_reports_removed_standards(self, df_with_standards, simple_experiment_2x3, both_config, intsta_df):
        """Test that 'both' method reports removed standards."""
        result = NormalizationService.normalize(
            df_with_standards, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert 'PC(15:0_15:0)_IS' in result.removed_standards
        assert 'PE(17:0_17:0)_IS' in result.removed_standards


class TestCombinedNormalizationClassFiltering:
    """Tests for class filtering in combined normalization."""

    def test_both_filters_by_class(self, multi_class_df, simple_experiment_2x3, both_config, intsta_df):
        """Test that 'both' method filters by selected classes."""
        result = NormalizationService.normalize(
            multi_class_df, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        classes = result.normalized_df['ClassKey'].unique()
        assert 'TG' not in classes

    def test_both_correct_method_applied(self, multi_class_df, simple_experiment_2x3, both_config, intsta_df):
        """Test that 'both' method reports correct method_applied."""
        result = NormalizationService.normalize(
            multi_class_df, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert 'Combined' in result.method_applied or 'both' in result.method_applied.lower()


# =============================================================================
# Column Renaming Helper Tests
# =============================================================================

class TestRenameIntensityToConcentration:
    """Tests for _rename_intensity_to_concentration helper."""

    def test_rename_basic(self):
        """Test basic column renaming."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [200.0],
        })
        result = NormalizationService._rename_intensity_to_concentration(df)
        assert 'concentration[s1]' in result.columns
        assert 'concentration[s2]' in result.columns

    def test_rename_removes_intensity(self):
        """Test that intensity columns are removed after rename."""
        df = pd.DataFrame({
            'intensity[s1]': [100.0],
        })
        result = NormalizationService._rename_intensity_to_concentration(df)
        assert 'intensity[s1]' not in result.columns

    def test_rename_preserves_non_intensity(self):
        """Test that non-intensity columns are preserved."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
            'OtherCol': ['value'],
            'intensity[s1]': [100.0],
        })
        result = NormalizationService._rename_intensity_to_concentration(df)
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert 'OtherCol' in result.columns

    def test_rename_empty_dataframe(self):
        """Test rename handles empty DataFrame."""
        df = pd.DataFrame(columns=['LipidMolec', 'intensity[s1]'])
        result = NormalizationService._rename_intensity_to_concentration(df)
        assert 'concentration[s1]' in result.columns

    def test_rename_preserves_values(self):
        """Test that values are preserved during rename."""
        df = pd.DataFrame({
            'intensity[s1]': [100.0, 200.0],
        })
        result = NormalizationService._rename_intensity_to_concentration(df)
        assert list(result['concentration[s1]']) == [100.0, 200.0]


# =============================================================================
# Helper Method Tests
# =============================================================================

class TestGetAvailableStandards:
    """Tests for get_available_standards method."""

    def test_returns_standard_names(self, intsta_df):
        """Test that get_available_standards returns standard names."""
        standards = NormalizationService.get_available_standards(intsta_df)
        assert 'PC(15:0_15:0)_IS' in standards
        assert 'PE(17:0_17:0)_IS' in standards
        assert 'TG(15:0_15:0_15:0)_IS' in standards

    def test_empty_intsta_df_returns_empty(self):
        """Test that empty intsta_df returns empty list."""
        assert NormalizationService.get_available_standards(pd.DataFrame()) == []

    def test_none_intsta_df_returns_empty(self):
        """Test that None intsta_df returns empty list."""
        assert NormalizationService.get_available_standards(None) == []

    def test_missing_lipidmolec_returns_empty(self):
        """Test that missing LipidMolec column returns empty list."""
        df = pd.DataFrame({'OtherCol': ['value']})
        assert NormalizationService.get_available_standards(df) == []

    def test_returns_unique_standards(self):
        """Test that duplicate standards are returned as unique."""
        df = pd.DataFrame({
            'LipidMolec': ['IS1', 'IS1', 'IS2'],
        })
        standards = NormalizationService.get_available_standards(df)
        assert len(standards) == 2


class TestGetStandardsByClass:
    """Tests for get_standards_by_class method."""

    def test_returns_grouped_standards(self, intsta_df):
        """Test that get_standards_by_class groups standards by class."""
        result = NormalizationService.get_standards_by_class(intsta_df)
        assert 'PC' in result
        assert 'PE' in result
        assert 'TG' in result
        assert 'PC(15:0_15:0)_IS' in result['PC']

    def test_empty_intsta_df_returns_empty(self):
        """Test that empty intsta_df returns empty dict."""
        assert NormalizationService.get_standards_by_class(pd.DataFrame()) == {}

    def test_none_intsta_df_returns_empty(self):
        """Test that None intsta_df returns empty dict."""
        assert NormalizationService.get_standards_by_class(None) == {}

    def test_missing_classkey_returns_empty(self):
        """Test that missing ClassKey column returns empty dict."""
        df = pd.DataFrame({'LipidMolec': ['Standard1']})
        assert NormalizationService.get_standards_by_class(df) == {}

    def test_missing_lipidmolec_returns_empty(self):
        """Test that missing LipidMolec column returns empty dict."""
        df = pd.DataFrame({'ClassKey': ['PC']})
        assert NormalizationService.get_standards_by_class(df) == {}

    def test_multiple_standards_per_class(self):
        """Test handling multiple standards per class."""
        df = pd.DataFrame({
            'LipidMolec': ['PC_IS1', 'PC_IS2', 'PE_IS1'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        result = NormalizationService.get_standards_by_class(df)
        assert len(result['PC']) == 2


# =============================================================================
# Validation Setup Tests
# =============================================================================

class TestValidateNormalizationSetupBasic:
    """Basic tests for validate_normalization_setup method."""

    def test_valid_none_config_no_errors(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that valid 'none' config returns no errors."""
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, none_config, simple_experiment_2x3
        )
        assert errors == []

    def test_empty_df_returns_error(self, simple_experiment_2x3, none_config):
        """Test that empty DataFrame returns error."""
        errors = NormalizationService.validate_normalization_setup(
            pd.DataFrame(), none_config, simple_experiment_2x3
        )
        assert len(errors) > 0

    def test_none_df_returns_error(self, simple_experiment_2x3, none_config):
        """Test that None DataFrame returns error."""
        errors = NormalizationService.validate_normalization_setup(
            None, none_config, simple_experiment_2x3
        )
        assert len(errors) > 0


class TestValidateNormalizationSetupColumns:
    """Tests for column validation in validate_normalization_setup."""

    def test_missing_classkey_returns_error(self, simple_experiment_2x3, none_config):
        """Test that missing ClassKey returns error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'intensity[s1]': [100.0],
        })
        errors = NormalizationService.validate_normalization_setup(
            df, none_config, simple_experiment_2x3
        )
        assert any('ClassKey' in e for e in errors)

    def test_missing_lipidmolec_returns_error(self, simple_experiment_2x3, none_config):
        """Test that missing LipidMolec returns error."""
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        errors = NormalizationService.validate_normalization_setup(
            df, none_config, simple_experiment_2x3
        )
        assert any('LipidMolec' in e for e in errors)

    def test_missing_intensity_returns_error(self, simple_experiment_2x3, none_config):
        """Test that missing intensity columns returns error."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
        })
        errors = NormalizationService.validate_normalization_setup(
            df, none_config, simple_experiment_2x3
        )
        assert any('intensity' in e.lower() for e in errors)


class TestValidateNormalizationSetupIS:
    """Tests for IS validation in validate_normalization_setup."""

    def test_missing_intsta_df_for_is_returns_error(self, basic_lipid_df, simple_experiment_2x3, internal_standard_config):
        """Test that missing intsta_df for IS method returns error."""
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, internal_standard_config, simple_experiment_2x3, intsta_df=None
        )
        assert any('internal standards' in e.lower() for e in errors)

    def test_missing_standard_in_intsta_df_returns_error(self, basic_lipid_df, simple_experiment_2x3, intsta_df):
        """Test that missing mapped standard returns error."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'NonExistent'},
            intsta_concentrations={'NonExistent': 10.0}
        )
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert any('not found' in e.lower() for e in errors)

    def test_valid_is_config_no_errors(self, basic_lipid_df, simple_experiment_2x3, internal_standard_config, intsta_df):
        """Test that valid IS config returns no errors."""
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, internal_standard_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert errors == []


class TestValidateNormalizationSetupProtein:
    """Tests for protein validation in validate_normalization_setup."""

    def test_missing_protein_concentrations_pydantic_error(self):
        """Test that missing protein concentrations raises Pydantic error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="protein_concentrations"):
            NormalizationConfig(method='protein')

    def test_missing_sample_protein_returns_error(self, basic_lipid_df, simple_experiment_2x3):
        """Test that missing sample protein concentration returns error."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 1.0, 's2': 1.0}  # Missing s3-s6
        )
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, config, simple_experiment_2x3
        )
        assert any('missing' in e.lower() for e in errors)

    def test_valid_protein_config_no_errors(self, basic_lipid_df, simple_experiment_2x3, protein_config):
        """Test that valid protein config returns no errors."""
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, protein_config, simple_experiment_2x3
        )
        assert errors == []


class TestValidateNormalizationSetupBoth:
    """Tests for 'both' validation in validate_normalization_setup."""

    def test_valid_both_config_no_errors(self, basic_lipid_df, simple_experiment_2x3, both_config, intsta_df):
        """Test that valid 'both' config returns no errors."""
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, both_config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert errors == []


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCasesSingleLipid:
    """Tests for single lipid edge cases."""

    def test_single_lipid_dataframe(self, simple_experiment_2x3, none_config):
        """Test normalization with single lipid."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
            'intensity[s5]': [1400.0],
            'intensity[s6]': [1500.0],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert len(result.normalized_df) == 1

    def test_single_lipid_is_normalization(self, simple_experiment_2x3, intsta_df):
        """Test IS normalization with single lipid."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
            'intensity[s5]': [1400.0],
            'intensity[s6]': [1500.0],
        })
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(15:0_15:0)_IS'},
            intsta_concentrations={'PC(15:0_15:0)_IS': 10.0}
        )
        result = NormalizationService.normalize(df, config, simple_experiment_2x3, intsta_df=intsta_df)
        assert len(result.normalized_df) == 1


class TestEdgeCasesSingleSample:
    """Tests for single sample edge cases."""

    def test_single_sample_per_condition(self, single_sample_experiment, none_config):
        """Test normalization with single sample per condition."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
        })
        result = NormalizationService.normalize(df, none_config, single_sample_experiment)
        assert 'concentration[s1]' in result.normalized_df.columns
        assert 'concentration[s2]' in result.normalized_df.columns


class TestEdgeCasesExtremeValues:
    """Tests for extreme value edge cases."""

    def test_very_large_values(self, simple_experiment_2x3, none_config):
        """Test normalization with very large intensity values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e15],
            'intensity[s2]': [1e15],
            'intensity[s3]': [1e15],
            'intensity[s4]': [1e15],
            'intensity[s5]': [1e15],
            'intensity[s6]': [1e15],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 1e15

    def test_very_small_values(self, simple_experiment_2x3, none_config):
        """Test normalization with very small intensity values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e-15],
            'intensity[s2]': [1e-15],
            'intensity[s3]': [1e-15],
            'intensity[s4]': [1e-15],
            'intensity[s5]': [1e-15],
            'intensity[s6]': [1e-15],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 1e-15

    def test_mixed_zero_and_nonzero(self, simple_experiment_2x3, none_config):
        """Test normalization with mixed zero and non-zero values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [1000.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [2000.0],
            'intensity[s5]': [0.0],
            'intensity[s6]': [3000.0],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 0.0
        assert result.normalized_df.loc[0, 'concentration[s2]'] == 1000.0

    def test_all_zeros_in_intensity(self, simple_experiment_2x3, none_config):
        """Test normalization with all zero intensity values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0],
            'intensity[s5]': [0.0],
            'intensity[s6]': [0.0],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert all(result.normalized_df.loc[0, f'concentration[s{i}]'] == 0.0 for i in range(1, 7))


class TestEdgeCasesUnknownMethod:
    """Tests for unknown method edge cases."""

    def test_unknown_method_raises_error(self, basic_lipid_df, simple_experiment_2x3):
        """Test that unknown normalization method raises error."""
        config = NormalizationConfig(method='none')
        object.__setattr__(config, 'method', 'unknown_method')
        with pytest.raises(ValueError, match="Unknown normalization method"):
            NormalizationService.normalize(basic_lipid_df, config, simple_experiment_2x3)


class TestEdgeCasesEmptyClasses:
    """Tests for empty class selection edge cases."""

    def test_empty_selected_classes_is(self, multi_class_df, simple_experiment_2x3, intsta_df):
        """Test IS normalization with empty selected_classes uses all."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=[],
            internal_standards={
                'PC': 'PC(15:0_15:0)_IS',
                'PE': 'PE(17:0_17:0)_IS',
                'TG': 'TG(15:0_15:0_15:0)_IS',
            },
            intsta_concentrations={
                'PC(15:0_15:0)_IS': 10.0,
                'PE(17:0_17:0)_IS': 5.0,
                'TG(15:0_15:0_15:0)_IS': 8.0,
            }
        )
        result = NormalizationService.normalize(multi_class_df, config, simple_experiment_2x3, intsta_df=intsta_df)
        classes = result.normalized_df['ClassKey'].unique()
        assert len(classes) == 3


# =============================================================================
# Type Handling Tests
# =============================================================================

class TestTypeHandlingInteger:
    """Tests for integer type handling."""

    def test_integer_intensity_values(self, simple_experiment_2x3, none_config):
        """Test normalization with integer intensity values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000],
            'intensity[s2]': [1100],
            'intensity[s3]': [1200],
            'intensity[s4]': [1300],
            'intensity[s5]': [1400],
            'intensity[s6]': [1500],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 1000

    def test_integer_protein_concentrations(self, basic_lipid_df, simple_experiment_2x3):
        """Test normalization with integer protein concentrations."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={
                's1': 2, 's2': 2, 's3': 2,
                's4': 2, 's5': 2, 's6': 2,
            }
        )
        result = NormalizationService.normalize(basic_lipid_df, config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'concentration[s1]'] == 500.0


class TestTypeHandlingStrings:
    """Tests for string column handling."""

    def test_string_column_preservation(self, simple_experiment_2x3, none_config):
        """Test that string columns are preserved correctly."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0],
            'intensity[s5]': [1400.0],
            'intensity[s6]': [1500.0],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert result.normalized_df.loc[0, 'LipidMolec'] == 'PC(16:0_18:1)'
        assert result.normalized_df.loc[0, 'ClassKey'] == 'PC'

    def test_special_characters_in_lipid_names(self, simple_experiment_2x3, none_config):
        """Test handling of special characters in lipid names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0/18:1(9Z))', 'PE(P-18:0/20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2100.0],
            'intensity[s3]': [1200.0, 2200.0],
            'intensity[s4]': [1300.0, 2300.0],
            'intensity[s5]': [1400.0, 2400.0],
            'intensity[s6]': [1500.0, 2500.0],
        })
        result = NormalizationService.normalize(df, none_config, simple_experiment_2x3)
        assert 'PC(16:0/18:1(9Z))' in result.normalized_df['LipidMolec'].values


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegrationInternalStandards:
    """Integration tests for internal standards workflow."""

    def test_full_workflow_is(self, simple_experiment_2x3, intsta_df):
        """Test complete workflow with internal standards normalization."""
        df = pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)', 'PC(18:0_18:1)', 'PC(15:0_15:0)_IS',
                'PE(18:0_20:4)', 'PE(17:0_17:0)_IS',
            ],
            'ClassKey': ['PC', 'PC', 'PC', 'PE', 'PE'],
            'intensity[s1]': [1000.0, 1100.0, 500.0, 2000.0, 600.0],
            'intensity[s2]': [1050.0, 1150.0, 510.0, 2050.0, 610.0],
            'intensity[s3]': [1100.0, 1200.0, 520.0, 2100.0, 620.0],
            'intensity[s4]': [1150.0, 1250.0, 530.0, 2150.0, 630.0],
            'intensity[s5]': [1200.0, 1300.0, 540.0, 2200.0, 640.0],
            'intensity[s6]': [1250.0, 1350.0, 550.0, 2250.0, 650.0],
        })

        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(15:0_15:0)_IS',
                'PE': 'PE(17:0_17:0)_IS',
            },
            intsta_concentrations={
                'PC(15:0_15:0)_IS': 10.0,
                'PE(17:0_17:0)_IS': 5.0,
            }
        )

        errors = NormalizationService.validate_normalization_setup(
            df, config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert errors == []

        result = NormalizationService.normalize(df, config, simple_experiment_2x3, intsta_df=intsta_df)

        assert len(result.normalized_df) == 3
        assert 'PC(15:0_15:0)_IS' not in result.normalized_df['LipidMolec'].values
        assert len(result.removed_standards) == 2


class TestIntegrationProtein:
    """Integration tests for protein normalization workflow."""

    def test_full_workflow_protein(self, basic_lipid_df, simple_experiment_2x3, protein_config):
        """Test complete workflow with protein normalization."""
        errors = NormalizationService.validate_normalization_setup(
            basic_lipid_df, protein_config, simple_experiment_2x3
        )
        assert errors == []

        result = NormalizationService.normalize(basic_lipid_df, protein_config, simple_experiment_2x3)

        for col in result.normalized_df.columns:
            if col.startswith('concentration['):
                orig_col = col.replace('concentration', 'intensity')
                for i in range(len(result.normalized_df)):
                    assert result.normalized_df.loc[i, col] < basic_lipid_df.loc[i, orig_col]


class TestIntegrationCombined:
    """Integration tests for combined normalization workflow."""

    def test_full_workflow_combined(self, simple_experiment_2x3, intsta_df, protein_concentrations):
        """Test complete workflow with combined normalization."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1050.0, 2050.0],
            'intensity[s3]': [1100.0, 2100.0],
            'intensity[s4]': [1150.0, 2150.0],
            'intensity[s5]': [1200.0, 2200.0],
            'intensity[s6]': [1250.0, 2250.0],
        })

        config = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(15:0_15:0)_IS',
                'PE': 'PE(17:0_17:0)_IS',
            },
            intsta_concentrations={
                'PC(15:0_15:0)_IS': 10.0,
                'PE(17:0_17:0)_IS': 5.0,
            },
            protein_concentrations=protein_concentrations
        )

        errors = NormalizationService.validate_normalization_setup(
            df, config, simple_experiment_2x3, intsta_df=intsta_df
        )
        assert errors == []

        result = NormalizationService.normalize(df, config, simple_experiment_2x3, intsta_df=intsta_df)

        # Verify combined formula: ((1000 / 500) * 10) / 1.5 = 13.33
        pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        expected = ((1000 / 500) * 10) / 1.5
        assert abs(pc_row['concentration[s1]'] - expected) < 0.01


class TestIntegrationPreserveStructure:
    """Integration tests for structure preservation."""

    def test_preserve_dataframe_structure(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that DataFrame structure is preserved after normalization."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        assert len(result.normalized_df) == len(basic_lipid_df)
        assert 'LipidMolec' in result.normalized_df.columns
        assert 'ClassKey' in result.normalized_df.columns

    def test_preserve_lipid_order(self, basic_lipid_df, simple_experiment_2x3, none_config):
        """Test that lipid order is preserved."""
        result = NormalizationService.normalize(basic_lipid_df, none_config, simple_experiment_2x3)
        original_lipids = basic_lipid_df['LipidMolec'].tolist()
        result_lipids = result.normalized_df['LipidMolec'].tolist()
        assert original_lipids == result_lipids
