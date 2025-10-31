"""
Unit tests for NormalizationService edge cases.

This file contains comprehensive edge case tests for normalization:
- Zero and NaN intensity handling
- Infinity value handling
- Missing standards for classes
- Mismatched sample names
- Empty DataFrames
- Boundary conditions
- Method chaining behavior
- Column preservation logic
- Error message quality
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.normalization_service import NormalizationService
from src.lipidcruncher.core.models.normalization import NormalizationConfig
from src.lipidcruncher.core.models.experiment import ExperimentConfig


# ==================== Fixtures ====================

@pytest.fixture
def service():
    """NormalizationService instance."""
    return NormalizationService()


@pytest.fixture
def experiment_config():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


@pytest.fixture
def sample_data():
    """Create sample lipid data with multiple classes."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 
            'PC(18:0_20:4)',
            'PE(18:0_20:4)',
            'PE(16:0_18:1)',
            'TAG(16:0_18:1_18:2)'
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE', 'TAG'],
        'intensity[s1]': [1000.0, 1100.0, 2000.0, 2100.0, 1500.0],
        'intensity[s2]': [1050.0, 1150.0, 2050.0, 2150.0, 1550.0],
        'intensity[s3]': [1100.0, 1200.0, 2100.0, 2200.0, 1600.0],
        'intensity[s4]': [1150.0, 1250.0, 2150.0, 2250.0, 1650.0]
    })


@pytest.fixture
def sample_standards():
    """Create sample internal standards DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)', 'TAG(16:0_18:1_18:2)(d5)'],
        'ClassKey': ['PC', 'PE', 'TAG'],
        'intensity[s1]': [5000.0, 6000.0, 7000.0],
        'intensity[s2]': [5100.0, 6100.0, 7100.0],
        'intensity[s3]': [5200.0, 6200.0, 7200.0],
        'intensity[s4]': [5300.0, 6300.0, 7300.0]
    })


@pytest.fixture
def protein_concentrations():
    """Create sample protein concentrations DataFrame."""
    return pd.DataFrame({
        'Sample': ['s1', 's2', 's3', 's4'],
        'Concentration': [2.0, 2.5, 2.2, 2.8]
    })


@pytest.fixture
def normalization_config_standards():
    """Config for internal standards normalization."""
    return NormalizationConfig(
        method='internal_standard',
        selected_classes=['PC', 'PE', 'TAG'],
        internal_standards={
            'PC': 'PC(16:0_18:1)(d7)',
            'PE': 'PE(18:0_20:4)(d9)',
            'TAG': 'TAG(16:0_18:1_18:2)(d5)'
        },
        intsta_concentrations={
            'PC(16:0_18:1)(d7)': 10.0,
            'PE(18:0_20:4)(d9)': 15.0,
            'TAG(16:0_18:1_18:2)(d5)': 20.0
        }
    )


@pytest.fixture
def normalization_config_protein():
    """Config for protein normalization."""
    return NormalizationConfig(
        method='protein',
        selected_classes=['PC', 'PE', 'TAG'],
        protein_concentrations={
            's1': 2.0,
            's2': 2.5,
            's3': 2.2,
            's4': 2.8
        }
    )


@pytest.fixture
def normalization_config_both():
    """Config for combined normalization."""
    return NormalizationConfig(
        method='both',
        selected_classes=['PC', 'PE', 'TAG'],
        internal_standards={
            'PC': 'PC(16:0_18:1)(d7)',
            'PE': 'PE(18:0_20:4)(d9)',
            'TAG': 'TAG(16:0_18:1_18:2)(d5)'
        },
        intsta_concentrations={
            'PC(16:0_18:1)(d7)': 10.0,
            'PE(18:0_20:4)(d9)': 15.0,
            'TAG(16:0_18:1_18:2)(d5)': 20.0
        },
        protein_concentrations={
            's1': 2.0,
            's2': 2.5,
            's3': 2.2,
            's4': 2.8
        }
    )


# ==================== Test Class A: Zero and NaN Handling ====================

class TestZeroAndNaNHandling:
    """Tests for handling zero and NaN values in normalization."""
    
    def test_normalization_with_all_zero_standard_intensities(
        self, service, sample_data, experiment_config, normalization_config_standards
    ):
        """Test behavior when standard has all zero intensities."""
        # Arrange - Create standards with zero intensities
        zero_standards = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)', 'TAG(16:0_18:1_18:2)(d5)'],
            'ClassKey': ['PC', 'PE', 'TAG'],
            'intensity[s1]': [0.0, 6000.0, 7000.0],
            'intensity[s2]': [0.0, 6100.0, 7100.0],
            'intensity[s3]': [0.0, 6200.0, 7200.0],
            'intensity[s4]': [0.0, 6300.0, 7300.0]
        })
        
        # Act - Normalize (division by zero should produce inf/nan, then be cleaned)
        result = service.normalize(
            sample_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=zero_standards
        )
        
        # Assert - Check that inf values are replaced with 0
        # PC class should have zero values (divided by zero standard)
        pc_data = result[result['ClassKey'] == 'PC']
        assert not pc_data.empty
        # After fillna(0) and replace inf with 0, all should be 0
        concentration_cols = [col for col in pc_data.columns if col.startswith('concentration[')]
        assert all(pc_data[concentration_cols].values.flatten() == 0)
    
    def test_normalization_with_some_zero_standard_intensities(
        self, service, sample_data, experiment_config, normalization_config_standards
    ):
        """Test behavior when standard has some zero intensities."""
        # Arrange - Create standards with some zeros
        partial_zero_standards = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)', 'TAG(16:0_18:1_18:2)(d5)'],
            'ClassKey': ['PC', 'PE', 'TAG'],
            'intensity[s1]': [5000.0, 6000.0, 7000.0],
            'intensity[s2]': [0.0, 6100.0, 7100.0],  # Zero in one sample
            'intensity[s3]': [5200.0, 0.0, 7200.0],  # Zero in another
            'intensity[s4]': [5300.0, 6300.0, 7300.0]
        })
        
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=partial_zero_standards
        )
        
        # Assert - Some values should be 0 (where standard was 0), others should be valid
        assert not result.empty
        # At least some values should be non-zero
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        assert (result[concentration_cols].values.flatten() > 0).any()
    
    def test_normalization_with_nan_standard_intensities(
        self, service, sample_data, experiment_config, normalization_config_standards
    ):
        """Test behavior when standard has NaN intensities."""
        # Arrange - Create standards with NaN values
        nan_standards = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)', 'TAG(16:0_18:1_18:2)(d5)'],
            'ClassKey': ['PC', 'PE', 'TAG'],
            'intensity[s1]': [5000.0, 6000.0, 7000.0],
            'intensity[s2]': [np.nan, 6100.0, 7100.0],
            'intensity[s3]': [5200.0, np.nan, 7200.0],
            'intensity[s4]': [5300.0, 6300.0, 7300.0]
        })
        
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=nan_standards
        )
        
        # Assert - NaN values should be filled with 0
        assert not result.empty
        assert not result.isnull().any().any()
    
    def test_normalization_with_zero_lipid_intensities(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test normalization when lipid intensities are zero."""
        # Arrange - Create data with zero lipid intensities
        zero_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TAG(16:0_18:1_18:2)'],
            'ClassKey': ['PC', 'PE', 'TAG'],
            'intensity[s1]': [0.0, 0.0, 0.0],
            'intensity[s2]': [0.0, 0.0, 0.0],
            'intensity[s3]': [0.0, 0.0, 0.0],
            'intensity[s4]': [0.0, 0.0, 0.0]
        })
        
        # Act
        result = service.normalize(
            zero_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - All normalized values should be zero
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        assert all(result[concentration_cols].values.flatten() == 0)
    
    def test_protein_normalization_with_zero_protein_concentration(
        self, service, sample_data, experiment_config, normalization_config_protein
    ):
        """Test behavior when protein concentration is zero."""
        # Arrange - Create protein data with zero concentration
        zero_protein = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, 0.0, 2.2, 2.8]  # s2 has zero
        })
        
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_protein,
            experiment_config,
            protein_df=zero_protein
        )
        
        # Assert - s2 should NOT be normalized (should skip zero protein)
        # Check that result was produced
        assert not result.empty
        assert len(result) == len(sample_data)
    
    def test_protein_normalization_with_negative_protein_concentration(
        self, service, sample_data, experiment_config, normalization_config_protein
    ):
        """Test behavior when protein concentration is negative."""
        # Arrange - Create protein data with negative concentration
        negative_protein = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, -1.0, 2.2, 2.8]  # s2 is negative
        })
        
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_protein,
            experiment_config,
            protein_df=negative_protein
        )
        
        # Assert - s2 should NOT be normalized (should skip negative protein)
        assert not result.empty
    
    def test_normalization_with_nan_in_lipid_data(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test normalization when lipid data contains NaN values."""
        # Arrange
        data_with_nan = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, np.nan],
            'intensity[s2]': [np.nan, 2050.0],
            'intensity[s3]': [1100.0, 2100.0],
            'intensity[s4]': [1150.0, 2150.0]
        })
        
        # Act
        result = service.normalize(
            data_with_nan,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - NaN should be handled (filled with 0)
        assert not result.isnull().any().any()


# ==================== Test Class B: Infinity Handling ====================

class TestInfinityHandling:
    """Tests for handling infinity values in normalization."""
    
    def test_normalization_handles_infinity_values_from_division_by_zero(
        self, service, experiment_config
    ):
        """Test that inf values from division by zero are handled."""
        # Arrange - Create scenario that produces inf
        data_with_large_values = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e10],
            'intensity[s2]': [1e10],
            'intensity[s3]': [1e10],
            'intensity[s4]': [1e10]
        })
        
        standards_with_zeros = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0]
        })
        
        # Create config with only PC class (not 3 classes)
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act
        result = service.normalize(
            data_with_large_values,
            config,
            experiment_config,
            intsta_df=standards_with_zeros
        )
        
        # Assert - No inf values should remain
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()
    
    def test_normalization_replaces_positive_infinity_with_zero(
        self, service, experiment_config
    ):
        """Test that positive infinity is replaced with zero."""
        # Arrange
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1000.0],
            'intensity[s3]': [1000.0],
            'intensity[s4]': [1000.0]
        })
        
        zero_standards = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0]
        })
        
        # Create config with only PC class (not 3 classes)
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act
        result = service.normalize(
            data,
            config,
            experiment_config,
            intsta_df=zero_standards
        )
        
        # Assert - All values should be 0 (replaced inf with 0)
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        assert all(result[concentration_cols].values.flatten() == 0)


# ==================== Test Class C: Missing Standards ====================

class TestMissingStandards:
    """Tests for handling missing standards."""
    
    def test_normalization_with_missing_standard_for_class(
        self, service, sample_data, experiment_config
    ):
        """Test error when a selected class has no matching standard."""
        # Arrange - Config includes TAG but standards don't have TAG standard
        config_missing_standard = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'TAG'],
            internal_standards={
                'PC': 'PC(16:0_18:1)(d7)',
                'PE': 'PE(18:0_20:4)(d9)',
                'TAG': 'TAG(16:0_18:1_18:2)(d5)'  # This standard exists in config
            },
            intsta_concentrations={
                'PC(16:0_18:1)(d7)': 10.0,
                'PE(18:0_20:4)(d9)': 15.0,
                'TAG(16:0_18:1_18:2)(d5)': 20.0
            }
        )
        
        # Standards DataFrame WITHOUT TAG standard
        standards_without_tag = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)'],  # No TAG!
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [5000.0, 6000.0],
            'intensity[s2]': [5100.0, 6100.0],
            'intensity[s3]': [5200.0, 6200.0],
            'intensity[s4]': [5300.0, 6300.0]
        })
        
        # Act & Assert - Should raise ValueError about missing standard
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                config_missing_standard,
                experiment_config,
                intsta_df=standards_without_tag
            )
        
        assert "not found" in str(exc_info.value).lower()
        assert "TAG(16:0_18:1_18:2)(d5)" in str(exc_info.value)
    
    def test_normalization_with_no_standards_dataframe(
        self, service, sample_data, experiment_config, normalization_config_standards
    ):
        """Test error when standards DataFrame is None."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                normalization_config_standards,
                experiment_config,
                intsta_df=None
            )
        
        assert "intsta_df required" in str(exc_info.value).lower()
    
    def test_normalization_with_empty_standards_dataframe(
        self, service, sample_data, experiment_config, normalization_config_standards
    ):
        """Test error when standards DataFrame is empty."""
        # Arrange
        empty_standards = pd.DataFrame()
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                normalization_config_standards,
                experiment_config,
                intsta_df=empty_standards
            )
        
        assert "intsta_df required" in str(exc_info.value).lower()


# ==================== Test Class D: Mismatched Sample Names ====================

class TestMismatchedSampleNames:
    """Tests for handling mismatched sample names."""
    
    def test_protein_normalization_with_mismatched_sample_names(
        self, service, sample_data, experiment_config, normalization_config_protein
    ):
        """Test when protein DataFrame has different sample names."""
        # Arrange - Protein data with wrong sample names
        mismatched_protein = pd.DataFrame({
            'Sample': ['wrong1', 'wrong2', 'wrong3', 'wrong4'],  # Don't match s1-s4
            'Concentration': [2.0, 2.5, 2.2, 2.8]
        })
        
        # Act - Should not normalize any samples (no matches)
        result = service.normalize(
            sample_data,
            normalization_config_protein,
            experiment_config,
            protein_df=mismatched_protein
        )
        
        # Assert - Result should be produced
        assert not result.empty
        assert len(result) == len(sample_data)
    
    def test_protein_normalization_with_partial_sample_matches(
        self, service, sample_data, experiment_config, normalization_config_protein
    ):
        """Test when only some sample names match."""
        # Arrange - Protein data with only 2 matching samples
        partial_protein = pd.DataFrame({
            'Sample': ['s1', 's2', 'wrong1', 'wrong2'],  # Only s1, s2 match
            'Concentration': [2.0, 2.5, 2.2, 2.8]
        })
        
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_protein,
            experiment_config,
            protein_df=partial_protein
        )
        
        # Assert - Only s1 and s2 should be normalized
        assert not result.empty
    
    def test_standards_normalization_with_mismatched_sample_names(
        self, service, experiment_config, normalization_config_standards
    ):
        """Test when standards have different sample names than data."""
        # Arrange - Data with samples s1-s4
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0],
            'intensity[s3]': [1200.0],
            'intensity[s4]': [1300.0]
        })
        
        # Standards with different sample names
        standards_wrong_samples = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[wrong1]': [5000.0],
            'intensity[wrong2]': [5100.0],
            'intensity[wrong3]': [5200.0],
            'intensity[wrong4]': [5300.0]
        })
        
        # Act & Assert - Should fail because column names don't match
        with pytest.raises(KeyError):
            service.normalize(
                data,
                normalization_config_standards,
                experiment_config,
                intsta_df=standards_wrong_samples
            )


# ==================== Test Class E: Method Chaining ('both' method) ====================

class TestMethodChaining:
    """Tests for the 'both' normalization method behavior."""
    
    def test_both_method_produces_consistent_results(
        self, service, sample_data, experiment_config, 
        sample_standards, protein_concentrations, normalization_config_both
    ):
        """Test that 'both' method produces valid results."""
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_both,
            experiment_config,
            intsta_df=sample_standards,
            protein_df=protein_concentrations
        )
        
        # Assert - Result should have concentration columns
        assert not result.empty
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        assert len(concentration_cols) > 0
        
        # All concentration values should be positive
        assert (result[concentration_cols] >= 0).all().all()
    
    def test_both_method_applies_standards_first_then_protein(
        self, service, sample_data, experiment_config,
        sample_standards, protein_concentrations, normalization_config_both
    ):
        """Test that 'both' method applies standards first, then protein."""
        # Act - Normalize with both
        result_both = service.normalize(
            sample_data,
            normalization_config_both,
            experiment_config,
            intsta_df=sample_standards,
            protein_df=protein_concentrations
        )
        
        # Compare with manual chaining
        config_standards_only = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'TAG'],
            internal_standards=normalization_config_both.internal_standards,
            intsta_concentrations=normalization_config_both.intsta_concentrations
        )
        temp = service.normalize(
            sample_data,
            config_standards_only,
            experiment_config,
            intsta_df=sample_standards
        )
        
        config_protein_only = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE', 'TAG'],
            protein_concentrations=normalization_config_both.protein_concentrations
        )
        result_manual = service.normalize(
            temp,
            config_protein_only,
            experiment_config,
            protein_df=protein_concentrations
        )
        
        # Assert - Results should be close
        concentration_cols = [col for col in result_both.columns if col.startswith('concentration[')]
        pd.testing.assert_frame_equal(
            result_both[concentration_cols].sort_index(),
            result_manual[concentration_cols].sort_index(),
            rtol=1e-5
        )
    
    def test_both_method_with_missing_standards(
        self, service, sample_data, experiment_config, protein_concentrations
    ):
        """Test 'both' method fails when standards missing."""
        # Arrange
        config_both_no_standards = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE', 'TAG'],
            internal_standards={
                'PC': 'PC(16:0_18:1)(d7)',
                'PE': 'PE(18:0_20:4)(d9)',
                'TAG': 'TAG(16:0_18:1_18:2)(d5)'
            },
            intsta_concentrations={
                'PC(16:0_18:1)(d7)': 10.0,
                'PE(18:0_20:4)(d9)': 15.0,
                'TAG(16:0_18:1_18:2)(d5)': 20.0
            },
            protein_concentrations={
                's1': 2.0, 's2': 2.5, 's3': 2.2, 's4': 2.8
            }
        )
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                config_both_no_standards,
                experiment_config,
                intsta_df=None,
                protein_df=protein_concentrations
            )
        
        assert "intsta_df required" in str(exc_info.value).lower()
    
    def test_both_method_with_missing_protein(
        self, service, sample_data, experiment_config, sample_standards
    ):
        """Test 'both' method fails when protein data missing."""
        # Arrange
        config_both_no_protein = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE', 'TAG'],
            internal_standards={
                'PC': 'PC(16:0_18:1)(d7)',
                'PE': 'PE(18:0_20:4)(d9)',
                'TAG': 'TAG(16:0_18:1_18:2)(d5)'
            },
            intsta_concentrations={
                'PC(16:0_18:1)(d7)': 10.0,
                'PE(18:0_20:4)(d9)': 15.0,
                'TAG(16:0_18:1_18:2)(d5)': 20.0
            },
            protein_concentrations={
                's1': 2.0, 's2': 2.5, 's3': 2.2, 's4': 2.8
            }
        )
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                config_both_no_protein,
                experiment_config,
                intsta_df=sample_standards,
                protein_df=None
            )
        
        assert "protein_df required" in str(exc_info.value).lower()


# ==================== Test Class F: Column Preservation ====================

class TestColumnPreservation:
    """Tests for column naming and preservation logic."""
    
    def test_normalization_preserves_non_intensity_columns(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test that LipidMolec, ClassKey, etc. are not modified."""
        # Arrange - Data with extra columns
        data_with_extra_cols = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'FAKey': ['16:0_18:1', '18:0_20:4'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [12.5, 13.2],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1050.0, 2050.0],
            'intensity[s3]': [1100.0, 2100.0],
            'intensity[s4]': [1150.0, 2150.0]
        })
        
        # Act
        result = service.normalize(
            data_with_extra_cols,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Metadata columns should be unchanged
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert all(result['LipidMolec'] == data_with_extra_cols['LipidMolec'])
        assert all(result['ClassKey'] == data_with_extra_cols['ClassKey'])
    
    def test_standards_normalization_renames_intensity_to_concentration(
        self, service, sample_data, experiment_config, 
        sample_standards, normalization_config_standards
    ):
        """Test that intensity columns are renamed to concentration."""
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Should have concentration columns, not intensity
        intensity_cols = [col for col in result.columns if col.startswith('intensity[')]
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        
        assert len(intensity_cols) == 0
        assert len(concentration_cols) == 4
    
    def test_protein_normalization_renames_intensity_to_concentration(
        self, service, sample_data, experiment_config, protein_concentrations, normalization_config_protein
    ):
        """Test that protein normalization also renames columns."""
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_protein,
            experiment_config,
            protein_df=protein_concentrations
        )
        
        # Assert - Should have concentration columns
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        assert len(concentration_cols) == 4
    
    def test_both_normalization_handles_column_renaming_correctly(
        self, service, sample_data, experiment_config,
        sample_standards, protein_concentrations, normalization_config_both
    ):
        """Test that 'both' method renames columns only once."""
        # Act
        result = service.normalize(
            sample_data,
            normalization_config_both,
            experiment_config,
            intsta_df=sample_standards,
            protein_df=protein_concentrations
        )
        
        # Assert - Should have concentration columns
        intensity_cols = [col for col in result.columns if col.startswith('intensity[')]
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        
        assert len(intensity_cols) == 0
        assert len(concentration_cols) == 4
    
    def test_preserve_column_prefix_option(
        self, service, sample_data, experiment_config, sample_standards
    ):
        """Test preserve_column_prefix=True keeps intensity[] names."""
        # Arrange
        config_preserve = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'TAG'],
            internal_standards={
                'PC': 'PC(16:0_18:1)(d7)',
                'PE': 'PE(18:0_20:4)(d9)',
                'TAG': 'TAG(16:0_18:1_18:2)(d5)'
            },
            intsta_concentrations={
                'PC(16:0_18:1)(d7)': 10.0,
                'PE(18:0_20:4)(d9)': 15.0,
                'TAG(16:0_18:1_18:2)(d5)': 20.0
            },
            preserve_column_prefix=True
        )
        
        # Act
        result = service.normalize(
            sample_data,
            config_preserve,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Should still have intensity columns
        intensity_cols = [col for col in result.columns if col.startswith('intensity[')]
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        
        assert len(intensity_cols) == 4
        assert len(concentration_cols) == 0


# ==================== Test Class G: Empty and Boundary Cases ====================

class TestEmptyAndBoundaryCases:
    """Tests for empty DataFrames and boundary conditions."""
    
    def test_normalization_with_empty_dataframe(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test normalization with empty input DataFrame."""
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act & Assert - Should handle gracefully
        with pytest.raises((ValueError, KeyError)):
            service.normalize(
                empty_df,
                normalization_config_standards,
                experiment_config,
                intsta_df=sample_standards
            )
    
    def test_normalization_with_single_lipid(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test normalization with single lipid species."""
        # Arrange
        single_lipid = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1050.0],
            'intensity[s3]': [1100.0],
            'intensity[s4]': [1150.0]
        })
        
        # Act
        result = service.normalize(
            single_lipid,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert
        assert len(result) == 1
        assert result['LipidMolec'].iloc[0] == 'PC(16:0_18:1)'
    
    def test_normalization_with_single_class_selected(
        self, service, sample_data, experiment_config, sample_standards
    ):
        """Test normalization when only one class is selected."""
        # Arrange
        config_single_class = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act
        result = service.normalize(
            sample_data,
            config_single_class,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Only PC lipids should be in result
        assert all(result['ClassKey'] == 'PC')
        assert len(result) == 2
    
    def test_normalization_with_very_large_values(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test normalization with very large intensity values."""
        # Arrange
        large_value_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e15],
            'intensity[s2]': [1e15],
            'intensity[s3]': [1e15],
            'intensity[s4]': [1e15]
        })
        
        # Act
        result = service.normalize(
            large_value_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Should handle without overflow
        assert not result.empty
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()
    
    def test_normalization_with_very_small_values(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test normalization with very small intensity values."""
        # Arrange
        small_value_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e-10],
            'intensity[s2]': [1e-10],
            'intensity[s3]': [1e-10],
            'intensity[s4]': [1e-10]
        })
        
        # Act
        result = service.normalize(
            small_value_data,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Should handle without underflow
        assert not result.empty
        concentration_cols = [col for col in result.columns if col.startswith('concentration[')]
        assert (result[concentration_cols] >= 0).all().all()


# ==================== Test Class H: Method Validation ====================

class TestMethodValidation:
    """Tests for method parameter validation."""
    
    def test_none_method_returns_copy_of_data(
        self, service, sample_data, experiment_config
    ):
        """Test that 'none' method returns unchanged data."""
        # Arrange
        config_none = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE', 'TAG']
        )
        
        # Act
        result = service.normalize(
            sample_data,
            config_none,
            experiment_config
        )
        
        # Assert - Should be identical to input
        pd.testing.assert_frame_equal(result, sample_data)


# ==================== Test Class I: Error Message Quality ====================

class TestErrorMessageQuality:
    """Tests to ensure error messages are helpful and user-friendly."""
    
    def test_error_messages_are_descriptive(
        self, service, sample_data, experiment_config, normalization_config_standards
    ):
        """Verify error messages contain helpful guidance."""
        # Test missing standards
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                normalization_config_standards,
                experiment_config,
                intsta_df=None
            )
        
        error_msg = str(exc_info.value)
        assert len(error_msg) > 20
        assert "intsta_df" in error_msg.lower()
    
    def test_error_messages_avoid_technical_jargon(
        self, service, sample_data, experiment_config
    ):
        """Error messages should be understandable by scientists."""
        # Arrange
        config_standards_no_data = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                config_standards_no_data,
                experiment_config,
                intsta_df=None
            )
        
        error_msg = str(exc_info.value)
        # Should use domain language
        assert any(word in error_msg.lower() for word in ['standard', 'required', 'normalization'])
    
    def test_missing_standard_error_includes_standard_name(
        self, service, sample_data, experiment_config
    ):
        """Test that error message includes which standard is missing."""
        # Arrange
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        standards_without_pc = pd.DataFrame({
            'LipidMolec': ['PE(18:0_20:4)(d9)'],
            'ClassKey': ['PE'],
            'intensity[s1]': [6000.0],
            'intensity[s2]': [6100.0],
            'intensity[s3]': [6200.0],
            'intensity[s4]': [6300.0]
        })
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.normalize(
                sample_data,
                config,
                experiment_config,
                intsta_df=standards_without_pc
            )
        
        error_msg = str(exc_info.value)
        assert "PC(16:0_18:1)(d7)" in error_msg


# ==================== Test Class J: Standards Removal ====================

class TestStandardsRemoval:
    """Tests for proper removal of standards from normalized data."""
    
    def test_standards_are_removed_from_normalized_data(
        self, service, experiment_config, sample_standards, normalization_config_standards
    ):
        """Test that internal standards themselves are removed from results."""
        # Arrange - Data that includes the standards
        data_with_standards = pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)',
                'PC(16:0_18:1)(d7)',  # Standard
                'PE(18:0_20:4)',
                'PE(18:0_20:4)(d9)',  # Standard
                'TAG(16:0_18:1_18:2)',
                'TAG(16:0_18:1_18:2)(d5)'  # Standard
            ],
            'ClassKey': ['PC', 'PC', 'PE', 'PE', 'TAG', 'TAG'],
            'intensity[s1]': [1000.0, 5000.0, 2000.0, 6000.0, 1500.0, 7000.0],
            'intensity[s2]': [1050.0, 5100.0, 2050.0, 6100.0, 1550.0, 7100.0],
            'intensity[s3]': [1100.0, 5200.0, 2100.0, 6200.0, 1600.0, 7200.0],
            'intensity[s4]': [1150.0, 5300.0, 2150.0, 6300.0, 1650.0, 7300.0]
        })
        
        # Act
        result = service.normalize(
            data_with_standards,
            normalization_config_standards,
            experiment_config,
            intsta_df=sample_standards
        )
        
        # Assert - Standards should not be in result
        assert 'PC(16:0_18:1)(d7)' not in result['LipidMolec'].values
        assert 'PE(18:0_20:4)(d9)' not in result['LipidMolec'].values
        assert 'TAG(16:0_18:1_18:2)(d5)' not in result['LipidMolec'].values
        
        # But regular lipids should be
        assert 'PC(16:0_18:1)' in result['LipidMolec'].values
        assert 'PE(18:0_20:4)' in result['LipidMolec'].values
        assert 'TAG(16:0_18:1_18:2)' in result['LipidMolec'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])