"""
Unit tests for NormalizationService.
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.normalization_service import NormalizationService
from src.lipidcruncher.core.models.normalization import NormalizationConfig
from src.lipidcruncher.core.models.experiment import ExperimentConfig


@pytest.fixture
def sample_lipid_data():
    """Create sample lipid data for testing."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(18:0_18:1)', 'PE(18:0_20:4)', 'PE(16:0_18:1)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [100.0, 200.0, 150.0, 120.0],
        'intensity[s2]': [110.0, 210.0, 160.0, 130.0],
        'intensity[s3]': [105.0, 205.0, 155.0, 125.0]
    })


@pytest.fixture
def sample_internal_standards():
    """Create sample internal standards data."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d7)'],
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [50.0, 60.0],
        'intensity[s2]': [55.0, 65.0],
        'intensity[s3]': [52.0, 62.0]
    })


@pytest.fixture
def sample_protein_data():
    """Create sample protein concentration data."""
    return pd.DataFrame({
        'Sample': ['s1', 's2', 's3'],
        'Concentration': [2.0, 2.5, 2.2]
    })


@pytest.fixture
def sample_experiment():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[1, 2]
    )


class TestNormalizationService:
    """Test suite for NormalizationService."""
    
    def test_normalize_none_method(self, sample_lipid_data, sample_experiment):
        """Test that 'none' method returns unchanged data."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE']
        )
        
        result = service.normalize(sample_lipid_data, config, sample_experiment)
        
        # Should return a copy with no changes
        pd.testing.assert_frame_equal(result, sample_lipid_data)
    
    def test_normalize_internal_standard_success(
        self, 
        sample_lipid_data, 
        sample_internal_standards,
        sample_experiment
    ):
        """Test successful internal standard normalization."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)', 'PE': 'PE(18:0_20:4)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0, 'PE(18:0_20:4)(d7)': 1.5}
        )
        
        result = service.normalize(
            sample_lipid_data,
            config,
            sample_experiment,
            intsta_df=sample_internal_standards
        )
        
        # Check that result has concentration columns
        assert 'concentration[s1]' in result.columns
        assert 'concentration[s2]' in result.columns
        assert 'concentration[s3]' in result.columns
        
        # Check that intensity columns are gone
        assert 'intensity[s1]' not in result.columns
        
        # Check that we have all lipids
        assert len(result) == 4
    
    def test_normalize_internal_standard_missing_intsta_df(
        self,
        sample_lipid_data,
        sample_experiment
    ):
        """Test that missing intsta_df raises error."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}
        )
        
        with pytest.raises(ValueError, match="intsta_df required"):
            service.normalize(sample_lipid_data, config, sample_experiment)
    
    def test_normalize_internal_standard_missing_standard(
        self,
        sample_lipid_data,
        sample_internal_standards,
        sample_experiment
    ):
        """Test that missing standard in intsta_df raises error."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'NonexistentStandard'},
            intsta_concentrations={'NonexistentStandard': 1.0}
        )
        
        with pytest.raises(ValueError, match="not found"):
            service.normalize(
                sample_lipid_data,
                config,
                sample_experiment,
                intsta_df=sample_internal_standards
            )
    
    def test_normalize_protein_success(
        self,
        sample_lipid_data,
        sample_protein_data,
        sample_experiment
    ):
        """Test successful protein normalization."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE'],
            protein_concentrations={'s1': 2.0, 's2': 2.5, 's3': 2.2}
        )
        
        result = service.normalize(
            sample_lipid_data,
            config,
            sample_experiment,
            protein_df=sample_protein_data
        )
        
        # Check that result has concentration columns
        assert 'concentration[s1]' in result.columns
        
        # Check calculation for one value
        # Original: 100.0, Protein: 2.0, Expected: 50.0
        assert result.loc[0, 'concentration[s1]'] == pytest.approx(50.0)
    
    def test_normalize_protein_missing_protein_df(
        self,
        sample_lipid_data,
        sample_experiment
    ):
        """Test that missing protein_df raises error."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.0, 's2': 2.5, 's3': 2.2}
        )
        
        with pytest.raises(ValueError, match="protein_df required"):
            service.normalize(sample_lipid_data, config, sample_experiment)
    
    def test_normalize_both_methods(
        self,
        sample_lipid_data,
        sample_internal_standards,
        sample_protein_data,
        sample_experiment
    ):
        """Test combined normalization (both internal standards and protein)."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)', 'PE': 'PE(18:0_20:4)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0, 'PE(18:0_20:4)(d7)': 1.5},
            protein_concentrations={'s1': 2.0, 's2': 2.5, 's3': 2.2}
        )
        
        result = service.normalize(
            sample_lipid_data,
            config,
            sample_experiment,
            intsta_df=sample_internal_standards,
            protein_df=sample_protein_data
        )
        
        # Should have concentration columns
        assert 'concentration[s1]' in result.columns
        
        # Result should be different from single method
        assert len(result) == 4
    
    def test_normalize_preserve_column_prefix(
        self,
        sample_lipid_data,
        sample_internal_standards,
        sample_experiment
    ):
        """Test that preserve_column_prefix keeps intensity prefix."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0},
            preserve_column_prefix=True
        )
        
        result = service.normalize(
            sample_lipid_data,
            config,
            sample_experiment,
            intsta_df=sample_internal_standards
        )
        
        # Should still have intensity columns
        assert 'intensity[s1]' in result.columns
        assert 'concentration[s1]' not in result.columns
    
    def test_normalize_filters_selected_classes(
        self,
        sample_lipid_data,
        sample_internal_standards,
        sample_experiment
    ):
        """Test that only selected classes are normalized."""
        service = NormalizationService()
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],  # Only PC, not PE
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}
        )
        
        result = service.normalize(
            sample_lipid_data,
            config,
            sample_experiment,
            intsta_df=sample_internal_standards
        )
        
        # Should only have PC lipids
        assert len(result) == 2
        assert all(result['ClassKey'] == 'PC')
    
    def test_normalize_handles_inf_values(
        self,
        sample_internal_standards,
        sample_experiment
    ):
        """Test that inf values are replaced with 0."""
        service = NormalizationService()
        
        # Create data that will produce inf (divide by zero)
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0]
        })
        
        # Create standards with zeros
        intsta_with_zeros = sample_internal_standards.copy()
        intsta_with_zeros.loc[0, 'intensity[s1]'] = 0.0
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}
        )
        
        result = service.normalize(
            lipid_data,
            config,
            sample_experiment,
            intsta_df=intsta_with_zeros
        )
        
        # Inf should be replaced with 0
        assert result.loc[0, 'concentration[s1]'] == 0.0
    
    def test_normalize_handles_nan_values(
        self,
        sample_internal_standards,
        sample_experiment
    ):
        """Test that NaN values are replaced with 0."""
        service = NormalizationService()
        
        # Create data with NaN
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [np.nan],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}
        )
        
        result = service.normalize(
            lipid_data,
            config,
            sample_experiment,
            intsta_df=sample_internal_standards
        )
        
        # NaN should be replaced with 0
        assert result.loc[0, 'concentration[s1]'] == 0.0
