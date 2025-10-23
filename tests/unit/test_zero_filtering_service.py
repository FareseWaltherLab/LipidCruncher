"""
Unit tests for ZeroFilteringService.
"""
import pytest
import pandas as pd
from src.lipidcruncher.core.services.zero_filtering_service import ZeroFilteringService
from src.lipidcruncher.core.models.experiment import ExperimentConfig


@pytest.fixture
def experiment_config():
    """Create experiment configuration with BQC."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['BQC', 'Control', 'Treatment'],
        number_of_samples_list=[4, 3, 3]
    )


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TAG(16:0_18:1_18:2)', 'LPC(16:0)'],
        'ClassKey': ['PC', 'PE', 'TAG', 'LPC'],
        # BQC samples (s1-s4)
        'intensity[s1]': [100.0, 0.0, 50.0, 0.0],
        'intensity[s2]': [110.0, 0.0, 55.0, 0.0],
        'intensity[s3]': [105.0, 10.0, 52.0, 5.0],
        'intensity[s4]': [108.0, 0.0, 53.0, 0.0],
        # Control samples (s5-s7)
        'intensity[s5]': [200.0, 0.0, 100.0, 10.0],
        'intensity[s6]': [210.0, 0.0, 110.0, 0.0],
        'intensity[s7]': [205.0, 5.0, 105.0, 0.0],
        # Treatment samples (s8-s10)
        'intensity[s8]': [300.0, 0.0, 150.0, 0.0],
        'intensity[s9]': [310.0, 10.0, 160.0, 5.0],
        'intensity[s10]': [305.0, 0.0, 155.0, 0.0],
    })


class TestZeroFilteringService:
    """Test suite for ZeroFilteringService."""
    
    def test_no_filtering_when_all_good(self, sample_data, experiment_config):
        """Test that lipids with good values are kept."""
        service = ZeroFilteringService()
        
        # PC and TAG have good values everywhere
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # PC and TAG should remain
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
        assert 'TAG(16:0_18:1_18:2)' in filtered_df['LipidMolec'].values
    
    def test_bqc_filtering(self, sample_data, experiment_config):
        """Test that lipids failing BQC threshold are removed."""
        service = ZeroFilteringService()
        
        # PE has 3/4 zeros in BQC (75% > 50% threshold)
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # PE should be removed (fails BQC)
        assert 'PE(18:0_20:4)' in removed
        assert 'PE(18:0_20:4)' not in filtered_df['LipidMolec'].values
    
    def test_all_conditions_fail(self, sample_data, experiment_config):
        """Test lipid removed when all non-BQC conditions fail."""
        service = ZeroFilteringService()
        
        # LPC has mostly zeros everywhere
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # LPC should be removed
        assert 'LPC(16:0)' in removed
    
    def test_no_bqc_label(self, sample_data, experiment_config):
        """Test filtering without BQC condition."""
        service = ZeroFilteringService()
        
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label=None  # No BQC
        )
        
        # Should only remove if ALL conditions have ≥75% zeros
        assert len(filtered_df) >= 1
    
    def test_custom_threshold(self, experiment_config):
        """Test filtering with custom threshold."""
        service = ZeroFilteringService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5.0],
            'intensity[s2]': [5.0],
            'intensity[s3]': [5.0],
            'intensity[s4]': [5.0],
            'intensity[s5]': [100.0],
            'intensity[s6]': [110.0],
            'intensity[s7]': [105.0],
            'intensity[s8]': [200.0],
            'intensity[s9]': [210.0],
            'intensity[s10]': [205.0],
        })
        
        # With threshold=10, all BQC values (5.0) should be considered zeros
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=10.0,
            bqc_label='BQC'
        )
        
        # PC should be removed (100% of BQC ≤ threshold)
        assert 'PC(16:0_18:1)' in removed
    
    def test_empty_dataframe(self, experiment_config):
        """Test handling of empty DataFrame."""
        service = ZeroFilteringService()
        
        empty_df = pd.DataFrame()
        filtered_df, removed = service.filter_by_zeros(
            empty_df,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        assert filtered_df.empty
        assert removed == []
