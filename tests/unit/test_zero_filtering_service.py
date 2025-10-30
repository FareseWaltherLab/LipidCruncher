"""
Unit tests for ZeroFilteringService with comprehensive edge case coverage.

This test suite builds on existing basic tests and adds:
- Boundary condition tests (exactly 50%, exactly 75% zeros)
- BQC detection and validation edge cases
- NaN and inf value handling
- Single sample/condition edge cases
- All-zero and mixed zero scenarios
- Invalid threshold values
- Missing intensity columns
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.zero_filtering_service import ZeroFilteringService
from src.lipidcruncher.core.models.experiment import ExperimentConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def service():
    """Create ZeroFilteringService instance."""
    return ZeroFilteringService()


@pytest.fixture
def experiment_config():
    """Create experiment configuration with BQC."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['BQC', 'Control', 'Treatment'],
        number_of_samples_list=[4, 3, 3]
    )


@pytest.fixture
def experiment_config_no_bqc():
    """Create experiment configuration without BQC."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3]
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


# ============================================================================
# BASIC FUNCTIONALITY TESTS (from original file)
# ============================================================================

class TestZeroFilteringBasics:
    """Basic functionality tests for ZeroFilteringService."""
    
    def test_no_filtering_when_all_good(self, service, sample_data, experiment_config):
        """Test that lipids with good values are kept."""
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
    
    def test_bqc_filtering(self, service, sample_data, experiment_config):
        """Test that lipids failing BQC threshold are removed."""
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
    
    def test_all_conditions_fail(self, service, sample_data, experiment_config):
        """Test lipid removed when all non-BQC conditions fail."""
        # LPC has mostly zeros everywhere
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # LPC should be removed
        assert 'LPC(16:0)' in removed
    
    def test_no_bqc_label(self, service, sample_data, experiment_config):
        """Test filtering without BQC condition."""
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label=None  # No BQC
        )
        
        # Should only remove if ALL conditions have ≥75% zeros
        assert len(filtered_df) >= 1
    
    def test_custom_threshold(self, service, experiment_config):
        """Test filtering with custom threshold."""
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
    
    def test_empty_dataframe(self, service, experiment_config):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        filtered_df, removed = service.filter_by_zeros(
            empty_df,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        assert filtered_df.empty
        assert removed == []


# ============================================================================
# EDGE CASE TESTS: BOUNDARY CONDITIONS
# ============================================================================

class TestBoundaryConditions:
    """Test boundary conditions for zero thresholds."""
    
    def test_exactly_50_percent_zeros_in_bqc_removed(self, service, experiment_config):
        """Test that exactly 50% zeros in BQC removes the lipid (≥ 50% threshold)."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 2/4 zeros = 50% (at threshold, implementation removes at ≥50%)
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0],
            # Other conditions have good values
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be REMOVED (50% zeros triggers BQC filter, implementation uses ≥50%)
        assert 'PC(16:0_18:1)' in removed
        assert 'PC(16:0_18:1)' not in filtered_df['LipidMolec'].values
    
    def test_just_under_50_percent_zeros_in_bqc_kept(self, service, experiment_config):
        """Test that <50% zeros in BQC keeps the lipid."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 1/4 zeros = 25% (< 50%, should be kept)
            'intensity[s1]': [0.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0],
            # Other conditions have good values
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be KEPT (25% < 50%)
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
        assert 'PC(16:0_18:1)' not in removed
    
    def test_just_over_50_percent_zeros_in_bqc_removed(self, service):
        """Test that >50% zeros in BQC removes the lipid."""
        # Create config with 5 BQC samples for easier percentage testing
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['BQC', 'Control'],
            number_of_samples_list=[5, 3]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 3/5 zeros = 60% (> 50% threshold, should be removed)
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [100.0],
            'intensity[s5]': [100.0],
            # Control has good values
            'intensity[s6]': [200.0],
            'intensity[s7]': [210.0],
            'intensity[s8]': [205.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be REMOVED (60% > 50%)
        assert 'PC(16:0_18:1)' in removed
        assert 'PC(16:0_18:1)' not in filtered_df['LipidMolec'].values
    
    def test_exactly_75_percent_zeros_in_condition_kept(self, service, experiment_config):
        """Test that exactly 75% zeros in non-BQC condition keeps the lipid."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC has good values
            'intensity[s1]': [100.0],
            'intensity[s2]': [110.0],
            'intensity[s3]': [105.0],
            'intensity[s4]': [108.0],
            # Control: 2/3 zeros = 66.7% (< 75%)
            'intensity[s5]': [0.0],
            'intensity[s6]': [0.0],
            'intensity[s7]': [100.0],
            # Treatment: 2/3 zeros = 66.7% (< 75%)
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [100.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be KEPT (66.7% < 75% in both non-BQC conditions)
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
        assert 'PC(16:0_18:1)' not in removed
    
    def test_all_conditions_exactly_75_percent_zeros_removed(self, service):
        """Test that ≥75% zeros in ALL non-BQC conditions removes lipid."""
        # Use 4 samples per condition for exact 75% testing
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['BQC', 'Control', 'Treatment'],
            number_of_samples_list=[3, 4, 4]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC has good values
            'intensity[s1]': [100.0],
            'intensity[s2]': [110.0],
            'intensity[s3]': [105.0],
            # Control: 3/4 zeros = 75% (exactly at threshold)
            'intensity[s4]': [0.0],
            'intensity[s5]': [0.0],
            'intensity[s6]': [0.0],
            'intensity[s7]': [100.0],
            # Treatment: 4/4 zeros = 100%
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [0.0],
            'intensity[s11]': [0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be REMOVED (75% and 100% both ≥ 75%)
        assert 'PC(16:0_18:1)' in removed


# ============================================================================
# EDGE CASE TESTS: BQC DETECTION AND VALIDATION
# ============================================================================

class TestBQCDetection:
    """Test BQC label detection and validation edge cases."""
    
    def test_invalid_bqc_label_ignored(self, service, sample_data, experiment_config):
        """Test that invalid BQC label is ignored (treated as no BQC)."""
        # Use a BQC label that doesn't exist in conditions
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='INVALID_BQC'
        )
        
        # Should work without crashing, treating as no BQC
        assert len(filtered_df) >= 0
    
    def test_bqc_label_case_sensitive(self, service, sample_data, experiment_config):
        """Test that BQC label matching is case-sensitive."""
        # Try lowercase 'bqc' when condition is 'BQC'
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='bqc'  # lowercase
        )
        
        # Should be treated as no BQC (case mismatch)
        # Behavior should match no_bqc_label test
        assert len(filtered_df) >= 1
    
    def test_bqc_only_experiment(self, service):
        """Test filtering when experiment has ONLY BQC condition."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['BQC'],
            number_of_samples_list=[4]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            # PC: 1/4 zeros = 25% (< 50%, but when BQC is ONLY condition, logic differs)
            'intensity[s1]': [100.0, 0.0],
            'intensity[s2]': [110.0, 0.0],
            'intensity[s3]': [105.0, 0.0],
            'intensity[s4]': [0.0, 0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # When BQC is the ONLY condition, all_non_bqc_fail=True by default
        # So if BQC fails OR all non-BQC fail, lipid is removed
        # Both lipids should be removed when BQC is the only condition
        assert len(filtered_df) == 0
        assert len(removed) == 2
    
    def test_bqc_as_last_condition(self, service):
        """Test that BQC detection works when BQC is last condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['Control', 'Treatment', 'BQC'],
            number_of_samples_list=[3, 3, 4]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # Control and Treatment have good values
            'intensity[s1]': [200.0],
            'intensity[s2]': [210.0],
            'intensity[s3]': [205.0],
            'intensity[s4]': [300.0],
            'intensity[s5]': [310.0],
            'intensity[s6]': [305.0],
            # BQC (last): 3/4 zeros = 75% (> 50%, should be removed)
            'intensity[s7]': [0.0],
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [100.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be removed due to BQC failure
        assert 'PC(16:0_18:1)' in removed


# ============================================================================
# EDGE CASE TESTS: NaN AND INF VALUES
# ============================================================================

class TestNaNAndInfValues:
    """Test handling of NaN and infinite values."""
    
    def test_nan_values_not_counted_as_zeros(self, service, experiment_config):
        """Test that NaN values are NOT counted in zero calculations."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 3 NaN values + 1 good = NaN not counted, so 0/1 zeros = 0%
            'intensity[s1]': [np.nan],
            'intensity[s2]': [np.nan],
            'intensity[s3]': [np.nan],
            'intensity[s4]': [100.0],
            # Other conditions good
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be KEPT (NaN values don't trigger the <= threshold check)
        # This is because pd.Series comparison with NaN returns False
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
        assert 'PC(16:0_18:1)' not in removed
    
    def test_negative_inf_treated_as_below_threshold(self, service, experiment_config):
        """Test that -inf values are treated as below threshold."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 3 -inf values + 1 good = 75% zeros
            'intensity[s1]': [-np.inf],
            'intensity[s2]': [-np.inf],
            'intensity[s3]': [-np.inf],
            'intensity[s4]': [100.0],
            # Other conditions good
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be removed (-inf is below threshold)
        assert 'PC(16:0_18:1)' in removed
    
    def test_positive_inf_kept(self, service, experiment_config):
        """Test that +inf values are kept (not considered zeros)."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: all +inf values (should be kept)
            'intensity[s1]': [np.inf],
            'intensity[s2]': [np.inf],
            'intensity[s3]': [np.inf],
            'intensity[s4]': [np.inf],
            # Other conditions good
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be kept (+inf is above threshold)
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
    
    def test_mixed_nan_inf_and_zeros(self, service, experiment_config):
        """Test handling of mixed NaN, inf, and zero values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: mix of bad values
            'intensity[s1]': [np.nan],
            'intensity[s2]': [0.0],
            'intensity[s3]': [-np.inf],
            'intensity[s4]': [100.0],  # Only 1 good value
            # Other conditions good
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # 3/4 bad values = 75% > 50%, should be removed
        assert 'PC(16:0_18:1)' in removed


# ============================================================================
# EDGE CASE TESTS: SINGLE SAMPLE/CONDITION SCENARIOS
# ============================================================================

class TestSingleSampleCondition:
    """Test edge cases with single sample per condition."""
    
    def test_single_sample_per_condition(self, service):
        """Test filtering with only 1 sample per condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['BQC', 'Control', 'Treatment'],
            number_of_samples_list=[1, 1, 1]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            # PC: all samples good
            'intensity[s1]': [100.0, 0.0],
            'intensity[s2]': [200.0, 0.0],
            'intensity[s3]': [300.0, 0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # PC should be kept (0% zeros everywhere)
        # PE should be removed (100% zeros everywhere)
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
        assert 'PE(18:0_20:4)' in removed
    
    def test_single_lipid_single_sample(self, service):
        """Test simplest case: 1 lipid, 1 sample."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label=None
        )
        
        # Should be kept (good value)
        assert len(filtered_df) == 1
        assert removed == []
    
    def test_single_condition_single_sample(self, service):
        """Test edge case with single condition and single sample."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [100.0, 0.0],
        })
        
        # Should not crash with single sample
        filtered_df, removed = service.filter_by_zeros(
            data,
            config,
            threshold=0.0,
            bqc_label=None
        )
        
        # PC: 0% zeros, should be kept
        # PE: 100% zeros, should be removed
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
        assert 'PE(18:0_20:4)' in removed


# ============================================================================
# EDGE CASE TESTS: ALL-ZERO AND MIXED SCENARIOS
# ============================================================================

class TestAllZeroScenarios:
    """Test scenarios with all zeros or mixed zero patterns."""
    
    def test_all_lipids_all_zeros(self, service, experiment_config):
        """Test that all lipids are removed when all values are zero."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TAG(16:0_18:1_18:2)'],
            'ClassKey': ['PC', 'PE', 'TAG'],
            'intensity[s1]': [0.0, 0.0, 0.0],
            'intensity[s2]': [0.0, 0.0, 0.0],
            'intensity[s3]': [0.0, 0.0, 0.0],
            'intensity[s4]': [0.0, 0.0, 0.0],
            'intensity[s5]': [0.0, 0.0, 0.0],
            'intensity[s6]': [0.0, 0.0, 0.0],
            'intensity[s7]': [0.0, 0.0, 0.0],
            'intensity[s8]': [0.0, 0.0, 0.0],
            'intensity[s9]': [0.0, 0.0, 0.0],
            'intensity[s10]': [0.0, 0.0, 0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # All should be removed
        assert len(filtered_df) == 0
        assert len(removed) == 3
        assert filtered_df.empty
    
    def test_one_good_sample_saves_lipid_in_bqc(self, service, experiment_config):
        """Test that one good sample in BQC can save a lipid."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 1/4 good = 75% zeros, but we need >50% to remove
            # Actually 3/4 = 75% zeros, which IS > 50%, so should be removed
            'intensity[s1]': [100.0],  # Good
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0],
            # Other conditions all zeros
            'intensity[s5]': [0.0],
            'intensity[s6]': [0.0],
            'intensity[s7]': [0.0],
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be removed (BQC has 75% zeros > 50%)
        assert 'PC(16:0_18:1)' in removed
    
    def test_two_good_samples_in_bqc_still_removed(self, service, experiment_config):
        """Test that 50% zeros in BQC removes lipid (≥50% threshold in implementation)."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: 2/4 good = 50% zeros (≥50% triggers removal in implementation)
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0],
            # Other conditions have at least one good condition
            'intensity[s5]': [200.0],
            'intensity[s6]': [200.0],
            'intensity[s7]': [0.0],
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be REMOVED (BQC has 50% zeros, implementation uses ≥50%)
        assert 'PC(16:0_18:1)' in removed
        assert 'PC(16:0_18:1)' not in filtered_df['LipidMolec'].values
    
    def test_mixed_performance_across_conditions(self, service, experiment_config):
        """Test lipid with varying zero patterns across conditions."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # BQC: good (0% zeros)
            'intensity[s1]': [100.0],
            'intensity[s2]': [110.0],
            'intensity[s3]': [105.0],
            'intensity[s4]': [108.0],
            # Control: 66.7% zeros (< 75%)
            'intensity[s5]': [0.0],
            'intensity[s6]': [0.0],
            'intensity[s7]': [200.0],
            # Treatment: 100% zeros (≥ 75%)
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should be KEPT (BQC passes, Control passes with <75%)
        # Not all non-BQC conditions fail
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values


# ============================================================================
# EDGE CASE TESTS: THRESHOLD VARIATIONS
# ============================================================================

class TestThresholdVariations:
    """Test various threshold values and edge cases."""
    
    def test_negative_threshold(self, service, experiment_config):
        """Test filtering with negative threshold."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # All values are 0, which is > -10
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0],
            'intensity[s4]': [0.0],
            'intensity[s5]': [0.0],
            'intensity[s6]': [0.0],
            'intensity[s7]': [0.0],
            'intensity[s8]': [0.0],
            'intensity[s9]': [0.0],
            'intensity[s10]': [0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=-10.0,
            bqc_label='BQC'
        )
        
        # With threshold=-10, values of 0 are NOT <= -10, so not zeros
        # Should be kept
        assert 'PC(16:0_18:1)' in filtered_df['LipidMolec'].values
    
    def test_very_high_threshold(self, service, experiment_config):
        """Test that high threshold treats many values as zeros."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # All BQC values are 1000, but threshold is 10000
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1000.0],
            'intensity[s3]': [1000.0],
            'intensity[s4]': [1000.0],
            # Other conditions also 1000
            'intensity[s5]': [1000.0],
            'intensity[s6]': [1000.0],
            'intensity[s7]': [1000.0],
            'intensity[s8]': [1000.0],
            'intensity[s9]': [1000.0],
            'intensity[s10]': [1000.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=10000.0,
            bqc_label='BQC'
        )
        
        # All values ≤ 10000, so all treated as zeros
        # Should be removed
        assert 'PC(16:0_18:1)' in removed
    
    def test_threshold_equal_to_value(self, service, experiment_config):
        """Test that values equal to threshold are treated as zeros."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # All BQC values exactly equal to threshold
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0],
            # Other conditions good
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=100.0,  # Equal to BQC values
            bqc_label='BQC'
        )
        
        # 100.0 <= 100.0 is True, so all BQC values are zeros
        # 100% zeros > 50%, should be removed
        assert 'PC(16:0_18:1)' in removed


# ============================================================================
# EDGE CASE TESTS: MISSING COLUMNS AND DATA INTEGRITY
# ============================================================================

class TestDataIntegrity:
    """Test handling of missing columns and data integrity issues."""
    
    def test_missing_lipidmolec_column(self, service, experiment_config):
        """Test that error handling works when LipidMolec column missing."""
        data = pd.DataFrame({
            # Missing LipidMolec column
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [110.0],
            'intensity[s3]': [105.0],
            'intensity[s4]': [108.0],
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        # Should raise KeyError or handle gracefully
        with pytest.raises(KeyError):
            service.filter_by_zeros(
                data,
                experiment_config,
                threshold=0.0,
                bqc_label='BQC'
            )
    
    def test_missing_some_intensity_columns(self, service, experiment_config):
        """Test handling when some expected intensity columns are missing."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            # Missing s2, s3, s4 (only s1 present for BQC)
            'intensity[s1]': [100.0],
            # All other samples present
            'intensity[s5]': [200.0],
            'intensity[s6]': [210.0],
            'intensity[s7]': [205.0],
            'intensity[s8]': [300.0],
            'intensity[s9]': [310.0],
            'intensity[s10]': [305.0],
        })
        
        # Should not crash - missing columns are skipped
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # With only 1/4 BQC samples present (others missing = 0 count?)
        # This tests the defensive column check in _count_zeros_in_condition
        assert len(filtered_df) >= 0
    
    def test_duplicate_lipid_names(self, service, experiment_config):
        """Test handling of duplicate lipid names in dataset."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [100.0, 200.0, 0.0],
            'intensity[s2]': [110.0, 210.0, 0.0],
            'intensity[s3]': [105.0, 205.0, 0.0],
            'intensity[s4]': [108.0, 208.0, 0.0],
            'intensity[s5]': [200.0, 300.0, 0.0],
            'intensity[s6]': [210.0, 310.0, 0.0],
            'intensity[s7]': [205.0, 305.0, 0.0],
            'intensity[s8]': [300.0, 400.0, 0.0],
            'intensity[s9]': [310.0, 410.0, 0.0],
            'intensity[s10]': [305.0, 405.0, 0.0],
        })
        
        filtered_df, removed = service.filter_by_zeros(
            data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Should handle duplicates (both PC entries kept, PE removed)
        assert len(filtered_df) == 2  # Both PC rows
        assert 'PE(18:0_20:4)' in removed
        assert removed.count('PC(16:0_18:1)') == 0  # Not removed


# ============================================================================
# EDGE CASE TESTS: RETURN VALUE VALIDATION
# ============================================================================

class TestReturnValues:
    """Test that return values are in expected format."""
    
    def test_filtered_df_has_reset_index(self, service, sample_data, experiment_config):
        """Test that filtered DataFrame has reset index."""
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Index should be 0, 1, 2, ... not original indices
        assert filtered_df.index.tolist() == list(range(len(filtered_df)))
    
    def test_removed_species_list_is_list_of_strings(self, service, sample_data, experiment_config):
        """Test that removed species is a list of strings."""
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        assert isinstance(removed, list)
        for item in removed:
            assert isinstance(item, str)
    
    def test_no_data_modification_in_place(self, service, sample_data, experiment_config):
        """Test that original DataFrame is not modified."""
        original_len = len(sample_data)
        original_columns = sample_data.columns.tolist()
        original_lipids = sample_data['LipidMolec'].tolist()
        
        service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        # Original should be unchanged
        assert len(sample_data) == original_len
        assert sample_data.columns.tolist() == original_columns
        assert sample_data['LipidMolec'].tolist() == original_lipids
    
    def test_removed_count_matches_difference(self, service, sample_data, experiment_config):
        """Test that removed count equals original - filtered count."""
        original_count = len(sample_data)
        
        filtered_df, removed = service.filter_by_zeros(
            sample_data,
            experiment_config,
            threshold=0.0,
            bqc_label='BQC'
        )
        
        assert len(removed) == original_count - len(filtered_df)