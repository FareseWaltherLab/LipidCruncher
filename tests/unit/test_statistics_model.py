"""
Unit tests for StatisticalTestConfig model.
"""
import pytest
from pydantic import ValidationError
from src.lipidcruncher.core.models.statistics import StatisticalTestConfig


class TestStatisticalTestConfig:
    """Test suite for StatisticalTestConfig model."""
    
    def test_valid_manual_parametric_config(self):
        """Test creating valid manual mode parametric config."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='parametric',
            correction_method='fdr_bh',
            posthoc_correction='standard',
            auto_transform=True
        )
        
        assert config.mode == 'manual'
        assert config.test_type == 'parametric'
        assert config.is_parametric() == True
        assert config.is_auto_mode() == False
    
    def test_valid_auto_mode_config(self):
        """Test creating valid auto mode config."""
        config = StatisticalTestConfig(
            mode='auto',
            test_type='auto',
            correction_method='auto',
            posthoc_correction='auto'
        )
        
        assert config.is_auto_mode() == True
        assert config.is_parametric() == True  # Auto mode uses parametric
    
    def test_valid_non_parametric_config(self):
        """Test creating valid non-parametric config."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='non_parametric',
            correction_method='bonferroni',
            auto_transform=False
        )
        
        assert config.is_non_parametric() == True
        assert config.auto_transform == False
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = StatisticalTestConfig(test_type='parametric')
        
        assert config.mode == 'manual'
        assert config.correction_method == 'uncorrected'
        assert config.posthoc_correction == 'standard'
        assert config.alpha == 0.05
        assert config.auto_transform == True
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValidationError, match="mode must be one of"):
            StatisticalTestConfig(
                mode='invalid_mode',
                test_type='parametric'
            )
    
    def test_invalid_test_type_raises_error(self):
        """Test that invalid test type raises error."""
        with pytest.raises(ValidationError, match="test_type must be one of"):
            StatisticalTestConfig(test_type='welch')
    
    def test_invalid_correction_method_raises_error(self):
        """Test that invalid correction method raises error."""
        with pytest.raises(ValidationError, match="correction_method must be one of"):
            StatisticalTestConfig(
                test_type='parametric',
                correction_method='invalid_method'
            )
    
    def test_invalid_posthoc_correction_raises_error(self):
        """Test that invalid post-hoc correction raises error."""
        with pytest.raises(ValidationError, match="posthoc_correction must be one of"):
            StatisticalTestConfig(
                test_type='parametric',
                posthoc_correction='tukey'
            )
    
    def test_alpha_bounds(self):
        """Test that alpha is bounded between 0 and 1."""
        # Alpha too low
        with pytest.raises(ValidationError):
            StatisticalTestConfig(test_type='parametric', alpha=0.0)
        
        # Alpha too high
        with pytest.raises(ValidationError):
            StatisticalTestConfig(test_type='parametric', alpha=1.0)
        
        # Valid alpha
        config = StatisticalTestConfig(test_type='parametric', alpha=0.01)
        assert config.alpha == 0.01
    
    def test_condition_pairs_validation(self):
        """Test validation of condition pairs."""
        # Valid pairs
        config = StatisticalTestConfig(
            test_type='parametric',
            conditions_to_compare=[('Control', 'Treatment'), ('Control', 'High_Dose')]
        )
        assert len(config.conditions_to_compare) == 2
    
    def test_condition_pair_same_condition_raises_error(self):
        """Test that comparing condition to itself raises error."""
        with pytest.raises(ValidationError, match="Cannot compare condition to itself"):
            StatisticalTestConfig(
                test_type='parametric',
                conditions_to_compare=[('Control', 'Control')]
            )
    
    def test_condition_pair_wrong_length_raises_error(self):
        """Test that pairs with wrong number of elements raise error."""
        # Pydantic catches this automatically, just verify it raises ValidationError
        with pytest.raises(ValidationError):
            StatisticalTestConfig(
                test_type='parametric',
                conditions_to_compare=[('Control', 'Treatment', 'Extra')]
            )
    
    def test_requires_posthoc_two_conditions(self):
        """Test that post-hoc not required for 2 conditions."""
        config = StatisticalTestConfig(test_type='parametric')
        assert config.requires_posthoc(n_conditions=2) == False
    
    def test_requires_posthoc_three_conditions(self):
        """Test that post-hoc required for 3+ conditions."""
        config = StatisticalTestConfig(
            test_type='parametric',
            posthoc_correction='standard'
        )
        assert config.requires_posthoc(n_conditions=3) == True
    
    def test_requires_posthoc_uncorrected(self):
        """Test that post-hoc not required if uncorrected."""
        config = StatisticalTestConfig(
            test_type='parametric',
            posthoc_correction='uncorrected'
        )
        assert config.requires_posthoc(n_conditions=3) == False
