"""
Unit tests for ExperimentConfig model.
"""
import pytest
from pydantic import ValidationError
from src.lipidcruncher.core.models.experiment import ExperimentConfig


class TestExperimentConfig:
    """Test suite for ExperimentConfig model."""
    
    def test_valid_experiment_creation(self):
        """Test creating a valid experiment configuration."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 3]
        )
        
        assert config.n_conditions == 2
        assert config.conditions_list == ['Control', 'Treatment']
        assert config.number_of_samples_list == [3, 3]
        assert config.samples_list == ['s1', 's2', 's3', 's4', 's5', 's6']
    
    def test_samples_generated_automatically(self):
        """Test that samples_list is auto-generated."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[5]
        )
        
        assert len(config.samples_list) == 5
        assert config.samples_list == ['s1', 's2', 's3', 's4', 's5']
    
    def test_mismatched_conditions_count_raises_error(self):
        """Test error when conditions_list length doesn't match n_conditions."""
        with pytest.raises(ValidationError, match="must match n_conditions"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['Control'],  # Only 1 condition for n_conditions=2
                number_of_samples_list=[3, 3]
            )
    
    def test_mismatched_samples_count_raises_error(self):
        """Test error when number_of_samples_list length doesn't match n_conditions."""
        with pytest.raises(ValidationError, match="must match"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['Control', 'Treatment'],
                number_of_samples_list=[3]  # Only 1 count for 2 conditions
            )
    
    def test_negative_sample_count_raises_error(self):
        """Test that negative sample counts are rejected."""
        with pytest.raises(ValidationError, match="positive"):
            ExperimentConfig(
                n_conditions=1,
                conditions_list=['Control'],
                number_of_samples_list=[-1]
            )
    
    def test_zero_sample_count_raises_error(self):
        """Test that zero sample counts are rejected."""
        with pytest.raises(ValidationError, match="positive"):
            ExperimentConfig(
                n_conditions=1,
                conditions_list=['Control'],
                number_of_samples_list=[0]
            )
    
    def test_get_samples_for_condition(self):
        """Test getting samples for a specific condition."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[2, 3]
        )
        
        control_samples = config.get_samples_for_condition('Control')
        assert control_samples == ['s1', 's2']
        
        treatment_samples = config.get_samples_for_condition('Treatment')
        assert treatment_samples == ['s3', 's4', 's5']
    
    def test_get_samples_for_nonexistent_condition_raises_error(self):
        """Test error when requesting nonexistent condition."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[3]
        )
        
        with pytest.raises(ValueError, match="not found"):
            config.get_samples_for_condition('Nonexistent')
