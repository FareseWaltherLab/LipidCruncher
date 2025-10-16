"""
Experiment configuration model with validation.
"""
from typing import List
from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    """
    Configuration for a lipidomics experiment.
    
    Attributes:
        n_conditions: Number of experimental conditions
        conditions_list: Names of each condition (e.g., ['Control', 'Treatment'])
        number_of_samples_list: Number of samples per condition
        samples_list: Generated list of sample IDs (e.g., ['s1', 's2', ...])
    """
    n_conditions: int = Field(gt=0, description="Number of conditions (must be positive)")
    conditions_list: List[str] = Field(min_length=1, description="List of condition names")
    number_of_samples_list: List[int] = Field(min_length=1, description="Number of samples per condition")
    samples_list: List[str] = Field(default_factory=list, description="Generated sample IDs")
    
    @field_validator('number_of_samples_list')
    @classmethod
    def validate_sample_counts(cls, v):
        """Ensure all sample counts are positive."""
        if any(count <= 0 for count in v):
            raise ValueError("All sample counts must be positive integers")
        return v
    
    @field_validator('conditions_list', 'number_of_samples_list')
    @classmethod
    def validate_lists_match_n_conditions(cls, v, info):
        """Ensure list lengths match n_conditions."""
        if 'n_conditions' in info.data:
            n_conditions = info.data['n_conditions']
            if len(v) != n_conditions:
                raise ValueError(f"List length ({len(v)}) must match n_conditions ({n_conditions})")
        return v
    
    def model_post_init(self, __context):
        """Generate samples_list after validation if not provided."""
        if not self.samples_list:
            self.samples_list = self._generate_samples()
    
    def _generate_samples(self) -> List[str]:
        """Generate sample IDs like ['s1', 's2', 's3', ...]"""
        total_samples = sum(self.number_of_samples_list)
        return [f's{i+1}' for i in range(total_samples)]
    
    def get_samples_for_condition(self, condition: str) -> List[str]:
        """
        Get the sample IDs for a specific condition.
        
        Args:
            condition: Name of the condition
            
        Returns:
            List of sample IDs for that condition
            
        Raises:
            ValueError: If condition not found
        """
        if condition not in self.conditions_list:
            raise ValueError(f"Condition '{condition}' not found in {self.conditions_list}")
        
        condition_idx = self.conditions_list.index(condition)
        start_idx = sum(self.number_of_samples_list[:condition_idx])
        end_idx = start_idx + self.number_of_samples_list[condition_idx]
        
        return self.samples_list[start_idx:end_idx]
