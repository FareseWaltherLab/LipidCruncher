"""
Data models for experiment configuration.
"""
from typing import List
from pydantic import BaseModel, field_validator, computed_field


class ExperimentConfig(BaseModel):
    """
    Configuration for a lipidomics experiment.
    Defines the experimental conditions and sample organization.
    """
    n_conditions: int
    conditions_list: List[str]
    number_of_samples_list: List[int]
    
    @field_validator('n_conditions')
    @classmethod
    def validate_n_conditions(cls, v):
        if v < 1:
            raise ValueError("Number of conditions must be at least 1")
        return v
    
    @field_validator('conditions_list')
    @classmethod
    def validate_conditions_list(cls, v, info):
        n_conditions = info.data.get('n_conditions')
        if n_conditions and len(v) != n_conditions:
            raise ValueError(f"conditions_list length ({len(v)}) must match n_conditions ({n_conditions})")
        if len(v) != len(set(v)):
            raise ValueError("Condition names must be unique")
        return v
    
    @field_validator('number_of_samples_list')
    @classmethod
    def validate_number_of_samples_list(cls, v, info):
        n_conditions = info.data.get('n_conditions')
        if n_conditions and len(v) != n_conditions:
            raise ValueError(f"number_of_samples_list length ({len(v)}) must match n_conditions ({n_conditions})")
        if any(n < 1 for n in v):
            raise ValueError("Each condition must have at least 1 sample")
        return v
    
    @computed_field
    @property
    def samples_list(self) -> List[str]:
        """Generate list of all sample names (s1, s2, s3, ...)."""
        total_samples = sum(self.number_of_samples_list)
        return [f's{i+1}' for i in range(total_samples)]
    
    @computed_field
    @property
    def individual_samples_list(self) -> List[List[str]]:
        """
        Generate list of sample lists for each condition.
        Example: [['s1', 's2'], ['s3', 's4', 's5']]
        """
        result = []
        sample_idx = 0
        for n_samples in self.number_of_samples_list:
            condition_samples = [f's{sample_idx + i + 1}' for i in range(n_samples)]
            result.append(condition_samples)
            sample_idx += n_samples
        return result
    
    @computed_field
    @property
    def total_samples(self) -> int:
        """Total number of samples across all conditions."""
        return sum(self.number_of_samples_list)
    
    def get_condition_for_sample(self, sample: str) -> str:
        """
        Get the condition name for a given sample.
        
        Args:
            sample: Sample name (e.g., 's1', 's2')
        
        Returns:
            Condition name
        """
        for condition, samples in zip(self.conditions_list, self.individual_samples_list):
            if sample in samples:
                return condition
        raise ValueError(f"Sample {sample} not found in experiment")
