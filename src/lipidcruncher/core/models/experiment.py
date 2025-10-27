"""
Enhanced Experiment configuration model with all derived fields.
Replaces the old mutable Experiment class with immutable Pydantic model.
"""
from pydantic import BaseModel, Field, field_validator, computed_field
from typing import List
import pandas as pd


class ExperimentConfig(BaseModel):
    """
    Experiment configuration with automatic derived field calculations.
    
    Attributes:
        n_conditions: Number of experimental conditions
        conditions_list: Labels for each condition
        number_of_samples_list: Number of samples per condition
    
    Computed Properties:
        full_samples_list: All sample labels (s1, s2, s3, ...)
        aggregate_number_of_samples_list: Cumulative sample counts
        individual_samples_list: Samples grouped by condition
        extensive_conditions_list: Flat list replicating conditions for each sample
    """
    
    n_conditions: int = Field(gt=0, description="Number of experimental conditions")
    conditions_list: List[str] = Field(description="Labels for each condition")
    number_of_samples_list: List[int] = Field(description="Number of samples per condition")
    
    @field_validator('conditions_list')
    @classmethod
    def validate_conditions_not_empty(cls, v):
        """Ensure all condition labels are non-empty strings."""
        if not all(label and label.strip() for label in v):
            raise ValueError("All condition labels must be non-empty")
        return v
    
    @field_validator('number_of_samples_list')
    @classmethod
    def validate_sample_counts(cls, v):
        """Ensure all sample counts are positive."""
        if not all(count > 0 for count in v):
            raise ValueError("All sample counts must be greater than 0")
        return v
    
    @computed_field
    @property
    def full_samples_list(self) -> List[str]:
        """
        Generate complete list of sample labels (s1, s2, s3, ...).
        
        Returns:
            List of sample labels
        """
        return [f's{i+1}' for i in range(sum(self.number_of_samples_list))]
    
    @computed_field
    @property
    def aggregate_number_of_samples_list(self) -> List[int]:
        """
        Calculate cumulative sample counts for each condition.
        
        Returns:
            List of cumulative counts [n1, n1+n2, n1+n2+n3, ...]
        """
        return [sum(self.number_of_samples_list[:i+1]) 
                for i in range(len(self.number_of_samples_list))]
    
    @computed_field
    @property
    def individual_samples_list(self) -> List[List[str]]:
        """
        Group sample labels by their experimental condition.
        
        Returns:
            List of lists, where each sublist contains samples for one condition
        """
        result = []
        start_index = 0
        for num_samples in self.number_of_samples_list:
            end_index = start_index + num_samples
            result.append(self.full_samples_list[start_index:end_index])
            start_index = end_index
        return result
    
    @computed_field
    @property
    def extensive_conditions_list(self) -> List[str]:
        """
        Create flat list replicating condition labels for each sample.
        Example: ['WT', 'WT', 'WT', 'KO', 'KO', 'KO'] for 3 samples each
        
        Returns:
            Flat list of condition labels
        """
        return [condition 
                for condition, num_samples in zip(self.conditions_list, self.number_of_samples_list) 
                for _ in range(num_samples)]
    
    def remove_bad_samples(self, bad_samples: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove bad samples from both the experiment configuration and dataframe.
        Creates a new ExperimentConfig with updated sample lists.
        
        Args:
            bad_samples: List of sample labels to remove
            df: DataFrame to update
            
        Returns:
            Updated DataFrame with bad sample columns removed
        """
        # Remove bad sample columns from DataFrame
        for sample in bad_samples:
            col_name = f'concentration[{sample}]'
            if col_name in df.columns:
                df = df.drop(columns=[col_name])
        
        # Update sample lists
        updated_full_samples = [s for s in self.full_samples_list if s not in bad_samples]
        
        # Count samples per condition after removal
        from collections import OrderedDict
        condition_counts = OrderedDict()
        
        for sample in updated_full_samples:
            # Find which condition this sample belongs to
            original_index = self.full_samples_list.index(sample)
            condition = self.extensive_conditions_list[original_index]
            
            if condition not in condition_counts:
                condition_counts[condition] = 0
            condition_counts[condition] += 1
        
        # Filter out conditions with no samples
        updated_conditions = [cond for cond, count in condition_counts.items() if count > 0]
        updated_sample_counts = [count for count in condition_counts.values() if count > 0]
        
        # Update self with new values
        self.n_conditions = len(updated_conditions)
        self.conditions_list = updated_conditions
        self.number_of_samples_list = updated_sample_counts
        
        return df
    
    model_config = {
        "frozen": False,  # Allow mutation for bad sample removal
        "validate_assignment": True  # Validate on assignment
    }
    
    @classmethod
    def from_user_input(
        cls,
        n_conditions: int,
        conditions_list: List[str],
        number_of_samples_list: List[int]
    ) -> "ExperimentConfig":
        """
        Factory method to create ExperimentConfig from user input.
        Validates inputs and raises ValueError if invalid.
        
        Args:
            n_conditions: Number of conditions
            conditions_list: Condition labels
            number_of_samples_list: Sample counts per condition
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            ValueError: If validation fails
        """
        return cls(
            n_conditions=n_conditions,
            conditions_list=conditions_list,
            number_of_samples_list=number_of_samples_list
        )