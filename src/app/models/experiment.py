"""
Experiment configuration model.
"""
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator
from typing import List, Optional


class ExperimentConfig(BaseModel):
    """
    Experiment configuration with automatic derived field calculations.

    Attributes:
        n_conditions: Number of experimental conditions (auto-derived from
            conditions_list if not provided; validated for consistency if provided)
        conditions_list: Labels for each condition
        number_of_samples_list: Number of samples per condition

    Computed Properties:
        full_samples_list: All sample labels (s1, s2, s3, ...)
        aggregate_number_of_samples_list: Cumulative sample counts
        individual_samples_list: Samples grouped by condition
        extensive_conditions_list: Flat list replicating conditions for each sample
    """
    model_config = {"frozen": True}

    n_conditions: int = Field(gt=0, description="Number of experimental conditions")
    conditions_list: List[str] = Field(description="Labels for each condition")
    number_of_samples_list: List[int] = Field(description="Number of samples per condition")

    @model_validator(mode='before')
    @classmethod
    def auto_derive_n_conditions(cls, data):
        """Auto-derive n_conditions from conditions_list when not provided."""
        if isinstance(data, dict):
            conditions = data.get('conditions_list')
            if conditions is not None and 'n_conditions' not in data:
                data = {**data, 'n_conditions': len(conditions)}
        return data

    def __hash__(self) -> int:
        return hash((self.n_conditions, tuple(self.conditions_list), tuple(self.number_of_samples_list)))

    @field_validator('conditions_list')
    @classmethod
    def validate_conditions_not_empty(cls, v):
        """Ensure all condition labels are non-empty strings and unique."""
        if not all(label and label.strip() for label in v):
            raise ValueError("All condition labels must be non-empty")
        if len(v) != len(set(v)):
            duplicates = [label for label in v if v.count(label) > 1]
            raise ValueError(
                f"Condition labels must be unique, found duplicates: {sorted(set(duplicates))}"
            )
        return v

    @field_validator('number_of_samples_list')
    @classmethod
    def validate_sample_counts(cls, v):
        """Ensure all sample counts are positive."""
        if not all(count > 0 for count in v):
            raise ValueError("All sample counts must be greater than 0")
        return v

    @model_validator(mode='after')
    def validate_list_lengths(self) -> 'ExperimentConfig':
        """Ensure conditions_list and number_of_samples_list match n_conditions."""
        if len(self.conditions_list) != self.n_conditions:
            raise ValueError(
                f"Length of conditions_list ({len(self.conditions_list)}) "
                f"must match n_conditions ({self.n_conditions})"
            )
        if len(self.number_of_samples_list) != self.n_conditions:
            raise ValueError(
                f"Length of number_of_samples_list ({len(self.number_of_samples_list)}) "
                f"must match n_conditions ({self.n_conditions})"
            )
        return self

    @computed_field
    @property
    def full_samples_list(self) -> List[str]:
        """Generate complete list of sample labels (s1, s2, s3, ...)."""
        return [f's{i+1}' for i in range(sum(self.number_of_samples_list))]

    @computed_field
    @property
    def aggregate_number_of_samples_list(self) -> List[int]:
        """Calculate cumulative sample counts [n1, n1+n2, n1+n2+n3, ...]."""
        return [sum(self.number_of_samples_list[:i+1])
                for i in range(len(self.number_of_samples_list))]

    @computed_field
    @property
    def individual_samples_list(self) -> List[List[str]]:
        """Group sample labels by their experimental condition."""
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
        """
        return [condition
                for condition, num_samples in zip(self.conditions_list, self.number_of_samples_list)
                for _ in range(num_samples)]

    def without_samples(self, samples_to_remove: List[str]) -> "ExperimentConfig":
        """
        Create a new ExperimentConfig with specified samples removed.

        Args:
            samples_to_remove: List of sample labels to remove (e.g., ['s1', 's3'])

        Returns:
            New ExperimentConfig with updated sample counts

        Raises:
            ValueError: If removing samples would leave a condition with no samples
        """
        if not samples_to_remove:
            return self

        # Validate that all samples to remove actually exist
        unknown = set(samples_to_remove) - set(self.full_samples_list)
        if unknown:
            raise ValueError(
                f"Cannot remove unknown samples: {sorted(unknown)}. "
                f"Valid samples are: {self.full_samples_list}"
            )

        # Count samples per condition after removal
        new_conditions = []
        new_sample_counts = []

        for condition, samples in zip(self.conditions_list, self.individual_samples_list):
            remaining_in_condition = [s for s in samples if s not in samples_to_remove]
            if remaining_in_condition:
                new_conditions.append(condition)
                new_sample_counts.append(len(remaining_in_condition))

        if not new_conditions:
            raise ValueError("Cannot remove all samples from all conditions")

        return ExperimentConfig(
            n_conditions=len(new_conditions),
            conditions_list=new_conditions,
            number_of_samples_list=new_sample_counts
        )
