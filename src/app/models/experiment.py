"""
Experiment configuration model.
"""
from pydantic import BaseModel, Field, field_validator, computed_field
from typing import List


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

        # Calculate remaining samples per condition
        remaining_samples = [s for s in self.full_samples_list if s not in samples_to_remove]

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

    @classmethod
    def from_user_input(
        cls,
        n_conditions: int,
        conditions_list: List[str],
        number_of_samples_list: List[int]
    ) -> "ExperimentConfig":
        """
        Factory method to create ExperimentConfig from user input.

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
