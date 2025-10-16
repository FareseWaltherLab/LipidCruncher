"""
Lipid data models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import pandas as pd


class LipidMeasurement(BaseModel):
    """
    Unified lipid measurement model supporting all data formats.
    
    Core fields (present in all formats):
        lipid_name: Lipid molecule identifier (LipidMolec)
        lipid_class: Lipid class (ClassKey)
        values: Intensity measurement values
        sample_ids: Sample identifiers
    
    Format-specific fields (optional):
        grade: Quality grade (A, B, C, D) - LipidSearch only
        calc_mass: Calculated mass - LipidSearch only
        base_rt: Base retention time - LipidSearch only
        fa_key: Fatty acid key - LipidSearch only
        sample_id_rate: Sample identification rate % - LipidSearch only
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core fields - present in ALL formats
    lipid_name: str = Field(description="Lipid molecule identifier (LipidMolec)")
    lipid_class: str = Field(description="Lipid class (ClassKey)")
    values: List[float] = Field(description="Intensity measurement values")
    sample_ids: List[str] = Field(description="Sample identifiers")
    
    # LipidSearch-specific fields (Optional)
    grade: Optional[str] = Field(default=None, description="Quality grade (LipidSearch only)")
    calc_mass: Optional[float] = Field(default=None, description="Calculated mass (LipidSearch only)")
    base_rt: Optional[float] = Field(default=None, description="Base retention time (LipidSearch only)")
    fa_key: Optional[str] = Field(default=None, description="Fatty acid key (LipidSearch only)")
    sample_id_rate: Optional[float] = Field(default=None, description="Sample identification rate % (LipidSearch only)")
    
    @field_validator('grade')
    @classmethod
    def validate_grade(cls, v):
        """Validate grade is one of A, B, C, D if provided."""
        if v is not None and v not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"Grade must be A, B, C, or D, got '{v}'")
        return v
    
    @field_validator('calc_mass', 'base_rt', 'sample_id_rate')
    @classmethod
    def validate_positive_if_present(cls, v):
        """Ensure numeric fields are positive if provided."""
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative")
        return v
    
    @model_validator(mode='after')
    def validate_lengths_match(self):
        """Ensure values and sample_ids have same length."""
        if len(self.values) != len(self.sample_ids):
            raise ValueError(
                f"values and sample_ids must have same length: "
                f"got {len(self.values)} values and {len(self.sample_ids)} sample_ids"
            )
        return self
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series with sample_ids as index."""
        return pd.Series(self.values, index=self.sample_ids, name=self.lipid_name)
    
    def is_lipidsearch_format(self) -> bool:
        """Check if this measurement came from LipidSearch format."""
        return self.grade is not None


class LipidDataset(BaseModel):
    """
    Collection of lipid measurements for an experiment.
    
    Attributes:
        measurements: List of lipid measurements
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    measurements: List[LipidMeasurement] = Field(default_factory=list, description="Lipid measurements")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all measurements to a DataFrame."""
        if not self.measurements:
            return pd.DataFrame()
        
        series_list = [m.to_series() for m in self.measurements]
        return pd.concat(series_list, axis=1).T
    
    def filter_by_class(self, lipid_classes: List[str]) -> 'LipidDataset':
        """Filter measurements by lipid class."""
        filtered = [m for m in self.measurements if m.lipid_class in lipid_classes]
        return LipidDataset(measurements=filtered)
    
    def filter_by_grade(self, grades: List[str]) -> 'LipidDataset':
        """
        Filter measurements by quality grade.
        Only applies to LipidSearch data - others are unchanged.
        """
        filtered = [m for m in self.measurements if m.grade is None or m.grade in grades]
        return LipidDataset(measurements=filtered)
    
    def get_lipid_classes(self) -> List[str]:
        """Get unique list of lipid classes in dataset."""
        return list(set(m.lipid_class for m in self.measurements))
    
    def get_lipidsearch_measurements(self) -> 'LipidDataset':
        """Get only measurements from LipidSearch format (have grade info)."""
        filtered = [m for m in self.measurements if m.is_lipidsearch_format()]
        return LipidDataset(measurements=filtered)
    
    def count(self) -> int:
        """Get number of measurements."""
        return len(self.measurements)
