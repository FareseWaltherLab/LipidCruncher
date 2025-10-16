"""
Normalization configuration models.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class NormalizationConfig(BaseModel):
    """
    Configuration for lipid data normalization.
    
    Attributes:
        method: Normalization method ('internal_standard', 'protein', 'both', 'none')
        selected_classes: List of lipid classes to include
        internal_standards: Dict mapping lipid class to internal standard species
        intsta_concentrations: Dict mapping internal standard to concentration (pmol)
        protein_concentrations: Dict mapping sample ID to protein concentration (for BCA normalization)
        preserve_column_prefix: If True, keeps 'intensity[' prefix instead of changing to 'concentration['
    """
    method: str = Field(description="Normalization method to use")
    selected_classes: List[str] = Field(default_factory=list, description="Lipid classes to include")
    internal_standards: Dict[str, str] = Field(default_factory=dict, description="Class to internal standard mapping")
    intsta_concentrations: Dict[str, float] = Field(default_factory=dict, description="Internal standard concentrations (pmol)")
    protein_concentrations: Dict[str, float] = Field(default_factory=dict, description="Protein concentrations per sample (for BCA normalization)")
    preserve_column_prefix: bool = Field(default=False, description="Keep 'intensity[' prefix instead of changing to 'concentration['")
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v):
        """Validate normalization method."""
        valid_methods = ['internal_standard', 'protein', 'both', 'none']
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{v}'")
        return v
    
    @field_validator('intsta_concentrations')
    @classmethod
    def validate_concentrations_positive(cls, v):
        """Ensure all internal standard concentrations are positive."""
        for intsta, conc in v.items():
            if conc <= 0:
                raise ValueError(f"Concentration for {intsta} must be positive, got {conc}")
        return v
    
    @field_validator('protein_concentrations')
    @classmethod
    def validate_protein_concentrations_positive(cls, v):
        """Ensure all protein concentrations are positive."""
        for sample, conc in v.items():
            if conc <= 0:
                raise ValueError(f"Protein concentration for {sample} must be positive, got {conc}")
        return v
    
    def requires_internal_standards(self) -> bool:
        """Check if this configuration requires internal standards."""
        return self.method in ['internal_standard', 'both']
    
    def requires_protein_data(self) -> bool:
        """Check if this configuration requires protein concentration data."""
        return self.method in ['protein', 'both']
    
    def validate_complete(self) -> None:
        """
        Validate that configuration is complete for the chosen method.
        
        Raises:
            ValueError: If required fields are missing for the chosen method
        """
        if self.method == 'internal_standard' or self.method == 'both':
            if not self.internal_standards:
                raise ValueError("internal_standards required for 'internal_standard' or 'both' method")
            if not self.intsta_concentrations:
                raise ValueError("intsta_concentrations required for 'internal_standard' or 'both' method")
            
            # Check all internal standards have concentrations
            missing = set(self.internal_standards.values()) - set(self.intsta_concentrations.keys())
            if missing:
                raise ValueError(f"Missing concentrations for internal standards: {missing}")
        
        if self.method == 'protein' or self.method == 'both':
            if not self.protein_concentrations:
                raise ValueError("protein_concentrations required for 'protein' or 'both' method")
