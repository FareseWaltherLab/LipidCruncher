"""
Data models for normalization configuration and results.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, field_validator, model_validator


class NormalizationConfig(BaseModel):
    """
    Configuration for data normalization.
    Supports internal standards, protein-based, or combined normalization.
    """
    method: Literal['none', 'internal_standard', 'protein', 'both']
    selected_classes: List[str]
    internal_standards: Optional[Dict[str, str]] = None
    intsta_concentrations: Optional[Dict[str, float]] = None
    protein_concentrations: Optional[Dict[str, float]] = None
    preserve_column_prefix: bool = False
    
    @field_validator('intsta_concentrations')
    @classmethod
    def validate_concentrations_positive(cls, v):
        """Ensure all concentrations are positive."""
        if v is not None:
            for standard, conc in v.items():
                if conc <= 0:
                    raise ValueError(f"Concentration for {standard} must be positive, got {conc}")
        return v
    
    @field_validator('protein_concentrations')
    @classmethod
    def validate_protein_concentrations_positive(cls, v):
        """Ensure all protein concentrations are positive."""
        if v is not None:
            for sample, conc in v.items():
                if conc <= 0:
                    raise ValueError(f"Protein concentration for {sample} must be positive, got {conc}")
        return v
    
    @model_validator(mode='after')
    def validate_method_requirements(self):
        """Validate that required fields are present for each method."""
        if self.method == 'internal_standard':
            if not self.internal_standards or not self.intsta_concentrations:
                raise ValueError("internal_standard method requires internal_standards and intsta_concentrations")
        
        elif self.method == 'protein':
            if not self.protein_concentrations:
                raise ValueError("protein method requires protein_concentrations")
        
        elif self.method == 'both':
            if not self.internal_standards or not self.intsta_concentrations:
                raise ValueError("both method requires internal_standards and intsta_concentrations")
            if not self.protein_concentrations:
                raise ValueError("both method requires protein_concentrations")
        
        return self
    
    def validate_complete(self) -> bool:
        """
        Validate that the configuration is complete for the selected method.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is incomplete
        """
        # This is handled by model_validator above
        return True


class NormalizationResult(BaseModel):
    """Results from normalization operation."""
    method: str
    lipids_normalized: int
    classes_normalized: List[str]
    standards_used: Optional[List[str]] = None
