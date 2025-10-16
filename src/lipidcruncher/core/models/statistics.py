"""
Statistical testing configuration models.
"""
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class StatisticalTestConfig(BaseModel):
    """
    Configuration for statistical testing.
    
    Attributes:
        mode: Analysis mode ('auto' or 'manual')
        test_type: Type of test ('parametric', 'non_parametric', 'auto')
        correction_method: Level 1 correction - between class/group ('uncorrected', 'fdr_bh', 'bonferroni', 'auto')
        posthoc_correction: Level 2 correction - within class for 3+ conditions ('uncorrected', 'standard', 'bonferroni_all', 'auto')
        alpha: Significance level (default 0.05)
        auto_transform: If True, applies log10 transformation to normalize data
        conditions_to_compare: List of condition pairs to compare (for pairwise tests)
    """
    mode: str = Field(default='manual', description="Analysis mode")
    test_type: str = Field(description="Statistical test to perform")
    correction_method: str = Field(default='uncorrected', description="Level 1: Between-class correction method")
    posthoc_correction: str = Field(default='standard', description="Level 2: Within-class post-hoc correction")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance threshold")
    auto_transform: bool = Field(default=True, description="Apply log10 transformation")
    conditions_to_compare: List[Tuple[str, str]] = Field(default_factory=list, description="Pairs of conditions to compare")
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        """Validate analysis mode."""
        valid_modes = ['auto', 'manual']
        if v not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{v}'")
        return v
    
    @field_validator('test_type')
    @classmethod
    def validate_test_type(cls, v):
        """Validate statistical test type."""
        valid_tests = ['parametric', 'non_parametric', 'auto']
        if v not in valid_tests:
            raise ValueError(f"test_type must be one of {valid_tests}, got '{v}'")
        return v
    
    @field_validator('correction_method')
    @classmethod
    def validate_correction_method(cls, v):
        """Validate Level 1 correction method."""
        valid_methods = ['uncorrected', 'fdr_bh', 'bonferroni', 'auto']
        if v not in valid_methods:
            raise ValueError(f"correction_method must be one of {valid_methods}, got '{v}'")
        return v
    
    @field_validator('posthoc_correction')
    @classmethod
    def validate_posthoc_correction(cls, v):
        """Validate Level 2 post-hoc correction method."""
        valid_methods = ['uncorrected', 'standard', 'bonferroni_all', 'auto']
        if v not in valid_methods:
            raise ValueError(f"posthoc_correction must be one of {valid_methods}, got '{v}'")
        return v
    
    @field_validator('conditions_to_compare')
    @classmethod
    def validate_condition_pairs(cls, v):
        """Validate condition pairs are unique and properly formatted."""
        if not v:
            return v
        
        # Check each pair has exactly 2 elements
        for pair in v:
            if len(pair) != 2:
                raise ValueError(f"Each comparison pair must have exactly 2 conditions, got {pair}")
            if pair[0] == pair[1]:
                raise ValueError(f"Cannot compare condition to itself: {pair}")
        
        return v
    
    def is_auto_mode(self) -> bool:
        """Check if running in auto mode."""
        return self.mode == 'auto'
    
    def is_parametric(self) -> bool:
        """Check if using parametric tests."""
        return self.test_type == 'parametric' or (self.test_type == 'auto' and self.mode == 'auto')
    
    def is_non_parametric(self) -> bool:
        """Check if using non-parametric tests."""
        return self.test_type == 'non_parametric'
    
    def requires_posthoc(self, n_conditions: int) -> bool:
        """Check if post-hoc tests are needed (3+ conditions)."""
        return n_conditions > 2 and self.posthoc_correction != 'uncorrected'
