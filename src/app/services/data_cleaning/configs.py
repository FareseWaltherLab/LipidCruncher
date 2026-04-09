"""
Configuration classes for data cleaning.
"""
from typing import Dict, FrozenSet, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class GradeFilterConfig(BaseModel):
    """Configuration for LipidSearch grade filtering.

    Args:
        grade_config: Optional dict mapping ClassKey to list of acceptable grades.
                     If None, uses default filtering (A/B/C for all classes).
    """
    model_config = {"frozen": True}

    grade_config: Optional[Dict[str, List[str]]] = None

    @field_validator('grade_config')
    @classmethod
    def validate_grades(cls, v: Optional[Dict[str, List[str]]]) -> Optional[Dict[str, List[str]]]:
        if v is None:
            return v
        valid_grades = {'A', 'B', 'C', 'D'}
        for class_key, grades in v.items():
            invalid = set(grades) - valid_grades
            if invalid:
                raise ValueError(
                    f"Invalid grade(s) {invalid} for class '{class_key}'. "
                    f"Valid grades are: {sorted(valid_grades)}"
                )
        return v

    def __hash__(self) -> int:
        if self.grade_config is None:
            return hash(None)
        return hash(tuple(sorted((k, tuple(v)) for k, v in self.grade_config.items())))

    @property
    def is_default(self) -> bool:
        """Check if using default filtering."""
        return self.grade_config is None


class QualityFilterConfig(BaseModel):
    """Configuration for MS-DIAL quality filtering.

    Args:
        total_score_threshold: Minimum Total Score (0-100). 0 means no filtering.
        require_msms: Whether to require MS/MS matched = TRUE.
    """
    model_config = {"frozen": True}

    total_score_threshold: int = Field(default=0, ge=0, le=100)
    require_msms: bool = False

    def __hash__(self) -> int:
        return hash((self.total_score_threshold, self.require_msms))

    @classmethod
    def strict(cls) -> "QualityFilterConfig":
        """Create strict quality config (Score >= 80, MS/MS required)."""
        return cls(total_score_threshold=80, require_msms=True)

    @classmethod
    def moderate(cls) -> "QualityFilterConfig":
        """Create moderate quality config (Score >= 60)."""
        return cls(total_score_threshold=60, require_msms=False)

    @classmethod
    def permissive(cls) -> "QualityFilterConfig":
        """Create permissive quality config (Score >= 40)."""
        return cls(total_score_threshold=40, require_msms=False)

    @classmethod
    def no_filtering(cls) -> "QualityFilterConfig":
        """Create config with no quality filtering."""
        return cls(total_score_threshold=0, require_msms=False)


class CleaningResult(BaseModel):
    """Result object containing cleaned data and metadata.

    Args:
        cleaned_df: Cleaned DataFrame without internal standards.
        internal_standards_df: DataFrame containing only internal standards.
        filter_messages: Optional list of filter messages (e.g., rows removed).
    """
    model_config = {"arbitrary_types_allowed": True}

    cleaned_df: pd.DataFrame = Field(default_factory=pd.DataFrame)
    internal_standards_df: pd.DataFrame = Field(default_factory=pd.DataFrame)
    filter_messages: List[str] = Field(default_factory=list)