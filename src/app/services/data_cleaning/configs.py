"""
Configuration classes for data cleaning.
"""
import pandas as pd
from typing import Dict, List, Optional


class GradeFilterConfig:
    """Configuration for LipidSearch grade filtering."""

    def __init__(self, grade_config: Optional[Dict[str, List[str]]] = None):
        """
        Initialize grade filter configuration.

        Args:
            grade_config: Optional dict mapping ClassKey to list of acceptable grades.
                         If None, uses default filtering (A/B/C for all classes).
        """
        self.grade_config = grade_config

    @property
    def is_default(self) -> bool:
        """Check if using default filtering."""
        return self.grade_config is None


class QualityFilterConfig:
    """Configuration for MS-DIAL quality filtering."""

    def __init__(
        self,
        total_score_threshold: int = 0,
        require_msms: bool = False
    ):
        """
        Initialize quality filter configuration.

        Args:
            total_score_threshold: Minimum Total Score (0-100). 0 means no filtering.
            require_msms: Whether to require MS/MS matched = TRUE.
        """
        self.total_score_threshold = total_score_threshold
        self.require_msms = require_msms

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


class CleaningResult:
    """Result object containing cleaned data and metadata."""

    def __init__(
        self,
        cleaned_df: pd.DataFrame,
        internal_standards_df: pd.DataFrame,
        filter_messages: Optional[List[str]] = None
    ):
        """
        Initialize cleaning result.

        Args:
            cleaned_df: Cleaned DataFrame without internal standards.
            internal_standards_df: DataFrame containing only internal standards.
            filter_messages: Optional list of filter messages (e.g., rows removed).
        """
        self.cleaned_df = cleaned_df
        self.internal_standards_df = internal_standards_df
        self.filter_messages = filter_messages or []
