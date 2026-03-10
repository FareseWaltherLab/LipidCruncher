"""
Configuration classes for data cleaning.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class GradeFilterConfig:
    """Configuration for LipidSearch grade filtering.

    Args:
        grade_config: Optional dict mapping ClassKey to list of acceptable grades.
                     If None, uses default filtering (A/B/C for all classes).
    """
    grade_config: Optional[Dict[str, List[str]]] = None

    def __hash__(self) -> int:
        if self.grade_config is None:
            return hash(None)
        return hash(tuple(sorted((k, tuple(v)) for k, v in self.grade_config.items())))

    @property
    def is_default(self) -> bool:
        """Check if using default filtering."""
        return self.grade_config is None


@dataclass
class QualityFilterConfig:
    """Configuration for MS-DIAL quality filtering.

    Args:
        total_score_threshold: Minimum Total Score (0-100). 0 means no filtering.
        require_msms: Whether to require MS/MS matched = TRUE.
    """
    total_score_threshold: int = 0
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


@dataclass
class CleaningResult:
    """Result object containing cleaned data and metadata.

    Args:
        cleaned_df: Cleaned DataFrame without internal standards.
        internal_standards_df: DataFrame containing only internal standards.
        filter_messages: Optional list of filter messages (e.g., rows removed).
    """
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    internal_standards_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    filter_messages: List[str] = field(default_factory=list)
