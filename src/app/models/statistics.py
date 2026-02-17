"""
Statistical testing configuration model.
"""
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class StatisticalTestConfig(BaseModel):
    """
    Configuration for statistical testing.

    Attributes:
        mode: Analysis mode ('auto' or 'manual')
        test_type: Type of statistical test to perform
        correction_method: Level 1 correction - between lipid classes
        posthoc_correction: Level 2 correction - within class for 3+ conditions
        alpha: Significance threshold (default 0.05)
        auto_transform: If True, applies log10 transformation to normalize data
        conditions_to_compare: List of condition pairs for pairwise comparison
    """
    mode: Literal['auto', 'manual'] = 'manual'
    test_type: Literal['parametric', 'non_parametric', 'auto'] = 'parametric'
    correction_method: Literal['uncorrected', 'fdr_bh', 'bonferroni', 'auto'] = 'fdr_bh'
    posthoc_correction: Literal['uncorrected', 'tukey', 'bonferroni', 'auto'] = 'tukey'
    alpha: float = Field(default=0.05, gt=0, lt=1)
    auto_transform: bool = True
    conditions_to_compare: List[Tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_auto_mode_settings(self):
        """In auto mode, test_type/correction_method/posthoc_correction should be 'auto'."""
        if self.mode == 'auto':
            if self.test_type != 'auto':
                raise ValueError(
                    "When mode is 'auto', test_type should be 'auto'"
                )
            if self.correction_method != 'auto':
                raise ValueError(
                    "When mode is 'auto', correction_method should be 'auto'"
                )
            if self.posthoc_correction != 'auto':
                raise ValueError(
                    "When mode is 'auto', posthoc_correction should be 'auto'"
                )
        return self

    @model_validator(mode='after')
    def validate_manual_mode_no_auto(self):
        """In manual mode, 'auto' is not a valid choice for test settings."""
        if self.mode == 'manual':
            if self.test_type == 'auto':
                raise ValueError(
                    "When mode is 'manual', test_type cannot be 'auto'. "
                    "Choose 'parametric' or 'non_parametric'"
                )
            if self.correction_method == 'auto':
                raise ValueError(
                    "When mode is 'manual', correction_method cannot be 'auto'. "
                    "Choose 'uncorrected', 'fdr_bh', or 'bonferroni'"
                )
            if self.posthoc_correction == 'auto':
                raise ValueError(
                    "When mode is 'manual', posthoc_correction cannot be 'auto'. "
                    "Choose 'uncorrected', 'tukey', or 'bonferroni'"
                )
        return self

    @field_validator('conditions_to_compare')
    @classmethod
    def validate_condition_pairs(cls, v):
        """Validate condition pairs are properly formatted."""
        if not v:
            return v

        for pair in v:
            if len(pair) != 2:
                raise ValueError(
                    f"Each comparison pair must have exactly 2 conditions, got {len(pair)}"
                )
            if pair[0] == pair[1]:
                raise ValueError(
                    f"Cannot compare a condition to itself: '{pair[0]}'"
                )
            if not pair[0] or not pair[1]:
                raise ValueError("Condition names cannot be empty")

        return v

    def is_auto_mode(self) -> bool:
        """Check if running in auto mode."""
        return self.mode == 'auto'

    def is_parametric(self) -> bool:
        """Check if using parametric tests."""
        return self.test_type == 'parametric'

    def is_non_parametric(self) -> bool:
        """Check if using non-parametric tests."""
        return self.test_type == 'non_parametric'

    def requires_posthoc(self, n_conditions: int) -> bool:
        """
        Check if post-hoc tests are needed (3+ conditions and correction enabled).

        Args:
            n_conditions: Number of conditions being compared

        Returns:
            True if post-hoc tests should be performed
        """
        return n_conditions > 2 and self.posthoc_correction != 'uncorrected'

    def get_correction_display_name(self) -> str:
        """Get human-readable name for the correction method."""
        names = {
            'uncorrected': 'Uncorrected',
            'fdr_bh': 'FDR (Benjamini-Hochberg)',
            'bonferroni': 'Bonferroni',
            'auto': 'Auto'
        }
        return names.get(self.correction_method, self.correction_method)

    def get_posthoc_display_name(self) -> str:
        """Get human-readable name for the post-hoc correction method."""
        names = {
            'uncorrected': 'Uncorrected',
            'tukey': "Tukey's HSD",
            'bonferroni': 'Bonferroni',
            'auto': 'Auto'
        }
        return names.get(self.posthoc_correction, self.posthoc_correction)

    @classmethod
    def create_auto(cls, auto_transform: bool = True) -> 'StatisticalTestConfig':
        """
        Factory method to create an auto-mode configuration.

        Args:
            auto_transform: Whether to apply log10 transformation

        Returns:
            StatisticalTestConfig configured for auto mode
        """
        return cls(
            mode='auto',
            test_type='auto',
            correction_method='auto',
            posthoc_correction='auto',
            auto_transform=auto_transform
        )

    @classmethod
    def create_manual(
        cls,
        test_type: Literal['parametric', 'non_parametric'] = 'parametric',
        correction_method: Literal['uncorrected', 'fdr_bh', 'bonferroni'] = 'fdr_bh',
        posthoc_correction: Literal['uncorrected', 'tukey', 'bonferroni'] = 'tukey',
        alpha: float = 0.05,
        auto_transform: bool = True,
        conditions_to_compare: Optional[List[Tuple[str, str]]] = None
    ) -> 'StatisticalTestConfig':
        """
        Factory method to create a manual-mode configuration.

        Args:
            test_type: 'parametric' or 'non_parametric'
            correction_method: Level 1 correction method
            posthoc_correction: Level 2 post-hoc correction method
            alpha: Significance threshold (default 0.05)
            auto_transform: Whether to apply log10 transformation
            conditions_to_compare: Optional list of condition pairs

        Returns:
            StatisticalTestConfig configured for manual mode
        """
        return cls(
            mode='manual',
            test_type=test_type,
            correction_method=correction_method,
            posthoc_correction=posthoc_correction,
            alpha=alpha,
            auto_transform=auto_transform,
            conditions_to_compare=conditions_to_compare or []
        )
