"""
Zero filtering service for removing lipid species with too many zero/below-detection values.
"""
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from app.constants import LIPIDSEARCH_DETECTION_THRESHOLD
from ..models.experiment import ExperimentConfig


@dataclass
class ZeroFilterConfig:
    """
    Configuration for zero filtering thresholds.

    Attributes:
        detection_threshold: Values <= this are considered "zero" (default 0.0).
            For LipidSearch 5.0, a value of 30000 is recommended to account for noise floor.
        bqc_threshold: Proportion threshold for BQC condition (default 0.5 = 50%).
            If >= this proportion of BQC samples are zeros, lipid is flagged.
        non_bqc_threshold: Proportion threshold for non-BQC conditions (default 0.75 = 75%).
            If ALL non-BQC conditions have >= this proportion zeros, lipid is flagged.
    """
    detection_threshold: float = 0.0
    bqc_threshold: float = 0.5
    non_bqc_threshold: float = 0.75

    def __post_init__(self):
        """Validate thresholds are in valid ranges."""
        if self.detection_threshold < 0:
            raise ValueError("detection_threshold must be >= 0")
        if not 0 <= self.bqc_threshold <= 1:
            raise ValueError("bqc_threshold must be between 0 and 1")
        if not 0 <= self.non_bqc_threshold <= 1:
            raise ValueError("non_bqc_threshold must be between 0 and 1")

    @classmethod
    def for_lipidsearch(cls) -> "ZeroFilterConfig":
        """Create config optimized for LipidSearch 5.0 data (higher noise floor)."""
        return cls(detection_threshold=LIPIDSEARCH_DETECTION_THRESHOLD)

    @classmethod
    def strict(cls) -> "ZeroFilterConfig":
        """Create strict config (lower thresholds = keep more data)."""
        return cls(bqc_threshold=0.25, non_bqc_threshold=0.50)

    @classmethod
    def permissive(cls) -> "ZeroFilterConfig":
        """Create permissive config (higher thresholds = remove more data)."""
        return cls(bqc_threshold=0.75, non_bqc_threshold=0.90)


@dataclass
class ZeroFilteringResult:
    """
    Result of zero filtering operation.

    Attributes:
        filtered_df: DataFrame with lipids passing the filter.
        removed_species: List of LipidMolec names that were removed.
        species_before: Count of lipid species before filtering.
        species_after: Count of lipid species after filtering.
    """
    filtered_df: pd.DataFrame
    removed_species: List[str]
    species_before: int
    species_after: int

    @property
    def species_removed_count(self) -> int:
        """Number of species removed."""
        return self.species_before - self.species_after

    @property
    def removal_percentage(self) -> float:
        """Percentage of species removed."""
        if self.species_before == 0:
            return 0.0
        return (self.species_removed_count / self.species_before) * 100


class ZeroFilteringService:
    """
    Service for filtering lipid species based on zero/low intensity values.

    Filtering logic:
    - For each lipid species, count zero/below-detection values per condition
    - BQC condition (if present): Remove if >= bqc_threshold of samples are zeros
    - Non-BQC conditions: Remove if ALL conditions have >= non_bqc_threshold zeros

    A lipid is kept if:
    1. BQC passes (< threshold zeros) OR no BQC condition exists
    2. AND at least one non-BQC condition passes (< threshold zeros)

    All methods are stateless static methods.
    """

    @staticmethod
    def filter_zeros(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        bqc_label: Optional[str] = None,
        config: Optional[ZeroFilterConfig] = None
    ) -> ZeroFilteringResult:
        """
        Filter lipid species based on zero/low intensity values.

        Args:
            df: DataFrame with LipidMolec column and intensity columns.
                Intensity columns should be named 'intensity[sample_name]'.
            experiment: Experiment configuration with condition/sample information.
            bqc_label: Label for BQC (Batch Quality Control) condition, if present.
                If None or not found in conditions_list, BQC filtering is skipped.
            config: Zero filtering configuration. Uses defaults if None.

        Returns:
            ZeroFilteringResult containing filtered_df and removed species list.

        Raises:
            ValueError: If DataFrame is missing required columns.
        """
        if config is None:
            config = ZeroFilterConfig()

        # Validate input
        ZeroFilteringService._validate_input(df, experiment)

        if df.empty:
            return ZeroFilteringResult(
                filtered_df=df.copy(),
                removed_species=[],
                species_before=0,
                species_after=0
            )

        # Get all species before filtering
        all_species = df['LipidMolec'].tolist()
        species_before = len(all_species)

        # Validate BQC label
        has_bqc = (
            bqc_label is not None and
            bqc_label in experiment.conditions_list
        )

        # Vectorized evaluation of all lipids
        keep_mask = ZeroFilteringService._evaluate_all_lipids(
            df=df,
            experiment=experiment,
            config=config,
            bqc_label=bqc_label if has_bqc else None
        )

        # Create filtered dataframe
        filtered_df = df[keep_mask].reset_index(drop=True)

        # Compute removed species
        kept_species = set(filtered_df['LipidMolec'].tolist())
        removed_species = [s for s in all_species if s not in kept_species]

        return ZeroFilteringResult(
            filtered_df=filtered_df,
            removed_species=removed_species,
            species_before=species_before,
            species_after=len(filtered_df)
        )

    @staticmethod
    def _validate_input(df: pd.DataFrame, experiment: ExperimentConfig) -> None:
        """
        Validate that input DataFrame has required structure.

        Raises:
            ValueError: If DataFrame is missing LipidMolec column.
        """
        if 'LipidMolec' not in df.columns:
            raise ValueError(
                "DataFrame must contain 'LipidMolec' column. "
                "Ensure data has been cleaned and standardized first."
            )

    @staticmethod
    def _evaluate_all_lipids(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        config: ZeroFilterConfig,
        bqc_label: Optional[str]
    ) -> pd.Series:
        """
        Vectorized evaluation of which lipids to keep.

        Args:
            df: DataFrame with intensity columns.
            experiment: Experiment configuration.
            config: Zero filtering configuration.
            bqc_label: BQC condition label (already validated), or None.

        Returns:
            Boolean Series — True for lipids to keep.
        """
        n_rows = len(df)
        # If BQC exists, default to fail (must pass); if no BQC, default to pass
        bqc_pass = pd.Series(bqc_label is None, index=df.index)
        non_bqc_any_pass = pd.Series(False, index=df.index)

        for cond_idx, cond_samples in enumerate(experiment.individual_samples_list):
            if not cond_samples:
                continue

            condition_name = experiment.conditions_list[cond_idx]
            is_bqc = condition_name == bqc_label
            threshold_pct = config.bqc_threshold if is_bqc else config.non_bqc_threshold

            # Vectorized zero counting for this condition
            zero_proportion = ZeroFilteringService._condition_zero_proportion(
                df, cond_samples, config.detection_threshold
            )

            passes = zero_proportion < threshold_pct

            if is_bqc:
                bqc_pass = passes
            else:
                non_bqc_any_pass = non_bqc_any_pass | passes

        return bqc_pass & non_bqc_any_pass

    @staticmethod
    def _condition_zero_proportion(
        df: pd.DataFrame,
        samples: List[str],
        detection_threshold: float
    ) -> pd.Series:
        """
        Compute per-row zero proportion for a set of samples (vectorized).

        Treats NaN and values <= detection_threshold as zeros.

        Args:
            df: DataFrame with intensity columns.
            samples: List of sample names in the condition.
            detection_threshold: Values <= this are counted as zero.

        Returns:
            Series of zero proportions (0.0 to 1.0) per row.
        """
        cols = [f'intensity[{s}]' for s in samples if f'intensity[{s}]' in df.columns]
        if not cols:
            # No columns found means no zeros can be counted → 0% zeros → passes
            return pd.Series(0.0, index=df.index)

        subset = df[cols]
        zero_counts = (subset.isna() | (subset <= detection_threshold)).sum(axis=1)
        return zero_counts / len(cols)

    @staticmethod
    def get_zero_statistics(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        config: Optional[ZeroFilterConfig] = None
    ) -> pd.DataFrame:
        """
        Get zero value statistics for each lipid species.

        Useful for understanding the data before applying filtering.

        Args:
            df: DataFrame with LipidMolec and intensity columns.
            experiment: Experiment configuration.
            config: Zero filtering config (for detection_threshold). Uses defaults if None.

        Returns:
            DataFrame with columns:
            - LipidMolec: Lipid species name
            - total_zeros: Total count of zeros across all samples
            - total_samples: Total number of samples
            - zero_percentage: Percentage of zeros overall
            - zeros_per_condition: Dict of {condition: zero_count}
        """
        if config is None:
            config = ZeroFilterConfig()

        ZeroFilteringService._validate_input(df, experiment)

        total_samples = sum(experiment.number_of_samples_list)

        # Compute zero counts per condition (vectorized)
        condition_zero_counts = {}
        total_zeros = pd.Series(0, index=df.index)

        for cond_idx, cond_samples in enumerate(experiment.individual_samples_list):
            condition_name = experiment.conditions_list[cond_idx]
            cols = [f'intensity[{s}]' for s in cond_samples if f'intensity[{s}]' in df.columns]
            if cols:
                subset = df[cols]
                cond_zeros = (subset.isna() | (subset <= config.detection_threshold)).sum(axis=1)
            else:
                cond_zeros = pd.Series(0, index=df.index)
            condition_zero_counts[condition_name] = cond_zeros
            total_zeros = total_zeros + cond_zeros

        zero_percentage = (total_zeros / total_samples * 100).round(2) if total_samples > 0 else 0

        stats_df = pd.DataFrame({
            'LipidMolec': df['LipidMolec'].values,
            'total_zeros': total_zeros.values,
            'total_samples': total_samples,
            'zero_percentage': zero_percentage if isinstance(zero_percentage, pd.Series) else 0,
            'zeros_per_condition': [
                {cond: int(condition_zero_counts[cond].iloc[i])
                 for cond in experiment.conditions_list}
                for i in range(len(df))
            ]
        })

        return stats_df
