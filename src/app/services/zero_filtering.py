"""
Zero filtering service for removing lipid species with too many zero/below-detection values.
"""
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

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
        return cls(detection_threshold=30000.0)

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

        # Evaluate each lipid
        to_keep = []
        for idx, row in df.iterrows():
            should_keep = ZeroFilteringService._evaluate_lipid(
                row=row,
                experiment=experiment,
                config=config,
                bqc_label=bqc_label if has_bqc else None
            )
            if should_keep:
                to_keep.append(idx)

        # Create filtered dataframe
        filtered_df = df.loc[to_keep].reset_index(drop=True)

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
    def _evaluate_lipid(
        row: pd.Series,
        experiment: ExperimentConfig,
        config: ZeroFilterConfig,
        bqc_label: Optional[str]
    ) -> bool:
        """
        Evaluate whether a single lipid should be kept.

        Args:
            row: DataFrame row containing lipid data.
            experiment: Experiment configuration.
            config: Zero filtering configuration.
            bqc_label: BQC condition label (already validated), or None.

        Returns:
            True if lipid should be kept, False if it should be removed.
        """
        # Initialize tracking variables
        # If BQC exists, default to fail (must pass to be kept)
        # If no BQC, default to pass (BQC check is skipped)
        bqc_fail = bqc_label is not None
        non_bqc_all_fail = True

        for cond_idx, cond_samples in enumerate(experiment.individual_samples_list):
            if not cond_samples:
                continue

            # Count zeros in this condition
            zero_count = ZeroFilteringService._count_zeros(
                row=row,
                samples=cond_samples,
                threshold=config.detection_threshold
            )

            n_samples = len(cond_samples)
            zero_proportion = zero_count / n_samples if n_samples > 0 else 1.0

            # Check if this is BQC condition
            condition_name = experiment.conditions_list[cond_idx]
            is_bqc = condition_name == bqc_label

            # Determine which threshold to use
            threshold = config.bqc_threshold if is_bqc else config.non_bqc_threshold

            # Check if condition passes (proportion below threshold)
            if zero_proportion < threshold:
                if is_bqc:
                    bqc_fail = False
                else:
                    non_bqc_all_fail = False

        # Keep lipid if: BQC passes AND at least one non-BQC passes
        return not bqc_fail and not non_bqc_all_fail

    @staticmethod
    def _count_zeros(
        row: pd.Series,
        samples: List[str],
        threshold: float
    ) -> int:
        """
        Count how many samples have values at or below the detection threshold.

        Args:
            row: DataFrame row containing intensity values.
            samples: List of sample names in the condition.
            threshold: Detection threshold (values <= this are counted).

        Returns:
            Count of zero/below-detection values.
        """
        zero_count = 0

        for sample in samples:
            col = f'intensity[{sample}]'
            if col in row.index:
                value = row[col]
                # Handle NaN as zero
                if pd.isna(value) or value <= threshold:
                    zero_count += 1

        return zero_count

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

        stats = []
        total_samples = sum(experiment.number_of_samples_list)

        for _, row in df.iterrows():
            lipid = row['LipidMolec']
            total_zeros = 0
            zeros_per_condition = {}

            for cond_idx, cond_samples in enumerate(experiment.individual_samples_list):
                condition_name = experiment.conditions_list[cond_idx]
                zero_count = ZeroFilteringService._count_zeros(
                    row=row,
                    samples=cond_samples,
                    threshold=config.detection_threshold
                )
                zeros_per_condition[condition_name] = zero_count
                total_zeros += zero_count

            zero_percentage = (total_zeros / total_samples * 100) if total_samples > 0 else 0

            stats.append({
                'LipidMolec': lipid,
                'total_zeros': total_zeros,
                'total_samples': total_samples,
                'zero_percentage': round(zero_percentage, 2),
                'zeros_per_condition': zeros_per_condition
            })

        return pd.DataFrame(stats)
