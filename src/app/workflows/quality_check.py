"""
Quality check workflow.

The Quality Check page's decisions: input validation, the BQC CoV filter,
retention time availability, which conditions can be correlated, and PCA
sample removal. The plots come from the plotter services through
StreamlitAdapter.

Unlike DataIngestionWorkflow/NormalizationWorkflow, this workflow does NOT
have a single run() method. Quality check is interactive — users make
decisions between steps (BQC filtering, PCA sample removal). Each method
is called individually from the UI layer.

All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..models.experiment import ExperimentConfig
from ..services.format_detection import DataFormat
from ..services.validation import get_matching_concentration_columns
from ..services.quality_check import (
    QualityCheckService,
    BQCFilterResult,
    RetentionTimeDataResult,
    SampleRemovalResult,
)


@dataclass(frozen=True)
class QualityCheckConfig:
    """Configuration for the quality check workflow.

    Attributes:
        format_type: Data format (affects retention time availability).
    """
    format_type: DataFormat = DataFormat.GENERIC


class QualityCheckWorkflow:
    """Orchestrates quality check steps for lipidomic data.

    Each public method corresponds to one section of the QC module.
    The UI layer calls them individually, allowing user interaction
    between steps (e.g., selecting lipids to keep after BQC, removing
    outlier samples after PCA).

    All methods are static — no instance state is stored.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_inputs(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Validate that data is ready for quality checks.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.

        Returns:
            List of validation error messages (empty if all valid).
        """
        errors = []

        if df is None or df.empty:
            errors.append(
                "Input DataFrame is empty. Cannot run quality checks."
            )
            return errors

        available = ', '.join(list(df.columns)[:15])

        if 'LipidMolec' not in df.columns:
            errors.append(f"Input DataFrame missing 'LipidMolec' column. Available columns: [{available}]")

        # Check for concentration columns
        conc_cols = [
            col for col in df.columns if col.startswith('concentration[')
        ]
        if not conc_cols:
            errors.append(
                "Input DataFrame has no concentration columns. "
                f"Expected 'concentration[s1]', etc. Available columns: [{available}]. "
                "Run normalization before quality checks."
            )
            return errors

        # Check that at least some concentration columns match the experiment
        matching = get_matching_concentration_columns(df, experiment)
        if not matching:
            expected = ', '.join([f"concentration[{s}]" for s in experiment.full_samples_list[:5]])
            found = ', '.join(conc_cols[:5])
            errors.append(
                f"No concentration columns match the experiment's sample labels. "
                f"Expected: [{expected}, ...]. Found: [{found}]"
            )

        if len(matching) < 2:
            errors.append(
                f"At least 2 samples are required for quality checks, "
                f"but only {len(matching)} matched: {matching}"
            )

        return errors

    # ------------------------------------------------------------------
    # BQC filter
    # ------------------------------------------------------------------

    @staticmethod
    def high_cov_lipids(prepared_df: pd.DataFrame, cov_threshold: float) -> List[str]:
        """Lipids whose BQC CoV is at or above the threshold (the ones the
        filter removes)."""
        return QualityCheckService.high_cov_lipids(prepared_df, cov_threshold)

    @staticmethod
    def apply_bqc_filter(
        df: pd.DataFrame,
        high_cov_lipids: List[str],
        lipids_to_keep: Optional[List[str]] = None,
    ) -> BQCFilterResult:
        """Apply BQC filter after user selects which high-CoV lipids to keep.

        Args:
            df: Original DataFrame (NOT the prepared_df with cov/mean columns).
            high_cov_lipids: Lipids with CoV >= threshold.
            lipids_to_keep: Lipids to retain despite high CoV.

        Returns:
            BQCFilterResult with filtered DataFrame and removal statistics.
        """
        return QualityCheckService.filter_by_bqc(
            df=df,
            high_cov_lipids=high_cov_lipids,
            lipids_to_keep=lipids_to_keep,
        )

    # ------------------------------------------------------------------
    # Retention Time
    # ------------------------------------------------------------------

    @staticmethod
    def check_retention_time_availability(
        df: pd.DataFrame,
        config: QualityCheckConfig,
    ) -> RetentionTimeDataResult:
        """Check if retention time data is available.

        Retention time plots are only relevant for LipidSearch and
        MS-DIAL formats that include BaseRt and CalcMass columns.

        Args:
            df: DataFrame to check.
            config: Quality check configuration with format_type.

        Returns:
            RetentionTimeDataResult with availability flag and classes.
        """
        if config.format_type not in (DataFormat.LIPIDSEARCH, DataFormat.MSDIAL):
            return RetentionTimeDataResult(available=False, lipid_classes=[])

        return QualityCheckService.check_retention_time_availability(df)

    # ------------------------------------------------------------------
    # Pairwise Correlation
    # ------------------------------------------------------------------

    @staticmethod
    def get_eligible_correlation_conditions(
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Get conditions eligible for pairwise correlation (>1 replicate).

        Args:
            experiment: Experiment configuration.

        Returns:
            List of condition labels with more than one replicate.
        """
        return QualityCheckService.get_correlation_eligible_conditions(
            experiment
        )

    # ------------------------------------------------------------------
    # PCA Sample Removal
    # ------------------------------------------------------------------

    @staticmethod
    def remove_samples(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        samples_to_remove: List[str],
    ) -> SampleRemovalResult:
        """Remove outlier samples identified from PCA.

        After PCA visualization, the user may select samples to remove.
        This updates both the DataFrame and ExperimentConfig.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Current experiment configuration.
            samples_to_remove: Sample labels to remove.

        Returns:
            SampleRemovalResult with updated df and experiment.

        Raises:
            ValueError: If removal would leave <2 samples.
        """
        return QualityCheckService.remove_samples(
            df=df,
            experiment=experiment,
            samples_to_remove=samples_to_remove,
        )
