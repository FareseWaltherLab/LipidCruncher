"""
Quality check workflow.

Orchestrates the quality check pipeline steps:
Box plots → BQC assessment → Retention time → Correlation → PCA

Unlike DataIngestionWorkflow/NormalizationWorkflow, this workflow does NOT
have a single run() method. Quality check is interactive — users make
decisions between steps (BQC filtering, PCA sample removal). Each method
is called individually from the UI layer.

All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.constants import COV_THRESHOLD_DEFAULT
from ..models.experiment import ExperimentConfig
from ..services.format_detection import DataFormat
from ..services.validation import get_matching_concentration_columns
from ..services.quality_check import (
    QualityCheckService,
    BoxPlotResult,
    BQCPrepareResult,
    BQCFilterResult,
    RetentionTimeDataResult,
    CorrelationResult,
    PCAResult,
    SampleRemovalResult,
)


@dataclass
class QualityCheckConfig:
    """Configuration for the quality check workflow.

    Attributes:
        bqc_label: Label of the BQC condition, or None if no BQC.
        format_type: Data format (affects retention time availability).
        cov_threshold: CoV threshold for BQC assessment (default 30%).
    """
    bqc_label: Optional[str] = None
    format_type: DataFormat = DataFormat.GENERIC
    cov_threshold: float = float(COV_THRESHOLD_DEFAULT)


@dataclass
class QualityCheckNonInteractiveResult:
    """Result of running all QC steps without user interaction.

    Attributes:
        box_plot: BoxPlotResult or None.
        bqc: BQCPrepareResult or None (None if no BQC samples).
        retention_time: RetentionTimeDataResult or None.
        correlations: Dict mapping condition name to CorrelationResult.
        pca: PCAResult or None.
        validation_errors: List of validation error strings.
    """
    box_plot: Optional[BoxPlotResult] = None
    bqc: Optional[BQCPrepareResult] = None
    retention_time: Optional[RetentionTimeDataResult] = None
    correlations: Dict[str, CorrelationResult] = dataclass_field(default_factory=dict)
    pca: Optional[PCAResult] = None
    validation_errors: List[str] = dataclass_field(default_factory=list)


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

        if 'LipidMolec' not in df.columns:
            errors.append("Input DataFrame missing 'LipidMolec' column.")

        # Check for concentration columns
        conc_cols = [
            col for col in df.columns if col.startswith('concentration[')
        ]
        if not conc_cols:
            errors.append(
                "Input DataFrame has no concentration columns. "
                "Run normalization before quality checks."
            )
            return errors

        # Check that at least some concentration columns match the experiment
        matching = get_matching_concentration_columns(df, experiment)
        if not matching:
            errors.append(
                "No concentration columns match the experiment's sample labels. "
                "Expected columns like 'concentration[s1]', 'concentration[s2]', etc."
            )

        if len(matching) < 2:
            errors.append(
                f"At least 2 samples are required for quality checks, "
                f"but only {len(matching)} matched."
            )

        return errors

    # ------------------------------------------------------------------
    # Section 1: Box Plots
    # ------------------------------------------------------------------

    @staticmethod
    def run_box_plots(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> BoxPlotResult:
        """Prepare data for box plots (missing values + concentration distributions).

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.

        Returns:
            BoxPlotResult with mean_area_df and missing_values_percent.

        Raises:
            ValueError: If inputs are invalid.
        """
        return QualityCheckService.prepare_box_plot_data(df, experiment)

    # ------------------------------------------------------------------
    # Section 2: BQC Quality Assessment
    # ------------------------------------------------------------------

    @staticmethod
    def run_bqc_assessment(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        config: QualityCheckConfig,
    ) -> Optional[BQCPrepareResult]:
        """Run BQC quality assessment if BQC samples exist.

        Returns None if no BQC label is configured or BQC label is not
        found in the experiment's conditions.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.
            config: Quality check configuration with bqc_label and cov_threshold.

        Returns:
            BQCPrepareResult with CoV data and high-CoV lipids, or None.
        """
        if config.bqc_label is None:
            return None

        if config.bqc_label not in experiment.conditions_list:
            return None

        return QualityCheckService.prepare_bqc_data(
            df=df,
            experiment=experiment,
            bqc_label=config.bqc_label,
            cov_threshold=config.cov_threshold,
        )

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
    # Section 3: Retention Time
    # ------------------------------------------------------------------

    @staticmethod
    def check_retention_time_availability(
        df: pd.DataFrame,
        config: QualityCheckConfig,
    ) -> RetentionTimeDataResult:
        """Check if retention time data is available.

        Retention time plots are only relevant for LipidSearch 5.0 and
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
    # Section 4: Pairwise Correlation
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

    @staticmethod
    def run_correlation(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        condition: str,
        bqc_label: Optional[str] = None,
    ) -> CorrelationResult:
        """Compute pairwise correlation for a single condition.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.
            condition: Condition to analyze.
            bqc_label: If set, uses technical replicate thresholds.

        Returns:
            CorrelationResult with correlation matrix and heatmap params.

        Raises:
            ValueError: If condition not found or has <2 replicates.
        """
        return QualityCheckService.compute_correlation(
            df=df,
            experiment=experiment,
            condition=condition,
            bqc_label=bqc_label,
        )

    @staticmethod
    def run_all_correlations(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        bqc_label: Optional[str] = None,
    ) -> Dict[str, CorrelationResult]:
        """Compute correlations for all eligible conditions at once.

        Convenience method that runs correlation for every condition
        with >1 replicate.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.
            bqc_label: If set, uses technical replicate thresholds.

        Returns:
            Dictionary mapping condition label to CorrelationResult.
        """
        eligible = QualityCheckService.get_correlation_eligible_conditions(
            experiment
        )
        results = {}
        for condition in eligible:
            results[condition] = QualityCheckService.compute_correlation(
                df=df,
                experiment=experiment,
                condition=condition,
                bqc_label=bqc_label,
            )
        return results

    # ------------------------------------------------------------------
    # Section 5: PCA Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def run_pca(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> PCAResult:
        """Run PCA analysis on concentration data.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.

        Returns:
            PCAResult with PC scores, labels, and condition mapping.

        Raises:
            ValueError: If fewer than 2 samples available.
        """
        return QualityCheckService.compute_pca(df, experiment)

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

    # ------------------------------------------------------------------
    # Full Pipeline (non-interactive)
    # ------------------------------------------------------------------

    @staticmethod
    def run_non_interactive(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        config: QualityCheckConfig,
    ) -> QualityCheckNonInteractiveResult:
        """Run all QC steps without user interaction.

        Useful for integration tests and batch processing. Runs each
        step that is applicable and returns all results.

        Does NOT apply BQC filtering or sample removal — those require
        user decisions.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.
            config: Quality check configuration.

        Returns:
            QualityCheckNonInteractiveResult with all QC step results.
        """
        result = QualityCheckNonInteractiveResult()

        # Validate
        errors = QualityCheckWorkflow.validate_inputs(df, experiment)
        if errors:
            result.validation_errors = errors
            return result

        # Box plots
        result.box_plot = QualityCheckWorkflow.run_box_plots(df, experiment)

        # BQC assessment
        result.bqc = QualityCheckWorkflow.run_bqc_assessment(
            df, experiment, config
        )

        # Retention time
        result.retention_time = (
            QualityCheckWorkflow.check_retention_time_availability(df, config)
        )

        # Correlations
        result.correlations = QualityCheckWorkflow.run_all_correlations(
            df, experiment, bqc_label=config.bqc_label
        )

        # PCA
        result.pca = QualityCheckWorkflow.run_pca(df, experiment)

        return result