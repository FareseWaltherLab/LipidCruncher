"""
Quality check service for lipidomic data.

Pure business logic - no Streamlit dependencies.
Handles box plot data preparation, BQC quality assessment, retention time
availability, pairwise correlation, and PCA analysis.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA

from ..models.experiment import ExperimentConfig
from ..services.validation import (
    validate_dataframe_not_empty,
    validate_concentration_columns,
)


# ============================================================
# Result Dataclasses
# ============================================================

@dataclass
class BoxPlotResult:
    """Result of box plot data preparation.

    Attributes:
        mean_area_df: DataFrame with concentration columns only (one per sample).
        missing_values_percent: Percentage of zero values per sample
            (parallel to available_samples).
        available_samples: Sample labels whose concentration columns exist in df.
    """
    mean_area_df: pd.DataFrame
    missing_values_percent: List[float]
    available_samples: List[str]


@dataclass
class BQCPrepareResult:
    """Result of BQC data preparation (before user filtering decision).

    Attributes:
        prepared_df: DataFrame with 'cov' and 'mean' (log10) columns appended.
        bqc_sample_index: Index of the BQC condition in conditions_list.
        bqc_samples: List of BQC sample labels.
        reliable_data_percent: Percentage of lipids with CoV below threshold.
        high_cov_lipids: List of LipidMolec names with CoV >= threshold.
        high_cov_details: DataFrame of high-CoV lipids with columns
            [LipidMolec, ClassKey, cov, mean].
    """
    prepared_df: pd.DataFrame
    bqc_sample_index: int
    bqc_samples: List[str]
    reliable_data_percent: float
    high_cov_lipids: List[str]
    high_cov_details: pd.DataFrame


@dataclass
class BQCFilterResult:
    """Result of BQC filtering (after user selects which lipids to keep).

    Attributes:
        filtered_df: DataFrame with high-CoV lipids removed (except those
            the user chose to keep), sorted by ClassKey, index reset.
        removed_lipids: List of LipidMolec names that were removed.
        kept_despite_high_cov: List of LipidMolec names kept despite high CoV.
        lipids_before: Count of lipid species before filtering.
        lipids_after: Count of lipid species after filtering.
    """
    filtered_df: pd.DataFrame
    removed_lipids: List[str]
    kept_despite_high_cov: List[str]
    lipids_before: int
    lipids_after: int

    @property
    def removed_count(self) -> int:
        """Number of lipids removed."""
        return self.lipids_before - self.lipids_after

    @property
    def removed_percentage(self) -> float:
        """Percentage of lipids removed."""
        if self.lipids_before == 0:
            return 0.0
        return (self.removed_count / self.lipids_before) * 100


@dataclass
class RetentionTimeDataResult:
    """Result of retention time data availability check.

    Attributes:
        available: Whether retention time data is available
            (BaseRt and CalcMass columns present).
        lipid_classes: List of available lipid classes (sorted by frequency).
    """
    available: bool
    lipid_classes: List[str]


@dataclass
class CorrelationResult:
    """Result of pairwise correlation analysis for a single condition.

    Attributes:
        condition: The condition label analyzed.
        correlation_df: Pearson correlation matrix (samples x samples).
        v_min: Minimum value for heatmap color scale (always 0.5).
        threshold: Threshold value for the heatmap (0.7 biological, 0.8 technical).
        sample_type: 'biological replicates' or 'technical replicates'.
        condition_samples: List of sample labels in this condition.
    """
    condition: str
    correlation_df: pd.DataFrame
    v_min: float
    threshold: float
    sample_type: str
    condition_samples: List[str]


@dataclass
class PCAResult:
    """Result of PCA computation.

    Attributes:
        pc_df: DataFrame with PC1, PC2 columns (one row per sample).
        pc_labels: Axis labels with variance explained,
            e.g. ['PC1 (45%)', 'PC2 (30%)'].
        available_samples: Sample labels included in PCA.
        conditions: Condition labels per sample (parallel to available_samples).
    """
    pc_df: pd.DataFrame
    pc_labels: List[str]
    available_samples: List[str]
    conditions: List[str]


@dataclass
class SampleRemovalResult:
    """Result of removing samples from DataFrame and experiment.

    Attributes:
        updated_df: DataFrame with removed sample columns dropped and
            remaining columns renamed to match new experiment labels.
        updated_experiment: New ExperimentConfig without removed samples.
        removed_samples: Sample labels that were actually removed.
        samples_before: Total sample count before removal.
        samples_after: Total sample count after removal.
    """
    updated_df: pd.DataFrame
    updated_experiment: ExperimentConfig
    removed_samples: List[str]
    samples_before: int
    samples_after: int


# ============================================================
# Service
# ============================================================

class QualityCheckService:
    """Service for quality check computations on lipidomic data.

    All methods are static - no instance state required.
    Reimplements legacy module computations (BoxPlot, BQCQualityCheck,
    Correlation, PCAAnalysis) to avoid transitive Streamlit imports.
    Plot rendering stays in legacy modules, called from UI layer.
    """

    # ============================================================
    # Section 1: Box Plot Data
    # ============================================================

    @staticmethod
    def prepare_box_plot_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> BoxPlotResult:
        """Prepare data for box plots: extract concentration columns and
        calculate missing values percentages.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration with sample information.

        Returns:
            BoxPlotResult with mean_area_df and missing_values_percent.

        Raises:
            ValueError: If DataFrame is empty or has no concentration columns.
        """
        QualityCheckService._validate_dataframe(df)
        available = QualityCheckService._validate_concentration_columns(
            df, experiment
        )

        conc_cols = [f'concentration[{s}]' for s in available]
        mean_area_df = df[conc_cols].copy()

        missing_pct = []
        n_rows = len(mean_area_df)
        for col in conc_cols:
            if n_rows == 0:
                missing_pct.append(0.0)
            else:
                zero_count = (mean_area_df[col] == 0).sum()
                missing_pct.append(zero_count / n_rows * 100)

        return BoxPlotResult(
            mean_area_df=mean_area_df,
            missing_values_percent=missing_pct,
            available_samples=available,
        )

    # ============================================================
    # Section 2: BQC Quality Assessment
    # ============================================================

    @staticmethod
    def calculate_coefficient_of_variation(numbers: "np.typing.ArrayLike") -> Optional[float]:
        """Calculate coefficient of variation (CoV) for an array of numbers.

        Includes zero values. Uses sample standard deviation (ddof=1).

        Args:
            numbers: Array-like of numerical values.

        Returns:
            CoV as a percentage, or None if <2 values or mean is zero.
        """
        arr = np.array(numbers, dtype=float)
        if len(arr) < 2:
            return None
        mean_val = np.mean(arr)
        if mean_val == 0:
            return None
        return float(np.std(arr, ddof=1) / mean_val * 100)

    @staticmethod
    def calculate_mean_including_zeros(numbers: "np.typing.ArrayLike") -> Optional[float]:
        """Calculate mean of numbers, including zeros.

        Args:
            numbers: Array-like of numerical values.

        Returns:
            Mean value, or None if fewer than 2 values.
        """
        arr = np.array(numbers, dtype=float)
        if len(arr) < 2:
            return None
        return float(np.mean(arr))

    @staticmethod
    def prepare_bqc_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        bqc_label: str,
        cov_threshold: float = 30.0,
    ) -> BQCPrepareResult:
        """Prepare BQC quality assessment data.

        Identifies BQC samples, calculates CoV and mean per lipid,
        determines reliability percentage.
        """
        QualityCheckService._validate_dataframe(df)
        QualityCheckService._validate_cov_threshold(cov_threshold)
        bqc_index = QualityCheckService._validate_bqc_label(experiment, bqc_label)

        bqc_samples = experiment.individual_samples_list[bqc_index]
        auc_columns = [f'concentration[{s}]' for s in bqc_samples]

        missing = [c for c in auc_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"BQC concentration columns not found in DataFrame: {missing}"
            )

        work_df = QualityCheckService._compute_cov_and_mean(df, auc_columns)
        reliable_pct, high_cov_lipids, high_cov_details = (
            QualityCheckService._identify_high_cov_lipids(work_df, cov_threshold)
        )

        return BQCPrepareResult(
            prepared_df=work_df,
            bqc_sample_index=bqc_index,
            bqc_samples=bqc_samples,
            reliable_data_percent=reliable_pct,
            high_cov_lipids=high_cov_lipids,
            high_cov_details=high_cov_details,
        )

    @staticmethod
    def _compute_cov_and_mean(df: pd.DataFrame, auc_columns: List[str]) -> pd.DataFrame:
        """Calculate CoV and log10 mean for BQC samples."""
        work_df = df.copy()
        work_df['cov'] = work_df[auc_columns].apply(
            QualityCheckService.calculate_coefficient_of_variation, axis=1
        )
        work_df['mean'] = work_df[auc_columns].apply(
            QualityCheckService.calculate_mean_including_zeros, axis=1
        )
        valid_mask = work_df['mean'].notnull() & (work_df['mean'] > 0)
        work_df.loc[valid_mask, 'mean'] = np.log10(work_df.loc[valid_mask, 'mean'])
        return work_df

    @staticmethod
    def _identify_high_cov_lipids(
        work_df: pd.DataFrame, cov_threshold: float
    ) -> tuple:
        """Identify lipids above CoV threshold. Returns (reliable_pct, lipid_list, details_df)."""
        valid_cov = work_df['cov'].dropna()
        if len(valid_cov) == 0:
            reliable_pct = 0.0
        else:
            reliable_pct = round(
                (valid_cov < cov_threshold).sum() / len(work_df) * 100, 1
            )

        high_cov_mask = work_df['cov'].notnull() & (work_df['cov'] >= cov_threshold)
        high_cov_lipids = work_df.loc[high_cov_mask, 'LipidMolec'].tolist()

        detail_cols = ['LipidMolec', 'ClassKey', 'cov', 'mean']
        available_detail_cols = [c for c in detail_cols if c in work_df.columns]
        high_cov_details = work_df.loc[high_cov_mask, available_detail_cols].copy()

        return reliable_pct, high_cov_lipids, high_cov_details

    @staticmethod
    def filter_by_bqc(
        df: pd.DataFrame,
        high_cov_lipids: List[str],
        lipids_to_keep: Optional[List[str]] = None,
    ) -> BQCFilterResult:
        """Filter lipids based on BQC CoV analysis.

        Removes high-CoV lipids from the dataset, except those the user
        chose to retain.

        Args:
            df: Original DataFrame to filter (the full dataset, not prepared_df).
            high_cov_lipids: List of LipidMolec names with CoV >= threshold
                (from BQCPrepareResult.high_cov_lipids).
            lipids_to_keep: LipidMolec names to keep despite high CoV.
                If None or empty, all high-CoV lipids are removed.

        Returns:
            BQCFilterResult with filtered DataFrame and removal statistics.
        """
        if lipids_to_keep is None:
            lipids_to_keep = []

        lipids_before = len(df)

        lipids_to_remove = [
            lipid for lipid in high_cov_lipids
            if lipid not in lipids_to_keep
        ]

        if lipids_to_remove:
            filtered_df = df[
                ~df['LipidMolec'].isin(set(lipids_to_remove))
            ].copy()
        else:
            filtered_df = df.copy()

        if 'ClassKey' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(
                by='ClassKey'
            ).reset_index(drop=True)
        else:
            filtered_df = filtered_df.reset_index(drop=True)

        return BQCFilterResult(
            filtered_df=filtered_df,
            removed_lipids=lipids_to_remove,
            kept_despite_high_cov=lipids_to_keep,
            lipids_before=lipids_before,
            lipids_after=len(filtered_df),
        )

    # ============================================================
    # Section 3: Retention Time Data
    # ============================================================

    @staticmethod
    def check_retention_time_availability(
        df: pd.DataFrame,
    ) -> RetentionTimeDataResult:
        """Check whether the DataFrame has retention time and mass data.

        The service does NOT render retention time plots. The UI layer calls
        legacy RetentionTime module directly for plotting.

        Args:
            df: DataFrame to check for BaseRt and CalcMass columns.

        Returns:
            RetentionTimeDataResult indicating availability and listing classes.
        """
        has_rt = 'BaseRt' in df.columns and 'CalcMass' in df.columns
        classes: List[str] = []
        if has_rt and 'ClassKey' in df.columns:
            classes = df['ClassKey'].value_counts().index.tolist()

        return RetentionTimeDataResult(available=has_rt, lipid_classes=classes)

    # ============================================================
    # Section 4: Pairwise Correlation
    # ============================================================

    @staticmethod
    def get_correlation_eligible_conditions(
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Get conditions eligible for pairwise correlation (>1 replicate).

        Args:
            experiment: Experiment configuration.

        Returns:
            List of condition labels with more than one replicate.
        """
        return [
            cond for cond, n in zip(
                experiment.conditions_list,
                experiment.number_of_samples_list,
            )
            if n > 1
        ]

    @staticmethod
    def compute_correlation(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        condition: str,
        bqc_label: Optional[str] = None,
    ) -> CorrelationResult:
        """Compute pairwise Pearson correlation for samples within a condition.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.
            condition: Condition label to analyze.
            bqc_label: If not None, uses 'technical replicates' thresholds;
                otherwise 'biological replicates'.

        Returns:
            CorrelationResult with correlation matrix and heatmap parameters.

        Raises:
            ValueError: If condition not found or has fewer than 2 replicates.
        """
        QualityCheckService._validate_dataframe(df)
        cond_index = QualityCheckService._validate_condition(
            experiment, condition
        )

        n_samples = experiment.number_of_samples_list[cond_index]
        if n_samples < 2:
            raise ValueError(
                f"Condition '{condition}' has only {n_samples} sample(s). "
                f"Correlation requires at least 2 replicates."
            )

        sample_type = (
            'technical replicates' if bqc_label is not None
            else 'biological replicates'
        )

        cond_samples = experiment.individual_samples_list[cond_index]

        conc_cols = [f'concentration[{s}]' for s in cond_samples]
        missing = [c for c in conc_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing concentration columns for condition '{condition}': "
                f"{missing}"
            )

        mean_area_df = df[conc_cols].copy()
        mean_area_df.columns = cond_samples

        v_min = 0.5
        threshold = 0.7 if sample_type == 'biological replicates' else 0.8
        correlation_df = mean_area_df.corr()

        return CorrelationResult(
            condition=condition,
            correlation_df=correlation_df,
            v_min=v_min,
            threshold=threshold,
            sample_type=sample_type,
            condition_samples=cond_samples,
        )

    # ============================================================
    # Section 5: PCA Analysis
    # ============================================================

    @staticmethod
    def compute_pca(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> PCAResult:
        """Run PCA with 2 components on concentration data.

        Standardizes data (z-score) then applies PCA. Maps samples to their
        experimental conditions.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.

        Returns:
            PCAResult with PC scores, labels, samples, and conditions.

        Raises:
            ValueError: If fewer than 2 samples have concentration columns.
        """
        QualityCheckService._validate_dataframe(df)
        available = QualityCheckService._validate_concentration_columns(
            df, experiment
        )

        if len(available) < 2:
            raise ValueError(
                f"PCA requires at least 2 samples, but only {len(available)} "
                f"sample(s) have concentration columns."
            )

        conc_cols = [f'concentration[{s}]' for s in available]
        data_matrix = df[conc_cols].T  # (samples x lipids)

        scaled_data = StandardScaler().fit_transform(data_matrix)
        pca = SklearnPCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_

        pc_df = pd.DataFrame(
            data=principal_components,
            columns=['PC1', 'PC2'],
        )
        pc_labels = [
            f'PC{i + 1} ({var:.0%})'
            for i, var in enumerate(explained_variance)
        ]

        conditions = []
        for sample in available:
            idx = experiment.full_samples_list.index(sample)
            conditions.append(experiment.extensive_conditions_list[idx])

        return PCAResult(
            pc_df=pc_df,
            pc_labels=pc_labels,
            available_samples=available,
            conditions=conditions,
        )

    @staticmethod
    def remove_samples(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        samples_to_remove: List[str],
    ) -> SampleRemovalResult:
        """Remove samples from DataFrame and create updated ExperimentConfig."""
        if not samples_to_remove:
            raise ValueError(
                "samples_to_remove cannot be empty. "
                "Pass a non-empty list of sample labels."
            )

        actual_to_remove = [
            s for s in samples_to_remove if s in experiment.full_samples_list
        ]
        total_samples = len(experiment.full_samples_list)
        remaining = total_samples - len(actual_to_remove)

        if remaining < 2:
            raise ValueError(
                f"Cannot remove {len(actual_to_remove)} sample(s): only "
                f"{remaining} would remain, but at least 2 are required."
            )

        new_experiment = experiment.without_samples(actual_to_remove)
        updated_df = QualityCheckService._drop_and_rename_columns(
            df, experiment, new_experiment, actual_to_remove
        )

        return SampleRemovalResult(
            updated_df=updated_df,
            updated_experiment=new_experiment,
            removed_samples=actual_to_remove,
            samples_before=total_samples,
            samples_after=len(new_experiment.full_samples_list),
        )

    @staticmethod
    def _drop_and_rename_columns(
        df: pd.DataFrame,
        old_experiment: ExperimentConfig,
        new_experiment: ExperimentConfig,
        removed: List[str],
    ) -> pd.DataFrame:
        """Drop removed sample columns and rename survivors to new labels."""
        updated_df = df.copy()
        for sample in removed:
            col = f'concentration[{sample}]'
            if col in updated_df.columns:
                updated_df = updated_df.drop(columns=[col])

        old_surviving = [s for s in old_experiment.full_samples_list if s not in removed]
        rename_map = {
            f'concentration[{old}]': f'concentration[{new}]'
            for old, new in zip(old_surviving, new_experiment.full_samples_list)
            if old != new and f'concentration[{old}]' in updated_df.columns
        }
        if rename_map:
            updated_df = updated_df.rename(columns=rename_map)
        return updated_df

    # ============================================================
    # Private Validation Methods
    # ============================================================

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Validate that the DataFrame is not None or empty."""
        validate_dataframe_not_empty(df)

    @staticmethod
    def _validate_concentration_columns(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Validate that concentration columns exist and return available samples."""
        return validate_concentration_columns(df, experiment)

    @staticmethod
    def _validate_bqc_label(
        experiment: ExperimentConfig,
        bqc_label: str,
    ) -> int:
        """Validate BQC label exists in conditions_list, return its index.

        Raises:
            ValueError: If bqc_label not found.
        """
        if bqc_label not in experiment.conditions_list:
            raise ValueError(
                f"BQC label '{bqc_label}' not found in conditions: "
                f"{experiment.conditions_list}"
            )
        return experiment.conditions_list.index(bqc_label)

    @staticmethod
    def _validate_condition(
        experiment: ExperimentConfig,
        condition: str,
    ) -> int:
        """Validate condition exists in conditions_list, return its index.

        Raises:
            ValueError: If condition not found.
        """
        if condition not in experiment.conditions_list:
            raise ValueError(
                f"Condition '{condition}' not found in conditions: "
                f"{experiment.conditions_list}"
            )
        return experiment.conditions_list.index(condition)

    @staticmethod
    def _validate_cov_threshold(threshold: float) -> None:
        """Validate that CoV threshold is positive.

        Raises:
            ValueError: If threshold is not positive.
        """
        if threshold <= 0:
            raise ValueError(
                f"CoV threshold must be positive, got {threshold}."
            )
