"""
Quality check service for lipidomic data.

Pure business logic - no Streamlit dependencies.
Handles the Quality Check decisions that change the data: the BQC CoV
filter and PCA sample removal, plus retention time availability and which
conditions can be correlated. The plots themselves come from the plotter
services under ``services/plotting``.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..models.experiment import ExperimentConfig
from .exceptions import ValidationError


# ============================================================
# Result Dataclasses
# ============================================================

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
class SampleRemovalResult:
    """Result of removing samples from DataFrame and experiment.

    Attributes:
        updated_df: DataFrame with removed sample columns dropped and
            remaining columns renamed to match new experiment labels.
        updated_experiment: New ExperimentConfig without removed samples.
        removed_samples: Sample labels that were actually removed.
        samples_before: Total sample count before removal.
        samples_after: Total sample count after removal.
        old_to_new: Mapping of ``concentration[s_old] -> concentration[s_new]``
            for every surviving sample (identity entries included).
    """
    updated_df: pd.DataFrame
    updated_experiment: ExperimentConfig
    removed_samples: List[str]
    samples_before: int
    samples_after: int
    old_to_new: Dict[str, str]


# ============================================================
# Service
# ============================================================

class QualityCheckService:
    """Service for quality check computations on lipidomic data.

    All methods are static - no instance state required.
    """

    # ============================================================
    # BQC Filter
    # ============================================================

    @staticmethod
    def high_cov_lipids(prepared_df: pd.DataFrame, cov_threshold: float) -> List[str]:
        """Lipids whose BQC CoV is at or above the threshold.

        These are the lipids the BQC filter removes; the CoV scatter plot
        draws the same ones red. A lipid without a CoV (NaN) is not one.

        Args:
            prepared_df: DataFrame with 'LipidMolec' and 'cov' columns (from
                the BQC CoV plot).
            cov_threshold: CoV threshold in percent.
        """
        return prepared_df.loc[prepared_df['cov'] >= cov_threshold, 'LipidMolec'].tolist()

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
                (from high_cov_lipids).
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
    # Retention Time Data
    # ============================================================

    @staticmethod
    def check_retention_time_availability(
        df: pd.DataFrame,
    ) -> RetentionTimeDataResult:
        """Check whether the DataFrame has retention time and mass data.

        The service does NOT render retention time plots. The UI layer calls
        RetentionTimePlotterService directly for plotting.

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
    # Pairwise Correlation
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

    # ============================================================
    # PCA Sample Removal
    # ============================================================

    @staticmethod
    def remove_samples(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        samples_to_remove: List[str],
    ) -> SampleRemovalResult:
        """Remove samples from DataFrame and create updated ExperimentConfig."""
        if not samples_to_remove:
            raise ValidationError(
                "samples_to_remove cannot be empty. "
                "Pass a non-empty list of sample labels."
            )

        actual_to_remove = [
            s for s in samples_to_remove if s in experiment.full_samples_list
        ]
        total_samples = len(experiment.full_samples_list)
        remaining = total_samples - len(actual_to_remove)

        if remaining < 2:
            raise ValidationError(
                f"Cannot remove {len(actual_to_remove)} sample(s): only "
                f"{remaining} would remain, but at least 2 are required."
            )

        new_experiment = experiment.without_samples(actual_to_remove)
        updated_df = QualityCheckService._drop_and_rename_columns(
            df, experiment, new_experiment, actual_to_remove
        )
        survivors = [
            s for s in experiment.full_samples_list if s not in actual_to_remove
        ]

        return SampleRemovalResult(
            updated_df=updated_df,
            updated_experiment=new_experiment,
            removed_samples=actual_to_remove,
            samples_before=total_samples,
            samples_after=len(new_experiment.full_samples_list),
            old_to_new={
                f'concentration[{old}]': f'concentration[{new}]'
                for old, new in zip(survivors, new_experiment.full_samples_list)
            },
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
