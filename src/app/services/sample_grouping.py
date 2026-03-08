"""
Sample grouping service for lipidomics data.

Validates dataset structure, builds sample-to-condition mappings,
and handles manual regrouping with column reordering.

Pure logic — no Streamlit dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.models.experiment import ExperimentConfig


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class DatasetValidationResult:
    """Result of validating a dataset against an experiment config.

    Attributes:
        valid: Whether the dataset is valid.
        message: Status or error message.
        n_intensity_cols: Number of intensity columns found.
        expected_samples: Number of samples expected from experiment.
    """
    valid: bool
    message: str
    n_intensity_cols: int = 0
    expected_samples: int = 0


@dataclass
class GroupingResult:
    """Result of building sample-to-condition mapping.

    Attributes:
        group_df: DataFrame with 'sample name' and 'condition' columns.
        sample_names: Ordered list of sample names (e.g., ['s1', 's2', ...]).
    """
    group_df: pd.DataFrame
    sample_names: List[str]


@dataclass
class RegroupingResult:
    """Result of manual sample regrouping.

    Attributes:
        group_df: Updated group DataFrame with reordered samples.
        reordered_df: DataFrame with reordered and renamed intensity columns.
        old_to_new: Mapping from old column names to new column names.
        name_df: DataFrame showing old name → updated name → condition.
    """
    group_df: pd.DataFrame
    reordered_df: pd.DataFrame
    old_to_new: Dict[str, str]
    name_df: pd.DataFrame


# =============================================================================
# Service
# =============================================================================


class SampleGroupingService:
    """Pure business logic for sample grouping — no Streamlit dependencies."""

    @staticmethod
    def validate_dataset(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        data_format: Optional[str] = None,
    ) -> DatasetValidationResult:
        """Validate that a dataset matches the experiment configuration.

        Args:
            df: Standardized DataFrame with intensity[...] columns.
            experiment: ExperimentConfig defining the expected structure.
            data_format: Optional format string (e.g., 'MS-DIAL') for
                format-specific error messages.

        Returns:
            DatasetValidationResult with validation status and details.
        """
        if df is None or df.empty:
            return DatasetValidationResult(
                valid=False,
                message="Dataset is empty or None.",
            )

        if 'LipidMolec' not in df.columns:
            return DatasetValidationResult(
                valid=False,
                message="Missing LipidMolec column.",
            )

        intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
        if not intensity_cols:
            return DatasetValidationResult(
                valid=False,
                message="No intensity columns found.",
            )

        expected = len(experiment.full_samples_list)
        actual = len(intensity_cols)

        if expected != actual:
            if data_format == 'MS-DIAL' and actual == 2 * expected:
                msg = (
                    f"MS-DIAL Data Error: Found {actual} intensity columns "
                    f"but expected {expected} samples. Your MS-DIAL export "
                    "contains both raw and normalized data, but the 'Lipid IS' "
                    "column is missing. Re-export your data from MS-DIAL and "
                    "ensure the 'Lipid IS' column is included."
                )
            else:
                msg = (
                    f"Number of samples in data ({actual}) doesn't match "
                    f"experiment setup ({expected})."
                )
            return DatasetValidationResult(
                valid=False,
                message=msg,
                n_intensity_cols=actual,
                expected_samples=expected,
            )

        return DatasetValidationResult(
            valid=True,
            message="Dataset is valid.",
            n_intensity_cols=actual,
            expected_samples=expected,
        )

    @staticmethod
    def extract_sample_names(df: pd.DataFrame) -> List[str]:
        """Extract sample names from intensity[...] columns.

        Args:
            df: DataFrame with intensity[sample_name] columns.

        Returns:
            List of sample names extracted from column headers.
        """
        names = []
        for col in df.columns:
            match = re.match(r"intensity\[(.+)\]$", col)
            if match:
                names.append(match.group(1))
        return names

    @staticmethod
    def _extract_sample_numbers(df: pd.DataFrame) -> List[int]:
        """Extract sorted sample numbers from intensity[sN] columns.

        Args:
            df: DataFrame with intensity[s1], intensity[s2], ... columns.

        Returns:
            Sorted list of integer sample numbers.
        """
        numbers = []
        for col in df.columns:
            match = re.match(r"intensity\[s(\d+)\]$", col)
            if match:
                numbers.append(int(match.group(1)))
        return sorted(numbers)

    @staticmethod
    def build_group_df(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        data_format: Optional[str] = None,
        workbench_conditions: Optional[Dict[str, str]] = None,
    ) -> GroupingResult:
        """Build a sample-to-condition mapping DataFrame.

        Args:
            df: Standardized DataFrame with intensity[sN] columns.
            experiment: ExperimentConfig defining conditions and samples.
            data_format: Optional format string for format-specific behavior.
            workbench_conditions: Optional mapping of sample name to condition
                (used for Metabolomics Workbench auto-detected conditions).

        Returns:
            GroupingResult with group_df and sample_names.

        Raises:
            ValueError: If no intensity columns found.
        """
        sample_numbers = SampleGroupingService._extract_sample_numbers(df)
        if not sample_numbers:
            raise ValueError("No intensity[sN] columns found in DataFrame.")

        sample_names = ['s' + str(i) for i in sample_numbers]

        if (
            data_format == 'Metabolomics Workbench'
            and workbench_conditions is not None
        ):
            conditions = [
                workbench_conditions.get(sample, '')
                for sample in sample_names
            ]
        else:
            conditions = list(experiment.extensive_conditions_list)

        group_df = pd.DataFrame({
            'sample name': sample_names,
            'condition': conditions,
        })

        return GroupingResult(group_df=group_df, sample_names=sample_names)

    @staticmethod
    def regroup_samples(
        df: pd.DataFrame,
        group_df: pd.DataFrame,
        selections: Dict[str, List[str]],
        experiment: ExperimentConfig,
    ) -> RegroupingResult:
        """Regroup samples based on user selections.

        Reorders samples according to the selections dict, renames intensity
        columns to the new order, and generates a name mapping table.

        Args:
            df: Standardized DataFrame with intensity[...] columns.
            group_df: Current sample-to-condition mapping DataFrame.
            selections: Dict mapping condition name to list of selected samples.
            experiment: ExperimentConfig defining conditions.

        Returns:
            RegroupingResult with reordered data.

        Raises:
            ValueError: If selections don't cover all conditions.
        """
        if set(selections.keys()) != set(experiment.conditions_list):
            raise ValueError("Selections do not cover all experiment conditions.")

        # Build ordered sample list
        ordered_samples = [
            sample
            for condition in experiment.conditions_list
            for sample in selections[condition]
        ]

        # Update group_df
        updated_group_df = group_df.copy()
        updated_group_df['sample name'] = ordered_samples

        # Build column rename mapping
        old_to_new = {
            f'intensity[{sample}]': f'intensity[s{i + 1}]'
            for i, sample in enumerate(ordered_samples)
        }

        # Reorder columns in DataFrame
        reordered_df = SampleGroupingService._reorder_intensity_columns(
            df, old_to_new,
        )

        # Build name mapping table
        name_df = SampleGroupingService._build_name_mapping(
            updated_group_df, experiment,
        )

        return RegroupingResult(
            group_df=updated_group_df,
            reordered_df=reordered_df,
            old_to_new=old_to_new,
            name_df=name_df,
        )

    @staticmethod
    def _reorder_intensity_columns(
        df: pd.DataFrame,
        old_to_new: Dict[str, str],
    ) -> pd.DataFrame:
        """Reorder and rename intensity columns based on a mapping.

        Args:
            df: DataFrame with intensity[...] columns.
            old_to_new: Mapping from old column names to new column names.

        Returns:
            New DataFrame with columns reordered and renamed.
        """
        df = df.copy()

        static_cols = [c for c in df.columns if not c.startswith('intensity[')]
        intensity_cols = sorted(
            [c for c in df.columns if c.startswith('intensity[')]
        )

        result_df = pd.DataFrame(index=df.index)
        for col in static_cols:
            result_df[col] = df[col]

        new_intensity_cols = [old_to_new.get(col, col) for col in intensity_cols]
        for new_col, old_col in zip(new_intensity_cols, intensity_cols):
            result_df[new_col] = df[old_col]

        return result_df

    @staticmethod
    def _build_name_mapping(
        group_df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> pd.DataFrame:
        """Build a name mapping table showing old → new sample names.

        Args:
            group_df: DataFrame with 'sample name' and 'condition' columns.
            experiment: ExperimentConfig for total sample count.

        Returns:
            DataFrame with 'old name', 'updated name', and 'condition' columns.
        """
        total_samples = sum(experiment.number_of_samples_list)
        updated_names = ['s' + str(i + 1) for i in range(total_samples)]

        return pd.DataFrame({
            'old name': group_df['sample name'].values,
            'updated name': updated_names,
            'condition': group_df['condition'].values,
        })
