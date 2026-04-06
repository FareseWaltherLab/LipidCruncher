"""
MS-DIAL format data cleaning.
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import List, Optional, Tuple

from ...models.experiment import ExperimentConfig
from .base import BaseDataCleaner
from .configs import QualityFilterConfig


class MSDIALCleaner(BaseDataCleaner):
    """
    Cleaner for MS-DIAL format data.

    Handles:
    - Quality filtering (Total Score, MS/MS matched)
    - Column extraction and standardization
    - Duplicate removal (keep highest score)
    """

    @staticmethod
    def clean(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        quality_config: Optional[QualityFilterConfig] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean MS-DIAL format data.

        Args:
            df: MS-DIAL DataFrame (already column-standardized).
            experiment: Experiment configuration.
            quality_config: Optional quality filtering configuration.

        Returns:
            Tuple of (cleaned_df, messages_list).

        Raises:
            ValueError: If dataset is empty or becomes empty after cleaning.
        """
        messages = []

        # Validate input
        if MSDIALCleaner.is_effectively_empty(df):
            raise ValueError(
                "Dataset is empty. Please upload a valid MS-DIAL data file."
            )

        initial_count = len(df)

        # Step 1: Apply quality filtering
        df, filter_msgs = MSDIALCleaner._step_apply_quality_filter(df, quality_config)
        messages.extend(filter_msgs)

        # Step 2: Extract relevant columns
        df = MSDIALCleaner._extract_columns(df, experiment.full_samples_list)

        # Step 3: Convert intensity columns to numeric
        df = MSDIALCleaner.convert_columns_to_numeric(df, experiment.full_samples_list)

        # Step 4: Remove invalid lipid rows
        df, msg = MSDIALCleaner._step_remove_invalid_rows(df)
        if msg:
            messages.append(msg)

        # Step 5: Remove duplicates (keep highest score)
        df, msg = MSDIALCleaner._step_remove_duplicates(df)
        if msg:
            messages.append(msg)

        # Step 6: Remove all-zero rows
        df, msg = MSDIALCleaner._step_remove_zero_rows(df)
        if msg:
            messages.append(msg)

        if df.empty:
            raise ValueError(
                "No lipid species with non-zero intensity values found."
            )

        messages.append(MSDIALCleaner._make_summary_message(initial_count, len(df)))

        return df.reset_index(drop=True), messages

    # ==================== Cleaning Steps ====================

    @staticmethod
    def _step_apply_quality_filter(
        df: pd.DataFrame,
        quality_config: Optional[QualityFilterConfig]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply quality filtering step."""
        df, messages = MSDIALCleaner._apply_quality_filter(df, quality_config)

        if df.empty:
            raise ValueError(
                "No data remaining after quality filtering. "
                "Please adjust your quality filter settings."
            )

        return df, messages

    @staticmethod
    def _step_remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove duplicates step (keeps highest Total score)."""
        has_score = 'Total score' in df.columns
        pre_count = len(df)
        df = MSDIALCleaner._remove_duplicates(df)

        removed = pre_count - len(df)
        if removed > 0:
            strategy = "highest Total score" if has_score else "first occurrence (Total score column not found)"
            msg = f"Removed {removed} duplicate entries (kept {strategy})"
        else:
            msg = None
        return df, msg

    # ==================== Quality Filtering ====================

    @staticmethod
    def _apply_quality_filter(
        df: pd.DataFrame,
        quality_config: Optional[QualityFilterConfig]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply MS-DIAL quality filtering.

        Returns:
            Tuple of (filtered_df, messages_list)
        """
        messages = []

        if quality_config is None:
            return df.copy(), messages

        filtered_df = df.copy()

        # Apply Total Score filter
        filtered_df, score_msgs = MSDIALCleaner._apply_score_filter(
            filtered_df, quality_config.total_score_threshold
        )
        messages.extend(score_msgs)

        # Apply MS/MS matched filter
        filtered_df, msms_msg = MSDIALCleaner._apply_msms_filter(
            filtered_df, quality_config.require_msms
        )
        if msms_msg:
            messages.append(msms_msg)

        return filtered_df, messages

    @staticmethod
    def _apply_score_filter(
        df: pd.DataFrame,
        threshold: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Apply Total Score filtering."""
        messages = []

        if 'Total score' not in df.columns or threshold <= 0:
            return df, messages

        initial_count = len(df)
        df = df.copy()
        df['Total score'] = pd.to_numeric(df['Total score'], errors='coerce')

        valid_scores = df['Total score'].notna().sum()
        if valid_scores == 0:
            messages.append(
                "Warning: Total score column contains no valid numeric values. "
                "Skipping score filter."
            )
            return df, messages

        df = df[df['Total score'] >= threshold]
        removed = initial_count - len(df)

        if removed > 0:
            messages.append(
                f"Quality filter: Removed {removed} entries with "
                f"Total score < {threshold}"
            )

        return df, messages

    @staticmethod
    def _apply_msms_filter(
        df: pd.DataFrame,
        require_msms: bool
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply MS/MS matched filtering."""
        if not require_msms or 'MS/MS matched' not in df.columns:
            return df, None

        pre_count = len(df)
        df = df[
            df['MS/MS matched'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
        ]
        removed = pre_count - len(df)

        msg = None
        if removed > 0:
            msg = f"MS/MS filter: Removed {removed} entries without MS/MS validation"

        return df, msg

    # ==================== Column Extraction ====================

    @staticmethod
    def _extract_columns(
        df: pd.DataFrame,
        full_samples_list: List[str]
    ) -> pd.DataFrame:
        """
        Extract relevant columns for MS-DIAL data.

        Keeps: LipidMolec, ClassKey, BaseRt (optional), CalcMass (optional),
               intensity columns, and Total score for deduplication.
        """
        col_map = {col.lower(): col for col in df.columns}
        columns_to_keep = []
        rename_dict = {}

        # Required columns
        columns_to_keep, rename_dict = MSDIALCleaner._add_required_columns(
            col_map, columns_to_keep, rename_dict
        )

        # Optional columns
        columns_to_keep, rename_dict = MSDIALCleaner._add_optional_columns(
            col_map, columns_to_keep, rename_dict
        )

        # Intensity columns
        columns_to_keep, rename_dict = MSDIALCleaner._add_intensity_columns(
            df, full_samples_list, columns_to_keep, rename_dict
        )

        result_df = df[columns_to_keep].copy()
        return result_df.rename(columns=rename_dict)

    @staticmethod
    def _add_optional_columns(
        col_map: dict,
        columns: List[str],
        rename_dict: dict
    ) -> Tuple[List[str], dict]:
        """Add optional columns (BaseRt, CalcMass, Total score)."""
        optional = [
            ('basert', 'BaseRt'),
            ('calcmass', 'CalcMass'),
            ('total score', 'Total score')
        ]

        for col_lower, col_standard in optional:
            if col_lower in col_map:
                columns.append(col_map[col_lower])
                rename_dict[col_map[col_lower]] = col_standard

        return columns, rename_dict

    # ==================== Duplicate Removal ====================

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates, keeping entry with highest Total score.
        """
        if df.empty:
            return df

        df = df.copy()

        if 'Total score' in df.columns:
            df['Total score'] = pd.to_numeric(df['Total score'], errors='coerce')
            df = df.sort_values('Total score', ascending=False)
            df = df.drop_duplicates(subset=['LipidMolec'], keep='first')
            df = df.drop(columns=['Total score'])
        else:
            df = df.drop_duplicates(subset=['LipidMolec'], keep='first')

        return df.reset_index(drop=True)
