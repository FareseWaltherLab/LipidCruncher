"""
LipidSearch 5.0 format data cleaning.
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ...models.experiment import ExperimentConfig
from .base import BaseDataCleaner
from .configs import GradeFilterConfig


class LipidSearchCleaner(BaseDataCleaner):
    """
    Cleaner for LipidSearch 5.0 format data.

    Handles:
    - Grade filtering (A/B/C/D)
    - FA key validation
    - Lipid name standardization
    - Best AUC selection per lipid
    """

    # Columns to keep in cleaned output
    OUTPUT_COLUMNS: List[str] = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt']

    @staticmethod
    def clean(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[GradeFilterConfig] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean LipidSearch 5.0 format data.

        Args:
            df: Raw LipidSearch DataFrame.
            experiment: Experiment configuration.
            grade_config: Optional grade filtering configuration.

        Returns:
            Tuple of (cleaned_df, messages_list).

        Raises:
            ValueError: If dataset is empty or becomes empty after cleaning.
        """
        messages = []

        # Validate input
        if LipidSearchCleaner.is_effectively_empty(df):
            raise ValueError(
                "Dataset is empty. Please upload a valid LipidSearch data file."
            )

        # Step 1: Remove rows with missing FA keys
        df, msg = LipidSearchCleaner._step_remove_missing_fa_keys(df)
        if msg:
            messages.append(msg)

        # Step 2: Convert intensity columns to numeric
        df = LipidSearchCleaner.convert_columns_to_numeric(df, experiment.full_samples_list)

        # Step 3: Apply grade filtering
        grade_dict = grade_config.grade_config if grade_config else None
        df, msg = LipidSearchCleaner._step_apply_grade_filter(df, grade_dict)
        if msg:
            messages.append(msg)

        # Step 4: Standardize lipid names
        df = LipidSearchCleaner._standardize_lipid_names(df)

        # Step 5: Select best AUC for each lipid
        df, msg = LipidSearchCleaner._step_select_best_auc(df, experiment, grade_dict)
        if msg:
            messages.append(msg)

        # Step 6: Final cleanup
        df = LipidSearchCleaner._final_cleanup(df)

        return df, messages

    # ==================== Cleaning Steps ====================

    @staticmethod
    def _step_remove_missing_fa_keys(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove rows with missing FA keys (except Ch class)."""
        initial_count = len(df)
        df = LipidSearchCleaner._remove_missing_fa_keys(df)

        if df.empty:
            raise ValueError(
                "No valid lipid species found. "
                "Dataset contains no rows with valid fatty acid (FA) keys."
            )

        removed = initial_count - len(df)
        msg = f"Removed {removed} rows with missing FA keys" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_apply_grade_filter(
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Apply grade filtering."""
        initial_count = len(df)
        df = LipidSearchCleaner._apply_grade_filter(df, grade_config)

        if df.empty:
            raise ValueError(
                "No lipid species remain after grade filtering. "
                "Please adjust your grade filter settings."
            )

        removed = initial_count - len(df)
        msg = f"Grade filter: Removed {removed} entries" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_select_best_auc(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """Select best AUC for each lipid."""
        initial_lipids = df['LipidMolec'].nunique()
        df = LipidSearchCleaner._select_best_auc(df, experiment, grade_config)

        if df.empty:
            raise ValueError(
                "No lipid species remain after quality filtering."
            )

        msg = f"AUC selection: Consolidated to {len(df)} unique lipids"
        return df, msg

    # ==================== FA Key Handling ====================

    @staticmethod
    def _remove_missing_fa_keys(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing FA keys, with exceptions for Ch class.

        Cholesterol molecules are kept even without FA keys.
        """
        if df.empty:
            return df

        df = df.copy()
        df['LipidMolec'] = df['LipidMolec'].fillna('').astype(str)

        keep_mask = (
            df['FAKey'].notna() |
            (df['ClassKey'] == 'Ch') |
            df['LipidMolec'].str.contains(r'^Ch-D', regex=True, na=False)
        )
        return df[keep_mask].copy()

    # ==================== Grade Filtering ====================

    @staticmethod
    def _apply_grade_filter(
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Apply grade filtering to the dataframe."""
        if grade_config is None:
            # Default: accept A, B, C grades
            return df[df['TotalGrade'].isin(['A', 'B', 'C'])].copy()

        # Custom filtering based on config
        filtered_dfs = []
        for class_key, grades in grade_config.items():
            if grades:
                class_df = df[
                    (df['ClassKey'] == class_key) &
                    (df['TotalGrade'].isin(grades))
                ]
                filtered_dfs.append(class_df)

        if filtered_dfs:
            return pd.concat(filtered_dfs, ignore_index=True)
        return df.iloc[0:0].copy()

    # ==================== Lipid Name Standardization ====================

    @staticmethod
    def _standardize_lipid_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize lipid names in the LipidMolec column."""
        df = df.copy()
        df['LipidMolec_modified'] = df.apply(
            lambda row: LipidSearchCleaner._standardize_single_name(
                row['ClassKey'], row['FAKey']
            ),
            axis=1
        )
        df = df.drop(columns=['LipidMolec'])
        return df.rename(columns={'LipidMolec_modified': 'LipidMolec'})

    @staticmethod
    def _standardize_single_name(class_key: str, fa_key) -> str:
        """
        Standardize a single lipid name.

        Args:
            class_key: Lipid class (e.g., 'PC', 'PE', 'Ch')
            fa_key: Fatty acid key (e.g., '18:0_20:4')

        Returns:
            Standardized lipid name (e.g., 'PC(18:0_20:4)')
        """
        fa_key = "" if pd.isna(fa_key) else str(fa_key)

        # Special handling for Cholesterol
        if class_key == 'Ch':
            return LipidSearchCleaner._standardize_cholesterol_name(fa_key)

        # Handle internal standards (deuterated)
        fa_key, internal_standard = LipidSearchCleaner._extract_internal_standard_suffix(fa_key)

        if not fa_key or fa_key == '()':
            return f"{class_key}(){internal_standard}"

        # Sort fatty acids for consistent naming
        sorted_fatty_acids = '_'.join(sorted(fa_key.strip('()').split('_')))
        return f"{class_key}({sorted_fatty_acids}){internal_standard}"

    @staticmethod
    def _standardize_cholesterol_name(fa_key: str) -> str:
        """Standardize cholesterol lipid name."""
        if not fa_key or fa_key.startswith('D'):
            if fa_key.startswith('D'):
                return f"Ch-{fa_key}()"
            return "Ch()"
        return f"Ch({fa_key})"

    @staticmethod
    def _extract_internal_standard_suffix(fa_key: str) -> Tuple[str, str]:
        """
        Extract internal standard suffix from FA key.

        Returns:
            Tuple of (fa_key_without_suffix, suffix)
        """
        if '+D' in fa_key:
            parts = fa_key.split('+', 1)
            return parts[0], '+' + parts[1]
        return fa_key, ''

    # ==================== AUC Selection ====================

    @staticmethod
    def _select_best_auc(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Select best AUC for each lipid based on grade priority.

        When multiple entries exist for the same lipid, select based on:
        1. Highest TotalSmpIDRate(%)
        2. Best grade (A > B > C)
        """
        sample_names = experiment.full_samples_list
        clean_df = LipidSearchCleaner._initialize_output_df(sample_names)

        if grade_config is None:
            clean_df = LipidSearchCleaner._process_default_grades(df, clean_df, sample_names)
        else:
            clean_df = LipidSearchCleaner._process_custom_grades(
                df, clean_df, sample_names, grade_config
            )

        return clean_df

    @staticmethod
    def _initialize_output_df(full_samples_list: List[str]) -> pd.DataFrame:
        """Initialize empty DataFrame with correct columns."""
        columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                  [f'intensity[{sample}]' for sample in full_samples_list]
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _process_default_grades(
        df: pd.DataFrame,
        clean_df: pd.DataFrame,
        sample_names: List[str]
    ) -> pd.DataFrame:
        """Process grades using default rules (A/B for all, C for LPC/SM)."""
        for grade in ['A', 'B', 'C']:
            class_filter = ['LPC', 'SM'] if grade == 'C' else None
            clean_df = LipidSearchCleaner._process_grade(
                df, clean_df, [grade], sample_names, class_filter
            )
        return clean_df

    @staticmethod
    def _process_custom_grades(
        df: pd.DataFrame,
        clean_df: pd.DataFrame,
        sample_names: List[str],
        grade_config: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Process grades using custom configuration."""
        for grade in ['A', 'B', 'C']:
            classes_for_grade = [
                class_key for class_key, grades in grade_config.items()
                if grade in grades
            ]
            if classes_for_grade:
                clean_df = LipidSearchCleaner._process_grade(
                    df, clean_df, [grade], sample_names, classes_for_grade
                )
        return clean_df

    @staticmethod
    def _process_grade(
        df: pd.DataFrame,
        clean_df: pd.DataFrame,
        grades: List[str],
        sample_names: List[str],
        class_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process lipids of specific grades."""
        temp_df = df[df['TotalGrade'].isin(grades)]
        if class_filter:
            temp_df = temp_df[temp_df['ClassKey'].isin(class_filter)]

        for lipid in temp_df['LipidMolec'].unique():
            if lipid not in clean_df['LipidMolec'].values:
                clean_df = LipidSearchCleaner._add_best_lipid_entry(
                    temp_df, clean_df, lipid
                )
        return clean_df

    @staticmethod
    def _add_best_lipid_entry(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        lipid: str
    ) -> pd.DataFrame:
        """Add the best quality entry for a lipid to target DataFrame."""
        lipid_df = source_df[source_df['LipidMolec'] == lipid]
        best_row = LipidSearchCleaner._select_best_row(lipid_df)

        new_row = {col: best_row[col] for col in target_df.columns}
        return pd.concat([target_df, pd.DataFrame([new_row])], ignore_index=True)

    @staticmethod
    def _select_best_row(lipid_df: pd.DataFrame) -> pd.Series:
        """Select the best row based on quality rate and grade."""
        max_quality = lipid_df['TotalSmpIDRate(%)'].max()
        best_quality_df = lipid_df[lipid_df['TotalSmpIDRate(%)'] == max_quality].copy()

        grade_priority = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        best_quality_df['grade_priority'] = best_quality_df['TotalGrade'].map(grade_priority)

        return best_quality_df.sort_values('grade_priority').iloc[0]

    # ==================== Final Cleanup ====================

    @staticmethod
    def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Remove temporary columns and reset index."""
        if 'TotalSmpIDRate(%)' in df.columns:
            df = df.drop(columns=['TotalSmpIDRate(%)'])
        return df.reset_index(drop=True)
