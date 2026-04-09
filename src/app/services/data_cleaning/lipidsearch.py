"""
LipidSearch 5.0 format data cleaning.
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ...models.experiment import ExperimentConfig
from ...constants import (
    normalize_hydroxyl, sort_chains_lipid_maps,
    remove_phantom_chains, format_lipid_name,
    LYSO_CLASSES,
)
from .base import BaseDataCleaner
from .configs import GradeFilterConfig
from .exceptions import EmptyDataError, ConfigurationError


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
    OUTPUT_COLUMNS: Tuple[str, ...] = ('LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt')

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
        initial_count = len(df)

        # Validate input
        if LipidSearchCleaner.is_effectively_empty(df):
            raise EmptyDataError(
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

        messages.append(LipidSearchCleaner._make_summary_message(initial_count, len(df)))
        return df, messages

    # ==================== Cleaning Steps ====================

    @staticmethod
    def _step_remove_missing_fa_keys(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove rows with missing FA keys (except Ch class)."""
        initial_count = len(df)
        df = LipidSearchCleaner._remove_missing_fa_keys(df)

        if df.empty:
            raise EmptyDataError(
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
            raise ConfigurationError(
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
            raise EmptyDataError(
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
    def _standardize_single_name(class_key: str, fa_key: object) -> str:
        """Standardize a single lipid name to LIPID MAPS shorthand notation.

        Args:
            class_key: Lipid class (e.g., 'PC', 'PE', 'Ch')
            fa_key: Fatty acid key (e.g., '18:0_20:4'). May be str, float (NaN), or None.

        Returns:
            Standardized lipid name (e.g., 'PC 18:0_20:4')
        """
        fa_key = "" if pd.isna(fa_key) else str(fa_key)

        # Special handling for Cholesterol
        if class_key == 'Ch':
            return LipidSearchCleaner._standardize_cholesterol_name(fa_key)

        # Handle internal standards (deuterated)
        fa_key, internal_standard = LipidSearchCleaner._extract_internal_standard_suffix(fa_key)

        if not fa_key or fa_key == '()':
            return f"{class_key}{internal_standard}" if internal_standard else class_key

        # Parse chains
        raw = fa_key.strip('()')
        chains = raw.split('_')

        # Remove phantom 0:0 chains for lyso-species
        if class_key in LYSO_CLASSES:
            chains = remove_phantom_chains(chains)

        # Normalize hydroxyl notation: ;2O → ;O2
        chains = [normalize_hydroxyl(c) for c in chains]

        # Sort per LIPID MAPS rules
        chains = sort_chains_lipid_maps(chains, class_key)

        return format_lipid_name(class_key, chains, internal_standard)

    @staticmethod
    def _standardize_cholesterol_name(fa_key: str) -> str:
        """Standardize cholesterol lipid name."""
        if not fa_key or fa_key.startswith('D'):
            if fa_key.startswith('D'):
                return f"Ch-{fa_key}"
            return "Ch"
        return f"Ch {fa_key}"

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

    # Grade priority for sorting: lower number = better grade
    _GRADE_PRIORITY: Dict[str, int] = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    @staticmethod
    def _select_best_auc(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Select best AUC for each lipid based on grade priority.

        When multiple entries exist for the same lipid, select based on:
        1. Best grade (A > B > C > D)
        2. Highest TotalSmpIDRate(%)

        Default mode: A/B for all classes, C only for LPC/SM.
        Custom mode: each class has its own allowed grades from config.
        """
        # Filter to eligible entries based on grade config
        eligible = LipidSearchCleaner._build_eligibility_mask(df, grade_config)
        df = df[eligible].copy()

        if df.empty:
            sample_cols = [f'intensity[{s}]' for s in experiment.full_samples_list]
            return pd.DataFrame(
                columns=['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + sample_cols
            )

        # Sort by best grade first, then highest quality rate
        df['_grade_priority'] = df['TotalGrade'].map(LipidSearchCleaner._GRADE_PRIORITY).fillna(99)
        df = df.sort_values(
            ['_grade_priority', 'TotalSmpIDRate(%)'],
            ascending=[True, False]
        )

        # Keep first (best) entry per lipid
        df = df.drop_duplicates(subset=['LipidMolec'], keep='first')
        df = df.drop(columns=['_grade_priority'])

        return df.reset_index(drop=True)

    @staticmethod
    def _build_eligibility_mask(
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> 'pd.Series':
        """Build boolean mask for entries eligible for AUC selection."""
        if grade_config is None:
            # Default: A/B for all classes, C only for LPC/SM
            return (
                df['TotalGrade'].isin(['A', 'B']) |
                (df['TotalGrade'].eq('C') & df['ClassKey'].isin(['LPC', 'SM']))
            )

        # Custom: each class has its own allowed grades
        eligible = pd.Series(False, index=df.index)
        for class_key, grades in grade_config.items():
            eligible = eligible | (
                df['ClassKey'].eq(class_key) & df['TotalGrade'].isin(grades)
            )
        return eligible

    @staticmethod
    def _select_best_row(lipid_df: pd.DataFrame) -> pd.Series:
        """Select the best row based on quality rate and grade."""
        max_quality = lipid_df['TotalSmpIDRate(%)'].max()
        best_quality_df = lipid_df[lipid_df['TotalSmpIDRate(%)'] == max_quality].copy()

        best_quality_df['grade_priority'] = best_quality_df['TotalGrade'].map(
            LipidSearchCleaner._GRADE_PRIORITY
        ).fillna(99)

        return best_quality_df.sort_values('grade_priority').iloc[0]

    # ==================== Final Cleanup ====================

    @staticmethod
    def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Remove temporary columns and reset index."""
        if 'TotalSmpIDRate(%)' in df.columns:
            df = df.drop(columns=['TotalSmpIDRate(%)'])
        return df.reset_index(drop=True)
