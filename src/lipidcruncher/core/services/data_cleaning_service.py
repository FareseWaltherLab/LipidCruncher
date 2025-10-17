"""
Data cleaning service for lipid data.
Handles cleaning for LipidSearch, Generic, and Metabolomics Workbench formats.
Pure logic - no UI dependencies.
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from ..models.experiment import ExperimentConfig


class DataCleaningService:
    """
    Service for cleaning and preprocessing lipidomic data.
    Handles format-specific cleaning operations.
    """
    
    def clean_lipidsearch_data(
        self,
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Clean LipidSearch format data.
        
        Args:
            df: Raw LipidSearch DataFrame
            experiment: Experiment configuration
            grade_config: Optional custom grade filtering configuration
                         Dict mapping ClassKey to list of acceptable grades
        
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing FA keys (except Ch class)
        df = self._remove_missing_fa_keys(df)
        
        # Convert columns to numeric
        df = self._convert_columns_to_numeric(df, experiment.samples_list)
        
        # Apply grade filtering
        df = self._apply_grade_filter(df, grade_config)
        
        # Standardize lipid names
        df = self._correct_lipid_molec_column(df)
        
        # Select best AUC for each lipid
        df = self._select_best_auc(df, experiment, grade_config)
        
        # Final cleanup
        df = self._final_cleanup_lipidsearch(df)
        
        return df
    
    def clean_generic_data(
        self,
        df: pd.DataFrame,
        experiment: ExperimentConfig
    ) -> pd.DataFrame:
        """
        Clean Generic format data.
        
        Args:
            df: Raw Generic DataFrame
            experiment: Experiment configuration
        
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with invalid lipid names
        df = self._remove_invalid_lipid_rows(df)
        
        if df.empty:
            raise ValueError("No valid data remains after removing invalid lipid names")
        
        # Convert columns to numeric
        df = self._convert_columns_to_numeric(df, experiment.samples_list)
        
        # Remove duplicates based on LipidMolec
        df = df.drop_duplicates(subset=['LipidMolec'])
        
        # Get intensity columns
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        
        # Remove rows where all intensity values are zero or null
        df = df[~(df[intensity_cols] == 0).all(axis=1)]
        df = df[~df[intensity_cols].isnull().all(axis=1)]
        
        df = df.reset_index(drop=True)
        
        return df
    
    def extract_internal_standards(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract internal standards from the cleaned dataframe.
        
        Args:
            df: Cleaned DataFrame
        
        Returns:
            Tuple of (df_without_standards, standards_df)
        """
        # Identify internal standards by patterns
        # Standards typically have (d7), (d9), or similar deuteration markers
        standard_pattern = r'\(d\d+\)|\+D\d+'
        
        # Find rows that match standard patterns
        is_standard = df['LipidMolec'].str.contains(standard_pattern, regex=True, na=False)
        
        standards_df = df[is_standard].copy()
        cleaned_df = df[~is_standard].copy()
        
        return cleaned_df, standards_df
    
    # ==================== LipidSearch-specific methods ====================
    
    def _remove_missing_fa_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing FA keys, with exceptions for Ch class molecules.
        """
        keep_mask = (
            df['FAKey'].notna() |
            (df['ClassKey'] == 'Ch') |
            df['LipidMolec'].str.contains(r'^Ch-D', regex=True, na=False)
        )
        return df[keep_mask]
    
    def _apply_grade_filter(
        self,
        df: pd.DataFrame,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> pd.DataFrame:
        """
        Apply grade filtering to the dataframe.
        """
        if grade_config is None:
            # Default behavior: A, B, C for all
            acceptable_grades = ['A', 'B', 'C']
            return df[df['TotalGrade'].isin(acceptable_grades)]
        else:
            # Custom filtering based on user configuration
            filtered_dfs = []
            for class_key, grades in grade_config.items():
                if grades:  # Only process if grades are selected
                    class_df = df[
                        (df['ClassKey'] == class_key) & 
                        (df['TotalGrade'].isin(grades))
                    ]
                    filtered_dfs.append(class_df)
            
            if filtered_dfs:
                return pd.concat(filtered_dfs, ignore_index=True)
            else:
                # Return empty dataframe with same structure if nothing selected
                return df.iloc[0:0]
    
    def _correct_lipid_molec_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize lipid names in the LipidMolec column.
        """
        df['LipidMolec_modified'] = df.apply(
            lambda row: self._standardize_lipid_name(row['ClassKey'], row['FAKey']),
            axis=1
        )
        df = df.drop(columns='LipidMolec')
        df = df.rename(columns={'LipidMolec_modified': 'LipidMolec'})
        return df
    
    def _standardize_lipid_name(self, class_key: str, fa_key: str) -> str:
        """
        Standardize lipid names with special handling for Ch class molecules.
        """
        if pd.isna(fa_key):
            fa_key = ""
        else:
            fa_key = str(fa_key)
        
        if class_key == 'Ch':
            if not fa_key or fa_key.startswith('D'):
                if fa_key.startswith('D'):
                    return f"Ch-{fa_key}()"
                else:
                    return f"Ch()"
        
        if '+D' in fa_key:
            fa_key, internal_standard = fa_key.split('+', 1)
            internal_standard = '+' + internal_standard
        else:
            internal_standard = ''
        
        if not fa_key or fa_key == '()':
            return f"{class_key}(){internal_standard}"
        
        sorted_fatty_acids = '_'.join(sorted(fa_key.strip('()').split('_')))
        return f"{class_key}({sorted_fatty_acids}){internal_standard}"
    
    def _select_best_auc(
        self,
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[Dict[str, List[str]]]
    ) -> pd.DataFrame:
        """
        Select best AUC for each lipid based on grade priority.
        """
        clean_df = self._initialize_clean_df(experiment.samples_list)
        sample_names = experiment.samples_list
        
        if grade_config is None:
            # Default behavior
            for grade in ['A', 'B', 'C']:
                additional_condition = ['LPC', 'SM'] if grade == 'C' else None
                clean_df = self._process_lipid_grades(
                    df, clean_df, [grade], sample_names, additional_condition
                )
        else:
            # Custom behavior based on grade_config
            for grade in ['A', 'B', 'C']:
                classes_for_grade = [
                    class_key for class_key, grades in grade_config.items() 
                    if grade in grades
                ]
                if classes_for_grade:
                    clean_df = self._process_lipid_grades(
                        df, clean_df, [grade], sample_names, classes_for_grade
                    )
        
        return clean_df
    
    def _initialize_clean_df(self, full_samples_list: List[str]) -> pd.DataFrame:
        """Initialize empty DataFrame with correct columns."""
        columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                  [f'intensity[{sample}]' for sample in full_samples_list]
        return pd.DataFrame(columns=columns)
    
    def _process_lipid_grades(
        self,
        df: pd.DataFrame,
        clean_df: pd.DataFrame,
        grades: List[str],
        sample_names: List[str],
        additional_condition: Optional[List[str]]
    ) -> pd.DataFrame:
        """
        Process lipid grades with flexible class filtering.
        """
        temp_df = df[df['TotalGrade'].isin(grades)]
        if additional_condition:
            temp_df = temp_df[temp_df['ClassKey'].isin(additional_condition)]
        
        for lipid in temp_df['LipidMolec'].unique():
            if lipid not in clean_df['LipidMolec'].unique():
                clean_df = self._add_lipid_to_clean_df(temp_df, clean_df, lipid, sample_names)
        return clean_df
    
    def _add_lipid_to_clean_df(
        self,
        temp_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        lipid: str,
        sample_names: List[str]
    ) -> pd.DataFrame:
        """
        Add a lipid to the clean DataFrame, selecting the best quality entry.
        """
        isolated_df = temp_df[temp_df['LipidMolec'] == lipid]
        max_peak_quality = max(isolated_df['TotalSmpIDRate(%)'])
        max_quality_df = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality]
        
        grade_priority = {'A': 1, 'B': 2, 'C': 3}
        max_quality_df = max_quality_df.copy()
        max_quality_df['grade_priority'] = max_quality_df['TotalGrade'].map(grade_priority)
        isolated_row = max_quality_df.sort_values('grade_priority').iloc[0]
        
        new_row = {col: isolated_row[col] for col in clean_df.columns}
        clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
        return clean_df
    
    def _final_cleanup_lipidsearch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup for LipidSearch data."""
        return df.drop(columns=['TotalSmpIDRate(%)']).reset_index(drop=True)
    
    # ==================== Generic format methods ====================
    
    def _remove_invalid_lipid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with invalid lipid names.
        Invalid names include: empty strings, '#', or strings with only special characters.
        """
        if 'LipidMolec' not in df.columns:
            raise ValueError("DataFrame must have 'LipidMolec' column")
        
        # Define invalid patterns
        invalid_mask = (
            df['LipidMolec'].isna() |
            (df['LipidMolec'].str.strip() == '') |
            (df['LipidMolec'] == '#') |
            df['LipidMolec'].str.match(r'^[^\w]+$', na=False)  # Only special characters
        )
        
        return df[~invalid_mask]
    
    # ==================== Common methods ====================
    
    def _convert_columns_to_numeric(
        self,
        df: pd.DataFrame,
        full_samples_list: List[str]
    ) -> pd.DataFrame:
        """
        Convert intensity columns to numeric type and handle null/negative values.
        """
        value_cols = [f'intensity[{sample}]' for sample in full_samples_list]
        df = df.copy()
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
        return df
