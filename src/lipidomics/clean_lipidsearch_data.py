import streamlit as st
import pandas as pd
import numpy as np
import re

class CleanLipidSearchData:
    """
    Class for cleaning and preprocessing lipidomics data with enhanced grade filtering.
    """
    def __init__(self):
        pass

    @st.cache_data(ttl=3600)
    def _convert_columns_to_numeric(_self, df, full_samples_list):
        value_cols = [f'intensity[{sample}]' for sample in full_samples_list]
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
        return df

    @st.cache_data(ttl=3600)
    def _apply_filter(_self, df, grade_config=None):
        """
        Apply grade filtering to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            grade_config (dict, optional): Dictionary mapping ClassKey to list of acceptable grades.
                                          If None, uses default filtering.
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if grade_config is None:
            # Default behavior: A, B for all; A, B, C for LPC and SM
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

    def _correct_lipid_molec_column(self, df):
        df['LipidMolec_modified'] = df.apply(
            lambda row: self._standardize_lipid_name(row['ClassKey'], row['FAKey']), axis=1
        )
        df.drop(columns='LipidMolec', inplace=True)
        df.rename(columns={'LipidMolec_modified': 'LipidMolec'}, inplace=True)
        return df

    def _standardize_lipid_name(self, class_key, fa_key):
        """
        Standardizes lipid names with special handling for Ch class molecules
        and null/NaN FAKey values.
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

    def _select_AUC(self, df, experiment, grade_config=None):
        """
        Select best AUC for each lipid based on grade priority.
        
        Args:
            df (pd.DataFrame): Input dataframe
            experiment: Experiment object
            grade_config (dict, optional): Dictionary mapping ClassKey to list of acceptable grades
        
        Returns:
            pd.DataFrame: Cleaned dataframe with best peaks selected
        """
        clean_df = self._initialize_clean_df(experiment.full_samples_list)
        sample_names = experiment.full_samples_list
        
        if grade_config is None:
            # Default behavior
            for grade in ['A', 'B', 'C']:
                additional_condition = ['LPC', 'SM'] if grade == 'C' else None
                clean_df = self._process_lipid_grades(df, clean_df, [grade], sample_names, additional_condition)
        else:
            # Custom behavior based on grade_config
            # Process grades in order of priority: A, then B, then C
            for grade in ['A', 'B', 'C']:
                # Find classes that accept this grade
                classes_for_grade = [
                    class_key for class_key, grades in grade_config.items() 
                    if grade in grades
                ]
                if classes_for_grade:
                    clean_df = self._process_lipid_grades(
                        df, clean_df, [grade], sample_names, classes_for_grade
                    )
        
        return clean_df

    @st.cache_data(ttl=3600)
    def _initialize_clean_df(_self, full_samples_list):
        columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                  [f'intensity[{sample}]' for sample in full_samples_list]
        return pd.DataFrame(columns=columns)

    @st.cache_data(ttl=3600)
    def _process_lipid_grades(_self, df, clean_df, grades, sample_names, additional_condition=None):
        """
        Process lipid grades with flexible class filtering.
        
        Args:
            df: Input dataframe
            clean_df: Accumulating clean dataframe
            grades: List of grades to process
            sample_names: List of sample names
            additional_condition: Either list of class names to include, or None for all classes
        
        Returns:
            Updated clean_df
        """
        temp_df = df[df['TotalGrade'].isin(grades)]
        if additional_condition:
            temp_df = temp_df[temp_df['ClassKey'].isin(additional_condition)]
        
        for lipid in temp_df['LipidMolec'].unique():
            if lipid not in clean_df['LipidMolec'].unique():
                clean_df = _self._add_lipid_to_clean_df(temp_df, clean_df, lipid, sample_names)
        return clean_df

    def _add_lipid_to_clean_df(self, temp_df, clean_df, lipid, sample_names):
        isolated_df = temp_df[temp_df['LipidMolec'] == lipid]
        max_peak_quality = max(isolated_df['TotalSmpIDRate(%)'])
        max_quality_df = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality]
        
        grade_priority = {'A': 1, 'B': 2, 'C': 3}
        max_quality_df['grade_priority'] = max_quality_df['TotalGrade'].map(grade_priority)
        isolated_row = max_quality_df.sort_values('grade_priority').iloc[0]
        max_quality_df = max_quality_df.drop(columns=['grade_priority'])
        
        new_row = {col: isolated_row[col] for col in clean_df.columns}
        clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
        return clean_df

    @st.cache_data(ttl=3600)
    def _remove_missing_fa_keys(_self, df):
        """
        Removes rows with missing FA keys, with exceptions for Ch class molecules.
        """
        keep_mask = (
            df['FAKey'].notna() |
            (df['ClassKey'] == 'Ch') |
            df['LipidMolec'].str.contains(r'^Ch-D', regex=True, na=False)
        )
        return df[keep_mask]

    def final_cleanup(self, df):
        return df.drop(columns=['TotalSmpIDRate(%)']).reset_index(drop=True)

    def data_cleaner(self, df, name_df, experiment, grade_config=None):
        """
        Main cleaning function with optional custom grade configuration.
        
        Args:
            df: Input dataframe
            name_df: Name mapping dataframe
            experiment: Experiment object
            grade_config (dict, optional): Custom grade filtering configuration
        
        Returns:
            Cleaned dataframe
        """
        df = self._remove_missing_fa_keys(df)
        df = self._convert_columns_to_numeric(df, experiment.full_samples_list)
        df = self._apply_filter(df, grade_config)
        df = self._correct_lipid_molec_column(df)
        df = self._select_AUC(df, experiment, grade_config)
        return self.final_cleanup(df)

    @st.cache_data(ttl=3600)
    def extract_internal_standards(_self, df):
        """
        Extracts internal standards from the dataframe with improved pattern matching.
        """
        df = df.copy()
        
        deuterium_mask = df['LipidMolec'].str.contains(r'[+-]D\d+', regex=True, na=False)
        ch_d_mask = df['LipidMolec'].str.contains(r'^Ch-D\d+', regex=True, na=False)
        standard_notation_mask = df['LipidMolec'].str.contains(r':\(s\)', regex=True, na=False)
        chemical_mod_mask = df['LipidMolec'].str.contains(r'\+[A-Z]{2,}', regex=True, na=False)
        
        standards_mask = deuterium_mask | ch_d_mask | (standard_notation_mask & ~chemical_mod_mask)
        
        internal_standards_df = df[standards_mask].reset_index(drop=True)
        non_standards_df = df[~standards_mask].reset_index(drop=True)
        
        return non_standards_df, internal_standards_df