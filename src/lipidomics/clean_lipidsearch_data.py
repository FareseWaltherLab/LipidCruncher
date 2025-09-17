import streamlit as st
import pandas as pd
import numpy as np
import re

class CleanLipidSearchData:
    """
    Class for cleaning and preprocessing lipidomics data.
    """
    def __init__(self):
        pass

    @st.cache_data(ttl=3600)
    def _convert_columns_to_numeric(_self, df, full_samples_list):
        value_cols = [f'intensity[{sample}]' for sample in full_samples_list]
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)
        return df

    @st.cache_data(ttl=3600)
    def _apply_filter(_self, df):
        acceptable_grades = ['A', 'B', 'C']
        return df[df['TotalGrade'].isin(acceptable_grades)]

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
       
        Args:
            class_key (str): The lipid class key (e.g., PC, PE, Ch)
            fa_key (str, float, None): The fatty acid composition, may be NaN
           
        Returns:
            str: Standardized lipid name
        """
        # Handle null/NaN FAKey
        if pd.isna(fa_key):
            fa_key = ""
        else:
            # Convert to string if it's not already
            fa_key = str(fa_key)
       
        # Special handling for cholesterol class - they often have different formatting
        if class_key == 'Ch':
            # Check if fa_key is empty or just contains deuterium labeling
            if not fa_key or fa_key.startswith('D'):
                # Handle different formats of Ch
                if fa_key.startswith('D'):
                    return f"Ch-{fa_key}()"
                else:
                    return f"Ch()"
       
        # Standard processing for other lipid classes
        if '+D' in fa_key:
            fa_key, internal_standard = fa_key.split('+', 1)
            internal_standard = '+' + internal_standard
        else:
            internal_standard = ''
           
        # Handle empty fa_key
        if not fa_key or fa_key == '()':
            return f"{class_key}(){internal_standard}"
           
        # Sort fatty acids for consistent representation
        sorted_fatty_acids = '_'.join(sorted(fa_key.strip('()').split('_')))
        return f"{class_key}({sorted_fatty_acids}){internal_standard}"

    def _select_AUC(self, df, experiment):
        clean_df = self._initialize_clean_df(experiment.full_samples_list)
        sample_names = experiment.full_samples_list
        for grade in ['A', 'B', 'C']:
            additional_condition = ['LPC', 'SM'] if grade == 'C' else None
            clean_df = self._process_lipid_grades(df, clean_df, [grade], sample_names, additional_condition)
        return clean_df

    @st.cache_data(ttl=3600)
    def _initialize_clean_df(_self, full_samples_list):
        columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                  [f'intensity[{sample}]' for sample in full_samples_list]
        return pd.DataFrame(columns=columns)

    @st.cache_data(ttl=3600)
    def _process_lipid_grades(_self, df, clean_df, grades, sample_names, additional_condition=None):
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
        # Filter rows with maximum TotalSmpIDRate(%)
        max_quality_df = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality]
        # Define grade priority (A > B > C)
        grade_priority = {'A': 1, 'B': 2, 'C': 3}
        # Select row with highest grade (lowest priority value)
        max_quality_df['grade_priority'] = max_quality_df['TotalGrade'].map(grade_priority)
        isolated_row = max_quality_df.sort_values('grade_priority').iloc[0]
        # Drop temporary column
        max_quality_df = max_quality_df.drop(columns=['grade_priority'])
        new_row = {col: isolated_row[col] for col in clean_df.columns}
        clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
        return clean_df

    @st.cache_data(ttl=3600)
    def _remove_missing_fa_keys(_self, df):
        """
        Removes rows with missing FA keys, with exceptions for:
        1. Ch class molecules
        2. Ch-D standard molecules (like Ch-D7)
       
        Args:
            df (pd.DataFrame): Input dataframe
           
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        # Create a mask for rows to keep:
        # Either they have a non-null FAKey, OR
        # They belong to the Ch class, OR
        # Their LipidMolec starts with "Ch-D"
        keep_mask = (
            df['FAKey'].notna() |
            (df['ClassKey'] == 'Ch') |
            df['LipidMolec'].str.contains(r'^Ch-D', regex=True, na=False)
        )
       
        # Return the filtered dataframe
        return df[keep_mask]

    def final_cleanup(self, df):
        return df.drop(columns=['TotalSmpIDRate(%)']).reset_index(drop=True)

    def data_cleaner(self, df, name_df, experiment):
        df = self._remove_missing_fa_keys(df)
        df = self._convert_columns_to_numeric(df, experiment.full_samples_list)
        df = self._apply_filter(df)
        df = self._correct_lipid_molec_column(df)
        df = self._select_AUC(df, experiment)
        return self.final_cleanup(df)

    @st.cache_data(ttl=3600)
    def extract_internal_standards(_self, df):
        """
        Extracts internal standards from the dataframe with improved pattern matching.
       
        True internal standards are identified as:
        1. Deuterium labeling patterns (e.g., +D7, -D7)
        2. Ch-D7 pattern specifically
        3. The :(s) notation, EXCEPT when combined with known chemical modifiers
        """
        # Create a copy of the dataframe
        df = df.copy()
       
        # Create pattern for deuterium-labeled standards
        deuterium_mask = df['LipidMolec'].str.contains(r'[+-]D\d+', regex=True, na=False)
       
        # Specific pattern for Ch-D7 type molecules
        ch_d_mask = df['LipidMolec'].str.contains(r'^Ch-D\d+', regex=True, na=False)
       
        # Handle the :(s) notation
        standard_notation_mask = df['LipidMolec'].str.contains(r':\(s\)', regex=True, na=False)
       
        # Create a mask for chemical modifications (+XXXX)
        chemical_mod_mask = df['LipidMolec'].str.contains(r'\+[A-Z]{2,}', regex=True, na=False)
       
        # A true standard should be one of:
        # 1. Have deuterium labeling, OR
        # 2. Be a Ch-D7 type molecule, OR
        # 3. Have :(s) notation WITHOUT chemical modifications
        standards_mask = deuterium_mask | ch_d_mask | (standard_notation_mask & ~chemical_mod_mask)
       
        # Extract standards and non-standards
        internal_standards_df = df[standards_mask].reset_index(drop=True)
        non_standards_df = df[~standards_mask].reset_index(drop=True)
       
        return non_standards_df, internal_standards_df

    @st.cache_data(ttl=3600)
    def log_transform_df(_self, df, number_of_samples):
        abundance_cols = [f'MeanArea[s{i}]' for i in range(1, number_of_samples + 1)]
        df[abundance_cols] = df[abundance_cols].replace(0, 1)
        df[abundance_cols] = np.log10(df[abundance_cols])
        return df