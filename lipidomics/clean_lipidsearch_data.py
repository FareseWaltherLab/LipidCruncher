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
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
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
        if '+D' in fa_key:
            fa_key, internal_standard = fa_key.split('+', 1)
            internal_standard = '+' + internal_standard
        else:
            internal_standard = ''
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
        isolated_row = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality].iloc[0]
        new_row = {col: isolated_row[col] for col in clean_df.columns}
        clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
        return clean_df

    @st.cache_data(ttl=3600)
    def _remove_missing_fa_keys(_self, df):
        return df.dropna(subset=['FAKey'])

    def final_cleanup(self, df):
        return df.drop(columns=['TotalSmpIDRate(%)']).reset_index(drop=True)

    def data_cleaner(self, df, name_df, experiment):
        df = self._remove_missing_fa_keys(df)
        #df = self._update_column_names(df, name_df)
        #df = self._extract_relevant_columns(df, experiment.full_samples_list)
        df = self._convert_columns_to_numeric(df, experiment.full_samples_list)
        df = self._apply_filter(df)
        df = self._correct_lipid_molec_column(df)
        df = self._select_AUC(df, experiment)
        return self.final_cleanup(df)

    @st.cache_data(ttl=3600)
    def extract_internal_standards(_self, df):
        patterns = [
            r'\+D\d+', r':\(s\)', r'\+\d*C\d*', r'\+\d*N\d*', r'SPLASH'
        ]
        combined_pattern = '|'.join(f'(?:{pattern})' for pattern in patterns)
        internal_standards_df = df[df['LipidMolec'].str.contains(combined_pattern, regex=True, na=False)].reset_index(drop=True)
        non_standards_df = df[~df['LipidMolec'].str.contains(combined_pattern, regex=True, na=False)].reset_index(drop=True)
        return non_standards_df, internal_standards_df

    @st.cache_data(ttl=3600)
    def log_transform_df(_self, df, number_of_samples):
        abundance_cols = [f'MeanArea[s{i}]' for i in range(1, number_of_samples + 1)]
        df[abundance_cols] = df[abundance_cols].replace(0, 1)
        df[abundance_cols] = np.log10(df[abundance_cols])
        return df
