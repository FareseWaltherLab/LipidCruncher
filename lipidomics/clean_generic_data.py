import streamlit as st
import pandas as pd
import numpy as np

class CleanGenericData:
    """
    Handles cleaning and preprocessing of generic format lipidomics data.
    """
    
    @st.cache_data(ttl=3600)
    def _extract_relevant_columns(_self, df, full_samples_list):
        """
        Extracts relevant columns for analysis from the DataFrame.
        
        Args:
            df (pd.DataFrame): The dataset to be processed
            full_samples_list (list): List of all sample names in the experiment
            
        Returns:
            pd.DataFrame: DataFrame containing only the essential columns
        """
        try:        
            static_cols = ['LipidMolec']
            intensity_cols = [f'Intensity[s{i+1}]' for i in range(len(full_samples_list))]
            relevant_cols = static_cols + intensity_cols
            
            # Check if all required columns exist
            missing_cols = [col for col in relevant_cols if col not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing columns: {missing_cols}")
            
            return df[relevant_cols].copy()
            
        except KeyError as e:
            return pd.DataFrame()

    def _update_column_names(self, df, name_df):
        """Updates column names in the DataFrame."""
        try:
            rename_dict = {
                f'Intensity[{old_name}]': f'intensity[s{i+1}]'
                for i, (old_name, _) in enumerate(zip(name_df['old name'], name_df['updated name']))
            }
            return df.rename(columns=rename_dict)
        except KeyError:
            return pd.DataFrame()

    def _convert_columns_to_numeric(self, df, full_samples_list):
        """Converts intensity data columns to numeric type."""
        try:
            intensity_cols = [f'intensity[s{i+1}]' for i in range(len(full_samples_list))]
            df_copy = df.copy()
            df_copy[intensity_cols] = df_copy[intensity_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            return df_copy
        except KeyError:
            return pd.DataFrame()

    def data_cleaner(self, df, name_df, experiment):
        """Main method to clean generic format data."""
        try:
            cleaned_df = df.copy()
            
            cleaned_df = self._extract_relevant_columns(cleaned_df, experiment.full_samples_list)
            if cleaned_df.empty:
                return cleaned_df
                
            cleaned_df = self._update_column_names(cleaned_df, name_df)
            cleaned_df = self._convert_columns_to_numeric(cleaned_df, experiment.full_samples_list)
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            return cleaned_df
            
        except Exception:
            st.error("Error in data cleaning. Please ensure your data follows the generic format requirements.")
            return pd.DataFrame()

    def extract_internal_standards(self, df):
        """For generic format, no internal standards processing is needed."""
        return df, pd.DataFrame(columns=df.columns)