import streamlit as st
import pandas as pd
import numpy as np

class CleanGenericData:
    """
    Handles cleaning and preprocessing of generic format lipidomics data.
    Provides a simplified version of data cleaning compared to LipidSearch format.
    """
    
    def __init__(self):
        """Initialize the CleanGenericData class."""
        pass
        
    @st.cache_data(ttl=3600)
    def _extract_relevant_columns(self, df, full_samples_list):
        """
        Extracts relevant columns for analysis from the DataFrame.
        For generic format, only LipidMolec and MeanArea columns are needed.
        
        Args:
            df (pd.DataFrame): The dataset to be processed
            full_samples_list (list): List of all sample names in the experiment
            
        Returns:
            pd.DataFrame: DataFrame containing only the essential columns
        """
        try:
            static_cols = ['LipidMolec']
            mean_area_cols = ['MeanArea[' + sample + ']' for sample in full_samples_list]
            relevant_cols = static_cols + mean_area_cols
            return df[relevant_cols]
        except KeyError as e:
            print(f"Error in extracting columns: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def _update_column_names(self, df, name_df):
        """
        Updates column names in the DataFrame to reflect new sample names.
        
        Args:
            df (pd.DataFrame): DataFrame with columns to be renamed
            name_df (pd.DataFrame): DataFrame containing old and updated sample names
            
        Returns:
            pd.DataFrame: DataFrame with updated column names
        """
        try:
            rename_dict = {
                f'MeanArea[{old}]': f'MeanArea[{new}]' 
                for old, new in zip(name_df['old name'], name_df['updated name'])
            }
            return df.rename(columns=rename_dict)
        except KeyError as e:
            print(f"Error in updating column names: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def _convert_columns_to_numeric(self, df, full_samples_list):
        """
        Converts abundance data columns to numeric type.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data
            full_samples_list (list): List of all sample names
            
        Returns:
            pd.DataFrame: DataFrame with numeric abundance columns
        """
        try:
            auc_cols = [f'MeanArea[{sample}]' for sample in full_samples_list]
            df_copy = df.copy()
            df_copy[auc_cols] = df_copy[auc_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            return df_copy
        except KeyError as e:
            print(f"Error in converting columns to numeric: {e}")
            return pd.DataFrame()

    def data_cleaner(self, df, name_df, experiment):
        """
        Main method to clean generic format data.
        Simplified version of LipidSearch cleaning process.
        
        Args:
            df (pd.DataFrame): The DataFrame to be cleaned
            name_df (pd.DataFrame): DataFrame with old and new sample names for renaming
            experiment (Experiment): Object containing experiment setup details
            
        Returns:
            pd.DataFrame: Cleaned DataFrame ready for analysis
        """
        try:
            # Create a copy of the input DataFrame to avoid modifying the original
            cleaned_df = df.copy()
            
            # Update column names based on the name mapping DataFrame
            cleaned_df = self._update_column_names(cleaned_df, name_df)
            
            # Extract relevant columns based on the experiment's sample list
            cleaned_df = self._extract_relevant_columns(cleaned_df, experiment.full_samples_list)
            
            # Convert specified columns to numeric type
            cleaned_df = self._convert_columns_to_numeric(cleaned_df, experiment.full_samples_list)
            
            # Reset index for clean output
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            return cleaned_df
            
        except Exception as e:
            print(f"An error occurred during data cleaning: {e}")
            return pd.DataFrame()

    def extract_internal_standards(self, df):
        """
        For generic format, no internal standards processing is needed.
        Returns the same dataframe as non-standards and empty dataframe as standards.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            tuple: (original DataFrame, empty DataFrame for standards)
        """
        return df, pd.DataFrame(columns=df.columns)