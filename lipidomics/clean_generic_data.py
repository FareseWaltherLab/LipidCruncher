import streamlit as st

class CleanDataGeneric:
    """
    Handles cleaning and preprocessing of generic format lipidomics data.
    Extracted from existing CleanData class, keeping only relevant parts.
    """
    
    def __init__(self):
        pass
        
    @st.cache_data(ttl=3600)
    def _extract_relevant_columns(_self, df, full_samples_list):
        """
        Extracts relevant columns for analysis from the DataFrame.
        For generic format, only LipidMolec and MeanArea columns are needed.
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
    def _update_column_names(_self, df, name_df):
        """
        Updates column names in the DataFrame to reflect new sample names.
        """
        try:
            rename_dict = {
                f'MeanArea[{old}]': f'MeanArea[{new}]' 
                for old, new in zip(name_df['old name'], name_df['updated name'])
            }
            df.rename(columns=rename_dict, inplace=True)
            return df
        except KeyError as e:
            print(f"Error in updating column names: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def _convert_columns_to_numeric(_self, df, full_samples_list):
        """
        Converts abundance data columns to numeric type.
        """
        try:
            auc_cols = [f'MeanArea[{sample}]' for sample in full_samples_list]
            df[auc_cols] = df[auc_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            return df
        except KeyError as e:
            print(f"Error in converting columns to numeric: {e}")
            return pd.DataFrame()

    def data_cleaner(self, df, name_df, experiment):
        """
        Main method to clean generic format data.
        Simplified version of LipidSearch cleaning process.
        """
        try:
            # Update column names based on the name mapping DataFrame
            df = self._update_column_names(df, name_df)
    
            # Extract relevant columns based on the experiment's sample list
            df = self._extract_relevant_columns(df, experiment.full_samples_list)
    
            # Convert specified columns to numeric type
            df = self._convert_columns_to_numeric(df, experiment.full_samples_list)
    
            return df
            
        except Exception as e:
            print(f"An error occurred during data cleaning: {e}")
            return pd.DataFrame()

    def extract_internal_standards(self, df):
        """
        For generic format, no internal standards processing is needed.
        Returns the same dataframe as non-standards and empty dataframe as standards.
        """
        return df, pd.DataFrame(columns=df.columns)