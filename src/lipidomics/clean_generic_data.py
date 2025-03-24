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
        Preserves correct case for ClassKey.
        """
        try:        
            # Create case-insensitive column mapping
            col_map = {col.lower(): col for col in df.columns}
            
            # Define the exact case we want for static columns
            static_cols = {'lipidmolec': 'LipidMolec', 'classkey': 'ClassKey'}
            static_col_mapping = {}
            
            for col_lower, col_desired in static_cols.items():
                if col_lower in col_map:
                    static_col_mapping[col_map[col_lower]] = col_desired
                else:
                    raise KeyError(f"Missing column: {col_desired}")
            
            # Handle intensity columns
            intensity_col_mapping = {}
            for sample in full_samples_list:
                expected_col = f'intensity[{sample}]'.lower()
                found = False
                for actual_col in df.columns:
                    if actual_col.lower() == expected_col:
                        intensity_col_mapping[actual_col] = f'intensity[{sample}]'
                        found = True
                        break
                if not found:
                    raise KeyError(f"Missing column: intensity[{sample}]")
            
            # Combine mappings and create new DataFrame
            rename_dict = {**static_col_mapping, **intensity_col_mapping}
            result_df = df[list(rename_dict.keys())].copy()
            result_df = result_df.rename(columns=rename_dict)
            
            return result_df
            
        except KeyError as e:
            st.error(f"Error extracting columns: {str(e)}")
            return pd.DataFrame()

    def _update_column_names(self, df, name_df):
        """Updates column names in the DataFrame."""
        try:
            rename_dict = {
                f'Intensity[{old_name}]': f'intensity[{new_name}]'
                for old_name, new_name in zip(name_df['old name'], name_df['updated name'])
            }
            return df.rename(columns=rename_dict)
        except KeyError as e:
            st.error(f"Error updating column names: {str(e)}")
            return pd.DataFrame()

    def _convert_columns_to_numeric(self, df, full_samples_list):
        """Converts intensity data columns to numeric type."""
        try:
            intensity_cols = [f'intensity[{sample}]' for sample in full_samples_list]
            df_copy = df.copy()
            df_copy[intensity_cols] = df_copy[intensity_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            return df_copy
        except KeyError as e:
            st.error(f"Error converting columns to numeric: {str(e)}")
            return pd.DataFrame()

    def data_cleaner(self, df, name_df, experiment):
        """
        Main method to clean generic format data.
        
        This method performs several cleaning operations:
        1. Removes rows with invalid lipid names (empty, '#', special characters only)
        2. Removes rows where all intensity values are zero or null
        3. Converts missing values to zero
        4. Removes duplicate entries based on LipidMolec
        
        Args:
            df (pd.DataFrame): Input DataFrame to clean
            name_df (pd.DataFrame): DataFrame containing name mappings
            experiment (Experiment): Experiment object containing setup information
            
        Returns:
            pd.DataFrame: Cleaned DataFrame or empty DataFrame if error occurs
        """
        try:
            cleaned_df = df.copy()
            
            # Extract relevant columns including ClassKey
            cleaned_df = self._extract_relevant_columns(cleaned_df, experiment.full_samples_list)
            if cleaned_df.empty:
                return cleaned_df
                
            # Remove rows with invalid lipid names
            cleaned_df = self._remove_invalid_lipid_rows(cleaned_df)
            if cleaned_df.empty:
                st.error("No valid data remains after removing invalid lipid names")
                return cleaned_df
                
            cleaned_df = self._update_column_names(cleaned_df, name_df)
            cleaned_df = self._convert_columns_to_numeric(cleaned_df, experiment.full_samples_list)
            
            # Remove duplicates based on LipidMolec
            cleaned_df = cleaned_df.drop_duplicates(subset=['LipidMolec'])
            
            # Get intensity columns
            intensity_cols = [col for col in cleaned_df.columns if col.startswith('intensity[')]
            
            # Remove rows where all intensity values are zero or null
            cleaned_df = cleaned_df[~(cleaned_df[intensity_cols] == 0).all(axis=1)]
            cleaned_df = cleaned_df[~cleaned_df[intensity_cols].isnull().all(axis=1)]
            
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            return cleaned_df
            
        except Exception as e:
            st.error(f"Error in data cleaning: {str(e)}")
            return pd.DataFrame()
    
    def _remove_invalid_lipid_rows(self, df):
        """
        Remove rows with invalid lipid names.
        
        Invalid lipid names include:
        - Empty strings or pure whitespace
        - Single special characters like '#'
        - Strings containing only special characters
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: DataFrame with invalid lipid rows removed
        """
        try:
            # Create a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Get the name of the lipid molecule column (case insensitive)
            lipid_col = [col for col in df_copy.columns if col.lower() == 'lipidmolec'][0]
            
            # Define invalid patterns
            invalid_patterns = [
                r'^\s*$',           # Empty strings or pure whitespace
                r'^[#@$%&*]+$',     # Strings of only special characters
                r'^[0-9]+$',        # Strings of only numbers
                r'^#\s*$',          # Just # with optional whitespace
                r'^nan$',           # Literal 'nan'
                r'^null$',          # Literal 'null'
                r'^none$'           # Literal 'none'
            ]
            
            # Combine patterns
            combined_pattern = '|'.join(invalid_patterns)
            
            # Create mask for valid lipid names
            valid_mask = ~df_copy[lipid_col].str.match(combined_pattern, case=False, na=True)
            
            # Also remove NaN values
            valid_mask = valid_mask & ~df_copy[lipid_col].isna()
            
            # Apply the mask and return
            return df_copy[valid_mask]
            
        except Exception as e:
            st.error(f"Error removing invalid lipid rows: {str(e)}")
            return pd.DataFrame()
    
    def extract_internal_standards(self, df):
        """Extract internal standards based on ClassKey and deuterated standards."""
        try:
            # Find ClassKey column with case-insensitive search if needed
            if 'ClassKey' not in df.columns:
                col_map = {col.lower(): col for col in df.columns}
                if 'classkey' in col_map:
                    df = df.rename(columns={col_map['classkey']: 'ClassKey'})
            
            # Define patterns for internal standards
            patterns = [
                r'\(d\d+\)',  # Matches deuterated standards like (d7)
                r'ISTD',      # Case-insensitive ISTD in ClassKey
                r':\(s\)',    # Standards marked with :(s)
                r'\+D\d+',    # Alternative deuterated format
                r'SPLASH'     # SPLASH standards
            ]
            
            # Create combined pattern
            combined_pattern = '|'.join(f'(?:{pattern})' for pattern in patterns)
            
            # Check both ClassKey and LipidMolec for standards
            is_standard = (
                df['ClassKey'].str.contains('ISTD', case=False, na=False) |
                df['LipidMolec'].str.contains(combined_pattern, regex=True, na=False)
            )
            
            if is_standard.any():
                intsta_df = df[is_standard].copy()
                cleaned_df = df[~is_standard].copy()
                return cleaned_df, intsta_df
            else:
                return df, pd.DataFrame(columns=df.columns)
                
        except Exception as e:
            st.error(f"Error extracting internal standards: {str(e)}")
            return df, pd.DataFrame(columns=df.columns)
    