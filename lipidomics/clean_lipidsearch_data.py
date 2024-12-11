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
    def _extract_relevant_columns(_self, df, full_samples_list):
        """
        Extracts relevant columns for analysis from the DataFrame.
    
        This method focuses on keeping columns essential for lipidomics analysis,
        including static columns and dynamic 'MeanArea' columns based on the given
        experiment's sample list.
    
        Args:
            df (pd.DataFrame): The dataset to be processed.
            experiment (Experiment): Object containing details about the experiment setup.
    
        Returns:
            pd.DataFrame: A DataFrame containing only the essential columns for analysis.
    
        Note:
            The 'FAKey' column is included as it is crucial for further data processing steps.
    
        Raises:
            KeyError: If one or more expected columns are not present in the DataFrame.
        """
        try:
            static_cols = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey']
            mean_area_cols = ['MeanArea[' + sample + ']' for sample in full_samples_list]
            relevant_cols = static_cols + mean_area_cols
            return df[relevant_cols]
        except KeyError as e:
            print(f"Error in extracting columns: {e}")
            return pd.DataFrame()


    @st.cache_data(ttl=3600)
    def _update_column_names(_self, df, name_df):
        """
        Updates column names in the DataFrame to reflect new sample names based on the name mapping DataFrame.
        Handles potential mismatches or missing columns gracefully.
    
        Args:
            df (pd.DataFrame): The dataset with columns to be renamed.
            name_df (pd.DataFrame): DataFrame containing old and updated sample names.
    
        Returns:
            pd.DataFrame: The DataFrame with updated column names, or an empty DataFrame if an error occurs.
        """
        try:
            rename_dict = {
                f'MeanArea[{old_name}]': f'MeanArea[{new_name}]'
                for old_name, new_name in zip(name_df['old name'], name_df['updated name'])
            }
            return df.rename(columns=rename_dict)
        except KeyError as e:
            print(f"Error in updating column names: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def _create_rename_dict(_self, name_df):
        """
        Creates a dictionary for renaming DataFrame columns based on a mapping in name_df.
    
        Args:
            name_df (pd.DataFrame): DataFrame with columns ['old name', 'updated name'] representing the mapping.
    
        Returns:
            dict: A dictionary where keys are old column names and values are updated column names.
    
        Raises:
            ValueError: If 'old name' or 'updated name' columns are missing in name_df.
        """
        try:
            if not {'old name', 'updated name'}.issubset(name_df.columns):
                raise ValueError("Columns 'old name' and 'updated name' must be present in name_df")
    
            # Using dictionary comprehension for concise and efficient mapping
            return {f'MeanArea[{old_name}]': f'MeanArea[{new_name}]' 
                    for old_name, new_name in zip(name_df['old name'], name_df['updated name'])}
        except ValueError as e:
            print(f"Error in creating rename dictionary: {e}")
            return {}  # Return an empty dictionary or handle as needed


    @st.cache_data(ttl=3600)
    def _convert_columns_to_numeric(_self, df, full_samples_list):
        """
        Converts abundance data columns in the DataFrame to numeric type, with non-numeric values replaced by zeros.
    
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            experiment (Experiment): Object with details about the experiment setup.
    
        Returns:
            pd.DataFrame: The DataFrame with abundance columns converted to numeric type.
    
        Raises:
            KeyError: If any required column is missing.
        """
        try:
            auc_cols = [f'MeanArea[{sample}]' for sample in full_samples_list]
    
            # Verify if all required columns are present in the DataFrame
            missing_cols = set(auc_cols) - set(df.columns)
            if missing_cols:
                raise KeyError(f"Missing columns in DataFrame: {missing_cols}")
    
            # Efficiently convert columns to numeric type
            df[auc_cols] = df[auc_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            return df
        except KeyError as e:
            print(f"Error in converting columns to numeric: {e}")
            return pd.DataFrame()  # Return an empty DataFrame or handle as needed


    @st.cache_data(ttl=3600)
    def _apply_filter(_self, df):
        """
        Filters the DataFrame based on 'TotalGrade' values to retain only high-quality lipidomics data.
    
        Args:
            df (pd.DataFrame): The DataFrame to be filtered.
    
        Returns:
            pd.DataFrame: A DataFrame containing rows with 'TotalGrade' as 'A', 'B', or 'C'.
    
        Raises:
            ValueError: If the 'TotalGrade' column is missing or contains unexpected values.
        """
        try:
            # Verify if 'TotalGrade' column exists
            if 'TotalGrade' not in df.columns:
                raise ValueError("Missing 'TotalGrade' column in DataFrame")
    
            # Define acceptable grades
            acceptable_grades = ['A', 'B', 'C']
    
            # Filter the DataFrame based on acceptable grades
            filtered_df = df[df['TotalGrade'].isin(acceptable_grades)]
    
            # Check if filtered DataFrame is empty
            if filtered_df.empty:
                raise ValueError("No rows with 'TotalGrade' as 'A', 'B', or 'C' found")
    
            return filtered_df
        except ValueError as e:
            print(f"Error in applying filter: {e}")
            return pd.DataFrame()  # Return an empty DataFrame or handle as needed


    def _correct_lipid_molec_column(self, df):
        """
        Standardizes lipid molecule names in the DataFrame to ensure consistent naming.
        This includes standardizing names of internal standard lipids.
    
        Args:
            df (pd.DataFrame): DataFrame with lipid molecule data.
    
        Returns:
            pd.DataFrame: The DataFrame with standardized lipid molecule names.
    
        Note:
            The method applies standardization rules for both regular and internal standard lipids.
            Internal standard lipids are identified and processed differently to preserve their markers.
    
        Raises:
            Exception: For any unexpected errors during standardization.
        """
        try:
            # Apply standardization to each lipid molecule name
            df['LipidMolec_modified'] = df.apply(
                lambda row: self._standardize_lipid_name(row['ClassKey'], row['FAKey']), axis=1
            )
    
            # Replace the original 'LipidMolec' column with the modified one
            df.drop(columns='LipidMolec', inplace=True)
            df.rename(columns={'LipidMolec_modified': 'LipidMolec'}, inplace=True)
            
            return df
        except Exception as e:
            # Handle any other unexpected errors
            print(f"An error occurred in standardizing lipid names: {e}")
            return pd.DataFrame()  # Return an empty DataFrame or handle as needed



    def _standardize_lipid_name(self, class_key, fa_key):
        """
        Standardizes a lipid molecule name based on its class key and fatty acids.
        This includes handling internal standard markers within lipid names.
    
        Args:
            class_key (str): Class key of the lipid molecule.
            fa_key (str): Fatty acids part of the lipid molecule.
    
        Returns:
            str: Standardized lipid molecule name.
    
        Note:
            The method checks if the fatty acid key contains an internal standard part and processes it accordingly.
            This ensures that internal standard markers are preserved in the standardized name.
            It also checks that both class_key and fa_key are strings.
        """
        if not isinstance(class_key, str) or not isinstance(fa_key, str):
            raise ValueError("class_key and fa_key must be strings")
    
        # Handle internal standards in the fatty acid key
        if '+D' in fa_key:
            # Splitting the fatty acid key and internal standard marker
            fa_key, internal_standard = fa_key.split('+', 1)
            internal_standard = '+' + internal_standard  # Re-adding '+' for formatting
        else:
            internal_standard = ''
        
        # Sorting and joining fatty acid chains for standardization
        sorted_fatty_acids = '_'.join(sorted(fa_key.strip('()').split('_')))
        return f"{class_key}({sorted_fatty_acids}){internal_standard}"



    def _select_AUC(self, df, experiment):
        """
        Selects the highest quality peak for each unique lipid in the dataset.
    
        This method processes lipids based on their quality grade ('A', 'B', or 'C'), ensuring
        that only the best data is used for analysis. Special attention is given to SM and LPC lipids
        for 'C' grade processing.
    
        Args:
            df (pd.DataFrame): The DataFrame containing lipidomics data.
            experiment (Experiment): The experiment object containing sample information.
    
        Returns:
            pd.DataFrame: The cleaned DataFrame with selected lipids based on peak quality.
    
        Raises:
            ValueError: If the DataFrame is empty or the experiment object is not provided.
        """
        try:
            if df.empty or not experiment:
                raise ValueError("Empty DataFrame or missing experiment details provided.")
            
            clean_df = self._initialize_clean_df(experiment.full_samples_list)
            sample_names = experiment.full_samples_list
    
            for grade in ['A', 'B', 'C']:
                additional_condition = ['LPC', 'SM'] if grade == 'C' else None
                clean_df = self._process_lipid_grades(df, clean_df, [grade], sample_names, additional_condition)
    
            return clean_df
        except Exception as e:
            print(f"Error in selecting AUC: {e}")
            return pd.DataFrame()  # Return an empty DataFrame or handle as needed


    @st.cache_data(ttl=3600)
    def _initialize_clean_df(_self, full_samples_list):
        """
        Initializes a DataFrame structured for cleaned data, based on experiment setup.
    
        This method creates a DataFrame with essential columns, including lipid molecule information
        and abundance data columns for each sample in the experiment.
    
        Args:
            full_samples_list (list): A list of sample names from the experiment.
    
        Returns:
            pd.DataFrame: An initialized DataFrame with specified columns but no data.
    
        Raises:
            ValueError: If the full_samples_list is empty or not provided.
        """
        try:
            if not full_samples_list:
                raise ValueError("No sample names provided for initializing DataFrame.")
            
            columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                      [f'MeanArea[{sample}]' for sample in full_samples_list]
            return pd.DataFrame(columns=columns)
        except Exception as e:
            print(f"Error in initializing DataFrame: {e}")
            return pd.DataFrame()  # Return an empty DataFrame or handle as needed

    
    @st.cache_data(ttl=3600)
    def _process_lipid_grades(_self, df, clean_df, grades, sample_names, additional_condition=None):
        """
        Processes and updates the DataFrame based on lipid grades and additional conditions.
    
        Filters the DataFrame for specified lipid grades and updates the clean DataFrame with
        the highest quality data for each unique lipid.
    
        Args:
            df (pd.DataFrame): The DataFrame containing lipidomics data.
            clean_df (pd.DataFrame): The DataFrame to be updated with selected lipid data.
            grades (list): Grades to be considered for filtering.
            sample_names (list): Sample names for extracting abundance data.
            additional_condition (list, optional): Additional condition for class keys filtering.
    
        Returns:
            pd.DataFrame: Updated DataFrame with selected lipid data.
        """
        # Filter based on grades and additional conditions
        temp_df = df[df['TotalGrade'].isin(grades)]
        if additional_condition:
            temp_df = temp_df[temp_df['ClassKey'].isin(additional_condition)]
        
        # Update clean DataFrame with selected lipid data
        for lipid in temp_df['LipidMolec'].unique():
            if lipid not in clean_df['LipidMolec'].unique():
                clean_df = _self._add_lipid_to_clean_df(temp_df, clean_df, lipid, sample_names)
        
        return clean_df


    
    def _add_lipid_to_clean_df(self, temp_df, clean_df, lipid, sample_names):
        """
        Updates the clean DataFrame with the highest quality data for a given lipid.
    
        This function finds the lipid in the temporary DataFrame and updates the clean DataFrame
        with the highest quality data. It either updates an existing row or adds a new row.
    
        Args:
            temp_df (pd.DataFrame): Temporary DataFrame containing data for a specific grade.
            clean_df (pd.DataFrame): DataFrame to be updated with the highest quality lipid data.
            lipid (str): The lipid molecule to be updated.
            sample_names (list): List of sample names for extracting abundance data.
    
        Returns:
            pd.DataFrame: Updated clean DataFrame with the lipid data.
        """
        try:
            isolated_df = temp_df[temp_df['LipidMolec'] == lipid]
            max_peak_quality = max(isolated_df['TotalSmpIDRate(%)'])
            isolated_row = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality].iloc[0]
    
            row_index = clean_df.index[clean_df['LipidMolec'] == lipid].tolist()
            if row_index:
                for col in ['ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + [f'MeanArea[{sample}]' for sample in sample_names]:
                    clean_df.at[row_index[0], col] = isolated_row[col]
            else:
                new_row = {col: isolated_row[col] for col in clean_df.columns}
                clean_df = clean_df.append(new_row, ignore_index=True)
            return clean_df
        except Exception as e:
            print(f"Error in updating clean DataFrame: {e}")
            return clean_df  # Return the original DataFrame or handle as needed
 
    @st.cache_data(ttl=3600)
    def _remove_missing_fa_keys(_self, df):
        """
        Removes rows from the DataFrame where the FAKey is None.
    
        Args:
            df (pd.DataFrame): The DataFrame to be processed.
    
        Returns:
            pd.DataFrame: The DataFrame with rows having None FAKey removed.
        """
        return df.dropna(subset=['FAKey'])

    def final_cleanup(self, df):
        """
        Performs final cleanup steps on the DataFrame.
    
        Args:
            df (pd.DataFrame): The DataFrame to be cleaned up.
    
        Returns:
            pd.DataFrame: The cleaned DataFrame with the 'TotalSmpIDRate(%)' column removed and the index reset.
        """
        return df.drop(columns=['TotalSmpIDRate(%)']).reset_index(drop=True)    
    
    def data_cleaner(self, df, name_df, experiment):
        """
        Orchestrates a comprehensive cleaning process on lipidomics data.
        
        Args:
            df (pd.DataFrame): The DataFrame to be cleaned.
            name_df (pd.DataFrame): DataFrame with old and new sample names for renaming.
            experiment (Experiment): The experiment object containing sample information.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame ready for downstream analysis.
        
        Raises:
            Exception: General exception for any errors that occur during the data cleaning process.
        """
        try:
            # Remove rows where FAKey is None
            df = self._remove_missing_fa_keys(df)
            
            # Update column names based on the name mapping DataFrame
            df = self._update_column_names(df, name_df)
    
            # Extract relevant columns based on the experiment's sample list
            df = self._extract_relevant_columns(df, experiment.full_samples_list)
    
            # Convert specified columns to numeric type
            df = self._convert_columns_to_numeric(df, experiment.full_samples_list)
    
            # Filter DataFrame based on 'TotalGrade' values
            df = self._apply_filter(df)
    
            # Standardize lipid molecule names
            df = self._correct_lipid_molec_column(df)
    
            # Select the highest quality peak for each unique lipid
            df = self._select_AUC(df, experiment)
    
            # Perform final cleanup steps
            df = self.final_cleanup(df)
            
            # Final step: rename MeanArea columns to intensity format
            rename_dict = {
                f'MeanArea[{sample}]': f'intensity[s{i+1}]'
                for i, sample in enumerate(experiment.full_samples_list)
            }
            df = df.rename(columns=rename_dict)
    
            return df
        except Exception as e:
            print(f"An error occurred during data cleaning: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def extract_internal_standards(_self, df):
        """
        Separates internal standards from other lipids in the dataset.
        
        Internal standards typically contain:
        - Deuterated forms (e.g., +D3, +D4, +D7)
        - ':(s)' suffix
        - 13C labeled forms (e.g., +13C)
        - 15N labeled forms
        
        Args:
            df (pd.DataFrame): Input DataFrame with intensity/MeanArea columns
                
        Returns:
            tuple: (non_standards_df, internal_standards_df)
            - non_standards_df: DataFrame containing regular lipids
            - internal_standards_df: DataFrame containing internal standards
        """
        try:
            # Define patterns for internal standards
            internal_std_patterns = [
                r'\+D\d+',      # Matches deuterated forms like +D3, +D7
                r':\(s\)',      # Matches :(s) suffix
                r'\+\d*C\d*',   # Matches carbon labeling like +13C
                r'\+\d*N\d*',   # Matches nitrogen labeling like +15N
                r'SPLASH'       # Matches SPLASH standards
            ]
            
            # Combine patterns into a single regex
            combined_pattern = '|'.join(f'(?:{pattern})' for pattern in internal_std_patterns)
            
            # Separate standards using the combined pattern
            internal_standards_df = df[df['LipidMolec'].str.contains(combined_pattern, regex=True, na=False)].reset_index(drop=True)
            non_standards_df = df[~df['LipidMolec'].str.contains(combined_pattern, regex=True, na=False)].reset_index(drop=True)
            
            # Get columns to rename (either MeanArea or intensity)
            value_cols = [col for col in df.columns if col.startswith(('MeanArea[', 'intensity['))]
            
            # Create renaming dictionary
            rename_dict = {
                col: f'intensity[s{i+1}]'
                for i, col in enumerate(value_cols)
            }
            
            # Apply renaming to both DataFrames
            internal_standards_df = internal_standards_df.rename(columns=rename_dict)
            non_standards_df = non_standards_df.rename(columns=rename_dict)
            
            # Validate separation
            if len(df) != len(internal_standards_df) + len(non_standards_df):
                raise ValueError("Row count mismatch after separation")
                
            return non_standards_df, internal_standards_df
            
        except Exception as e:
            print(f"Error separating internal standards: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def log_transform_df(_self, df, number_of_samples):
        """
        Applies a logarithmic transformation to abundance columns in the dataset.
        
        Args:
            df (pd.DataFrame): The dataset to be transformed.
            number_of_samples (int): The total number of samples in the experiment.
        
        Returns:
            pd.DataFrame: The dataset with log-transformed abundance columns.
        
        Note:
            This transformation helps to normalize data distribution for statistical analysis and visualization.
        """
        try:
            abundance_cols = [f'MeanArea[s{i}]' for i in range(1, number_of_samples + 1)]
            df_log_transformed = df.copy()
            df_log_transformed[abundance_cols] = df_log_transformed[abundance_cols].replace(0, 1)
            df_log_transformed[abundance_cols] = np.log10(df_log_transformed[abundance_cols])
        except Exception as e:
            print(f"Error in log transformation: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error
    
        return df_log_transformed