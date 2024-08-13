import pandas as pd
import numpy as np
import streamlit as st

class NormalizeData:
    """
    Class for normalizing lipidomic data using internal standards.
    """

    def _filter_data_by_class(self, df, selected_class_list):
        """
        Filters the dataset based on the selected lipid classes.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the lipidomic data.
            selected_class_list (list): A list of lipid classes selected for normalization.
    
        Returns:
            pd.DataFrame: A filtered DataFrame containing only the rows with lipid classes from the selected list.
        """
        try:
            return df[df['ClassKey'].isin(selected_class_list)]
        except KeyError as e:
            print(f"KeyError in _filter_data_by_class: {e}")
            return pd.DataFrame()

    def _process_internal_standards(self, selected_df, added_intsta_species_lst, intsta_concentration_dict, intsta_df, full_samples_list, selected_class_list):
        """
        Processes the internal standards and performs normalization calculations for each lipid class.
    
        Parameters:
            selected_df (pd.DataFrame): The DataFrame filtered for selected lipid classes.
            added_intsta_species_lst (list): List of internal standard species selected for each lipid class.
            intsta_concentration_dict (dict): Dictionary mapping internal standards to their concentrations.
            intsta_df (pd.DataFrame): DataFrame containing internal standards data.
            full_samples_list (list): List of all sample names in the experiment.
            selected_class_list (list): List of selected lipid classes for normalization.
    
        Returns:
            pd.DataFrame: A DataFrame with normalized data for each lipid class based on the internal standards.
        """
        norm_df = pd.DataFrame()
        try:
            for intsta_species, lipid_class in zip(added_intsta_species_lst, selected_class_list):
                concentration = intsta_concentration_dict[intsta_species]
                intsta_auc = self._compute_intsta_auc(intsta_df, intsta_species, full_samples_list)
                class_norm_df = self._compute_normalized_auc(selected_df, full_samples_list, lipid_class, intsta_auc, concentration)
                norm_df = pd.concat([norm_df, class_norm_df], axis=0)
            return norm_df
        except Exception as e:
            print(f"Error in _process_internal_standards: {e}")
            return pd.DataFrame()

    def normalize_data(self, selected_class_list, added_intsta_species_lst, intsta_concentration_dict, df, intsta_df, experiment):
        """
        Main method to normalize the data based on selected classes, internal standards, and their concentrations.
    
        Parameters:
            selected_class_list (list): Selected classes for normalization.
            added_intsta_species_lst (list): Selected internal standards for each class.
            intsta_concentration_dict (dict): Concentrations of the selected internal standards.
            df (pd.DataFrame): Main dataset without internal standards.
            intsta_df (pd.DataFrame): Dataset containing internal standards.
            experiment: Object containing details about the experiment.
    
        Returns:
            pd.DataFrame: The resulting DataFrame with normalized data.
        """
        try:
            selected_df = self._filter_data_by_class(df, selected_class_list)
            full_samples_list = experiment.full_samples_list
            norm_df = self._process_internal_standards(selected_df, added_intsta_species_lst, intsta_concentration_dict, intsta_df, full_samples_list, selected_class_list)
            norm_df.reset_index(drop=True, inplace=True)
            norm_df.fillna(0, inplace=True)
            norm_df.replace([np.inf, -np.inf], 0, inplace=True)
            return norm_df
        except Exception as e:
            print(f"Error in normalize_data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def _compute_intsta_auc(_self, intsta_df, intsta_species, full_samples_list):
        """
        Computes the average area under curve (AUC) for the selected internal standard.

        Parameters:
            intsta_df (pd.DataFrame): Dataset containing internal standards.
            intsta_species (str): Internal standard species.
            full_samples_list (list): List of all samples.

        Returns:
            np.array: Array of AUC values for the internal standard.
        """
        try:
            return intsta_df[['MeanArea[' + sample + ']' for sample in full_samples_list]][intsta_df['LipidMolec'] == intsta_species].values.reshape(len(full_samples_list),)
        except Exception as e:
            print(f"Error in _compute_intsta_auc: {e}")
            return np.array([])

    @st.cache_data(ttl=3600)
    def _compute_normalized_auc(_self, selected_df, full_samples_list, lipid_class, intsta_auc, concentration):
        """
        Normalizes the AUC data for a specific lipid class using the internal standard.

        Parameters:
            selected_df (pd.DataFrame): Filtered dataset for a selected lipid class.
            full_samples_list (list): List of all samples.
            lipid_class (str): Lipid class being normalized.
            intsta_auc (np.array): AUC values for the internal standard.
            concentration (float): Concentration of the internal standard.

        Returns:
            pd.DataFrame: Dataframe with normalized AUC values for the lipid class.
        """
        try:
            # Filter for the specific lipid class
            class_df = selected_df[selected_df['ClassKey'] == lipid_class]
    
            # Extract the 'MeanArea' columns
            mean_area_cols = ['MeanArea[' + sample + ']' for sample in full_samples_list]
            mean_area_df = class_df[mean_area_cols]
    
            # Perform the normalization calculation using vectorized operations
            normalized_mean_area_df = (mean_area_df.divide(intsta_auc, axis='columns') * concentration)
    
            # Concatenate the non-numeric columns with the calculated normalized data
            non_numeric_cols = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt']
            result_df = pd.concat([class_df[non_numeric_cols].reset_index(drop=True), normalized_mean_area_df.reset_index(drop=True)], axis=1)
            
            return result_df
        except Exception as e:
            print(f"Error in _compute_normalized_auc: {e}")
            return pd.DataFrame()
        
    @st.cache_data(ttl=3600)
    def normalize_using_bca(_self, df, protein_df):
        """
        Normalize lipid intensities in the DataFrame using protein concentrations from BCA assay.
        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            protein_df (pd.DataFrame): DataFrame with 'Sample' as index and 'Concentration' columns.
        Returns:
            pd.DataFrame: DataFrame with normalized lipid intensities.
        """
        # Check if 'Sample' is a column and set it as index if true
        if 'Sample' in protein_df.columns:
            protein_df.set_index('Sample', inplace=True)
        elif 'Sample' not in protein_df.index:
            raise KeyError("Protein DataFrame must have 'Sample' as an index or column.")
        
        # Extract sample names from df column names, assuming format like 'MeanArea[s1]'
        corrected_columns = {col: col[col.find('[')+1:col.find(']')] for col in df.columns if 'MeanArea[' in col}
    
        # Apply normalization only to the relevant columns
        for col, corrected_col in corrected_columns.items():
            if corrected_col in protein_df.index:
                df[col] = df[col] / protein_df.loc[corrected_col, 'Concentration']
        
        return df