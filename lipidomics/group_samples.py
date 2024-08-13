import pandas as pd
import streamlit as st
import re

class GroupSamples:
    """
    Handles grouping of samples for lipidomics data analysis.

    This class provides functionality to validate datasets, construct and manipulate dataframes for grouping samples 
    based on experimental conditions.

    Attributes:
        experiment (Experiment): An object representing the experimental setup.
        _mean_area_col_list_cache (list): Cache for storing the list of mean area columns.
    """

    def __init__(self, experiment):
        """
        Initializes the GroupSamples class with the given experimental setup.

        Args:
            experiment (Experiment): The experimental setup to be used for sample grouping.
        """
        self.experiment = experiment
        
    def check_dataset_validity(self, df):
        """
        Validates if the dataset contains the necessary columns required for LipidSearch analysis.

        Args:
            df (pd.DataFrame): The dataset to be validated.

        Returns:
            bool: True if the dataset contains the required columns, False otherwise.
        """
        required_static_columns = {'LipidMolec', 'ClassKey', 'BaseRt', 'FAKey'}
        if not required_static_columns.issubset(df.columns):
            return False

        # Dynamically check for 'MeanArea' columns based on the sample names in the dataset
        mean_area_columns_present = all(
            any(col.startswith(f"MeanArea[{sample_name}]") for col in df.columns)
            for sample_name in self._extract_sample_names(df)
        )

        return mean_area_columns_present

    def _extract_sample_names(self, df):
        """
        Extracts sample names from the 'MeanArea' columns in the DataFrame.

        Args:
            df (pd.DataFrame): The dataset to extract sample names from.

        Returns:
            list: A list of extracted sample names.
        """
        sample_names = set()
        for col in df.columns:
            match = re.match(r"MeanArea\[(.+)\]$", col)
            if match:
                sample_names.add(match.group(1))

        return list(sample_names)
            
    def check_input_validity(self, df):
        """
        Validates if the total number of samples in the dataset matches with the experimental setup.

        Args:
            df (pd.DataFrame): The dataset to be validated.

        Returns:
            bool: True if the total number of samples matches, False otherwise.
        """
        return len(self.build_mean_area_col_list(df)) == len(self.experiment.full_samples_list)
        
    @st.cache_data(ttl=3600)
    def build_mean_area_col_list(_self, df):
        """
        Extracts and caches the list of mean area columns from the dataset.
    
        Args:
            df (pd.DataFrame): The dataset from which to extract the columns.
    
        Returns:
            list: The list of mean area column indices.
    
        Raises:
            ValueError: If mean area columns are not found or follow an unexpected format.
        """
    
        mean_area_cols = []
        for col in df.columns:
            match = re.match(r"MeanArea\[s(\d+)\]$", col)
            if match:
                mean_area_cols.append(int(match.group(1)))
    
        if not mean_area_cols:
            raise ValueError("Mean area columns not found or follow an unexpected format.")

        return sorted(mean_area_cols)

    
    def build_group_df(self, df):
        """
        Constructs a DataFrame that maps sample names to their conditions.

        Args:
            df (pd.DataFrame): The dataset used to construct the group DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with sample names and their corresponding conditions.
        """
        sample_names = ['s' + str(i) for i in self.build_mean_area_col_list(df)]
        return pd.DataFrame({'sample name': sample_names, 'condition': self.experiment.extensive_conditions_list})
    
    def group_samples(self, group_df, selections):
        """
        Reorders samples in the group DataFrame based on user selections.
    
        Args:
            group_df (pd.DataFrame): The initial group DataFrame.
            selections (dict): A dictionary mapping conditions to selected sample names.
    
        Returns:
            pd.DataFrame: The updated group DataFrame with reordered samples.
    
        Raises:
            ValueError: If selections do not align with the experiment setup or if data integrity issues are detected.
        """
        # Check that all conditions are represented in selections
        if set(selections.keys()) != set(self.experiment.conditions_list):
            raise ValueError("Selections do not cover all experiment conditions.")
    
        # Verify that the total number of selected samples matches the experiment setup
        total_selected_samples = sum(len(samples) for samples in selections.values())
        if total_selected_samples != sum(self.experiment.number_of_samples_list):
            raise ValueError("The number of selected samples does not match the experiment setup.")
    
        # Ensure no duplication or omission of samples
        all_selected_samples = set(sample for samples in selections.values() for sample in samples)
        if len(all_selected_samples) != total_selected_samples:
            raise ValueError("Duplication or omission detected in selected samples.")
    
        ordered_samples = []
        for condition in self.experiment.conditions_list:
            selected_samples = selections[condition]
            ordered_samples.extend(selected_samples)
    
        group_df['sample name'] = ordered_samples
        return group_df

    
    def update_sample_names(self, group_df):
        """
        Updates the sample names in the group DataFrame to a standardized format.

        Args:
            group_df (pd.DataFrame): The DataFrame with original sample names.

        Returns:
            pd.DataFrame: A DataFrame with old and updated sample names and conditions.
        """
        total_samples = sum(self.experiment.number_of_samples_list)
        updated_names = ['s' + str(i+1) for i in range(total_samples)]
        return pd.DataFrame({'old name': group_df['sample name'], 'updated name': updated_names, 'condition': group_df['condition']})
