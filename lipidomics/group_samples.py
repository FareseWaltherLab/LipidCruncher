import pandas as pd
import streamlit as st
import re

class GroupSamples:
    """Handles grouping of samples for lipidomics data analysis."""

    def __init__(self, experiment, data_format='lipidsearch'):
        """Initialize with experiment setup and data format."""
        self.experiment = experiment
        self.data_format = data_format
        
    def check_dataset_validity(self, df):
        """Validates dataset columns based on format."""
        # Check for required static columns based on format
        if self.data_format == 'lipidsearch':
            required_static_columns = {'LipidMolec', 'ClassKey', 'BaseRt', 'FAKey'}
        else:
            required_static_columns = {'LipidMolec'}
            
        if not required_static_columns.issubset(df.columns):
            return False

        # Check for intensity columns (now standardized format for both types)
        value_columns = [col for col in df.columns if col.startswith('intensity[')]
        return bool(value_columns)

    def _extract_sample_names(self, df):
        """Extracts sample names from the value columns."""
        sample_names = set()
        
        for col in df.columns:
            match = re.match(r"intensity\[(.+)\]$", col)
            if match:
                sample_names.add(match.group(1))
                
        return list(sample_names)
            
    def check_input_validity(self, df):
        """Validates if the total number of samples matches the experiment setup."""
        return len(self.build_mean_area_col_list(df)) == len(self.experiment.full_samples_list)
        
    def build_mean_area_col_list(self, df):
        """Returns list of sample indices from value columns."""
        value_cols = []
        
        for col in df.columns:
            match = re.match(r"intensity\[s(\d+)\]$", col)
            if match:
                value_cols.append(int(match.group(1)))
    
        if not value_cols:
            raise ValueError("No intensity columns found or they follow an unexpected format")

        return sorted(value_cols)
    
    def build_group_df(self, df):
        """Constructs sample to condition mapping DataFrame."""
        sample_names = ['s' + str(i) for i in self.build_mean_area_col_list(df)]
        return pd.DataFrame({
            'sample name': sample_names, 
            'condition': self.experiment.extensive_conditions_list
        })
    
    def group_samples(self, group_df, selections):
        """Reorders samples based on user selections."""
        if set(selections.keys()) != set(self.experiment.conditions_list):
            raise ValueError("Selections do not cover all experiment conditions.")
    
        total_selected_samples = sum(len(samples) for samples in selections.values())
        if total_selected_samples != sum(self.experiment.number_of_samples_list):
            raise ValueError("The number of selected samples does not match the experiment setup.")
    
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
        """Updates sample names to a standardized format."""
        total_samples = sum(self.experiment.number_of_samples_list)
        updated_names = ['s' + str(i+1) for i in range(total_samples)]
        return pd.DataFrame({
            'old name': group_df['sample name'], 
            'updated name': updated_names, 
            'condition': group_df['condition']
        })