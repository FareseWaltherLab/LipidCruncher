import pandas as pd
import re

class GroupSamples:
    """Handles grouping of samples for lipidomics data analysis."""
    
    def __init__(self, experiment, data_format=None):
        """
        Initialize with experiment setup.
        """
        self.experiment = experiment

    def check_dataset_validity(self, df):
        """Validates dataset columns."""
        required_static_columns = {'LipidMolec'}
        if not required_static_columns.issubset(df.columns):
            return False
        
        # Check for intensity columns in standardized format
        value_columns = [col for col in df.columns if col.startswith('intensity[')]
        return bool(value_columns)

    def extract_sample_names(self, df):
        """Extracts sample names from the standardized intensity columns."""
        sample_names = {
            match.group(1) for col in df.columns
            if (match := re.match(r"intensity\[(.+)\]$", col))
        }
        return list(sample_names)

    def check_input_validity(self, df):
        """Validates if the total number of samples matches the experiment setup."""
        n_samples = len(self.build_mean_area_col_list(df))
        expected_samples = len(self.experiment.full_samples_list)
        return n_samples == expected_samples

    def build_mean_area_col_list(self, df):
        """Returns list of sample indices from standardized intensity columns."""
        value_cols = [
            int(match.group(1)) for col in df.columns
            if (match := re.match(r"intensity\[s(\d+)\]$", col))
        ]
        if not value_cols:
            raise ValueError("No intensity columns found or they follow an unexpected format")
        return sorted(value_cols)

    def build_group_df(self, df):
        """Constructs sample-to-condition mapping DataFrame."""
        sample_names = ['s' + str(i) for i in self.build_mean_area_col_list(df)]
        group_df = pd.DataFrame({
            'sample name': sample_names,
            'condition': self.experiment.extensive_conditions_list
        })
        return group_df

    def group_samples(self, group_df, selections):
        """
        Reorders samples based on user selections and returns updated group_df 
        and column mapping.
        """
        if set(selections.keys()) != set(self.experiment.conditions_list):
            raise ValueError("Selections do not cover all experiment conditions.")
        
        # Create the new ordered sample list
        ordered_samples = [
            sample for condition in self.experiment.conditions_list
            for sample in selections[condition]
        ]
        
        # Update group_df with new sample ordering
        group_df['sample name'] = ordered_samples
        
        # Create mapping for column reordering
        old_to_new = {
            f'intensity[{sample}]': f'intensity[s{i + 1}]'
            for i, sample in enumerate(ordered_samples)
        }
        return group_df, old_to_new

    def reorder_intensity_columns(self, df, old_to_new):
        """
        Reorders and renames intensity columns in the DataFrame based on the mapping.
        """
        df = df.copy()
        
        # Separate static and intensity columns
        static_cols = [col for col in df.columns if not col.startswith('intensity[')]
        intensity_cols = sorted([col for col in df.columns if col.startswith('intensity[')])
        
        # Create new DataFrame and copy static columns
        result_df = pd.DataFrame(index=df.index)
        for col in static_cols:
            result_df[col] = df[col]
        
        # Create the new intensity columns list
        new_intensity_cols = [
            old_to_new.get(col, col) for col in intensity_cols
        ]
        
        # Copy data column by column
        for new_col, old_col in zip(new_intensity_cols, intensity_cols):
            result_df[new_col] = df[old_col]
        
        return result_df

    def update_sample_names(self, group_df):
        """Updates sample names to a standardized format."""
        total_samples = sum(self.experiment.number_of_samples_list)
        updated_names = ['s' + str(i + 1) for i in range(total_samples)]
        
        name_df = pd.DataFrame({
            'old name': group_df['sample name'],
            'updated name': updated_names,
            'condition': group_df['condition']
        })
        return name_df
