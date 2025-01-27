import pandas as pd
import numpy as np
import streamlit as st

class NormalizeData:
    """
    Class for normalizing lipidomic data using internal standards and BCA assay data.
    Format-agnostic implementation focusing only on essential columns for normalization.
    """
    
    def _filter_data_by_class(self, df, selected_class_list):
        """Filter dataset based on selected lipid classes."""
        return df[df['ClassKey'].isin(selected_class_list)]

    @st.cache_data(ttl=3600)
    def _compute_intsta_auc(_self, intsta_df, intsta_species, full_samples_list):
        """Compute AUC values for the selected internal standard."""
        cols = [f"intensity[{sample}]" for sample in full_samples_list]
        filtered_intsta = intsta_df[intsta_df['LipidMolec'] == intsta_species]
        return filtered_intsta[cols].values.reshape(len(full_samples_list),)

    @st.cache_data(ttl=3600)
    def _compute_normalized_auc(_self, selected_df, full_samples_list, lipid_class, intsta_auc, concentration):
        """Normalize AUC values for a specific lipid class using internal standard."""
        class_df = selected_df[selected_df['ClassKey'] == lipid_class]
        intensity_cols = [f"intensity[{sample}]" for sample in full_samples_list]
        intensity_df = class_df[intensity_cols]
        normalized_intensity_df = (intensity_df.divide(intsta_auc, axis='columns') * concentration)
        
        # Only keep essential columns needed for normalization
        essential_cols = ['LipidMolec', 'ClassKey']
        
        result_df = pd.concat([class_df[essential_cols].reset_index(drop=True), 
                             normalized_intensity_df.reset_index(drop=True)], axis=1)
        return result_df

    def _process_internal_standards(self, selected_df, added_intsta_species_lst, intsta_concentration_dict, intsta_df, full_samples_list, selected_class_list):
        """Process all internal standards and compute normalized values."""
        norm_df = pd.DataFrame()
        for intsta_species, lipid_class in zip(added_intsta_species_lst, selected_class_list):
            concentration = intsta_concentration_dict[intsta_species]
            intsta_auc = self._compute_intsta_auc(intsta_df, intsta_species, full_samples_list)
            class_norm_df = self._compute_normalized_auc(selected_df, full_samples_list, 
                                                       lipid_class, intsta_auc, concentration)
            norm_df = pd.concat([norm_df, class_norm_df], axis=0)
        return norm_df

    def _rename_intensity_columns(self, df):
        """Rename intensity columns to concentration columns."""
        renamed_cols = {col: col.replace('intensity[', 'concentration[') 
                       for col in df.columns if 'intensity[' in col}
        return df.rename(columns=renamed_cols)

    def normalize_data(self, selected_class_list, added_intsta_species_lst, intsta_concentration_dict, df, intsta_df, experiment):
        """
        Normalize data using internal standards.
        
        Args:
            selected_class_list (list): List of lipid classes to normalize
            added_intsta_species_lst (list): List of standards used for normalization
            intsta_concentration_dict (dict): Dictionary of standard concentrations
            df (pd.DataFrame): DataFrame to normalize
            intsta_df (pd.DataFrame): DataFrame containing standards data
            experiment (Experiment): Experiment object containing sample information
            
        Returns:
            pd.DataFrame: Normalized DataFrame with standards removed
        """
        selected_df = self._filter_data_by_class(df, selected_class_list)
        full_samples_list = experiment.full_samples_list
        
        # Remove standards from the dataset before normalization
        standards_to_remove = set(added_intsta_species_lst)
        selected_df = selected_df[~selected_df['LipidMolec'].isin(standards_to_remove)]
        
        norm_df = self._process_internal_standards(selected_df, added_intsta_species_lst,
                                                intsta_concentration_dict, intsta_df,
                                                full_samples_list, selected_class_list)
        
        if not norm_df.empty:
            norm_df.reset_index(drop=True, inplace=True)
            norm_df.fillna(0, inplace=True)
            norm_df.replace([np.inf, -np.inf], 0, inplace=True)
            norm_df = self._rename_intensity_columns(norm_df)
            
        return norm_df

    def normalize_using_bca(self, df, protein_df, preserve_prefix=False):
        """
        Normalize lipid intensities using protein concentrations from BCA assay.
        
        Args:
            df (pd.DataFrame): DataFrame to normalize
            protein_df (pd.DataFrame): DataFrame with protein concentrations
            preserve_prefix (bool): If True, keeps 'intensity[' prefix instead of changing to 'concentration['
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        try:
            normalized_df = df.copy()
            
            # Set up protein concentrations
            if 'Sample' in protein_df.columns:
                protein_df.set_index('Sample', inplace=True)
                
            # Find and normalize intensity columns
            intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
            
            for col in intensity_cols:
                sample_name = col[col.find('[')+1:col.find(']')]
                if sample_name in protein_df.index:
                    conc_value = protein_df.loc[sample_name, 'Concentration']
                    if conc_value <= 0:
                        st.warning(f"Skipping {sample_name} - protein concentration is zero or negative")
                        continue
                    normalized_df[col] = df[col] / conc_value
            
            # Only rename columns if not preserving prefix
            if not preserve_prefix:
                normalized_df = self._rename_intensity_columns(normalized_df)
            
            return normalized_df
            
        except Exception as e:
            st.error(f"Error in BCA normalization: {str(e)}")
            return df
        
    def process_standards_file(self, standards_df, cleaned_df):
        """
        Process and validate uploaded standards file.
        
        Args:
            standards_df (pd.DataFrame): DataFrame containing standards data
            cleaned_df (pd.DataFrame): DataFrame containing the cleaned dataset
            
        Returns:
            pd.DataFrame or None: Processed standards DataFrame if valid, None otherwise
        """
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must contain 'LipidMolec' column")

        # Extract ClassKey from LipidMolec
        standards_df['ClassKey'] = standards_df['LipidMolec'].apply(
            lambda x: x.split('(')[0] if '(' in x else x
        )

        # Validate standards exist in dataset
        valid_standards = standards_df['LipidMolec'].isin(cleaned_df['LipidMolec'])
        if not valid_standards.all():
            invalid_standards = standards_df[~valid_standards]['LipidMolec'].tolist()
            raise ValueError(f"The following standards were not found in the dataset: {', '.join(invalid_standards)}")
        
        # Get intensity columns from cleaned_df
        intensity_cols = [col for col in cleaned_df.columns if col.startswith('intensity[')]
        if not intensity_cols:
            raise ValueError("No intensity columns found in the cleaned dataset")
            
        # Create result DataFrame with standards and their intensity values
        result_df = pd.DataFrame()
        for standard in standards_df['LipidMolec']:
            # Get the row from cleaned_df for this standard
            standard_data = cleaned_df[cleaned_df['LipidMolec'] == standard].copy()
            if not standard_data.empty:
                # Add ClassKey if not present
                if 'ClassKey' not in standard_data.columns:
                    standard_data['ClassKey'] = standards_df[standards_df['LipidMolec'] == standard]['ClassKey'].values[0]
                result_df = pd.concat([result_df, standard_data[['LipidMolec', 'ClassKey'] + intensity_cols]], ignore_index=True)
            
        return result_df

    def validate_standards(self, standards_df):
        """
        Validate that standards DataFrame has required columns and format.
        
        Args:
            standards_df (pd.DataFrame): DataFrame containing standards data
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_cols = ['LipidMolec', 'ClassKey']
        if not all(col in standards_df.columns for col in required_cols):
            return False
            
        # Add any additional validation logic here
        return True