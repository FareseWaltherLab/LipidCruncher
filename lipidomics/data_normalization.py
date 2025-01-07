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
        """
        selected_df = self._filter_data_by_class(df, selected_class_list)
        full_samples_list = experiment.full_samples_list
        
        norm_df = self._process_internal_standards(selected_df, added_intsta_species_lst,
                                                 intsta_concentration_dict, intsta_df,
                                                 full_samples_list, selected_class_list)
        
        if not norm_df.empty:
            norm_df.reset_index(drop=True, inplace=True)
            norm_df.fillna(0, inplace=True)
            norm_df.replace([np.inf, -np.inf], 0, inplace=True)
            norm_df = self._rename_intensity_columns(norm_df)
            
        return norm_df

    def normalize_using_bca(self, df, protein_df):
        """
        Normalize lipid intensities using protein concentrations from BCA assay.
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
            
            # Rename the columns to concentration after normalization
            normalized_df = self._rename_intensity_columns(normalized_df)
            return normalized_df
            
        except Exception as e:
            st.error(f"Error in BCA normalization: {str(e)}")
            return df