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

            # Set up protein concentrations (use a copy to avoid modifying the original)
            protein_lookup = protein_df.copy()
            if 'Sample' in protein_lookup.columns:
                protein_lookup.set_index('Sample', inplace=True)
                
            # Find and normalize intensity columns
            intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
            
            for col in intensity_cols:
                sample_name = col[col.find('[')+1:col.find(']')]
                if sample_name in protein_lookup.index:
                    conc_value = protein_lookup.loc[sample_name, 'Concentration']
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
        
    def process_standards_file(self, standards_df, cleaned_df, intsta_df, standards_in_dataset):
        """
        Process and validate uploaded standards file based on user selection.
        
        Args:
            standards_df (pd.DataFrame): DataFrame containing standards data from the uploaded file
            cleaned_df (pd.DataFrame): DataFrame containing the cleaned dataset (non-standards)
            intsta_df (pd.DataFrame): DataFrame containing the internal standards dataset
            standards_in_dataset (bool): True if standards exist in dataset (legacy mode), 
                                        False if standards are external (new mode)

        Returns:
            pd.DataFrame: Processed standards DataFrame with intensity values

        Raises:
            ValueError: If the standards file is invalid or has format issues
        """
        # Get expected intensity columns from the main dataset
        full_df = pd.concat([cleaned_df, intsta_df], ignore_index=True)
        expected_intensity_cols = [col for col in full_df.columns if col.startswith('intensity[')]
        
        if not expected_intensity_cols:
            raise ValueError("No intensity columns found in the main dataset")
        
        # Standardize column names: 1st col = LipidMolec, 2nd col = ClassKey, rest = intensity columns
        standards_df = self._standardize_standards_columns(standards_df, expected_intensity_cols)
        
        if standards_in_dataset:
            # LEGACY MODE: Extract standards from main dataset
            return self._extract_standards_from_dataset(standards_df, full_df, expected_intensity_cols)
        else:
            # NEW MODE: Use complete standards dataset with intensity values
            return self._validate_complete_standards(standards_df, expected_intensity_cols)

    def _standardize_standards_columns(self, df, expected_intensity_cols):
        """
        Standardize column names: 
        - 1st column â†’ LipidMolec
        - 2nd column â†’ ClassKey (if exists and not numeric)
        - Remaining columns â†’ intensity[s1], intensity[s2], etc.
        """
        df = df.copy()
        cols = df.columns.tolist()
        
        # First column is always LipidMolec
        new_col_names = {cols[0]: 'LipidMolec'}
        
        # Check if we have more than 1 column
        if len(cols) > 1:
            # If second column looks like intensity data (numeric), then no ClassKey provided
            # Otherwise, second column is ClassKey
            if df[cols[1]].dtype in ['int64', 'float64'] or pd.to_numeric(df[cols[1]], errors='coerce').notna().all():
                # No ClassKey, all remaining columns are intensities
                has_classkey = False
                intensity_start_idx = 1
            else:
                # Second column is ClassKey
                new_col_names[cols[1]] = 'ClassKey'
                has_classkey = True
                intensity_start_idx = 2
            
            # Rename remaining columns as intensity columns
            num_intensity_cols = len(cols) - intensity_start_idx
            if num_intensity_cols > 0:
                for i, col_idx in enumerate(range(intensity_start_idx, len(cols))):
                    if i < len(expected_intensity_cols):
                        new_col_names[cols[col_idx]] = expected_intensity_cols[i]
        
        # Rename columns
        df = df.rename(columns=new_col_names)
        
        # If ClassKey not present, infer from LipidMolec
        if 'ClassKey' not in df.columns:
            df['ClassKey'] = df['LipidMolec'].apply(
                lambda x: x.split('(')[0] if '(' in x else x
            )
        
        return df

    def _validate_complete_standards(self, standards_df, expected_intensity_cols):
        """
        Validate and process complete standards dataset (new mode).
        """
        # Check for required columns
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must have LipidMolec as the first column")
        
        if 'ClassKey' not in standards_df.columns:
            raise ValueError("Standards file must have ClassKey as the second column or it will be inferred from LipidMolec")
        
        # Check for intensity columns
        uploaded_intensity_cols = [col for col in standards_df.columns if col.startswith('intensity[')]
        
        if not uploaded_intensity_cols:
            raise ValueError(
                f"Standards file must contain intensity columns. "
                f"Expected columns after LipidMolec and ClassKey: {', '.join(expected_intensity_cols)}. "
                f"The columns after LipidMolec and ClassKey should contain intensity values for each sample."
            )
        
        # Validate that all required intensity columns are present
        missing_cols = set(expected_intensity_cols) - set(uploaded_intensity_cols)
        if missing_cols:
            raise ValueError(
                f"Standards file is missing the following intensity columns: {', '.join(sorted(missing_cols))}. "
                f"Your main dataset has {len(expected_intensity_cols)} samples, so your standards file must have "
                f"the same number of intensity columns (columns 3 onwards in your CSV)."
            )
        
        # Select only the necessary columns
        result_df = standards_df[['LipidMolec', 'ClassKey'] + expected_intensity_cols].copy()
        
        # Convert intensity columns to numeric
        for col in expected_intensity_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)
        
        st.success(f"✓ Loaded {len(result_df)} standards with complete intensity data")
        return result_df

    def remove_standards_from_main_dataset(self, cleaned_df, standards_df):
        """
        Remove any lipids from the main dataset that match the uploaded standards.
        This ensures standards are not included in the analysis.
        
        Args:
            cleaned_df (pd.DataFrame): The main cleaned dataset
            standards_df (pd.DataFrame): The standards dataset
            
        Returns:
            tuple: (updated_cleaned_df, list of removed lipid names)
        """
        if standards_df is None or standards_df.empty:
            return cleaned_df, []
        
        # Get standard lipid names
        standard_names = set(standards_df['LipidMolec'].unique())
        
        # Find which standards exist in the main dataset
        existing_in_dataset = cleaned_df[cleaned_df['LipidMolec'].isin(standard_names)]['LipidMolec'].unique().tolist()
        
        if existing_in_dataset:
            # Remove these lipids from cleaned_df
            cleaned_df = cleaned_df[~cleaned_df['LipidMolec'].isin(standard_names)].copy()
        
        return cleaned_df, existing_in_dataset
    
    def _extract_standards_from_dataset(self, standards_df, full_df, expected_intensity_cols):
        """
        Extract standards from the main dataset (legacy mode).
        """
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must have LipidMolec as the first column")
        
        # Deduplicate standards - important to avoid extracting the same standard multiple times
        unique_standards = standards_df['LipidMolec'].drop_duplicates()
        
        # Validate that all standards exist in the combined dataset
        valid_standards = unique_standards.isin(full_df['LipidMolec'])
        if not valid_standards.all():
            invalid_standards = unique_standards[~valid_standards].tolist()
            raise ValueError(
                f"The following standards were not found in the dataset: {', '.join(invalid_standards)}. "
                f"If these are external standards, please select 'No' for the mode question and provide intensity values."
            )
        
        # Extract standards from the main dataset
        result_df = pd.DataFrame()
        for standard in unique_standards:
            standard_data = full_df[full_df['LipidMolec'] == standard].copy()
            if not standard_data.empty:
                if 'ClassKey' not in standard_data.columns:
                    # Try to get ClassKey from uploaded file
                    if 'ClassKey' in standards_df.columns:
                        standard_data['ClassKey'] = standards_df[standards_df['LipidMolec'] == standard]['ClassKey'].values[0]
                    else:
                        # Infer from LipidMolec
                        standard_data['ClassKey'] = standard.split('(')[0] if '(' in standard else standard
                standard_data = standard_data[['LipidMolec', 'ClassKey'] + expected_intensity_cols]
                result_df = pd.concat([result_df, standard_data], ignore_index=True)
        
        st.success(f"✓ Extracted {len(result_df)} standards from the main dataset")
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