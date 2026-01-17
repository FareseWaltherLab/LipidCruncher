import streamlit as st
import pandas as pd
import numpy as np
import re


class CleanMSDIALData:
    """
    Handles cleaning and preprocessing of MS-DIAL format lipidomics data.
    
    MS-DIAL exports alignment results with specific columns and quality metrics.
    This class provides:
    - Quality filtering based on Total Score and MS/MS matched status
    - Column extraction and standardization
    - Internal standards detection using deuterium label patterns
    - Data type conversion and cleanup
    
    Follows the same patterns as CleanLipidSearchData and CleanGenericData.
    """
    
    def __init__(self):
        pass
    
    def _convert_columns_to_numeric(self, df, full_samples_list):
        """
        Converts intensity data columns to numeric type and handles null/negative values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            full_samples_list (list): List of sample identifiers
            
        Returns:
            pd.DataFrame: Dataframe with numeric intensity columns
        """
        try:
            intensity_cols = [f'intensity[{sample}]' for sample in full_samples_list]
            df_copy = df.copy()
            df_copy[intensity_cols] = df_copy[intensity_cols].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(0).clip(lower=0)
            return df_copy
        except KeyError as e:
            st.error(f"Error converting columns to numeric: {str(e)}")
            return pd.DataFrame()
    
    def _extract_relevant_columns(self, df, full_samples_list):
        """
        Extracts relevant columns for analysis from the DataFrame.
        
        For MS-DIAL data, this includes:
        - LipidMolec (standardized from Metabolite name)
        - ClassKey (from Ontology)
        - BaseRt (from Average Rt(min), optional)
        - CalcMass (from Average Mz, optional)
        - intensity[sN] columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            full_samples_list (list): List of sample identifiers
            
        Returns:
            pd.DataFrame: Dataframe with only relevant columns
        """
        try:
            # Create case-insensitive column mapping
            col_map = {col.lower(): col for col in df.columns}
            
            # Required columns with their standardized names
            required_cols = {
                'lipidmolec': 'LipidMolec',
                'classkey': 'ClassKey'
            }
            
            # Optional columns (may not be present after standardization)
            optional_cols = {
                'basert': 'BaseRt',
                'calcmass': 'CalcMass'
            }
            
            # Build column mapping for required columns
            selected_cols = {}
            for col_lower, col_desired in required_cols.items():
                if col_lower in col_map:
                    selected_cols[col_map[col_lower]] = col_desired
                else:
                    raise KeyError(f"Missing required column: {col_desired}")
            
            # Add optional columns if present
            for col_lower, col_desired in optional_cols.items():
                if col_lower in col_map:
                    selected_cols[col_map[col_lower]] = col_desired
            
            # Handle intensity columns
            for sample in full_samples_list:
                expected_col = f'intensity[{sample}]'.lower()
                found = False
                for actual_col in df.columns:
                    if actual_col.lower() == expected_col:
                        selected_cols[actual_col] = f'intensity[{sample}]'
                        found = True
                        break
                if not found:
                    raise KeyError(f"Missing intensity column: intensity[{sample}]")
            
            # Create new DataFrame with selected columns
            result_df = df[list(selected_cols.keys())].copy()
            result_df = result_df.rename(columns=selected_cols)
            
            return result_df
            
        except KeyError as e:
            st.error(f"Error extracting columns: {str(e)}")
            return pd.DataFrame()
    
    def _apply_quality_filter(self, df, quality_config):
        """
        Apply quality filtering based on Total Score and MS/MS matched.
        
        Returns:
            tuple: (filtered_df, messages_list) where messages_list contains filter result strings
        """
        messages = []
        
        if quality_config is None:
            return df, messages
        
        filtered_df = df.copy()
        initial_count = len(filtered_df)
        
        # Apply Total Score filter if column exists
        if 'Total score' in filtered_df.columns:
            threshold = quality_config.get('total_score_threshold', 60)
            # Convert to numeric, handling any non-numeric values
            filtered_df['Total score'] = pd.to_numeric(
                filtered_df['Total score'], errors='coerce'
            )
            
            # Check if we have any valid scores
            valid_scores = filtered_df['Total score'].notna().sum()
            if valid_scores == 0:
                messages.append("⚠️ Total score column exists but contains no valid numeric values. Skipping score filter.")
            elif threshold > 0:
                # Only filter if threshold > 0 (skip if "No filtering" selected)
                filtered_df = filtered_df[
                    filtered_df['Total score'] >= threshold
                ]
                score_filtered = initial_count - len(filtered_df)
                if score_filtered > 0:
                    messages.append(f"Quality filter: Removed {score_filtered} entries with Total score < {threshold}")
        
        # Apply MS/MS matched filter if required and column exists
        if quality_config.get('require_msms', False):
            if 'MS/MS matched' in filtered_df.columns:
                pre_msms_count = len(filtered_df)
                # Handle various TRUE representations
                filtered_df = filtered_df[
                    filtered_df['MS/MS matched'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
                ]
                msms_filtered = pre_msms_count - len(filtered_df)
                if msms_filtered > 0:
                    messages.append(f"MS/MS filter: Removed {msms_filtered} entries without MS/MS validation")
        
        return filtered_df, messages
    
    def _remove_invalid_lipid_rows(self, df):
        """
        Remove rows with invalid lipid names.
        
        Invalid lipid names include:
        - Empty strings or pure whitespace
        - 'Unknown' entries (from failed standardization)
        - Single special characters
        - Strings containing only special characters/numbers
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: DataFrame with invalid lipid rows removed
        """
        try:
            df_copy = df.copy()
            
            # Find the LipidMolec column
            lipid_col = 'LipidMolec'
            if lipid_col not in df_copy.columns:
                col_map = {col.lower(): col for col in df_copy.columns}
                if 'lipidmolec' in col_map:
                    lipid_col = col_map['lipidmolec']
                else:
                    return df_copy  # No lipid column found, return unchanged
            
            initial_count = len(df_copy)
            
            # Count different types of invalid entries for diagnostics
            nan_count = df_copy[lipid_col].isna().sum()
            empty_count = (df_copy[lipid_col].astype(str).str.strip() == '').sum()
            unknown_count = (df_copy[lipid_col].astype(str).str.upper() == 'UNKNOWN').sum()
            
            # Define invalid patterns
            invalid_patterns = [
                r'^\s*$',           # Empty strings or pure whitespace
                r'^Unknown$',       # Unknown from failed standardization
                r'^[#@$%&*]+$',     # Strings of only special characters
                r'^[0-9]+$',        # Strings of only numbers
                r'^nan$',           # Literal 'nan'
                r'^null$',          # Literal 'null'
                r'^none$',          # Literal 'none'
            ]
            
            # Combine patterns
            combined_pattern = '|'.join(invalid_patterns)
            
            # Create mask for valid lipid names
            valid_mask = ~df_copy[lipid_col].str.match(combined_pattern, case=False, na=True)
            
            # Also remove NaN values
            valid_mask = valid_mask & ~df_copy[lipid_col].isna()
            
            result_df = df_copy[valid_mask]
            removed_count = initial_count - len(result_df)
            
            # Show diagnostic info if many rows removed
            if removed_count > 0:
                st.info(f"Removed {removed_count} entries with invalid lipid names: "
                       f"{nan_count} NaN, {empty_count} empty, {unknown_count} 'Unknown'")
                
                # If everything was removed, show sample of what lipid names looked like
                if len(result_df) == 0:
                    sample_names = df_copy[lipid_col].dropna().head(10).tolist()
                    if sample_names:
                        st.warning(f"Sample lipid names from dataset: {sample_names[:5]}")
                    else:
                        st.warning("All lipid name entries are NaN or empty.")
            
            return result_df
            
        except Exception as e:
            st.error(f"Error removing invalid lipid rows: {str(e)}")
            return df
    
    def _remove_duplicates(self, df):
        """
        Remove duplicate entries based on LipidMolec.
        
        For MS-DIAL data, when duplicates exist, keep the entry with:
        1. Highest Total score (if available)
        2. First occurrence (if no score)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with duplicates removed
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # Check if Total score column exists for smart deduplication
        if 'Total score' in df_copy.columns:
            # Sort by Total score descending, then keep first occurrence
            df_copy['Total score'] = pd.to_numeric(df_copy['Total score'], errors='coerce')
            df_copy = df_copy.sort_values('Total score', ascending=False)
        
        # Remove duplicates keeping first (highest score if sorted)
        df_copy = df_copy.drop_duplicates(subset=['LipidMolec'], keep='first')
        
        return df_copy.reset_index(drop=True)
    
    def data_cleaner(self, df, name_df, experiment, quality_config=None):
        """
        Main cleaning function for MS-DIAL format data.
        
        Performs the following operations:
        1. Apply quality filtering (Total Score, MS/MS matched)
        2. Extract relevant columns
        3. Convert intensity columns to numeric
        4. Remove invalid lipid rows
        5. Remove duplicates
        6. Remove all-zero rows
        
        Args:
            df (pd.DataFrame): Input dataframe (already standardized)
            name_df (pd.DataFrame): Name mapping dataframe (may not be used for MS-DIAL)
            experiment (Experiment): Experiment object containing setup information
            quality_config (dict, optional): Quality filtering configuration:
                - 'total_score_threshold': Minimum Total score (default: 60)
                - 'require_msms': Whether to require MS/MS matched = TRUE (default: False)
                
        Returns:
            pd.DataFrame: Cleaned dataframe or empty DataFrame if error occurs
        """
        try:
            cleaned_df = df.copy()
            
            # Step 1: Apply quality filtering (before column extraction)
            cleaned_df, filter_messages = self._apply_quality_filter(cleaned_df, quality_config)
            # Store messages for display in main_app.py (always update to clear stale messages)
            st.session_state.msdial_filter_messages = filter_messages  # Will be empty list if no filtering
            if cleaned_df.empty:
                st.warning("No data remaining after quality filtering")
                return cleaned_df
            
            # Step 2: Extract relevant columns
            cleaned_df = self._extract_relevant_columns(cleaned_df, experiment.full_samples_list)
            if cleaned_df.empty:
                return cleaned_df
            
            # Step 3: Convert intensity columns to numeric
            cleaned_df = self._convert_columns_to_numeric(cleaned_df, experiment.full_samples_list)
            if cleaned_df.empty:
                return cleaned_df
            
            # Step 4: Remove invalid lipid rows
            cleaned_df = self._remove_invalid_lipid_rows(cleaned_df)
            if cleaned_df.empty:
                st.error("No valid data remains after removing invalid lipid names")
                return cleaned_df
            
            # Step 5: Remove duplicates
            cleaned_df = self._remove_duplicates(cleaned_df)
            
            # Step 6: Remove rows where all intensity values are zero
            intensity_cols = [col for col in cleaned_df.columns if col.startswith('intensity[')]
            if intensity_cols:
                cleaned_df = cleaned_df[~(cleaned_df[intensity_cols] == 0).all(axis=1)]
                cleaned_df = cleaned_df[~cleaned_df[intensity_cols].isnull().all(axis=1)]
            
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            return cleaned_df
            
        except Exception as e:
            st.error(f"Error in MS-DIAL data cleaning: {str(e)}")
            return pd.DataFrame()
    
    def extract_internal_standards(self, df):
        """
        Extract internal standards from MS-DIAL data based on common patterns.
        
        MS-DIAL internal standards can be identified by:
        - Deuterium labels: (d5), (d7), (d9), etc. in lipid name
        - ISTD marker in ClassKey/Ontology
        - SPLASH standard naming patterns
        - 'IS' or 'Internal Standard' in any relevant field
        
        Args:
            df (pd.DataFrame): Input dataframe with LipidMolec and ClassKey columns
            
        Returns:
            tuple: (cleaned_df, internal_standards_df)
                - cleaned_df: DataFrame with internal standards removed
                - internal_standards_df: DataFrame containing only internal standards
        """
        try:
            df_copy = df.copy()
            
            # Ensure required columns exist
            if 'LipidMolec' not in df_copy.columns:
                st.warning("LipidMolec column not found for internal standards extraction")
                return df_copy, pd.DataFrame(columns=df_copy.columns)
            
            # Define patterns for internal standard detection in lipid names
            lipid_patterns = [
                r'\(d\d+\)',        # Deuterium labels: (d5), (d7), (d9)
                r'-d\d+[_\)\s]',    # Alternative deuterium: -d7_, -d9)
                r'\[d\d+\]',        # Bracketed deuterium: [d7]
                r'ISTD',            # ISTD marker (case insensitive)
                r'SPLASH',          # SPLASH lipidomix standards
                r'Internal\s*Standard',  # Full text
                r'_IS$',            # Suffix _IS
                r'\(IS\)',          # (IS) marker
                r':\(s\)',          # :(s) notation
            ]
            
            # Combine lipid name patterns
            combined_lipid_pattern = '|'.join(f'(?:{pattern})' for pattern in lipid_patterns)
            
            # Check LipidMolec for patterns (case insensitive)
            is_standard_lipid = df_copy['LipidMolec'].str.contains(
                combined_lipid_pattern, 
                regex=True, 
                case=False, 
                na=False
            )
            
            # Check ClassKey for ISTD marker if column exists
            is_standard_class = pd.Series([False] * len(df_copy), index=df_copy.index)
            if 'ClassKey' in df_copy.columns:
                is_standard_class = df_copy['ClassKey'].str.contains(
                    r'ISTD|IS|Internal', 
                    regex=True, 
                    case=False, 
                    na=False
                )
            
            # Combine both checks
            is_standard = is_standard_lipid | is_standard_class
            
            if is_standard.any():
                internal_standards_df = df_copy[is_standard].copy().reset_index(drop=True)
                cleaned_df = df_copy[~is_standard].copy().reset_index(drop=True)
                
                n_standards = len(internal_standards_df)
                # Message removed - info already shown in Manage Standards section
                
                return cleaned_df, internal_standards_df
            else:
                # No internal standards found
                return df_copy, pd.DataFrame(columns=df_copy.columns)
                
        except Exception as e:
            st.error(f"Error extracting internal standards: {str(e)}")
            return df, pd.DataFrame(columns=df.columns)


def get_quality_filter_options():
    """
    Returns predefined quality filtering options for MS-DIAL data.
    
    These match the three-tier system defined in the integration plan:
    - Strict: Score >= 80, MS/MS required (publication-ready)
    - Moderate: Score >= 60, MS/MS not required (exploratory)
    - Permissive: Score >= 40, MS/MS not required (discovery)
    
    Returns:
        dict: Dictionary mapping option names to configuration dicts
    """
    return {
        'Strict (Score â‰¥80, MS/MS required)': {
            'total_score_threshold': 80,
            'require_msms': True
        },
        'Moderate (Score â‰¥60)': {
            'total_score_threshold': 60,
            'require_msms': False
        },
        'Permissive (Score â‰¥40)': {
            'total_score_threshold': 40,
            'require_msms': False
        },
        'No filtering': {
            'total_score_threshold': 0,
            'require_msms': False
        }
    }