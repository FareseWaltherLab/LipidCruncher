import pandas as pd
import re
import streamlit as st

class DataFormatHandler:
    """
    Handles data validation and standardization for different data formats.
    Works in conjunction with the Experiment class for sample management.
    All formats are standardized to use intensity[sample_name] columns.
    """
    
    @staticmethod
    def _infer_class_key(lipid_molec):
        """
        Infers ClassKey from LipidMolec naming convention.
        Expected format: Class(chain details)
        Example: CL(18:2/16:0/18:1/18:2) -> CL
        
        Args:
            lipid_molec (str): Lipid molecule identifier
            
        Returns:
            str: Inferred class key
        """
        try:
            # Extract everything before the first parenthesis
            match = re.match(r'^([^(]+)', lipid_molec)
            if match:
                return match.group(1).strip()
            return "Unknown"
        except:
            return "Unknown"
    
    @staticmethod
    def validate_and_preprocess(df, data_format):
        """
        Validates and standardizes data format.
        
        Args:
            df (pd.DataFrame): Input dataframe
            data_format (str): Either 'lipidsearch' or 'generic'
            
        Returns:
            tuple: (standardized_df, success, message)
        """
        try:
            # First validate the format
            if data_format == 'lipidsearch':
                success, message = DataFormatHandler._validate_lipidsearch(df)
                if not success:
                    return None, False, message
                    
                # Standardize LipidSearch format
                standardized_df = DataFormatHandler._standardize_lipidsearch(df)
                
            else:  # generic format
                success, message = DataFormatHandler._validate_generic(df)
                if not success:
                    return None, False, message
                    
                # Standardize generic format
                standardized_df = DataFormatHandler._standardize_generic(df)
            
            return standardized_df, True, "Data successfully standardized to generic format"
            
        except Exception as e:
            return None, False, f"Error during preprocessing: {str(e)}"
    
    @staticmethod
    def _validate_lipidsearch(df):
        """Validates LipidSearch format data."""
        required_cols = [
            'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
            'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey'
        ]
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
            
        # Check for MeanArea columns
        if not any(col.startswith('MeanArea[') for col in df.columns):
            return False, "No MeanArea columns found"
            
        return True, "Valid LipidSearch format"
    
    @staticmethod
    def _validate_generic(df):
        """
        Validates generic format data with case-insensitive column matching.
        Only requires LipidMolec and intensity columns.
        """
        # Create case-insensitive column mapping
        col_map = {col.lower(): col for col in df.columns}
        
        # Check for LipidMolec column
        if 'lipidmolec' not in col_map:
            return False, "Missing required column: LipidMolec"
            
        # Check for intensity columns - case insensitive
        has_intensity = any('intensity[' in col.lower().replace(' ', '') for col in df.columns)
        if not has_intensity:
            return False, "No intensity columns found"
        
        # Validate lipid naming format in LipidMolec column
        lipid_col = col_map['lipidmolec']
        # A more permissive regex that allows numbers in class names and modifications after chain details
        invalid_formats = df[lipid_col].apply(lambda x: not re.match(r'^[A-Za-z][A-Za-z0-9]*\([^()]*(?:\+[A-Za-z0-9]+)?[^()]*\)$', str(x)))
        if invalid_formats.any():
            # Check if we actually have invalid formats or if they're just nulls/empty strings
            real_invalids = df[lipid_col][invalid_formats].apply(lambda x: bool(str(x).strip()))
            if real_invalids.any():
                invalid_examples = df[lipid_col][real_invalids].head(3).tolist()
                return False, f"Invalid lipid naming format found. Examples: {', '.join(map(str, invalid_examples))}"
            
        return True, "Valid generic format"
    
    @staticmethod
    def _standardize_lipidsearch(df):
        """Standardizes LipidSearch format to use intensity[sample_name] columns."""
        df = df.copy()
        
        # Get all MeanArea columns and create new standardized names
        meanarea_cols = [col for col in df.columns if col.startswith('MeanArea[')]
        rename_dict = {}
        
        for col in meanarea_cols:
            # Extract sample name and create new column name
            sample_name = col[col.find('[')+1:col.find(']')]
            new_col = f'intensity[{sample_name}]'
            rename_dict[col] = new_col
            
        # Rename columns
        df = df.rename(columns=rename_dict)
        
        return df
    
    @staticmethod
    def _standardize_generic(df):
        """
        Standardizes generic format ensuring consistent intensity column naming
        and infers ClassKey from LipidMolec.
        """
        df = df.copy()
        
        # Create case-insensitive column mapping
        col_map = {col.lower(): col for col in df.columns}
        
        # Standardize LipidMolec
        if 'lipidmolec' in col_map:
            lipid_col = col_map['lipidmolec']
            if lipid_col != 'LipidMolec':
                df = df.rename(columns={lipid_col: 'LipidMolec'})
        
        # Infer ClassKey from LipidMolec
        df['ClassKey'] = df['LipidMolec'].apply(DataFormatHandler._infer_class_key)
        
        # Handle intensity columns with case insensitive matching
        intensity_cols = [
            col for col in df.columns 
            if 'intensity[' in col.lower().replace(' ', '')
        ]
        
        for col in intensity_cols:
            # Extract sample name ignoring case and spaces
            col_clean = col.lower().replace(' ', '')
            start_idx = col_clean.find('[') + 1
            end_idx = col_clean.find(']')
            
            if start_idx > 0 and end_idx > start_idx:
                sample_name = col[start_idx:end_idx].strip()
                new_col = f'intensity[{sample_name}]'
                if col != new_col:
                    df = df.rename(columns={col: new_col})
            
        return df