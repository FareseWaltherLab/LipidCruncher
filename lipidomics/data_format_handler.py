import pandas as pd
import streamlit as st

class DataFormatHandler:
    """
    Handles data validation and standardization for different data formats.
    Works in conjunction with the Experiment class for sample management.
    All formats are standardized to use intensity[sample_name] columns.
    """
    
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
            
            return standardized_df, True, f"Data successfully standardized to generic format"
            
        except Exception as e:
            return None, False, f"Error during preprocessing: {str(e)}"
    
    @staticmethod
    def _validate_lipidsearch(df):
        """
        Validates LipidSearch format data.
        
        Returns:
            tuple: (success, message)
        """
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
        Validates generic format data.
        
        Returns:
            tuple: (success, message)
        """
        required_cols = ['LipidMolec', 'ClassKey']
        
        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
            
        # Check for intensity columns - case insensitive
        intensity_cols = [col for col in df.columns if 'intensity[' in col.lower()]
        if not intensity_cols:
            return False, "No intensity columns found"
            
        return True, "Valid generic format"
    
    @staticmethod
    def _standardize_lipidsearch(df):
        """
        Standardizes LipidSearch format to use intensity[sample_name] columns.
        """
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
        Standardizes generic format ensuring consistent intensity column naming.
        """
        df = df.copy()
        
        # Get all intensity columns (case insensitive)
        intensity_cols = [col for col in df.columns if 'intensity[' in col.lower()]
        rename_dict = {}
        
        for col in intensity_cols:
            # Extract sample name and create standardized name
            sample_name = col[col.lower().find('[')+1:col.lower().find(']')]
            new_col = f'intensity[{sample_name}]'
            if col != new_col:
                rename_dict[col] = new_col
                
        # Rename columns if needed
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
        return df
    
    @staticmethod
    def update_sample_columns(df, experiment):
        """
        Updates intensity column names to match the experiment's sample list.
        This ensures column names align with the Experiment class sample management.
        
        Args:
            df (pd.DataFrame): Input dataframe with standardized intensity columns
            experiment (Experiment): Experiment object with sample information
            
        Returns:
            pd.DataFrame: DataFrame with updated column names
        """
        df = df.copy()
        
        # Get current intensity columns
        current_cols = [col for col in df.columns if col.startswith('intensity[')]
        
        # Create mapping to experiment sample names
        rename_dict = {}
        for i, sample in enumerate(experiment.full_samples_list):
            old_col = current_cols[i] if i < len(current_cols) else None
            if old_col and old_col != f'intensity[{sample}]':
                rename_dict[old_col] = f'intensity[{sample}]'
        
        # Rename columns if needed
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        return df