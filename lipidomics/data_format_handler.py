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
    def _standardize_lipid_name(lipid_name):
        """
        Standardizes lipid names to a consistent format: Class(chain details with underscore separator)
        Examples:
            LPC O-17:4 -> LPC(O-17:4)
            Cer d18:0/C24:0 -> Cer(d18:0_C24:0)
            CE 14:0;0 -> CE(14:0)
            DAG 10:0;0_16:0;0 -> DAG(10:0_16:0)
            CerG1(d13:0_25:2) -> CerG1(d13:0_25:2)
        """
        try:
            # Strip any whitespace and handle empty/null cases
            if not lipid_name or pd.isna(lipid_name):
                return "Unknown"
            
            lipid_name = str(lipid_name).strip()
            
            # Remove all ;0 suffixes, not just at the end
            lipid_name = re.sub(r';0(?=[\s/_)]|$)', '', lipid_name)
            
            # If already in standard format Class(chains), just clean up the separators
            if '(' in lipid_name and ')' in lipid_name:
                class_name = lipid_name[:lipid_name.find('(')]
                chain_info = lipid_name[lipid_name.find('(')+1:lipid_name.find(')')]
                # Convert separators to underscore
                chain_info = chain_info.replace('/', '_')
                return f"{class_name}({chain_info})"
            
            # Handle space-separated format (e.g., "CE 14:0" or "LPC O-17:4")
            if ' ' in lipid_name:
                parts = lipid_name.split(maxsplit=1)
                class_name = parts[0]
                chain_info = parts[1]
                return f"{class_name}({chain_info})"
            
            # If no obvious separation, try to find where class name ends
            match = re.match(r'^([A-Za-z][A-Za-z0-9]*)[- ]?(.*)', lipid_name)
            if match:
                class_name, chain_info = match.groups()
                chain_info = chain_info.replace('/', '_')
                return f"{class_name}({chain_info})"
            
            return lipid_name
            
        except Exception as e:
            print(f"Error standardizing lipid name '{lipid_name}': {str(e)}")
            return lipid_name

    @staticmethod
    def _infer_class_key(lipid_molec):
        """
        Infers ClassKey from LipidMolec naming convention.
        Expected format: Class(chain details)
        Example: CL(18:2/16:0/18:1/18:2) -> CL
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
    def _contains_letters(column):
        """Check if a column contains any letters in its values."""
        return column.astype(str).str.contains('[A-Za-z]').any()
    
    @staticmethod
    def _validate_generic(df):
        """
        Validates generic format data.
        First column should contain letters (lipid names).
        """
        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns: lipid names and one intensity column"
            
        # First column should contain letters (lipid names)
        if not DataFormatHandler._contains_letters(df.iloc[:, 0]):
            return False, "First column doesn't appear to contain lipid names (no letters found)."
            
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
        Standardizes generic format:
        - First column becomes 'LipidMolec'
        - All other columns become intensity[s1], intensity[s2], etc.
        Also stores the column name mapping in session state.
        """
        df = df.copy()
        
        # Create mapping dictionary for column names
        original_cols = df.columns.tolist()
        standardized_cols = ['LipidMolec']  # First column becomes LipidMolec
        
        # Generate standardized names for intensity columns
        for i in range(1, len(original_cols)):
            standardized_cols.append(f'intensity[s{i}]')
        
        # Create and store the mapping DataFrame
        mapping_df = pd.DataFrame({
            'original_name': original_cols,
            'standardized_name': standardized_cols
        })
        
        # Store in session state for display
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = mapping_df
            
        # Store original number of intensity columns for later validation
        if 'n_intensity_cols' not in st.session_state:
            st.session_state.n_intensity_cols = len(original_cols) - 1
        
        # Rename columns
        df.columns = standardized_cols
        
        # Standardize lipid names in LipidMolec column
        df['LipidMolec'] = df['LipidMolec'].apply(DataFormatHandler._standardize_lipid_name)
        
        # Infer ClassKey from standardized LipidMolec
        df['ClassKey'] = df['LipidMolec'].apply(DataFormatHandler._infer_class_key)
        
        return df