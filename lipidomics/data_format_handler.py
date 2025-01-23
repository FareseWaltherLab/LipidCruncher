import pandas as pd
import re
import io
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
        Standardizes lipid names to a consistent format: Class(chain details)(modifications)
        Examples:
            LPC O-17:4 -> LPC(O-17:4)
            Cer d18:0/C24:0 -> Cer(d18:0_C24:0)
            CE 14:0;0 -> CE(14:0)
            LPC 18:1(d7) -> LPC(18:1)(d7)
            CerG1(d13:0_25:2) -> CerG1(d13:0_25:2)
        """
        try:
            # Strip any whitespace and handle empty/null cases
            if not lipid_name or pd.isna(lipid_name):
                return "Unknown"
            
            lipid_name = str(lipid_name).strip()
            
            # Remove all ;0 suffixes
            lipid_name = re.sub(r';0(?=[\s/_)]|$)', '', lipid_name)
            
            # Extract deuteration or other modifications if present
            modification = ""
            mod_match = re.search(r'\(d\d+\)', lipid_name)
            if mod_match:
                modification = mod_match.group()
                lipid_name = lipid_name.replace(modification, '').strip()
            
            # If already in standard format Class(chains), just clean up the separators
            if '(' in lipid_name and ')' in lipid_name:
                class_name = lipid_name[:lipid_name.find('(')]
                chain_info = lipid_name[lipid_name.find('(')+1:lipid_name.find(')')]
                # Convert separators to underscore
                chain_info = chain_info.replace('/', '_')
                result = f"{class_name}({chain_info})"
            else:
                # Handle space-separated format
                if ' ' in lipid_name:
                    parts = lipid_name.split(maxsplit=1)
                    class_name = parts[0]
                    chain_info = parts[1]
                    result = f"{class_name}({chain_info})"
                else:
                    # If no obvious separation, try to find where class name ends
                    match = re.match(r'^([A-Za-z][A-Za-z0-9]*)[- ]?(.*)', lipid_name)
                    if match:
                        class_name, chain_info = match.groups()
                        chain_info = chain_info.replace('/', '_')
                        result = f"{class_name}({chain_info})"
                    else:
                        result = lipid_name
            
            # Add back modification if present
            if modification:
                result = f"{result}{modification}"
            
            return result
                
        except Exception as e:
            print(f"Error standardizing lipid name '{lipid_name}': {str(e)}")
            return lipid_name
    
    @staticmethod
    def _infer_class_key(lipid_molec):
        """
        Infers ClassKey from LipidMolec naming convention.
        Expected format: Class(chain details)(modifications)
        Examples:
            CL(18:2/16:0/18:1/18:2) -> CL
            LPC(18:1)(d7) -> LPC
        """
        try:
            # Extract everything before the first parenthesis or space
            match = re.match(r'^([A-Za-z][A-Za-z0-9]*)', lipid_molec)
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
            df (pd.DataFrame or str): Input dataframe or raw text for Metabolomics Workbench
            data_format (str): 'lipidsearch', 'generic', or 'metabolomics_workbench'
            
        Returns:
            tuple: (standardized_df, success, message)
        """
        try:
            # First validate the format
            if data_format == 'lipidsearch':
                success, message = DataFormatHandler._validate_lipidsearch(df)
                if not success:
                    return None, False, message
                standardized_df = DataFormatHandler._standardize_lipidsearch(df)
                
            elif data_format == 'Metabolomics Workbench':
                if isinstance(df, str):
                    success, message = DataFormatHandler._validate_metabolomics_workbench(df)
                    if not success:
                        return None, False, message
                    standardized_df = DataFormatHandler._standardize_metabolomics_workbench(df)
                else:
                    return None, False, "Invalid input type for Metabolomics Workbench format"
                
            else:  # generic format
                success, message = DataFormatHandler._validate_generic(df)
                if not success:
                    return None, False, message
                standardized_df = DataFormatHandler._standardize_generic(df)
            
            return standardized_df, True, "Data successfully standardized"
            
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
    
    @staticmethod
    def _validate_metabolomics_workbench(text_data):
        """
        Validates Metabolomics Workbench format text data.
        """
        try:
            # Check for required section markers
            if 'MS_METABOLITE_DATA_START' not in text_data or 'MS_METABOLITE_DATA_END' not in text_data:
                return False, "Missing required data section markers"
            
            # Extract data section
            start_idx = text_data.find('MS_METABOLITE_DATA_START')
            end_idx = text_data.find('MS_METABOLITE_DATA_END')
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                return False, "Invalid data section markers"
            
            # Get the data section
            data_lines = text_data[start_idx:end_idx].strip().split('\n')
            data_lines = [line.strip() for line in data_lines if line.strip()]
            
            # Need at least marker + samples + factors + one data row
            if len(data_lines) < 4:
                return False, "Insufficient data rows"
                
            return True, "Valid Metabolomics Workbench format"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def _standardize_metabolomics_workbench(text_data):
        """
        Standardizes Metabolomics Workbench format data.
        """
        try:
            # Find the data section
            start_idx = text_data.find('MS_METABOLITE_DATA_START')
            end_idx = text_data.find('MS_METABOLITE_DATA_END')
            
            # Extract and split the data section into lines
            data_section = text_data[start_idx:end_idx].strip().split('\n')
            data_section = [line.strip() for line in data_section if line.strip()]
            
            # Remove the MS_METABOLITE_DATA_START line
            data_section = [line for line in data_section if 'MS_METABOLITE_DATA_START' not in line]
            
            # Parse header rows and split by comma since file is CSV
            samples_line = data_section[0].split(',')
            factors_line = data_section[1].split(',')
            
            # Extract sample names and conditions, skipping the first column header
            sample_names = samples_line[1:]  # Skip 'Samples' column
            conditions = factors_line[1:]    # Skip 'Factors' column
            
            # Create list of data rows
            data_rows = []
            for line in data_section[2:]:  # Skip header rows
                if line.strip():  # Ignore empty lines
                    values = line.split(',')  # Split by comma
                    if len(values) == len(sample_names) + 1:  # +1 for lipid name column
                        data_rows.append(values)
            
            # Create DataFrame
            columns = ['LipidMolec'] + [f'intensity[s{i+1}]' for i in range(len(sample_names))]
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Store experimental conditions and sample names in session state
            st.session_state.workbench_conditions = {
                f's{i+1}': condition.strip() 
                for i, condition in enumerate(conditions)
            }
            st.session_state.workbench_samples = {
                f's{i+1}': name.strip() 
                for i, name in enumerate(sample_names)
            }
            
            # Clean and standardize lipid names
            df['LipidMolec'] = df['LipidMolec'].apply(DataFormatHandler._standardize_lipid_name)
            
            # Infer ClassKey from standardized LipidMolec
            df['ClassKey'] = df['LipidMolec'].apply(DataFormatHandler._infer_class_key)
            
            # Convert intensity columns to numeric, replacing any non-numeric values with NaN
            intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
            for col in intensity_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Replace NaN with 0
            df[intensity_cols] = df[intensity_cols].fillna(0)
            
            return df
            
        except Exception as e:
            st.error(f"Error in standardize_metabolomics_workbench: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _parse_workbench_conditions(conditions):
        """
        Parses condition strings from Metabolomics Workbench format.
        Example: "Diet:Normal | BileAcid:water" -> {'Diet': 'Normal', 'BileAcid': 'water'}
        """
        parsed_conditions = {}
        for condition in conditions:
            if isinstance(condition, str):
                factors = condition.split('|')
                for factor in factors:
                    factor = factor.strip()
                    if ':' in factor:
                        key, value = factor.split(':')
                        key = key.strip()
                        value = value.strip()
                        if key not in parsed_conditions:
                            parsed_conditions[key] = []
                        if value not in parsed_conditions[key]:
                            parsed_conditions[key].append(value)
        
        return parsed_conditions