"""
Format preprocessing service for lipid data.
Handles validation and standardization of different data formats (LipidSearch, Generic, Metabolomics Workbench).
Pure logic - no UI dependencies.
"""
import pandas as pd
import re
from typing import Tuple, Optional


class FormatPreprocessingService:
    """
    Service for preprocessing and standardizing different lipidomic data formats.
    Converts all formats to a common structure with intensity[sample] columns.
    """
    
    def validate_and_preprocess(
        self,
        df: pd.DataFrame,
        format_type: str
    ) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """
        Validate and preprocess data based on format type.
        
        Args:
            df: Input DataFrame (or string for Metabolomics Workbench)
            format_type: One of 'lipidsearch', 'generic', 'metabolomics_workbench'
        
        Returns:
            Tuple of (preprocessed_df, success, message)
        """
        try:
            if format_type == 'lipidsearch':
                return self._process_lipidsearch(df)
            elif format_type == 'generic':
                return self._process_generic(df)
            elif format_type == 'metabolomics_workbench':
                if isinstance(df, str):
                    return self._process_metabolomics_workbench(df)
                else:
                    return None, False, "Metabolomics Workbench format requires text input"
            else:
                return None, False, f"Unknown format type: {format_type}"
                
        except Exception as e:
            return None, False, f"Error during preprocessing: {str(e)}"
    
    # ============================================================================
    # LIPIDSEARCH FORMAT
    # ============================================================================
    
    def _process_lipidsearch(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """Process LipidSearch format data."""
        # Validate
        success, message = self._validate_lipidsearch(df)
        if not success:
            return None, False, message
        
        # Standardize
        standardized_df = self._standardize_lipidsearch(df)
        return standardized_df, True, f"LipidSearch data preprocessed: {len(standardized_df)} lipids"
    
    def _validate_lipidsearch(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate LipidSearch format has required columns."""
        required_cols = [
            'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
            'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Check for MeanArea columns
        if not any(col.startswith('MeanArea[') for col in df.columns):
            return False, "No MeanArea[] columns found"
        
        return True, "Valid LipidSearch format"
    
    def _standardize_lipidsearch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize LipidSearch format.
        Renames MeanArea[sample] → intensity[sample] columns.
        """
        df = df.copy()
        
        # Find all MeanArea columns
        meanarea_cols = [col for col in df.columns if col.startswith('MeanArea[')]
        
        # Create rename mapping
        rename_dict = {}
        for col in meanarea_cols:
            # Extract sample name: MeanArea[s1] → s1
            sample_name = col[col.find('[')+1:col.find(']')]
            rename_dict[col] = f'intensity[{sample_name}]'
        
        # Rename columns
        df = df.rename(columns=rename_dict)
        
        return df
    
    # ============================================================================
    # GENERIC FORMAT
    # ============================================================================
    
    def _process_generic(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """Process Generic format data."""
        # Validate
        success, message = self._validate_generic(df)
        if not success:
            return None, False, message
        
        # Standardize
        standardized_df, column_mapping = self._standardize_generic(df)
        return standardized_df, True, f"Generic data preprocessed: {len(standardized_df)} lipids"
    
    def _validate_generic(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate generic format."""
        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns: lipid names and one intensity column"
        
        # First column should contain letters (lipid names)
        if not self._contains_letters(df.iloc[:, 0]):
            return False, "First column doesn't appear to contain lipid names (no letters found)"
        
        return True, "Valid generic format"
    
    def _standardize_generic(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize generic format.
        - First column → 'LipidMolec'
        - Other columns → intensity[s1], intensity[s2], ...
        - Adds ClassKey inferred from lipid names
        
        Returns:
            Tuple of (standardized_df, column_mapping_df)
        """
        df = df.copy()
        
        # Store original column names for mapping
        original_cols = df.columns.tolist()
        standardized_cols = ['LipidMolec'] + [f'intensity[s{i}]' for i in range(1, len(original_cols))]
        
        # Create mapping DataFrame
        column_mapping = pd.DataFrame({
            'original_name': original_cols,
            'standardized_name': standardized_cols
        })
        
        # Rename columns
        df.columns = standardized_cols
        
        # Standardize lipid names
        df['LipidMolec'] = df['LipidMolec'].apply(self._standardize_lipid_name)
        
        # Infer ClassKey from lipid names
        df['ClassKey'] = df['LipidMolec'].apply(self._infer_class_key)
        
        return df, column_mapping
    
    # ============================================================================
    # METABOLOMICS WORKBENCH FORMAT
    # ============================================================================
    
    def _process_metabolomics_workbench(
        self,
        text_data: str
    ) -> Tuple[Optional[pd.DataFrame], bool, str]:
        """Process Metabolomics Workbench format data."""
        # Validate
        success, message = self._validate_metabolomics_workbench(text_data)
        if not success:
            return None, False, message
        
        # Standardize
        standardized_df, conditions_map = self._standardize_metabolomics_workbench(text_data)
        return standardized_df, True, f"Metabolomics Workbench data preprocessed: {len(standardized_df)} lipids"
    
    def _validate_metabolomics_workbench(self, text_data: str) -> Tuple[bool, str]:
        """Validate Metabolomics Workbench format."""
        try:
            # Check for required section markers
            if 'MS_METABOLITE_DATA_START' not in text_data or 'MS_METABOLITE_DATA_END' not in text_data:
                return False, "Missing required data section markers (MS_METABOLITE_DATA_START/END)"
            
            # Extract data section
            start_idx = text_data.find('MS_METABOLITE_DATA_START')
            end_idx = text_data.find('MS_METABOLITE_DATA_END')
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                return False, "Invalid data section markers"
            
            # Get data lines
            data_lines = text_data[start_idx:end_idx].strip().split('\n')
            data_lines = [line.strip() for line in data_lines if line.strip()]
            
            # Need: marker + samples + factors + at least one data row
            if len(data_lines) < 4:
                return False, "Insufficient data rows"
            
            return True, "Valid Metabolomics Workbench format"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _standardize_metabolomics_workbench(
        self,
        text_data: str
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Standardize Metabolomics Workbench format.
        
        Returns:
            Tuple of (standardized_df, conditions_mapping_dict)
        """
        try:
            # Find data section
            start_idx = text_data.find('MS_METABOLITE_DATA_START')
            end_idx = text_data.find('MS_METABOLITE_DATA_END')
            
            # Extract and clean data lines
            data_section = text_data[start_idx:end_idx].strip().split('\n')
            data_section = [line.strip() for line in data_section if line.strip()]
            data_section = [line for line in data_section if 'MS_METABOLITE_DATA_START' not in line]
            
            # Parse header rows (CSV format)
            samples_line = data_section[0].split(',')
            factors_line = data_section[1].split(',')
            
            # Extract sample names and conditions (skip first column header)
            sample_names = samples_line[1:]
            conditions = factors_line[1:]
            
            # Clean condition names (remove "Condition:" prefix)
            cleaned_conditions = []
            for condition in conditions:
                condition = condition.strip()
                if condition.startswith('Condition:'):
                    condition = condition.replace('Condition:', '').strip()
                cleaned_conditions.append(condition)
            
            # Parse data rows
            data_rows = []
            for line in data_section[2:]:
                if line.strip():
                    values = line.split(',')
                    if len(values) == len(sample_names) + 1:
                        data_rows.append(values)
            
            # Create DataFrame
            columns = ['LipidMolec'] + [f'intensity[s{i+1}]' for i in range(len(sample_names))]
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Create conditions mapping
            conditions_map = {
                f's{i+1}': condition
                for i, condition in enumerate(cleaned_conditions)
            }
            
            # Standardize lipid names
            df['LipidMolec'] = df['LipidMolec'].apply(self._standardize_lipid_name)
            
            # Infer ClassKey
            df['ClassKey'] = df['LipidMolec'].apply(self._infer_class_key)
            
            # Convert intensity columns to numeric
            intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
            for col in intensity_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN with 0
            df[intensity_cols] = df[intensity_cols].fillna(0)
            
            return df, conditions_map
            
        except Exception as e:
            raise ValueError(f"Error standardizing Metabolomics Workbench data: {str(e)}")
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _standardize_lipid_name(self, lipid_name: str) -> str:
        """
        Standardize lipid names to consistent format: Class(chain_details)(modifications)
        
        Examples:
            'LPC O-17:4' → 'LPC(O-17:4)'
            'Cer d18:0/C24:0' → 'Cer(d18:0_C24:0)'
            'CE 14:0;0' → 'CE(14:0)'
        """
        try:
            # Handle empty/null cases
            if not lipid_name or pd.isna(lipid_name):
                return "Unknown"
            
            lipid_name = str(lipid_name).strip()
            
            # Remove ;N suffixes (oxidation state markers)
            lipid_name = re.sub(r';[0-9]+(?=[\s/_)]|$)', '', lipid_name)
            
            # Extract modifications like (d7)
            modification = ""
            mod_match = re.search(r'\(d\d+\)', lipid_name)
            if mod_match:
                modification = mod_match.group()
                lipid_name = lipid_name.replace(modification, '').strip()
            
            # If already in standard format Class(chains)
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
                    chain_info = parts[1] if len(parts) > 1 else ''
                    chain_info = chain_info.replace('/', '_')
                    result = f"{class_name}({chain_info})"
                else:
                    # Try to find where class name ends
                    match = re.match(r'^([A-Za-z][A-Za-z0-9]*)[- ]?(.*)', lipid_name)
                    if match:
                        class_name, chain_info = match.groups()
                        chain_info = chain_info.replace('/', '_')
                        result = f"{class_name}({chain_info})"
                    else:
                        result = lipid_name
            
            # Add back modification
            if modification:
                result = f"{result}{modification}"
            
            return result
            
        except Exception:
            return lipid_name
    
    def _infer_class_key(self, lipid_molec: str) -> str:
        """
        Infer ClassKey from LipidMolec.
        Extracts everything before the first parenthesis or space.
        
        Examples:
            'CL(18:2/16:0/18:1/18:2)' → 'CL'
            'LPC(18:1)(d7)' → 'LPC'
        """
        try:
            match = re.match(r'^([A-Za-z][A-Za-z0-9]*)', lipid_molec)
            if match:
                return match.group(1).strip()
            return "Unknown"
        except:
            return "Unknown"
    
    def _contains_letters(self, column: pd.Series) -> bool:
        """Check if a column contains any letters."""
        return column.astype(str).str.contains('[A-Za-z]').any()