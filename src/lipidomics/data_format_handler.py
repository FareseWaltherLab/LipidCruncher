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
    
    # MS-DIAL known metadata columns (to exclude from sample detection)
    MSDIAL_METADATA_COLUMNS = {
        'Alignment ID', 'Average Rt(min)', 'Average Mz', 'Metabolite name',
        'Adduct type', 'Post curation result', 'Fill %', 'MS/MS assigned',
        'Reference RT', 'Reference m/z', 'Formula', 'Ontology', 'INCHIKEY',
        'SMILES', 'Annotation tag', 'Annotation tag (VS1.0)', 'RT matched', 
        'm/z matched', 'MS/MS matched', 'Comment', 'Manually modified',
        'Isotope tracking parent ID', 'Isotope tracking weight number',
        'Total score', 'RT similarity', 'Dot product', 'Reverse dot product',
        'Fragment presence %', 'S/N average', 'Spectrum reference file name',
        'MS1 isotopic spectrum', 'MS/MS spectrum', 'Lipid IS'
    }
    
    @staticmethod
    def _standardize_lipid_name(lipid_name):
        """
        Standardizes lipid names to a consistent format: Class(chain details)(modifications)
        
        Preserves hydroxyl notation (;O, ;2O, ;3O) which is biologically meaningful.
        
        Examples:
            LPC O-17:4 -> LPC(O-17:4)
            Cer d18:0/C24:0 -> Cer(d18:0_C24:0)
            CE 14:0;0 -> CE(14:0)
            LPC 18:1(d7) -> LPC(18:1)(d7)
            CerG1(d13:0_25:2) -> CerG1(d13:0_25:2)
            Cer 18:1;2O/24:0 -> Cer(18:1;2O_24:0)  # MS-DIAL format with hydroxyl
            Cer 18:0;3O/24:0;(2OH) -> Cer(18:0;3O_24:0;(2OH))  # Complex hydroxylation
        """
        try:
            # Strip any whitespace and handle empty/null cases
            if not lipid_name or pd.isna(lipid_name):
                return "Unknown"
            
            lipid_name = str(lipid_name).strip()
            
            # Remove ;N suffixes ONLY when N is a plain number (not followed by O)
            # This preserves ;2O, ;3O (hydroxyl groups) but removes ;0, ;1, etc.
            # Pattern: semicolon + digits + NOT followed by 'O'
            lipid_name = re.sub(r';(\d+)(?!O)(?=[\s/_)]|$)', '', lipid_name)
            
            # Extract deuteration modifications like (d7), (d9) at the END of the name
            modification = ""
            mod_match = re.search(r'\(d\d+\)$', lipid_name)
            if mod_match:
                modification = mod_match.group()
                lipid_name = lipid_name[:mod_match.start()].strip()
            
            # Check if already in standard format Class(chains) 
            # BUT be careful with embedded parentheses like ;(2OH)
            # We check if there's a ( that comes BEFORE any chain info indicators
            first_paren = lipid_name.find('(')
            first_space = lipid_name.find(' ')
            first_colon = lipid_name.find(':')
            
            # If lipid already has parentheses around chain info (not embedded like ;(2OH))
            # Standard format: Class(chains) where ( appears before : or at start of chain info
            if first_paren > 0 and (first_colon < 0 or first_paren < first_colon):
                # Already has parentheses that look like class wrapper
                class_name = lipid_name[:first_paren]
                # Find the matching closing paren
                paren_depth = 0
                chain_end = first_paren
                for i in range(first_paren, len(lipid_name)):
                    if lipid_name[i] == '(':
                        paren_depth += 1
                    elif lipid_name[i] == ')':
                        paren_depth -= 1
                        if paren_depth == 0:
                            chain_end = i
                            break
                chain_info = lipid_name[first_paren+1:chain_end]
                # Get any trailing info (like additional modifications)
                trailing = lipid_name[chain_end+1:].strip()
                # Convert separators to underscore
                chain_info = chain_info.replace('/', '_')
                result = f"{class_name}({chain_info})"
                if trailing:
                    result = f"{result}{trailing}"
            else:
                # Handle space-separated format (common in MS-DIAL)
                # Format: "Class chain1/chain2" or "Class chain1/chain2;(mod)"
                if first_space > 0:
                    class_name = lipid_name[:first_space]
                    chain_info = lipid_name[first_space+1:]
                    # Convert / to _ for chain separation
                    chain_info = chain_info.replace('/', '_')
                    result = f"{class_name}({chain_info})"
                else:
                    # If no obvious separation, try to find where class name ends
                    match = re.match(r'^([A-Za-z][A-Za-z0-9]*)[- ]?(.*)', lipid_name)
                    if match:
                        class_name, chain_info = match.groups()
                        chain_info = chain_info.replace('/', '_')
                        if chain_info:
                            result = f"{class_name}({chain_info})"
                        else:
                            result = class_name
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
            data_format (str): 'lipidsearch', 'generic', 'metabolomics_workbench', or 'msdial'
            
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
            
            elif data_format == 'msdial':
                success, message = DataFormatHandler._validate_msdial(df)
                if not success:
                    return None, False, message
                standardized_df = DataFormatHandler._standardize_msdial(df)
                
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
    
    # =========================================================================
    # MS-DIAL Format Methods
    # =========================================================================
    
    @staticmethod
    def _detect_msdial_header_row(df):
        """
        Auto-detect where the actual data headers start in MS-DIAL format.
        
        MS-DIAL exports have metadata rows at the top (variable number, often 1-9)
        containing sample-level info like Category, Tissue, Genotype, etc.
        The actual column headers appear in the row where 'Alignment ID' is found
        as text, followed by data rows with numeric Alignment IDs.
        
        Returns:
            int: Index of the header row (0-based), or 0 if no metadata detected
        """
        # Check if first column header is already 'Alignment ID' (no metadata rows)
        if df.columns[0] == 'Alignment ID':
            return -1  # Headers are already in column names, no extra header row
        
        # Scan rows to find where headers are
        for idx in range(min(20, len(df))):  # Check first 20 rows max
            row = df.iloc[idx]
            first_val = str(row.iloc[0]).strip()
            
            # The header row will have "Alignment ID" as text in first column
            if first_val == 'Alignment ID':
                return idx
            
            # Alternative: check for 'Metabolite name' which is more unique
            if 'Metabolite name' in row.values:
                return idx
        
        # If we didn't find explicit headers, check if data looks valid already
        # (i.e., first row might be data if columns are correct)
        return -1  # Assume no metadata header rows
    
    @staticmethod
    def _detect_msdial_sample_columns(df, columns=None):
        """
        Detect sample intensity columns in MS-DIAL data.
        
        Sample columns are those that:
        1. Are NOT in the known metadata columns set
        2. Contain numeric data
        3. Are NOT the 'Lipid IS' column or columns after it (if normalized data exists)
        
        Args:
            df: DataFrame to analyze (should have correct headers as columns)
            columns: Optional list of column names (if different from df.columns)
            
        Returns:
            tuple: (raw_sample_cols, normalized_sample_cols, lipid_is_col_idx)
        """
        raw_cols, norm_cols, lipid_is_idx, _, _ = \
            DataFormatHandler._detect_msdial_sample_columns_with_indices(df, columns)
        return raw_cols, norm_cols, lipid_is_idx
    
    @staticmethod
    def _detect_msdial_sample_columns_with_indices(df, columns=None):
        """
        Detect sample intensity columns in MS-DIAL data, returning both names and indices.
        
        This handles the case where sample columns have duplicate names (raw vs normalized).
        
        Args:
            df: DataFrame to analyze (should have correct headers as columns)
            columns: Optional list of column names (if different from df.columns)
            
        Returns:
            tuple: (raw_sample_cols, normalized_sample_cols, lipid_is_col_idx, 
                   raw_sample_indices, normalized_sample_indices)
        """
        if columns is None:
            columns = df.columns.tolist()
        
        raw_sample_cols = []
        normalized_sample_cols = []
        raw_sample_indices = []
        normalized_sample_indices = []
        lipid_is_idx = None
        
        # Find the 'Lipid IS' column which separates raw and normalized data
        for idx, col in enumerate(columns):
            if col == 'Lipid IS':
                lipid_is_idx = idx
                break
        
        # Iterate through columns BY INDEX to handle duplicates properly
        for idx, col in enumerate(columns):
            # Skip known metadata columns
            if col in DataFormatHandler.MSDIAL_METADATA_COLUMNS:
                continue
            
            # Skip 'Lipid IS' column itself
            if col == 'Lipid IS':
                continue
            
            # Skip NaN or empty column names
            if pd.isna(col) or (isinstance(col, str) and not col.strip()):
                continue
                
            # Check if column contains numeric data (sample intensity)
            # Use iloc to access by position to avoid duplicate column issues
            try:
                col_data = df.iloc[:, idx]
                # Try to determine if this is a sample column
                # Sample columns should have mostly numeric values
                numeric_values = pd.to_numeric(col_data, errors='coerce')
                numeric_count = numeric_values.notna().sum()
                
                if numeric_count > len(col_data) * 0.5:  # >50% numeric
                    if lipid_is_idx is not None:
                        if idx < lipid_is_idx:
                            raw_sample_cols.append(col)
                            raw_sample_indices.append(idx)
                        elif idx > lipid_is_idx:
                            normalized_sample_cols.append(col)
                            normalized_sample_indices.append(idx)
                    else:
                        # No Lipid IS column found, assume all are raw
                        raw_sample_cols.append(col)
                        raw_sample_indices.append(idx)
            except Exception as e:
                # Skip columns that cause errors
                continue
        
        return raw_sample_cols, normalized_sample_cols, lipid_is_idx, raw_sample_indices, normalized_sample_indices
    
    @staticmethod
    def _validate_msdial(df):
        """
        Validates MS-DIAL format with flexible column detection.
        
        Required: 
            - Metabolite name column (or similar lipid identifier)
            - At least one sample intensity column
            
        Optional (will use if present, gracefully degrade if not):
            - Ontology (for ClassKey)
            - Total score, MS/MS matched (for quality filtering)
            - Average Rt(min), Average Mz (for additional info)
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # First, detect if there are metadata header rows
            header_row_idx = DataFormatHandler._detect_msdial_header_row(df)
            
            # Get the actual column names
            if header_row_idx >= 0:
                # Headers are in a data row, need to extract them
                actual_columns = df.iloc[header_row_idx].tolist()
            else:
                # Headers are already the column names
                actual_columns = df.columns.tolist()
            
            # Check for lipid name column (required)
            lipid_col = None
            lipid_col_idx = None
            possible_lipid_cols = ['Metabolite name', 'Metabolite', 'Name', 'Lipid', 'LipidMolec']
            
            for col_name in possible_lipid_cols:
                if col_name in actual_columns:
                    lipid_col = col_name
                    lipid_col_idx = actual_columns.index(col_name)
                    break
            
            if lipid_col is None:
                return False, "No lipid name column found. Expected 'Metabolite name' or similar column."
            
            # Detect sample columns - now returns indices too
            # For validation, we need to work with the actual data
            if header_row_idx >= 0:
                # Create a temporary df with correct headers for analysis
                temp_df = df.iloc[header_row_idx + 1:].copy()
                temp_df.columns = actual_columns
            else:
                temp_df = df
            
            raw_samples, norm_samples, lipid_is_idx, raw_indices, norm_indices = \
                DataFormatHandler._detect_msdial_sample_columns_with_indices(
                    temp_df, actual_columns
                )
            
            total_samples = len(raw_samples) + len(norm_samples)
            if total_samples < 1:
                return False, "No sample intensity columns detected. Check that your export includes sample data."
            
            # Build column index mapping for metadata columns
            col_indices = {}
            for idx, col in enumerate(actual_columns):
                if col not in col_indices:  # Only store first occurrence for metadata
                    col_indices[col] = idx
            
            # Build info about available optional columns for later use
            available_features = {
                'has_ontology': 'Ontology' in actual_columns,
                'has_quality_score': 'Total score' in actual_columns,
                'has_msms_matched': 'MS/MS matched' in actual_columns,
                'has_rt': 'Average Rt(min)' in actual_columns,
                'has_mz': 'Average Mz' in actual_columns,
                'has_normalized_data': len(norm_samples) > 0,
                'lipid_column': lipid_col,
                'lipid_column_index': lipid_col_idx,
                'raw_sample_columns': raw_samples,
                'raw_sample_indices': raw_indices,
                'normalized_sample_columns': norm_samples,
                'normalized_sample_indices': norm_indices,
                'header_row_index': header_row_idx,
                'actual_columns': actual_columns,
                'column_indices': col_indices
            }
            
            # Store for use in standardization
            st.session_state.msdial_features = available_features
            
            # Build informative message
            msg_parts = [f"Valid MS-DIAL format: {len(raw_samples)} samples detected"]
            if len(norm_samples) > 0:
                msg_parts.append(f"(+ {len(norm_samples)} normalized columns available)")
            if not available_features['has_ontology']:
                msg_parts.append("Note: No 'Ontology' column - ClassKey will be inferred from lipid names")
            if not available_features['has_quality_score']:
                msg_parts.append("Note: No 'Total score' column - quality filtering unavailable")
            
            return True, " | ".join(msg_parts)
            
        except Exception as e:
            return False, f"MS-DIAL validation error: {str(e)}"
    
    @staticmethod
    def _standardize_msdial(df):
        """
        Standardizes MS-DIAL format to LipidCruncher standard format.
        
        Transforms:
            - Metabolite name -> LipidMolec (with standardized naming)
            - Ontology -> ClassKey (or inferred from lipid name)
            - Average Rt(min) -> BaseRt
            - Average Mz -> CalcMass
            - Sample columns -> intensity[s1], intensity[s2], etc.
            
        Returns:
            pd.DataFrame: Standardized dataframe
        """
        try:
            # Retrieve the features detected during validation
            features = st.session_state.get('msdial_features', {})
            
            header_row_idx = features.get('header_row_index', -1)
            actual_columns = features.get('actual_columns', df.columns.tolist())
            lipid_col = features.get('lipid_column', 'Metabolite name')
            lipid_col_idx = features.get('lipid_column_index', 3)  # Default index for 'Metabolite name'
            raw_sample_cols = features.get('raw_sample_columns', [])
            raw_sample_indices = features.get('raw_sample_indices', [])
            norm_sample_cols = features.get('normalized_sample_columns', [])
            norm_sample_indices = features.get('normalized_sample_indices', [])
            col_indices = features.get('column_indices', {})
            
            # Skip metadata rows if present
            if header_row_idx >= 0:
                # Store metadata for potential future use
                metadata_df = df.iloc[:header_row_idx].copy()
                st.session_state.msdial_metadata = metadata_df
                
                # Get actual data rows - DO NOT set columns (use iloc instead)
                data_df = df.iloc[header_row_idx + 1:].copy()
            else:
                data_df = df.copy()
            
            # Reset index
            data_df = data_df.reset_index(drop=True)
            
            # Read from the widget key directly (msdial_data_type_selection) instead of the manually-set value
            # This ensures we get the current selection, not the previous run's value
            data_type_selection = st.session_state.get('msdial_data_type_selection', f"Raw intensity values ({len(raw_sample_cols)} samples)")
            use_normalized = "Pre-normalized" in data_type_selection
            
            if use_normalized and len(norm_sample_cols) > 0:
                sample_cols_to_use = norm_sample_cols
                sample_indices_to_use = norm_sample_indices
                st.session_state.msdial_data_type = 'normalized'
            else:
                sample_cols_to_use = raw_sample_cols
                sample_indices_to_use = raw_sample_indices
                st.session_state.msdial_data_type = 'raw'
            
            # Build the standardized dataframe using POSITIONAL indexing (iloc)
            standardized_data = {}
            
            # LipidMolec - standardize lipid names (use positional index)
            standardized_data['LipidMolec'] = data_df.iloc[:, lipid_col_idx].apply(
                DataFormatHandler._standardize_lipid_name
            )
            
            # ClassKey - use Ontology if available, otherwise infer
            if features.get('has_ontology', False) and 'Ontology' in col_indices:
                ontology_idx = col_indices['Ontology']
                standardized_data['ClassKey'] = data_df.iloc[:, ontology_idx].fillna('Unknown')
            else:
                # Infer from standardized lipid names
                standardized_data['ClassKey'] = standardized_data['LipidMolec'].apply(
                    DataFormatHandler._infer_class_key
                )
            
            # Optional columns - use positional indexing
            if features.get('has_rt', False) and 'Average Rt(min)' in col_indices:
                rt_idx = col_indices['Average Rt(min)']
                standardized_data['BaseRt'] = pd.to_numeric(
                    data_df.iloc[:, rt_idx], errors='coerce'
                )
            
            if features.get('has_mz', False) and 'Average Mz' in col_indices:
                mz_idx = col_indices['Average Mz']
                standardized_data['CalcMass'] = pd.to_numeric(
                    data_df.iloc[:, mz_idx], errors='coerce'
                )
            
            # Store quality columns for potential filtering
            if features.get('has_quality_score', False) and 'Total score' in col_indices:
                score_idx = col_indices['Total score']
                st.session_state.msdial_quality_scores = pd.to_numeric(
                    data_df.iloc[:, score_idx], errors='coerce'
                )
            
            if features.get('has_msms_matched', False) and 'MS/MS matched' in col_indices:
                msms_idx = col_indices['MS/MS matched']
                st.session_state.msdial_msms_matched = data_df.iloc[:, msms_idx].apply(
                    lambda x: str(x).upper() == 'TRUE'
                )
            
            # Sample intensity columns - use positional indexing to avoid duplicate name issues
            column_mapping = {}
            for i, (col_name, col_idx) in enumerate(zip(sample_cols_to_use, sample_indices_to_use), 1):
                new_col_name = f'intensity[s{i}]'
                standardized_data[new_col_name] = pd.to_numeric(
                    data_df.iloc[:, col_idx], errors='coerce'
                ).fillna(0)
                column_mapping[col_name] = new_col_name
            
            # Create the standardized dataframe
            result_df = pd.DataFrame(standardized_data)
            
            # Build complete column mapping including metadata columns (like generic format)
            complete_mapping = []
            
            # Add metadata column mappings
            complete_mapping.append({
                'original_name': lipid_col,
                'standardized_name': 'LipidMolec'
            })
            
            if features.get('has_ontology', False) and 'Ontology' in col_indices:
                complete_mapping.append({
                    'original_name': 'Ontology',
                    'standardized_name': 'ClassKey'
                })
            else:
                complete_mapping.append({
                    'original_name': f'{lipid_col} (inferred)',
                    'standardized_name': 'ClassKey'
                })
            
            if features.get('has_rt', False) and 'Average Rt(min)' in col_indices:
                complete_mapping.append({
                    'original_name': 'Average Rt(min)',
                    'standardized_name': 'BaseRt'
                })
            
            if features.get('has_mz', False) and 'Average Mz' in col_indices:
                complete_mapping.append({
                    'original_name': 'Average Mz',
                    'standardized_name': 'CalcMass'
                })
            
            # Add intensity column mappings
            for orig_name, std_name in column_mapping.items():
                complete_mapping.append({
                    'original_name': orig_name,
                    'standardized_name': std_name
                })
            
            # Store complete mapping for display
            # Put standardized_name first for better visibility in sidebar
            mapping_df = pd.DataFrame(complete_mapping)
            mapping_df = mapping_df[['standardized_name', 'original_name']]  # Swap column order
            
            # Validate that mapping_df has both required columns
            if 'original_name' not in mapping_df.columns or 'standardized_name' not in mapping_df.columns:
                st.error(f"Error: Column mapping DataFrame is malformed. Columns: {mapping_df.columns.tolist()}")
                st.error(f"Complete mapping list: {complete_mapping}")
            else:
                st.session_state.column_mapping = mapping_df
                
            st.session_state.n_intensity_cols = len(sample_cols_to_use)
            
            # Store original sample names for grouping UI
            st.session_state.msdial_sample_names = {
                f's{i}': col for i, col in enumerate(sample_cols_to_use, 1)
            }
            
            return result_df
            
        except Exception as e:
            st.error(f"Error standardizing MS-DIAL data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_msdial_quality_filter_options():
        """
        Returns available quality filtering options for MS-DIAL data.
        Call this after validation to determine what UI options to show.
        
        Returns:
            dict: Available filtering options and their status
        """
        features = st.session_state.get('msdial_features', {})
        
        return {
            'quality_filtering_available': features.get('has_quality_score', False),
            'msms_filtering_available': features.get('has_msms_matched', False),
            'normalized_data_available': features.get('has_normalized_data', False),
            'raw_sample_count': len(features.get('raw_sample_columns', [])),
            'normalized_sample_count': len(features.get('normalized_sample_columns', []))
        }
    
    @staticmethod
    def apply_msdial_quality_filter(df, min_score=None, require_msms=False):
        """
        Apply quality filtering to MS-DIAL data.
        
        Args:
            df: Standardized dataframe
            min_score: Minimum Total score threshold (0-100)
            require_msms: If True, only keep rows where MS/MS matched is TRUE
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        mask = pd.Series([True] * len(df))
        
        # Apply score filter
        if min_score is not None:
            quality_scores = st.session_state.get('msdial_quality_scores')
            if quality_scores is not None and len(quality_scores) == len(df):
                mask = mask & (quality_scores >= min_score)
        
        # Apply MS/MS filter
        if require_msms:
            msms_matched = st.session_state.get('msdial_msms_matched')
            if msms_matched is not None and len(msms_matched) == len(df):
                mask = mask & msms_matched
        
        filtered_df = df[mask].reset_index(drop=True)
        
        # Also filter the quality columns in session state
        if 'msdial_quality_scores' in st.session_state:
            st.session_state.msdial_quality_scores = st.session_state.msdial_quality_scores[mask].reset_index(drop=True)
        if 'msdial_msms_matched' in st.session_state:
            st.session_state.msdial_msms_matched = st.session_state.msdial_msms_matched[mask].reset_index(drop=True)
        
        return filtered_df

    # =========================================================================
    # Existing Format Methods (LipidSearch, Generic, Metabolomics Workbench)
    # =========================================================================
    
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
        # Put standardized_name first for better visibility in sidebar
        mapping_df = pd.DataFrame({
            'standardized_name': standardized_cols,
            'original_name': original_cols
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
            
            # Clean up condition names - remove "Condition:" prefix if present
            cleaned_conditions = []
            for condition in conditions:
                condition = condition.strip()
                if condition.startswith('Condition:'):
                    condition = condition.replace('Condition:', '').strip()
                cleaned_conditions.append(condition)
            
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
            # Use cleaned conditions without the "Condition:" prefix
            st.session_state.workbench_conditions = {
                f's{i+1}': condition 
                for i, condition in enumerate(cleaned_conditions)
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
        Handles conditions from Metabolomics Workbench format.
        Each condition string is treated as a complete condition, without parsing.
        Examples:
            "WT" -> "WT"
            "Diet:Normal | BileAcid:water" -> "Diet:Normal | BileAcid:water"
        """
        unique_conditions = set()
        for condition in conditions:
            if isinstance(condition, str):
                unique_conditions.add(condition.strip())
        return list(unique_conditions)