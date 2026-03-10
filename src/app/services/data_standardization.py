"""
Data standardization service for lipid data.

Validates and standardizes data from all supported formats into a common schema:
- LipidMolec column (standardized lipid names)
- ClassKey column (inferred from lipid names)
- intensity[s1], intensity[s2], ... columns

Pure logic — no Streamlit dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from app.services.format_detection import DataFormat, FormatDetectionService


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class StandardizationResult:
    """Result of validating and standardizing a dataset.

    Attributes:
        success: Whether standardization succeeded.
        message: Status or error message.
        standardized_df: The standardized DataFrame (None on failure).
        column_mapping: DataFrame with 'standardized_name' and 'original_name' columns.
        n_intensity_cols: Number of intensity columns in the standardized data.
        msdial_features: MS-DIAL–specific feature dict (None for other formats).
        msdial_sample_names: Map of s1->original_name for MS-DIAL (None for others).
        workbench_conditions: Map of s1->condition for Metabolomics Workbench.
        workbench_samples: Map of s1->sample_name for Metabolomics Workbench.
    """
    success: bool
    message: str
    standardized_df: Optional[pd.DataFrame] = None
    column_mapping: Optional[pd.DataFrame] = None
    n_intensity_cols: int = 0
    msdial_features: Optional[Dict] = None
    msdial_sample_names: Optional[Dict[str, str]] = None
    workbench_conditions: Optional[Dict[str, str]] = None
    workbench_samples: Optional[Dict[str, str]] = None


@dataclass
class MSDIALOverrideResult:
    """Result of applying MS-DIAL sample column override.

    Attributes:
        standardized_df: Rebuilt DataFrame with only selected intensity columns.
        column_mapping: Updated column mapping DataFrame.
        n_intensity_cols: Number of intensity columns after override.
        sample_names: Updated s1->original_name mapping.
        raw_sample_columns: Updated raw sample column list.
        normalized_sample_columns: Updated normalized sample column list.
    """
    standardized_df: pd.DataFrame
    column_mapping: pd.DataFrame
    n_intensity_cols: int
    sample_names: Dict[str, str]
    raw_sample_columns: List[str]
    normalized_sample_columns: List[str]


# =============================================================================
# Service
# =============================================================================


class DataStandardizationService:
    """
    Validates and standardizes lipid data from all supported formats.

    All methods are static and pure — no Streamlit session state access.
    Returns StandardizationResult with all data the caller needs.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def validate_and_standardize(
        data: Union[pd.DataFrame, str],
        data_format: DataFormat,
        msdial_use_normalized: bool = False,
    ) -> StandardizationResult:
        """Validate and standardize data for the given format.

        Args:
            data: Raw DataFrame, or raw text string for Metabolomics Workbench.
            data_format: The target format enum.
            msdial_use_normalized: For MS-DIAL, use pre-normalized columns
                instead of raw. Ignored for other formats.

        Returns:
            StandardizationResult with all outputs.
        """
        try:
            if data_format == DataFormat.LIPIDSEARCH:
                return DataStandardizationService._process_lipidsearch(data)
            elif data_format == DataFormat.MSDIAL:
                return DataStandardizationService._process_msdial(
                    data, msdial_use_normalized
                )
            elif data_format == DataFormat.METABOLOMICS_WORKBENCH:
                return DataStandardizationService._process_metabolomics_workbench(data)
            elif data_format == DataFormat.GENERIC:
                return DataStandardizationService._process_generic(data)
            else:
                return StandardizationResult(
                    success=False,
                    message=f"Unsupported format: {data_format}",
                )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            return StandardizationResult(
                success=False,
                message=f"Error during preprocessing: {e}",
            )

    # ------------------------------------------------------------------
    # MS-DIAL sample override
    # ------------------------------------------------------------------

    @staticmethod
    def apply_msdial_sample_override(
        df: pd.DataFrame,
        column_mapping: pd.DataFrame,
        manual_samples: List[str],
        features: Dict,
    ) -> MSDIALOverrideResult:
        """Rebuild DataFrame and column mapping after manual MS-DIAL sample override.

        Pure logic — no session state access.

        Args:
            df: Current standardized DataFrame with intensity[sN] columns.
            column_mapping: Current column mapping DataFrame with
                'standardized_name' and 'original_name' columns.
            manual_samples: User-selected original sample column names.
            features: MS-DIAL features dict (needs 'normalized_sample_columns').

        Returns:
            MSDIALOverrideResult with rebuilt DataFrame, mapping, and updated lists.
        """
        # Update feature lists
        raw_sample_columns = list(manual_samples)
        current_norm_samples = features.get('normalized_sample_columns', [])
        normalized_sample_columns = [s for s in current_norm_samples if s in manual_samples]

        # Build reverse lookup: standardized_name -> original_name
        std_to_orig = dict(zip(
            column_mapping['standardized_name'],
            column_mapping['original_name'],
        ))

        # Identify which intensity columns to keep (by original name)
        intensity_cols_to_keep = [
            col for col in df.columns
            if col.startswith('intensity[') and std_to_orig.get(col) in manual_samples
        ]

        # Build new DataFrame with metadata + selected intensity columns
        non_intensity_cols = [col for col in df.columns if not col.startswith('intensity[')]
        new_df = df[non_intensity_cols + intensity_cols_to_keep].copy()

        # Build new column mapping (metadata rows + sequential intensity rows)
        new_mapping_rows = [
            {'standardized_name': row['standardized_name'], 'original_name': row['original_name']}
            for _, row in column_mapping.iterrows()
            if not row['standardized_name'].startswith('intensity[')
        ]

        rename_map = {}
        for i, old_col in enumerate(intensity_cols_to_keep, 1):
            new_col = f'intensity[s{i}]'
            rename_map[old_col] = new_col
            new_mapping_rows.append({
                'standardized_name': new_col,
                'original_name': std_to_orig.get(old_col, old_col),
            })

        new_df = new_df.rename(columns=rename_map)

        sample_names = {
            f's{i}': orig for i, orig in enumerate(manual_samples, 1)
        }

        return MSDIALOverrideResult(
            standardized_df=new_df,
            column_mapping=pd.DataFrame(new_mapping_rows),
            n_intensity_cols=len(intensity_cols_to_keep),
            sample_names=sample_names,
            raw_sample_columns=raw_sample_columns,
            normalized_sample_columns=normalized_sample_columns,
        )

    # ------------------------------------------------------------------
    # Lipid name helpers (ported from DataFormatHandler)
    # ------------------------------------------------------------------

    @staticmethod
    def standardize_lipid_name(lipid_name) -> str:
        """Standardize a lipid name to Class(chain_details)(modifications).

        Examples:
            LPC O-17:4 -> LPC(O-17:4)
            Cer d18:0/C24:0 -> Cer(d18:0_C24:0)
            LPC 18:1(d7) -> LPC(18:1)(d7)
            Cer 18:1;2O/24:0 -> Cer(18:1;2O_24:0)
        """
        try:
            if not lipid_name or pd.isna(lipid_name):
                return "Unknown"

            lipid_name = str(lipid_name).strip()

            # Extract deuteration modifications like (d7), (d9) at the END
            modification = ""
            mod_match = re.search(r'\(d\d+\)$', lipid_name)
            if mod_match:
                modification = mod_match.group()
                lipid_name = lipid_name[:mod_match.start()].strip()

            first_paren = lipid_name.find('(')
            first_colon = lipid_name.find(':')

            if first_paren > 0 and (first_colon < 0 or first_paren < first_colon):
                # Already has parentheses around chain info
                class_name = lipid_name[:first_paren]
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
                chain_info = lipid_name[first_paren + 1:chain_end]
                trailing = lipid_name[chain_end + 1:].strip()
                chain_info = chain_info.replace('/', '_')
                result = f"{class_name}({chain_info})"
                if trailing:
                    result = f"{result}{trailing}"
            else:
                first_space = lipid_name.find(' ')
                if first_space > 0:
                    class_name = lipid_name[:first_space]
                    chain_info = lipid_name[first_space + 1:].replace('/', '_')
                    result = f"{class_name}({chain_info})"
                else:
                    match = re.match(r'^([A-Za-z][A-Za-z0-9]*)[- ]?(.*)', lipid_name)
                    if match:
                        class_name, chain_info = match.groups()
                        chain_info = chain_info.replace('/', '_')
                        result = f"{class_name}({chain_info})" if chain_info else class_name
                    else:
                        result = lipid_name

            if modification:
                result = f"{result}{modification}"

            return result

        except (TypeError, AttributeError, ValueError, IndexError):
            return str(lipid_name) if lipid_name else "Unknown"

    @staticmethod
    def infer_class_key(lipid_molec) -> str:
        """Infer ClassKey from a standardized LipidMolec name.

        Examples:
            CL(18:2/16:0/18:1/18:2) -> CL
            LPC(18:1)(d7) -> LPC
            SPLASH(LPC 18:1)(d7) -> LPC
        """
        try:
            splash_match = re.match(
                r'^SPLASH\s*\(\s*([A-Za-z]+)', str(lipid_molec), re.IGNORECASE
            )
            if splash_match:
                return splash_match.group(1).strip()

            match = re.match(r'^([A-Za-z][A-Za-z0-9]*)', str(lipid_molec))
            if match:
                return match.group(1).strip()
            return "Unknown"
        except (TypeError, AttributeError):
            return "Unknown"

    # ------------------------------------------------------------------
    # LipidSearch 5.0
    # ------------------------------------------------------------------

    @staticmethod
    def _process_lipidsearch(df: pd.DataFrame) -> StandardizationResult:
        """Validate and standardize LipidSearch 5.0 data."""
        if not isinstance(df, pd.DataFrame):
            return StandardizationResult(False, "Expected a DataFrame for LipidSearch format")

        # Validate
        required_cols = [
            'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
            'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey',
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return StandardizationResult(
                False, f"Missing required columns: {', '.join(missing)}"
            )

        meanarea_cols = [c for c in df.columns if c.startswith('MeanArea[')]
        if not meanarea_cols:
            return StandardizationResult(False, "No MeanArea columns found")

        # Standardize: rename MeanArea[x] -> intensity[x]
        result_df = df.copy()
        rename_dict = {}
        for col in meanarea_cols:
            sample = col.split('[', 1)[1].rstrip(']')
            rename_dict[col] = f'intensity[{sample}]'
        result_df = result_df.rename(columns=rename_dict)

        # Build column mapping (only renamed columns)
        mapping_rows = []
        for orig, std in rename_dict.items():
            mapping_rows.append({'standardized_name': std, 'original_name': orig})
        mapping_df = pd.DataFrame(mapping_rows) if mapping_rows else None

        return StandardizationResult(
            success=True,
            message="Data successfully standardized",
            standardized_df=result_df,
            column_mapping=mapping_df,
            n_intensity_cols=len(meanarea_cols),
        )

    # ------------------------------------------------------------------
    # Generic
    # ------------------------------------------------------------------

    @staticmethod
    def _is_classkey_column(df: pd.DataFrame, col_idx: int) -> bool:
        """Check if a column looks like a ClassKey column."""
        if col_idx >= len(df.columns):
            return False

        col_name = df.columns[col_idx]
        if str(col_name).lower().strip() == 'classkey':
            return True

        col_values = df.iloc[:, col_idx]
        try:
            numeric_count = pd.to_numeric(col_values, errors='coerce').notna().sum()
            if numeric_count / len(col_values) > 0.5:
                return False

            str_values = col_values.astype(str)
            avg_len = str_values.str.len().mean()
            total_len = str_values.str.len().sum()
            if total_len == 0:
                return False
            alpha_ratio = (
                str_values.str.replace(r'[^A-Za-z]', '', regex=True).str.len().sum()
                / total_len
            )
            return avg_len < 15 and alpha_ratio > 0.7
        except (ValueError, TypeError, ZeroDivisionError):
            return False

    @staticmethod
    def _process_generic(df: pd.DataFrame) -> StandardizationResult:
        """Validate and standardize Generic format data."""
        if not isinstance(df, pd.DataFrame):
            return StandardizationResult(False, "Expected a DataFrame for Generic format")

        if len(df.columns) < 2:
            return StandardizationResult(
                False, "Dataset must have at least 2 columns: lipid names and one intensity column"
            )

        # First column should contain letters (lipid names)
        if not df.iloc[:, 0].astype(str).str.contains('[A-Za-z]').any():
            return StandardizationResult(
                False, "First column doesn't appear to contain lipid names (no letters found)."
            )

        result_df = df.copy()
        original_cols = df.columns.tolist()
        standardized_cols = ['LipidMolec']

        has_classkey = (
            len(original_cols) > 2
            and DataStandardizationService._is_classkey_column(df, 1)
        )

        if has_classkey:
            standardized_cols.append('ClassKey')
            intensity_start = 2
        else:
            intensity_start = 1

        for i in range(intensity_start, len(original_cols)):
            standardized_cols.append(f'intensity[s{i - intensity_start + 1}]')

        n_intensity = len(original_cols) - intensity_start

        # Column mapping
        mapping_df = pd.DataFrame({
            'standardized_name': standardized_cols,
            'original_name': original_cols,
        })

        # Rename columns
        result_df.columns = standardized_cols

        # Standardize lipid names
        result_df['LipidMolec'] = result_df['LipidMolec'].apply(
            DataStandardizationService.standardize_lipid_name
        )

        # Infer ClassKey if not present
        if not has_classkey:
            result_df['ClassKey'] = result_df['LipidMolec'].apply(
                DataStandardizationService.infer_class_key
            )

        return StandardizationResult(
            success=True,
            message="Data successfully standardized",
            standardized_df=result_df,
            column_mapping=mapping_df,
            n_intensity_cols=n_intensity,
        )

    # ------------------------------------------------------------------
    # MS-DIAL
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_msdial_sample_columns(
        df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Optional[int], List[int], List[int]]:
        """Detect sample intensity columns in MS-DIAL data.

        Returns:
            (raw_sample_cols, normalized_sample_cols, lipid_is_idx,
             raw_sample_indices, normalized_sample_indices)
        """
        if columns is None:
            columns = df.columns.tolist()

        metadata_cols = FormatDetectionService.MSDIAL_METADATA_COLUMNS

        raw_cols: List[str] = []
        norm_cols: List[str] = []
        raw_indices: List[int] = []
        norm_indices: List[int] = []
        lipid_is_idx = None

        for idx, col in enumerate(columns):
            if col == 'Lipid IS':
                lipid_is_idx = idx
                break

        for idx, col in enumerate(columns):
            if col in metadata_cols or col == 'Lipid IS':
                continue
            if pd.isna(col) or (isinstance(col, str) and not col.strip()):
                continue

            try:
                col_data = df.iloc[:, idx]
                numeric_values = pd.to_numeric(col_data, errors='coerce')
                numeric_count = numeric_values.notna().sum()

                if numeric_count > len(col_data) * 0.5:
                    if lipid_is_idx is not None:
                        if idx < lipid_is_idx:
                            raw_cols.append(col)
                            raw_indices.append(idx)
                        elif idx > lipid_is_idx:
                            norm_cols.append(col)
                            norm_indices.append(idx)
                    else:
                        raw_cols.append(col)
                        raw_indices.append(idx)
            except (ValueError, TypeError, IndexError):
                continue

        return raw_cols, norm_cols, lipid_is_idx, raw_indices, norm_indices

    @staticmethod
    def _detect_msdial_structure(
        df: pd.DataFrame,
    ) -> Tuple[int, List[str], str, int, Dict[str, int], Dict]:
        """Detect MS-DIAL file structure: header row, columns, and features.

        Returns:
            (header_row_idx, actual_columns, lipid_col, lipid_col_idx,
             col_indices, features)

        Raises:
            ValueError: If no lipid column or sample columns are found.
        """
        header_row_idx = FormatDetectionService._detect_msdial_header_row(df)

        if header_row_idx >= 0:
            actual_columns = df.iloc[header_row_idx].tolist()
        else:
            actual_columns = df.columns.tolist()

        # Find lipid name column
        lipid_col = None
        lipid_col_idx = None
        for col_name in ['Metabolite name', 'Metabolite', 'Name', 'Lipid', 'LipidMolec']:
            if col_name in actual_columns:
                lipid_col = col_name
                lipid_col_idx = actual_columns.index(col_name)
                break

        if lipid_col is None:
            raise ValueError(
                "No lipid name column found. Expected 'Metabolite name' column."
            )

        # Create temp df with correct headers for sample detection
        if header_row_idx >= 0:
            temp_df = df.iloc[header_row_idx + 1:].copy()
            temp_df.columns = actual_columns
        else:
            temp_df = df

        raw_samples, norm_samples, lipid_is_idx, raw_indices, norm_indices = (
            DataStandardizationService._detect_msdial_sample_columns(
                temp_df, actual_columns
            )
        )

        if len(raw_samples) + len(norm_samples) < 1:
            raise ValueError(
                "No sample intensity columns detected. "
                "Check that your export includes sample data."
            )

        # Build column index mapping (first occurrence only)
        col_indices: Dict[str, int] = {}
        for idx, col in enumerate(actual_columns):
            if col not in col_indices:
                col_indices[col] = idx

        features = {
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
            'column_indices': col_indices,
        }

        return header_row_idx, actual_columns, lipid_col, lipid_col_idx, col_indices, features

    @staticmethod
    def _standardize_msdial_columns(
        df: pd.DataFrame,
        header_row_idx: int,
        lipid_col_idx: int,
        col_indices: Dict[str, int],
        features: Dict,
        sample_cols: List[str],
        sample_indices: List[int],
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Standardize MS-DIAL columns into the common schema.

        Returns:
            (result_df, column_mapping_dict) where column_mapping_dict maps
            original sample column names to intensity[sN] names.
        """
        if header_row_idx >= 0:
            data_df = df.iloc[header_row_idx + 1:].copy()
        else:
            data_df = df.copy()
        data_df = data_df.reset_index(drop=True)

        standardized_data: Dict[str, pd.Series] = {}

        # LipidMolec + ClassKey
        standardized_data['LipidMolec'] = data_df.iloc[:, lipid_col_idx].apply(
            DataStandardizationService.standardize_lipid_name
        )
        standardized_data['ClassKey'] = standardized_data['LipidMolec'].apply(
            DataStandardizationService.infer_class_key
        )

        # Optional metadata columns
        if features['has_rt'] and 'Average Rt(min)' in col_indices:
            standardized_data['BaseRt'] = pd.to_numeric(
                data_df.iloc[:, col_indices['Average Rt(min)']], errors='coerce'
            )
        if features['has_mz'] and 'Average Mz' in col_indices:
            standardized_data['CalcMass'] = pd.to_numeric(
                data_df.iloc[:, col_indices['Average Mz']], errors='coerce'
            )

        # Quality columns (preserved for filtering)
        if features['has_quality_score'] and 'Total score' in col_indices:
            standardized_data['Total score'] = pd.to_numeric(
                data_df.iloc[:, col_indices['Total score']], errors='coerce'
            )
        if features['has_msms_matched'] and 'MS/MS matched' in col_indices:
            standardized_data['MS/MS matched'] = data_df.iloc[
                :, col_indices['MS/MS matched']
            ]

        # Sample intensity columns
        column_mapping_dict: Dict[str, str] = {}
        for i, (col_name, col_idx) in enumerate(
            zip(sample_cols, sample_indices), 1
        ):
            new_col = f'intensity[s{i}]'
            standardized_data[new_col] = pd.to_numeric(
                data_df.iloc[:, col_idx], errors='coerce'
            ).fillna(0)
            column_mapping_dict[col_name] = new_col

        return pd.DataFrame(standardized_data), column_mapping_dict

    @staticmethod
    def _build_msdial_mapping(
        lipid_col: str,
        col_indices: Dict[str, int],
        features: Dict,
        column_mapping_dict: Dict[str, str],
    ) -> pd.DataFrame:
        """Build the column mapping DataFrame for MS-DIAL."""
        mapping_rows = [
            {'original_name': lipid_col, 'standardized_name': 'LipidMolec'}
        ]
        if features['has_rt'] and 'Average Rt(min)' in col_indices:
            mapping_rows.append({
                'original_name': 'Average Rt(min)',
                'standardized_name': 'BaseRt',
            })
        if features['has_mz'] and 'Average Mz' in col_indices:
            mapping_rows.append({
                'original_name': 'Average Mz',
                'standardized_name': 'CalcMass',
            })
        for orig, std in column_mapping_dict.items():
            mapping_rows.append({
                'original_name': orig,
                'standardized_name': std,
            })
        return pd.DataFrame(mapping_rows)[['standardized_name', 'original_name']]

    @staticmethod
    def _process_msdial(
        df: pd.DataFrame, use_normalized: bool = False
    ) -> StandardizationResult:
        """Validate and standardize MS-DIAL data."""
        if not isinstance(df, pd.DataFrame):
            return StandardizationResult(False, "Expected a DataFrame for MS-DIAL format")

        try:
            header_row_idx, actual_columns, lipid_col, lipid_col_idx, col_indices, features = (
                DataStandardizationService._detect_msdial_structure(df)
            )

            # Choose raw vs normalized sample columns
            norm_samples = features['normalized_sample_columns']
            if use_normalized and len(norm_samples) > 0:
                sample_cols = norm_samples
                sample_indices = features['normalized_sample_indices']
            else:
                sample_cols = features['raw_sample_columns']
                sample_indices = features['raw_sample_indices']

            result_df, column_mapping_dict = (
                DataStandardizationService._standardize_msdial_columns(
                    df, header_row_idx, lipid_col_idx,
                    col_indices, features, sample_cols, sample_indices,
                )
            )

            mapping_df = DataStandardizationService._build_msdial_mapping(
                lipid_col, col_indices, features, column_mapping_dict,
            )

            sample_names = {
                f's{i}': col for i, col in enumerate(sample_cols, 1)
            }

            return StandardizationResult(
                success=True,
                message="Data successfully standardized",
                standardized_df=result_df,
                column_mapping=mapping_df,
                n_intensity_cols=len(sample_cols),
                msdial_features=features,
                msdial_sample_names=sample_names,
            )

        except ValueError as e:
            return StandardizationResult(False, str(e))
        except (KeyError, IndexError, TypeError) as e:
            return StandardizationResult(
                False, f"MS-DIAL standardization error: {e}"
            )

    # ------------------------------------------------------------------
    # Metabolomics Workbench
    # ------------------------------------------------------------------

    @staticmethod
    def _process_metabolomics_workbench(
        text_data: Union[str, pd.DataFrame],
    ) -> StandardizationResult:
        """Validate and standardize Metabolomics Workbench text data."""
        if not isinstance(text_data, str):
            return StandardizationResult(
                False, "Invalid input type for Metabolomics Workbench format"
            )

        # Validate markers
        if 'MS_METABOLITE_DATA_START' not in text_data or 'MS_METABOLITE_DATA_END' not in text_data:
            return StandardizationResult(False, "Missing required data section markers")

        start_idx = text_data.find('MS_METABOLITE_DATA_START')
        end_idx = text_data.find('MS_METABOLITE_DATA_END')
        if start_idx >= end_idx:
            return StandardizationResult(False, "Invalid data section markers")

        try:
            data_section = text_data[start_idx:end_idx].strip().split('\n')
            data_section = [line.strip() for line in data_section if line.strip()]
            data_section = [
                line for line in data_section if 'MS_METABOLITE_DATA_START' not in line
            ]

            if len(data_section) < 3:
                return StandardizationResult(False, "Insufficient data rows")

            samples_line = data_section[0].split(',')
            factors_line = data_section[1].split(',')

            sample_names = samples_line[1:]
            conditions = factors_line[1:]

            cleaned_conditions = []
            for cond in conditions:
                cond = cond.strip()
                if cond.startswith('Condition:'):
                    cond = cond.replace('Condition:', '').strip()
                cleaned_conditions.append(cond)

            data_rows = []
            for line in data_section[2:]:
                if line.strip():
                    values = line.split(',')
                    if len(values) == len(sample_names) + 1:
                        data_rows.append(values)

            if not data_rows:
                return StandardizationResult(False, "No valid data rows found")

            columns = ['LipidMolec'] + [
                f'intensity[s{i + 1}]' for i in range(len(sample_names))
            ]
            df = pd.DataFrame(data_rows, columns=columns)

            # Standardize lipid names and infer ClassKey
            df['LipidMolec'] = df['LipidMolec'].apply(
                DataStandardizationService.standardize_lipid_name
            )
            df['ClassKey'] = df['LipidMolec'].apply(
                DataStandardizationService.infer_class_key
            )

            # Convert intensity columns to numeric
            intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
            for col in intensity_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df[intensity_cols] = df[intensity_cols].fillna(0)

            wb_conditions = {
                f's{i + 1}': cond for i, cond in enumerate(cleaned_conditions)
            }
            wb_samples = {
                f's{i + 1}': name.strip() for i, name in enumerate(sample_names)
            }

            return StandardizationResult(
                success=True,
                message="Data successfully standardized",
                standardized_df=df,
                n_intensity_cols=len(sample_names),
                workbench_conditions=wb_conditions,
                workbench_samples=wb_samples,
            )

        except (ValueError, KeyError, IndexError, TypeError) as e:
            return StandardizationResult(
                False, f"Error standardizing Metabolomics Workbench data: {e}"
            )
