"""
Format detection service for lipid data.
Automatically identifies the format of uploaded data based on column signatures.
Pure logic - no Streamlit dependencies.
"""
from enum import Enum
from typing import Union, List, Set
import pandas as pd


class DataFormat(str, Enum):
    """Supported data formats."""
    LIPIDSEARCH = "LipidSearch 5.0"
    MSDIAL = "MS-DIAL"
    METABOLOMICS_WORKBENCH = "Metabolomics Workbench"
    GENERIC = "Generic Format"
    UNKNOWN = "Unknown"


class FormatDetectionService:
    """
    Service for automatically detecting the format of lipidomics data.

    Detection is based on column signatures and data patterns unique to each format.
    All methods are stateless static methods for easy testing.
    """

    # LipidSearch 5.0 required columns
    LIPIDSEARCH_REQUIRED_COLUMNS: Set[str] = {
        'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
        'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey'
    }

    # MS-DIAL known metadata columns (used for sample column detection)
    MSDIAL_METADATA_COLUMNS: Set[str] = {
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

    # MS-DIAL signature columns (presence of these suggests MS-DIAL format)
    MSDIAL_SIGNATURE_COLUMNS: Set[str] = {
        'Alignment ID', 'Metabolite name', 'Average Rt(min)', 'Average Mz',
        'Adduct type', 'Total score', 'MS/MS matched', 'Ontology'
    }

    # Metabolomics Workbench markers
    MW_START_MARKER: str = 'MS_METABOLITE_DATA_START'
    MW_END_MARKER: str = 'MS_METABOLITE_DATA_END'

    @staticmethod
    def detect_format(data: Union[pd.DataFrame, str]) -> DataFormat:
        """
        Detect the format of the input data.

        Args:
            data: Either a DataFrame (for most formats) or a string
                  (for Metabolomics Workbench raw text)

        Returns:
            DataFormat enum indicating the detected format
        """
        # Check for Metabolomics Workbench (text format)
        if isinstance(data, str):
            if FormatDetectionService._is_metabolomics_workbench(data):
                return DataFormat.METABOLOMICS_WORKBENCH
            return DataFormat.UNKNOWN

        # Must be a DataFrame for other formats
        if not isinstance(data, pd.DataFrame):
            return DataFormat.UNKNOWN

        if data.empty:
            return DataFormat.UNKNOWN

        # Check formats in order of specificity (most specific first)
        if FormatDetectionService._is_lipidsearch(data):
            return DataFormat.LIPIDSEARCH

        if FormatDetectionService._is_msdial(data):
            return DataFormat.MSDIAL

        if FormatDetectionService._is_generic(data):
            return DataFormat.GENERIC

        return DataFormat.UNKNOWN

    @staticmethod
    def _is_metabolomics_workbench(text_data: str) -> bool:
        """
        Check if text data is in Metabolomics Workbench format.

        Metabolomics Workbench format is characterized by:
        - MS_METABOLITE_DATA_START marker
        - MS_METABOLITE_DATA_END marker
        - Data section between markers with samples row and factors row
        """
        if not text_data:
            return False

        has_start = FormatDetectionService.MW_START_MARKER in text_data
        has_end = FormatDetectionService.MW_END_MARKER in text_data

        if not (has_start and has_end):
            return False

        # Verify markers are in correct order
        start_idx = text_data.find(FormatDetectionService.MW_START_MARKER)
        end_idx = text_data.find(FormatDetectionService.MW_END_MARKER)

        return start_idx < end_idx

    @staticmethod
    def _is_lipidsearch(df: pd.DataFrame) -> bool:
        """
        Check if DataFrame is in LipidSearch 5.0 format.

        LipidSearch format requires:
        - All required columns present
        - At least one MeanArea[*] column for intensity values
        """
        columns = set(df.columns)

        # Check for required columns
        if not FormatDetectionService.LIPIDSEARCH_REQUIRED_COLUMNS.issubset(columns):
            return False

        # Check for MeanArea intensity columns
        has_meanarea = any(col.startswith('MeanArea[') for col in df.columns)

        return has_meanarea

    @staticmethod
    def _is_msdial(df: pd.DataFrame) -> bool:
        """
        Check if DataFrame is in MS-DIAL format.

        MS-DIAL format detection:
        - Has 'Metabolite name' column (can be in data rows if metadata headers present)
        - May have 'Alignment ID' as first column
        - Contains characteristic MS-DIAL metadata columns
        - Sample intensity columns are at the end (not in metadata set)
        """
        columns = set(df.columns)

        # Direct check: 'Metabolite name' in columns
        if 'Metabolite name' in columns:
            # Verify it has at least some MS-DIAL signature columns
            signature_match = columns & FormatDetectionService.MSDIAL_SIGNATURE_COLUMNS
            if len(signature_match) >= 2:  # At least 2 signature columns
                return True

        # Check for 'Alignment ID' as first column (common in MS-DIAL)
        if df.columns[0] == 'Alignment ID':
            return True

        # Check for metadata header rows (MS-DIAL exports often have these)
        # Look for 'Alignment ID' or 'Metabolite name' in the data
        header_row_idx = FormatDetectionService._detect_msdial_header_row(df)
        if header_row_idx >= 0:
            # Found MS-DIAL style header row in data
            return True

        return False

    @staticmethod
    def _detect_msdial_header_row(df: pd.DataFrame) -> int:
        """
        Detect if there are metadata header rows in MS-DIAL data.

        MS-DIAL exports can have metadata rows at the top before actual column headers.

        Returns:
            Index of the header row (0-based), or -1 if no metadata rows detected
        """
        # Check first 20 rows for header markers
        max_rows = min(20, len(df))

        for idx in range(max_rows):
            row = df.iloc[idx]
            first_val = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''

            # The header row will have "Alignment ID" as text in first column
            if first_val == 'Alignment ID':
                return idx

            # Alternative: check for 'Metabolite name' which is more unique
            row_values = [str(v).strip() for v in row.values if pd.notna(v)]
            if 'Metabolite name' in row_values:
                return idx

        return -1

    @staticmethod
    def _is_generic(df: pd.DataFrame) -> bool:
        """
        Check if DataFrame is in generic format.

        Generic format:
        - First column contains lipid names (has letters)
        - At least 2 columns total (lipid names + at least one intensity column)
        - Other columns are numeric (intensity values)
        """
        if len(df.columns) < 2:
            return False

        # First column should contain letters (lipid names)
        first_col = df.iloc[:, 0]
        has_letters = first_col.astype(str).str.contains('[A-Za-z]').any()

        if not has_letters:
            return False

        # Check if at least one other column appears to be numeric
        for col_idx in range(1, min(len(df.columns), 5)):  # Check first few columns
            col = df.iloc[:, col_idx]
            numeric_count = pd.to_numeric(col, errors='coerce').notna().sum()
            if numeric_count > len(col) * 0.5:  # >50% numeric values
                return True

        return False

    @staticmethod
    def get_format_display_name(format_type: DataFormat) -> str:
        """
        Get the user-friendly display name for a format.

        Args:
            format_type: DataFormat enum value

        Returns:
            User-friendly string name
        """
        return format_type.value

    @staticmethod
    def get_all_formats() -> List[DataFormat]:
        """
        Get list of all supported formats (excluding UNKNOWN).

        Returns:
            List of DataFormat values
        """
        return [
            DataFormat.GENERIC,
            DataFormat.METABOLOMICS_WORKBENCH,
            DataFormat.LIPIDSEARCH,
            DataFormat.MSDIAL
        ]

    @staticmethod
    def get_format_from_string(format_string: str) -> DataFormat:
        """
        Convert a format string to DataFormat enum.

        Args:
            format_string: Format name as string

        Returns:
            Corresponding DataFormat enum value, or UNKNOWN if not recognized
        """
        format_mapping = {
            'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
            'lipidsearch': DataFormat.LIPIDSEARCH,
            'MS-DIAL': DataFormat.MSDIAL,
            'msdial': DataFormat.MSDIAL,
            'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
            'metabolomics_workbench': DataFormat.METABOLOMICS_WORKBENCH,
            'Generic Format': DataFormat.GENERIC,
            'generic': DataFormat.GENERIC,
        }
        return format_mapping.get(format_string, DataFormat.UNKNOWN)
