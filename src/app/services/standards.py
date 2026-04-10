"""
Internal standards management service for lipid data.

Pure business logic - no Streamlit dependencies.
Handles detection, extraction, validation, and processing of internal standards.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from app.constants import (
    INTERNAL_STANDARD_LIPID_PATTERNS,
    INTERNAL_STANDARD_CLASS_PATTERN,
)
from app.services.data_standardization import DataStandardizationService

# Canonical class inference function — used throughout this module
# instead of a local _infer_class, so all class inference is consistent.
_infer_class_key = DataStandardizationService.infer_class_key


@dataclass
class StandardsExtractionResult:
    """
    Result of extracting internal standards from a DataFrame.

    Attributes:
        data_df: DataFrame with standards removed
        standards_df: DataFrame containing only internal standards
        standards_count: Number of standards extracted
        detection_patterns_matched: Patterns that matched for detection
    """
    data_df: pd.DataFrame
    standards_df: pd.DataFrame
    standards_count: int
    detection_patterns_matched: List[str] = field(default_factory=list)


@dataclass
class StandardsValidationResult:
    """
    Result of validating a standards DataFrame.

    Attributes:
        is_valid: Whether the standards DataFrame is valid
        errors: List of validation errors
        warnings: List of validation warnings (non-blocking)
        valid_standards_count: Number of valid standards
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    valid_standards_count: int = 0


@dataclass
class StandardsProcessingResult:
    """
    Result of processing an uploaded standards file.

    Attributes:
        standards_df: Processed standards DataFrame with correct columns
        duplicates_removed: Number of duplicate standards removed
        standards_count: Number of standards after processing
        source_mode: 'extract' if extracted from main dataset, 'complete' if external
    """
    standards_df: pd.DataFrame
    duplicates_removed: int
    standards_count: int
    source_mode: str


class StandardsService:
    """
    Service for managing internal standards in lipidomic data.

    All methods are static - no instance state required.
    Handles detection, extraction, validation, and processing of internal standards.
    """

    # Internal standard patterns (imported from app.constants)
    INTERNAL_STANDARD_LIPID_PATTERNS: List[str] = INTERNAL_STANDARD_LIPID_PATTERNS
    INTERNAL_STANDARD_CLASS_PATTERN: str = INTERNAL_STANDARD_CLASS_PATTERN

    # ==================== Detection Methods ====================

    @staticmethod
    def detect_standards(df: pd.DataFrame) -> pd.Series:
        """
        Detect internal standards in a DataFrame based on name and class patterns.

        Args:
            df: DataFrame with LipidMolec column (and optionally ClassKey)

        Returns:
            Boolean Series where True indicates an internal standard

        Raises:
            ValueError: If DataFrame is empty or missing LipidMolec column
        """
        if df is None or df.empty:
            raise ValueError(
                "DataFrame is empty. Cannot detect internal standards."
            )

        if 'LipidMolec' not in df.columns:
            raise ValueError(
                "DataFrame must have 'LipidMolec' column to detect internal standards."
            )

        # Check lipid name patterns
        is_standard_lipid = StandardsService._check_lipid_patterns(df)

        # Check class patterns if ClassKey exists
        is_standard_class = StandardsService._check_class_patterns(df)

        return is_standard_lipid | is_standard_class

    @staticmethod
    def _check_lipid_patterns(df: pd.DataFrame) -> pd.Series:
        """
        Check LipidMolec column for internal standard patterns.

        Args:
            df: DataFrame with LipidMolec column

        Returns:
            Boolean Series indicating matches
        """
        combined_pattern = '|'.join(
            f'(?:{p})' for p in StandardsService.INTERNAL_STANDARD_LIPID_PATTERNS
        )

        return df['LipidMolec'].str.contains(
            combined_pattern,
            regex=True,
            case=False,
            na=False
        )

    @staticmethod
    def _check_class_patterns(df: pd.DataFrame) -> pd.Series:
        """
        Check ClassKey column for internal standard markers.

        Args:
            df: DataFrame (may or may not have ClassKey column)

        Returns:
            Boolean Series indicating matches (all False if no ClassKey column)
        """
        if 'ClassKey' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        return df['ClassKey'].str.contains(
            StandardsService.INTERNAL_STANDARD_CLASS_PATTERN,
            regex=True,
            case=False,
            na=False
        )

    @staticmethod
    def get_matched_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get which patterns matched for each detected standard.

        Args:
            df: DataFrame with LipidMolec column

        Returns:
            Dictionary mapping lipid names to list of matched patterns
        """
        if df is None or df.empty or 'LipidMolec' not in df.columns:
            return {}

        matches = {}

        for lipid in df['LipidMolec'].unique():
            lipid_matches = []

            # Check each lipid pattern
            for pattern in StandardsService.INTERNAL_STANDARD_LIPID_PATTERNS:
                if pd.Series([lipid]).str.contains(
                    pattern, regex=True, case=False, na=False
                ).any():
                    lipid_matches.append(f"lipid:{pattern}")

            # Check class pattern if ClassKey exists
            if 'ClassKey' in df.columns:
                class_value = df[df['LipidMolec'] == lipid]['ClassKey'].iloc[0] \
                    if not df[df['LipidMolec'] == lipid].empty else None

                if class_value and pd.Series([class_value]).str.contains(
                    StandardsService.INTERNAL_STANDARD_CLASS_PATTERN,
                    regex=True, case=False, na=False
                ).any():
                    lipid_matches.append(
                        f"class:{StandardsService.INTERNAL_STANDARD_CLASS_PATTERN}"
                    )

            if lipid_matches:
                matches[lipid] = lipid_matches

        return matches

    # ==================== Extraction Methods ====================

    @staticmethod
    def extract_standards(df: pd.DataFrame) -> StandardsExtractionResult:
        """
        Extract internal standards from a DataFrame, returning separate DataFrames.

        Args:
            df: DataFrame containing lipid data

        Returns:
            StandardsExtractionResult with data (without standards) and standards DataFrames

        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df is None or df.empty:
            raise ValueError(
                "DataFrame is empty. Cannot extract internal standards."
            )

        if 'LipidMolec' not in df.columns:
            raise ValueError(
                "DataFrame must have 'LipidMolec' column to extract internal standards."
            )

        # Detect standards
        is_standard = StandardsService.detect_standards(df)

        # Split DataFrames
        standards_df = df[is_standard].copy().reset_index(drop=True)
        data_df = df[~is_standard].copy().reset_index(drop=True)

        # Get matched patterns for informational purposes
        matched = StandardsService.get_matched_patterns(standards_df)
        all_patterns = []
        for patterns in matched.values():
            all_patterns.extend(patterns)
        unique_patterns = list(set(all_patterns))

        return StandardsExtractionResult(
            data_df=data_df,
            standards_df=standards_df,
            standards_count=len(standards_df),
            detection_patterns_matched=unique_patterns
        )

    @staticmethod
    def remove_standards_from_dataset(
        df: pd.DataFrame,
        standards_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove specific standards from a dataset.

        Args:
            df: Main dataset DataFrame
            standards_df: DataFrame containing standards to remove

        Returns:
            Tuple of (filtered DataFrame, list of removed lipid names)

        Raises:
            ValueError: If either DataFrame is None
        """
        if df is None:
            raise ValueError("Main DataFrame cannot be None.")

        if df.empty:
            return df.copy(), []

        if standards_df is None or standards_df.empty:
            return df.copy(), []

        if 'LipidMolec' not in df.columns:
            raise ValueError("Main DataFrame must have 'LipidMolec' column.")

        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards DataFrame must have 'LipidMolec' column.")

        # Get standard lipid names
        standard_names = set(standards_df['LipidMolec'].unique())

        # Find which standards exist in the main dataset
        existing_in_dataset = df[
            df['LipidMolec'].isin(standard_names)
        ]['LipidMolec'].unique().tolist()

        # Remove standards from main dataset
        filtered_df = df[~df['LipidMolec'].isin(standard_names)].copy()

        return filtered_df, existing_in_dataset

    # ==================== Validation Methods ====================

    @staticmethod
    def validate_standards(
        standards_df: pd.DataFrame,
        cleaned_df: Optional[pd.DataFrame] = None,
        check_existence: bool = False
    ) -> StandardsValidationResult:
        """Validate a standards DataFrame."""
        errors = []
        warnings = []

        # Early returns for structural issues
        if standards_df is None:
            errors.append("Standards DataFrame is None.")
            return StandardsValidationResult(is_valid=False, errors=errors, warnings=warnings)
        if standards_df.empty:
            errors.append("Standards DataFrame is empty.")
            return StandardsValidationResult(is_valid=False, errors=errors, warnings=warnings)
        if 'LipidMolec' not in standards_df.columns:
            errors.append("Standards DataFrame must have 'LipidMolec' column.")
            return StandardsValidationResult(is_valid=False, errors=errors, warnings=warnings)

        valid_count = StandardsService._validate_standards_content(
            standards_df, errors, warnings
        )

        if check_existence and cleaned_df is not None:
            StandardsService._check_standards_existence(standards_df, cleaned_df, errors)

        return StandardsValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            valid_standards_count=valid_count
        )

    @staticmethod
    def _validate_standards_content(
        standards_df: pd.DataFrame,
        errors: List[str],
        warnings: List[str],
    ) -> int:
        """Check lipid names, duplicates, ClassKey, and intensity columns. Returns valid count."""
        empty_lipids = standards_df['LipidMolec'].isna().sum()
        empty_lipids += (standards_df['LipidMolec'].astype(str).str.strip() == '').sum()
        if empty_lipids > 0:
            warnings.append(f"{empty_lipids} standard(s) have empty or null names.")

        duplicate_count = standards_df['LipidMolec'].duplicated().sum()
        if duplicate_count > 0:
            warnings.append(
                f"{duplicate_count} duplicate standard(s) found. "
                "Only the first occurrence will be used."
            )

        valid_standards = standards_df['LipidMolec'].dropna()
        valid_standards = valid_standards[valid_standards.astype(str).str.strip() != '']
        valid_count = valid_standards.nunique()

        if 'ClassKey' in standards_df.columns:
            missing_class = standards_df['ClassKey'].isna().sum()
            if missing_class > 0:
                warnings.append(
                    f"{missing_class} standard(s) have missing ClassKey. "
                    "ClassKey will be inferred from lipid name."
                )

        intensity_cols = [col for col in standards_df.columns if col.startswith('intensity[')]
        if not intensity_cols:
            warnings.append(
                "No intensity columns found in standards DataFrame. "
                "Standards may need intensity data for normalization."
            )

        return valid_count

    @staticmethod
    def _check_standards_existence(
        standards_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        errors: List[str],
    ) -> None:
        """Check if standards exist in the cleaned dataset."""
        if 'LipidMolec' not in cleaned_df.columns:
            return
        available_lipids = set(cleaned_df['LipidMolec'].unique())
        missing = set(standards_df['LipidMolec'].unique()) - available_lipids
        if missing:
            preview = list(missing)[:5]
            msg = f"Standards not found in dataset: {', '.join(preview)}"
            if len(missing) > 5:
                msg += f" and {len(missing) - 5} more"
            errors.append(msg)

    @staticmethod
    def validate_intensity_columns(
        standards_df: pd.DataFrame,
        expected_samples: List[str]
    ) -> StandardsValidationResult:
        """
        Validate that standards DataFrame has all required intensity columns.

        Args:
            standards_df: Standards DataFrame to validate
            expected_samples: List of expected sample names

        Returns:
            StandardsValidationResult with validation status
        """
        errors = []
        warnings = []

        if standards_df is None or standards_df.empty:
            errors.append("Standards DataFrame is empty.")
            return StandardsValidationResult(
                is_valid=False, errors=errors, warnings=warnings
            )

        expected_cols = [f"intensity[{s}]" for s in expected_samples]
        actual_cols = [col for col in standards_df.columns if col.startswith('intensity[')]

        missing_cols = set(expected_cols) - set(actual_cols)
        if missing_cols:
            errors.append(
                f"Standards DataFrame missing {len(missing_cols)} intensity column(s): "
                f"{', '.join(sorted(missing_cols)[:5])}"
                + (" ..." if len(missing_cols) > 5 else "")
            )

        extra_cols = set(actual_cols) - set(expected_cols)
        if extra_cols:
            warnings.append(
                f"Standards DataFrame has {len(extra_cols)} extra intensity column(s). "
                "These will be ignored."
            )

        valid_count = len(standards_df) if not errors else 0

        return StandardsValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            valid_standards_count=valid_count
        )

    # ==================== Processing Methods ====================

    @staticmethod
    def process_standards_file(
        uploaded_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        standards_in_dataset: bool = True
    ) -> StandardsProcessingResult:
        """
        Process an uploaded standards file.

        Args:
            uploaded_df: DataFrame from uploaded standards file
            cleaned_df: Main cleaned dataset
            standards_in_dataset: True if standards exist in main dataset (extract mode),
                                  False if uploading complete external standards

        Returns:
            StandardsProcessingResult with processed standards

        Raises:
            ValueError: If validation fails or required data is missing
        """
        if uploaded_df is None or uploaded_df.empty:
            raise ValueError("Uploaded standards file is empty.")

        if cleaned_df is None or cleaned_df.empty:
            raise ValueError("Cleaned dataset is empty. Upload data first.")

        # Get expected intensity columns from cleaned data
        expected_intensity_cols = [
            col for col in cleaned_df.columns
            if col.startswith('intensity[')
        ]
        if not expected_intensity_cols:
            raise ValueError("No intensity columns found in the main dataset.")

        # Standardize column names
        standardized_df = StandardsService._standardize_columns(
            uploaded_df, expected_intensity_cols
        )

        # Remove duplicates
        original_count = len(standardized_df)
        standardized_df = standardized_df.drop_duplicates(
            subset=['LipidMolec'], keep='first'
        )
        duplicates_removed = original_count - len(standardized_df)

        if standards_in_dataset:
            # Extract mode: standards exist in main dataset
            result_df = StandardsService._extract_standards_from_dataset(
                standardized_df, cleaned_df, expected_intensity_cols
            )
            source_mode = 'extract'
        else:
            # Complete mode: using external standards with their own intensity values
            result_df = StandardsService._validate_complete_standards(
                standardized_df, expected_intensity_cols
            )
            source_mode = 'complete'

        return StandardsProcessingResult(
            standards_df=result_df,
            duplicates_removed=duplicates_removed,
            standards_count=len(result_df),
            source_mode=source_mode
        )

    @staticmethod
    def _standardize_columns(
        df: pd.DataFrame,
        expected_intensity_cols: List[str]
    ) -> pd.DataFrame:
        """
        Standardize column names for a standards DataFrame.

        Column mapping:
        - 1st column -> LipidMolec
        - 2nd column -> ClassKey (if non-numeric), else intensity column
        - Remaining columns -> intensity[s1], intensity[s2], etc.

        Args:
            df: DataFrame to standardize
            expected_intensity_cols: Expected intensity column names

        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        cols = df.columns.tolist()

        if not cols:
            raise ValueError("Standards file has no columns.")

        new_col_names = {cols[0]: 'LipidMolec'}

        if len(cols) > 1:
            # Check if second column is ClassKey (non-numeric) or intensity (numeric)
            second_col_values = df[cols[1]]
            is_numeric = (
                pd.api.types.is_numeric_dtype(second_col_values) or
                pd.to_numeric(second_col_values, errors='coerce').notna().all()
            )

            if is_numeric:
                # No ClassKey provided, all remaining columns are intensities
                has_classkey = False
                intensity_start_idx = 1
            else:
                # Second column is ClassKey
                new_col_names[cols[1]] = 'ClassKey'
                has_classkey = True
                intensity_start_idx = 2

            # Rename remaining columns as intensity columns
            for i, col_idx in enumerate(range(intensity_start_idx, len(cols))):
                if i < len(expected_intensity_cols):
                    new_col_names[cols[col_idx]] = expected_intensity_cols[i]

        # Rename columns
        df = df.rename(columns=new_col_names)

        # Infer ClassKey from LipidMolec if not present
        if 'ClassKey' not in df.columns:
            df['ClassKey'] = df['LipidMolec'].apply(_infer_class_key)

        return df

    @staticmethod
    def _extract_standards_from_dataset(
        standards_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        expected_intensity_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extract standards from the main dataset (extract mode).

        Args:
            standards_df: DataFrame with standard names (LipidMolec column)
            cleaned_df: Main dataset to extract intensity values from
            expected_intensity_cols: Expected intensity column names

        Returns:
            DataFrame with standards and their intensity values from main dataset

        Raises:
            ValueError: If standards not found in dataset
        """
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must have LipidMolec column.")

        # Normalize uploaded standard names to LIPID MAPS notation so they
        # match the names in the cleaned dataset (which have already been
        # standardized during data ingestion).
        standards_df = standards_df.copy()
        standards_df['LipidMolec'] = standards_df['LipidMolec'].apply(
            DataStandardizationService.standardize_lipid_name
        )

        # Get unique standard names
        unique_standards = standards_df['LipidMolec'].unique()

        # Validate that standards exist in the dataset
        available_lipids = set(cleaned_df['LipidMolec'].unique())
        missing_standards = [s for s in unique_standards if s not in available_lipids]

        if missing_standards:
            preview = missing_standards[:5]
            msg = f"Standards not found in dataset: {', '.join(preview)}"
            if len(missing_standards) > 5:
                msg += f" and {len(missing_standards) - 5} more"
            raise ValueError(
                msg + ". If these are external standards, use the 'complete' mode."
            )

        # Build result DataFrame by extracting from main dataset
        result_rows = []
        for standard_name in unique_standards:
            standard_data = cleaned_df[cleaned_df['LipidMolec'] == standard_name]
            if not standard_data.empty:
                row = {'LipidMolec': standard_name}

                # Get ClassKey
                if 'ClassKey' in standard_data.columns:
                    row['ClassKey'] = standard_data['ClassKey'].iloc[0]
                elif 'ClassKey' in standards_df.columns:
                    class_val = standards_df[
                        standards_df['LipidMolec'] == standard_name
                    ]['ClassKey']
                    row['ClassKey'] = class_val.iloc[0] if not class_val.empty else \
                        _infer_class_key(standard_name)
                else:
                    row['ClassKey'] = _infer_class_key(standard_name)

                # Get intensity values
                for col in expected_intensity_cols:
                    if col in standard_data.columns:
                        row[col] = standard_data[col].iloc[0]
                    else:
                        row[col] = 0.0

                result_rows.append(row)

        if not result_rows:
            raise ValueError("No valid standards could be extracted from the dataset.")

        result_df = pd.DataFrame(result_rows)

        # Ensure correct column order
        ordered_cols = ['LipidMolec', 'ClassKey'] + expected_intensity_cols
        available_ordered = [c for c in ordered_cols if c in result_df.columns]
        result_df = result_df[available_ordered]

        return result_df

    @staticmethod
    def _validate_complete_standards(
        standards_df: pd.DataFrame,
        expected_intensity_cols: List[str]
    ) -> pd.DataFrame:
        """
        Validate and process complete standards dataset (external standards mode).

        Args:
            standards_df: DataFrame with standards and intensity values
            expected_intensity_cols: Expected intensity column names

        Returns:
            Validated standards DataFrame

        Raises:
            ValueError: If validation fails
        """
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must have LipidMolec as the first column.")

        # Check for intensity columns
        uploaded_intensity_cols = [
            col for col in standards_df.columns
            if col.startswith('intensity[')
        ]

        if not uploaded_intensity_cols:
            raise ValueError(
                "Standards file must contain intensity columns. "
                f"Expected columns: {', '.join(expected_intensity_cols[:3])}..."
            )

        # Check for missing intensity columns
        missing_cols = set(expected_intensity_cols) - set(uploaded_intensity_cols)
        if missing_cols:
            raise ValueError(
                f"Standards file is missing {len(missing_cols)} intensity column(s). "
                f"Your main dataset has {len(expected_intensity_cols)} samples."
            )

        # Select and order columns
        result_cols = ['LipidMolec', 'ClassKey'] + expected_intensity_cols
        available_cols = [c for c in result_cols if c in standards_df.columns]

        # Ensure ClassKey exists
        if 'ClassKey' not in standards_df.columns:
            standards_df = standards_df.copy()
            standards_df['ClassKey'] = standards_df['LipidMolec'].apply(
                _infer_class_key
            )
            available_cols = [c for c in result_cols if c in standards_df.columns]

        result_df = standards_df[available_cols].copy()

        # Convert intensity columns to numeric
        for col in expected_intensity_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

        return result_df

    # Backward-compatible alias — delegates to the canonical implementation
    # in DataStandardizationService.infer_class_key.
    _infer_class = staticmethod(_infer_class_key)

    # ==================== Utility Methods ====================

    @staticmethod
    def get_available_standards(standards_df: pd.DataFrame) -> List[str]:
        """
        Get list of available internal standard names.

        Args:
            standards_df: DataFrame with internal standards

        Returns:
            List of standard lipid molecule names
        """
        if standards_df is None or standards_df.empty:
            return []

        if 'LipidMolec' not in standards_df.columns:
            return []

        return list(standards_df['LipidMolec'].unique())

    @staticmethod
    def get_standards_by_class(standards_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group available internal standards by their lipid class.

        Args:
            standards_df: DataFrame with internal standards

        Returns:
            Dictionary mapping ClassKey -> list of standard names
        """
        if standards_df is None or standards_df.empty:
            return {}

        if 'LipidMolec' not in standards_df.columns:
            return {}

        # Ensure ClassKey exists
        if 'ClassKey' not in standards_df.columns:
            df = standards_df.copy()
            df['ClassKey'] = df['LipidMolec'].apply(_infer_class_key)
        else:
            df = standards_df

        return df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()

    @staticmethod
    def get_classes_with_standards(standards_df: pd.DataFrame) -> Set[str]:
        """
        Get set of lipid classes that have available standards.

        Args:
            standards_df: DataFrame with internal standards

        Returns:
            Set of ClassKey values that have standards
        """
        by_class = StandardsService.get_standards_by_class(standards_df)
        return set(by_class.keys())

    @staticmethod
    def get_classes_without_standards(
        data_df: pd.DataFrame,
        standards_df: pd.DataFrame
    ) -> Set[str]:
        """
        Get lipid classes in the data that don't have matching standards.

        Args:
            data_df: Main lipid data DataFrame
            standards_df: DataFrame with internal standards

        Returns:
            Set of ClassKey values without standards
        """
        if data_df is None or data_df.empty or 'ClassKey' not in data_df.columns:
            return set()

        data_classes = set(data_df['ClassKey'].unique())
        standards_classes = StandardsService.get_classes_with_standards(standards_df)

        return data_classes - standards_classes

    @staticmethod
    def suggest_standards_for_classes(
        standards_df: pd.DataFrame,
        target_classes: List[str]
    ) -> Dict[str, Optional[str]]:
        """
        Suggest standards for given lipid classes.

        For each class:
        - If a class-specific standard exists, suggest it
        - Otherwise, suggest None (user must choose from all available)

        Args:
            standards_df: DataFrame with internal standards
            target_classes: List of lipid classes needing standards

        Returns:
            Dictionary mapping class -> suggested standard (or None)
        """
        if standards_df is None or standards_df.empty:
            return {cls: None for cls in target_classes}

        by_class = StandardsService.get_standards_by_class(standards_df)

        suggestions = {}
        for lipid_class in target_classes:
            if lipid_class in by_class and by_class[lipid_class]:
                # Suggest first class-specific standard
                suggestions[lipid_class] = by_class[lipid_class][0]
            else:
                # No class-specific standard available
                suggestions[lipid_class] = None

        return suggestions

    @staticmethod
    def create_default_mapping(
        data_df: pd.DataFrame,
        standards_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Create a default class-to-standard mapping.

        Uses class-specific standards where available.
        For classes without specific standards, uses the first available standard.

        Args:
            data_df: Main lipid data DataFrame
            standards_df: DataFrame with internal standards

        Returns:
            Dictionary mapping ClassKey -> standard LipidMolec

        Raises:
            ValueError: If no standards available
        """
        if standards_df is None or standards_df.empty:
            raise ValueError("No internal standards available for mapping.")

        all_standards = StandardsService.get_available_standards(standards_df)
        if not all_standards:
            raise ValueError("No internal standards available for mapping.")

        by_class = StandardsService.get_standards_by_class(standards_df)
        default_standard = all_standards[0]

        if data_df is None or data_df.empty or 'ClassKey' not in data_df.columns:
            return {}

        mapping = {}
        for lipid_class in data_df['ClassKey'].unique():
            if lipid_class in by_class and by_class[lipid_class]:
                mapping[lipid_class] = by_class[lipid_class][0]
            else:
                mapping[lipid_class] = default_standard

        return mapping

    @staticmethod
    def count_standards(df: pd.DataFrame) -> int:
        """
        Count the number of internal standards in a DataFrame.

        Args:
            df: DataFrame to check

        Returns:
            Number of detected internal standards
        """
        if df is None or df.empty or 'LipidMolec' not in df.columns:
            return 0

        is_standard = StandardsService.detect_standards(df)
        return is_standard.sum()

    @staticmethod
    def has_standards(df: pd.DataFrame) -> bool:
        """
        Check if DataFrame contains any internal standards.

        Args:
            df: DataFrame to check

        Returns:
            True if at least one internal standard is detected
        """
        return StandardsService.count_standards(df) > 0
