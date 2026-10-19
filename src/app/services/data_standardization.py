"""
Data standardization service for lipid data.

Validates and standardizes data from all supported formats into a common schema:
- LipidMolec column (standardized lipid names)
- ClassKey column (inferred from lipid names)
- intensity[s1], intensity[s2], ... columns

Pure logic — no Streamlit dependencies.
"""

import csv
from typing import Union

import pandas as pd

from app.services import lipid_names
from app.services.standardization import generic, lipidsearch, msdial
from app.services.standardization.msdial import (  # noqa: F401 - re-exported
    MSDIALFeatures, MSDIALOverrideResult, MSDIALStructure,
)
from app.services.standardization.result import StandardizationResult
from app.services.format_detection import DataFormat


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
                return lipidsearch.process(data)
            elif data_format == DataFormat.MSDIAL:
                return msdial.process(data, msdial_use_normalized)
            elif data_format == DataFormat.METABOLOMICS_WORKBENCH:
                return DataStandardizationService._process_metabolomics_workbench(data)
            elif data_format == DataFormat.GENERIC:
                return generic.process(data)
            else:
                return StandardizationResult(
                    success=False,
                    message=f"Unsupported format: {data_format}",
                )
        except (ValueError, KeyError) as e:
            format_name = data_format.value if hasattr(data_format, 'value') else str(data_format)
            return StandardizationResult(
                success=False,
                message=(
                    f"Error during {format_name} data standardization.\n"
                    f"  Type: {type(e).__name__}\n"
                    f"  Detail: {e}\n"
                    f"  Hint: Verify the uploaded file matches the selected format."
                ),
            )

    # ------------------------------------------------------------------
    # Lipid name helpers (services/lipid_names.py)
    # ------------------------------------------------------------------

    standardize_lipid_name = staticmethod(lipid_names.standardize_lipid_name)
    infer_class_key = staticmethod(lipid_names.infer_class_key)

    # ------------------------------------------------------------------
    # LipidSearch (services/standardization/lipidsearch.py)
    # ------------------------------------------------------------------

    _process_lipidsearch = staticmethod(lipidsearch.process)
    standardize_lipidsearch_with_alignment = staticmethod(
        lipidsearch.standardize_with_alignment
    )

    # ------------------------------------------------------------------
    # Generic (services/standardization/generic.py)
    # ------------------------------------------------------------------

    _is_classkey_column = staticmethod(generic.is_classkey_column)
    _process_generic = staticmethod(generic.process)
    _build_generic_column_map = staticmethod(generic.build_column_map)

    # ------------------------------------------------------------------
    # MS-DIAL (services/standardization/msdial.py)
    # ------------------------------------------------------------------

    apply_msdial_sample_override = staticmethod(msdial.apply_sample_override)
    _detect_msdial_sample_columns = staticmethod(msdial.detect_sample_columns)
    _detect_msdial_structure = staticmethod(msdial.detect_structure)
    _process_msdial = staticmethod(msdial.process)

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

            # csv.reader keeps a quoted name such as "1,2-DAG 34:1" as one
            # field; a plain split on ',' silently dropped those rows.
            parsed = list(csv.reader(data_section))
            samples_line = parsed[0]
            factors_line = parsed[1]

            sample_names = samples_line[1:]
            conditions = factors_line[1:]

            cleaned_conditions = []
            for cond in conditions:
                cond = cond.strip()
                if cond.startswith('Condition:'):
                    cond = cond.replace('Condition:', '').strip()
                cleaned_conditions.append(cond)

            data_rows = []
            for values in parsed[2:]:
                if len(values) == len(sample_names) + 1:
                    data_rows.append(values)

            if not data_rows:
                return StandardizationResult(False, "No valid data rows found")

            # Number samples condition by condition (stable within a condition):
            # the experiment assigns s1..sN to conditions contiguously, so an
            # interleaved factor order would otherwise put samples in the wrong
            # condition.
            first_seen = list(dict.fromkeys(cleaned_conditions))
            order = sorted(
                range(len(sample_names)),
                key=lambda i: first_seen.index(cleaned_conditions[i]),
            )
            sample_names = [sample_names[i] for i in order]
            cleaned_conditions = [cleaned_conditions[i] for i in order]
            data_rows = [[row[0]] + [row[i + 1] for i in order] for row in data_rows]

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
            column_mapping = pd.DataFrame({
                'standardized_name': ['LipidMolec'] + intensity_cols,
                'original_name': [samples_line[0].strip()] + [name.strip() for name in sample_names],
            })

            return StandardizationResult(
                success=True,
                message="Data successfully standardized",
                standardized_df=df,
                column_mapping=column_mapping,
                n_intensity_cols=len(sample_names),
                workbench_conditions=wb_conditions,
                workbench_samples=wb_samples,
            )

        except (ValueError, KeyError, IndexError) as e:
            return StandardizationResult(
                False, f"Error standardizing Metabolomics Workbench data: {e}"
            )
