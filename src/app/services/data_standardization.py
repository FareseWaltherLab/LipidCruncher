"""
Data standardization service for lipid data.

Validates and standardizes data from all supported formats into a common schema:
- LipidMolec column (standardized lipid names)
- ClassKey column (inferred from lipid names)
- intensity[s1], intensity[s2], ... columns

The entry point; the work lives in ``services/lipid_names.py`` and one
module per format under ``services/standardization/``. The static methods
below keep the names the rest of the app and the tests import.

Pure logic — no Streamlit dependencies.
"""

from typing import Union

import pandas as pd

from app.services import lipid_names
from app.services.standardization import generic, lipidsearch, msdial, workbench
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
                return workbench.process(data)
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
    # Metabolomics Workbench (services/standardization/workbench.py)
    # ------------------------------------------------------------------

    _process_metabolomics_workbench = staticmethod(workbench.process)

