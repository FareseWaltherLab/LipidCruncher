"""
Isotope tracing workflow.

Orchestrates the stable-isotope-tracing path: validate the three input files →
run IsoCorrectoR → typed result. Mirrors DataIngestionWorkflow: the workflow
owns success/error state, the service raises typed exceptions. No Streamlit
dependencies.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ..models.isotope_tracing import IsotopeCorrectionConfig
from ..services.isotope_correction import (
    IsotopeCorrectionService,
    CorrectionResult,
    validate_inputs,
    RRuntimeError,
)


@dataclass
class IsotopeTracingResult:
    """Complete result of the isotope tracing workflow."""
    success: bool = False
    correction: Optional[CorrectionResult] = None
    validation_errors: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class IsotopeTracingWorkflow:
    """Validate inputs and run natural-isotope-abundance correction.

    All methods are static — no instance state.
    """

    @staticmethod
    def run(
        measurement_df: pd.DataFrame,
        molecule_df: pd.DataFrame,
        element_df: pd.DataFrame,
        config: IsotopeCorrectionConfig,
    ) -> IsotopeTracingResult:
        """Execute the isotope tracing workflow.

        Pipeline:
        1. Structurally validate the three input files (pure Python).
        2. Run IsoCorrectoR via the service.

        Args:
            measurement_df: Measurement file.
            molecule_df: Molecule file.
            element_df: Element file.
            config: IsoCorrectoR settings.

        Returns:
            IsotopeTracingResult — success with the correction, or failure with
            validation_errors (bad inputs) or error_message (R runtime failure).
        """
        validation_errors = validate_inputs(
            measurement_df, molecule_df, element_df,
            ultra_high_res=config.ultra_high_res,
        )
        if validation_errors:
            return IsotopeTracingResult(
                success=False, validation_errors=validation_errors
            )

        try:
            correction = IsotopeCorrectionService.run_correction(
                measurement_df, molecule_df, element_df, config
            )
        except RRuntimeError as e:
            return IsotopeTracingResult(success=False, error_message=str(e))

        return IsotopeTracingResult(success=True, correction=correction)
