"""
The isotope tracing page's part of the adapter: its session state, its reset
stage and the IsoCorrectoR calls. StreamlitAdapter inherits IsotopeAdapter;
SessionState inherits IsotopeState.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from ..models.isotope_tracing import IsotopeCorrectionConfig
from ..workflows.isotope_tracing import IsotopeTracingWorkflow, IsotopeTracingResult


@dataclass
class IsotopeState:
    """The isotope tracing page's session state fields (see SessionState)."""

    # --- Isotope tracing (owner: isotope_tracing_page.py) ---
    iso_measurement_df: Optional[pd.DataFrame] = None
    iso_molecule_df: Optional[pd.DataFrame] = None
    iso_element_df: Optional[pd.DataFrame] = None
    iso_result: Optional[IsotopeTracingResult] = None
    # Fingerprint of the inputs + settings iso_result was computed from.
    iso_result_key: Optional[str] = None


# The 'isotope' stage of _STAGES (see _Stage).
ISOTOPE_KEYS = dict(
    fields=frozenset({
        'iso_measurement_df', 'iso_molecule_df', 'iso_element_df',
        'iso_result', 'iso_result_key',
    }),
    widgets=frozenset({
        'iso_measurement_uploader', 'iso_molecule_uploader',
        'iso_element_uploader', 'iso_load_sample', 'iso_run',
        'iso_correct_tracer_impurity', 'iso_correct_tracer_element_core',
        'iso_calculate_mean_enrichment', 'iso_ultra_high_res',
        'iso_correct_also_monoisotopic', 'iso_calculation_threshold_uhr',
        'iso_plot_molecule', 'iso_plot_conditions', 'iso_plot_isotopologues',
        'iso_plot_rescale',
    }),
    # The conditions the plot's condition picker last offered.
    plain=frozenset({'iso_plot_condition_options'}),
    # iso_plot_groups_<digest>: the sample/condition table, one per sample set.
    prefixes=('iso_dl_', 'iso_plot_groups_'),
)


class _UncachedResult(Exception):
    """Carries a result out of a cached function without it being cached."""

    def __init__(self, result):
        super().__init__()
        self.result = result


@st.cache_resource(show_spinner=False)
def _isocorrector_available() -> bool:
    """Whether R + IsoCorrectoR can run here, probed once per process.

    The probe starts Rscript (~0.7 s), which every rerun of the isotope
    tracing page used to pay. R comes with the image, so the answer cannot
    change under a running process.
    """
    from ..services.isotope_correction import check_isocorrector_available
    return check_isocorrector_available()


class IsotopeAdapter:
    """The isotope tracing page's adapter calls; StreamlitAdapter inherits them."""

    @staticmethod
    def isocorrector_available() -> bool:
        """Whether the isotope tracing page can run (R + IsoCorrectoR)."""
        return _isocorrector_available()

    @classmethod
    def run_isotope_correction(
        cls,
        measurement_df: pd.DataFrame,
        molecule_df: pd.DataFrame,
        element_df: pd.DataFrame,
        config: IsotopeCorrectionConfig,
    ) -> IsotopeTracingResult:
        """Isotope tracing workflow, cached on success.

        Keyed on the three input DataFrames and the config, so re-running with
        unchanged inputs skips the (slow) R subprocess. A run that failed in R
        (a timeout, a crash, inputs IsoCorrectoR rejected) is not cached, so
        retrying really re-runs it.

        Args:
            measurement_df: Measurement file.
            molecule_df: Molecule file.
            element_df: Element file.
            config: IsoCorrectoR settings.

        Returns:
            IsotopeTracingResult — success with the correction, or failure with
            validation_errors / error_message.
        """
        try:
            return cls._run_isotope_correction_cached(
                measurement_df, molecule_df, element_df, config
            )
        except _UncachedResult as uncached:
            return uncached.result

    @staticmethod
    @st.cache_data(
        show_spinner="Running IsoCorrectoR...",
        hash_funcs={IsotopeCorrectionConfig: hash},
    )
    def _run_isotope_correction_cached(
        measurement_df: pd.DataFrame,
        molecule_df: pd.DataFrame,
        element_df: pd.DataFrame,
        config: IsotopeCorrectionConfig,
    ) -> IsotopeTracingResult:
        result = IsotopeTracingWorkflow.run(
            measurement_df, molecule_df, element_df, config
        )
        if result.error_message:
            raise _UncachedResult(result)  # Streamlit does not cache exceptions
        return result
