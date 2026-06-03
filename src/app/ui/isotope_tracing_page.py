"""
Stable isotope tracing page UI.

A workflow fully separate from the lipidomics pipeline: upload the three
IsoCorrectoR input files (Measurement, Molecule, Element), run
natural-isotope-abundance correction via the real IsoCorrectoR R package, then
view and download the corrected tables.

UI-layer only — all data work goes through StreamlitAdapter / the workflow.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.constants import PAGE_LANDING
from app.models.isotope_tracing import IsotopeCorrectionConfig
from app.services.isotope_correction import check_isocorrector_available
from app.ui.landing_page import display_logo

# Golden sample dataset shipped with the repo (this file: src/app/ui/...).
_SAMPLE_DIR = Path(__file__).parents[3] / "sample_datasets" / "isotope_tracing"

# (CorrectionResult attribute, display label) for the output tables, corrected first.
_OUTPUT_TABLES = [
    ("corrected", "Corrected"),
    ("corrected_fractions", "Corrected Fractions"),
    ("mean_enrichment", "Mean Enrichment"),
    ("raw_data", "Raw Data"),
    ("residuals", "Residuals"),
    ("relative_residuals", "Relative Residuals"),
]


def _go_home() -> None:
    """Clear isotope state and return to the landing page.

    Runs as an on_click callback — Streamlit reruns automatically afterward,
    so no explicit st.rerun() (which is illegal inside a callback).
    """
    StreamlitAdapter.reset_module_state("iso_")
    st.session_state.page = PAGE_LANDING


# Setting widget key -> default value (defaults match IsoCorrectoR, except
# tracer-impurity which we default on). Used to seed the widgets so the sample
# loader can override them cleanly.
_SETTING_DEFAULTS = {
    "iso_correct_tracer_impurity": IsotopeCorrectionConfig().correct_tracer_impurity,
    "iso_correct_tracer_element_core": IsotopeCorrectionConfig().correct_tracer_element_core,
    "iso_calculate_mean_enrichment": IsotopeCorrectionConfig().calculate_mean_enrichment,
    "iso_ultra_high_res": IsotopeCorrectionConfig().ultra_high_res,
    "iso_correct_also_monoisotopic": IsotopeCorrectionConfig().correct_also_monoisotopic,
    "iso_calculation_threshold_uhr": IsotopeCorrectionConfig().calculation_threshold_uhr,
}


def _load_sample_data() -> None:
    """Load the bundled golden input files (and their matching settings).

    The sample is an ultra-high-resolution dataset (measurement IDs like
    ``BMP_18:1_18:1_C0``), so UHR mode must be on for IsoCorrectoR to parse it.
    Seeds the settings to the configuration that produced the golden output.
    """
    st.session_state.iso_measurement_df = pd.read_csv(_SAMPLE_DIR / "MeasurementFile.csv")
    st.session_state.iso_molecule_df = pd.read_csv(_SAMPLE_DIR / "MoleculeFile.csv")
    st.session_state.iso_element_df = pd.read_csv(_SAMPLE_DIR / "ElementFile.csv")
    st.session_state.iso_ultra_high_res = True


def _resolve_input(uploaded, session_key: str) -> Optional[pd.DataFrame]:
    """An uploaded file takes precedence; otherwise use what's in session."""
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return st.session_state.get(session_key)


def _config_from_widgets() -> IsotopeCorrectionConfig:
    """Build the correction config from the sidebar setting widgets."""
    return IsotopeCorrectionConfig(
        correct_tracer_impurity=st.session_state.iso_correct_tracer_impurity,
        correct_tracer_element_core=st.session_state.iso_correct_tracer_element_core,
        calculate_mean_enrichment=st.session_state.iso_calculate_mean_enrichment,
        ultra_high_res=st.session_state.iso_ultra_high_res,
        correct_also_monoisotopic=st.session_state.iso_correct_also_monoisotopic,
        calculation_threshold_uhr=st.session_state.iso_calculation_threshold_uhr,
    )


def _download_table(df: pd.DataFrame, label: str, key: str) -> None:
    """Download a result table as CSV, preserving the isotopologue-ID index."""
    st.download_button(
        label=f"Download {label} CSV",
        data=df.to_csv(index=True).encode("utf-8"),
        file_name=f"IsoCorrectoR_{label.replace(' ', '')}.csv",
        mime="text/csv",
        key=key,
    )


def _display_sidebar(r_available: bool):
    """Render the sidebar (uploaders, sample data, settings, actions).

    Returns the (measurement_df, molecule_df, element_df, run_clicked) tuple.
    """
    # Seed setting widgets once so the sample loader / Run can read them by key.
    for key, default in _SETTING_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown("### Stable Isotope Tracing")

        st.button("Load sample data", key="iso_load_sample", on_click=_load_sample_data,
                  use_container_width=True, help="Load the bundled IsoCorrectoR example dataset.")

        measurement_up = st.file_uploader("Measurement file (CSV)", type="csv", key="iso_measurement_uploader")
        molecule_up = st.file_uploader("Molecule file (CSV)", type="csv", key="iso_molecule_uploader")
        element_up = st.file_uploader("Element file (CSV)", type="csv", key="iso_element_uploader")

        measurement_df = _resolve_input(measurement_up, "iso_measurement_df")
        molecule_df = _resolve_input(molecule_up, "iso_molecule_df")
        element_df = _resolve_input(element_up, "iso_element_df")

        with st.expander("Correction settings", expanded=False):
            st.checkbox("Correct tracer impurity", key="iso_correct_tracer_impurity")
            st.checkbox("Correct tracer element in core molecule", key="iso_correct_tracer_element_core")
            st.checkbox("Calculate mean enrichment", key="iso_calculate_mean_enrichment")
            st.checkbox("Ultra-high-resolution mode", key="iso_ultra_high_res")
            st.checkbox("Also correct monoisotopic peak", key="iso_correct_also_monoisotopic")
            st.number_input("UHR calculation limit", min_value=0.0, step=1.0,
                            key="iso_calculation_threshold_uhr",
                            help="Limit value used by the ultra-high-resolution calculation.")

        have_all = measurement_df is not None and molecule_df is not None and element_df is not None
        run_clicked = st.button(
            "Run correction", type="primary", use_container_width=True,
            disabled=not (have_all and r_available),
            key="iso_run",
        )
        if not r_available:
            st.caption("Run is disabled: R runtime unavailable.")
        elif not have_all:
            st.caption("Upload all three files (or load sample data) to enable.")

        st.markdown("---")
        st.button("← Back to Home", key="iso_back_home", on_click=_go_home, use_container_width=True)

    return measurement_df, molecule_df, element_df, run_clicked


def _display_results(result) -> None:
    """Render validation errors / R errors / the output tables and log."""
    if result is None:
        return

    if not result.success:
        if result.validation_errors:
            st.error("The input files could not be validated:")
            for err in result.validation_errors:
                st.markdown(f"- {err}")
        else:
            st.error(result.error_message or "Correction failed.")
        return

    correction = result.correction
    st.success("Correction complete.")

    # Primary output: the corrected intensities.
    st.markdown("#### Corrected intensities")
    st.dataframe(correction.corrected, use_container_width=True)
    _download_table(correction.corrected, "Corrected", key="iso_dl_corrected_main")

    # All other IsoCorrectoR outputs.
    with st.expander("All IsoCorrectoR outputs", expanded=False):
        for attr, label in _OUTPUT_TABLES:
            table = getattr(correction, attr)
            if table is None:
                continue
            st.markdown(f"**{label}**")
            st.dataframe(table, use_container_width=True)
            _download_table(table, label, key=f"iso_dl_{attr}")
            st.markdown("---")

    if correction.log_text:
        with st.expander("IsoCorrectoR run log", expanded=False):
            st.code(correction.log_text)


def display_isotope_tracing_page() -> None:
    """Display the stable isotope tracing page."""
    _, center, _ = st.columns([1, 3, 1])

    r_available = check_isocorrector_available()

    measurement_df, molecule_df, element_df, run_clicked = _display_sidebar(r_available)

    with center:
        display_logo(centered=True)
        st.markdown(
            "Run natural-isotope-abundance correction on stable isotope tracing data "
            "using the IsoCorrectoR package. Upload the **Measurement**, **Molecule**, "
            "and **Element** files in the sidebar, or load the sample dataset."
        )

        if not r_available:
            st.warning(
                "**R runtime unavailable.** The stable isotope tracing module needs R "
                "with the IsoCorrectoR package installed. The rest of LipidCruncher is "
                "unaffected — use **← Back to Home** to return to general lipidomics."
            )

        if run_clicked:
            config = _config_from_widgets()
            st.session_state.iso_config = config
            st.session_state.iso_measurement_df = measurement_df
            st.session_state.iso_molecule_df = molecule_df
            st.session_state.iso_element_df = element_df
            st.session_state.iso_result = StreamlitAdapter.run_isotope_correction(
                measurement_df, molecule_df, element_df, config
            )

        if st.session_state.get("iso_result") is None and not run_clicked:
            st.info("Provide the three input files and click **Run correction** in the sidebar.")

        _display_results(st.session_state.get("iso_result"))
