"""
UI tests for the stable isotope tracing page.

Uses AppTest.from_function (mirroring tests/ui/conftest.py) to render the page
without importing main_app.py. R-gated: the page runs the real IsoCorrectoR, so
these are skipped where Rscript/IsoCorrectoR isn't installed (CI without R still
passes). The pure-Python validation logic itself is covered by
tests/unit/test_isotope_tracing_validation.py.
"""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# Pre-import to avoid circular import when AppTest runs the script (see conftest).
import app.services.format_detection  # noqa: F401
import app.constants  # noqa: F401
from app.services.isotope_correction import check_isocorrector_available

DEFAULT_TIMEOUT = 120

pytestmark = pytest.mark.skipif(
    not check_isocorrector_available(),
    reason="Rscript / IsoCorrectoR not available in this environment",
)


def isotope_page_script():
    """Render the isotope tracing page in isolation."""
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    import app.ui.landing_page as lp
    lp.PDF2IMAGE_AVAILABLE = False  # avoid PDF conversion timeouts
    from app.ui.isotope_tracing_page import display_isotope_tracing_page
    display_isotope_tracing_page()


def _make_app():
    return AppTest.from_function(isotope_page_script, default_timeout=DEFAULT_TIMEOUT)


class TestSampleRun:
    """Load sample data → run → reproduce the golden output through the UI."""

    def test_sample_load_enables_uhr_and_reproduces_golden(self):
        at = _make_app().run()
        at.button(key="iso_load_sample").click().run()
        # Loading the (UHR) sample dataset must enable UHR mode.
        assert at.session_state["iso_ultra_high_res"] is True

        run_btn = at.button(key="iso_run")
        assert not run_btn.disabled
        run_btn.click().run()

        result = at.session_state["iso_result"]
        assert result.success, result.error_message
        corrected = result.correction.corrected
        value = corrected.loc["PC_34:1_C0"].iloc[0]
        assert abs(value - 85890702767.1415) < 1.0
        # Success banner and the corrected table render.
        assert any("complete" in m.value.lower() for m in at.success)
        assert len(at.dataframe) >= 1


class TestValidationPath:
    """Bad inputs surface validation errors in the UI without crashing."""

    def test_mismatched_molecule_file_shows_errors(self):
        at = _make_app()
        at.session_state["iso_measurement_df"] = pd.DataFrame(
            {"ids": ["BMP_18:1_18:1_C0"], "sample_a": [100.0]}
        )
        at.session_state["iso_molecule_df"] = pd.DataFrame({"Molecule": ["Unrelated"]})
        at.session_state["iso_element_df"] = pd.DataFrame({"Element": ["C"]})
        at.run()

        at.button(key="iso_run").click().run()
        result = at.session_state["iso_result"]
        assert result.success is False
        assert result.validation_errors
        assert len(at.error) > 0
