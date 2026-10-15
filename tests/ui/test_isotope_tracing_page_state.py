"""Isotope tracing page state, without R: the R run is replaced by a fixed
result, so these run in CI too."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# Pre-import to avoid circular import when AppTest runs the script.
import app.constants  # noqa: F401
import app.services.format_detection  # noqa: F401
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.isotope_correction.service import CorrectionResult
from app.workflows.isotope_tracing import IsotopeTracingResult


def page_script():
    import app.ui.isotope_tracing_page as page
    import app.ui.landing_page as lp
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    lp.PDF2IMAGE_AVAILABLE = False
    page.check_isocorrector_available = lambda: True
    page.display_isotope_tracing_page()


@pytest.fixture
def runs(monkeypatch):
    runs = []
    table = pd.DataFrame({"s1": [1.0]}, index=["PC_34:1_C0"])

    def _run(measurement_df, molecule_df, element_df, config):
        runs.append(config)
        return IsotopeTracingResult(success=True, correction=CorrectionResult(
            corrected=table, raw_data=table,
        ))

    monkeypatch.setattr(StreamlitAdapter, "run_isotope_correction",
                        staticmethod(_run))
    return runs


def _shows_results(at):
    return any(s.value == "Correction complete." for s in at.success)


def _stale(at):
    return any("changed since the last run" in i.value for i in at.info)


class TestResultsMatchTheInputs:
    """Regression (IQ4): the last result stayed on screen, under "Correction
    complete" and with its downloads, after a new upload, Load sample data
    or a settings change, until Run was clicked again."""

    def _ran(self, runs):
        at = AppTest.from_function(page_script, default_timeout=30).run()
        at.button(key="iso_load_sample").click().run()
        at.button(key="iso_run").click().run()
        assert len(runs) == 1 and _shows_results(at)
        return at

    def test_changing_a_setting_hides_the_old_result(self, runs):
        at = self._ran(runs)
        at.checkbox(key="iso_correct_tracer_impurity").uncheck().run()
        assert not _shows_results(at)
        assert _stale(at)

    def test_reloading_different_data_hides_the_old_result(self, runs):
        at = self._ran(runs)
        at.session_state["iso_element_df"] = pd.DataFrame({"Element": ["C"]})
        at.run()
        assert not _shows_results(at)
        assert _stale(at)

    def test_unrelated_rerun_keeps_the_result(self, runs):
        at = self._ran(runs)
        at.run()
        assert _shows_results(at) and not _stale(at)

    def test_running_again_shows_the_new_result(self, runs):
        at = self._ran(runs)
        at.checkbox(key="iso_correct_tracer_impurity").uncheck().run()
        at.button(key="iso_run").click().run()
        assert len(runs) == 2
        assert runs[1].correct_tracer_impurity is False
        assert _shows_results(at) and not _stale(at)


class TestSampleDataResetsTheSettings:
    """Regression (IQ8): Load sample data set only UHR mode, so a setting
    changed earlier (e.g. tracer impurity off) carried over and the sample
    run no longer reproduced the golden output it is meant to show."""

    def test_every_setting_returns_to_the_golden_configuration(self, runs):
        from app.models.isotope_tracing import IsotopeCorrectionConfig

        at = AppTest.from_function(page_script, default_timeout=30).run()
        at.checkbox(key="iso_correct_tracer_impurity").uncheck().run()
        at.checkbox(key="iso_calculate_mean_enrichment").uncheck().run()
        at.button(key="iso_load_sample").click().run()
        at.button(key="iso_run").click().run()
        assert runs[-1] == IsotopeCorrectionConfig(ultra_high_res=True)
