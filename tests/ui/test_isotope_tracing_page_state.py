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
    monkeypatch.setattr(StreamlitAdapter, "isocorrector_available",
                        staticmethod(lambda: True))
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


def page_script_without_r():
    import app.ui.isotope_tracing_page as page
    import app.ui.landing_page as lp
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    lp.PDF2IMAGE_AVAILABLE = False
    page.display_isotope_tracing_page()


def test_without_r_the_page_says_so_and_cannot_run(runs, monkeypatch):
    monkeypatch.setattr(StreamlitAdapter, "isocorrector_available",
                        staticmethod(lambda: False))
    at = AppTest.from_function(page_script_without_r, default_timeout=30).run()
    assert not at.exception
    assert any("R runtime unavailable" in w.value for w in at.warning)
    at.sidebar.button(key="iso_load_sample").click().run()
    assert at.sidebar.button(key="iso_run").disabled
    assert any("Run is disabled: R runtime unavailable." in c.value
               for c in at.sidebar.caption)
    assert runs == []


def test_r_is_probed_once_per_process(monkeypatch):
    """IQ7: every rerun of the page started Rscript to check for R."""
    import app.services.isotope_correction as iso
    from app.adapters.streamlit_adapter import _isocorrector_available

    probes = []

    def _probe():
        probes.append(1)
        return False

    monkeypatch.setattr(iso, "check_isocorrector_available", _probe)
    _isocorrector_available.clear()
    try:
        at = AppTest.from_function(page_script_without_r, default_timeout=30)
        at.run()
        at.run()
        AppTest.from_function(page_script_without_r, default_timeout=30).run()
    finally:
        _isocorrector_available.clear()
    assert not at.exception
    assert len(probes) == 1
