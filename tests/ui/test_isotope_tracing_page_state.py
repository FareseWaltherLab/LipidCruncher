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


class TestRemovedUploadLeavesTheInputs:
    """Regression: Run kept a copy of the uploaded files, so after removing
    one from its uploader (its ×) the old result stayed on screen under
    "Correction complete", and Run stayed enabled on the removed file."""

    def test_removed_file_is_not_used(self, runs, fake_uploads):
        from tests.ui.conftest import FakeUpload
        from app.ui.isotope_tracing_page import _SAMPLE_DIR

        for name in ("Measurement", "Molecule", "Element"):
            fake_uploads[f"{name} file"] = FakeUpload(
                f"{name}.csv", (_SAMPLE_DIR / f"{name}File.csv").read_bytes(),
            )
        at = AppTest.from_function(page_script, default_timeout=30).run()
        at.button(key="iso_run").click().run()
        assert len(runs) == 1 and _shows_results(at)

        del fake_uploads["Measurement file"]
        at.run()
        assert not _shows_results(at)
        assert at.sidebar.button(key="iso_run").disabled


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


class TestGeneratedInputFiles:
    """The Molecule and Element files written from the Measurement file's
    lipids (power-user request: users wrote them by hand)."""

    @pytest.fixture
    def frames(self, monkeypatch):
        frames = []

        def _run(measurement_df, molecule_df, element_df, config):
            frames.append((molecule_df, element_df))
            table = pd.DataFrame({"s1": [1.0]}, index=["PC_34:1_C0"])
            return IsotopeTracingResult(success=True, correction=CorrectionResult(
                corrected=table, raw_data=table,
            ))

        monkeypatch.setattr(StreamlitAdapter, "run_isotope_correction",
                            staticmethod(_run))
        monkeypatch.setattr(StreamlitAdapter, "isocorrector_available",
                            staticmethod(lambda: True))
        return frames

    @staticmethod
    def _generate(at):
        at.sidebar.radio(key="iso_input_source").set_value(
            "Generate from the Measurement file").run()
        assert not at.exception, at.exception
        return at

    def test_run_uses_the_generated_files(self, frames):
        at = AppTest.from_function(page_script, default_timeout=30).run()
        at.button(key="iso_load_sample").click().run()
        self._generate(at)
        assert not at.sidebar.button(key="iso_run").disabled
        at.sidebar.button(key="iso_run").click().run()
        molecule_df, element_df = frames[-1]
        assert molecule_df["MS ion or MS/MS product ion"].tolist() == [
            "C42H78O10P1LabC6", "C42H81N1O8P1LabC6",
        ]
        assert sorted(element_df["Element"]) == ["C", "H", "N", "O", "P"]
        assert _shows_results(at)

    def test_purity_reaches_the_element_file(self, frames):
        at = AppTest.from_function(page_script, default_timeout=30).run()
        at.button(key="iso_load_sample").click().run()
        self._generate(at)
        at.number_input(key="iso_tracer_purity").set_value(0.95).run()
        at.sidebar.button(key="iso_run").click().run()
        element_df = frames[-1][1].set_index("Element")
        assert element_df.loc["C", "Tracer purity"] == 0.95

    def test_unknown_lipid_class_blocks_the_run(self, frames):
        at = AppTest.from_function(page_script, default_timeout=30)
        at.session_state["iso_measurement_df"] = pd.DataFrame({
            "Measurements/Samples": ["Mystery_1:0_0", "Mystery_1:0_1"],
            "s1": [1.0, 2.0],
        })
        at.run()
        self._generate(at)
        assert any("enter its formula" in w.value for w in at.warning)
        assert at.sidebar.button(key="iso_run").disabled
        assert frames == []

    def test_without_a_measurement_file_it_asks_for_one(self, frames):
        at = self._generate(AppTest.from_function(page_script, default_timeout=30).run())
        assert any("Upload the Measurement file" in i.value for i in at.info)
        assert at.sidebar.button(key="iso_run").disabled
