"""IsotopeCorrectionService without R: the subprocess is replaced, so these
run in CI (which has no R) and pin how R's outcome is reported, how the
output files are read back, and how R is detected."""
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from app.models.isotope_tracing import IsotopeCorrectionConfig
from app.services.isotope_correction import service as svc
from app.services.isotope_correction.exceptions import (
    IsotopeInputError,
    RRuntimeError,
)


@pytest.fixture
def fake_r(monkeypatch):
    """Make run_correction see an R that exits with the given outcome."""
    monkeypatch.setattr(svc, "require_isocorrector", lambda: "Rscript")

    def _set(returncode=1, stdout="", stderr=""):
        def _run(args, **kwargs):
            return subprocess.CompletedProcess(args, returncode, stdout, stderr)
        monkeypatch.setattr(svc.subprocess, "run", _run)
    return _set


def _run():
    df = pd.DataFrame({"a": [1]})
    return svc.IsotopeCorrectionService.run_correction(
        df, df, df, IsotopeCorrectionConfig(),
    )


class TestRFailuresAreReadable:
    """Regression (IQ2): the whole of stderr (~1.7 KB of readr column specs
    and a banner) was the message, with IsoCorrectoR's reason in the middle
    and a pointer to a result log in a directory already deleted."""

    def test_rejected_inputs_report_only_the_reason(self, fake_r):
        noise = "Column specification\n" + "-" * 40 + "\n" * 30
        fake_r(stderr=(
            noise
            + "ISOCORRECTOR_ERROR: One or more elements in the provided "
              "formula were not found in the element file.\n"
            + noise
        ))
        with pytest.raises(IsotopeInputError) as err:
            _run()
        assert str(err.value) == (
            "IsoCorrectoR could not correct these files: One or more "
            "elements in the provided formula were not found in the "
            "element file."
        )

    def test_other_failures_show_the_end_of_stderr(self, fake_r):
        fake_r(stderr="x" * 5000 + "\nError in foo(): the real cause\n")
        with pytest.raises(RRuntimeError) as err:
            _run()
        assert "the real cause" in str(err.value)
        assert len(str(err.value)) < 1000


class TestFailedRunsAreNotCached:
    """Regression (IQ5): the cached adapter stored failed results too, so a
    timeout or R crash was replayed instantly on every retry."""

    @pytest.fixture
    def workflow_calls(self, monkeypatch):
        from app.adapters.streamlit_adapter import StreamlitAdapter
        from app.workflows.isotope_tracing import (
            IsotopeTracingResult,
            IsotopeTracingWorkflow,
        )

        calls = []
        outcome = {"result": IsotopeTracingResult(
            success=False, error_message="IsoCorrectoR did not finish",
        )}

        def _run(*args):
            calls.append(args)
            return outcome["result"]

        monkeypatch.setattr(IsotopeTracingWorkflow, "run", staticmethod(_run))
        StreamlitAdapter._run_isotope_correction_cached.clear()
        yield calls, outcome
        StreamlitAdapter._run_isotope_correction_cached.clear()

    def _call(self):
        from app.adapters.streamlit_adapter import StreamlitAdapter

        df = pd.DataFrame({"a": [1]})
        return StreamlitAdapter.run_isotope_correction(
            df, df, df, IsotopeCorrectionConfig(),
        )

    def test_a_failed_run_is_retried(self, workflow_calls):
        calls, _ = workflow_calls
        first, second = self._call(), self._call()
        assert len(calls) == 2
        assert first.error_message == second.error_message

    def test_a_successful_run_is_cached(self, workflow_calls):
        from app.workflows.isotope_tracing import IsotopeTracingResult

        calls, outcome = workflow_calls
        outcome["result"] = IsotopeTracingResult(success=True)
        assert self._call().success and self._call().success
        assert len(calls) == 1


GOLDEN_OUTPUT = (Path(__file__).parents[2] / "sample_datasets"
                 / "isotope_tracing" / "golden_output")
RUN_DIR = "2026-01-01_000000"  # IsoCorrectoR's timestamped result folder

# Which file each CorrectionResult table must come from. Spelled out here,
# not read from the service, so a swapped mapping there fails.
GOLDEN_FILES = {
    "corrected": "Corrected",
    "raw_data": "RawData",
    "corrected_fractions": "CorrectedFractions",
    "mean_enrichment": "MeanEnrichment",
    "residuals": "Residuals",
    "relative_residuals": "RelativeResiduals",
}


def _copy_golden(output_dir):
    shutil.copytree(GOLDEN_OUTPUT, Path(output_dir) / RUN_DIR)


def _golden(suffix):
    return pd.read_csv(
        GOLDEN_OUTPUT / f"IsoCorrectoR_result_{suffix}.csv", index_col=0,
    )


class TestParseOutputs:
    """IQ9: reading IsoCorrectoR's files back ran only where R is installed."""

    def test_each_table_comes_from_its_own_file(self, tmp_path):
        _copy_golden(tmp_path)
        result = svc.IsotopeCorrectionService._parse_outputs(str(tmp_path))
        for field_name, suffix in GOLDEN_FILES.items():
            pd.testing.assert_frame_equal(
                getattr(result, field_name), _golden(suffix),
                obj=field_name,
            )
        assert result.corrected_monoisotopic is None
        assert result.corrected_monoisotopic_fractions is None

    def test_log_and_its_warnings_are_kept(self, tmp_path):
        _copy_golden(tmp_path)
        result = svc.IsotopeCorrectionService._parse_outputs(str(tmp_path))
        log = (GOLDEN_OUTPUT / "IsoCorrectoR_result.log").read_text()
        assert result.log_text == log
        assert result.warnings == (
            svc.IsotopeCorrectionService.parse_log_warnings(log)
        )

    def test_missing_corrected_table_is_an_error(self, tmp_path):
        _copy_golden(tmp_path)
        (tmp_path / RUN_DIR / "IsoCorrectoR_result_Corrected.csv").unlink()
        with pytest.raises(RRuntimeError, match="expected output files"):
            svc.IsotopeCorrectionService._parse_outputs(str(tmp_path))

    def test_no_result_folder_is_an_error(self, tmp_path):
        with pytest.raises(RRuntimeError, match="no output directory"):
            svc.IsotopeCorrectionService._parse_outputs(str(tmp_path))


class TestRunOutcomes:
    """What run_correction makes of each way the R process can end."""

    def test_success_parses_what_r_wrote(self, monkeypatch):
        monkeypatch.setattr(svc, "require_isocorrector", lambda: "Rscript")
        seen = {}

        def _run(args, **kwargs):
            with open(args[2]) as fh:
                config = json.load(fh)
            seen["config"] = config
            seen["measurement"] = pd.read_csv(config["measurement_file"])
            _copy_golden(config["output_dir"])
            return subprocess.CompletedProcess(args, 0, "ISOCORRECTOR_OK\n", "")

        monkeypatch.setattr(svc.subprocess, "run", _run)
        measurement = pd.DataFrame({"ID": ["PC_34:1_C0"], "s1": [5.0]})
        result = svc.IsotopeCorrectionService.run_correction(
            measurement, measurement, measurement,
            IsotopeCorrectionConfig(ultra_high_res=True,
                                    calculation_threshold_uhr=6),
        )
        pd.testing.assert_frame_equal(result.corrected, _golden("Corrected"))
        assert seen["config"]["ultra_high_res"] is True
        assert seen["config"]["calculation_threshold_uhr"] == 6
        # Written without the index: IsoCorrectoR reads column 1 as the IDs.
        pd.testing.assert_frame_equal(seen["measurement"], measurement)

    def test_exit_0_without_the_ok_marker_is_a_failure(self, fake_r):
        fake_r(returncode=0, stdout="", stderr="it stopped early")
        with pytest.raises(RRuntimeError, match="it stopped early"):
            _run()

    def test_timeout(self, monkeypatch):
        monkeypatch.setattr(svc, "require_isocorrector", lambda: "Rscript")

        def _run_timeout(args, **kwargs):
            raise subprocess.TimeoutExpired(args, 600)

        monkeypatch.setattr(svc.subprocess, "run", _run_timeout)
        with pytest.raises(RRuntimeError, match="did not finish within"):
            _run()

    def test_rscript_cannot_start(self, monkeypatch):
        monkeypatch.setattr(svc, "require_isocorrector", lambda: "Rscript")

        def _run_oserror(args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(svc.subprocess, "run", _run_oserror)
        with pytest.raises(RRuntimeError, match="Failed to launch Rscript"):
            _run()


class TestRRuntimeDetection:
    @pytest.fixture
    def probe(self, monkeypatch):
        from app.services.isotope_correction import r_runtime

        def _set(rscript="/usr/bin/Rscript", returncode=0, raises=None):
            monkeypatch.setattr(r_runtime.shutil, "which",
                                lambda name: rscript)

            def _run(args, **kwargs):
                if raises is not None:
                    raise raises
                return subprocess.CompletedProcess(args, returncode)

            monkeypatch.setattr(r_runtime.subprocess, "run", _run)
            return r_runtime
        return _set

    def test_no_rscript(self, probe):
        r_runtime = probe(rscript=None)
        assert r_runtime.check_isocorrector_available() is False
        with pytest.raises(RRuntimeError, match="Rscript was not found"):
            r_runtime.require_isocorrector()

    def test_package_missing(self, probe):
        r_runtime = probe(returncode=1)
        assert r_runtime.check_isocorrector_available() is False
        with pytest.raises(RRuntimeError, match="could not be loaded"):
            r_runtime.require_isocorrector()

    @pytest.mark.parametrize("error", [
        subprocess.TimeoutExpired("Rscript", 30), OSError("exec format"),
    ])
    def test_probe_errors_mean_unavailable(self, probe, error):
        assert probe(raises=error).check_isocorrector_available() is False

    def test_available(self, probe):
        r_runtime = probe()
        assert r_runtime.check_isocorrector_available() is True
        assert r_runtime.require_isocorrector() == "/usr/bin/Rscript"


class TestWorkflow:
    def test_r_failure_becomes_an_error_message(self, monkeypatch):
        from app.workflows import isotope_tracing as wf

        monkeypatch.setattr(wf, "validate_inputs", lambda *a, **k: [])

        def _fail(*args):
            raise RRuntimeError("R crashed")

        monkeypatch.setattr(wf.IsotopeCorrectionService, "run_correction",
                            staticmethod(_fail))
        df = pd.DataFrame({"a": [1]})
        result = wf.IsotopeTracingWorkflow.run(
            df, df, df, IsotopeCorrectionConfig(),
        )
        assert not result.success
        assert result.error_message == "R crashed"
        assert result.validation_errors == []

    def test_invalid_inputs_never_reach_r(self, monkeypatch):
        from app.workflows import isotope_tracing as wf

        def _never(*args):
            raise AssertionError("R must not run on invalid inputs")

        monkeypatch.setattr(wf.IsotopeCorrectionService, "run_correction",
                            staticmethod(_never))
        empty = pd.DataFrame()
        result = wf.IsotopeTracingWorkflow.run(
            empty, empty, empty, IsotopeCorrectionConfig(),
        )
        assert not result.success
        assert "Measurement file is empty." in result.validation_errors
