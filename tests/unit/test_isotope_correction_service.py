"""IsotopeCorrectionService without R: the subprocess is replaced, so these
run in CI (which has no R) and pin how R's outcome is reported."""
import subprocess

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
