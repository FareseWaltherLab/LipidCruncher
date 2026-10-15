"""IsoCorrectoR runs that complete "with warnings" (IQ1).

Regression: the wrapper accepted only success == "TRUE". For a blank
intensity (which the in-app help tells users to leave) or a missing
isotopologue row IsoCorrectoR returns "WARNINGS" after writing every output,
and the whole run was reported as a failure.
"""
from pathlib import Path

import pandas as pd
import pytest

from app.models.isotope_tracing import IsotopeCorrectionConfig
from app.services.isotope_correction import check_isocorrector_available
from app.services.isotope_correction.service import IsotopeCorrectionService
from app.workflows.isotope_tracing import IsotopeTracingWorkflow

DATA = Path(__file__).parents[2] / "sample_datasets" / "isotope_tracing"

_LOG = """ISOCORRECTOR LOG-FILE - 2026-10-15 10:01:17
IsoCorrectoR-version: 1.30.0

WARNINGS HAVE OCCURRED, THE CORRECTION MAY BE FAULTY. WARNING(s):

In measurement data file: Measurement data of molecule BMP_18:1_18:1 in sample S1 contains NA values. The correction performed may be less acurate.
Be especially careful when considering fraction and mean enrichment values from samples with missing values.
In measurement data file: The expected measurement ID 'PC_34:1_C6' could not be found in the measurement data file.
Correction will be performed, however the results may be less accurate.

FILES AND DIRECTORIES

Measurement File: /tmp/x.csv
"""


class TestParseLogWarnings:
    def test_each_warning_becomes_one_string(self):
        warnings = IsotopeCorrectionService.parse_log_warnings(_LOG)
        assert len(warnings) == 2
        assert warnings[0].startswith("In measurement data file: Measurement")
        assert "Be especially careful" in warnings[0]
        assert "'PC_34:1_C6' could not be found" in warnings[1]

    def test_clean_log_has_none(self):
        assert IsotopeCorrectionService.parse_log_warnings(
            "ISOCORRECTOR LOG-FILE\n\nFILES AND DIRECTORIES\n"
        ) == []


@pytest.mark.skipif(
    not check_isocorrector_available(),
    reason="Rscript / IsoCorrectoR not available in this environment",
)
class TestRunsWithWarningsSucceed:
    @pytest.fixture
    def inputs(self):
        return [pd.read_csv(DATA / f) for f in (
            "MeasurementFile.csv", "MoleculeFile.csv", "ElementFile.csv",
        )]

    def _run(self, measurement, molecule, element):
        return IsotopeTracingWorkflow.run(
            measurement, molecule, element,
            IsotopeCorrectionConfig(ultra_high_res=True),
        )

    def test_blank_intensity_is_a_warning_not_a_failure(self, inputs):
        measurement, molecule, element = inputs
        measurement.iloc[3, 2] = None
        result = self._run(measurement, molecule, element)
        assert result.success, result.error_message
        assert any("NA values" in w for w in result.correction.warnings)

    def test_missing_isotopologue_row_is_a_warning(self, inputs):
        measurement, molecule, element = inputs
        measurement = measurement[measurement.iloc[:, 0] != "PC_34:1_C6"]
        result = self._run(measurement, molecule, element)
        assert result.success, result.error_message
        assert any("PC_34:1_C6" in w for w in result.correction.warnings)

    def test_clean_run_has_no_warnings(self, inputs):
        result = self._run(*inputs)
        assert result.success
        assert result.correction.warnings == []


@pytest.mark.skipif(
    not check_isocorrector_available(),
    reason="Rscript / IsoCorrectoR not available in this environment",
)
def test_rejected_inputs_say_why_in_one_line():
    """Regression (IQ2): the reason was buried in ~1.7 KB of stderr."""
    measurement, molecule, element = [pd.read_csv(DATA / f) for f in (
        "MeasurementFile.csv", "MoleculeFile.csv", "ElementFile.csv",
    )]
    result = IsotopeTracingWorkflow.run(
        measurement, molecule, element[element.Element != "N"],
        IsotopeCorrectionConfig(ultra_high_res=True),
    )
    assert not result.success
    assert result.error_message.startswith(
        "IsoCorrectoR could not correct these files:"
    )
    assert "element file" in result.error_message
    assert len(result.error_message) < 300
