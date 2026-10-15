"""
Golden-dataset parity test for the isotope tracing pipeline.

Acceptance gate (Phase 1): our pipeline must reproduce the superuser's
IsoCorrectoR output on the scientifically meaningful tables: the same rows
and columns, the same non-finite cells, and values within a per-table
tolerance set from what is actually observed (IsoCorrectoR 1.30.0 against
the golden 1.28.0 output): RawData and MeanEnrichment identical,
CorrectedFractions within 1e-15, Corrected and Residuals within 3e-5 on
intensities up to 1e11. Not bit-for-bit: floating-point noise near zero
differs between versions.

Gated on R availability — skipped where Rscript/IsoCorrectoR isn't installed,
so CI without R still passes. Run it where R is installed.

RelativeResiduals values are not compared: they are numerically degenerate
(residual / a corrected value of ~0 → ±Inf or huge values) and those cells
differ between IsoCorrectoR versions. Its rows and columns are checked.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.models.isotope_tracing import IsotopeCorrectionConfig
from app.services.isotope_correction import check_isocorrector_available
from app.workflows.isotope_tracing import IsotopeTracingWorkflow

pytestmark = pytest.mark.skipif(
    not check_isocorrector_available(),
    reason="Rscript / IsoCorrectoR not available in this environment",
)

GOLDEN_DIR = (
    Path(__file__).parent.parent.parent
    / "sample_datasets"
    / "isotope_tracing"
)
GOLDEN_OUTPUT_DIR = GOLDEN_DIR / "golden_output"

# Settings recorded in the golden run's IsoCorrectoR_result.log.
GOLDEN_CONFIG = IsotopeCorrectionConfig(
    correct_tracer_impurity=True,
    correct_tracer_element_core=True,
    calculate_mean_enrichment=True,
    ultra_high_res=True,
    correct_also_monoisotopic=False,
    calculation_threshold_uhr=8,
)

# (CorrectionResult field, golden output-file suffix, absolute tolerance),
# each a few times the difference observed (see the module docstring).
PARITY_TABLES = [
    ("corrected", "Corrected", 1e-4),
    ("corrected_fractions", "CorrectedFractions", 1e-12),
    ("mean_enrichment", "MeanEnrichment", 1e-12),
    ("residuals", "Residuals", 1e-4),
    ("raw_data", "RawData", 0.0),
]


def _golden(suffix: str) -> pd.DataFrame:
    return pd.read_csv(
        GOLDEN_OUTPUT_DIR / f"IsoCorrectoR_result_{suffix}.csv", index_col=0
    )


def _assert_same_labels(actual: pd.DataFrame, golden: pd.DataFrame, name: str):
    pd.testing.assert_index_equal(actual.index, golden.index, obj=f"{name} rows")
    pd.testing.assert_index_equal(
        actual.columns, golden.columns, obj=f"{name} columns",
    )


def _assert_parity(actual: pd.DataFrame, golden: pd.DataFrame, name: str,
                   atol: float):
    """Same labels and non-finite cells; finite values within ``atol``."""
    _assert_same_labels(actual, golden, name)
    a = actual.values.astype(float)
    g = golden.values.astype(float)
    assert np.array_equal(~np.isfinite(a), ~np.isfinite(g)), (
        f"{name}: non-finite cell positions differ"
    )
    finite = np.isfinite(g)
    np.testing.assert_allclose(
        a[finite], g[finite], rtol=1e-12, atol=atol, err_msg=name,
    )


@pytest.fixture(scope="module")
def correction_result():
    measurement = pd.read_csv(GOLDEN_DIR / "MeasurementFile.csv")
    molecule = pd.read_csv(GOLDEN_DIR / "MoleculeFile.csv")
    element = pd.read_csv(GOLDEN_DIR / "ElementFile.csv")

    result = IsotopeTracingWorkflow.run(measurement, molecule, element, GOLDEN_CONFIG)
    assert result.success, (
        f"workflow failed: validation={result.validation_errors} "
        f"error={result.error_message}"
    )
    return result.correction


@pytest.mark.parametrize("field_name,suffix,atol", PARITY_TABLES)
def test_table_matches_golden(correction_result, field_name, suffix, atol):
    actual = getattr(correction_result, field_name)
    assert actual is not None, f"{field_name} table was not produced"
    _assert_parity(actual, _golden(suffix), field_name, atol)


def test_relative_residuals_have_the_golden_labels(correction_result):
    actual = correction_result.relative_residuals
    assert actual is not None
    _assert_same_labels(actual, _golden("RelativeResiduals"), "relative_residuals")


def test_run_log_captured(correction_result):
    assert "ISOCORRECTOR" in correction_result.log_text.upper()
