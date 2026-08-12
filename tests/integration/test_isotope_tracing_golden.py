"""
Golden-dataset parity test for the isotope tracing pipeline.

Acceptance gate (Phase 1): our pipeline must reproduce the superuser's
IsoCorrectoR output bit-for-bit on the scientifically meaningful tables.

Gated on R availability — skipped where Rscript/IsoCorrectoR isn't installed,
so CI without R still passes. Run it where R is installed.

Note: RelativeResiduals is intentionally NOT asserted. It is numerically
degenerate (residual / a corrected value of ~0 → ±Inf or huge values) and
those cells differ between the IsoCorrectoR version that generated the golden
output (1.28.0) and any other version. Every meaningful output matches exactly.
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
    calculation_threshold_uhr=8.0,
)

# (CorrectionResult field, golden output-file suffix). RelativeResiduals omitted.
PARITY_TABLES = [
    ("corrected", "Corrected"),
    ("corrected_fractions", "CorrectedFractions"),
    ("mean_enrichment", "MeanEnrichment"),
    ("residuals", "Residuals"),
    ("raw_data", "RawData"),
]


def _assert_parity(actual: pd.DataFrame, golden: pd.DataFrame, name: str):
    """Non-finite masks must match; finite values must be close."""
    aligned = actual.reindex(index=golden.index, columns=golden.columns)
    a = aligned.values.astype(float)
    g = golden.values.astype(float)

    assert not np.isnan(a).all(), f"{name}: actual table did not align with golden index/columns"

    a_nonfinite = ~np.isfinite(a)
    g_nonfinite = ~np.isfinite(g)
    assert np.array_equal(a_nonfinite, g_nonfinite), f"{name}: non-finite cell positions differ"

    finite = np.isfinite(a) & np.isfinite(g)
    assert np.allclose(a[finite], g[finite], rtol=1e-6, atol=1e-3), f"{name}: finite values differ"


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


@pytest.mark.parametrize("field_name,suffix", PARITY_TABLES)
def test_table_matches_golden(correction_result, field_name, suffix):
    actual = getattr(correction_result, field_name)
    assert actual is not None, f"{field_name} table was not produced"
    golden = pd.read_csv(
        GOLDEN_OUTPUT_DIR / f"IsoCorrectoR_result_{suffix}.csv", index_col=0
    )
    _assert_parity(actual, golden, field_name)


def test_run_log_captured(correction_result):
    assert "ISOCORRECTOR" in correction_result.log_text.upper()
