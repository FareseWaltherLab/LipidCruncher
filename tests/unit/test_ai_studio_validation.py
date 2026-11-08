"""
Unit tests for AI Studio validation logic.

Tests the format diagnostic (transform tab) — no AI calls needed.
"""
import pandas as pd
import pytest

from app.services.format_detection import DataFormat
from app.workflows.ai_chat import (
    AIChatWorkflow,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def lipidsearch_df():
    """DataFrame in LipidSearch 5.0 format."""
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1"],
        "ClassKey": ["PC"],
        "CalcMass": [758.5],
        "BaseRt": [10.1],
        "TotalGrade": ["A"],
        "TotalSmpIDRate(%)": [100],
        "FAKey": ["16:0_18:1"],
        "MeanArea[s1]": [1000.0],
    })


@pytest.fixture
def generic_df():
    """DataFrame in Generic format."""
    return pd.DataFrame({
        "Lipid": ["PC 34:1", "PE 36:2"],
        "Sample1": [100, 200],
        "Sample2": [110, 190],
    })


@pytest.fixture
def unknown_df():
    """DataFrame not matching any supported format."""
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "compound_name": ["PC(34:1)", "PE(36:2)", "SM(42:1)"],
        "formula": ["C42H82", "C41H78", "C25H51"],
        "retention_time": [10.1, 11.2, 9.8],
        "area_1": [1000, 2000, 3000],
        "area_2": [1100, 1900, 3100],
    })


# =========================================================================
# Format Diagnostic Tests
# =========================================================================

class TestDiagnoseFormat:

    def test_lipidsearch_detected_as_supported(self, lipidsearch_df):
        result = AIChatWorkflow.diagnose_format(lipidsearch_df)
        assert result.is_supported is True
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert "LipidSearch" in result.message

    def test_generic_detected_as_supported(self, generic_df):
        result = AIChatWorkflow.diagnose_format(generic_df)
        assert result.is_supported is True
        assert result.detected_format == DataFormat.GENERIC

    def test_unknown_detected_as_unsupported(self, unknown_df):
        result = AIChatWorkflow.diagnose_format(unknown_df)
        assert result.is_supported is False
        assert result.detected_format == DataFormat.UNKNOWN

    def test_unknown_identifies_lipid_column(self, unknown_df):
        result = AIChatWorkflow.diagnose_format(unknown_df)
        assert result.lipid_name_column == "compound_name"

    def test_unknown_identifies_intensity_columns(self, unknown_df):
        result = AIChatWorkflow.diagnose_format(unknown_df)
        assert "area_1" in result.intensity_columns
        assert "area_2" in result.intensity_columns

    def test_unknown_identifies_metadata_columns(self, unknown_df):
        result = AIChatWorkflow.diagnose_format(unknown_df)
        assert "retention_time" in result.metadata_columns
        assert "formula" in result.metadata_columns

    def test_unknown_message_mentions_columns(self, unknown_df):
        result = AIChatWorkflow.diagnose_format(unknown_df)
        assert "compound_name" in result.message
        assert "area_1" in result.message

    def test_supported_format_has_no_column_analysis(self, lipidsearch_df):
        result = AIChatWorkflow.diagnose_format(lipidsearch_df)
        assert result.lipid_name_column is None
        assert result.intensity_columns == []
        assert result.metadata_columns == []

    def test_empty_dataframe(self):
        result = AIChatWorkflow.diagnose_format(pd.DataFrame())
        assert result.is_supported is False
        assert result.detected_format == DataFormat.UNKNOWN


# =========================================================================
# Format Diagnostic with Session Context
# =========================================================================

class TestDiagnosticInSession:

    def test_unknown_format_adds_preanalysis_to_schema(self, unknown_df):
        diagnostic = AIChatWorkflow.diagnose_format(unknown_df)
        session = AIChatWorkflow.start_transform_session(
            unknown_df, diagnostic
        )
        assert "Pre-Analysis" in session.schema_context
        assert "compound_name" in session.schema_context

    def test_supported_format_no_preanalysis(self, generic_df):
        diagnostic = AIChatWorkflow.diagnose_format(generic_df)
        session = AIChatWorkflow.start_transform_session(
            generic_df, diagnostic
        )
        assert "Pre-Analysis" not in session.schema_context

    def test_no_diagnostic_no_preanalysis(self, generic_df):
        session = AIChatWorkflow.start_transform_session(generic_df)
        assert "Pre-Analysis" not in session.schema_context
