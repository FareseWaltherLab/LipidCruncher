"""
Unit tests for AI Studio validation logic.

Tests format diagnostic (transform tab) and Module 1 output
validation (viz tab) — no AI calls needed.
"""
import pandas as pd
import pytest

from app.services.format_detection import DataFormat
from app.workflows.ai_chat import (
    AIChatWorkflow,
    FormatDiagnosticResult,
    Module1ValidationResult,
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


@pytest.fixture
def module1_intensity_df():
    """DataFrame that has been through Module 1 (intensity columns)."""
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1", "PE 36:2"],
        "ClassKey": ["PC", "PE"],
        "intensity[s1]": [100.0, 200.0],
        "intensity[s2]": [110.0, 190.0],
    })


@pytest.fixture
def module1_concentration_df():
    """DataFrame that has been through Module 1 (concentration columns)."""
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1", "PE 36:2"],
        "ClassKey": ["PC", "PE"],
        "concentration[s1]": [10.0, 20.0],
        "concentration[s2]": [11.0, 19.0],
    })


@pytest.fixture
def raw_df():
    """Raw DataFrame that has NOT been through Module 1."""
    return pd.DataFrame({
        "lipid_name": ["PC(34:1)", "PE(36:2)"],
        "sample1": [100, 200],
        "sample2": [110, 190],
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


# =========================================================================
# Module 1 Validation Tests
# =========================================================================

class TestValidateModule1Output:

    def test_valid_intensity_data(self, module1_intensity_df):
        result = AIChatWorkflow.validate_module1_output(module1_intensity_df)
        assert result.is_valid is True
        assert result.has_lipid_molec is True
        assert result.has_class_key is True
        assert len(result.intensity_columns) == 2

    def test_valid_concentration_data(self, module1_concentration_df):
        result = AIChatWorkflow.validate_module1_output(
            module1_concentration_df
        )
        assert result.is_valid is True
        assert len(result.concentration_columns) == 2

    def test_raw_data_invalid(self, raw_df):
        result = AIChatWorkflow.validate_module1_output(raw_df)
        assert result.is_valid is False
        assert result.has_lipid_molec is False
        assert result.has_class_key is False

    def test_missing_classkey(self):
        df = pd.DataFrame({
            "LipidMolec": ["PC 34:1"],
            "intensity[s1]": [100.0],
        })
        result = AIChatWorkflow.validate_module1_output(df)
        assert result.is_valid is False
        assert result.has_lipid_molec is True
        assert result.has_class_key is False

    def test_missing_lipidmolec(self):
        df = pd.DataFrame({
            "ClassKey": ["PC"],
            "intensity[s1]": [100.0],
        })
        result = AIChatWorkflow.validate_module1_output(df)
        assert result.is_valid is False
        assert result.has_lipid_molec is False
        assert result.has_class_key is True

    def test_missing_sample_columns(self):
        df = pd.DataFrame({
            "LipidMolec": ["PC 34:1"],
            "ClassKey": ["PC"],
            "some_value": [100.0],
        })
        result = AIChatWorkflow.validate_module1_output(df)
        assert result.is_valid is False
        assert len(result.intensity_columns) == 0
        assert len(result.concentration_columns) == 0

    def test_valid_message_includes_counts(self, module1_intensity_df):
        result = AIChatWorkflow.validate_module1_output(module1_intensity_df)
        assert "2" in result.message  # 2 lipid species or 2 intensity columns
        assert "intensity" in result.message

    def test_invalid_message_lists_missing(self, raw_df):
        result = AIChatWorkflow.validate_module1_output(raw_df)
        assert "LipidMolec" in result.message
        assert "ClassKey" in result.message
        assert "Module 1" in result.message
