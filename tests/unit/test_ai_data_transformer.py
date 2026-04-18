"""
Unit tests for the AI data transformer service.

Tests validation logic directly and transformation execution with
pre-written code (no AI calls needed for these tests).
"""
import numpy as np
import pandas as pd
import pytest

from app.services.ai_chat.data_transformer import DataTransformerService, TransformationResult
from app.models.chat import SandboxConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source_df():
    """A non-standard DataFrame that needs transformation."""
    return pd.DataFrame({
        "lipid_name": ["PC(34:1)", "PE(36:2)", "SM(42:1)"],
        "lipid_class": ["PC", "PE", "SM"],
        "area_sample1": [100.0, 200.0, 300.0],
        "area_sample2": [110.0, 190.0, 310.0],
    })


@pytest.fixture
def valid_generic_df():
    """A properly formatted Generic DataFrame (col1=lipid names, cols2+=numeric)."""
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1", "PE 36:2", "SM 42:1"],
        "Sample1": [100.0, 200.0, 300.0],
        "Sample2": [110.0, 190.0, 310.0],
    })


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

class TestValidateGenericSchema:

    def test_valid_schema_passes(self, valid_generic_df):
        is_valid, errors = DataTransformerService.validate_generic_schema(valid_generic_df)
        assert is_valid is True
        assert errors == []

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Lipid", "Sample1"])
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_single_column_rejected(self):
        df = pd.DataFrame({"Lipid": ["PC 34:1", "PE 36:2"]})
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("at least 2 columns" in e.lower() for e in errors)

    def test_first_column_all_nan(self):
        df = pd.DataFrame({
            "Lipid": [None, None],
            "Sample1": [100.0, 200.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("empty" in e.lower() and "NaN" in e for e in errors)

    def test_non_numeric_column_after_first(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1"],
            "ClassKey": ["PC"],
            "Sample1": [100.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("non-numeric" in e.lower() for e in errors)
        assert any("ClassKey" in e for e in errors)

    def test_multiple_non_numeric_columns_reported(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1"],
            "ClassKey": ["PC"],
            "Score": ["high"],
            "Sample1": [100.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("ClassKey" in e for e in errors)
        assert any("Score" in e for e in errors)

    def test_multiple_numeric_columns_valid(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1"],
            "S1": [100.0],
            "S2": [110.0],
            "S3": [120.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is True

    def test_any_column_names_accepted(self):
        """Column names are arbitrary — only data types matter."""
        df = pd.DataFrame({
            "my_lipids": ["PC 34:1", "PE 36:2"],
            "KO-1": [100.0, 200.0],
            "WT-2": [110.0, 190.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is True


# ---------------------------------------------------------------------------
# Execute Transformation (with real sandbox, no AI)
# ---------------------------------------------------------------------------

class TestExecuteTransformation:

    def test_successful_transformation(self, source_df):
        code = (
            "result_df = pd.DataFrame()\n"
            "result_df['LipidMolec'] = df['lipid_name'].str.replace("
            "r'^([A-Za-z]+)\\((.+)\\)$', r'\\1 \\2', regex=True)\n"
            "result_df['Sample1'] = pd.to_numeric(df['area_sample1'])\n"
            "result_df['Sample2'] = pd.to_numeric(df['area_sample2'])\n"
        )
        result = DataTransformerService.execute_transformation(code, source_df)
        assert result.success is True
        assert result.result_df is not None
        assert "LipidMolec" in result.result_df.columns
        assert "Sample1" in result.result_df.columns
        assert result.result_df["LipidMolec"].iloc[0] == "PC 34:1"

    def test_code_with_syntax_error(self, source_df):
        result = DataTransformerService.execute_transformation("def f(", source_df)
        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_code_that_produces_no_result_df(self, source_df):
        code = "x = df.copy()  # forgot to name it result_df"
        result = DataTransformerService.execute_transformation(code, source_df)
        assert result.success is False
        assert any("result_df" in e for e in result.validation_errors)

    def test_code_producing_invalid_schema(self, source_df):
        code = "result_df = pd.DataFrame({'wrong_col': [1, 2, 3]})"
        result = DataTransformerService.execute_transformation(code, source_df)
        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_code_with_runtime_error(self, source_df):
        code = "result_df = df['nonexistent_column']"
        result = DataTransformerService.execute_transformation(code, source_df)
        assert result.success is False

    def test_forbidden_code_blocked(self, source_df):
        code = "import os\nresult_df = df"
        result = DataTransformerService.execute_transformation(code, source_df)
        assert result.success is False
