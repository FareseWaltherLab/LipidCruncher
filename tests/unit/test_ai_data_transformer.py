"""
Unit tests for the AI data transformer service.

Tests validation logic directly and transformation execution with
pre-written code (no AI calls needed for these tests).
"""
import pandas as pd
import pytest

from app.services.ai_chat.data_transformer import DataTransformerService


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
            "Comment": ["peak eluted late, check shape"],
            "Sample1": [100.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("non-numeric" in e.lower() for e in errors)
        assert any("Comment" in e for e in errors)

    # Regression (AQ8): these disagreed with how the Generic reader
    # (DataStandardizationService) reads the uploaded file.

    def test_classkey_second_column_is_accepted(self):
        """The reader takes an optional ClassKey column 2; the validator
        rejected it, on exactly the raw-import case the tab is for."""
        df = pd.DataFrame({
            "LipidMolec": ["PC 34:1", "PE 36:2"],
            "ClassKey": ["PC", "PE"],
            "Sample1": [100.0, 200.0],
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid, errors
        assert DataTransformerService.detect_metadata_columns(df) == []

    def test_numeric_first_column_is_rejected(self):
        df = pd.DataFrame({"Id": [1, 2], "Sample1": [100.0, 200.0]})
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("lipid names" in e for e in errors)

    def test_empty_sample_column_is_rejected(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1", "PE 36:2"],
            "Sample1": [100.0, 200.0],
            "Sample2": [float("nan")] * 2,
        })
        is_valid, errors = DataTransformerService.validate_generic_schema(df)
        assert is_valid is False
        assert any("Sample2" in e and "no values" in e for e in errors)

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


# ---------------------------------------------------------------------------
# Metadata Detection
# ---------------------------------------------------------------------------

class TestDetectMetadataColumns:

    def test_clean_generic_no_metadata(self, valid_generic_df):
        result = DataTransformerService.detect_metadata_columns(valid_generic_df)
        assert result == []

    def test_detects_score_column(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1"],
            "Score": [95.0],
            "Sample1": [100.0],
        })
        result = DataTransformerService.detect_metadata_columns(df)
        assert "Score" in result

    def test_detects_mass_and_rt(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1"],
            "CalcMass": [758.5],
            "RetentionTime": [10.1],
            "Sample1": [100.0],
        })
        result = DataTransformerService.detect_metadata_columns(df)
        assert "CalcMass" in result
        assert "RetentionTime" in result

    def test_ignores_first_column(self):
        """First column (lipid names) should never be flagged."""
        df = pd.DataFrame({
            "class_name": ["PC 34:1"],
            "Sample1": [100.0],
        })
        result = DataTransformerService.detect_metadata_columns(df)
        assert "class_name" not in result

    def test_detects_case_insensitive(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1"],
            "TOTAL_SCORE": [95.0],
            "sample1": [100.0],
        })
        result = DataTransformerService.detect_metadata_columns(df)
        assert "TOTAL_SCORE" in result


# ---------------------------------------------------------------------------
# Validation Summary
# ---------------------------------------------------------------------------

class TestBuildValidationSummary:

    def test_clean_generic_summary(self, valid_generic_df):
        summary = DataTransformerService.build_validation_summary(valid_generic_df)
        assert "3 lipid species" in summary
        assert "2 sample columns" in summary
        assert "no metadata detected" in summary

    def test_summary_flags_metadata(self):
        df = pd.DataFrame({
            "Lipid": ["PC 34:1", "PE 36:2"],
            "Score": [95, 88],
            "Sample1": [100.0, 200.0],
        })
        summary = DataTransformerService.build_validation_summary(df)
        assert "2 lipid species" in summary
        assert "metadata" in summary.lower()
        assert "Score" in summary

    def test_summary_warns_non_lipid_first_column(self):
        df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Sample1": [100.0, 200.0, 300.0],
        })
        summary = DataTransformerService.build_validation_summary(df)
        assert "may not be lipid names" in summary


# The code the AI wrote for "keep species with retention time above 5 and
# quality score above 70, then transform to generic" (2026-10-20). The names
# get a fresh 0..n index (.values) while the sample columns keep the filtered
# frame's index, so pandas lines them up by label: values land on the wrong
# lipids or vanish.
_MISALIGNED_FILTER = """
def transform(df):
    filtered = df[(df['RetentionTime'] > 5) & (df['QualityScore'] > 70)].copy()
    sample_cols = ['KO-1', 'KO-2', 'KO-3', 'KOWT-1', 'KOWT-2', 'KOWT-3',
                   'mut1-1', 'mut1-2', 'mut1-3', 'mut2-1', 'mut2-2', 'mut2-3',
                   'WT-1', 'WT-2', 'WT-3']
    result = pd.DataFrame()
    result['feature'] = filtered['feature'].values
    for col in sample_cols:
        result[col] = pd.to_numeric(filtered[col], errors='coerce')
    return result.reset_index(drop=True)

result_df = transform(df)
"""

_SAMPLES = ['KO-1', 'KO-2', 'KO-3', 'KOWT-1', 'KOWT-2', 'KOWT-3',
            'mut1-1', 'mut1-2', 'mut1-3', 'mut2-1', 'mut2-2', 'mut2-3',
            'WT-1', 'WT-2', 'WT-3']


@pytest.fixture
def studio_df():
    from pathlib import Path
    path = Path(__file__).parents[1] / 'fixtures' / 'ai_studio_test_dataset.csv'
    return pd.read_csv(path)


class TestValuesStayWithTheirLipid:
    """Regression: a transform that filtered rows and then combined columns
    by index put each lipid's values on another lipid (and blanked others),
    and passed every schema check."""

    def test_misaligned_filter_is_refused(self, studio_df):
        result = DataTransformerService.execute_transformation(_MISALIGNED_FILTER, studio_df)
        assert result.success is False
        message = " ".join(result.validation_errors)
        assert "'CE 19:1;0' has the values of 'CE 16:1;0'" in message
        assert "reset_index(drop=True)" in message

    def test_the_corrected_filter_passes(self, studio_df):
        code = _MISALIGNED_FILTER.replace(
            ".copy()", ".copy().reset_index(drop=True)", 1,
        )
        result = DataTransformerService.execute_transformation(code, studio_df)
        assert result.success is True, result.validation_errors
        kept = studio_df[(studio_df['RetentionTime'] > 5) & (studio_df['QualityScore'] > 70)]
        assert len(result.result_df) == len(kept)

    @pytest.mark.parametrize('values', [
        "df[_SAMPLES] / 2",                     # scaled: every value changes
        "df[_SAMPLES].fillna(0)",               # gaps filled
    ])
    def test_changed_values_are_not_mistaken_for_moved_ones(self, studio_df, values):
        code = (
            f"_SAMPLES = {_SAMPLES!r}\n"
            f"result_df = pd.concat([df[['feature']], {values}], axis=1)\n"
        )
        result = DataTransformerService.execute_transformation(code, studio_df)
        assert result.success is True, result.validation_errors

    def test_renamed_lipids_are_not_checked(self, studio_df):
        code = (
            f"_SAMPLES = {_SAMPLES!r}\n"
            "result_df = df[['feature'] + _SAMPLES].copy()\n"
            "result_df['feature'] = 'X ' + result_df['feature']\n"
            "result_df = result_df.iloc[::-1]\n"
        )
        result = DataTransformerService.execute_transformation(code, studio_df)
        assert result.success is True, result.validation_errors


class TestMetadataKeptAsSamples:
    """A numeric metadata column left in result_df passed the schema check
    and was then analysed as a sample; only the summary text hinted at it."""

    def test_retention_time_kept_as_a_sample_is_refused(self, studio_df):
        code = (
            f"_SAMPLES = {_SAMPLES!r}\n"
            "result_df = df[['feature', 'RetentionTime'] + _SAMPLES]\n"
        )
        result = DataTransformerService.execute_transformation(code, studio_df)
        assert result.success is False
        message = " ".join(result.validation_errors)
        assert "'RetentionTime'" in message
        assert "drop these" in message

    def test_the_same_transform_without_it_passes(self, studio_df):
        code = (
            f"_SAMPLES = {_SAMPLES!r}\n"
            "result_df = df[['feature'] + _SAMPLES]\n"
        )
        result = DataTransformerService.execute_transformation(code, studio_df)
        assert result.success is True, result.validation_errors
