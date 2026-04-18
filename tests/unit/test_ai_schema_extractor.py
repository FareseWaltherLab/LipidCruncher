"""
Unit tests for the AI schema extractor service.
"""
import pandas as pd
import pytest

from app.services.ai_chat.schema_extractor import SchemaExtractorService


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1", "PE 36:2", "SM 42:1"],
        "ClassKey": ["PC", "PE", "SM"],
        "intensity[s1]": [100.0, 200.0, 300.0],
        "intensity[s2]": [110.0, 190.0, 310.0],
    })


class TestExtractSchema:

    def test_includes_shape(self, sample_df):
        schema = SchemaExtractorService.extract_schema(sample_df)
        assert "3 rows x 4 columns" in schema

    def test_includes_column_names(self, sample_df):
        schema = SchemaExtractorService.extract_schema(sample_df)
        assert "LipidMolec" in schema
        assert "ClassKey" in schema
        assert "intensity[s1]" in schema
        assert "intensity[s2]" in schema

    def test_includes_dtypes(self, sample_df):
        schema = SchemaExtractorService.extract_schema(sample_df)
        assert "object" in schema
        assert "float64" in schema

    def test_includes_sample_rows(self, sample_df):
        schema = SchemaExtractorService.extract_schema(sample_df)
        assert "PC 34:1" in schema
        assert "Sample rows" in schema

    def test_non_null_counts(self, sample_df):
        schema = SchemaExtractorService.extract_schema(sample_df)
        assert "3/3 non-null" in schema

    def test_with_nulls(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        schema = SchemaExtractorService.extract_schema(df)
        assert "2/3 non-null" in schema

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        schema = SchemaExtractorService.extract_schema(df)
        assert "0 rows" in schema

    def test_limits_sample_rows(self):
        df = pd.DataFrame({"x": range(100)})
        schema = SchemaExtractorService.extract_schema(df)
        assert "Sample rows (3)" in schema


class TestExtractGenericFormatSpec:

    def test_mentions_column_structure(self):
        spec = SchemaExtractorService.extract_generic_format_spec()
        assert "Column 1" in spec
        assert "Lipid names" in spec or "lipid names" in spec.lower()
        assert "Columns 2+" in spec
        assert "Sample intensities" in spec or "sample intensities" in spec.lower()

    def test_mentions_no_extra_columns(self):
        spec = SchemaExtractorService.extract_generic_format_spec()
        assert "NO extra columns" in spec or "no extra columns" in spec.lower()

    def test_mentions_lipid_maps(self):
        spec = SchemaExtractorService.extract_generic_format_spec()
        assert "LIPID MAPS" in spec
