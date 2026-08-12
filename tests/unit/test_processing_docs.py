"""Unit tests for the in-app "About Data Standardization and Filtering" text.

These docs describe what the cleaners actually do, so they rot silently when
the pipeline changes — nothing fails, the panel just starts lying. The tests
here are deliberately narrow: they guard the structural contract (every format
has docs) and the handful of load-bearing claims that have drifted before,
rather than pinning prose that is meant to be edited freely.
"""
import pytest

from app.constants import (
    FORMAT_GENERIC, FORMAT_LIPIDSEARCH, FORMAT_METABOLOMICS_WORKBENCH,
    FORMAT_MSDIAL, FORMAT_OPTIONS,
)
from app.ui.content import get_processing_docs
from app.ui.content.processing_docs import PROCESSING_DOCS


class TestEveryFormatIsDocumented:
    """Adding a format without docs should fail here, not surface as an empty
    expander in the app."""

    @pytest.mark.parametrize("data_format", FORMAT_OPTIONS)
    def test_format_has_non_empty_docs(self, data_format):
        assert data_format in PROCESSING_DOCS
        assert PROCESSING_DOCS[data_format].strip()

    def test_unknown_format_falls_back_to_generic(self):
        assert get_processing_docs("Nonexistent Format") == (
            PROCESSING_DOCS[FORMAT_GENERIC]
        )

    @pytest.mark.parametrize("data_format", FORMAT_OPTIONS)
    def test_every_doc_mentions_zero_filtering(self, data_format):
        """Zero filtering runs for every format and is user-configurable, so
        each pipeline table has to end with it."""
        assert "Zero Filtering" in get_processing_docs(data_format)


class TestLipidSearchDualPolarityIsDocumented:
    """Regression: the docs described only the flat `MeanArea` layout, so users
    of condition-grouped 5.2 exports had no explanation of why an Alignment
    Setting file was being demanded — the one step they cannot skip."""

    @pytest.fixture
    def docs(self):
        return get_processing_docs(FORMAT_LIPIDSEARCH)

    def test_both_intensity_layouts_are_described(self, docs):
        assert "MeanArea" in docs
        assert "OriginalArea" in docs

    def test_alignment_file_requirement_is_stated(self, docs):
        assert "Alignment Setting file" in docs

    def test_covers_both_lipidsearch_versions(self, docs):
        assert "5.0" in docs and "5.2" in docs

    def test_grade_beats_id_rate_in_best_peak_selection(self, docs):
        """The selection rule sorts by grade priority first and only then by
        TotalSmpIDRate(%). The docs used to name the ID rate alone, which is
        the opposite outcome whenever the two disagree."""
        assert "TotalSmpIDRate(%)" in docs
        assert "best grade first" in docs


class TestDocumentedGradeDefaultsMatchTheCleaner:
    """The grade table is a promise about default behaviour; tie it to the
    code that implements it."""

    def test_default_eligibility_matches_the_documented_table(self):
        import pandas as pd
        from app.services.data_cleaning.lipidsearch import LipidSearchCleaner

        df = pd.DataFrame({
            'TotalGrade': ['A', 'B', 'C', 'C', 'D'],
            'ClassKey': ['PC', 'PC', 'PC', 'LPC', 'PC'],
        })
        eligible = LipidSearchCleaner._build_eligibility_mask(df, None)

        # A and B kept for any class; C only for LPC/SM; D never.
        assert list(eligible) == [True, True, False, True, False]

        docs = get_processing_docs(FORMAT_LIPIDSEARCH)
        assert "Keep for LPC/SM only" in docs
