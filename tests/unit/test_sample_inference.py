"""Unit tests for SampleInferenceService (AI Studio sidebar guesses)."""
import pandas as pd
import pytest

from app.services.sample_inference import SampleInferenceService as S


class TestNumericColumns:
    def test_text_and_empty_columns_are_not_numeric(self):
        df = pd.DataFrame({
            'lipid': ['PC 34:1', 'PE 36:2'],
            'notes': [None, None],
            'mixed': ['n/a', '3.5'],
            'KO-1': [1.0, 2.0],
        })
        assert S.numeric_columns(df) == ['mixed', 'KO-1']


class TestColumnFamilies:
    def test_groups_bracketed_columns_by_measurement(self):
        cols = ['Area[KO-1]', 'Area[KO-2]', 'Height[KO-1]', 'Height[KO-2]', 'RT', 'Grade[KO-1]']
        assert S.column_families(cols) == {
            'Area': ['Area[KO-1]', 'Area[KO-2]'],
            'Height': ['Height[KO-1]', 'Height[KO-2]'],
        }

    def test_flat_names_have_no_families(self):
        assert S.column_families(['KO-1', 'KO-2', 'RetentionTime']) == {}

    @pytest.mark.parametrize("column, expected", [
        ('Area[KO-1]', 'KO-1'), ('intensity[s3]', 's3'), ('KO-1', 'KO-1'),
    ])
    def test_sample_name_strips_the_measurement(self, column, expected):
        assert S.sample_name(column) == expected


class TestLooksLikeMetadata:
    @pytest.mark.parametrize("name", [
        'RetentionTime', 'PrecursorMZ', 'QualityScore', 'Notes', 'CalcMass',
        'Average Rt(min)', 'm/z', 'RT', 'S/N',
    ])
    def test_metadata_names(self, name):
        assert S.looks_like_metadata(name)

    @pytest.mark.parametrize("name", [
        'KO-1', 'WT_3', 'Heart_1', 'Start_2', 'mut1-1', 'Sample10',
    ])
    def test_sample_names(self, name):
        """'Heart' and 'Start' contain 'rt' but are not the word RT."""
        assert not S.looks_like_metadata(name)


class TestGuessConditions:
    def test_strips_replicate_numbers(self):
        names = ['KO-1', 'KO-2', 'KOWT-1', 'KOWT-2', 'mut1-1', 'mut1-2', 'WT_rep1', 'WT_rep2']
        assert S.guess_conditions(names) == [
            'KO', 'KO', 'KOWT', 'KOWT', 'mut1', 'mut1', 'WT', 'WT',
        ]

    def test_digits_without_separator(self):
        assert S.guess_conditions(['WT1', 'WT2', 'Liver1', 'Liver2']) == [
            'WT', 'WT', 'Liver', 'Liver',
        ]

    @pytest.mark.parametrize("names", [
        ['s1', 's2', 's3', 's4'],          # one condition only
        ['A1', 'B1', 'C1'],                # no condition with two samples
        ['1', '2', '3', '4'],              # nothing left after the number
        ['S1A', 'S2A', 'S1B', 'S2B'],      # no trailing replicate number
    ])
    def test_no_grouping_returns_none(self, names):
        assert S.guess_conditions(names) is None
