"""
Tests for the CSV conversion behind every CSV download button.
"""

import io

import pandas as pd

from app.ui.download_utils import convert_df


def _read(csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(csv_bytes))


class TestConvertDfIndex:
    """An informative index is written out; a positional one is not."""

    def test_multiindex_row_labels_are_kept(self):
        # Regression: the species heatmaps' (LipidMolec, ClassKey) index was
        # dropped, leaving only concentration[...] columns in the CSV.
        df = pd.DataFrame(
            {'concentration[s1]': [0.5, -0.5]},
            index=pd.MultiIndex.from_tuples(
                [('PC 34:1', 'PC'), ('PE 36:2', 'PE')],
                names=['LipidMolec', 'ClassKey'],
            ),
        )
        out = _read(convert_df(df))
        assert list(out.columns) == ['LipidMolec', 'ClassKey', 'concentration[s1]']
        assert out['LipidMolec'].tolist() == ['PC 34:1', 'PE 36:2']

    def test_named_index_is_kept(self):
        # Regression: the class-aggregated heatmap's ClassKey index was dropped.
        df = pd.DataFrame(
            {'concentration[s1]': [1.0, -1.0]},
            index=pd.Index(['PC', 'PE'], name='ClassKey'),
        )
        out = _read(convert_df(df))
        assert list(out.columns) == ['ClassKey', 'concentration[s1]']
        assert out['ClassKey'].tolist() == ['PC', 'PE']

    def test_positional_index_is_dropped(self):
        df = pd.DataFrame({'a': [1, 2]})
        assert list(_read(convert_df(df)).columns) == ['a']
        # A filtered or concatenated frame has a non-contiguous integer index.
        gappy = pd.concat([df, df])
        assert list(_read(convert_df(gappy)).columns) == ['a']


class TestConvertDfSampleNames:
    """Sample names replace the s-labels in the written CSV."""

    def test_names_applied_after_index_is_written(self):
        df = pd.DataFrame(
            {'concentration[s1]': [1.0], 'concentration[s2]': [2.0]},
            index=pd.Index(['PC'], name='ClassKey'),
        )
        out = _read(convert_df(df, {'s1': 'ID_01'}))
        assert list(out.columns) == [
            'ClassKey', 'concentration[ID_01]', 'concentration[s2]',
        ]

    def test_without_names_labels_are_kept(self):
        df = pd.DataFrame({'concentration[s1]': [1.0]})
        assert list(_read(convert_df(df)).columns) == ['concentration[s1]']
