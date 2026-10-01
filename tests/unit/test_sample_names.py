"""
Tests for sample display-name handling.

Covers the pure helpers in ``app.ui.sample_names`` that build, format, and
remap the ``{internal label -> display name}`` map used by the sidebar.
"""

import pandas as pd

from app.ui.sample_names import (
    build_names_from_mapping,
    display_label,
    name_samples_for_csv,
    remap_names_after_exclusion,
    remap_names_after_regroup,
)


class TestBuildNamesFromMapping:
    """build_names_from_mapping parses/cleans the column-mapping table."""

    def test_extracts_generic_headers(self):
        cm = pd.DataFrame({
            'standardized_name': ['LipidMolec', 'intensity[s1]', 'intensity[s2]'],
            'original_name': ['Lipid', 'mouse_liver_001', 'mouse_liver_002'],
        })
        assert build_names_from_mapping(cm) == {
            's1': 'mouse_liver_001', 's2': 'mouse_liver_002',
        }

    def test_strips_wrapper_prefix(self):
        cm = pd.DataFrame({
            'standardized_name': ['intensity[s1]'],
            'original_name': ['MeanArea[SampleA]'],
        })
        assert build_names_from_mapping(cm) == {'s1': 'SampleA'}

    def test_composite_dual_polarity_header_left_intact(self):
        # LipidSearch 5.2 dual-polarity merges two per-file tokens into one
        # header; a greedy wrapper strip would mangle it to
        # "s1-1] + OriginalArea[s1-2". It must stay whole and match the
        # column-standardization table.
        cm = pd.DataFrame({
            'standardized_name': ['intensity[s1]'],
            'original_name': ['OriginalArea[s1-1] + OriginalArea[s1-2]'],
        })
        assert build_names_from_mapping(cm) == {
            's1': 'OriginalArea[s1-1] + OriginalArea[s1-2]'
        }

    def test_omits_names_identical_to_label(self):
        # LipidSearch flat exports often carry MeanArea[s1] -> inner == label.
        cm = pd.DataFrame({
            'standardized_name': ['intensity[s1]', 'intensity[s2]'],
            'original_name': ['MeanArea[s1]', 'real name'],
        })
        assert build_names_from_mapping(cm) == {'s2': 'real name'}

    def test_handles_missing_or_empty(self):
        assert build_names_from_mapping(None) == {}
        assert build_names_from_mapping(pd.DataFrame()) == {}


class TestDisplayFormatting:
    """display_label renders 's3 — name'."""

    def test_display_label_with_name(self):
        assert display_label('s3', {'s3': 'mouse liver #5'}) == 's3 — mouse liver #5'

    def test_display_label_without_name(self):
        assert display_label('s3', {'s1': 'x'}) == 's3'
        assert display_label('s3', None) == 's3'


class TestRemapAfterRegroup:
    """Names follow the intensity-column rename map from regrouping."""

    def test_permutation(self):
        # regroup moves old s4 -> new s1, old s1 -> new s2.
        old_to_new = {'intensity[s4]': 'intensity[s1]', 'intensity[s1]': 'intensity[s2]'}
        assert remap_names_after_regroup({'s1': 'a', 's4': 'd'}, old_to_new) == {
            's1': 'd', 's2': 'a',
        }


class TestRemapAfterExclusion:
    """Names follow the survivors when sample exclusion renumbers them."""

    def test_survivors_take_the_freed_labels(self):
        # Excluding s2 renumbers old s3 -> s2 and old s4 -> s3; the name must
        # travel with the sample, not stay on the label.
        names = {'s1': 'A', 's2': 'B', 's3': 'C', 's4': 'D'}
        assert remap_names_after_exclusion(
            names, ['s1', 's2', 's3', 's4'], ['s2'], ['s1', 's2', 's3'],
        ) == {'s1': 'A', 's2': 'C', 's3': 'D'}

    def test_several_removed_and_unnamed_survivor(self):
        names = {'s1': 'A', 's4': 'D', 's5': 'E'}
        assert remap_names_after_exclusion(
            names, ['s1', 's2', 's3', 's4', 's5'], ['s1', 's3'],
            ['s1', 's2', 's3'],
        ) == {'s2': 'D', 's3': 'E'}

    def test_shared_name_keeps_its_original_label(self):
        """Regression: the suffix used the post-exclusion label, so
        'QC (s4)' meant original s4 before the exclusion and original s5
        after it."""
        names = {'s4': 'QC', 's5': 'QC'}
        labels = ['s1', 's2', 's3', 's4', 's5']
        remapped = remap_names_after_exclusion(names, labels, ['s1'], labels[:4])
        assert remapped == {'s3': 'QC (s4)', 's4': 'QC (s5)'}
        post = pd.DataFrame(columns=['concentration[s3]', 'concentration[s4]'])
        assert list(name_samples_for_csv(post, remapped).columns) == [
            'concentration[QC (s4)]', 'concentration[QC (s5)]',
        ]

    def test_no_names(self):
        assert remap_names_after_exclusion(None, ['s1', 's2'], ['s1'], ['s1']) == {}
        assert remap_names_after_exclusion({}, ['s1', 's2'], ['s1'], ['s1']) == {}


class TestNameSamplesForCsv:
    """name_samples_for_csv puts display names into CSV headers and labels."""

    def test_keeps_prefix_and_leaves_unnamed_and_other_columns(self):
        df = pd.DataFrame(columns=[
            'LipidMolec', 'concentration[s1]', 'concentration[s2]', 'intensity[s1]',
        ])
        out = name_samples_for_csv(df, {'s1': 'ID_01'})
        assert list(out.columns) == [
            'LipidMolec', 'concentration[ID_01]', 'concentration[s2]',
            'intensity[ID_01]',
        ]

    def test_label_does_not_match_a_longer_label(self):
        # s1's name must not leak onto s10/s11.
        df = pd.DataFrame(columns=['concentration[s1]', 'concentration[s10]'])
        out = name_samples_for_csv(df, {'s1': 'A'})
        assert list(out.columns) == ['concentration[A]', 'concentration[s10]']

    def test_blank_names_keep_the_label(self):
        df = pd.DataFrame(columns=['concentration[s1]', 'concentration[s2]'])
        out = name_samples_for_csv(df, {'s1': '   ', 's2': ''})
        assert list(out.columns) == ['concentration[s1]', 'concentration[s2]']

    def test_names_are_stripped(self):
        df = pd.DataFrame(columns=['concentration[s1]'])
        out = name_samples_for_csv(df, {'s1': '  ID_01 '})
        assert list(out.columns) == ['concentration[ID_01]']

    def test_duplicate_names_get_their_label(self):
        df = pd.DataFrame(columns=[
            'concentration[s1]', 'concentration[s2]', 'concentration[s3]',
        ])
        out = name_samples_for_csv(df, {'s1': 'QC', 's2': 'QC', 's3': 'X'})
        assert list(out.columns) == [
            'concentration[QC (s1)]', 'concentration[QC (s2)]',
            'concentration[X]',
        ]

    def test_name_equal_to_another_label_is_disambiguated(self):
        # Naming s1 "s2" would otherwise give two concentration[s2] headers.
        df = pd.DataFrame(columns=['concentration[s1]', 'concentration[s2]'])
        out = name_samples_for_csv(df, {'s1': 's2'})
        assert list(out.columns) == ['concentration[s2 (s1)]', 'concentration[s2]']
        assert out.columns.is_unique

    def test_bare_label_columns_and_sample_values(self):
        # The shape of the correlation matrix and the PCA/box-plot tables.
        df = pd.DataFrame({
            'Sample': ['s1', 's2'], 's1': [1.0, 0.9], 's2': [0.9, 1.0],
        })
        out = name_samples_for_csv(df, {'s1': 'A'})
        assert list(out.columns) == ['Sample', 'A', 's2']
        assert out['Sample'].tolist() == ['A', 's2']

    def test_name_clashing_with_another_column_keeps_its_label(self):
        df = pd.DataFrame({'Sample': ['s1'], 's1': [1.0]})
        out = name_samples_for_csv(df, {'s1': 'Sample'})
        assert list(out.columns) == ['Sample', 's1']

    def test_no_names_returns_frame_unchanged(self):
        df = pd.DataFrame(columns=['concentration[s1]'])
        assert name_samples_for_csv(df, None) is df
        assert name_samples_for_csv(df, {}) is df

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({'Sample': ['s1'], 'concentration[s1]': [1.0]})
        name_samples_for_csv(df, {'s1': 'A'})
        assert list(df.columns) == ['Sample', 'concentration[s1]']
        assert df['Sample'].tolist() == ['s1']
