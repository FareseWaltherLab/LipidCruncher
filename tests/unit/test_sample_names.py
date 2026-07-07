"""
Tests for sample display-name handling.

Covers the pure helpers in ``app.ui.sample_names`` that build, format, and
remap the ``{internal label -> display name}`` map used by the sidebar.
"""

import pandas as pd

from app.ui.sample_names import (
    build_names_from_mapping,
    display_label,
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
