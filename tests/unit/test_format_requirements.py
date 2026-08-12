"""Unit tests for the in-app "Data Format Requirements" panel.

This panel tells users what to upload, so a wrong claim here costs them a
failed run or a mangled dataset. Like the processing docs, it drifts silently.
These tests pin the claims to the code that implements them: each one first
asserts what the standardizer actually does, then asserts the panel says so.
"""
import pandas as pd
import pytest

from app.services.data_standardization import DataStandardizationService
from app.ui.format_requirements import (
    GENERIC_REQUIREMENTS,
    LIPIDSEARCH_REQUIREMENTS,
    METABOLOMICS_WORKBENCH_REQUIREMENTS,
    MSDIAL_REQUIREMENTS,
)

ALL_REQUIREMENTS = [
    GENERIC_REQUIREMENTS,
    LIPIDSEARCH_REQUIREMENTS,
    MSDIAL_REQUIREMENTS,
    METABOLOMICS_WORKBENCH_REQUIREMENTS,
]


class TestAllFormatsHaveRequirements:
    @pytest.mark.parametrize("text", ALL_REQUIREMENTS)
    def test_non_empty(self, text):
        assert text.strip()


class TestGenericClassKeyColumn:
    """Regression: the panel said "No extra columns allowed! Remove any
    metadata columns" — but an optional lipid-class column in position 2 is
    supported, and keeping it is better than inferring the class from the
    name. Users were being told to delete a column the app wants."""

    def test_second_column_is_taken_as_classkey(self):
        df = pd.DataFrame({
            'Lipid': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'ClassKey': ['PC', 'PE'],
            'A': [1.0, 2.0],
            'B': [3.0, 4.0],
        })
        cols, has_classkey, n_intensity = (
            DataStandardizationService._build_generic_column_map(df)
        )
        assert has_classkey is True
        assert cols == ['LipidMolec', 'ClassKey', 'intensity[s1]', 'intensity[s2]']
        assert n_intensity == 2

    def test_numeric_second_column_stays_a_sample(self):
        df = pd.DataFrame({
            'Lipid': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'A': [1.0, 2.0],
            'B': [3.0, 4.0],
        })
        cols, has_classkey, n_intensity = (
            DataStandardizationService._build_generic_column_map(df)
        )
        assert has_classkey is False
        assert cols == ['LipidMolec', 'intensity[s1]', 'intensity[s2]']

    def test_everything_after_the_class_column_becomes_a_sample(self):
        """The reason the "remove metadata" warning still belongs in the panel:
        trailing columns are renamed to intensity[*] whatever they hold."""
        df = pd.DataFrame({
            'Lipid': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'A': [1.0],
            'Notes': ['some annotation'],
        })
        cols, _, n_intensity = (
            DataStandardizationService._build_generic_column_map(df)
        )
        assert cols == ['LipidMolec', 'ClassKey', 'intensity[s1]', 'intensity[s2]']
        assert n_intensity == 2

    def test_panel_documents_the_optional_class_column(self):
        assert "ClassKey" in GENERIC_REQUIREMENTS
        assert "optional" in GENERIC_REQUIREMENTS.lower()

    def test_panel_no_longer_forbids_all_extra_columns(self):
        assert "No extra columns allowed" not in GENERIC_REQUIREMENTS
        # …but still warns about genuine metadata columns.
        assert "phantom sample" in GENERIC_REQUIREMENTS


class TestMSDIALLipidColumnNames:
    """Regression: the panel called `Metabolite name` "exact name required".
    Four other headers are accepted."""

    ACCEPTED = ['Metabolite name', 'Metabolite', 'Name', 'Lipid', 'LipidMolec']

    @staticmethod
    def _frame(lipid_col_name):
        return pd.DataFrame({
            lipid_col_name: ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'Average Rt(min)': [1.0, 2.0],
            'S1': [10.0, 20.0],
            'S2': [30.0, 40.0],
        })

    @pytest.mark.parametrize("name", ACCEPTED)
    def test_each_documented_name_is_accepted(self, name):
        structure = DataStandardizationService._detect_msdial_structure(
            self._frame(name)
        )
        assert structure.lipid_col == name

    def test_an_undocumented_name_is_rejected(self):
        with pytest.raises(ValueError, match="No lipid name column found"):
            DataStandardizationService._detect_msdial_structure(
                self._frame('Bogus')
            )

    @pytest.mark.parametrize("name", ACCEPTED)
    def test_panel_lists_every_accepted_name(self, name):
        assert name in MSDIAL_REQUIREMENTS


class TestMSDIALSampleColumnsAreFoundByContent:
    """Regression: the panel said sample columns "must be LAST columns".
    Position is irrelevant — detection is "not a known metadata column and
    mostly numeric", so a custom numeric column anywhere becomes a sample."""

    def test_a_sample_column_before_metadata_is_still_detected(self):
        df = pd.DataFrame({
            'Metabolite name': ['PC 16:0_18:1'],
            'S1': [10.0],
            'Average Rt(min)': [1.0],   # metadata AFTER a sample column
            'S2': [30.0],
        })
        raw_cols, _, _, _, _ = (
            DataStandardizationService._detect_msdial_sample_columns(df)
        )
        assert raw_cols == ['S1', 'S2']

    def test_an_unknown_numeric_column_becomes_a_phantom_sample(self):
        df = pd.DataFrame({
            'Metabolite name': ['PC 16:0_18:1'],
            'My Custom Score': [42.0],   # not an MS-DIAL metadata column
            'S1': [10.0],
        })
        raw_cols, _, _, _, _ = (
            DataStandardizationService._detect_msdial_sample_columns(df)
        )
        assert raw_cols == ['My Custom Score', 'S1']

    def test_panel_warns_about_this_instead_of_claiming_position_matters(self):
        assert "must be LAST columns" not in MSDIAL_REQUIREMENTS
        assert "phantom sample" in MSDIAL_REQUIREMENTS


class TestLipidSearchDualPolarityAndDelimiter:
    def test_both_versions_and_layouts_are_covered(self):
        assert "5.0" in LIPIDSEARCH_REQUIREMENTS
        assert "5.2" in LIPIDSEARCH_REQUIREMENTS
        assert "MeanArea" in LIPIDSEARCH_REQUIREMENTS
        assert "OriginalArea" in LIPIDSEARCH_REQUIREMENTS

    def test_alignment_file_requirement_is_stated(self):
        assert "Alignment Setting file" in LIPIDSEARCH_REQUIREMENTS

    def test_delimiter_detection_is_mentioned(self):
        """5.2 exports are tab-delimited despite the .csv extension; users
        should not go converting them by hand."""
        assert "tab-delimited" in LIPIDSEARCH_REQUIREMENTS
