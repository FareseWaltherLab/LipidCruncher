"""
Module 1 format-specific filtering, driven through the full app.

LipidSearch grade filtering (per-class customization, an emptied class, every
class emptied, selections kept across Next / Back) and MS-DIAL quality
filtering (presets, the MS/MS checkbox following the preset, a custom score
threshold), checked against the data the cleaning step produced.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.constants import (
    FORMAT_LIPIDSEARCH, FORMAT_MSDIAL, MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS,
    PAGE_APP,
)
from app.services.data_standardization import MSDIALFeatures
from tests.ui.conftest import DEFAULT_TIMEOUT as WIDGET_TIMEOUT
from tests.ui.conftest import quality_filtering_script

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')


def _module1(data_format):
    """The app on Module 1 with ``data_format``'s sample data confirmed."""
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.selectbox[0].set_value(data_format).run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    assert not at.exception, at.exception
    return at


def _next_and_back(at):
    """Leave Module 1 for Quality Check and come back. Set the module rather
    than click Next / Back: after their st.rerun() AppTest's element tree
    keeps the previous page's widgets, whose dropped keys the next run cannot
    read. Both only clear the qc stage, empty here."""
    for module in (MODULE_QC_ANALYSIS, MODULE_DATA_PROCESSING):
        at.session_state['module'] = module
        at.run()
        assert not at.exception, at.exception


def _rows(at, lipid_class):
    cleaned = at.session_state['cleaned_df']
    return cleaned[cleaned['ClassKey'] == lipid_class]


class TestCustomGrades:

    def test_emptied_class_is_dropped(self):
        at = _module1(FORMAT_LIPIDSEARCH)
        at.radio(key='grade_filter_mode_radio').set_value('Customize by Class').run()
        assert len(_rows(at, 'PC')) > 0
        at.multiselect(key='grade_select_PC').set_value([]).run()
        assert len(_rows(at, 'PC')) == 0
        assert any('Will be excluded' in e.value for e in at.error)

    def test_narrower_grades_keep_fewer_species(self):
        at = _module1(FORMAT_LIPIDSEARCH)
        at.radio(key='grade_filter_mode_radio').set_value('Customize by Class').run()
        default_pe = set(_rows(at, 'PE')['LipidMolec'])  # A, B
        at.multiselect(key='grade_select_PE').set_value(['A']).run()
        assert set(_rows(at, 'PE')['LipidMolec']) < default_pe

    def test_selections_survive_next_and_back(self):
        at = _module1(FORMAT_LIPIDSEARCH)
        at.radio(key='grade_filter_mode_radio').set_value('Customize by Class').run()
        at.multiselect(key='grade_select_PC').set_value([]).run()
        at.multiselect(key='grade_select_PE').set_value(['A']).run()
        pe_rows = len(_rows(at, 'PE'))
        _next_and_back(at)
        assert at.radio(key='grade_filter_mode_radio').value == 'Customize by Class'
        assert at.multiselect(key='grade_select_PC').value == []
        assert at.multiselect(key='grade_select_PE').value == ['A']
        assert len(_rows(at, 'PC')) == 0
        assert len(_rows(at, 'PE')) == pe_rows

    def test_every_class_emptied_stops_module1(self):
        at = _module1(FORMAT_LIPIDSEARCH)
        at.radio(key='grade_filter_mode_radio').set_value('Customize by Class').run()
        for widget in [m for m in at.multiselect if (m.key or '').startswith('grade_select_')]:
            at.multiselect(key=widget.key).set_value([]).run()
        assert not at.exception, at.exception
        assert any('No lipid species remain' in e.value for e in at.error)
        assert not any(b.label.startswith('Next') for b in at.button)


def _quality_config(at):
    config = at.session_state['last_quality_config']
    return config.total_score_threshold, config.require_msms


class TestMsdialQualityLevels:

    def test_preset_sets_threshold_and_msms_checkbox(self):
        at = _module1(FORMAT_MSDIAL)
        assert _quality_config(at) == (60, False)
        at.radio(key='msdial_quality_level_radio').set_value(
            'Strict (Score ≥80, MS/MS required)'
        ).run()
        assert _quality_config(at) == (80, True)
        assert at.checkbox(key='msdial_custom_msms').value is True
        at.radio(key='msdial_quality_level_radio').set_value('No filtering').run()
        assert _quality_config(at) == (0, False)
        assert at.checkbox(key='msdial_custom_msms').value is False

    def test_msms_checkbox_overrides_preset(self):
        at = _module1(FORMAT_MSDIAL)
        at.checkbox(key='msdial_custom_msms').check().run()
        assert _quality_config(at) == (60, True)

    def test_custom_score_overrides_preset(self):
        at = _module1(FORMAT_MSDIAL)
        rows_at_60 = len(at.session_state['ingestion_result'].cleaned_df)
        at.checkbox(key='msdial_show_custom_threshold').check().run()
        at.slider(key='msdial_custom_score').set_value(90).run()
        assert _quality_config(at) == (90, False)
        assert len(at.session_state['ingestion_result'].cleaned_df) < rows_at_60
        assert any('Score ≥ 90' in m.value for m in at.markdown)


class TestQualityFilterFallbacks:
    """Exports without a Total score column, or without either column."""

    def _run(self, has_quality_score, has_msms_matched):
        at = AppTest.from_function(quality_filtering_script, default_timeout=WIDGET_TIMEOUT)
        at.session_state['msdial_features'] = MSDIALFeatures(
            has_quality_score=has_quality_score,
            has_msms_matched=has_msms_matched,
            raw_sample_columns=['s1', 's2', 's3'],
            normalized_sample_columns=[],
        )
        at.run()
        assert not at.exception, at.exception
        return at

    def test_msms_only(self):
        at = self._run(has_quality_score=False, has_msms_matched=True)
        assert [t.value for t in at.text] == ['threshold:0', 'msms:False']
        at.checkbox(key='msdial_msms_only').check().run()
        assert [t.value for t in at.text] == ['threshold:0', 'msms:True']

    def test_no_quality_columns(self):
        at = self._run(has_quality_score=False, has_msms_matched=False)
        assert [t.value for t in at.text] == ['no_quality_config']
        assert any('Quality filtering unavailable' in w.value for w in at.warning)
