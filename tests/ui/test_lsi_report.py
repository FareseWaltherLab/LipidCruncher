"""
UI tests for the LSI compliance report, driven through the full app.

Each test loads a sample dataset, normalizes it, moves on to Quality Check &
Analysis and reads the report's CSV download, so the auto-filled fields are
checked against what the user actually did in Modules 1 and 2.
"""

import io
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.constants import FORMAT_LIPIDSEARCH, PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')


def _run_to_analysis(method, data_format=None, drop_classes=(), setup=None):
    """Sample data → confirm → ``setup(at)`` → normalization ``method`` → Next."""
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    if data_format is not None:
        at.sidebar.selectbox[0].set_value(data_format).run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    if setup is not None:
        setup(at)
        assert not at.exception, at.exception
    if drop_classes:
        classes = at.multiselect(key='temp_selected_classes')
        classes.set_value([c for c in classes.options if c not in drop_classes]).run()
    at.radio(key='norm_method_selection').set_value(method).run()
    assert not at.exception, at.exception
    [b for b in at.button if b.label.startswith('Next')][0].click().run()
    assert not at.exception, at.exception
    return at


def _report(captured):
    """The report CSV as {item: value}."""
    assert 'lsi_download_csv' in captured, "no LSI CSV download rendered"
    df = pd.read_csv(io.BytesIO(captured['lsi_download_csv']))
    return dict(zip(df['Item'], df['Value']))


class TestNormalizationReported:
    """Regression: the report said "Not configured" whatever normalization
    ran, because it read a session key nothing wrote and a result field
    the normalization result did not have."""

    def test_total_intensity(self, captured_downloads):
        _run_to_analysis('Total Intensity')
        assert _report(captured_downloads)['Normalization method'] == (
            'Total Intensity Normalization'
        )

    def test_none(self, captured_downloads):
        _run_to_analysis('None (pre-normalized data)')
        assert _report(captured_downloads)['Normalization method'] == 'None'

    def test_internal_standards_with_mappings(self, captured_downloads):
        # The sample's PG standard is zero in the BQC samples.
        at = _run_to_analysis('Internal Standards', FORMAT_LIPIDSEARCH, drop_classes={'PG'})
        report = _report(captured_downloads)
        assert report['Normalization method'] == 'Internal Standard Normalization'
        mapping = at.session_state['class_standard_map']
        assert report['Standard-to-class mappings'] == '; '.join(
            f'{cls} → {std}' for cls, std in sorted(mapping.items())
        )


GRADE_ITEM = 'Cleaning: Grade filter (LipidSearch)'


class TestGradeFilterReported:
    """Regression: the grade filter LipidSearch data went through never
    reached the report, which read a grade_config session key nothing
    wrote."""

    def test_default_grades(self, captured_downloads):
        _run_to_analysis('None (pre-normalized data)', FORMAT_LIPIDSEARCH)
        assert _report(captured_downloads)[GRADE_ITEM] == (
            'Default: A, B for all classes; A, B, C for LPC, SM'
        )

    def test_allow_c_for_all_classes(self, captured_downloads):
        def allow_c(at):
            at.radio(key='grade_filter_mode_radio').set_value(
                'Allow C for all classes'
            ).run()

        at = _run_to_analysis(
            'None (pre-normalized data)', FORMAT_LIPIDSEARCH, setup=allow_c,
        )
        entries = _report(captured_downloads)[GRADE_ITEM].split('; ')
        classes = sorted(at.session_state['standardized_df']['ClassKey'].unique())
        assert entries == [f'{cls}: A, B, C' for cls in classes]

    def test_not_reported_for_other_formats(self, captured_downloads):
        _run_to_analysis('None (pre-normalized data)')
        assert GRADE_ITEM not in _report(captured_downloads)
