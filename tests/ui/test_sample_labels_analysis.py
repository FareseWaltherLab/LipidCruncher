"""
UI tests for the "Show samples as" switch in Analysis plots and the report.

With original names chosen, the lipidomic heatmap's sample axis, the
internal-standards consistency plots and the PDF cover's Sample IDs show the
same text the tables and CSVs use; with standardized labels chosen, they show
s1, s2, ... exactly as before names existed. The figures stored for the PDF
report are the ones shown, so they carry the same labels.

Covers:
1. Lipidomic heatmap (space 'qc'), including a flip of the switch and the
   renumbering that Quality Check's sample exclusion causes.
2. Internal standards consistency plots (space 'upload').
3. The PDF report cover's Sample IDs.
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    analysis_module_script,
    internal_standards_script,
    make_analysis_dataframe,
    make_cleaned_dataframe,
    make_intsta_dataframe,
    qc_and_analysis_script,
)


NAMES = {
    's1': 'CLEANUP_BLANK_01', 's2': 'ID_02', 's3': 'QC',
    's4': 'QC', 's5': 'SS48_JW_03',
}  # s6 unnamed
SHOWN = ['CLEANUP_BLANK_01', 'ID_02', 'QC (s3)', 'QC (s4)', 'SS48_JW_03', 's6']
LABELS = ['s1', 's2', 's3', 's4', 's5', 's6']


def _experiment():
    from app.models.experiment import ExperimentConfig
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )


def _select(at, key, word):
    option = [o for o in at.radio(key=key).options if word in o][0]
    at.radio(key=key).set_value(option).run()
    assert not at.exception
    return at


def _heatmap_app(mode='original'):
    at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = _experiment()
    at.session_state['qc_sample_names'] = dict(NAMES)
    at.session_state['sample_label_mode'] = mode
    at.run()
    assert not at.exception
    return _select(at, 'analysis_radio', 'Lipidomic Heatmap')


def _heatmap_x(at):
    return list(at.session_state['analysis_heatmap_fig'].data[0].x)


# =============================================================================
# 1. Lipidomic heatmap
# =============================================================================

class TestHeatmapSampleAxis:

    @pytest.mark.parametrize(
        'heatmap_type',
        ['Clustered', 'Regular', 'Grouped by Class', 'Aggregated by Class'],
    )
    def test_original_names_on_the_axis(self, heatmap_type):
        at = _select(_heatmap_app(), 'heatmap_type', heatmap_type)
        assert _heatmap_x(at) == SHOWN
        # The PDF report embeds the figure that is shown.
        assert list(at.session_state['analysis_all_plots']['heatmap'].data[0].x) == SHOWN

    def test_standardized_labels_on_the_axis(self):
        at = _heatmap_app(mode='standardized')
        assert _heatmap_x(at) == LABELS
        assert at.session_state['analysis_heatmap_fig'].layout.xaxis.tickangle == 45

    def test_flipping_the_switch_redraws_the_axis(self):
        """The cached heatmap is keyed on the names, so a flip is not served
        the other mode's figure."""
        at = _heatmap_app()
        assert _heatmap_x(at) == SHOWN
        at.session_state['sample_label_mode'] = 'standardized'
        at.run()
        assert not at.exception
        assert _heatmap_x(at) == LABELS
        at.session_state['sample_label_mode'] = 'original'
        at.run()
        assert _heatmap_x(at) == SHOWN

    def test_names_follow_the_survivors_after_exclusion(self):
        """Excluding s2 renumbers old s3 to s2; its column must still be
        called by s3's name."""
        at = AppTest.from_function(qc_and_analysis_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
        at.session_state['_test_experiment'] = _experiment()
        at.session_state['sample_names'] = dict(NAMES)
        at.run()
        assert not at.exception
        at.multiselect(key='pca_samples_remove').set_value(['s2']).run()
        assert not at.exception
        _select(at, 'analysis_radio', 'Lipidomic Heatmap')
        assert _heatmap_x(at) == [
            'CLEANUP_BLANK_01', 'QC (s3)', 'QC (s4)', 'SS48_JW_03', 's5',
        ]


# =============================================================================
# 2. Internal standards consistency plots
# =============================================================================

def _standards_app(mode):
    at = AppTest.from_function(internal_standards_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_auto_detected_df'] = make_intsta_dataframe()
    at.session_state['_test_experiment'] = _experiment()
    at.session_state['sample_names'] = dict(NAMES)
    at.session_state['sample_label_mode'] = mode
    at.run()
    assert not at.exception
    return at


class TestStandardsPlotSampleAxis:

    @pytest.mark.parametrize('mode, expected', [
        ('original', SHOWN), ('standardized', LABELS),
    ])
    def test_stored_figures_show_the_chosen_labels(self, mode, expected):
        figs = _standards_app(mode).session_state['standards_consistency_figs']
        assert figs
        for fig in figs:
            assert [x for trace in fig.data for x in trace.x] == expected


# =============================================================================
# 3. PDF report cover
# =============================================================================

class TestPdfCoverSampleIds:

    @pytest.fixture
    def captured_metadata(self, monkeypatch):
        import app.ui.main_content.analysis._entry as entry
        captured = {}

        def _fake_report(analysis_plots, metadata, **kwargs):
            captured['metadata'] = metadata
            return b'%PDF-'

        monkeypatch.setattr(entry, 'generate_pdf_report', _fake_report)
        return captured

    @pytest.mark.parametrize('mode, expected', [
        ('original', SHOWN), ('standardized', LABELS),
    ])
    def test_cover_lists_the_chosen_labels(self, captured_metadata, mode, expected):
        at = _heatmap_app(mode)
        at.button(key='generate_pdf_report').click().run()
        assert not at.exception
        details = captured_metadata['metadata'].conditions_detail
        assert [s for _, _, samples in details for s in samples] == expected
