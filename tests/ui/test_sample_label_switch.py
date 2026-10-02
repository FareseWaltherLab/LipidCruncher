"""
UI tests for the "Show samples as" switch in plots, pickers and summaries.

With original names chosen, every place that names a sample shows the same
text the tables and CSVs use; with standardized labels chosen, it shows
s1, s2, ... exactly as before names existed. The data stays keyed by
s-label either way.

Covers:
1. Quality Check plots (missing values, box plot, correlation, PCA), including
   the figures stored for the PDF report.
2. The PCA "Exclude Samples" picker, whose selection survives a flip.
3. The LSI report's "Outlier samples removed (PCA)" field.
4. Normalization's per-sample protein inputs and their zero-value error.
5. The sidebar's Confirm Inputs summary and manual-regroup pickers.
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    full_sidebar_script,
    make_analysis_dataframe,
    make_cleaned_dataframe,
    normalization_script,
    qc_module_script,
)


NAMES = {
    's1': 'CLEANUP_BLANK_01', 's2': '140509_FT_01', 's3': 'ID_03',
    's4': 'QC', 's5': 'QC', 's6': 'ID_06',
}
# The text shown for each sample: a shared name gets its label appended.
SHOWN = [
    'CLEANUP_BLANK_01', '140509_FT_01', 'ID_03', 'QC (s4)', 'QC (s5)', 'ID_06',
]
LABELS = ['s1', 's2', 's3', 's4', 's5', 's6']


def _experiment():
    from app.models.experiment import ExperimentConfig
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )


def _qc_app(mode='original'):
    at = AppTest.from_function(qc_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = _experiment()
    at.session_state['sample_names'] = dict(NAMES)
    at.session_state['sample_label_mode'] = mode
    at.run()
    assert not at.exception
    return at


def _missing_values_bars(at):
    fig = at.session_state['qc_box_plot_fig1']
    return [t.y[0] for t in fig.data if t.y[0] is not None]


def _boxes(at):
    fig = at.session_state['qc_box_plot_fig2']
    return [t.name for t in fig.data if t.showlegend is False]


def _correlation_ticks(at):
    ax = at.session_state['qc_correlation_plots']['Control'].axes[0]
    return [t.get_text() for t in ax.get_xticklabels()]


def _pca_hover(at):
    fig = at.session_state['qc_pca_plot']
    return [text for t in fig.data if t.mode == 'markers' for text in t.text]


# =============================================================================
# 1. Quality Check plots
# =============================================================================

class TestQualityCheckPlots:
    """Regression: the plots (and the PDF copies of them) kept s1, s2, ...
    while the tables and CSVs next to them showed the names."""

    def test_original_names(self):
        at = _qc_app('original')
        assert _missing_values_bars(at) == SHOWN
        assert _boxes(at) == SHOWN
        assert _correlation_ticks(at) == SHOWN[:3]
        assert _pca_hover(at) == SHOWN

    def test_standardized_labels(self):
        at = _qc_app('standardized')
        assert _missing_values_bars(at) == LABELS
        assert _boxes(at) == LABELS
        assert _correlation_ticks(at) == LABELS[:3]
        assert _pca_hover(at) == LABELS

    def test_flipping_the_switch_relabels_the_plots(self):
        at = _qc_app('original')
        at.session_state['sample_label_mode'] = 'standardized'
        at.run()
        assert _boxes(at) == LABELS
        assert _pca_hover(at) == LABELS
        at.session_state['sample_label_mode'] = 'original'
        at.run()
        assert _boxes(at) == SHOWN
        assert _pca_hover(at) == SHOWN

    def test_flip_relabels_every_stored_correlation_heatmap(self):
        """The PDF holds a heatmap per condition viewed; a flip must relabel
        all of them, not only the one on screen."""
        at = _qc_app('original')
        at.selectbox(key='corr_condition').set_value('Treatment').run()
        at.session_state['sample_label_mode'] = 'standardized'
        at.run()
        assert not at.exception
        ticks = {
            condition: [t.get_text() for t in fig.axes[0].get_xticklabels()]
            for condition, fig in at.session_state['qc_correlation_plots'].items()
        }
        assert ticks == {'Control': LABELS[:3], 'Treatment': LABELS[3:]}

    def test_plot_and_table_agree_on_correlation(self):
        at = _qc_app('original')
        table = [
            d.value for d in at.dataframe if 'Sample' in d.value.columns
        ][0]
        assert table['Sample'].tolist() == _correlation_ticks(at)

    def test_pca_after_exclusion_names_the_survivors(self):
        """Excluding s2 renumbers old s3 to s2; its point is still ID_03."""
        at = _qc_app('original')
        at.multiselect(key='pca_samples_remove').set_value(['s2']).run()
        assert not at.exception
        assert _pca_hover(at) == [
            'CLEANUP_BLANK_01', 'ID_03', 'QC (s4)', 'QC (s5)', 'ID_06',
        ]
        assert at.session_state['qc_pca_plot'] is not None


# =============================================================================
# 2. PCA "Exclude Samples" picker
# =============================================================================

class TestExcludeSamplesPicker:
    """Options read as names; values stay s-labels."""

    def test_options_follow_the_switch(self):
        at = _qc_app('original')
        assert at.multiselect(key='pca_samples_remove').options == SHOWN
        at = _qc_app('standardized')
        assert at.multiselect(key='pca_samples_remove').options == LABELS

    def test_selection_is_s_labels(self):
        at = _qc_app('original')
        at.multiselect(key='pca_samples_remove').set_value(['s4']).run()
        assert at.multiselect(key='pca_samples_remove').value == ['s4']
        assert at.session_state['qc_samples_removed'] == ['s4']

    def test_flipping_the_switch_keeps_the_selection(self):
        at = _qc_app('original')
        at.multiselect(key='pca_samples_remove').set_value(['s2', 's5']).run()
        for mode in ('standardized', 'original'):
            at.session_state['sample_label_mode'] = mode
            at.run()
            assert not at.exception
            assert at.multiselect(key='pca_samples_remove').value == ['s2', 's5']
            assert at.session_state['qc_samples_removed'] == ['s2', 's5']
            assert at.text[-1].value == 'qc_samples:4'


# =============================================================================
# 3. LSI report
# =============================================================================

def _lsi_summary_script():
    import streamlit as st
    from app.ui.main_content.lsi_report import _build_qc_summary
    st.text(_build_qc_summary()["Outlier samples removed (PCA)"])


class TestLsiOutlierSummary:
    @pytest.mark.parametrize('mode, expected', [
        ('original', 'CLEANUP_BLANK_01, QC (s4)'),
        ('standardized', 's1, s4'),
    ])
    def test_removed_samples_follow_the_switch(self, mode, expected):
        """Regression: the LSI PDF/CSV listed s-labels even with names on."""
        at = AppTest.from_function(_lsi_summary_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['qc_input_sample_names'] = dict(NAMES)
        at.session_state['qc_samples_removed'] = ['s1', 's4']
        at.session_state['sample_label_mode'] = mode
        at.run()
        assert not at.exception
        assert at.text[0].value == expected


# =============================================================================
# 4. Normalization protein inputs
# =============================================================================

def _protein_app(mode='original'):
    at = AppTest.from_function(normalization_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_intsta_df'] = None
    at.session_state['_test_experiment'] = _experiment()
    at.session_state['sample_names'] = dict(NAMES)
    at.session_state['sample_label_mode'] = mode
    at.run()
    at.radio(key='norm_method_selection').set_value('Protein-based').run()
    assert not at.exception
    return at


class TestProteinInputs:
    def test_labels_follow_the_switch(self):
        at = _protein_app('original')
        labels = [at.number_input(key=f'protein_{s}').label for s in LABELS]
        assert labels == [f'{text}:' for text in SHOWN]
        at = _protein_app('standardized')
        labels = [at.number_input(key=f'protein_{s}').label for s in LABELS]
        assert labels == [f'{s}:' for s in LABELS]

    def test_flipping_the_switch_keeps_entered_values(self):
        at = _protein_app('original')
        at.number_input(key='protein_s2').set_value(42.5).run()
        at.session_state['sample_label_mode'] = 'standardized'
        at.run()
        assert not at.exception
        assert at.number_input(key='protein_s2').label == 's2:'
        assert at.number_input(key='protein_s2').value == 42.5
        at.session_state['sample_label_mode'] = 'original'
        at.run()
        assert at.number_input(key='protein_s2').value == 42.5

    def test_zero_value_error_names_the_sample(self):
        at = _protein_app('original')
        at.number_input(key='protein_s4').set_value(0.0).run()
        errors = [e.value for e in at.error]
        assert any('QC (s4)' in e for e in errors), errors


# =============================================================================
# 5. Sidebar: Confirm Inputs summary and manual regroup pickers
# =============================================================================

def _confirm_script():
    from app.models.experiment import ExperimentConfig
    from app.ui.sidebar.confirm_inputs import display_confirm_inputs
    display_confirm_inputs(ExperimentConfig(
        n_conditions=2, conditions_list=['A', 'B'],
        number_of_samples_list=[3, 6],
    ))


class TestConfirmSummary:
    @pytest.mark.parametrize('mode, expected', [
        ('original', [
            '• N-1, N2, s3 correspond to A',
            '• N4 to N9 (total 6) correspond to B',
        ]),
        ('standardized', [
            '• s1-s2-s3 correspond to A',
            '• s4 to s9 (total 6) correspond to B',
        ]),
    ])
    def test_summary_follows_the_switch(self, mode, expected):
        at = AppTest.from_function(_confirm_script, default_timeout=DEFAULT_TIMEOUT)
        names = {f's{i}': f'N{i}' for i in range(2, 10)}
        names['s1'] = 'N-1'
        names.pop('s3')
        at.session_state['sample_names'] = names
        at.session_state['sample_label_mode'] = mode
        at.run()
        assert not at.exception
        assert [t.value for t in at.sidebar.text] == expected


class TestRegroupPickers:
    """The regroup pickers show names from the stable pre-regroup snapshot.

    Regression guard: a format_func reading the live sample_names (which the
    regroup rewrites mid-run) once reset these widgets and blanked the
    selections; flipping the switch must not blank them either.
    """

    @staticmethod
    def _counts(at):
        return [
            at.sidebar.number_input[i + 1].value
            for i in range(at.sidebar.number_input[0].value)
        ]

    def _regroup_app(self):
        at = AppTest.from_function(full_sidebar_script, default_timeout=60)
        at.run()
        at.sidebar.selectbox[0].set_value('MS-DIAL').run()
        at.sidebar.button(key='load_sample').click().run()
        n = sum(self._counts(at))
        at.session_state['sample_names'] = {f's{i}': f'N{i}' for i in range(1, n + 1)}
        at.run()
        at.sidebar.radio(key='grouping_radio').set_value('No').run()
        assert not at.exception
        return at

    def test_options_show_names_and_values_stay_labels(self):
        at = self._regroup_app()
        key = [m.key for m in at.sidebar.multiselect if m.key.startswith('select_')][0]
        assert at.sidebar.multiselect(key=key).options[:3] == ['N1', 'N2', 'N3']
        at.sidebar.multiselect(key=key).set_value(['s2']).run()
        assert at.sidebar.multiselect(key=key).value == ['s2']

    def test_flip_after_regroup_then_confirm(self):
        """The switch appears once the regroup is complete; flipping it there
        relabels the pickers without blanking them or undoing the regroup."""
        at = self._regroup_app()
        keys = [m.key for m in at.sidebar.multiselect if m.key.startswith('select_')]
        counts = self._counts(at)
        shuffled = [f's{i}' for i in range(sum(counts), 0, -1)]
        picks, idx = [], 0
        for key, n in zip(keys, counts):
            picks.append(shuffled[idx:idx + n])
            at.sidebar.multiselect(key=key).set_value(picks[-1]).run()
            idx += n
        assert at.session_state['grouping_complete'] is True

        for mode in ('standardized', 'original'):
            at.sidebar.radio(key='_sample_label_mode_radio').set_value(mode).run()
            assert not at.exception
            assert [at.sidebar.multiselect(key=k).value for k in keys] == picks
            assert at.session_state['grouping_complete'] is True
            first = at.sidebar.multiselect(key=keys[0]).options[0]
            assert first == ('s1' if mode == 'standardized' else 'N1')

        at.sidebar.checkbox(key='confirm_checkbox').set_value(True).run()
        assert not at.exception
        assert at.session_state['confirmed'] is True
        assert [at.sidebar.multiselect(key=k).value for k in keys] == picks
        # Names followed their samples through the regroup.
        assert at.session_state['sample_names']['s1'] == f'N{sum(counts)}'
