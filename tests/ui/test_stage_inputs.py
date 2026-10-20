"""
Outputs computed from a stage's input go when that input changes.

Only the selected analysis page re-renders, but every page's figures stay in
the PDF report (and their tests in the LSI report). When Quality Check's
output changes while another page is shown, the pages not shown would
otherwise keep results computed from the data before the change.
"""

from tests.ui.test_lsi_report import _run_to_analysis, _show_analysis


def _bar_results(at):
    return (
        'bar_chart' in at.session_state['analysis_all_plots'],
        'Abundance bar chart' in at.session_state['analysis_statistics'],
    )


class TestAnalysisOutputsFollowQcOutput:

    def test_kept_while_qc_output_unchanged(self):
        at = _run_to_analysis('None (pre-normalized data)')
        _show_analysis(at, 'Pie')
        at.run()
        assert _bar_results(at) == (True, True)

    def test_dropped_when_cov_filter_changes_qc_output(self):
        """Regression: after the CoV filter removed 338 of 943 lipids, the
        bar chart's PDF figure and LSI statistics still came from all 943."""
        at = _run_to_analysis('None (pre-normalized data)')
        _show_analysis(at, 'Pie')
        at.number_input(key='bqc_cov_threshold').set_value(10).run()
        at.radio(key='bqc_filter_choice').set_value('Yes').run()
        assert not at.exception, at.exception
        assert _bar_results(at) == (False, False)
        # The page on screen was redrawn from the new data.
        assert any(k.startswith('pie') for k in at.session_state['analysis_all_plots'])

    def test_dropped_when_pca_removes_a_sample(self):
        at = _run_to_analysis('None (pre-normalized data)')
        _show_analysis(at, 'Pie')
        at.multiselect(key='pca_samples_remove').set_value(['s1']).run()
        assert not at.exception, at.exception
        assert _bar_results(at) == (False, False)

    def test_redrawn_page_comes_back(self):
        at = _run_to_analysis('None (pre-normalized data)')
        _show_analysis(at, 'Pie')
        at.multiselect(key='pca_samples_remove').set_value(['s1']).run()
        _show_analysis(at, 'Bar Chart')
        assert _bar_results(at) == (True, True)
        assert at.session_state['analysis_statistics'][
            'Abundance bar chart'
        ].summary.parameters['n_conditions'] == 3

    def test_dropped_when_sample_labels_switch(self):
        """The figures show sample names or labels; switching what is shown
        must not leave figures drawn with the other."""
        def name_samples(at):
            at.session_state['sample_names'] = {f's{i}': f'N{i}' for i in range(1, 13)}
            at.run()

        at = _run_to_analysis('None (pre-normalized data)', setup=name_samples)
        _show_analysis(at, 'Pie')
        at.run()
        assert _bar_results(at) == (True, True)
        at.sidebar.radio(key='_sample_label_mode_radio').set_value('standardized').run()
        assert not at.exception, at.exception
        assert _bar_results(at) == (False, False)
