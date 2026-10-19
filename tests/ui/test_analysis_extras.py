"""Analysis-page features without UI tests until now: the pathway layout's
save/load and edge editing, the volcano's label adjustment, and the
detailed-statistics tables (with post-hoc comparisons)."""

import json

import pytest

from tests.ui.conftest import FakeUpload


def _open(at, word):
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if word in o][0]).run()
    assert not at.exception
    return at


class TestPathwayLayoutFile:

    def test_saved_layout_loads_back(self, analysis_generic_app, captured_downloads, fake_uploads):
        at = _open(analysis_generic_app, 'Pathway')
        at.text_input(key='pathway_add_node_name').set_value('MyLipid')
        at.number_input(key='pathway_add_node_x').set_value(1.0)
        at.number_input(key='pathway_add_node_y').set_value(2.0)
        at.button(key='pathway_add_node_btn').click().run()
        assert 'MyLipid' in at.session_state['pathway_class_selector']
        layout = list(at.session_state['pathway_class_selector'])
        saved = captured_downloads['pathway_download_config']

        at.button(key='pathway_start_default').click().run()
        assert 'MyLipid' not in at.session_state['pathway_class_selector']

        fake_uploads['Upload Configuration'] = FakeUpload('layout.json', saved.encode())
        at.run()
        at.run()
        assert not at.exception
        assert at.session_state['pathway_class_selector'] == layout
        assert tuple(at.session_state['analysis_pathway_custom_nodes']['MyLipid']) == (1.0, 2.0)

    @pytest.mark.parametrize('content, message', [
        (b'[1, 2]', 'expected a JSON object'),
        (b'{"custom_nodes": {}}', "must contain 'active_classes'"),
        (b'{not json', 'Could not parse JSON'),
    ])
    def test_bad_layout_files_are_reported(self, analysis_generic_app, fake_uploads, content, message):
        at = _open(analysis_generic_app, 'Pathway')
        fake_uploads['Upload Configuration'] = FakeUpload('layout.json', content)
        at.run()
        assert not at.exception
        assert any(message in e.value for e in at.error)


class TestPathwayEdges:

    def test_add_then_remove_an_edge(self, analysis_generic_app):
        at = _open(analysis_generic_app, 'Pathway')
        at.selectbox(key='pathway_add_edge_source').set_value('PC').run()
        at.selectbox(key='pathway_add_edge_target').set_value('TG').run()
        at.button(key='pathway_add_edge_btn').click().run()
        assert ('PC', 'TG') in [tuple(e) for e in at.session_state['analysis_pathway_added_edges']]

        remove = at.selectbox(key='pathway_edge_to_remove_select')
        remove.set_value([o for o in remove.options if 'PC' in o and 'TG' in o][0]).run()
        at.button(key='pathway_remove_edge_btn').click().run()
        assert not at.exception
        assert ('PC', 'TG') not in [tuple(e) for e in at.session_state['analysis_pathway_added_edges']]

    def test_removing_a_default_edge_records_it(self, analysis_generic_app):
        at = _open(analysis_generic_app, 'Pathway')
        remove = at.selectbox(key='pathway_edge_to_remove_select')
        edge = remove.options[0]
        remove.set_value(edge).run()
        at.button(key='pathway_remove_edge_btn').click().run()
        assert not at.exception
        assert tuple(edge.split(' — ')) in [
            tuple(e) for e in at.session_state['analysis_pathway_removed_edges']
        ]


def _annotation(figure, text):
    return [a for a in figure.layout.annotations if a.text == text][0]


class TestVolcanoLabelAdjustment:

    def test_offset_moves_the_label(self, analysis_generic_app):
        at = _open(analysis_generic_app, 'Volcano')
        at.number_input(key='volcano_top_n').set_value(1).run()
        at.checkbox(key='volcano_enable_adjust').check().run()
        lipid = [n.key for n in at.number_input if n.key.startswith('volcano_label_x_')][0]
        lipid = lipid[len('volcano_label_x_'):]
        before = _annotation(at.session_state['analysis_all_plots']['volcano'], lipid)

        at.number_input(key=f'volcano_label_x_{lipid}').set_value(2.0).run()
        assert not at.exception
        after = _annotation(at.session_state['analysis_all_plots']['volcano'], lipid)
        assert (after.ax, after.x) != (before.ax, before.x)


class TestDetailedStatistics:

    def test_bar_chart_table(self, analysis_generic_app):
        at = analysis_generic_app
        at.checkbox(key='bar_detailed_stats').check().run()
        assert not at.exception
        tables = [d.value for d in at.dataframe if 'p-value' in d.value.columns]
        assert tables and set(tables[0]['Class']) == {'PC', 'PE'}

    def test_post_hoc_comparisons_for_three_conditions(self):
        """With a real difference between conditions, the omnibus test is
        significant and the pairwise comparisons are listed."""
        from streamlit.testing.v1 import AppTest
        from app.models.experiment import ExperimentConfig
        from tests.ui.conftest import (
            DEFAULT_TIMEOUT, analysis_module_script, make_analysis_dataframe,
        )

        at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_df'] = make_analysis_dataframe(
            n_lipids=20, n_samples=9, condition_effect=True,
        )
        at.session_state['_test_experiment'] = ExperimentConfig(
            conditions_list=['Control', 'Treatment', 'Vehicle'],
            number_of_samples_list=[3, 3, 3],
        )
        at.run()
        at.checkbox(key='bar_detailed_stats').check().run()
        assert not at.exception
        tables = [d.value for d in at.dataframe if 'Comparison' in d.value.columns]
        assert tables, "no post-hoc table"
        assert set(tables[0]['Comparison']) >= {'Control vs Treatment'}

    def test_volcano_table(self, analysis_generic_app):
        at = _open(analysis_generic_app, 'Volcano')
        at.checkbox(key='volcano_detailed_stats').check().run()
        assert not at.exception
        tables = [d.value for d in at.dataframe if 'Lipid' in d.value.columns]
        assert tables and 'p-value' in tables[0].columns


def test_layout_json_is_what_the_download_offers(analysis_generic_app, captured_downloads):
    at = _open(analysis_generic_app, 'Pathway')
    config = json.loads(captured_downloads['pathway_download_config'])
    assert config['active_classes'] == at.session_state['pathway_class_selector']
