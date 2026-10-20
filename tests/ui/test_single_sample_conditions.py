"""
Mean-only analyses accept a condition with a single sample (D9).

Pie charts and heatmaps already offered such conditions; chain length,
FACH and pathway used the statistics pages' selector, which needs two or
more samples, and hid them without saying so. The statistics pages (bar
chart, saturation, volcano) still need replicates.
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import DEFAULT_TIMEOUT, analysis_module_script, make_analysis_dataframe


@pytest.fixture
def single_sample_app():
    from app.models.experiment import ExperimentConfig
    at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(
        n_lipids=20, n_samples=7, detailed_fa=True,
    )
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=3, conditions_list=['Control', 'Treatment', 'Pooled'],
        number_of_samples_list=[3, 3, 1],
    )
    return at.run()


def _show(at, word):
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if word in o][0]).run()
    assert not at.exception, at.exception
    return at


class TestOffered:

    def test_chain_length(self, single_sample_app):
        at = _show(single_sample_app, 'Chain Length')
        assert 'Pooled' in at.multiselect(key='clen_conditions').options
        at.multiselect(key='clen_conditions').set_value(['Pooled']).run()
        assert not at.exception, at.exception
        assert 'chain_length' in at.session_state['analysis_all_plots']

    def test_fach(self, single_sample_app):
        at = _show(single_sample_app, 'Fatty Acid Composition')
        assert 'Pooled' in at.multiselect(key='fach_conditions').options
        at.multiselect(key='fach_conditions').set_value(['Control', 'Pooled']).run()
        assert not at.exception, at.exception

    def test_pathway(self, single_sample_app):
        at = _show(single_sample_app, 'Pathway')
        assert 'Pooled' in at.selectbox(key='pathway_control').options
        at.selectbox(key='pathway_control').set_value('Pooled').run()
        assert not at.exception, at.exception
        assert 'pathway' in at.session_state['analysis_all_plots']


class TestStatisticsPagesStillNeedReplicates:

    @pytest.mark.parametrize('word, key', [
        ('Bar Chart', 'bar_conditions'),
        ('Saturation', 'sat_conditions'),
    ])
    def test_not_offered(self, single_sample_app, word, key):
        at = _show(single_sample_app, word)
        assert 'Pooled' not in at.multiselect(key=key).options

    def test_volcano(self, single_sample_app):
        at = _show(single_sample_app, 'Volcano')
        assert 'Pooled' not in at.selectbox(key='volcano_control').options
