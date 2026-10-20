"""
The multiple-testing correction counts only what was tested: the selected
classes (or their species). The pages say so when classes are left out,
since adjusted p-values then change with the class selection (M2).
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import DEFAULT_TIMEOUT, analysis_module_script, make_analysis_dataframe

NOTE = 'covers only the'


def _app(word):
    from app.models.experiment import ExperimentConfig
    at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(
        n_lipids=20, n_samples=6, detailed_fa=True,
    )
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2, conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.run()
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if word in o][0]).run()
    assert not at.exception, at.exception
    return at


def _notes(at):
    return [w.value for w in at.warning if NOTE in w.value]


def _fdr(prefix):
    """Manual mode with an FDR correction (Auto skips it for few classes)."""
    def setup(at):
        at.radio(key=f'{prefix}_stats_mode').set_value('Manual').run()
        at.selectbox(key=f'{prefix}_correction').set_value('fdr_bh').run()
    return setup


@pytest.mark.parametrize('word, classes_key, setup', [
    ('Bar Chart', 'bar_classes', _fdr('bar')),
    ('Saturation', 'sat_classes', _fdr('sat')),
    ('Volcano', 'volcano_classes',
     lambda at: at.selectbox(key='volcano_correction').set_value('fdr_bh').run()),
])
class TestCorrectionScopeNote:

    def _configure(self, at, setup):
        setup(at)
        assert not at.exception, at.exception

    def test_shown_when_classes_are_left_out(self, word, classes_key, setup):
        at = _app(word)
        self._configure(at, setup)
        at.multiselect(key=classes_key).set_value(['PC']).run()
        assert not at.exception, at.exception
        assert len(_notes(at)) == 1

    def test_not_shown_with_every_class(self, word, classes_key, setup):
        at = _app(word)
        self._configure(at, setup)
        assert _notes(at) == []


def test_not_shown_without_a_correction():
    at = _app('Volcano')   # the volcano defaults to uncorrected
    at.multiselect(key='volcano_classes').set_value(['PC']).run()
    assert _notes(at) == []
