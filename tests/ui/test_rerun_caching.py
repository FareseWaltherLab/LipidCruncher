"""
A rerun with unchanged inputs does not redo heavy work (M17).

Every widget interaction reruns the whole page, so work done in the UI
without the adapter's cache is repeated each time: the zero filter, the
internal-standards plots, an uploaded standards table, the saturation CSV
and the volcano's distribution plot did that.
"""

import pytest
import streamlit

from app.services.plotting.saturation_plot import SaturationPlotterService
from app.services.plotting.standards_plotter import StandardsPlotterService
from app.services.plotting.volcano_plot import VolcanoPlotterService
from app.services.standards import StandardsService
from app.services.zero_filtering import ZeroFilteringService
from tests.ui.conftest import FakeUpload


@pytest.fixture
def count(monkeypatch):
    """count(cls, name) -> list that grows by one per call; caches cleared."""
    streamlit.cache_data.clear()

    def _count(cls, name):
        calls = []
        original = getattr(cls, name)
        monkeypatch.setattr(cls, name, staticmethod(
            lambda *a, **k: calls.append(1) or original(*a, **k)
        ))
        return calls
    return _count


def _rerun_adds_no_calls(at, calls):
    assert calls, "the work did not run at all"
    before = len(calls)
    at.run()
    assert not at.exception, at.exception
    assert len(calls) == before


def _show(at, word):
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if word in o][0]).run()
    assert not at.exception, at.exception


def test_zero_filter(zero_filter_generic_app, count):
    calls = count(ZeroFilteringService, 'filter_zeros')
    zero_filter_generic_app.run()
    _rerun_adds_no_calls(zero_filter_generic_app, calls)


def test_standards_plots(intsta_with_standards_app, count):
    calls = count(StandardsPlotterService, 'create_consistency_plots')
    intsta_with_standards_app.run()
    _rerun_adds_no_calls(intsta_with_standards_app, calls)


def test_uploaded_standards_table(intsta_with_standards_app, fake_uploads, count):
    calls = count(StandardsService, 'process_standards_file')
    upload = FakeUpload('standards.csv', (
        'LipidMolec,ClassKey,s1,s2,s3,s4,s5,s6\n'
        'PC 15:0_18:1(d7),PC,100,110,120,130,140,150\n'
    ).encode())
    fake_uploads['Upload standards CSV'] = upload
    at = intsta_with_standards_app
    at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
    upload.seek(0)   # Streamlit hands each run a fresh file object
    _rerun_adds_no_calls(at, calls)


def test_saturation_csv(analysis_detailed_fa_app, count):
    calls = count(SaturationPlotterService, 'calculate_sfa_mufa_pufa')
    _show(analysis_detailed_fa_app, 'Saturation')
    _rerun_adds_no_calls(analysis_detailed_fa_app, calls)


def test_volcano_distribution_plot(analysis_generic_app, count):
    calls = count(VolcanoPlotterService, 'create_distribution_plot')
    _show(analysis_generic_app, 'Volcano')
    _rerun_adds_no_calls(analysis_generic_app, calls)
