"""Unit tests for PlotterServiceProtocol conformance.

Verifies that plotting services implement the expected interface pattern
defined in app/services/plotting/base.py.
"""
import pytest

from app.services.plotting.base import PlotterServiceProtocol
from app.services.plotting.abundance_bar_chart import BarChartPlotterService
from app.services.plotting.abundance_pie_chart import PieChartPlotterService
from app.services.plotting.volcano_plot import VolcanoPlotterService


# =============================================================================
# Protocol Conformance
# =============================================================================


class TestProtocolConformance:
    """Verify plotters with generate_color_mapping satisfy the protocol."""

    @pytest.mark.parametrize("plotter_class", [
        BarChartPlotterService,
        PieChartPlotterService,
        VolcanoPlotterService,
    ])
    def test_has_generate_color_mapping(self, plotter_class):
        """Plotter should have a generate_color_mapping static method."""
        assert hasattr(plotter_class, 'generate_color_mapping')
        assert callable(plotter_class.generate_color_mapping)

    @pytest.mark.parametrize("plotter_class", [
        BarChartPlotterService,
        PieChartPlotterService,
        VolcanoPlotterService,
    ])
    def test_color_mapping_returns_dict(self, plotter_class):
        """generate_color_mapping should return a dict of str → str."""
        result = plotter_class.generate_color_mapping(['A', 'B', 'C'])
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result)
        assert all(isinstance(v, str) for v in result.values())

    @pytest.mark.parametrize("plotter_class", [
        BarChartPlotterService,
        PieChartPlotterService,
        VolcanoPlotterService,
    ])
    def test_color_mapping_covers_all_items(self, plotter_class):
        """Every input item should appear as a key in the mapping."""
        items = ['Control', 'Treatment', 'Vehicle']
        result = plotter_class.generate_color_mapping(items)
        for item in items:
            assert item in result

    @pytest.mark.parametrize("plotter_class", [
        BarChartPlotterService,
        PieChartPlotterService,
        VolcanoPlotterService,
    ])
    def test_color_mapping_empty_input(self, plotter_class):
        """Empty input should return empty dict."""
        result = plotter_class.generate_color_mapping([])
        assert result == {}

    @pytest.mark.parametrize("plotter_class", [
        BarChartPlotterService,
        PieChartPlotterService,
        VolcanoPlotterService,
    ])
    def test_color_mapping_single_item(self, plotter_class):
        """Single item should return a dict with one entry."""
        result = plotter_class.generate_color_mapping(['A'])
        assert len(result) == 1
        assert 'A' in result


# =============================================================================
# Protocol Runtime Check
# =============================================================================


class TestRuntimeCheckable:
    """Verify the protocol is @runtime_checkable and works with isinstance."""

    @pytest.mark.parametrize("plotter_class", [
        BarChartPlotterService,
        PieChartPlotterService,
        VolcanoPlotterService,
    ])
    def test_isinstance_check(self, plotter_class):
        """Plotters with generate_color_mapping should satisfy the protocol."""
        assert isinstance(plotter_class, PlotterServiceProtocol)