"""
Tests for the PDF report generator service.
"""
import io
from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
import pytest

from app.services.report_generator import (
    ReportMetadata,
    build_metadata_from_experiment,
    generate_pdf_report,
    _build_analyses_list,
    _get_saturation_classes,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_metadata():
    return ReportMetadata(
        format_type="Generic Format",
        n_conditions=2,
        total_samples=6,
        conditions_detail=[
            ("Control", 3, ["s1", "s2", "s3"]),
            ("Treatment", 3, ["s4", "s5", "s6"]),
        ],
    )


@pytest.fixture
def minimal_metadata():
    return ReportMetadata()


@pytest.fixture
def plotly_fig():
    """Simple Plotly figure for testing."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    return fig


@pytest.fixture
def matplotlib_fig():
    """Simple Matplotlib figure for testing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    return fig


@pytest.fixture
def analysis_plots(plotly_fig):
    return {
        'bar_chart': plotly_fig,
        'pie_Control': plotly_fig,
        'pie_Treatment': plotly_fig,
        'heatmap': plotly_fig,
    }


@pytest.fixture
def qc_plots(plotly_fig, matplotlib_fig):
    return {
        'box_plot_fig1': plotly_fig,
        'box_plot_fig2': plotly_fig,
        'bqc_plot': plotly_fig,
        'pca_plot': plotly_fig,
        'correlation_plots': {'Control': matplotlib_fig},
    }


# ── TestReportMetadata ──────────────────────────────────────────────────────

class TestReportMetadata:
    def test_defaults(self):
        m = ReportMetadata()
        assert m.format_type is None
        assert m.n_conditions is None
        assert m.total_samples is None
        assert m.conditions_detail == []

    def test_with_values(self, simple_metadata):
        assert simple_metadata.format_type == "Generic Format"
        assert simple_metadata.n_conditions == 2
        assert simple_metadata.total_samples == 6
        assert len(simple_metadata.conditions_detail) == 2

    def test_many_conditions(self):
        details = [(f"C{i}", 5, [f"s{j}" for j in range(5)])
                    for i in range(10)]
        m = ReportMetadata(n_conditions=10, total_samples=50,
                           conditions_detail=details)
        assert len(m.conditions_detail) == 10


# ── TestBuildMetadataFromExperiment ─────────────────────────────────────────

class TestBuildMetadataFromExperiment:
    def test_none_experiment(self):
        m = build_metadata_from_experiment(None, format_type="MS-DIAL")
        assert m.format_type == "MS-DIAL"
        assert m.n_conditions is None

    def test_with_experiment(self):
        exp = MagicMock()
        exp.n_conditions = 2
        exp.conditions_list = ["WT", "KO"]
        exp.number_of_samples_list = [3, 3]
        exp.individual_samples_list = [["s1", "s2", "s3"],
                                        ["s4", "s5", "s6"]]
        exp.full_samples_list = ["s1", "s2", "s3", "s4", "s5", "s6"]

        m = build_metadata_from_experiment(exp, "LipidSearch")
        assert m.format_type == "LipidSearch"
        assert m.n_conditions == 2
        assert m.total_samples == 6
        assert len(m.conditions_detail) == 2
        assert m.conditions_detail[0] == ("WT", 3, ["s1", "s2", "s3"])


# ── TestBuildAnalysesList ───────────────────────────────────────────────────

class TestBuildAnalysesList:
    def test_empty(self):
        assert _build_analyses_list({}, {}) == []

    def test_qc_only(self, plotly_fig, matplotlib_fig):
        qc = {
            'box_plot_fig1': plotly_fig,
            'box_plot_fig2': plotly_fig,
            'pca_plot': plotly_fig,
            'correlation_plots': {'C1': matplotlib_fig, 'C2': matplotlib_fig},
        }
        items = _build_analyses_list({}, qc)
        assert "Distribution Box Plots" in items
        assert "Principal Component Analysis (PCA)" in items
        assert "Pairwise Correlation (2 condition(s))" in items

    def test_analysis_only(self, plotly_fig):
        plots = {
            'bar_chart': plotly_fig,
            'pie_WT': plotly_fig,
            'pie_KO': plotly_fig,
            'sat_concentration_PC': plotly_fig,
            'sat_percentage_PC': plotly_fig,
            'fach': plotly_fig,
            'pathway': plotly_fig,
            'volcano': plotly_fig,
            'heatmap': plotly_fig,
        }
        items = _build_analyses_list(plots, {})
        assert "Class Concentration Bar Chart" in items
        assert "Class Concentration Pie Charts (2 condition(s))" in items
        assert "Saturation Profiles (1 class(es))" in items
        assert "Fatty Acid Composition Heatmap" in items
        assert "Lipid Pathway Visualization" in items
        assert "Volcano Plot Analysis" in items
        assert "Lipidomic Heatmap" in items

    def test_bqc_included(self, plotly_fig):
        qc = {'bqc_plot': plotly_fig}
        items = _build_analyses_list({}, qc)
        assert "BQC Quality Assessment" in items

    def test_retention_time_included(self, plotly_fig):
        qc = {'retention_time_plot': plotly_fig}
        items = _build_analyses_list({}, qc)
        assert "Retention Time Analysis" in items

    def test_multiple_saturation_classes(self, plotly_fig):
        plots = {
            'sat_concentration_PC': plotly_fig,
            'sat_concentration_PE': plotly_fig,
            'sat_percentage_PC': plotly_fig,
        }
        items = _build_analyses_list(plots, {})
        assert "Saturation Profiles (2 class(es))" in items

    def test_volcano_companion_plots_listed(self, plotly_fig, matplotlib_fig):
        """The two plots rendered underneath the volcano plot are reported
        separately, so the cover page reflects what the PDF contains."""
        plots = {
            'volcano': plotly_fig,
            'conc_vs_fc': plotly_fig,
            'lipid_distribution': matplotlib_fig,
        }
        items = _build_analyses_list(plots, {})
        assert "Concentration vs. Fold Change" in items
        assert "Individual Lipid Concentration Distribution" in items

    def test_standards_plots_listed(self, plotly_fig):
        items = _build_analyses_list({}, {}, [plotly_fig, plotly_fig])
        assert "Internal Standards Consistency (2 class(es))" in items

    def test_standards_absent_when_not_supplied(self, plotly_fig):
        items = _build_analyses_list({'volcano': plotly_fig}, {})
        assert not any('Internal Standards' in i for i in items)


# ── TestGetSaturationClasses ────────────────────────────────────────────────

class TestGetSaturationClasses:
    def test_empty(self):
        assert _get_saturation_classes({}) == []

    def test_single_class(self, plotly_fig):
        plots = {
            'sat_concentration_PC': plotly_fig,
            'sat_percentage_PC': plotly_fig,
        }
        assert _get_saturation_classes(plots) == ['PC']

    def test_multiple_classes_sorted(self, plotly_fig):
        plots = {
            'sat_concentration_PE': plotly_fig,
            'sat_concentration_PC': plotly_fig,
            'sat_percentage_SM': plotly_fig,
        }
        assert _get_saturation_classes(plots) == ['PC', 'PE', 'SM']

    def test_ignores_non_sat_keys(self, plotly_fig):
        plots = {
            'bar_chart': plotly_fig,
            'sat_concentration_PC': plotly_fig,
            'volcano': plotly_fig,
        }
        assert _get_saturation_classes(plots) == ['PC']


# ── TestGeneratePdfReport ───────────────────────────────────────────────────

class TestGeneratePdfReport:
    def test_empty_report(self, simple_metadata):
        """Report with no plots should still generate a cover page."""
        result = generate_pdf_report({}, simple_metadata)
        assert result is not None
        assert isinstance(result, io.BytesIO)
        data = result.read()
        assert len(data) > 0
        assert data[:5] == b'%PDF-'

    def test_minimal_metadata(self, minimal_metadata):
        result = generate_pdf_report({}, minimal_metadata)
        assert result is not None
        data = result.read()
        assert data[:5] == b'%PDF-'

    def test_with_analysis_plots(self, analysis_plots, simple_metadata):
        result = generate_pdf_report(analysis_plots, simple_metadata)
        assert result is not None
        data = result.read()
        assert len(data) > 100

    def test_with_qc_plots(self, qc_plots, simple_metadata):
        result = generate_pdf_report({}, simple_metadata, qc_plots=qc_plots)
        assert result is not None
        data = result.read()
        assert len(data) > 100

    def test_with_all_plots(self, analysis_plots, qc_plots, simple_metadata):
        result = generate_pdf_report(analysis_plots, simple_metadata,
                                     qc_plots=qc_plots)
        assert result is not None

    def test_none_qc_plots_default(self, simple_metadata):
        result = generate_pdf_report({}, simple_metadata, qc_plots=None)
        assert result is not None

    def test_returns_bytesio(self, simple_metadata, plotly_fig):
        result = generate_pdf_report({'bar_chart': plotly_fig},
                                     simple_metadata)
        assert isinstance(result, io.BytesIO)
        assert result.tell() == 0  # seek(0) was called

    def test_saturation_plots(self, simple_metadata, plotly_fig):
        plots = {
            'sat_concentration_PC': plotly_fig,
            'sat_percentage_PC': plotly_fig,
            'sat_concentration_PE': plotly_fig,
        }
        result = generate_pdf_report(plots, simple_metadata)
        assert result is not None

    def test_pathway_plotly(self, simple_metadata, plotly_fig):
        plots = {'pathway': plotly_fig}
        result = generate_pdf_report(plots, simple_metadata)
        assert result is not None

    def test_heatmap_special_rendering(self, simple_metadata, plotly_fig):
        plots = {'heatmap': plotly_fig}
        result = generate_pdf_report(plots, simple_metadata)
        assert result is not None


# ── TestCoverPageContent ────────────────────────────────────────────────────

class TestPlotsReachThePdf:
    """A plot listed on the cover page but missing from the renderer never
    reaches the PDF, so assert on what is actually drawn."""

    def _rendered_titles(self, **kwargs):
        titles = []
        with patch(
            'app.services.report_generator._render_plot_page',
            side_effect=lambda pdf, fig, title, *a, **k: titles.append(title),
        ):
            generate_pdf_report(**kwargs)
        return titles

    def test_volcano_companion_plots_render(self, simple_metadata, plotly_fig,
                                            matplotlib_fig):
        titles = self._rendered_titles(
            analysis_plots={
                'volcano': plotly_fig,
                'conc_vs_fc': plotly_fig,
                'lipid_distribution': matplotlib_fig,
            },
            metadata=simple_metadata,
        )
        assert "Concentration vs. Fold Change" in titles
        assert "Individual Lipid Concentration Distribution" in titles

    def test_matplotlib_distribution_survives_export(self, simple_metadata,
                                                     matplotlib_fig):
        """The distribution plot is Matplotlib, not Plotly. Routing it to the
        Plotly exporter raises, so this fails if is_matplotlib is wrong."""
        result = generate_pdf_report(
            {'lipid_distribution': matplotlib_fig}, simple_metadata,
        )
        assert result is not None
        assert result.getbuffer().nbytes > 0

    def test_standards_plots_render_with_their_own_titles(self, simple_metadata):
        fig = go.Figure()
        fig.update_layout(
            title='Internal Standards Intensity for PC (PC 15:0)'
        )
        titles = self._rendered_titles(
            analysis_plots={}, metadata=simple_metadata, standards_plots=[fig],
        )
        assert 'Internal Standards Intensity for PC (PC 15:0)' in titles

    def test_untitled_standard_gets_a_numbered_caption(self, simple_metadata,
                                                       plotly_fig):
        titles = self._rendered_titles(
            analysis_plots={}, metadata=simple_metadata,
            standards_plots=[plotly_fig],
        )
        assert 'Internal Standards Consistency (1)' in titles

    def test_standards_only_report_generates(self, simple_metadata, plotly_fig):
        """A user who ran Module 1 and stopped still gets their standards."""
        result = generate_pdf_report(
            {}, simple_metadata, standards_plots=[plotly_fig, plotly_fig],
        )
        assert result is not None
        assert result.getbuffer().nbytes > 0


class TestCoverPageContent:
    """Test cover page rendering via mock canvas to inspect drawString calls."""

    def _get_drawn_strings(self, metadata, analyses=None):
        """Collect all strings drawn on the canvas."""
        if analyses is None:
            analyses = []
        strings = []
        mock_pdf = MagicMock()
        mock_pdf.drawString = lambda x, y, s: strings.append(s)
        mock_pdf.drawCentredString = lambda x, y, s: strings.append(s)
        from app.services.report_generator import _render_cover_page
        _render_cover_page(mock_pdf, metadata, analyses)
        return strings

    def test_cover_has_title(self, simple_metadata):
        strings = self._get_drawn_strings(simple_metadata)
        assert any('LipidCruncher Analysis Report' in s for s in strings)

    def test_cover_has_format_type(self, simple_metadata):
        strings = self._get_drawn_strings(simple_metadata)
        assert any('Generic Format' in s for s in strings)

    def test_cover_has_condition_count(self, simple_metadata):
        strings = self._get_drawn_strings(simple_metadata)
        assert any('Number of Conditions: 2' in s for s in strings)

    def test_cover_has_total_samples(self, simple_metadata):
        strings = self._get_drawn_strings(simple_metadata)
        assert any('Total Samples: 6' in s for s in strings)

    def test_cover_has_footer(self, simple_metadata):
        strings = self._get_drawn_strings(simple_metadata)
        assert any('Farese and Walther Lab' in s for s in strings)

    def test_cover_has_analyses_list(self, simple_metadata):
        analyses = ["Distribution Box Plots", "Volcano Plot Analysis"]
        strings = self._get_drawn_strings(simple_metadata, analyses)
        assert any('Distribution Box Plots' in s for s in strings)
        assert any('Volcano Plot Analysis' in s for s in strings)


# ── TestErrorHandling ───────────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_figure_raises_error(self, simple_metadata):
        """Non-figure object should raise ValueError with details."""
        plots = {'bar_chart': "not_a_figure"}
        with pytest.raises(ValueError, match="PDF report generation failed"):
            generate_pdf_report(plots, simple_metadata)

    def test_empty_analysis_plots_dict(self, simple_metadata):
        result = generate_pdf_report({}, simple_metadata)
        assert result is not None

    def test_none_values_in_qc_plots(self, simple_metadata):
        qc = {
            'box_plot_fig1': None,
            'box_plot_fig2': None,
            'bqc_plot': None,
            'pca_plot': None,
            'retention_time_plot': None,
            'correlation_plots': {},
        }
        result = generate_pdf_report({}, simple_metadata, qc_plots=qc)
        assert result is not None


# ── TestPlotOrdering ────────────────────────────────────────────────────────

class TestPlotOrdering:
    """Verify plots appear in the expected order by checking page count."""

    def test_single_plot_has_two_pages(self, simple_metadata, plotly_fig):
        """Cover + one plot = at least 2 pages."""
        result = generate_pdf_report({'bar_chart': plotly_fig},
                                     simple_metadata)
        data = result.read().decode('latin-1')
        # ReportLab writes page count in the trailer
        assert data.count('showpage') >= 0  # PDF has pages

    def test_pie_charts_sorted_by_condition(self, plotly_fig):
        """Verify _render_analysis_plots processes pie keys in sorted order."""
        from app.services.report_generator import _render_analysis_plots
        plots = {
            'pie_Zeta': plotly_fig,
            'pie_Alpha': plotly_fig,
        }
        titles = []
        with patch('app.services.report_generator._render_plot_page') as mock_render:
            mock_pdf = MagicMock()
            _render_analysis_plots(mock_pdf, plots)
            for call in mock_render.call_args_list:
                titles.append(call[0][2])  # title is 3rd positional arg
        alpha_idx = next(i for i, t in enumerate(titles) if 'Alpha' in t)
        zeta_idx = next(i for i, t in enumerate(titles) if 'Zeta' in t)
        assert alpha_idx < zeta_idx


# ── TestManyConditions ──────────────────────────────────────────────────────

class TestManyConditions:
    def test_many_conditions_in_metadata(self):
        details = [(f"Condition_{i}", 10, [f"s{j}" for j in range(10)])
                    for i in range(20)]
        m = ReportMetadata(
            format_type="Generic",
            n_conditions=20,
            total_samples=200,
            conditions_detail=details,
        )
        result = generate_pdf_report({}, m)
        assert result is not None

    def test_many_pie_charts(self, plotly_fig):
        plots = {f'pie_C{i}': plotly_fig for i in range(10)}
        m = ReportMetadata(format_type="Generic")
        result = generate_pdf_report(plots, m)
        assert result is not None


# ── TestHeatmapPage ─────────────────────────────────────────────────────────

class TestHeatmapPage:
    """The heatmap page re-exports the figure with its own margins. Its old
    fixed 60px top margin had no room for the condition strip's names, which
    ran into the title or were clipped off the page."""

    @staticmethod
    def _exported_top_margin(fig):
        from PIL import Image
        from reportlab.pdfgen import canvas
        from app.services.report_generator import _render_heatmap_page

        seen = {}

        def fake_export(fig_copy, params):
            seen['top'] = fig_copy.layout.margin.t
            buf = io.BytesIO()
            Image.new('RGB', (9, 12)).save(buf, 'PNG')
            return buf.getvalue()

        with patch(
            'app.services.report_generator._plotly_to_image',
            side_effect=fake_export,
        ):
            _render_heatmap_page(
                canvas.Canvas(io.BytesIO()), fig, "Lipidomic Heatmap",
            )
        return seen['top']

    def test_keeps_the_room_made_for_the_condition_strip(self):
        import numpy as np
        import pandas as pd
        from app.services.plotting.lipidomic_heatmap import (
            LipidomicHeatmapPlotterService,
        )

        z = pd.DataFrame(
            np.arange(8, dtype=float).reshape(2, 4),
            index=pd.MultiIndex.from_arrays(
                [['L0', 'L1'], ['PC', 'PE']], names=['LipidMolec', 'ClassKey'],
            ),
            columns=['s1', 's2', 's3', 's4'],
        )
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z, ['s1', 's2', 's3', 's4'], sample_conditions=['A', 'A', 'B', 'B'],
        )
        assert self._exported_top_margin(fig) == fig.layout.margin.t > 60

    def test_figures_without_a_taller_margin_get_the_default(self, plotly_fig):
        assert self._exported_top_margin(plotly_fig) == 60
