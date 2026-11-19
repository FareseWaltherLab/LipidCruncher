"""
The sidebar's condition color scheme reaches every figure colored by
condition, and bars and boxes draw as a light fill outlined in the color.

Covers the palettes themselves, the tint/outline helpers, and each figure:
the Quality Check missing-values bars, box plot and PCA, the internal
standards bars, the class concentration bar chart (with a dot per sample)
and every lipidomic heatmap's condition strip.
"""

import re

import numpy as np
import pandas as pd
import pytest

from app.constants import (
    CONDITION_COLORS,
    CONDITION_PALETTES,
    DEFAULT_CONDITION_PALETTE,
)
from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import (
    OUTLINE_WIDTH,
    generate_condition_color_mapping,
    outlined_marker,
    tint,
)
from app.services.plotting.abundance_bar_chart import BarChartPlotterService
from app.services.plotting.box_plot import BoxPlotService
from app.services.plotting.pca import PCAPlotterService
from app.services.plotting.standards_plotter import StandardsPlotterService
from app.workflows.analysis import AnalysisWorkflow

SOFT = CONDITION_PALETTES['Soft']
SAMPLES = ['s1', 's2', 's3', 's4']
CONDITIONS = ['Control', 'Treatment']


@pytest.fixture
def experiment():
    return ExperimentConfig(
        n_conditions=2, conditions_list=CONDITIONS, number_of_samples_list=[2, 2],
    )


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        'LipidMolec': [f'{c}({i}:0)' for c in ('PC', 'PE') for i in range(3)],
        'ClassKey': ['PC'] * 3 + ['PE'] * 3,
    })
    for s in SAMPLES:
        frame[f'concentration[{s}]'] = rng.uniform(10, 100, len(frame))
    return frame


# ── Palettes and helpers ──────────────────────────────────────────────


class TestPalettes:
    def test_default_is_offered(self):
        assert DEFAULT_CONDITION_PALETTE in CONDITION_PALETTES

    def test_classic_keeps_the_original_colors(self):
        assert CONDITION_PALETTES['Classic'] == CONDITION_COLORS

    @pytest.mark.parametrize('name', list(CONDITION_PALETTES))
    def test_colors_are_distinct_hex(self, name):
        colors = CONDITION_PALETTES[name]
        assert len(colors) >= 8
        assert len(set(colors)) == len(colors)
        for color in colors:
            assert re.fullmatch(r'#[0-9a-f]{6}', color), color

    def test_mapping_uses_the_named_palette(self):
        assert generate_condition_color_mapping(CONDITIONS, 'Soft') == {
            'Control': SOFT[0], 'Treatment': SOFT[1],
        }

    def test_mapping_defaults_to_classic(self):
        assert generate_condition_color_mapping(CONDITIONS) == {
            'Control': CONDITION_COLORS[0], 'Treatment': CONDITION_COLORS[1],
        }

    def test_mapping_cycles_past_the_palette(self):
        conditions = [f'C{i}' for i in range(len(SOFT) + 1)]
        mapping = generate_condition_color_mapping(conditions, 'Soft')
        assert mapping[conditions[-1]] == SOFT[0]


class TestTint:
    def test_mixes_with_white(self):
        assert tint('#000000', 0.2) == '#cccccc'

    def test_white_stays_white(self):
        assert tint('#ffffff') == '#ffffff'

    def test_full_strength_is_the_color(self):
        assert tint('#2f5597', 1.0) == '#2f5597'

    def test_outlined_marker(self):
        marker = outlined_marker('#2f5597')
        assert marker == dict(
            color=tint('#2f5597'),
            line=dict(color='#2f5597', width=OUTLINE_WIDTH),
        )


# ── Quality Check ─────────────────────────────────────────────────────


class TestQualityCheckFigures:
    def test_missing_values_bars(self, experiment):
        fig = BoxPlotService.plot_missing_values(
            SAMPLES, [0, 10, 20, 30],
            experiment.conditions_list, experiment.individual_samples_list,
            palette='Soft',
        )
        bars = [t for t in fig.data if t.showlegend is False]
        assert [t.marker.line.color for t in bars] == [SOFT[0]] * 2 + [SOFT[1]] * 2
        assert [t.marker.color for t in bars] == [tint(SOFT[0])] * 2 + [tint(SOFT[1])] * 2
        legend = [t for t in fig.data if t.showlegend]
        assert [t.marker.line.color for t in legend] == [SOFT[0], SOFT[1]]

    def test_box_plot(self, df, experiment):
        mean_area_df = BoxPlotService.create_mean_area_df(df, SAMPLES)
        fig = BoxPlotService.plot_box_plot(
            mean_area_df, SAMPLES,
            experiment.conditions_list, experiment.individual_samples_list,
            palette='Soft',
        )
        boxes = [t for t in fig.data if t.showlegend is False]
        assert [t.line.color for t in boxes] == [SOFT[0]] * 2 + [SOFT[1]] * 2
        assert [t.fillcolor for t in boxes] == [tint(SOFT[0])] * 2 + [tint(SOFT[1])] * 2
        assert all(t.line.width == OUTLINE_WIDTH for t in boxes)

    def test_pca(self, df, experiment):
        fig, _ = PCAPlotterService.plot_pca(
            df, SAMPLES, experiment.extensive_conditions_list, palette='Soft',
        )
        markers = {t.name: t.marker.color for t in fig.data if t.mode == 'markers'}
        assert markers == {'Control': SOFT[0], 'Treatment': SOFT[1]}


# ── Internal standards ────────────────────────────────────────────────


class TestStandardsFigures:
    def _intsta(self, standards):
        frame = pd.DataFrame({
            'LipidMolec': standards, 'ClassKey': ['PC'] * len(standards),
        })
        for i, s in enumerate(SAMPLES):
            frame[f'intensity[{s}]'] = [100.0 * (i + 1)] * len(standards)
        return frame

    @pytest.mark.parametrize('standards', [
        ['PC(15:0)+D7'], ['PC(15:0)+D7', 'PC(17:0)+D7'],
    ])
    def test_bars_follow_the_palette(self, standards):
        figs = StandardsPlotterService.create_consistency_plots(
            self._intsta(standards), SAMPLES,
            sample_conditions=['Control', 'Control', 'Treatment', 'Treatment'],
            palette='Soft',
        )
        outlines = {t.name: t.marker.line.color for t in figs[0].data}
        assert outlines == {'Control': SOFT[0], 'Treatment': SOFT[1]}
        assert {t.marker.color for t in figs[0].data} == {tint(SOFT[0]), tint(SOFT[1])}


# ── Class concentration bar chart ─────────────────────────────────────


class TestClassBarChart:
    def _chart(self, df, experiment, scale='linear', palette='Soft'):
        return AnalysisWorkflow.run_bar_chart(
            df, experiment, CONDITIONS, ['PC', 'PE'], scale=scale, palette=palette,
        ).figure

    def test_bars_follow_the_palette(self, df, experiment):
        bars = [t for t in self._chart(df, experiment).data if t.type == 'bar']
        assert [t.marker.line.color for t in bars] == [SOFT[0], SOFT[1]]
        assert [t.marker.color for t in bars] == [tint(SOFT[0]), tint(SOFT[1])]

    def test_one_dot_per_sample_and_class(self, df, experiment):
        dots = [t for t in self._chart(df, experiment).data if t.type == 'scatter']
        assert [t.legendgroup for t in dots] == CONDITIONS
        assert [t.marker.color for t in dots] == [SOFT[0], SOFT[1]]
        # 2 classes x 2 samples per condition.
        assert [len(t.x) for t in dots] == [4, 4]

    def test_dots_are_the_sample_class_totals(self, df, experiment):
        dots = [t for t in self._chart(df, experiment).data if t.type == 'scatter']
        pc = df[df['ClassKey'] == 'PC']
        expected = [pc[f'concentration[{s}]'].sum() for s in ('s1', 's2')]
        assert list(dots[0].x[:2]) == pytest.approx(expected)

    def test_dots_sit_on_their_bar(self, df, experiment):
        fig = self._chart(df, experiment)
        bars = [t for t in fig.data if t.type == 'bar']
        dots = [t for t in fig.data if t.type == 'scatter']
        for bar, dot in zip(bars, dots):
            for class_index, centre in enumerate(bar.y):
                ys = dot.y[class_index * 2:class_index * 2 + 2]
                assert all(abs(y - centre) <= bar.width / 2 for y in ys)
                assert ys[0] != ys[1]  # spread, so equal values stay visible

    def test_log10_dots(self, df, experiment):
        dots = [
            t for t in self._chart(df, experiment, scale='log10').data
            if t.type == 'scatter'
        ]
        pc = df[df['ClassKey'] == 'PC']
        expected = [np.log10(pc[f'concentration[{s}]'].sum()) for s in ('s1', 's2')]
        assert list(dots[0].x[:2]) == pytest.approx(expected)

    def test_log10_dots_replace_zeros_like_the_mean(self, experiment):
        """A zero total is drawn where the log10 mean counted it: at a tenth
        of the smallest positive total."""
        frame = pd.DataFrame({
            'LipidMolec': ['PC(1:0)'], 'ClassKey': ['PC'],
            'concentration[s1]': [0.0], 'concentration[s2]': [100.0],
            'concentration[s3]': [10.0], 'concentration[s4]': [10.0],
        })
        bar_data = BarChartPlotterService.create_mean_std_data(
            frame, experiment, CONDITIONS, ['PC'],
        )
        control = bar_data.sample_values[bar_data.sample_values['Condition'] == 'Control']
        assert control['log10_value'].tolist() == pytest.approx([1.0, 2.0])
        assert bar_data.abundance_df['log10_mean_AUC_Control'][0] == pytest.approx(1.5)


# ── Lipidomic heatmap ─────────────────────────────────────────────────


class TestHeatmapStrip:
    @pytest.mark.parametrize('heatmap_type', [
        'regular', 'clustered', 'class_grouped', 'class_aggregated',
    ])
    def test_strip_follows_the_palette(self, df, experiment, heatmap_type):
        result = AnalysisWorkflow.run_heatmap(
            df, experiment, CONDITIONS, ['PC', 'PE'],
            heatmap_type=heatmap_type, n_clusters=2, palette='Soft',
        )
        rects = [s for s in result.figure.layout.shapes if s.type == 'rect']
        assert [r.fillcolor for r in rects] == [SOFT[0], SOFT[1]]
