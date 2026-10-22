"""Checks that an AI answer table still matches the data it came from
(services/ai_chat/output_checks.py), for tables of any layout."""
from pathlib import Path

import pandas as pd
import pytest

from app.services.ai_chat.output_checks import (
    check_direction_colors,
    check_lipid_table,
    computed_values,
    numbers_not_computed,
)

_KO = ['KO-1', 'KO-2', 'KO-3']


@pytest.fixture(scope='module')
def studio_df():
    path = Path(__file__).parents[1] / 'fixtures' / 'ai_studio_test_dataset.csv'
    return pd.read_csv(path)


def _misaligned(df):
    """Names with a fresh index next to columns with the filtered one."""
    kept = df[df['QualityScore'] > 70]
    table = pd.DataFrame({'feature': kept['feature'].values})
    for col in _KO:
        table[col] = kept[col]
    return table


def test_misaligned_per_lipid_table_is_caught(studio_df):
    (message,) = check_lipid_table(_misaligned(studio_df), studio_df)
    assert "Values do not stay with their lipid" in message


def test_names_need_not_be_the_first_column_or_a_column(studio_df):
    kept = studio_df[studio_df['QualityScore'] > 70].reset_index(drop=True)
    good = kept[_KO + ['feature']]
    assert check_lipid_table(good, studio_df) == []
    assert check_lipid_table(good.set_index('feature'), studio_df) == []
    bad = _misaligned(studio_df)[_KO + ['feature']]
    assert check_lipid_table(bad.set_index('feature'), studio_df) != []


def test_summary_tables_are_not_checked(studio_df):
    means = studio_df.groupby('LipidClass')[_KO].mean().reset_index()
    assert check_lipid_table(means, studio_df) == []
    counts = pd.DataFrame({'n': [len(studio_df)]})
    assert check_lipid_table(counts, studio_df) == []


class TestNumbersNotComputed:
    """B2: the explanation streams before the code runs, so a number in it
    is a claim. Only numbers no computed value rounds to are reported."""

    computed = computed_values(
        "Kept 757 of 1394 lipids\n",
        pd.DataFrame({"LipidMolec": ["PE 18:0_20:4"], "p_value": [1.069e-06],
                      "log2FC": [1.987], "share": [0.5430]}),
    )

    def test_rounded_computed_numbers_pass(self):
        prose = ("We kept 757 of 1,394 lipids (54.3%); PE 18:0_20:4 has "
                 "p = 1.07e-06 and log2FC 1.99.")
        assert numbers_not_computed(prose, self.computed) == ()

    def test_times_ten_notation_is_read_as_one_number(self):
        """First eval run with written answers (2026-10-22): "p = 9.27×10⁻⁶"
        was read as 9.27 and flagged."""
        computed = computed_values("", pd.DataFrame({"p": [9.269e-06, 1.065e-06]}))
        prose = "p = 9.27×10⁻⁶ and p = 1.07 x 10^-6; not 3.5×10⁻⁶."
        assert numbers_not_computed(prose, computed) == ("3.5×10⁻⁶",)

    def test_numbers_the_code_never_produced_are_reported(self):
        prose = "There are 24 species, with a mean of 412.7, and 61.2% are PCs."
        assert numbers_not_computed(prose, self.computed) == ("24", "412.7", "61.2%")

    def test_names_small_numbers_and_the_question_are_not_claims(self):
        prose = ("1. In WT-1, s3 and C18 chains of PC 16:0_18:1, across 2 "
                 "conditions, the top 10 are shown.")
        assert numbers_not_computed(prose, self.computed, context="top 10") == ()

    def test_plotted_values_count_as_computed(self):
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=["PC", "PE"], y=[6350.0, 4379.0]))
        assert numbers_not_computed(
            "PC totals 6,350 and PE 4,379.", computed_values(figure=fig),
        ) == ()


# The model's code from 2026-10-21 for "bar chart of the 10 most significant
# lipids ... colored by direction": colors listed, then the bars re-sorted.
MISCOLORED_BARS = """
top10 = lc.compare_species('WT', 'KO').head(10).copy()
top10['Direction'] = top10['log2FC'].apply(lambda x: 'Up in KO' if x > 0 else 'Down in KO')
color_map = {'Up in KO': '#d62728', 'Down in KO': '#1f77b4'}
bar_colors = [color_map[d] for d in top10['Direction']]
top10 = top10.sort_values('log2FC')
fig = go.Figure([go.Bar(x=top10['log2FC'], y=top10['LipidMolec'], orientation='h',
                        marker_color=bar_colors, showlegend=False)])
for label, color in color_map.items():
    fig.add_trace(go.Bar(x=[None], y=[None], orientation='h', marker_color=color,
                         name=label))
"""


class TestDirectionColors:
    """B3: values right, colors wrong. A chart colored by direction must
    have each color on one side of zero."""

    @pytest.fixture(scope='class')
    def lc(self):
        from app.models.experiment import ExperimentConfig
        from app.workflows.pipeline_api import PipelineAPI
        from tests.evals import reference_analysis as ref
        return PipelineAPI(ref.load_canonical(), ExperimentConfig(
            conditions_list=["WT", "KO"], number_of_samples_list=[3, 3]))

    def _figure(self, code, lc):
        import plotly.graph_objects as go
        scope = {"lc": lc, "go": go}
        exec(code, scope)
        return scope["fig"]

    def test_colors_listed_before_a_resort_are_caught(self, lc):
        (warning,) = check_direction_colors(self._figure(MISCOLORED_BARS, lc))
        assert "2 of 10 points" in warning
        assert "'SM d18:1_16:0' is colored 'Up in KO'" in warning

    def test_colors_listed_after_the_sort_pass(self, lc):
        code = MISCOLORED_BARS.replace(
            "top10 = top10.sort_values('log2FC')\n", "",
        ).replace(
            "top10['Direction'] = ",
            "top10 = top10.sort_values('log2FC')\ntop10['Direction'] = ",
        )
        assert check_direction_colors(self._figure(code, lc)) == []

    def test_one_trace_per_direction(self):
        import plotly.graph_objects as go
        good = go.Figure([go.Bar(name="Higher in WT", x=["a", "b"], y=[-1.0, -2.0]),
                          go.Bar(name="Higher in KO", x=["c"], y=[1.5])])
        assert check_direction_colors(good) == []   # which side is "up" is not assumed
        bad = go.Figure([go.Bar(name="Increased", x=["a", "b"], y=[1.0, -2.0]),
                         go.Bar(name="Decreased", x=["c"], y=[-1.5])])
        (warning,) = check_direction_colors(bad)
        assert "'b' is colored 'Increased'" in warning

    def test_colors_that_are_not_directions_are_not_checked(self):
        import plotly.graph_objects as go
        by_class = go.Figure([go.Bar(name="PC", x=["a", "b"], y=[1.0, -2.0]),
                              go.Bar(name="PE", x=["c"], y=[-1.5])])
        assert check_direction_colors(by_class) == []
