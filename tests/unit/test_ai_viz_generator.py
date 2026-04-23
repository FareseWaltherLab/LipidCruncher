"""
Unit tests for the AI visualization generator service.

Tests execute_viz_code with pre-written code (no AI calls needed),
and workflow viz methods with a mocked AI client.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.models.chat import AIConfig, ChatMode, SandboxConfig
from app.models.experiment import ExperimentConfig
from app.services.ai_chat.viz_generator import VizGeneratorService, _build_experiment_context
from app.workflows.ai_chat import AIChatWorkflow, VizStepResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analysis_df():
    """Analysis-ready DataFrame with Generic format."""
    return pd.DataFrame({
        "LipidMolec": ["PC 34:1", "PE 36:2", "SM 42:1", "PC 36:2"],
        "ClassKey": ["PC", "PE", "SM", "PC"],
        "intensity[s1]": [100.0, 200.0, 300.0, 150.0],
        "intensity[s2]": [110.0, 190.0, 310.0, 160.0],
        "intensity[s3]": [90.0, 210.0, 290.0, 140.0],
        "intensity[s4]": [120.0, 180.0, 320.0, 170.0],
    })


@pytest.fixture
def experiment():
    return ExperimentConfig(
        conditions_list=["Control", "Treatment"],
        number_of_samples_list=[2, 2],
    )


@pytest.fixture
def viz_session(analysis_df, experiment):
    return AIChatWorkflow.start_viz_session(analysis_df, experiment)


# ---------------------------------------------------------------------------
# Experiment Context Builder
# ---------------------------------------------------------------------------

class TestBuildExperimentContext:

    def test_includes_conditions(self, experiment):
        ctx = _build_experiment_context(experiment)
        assert "Control" in ctx
        assert "Treatment" in ctx

    def test_includes_sample_count(self, experiment):
        ctx = _build_experiment_context(experiment)
        assert "2" in ctx

    def test_includes_sample_labels(self, experiment):
        ctx = _build_experiment_context(experiment)
        assert "s1" in ctx
        assert "s4" in ctx

    def test_includes_mapping(self, experiment):
        ctx = _build_experiment_context(experiment)
        assert "Control: s1, s2" in ctx
        assert "Treatment: s3, s4" in ctx


# ---------------------------------------------------------------------------
# Execute Viz Code (real sandbox, no AI)
# ---------------------------------------------------------------------------

class TestExecuteVizCode:

    def test_plotly_express_bar(self, analysis_df):
        code = "fig = px.bar(df, x='LipidMolec', y='intensity[s1]', title='Test')"
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is True
        assert result.figure is not None
        assert isinstance(result.figure, go.Figure)

    def test_plotly_go_scatter(self, analysis_df):
        code = (
            "fig = go.Figure()\n"
            "fig.add_trace(go.Scatter(x=df['LipidMolec'], y=df['intensity[s1]'], mode='markers'))\n"
            "fig.update_layout(title='Scatter Test')\n"
        )
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is True
        assert result.figure is not None

    def test_plotly_pie_chart(self, analysis_df):
        code = (
            "class_totals = df.groupby('ClassKey')['intensity[s1]'].sum()\n"
            "fig = px.pie(values=class_totals.values, names=class_totals.index, title='Composition')\n"
        )
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is True
        assert result.figure is not None

    def test_plotly_box_plot(self, analysis_df):
        code = (
            "intensity_cols = [c for c in df.columns if c.startswith('intensity')]\n"
            "melted = df.melt(id_vars=['LipidMolec', 'ClassKey'], value_vars=intensity_cols)\n"
            "fig = px.box(melted, x='ClassKey', y='value', title='Distribution')\n"
        )
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is True
        assert result.figure is not None

    def test_code_without_fig_returns_no_figure(self, analysis_df):
        code = "x = df['intensity[s1]'].mean()"
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is True
        assert result.figure is None

    def test_runtime_error_in_viz_code(self, analysis_df):
        code = "fig = px.bar(df, x='nonexistent_col', y='intensity[s1]')"
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is False

    def test_forbidden_import_blocked(self, analysis_df):
        code = "import os\nfig = px.bar(df, x='ClassKey', y='intensity[s1]')"
        result = VizGeneratorService.execute_viz_code(code, analysis_df)
        assert result.success is False

    def test_df_is_copy(self, analysis_df):
        """Viz code cannot modify the original DataFrame."""
        original_val = analysis_df["intensity[s1]"].iloc[0]
        code = "df['intensity[s1]'] = 0\nfig = px.bar(df, x='ClassKey', y='intensity[s1]')"
        VizGeneratorService.execute_viz_code(code, analysis_df)
        assert analysis_df["intensity[s1]"].iloc[0] == original_val


# ---------------------------------------------------------------------------
# Workflow — Start Viz Session
# ---------------------------------------------------------------------------

class TestStartVizSession:

    def test_creates_visualization_session(self, analysis_df, experiment):
        session = AIChatWorkflow.start_viz_session(analysis_df, experiment)
        assert session.mode == ChatMode.VISUALIZATION
        assert session.messages == ()

    def test_schema_context_populated(self, analysis_df, experiment):
        session = AIChatWorkflow.start_viz_session(analysis_df, experiment)
        assert "LipidMolec" in session.schema_context
        assert "intensity[s1]" in session.schema_context


# ---------------------------------------------------------------------------
# Workflow — Handle Viz Message (mocked AI)
# ---------------------------------------------------------------------------

class TestHandleVizMessage:
    """Tests mock execute_tool_loop which returns the full response text
    (including code fences). extract_code_block then splits it into
    explanation + code."""

    @patch("app.services.ai_chat.viz_generator.AIClientService.execute_tool_loop")
    def test_successful_viz(self, mock_loop, analysis_df, experiment, viz_session):
        mock_loop.return_value = (
            "Here's a bar chart of intensities.\n\n"
            "```python\n"
            "fig = px.bar(df, x='LipidMolec', y='intensity[s1]', title='Intensities')\n"
            "```"
        )

        result = AIChatWorkflow.handle_viz_message(
            user_message="Show a bar chart of intensities",
            session=viz_session,
            analysis_df=analysis_df,
            experiment=experiment,
        )

        assert isinstance(result, VizStepResult)
        assert result.sandbox_result is not None
        assert result.sandbox_result.success is True
        assert result.sandbox_result.figure is not None
        assert len(result.session.messages) == 2

    @patch("app.services.ai_chat.viz_generator.AIClientService.execute_tool_loop")
    def test_ai_asks_clarification(self, mock_loop, analysis_df, experiment, viz_session):
        mock_loop.return_value = (
            "Could you clarify which samples you'd like to compare?"
        )

        result = AIChatWorkflow.handle_viz_message(
            "Show me a comparison",
            viz_session,
            analysis_df,
            experiment,
        )

        assert result.sandbox_result is None
        assert "clarify" in result.assistant_message.lower()

    @patch("app.services.ai_chat.viz_generator.AIClientService.execute_tool_loop")
    def test_code_error_reported(self, mock_loop, analysis_df, experiment, viz_session):
        mock_loop.return_value = (
            "Here's the chart.\n\n"
            "```python\n"
            "fig = px.bar(df, x='nonexistent', y='also_nonexistent')\n"
            "```"
        )

        result = AIChatWorkflow.handle_viz_message(
            "Make a chart",
            viz_session,
            analysis_df,
            experiment,
        )

        assert result.sandbox_result is not None
        assert result.sandbox_result.success is False
        assert "error" in result.assistant_message.lower()

    @patch("app.services.ai_chat.viz_generator.AIClientService.execute_tool_loop")
    def test_code_without_fig_reported(self, mock_loop, analysis_df, experiment, viz_session):
        mock_loop.return_value = (
            "Computing mean.\n\n"
            "```python\n"
            "x = df['intensity[s1]'].mean()\n"
            "```"
        )

        result = AIChatWorkflow.handle_viz_message(
            "Calculate mean",
            viz_session,
            analysis_df,
            experiment,
        )

        assert result.sandbox_result is not None
        assert result.sandbox_result.success is True
        assert result.sandbox_result.figure is None
        assert "didn't produce a figure" in result.assistant_message

    @patch("app.services.ai_chat.viz_generator.AIClientService.execute_tool_loop")
    def test_session_accumulates(self, mock_loop, analysis_df, experiment, viz_session):
        mock_loop.return_value = "Got it."
        r1 = AIChatWorkflow.handle_viz_message("First", viz_session, analysis_df, experiment)
        assert len(r1.session.messages) == 2

        mock_loop.return_value = "Sure."
        r2 = AIChatWorkflow.handle_viz_message("Second", r1.session, analysis_df, experiment)
        assert len(r2.session.messages) == 4
