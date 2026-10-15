"""
Unit tests for the AI chat workflow.

Tests workflow orchestration with a mocked AI client to avoid real API calls.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from app.models.chat import ChatMode
from app.workflows.ai_chat import AIChatWorkflow, TransformStepResult
from tests.ai_turns import canned_stream, finish


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source_df():
    """Non-standard DataFrame for transformation tests."""
    return pd.DataFrame({
        "lipid_name": ["PC(34:1)", "PE(36:2)", "SM(42:1)"],
        "lipid_class": ["PC", "PE", "SM"],
        "area_s1": [100.0, 200.0, 300.0],
        "area_s2": [110.0, 190.0, 310.0],
    })


@pytest.fixture
def transform_session(source_df):
    """A fresh transform session."""
    return AIChatWorkflow.start_transform_session(source_df)


# Mock response with valid transformation code
MOCK_CODE_RESPONSE = """\
I'll rename the columns to match the Generic format.

```python
result_df = pd.DataFrame()
result_df['LipidMolec'] = df['lipid_name'].str.replace(r'^([A-Za-z]+)\\((.+)\\)$', r'\\1 \\2', regex=True)
result_df['Sample1'] = pd.to_numeric(df['area_s1'])
result_df['Sample2'] = pd.to_numeric(df['area_s2'])
```
"""

# Mock response with no code (AI asking a clarifying question)
MOCK_QUESTION_RESPONSE = (
    "I see your data has columns lipid_name, lipid_class, area_s1, area_s2. "
    "Could you confirm that area_s1 and area_s2 are raw intensity values for two samples?"
)


# ---------------------------------------------------------------------------
# Start Session
# ---------------------------------------------------------------------------

class TestStartTransformSession:

    def test_creates_data_transform_session(self, source_df):
        session = AIChatWorkflow.start_transform_session(source_df)
        assert session.mode == ChatMode.DATA_TRANSFORM
        assert session.messages == ()

    def test_schema_context_populated(self, source_df):
        session = AIChatWorkflow.start_transform_session(source_df)
        assert "lipid_name" in session.schema_context
        assert "4 columns" in session.schema_context

    def test_schema_includes_sample_rows(self, source_df):
        session = AIChatWorkflow.start_transform_session(source_df)
        assert "PC(34:1)" in session.schema_context


# ---------------------------------------------------------------------------
# Transform turn — with mocked AI
# ---------------------------------------------------------------------------

class TestTransformTurn:

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_successful_transformation(self, mock_send, source_df, transform_session):
        mock_send.side_effect = canned_stream(
            "I'll rename the columns to match the Generic format.\n\n"
            "```python\n"
            "result_df = pd.DataFrame()\n"
            "result_df['LipidMolec'] = df['lipid_name'].str.replace("
            "r'^([A-Za-z]+)\\\\((.+)\\\\)$', r'\\\\1 \\\\2', regex=True)\n"
            "result_df['Sample1'] = pd.to_numeric(df['area_s1'])\n"
            "result_df['Sample2'] = pd.to_numeric(df['area_s2'])\n"
            "```"
        )

        result = finish(AIChatWorkflow.stream_transform_message(
            user_message="Transform my data",
            session=transform_session,
            source_df=source_df,
        ))

        assert isinstance(result, TransformStepResult)
        assert result.transformation is not None
        assert result.transformation.success is True
        assert result.transformation.result_df is not None
        assert "LipidMolec" in result.transformation.result_df.columns
        assert len(result.session.messages) == 2  # user + assistant

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_ai_asks_question(self, mock_send, source_df, transform_session):
        mock_send.side_effect = canned_stream(MOCK_QUESTION_RESPONSE)

        result = finish(AIChatWorkflow.stream_transform_message(
            user_message="Please transform my data",
            session=transform_session,
            source_df=source_df,
        ))

        assert result.transformation is None
        assert "confirm" in result.assistant_message.lower()
        assert len(result.session.messages) == 2

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_failed_transformation_reports_errors(self, mock_send, source_df, transform_session):
        mock_send.side_effect = canned_stream(
            "Here's the transformation.\n\n"
            "```python\n"
            "result_df = pd.DataFrame({'wrong': [1]})\n"
            "```"
        )

        result = finish(AIChatWorkflow.stream_transform_message(
            user_message="Transform it",
            session=transform_session,
            source_df=source_df,
        ))

        assert result.transformation is not None
        assert result.transformation.success is False
        assert len(result.transformation.validation_errors) > 0
        assert "issues" in result.assistant_message.lower()

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_session_accumulates_messages(self, mock_send, source_df, transform_session):
        mock_send.side_effect = canned_stream("Got it.")

        result1 = finish(AIChatWorkflow.stream_transform_message(
            "First message", transform_session, source_df
        ))
        assert len(result1.session.messages) == 2

        mock_send.side_effect = canned_stream("Sure.")
        result2 = finish(AIChatWorkflow.stream_transform_message(
            "Second message", result1.session, source_df
        ))
        assert len(result2.session.messages) == 4  # 2 from first turn + 2 from second

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_code_included_in_assistant_message(self, mock_send, source_df, transform_session):
        code = "result_df = df.rename(columns={'lipid_name': 'LipidMolec'})"
        mock_send.side_effect = canned_stream(f"Renaming columns.\n\n```python\n{code}\n```")

        result = finish(AIChatWorkflow.stream_transform_message(
            "Transform", transform_session, source_df
        ))

        assistant_msg = result.session.messages[-1]
        assert assistant_msg.code == code


class TestStatisticalChoicesAreShown:
    """Regression (AQ5): the user was never told which statistical choices
    `lc` made. The answer now ends with a note listing them."""

    def test_data_questions_answer_lists_the_choices(self):
        from app.models.chat import ChatMode, ChatSession, SandboxConfig
        from app.models.experiment import ExperimentConfig
        from app.workflows.ai_chat import AIChatWorkflow
        from tests.evals import reference_analysis as ref

        result = AIChatWorkflow._finalize_data_questions(
            ChatSession(mode=ChatMode.DATA_QUESTIONS),
            "Comparing WT and KO.\n\n```python\n"
            "result_df = lc.compare_species('WT', 'KO')\n```",
            ref.load_canonical(),
            SandboxConfig(),
            None,
            ExperimentConfig(
                conditions_list=["WT", "KO"], number_of_samples_list=[3, 3],
            ),
        )
        assert result.result_df is not None
        assert "**Statistical choices**" in result.assistant_message
        assert "no multiple-testing correction" in result.assistant_message
        assert "main app's Volcano defaults" in result.assistant_message

    def test_no_note_when_lc_ran_no_statistics(self):
        from app.models.chat import ChatMode, ChatSession, SandboxConfig
        from app.workflows.ai_chat import AIChatWorkflow
        from tests.evals import reference_analysis as ref

        result = AIChatWorkflow._finalize_data_questions(
            ChatSession(mode=ChatMode.DATA_QUESTIONS),
            "Counting.\n\n```python\nprint(len(df))\n```",
            ref.load_canonical(), SandboxConfig(), None, None,
        )
        assert "Statistical choices" not in result.assistant_message


class TestFigureHasData:
    """Regression (AQ2): only x/y were checked, so pies, treemaps, heatmaps
    and other non-cartesian figures (including lc.pie_charts()) were told
    to the user as "an empty figure (no data was plotted)"."""

    @pytest.fixture
    def frame(self):
        import pandas as pd
        return pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "v": [1, 2]})

    def test_non_cartesian_figures_have_data(self, frame):
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        from app.workflows.ai_chat import figure_has_data

        figures = [
            px.pie(values=[1, 2], names=["a", "b"]),
            px.imshow(np.ones((2, 2))),
            go.Figure(go.Heatmap(z=[[1, 2]])),
            px.treemap(frame, path=["a", "b"], values="v"),
            px.sunburst(frame, path=["a", "b"], values="v"),
            px.parallel_coordinates(pd.DataFrame({"x": [1, 2], "y": [3, 4]})),
            go.Figure(go.Table(header={"values": ["a"]},
                               cells={"values": [[1, 2]]})),
            px.scatter(x=[1], y=[2]),
        ]
        assert all(figure_has_data(f) for f in figures)

    def test_empty_figures_have_none(self):
        import plotly.graph_objects as go

        from app.workflows.ai_chat import figure_has_data

        assert not figure_has_data(go.Figure(go.Scatter(x=[], y=[])))
        assert not figure_has_data(go.Figure())
        assert not figure_has_data(None)

    def test_lc_pie_chart_is_not_called_empty(self):
        from app.models.chat import ChatMode, ChatSession, SandboxConfig
        from app.models.experiment import ExperimentConfig
        from app.workflows.ai_chat import AIChatWorkflow
        from tests.evals import reference_analysis as ref

        result = AIChatWorkflow._finalize_viz(
            ChatSession(mode=ChatMode.VISUALIZATION),
            "A pie chart of WT.",
            "fig = lc.pie_charts()['WT'].figure",
            ref.load_canonical(),
            SandboxConfig(),
            None,
            ExperimentConfig(
                conditions_list=["WT", "KO"], number_of_samples_list=[3, 3],
            ),
        )
        assert result.sandbox_result.figure is not None
        assert "empty figure" not in result.assistant_message
        assert "Here's the visualization" in result.assistant_message


class TestVizKeepsPrintedOutput:
    """Regression (AQ7): Custom Viz showed printed output only in an
    expander drawn just before a rerun, and never put it in the message, so
    it vanished at once and later turns could not see it."""

    def test_printed_output_is_part_of_the_answer(self):
        from app.models.chat import ChatMode, ChatSession, SandboxConfig
        from app.workflows.ai_chat import AIChatWorkflow
        from tests.evals import reference_analysis as ref

        result = AIChatWorkflow._finalize_viz(
            ChatSession(mode=ChatMode.VISUALIZATION),
            "Plotting.",
            "print('rows plotted: 24')\n"
            "fig = px.bar(df, x='ClassKey', y='concentration[s1]')",
            ref.load_canonical(), SandboxConfig(), None, None,
        )
        assert "```\nrows plotted: 24\n```" in result.assistant_message
        assert "rows plotted: 24" in result.session.messages[-1].content


class TestRegressions:
    """AQ11: fixes that had no test."""

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_printing_without_a_result_is_an_info_turn(
        self, mock_loop, source_df, transform_session,
    ):
        """8af20ad: a turn that prints what it found (to plan a transform)
        and sets no result_df is a successful info-only turn, its output
        shown as the answer, not a failed transformation."""
        mock_loop.side_effect = canned_stream(
            "Listing the classes.\n\n"
            "```python\nprint(sorted(df['lipid_class'].unique()))\n```"
        )
        result = finish(AIChatWorkflow.stream_transform_message(
            "Which classes are there?", transform_session, source_df,
        ))
        t = result.transformation
        assert t.success and t.is_info_only
        assert t.result_df is None
        assert "['PC', 'PE', 'SM']" in result.assistant_message

    @patch("app.services.ai_chat.ai_client.AIClientService.stream_tool_loop")
    def test_bare_code_block_with_a_table_is_not_an_empty_answer(
        self, mock_loop, source_df,
    ):
        """d9f3ac6: a reply that is only a code block setting result_df got
        the "I wasn't able to generate a response" fallback."""
        mock_loop.side_effect = canned_stream(
            "```python\nresult_df = df.head(2)\n```"
        )
        session = AIChatWorkflow.start_data_questions_session(source_df)
        result = finish(AIChatWorkflow.stream_data_questions_message(
            "first two rows", session, source_df,
        ))
        assert result.result_df is not None and len(result.result_df) == 2
        assert result.assistant_message == "Here's the result:"
