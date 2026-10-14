"""
Unit tests for the AI chat workflow.

Tests workflow orchestration with a mocked AI client to avoid real API calls.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from app.models.chat import ChatMode
from app.workflows.ai_chat import AIChatWorkflow, TransformStepResult


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
# Handle Transform Message — with mocked AI
# ---------------------------------------------------------------------------

class TestHandleTransformMessage:

    @patch("app.services.ai_chat.data_transformer.AIClientService.execute_tool_loop")
    def test_successful_transformation(self, mock_send, source_df, transform_session):
        mock_send.return_value = (
            "I'll rename the columns to match the Generic format.\n\n"
            "```python\n"
            "result_df = pd.DataFrame()\n"
            "result_df['LipidMolec'] = df['lipid_name'].str.replace("
            "r'^([A-Za-z]+)\\\\((.+)\\\\)$', r'\\\\1 \\\\2', regex=True)\n"
            "result_df['Sample1'] = pd.to_numeric(df['area_s1'])\n"
            "result_df['Sample2'] = pd.to_numeric(df['area_s2'])\n"
            "```"
        )

        result = AIChatWorkflow.handle_transform_message(
            user_message="Transform my data",
            session=transform_session,
            source_df=source_df,
        )

        assert isinstance(result, TransformStepResult)
        assert result.transformation is not None
        assert result.transformation.success is True
        assert result.transformation.result_df is not None
        assert "LipidMolec" in result.transformation.result_df.columns
        assert len(result.session.messages) == 2  # user + assistant

    @patch("app.services.ai_chat.data_transformer.AIClientService.execute_tool_loop")
    def test_ai_asks_question(self, mock_send, source_df, transform_session):
        mock_send.return_value = MOCK_QUESTION_RESPONSE

        result = AIChatWorkflow.handle_transform_message(
            user_message="Please transform my data",
            session=transform_session,
            source_df=source_df,
        )

        assert result.transformation is None
        assert "confirm" in result.assistant_message.lower()
        assert len(result.session.messages) == 2

    @patch("app.services.ai_chat.data_transformer.AIClientService.execute_tool_loop")
    def test_failed_transformation_reports_errors(self, mock_send, source_df, transform_session):
        mock_send.return_value = (
            "Here's the transformation.\n\n"
            "```python\n"
            "result_df = pd.DataFrame({'wrong': [1]})\n"
            "```"
        )

        result = AIChatWorkflow.handle_transform_message(
            user_message="Transform it",
            session=transform_session,
            source_df=source_df,
        )

        assert result.transformation is not None
        assert result.transformation.success is False
        assert len(result.transformation.validation_errors) > 0
        assert "issues" in result.assistant_message.lower()

    @patch("app.services.ai_chat.data_transformer.AIClientService.execute_tool_loop")
    def test_session_accumulates_messages(self, mock_send, source_df, transform_session):
        mock_send.return_value = "Got it."

        result1 = AIChatWorkflow.handle_transform_message(
            "First message", transform_session, source_df
        )
        assert len(result1.session.messages) == 2

        mock_send.return_value = "Sure."
        result2 = AIChatWorkflow.handle_transform_message(
            "Second message", result1.session, source_df
        )
        assert len(result2.session.messages) == 4  # 2 from first turn + 2 from second

    @patch("app.services.ai_chat.data_transformer.AIClientService.execute_tool_loop")
    def test_code_included_in_assistant_message(self, mock_send, source_df, transform_session):
        code = "result_df = df.rename(columns={'lipid_name': 'LipidMolec'})"
        mock_send.return_value = f"Renaming columns.\n\n```python\n{code}\n```"

        result = AIChatWorkflow.handle_transform_message(
            "Transform", transform_session, source_df
        )

        assistant_msg = result.session.messages[-1]
        assert assistant_msg.code == code
