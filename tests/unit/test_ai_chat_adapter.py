"""
Unit tests for the AI chat adapter integration.

Tests that SessionState has the correct chat_* keys, that adapter
methods delegate correctly, and that the chat UI modules import cleanly.
"""
from dataclasses import fields
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from app.adapters.streamlit_adapter import SessionState
from app.models.chat import ChatMode, ChatSession
from app.models.experiment import ExperimentConfig
from app.workflows.ai_chat import AIChatWorkflow


# ---------------------------------------------------------------------------
# SessionState chat keys
# ---------------------------------------------------------------------------

class TestSessionStateChatKeys:
    """Verify chat_* keys exist with correct defaults."""

    def test_chat_transform_session_default_none(self):
        state = SessionState()
        assert state.chat_transform_session is None

    def test_chat_viz_session_default_none(self):
        state = SessionState()
        assert state.chat_viz_session is None

    def test_chat_mode_default_none(self):
        state = SessionState()
        assert state.chat_mode is None

    def test_chat_transform_result_default_none(self):
        state = SessionState()
        assert state.chat_transform_result is None

    def test_chat_viz_figures_default_empty_dict(self):
        state = SessionState()
        assert state.chat_viz_figures == {}

    def test_chat_message_count_default_zero(self):
        state = SessionState()
        assert state.chat_message_count == 0

    def test_chat_execution_count_default_zero(self):
        state = SessionState()
        assert state.chat_execution_count == 0

    def test_chat_keys_in_session_state_fields(self):
        """All chat_* fields exist in the SessionState dataclass."""
        field_names = {f.name for f in fields(SessionState)}
        expected = {
            'chat_transform_session', 'chat_viz_session', 'chat_mode',
            'chat_transform_result', 'chat_viz_figures',
            'chat_message_count', 'chat_execution_count',
        }
        assert expected.issubset(field_names)


# ---------------------------------------------------------------------------
# Workflow session creation (tested via adapter entry points)
# ---------------------------------------------------------------------------

class TestWorkflowSessionCreation:

    def test_start_transform_session(self):
        df = pd.DataFrame({
            "lipid": ["PC 34:1", "PE 36:2"],
            "area_1": [100.0, 200.0],
        })
        session = AIChatWorkflow.start_transform_session(df)
        assert session.mode == ChatMode.DATA_TRANSFORM
        assert "lipid" in session.schema_context
        assert len(session.messages) == 0

    def test_start_viz_session(self):
        df = pd.DataFrame({
            "LipidMolec": ["PC 34:1"],
            "ClassKey": ["PC"],
            "intensity[s1]": [100.0],
            "intensity[s2]": [110.0],
        })
        experiment = ExperimentConfig(
            conditions_list=["Control", "Treatment"],
            number_of_samples_list=[1, 1],
        )
        session = AIChatWorkflow.start_viz_session(df, experiment)
        assert session.mode == ChatMode.VISUALIZATION
        assert "LipidMolec" in session.schema_context

    def test_transform_session_preserves_schema(self):
        """Schema context is set once and persists through messages."""
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
        session = AIChatWorkflow.start_transform_session(df)
        assert "col_a" in session.schema_context
        assert "col_b" in session.schema_context


# ---------------------------------------------------------------------------
# Module import smoke tests
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Verify all Phase 4 modules import without errors."""

    def test_import_chat_panel(self):
        from app.ui.ai_chat import chat_panel
        assert hasattr(chat_panel, 'display_transform_chat')
        assert hasattr(chat_panel, 'display_viz_chat')

    def test_import_data_transform_ui(self):
        from app.ui.ai_chat import _data_transform_ui
        assert hasattr(_data_transform_ui, 'render_transform_chat')

    def test_import_viz_generation_ui(self):
        from app.ui.ai_chat import _viz_generation_ui
        assert hasattr(_viz_generation_ui, 'render_viz_chat')

    def test_import_ai_chat_package(self):
        from app.ui.ai_chat import display_transform_chat, display_viz_chat
        assert callable(display_transform_chat)
        assert callable(display_viz_chat)


# ---------------------------------------------------------------------------
# Interaction logging: adapter supplies the session-state context
# ---------------------------------------------------------------------------

class TestLogAiTurnContext:
    def test_log_ai_turn_passes_model_dataset_history_and_figure(self):
        import plotly.graph_objects as go
        from app.adapters.streamlit_adapter import StreamlitAdapter
        from app.models.chat import AIConfig, ChatMessage

        df = pd.DataFrame({"LipidMolec": ["a"], "s1": [1.0]})
        exp = ExperimentConfig(conditions_list=["WT"], number_of_samples_list=[1])
        fig = go.Figure(data=[go.Bar(x=["a"], y=[1])])
        history = (ChatMessage(role="user", content="earlier"),)

        with patch("app.adapters.streamlit_adapter.st") as mock_st, \
             patch("app.services.ai_chat.interaction_logger."
                   "InteractionLoggingService.log_turn") as log_turn:
            mock_st.session_state = {
                "chat_studio_file_id": "run.csv_42",
                "chat_studio_detected_format": "Generic",
            }
            turn_id = StreamlitAdapter.log_ai_turn(
                tab="visualization", question="plot", answer="ok",
                produced_figure=True, history=history, df=df,
                experiment=exp, figure=fig,
            )

        kwargs = log_turn.call_args.kwargs
        assert kwargs["turn_id"] == turn_id
        assert kwargs["model"] == AIConfig().model
        assert kwargs["dataset"]["source"] == "run.csv_42"
        assert kwargs["dataset"]["format"] == "Generic"
        assert kwargs["dataset"]["rows"] == 1
        assert kwargs["dataset"]["conditions"] == {"WT": 1}
        assert kwargs["history"] is history
        assert '"bar"' in kwargs["figure_json"]

    def test_pipeline_import_source_wins_over_file_id(self):
        from app.adapters.streamlit_adapter import StreamlitAdapter

        with patch("app.adapters.streamlit_adapter.st") as mock_st, \
             patch("app.services.ai_chat.interaction_logger."
                   "InteractionLoggingService.log_turn") as log_turn:
            mock_st.session_state = {
                "chat_studio_file_id": "_pipeline_import",
                "chat_studio_pipeline_source": "normalized",
            }
            StreamlitAdapter.log_ai_turn(
                tab="data_questions", question="q", answer="a",
                df=pd.DataFrame({"x": [1]}),
            )
        assert log_turn.call_args.kwargs["dataset"]["source"] == "normalized"
