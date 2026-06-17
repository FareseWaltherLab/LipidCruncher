"""
Unit tests for AI chat Pydantic models.
"""
from datetime import datetime

import pytest

from app.models.chat import (
    ChatMessage,
    ChatSession,
    ChatMode,
    AIConfig,
    SandboxConfig,
)


class TestChatMessage:

    def test_create_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.code is None
        assert msg.figure_key is None

    def test_create_assistant_message_with_code(self):
        msg = ChatMessage(role="assistant", content="Here's the code", code="x = 1")
        assert msg.role == "assistant"
        assert msg.code == "x = 1"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="role must be"):
            ChatMessage(role="system", content="test")

    def test_frozen(self):
        msg = ChatMessage(role="user", content="test")
        with pytest.raises(Exception):
            msg.content = "changed"

    def test_hashable(self):
        msg = ChatMessage(role="user", content="test", timestamp=datetime(2026, 1, 1))
        assert hash(msg) == hash(msg)
        s = {msg}
        assert len(s) == 1


class TestChatSession:

    def test_create_empty_session(self):
        session = ChatSession(mode=ChatMode.DATA_TRANSFORM)
        assert session.mode == ChatMode.DATA_TRANSFORM
        assert session.messages == ()
        assert session.schema_context == ""

    def test_add_message_returns_new_session(self):
        session = ChatSession(mode=ChatMode.VISUALIZATION)
        msg = ChatMessage(role="user", content="Show bar chart")
        new_session = session.add_message(msg)

        assert len(session.messages) == 0  # original unchanged
        assert len(new_session.messages) == 1
        assert new_session.messages[0].content == "Show bar chart"
        assert new_session.mode == ChatMode.VISUALIZATION

    def test_to_api_messages(self):
        session = ChatSession(mode=ChatMode.DATA_TRANSFORM)
        session = session.add_message(ChatMessage(role="user", content="Hello"))
        session = session.add_message(ChatMessage(role="assistant", content="Hi"))

        api_msgs = session.to_api_messages()
        assert len(api_msgs) == 2
        assert api_msgs[0] == {"role": "user", "content": "Hello"}
        assert api_msgs[1] == {"role": "assistant", "content": "Hi"}

    def test_to_api_messages_includes_assistant_code(self):
        session = ChatSession(mode=ChatMode.VISUALIZATION)
        session = session.add_message(ChatMessage(role="user", content="Plot it"))
        session = session.add_message(ChatMessage(
            role="assistant",
            content="Here's the visualization.",
            code="fig = px.bar(df, x='ClassKey')",
        ))

        api_msgs = session.to_api_messages()
        assert "Here's the visualization." in api_msgs[1]["content"]
        assert "fig = px.bar(df, x='ClassKey')" in api_msgs[1]["content"]
        assert "```python" in api_msgs[1]["content"]

    def test_to_api_messages_no_code_unchanged(self):
        session = ChatSession(mode=ChatMode.VISUALIZATION)
        session = session.add_message(ChatMessage(
            role="assistant", content="Just text, no code",
        ))
        api_msgs = session.to_api_messages()
        assert api_msgs[0]["content"] == "Just text, no code"

    def test_to_api_messages_user_code_field_ignored(self):
        # User messages don't normally have code, but if they do (defensive),
        # the user's content should be sent as-is, not wrapped.
        session = ChatSession(mode=ChatMode.VISUALIZATION)
        session = session.add_message(ChatMessage(
            role="user", content="Question", code="some code",
        ))
        api_msgs = session.to_api_messages()
        assert api_msgs[0]["content"] == "Question"

    def test_frozen(self):
        session = ChatSession(mode=ChatMode.DATA_TRANSFORM)
        with pytest.raises(Exception):
            session.mode = ChatMode.VISUALIZATION

    def test_hashable(self):
        session = ChatSession(mode=ChatMode.DATA_TRANSFORM, schema_context="cols: a, b")
        assert hash(session) == hash(session)


class TestAIConfig:

    def test_defaults(self):
        config = AIConfig()
        assert config.model == "claude-sonnet-4-6"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0

    def test_custom_values(self):
        config = AIConfig(model="claude-haiku-4-5-20251001", max_tokens=1024, temperature=0.5)
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_tokens == 1024
        assert config.temperature == 0.5

    def test_temperature_bounds(self):
        with pytest.raises(ValueError):
            AIConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            AIConfig(temperature=1.1)

    def test_frozen(self):
        config = AIConfig()
        with pytest.raises(Exception):
            config.temperature = 0.5

    def test_hashable(self):
        config = AIConfig()
        assert hash(config) == hash(config)


class TestSandboxConfig:

    def test_defaults(self):
        config = SandboxConfig()
        assert config.execution_timeout == 10
        assert config.max_output_rows == 50000
        assert config.max_figure_size_bytes == 10 * 1024 * 1024

    def test_timeout_bounds(self):
        with pytest.raises(ValueError):
            SandboxConfig(execution_timeout=0)
        with pytest.raises(ValueError):
            SandboxConfig(execution_timeout=61)

    def test_frozen(self):
        config = SandboxConfig()
        with pytest.raises(Exception):
            config.execution_timeout = 30

    def test_hashable(self):
        config = SandboxConfig()
        assert hash(config) == hash(config)
