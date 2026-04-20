"""
AI chat session models.

Frozen Pydantic models for chat messages, sessions, configuration,
and sandbox execution settings.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict, Any

from pydantic import BaseModel, Field, field_validator


class ChatMode(str, Enum):
    """Chat interaction mode."""
    DATA_TRANSFORM = "data_transform"
    VISUALIZATION = "visualization"
    DATA_QUESTIONS = "data_questions"


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Message author — 'user' or 'assistant'.
        content: Text content of the message.
        timestamp: When the message was created.
        code: Optional code block extracted from or generated in this message.
        figure_key: Optional session state key for an associated figure.
    """
    model_config = {"frozen": True}

    role: str = Field(description="Message author: 'user' or 'assistant'")
    content: str = Field(description="Text content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)
    code: Optional[str] = Field(default=None, description="Code block associated with this message")
    figure_key: Optional[str] = Field(default=None, description="Session state key for associated figure")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v

    def __hash__(self) -> int:
        return hash((self.role, self.content, self.timestamp, self.code, self.figure_key))


class ChatSession(BaseModel):
    """An immutable chat session with message history.

    Attributes:
        mode: The chat mode (data transformation or visualization).
        messages: Tuple of messages in chronological order.
        schema_context: Pre-computed schema description of the source DataFrame.
    """
    model_config = {"frozen": True}

    mode: ChatMode = Field(description="Chat interaction mode")
    messages: Tuple[ChatMessage, ...] = Field(default=(), description="Message history")
    schema_context: str = Field(default="", description="DataFrame schema context for AI")

    def __hash__(self) -> int:
        return hash((self.mode, self.messages, self.schema_context))

    def add_message(self, message: ChatMessage) -> "ChatSession":
        """Return a new ChatSession with the message appended."""
        return ChatSession(
            mode=self.mode,
            messages=self.messages + (message,),
            schema_context=self.schema_context,
        )

    def to_api_messages(self) -> list:
        """Convert message history to Anthropic API format.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]


class AIConfig(BaseModel):
    """Configuration for AI API calls.

    Attributes:
        model: Anthropic model identifier.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0 = deterministic).
    """
    model_config = {"frozen": True}

    model: str = Field(default="claude-sonnet-4-20250514", description="Anthropic model ID")
    max_tokens: int = Field(default=4096, ge=1, le=8192, description="Max response tokens")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")

    def __hash__(self) -> int:
        return hash((self.model, self.max_tokens, self.temperature))


class SandboxConfig(BaseModel):
    """Configuration for sandboxed code execution.

    Attributes:
        execution_timeout: Maximum seconds for code execution.
        max_output_rows: Maximum rows in a result DataFrame.
        max_figure_size_bytes: Maximum figure size in bytes (10 MB default).
    """
    model_config = {"frozen": True}

    execution_timeout: int = Field(default=10, ge=1, le=60, description="Execution timeout in seconds")
    max_output_rows: int = Field(default=50000, ge=1, description="Max rows in result DataFrame")
    max_figure_size_bytes: int = Field(default=10 * 1024 * 1024, ge=1, description="Max figure size in bytes")

    def __hash__(self) -> int:
        return hash((self.execution_timeout, self.max_output_rows, self.max_figure_size_bytes))
