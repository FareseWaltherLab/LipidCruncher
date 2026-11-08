"""
AI chat service package.

Provides AI-powered data transformation, data questions and figures
via conversational interface with Claude.
"""
from .exceptions import AIChatError, AIClientError, SandboxError
from .ai_client import AIClientService
from .schema_extractor import SchemaExtractorService
from .sandbox import SandboxService, SandboxResult, validate_code
from .data_transformer import DataTransformerService, TransformationResult
from .codebase_tools import TOOL_DEFINITIONS, execute_tool
from .prompt_templates import (
    data_transform_system_prompt,
    data_questions_system_prompt,
    build_transform_context,
)

__all__ = [
    "AIChatError",
    "AIClientError",
    "SandboxError",
    "AIClientService",
    "SchemaExtractorService",
    "SandboxService",
    "SandboxResult",
    "validate_code",
    "DataTransformerService",
    "TransformationResult",
    "TOOL_DEFINITIONS",
    "execute_tool",
    "data_transform_system_prompt",
    "data_questions_system_prompt",
    "build_transform_context",
]
