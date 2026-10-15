"""
AI chat service package.

Provides AI-powered data transformation and visualization generation
via conversational interface with Claude.
"""
from .exceptions import AIChatError, AIClientError, SandboxError
from .ai_client import AIClientService
from .schema_extractor import SchemaExtractorService
from .sandbox import SandboxService, SandboxResult, validate_code
from .data_transformer import DataTransformerService, TransformationResult
from .viz_generator import VizGeneratorService
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
    "VizGeneratorService",
    "TOOL_DEFINITIONS",
    "execute_tool",
    "data_transform_system_prompt",
    "data_questions_system_prompt",
    "build_transform_context",
]
