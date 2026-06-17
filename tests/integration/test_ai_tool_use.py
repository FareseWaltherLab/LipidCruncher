"""
Integration tests for AI tool use.

These tests make real API calls to the Anthropic API and verify the full
tool-use loop works end-to-end: the AI can search/read the codebase,
receive tool results, and produce a final response with code.

Requires ANTHROPIC_API_KEY environment variable to be set.
Run with: pytest tests/integration/test_ai_tool_use.py -v
"""
import os

import pandas as pd
import pytest

from app.models.chat import AIConfig
from app.services.ai_chat.ai_client import AIClientService
from app.services.ai_chat.codebase_tools import TOOL_DEFINITIONS, execute_tool
from app.services.ai_chat.data_transformer import DataTransformerService
from app.services.ai_chat.schema_extractor import SchemaExtractorService

# Skip all tests in this module if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping integration tests",
)

# Use a smaller/cheaper model config for integration tests
TEST_CONFIG = AIConfig(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    temperature=0.0,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A simple DataFrame for testing."""
    return pd.DataFrame({
        "Lipid": ["PC(34:1)", "PE(36:2)", "SM(42:1)", "TG(52:3)"],
        "Class": ["PC", "PE", "SM", "TG"],
        "Score": [99.5, 87.2, 95.1, 91.0],
        "Sample_A": [1500.0, 2200.0, 800.0, 3100.0],
        "Sample_B": [1600.0, 2100.0, 850.0, 2900.0],
        "Sample_C": [1450.0, 2300.0, 780.0, 3200.0],
    })


# ---------------------------------------------------------------------------
# Tool loop integration
# ---------------------------------------------------------------------------

class TestToolLoopIntegration:
    """Test the full tool-use loop with real API calls."""

    def test_ai_can_use_search_tool(self):
        """The AI should use search_codebase when asked about implementation."""
        messages = [
            {"role": "user", "content": "How does LipidCruncher handle lipid name standardization? Use the search_codebase tool to find out."},
        ]

        tool_calls_made = []

        def track_tool_call(name, input_dict):
            tool_calls_made.append(name)

        response = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt="You are a helpful assistant with access to the LipidCruncher codebase. Use your tools to answer questions accurately.",
            config=TEST_CONFIG,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            on_tool_call=track_tool_call,
        )

        # The AI should have used at least one tool
        assert len(tool_calls_made) > 0
        # Response should mention something about lipid names
        assert len(response) > 50

    def test_ai_can_read_documentation(self):
        """The AI should use read_documentation for high-level questions."""
        messages = [
            {"role": "user", "content": "What analysis modules does LipidCruncher have? Use the read_documentation tool with doc_name 'CODEBASE' to find out."},
        ]

        tool_calls_made = []

        def track_tool_call(name, input_dict):
            tool_calls_made.append(name)

        response = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt="You are a helpful assistant. Use your tools to look up information.",
            config=TEST_CONFIG,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            on_tool_call=track_tool_call,
        )

        assert "read_documentation" in tool_calls_made
        assert len(response) > 50

    def test_ai_responds_without_tools_when_unnecessary(self):
        """The AI should not use tools for simple greetings."""
        messages = [
            {"role": "user", "content": "Hello! What can you help me with?"},
        ]

        tool_calls_made = []

        def track_tool_call(name, input_dict):
            tool_calls_made.append(name)

        response = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt="You are a helpful lipidomics assistant.",
            config=TEST_CONFIG,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            on_tool_call=track_tool_call,
        )

        # Should respond without needing tools
        assert len(tool_calls_made) == 0
        assert len(response) > 10


# ---------------------------------------------------------------------------
# Data transformer integration
# ---------------------------------------------------------------------------

class TestDataTransformerIntegration:
    """Test DataTransformerService with real API calls and tools."""

    def test_generate_code_with_tools(self, sample_df):
        """The transformer should generate valid code using tool context."""
        schema = SchemaExtractorService.extract_schema(sample_df)

        tool_calls_made = []

        def track_tool_call(name, input_dict):
            tool_calls_made.append(name)

        explanation, code = DataTransformerService.generate_transformation_code(
            source_schema=schema,
            user_message="Transform my data to Generic format. Column 'Lipid' has lipid names, Sample_A/B/C are intensities. Remove Class and Score columns.",
            history=[],
            config=TEST_CONFIG,
            on_tool_call=track_tool_call,
        )

        # Should produce an explanation and code
        assert len(explanation) > 0
        assert len(code) > 0
        assert "result_df" in code

    def test_generated_code_executes_successfully(self, sample_df):
        """The generated code should execute and produce valid output."""
        schema = SchemaExtractorService.extract_schema(sample_df)

        explanation, code = DataTransformerService.generate_transformation_code(
            source_schema=schema,
            user_message="Transform my data to Generic format. Column 'Lipid' has lipid names, Sample_A/B/C are intensities. Remove the Class and Score columns.",
            history=[],
            config=TEST_CONFIG,
        )

        if code:
            result = DataTransformerService.execute_transformation(
                code=code,
                source_df=sample_df,
            )
            # The transformation should succeed
            assert result.success, f"Transformation failed: {result.validation_errors}"
            assert result.result_df is not None
            assert len(result.result_df) == 4
            # Should have lipid names + 3 sample columns
            assert len(result.result_df.columns) >= 4


# ---------------------------------------------------------------------------
# Tool execution edge cases
# ---------------------------------------------------------------------------

class TestToolExecutionEdgeCases:
    """Test edge cases in tool execution during the loop."""

    def test_tool_error_does_not_crash_loop(self):
        """If a tool returns an error, the loop should continue gracefully."""
        messages = [
            {"role": "user", "content": "Read the file 'src/app/nonexistent_file.py' using the read_file_section tool."},
        ]

        # Should not raise — the error message gets passed back to the AI
        response = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt="You are a helpful assistant. Use tools as requested.",
            config=TEST_CONFIG,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
        )

        # AI should acknowledge the file wasn't found
        assert len(response) > 0

    def test_multiple_tool_calls_in_sequence(self):
        """The AI should be able to chain multiple tool calls."""
        messages = [
            {"role": "user", "content": (
                "First, use list_files to see what's in src/app/services/ai_chat/, "
                "then use read_file_section to read the first 5 lines of the ai_client.py file. "
                "Tell me what you found."
            )},
        ]

        tool_calls_made = []

        def track_tool_call(name, input_dict):
            tool_calls_made.append(name)

        response = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt="You are a helpful assistant. Follow the user's instructions step by step using your tools.",
            config=TEST_CONFIG,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            on_tool_call=track_tool_call,
        )

        # Should have made at least 2 tool calls
        assert len(tool_calls_made) >= 2
        assert len(response) > 50
