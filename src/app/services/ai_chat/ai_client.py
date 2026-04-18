"""
Anthropic API client wrapper.

Stateless static methods for communicating with the Claude API.
All methods are pure — no Streamlit imports, no instance state.
"""
import logging
import os
import re
from typing import Any, Callable, List, Optional, Tuple

from ...models.chat import AIConfig
from .exceptions import AIClientError

logger = logging.getLogger(__name__)

# Default cap on tool-use round-trips per user message
_DEFAULT_MAX_TOOL_ITERATIONS = 5


class AIClientService:
    """Static wrapper around the Anthropic Python SDK.

    Handles client creation, message sending, code extraction,
    and tool-use loop execution.
    """

    @staticmethod
    def create_client():
        """Create an Anthropic client from the ANTHROPIC_API_KEY env var.

        Returns:
            anthropic.Anthropic client instance.

        Raises:
            AIClientError: If the API key is not set or the SDK is not installed.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AIClientError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Please set it to your Anthropic API key."
            )
        try:
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise AIClientError(
                "The 'anthropic' package is not installed. "
                "Install it with: pip install anthropic"
            )

    # ------------------------------------------------------------------
    # Original (non-tool) methods — kept for backward compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def send_message(
        messages: list,
        system_prompt: str,
        config: AIConfig,
    ) -> str:
        """Send a conversation to the Claude API and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: System prompt providing context and instructions.
            config: AI configuration (model, max_tokens, temperature).

        Returns:
            Response text from the AI.

        Raises:
            AIClientError: On any API communication failure.
        """
        try:
            client = AIClientService.create_client()
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
        except AIClientError:
            raise
        except Exception as e:
            raise AIClientError(f"API call failed: {e}") from e

    @staticmethod
    def send_message_for_code(
        messages: list,
        system_prompt: str,
        config: AIConfig,
    ) -> Tuple[str, str]:
        """Send a conversation and extract a code block from the response.

        The AI is expected to return an explanation followed by a Python
        code block delimited by ```python ... ```.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: System prompt providing context and instructions.
            config: AI configuration (model, max_tokens, temperature).

        Returns:
            Tuple of (explanation_text, code_block). If no code block is
            found, code_block is an empty string.

        Raises:
            AIClientError: On any API communication failure.
        """
        response_text = AIClientService.send_message(messages, system_prompt, config)
        explanation, code = AIClientService.extract_code_block(response_text)
        return explanation, code

    @staticmethod
    def extract_code_block(text: str) -> Tuple[str, str]:
        """Extract a Python code block from AI response text.

        Looks for ```python ... ``` delimiters. If multiple blocks exist,
        returns the first one.

        Args:
            text: Full AI response text.

        Returns:
            Tuple of (text_without_code_block, code_block).
            If no code block found, returns (text, "").
        """
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return text, ""

        code = match.group(1).strip()
        explanation = text[:match.start()].strip()
        return explanation, code

    # ------------------------------------------------------------------
    # Tool-aware methods
    # ------------------------------------------------------------------

    @staticmethod
    def send_message_with_tools(
        messages: list,
        system_prompt: str,
        config: AIConfig,
        tools: List[dict],
        tool_choice: Optional[dict] = None,
    ):
        """Send a conversation with tool definitions and return the full response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: System prompt providing context and instructions.
            config: AI configuration (model, max_tokens, temperature).
            tools: List of tool definition dicts (Anthropic API format).
            tool_choice: Optional tool_choice parameter (e.g. {"type": "auto"}).

        Returns:
            The full Anthropic Message object (not just the text).

        Raises:
            AIClientError: On any API communication failure.
        """
        try:
            client = AIClientService.create_client()
            kwargs: dict[str, Any] = dict(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            return client.messages.create(**kwargs)
        except AIClientError:
            raise
        except Exception as e:
            raise AIClientError(f"API call with tools failed: {e}") from e

    @staticmethod
    def extract_tool_use_blocks(response) -> List[dict]:
        """Extract tool_use content blocks from an API response.

        Args:
            response: Anthropic Message object.

        Returns:
            List of dicts with 'id', 'name', and 'input' keys for each
            tool_use block in the response.
        """
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return tool_calls

    @staticmethod
    def extract_text_from_response(response) -> str:
        """Extract concatenated text content from an API response.

        Args:
            response: Anthropic Message object.

        Returns:
            All text blocks joined together. Empty string if no text blocks.
        """
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def execute_tool_loop(
        messages: list,
        system_prompt: str,
        config: AIConfig,
        tools: List[dict],
        tool_executor: Callable[[str, dict], str],
        max_iterations: int = _DEFAULT_MAX_TOOL_ITERATIONS,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ) -> str:
        """Run the tool-use loop until the AI produces a final text response.

        Sends the conversation with tools. If the AI requests a tool call,
        executes it locally via ``tool_executor``, appends the result, and
        calls the API again — repeating until the AI responds with text only
        (stop_reason == "end_turn") or the iteration cap is reached.

        Args:
            messages: Conversation history (will NOT be mutated — a copy is used).
            system_prompt: System prompt.
            config: AI configuration.
            tools: Tool definitions for the API.
            tool_executor: Callable(name, input_dict) -> result_string.
                This is typically ``codebase_tools.execute_tool``.
            max_iterations: Maximum number of API round-trips (default 5).
            on_tool_call: Optional callback invoked with (tool_name, tool_input)
                before each tool execution. Useful for UI status updates.

        Returns:
            The final text response from the AI (after all tool calls complete).

        Raises:
            AIClientError: On any API communication failure.
        """
        # Work on a copy so caller's list is not mutated
        loop_messages = list(messages)

        for iteration in range(max_iterations):
            response = AIClientService.send_message_with_tools(
                loop_messages, system_prompt, config, tools,
            )

            # If the AI didn't request any tools, we're done
            tool_calls = AIClientService.extract_tool_use_blocks(response)
            if not tool_calls:
                return AIClientService.extract_text_from_response(response)

            # Append the assistant's response (with tool_use blocks) to messages
            loop_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Execute each tool and build tool_result blocks
            tool_results = []
            for tool_call in tool_calls:
                if on_tool_call is not None:
                    on_tool_call(tool_call["name"], tool_call["input"])

                try:
                    result_text = tool_executor(
                        tool_call["name"], tool_call["input"]
                    )
                except Exception as e:
                    logger.warning("Tool %s failed: %s", tool_call["name"], e)
                    result_text = f"Error executing tool: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result_text,
                })

            # Append tool results as user message
            loop_messages.append({
                "role": "user",
                "content": tool_results,
            })

        # If we exhausted iterations, make one final call without tools
        # to force a text response
        logger.warning(
            "Tool loop reached %d iterations — forcing final response",
            max_iterations,
        )
        return AIClientService.send_message(loop_messages, system_prompt, config)
