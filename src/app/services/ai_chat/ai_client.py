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
_DEFAULT_MAX_TOOL_ITERATIONS = 4

# Client tuning. The SDK already retries 429/5xx/connection errors with
# exponential backoff; we bump the default (2) slightly and cap the per-request
# wait so a hung call surfaces as an error instead of blocking the UI forever.
_MAX_RETRIES = 4
_REQUEST_TIMEOUT_SECONDS = 120.0


def _friendly_error(exc: Exception, context: str = "") -> AIClientError:
    """Map a low-level Anthropic SDK exception to a user-friendly AIClientError.

    Recognized failure modes (rate limit, timeout, connection, auth, server
    error) get a short, actionable message suitable for display. Unrecognized
    exceptions fall back to the raw error text (with an optional ``context``
    prefix) so debugging detail is never lost.
    """
    try:
        import anthropic
    except ImportError:
        anthropic = None

    if anthropic is not None:
        # Subclass order matters: AuthenticationError / RateLimitError are
        # APIStatusError subclasses, and APITimeoutError is an
        # APIConnectionError subclass — check the specific types first.
        if isinstance(exc, anthropic.AuthenticationError):
            return AIClientError(
                "The AI service rejected the API key. Please check that "
                "ANTHROPIC_API_KEY is set correctly."
            )
        if isinstance(exc, anthropic.RateLimitError):
            return AIClientError(
                "The AI is busy right now (rate limit reached). Please wait a "
                "few seconds and try again."
            )
        if isinstance(exc, anthropic.APITimeoutError):
            return AIClientError(
                "The AI took too long to respond. Please try again."
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return AIClientError(
                "Couldn't reach the AI service. Please check your internet "
                "connection and try again."
            )
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", None)
            if status is not None and status >= 500:
                return AIClientError(
                    "The AI service is temporarily unavailable. Please try "
                    "again in a moment."
                )
            return AIClientError(
                f"The AI service returned an error (HTTP {status}). "
                "Please try again."
            )

    prefix = f"{context}: " if context else ""
    return AIClientError(
        f"{prefix}Unexpected error contacting the AI service: {exc}"
    )


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
            return anthropic.Anthropic(
                api_key=api_key,
                max_retries=_MAX_RETRIES,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except ImportError:
            raise AIClientError(
                "The 'anthropic' package is not installed. "
                "Install it with: pip install anthropic"
            )

    @staticmethod
    def _system_blocks(system_prompt: str) -> list:
        """Wrap the system prompt in a cache-enabled content block.

        The system prompt (role + app knowledge + few-shot examples) is a
        large static prefix reused across every turn and every tool-loop
        round-trip, so we mark it for prompt caching to cut latency and cost.
        The breakpoint also caches the preceding tool definitions.
        """
        return [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

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
                system=AIClientService._system_blocks(system_prompt),
                messages=messages,
                tools=tools,
            )
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            return client.messages.create(**kwargs)
        except AIClientError:
            raise
        except Exception as e:
            raise _friendly_error(e) from e

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
            logger.info("Tool loop iteration %d/%d", iteration + 1, max_iterations)
            response = AIClientService.send_message_with_tools(
                loop_messages, system_prompt, config, tools,
            )

            # Log what we got back
            block_types = [b.type for b in response.content]
            logger.info("Response blocks: %s, stop_reason: %s",
                        block_types, response.stop_reason)

            # If the AI didn't request any tools, we're done
            tool_calls = AIClientService.extract_tool_use_blocks(response)
            if not tool_calls:
                text = AIClientService.extract_text_from_response(response)
                logger.info("Final text response: %d chars", len(text))
                return text

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

        # If we exhausted iterations, make one final call with tools still
        # present (required because the conversation contains tool_result
        # messages) but ask the model to stop using them.
        logger.warning(
            "Tool loop reached %d iterations — forcing final response",
            max_iterations,
        )
        loop_messages.append({
            "role": "user",
            "content": (
                "Please provide your final answer now based on what you've "
                "found so far. Do not make any more tool calls."
            ),
        })
        response = AIClientService.send_message_with_tools(
            loop_messages, system_prompt, config, tools,
        )
        return AIClientService.extract_text_from_response(response)

    @staticmethod
    def stream_tool_loop(
        messages: list,
        system_prompt: str,
        config: AIConfig,
        tools: List[dict],
        tool_executor: Callable[[str, dict], str],
        max_iterations: int = _DEFAULT_MAX_TOOL_ITERATIONS,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ):
        """Streaming variant of ``execute_tool_loop``.

        Runs the same tool-use loop, but streams each model turn and yields
        text chunks as they arrive (suitable for ``st.write_stream``). Tool
        calls between turns are executed locally via ``tool_executor`` and
        appended, then the loop continues. Yields only ``str`` chunks — no
        Streamlit, no session state — so the layer boundary is preserved.

        Yields:
            Text fragments from the assistant as they are generated.
        """
        client = AIClientService.create_client()
        loop_messages = list(messages)

        def _stream_turn():
            try:
                with client.messages.stream(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    system=AIClientService._system_blocks(system_prompt),
                    messages=loop_messages,
                    tools=tools,
                ) as stream:
                    for text in stream.text_stream:
                        yield text
                    _stream_turn.final = stream.get_final_message()
            except AIClientError:
                raise
            except Exception as e:
                raise _friendly_error(e, context="streaming") from e

        for iteration in range(max_iterations):
            logger.info(
                "Streaming tool loop iteration %d/%d", iteration + 1, max_iterations
            )
            yield from _stream_turn()
            final = _stream_turn.final

            loop_messages.append({"role": "assistant", "content": final.content})

            tool_calls = AIClientService.extract_tool_use_blocks(final)
            if not tool_calls:
                return

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
            loop_messages.append({"role": "user", "content": tool_results})

        # Iterations exhausted with tools still pending — force a final answer.
        logger.warning(
            "Streaming tool loop reached %d iterations — forcing final response",
            max_iterations,
        )
        loop_messages.append({
            "role": "user",
            "content": (
                "Please provide your final answer now based on what you've "
                "found so far. Do not make any more tool calls."
            ),
        })
        yield from _stream_turn()

    @staticmethod
    def _filter_to_explanation(chunks, holder: dict):
        """Yield only the prose before the first ``` code fence.

        Consumes raw text ``chunks`` and accumulates the COMPLETE raw text
        (explanation + code) into ``holder['full']`` for later code
        extraction, while yielding only the explanation portion for live
        display. Once a code fence appears, nothing more is yielded.
        """
        yielded = 0
        stopped = False
        for chunk in chunks:
            holder["full"] += chunk
            if stopped:
                continue
            text = holder["full"]
            fence = text.find("```")
            if fence != -1:
                if fence > yielded:
                    yield text[yielded:fence]
                yielded = fence
                stopped = True
            else:
                # Hold back a trailing run of backticks — it may be the
                # start of a fence split across chunk boundaries.
                end = len(text)
                while end > yielded and text[end - 1] == "`":
                    end -= 1
                if end > yielded:
                    yield text[yielded:end]
                    yielded = end
        if not stopped and len(holder["full"]) > yielded:
            yield holder["full"][yielded:]

    @staticmethod
    def stream_explanation(
        messages: list,
        system_prompt: str,
        config: AIConfig,
        tools: List[dict],
        tool_executor: Callable[[str, dict], str],
        holder: dict,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ):
        """Stream a tool-loop response, yielding only the explanation prose.

        Wraps ``stream_tool_loop`` and filters its output so the caller sees
        only the text before the code block (suitable for ``st.write_stream``).
        The complete raw response (explanation + code) is accumulated into
        ``holder['full']`` so the caller can extract and execute the code
        after streaming finishes.
        """
        raw = AIClientService.stream_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=config,
            tools=tools,
            tool_executor=tool_executor,
            on_tool_call=on_tool_call,
        )
        yield from AIClientService._filter_to_explanation(raw, holder)
