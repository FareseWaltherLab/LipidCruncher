"""
Anthropic API client wrapper.

Stateless static methods for communicating with the Claude API.
All methods are pure — no Streamlit imports, no instance state.
"""
import os
import re
from typing import Tuple

from ...models.chat import AIConfig
from .exceptions import AIClientError


class AIClientService:
    """Static wrapper around the Anthropic Python SDK.

    Handles client creation, message sending, and code extraction
    from AI responses.
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
