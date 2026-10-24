"""
LipidCruncher Questions: the AI chat workflow's conversational tab about
the app. Pure logic — no Streamlit dependencies.
"""
from typing import Optional

from ..models.chat import AIConfig, ChatMessage, ChatMode, ChatSession
from ..services.ai_chat.ai_client import AIClientService
from ..services.ai_chat.codebase_tools import TOOL_DEFINITIONS, execute_tool
from ..services.ai_chat.prompt_templates import lipidcruncher_questions_system_prompt


class LCQuestionsWorkflow:
    """The LipidCruncher Questions tab: a conversation about the app itself,
    with codebase and documentation tools, no code execution. Static methods
    only; AIChatWorkflow inherits them."""

    @staticmethod
    def start_lc_questions_session() -> ChatSession:
        """Initialize a new LipidCruncher questions chat session.

        No DataFrame needed — this tab answers questions about the app itself.

        Returns:
            A new ChatSession in LIPIDCRUNCHER_QUESTIONS mode.
        """
        return ChatSession(
            mode=ChatMode.LIPIDCRUNCHER_QUESTIONS,
            schema_context="",
        )

    @staticmethod
    def stream_lc_questions_message(
        user_message: str,
        session: ChatSession,
        ai_config: Optional[AIConfig] = None,
    ):
        """Handle one user message in the LipidCruncher questions chat.

        Purely conversational — no code execution; the AI may search the
        codebase and docs with tools. Appends the user message to the
        session, then returns a generator that streams the assistant's
        answer token-by-token. The caller (UI) consumes the generator —
        e.g. via ``st.write_stream`` — and appends the assistant message to
        the session from the accumulated text.

        Returns:
            Tuple of (session_with_user_message, text_chunk_generator).
        """
        if ai_config is None:
            ai_config = AIConfig()

        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        history = session.to_api_messages()[:-1]
        messages = history + [{"role": "user", "content": user_message}]
        system_prompt = lipidcruncher_questions_system_prompt()

        generator = AIClientService.stream_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=ai_config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
        )
        return session, generator
