"""
Data Questions: the AI chat workflow's code execution and answer writing
for the Data Questions tab. Pure logic — no Streamlit dependencies.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from ..models.chat import AIConfig, ChatMessage, ChatMode, ChatSession, SandboxConfig
from ..models.experiment import ExperimentConfig
from ..services.ai_chat.ai_client import AIClientService
from ..services.ai_chat.codebase_tools import TOOL_DEFINITIONS, execute_tool
from ..services.ai_chat.exceptions import AIClientError
from ..services.ai_chat.output_checks import check_lipid_table
from ..services.ai_chat.prompt_templates import (
    answer_from_results_system_prompt,
    build_answer_request,
    build_experiment_context,
    data_questions_system_prompt,
)
from ..services.ai_chat.sandbox import SandboxService
from ..services.ai_chat.schema_extractor import SchemaExtractorService
from ..services.format_detection import DataFormat, FormatDetectionService
from .ai_chat_shared import _methods_note, _repeated_lipids_note, _unverified_numbers
from .pipeline_api import PipelineAPI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQuestionsStepResult:
    """Result of a single data questions conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        assistant_message: The full assistant response text for display.
        result_df: Optional DataFrame if the AI computed a summary table.
        code: Optional code that was executed to compute the answer.
        unverified_numbers: Numbers the explanation states that the code
            never produced (logged, not shown; see ``_unverified_numbers``).
    """
    session: ChatSession
    assistant_message: str = ""
    result_df: Optional[pd.DataFrame] = None
    code: str = ""
    unverified_numbers: Tuple[str, ...] = ()


def _answer_from_results(
    question: str,
    plan: str,
    code: str,
    stdout: str,
    result_df: Optional[pd.DataFrame],
    ai_config: AIConfig,
) -> Optional[str]:
    """A second, small call that writes the Data Questions answer from the
    code's actual output, so its numbers are results, not guesses.

    None if the call fails: the turn then keeps the plan and the printed
    output, as before, rather than losing a computed result.
    """
    table = None
    if result_df is not None:
        shown = result_df.head(50)
        table = (
            f"{len(result_df)} rows x {result_df.shape[1]} columns"
            + (", first 50 shown" if len(result_df) > 50 else "") + "\n"
            + shown.to_csv(
                index=not isinstance(result_df.index, pd.RangeIndex),
            )[:8000]
        )
    try:
        answer = AIClientService.complete(
            messages=[{"role": "user", "content": build_answer_request(
                question, plan, code, stdout, table,
            )}],
            system_prompt=answer_from_results_system_prompt(),
            config=ai_config.model_copy(update={"max_tokens": 1024}),
        )
    except AIClientError as e:
        logger.warning("Writing the answer from results failed: %s", e)
        return None
    return answer.strip() or None


class DataQuestionsWorkflow:
    """The Data Questions tab: the model's pandas code run in the sandbox,
    then the answer written from what it printed. Static methods only;
    AIChatWorkflow inherits them."""

    @staticmethod
    def start_data_questions_session(
        df: pd.DataFrame,
        format_label_override: Optional[str] = None,
    ) -> ChatSession:
        """Initialize a new data questions chat session.

        Args:
            df: The user's uploaded DataFrame.
            format_label_override: Optional pre-known format label to use
                instead of re-running detection on `df`. Use this for
                pipeline imports, where `df` is the standardized /
                normalized DataFrame and detection on it would lose the
                user's original format choice (e.g. LipidSearch raw with
                renamed columns matches Generic by structure).

        Returns:
            A new ChatSession in DATA_QUESTIONS mode with schema context.
        """
        if format_label_override is not None:
            format_label = format_label_override
        else:
            detected = FormatDetectionService.detect_format(df)
            # User-facing label: avoid the bare word "Unknown" — readers
            # are non-technical lipidomics researchers, and "non-standard"
            # signals "doesn't match a recognized format" without sounding
            # like a system error.
            format_label = (
                'non-standard' if detected == DataFormat.UNKNOWN
                else detected.value
            )
        schema = SchemaExtractorService.extract_schema(df)
        schema_context = (
            f"Auto-detected format: {format_label}\n\n{schema}"
            + _repeated_lipids_note(df)
        )
        return ChatSession(
            mode=ChatMode.DATA_QUESTIONS,
            schema_context=schema_context,
        )

    @staticmethod
    def _finalize_data_questions(
        session: ChatSession,
        response_text: str,
        source_df: pd.DataFrame,
        sandbox_config: SandboxConfig,
        numeric_columns: Optional[List[str]],
        experiment: Optional[ExperimentConfig] = None,
        ai_config: Optional[AIConfig] = None,
        bqc_label: Optional[str] = None,
    ) -> DataQuestionsStepResult:
        """Extract/execute any code in the AI response, have the answer
        written from the code's output, append it to the session, and
        return the step result.
        """
        explanation, code = AIClientService.extract_code_block(response_text)
        result_df = None
        stdout = ""
        answer = None

        if code:
            sandbox_result = SandboxService.execute(
                code, source_df, sandbox_config,
                numeric_columns=numeric_columns,
                lc=PipelineAPI(source_df, experiment, bqc_label),
            )
            if sandbox_result.success:
                stdout = sandbox_result.stdout
            table_errors = (
                check_lipid_table(sandbox_result.result_df, source_df)
                if sandbox_result.success
                and sandbox_result.result_df is not None
                else []
            )
            if table_errors:
                # Not an answer: the message goes back to the model on the
                # next turn, as a failed transform's does.
                assistant_text = (
                    f"{explanation}\n\n"
                    f"I computed a table, but it failed a check, so it is not "
                    f"shown: {' '.join(table_errors)}"
                )
            elif sandbox_result.success:
                result_df = sandbox_result.result_df
                # The explanation was written before the code ran; the
                # answer is written from what the code actually produced.
                question = session.messages[-1].content if session.messages else ""
                answer = _answer_from_results(
                    question, explanation, code,
                    stdout, result_df, ai_config or AIConfig(),
                )
                parts = []
                if answer:
                    parts.append(answer)
                elif explanation.strip():
                    parts.append(explanation.strip())
                if sandbox_result.stdout.strip():
                    # Fence it: printed output is rendered as markdown, which
                    # collapses the newlines and column padding a printed
                    # DataFrame depends on, turning a table into one mangled
                    # line. Matches how the transform tab shows stdout.
                    parts.append(
                        f"```\n{sandbox_result.stdout.strip()}\n```"
                    )
                # If the AI returned a bare code block (no prose) and the
                # code only assigns to result_df (no print()), parts ends
                # up empty even though we have a valid table to show.
                # Without this lead-in the empty assistant_text triggers
                # the "I wasn't able to generate a response" fallback,
                # which contradicts the table rendered alongside it.
                if not parts and result_df is not None:
                    parts.append("Here's the result:")
                if sandbox_result.methods:
                    parts.append(_methods_note(sandbox_result.methods))
                assistant_text = "\n\n".join(parts)
            else:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"I tried to compute the answer but encountered an error: "
                    f"{sandbox_result.error_message}"
                )
        else:
            assistant_text = explanation
            code = ""

        if not assistant_text.strip():
            assistant_text = (
                "I wasn't able to generate a response. "
                "Please try rephrasing your question."
            )

        assistant_msg = ChatMessage(
            role="assistant",
            content=assistant_text,
            code=code if code else None,
        )
        session = session.add_message(assistant_msg)

        return DataQuestionsStepResult(
            session=session,
            assistant_message=assistant_text,
            result_df=result_df,
            code=code,
            unverified_numbers=_unverified_numbers(
                session, f"{explanation}\n\n{answer or ''}", code,
                stdout=stdout, result_df=result_df,
            ),
        )

    @staticmethod
    def stream_data_questions_message(
        user_message: str,
        session: ChatSession,
        source_df: pd.DataFrame,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
        experiment: Optional[ExperimentConfig] = None,
        bqc_label: Optional[str] = None,
    ):
        """Handle one user message in the data questions chat.

        Appends the user message, then returns ``(explanation_generator,
        finalize)``. The caller streams the explanation, then calls
        ``finalize()`` to execute any code and obtain a
        ``DataQuestionsStepResult``.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        context = f"<dataframe_schema>\n{session.schema_context}\n</dataframe_schema>"
        exp_context = build_experiment_context(experiment)
        if exp_context:
            context = f"{context}\n\n{exp_context}"
        augmented_message = f"{context}\n\n{user_message}"
        history = session.to_api_messages()[:-1]
        messages = history + [{"role": "user", "content": augmented_message}]
        system_prompt = data_questions_system_prompt()

        holder = {"full": ""}
        generator = AIClientService.stream_explanation(
            messages=messages,
            system_prompt=system_prompt,
            config=ai_config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            holder=holder,
        )

        def finalize() -> DataQuestionsStepResult:
            return DataQuestionsWorkflow._finalize_data_questions(
                session, holder["full"], source_df,
                sandbox_config, numeric_columns, experiment, ai_config,
                bqc_label=bqc_label,
            )

        return generator, finalize
