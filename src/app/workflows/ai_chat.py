"""
AI chat workflow.

Orchestrates AI-powered data transformation, visualization
generation, and data questions. Pure logic — no Streamlit dependencies.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..models.chat import (
    AIConfig,
    ChatMessage,
    ChatMode,
    ChatSession,
    SandboxConfig,
)
from ..models.experiment import ExperimentConfig
from ..services.ai_chat.ai_client import AIClientService
from ..services.ai_chat.codebase_tools import TOOL_DEFINITIONS, execute_tool
from ..services.ai_chat.data_transformer import DataTransformerService, TransformationResult
from ..services.ai_chat.prompt_templates import data_questions_system_prompt, unified_system_prompt
from ..services.ai_chat.sandbox import SandboxResult, SandboxService
from ..services.ai_chat.schema_extractor import SchemaExtractorService
from ..services.ai_chat.viz_generator import VizGeneratorService


@dataclass(frozen=True)
class TransformStepResult:
    """Result of a single transform conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        transformation: Transformation result (None if AI only asked a question).
        assistant_message: The full assistant response text for display.
    """
    session: ChatSession
    transformation: Optional[TransformationResult] = None
    assistant_message: str = ""


@dataclass(frozen=True)
class VizStepResult:
    """Result of a single visualization conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        sandbox_result: Sandbox execution result (None if AI only asked a question).
        assistant_message: The full assistant response text for display.
    """
    session: ChatSession
    sandbox_result: Optional[SandboxResult] = None
    assistant_message: str = ""


@dataclass(frozen=True)
class DataQuestionsStepResult:
    """Result of a single data questions conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        assistant_message: The full assistant response text for display.
        result_df: Optional DataFrame if the AI computed a summary table.
        code: Optional code that was executed to compute the answer.
    """
    session: ChatSession
    assistant_message: str = ""
    result_df: Optional[pd.DataFrame] = None
    code: str = ""


@dataclass(frozen=True)
class UnifiedStepResult:
    """Result of a single unified chat conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        assistant_message: The full assistant response text for display.
        result_df: Optional DataFrame (summary table or transformation result).
        figure: Optional Plotly figure.
        code: Code that was executed (if any).
        is_transformation: True if result_df is a Generic format transformation.
    """
    session: ChatSession
    assistant_message: str = ""
    result_df: Optional[pd.DataFrame] = None
    figure: object = None
    code: str = ""
    is_transformation: bool = False


class AIChatWorkflow:
    """Orchestrates AI chat interactions.

    All methods are stateless static methods. No Streamlit imports.
    """

    @staticmethod
    def start_transform_session(source_df: pd.DataFrame) -> ChatSession:
        """Initialize a new data transformation chat session.

        Pre-loads the DataFrame schema into the session context so
        the AI has structural information without raw data.

        Args:
            source_df: The user's uploaded DataFrame.

        Returns:
            A new ChatSession in DATA_TRANSFORM mode with schema context.
        """
        schema = SchemaExtractorService.extract_schema(source_df)
        return ChatSession(
            mode=ChatMode.DATA_TRANSFORM,
            schema_context=schema,
        )

    @staticmethod
    def handle_transform_message(
        user_message: str,
        session: ChatSession,
        source_df: pd.DataFrame,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> TransformStepResult:
        """Handle one user message in a data transformation chat.

        Flow:
        1. Append user message to session.
        2. Send conversation to AI to generate transformation code.
        3. If code is returned, execute in sandbox and validate output.
        4. Append assistant response to session.
        5. Return updated session + transformation result.

        Args:
            user_message: The user's latest message.
            session: Current chat session.
            source_df: The user's original DataFrame.
            ai_config: AI configuration. Uses defaults if None.
            sandbox_config: Sandbox configuration. Uses defaults if None.

        Returns:
            TransformStepResult with updated session and transformation outcome.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        # 1. Append user message
        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        # 2. Generate transformation code via AI
        history = session.to_api_messages()[:-1]  # exclude the latest (we pass it separately)

        explanation, code = DataTransformerService.generate_transformation_code(
            source_schema=session.schema_context,
            user_message=user_message,
            history=history,
            config=ai_config,
        )

        # 3. If code was generated, execute and validate
        transformation = None
        if code:
            transformation = DataTransformerService.execute_transformation(
                code=code,
                source_df=source_df,
                sandbox_config=sandbox_config,
            )
            # Include explanation in the transformation result
            transformation = TransformationResult(
                success=transformation.success,
                result_df=transformation.result_df,
                explanation=explanation,
                code=code,
                sandbox_result=transformation.sandbox_result,
                validation_errors=transformation.validation_errors,
            )

            # Build assistant response text
            if transformation.success:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"The transformation was successful! "
                    f"The result has {len(transformation.result_df)} rows and "
                    f"{len(transformation.result_df.columns)} columns. "
                    f"You can download the transformed file below."
                )
            else:
                error_summary = "; ".join(transformation.validation_errors)
                assistant_text = (
                    f"{explanation}\n\n"
                    f"The transformation encountered issues: {error_summary}\n\n"
                    f"Please describe what needs to be fixed and I'll adjust the code."
                )
        else:
            # AI responded with a question or explanation, no code
            assistant_text = explanation

        # 4. Append assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=assistant_text,
            code=code if code else None,
        )
        session = session.add_message(assistant_msg)

        return TransformStepResult(
            session=session,
            transformation=transformation,
            assistant_message=assistant_text,
        )

    # ------------------------------------------------------------------
    # Visualization mode
    # ------------------------------------------------------------------

    @staticmethod
    def start_viz_session(
        analysis_df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> ChatSession:
        """Initialize a new visualization chat session.

        Pre-loads the DataFrame schema into the session context.

        Args:
            analysis_df: The analysis-ready DataFrame.
            experiment: Experiment configuration (conditions, samples).

        Returns:
            A new ChatSession in VISUALIZATION mode with schema context.
        """
        schema = SchemaExtractorService.extract_schema(analysis_df)
        return ChatSession(
            mode=ChatMode.VISUALIZATION,
            schema_context=schema,
        )

    @staticmethod
    def handle_viz_message(
        user_message: str,
        session: ChatSession,
        analysis_df: pd.DataFrame,
        experiment: ExperimentConfig,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> VizStepResult:
        """Handle one user message in a visualization chat.

        Flow:
        1. Append user message to session.
        2. Send conversation to AI to generate visualization code.
        3. If code is returned, execute in sandbox.
        4. Append assistant response to session.
        5. Return updated session + sandbox result.

        Args:
            user_message: The user's latest message.
            session: Current chat session.
            analysis_df: The analysis-ready DataFrame.
            experiment: Experiment configuration.
            ai_config: AI configuration. Uses defaults if None.
            sandbox_config: Sandbox configuration. Uses defaults if None.

        Returns:
            VizStepResult with updated session and execution outcome.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        # 1. Append user message
        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        # 2. Generate viz code via AI
        history = session.to_api_messages()[:-1]

        explanation, code = VizGeneratorService.generate_viz_code(
            user_request=user_message,
            schema_context=session.schema_context,
            experiment=experiment,
            history=history,
            config=ai_config,
        )

        # 3. If code was generated, execute in sandbox
        sandbox_result = None
        if code:
            sandbox_result = VizGeneratorService.execute_viz_code(
                code=code,
                df=analysis_df,
                sandbox_config=sandbox_config,
            )

            if sandbox_result.success and sandbox_result.figure is not None:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"Here's the visualization you requested."
                )
            elif sandbox_result.success and sandbox_result.figure is None:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"The code executed but didn't produce a figure. "
                    f"Make sure the code assigns a Plotly figure to `fig`."
                )
            else:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"The visualization code encountered an error: "
                    f"{sandbox_result.error_message}\n\n"
                    f"Please describe what you'd like changed and I'll fix it."
                )
        else:
            assistant_text = explanation

        # 4. Append assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=assistant_text,
            code=code if code else None,
        )
        session = session.add_message(assistant_msg)

        return VizStepResult(
            session=session,
            sandbox_result=sandbox_result,
            assistant_message=assistant_text,
        )

    # ------------------------------------------------------------------
    # Data Questions mode
    # ------------------------------------------------------------------

    @staticmethod
    def start_data_questions_session(df: pd.DataFrame) -> ChatSession:
        """Initialize a new data questions chat session.

        Args:
            df: The user's uploaded DataFrame.

        Returns:
            A new ChatSession in DATA_QUESTIONS mode with schema context.
        """
        schema = SchemaExtractorService.extract_schema(df)
        return ChatSession(
            mode=ChatMode.DATA_QUESTIONS,
            schema_context=schema,
        )

    @staticmethod
    def handle_data_questions_message(
        user_message: str,
        session: ChatSession,
        source_df: pd.DataFrame,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> DataQuestionsStepResult:
        """Handle one user message in a data questions chat.

        Flow:
        1. Append user message to session.
        2. Send conversation to AI (with tools) to answer the question.
        3. If code is returned, execute in sandbox to compute the answer.
        4. Append assistant response to session.
        5. Return updated session + optional result DataFrame.

        Args:
            user_message: The user's latest message.
            session: Current chat session.
            source_df: The user's DataFrame.
            ai_config: AI configuration. Uses defaults if None.
            sandbox_config: Sandbox configuration. Uses defaults if None.

        Returns:
            DataQuestionsStepResult with updated session and answer.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        # 1. Append user message
        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        # 2. Build context and call AI with tools
        context = f"## DataFrame Schema\n{session.schema_context}"
        augmented_message = f"{user_message}\n\n{context}"
        history = session.to_api_messages()[:-1]
        messages = history + [{"role": "user", "content": augmented_message}]

        system_prompt = data_questions_system_prompt()

        response_text = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=ai_config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
        )

        # 3. Extract and execute code if present
        explanation, code = AIClientService.extract_code_block(response_text)
        result_df = None

        if code:
            sandbox_result = SandboxService.execute(code, source_df, sandbox_config)
            if sandbox_result.success and sandbox_result.result_df is not None:
                result_df = sandbox_result.result_df
                assistant_text = explanation
            elif sandbox_result.success:
                # Code ran but no result_df — just show the explanation
                assistant_text = explanation
            else:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"I tried to compute the answer but encountered an error: "
                    f"{sandbox_result.error_message}"
                )
        else:
            assistant_text = explanation
            code = ""

        # 4. Append assistant message
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
        )

    # ------------------------------------------------------------------
    # Unified mode (single chat for all capabilities)
    # ------------------------------------------------------------------

    @staticmethod
    def start_unified_session(df: pd.DataFrame) -> ChatSession:
        """Initialize a unified AI Studio chat session.

        Args:
            df: The user's uploaded DataFrame.

        Returns:
            A new ChatSession in DATA_QUESTIONS mode with schema context.
        """
        schema = SchemaExtractorService.extract_schema(df)
        return ChatSession(
            mode=ChatMode.DATA_QUESTIONS,
            schema_context=schema,
        )

    @staticmethod
    def handle_unified_message(
        user_message: str,
        session: ChatSession,
        source_df: pd.DataFrame,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> UnifiedStepResult:
        """Handle one user message in the unified chat.

        The AI decides whether to answer a question, transform data,
        or generate a visualization based on the user's request.

        Args:
            user_message: The user's latest message.
            session: Current chat session.
            source_df: The user's DataFrame.
            ai_config: AI configuration. Uses defaults if None.
            sandbox_config: Sandbox configuration. Uses defaults if None.

        Returns:
            UnifiedStepResult with updated session and results.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        # 1. Append user message
        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        # 2. Build context and call AI with tools
        context = f"## DataFrame Schema\n{session.schema_context}"
        augmented_message = f"{user_message}\n\n{context}"
        history = session.to_api_messages()[:-1]
        messages = history + [{"role": "user", "content": augmented_message}]

        system_prompt = unified_system_prompt()

        response_text = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=ai_config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
        )

        # 3. Extract and execute code if present
        explanation, code = AIClientService.extract_code_block(response_text)
        result_df = None
        figure = None
        is_transformation = False

        if code:
            sandbox_result = SandboxService.execute(code, source_df, sandbox_config)

            if sandbox_result.success:
                # Check what was produced
                if sandbox_result.figure is not None:
                    figure = sandbox_result.figure
                    assistant_text = explanation
                elif sandbox_result.result_df is not None:
                    result_df = sandbox_result.result_df
                    # Detect if this is a transformation (Generic format)
                    is_transformation = _looks_like_generic_format(result_df)
                    if is_transformation:
                        assistant_text = (
                            f"{explanation}\n\n"
                            f"The transformation was successful! "
                            f"The result has {len(result_df)} rows and "
                            f"{len(result_df.columns)} columns. "
                            f"You can download the transformed file below."
                        )
                    else:
                        assistant_text = explanation
                else:
                    assistant_text = explanation
            else:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"I encountered an error: {sandbox_result.error_message}"
                )
        else:
            assistant_text = explanation
            code = ""

        # 4. Append assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=assistant_text,
        )
        session = session.add_message(assistant_msg)

        return UnifiedStepResult(
            session=session,
            assistant_message=assistant_text,
            result_df=result_df,
            figure=figure,
            code=code,
            is_transformation=is_transformation,
        )


def _looks_like_generic_format(df: pd.DataFrame) -> bool:
    """Heuristic: does this DataFrame look like a Generic format transformation?

    Generic format = column 1 is strings (lipid names), columns 2+ are numeric.
    A summary table (counts, stats) typically has few rows or non-lipid-like first column.
    """
    if df.empty or len(df.columns) < 3:
        return False
    # First column should be string-like
    first_col = df.iloc[:, 0]
    if not (first_col.dtype == object or first_col.dtype.name == 'string'):
        return False
    # Remaining columns should be mostly numeric
    numeric_count = sum(
        1 for col in df.columns[1:]
        if pd.api.types.is_numeric_dtype(df[col])
    )
    # At least 2 numeric columns and most columns are numeric
    return numeric_count >= 2 and numeric_count >= (len(df.columns) - 1) * 0.7
