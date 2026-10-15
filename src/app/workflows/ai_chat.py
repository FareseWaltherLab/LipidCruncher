"""
AI chat workflow.

Orchestrates AI-powered data transformation, visualization
generation, and data questions. Pure logic — no Streamlit dependencies.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

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
from ..services.ai_chat.prompt_templates import (
    build_experiment_context,
    data_questions_system_prompt,
    lipidcruncher_questions_system_prompt,
    unified_system_prompt,
)
from ..services.ai_chat.sandbox import SandboxResult, SandboxService
from .pipeline_api import PipelineAPI
from ..services.ai_chat.schema_extractor import SchemaExtractorService
from ..services.ai_chat.viz_generator import VizGeneratorService
from ..services.format_detection import (
    FormatDetectionService,
    DataFormat,
    GENERIC_METADATA_NAME_PATTERNS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormatDiagnosticResult:
    """Result of format diagnostic on uploaded data.

    Attributes:
        detected_format: The detected DataFormat.
        is_supported: True if data is already in a supported format.
        message: User-facing diagnostic message.
        lipid_name_column: Guessed lipid name column (if UNKNOWN format).
        intensity_columns: Guessed intensity columns (if UNKNOWN format).
        metadata_columns: Guessed metadata columns to remove (if UNKNOWN format).
    """
    detected_format: DataFormat
    is_supported: bool
    message: str
    lipid_name_column: Optional[str] = None
    intensity_columns: List[str] = None
    metadata_columns: List[str] = None

    def __post_init__(self):
        # Default mutable fields
        if self.intensity_columns is None:
            object.__setattr__(self, 'intensity_columns', [])
        if self.metadata_columns is None:
            object.__setattr__(self, 'metadata_columns', [])


@dataclass(frozen=True)
class Module1ValidationResult:
    """Result of validating whether data has been through Module 1.

    Attributes:
        is_valid: True if data has Module 1 pipeline markers.
        has_lipid_molec: True if LipidMolec column exists.
        has_class_key: True if ClassKey column exists.
        intensity_columns: List of intensity[sN] columns found.
        concentration_columns: List of concentration[sN] columns found.
        message: User-facing validation message.
    """
    is_valid: bool
    has_lipid_molec: bool
    has_class_key: bool
    intensity_columns: List[str]
    concentration_columns: List[str]
    message: str


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
class LCQuestionsStepResult:
    """Result of a single LipidCruncher questions conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        assistant_message: The full assistant response text for display.
    """
    session: ChatSession
    assistant_message: str = ""


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
    def diagnose_format(df: pd.DataFrame) -> FormatDiagnosticResult:
        """Run format diagnostic on uploaded data.

        Checks whether data is already in a supported format. If not,
        analyzes the schema to guess which columns are lipid names,
        sample intensities, and metadata.

        Args:
            df: The user's uploaded DataFrame.

        Returns:
            FormatDiagnosticResult with detection outcome and column analysis.
        """
        detected = FormatDetectionService.detect_format(df)

        if detected != DataFormat.UNKNOWN:
            return FormatDiagnosticResult(
                detected_format=detected,
                is_supported=True,
                message=(
                    f"Your data is already in **{detected.value}** format. "
                    f"No transformation needed — upload it directly to "
                    f"LipidCruncher using the main pipeline."
                ),
            )

        # UNKNOWN format — analyze columns
        lipid_col = None
        intensity_cols = []
        metadata_cols = []

        # Reuse the shared pattern set from format_detection so the
        # detector and the diagnostic stay aligned.
        metadata_patterns = GENERIC_METADATA_NAME_PATTERNS

        for i, col in enumerate(df.columns):
            col_lower = col.lower().strip()
            series = df[col]

            # First string column with letter content → likely lipid names
            if lipid_col is None and series.dtype == object:
                has_letters = series.astype(str).str.contains('[A-Za-z]').any()
                if has_letters:
                    lipid_col = col
                    continue

            # Check if column name matches metadata patterns
            is_metadata = any(p in col_lower for p in metadata_patterns)
            if is_metadata:
                metadata_cols.append(col)
                continue

            # Non-numeric string columns → likely metadata
            if series.dtype == object:
                numeric_ratio = (
                    pd.to_numeric(series, errors='coerce').notna().sum()
                    / max(len(series), 1)
                )
                if numeric_ratio < 0.5:
                    metadata_cols.append(col)
                    continue

            # Numeric columns → likely intensities
            if pd.api.types.is_numeric_dtype(series):
                intensity_cols.append(col)

        # Build diagnostic message
        parts = ["Your data doesn't match any supported format. "
                 "Here's what I found:\n"]
        if lipid_col:
            parts.append(f"- **Lipid names column**: `{lipid_col}`")
        else:
            parts.append("- **Lipid names column**: could not determine "
                         "— please tell me which column has lipid names")
        parts.append(
            f"- **Sample intensity columns** ({len(intensity_cols)}): "
            + (", ".join(f"`{c}`" for c in intensity_cols[:8])
               + (f" ... and {len(intensity_cols) - 8} more"
                  if len(intensity_cols) > 8 else "")
               if intensity_cols else "none detected")
        )
        if metadata_cols:
            parts.append(
                f"- **Metadata columns to remove** ({len(metadata_cols)}): "
                + ", ".join(f"`{c}`" for c in metadata_cols[:8])
                + (f" ... and {len(metadata_cols) - 8} more"
                   if len(metadata_cols) > 8 else "")
            )

        parts.append("\nWould you like me to transform this data to "
                     "Generic format? Let me know if you'd also like any "
                     "filtering (e.g., remove low-quality rows by score or "
                     "grade — you tell me the column and threshold).")

        return FormatDiagnosticResult(
            detected_format=detected,
            is_supported=False,
            message="\n".join(parts),
            lipid_name_column=lipid_col,
            intensity_columns=intensity_cols,
            metadata_columns=metadata_cols,
        )

    @staticmethod
    def start_transform_session(
        source_df: pd.DataFrame,
        diagnostic: Optional[FormatDiagnosticResult] = None,
    ) -> ChatSession:
        """Initialize a new data transformation chat session.

        Pre-loads the DataFrame schema into the session context so
        the AI has structural information without raw data. If a
        diagnostic is provided and the format is UNKNOWN, the
        pre-analyzed column structure is appended to the schema.

        Args:
            source_df: The user's uploaded DataFrame.
            diagnostic: Optional format diagnostic result.

        Returns:
            A new ChatSession in DATA_TRANSFORM mode with schema context.
        """
        from ..services.ai_chat.prompt_templates import build_diagnostic_context

        schema = SchemaExtractorService.extract_schema(source_df)

        # Append diagnostic context if format is unknown
        if (diagnostic is not None
                and not diagnostic.is_supported):
            diag_context = build_diagnostic_context(
                lipid_name_column=diagnostic.lipid_name_column,
                intensity_columns=diagnostic.intensity_columns,
                metadata_columns=diagnostic.metadata_columns,
            )
            schema = f"{schema}\n\n{diag_context}"

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
        numeric_columns: Optional[List[str]] = None,
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
            numeric_columns: Optional list of columns to pre-coerce to
                numeric in the sandbox (typically the sample-intensity
                columns the user specified in the sidebar).

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

        # 3-4. Execute (if any) and append assistant message
        return AIChatWorkflow._finalize_transform(
            session, explanation, code, source_df,
            sandbox_config, numeric_columns,
        )

    @staticmethod
    def _finalize_transform(
        session: ChatSession,
        explanation: str,
        code: str,
        source_df: pd.DataFrame,
        sandbox_config: SandboxConfig,
        numeric_columns: Optional[List[str]],
    ) -> TransformStepResult:
        """Execute generated transform code (if any), build the assistant
        message, append it to the session, and return the step result.

        Shared by the blocking and streaming transform paths so the
        post-AI logic lives in exactly one place.
        """
        transformation = None
        if code:
            transformation = DataTransformerService.execute_transformation(
                code=code,
                source_df=source_df,
                sandbox_config=sandbox_config,
                numeric_columns=numeric_columns,
            )
            # Include explanation in the transformation result
            transformation = TransformationResult(
                success=transformation.success,
                result_df=transformation.result_df,
                is_info_only=transformation.is_info_only,
                explanation=explanation,
                code=code,
                sandbox_result=transformation.sandbox_result,
                validation_errors=transformation.validation_errors,
            )

            # Build assistant response text
            if transformation.success and transformation.is_info_only:
                # Info-gathering turn: surface the printed output. Don't
                # claim a transformation was produced.
                stdout = (
                    transformation.sandbox_result.stdout
                    if transformation.sandbox_result else ""
                )
                assistant_text = (
                    f"{explanation}\n\n```\n{stdout.rstrip()}\n```"
                )
            elif transformation.success:
                summary = DataTransformerService.build_validation_summary(
                    transformation.result_df
                )
                assistant_text = (
                    f"{explanation}\n\n"
                    f"The transformation was successful! "
                    f"**Result:** {summary}. "
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

    @staticmethod
    def stream_transform_message(
        user_message: str,
        session: ChatSession,
        source_df: pd.DataFrame,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
    ):
        """Streaming variant of ``handle_transform_message``.

        Appends the user message, then returns ``(explanation_generator,
        finalize)``. The caller streams the explanation (e.g. via
        ``st.write_stream``), then calls ``finalize()`` to execute the code
        and obtain a ``TransformStepResult``.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)
        history = session.to_api_messages()[:-1]

        holder = {"full": ""}
        generator = DataTransformerService.stream_transformation_code(
            source_schema=session.schema_context,
            user_message=user_message,
            history=history,
            config=ai_config,
            holder=holder,
        )

        def finalize() -> TransformStepResult:
            explanation, code = AIClientService.extract_code_block(holder["full"])
            return AIChatWorkflow._finalize_transform(
                session, explanation, code, source_df,
                sandbox_config, numeric_columns,
            )

        return generator, finalize

    # ------------------------------------------------------------------
    # Visualization mode
    # ------------------------------------------------------------------

    @staticmethod
    def validate_module1_output(df: pd.DataFrame) -> Module1ValidationResult:
        """Check whether data has been processed through Module 1.

        Module 1 output has: LipidMolec, ClassKey, and intensity[sN]
        or concentration[sN] columns.

        Args:
            df: The DataFrame to validate.

        Returns:
            Module1ValidationResult with validation outcome.
        """
        cols = set(df.columns)
        has_lipid_molec = 'LipidMolec' in cols
        has_class_key = 'ClassKey' in cols
        intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
        concentration_cols = [c for c in df.columns
                             if c.startswith('concentration[')]

        sample_cols = intensity_cols or concentration_cols
        is_valid = has_lipid_molec and has_class_key and len(sample_cols) > 0

        if is_valid:
            col_type = "concentration" if concentration_cols else "intensity"
            message = (
                f"Data has been processed through Module 1: "
                f"**{len(df)}** lipid species, "
                f"**{len(sample_cols)}** {col_type} columns. "
                f"Ready for custom visualizations."
            )
        else:
            missing = []
            if not has_lipid_molec:
                missing.append("`LipidMolec` (standardized lipid names)")
            if not has_class_key:
                missing.append("`ClassKey` (lipid class)")
            if not sample_cols:
                missing.append(
                    "`intensity[sN]` or `concentration[sN]` columns"
                )
            message = (
                "Your data doesn't appear to have been processed through "
                "LipidCruncher's Module 1 pipeline. Missing: "
                + ", ".join(missing) + ".\n\n"
                "For best results, run your data through the main pipeline "
                "first (**Start Crunching** on the home page), then return "
                "here for custom visualizations.\n\n"
                "You can still try creating visualizations, but the AI may "
                "not be able to use LipidCruncher-specific column names."
            )

        return Module1ValidationResult(
            is_valid=is_valid,
            has_lipid_molec=has_lipid_molec,
            has_class_key=has_class_key,
            intensity_columns=intensity_cols,
            concentration_columns=concentration_cols,
            message=message,
        )

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
        numeric_columns: Optional[List[str]] = None,
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
            numeric_columns: Optional list of columns to pre-coerce to
                numeric in the sandbox (typically the renamed
                intensity[*]/concentration[*] columns produced by the
                apply step).

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

        # 3-4. Execute (if any) and append assistant message
        return AIChatWorkflow._finalize_viz(
            session, explanation, code, analysis_df,
            sandbox_config, numeric_columns, experiment,
        )

    @staticmethod
    def _finalize_viz(
        session: ChatSession,
        explanation: str,
        code: str,
        analysis_df: pd.DataFrame,
        sandbox_config: SandboxConfig,
        numeric_columns: Optional[List[str]],
        experiment: Optional[ExperimentConfig] = None,
    ) -> VizStepResult:
        """Execute generated viz code (if any), build the assistant message,
        append it to the session, and return the step result.

        Shared by the blocking and streaming viz paths.
        """
        sandbox_result = None
        if code:
            sandbox_result = VizGeneratorService.execute_viz_code(
                code=code,
                df=analysis_df,
                sandbox_config=sandbox_config,
                numeric_columns=numeric_columns,
                experiment=experiment,
                lc=PipelineAPI(analysis_df, experiment),
            )

            if sandbox_result.success and sandbox_result.figure is not None:
                if _figure_is_empty(sandbox_result.figure):
                    assistant_text = (
                        f"{explanation}\n\n"
                        f"The code ran without errors but produced an empty "
                        f"figure (no data was plotted). This usually means a "
                        f"filter, group-by, or column-name mismatch eliminated "
                        f"all rows before plotting. Expand the code below the "
                        f"empty plot to see what was generated, then describe "
                        f"what to fix and I'll adjust."
                    )
                else:
                    assistant_text = (
                        f"{explanation}\n\n"
                        f"Here's the visualization you requested."
                    )
                    if sandbox_result.methods:
                        assistant_text += (
                            "\n\n" + _methods_note(sandbox_result.methods)
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

    @staticmethod
    def stream_viz_message(
        user_message: str,
        session: ChatSession,
        analysis_df: pd.DataFrame,
        experiment: ExperimentConfig,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
    ):
        """Streaming variant of ``handle_viz_message``.

        Appends the user message, then returns ``(explanation_generator,
        finalize)``. The caller streams the explanation, then calls
        ``finalize()`` to execute the code and obtain a ``VizStepResult``.
        """
        if ai_config is None:
            ai_config = AIConfig()
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)
        history = session.to_api_messages()[:-1]

        holder = {"full": ""}
        generator = VizGeneratorService.stream_viz_code(
            user_request=user_message,
            schema_context=session.schema_context,
            experiment=experiment,
            history=history,
            config=ai_config,
            holder=holder,
        )

        def finalize() -> VizStepResult:
            explanation, code = AIClientService.extract_code_block(holder["full"])
            return AIChatWorkflow._finalize_viz(
                session, explanation, code, analysis_df,
                sandbox_config, numeric_columns, experiment,
            )

        return generator, finalize

    # ------------------------------------------------------------------
    # Data Questions mode
    # ------------------------------------------------------------------

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
        )
        return ChatSession(
            mode=ChatMode.DATA_QUESTIONS,
            schema_context=schema_context,
        )

    @staticmethod
    def handle_data_questions_message(
        user_message: str,
        session: ChatSession,
        source_df: pd.DataFrame,
        ai_config: Optional[AIConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
        experiment: Optional[ExperimentConfig] = None,
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
            numeric_columns: Optional list of columns to pre-coerce to
                numeric in the sandbox (typically the sample-intensity
                columns the user specified in the sidebar).

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

        # 2. Build context and call AI with tools — long-form data above the
        # question (Anthropic long-context guidance).
        context = f"<dataframe_schema>\n{session.schema_context}\n</dataframe_schema>"
        exp_context = build_experiment_context(experiment)
        if exp_context:
            context = f"{context}\n\n{exp_context}"
        augmented_message = f"{context}\n\n{user_message}"
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

        # 3-4. Extract/execute code (if any) and append assistant message
        return AIChatWorkflow._finalize_data_questions(
            session, response_text, source_df,
            sandbox_config, numeric_columns, experiment,
        )

    @staticmethod
    def _finalize_data_questions(
        session: ChatSession,
        response_text: str,
        source_df: pd.DataFrame,
        sandbox_config: SandboxConfig,
        numeric_columns: Optional[List[str]],
        experiment: Optional[ExperimentConfig] = None,
    ) -> DataQuestionsStepResult:
        """Extract/execute any code in the AI response, build the assistant
        message, append it to the session, and return the step result.

        Shared by the blocking and streaming data-questions paths.
        """
        explanation, code = AIClientService.extract_code_block(response_text)
        result_df = None

        if code:
            sandbox_result = SandboxService.execute(
                code, source_df, sandbox_config,
                numeric_columns=numeric_columns,
                experiment=experiment,
                lc=PipelineAPI(source_df, experiment),
            )
            if sandbox_result.success:
                result_df = sandbox_result.result_df
                parts = []
                if explanation.strip():
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
    ):
        """Streaming variant of ``handle_data_questions_message``.

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
            return AIChatWorkflow._finalize_data_questions(
                session, holder["full"], source_df,
                sandbox_config, numeric_columns, experiment,
            )

        return generator, finalize

    # ------------------------------------------------------------------
    # LipidCruncher Questions mode
    # ------------------------------------------------------------------

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
    def handle_lc_questions_message(
        user_message: str,
        session: ChatSession,
        ai_config: Optional[AIConfig] = None,
    ) -> LCQuestionsStepResult:
        """Handle one user message in the LipidCruncher questions chat.

        Purely conversational — no code execution. Uses the tool loop
        so the AI can search the codebase if needed.

        Args:
            user_message: The user's latest message.
            session: Current chat session.
            ai_config: AI configuration. Uses defaults if None.

        Returns:
            LCQuestionsStepResult with updated session and response.
        """
        if ai_config is None:
            ai_config = AIConfig()

        # 1. Append user message
        user_msg = ChatMessage(role="user", content=user_message)
        session = session.add_message(user_msg)

        # 2. Call AI with tools (for codebase search)
        history = session.to_api_messages()[:-1]
        messages = history + [{"role": "user", "content": user_message}]

        system_prompt = lipidcruncher_questions_system_prompt()

        response_text = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=ai_config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
        )

        assistant_text = response_text.strip()
        if not assistant_text:
            assistant_text = (
                "I wasn't able to generate a response. "
                "Please try rephrasing your question."
            )

        # 3. Append assistant message
        assistant_msg = ChatMessage(role="assistant", content=assistant_text)
        session = session.add_message(assistant_msg)

        return LCQuestionsStepResult(
            session=session,
            assistant_message=assistant_text,
        )

    @staticmethod
    def stream_lc_questions_message(
        user_message: str,
        session: ChatSession,
        ai_config: Optional[AIConfig] = None,
    ):
        """Streaming variant of ``handle_lc_questions_message``.

        Appends the user message to the session, then returns a generator
        that streams the assistant's answer token-by-token. The caller (UI)
        consumes the generator — e.g. via ``st.write_stream`` — and appends
        the assistant message to the session from the accumulated text.

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

        # 2. Build context and call AI with tools — long-form data above the
        # question (Anthropic long-context guidance).
        context = f"<dataframe_schema>\n{session.schema_context}\n</dataframe_schema>"
        augmented_message = f"{context}\n\n{user_message}"
        history = session.to_api_messages()[:-1]
        messages = history + [{"role": "user", "content": augmented_message}]

        system_prompt = unified_system_prompt()

        logger.info("Calling AI with %d messages", len(messages))
        response_text = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=ai_config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
        )
        logger.info("AI response: %d chars, first 200: %s",
                     len(response_text), response_text[:200])

        # 3. Extract and execute code if present
        explanation, code = AIClientService.extract_code_block(response_text)
        result_df = None
        figure = None
        is_transformation = False

        if code:
            sandbox_result = SandboxService.execute(
                code, source_df, sandbox_config,
                lc=PipelineAPI(source_df, None),
            )

            if sandbox_result.success:
                # Collect all outputs (figure and result_df can coexist)
                figure = sandbox_result.figure
                result_df = sandbox_result.result_df

                # Build the text response
                parts = []
                if explanation.strip():
                    parts.append(explanation)

                if result_df is not None:
                    is_transformation = _looks_like_generic_format(result_df)
                    if is_transformation:
                        parts.append(
                            f"The transformation was successful! "
                            f"The result has {len(result_df)} rows and "
                            f"{len(result_df.columns)} columns. "
                            f"You can download the transformed file below."
                        )

                # Include captured print output (e.g., statistics, summaries)
                if sandbox_result.stdout.strip():
                    parts.append(sandbox_result.stdout.strip())

                if figure is not None and not parts:
                    parts.append("Here's the visualization you requested.")

                assistant_text = "\n\n".join(parts) if parts else "Done."
            else:
                assistant_text = (
                    f"{explanation}\n\n"
                    f"I encountered an error: {sandbox_result.error_message}"
                ) if explanation.strip() else (
                    f"I encountered an error: {sandbox_result.error_message}"
                )
        else:
            assistant_text = explanation
            code = ""

        # Guard against empty response
        if not assistant_text.strip():
            logger.warning("Empty assistant_text — response_text was: %s", response_text[:500])
            assistant_text = "I wasn't able to generate a response. Please try rephrasing your question."

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


def _methods_note(methods) -> str:
    """The statistical choices `lc` made, as a note under the answer."""
    lines = "\n".join(f"- {m}" for m in methods)
    return (
        f"**Statistical choices** (ask if you want any of them changed):\n"
        f"{lines}"
    )


def _figure_is_empty(fig) -> bool:
    """True if the figure has no traces or every trace has empty x and y."""
    if fig is None or not getattr(fig, "data", None):
        return True
    for trace in fig.data:
        x = getattr(trace, "x", None)
        y = getattr(trace, "y", None)
        if (x is not None and len(x) > 0) or (y is not None and len(y) > 0):
            return False
    return True


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
