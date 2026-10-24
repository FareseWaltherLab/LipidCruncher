"""
Transform to Generic: the AI chat workflow's format diagnosis and code
execution for the Transform tab. Pure logic — no Streamlit dependencies.
"""
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ..models.chat import AIConfig, ChatMessage, ChatMode, ChatSession, SandboxConfig
from ..services.ai_chat.ai_client import AIClientService
from ..services.ai_chat.data_transformer import DataTransformerService, TransformationResult
from ..services.ai_chat.schema_extractor import SchemaExtractorService
from ..services.format_detection import (
    FormatDetectionService,
    DataFormat,
    looks_like_metadata_name,
)
from .ai_chat_shared import _repeated_lipids_note


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


_RAW_IMPORT_MESSAGE = (
    "This is **raw** data imported from the main pipeline "
    "— column names are already standardized to "
    "`intensity[*]`, but the data hasn't been filtered "
    "or normalized.\n\n"
    "For the standard workflow, run it through the main "
    "pipeline (Cleaning → Zero Filtering → Normalization) "
    "and re-import the **Normalized** version. Or, if you "
    "want a custom filter/normalization, ask me here and "
    "I'll generate the code."
)


_RAW_IMPORT_NOTE = (
    "\n\n## Important context for this dataset\n"
    "This data was imported from the LipidCruncher main "
    "pipeline. It is **already in Generic shape** "
    "(LipidMolec + intensity[*] columns) — no shape "
    "transformation is needed. When the user asks for "
    "filtering, deduplication, normalization, or other "
    "content-level operations, treat those as standalone "
    "tasks. Do NOT frame your response as 'preprocessing "
    "before Generic conversion' — the data is already "
    "Generic-shaped. The result_df should still be in "
    "Generic shape (lipid name column + numeric sample "
    "columns) so the user can download and re-upload it."
)


class TransformWorkflow:
    """The Transform to Generic tab: diagnose the upload, then turn the
    model's code into a checked Generic-format table. Static methods only;
    AIChatWorkflow inherits them."""

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

        for i, col in enumerate(df.columns):
            series = df[col]

            # First string column with letter content → likely lipid names
            if lipid_col is None and series.dtype == object:
                has_letters = series.astype(str).str.contains('[A-Za-z]').any()
                if has_letters:
                    lipid_col = col
                    continue

            # The detector's own metadata-name rule, so the two agree.
            if looks_like_metadata_name(col):
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
            schema_context=schema + _repeated_lipids_note(source_df),
        )

    @staticmethod
    def start_transform_chat(
        source_df: pd.DataFrame,
        raw_pipeline_import: bool = False,
    ) -> Tuple[FormatDiagnosticResult, ChatSession]:
        """Diagnose the data and open the Transform tab's conversation.

        Raw pipeline imports trip the Generic-shape detection (LipidMolec +
        intensity[*]) but are still unfiltered and un-normalized, so the
        usual "no transformation needed" diagnosis would mislead. For them
        the diagnosis points at the main pipeline while keeping the tab
        usable (``is_supported=False``), and the AI is told the data is
        already Generic-shaped, so filter / normalize requests are not
        framed as preprocessing before a conversion.

        Returns:
            ``(diagnostic, session)``.
        """
        diagnostic = TransformWorkflow.diagnose_format(source_df)
        if raw_pipeline_import:
            diagnostic = replace(
                diagnostic,
                is_supported=False,
                message=_RAW_IMPORT_MESSAGE,
                lipid_name_column=(
                    'LipidMolec' if 'LipidMolec' in source_df.columns else None
                ),
                intensity_columns=[
                    c for c in source_df.columns
                    if str(c).startswith('intensity[')
                ],
                # The extra columns (CalcMass, BaseRt, ...) of raw pipeline
                # data aren't drop candidates the way they are in an
                # arbitrary CSV.
                metadata_columns=[],
            )
        session = TransformWorkflow.start_transform_session(source_df, diagnostic)
        if raw_pipeline_import:
            session = session.model_copy(update={
                'schema_context': session.schema_context + _RAW_IMPORT_NOTE,
            })
        return diagnostic, session

    @staticmethod
    def _finalize_transform(
        session: ChatSession,
        explanation: str,
        code: str,
        source_df: pd.DataFrame,
        sandbox_config: SandboxConfig,
        numeric_columns: Optional[List[str]],
        keep_columns: Sequence[str] = (),
        sample_columns: Optional[Sequence[str]] = None,
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
                keep_columns=keep_columns,
                sample_columns=sample_columns,
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
                report=transformation.report,
                metadata_columns=transformation.metadata_columns,
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
                    f"**Compared with your upload:** "
                    f"{transformation.report.summary()}. "
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
        keep_columns: Sequence[str] = (),
        sample_columns: Optional[Sequence[str]] = None,
    ):
        """Handle one user message in a data transformation chat.

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
        # The model sees which columns the user confirmed as samples; the
        # session keeps the user's own words.
        model_message = user_message
        if sample_columns:
            model_message = (
                "Sample columns the user confirmed (result_df's sample "
                "columns must come from these): "
                + ", ".join(f"`{c}`" for c in sample_columns)
                + f"\n\n{user_message}"
            )
        generator = DataTransformerService.stream_transformation_code(
            source_schema=session.schema_context,
            user_message=model_message,
            history=history,
            config=ai_config,
            holder=holder,
        )

        def finalize() -> TransformStepResult:
            explanation, code = AIClientService.extract_code_block(holder["full"])
            return TransformWorkflow._finalize_transform(
                session, explanation, code, source_df,
                sandbox_config, numeric_columns, keep_columns, sample_columns,
            )

        return generator, finalize

    @staticmethod
    def keep_transform_columns(
        session: ChatSession,
        source_df: pd.DataFrame,
        keep_columns: Sequence[str],
        numeric_columns: Optional[List[str]] = None,
        sample_columns: Optional[Sequence[str]] = None,
    ) -> TransformStepResult:
        """Re-run the last transform, keeping columns the user confirmed
        are samples although their names look like metadata.

        No AI call: the last assistant message's code runs again with
        ``keep_columns`` allowed, and the outcome is appended as a new
        assistant message, so the model sees the choice on the next turn.
        """
        code = next(
            (m.code for m in reversed(session.messages)
             if m.role == "assistant" and m.code),
            "",
        )
        names = ", ".join(f"`{c}`" for c in keep_columns)
        return TransformWorkflow._finalize_transform(
            session,
            f"You confirmed these are samples: {names}. They stay in the "
            f"result.",
            code, source_df, SandboxConfig(), numeric_columns, keep_columns,
            sample_columns,
        )
