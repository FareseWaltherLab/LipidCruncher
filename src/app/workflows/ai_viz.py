"""
Custom Visualization: the AI chat workflow's figure generation for the
Custom Visualization tab. Pure logic — no Streamlit dependencies.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from ..models.chat import AIConfig, ChatMessage, ChatMode, ChatSession, SandboxConfig
from ..models.experiment import ExperimentConfig
from ..services.ai_chat.ai_client import AIClientService
from ..services.ai_chat.output_checks import check_direction_colors
from ..services.ai_chat.sandbox import SandboxResult
from ..services.ai_chat.schema_extractor import SchemaExtractorService
from ..services.ai_chat.viz_generator import VizGeneratorService
from .ai_chat_shared import _methods_note, _unverified_numbers
from .pipeline_api import PipelineAPI


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
class VizStepResult:
    """Result of a single visualization conversation turn.

    Attributes:
        session: Updated chat session with new messages appended.
        sandbox_result: Sandbox execution result (None if AI only asked a question).
        assistant_message: The full assistant response text for display.
        unverified_numbers: Numbers the explanation states that the code
            never produced (logged, not shown; see ``_unverified_numbers``).
    """
    session: ChatSession
    sandbox_result: Optional[SandboxResult] = None
    assistant_message: str = ""
    unverified_numbers: Tuple[str, ...] = ()


# Trace attributes that carry plotted data. Only x/y was checked before, so
# pies, sunbursts and treemaps (values/labels), heatmaps and images (z) and
# parallel coordinates (dimensions) were all reported as empty figures.
_TRACE_DATA_ATTRS = (
    "x", "y", "z", "values", "labels", "r", "lat", "lon", "dimensions", "cells",
)


def figure_has_data(fig) -> bool:
    """True if any trace of ``fig`` holds at least one data point."""
    for trace in getattr(fig, "data", None) or ():
        for attr in _TRACE_DATA_ATTRS:
            value = getattr(trace, attr, None)
            if attr == "cells":  # go.Table: the data is in cells.values
                value = getattr(value, "values", None)
            try:
                if value is not None and len(value) > 0:
                    return True
            except TypeError:
                continue
    return False


class VizWorkflow:
    """The Custom Visualization tab: Module 1 output check, then the model's
    plotting code run in the sandbox. Static methods only; AIChatWorkflow
    inherits them."""

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
    def _finalize_viz(
        session: ChatSession,
        explanation: str,
        code: str,
        analysis_df: pd.DataFrame,
        sandbox_config: SandboxConfig,
        numeric_columns: Optional[List[str]],
        experiment: Optional[ExperimentConfig] = None,
        bqc_label: Optional[str] = None,
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
                lc=PipelineAPI(analysis_df, experiment, bqc_label),
            )

            if sandbox_result.success and sandbox_result.figure is not None:
                if not figure_has_data(sandbox_result.figure):
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
                    # Shown to the user, and seen by the model next turn.
                    for warning in check_direction_colors(sandbox_result.figure):
                        assistant_text += f"\n\n{warning}"
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
            if sandbox_result.success:
                # Kept in the message, fenced as in the other tabs, so the
                # transcript and later turns see what the code printed.
                if sandbox_result.stdout.strip():
                    assistant_text += (
                        f"\n\n```\n{sandbox_result.stdout.strip()}\n```"
                    )
                if sandbox_result.methods:
                    assistant_text += (
                        "\n\n" + _methods_note(sandbox_result.methods)
                    )
        else:
            assistant_text = explanation

        assistant_msg = ChatMessage(
            role="assistant",
            content=assistant_text,
            code=code if code else None,
        )
        session = session.add_message(assistant_msg)

        ran = sandbox_result is not None and sandbox_result.success
        return VizStepResult(
            session=session,
            sandbox_result=sandbox_result,
            assistant_message=assistant_text,
            unverified_numbers=_unverified_numbers(
                session, explanation, code,
                stdout=sandbox_result.stdout if ran else "",
                figure=sandbox_result.figure if ran else None,
            ),
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
        bqc_label: Optional[str] = None,
    ):
        """Handle one user message in the visualization chat.

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
            return VizWorkflow._finalize_viz(
                session, explanation, code, analysis_df,
                sandbox_config, numeric_columns, experiment,
                bqc_label=bqc_label,
            )

        return generator, finalize
