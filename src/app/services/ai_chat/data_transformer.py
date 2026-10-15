"""
AI-powered data transformation service.

Generates transformation code via the AI, executes it in the sandbox,
and validates that the output conforms to the Generic format schema.
All methods are stateless static methods — no Streamlit imports.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

from ...models.chat import AIConfig, SandboxConfig
from ..data_standardization import DataStandardizationService
from ..format_detection import GENERIC_METADATA_NAME_PATTERNS
from .ai_client import AIClientService
from .codebase_tools import TOOL_DEFINITIONS, execute_tool
from .prompt_templates import build_transform_context, data_transform_system_prompt
from .sandbox import SandboxResult, SandboxService
from .schema_extractor import SchemaExtractorService


@dataclass(frozen=True)
class TransformationResult:
    """Result of a data transformation attempt.

    Attributes:
        success: Whether the turn ran cleanly. True for both
            transformation and info-gathering paths; False on sandbox
            error or schema-validation failure.
        result_df: Transformed DataFrame in Generic format. None on
            info-gathering turns (where the AI printed instead of
            transforming) and on failures.
        is_info_only: True when the AI ran code that only printed
            information (no `result_df` set) — typically because the
            user asked to see something to plan a transform. The chat
            UI shows the stdout instead of a "transformation
            successful" / preview banner.
        explanation: AI's explanation of what the code does.
        code: The generated Python code.
        sandbox_result: Raw sandbox execution result.
        validation_errors: Schema validation errors (if any).
    """
    success: bool
    result_df: Optional[pd.DataFrame] = None
    is_info_only: bool = False
    explanation: str = ""
    code: str = ""
    sandbox_result: Optional[SandboxResult] = None
    validation_errors: Tuple[str, ...] = ()


class DataTransformerService:
    """Transforms arbitrary DataFrames into the Generic format via AI.

    All methods are stateless static methods.
    """

    @staticmethod
    def generate_transformation_code(
        source_schema: str,
        user_message: str,
        history: list,
        config: Optional[AIConfig] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ) -> Tuple[str, str]:
        """Ask the AI to generate transformation code.

        Uses the tool-aware API so the AI can search/read the codebase
        before generating code.

        Args:
            source_schema: Schema description of the source DataFrame.
            user_message: The user's latest message/instructions.
            history: Previous conversation messages in API format.
            config: AI configuration. Uses defaults if None.
            on_tool_call: Optional callback(tool_name, tool_input) for
                UI status updates (e.g. "Searching codebase...").

        Returns:
            Tuple of (explanation, code).

        Raises:
            AIClientError: On API communication failure.
        """
        if config is None:
            config = AIConfig()

        target_schema = SchemaExtractorService.extract_generic_format_spec()
        context = build_transform_context(source_schema, target_schema)

        system_prompt = data_transform_system_prompt()

        # Prepend context to the user's message — long-form data goes above
        # the question (Anthropic long-context guidance).
        augmented_message = f"{context}\n\n{user_message}"
        messages = history + [{"role": "user", "content": augmented_message}]

        # Use the tool loop so the AI can look up implementation details
        response_text = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt=system_prompt,
            config=config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            on_tool_call=on_tool_call,
        )

        return AIClientService.extract_code_block(response_text)

    @staticmethod
    def stream_transformation_code(
        source_schema: str,
        user_message: str,
        history: list,
        config: AIConfig,
        holder: dict,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ):
        """Streaming variant of ``generate_transformation_code``.

        Returns a generator that yields the explanation prose as it streams.
        The complete raw response (explanation + code) is accumulated into
        ``holder['full']`` so the caller can extract and execute the code
        once streaming completes. Mirrors the message assembly of the
        non-streaming method exactly.
        """
        target_schema = SchemaExtractorService.extract_generic_format_spec()
        context = build_transform_context(source_schema, target_schema)
        system_prompt = data_transform_system_prompt()

        augmented_message = f"{context}\n\n{user_message}"
        messages = history + [{"role": "user", "content": augmented_message}]

        return AIClientService.stream_explanation(
            messages=messages,
            system_prompt=system_prompt,
            config=config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            holder=holder,
            on_tool_call=on_tool_call,
        )

    @staticmethod
    def execute_transformation(
        code: str,
        source_df: pd.DataFrame,
        sandbox_config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
    ) -> TransformationResult:
        """Execute transformation code and validate the output.

        Args:
            code: Python code that should produce a ``result_df``.
            source_df: The user's original DataFrame.
            sandbox_config: Sandbox configuration. Uses defaults if None.
            numeric_columns: Optional list of columns to pre-coerce to
                numeric before exposing the DataFrame to user code.

        Returns:
            TransformationResult with the outcome.
        """
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        sandbox_result = SandboxService.execute(
            code, source_df, sandbox_config, numeric_columns=numeric_columns,
        )

        if not sandbox_result.success:
            return TransformationResult(
                success=False,
                code=code,
                sandbox_result=sandbox_result,
                validation_errors=(sandbox_result.error_message,),
            )

        if sandbox_result.result_df is None:
            # Info-gathering turn (e.g. "list internal standards" before a
            # custom normalization): the AI ran code that only printed.
            # Treat as a success — the print output IS the answer — and
            # let the handler/UI surface stdout.
            stdout = sandbox_result.stdout or ""
            if stdout.strip():
                return TransformationResult(
                    success=True,
                    result_df=None,
                    is_info_only=True,
                    code=code,
                    sandbox_result=sandbox_result,
                )
            return TransformationResult(
                success=False,
                code=code,
                sandbox_result=sandbox_result,
                validation_errors=(
                    "Code executed successfully but did not produce a 'result_df'. "
                    "Make sure the code assigns the transformed DataFrame to 'result_df'.",
                ),
            )

        is_valid, errors = DataTransformerService.validate_generic_schema(
            sandbox_result.result_df
        )

        if not is_valid:
            return TransformationResult(
                success=False,
                result_df=sandbox_result.result_df,
                code=code,
                sandbox_result=sandbox_result,
                validation_errors=tuple(errors),
            )

        return TransformationResult(
            success=True,
            result_df=sandbox_result.result_df,
            code=code,
            sandbox_result=sandbox_result,
        )

    @staticmethod
    def validate_generic_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that a DataFrame conforms to the Generic upload format.

        Mirrors how the Generic reader (``DataStandardizationService``)
        will read the uploaded file:
          - Column 1: lipid names (text)
          - Column 2: optionally a ClassKey column (the reader's own test)
          - Remaining columns: sample intensities (numeric, not all empty)

        LipidCruncher handles all further standardization (renaming columns,
        extracting ClassKey, normalizing lipid names) after upload.

        Args:
            df: DataFrame to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        if df.empty:
            errors.append("DataFrame is empty — no rows produced.")
            return False, errors

        if len(df.columns) < 2:
            errors.append(
                "Need at least 2 columns: one for lipid names and "
                "one or more for sample intensities."
            )
            return False, errors

        # Column 1 should contain lipid names (strings)
        first_col = df.iloc[:, 0]
        if first_col.isna().all():
            errors.append(
                f"First column ('{df.columns[0]}') is entirely empty/NaN — "
                f"it should contain lipid names."
            )
        elif not first_col.astype(str).str.contains('[A-Za-z]').any():
            errors.append(
                f"First column ('{df.columns[0]}') has no text — it should "
                f"contain lipid names, not numbers."
            )

        sample_cols = DataTransformerService._sample_columns(df)
        if not sample_cols:
            errors.append(
                "No sample intensity columns after the lipid names "
                "(and ClassKey) columns."
            )

        # The sample columns should be numeric intensities
        non_numeric_cols = []
        for col in sample_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)

        if non_numeric_cols:
            if len(non_numeric_cols) <= 3:
                col_list = ", ".join(f"'{c}'" for c in non_numeric_cols)
            else:
                col_list = (
                    ", ".join(f"'{c}'" for c in non_numeric_cols[:3])
                    + f", and {len(non_numeric_cols) - 3} more"
                )
            errors.append(
                f"Non-numeric sample columns: {col_list}. After the lipid "
                f"names (and an optional ClassKey column) every column "
                f"should be a sample intensity. Remove any metadata columns "
                f"(mass, score, etc.)."
            )

        empty_cols = [
            c for c in sample_cols
            if c not in non_numeric_cols and df[c].isna().all()
        ]
        if empty_cols:
            errors.append(
                "Sample columns with no values: "
                + ", ".join(f"'{c}'" for c in empty_cols[:3])
                + (f", and {len(empty_cols) - 3} more" if len(empty_cols) > 3 else "")
                + "."
            )

        return len(errors) == 0, errors

    @staticmethod
    def _sample_columns(df: pd.DataFrame) -> List:
        """The columns the Generic reader will treat as samples."""
        has_classkey = (
            len(df.columns) > 2
            and DataStandardizationService._is_classkey_column(df, 1)
        )
        return list(df.columns[2 if has_classkey else 1:])

    @staticmethod
    def detect_metadata_columns(df: pd.DataFrame) -> List[str]:
        """Check for columns that look like leftover metadata.

        Scans column names (excluding the first, which is lipid names)
        for common metadata patterns. These shouldn't be in the Generic
        format — only lipid names + numeric intensities.

        Args:
            df: The transformed DataFrame.

        Returns:
            List of suspicious column names that may be metadata.
        """
        suspicious = []
        for col in DataTransformerService._sample_columns(df):
            col_lower = str(col).lower().strip()
            if any(p in col_lower for p in GENERIC_METADATA_NAME_PATTERNS):
                suspicious.append(col)
        return suspicious

    @staticmethod
    def build_validation_summary(df: pd.DataFrame) -> str:
        """Build a validation summary for a successfully transformed DataFrame.

        Args:
            df: The transformed DataFrame in Generic format.

        Returns:
            A summary string describing the result.
        """
        n_species = len(df)
        n_samples = len(DataTransformerService._sample_columns(df))
        metadata_cols = DataTransformerService.detect_metadata_columns(df)

        # Check first column looks like lipid names
        first_col = df.iloc[:, 0]
        has_letters = first_col.astype(str).str.contains('[A-Za-z]').any()

        parts = []
        if has_letters:
            parts.append(f"{n_species} lipid species")
        else:
            parts.append(f"{n_species} rows (first column may not be lipid names)")
        parts.append(f"{n_samples} sample columns")

        if metadata_cols:
            col_list = ", ".join(f"`{c}`" for c in metadata_cols[:5])
            if len(metadata_cols) > 5:
                col_list += f" and {len(metadata_cols) - 5} more"
            parts.append(f"possible metadata still present: {col_list}")
        else:
            parts.append("no metadata detected")

        return ", ".join(parts)
