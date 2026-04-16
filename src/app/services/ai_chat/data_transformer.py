"""
AI-powered data transformation service.

Generates transformation code via the AI, executes it in the sandbox,
and validates that the output conforms to the Generic format schema.
All methods are stateless static methods — no Streamlit imports.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from ...models.chat import AIConfig, SandboxConfig
from .ai_client import AIClientService
from .exceptions import TransformationError
from .prompt_templates import build_transform_context, data_transform_system_prompt
from .sandbox import SandboxResult, SandboxService
from .schema_extractor import SchemaExtractorService


@dataclass(frozen=True)
class TransformationResult:
    """Result of a data transformation attempt.

    Attributes:
        success: Whether the transformation produced valid output.
        result_df: Transformed DataFrame in Generic format (if successful).
        explanation: AI's explanation of what the code does.
        code: The generated Python code.
        sandbox_result: Raw sandbox execution result.
        validation_errors: Schema validation errors (if any).
    """
    success: bool
    result_df: Optional[pd.DataFrame] = None
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
    ) -> Tuple[str, str]:
        """Ask the AI to generate transformation code.

        Args:
            source_schema: Schema description of the source DataFrame.
            user_message: The user's latest message/instructions.
            history: Previous conversation messages in API format.
            config: AI configuration. Uses defaults if None.

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

        # Append context to the user's message
        augmented_message = f"{user_message}\n\n{context}"
        messages = history + [{"role": "user", "content": augmented_message}]

        return AIClientService.send_message_for_code(messages, system_prompt, config)

    @staticmethod
    def execute_transformation(
        code: str,
        source_df: pd.DataFrame,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> TransformationResult:
        """Execute transformation code and validate the output.

        Args:
            code: Python code that should produce a ``result_df``.
            source_df: The user's original DataFrame.
            sandbox_config: Sandbox configuration. Uses defaults if None.

        Returns:
            TransformationResult with the outcome.
        """
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        sandbox_result = SandboxService.execute(code, source_df, sandbox_config)

        if not sandbox_result.success:
            return TransformationResult(
                success=False,
                code=code,
                sandbox_result=sandbox_result,
                validation_errors=(sandbox_result.error_message,),
            )

        if sandbox_result.result_df is None:
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
        """Validate that a DataFrame conforms to the Generic format.

        Checks for required columns: LipidMolec, ClassKey, and at least
        one intensity[sN] column.

        Args:
            df: DataFrame to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: List[str] = []

        if df.empty:
            errors.append("DataFrame is empty — no rows produced.")
            return False, errors

        if "LipidMolec" not in df.columns:
            errors.append("Missing required column: 'LipidMolec'")

        if "ClassKey" not in df.columns:
            errors.append("Missing required column: 'ClassKey'")

        intensity_pattern = re.compile(r"^intensity\[s\d+\]$")
        intensity_cols = [c for c in df.columns if intensity_pattern.match(c)]
        if not intensity_cols:
            errors.append(
                "No intensity columns found. Expected columns like "
                "'intensity[s1]', 'intensity[s2]', etc."
            )

        # Check that intensity columns are numeric
        for col in intensity_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric but has dtype {df[col].dtype}")

        # Check LipidMolec has actual values
        if "LipidMolec" in df.columns and df["LipidMolec"].isna().all():
            errors.append("'LipidMolec' column is entirely empty/NaN")

        if "ClassKey" in df.columns and df["ClassKey"].isna().all():
            errors.append("'ClassKey' column is entirely empty/NaN")

        return len(errors) == 0, errors
