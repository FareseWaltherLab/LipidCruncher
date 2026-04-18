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
        """Validate that a DataFrame conforms to the Generic upload format.

        The Generic format is simple:
          - Column 1: lipid names (string)
          - Columns 2+: sample intensities (numeric)

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

        # Columns 2+ should be numeric (sample intensities)
        non_numeric_cols = []
        for col in df.columns[1:]:
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
                f"Non-numeric columns found after column 1: {col_list}. "
                f"Columns 2+ should contain only sample intensities. "
                f"Remove any metadata columns (class, mass, score, etc.)."
            )

        return len(errors) == 0, errors
