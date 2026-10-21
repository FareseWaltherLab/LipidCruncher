"""
AI-powered data transformation service.

Generates transformation code via the AI, executes it in the sandbox,
and validates that the output conforms to the Generic format schema.
All methods are stateless static methods — no Streamlit imports.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...models.chat import AIConfig, SandboxConfig
from ..data_standardization import DataStandardizationService
from ..format_detection import looks_like_metadata_name
from .ai_client import AIClientService
from .codebase_tools import TOOL_DEFINITIONS, execute_tool
from .prompt_templates import build_transform_context, data_transform_system_prompt
from .sandbox import SandboxResult, SandboxService
from .schema_extractor import SchemaExtractorService


@dataclass(frozen=True)
class TransformReport:
    """What an accepted transform did to the upload, computed by the app.

    Lipids are matched by name and columns by name, as in
    ``check_values_stay_with_their_lipid``; a renamed lipid counts as
    removed and as not in the upload. Value counts cover lipid x column
    cells present on both sides (``cells_compared``).
    """
    rows_uploaded: int
    rows_result: int
    lipids_removed: Tuple[str, ...]
    lipids_not_in_upload: Tuple[str, ...]
    sample_columns: Tuple[str, ...]
    columns_dropped: Tuple[str, ...]
    cells_compared: int
    values_changed: int
    values_appeared: int
    values_missing: int
    duplicate_names: Tuple[str, ...]

    def summary(self) -> str:
        """One line for the assistant message."""
        kept = self.rows_uploaded - len(self.lipids_removed)
        if kept:
            parts = [
                f"{kept:,} of {self.rows_uploaded:,} lipids kept, "
                f"{len(self.lipids_removed):,} removed",
            ]
            if self.lipids_not_in_upload:
                parts.append(
                    f"{len(self.lipids_not_in_upload):,} lipid names not in "
                    f"the upload"
                )
        else:
            parts = [
                f"{self.rows_result:,} lipids, none named as in the upload, "
                f"so they could not be compared with it"
            ]
        parts.append(f"{len(self.sample_columns)} sample columns")
        if self.columns_dropped:
            parts.append(
                f"{len(self.columns_dropped)} columns dropped ("
                + ", ".join(f"`{c}`" for c in self.columns_dropped[:5])
                + (", ..." if len(self.columns_dropped) > 5 else "")
                + ")"
            )
        if self.cells_compared:
            parts.append(f"{self.values_changed:,} values changed")
            if self.values_appeared:
                parts.append(f"{self.values_appeared:,} values filled in")
            if self.values_missing:
                parts.append(f"{self.values_missing:,} values now missing")
        elif kept:
            parts.append(
                "values not compared (no sample column kept under its "
                "uploaded name)"
            )
        if self.duplicate_names:
            parts.append(
                f"{len(self.duplicate_names):,} lipid names appear more than once"
            )
        return "; ".join(parts)


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
        report: What an accepted transform changed relative to the
            upload. None unless the transform was accepted.
    """
    success: bool
    result_df: Optional[pd.DataFrame] = None
    is_info_only: bool = False
    explanation: str = ""
    code: str = ""
    sandbox_result: Optional[SandboxResult] = None
    validation_errors: Tuple[str, ...] = ()
    report: Optional[TransformReport] = None


class DataTransformerService:
    """Transforms arbitrary DataFrames into the Generic format via AI.

    All methods are stateless static methods.
    """

    @staticmethod
    def stream_transformation_code(
        source_schema: str,
        user_message: str,
        history: list,
        config: AIConfig,
        holder: dict,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ):
        """Ask the AI for transformation code, streaming its explanation.

        Uses the tool-aware loop so the AI can search/read the codebase
        before writing code. Returns a generator that yields the
        explanation prose as it streams. The complete raw response
        (explanation + code) is accumulated into ``holder['full']`` so the
        caller can extract and execute the code once streaming completes.
        """
        target_schema = SchemaExtractorService.extract_generic_format_spec()
        context = build_transform_context(source_schema, target_schema)
        system_prompt = data_transform_system_prompt()

        # Long-form data above the question (Anthropic long-context guidance).
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
        if is_valid:
            # A numeric metadata column passes the schema check, and the
            # Generic reader would then analyse it as a sample.
            metadata = DataTransformerService.detect_metadata_columns(
                sandbox_result.result_df
            )
            if metadata:
                errors = [
                    "Metadata columns kept as samples: "
                    + ", ".join(f"'{c}'" for c in metadata)
                    + ". Every column after the lipid names (and ClassKey) "
                    "is analysed as a sample, so drop these from result_df."
                ]
            errors += DataTransformerService.check_values_stay_with_their_lipid(
                sandbox_result.result_df, source_df,
            )
            is_valid = not errors

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
            report=DataTransformerService.compare_to_upload(
                sandbox_result.result_df, source_df,
            ),
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
    def check_values_stay_with_their_lipid(
        result_df: pd.DataFrame, source_df: pd.DataFrame,
    ) -> List[str]:
        """Catch values that moved to another lipid during the transform.

        Generated code that filters rows and then builds ``result_df`` by
        index (``result['name'] = filtered['name'].values`` next to
        ``result[col] = filtered[col]``) lines the columns up by row label,
        so lipids get other lipids' values or none. The result looks like a
        valid Generic table, so the schema check passes it.

        Compares each output row with the uploaded row of the same lipid,
        over the sample columns kept under their names. A transform may
        change values (scaling, filling gaps), so only an exact copy of a
        *different* lipid's row is treated as moved. Lipids or columns that
        were renamed, and names that are not unique, are not checked.

        Returns:
            One error message when values moved, otherwise an empty list.
        """
        names = result_df.iloc[:, 0].astype(str)
        name_col = DataTransformerService._name_column(names, source_df)
        columns = [
            c for c in DataTransformerService._sample_columns(result_df)
            if c in source_df.columns and c != name_col
        ]
        if name_col is None or not columns:
            return []

        source_names = source_df[name_col].astype(str)
        unique = set(source_names[~source_names.duplicated(keep=False)])
        unique &= set(names[~names.duplicated(keep=False)])
        if not unique:
            return []

        def _rows(df, name_series):
            values = df[columns].apply(pd.to_numeric, errors='coerce')
            values.index = name_series.values
            return values

        source = _rows(source_df, source_names)
        output = _rows(result_df, names)

        def _key(row) -> tuple:
            return tuple('nan' if pd.isna(v) else f'{v:.10g}' for v in row)

        owner = {}
        for name, row in source.iterrows():
            if row.notna().sum() >= 2:
                owner.setdefault(_key(row), set()).add(name)

        moved, lost = [], []
        for name, row in output.iterrows():
            if name not in unique:
                continue
            own = source.loc[name]
            if row.isna().all() and own.notna().any():
                lost.append(name)
                continue
            key = _key(row)
            others = owner.get(key, set()) - {name}
            if others and key != _key(own):
                moved.append((name, sorted(others)[0]))

        if not moved:
            return []
        example = moved[0]
        message = (
            f"Values do not stay with their lipid: {len(moved)} row(s) carry "
            f"another lipid's values from the uploaded file (e.g. "
            f"'{example[0]}' has the values of '{example[1]}')"
        )
        if lost:
            message += (
                f", and {len(lost)} row(s) lost all their values "
                f"(e.g. '{lost[0]}')"
            )
        message += (
            ". This happens when columns of a filtered DataFrame are combined "
            "by index with a new one: call .reset_index(drop=True) on the "
            "filtered DataFrame before building result_df, or use .values "
            "for every column."
        )
        return [message]

    @staticmethod
    def compare_to_upload(
        result_df: pd.DataFrame, source_df: pd.DataFrame,
    ) -> TransformReport:
        """Report what a transform changed, so the user can check it
        without reading the code (a filter that used ``>`` instead of
        ``>=`` passes every check, but shows here as lipids removed)."""
        names = result_df.iloc[:, 0].astype(str)
        name_col = DataTransformerService._name_column(names, source_df)
        source_names = (
            source_df[name_col].astype(str) if name_col is not None
            else pd.Series([], dtype=str)
        )
        samples = DataTransformerService._sample_columns(result_df)
        columns = [c for c in samples if c in source_df.columns and c != name_col]

        source_set, result_set = set(source_names), set(names)
        compared = changed = appeared = missing = 0
        both = sorted(
            set(source_names[~source_names.duplicated(keep=False)])
            & set(names[~names.duplicated(keep=False)])
        )
        if columns and both:
            def _cells(df, name_series):
                values = df[columns].apply(pd.to_numeric, errors='coerce')
                values.index = name_series.values
                return values.loc[both]

            before = _cells(source_df, source_names)
            after = _cells(result_df, names)
            compared = before.size
            was, now = before.notna(), after.notna()
            differs = ~np.isclose(before, after, rtol=1e-9, atol=0)
            changed = int((was & now & differs).sum().sum())
            appeared = int((~was & now).sum().sum())
            missing = int((was & ~now).sum().sum())

        return TransformReport(
            rows_uploaded=len(source_df),
            rows_result=len(result_df),
            lipids_removed=tuple(
                n for n in dict.fromkeys(source_names) if n not in result_set
            ),
            lipids_not_in_upload=tuple(
                n for n in dict.fromkeys(names) if n not in source_set
            ),
            sample_columns=tuple(str(c) for c in samples),
            columns_dropped=tuple(
                str(c) for c in source_df.columns
                if c not in result_df.columns and c != name_col
            ),
            cells_compared=compared,
            values_changed=changed,
            values_appeared=appeared,
            values_missing=missing,
            duplicate_names=tuple(dict.fromkeys(names[names.duplicated()])),
        )

    @staticmethod
    def _name_column(names: pd.Series, source_df: pd.DataFrame):
        """The upload's text column that shares the most names with
        ``names`` (the result's lipid names); None if it has none."""
        return max(
            (c for c in source_df.columns if source_df[c].dtype == object),
            key=lambda c: names.isin(source_df[c].astype(str)).sum(),
            default=None,
        )

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
            if looks_like_metadata_name(col):
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
