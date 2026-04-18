"""
DataFrame schema extraction for AI context.

Converts a DataFrame into a concise text description that can be
sent to the AI without exposing raw data — only schema, dtypes,
shape, and a small sample.
"""
import pandas as pd


class SchemaExtractorService:
    """Extracts schema descriptions from DataFrames for AI context.

    All methods are stateless static methods. No Streamlit imports.
    """

    SAMPLE_ROWS = 3

    @staticmethod
    def extract_schema(df: pd.DataFrame) -> str:
        """Extract a text description of a DataFrame's schema.

        Includes shape, column names with dtypes, and a few sample rows.
        Never sends the full dataset — only structural information.

        Args:
            df: Source DataFrame to describe.

        Returns:
            Multi-line text description suitable for an AI system prompt.
        """
        lines = [
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            "",
            "Columns:",
        ]

        for col in df.columns:
            non_null = df[col].notna().sum()
            lines.append(f"  - {col} ({df[col].dtype}, {non_null}/{len(df)} non-null)")

        n_sample = min(SchemaExtractorService.SAMPLE_ROWS, len(df))
        if n_sample > 0:
            sample = df.head(n_sample).to_string(index=False, max_colwidth=40)
            lines.extend(["", f"Sample rows ({n_sample}):", sample])

        return "\n".join(lines)

    @staticmethod
    def extract_generic_format_spec() -> str:
        """Return the target Generic format specification.

        This describes the column structure that the AI must produce
        when transforming arbitrary data into the Generic format.

        Returns:
            Text specification of the Generic format.
        """
        return (
            "Target format: LipidCruncher Generic\n"
            "\n"
            "Structure (very simple):\n"
            "  - Column 1: Lipid names (any naming convention — LipidCruncher will\n"
            "    standardize them to LIPID MAPS notation automatically).\n"
            "    Examples: 'PC(34:1)', 'PC 34:1', 'PE 36:2', 'Cer(d18:1/24:0)'\n"
            "  - Columns 2+: Sample intensities — one column per sample.\n"
            "    Column names can be anything (e.g., 'KO-1', 'WT-2', 'Sample_A').\n"
            "    LipidCruncher will rename them to intensity[s1], intensity[s2], etc.\n"
            "\n"
            "Rules:\n"
            "  - Each row is one lipid species.\n"
            "  - NO extra columns allowed — only lipid names + sample intensities.\n"
            "    Remove any metadata columns (mass, retention time, scores, etc.).\n"
            "  - Missing/NaN values are fine — LipidCruncher handles them.\n"
            "\n"
            "What LipidCruncher does automatically after upload:\n"
            "  - Standardizes lipid names to LIPID MAPS shorthand notation\n"
            "  - Extracts ClassKey from lipid names\n"
            "  - Renames sample columns to intensity[s1], intensity[s2], ...\n"
        )
