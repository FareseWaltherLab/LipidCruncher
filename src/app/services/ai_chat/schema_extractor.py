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
            "Required columns:\n"
            "  - LipidMolec (str): Lipid species name using LIPID MAPS shorthand notation.\n"
            "    Examples: 'PC 34:1', 'PE 36:2', 'Cer 18:1;O2/24:0'\n"
            "  - ClassKey (str): Lipid class abbreviation.\n"
            "    Examples: 'PC', 'PE', 'Cer', 'TG', 'SM'\n"
            "  - intensity[s1], intensity[s2], ... (float): One column per sample,\n"
            "    named intensity[s1] through intensity[sN]. Contains raw intensity/area values.\n"
            "\n"
            "Optional columns:\n"
            "  - CalcMass (float): Calculated molecular mass.\n"
            "  - BaseRt (float): Retention time in minutes.\n"
            "\n"
            "Rules:\n"
            "  - Each row is one lipid species.\n"
            "  - Sample columns must be named exactly intensity[s1], intensity[s2], etc.\n"
            "  - Lipid names should follow LIPID MAPS notation: 'Class TotalC:TotalDB'\n"
            "    (e.g., 'PC 34:1', not 'PC(34:1)').\n"
            "  - ClassKey must match the class prefix in LipidMolec.\n"
            "  - Missing values should be 0 or NaN.\n"
        )
