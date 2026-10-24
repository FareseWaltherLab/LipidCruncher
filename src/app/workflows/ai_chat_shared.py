"""
Notes and checks shared by the AI chat tabs' workflows: numbers the model
stated that its code never produced, the statistical-choices note, and the
repeated-lipids context. Pure logic — no Streamlit dependencies.
"""
from typing import Optional, Tuple

import pandas as pd

from ..models.chat import ChatSession
from ..services.ai_chat.output_checks import computed_values, numbers_not_computed


def _unverified_numbers(
    session: ChatSession,
    explanation: str,
    code: str,
    stdout: str = "",
    result_df: Optional[pd.DataFrame] = None,
    figure=None,
) -> Tuple[str, ...]:
    """Numbers in the model's explanation that its code never produced.

    ``session`` ends with this turn's user and assistant messages; the
    question, the code and the data's schema are the numbers' legitimate
    sources besides the computed output. Logged so the rate can be
    measured before anything is shown to users (plan B2)."""
    question = session.messages[-2].content if len(session.messages) >= 2 else ""
    return numbers_not_computed(
        explanation,
        computed_values(stdout, result_df, figure),
        context="\n".join([question, code or "", session.schema_context or ""]),
    )


def _repeated_lipids_note(df: pd.DataFrame) -> str:
    """Context for data with one row per identification, not per lipid.

    A raw LipidSearch export (uploaded here, or imported as "Raw uploaded
    data") lists a lipid once per adduct / polarity it was seen in.
    Counting or summing those rows disagrees with the pipeline, which
    keeps one row per lipid. Empty when every lipid name is unique.
    """
    if 'LipidMolec' not in df.columns:
        return ""
    names = df['LipidMolec'].astype(str)
    repeated = names[names.duplicated(keep=False)]
    if repeated.empty:
        return ""
    why = (
        "the same lipid identified as different adducts (see MainIon) or in "
        "both polarities" if 'MainIon' in df.columns
        else "the same lipid identified more than once"
    )
    return (
        "\n\n## Rows are identifications, not lipids\n"
        f"{repeated.nunique()} lipid names appear on more than one row "
        f"({len(repeated)} rows; e.g. `{repeated.iloc[0]}`): {why}. Counting "
        f"rows counts such a lipid more than once ({names.nunique()} distinct "
        f"names in {len(names)} rows), and its rows are on different signal "
        "scales, so never add them. LipidCruncher's pipeline keeps one row per "
        "lipid (in its class's internal-standard adduct, otherwise the best "
        "grade). For counts, totals or comparisons, say this, count distinct "
        "lipids, and suggest processing the file in the main app (Module 1) "
        "and importing the QC-filtered data, which has one row per lipid."
    )


def _methods_note(methods) -> str:
    """The statistical choices `lc` made, as a note under the answer."""
    lines = "\n".join(f"- {m}" for m in methods)
    return (
        f"**Statistical choices** (ask if you want any of them changed):\n"
        f"{lines}"
    )
