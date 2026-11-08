"""
AI chat workflow.

Orchestrates AI-powered data transformation and data questions (with
figures). AIChatWorkflow is the one class the adapter calls; each tab's
methods live in their own module (ai_transform.py, ai_data_questions.py,
ai_lc_questions.py, shared notes in ai_chat_shared.py), which it
inherits. Pure logic — no Streamlit
dependencies.
"""
from .ai_data_questions import (  # noqa: F401 (re-exported)
    DataQuestionsStepResult,
    DataQuestionsWorkflow,
    figure_has_data,
)
from .ai_lc_questions import LCQuestionsWorkflow
from .ai_transform import (  # noqa: F401 (re-exported)
    FormatDiagnosticResult,
    TransformStepResult,
    TransformWorkflow,
)


class AIChatWorkflow(
    TransformWorkflow, DataQuestionsWorkflow, LCQuestionsWorkflow,
):
    """Orchestrates AI chat interactions.

    All methods are stateless static methods. No Streamlit imports.
    """
