"""
AI chat workflow.

Orchestrates AI-powered data transformation, visualization generation and
data questions. AIChatWorkflow is the one class the adapter calls; each
tab's methods live in their own module (ai_transform.py, ai_viz.py,
ai_data_questions.py, ai_lc_questions.py, shared notes in
ai_chat_shared.py), which it inherits. Pure logic — no Streamlit
dependencies.
"""
from .ai_data_questions import DataQuestionsStepResult, DataQuestionsWorkflow  # noqa: F401 (re-exported)
from .ai_lc_questions import LCQuestionsWorkflow
from .ai_transform import (  # noqa: F401 (re-exported)
    FormatDiagnosticResult,
    TransformStepResult,
    TransformWorkflow,
)
from .ai_viz import (  # noqa: F401 (re-exported)
    Module1ValidationResult,
    VizStepResult,
    VizWorkflow,
    figure_has_data,
)


class AIChatWorkflow(
    TransformWorkflow, VizWorkflow, DataQuestionsWorkflow, LCQuestionsWorkflow,
):
    """Orchestrates AI chat interactions.

    All methods are stateless static methods. No Streamlit imports.
    """
