"""
AI-powered visualization generation service.

Generates custom Plotly/Matplotlib visualizations from natural language
requests. All methods are stateless static methods — no Streamlit imports.
"""
from typing import Optional, Tuple

import pandas as pd

from ...models.chat import AIConfig, SandboxConfig
from ...models.experiment import ExperimentConfig
from .ai_client import AIClientService
from .sandbox import SandboxResult, SandboxService
from .schema_extractor import SchemaExtractorService


def _viz_system_prompt() -> str:
    """System prompt for the visualization chat mode."""
    return """\
You are a lipidomics visualization assistant for LipidCruncher.

Your job is to create custom data visualizations from the user's analyzed lipidomic data \
based on their natural language requests.

## Instructions

1. The user will describe what visualization they want. You will also receive the DataFrame \
schema and experiment structure.
2. Write Python code that produces a Plotly figure assigned to the variable `fig`.
3. Always wrap your code in a ```python code block.
4. Available variables in the execution environment:
   - `df` — the user's analyzed DataFrame (pandas)
   - `pd` — pandas
   - `np` — numpy
   - `px` — plotly.express
   - `go` — plotly.graph_objects
   - `plt` — matplotlib.pyplot (use only if the user specifically requests matplotlib)
   - `sns` — seaborn (use only if needed with matplotlib)
   - `math`, `statistics` — standard library math modules
5. Do NOT import any modules — they are pre-loaded.
6. The result MUST be a Plotly figure assigned to `fig`. For example:
   ```
   fig = px.bar(df, x='ClassKey', y='intensity[s1]')
   ```

## App Color Palette

Use these colors for consistency with the rest of the application:
- Conditions: '#1f77b4' (blue), '#ff7f0e' (orange), '#2ca02c' (green), '#d62728' (red), '#9467bd' (purple)
- Lipid classes: use plotly's default color sequence or the condition colors above

## Visualization Tips

- For comparing conditions: use grouped bar charts, box plots, or violin plots
- For composition: use pie charts or stacked bar charts
- For distributions: use histograms, KDE plots, or box plots
- For relationships: use scatter plots or heatmaps
- For volcano plots: use scatter with -log10(p-value) vs log2(fold-change)
- Always add clear titles, axis labels, and legends
- Use `fig.update_layout()` for customization (font size, margins, etc.)
- Handle missing/NaN values gracefully

## Important
- Always produce a Plotly `fig` variable.
- If the user's request is ambiguous, ask for clarification before generating code.
- If the data doesn't support the requested visualization, explain why and suggest alternatives.
"""


def _build_experiment_context(experiment: ExperimentConfig) -> str:
    """Build a text description of the experiment structure for AI context.

    Args:
        experiment: The experiment configuration.

    Returns:
        Text description of the experiment.
    """
    lines = [
        f"Number of conditions: {experiment.n_conditions}",
        f"Conditions: {', '.join(experiment.conditions_list)}",
        f"Samples per condition: {', '.join(str(n) for n in experiment.number_of_samples_list)}",
        f"Sample labels: {', '.join(experiment.full_samples_list)}",
        "",
        "Sample-to-condition mapping:",
    ]
    for condition, samples in zip(experiment.conditions_list, experiment.individual_samples_list):
        lines.append(f"  {condition}: {', '.join(samples)}")
    return "\n".join(lines)


class VizGeneratorService:
    """Generates custom visualizations via AI.

    All methods are stateless static methods.
    """

    @staticmethod
    def generate_viz_code(
        user_request: str,
        schema_context: str,
        experiment: ExperimentConfig,
        history: list,
        config: Optional[AIConfig] = None,
    ) -> Tuple[str, str]:
        """Ask the AI to generate visualization code.

        Args:
            user_request: The user's visualization request.
            schema_context: Schema description of the analysis DataFrame.
            experiment: Experiment configuration (conditions, samples).
            history: Previous conversation messages in API format.
            config: AI configuration. Uses defaults if None.

        Returns:
            Tuple of (explanation, code).

        Raises:
            AIClientError: On API communication failure.
        """
        if config is None:
            config = AIConfig()

        experiment_context = _build_experiment_context(experiment)
        context = (
            f"## DataFrame Schema\n{schema_context}\n\n"
            f"## Experiment Structure\n{experiment_context}"
        )

        augmented_message = f"{user_request}\n\n{context}"
        messages = history + [{"role": "user", "content": augmented_message}]

        return AIClientService.send_message_for_code(
            messages, _viz_system_prompt(), config
        )

    @staticmethod
    def execute_viz_code(
        code: str,
        df: pd.DataFrame,
        sandbox_config: Optional[SandboxConfig] = None,
    ) -> SandboxResult:
        """Execute visualization code in the sandbox.

        Args:
            code: Python code that should produce a ``fig`` variable.
            df: The analysis DataFrame.
            sandbox_config: Sandbox configuration. Uses defaults if None.

        Returns:
            SandboxResult — check ``result.figure`` for the generated plot.
        """
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        return SandboxService.execute(code, df, sandbox_config)
