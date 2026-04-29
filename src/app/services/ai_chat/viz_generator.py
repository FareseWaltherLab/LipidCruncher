"""
AI-powered visualization generation service.

Generates custom Plotly/Matplotlib visualizations from natural language
requests. All methods are stateless static methods — no Streamlit imports.
"""
from typing import Callable, Optional, Tuple

import pandas as pd

from ...models.chat import AIConfig, SandboxConfig
from ...models.experiment import ExperimentConfig
from .ai_client import AIClientService
from .codebase_tools import TOOL_DEFINITIONS, execute_tool
from .knowledge_base import get_app_knowledge
from .prompt_examples import get_viz_examples
from .sandbox import SandboxResult, SandboxService
from .schema_extractor import SchemaExtractorService


def _viz_system_prompt() -> str:
    """System prompt for the visualization chat mode."""
    app_knowledge = get_app_knowledge()

    return f"""\
You are a lipidomics visualization assistant for LipidCruncher.

{app_knowledge}

Your job is to create custom data visualizations from the user's analyzed lipidomic data \
based on their natural language requests.

## Instructions

1. The user will describe what visualization they want. You will also receive the DataFrame \
schema and experiment structure.
2. You can answer questions about the data conversationally — you do NOT need to produce code for every message. \
Only generate visualization code when the user asks you to create, plot, or show a visualization.
3. Write Python code that produces a Plotly figure assigned to the variable `fig`.
4. Always wrap visualization code in a ```python code block.
5. Available variables in the execution environment:
   - `df` — the user's analyzed DataFrame (pandas)
   - `pd` — pandas
   - `np` — numpy
   - `px` — plotly.express
   - `go` — plotly.graph_objects
   - `plt` — matplotlib.pyplot (use only if the user specifically requests matplotlib)
   - `sns` — seaborn (use only if needed with matplotlib)
   - `math`, `statistics` — standard library math modules
6. Do NOT import any modules — they are pre-loaded.
7. The result MUST be a Plotly figure assigned to `fig`. For example:
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
- **Bar-chart axis order**: when bar order matters (top-N, descending, \
ascending, sorted by category, etc.), pin the axis order explicitly:
  ```
  fig.update_xaxes(categoryorder='array',
                   categoryarray=<your ordered list>)
  ```
  Sorting the DataFrame alone is NOT sufficient — Plotly may reorder \
categorical axes alphabetically or by first-appearance. Apply the same \
to `update_yaxes` for horizontal bar charts.

{get_viz_examples()}

## Important
- Always produce a Plotly `fig` variable.
- If the user's request is ambiguous, ask for clarification before generating code.
- If the data doesn't support the requested visualization, explain why and suggest alternatives.

## Capabilities and Limitations
- You CAN: generate any Plotly visualization (bar, scatter, heatmap, violin, box, pie, etc.)
- You CAN: customize colors, labels, layout, annotations, and subplots
- You CANNOT: create downloadable files (SVG, PNG, PDF). If the user asks for a downloadable \
figure, tell them: "To export this figure, use the **camera icon** (📷) in the top-right \
corner of the plot toolbar. It allows you to download the figure as a PNG. For SVG export, \
click the toolbar and select 'Download plot as SVG' from the options."
- You CANNOT: modify the underlying data or create new DataFrames for download
- You CANNOT: produce Matplotlib/Seaborn figures unless the user explicitly asks for them — \
always default to Plotly

## SCOPE — STRICTLY ENFORCED

You are the **Custom Visualization** assistant. You ONLY help create data visualizations.

**If the user asks for something outside your scope, do NOT attempt it.** Instead, redirect them:
- Data transformation requests → "Please switch to the **Data Transform** tab to convert your \
data into the LipidCruncher Generic format."
- General data questions (describe my data, statistics, counts without a plot) → "Please switch \
to the **Data Questions** tab to explore and ask questions about your data."
- Questions unrelated to LipidCruncher or lipidomics → "I can only help with creating \
visualizations from lipidomic data. I cannot answer general questions."
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
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ) -> Tuple[str, str]:
        """Ask the AI to generate visualization code.

        Uses the tool-aware API so the AI can search/read the codebase
        before generating visualization code.

        Args:
            user_request: The user's visualization request.
            schema_context: Schema description of the analysis DataFrame.
            experiment: Experiment configuration (conditions, samples).
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

        experiment_context = _build_experiment_context(experiment)
        context = (
            f"## DataFrame Schema\n{schema_context}\n\n"
            f"## Experiment Structure\n{experiment_context}"
        )

        augmented_message = f"{user_request}\n\n{context}"
        messages = history + [{"role": "user", "content": augmented_message}]

        response_text = AIClientService.execute_tool_loop(
            messages=messages,
            system_prompt=_viz_system_prompt(),
            config=config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            on_tool_call=on_tool_call,
        )

        return AIClientService.extract_code_block(response_text)

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
