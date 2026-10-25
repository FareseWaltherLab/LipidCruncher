"""
AI-powered visualization generation service.

Generates custom Plotly visualizations from natural language
requests. All methods are stateless static methods — no Streamlit imports.
"""
from typing import Callable, List, Optional

import pandas as pd

from ...models.chat import AIConfig, SandboxConfig
from ...models.experiment import ExperimentConfig
from .ai_client import AIClientService
from .codebase_tools import TOOL_DEFINITIONS, execute_tool
from .knowledge_base import get_app_knowledge
from .prompt_examples import get_viz_examples
from .prompt_templates import (
    METHODOLOGY_CLARIFICATION_RULES,
    PIPELINE_FIRST_RULES,
    analysis_modules_text,
    data_block,
    with_data_note,
)
from .sandbox import SandboxResult, SandboxService


def _viz_system_prompt() -> str:
    """System prompt for the visualization chat mode."""
    app_knowledge = get_app_knowledge()

    return with_data_note(f"""\
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
   - `lc` — LipidCruncher's own analysis pipeline (see "Pipeline first" below — check this before writing your own analysis)
   - `pd` — pandas
   - `np` — numpy
   - `px` — plotly.express
   - `go` — plotly.graph_objects
   - `math`, `statistics` — standard library math modules
   - `re` — standard library regex module (useful for parsing lipid names, e.g. extracting fatty acid chain composition like C16/C18)
6. Modules are pre-loaded; {analysis_modules_text()} may also be imported when you genuinely need an analysis `lc` does not cover (import the submodule itself: `from scipy import stats`, not `import scipy`). Nothing else is importable — in particular not matplotlib or seaborn, so if the user asks for one of those, say so and draw the plot with Plotly.
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
- For volcano plots: use `lc.volcano(...)`, the app's own volcano plot
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

- **Coloring bars by category — use `lc.colored_bars`**: when each x \
position has exactly ONE bar colored by some category (top-N lipids by \
ClassKey, fold changes by direction), put the category in a column of the \
table, sort and filter the table as you need, then:
  ```
  top10['Direction'] = np.where(top10['log2FC'] > 0, 'Up in KO', 'Down in KO')
  top10 = top10.sort_values('log2FC')
  fig = lc.colored_bars(top10, x='log2FC', y='LipidMolec', color_by='Direction',
                        colors={{'Up in KO': '#d62728', 'Down in KO': '#1f77b4'}},
                        orientation='h').figure
  ```
  It draws one evenly spaced bar per row in the table's order, reads each \
bar's color from its own row (so sorting first is safe), adds a legend of \
the categories and pins the category axis order. Do NOT use \
`px.bar(..., color='ClassKey')` for this: it splits the data into one \
trace per category, and Plotly's per-trace offsetgroup layout produces \
uneven bar spacing.

  **Any other per-point list you build yourself** (colors, text, hover \
labels): build it last, from the DataFrame in its final order — do every \
sort and filter first. A list built before a re-sort labels each point \
with another row's value — e.g. a falling lipid drawn in the "up" color.

  When multiple bars genuinely share an x position (e.g., grouped/ \
stacked bars per condition), `px.bar(..., color=...)` is fine — use \
`barmode='stack'` or `'group'` as appropriate.

<examples>
{get_viz_examples()}
</examples>

{PIPELINE_FIRST_RULES}

{METHODOLOGY_CLARIFICATION_RULES}

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
- Data transformation requests → "Please switch to the **Transform to Generic** tab to convert your \
data into the LipidCruncher Generic format."
- General data questions (describe my data, statistics, counts without a plot) → "Please switch \
to the **Data Questions** tab to explore and ask questions about your data."
- Questions unrelated to LipidCruncher or lipidomics → "I can only help with creating \
visualizations from lipidomic data. I cannot answer general questions."
""")


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
    def stream_viz_code(
        user_request: str,
        schema_context: str,
        experiment: ExperimentConfig,
        history: list,
        config: AIConfig,
        holder: dict,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
    ):
        """Ask the AI for visualization code, streaming its explanation.

        Uses the tool-aware loop so the AI can search/read the codebase
        before writing code. Returns a generator that yields the
        explanation prose as it streams. The complete raw response
        (explanation + code) is accumulated into ``holder['full']`` so the
        caller can extract and execute the code once streaming completes.
        """
        experiment_context = _build_experiment_context(experiment)
        context = (
            f"{data_block('dataframe_schema', schema_context)}\n\n"
            f"<experiment_structure>\n{experiment_context}\n</experiment_structure>"
        )

        augmented_message = f"{context}\n\n{user_request}"
        messages = history + [{"role": "user", "content": augmented_message}]

        return AIClientService.stream_explanation(
            messages=messages,
            system_prompt=_viz_system_prompt(),
            config=config,
            tools=TOOL_DEFINITIONS,
            tool_executor=execute_tool,
            holder=holder,
            on_tool_call=on_tool_call,
        )

    @staticmethod
    def execute_viz_code(
        code: str,
        df: pd.DataFrame,
        sandbox_config: Optional[SandboxConfig] = None,
        numeric_columns: Optional[List[str]] = None,
        lc=None,
    ) -> SandboxResult:
        """Execute visualization code in the sandbox.

        Args:
            code: Python code that should produce a ``fig`` variable.
            df: The analysis DataFrame.
            sandbox_config: Sandbox configuration. Uses defaults if None.
            numeric_columns: Optional list of columns to pre-coerce to
                numeric before exposing the DataFrame to user code.
            lc: Optional ``PipelineAPI`` bound into the code's globals as
                ``lc``.

        Returns:
            SandboxResult — check ``result.figure`` for the generated plot.
        """
        if sandbox_config is None:
            sandbox_config = SandboxConfig()

        return SandboxService.execute(
            code, df, sandbox_config, numeric_columns=numeric_columns,
            lc=lc,
        )
