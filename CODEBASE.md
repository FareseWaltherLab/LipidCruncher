# LipidCruncher — Codebase Documentation

A professional lipidomics data analysis application built with Streamlit, developed by the Farese & Walther Lab at Memorial Sloan Kettering Cancer Center. LipidCruncher enables researchers to process, analyze, and visualize lipidomic data from multiple sources without requiring bioinformatics expertise.

The app also includes an **AI Studio** — a standalone page with a Claude-powered chat that lets users ask questions about their data, transform arbitrary CSVs into LipidCruncher's Generic format, and generate custom Plotly visualizations from natural language. AI-generated code runs in a sandboxed execution environment.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Application Flow](#application-flow)
5. [Layer-by-Layer Reference](#layer-by-layer-reference)
   - [Models](#models)
   - [Services](#services)
   - [Workflows](#workflows)
   - [Adapter](#adapter)
   - [UI](#ui)
6. [Supported Data Formats](#supported-data-formats)
7. [Session State Management](#session-state-management)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)
11. [Extending the Application](#extending-the-application)

---

## Quick Start

```bash
# Create and activate virtual environment
python -m venv lipidcruncher_env
source lipidcruncher_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/main_app.py

# Run tests
pytest tests/ -v
```

The app opens at `http://localhost:8501`.

---

## Architecture Overview

LipidCruncher follows a **layered architecture** with strict dependency rules — each layer only imports from the layer below it:

```
┌─────────────────────────────────────────────────┐
│  UI Layer  (Streamlit components)               │
│  src/main_app.py + src/app/ui/                  │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Adapter Layer  (session state + caching)        │
│  src/app/adapters/streamlit_adapter.py           │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Workflow Layer  (orchestration)                  │
│  src/app/workflows/                              │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Services Layer  (business logic)                │
│  src/app/services/                               │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Models Layer  (data structures)                 │
│  src/app/models/                                 │
└─────────────────────────────────────────────────┘
```

**Key design principles:**

- **Services and workflows are pure** — they have zero Streamlit dependencies, use only static methods, and are fully testable in isolation.
- **Models are immutable** — Pydantic models with `frozen=True` make them hashable and safe for `@st.cache_data`.
- **The adapter is the only bridge** — `StreamlitAdapter` is the sole point where Streamlit concerns (caching, session state) meet business logic.
- **UI modules are thin** — they call adapter/workflow methods and render results; no data processing logic lives here.
- **AI-generated code is untrusted** — the AI Studio adds an extra security layer: every model output goes through `SandboxService` (AST validation + curated builtins + timeout) before execution. The sandbox is the security boundary, not AI compliance.

---

## Directory Structure

```
LipidCruncher/
├── src/
│   ├── main_app.py                          # Entry point (enum-based page routing)
│   └── app/
│       ├── constants.py                     # Shared constants, enums, format mappings
│       ├── lipid_nomenclature.py            # LIPID MAPS nomenclature functions
│       ├── models/                          # Pydantic data models
│       │   ├── experiment.py                #   ExperimentConfig
│       │   ├── normalization.py             #   NormalizationConfig
│       │   ├── statistics.py                #   StatisticalTestConfig
│       │   ├── chat.py                      #   ChatMessage, ChatSession, ChatMode, AIConfig, SandboxConfig
│       │   └── isotope_tracing.py           #   IsotopeCorrectionConfig
│       ├── services/                        # Pure business logic
│       │   ├── __init__.py                  #   Public API: re-exports services + exceptions
│       │   ├── exceptions.py                #   Typed exception hierarchy (ServiceError, etc.)
│       │   ├── format_detection.py          #   DataFormat enum + detection
│       │   ├── data_standardization.py      #   Column name standardization
│       │   ├── lipidsearch_alignment.py     #   LipidSearch 5.2 dual-polarity pairing + merge
│       │   ├── data_cleaning/               #   Format-specific cleaning
│       │   │   ├── __init__.py              #     Registry dispatch (DataCleaningService)
│       │   │   ├── exceptions.py            #     Domain-specific cleaning exceptions
│       │   │   ├── base.py                  #     BaseDataCleaner
│       │   │   ├── configs.py               #     GradeFilterConfig, QualityFilterConfig
│       │   │   ├── lipidsearch.py           #     LipidSearchCleaner
│       │   │   ├── msdial.py               #     MSDIALCleaner
│       │   │   └── generic.py               #     GenericCleaner
│       │   ├── normalization.py             #   NormalizationService
│       │   ├── quality_check.py             #   QualityCheckService
│       │   ├── statistical_testing.py       #   StatisticalTestingService
│       │   ├── standards.py                 #   Standards extraction/validation
│       │   ├── validation.py                #   Data validation helpers
│       │   ├── zero_filtering.py            #   ZeroFilteringService
│       │   ├── sample_grouping.py           #   Sample assignment to conditions
│       │   ├── report_generator.py          #   PDF report generation
│       │   ├── lsi_report.py               #   LSI compliance checklist generation
│       │   ├── ai_chat/                     #   AI Studio service layer (Anthropic API + sandbox)
│       │   │   ├── __init__.py              #     Public API: re-exports AI services + exceptions
│       │   │   ├── exceptions.py            #     AIChatError, AIClientError, SandboxError, TransformationError
│       │   │   ├── ai_client.py             #     Anthropic SDK wrapper + tool-use loop (blocking + streaming)
│       │   │   ├── sandbox.py               #     Sandboxed execution, parent side (validate → spawn child → time out)
│       │   │   ├── sandbox_runner.py        #     Child-process entry point that actually runs the AI's code
│       │   │   ├── interaction_logger.py    #     One JSON record per chat turn / feedback event (stdout → CloudWatch)
│       │   │   ├── schema_extractor.py      #     DataFrame → schema description for prompts
│       │   │   ├── codebase_tools.py        #     search_codebase / read_file_section / list_files / read_documentation tools
│       │   │   ├── knowledge_base.py        #     Condensed app knowledge embedded in system prompts
│       │   │   ├── prompt_templates.py      #     Per-mode system prompts (transform, viz, data Q, LC Q, unified)
│       │   │   ├── prompt_examples.py       #     Few-shot example prompts shown in chat UI
│       │   │   ├── data_transformer.py      #     DataTransformerService (CSV → Generic format)
│       │   │   └── viz_generator.py         #     VizGeneratorService (NL → Plotly figure)
│       │   ├── isotope_correction/          #   Stable isotope tracing (IsoCorrectoR)
│       │   │   ├── service.py               #     IsotopeCorrectionService (Rscript subprocess)
│       │   │   ├── run_isocorrector.R       #     Bundled trusted R wrapper
│       │   │   ├── validation.py            #     Pure-Python input validation
│       │   │   ├── r_runtime.py             #     R/IsoCorrectoR availability detection
│       │   │   └── exceptions.py            #     IsotopeError hierarchy
│       │   └── plotting/                    #   Visualization services
│       │       ├── __init__.py              #     Public API: re-exports all plotters + shared utils
│       │       ├── base.py                  #     PlotterServiceProtocol (interface contract)
│       │       ├── _shared.py               #     Shared utilities (colors, significance, validation)
│       │       ├── box_plot.py
│       │       ├── bqc_plotter.py
│       │       ├── correlation.py
│       │       ├── pca.py
│       │       ├── retention_time.py
│       │       ├── standards_plotter.py
│       │       ├── abundance_bar_chart.py
│       │       ├── abundance_pie_chart.py
│       │       ├── saturation_plot.py
│       │       ├── chain_length_plot.py
│       │       ├── fach.py
│       │       ├── pathway_viz.py
│       │       ├── volcano_plot.py
│       │       └── lipidomic_heatmap.py
│       ├── workflows/                       # Orchestration layer
│       │   ├── data_ingestion.py            #   DataIngestionWorkflow
│       │   ├── normalization.py             #   NormalizationWorkflow
│       │   ├── quality_check.py             #   QualityCheckWorkflow
│       │   ├── analysis.py                  #   AnalysisWorkflow
│       │   ├── ai_chat.py                   #   AIChatWorkflow (transform / viz / data Q / LC Q / unified)
│       │   ├── pipeline_api.py              #   PipelineAPI — the `lc` object AI-generated code calls into
│       │   └── isotope_tracing.py           #   IsotopeTracingWorkflow
│       ├── adapters/
│       │   └── streamlit_adapter.py         #   StreamlitAdapter
│       └── ui/                              # Streamlit UI components
│           ├── landing_page.py
│           ├── isotope_tracing_page.py      #   Stable isotope tracing page
│           ├── format_requirements.py
│           ├── zero_filtering.py
│           ├── standards_plots.py
│           ├── download_utils.py
│           ├── st_helpers.py                #   Shared UI helpers (export buttons, section headers)
│           ├── ai_studio_page.py            #   Standalone AI Studio page (4 tabs + sidebar data controls)
│           ├── ai_chat/                     #   AI chat panels (per-tab UI)
│           │   ├── __init__.py              #     Re-exports tab-specific display_* functions
│           │   ├── chat_panel.py            #     Shared chat-rendering helpers
│           │   ├── _lipidcruncher_questions_ui.py  # Tab 1: app Q&A
│           │   ├── _data_questions_ui.py    #     Tab 2: dataset exploration
│           │   ├── _data_transform_ui.py    #     Tab 3: CSV → Generic transformation
│           │   ├── _viz_generation_ui.py    #     Tab 4: custom visualization
│           │   ├── _interaction_log.py      #     Per-turn logging + thumbs / comment feedback widgets
│           │   └── _unified_chat_ui.py      #     Single-chat variant (legacy / fallback)
│           ├── sidebar/                     #   Sidebar input components
│           │   ├── file_upload.py
│           │   ├── column_mapping.py
│           │   ├── experiment_config.py
│           │   ├── sample_grouping.py
│           │   └── confirm_inputs.py
│           ├── main_content/                #   Main area content
│           │   ├── data_processing.py
│           │   ├── internal_standards.py
│           │   ├── normalization.py
│           │   ├── quality_check.py
│           │   ├── lsi_report.py            #   LSI compliance report UI
│           │   └── analysis/                #   Analysis visualizations
│           │       ├── _entry.py
│           │       ├── _shared.py
│           │       ├── _utils.py
│           │       ├── _bar_chart.py
│           │       ├── _pie_charts.py
│           │       ├── _saturation.py
│           │       ├── _chain_length.py
│           │       ├── _fach.py
│           │       ├── _pathway.py
│           │       ├── _pathway_state.py    #     Pathway session state management
│           │       ├── _pathway_editor.py   #     Pathway layout customization panel
│           │       ├── _volcano.py
│           │       └── _heatmap.py
│           └── content/                     #   Static docs/help text
│               ├── sample_data.py
│               ├── processing_docs.py
│               ├── normalization_docs.py
│               ├── standards_help.py
│               └── analysis_docs.py         #   Statistical testing & saturation docs
├── tests/
│   ├── conftest.py                          # Shared fixtures
│   ├── unit/                                # 64 unit test modules (incl. AI chat models, sandbox isolation, codebase tools, transformer, viz generator, schema extractor, scope, validation, adapter, workflow)
│   ├── integration/                         # 6 integration test modules (incl. test_ai_tool_use.py, test_isotope_tracing_golden.py)
│   └── ui/                                  # 10 UI test modules (incl. test_navigation.py for AI Studio routing)
├── sample_datasets/                         # Test data (one per format)
│   ├── lipidsearch5_test_dataset.csv
│   ├── msdial_test_dataset.csv
│   ├── generic_test_dataset.csv
│   ├── mw_test_dataset.csv
│   └── isotope_tracing/                     # IsoCorrectoR inputs + golden output
├── images/                                  # UI assets (logo, module PDFs)
├── .streamlit/config.toml                   # Server settings (upload cap: maxUploadSize = 1024 MB)
├── requirements.txt
├── pytest.ini
├── Dockerfile                               # Single image: app + AI Studio + R/IsoCorrectoR
└── README.md
```

---

## Application Flow

The app has four pages: a **landing page**, the **main app page**, an **AI Studio page**, and the **isotope tracing page**. The landing page asks *where do you want to start?* and offers all three destinations as stacked buttons — the main lipidomics app (`Page.APP`), the stable isotope tracing workflow (`Page.ISOTOPE_TRACING`), and the AI Studio (`Page.AI_STUDIO`) — each with a one-line caption. The main app page has two modules that the user progresses through sequentially. The AI Studio is also reachable from the end of Module 2 (so users can take pipeline output into AI-driven exploration); a round-trip back to the main app preserves sidebar widget state via `StreamlitAdapter.preserve_main_app_widgets()` / `restore_navigation_widgets()`.

### Module 1: Data Cleaning, Filtering, & Normalization

```
CSV Upload → Format Detection → Column Standardization → Data Cleaning
    → Zero Filtering → Internal Standards Management → Normalization
```

1. **Format detection** — auto-detects or user selects the input format.
2. **Column standardization** — maps format-specific column names (e.g., `MeanArea[s1]`) to standard names (`intensity[s1]`).
3. **Data cleaning** — format-specific pipeline (grade filtering for LipidSearch, score filtering for MS-DIAL, basic cleanup for Generic).
4. **Zero filtering** — removes lipid species with excessive missing values (configurable thresholds).
5. **Internal standards** — auto-detected via regex patterns; user can review, remove, or upload custom standards.
6. **Normalization** — five options: none, internal standard, protein, internal standards + protein, or total intensity.

**Output:** `normalized_df` stored in session state.

### Module 2: Quality Check & Analysis (single page)

**Quality Check section:**
```
Box Plots → BQC Assessment → Retention Time → Correlation → PCA
```

- **Box plots** — concentration distributions and missing value percentages per sample.
- **BQC assessment** — coefficient of variation across BQC samples; user can remove high-CoV lipids.
- **Retention time** — RT plots (LipidSearch/MS-DIAL only, when BaseRt column exists).
- **Pairwise correlation** — Pearson correlation heatmaps between replicates.
- **PCA** — principal component analysis; user can remove outlier samples.

**Analysis section:**
```
Bar Charts | Pie Charts | Saturation | Chain Length | FACH | Pathway | Volcano | Heatmap
```

Each analysis type is independent and user-selectable. All produce interactive plots with export options (SVG, CSV, PDF).

### AI Studio (separate page)

Reached via `Page.AI_STUDIO`. Independent of the main pipeline — users can upload a fresh CSV or import the current pipeline DataFrame in one of three states (`chat_studio_pipeline_source`): `raw` (standardized column names, not yet filtered or normalized), `normalized` (Module 1 output), or `qc_filtered` (Module 2 output, with high-CoV lipids and PCA-removed samples excluded). The page exposes four tabs:

1. **LipidCruncher Questions** — Q&A about the app's features and methods. No data required. The AI may use codebase / documentation tools to ground its answers.
2. **Data Questions** — explore and summarize the uploaded dataset. AI generates pandas code that runs in the sandbox; results appear as inline tables and printed text.
3. **Transform to Generic** — convert an arbitrary CSV into LipidCruncher's Generic format. A format diagnostic guesses lipid-name / intensity / metadata columns; the AI iterates on transformation code until the output validates.
4. **Custom Visualization** — generate Plotly figures from natural-language requests against Module 1 output (gated by `Module1ValidationResult`).

All AI-generated code runs through `SandboxService`, which executes it in a hardened child process (`sandbox_runner.py`) with AST validation and curated builtins in front as defense-in-depth. The AI runs in a tool-use loop (`AIClientService.execute_tool_loop`, or `stream_tool_loop` for the token-by-token variant every tab uses) so it can `search_codebase`, `read_file_section`, `list_files`, or `read_documentation` before answering, and generated code can call into the pipeline itself through `PipelineAPI` (exposed as `lc`). Each turn — and each thumbs-up/down with optional comment — is logged as a JSON record by `interaction_logger`.

In-app guidance sits alongside the tabs: page-level *How to use the AI Studio* and *How the AI works — and its limits* expanders (what reaches the API, where code runs, session caps of 50 messages / 20 executions), plus an *About this tab* expander in each of the four tabs.

### Stable Isotope Tracing (separate workflow)

Reached from the landing-page mode choice (`Page.ISOTOPE_TRACING`); fully independent of Modules 1–2.

```
Upload 3 files (Measurement, Molecule, Element) → Validate → IsoCorrectoR (Rscript) → Corrected tables + CSV download
```

1. **Upload / sample data** — three IsoCorrectoR CSVs, or the bundled example.
2. **Settings** — `IsotopeCorrectionConfig` (tracer-impurity, UHR mode, etc.).
3. **Validation** — pure-Python structural checks before R is invoked.
4. **Correction** — `IsotopeCorrectionService` runs the real IsoCorrectoR via an `Rscript` subprocess.
5. **Results** — corrected intensities, fractions, mean enrichment, residuals, raw data — viewed and downloaded as CSV; run log shown.

Requires R + IsoCorrectoR; degrades gracefully (clear notice) where R is absent.

---

## Layer-by-Layer Reference

### Models

Located in `src/app/models/`. All are frozen Pydantic `BaseModel` subclasses (immutable, hashable).

| Model | File | Purpose |
|---|---|---|
| `ExperimentConfig` | `experiment.py` | Defines experimental setup: number of conditions, condition labels, sample counts. Auto-derives `n_conditions` from `conditions_list` if omitted. Validates condition label uniqueness. Computed properties generate sample labels (s1, s2, ...) and group them by condition. Provides `without_samples()` for creating configs after sample removal. |
| `NormalizationConfig` | `normalization.py` | Normalization method (`none`, `internal_standard`, `protein`, `both`, `total_intensity`), selected lipid classes, standard-to-class mappings, concentrations. |
| `StatisticalTestConfig` | `statistics.py` | Test type (`parametric`, `non_parametric`, `auto`), multiple-testing correction (`fdr_bh`, `bonferroni`), post-hoc method (`tukey`, `bonferroni`), `auto_transform` flag for log10 transformation. |
| `ChatMessage` | `chat.py` | A single chat message: `role` ("user" / "assistant"), `content`, optional `code`, optional `figure_key`. Implements `__hash__` so `ChatSession` (which holds a tuple of messages) stays hashable for caching. |
| `ChatSession` | `chat.py` | Immutable chat history: `mode` (`ChatMode` enum), tuple of `ChatMessage`, and `schema_context` (pre-computed DataFrame schema). `add_message()` returns a new session; `to_api_messages()` formats history for the Anthropic API and re-injects each assistant message's stored `code` so the model sees its own previous attempts. |
| `ChatMode` | `chat.py` | Enum: `DATA_TRANSFORM`, `VISUALIZATION`, `DATA_QUESTIONS`, `LIPIDCRUNCHER_QUESTIONS`. |
| `AIConfig` | `chat.py` | Anthropic API config: `model`, `max_tokens` (1–8192), `temperature` (0.0–1.0). Defaults to a deterministic Sonnet call. |
| `SandboxConfig` | `chat.py` | Limits for sandboxed execution: `execution_timeout` (seconds), `max_output_rows`, `max_figure_size_bytes`. |
| `IsotopeCorrectionConfig` | `isotope_tracing.py` | One field per IsoCorrectoR `IsoCorrection()` setting (`correct_tracer_impurity`, `correct_tracer_element_core`, `calculate_mean_enrichment`, `ultra_high_res`, `correct_also_monoisotopic`, `calculation_threshold_uhr`). Used by the stable isotope tracing workflow. |

### Services

Located in `src/app/services/`. All classes use **static methods only** — no instance state, no Streamlit imports. A unified public API (`services/__init__.py`) re-exports all services, exceptions, and result dataclasses.

#### Exception Hierarchy (`services/exceptions.py`)

Typed exceptions replace scattered `ValueError` raises, enabling callers to distinguish error categories. All inherit from `ValueError` for backward compatibility.

```
ServiceError (base)
├── ConfigurationError   — user-fixable settings (overly strict filters, invalid config)
├── EmptyDataError       — empty or missing input data
└── ValidationError      — data fails structural or content validation
```

Domain-specific subclasses exist in `data_cleaning/exceptions.py` (`DataCleaningError`, etc.) using multiple inheritance so callers can catch at either the domain or service level.

#### Data Cleaning (`services/data_cleaning/`)

| Class | Purpose |
|---|---|
| `BaseDataCleaner` | Common utilities: lipid column detection, numeric conversion, internal standard extraction (regex-based), lipid name validation. |
| `LipidSearchCleaner` | Grade-based filtering (A/B/C/D per class), AUC deduplication (highest abundance), column standardization (`MeanArea[*]` → `intensity[*]`). |
| `MSDIALCleaner` | Score threshold filtering, optional MS/MS requirement, deduplication by highest score, sample column auto-detection. |
| `GenericCleaner` | Basic cleanup: invalid lipid removal, zero column check, standard extraction. Used for Generic and Metabolomics Workbench formats. |

Configuration dataclasses: `GradeFilterConfig` (per-class grade selections), `QualityFilterConfig` (score threshold, MS/MS flag). Results returned as `CleaningResult` dataclass.

Cleaning is dispatched via a **registry pattern** in `data_cleaning/__init__.py`: `DataCleaningService.clean_data()` looks up the format in `_CLEANER_REGISTRY` (a `Dict[DataFormat, type]` mapping) and delegates to the appropriate cleaner. Adding a new format requires only a registry entry and a new cleaner class.

#### Core Services

| Service | File | Purpose |
|---|---|---|
| `FormatDetectionService` | `format_detection.py` | Auto-detects `DataFormat` enum from column signatures. Checks most-specific format first (LipidSearch → MS-DIAL → Metabolomics Workbench → Generic). |
| `DataStandardizationService` | `data_standardization.py` | Converts format-specific column names to standard `intensity[*]` naming. For LipidSearch 5.2 dual-polarity exports, merges each sample's positive/negative files via the alignment map. |
| `lipidsearch_alignment` | `lipidsearch_alignment.py` | Parses the LipidSearch Alignment Setting file, pairs each biological sample's positive/negative runs by raw-filename base, and merges their per-file `OriginalArea[s{cond}-{file}]` columns into flat `intensity[s1..sN]`. |
| `NormalizationService` | `normalization.py` | Four methods: internal standard (divide by standard intensity), protein (divide by protein concentration), internal standards + protein (standard then protein), total intensity (divide by per-sample total, scale by median). |
| `QualityCheckService` | `quality_check.py` | Box plot data prep, BQC CoV calculation, RT availability check, pairwise Pearson correlation, PCA (via scikit-learn). |
| `StatisticalTestingService` | `statistical_testing.py` | Parametric (Welch's t-test, Alexander-Govern ANOVA) and non-parametric (Mann-Whitney, Kruskal-Wallis) tests with two-level correction (between-class and post-hoc). Auto mode selects parametric with log10 transform, FDR for 2+ classes, Tukey for 3+ conditions. Uses dataset-wide zero replacement for consistent detection-floor handling. |
| `ZeroFilteringService` | `zero_filtering.py` | Removes lipids exceeding configurable zero-value thresholds (default: 75% for non-BQC, 50% for BQC samples). |
| `StandardsService` | `standards.py` | Standards extraction, validation, and class-to-standard mapping. |
| `ReportGeneratorService` | `report_generator.py` | Generates PDF reports from collected plots and analysis results. |
| `LSIReportService` | `lsi_report.py` | Generates Lipidomics Standards Initiative (LSI) compliance checklists with auto-filled fields from the analysis pipeline. Supports PDF and CSV export. |

#### AI Chat Services (`services/ai_chat/`)

All services are stateless static methods. The Anthropic API key is read from `ANTHROPIC_API_KEY` (env var or `st.secrets`) at call time; the service layer never imports Streamlit.

| Service / Module | File | Purpose |
|---|---|---|
| `AIClientService` | `ai_client.py` | Anthropic SDK wrapper. `send_message()` for plain calls, `execute_tool_loop()` for tool-use (caps round-trips at 4), the streaming counterparts `stream_tool_loop()` / `stream_explanation()` that every tab renders through `st.write_stream`, and `extract_code_block()` to pull ` ```python ` blocks out of responses. System prompts are marked for prompt caching. |
| `SandboxService` + `validate_code()` | `sandbox.py` | Runs AI-generated Python. **The security boundary is the child process**, not the parser: code executes via `sandbox_runner.py` in a separate process with no inherited environment (so `ANTHROPIC_API_KEY` is unreachable), no filesystem writes, no network, no subprocesses, and a hard kill on timeout. The AST validator (import whitelist — `math`, `statistics`, `re`, plus the pre-loaded `pd`/`np`/`px`/`go`; block-list of dunders and dangerous builtins) and the curated `_SAFE_BUILTINS` map run in front as defense-in-depth — never sufficient alone, since `pd`/`np` expose real modules by ordinary attribute access. Returns `SandboxResult(success, figure, result_df, stdout, error_message, execution_time)`. |
| `sandbox_runner` | `sandbox_runner.py` | The child-process entry point (`python -m app.services.ai_chat.sandbox_runner <fd>`): reads a pickled request from stdin, hardens itself best-effort (`RLIMIT_FSIZE` 0 → no file writes, `RLIMIT_CPU`, `RLIMIT_NPROC` 0 → no subprocesses, `socket` neutered in place → no network), executes the code, and writes the pickled response to a dedicated pipe. Covered end to end by `tests/unit/test_sandbox_isolation.py`. |
| `SchemaExtractorService` | `schema_extractor.py` | Builds a compact textual schema (column names, dtypes, example values) for inclusion in prompts — keeps raw data out of the API call. |
| `DataTransformerService` | `data_transformer.py` | Drives the CSV → Generic format transformation: builds prompt context, calls AI, runs the returned code in the sandbox, validates that the output is Generic-shaped. Returns `TransformationResult(success, result_df, explanation, code, sandbox_result, validation_errors)`. |
| `VizGeneratorService` | `viz_generator.py` | Generates Plotly figures from natural-language requests using `ExperimentConfig` context. Same loop: prompt → AI → sandbox → result. |
| `codebase_tools` | `codebase_tools.py` | Tool definitions for the Anthropic tools API plus an `execute_tool` dispatcher. Four read-only tools, each path-validated against an allowlist (`src/app`, `tests`, `docs`, plus `CODEBASE.md`) with output capped at 8 KB (40 KB for documentation): `search_codebase`, `read_file_section`, `list_files`, `read_documentation` (paper, supplementary methods, release notes, CODEBASE, etc.). |
| `knowledge_base` | `knowledge_base.py` | `get_app_knowledge()` — condensed reference (formats, modules, terminology) injected into system prompts so the assistant doesn't need to query the codebase for basic context. |
| `prompt_templates` | `prompt_templates.py` | Per-mode system prompts (`data_transform_system_prompt`, `data_questions_system_prompt`, `lipidcruncher_questions_system_prompt`, `unified_system_prompt`) and `build_transform_context` / `build_diagnostic_context` helpers. Includes `METHODOLOGY_CLARIFICATION_RULES` so the AI clarifies methodology before running aggregations or ratios. |
| `prompt_examples` | `prompt_examples.py` | Few-shot example prompts shown as clickable buttons in each AI Studio tab. |
| `interaction_logger` | `interaction_logger.py` | Emits one JSON record per chat turn and per feedback event to stdout (shipped to CloudWatch Logs by the container's `awslogs` driver). Pure — the UI/adapter supplies the fields, including an anonymous session id; emission failures are swallowed so logging can never break a turn. |

Exception hierarchy in `ai_chat/exceptions.py`:

```
AIChatError (base)
├── AIClientError       — API authentication / SDK / network failures
├── SandboxError        — code rejected by AST validator or runtime failure
└── TransformationError — output failed Generic-format validation
```
#### Isotope Correction (`services/isotope_correction/`)

The stable isotope tracing module. Runs the real **IsoCorrectoR** R/Bioconductor package via an `Rscript` subprocess (no `rpy2`) — the app is Python, IsoCorrectoR is R.

| File | Purpose |
|---|---|
| `service.py` | `IsotopeCorrectionService.run_correction()` — writes the three input DataFrames + a JSON config to a temp dir, invokes the R wrapper, parses the CSV outputs back into a `CorrectionResult` (corrected intensities, fractions, mean enrichment, residuals, raw data, run log). |
| `run_isocorrector.R` | **Bundled, trusted** R wrapper: reads the JSON config, calls `IsoCorrection()`, writes outputs. The executed R is *ours*; the user supplies data files, not code (so the AI-sandbox boundary does not apply). Passed file-path **arguments**, never shell-interpolated. |
| `validation.py` | Pure-Python structural validation of the three files *before* R runs (molecule-name consistency, numeric intensities, and a resolution-mode/ID-format check that catches UHR-vs-standard mismatches with an actionable message). |
| `r_runtime.py` | `find_rscript()` / `check_isocorrector_available()` / `require_isocorrector()` — graceful detection so the rest of the app runs where R is absent. |
| `exceptions.py` | `IsotopeError` → `RRuntimeError` (R missing/crashed) and `IsotopeInputError` (bad inputs). |

#### Plotting Services (`services/plotting/`)

Each module provides a static service class that returns Plotly or Matplotlib figures. All plotters follow the contract documented in `PlotterServiceProtocol` (`base.py`): static methods only, no Streamlit imports, validate inputs, return `go.Figure` or `plt.Figure`.

Shared utilities live in `_shared.py`: `p_value_to_marker()` (significance strings), `generate_class_color_mapping()` / `generate_condition_color_mapping()` (consistent color assignment from a 20-color palette), and `validate_dataframe()` (non-null, non-empty, required columns).

| Service | Output |
|---|---|
| `BoxPlotService` | Missing value bars + concentration box plots |
| `BQCPlotterService` | CoV scatter plot (high-CoV lipids highlighted) |
| `CorrelationPlotterService` | Pairwise Pearson correlation heatmap |
| `PCAPlotterService` | PCA biplot colored by condition |
| `RetentionTimePlotterService` | Single and multi-class RT scatter plots |
| `StandardsPlotterService` | Internal standards visualization |
| `BarChartPlotterService` | Mean ± std per condition per class (with optional stats) |
| `PieChartPlotterService` | Class composition per condition |
| `SaturationPlotterService` | Saturation degree distribution |
| `ChainLengthPlotterService` | Carbon chain length and double bond bubble charts |
| `FACHPlotterService` | Fatty acid composition heatmap |
| `PathwayVizService` | Interactive Plotly metabolic pathway network (data-driven, editable layout, 28 lipid classes) |
| `VolcanoPlotterService` | Fold-change vs. p-value scatter |
| `LipidomicHeatmapService` | Heatmaps in four modes — `regular`, `clustered` (k-means), `class_grouped` (one row per species, blocked by class, paged), `class_aggregated` (one row per class) — coloured by row Z-score or by log2 fold change against a control condition; under `class_grouped` + log2FC, species are sorted within each block by mean fold change |

### Workflows

Located in `src/app/workflows/`. These orchestrate multi-step pipelines by calling services in sequence. Pure logic — no Streamlit.

| Workflow | Key Method | What It Does |
|---|---|---|
| `DataIngestionWorkflow` | `run()` → `IngestionResult` | Format detection → cleaning → zero filtering → standards validation. Returns cleaned DataFrame + internal standards + messages. |
| `NormalizationWorkflow` | `run()` → `NormalizationWorkflowResult` | Validates inputs → preserves essential columns (CalcMass, BaseRt) → applies normalization → restores columns. Also provides `suggest_standard_mappings()` and `preview_normalization()`. |
| `QualityCheckWorkflow` | Individual step methods | Non-sequential: `run_box_plots()`, `run_bqc_assessment()`, `apply_bqc_filter()`, `run_correlation()`, `run_pca()`, `remove_samples()`. Each step can be run independently. |
| `AnalysisWorkflow` | Individual analysis methods | Non-sequential: `run_bar_chart()`, `run_pie_charts()`, `run_saturation()`, `run_chain_length()`, `run_fach()`, `run_pathway()`, `run_volcano()`, `run_heatmap()`. |
| `AIChatWorkflow` | One method per chat mode | Orchestrates the AI Studio chat loop. `diagnose_format()` decides if a CSV needs transformation; `validate_module1_output()` gates the viz tab. Per-mode session starters (`start_transform_session`, `start_viz_session`, `start_data_questions_session`, `start_lc_questions_session`, `start_unified_session`) and message handlers (`handle_transform_message`, `handle_viz_message`, `handle_data_questions_message`, `handle_lc_questions_message`, `handle_unified_message`). Each handler appends user/assistant messages to an immutable `ChatSession`, calls `AIClientService` (with tools where appropriate), runs returned code in the sandbox, and returns a typed result dataclass (`FormatDiagnosticResult`, `Module1ValidationResult`, `TransformStepResult`, `VizStepResult`, `DataQuestionsStepResult`, `LCQuestionsStepResult`, `UnifiedStepResult`). |
| `PipelineAPI` | Instance methods, injected as `lc` | Not a workflow but a workflow-layer facade: gives AI-generated sandbox code the pipeline's own analytics instead of re-invented heuristics — `conditions()`, `samples_for()`, `compare_species()`, `compare_classes()`, `class_summary()`, and one method per Module 3 analysis (`bar_chart()`, `pie_charts()`, `saturation()`, `chain_length()`, `fatty_acid_composition()`, `pathway()`, `volcano()`, `heatmap()`), each returning an `LCResult`. |
| `IsotopeTracingWorkflow` | `run()` → `IsotopeTracingResult` | Validate the three input files → run `IsotopeCorrectionService` → typed result (success + `CorrectionResult`, or `validation_errors` / `error_message`). Separate from the lipidomics pipeline. |

### Adapter

`src/app/adapters/streamlit_adapter.py` — the `StreamlitAdapter` class provides:

- **`SessionState` dataclass** — defines all session state keys with defaults and documents ownership (which UI file manages which keys).
- **`initialize_session_state()`** — populates `st.session_state` from `SessionState` defaults on first run.
- **`reset_data_state()`** — clears all data keys (full reset on new upload).
- **`reset_module_state(*prefixes)`** — clears keys matching given prefixes (e.g., `'qc_'`, `'analysis_'`).
- **Widget value preservation** — `restore_widget_value()` / `save_widget_value()` persist widget selections across module navigation.
- **Cached wrappers** — every workflow/service call that should be cached is wrapped with `@st.cache_data` and custom hash functions for Pydantic models and DataFrames.
- **AI Studio state** — `chat_studio_df`, `chat_studio_df_original`, `chat_studio_experiment`, `chat_studio_pipeline_source`, `studio_applied_snapshot` (set when the user confirms the inputs; the gate the three data tabs check), plus per-mode session keys `chat_transform_session`, `chat_viz_session`, `chat_data_questions_session`, `chat_unified_session`, and result caches (`chat_transform_result`, `chat_viz_figures`, `chat_result_tables`).
- **AI streaming wrappers** — `stream_ai_transform()`, `stream_ai_viz()` and `stream_ai_data_questions()` return `(explanation_generator, finalize)`: the UI streams the explanation through `st.write_stream`, then calls `finalize()` to run the code and get the typed step result. `stream_ai_lc_questions()` returns `(session_with_user_message, text_chunk_generator)` — no code to execute, so the UI just accumulates the text. `get_cached_lc_answer()` / `set_cached_lc_answer()` cache the canonical answers to the example LipidCruncher questions (instant repeats, and a fallback when the API is down), and `log_ai_feedback()` records thumbs / comments.
- **Isotope tracing** — `run_isotope_correction()` wraps `IsotopeTracingWorkflow` for the isotope page.
- **Cross-page widget preservation** — `preserve_main_app_widgets()` / `preserve_ai_studio_widgets()` snapshot widget values into `_nav_preserved_*` keys before navigation; `restore_navigation_widgets()` re-hydrates them on return. Required because Streamlit garbage-collects widget keys when their owning page is not rendered.

### UI

Located in `src/app/ui/`. Thin presentation layer that calls adapter/workflow methods and renders results.

**Sidebar components** (`ui/sidebar/`):
| File | Responsibility |
|---|---|
| `file_upload.py` | CSV upload widget, sample data loader |
| `column_mapping.py` | Displays standardized column names, MS-DIAL sample override |
| `experiment_config.py` | Number of conditions, condition labels, sample counts |
| `sample_grouping.py` | Assign samples to conditions, BQC label selection |
| `confirm_inputs.py` | Final confirmation checkbox before processing |

**Main content** (`ui/main_content/`):
| File | Responsibility |
|---|---|
| `data_processing.py` | Format docs, filter config, ingestion pipeline, filtered data display |
| `internal_standards.py` | Review/edit auto-detected standards, upload custom standards |
| `normalization.py` | Normalization method selection, standard mapping, protein input |
| `quality_check.py` | All QC steps rendered with interactive controls |
| `lsi_report.py` | LSI compliance checklist generation and export (PDF/CSV) |
| `analysis/` | Subpackage with one module per analysis type (`_bar_chart.py`, `_volcano.py`, etc.) plus shared utilities. The pathway module is decomposed into `_pathway.py` (main rendering), `_pathway_state.py` (session state accessors), and `_pathway_editor.py` (layout customization panel). |

**Static content** (`ui/content/`): Documentation text for processing steps, normalization methods, sample data descriptions, standards help, and analysis methodology (statistical testing guide, saturation profile calculations).

**Utilities**: `landing_page.py` (home page), `format_requirements.py` (format-specific column requirements), `zero_filtering.py` (interactive threshold sliders), `standards_plots.py` (standards visualization), `download_utils.py` (SVG/CSV/PDF export helpers), `st_helpers.py` (shared UI helpers: export buttons, section headers, widget persistence).

**AI Studio** (`ui/ai_studio_page.py` + `ui/ai_chat/`):
| File | Responsibility |
|---|---|
| `ai_studio_page.py` | Top-level page: data-source sidebar (upload / import from main pipeline), tab routing, round-trip navigation back to the main app. Locks data-dependent tabs behind a "Confirm" step. Carries the two page-level help expanders (*How to use the AI Studio*, *How the AI works — and its limits*); each tab module carries its own *About this tab* expander. |
| `ai_chat/chat_panel.py` | Shared chat-rendering helpers used by every tab (message rendering, code expanders, example-prompt buttons, figure / table display). |
| `ai_chat/_lipidcruncher_questions_ui.py` | Tab 1 — Q&A about LipidCruncher itself. |
| `ai_chat/_data_questions_ui.py` | Tab 2 — natural-language exploration of the loaded DataFrame. |
| `ai_chat/_data_transform_ui.py` | Tab 3 — CSV → Generic format transformation, with download of the result. |
| `ai_chat/_viz_generation_ui.py` | Tab 4 — natural-language Plotly figure generation. Gated by `Module1ValidationResult`. |
| `ai_chat/_interaction_log.py` | `record_turn()` (one structured log record per turn) and `render_feedback()` (thumbs up/down + optional comment under each assistant message). |
| `ai_chat/_unified_chat_ui.py` | Single-chat variant (legacy / fallback before tabs were reintroduced). |
**Isotope tracing page** (`isotope_tracing_page.py`): a standalone page (route `Page.ISOTOPE_TRACING`, reached from the landing-page mode choice) for the stable isotope tracing workflow — three file uploaders + sample-data loader, a settings panel mapping 1:1 to `IsotopeCorrectionConfig`, in-app help expanders (*How to Run a Correction* with a rejection-message → fix table, *Input File Requirements*, *About Isotope Correction*), run button, result tables with per-table CSV download and a *How to Read These Tables* expander explaining each IsoCorrectoR output and what a healthy run looks like, run-log expander, and a graceful "R runtime unavailable" notice.

---

## Supported Data Formats

| Format | Enum Value | Detection Signature | Key Columns |
|---|---|---|---|
| LipidSearch | `LIPIDSEARCH` | Flat `MeanArea[*]` (5.0 / sample-grouped 5.2) **or** per-file `OriginalArea[s{cond}-{file}]` (5.2 dual-polarity, condition-grouped — requires the Alignment Setting file) + required metadata | `LipidMolec`, `ClassKey`, `BaseRt`, `CalcMass`, `TotalGrade`, `MeanArea[s1]`… / `OriginalArea[s1-1]`… |
| MS-DIAL | `MSDIAL` | `Metabolite name` + signature columns OR `Alignment ID` | `Metabolite name`, `Ontology`, `Total score`, sample columns auto-detected |
| Generic | `GENERIC` | Lipid names in first column + numeric data | `LipidMolec`, `ClassKey`, numeric sample columns |
| Metabolomics Workbench | `METABOLOMICS_WORKBENCH` | `START`/`END` text markers | Tab-delimited with metadata headers |

Format detection priority: LipidSearch (most specific) → MS-DIAL → Metabolomics Workbench → Generic (fallback).

### Lipid Name Nomenclature (LIPID MAPS)

All lipid names are standardized to [LIPID MAPS shorthand notation](https://www.lipidmaps.org/) (Liebisch et al. 2020):

| Feature | Old Format | LIPID MAPS Format |
|---------|-----------|-------------------|
| Class separator | Parentheses: `PC(34:1)` | Space: `PC 34:1` |
| Chain separator (general) | Underscore: `PC(16:0_18:1)` | Underscore: `PC 16:0_18:1` |
| Chain separator (sphingolipids) | Underscore: `Cer(d18:1_24:0)` | Slash: `Cer 18:1;O2/24:0` |
| Hydroxyl notation | `;2O` | `;O2` (element before count) |
| Lyso species | `LPC(20:0_0:0)` | `LPC 20:0` (phantom chain removed) |
| Chain ordering | Alphabetical | Ascending by carbon count, then double bonds |

Nomenclature functions live in `lipid_nomenclature.py`: `parse_lipid_name()`, `normalize_hydroxyl()`, `sort_chains_lipid_maps()`, `remove_phantom_chains()`, `format_lipid_name()`. All lipid class sets (`SPHINGOLIPID_CLASSES`, `SINGLE_CHAIN_CLASSES`, `LYSO_CLASSES`) are `frozenset` for immutability. The functions are re-exported from `constants.py` for backward compatibility.

---

## Session State Management

All session state is managed through the `SessionState` dataclass in `streamlit_adapter.py`. Key principles:

1. **Single ownership** — each key is owned by exactly one UI file (documented in `SessionState` docstring).
2. **Prefixed resets** — `reset_module_state('qc_')` clears all QC keys without touching other modules.
3. **Widget preservation** — when the user navigates between modules, widget values are saved to `_preserved_*` keys and restored on return. Cross-page navigation between the main app and AI Studio uses a separate `_nav_preserved_*` snapshot via `preserve_main_app_widgets()` / `preserve_ai_studio_widgets()` / `restore_navigation_widgets()`.
4. **Cascade resets** — navigating back to Module 1 resets both QC and Analysis state. Going home resets everything.
5. **Page routing** — the active page is controlled by a `Page` enum (`LANDING`, `APP`, `AI_STUDIO`, `ISOTOPE_TRACING`) stored in `st.session_state.page`; aliases `PAGE_LANDING`, `PAGE_APP`, `PAGE_AI_STUDIO`, `PAGE_ISOTOPE_TRACING` are exported from `constants.py` for backward compatibility.

**Important state keys and their flow:**

```
raw_df → standardized_df → cleaned_df → normalized_df → qc_continuation_df → analysis input
```

Each DataFrame flows forward through the pipeline. If the user changes something upstream (e.g., re-uploads data), downstream state is automatically cleared.

The isotope tracing page owns the `iso_`-prefixed keys (`iso_measurement_df`, `iso_molecule_df`, `iso_element_df`, `iso_config`, `iso_result`), cleared together via `reset_module_state('iso_')` when leaving the page.

---

## Testing

### Configuration

```ini
# pytest.ini
[pytest]
pythonpath = src
testpaths = tests
```

This allows imports like `from app.services.normalization import NormalizationService` in tests.

### Test Organization

| Directory | Count | Scope |
|---|---|---|
| `tests/unit/` | 64 modules | Individual services, models, plotters, adapter, sandbox isolation |
| `tests/integration/` | 6 modules | End-to-end pipeline per module, AI tool use, isotope golden run |
| `tests/ui/` | 10 modules | Streamlit component rendering (AppTest framework) |

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_normalization.py -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src/app --cov-report=term-missing
```

### Test Patterns

- **Service tests** — pure function testing, no mocking needed. Create test DataFrames, call static methods, assert results.
- **Workflow tests** — integration-style, testing multi-step pipelines with realistic data.
- **UI tests** — use mocking for `st.session_state` and Streamlit widgets.
- **Fixtures** — shared in `tests/conftest.py`: sample DataFrames for all formats, `ExperimentConfig` instances, filter configs.

### Sample Datasets

`sample_datasets/` contains one test CSV per format, used by both the test suite and the app's "Load Sample Data" feature. The `isotope_tracing/` subfolder holds the IsoCorrectoR golden dataset (the three input files plus `golden_output/`), used by the sample-data loader and the R-gated parity test.

---

## Deployment

### Docker

```bash
# Build
docker build -t lipidcruncher .

# Run
docker run -p 8501:8501 lipidcruncher
```

The Dockerfile uses `python:3.11-slim`, installs `poppler-utils` (for PDF generation) plus R + Bioconductor + IsoCorrectoR + `jsonlite` (for the stable isotope tracing module), and runs `streamlit run src/main_app.py`.

**One image, many deployments.** There is a single Dockerfile; what an image contains is decided by the branch it is built from, and deployments are separated by ECR tag and ECS service rather than by a second Dockerfile. Build for Fargate with `--platform linux/amd64`. The R layer dominates a cold build but sits before the application `COPY`, so it stays cached across app changes. `ANTHROPIC_API_KEY` is injected at run time (ECS `secrets`), never baked into the image; without it every page works except the AI Studio. See `docs/isotope_deployment.md` for the separate-deployment guide.

### Local Development

```bash
source lipidcruncher_env/bin/activate
streamlit run src/main_app.py
```

The app runs on port 8501 by default. Server settings live in `.streamlit/config.toml`; `maxUploadSize = 1024` sets the 1 GB upload cap, and the sidebar reads it back with `st.get_option("server.maxUploadSize")` rather than hard-coding a number.

---

## Troubleshooting

### Common Issues

**"No normalized data available"** — The user tried to access Module 2 without completing normalization in Module 1. Ensure `normalized_df` is set in session state before navigating.

**Format detection fails** — Check that the uploaded CSV has the expected column signatures. Use `FormatDetectionService.detect_format()` directly to debug. The detection order matters: it checks LipidSearch first, then MS-DIAL, then Metabolomics Workbench, then Generic.

**Internal standards not detected** — The regex patterns in `constants.py` (`INTERNAL_STANDARD_LIPID_PATTERNS`) define what's recognized. If a lab uses non-standard naming, the patterns may need to be extended.

**Caching issues** — If the app shows stale data after code changes, clear the Streamlit cache (`st.cache_data.clear()`) or restart the app. The adapter's `@st.cache_data` decorators cache based on input hashes.

**Session state key errors** — All valid keys are defined in the `SessionState` dataclass. If a key is missing, ensure `StreamlitAdapter.initialize_session_state()` runs before any access (it's called at module level in `main_app.py`).

**Statistical tests return NaN** — This happens when a condition has too few samples or all values are identical. The `StatisticalTestingService` handles this gracefully but the UI should indicate it.

**BQC steps unavailable** — BQC assessment requires BQC samples to be labeled during experiment configuration. If no BQC label is set, BQC-related QC steps are skipped.

**PDF report generation fails** — Requires `poppler-utils` system package (installed in Docker). For local dev on macOS: `brew install poppler`.

**AI Studio: "ANTHROPIC_API_KEY environment variable is not set"** — Set the key in your shell (`export ANTHROPIC_API_KEY=...`) or add it to `.streamlit/secrets.toml` as `ANTHROPIC_API_KEY = "..."`. `main_app.py` copies the secret into `os.environ` at startup so the service layer can read it.

**AI Studio: "Import of '...' is not allowed"** — The AI generated code that uses an import outside the sandbox whitelist. Either ask the AI to rewrite without that library, or extend `_WHITELISTED_IMPORTS` in `services/ai_chat/sandbox.py` if the import is genuinely safe.

**AI Studio: empty figure / no result** — `AIChatWorkflow` detects empty Plotly figures and surfaces the generated code below the empty plot for troubleshooting; usually a filter, group-by, or column-name mismatch eliminated all rows before plotting.

### Debugging Tips

- **Check session state**: In a Streamlit callback or after an error, inspect `st.session_state` to see what data is available at each stage.
- **Test services in isolation**: Since all services are pure static methods, you can test any business logic in a Python REPL without running Streamlit.
- **Check the adapter**: If a cached result seems wrong, the issue is usually in the hash function or the cache key. Look at the `@st.cache_data` wrappers in `streamlit_adapter.py`.

---

## Extending the Application

### Adding a New Analysis Type

1. **Create a plotting service** in `services/plotting/new_plot.py` with a static method returning a Plotly/Matplotlib figure.
2. **Add a method** to `AnalysisWorkflow` in `workflows/analysis.py` that calls the new service.
3. **Add a cached wrapper** in `StreamlitAdapter` if the computation is expensive.
4. **Create a UI module** at `ui/main_content/analysis/_new_plot.py` following the pattern of existing analysis modules.
5. **Register it** in `ui/main_content/analysis/_entry.py` to add it to the analysis type selector.
6. **Add tests** in `tests/unit/` for the plotter service and update integration tests.

### Adding a New Data Format

1. **Add an enum value** to `DataFormat` in `services/format_detection.py`.
2. **Add detection logic** in `FormatDetectionService.detect_format()` — place it in the correct priority order.
3. **Create a cleaner** in `services/data_cleaning/` extending `BaseDataCleaner`.
4. **Register the cleaner** in `_CLEANER_REGISTRY` in `data_cleaning/__init__.py`.
5. **Add standardization logic** in `services/data_standardization.py`.
6. **Update the format display mapping** in `constants.py` (`get_format_display_to_enum()`).
7. **Add a sample dataset** in `sample_datasets/`.
8. **Add format requirements** in `ui/format_requirements.py`.
9. **Add tests** covering detection, cleaning, and end-to-end ingestion.

### Adding a New AI Studio Tab

1. **Add a `ChatMode` enum value** in `models/chat.py` if the new tab has its own conversation type.
2. **Add a system prompt** in `services/ai_chat/prompt_templates.py` that defines scope and tone.
3. **Add session start + handler methods** to `AIChatWorkflow` in `workflows/ai_chat.py`. Use `AIClientService.execute_tool_loop` if the AI needs codebase / documentation tools, otherwise `send_message`.
4. **Add a UI module** at `ui/ai_chat/_<tab>_ui.py` calling the workflow methods through the adapter.
5. **Register the tab** in `ui/ai_studio_page.py` and re-export the display function from `ui/ai_chat/__init__.py`.
6. **Add session-state keys** with the `chat_*` prefix in `SessionState` in `streamlit_adapter.py`.
7. **Add tests** under `tests/unit/test_ai_chat_*` for the workflow and `tests/integration/test_ai_tool_use.py` for any new tools.

### Extending the Sandbox

Sandbox changes are security-critical — review carefully. Never route AI code around the child process: the AST validator and builtin allowlist are defense-in-depth, not the boundary.

1. **New safe builtin** — add to `_SAFE_BUILTINS` in `services/ai_chat/sandbox.py`. Avoid anything that touches I/O, threading, or attribute introspection.
2. **New whitelisted import** — add to `_WHITELISTED_IMPORTS`. Confirm the module has no `eval`/`exec`/`__import__`-style escape hatches.
3. **Changing the child process** — process hardening lives in `_harden()` in `sandbox_runner.py`; each control is best-effort and independent, so one unavailable on the host must not prevent the others from applying.
4. **Update tests** — `tests/unit/test_ai_sandbox.py` covers AST validation, blocked dunders / builtins, timeout, and output limits; `tests/unit/test_sandbox_isolation.py` drives each escape (env, filesystem, network, subprocess) end to end through the real child process. Both must stay passing.

### Adding a New QC Step

1. **Add a method** to `QualityCheckService` in `services/quality_check.py`.
2. **Add an orchestration method** to `QualityCheckWorkflow` in `workflows/quality_check.py`.
3. **Add a cached wrapper** in `StreamlitAdapter` if needed.
4. **Add UI rendering** in `ui/main_content/quality_check.py`.
5. **Add session state keys** to `SessionState` with the `qc_` prefix.
6. **Add tests**.

### Key Conventions

- **Services are stateless** — all methods are `@staticmethod`. No instance variables.
- **Return dataclasses, not tuples** — workflow results are typed dataclasses for clarity.
- **Typed exceptions** — raise `ConfigurationError`, `EmptyDataError`, or `ValidationError` (from `services/exceptions.py`) instead of bare `ValueError`.
- **Immutable collections** — constants use `Tuple` and `frozenset` instead of mutable `List`/`set` to prevent accidental mutation in cached contexts.
- **Enum-based routing** — `Page` and `Module` enums in `constants.py` replace raw strings for type-safe page/module navigation. Backward-compatible aliases (`PAGE_LANDING`, `MODULE_DATA_PROCESSING`, etc.) are provided.
- **Prefix session state keys** — by module: `qc_*`, `analysis_*`, `_preserved_*`, etc.
- **Plotting services return figures** — the UI layer handles rendering with `st.plotly_chart()` or `st.pyplot()`. All plotters follow `PlotterServiceProtocol` (`plotting/base.py`).
- **Constants go in `constants.py`** — thresholds, regex patterns, format mappings. Nomenclature functions live in `lipid_nomenclature.py`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.50.0 | UI framework |
| pandas | 1.5.3 | Data manipulation |
| numpy | 1.24.3 | Numerical computing |
| scipy | 1.13.1 | Statistical tests (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis, Shapiro-Wilk) |
| scikit-learn | 1.3.0 | PCA, StandardScaler |
| statsmodels | 0.14.0 | Multiple testing correction (FDR, Bonferroni) |
| plotly | 5.18.0 | Interactive visualizations |
| matplotlib | 3.7.1 | Static plots (heatmaps, FACH) |
| seaborn | 0.12.2 | Statistical plot styling |
| bokeh | 2.4.3 | Additional visualization support |
| kaleido | 0.2.1 | Plotly static image export |
| reportlab | 3.6.12 | PDF generation |
| pdf2image | 1.17.0 | PDF to image conversion |
| svglib | 1.5.1 | SVG rendering for PDF reports |
| pillow | 9.5.0 | Image processing |
| openpyxl | 3.1.2 | Excel export |
| selenium | 4.10.0 | Browser automation (export fallback) |
| anthropic | ≥0.52.0 | Claude API SDK for the AI Studio (chat completion + tool use). Reads `ANTHROPIC_API_KEY` from environment or `st.secrets`. |
