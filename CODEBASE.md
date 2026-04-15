# LipidCruncher — Codebase Documentation

A professional lipidomics data analysis application built with Streamlit, developed by the Farese & Walther Lab at Memorial Sloan Kettering Cancer Center. LipidCruncher enables researchers to process, analyze, and visualize lipidomic data from multiple sources without requiring bioinformatics expertise.

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
│       │   └── statistics.py                #   StatisticalTestConfig
│       ├── services/                        # Pure business logic
│       │   ├── __init__.py                  #   Public API: re-exports services + exceptions
│       │   ├── exceptions.py                #   Typed exception hierarchy (ServiceError, etc.)
│       │   ├── format_detection.py          #   DataFormat enum + detection
│       │   ├── data_standardization.py      #   Column name standardization
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
│       │   └── analysis.py                  #   AnalysisWorkflow
│       ├── adapters/
│       │   └── streamlit_adapter.py         #   StreamlitAdapter
│       └── ui/                              # Streamlit UI components
│           ├── landing_page.py
│           ├── format_requirements.py
│           ├── zero_filtering.py
│           ├── standards_plots.py
│           ├── download_utils.py
│           ├── st_helpers.py                #   Shared UI helpers (export buttons, section headers)
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
│   ├── unit/                                # 33 unit test modules
│   ├── integration/                         # 4 integration test modules
│   └── ui/                                  # 5 UI test modules
├── sample_datasets/                         # Test data (one per format)
│   ├── lipidsearch5_test_dataset.csv
│   ├── msdial_test_dataset.csv
│   ├── generic_test_dataset.csv
│   └── mw_test_dataset.csv
├── images/                                  # UI assets (logo, module PDFs)
├── requirements.txt
├── pytest.ini
├── Dockerfile
└── README.md
```

---

## Application Flow

The app has two pages: a **landing page** and the **main app page**. The main app page has two modules that the user progresses through sequentially.

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

---

## Layer-by-Layer Reference

### Models

Located in `src/app/models/`. All are frozen Pydantic `BaseModel` subclasses (immutable, hashable).

| Model | File | Purpose |
|---|---|---|
| `ExperimentConfig` | `experiment.py` | Defines experimental setup: number of conditions, condition labels, sample counts. Auto-derives `n_conditions` from `conditions_list` if omitted. Validates condition label uniqueness. Computed properties generate sample labels (s1, s2, ...) and group them by condition. Provides `without_samples()` for creating configs after sample removal. |
| `NormalizationConfig` | `normalization.py` | Normalization method (`none`, `internal_standard`, `protein`, `both`, `total_intensity`), selected lipid classes, standard-to-class mappings, concentrations. |
| `StatisticalTestConfig` | `statistics.py` | Test type (`parametric`, `non_parametric`, `auto`), multiple-testing correction (`fdr_bh`, `bonferroni`), post-hoc method (`tukey`, `bonferroni`), `auto_transform` flag for log10 transformation. |

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
| `DataStandardizationService` | `data_standardization.py` | Converts format-specific column names to standard `intensity[*]` naming. |
| `NormalizationService` | `normalization.py` | Four methods: internal standard (divide by standard intensity), protein (divide by protein concentration), internal standards + protein (standard then protein), total intensity (divide by per-sample total, scale by median). |
| `QualityCheckService` | `quality_check.py` | Box plot data prep, BQC CoV calculation, RT availability check, pairwise Pearson correlation, PCA (via scikit-learn). |
| `StatisticalTestingService` | `statistical_testing.py` | Parametric (Welch's t-test, Alexander-Govern ANOVA) and non-parametric (Mann-Whitney, Kruskal-Wallis) tests with two-level correction (between-class and post-hoc). Auto mode selects parametric with log10 transform, FDR for 2+ classes, Tukey for 3+ conditions. Uses dataset-wide zero replacement for consistent detection-floor handling. |
| `ZeroFilteringService` | `zero_filtering.py` | Removes lipids exceeding configurable zero-value thresholds (default: 75% for non-BQC, 50% for BQC samples). |
| `StandardsService` | `standards.py` | Standards extraction, validation, and class-to-standard mapping. |
| `ReportGeneratorService` | `report_generator.py` | Generates PDF reports from collected plots and analysis results. |

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
| `LipidomicHeatmapService` | Clustered heatmap with k-means |

### Workflows

Located in `src/app/workflows/`. These orchestrate multi-step pipelines by calling services in sequence. Pure logic — no Streamlit.

| Workflow | Key Method | What It Does |
|---|---|---|
| `DataIngestionWorkflow` | `run()` → `IngestionResult` | Format detection → cleaning → zero filtering → standards validation. Returns cleaned DataFrame + internal standards + messages. |
| `NormalizationWorkflow` | `run()` → `NormalizationWorkflowResult` | Validates inputs → preserves essential columns (CalcMass, BaseRt) → applies normalization → restores columns. Also provides `suggest_standard_mappings()` and `preview_normalization()`. |
| `QualityCheckWorkflow` | Individual step methods | Non-sequential: `run_box_plots()`, `run_bqc_assessment()`, `apply_bqc_filter()`, `run_correlation()`, `run_pca()`, `remove_samples()`. Each step can be run independently. |
| `AnalysisWorkflow` | Individual analysis methods | Non-sequential: `run_bar_chart()`, `run_pie_charts()`, `run_saturation()`, `run_chain_length()`, `run_fach()`, `run_pathway()`, `run_volcano()`, `run_heatmap()`. |

### Adapter

`src/app/adapters/streamlit_adapter.py` — the `StreamlitAdapter` class provides:

- **`SessionState` dataclass** — defines all session state keys with defaults and documents ownership (which UI file manages which keys).
- **`initialize_session_state()`** — populates `st.session_state` from `SessionState` defaults on first run.
- **`reset_data_state()`** — clears all data keys (full reset on new upload).
- **`reset_module_state(*prefixes)`** — clears keys matching given prefixes (e.g., `'qc_'`, `'analysis_'`).
- **Widget value preservation** — `restore_widget_value()` / `save_widget_value()` persist widget selections across module navigation.
- **Cached wrappers** — every workflow/service call that should be cached is wrapped with `@st.cache_data` and custom hash functions for Pydantic models and DataFrames.

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
| `analysis/` | Subpackage with one module per analysis type (`_bar_chart.py`, `_volcano.py`, etc.) plus shared utilities. The pathway module is decomposed into `_pathway.py` (main rendering), `_pathway_state.py` (session state accessors), and `_pathway_editor.py` (layout customization panel). |

**Static content** (`ui/content/`): Documentation text for processing steps, normalization methods, sample data descriptions, standards help, and analysis methodology (statistical testing guide, saturation profile calculations).

**Utilities**: `landing_page.py` (home page), `format_requirements.py` (format-specific column requirements), `zero_filtering.py` (interactive threshold sliders), `standards_plots.py` (standards visualization), `download_utils.py` (SVG/CSV/PDF export helpers), `st_helpers.py` (shared UI helpers: export buttons, section headers, widget persistence).

---

## Supported Data Formats

| Format | Enum Value | Detection Signature | Key Columns |
|---|---|---|---|
| LipidSearch 5.0 | `LIPIDSEARCH` | `MeanArea[*]` columns + required metadata | `LipidMolec`, `ClassKey`, `BaseRt`, `CalcMass`, `Grade`, `MeanArea[s1]`... |
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
3. **Widget preservation** — when the user navigates between modules, widget values are saved to `_preserved_*` keys and restored on return.
4. **Cascade resets** — navigating back to Module 1 resets both QC and Analysis state. Going home resets everything.

**Important state keys and their flow:**

```
raw_df → standardized_df → cleaned_df → normalized_df → qc_continuation_df → analysis input
```

Each DataFrame flows forward through the pipeline. If the user changes something upstream (e.g., re-uploads data), downstream state is automatically cleared.

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
| `tests/unit/` | 33 modules | Individual services, models, plotters, adapter |
| `tests/integration/` | 4 modules | End-to-end pipeline per module |
| `tests/ui/` | 5 modules | Streamlit component rendering (AppTest framework) |

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

`sample_datasets/` contains one test CSV per format, used by both the test suite and the app's "Load Sample Data" feature.

---

## Deployment

### Docker

```bash
# Build
docker build -t lipidcruncher .

# Run
docker run -p 8501:8501 lipidcruncher
```

The Dockerfile uses `python:3.11-slim`, installs `poppler-utils` (for PDF generation), and runs `streamlit run src/main_app.py`.

### Local Development

```bash
source lipidcruncher_env/bin/activate
streamlit run src/main_app.py
```

The app runs on port 8501 by default.

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
