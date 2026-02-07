# LipidCruncher Project Knowledge

**Last Updated:** February 7, 2026
**Current Branch:** `refactor/v3.0`

---

## Current Progress

### ✅ Phase 1: Setup (COMPLETE)

| Task | Status | Commit |
|------|--------|--------|
| Create `refactor/v3.0` branch from `main` | ✅ Done | `5ab5142` |
| Rename `main_app.py` → `old_main_app.py` | ✅ Done | `5ab5142` |
| Create folder structure `src/app/` | ✅ Done | `5ab5142` |
| Create new minimal `main_app.py` | ✅ Done | `5ab5142` |
| Create `tests/fixtures/` structure | ✅ Done | `5ab5142` |

**Decisions:**
- Skip upfront integration tests — `old_main_app.py` is tightly coupled to Streamlit, making it hard to test. Write unit tests as each service is extracted instead.
- Start `main_app.py` fresh — don't import from `old_main_app.py`. Build features incrementally using the new architecture. Use `old_main_app.py` as reference only.

### ✅ Phase 2: Extract Models (COMPLETE)

| Model | Status | Tests | Commit |
|-------|--------|-------|--------|
| ExperimentConfig | ✅ Done | 105 tests | `f7b3641` |
| NormalizationConfig | ✅ Done | 108 tests | `58349ed` |
| StatisticalTestConfig | ✅ Done | 137 tests | `ac35c31` |

**Created Files:**
- `src/app/models/experiment.py` — ExperimentConfig with computed sample lists
- `src/app/models/normalization.py` — NormalizationConfig with method validation
- `src/app/models/statistics.py` — StatisticalTestConfig with mode/correction validation
- `tests/unit/test_experiment_config.py` — 105 tests (includes type handling, input validation, boundary conditions)
- `tests/unit/test_normalization_config.py` — 108 tests (includes type handling, input validation, boundary conditions)
- `tests/unit/test_statistics_config.py` — 137 tests (includes type handling, input validation, boundary conditions)
- `pytest.ini` — Test configuration with pythonpath

### ✅ Phase 3: Extract Services (COMPLETE)

| Service | Status | Tests | Commit |
|---------|--------|-------|--------|
| FormatDetectionService | ✅ Done | 133 tests | `af1a42f` |
| DataCleaningService | ✅ Done | 158 tests | `ef09322` |
| ZeroFilteringService | ✅ Done | 102 tests | `7c66478` |
| NormalizationService | ✅ Done | 115 tests | `4ce8397` |
| StandardsService | ✅ Done | 167 tests | `f8a6cbe` |

**Created Files:**
- `src/app/services/format_detection.py` — Auto-detect data format from column signatures
- `tests/unit/test_format_detection.py` — 133 tests
- `tests/unit/test_standards.py` — 167 tests (includes MW format edge cases)
- `src/app/services/data_cleaning/` — Modular package for data cleaning:
  - `__init__.py` — Main DataCleaningService with format dispatching
  - `base.py` — Common methods (validation, conversion, row filtering)
  - `configs.py` — GradeFilterConfig, QualityFilterConfig, CleaningResult
  - `lipidsearch.py` — LipidSearch 5.0 cleaner (grade filtering, AUC selection)
  - `msdial.py` — MS-DIAL cleaner (quality filtering, deduplication)
  - `generic.py` — Generic/Metabolomics Workbench cleaner
- `tests/unit/test_data_cleaning.py` — 158 tests (includes ClassKey edge cases)
- `src/app/services/zero_filtering.py` — Zero value filtering with configurable thresholds
  - `ZeroFilterConfig` — Configurable thresholds (detection, BQC, non-BQC)
  - `ZeroFilteringResult` — Result with filtered_df and removed species
  - `ZeroFilteringService` — Static methods for filtering and statistics
- `tests/unit/test_zero_filtering.py` — 102 tests
- `src/app/services/normalization.py` — Data normalization service
  - `NormalizationResult` — Result with normalized_df, removed_standards, method_applied
  - `NormalizationService` — Static methods for IS, protein, and combined normalization
- `tests/unit/test_normalization.py` — 115 tests
- `src/app/services/standards.py` — Internal standards management service
  - `StandardsExtractionResult` — Result with data_df, standards_df, patterns matched
  - `StandardsValidationResult` — Validation result with errors/warnings
  - `StandardsProcessingResult` — Result with processed standards and source mode
  - `StandardsService` — Static methods for detection, extraction, validation, and processing
- `tests/unit/test_standards.py` — 153 tests

### 🔄 Phase 4: Extract Workflows & UI (IN PROGRESS)

| Component | Status | Tests | Commit |
|-----------|--------|-------|--------|
| StreamlitAdapter | ✅ Done | 75 tests | `b077cb1` |
| DataIngestionWorkflow | ✅ Done | 125 tests | `b077cb1` |
| NormalizationWorkflow | ✅ Done | 98 tests | `341ba71` |
| **Module 1 UI** | ✅ Done | - | `4a428fd` |

**Created Files:**
- `src/app/adapters/streamlit_adapter.py` — Session state management and caching wrappers
  - `SessionState` — Type-safe container for all session state variables
  - `StreamlitAdapter` — Static methods for session state, caching, and service wrappers
- `src/app/workflows/data_ingestion.py` — Data ingestion pipeline orchestration
  - `IngestionConfig` — Configuration for the ingestion workflow
  - `IngestionResult` — Complete result with cleaned_df, standards, and validation status
  - `DataIngestionWorkflow` — Orchestrates format detection → cleaning → zero filtering → standards
- `src/app/workflows/normalization.py` — Normalization pipeline orchestration
  - `NormalizationWorkflowConfig` — Configuration for normalization workflow
  - `NormalizationWorkflowResult` — Complete result with normalized_df, method_applied, statistics
  - `NormalizationWorkflow` — Orchestrates class selection → method config → normalization → column restoration
- `tests/unit/test_data_ingestion_workflow.py` — 125 tests (comprehensive fixtures for all formats)
- `tests/unit/test_streamlit_adapter.py` — 75 tests (SessionState, utility methods, mocked session state)
- `tests/unit/test_normalization_workflow.py` — 98 tests (all methods, edge cases, integration scenarios)

### ⬜ Phase 5: Polish (NOT STARTED)

### Next Steps — Page-by-Page UI Build

**Strategy:** Build the UI module by module, wiring up existing workflows before extracting new ones.

#### Module 1: Filter and Normalize (COMPLETE)
Workflows and UI implemented:
- ✅ `DataIngestionWorkflow` — upload → format detection → cleaning → zero filtering → standards
- ✅ `NormalizationWorkflow` — normalization pipeline
- ✅ Landing page with "Start Analysis" button
- ✅ Sidebar: file upload, sample grouping, experiment config
- ✅ Main area: data preview, filtering options, normalization settings
- ✅ Workflows wired up and tested
- ✅ Landing page extracted to `src/app/ui/landing_page.py`
- ✅ Format requirements extracted to `src/app/ui/format_requirements.py`
- ✅ Zero filtering UI extracted to `src/app/ui/zero_filtering.py` (with detection threshold input)

**Remaining Tasks (UI Polish):**
1. ✅ Check for large methods in `main_app.py` — break down methods >50 lines (`0d37dba`)
2. ✅ Fix Metabolomics Workbench format loading (read as text, not CSV) (`183187a`)
3. ✅ Fix sample column detection after standardization (use `intensity[...]` pattern) (`183187a`)
4. ✅ Match all UI instructions to old_main_app.py:
   - ✅ "Try Sample Data" section — verified identical to old app
   - ✅ Format requirements — verified identical to old app
   - ✅ User-facing text — verified consistent with old app
5. ✅ Make sidebar identical to old app (except Define Experiment section):
   - ✅ Add "Column Name Standardization" section - display mapping table after file upload
   - ✅ Add "Override Sample Detection" expander for MS-DIAL format
   - ✅ Add "Group Samples" dataframe display (shows sample-condition mapping)
   - ✅ Add "Are your samples properly grouped together?" radio with manual regrouping option
   - ✅ Add explicit "Specify Label of BQC Samples" section with Yes/No radio
   - ✅ Add "Confirm Inputs" section with summary and checkbox confirmation
6. ✅ Fix UI flow gating — content only shows after "Confirm Inputs" checked (`162178f`)

**Completed Tasks:**
1. ✅ **Remove manual buttons** - Got rid of "Process Data" and "Apply Normalization" buttons (`df50fdc`)
   - Flow now runs automatically after sidebar confirmation (like old app)
   - Data cleaning → zero filtering → normalization flows without clicks
2. ✅ **Automatic flow execution** - After confirm checkbox, all processing runs at once (`df50fdc`)
3. ✅ **Add "Manage Internal Standards" expander** - Allow upload/edit of standards
   - Radio toggle between "Automatic Detection" and "Upload Custom Standards"
   - For custom upload: choice of "Extract from dataset" vs "Complete external standards"
   - File uploader with format guidance
   - Clear custom standards option
4. ✅ **Add lipid class multiselect** - Select which classes to include in normalization (`df50fdc`)
5. ✅ **Add "About Normalization" section** - Documentation expander for normalization (`df50fdc`)
6. ✅ **Match old app expander structure exactly:**
   - ✅ "📖 About Data Standardization and Filtering" expander
   - ✅ "⚙️ Configure Grade/Quality Filtering" expander (format-specific)
   - ✅ "⚙️ Configure Zero Filtering" expander
   - ✅ "📋 Final Filtered Data" section (always visible)
   - ✅ "Manage Internal Standards" expander
   - ✅ "📖 About Normalization" expander
   - ✅ Normalization method selection (radio)
   - ✅ Class selection (multiselect)
   - ✅ Standard-to-class mapping (if IS method selected)
   - ✅ Protein concentrations input (if protein method selected)
   - ✅ Normalized data display
7. ✅ **Verify zero filtering is being applied correctly** — Added detection threshold input, extracted to `src/app/ui/zero_filtering.py` (`4e7efd7`)
8. ✅ **Match exact text/labels from old app** (`4a428fd`)
   - Removed "Data cleaned successfully" message and "Processing Details" expander
   - Added "File uploaded successfully!" in sidebar after upload
   - Removed zero filtering summary line outside expander
   - Moved "Final Filtered Data" outside expander (matches old app)
   - Removed large "Normalization" subheader and redundant dividers/titles
   - Removed Lipids/Samples/Standards metrics from normalization results
   - Centered main page with `st.columns([1, 3, 1])` matching landing page width
   - Format requirements expander at top of centered content
9. ✅ **Add internal standards consistency plots** (`f9e99f5`) - Bar charts showing standards across samples
   - Condition multiselect to filter which samples to display
   - Uses `InternalStandardsPlotter.create_consistency_plots()` from legacy modules
   - Extracted to `src/app/ui/standards_plots.py`
10. ✅ **Make normalization section UI identical to old app** (`1d78365`)
   - Class multiselect now matches old app (direct multiselect with session state, no expander/checkbox)
   - Double-column layout kept for standard-to-class mapping (new design improvement)
   - Protein input matches old app: Manual Input / Upload CSV File, 3-column grid, always visible
   - Standards mapping and concentrations merged into single expander with validation
   - Added MS-DIAL pre-normalized check to disable IS normalization options
   - Added normalization method session state initialization and invalid selection guard

**`main_app.py` now includes:**
- Landing page with module descriptions
- Format selection and requirements display
- Sample data loading (all 4 formats working)
- File upload with format detection (Metabolomics Workbench reads as text)
- Column name standardization display (with MS-DIAL sample override)
- Group samples dataframe display with manual regrouping option
- Explicit BQC sample specification (Yes/No radio)
- Confirm inputs section with sample-condition summary
- Data processing documentation expander (format-specific pipeline docs)
- Grade filtering configuration for LipidSearch (A/B/C/D per class)
- Quality filtering configuration for MS-DIAL (Score threshold + MS/MS)
- Data type selection for MS-DIAL (raw vs pre-normalized)
- Zero filtering configuration with interactive live preview
- Final filtered data preview before normalization
- **Manage Internal Standards expander** (auto-detect or upload custom standards)
- **Internal standards consistency plots** (bar charts with condition filtering)
- Normalization UI (class selection, method selection, IS mapping, protein concentrations)

**Bugs Fixed (`f65bb20`):**
1. ✅ **MS-DIAL quality filtering summary** — Filter results now display inside the quality filtering expander (matching old app)
2. ✅ **Final filtered data sorting** — Data sorted by ClassKey so all species of the same class are grouped together

**Bugs Fixed (`d71dd1a`):**
1. ✅ **Sample grouping not applied** — Reordered DataFrame from manual regrouping now correctly passed to data processing workflow
2. ✅ **MS-DIAL override sample detection** — Removing samples via override now updates both raw and normalized column counts

**Bugs Fixed (`5e1cd8a`):**
1. ✅ **Internal standards upload mode switching** — Switching between "Extract from dataset" and "Uploading complete standards data" now clears previously uploaded standards
2. ✅ **Internal standards example naming** — Updated example lipid names to match LipidCruncher convention: `PC(15:0_18:1)+D7:(s)` instead of `PC(15:0-18:1(d7))`

**Bugs Fixed (`4b728ac`):**
1. ✅ **Generic format ClassKey column detection** — When Generic format has an optional ClassKey column (2nd column between LipidMolec and intensity columns), it was incorrectly counted as an intensity column. Fix detects ClassKey by header name (case-insensitive) or by value pattern (short alphabetic strings like lipid class names) and preserves it as metadata.

**Performance Improvement (`745dcc8`):**
1. ✅ **Add st.cache_data caching to workflow calls** — Previously, every UI interaction (slider moved, checkbox clicked) caused full data reprocessing. Now cached results are returned instantly when inputs haven't changed.
   - Added `StreamlitAdapter.run_ingestion()` — Cached wrapper for `DataIngestionWorkflow.run()`
   - Added `StreamlitAdapter.run_normalization()` — Cached wrapper for `NormalizationWorkflow.run()`
   - Updated `main_app.py` to use cached adapter methods instead of direct workflow calls
   - Fixed multiselect default validation to handle data changes gracefully (prevents "default value must exist in options" error)

#### Module 2: Quality Check (NOT STARTED)
1. ⬜ Extract `QualityCheckWorkflow` — box plots, BQC analysis, outlier detection
2. ⬜ Build Module 2 UI

#### Module 3: Visualize and Analyze (NOT STARTED)
1. ⬜ Extract `AnalysisWorkflow` — statistical tests, volcano plots, heatmaps
2. ⬜ Build Module 3 UI

---

## Extraction Workflow

For each component (model, service, etc.):

1. **Identify what's needed** — read `old_main_app.py` to understand full requirements (reference only, don't import)
2. **Check v2.0 for clean implementations** — v2.0 has well-structured code worth reusing
3. **Use/adapt v2.0 code** where it fits the requirements
4. **Supplement from `old_main_app.py`** where v2.0 is missing features
5. **Write unit tests** for the extracted component
6. **Add to `main_app.py`** and test the feature
7. **Commit**

---

## Project Overview

**LipidCruncher** is a Streamlit-based web application for lipidomics data analysis:

```
Upload CSV → Format Detection → Data Cleaning → Zero Filtering → Normalization → Statistical Analysis → Visualization
```

### Supported Data Formats
| Format | Description |
|--------|-------------|
| LipidSearch 5.0 | Mass spec output with grade filtering (A/B/C/D) |
| MS-DIAL | Mass spec with quality scores and MS/MS matching |
| Generic | Simple CSV (LipidMolec + intensity columns) |
| Metabolomics Workbench | Public repository format with embedded metadata |

### Normalization Methods
1. None (raw data)
2. Internal Standards (lipid class-specific)
3. Protein Concentration (sample-level)
4. Both (internal standards first, then protein)

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production code - fully functional, monolithic |
| `refactor/v2.0` | **Valuable resource** - clean architecture AND reusable code (may be missing some main features) |
| `refactor/v3.0` | **Active development** - combining best of both branches |

### Key Principle
- **Check v2.0 first** for clean, well-structured implementations
- **Use `old_main_app.py`** to understand full requirements and fill gaps where v2.0 is missing features
- **Refactor incrementally** with unit tests for each extracted component

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        UI LAYER                             │
│    main_app.py → app/ui/*.py                               │
│    (Streamlit widgets, user input, display only)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     WORKFLOW LAYER                          │
│         app/workflows/*.py                                  │
│    (Orchestration - coordinates services, handles flow)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      ADAPTER LAYER                          │
│         app/adapters/streamlit_adapter.py                  │
│    (Bridge between UI and services, session state)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                           │
│         app/services/*.py                                  │
│    (Pure business logic - NO Streamlit dependencies)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                            │
│         app/models/*.py                                    │
│    (Pydantic models with validation)                       │
└─────────────────────────────────────────────────────────────┘
```

### Layer Rules

| Layer | Can Import | Cannot Import |
|-------|-----------|---------------|
| Model | Pydantic, pandas | Streamlit, Services |
| Service | Models, pandas, numpy | Streamlit |
| Adapter | Services, Models, Streamlit | UI |
| Workflow | Adapter, Services, Streamlit | - |
| UI | Workflows, Streamlit | Services directly |

---

## Directory Structure

```
LipidCruncher/
├── src/
│   ├── main_app.py                    # New refactored app (TARGET: <500 lines)
│   ├── old_main_app.py                # Original monolithic app (reference)
│   ├── app/                           # Refactored package
│   │   ├── __init__.py
│   │   ├── models/                    # Pydantic data models
│   │   │   └── __init__.py
│   │   ├── services/                  # Business logic (NO Streamlit)
│   │   │   └── __init__.py
│   │   ├── adapters/                  # Streamlit adapter
│   │   │   └── __init__.py
│   │   ├── workflows/                 # Multi-step orchestration
│   │   │   └── __init__.py
│   │   └── ui/                        # UI components
│   │       └── __init__.py
│   └── lipidomics/                    # Legacy visualization modules (keep for now)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                      # Generated edge case datasets
│       ├── edge_cases/
│       └── generators/
└── sample_datasets/
```

---

## Testing Strategy

### Unit Test Depth & Coverage Requirements

**Target: 100+ tests per service** — This project maintains comprehensive unit test coverage. Each service should have approximately 100-150 tests covering:

1. **Core Functionality Tests** — Test each public method's primary purpose
2. **Edge Cases** — Empty DataFrames, None values, missing columns, single rows
3. **Boundary Conditions** — Threshold values, large datasets, special characters
4. **Type Handling** — String/int/float coercion, object dtypes, mixed types
5. **Error Conditions** — Invalid inputs, missing required data, validation failures
6. **Integration Scenarios** — Realistic multi-step workflows

**⚠️ IMPORTANT: Maintain Test Depth for All New Code**

All future tests MUST maintain the same level of depth and coverage as existing tests. Reference these examples:
- `test_format_detection.py` — 133 tests
- `test_data_cleaning.py` — 158 tests (includes bug fix regression tests)
- `test_standards.py` — 167 tests (includes bug fix regression tests)
- `test_data_ingestion_workflow.py` — 125 tests
- `test_streamlit_adapter.py` — 75 tests

**Bug Fix Regression Tests:**
When fixing bugs, add targeted edge case tests to prevent regressions:
- `TestGenericCleanerClassKeyEdgeCases` — 7 tests for ClassKey detection
- `TestSampleColumnDetection` — 5 tests for intensity[...] pattern
- `TestStandardsColumnStandardization` — 8 tests for column standardization
- `TestMetabolomicsWorkbenchParsing` — 6 tests for MW format detection

When writing new tests:
- Create comprehensive fixtures covering all data formats (LipidSearch, MS-DIAL, Generic)
- Include edge case fixtures (empty, single row, all zeros, NaN values, duplicates, special characters)
- Test each public method with multiple scenarios
- Use `MockSessionState` for Streamlit session state mocking (supports both dict and attribute access)
- Use keyword arguments for Pydantic models like `ExperimentConfig`

**Test Structure Pattern:**
- Group tests by class (e.g., `TestDetectStandards`, `TestValidateStandards`)
- Use descriptive test names (e.g., `test_detect_d7_pattern`, `test_empty_dataframe_raises_error`)
- Use pytest fixtures for reusable test data
- Follow Arrange-Act-Assert pattern

### Unit Tests (As You Extract)

Write unit tests for each service as it's extracted. Services are pure Python (no Streamlit), making them easy to test.

```python
# Example: test for FormatDetectionService
def test_detect_lipidsearch_format():
    df = pd.read_csv("sample_datasets/lipidsearch5_sample_dataset.csv")
    result = FormatDetectionService.detect_format(df)
    assert result == "LipidSearch 5.0"
```

### Integration Tests (After Services Extracted)

Once services are extracted, create integration tests that run data through the full pipeline:
- 4 sample datasets (LipidSearch, MS-DIAL, Generic, Metabolomics Workbench)
- Edge case datasets as needed

### Testing After Each Change
1. Run unit tests: `pytest tests/unit/ -v`
2. Manual app test: `streamlit run src/main_app.py`
3. Commit if passing

---

## Coding Patterns

### Services: Stateless Static Methods
```python
# CORRECT
class DataCleaningService:
    @staticmethod
    def clean_data(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
        # Pure function, no state
        pass

# WRONG
class DataCleaningService:
    def __init__(self):
        self.df = None  # NO state storage
```

### Error Messages: User-Friendly
```python
# CORRECT
if df.empty:
    raise ValueError(
        "Dataset is empty. Please upload a file with data rows."
    )

# WRONG
if df.empty:
    raise ValueError("Empty dataframe")
```

### Tests: Arrange-Act-Assert
```python
def test_empty_dataframe_raises_error(self):
    # Arrange
    empty_df = pd.DataFrame()

    # Act & Assert
    with pytest.raises(ValueError, match="empty"):
        service.clean_data(empty_df, config)
```

### Caching: In Adapters, Not Services
```python
# Services - NO caching (pure, testable)
class NormalizationService:
    @staticmethod
    def normalize(df, config):
        ...

# Adapter - ADD caching
@st.cache_data
def normalize_data(df, config):
    return NormalizationService.normalize(df, config)
```

---

## Refactoring Phases

### Phase 1: Setup ✅
1. Create `refactor/v3.0` branch from `main`
2. Rename `main_app.py` → `old_main_app.py`
3. Create folder structure: `src/app/{models, services, adapters, workflows, ui}`
4. Create new minimal `main_app.py`

### Phase 2: Extract Models
Extract Pydantic models for configuration objects:
- ✅ ExperimentConfig — experiment setup with computed sample lists
- ✅ NormalizationConfig — normalization method with validation
- ⬜ StatisticalTestConfig — statistical testing options (mode, test_type, corrections)

**StatisticalTestConfig Details (from old_main_app.py:3481-3577):**
- `mode`: 'auto' or 'manual'
- `test_type`: 'parametric', 'non_parametric', 'auto'
- `correction_method`: 'uncorrected', 'fdr_bh', 'bonferroni', 'auto' (Level 1)
- `posthoc_correction`: 'uncorrected', 'tukey', 'bonferroni', 'auto' (Level 2)
- `alpha`: 0.05 (significance threshold)
- `auto_transform`: bool (log10 transformation)

### Phase 3: Extract Services (one at a time)
For each service:
1. Create service file with static methods
2. Write unit tests
3. Update `main_app.py` to use service
4. **Run integration tests** to verify nothing broke
5. Commit

Priority order:
1. FormatDetectionService
2. DataCleaningService
3. ZeroFilteringService
4. NormalizationService
5. StandardsService

### Phase 4: Extract Workflows & UI
- Create workflows to orchestrate services
- Extract UI components from main_app.py
- Create StreamlitAdapter for session state management

### Phase 5: Polish
- Reduce main_app.py to <500 lines
- Consolidate statistical testing logic
- Ensure all edge cases pass

---

## Development Workflow

### After Each Change
1. Run integration tests: `pytest tests/integration/ -v`
2. Run unit tests: `pytest tests/unit/ -v`
3. Test the app: `streamlit run src/main_app.py`
4. Commit if working: `git add . && git commit -m "description"`

### Key Commands
```bash
# Run app
streamlit run src/main_app.py

# Run all tests
pytest tests/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run specific tests
pytest tests/ -k "normalization" -v

# Check branch
git branch
```

---

## Reference: v2.0 Code

Check these for reusable code (in `refactor/v2.0` branch):

### Models
| Model | Location | Purpose |
|-------|----------|---------|
| ExperimentConfig | `src/lipidcruncher/core/models/experiment.py` | Experiment setup config |
| LipidData | `src/lipidcruncher/core/models/lipid_data.py` | Data structure |
| NormalizationConfig | `src/lipidcruncher/core/models/normalization.py` | Normalization settings |
| StatisticsConfig | `src/lipidcruncher/core/models/statistics.py` | Statistical test settings |

### Services

| Service | Location | Purpose |
|---------|----------|---------|
| DataCleaningService | `core/services/data_cleaning_service.py` | Grade filtering, duplicate removal |
| NormalizationService | `core/services/normalization_service.py` | IS/protein normalization |
| ZeroFilteringService | `core/services/zero_filtering_service.py` | Zero value handling |
| StandardsService | `core/services/standards_service.py` | Internal standards management |
| FormatPreprocessingService | `core/services/format_preprocessing_service.py` | Format-specific preprocessing |
| SampleGroupingService | `core/services/sample_grouping_service.py` | Group assignment |

---

## Sample Data for Testing

| File | Format |
|------|--------|
| `sample_datasets/lipidsearch5_sample_dataset.csv` | LipidSearch 5.0 |
| `sample_datasets/generic_sample_dataset.csv` | Generic |
| `sample_datasets/metabolomic_workbench_sample_data.csv` | Metabolomics Workbench |
| `sample_datasets/msdial_test_dataset.csv` | MS-DIAL |

---

## Rules Summary

**DO:**
- Run integration tests after every change
- Keep services pure (no Streamlit)
- Use Pydantic models
- Write unit tests for new services
- Test edge cases with generated datasets
- Commit after each working change
- Use type hints
- **Follow target architecture** — place new UI components in `src/app/ui/`, not in `main_app.py`

**DON'T:**
- Import Streamlit in services
- Store state in service classes
- Make large changes without commits
- Use v2.0 code without checking if `old_main_app.py` has additional features needed
- Add new UI functions to `main_app.py` — extract to `src/app/ui/` instead
