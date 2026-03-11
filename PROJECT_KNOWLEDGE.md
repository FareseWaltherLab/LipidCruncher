# LipidCruncher Project Knowledge

**Last Updated:** March 10, 2026
**Current Branch:** `refactor/v3.0`

---

## ✅ Refactor main_app.py (2002 → 206 lines) — COMPLETE

**Problem:** `main_app.py` is bloated with inline content, large methods, and anti-patterns.

**Current state:** 206 lines | **Target:** <500 lines ✅

### Refactoring Plan

#### Step 1: Extract Static Content to `src/app/ui/content/`
Move inline markdown docs and data dictionaries out of main_app.py.

| Extract From | To File | Content |
|--------------|---------|---------|
| `get_sample_data_info()` | `content/sample_data.py` | Sample dataset descriptions |
| `display_data_processing_docs()` | `content/processing_docs.py` | Pipeline documentation (LipidSearch, MS-DIAL, etc.) |
| Normalization docs | `content/normalization_docs.py` | Method formulas table, column renaming note |
| Standards CSV help | `content/standards_help.py` | CSV format examples for standards upload |

**Estimated reduction:** ~300 lines

#### Step 2: Extract Sidebar Components to `src/app/ui/sidebar/`
Break down sidebar functions into focused modules.

| Function(s) | To File | Lines |
|-------------|---------|-------|
| `display_file_upload`, `load_sample_dataset` | `sidebar/file_upload.py` | ~100 |
| `display_column_mapping` | `sidebar/column_mapping.py` | ~130 |
| `display_experiment_definition` | `sidebar/experiment_config.py` | ~90 |
| `display_group_samples`, `display_sample_grouping` | `sidebar/sample_grouping.py` | ~170 |
| `display_bqc_section`, `display_confirm_inputs` | `sidebar/confirm_inputs.py` | ~70 |

**Estimated reduction:** ~560 lines

#### Step 3: Extract Main Content Components to `src/app/ui/main_content/`
Move processing config and normalization UI.

| Function(s) | To File | Lines |
|-------------|---------|-------|
| `display_grade_filtering_config`, `display_quality_filtering_config`, `display_msdial_data_type_selection` | `main_content/data_processing.py` | ~250 |
| `display_manage_internal_standards` | `main_content/internal_standards.py` | ~240 |
| `display_normalization_ui`, `_display_class_selection`, `_display_internal_standards_config`, `_display_protein_config`, `_run_normalization` | `main_content/normalization.py` | ~400 |

**Estimated reduction:** ~890 lines

#### Step 4: Fix Anti-Patterns ✅
- [x] `SampleColumnTracker` class — already removed during Step 2/3 extraction
- [x] Decompose `display_app_page()` (~230 lines → ~115 lines) by extracting:
  - `build_filter_configs()` — format-specific filter config building
  - `run_ingestion_pipeline()` — cached ingestion execution + session state + error handling
  - `display_final_filtered_data()` — data table with ClassKey sorting + download button
- [x] Remove unused imports (`Path`, `BASE_DIR`, `load_sample_dataset`, `DataFormat`, `GradeFilterConfig`, `QualityFilterConfig`, `IngestionResult`)

#### Final Target Structure
After refactoring, `main_app.py` should only contain:
- Imports
- (safe_rerun removed after Streamlit upgrade)
- `display_app_page()` orchestration (~100 lines calling extracted modules)
- `main()` entry point

```
src/app/ui/
├── content/                    # Static markdown content
│   ├── sample_data.py
│   ├── processing_docs.py
│   ├── normalization_docs.py
│   └── standards_help.py
├── sidebar/                    # Sidebar UI components
│   ├── file_upload.py
│   ├── column_mapping.py
│   ├── experiment_config.py
│   ├── sample_grouping.py
│   └── confirm_inputs.py
├── main_content/               # Main area UI components
│   ├── data_processing.py
│   ├── internal_standards.py
│   └── normalization.py
├── landing_page.py             # (existing)
├── format_requirements.py      # (existing)
├── zero_filtering.py           # (existing)
└── standards_plots.py          # (existing)
```

### Progress

| Step | Status | Lines Removed |
|------|--------|---------------|
| Step 1: Extract content | ✅ Done | 171 (2002 → 1831) |
| Step 2: Extract sidebar | ✅ Done | 609 (1831 → 1222) |
| Step 3: Extract main content | ✅ Done | 892 (1222 → 330) |
| Step 4: Fix anti-patterns | ✅ Done | 124 (330 → 206) |

**Step 1 Files Created:**
- `src/app/ui/content/__init__.py`
- `src/app/ui/content/sample_data.py` — Sample dataset info dict
- `src/app/ui/content/processing_docs.py` — Pipeline docs per format
- `src/app/ui/content/normalization_docs.py` — Method formulas table
- `src/app/ui/content/standards_help.py` — CSV format examples

**Step 2 Files Created:**
- `src/app/ui/sidebar/__init__.py` — Package init with exports
- `src/app/ui/sidebar/file_upload.py` — Format selection, sample data, file upload
- `src/app/ui/sidebar/column_mapping.py` — Column standardization, MS-DIAL sample override
- `src/app/ui/sidebar/experiment_config.py` — Experiment definition (conditions, samples)
- `src/app/ui/sidebar/sample_grouping.py` — Sample grouping, manual regrouping
- `src/app/ui/sidebar/confirm_inputs.py` — BQC specification, input confirmation

**Step 3 Files Created:**
- `src/app/ui/main_content/__init__.py` — Package init with exports
- `src/app/ui/main_content/data_processing.py` — Processing docs, grade filtering, MS-DIAL data type, quality filtering
- `src/app/ui/main_content/internal_standards.py` — Standards management (auto-detect, upload, consistency plots)
- `src/app/ui/main_content/normalization.py` — Class selection, IS config, protein config, normalization execution

**Step 4 Changes:**
- `src/app/ui/main_content/data_processing.py` — Added `build_filter_configs()`, `run_ingestion_pipeline()`, `display_final_filtered_data()`
- `src/main_app.py` — Simplified from 330 → 206 lines; `display_app_page()` reduced from ~230 → ~115 lines

**Step 5: Refactor long UI methods (`85832f9`):**
- `display_manage_internal_standards()` (211→47 lines) — extracted `_display_auto_detected_standards()`, `_display_custom_upload()`, `_display_preserved_custom_standards()`, `_process_uploaded_standards()`
- `display_column_mapping()` (126→40 lines) — extracted `_apply_msdial_sample_override()`
- `_display_protein_config()` (107→25 lines) — extracted `_display_manual_protein_input()`, `_display_csv_protein_upload()`

---

## ✅ AI Handoff: Integration Tests Complete

**Status:** Integration tests complete with **73 tests passing**.

**Files Created:**
- `tests/integration/test_module1_pipeline.py` - 73 integration tests for Module 1

**Completed Work:**
1. ✅ Fixed MS-DIAL sample column detection by creating custom loader for tests
2. ✅ Added TestNormalizationNone (4 tests)
3. ✅ Added TestNormalizationInternalStandard (6 tests)
4. ✅ Added TestNormalizationProtein (4 tests)
5. ✅ Added TestNormalizationBoth (3 tests)
6. ✅ Added TestIngestionNormalizationIntegration (7 tests)
7. ✅ Added TestEdgeCases (12 tests)
8. ✅ Added TestErrorHandling (9 tests)

**Test Summary:**
- TestLipidSearchPipeline: 11 tests
- TestMSDIALPipeline: 8 tests
- TestGenericPipeline: 6 tests
- TestMetabolomicsWorkbenchPipeline: 4 tests
- TestNormalizationNone: 4 tests
- TestNormalizationInternalStandard: 6 tests
- TestNormalizationProtein: 4 tests
- TestNormalizationBoth: 3 tests
- TestIngestionNormalizationIntegration: 7 tests
- TestEdgeCases: 12 tests
- TestErrorHandling: 9 tests
- **Total: 73 tests**

**Ready for Next Phase:**
- Module 2: Quality Check (QualityCheckWorkflow extraction)
- Module 3: Visualize and Analyze (AnalysisWorkflow extraction)

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
| QualityCheckService | ✅ Done | 197 tests | `05d751a`, `7cdbbdb` |
| QualityCheckWorkflow | ✅ Done | 136 tests | `7c24e30`, `7cdbbdb` |
| **Module 2 UI** | ✅ Done | - | `9b111cd` |

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

**Bugs Fixed (`caf00d2`, `9797440`, `b271999`) — SessionState pre-population pattern:**
`SessionState` initializes all keys to `None`, so `'key' not in st.session_state` checks always fail (key exists, value is `None`). Fix: use `st.session_state.get('key') is None` instead.
1. ✅ **Generic format column mapping not displayed** (`caf00d2`) — `_standardize_generic()` never set `column_mapping`. Same fix for `n_intensity_cols`.
2. ✅ **Zero filtering slider defaults not applied** (`9797440`) — Non-BQC (75%) and BQC (50%) thresholds defaulted to `None` instead of intended values.
3. ✅ **Protein input method default not set** (`b271999`) — `protein_input_method` defaulted to `None` instead of "Manual Input", showing "Upload CSV File" first.

**Bugs Fixed (`aeaa5df`) — Module navigation widget state loss:**
When navigating from Module 1 to Module 2 and back, Streamlit removes widget keys for non-rendered widgets. `initialize_session_state()` re-creates them with defaults, losing user selections. Fix: restore widget keys from separate persistence keys before rendering.
1. ✅ **`standards_source_radio`** — Restore from `standards_source` (guard: `not in st.session_state`)
2. ✅ **`standards_location_radio`** — Restore from `custom_standards_mode` (guard: `not in st.session_state`)
3. ✅ **`norm_method_selection`** — Added `_preserved_norm_method_selection` to `SessionState` + `on_change` callback (same pattern as grade filtering)
4. ✅ **`protein_input_method`** — Restore from `protein_input_method_prev` instead of hardcoding "Manual Input"

**Pattern:** Widgets with `on_change` callbacks (grade filtering, norm method) can set the widget key on every render since the callback updates the persistence key before the next rerun. Widgets without `on_change` (standards source/location) must guard with `not in st.session_state` to avoid overwriting user clicks.

**Performance Improvement (`745dcc8`):**
1. ✅ **Add st.cache_data caching to workflow calls** — Previously, every UI interaction (slider moved, checkbox clicked) caused full data reprocessing. Now cached results are returned instantly when inputs haven't changed.
   - Added `StreamlitAdapter.run_ingestion()` — Cached wrapper for `DataIngestionWorkflow.run()`
   - Added `StreamlitAdapter.run_normalization()` — Cached wrapper for `NormalizationWorkflow.run()`
   - Updated `main_app.py` to use cached adapter methods instead of direct workflow calls
   - Fixed multiselect default validation to handle data changes gracefully (prevents "default value must exist in options" error)

#### Integration Tests for Module 1 (COMPLETE ✅)

**File:** `tests/integration/test_module1_pipeline.py`
**Result:** 73 tests, all passing

| # | Task | Tests | Status |
|---|------|-------|--------|
| 1 | Setup: imports, constants, helper functions | - | ✅ |
| 2 | Fixtures: experiment configs for each format | - | ✅ |
| 3 | Fixtures: sample dataset loaders (all 4 formats) | - | ✅ (Custom MS-DIAL loader) |
| 4 | Fixtures: edge case DataFrames (empty, zeros, NaN, etc.) | - | ✅ |
| 5 | Fixtures: internal standards and protein concentrations | - | ✅ |
| 6 | `TestLipidSearchPipeline` - full pipeline for LipidSearch | 11 | ✅ |
| 7 | `TestMSDIALPipeline` - full pipeline for MS-DIAL | 8 | ✅ |
| 8 | `TestGenericPipeline` - full pipeline for Generic | 6 | ✅ |
| 9 | `TestMetabolomicsWorkbenchPipeline` - full pipeline for MW | 4 | ✅ |
| 10 | `TestNormalizationNone` - 'none' method tests | 4 | ✅ |
| 11 | `TestNormalizationInternalStandard` - IS method tests | 6 | ✅ |
| 12 | `TestNormalizationProtein` - protein method tests | 4 | ✅ |
| 13 | `TestNormalizationBoth` - combined method tests | 3 | ✅ |
| 14 | `TestIngestionNormalizationIntegration` - end-to-end flow | 7 | ✅ |
| 19 | `TestEdgeCases` - programmatic edge cases | 12 | ✅ |
| 20 | `TestErrorHandling` - error propagation tests | 9 | ✅ |

**Existing Unit Test Coverage (for reference):**
- Zero filtering: 102 tests in `test_zero_filtering.py`
- Grade filtering (LipidSearch): 12+ tests in `test_data_cleaning.py`
- Quality filtering (MS-DIAL): Tests in `test_data_cleaning.py`
- External standards: 8+ tests in `test_data_ingestion_workflow.py`, `test_standards.py`
- Protein concentrations: 20+ tests in `test_normalization.py`, `test_normalization_config.py`

**Rationale:** Module 1 workflows are now pure Python with clean interfaces. Integration tests will:
- Ensure the orchestration between services works correctly
- Provide a safety net before building Modules 2 & 3 on top
- Lock down expected behavior (Modules 2 & 3 are downstream consumers)

#### Streamlit Upgrade + UI Tests

**Problem:** Streamlit 1.22.0 lacks `st.rerun()` and the `AppTest` testing framework. This forced `safe_rerun()` compatibility wrappers across 5 files and left the entire UI layer untested — the MS-DIAL override bug (`d9451cf`) wasn't caught by the 1390 existing tests.

**Phase A: Upgrade Streamlit ✅ (`0978340`)**

Upgraded to Streamlit 1.50.0, removed `safe_rerun()` wrappers, fixed deprecated `use_column_width` parameter.

**Phase B: Add UI Tests ✅ (COMPLETE — 26/26 tests passing)**

**Approach:** `AppTest.from_function()` with focused wrapper functions that import UI components directly (never import `main_app.py` — it has `st.set_page_config()` at module level). Pre-populate session state for data-dependent tests.

**Files to create:**
- `tests/ui/__init__.py` — Package init (empty)
- `tests/ui/conftest.py` — Wrapper functions + shared fixtures
- `tests/ui/test_module1_ui.py` — 26 tests in 8 classes

**Key decisions:**
- Set `PDF2IMAGE_AVAILABLE = False` in wrappers that render landing page (avoid PDF conversion timeouts)
- Use `default_timeout=15` (default 3s too short for import-heavy scripts)
- Access widgets by key where possible, by index otherwise
- `st.file_uploader` is NOT supported by AppTest — use sample data loading instead
- Widgets inside `st.expander()` ARE accessible by key (expanders are layout containers; the script runs top-to-bottom and registers all widgets)
- `st.image()` renders without crashing but isn't inspectable
- `st.rerun()` is handled automatically by AppTest's `.run()`

##### AppTest API Quick Reference (Streamlit 1.50.0)

```python
from streamlit.testing.v1 import AppTest

# Create and run
at = AppTest.from_function(my_func, default_timeout=15)
at.session_state['key'] = value  # Pre-populate BEFORE first run
at.run()

# Access widgets (main area)
at.button[0]                        # By index
at.button(key='my_key')             # By key
at.radio(key='my_radio')            # Radio by key
at.selectbox[0]                     # Selectbox by index

# Access sidebar widgets
at.sidebar.selectbox[0]             # Sidebar selectbox
at.sidebar.button(key='load_sample') # Sidebar button by key
at.sidebar.radio(key='bqc_radio')   # Sidebar radio by key
at.sidebar.checkbox(key='confirm_checkbox')

# Widget properties
widget.value                        # Current value
widget.label                        # Display label
widget.options                      # Options list (selectbox, radio)

# Interact (MUST call .run() after each interaction)
at.button[0].click().run()
at.selectbox[0].set_value('MS-DIAL').run()
at.radio(key='bqc_radio').set_value('Yes').run()
at.checkbox(key='confirm_checkbox').check().run()
at.number_input[0].set_value(3).run()
at.text_input(key='cond_name_0').set_value('WT').run()

# Read session state after run
at.session_state['page']            # Read value
at.session_state['raw_df']          # Read DataFrame

# Read rendered text elements
at.text[0].value                    # st.text() output
at.markdown[0].value                # st.markdown() output
```

##### Widget Keys Reference (from UI source files)

| Widget | Source File | Key | Access Pattern |
|--------|-----------|-----|----------------|
| Format selectbox | `sidebar/file_upload.py:45` | None | `at.sidebar.selectbox[0]` |
| Load Sample button | `sidebar/file_upload.py:102` | `load_sample` | `at.sidebar.button(key='load_sample')` |
| Clear & Upload button | `sidebar/file_upload.py:113` | None | by index |
| Conditions count | `sidebar/experiment_config.py:112` | None | `at.sidebar.number_input[0]` |
| Condition name | `sidebar/experiment_config.py:126` | `cond_name_{i}` | `at.text_input(key='cond_name_0')` |
| Sample count | `sidebar/experiment_config.py:133` | `n_samples_{i}` | `at.number_input(key='n_samples_0')` |
| Grouping radio | `sidebar/sample_grouping.py:69` | `grouping_radio` | `at.sidebar.radio(key='grouping_radio')` |
| BQC radio | `sidebar/confirm_inputs.py:30` | `bqc_radio` | `at.sidebar.radio(key='bqc_radio')` |
| BQC label radio | `sidebar/confirm_inputs.py:47` | `bqc_label_radio` | `at.sidebar.radio(key='bqc_label_radio')` |
| Confirm checkbox | `sidebar/confirm_inputs.py:91` | `confirm_checkbox` | `at.sidebar.checkbox(key='confirm_checkbox')` |
| Start Crunching button | `ui/landing_page.py:160` | None | `at.button[0]` |
| Back to Home button | `main_app.py:84,183` | None | filter by label |
| MS-DIAL data type radio | `main_content/data_processing.py:160` | `msdial_data_type_radio` | `at.radio(key='msdial_data_type_radio')` |
| Grade filter mode radio | `main_content/data_processing.py:76` | `grade_filter_mode_radio` | `at.radio(key='grade_filter_mode_radio')` |
| Quality level radio | `main_content/data_processing.py:219` | `msdial_quality_level_radio` | `at.radio(key='msdial_quality_level_radio')` |
| Standards source radio | `main_content/internal_standards.py` | `standards_source_radio` | `at.radio(key='standards_source_radio')` |

##### Session State Keys (from `streamlit_adapter.py` SessionState dataclass)

**Data states:**
- `raw_df`: DataFrame or None — raw uploaded data
- `standardized_df`: DataFrame or None — after column standardization
- `cleaned_df`: DataFrame or None — after filtering
- `intsta_df`: DataFrame or None — internal standards
- `normalized_df`: DataFrame or None — after normalization
- `continuation_df`: DataFrame or None — current working data

**Experiment:**
- `experiment`: ExperimentConfig or None
- `format_type`: DataFormat or None
- `bqc_label`: str or None

**Flow control:**
- `page`: str = `'landing'` — page routing (`'landing'` or `'app'`)
- `confirmed`: bool = False — confirm checkbox state
- `grouping_complete`: bool = True
- `using_sample_data`: bool = False

**MS-DIAL specific:**
- `msdial_features`: dict = {} — format detection results (keys: `has_normalized_data`, `raw_sample_columns`, `normalized_sample_columns`, `has_quality_score`, `has_msms_matched`)
- `msdial_use_normalized`: bool = False
- `msdial_data_type_index`: int = 0
- `_msdial_override_samples`: list or absent — saved sample override (set by `_apply_msdial_sample_override()`, popped by `reset_data_state()` at line 201)

**Session state initialization:** `StreamlitAdapter.initialize_session_state()` only sets defaults if keys don't already exist. Pre-setting session state before `.run()` works correctly.

##### main_app.py Structure (206 lines)

```python
# Module level (lines 20-56):
st.set_page_config(...)              # CANNOT import this file in tests
StreamlitAdapter.initialize_session_state()

# display_app_page() orchestration (lines 63-186):
#   1. display_format_selection()                   → sidebar selectbox
#   2. display_file_upload(data_format)              → sidebar expander + file_uploader
#   3. standardize_uploaded_data(raw_df, data_format) → column standardization
#   4. display_column_mapping(standardized_df, data_format) → mapping table + MS-DIAL override
#   5. display_sample_grouping(df, data_format)      → experiment + grouping + BQC + confirm
#   6. build_filter_configs(data_format, raw_df)     → grade/quality filtering
#   7. run_ingestion_pipeline(...)                   → cached ingestion workflow
#   8. display_zero_filtering_config(...)            → zero filtering
#   9. display_final_filtered_data(cleaned_df)       → data table
#  10. display_manage_internal_standards(...)         → standards management
#  11. display_normalization_ui(...)                  → normalization
#  12. Back to Home button

# main() (lines 193-198):
#   if page == 'landing': display_landing_page()
#   elif page == 'app': display_app_page()
```

**Critical:** When no data is loaded (`raw_df is None`), only the "Back to Home" button and an info message render. The MS-DIAL override restoration logic (lines 96-102) checks `_msdial_override_samples` in session state after re-standardization.

##### Wrapper Functions (code patterns for `conftest.py`)

Each wrapper must: (1) import inside the function, (2) call `StreamlitAdapter.initialize_session_state()`, (3) never call `st.set_page_config()`.

**`landing_page_script`** — Groups 1:
```python
def landing_page_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    import app.ui.landing_page as lp
    lp.PDF2IMAGE_AVAILABLE = False  # Avoid PDF conversion timeouts
    from app.ui.landing_page import display_landing_page
    display_landing_page()
```

**`format_and_upload_script`** — Groups 2, 3:
```python
def format_and_upload_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.sidebar.file_upload import display_format_selection, display_file_upload
    from app.ui.sidebar.column_mapping import standardize_uploaded_data
    data_format = display_format_selection()
    raw_df = display_file_upload(data_format)
    if raw_df is not None:
        if st.session_state.get('standardized_df') is None:
            std_df = standardize_uploaded_data(raw_df, data_format)
            st.session_state.standardized_df = std_df
        st.text(f"data_loaded:{raw_df.shape[0]}x{raw_df.shape[1]}")
    else:
        st.text("no_data")
```

**`full_sidebar_script`** — Groups 4, 5:
```python
def full_sidebar_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.sidebar.file_upload import display_format_selection, display_file_upload
    from app.ui.sidebar.column_mapping import standardize_uploaded_data, display_column_mapping
    from app.ui.sidebar.sample_grouping import display_sample_grouping
    data_format = display_format_selection()
    raw_df = display_file_upload(data_format)
    if raw_df is not None:
        if st.session_state.get('standardized_df') is None:
            std_df = standardize_uploaded_data(raw_df, data_format)
            if std_df is not None:
                st.session_state.standardized_df = std_df
        std_df = st.session_state.get('standardized_df')
        if std_df is not None:
            mapping_valid, modified_df = display_column_mapping(std_df, data_format)
            if mapping_valid:
                if modified_df is not None:
                    std_df = modified_df
                    st.session_state.standardized_df = std_df
                experiment, bqc_label = display_sample_grouping(std_df, data_format)
                if experiment is not None:
                    st.text(f"confirmed:{experiment.n_conditions}c")
                else:
                    st.text("not_confirmed")
    else:
        st.text("no_data")
```

**`app_page_script`** — Group 8:
```python
def app_page_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    import app.ui.landing_page as lp
    lp.PDF2IMAGE_AVAILABLE = False
    from app.ui.landing_page import display_logo
    from app.ui.format_requirements import display_format_requirements
    from app.ui.sidebar import display_format_selection, display_file_upload
    _, center, _ = st.columns([1, 3, 1])
    data_format = display_format_selection()
    with center:
        display_logo(centered=True)
        display_format_requirements(data_format)
    raw_df = display_file_upload(data_format)
    if raw_df is None:
        with center:
            st.info("Upload a dataset or load sample data to begin.")
            if st.button("← Back to Home"):
                st.session_state.page = 'landing'
                st.rerun()
```

**`msdial_data_type_script`** — Group 6:
```python
def msdial_data_type_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import display_msdial_data_type_selection
    display_msdial_data_type_selection()
    use_norm = st.session_state.get('msdial_use_normalized', False)
    st.text(f"use_normalized:{use_norm}")
```

**`override_preservation_script`** — Group 7 (test 2):
```python
def override_preservation_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import display_msdial_data_type_selection
    display_msdial_data_type_selection()
    override = st.session_state.get('_msdial_override_samples')
    st.text(f"override:{override}")
    st.text(f"std_df_cleared:{st.session_state.get('standardized_df') is None}")
```

**`override_reset_script`** — Group 7 (test 3):
```python
def override_reset_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    override = st.session_state.get('_msdial_override_samples')
    st.text(f"override:{override}")
    if st.button("Reset", key='reset_btn'):
        StreamlitAdapter.reset_data_state()
        st.rerun()
```

##### Fixtures (for `conftest.py`)

```python
DEFAULT_TIMEOUT = 15

@pytest.fixture
def landing_app():
    return AppTest.from_function(landing_page_script, default_timeout=DEFAULT_TIMEOUT).run()

@pytest.fixture
def format_upload_app():
    return AppTest.from_function(format_and_upload_script, default_timeout=DEFAULT_TIMEOUT).run()

@pytest.fixture
def full_sidebar_app():
    return AppTest.from_function(full_sidebar_script, default_timeout=DEFAULT_TIMEOUT).run()

@pytest.fixture
def generic_sidebar_app(full_sidebar_app):
    """Generic data loaded + 3x4=12 experiment config matching 12-sample dataset."""
    at = full_sidebar_app
    at.sidebar.button(key='load_sample').click().run()
    # Set 3 conditions x 4 samples = 12 (matches Generic dataset)
    at.sidebar.number_input[0].set_value(3).run()
    at.number_input(key='n_samples_0').set_value(4).run()
    at.number_input(key='n_samples_1').set_value(4).run()
    at.number_input(key='n_samples_2').set_value(4).run()
    return at

@pytest.fixture
def msdial_features_dict():
    return {
        'has_normalized_data': True,
        'raw_sample_columns': ['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
        'normalized_sample_columns': ['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
        'has_quality_score': True, 'has_msms_matched': False,
    }

@pytest.fixture
def msdial_data_type_app(msdial_features_dict):
    at = AppTest.from_function(msdial_data_type_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['msdial_features'] = msdial_features_dict
    return at.run()
```

##### Test Groups — Detailed Assertions

**Group 1: TestLandingPageNavigation (3 tests)** — Uses `landing_app`
1. `test_start_crunching_button_renders`: `assert len(at.button) >= 1` and `"Start Crunching" in at.button[0].label`
2. `test_start_crunching_sets_page_to_app`: `at.button[0].click().run()` → `assert at.session_state['page'] == 'app'`
3. `test_landing_page_initial_state`: `assert at.session_state['page'] == 'landing'`

**Group 2: TestFormatSelection (3 tests)** — Uses `format_upload_app`
1. `test_format_selectbox_has_four_options`: `assert at.sidebar.selectbox[0].options == ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0', 'MS-DIAL']`
2. `test_format_default_is_generic`: `assert at.sidebar.selectbox[0].value == 'Generic Format'`
3. `test_format_can_switch_to_msdial`: `.set_value('MS-DIAL').run()` → `assert value == 'MS-DIAL'`

**Group 3: TestSampleDataLoading (4 tests)** — Uses `format_upload_app`
1. `test_load_generic_sample_data`: click `load_sample` → `assert at.session_state['raw_df'].shape[0] == 943` and `using_sample_data is True`
2. `test_load_lipidsearch_sample_data`: switch to LipidSearch → click → `assert 'LipidMolec' in raw_df.columns`
3. `test_load_msdial_sample_data`: switch to MS-DIAL → click → `assert at.session_state['msdial_features']['has_normalized_data'] is True`
4. `test_load_metabolomics_workbench_sample_data`: switch to MW → click → `assert raw_df is not None`

**Group 4: TestExperimentDefinition (4 tests)** — Uses `full_sidebar_app` + load sample
1. `test_default_conditions_is_two`: load sample → `assert at.sidebar.number_input[0].value == 2`
2. `test_condition_text_inputs_created`: load sample → `assert at.text_input(key='cond_name_0').value == 'Condition_1'`
3. `test_sample_count_validation_blocks_progress`: default 2x3=6 ≠ 12 → no `grouping_radio` or `confirm_checkbox` rendered
4. `test_correct_sample_count_enables_grouping`: set 3x4=12 → `grouping_radio` and `confirm_checkbox` exist

**Group 5: TestConfirmInputsBQC (4 tests)** — Uses `generic_sidebar_app`
1. `test_bqc_radio_default_is_no`: `assert at.sidebar.radio(key='bqc_radio').value == 'No'`
2. `test_confirm_checkbox_default_unchecked`: `assert at.sidebar.checkbox(key='confirm_checkbox').value is False`
3. `test_checking_confirm_enables_flow`: `.check().run()` → `assert "confirmed:3c" in at.text[0].value`
4. `test_bqc_yes_shows_label_selector`: `.set_value('Yes').run()` → `at.sidebar.radio(key='bqc_label_radio')` exists

**Group 6: TestMSDIALDataTypeSelection (3 tests)** — Uses `msdial_data_type_app`
1. `test_data_type_radio_exists`: `at.radio(key='msdial_data_type_radio')` is not None
2. `test_data_type_default_is_raw`: `"Raw" in radio.value` and text contains `"use_normalized:False"`
3. `test_data_type_can_switch_to_normalized`: set to pre-normalized option → text contains `"use_normalized:True"`

**Group 7: TestMSDIALOverridePreservation (3 tests)** — Regression for d9451cf
1. `test_override_saved_to_session_state`: Pre-set `_msdial_override_samples = ['s1', 's2', 's3']` → persists after run
2. `test_override_preserved_after_data_type_switch`: Pre-set override + switch data type radio → override still in session state, `standardized_df` is None (callback ran, cleared data but NOT override)
3. `test_override_cleared_on_reset`: Pre-set override → click reset button → `_msdial_override_samples` removed from session state (verified: `reset_data_state()` calls `st.session_state.pop('_msdial_override_samples', None)` at `streamlit_adapter.py:201`)

**Group 8: TestBackToHome (2 tests)** — Uses `app_page_script`
1. `test_back_button_exists_when_no_data`: set `page='app'` → `any("Back to Home" in b.label for b in at.button)`
2. `test_back_button_resets_to_landing`: click back button → `assert at.session_state['page'] == 'landing'`

##### Sample Dataset Details

| Format | File | Rows | Sample Cols | Experiment Config |
|--------|------|------|-------------|-------------------|
| Generic | `generic_test_dataset.csv` | 943 | 12 (`intensity[s1]`..`intensity[s12]`) | 3 cond x 4 samples |
| LipidSearch | `lipidsearch5_test_dataset.csv` | ~943 | 12 | 3 cond x 4 samples |
| MS-DIAL | `msdial_test_dataset.csv` | ~1247 | 7 raw + 7 normalized | 3 cond, varied |
| Metabolomics Workbench | `mw_test_dataset.csv` | varies | 44 | 2x2 factorial (auto-detected) |

**Target:** 26 new UI tests + all 1390 existing tests passing.

#### ✅ Code Review Cleanup (COMPLETE — all Critical + High + Medium fixed)
All issues from the **Code Review Issues (February 16, 2026)** section below are fixed except low-priority L1-L4, L7-L8 (deferred).
**Test count:** 1972 passing

#### Module 2: Quality Check (COMPLETE)

**Reference:** Old app `old_main_app.py` lines 1619-2403 has the full working Module 2.

**Scope — 5 sections (all in expanders):**
1. **Box Plots** — Missing values distribution + concentration box plots (always shown)
2. **BQC Quality Assessment** — CoV analysis with filtering (only if BQC samples exist)
3. **Retention Time Plots** — Mass vs retention time (LipidSearch 5.0 AND MS-DIAL, if BaseRt+CalcMass columns exist)
4. **Pairwise Correlation** — Correlation heatmaps per condition
5. **PCA Analysis** — PCA plot with confidence ellipses + sample removal

**NOT in scope:** Module 3 analysis features (bar/pie charts, volcano plots, heatmaps, etc.)

##### Implementation Plan

**Step 1: ✅ Create `QualityCheckService` — Pure Business Logic (NO Streamlit) (`05d751a`)**
**File:** `src/app/services/quality_check.py`

All computation reimplemented inline (no legacy module imports) to avoid transitive Streamlit imports.

**Result Dataclasses:**
- `BoxPlotResult` — `mean_area_df`, `missing_values_percent`, `available_samples`
- `BQCPrepareResult` — `prepared_df` (with cov/mean cols), `bqc_sample_index`, `bqc_samples`, `reliable_data_percent`, `high_cov_lipids`, `high_cov_details`
- `BQCFilterResult` — `filtered_df`, `removed_lipids`, `kept_despite_high_cov`, `lipids_before`, `lipids_after` + computed `removed_count`/`removed_percentage`
- `RetentionTimeDataResult` — `available` (bool), `lipid_classes`
- `CorrelationResult` — `condition`, `correlation_df`, `v_min`, `threshold`, `sample_type`, `condition_samples`
- `PCAResult` — `pc_df` (PC1/PC2), `pc_labels`, `available_samples`, `conditions`
- `SampleRemovalResult` — `updated_df`, `updated_experiment`, `removed_samples`, `samples_before`, `samples_after`

**Public Methods:**
- `prepare_box_plot_data(df, experiment)` → BoxPlotResult
- `prepare_bqc_data(df, experiment, bqc_label, cov_threshold=30)` → BQCPrepareResult
- `filter_by_bqc(df, high_cov_lipids, lipids_to_keep=None)` → BQCFilterResult
- `check_retention_time_availability(df)` → RetentionTimeDataResult
- `get_correlation_eligible_conditions(experiment)` → List[str]
- `compute_correlation(df, experiment, condition, bqc_label=None)` → CorrelationResult
- `compute_pca(df, experiment)` → PCAResult
- `remove_samples(df, experiment, samples_to_remove)` → SampleRemovalResult

**Also public (reusable computation):**
- `calculate_coefficient_of_variation(numbers)` → Optional[float]
- `calculate_mean_including_zeros(numbers)` → Optional[float]

**Two-step interaction patterns:**
- BQC: `prepare_bqc_data()` → UI shows scatter plot + multiselect → `filter_by_bqc()`
- PCA: `compute_pca()` → UI shows plot + multiselect → `remove_samples()`

**Plot rendering stays in legacy modules** — called directly from UI layer:
- `lp.BoxPlot.plot_missing_values()`, `plot_box_plot()`
- `lp.BQCQualityCheck.create_cov_scatter_plot_with_threshold()`
- `lp.Correlation.render_correlation_plot()`
- `lp.PCAAnalysis.plot_pca()`
- `lp.RetentionTime.plot_single_retention()`, `plot_multi_retention()`

**Step 2: ✅ Write Unit Tests for QualityCheckService (`3c1795b`, `7cdbbdb`)**
**File:** `tests/unit/test_quality_check_service.py`
**Result:** 197 tests, all passing

Test classes (14 total):
- `TestCoVCalculation` (13) — formula verification, boundary (single/empty/zeros/negative/mixed sign), input types (numpy, pandas, float)
- `TestMeanCalculation` (9) — formula, zeros, input types, return type
- `TestPrepareBoxPlotData` (13) — structure, missing values %, partial columns, errors, immutability
- `TestPrepareBQCData` (27) — index/samples, CoV/mean/log10, reliability %, high-CoV identification, boundary (first condition, threshold boundary, many BQC samples), row/column preservation, errors
- `TestFilterByBQC` (18) — keep none/some/all, computed properties, ClassKey sorting, index reset, edge cases (nonexistent lipid, no ClassKey, duplicates, keep list extras, all removed, order, data preservation)
- `TestRetentionTimeAvailability` (8) — column presence/absence, class frequency sorting, no ClassKey
- `TestCorrelationEligibleConditions` (8) — multi/single/mixed replicates, exactly-2 boundary, three conditions, many varied
- `TestComputeCorrelation` (21) — matrix shape/symmetry/diagonal, bio vs tech thresholds, value range [-1,1], anticorrelation, three conditions, errors, immutability, zeros
- `TestComputePCA` (15) — shape, columns, labels, variance explained, conditions mapping, partial samples, errors, boundary (2 samples)
- `TestRemoveSamples` (19) — single/multiple removal, column renaming, experiment update, condition drop, errors, boundary (N-2), nonexistent samples, column count, three conditions, cross-condition renaming
- `TestValidation` (14) — all 5 private validators (DataFrame, concentration columns, BQC label, condition, CoV threshold)
- `TestEdgeCases` (14) — NaN, large dataset, identical samples, constant values, single lipid, special characters, pipeline flows
- `TestTypeCoercion` (15) — string numbers, int/float/int64/float32/object dtypes across all methods (box plot, BQC, correlation, PCA, filter, remove_samples, CoV, mean)
- `TestMultiStepPipelines` (6) — full QC pipeline (no BQC), full QC pipeline (with BQC), PCA→remove→PCA, BQC filter→remove→correlation, retention time in pipeline, box plot after removal

**Step 3: ✅ Create `QualityCheckWorkflow` + Unit Tests**
**Files:** `src/app/workflows/quality_check.py`, `tests/unit/test_quality_check_workflow.py`
**Result:** 136 tests, all passing

Unlike DataIngestionWorkflow/NormalizationWorkflow, this workflow does NOT have a single `run()` method. Quality check is interactive — users make decisions between steps (BQC filtering, PCA sample removal). Each method is called individually from the UI layer.

**Config:**
- `QualityCheckConfig` — `bqc_label`, `format_type` (DataFormat), `cov_threshold` (default 30.0)

**Public Methods:**
- `validate_inputs(df, experiment)` → List[str] (validation errors)
- `run_box_plots(df, experiment)` → BoxPlotResult
- `run_bqc_assessment(df, experiment, config)` → Optional[BQCPrepareResult] (None if no BQC)
- `apply_bqc_filter(df, high_cov_lipids, lipids_to_keep)` → BQCFilterResult
- `check_retention_time_availability(df, config)` → RetentionTimeDataResult (format-aware)
- `get_eligible_correlation_conditions(experiment)` → List[str]
- `run_correlation(df, experiment, condition, bqc_label)` → CorrelationResult
- `run_all_correlations(df, experiment, bqc_label)` → Dict[str, CorrelationResult]
- `run_pca(df, experiment)` → PCAResult
- `remove_samples(df, experiment, samples_to_remove)` → SampleRemovalResult
- `run_non_interactive(df, experiment, config)` → Dict (all QC steps without user interaction)

Test classes (13 total):
- `TestQualityCheckConfig` (10) — defaults, custom values, all format types, empty/whitespace BQC label, threshold boundaries (very small/large/integer), dataclass equality/inequality
- `TestValidateInputs` (12) — valid, None/empty df, missing columns, no matching samples, boundary
- `TestRunBoxPlots` (9) — return type, shape, samples, missing values, zeros, errors, large dataset
- `TestRunBQCAssessment` (11) — None when no BQC, valid result, samples, cov/mean columns, threshold, immutability
- `TestApplyBQCFilter` (9) — remove/keep lipids, empty list, computed properties, sorted, index reset
- `TestCheckRetentionTimeAvailability` (7) — Generic/MW unavailable, LS/MSDIAL with/without columns, class frequency
- `TestGetEligibleCorrelationConditions` (4) — all eligible, none eligible, mixed, large
- `TestRunCorrelation` (10) — return type, matrix shape, bio/tech thresholds, samples, diagonal, range, errors
- `TestRunAllCorrelations` (6) — dict of results, skips single replicate, empty, bqc passthrough, three conditions
- `TestRunPCA` (8) — return type, shape, columns, labels, samples, conditions, errors, large
- `TestRemoveSamples` (9) — return type, single/multiple removal, updated experiment/df, errors, immutability
- `TestRunNonInteractive` (12) — all keys, populated results, BQC with/without, RT formats, validation errors, large
- `TestEdgeCases` (15) — single lipid, all zeros, NaN, special chars, min samples, identical, multi-step pipelines
- `TestTypeCoercion` (9) — string numbers, int/int64/float32 concentrations across correlation, BQC, PCA, remove_samples, filter, non-interactive
- `TestImmutability` (6) — all methods preserve input data

**Test depth assessment:** 136 workflow tests are comparable to Data Ingestion (137) and better than Normalization (98). The workflow is a thin delegation layer — deep computation is covered by the 197 QC Service tests. Type coercion and config edge case coverage now matches Module 1 depth.

**Step 4: ✅ Update `StreamlitAdapter` — Session State (`d062f2c`)**
**File:** `src/app/adapters/streamlit_adapter.py`

Added 6 QC session state keys to `SessionState` dataclass:
- `qc_continuation_df`: DataFrame or None — QC working copy (may have samples removed)
- `qc_bqc_plot`: figure or None
- `qc_cov_threshold`: int = 30
- `qc_correlation_plots`: Dict = {}
- `qc_pca_plot`: figure or None
- `qc_samples_removed`: List[str] = []

Automatically handled by existing `initialize_session_state()` and `reset_data_state()` (both iterate `fields(SessionState)`).

**Step 5: ✅ Build Module 2 UI + Wire into main_app.py (`9b111cd`)**

Two files created, three modified. Step 6 (module routing) merged into this step.

**Files Created:**
- `src/app/ui/download_utils.py` (82 lines) — Reusable download helpers:
  - `plotly_svg_download_button(fig, filename, key)` — Plotly figure → SVG download
  - `matplotlib_svg_download_button(fig, filename, key)` — Matplotlib figure → SVG download
  - `convert_df(df)` — DataFrame → CSV bytes
  - `csv_download_button(df, filename, key)` — DataFrame → CSV download
- `src/app/ui/main_content/quality_check.py` (596 lines) — Full QC UI with 5 expander sections:

| Function | Expander Title | Modifies Data? | Plot Lib |
|----------|---------------|----------------|----------|
| `_display_box_plots(df, experiment)` | "View Distributions of AUC: Scan Data & Detect Atypical Patterns" | No | Plotly (lp.BoxPlot) |
| `_display_bqc_assessment(df, experiment, bqc_label)` | "Quality Check Using BQC Samples" | Yes (filters lipids) | Plotly (lp.BQCQualityCheck) |
| `_display_retention_time_plots(df, config)` | "Retention Time Analysis" | No | Plotly (lp.RetentionTime) |
| `_display_correlation_analysis(df, experiment, bqc_label)` | "Pairwise Correlation Analysis" | No | Matplotlib (lp.Correlation) |
| `_display_pca_analysis(df, experiment)` | "Principal Component Analysis (PCA)" | Yes (removes samples) | Plotly (lp.PCAAnalysis) |

**Files Modified:**
- `src/app/ui/main_content/__init__.py` — Added `display_quality_check_module` import and `__all__` entry
- `src/app/adapters/streamlit_adapter.py` — Added 10 QC widget keys to `_WIDGET_KEYS`
- `src/main_app.py` (206 → 269 lines) — Module routing with `_display_module1()`, `_display_module2()`, `_reset_qc_state()`

**Module routing in main_app.py:**
- `_display_module1()` — existing Steps 1-7 + "Next: Quality Check & Analysis →" button (only visible after normalization)
- `_display_module2()` — `display_quality_check_module()` + "Back to Data Processing" / "Back to Home" buttons
- `_reset_qc_state()` — clears all 6 QC session state keys on navigation

**Data flow:** `normalized_df` → box plots (read-only) → BQC (may filter lipids) → RT (read-only) → correlation (read-only) → PCA (may remove samples) → `qc_continuation_df`

**Legacy module calls (rendering):**
- `lp.BoxPlot.create_mean_area_df()`, `.calculate_missing_values_percentage()`, `.plot_missing_values()`, `.plot_box_plot()`
- `lp.BQCQualityCheck.generate_and_display_cov_plot(df, experiment, bqc_sample_index, cov_threshold)`
- `lp.RetentionTime.plot_single_retention(df)`, `.plot_multi_retention(df, classes)`
- `lp.Correlation.prepare_data_for_correlation()`, `.compute_correlation()`, `.render_correlation_plot()`
- `lp.PCAAnalysis.plot_pca(df, full_samples_list, extensive_conditions_list)`

**Workflow calls (business logic):**
- `QualityCheckWorkflow.validate_inputs()`, `.check_retention_time_availability()`, `.get_eligible_correlation_conditions()`, `.apply_bqc_filter()`, `.remove_samples()`

**Widget keys:**
- `bqc_cov_threshold`, `bqc_filter_choice`, `bqc_lipids_to_keep`, `bqc_csv_download`, `bqc_filtered_download`
- `rt_viewing_mode`, `rt_class_selection`
- `corr_condition`, `corr_csv_download`
- `pca_samples_remove`, `pca_csv_download`

**Step 6: Integration Tests ✅**
**File:** `tests/integration/test_module2_pipeline.py`
**Result:** 65 tests in 11 classes — all passing

**Helper Functions:**
- Reuse `load_lipidsearch_sample()`, `load_msdial_sample()`, `load_generic_sample()`, `load_mw_sample()` (copy from Module 1 tests — test files should be self-contained)
- `run_module1_pipeline(raw_df, experiment, data_format) -> pd.DataFrame` — chains `DataIngestionWorkflow.run()` + `NormalizationWorkflow.run(method='none')` to produce normalized DataFrames with `concentration[]` columns for QC input
- `make_qc_dataframe(lipids, classes, n_samples, values_fn)` — builds synthetic QC-ready DataFrames for edge cases

**Fixtures:**
- Experiment configs: `lipidsearch_experiment` (3×4=12), `msdial_experiment` (1+3+3=7), `generic_experiment` (3×4=12), `mw_experiment` (2×22=44)
- Normalized DataFrames (cached via `run_module1_pipeline`): `lipidsearch_normalized_df`, `msdial_normalized_df`, `generic_normalized_df`, `mw_normalized_df`
- QC configs: `lipidsearch_qc_config` (LIPIDSEARCH, bqc='BQC'), `generic_qc_config` (GENERIC, bqc='BQC'), `no_bqc_config` (GENERIC, bqc=None)
- Edge case DataFrames (synthetic): `single_lipid_df`, `two_sample_df`, `uniform_df`, `high_cov_all_df`

**Test Classes:**

| Class | Tests | Focus |
|-------|-------|-------|
| TestLipidSearchEndToEnd | 8 | Full QC pipeline with RT, BQC, all 12 samples |
| TestMSDIALEndToEnd | 6 | 7 samples, Blank excluded from correlation, RT available |
| TestGenericEndToEnd | 5 | No RT, bio vs tech replicate thresholds |
| TestMWEndToEnd | 4 | 44-sample scale, large correlation matrices |
| TestBQCCascadingEffects | 9 | BQC filter → downstream correlation/PCA/box plots |
| TestPCASampleRemoval | 10 | Remove → re-run PCA/correlation, experiment update, renaming |
| TestFormatSpecificBehavior | 5 | RT availability per format, column preservation |
| TestDataIntegrity | 5 | Column preservation, sample count consistency, non-negative values |
| TestEdgeCases | 6 | Single lipid, 2 samples, all removed, uniform data |
| TestErrorHandling | 4 | Missing columns, invalid BQC label, single replicate |
| TestNonInteractivePipeline | 3 | `run_non_interactive` consistency and completeness |

**Key differentiation from unit tests:**
- Multi-step chains: BQC assess → filter → correlate on filtered; PCA → remove → PCA again
- Real sample data through full Module 1 pipeline first
- Cascading state changes: experiment config updates after removal, column renaming propagation
- Cross-method data shape validation: filtered lipid count carries through to PCA/correlation

##### Files Created/Modified

| File | Action | Status |
|------|--------|--------|
| `src/app/services/quality_check.py` | CREATE | ✅ Step 1 |
| `tests/unit/test_quality_check_service.py` | CREATE | ✅ Step 2 |
| `src/app/workflows/quality_check.py` | CREATE | ✅ Step 3 |
| `tests/unit/test_quality_check_workflow.py` | CREATE | ✅ Step 3 |
| `src/app/adapters/streamlit_adapter.py` | MODIFY (QC session state + widget keys) | ✅ Steps 4-5 |
| `src/app/ui/download_utils.py` | CREATE | ✅ Step 5 |
| `src/app/ui/main_content/quality_check.py` | CREATE | ✅ Step 5 |
| `src/app/ui/main_content/__init__.py` | MODIFY (add export) | ✅ Step 5 |
| `src/main_app.py` | MODIFY (module routing) | ✅ Step 5 |
| `tests/integration/test_module2_pipeline.py` | CREATE | ✅ Step 6 |

##### Legacy Modules to Reuse (NOT rewrite)

| Module | Path | Used For |
|--------|------|----------|
| `BoxPlot` | `src/lipidomics/box_plot.py` | Missing values bar chart, box plots |
| `BQCQualityCheck` | `src/lipidomics/bqc_check.py` | CoV calculations and scatter plots |
| `Correlation` | `src/lipidomics/correlation_heatmap.py` | Correlation heatmaps |
| `PCAAnalysis` | `src/lipidomics/pca.py` | PCA computation and plotting |
| `RetentionTime` | `src/lipidomics/retention_time_plot.py` | Retention time plots (LipidSearch only) |

#### Code Quality Sprint: A-Grade Before Module 3

**Principle:** No legacy module (`src/lipidomics/`) imports in the new refactored code. Everything must go through the clean architecture (services → workflows → adapter → UI).

##### Phase 1: Session State Overhaul ✅ (`6b6b7f4`)

**Problem:** Dynamic widget keys leaked across resets, no ownership model for session state keys.

**Changes to `src/app/adapters/streamlit_adapter.py`:**
- **Step 1.1:** SessionState dataclass already had all 54 data/state keys — confirmed complete
- **Step 1.2:** Added 16 missing static widget keys to `_WIDGET_KEYS` (sidebar radios, data processing, standards, normalization, QC download buttons). Added `_DYNAMIC_KEY_PREFIXES` tuple with 9 prefixes (`protein_`, `conc_`, `standard_selection_`, `grade_select_`, `cond_name_`, `n_samples_`, `select_`, `qc_rt_svg_individual_`, `rt_csv_individual_`). Updated `reset_data_state()` with pattern-based cleanup loop.
- **Step 1.3:** Reorganized SessionState fields by owner file with `# --- Owner (owner: file.py) ---` section comments and docstring ownership reference
- **Step 1.4:** All `_preserved_*` keys verified present in SessionState dataclass — pattern is consistent

##### Phase 2: Eliminate Legacy Module Imports

**8 legacy classes to replace across 5 files. Each replacement: rewrite as service/utility → update UI to call new code → add tests.**

**Step 2.1: ✅ Replace `DataFormatHandler`** (`e70e0b3`)
- Created `src/app/services/data_standardization.py` — `DataStandardizationService` + `StandardizationResult` dataclass
  - Pure Python (no Streamlit), returns all outputs via result dataclass
  - Ported: `validate_and_preprocess()` → `validate_and_standardize()`, `_standardize_lipid_name()` → `standardize_lipid_name()`, `_infer_class_key()` → `infer_class_key()`
  - Reuses `FormatDetectionService.MSDIAL_METADATA_COLUMNS` and `._detect_msdial_header_row()` (no duplication)
- Updated `src/app/ui/sidebar/file_upload.py` — uses `DataStandardizationService` for Metabolomics Workbench parsing
- Updated `src/app/ui/sidebar/column_mapping.py` — uses `DataStandardizationService` for all formats, `FormatDetectionService.MSDIAL_METADATA_COLUMNS` for override UI
- Removed `FORMAT_DISPLAY_TO_INTERNAL` from `src/app/constants.py` (no longer needed)
- Tests: 174 in `tests/unit/test_data_standardization.py`

**Step 2.2: ✅ Replace `GroupSamples`** (used in sample_grouping.py)
- Created `src/app/services/sample_grouping.py` — `SampleGroupingService` + 3 result dataclasses (`DatasetValidationResult`, `GroupingResult`, `RegroupingResult`)
  - `validate_dataset()` — replaces `check_dataset_validity()` + `check_input_validity()`
  - `extract_sample_names()` — replaces `extract_sample_names()`
  - `build_group_df()` — replaces `build_group_df()` + `build_mean_area_col_list()`, accepts `workbench_conditions` param instead of reading `st.session_state`
  - `regroup_samples()` — replaces `group_samples()` + `reorder_intensity_columns()` + `update_sample_names()` in a single call
- Updated `src/app/ui/sidebar/sample_grouping.py` — removed `TempExperiment` wrapper and `GroupSamples` import, uses `SampleGroupingService` directly
- Tests: 63 in `tests/unit/test_sample_grouping_service.py`

**Step 2.3: ✅ Replace `InternalStandardsPlotter`** (used in standards_plots.py)
- Created `src/app/services/plotting/standards_plotter.py` — `StandardsPlotterService` with `create_consistency_plots()`
  - Pure Python (no Streamlit), extracted `_build_single_standard_plot()` and `_build_multi_standard_plot()` helpers
  - Identical output to legacy `InternalStandardsPlotter`
- Updated `src/app/ui/standards_plots.py` — uses `StandardsPlotterService` instead of `from lipidomics import InternalStandardsPlotter`
- Tests: 41 in `tests/unit/test_standards_plotter.py`

**Step 2.4: ✅ Replace QC Plotting Classes** (`4594ee1`)
- Created `src/app/services/plotting/box_plot.py` — `BoxPlotService` with `create_mean_area_df()`, `calculate_missing_values_percentage()`, `plot_missing_values()`, `plot_box_plot()`
- Created `src/app/services/plotting/bqc_plotter.py` — `BQCPlotterService` with `prepare_dataframe_for_plot()`, `generate_cov_plot_data()`, `generate_and_display_cov_plot()`, `create_cov_scatter_plot_with_threshold()`
- Created `src/app/services/plotting/retention_time.py` — `RetentionTimePlotterService` with `plot_single_retention()`, `plot_multi_retention()`
- Created `src/app/services/plotting/correlation.py` — `CorrelationPlotterService` with `prepare_data_for_correlation()`, `compute_correlation()`, `render_correlation_plot()`
- Created `src/app/services/plotting/pca.py` — `PCAPlotterService` with `plot_pca()` (includes confidence ellipses)
- Updated `src/app/ui/main_content/quality_check.py` — removed `import lipidomics as lp`, uses new plotting services directly
- Updated `src/app/services/plotting/__init__.py` — exports all 6 plotting services
- All pure Python (no Streamlit), take data in → return figure out
- Tests: 170 across 5 files (`test_box_plot_service.py`: 47, `test_bqc_plotter_service.py`: 38, `test_retention_time_plotter.py`: 30, `test_correlation_plotter.py`: 24, `test_pca_plotter.py`: 31)
- **Phase 2 complete:** Zero `import lipidomics` references remain in `src/app/`

##### Phase 3: DRY & Code Quality

**Step 3.1: Extract Shared Constants ✅**
Created `src/app/constants.py` with:
- `get_format_display_to_enum()` — lazy function (avoids circular import), replaced inline dicts in data_processing.py, quality_check.py, normalization.py
- `FORMAT_DISPLAY_TO_INTERNAL` — replaced inline dict in column_mapping.py
- `INTERNAL_STANDARD_LIPID_PATTERNS` / `INTERNAL_STANDARD_CLASS_PATTERN` — deduplicated from standards.py + base.py
- `LIPIDSEARCH_DETECTION_THRESHOLD` (30000.0) — used by zero_filtering service + UI
- `COV_THRESHOLD_DEFAULT` (30) — used by main_app.py, streamlit_adapter.py, quality_check workflow
- `ZERO_FILTER_NON_BQC_DEFAULT` (75) / `ZERO_FILTER_BQC_DEFAULT` (50) — used by zero_filtering UI
- Note: `MSDIAL_METADATA_COLUMNS` left in `format_detection.py` — only used there (no duplication in refactored code)

**Step 3.2: Break Down Long Methods ✅ (`5a92b2b`)**
- `_display_bqc_assessment()` (145→25 lines) → extracted `_render_bqc_scatter()`, `_render_bqc_filtering()`
- `display_quality_filtering_config()` (119→20 lines) → extracted `_display_score_filtering()`, `_display_msms_only_filtering()`, `_display_quality_filter_summary()`, `_display_cached_filter_results()`
- `_apply_internal_standard()` (100→55 lines) → extracted `_normalize_by_class()`
- `display_normalization_ui()` (102→30 lines) → extracted `_get_normalization_options()`, `_display_method_selection()`, `_collect_method_config()`, `_display_normalization_results()`
- `display_group_samples()` (99→45 lines) → extracted `_handle_manual_regrouping()`
- `display_experiment_definition()` (98→20 lines) → extracted `_display_workbench_auto_detection()`, `_display_manual_experiment()`

**Step 3.3: Merge Duplicate Methods ✅ (`c2194ab`)**
- `get_intensity_column_samples()` / `get_concentration_column_samples()` → `get_column_samples(df, prefix)` with backward-compatible wrappers
- 14 inline CSV download patterns across 5 files → `csv_download_button()` from `download_utils.py`
- `convert_df` no longer imported anywhere in app code (still exported from `download_utils.py`)
- Deduplication logic in cleaners: skipped — fundamentally different strategies (AUC+Grade vs Total Score vs first-occurrence)

**Step 3.4: Move Business Logic Out of UI ✅**
- `_apply_msdial_sample_override()` → pure logic extracted to `DataStandardizationService.apply_msdial_sample_override()` with `MSDIALOverrideResult` dataclass; UI wrapper in `column_mapping.py` only handles session state updates. 17 new tests in `TestApplyMSDIALSampleOverride`.
- Metabolomics Workbench session state handling → deduplicated via `_store_workbench_result()` helper in `file_upload.py` (eliminated duplicate session state code between `load_sample_dataset()` and `display_file_upload()`).

##### Phase 4: Testing & Caching Improvements ✅

**Step 4.1: ✅ Shared Test Fixtures**
Created `tests/conftest.py` with factory functions:
- `make_experiment(n_conditions, samples_per_condition, conditions_list, number_of_samples_list)` — flexible experiment builder
- `make_dataframe(n_lipids, n_samples, classes, lipids, with_classkey, prefix, value_fn)` — flexible DataFrame builder
- Shared fixtures: `simple_experiment_2x3`, `simple_experiment_2x2`, `three_condition_experiment`
- Updated 8 test files to use `make_experiment()`: `test_normalization.py`, `test_zero_filtering.py`, `test_quality_check_service.py`, `test_quality_check_workflow.py`, `test_data_cleaning.py`, `test_data_ingestion_workflow.py`, `test_normalization_workflow.py`, `test_sample_grouping_service.py`
- Updated `test_module1_pipeline.py` integration test
- Created `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` (fixes `from tests.conftest import` resolution)

**Step 4.2: ✅ Cache `load_module_image()`**
Extracted `_convert_pdf_to_png(pdf_path)` with `@st.cache_data` — PDF→PNG conversion now cached, `load_module_image()` only calls `st.image()`.

**Step 4.3: ✅ Verify Hash Functions**
Both `GradeFilterConfig.__hash__()` and `QualityFilterConfig.__hash__()` confirmed present and used in `StreamlitAdapter.run_ingestion()` via `hash_funcs={}`.

**Step 4.4: ✅ Add Type Hints to UI Layer**
Added `Optional`, `Tuple`, return types, and parameter types to:
- `main_app.py` — `_reset_qc_state`, `display_app_page`, `_display_module1`, `_display_module2`, `main`
- `quality_check.py` — all 9 functions (`display_quality_check_module`, `_display_box_plots`, `_display_bqc_assessment`, `_render_bqc_scatter`, `_render_bqc_filtering`, `_display_retention_time_plots`, `_display_correlation_analysis`, `_display_pca_analysis`)
- `file_upload.py` — `_store_workbench_result`, `load_sample_dataset`, `display_file_upload` (→ `Optional[pd.DataFrame]`)
- `column_mapping.py` — `standardize_uploaded_data` (→ `Optional[pd.DataFrame]`), `display_column_mapping` (→ `Tuple[bool, Optional[pd.DataFrame]]`)

##### Phase 5: Cleanup ✅

**Step 5.1: ✅ Remove Dead Code**
- Deleted `src/lipidomics/standards_manager.py` (imported nowhere)
- `fields` import in `streamlit_adapter.py:7` — actually used (lines 210, 224), no action needed
- Removed unused export `load_module_image` from `src/app/ui/__init__.py`
- `src/lipidomics/__init__.py` — all 20 exports used by `old_main_app.py`, kept as-is

**Step 5.2: ✅ Use `pd.api.types.is_numeric_dtype()`**
Replaced `dtype in ['int64', 'float64']` with `pd.api.types.is_numeric_dtype()` in `standards.py:549`

**Step 5.3: ✅ Fix Fragile String Parsing**
Replaced `col[col.find('[') + 1:col.find(']')]` with `col.split('[', 1)[1].rstrip(']')` in:
- `normalization.py:300`
- `data_standardization.py:329`

##### Execution Order

| Order | Phase | Estimated Scope |
|-------|-------|-----------------|
| 1 | Phase 1 (Session State) ✅ | Foundational — everything else depends on clean state |
| 2 | Phase 3.1 (Constants) ✅ | Quick win, unblocks Phase 2 |
| 3 | Phase 2 (Legacy Elimination) ✅ | All 8 legacy classes replaced (Steps 2.1-2.4) |
| 4 | Phase 3.2-3.4 (DRY/Long Methods) ✅ | Improves maintainability |
| 5 | Phase 4 (Testing/Caching) ✅ | Safety net improvements |
| 6 | Phase 5 (Cleanup) ✅ | Final polish |

#### Code Review Before Module 3 (March 9, 2026)

Senior review of session state, test depth, and UI coverage. All issues must be resolved before starting Module 3.

##### Issue 1: Missing Widget Keys in `_WIDGET_KEYS` ✅ (`62eb3ad`)

Added 19 missing widget keys (6 stateful + 13 download/action buttons) to `_WIDGET_KEYS` so they are properly cleared by `reset_data_state()`.

##### Issue 2+3: Dead SessionState Fields + Dead Method ✅

Removed 3 dead fields (`grade_filter_mode_saved`, `grade_selections_saved`, `msdial_quality_level_index`) and dead method `reset_normalization_state()`.

##### Issue 4: Plotter Tests Lack Error Handling + Type Coercion ✅ (`d1c22cb`)

**Fixed:** Added 76 new tests (error handling, type coercion, NaN, boundary) across all 5 plotter test files. Total test count: 2325.

| Plotter Test File | Before | Added | After | `pytest.raises` | Type Coercion | NaN Tests | Boundary |
|-------------------|--------|-------|-------|-----------------|---------------|-----------|----------|
| `test_box_plot_service.py` | 37 | 14 | 51 | 4 | 4 | 4 | 2 |
| `test_bqc_plotter_service.py` | 39 | 22 | 61 | 4 | 7 | 2 | 4 |
| `test_retention_time_plotter.py` | 27 | 14 | 41 | 5 | 3 | 3 | 3 |
| `test_correlation_plotter.py` | 21 | 13 | 34 | 4 | 3 | 4 | 2 |
| `test_pca_plotter.py` | 33 | 13 | 46 | 4 | 3 | 2 | 3 |

New test classes per file: `TestErrorHandling`, `TestTypeCoercion`, `TestNaNHandling`, `TestBoundary`.

##### Issue 5: Module 2 QC UI Has Zero Tests (HIGH) ✅

**File:** `tests/ui/test_module2_ui.py` — 23 tests, all passing

**Approach:** Single `qc_module_script` wrapper parameterized via session state (`_test_df`, `_test_experiment`, `_test_bqc_label`, `_test_format_type`). Data builders in conftest.py: `make_qc_dataframe()` and `make_qc_bqc_dataframe()`.

**Fixtures:** `qc_generic_app` (2×3, Generic, no BQC), `qc_bqc_app` (3+3+2, BQC with 3 high-CoV lipids), `qc_lipidsearch_app` (2×3, LipidSearch with RT), `qc_small_app` (2×2, for PCA min-2 test).

**Fix:** Added `import app.services.format_detection` + `import app.constants` to `tests/ui/conftest.py` to pre-load modules. Note: the root circular import (`app.constants` → `app.services` → `data_cleaning.base` → `app.constants`) was later fixed by converting `FORMAT_DISPLAY_TO_ENUM` to a lazy function `get_format_display_to_enum()` in `constants.py`.

| Class | Tests | Section |
|-------|-------|---------|
| TestQCEntryPoint | 2 | Validation errors (empty df), renders without exception |
| TestBoxPlots | 3 | No exceptions, correct row/sample counts in output |
| TestBQCAssessment | 7 | Skip when no BQC, threshold default/storage, filter radio, filter yes removes lipids |
| TestRetentionTime | 4 | Hidden for Generic, visible for LipidSearch, comparison mode, class multiselect |
| TestCorrelation | 3 | Condition selectbox, bio vs tech replicate info |
| TestPCA | 4 | Multiselect exists, plot stored, removal updates state, min-2 error |

##### Issue 6: Module Navigation Untested (HIGH) ✅ (`4228be8`)

**File:** `tests/ui/test_navigation.py` — 9 tests, all passing

**Approach:** Two wrapper scripts (`module1_nav_script`, `module2_nav_script`) replicate `main_app.py` navigation logic without importing it. Pre-populated QC state verifies `_reset_qc_state()` clears all 6 keys.

| Class | Tests | Coverage |
|-------|-------|---------|
| TestNextToQualityCheck | 3 | Sets module, resets all 6 QC keys, hidden without normalized_df |
| TestBackToDataProcessing | 2 | Resets module to Module 1, clears all QC state |
| TestBackToHomeFromModule2 | 2 | Sets page to landing, clears data state |
| TestStatePreservation | 2 | normalized_df survives round-trip, Module 2 error gate |

##### Issue 7: Module 1 Main Content UI Untested (MEDIUM) ✅

**Problem:** The most complex Module 1 UI components had zero UI tests:
- Normalization UI (469 lines) — class selection, method radio, IS mapping, protein input
- Zero filtering UI (139 lines) — sliders, detection threshold, live preview
- Internal standards UI (279 lines) — auto-detect, custom upload, clear button
- Grade/quality filtering (data_processing.py) — grade radio, per-class multiselects

**Fix:** Added 29 tests in `tests/ui/test_module1_main_content.py`:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestZeroFiltering | 5 | Slider defaults, BQC presence/absence, detection threshold per format |
| TestInternalStandards | 5 | Source radio, auto-detect display, no-standards warning, custom upload switch |
| TestNormalization | 8 | Class multiselect, method radio (with/without standards), protein config, normalization execution |
| TestGradeFiltering | 4 | Default/custom modes, per-class multiselects, default grade values (A/B vs A/B/C) |
| TestQualityFiltering | 4 | Preset options, default moderate, MS/MS checkbox |
| TestColumnMapping | 3 | Generic mapping display, MS-DIAL override multiselect, no override for Generic |

**Wrapper scripts added to `tests/ui/conftest.py`:** `zero_filtering_script`, `internal_standards_script`, `normalization_script`, `grade_filtering_script`, `quality_filtering_script`, `column_mapping_script`.

**Data builders added:** `make_cleaned_dataframe()`, `make_intsta_dataframe()`, `make_grade_dataframe()`.

##### Execution Plan

| Order | Issue | Scope | Priority | Status |
|-------|-------|-------|----------|--------|
| 1 | Issue 1: Missing widget keys | Add to `_WIDGET_KEYS` | HIGH | ✅ `62eb3ad` |
| 2 | Issue 2+3: Dead fields/method | Remove from `streamlit_adapter.py` | MEDIUM | ✅ |
| 3 | Issue 4: Plotter test gaps | 76 new tests across 5 files | HIGH | ✅ `d1c22cb` |
| 4 | Issue 5: Module 2 UI tests | 23 new tests | HIGH | ✅ |
| 5 | Issue 6: Navigation tests | 9 new tests | HIGH | ✅ `4228be8` |
| 6 | Issue 7: Module 1 main content tests | 29 new tests | MEDIUM | ✅ |

**All 6 issues complete.** ✅ Ready for Module 3. Total test count: 2386.

#### Senior Code Review (March 10, 2026) — Pre-Module 3 Fixes

**9 issues resolved (`34d5984`).** All 2357 tests passing.

| # | Issue | Severity | Location | Status |
|---|-------|----------|----------|--------|
| 1 | **Fix config comparison bug** — `run_ingestion_pipeline` compares dict `.get()` against `QualityFilterConfig` object attributes (type mismatch, runtime bug) | HIGH | `data_processing.py:293-295` | ✅ |
| 2 | **Refactor `_process_msdial()`** — 180 lines → split into `_detect_msdial_structure()`, `_standardize_msdial_columns()`, `_build_msdial_mapping()` | HIGH | `data_standardization.py` | ✅ |
| 3 | **Standardize widget preservation pattern** — Added `restore_widget_value()`/`save_widget_value()` to `StreamlitAdapter`. Applied to QC widgets (`bqc_filter_choice`, `rt_viewing_mode`, `pca_samples_remove`) with `_preserved_*` session state keys | HIGH | `streamlit_adapter.py`, `quality_check.py`, `main_app.py` | ✅ |
| 4 | **Replace broad `except Exception`** — Replaced 6 occurrences with specific types (`ValueError`, `KeyError`, `IndexError`, `TypeError`, `AttributeError`, `ZeroDivisionError`) | MEDIUM | `data_standardization.py` | ✅ |
| 5 | **Type `run_non_interactive()` return** — Created `QualityCheckNonInteractiveResult` dataclass with typed fields | MEDIUM | `workflows/quality_check.py` | ✅ |
| 6 | **Extract shared validation utilities** — Created `src/app/services/validation.py` with `validate_dataframe_not_empty()`, `validate_concentration_columns()`, `get_matching_concentration_columns()`. Deduplicated QC service, QC workflow, and QC UI | MEDIUM | `validation.py`, `quality_check.py` (3 files) | ✅ |
| 7 | **Convert `GradeFilterConfig`/`QualityFilterConfig`/`CleaningResult`** to `@dataclass` with custom `__hash__` | MEDIUM | `data_cleaning/configs.py` | ✅ |
| 8 | **Adopt `@pytest.mark.parametrize`** — Invalid lipid rows (7→1 parametrized), filter presets (4→1), RT format availability (3→2 parametrized) | LOW | `test_data_cleaning.py`, `test_quality_check_workflow.py` | ✅ |
| 9 | **Consolidate fixture definitions** — Moved `bqc_experiment` to shared `tests/conftest.py`, removed `three_condition_experiment` duplicates from 4 unit test files | LOW | `tests/conftest.py` + 4 test files | ✅ |

**Files Created:**
- `src/app/services/validation.py` — Shared DataFrame/concentration column validation utilities

**Files Modified (source):**
- `src/app/adapters/streamlit_adapter.py` — Added `restore_widget_value()`, `save_widget_value()`, 3 `_preserved_*` QC keys
- `src/app/services/data_cleaning/configs.py` — Converted all 3 classes to `@dataclass`
- `src/app/services/data_standardization.py` — Split `_process_msdial()` into 3 helpers, narrowed exception types
- `src/app/services/quality_check.py` — Delegates to shared validation utilities
- `src/app/ui/main_content/data_processing.py` — Fixed config comparison bug (`.get()` → attribute access)
- `src/app/ui/main_content/quality_check.py` — Added widget preservation for 3 QC widgets
- `src/app/workflows/quality_check.py` — Added `QualityCheckNonInteractiveResult` dataclass, uses shared validation
- `src/main_app.py` — `_reset_qc_state()` clears 3 new `_preserved_*` keys

**Files Modified (tests):**
- `tests/conftest.py` — Added shared `bqc_experiment` fixture
- `tests/ui/conftest.py` — Updated `_reset_qc_state()` with preserved keys
- `tests/ui/test_navigation.py` — Verifies preserved keys cleared on navigation
- `tests/unit/test_data_cleaning.py` — Parametrized invalid lipid rows + filter presets
- `tests/unit/test_normalization.py` — Removed duplicate `three_condition_experiment`
- `tests/unit/test_normalization_workflow.py` — Removed duplicate `three_condition_experiment`
- `tests/unit/test_quality_check_service.py` — Removed duplicate `bqc_experiment` + `three_condition_experiment`
- `tests/unit/test_quality_check_workflow.py` — Removed duplicates, parametrized RT tests, dict→attribute access
- `tests/integration/test_module2_pipeline.py` — dict→attribute access for `run_non_interactive` results

#### Fix Circular Import in `constants.py` (March 10, 2026)

**Problem:** `constants.py` imported `DataFormat` at module level from `app.services.format_detection`, which triggered `services/__init__.py` → `data_cleaning` → `data_cleaning/base.py` → `app.constants` (still loading) → `ImportError`.

**Fix:** Converted `FORMAT_DISPLAY_TO_ENUM` dict constant to `get_format_display_to_enum()` lazy function with deferred import. Updated 4 consumers: `normalization.py`, `quality_check.py`, `data_processing.py`, `column_mapping.py`. All 2357 tests passing.

#### Senior Code Review (March 11, 2026) — Pre-Module 3 Fixes

Broad review of all ~30 source files, ~23 UI files, ~29 test files (2,460+ tests, 31K lines of test code). Many initial findings were false positives after verification. 8 issues to fix before Module 3. **Test count: 2388.**

##### Issue 1: LipidSearch NaN grade priority (BUG) ✅ (`f37670d`)
**File:** `src/app/services/data_cleaning/lipidsearch.py:357`
**Fix:** Added `.fillna(99)` after `.map(grade_priority)` so NaN grades get lowest priority. Added 2 tests for NaN vs valid grade and all-NaN scenarios.

##### Issue 2: Duplicate button labels without unique keys (BUG) ✅ (`8bdde51`)
**File:** `src/main_app.py`
**Fix:** Added unique keys to all 5 duplicate buttons: `back_home_no_data`, `back_home_module1`, `back_home_module2`, `back_processing_error`, `back_processing_module2`.

##### Issue 3: Long methods in services (CODE QUALITY) ✅ (`c4fad23`)
Extracted 14 helpers across 5 files, all methods now under 50 lines:
- `normalization.py`: `_apply_protein` (78→44) → `_normalize_by_protein`; `validate_normalization_setup` (76→30) → `_validate_internal_standards_setup`, `_validate_protein_setup`
- `standards.py`: `validate_standards` (99→30) → `_validate_standards_content`, `_check_standards_existence`
- `quality_check.py`: `prepare_bqc_data` (75→37) → `_compute_cov_and_mean`, `_identify_high_cov_lipids`; `remove_samples` (72→33) → `_drop_and_rename_columns`
- `box_plot.py`: `plot_box_plot` (89→20) → `_add_box_traces`, `_apply_box_plot_layout`
- `bqc_plotter.py`: `create_cov_scatter_plot` (96→13) → `_prepare_cov_data`, `_add_cov_scatter_traces`, `_apply_cov_layout`

##### Issue 4: Inconsistent session state access in UI (CODE QUALITY)
**Problem:** UI files mix three patterns: (a) direct `st.session_state['key']`, (b) `StreamlitAdapter.save/restore_widget_value()`, (c) `st.session_state.get('key')`.
**Fix:** Add docstring in `streamlit_adapter.py` defining valid patterns and when to use each. Prevents Module 3 from amplifying inconsistency.

##### Issue 5: Missing type hints on UI functions (CODE QUALITY)
**Problem:** ~30 UI functions lack return type annotations.
**Files:** `normalization.py`, `internal_standards.py`, `experiment_config.py`, `sample_grouping.py`, `confirm_inputs.py`, `standards_plots.py`, `zero_filtering.py`
**Fix:** Add return type annotations to all public UI functions.

##### Issue 6: Complex tuple return in `_detect_msdial_structure()` (CODE SMELL)
**File:** `src/app/services/data_standardization.py:505`
**Problem:** Returns 6-element tuple with bare `Dict` type. Hard to understand, error-prone.
**Fix:** Create `MSDIALStructure` dataclass to replace the tuple.

##### Issue 7: Test fixture duplication (TEST QUALITY)
**Problem:** Multiple test files redefine fixtures from `conftest.py` (e.g., local `simple_experiment`, local `three_condition_experiment`).
**Fix:** Replace local fixtures with shared conftest fixtures.

##### Issue 8: `pytest.raises` without `match=` (TEST QUALITY)
**Problem:** ~40 uses of `pytest.raises(ValueError)` without checking error messages in model tests.
**Fix:** Add `match="..."` to critical validation tests.

##### Execution Order

| Order | Issue | Scope | Status |
|-------|-------|-------|--------|
| 1 | Issue 1: LipidSearch NaN bug | 1 file, 2 new tests | ✅ `f37670d` |
| 2 | Issue 2: Button key collision | 1 file, 5 keys | ✅ `8bdde51` |
| 3 | Issue 3: Long methods | 5 files, 14 helpers extracted | ✅ `c4fad23` |
| 4 | Issue 6: Tuple → dataclass | 1 file, ~30 lines + update callers | |
| 5 | Issue 5: Type hints | 7 UI files | |
| 6 | Issue 4: Session state pattern docs | 1 file, docstring | |
| 7 | Issue 7: Fixture dedup | ~5 test files | |
| 8 | Issue 8: pytest.raises match | 3 test files | |

##### Out of Scope (deferred to Phase 5: Polish)
- Session state architecture overhaul (too risky pre-Module 3)
- Business logic refactoring in UI orchestration (works correctly)
- Test assertion count reduction (cosmetic)
- Mock-based testing adoption (current approach valid)
- Widget key enum/registry (nice-to-have)
- Plotting library inconsistency (matplotlib vs plotly — legacy artifact)

##### Verified False Positives (NOT bugs)
- `standards.py:633-634` `.iloc[0]` on Series — guard clause prevents error
- `normalization.py:430` reshape without bounds check — prior validation makes it safe
- `bqc_plotter.py:141-143` NaN filter array lengths — filtered consistently
- `quality_check.py:69-73` error flow continues — actually returns early
- `zero_filtering.py:100-171` filter_zeros 72 lines — well-structured, appropriate size

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

## Code Review Issues (February 16, 2026)

Full code review before starting Module 2. All issues listed by priority.

### Critical — ✅ ALL DONE

#### C1. ✅ DataFrame hash function removed
**Fix applied:** Removed `compute_df_hash()` entirely. The `_df_hash` parameter was prefixed with `_` (excluded from Streamlit caching) and `df` was already hashed natively by `@st.cache_data`. Removed `_df_hash` param from `run_ingestion()`, `run_normalization()`, and both call sites in `data_processing.py` and `normalization.py`.

#### C2. ✅ All session state keys added to SessionState dataclass
**Fix applied:** Added ~26 missing keys to `SessionState` dataclass organized by category (normalization UI state, standards, zero filtering preservation, sidebar state). Also added `_PRESERVE_ON_RESET` and `_WIDGET_KEYS` module-level sets. Refactored `initialize_session_state()` to loop over `fields(SessionState)` instead of manual per-key code.

#### C3. ✅ `reset_data_state()` now clears all state programmatically
**Fix applied:** `reset_data_state()` now iterates `fields(SessionState)` and resets all except `page`/`module`. Also pops `_WIDGET_KEYS` (Streamlit widget keys) to prevent stale widget state. Also fixed `reset_normalization_state()` to clear the new normalization UI keys.

### High Priority — ✅ ALL DONE

#### H1. ✅ Unused typed accessors deleted
**Fix applied:** Removed 12 getter/setter methods (60 lines) and their 9 tests.

#### H2. ✅ Unused cached service wrappers deleted
**Fix applied:** Removed `clean_data()`, `filter_zeros()`, `normalize_data()` (~115 lines). Also removed unused imports (`DataCleaningService`, `CleaningResult`, `ZeroFilteringService`, `ZeroFilterConfig`, `ZeroFilteringResult`, `NormalizationService`, `NormalizationResult`, `StandardsService`, `StandardsExtractionResult`, `StandardsValidationResult`).

#### H3. ✅ Type hint bug fixed
**Fix applied:** Changed to `Optional[List[Tuple[str, str]]] = None` and added `Optional` import.

#### H4. ✅ `StatisticalTestConfig` added to `models/__init__.py`

#### H5. ✅ Missing exports added to `services/__init__.py`
**Fix applied:** Added `ZeroFilteringService`, `ZeroFilterConfig`, `ZeroFilteringResult`, `NormalizationService`, `NormalizationResult`, `StandardsService`, `StandardsExtractionResult`, `StandardsValidationResult`, `StandardsProcessingResult`.

### Medium Priority — ✅ ALL DONE

#### M1. Duplicated column extraction logic across cleaners
**Files:** `src/app/services/data_cleaning/generic.py:149-187` and `src/app/services/data_cleaning/msdial.py:262-320`
**Problem:** `_add_required_columns()` and `_add_intensity_columns()` are nearly identical in both files. The only difference is MS-DIAL also has `_add_optional_columns()`. This violates DRY and means bug fixes must be applied in two places.
**Fix:** Move `_add_required_columns()` and `_add_intensity_columns()` to `BaseDataCleaner` in `base.py`. MS-DIAL can call the base versions and add its own `_add_optional_columns()`.

#### M2. Duplicated step methods across cleaners
**Files:** `generic.py:84-117` and `msdial.py:102-134`
**Problem:** `_step_remove_invalid_rows()`, `_step_remove_duplicates()`, `_step_remove_zero_rows()` are nearly identical in GenericCleaner and MSDIALCleaner. Only the error messages differ slightly.
**Fix:** Move to `BaseDataCleaner` with configurable error messages, or extract a shared `_counting_step()` helper.

#### M3. Broad `except Exception` catches in UI layer
**Files:** 8 occurrences across UI files:
- `landing_page.py:64,88`
- `file_upload.py:150`
- `normalization.py:229,347`
- `internal_standards.py:201`
- `data_processing.py:411`
- `sample_grouping.py:172`

**Problem:** Catching bare `Exception` swallows unexpected errors (`TypeError`, `AttributeError`, `KeyError`) that indicate real bugs, not user input errors. These become invisible and make debugging difficult.
**Fix:** Replace with specific exception types. For file parsing, catch `ValueError`, `KeyError`, `pd.errors.ParserError`. For UI rendering, let unexpected exceptions propagate so they're visible in the Streamlit error display.

#### M4. Serialization boilerplate in cached wrappers
**File:** `src/app/adapters/streamlit_adapter.py:411-544`
**Problem:** Every cached method manually converts Pydantic models to dicts, passes them, then reconstructs with `ExperimentConfig(**experiment_dict)`. This is ~30 lines of boilerplate per method that exists solely because `@st.cache_data` needs hashable arguments.
**Fix:** Define `__hash__` on `ExperimentConfig`, `NormalizationConfig`, `GradeFilterConfig`, and `QualityFilterConfig`. Then pass the objects directly — Streamlit will hash them. This eliminates `experiment_to_dict()`, `config_to_dict()`, and all the reconstruct-from-dict code.

#### M5. ✅ `StreamlitAdapter` class size — no split needed
**File:** `src/app/adapters/streamlit_adapter.py`
**Assessment:** After H1/H2/M4 cleanups, the class is 191 lines (file is 307 lines total). Well-organized with clear sections (session state init/reset, service wrapper, cached workflows). No split needed at current size. Reassess when Modules 2/3 add more cached wrappers.

#### M6. ✅ `NormalizationConfig` cross-field validation added
**File:** `src/app/models/normalization.py`
**Assessment:**
- Empty dicts already fail validation (`not {}` is `True`) — existing tests confirm this
- `protein_concentrations` vs samples: NormalizationConfig doesn't know about samples — belongs at workflow level
**Fix applied:** Added cross-field validation in `validate_method_requirements()`: every standard name (value in `internal_standards`) must have a matching key in `intsta_concentrations`. Added 5 new tests (112 total, up from 108).

### Low Priority

#### L1. `ExperimentConfig.without_samples()` inconsistent return
**File:** `src/app/models/experiment.py:109`
**Problem:** Returns `self` (same instance) when `samples_to_remove` is empty, but creates a new instance otherwise. Callers may not expect identity semantics to vary.
**Fix:** Always create a new instance for consistency.

#### L2. `ExperimentConfig.without_samples()` silently ignores invalid samples
**File:** `src/app/models/experiment.py:112-122`
**Problem:** If `samples_to_remove` contains sample names not in the experiment, they are silently ignored. This could mask bugs in callers.
**Fix:** Add a warning or raise when removing samples that don't exist.

#### L3. `st.set_page_config()` at module level prevents testing
**File:** `src/main_app.py:20-25`
**Problem:** `st.set_page_config()` runs at import time, which prevents importing `main_app.py` in tests. The workaround (wrapper functions in `conftest.py`) means the actual `display_app_page()` orchestration is never tested via AppTest.
**Fix:** Move `st.set_page_config()` inside `main()` with an `if not _called` guard, or accept this as a known limitation.

#### L4. No UI tests for main content area
**Problem:** The 26 UI tests cover landing page, sidebar, and MS-DIAL data type selection, but nothing in the main content area (data processing, zero filtering, internal standards, normalization). These are the most complex UI components.
**Fix:** Add UI tests for main content components after Module 2, when the test patterns are well-established.

#### L5. ✅ `using_sample_data` and `ingestion_result` added to `SessionState` (fixed in C2)

#### L6. ✅ `_msdial_override_samples` now in `SessionState` with consistent reset (fixed in C2/C3)

#### L7. Cached workflow wrappers return bare tuples
**Files:** `streamlit_adapter.py:422-472` (7-element tuple), `streamlit_adapter.py:487-544` (6-element tuple)
**Problem:** Return types are `Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str, bool, List[str], List[str], List[str]]`. Callers must destructure by position — easy to misorder.
**Fix:** Return `IngestionResult` and `NormalizationWorkflowResult` directly. If `@st.cache_data` can't hash them, define `__hash__` or use `@st.cache_resource`.

#### L8. `NormalizationConfig` dict keys not documented in type hints
**File:** `src/app/models/normalization.py:23`
**Problem:** `internal_standards: Optional[Dict[str, str]]` — type doesn't convey that keys are lipid classes and values are standard names. Similarly for other dict fields.
**Fix:** Add type aliases: `ClassToStandard = Dict[str, str]`, `SampleToConcentration = Dict[str, float]`, etc.

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
