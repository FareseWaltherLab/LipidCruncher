# LipidCruncher Project Knowledge

**Last Updated:** January 26, 2026
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

### 🔄 Phase 2: Extract Models (IN PROGRESS)
### ⬜ Phase 3: Extract Services (NOT STARTED)
### ⬜ Phase 4: Extract Workflows & UI (NOT STARTED)
### ⬜ Phase 5: Polish (NOT STARTED)

### Next Steps
1. **Extract Pydantic models** (ExperimentConfig, NormalizationConfig, StatisticalTestConfig)
2. Then proceed to Phase 3: Extract FormatDetectionService (with unit tests)

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
- ExperimentConfig
- NormalizationConfig
- StatisticalTestConfig

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

**DON'T:**
- Import Streamlit in services
- Store state in service classes
- Make large changes without commits
- Use v2.0 code without checking if `old_main_app.py` has additional features needed
