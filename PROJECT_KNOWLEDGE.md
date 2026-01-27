# LipidCruncher Project Knowledge

**Last Updated:** January 2026
**Current Branch:** `refactor/v3.0`

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
| `refactor/v2.0` | **Reference only** - has clean architecture patterns but missing main's features |
| `refactor/v3.0` | **Active development** - refactoring from main using v2.0 patterns |

### Key Principle
- **Start from `main`** (working code with all features)
- **Use `refactor/v2.0` as reference** (architecture patterns, not code to copy)
- **Refactor incrementally** with testing at each step

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

### Integration Testing (From Day 1)

Create integration tests BEFORE extracting services to establish a "golden path" baseline.

#### Sample Dataset Tests
```python
def test_full_pipeline_lipidsearch():
    """sample_datasets/lipidsearch5_sample_dataset.csv through full pipeline"""

def test_full_pipeline_msdial():
    """sample_datasets/msdial_test_dataset.csv through full pipeline"""

def test_full_pipeline_generic():
    """sample_datasets/generic_sample_dataset.csv through full pipeline"""

def test_full_pipeline_metabolomics_workbench():
    """sample_datasets/metabolomic_workbench_sample_data.csv through full pipeline"""
```

#### Edge Case Datasets (Generated)
Create `tests/fixtures/generators/` with scripts to generate edge case CSVs:

| Edge Case | Description | File |
|-----------|-------------|------|
| Empty file | No data rows | `edge_empty.csv` |
| Single row | Minimal valid data | `edge_single_row.csv` |
| All zeros | All intensity values are 0 | `edge_all_zeros.csv` |
| Missing columns | Required columns missing | `edge_missing_cols.csv` |
| Extra columns | Unexpected columns present | `edge_extra_cols.csv` |
| Special characters | Lipid names with special chars | `edge_special_chars.csv` |
| Large numbers | Very large intensity values | `edge_large_numbers.csv` |
| Negative values | Negative intensities | `edge_negative_values.csv` |
| Mixed case | Column names in mixed case | `edge_mixed_case.csv` |
| Duplicate lipids | Same lipid multiple times | `edge_duplicates.csv` |
| Unicode | Unicode characters in names | `edge_unicode.csv` |
| Whitespace | Leading/trailing whitespace | `edge_whitespace.csv` |
| NaN values | Various NaN representations | `edge_nan_values.csv` |
| One sample per group | Minimal group sizes | `edge_one_per_group.csv` |
| Unbalanced groups | Very different group sizes | `edge_unbalanced.csv` |
| No internal standards | Data without IS lipids | `edge_no_standards.csv` |
| All grades filtered | All lipids have grade D | `edge_all_filtered.csv` |

#### Testing After Each Change
1. Run integration tests: `pytest tests/integration/ -v`
2. Run unit tests: `pytest tests/unit/ -v`
3. Manual app test: `streamlit run src/main_app.py`
4. Commit if passing

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

### Phase 1: Setup
1. Create `refactor/v3.0` branch from `main`
2. Rename `main_app.py` → `old_main_app.py`
3. Create folder structure: `src/app/{models, services, adapters, workflows, ui}`
4. Create new minimal `main_app.py`
5. **Create integration test baseline** using `old_main_app.py` functions
6. **Generate edge case test datasets**

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

## Reference: v2.0 Services

Use these as architectural reference (in `refactor/v2.0` branch):

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
- Skip integration testing
- Make large changes without commits
- Copy code blindly from v2.0
