# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (required before any Python command). Python 3.11, matching
# the Dockerfile and CI; recreate with `python3.11 -m venv lipidcruncher_env`.
source lipidcruncher_env/bin/activate

# Run the app
streamlit run src/main_app.py

# Run all tests (add -n auto to use every core; CI does)
pytest tests/ -v

# Run a single test file / test
pytest tests/unit/test_normalization.py -v
pytest tests/unit/test_normalization.py::TestInternalStandardFormula::test_is_normalization_applies_correct_formula -v

# Coverage
pytest tests/ --cov=src/app --cov-report=term-missing

# Lint (CI fails on any finding; config in ruff.toml)
ruff check src tests
```

`pytest.ini` sets `pythonpath = src`, so tests import as `from app.services...` (not `src.app.services...`).

## Testing

- **Every bug we fix gets a regression test.** Add a test that fails on the buggy code and passes after the fix, in the same change as the fix.
- **`ruff check src tests` must pass** before committing; CI runs it alongside the tests.

## Architecture (read CODEBASE.md for full detail)

Strict layered architecture; each layer only imports from the layer below:

```
UI (src/main_app.py, src/app/ui/)
  → Adapter (src/app/adapters/streamlit_adapter.py)
    → Workflows (src/app/workflows/)
      → Services (src/app/services/)
        → Models (src/app/models/)
```

**Load-bearing invariants — do not break:**

- **Services, workflows, and models have zero Streamlit imports.** They use only static methods (services/workflows) or frozen Pydantic (models). This is what makes them unit-testable. If you find yourself wanting `st.something` in those layers, stop — the work belongs in the UI layer or the adapter.
- **`StreamlitAdapter` is the only bridge** between Streamlit (session state, `@st.cache_data`) and business logic. UI modules call adapter/workflow methods and render results; data manipulation does not live in UI files. UI files may read and write their own session-state keys directly (widget keys, page-local values), but every key belongs to a stage in the registry in `src/app/adapters/session_state.py` (`tests/unit/test_streamlit_adapter.py` scans the UI for unregistered keys), and invalidation only ever goes through the adapter (`invalidate_from`, `invalidate_results_from`, `track_inputs`), never through hand-kept lists of keys to clear.
- **Models are frozen Pydantic** (`ConfigDict(frozen=True)`) so they're hashable and safe to pass to `@st.cache_data`. Don't add mutable fields; use `.model_copy(update=...)` or builder methods like `ExperimentConfig.without_samples()`.

## Supported data formats

The sidebar format dropdown decides how a file is read: `resolve_format_enum` (`src/app/constants.py`) maps it to a `DataFormat`, which routes to a cleaner via the registry in `src/app/services/data_cleaning/__init__.py`. `FormatDetectionService.detect_format` only feeds a sidebar warning when the file looks like a different format. Adding a format = new cleaner class + registry entry + a `DataFormat` enum value + its entry in `FORMAT_OPTIONS` / `get_format_display_to_enum` in `constants.py` (without it, `resolve_format_enum` silently routes the new format to Generic).

LipidSearch uploads (the `LipidSearch` format — handles both 5.0 and 5.2) auto-detect the delimiter (tab vs comma) and the layout: flat `MeanArea[*]` (5.0 / sample-grouped 5.2) vs. condition-grouped **dual-polarity** `OriginalArea[s{cond}-{file}]` (5.2). The dual-polarity layout requires the **Alignment Setting file** to pair each sample's positive/negative runs before merging into `intensity[s1..sN]` — see `src/app/services/lipidsearch_alignment.py`.

## Further reading

- `CODEBASE.md` — full layer-by-layer reference (services, workflows, adapter contract, session-state keys, troubleshooting, extension guide).
- `README.md` — user-facing overview, install, citation.
- `docs/lipidcruncher_paper.md`, `docs/supplementary_methods.md` — scientific methodology; consult these before changing statistical or normalization defaults.
