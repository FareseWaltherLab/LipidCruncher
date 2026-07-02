# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv (required before any Python command)
source lipidcruncher_env/bin/activate

# Run the app
streamlit run src/main_app.py

# Run all tests
pytest tests/ -v

# Run a single test file / test
pytest tests/unit/test_normalization.py -v
pytest tests/unit/test_normalization.py::TestNormalizationService::test_internal_standard -v

# Coverage
pytest tests/ --cov=src/app --cov-report=term-missing
```

`pytest.ini` sets `pythonpath = src`, so tests import as `from app.services...` (not `src.app.services...`).

## Testing

- **Every bug we fix gets a regression test.** Add a test that fails on the buggy code and passes after the fix, in the same change as the fix.

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
- **`StreamlitAdapter` is the only bridge** between Streamlit (session state, `@st.cache_data`) and business logic. UI modules call adapter/workflow methods and render results; data manipulation does not live in UI files.
- **Models are frozen Pydantic** (`ConfigDict(frozen=True)`) so they're hashable and safe to pass to `@st.cache_data`. Don't add mutable fields; use `.model_copy(update=...)` or builder methods like `ExperimentConfig.without_samples()`.

## Supported data formats

Format detection auto-routes to a cleaner via the registry in `src/app/services/data_cleaning/__init__.py`. Adding a format = new cleaner class + registry entry + a `DataFormat` enum value. See `format_detection.py` for the detection signature order (most specific first).

LipidSearch uploads (the `LipidSearch` format — handles both 5.0 and 5.2) auto-detect the delimiter (tab vs comma) and the layout: flat `MeanArea[*]` (5.0 / sample-grouped 5.2) vs. condition-grouped **dual-polarity** `OriginalArea[s{cond}-{file}]` (5.2). The dual-polarity layout requires the **Alignment Setting file** to pair each sample's positive/negative runs before merging into `intensity[s1..sN]` — see `src/app/services/lipidsearch_alignment.py`.

## Further reading

- `CODEBASE.md` — full layer-by-layer reference (services, workflows, adapter contract, session-state keys, troubleshooting, extension guide).
- `README.md` — user-facing overview, install, citation.
- `docs/lipidcruncher_paper.md`, `docs/supplementary_methods.md` — scientific methodology; consult these before changing statistical or normalization defaults.
