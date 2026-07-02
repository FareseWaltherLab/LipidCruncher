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
- **AI-generated code is untrusted.** Everything from the AI Studio runs through `SandboxService` (AST validator + curated builtins + timeout) in `src/app/services/ai_chat/sandbox.py`. The sandbox is the security boundary — never bypass it, and never widen its allowlists without justification.

## AI Studio specifics

- The AI Studio (`src/app/ui/ai_studio_page.py`) is a separate page with four tabs (LipidCruncher Questions, Data Questions, Transform to Generic, Custom Visualization). Each tab has its own chat panel under `src/app/ui/ai_chat/_*_ui.py` and a corresponding service under `src/app/services/ai_chat/`.
- AI chat tabs should **defer to LipidCruncher's main pipeline services** for analytic choices (normalization rules, zero-filter thresholds, statistical defaults) rather than re-inventing looser heuristics. If you need a rule, look in `src/app/services/` first.
- The Anthropic API key is read from `ANTHROPIC_API_KEY` (env var, or `st.secrets["ANTHROPIC_API_KEY"]` which `main_app.py` promotes into the env). The AI service layer itself never touches Streamlit.

## Supported data formats

Format detection auto-routes to a cleaner via the registry in `src/app/services/data_cleaning/__init__.py`. Adding a format = new cleaner class + registry entry + a `DataFormat` enum value. See `format_detection.py` for the detection signature order (most specific first).

LipidSearch uploads (the `LipidSearch` format — handles both 5.0 and 5.2) auto-detect the delimiter (tab vs comma) and the layout: flat `MeanArea[*]` (5.0 / sample-grouped 5.2) vs. condition-grouped **dual-polarity** `OriginalArea[s{cond}-{file}]` (5.2). The dual-polarity layout requires the **Alignment Setting file** to pair each sample's positive/negative runs before merging into `intensity[s1..sN]` — see `src/app/services/lipidsearch_alignment.py`.

## Further reading

- `CODEBASE.md` — full layer-by-layer reference (services, workflows, adapter contract, session-state keys, troubleshooting, extension guide).
- `README.md` — user-facing overview, install, citation.
- `docs/lipidcruncher_paper.md`, `docs/supplementary_methods.md` — scientific methodology; consult these (or have the AI Studio's `read_documentation` tool read them) before changing statistical or normalization defaults.
