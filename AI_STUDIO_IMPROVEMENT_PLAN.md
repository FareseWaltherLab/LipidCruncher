# AI Studio Improvement Plan

## What Has Been Done

### Tool Use (Steps 1-6 — Complete)
- Codebase tools: `search_codebase`, `read_file_section`, `list_files`, `read_documentation`
- Tool-aware API: `execute_tool_loop()` with iteration cap and callbacks
- All services wired to use tools: `DataTransformerService`, `VizGeneratorService`
- Integration tests (7 tests, skip when no API key)

### Unified Chat Attempt
- Merged all tabs into single chat — caused scope confusion
- AI mixed up transform/viz/questions tasks
- Code-only responses returned empty messages (now fixed)
- Error handling improved (errors persisted in chat history)

### Key Bugs Fixed
- Sandbox `import pandas` error (added no-import instruction)
- `if/elif` only picking `fig` OR `result_df` (now collects both)
- Empty response when AI returns code without explanation (now uses stdout)
- Silent error handling (errors now show inline in chat)

---

## Current Problems

1. **Unified chat doesn't enforce scope** — AI tries to do everything and does nothing well
2. **No data format diagnostic** — Transform tab doesn't check if data already fits a format
3. **No Module 1 validation for viz** — Custom viz doesn't verify data went through the pipeline
4. **Few-shot examples too few** — Only 4 examples; need more per-tab examples
5. **UI lacks polish** — No example prompts, no visual guidance for users
6. **No output validation** — Transform output not validated against Generic schema before offering download

---

## Plan

### Step 1: Restore 4-Tab Structure with Strict Scope Prompts

Revert AI Studio page to 4 separate tabs, each with its own system prompt, workflow handler, and UI. Each prompt has a hard SCOPE section that refuses out-of-scope requests.

**Tabs (in order):**
1. **LipidCruncher Questions** — Ask about app features, methods, analysis options
2. **Data Questions** — Explore and summarize the uploaded dataset
3. **Transform to Generic** — Convert data to LipidCruncher Generic format
4. **Custom Visualization** — Create plots from Module 1 output

**Files to change:**
- `src/app/ui/ai_studio_page.py` — 4 tabs instead of unified chat
- `src/app/services/ai_chat/prompt_templates.py` — 4 separate system prompts
- `src/app/ui/ai_chat/chat_panel.py` — add `display_lipidcruncher_questions_chat()`
- `src/app/ui/ai_chat/__init__.py` — export new display function
- `src/app/ui/ai_chat/_lipidcruncher_questions_ui.py` — NEW: LC questions chat UI
- Reuse existing `_data_questions_ui.py`, `_data_transform_ui.py`, `_viz_generation_ui.py`

### Step 2: Transform Tab — Format Diagnostic

Before accepting transform requests, run an automatic diagnostic:

1. On session start, use `FormatDetectionService.detect_format(df)` on the uploaded data
2. If format is LipidSearch/MS-DIAL/Generic/MW → tell user: "Your data is already in **{format}** format. No transformation needed — upload it directly to LipidCruncher."
3. If UNKNOWN → analyze the schema and tell the user:
   - What column appears to contain lipid names
   - Which columns look like sample intensities
   - Which columns are metadata that need to be removed
   - Ask: "Do you want any specific filtering (e.g., remove certain lipid classes, filter by score)?"
4. Only generate code after the user confirms the plan

**Files to change:**
- `src/app/workflows/ai_chat.py` — add diagnostic logic to `start_transform_session()`
- `src/app/services/ai_chat/prompt_templates.py` — add diagnostic instructions to transform prompt
- `src/app/ui/ai_chat/_data_transform_ui.py` — show diagnostic result on session start

### Step 3: Custom Viz — Module 1 Output Validation

Before accepting viz requests, validate the data looks like Module 1 output:

- Check for `LipidMolec` column (standardized lipid names)
- Check for `ClassKey` column (extracted lipid classes)  
- Check for `intensity[sN]` or `concentration[sN]` columns
- If missing → show message: "Your data doesn't appear to have been processed through LipidCruncher Module 1. Please run your data through the analysis pipeline first (Start Crunching → Module 1), then return here for custom visualizations."
- If present → proceed normally

**Files to change:**
- `src/app/workflows/ai_chat.py` — add validation to `start_viz_session()`
- `src/app/ui/ai_chat/_viz_generation_ui.py` — show validation result

### Step 4: Example Prompts in UI

Each tab shows clickable example prompts that auto-fill the chat input:

**LipidCruncher Questions:**
- "What normalization methods are available?"
- "How does LipidCruncher handle missing values?"
- "What statistical tests can I use for 3 conditions?"

**Data Questions:**
- "Describe my dataset"
- "How many lipid species are in each class?"
- "Are there any missing values?"

**Transform to Generic:**
- "Transform my data to Generic format"
- "What columns need to be removed?"
- "My lipid names are in column X, samples are columns Y-Z"

**Custom Visualization:**
- "Create a bar chart of species count per class"
- "Show a box plot of concentrations across conditions"
- "Make a volcano plot for KO vs WT"

**Files to change:**
- All 4 `_*_ui.py` files — add example prompt buttons in the intro message area

### Step 5: Expand Few-Shot Examples Library

Create a separate examples file (`src/app/services/ai_chat/prompt_examples.py`) with 8-12 examples per tab, loaded into system prompts. Examples cover:

**Transform examples (8+):**
- Data with extra metadata columns
- Long-to-wide pivot
- Score-based filtering before transform
- Multi-sheet-like data with condition prefixes
- Data with non-standard lipid name formats
- Data where column 1 isn't lipid names

**Viz examples (8+):**
- Bar chart with error bars per class
- Grouped bar chart across conditions
- Volcano plot (fold-change vs p-value)
- Box/violin plot per condition
- Pie chart of class composition
- Scatter plot with regression
- Heatmap of correlations
- Combined figure + table output

**Data questions examples (4+):**
- Describe dataset
- Species per class count
- Missing value analysis
- Summary statistics

**Files to change:**
- `src/app/services/ai_chat/prompt_examples.py` — NEW: structured example library
- `src/app/services/ai_chat/prompt_templates.py` — import and include examples in prompts

### Step 6: Output Validation for Transforms

After sandbox executes transformation code, validate the output before offering download:

1. Run `DataTransformerService.validate_generic_schema(result_df)` — already exists
2. If invalid → show errors and ask AI to fix (same as current behavior)
3. If valid → additionally check:
   - First column contains string values that look like lipid names
   - Remaining columns are all numeric
   - No obvious metadata columns left (check for common names: "Score", "Mass", "RT", etc.)
4. Show a validation summary: "✓ 450 lipid species, 12 sample columns, no metadata detected"

**Files to change:**
- `src/app/workflows/ai_chat.py` — add post-transform validation summary
- `src/app/services/ai_chat/data_transformer.py` — add metadata column detection

### Step 7: Tests

- Unit tests for format diagnostic in transform workflow
- Unit tests for Module 1 validation in viz workflow
- Unit tests for new prompt templates (scope enforcement)
- Update existing workflow mocks for the tab-specific handlers

**Files to change:**
- `tests/unit/test_ai_chat_workflow.py` — update mocks, add diagnostic tests
- `tests/unit/test_ai_studio_validation.py` — NEW: format and Module 1 validation tests

---

## Implementation Order

1. **Step 1** — Restore 4 tabs with strict prompts (biggest architectural change)
2. **Step 5** — Expand examples library (improves quality immediately)
3. **Step 2** — Transform format diagnostic
4. **Step 3** — Viz Module 1 validation
5. **Step 4** — Example prompts in UI
6. **Step 6** — Output validation
7. **Step 7** — Tests

---

## Key Design Decisions

### Why 4 tabs instead of unified chat?
Each tab has a focused system prompt with clear boundaries. The AI performs significantly better when given a narrow scope with specific examples vs. a broad "do everything" prompt. Users also get clearer mental models of what each tab does.

### Why format diagnostic before transform?
Prevents users from accidentally transforming data that's already in a supported format. Also gives the AI a structured plan to follow rather than guessing what to do.

### Why Module 1 validation for viz?
The visualization prompt assumes columns like `ClassKey`, `concentration[s1]`, etc. If the data hasn't been through Module 1, the AI generates code that references non-existent columns and fails.

### How to expand examples without blowing up token count?
The examples file uses a function per tab that returns only the relevant examples. Each prompt includes ~8 examples (~1500 tokens) — manageable within the context window. Examples are structured as input/output pairs that the AI can pattern-match against.