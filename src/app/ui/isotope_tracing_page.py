"""
Stable isotope tracing page UI.

A workflow fully separate from the lipidomics pipeline: upload the three
IsoCorrectoR input files (Measurement, Molecule, Element), run
natural-isotope-abundance correction via the real IsoCorrectoR R package, then
view and download the corrected tables.

UI-layer only — all data work goes through StreamlitAdapter / the workflow.
"""
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.constants import PAGE_LANDING
from app.models.isotope_tracing import IsotopeCorrectionConfig
from app.ui.landing_page import display_logo
from app.ui.st_helpers import display_export_buttons
from app.workflows.isotope_tracing import IsotopeTracingWorkflow

# Golden sample dataset shipped with the repo (this file: src/app/ui/...).
_SAMPLE_DIR = Path(__file__).parents[3] / "sample_datasets" / "isotope_tracing"


# =============================================================================
# In-app documentation (mirrors format_requirements.py / data-processing docs)
# =============================================================================

_HOW_TO_RUN_DOC = """
### 🚀 Running a Correction

1. **Load your inputs** in the sidebar — upload the Measurement, Molecule, and
   Element CSVs, or click **Load sample data** for the bundled ¹³C example.
   No Molecule or Element file? Choose **Generate from the Measurement file**
   and check the formulas, ion forms and labelled atoms on the page.
2. **Match the resolution mode to your isotopologue IDs** under *Correction
   settings*: **Ultra-high-resolution mode** ON for element-specific IDs
   (`_C0`, `_C1`), OFF for plain mass-shift IDs (`_0`, `_1`).
3. **Check the remaining settings.** The defaults — correct tracer impurity,
   correct tracer element in the core molecule, calculate mean enrichment —
   suit most tracing experiments.
4. **Click Run correction.** The three files are checked structurally first,
   so most mistakes are reported before IsoCorrectoR is invoked.
5. **Read the outputs.** The isotopologue distribution plot comes first: pick
   a lipid, the conditions (in plot order) and the isotopologues to show, and
   optionally rescale the shown isotopologues to 100%. Regroup samples under
   *Sample conditions* if their names don't follow `<condition>_<replicate>`.
   The corrected intensities follow; the remaining IsoCorrectoR tables and the run
   log are in expanders, each downloadable as CSV.

**If the run is rejected**

| Message | What to do |
|---------|------------|
| *…do not match any molecule in the Molecule file* | Every measurement ID must begin with a molecule name from the Molecule file (`PC_34:1` → `PC_34:1_C0`). Check spelling and separators. |
| *Measurement IDs use ultra-high-resolution / standard-resolution labeling* | The ID style and the *Ultra-high-resolution mode* setting disagree — flip the setting, or relabel the isotopologues. |
| *…contains non-numeric intensity values* | Remove text (`N/A`, `n.d.`, thousands separators) from the sample columns; leave missing values blank (IsoCorrectoR then warns that results for that sample may be less accurate). |
| *R runtime unavailable* | This deployment has no R + IsoCorrectoR installed. The rest of LipidCruncher is unaffected. |
"""

_INPUT_REQUIREMENTS_DOC = """
### 🧪 IsoCorrectoR Input Files

Stable isotope tracing needs **three CSV files** in the IsoCorrectoR format.

**1. Measurement file** — the measured isotopologue intensities

| Position | Content |
|----------|---------|
| Column 1 | Isotopologue IDs: `<Molecule>_<label>` |
| Columns 2+ | One column per sample, with the measured intensity |

The label suffix must match your **resolution mode**:
- **Standard resolution** — total mass shift: `_0`, `_1`, `_2`, …
- **Ultra-high resolution** — element-specific: `_C0`, `_C1`, … (turn on *Ultra-high-resolution mode*)

```
Measurements/Samples,Sample_A,Sample_B
BMP_18:1_18:1_C0,26588314,24796338
BMP_18:1_18:1_C1,11962732,10856478
PC_34:1_C0,85698314,77001200
```

**Sample names set the conditions in the plots.** Name each sample
`<condition>_<replicate>`, e.g. `WT_13C_1`, `WT_13C_2`, `WT_13C_3`: samples that
differ only in the trailing number are replicates of one condition (`WT_13C`).
Capitals don't matter: `WT_Neg_1` and `WT_neg_2` are the same condition.
A name without a trailing `_<number>` is a condition of its own. Either way,
the grouping can be edited by hand on the results page (*Sample conditions*).

**2. Molecule file** — one row per molecule (or let LipidCruncher generate it)

| Column | Content |
|--------|---------|
| `Molecule` | Name — must match the **prefix** of the measurement IDs |
| `MS ion or MS/MS product ion` | Full elemental formula of the measured ion, labelled atoms included, then `Lab` + the labelled element + how many of its atoms can carry the label (e.g. `C42H78O10P1LabC6` for BMP 18:1_18:1 as [M-H]⁻) |
| `MS/MS neutral loss` | Neutral-loss formula for MS/MS data (leave blank for MS) |

**3. Element file** — one row per element (or let LipidCruncher generate it)

| Column | Content |
|--------|---------|
| `Element` | Symbol (`C`, `H`, `N`, `O`, `P`, …) |
| `Isotope abundance_Mass shift` | Natural abundances, e.g. `0.0107_1/0.9893_0` |
| `Tracer isotope mass shift` | Mass shift of the tracer (blank for non-tracer elements) |
| `Tracer purity` | Label purity, e.g. `0.99` (blank for non-tracer elements) |

💡 **No files?** Click **Load sample data** for a bundled example (a ¹³C HepG2 tracing dataset).
"""

_ABOUT_CORRECTION_DOC = """
### 📖 About Natural Isotope Abundance Correction

Stable isotope tracing feeds cells a nutrient labeled with a heavy but **stable**
(non-radioactive) isotope — e.g. ¹³C — and measures how much label ends up in each
lipid. The catch: nature already contains a little heavy isotope (~1.1% of all
carbon is ¹³C), so part of every "heavy" signal is background, not your tracer.

This module runs the real **IsoCorrectoR** package to remove that natural-abundance
background, leaving the signal that reflects **only the label you fed in**.

**Settings**
- **Correct tracer impurity** — accounts for the label reagent not being 100% pure.
- **Correct tracer element in core molecule** — corrects natural heavy atoms of the
  tracer element elsewhere in the molecule.
- **Calculate mean enrichment** — also reports the average degree of labeling per molecule.
- **Ultra-high-resolution mode** — ON for high-resolution instruments (Orbitrap,
  FT-ICR) whose IDs label the specific element (`_C0`); OFF for standard resolution
  (`_0`). **The ID format must match this setting.**
- **Also output monoisotopic correction** — adds IsoCorrectoR's monoisotopic
  correction results as two extra tables (Corrected Monoisotopic and its
  fractions); the main correction is unchanged.
- **UHR calculation limit** — advanced; the default is appropriate for most data.

**Outputs** — an isotopologue distribution plot (each sample's corrected fractions as a
stacked bar, grouped by condition), then corrected intensities, corrected fractions, mean
enrichment, residuals, and raw data (plus the monoisotopic tables when asked for), each
downloadable as CSV.
"""


_OUTPUTS_DOC = """
### 📊 What Each Output Table Means

| Table | What it contains |
|-------|------------------|
| **Corrected** | Isotopologue intensities after natural-abundance correction (and tracer-impurity correction, if enabled) — the numbers to carry into downstream quantification. |
| **Corrected Fractions** | The same corrected values as fractions of each molecule's total signal in that sample (they sum to 1 per molecule per sample) — the labeling *pattern*, independent of how much lipid was there. |
| **Mean Enrichment** | The average fraction of tracer-element positions carrying the label, per molecule per sample: 0 = unlabeled, 1 = fully labeled. One number summarizing each labeling pattern. |
| **Raw Data** | Your measured intensities as IsoCorrectoR read them — use it to confirm the files were parsed the way you expected. |
| **Residuals** | Measured minus back-calculated intensities: the part of the signal the correction model could not explain. |
| **Relative Residuals** | The same discrepancy relative to the measured signal — the scale-free version, and the one worth reading. |
| **Corrected Monoisotopic** / **… Fractions** | Only with *Also output monoisotopic correction*: IsoCorrectoR's monoisotopic correction results, as intensities and as fractions. |

**What to look for**

- **Relative residuals near zero.** Large values flag a mismatch between your data
  and the model: a wrong elemental formula in the Molecule file, a missing
  isotopologue row, an isobaric co-elution, or a peak too small to correct.
- **Corrected values that go negative.** Correction *subtracts* the
  natural-abundance contribution, so a noisy low-intensity isotopologue can dip
  slightly below zero — a few near-zero values are normal. Systematically
  negative values point to the same causes as large residuals.
- **Mean enrichment between 0 and 1**, and consistent across replicates of a
  condition. Replicates that disagree are a labeling or sample-prep problem, not
  a correction problem.
- **Corrected vs. raw.** Heavy isotopologues should shrink once their
  natural-abundance share is removed. If nothing changed, check that the tracer
  element's mass shift and purity are actually filled in in the Element file.
"""


def _display_docs() -> None:
    """Render the how-to, input-requirements and about-correction expanders."""
    with st.expander("🚀 How to Run a Correction", expanded=False):
        st.markdown(_HOW_TO_RUN_DOC)
    with st.expander("📋 Input File Requirements", expanded=False):
        st.markdown(_INPUT_REQUIREMENTS_DOC)
    with st.expander("📖 About Isotope Correction", expanded=False):
        st.markdown(_ABOUT_CORRECTION_DOC)

# (CorrectionResult attribute, display label) for the output tables, corrected first.
_OUTPUT_TABLES = [
    ("corrected", "Corrected"),
    ("corrected_fractions", "Corrected Fractions"),
    ("mean_enrichment", "Mean Enrichment"),
    ("raw_data", "Raw Data"),
    ("residuals", "Residuals"),
    ("relative_residuals", "Relative Residuals"),
    ("corrected_monoisotopic", "Corrected Monoisotopic"),
    ("corrected_monoisotopic_fractions", "Corrected Monoisotopic Fractions"),
]


# Where the Molecule and Element files come from (the sidebar radio).
_UPLOAD = "Upload my own"
_GENERATE = "Generate from the Measurement file"

_BUILDER_DOC = """
One row per lipid in the Measurement file. Check each row; the files are
written from it.

- **Formula** — the neutral lipid's full formula, worked out from its class
  and chains. Labelled atoms stay in it: BMP 22:6_22:6 is C50, not 50 − 6.
  Blank when the class isn't known (or the name has modifications): type it in.
- **Ion** — the ion you measured; the Molecule file holds the ion's formula.
- **Labelled atoms** — how many atoms of the labelled element can carry the
  label, which depends on your tracer and labelling approach. The suggestion
  is the heaviest isotopologue in the Measurement file.

The Element file uses natural isotope abundances plus the labelled element and
its purity chosen here.
"""


def _go_home() -> None:
    """Clear isotope state and return to the landing page.

    Runs as an on_click callback — Streamlit reruns automatically afterward,
    so no explicit st.rerun() (which is illegal inside a callback).
    """
    StreamlitAdapter.clear_stage('isotope')
    st.session_state.page = PAGE_LANDING


# Setting widget key -> default value (defaults match IsoCorrectoR, except
# tracer-impurity which we default on). Used to seed the widgets so the sample
# loader can override them cleanly.
_SETTING_DEFAULTS = {
    "iso_correct_tracer_impurity": IsotopeCorrectionConfig().correct_tracer_impurity,
    "iso_correct_tracer_element_core": IsotopeCorrectionConfig().correct_tracer_element_core,
    "iso_calculate_mean_enrichment": IsotopeCorrectionConfig().calculate_mean_enrichment,
    "iso_ultra_high_res": IsotopeCorrectionConfig().ultra_high_res,
    "iso_correct_also_monoisotopic": IsotopeCorrectionConfig().correct_also_monoisotopic,
    "iso_calculation_threshold_uhr": IsotopeCorrectionConfig().calculation_threshold_uhr,
}


def _load_sample_data() -> None:
    """Load the bundled golden input files (and their matching settings).

    The sample is an ultra-high-resolution dataset (measurement IDs like
    ``BMP_18:1_18:1_C0``), so UHR mode must be on for IsoCorrectoR to parse it.
    Seeds the settings to the configuration that produced the golden output.
    """
    st.session_state.iso_measurement_df = pd.read_csv(_SAMPLE_DIR / "MeasurementFile.csv")
    st.session_state.iso_molecule_df = pd.read_csv(_SAMPLE_DIR / "MoleculeFile.csv")
    st.session_state.iso_element_df = pd.read_csv(_SAMPLE_DIR / "ElementFile.csv")
    for key, default in _SETTING_DEFAULTS.items():
        st.session_state[key] = default
    st.session_state.iso_ultra_high_res = True


def _resolve_input(uploaded, session_key: str) -> Optional[pd.DataFrame]:
    """An uploaded file takes precedence; otherwise the sample data, if loaded."""
    if uploaded is not None:
        return pd.read_csv(uploaded)
    return st.session_state.get(session_key)


def _config_from_widgets() -> IsotopeCorrectionConfig:
    """Build the correction config from the sidebar setting widgets."""
    return IsotopeCorrectionConfig(
        correct_tracer_impurity=st.session_state.iso_correct_tracer_impurity,
        correct_tracer_element_core=st.session_state.iso_correct_tracer_element_core,
        calculate_mean_enrichment=st.session_state.iso_calculate_mean_enrichment,
        ultra_high_res=st.session_state.iso_ultra_high_res,
        correct_also_monoisotopic=st.session_state.iso_correct_also_monoisotopic,
        calculation_threshold_uhr=st.session_state.iso_calculation_threshold_uhr,
    )


def _inputs_key(measurement_df, molecule_df, element_df, config) -> Optional[str]:
    """Fingerprint of the three inputs and the settings, to tell whether the
    result on screen still belongs to what the sidebar shows."""
    if measurement_df is None or molecule_df is None or element_df is None:
        return None
    digest = hashlib.sha256(config.model_dump_json().encode())
    for df in (measurement_df, molecule_df, element_df):
        digest.update(repr(list(df.columns)).encode())
        digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()


def _download_table(df: pd.DataFrame, label: str, key: str) -> None:
    """Download a result table as CSV, preserving the isotopologue-ID index."""
    st.download_button(
        label=f"Download {label} CSV",
        data=df.to_csv(index=True).encode("utf-8"),
        file_name=f"IsoCorrectoR_{label.replace(' ', '')}.csv",
        mime="text/csv",
        key=key,
    )


def _display_inputs():
    """Render the top of the sidebar: sample data and the input files.

    Returns (measurement_df, molecule_df, element_df, generate): with
    ``generate`` the Molecule and Element files come from the page's
    builder instead, and the two frames returned here are None.
    """
    # Seed setting widgets once so the sample loader / Run can read them by key.
    for key, default in _SETTING_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.markdown("### Stable Isotope Tracing")

        st.button("Load sample data", key="iso_load_sample", on_click=_load_sample_data,
                  width='stretch', help="Load the bundled IsoCorrectoR example dataset.")

        measurement_up = st.file_uploader("Measurement file (CSV)", type="csv", key="iso_measurement_uploader")
        measurement_df = _resolve_input(measurement_up, "iso_measurement_df")

        source = st.radio(
            "Molecule and Element files", [_UPLOAD, _GENERATE], key="iso_input_source",
            help="LipidCruncher can write both files from the lipids in your "
                 "Measurement file; you check them on the page.",
        )
        if source == _GENERATE:
            return measurement_df, None, None, True

        molecule_up = st.file_uploader("Molecule file (CSV)", type="csv", key="iso_molecule_uploader")
        element_up = st.file_uploader("Element file (CSV)", type="csv", key="iso_element_uploader")
        molecule_df = _resolve_input(molecule_up, "iso_molecule_df")
        element_df = _resolve_input(element_up, "iso_element_df")

    return measurement_df, molecule_df, element_df, False


def _display_run(r_available: bool, have_all: bool, generate: bool) -> bool:
    """Render the rest of the sidebar (settings, Run, Back); True when Run
    was clicked."""
    with st.sidebar:
        with st.expander("Correction settings", expanded=False):
            st.checkbox("Correct tracer impurity", key="iso_correct_tracer_impurity")
            st.checkbox("Correct tracer element in core molecule", key="iso_correct_tracer_element_core")
            st.checkbox("Calculate mean enrichment", key="iso_calculate_mean_enrichment")
            st.checkbox("Ultra-high-resolution mode", key="iso_ultra_high_res")
            st.checkbox("Also output monoisotopic correction", key="iso_correct_also_monoisotopic")
            st.number_input("UHR calculation limit", min_value=0, step=1,
                            key="iso_calculation_threshold_uhr",
                            help="Limit value used by the ultra-high-resolution calculation.")

        run_clicked = st.button(
            "Run correction", type="primary", width='stretch',
            disabled=not (have_all and r_available),
            key="iso_run",
        )
        if not r_available:
            st.caption("Run is disabled: R runtime unavailable.")
        elif not have_all and generate:
            st.caption("Upload the Measurement file (or load sample data) and "
                       "complete the Molecule and Element files on the page to enable.")
        elif not have_all:
            st.caption("Upload all three files (or load sample data) to enable.")

        st.markdown("---")
        st.button("← Back to Home", key="iso_back_home", on_click=_go_home, width='stretch')

    return run_clicked


def _download_input(df: pd.DataFrame, file_name: str, key: str) -> None:
    """Download a generated input file in IsoCorrectoR's format."""
    st.download_button(
        label=f"Download {file_name}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name, mime="text/csv", key=key,
    )


def _display_input_builder(measurement_df: Optional[pd.DataFrame]):
    """Generate the Molecule and Element files from the Measurement file's
    lipids, after the user checks them. Returns (molecule_df, element_df),
    both None until every lipid is complete."""
    st.markdown("#### Molecule and Element files")
    if measurement_df is None:
        st.info("Upload the Measurement file (or load sample data) to generate them.")
        return None, None
    st.markdown(_BUILDER_DOC)

    untagged = IsotopeTracingWorkflow.untagged_ids(measurement_df)
    if untagged:
        shown = ", ".join(f"`{i}`" for i in untagged[:3])
        more = f" and {len(untagged) - 3} more" if len(untagged) > 3 else ""
        st.error(
            f"{len(untagged)} ID{'s' if len(untagged) != 1 else ''} in the Measurement "
            f"file {'do' if len(untagged) != 1 else 'does'} not end in an isotopologue "
            f"number: {shown}{more}. Each row is one isotopologue, so its ID is the "
            "lipid plus `_0`, `_1`, … (or `_C0`, `_C1`, … in ultra-high resolution), "
            "e.g. `BMP_22:6_22:6_0`."
        )
        return None, None

    table = IsotopeTracingWorkflow.molecule_table(measurement_df)
    # One table and tracer per lipid set, so a new dataset never inherits old edits.
    digest = hashlib.sha256("\x00".join(table["Molecule"]).encode()).hexdigest()[:12]
    tracer_key = f"iso_build_tracer_{digest}"
    if tracer_key not in st.session_state:
        st.session_state[tracer_key] = IsotopeTracingWorkflow.default_tracer(measurement_df)

    tracers = IsotopeTracingWorkflow.TRACERS
    left, right = st.columns(2)
    tracer = left.selectbox(
        "Labelled element", list(tracers), key=tracer_key,
        format_func=lambda e: f"{tracers[e]} ({e})",
    )
    purity = right.number_input(
        "Tracer purity", min_value=0.0, max_value=1.0, step=0.001, format="%.3f",
        value=0.99, key="iso_tracer_purity",
        help="Isotopic purity of the labelled nutrient, e.g. 0.99 for 99% ¹³C.",
    )
    edited = st.data_editor(
        table, key=f"iso_build_table_{digest}", disabled=["Molecule"],
        hide_index=True, width='stretch',
        column_config={
            "Formula": st.column_config.TextColumn(
                "Formula (neutral lipid)",
                help="Every atom of the neutral lipid, labelled ones included; "
                     "the ion form is added to it. Blank when LipidCruncher "
                     "doesn't know the lipid class: type it in.",
            ),
            "Ion": st.column_config.SelectboxColumn(
                "Ion", options=IsotopeTracingWorkflow.ION_FORMS, required=True,
                help="The ion you measured. The Molecule file holds the ion's formula.",
            ),
            "Labelled atoms": st.column_config.NumberColumn(
                f"Labelled {tracer} atoms", min_value=1, step=1,
                help="How many atoms can carry the label (the 6 in LabC6). It "
                     "depends on the tracer and the labelling approach. "
                     "Suggested from the heaviest isotopologue measured.",
            ),
        },
    )

    molecule_df, element_df, problems = IsotopeTracingWorkflow.build_input_files(
        edited, tracer, purity,
    )
    if problems:
        st.warning("Complete the table to generate the files:\n\n"
                   + "\n".join(f"- {p}" for p in problems))
        return None, None

    with st.expander("Generated files", expanded=False):
        st.markdown("**Molecule file**")
        st.dataframe(molecule_df, hide_index=True, width='stretch')
        _download_input(molecule_df, "MoleculeFile.csv", "iso_dl_molecule_file")
        st.markdown("**Element file**")
        # Blank, not "None", for the non-tracer elements, as in the CSV.
        st.dataframe(element_df.astype(object).where(element_df.notna(), ""),
                     hide_index=True, width='stretch')
        _download_input(element_df, "ElementFile.csv", "iso_dl_element_file")
    return molecule_df, element_df


def _display_results(result) -> None:
    """Render validation errors / R errors / the output tables and log."""
    if result is None:
        return

    if not result.success:
        if result.validation_errors:
            st.error("The input files could not be validated:")
            for err in result.validation_errors:
                st.markdown(f"- {err}")
        else:
            st.error(result.error_message or "Correction failed.")
        return

    correction = result.correction
    if correction.warnings:
        st.warning(
            "Correction complete, but IsoCorrectoR warns the results may be "
            "less accurate:\n\n"
            + "\n".join(f"- {w}" for w in correction.warnings)
        )
    else:
        st.success("Correction complete.")

    # How to read the tables, next to the tables themselves.
    with st.expander("📖 How to Read These Tables", expanded=False):
        st.markdown(_OUTPUTS_DOC)

    if correction.corrected_fractions is not None:
        _display_distribution_plot(correction.corrected_fractions)

    # Primary output: the corrected intensities.
    st.markdown("#### Corrected intensities")
    st.dataframe(correction.corrected, width='stretch')
    _download_table(correction.corrected, "Corrected", key="iso_dl_corrected_main")

    # All other IsoCorrectoR outputs.
    with st.expander("All IsoCorrectoR outputs", expanded=False):
        for attr, label in _OUTPUT_TABLES:
            table = getattr(correction, attr)
            if table is None:
                continue
            st.markdown(f"**{label}**")
            st.dataframe(table, width='stretch')
            _download_table(table, label, key=f"iso_dl_{attr}")
            st.markdown("---")

    if correction.log_text:
        with st.expander("IsoCorrectoR run log", expanded=False):
            st.code(correction.log_text)


def _keep_valid(key: str, options: list, default: Optional[list] = None) -> None:
    """Drop a stored selection that a new result no longer offers, so the
    widget falls back to its default instead of failing. A multiselect's
    default is seeded here rather than passed to the widget, which
    Streamlit warns about once its key has been set."""
    value = st.session_state.get(key)
    if isinstance(value, list):
        kept = [v for v in value if v in options]
        if kept or not value:  # an empty selection is the user's choice
            st.session_state[key] = kept
        else:
            st.session_state[key] = default
    elif value is None:
        if default is not None:
            st.session_state[key] = default
    elif value not in options:
        del st.session_state[key]


def _sync_conditions(options: list) -> None:
    """Keep the condition picker in step with the conditions on offer:
    drop ones that are gone, and add new ones (a condition renamed in the
    table, or a new dataset's) at the end."""
    previous = st.session_state.get("iso_plot_condition_options")
    st.session_state.iso_plot_condition_options = options
    value = st.session_state.get("iso_plot_conditions")
    if value is None or previous is None:
        _keep_valid("iso_plot_conditions", options, default=options)
        return
    kept = [v for v in value if v in options]
    kept += [o for o in options if o not in previous and o not in kept]
    st.session_state.iso_plot_conditions = kept


def _sample_groups(samples: list) -> dict:
    """Condition -> samples: read from the names, then editable by hand."""
    auto = IsotopeTracingWorkflow.sample_groups(samples)
    table = pd.DataFrame(
        [(s, c) for c, members in auto.items() for s in members],
        columns=["Sample", "Condition"],
    )
    # One table per sample set, so a new dataset never inherits old edits.
    digest = hashlib.sha256("\x00".join(samples).encode()).hexdigest()[:12]
    with st.expander("Sample conditions", expanded=False):
        st.caption(
            "Read from the sample names. Type a condition to regroup a "
            "sample; give samples the same condition to make them replicates."
        )
        edited = st.data_editor(
            table, key=f"iso_plot_groups_{digest}", disabled=["Sample"],
            hide_index=True, width='stretch',
        )
    return IsotopeTracingWorkflow.groups_from_table(
        edited["Sample"].tolist(), edited["Condition"].tolist(),
    )


def _display_distribution_plot(fractions: pd.DataFrame) -> None:
    """Stacked bars of a molecule's corrected fractions, by condition."""
    st.markdown("#### Isotopologue distribution")
    st.markdown(
        "Each bar is one sample's corrected fractions, stacked from the "
        "unlabelled form (_0) up; bars are grouped by condition. Conditions "
        "are read from the sample names: a trailing `_1`, `_2`, `_3` marks "
        "replicates of the same condition. Regroup samples under **Sample "
        "conditions**."
    )

    molecules = IsotopeTracingWorkflow.molecules(fractions)
    groups = _sample_groups(list(fractions.columns))
    labels = IsotopeTracingWorkflow.isotopologue_labels(fractions)
    _keep_valid("iso_plot_molecule", molecules)
    _sync_conditions(list(groups))
    _keep_valid("iso_plot_isotopologues", labels, default=labels)

    molecule = st.selectbox("Lipid", molecules, key="iso_plot_molecule")
    conditions = st.multiselect(
        "Conditions, in plot order", list(groups), key="iso_plot_conditions",
        help="Remove conditions to hide them; re-add one to move it to the end.",
    )
    chosen_labels = st.multiselect(
        "Isotopologues", labels, key="iso_plot_isotopologues",
        help=(
            "Leave some out to focus on others. The bars then show only the "
            "chosen fractions and no longer reach 1, unless you rescale them."
        ),
    )
    rescale = st.checkbox(
        "Rescale bars to 100% of the shown isotopologues",
        key="iso_plot_rescale",
        help=(
            "Each bar becomes the shown isotopologues' share of their own "
            "total, e.g. how the label is spread over +1 to +6. Where the "
            "shown fractions are tiny (unlabelled controls), rescaling "
            "magnifies noise."
        ),
    )
    st.caption(" · ".join(
        f"{c}: {len(groups[c])} replicate{'s' if len(groups[c]) != 1 else ''}"
        for c in conditions
    ))
    if not conditions or not chosen_labels:
        st.warning("Select at least one condition and one isotopologue.")
        return

    fig, data = IsotopeTracingWorkflow.distribution_plot(
        fractions, molecule, {c: groups[c] for c in conditions}, chosen_labels, rescale,
    )
    st.plotly_chart(fig, width='stretch')
    stem = f"{molecule}_isotopologue_distribution".replace(':', '-')
    display_export_buttons(
        fig, data, f"{stem}.svg", f"{stem}.csv",
        "iso_dl_plot_svg", "iso_dl_plot_csv",
    )


def display_isotope_tracing_page() -> None:
    """Display the stable isotope tracing page."""
    _, center, _ = st.columns([1, 3, 1])

    r_available = StreamlitAdapter.isocorrector_available()

    measurement_df, molecule_df, element_df, generate = _display_inputs()

    with center:
        display_logo(centered=True)
        st.markdown(
            "Run natural-isotope-abundance correction on stable isotope tracing data "
            "using the IsoCorrectoR package. Upload the **Measurement**, **Molecule**, "
            "and **Element** files in the sidebar, or load the sample dataset."
        )

        _display_docs()

        if not r_available:
            st.warning(
                "**R runtime unavailable.** The stable isotope tracing module needs R "
                "with the IsoCorrectoR package installed. The rest of LipidCruncher is "
                "unaffected — use **← Back to Home** to return to general lipidomics."
            )

        if generate:
            molecule_df, element_df = _display_input_builder(measurement_df)
        have_all = measurement_df is not None and molecule_df is not None and element_df is not None
        run_clicked = _display_run(r_available, have_all, generate)

        config = _config_from_widgets()
        current_key = _inputs_key(measurement_df, molecule_df, element_df, config)
        if run_clicked:
            st.session_state.iso_result = StreamlitAdapter.run_isotope_correction(
                measurement_df, molecule_df, element_df, config
            )
            st.session_state.iso_result_key = current_key

        result = st.session_state.get("iso_result")
        if result is None:
            st.info("Provide the three input files and click **Run correction** in the sidebar.")
        elif st.session_state.get("iso_result_key") != current_key:
            # A new upload, the sample data or a changed setting: the last
            # result (and its downloads) no longer match the sidebar.
            st.info(
                "The input files or settings have changed since the last run. "
                "Click **Run correction** to update the results."
            )
        else:
            _display_results(result)
