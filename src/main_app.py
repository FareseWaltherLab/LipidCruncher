"""
LipidCruncher - Lipidomics Data Analysis Application

Refactored architecture:
    UI Layer (this file)
        → Workflows (app/workflows/)
            → Adapters (app/adapters/)
                → Services (app/services/)
                    → Models (app/models/)

Reference: old_main_app.py contains the original monolithic implementation.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="LipidCruncher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = Path(__file__).parent
SAMPLE_DATA_DIR = BASE_DIR.parent / "sample_datasets"

# =============================================================================
# Imports - Refactored Components
# =============================================================================

from app.adapters.streamlit_adapter import StreamlitAdapter, SessionState
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.services.format_detection import FormatDetectionService, DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.services.zero_filtering import ZeroFilterConfig
from app.workflows.data_ingestion import DataIngestionWorkflow, IngestionConfig
from app.workflows.normalization import NormalizationWorkflow, NormalizationWorkflowConfig, NormalizationWorkflowResult
from app.ui.landing_page import display_landing_page, display_logo
from app.ui.format_requirements import display_format_requirements
from app.ui.zero_filtering import display_zero_filtering_config

# Legacy modules for UI compatibility (GroupSamples, DataFormatHandler)
from lipidomics.group_samples import GroupSamples
from lipidomics.data_format_handler import DataFormatHandler


# =============================================================================
# Streamlit Compatibility
# =============================================================================

def safe_rerun():
    """Rerun the app, compatible with both old and new Streamlit versions."""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()


# =============================================================================
# Session State Initialization
# =============================================================================

StreamlitAdapter.initialize_session_state()


# =============================================================================
# UI Components - Format Selection & File Upload
# =============================================================================

def display_format_selection() -> str:
    """Display format selection dropdown in sidebar."""
    return st.sidebar.selectbox(
        'Select Data Format',
        ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0', 'MS-DIAL']
    )


def get_sample_data_info(data_format: str) -> dict:
    """Get sample dataset info including file path and description."""
    sample_info = {
        'Generic Format': {
            'file': 'generic_test_dataset.csv',
            'description': """ADGAT-DKO case study (normalized): inguinal white adipose tissue, WT vs ADGAT-DKO.

**Sample order:**
1. WT (s1–s4, n=4)
2. ADGAT-DKO (s5–s8, n=4)
3. BQC (s9–s12, n=4)"""
        },
        'LipidSearch 5.0': {
            'file': 'lipidsearch5_test_dataset.csv',
            'description': """ADGAT-DKO case study (raw): inguinal white adipose tissue, WT vs ADGAT-DKO. Includes quality grades and retention times.

**Sample order:**
1. WT (s1–s4, n=4)
2. ADGAT-DKO (s5–s8, n=4)
3. BQC (s9–s12, n=4)"""
        },
        'MS-DIAL': {
            'file': 'msdial_test_dataset.csv',
            'description': """Mouse adrenal gland lipidomics: fads2 knockout vs wild-type.

**Sample order:**
1. Blank (n=1)
2. fads2 KO (n=3)
3. Wild-type (n=3)"""
        },
        'Metabolomics Workbench': {
            'file': 'mw_test_dataset.csv',
            'description': """Mouse serum HFD study: 2×2 factorial (Normal/HFD × Water/DCA).

**Sample order:**
1. Normal+Water (S1A–S11A, n=11)
2. Normal+DCA (S1B–S11B, n=11)
3. HFD+Water (S1C–S11C, n=11)
4. HFD+DCA (S1D–S11D, n=11)
5. Blank (n=2)
6. TQC (n=12)"""
        },
    }
    return sample_info.get(data_format)


def load_sample_dataset(data_format: str) -> pd.DataFrame:
    """Load a sample dataset for the selected format."""
    info = get_sample_data_info(data_format)
    if info:
        filepath = SAMPLE_DATA_DIR / info['file']
        if filepath.exists():
            st.session_state.sample_data_file = info['file']

            if data_format == 'Metabolomics Workbench':
                # Metabolomics Workbench needs raw text for parsing special markers
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                # Process through handler which returns standardized DataFrame
                standardized_df, success, message = DataFormatHandler.validate_and_preprocess(
                    text_content, 'Metabolomics Workbench'
                )
                if success:
                    # Store as already standardized
                    st.session_state.standardized_df = standardized_df
                    return standardized_df
                else:
                    st.sidebar.error(message)
                    return None
            else:
                return pd.read_csv(filepath)
    return None


def display_file_upload(data_format: str) -> pd.DataFrame:
    """Display file upload widget and sample data option."""

    # Sample data option
    with st.sidebar.expander("🧪 Try Sample Data", expanded=False):
        info = get_sample_data_info(data_format)
        if info:
            st.markdown(f"**{data_format} Example:**")
            st.markdown(info['description'])
            st.markdown("---")
        if st.button("Load Sample Data", key="load_sample"):
            sample_df = load_sample_dataset(data_format)
            if sample_df is not None:
                st.session_state.using_sample_data = True
                st.session_state.raw_df = sample_df
                safe_rerun()

    # Check if using sample data
    if st.session_state.get('using_sample_data') and st.session_state.get('raw_df') is not None:
        sample_file = st.session_state.get('sample_data_file', 'sample data')
        st.sidebar.info(f"📁 Using sample: {sample_file}")
        if st.sidebar.button("Clear & Upload Your Data"):
            st.session_state.using_sample_data = False
            st.session_state.sample_data_file = None
            StreamlitAdapter.reset_data_state()
            safe_rerun()
        return st.session_state.raw_df

    # File upload
    file_types = ['csv'] if data_format == 'Metabolomics Workbench' else ['csv', 'txt']
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset',
        type=file_types,
        help="Limit 800MB per file"
    )

    if uploaded_file is not None:
        try:
            if data_format == 'Metabolomics Workbench':
                # Metabolomics Workbench needs raw text for parsing special markers
                text_content = uploaded_file.getvalue().decode('utf-8')
                # Process through handler which returns standardized DataFrame
                standardized_df, success, message = DataFormatHandler.validate_and_preprocess(
                    text_content, 'Metabolomics Workbench'
                )
                if success:
                    st.sidebar.success("File uploaded and processed successfully!")
                    st.session_state.raw_df = standardized_df
                    st.session_state.standardized_df = standardized_df
                    return standardized_df
                else:
                    st.sidebar.error(message)
                    return None
            else:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
                st.session_state.raw_df = df
                return df
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None

    return None


# =============================================================================
# UI Components - Column Standardization
# =============================================================================

def standardize_uploaded_data(df: pd.DataFrame, data_format: str) -> pd.DataFrame:
    """
    Standardize uploaded data and create column mapping.
    Returns the standardized DataFrame.
    """
    # Metabolomics Workbench is already standardized during file loading
    # (it requires raw text parsing, done in load_sample_dataset/display_file_upload)
    if data_format == 'Metabolomics Workbench':
        st.session_state.format_type = data_format
        return df

    format_map = {
        'LipidSearch 5.0': 'lipidsearch',
        'MS-DIAL': 'msdial',
        'Generic Format': 'generic',
    }
    internal_format = format_map.get(data_format, 'generic')

    # Use DataFormatHandler to validate and standardize
    standardized_df, success, message = DataFormatHandler.validate_and_preprocess(
        df, internal_format
    )

    if not success:
        st.sidebar.error(message)
        return None

    st.session_state.format_type = data_format
    return standardized_df


def display_column_mapping(df: pd.DataFrame, data_format: str) -> tuple:
    """
    Display column mapping in the sidebar.
    For MS-DIAL, includes override sample detection expander.

    Returns:
        tuple: (success: bool, modified_df: pd.DataFrame or None)
    """
    if st.session_state.get('column_mapping') is None:
        return True, None

    st.sidebar.subheader("Column Name Standardization")

    # Display mapping table
    mapping_df = st.session_state.column_mapping.copy()
    st.sidebar.dataframe(
        mapping_df.reset_index(drop=True),
        use_container_width=True
    )

    # MS-DIAL: Optional override for sample column detection
    if data_format == 'MS-DIAL':
        with st.sidebar.expander("🔧 Override Sample Detection (Optional)", expanded=False):
            st.write("Only change this if auto-detection incorrectly classified columns.")

            features = st.session_state.get('msdial_features', {})
            detected_samples = features.get('raw_sample_columns', [])
            all_columns = features.get('actual_columns', [])

            # Exclude known metadata columns
            available_for_samples = [
                col for col in all_columns
                if col not in DataFormatHandler.MSDIAL_METADATA_COLUMNS
            ]

            manual_samples = st.multiselect(
                "Sample columns:",
                options=available_for_samples,
                default=detected_samples,
                key='manual_sample_override',
                help="Select all columns containing sample intensity data"
            )

            if manual_samples and manual_samples != detected_samples:
                # Update the feature list
                st.session_state.msdial_features['raw_sample_columns'] = manual_samples

                # Get current DataFrame and rebuild intensity columns
                current_df = df.copy()
                current_mapping = st.session_state.column_mapping

                # Build reverse lookup: standardized_name -> original_name
                std_to_orig = dict(zip(
                    current_mapping['standardized_name'],
                    current_mapping['original_name']
                ))

                # Identify which intensity columns to keep (by original name)
                intensity_cols_to_keep = []
                for col in current_df.columns:
                    if col.startswith('intensity['):
                        orig_name = std_to_orig.get(col)
                        if orig_name in manual_samples:
                            intensity_cols_to_keep.append(col)

                # Get non-intensity columns (metadata)
                non_intensity_cols = [col for col in current_df.columns if not col.startswith('intensity[')]

                # Build new DataFrame with only selected intensity columns
                new_df = current_df[non_intensity_cols + intensity_cols_to_keep].copy()

                # Rename intensity columns to be sequential
                rename_map = {}
                new_mapping_rows = []

                # Keep metadata mappings
                for _, row in current_mapping.iterrows():
                    if not row['standardized_name'].startswith('intensity['):
                        new_mapping_rows.append({
                            'standardized_name': row['standardized_name'],
                            'original_name': row['original_name']
                        })

                # Create new sequential intensity column names
                for i, old_col in enumerate(intensity_cols_to_keep, 1):
                    new_col = f'intensity[s{i}]'
                    rename_map[old_col] = new_col
                    orig_name = std_to_orig.get(old_col, old_col)
                    new_mapping_rows.append({
                        'standardized_name': new_col,
                        'original_name': orig_name
                    })

                new_df = new_df.rename(columns=rename_map)

                # Update session state
                st.session_state.n_intensity_cols = len(intensity_cols_to_keep)
                st.session_state.column_mapping = pd.DataFrame(new_mapping_rows)

                # Update sample name mapping
                st.session_state.msdial_sample_names = {
                    f's{i}': orig for i, orig in enumerate(manual_samples, 1)
                }

                st.success(f"✓ Using {len(manual_samples)} manually selected samples")

                # Display updated mapping table
                st.write("**Updated column mapping:**")
                st.dataframe(
                    st.session_state.column_mapping.reset_index(drop=True),
                    use_container_width=True
                )

                return True, new_df

    return True, None


# =============================================================================
# UI Components - Sample Grouping
# =============================================================================

def detect_sample_columns(df: pd.DataFrame, data_format: str) -> list:
    """Detect intensity/sample columns based on format.

    After standardization, all formats use intensity[...] columns for sample data.
    """
    # After standardization, all formats use intensity[...] columns
    return [col for col in df.columns if col.startswith('intensity[')]


def extract_sample_names(columns: list, data_format: str) -> list:
    """Extract clean sample names from column names.

    After standardization, all formats use intensity[sample_name] pattern.
    """
    names = []
    for col in columns:
        if col.startswith('intensity[') and col.endswith(']'):
            # Extract name from intensity[name]
            names.append(col[10:-1])  # Remove 'intensity[' and ']'
        else:
            names.append(col)
    return names


def display_experiment_definition(df: pd.DataFrame, data_format: str, sample_cols: list) -> tuple:
    """
    Display experiment definition UI.
    Returns (n_conditions, conditions_list, number_of_samples_list) or (None, None, None) if invalid.
    """
    st.sidebar.subheader("Define Experiment")

    # Handle Metabolomics Workbench auto-detection
    if data_format == 'Metabolomics Workbench' and 'workbench_conditions' in st.session_state:
        conditions_in_order = [
            st.session_state.workbench_conditions[f's{i+1}']
            for i in range(len(st.session_state.workbench_samples))
        ]

        # Get unique conditions in order of first appearance
        unique_conditions = []
        seen = set()
        for condition in conditions_in_order:
            if condition not in seen:
                seen.add(condition)
                unique_conditions.append(condition)

        use_detected = st.sidebar.checkbox("Use detected experimental setup", value=True)

        if use_detected:
            n_conditions = len(unique_conditions)
            conditions_list = unique_conditions

            # Count samples per condition
            sample_counts = {}
            for condition in conditions_list:
                count = sum(1 for x in st.session_state.workbench_conditions.values() if x == condition)
                sample_counts[condition] = count

            number_of_samples_list = [sample_counts[condition] for condition in conditions_list]

            # Display the detected setup
            st.sidebar.write("Using detected setup:")
            for cond, count in zip(conditions_list, number_of_samples_list):
                st.sidebar.text(f"• {cond}: {count} samples")

            return n_conditions, conditions_list, number_of_samples_list

    # Manual experiment definition
    n_conditions = st.sidebar.number_input(
        'Number of conditions',
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    conditions_list = []
    number_of_samples_list = []

    for i in range(n_conditions):
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            cond_name = st.text_input(
                f'Condition {i + 1}',
                value=f'Condition_{i + 1}',
                key=f'cond_name_{i}'
            )
            conditions_list.append(cond_name)
        with col2:
            n_samples = st.number_input(
                'Samples',
                min_value=1,
                max_value=len(sample_cols),
                value=min(3, len(sample_cols)),
                key=f'n_samples_{i}'
            )
            number_of_samples_list.append(n_samples)

    # Validate all condition labels are non-empty
    if not all(cond and cond.strip() for cond in conditions_list):
        st.sidebar.error("All condition labels must be non-empty.")
        return None, None, None

    # Validate total samples
    total_assigned = sum(number_of_samples_list)
    if total_assigned != len(sample_cols):
        st.sidebar.warning(
            f"Assigned {total_assigned} samples but dataset has {len(sample_cols)}. "
            "Please adjust sample counts."
        )
        return None, None, None

    return n_conditions, conditions_list, number_of_samples_list


def display_group_samples(df: pd.DataFrame, experiment: ExperimentConfig, data_format: str) -> tuple:
    """
    Display group samples section with dataframe and manual regrouping option.
    Returns (group_df, updated_df).
    """
    st.sidebar.subheader('Group Samples')

    # Create a temporary experiment-like object for GroupSamples
    class TempExperiment:
        def __init__(self, config: ExperimentConfig):
            self.conditions_list = config.conditions_list
            self.number_of_samples_list = config.number_of_samples_list
            self.full_samples_list = config.full_samples_list
            self.extensive_conditions_list = config.extensive_conditions_list
            self.individual_samples_list = config.individual_samples_list

    temp_exp = TempExperiment(experiment)
    grouped_samples = GroupSamples(temp_exp, data_format)

    # Check dataset validity
    if not grouped_samples.check_dataset_validity(df):
        st.sidebar.error("Invalid dataset format!")
        return None, df

    # Build and display group_df
    group_df = grouped_samples.build_group_df(df)
    if group_df.empty:
        st.sidebar.error("Error building sample groups!")
        return None, df

    st.sidebar.dataframe(group_df, use_container_width=True)

    # Manual regrouping option
    st.sidebar.write('Are your samples properly grouped together?')
    grouping_correct = st.sidebar.radio('', ['Yes', 'No'], key='grouping_radio')

    if grouping_correct == 'No':
        # Store original column order if not already stored
        if st.session_state.get('original_column_order') is None:
            st.session_state.original_column_order = df.columns.tolist()

        st.session_state.grouping_complete = False
        selections = {}
        remaining_samples = group_df['sample name'].tolist()

        # Keep track of expected samples per condition
        expected_samples = dict(zip(experiment.conditions_list, experiment.number_of_samples_list))

        # Process each condition
        for condition in experiment.conditions_list:
            st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")

            selected_samples = st.sidebar.multiselect(
                f'Pick the samples that belong to condition {condition}',
                remaining_samples,
                key=f'select_{condition}'
            )

            selections[condition] = selected_samples

            # Update remaining samples only if correct number selected
            if len(selected_samples) == expected_samples[condition]:
                remaining_samples = [s for s in remaining_samples if s not in selected_samples]

        # Verify all conditions have correct number of samples
        all_correct = all(
            len(selections[condition]) == expected_samples[condition]
            for condition in experiment.conditions_list
        )

        if all_correct:
            try:
                # Update the group_df and get column mapping
                group_df, old_to_new = grouped_samples.group_samples(group_df, selections)

                # Reorder and rename columns in the DataFrame
                df_reordered = grouped_samples.reorder_intensity_columns(df, old_to_new)

                st.session_state.grouping_complete = True

                # Generate and display name_df to show the new sample order
                name_df = grouped_samples.update_sample_names(group_df)
                st.sidebar.write("New sample order after regrouping:")
                st.sidebar.dataframe(name_df, use_container_width=True)

                return group_df, df_reordered

            except ValueError as e:
                st.sidebar.error(f"Error updating groups: {str(e)}")
                st.session_state.grouping_complete = False
                return group_df, df
        else:
            st.session_state.grouping_complete = False
            return group_df, df
    else:
        st.session_state.grouping_complete = True
        # Restore original column order if it was stored
        if st.session_state.get('original_column_order') is not None:
            df = df.reindex(columns=st.session_state.original_column_order)

    return group_df, df


def display_bqc_section(experiment: ExperimentConfig) -> str:
    """
    Display BQC sample specification section.
    Returns the BQC label or None.
    """
    st.sidebar.subheader("Specify Label of BQC Samples")

    bqc_ans = st.sidebar.radio(
        'Do you have Batch Quality Control (BQC) samples?',
        ['Yes', 'No'],
        index=1,  # Default to 'No'
        key='bqc_radio'
    )

    bqc_label = None
    if bqc_ans == 'Yes':
        # Filter to conditions with 2+ samples
        conditions_with_two_plus = [
            condition for condition, n_samples
            in zip(experiment.conditions_list, experiment.number_of_samples_list)
            if n_samples > 1
        ]

        if conditions_with_two_plus:
            bqc_label = st.sidebar.radio(
                'Which label corresponds to BQC samples?',
                conditions_with_two_plus,
                index=0,
                key='bqc_label_radio'
            )
        else:
            st.sidebar.warning("No conditions with 2+ samples available for BQC.")

    return bqc_label


def display_confirm_inputs(experiment: ExperimentConfig) -> bool:
    """
    Display confirm inputs section with summary.
    Returns True if user confirms.
    """
    st.sidebar.subheader("Confirm Inputs")

    # Display total number of samples
    total_samples = sum(experiment.number_of_samples_list)
    st.sidebar.write(f"There are a total of {total_samples} samples.")

    # Display sample-condition pairings
    for i, condition in enumerate(experiment.conditions_list):
        if condition and condition.strip():
            samples = experiment.individual_samples_list[i]

            if len(samples) > 5:
                display_text = f"• {samples[0]} to {samples[-1]} (total {len(samples)}) correspond to {condition}"
            else:
                display_text = f"• {'-'.join(samples)} correspond to {condition}"

            # Use st.sidebar.text() to avoid markdown parsing of pipe characters
            st.sidebar.text(display_text)
        else:
            st.sidebar.error(f"Empty condition found at index {i}")

    # Confirmation checkbox
    return st.sidebar.checkbox("Confirm the inputs by checking this box", key='confirm_checkbox')


def display_sample_grouping(df: pd.DataFrame, data_format: str):
    """
    Display complete sample grouping UI in sidebar.
    Includes: experiment definition, group samples, BQC, and confirmation.
    """
    # Detect sample columns
    sample_cols = detect_sample_columns(df, data_format)
    sample_names = extract_sample_names(sample_cols, data_format)

    if not sample_cols:
        st.sidebar.error("No sample columns detected!")
        return None, None

    # Step 1: Define Experiment
    result = display_experiment_definition(df, data_format, sample_cols)
    if result[0] is None:
        return None, None

    n_conditions, conditions_list, number_of_samples_list = result

    # Create ExperimentConfig
    try:
        experiment = ExperimentConfig(
            n_conditions=n_conditions,
            conditions_list=conditions_list,
            number_of_samples_list=number_of_samples_list
        )
    except Exception as e:
        st.sidebar.error(f"Invalid experiment setup: {e}")
        return None, None

    # Step 2: Group Samples (with manual regrouping option)
    group_df, updated_df = display_group_samples(df, experiment, data_format)

    if group_df is None:
        return None, None

    # Check if grouping is complete
    if not st.session_state.get('grouping_complete', False):
        st.sidebar.error("Please complete sample grouping before proceeding.")
        return None, None

    # Step 3: BQC Section
    bqc_label = display_bqc_section(experiment)

    # Step 4: Confirm Inputs
    confirmed = display_confirm_inputs(experiment)

    if confirmed:
        st.session_state.experiment = experiment
        st.session_state.bqc_label = bqc_label
        st.session_state.confirmed = True
        st.session_state.standardized_df = updated_df
        return experiment, bqc_label
    else:
        # Checkbox unchecked - clear confirmed state
        st.session_state.confirmed = False
        return None, None


# =============================================================================
# UI Components - Data Processing Documentation
# =============================================================================

def display_data_processing_docs(data_format: str):
    """Display format-specific data processing documentation."""
    cleaning_docs = {
        'LipidSearch 5.0': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Column Standardization | Extract LipidMolec, ClassKey, CalcMass, BaseRt, TotalGrade, TotalSmpIDRate(%), FAKey, MeanArea columns |
| 2. Data Type Conversion | Convert MeanArea to numeric (non-numeric → 0) |
| 3. Lipid Name Standardization | Standardize to `Class(chains)` format |
| 4. Grade Filtering | Filter by quality grade (**configurable below**) |
| 5. Best Peak Selection | Keep entry with highest TotalSmpIDRate(%) per lipid |
| 6. Missing FA Keys | Remove rows without FAKey (except Ch class, deuterated standards) |
| 7. Duplicate Removal | Remove duplicates by LipidMolec |
| 8. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### ⚙️ Grade Filtering (Configurable)

LipidSearch assigns quality grades to each identification:

| Grade | Confidence | Default Action |
|-------|------------|----------------|
| A | Highest | ✓ Keep |
| B | Good | ✓ Keep |
| C | Lower | ✓ Keep for LPC/SM only |
| D | Lowest | ✗ Remove |

**→ Configure in "Configure Grade Filtering" section below.**
        """,

        'MS-DIAL': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Header Detection | Auto-detect data start row (skip metadata rows) |
| 2. Column Mapping | `Metabolite name` → LipidMolec, `Average Rt(min)` → BaseRt, `Average Mz` → CalcMass |
| 3. ClassKey Inference | Extract class from lipid name (e.g., `Cer(18:1;2O_24:0)` → `Cer`) |
| 4. Lipid Name Standardization | Standardize format, preserve hydroxyl notation (`;2O`, `;3O`) |
| 5. Quality Filtering | Filter by Total Score and/or MS/MS validation (**configurable below**) |
| 6. Data Type Selection | Choose raw or pre-normalized (if both available) |
| 7. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 8. Smart Deduplication | Keep entry with highest Total Score per lipid |
| 9. Internal Standards | Auto-detect: `(d5)`, `(d7)`, `(d9)`, `ISTD`, `SPLASH` patterns |
| 10. Duplicate Removal | Remove remaining duplicates by LipidMolec |
| 11. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### ⚙️ Quality Filtering (Configurable)

MS-DIAL provides quality metrics for filtering:

| Preset | Total Score | MS/MS Required | Use Case |
|--------|-------------|----------------|----------|
| Strict | ≥80 | Yes | Publication-ready |
| Moderate | ≥60 | No | Exploratory analysis |
| Permissive | ≥40 | No | Discovery |

**→ Configure in "Configure Quality Filtering" section below.**
        """,

        'Metabolomics Workbench': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Section Extraction | Extract data between `MS_METABOLITE_DATA_START` and `MS_METABOLITE_DATA_END` |
| 2. Header Processing | Row 1 → sample names, Row 2 → conditions |
| 3. Column Standardization | First column → LipidMolec, remaining → `intensity[s1]`, `intensity[s2]`, ... |
| 4. Lipid Name Standardization | Standardize to `Class(chains)` format |
| 5. ClassKey Extraction | Extract class from lipid name |
| 6. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 7. Conditions Storage | Store conditions for experiment setup suggestions |
| 8. Zero Filtering | Remove species failing zero threshold (**configurable below**) |
        """,

        'Generic Format': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Column Standardization | First column → LipidMolec, remaining → `intensity[s1]`, `intensity[s2]`, ... |
| 2. Lipid Name Standardization | Standardize to `Class(chains)` format, preserve hydroxyl notation |
| 3. ClassKey Extraction | Extract class from lipid name (e.g., `PC(16:0_18:1)` → `PC`) |
| 4. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 5. Invalid Lipid Removal | Remove empty names, single special characters |
| 6. Duplicate Removal | Remove duplicates by LipidMolec |
| 7. Zero Filtering | Remove species failing zero threshold (**configurable below**) |
        """
    }

    with st.expander("📖 About Data Standardization and Filtering", expanded=False):
        st.markdown(cleaning_docs.get(data_format, cleaning_docs['Generic Format']))

        # Zero filtering explanation (applies to all formats)
        st.markdown("---")
        st.markdown("""
#### 🔧 Zero Filtering (Configurable)

Removes lipid species with too many zero/below-detection values:

| Condition Type | Default Threshold | Action |
|----------------|-------------------|--------|
| BQC (if present) | ≥50% zeros | Remove species |
| All non-BQC conditions | ≥75% zeros each | Remove species |

*Thresholds are adjustable in "Configure Zero Filtering" section below.*
        """)


# =============================================================================
# UI Components - Format-Specific Filtering
# =============================================================================

def display_grade_filtering_config(df: pd.DataFrame) -> dict:
    """
    Display LipidSearch grade filtering configuration.

    Returns:
        dict: Grade config mapping class to acceptable grades, or None for defaults
    """
    # Check if the required columns exist
    if 'ClassKey' not in df.columns or 'TotalGrade' not in df.columns:
        return None

    # Get unique classes from the data
    all_classes = sorted(df['ClassKey'].dropna().unique())
    if not all_classes:
        return None

    with st.expander("⚙️ Configure Grade Filtering", expanded=False):
        # Initialize session state for persistence
        if 'grade_filter_mode' not in st.session_state:
            st.session_state.grade_filter_mode = "Use Default Settings"
        if 'grade_selections' not in st.session_state:
            st.session_state.grade_selections = {}

        options = ["Use Default Settings", "Customize by Class"]
        widget_key = "grade_filter_mode_radio"

        # Initialize widget key from persisted value BEFORE rendering
        persisted_value = st.session_state.get('grade_filter_mode', "Use Default Settings")
        if persisted_value in options:
            st.session_state[widget_key] = persisted_value

        def on_grade_mode_change():
            st.session_state.grade_filter_mode = st.session_state[widget_key]

        use_custom = st.radio(
            "Grade filtering mode:",
            options,
            horizontal=True,
            key=widget_key,
            on_change=on_grade_mode_change
        )
        st.session_state.grade_filter_mode = use_custom

        if use_custom == "Use Default Settings":
            st.success("✓ Default: A/B for all classes, plus C for LPC and SM.")
            return None

        # Custom settings
        st.markdown("---")
        grade_config = {}
        cols = st.columns(3)

        for idx, lipid_class in enumerate(all_classes):
            with cols[idx % 3]:
                # Get saved or default grades
                if lipid_class in st.session_state.grade_selections:
                    default_grades = st.session_state.grade_selections[lipid_class]
                elif lipid_class in ['LPC', 'SM']:
                    default_grades = ['A', 'B', 'C']
                else:
                    default_grades = ['A', 'B']

                selected_grades = st.multiselect(
                    f"**{lipid_class}**",
                    options=['A', 'B', 'C', 'D'],
                    default=default_grades,
                    key=f"grade_select_{lipid_class}"
                )

                st.session_state.grade_selections[lipid_class] = selected_grades

                if not selected_grades:
                    st.error("⚠️ Will be excluded!")

                grade_config[lipid_class] = selected_grades

        return grade_config


def display_msdial_data_type_selection():
    """Display MS-DIAL data type selection (raw vs pre-normalized)."""
    features = st.session_state.get('msdial_features', {})
    has_normalized_data = features.get('has_normalized_data', False)
    raw_samples = features.get('raw_sample_columns', [])
    norm_samples = features.get('normalized_sample_columns', [])

    if has_normalized_data and len(norm_samples) > 0:
        st.markdown("##### 📊 Data Type Selection")
        st.markdown(f"""
Your MS-DIAL export contains both raw and pre-normalized intensity values:
- **Raw data**: {len(raw_samples)} sample columns
- **Normalized data**: {len(norm_samples)} sample columns (after 'Lipid IS' column)
        """)

        options = [
            f"Raw intensity values ({len(raw_samples)} samples)",
            f"Pre-normalized values ({len(norm_samples)} samples)"
        ]

        # Initialize widget key from persisted index BEFORE rendering radio
        # This ensures consistency when navigating between pages
        widget_key = "msdial_data_type_radio"
        persisted_index = st.session_state.get('msdial_data_type_index', 0)
        if persisted_index < len(options):
            st.session_state[widget_key] = options[persisted_index]

        def on_data_type_change():
            """Callback to update session state immediately when selection changes."""
            selection = st.session_state[widget_key]
            new_index = options.index(selection) if selection in options else 0
            st.session_state.msdial_data_type_index = new_index
            st.session_state.msdial_use_normalized = "Pre-normalized" in selection
            # Clear cached data to force re-standardization with new selection
            st.session_state.standardized_df = None
            st.session_state.cleaned_df = None
            st.session_state.pre_filter_df = None
            st.session_state.continuation_df = None

        data_type = st.radio(
            "Select which data to use:",
            options,
            key=widget_key,
            on_change=on_data_type_change,
            help="Choose raw data if you want to apply LipidCruncher's normalization. Choose pre-normalized if MS-DIAL already normalized your data."
        )

        # Keep session state in sync after render
        use_normalized = "Pre-normalized" in data_type
        st.session_state.msdial_data_type_index = options.index(data_type) if data_type in options else 0
        st.session_state.msdial_use_normalized = use_normalized

        if use_normalized:
            st.info("📌 Using pre-normalized data. LipidCruncher's internal standard normalization will be skipped.")
        else:
            st.info("📌 Using raw intensity data. You can apply normalization in the next step.")

        st.markdown("---")


def display_quality_filtering_config() -> dict:
    """
    Display MS-DIAL quality filtering configuration.

    Returns:
        dict: Quality config with 'total_score_threshold' and 'require_msms' keys
    """
    features = st.session_state.get('msdial_features', {})
    quality_filtering_available = features.get('has_quality_score', False)
    msms_filtering_available = features.get('has_msms_matched', False)

    if not quality_filtering_available and not msms_filtering_available:
        st.warning("Quality filtering unavailable — no 'Total score' or 'MS/MS matched' columns found.")
        return None

    with st.expander("⚙️ Configure Quality Filtering", expanded=False):
        # Initialize session state
        if 'msdial_quality_level' not in st.session_state:
            st.session_state.msdial_quality_level = 'Moderate (Score ≥60)'

        if quality_filtering_available:
            quality_options = {
                'Strict (Score ≥80, MS/MS required)': {'total_score_threshold': 80, 'require_msms': True},
                'Moderate (Score ≥60)': {'total_score_threshold': 60, 'require_msms': False},
                'Permissive (Score ≥40)': {'total_score_threshold': 40, 'require_msms': False},
                'No filtering': {'total_score_threshold': 0, 'require_msms': False}
            }
            quality_options_list = list(quality_options.keys())
            widget_key = "msdial_quality_level_radio"

            # Initialize widget key from persisted value BEFORE rendering
            persisted_value = st.session_state.get('msdial_quality_level', 'Moderate (Score ≥60)')
            if persisted_value in quality_options_list:
                st.session_state[widget_key] = persisted_value

            def on_quality_level_change():
                st.session_state.msdial_quality_level = st.session_state[widget_key]

            selected_option = st.radio(
                "Quality filtering level:",
                quality_options_list,
                horizontal=True,
                key=widget_key,
                on_change=on_quality_level_change
            )
            st.session_state.msdial_quality_level = selected_option

            quality_config = quality_options[selected_option].copy()

            # MS/MS validation override (if available)
            if msms_filtering_available:
                col1, col2 = st.columns(2)
                with col1:
                    custom_msms = st.checkbox(
                        "Require MS/MS validation",
                        value=quality_config['require_msms'],
                        key="msdial_custom_msms"
                    )
                    quality_config['require_msms'] = custom_msms

            # Advanced: custom score threshold
            show_custom = st.checkbox("Customize score threshold", value=False, key="msdial_show_custom_threshold")
            if show_custom:
                custom_score = st.slider(
                    "Minimum Total Score:",
                    min_value=0,
                    max_value=100,
                    value=quality_config['total_score_threshold'],
                    step=5,
                    key="msdial_custom_score"
                )
                quality_config['total_score_threshold'] = custom_score

            # Summary
            st.markdown("---")
            st.markdown(f"**Current settings:** Score ≥ {quality_config['total_score_threshold']}, "
                       f"MS/MS required: {'Yes' if quality_config['require_msms'] else 'No'}")

            # Display filter results from previous processing run
            ingestion_result = st.session_state.get('ingestion_result')
            if ingestion_result and ingestion_result.cleaning_messages:
                st.markdown("**Filter Results:**")
                for msg in ingestion_result.cleaning_messages:
                    st.info(msg)

            return quality_config

        elif msms_filtering_available:
            # Only MS/MS filtering available
            require_msms = st.checkbox(
                "Require MS/MS validation",
                value=False,
                key="msdial_msms_only"
            )
            return {'total_score_threshold': 0, 'require_msms': require_msms}

    return None



# =============================================================================
# UI Components - Manage Internal Standards
# =============================================================================

def display_manage_internal_standards(
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Display the Manage Internal Standards expander.

    Allows users to:
    - Use automatically detected standards
    - Upload custom standards (from dataset or external)
    - Clear custom standards
    - View consistency plots for internal standards

    Args:
        cleaned_df: The cleaned data DataFrame (for extracting standards)
        auto_detected_df: Auto-detected internal standards DataFrame

    Returns:
        The active internal standards DataFrame to use for normalization
    """
    from app.services.standards import StandardsService
    from app.ui.standards_plots import display_standards_consistency_plots

    # Track the active standards DataFrame (set in branches below)
    active_standards_df = pd.DataFrame()

    with st.expander("Manage Internal Standards", expanded=False):
        st.markdown("""
Auto-detection identifies deuterated standards (`(d5)`, `(d7)`, `(d9)`),
`ISTD`/`IS` markers in class names, and SPLASH LIPIDOMIX® patterns.
        """)

        # Initialize session state for standards management
        if 'standards_source' not in st.session_state:
            st.session_state.standards_source = "Automatic Detection"
        if 'custom_standards_df' not in st.session_state:
            st.session_state.custom_standards_df = None
        if 'original_auto_intsta_df' not in st.session_state:
            st.session_state.original_auto_intsta_df = auto_detected_df

        # Standards source selection
        st.markdown("##### 🏷️ Standards Source")

        standards_source = st.radio(
            "Standards source:",
            ["Automatic Detection", "Upload Custom Standards"],
            horizontal=True,
            key="standards_source_radio",
            label_visibility="collapsed"
        )
        st.session_state.standards_source = standards_source

        if standards_source == "Automatic Detection":
            # Clear custom standards when switching to automatic
            if st.session_state.custom_standards_df is not None:
                st.session_state.custom_standards_df = None

            # Show auto-detected standards
            if auto_detected_df is not None and not auto_detected_df.empty:
                st.success(f"✓ Found {len(auto_detected_df)} standards")
                st.dataframe(auto_detected_df, use_container_width=True)

                # Download button
                csv = auto_detected_df.to_csv(index=False)
                st.download_button(
                    label="Download Detected Standards",
                    data=csv,
                    file_name="detected_standards.csv",
                    mime="text/csv",
                    key="download_auto_standards"
                )
                active_standards_df = auto_detected_df
            else:
                st.warning("No internal standards automatically detected in dataset.")
                active_standards_df = pd.DataFrame()

        else:  # Upload Custom Standards
            st.markdown("---")

            # Mode selection: standards in dataset or external
            st.markdown("**Are standards present in your main dataset?**")

            standards_location = st.radio(
                "Standards location:",
                options=[
                    "Yes — Extract from dataset",
                    "No — Uploading complete standards data"
                ],
                key="standards_location_radio",
                horizontal=True,
                label_visibility="collapsed"
            )

            use_extract_mode = "Yes" in standards_location

            # Format guidance
            st.markdown("---")
            if use_extract_mode:
                st.markdown("""
**CSV format:** Single column with lipid names (must exist in your dataset).

Example:
```
LipidMolec
PC(15:0-18:1(d7))
PE(15:0-18:1(d7))
SM(18:1(d9))
```
                """)
            else:
                st.markdown("""
**CSV format:** 1st column = lipid names, remaining columns = intensity values per sample.

Example:
```
LipidMolec,s1,s2,s3,s4
PC(15:0-18:1(d7)),1000,1200,1100,1050
PE(15:0-18:1(d7)),800,850,820,810
```
                """)

            # File uploader
            uploaded_file = st.file_uploader(
                "Upload standards CSV",
                type=['csv'],
                key="standards_file_uploader"
            )

            # Show preserved custom standards if file uploader is empty
            if uploaded_file is None and st.session_state.custom_standards_df is not None:
                st.success(f"✓ Using {len(st.session_state.custom_standards_df)} custom standards")
                st.dataframe(st.session_state.custom_standards_df, use_container_width=True)

                # Clear button
                if st.button("Clear Custom Standards", key="clear_custom_standards"):
                    st.session_state.custom_standards_df = None
                    st.session_state.standards_source = "Automatic Detection"
                    safe_rerun()

                active_standards_df = st.session_state.custom_standards_df

            # Process uploaded file
            elif uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)

                    # Process using StandardsService
                    result = StandardsService.process_standards_file(
                        uploaded_df=uploaded_df,
                        cleaned_df=cleaned_df,
                        standards_in_dataset=use_extract_mode
                    )

                    if result.standards_df is not None and not result.standards_df.empty:
                        st.session_state.custom_standards_df = result.standards_df

                        # For external standards, remove them from main dataset if present
                        if not use_extract_mode:
                            filtered_df, removed_lipids = StandardsService.remove_standards_from_dataset(
                                cleaned_df,
                                result.standards_df
                            )
                            if removed_lipids:
                                st.session_state.cleaned_df = filtered_df
                                preview = removed_lipids[:5]
                                more = f"... and {len(removed_lipids) - 5} more" if len(removed_lipids) > 5 else ""
                                st.warning(
                                    f"⚠️ Removed {len(removed_lipids)} standard(s) from main dataset: "
                                    f"{', '.join(preview)}{more}"
                                )

                        # Show processing info
                        if result.duplicates_removed > 0:
                            st.info(f"Removed {result.duplicates_removed} duplicate standard(s).")

                        st.success(f"✓ Loaded {result.standards_count} custom standards (mode: {result.source_mode})")
                        st.dataframe(result.standards_df, use_container_width=True)

                        # Download button
                        csv = result.standards_df.to_csv(index=False)
                        st.download_button(
                            label="Download Custom Standards",
                            data=csv,
                            file_name="custom_standards.csv",
                            mime="text/csv",
                            key="download_custom_standards"
                        )

                        active_standards_df = result.standards_df
                    else:
                        st.error("No valid standards found in uploaded file.")
                        active_standards_df = auto_detected_df if auto_detected_df is not None else pd.DataFrame()

                except ValueError as ve:
                    st.error(str(ve))
                    active_standards_df = auto_detected_df if auto_detected_df is not None else pd.DataFrame()
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    active_standards_df = auto_detected_df if auto_detected_df is not None else pd.DataFrame()

            else:
                # No file uploaded yet, use auto-detected as fallback
                if auto_detected_df is not None and not auto_detected_df.empty:
                    st.info("Upload a CSV file or switch to Automatic Detection.")
                    active_standards_df = auto_detected_df
                else:
                    st.warning("No standards available. Please upload a standards file.")
                    active_standards_df = pd.DataFrame()

        # Show consistency plots if we have standards and experiment config
        if (active_standards_df is not None and
            not active_standards_df.empty and
            'experiment' in st.session_state and
            st.session_state.experiment is not None):
            display_standards_consistency_plots(
                intsta_df=active_standards_df,
                experiment=st.session_state.experiment
            )

    return active_standards_df


# =============================================================================
# UI Components - Normalization
# =============================================================================

def _display_class_selection(cleaned_df: pd.DataFrame) -> list:
    """Display lipid class selection UI. Returns selected classes or empty list."""
    available_classes = NormalizationWorkflow.get_available_classes(cleaned_df)

    # Initialize session state for persistence (matches old app pattern)
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = available_classes.copy()

    def update_selected_classes():
        st.session_state.selected_classes = st.session_state.temp_selected_classes

    selected_classes = st.multiselect(
        'Classes to analyze:',
        options=available_classes,
        default=available_classes if not st.session_state.selected_classes else st.session_state.selected_classes,
        key='temp_selected_classes',
        on_change=update_selected_classes
    )

    return selected_classes


def _display_internal_standards_config(intsta_df: pd.DataFrame, selected_classes: list) -> tuple:
    """Display internal standards configuration UI. Returns (internal_standards, intsta_concentrations) or (None, None)."""
    if intsta_df is None or intsta_df.empty:
        st.error("No internal standards detected. Please upload standards or use a different method.")
        return None, None

    standards_by_class = NormalizationWorkflow.get_standards_by_class(intsta_df)
    all_standards = NormalizationWorkflow.get_available_standards(intsta_df)

    if not all_standards:
        st.error("No internal standards available.")
        return None, None

    internal_standards = {}
    intsta_concentrations = {}

    # Retrieve saved mappings for session state persistence
    saved_mappings = st.session_state.get('class_standard_map', {})

    with st.expander("⚙️ Internal Standards Mapping", expanded=True):
        # Show warnings for classes without specific standards
        for lipid_class in selected_classes:
            class_standards = standards_by_class.get(lipid_class, [])
            if not class_standards:
                if saved_mappings and lipid_class in saved_mappings and saved_mappings[lipid_class] in all_standards:
                    default_standard = saved_mappings[lipid_class]
                else:
                    default_standard = all_standards[0]
                st.warning(f"No specific standards available for {lipid_class}. Defaulting to {default_standard}.")

        # Class-to-standard mapping (double-column layout)
        for lipid_class in selected_classes:
            # Determine default standard (saved mapping -> class-specific -> first available)
            default_std = None
            if saved_mappings and lipid_class in saved_mappings:
                default_std = saved_mappings[lipid_class]
                if default_std not in all_standards:
                    default_std = None
            if not default_std:
                class_standards = standards_by_class.get(lipid_class, [])
                default_std = class_standards[0] if class_standards else all_standards[0]

            default_idx = all_standards.index(default_std) if default_std in all_standards else 0

            col1, col2 = st.columns([1, 2])
            with col1:
                st.text(lipid_class)
            with col2:
                selected_std = st.selectbox(
                    f'Select internal standard for {lipid_class}',
                    options=all_standards,
                    index=default_idx,
                    key=f'standard_selection_{lipid_class}',
                    label_visibility="collapsed"
                )
                internal_standards[lipid_class] = selected_std

        # Save mapping to session state for persistence
        st.session_state.class_standard_map = internal_standards

        # Standard concentrations (always visible, not optional)
        if st.session_state.get('standard_concentrations') is None:
            st.session_state.standard_concentrations = {}

        st.write("Enter the concentration of each selected internal standard (µM):")

        all_concentrations_valid = True
        for std in set(internal_standards.values()):
            widget_key = f"conc_{std}"
            # Initialize from preserved concentrations if not already set
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state.standard_concentrations.get(std, 1.0)

            conc = st.number_input(
                f"Concentration (µM) for {std}",
                min_value=0.0,
                step=0.1,
                key=widget_key
            )

            # Sync back for session preservation
            st.session_state.standard_concentrations[std] = conc

            if conc <= 0:
                st.error(f"Please enter a valid concentration for {std}")
                all_concentrations_valid = False
            intsta_concentrations[std] = conc

        if not all_concentrations_valid:
            st.error("Please enter valid concentrations for all standards")
            return None, None

    return internal_standards, intsta_concentrations


def _display_protein_config(experiment: ExperimentConfig) -> dict:
    """Display protein concentration configuration UI. Returns protein_concentrations dict or None."""

    sample_names = experiment.full_samples_list
    protein_concentrations = {}

    with st.expander("⚙️ Protein Concentration Data", expanded=True):
        # Initialize method selection key if not present
        if 'protein_input_method' not in st.session_state:
            st.session_state.protein_input_method = "Manual Input"

        # Track previous method to detect changes
        prev_method = st.session_state.get('protein_input_method_prev')

        method = st.radio(
            "Input method:",
            ["Manual Input", "Upload CSV File"],
            key='protein_input_method',
            horizontal=True
        )

        # Detect method change and clear stale data
        if prev_method is not None and prev_method != method:
            if 'protein_df' in st.session_state:
                del st.session_state.protein_df
            for sample in sample_names:
                widget_key = f"protein_{sample}"
                if widget_key in st.session_state:
                    del st.session_state[widget_key]

        st.session_state.protein_input_method_prev = method

        if method == "Manual Input":
            # Get preserved protein data for restoring values
            preserved_protein_df = st.session_state.get('protein_df')
            preserved_values = {}
            if preserved_protein_df is not None and isinstance(preserved_protein_df, dict):
                preserved_values = preserved_protein_df
            elif preserved_protein_df is not None and hasattr(preserved_protein_df, 'columns'):
                if 'Sample' in preserved_protein_df.columns and 'Concentration' in preserved_protein_df.columns:
                    preserved_values = dict(zip(preserved_protein_df['Sample'], preserved_protein_df['Concentration']))

            # Initialize session state for all samples BEFORE widgets render
            for sample in sample_names:
                widget_key = f"protein_{sample}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = float(preserved_values.get(sample, 1.0))

            # 3-column flat grid layout (matches old app)
            cols = st.columns(3)
            for idx, sample in enumerate(sample_names):
                with cols[idx % 3]:
                    concentration = st.number_input(
                        f'{sample}:',
                        min_value=0.0,
                        max_value=1000000.0,
                        step=0.1,
                        key=f"protein_{sample}"
                    )
                    protein_concentrations[sample] = concentration

            # Preserve in session state
            st.session_state.protein_df = protein_concentrations.copy()
            return protein_concentrations

        else:  # Upload CSV File
            st.markdown("**CSV format:** Single column named `Concentration` with one value per sample (in order).")

            preserved_protein_df = st.session_state.get('protein_df')

            uploaded_file = st.file_uploader("Upload CSV", type="csv", key="protein_csv_upload")

            if uploaded_file is not None:
                try:
                    csv_df = pd.read_csv(uploaded_file)

                    if 'Concentration' not in csv_df.columns:
                        st.error(f"CSV must contain a column named 'Concentration'. Found: {list(csv_df.columns)}")
                        return None

                    if len(csv_df) != len(sample_names):
                        st.error(f"Row count ({len(csv_df)}) doesn't match sample count ({len(sample_names)})")
                        return None

                    csv_df['Concentration'] = pd.to_numeric(csv_df['Concentration'], errors='coerce')
                    if csv_df['Concentration'].isna().any():
                        st.error("Some concentration values couldn't be converted to numbers.")
                        return None

                    # Convert to dict for NormalizationConfig compatibility
                    protein_concentrations = dict(zip(sample_names, csv_df['Concentration'].tolist()))
                    st.session_state.protein_df = protein_concentrations.copy()
                    st.success(f"✓ Loaded {len(protein_concentrations)} concentration values")
                    return protein_concentrations

                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
                    return None

            # No new file uploaded - check preserved data
            if preserved_protein_df is not None and isinstance(preserved_protein_df, dict):
                if len(preserved_protein_df) == len(sample_names):
                    st.success(f"✓ Using previously loaded {len(preserved_protein_df)} concentration values")
                    return preserved_protein_df

            st.info("Please upload a CSV file with protein concentrations.")
            return None


def _run_normalization(
    cleaned_df: pd.DataFrame,
    intsta_df: pd.DataFrame,
    experiment: ExperimentConfig,
    data_format: str,
    config_method: str,
    selected_classes: list,
    internal_standards: dict,
    intsta_concentrations: dict,
    protein_concentrations: dict
) -> NormalizationWorkflowResult:
    """Run normalization workflow and return result."""
    try:
        norm_config = NormalizationConfig(
            method=config_method,
            selected_classes=selected_classes,
            internal_standards=internal_standards if internal_standards else None,
            intsta_concentrations=intsta_concentrations if intsta_concentrations else None,
            protein_concentrations=protein_concentrations if protein_concentrations else None
        )

        format_map = {
            'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
            'MS-DIAL': DataFormat.MSDIAL,
            'Generic Format': DataFormat.GENERIC,
            'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
        }

        workflow_config = NormalizationWorkflowConfig(
            experiment=experiment,
            normalization=norm_config,
            data_format=format_map.get(data_format, DataFormat.GENERIC)
        )

        result = NormalizationWorkflow.run(
            df=cleaned_df,
            config=workflow_config,
            intsta_df=intsta_df
        )

        if result.success:
            st.session_state.normalization_result = result
            st.session_state.normalized_df = result.normalized_df
            st.session_state.continuation_df = result.normalized_df

        return result

    except Exception as e:
        st.error(f"Normalization error: {e}")
        return None


def display_normalization_ui(cleaned_df: pd.DataFrame, intsta_df: pd.DataFrame, experiment: ExperimentConfig, data_format: str):
    """Display normalization options and apply normalization automatically."""
    # About Normalization Methods (documentation)
    with st.expander("📖 About Normalization Methods", expanded=False):
        st.markdown("""
### Normalization Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **None** | Raw values | Data already normalized externally |
| **Internal Standards** | `(Intensity_lipid / Intensity_standard) × Conc_standard` | Correct for extraction/ionization variability |
| **Protein-based** | `Intensity_lipid / Protein_conc` | Normalize to starting material (e.g., BCA assay) |
| **Both** | `(Intensity_lipid / Intensity_standard) × (Conc_standard / Protein_conc)` | Combined correction |

After normalization, `intensity[...]` columns become `concentration[...]` columns.
        """)

    # Class selection
    st.markdown("##### 🎯 Select Lipid Classes")
    selected_classes = _display_class_selection(cleaned_df)
    if not selected_classes:
        st.warning("Please select at least one lipid class.")
        return None

    # Check if we have standards available
    has_standards = intsta_df is not None and not intsta_df.empty

    # Check if using pre-normalized MS-DIAL data (IS normalization not applicable)
    is_msdial_prenormalized = (data_format == 'MS-DIAL' and
                               st.session_state.get('msdial_use_normalized', False))

    # Determine available normalization options
    if is_msdial_prenormalized:
        normalization_options = ['None (pre-normalized data)', 'Protein-based']
        st.markdown("*Internal standards options unavailable — using pre-normalized MS-DIAL data.*")
    elif has_standards:
        normalization_options = ['None (pre-normalized data)', 'Internal Standards', 'Protein-based', 'Both']
    else:
        normalization_options = ['None (pre-normalized data)', 'Protein-based']
        st.markdown("*Internal standards options unavailable — no standards detected or uploaded.*")

    # Initialize session state for normalization method
    if 'norm_method_selection' not in st.session_state:
        st.session_state['norm_method_selection'] = 'None (pre-normalized data)'

    # Handle case where saved method is no longer available (e.g., standards removed)
    current_selection = st.session_state.get('norm_method_selection')
    if current_selection not in normalization_options:
        st.session_state['norm_method_selection'] = 'None (pre-normalized data)'

    # Method selection
    st.markdown("##### ⚙️ Normalization Method")
    method = st.radio(
        "Method:",
        options=normalization_options,
        horizontal=True,
        key='norm_method_selection'
    )

    method_map = {
        'None (pre-normalized data)': 'none',
        'Internal Standards': 'internal_standard',
        'Protein-based': 'protein',
        'Both': 'both'
    }
    config_method = method_map.get(method, 'none')

    # Collect configuration based on method
    internal_standards = {}
    intsta_concentrations = {}
    protein_concentrations = {}

    if method in ['Internal Standards', 'Both']:
        internal_standards, intsta_concentrations = _display_internal_standards_config(intsta_df, selected_classes)
        if internal_standards is None:
            return None

    if method in ['Protein-based', 'Both']:
        protein_concentrations = _display_protein_config(experiment)
        if protein_concentrations is None:
            return None

    # Apply normalization automatically (no button needed)
    result = _run_normalization(
        cleaned_df, intsta_df, experiment, data_format,
        config_method, selected_classes,
        internal_standards, intsta_concentrations, protein_concentrations
    )

    # Display results
    if result and result.success:
        st.markdown("##### 📊 Final Normalized Data")

        if result.normalized_df is not None:
            st.dataframe(result.normalized_df, use_container_width=True)
            csv = result.normalized_df.to_csv(index=False)
            st.download_button(
                label="Download Normalized Data",
                data=csv,
                file_name="normalized_data.csv",
                mime="text/csv",
                key="download_normalized_data"
            )
    elif result:
        for error in result.validation_errors:
            st.error(error)

    return st.session_state.get('normalized_df')


# =============================================================================
# Main App Page
# =============================================================================

def display_app_page():
    """Display the main application page (Module 1) with centered layout matching landing page."""
    # Centered layout matching landing page width
    _, center, _ = st.columns([1, 3, 1])

    # Sidebar: Format selection (must happen before center content that depends on it)
    data_format = display_format_selection()

    with center:
        display_logo(centered=True)
        st.markdown("Process, analyze and visualize lipidomic data from multiple sources.")
        display_format_requirements(data_format)

    # Sidebar: File upload
    raw_df = display_file_upload(data_format)

    if raw_df is None:
        with center:
            st.info("Upload a dataset or load sample data to begin.")

            # Back to landing button
            if st.button("← Back to Home"):
                st.session_state.page = 'landing'
                safe_rerun()
        return

    # Standardize data if not already done
    if st.session_state.get('standardized_df') is None:
        standardized_df = standardize_uploaded_data(raw_df, data_format)
        if standardized_df is None:
            return
        st.session_state.standardized_df = standardized_df
    else:
        standardized_df = st.session_state.standardized_df

    # Sidebar: Column Name Standardization (display mapping)
    mapping_valid, modified_df = display_column_mapping(standardized_df, data_format)
    if not mapping_valid:
        return

    # Use modified df if sample override was applied
    if modified_df is not None:
        standardized_df = modified_df
        st.session_state.standardized_df = standardized_df

    # Use standardized df for sample grouping
    df = standardized_df

    # Sidebar: Sample grouping (includes experiment def, grouping, BQC, confirm)
    experiment, bqc_label = display_sample_grouping(df, data_format)

    if experiment is None:
        with center:
            st.info("Configure your experiment in the sidebar and check 'Confirm Inputs' to proceed.")
        return

    # ==========================================================================
    # Main area: Processing Module (Automatic Flow)
    # ==========================================================================
    with center:
        st.subheader("Data Standardization, Filtering, and Normalization")

        # Step 1: Data processing documentation
        display_data_processing_docs(data_format)

        # Step 2: Format-specific filtering configuration
        grade_config = None
        quality_config = None

        if data_format == 'LipidSearch 5.0':
            grade_config_dict = display_grade_filtering_config(raw_df)
            if grade_config_dict is not None:
                grade_config = GradeFilterConfig(grade_config=grade_config_dict)

        elif data_format == 'MS-DIAL':
            # Data type selection (raw vs normalized)
            display_msdial_data_type_selection()

            # Quality filtering
            quality_config_dict = display_quality_filtering_config()
            if quality_config_dict is not None:
                quality_config = QualityFilterConfig(
                    total_score_threshold=quality_config_dict.get('total_score_threshold', 0),
                    require_msms=quality_config_dict.get('require_msms', False)
                )

        # Step 3: Automatic data processing (runs on every confirmation)
        # Build config hash to detect when settings change
        config_hash = f"{data_format}_{grade_config}_{quality_config}"

        # Process data automatically (no button needed)
        format_map = {
            'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
            'MS-DIAL': DataFormat.MSDIAL,
            'Generic Format': DataFormat.GENERIC,
            'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
        }

        # Run ingestion workflow (clean data WITHOUT zero filtering - user configures thresholds after)
        config = IngestionConfig(
            experiment=experiment,
            data_format=format_map.get(data_format),
            bqc_label=bqc_label,
            apply_zero_filter=False,  # Don't apply yet - let user configure
            grade_config=grade_config,
            quality_config=quality_config,
        )

        try:
            result = DataIngestionWorkflow.run(df, config)
            st.session_state.ingestion_result = result
            st.session_state.pre_filter_df = result.cleaned_df  # Store pre-filter version

            if result.is_valid:
                st.session_state.cleaned_df = result.cleaned_df
                st.session_state.intsta_df = result.internal_standards_df
                st.session_state.continuation_df = result.cleaned_df
        except Exception as e:
            st.error(f"Processing error: {e}")
            return

        # Step 4: Display results (automatic - no button needed)
        if not result.is_valid:
            for error in result.validation_errors:
                st.error(error)
            return

        # Show warnings
        for warning in result.validation_warnings:
            st.warning(warning)

        # Step 5: Zero Filtering Configuration (interactive)
        pre_filter_df = st.session_state.get('pre_filter_df', result.cleaned_df)
        if pre_filter_df is not None and not pre_filter_df.empty:
            filtered_df, removed_species, zero_config = display_zero_filtering_config(
                pre_filter_df, experiment, bqc_label,
                data_format=st.session_state.get('format_type')
            )

            # Update session state with filtered data
            if filtered_df is not None:
                st.session_state.cleaned_df = filtered_df
                st.session_state.continuation_df = filtered_df

        # Step 6: Show final filtered data (outside expander, matching old app)
        # Sort by ClassKey so all species of the same class are grouped together
        if 'ClassKey' in st.session_state.cleaned_df.columns:
            st.session_state.cleaned_df = st.session_state.cleaned_df.sort_values('ClassKey').reset_index(drop=True)
            st.session_state.continuation_df = st.session_state.cleaned_df

        st.markdown("##### 📋 Final Filtered Data (Pre-Normalization)")
        st.dataframe(st.session_state.cleaned_df, use_container_width=True)
        csv = st.session_state.cleaned_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="final_filtered_data.csv",
            mime="text/csv",
            key="download_filtered_data"
        )

        # Step 7: Manage Internal Standards
        auto_detected_intsta_df = st.session_state.get('intsta_df', result.internal_standards_df)
        intsta_df = display_manage_internal_standards(
            cleaned_df=st.session_state.cleaned_df,
            auto_detected_df=auto_detected_intsta_df
        )
        # Update session state with the active standards
        st.session_state.intsta_df = intsta_df

        # Step 8: Normalization (automatic - no button needed)
        display_normalization_ui(
            cleaned_df=st.session_state.cleaned_df,
            intsta_df=intsta_df,
            experiment=experiment,
            data_format=data_format
        )

        # Navigation
        st.markdown("---")
        if st.button("← Back to Home"):
            st.session_state.page = 'landing'
            StreamlitAdapter.reset_data_state()
            safe_rerun()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main application entry point."""
    if st.session_state.page == 'landing':
        display_landing_page()
    elif st.session_state.page == 'app':
        display_app_page()


if __name__ == "__main__":
    main()
