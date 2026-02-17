"""
Experiment Configuration UI Components for LipidCruncher sidebar.

This module contains:
- detect_sample_columns: Detect intensity/sample columns based on format
- extract_sample_names: Extract clean sample names from column names
- display_experiment_definition: Display experiment definition UI (conditions, samples)
"""

import streamlit as st
import pandas as pd


# =============================================================================
# Helper Functions
# =============================================================================

def detect_sample_columns(df: pd.DataFrame, data_format: str) -> list:
    """Detect intensity/sample columns based on format.

    After standardization, all formats use intensity[...] columns for sample data.

    Args:
        df: Standardized DataFrame
        data_format: The selected data format string

    Returns:
        List of intensity column names
    """
    # After standardization, all formats use intensity[...] columns
    return [col for col in df.columns if col.startswith('intensity[')]


def extract_sample_names(columns: list, data_format: str) -> list:
    """Extract clean sample names from column names.

    After standardization, all formats use intensity[sample_name] pattern.

    Args:
        columns: List of intensity column names
        data_format: The selected data format string

    Returns:
        List of clean sample names
    """
    names = []
    for col in columns:
        if col.startswith('intensity[') and col.endswith(']'):
            # Extract name from intensity[name]
            names.append(col[10:-1])  # Remove 'intensity[' and ']'
        else:
            names.append(col)
    return names


# =============================================================================
# UI Components
# =============================================================================

def display_experiment_definition(df: pd.DataFrame, data_format: str, sample_cols: list) -> tuple:
    """
    Display experiment definition UI.

    Args:
        df: Standardized DataFrame
        data_format: The selected data format string
        sample_cols: List of detected sample columns

    Returns:
        Tuple of (n_conditions, conditions_list, number_of_samples_list)
        or (None, None, None) if invalid
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
        col1, col2 = st.sidebar.columns([1, 1])
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
