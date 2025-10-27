"""
UI components for data normalization workflow.
Separates UI logic from business logic.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Tuple


def select_lipid_classes_ui(cleaned_df: pd.DataFrame) -> Optional[List[str]]:
    """
    Display UI for selecting lipid classes to normalize.
    
    Args:
        cleaned_df: Cleaned dataframe with ClassKey column
    
    Returns:
        List of selected class names, or None if none selected
    """
    if 'ClassKey' not in cleaned_df.columns:
        st.error("ClassKey column is required for normalization.")
        return None
    
    all_classes = sorted(cleaned_df['ClassKey'].unique())
    
    # Initialize session state with ALL classes selected by default
    if 'selected_classes' not in st.session_state or st.session_state.selected_classes is None:
        st.session_state.selected_classes = all_classes  # Start with all selected
    
    def update_selected_classes():
        st.session_state.selected_classes = st.session_state.temp_selected_classes
    
    selected_classes = st.multiselect(
        'Select lipid classes you would like to analyze:',
        options=all_classes,
        default=st.session_state.selected_classes,  # Use persisted selection
        key='temp_selected_classes',
        on_change=update_selected_classes
    )
    
    if not selected_classes:
        st.warning("Please select at least one lipid class to proceed with normalization.")
        return None
    
    return selected_classes


def select_normalization_method_ui() -> Tuple[bool, bool]:
    """
    Display UI for selecting normalization method.
    
    Returns:
        Tuple of (do_protein, do_standards)
    """
    st.subheader("Select Normalization Method")
    
    norm_method = st.radio(
        "Choose normalization approach:",
        ["None", "Internal Standards Only", "Protein-based Only", "Both Internal Standards and Protein"],
        index=0
    )
    
    do_protein = norm_method in ["Protein-based Only", "Both Internal Standards and Protein"]
    do_standards = norm_method in ["Internal Standards Only", "Both Internal Standards and Protein"]
    
    return do_protein, do_standards


def collect_protein_concentrations_ui(experiment_config) -> Optional[pd.DataFrame]:
    """
    Display UI for collecting protein concentrations.
    
    Args:
        experiment_config: ExperimentConfig with sample information
    
    Returns:
        DataFrame with Sample and Concentration columns, or None
    """
    st.subheader("Protein Concentration Input")
    
    method = st.radio(
        "Select the method for providing protein concentrations:",
        ["Manual Input", "Upload Excel File"],
        index=1,
        key="protein_input_method"
    )
    
    if method == "Manual Input":
        protein_concentrations = {}
        
        st.write("Enter protein concentration (mg/mL) for each sample:")
        for sample in experiment_config.full_samples_list:
            conc = st.number_input(
                f"Concentration for {sample}:",
                min_value=0.0,
                value=2.0,
                step=0.1,
                key=f"protein_conc_{sample}"
            )
            protein_concentrations[sample] = conc
        
        return pd.DataFrame({
            'Sample': list(protein_concentrations.keys()),
            'Concentration': list(protein_concentrations.values())
        })
    
    else:  # Upload Excel File
        uploaded_file = st.file_uploader(
            "Upload Excel file with protein concentrations",
            type=['xlsx', 'xls'],
            key="protein_file_upload"
        )
        
        if uploaded_file is not None:
            try:
                protein_df = pd.read_excel(uploaded_file)
                
                # Validate columns
                if 'Sample' not in protein_df.columns or 'Concentration' not in protein_df.columns:
                    st.error("Excel file must have 'Sample' and 'Concentration' columns")
                    return None
                
                # Validate samples match
                file_samples = set(protein_df['Sample'])
                expected_samples = set(experiment_config.full_samples_list)
                
                if file_samples != expected_samples:
                    st.error(f"Sample mismatch. Expected: {expected_samples}, Got: {file_samples}")
                    return None
                
                st.success("✅ Protein concentrations loaded successfully")
                st.dataframe(protein_df)
                
                return protein_df
                
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
        
        return None


def collect_internal_standards_ui(
    selected_classes: List[str],
    intsta_df: pd.DataFrame
) -> Optional[Tuple[Dict[str, str], Dict[str, float]]]:
    """
    Display UI for selecting internal standards and their concentrations.
    
    Args:
        selected_classes: List of lipid classes to normalize
        intsta_df: DataFrame of available internal standards
    
    Returns:
        Tuple of (class_to_standard_mapping, standard_concentrations) or None
    """
    st.subheader("Internal Standards Configuration")
    
    if intsta_df.empty:
        st.error("No internal standards available. Please ensure standards were detected during cleaning.")
        return None
    
    # Check for intensity columns
    intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
    if not intensity_cols:
        st.error("Internal standards data does not contain properly formatted intensity columns")
        return None
    
    # Group standards by class
    standards_by_class = {}
    if 'ClassKey' in intsta_df.columns:
        standards_by_class = intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()
    
    # Select standard for each class
    st.write("### Step 1: Select Internal Standard for Each Class")
    class_to_standard = {}
    
    for lipid_class in selected_classes:
        available_standards = standards_by_class.get(lipid_class, intsta_df['LipidMolec'].tolist())
        
        if not available_standards:
            st.warning(f"No standards available for class {lipid_class}")
            continue
        
        selected_std = st.selectbox(
            f"Standard for {lipid_class}:",
            options=available_standards,
            key=f"standard_select_{lipid_class}"
        )
        
        class_to_standard[lipid_class] = selected_std
    
    if len(class_to_standard) != len(selected_classes):
        st.error("Please select standards for all lipid classes")
        return None
    
    # Collect concentrations
    st.write("### Step 2: Enter Concentrations for Selected Standards")
    
    unique_standards = set(class_to_standard.values())
    
    # Initialize session state for persistence
    if 'standard_concentrations' not in st.session_state:
        st.session_state.standard_concentrations = {}
    
    standard_concentrations = {}
    all_valid = True
    
    for standard in unique_standards:
        default_value = st.session_state.standard_concentrations.get(standard, 1.0)
        
        conc = st.number_input(
            f"Concentration (µM) for {standard}:",
            min_value=0.0,
            value=default_value,
            step=0.1,
            key=f"conc_{standard}"
        )
        
        st.session_state.standard_concentrations[standard] = conc
        
        if conc <= 0:
            st.error(f"Please enter a valid concentration for {standard}")
            all_valid = False
        
        standard_concentrations[standard] = conc
    
    if not all_valid:
        return None
    
    return class_to_standard, standard_concentrations


def display_normalization_info():
    """Display information about normalization methods in readable format."""
    with st.expander("ℹ️ About Normalization Methods"):
        st.markdown("### Data Normalization Methods")
        
        st.markdown("**None**: Use raw intensity values without normalization. This is suitable if your data has already been normalized externally.")
        
        st.markdown("**Internal Standards**: Normalize lipid measurements using spiked-in internal standards of known concentration. For each lipid class, you'll select an appropriate internal standard. The normalization formula is:")
        st.code("Concentration = (Intensity_lipid / Intensity_standard) × Concentration_standard")
        
        st.markdown("**Protein-based**: Normalize lipid intensities against protein concentration (e.g., determined by a BCA assay). This adjusts for differences in starting material. The normalization formula is:")
        st.code("Concentration = Intensity_lipid / Protein_concentration")
        
        st.markdown("**Both**: Apply both internal standards and protein normalization sequentially:")
        st.code("Concentration = (Intensity_lipid / Intensity_standard) × Concentration_standard / Protein_concentration")
