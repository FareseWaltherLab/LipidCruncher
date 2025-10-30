"""
Normalization workflow orchestrator.
Coordinates UI interactions and service calls for data normalization.
"""
import streamlit as st
import pandas as pd
from typing import Optional
from ..ui.normalization_ui import (
    select_lipid_classes_ui,
    select_normalization_method_ui,
    collect_protein_concentrations_ui,
    collect_internal_standards_ui,
    display_normalization_info
)
from ..adapters.streamlit_adapter import StreamlitDataAdapter
from ..core.models.experiment import ExperimentConfig
from ..core.models.normalization import NormalizationConfig


class NormalizationWorkflow:
    """
    Orchestrates the complete normalization workflow.
    Separates UI logic from business logic.
    """
    
    def __init__(self):
        self.adapter = StreamlitDataAdapter()
    
    def run(
        self,
        cleaned_df: pd.DataFrame,
        intsta_df: pd.DataFrame,
        experiment,  # Old Experiment object
        format_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Run the complete normalization workflow.
        
        Args:
            cleaned_df: Cleaned data DataFrame
            intsta_df: Internal standards DataFrame
            experiment: Experiment object (old style)
            format_type: Data format type
        
        Returns:
            Normalized DataFrame or None if workflow incomplete
        """
        # Convert old Experiment to new ExperimentConfig
        experiment_config = ExperimentConfig(
            n_conditions=experiment.n_conditions,
            conditions_list=experiment.conditions_list,
            number_of_samples_list=experiment.number_of_samples_list
        )
        
        # Store essential columns for LipidSearch format
        stored_columns = {}
        if format_type == 'LipidSearch 5.0':
            essential_columns = ['CalcMass', 'BaseRt']
            stored_columns = {
                col: cleaned_df[col] 
                for col in essential_columns 
                if col in cleaned_df.columns
            }
        
        # Display info about normalization
        display_normalization_info()
        
        # Step 1: Select lipid classes
        selected_classes = select_lipid_classes_ui(cleaned_df)
        if selected_classes is None:
            return None
        
        # Filter data by selected classes
        filtered_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_classes)].copy()
        
        # Step 2: Select normalization method
        do_protein, do_standards = select_normalization_method_ui()
        
        if not do_protein and not do_standards:
            st.info("No normalization selected. Using raw intensity values.")
            final_df = self._finalize_dataframe(filtered_df, stored_columns, format_type)
            
            # Display the data
            st.subheader("View Dataset (No Normalization Applied)")
            st.write(final_df)
            
            # Download button
            csv = final_df.to_csv(index=False)
            st.download_button(
                label="Download Data",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv",
                key="download_filtered_data"
            )
            
            return final_df
        
        # Initialize normalized_df
        normalized_df = filtered_df.copy()
        
        # Step 3: Collect inputs and normalize
        try:
            # Validate that internal standards are available when needed
            if do_standards and (intsta_df is None or intsta_df.empty):
                st.error("Internal standards normalization requires internal standards data. "
                        "Please ensure your dataset contains internal standards or select a different normalization method.")
                return None

            # Handle internal standards normalization FIRST (matches processing order in NormalizationService)
            internal_standards = None
            intsta_concentrations = None
            
            if do_standards:
                with st.expander("📊 Internal Standards Normalization", expanded=True):
                    result = collect_internal_standards_ui(selected_classes, intsta_df)
                    if result is not None:
                        internal_standards, intsta_concentrations = result
            
            # Handle protein normalization SECOND (matches processing order in NormalizationService)
            protein_df = None
            protein_concentrations = None
            
            if do_protein:
                with st.expander("🧪 Protein-based Normalization", expanded=True):
                    protein_df = collect_protein_concentrations_ui(experiment_config)
                    if protein_df is not None:
                        # Convert to dict
                        protein_concentrations = dict(zip(protein_df['Sample'], protein_df['Concentration']))
            
            # Check if all required inputs are collected before proceeding
            ready_to_normalize = True
            missing_inputs = []
            
            if do_standards and (internal_standards is None or intsta_concentrations is None):
                ready_to_normalize = False
                missing_inputs.append("internal standards configuration")
            
            if do_protein and protein_concentrations is None:
                ready_to_normalize = False
                missing_inputs.append("protein concentrations")
            
            if not ready_to_normalize:
                st.info(f"⏳ Please complete: {', '.join(missing_inputs)}")
                return None
            
            # Determine normalization method
            if do_protein and do_standards:
                method = 'both'
            elif do_protein:
                method = 'protein'
            else:
                method = 'internal_standard'
            
            # Build NormalizationConfig
            norm_config = NormalizationConfig(
                method=method,
                selected_classes=selected_classes,
                internal_standards=internal_standards,
                intsta_concentrations=intsta_concentrations,
                protein_concentrations=protein_concentrations
            )
            
            # Validate configuration
            try:
                norm_config.validate_complete()
            except ValueError as e:
                st.error(f"Configuration error: {str(e)}")
                return None
            
            # Perform normalization
            with st.spinner("Normalizing data..."):
                normalized_df = self.adapter.normalize_data(
                    filtered_df,
                    norm_config,
                    experiment_config,
                    intsta_df=intsta_df if not intsta_df.empty else None,
                    protein_df=protein_df
                )
            
            # Finalize
            final_df = self._finalize_dataframe(normalized_df, stored_columns, format_type)
            
            # Display normalized data
            st.success("✅ Normalization completed successfully!")
            
            st.subheader("View Normalized Dataset")
            st.write(final_df)
            
            # Download button
            csv = final_df.to_csv(index=False)
            st.download_button(
                label="Download Normalized Data",
                data=csv,
                file_name="normalized_data.csv",
                mime="text/csv",
                key="download_normalized_data"
            )
            
            return final_df
        
        except Exception as e:
            st.error(f"Error during normalization: {str(e)}")
            st.exception(e)
            return None
    
    def _finalize_dataframe(
        self,
        df: pd.DataFrame,
        stored_columns: dict,
        format_type: str
    ) -> pd.DataFrame:
        """
        Add back stored columns and perform final cleanup.
        
        Args:
            df: Normalized DataFrame
            stored_columns: Dict of columns to restore
            format_type: Data format type
        
        Returns:
            Finalized DataFrame
        """
        # Restore essential columns for LipidSearch
        if format_type == 'LipidSearch 5.0':
            for col_name, col_data in stored_columns.items():
                if col_name not in df.columns:
                    df[col_name] = col_data
        
        return df