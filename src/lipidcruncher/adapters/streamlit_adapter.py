"""
Adapter layer between Streamlit UI and core services.
Handles conversion between Streamlit state and service models.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Tuple, Dict, List
from ..core.services.data_cleaning_service import DataCleaningService
from ..core.services.normalization_service import NormalizationService
from ..core.services.zero_filtering_service import ZeroFilteringService
from ..core.services.format_preprocessing_service import FormatPreprocessingService
from ..core.models.experiment import ExperimentConfig
from ..core.models.normalization import NormalizationConfig


class StreamlitDataAdapter:
    """
    Adapter for data cleaning and normalization in Streamlit app.
    Converts between Streamlit state and core service models.
    """
    
    def __init__(self):
        self.cleaning_service = DataCleaningService()
        self.normalization_service = NormalizationService()
        self.zero_filter_service = ZeroFilteringService()
        self.format_preprocessing_service = FormatPreprocessingService()
    
    def clean_data(
        self,
        df: pd.DataFrame,
        experiment_config: ExperimentConfig,
        data_format: str,
        grade_config: Optional[Dict[str, List[str]]] = None,
        apply_zero_filter: bool = False,
        zero_filter_threshold: float = 0.0,
        bqc_label: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Clean uploaded data based on format, with optional zero filtering.
        
        Args:
            df: Raw uploaded DataFrame
            experiment_config: Experiment configuration
            data_format: Format type ('LipidSearch 5.0', 'Generic Format', 'Metabolomics Workbench')
            grade_config: Optional grade filtering config for LipidSearch
            apply_zero_filter: Whether to apply zero filtering
            zero_filter_threshold: Threshold for zero filtering
            bqc_label: Label for BQC condition (if present)
        
        Returns:
            Tuple of (cleaned_df, internal_standards_df, removed_species)
        """
        try:
            # Step 1: Map UI format names to service format types
            format_type_map = {
                'LipidSearch 5.0': 'lipidsearch',
                'Generic Format': 'generic',
                'Metabolomics Workbench': 'metabolomics_workbench'
            }
            
            format_type = format_type_map.get(data_format, 'generic')
            
            # Step 2: Preprocess/standardize the data format
            preprocessed_df, success, message = self.format_preprocessing_service.validate_and_preprocess(
                df, format_type
            )
            
            if not success:
                st.error(f"❌ Preprocessing failed: {message}")
                raise ValueError(f"Preprocessing failed: {message}")
            
            if preprocessed_df is None:
                st.error("❌ Preprocessing returned no data")
                raise ValueError("Preprocessing returned no data")
            
            st.info(f"✅ Data preprocessed: {message}")
            
            # Step 3: Clean the preprocessed data
            if data_format == 'LipidSearch 5.0':
                cleaned_df = self.cleaning_service.clean_lipidsearch_data(
                    preprocessed_df,
                    experiment_config,
                    grade_config
                )
            else:
                # Generic and Metabolomics Workbench both use generic cleaning
                cleaned_df = self.cleaning_service.clean_generic_data(
                    preprocessed_df,
                    experiment_config
                )
            
            # Step 4: Apply zero filtering (if requested)
            removed_species = []
            if apply_zero_filter:
                cleaned_df, removed_species = self.zero_filter_service.filter_by_zeros(
                    cleaned_df,
                    experiment_config,
                    threshold=zero_filter_threshold,
                    bqc_label=bqc_label
                )
                
                if removed_species:
                    n_removed = len(removed_species)
                    st.warning(f"🔍 Removed {n_removed} species based on zero filter")
            
            # Step 5: Extract internal standards
            cleaned_df, standards_df = self.cleaning_service.extract_internal_standards(cleaned_df)
            
            st.success(f"✅ Data cleaned successfully: {len(cleaned_df)} lipids, {len(standards_df)} standards")
            
            return cleaned_df, standards_df, removed_species
            
        except Exception as e:
            st.error(f"❌ Error during data cleaning: {str(e)}")
            raise
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        norm_config: NormalizationConfig,
        experiment_config: ExperimentConfig,
        intsta_df: Optional[pd.DataFrame] = None,
        protein_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Normalize cleaned data.
        
        Args:
            df: Cleaned DataFrame
            norm_config: Normalization configuration
            experiment_config: Experiment configuration
            intsta_df: Optional internal standards DataFrame
            protein_df: Optional protein concentrations DataFrame
        
        Returns:
            Normalized DataFrame
        """
        try:
            normalized_df = self.normalization_service.normalize(
                df,
                norm_config,
                experiment_config,
                intsta_df=intsta_df,
                protein_df=protein_df
            )
            
            st.success(f"✅ Normalization complete: {len(normalized_df)} lipids normalized")
            
            return normalized_df
            
        except Exception as e:
            st.error(f"❌ Error during normalization: {str(e)}")
            raise
    
    @staticmethod
    def build_normalization_config_from_ui(
        method: str,
        selected_classes: List[str],
        internal_standards: Optional[Dict[str, str]] = None,
        intsta_concentrations: Optional[Dict[str, float]] = None,
        protein_concentrations: Optional[Dict[str, float]] = None
    ) -> NormalizationConfig:
        """
        Build NormalizationConfig from Streamlit UI inputs.
        
        Args:
            method: Normalization method
            selected_classes: List of selected lipid classes
            internal_standards: Dict mapping class to standard name
            intsta_concentrations: Dict of standard concentrations
            protein_concentrations: Dict of protein concentrations
        
        Returns:
            NormalizationConfig instance
        """
        return NormalizationConfig(
            method=method,
            selected_classes=selected_classes,
            internal_standards=internal_standards,
            intsta_concentrations=intsta_concentrations,
            protein_concentrations=protein_concentrations
        )
    
    @staticmethod
    def create_experiment_from_streamlit(
        n_conditions: int,
        conditions_list: List[str],
        number_of_samples_list: List[int]
    ) -> ExperimentConfig:
        """
        Create ExperimentConfig from Streamlit session state.
        
        Args:
            n_conditions: Number of experimental conditions
            conditions_list: List of condition names
            number_of_samples_list: List of sample counts per condition
        
        Returns:
            ExperimentConfig instance
        """
        return ExperimentConfig(
            n_conditions=n_conditions,
            conditions_list=conditions_list,
            number_of_samples_list=number_of_samples_list
        )