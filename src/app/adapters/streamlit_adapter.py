"""
Streamlit adapter layer.

Bridge between UI and services - handles session state management and caching.
This module contains all Streamlit-specific code that wraps the pure business logic services.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig
from ..services.format_detection import FormatDetectionService, DataFormat
from ..services.data_cleaning import DataCleaningService, CleaningResult, GradeFilterConfig, QualityFilterConfig
from ..services.zero_filtering import ZeroFilteringService, ZeroFilterConfig, ZeroFilteringResult
from ..services.normalization import NormalizationService, NormalizationResult
from ..services.standards import StandardsService, StandardsExtractionResult, StandardsValidationResult


@dataclass
class SessionState:
    """
    Type-safe container for all session state variables.

    Defines the complete shape of session state with default values.
    Use this as a reference for what keys exist in session state.
    """
    # Data states
    raw_df: Optional[pd.DataFrame] = None
    cleaned_df: Optional[pd.DataFrame] = None
    intsta_df: Optional[pd.DataFrame] = None
    normalized_df: Optional[pd.DataFrame] = None
    continuation_df: Optional[pd.DataFrame] = None
    original_column_order: Optional[List[str]] = None

    # Experiment configuration
    experiment: Optional[ExperimentConfig] = None
    format_type: Optional[DataFormat] = None
    bqc_label: Optional[str] = None

    # Process control
    page: str = 'landing'
    module: str = "Data Cleaning, Filtering, & Normalization"
    confirmed: bool = False
    grouping_complete: bool = True

    # Normalization settings
    normalization_method: str = 'None'
    normalization_inputs: Dict[str, Any] = field(default_factory=dict)
    selected_classes: List[str] = field(default_factory=list)
    create_norm_dataset: bool = False

    # Format-specific configurations
    grade_config: Optional[GradeFilterConfig] = None
    grade_filter_mode_saved: int = 0  # 0 = default, 1 = customize
    grade_selections_saved: Dict[str, List[str]] = field(default_factory=dict)

    msdial_quality_config: Optional[QualityFilterConfig] = None
    msdial_features: Dict[str, Any] = field(default_factory=dict)
    msdial_use_normalized: bool = False
    msdial_data_type_index: int = 0  # 0 = raw, 1 = pre-normalized
    msdial_quality_level_index: int = 1  # 0 = relaxed, 1 = moderate, 2 = strict

    # Standards settings
    original_auto_intsta_df: Optional[pd.DataFrame] = None
    preserved_intsta_df: Optional[pd.DataFrame] = None
    preserved_standards_mode: Optional[str] = None


class StreamlitAdapter:
    """
    Adapter for managing Streamlit session state and caching.

    Responsibilities:
    1. Initialize session state with default values
    2. Provide type-safe accessors for session state
    3. Provide cached wrappers for service calls
    4. Handle UI data conversion to service models

    All methods are static - no instance state is stored.
    """

    # ==================== Session State Initialization ====================

    @staticmethod
    def initialize_session_state() -> None:
        """
        Initialize all session state variables with default values.

        Call this at the beginning of main() to ensure all required
        session state keys exist before they are accessed.
        """
        defaults = SessionState()

        # Data states
        if 'raw_df' not in st.session_state:
            st.session_state.raw_df = defaults.raw_df
        if 'cleaned_df' not in st.session_state:
            st.session_state.cleaned_df = defaults.cleaned_df
        if 'intsta_df' not in st.session_state:
            st.session_state.intsta_df = defaults.intsta_df
        if 'normalized_df' not in st.session_state:
            st.session_state.normalized_df = defaults.normalized_df
        if 'continuation_df' not in st.session_state:
            st.session_state.continuation_df = defaults.continuation_df
        if 'original_column_order' not in st.session_state:
            st.session_state.original_column_order = defaults.original_column_order

        # Experiment configuration
        if 'experiment' not in st.session_state:
            st.session_state.experiment = defaults.experiment
        if 'format_type' not in st.session_state:
            st.session_state.format_type = defaults.format_type
        if 'bqc_label' not in st.session_state:
            st.session_state.bqc_label = defaults.bqc_label

        # Process control
        if 'page' not in st.session_state:
            st.session_state.page = defaults.page
        if 'module' not in st.session_state:
            st.session_state.module = defaults.module
        if 'confirmed' not in st.session_state:
            st.session_state.confirmed = defaults.confirmed
        if 'grouping_complete' not in st.session_state:
            st.session_state.grouping_complete = defaults.grouping_complete

        # Normalization settings
        if 'normalization_method' not in st.session_state:
            st.session_state.normalization_method = defaults.normalization_method
        if 'normalization_inputs' not in st.session_state:
            st.session_state.normalization_inputs = defaults.normalization_inputs
        if 'selected_classes' not in st.session_state:
            st.session_state.selected_classes = defaults.selected_classes
        if 'create_norm_dataset' not in st.session_state:
            st.session_state.create_norm_dataset = defaults.create_norm_dataset

        # LipidSearch grade configuration
        if 'grade_config' not in st.session_state:
            st.session_state.grade_config = defaults.grade_config
        if 'grade_filter_mode_saved' not in st.session_state:
            st.session_state.grade_filter_mode_saved = defaults.grade_filter_mode_saved
        if 'grade_selections_saved' not in st.session_state:
            st.session_state.grade_selections_saved = defaults.grade_selections_saved

        # MS-DIAL quality configuration
        if 'msdial_quality_config' not in st.session_state:
            st.session_state.msdial_quality_config = defaults.msdial_quality_config
        if 'msdial_features' not in st.session_state:
            st.session_state.msdial_features = defaults.msdial_features
        if 'msdial_use_normalized' not in st.session_state:
            st.session_state.msdial_use_normalized = defaults.msdial_use_normalized
        if 'msdial_data_type_index' not in st.session_state:
            st.session_state.msdial_data_type_index = defaults.msdial_data_type_index
        if 'msdial_quality_level_index' not in st.session_state:
            st.session_state.msdial_quality_level_index = defaults.msdial_quality_level_index

        # Standards settings
        if 'original_auto_intsta_df' not in st.session_state:
            st.session_state.original_auto_intsta_df = defaults.original_auto_intsta_df
        if 'preserved_intsta_df' not in st.session_state:
            st.session_state.preserved_intsta_df = defaults.preserved_intsta_df
        if 'preserved_standards_mode' not in st.session_state:
            st.session_state.preserved_standards_mode = defaults.preserved_standards_mode

    @staticmethod
    def reset_data_state() -> None:
        """Reset all data-related session state for a fresh upload."""
        st.session_state.raw_df = None
        st.session_state.cleaned_df = None
        st.session_state.intsta_df = None
        st.session_state.normalized_df = None
        st.session_state.continuation_df = None
        st.session_state.experiment = None
        st.session_state.confirmed = False
        st.session_state.grade_config = None
        st.session_state.msdial_quality_config = None
        st.session_state.original_auto_intsta_df = None

    @staticmethod
    def reset_normalization_state() -> None:
        """Reset normalization-related session state."""
        st.session_state.normalization_method = 'None'
        st.session_state.normalization_inputs = {}
        st.session_state.selected_classes = []
        st.session_state.create_norm_dataset = False
        st.session_state.normalized_df = None

    # ==================== State Accessors ====================

    @staticmethod
    def get_experiment() -> Optional[ExperimentConfig]:
        """Get current experiment configuration."""
        return st.session_state.get('experiment')

    @staticmethod
    def set_experiment(experiment: ExperimentConfig) -> None:
        """Set experiment configuration."""
        st.session_state.experiment = experiment

    @staticmethod
    def get_format_type() -> Optional[DataFormat]:
        """Get current data format type."""
        return st.session_state.get('format_type')

    @staticmethod
    def set_format_type(format_type: DataFormat) -> None:
        """Set data format type."""
        st.session_state.format_type = format_type

    @staticmethod
    def get_cleaned_df() -> Optional[pd.DataFrame]:
        """Get cleaned DataFrame."""
        return st.session_state.get('cleaned_df')

    @staticmethod
    def set_cleaned_df(df: pd.DataFrame) -> None:
        """Set cleaned DataFrame."""
        st.session_state.cleaned_df = df

    @staticmethod
    def get_intsta_df() -> Optional[pd.DataFrame]:
        """Get internal standards DataFrame."""
        return st.session_state.get('intsta_df')

    @staticmethod
    def set_intsta_df(df: pd.DataFrame) -> None:
        """Set internal standards DataFrame."""
        st.session_state.intsta_df = df

    @staticmethod
    def get_continuation_df() -> Optional[pd.DataFrame]:
        """Get continuation DataFrame (for downstream analysis)."""
        return st.session_state.get('continuation_df')

    @staticmethod
    def set_continuation_df(df: pd.DataFrame) -> None:
        """Set continuation DataFrame."""
        st.session_state.continuation_df = df

    @staticmethod
    def get_bqc_label() -> Optional[str]:
        """Get BQC condition label."""
        return st.session_state.get('bqc_label')

    @staticmethod
    def set_bqc_label(label: str) -> None:
        """Set BQC condition label."""
        st.session_state.bqc_label = label

    # ==================== Service Wrappers ====================

    @staticmethod
    def detect_format(df: pd.DataFrame) -> DataFormat:
        """
        Detect data format from DataFrame.

        This is a thin wrapper around FormatDetectionService.
        Not cached because format detection is fast (just column checks).

        Args:
            df: DataFrame to detect format for

        Returns:
            Detected DataFormat
        """
        return FormatDetectionService.detect_format(df)

    @staticmethod
    @st.cache_data(show_spinner="Cleaning data...")
    def clean_data(
        _df_hash: str,
        df: pd.DataFrame,
        experiment_dict: Dict[str, Any],
        format_type: str,
        grade_config_dict: Optional[Dict[str, Any]] = None,
        quality_config_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Cached data cleaning wrapper.

        Args:
            _df_hash: Hash of DataFrame for cache invalidation
            df: Raw DataFrame to clean
            experiment_dict: Serialized ExperimentConfig
            format_type: Format type string
            grade_config_dict: Serialized GradeFilterConfig
            quality_config_dict: Serialized QualityFilterConfig

        Returns:
            Tuple of (cleaned_df, internal_standards_df, messages)
        """
        # Reconstruct objects from dicts
        experiment = ExperimentConfig(**experiment_dict)
        data_format = DataFormat(format_type)

        grade_config = GradeFilterConfig(**grade_config_dict) if grade_config_dict else None
        quality_config = QualityFilterConfig(**quality_config_dict) if quality_config_dict else None

        result = DataCleaningService.clean_data(
            df=df,
            experiment=experiment,
            data_format=data_format,
            grade_config=grade_config,
            quality_config=quality_config
        )

        return result.cleaned_df, result.internal_standards_df, result.filter_messages

    @staticmethod
    @st.cache_data(show_spinner="Filtering zero values...")
    def filter_zeros(
        _df_hash: str,
        df: pd.DataFrame,
        experiment_dict: Dict[str, Any],
        bqc_label: Optional[str],
        detection_threshold: float = 0.0,
        bqc_threshold: float = 0.5,
        non_bqc_threshold: float = 0.75
    ) -> Tuple[pd.DataFrame, List[str], int, int]:
        """
        Cached zero filtering wrapper.

        Args:
            _df_hash: Hash of DataFrame for cache invalidation
            df: Cleaned DataFrame to filter
            experiment_dict: Serialized ExperimentConfig
            bqc_label: BQC condition label
            detection_threshold: Value threshold for "zero"
            bqc_threshold: BQC proportion threshold
            non_bqc_threshold: Non-BQC proportion threshold

        Returns:
            Tuple of (filtered_df, removed_species, species_before, species_after)
        """
        experiment = ExperimentConfig(**experiment_dict)
        config = ZeroFilterConfig(
            detection_threshold=detection_threshold,
            bqc_threshold=bqc_threshold,
            non_bqc_threshold=non_bqc_threshold
        )

        result = ZeroFilteringService.filter(df, experiment, config, bqc_label)

        return (
            result.filtered_df,
            result.removed_species,
            result.species_before,
            result.species_after
        )

    @staticmethod
    @st.cache_data(show_spinner="Normalizing data...")
    def normalize_data(
        _df_hash: str,
        df: pd.DataFrame,
        config_dict: Dict[str, Any],
        experiment_dict: Dict[str, Any],
        intsta_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Cached normalization wrapper.

        Args:
            _df_hash: Hash of DataFrame for cache invalidation
            df: Cleaned DataFrame to normalize
            config_dict: Serialized NormalizationConfig
            experiment_dict: Serialized ExperimentConfig
            intsta_df: Internal standards DataFrame

        Returns:
            Tuple of (normalized_df, removed_standards, method_applied)
        """
        config = NormalizationConfig(**config_dict)
        experiment = ExperimentConfig(**experiment_dict)

        result = NormalizationService.normalize(
            df=df,
            config=config,
            experiment=experiment,
            intsta_df=intsta_df
        )

        return result.normalized_df, result.removed_standards, result.method_applied

    # ==================== Utility Methods ====================

    @staticmethod
    def compute_df_hash(df: pd.DataFrame) -> str:
        """
        Compute a hash for a DataFrame for cache invalidation.

        Args:
            df: DataFrame to hash

        Returns:
            String hash of the DataFrame
        """
        import hashlib
        # Use shape + column names + first/last values for quick hash
        content = f"{df.shape}|{list(df.columns)}|{df.iloc[0].tolist() if len(df) > 0 else []}|{df.iloc[-1].tolist() if len(df) > 0 else []}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def experiment_to_dict(experiment: ExperimentConfig) -> Dict[str, Any]:
        """
        Convert ExperimentConfig to dict for caching.

        Args:
            experiment: ExperimentConfig instance

        Returns:
            Dict representation
        """
        return {
            'n_conditions': experiment.n_conditions,
            'conditions_list': experiment.conditions_list,
            'number_of_samples_list': experiment.number_of_samples_list,
        }

    @staticmethod
    def config_to_dict(config: Any) -> Optional[Dict[str, Any]]:
        """
        Convert a dataclass config to dict for caching.

        Args:
            config: Dataclass instance or None

        Returns:
            Dict representation or None
        """
        if config is None:
            return None
        from dataclasses import asdict
        return asdict(config)
