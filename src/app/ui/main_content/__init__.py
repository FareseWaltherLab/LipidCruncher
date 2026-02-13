"""
Main content UI components for LipidCruncher.

This package contains extracted main content area components from main_app.py:
- data_processing: Processing docs, grade filtering, MS-DIAL config, quality filtering
- internal_standards: Internal standards management (auto-detect, upload, plots)
- normalization: Normalization UI (class selection, IS config, protein config, execution)
"""

from app.ui.main_content.data_processing import (
    display_data_processing_docs,
    display_grade_filtering_config,
    display_msdial_data_type_selection,
    display_quality_filtering_config,
    build_filter_configs,
    run_ingestion_pipeline,
    display_final_filtered_data,
)
from app.ui.main_content.internal_standards import (
    display_manage_internal_standards,
)
from app.ui.main_content.normalization import (
    display_normalization_ui,
)

__all__ = [
    # data_processing
    'display_data_processing_docs',
    'display_grade_filtering_config',
    'display_msdial_data_type_selection',
    'display_quality_filtering_config',
    'build_filter_configs',
    'run_ingestion_pipeline',
    'display_final_filtered_data',
    # internal_standards
    'display_manage_internal_standards',
    # normalization
    'display_normalization_ui',
]
