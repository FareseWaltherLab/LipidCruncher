"""
Sidebar UI components for LipidCruncher.

This package contains extracted sidebar components from main_app.py:
- file_upload: Format selection, sample data loading, file upload
- column_mapping: Column standardization and MS-DIAL sample override
- experiment_config: Experiment definition (conditions, samples)
- sample_grouping: Sample grouping and manual regrouping
- confirm_inputs: BQC specification and input confirmation
"""

from app.ui.sidebar.file_upload import (
    display_format_selection,
    load_sample_dataset,
    display_file_upload,
)
from app.ui.sidebar.column_mapping import (
    standardize_uploaded_data,
    display_column_mapping,
)
from app.ui.sidebar.experiment_config import (
    detect_sample_columns,
    extract_sample_names,
    display_experiment_definition,
)
from app.ui.sidebar.sample_grouping import (
    display_group_samples,
    display_sample_grouping,
)
from app.ui.sidebar.confirm_inputs import (
    display_bqc_section,
    display_confirm_inputs,
)

__all__ = [
    # file_upload
    'display_format_selection',
    'load_sample_dataset',
    'display_file_upload',
    # column_mapping
    'standardize_uploaded_data',
    'display_column_mapping',
    # experiment_config
    'detect_sample_columns',
    'extract_sample_names',
    'display_experiment_definition',
    # sample_grouping
    'display_group_samples',
    'display_sample_grouping',
    # confirm_inputs
    'display_bqc_section',
    'display_confirm_inputs',
]
