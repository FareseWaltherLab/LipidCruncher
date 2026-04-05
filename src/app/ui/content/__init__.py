"""Static content modules for UI documentation and help text."""

from .sample_data import SAMPLE_DATA_INFO, get_sample_data_info
from .processing_docs import PROCESSING_DOCS, ZERO_FILTERING_DOCS, get_processing_docs
from .normalization_docs import NORMALIZATION_METHODS_DOCS
from .standards_help import STANDARDS_EXTRACT_HELP, STANDARDS_COMPLETE_HELP, PROTEIN_CSV_HELP
from .analysis_docs import STATISTICAL_TESTING_DOCS, SATURATION_PROFILE_DOCS

__all__ = [
    'SAMPLE_DATA_INFO',
    'get_sample_data_info',
    'PROCESSING_DOCS',
    'ZERO_FILTERING_DOCS',
    'get_processing_docs',
    'NORMALIZATION_METHODS_DOCS',
    'STANDARDS_EXTRACT_HELP',
    'STANDARDS_COMPLETE_HELP',
    'PROTEIN_CSV_HELP',
    'STATISTICAL_TESTING_DOCS',
    'SATURATION_PROFILE_DOCS',
]
