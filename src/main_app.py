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

import streamlit as st

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="LipidCruncher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Imports (uncomment as components are extracted)
# =============================================================================

# Phase 2: Models
# from app.models import ExperimentConfig, NormalizationConfig

# Phase 3: Services
# from app.services import FormatDetectionService
# from app.services import DataCleaningService
# from app.services import NormalizationService

# Phase 4: Workflows & UI
# from app.workflows import DataPipelineWorkflow
# from app.ui import FileUploadComponent

# =============================================================================
# Main Application
# =============================================================================

st.title("LipidCruncher")
st.caption("Lipidomics Data Analysis Application")

st.info("🚧 Refactoring in progress. Features will be added incrementally.")

# TODO: Add file upload
# TODO: Add format detection
# TODO: Add data cleaning
# TODO: Add normalization
# TODO: Add statistical analysis
# TODO: Add visualization
