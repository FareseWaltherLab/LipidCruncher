"""
LipidCruncher - Lipidomics Data Analysis Application

This is the refactored entry point. During the transition period,
it imports from old_main_app.py. As services are extracted, this file
will be updated to use the new architecture:

    UI Layer (this file)
        → Workflows (app/workflows/)
            → Adapters (app/adapters/)
                → Services (app/services/)
                    → Models (app/models/)

Target: < 500 lines after refactoring is complete.
"""

# =============================================================================
# TRANSITION PERIOD: Import and run the old monolithic app
# As we extract services, we'll gradually replace these imports
# =============================================================================

# Import everything from old_main_app to maintain functionality
from old_main_app import *

# =============================================================================
# REFACTORED IMPORTS (uncomment as services are extracted)
# =============================================================================

# Phase 2: Models
# from app.models import ExperimentConfig, NormalizationConfig

# Phase 3: Services
# from app.services import FormatDetectionService
# from app.services import DataCleaningService
# from app.services import ZeroFilteringService
# from app.services import NormalizationService
# from app.services import StandardsService

# Phase 4: Workflows & Adapters
# from app.adapters import StreamlitAdapter
# from app.workflows import DataPipelineWorkflow

# Phase 4: UI Components
# from app.ui import FileUploadComponent
# from app.ui import GroupingComponent
# from app.ui import NormalizationComponent

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Currently runs the old app via the wildcard import above
    # After refactoring, this will orchestrate the new architecture
    pass
