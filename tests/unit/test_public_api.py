"""Smoke tests for public API barrel files.

Verifies that all expected names are importable from:
- app.services (services/__init__.py)
- app.services.plotting (plotting/__init__.py)

Prevents accidental export regressions when refactoring.
"""
import pytest


# =============================================================================
# Services Public API
# =============================================================================


class TestServicesPublicAPI:
    """Verify app.services exports all expected names."""

    @pytest.mark.parametrize("name", [
        'ServiceError',
        'ServiceConfigurationError',
        'ServiceEmptyDataError',
        'ServiceValidationError',
        'FormatDetectionService',
        'DataFormat',
        'DataCleaningService',
        'GradeFilterConfig',
        'QualityFilterConfig',
        'CleaningResult',
        'DataCleaningError',
        'ConfigurationError',
        'EmptyDataError',
        'BaseDataCleaner',
        'LipidSearchCleaner',
        'MSDIALCleaner',
        'GenericCleaner',
        'ZeroFilteringService',
        'ZeroFilterConfig',
        'ZeroFilteringResult',
        'NormalizationService',
        'NormalizationResult',
        'StandardsService',
        'StandardsExtractionResult',
        'StandardsValidationResult',
        'StandardsProcessingResult',
        'QualityCheckService',
        'BoxPlotResult',
        'BQCPrepareResult',
        'BQCFilterResult',
        'RetentionTimeDataResult',
        'CorrelationResult',
        'PCAResult',
        'SampleRemovalResult',
    ])
    def test_services_exports(self, name):
        """Each name in __all__ should be importable from app.services."""
        import app.services as svc
        assert hasattr(svc, name), f"app.services missing export: {name}"

    def test_all_matches_dir(self):
        """__all__ should list every public export."""
        import app.services as svc
        for name in svc.__all__:
            assert hasattr(svc, name), f"__all__ lists '{name}' but it's not defined"


# =============================================================================
# Plotting Public API
# =============================================================================


class TestPlottingPublicAPI:
    """Verify app.services.plotting exports all expected names."""

    @pytest.mark.parametrize("name", [
        'BarChartPlotterService',
        'FACHPlotterService',
        'LipidomicHeatmapPlotterService',
        'PathwayVizPlotterService',
        'PieChartPlotterService',
        'BoxPlotService',
        'BQCPlotterService',
        'CorrelationPlotterService',
        'PCAPlotterService',
        'RetentionTimePlotterService',
        'SaturationPlotterService',
        'StandardsPlotterService',
        'VolcanoPlotterService',
    ])
    def test_plotting_exports(self, name):
        """Each name in __all__ should be importable from app.services.plotting."""
        import app.services.plotting as plt_svc
        assert hasattr(plt_svc, name), f"app.services.plotting missing export: {name}"

    def test_plotting_all_matches_dir(self):
        """__all__ should list every public export."""
        import app.services.plotting as plt_svc
        for name in plt_svc.__all__:
            assert hasattr(plt_svc, name), (
                f"__all__ lists '{name}' but it's not defined"
            )

    @pytest.mark.parametrize("name", [
        'generate_class_color_mapping',
        'generate_condition_color_mapping',
        'get_effective_p_value',
        'p_value_to_marker',
        'validate_dataframe',
    ])
    def test_shared_utilities_exported(self, name):
        """Shared plotting utilities should be importable from the package."""
        import app.services.plotting as plt_svc
        assert hasattr(plt_svc, name), (
            f"app.services.plotting missing shared utility: {name}"
        )