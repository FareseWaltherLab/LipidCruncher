"""Unit tests for the shared exception hierarchy in app/services/exceptions.py.

Covers: inheritance chain, message propagation, catchability by parent types,
and the data cleaning sub-hierarchy in app/services/data_cleaning/exceptions.py.
"""
import pytest

from app.services.exceptions import (
    ConfigurationError,
    EmptyDataError,
    ServiceError,
    ValidationError,
)
from app.services.data_cleaning.exceptions import (
    ConfigurationError as CleaningConfigurationError,
    DataCleaningError,
    EmptyDataError as CleaningEmptyDataError,
)


# =============================================================================
# ServiceError Hierarchy
# =============================================================================


class TestServiceError:
    """Tests for the base ServiceError class."""

    def test_is_value_error(self):
        """ServiceError inherits from ValueError for backward compatibility."""
        assert issubclass(ServiceError, ValueError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ServiceError, match="test message"):
            raise ServiceError("test message")

    def test_caught_as_value_error(self):
        """Existing code catching ValueError should still work."""
        with pytest.raises(ValueError):
            raise ServiceError("test")

    def test_message_preserved(self):
        err = ServiceError("specific message")
        assert str(err) == "specific message"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_is_service_error(self):
        assert issubclass(ConfigurationError, ServiceError)

    def test_is_value_error(self):
        assert issubclass(ConfigurationError, ValueError)

    def test_caught_by_service_error(self):
        with pytest.raises(ServiceError):
            raise ConfigurationError("bad config")

    def test_not_caught_by_sibling(self):
        """ConfigurationError should not be caught by EmptyDataError."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("bad config")
        # Ensure it's NOT an EmptyDataError
        assert not issubclass(ConfigurationError, EmptyDataError)


class TestEmptyDataError:
    """Tests for EmptyDataError."""

    def test_is_service_error(self):
        assert issubclass(EmptyDataError, ServiceError)

    def test_caught_by_service_error(self):
        with pytest.raises(ServiceError):
            raise EmptyDataError("no data")

    def test_not_configuration_error(self):
        assert not issubclass(EmptyDataError, ConfigurationError)


class TestValidationError:
    """Tests for ValidationError."""

    def test_is_service_error(self):
        assert issubclass(ValidationError, ServiceError)

    def test_caught_by_service_error(self):
        with pytest.raises(ServiceError):
            raise ValidationError("invalid data")

    def test_distinct_from_siblings(self):
        assert not issubclass(ValidationError, ConfigurationError)
        assert not issubclass(ValidationError, EmptyDataError)


# =============================================================================
# Data Cleaning Sub-Hierarchy
# =============================================================================


class TestDataCleaningError:
    """Tests for DataCleaningError."""

    def test_is_service_error(self):
        assert issubclass(DataCleaningError, ServiceError)

    def test_is_value_error(self):
        assert issubclass(DataCleaningError, ValueError)

    def test_can_be_raised(self):
        with pytest.raises(DataCleaningError, match="cleaning failed"):
            raise DataCleaningError("cleaning failed")


class TestCleaningConfigurationError:
    """Tests for data_cleaning.ConfigurationError (diamond inheritance)."""

    def test_is_data_cleaning_error(self):
        assert issubclass(CleaningConfigurationError, DataCleaningError)

    def test_is_service_configuration_error(self):
        """Should also be caught by the top-level ConfigurationError."""
        assert issubclass(CleaningConfigurationError, ConfigurationError)

    def test_caught_by_either_parent(self):
        """Should be catchable as both DataCleaningError and ConfigurationError."""
        with pytest.raises(DataCleaningError):
            raise CleaningConfigurationError("strict filter")
        with pytest.raises(ConfigurationError):
            raise CleaningConfigurationError("strict filter")

    def test_caught_by_service_error(self):
        with pytest.raises(ServiceError):
            raise CleaningConfigurationError("strict filter")

    def test_caught_by_value_error(self):
        with pytest.raises(ValueError):
            raise CleaningConfigurationError("strict filter")


class TestCleaningEmptyDataError:
    """Tests for data_cleaning.EmptyDataError (diamond inheritance)."""

    def test_is_data_cleaning_error(self):
        assert issubclass(CleaningEmptyDataError, DataCleaningError)

    def test_is_service_empty_data_error(self):
        assert issubclass(CleaningEmptyDataError, EmptyDataError)

    def test_caught_by_either_parent(self):
        with pytest.raises(DataCleaningError):
            raise CleaningEmptyDataError("empty upload")
        with pytest.raises(EmptyDataError):
            raise CleaningEmptyDataError("empty upload")


# =============================================================================
# Cross-Hierarchy Discrimination
# =============================================================================


class TestExceptionDiscrimination:
    """Verify that except clauses can distinguish between exception types."""

    def test_catch_specific_before_general(self):
        """Specific exception types should be distinguishable."""
        try:
            raise CleaningConfigurationError("too strict")
        except CleaningConfigurationError:
            caught = 'cleaning_config'
        except ConfigurationError:
            caught = 'config'
        except ServiceError:
            caught = 'service'
        assert caught == 'cleaning_config'

    def test_unrelated_exceptions_not_caught(self):
        """ValidationError should not be caught by EmptyDataError handler."""
        with pytest.raises(ValidationError):
            try:
                raise ValidationError("bad columns")
            except EmptyDataError:
                pass  # Should not catch

    @pytest.mark.parametrize("exc_class,expected_parents", [
        (ServiceError, [ValueError]),
        (ConfigurationError, [ServiceError, ValueError]),
        (EmptyDataError, [ServiceError, ValueError]),
        (ValidationError, [ServiceError, ValueError]),
        (DataCleaningError, [ServiceError, ValueError]),
        (CleaningConfigurationError, [DataCleaningError, ConfigurationError, ServiceError]),
        (CleaningEmptyDataError, [DataCleaningError, EmptyDataError, ServiceError]),
    ])
    def test_full_inheritance_chain(self, exc_class, expected_parents):
        """Each exception should be a subclass of all expected parents."""
        for parent in expected_parents:
            assert issubclass(exc_class, parent), (
                f"{exc_class.__name__} should be subclass of {parent.__name__}"
            )