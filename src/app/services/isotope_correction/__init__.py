"""Stable isotope tracing — natural-isotope-abundance correction via IsoCorrectoR."""
from .service import IsotopeCorrectionService, CorrectionResult
from .validation import validate_inputs
from .r_runtime import (
    find_rscript,
    check_isocorrector_available,
    require_isocorrector,
)
from .exceptions import IsotopeError, RRuntimeError, IsotopeInputError

__all__ = [
    "IsotopeCorrectionService",
    "CorrectionResult",
    "validate_inputs",
    "find_rscript",
    "check_isocorrector_available",
    "require_isocorrector",
    "IsotopeError",
    "RRuntimeError",
    "IsotopeInputError",
]
