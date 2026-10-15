"""
Typed exceptions for the isotope-correction service.

Lets the UI distinguish "the R runtime / IsoCorrectoR is unavailable or crashed"
(environmental — show a graceful notice) from "your input files are malformed"
(user-fixable) without string-matching error messages.
"""


class IsotopeError(Exception):
    """Base exception for the isotope-correction service."""
    pass


class RRuntimeError(IsotopeError):
    """Raised when Rscript / IsoCorrectoR is missing or the R run fails for a
    reason IsoCorrectoR did not attribute to the inputs (a crash, a timeout).
    """
    pass


class IsotopeInputError(IsotopeError):
    """Raised when IsoCorrectoR rejects the input files.

    User-fixable, e.g. a formula element missing from the Element file.
    (Structural problems are caught earlier by ``validate_inputs``, which
    returns messages rather than raising.)
    """
    pass
