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
    """Raised when Rscript / IsoCorrectoR is missing or the R run fails.

    This is an environmental failure, not a user-input problem. The UI should
    surface it as an "R runtime unavailable" notice so the rest of the app
    keeps working on hosts without R.
    """
    pass


class IsotopeInputError(IsotopeError):
    """Raised when the three input files fail structural validation.

    User-fixable: missing columns, molecule-name mismatch, non-numeric
    intensities, etc. Surfaced before R is ever invoked.
    """
    pass
