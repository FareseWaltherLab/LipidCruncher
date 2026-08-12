"""
R runtime detection for the isotope-correction service.

Locates ``Rscript`` and checks that the IsoCorrectoR package can be loaded, so
the rest of the app degrades gracefully where R is absent (e.g. local machines
without R). No Streamlit imports — pure environment probing.
"""
import shutil
import subprocess
from typing import Optional

from .exceptions import RRuntimeError

# Short timeout: this only loads a package and prints a version string.
_PROBE_TIMEOUT_SECONDS = 30


def find_rscript() -> Optional[str]:
    """Return the path to ``Rscript`` on PATH, or None if not found."""
    return shutil.which("Rscript")


def check_isocorrector_available() -> bool:
    """Return True iff Rscript exists and IsoCorrectoR + jsonlite can load.

    Swallows all failures and returns False — this is a capability probe, not
    an operation, so callers can branch on availability without try/except.
    """
    rscript = find_rscript()
    if rscript is None:
        return False
    try:
        proc = subprocess.run(
            [
                rscript,
                "-e",
                'quit(status = !(requireNamespace("IsoCorrectoR", quietly=TRUE) '
                '&& requireNamespace("jsonlite", quietly=TRUE)))',
            ],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        return proc.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def require_isocorrector() -> str:
    """Return the Rscript path, or raise RRuntimeError if R/IsoCorrectoR is unusable."""
    rscript = find_rscript()
    if rscript is None:
        raise RRuntimeError(
            "Rscript was not found on PATH. The stable isotope tracing module "
            "requires R with the IsoCorrectoR package installed."
        )
    if not check_isocorrector_available():
        raise RRuntimeError(
            "R is available but the IsoCorrectoR (and/or jsonlite) package could "
            "not be loaded. Install them in the R runtime to use isotope tracing."
        )
    return rscript
