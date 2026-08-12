"""
IsotopeCorrectionService — runs the real IsoCorrectoR via a trusted R subprocess.

Flow: write the three input DataFrames to a temp working directory, invoke the
bundled ``run_isocorrector.R`` wrapper with a JSON config, then parse the
CSV outputs (written by IsoCorrectoR into a timestamped subdirectory) back into
DataFrames.

Security boundary (the R code is OURS and trusted; the user supplies data, not
code — so the AST sandbox does not apply here): we (a) validate inputs upstream,
(b) pass file paths as a JSON config argument — never shell-interpolated — and
(c) run in an isolated temp directory. No Streamlit imports.
"""
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app.models.isotope_tracing import IsotopeCorrectionConfig
from .exceptions import RRuntimeError
from .r_runtime import require_isocorrector

# Path to the bundled, trusted R wrapper (lives next to this module).
_WRAPPER_PATH = os.path.join(os.path.dirname(__file__), "run_isocorrector.R")

# IsoCorrectoR can be slow on large measurement matrices; cap generously.
_RUN_TIMEOUT_SECONDS = 600

# Output file stem written by the wrapper (FileOut = "IsoCorrectoR_result").
_RESULT_STEM = "IsoCorrectoR_result"


@dataclass
class CorrectionResult:
    """Successful IsoCorrectoR run: the parsed output tables plus the run log.

    Tables are None when IsoCorrectoR did not produce them for the given
    settings (e.g. ``mean_enrichment`` when CalculateMeanEnrichment is False).
    """
    corrected: pd.DataFrame
    raw_data: pd.DataFrame
    corrected_fractions: Optional[pd.DataFrame] = None
    mean_enrichment: Optional[pd.DataFrame] = None
    residuals: Optional[pd.DataFrame] = None
    relative_residuals: Optional[pd.DataFrame] = None
    log_text: str = ""


# Maps a CorrectionResult field to the output-file suffix IsoCorrectoR writes.
_OUTPUT_SUFFIXES = {
    "corrected": "Corrected",
    "raw_data": "RawData",
    "corrected_fractions": "CorrectedFractions",
    "mean_enrichment": "MeanEnrichment",
    "residuals": "Residuals",
    "relative_residuals": "RelativeResiduals",
}


class IsotopeCorrectionService:
    """Runs natural-isotope-abundance correction. All methods are static."""

    @staticmethod
    def run_correction(
        measurement_df: pd.DataFrame,
        molecule_df: pd.DataFrame,
        element_df: pd.DataFrame,
        config: IsotopeCorrectionConfig,
    ) -> CorrectionResult:
        """Run IsoCorrectoR on the three input tables and return parsed outputs.

        Args:
            measurement_df: Measurement file as a DataFrame.
            molecule_df: Molecule file as a DataFrame.
            element_df: Element file as a DataFrame.
            config: IsoCorrectoR settings.

        Returns:
            CorrectionResult with the parsed output tables and run log.

        Raises:
            RRuntimeError: If R/IsoCorrectoR is unavailable or the run fails.
        """
        rscript = require_isocorrector()

        with tempfile.TemporaryDirectory(prefix="lc_isocorrector_") as work_dir:
            measurement_path = os.path.join(work_dir, "MeasurementFile.csv")
            molecule_path = os.path.join(work_dir, "MoleculeFile.csv")
            element_path = os.path.join(work_dir, "ElementFile.csv")
            output_dir = os.path.join(work_dir, "out")
            os.makedirs(output_dir, exist_ok=True)

            # index=False reproduces the original CSV layout (first column is the
            # ID/name column IsoCorrectoR reads positionally).
            measurement_df.to_csv(measurement_path, index=False)
            molecule_df.to_csv(molecule_path, index=False)
            element_df.to_csv(element_path, index=False)

            config_path = os.path.join(work_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "measurement_file": measurement_path,
                        "molecule_file": molecule_path,
                        "element_file": element_path,
                        "correct_tracer_impurity": config.correct_tracer_impurity,
                        "correct_tracer_element_core": config.correct_tracer_element_core,
                        "calculate_mean_enrichment": config.calculate_mean_enrichment,
                        "ultra_high_res": config.ultra_high_res,
                        "correct_also_monoisotopic": config.correct_also_monoisotopic,
                        "calculation_threshold_uhr": config.calculation_threshold_uhr,
                        "output_dir": output_dir,
                    },
                    f,
                )

            try:
                proc = subprocess.run(
                    [rscript, _WRAPPER_PATH, config_path],
                    capture_output=True,
                    text=True,
                    timeout=_RUN_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                raise RRuntimeError(
                    f"IsoCorrectoR did not finish within {_RUN_TIMEOUT_SECONDS} seconds."
                )
            except OSError as e:
                raise RRuntimeError(f"Failed to launch Rscript: {e}")

            if proc.returncode != 0 or "ISOCORRECTOR_OK" not in proc.stdout:
                raise RRuntimeError(
                    "IsoCorrectoR run failed.\n"
                    f"stderr:\n{proc.stderr.strip()}"
                )

            return IsotopeCorrectionService._parse_outputs(output_dir)

    @staticmethod
    def _parse_outputs(output_dir: str) -> CorrectionResult:
        """Parse the CSVs IsoCorrectoR wrote into the timestamped result subdir."""
        result_dir = IsotopeCorrectionService._locate_result_dir(output_dir)

        tables = {}
        for field_name, suffix in _OUTPUT_SUFFIXES.items():
            path = os.path.join(result_dir, f"{_RESULT_STEM}_{suffix}.csv")
            tables[field_name] = (
                pd.read_csv(path, index_col=0) if os.path.exists(path) else None
            )

        log_path = os.path.join(result_dir, f"{_RESULT_STEM}.log")
        log_text = ""
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_text = f.read()

        if tables["corrected"] is None or tables["raw_data"] is None:
            raise RRuntimeError(
                "IsoCorrectoR finished but the expected output files were not found "
                f"in {result_dir}."
            )

        return CorrectionResult(log_text=log_text, **tables)

    @staticmethod
    def _locate_result_dir(output_dir: str) -> str:
        """Return the timestamped subdir IsoCorrectoR created under output_dir.

        IsoCorrectoR writes results into a single ``YYYY-MM-DD_HHMMSS`` subfolder
        of DirOut. We isolate DirOut per run, so there is exactly one.
        """
        subdirs = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, name))
        ]
        if not subdirs:
            raise RRuntimeError(
                f"IsoCorrectoR produced no output directory under {output_dir}."
            )
        # Newest, in case anything stray exists; normally there is exactly one.
        return max(subdirs, key=os.path.getmtime)
