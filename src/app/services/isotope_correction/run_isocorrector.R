#!/usr/bin/env Rscript
#
# Trusted IsoCorrectoR wrapper for LipidCruncher.
#
# This script is BUNDLED and TRUSTED — it is part of the application, not
# user-supplied. The user provides *data files*, never code. It is invoked as a
# subprocess by IsotopeCorrectionService with a single argument: the path to a
# JSON file describing the input file paths, the IsoCorrection() settings, and
# the output directory. Passing paths/settings via a JSON file (not shell
# interpolation) keeps the call injection-free.
#
# On success it prints "ISOCORRECTOR_OK" on the last line and exits 0; the
# corrected CSVs are written to the output directory. On failure it prints the
# error to stderr and exits non-zero so the Python layer can surface it.

suppressMessages({
  ok <- requireNamespace("IsoCorrectoR", quietly = TRUE) &&
        requireNamespace("jsonlite", quietly = TRUE)
})
if (!ok) {
  stop("IsoCorrectoR and jsonlite must be installed in the R runtime.")
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: run_isocorrector.R <config.json>")
}

cfg <- jsonlite::fromJSON(args[[1]])

result <- IsoCorrectoR::IsoCorrection(
  MeasurementFile          = cfg$measurement_file,
  MoleculeFile             = cfg$molecule_file,
  ElementFile              = cfg$element_file,
  CorrectTracerImpurity    = isTRUE(cfg$correct_tracer_impurity),
  CorrectTracerElementCore = isTRUE(cfg$correct_tracer_element_core),
  CalculateMeanEnrichment  = isTRUE(cfg$calculate_mean_enrichment),
  UltraHighRes             = isTRUE(cfg$ultra_high_res),
  CorrectAlsoMonoisotopic  = isTRUE(cfg$correct_also_monoisotopic),
  CalculationThreshold_UHR = cfg$calculation_threshold_uhr,
  DirOut                   = cfg$output_dir,
  # IsoCorrectoR prepends "IsoCorrectoR_" to FileOut, so this yields
  # "IsoCorrectoR_result_*.csv" — matching the golden output naming.
  FileOut                  = "result",
  FileOutFormat            = "csv",
  ReturnResultsObject      = FALSE,
  verbose                  = FALSE
)

if (!identical(result$success, "TRUE") && !isTRUE(result$success)) {
  stop("IsoCorrectoR reported failure; see the result log in the output directory.")
}

cat("ISOCORRECTOR_OK\n")
