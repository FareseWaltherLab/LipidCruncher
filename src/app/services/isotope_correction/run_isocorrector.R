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
# corrected CSVs are written to the output directory. When IsoCorrectoR rejects
# the inputs it prints "ISOCORRECTOR_ERROR: <reason>" to stderr and exits 1, so
# the Python layer can show the reason rather than the whole of stderr.

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

# readr otherwise prints a column specification for every input file, which
# buries the one line that matters when something goes wrong.
options(readr.show_col_types = FALSE)

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

# "WARNINGS" means the correction finished and every output was written,
# but IsoCorrectoR logged a warning (e.g. a missing intensity or a missing
# isotopologue row); the Python side reads those from the result log.
if (!(as.character(result$success) %in% c("TRUE", "WARNINGS"))) {
  cat("ISOCORRECTOR_ERROR: ", paste(as.character(result$error), collapse = " "),
      "\n", sep = "", file = stderr())
  quit(status = 1)
}

cat("ISOCORRECTOR_OK\n")
