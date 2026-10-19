"""
Per-format standardization: one module per supported upload format, each
turning a raw upload into the common schema (LipidMolec, ClassKey,
intensity[s1..sN]). ``DataStandardizationService`` in
``services/data_standardization.py`` is the entry point.
"""
