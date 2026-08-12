"""
Stable isotope tracing configuration model.

Mirrors the IsoCorrectoR ``IsoCorrection()`` settings one-to-one. Frozen
Pydantic so it is hashable and safe to pass to ``@st.cache_data`` (same
convention as the other config models).
"""
from pydantic import BaseModel, Field


class IsotopeCorrectionConfig(BaseModel):
    """
    Settings for a natural-isotope-abundance correction run.

    Each field maps directly to an ``IsoCorrection()`` argument. Defaults match
    IsoCorrectoR's own defaults except ``correct_tracer_impurity``, which we
    default to True to match the superuser's golden-dataset run.

    Attributes:
        correct_tracer_impurity: Correct for tracer (label) impurity.
        correct_tracer_element_core: Correct the tracer element in the core molecule.
        calculate_mean_enrichment: Also compute mean enrichment output.
        ultra_high_res: Ultra-high-resolution correction mode.
        correct_also_monoisotopic: Apply correction to the monoisotopic peak too.
        calculation_threshold_uhr: Limit value used by the UHR calculation.
    """
    model_config = {"frozen": True}

    correct_tracer_impurity: bool = True
    correct_tracer_element_core: bool = True
    calculate_mean_enrichment: bool = True
    ultra_high_res: bool = False
    correct_also_monoisotopic: bool = False
    calculation_threshold_uhr: float = Field(default=8.0, ge=0.0)
