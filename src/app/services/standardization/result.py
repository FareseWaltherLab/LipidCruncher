"""
The result every format's standardization returns.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from app.services.standardization.msdial import MSDIALFeatures


@dataclass
class StandardizationResult:
    """Result of validating and standardizing a dataset.

    Attributes:
        success: Whether standardization succeeded.
        message: Status or error message.
        standardized_df: The standardized DataFrame (None on failure).
        column_mapping: DataFrame with 'standardized_name' and 'original_name' columns.
        n_intensity_cols: Number of intensity columns in the standardized data.
        msdial_features: MS-DIAL–specific feature dict (None for other formats).
        msdial_sample_names: Map of s1->original_name for MS-DIAL (None for others).
        workbench_conditions: Map of s1->condition for Metabolomics Workbench.
        workbench_samples: Map of s1->sample_name for Metabolomics Workbench.
    """
    success: bool
    message: str
    standardized_df: Optional[pd.DataFrame] = None
    column_mapping: Optional[pd.DataFrame] = None
    n_intensity_cols: int = 0
    msdial_features: Optional["MSDIALFeatures"] = None
    msdial_sample_names: Optional[Dict[str, str]] = None
    workbench_conditions: Optional[Dict[str, str]] = None
    workbench_samples: Optional[Dict[str, str]] = None
    # LipidSearch 5.2 dual-polarity alignment: experiment metadata read from
    # the Alignment Setting file. (The sidebar pre-fills the experiment from
    # its own parse of the file; these fields are not read by the UI.)
    lipidsearch_conditions: Optional[List[str]] = None
    lipidsearch_samples_per_condition: Optional[List[int]] = None
    # Map of s1->raw-filename-derived sample name from the alignment file.
    lipidsearch_sample_names: Optional[Dict[str, str]] = None
