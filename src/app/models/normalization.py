"""
Normalization configuration model.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, field_validator, model_validator


class NormalizationConfig(BaseModel):
    """
    Configuration for data normalization.

    Supports internal standards, protein-based, or combined normalization.

    Attributes:
        method: Normalization method to apply
        selected_classes: Lipid classes to normalize (filters the dataset)
        internal_standards: Mapping of lipid class -> standard lipid molecule name
        intsta_concentrations: Concentration of each internal standard (in µM or similar)
        protein_concentrations: Protein concentration for each sample (for BCA normalization)
    """
    model_config = {"frozen": True}

    method: Literal['none', 'internal_standard', 'protein', 'both'] = 'none'
    selected_classes: List[str] = []
    internal_standards: Optional[Dict[str, str]] = None
    intsta_concentrations: Optional[Dict[str, float]] = None
    protein_concentrations: Optional[Dict[str, float]] = None

    def __hash__(self) -> int:
        return hash((
            self.method,
            tuple(self.selected_classes),
            tuple(sorted(self.internal_standards.items())) if self.internal_standards else None,
            tuple(sorted(self.intsta_concentrations.items())) if self.intsta_concentrations else None,
            tuple(sorted(self.protein_concentrations.items())) if self.protein_concentrations else None,
        ))

    @field_validator('intsta_concentrations')
    @classmethod
    def validate_intsta_concentrations_positive(cls, v):
        """Ensure all internal standard concentrations are positive."""
        if v is not None:
            for standard, conc in v.items():
                if conc <= 0:
                    raise ValueError(
                        f"Internal standard concentration for '{standard}' must be positive, got {conc}"
                    )
        return v

    @field_validator('protein_concentrations')
    @classmethod
    def validate_protein_concentrations_positive(cls, v):
        """Ensure all protein concentrations are positive."""
        if v is not None:
            for sample, conc in v.items():
                if conc <= 0:
                    raise ValueError(
                        f"Protein concentration for sample '{sample}' must be positive, got {conc}"
                    )
        return v

    @model_validator(mode='after')
    def validate_method_requirements(self):
        """Validate that required fields are present for each normalization method."""
        if self.method == 'internal_standard':
            if not self.internal_standards:
                raise ValueError(
                    "Internal standard normalization requires 'internal_standards' mapping"
                )
            if not self.intsta_concentrations:
                raise ValueError(
                    "Internal standard normalization requires 'intsta_concentrations'"
                )

        elif self.method == 'protein':
            if not self.protein_concentrations:
                raise ValueError(
                    "Protein normalization requires 'protein_concentrations'"
                )

        elif self.method == 'both':
            if not self.internal_standards:
                raise ValueError(
                    "Combined normalization requires 'internal_standards' mapping"
                )
            if not self.intsta_concentrations:
                raise ValueError(
                    "Combined normalization requires 'intsta_concentrations'"
                )
            if not self.protein_concentrations:
                raise ValueError(
                    "Combined normalization requires 'protein_concentrations'"
                )

        # Cross-field: every standard in internal_standards must have a concentration
        if self.internal_standards and self.intsta_concentrations:
            for lipid_class, standard_name in self.internal_standards.items():
                if standard_name not in self.intsta_concentrations:
                    raise ValueError(
                        f"Missing concentration for internal standard '{standard_name}' "
                        f"(assigned to class '{lipid_class}'). "
                        f"Every standard must have an entry in 'intsta_concentrations'."
                    )

        return self

    def requires_internal_standards(self) -> bool:
        """Check if this method needs internal standard data."""
        return self.method in ('internal_standard', 'both')

    def requires_protein(self) -> bool:
        """Check if this method needs protein concentration data."""
        return self.method in ('protein', 'both')

    def get_standard_for_class(self, lipid_class: str) -> Optional[str]:
        """
        Get the internal standard assigned to a lipid class.

        Args:
            lipid_class: The lipid class to look up

        Returns:
            Standard lipid molecule name, or None if not mapped
        """
        if self.internal_standards is None:
            return None
        return self.internal_standards.get(lipid_class)

    def get_standard_concentration(self, standard: str) -> Optional[float]:
        """
        Get the concentration of an internal standard.

        Args:
            standard: Standard lipid molecule name

        Returns:
            Concentration value, or None if not found
        """
        if self.intsta_concentrations is None:
            return None
        return self.intsta_concentrations.get(standard)

    def get_protein_concentration(self, sample: str) -> Optional[float]:
        """
        Get the protein concentration for a sample.

        Args:
            sample: Sample label (e.g., 's1')

        Returns:
            Protein concentration, or None if not found
        """
        if self.protein_concentrations is None:
            return None
        return self.protein_concentrations.get(sample)
