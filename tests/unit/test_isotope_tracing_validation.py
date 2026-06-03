"""Unit tests for isotope-tracing input validation and config (pure Python, no R)."""
import pandas as pd
import pytest

from app.models.isotope_tracing import IsotopeCorrectionConfig
from app.services.isotope_correction import validate_inputs


def _valid_inputs():
    """Minimal well-formed Measurement / Molecule / Element trio."""
    measurement = pd.DataFrame(
        {
            "Measurements/Samples": ["BMP_18:1_18:1_C0", "BMP_18:1_18:1_C1", "PC_34:1_C0"],
            "sample_a": [100, 50, 200],
            "sample_b": [110, 40, 210],
        }
    )
    molecule = pd.DataFrame(
        {
            "Molecule": ["BMP_18:1_18:1", "PC_34:1"],
            "MS ion or MS/MS product ion": ["C42H79O10P1LabC6", "C42H82N1O8P1LabC6"],
            "MS/MS neutral loss": ["", ""],
        }
    )
    element = pd.DataFrame(
        {
            "Element": ["C", "H"],
            "Isotope abundance_Mass shift": ["0.0107_1/0.9893_0", "0.000115_1/0.999885_0"],
            "Tracer isotope mass shift": [1, None],
            "Tracer purity": [0.99, None],
        }
    )
    return measurement, molecule, element


class TestIsotopeCorrectionConfig:
    def test_defaults_match_golden_run(self):
        config = IsotopeCorrectionConfig()
        assert config.correct_tracer_impurity is True
        assert config.correct_tracer_element_core is True
        assert config.calculate_mean_enrichment is True
        assert config.ultra_high_res is False
        assert config.calculation_threshold_uhr == 8.0

    def test_is_frozen_and_hashable(self):
        config = IsotopeCorrectionConfig()
        assert hash(config) == hash(IsotopeCorrectionConfig())
        with pytest.raises(Exception):
            config.ultra_high_res = True

    def test_rejects_negative_threshold(self):
        with pytest.raises(Exception):
            IsotopeCorrectionConfig(calculation_threshold_uhr=-1)


class TestValidateInputs:
    def test_valid_inputs_pass(self):
        assert validate_inputs(*_valid_inputs()) == []

    def test_empty_measurement_flagged(self):
        _, molecule, element = _valid_inputs()
        errors = validate_inputs(pd.DataFrame(), molecule, element)
        assert any("Measurement file is empty" in e for e in errors)

    def test_no_sample_columns_flagged(self):
        measurement, molecule, element = _valid_inputs()
        only_ids = measurement[["Measurements/Samples"]]
        errors = validate_inputs(only_ids, molecule, element)
        assert any("at least one sample column" in e for e in errors)

    def test_non_numeric_intensity_flagged(self):
        measurement, molecule, element = _valid_inputs()
        measurement = measurement.copy()
        measurement.loc[0, "sample_a"] = "not_a_number"
        errors = validate_inputs(measurement, molecule, element)
        assert any("non-numeric" in e for e in errors)

    def test_molecule_name_mismatch_flagged(self):
        measurement, _, element = _valid_inputs()
        wrong_molecule = pd.DataFrame({"Molecule": ["SomethingElse"]})
        errors = validate_inputs(measurement, wrong_molecule, element)
        assert any("do not match any molecule" in e for e in errors)

    def test_prefix_matching_handles_underscores_in_molecule_name(self):
        # "BMP_18:1_18:1_C1" must match molecule "BMP_18:1_18:1" despite underscores.
        measurement, molecule, element = _valid_inputs()
        assert validate_inputs(measurement, molecule, element) == []


class TestResolutionModeConsistency:
    """The UHR setting must match the isotopologue-ID labeling style."""

    def _standard_inputs(self):
        """Inputs with standard-resolution IDs ('_0', '_1')."""
        measurement, molecule, element = _valid_inputs()
        measurement = measurement.copy()
        measurement["Measurements/Samples"] = [
            "BMP_18:1_18:1_0", "BMP_18:1_18:1_1", "PC_34:1_0"
        ]
        return measurement, molecule, element

    def test_uhr_ids_without_uhr_flag_flagged(self):
        # Sample IDs are UHR-style ("_C0"); running with UHR off must be caught.
        errors = validate_inputs(*_valid_inputs(), ultra_high_res=False)
        assert any("ultra-high-resolution labeling" in e for e in errors)
        assert any("Enable 'Ultra-high-resolution mode'" in e for e in errors)

    def test_uhr_ids_with_uhr_flag_ok(self):
        assert validate_inputs(*_valid_inputs(), ultra_high_res=True) == []

    def test_standard_ids_with_uhr_flag_flagged(self):
        errors = validate_inputs(*self._standard_inputs(), ultra_high_res=True)
        assert any("standard-resolution labeling" in e for e in errors)
        assert any("Disable 'Ultra-high-resolution mode'" in e for e in errors)

    def test_standard_ids_without_uhr_flag_ok(self):
        assert validate_inputs(*self._standard_inputs(), ultra_high_res=False) == []

    def test_mode_check_skipped_when_flag_is_none(self):
        # Default call (no flag) must not perform the mode check.
        assert validate_inputs(*_valid_inputs()) == []
