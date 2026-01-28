"""Unit tests for NormalizationConfig model."""
import pytest
from app.models.normalization import NormalizationConfig


class TestNormalizationConfigCreation:
    """Tests for creating NormalizationConfig instances."""

    def test_none_method_minimal(self):
        """Test creating config with no normalization."""
        config = NormalizationConfig(method='none')
        assert config.method == 'none'
        assert config.selected_classes == []

    def test_none_method_with_classes(self):
        """Test none method with class selection."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE', 'TG']
        )
        assert config.selected_classes == ['PC', 'PE', 'TG']

    def test_internal_standard_method_valid(self):
        """Test valid internal standard config."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0, 'PE(17:0/17:0)': 0.5}
        )
        assert config.method == 'internal_standard'
        assert config.internal_standards == {'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'}

    def test_protein_method_valid(self):
        """Test valid protein normalization config."""
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.5, 's2': 3.0, 's3': 2.8}
        )
        assert config.method == 'protein'
        assert config.protein_concentrations == {'s1': 2.5, 's2': 3.0, 's3': 2.8}

    def test_both_method_valid(self):
        """Test valid combined normalization config."""
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0},
            protein_concentrations={'s1': 2.5, 's2': 3.0}
        )
        assert config.method == 'both'

    def test_default_method_is_none(self):
        """Test that default method is 'none'."""
        config = NormalizationConfig()
        assert config.method == 'none'


class TestNormalizationConfigValidation:
    """Tests for validation rules."""

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            NormalizationConfig(method='invalid')

    def test_internal_standard_missing_standards(self):
        """Test internal_standard method requires internal_standards."""
        with pytest.raises(ValueError, match="internal_standards"):
            NormalizationConfig(
                method='internal_standard',
                intsta_concentrations={'PC(15:0/18:1)': 1.0}
            )

    def test_internal_standard_missing_concentrations(self):
        """Test internal_standard method requires intsta_concentrations."""
        with pytest.raises(ValueError, match="intsta_concentrations"):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'}
            )

    def test_protein_missing_concentrations(self):
        """Test protein method requires protein_concentrations."""
        with pytest.raises(ValueError, match="protein_concentrations"):
            NormalizationConfig(method='protein')

    def test_both_missing_internal_standards(self):
        """Test both method requires internal_standards."""
        with pytest.raises(ValueError, match="internal_standards"):
            NormalizationConfig(
                method='both',
                intsta_concentrations={'PC(15:0/18:1)': 1.0},
                protein_concentrations={'s1': 2.5}
            )

    def test_both_missing_intsta_concentrations(self):
        """Test both method requires intsta_concentrations."""
        with pytest.raises(ValueError, match="intsta_concentrations"):
            NormalizationConfig(
                method='both',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                protein_concentrations={'s1': 2.5}
            )

    def test_both_missing_protein_concentrations(self):
        """Test both method requires protein_concentrations."""
        with pytest.raises(ValueError, match="protein_concentrations"):
            NormalizationConfig(
                method='both',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations={'PC(15:0/18:1)': 1.0}
            )

    def test_negative_intsta_concentration(self):
        """Test negative internal standard concentration raises error."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations={'PC(15:0/18:1)': -1.0}
            )

    def test_zero_intsta_concentration(self):
        """Test zero internal standard concentration raises error."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations={'PC(15:0/18:1)': 0.0}
            )

    def test_negative_protein_concentration(self):
        """Test negative protein concentration raises error."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': -2.5}
            )

    def test_zero_protein_concentration(self):
        """Test zero protein concentration raises error."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': 0.0}
            )

    def test_mixed_valid_invalid_intsta_concentrations(self):
        """Test that one invalid concentration fails even if others are valid."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
                intsta_concentrations={'PC(15:0/18:1)': 1.0, 'PE(17:0/17:0)': -0.5}
            )

    def test_mixed_valid_invalid_protein_concentrations(self):
        """Test that one invalid protein concentration fails."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': 2.5, 's2': 0.0, 's3': 3.0}
            )


class TestNormalizationConfigHelperMethods:
    """Tests for helper methods."""

    def test_requires_internal_standards_true(self):
        """Test requires_internal_standards for methods that need it."""
        config_is = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert config_is.requires_internal_standards() is True

        config_both = NormalizationConfig(
            method='both',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0},
            protein_concentrations={'s1': 2.5}
        )
        assert config_both.requires_internal_standards() is True

    def test_requires_internal_standards_false(self):
        """Test requires_internal_standards for methods that don't need it."""
        config_none = NormalizationConfig(method='none')
        assert config_none.requires_internal_standards() is False

        config_protein = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        assert config_protein.requires_internal_standards() is False

    def test_requires_protein_true(self):
        """Test requires_protein for methods that need it."""
        config_protein = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        assert config_protein.requires_protein() is True

        config_both = NormalizationConfig(
            method='both',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0},
            protein_concentrations={'s1': 2.5}
        )
        assert config_both.requires_protein() is True

    def test_requires_protein_false(self):
        """Test requires_protein for methods that don't need it."""
        config_none = NormalizationConfig(method='none')
        assert config_none.requires_protein() is False

        config_is = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert config_is.requires_protein() is False

    def test_get_standard_for_class_exists(self):
        """Test getting standard for a mapped class."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0, 'PE(17:0/17:0)': 0.5}
        )
        assert config.get_standard_for_class('PC') == 'PC(15:0/18:1)'
        assert config.get_standard_for_class('PE') == 'PE(17:0/17:0)'

    def test_get_standard_for_class_not_exists(self):
        """Test getting standard for an unmapped class."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert config.get_standard_for_class('TG') is None

    def test_get_standard_for_class_none_method(self):
        """Test getting standard when no standards configured."""
        config = NormalizationConfig(method='none')
        assert config.get_standard_for_class('PC') is None

    def test_get_standard_concentration_exists(self):
        """Test getting concentration for an existing standard."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.5}
        )
        assert config.get_standard_concentration('PC(15:0/18:1)') == 1.5

    def test_get_standard_concentration_not_exists(self):
        """Test getting concentration for a non-existent standard."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.5}
        )
        assert config.get_standard_concentration('PE(17:0/17:0)') is None

    def test_get_protein_concentration_exists(self):
        """Test getting protein concentration for an existing sample."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5, 's2': 3.0}
        )
        assert config.get_protein_concentration('s1') == 2.5
        assert config.get_protein_concentration('s2') == 3.0

    def test_get_protein_concentration_not_exists(self):
        """Test getting protein concentration for a non-existent sample."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        assert config.get_protein_concentration('s99') is None

    def test_get_protein_concentration_none_method(self):
        """Test getting protein concentration when no protein data configured."""
        config = NormalizationConfig(method='none')
        assert config.get_protein_concentration('s1') is None


class TestNormalizationConfigSerialization:
    """Tests for model serialization."""

    def test_to_dict_none_method(self):
        """Test serializing config with none method."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE']
        )
        data = config.model_dump()
        assert data['method'] == 'none'
        assert data['selected_classes'] == ['PC', 'PE']
        assert data['internal_standards'] is None

    def test_to_dict_full_config(self):
        """Test serializing a full config."""
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0},
            protein_concentrations={'s1': 2.5, 's2': 3.0}
        )
        data = config.model_dump()
        assert data['method'] == 'both'
        assert data['internal_standards'] == {'PC': 'PC(15:0/18:1)'}
        assert data['intsta_concentrations'] == {'PC(15:0/18:1)': 1.0}
        assert data['protein_concentrations'] == {'s1': 2.5, 's2': 3.0}

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            'method': 'protein',
            'selected_classes': ['PC', 'PE'],
            'protein_concentrations': {'s1': 2.5, 's2': 3.0}
        }
        config = NormalizationConfig.model_validate(data)
        assert config.method == 'protein'
        assert config.protein_concentrations == {'s1': 2.5, 's2': 3.0}

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'TG'],
            internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0, 'PE(17:0/17:0)': 0.5}
        )
        json_str = config.model_dump_json()
        restored = NormalizationConfig.model_validate_json(json_str)

        assert restored.method == config.method
        assert restored.selected_classes == config.selected_classes
        assert restored.internal_standards == config.internal_standards
        assert restored.intsta_concentrations == config.intsta_concentrations


class TestNormalizationConfigEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_selected_classes(self):
        """Test config with empty selected classes."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=[],
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert config.selected_classes == []

    def test_many_classes_and_standards(self):
        """Test config with many lipid classes."""
        classes = ['PC', 'PE', 'PS', 'PI', 'PG', 'PA', 'TG', 'DG', 'MG', 'CE']
        standards = {c: f'{c}(15:0/18:1)' for c in classes}
        concentrations = {f'{c}(15:0/18:1)': 1.0 for c in classes}

        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=classes,
            internal_standards=standards,
            intsta_concentrations=concentrations
        )
        assert len(config.selected_classes) == 10
        assert len(config.internal_standards) == 10

    def test_many_samples_protein(self):
        """Test config with many samples for protein normalization."""
        samples = {f's{i}': 2.5 + (i * 0.1) for i in range(1, 101)}

        config = NormalizationConfig(
            method='protein',
            protein_concentrations=samples
        )
        assert len(config.protein_concentrations) == 100
        assert config.get_protein_concentration('s1') == pytest.approx(2.6)
        assert config.get_protein_concentration('s100') == pytest.approx(12.5)

    def test_special_characters_in_standard_names(self):
        """Test standard names with special characters."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1(9Z))'},
            intsta_concentrations={'PC(15:0/18:1(9Z))': 1.0}
        )
        assert config.get_standard_for_class('PC') == 'PC(15:0/18:1(9Z))'

    def test_very_small_concentrations(self):
        """Test very small but positive concentrations."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 0.0001}
        )
        assert config.get_standard_concentration('PC(15:0/18:1)') == 0.0001

    def test_very_large_concentrations(self):
        """Test very large concentrations."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 1000000.0}
        )
        assert config.get_protein_concentration('s1') == 1000000.0

    def test_float_precision(self):
        """Test float precision is maintained."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.123456789}
        )
        assert config.get_protein_concentration('s1') == 2.123456789

    def test_empty_dict_for_internal_standards(self):
        """Test empty dict vs None for internal_standards."""
        # Empty dict should fail validation for internal_standard method
        with pytest.raises(ValueError, match="internal_standards"):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={},
                intsta_concentrations={'PC(15:0/18:1)': 1.0}
            )

    def test_empty_dict_for_protein_concentrations(self):
        """Test empty dict fails for protein method."""
        with pytest.raises(ValueError, match="protein_concentrations"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={}
            )

    def test_single_class_single_standard(self):
        """Test minimal valid config with single class and standard."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert len(config.internal_standards) == 1
        assert config.get_standard_for_class('PC') == 'PC(15:0/18:1)'

    def test_single_sample_protein(self):
        """Test minimal valid config with single sample."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        assert len(config.protein_concentrations) == 1

    def test_case_sensitivity_class_lookup(self):
        """Test that class lookup is case-sensitive."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert config.get_standard_for_class('PC') == 'PC(15:0/18:1)'
        assert config.get_standard_for_class('pc') is None
        assert config.get_standard_for_class('Pc') is None

    def test_case_sensitivity_sample_lookup(self):
        """Test that sample lookup is case-sensitive."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5, 'S1': 3.0}
        )
        assert config.get_protein_concentration('s1') == 2.5
        assert config.get_protein_concentration('S1') == 3.0

    def test_unicode_in_class_names(self):
        """Test unicode characters in lipid class names."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC-α', 'PE-β'],
            internal_standards={'PC-α': 'PC-α(15:0/18:1)', 'PE-β': 'PE-β(17:0/17:0)'},
            intsta_concentrations={'PC-α(15:0/18:1)': 1.0, 'PE-β(17:0/17:0)': 0.5}
        )
        assert config.get_standard_for_class('PC-α') == 'PC-α(15:0/18:1)'

    def test_long_standard_names(self):
        """Test very long standard names (realistic for complex lipids)."""
        long_name = 'TG(16:0/18:1(9Z)/18:2(9Z,12Z))[iso3]'
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'TG': long_name},
            intsta_concentrations={long_name: 1.0}
        )
        assert config.get_standard_for_class('TG') == long_name
        assert config.get_standard_concentration(long_name) == 1.0

    def test_selected_classes_not_in_standards(self):
        """Test that selected_classes can include classes without standards."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'TG'],  # TG has no standard
            internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0, 'PE(17:0/17:0)': 0.5}
        )
        # This should be allowed - some classes may not need normalization
        assert config.get_standard_for_class('TG') is None
        assert config.get_standard_for_class('PC') == 'PC(15:0/18:1)'

    def test_extra_standards_not_used(self):
        """Test that standards can exist without being in selected_classes."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0, 'PE(17:0/17:0)': 0.5}
        )
        # PE standard exists but PE is not in selected_classes - this is allowed
        assert config.get_standard_for_class('PE') == 'PE(17:0/17:0)'

    def test_all_method_literals_accepted(self):
        """Test all valid method values are accepted."""
        # none
        config_none = NormalizationConfig(method='none')
        assert config_none.method == 'none'

        # internal_standard
        config_is = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        assert config_is.method == 'internal_standard'

        # protein
        config_prot = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        assert config_prot.method == 'protein'

        # both
        config_both = NormalizationConfig(
            method='both',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1.0},
            protein_concentrations={'s1': 2.5}
        )
        assert config_both.method == 'both'

    def test_duplicate_classes_in_selected(self):
        """Test that duplicate classes in selected_classes are preserved."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE', 'PC']  # PC appears twice
        )
        assert config.selected_classes == ['PC', 'PE', 'PC']

    def test_whitespace_in_sample_names(self):
        """Test sample names with whitespace are preserved."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'sample 1': 2.5, 'sample_2': 3.0}
        )
        assert config.get_protein_concentration('sample 1') == 2.5
        assert config.get_protein_concentration('sample_2') == 3.0

    def test_numeric_sample_names(self):
        """Test numeric-looking sample names."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'1': 2.5, '2': 3.0, '10': 4.0}
        )
        assert config.get_protein_concentration('1') == 2.5
        assert config.get_protein_concentration('10') == 4.0


class TestNormalizationConfigRealWorldScenarios:
    """Tests simulating real-world usage patterns."""

    def test_lipidomics_workflow_internal_standard(self):
        """Test typical lipidomics internal standard normalization."""
        # Typical lipid classes in a lipidomics experiment
        classes = ['PC', 'PE', 'PI', 'PS', 'SM', 'Cer', 'TG', 'DG']

        # Standards for major classes
        standards = {
            'PC': 'PC(17:0/17:0)',
            'PE': 'PE(17:0/17:0)',
            'PI': 'PI(17:0/17:0)',
            'SM': 'SM(d18:1/17:0)',
            'Cer': 'Cer(d18:1/17:0)',
            'TG': 'TG(17:0/17:0/17:0)',
        }

        concentrations = {std: 1.0 for std in standards.values()}

        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=classes,
            internal_standards=standards,
            intsta_concentrations=concentrations
        )

        # Verify all mapped classes work
        assert config.get_standard_for_class('PC') == 'PC(17:0/17:0)'
        assert config.get_standard_concentration('PC(17:0/17:0)') == 1.0

        # PS and DG have no standards
        assert config.get_standard_for_class('PS') is None
        assert config.get_standard_for_class('DG') is None

    def test_lipidomics_workflow_protein_normalized(self):
        """Test typical protein-normalized lipidomics workflow."""
        # 4 conditions, 3 samples each = 12 samples
        protein_data = {
            's1': 1.2, 's2': 1.3, 's3': 1.1,  # Condition A
            's4': 1.4, 's5': 1.5, 's6': 1.3,  # Condition B
            's7': 1.1, 's8': 1.2, 's9': 1.0,  # Condition C
            's10': 1.3, 's11': 1.4, 's12': 1.2,  # Condition D
        }

        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE', 'TG'],
            protein_concentrations=protein_data
        )

        assert config.requires_protein() is True
        assert config.requires_internal_standards() is False
        assert config.get_protein_concentration('s1') == 1.2
        assert config.get_protein_concentration('s12') == 1.2

    def test_lipidomics_workflow_combined(self):
        """Test combined IS + protein normalization workflow."""
        standards = {
            'PC': 'PC(17:0/17:0)',
            'PE': 'PE(17:0/17:0)',
        }
        concentrations = {
            'PC(17:0/17:0)': 0.5,
            'PE(17:0/17:0)': 0.5,
        }
        protein_data = {
            's1': 1.2, 's2': 1.3, 's3': 1.1,
            's4': 1.4, 's5': 1.5, 's6': 1.3,
        }

        config = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE'],
            internal_standards=standards,
            intsta_concentrations=concentrations,
            protein_concentrations=protein_data
        )

        assert config.requires_protein() is True
        assert config.requires_internal_standards() is True

    def test_changing_method_requires_new_config(self):
        """Test that changing normalization method requires creating new config."""
        # Start with none
        config1 = NormalizationConfig(method='none', selected_classes=['PC'])

        # Can't just switch to protein - need to provide protein data
        config2 = NormalizationConfig(
            method='protein',
            selected_classes=config1.selected_classes,
            protein_concentrations={'s1': 2.5}
        )

        assert config1.method == 'none'
        assert config2.method == 'protein'

    def test_metabolomics_workbench_sample_naming(self):
        """Test with Metabolomics Workbench style sample names."""
        # MW uses descriptive sample names
        protein_data = {
            'Control_1': 1.2,
            'Control_2': 1.3,
            'Control_3': 1.1,
            'Treatment_A_1': 1.4,
            'Treatment_A_2': 1.5,
            'Treatment_A_3': 1.3,
        }

        config = NormalizationConfig(
            method='protein',
            protein_concentrations=protein_data
        )

        assert config.get_protein_concentration('Control_1') == 1.2
        assert config.get_protein_concentration('Treatment_A_1') == 1.4


class TestNormalizationConfigTypeHandling:
    """Tests for type coercion and type error handling."""

    def test_integer_intsta_concentration_coerced_to_float(self):
        """Test that integer concentrations are coerced to floats."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': 1}  # int, not float
        )
        assert config.intsta_concentrations['PC(15:0/18:1)'] == 1.0
        assert isinstance(config.intsta_concentrations['PC(15:0/18:1)'], float)

    def test_integer_protein_concentration_coerced_to_float(self):
        """Test that integer protein concentrations are coerced to floats."""
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2, 's2': 3}  # ints
        )
        assert config.protein_concentrations['s1'] == 2.0
        assert isinstance(config.protein_concentrations['s1'], float)

    def test_string_method_literal_required(self):
        """Test that method must be a valid string literal."""
        with pytest.raises(ValueError):
            NormalizationConfig(method='NONE')  # wrong case

    def test_integer_method_raises_error(self):
        """Test that integer method raises validation error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(method=1)

    def test_none_method_value_raises_error(self):
        """Test that None as method value raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(method=None)

    def test_list_instead_of_dict_for_internal_standards_raises_error(self):
        """Test that list instead of dict for internal_standards raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='internal_standard',
                internal_standards=['PC(15:0/18:1)', 'PE(17:0/17:0)'],
                intsta_concentrations={'PC(15:0/18:1)': 1.0}
            )

    def test_list_instead_of_dict_for_concentrations_raises_error(self):
        """Test that list instead of dict for concentrations raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations=[1.0, 0.5]
            )

    def test_list_instead_of_dict_for_protein_concentrations_raises_error(self):
        """Test that list instead of dict for protein_concentrations raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='protein',
                protein_concentrations=[2.5, 3.0]
            )

    def test_tuple_for_selected_classes_works(self):
        """Test that tuple is accepted for selected_classes."""
        config = NormalizationConfig(
            method='none',
            selected_classes=('PC', 'PE', 'TG')  # tuple instead of list
        )
        # Pydantic converts tuples to lists
        assert isinstance(config.selected_classes, list)
        assert config.selected_classes == ['PC', 'PE', 'TG']

    def test_set_for_selected_classes_handling(self):
        """Test that set input for selected_classes is handled."""
        # Sets might work but lose order
        try:
            config = NormalizationConfig(
                method='none',
                selected_classes={'PC', 'PE'}  # set
            )
            assert len(config.selected_classes) == 2
        except (ValueError, TypeError):
            pass  # Also acceptable

    def test_numeric_string_concentration_coerced_to_float(self):
        """Test that numeric string concentration values are coerced to floats."""
        # Pydantic v2 coerces numeric strings to floats
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': '1.0'}  # string coerced
        )
        assert config.intsta_concentrations['PC(15:0/18:1)'] == 1.0
        assert isinstance(config.intsta_concentrations['PC(15:0/18:1)'], float)

    def test_numeric_string_protein_concentration_coerced_to_float(self):
        """Test that numeric string protein concentration values are coerced."""
        # Pydantic v2 coerces numeric strings to floats
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': '2.5'}  # string coerced
        )
        assert config.protein_concentrations['s1'] == 2.5
        assert isinstance(config.protein_concentrations['s1'], float)

    def test_non_numeric_string_concentration_raises_error(self):
        """Test that non-numeric string concentration values raise error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations={'PC(15:0/18:1)': 'not_a_number'}
            )

    def test_non_numeric_string_protein_concentration_raises_error(self):
        """Test that non-numeric string protein concentration values raise error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': 'invalid'}
            )

    def test_none_value_in_internal_standards_dict(self):
        """Test that None value in internal_standards dict raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': None},  # None value
                intsta_concentrations={'PC(15:0/18:1)': 1.0}
            )

    def test_none_value_in_intsta_concentrations_dict(self):
        """Test that None value in intsta_concentrations dict raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations={'PC(15:0/18:1)': None}  # None value
            )

    def test_none_value_in_protein_concentrations_dict(self):
        """Test that None value in protein_concentrations dict raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': None}  # None value
            )

    def test_none_in_selected_classes_list(self):
        """Test that None in selected_classes list raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='none',
                selected_classes=['PC', None, 'PE']
            )

    def test_integer_in_selected_classes_raises_error(self):
        """Test that integer in selected_classes raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='none',
                selected_classes=['PC', 123, 'PE']
            )

    def test_nested_dict_for_internal_standards_raises_error(self):
        """Test that nested dict for internal_standards raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': {'name': 'PC(15:0/18:1)'}},  # nested
                intsta_concentrations={'PC(15:0/18:1)': 1.0}
            )

    def test_nested_list_in_selected_classes_raises_error(self):
        """Test that nested list in selected_classes raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='none',
                selected_classes=[['PC'], ['PE']]  # nested lists
            )

    def test_boolean_concentration_coerced(self):
        """Test that boolean concentrations are coerced (True=1.0)."""
        # True should become 1.0, which is positive and valid
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)'},
            intsta_concentrations={'PC(15:0/18:1)': True}  # True = 1.0
        )
        assert config.intsta_concentrations['PC(15:0/18:1)'] == 1.0

    def test_boolean_false_concentration_fails_validation(self):
        """Test that False concentration (0.0) fails positive validation."""
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='internal_standard',
                internal_standards={'PC': 'PC(15:0/18:1)'},
                intsta_concentrations={'PC(15:0/18:1)': False}  # False = 0.0
            )

    def test_dict_instead_of_list_for_selected_classes_raises(self):
        """Test that dict instead of list for selected_classes raises error."""
        with pytest.raises((ValueError, TypeError)):
            NormalizationConfig(
                method='none',
                selected_classes={'PC': 1, 'PE': 2}
            )

    def test_mixed_types_in_concentration_values(self):
        """Test mixed int/float concentration values work."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': 'PC(15:0/18:1)', 'PE': 'PE(17:0/17:0)'},
            intsta_concentrations={
                'PC(15:0/18:1)': 1,    # int
                'PE(17:0/17:0)': 0.5   # float
            }
        )
        assert config.intsta_concentrations['PC(15:0/18:1)'] == 1.0
        assert config.intsta_concentrations['PE(17:0/17:0)'] == 0.5


class TestNormalizationConfigInputValidation:
    """Tests for input validation edge cases."""

    def test_empty_string_key_in_internal_standards(self):
        """Test empty string as key in internal_standards."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'': 'PC(15:0/18:1)'},  # empty key
            intsta_concentrations={'PC(15:0/18:1)': 1.0}
        )
        # Empty string keys are technically valid for dict
        assert config.internal_standards.get('') == 'PC(15:0/18:1)'

    def test_empty_string_value_in_internal_standards(self):
        """Test empty string as value in internal_standards."""
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': ''},  # empty value
            intsta_concentrations={'': 1.0}  # matching empty key
        )
        assert config.get_standard_for_class('PC') == ''

    def test_empty_string_in_selected_classes(self):
        """Test empty string in selected_classes is allowed."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', '', 'PE']
        )
        assert '' in config.selected_classes

    def test_whitespace_only_class_in_selected_classes(self):
        """Test whitespace-only class name is allowed."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', '   ', 'PE']
        )
        assert '   ' in config.selected_classes

    def test_generator_for_selected_classes(self):
        """Test generator input for selected_classes."""
        try:
            config = NormalizationConfig(
                method='none',
                selected_classes=(x for x in ['PC', 'PE', 'TG'])
            )
            assert config.selected_classes == ['PC', 'PE', 'TG']
        except (ValueError, TypeError):
            pass  # Also acceptable

    def test_inf_concentration_handling(self):
        """Test infinity concentration handling."""
        import math
        # Infinity should fail the positive check? Actually inf > 0 is True
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': math.inf}
        )
        assert config.protein_concentrations['s1'] == math.inf

    def test_nan_concentration_handling(self):
        """Test NaN concentration handling."""
        import math
        # NaN comparisons are tricky - NaN <= 0 is False, but NaN > 0 is also False
        # The validator uses `if conc <= 0` so NaN should pass the check
        # but this may not be desired behavior
        try:
            config = NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': math.nan}
            )
            # If it passes, NaN is stored
            assert math.isnan(config.protein_concentrations['s1'])
        except ValueError:
            pass  # Also acceptable if validation catches NaN

    def test_very_small_positive_concentration(self):
        """Test very small positive concentration (near machine epsilon)."""
        import sys
        tiny = sys.float_info.min
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': tiny}
        )
        assert config.protein_concentrations['s1'] == tiny

    def test_negative_zero_concentration_fails(self):
        """Test negative zero concentration fails validation."""
        # -0.0 equals 0.0 in Python, so should fail
        with pytest.raises(ValueError, match="positive"):
            NormalizationConfig(
                method='protein',
                protein_concentrations={'s1': -0.0}
            )


class TestNormalizationConfigBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_very_large_concentration_dictionary(self):
        """Test with many entries in concentration dictionary."""
        standards = {f'Class_{i}': f'Standard_{i}' for i in range(100)}
        concentrations = {f'Standard_{i}': 1.0 + i * 0.01 for i in range(100)}

        config = NormalizationConfig(
            method='internal_standard',
            internal_standards=standards,
            intsta_concentrations=concentrations
        )
        assert len(config.internal_standards) == 100
        assert len(config.intsta_concentrations) == 100

    def test_very_long_class_name(self):
        """Test very long lipid class name."""
        long_class = 'X' * 1000
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={long_class: 'Standard'},
            intsta_concentrations={'Standard': 1.0}
        )
        assert config.get_standard_for_class(long_class) == 'Standard'

    def test_very_long_standard_name(self):
        """Test very long internal standard name."""
        long_standard = 'Y' * 1000
        config = NormalizationConfig(
            method='internal_standard',
            internal_standards={'PC': long_standard},
            intsta_concentrations={long_standard: 1.0}
        )
        assert config.get_standard_for_class('PC') == long_standard
        assert config.get_standard_concentration(long_standard) == 1.0

    def test_very_long_sample_name(self):
        """Test very long sample name."""
        long_sample = 'Z' * 1000
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={long_sample: 2.5}
        )
        assert config.get_protein_concentration(long_sample) == 2.5

    def test_many_selected_classes(self):
        """Test with many selected classes."""
        classes = [f'Class_{i}' for i in range(500)]
        config = NormalizationConfig(
            method='none',
            selected_classes=classes
        )
        assert len(config.selected_classes) == 500

    def test_maximum_float_concentration(self):
        """Test maximum float concentration."""
        import sys
        config = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': sys.float_info.max}
        )
        assert config.protein_concentrations['s1'] == sys.float_info.max


class TestNormalizationConfigCopyAndEquality:
    """Tests for model copy and equality behavior."""

    def test_equal_configs(self):
        """Test that two configs with same values are equal."""
        config1 = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE'],
            protein_concentrations={'s1': 2.5, 's2': 3.0}
        )
        config2 = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE'],
            protein_concentrations={'s1': 2.5, 's2': 3.0}
        )
        assert config1 == config2

    def test_unequal_method(self):
        """Test configs with different methods are not equal."""
        config1 = NormalizationConfig(method='none')
        config2 = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        assert config1 != config2

    def test_unequal_concentrations(self):
        """Test configs with different concentrations are not equal."""
        config1 = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        config2 = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 3.0}
        )
        assert config1 != config2

    def test_model_copy(self):
        """Test that model_copy creates independent copy."""
        config1 = NormalizationConfig(
            method='protein',
            protein_concentrations={'s1': 2.5}
        )
        config2 = config1.model_copy()

        assert config1 == config2
        assert config1 is not config2

    def test_model_copy_deep(self):
        """Test deep copy of model."""
        config1 = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE'],
            protein_concentrations={'s1': 2.5}
        )
        config2 = config1.model_copy(deep=True)

        assert config1 == config2
        assert config1.selected_classes is not config2.selected_classes
        assert config1.protein_concentrations is not config2.protein_concentrations


class TestNormalizationConfigModelSchema:
    """Tests for model JSON schema."""

    def test_json_schema_generation(self):
        """Test that JSON schema can be generated."""
        schema = NormalizationConfig.model_json_schema()
        assert 'properties' in schema
        assert 'method' in schema['properties']
        assert 'selected_classes' in schema['properties']

    def test_json_schema_method_enum(self):
        """Test that method has enum values in schema."""
        schema = NormalizationConfig.model_json_schema()
        method_schema = schema['properties']['method']
        # Check that the valid method values are documented
        assert 'enum' in method_schema or 'const' in method_schema or 'anyOf' in method_schema
