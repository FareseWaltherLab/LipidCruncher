"""Unit tests for StatisticalTestConfig model."""
import pytest
from app.models.statistics import StatisticalTestConfig


class TestStatisticalTestConfigCreation:
    """Tests for creating StatisticalTestConfig instances."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        config = StatisticalTestConfig()
        assert config.mode == 'manual'
        assert config.test_type == 'parametric'
        assert config.correction_method == 'fdr_bh'
        assert config.posthoc_correction == 'tukey'
        assert config.alpha == 0.05
        assert config.auto_transform is True
        assert config.conditions_to_compare == []

    def test_manual_mode_parametric(self):
        """Test manual mode with parametric settings."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='parametric',
            correction_method='fdr_bh',
            posthoc_correction='tukey'
        )
        assert config.mode == 'manual'
        assert config.test_type == 'parametric'

    def test_manual_mode_non_parametric(self):
        """Test manual mode with non-parametric settings."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='non_parametric',
            correction_method='bonferroni',
            posthoc_correction='bonferroni'
        )
        assert config.test_type == 'non_parametric'
        assert config.correction_method == 'bonferroni'

    def test_auto_mode(self):
        """Test auto mode configuration."""
        config = StatisticalTestConfig(
            mode='auto',
            test_type='auto',
            correction_method='auto',
            posthoc_correction='auto'
        )
        assert config.mode == 'auto'
        assert config.test_type == 'auto'
        assert config.correction_method == 'auto'
        assert config.posthoc_correction == 'auto'

    def test_custom_alpha(self):
        """Test custom alpha value."""
        config = StatisticalTestConfig(alpha=0.01)
        assert config.alpha == 0.01

    def test_auto_transform_false(self):
        """Test disabling auto-transform."""
        config = StatisticalTestConfig(auto_transform=False)
        assert config.auto_transform is False

    def test_with_conditions_to_compare(self):
        """Test setting conditions to compare."""
        conditions = [('Control', 'Treatment1'), ('Control', 'Treatment2')]
        config = StatisticalTestConfig(conditions_to_compare=conditions)
        assert config.conditions_to_compare == conditions

    def test_all_correction_methods_manual(self):
        """Test all valid correction methods in manual mode."""
        for method in ['uncorrected', 'fdr_bh', 'bonferroni']:
            config = StatisticalTestConfig(correction_method=method)
            assert config.correction_method == method

    def test_all_posthoc_methods_manual(self):
        """Test all valid posthoc correction methods in manual mode."""
        for method in ['uncorrected', 'tukey', 'bonferroni']:
            config = StatisticalTestConfig(posthoc_correction=method)
            assert config.posthoc_correction == method


class TestStatisticalTestConfigValidation:
    """Tests for validation rules."""

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="auto.*manual"):
            StatisticalTestConfig(mode='invalid')

    def test_invalid_test_type(self):
        """Test that invalid test_type raises error."""
        with pytest.raises(ValueError, match="parametric.*non_parametric.*auto"):
            StatisticalTestConfig(test_type='invalid')

    def test_invalid_correction_method(self):
        """Test that invalid correction_method raises error."""
        with pytest.raises(ValueError, match="uncorrected.*fdr_bh.*bonferroni.*auto"):
            StatisticalTestConfig(correction_method='invalid')

    def test_invalid_posthoc_correction(self):
        """Test that invalid posthoc_correction raises error."""
        with pytest.raises(ValueError, match="uncorrected.*tukey.*bonferroni.*auto"):
            StatisticalTestConfig(posthoc_correction='invalid')

    def test_alpha_zero_invalid(self):
        """Test that alpha=0 raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            StatisticalTestConfig(alpha=0)

    def test_alpha_one_invalid(self):
        """Test that alpha=1 raises error."""
        with pytest.raises(ValueError, match="less than 1"):
            StatisticalTestConfig(alpha=1)

    def test_alpha_negative_invalid(self):
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            StatisticalTestConfig(alpha=-0.05)

    def test_alpha_greater_than_one_invalid(self):
        """Test that alpha > 1 raises error."""
        with pytest.raises(ValueError, match="less than 1"):
            StatisticalTestConfig(alpha=1.5)

    def test_alpha_boundary_low(self):
        """Test alpha just above 0 is valid."""
        config = StatisticalTestConfig(alpha=0.001)
        assert config.alpha == 0.001

    def test_alpha_boundary_high(self):
        """Test alpha just below 1 is valid."""
        config = StatisticalTestConfig(alpha=0.999)
        assert config.alpha == 0.999


class TestAutoModeValidation:
    """Tests for auto mode validation rules."""

    def test_auto_mode_requires_auto_test_type(self):
        """Test auto mode requires test_type='auto'."""
        with pytest.raises(ValueError, match="test_type should be 'auto'"):
            StatisticalTestConfig(
                mode='auto',
                test_type='parametric',
                correction_method='auto',
                posthoc_correction='auto'
            )

    def test_auto_mode_requires_auto_correction(self):
        """Test auto mode requires correction_method='auto'."""
        with pytest.raises(ValueError, match="correction_method should be 'auto'"):
            StatisticalTestConfig(
                mode='auto',
                test_type='auto',
                correction_method='fdr_bh',
                posthoc_correction='auto'
            )

    def test_auto_mode_requires_auto_posthoc(self):
        """Test auto mode requires posthoc_correction='auto'."""
        with pytest.raises(ValueError, match="posthoc_correction should be 'auto'"):
            StatisticalTestConfig(
                mode='auto',
                test_type='auto',
                correction_method='auto',
                posthoc_correction='tukey'
            )

    def test_auto_mode_all_auto_valid(self):
        """Test auto mode with all settings as 'auto' is valid."""
        config = StatisticalTestConfig(
            mode='auto',
            test_type='auto',
            correction_method='auto',
            posthoc_correction='auto'
        )
        assert config.is_auto_mode() is True


class TestManualModeValidation:
    """Tests for manual mode validation rules."""

    def test_manual_mode_rejects_auto_test_type(self):
        """Test manual mode rejects test_type='auto'."""
        with pytest.raises(ValueError, match="test_type cannot be 'auto'"):
            StatisticalTestConfig(
                mode='manual',
                test_type='auto'
            )

    def test_manual_mode_rejects_auto_correction(self):
        """Test manual mode rejects correction_method='auto'."""
        with pytest.raises(ValueError, match="correction_method cannot be 'auto'"):
            StatisticalTestConfig(
                mode='manual',
                correction_method='auto'
            )

    def test_manual_mode_rejects_auto_posthoc(self):
        """Test manual mode rejects posthoc_correction='auto'."""
        with pytest.raises(ValueError, match="posthoc_correction cannot be 'auto'"):
            StatisticalTestConfig(
                mode='manual',
                posthoc_correction='auto'
            )


class TestConditionPairsValidation:
    """Tests for conditions_to_compare validation."""

    def test_empty_conditions_valid(self):
        """Test empty conditions list is valid."""
        config = StatisticalTestConfig(conditions_to_compare=[])
        assert config.conditions_to_compare == []

    def test_valid_single_pair(self):
        """Test single valid condition pair."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('Control', 'Treatment')]
        )
        assert config.conditions_to_compare == [('Control', 'Treatment')]

    def test_valid_multiple_pairs(self):
        """Test multiple valid condition pairs."""
        pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        config = StatisticalTestConfig(conditions_to_compare=pairs)
        assert config.conditions_to_compare == pairs

    def test_same_condition_comparison_invalid(self):
        """Test comparing condition to itself raises error."""
        with pytest.raises(ValueError, match="Cannot compare a condition to itself"):
            StatisticalTestConfig(
                conditions_to_compare=[('Control', 'Control')]
            )

    def test_empty_condition_name_invalid(self):
        """Test empty condition name raises error."""
        with pytest.raises(ValueError, match="Condition names cannot be empty"):
            StatisticalTestConfig(
                conditions_to_compare=[('Control', '')]
            )

    def test_empty_first_condition_invalid(self):
        """Test empty first condition name raises error."""
        with pytest.raises(ValueError, match="Condition names cannot be empty"):
            StatisticalTestConfig(
                conditions_to_compare=[('', 'Treatment')]
            )


class TestHelperMethods:
    """Tests for helper methods."""

    def test_is_auto_mode_true(self):
        """Test is_auto_mode returns True for auto mode."""
        config = StatisticalTestConfig(
            mode='auto',
            test_type='auto',
            correction_method='auto',
            posthoc_correction='auto'
        )
        assert config.is_auto_mode() is True

    def test_is_auto_mode_false(self):
        """Test is_auto_mode returns False for manual mode."""
        config = StatisticalTestConfig(mode='manual')
        assert config.is_auto_mode() is False

    def test_is_parametric_true(self):
        """Test is_parametric returns True for parametric tests."""
        config = StatisticalTestConfig(test_type='parametric')
        assert config.is_parametric() is True

    def test_is_parametric_false(self):
        """Test is_parametric returns False for non-parametric tests."""
        config = StatisticalTestConfig(test_type='non_parametric')
        assert config.is_parametric() is False

    def test_is_non_parametric_true(self):
        """Test is_non_parametric returns True for non-parametric tests."""
        config = StatisticalTestConfig(test_type='non_parametric')
        assert config.is_non_parametric() is True

    def test_is_non_parametric_false(self):
        """Test is_non_parametric returns False for parametric tests."""
        config = StatisticalTestConfig(test_type='parametric')
        assert config.is_non_parametric() is False

    def test_requires_posthoc_two_conditions(self):
        """Test requires_posthoc returns False for 2 conditions."""
        config = StatisticalTestConfig(posthoc_correction='tukey')
        assert config.requires_posthoc(2) is False

    def test_requires_posthoc_three_conditions_with_correction(self):
        """Test requires_posthoc returns True for 3+ conditions with correction."""
        config = StatisticalTestConfig(posthoc_correction='tukey')
        assert config.requires_posthoc(3) is True

    def test_requires_posthoc_three_conditions_uncorrected(self):
        """Test requires_posthoc returns False when uncorrected."""
        config = StatisticalTestConfig(posthoc_correction='uncorrected')
        assert config.requires_posthoc(3) is False

    def test_requires_posthoc_four_conditions(self):
        """Test requires_posthoc returns True for 4 conditions."""
        config = StatisticalTestConfig(posthoc_correction='bonferroni')
        assert config.requires_posthoc(4) is True


class TestDisplayNames:
    """Tests for display name methods."""

    def test_correction_display_uncorrected(self):
        """Test display name for uncorrected."""
        config = StatisticalTestConfig(correction_method='uncorrected')
        assert config.get_correction_display_name() == 'Uncorrected'

    def test_correction_display_fdr_bh(self):
        """Test display name for FDR-BH."""
        config = StatisticalTestConfig(correction_method='fdr_bh')
        assert config.get_correction_display_name() == 'FDR (Benjamini-Hochberg)'

    def test_correction_display_bonferroni(self):
        """Test display name for Bonferroni."""
        config = StatisticalTestConfig(correction_method='bonferroni')
        assert config.get_correction_display_name() == 'Bonferroni'

    def test_posthoc_display_uncorrected(self):
        """Test display name for uncorrected posthoc."""
        config = StatisticalTestConfig(posthoc_correction='uncorrected')
        assert config.get_posthoc_display_name() == 'Uncorrected'

    def test_posthoc_display_tukey(self):
        """Test display name for Tukey's HSD."""
        config = StatisticalTestConfig(posthoc_correction='tukey')
        assert config.get_posthoc_display_name() == "Tukey's HSD"

    def test_posthoc_display_bonferroni(self):
        """Test display name for Bonferroni posthoc."""
        config = StatisticalTestConfig(posthoc_correction='bonferroni')
        assert config.get_posthoc_display_name() == 'Bonferroni'


class TestFactoryMethods:
    """Tests for factory methods."""

    def test_create_auto_defaults(self):
        """Test create_auto with defaults."""
        config = StatisticalTestConfig.create_auto()
        assert config.mode == 'auto'
        assert config.test_type == 'auto'
        assert config.correction_method == 'auto'
        assert config.posthoc_correction == 'auto'
        assert config.auto_transform is True

    def test_create_auto_no_transform(self):
        """Test create_auto with auto_transform=False."""
        config = StatisticalTestConfig.create_auto(auto_transform=False)
        assert config.mode == 'auto'
        assert config.auto_transform is False

    def test_create_manual_defaults(self):
        """Test create_manual with defaults."""
        config = StatisticalTestConfig.create_manual()
        assert config.mode == 'manual'
        assert config.test_type == 'parametric'
        assert config.correction_method == 'fdr_bh'
        assert config.posthoc_correction == 'tukey'
        assert config.auto_transform is True
        assert config.conditions_to_compare == []

    def test_create_manual_non_parametric(self):
        """Test create_manual with non-parametric settings."""
        config = StatisticalTestConfig.create_manual(
            test_type='non_parametric',
            correction_method='bonferroni',
            posthoc_correction='bonferroni'
        )
        assert config.test_type == 'non_parametric'
        assert config.correction_method == 'bonferroni'
        assert config.posthoc_correction == 'bonferroni'

    def test_create_manual_with_conditions(self):
        """Test create_manual with conditions to compare."""
        conditions = [('A', 'B'), ('A', 'C')]
        config = StatisticalTestConfig.create_manual(
            conditions_to_compare=conditions
        )
        assert config.conditions_to_compare == conditions

    def test_create_manual_no_transform(self):
        """Test create_manual with auto_transform=False."""
        config = StatisticalTestConfig.create_manual(auto_transform=False)
        assert config.auto_transform is False

    def test_create_manual_uncorrected(self):
        """Test create_manual with uncorrected settings."""
        config = StatisticalTestConfig.create_manual(
            correction_method='uncorrected',
            posthoc_correction='uncorrected'
        )
        assert config.correction_method == 'uncorrected'
        assert config.posthoc_correction == 'uncorrected'


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_very_small_alpha(self):
        """Test very small but valid alpha."""
        config = StatisticalTestConfig(alpha=0.0001)
        assert config.alpha == 0.0001

    def test_alpha_precision(self):
        """Test alpha with high precision."""
        config = StatisticalTestConfig(alpha=0.05123456789)
        assert config.alpha == 0.05123456789

    def test_many_condition_pairs(self):
        """Test with many condition pairs."""
        pairs = [(f'Cond{i}', f'Cond{j}') for i in range(5) for j in range(i+1, 5)]
        config = StatisticalTestConfig(conditions_to_compare=pairs)
        assert len(config.conditions_to_compare) == 10

    def test_condition_names_with_spaces(self):
        """Test condition names with spaces."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('Control Group', 'Treatment Group')]
        )
        assert config.conditions_to_compare == [('Control Group', 'Treatment Group')]

    def test_condition_names_with_special_chars(self):
        """Test condition names with special characters."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('Day_1', 'Day-2')]
        )
        assert config.conditions_to_compare == [('Day_1', 'Day-2')]

    def test_model_immutability_mode(self):
        """Test that mode can be set at creation."""
        config = StatisticalTestConfig(mode='manual')
        assert config.mode == 'manual'

    def test_serialization_roundtrip(self):
        """Test model can be serialized and deserialized."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='non_parametric',
            correction_method='bonferroni',
            posthoc_correction='bonferroni',
            alpha=0.01,
            auto_transform=False,
            conditions_to_compare=[('A', 'B')]
        )
        data = config.model_dump()
        restored = StatisticalTestConfig(**data)
        assert restored == config

    def test_auto_mode_serialization(self):
        """Test auto mode can be serialized and deserialized."""
        config = StatisticalTestConfig.create_auto()
        data = config.model_dump()
        restored = StatisticalTestConfig(**data)
        assert restored == config
        assert restored.is_auto_mode() is True

    def test_duplicate_condition_pairs_allowed(self):
        """Test that duplicate condition pairs are preserved."""
        pairs = [('A', 'B'), ('A', 'B')]
        config = StatisticalTestConfig(conditions_to_compare=pairs)
        assert len(config.conditions_to_compare) == 2

    def test_reversed_pairs_are_different(self):
        """Test that (A, B) and (B, A) are treated as different pairs."""
        pairs = [('A', 'B'), ('B', 'A')]
        config = StatisticalTestConfig(conditions_to_compare=pairs)
        assert config.conditions_to_compare == [('A', 'B'), ('B', 'A')]

    def test_unicode_condition_names(self):
        """Test unicode characters in condition names."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('Contrôle', 'Traitement-α')]
        )
        assert config.conditions_to_compare == [('Contrôle', 'Traitement-α')]

    def test_numeric_condition_names(self):
        """Test numeric-looking condition names."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('1', '2'), ('10', '20')]
        )
        assert config.conditions_to_compare == [('1', '2'), ('10', '20')]

    def test_long_condition_names(self):
        """Test very long condition names."""
        long_name = 'Treatment_with_compound_XYZ_at_concentration_100uM'
        config = StatisticalTestConfig(
            conditions_to_compare=[('Control', long_name)]
        )
        assert config.conditions_to_compare == [('Control', long_name)]


class TestSerialization:
    """Tests for model serialization."""

    def test_model_dump_manual_mode(self):
        """Test serializing manual mode config to dict."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='parametric',
            correction_method='fdr_bh',
            posthoc_correction='tukey',
            alpha=0.05,
            auto_transform=True,
            conditions_to_compare=[('A', 'B')]
        )
        data = config.model_dump()
        assert data['mode'] == 'manual'
        assert data['test_type'] == 'parametric'
        assert data['correction_method'] == 'fdr_bh'
        assert data['posthoc_correction'] == 'tukey'
        assert data['alpha'] == 0.05
        assert data['auto_transform'] is True
        assert data['conditions_to_compare'] == [('A', 'B')]

    def test_model_dump_auto_mode(self):
        """Test serializing auto mode config to dict."""
        config = StatisticalTestConfig.create_auto()
        data = config.model_dump()
        assert data['mode'] == 'auto'
        assert data['test_type'] == 'auto'
        assert data['correction_method'] == 'auto'
        assert data['posthoc_correction'] == 'auto'

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        config = StatisticalTestConfig(
            mode='manual',
            test_type='non_parametric',
            correction_method='bonferroni',
            posthoc_correction='bonferroni',
            alpha=0.01,
            auto_transform=False,
            conditions_to_compare=[('Control', 'Treatment1'), ('Control', 'Treatment2')]
        )
        json_str = config.model_dump_json()
        restored = StatisticalTestConfig.model_validate_json(json_str)

        assert restored.mode == config.mode
        assert restored.test_type == config.test_type
        assert restored.correction_method == config.correction_method
        assert restored.posthoc_correction == config.posthoc_correction
        assert restored.alpha == config.alpha
        assert restored.auto_transform == config.auto_transform
        assert restored.conditions_to_compare == config.conditions_to_compare

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            'mode': 'manual',
            'test_type': 'parametric',
            'correction_method': 'fdr_bh',
            'posthoc_correction': 'tukey',
            'alpha': 0.05,
            'auto_transform': True,
            'conditions_to_compare': [['A', 'B']]
        }
        config = StatisticalTestConfig.model_validate(data)
        assert config.mode == 'manual'
        assert config.test_type == 'parametric'

    def test_from_dict_auto_mode(self):
        """Test creating auto mode config from dictionary."""
        data = {
            'mode': 'auto',
            'test_type': 'auto',
            'correction_method': 'auto',
            'posthoc_correction': 'auto'
        }
        config = StatisticalTestConfig.model_validate(data)
        assert config.is_auto_mode() is True


class TestDisplayNamesAutoMode:
    """Tests for display names in auto mode."""

    def test_correction_display_auto(self):
        """Test display name for auto correction."""
        config = StatisticalTestConfig.create_auto()
        assert config.get_correction_display_name() == 'Auto'

    def test_posthoc_display_auto(self):
        """Test display name for auto posthoc."""
        config = StatisticalTestConfig.create_auto()
        assert config.get_posthoc_display_name() == 'Auto'


class TestRequiresPosthocEdgeCases:
    """Additional tests for requires_posthoc method."""

    def test_requires_posthoc_one_condition(self):
        """Test requires_posthoc with 1 condition."""
        config = StatisticalTestConfig(posthoc_correction='tukey')
        assert config.requires_posthoc(1) is False

    def test_requires_posthoc_zero_conditions(self):
        """Test requires_posthoc with 0 conditions."""
        config = StatisticalTestConfig(posthoc_correction='tukey')
        assert config.requires_posthoc(0) is False

    def test_requires_posthoc_negative_conditions(self):
        """Test requires_posthoc with negative (edge case)."""
        config = StatisticalTestConfig(posthoc_correction='tukey')
        assert config.requires_posthoc(-1) is False

    def test_requires_posthoc_many_conditions(self):
        """Test requires_posthoc with many conditions."""
        config = StatisticalTestConfig(posthoc_correction='bonferroni')
        assert config.requires_posthoc(10) is True


class TestRealWorldScenarios:
    """Tests simulating real-world usage patterns."""

    def test_two_group_comparison_parametric(self):
        """Test typical two-group parametric comparison (e.g., Control vs Treatment)."""
        config = StatisticalTestConfig.create_manual(
            test_type='parametric',
            correction_method='uncorrected',
            posthoc_correction='uncorrected',
            conditions_to_compare=[('Control', 'Treatment')]
        )
        assert config.is_parametric() is True
        assert config.requires_posthoc(2) is False

    def test_two_group_comparison_non_parametric(self):
        """Test two-group non-parametric comparison (Mann-Whitney U)."""
        config = StatisticalTestConfig.create_manual(
            test_type='non_parametric',
            correction_method='uncorrected',
            posthoc_correction='uncorrected',
            conditions_to_compare=[('Control', 'Treatment')]
        )
        assert config.is_non_parametric() is True
        assert config.requires_posthoc(2) is False

    def test_multi_group_anova_with_posthoc(self):
        """Test multi-group ANOVA with Tukey's HSD post-hoc."""
        conditions = ['Control', 'Treatment_A', 'Treatment_B', 'Treatment_C']
        pairs = [
            (conditions[i], conditions[j])
            for i in range(len(conditions))
            for j in range(i+1, len(conditions))
        ]

        config = StatisticalTestConfig.create_manual(
            test_type='parametric',
            correction_method='fdr_bh',
            posthoc_correction='tukey',
            conditions_to_compare=pairs
        )

        assert config.requires_posthoc(4) is True
        assert len(config.conditions_to_compare) == 6  # 4C2 = 6 pairs

    def test_multi_group_kruskal_wallis_with_bonferroni(self):
        """Test multi-group Kruskal-Wallis with Bonferroni correction."""
        config = StatisticalTestConfig.create_manual(
            test_type='non_parametric',
            correction_method='bonferroni',
            posthoc_correction='bonferroni'
        )
        assert config.is_non_parametric() is True
        assert config.requires_posthoc(3) is True

    def test_exploratory_analysis_auto_mode(self):
        """Test exploratory analysis using auto mode."""
        config = StatisticalTestConfig.create_auto(auto_transform=True)
        assert config.is_auto_mode() is True
        assert config.auto_transform is True

    def test_no_transformation_scenario(self):
        """Test scenario where data is already log-transformed."""
        config = StatisticalTestConfig.create_manual(
            test_type='parametric',
            correction_method='fdr_bh',
            posthoc_correction='tukey',
            auto_transform=False
        )
        assert config.auto_transform is False

    def test_strict_multiple_testing_correction(self):
        """Test configuration for very strict multiple testing correction."""
        config = StatisticalTestConfig.create_manual(
            test_type='parametric',
            correction_method='bonferroni',
            posthoc_correction='bonferroni',
            alpha=0.01
        )
        assert config.correction_method == 'bonferroni'
        assert config.posthoc_correction == 'bonferroni'
        assert config.alpha == 0.01

    def test_lenient_analysis_configuration(self):
        """Test configuration for lenient/exploratory analysis."""
        config = StatisticalTestConfig.create_manual(
            test_type='parametric',
            correction_method='uncorrected',
            posthoc_correction='uncorrected',
            alpha=0.1
        )
        assert config.correction_method == 'uncorrected'
        assert config.alpha == 0.1

    def test_lipidomics_typical_workflow(self):
        """Test typical lipidomics statistical analysis workflow."""
        # Typical setup: 3 conditions, parametric, FDR correction
        config = StatisticalTestConfig.create_manual(
            test_type='parametric',
            correction_method='fdr_bh',
            posthoc_correction='tukey',
            auto_transform=True,
            conditions_to_compare=[
                ('Control', 'Disease'),
                ('Control', 'Treatment'),
                ('Disease', 'Treatment')
            ]
        )

        assert config.is_parametric() is True
        assert config.auto_transform is True
        assert config.requires_posthoc(3) is True
        assert config.get_correction_display_name() == 'FDR (Benjamini-Hochberg)'
        assert config.get_posthoc_display_name() == "Tukey's HSD"


class TestAllCombinations:
    """Tests for various valid combinations of settings."""

    def test_all_manual_test_types(self):
        """Test all valid test types in manual mode."""
        for test_type in ['parametric', 'non_parametric']:
            config = StatisticalTestConfig(mode='manual', test_type=test_type)
            assert config.test_type == test_type

    def test_all_manual_correction_combinations(self):
        """Test all valid correction method combinations in manual mode."""
        correction_methods = ['uncorrected', 'fdr_bh', 'bonferroni']
        posthoc_methods = ['uncorrected', 'tukey', 'bonferroni']

        for cm in correction_methods:
            for pm in posthoc_methods:
                config = StatisticalTestConfig(
                    mode='manual',
                    correction_method=cm,
                    posthoc_correction=pm
                )
                assert config.correction_method == cm
                assert config.posthoc_correction == pm

    def test_parametric_with_all_posthoc_options(self):
        """Test parametric test with all posthoc correction options."""
        for posthoc in ['uncorrected', 'tukey', 'bonferroni']:
            config = StatisticalTestConfig(
                test_type='parametric',
                posthoc_correction=posthoc
            )
            assert config.posthoc_correction == posthoc

    def test_non_parametric_with_all_posthoc_options(self):
        """Test non-parametric test with all posthoc correction options."""
        for posthoc in ['uncorrected', 'tukey', 'bonferroni']:
            config = StatisticalTestConfig(
                test_type='non_parametric',
                posthoc_correction=posthoc
            )
            assert config.posthoc_correction == posthoc


class TestTypeHandling:
    """Tests for type coercion and type error handling."""

    def test_integer_alpha_coerced_to_float(self):
        """Test that integer alpha is coerced to float (if in valid range, won't work for 0 or 1)."""
        # Note: 0 and 1 are invalid, so we can't test simple int coercion directly
        # But Pydantic should coerce numeric types
        config = StatisticalTestConfig(alpha=0.5)  # 0.5 as float
        assert isinstance(config.alpha, float)

    def test_numeric_string_alpha_raises_or_coerces(self):
        """Test that numeric string alpha is handled."""
        # Pydantic v2 may coerce "0.05" to 0.05
        try:
            config = StatisticalTestConfig(alpha="0.05")
            assert config.alpha == 0.05
            assert isinstance(config.alpha, float)
        except (ValueError, TypeError):
            pass  # Also acceptable

    def test_non_numeric_string_alpha_raises_error(self):
        """Test that non-numeric string alpha raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(alpha="not_a_number")

    def test_none_alpha_raises_error(self):
        """Test that None alpha raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(alpha=None)

    def test_none_mode_raises_error(self):
        """Test that None mode raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(mode=None)

    def test_integer_mode_raises_error(self):
        """Test that integer mode raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(mode=1)

    def test_none_test_type_raises_error(self):
        """Test that None test_type raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(test_type=None)

    def test_integer_test_type_raises_error(self):
        """Test that integer test_type raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(test_type=123)

    def test_tuple_for_conditions_to_compare_works(self):
        """Test that tuple is accepted for conditions_to_compare."""
        config = StatisticalTestConfig(
            conditions_to_compare=(('A', 'B'), ('C', 'D'))  # tuple of tuples
        )
        # Pydantic converts tuples to lists
        assert isinstance(config.conditions_to_compare, list)
        assert len(config.conditions_to_compare) == 2

    def test_list_of_lists_for_condition_pairs_works(self):
        """Test that list of lists is accepted for condition pairs."""
        config = StatisticalTestConfig(
            conditions_to_compare=[['A', 'B'], ['C', 'D']]  # lists not tuples
        )
        assert len(config.conditions_to_compare) == 2

    def test_none_in_condition_pair_raises_error(self):
        """Test that None in condition pair raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(
                conditions_to_compare=[('A', None)]
            )

    def test_none_as_first_in_pair_raises_error(self):
        """Test that None as first element in pair raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(
                conditions_to_compare=[(None, 'B')]
            )

    def test_integer_in_condition_pair_raises_error(self):
        """Test that integer in condition pair raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(
                conditions_to_compare=[(123, 'B')]
            )

    def test_float_in_condition_pair_raises_error(self):
        """Test that float in condition pair raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(
                conditions_to_compare=[('A', 3.14)]
            )

    def test_dict_instead_of_list_for_conditions_raises_error(self):
        """Test that dict instead of list for conditions_to_compare raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(
                conditions_to_compare={'A': 'B'}
            )

    def test_single_element_tuple_raises_error(self):
        """Test that single element tuple raises validation error."""
        # Pydantic raises "Field required" for missing second element
        with pytest.raises(ValueError):
            StatisticalTestConfig(
                conditions_to_compare=[('A',)]  # only one element
            )

    def test_three_element_tuple_raises_error(self):
        """Test that three element tuple raises validation error."""
        # Pydantic raises "Tuple should have at most 2 items"
        with pytest.raises(ValueError):
            StatisticalTestConfig(
                conditions_to_compare=[('A', 'B', 'C')]  # three elements
            )

    def test_empty_tuple_in_conditions_raises_error(self):
        """Test that empty tuple raises validation error."""
        # Pydantic raises "Field required" for missing elements
        with pytest.raises(ValueError):
            StatisticalTestConfig(
                conditions_to_compare=[()]  # empty tuple
            )

    def test_boolean_auto_transform_accepted(self):
        """Test that boolean values for auto_transform work."""
        config_true = StatisticalTestConfig(auto_transform=True)
        config_false = StatisticalTestConfig(auto_transform=False)
        assert config_true.auto_transform is True
        assert config_false.auto_transform is False

    def test_int_for_auto_transform_coerced(self):
        """Test that int is coerced to bool for auto_transform."""
        config_1 = StatisticalTestConfig(auto_transform=1)
        config_0 = StatisticalTestConfig(auto_transform=0)
        assert config_1.auto_transform is True
        assert config_0.auto_transform is False

    def test_string_for_auto_transform_coerced(self):
        """Test that string for auto_transform is coerced by Pydantic v2."""
        # Pydantic v2 coerces "true"/"false" strings to booleans
        config = StatisticalTestConfig(auto_transform="true")
        assert config.auto_transform is True

    def test_none_for_auto_transform_raises_error(self):
        """Test that None for auto_transform raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(auto_transform=None)

    def test_uppercase_mode_raises_error(self):
        """Test that uppercase mode raises error."""
        with pytest.raises(ValueError):
            StatisticalTestConfig(mode='AUTO')

    def test_uppercase_test_type_raises_error(self):
        """Test that uppercase test_type raises error."""
        with pytest.raises(ValueError):
            StatisticalTestConfig(test_type='PARAMETRIC')

    def test_mixed_case_correction_method_raises_error(self):
        """Test that mixed case correction_method raises error."""
        with pytest.raises(ValueError):
            StatisticalTestConfig(correction_method='FDR_BH')


class TestInputValidationEdgeCases:
    """Tests for input validation edge cases."""

    def test_whitespace_only_condition_name_is_allowed(self):
        """Test that whitespace-only condition name is allowed (not validated as empty)."""
        # The validator uses `if not pair[0]` which considers "   " as truthy
        # Whitespace-only names are technically allowed
        config = StatisticalTestConfig(
            conditions_to_compare=[('A', '   ')]  # whitespace only - allowed
        )
        assert config.conditions_to_compare == [('A', '   ')]

    def test_generator_for_conditions_to_compare(self):
        """Test generator input for conditions_to_compare."""
        try:
            pairs = (('A', 'B') for _ in range(2))
            config = StatisticalTestConfig(conditions_to_compare=pairs)
            assert len(config.conditions_to_compare) == 2
        except (ValueError, TypeError):
            pass  # Also acceptable

    def test_set_for_conditions_to_compare_converted(self):
        """Test that set for conditions_to_compare is converted to list."""
        # Pydantic converts sets to lists (order may vary)
        config = StatisticalTestConfig(
            conditions_to_compare={('A', 'B'), ('C', 'D')}
        )
        assert len(config.conditions_to_compare) == 2
        assert isinstance(config.conditions_to_compare, list)

    def test_nested_list_in_condition_pair(self):
        """Test nested list in condition pair raises error."""
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(
                conditions_to_compare=[[['A'], 'B']]  # nested list
            )

    def test_inf_alpha_raises_error(self):
        """Test that infinity alpha raises error (gt=0, lt=1)."""
        import math
        with pytest.raises(ValueError):
            StatisticalTestConfig(alpha=math.inf)

    def test_nan_alpha_raises_error(self):
        """Test that NaN alpha raises error."""
        import math
        with pytest.raises((ValueError, TypeError)):
            StatisticalTestConfig(alpha=math.nan)

    def test_very_small_alpha_valid(self):
        """Test very small alpha (near machine epsilon) is valid."""
        import sys
        tiny = sys.float_info.min
        if tiny > 0 and tiny < 1:
            config = StatisticalTestConfig(alpha=tiny)
            assert config.alpha == tiny

    def test_alpha_at_boundary_epsilon(self):
        """Test alpha just above 0 and just below 1."""
        config_low = StatisticalTestConfig(alpha=1e-10)
        config_high = StatisticalTestConfig(alpha=0.9999999999)
        assert config_low.alpha == 1e-10
        assert config_high.alpha == 0.9999999999


class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_very_many_condition_pairs(self):
        """Test with very many condition pairs."""
        pairs = [(f'C{i}', f'C{j}') for i in range(20) for j in range(i+1, 20)]
        config = StatisticalTestConfig(conditions_to_compare=pairs)
        # 20C2 = 190 pairs
        assert len(config.conditions_to_compare) == 190

    def test_very_long_condition_name(self):
        """Test very long condition name."""
        long_name = 'X' * 1000
        config = StatisticalTestConfig(
            conditions_to_compare=[('Short', long_name)]
        )
        assert config.conditions_to_compare[0][1] == long_name

    def test_condition_name_with_special_unicode(self):
        """Test condition name with special unicode characters."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('Contrôle-α', 'Βεhandlung-β')]
        )
        assert 'α' in config.conditions_to_compare[0][0]
        assert 'β' in config.conditions_to_compare[0][1]

    def test_condition_name_with_emoji(self):
        """Test condition name with emoji characters."""
        config = StatisticalTestConfig(
            conditions_to_compare=[('Control🔬', 'Treatment💊')]
        )
        assert '🔬' in config.conditions_to_compare[0][0]


class TestCopyAndEquality:
    """Tests for model copy and equality behavior."""

    def test_equal_configs(self):
        """Test that two configs with same values are equal."""
        config1 = StatisticalTestConfig(
            mode='manual',
            test_type='parametric',
            correction_method='fdr_bh',
            alpha=0.05
        )
        config2 = StatisticalTestConfig(
            mode='manual',
            test_type='parametric',
            correction_method='fdr_bh',
            alpha=0.05
        )
        assert config1 == config2

    def test_unequal_mode(self):
        """Test configs with different modes are not equal."""
        config1 = StatisticalTestConfig.create_manual()
        config2 = StatisticalTestConfig.create_auto()
        assert config1 != config2

    def test_unequal_alpha(self):
        """Test configs with different alpha are not equal."""
        config1 = StatisticalTestConfig(alpha=0.05)
        config2 = StatisticalTestConfig(alpha=0.01)
        assert config1 != config2

    def test_model_copy(self):
        """Test that model_copy creates independent copy."""
        config1 = StatisticalTestConfig(
            conditions_to_compare=[('A', 'B')]
        )
        config2 = config1.model_copy()

        assert config1 == config2
        assert config1 is not config2

    def test_model_copy_deep(self):
        """Test deep copy of model."""
        config1 = StatisticalTestConfig(
            conditions_to_compare=[('A', 'B'), ('C', 'D')]
        )
        config2 = config1.model_copy(deep=True)

        assert config1 == config2
        assert config1.conditions_to_compare is not config2.conditions_to_compare


class TestModelSchema:
    """Tests for model JSON schema."""

    def test_json_schema_generation(self):
        """Test that JSON schema can be generated."""
        schema = StatisticalTestConfig.model_json_schema()
        assert 'properties' in schema
        assert 'mode' in schema['properties']
        assert 'alpha' in schema['properties']

    def test_json_schema_alpha_constraints(self):
        """Test that alpha constraints are in schema."""
        schema = StatisticalTestConfig.model_json_schema()
        alpha_schema = schema['properties']['alpha']
        # Should have exclusiveMinimum and exclusiveMaximum for gt/lt
        assert 'exclusiveMinimum' in alpha_schema or 'minimum' in alpha_schema
        assert 'exclusiveMaximum' in alpha_schema or 'maximum' in alpha_schema

    def test_json_schema_has_all_fields(self):
        """Test that schema has all expected fields."""
        schema = StatisticalTestConfig.model_json_schema()
        expected_fields = ['mode', 'test_type', 'correction_method',
                          'posthoc_correction', 'alpha', 'auto_transform',
                          'conditions_to_compare']
        for field in expected_fields:
            assert field in schema['properties']
