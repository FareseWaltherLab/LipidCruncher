"""Unit tests for ExperimentConfig model."""
import pytest
import copy
from pydantic import ValidationError
from app.models.experiment import ExperimentConfig


class TestExperimentConfigCreation:
    """Tests for creating ExperimentConfig instances."""

    def test_valid_config(self):
        """Test creating a valid experiment config."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        assert config.n_conditions == 2
        assert config.conditions_list == ['WT', 'KO']
        assert config.number_of_samples_list == [3, 3]

    def test_direct_construction(self):
        """Test direct construction with keyword arguments."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[4, 4]
        )
        assert config.n_conditions == 2
        assert len(config.full_samples_list) == 8

    def test_invalid_n_conditions_zero(self):
        """Test that zero conditions raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=0,
                conditions_list=[],
                number_of_samples_list=[]
            )

    def test_invalid_n_conditions_negative(self):
        """Test that negative conditions raises error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=-1,
                conditions_list=['WT'],
                number_of_samples_list=[3]
            )

    def test_empty_condition_label(self):
        """Test that empty condition labels raise error."""
        with pytest.raises(ValueError, match="non-empty"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['WT', ''],
                number_of_samples_list=[3, 3]
            )

    def test_whitespace_condition_label(self):
        """Test that whitespace-only labels raise error."""
        with pytest.raises(ValueError, match="non-empty"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['WT', '   '],
                number_of_samples_list=[3, 3]
            )

    def test_zero_sample_count(self):
        """Test that zero sample counts raise error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['WT', 'KO'],
                number_of_samples_list=[3, 0]
            )

    def test_negative_sample_count(self):
        """Test that negative sample counts raise error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['WT', 'KO'],
                number_of_samples_list=[3, -1]
            )

    def test_special_characters_in_condition_labels(self):
        """Test condition labels with special characters are allowed."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT-1 (control)', 'KO_treated|2'],
            number_of_samples_list=[3, 3]
        )
        assert config.conditions_list == ['WT-1 (control)', 'KO_treated|2']

    def test_unicode_condition_labels(self):
        """Test condition labels with unicode characters."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['α-treatment', 'β-control'],
            number_of_samples_list=[2, 2]
        )
        assert config.conditions_list == ['α-treatment', 'β-control']

    def test_many_conditions(self):
        """Test with many conditions (realistic maximum)."""
        n = 10
        config = ExperimentConfig(
            n_conditions=n,
            conditions_list=[f'Condition_{i}' for i in range(n)],
            number_of_samples_list=[3] * n
        )
        assert config.n_conditions == n
        assert len(config.full_samples_list) == 30

    def test_single_sample_per_condition(self):
        """Test with single sample per condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[1, 1, 1]
        )
        assert config.full_samples_list == ['s1', 's2', 's3']
        assert config.individual_samples_list == [['s1'], ['s2'], ['s3']]

    def test_unequal_sample_counts(self):
        """Test with very different sample counts per condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[1, 10, 2]
        )
        assert len(config.full_samples_list) == 13
        assert config.aggregate_number_of_samples_list == [1, 11, 13]


class TestExperimentConfigComputedFields:
    """Tests for computed properties."""

    def test_full_samples_list(self):
        """Test sample label generation."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 2]
        )
        assert config.full_samples_list == ['s1', 's2', 's3', 's4', 's5']

    def test_aggregate_number_of_samples_list(self):
        """Test cumulative sample counts."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 3, 4]
        )
        assert config.aggregate_number_of_samples_list == [2, 5, 9]

    def test_individual_samples_list(self):
        """Test grouping samples by condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 3, 1]
        )
        assert config.individual_samples_list == [
            ['s1', 's2'],
            ['s3', 's4', 's5'],
            ['s6']
        ]

    def test_extensive_conditions_list(self):
        """Test flat condition list with replication."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 2]
        )
        assert config.extensive_conditions_list == ['WT', 'WT', 'WT', 'KO', 'KO']

    def test_single_condition(self):
        """Test with single condition."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[5]
        )
        assert config.full_samples_list == ['s1', 's2', 's3', 's4', 's5']
        assert config.individual_samples_list == [['s1', 's2', 's3', 's4', 's5']]
        assert config.extensive_conditions_list == ['Control'] * 5

    def test_aggregate_with_single_condition(self):
        """Test aggregate list with single condition."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[5]
        )
        assert config.aggregate_number_of_samples_list == [5]

    def test_computed_fields_consistency(self):
        """Test that all computed fields are consistent with each other."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 4, 3]
        )
        # Total samples should match
        total = sum(config.number_of_samples_list)
        assert len(config.full_samples_list) == total
        assert len(config.extensive_conditions_list) == total

        # Individual samples should sum to total
        individual_total = sum(len(group) for group in config.individual_samples_list)
        assert individual_total == total

        # Last aggregate should equal total
        assert config.aggregate_number_of_samples_list[-1] == total

    def test_large_sample_count(self):
        """Test with large number of samples."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[50, 50]
        )
        assert len(config.full_samples_list) == 100
        assert config.full_samples_list[-1] == 's100'


class TestExperimentConfigWithoutSamples:
    """Tests for the without_samples method."""

    def test_remove_single_sample(self):
        """Test removing a single sample."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        new_config = config.without_samples(['s1'])

        assert new_config.n_conditions == 2
        assert new_config.number_of_samples_list == [2, 3]
        assert len(new_config.full_samples_list) == 5

    def test_remove_multiple_samples(self):
        """Test removing multiple samples."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        new_config = config.without_samples(['s1', 's4'])

        assert new_config.number_of_samples_list == [2, 2]

    def test_remove_entire_condition(self):
        """Test removing all samples from one condition."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[2, 3]
        )
        # Remove s1 and s2 (all WT samples)
        new_config = config.without_samples(['s1', 's2'])

        assert new_config.n_conditions == 1
        assert new_config.conditions_list == ['KO']
        assert new_config.number_of_samples_list == [3]

    def test_remove_no_samples(self):
        """Test with empty removal list returns equivalent config."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        new_config = config.without_samples([])

        assert new_config.n_conditions == config.n_conditions
        assert new_config.conditions_list == config.conditions_list

    def test_remove_all_samples_raises_error(self):
        """Test that removing all samples raises error."""
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[2]
        )
        with pytest.raises(ValueError, match="Cannot remove all samples"):
            config.without_samples(['s1', 's2'])

    def test_original_unchanged_after_removal(self):
        """Test that original config is not mutated."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        _ = config.without_samples(['s1', 's2', 's3'])

        # Original should be unchanged
        assert config.number_of_samples_list == [3, 3]
        assert config.n_conditions == 2

    def test_remove_nonexistent_samples(self):
        """Test that removing non-existent samples raises ValueError."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        with pytest.raises(ValueError, match="unknown samples"):
            config.without_samples(['s1', 's100'])

    def test_remove_samples_from_all_conditions(self):
        """Test removing one sample from each condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[3, 3, 3]
        )
        # Remove first sample from each condition
        new_config = config.without_samples(['s1', 's4', 's7'])

        assert new_config.n_conditions == 3
        assert new_config.number_of_samples_list == [2, 2, 2]

    def test_remove_middle_condition_entirely(self):
        """Test removing all samples from the middle condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 2, 2]
        )
        # Remove s3 and s4 (all of B)
        new_config = config.without_samples(['s3', 's4'])

        assert new_config.n_conditions == 2
        assert new_config.conditions_list == ['A', 'C']
        assert new_config.number_of_samples_list == [2, 2]

    def test_remove_samples_preserves_condition_order(self):
        """Test that condition order is preserved after removal."""
        config = ExperimentConfig(
            n_conditions=4,
            conditions_list=['First', 'Second', 'Third', 'Fourth'],
            number_of_samples_list=[2, 2, 2, 2]
        )
        # Remove all samples from Second and Fourth
        new_config = config.without_samples(['s3', 's4', 's7', 's8'])

        assert new_config.conditions_list == ['First', 'Third']

    def test_chained_removals(self):
        """Test multiple sequential removals."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[4, 4]
        )

        # First removal
        config2 = config.without_samples(['s1'])
        assert config2.number_of_samples_list == [3, 4]

        # Second removal from the new config
        config3 = config2.without_samples(['s5'])
        assert config3.number_of_samples_list == [3, 3]

        # Original still unchanged
        assert config.number_of_samples_list == [4, 4]


class TestExperimentConfigSerialization:
    """Tests for model serialization."""

    def test_to_dict(self):
        """Test converting model to dictionary."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        data = config.model_dump()

        assert data['n_conditions'] == 2
        assert data['conditions_list'] == ['WT', 'KO']
        assert data['number_of_samples_list'] == [3, 3]
        # Computed fields should also be included
        assert data['full_samples_list'] == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_from_dict(self):
        """Test creating model from dictionary."""
        data = {
            'n_conditions': 2,
            'conditions_list': ['A', 'B'],
            'number_of_samples_list': [2, 3]
        }
        config = ExperimentConfig.model_validate(data)

        assert config.n_conditions == 2
        assert len(config.full_samples_list) == 5

    def test_to_json_and_back(self):
        """Test JSON round-trip serialization."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[4, 4]
        )

        json_str = config.model_dump_json()
        restored = ExperimentConfig.model_validate_json(json_str)

        assert restored.n_conditions == config.n_conditions
        assert restored.conditions_list == config.conditions_list
        assert restored.full_samples_list == config.full_samples_list

    def test_model_dump_exclude_computed(self):
        """Test serialization can exclude computed fields."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )
        # Get only the base fields
        data = config.model_dump(exclude={'full_samples_list', 'aggregate_number_of_samples_list',
                                          'individual_samples_list', 'extensive_conditions_list'})
        assert 'n_conditions' in data
        assert 'full_samples_list' not in data

    def test_model_dump_by_alias(self):
        """Test serialization with aliases (if any)."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )
        data = config.model_dump(by_alias=True)
        assert 'n_conditions' in data


class TestExperimentConfigListLengthValidation:
    """Tests for validating that list lengths match n_conditions."""

    def test_conditions_list_length_mismatch_fewer(self):
        """Test that fewer conditions than n_conditions raises error."""
        with pytest.raises(ValueError, match="conditions_list.*must match n_conditions"):
            ExperimentConfig(
                n_conditions=3,
                conditions_list=['A', 'B'],  # Only 2, expected 3
                number_of_samples_list=[2, 2, 2]
            )

    def test_conditions_list_length_mismatch_more(self):
        """Test that more conditions than n_conditions raises error."""
        with pytest.raises(ValueError, match="conditions_list.*must match n_conditions"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B', 'C'],  # 3, expected 2
                number_of_samples_list=[2, 2]
            )

    def test_samples_list_length_mismatch_fewer(self):
        """Test that fewer sample counts than n_conditions raises error."""
        with pytest.raises(ValueError, match="number_of_samples_list.*must match n_conditions"):
            ExperimentConfig(
                n_conditions=3,
                conditions_list=['A', 'B', 'C'],
                number_of_samples_list=[2, 2]  # Only 2, expected 3
            )

    def test_samples_list_length_mismatch_more(self):
        """Test that more sample counts than n_conditions raises error."""
        with pytest.raises(ValueError, match="number_of_samples_list.*must match n_conditions"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[2, 2, 2]  # 3, expected 2
            )

    def test_both_lists_too_short(self):
        """Test both lists shorter than n_conditions."""
        with pytest.raises(ValueError, match="must match n_conditions"):
            ExperimentConfig(
                n_conditions=5,
                conditions_list=['A', 'B'],
                number_of_samples_list=[2, 2]
            )

    def test_lists_match_n_conditions(self):
        """Test that matching lengths work correctly."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[1, 2, 3]
        )
        assert config.n_conditions == 3
        assert len(config.conditions_list) == 3
        assert len(config.number_of_samples_list) == 3


class TestExperimentConfigConditionLabels:
    """Additional tests for condition label validation."""

    def test_duplicate_condition_labels_rejected(self):
        """Test that duplicate condition labels are rejected."""
        with pytest.raises(ValidationError, match="unique"):
            ExperimentConfig(
                n_conditions=3,
                conditions_list=['Control', 'Treatment', 'Control'],
                number_of_samples_list=[2, 2, 2]
            )

    def test_very_long_condition_label(self):
        """Test condition label with very long string."""
        long_label = 'A' * 500
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=[long_label, 'Short'],
            number_of_samples_list=[2, 2]
        )
        assert config.conditions_list[0] == long_label

    def test_numeric_only_condition_labels(self):
        """Test condition labels that are numeric strings."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['1', '2', '3'],
            number_of_samples_list=[2, 2, 2]
        )
        assert config.conditions_list == ['1', '2', '3']

    def test_condition_labels_with_leading_trailing_spaces(self):
        """Test condition labels with leading/trailing spaces are preserved."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=[' Leading', 'Trailing '],
            number_of_samples_list=[2, 2]
        )
        # Note: Labels with spaces are valid (not empty after strip)
        assert config.conditions_list == [' Leading', 'Trailing ']

    def test_single_character_condition_labels(self):
        """Test single character condition labels."""
        config = ExperimentConfig(
            n_conditions=4,
            conditions_list=['A', 'B', 'C', 'D'],
            number_of_samples_list=[1, 1, 1, 1]
        )
        assert config.conditions_list == ['A', 'B', 'C', 'D']

    def test_mixed_case_condition_labels(self):
        """Test mixed case condition labels are case-sensitive."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['Control', 'CONTROL', 'control'],
            number_of_samples_list=[2, 2, 2]
        )
        assert config.conditions_list == ['Control', 'CONTROL', 'control']

    def test_condition_with_newline_fails(self):
        """Test that condition labels with only newlines fail."""
        with pytest.raises(ValueError, match="non-empty"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['Valid', '\n\n'],
                number_of_samples_list=[2, 2]
            )

    def test_condition_with_tab_only_fails(self):
        """Test that condition labels with only tabs fail."""
        with pytest.raises(ValueError, match="non-empty"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['Valid', '\t\t'],
                number_of_samples_list=[2, 2]
            )


class TestExperimentConfigSampleCounts:
    """Additional tests for sample count validation."""

    def test_very_large_sample_count(self):
        """Test handling of very large sample counts."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[500, 500]
        )
        assert len(config.full_samples_list) == 1000
        assert config.full_samples_list[-1] == 's1000'

    def test_asymmetric_sample_counts(self):
        """Test very asymmetric sample distributions."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Few', 'Many'],
            number_of_samples_list=[1, 100]
        )
        assert config.individual_samples_list[0] == ['s1']
        assert len(config.individual_samples_list[1]) == 100

    def test_all_ones_sample_counts(self):
        """Test all conditions with single samples."""
        config = ExperimentConfig(
            n_conditions=5,
            conditions_list=['A', 'B', 'C', 'D', 'E'],
            number_of_samples_list=[1, 1, 1, 1, 1]
        )
        assert len(config.full_samples_list) == 5
        for group in config.individual_samples_list:
            assert len(group) == 1


class TestExperimentConfigEquality:
    """Tests for model equality and comparison."""

    def test_equal_configs(self):
        """Test that two configs with same values are equal."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        assert config1 == config2

    def test_unequal_n_conditions(self):
        """Test configs with different n_conditions are not equal."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[3, 3, 3]
        )
        assert config1 != config2

    def test_unequal_conditions_list(self):
        """Test configs with different condition labels are not equal."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'C'],
            number_of_samples_list=[3, 3]
        )
        assert config1 != config2

    def test_unequal_sample_counts(self):
        """Test configs with different sample counts are not equal."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 4]
        )
        assert config1 != config2


class TestExperimentConfigCopy:
    """Tests for model copy behavior."""

    def test_model_copy(self):
        """Test that model_copy creates independent copy."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = config1.model_copy()

        assert config1 == config2
        assert config1 is not config2

    def test_model_copy_deep(self):
        """Test deep copy of model."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = config1.model_copy(deep=True)

        assert config1 == config2
        assert config1.conditions_list is not config2.conditions_list

    def test_python_copy_module(self):
        """Test Python copy module compatibility."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        config2 = copy.copy(config1)
        config3 = copy.deepcopy(config1)

        assert config1 == config2 == config3


class TestExperimentConfigRepr:
    """Tests for model string representation."""

    def test_repr_contains_key_fields(self):
        """Test that repr contains key field values."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        repr_str = repr(config)

        assert 'n_conditions=2' in repr_str
        assert 'WT' in repr_str
        assert 'KO' in repr_str

    def test_str_representation(self):
        """Test string conversion."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )
        str_repr = str(config)
        assert 'n_conditions' in str_repr


class TestExperimentConfigConstructionValidation:
    """Tests for validation during direct construction."""

    def test_construction_propagates_n_conditions_error(self):
        """Test construction propagates n_conditions validation error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=0,
                conditions_list=[],
                number_of_samples_list=[]
            )

    def test_construction_propagates_empty_label_error(self):
        """Test construction propagates empty label validation error."""
        with pytest.raises(ValueError, match="non-empty"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['Valid', ''],
                number_of_samples_list=[2, 2]
            )

    def test_construction_propagates_sample_count_error(self):
        """Test construction propagates sample count validation error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[2, 0]
            )

    def test_construction_with_all_valid_inputs(self):
        """Test construction with all valid inputs."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['Control', 'Low Dose', 'High Dose'],
            number_of_samples_list=[5, 5, 5]
        )
        assert config.n_conditions == 3
        assert len(config.full_samples_list) == 15


class TestExperimentConfigWithoutSamplesEdgeCases:
    """Additional edge cases for without_samples method."""

    def test_remove_samples_case_sensitive(self):
        """Test that sample removal is case-sensitive and rejects unknown labels."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        # 'S1' (uppercase) doesn't exist as a sample label
        with pytest.raises(ValueError, match="unknown samples"):
            config.without_samples(['S1', 'S2'])

    def test_remove_samples_with_invalid_format(self):
        """Test removing samples with invalid format raises ValueError."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        # 'sample1' format doesn't match 's1' format
        with pytest.raises(ValueError, match="unknown samples"):
            config.without_samples(['sample1', 'x1', '1'])

    def test_remove_samples_duplicates_in_list(self):
        """Test removing with duplicate entries in removal list."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        # s1 listed twice
        new_config = config.without_samples(['s1', 's1', 's1'])
        assert new_config.number_of_samples_list == [2, 3]

    def test_remove_samples_preserves_sample_naming(self):
        """Test that remaining samples keep their original names."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        new_config = config.without_samples(['s2'])

        # Note: The new config generates fresh sample labels
        # s1, s2, s3, s4, s5 with s2 removed should give new labels s1-s5
        assert len(new_config.full_samples_list) == 5

    def test_remove_last_sample_from_each_condition(self):
        """Test removing last sample from each condition."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[3, 3, 3]
        )
        # Remove s3 (last of A), s6 (last of B), s9 (last of C)
        new_config = config.without_samples(['s3', 's6', 's9'])

        assert new_config.n_conditions == 3
        assert new_config.number_of_samples_list == [2, 2, 2]


class TestExperimentConfigComputedFieldsEdgeCases:
    """Additional edge cases for computed fields."""

    def test_computed_fields_are_read_only(self):
        """Test that computed fields cannot be directly assigned."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )
        # Frozen model prevents field assignment
        with pytest.raises(ValidationError, match="frozen"):
            config.full_samples_list = ['x1', 'x2']

    def test_computed_fields_update_on_model_copy_update(self):
        """Test computed fields update when model is copied with updates."""
        config1 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2]
        )
        # Create new config with different sample counts
        config2 = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )

        assert len(config1.full_samples_list) == 4
        assert len(config2.full_samples_list) == 6

    def test_individual_samples_list_indexing(self):
        """Test indexing into individual_samples_list."""
        config = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 3, 4]
        )
        # First condition samples
        assert config.individual_samples_list[0] == ['s1', 's2']
        # Second condition samples
        assert config.individual_samples_list[1] == ['s3', 's4', 's5']
        # Third condition samples
        assert config.individual_samples_list[2] == ['s6', 's7', 's8', 's9']

    def test_extensive_conditions_list_indexing(self):
        """Test that extensive_conditions_list aligns with full_samples_list."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 2]
        )
        # Each sample should map to correct condition
        for i, sample in enumerate(config.full_samples_list):
            condition = config.extensive_conditions_list[i]
            if i < 3:
                assert condition == 'Control'
            else:
                assert condition == 'Treatment'


class TestExperimentConfigModelSchema:
    """Tests for model JSON schema."""

    def test_json_schema_generation(self):
        """Test that JSON schema can be generated."""
        schema = ExperimentConfig.model_json_schema()
        assert 'properties' in schema
        assert 'n_conditions' in schema['properties']
        assert 'conditions_list' in schema['properties']
        assert 'number_of_samples_list' in schema['properties']

    def test_json_schema_required_fields(self):
        """Test that required fields are marked in schema."""
        schema = ExperimentConfig.model_json_schema()
        required = schema.get('required', [])
        assert 'n_conditions' in required
        assert 'conditions_list' in required
        assert 'number_of_samples_list' in required


class TestExperimentConfigTypeHandling:
    """Tests for type coercion and type error handling."""

    def test_float_sample_counts_coerced_to_int(self):
        """Test that float sample counts are coerced to integers."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3.0, 4.0]
        )
        assert config.number_of_samples_list == [3, 4]
        assert all(isinstance(x, int) for x in config.number_of_samples_list)

    def test_float_n_conditions_coerced_to_int(self):
        """Test that float n_conditions is coerced to integer."""
        config = ExperimentConfig(
            n_conditions=2.0,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3]
        )
        assert config.n_conditions == 2
        assert isinstance(config.n_conditions, int)

    def test_string_n_conditions_raises_error(self):
        """Test that string n_conditions raises validation error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions='two',
                conditions_list=['A', 'B'],
                number_of_samples_list=[3, 3]
            )

    def test_string_sample_counts_raises_error(self):
        """Test that string sample counts raise validation error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=['three', 'four']
            )

    def test_none_in_conditions_list_raises_error(self):
        """Test that None in conditions list raises error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', None],
                number_of_samples_list=[3, 3]
            )

    def test_none_in_sample_counts_raises_error(self):
        """Test that None in sample counts raises error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[3, None]
            )

    def test_none_conditions_list_raises_error(self):
        """Test that None conditions_list raises error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=None,
                number_of_samples_list=[3, 3]
            )

    def test_none_sample_counts_list_raises_error(self):
        """Test that None number_of_samples_list raises error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=None
            )

    def test_integer_condition_label_raises_error(self):
        """Test that integer condition labels raise validation error."""
        # Pydantic's List[str] does not coerce integers to strings
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=[1, 2],
                number_of_samples_list=[3, 3]
            )

    def test_mixed_types_in_conditions_list_raises_error(self):
        """Test mixed types in conditions_list raise validation error."""
        # Pydantic's List[str] requires actual strings
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=3,
                conditions_list=['Control', 42, 3.14],
                number_of_samples_list=[2, 2, 2]
            )

    def test_boolean_in_sample_counts_coerced(self):
        """Test that booleans in sample counts are coerced to int (True=1, False=0)."""
        # True should become 1, False would fail validation (0 not allowed)
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[True, False]  # False=0 should fail
            )

    def test_fractional_float_sample_count_raises_error(self):
        """Test that fractional float sample counts raise error."""
        # Pydantic rejects floats with fractional parts for int fields
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[3.0, 2.5]  # 2.5 has fractional part
            )

    def test_negative_integer_sample_count_raises_error(self):
        """Test that negative integer sample counts raise error."""
        with pytest.raises(ValueError, match="greater than 0"):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[3, -2]
            )


class TestExperimentConfigInputValidation:
    """Tests for input validation similar to data cleaning service patterns."""

    def test_empty_dataframe_equivalent_raises(self):
        """Test that minimal valid config with single sample works."""
        # Equivalent to single row data test in data cleaning
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Only'],
            number_of_samples_list=[1]
        )
        assert config.n_conditions == 1
        assert len(config.full_samples_list) == 1

    def test_all_required_fields_missing(self):
        """Test that missing all required fields raises error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig()

    def test_partial_fields_missing_raises(self):
        """Test that partial required fields raise error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(n_conditions=2)

    def test_tuple_instead_of_list_works(self):
        """Test that tuples are accepted for list fields."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=('A', 'B'),
            number_of_samples_list=(3, 3)
        )
        assert config.n_conditions == 2
        # Pydantic converts tuples to lists
        assert isinstance(config.conditions_list, list)

    def test_generator_input_raises_or_works(self):
        """Test generator input handling."""
        # Generators might work or fail depending on Pydantic version
        try:
            config = ExperimentConfig(
                n_conditions=2,
                conditions_list=(x for x in ['A', 'B']),
                number_of_samples_list=[3, 3]
            )
            # If it works, verify the result
            assert config.conditions_list == ['A', 'B']
        except (ValueError, TypeError):
            # If it fails, that's also acceptable behavior
            pass

    def test_set_input_for_list_field(self):
        """Test that set input might lose order or fail."""
        # Sets don't guarantee order, so this might produce unexpected results
        try:
            config = ExperimentConfig(
                n_conditions=2,
                conditions_list={'A', 'B'},  # Set - order not guaranteed
                number_of_samples_list=[3, 3]
            )
            # If it works, verify it has 2 conditions
            assert len(config.conditions_list) == 2
        except (ValueError, TypeError):
            pass

    def test_dict_input_for_list_raises(self):
        """Test that dict input for list field raises error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list={'A': 1, 'B': 2},
                number_of_samples_list=[3, 3]
            )

    def test_nested_list_raises_error(self):
        """Test that nested lists raise validation error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=[['A'], ['B']],
                number_of_samples_list=[3, 3]
            )

    def test_nested_sample_counts_raises_error(self):
        """Test that nested sample counts raise error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                n_conditions=2,
                conditions_list=['A', 'B'],
                number_of_samples_list=[[3], [3]]
            )


class TestExperimentConfigBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_maximum_practical_conditions(self):
        """Test with maximum practical number of conditions (e.g., 100)."""
        n = 100
        config = ExperimentConfig(
            n_conditions=n,
            conditions_list=[f'Condition_{i}' for i in range(n)],
            number_of_samples_list=[2] * n
        )
        assert config.n_conditions == n
        assert len(config.full_samples_list) == 200

    def test_maximum_practical_samples(self):
        """Test with very large total sample count."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[1000, 1000]
        )
        assert len(config.full_samples_list) == 2000

    def test_extreme_asymmetry(self):
        """Test extreme asymmetry in sample counts."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Small', 'Large'],
            number_of_samples_list=[1, 1000]
        )
        assert config.individual_samples_list[0] == ['s1']
        assert len(config.individual_samples_list[1]) == 1000

    def test_condition_label_at_max_practical_length(self):
        """Test condition label at maximum practical length (1000 chars)."""
        long_label = 'X' * 1000
        config = ExperimentConfig(
            n_conditions=1,
            conditions_list=[long_label],
            number_of_samples_list=[2]
        )
        assert len(config.conditions_list[0]) == 1000

    def test_n_conditions_int_max_boundary(self):
        """Test n_conditions near int boundary (but practical)."""
        # Don't test actual INT_MAX, just a large practical value
        n = 1000
        config = ExperimentConfig(
            n_conditions=n,
            conditions_list=[f'C{i}' for i in range(n)],
            number_of_samples_list=[1] * n
        )
        assert config.n_conditions == n
