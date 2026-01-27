"""Unit tests for ExperimentConfig model."""
import pytest
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

    def test_from_user_input(self):
        """Test factory method."""
        config = ExperimentConfig.from_user_input(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[4, 4]
        )
        assert config.n_conditions == 2
        assert len(config.full_samples_list) == 8

    def test_invalid_n_conditions_zero(self):
        """Test that zero conditions raises error."""
        with pytest.raises(ValueError):
            ExperimentConfig(
                n_conditions=0,
                conditions_list=[],
                number_of_samples_list=[]
            )

    def test_invalid_n_conditions_negative(self):
        """Test that negative conditions raises error."""
        with pytest.raises(ValueError):
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
        """Test that removing non-existent samples is handled gracefully."""
        config = ExperimentConfig(
            n_conditions=2,
            conditions_list=['WT', 'KO'],
            number_of_samples_list=[3, 3]
        )
        # s100 doesn't exist - should be ignored
        new_config = config.without_samples(['s1', 's100'])

        assert new_config.number_of_samples_list == [2, 3]

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
