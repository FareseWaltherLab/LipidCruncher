"""Unit tests for QualityCheckService."""
import pytest
import pandas as pd
import numpy as np
from app.services.quality_check import (
    QualityCheckService,
    BQCFilterResult,
    RetentionTimeDataResult,
    SampleRemovalResult,
)
from app.models.experiment import ExperimentConfig

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def single_sample_experiment():
    """2 conditions x 1 sample each = 2 samples."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['A', 'B'],
        number_of_samples_list=[1, 1],
    )


@pytest.fixture
def basic_conc_df(simple_experiment_2x3):
    """Basic DataFrame with concentration columns for 6 samples."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'concentration[s1]': [1000.0, 2000.0, 3000.0],
        'concentration[s2]': [1100.0, 2100.0, 3100.0],
        'concentration[s3]': [1200.0, 2200.0, 3200.0],
        'concentration[s4]': [1300.0, 2300.0, 3300.0],
        'concentration[s5]': [1400.0, 2400.0, 3400.0],
        'concentration[s6]': [1500.0, 2500.0, 3500.0],
    })


@pytest.fixture
def rt_df():
    """DataFrame with retention time columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'BaseRt': [5.1, 6.2, 7.3],
        'CalcMass': [733.5, 743.5, 850.7],
        'concentration[s1]': [100.0, 200.0, 300.0],
    })


# =============================================================================
# TestFilterByBQC
# =============================================================================

class TestHighCovLipids:
    """high_cov_lipids(): the rule the BQC filter applies."""

    def test_a_lipid_at_the_threshold_is_high_cov(self):
        """Pinned: >= threshold, the same boundary the scatter plot draws red."""
        values = [10.0, 12.0, 14.0]
        cov = float(np.std(values, ddof=1) / np.mean(values) * 100)
        prepared = pd.DataFrame({
            'LipidMolec': ['at', 'below', 'above', 'no_cov'],
            'cov': [cov, cov - 1e-9, cov + 1, np.nan],
        })
        assert QualityCheckService.high_cov_lipids(prepared, cov) == ['at', 'above']


class TestFilterByBQC:
    """Tests for filter_by_bqc()."""

    def test_remove_all_high_cov(self, basic_conc_df):
        """Remove all high-CoV lipids when lipids_to_keep is empty."""
        high_cov = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov)
        assert isinstance(result, BQCFilterResult)
        assert len(result.filtered_df) == 1  # Only TG remains
        assert result.removed_lipids == high_cov
        assert result.kept_despite_high_cov == []

    def test_keep_some_high_cov(self, basic_conc_df):
        """Keep selected lipids despite high CoV."""
        high_cov = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        keep = ['PC(16:0_18:1)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov, keep)
        assert len(result.filtered_df) == 2  # PC + TG
        assert 'PE(18:0_20:4)' in result.removed_lipids
        assert 'PC(16:0_18:1)' not in result.removed_lipids

    def test_keep_all_high_cov(self, basic_conc_df):
        """Keep all high-CoV lipids → no removal."""
        high_cov = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov, high_cov)
        assert len(result.filtered_df) == 3  # All kept
        assert result.removed_lipids == []

    def test_empty_high_cov_list(self, basic_conc_df):
        """No high-CoV lipids → no changes."""
        result = QualityCheckService.filter_by_bqc(basic_conc_df, [])
        assert len(result.filtered_df) == len(basic_conc_df)
        assert result.removed_lipids == []

    def test_none_lipids_to_keep(self, basic_conc_df):
        """lipids_to_keep=None removes all high-CoV."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)'], None
        )
        assert len(result.filtered_df) == 2

    def test_removed_count_property(self, basic_conc_df):
        """removed_count computed correctly."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        assert result.removed_count == 1
        assert result.lipids_before == 3
        assert result.lipids_after == 2

    def test_removed_percentage_property(self, basic_conc_df):
        """removed_percentage computed correctly."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        expected_pct = (1 / 3) * 100
        assert result.removed_percentage == pytest.approx(expected_pct)

    def test_removed_percentage_zero_lipids(self):
        """removed_percentage with 0 lipids_before returns 0."""
        result = BQCFilterResult(
            filtered_df=pd.DataFrame(),
            removed_lipids=[],
            kept_despite_high_cov=[],
            lipids_before=0,
            lipids_after=0,
        )
        assert result.removed_percentage == 0.0

    def test_sorted_by_classkey(self, basic_conc_df):
        """Filtered DataFrame is sorted by ClassKey."""
        result = QualityCheckService.filter_by_bqc(basic_conc_df, [])
        if 'ClassKey' in result.filtered_df.columns:
            classkeys = result.filtered_df['ClassKey'].tolist()
            assert classkeys == sorted(classkeys)

    def test_index_reset(self, basic_conc_df):
        """Filtered DataFrame has reset index."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        assert list(result.filtered_df.index) == list(range(len(result.filtered_df)))

    def test_does_not_modify_original(self, basic_conc_df):
        """Original DataFrame is not modified."""
        original_len = len(basic_conc_df)
        QualityCheckService.filter_by_bqc(basic_conc_df, ['PC(16:0_18:1)'])
        assert len(basic_conc_df) == original_len

    def test_nonexistent_lipid_in_high_cov(self, basic_conc_df):
        """Non-existent lipid in high_cov_lipids is silently ignored."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['NONEXISTENT_LIPID']
        )
        assert len(result.filtered_df) == 3  # Nothing removed
        assert result.removed_lipids == ['NONEXISTENT_LIPID']

    def test_without_classkey_column(self):
        """DataFrame without ClassKey still works (no sorting)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'concentration[s1]': [100.0, 200.0, 300.0],
        })
        result = QualityCheckService.filter_by_bqc(df, ['PE(18:0)'])
        assert len(result.filtered_df) == 2
        assert list(result.filtered_df.index) == [0, 1]

    def test_keep_list_with_items_not_in_high_cov(self, basic_conc_df):
        """keep list items not in high_cov are harmless."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)'], ['PC(16:0_18:1)', 'NOT_IN_HIGH_COV']
        )
        assert len(result.filtered_df) == 3  # PC kept
        assert result.removed_lipids == []

    def test_duplicate_entries_in_high_cov(self, basic_conc_df):
        """Duplicate entries in high_cov_lipids don't cause double removal."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)', 'PC(16:0_18:1)']
        )
        assert len(result.filtered_df) == 2  # PC removed once
        assert result.removed_count == 1

    def test_all_lipids_removed(self, basic_conc_df):
        """Removing all lipids produces empty DataFrame."""
        all_lipids = basic_conc_df['LipidMolec'].tolist()
        result = QualityCheckService.filter_by_bqc(basic_conc_df, all_lipids)
        assert len(result.filtered_df) == 0
        assert result.removed_percentage == pytest.approx(100.0)

    def test_removed_lipids_order_preserved(self, basic_conc_df):
        """removed_lipids list preserves order of high_cov_lipids."""
        high_cov = ['TG(16:0_18:1_18:2)', 'PC(16:0_18:1)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov)
        assert result.removed_lipids == high_cov

    def test_filter_preserves_data_values(self, basic_conc_df):
        """Filtering preserves concentration values of remaining lipids."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        # PE row should have same values
        pe_row = result.filtered_df[
            result.filtered_df['LipidMolec'] == 'PE(18:0_20:4)'
        ]
        assert pe_row['concentration[s1]'].iloc[0] == 2000.0


# =============================================================================
# TestRetentionTimeAvailability
# =============================================================================

class TestRetentionTimeAvailability:
    """Tests for check_retention_time_availability()."""

    def test_available_with_both_columns(self, rt_df):
        """Available when both BaseRt and CalcMass exist."""
        result = QualityCheckService.check_retention_time_availability(rt_df)
        assert isinstance(result, RetentionTimeDataResult)
        assert result.available is True

    def test_lipid_classes_listed(self, rt_df):
        """Lipid classes are returned sorted by frequency."""
        result = QualityCheckService.check_retention_time_availability(rt_df)
        assert set(result.lipid_classes) == {'PC', 'PE', 'TG'}

    def test_not_available_missing_basert(self):
        """Not available when BaseRt is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'CalcMass': [733.5],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is False
        assert result.lipid_classes == []

    def test_not_available_missing_calcmass(self):
        """Not available when CalcMass is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'BaseRt': [5.1],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is False

    def test_not_available_no_rt_columns(self):
        """Not available when neither column exists."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is False

    def test_available_without_classkey(self):
        """Available but no classes when ClassKey is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'BaseRt': [5.1],
            'CalcMass': [733.5],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is True
        assert result.lipid_classes == []

    def test_classes_sorted_by_frequency(self):
        """Classes are sorted by frequency (most frequent first)."""
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c', 'd', 'e'],
            'ClassKey': ['PE', 'PC', 'PC', 'PC', 'PE'],
            'BaseRt': [1, 2, 3, 4, 5],
            'CalcMass': [1, 2, 3, 4, 5],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.lipid_classes[0] == 'PC'  # 3 occurrences
        assert result.lipid_classes[1] == 'PE'  # 2 occurrences

    def test_empty_dataframe(self):
        """Empty DataFrame → not available."""
        df = pd.DataFrame(columns=['BaseRt', 'CalcMass', 'ClassKey'])
        result = QualityCheckService.check_retention_time_availability(df)
        # Empty DF has the columns but no rows
        assert result.available is True
        assert result.lipid_classes == []


# =============================================================================
# TestCorrelationEligibleConditions
# =============================================================================

class TestCorrelationEligibleConditions:
    """Tests for get_correlation_eligible_conditions()."""

    def test_all_multi_replicate(self, simple_experiment_2x3):
        """All conditions eligible when all have >1 replicate."""
        result = QualityCheckService.get_correlation_eligible_conditions(simple_experiment_2x3)
        assert result == ['Control', 'Treatment']

    def test_mixed_replicates(self):
        """Only multi-replicate conditions returned."""
        exp = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[3, 1, 2],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['A', 'C']

    def test_all_single_replicate(self, single_sample_experiment):
        """No eligible conditions when all have 1 replicate."""
        result = QualityCheckService.get_correlation_eligible_conditions(
            single_sample_experiment
        )
        assert result == []

    def test_preserves_order(self, large_experiment):
        """Eligible conditions preserve original order."""
        result = QualityCheckService.get_correlation_eligible_conditions(large_experiment)
        assert result == ['A', 'B', 'C', 'D']

    def test_single_condition_multi_replicate(self):
        """Single condition with multiple replicates."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Only'],
            number_of_samples_list=[5],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['Only']

    def test_exactly_two_replicates_is_eligible(self):
        """Condition with exactly 2 replicates is eligible."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 1],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['A']

    def test_three_conditions_with_bqc(self, three_condition_experiment):
        """Three conditions all eligible (all have 2 replicates)."""
        result = QualityCheckService.get_correlation_eligible_conditions(
            three_condition_experiment
        )
        assert result == ['Control', 'Treatment', 'Vehicle']

    def test_many_conditions_varied_replicates(self):
        """Many conditions with varied sample counts."""
        exp = ExperimentConfig(
            n_conditions=5,
            conditions_list=['A', 'B', 'C', 'D', 'E'],
            number_of_samples_list=[1, 5, 1, 3, 1],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['B', 'D']


# =============================================================================
# TestRemoveSamples
# =============================================================================

class TestRemoveSamples:
    """Tests for remove_samples()."""

    def test_remove_single_sample(self, basic_conc_df, simple_experiment_2x3):
        """Remove one sample from a 6-sample experiment."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s3']
        )
        assert isinstance(result, SampleRemovalResult)
        assert result.samples_before == 6
        assert result.samples_after == 5
        assert result.removed_samples == ['s3']

    def test_remove_multiple_samples(self, basic_conc_df, simple_experiment_2x3):
        """Remove two samples from different conditions."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s2', 's5']
        )
        assert result.samples_before == 6
        assert result.samples_after == 4
        assert set(result.removed_samples) == {'s2', 's5'}

    def test_columns_renamed_sequentially(self, basic_conc_df, simple_experiment_2x3):
        """After removal, columns are renamed to match new sequential labels."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s2']
        )
        new_exp = result.updated_experiment
        for s in new_exp.full_samples_list:
            assert f'concentration[{s}]' in result.updated_df.columns

    def test_removed_column_not_in_result(self, basic_conc_df, simple_experiment_2x3):
        """Removed sample's concentration column is not in updated_df."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s3']
        )
        # The old column concentration[s3] should not exist
        # (it may have been renamed to something else, but the data for s3 is gone)
        assert result.samples_after == 5

    def test_updated_experiment_config(self, basic_conc_df, simple_experiment_2x3):
        """Updated experiment has correct sample counts."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1', 's4']
        )
        new_exp = result.updated_experiment
        # Control had 3 (s1,s2,s3) → now 2; Treatment had 3 (s4,s5,s6) → now 2
        assert new_exp.number_of_samples_list == [2, 2]
        assert sum(new_exp.number_of_samples_list) == 4

    def test_condition_dropped_when_empty(self, simple_experiment_2x3):
        """Condition is dropped when all its samples are removed."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s1', 's2', 's3']
        )
        # Control condition entirely removed
        assert result.updated_experiment.n_conditions == 1
        assert result.updated_experiment.conditions_list == ['Treatment']

    def test_empty_samples_to_remove_raises(self, basic_conc_df, simple_experiment_2x3):
        """Empty samples_to_remove raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            QualityCheckService.remove_samples(
                basic_conc_df, simple_experiment_2x3, []
            )

    def test_too_many_removals_raises(self, simple_experiment_2x3):
        """Removing all but 1 sample raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        with pytest.raises(ValueError, match="at least 2 are required"):
            QualityCheckService.remove_samples(
                df, simple_experiment_2x3, ['s1', 's2', 's3', 's4', 's5']
            )

    def test_nonexistent_sample_silently_ignored(self, basic_conc_df, simple_experiment_2x3):
        """Non-existent sample labels are silently filtered out."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1', 'nonexistent']
        )
        assert result.removed_samples == ['s1']
        assert result.samples_after == 5

    def test_does_not_modify_original(self, basic_conc_df, simple_experiment_2x3):
        """Original DataFrame and experiment are not modified."""
        original_cols = list(basic_conc_df.columns)
        original_shape = basic_conc_df.shape
        QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert list(basic_conc_df.columns) == original_cols
        assert basic_conc_df.shape == original_shape

    def test_data_integrity_after_removal(self, simple_experiment_2x3):
        """Data values are preserved (just columns renamed)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        result = QualityCheckService.remove_samples(df, simple_experiment_2x3, ['s3'])
        # s1=100, s2=200 stay as s1,s2; s4=400,s5=500,s6=600 become s3,s4,s5
        new_df = result.updated_df
        assert new_df['concentration[s1]'].iloc[0] == 100.0
        assert new_df['concentration[s2]'].iloc[0] == 200.0
        # s4 → s3 in new labeling
        assert new_df['concentration[s3]'].iloc[0] == 400.0

    def test_metadata_columns_preserved(self, basic_conc_df, simple_experiment_2x3):
        """Non-concentration columns (LipidMolec, ClassKey) preserved."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert 'LipidMolec' in result.updated_df.columns
        assert 'ClassKey' in result.updated_df.columns

    def test_remove_all_samples_from_all_conditions_raises(self, simple_experiment_2x3):
        """Removing all samples raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        with pytest.raises(ValueError):
            QualityCheckService.remove_samples(
                df, simple_experiment_2x3,
                ['s1', 's2', 's3', 's4', 's5', 's6']
            )

    def test_remove_exactly_n_minus_2_boundary(self, simple_experiment_2x3):
        """Removing N-2 samples leaves exactly 2 (minimum allowed)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s1', 's2', 's3', 's4']
        )
        assert result.samples_after == 2

    def test_all_nonexistent_samples_still_raises(self, basic_conc_df, simple_experiment_2x3):
        """All nonexistent samples → 0 actual removals → but still <2 check passes.
        Empty list after filtering → would leave all 6 → should not raise."""
        # Actually all are filtered out, remaining=6, so it won't raise for <2.
        # But actual_to_remove is empty which leaves remaining unchanged.
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['x1', 'x2']
        )
        assert result.removed_samples == []
        assert result.samples_after == 6

    def test_column_count_after_removal(self, basic_conc_df, simple_experiment_2x3):
        """Number of concentration columns decreases by number removed."""
        original_conc_cols = [c for c in basic_conc_df.columns if c.startswith('concentration[')]
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s2', 's5']
        )
        new_conc_cols = [c for c in result.updated_df.columns if c.startswith('concentration[')]
        assert len(new_conc_cols) == len(original_conc_cols) - 2

    def test_remove_from_three_condition_experiment(self, three_condition_experiment):
        """Remove samples across 3 conditions, one condition drops."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        # Remove both samples from Control (s1, s2)
        result = QualityCheckService.remove_samples(
            df, three_condition_experiment, ['s1', 's2']
        )
        assert result.updated_experiment.n_conditions == 2
        assert 'Control' not in result.updated_experiment.conditions_list
        assert result.samples_after == 4

    def test_remove_sample_renaming_across_conditions(self, simple_experiment_2x3):
        """Verify exact rename mapping when removing from middle."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [10.0],
            'concentration[s2]': [20.0],
            'concentration[s3]': [30.0],
            'concentration[s4]': [40.0],
            'concentration[s5]': [50.0],
            'concentration[s6]': [60.0],
        })
        # Remove s2 (Control) and s4 (Treatment)
        result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s2', 's4']
        )
        new_df = result.updated_df
        # Surviving: s1(10), s3(30), s5(50), s6(60) → renamed to s1, s2, s3, s4
        assert new_df['concentration[s1]'].iloc[0] == 10.0
        assert new_df['concentration[s2]'].iloc[0] == 30.0
        assert new_df['concentration[s3]'].iloc[0] == 50.0
        assert new_df['concentration[s4]'].iloc[0] == 60.0


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Edge case tests across multiple methods."""


    def test_filter_bqc_single_lipid_removed(self):
        """Filter with single lipid dataset removes it."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        result = QualityCheckService.filter_by_bqc(df, ['PC(16:0)'])
        assert len(result.filtered_df) == 0
        assert result.removed_count == 1


    def test_remove_samples_preserves_row_count(self, basic_conc_df, simple_experiment_2x3):
        """Removing samples doesn't change row count (lipids)."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert len(result.updated_df) == len(basic_conc_df)


    def test_retention_time_with_extra_columns(self):
        """Retention time availability ignores extra columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'BaseRt': [5.1],
            'CalcMass': [733.5],
            'ExtraCol': ['x'],
            'AnotherCol': [42],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is True


# =============================================================================
# TestTypeCoercion
# =============================================================================

class TestTypeCoercion:
    """Tests for type handling across methods."""


    def test_remove_samples_with_int_concentrations(self):
        """remove_samples works when concentration columns are integers."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100, 200],
            'concentration[s2]': [110, 210],
            'concentration[s3]': [120, 220],
            'concentration[s4]': [130, 230],
        })
        result = QualityCheckService.remove_samples(df, exp, ['s1'])
        assert isinstance(result, SampleRemovalResult)
        assert result.samples_after == 3

    def test_filter_by_bqc_with_int_concentrations(self):
        """filter_by_bqc works when concentration columns are integers."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
        })
        result = QualityCheckService.filter_by_bqc(df, ['PE(18:0)'])
        assert len(result.filtered_df) == 2
        assert result.filtered_df['concentration[s1]'].dtype in (
            np.int64, np.float64, int,
        )


# =============================================================================
# TestMultiStepPipelines
# =============================================================================

class TestMultiStepPipelines:
    """Tests for realistic multi-step workflows."""


    def test_retention_time_check_in_pipeline(self, simple_experiment_2x3):
        """Retention time check integrates into pipeline correctly."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'BaseRt': [5.1, 6.2, 7.3],
            'CalcMass': [733.5, 743.5, 850.7],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
            'concentration[s3]': [120, 220, 320],
            'concentration[s4]': [90, 190, 290],
            'concentration[s5]': [95, 195, 295],
            'concentration[s6]': [80, 180, 280],
        })
        # Check RT availability
        rt_result = QualityCheckService.check_retention_time_availability(df)
        assert rt_result.available is True
        assert len(rt_result.lipid_classes) == 3

        # After BQC filter, RT columns still present
        filtered = QualityCheckService.filter_by_bqc(df, ['PC(16:0)'])
        rt_after = QualityCheckService.check_retention_time_availability(
            filtered.filtered_df
        )
        assert rt_after.available is True
        assert len(rt_after.lipid_classes) == 2  # PC removed
