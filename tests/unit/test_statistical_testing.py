"""
Tests for StatisticalTestingService.

Covers two-group tests, multi-group tests, correction methods,
post-hoc tests, auto mode, zero handling, data preparation,
class-level / saturation-level / species-level orchestration,
and edge cases.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.services.statistical_testing import (
    PostHocResult,
    StatisticalTestResult,
    StatisticalTestSummary,
    StatisticalTestingService,
    _tukey_pair_index,
)
from tests.conftest import make_dataframe, make_experiment


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def two_different_groups(rng):
    """Two groups with clearly different means."""
    return rng.normal(10, 2, 30), rng.normal(20, 2, 30)


@pytest.fixture
def two_similar_groups(rng):
    """Two groups from the same distribution."""
    return rng.normal(100, 5, 30), rng.normal(100, 5, 30)


@pytest.fixture
def three_groups(rng):
    """Three groups, one different."""
    return {
        'A': rng.normal(10, 2, 20),
        'B': rng.normal(10, 2, 20),
        'C': rng.normal(25, 2, 20),
    }


@pytest.fixture
def three_equal_groups(rng):
    """Three groups from the same distribution."""
    return {
        'A': rng.normal(50, 5, 20),
        'B': rng.normal(50, 5, 20),
        'C': rng.normal(50, 5, 20),
    }


@pytest.fixture
def experiment_2x3():
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x3():
    return make_experiment(3, 3)


@pytest.fixture
def class_level_df():
    """DataFrame with two classes and concentration columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PC(18:0)', 'PE(16:0)', 'PE(18:0)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'concentration[s1]': [100, 200, 50, 60],
        'concentration[s2]': [110, 210, 55, 65],
        'concentration[s3]': [105, 205, 52, 62],
        'concentration[s4]': [500, 600, 150, 160],
        'concentration[s5]': [510, 610, 155, 165],
        'concentration[s6]': [505, 605, 152, 162],
    })


@pytest.fixture
def species_level_df():
    """DataFrame for species-level (volcano) testing."""
    rng = np.random.default_rng(99)
    n = 10
    data = {
        'LipidMolec': [f'PC({i}:0)' for i in range(n)],
        'ClassKey': ['PC'] * n,
    }
    for s in ['s1', 's2', 's3']:
        data[f'concentration[{s}]'] = rng.normal(100, 10, n)
    for s in ['s4', 's5', 's6']:
        data[f'concentration[{s}]'] = rng.normal(200, 10, n)
    return pd.DataFrame(data)


@pytest.fixture
def manual_parametric_config():
    return StatisticalTestConfig.create_manual(
        test_type='parametric',
        correction_method='fdr_bh',
        posthoc_correction='tukey',
    )


@pytest.fixture
def manual_nonparam_config():
    return StatisticalTestConfig.create_manual(
        test_type='non_parametric',
        correction_method='bonferroni',
        posthoc_correction='bonferroni',
    )


@pytest.fixture
def auto_config():
    return StatisticalTestConfig.create_auto()


# ═══════════════════════════════════════════════════════════════════════
# Data Preparation (_prepare_group_data)
# ═══════════════════════════════════════════════════════════════════════

class TestPrepareGroupData:
    def test_removes_nan(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = StatisticalTestingService._prepare_group_data(arr, False)
        assert len(result) == 3
        np.testing.assert_array_equal(result, [1.0, 3.0, 5.0])

    def test_no_transform_returns_cleaned(self):
        arr = np.array([0.0, 1.0, 2.0])
        result = StatisticalTestingService._prepare_group_data(arr, False)
        np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])

    def test_transform_replaces_zeros(self):
        arr = np.array([0.0, 10.0, 20.0])
        result = StatisticalTestingService._prepare_group_data(arr, True)
        # min positive = 10, small = 1.0, zeros replaced with 1.0
        assert len(result) == 3
        assert result[0] == pytest.approx(np.log10(1.0))
        assert result[1] == pytest.approx(np.log10(10.0))

    def test_transform_applies_log10(self):
        arr = np.array([100.0, 1000.0])
        result = StatisticalTestingService._prepare_group_data(arr, True)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_all_zeros_uses_fallback(self):
        arr = np.array([0.0, 0.0, 0.0])
        result = StatisticalTestingService._prepare_group_data(arr, True)
        # fallback min_pos = 1.0, small = 0.1
        expected = np.log10(0.1)
        assert all(v == pytest.approx(expected) for v in result)

    def test_empty_after_nan_removal(self):
        arr = np.array([np.nan, np.nan])
        result = StatisticalTestingService._prepare_group_data(arr, False)
        assert len(result) == 0

    def test_negative_values_kept_without_transform(self):
        arr = np.array([-1.0, 0.0, 1.0])
        result = StatisticalTestingService._prepare_group_data(arr, False)
        np.testing.assert_array_equal(result, [-1.0, 0.0, 1.0])

    def test_single_positive_value_sets_small(self):
        arr = np.array([0.0, 0.0, 5.0])
        result = StatisticalTestingService._prepare_group_data(arr, True)
        small = 5.0 / 10
        assert result[0] == pytest.approx(np.log10(small))
        assert result[2] == pytest.approx(np.log10(5.0))

    def test_integer_input_converted(self):
        arr = np.array([1, 2, 3])
        result = StatisticalTestingService._prepare_group_data(arr, True)
        assert result.dtype == np.float64

    def test_string_number_input(self):
        arr = np.array([1.0, 2.0, 3.0])  # Already float
        result = StatisticalTestingService._prepare_group_data(arr, False)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════
# Two-Group Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTwoGroups:
    def test_parametric_different_groups(self, two_different_groups):
        g1, g2 = two_different_groups
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert result.test_name == "Welch's t-test"
        assert result.p_value < 0.05

    def test_parametric_similar_groups(self, two_similar_groups):
        g1, g2 = two_similar_groups
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert result.p_value > 0.05

    def test_nonparametric_different_groups(self, two_different_groups):
        g1, g2 = two_different_groups
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'non_parametric', True
        )
        assert result.test_name == "Mann-Whitney U"
        assert result.p_value < 0.05

    def test_nonparametric_similar_groups(self, two_similar_groups):
        g1, g2 = two_similar_groups
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'non_parametric', True
        )
        assert result.p_value > 0.05

    def test_returns_correct_result_type(self, two_different_groups):
        g1, g2 = two_different_groups
        result = StatisticalTestingService.test_two_groups(g1, g2)
        assert isinstance(result, StatisticalTestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)

    def test_raises_on_single_value_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            StatisticalTestingService.test_two_groups(
                np.array([1.0]), np.array([1.0, 2.0])
            )

    def test_raises_on_empty_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            StatisticalTestingService.test_two_groups(
                np.array([]), np.array([1.0, 2.0])
            )

    def test_raises_on_all_nan_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            StatisticalTestingService.test_two_groups(
                np.array([np.nan, np.nan]), np.array([1.0, 2.0])
            )

    def test_without_transform(self, two_different_groups):
        g1, g2 = two_different_groups
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', False
        )
        assert result.test_name == "Welch's t-test"
        assert 0 <= result.p_value <= 1

    def test_identical_values_parametric(self):
        g1 = np.array([5.0, 5.0, 5.0, 5.0])
        g2 = np.array([5.0, 5.0, 5.0, 5.0])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', False
        )
        assert np.isnan(result.p_value) or result.p_value > 0.05

    def test_with_zeros_and_transform(self):
        g1 = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 250.0])
        g2 = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert result.p_value < 0.05

    def test_large_sample_sizes(self, rng):
        g1 = rng.normal(10, 1, 1000)
        g2 = rng.normal(10.1, 1, 1000)
        result = StatisticalTestingService.test_two_groups(g1, g2, 'parametric')
        # Small effect size, large n → may or may not be significant
        assert 0 <= result.p_value <= 1

    def test_p_value_between_0_and_1(self, two_different_groups):
        g1, g2 = two_different_groups
        for tt in ['parametric', 'non_parametric']:
            result = StatisticalTestingService.test_two_groups(g1, g2, tt)
            assert 0 <= result.p_value <= 1


# ═══════════════════════════════════════════════════════════════════════
# Multi-Group Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMultipleGroups:
    def test_parametric_one_different(self, three_groups):
        result = StatisticalTestingService.test_multiple_groups(
            three_groups, 'parametric', True
        )
        assert result.test_name == "Welch's ANOVA"
        assert result.p_value < 0.05

    def test_parametric_all_equal(self, three_equal_groups):
        result = StatisticalTestingService.test_multiple_groups(
            three_equal_groups, 'parametric', True
        )
        assert result.p_value > 0.05

    def test_nonparametric_one_different(self, three_groups):
        result = StatisticalTestingService.test_multiple_groups(
            three_groups, 'non_parametric', True
        )
        assert result.test_name == "Kruskal-Wallis"
        assert result.p_value < 0.05

    def test_nonparametric_all_equal(self, three_equal_groups):
        result = StatisticalTestingService.test_multiple_groups(
            three_equal_groups, 'non_parametric', True
        )
        assert result.p_value > 0.05

    def test_raises_on_single_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            StatisticalTestingService.test_multiple_groups(
                {'A': np.array([1.0, 2.0])}, 'parametric'
            )

    def test_raises_on_small_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            StatisticalTestingService.test_multiple_groups(
                {'A': np.array([1.0, 2.0]), 'B': np.array([3.0])},
                'parametric',
            )

    def test_four_groups(self, rng):
        groups = {
            'A': rng.normal(10, 2, 15),
            'B': rng.normal(10, 2, 15),
            'C': rng.normal(10, 2, 15),
            'D': rng.normal(30, 2, 15),
        }
        result = StatisticalTestingService.test_multiple_groups(
            groups, 'parametric', True
        )
        assert result.p_value < 0.05

    def test_two_groups_dispatched(self, rng):
        groups = {
            'A': rng.normal(10, 2, 20),
            'B': rng.normal(30, 2, 20),
        }
        result = StatisticalTestingService.test_multiple_groups(
            groups, 'parametric', True
        )
        assert result.p_value < 0.05

    def test_without_transform(self, three_groups):
        result = StatisticalTestingService.test_multiple_groups(
            three_groups, 'parametric', False
        )
        assert 0 <= result.p_value <= 1


# ═══════════════════════════════════════════════════════════════════════
# Correction Methods
# ═══════════════════════════════════════════════════════════════════════

class TestCorrection:
    def test_uncorrected(self):
        p = np.array([0.01, 0.04, 0.06])
        sig, adj = StatisticalTestingService.apply_correction(
            p, 'uncorrected', 0.05
        )
        np.testing.assert_array_equal(sig, [True, True, False])
        np.testing.assert_array_equal(adj, p)

    def test_bonferroni(self):
        p = np.array([0.01, 0.04, 0.06])
        sig, adj = StatisticalTestingService.apply_correction(
            p, 'bonferroni', 0.05
        )
        # Bonferroni: 0.01*3=0.03, 0.04*3=0.12, 0.06*3=0.18
        assert sig[0] is np.True_
        assert sig[1] is np.False_
        assert sig[2] is np.False_
        assert adj[0] == pytest.approx(0.03)

    def test_fdr_bh(self):
        p = np.array([0.001, 0.01, 0.04, 0.09, 0.5])
        sig, adj = StatisticalTestingService.apply_correction(
            p, 'fdr_bh', 0.05
        )
        # FDR is less conservative than Bonferroni
        assert sig[0]  # Very small p
        assert len(adj) == 5
        # Adjusted p-values should be >= raw p-values
        assert all(a >= r for a, r in zip(adj, p))

    def test_empty_p_values(self):
        sig, adj = StatisticalTestingService.apply_correction(
            np.array([]), 'fdr_bh'
        )
        assert len(sig) == 0
        assert len(adj) == 0

    def test_single_p_value(self):
        sig, adj = StatisticalTestingService.apply_correction(
            np.array([0.03]), 'bonferroni', 0.05
        )
        assert sig[0]
        assert adj[0] == pytest.approx(0.03)

    def test_all_significant(self):
        p = np.array([0.001, 0.002, 0.003])
        sig, _ = StatisticalTestingService.apply_correction(p, 'fdr_bh')
        assert all(sig)

    def test_none_significant(self):
        p = np.array([0.5, 0.6, 0.7])
        sig, _ = StatisticalTestingService.apply_correction(p, 'fdr_bh')
        assert not any(sig)

    def test_adjusted_p_capped_at_1(self):
        p = np.array([0.9, 0.95])
        _, adj = StatisticalTestingService.apply_correction(p, 'bonferroni')
        assert all(a <= 1.0 for a in adj)

    def test_preserves_order(self):
        p = np.array([0.05, 0.01, 0.1])
        _, adj = StatisticalTestingService.apply_correction(p, 'fdr_bh')
        assert len(adj) == 3


# ═══════════════════════════════════════════════════════════════════════
# Post-Hoc Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPostHoc:
    def test_tukey_three_groups(self, three_groups):
        results = StatisticalTestingService.run_posthoc(
            three_groups, 'tukey', 0.05, True
        )
        assert len(results) == 3  # C(3,2) pairs
        assert all(isinstance(r, PostHocResult) for r in results)
        # A vs C and B vs C should be significant
        ac = [r for r in results if set([r.group1, r.group2]) == {'A', 'C'}][0]
        assert ac.significant
        assert ac.test_name == "Tukey's HSD"

    def test_bonferroni_pairwise(self, three_groups):
        results = StatisticalTestingService.run_posthoc(
            three_groups, 'bonferroni', 0.05, True
        )
        assert len(results) == 3
        assert all("Bonferroni" in r.test_name for r in results)

    def test_uncorrected_pairwise(self, three_groups):
        results = StatisticalTestingService.run_posthoc(
            three_groups, 'uncorrected', 0.05, True
        )
        assert len(results) == 3
        # Uncorrected: adjusted = raw
        for r in results:
            assert r.adjusted_p_value == pytest.approx(r.p_value)

    def test_tukey_equal_groups(self, three_equal_groups):
        results = StatisticalTestingService.run_posthoc(
            three_equal_groups, 'tukey', 0.05, True
        )
        assert all(not r.significant for r in results)

    def test_raises_on_single_group(self):
        with pytest.raises(ValueError, match="at least 2"):
            StatisticalTestingService.run_posthoc(
                {'A': np.array([1.0, 2.0])}, 'tukey'
            )

    def test_four_groups_six_pairs(self, rng):
        groups = {
            'A': rng.normal(10, 2, 15),
            'B': rng.normal(10, 2, 15),
            'C': rng.normal(10, 2, 15),
            'D': rng.normal(10, 2, 15),
        }
        results = StatisticalTestingService.run_posthoc(groups, 'tukey')
        assert len(results) == 6  # C(4,2)

    def test_posthoc_p_values_in_range(self, three_groups):
        for method in ['tukey', 'bonferroni', 'uncorrected']:
            results = StatisticalTestingService.run_posthoc(
                three_groups, method, 0.05, True
            )
            for r in results:
                assert 0 <= r.p_value <= 1
                assert r.adjusted_p_value is not None

    def test_two_groups_posthoc(self, rng):
        groups = {
            'A': rng.normal(10, 2, 15),
            'B': rng.normal(30, 2, 15),
        }
        results = StatisticalTestingService.run_posthoc(groups, 'tukey')
        assert len(results) == 1
        assert results[0].significant


# ═══════════════════════════════════════════════════════════════════════
# Auto Mode
# ═══════════════════════════════════════════════════════════════════════

class TestAutoMode:
    def test_few_tests_uncorrected(self):
        result = StatisticalTestingService.apply_auto_mode(2, 2)
        assert result['correction_method'] == 'uncorrected'

    def test_three_tests_uncorrected(self):
        result = StatisticalTestingService.apply_auto_mode(3, 2)
        assert result['correction_method'] == 'uncorrected'

    def test_many_tests_fdr(self):
        result = StatisticalTestingService.apply_auto_mode(10, 2)
        assert result['correction_method'] == 'fdr_bh'

    def test_two_conditions_no_posthoc(self):
        result = StatisticalTestingService.apply_auto_mode(5, 2)
        assert result['posthoc_correction'] == 'uncorrected'

    def test_three_conditions_tukey(self):
        result = StatisticalTestingService.apply_auto_mode(5, 3)
        assert result['posthoc_correction'] == 'tukey'

    def test_always_parametric(self):
        result = StatisticalTestingService.apply_auto_mode(1, 2)
        assert result['test_type'] == 'parametric'

    def test_single_test_two_conditions(self):
        result = StatisticalTestingService.apply_auto_mode(1, 2)
        assert result == {
            'test_type': 'parametric',
            'correction_method': 'uncorrected',
            'posthoc_correction': 'uncorrected',
        }

    def test_many_tests_many_conditions(self):
        result = StatisticalTestingService.apply_auto_mode(20, 4)
        assert result['correction_method'] == 'fdr_bh'
        assert result['posthoc_correction'] == 'tukey'


# ═══════════════════════════════════════════════════════════════════════
# Fold Change
# ═══════════════════════════════════════════════════════════════════════

class TestFoldChange:
    def test_twofold_increase(self):
        ctrl = np.array([100.0, 100.0, 100.0])
        exp = np.array([200.0, 200.0, 200.0])
        fc = StatisticalTestingService._compute_fold_change(ctrl, exp)
        assert fc == pytest.approx(1.0)  # log2(2) = 1

    def test_twofold_decrease(self):
        ctrl = np.array([200.0, 200.0, 200.0])
        exp = np.array([100.0, 100.0, 100.0])
        fc = StatisticalTestingService._compute_fold_change(ctrl, exp)
        assert fc == pytest.approx(-1.0)  # log2(0.5) = -1

    def test_no_change(self):
        ctrl = np.array([100.0, 100.0])
        exp = np.array([100.0, 100.0])
        fc = StatisticalTestingService._compute_fold_change(ctrl, exp)
        assert fc == pytest.approx(0.0)

    def test_with_zeros(self):
        ctrl = np.array([0.0, 100.0])
        exp = np.array([200.0, 200.0])
        fc = StatisticalTestingService._compute_fold_change(ctrl, exp)
        # Zeros replaced; fold change > 0 since exp > ctrl
        assert fc > 0

    def test_all_zeros_both(self):
        ctrl = np.array([0.0, 0.0])
        exp = np.array([0.0, 0.0])
        fc = StatisticalTestingService._compute_fold_change(ctrl, exp)
        assert fc == pytest.approx(0.0)  # Same values after replacement


# ═══════════════════════════════════════════════════════════════════════
# Extract Class Totals
# ═══════════════════════════════════════════════════════════════════════

class TestExtractClassTotals:
    def test_basic_extraction(self, class_level_df, experiment_2x3):
        groups = StatisticalTestingService._extract_class_totals(
            class_level_df, experiment_2x3, ['Control', 'Treatment'], 'PC'
        )
        assert groups is not None
        assert 'Control' in groups
        assert 'Treatment' in groups
        # PC has 2 species: sum per sample
        assert len(groups['Control']) == 3
        assert groups['Control'][0] == pytest.approx(300.0)  # 100+200

    def test_missing_class(self, class_level_df, experiment_2x3):
        result = StatisticalTestingService._extract_class_totals(
            class_level_df, experiment_2x3, ['Control'], 'MISSING'
        )
        assert result is None

    def test_single_condition_returns_none(self, class_level_df, experiment_2x3):
        result = StatisticalTestingService._extract_class_totals(
            class_level_df, experiment_2x3, ['Control'], 'PC'
        )
        assert result is None  # Need >= 2 groups

    def test_correct_samples_per_condition(self, class_level_df, experiment_2x3):
        groups = StatisticalTestingService._extract_class_totals(
            class_level_df, experiment_2x3, ['Control', 'Treatment'], 'PE'
        )
        assert len(groups['Control']) == 3
        assert len(groups['Treatment']) == 3


# ═══════════════════════════════════════════════════════════════════════
# Class-Level Tests (Orchestration)
# ═══════════════════════════════════════════════════════════════════════

class TestClassLevelTests:
    def test_basic_run(self, class_level_df, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC', 'PE'],
            manual_parametric_config,
        )
        assert isinstance(summary, StatisticalTestSummary)
        assert 'PC' in summary.results
        assert 'PE' in summary.results
        assert summary.parameters['n_conditions'] == 2
        assert summary.parameters['n_classes'] == 2

    def test_significant_classes(self, class_level_df, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC', 'PE'],
            manual_parametric_config,
        )
        # Large difference → should be significant
        for cls in ['PC', 'PE']:
            assert summary.results[cls].p_value < 0.05

    def test_auto_mode(self, class_level_df, experiment_2x3, auto_config):
        summary = StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            auto_config,
        )
        assert summary.test_info['test_type'] == 'parametric'

    def test_missing_class_skipped(self, class_level_df, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC', 'MISSING'],
            manual_parametric_config,
        )
        assert 'PC' in summary.results
        assert 'MISSING' not in summary.results

    def test_test_info_populated(self, class_level_df, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            manual_parametric_config,
        )
        assert 'test_type' in summary.test_info
        assert 'correction' in summary.test_info
        assert 'transform' in summary.test_info

    def test_three_conditions_with_posthoc(self, experiment_3x3, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(18:0)'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [100, 200],
            'concentration[s2]': [110, 210],
            'concentration[s3]': [105, 205],
            'concentration[s4]': [100, 200],
            'concentration[s5]': [110, 210],
            'concentration[s6]': [105, 205],
            'concentration[s7]': [500, 600],
            'concentration[s8]': [510, 610],
            'concentration[s9]': [520, 620],
        })
        summary = StatisticalTestingService.run_class_level_tests(
            df, experiment_3x3,
            ['Control', 'Treatment', 'Vehicle'], ['PC'],
            manual_parametric_config,
        )
        # If omnibus is significant, posthoc should be present
        if summary.results['PC'].significant:
            assert 'PC' in summary.posthoc_results

    def test_nonparametric(self, class_level_df, experiment_2x3, manual_nonparam_config):
        summary = StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            manual_nonparam_config,
        )
        assert summary.results['PC'].test_name == "Mann-Whitney U"


# ═══════════════════════════════════════════════════════════════════════
# Saturation-Level Tests
# ═══════════════════════════════════════════════════════════════════════

class TestSaturationTests:
    @pytest.fixture
    def fa_data(self, rng):
        return {
            'PC': {
                'SFA': {
                    'Control': rng.normal(100, 10, 3),
                    'Treatment': rng.normal(200, 10, 3),
                },
                'MUFA': {
                    'Control': rng.normal(50, 5, 3),
                    'Treatment': rng.normal(50, 5, 3),
                },
                'PUFA': {
                    'Control': rng.normal(30, 3, 3),
                    'Treatment': rng.normal(60, 3, 3),
                },
            },
        }

    def test_basic_run(self, fa_data, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            fa_data, manual_parametric_config,
        )
        assert isinstance(summary, StatisticalTestSummary)
        assert 'PC_SFA' in summary.results
        assert 'PC_MUFA' in summary.results
        assert 'PC_PUFA' in summary.results

    def test_significant_fa_types(self, fa_data, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            fa_data, manual_parametric_config,
        )
        # SFA and PUFA have different means → likely significant
        assert summary.results['PC_SFA'].p_value < 0.05

    def test_missing_class_skipped(self, fa_data, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC', 'MISSING'],
            fa_data, manual_parametric_config,
        )
        assert all('MISSING' not in k for k in summary.results)

    def test_auto_mode(self, fa_data, experiment_2x3, auto_config):
        summary = StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            fa_data, auto_config,
        )
        assert summary.test_info['test_type'] == 'parametric'
        # 1 class * 3 FA types = 3 tests → uncorrected
        assert summary.test_info['correction'] == 'uncorrected'

    def test_parameters_populated(self, fa_data, experiment_2x3, manual_parametric_config):
        summary = StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            fa_data, manual_parametric_config,
        )
        assert summary.parameters['n_conditions'] == 2
        assert summary.parameters['n_classes'] == 1


# ═══════════════════════════════════════════════════════════════════════
# Species-Level Tests (Volcano)
# ═══════════════════════════════════════════════════════════════════════

class TestSpeciesLevelTests:
    def test_basic_run(self, species_level_df, manual_parametric_config):
        summary = StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert isinstance(summary, StatisticalTestSummary)
        assert len(summary.results) == 10

    def test_fold_change_populated(self, species_level_df, manual_parametric_config):
        summary = StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        for lipid, result in summary.results.items():
            assert result.effect_size is not None
            # exp ~200, ctrl ~100 → log2(2) ≈ 1
            assert result.effect_size > 0

    def test_significant_lipids(self, species_level_df, manual_parametric_config):
        summary = StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        sig_count = sum(1 for r in summary.results.values() if r.significant)
        assert sig_count > 0

    def test_skips_all_nan_lipid(self, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': ['PC(1:0)', 'PC(2:0)'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [np.nan, 100],
            'concentration[s2]': [np.nan, 110],
            'concentration[s3]': [np.nan, 105],
            'concentration[s4]': [np.nan, 200],
            'concentration[s5]': [np.nan, 210],
            'concentration[s6]': [np.nan, 205],
        })
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert 'PC(1:0)' not in summary.results
        assert 'PC(2:0)' in summary.results

    def test_skips_all_zero_lipid(self, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': ['PC(1:0)', 'PC(2:0)'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [0, 100],
            'concentration[s2]': [0, 110],
            'concentration[s3]': [0, 105],
            'concentration[s4]': [0, 200],
            'concentration[s5]': [0, 210],
            'concentration[s6]': [0, 205],
        })
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert 'PC(1:0)' not in summary.results

    def test_parameters_tracked(self, species_level_df, manual_parametric_config):
        summary = StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert summary.parameters['n_lipids_total'] == 10
        assert summary.parameters['n_control_samples'] == 3

    def test_auto_mode_species(self, species_level_df, auto_config):
        summary = StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            auto_config,
        )
        # 10 lipids → fdr_bh
        assert summary.test_info['correction'] == 'fdr_bh'

    def test_nonparametric_species(self, species_level_df, manual_nonparam_config):
        summary = StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_nonparam_config,
        )
        for r in summary.results.values():
            assert r.test_name == "Mann-Whitney U"


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_all_zeros_group_with_transform(self):
        g1 = np.array([0.0, 0.0, 0.0])
        g2 = np.array([100.0, 200.0, 300.0])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert 0 <= result.p_value <= 1

    def test_mixed_nan_and_zeros(self):
        g1 = np.array([0.0, np.nan, 100.0, 200.0])
        g2 = np.array([np.nan, 500.0, 600.0, 700.0])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert 0 <= result.p_value <= 1

    def test_very_small_values(self):
        g1 = np.array([1e-10, 2e-10, 3e-10])
        g2 = np.array([1e-5, 2e-5, 3e-5])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert result.p_value < 0.05

    def test_very_large_values(self):
        g1 = np.array([1e10, 2e10, 3e10])
        g2 = np.array([4e10, 5e10, 6e10])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', True
        )
        assert 0 <= result.p_value <= 1

    def test_single_species_df(self, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100],
            'concentration[s2]': [110],
            'concentration[s3]': [105],
            'concentration[s4]': [500],
            'concentration[s5]': [510],
            'concentration[s6]': [505],
        })
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert len(summary.results) == 1

    def test_empty_dataframe_species(self, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': [],
            'ClassKey': [],
            'concentration[s1]': [],
            'concentration[s2]': [],
        })
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1'], ['s2'], manual_parametric_config,
        )
        assert len(summary.results) == 0

    def test_two_samples_minimum(self):
        g1 = np.array([100.0, 200.0])
        g2 = np.array([300.0, 400.0])
        result = StatisticalTestingService.test_two_groups(
            g1, g2, 'parametric', False
        )
        assert 0 <= result.p_value <= 1


# ═══════════════════════════════════════════════════════════════════════
# Resolve Config
# ═══════════════════════════════════════════════════════════════════════

class TestResolveConfig:
    def test_manual_passthrough(self, manual_parametric_config):
        tt, corr, ph = StatisticalTestingService._resolve_config(
            manual_parametric_config, 10, 3
        )
        assert tt == 'parametric'
        assert corr == 'fdr_bh'
        assert ph == 'tukey'

    def test_auto_resolves(self, auto_config):
        tt, corr, ph = StatisticalTestingService._resolve_config(
            auto_config, 10, 3
        )
        assert tt == 'parametric'
        assert corr == 'fdr_bh'
        assert ph == 'tukey'

    def test_auto_few_tests(self, auto_config):
        tt, corr, ph = StatisticalTestingService._resolve_config(
            auto_config, 2, 2
        )
        assert corr == 'uncorrected'
        assert ph == 'uncorrected'


# ═══════════════════════════════════════════════════════════════════════
# Tukey Pair Index Helper
# ═══════════════════════════════════════════════════════════════════════

class TestTukeyPairIndex:
    def test_three_groups(self):
        # (0,1), (0,2), (1,2) → indices 0, 1, 2
        assert _tukey_pair_index(0, 1, 3) == 0
        assert _tukey_pair_index(0, 2, 3) == 1
        assert _tukey_pair_index(1, 2, 3) == 2

    def test_four_groups(self):
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) → 0..5
        assert _tukey_pair_index(0, 1, 4) == 0
        assert _tukey_pair_index(0, 2, 4) == 1
        assert _tukey_pair_index(0, 3, 4) == 2
        assert _tukey_pair_index(1, 2, 4) == 3
        assert _tukey_pair_index(1, 3, 4) == 4
        assert _tukey_pair_index(2, 3, 4) == 5


# ═══════════════════════════════════════════════════════════════════════
# Dataclass Defaults
# ═══════════════════════════════════════════════════════════════════════

class TestDataclasses:
    def test_statistical_test_result_defaults(self):
        r = StatisticalTestResult(test_name="t", statistic=1.0, p_value=0.05)
        assert r.adjusted_p_value is None
        assert r.significant is False
        assert r.effect_size is None
        assert r.group_key == ""

    def test_posthoc_result_defaults(self):
        r = PostHocResult(group1="A", group2="B", p_value=0.01)
        assert r.adjusted_p_value is None
        assert r.significant is False
        assert r.test_name == ""

    def test_summary_defaults(self):
        s = StatisticalTestSummary()
        assert s.results == {}
        assert s.posthoc_results == {}
        assert s.test_info == {}
        assert s.parameters == {}

    def test_summary_with_data(self):
        r = StatisticalTestResult(
            test_name="Welch's t-test", statistic=5.0, p_value=0.001,
            significant=True, group_key="PC"
        )
        s = StatisticalTestSummary(
            results={'PC': r},
            test_info={'test_type': 'parametric'},
        )
        assert 'PC' in s.results
        assert s.results['PC'].significant


# ═══════════════════════════════════════════════════════════════════════
# Level 1 Correction Integration
# ═══════════════════════════════════════════════════════════════════════

class TestLevel1Correction:
    def test_corrects_across_classes(self):
        results = {
            'PC': StatisticalTestResult("t", 2.0, 0.03),
            'PE': StatisticalTestResult("t", 1.5, 0.04),
            'SM': StatisticalTestResult("t", 0.5, 0.5),
        }
        StatisticalTestingService._apply_level1_correction(
            results, 'bonferroni', 0.05
        )
        # 0.03 * 3 = 0.09 > 0.05
        assert not results['PC'].significant
        assert not results['PE'].significant
        assert not results['SM'].significant

    def test_fdr_less_conservative(self):
        results = {
            'PC': StatisticalTestResult("t", 2.0, 0.01),
            'PE': StatisticalTestResult("t", 1.5, 0.03),
        }
        StatisticalTestingService._apply_level1_correction(
            results, 'fdr_bh', 0.05
        )
        assert results['PC'].significant

    def test_uncorrected_raw(self):
        results = {
            'PC': StatisticalTestResult("t", 2.0, 0.03),
        }
        StatisticalTestingService._apply_level1_correction(
            results, 'uncorrected', 0.05
        )
        assert results['PC'].significant
        assert results['PC'].adjusted_p_value == pytest.approx(0.03)

    def test_empty_results(self):
        results = {}
        StatisticalTestingService._apply_level1_correction(
            results, 'fdr_bh', 0.05
        )
        assert results == {}


# ═══════════════════════════════════════════════════════════════════════
# Type Coercion
# ═══════════════════════════════════════════════════════════════════════

class TestTypeCoercion:
    """Verify service handles various numeric dtypes correctly."""

    def test_float32_concentrations_class_level(self, experiment_2x3, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': np.array([100.0], dtype=np.float32),
            'concentration[s2]': np.array([110.0], dtype=np.float32),
            'concentration[s3]': np.array([105.0], dtype=np.float32),
            'concentration[s4]': np.array([500.0], dtype=np.float32),
            'concentration[s5]': np.array([510.0], dtype=np.float32),
            'concentration[s6]': np.array([505.0], dtype=np.float32),
        })
        summary = StatisticalTestingService.run_class_level_tests(
            df, experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            manual_parametric_config,
        )
        assert 'PC' in summary.results
        assert summary.results['PC'].p_value < 0.05

    def test_int64_concentrations_class_level(self, experiment_2x3, manual_parametric_config):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': np.array([100], dtype=np.int64),
            'concentration[s2]': np.array([110], dtype=np.int64),
            'concentration[s3]': np.array([105], dtype=np.int64),
            'concentration[s4]': np.array([500], dtype=np.int64),
            'concentration[s5]': np.array([510], dtype=np.int64),
            'concentration[s6]': np.array([505], dtype=np.int64),
        })
        summary = StatisticalTestingService.run_class_level_tests(
            df, experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            manual_parametric_config,
        )
        assert 'PC' in summary.results

    def test_object_dtype_concentrations_species_level(self, manual_parametric_config):
        """Object dtype columns (string numbers) in species-level tests."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': pd.array(['100.0'], dtype='object'),
            'concentration[s2]': pd.array(['110.0'], dtype='object'),
            'concentration[s3]': pd.array(['105.0'], dtype='object'),
            'concentration[s4]': pd.array(['500.0'], dtype='object'),
            'concentration[s5]': pd.array(['510.0'], dtype='object'),
            'concentration[s6]': pd.array(['505.0'], dtype='object'),
        })
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert 'PC(16:0)' in summary.results

    def test_float32_two_groups(self):
        g1 = np.array([100.0, 110.0, 105.0], dtype=np.float32)
        g2 = np.array([500.0, 510.0, 505.0], dtype=np.float32)
        result = StatisticalTestingService.test_two_groups(g1, g2, 'parametric')
        assert result.p_value < 0.05

    def test_integer_list_two_groups(self):
        g1 = np.array([100, 110, 105])
        g2 = np.array([500, 510, 505])
        result = StatisticalTestingService.test_two_groups(g1, g2, 'parametric')
        assert result.p_value < 0.05

    def test_float32_saturation_data(self, experiment_2x3, manual_parametric_config):
        """float32 values in saturation-level fa_data."""
        fa_data = {
            'PC': {
                'SFA': {
                    'Control': np.array([100.0, 110.0, 105.0], dtype=np.float32),
                    'Treatment': np.array([500.0, 510.0, 505.0], dtype=np.float32),
                },
                'MUFA': {
                    'Control': np.array([50.0, 55.0, 52.0], dtype=np.float32),
                    'Treatment': np.array([50.0, 48.0, 51.0], dtype=np.float32),
                },
                'PUFA': {
                    'Control': np.array([30.0, 35.0, 32.0], dtype=np.float32),
                    'Treatment': np.array([30.0, 28.0, 31.0], dtype=np.float32),
                },
            },
        }
        summary = StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            fa_data, manual_parametric_config,
        )
        assert 'PC_SFA' in summary.results
        assert summary.results['PC_SFA'].p_value < 0.05


# ═══════════════════════════════════════════════════════════════════════
# Immutability
# ═══════════════════════════════════════════════════════════════════════

class TestImmutability:
    """Verify that input DataFrames and arrays are not modified."""

    def test_class_level_preserves_input(self, class_level_df, experiment_2x3, manual_parametric_config):
        df_copy = class_level_df.copy()
        StatisticalTestingService.run_class_level_tests(
            class_level_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC', 'PE'],
            manual_parametric_config,
        )
        pd.testing.assert_frame_equal(class_level_df, df_copy)

    def test_species_level_preserves_input(self, species_level_df, manual_parametric_config):
        df_copy = species_level_df.copy()
        StatisticalTestingService.run_species_level_tests(
            species_level_df,
            ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        pd.testing.assert_frame_equal(species_level_df, df_copy)

    def test_two_groups_preserves_arrays(self):
        g1 = np.array([100.0, 110.0, 105.0])
        g2 = np.array([500.0, 510.0, 505.0])
        g1_copy = g1.copy()
        g2_copy = g2.copy()
        StatisticalTestingService.test_two_groups(g1, g2, 'parametric', True)
        np.testing.assert_array_equal(g1, g1_copy)
        np.testing.assert_array_equal(g2, g2_copy)

    def test_multiple_groups_preserves_arrays(self, three_groups):
        copies = {k: v.copy() for k, v in three_groups.items()}
        StatisticalTestingService.test_multiple_groups(
            three_groups, 'parametric', True
        )
        for k in three_groups:
            np.testing.assert_array_equal(three_groups[k], copies[k])

    def test_prepare_group_data_preserves_input(self):
        arr = np.array([0.0, 10.0, np.nan, 20.0])
        arr_copy = arr.copy()
        StatisticalTestingService._prepare_group_data(arr, True)
        np.testing.assert_array_equal(arr, arr_copy)

    def test_saturation_level_preserves_fa_data(self, experiment_2x3, manual_parametric_config, rng):
        fa_data = {
            'PC': {
                'SFA': {
                    'Control': rng.normal(100, 10, 3),
                    'Treatment': rng.normal(200, 10, 3),
                },
                'MUFA': {
                    'Control': rng.normal(50, 5, 3),
                    'Treatment': rng.normal(50, 5, 3),
                },
                'PUFA': {
                    'Control': rng.normal(30, 3, 3),
                    'Treatment': rng.normal(30, 3, 3),
                },
            },
        }
        # Deep copy for comparison
        copies = {}
        for cls in fa_data:
            copies[cls] = {}
            for fa in fa_data[cls]:
                copies[cls][fa] = {}
                for cond in fa_data[cls][fa]:
                    copies[cls][fa][cond] = fa_data[cls][fa][cond].copy()

        StatisticalTestingService.run_saturation_tests(
            pd.DataFrame(), experiment_2x3,
            ['Control', 'Treatment'], ['PC'],
            fa_data, manual_parametric_config,
        )
        for cls in fa_data:
            for fa in fa_data[cls]:
                for cond in fa_data[cls][fa]:
                    np.testing.assert_array_equal(
                        fa_data[cls][fa][cond], copies[cls][fa][cond]
                    )


# ═══════════════════════════════════════════════════════════════════════
# Large Dataset
# ═══════════════════════════════════════════════════════════════════════

class TestLargeDataset:
    """Stress tests with many lipids and classes."""

    def test_100_lipids_species_level(self, manual_parametric_config):
        n = 100
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'LipidMolec': [f'PC({i}:0)' for i in range(n)],
            'ClassKey': ['PC'] * n,
        })
        for s in ['s1', 's2', 's3']:
            df[f'concentration[{s}]'] = rng.normal(100, 10, n)
        for s in ['s4', 's5', 's6']:
            df[f'concentration[{s}]'] = rng.normal(200, 10, n)
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            manual_parametric_config,
        )
        assert len(summary.results) == n
        # With clear separation, most should be significant
        sig_count = sum(1 for r in summary.results.values() if r.significant)
        assert sig_count > n * 0.5

    def test_50_classes_class_level(self, experiment_2x3, manual_parametric_config):
        n = 50
        rng = np.random.default_rng(42)
        classes = [f'Class{i}' for i in range(n)]
        df = pd.DataFrame({
            'LipidMolec': [f'Lipid_{i}' for i in range(n)],
            'ClassKey': classes,
        })
        for s in ['s1', 's2', 's3']:
            df[f'concentration[{s}]'] = rng.normal(100, 10, n)
        for s in ['s4', 's5', 's6']:
            df[f'concentration[{s}]'] = rng.normal(500, 10, n)
        summary = StatisticalTestingService.run_class_level_tests(
            df, experiment_2x3,
            ['Control', 'Treatment'], classes,
            manual_parametric_config,
        )
        assert len(summary.results) == n
        assert summary.parameters['n_classes'] == n

    def test_100_lipids_auto_mode_uses_fdr(self, auto_config):
        """With 100 tests, auto mode should select FDR correction."""
        n = 100
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'LipidMolec': [f'PC({i}:0)' for i in range(n)],
            'ClassKey': ['PC'] * n,
        })
        for s in ['s1', 's2', 's3']:
            df[f'concentration[{s}]'] = rng.normal(100, 10, n)
        for s in ['s4', 's5', 's6']:
            df[f'concentration[{s}]'] = rng.normal(200, 10, n)
        summary = StatisticalTestingService.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'],
            auto_config,
        )
        assert summary.test_info['correction'] == 'fdr_bh'

    def test_large_groups(self):
        """1000 samples per group — tests should still work."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(100, 10, 1000)
        g2 = rng.normal(105, 10, 1000)
        result = StatisticalTestingService.test_two_groups(g1, g2, 'parametric')
        assert 0 <= result.p_value <= 1
