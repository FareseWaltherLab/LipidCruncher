"""Value-pinning tests for StatisticalTestingService.

The other statistics tests mostly assert thresholds (p < 0.05) on data where
a wrong formula gives the same verdict. Each test here uses data on which the
plausible wrong formula gives a different number, and pins the right one.
"""

import numpy as np
import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.services.statistical_testing import (
    StatisticalTestResult,
    StatisticalTestingService,
)

S = StatisticalTestingService


def _manual(test_type='parametric', correction='uncorrected', posthoc='tukey'):
    return StatisticalTestConfig.create_manual(
        test_type=test_type, correction_method=correction, posthoc_correction=posthoc,
    )


class TestPinnedOmnibusValues:

    # Unequal variances and unequal n: Welch 0.178, Student's t 0.017.
    G1 = np.array([10, 11, 9, 10.5, 9.5, 10.2])
    G2 = np.array([12, 20, 30])

    def test_two_groups_is_welch_not_student(self):
        result = S.test_two_groups(self.G1, self.G2, 'parametric', False)
        assert result.p_value == pytest.approx(0.177506, rel=1e-5)

    def test_two_groups_nonparametric_is_mann_whitney(self):
        result = S.test_two_groups(self.G1, self.G2, 'non_parametric', False)
        assert result.p_value == pytest.approx(0.0238095, rel=1e-5)

    # B has a much larger variance: Alexander-Govern 0.0017, classic F 0.42.
    GROUPS = {
        'A': np.array([10, 11, 9, 10.5]),
        'B': np.array([10, 30, 5, 20]),
        'C': np.array([14, 15, 13.5, 14.5]),
    }

    def test_multiple_groups_is_heteroscedastic_not_classic_anova(self):
        result = S.test_multiple_groups(self.GROUPS, 'parametric', False)
        assert result.p_value == pytest.approx(0.00172413, rel=1e-5)

    def test_multiple_groups_nonparametric_is_kruskal_wallis(self):
        result = S.test_multiple_groups(self.GROUPS, 'non_parametric', False)
        assert result.p_value == pytest.approx(0.2208906, rel=1e-5)


class TestPinnedPostHoc:

    def test_tukey_significance_follows_alpha(self):
        # A vs B has p = 0.0448: significant at alpha, not at alpha / 2.
        groups = {
            'A': np.array([10.0, 11, 9, 10.5]),
            'B': np.array([11.6, 12.6, 10.6, 12.1]),
            'C': np.array([14.0, 15, 13.5, 14.5]),
        }
        alpha = 0.05
        results = S.run_posthoc(groups, 'tukey', alpha, False)
        ab = next(r for r in results if {r.group1, r.group2} == {'A', 'B'})
        assert ab.p_value == pytest.approx(0.0448, abs=5e-4)
        assert ab.significant
        for r in results:
            assert r.significant == (r.p_value <= alpha)

    def test_posthoc_runs_only_after_a_significant_omnibus(self):
        """Docs S5.2: post-hoc only after a significant omnibus test."""
        experiment = ExperimentConfig(
            n_conditions=3, conditions_list=['A', 'B', 'C'], number_of_samples_list=[3, 3, 3],
        )
        values = [100, 110, 105, 102, 108, 104, 101, 107, 106]
        df = pd.DataFrame({'LipidMolec': ['PC 32:0'], 'ClassKey': ['PC']})
        for i, v in enumerate(values, start=1):
            df[f'concentration[s{i}]'] = [float(v)]

        summary = S.run_class_level_tests(df, experiment, ['A', 'B', 'C'], ['PC'], _manual())

        assert not summary.results['PC'].significant
        assert 'PC' not in summary.posthoc_results


class TestPinnedFoldChange:

    def test_uses_means_not_medians(self):
        # mean(ctrl) = 200 = mean(exp) -> 0.0; medians 100 vs 200 would give 1.0
        fc = S._compute_fold_change(np.array([100.0, 100, 400]), np.array([200.0, 200, 200]), 1.0)
        assert fc == pytest.approx(0.0)

    def test_zero_floor_is_the_value_passed_in(self):
        # the caller passes the dataset-wide floor (here 0.5) -> ctrl [0.5, 100]
        fc = S._compute_fold_change(np.array([0.0, 100]), np.array([50.0, 50]), 0.5)
        assert fc == pytest.approx(np.log2(50 / 50.25))

    def test_species_fold_change_floors_zeros_at_the_dataset_wide_value(self):
        """Regression: the fold change floored a zero at one-tenth of the
        lipid's own smallest positive value while the p-value on the same
        row used the dataset-wide floor (S2, supplementary methods)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 34:1', 'PE 36:2'],
            'ClassKey': ['PC', 'PE'],
            # PE's 10 is the dataset's smallest positive value -> floor 1.0
            'concentration[s1]': [0.0, 10.0], 'concentration[s2]': [100.0, 10.0],
            'concentration[s3]': [100.0, 10.0], 'concentration[s4]': [200.0, 10.0],
            'concentration[s5]': [200.0, 10.0], 'concentration[s6]': [200.0, 10.0],
        })
        summary = S.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'], _manual(),
        )
        ctrl_mean = (1.0 + 100 + 100) / 3          # not (10 + 100 + 100) / 3
        assert summary.results['PC 34:1'].effect_size == pytest.approx(np.log2(200 / ctrl_mean))


class TestPinnedSpeciesSkipCounts:

    def test_skip_reasons_are_counted(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC 32:0', 'PC 34:1', 'PC 36:2'],
            'ClassKey': ['PC', 'PC', 'PC'],
            'concentration[s1]': [100.0, 0.0, np.nan],
            'concentration[s2]': [110.0, 0.0, np.nan],
            'concentration[s3]': [105.0, 0.0, 100.0],
            'concentration[s4]': [200.0, 200.0, 200.0],
            'concentration[s5]': [210.0, 210.0, 210.0],
            'concentration[s6]': [205.0, 205.0, 205.0],
        })
        summary = S.run_species_level_tests(
            df, ['s1', 's2', 's3'], ['s4', 's5', 's6'], _manual(),
        )
        # Detected in one condition only: skipped, not tested with an extreme FC.
        assert list(summary.results) == ['PC 32:0']
        assert summary.parameters['n_skipped_all_zero'] == 1
        assert summary.parameters['n_skipped_insufficient_data'] == 1
        assert summary.parameters['n_lipids_skipped'] == 2


class TestPinnedCorrection:

    def test_nan_first_does_not_shift_adjusted_values(self):
        results = {
            'BAD': StatisticalTestResult("t", float('nan'), float('nan')),
            'PC': StatisticalTestResult("t", 3.0, 0.001),
            'PE': StatisticalTestResult("t", 2.5, 0.01),
        }
        S._apply_level1_correction(results, 'fdr_bh', 0.05)
        # BH over the two valid p-values: 0.001 -> 0.002, 0.01 -> 0.01
        assert results['PC'].adjusted_p_value == pytest.approx(0.002)
        assert results['PE'].adjusted_p_value == pytest.approx(0.01)
        assert np.isnan(results['BAD'].adjusted_p_value)


class TestDatasetFloorReachesClassTests:

    def test_class_level_uses_the_dataset_wide_zero_floor(self):
        """Class tests must replace zeros with the dataset-wide floor (here set
        by a tiny PE value), not one computed from the class totals alone."""
        experiment = ExperimentConfig(
            n_conditions=2, conditions_list=['Ctrl', 'KO'], number_of_samples_list=[3, 3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC 32:0', 'PE 34:1'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [0.0, 0.5],
            'concentration[s2]': [120.0, 0.6],
            'concentration[s3]': [100.0, 0.7],
            'concentration[s4]': [300.0, 0.5],
            'concentration[s5]': [280.0, 0.6],
            'concentration[s6]': [310.0, 0.7],
        })
        summary = S.run_class_level_tests(df, experiment, ['Ctrl', 'KO'], ['PC'], _manual())

        ctrl, ko = np.array([0.0, 120, 100]), np.array([300.0, 280, 310])
        expected = S.test_two_groups(ctrl, ko, 'parametric', True, global_small_value=0.05)
        class_only = S.test_two_groups(ctrl, ko, 'parametric', True)
        assert expected.p_value != pytest.approx(class_only.p_value)
        assert summary.results['PC'].p_value == pytest.approx(expected.p_value)
