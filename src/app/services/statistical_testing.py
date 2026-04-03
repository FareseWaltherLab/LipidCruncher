"""
Statistical testing service for lipidomic data analysis.

Provides a unified engine for two-group, multi-group, and post-hoc
statistical tests with multiple-testing correction. Used by abundance
bar chart, saturation plot, and volcano plot analyses.
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import alexandergovern, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from ..models.experiment import ExperimentConfig
from ..models.statistics import StatisticalTestConfig

MIN_POSITIVE_FALLBACK = 1.0
ZERO_REPLACEMENT_DIVISOR = 10


@dataclass
class StatisticalTestResult:
    """Result of a single statistical test (one group key)."""
    test_name: str
    statistic: float
    p_value: float
    adjusted_p_value: Optional[float] = None
    significant: bool = False
    effect_size: Optional[float] = None
    group_key: str = ""


@dataclass
class PostHocResult:
    """Result of a single pairwise post-hoc comparison."""
    group1: str
    group2: str
    p_value: float
    adjusted_p_value: Optional[float] = None
    significant: bool = False
    test_name: str = ""


@dataclass
class StatisticalTestSummary:
    """Complete results from a statistical testing run."""
    results: Dict[str, StatisticalTestResult] = field(default_factory=dict)
    posthoc_results: Dict[str, List[PostHocResult]] = field(default_factory=dict)
    test_info: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, object] = field(default_factory=dict)


class StatisticalTestingService:
    """Stateless service for running statistical tests on lipidomic data."""

    # ── Core test methods ──────────────────────────────────────────────

    @staticmethod
    def test_two_groups(
        group1_values: np.ndarray,
        group2_values: np.ndarray,
        test_type: Literal['parametric', 'non_parametric'] = 'parametric',
        auto_transform: bool = True,
        global_small_value: Optional[float] = None,
    ) -> StatisticalTestResult:
        """Run a two-group statistical test.

        Args:
            group1_values: Raw values for group 1.
            group2_values: Raw values for group 2.
            test_type: 'parametric' for Welch's t-test,
                       'non_parametric' for Mann-Whitney U.
            auto_transform: If True, apply zero replacement + log10.
            global_small_value: Dataset-wide zero-replacement threshold.
                If None, falls back to computing from these two groups.

        Returns:
            StatisticalTestResult with test name, statistic, and p-value.

        Raises:
            ValueError: If either group has fewer than 2 values or
                        all values are non-positive after filtering NaN.
        """
        small = global_small_value
        if small is None and auto_transform:
            small = StatisticalTestingService._compute_small_value_from_arrays(
                group1_values, group2_values
            )
        g1 = StatisticalTestingService._prepare_group_data(
            group1_values, auto_transform, small
        )
        g2 = StatisticalTestingService._prepare_group_data(
            group2_values, auto_transform, small
        )

        if len(g1) < 2 or len(g2) < 2:
            raise ValueError(
                "Each group must have at least 2 non-NaN values"
            )

        if test_type == 'non_parametric':
            stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
            name = "Mann-Whitney U"
        else:
            stat, p = stats.ttest_ind(g1, g2, equal_var=False)
            name = "Welch's t-test"

        return StatisticalTestResult(
            test_name=name, statistic=float(stat), p_value=float(p)
        )

    @staticmethod
    def test_multiple_groups(
        groups_dict: Dict[str, np.ndarray],
        test_type: Literal['parametric', 'non_parametric'] = 'parametric',
        auto_transform: bool = True,
        global_small_value: Optional[float] = None,
    ) -> StatisticalTestResult:
        """Run an omnibus test across 3+ groups.

        Args:
            groups_dict: Mapping of condition name → raw values array.
            test_type: 'parametric' for Welch's ANOVA (Alexander-Govern),
                       'non_parametric' for Kruskal-Wallis.
            auto_transform: If True, apply zero replacement + log10.
            global_small_value: Dataset-wide zero-replacement threshold.
                If None, falls back to computing from these groups.

        Returns:
            StatisticalTestResult with omnibus statistic and p-value.

        Raises:
            ValueError: If fewer than 2 groups or any group has < 2 values.
        """
        if len(groups_dict) < 2:
            raise ValueError("Need at least 2 groups for multi-group test")

        small = global_small_value
        if small is None and auto_transform:
            small = StatisticalTestingService._compute_small_value_from_arrays(
                *groups_dict.values()
            )
        prepared = {
            k: StatisticalTestingService._prepare_group_data(
                v, auto_transform, small
            )
            for k, v in groups_dict.items()
        }

        for k, v in prepared.items():
            if len(v) < 2:
                raise ValueError(
                    f"Group '{k}' must have at least 2 non-NaN values"
                )

        arrays = list(prepared.values())

        if test_type == 'non_parametric':
            stat, p = kruskal(*arrays)
            name = "Kruskal-Wallis"
        else:
            result = alexandergovern(*arrays)
            stat, p = result.statistic, result.pvalue
            name = "Welch's ANOVA"

        return StatisticalTestResult(
            test_name=name, statistic=float(stat), p_value=float(p)
        )

    # ── Correction ─────────────────────────────────────────────────────

    @staticmethod
    def apply_correction(
        p_values: np.ndarray,
        method: Literal['uncorrected', 'fdr_bh', 'bonferroni'] = 'fdr_bh',
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple-testing correction to a set of p-values.

        Args:
            p_values: Array of raw p-values.
            method: Correction method.
            alpha: Significance threshold.

        Returns:
            Tuple of (significant_flags, adjusted_p_values).
        """
        p_arr = np.asarray(p_values, dtype=float)

        if len(p_arr) == 0:
            return np.array([], dtype=bool), np.array([], dtype=float)

        if method == 'uncorrected':
            return p_arr <= alpha, p_arr.copy()

        reject, adj_p, _, _ = multipletests(p_arr, alpha=alpha, method=method)
        return np.asarray(reject), np.asarray(adj_p)

    # ── Post-hoc ───────────────────────────────────────────────────────

    @staticmethod
    def run_posthoc(
        groups_dict: Dict[str, np.ndarray],
        correction: Literal['tukey', 'bonferroni', 'uncorrected'] = 'tukey',
        alpha: float = 0.05,
        auto_transform: bool = True,
        global_small_value: Optional[float] = None,
    ) -> List[PostHocResult]:
        """Run pairwise post-hoc tests for 3+ groups.

        Args:
            groups_dict: Condition name → values array.
            correction: 'tukey' for Tukey's HSD,
                        'bonferroni' for Bonferroni-corrected pairwise tests,
                        'uncorrected' for raw pairwise tests.
            alpha: Significance threshold.
            auto_transform: If True, apply zero replacement + log10.
            global_small_value: Dataset-wide zero-replacement threshold.
                If None, falls back to computing from these groups.

        Returns:
            List of PostHocResult for each pairwise comparison.

        Raises:
            ValueError: If fewer than 2 groups.
        """
        if len(groups_dict) < 2:
            raise ValueError("Need at least 2 groups for post-hoc tests")

        small = global_small_value
        if small is None and auto_transform:
            small = StatisticalTestingService._compute_small_value_from_arrays(
                *groups_dict.values()
            )
        prepared = {
            k: StatisticalTestingService._prepare_group_data(
                v, auto_transform, small
            )
            for k, v in groups_dict.items()
        }

        if correction == 'tukey':
            return StatisticalTestingService._run_tukey(prepared, alpha)
        else:
            return StatisticalTestingService._run_pairwise(
                prepared, correction, alpha
            )

    @staticmethod
    def _run_tukey(
        prepared: Dict[str, np.ndarray], alpha: float
    ) -> List[PostHocResult]:
        """Run Tukey's HSD on prepared data."""
        all_data = []
        all_labels = []
        for label, vals in prepared.items():
            all_data.extend(vals)
            all_labels.extend([label] * len(vals))

        tukey = pairwise_tukeyhsd(all_data, all_labels, alpha=alpha)

        results = []
        for i in range(len(tukey.groupsunique)):
            for j in range(i + 1, len(tukey.groupsunique)):
                idx = _tukey_pair_index(
                    i, j, len(tukey.groupsunique)
                )
                g1 = str(tukey.groupsunique[i])
                g2 = str(tukey.groupsunique[j])
                p = float(tukey.pvalues[idx])
                results.append(PostHocResult(
                    group1=g1,
                    group2=g2,
                    p_value=p,
                    adjusted_p_value=p,
                    significant=p <= alpha,
                    test_name="Tukey's HSD",
                ))
        return results

    @staticmethod
    def _run_pairwise(
        prepared: Dict[str, np.ndarray],
        correction: Literal['bonferroni', 'uncorrected'],
        alpha: float,
    ) -> List[PostHocResult]:
        """Run pairwise Mann-Whitney U tests with optional Bonferroni."""
        keys = list(prepared.keys())
        pairs = list(combinations(keys, 2))
        raw_p = []

        for g1, g2 in pairs:
            _, p = mannwhitneyu(
                prepared[g1], prepared[g2], alternative='two-sided'
            )
            raw_p.append(float(p))

        raw_arr = np.array(raw_p)
        if correction == 'bonferroni':
            sig, adj = StatisticalTestingService.apply_correction(
                raw_arr, 'bonferroni', alpha
            )
        else:
            sig, adj = raw_arr <= alpha, raw_arr.copy()

        results = []
        for k, (g1, g2) in enumerate(pairs):
            results.append(PostHocResult(
                group1=g1,
                group2=g2,
                p_value=raw_p[k],
                adjusted_p_value=float(adj[k]),
                significant=bool(sig[k]),
                test_name=(
                    "Mann-Whitney U (Bonferroni)"
                    if correction == 'bonferroni'
                    else "Mann-Whitney U"
                ),
            ))
        return results

    # ── Orchestration methods ──────────────────────────────────────────

    @staticmethod
    def run_class_level_tests(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        config: StatisticalTestConfig,
    ) -> StatisticalTestSummary:
        """Run statistical tests at the lipid class level.

        For each selected class, sums species concentrations per sample,
        then compares across conditions (used by abundance bar chart).

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration[s*] cols.
            experiment: Experiment configuration.
            selected_conditions: Conditions to compare.
            selected_classes: Lipid classes to test.
            config: Statistical test configuration.

        Returns:
            StatisticalTestSummary with per-class results.
        """
        resolved = StatisticalTestingService._resolve_config(
            config, len(selected_classes), len(selected_conditions)
        )
        test_type, correction, posthoc_corr = resolved

        # Compute one dataset-wide small value for all tests
        global_small = (
            StatisticalTestingService._compute_dataset_small_value(df)
            if config.auto_transform else None
        )

        results: Dict[str, StatisticalTestResult] = {}
        posthoc_results: Dict[str, List[PostHocResult]] = {}

        for cls in selected_classes:
            groups = StatisticalTestingService._extract_class_totals(
                df, experiment, selected_conditions, cls
            )
            if groups is None:
                continue

            result = StatisticalTestingService._run_test_for_groups(
                groups, test_type, config.auto_transform, global_small
            )
            result.group_key = cls
            results[cls] = result

        # Apply Level 1 correction across classes
        StatisticalTestingService._apply_level1_correction(
            results, correction, config.alpha
        )

        # Post-hoc for multi-group if needed
        if len(selected_conditions) > 2:
            for cls in selected_classes:
                groups = StatisticalTestingService._extract_class_totals(
                    df, experiment, selected_conditions, cls
                )
                if groups is None or cls not in results:
                    continue
                if not results[cls].significant:
                    continue
                ph = StatisticalTestingService.run_posthoc(
                    groups, posthoc_corr, config.alpha, config.auto_transform,
                    global_small,
                )
                posthoc_results[cls] = ph

        return StatisticalTestSummary(
            results=results,
            posthoc_results=posthoc_results,
            test_info={
                'test_type': test_type,
                'correction': correction,
                'posthoc': posthoc_corr,
                'transform': 'log10' if config.auto_transform else 'none',
            },
            parameters={
                'n_conditions': len(selected_conditions),
                'n_classes': len(selected_classes),
                'n_tests': len(results),
                'alpha': config.alpha,
            },
        )

    @staticmethod
    def run_saturation_tests(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        fa_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        config: StatisticalTestConfig,
    ) -> StatisticalTestSummary:
        """Run statistical tests on saturation (SFA/MUFA/PUFA) data.

        Tests each (class, FA type) combination across conditions.

        Args:
            df: DataFrame with concentration columns (used to derive the
                dataset-wide zero-replacement threshold).
            experiment: Experiment configuration.
            selected_conditions: Conditions to compare.
            selected_classes: Lipid classes to test.
            fa_data: Nested dict of class → FA type → condition → values.
                     e.g. {'PC': {'SFA': {'Control': array, 'KO': array}}}
            config: Statistical test configuration.

        Returns:
            StatisticalTestSummary with per-(class, FA type) results.
        """
        fa_types = ['SFA', 'MUFA', 'PUFA']
        total_tests = len(selected_classes) * len(fa_types)
        resolved = StatisticalTestingService._resolve_config(
            config, total_tests, len(selected_conditions)
        )
        test_type, correction, posthoc_corr = resolved

        # Compute one dataset-wide small value for all tests
        global_small = (
            StatisticalTestingService._compute_dataset_small_value(df)
            if config.auto_transform else None
        )

        results: Dict[str, StatisticalTestResult] = {}
        posthoc_results: Dict[str, List[PostHocResult]] = {}
        skipped_keys = []

        for cls in selected_classes:
            if cls not in fa_data:
                continue
            for fa_type in fa_types:
                if fa_type not in fa_data[cls]:
                    continue
                key = f"{cls}_{fa_type}"
                groups = {
                    cond: fa_data[cls][fa_type][cond]
                    for cond in selected_conditions
                    if cond in fa_data[cls][fa_type]
                }
                if len(groups) < 2:
                    skipped_keys.append(key)
                    continue

                try:
                    result = StatisticalTestingService._run_test_for_groups(
                        groups, test_type, config.auto_transform, global_small
                    )
                except ValueError:
                    skipped_keys.append(key)
                    continue
                result.group_key = key
                results[key] = result

        StatisticalTestingService._apply_level1_correction(
            results, correction, config.alpha
        )

        if len(selected_conditions) > 2:
            for key, res in results.items():
                if not res.significant:
                    continue
                cls, fa_type = key.rsplit('_', 1)
                if cls not in fa_data or fa_type not in fa_data[cls]:
                    continue
                groups = {
                    cond: fa_data[cls][fa_type][cond]
                    for cond in selected_conditions
                    if cond in fa_data[cls][fa_type]
                }
                try:
                    ph = StatisticalTestingService.run_posthoc(
                        groups, posthoc_corr, config.alpha, config.auto_transform,
                        global_small,
                    )
                    posthoc_results[key] = ph
                except ValueError:
                    continue

        return StatisticalTestSummary(
            results=results,
            posthoc_results=posthoc_results,
            test_info={
                'test_type': test_type,
                'correction': correction,
                'posthoc': posthoc_corr,
                'transform': 'log10' if config.auto_transform else 'none',
            },
            parameters={
                'n_conditions': len(selected_conditions),
                'n_classes': len(selected_classes),
                'n_tests': len(results),
                'n_skipped': len(skipped_keys),
                'alpha': config.alpha,
            },
        )

    @staticmethod
    def run_species_level_tests(
        df: pd.DataFrame,
        control_samples: List[str],
        experimental_samples: List[str],
        config: StatisticalTestConfig,
    ) -> StatisticalTestSummary:
        """Run per-lipid statistical tests (used by volcano plot).

        Tests each lipid species independently across two conditions,
        then applies multiple-testing correction across all lipids.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration[s*] cols.
            control_samples: Sample names for control condition.
            experimental_samples: Sample names for experimental condition.
            config: Statistical test configuration.

        Returns:
            StatisticalTestSummary with per-lipid results including
            fold change and log2 fold change as effect_size.
        """
        resolved = StatisticalTestingService._resolve_config(
            config, len(df), 2
        )
        test_type, correction, _ = resolved

        # Compute one dataset-wide small value for all tests
        global_small = (
            StatisticalTestingService._compute_dataset_small_value(df)
            if config.auto_transform else None
        )

        results: Dict[str, StatisticalTestResult] = {}
        skipped_insufficient = []
        skipped_all_zero = []
        skipped_test_error = []

        ctrl_cols = [f'concentration[{s}]' for s in control_samples]
        exp_cols = [f'concentration[{s}]' for s in experimental_samples]

        for _, row in df.iterrows():
            lipid = row['LipidMolec']
            ctrl_vals = row[ctrl_cols].values.astype(float)
            exp_vals = row[exp_cols].values.astype(float)

            ctrl_clean = ctrl_vals[~np.isnan(ctrl_vals)]
            exp_clean = exp_vals[~np.isnan(exp_vals)]

            if len(ctrl_clean) < 2 or len(exp_clean) < 2:
                skipped_insufficient.append(lipid)
                continue

            # Check for all-zero groups
            if np.all(ctrl_clean == 0) or np.all(exp_clean == 0):
                skipped_all_zero.append(lipid)
                continue

            try:
                result = StatisticalTestingService.test_two_groups(
                    ctrl_clean, exp_clean, test_type, config.auto_transform,
                    global_small,
                )
            except ValueError:
                skipped_test_error.append(lipid)
                continue

            # Compute fold change on zero-adjusted original values
            fc = StatisticalTestingService._compute_fold_change(
                ctrl_clean, exp_clean
            )
            result.effect_size = fc
            result.group_key = lipid
            results[lipid] = result

        StatisticalTestingService._apply_level1_correction(
            results, correction, config.alpha
        )

        n_skipped = len(skipped_insufficient) + len(skipped_all_zero) + len(skipped_test_error)
        return StatisticalTestSummary(
            results=results,
            test_info={
                'test_type': test_type,
                'correction': correction,
                'transform': 'log10' if config.auto_transform else 'none',
            },
            parameters={
                'n_lipids_tested': len(results),
                'n_lipids_total': len(df),
                'n_lipids_skipped': n_skipped,
                'n_skipped_insufficient_data': len(skipped_insufficient),
                'n_skipped_all_zero': len(skipped_all_zero),
                'n_skipped_test_error': len(skipped_test_error),
                'n_control_samples': len(control_samples),
                'n_experimental_samples': len(experimental_samples),
                'alpha': config.alpha,
            },
        )

    # ── Auto mode ──────────────────────────────────────────────────────

    @staticmethod
    def apply_auto_mode(
        n_tests: int, n_conditions: int
    ) -> Dict[str, str]:
        """Determine smart defaults for auto mode.

        Args:
            n_tests: Number of tests to perform (classes, class×FA, lipids).
            n_conditions: Number of experimental conditions.

        Returns:
            Dict with 'test_type', 'correction_method', 'posthoc_correction'.
        """
        test_type = 'parametric'

        if n_tests <= 3:
            correction = 'uncorrected'
        else:
            correction = 'fdr_bh'

        if n_conditions <= 2:
            posthoc = 'uncorrected'
        else:
            posthoc = 'tukey'

        return {
            'test_type': test_type,
            'correction_method': correction,
            'posthoc_correction': posthoc,
        }

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _prepare_group_data(
        values: np.ndarray,
        auto_transform: bool,
        global_small_value: Optional[float] = None,
    ) -> np.ndarray:
        """Filter NaN, replace zeros, and optionally log10-transform.

        Args:
            values: Raw values array (may contain NaN and zeros).
            auto_transform: If True, replace zeros and apply log10.
            global_small_value: If provided, use this as the zero-replacement
                threshold instead of computing per-group. This ensures
                consistent zero handling across all groups in a comparison.

        Returns:
            Cleaned (and optionally transformed) array.
        """
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]

        if not auto_transform:
            return arr

        if global_small_value is not None:
            small_value = global_small_value
        else:
            positive = arr[arr > 0]
            min_pos = float(positive.min()) if len(positive) > 0 else MIN_POSITIVE_FALLBACK
            small_value = min_pos / ZERO_REPLACEMENT_DIVISOR

        arr = np.maximum(arr, small_value)
        return np.log10(arr)

    @staticmethod
    def _compute_dataset_small_value(df: pd.DataFrame) -> float:
        """Compute the zero-replacement value from the entire dataset.

        Finds the smallest non-zero concentration across all concentration
        columns and divides by ZERO_REPLACEMENT_DIVISOR. This provides a
        single, consistent detection-floor proxy for all statistical tests.

        Args:
            df: DataFrame with concentration[s*] columns.

        Returns:
            The small value to use for zero replacement.
        """
        conc_cols = [c for c in df.columns if c.startswith('concentration[')]
        if not conc_cols:
            return MIN_POSITIVE_FALLBACK / ZERO_REPLACEMENT_DIVISOR

        values = df[conc_cols].values.astype(float).ravel()
        values = values[~np.isnan(values)]
        positive = values[values > 0]

        if len(positive) == 0:
            return MIN_POSITIVE_FALLBACK / ZERO_REPLACEMENT_DIVISOR

        return float(positive.min()) / ZERO_REPLACEMENT_DIVISOR

    @staticmethod
    def _compute_small_value_from_arrays(*arrays: np.ndarray) -> float:
        """Compute a shared zero-replacement value from raw arrays.

        Fallback for when no DataFrame is available (e.g., direct
        test_two_groups calls without an orchestration method).

        Args:
            *arrays: One or more raw value arrays.

        Returns:
            The small value to use for zero replacement.
        """
        combined = np.concatenate([
            np.asarray(a, dtype=float)[~np.isnan(np.asarray(a, dtype=float))]
            for a in arrays
        ])
        positive = combined[combined > 0]
        min_pos = float(positive.min()) if len(positive) > 0 else MIN_POSITIVE_FALLBACK
        return min_pos / ZERO_REPLACEMENT_DIVISOR

    @staticmethod
    def _extract_class_totals(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        lipid_class: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Sum species concentrations per class per sample, grouped by condition.

        Returns:
            Dict of condition → array of total concentrations per sample,
            or None if the class is not in the DataFrame.
        """
        class_df = df[df['ClassKey'] == lipid_class]
        if class_df.empty:
            return None

        groups: Dict[str, np.ndarray] = {}
        for cond_idx, cond in enumerate(experiment.conditions_list):
            if cond not in selected_conditions:
                continue
            samples = experiment.individual_samples_list[cond_idx]
            cols = [f'concentration[{s}]' for s in samples]
            existing = [c for c in cols if c in class_df.columns]
            if not existing:
                continue
            totals = class_df[existing].sum(axis=0).values.astype(float)
            groups[cond] = totals

        return groups if len(groups) >= 2 else None

    @staticmethod
    def _run_test_for_groups(
        groups: Dict[str, np.ndarray],
        test_type: str,
        auto_transform: bool,
        global_small_value: Optional[float] = None,
    ) -> StatisticalTestResult:
        """Dispatch to two-group or multi-group test based on group count."""
        if len(groups) == 2:
            keys = list(groups.keys())
            return StatisticalTestingService.test_two_groups(
                groups[keys[0]], groups[keys[1]], test_type, auto_transform,
                global_small_value,
            )
        return StatisticalTestingService.test_multiple_groups(
            groups, test_type, auto_transform, global_small_value,
        )

    @staticmethod
    def _apply_level1_correction(
        results: Dict[str, StatisticalTestResult],
        correction: str,
        alpha: float,
    ) -> None:
        """Apply Level 1 correction across all results (mutates in place).

        NaN p-values (from failed tests) are excluded from correction
        to prevent statsmodels from propagating NaN to all adjusted values.
        """
        if not results:
            return

        keys = list(results.keys())
        p_values = np.array([results[k].p_value for k in keys])

        # Separate valid and NaN p-values to avoid NaN poisoning
        valid_mask = ~np.isnan(p_values)
        valid_keys = [k for k, m in zip(keys, valid_mask) if m]
        valid_p = p_values[valid_mask]

        if len(valid_p) > 0:
            sig, adj = StatisticalTestingService.apply_correction(
                valid_p, correction, alpha
            )
            for i, k in enumerate(valid_keys):
                results[k].adjusted_p_value = float(adj[i])
                results[k].significant = bool(sig[i])

        # Mark NaN p-value results as non-significant
        for k, m in zip(keys, valid_mask):
            if not m:
                results[k].adjusted_p_value = float('nan')
                results[k].significant = False

    @staticmethod
    def _resolve_config(
        config: StatisticalTestConfig,
        n_tests: int,
        n_conditions: int,
    ) -> Tuple[str, str, str]:
        """Resolve auto-mode config into concrete test/correction choices.

        Returns:
            Tuple of (test_type, correction_method, posthoc_correction).
        """
        if config.is_auto_mode():
            auto = StatisticalTestingService.apply_auto_mode(
                n_tests, n_conditions
            )
            return (
                auto['test_type'],
                auto['correction_method'],
                auto['posthoc_correction'],
            )
        return config.test_type, config.correction_method, config.posthoc_correction

    @staticmethod
    def _compute_fold_change(
        control: np.ndarray, experimental: np.ndarray
    ) -> float:
        """Compute log2 fold change (experimental / control).

        Uses zero-adjusted means. Returns 0.0 if control mean is zero.
        """
        positive = np.concatenate([control, experimental])
        positive = positive[positive > 0]
        min_pos = float(positive.min()) if len(positive) > 0 else MIN_POSITIVE_FALLBACK
        small = min_pos / ZERO_REPLACEMENT_DIVISOR

        ctrl_adj = np.maximum(control, small)
        exp_adj = np.maximum(experimental, small)

        mean_ctrl = np.mean(ctrl_adj)
        mean_exp = np.mean(exp_adj)

        if mean_ctrl == 0:
            return 0.0

        return float(np.log2(mean_exp / mean_ctrl))


def _tukey_pair_index(i: int, j: int, n_groups: int) -> int:
    """Map (i, j) pair indices to the linear index in Tukey results.

    Tukey's HSD results are stored as upper-triangle pairs in order:
    (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ..., (n-2, n-1).
    """
    return i * n_groups - i * (i + 1) // 2 + (j - i - 1)
