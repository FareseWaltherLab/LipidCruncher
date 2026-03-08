"""
Unit tests for SampleGroupingService.

Tests cover: dataset validation, sample name extraction, group_df building,
regrouping with column reordering, and name mapping.
"""

import pytest
import pandas as pd
import numpy as np

from app.models.experiment import ExperimentConfig
from app.services.sample_grouping import (
    DatasetValidationResult,
    GroupingResult,
    RegroupingResult,
    SampleGroupingService,
)


# =============================================================================
# Helpers
# =============================================================================


def make_experiment(n_conditions=2, samples_per=3):
    """Build an ExperimentConfig with equal samples per condition."""
    return ExperimentConfig(
        n_conditions=n_conditions,
        conditions_list=[f'Cond_{i+1}' for i in range(n_conditions)],
        number_of_samples_list=[samples_per] * n_conditions,
    )


def make_df(n_samples=6, n_lipids=5, with_classkey=True):
    """Build a minimal standardized DataFrame."""
    data = {'LipidMolec': [f'PC({i}:0)' for i in range(n_lipids)]}
    if with_classkey:
        data['ClassKey'] = ['PC'] * n_lipids
    for i in range(1, n_samples + 1):
        data[f'intensity[s{i}]'] = np.random.rand(n_lipids) * 1000
    return pd.DataFrame(data)


# =============================================================================
# TestDatasetValidationResult
# =============================================================================


class TestDatasetValidationResult:
    def test_default_values(self):
        r = DatasetValidationResult(valid=True, message="ok")
        assert r.valid is True
        assert r.n_intensity_cols == 0
        assert r.expected_samples == 0

    def test_with_all_fields(self):
        r = DatasetValidationResult(
            valid=False, message="err", n_intensity_cols=5, expected_samples=6,
        )
        assert r.valid is False
        assert r.n_intensity_cols == 5
        assert r.expected_samples == 6


# =============================================================================
# TestValidateDataset
# =============================================================================


class TestValidateDataset:
    def test_valid_dataset(self):
        exp = make_experiment(2, 3)
        df = make_df(6)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True
        assert result.n_intensity_cols == 6
        assert result.expected_samples == 6

    def test_none_dataframe(self):
        exp = make_experiment()
        result = SampleGroupingService.validate_dataset(None, exp)
        assert result.valid is False
        assert "empty" in result.message.lower()

    def test_empty_dataframe(self):
        exp = make_experiment()
        result = SampleGroupingService.validate_dataset(pd.DataFrame(), exp)
        assert result.valid is False

    def test_missing_lipidmolec(self):
        exp = make_experiment(1, 2)
        df = pd.DataFrame({
            'intensity[s1]': [1.0],
            'intensity[s2]': [2.0],
        })
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is False
        assert "LipidMolec" in result.message

    def test_no_intensity_columns(self):
        exp = make_experiment()
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)'], 'other': [1.0]})
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is False
        assert "intensity" in result.message.lower()

    def test_sample_count_mismatch(self):
        exp = make_experiment(2, 3)  # expects 6
        df = make_df(4)  # has 4
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is False
        assert "4" in result.message
        assert "6" in result.message

    def test_msdial_double_columns_error(self):
        exp = make_experiment(1, 3)  # expects 3
        df = make_df(6)  # has 6 = 2 * 3
        result = SampleGroupingService.validate_dataset(df, exp, data_format='MS-DIAL')
        assert result.valid is False
        assert "MS-DIAL" in result.message
        assert "Lipid IS" in result.message

    def test_msdial_non_double_mismatch(self):
        exp = make_experiment(1, 3)  # expects 3
        df = make_df(5)  # has 5 ≠ 2*3
        result = SampleGroupingService.validate_dataset(df, exp, data_format='MS-DIAL')
        assert result.valid is False
        assert "MS-DIAL" not in result.message  # generic error, not MS-DIAL specific

    def test_single_sample(self):
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )
        df = make_df(1)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True
        assert result.n_intensity_cols == 1

    def test_large_dataset(self):
        exp = make_experiment(5, 10)  # 50 samples
        df = make_df(50)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True
        assert result.n_intensity_cols == 50

    def test_data_format_none(self):
        exp = make_experiment(2, 3)
        df = make_df(6)
        result = SampleGroupingService.validate_dataset(df, exp, data_format=None)
        assert result.valid is True

    def test_result_counts_on_valid(self):
        exp = make_experiment(3, 4)  # 12 samples
        df = make_df(12)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.n_intensity_cols == 12
        assert result.expected_samples == 12


# =============================================================================
# TestExtractSampleNames
# =============================================================================


class TestExtractSampleNames:
    def test_basic_extraction(self):
        df = make_df(3)
        names = SampleGroupingService.extract_sample_names(df)
        assert names == ['s1', 's2', 's3']

    def test_non_standard_names(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[WT_rep1]': [1.0],
            'intensity[KO_rep1]': [2.0],
        })
        names = SampleGroupingService.extract_sample_names(df)
        assert names == ['WT_rep1', 'KO_rep1']

    def test_no_intensity_columns(self):
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)'], 'other': [1.0]})
        names = SampleGroupingService.extract_sample_names(df)
        assert names == []

    def test_preserves_column_order(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s3]': [3.0],
            'intensity[s1]': [1.0],
            'intensity[s2]': [2.0],
        })
        names = SampleGroupingService.extract_sample_names(df)
        assert names == ['s3', 's1', 's2']  # order from columns

    def test_mixed_columns(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1.0],
            'other_col': [99],
            'intensity[s2]': [2.0],
        })
        names = SampleGroupingService.extract_sample_names(df)
        assert names == ['s1', 's2']

    def test_special_characters_in_names(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[sample (1)]': [1.0],
            'intensity[sample-2]': [2.0],
        })
        names = SampleGroupingService.extract_sample_names(df)
        assert names == ['sample (1)', 'sample-2']


# =============================================================================
# TestExtractSampleNumbers
# =============================================================================


class TestExtractSampleNumbers:
    def test_basic_extraction(self):
        df = make_df(4)
        numbers = SampleGroupingService._extract_sample_numbers(df)
        assert numbers == [1, 2, 3, 4]

    def test_sorted_output(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s3]': [3.0],
            'intensity[s1]': [1.0],
            'intensity[s5]': [5.0],
        })
        numbers = SampleGroupingService._extract_sample_numbers(df)
        assert numbers == [1, 3, 5]

    def test_non_standard_names_excluded(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[WT_rep1]': [1.0],
            'intensity[s1]': [2.0],
        })
        numbers = SampleGroupingService._extract_sample_numbers(df)
        assert numbers == [1]

    def test_no_matches(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[WT_rep1]': [1.0],
        })
        numbers = SampleGroupingService._extract_sample_numbers(df)
        assert numbers == []


# =============================================================================
# TestBuildGroupDf
# =============================================================================


class TestBuildGroupDf:
    def test_basic_grouping(self):
        exp = make_experiment(2, 3)
        df = make_df(6)
        result = SampleGroupingService.build_group_df(df, exp)
        assert isinstance(result, GroupingResult)
        assert list(result.group_df.columns) == ['sample name', 'condition']
        assert len(result.group_df) == 6
        assert result.sample_names == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_conditions_match_experiment(self):
        exp = make_experiment(2, 3)
        df = make_df(6)
        result = SampleGroupingService.build_group_df(df, exp)
        conditions = result.group_df['condition'].tolist()
        assert conditions == exp.extensive_conditions_list

    def test_workbench_conditions(self):
        exp = make_experiment(2, 2)
        df = make_df(4)
        wb_conditions = {'s1': 'WT', 's2': 'WT', 's3': 'KO', 's4': 'KO'}
        result = SampleGroupingService.build_group_df(
            df, exp,
            data_format='Metabolomics Workbench',
            workbench_conditions=wb_conditions,
        )
        conditions = result.group_df['condition'].tolist()
        assert conditions == ['WT', 'WT', 'KO', 'KO']

    def test_workbench_without_conditions_uses_experiment(self):
        exp = make_experiment(2, 2)
        df = make_df(4)
        result = SampleGroupingService.build_group_df(
            df, exp, data_format='Metabolomics Workbench',
        )
        conditions = result.group_df['condition'].tolist()
        assert conditions == exp.extensive_conditions_list

    def test_workbench_partial_conditions(self):
        exp = make_experiment(2, 2)
        df = make_df(4)
        # Only s1 and s3 have conditions
        wb_conditions = {'s1': 'WT', 's3': 'KO'}
        result = SampleGroupingService.build_group_df(
            df, exp,
            data_format='Metabolomics Workbench',
            workbench_conditions=wb_conditions,
        )
        conditions = result.group_df['condition'].tolist()
        assert conditions == ['WT', '', 'KO', '']

    def test_non_workbench_ignores_workbench_conditions(self):
        exp = make_experiment(2, 2)
        df = make_df(4)
        wb_conditions = {'s1': 'WT', 's2': 'WT', 's3': 'KO', 's4': 'KO'}
        result = SampleGroupingService.build_group_df(
            df, exp,
            data_format='Generic Format',
            workbench_conditions=wb_conditions,
        )
        conditions = result.group_df['condition'].tolist()
        assert conditions == exp.extensive_conditions_list

    def test_no_intensity_columns_raises(self):
        exp = make_experiment(1, 1)
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)'], 'other': [1.0]})
        with pytest.raises(ValueError, match="No intensity"):
            SampleGroupingService.build_group_df(df, exp)

    def test_single_sample(self):
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )
        df = make_df(1)
        result = SampleGroupingService.build_group_df(df, exp)
        assert len(result.group_df) == 1
        assert result.sample_names == ['s1']

    def test_sample_names_match_group_df(self):
        exp = make_experiment(3, 2)
        df = make_df(6)
        result = SampleGroupingService.build_group_df(df, exp)
        assert result.sample_names == result.group_df['sample name'].tolist()


# =============================================================================
# TestRegroupSamples
# =============================================================================


class TestRegroupSamples:
    def setup_method(self):
        self.exp = make_experiment(2, 3)
        self.df = make_df(6, n_lipids=3)
        result = SampleGroupingService.build_group_df(self.df, self.exp)
        self.group_df = result.group_df

    def test_basic_regroup(self):
        selections = {
            'Cond_1': ['s4', 's5', 's6'],
            'Cond_2': ['s1', 's2', 's3'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert isinstance(result, RegroupingResult)

    def test_regroup_returns_correct_types(self):
        selections = {
            'Cond_1': ['s1', 's2', 's3'],
            'Cond_2': ['s4', 's5', 's6'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert isinstance(result.group_df, pd.DataFrame)
        assert isinstance(result.reordered_df, pd.DataFrame)
        assert isinstance(result.old_to_new, dict)
        assert isinstance(result.name_df, pd.DataFrame)

    def test_regroup_updates_group_df(self):
        selections = {
            'Cond_1': ['s4', 's5', 's6'],
            'Cond_2': ['s1', 's2', 's3'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert result.group_df['sample name'].tolist() == ['s4', 's5', 's6', 's1', 's2', 's3']

    def test_regroup_creates_column_mapping(self):
        selections = {
            'Cond_1': ['s4', 's5', 's6'],
            'Cond_2': ['s1', 's2', 's3'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert result.old_to_new == {
            'intensity[s4]': 'intensity[s1]',
            'intensity[s5]': 'intensity[s2]',
            'intensity[s6]': 'intensity[s3]',
            'intensity[s1]': 'intensity[s4]',
            'intensity[s2]': 'intensity[s5]',
            'intensity[s3]': 'intensity[s6]',
        }

    def test_regroup_reorders_dataframe_columns(self):
        # Store original values
        orig_s1_vals = self.df['intensity[s1]'].values.copy()
        orig_s4_vals = self.df['intensity[s4]'].values.copy()

        selections = {
            'Cond_1': ['s4', 's5', 's6'],
            'Cond_2': ['s1', 's2', 's3'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        # After regroup: s4→s1, s1→s4
        reordered = result.reordered_df
        assert 'intensity[s1]' in reordered.columns
        # The data from original s1 should now be in s4
        np.testing.assert_array_equal(reordered['intensity[s4]'].values, orig_s1_vals)
        np.testing.assert_array_equal(reordered['intensity[s1]'].values, orig_s4_vals)

    def test_regroup_preserves_static_columns(self):
        selections = {
            'Cond_1': ['s1', 's2', 's3'],
            'Cond_2': ['s4', 's5', 's6'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert 'LipidMolec' in result.reordered_df.columns
        assert 'ClassKey' in result.reordered_df.columns

    def test_regroup_preserves_row_count(self):
        selections = {
            'Cond_1': ['s1', 's2', 's3'],
            'Cond_2': ['s4', 's5', 's6'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert len(result.reordered_df) == len(self.df)

    def test_regroup_name_df_structure(self):
        selections = {
            'Cond_1': ['s4', 's5', 's6'],
            'Cond_2': ['s1', 's2', 's3'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        assert list(result.name_df.columns) == ['old name', 'updated name', 'condition']
        assert result.name_df['updated name'].tolist() == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_regroup_missing_condition_raises(self):
        selections = {'Cond_1': ['s1', 's2', 's3']}  # missing Cond_2
        with pytest.raises(ValueError, match="do not cover all"):
            SampleGroupingService.regroup_samples(
                self.df, self.group_df, selections, self.exp,
            )

    def test_regroup_extra_condition_raises(self):
        selections = {
            'Cond_1': ['s1', 's2', 's3'],
            'Cond_2': ['s4', 's5', 's6'],
            'Cond_3': [],
        }
        with pytest.raises(ValueError, match="do not cover all"):
            SampleGroupingService.regroup_samples(
                self.df, self.group_df, selections, self.exp,
            )

    def test_identity_regroup(self):
        """Regrouping with same order should produce identical data."""
        selections = {
            'Cond_1': ['s1', 's2', 's3'],
            'Cond_2': ['s4', 's5', 's6'],
        }
        result = SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        for i in range(1, 7):
            col = f'intensity[s{i}]'
            np.testing.assert_array_equal(
                result.reordered_df[col].values,
                self.df[col].values,
            )

    def test_regroup_does_not_mutate_input(self):
        orig_df = self.df.copy()
        orig_group = self.group_df.copy()
        selections = {
            'Cond_1': ['s4', 's5', 's6'],
            'Cond_2': ['s1', 's2', 's3'],
        }
        SampleGroupingService.regroup_samples(
            self.df, self.group_df, selections, self.exp,
        )
        pd.testing.assert_frame_equal(self.df, orig_df)
        pd.testing.assert_frame_equal(self.group_df, orig_group)


# =============================================================================
# TestReorderIntensityColumns
# =============================================================================


class TestReorderIntensityColumns:
    def test_basic_reorder(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [200.0],
        })
        mapping = {
            'intensity[s1]': 'intensity[s2]',
            'intensity[s2]': 'intensity[s1]',
        }
        result = SampleGroupingService._reorder_intensity_columns(df, mapping)
        assert result['intensity[s1]'].iloc[0] == 200.0
        assert result['intensity[s2]'].iloc[0] == 100.0

    def test_preserves_static_columns(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        mapping = {'intensity[s1]': 'intensity[s1]'}
        result = SampleGroupingService._reorder_intensity_columns(df, mapping)
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [200.0],
        })
        orig = df.copy()
        mapping = {
            'intensity[s1]': 'intensity[s2]',
            'intensity[s2]': 'intensity[s1]',
        }
        SampleGroupingService._reorder_intensity_columns(df, mapping)
        pd.testing.assert_frame_equal(df, orig)

    def test_partial_mapping(self):
        """Unmapped columns keep their original names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [200.0],
        })
        mapping = {'intensity[s1]': 'intensity[s3]'}  # only s1 mapped
        result = SampleGroupingService._reorder_intensity_columns(df, mapping)
        assert 'intensity[s3]' in result.columns
        assert 'intensity[s2]' in result.columns


# =============================================================================
# TestBuildNameMapping
# =============================================================================


class TestBuildNameMapping:
    def test_basic_mapping(self):
        exp = make_experiment(2, 2)
        group_df = pd.DataFrame({
            'sample name': ['s3', 's4', 's1', 's2'],
            'condition': ['Cond_1', 'Cond_1', 'Cond_2', 'Cond_2'],
        })
        name_df = SampleGroupingService._build_name_mapping(group_df, exp)
        assert list(name_df.columns) == ['old name', 'updated name', 'condition']
        assert name_df['old name'].tolist() == ['s3', 's4', 's1', 's2']
        assert name_df['updated name'].tolist() == ['s1', 's2', 's3', 's4']

    def test_single_sample(self):
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )
        group_df = pd.DataFrame({
            'sample name': ['s1'],
            'condition': ['A'],
        })
        name_df = SampleGroupingService._build_name_mapping(group_df, exp)
        assert len(name_df) == 1
        assert name_df['updated name'].tolist() == ['s1']


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    def test_many_samples(self):
        """Test with a large number of samples."""
        exp = make_experiment(10, 10)  # 100 samples
        df = make_df(100, n_lipids=10)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True

        grp = SampleGroupingService.build_group_df(df, exp)
        assert len(grp.group_df) == 100

    def test_single_condition_single_sample(self):
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Only'],
            number_of_samples_list=[1],
        )
        df = make_df(1, n_lipids=3)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True

        grp = SampleGroupingService.build_group_df(df, exp)
        assert grp.group_df['condition'].tolist() == ['Only']

    def test_unequal_samples_per_condition(self):
        exp = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 3, 1],
        )
        df = make_df(6)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True

        grp = SampleGroupingService.build_group_df(df, exp)
        conditions = grp.group_df['condition'].tolist()
        assert conditions == ['A', 'A', 'B', 'B', 'B', 'C']

    def test_dataframe_without_classkey(self):
        exp = make_experiment(1, 2)
        df = make_df(2, with_classkey=False)
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True

    def test_regroup_with_three_conditions(self):
        exp = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[2, 2, 2],
        )
        df = make_df(6, n_lipids=3)
        grp = SampleGroupingService.build_group_df(df, exp)

        # Reverse all assignments
        selections = {
            'A': ['s5', 's6'],
            'B': ['s3', 's4'],
            'C': ['s1', 's2'],
        }
        result = SampleGroupingService.regroup_samples(
            df, grp.group_df, selections, exp,
        )
        assert result.group_df['sample name'].tolist() == ['s5', 's6', 's3', 's4', 's1', 's2']

    def test_validate_with_extra_non_intensity_columns(self):
        """Extra metadata columns should not affect validation."""
        exp = make_experiment(1, 2)
        df = make_df(2, n_lipids=2)
        df['BaseRt'] = [1.5, 2.5]
        df['CalcMass'] = [700.0, 800.0]
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True


# =============================================================================
# TestImmutability
# =============================================================================


class TestImmutability:
    def test_validate_does_not_mutate(self):
        exp = make_experiment(2, 3)
        df = make_df(6)
        orig = df.copy()
        SampleGroupingService.validate_dataset(df, exp)
        pd.testing.assert_frame_equal(df, orig)

    def test_build_group_df_does_not_mutate(self):
        exp = make_experiment(2, 3)
        df = make_df(6)
        orig = df.copy()
        SampleGroupingService.build_group_df(df, exp)
        pd.testing.assert_frame_equal(df, orig)

    def test_extract_sample_names_does_not_mutate(self):
        df = make_df(4)
        orig = df.copy()
        SampleGroupingService.extract_sample_names(df)
        pd.testing.assert_frame_equal(df, orig)


# =============================================================================
# TestTypeCoercion
# =============================================================================


class TestTypeCoercion:
    def test_integer_intensity_values(self):
        """Intensity columns with integer dtype should work."""
        exp = make_experiment(1, 2)
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'intensity[s1]': [100, 200],
            'intensity[s2]': [300, 400],
        })
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True

    def test_string_intensity_values_validate(self):
        """Validation only checks column names, not values."""
        exp = make_experiment(1, 2)
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s1]': ['abc'],
            'intensity[s2]': ['def'],
        })
        result = SampleGroupingService.validate_dataset(df, exp)
        assert result.valid is True

    def test_numpy_int64_sample_numbers(self):
        """Columns with numpy int64 indices should work."""
        exp = make_experiment(1, 3)
        data = {'LipidMolec': ['PC(16:0)']}
        for i in range(1, 4):
            data[f'intensity[s{np.int64(i)}]'] = [float(i)]
        df = pd.DataFrame(data)
        numbers = SampleGroupingService._extract_sample_numbers(df)
        assert numbers == [1, 2, 3]
