"""Unit tests for DataStandardizationService."""
import pytest
import pandas as pd
import numpy as np
from app.services.data_standardization import (
    DataStandardizationService,
    MSDIALOverrideResult,
    StandardizationResult,
)
from app.services.format_detection import DataFormat


# =============================================================================
# Helper functions for building test data
# =============================================================================


def make_lipidsearch_df(n_samples=3):
    """Build a minimal LipidSearch 5.0 DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PE'],
        'CalcMass': [757.56, 767.54],
        'BaseRt': [10.5, 12.3],
        'TotalGrade': ['A', 'B'],
        'TotalSmpIDRate(%)': [100.0, 85.0],
        'FAKey': ['(16:0_18:1)', '(18:0_20:4)'],
        **{f'MeanArea[s{i}]': [1000.0 * i, 2000.0 * i] for i in range(1, n_samples + 1)},
    })


def make_generic_df(n_samples=4, with_classkey=False):
    """Build a minimal Generic format DataFrame."""
    cols = {'Lipids': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)']}
    if with_classkey:
        cols['Class'] = ['PC', 'PE', 'TG']
    for i in range(1, n_samples + 1):
        cols[f'sample{i}'] = [1000.0 * i, 2000.0 * i, 3000.0 * i]
    return pd.DataFrame(cols)


def make_msdial_df(n_samples=3, with_normalized=False, with_header_rows=False):
    """Build a minimal MS-DIAL DataFrame.

    When with_header_rows=True, simulates the raw export where metadata rows
    appear above the actual column names, and column names are generic (0,1,...).
    """
    metadata_cols = [
        'Alignment ID', 'Average Rt(min)', 'Average Mz',
        'Metabolite name', 'Adduct type', 'Post curation result',
        'Fill %', 'MS/MS assigned', 'Total score', 'MS/MS matched',
    ]
    sample_names = [f'sample_{i}' for i in range(1, n_samples + 1)]
    norm_names = [f'norm_{i}' for i in range(1, n_samples + 1)] if with_normalized else []
    lipid_is = ['Lipid IS'] if with_normalized else []

    all_cols = metadata_cols + sample_names + lipid_is + norm_names

    if with_header_rows:
        # Build a DF where first row is the real header embedded in data rows
        header_row = all_cols
        data_rows = []
        for row_idx in range(2):
            row = []
            for j, col in enumerate(all_cols):
                if col == 'Alignment ID':
                    row.append(str(row_idx + 1))
                elif col in ('Average Rt(min)', 'Average Mz'):
                    row.append(str(round(5.0 + row_idx * 0.5, 2)))
                elif col == 'Metabolite name':
                    row.append(['PC 16:0_18:1', 'PE 18:0_20:4'][row_idx])
                elif col == 'Adduct type':
                    row.append('[M+H]+')
                elif col in ('Post curation result', 'MS/MS assigned'):
                    row.append('')
                elif col == 'Fill %':
                    row.append('100')
                elif col == 'Total score':
                    row.append(str(90 - row_idx * 10))
                elif col == 'MS/MS matched':
                    row.append('True')
                elif col == 'Lipid IS':
                    row.append('')
                else:
                    row.append(str(1000.0 * (row_idx + 1) * (j + 1)))
            data_rows.append(row)

        generic_cols = list(range(len(all_cols)))
        df = pd.DataFrame([header_row] + data_rows, columns=generic_cols)
        return df
    else:
        # Columns already correct
        data = {}
        for col in all_cols:
            if col == 'Alignment ID':
                data[col] = list(range(1, 3))
            elif col in ('Average Rt(min)', 'Average Mz'):
                data[col] = [5.0, 5.5]
            elif col == 'Metabolite name':
                data[col] = ['PC 16:0_18:1', 'PE 18:0_20:4']
            elif col == 'Adduct type':
                data[col] = ['[M+H]+', '[M+H]+']
            elif col in ('Post curation result', 'MS/MS assigned'):
                data[col] = ['', '']
            elif col == 'Fill %':
                data[col] = [100, 100]
            elif col == 'Total score':
                data[col] = [90, 80]
            elif col == 'MS/MS matched':
                data[col] = ['True', 'True']
            elif col == 'Lipid IS':
                data[col] = ['', '']
            else:
                data[col] = [1000.0, 2000.0]
        return pd.DataFrame(data)


def make_mw_text(n_samples=3, n_lipids=2):
    """Build a Metabolomics Workbench format text block."""
    sample_names = [f'sample_{i}' for i in range(1, n_samples + 1)]
    conditions = [f'Condition:Group{(i % 2) + 1}' for i in range(n_samples)]

    lines = []
    lines.append('MS_METABOLITE_DATA_START')
    lines.append('Samples,' + ','.join(sample_names))
    lines.append('Factors,' + ','.join(conditions))

    lipid_names = ['PC 16:0_18:1', 'PE 18:0_20:4', 'TG 16:0_18:1_18:2',
                   'SM 18:1_24:0', 'LPC 18:1']
    for i in range(n_lipids):
        values = [str(1000.0 * (i + 1) * (j + 1)) for j in range(n_samples)]
        lines.append(f'{lipid_names[i % len(lipid_names)]},' + ','.join(values))

    lines.append('MS_METABOLITE_DATA_END')
    return '\n'.join(lines)


# =============================================================================
# StandardizationResult Dataclass Tests
# =============================================================================


class TestStandardizationResult:
    """Tests for the StandardizationResult dataclass."""

    def test_default_values(self):
        result = StandardizationResult(success=True, message="ok")
        assert result.success is True
        assert result.message == "ok"
        assert result.standardized_df is None
        assert result.column_mapping is None
        assert result.n_intensity_cols == 0
        assert result.msdial_features is None
        assert result.msdial_sample_names is None
        assert result.workbench_conditions is None
        assert result.workbench_samples is None

    def test_failure_result(self):
        result = StandardizationResult(success=False, message="error")
        assert result.success is False

    def test_with_all_fields(self):
        df = pd.DataFrame({'a': [1]})
        result = StandardizationResult(
            success=True,
            message="ok",
            standardized_df=df,
            column_mapping=df,
            n_intensity_cols=5,
            msdial_features={'key': 'val'},
            msdial_sample_names={'s1': 'sample1'},
            workbench_conditions={'s1': 'cond1'},
            workbench_samples={'s1': 'samp1'},
        )
        assert result.n_intensity_cols == 5
        assert result.msdial_features == {'key': 'val'}


# =============================================================================
# TestStandardizeLipidName
# =============================================================================


class TestStandardizeLipidName:
    """Tests for standardize_lipid_name."""

    def test_space_separated(self):
        assert DataStandardizationService.standardize_lipid_name('LPC O-17:4') == 'LPC(O-17:4)'

    def test_space_separated_pc(self):
        assert DataStandardizationService.standardize_lipid_name('PC 16:0_18:1') == 'PC(16:0_18:1)'

    def test_slash_separated(self):
        result = DataStandardizationService.standardize_lipid_name('Cer d18:0/C24:0')
        assert result == 'Cer(d18:0_C24:0)'

    def test_already_parenthesized(self):
        result = DataStandardizationService.standardize_lipid_name('CerG1(d13:0_25:2)')
        assert result == 'CerG1(d13:0_25:2)'

    def test_already_parenthesized_with_slash(self):
        result = DataStandardizationService.standardize_lipid_name('PC(16:0/18:1)')
        assert result == 'PC(16:0_18:1)'

    def test_deuterium_d7(self):
        result = DataStandardizationService.standardize_lipid_name('LPC 18:1(d7)')
        assert result == 'LPC(18:1)(d7)'

    def test_deuterium_d9(self):
        result = DataStandardizationService.standardize_lipid_name('PE 15:0_15:0(d9)')
        assert result == 'PE(15:0_15:0)(d9)'

    def test_msdial_hydroxyl_format(self):
        result = DataStandardizationService.standardize_lipid_name('Cer 18:1;2O/24:0')
        assert result == 'Cer(18:1;2O_24:0)'

    def test_none_returns_unknown(self):
        assert DataStandardizationService.standardize_lipid_name(None) == 'Unknown'

    def test_nan_returns_unknown(self):
        assert DataStandardizationService.standardize_lipid_name(float('nan')) == 'Unknown'

    def test_empty_string_returns_unknown(self):
        assert DataStandardizationService.standardize_lipid_name('') == 'Unknown'

    def test_single_word_no_chain(self):
        result = DataStandardizationService.standardize_lipid_name('Cholesterol')
        assert result == 'Cholesterol'

    def test_splash_standard(self):
        result = DataStandardizationService.standardize_lipid_name('SPLASH LPC 18:1')
        assert 'SPLASH' in result

    def test_complex_cardiolipin(self):
        result = DataStandardizationService.standardize_lipid_name('CL(18:2/16:0/18:1/18:2)')
        assert result == 'CL(18:2_16:0_18:1_18:2)'

    def test_preserves_parenthesized_name(self):
        result = DataStandardizationService.standardize_lipid_name('PC(16:0_18:1)')
        assert result == 'PC(16:0_18:1)'

    def test_whitespace_stripped(self):
        result = DataStandardizationService.standardize_lipid_name('  PC 16:0  ')
        assert result == 'PC(16:0)'

    def test_numeric_input(self):
        result = DataStandardizationService.standardize_lipid_name(12345)
        # Should not crash; returns string representation
        assert isinstance(result, str)

    def test_special_chars_in_name(self):
        result = DataStandardizationService.standardize_lipid_name('PE-NMe2 18:1/20:4')
        assert isinstance(result, str)

    def test_multiple_slashes(self):
        result = DataStandardizationService.standardize_lipid_name('TG 16:0/18:1/18:2')
        assert result == 'TG(16:0_18:1_18:2)'

    def test_o_ether_lipid(self):
        result = DataStandardizationService.standardize_lipid_name('PE O-18:0/20:4')
        assert result == 'PE(O-18:0_20:4)'


# =============================================================================
# TestInferClassKey
# =============================================================================


class TestInferClassKey:
    """Tests for infer_class_key."""

    def test_standard_pc(self):
        assert DataStandardizationService.infer_class_key('PC(16:0_18:1)') == 'PC'

    def test_standard_cl(self):
        assert DataStandardizationService.infer_class_key('CL(18:2_16:0)') == 'CL'

    def test_lpc(self):
        assert DataStandardizationService.infer_class_key('LPC(18:1)') == 'LPC'

    def test_splash_lpc(self):
        assert DataStandardizationService.infer_class_key('SPLASH(LPC 18:1)(d7)') == 'LPC'

    def test_splash_fa(self):
        assert DataStandardizationService.infer_class_key('SPLASH(FA 20:4)(d11)') == 'FA'

    def test_none_returns_none_string(self):
        # infer_class_key converts to str first, so None -> 'None' -> match 'None'
        result = DataStandardizationService.infer_class_key(None)
        assert result == 'None'

    def test_nan_returns_nan_string(self):
        # infer_class_key converts to str first, so nan -> 'nan' -> match 'nan'
        result = DataStandardizationService.infer_class_key(float('nan'))
        assert result == 'nan'

    def test_empty_string(self):
        assert DataStandardizationService.infer_class_key('') == 'Unknown'

    def test_numeric_only(self):
        assert DataStandardizationService.infer_class_key('12345') == 'Unknown'

    def test_single_letter(self):
        result = DataStandardizationService.infer_class_key('A')
        assert result == 'A'

    def test_complex_class_name(self):
        assert DataStandardizationService.infer_class_key('CerG1(d13:0_25:2)') == 'CerG1'

    def test_cholesterol(self):
        assert DataStandardizationService.infer_class_key('Cholesterol') == 'Cholesterol'


# =============================================================================
# TestLipidSearchStandardization
# =============================================================================


class TestLipidSearchStandardization:
    """Tests for LipidSearch standardization."""

    def test_valid_basic(self):
        df = make_lipidsearch_df()
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is True
        assert result.standardized_df is not None

    def test_renames_meanarea_to_intensity(self):
        df = make_lipidsearch_df(n_samples=3)
        result = DataStandardizationService._process_lipidsearch(df)
        sdf = result.standardized_df
        for i in range(1, 4):
            assert f'intensity[s{i}]' in sdf.columns
            assert f'MeanArea[s{i}]' not in sdf.columns

    def test_preserves_metadata_columns(self):
        df = make_lipidsearch_df()
        result = DataStandardizationService._process_lipidsearch(df)
        sdf = result.standardized_df
        for col in ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalGrade']:
            assert col in sdf.columns

    def test_column_mapping_correct(self):
        df = make_lipidsearch_df(n_samples=2)
        result = DataStandardizationService._process_lipidsearch(df)
        mapping = result.column_mapping
        assert mapping is not None
        assert 'standardized_name' in mapping.columns
        assert 'original_name' in mapping.columns
        assert len(mapping) == 2

    def test_n_intensity_cols(self):
        df = make_lipidsearch_df(n_samples=5)
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.n_intensity_cols == 5

    def test_missing_lipidmolec(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=['LipidMolec'])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False
        assert 'LipidMolec' in result.message

    def test_missing_classkey(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=['ClassKey'])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False

    def test_missing_calcmass(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=['CalcMass'])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False

    def test_missing_basert(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=['BaseRt'])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False

    def test_missing_totalgrade(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=['TotalGrade'])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False

    def test_missing_fakey(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=['FAKey'])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False

    def test_no_meanarea_columns(self):
        df = make_lipidsearch_df()
        df = df.drop(columns=[c for c in df.columns if c.startswith('MeanArea[')])
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is False
        assert 'MeanArea' in result.message

    def test_non_dataframe_input(self):
        result = DataStandardizationService._process_lipidsearch("not a df")
        assert result.success is False

    def test_empty_df_with_correct_columns(self):
        df = make_lipidsearch_df()
        empty = df.iloc[:0]
        result = DataStandardizationService._process_lipidsearch(empty)
        assert result.success is True
        assert len(result.standardized_df) == 0

    def test_intensity_values_preserved(self):
        df = make_lipidsearch_df(n_samples=2)
        result = DataStandardizationService._process_lipidsearch(df)
        sdf = result.standardized_df
        assert sdf['intensity[s1]'].tolist() == [1000.0, 2000.0]
        assert sdf['intensity[s2]'].tolist() == [2000.0, 4000.0]


# =============================================================================
# TestGenericStandardization
# =============================================================================


class TestGenericStandardization:
    """Tests for Generic format standardization."""

    def test_basic_without_classkey(self):
        df = make_generic_df(n_samples=3, with_classkey=False)
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        sdf = result.standardized_df
        assert 'LipidMolec' in sdf.columns
        assert 'ClassKey' in sdf.columns

    def test_basic_with_classkey(self):
        df = make_generic_df(n_samples=3, with_classkey=True)
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        sdf = result.standardized_df
        assert 'ClassKey' in sdf.columns

    def test_first_col_becomes_lipidmolec(self):
        df = make_generic_df()
        result = DataStandardizationService._process_generic(df)
        assert result.standardized_df.columns[0] == 'LipidMolec'

    def test_intensity_columns_named(self):
        df = make_generic_df(n_samples=3, with_classkey=False)
        result = DataStandardizationService._process_generic(df)
        sdf = result.standardized_df
        for i in range(1, 4):
            assert f'intensity[s{i}]' in sdf.columns

    def test_n_intensity_cols_without_classkey(self):
        df = make_generic_df(n_samples=4, with_classkey=False)
        result = DataStandardizationService._process_generic(df)
        assert result.n_intensity_cols == 4

    def test_n_intensity_cols_with_classkey(self):
        df = make_generic_df(n_samples=4, with_classkey=True)
        result = DataStandardizationService._process_generic(df)
        assert result.n_intensity_cols == 4

    def test_column_mapping_without_classkey(self):
        df = make_generic_df(n_samples=2, with_classkey=False)
        result = DataStandardizationService._process_generic(df)
        mapping = result.column_mapping
        assert mapping is not None
        assert len(mapping) == 3  # LipidMolec + 2 intensity

    def test_column_mapping_with_classkey(self):
        df = make_generic_df(n_samples=2, with_classkey=True)
        result = DataStandardizationService._process_generic(df)
        mapping = result.column_mapping
        assert mapping is not None
        assert len(mapping) == 4  # LipidMolec + ClassKey + 2 intensity

    def test_lipid_name_standardized(self):
        df = pd.DataFrame({
            'Lipids': ['PC 16:0/18:1', 'PE 18:0/20:4'],
            'sample1': [100, 200],
        })
        result = DataStandardizationService._process_generic(df)
        sdf = result.standardized_df
        assert sdf['LipidMolec'].iloc[0] == 'PC(16:0_18:1)'
        assert sdf['LipidMolec'].iloc[1] == 'PE(18:0_20:4)'

    def test_classkey_inferred_when_absent(self):
        df = make_generic_df(n_samples=2, with_classkey=False)
        result = DataStandardizationService._process_generic(df)
        sdf = result.standardized_df
        assert sdf['ClassKey'].iloc[0] == 'PC'
        assert sdf['ClassKey'].iloc[1] == 'PE'

    def test_minimum_two_columns(self):
        df = pd.DataFrame({'Lipids': ['PC(16:0_18:1)'], 'sample': [100.0]})
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        assert result.n_intensity_cols == 1

    def test_single_data_row(self):
        df = pd.DataFrame({
            'Lipids': ['PC(16:0_18:1)'],
            's1': [100.0],
            's2': [200.0],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        assert len(result.standardized_df) == 1

    def test_many_columns(self):
        cols = {'Lipids': ['PC(16:0_18:1)']}
        for i in range(1, 51):
            cols[f'sample{i}'] = [float(i)]
        df = pd.DataFrame(cols)
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        assert result.n_intensity_cols == 50

    def test_less_than_2_columns(self):
        df = pd.DataFrame({'Lipids': ['PC(16:0_18:1)']})
        result = DataStandardizationService._process_generic(df)
        assert result.success is False
        assert 'at least 2 columns' in result.message

    def test_first_column_all_numeric(self):
        df = pd.DataFrame({
            'col1': [123, 456, 789],
            'col2': [100, 200, 300],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is False
        assert 'lipid names' in result.message.lower()

    def test_non_dataframe_input(self):
        result = DataStandardizationService._process_generic("not a df")
        assert result.success is False

    def test_message_on_success(self):
        df = make_generic_df()
        result = DataStandardizationService._process_generic(df)
        assert 'successfully' in result.message.lower()


class TestIsClasskeyColumn:
    """Tests for _is_classkey_column helper."""

    def test_true_for_short_alpha_strings(self):
        df = pd.DataFrame({
            'lipids': ['PC(16:0)', 'PE(18:0)'],
            'class': ['PC', 'PE'],
            'vals': [100, 200],
        })
        assert DataStandardizationService._is_classkey_column(df, 1) == True

    def test_false_for_numeric_column(self):
        df = pd.DataFrame({
            'lipids': ['PC(16:0)', 'PE(18:0)'],
            'vals': [100.0, 200.0],
        })
        assert DataStandardizationService._is_classkey_column(df, 1) == False

    def test_true_for_classkey_named_column(self):
        df = pd.DataFrame({
            'lipids': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'vals': [100],
        })
        assert DataStandardizationService._is_classkey_column(df, 1) == True

    def test_false_for_out_of_range_index(self):
        df = pd.DataFrame({'a': [1], 'b': [2]})
        assert DataStandardizationService._is_classkey_column(df, 5) == False

    def test_false_for_long_strings(self):
        df = pd.DataFrame({
            'lipids': ['PC(16:0_18:1_20:4)', 'PE(18:0_20:4_22:6)'],
            'descriptions': [
                'This is a very long description text here',
                'Another very long description string here',
            ],
            'vals': [100, 200],
        })
        assert DataStandardizationService._is_classkey_column(df, 1) == False

    def test_false_for_mostly_numeric_strings(self):
        df = pd.DataFrame({
            'lipids': ['PC(16:0)', 'PE(18:0)'],
            'nums': ['12345', '67890'],
        })
        assert DataStandardizationService._is_classkey_column(df, 1) == False


# =============================================================================
# TestMSDIALStandardization
# =============================================================================


class TestMSDIALStandardization:
    """Tests for MS-DIAL standardization."""

    def test_valid_without_header_rows(self):
        df = make_msdial_df(n_samples=3, with_header_rows=False)
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        assert result.standardized_df is not None

    def test_valid_with_header_rows(self):
        df = make_msdial_df(n_samples=3, with_header_rows=True)
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True

    def test_lipidmolec_column_created(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert 'LipidMolec' in result.standardized_df.columns

    def test_classkey_inferred(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert 'ClassKey' in result.standardized_df.columns
        assert result.standardized_df['ClassKey'].iloc[0] == 'PC'

    def test_lipid_names_standardized(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert result.standardized_df['LipidMolec'].iloc[0] == 'PC(16:0_18:1)'

    def test_intensity_columns_named(self):
        df = make_msdial_df(n_samples=3)
        result = DataStandardizationService._process_msdial(df)
        sdf = result.standardized_df
        for i in range(1, 4):
            assert f'intensity[s{i}]' in sdf.columns

    def test_n_intensity_cols(self):
        df = make_msdial_df(n_samples=4)
        result = DataStandardizationService._process_msdial(df)
        assert result.n_intensity_cols == 4

    def test_raw_only_samples(self):
        df = make_msdial_df(n_samples=3, with_normalized=False)
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        assert result.n_intensity_cols == 3

    def test_raw_and_normalized_default_uses_raw(self):
        df = make_msdial_df(n_samples=2, with_normalized=True)
        result = DataStandardizationService._process_msdial(df, use_normalized=False)
        assert result.success is True
        assert result.n_intensity_cols == 2
        # Should use raw columns
        assert result.msdial_sample_names['s1'] == 'sample_1'

    def test_use_normalized_flag(self):
        df = make_msdial_df(n_samples=2, with_normalized=True)
        result = DataStandardizationService._process_msdial(df, use_normalized=True)
        assert result.success is True
        assert result.msdial_sample_names['s1'] == 'norm_1'

    def test_features_dict_keys(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        features = result.msdial_features
        assert features is not None
        expected_keys = [
            'has_ontology', 'has_quality_score', 'has_msms_matched',
            'has_rt', 'has_mz', 'has_normalized_data', 'lipid_column',
            'lipid_column_index', 'raw_sample_columns', 'raw_sample_indices',
            'normalized_sample_columns', 'normalized_sample_indices',
            'header_row_index', 'actual_columns', 'column_indices',
        ]
        for key in expected_keys:
            assert key in features

    def test_features_quality_score(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['has_quality_score'] is True

    def test_features_msms_matched(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['has_msms_matched'] is True

    def test_quality_columns_preserved(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        sdf = result.standardized_df
        assert 'Total score' in sdf.columns

    def test_optional_basert(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert 'BaseRt' in result.standardized_df.columns

    def test_optional_calcmass(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert 'CalcMass' in result.standardized_df.columns

    def test_column_mapping(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        mapping = result.column_mapping
        assert mapping is not None
        assert 'standardized_name' in mapping.columns
        assert 'original_name' in mapping.columns
        # At minimum: LipidMolec + BaseRt + CalcMass + 2 intensity
        assert len(mapping) >= 4

    def test_sample_names_map(self):
        df = make_msdial_df(n_samples=3)
        result = DataStandardizationService._process_msdial(df)
        sample_names = result.msdial_sample_names
        assert sample_names is not None
        assert sample_names['s1'] == 'sample_1'
        assert sample_names['s2'] == 'sample_2'
        assert sample_names['s3'] == 'sample_3'

    def test_missing_lipid_name_column(self):
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'SomeOtherCol': ['a', 'b'],
            'sample1': [100, 200],
        })
        result = DataStandardizationService._process_msdial(df)
        assert result.success is False
        assert 'lipid name' in result.message.lower()

    def test_no_sample_columns(self):
        # Only metadata columns, no numeric sample data
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'Average Rt(min)': [5.0, 5.5],
            'Average Mz': [700.0, 800.0],
            'Metabolite name': ['PC 16:0', 'PE 18:0'],
            'Adduct type': ['[M+H]+', '[M+H]+'],
        })
        result = DataStandardizationService._process_msdial(df)
        # The numeric metadata cols may or may not be detected as samples
        # depending on filtering, but at least it shouldn't crash
        assert isinstance(result, StandardizationResult)

    def test_non_dataframe_input(self):
        result = DataStandardizationService._process_msdial("not a df")
        assert result.success is False

    def test_intensity_values_numeric(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        sdf = result.standardized_df
        assert pd.api.types.is_numeric_dtype(sdf['intensity[s1]'])

    def test_nan_intensities_filled_zero(self):
        # Build a DF with enough rows so one NaN still passes >50% numeric threshold
        df = pd.DataFrame({
            'Alignment ID': [1, 2, 3],
            'Average Rt(min)': [5.0, 5.5, 6.0],
            'Average Mz': [700.0, 800.0, 900.0],
            'Metabolite name': ['PC 16:0', 'PE 18:0', 'SM 18:1'],
            'Adduct type': ['[M+H]+'] * 3,
            'Post curation result': [''] * 3,
            'Fill %': [100] * 3,
            'MS/MS assigned': [''] * 3,
            'Total score': [90, 80, 70],
            'MS/MS matched': ['True'] * 3,
            'sample_1': [np.nan, 2000.0, 3000.0],  # 1 NaN out of 3 -> 66% numeric
            'sample_2': [1000.0, 2000.0, 3000.0],
        })
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        # Find which intensity column corresponds to sample_1
        for k, v in result.msdial_sample_names.items():
            if v == 'sample_1':
                assert result.standardized_df[f'intensity[{k}]'].iloc[0] == 0.0
                break

    def test_features_has_normalized_data_false(self):
        df = make_msdial_df(n_samples=2, with_normalized=False)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['has_normalized_data'] is False

    def test_features_has_normalized_data_true(self):
        df = make_msdial_df(n_samples=2, with_normalized=True)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['has_normalized_data'] is True


# =============================================================================
# TestMetabolomicsWorkbenchStandardization
# =============================================================================


class TestMetabolomicsWorkbenchStandardization:
    """Tests for Metabolomics Workbench standardization."""

    def test_valid_text(self):
        text = make_mw_text(n_samples=3, n_lipids=2)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True
        assert result.standardized_df is not None

    def test_lipid_names_standardized(self):
        text = make_mw_text(n_samples=2, n_lipids=1)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        sdf = result.standardized_df
        assert sdf['LipidMolec'].iloc[0] == 'PC(16:0_18:1)'

    def test_classkey_inferred(self):
        text = make_mw_text(n_samples=2, n_lipids=2)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        sdf = result.standardized_df
        assert 'ClassKey' in sdf.columns
        assert sdf['ClassKey'].iloc[0] == 'PC'

    def test_condition_prefix_cleaned(self):
        text = make_mw_text(n_samples=3)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        conditions = result.workbench_conditions
        for key, cond in conditions.items():
            assert not cond.startswith('Condition:')

    def test_workbench_conditions_populated(self):
        text = make_mw_text(n_samples=3)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.workbench_conditions is not None
        assert len(result.workbench_conditions) == 3
        assert 's1' in result.workbench_conditions

    def test_workbench_samples_populated(self):
        text = make_mw_text(n_samples=3)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.workbench_samples is not None
        assert result.workbench_samples['s1'] == 'sample_1'

    def test_intensity_columns_numeric(self):
        text = make_mw_text(n_samples=2, n_lipids=2)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        sdf = result.standardized_df
        assert pd.api.types.is_numeric_dtype(sdf['intensity[s1]'])

    def test_n_intensity_cols(self):
        text = make_mw_text(n_samples=4, n_lipids=2)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.n_intensity_cols == 4

    def test_missing_start_marker(self):
        text = "some data\nMS_METABOLITE_DATA_END"
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is False
        assert 'markers' in result.message.lower()

    def test_missing_end_marker(self):
        text = "MS_METABOLITE_DATA_START\nsome data"
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is False

    def test_missing_both_markers(self):
        text = "just some random text"
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is False

    def test_invalid_marker_order(self):
        text = "MS_METABOLITE_DATA_END\nsome data\nMS_METABOLITE_DATA_START"
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is False

    def test_insufficient_data_rows(self):
        text = "MS_METABOLITE_DATA_START\nSamples,s1\nMS_METABOLITE_DATA_END"
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is False

    def test_non_string_input(self):
        df = pd.DataFrame({'a': [1]})
        result = DataStandardizationService._process_metabolomics_workbench(df)
        assert result.success is False
        assert 'type' in result.message.lower()

    def test_empty_data_section(self):
        text = (
            "MS_METABOLITE_DATA_START\n"
            "Samples,s1,s2\n"
            "Factors,Condition:A,Condition:B\n"
            "MS_METABOLITE_DATA_END"
        )
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is False


# =============================================================================
# TestValidateAndStandardize
# =============================================================================


class TestValidateAndStandardize:
    """Tests for the main validate_and_standardize entry point."""

    def test_dispatches_lipidsearch(self):
        df = make_lipidsearch_df()
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.LIPIDSEARCH
        )
        assert result.success is True

    def test_dispatches_generic(self):
        df = make_generic_df()
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.GENERIC
        )
        assert result.success is True

    def test_dispatches_msdial(self):
        df = make_msdial_df()
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.MSDIAL
        )
        assert result.success is True

    def test_dispatches_metabolomics_workbench(self):
        text = make_mw_text()
        result = DataStandardizationService.validate_and_standardize(
            text, DataFormat.METABOLOMICS_WORKBENCH
        )
        assert result.success is True

    def test_unsupported_format(self):
        df = make_generic_df()
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.UNKNOWN
        )
        assert result.success is False
        assert 'Unsupported' in result.message

    def test_exception_handling(self):
        # Pass incompatible types to trigger exception
        result = DataStandardizationService.validate_and_standardize(
            None, DataFormat.LIPIDSEARCH
        )
        assert result.success is False

    def test_msdial_use_normalized_param(self):
        df = make_msdial_df(n_samples=2, with_normalized=True)
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.MSDIAL, msdial_use_normalized=True
        )
        assert result.success is True
        assert result.msdial_sample_names['s1'] == 'norm_1'

    def test_result_fields_populated_lipidsearch(self):
        df = make_lipidsearch_df()
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.LIPIDSEARCH
        )
        assert result.standardized_df is not None
        assert result.column_mapping is not None
        assert result.n_intensity_cols > 0

    def test_result_fields_populated_msdial(self):
        df = make_msdial_df()
        result = DataStandardizationService.validate_and_standardize(
            df, DataFormat.MSDIAL
        )
        assert result.msdial_features is not None
        assert result.msdial_sample_names is not None

    def test_result_fields_populated_workbench(self):
        text = make_mw_text()
        result = DataStandardizationService.validate_and_standardize(
            text, DataFormat.METABOLOMICS_WORKBENCH
        )
        assert result.workbench_conditions is not None
        assert result.workbench_samples is not None

    def test_failure_result_has_no_df(self):
        result = DataStandardizationService.validate_and_standardize(
            "bad", DataFormat.LIPIDSEARCH
        )
        assert result.success is False
        assert result.standardized_df is None

    def test_exception_wrapping_message(self):
        result = DataStandardizationService.validate_and_standardize(
            12345, DataFormat.GENERIC
        )
        assert result.success is False
        assert 'Error' in result.message or 'Expected' in result.message


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests across all formats."""

    def test_large_dataframe_lipidsearch(self):
        n = 1000
        df = pd.DataFrame({
            'LipidMolec': [f'PC({i}:0_{i+1}:0)' for i in range(n)],
            'ClassKey': ['PC'] * n,
            'CalcMass': [700.0] * n,
            'BaseRt': [10.0] * n,
            'TotalGrade': ['A'] * n,
            'TotalSmpIDRate(%)': [100.0] * n,
            'FAKey': [f'({i}:0_{i+1}:0)' for i in range(n)],
            'MeanArea[s1]': np.random.rand(n) * 10000,
            'MeanArea[s2]': np.random.rand(n) * 10000,
        })
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is True
        assert len(result.standardized_df) == n

    def test_large_dataframe_generic(self):
        n = 1000
        cols = {'Lipids': [f'PC({i}:0)' for i in range(n)]}
        cols['sample1'] = np.random.rand(n) * 10000
        cols['sample2'] = np.random.rand(n) * 10000
        df = pd.DataFrame(cols)
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        assert len(result.standardized_df) == n

    def test_all_nan_intensities_generic(self):
        df = pd.DataFrame({
            'Lipids': ['PC(16:0)', 'PE(18:0)'],
            'sample1': [np.nan, np.nan],
            'sample2': [np.nan, np.nan],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is True

    def test_special_chars_in_lipid_names(self):
        df = pd.DataFrame({
            'Lipids': ['PC-O/P(16:0_18:1)', 'LPE(18:1e)'],
            'sample1': [100, 200],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is True

    def test_unicode_in_column_names(self):
        df = pd.DataFrame({
            'Lipid\u00e9': ['PC(16:0)', 'PE(18:0)'],
            'sample\u00fc1': [100, 200],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is True

    def test_duplicate_column_names_generic(self):
        df = pd.DataFrame(
            [[' PC(16:0)', 100, 200], ['PE(18:0)', 300, 400]],
            columns=['Lipids', 'sample1', 'sample1'],
        )
        result = DataStandardizationService._process_generic(df)
        # Should not crash
        assert isinstance(result, StandardizationResult)

    def test_empty_dataframe_generic(self):
        df = pd.DataFrame({'Lipids': pd.Series(dtype=str), 'sample1': pd.Series(dtype=float)})
        result = DataStandardizationService._process_generic(df)
        # Might fail because no letters found, or succeed with 0 rows
        assert isinstance(result, StandardizationResult)

    def test_whitespace_only_lipid_names(self):
        df = pd.DataFrame({
            'Lipids': ['  ', '   '],
            'sample1': [100, 200],
        })
        # No letters found -> should fail
        result = DataStandardizationService._process_generic(df)
        assert result.success is False

    def test_mixed_types_in_lipid_column(self):
        df = pd.DataFrame({
            'Lipids': ['PC(16:0)', 123, 'PE(18:0)'],
            'sample1': [100, 200, 300],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is True

    def test_very_long_lipid_name(self):
        long_name = 'A' * 1000
        result = DataStandardizationService.standardize_lipid_name(long_name)
        assert isinstance(result, str)


# =============================================================================
# TestImmutability
# =============================================================================


class TestImmutability:
    """Tests that input DataFrames are not modified."""

    def test_lipidsearch_input_unchanged(self):
        df = make_lipidsearch_df()
        original_cols = df.columns.tolist()
        original_values = df.values.copy()
        DataStandardizationService._process_lipidsearch(df)
        assert df.columns.tolist() == original_cols
        np.testing.assert_array_equal(df.values, original_values)

    def test_generic_input_unchanged(self):
        df = make_generic_df()
        original_cols = df.columns.tolist()
        original_values = df.values.copy()
        DataStandardizationService._process_generic(df)
        assert df.columns.tolist() == original_cols
        np.testing.assert_array_equal(df.values, original_values)

    def test_msdial_input_unchanged(self):
        df = make_msdial_df()
        original_cols = df.columns.tolist()
        original_shape = df.shape
        DataStandardizationService._process_msdial(df)
        assert df.columns.tolist() == original_cols
        assert df.shape == original_shape

    def test_generic_with_classkey_input_unchanged(self):
        df = make_generic_df(with_classkey=True)
        original_cols = df.columns.tolist()
        original_values = df.values.copy()
        DataStandardizationService._process_generic(df)
        assert df.columns.tolist() == original_cols
        np.testing.assert_array_equal(df.values, original_values)

    def test_lipidsearch_copy_is_independent(self):
        df = make_lipidsearch_df()
        result = DataStandardizationService._process_lipidsearch(df)
        result.standardized_df['intensity[s1]'] = 0
        assert df['MeanArea[s1]'].iloc[0] != 0


# =============================================================================
# TestTypeCoercion
# =============================================================================


class TestTypeCoercion:
    """Tests for type coercion in intensity columns."""

    def test_string_numbers_to_numeric_msdial(self):
        df = make_msdial_df(n_samples=2)
        # Convert sample values to strings
        df['sample_1'] = df['sample_1'].astype(str)
        df['sample_2'] = df['sample_2'].astype(str)
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        assert pd.api.types.is_numeric_dtype(result.standardized_df['intensity[s1]'])

    def test_string_numbers_to_numeric_workbench(self):
        text = make_mw_text(n_samples=2, n_lipids=2)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True
        assert pd.api.types.is_numeric_dtype(result.standardized_df['intensity[s1]'])

    def test_int_intensity_values_lipidsearch(self):
        df = make_lipidsearch_df(n_samples=2)
        df['MeanArea[s1]'] = df['MeanArea[s1]'].astype(int)
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is True

    def test_float_intensity_values_lipidsearch(self):
        df = make_lipidsearch_df(n_samples=2)
        result = DataStandardizationService._process_lipidsearch(df)
        assert result.success is True

    def test_mixed_types_msdial_intensity(self):
        df = make_msdial_df(n_samples=2)
        # Mix string and numeric
        col_idx = df.columns.tolist().index('sample_1')
        df.iloc[0, col_idx] = '500.5'
        df.iloc[1, col_idx] = 1000
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True

    def test_non_numeric_string_becomes_nan_then_zero_msdial(self):
        # Build DF with 3 rows so one non-numeric still passes >50% threshold
        df = pd.DataFrame({
            'Alignment ID': [1, 2, 3],
            'Average Rt(min)': [5.0, 5.5, 6.0],
            'Average Mz': [700.0, 800.0, 900.0],
            'Metabolite name': ['PC 16:0', 'PE 18:0', 'SM 18:1'],
            'Adduct type': ['[M+H]+'] * 3,
            'Post curation result': [''] * 3,
            'Fill %': [100] * 3,
            'MS/MS assigned': [''] * 3,
            'Total score': [90, 80, 70],
            'MS/MS matched': ['True'] * 3,
            'sample_1': ['not_a_number', 2000.0, 3000.0],
            'sample_2': [1000.0, 2000.0, 3000.0],
        })
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        for k, v in result.msdial_sample_names.items():
            if v == 'sample_1':
                assert result.standardized_df[f'intensity[{k}]'].iloc[0] == 0.0
                break

    def test_workbench_nan_intensity_filled(self):
        text = make_mw_text(n_samples=2, n_lipids=1)
        # Replace a value with non-numeric
        text = text.replace('1000.0', 'NA')
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True
        sdf = result.standardized_df
        assert sdf['intensity[s1]'].iloc[0] == 0.0

    def test_generic_preserves_numeric_types(self):
        df = pd.DataFrame({
            'Lipids': ['PC(16:0)', 'PE(18:0)'],
            'sample1': [100.5, 200.5],
            'sample2': [300.5, 400.5],
        })
        result = DataStandardizationService._process_generic(df)
        sdf = result.standardized_df
        assert sdf['intensity[s1]'].iloc[0] == 100.5


# =============================================================================
# TestDetectMSDIALSampleColumns
# =============================================================================


class TestDetectMSDIALSampleColumns:
    """Tests for _detect_msdial_sample_columns helper."""

    def test_raw_only(self):
        df = make_msdial_df(n_samples=3, with_normalized=False)
        raw, norm, lip_idx, raw_idx, norm_idx = (
            DataStandardizationService._detect_msdial_sample_columns(df)
        )
        assert len(raw) == 3
        assert len(norm) == 0
        assert lip_idx is None

    def test_with_lipid_is_separator(self):
        df = make_msdial_df(n_samples=2, with_normalized=True)
        raw, norm, lip_idx, raw_idx, norm_idx = (
            DataStandardizationService._detect_msdial_sample_columns(df)
        )
        assert len(raw) == 2
        assert len(norm) == 2
        assert lip_idx is not None

    def test_custom_columns_parameter(self):
        df = make_msdial_df(n_samples=2)
        columns = df.columns.tolist()
        raw, norm, lip_idx, raw_idx, norm_idx = (
            DataStandardizationService._detect_msdial_sample_columns(df, columns=columns)
        )
        assert len(raw) == 2

    def test_metadata_cols_excluded(self):
        df = make_msdial_df(n_samples=2)
        raw, norm, lip_idx, raw_idx, norm_idx = (
            DataStandardizationService._detect_msdial_sample_columns(df)
        )
        for col in raw:
            assert col not in ('Alignment ID', 'Average Rt(min)', 'Average Mz')


# =============================================================================
# Additional targeted tests for full coverage
# =============================================================================


class TestStandardizeLipidNameAdditional:
    """Additional edge-case tests for standardize_lipid_name."""

    def test_splash_with_parenthesized_class(self):
        result = DataStandardizationService.standardize_lipid_name('SPLASH(LPC 18:1)(d7)')
        # SPLASH already has parens so goes through paren branch
        assert 'SPLASH' in result
        assert '(d7)' in result

    def test_name_with_dash_separator(self):
        result = DataStandardizationService.standardize_lipid_name('SM d18:1/24:0')
        assert result == 'SM(d18:1_24:0)'

    def test_false_value(self):
        result = DataStandardizationService.standardize_lipid_name(False)
        assert result == 'Unknown'

    def test_zero_value(self):
        result = DataStandardizationService.standardize_lipid_name(0)
        assert result == 'Unknown'

    def test_list_input(self):
        # Should not crash
        result = DataStandardizationService.standardize_lipid_name([1, 2, 3])
        assert isinstance(result, str)

    def test_only_class_no_chain(self):
        result = DataStandardizationService.standardize_lipid_name('Cholesterol')
        assert result == 'Cholesterol'

    def test_complex_nested_parens(self):
        result = DataStandardizationService.standardize_lipid_name('CL(18:2(9Z,12Z)/16:0/18:1/18:2)')
        assert isinstance(result, str)
        assert 'CL' in result


class TestMetabolomicsWorkbenchAdditional:
    """Additional edge-case tests for Metabolomics Workbench."""

    def test_conditions_without_prefix(self):
        lines = [
            'MS_METABOLITE_DATA_START',
            'Samples,s1,s2',
            'Factors,GroupA,GroupB',
            'PC 16:0,100,200',
            'MS_METABOLITE_DATA_END',
        ]
        text = '\n'.join(lines)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True
        assert result.workbench_conditions['s1'] == 'GroupA'

    def test_extra_whitespace_in_lines(self):
        lines = [
            '  MS_METABOLITE_DATA_START  ',
            '  Samples , s1 , s2  ',
            '  Factors , Condition:A , Condition:B  ',
            '  PC 16:0 , 100 , 200  ',
            '  MS_METABOLITE_DATA_END  ',
        ]
        text = '\n'.join(lines)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        # Should handle whitespace in markers and data
        assert isinstance(result, StandardizationResult)

    def test_multiple_lipid_rows(self):
        text = make_mw_text(n_samples=2, n_lipids=5)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True
        assert len(result.standardized_df) == 5

    def test_single_sample(self):
        text = make_mw_text(n_samples=1, n_lipids=2)
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True
        assert result.n_intensity_cols == 1

    def test_text_before_and_after_markers(self):
        text = (
            'HEADER_INFO\n'
            'Some preamble text\n'
            + make_mw_text(n_samples=2, n_lipids=2) +
            '\nSome trailing text\n'
        )
        result = DataStandardizationService._process_metabolomics_workbench(text)
        assert result.success is True


class TestGenericStandardizationAdditional:
    """Additional edge case tests for Generic format."""

    def test_column_named_classkey(self):
        df = pd.DataFrame({
            'Lipids': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'sample1': [100, 200],
        })
        result = DataStandardizationService._process_generic(df)
        assert result.success is True
        # ClassKey column detected
        sdf = result.standardized_df
        assert 'ClassKey' in sdf.columns

    def test_no_classkey_column_infers_from_names(self):
        df = pd.DataFrame({
            'Lipids': ['TG(16:0_18:1_18:2)', 'DG(16:0_18:1)'],
            'sample1': [100, 200],
        })
        result = DataStandardizationService._process_generic(df)
        sdf = result.standardized_df
        assert sdf['ClassKey'].iloc[0] == 'TG'
        assert sdf['ClassKey'].iloc[1] == 'DG'

    def test_intensity_column_ordering(self):
        df = pd.DataFrame({
            'Lipids': ['PC(16:0)'],
            'A': [1.0],
            'B': [2.0],
            'C': [3.0],
        })
        result = DataStandardizationService._process_generic(df)
        sdf = result.standardized_df
        assert sdf['intensity[s1]'].iloc[0] == 1.0
        assert sdf['intensity[s2]'].iloc[0] == 2.0
        assert sdf['intensity[s3]'].iloc[0] == 3.0


class TestMSDIALStandardizationAdditional:
    """Additional tests for MS-DIAL edge cases."""

    def test_header_row_detection_with_alignment_id(self):
        df = make_msdial_df(n_samples=2, with_header_rows=True)
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        assert result.msdial_features['header_row_index'] >= 0

    def test_without_header_row_index_negative(self):
        df = make_msdial_df(n_samples=2, with_header_rows=False)
        result = DataStandardizationService._process_msdial(df)
        assert result.success is True
        assert result.msdial_features['header_row_index'] == -1

    def test_features_lipid_column_name(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['lipid_column'] == 'Metabolite name'

    def test_features_has_rt(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['has_rt'] is True

    def test_features_has_mz(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert result.msdial_features['has_mz'] is True

    def test_basert_numeric(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert pd.api.types.is_numeric_dtype(result.standardized_df['BaseRt'])

    def test_calcmass_numeric(self):
        df = make_msdial_df(n_samples=2)
        result = DataStandardizationService._process_msdial(df)
        assert pd.api.types.is_numeric_dtype(result.standardized_df['CalcMass'])


# =============================================================================
# MS-DIAL Sample Override Tests
# =============================================================================


def _make_override_inputs(n_samples=5, n_to_keep=3):
    """Build inputs for apply_msdial_sample_override tests.

    Returns (standardized_df, column_mapping, manual_samples, features).
    """
    original_samples = [f'sample_{i}' for i in range(1, n_samples + 1)]

    # Build a standardized DataFrame (as produced by _process_msdial)
    data = {
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PE'],
    }
    for i in range(1, n_samples + 1):
        data[f'intensity[s{i}]'] = [1000.0 * i, 2000.0 * i]
    df = pd.DataFrame(data)

    # Build column mapping
    mapping_rows = [{'standardized_name': 'LipidMolec', 'original_name': 'Metabolite name'}]
    for i, orig in enumerate(original_samples, 1):
        mapping_rows.append({
            'standardized_name': f'intensity[s{i}]',
            'original_name': orig,
        })
    column_mapping = pd.DataFrame(mapping_rows)

    # Features dict
    features = {
        'raw_sample_columns': original_samples,
        'normalized_sample_columns': original_samples[:3],
    }

    manual_samples = original_samples[:n_to_keep]
    return df, column_mapping, manual_samples, features


class TestApplyMSDIALSampleOverride:
    """Tests for DataStandardizationService.apply_msdial_sample_override."""

    def test_returns_override_result(self):
        df, mapping, manual, features = _make_override_inputs()
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert isinstance(result, MSDIALOverrideResult)

    def test_df_has_correct_intensity_columns(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        intensity_cols = [c for c in result.standardized_df.columns if c.startswith('intensity[')]
        assert intensity_cols == ['intensity[s1]', 'intensity[s2]', 'intensity[s3]']

    def test_df_preserves_metadata_columns(self):
        df, mapping, manual, features = _make_override_inputs()
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert 'LipidMolec' in result.standardized_df.columns
        assert 'ClassKey' in result.standardized_df.columns

    def test_n_intensity_cols_matches(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.n_intensity_cols == 3

    def test_sample_names_mapping(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.sample_names == {
            's1': 'sample_1', 's2': 'sample_2', 's3': 'sample_3'
        }

    def test_raw_sample_columns_updated(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.raw_sample_columns == ['sample_1', 'sample_2', 'sample_3']

    def test_normalized_sample_columns_filtered(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        # features has normalized_sample_columns = ['sample_1', 'sample_2', 'sample_3']
        # manual keeps first 3, so all normalized should be preserved
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.normalized_sample_columns == ['sample_1', 'sample_2', 'sample_3']

    def test_normalized_columns_filtered_to_manual_subset(self):
        """Only normalized samples that are in manual_samples are kept."""
        df, mapping, _, features = _make_override_inputs(n_samples=5, n_to_keep=5)
        features['normalized_sample_columns'] = ['sample_1', 'sample_3', 'sample_5']
        manual = ['sample_1', 'sample_2']  # sample_3 and sample_5 excluded
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.normalized_sample_columns == ['sample_1']

    def test_column_mapping_has_metadata_and_intensity_rows(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        # Should have 1 metadata row (LipidMolec) + 3 intensity rows
        assert len(result.column_mapping) == 4
        std_names = result.column_mapping['standardized_name'].tolist()
        assert std_names[0] == 'LipidMolec'
        assert std_names[1:] == ['intensity[s1]', 'intensity[s2]', 'intensity[s3]']

    def test_column_mapping_original_names_correct(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        orig_names = result.column_mapping['original_name'].tolist()
        assert orig_names == ['Metabolite name', 'sample_1', 'sample_2', 'sample_3']

    def test_data_values_preserved(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=2)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        # intensity[s1] was originally intensity[s1] with values [1000, 2000]
        assert result.standardized_df['intensity[s1]'].tolist() == [1000.0, 2000.0]
        assert result.standardized_df['intensity[s2]'].tolist() == [2000.0, 4000.0]

    def test_single_sample_override(self):
        df, mapping, _, features = _make_override_inputs(n_samples=5, n_to_keep=5)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, ['sample_3'], features
        )
        assert result.n_intensity_cols == 1
        assert result.sample_names == {'s1': 'sample_3'}
        intensity_cols = [c for c in result.standardized_df.columns if c.startswith('intensity[')]
        assert intensity_cols == ['intensity[s1]']

    def test_all_samples_kept(self):
        df, mapping, _, features = _make_override_inputs(n_samples=3, n_to_keep=3)
        manual = ['sample_1', 'sample_2', 'sample_3']
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.n_intensity_cols == 3
        assert len(result.standardized_df.columns) == len(df.columns)

    def test_input_df_not_mutated(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        original_cols = df.columns.tolist()
        DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert df.columns.tolist() == original_cols

    def test_input_mapping_not_mutated(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=3)
        original_len = len(mapping)
        DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert len(mapping) == original_len

    def test_empty_normalized_columns(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=3, n_to_keep=2)
        features['normalized_sample_columns'] = []
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert result.normalized_sample_columns == []

    def test_row_count_preserved(self):
        df, mapping, manual, features = _make_override_inputs(n_samples=5, n_to_keep=2)
        result = DataStandardizationService.apply_msdial_sample_override(
            df, mapping, manual, features
        )
        assert len(result.standardized_df) == len(df)
