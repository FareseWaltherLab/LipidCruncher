"""Module 1's file uploads: protein amounts as a CSV, and an external
internal-standards table. AppTest cannot upload files; ``fake_uploads``
hands the uploader a file instead."""

import numpy as np

from tests.ui.conftest import FakeUpload

AMOUNTS = [10.0, 20.0, 40.0, 10.0, 20.0, 40.0]


def _csv(text: str) -> FakeUpload:
    return FakeUpload('upload.csv', text.encode())


def _protein_csv(at, fake_uploads, upload):
    at.radio(key='norm_method_selection').set_value('Protein-based').run()
    fake_uploads['Upload CSV'] = upload
    at.radio(key='protein_input_method').set_value('Upload CSV File').run()
    assert not at.exception
    return at


def _errors(at):
    return [e.value for e in at.error]


class TestProteinCsv:

    def test_amounts_divide_their_own_samples(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Sample,Concentration\n' + ''.join(f's{i},{a}\n' for i, a in enumerate(AMOUNTS, 1))
        ))
        assert any('Loaded 6 concentration values' in s.value for s in at.success)
        cleaned = at.session_state['_test_cleaned_df'].set_index('LipidMolec')
        normalized = at.session_state['normalized_df'].set_index('LipidMolec')
        for i, amount in enumerate(AMOUNTS, 1):
            np.testing.assert_allclose(
                normalized[f'concentration[s{i}]'],
                cleaned.loc[normalized.index, f'intensity[s{i}]'] / amount,
            )

    def test_amounts_are_kept_when_the_upload_is_gone(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Concentration\n' + ''.join(f'{a}\n' for a in AMOUNTS)
        ))
        fake_uploads.clear()
        at.run()
        assert not at.exception
        assert any('previously loaded 6' in s.value for s in at.success)

    def test_missing_concentration_column(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv('Amount\n1\n2\n'))
        assert any("named 'Concentration'" in e for e in _errors(at))

    def test_wrong_number_of_rows(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv('Concentration\n1\n2\n'))
        assert any("Row count (2) doesn't match sample count (6)" in e for e in _errors(at))

    def test_values_that_are_not_numbers(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Concentration\n' + 'x\n' * 6
        ))
        assert any("couldn't be converted to numbers" in e for e in _errors(at))

    def test_empty_file(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(''))
        assert any('appears to be empty' in e for e in _errors(at))


def _standards_csv(at, fake_uploads, upload):
    fake_uploads['Upload standards CSV'] = upload
    at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
    assert not at.exception
    return at


_COMPLETE = (
    'LipidMolec,ClassKey,s1,s2,s3,s4,s5,s6\n'
    'PC 15:0_18:1(d7),PC,100,110,120,130,140,150\n'
    'PE 15:0_18:1(d7),PE,200,210,220,230,240,250\n'
)


class TestExternalStandardsCsv:

    def test_complete_table_is_loaded_per_sample(self, intsta_with_standards_app, fake_uploads):
        at = _standards_csv(intsta_with_standards_app, fake_uploads, _csv(_COMPLETE))
        assert any('Loaded 2 custom standards' in s.value for s in at.success)
        standards = at.session_state['custom_standards_df'].set_index('ClassKey')
        assert standards.loc['PC', 'intensity[s1]'] == 100
        assert standards.loc['PE', 'intensity[s6]'] == 250

    def test_a_standard_also_in_the_dataset_is_removed_from_it(
        self, intsta_with_standards_app, fake_uploads,
    ):
        cleaned = intsta_with_standards_app.session_state['_test_cleaned_df']
        in_dataset = cleaned['LipidMolec'].iloc[0]
        at = _standards_csv(intsta_with_standards_app, fake_uploads, _csv(
            f'LipidMolec,ClassKey,s1,s2,s3,s4,s5,s6\n{in_dataset},PC,1,2,3,4,5,6\n'
        ))
        assert any('Removed 1 standard(s) from main dataset' in w.value for w in at.warning)
        assert in_dataset not in set(at.session_state['cleaned_df']['LipidMolec'])

    def test_more_value_columns_than_samples_is_refused(
        self, intsta_with_standards_app, fake_uploads,
    ):
        at = _standards_csv(intsta_with_standards_app, fake_uploads, _csv(
            'LipidMolec,ClassKey,s1,s2,s3,s4,s5,s6,s7\n'
            'PC 15:0_18:1(d7),PC,1,2,3,4,5,6,7\n'
        ))
        assert _errors(at)
        assert at.session_state['custom_standards_df'] is None
