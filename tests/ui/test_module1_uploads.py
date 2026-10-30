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
            'Sample,Protein_ug\n' + ''.join(f's{i},{a}\n' for i, a in enumerate(AMOUNTS, 1))
        ))
        assert any('Loaded 6 protein amounts' in s.value for s in at.success)
        cleaned = at.session_state['_test_cleaned_df'].set_index('LipidMolec')
        normalized = at.session_state['normalized_df'].set_index('LipidMolec')
        for i, amount in enumerate(AMOUNTS, 1):
            np.testing.assert_allclose(
                normalized[f'concentration[s{i}]'],
                cleaned.loc[normalized.index, f'intensity[s{i}]'] / amount,
            )

    def _assert_divided_by(self, at, amount_of):
        cleaned = at.session_state['_test_cleaned_df'].set_index('LipidMolec')
        normalized = at.session_state['normalized_df'].set_index('LipidMolec')
        for sample, amount in amount_of.items():
            np.testing.assert_allclose(
                normalized[f'concentration[{sample}]'],
                cleaned.loc[normalized.index, f'intensity[{sample}]'] / amount,
            )

    def test_rows_follow_their_sample_column(self, norm_no_standards_app, fake_uploads):
        """Regression: rows were taken in order and the Sample column was
        ignored, so a file listing the samples in another order gave each
        sample another sample's amount."""
        amount_of = {f's{i}': a for i, a in enumerate(AMOUNTS, 1)}
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Sample,Protein_ug\n'
            + ''.join(f'{s},{a}\n' for s, a in reversed(amount_of.items()))
        ))
        assert any('matched by the Sample column' in s.value for s in at.success)
        self._assert_divided_by(at, amount_of)

    def test_rows_matched_by_sample_name(self, norm_no_standards_app, fake_uploads):
        at = norm_no_standards_app
        at.session_state['sample_names'] = {f's{i}': f'mouse {i}' for i in range(1, 7)}
        amount_of = {f's{i}': a for i, a in enumerate(AMOUNTS, 1)}
        at = _protein_csv(at, fake_uploads, _csv(
            'Sample,Protein_ug\n'
            + ''.join(f'mouse {s[1:]},{a}\n' for s, a in reversed(amount_of.items()))
        ))
        self._assert_divided_by(at, amount_of)

    def test_sample_column_naming_no_sample_keeps_row_order(
        self, norm_no_standards_app, fake_uploads,
    ):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Sample,Protein_ug\n' + ''.join(f'tube {i},{a}\n' for i, a in enumerate(AMOUNTS, 1))
        ))
        assert any('matched by row order' in s.value for s in at.success)
        self._assert_divided_by(at, {f's{i}': a for i, a in enumerate(AMOUNTS, 1)})

    def test_sample_column_partly_matching_is_refused(
        self, norm_no_standards_app, fake_uploads,
    ):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Sample,Protein_ug\ns1,1\ns2,1\ns3,1\ns4,1\ns5,1\nsx,1\n'
        ))
        assert any("not a sample of this experiment: sx" in e for e in _errors(at))
        assert at.session_state['normalized_df'] is None

    def test_sample_named_twice_is_refused(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Sample,Protein_ug\ns1,1\ns1,1\ns3,1\ns4,1\ns5,1\ns6,1\n'
        ))
        assert any("more than once: s1" in e for e in _errors(at))
        assert at.session_state['normalized_df'] is None

    def test_amounts_are_kept_when_the_upload_is_gone(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Protein_ug\n' + ''.join(f'{a}\n' for a in AMOUNTS)
        ))
        fake_uploads.clear()
        at.run()
        assert not at.exception
        assert any('previously loaded 6' in s.value for s in at.success)

    def test_old_concentration_column_still_read(self, norm_no_standards_app, fake_uploads):
        """Files made before the column was renamed Protein_ug keep working."""
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Concentration\n' + ''.join(f'{a}\n' for a in AMOUNTS)
        ))
        assert any('Loaded 6 protein amounts' in s.value for s in at.success)
        assert at.session_state['normalized_df'] is not None

    def test_missing_protein_column(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv('Amount\n1\n2\n'))
        assert any("named 'Protein_ug'" in e for e in _errors(at))

    def test_wrong_number_of_rows(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv('Protein_ug\n1\n2\n'))
        assert any("Row count (2) doesn't match sample count (6)" in e for e in _errors(at))

    def test_values_that_are_not_numbers(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Protein_ug\n' + 'x\n' * 6
        ))
        assert any("couldn't be converted to numbers" in e for e in _errors(at))

    def test_empty_file(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(''))
        assert any('appears to be empty' in e for e in _errors(at))

    def test_not_utf8(self, norm_no_standards_app, fake_uploads):
        upload = FakeUpload('upload.csv', 'Protein_ug\n'.encode('utf-16'))
        at = _protein_csv(norm_no_standards_app, fake_uploads, upload)
        assert any('save it as a UTF-8 encoded CSV' in e for e in _errors(at))
        assert at.session_state['normalized_df'] is None

    def test_malformed_rows(self, norm_no_standards_app, fake_uploads):
        at = _protein_csv(norm_no_standards_app, fake_uploads, _csv(
            'Protein_ug\n1\n2,3,4\n'
        ))
        assert any('Could not parse the protein concentration CSV' in e for e in _errors(at))
        assert at.session_state['normalized_df'] is None


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

    def test_empty_file(self, intsta_with_standards_app, fake_uploads):
        """Regression: pandas' EmptyDataError is a ValueError, and the
        ValueError handler came first, so this showed 'No columns to parse
        from file' instead of the message written for it."""
        at = _standards_csv(intsta_with_standards_app, fake_uploads, _csv(''))
        assert any('appears to be empty' in e for e in _errors(at))

    def test_clearing_returns_to_automatic_detection(
        self, intsta_with_standards_app, fake_uploads,
    ):
        """Regression: Clear Custom Standards set standards_source back to
        Automatic Detection, but the radio kept its own key, so the page
        stayed on Upload Custom Standards with no standards active."""
        at = _standards_csv(intsta_with_standards_app, fake_uploads, _csv(_COMPLETE))
        fake_uploads.clear()
        at.run()
        at.button(key='clear_custom_standards').click().run()
        assert not at.exception
        assert at.session_state['custom_standards_df'] is None
        assert at.radio(key='standards_source_radio').value == 'Automatic Detection'
        assert at.session_state['standards_source'] == 'Automatic Detection'
