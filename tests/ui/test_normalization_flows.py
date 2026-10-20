"""
Module 1 normalization: settings that cannot run must not leave the last
run's output behind (normalized_df gates Next and feeds Modules 2 and 3),
and settings that can must survive Next / Back.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.constants import (
    FORMAT_LIPIDSEARCH, FORMAT_MSDIAL, MODULE_DATA_PROCESSING, PAGE_APP,
)

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')
STANDARD = 'PC(15:0_18:1)+D7:(s)'


def _errors(at):
    assert not at.exception, at.exception
    return [e.value for e in at.error]


def _module1(data_format):
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.selectbox[0].set_value(data_format).run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    assert not at.exception, at.exception
    return at


def _has_next(at):
    return any(b.label.startswith('Next') for b in at.button)


class TestInvalidSettingsClearOutput:

    def test_no_classes_selected(self, norm_no_standards_app):
        at = norm_no_standards_app
        assert at.session_state['normalized_df'] is not None
        at.multiselect(key='temp_selected_classes').set_value([]).run()
        assert not at.exception
        assert 'Please select at least one lipid class.' in [w.value for w in at.warning]
        assert at.session_state['normalized_df'] is None

    def test_standard_concentration_not_positive(self, norm_with_standards_app):
        at = norm_with_standards_app
        at.radio(key='norm_method_selection').set_value('Internal Standards').run()
        assert at.session_state['normalized_df'] is not None
        at.number_input(key=f'conc_{STANDARD}').set_value(0.0).run()
        assert f'Please enter a valid concentration for {STANDARD}' in _errors(at)
        assert at.session_state['normalized_df'] is None

    def test_protein_amount_not_positive(self, norm_no_standards_app):
        at = norm_no_standards_app
        at.radio(key='norm_method_selection').set_value('Protein-based').run()
        assert at.session_state['normalized_df'] is not None
        at.number_input(key='protein_s2').set_value(0.0).run()
        assert any(
            'must be greater than zero' in e and e.endswith(': s2') for e in _errors(at)
        )
        assert at.session_state['normalized_df'] is None

    def test_failed_normalization(self):
        """The LipidSearch sample's PG standard is zero in the BQC samples,
        so normalizing PG by it fails: the earlier run's output must go."""
        at = _module1(FORMAT_LIPIDSEARCH)
        assert at.session_state['normalized_df'] is not None
        assert _has_next(at)
        at.radio(key='norm_method_selection').set_value('Internal Standards').run()
        assert any('has zero intensity in sample(s) s9, s10, s11, s12' in e for e in _errors(at))
        assert at.session_state['normalized_df'] is None
        assert at.session_state['normalization_result'] is None
        assert not _has_next(at)


class TestStandardMappingSurvivesNextAndBack:

    def test_custom_standard_kept(self):
        at = _module1(FORMAT_LIPIDSEARCH)
        at.multiselect(key='temp_selected_classes').set_value(
            [c for c in at.multiselect(key='temp_selected_classes').options if c != 'PG']
        ).run()
        at.radio(key='norm_method_selection').set_value('Internal Standards').run()
        # AcCa has no standard of its own and defaults to the first one.
        # (Not the PG standard: it is zero in the BQC samples.)
        acca = at.selectbox(key='standard_selection_AcCa')
        other = [o for o in acca.options if o != acca.value and not o.startswith('PG')][0]
        acca.set_value(other).run()
        assert not at.exception, at.exception

        [b for b in at.button if b.label.startswith('Next')][0].click().run()
        assert not at.exception, at.exception
        # Clicking Back trips an AppTest artifact (the element tree keeps
        # Module 2's widgets after st.rerun()); Back only clears the qc stage.
        at.session_state['module'] = MODULE_DATA_PROCESSING
        at.run()

        assert at.radio(key='norm_method_selection').value == 'Internal Standards'
        assert at.selectbox(key='standard_selection_AcCa').value == other
        assert f'No specific standards available for AcCa. Defaulting to {other}.' in [
            w.value for w in at.warning
        ]
        assert at.session_state['normalization_result'].config.internal_standards['AcCa'] == other


class TestMsdialPrenormalized:

    def test_standards_options_withdrawn(self):
        at = _module1(FORMAT_MSDIAL)
        at.radio(key='msdial_data_type_radio').set_value(
            [o for o in at.radio(key='msdial_data_type_radio').options if 'Pre-normalized' in o][0]
        ).run()
        assert not at.exception, at.exception
        assert at.radio(key='norm_method_selection').options == [
            'None (pre-normalized data)', 'Protein-based', 'Total Intensity',
        ]
        assert any(
            'using pre-normalized MS-DIAL data' in m.value for m in at.markdown
        )
