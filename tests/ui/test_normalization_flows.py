"""
Module 1 normalization: settings that cannot run must not leave the last
run's output behind (normalized_df gates Next and feeds Modules 2 and 3),
and settings that can must survive Next / Back.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.constants import (
    FORMAT_LIPIDSEARCH, FORMAT_MSDIAL, MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS,
    PAGE_APP,
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
        assert f'Please enter a valid amount for {STANDARD}' in _errors(at)
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


SPLASH = 'SPLASH LIPIDOMIX (330707)'
# SPLASH's PC 15:0-18:1(d7) is 160 µg/mL, MW 753.11; its PE is 5 µg/mL, 711.03.
SPLASH_PC_PMOL_PER_UL = 160 / 753.11 * 1000
SPLASH_PE_PMOL_PER_UL = 5 / 711.03 * 1000


class TestPremadeStandardMixes:

    @staticmethod
    def _amount(at, standard=STANDARD):
        return at.number_input(key=f'conc_{standard}').value

    @staticmethod
    def _pick_splash(at):
        at.radio(key='norm_method_selection').set_value('Internal Standards').run()
        at.multiselect(key='is_mix_selection').set_value([SPLASH]).run()
        assert not at.exception, at.exception
        return at

    def test_picking_a_mix_fills_its_standards(self, norm_with_standards_app):
        at = self._pick_splash(norm_with_standards_app)
        assert at.number_input(key=f'conc_{STANDARD}').label == f'Amount (pmol) for {STANDARD}'
        assert self._amount(at) == pytest.approx(10 * SPLASH_PC_PMOL_PER_UL)
        config = at.session_state['normalization_result'].config
        assert config.intsta_concentrations[STANDARD] == pytest.approx(
            10 * SPLASH_PC_PMOL_PER_UL
        )

    def test_volume_scales_the_amounts(self, norm_with_standards_app):
        at = self._pick_splash(norm_with_standards_app)
        at.number_input(key=f'is_mix_volume_{SPLASH}').set_value(2.0).run()
        assert self._amount(at) == pytest.approx(2 * SPLASH_PC_PMOL_PER_UL)

    def test_hand_edit_kept_until_the_mixes_change(self, norm_with_standards_app):
        at = self._pick_splash(norm_with_standards_app)
        at.number_input(key=f'conc_{STANDARD}').set_value(3.0).run()
        at.run()
        assert self._amount(at) == 3.0
        at.number_input(key=f'is_mix_volume_{SPLASH}').set_value(5.0).run()
        assert self._amount(at) == pytest.approx(5 * SPLASH_PC_PMOL_PER_UL)

    def test_standard_not_in_mix_left_for_hand_entry(self):
        """The LipidSearch sample's standards are SPLASH's; the PE standard
        is not in SphingoSPLASH and keeps its hand-entered default."""
        at = _module1(FORMAT_LIPIDSEARCH)
        at.radio(key='norm_method_selection').set_value('Internal Standards').run()
        at.multiselect(key='is_mix_selection').set_value(['SphingoSPLASH I (330734)']).run()
        assert not at.exception, at.exception
        pe = at.selectbox(key='standard_selection_PE').value
        assert self._amount(at, pe) == 1.0
        assert any(pe in c.value for c in at.caption if c.value.startswith('Not found'))

    def test_mixes_survive_next_and_back(self):
        at = _module1(FORMAT_LIPIDSEARCH)
        at.multiselect(key='temp_selected_classes').set_value(['PC', 'PE']).run()
        at.radio(key='norm_method_selection').set_value('Internal Standards').run()
        at.multiselect(key='is_mix_selection').set_value([SPLASH]).run()
        at.number_input(key=f'is_mix_volume_{SPLASH}').set_value(5.0).run()
        pe = at.selectbox(key='standard_selection_PE').value
        assert self._amount(at, pe) == pytest.approx(5 * SPLASH_PE_PMOL_PER_UL)

        for module in (MODULE_QC_ANALYSIS, MODULE_DATA_PROCESSING):
            at.session_state['module'] = module
            at.run()
            assert not at.exception, at.exception

        assert at.multiselect(key='is_mix_selection').value == [SPLASH]
        assert at.number_input(key=f'is_mix_volume_{SPLASH}').value == 5.0
        assert self._amount(at, pe) == pytest.approx(5 * SPLASH_PE_PMOL_PER_UL)


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

        # Leave and re-enter Module 1 by setting the module: after Next /
        # Back's st.rerun() AppTest's element tree keeps the previous page's
        # widgets, whose dropped keys the next run cannot read. Both only
        # clear the qc stage.
        for module in (MODULE_QC_ANALYSIS, MODULE_DATA_PROCESSING):
            at.session_state['module'] = module
            at.run()
            assert not at.exception, at.exception

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
