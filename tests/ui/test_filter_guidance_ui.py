"""
The filter explanations and the warnings for settings changed from their
defaults. Warnings sit under each filter's panel, outside the expander, so
they are seen while it is collapsed.
"""
import dataclasses

from streamlit.testing.v1 import AppTest

from tests.ui.conftest import DEFAULT_TIMEOUT, make_grade_dataframe


def _texts(elements):
    return [e.value for e in elements]


def _outside_expanders(at, kind):
    """Warnings or infos of the page that no expander holds."""
    inside = {e.value for x in at.expander for e in getattr(x, kind)}
    return [v for v in _texts(getattr(at, kind)) if v not in inside]


def filter_configs_script():
    """build_filter_configs for the format in ``_test_format``."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.ui.main_content.data_processing import build_filter_configs

    build_filter_configs(st.session_state['_test_format'], st.session_state.get('_test_df'))


class TestZeroFilter:

    def test_explains_the_rule_for_this_design(self, zero_filter_generic_app):
        at = zero_filter_generic_app
        explanation = " ".join(_texts(at.expander[0].markdown))
        assert "**at least 1 of 3 replicates in any one condition**" in explanation
        assert "Low-abundance lipids" in explanation
        assert "With the current settings (non-BQC 75%) and 3 samples per condition" in explanation
        assert "**Raising a threshold is looser**" in explanation

    def test_sliders_say_what_they_remove(self, zero_filter_bqc_app):
        at = zero_filter_bqc_app
        assert at.slider(key='non_bqc_zero_threshold').label == (
            "Non-BQC threshold: remove if missing in at least this % of samples, "
            "in every condition"
        )
        assert at.slider(key='bqc_zero_threshold').label == (
            "BQC threshold: remove if missing in at least this % of BQC samples"
        )
        assert "Higher is looser" in at.slider(key='non_bqc_zero_threshold').help

    def test_single_detection_warning_is_visible_while_collapsed(self, zero_filter_generic_app):
        """The default is kept; the user asked for a clear warning instead."""
        warnings = _outside_expanders(zero_filter_generic_app, 'warning')
        assert any(w.startswith("**A single detection keeps a lipid.**") for w in warnings)

    def test_lowered_non_bqc_threshold(self, zero_filter_generic_app):
        at = zero_filter_generic_app
        at.slider(key='non_bqc_zero_threshold').set_value(50).run()
        assert not any("single detection" in w for w in _texts(at.warning))
        infos = _outside_expanders(at, 'info')
        assert any(i.startswith("**Non-BQC zero threshold: 50% (default 75%).**") for i in infos)
        assert "**at least 2 of 3 replicates in any one condition**" in " ".join(
            _texts(at.expander[0].markdown)
        )

    def test_raised_non_bqc_threshold_warns(self, zero_filter_generic_app):
        at = zero_filter_generic_app
        at.slider(key='non_bqc_zero_threshold').set_value(90).run()
        warnings = _outside_expanders(at, 'warning')
        assert any(w.startswith("**Non-BQC zero threshold: 90% (default 75%).**") for w in warnings)

    def test_lowered_detection_threshold_warns(self, zero_filter_lipidsearch_app):
        at = zero_filter_lipidsearch_app
        assert not any("Detection threshold" in w for w in _texts(at.warning))
        at.number_input(key='zero_filter_detection_threshold').set_value(10000.0).run()
        warnings = _outside_expanders(at, 'warning')
        assert any(w.startswith("**Detection threshold: 10,000 (default 30,000).**")
                   for w in warnings)

    def test_raised_bqc_threshold_warns(self, zero_filter_bqc_app):
        at = zero_filter_bqc_app
        at.slider(key='bqc_zero_threshold').set_value(80).run()
        warnings = _outside_expanders(at, 'warning')
        assert any(w.startswith("**BQC zero threshold: 80% (default 50%).**") for w in warnings)


class TestGradeFilter:

    def _app(self):
        at = AppTest.from_function(filter_configs_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_format'] = 'LipidSearch'
        at.session_state['_test_df'] = make_grade_dataframe()
        return at.run()

    def test_explains_what_a_grade_means(self):
        at = self._app()
        assert any("**identified correctly**" in i for i in _texts(at.expander[0].info))

    def test_default_has_no_warning(self):
        assert not any("Grades" in w for w in _texts(self._app().warning))

    def test_allow_c_for_all_classes_warns_outside_the_panel(self):
        at = self._app()
        at.radio(key='grade_filter_mode_radio').set_value('Allow C for all classes').run()
        assert not at.exception, at.exception
        warnings = _outside_expanders(at, 'warning')
        assert any(w.startswith("**Grades allowed: C for PC, PE") and "spectra" in w
                   for w in warnings)


class TestMsdialQuality:

    def _app(self, features):
        at = AppTest.from_function(filter_configs_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_format'] = 'MS-DIAL'
        at.session_state['msdial_features'] = features
        return at.run()

    def test_permissive_preset_warns(self, msdial_features_dict):
        at = self._app(msdial_features_dict)
        assert not any("MS-DIAL score" in w for w in _texts(at.warning))
        at.radio(key='msdial_quality_level_radio').set_value('Permissive (Score ≥40)').run()
        assert not at.exception, at.exception
        warnings = _outside_expanders(at, 'warning')
        assert any(w.startswith("**MS-DIAL score threshold: 40 (default 60).**") for w in warnings)

    def test_strict_preset_is_a_note(self, msdial_features_dict):
        at = self._app(dataclasses.replace(msdial_features_dict, has_msms_matched=True))
        at.radio(key='msdial_quality_level_radio').set_value(
            'Strict (Score ≥80, MS/MS required)'
        ).run()
        infos = _outside_expanders(at, 'info')
        assert any(i.startswith("**MS-DIAL score threshold: 80 (default 60).**") for i in infos)
        assert any(i.startswith("**MS/MS match required") for i in infos)


class TestBqcCov:

    def test_explains_cov_as_a_rule_of_thumb(self, qc_bqc_app):
        text = " ".join(_texts(qc_bqc_app.markdown))
        assert "30% is a common rule of thumb" in text

    def test_raised_threshold_warns(self, qc_bqc_app):
        at = qc_bqc_app
        assert not any("CoV threshold" in w for w in _texts(at.warning))
        at.number_input(key='bqc_cov_threshold').set_value(40).run()
        assert any(w.startswith("**BQC CoV threshold: 40% (default 30%).**")
                   for w in _texts(at.warning))

    def test_lipids_kept_despite_high_cov_warn(self, qc_bqc_app):
        at = qc_bqc_app
        at.radio(key='bqc_filter_choice').set_value('Yes').run()
        keep = at.multiselect(key='bqc_lipids_to_keep')
        keep.set_value([o.rsplit(' (CoV:', 1)[0] for o in keep.options[:2]]).run()
        assert not at.exception, at.exception
        assert any(w.startswith("**2 lipids kept despite CoV ≥ 30%.**")
                   for w in _texts(at.warning))
