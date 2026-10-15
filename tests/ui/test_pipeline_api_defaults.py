"""`lc` must run the same statistics the analysis pages run by default (AQ5).

Renders each page's statistical-options panel untouched and compares the
config it returns with the one `lc` uses, so a change to either side fails
here instead of silently splitting the AI's answers from the app's."""
from streamlit.testing.v1 import AppTest

from app.workflows.pipeline_api import CLASS_STATS, VOLCANO_STATS


def options_script(page):
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    if page == "volcano":
        from app.ui.main_content.analysis._volcano import (
            _display_volcano_statistical_options,
        )
        config = _display_volcano_statistical_options()
    else:
        from app.ui.main_content.analysis._shared import (
            _display_statistical_options,
        )
        config = _display_statistical_options(page, 3, 2)
    st.text(f"config:{config.model_dump_json()}")


def _page_default(page):
    at = AppTest.from_function(
        options_script, args=(page,), default_timeout=15,
    ).run()
    assert not at.exception
    marker = next(t.value for t in at.text if t.value.startswith("config:"))
    return marker[len("config:"):]


def test_species_defaults_match_the_volcano_page():
    assert _page_default("volcano") == VOLCANO_STATS.model_dump_json()


def test_class_defaults_match_the_bar_chart_and_saturation_pages():
    assert _page_default("bar") == CLASS_STATS.model_dump_json()
    assert _page_default("sat") == CLASS_STATS.model_dump_json()
