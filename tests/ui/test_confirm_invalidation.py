"""Regrouping after an AI Studio round trip un-confirms the inputs (SQ2).

Leaving for AI Studio snapshots the Confirm checkbox
(`_nav_preserved_confirm_checkbox`) so it survives the page switch. While
the user regroups, the checkbox is not rendered and Streamlit drops its key;
restore_navigation_widgets() then put the stale True back from the
snapshot, so the regrouped data came back confirmed without the user
checking the new summary. Nothing tested
_invalidate_confirmation_if_grouping_changed, which drops the snapshot."""
from streamlit.testing.v1 import AppTest

# Pre-import to avoid circular import when AppTest runs the script.
import app.constants  # noqa: F401
import app.services.format_detection  # noqa: F401


def sidebar_after_navigation_script():
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    StreamlitAdapter.restore_navigation_widgets()  # as main_app does
    from tests.ui.conftest import full_sidebar_script
    full_sidebar_script()


def _confirm(at):
    boxes = [c for c in at.sidebar.checkbox if c.key == "confirm_checkbox"]
    return boxes[0].value if boxes else None


def test_regrouping_after_ai_studio_needs_a_new_confirm():
    at = AppTest.from_function(
        sidebar_after_navigation_script, default_timeout=60,
    ).run()
    at.sidebar.button(key="load_sample").click().run()
    assert _confirm(at) is True

    # Back from AI Studio: the snapshot preserve_main_app_widgets() took.
    at.session_state["_nav_preserved_confirm_checkbox"] = True

    at.sidebar.radio(key="grouping_radio").set_value("No").run()
    assert _confirm(at) is None  # not rendered mid-regroup
    at.run()
    at.sidebar.radio(key="grouping_radio").set_value("Yes").run()

    assert not at.exception
    assert _confirm(at) is False
    assert "not_confirmed" in [t.value for t in at.text]
