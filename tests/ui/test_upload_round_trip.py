"""The main-app uploader and the AI Studio round-trip (SQ1).

The uploader comes back empty after the user visits AI Studio, so the
stored raw_df stands in for it then. Regression: it stood in always, so
clearing the uploader (its ×) on the main page left the old dataset loaded
while the uploader showed nothing."""
import pandas as pd
from streamlit.testing.v1 import AppTest

# Pre-import to avoid circular import when AppTest runs the script.
import app.constants  # noqa: F401
import app.services.format_detection  # noqa: F401


def upload_script(kept):
    import pandas as pd
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.constants import FORMAT_GENERIC
    from app.ui.sidebar.file_upload import display_file_upload
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('raw_df') is None:
        st.session_state.raw_df = pd.DataFrame({'LipidMolec': ['PC 34:1']})
        st.session_state.upload_kept_across_pages = kept
    df = display_file_upload(FORMAT_GENERIC)
    st.text(f"returned:{df is not None}")


def _returned(kept):
    at = AppTest.from_function(
        upload_script, args=(kept,), default_timeout=15,
    ).run()
    assert not at.exception
    marker = next(t.value for t in at.text if t.value.startswith("returned:"))
    return marker.endswith("True"), at


def test_emptied_uploader_drops_the_dataset():
    returned, _ = _returned(kept=False)
    assert returned is False


def test_after_ai_studio_the_stored_upload_is_used_and_said_so():
    returned, at = _returned(kept=True)
    assert returned is True
    assert any("before opening AI Studio" in i.value for i in at.sidebar.info)


def test_leaving_for_ai_studio_keeps_the_upload(monkeypatch):
    from app.adapters import streamlit_adapter as adapter_module
    from app.adapters.streamlit_adapter import StreamlitAdapter

    state = {'raw_df': pd.DataFrame({'a': [1]})}
    monkeypatch.setattr(adapter_module.st, "session_state", state)
    StreamlitAdapter.preserve_main_app_widgets()
    assert state['upload_kept_across_pages'] is True
