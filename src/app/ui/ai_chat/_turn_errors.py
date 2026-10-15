"""
Each tab's last failed turn, shown with a one-click Retry until a turn
succeeds.

Every tab keeps its own keys. ``st.tabs`` runs every tab body on every
rerun, so with one shared key whichever tab rendered first showed (and
consumed) another tab's error, and a Retry there sent that tab's prompt to
the wrong chat.
"""
import streamlit as st


def record_failure(tab: str, user_input: str, error: str) -> None:
    st.session_state[f'chat_{tab}_last_error'] = error
    st.session_state[f'chat_{tab}_failed_input'] = user_input


def clear_failure(tab: str) -> None:
    st.session_state.pop(f'chat_{tab}_last_error', None)
    st.session_state.pop(f'chat_{tab}_failed_input', None)


def render_failure(tab: str, pending_prompt_key: str) -> None:
    """Show the tab's last error and a Retry button that resends its input.

    The error is read, not popped: the button has to exist on the rerun its
    own click triggers, or the click is lost.
    """
    error = st.session_state.get(f'chat_{tab}_last_error')
    if not error:
        return
    st.error(f"An error occurred: {error}")
    if st.button("Retry", key=f"chat_retry_{tab}"):
        st.session_state[pending_prompt_key] = st.session_state.get(
            f'chat_{tab}_failed_input'
        )
        clear_failure(tab)
        st.rerun()
