"""The BQC zero-filter threshold survives Next / Back.

Regression: Next and Back to Data Processing reset every session field
starting with `_preserved_bqc_`, meant for the QC page's BQC filter choice,
which also matched Module 1's `_preserved_bqc_zero_threshold`. A custom BQC
threshold reverted to the default on returning to Module 1 (changing the
zero filter's result) and was missing from the LSI report on Module 2/3."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.constants import PAGE_APP

MAIN_APP = Path(__file__).parents[2] / "src" / "main_app.py"


def _next(at):
    [b for b in at.button
     if b.label.startswith("Next: Quality Check")][0].click().run()


def test_custom_bqc_threshold_survives_next_and_back():
    at = AppTest.from_file(str(MAIN_APP), default_timeout=120)
    at.session_state["page"] = PAGE_APP
    at.run()
    at.sidebar.button(key="load_sample").click().run()
    at.sidebar.checkbox(key="confirm_checkbox").check().run()
    at.slider(key="bqc_zero_threshold").set_value(75).run()

    _next(at)
    assert at.session_state["_preserved_bqc_zero_threshold"] == 75

    [b for b in at.button if b.key == "back_processing_module2"][0].click().run()
    assert not at.exception
    assert at.slider(key="bqc_zero_threshold").value == 75
