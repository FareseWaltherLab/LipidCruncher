"""UI tests for the sample-count mismatch diagnosis.

Sample columns are detected by content: any numeric column the format does not
publish as metadata is read as a sample. So a user's own numeric annotation
column silently becomes an extra sample, and the only thing that catches it is
the count check against the experiment the user declared.

That check used to report the bare numbers, which reads as "I mis-declared my
experiment" rather than "a column of mine got absorbed". These tests cover the
diagnosis that now accompanies it.
"""
from streamlit.testing.v1 import AppTest

DEFAULT_TIMEOUT = 15


def sample_count_mismatch_script():
    """Render the grouping step with three detected sample columns against an
    experiment that declares two — the shape produced by one stray numeric
    column in the uploaded file."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.models.experiment import ExperimentConfig
    from app.ui.sidebar.sample_grouping import display_group_samples

    df = pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4'],
        'intensity[s1]': [1.0, 2.0],
        'intensity[s2]': [3.0, 4.0],
        'intensity[s3]': [5.0, 6.0],
    })
    # The mapping still holds the names the user actually wrote.
    st.session_state.column_mapping = pd.DataFrame([
        {'standardized_name': 'LipidMolec', 'original_name': 'Metabolite name'},
        {'standardized_name': 'intensity[s1]', 'original_name': 'WT_1'},
        {'standardized_name': 'intensity[s2]', 'original_name': 'WT_2'},
        {'standardized_name': 'intensity[s3]', 'original_name': 'My Custom Score'},
    ])
    experiment = ExperimentConfig(
        conditions_list=['WT'], number_of_samples_list=[2],
    )
    display_group_samples(df, experiment, st.session_state['_test_format'])


def _run(data_format):
    at = AppTest.from_function(
        sample_count_mismatch_script, default_timeout=DEFAULT_TIMEOUT,
    )
    at.session_state['_test_format'] = data_format
    return at.run()


def _captions(at):
    return " ".join(c.value for c in at.caption)


class TestMismatchIsStillReported:
    def test_the_bare_count_error_still_appears(self):
        at = _run('Generic Format')
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "3" in errors and "2" in errors


class TestDetectedColumnsAreNamed:
    """The whole point: let the user see which columns were taken, using the
    names from their file rather than intensity[s3]."""

    def test_original_column_names_are_listed(self):
        at = _run('Generic Format')
        captions = _captions(at)
        assert "My Custom Score" in captions
        assert "WT_1" in captions and "WT_2" in captions

    def test_the_renamed_labels_are_not_what_is_shown(self):
        """`intensity[s3]` means nothing to someone looking at their CSV."""
        at = _run('Generic Format')
        assert "intensity[s3]" not in _captions(at)


class TestFixHintMatchesTheFormat:
    """MS-DIAL has an override widget; the other formats need the file fixed.
    Pointing a Generic user at a widget that isn't rendered for them would be
    worse than saying nothing."""

    def test_msdial_points_at_the_override(self):
        captions = _captions(_run('MS-DIAL'))
        assert "Override Sample Detection" in captions

    def test_generic_points_at_the_file(self):
        captions = _captions(_run('Generic Format'))
        assert "Override Sample Detection" not in captions
        assert "re-upload" in captions


def matching_counts_script():
    """Same shape, but the experiment matches the data — the diagnosis must
    stay out of the way."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()
    from app.models.experiment import ExperimentConfig
    from app.ui.sidebar.sample_grouping import display_group_samples

    df = pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4'],
        'intensity[s1]': [1.0, 2.0],
        'intensity[s2]': [3.0, 4.0],
    })
    st.session_state.column_mapping = pd.DataFrame([
        {'standardized_name': 'intensity[s1]', 'original_name': 'WT_1'},
        {'standardized_name': 'intensity[s2]', 'original_name': 'WT_2'},
    ])
    experiment = ExperimentConfig(
        conditions_list=['WT'], number_of_samples_list=[2],
    )
    display_group_samples(df, experiment, 'Generic Format')


class TestNoDiagnosisWhenCountsAgree:
    def test_valid_dataset_shows_no_mismatch_captions(self):
        at = AppTest.from_function(
            matching_counts_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        assert not at.exception
        captions = " ".join(c.value for c in at.caption)
        assert "Read as samples" not in captions
