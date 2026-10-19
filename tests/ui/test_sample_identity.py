"""Pins: every per-sample value follows its sample through renumbering.

Three events renumber samples: a manual regroup in the sidebar, the sidebar's
sample exclusion, and Quality Check's PCA exclusion. Each test drives the
real app on the Generic sample dataset (WT s1-s4, ADGAT-DKO s5-s8, BQC
s9-s12) with sample k named ``Nk``, applies events, and checks what users
get: the CSV downloads (each name over its own sample's data, each name in
the condition the user gave that sample) and the per-sample inputs
(protein amounts, exclusions).
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.constants import PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')

UPLOAD_CONDITIONS = {
    **{f's{i}': 'WT' for i in range(1, 5)},
    **{f's{i}': 'ADGAT-DKO' for i in range(5, 9)},
    **{f's{i}': 'BQC' for i in range(9, 13)},
}
CONDITIONS = ['WT', 'ADGAT-DKO', 'BQC']
# Swap s1 and s5 between WT and ADGAT-DKO (labels as offered: upload labels).
SWAP = {
    'WT': ['s5', 's2', 's3', 's4'],
    'ADGAT-DKO': ['s1', 's6', 's7', 's8'],
    'BQC': ['s9', 's10', 's11', 's12'],
}


class Scenario:
    """The app plus what the test expects of each uploaded sample."""

    def __init__(self, captured):
        self.captured = captured
        self.at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
        self.at.session_state['page'] = PAGE_APP
        self.at.run()
        self.at.sidebar.button(key='load_sample').click().run()
        self.at.sidebar.checkbox(key='confirm_checkbox').check().run()
        self.at.session_state['sample_names'] = {
            f's{i}': f'N{i}' for i in range(1, 13)
        }
        self.at.run()
        self._ok()
        self.upload = self.at.session_state['standardized_df'].set_index('LipidMolec')
        self.condition = dict(UPLOAD_CONDITIONS)  # upload label -> condition
        self.excluded = set()  # upload labels

    def _ok(self):
        assert not self.at.exception, self.at.exception

    def _confirm(self):
        checkbox = self.at.sidebar.checkbox(key='confirm_checkbox')
        if not checkbox.value:
            checkbox.check().run()
        self._ok()

    def origin_of(self, column, frame):
        """The upload label whose intensities ``frame[column]`` holds."""
        frame = frame.set_index('LipidMolec') if 'LipidMolec' in frame else frame
        common = frame.index.intersection(self.upload.index)
        values = pd.to_numeric(frame.loc[common, column]).to_numpy(float)
        matches = [
            f's{i}' for i in range(1, 13)
            if np.allclose(
                values,
                pd.to_numeric(self.upload.loc[common, f'intensity[s{i}]']).to_numpy(float),
            )
        ]
        assert len(matches) == 1, (column, matches)
        return matches[0]

    def sidebar_label(self, upload_label):
        """The label the sidebar shows today for an uploaded sample."""
        df = self.at.session_state['standardized_df']
        for column in df.columns:
            if column.startswith('intensity[') and self.origin_of(column, df) == upload_label:
                return column[len('intensity['):-1]
        raise AssertionError(upload_label)

    # --- events -----------------------------------------------------------

    def regroup(self, selections=SWAP):
        self.at.sidebar.radio(key='grouping_radio').set_value('No').run()
        for condition, samples in selections.items():
            self.at.sidebar.multiselect(key=f'select_{condition}').set_value(samples).run()
        self._confirm()
        for condition, samples in selections.items():
            for s in samples:
                self.condition[s] = condition

    def undo_regroup(self):
        self.at.sidebar.radio(key='grouping_radio').set_value('Yes').run()
        self._confirm()
        self.condition = dict(UPLOAD_CONDITIONS)

    def exclude_in_sidebar(self, *upload_labels):
        """Exclude uploaded samples, picked under their current sidebar labels."""
        picks = [self.sidebar_label(s) for s in upload_labels]
        self.at.sidebar.multiselect(key='exclude_samples').set_value(picks).run()
        self._confirm()
        self.excluded = set(upload_labels)

    def next(self):
        [b for b in self.at.button if b.label.startswith('Next')][0].click().run()
        self._ok()

    def exclude_in_pca(self, *upload_labels):
        """Exclude uploaded samples in Quality Check, picked under their
        Module 1 labels (read off the data QC received)."""
        df = self.at.session_state['normalized_df']
        label_of = {
            self.origin_of(c, df): c[len('concentration['):-1]
            for c in df.columns if c.startswith('concentration[')
        }
        self.at.multiselect(key='pca_samples_remove').set_value(
            [label_of[s] for s in upload_labels]
        ).run()
        self._ok()
        self.excluded |= set(upload_labels)

    # --- checks -----------------------------------------------------------

    def kept(self):
        return [f's{i}' for i in range(1, 13) if f's{i}' not in self.excluded]

    def csv(self, key):
        assert key in self.captured, f"no download rendered for {key}"
        return pd.read_csv(io.BytesIO(self.captured[key]))

    def assert_module1_csv_follows(self):
        """Normalized data CSV: each kept sample's own data under its name."""
        out = self.csv('download_normalized_data')
        columns = [c for c in out.columns if c.startswith('concentration[')]
        assert sorted(columns) == sorted(f'concentration[N{s[1:]}]' for s in self.kept())
        for column in columns:
            name = column[len('concentration['):-1]
            assert self.origin_of(column, out) == f's{name[1:]}', column

    def assert_module1_conditions_follow(self):
        """The experiment Module 1 runs on groups each sample where the user put it."""
        experiment = self.at.session_state['experiment']
        df = self.at.session_state['normalized_df']  # no normalization: raw values
        for condition, samples in zip(
            experiment.conditions_list, experiment.individual_samples_list,
        ):
            for label in samples:
                origin = self.origin_of(f'concentration[{label}]', df)
                assert self.condition[origin] == condition, (label, origin)

    def assert_pca_csv_follows(self):
        """PCA CSV: each kept sample's name, in the condition it was given."""
        out = self.csv('pca_csv_download')
        expected = {f'N{s[1:]}': self.condition[s] for s in self.kept()}
        assert dict(zip(out['Sample'], out['Condition'])) == expected

    def assert_qc_output_follows(self):
        """The data analyses receive holds exactly the kept samples, in
        condition order, each name over its own sample's data."""
        df = self.at.session_state['qc_continuation_df']
        columns = [c for c in df.columns if c.startswith('concentration[')]
        origins = [self.origin_of(c, df) for c in columns]
        assert sorted(origins) == sorted(self.kept())
        pca = self.csv('pca_csv_download')
        assert pca['Sample'].tolist() == [f'N{o[1:]}' for o in origins]


def _scenario(captured_downloads):
    return Scenario(captured_downloads)


class TestModule1:
    """What Module 1 runs on and exports, after sidebar renumbering."""

    def test_regroup(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.regroup()
        sc.assert_module1_conditions_follow()
        sc.assert_module1_csv_follows()

    def test_sidebar_exclusion(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.exclude_in_sidebar('s2')
        sc.assert_module1_conditions_follow()
        sc.assert_module1_csv_follows()

    def test_exclusion_then_regroup(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.exclude_in_sidebar('s1')  # the regroup moves s1 to ADGAT-DKO
        sc.regroup()
        sc.assert_module1_conditions_follow()
        sc.assert_module1_csv_follows()

    def test_regroup_then_exclusion(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.regroup()
        sc.exclude_in_sidebar('s5')  # now in WT
        sc.assert_module1_conditions_follow()
        sc.assert_module1_csv_follows()

    def test_regroup_exclusion_then_undo_regroup(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.regroup()
        sc.exclude_in_sidebar('s5')
        sc.undo_regroup()
        assert sc.at.sidebar.multiselect(key='exclude_samples').value == ['s5']
        sc.assert_module1_conditions_follow()
        sc.assert_module1_csv_follows()


class TestQualityCheck:
    """What Quality Check and the analyses get, after every event."""

    def test_no_renumbering(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.next()
        sc.assert_pca_csv_follows()
        sc.assert_qc_output_follows()

    def test_pca_exclusion(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.next()
        sc.exclude_in_pca('s2')
        sc.assert_pca_csv_follows()
        sc.assert_qc_output_follows()

    def test_regroup_and_sidebar_exclusion_then_pca_exclusion(self, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.regroup()
        sc.exclude_in_sidebar('s2')
        sc.next()
        sc.assert_pca_csv_follows()
        sc.exclude_in_pca('s5')
        sc.assert_pca_csv_follows()
        sc.assert_qc_output_follows()

    def test_sidebar_change_on_qc_page_does_not_move_samples(self, captured_downloads):
        """The sidebar keeps running on Quality Check; its renumbering must
        not reach the data QC already holds."""
        sc = _scenario(captured_downloads)
        sc.exclude_in_sidebar('s2')
        sc.next()
        expected = sc.csv('pca_csv_download')
        sc.at.sidebar.radio(key='grouping_radio').set_value('No').run()
        for condition, samples in SWAP.items():
            sc.at.sidebar.multiselect(key=f'select_{condition}').set_value(samples).run()
        sc._confirm()
        pd.testing.assert_frame_equal(sc.csv('pca_csv_download'), expected)


class TestProteinAmounts:
    """Combinations not covered by test_sidebar_state.py."""

    @staticmethod
    def _enter(sc):
        sc.at.radio(key='norm_method_selection').set_value('Protein-based').run()
        for i in range(1, 13):
            sc.at.number_input(key=f'protein_s{i}').set_value(float(i * 10))
        sc.at.run()
        sc._ok()

    @staticmethod
    def _assert_amounts_follow(sc):
        """Each normalized column is one kept sample's intensities divided by
        that same sample's amount (sample k was given 10k)."""
        df = sc.at.session_state['normalized_df'].set_index('LipidMolec')
        common = df.index.intersection(sc.upload.index)
        found = []
        for column in [c for c in df.columns if c.startswith('concentration[')]:
            values = df.loc[common, column].astype(float).to_numpy()
            found += [
                f's{k}' for k in range(1, 13)
                if np.allclose(
                    values * k * 10,
                    sc.upload.loc[common, f'intensity[s{k}]'].astype(float).to_numpy(),
                )
            ]
        assert sorted(found) == sorted(sc.kept())

    def test_exclusion_then_regroup(self, captured_downloads):
        sc = _scenario(captured_downloads)
        self._enter(sc)
        sc.exclude_in_sidebar('s1')
        sc.regroup()
        self._assert_amounts_follow(sc)

    def test_regroup_then_exclusion(self, captured_downloads):
        sc = _scenario(captured_downloads)
        self._enter(sc)
        sc.regroup()
        sc.exclude_in_sidebar('s1')  # now in ADGAT-DKO, labelled s5
        self._assert_amounts_follow(sc)


class TestNameEdits:
    """A name typed into the sidebar's Sample Names editor stays with its
    sample. AppTest cannot type into a data_editor, so the editor is patched
    to apply one edit, as a user's typing would."""

    @staticmethod
    def _type_name(monkeypatch, sc, label, name):
        import streamlit

        original = streamlit.data_editor
        pending = [(label, name)]

        def _editor(data, *args, key=None, **kwargs):
            if key == 'sample_names_editor' and pending:
                edited_label, edited_name = pending.pop()
                data = data.copy()
                data.loc[data['sample'] == edited_label, 'name'] = edited_name
            return original(data, *args, key=key, **kwargs)

        monkeypatch.setattr(streamlit, 'data_editor', _editor)
        sc.at.run()
        sc._ok()

    def test_edit_during_a_regroup_survives_undoing_it(self, monkeypatch, captured_downloads):
        """Regression: names were rebuilt from the pre-regroup snapshot on
        every run, so a name typed while regrouped was lost on undo."""
        sc = _scenario(captured_downloads)
        sc.regroup()  # upload s5 is now sidebar s1
        self._type_name(monkeypatch, sc, 's1', 'Edited')
        assert sc.at.session_state['sample_names']['s1'] == 'Edited'

        sc.undo_regroup()
        assert sc.at.session_state['sample_names']['s5'] == 'Edited'
        assert sc.at.session_state['sample_names']['s1'] == 'N1'

    def test_edit_during_a_regroup_survives_changing_it(self, monkeypatch, captured_downloads):
        sc = _scenario(captured_downloads)
        sc.regroup()
        self._type_name(monkeypatch, sc, 's1', 'Edited')  # upload s5

        sc.regroup({
            'WT': ['s1', 's2', 's3', 's4'],
            'ADGAT-DKO': ['s5', 's6', 's7', 's8'],
            'BQC': ['s9', 's10', 's11', 's12'],
        })
        assert sc.at.session_state['sample_names']['s5'] == 'Edited'
        sc.next()
        assert 'Edited' in set(sc.csv('pca_csv_download')['Sample'])
