"""
UI tests for what CSV downloads contain.

Covers:
1. Heatmap CSVs keep their row labels (lipid class / species, plus the
   cluster for the clustered mode, in the figure's row order).
2. User sample names replace the s-labels in per-sample CSV headers.
3. After Quality Check excludes samples (which renumbers the survivors), the
   downstream CSVs carry each survivor's own name, not its new label's.
"""

import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    analysis_module_script,
    make_analysis_dataframe,
    make_cleaned_dataframe,
    normalization_script,
    qc_and_analysis_script,
    qc_module_script,
)


def _csv(captured, key) -> pd.DataFrame:
    assert key in captured, f"no download rendered for {key}"
    return pd.read_csv(io.BytesIO(captured[key]))


def _experiment():
    from app.models.experiment import ExperimentConfig
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )


def _select(at, key, word):
    option = [o for o in at.radio(key=key).options if word in o][0]
    at.radio(key=key).set_value(option).run()
    assert not at.exception
    return at


NAMES = {'s1': 'A', 's2': 'B', 's3': 'C', 's4': 'D', 's5': 'E', 's6': 'F'}


# =============================================================================
# 1. Heatmap CSV row labels
# =============================================================================

class TestHeatmapCsvRowLabels:
    """Regression: the heatmap CSV held only concentration[...] columns, so
    nobody could tell which row was which lipid class or species."""

    @pytest.fixture
    def heatmap_app(self, captured_downloads):
        at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
        at.session_state['_test_experiment'] = _experiment()
        at.run()
        return _select(at, 'analysis_radio', 'Lipidomic Heatmap')

    def test_aggregated_has_class_column(self, heatmap_app, captured_downloads):
        _select(heatmap_app, 'heatmap_type', 'Aggregated')
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        assert out.columns[0] == 'ClassKey'
        assert out['ClassKey'].tolist() == ['PC', 'PE']

    @pytest.mark.parametrize('mode', ['Regular', 'Grouped by Class'])
    def test_species_modes_have_lipid_and_class(
        self, heatmap_app, captured_downloads, mode,
    ):
        _select(heatmap_app, 'heatmap_type', mode)
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        assert list(out.columns[:2]) == ['LipidMolec', 'ClassKey']
        df = heatmap_app.session_state['_test_df']
        assert sorted(out['LipidMolec']) == sorted(df['LipidMolec'])

    def test_clustered_has_cluster_in_figure_order(
        self, heatmap_app, captured_downloads,
    ):
        _select(heatmap_app, 'heatmap_type', 'Clustered')
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        assert list(out.columns[:3]) == ['LipidMolec', 'ClassKey', 'Cluster']
        y = heatmap_app.session_state['analysis_heatmap_fig'].data[0].y
        species = list(y[-1]) if isinstance(y[0], (list, tuple)) else list(y)
        assert out['LipidMolec'].tolist() == species

    def test_cluster_composition_keeps_cluster_column(
        self, heatmap_app, captured_downloads,
    ):
        _select(heatmap_app, 'heatmap_type', 'Clustered')
        out = _csv(captured_downloads, 'analysis_csv_cluster')
        assert out.columns[0] == 'Cluster'


# =============================================================================
# 2. Sample names in per-sample CSV headers
# =============================================================================

class TestSampleNamesInCsv:
    """Named samples are renamed; unnamed ones keep their label."""

    def test_normalized_data_csv(self, captured_downloads):
        at = AppTest.from_function(normalization_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_cleaned_df'] = make_cleaned_dataframe(n_lipids=20, n_samples=6)
        at.session_state['_test_intsta_df'] = None
        at.session_state['_test_experiment'] = _experiment()
        at.session_state['sample_names'] = {'s1': 'ID_01', 's2': '  '}
        at.run()
        assert not at.exception
        out = _csv(captured_downloads, 'download_normalized_data')
        assert 'concentration[ID_01]' in out.columns
        assert 'concentration[s1]' not in out.columns
        # Blank name and unnamed samples keep the label.
        assert 'concentration[s2]' in out.columns
        assert 'concentration[s6]' in out.columns

    def test_correlation_csv_keeps_and_names_row_labels(self, captured_downloads):
        """Regression: the correlation matrix's row labels (its index) were
        dropped, leaving a square of numbers with no row names."""
        at = AppTest.from_function(qc_module_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
        at.session_state['_test_experiment'] = _experiment()
        at.session_state['sample_names'] = {'s1': 'A'}
        at.run()
        assert not at.exception
        out = _csv(captured_downloads, 'corr_csv_download')
        assert list(out.columns) == ['Sample', 'A', 's2', 's3']
        assert out['Sample'].tolist() == ['A', 's2', 's3']


# =============================================================================
# 3. Names after sample exclusion
# =============================================================================

class TestNamesAfterExclusion:
    """Excluding s2 renumbers old s3 to s2; downstream CSVs must call it C."""

    @pytest.fixture
    def app(self, captured_downloads):
        at = AppTest.from_function(qc_and_analysis_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
        at.session_state['_test_experiment'] = _experiment()
        at.session_state['sample_names'] = dict(NAMES)
        at.run()
        assert not at.exception
        return at

    def _exclude(self, at, samples):
        at.multiselect(key='pca_samples_remove').set_value(samples).run()
        assert not at.exception
        return at

    def test_heatmap_csv_names_follow_the_survivors(self, app, captured_downloads):
        self._exclude(app, ['s2'])
        _select(app, 'analysis_radio', 'Lipidomic Heatmap')
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        sample_cols = [c for c in out.columns if c.startswith('concentration[')]
        assert sample_cols == [
            'concentration[A]', 'concentration[C]', 'concentration[D]',
            'concentration[E]', 'concentration[F]',
        ]

    def test_values_under_a_name_belong_to_that_sample(self, app, captured_downloads):
        self._exclude(app, ['s2'])
        _select(app, 'analysis_radio', 'Volcano')
        out = _csv(captured_downloads, 'analysis_csv_dist')
        original = app.session_state['_test_df'].set_index('LipidMolec')
        rows_c = out[out['Sample'] == 'C']
        assert not rows_c.empty
        for _, row in rows_c.iterrows():
            assert row['Concentration'] == pytest.approx(
                original.loc[row['LipidMolec'], 'concentration[s3]']
            )
        assert 'B' not in set(out['Sample'])

    def test_pca_csv_names_follow_the_survivors(self, app, captured_downloads):
        self._exclude(app, ['s2'])
        out = _csv(captured_downloads, 'pca_csv_download')
        assert out['Sample'].tolist() == ['A', 'C', 'D', 'E', 'F']

    def test_pre_exclusion_csv_keeps_original_names(self, app, captured_downloads):
        """The box plot CSV is built before the exclusion, so B is still s2."""
        self._exclude(app, ['s2'])
        out = _csv(captured_downloads, 'qc_box_plot_csv')
        original = app.session_state['_test_df']
        assert list(out.columns) == [f'concentration[{n}]' for n in 'ABCDEF']
        assert out['concentration[B]'].tolist() == pytest.approx(
            original['concentration[s2]'].tolist()
        )

    def test_unnamed_survivor_keeps_its_new_label(self, captured_downloads):
        at = AppTest.from_function(qc_and_analysis_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
        at.session_state['_test_experiment'] = _experiment()
        at.session_state['sample_names'] = {'s3': 'C'}
        at.run()
        self._exclude(at, ['s1'])
        _select(at, 'analysis_radio', 'Lipidomic Heatmap')
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        sample_cols = [c for c in out.columns if c.startswith('concentration[')]
        # Old s2 (unnamed) is now s1; old s3 (C) is now s2.
        assert sample_cols[:2] == ['concentration[s1]', 'concentration[C]']

    def test_clearing_the_exclusion_restores_original_names(
        self, app, captured_downloads,
    ):
        self._exclude(app, ['s2'])
        self._exclude(app, [])
        _select(app, 'analysis_radio', 'Lipidomic Heatmap')
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        sample_cols = [c for c in out.columns if c.startswith('concentration[')]
        assert sample_cols == [f'concentration[{n}]' for n in 'ABCDEF']

    def test_group_exclusion(self, app, captured_downloads):
        """Excluding a whole condition renumbers Treatment's s4..s6 to s1..s3."""
        app.multiselect(key='pca_groups_remove').set_value(['Control']).run()
        assert not app.exception
        _select(app, 'analysis_radio', 'Lipidomic Heatmap')
        out = _csv(captured_downloads, 'analysis_csv_heatmap')
        sample_cols = [c for c in out.columns if c.startswith('concentration[')]
        assert sample_cols == [f'concentration[{n}]' for n in 'DEF']
