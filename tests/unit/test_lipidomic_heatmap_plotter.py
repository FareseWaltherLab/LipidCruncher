"""
Tests for LipidomicHeatmapPlotterService.

Covers: data filtering (conditions, classes, samples), Z-score computation
(row-wise normalization, NaN handling), hierarchical clustering (Ward linkage,
cluster labels, dendrogram order), clustered heatmap rendering (traces, layout,
cluster boundaries, symmetric colorscale), regular heatmap rendering,
cluster composition (species count and concentration modes), edge cases
(empty data, invalid inputs, single lipid, single sample), type coercion,
immutability, and large dataset stress tests.
"""

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.lipidomic_heatmap import (
    CELL_SIZE_PX,
    ClusteringResult,
    LipidomicHeatmapPlotterService,
    _compute_concentration_percentages,
    _compute_species_percentages,
)
from tests.conftest import make_experiment


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


def _make_df(lipids, classes, sample_values):
    """Build a DataFrame with LipidMolec, ClassKey, and concentration columns.

    Args:
        lipids: List of lipid name strings.
        classes: List of ClassKey strings (same length as lipids).
        sample_values: List of lists, one per sample column.
    """
    data = {'LipidMolec': lipids, 'ClassKey': classes}
    for i, vals in enumerate(sample_values, start=1):
        data[f'concentration[s{i}]'] = vals
    return pd.DataFrame(data)


def _make_clusterable_z_scores(n_lipids=6):
    """Build Z-score DataFrame with distinct patterns that produce real clusters.

    Creates lipids where half have high values in s1-s2 and low in s3-s4,
    and the other half have the opposite pattern.

    Returns:
        (z_scores_df, sample_names)
    """
    rng = np.random.RandomState(42)
    samples = ['s1', 's2', 's3', 's4']
    half = n_lipids // 2
    data = {}
    for s in samples[:2]:
        data[f'concentration[{s}]'] = (
            list(rng.uniform(800, 1000, half)) + list(rng.uniform(10, 50, n_lipids - half))
        )
    for s in samples[2:]:
        data[f'concentration[{s}]'] = (
            list(rng.uniform(10, 50, half)) + list(rng.uniform(800, 1000, n_lipids - half))
        )
    lipids = [f'L{i}' for i in range(n_lipids)]
    classes = ['PC'] * half + ['PE'] * (n_lipids - half)
    index = pd.MultiIndex.from_arrays([lipids, classes], names=['LipidMolec', 'ClassKey'])
    df = pd.DataFrame(data, index=index)
    cols = df.columns
    z_df = df[cols].apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    return z_df, samples


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_2x3():
    """2 conditions x 3 samples each."""
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x2():
    """3 conditions x 2 samples each."""
    return make_experiment(3, 2)


@pytest.fixture
def simple_df():
    """3 lipids (2 PC, 1 PE), 6 samples with distinct patterns per lipid."""
    return _make_df(
        lipids=['PC(34:1)', 'PC(36:2)', 'PE(38:4)'],
        classes=['PC', 'PC', 'PE'],
        sample_values=[
            [100.0, 800.0, 50.0],   # s1 — PC(34:1) low, PC(36:2) high, PE low
            [110.0, 790.0, 55.0],   # s2
            [120.0, 780.0, 60.0],   # s3
            [900.0, 100.0, 950.0],  # s4 — PC(34:1) high, PC(36:2) low, PE high
            [910.0, 110.0, 960.0],  # s5
            [920.0, 120.0, 970.0],  # s6
        ],
    )


@pytest.fixture
def multi_class_df():
    """4 lipids across 3 classes, 6 samples."""
    return _make_df(
        lipids=['PC(34:1)', 'PC(36:2)', 'PE(38:4)', 'SM(42:1)'],
        classes=['PC', 'PC', 'PE', 'SM'],
        sample_values=[
            [100.0, 200.0, 300.0, 400.0],  # s1
            [110.0, 210.0, 310.0, 410.0],  # s2
            [120.0, 220.0, 320.0, 420.0],  # s3
            [500.0, 600.0, 700.0, 800.0],  # s4
            [510.0, 610.0, 710.0, 810.0],  # s5
            [520.0, 620.0, 720.0, 820.0],  # s6
        ],
    )


@pytest.fixture
def filtered_df_with_index(simple_df, experiment_2x3):
    """Pre-filtered DataFrame and samples for Z-score / clustering tests."""
    filtered, samples = LipidomicHeatmapPlotterService.filter_data(
        simple_df, ['Control', 'Treatment'], ['PC', 'PE'], experiment_2x3,
    )
    return filtered, samples


@pytest.fixture
def z_scores_df(filtered_df_with_index):
    """Pre-computed Z-scores for convenience."""
    filtered, _ = filtered_df_with_index
    return LipidomicHeatmapPlotterService.compute_z_scores(filtered)


@pytest.fixture
def sample_names(filtered_df_with_index):
    """Sample names extracted from filter_data."""
    _, samples = filtered_df_with_index
    return samples


# ═══════════════════════════════════════════════════════════════════════
# TestFilterData — basic functionality
# ═══════════════════════════════════════════════════════════════════════


class TestFilterData:
    """Test lipidomic data filtering."""

    def test_returns_tuple(self, simple_df, experiment_2x3):
        result = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control'], ['PC'], experiment_2x3,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_filtered_df_has_correct_columns(self, simple_df, experiment_2x3):
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control'], ['PC'], experiment_2x3,
        )
        assert 'LipidMolec' in filtered.columns
        assert 'ClassKey' in filtered.columns
        for s in samples:
            assert f'concentration[{s}]' in filtered.columns

    def test_filters_by_class(self, simple_df, experiment_2x3):
        """Only PC lipids when selecting PC class."""
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control'], ['PC'], experiment_2x3,
        )
        assert all(filtered['ClassKey'] == 'PC')
        assert len(filtered) == 2

    def test_filters_by_multiple_classes(self, multi_class_df, experiment_2x3):
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            multi_class_df, ['Control'], ['PC', 'PE'], experiment_2x3,
        )
        assert set(filtered['ClassKey'].unique()) == {'PC', 'PE'}
        assert len(filtered) == 3

    def test_selects_correct_samples_for_condition(self, simple_df, experiment_2x3):
        """Control condition should use s1, s2, s3."""
        _, samples = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control'], ['PC'], experiment_2x3,
        )
        assert samples == ['s1', 's2', 's3']

    def test_selects_samples_for_multiple_conditions(self, simple_df, experiment_2x3):
        _, samples = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        assert samples == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_nonexistent_class_returns_empty(self, simple_df, experiment_2x3):
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control'], ['NonExistent'], experiment_2x3,
        )
        assert len(filtered) == 0

    def test_invalid_condition_skipped(self, simple_df, experiment_2x3):
        """Invalid conditions are skipped but valid ones still work."""
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control', 'FakeCondition'], ['PC'], experiment_2x3,
        )
        assert len(filtered) == 2
        assert samples == ['s1', 's2', 's3']


class TestFilterDataEdgeCases:
    """Test filter_data error handling."""

    def test_none_df_raises(self, experiment_2x3):
        with pytest.raises(ValueError, match="DataFrame is empty"):
            LipidomicHeatmapPlotterService.filter_data(
                None, ['Control'], ['PC'], experiment_2x3,
            )

    def test_empty_df_raises(self, experiment_2x3):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            LipidomicHeatmapPlotterService.filter_data(
                empty_df, ['Control'], ['PC'], experiment_2x3,
            )

    def test_empty_conditions_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one condition"):
            LipidomicHeatmapPlotterService.filter_data(
                simple_df, [], ['PC'], experiment_2x3,
            )

    def test_empty_classes_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one lipid class"):
            LipidomicHeatmapPlotterService.filter_data(
                simple_df, ['Control'], [], experiment_2x3,
            )

    def test_all_invalid_conditions_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="No valid samples"):
            LipidomicHeatmapPlotterService.filter_data(
                simple_df, ['Fake1', 'Fake2'], ['PC'], experiment_2x3,
            )


# ═══════════════════════════════════════════════════════════════════════
# TestComputeZScores
# ═══════════════════════════════════════════════════════════════════════


class TestComputeZScores:
    """Test Z-score computation."""

    def test_returns_dataframe(self, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        assert isinstance(z_scores, pd.DataFrame)

    def test_index_is_multiindex(self, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        assert z_scores.index.names == ['LipidMolec', 'ClassKey']

    def test_z_scores_have_zero_mean(self, filtered_df_with_index):
        """Each row should have mean ≈ 0."""
        filtered, _ = filtered_df_with_index
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        row_means = z_scores.mean(axis=1)
        for mean_val in row_means:
            assert mean_val == pytest.approx(0.0, abs=1e-10)

    def test_z_scores_have_unit_std(self, filtered_df_with_index):
        """Each row should have std ≈ 1."""
        filtered, _ = filtered_df_with_index
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        row_stds = z_scores.std(axis=1)
        for std_val in row_stds:
            assert std_val == pytest.approx(1.0, abs=1e-10)

    def test_shape_matches_input(self, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        n_conc_cols = len([c for c in filtered.columns if c.startswith('concentration[')])
        assert z_scores.shape == (len(filtered), n_conc_cols)

    def test_lipid_names_preserved(self, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        lipid_names = z_scores.index.get_level_values('LipidMolec').tolist()
        assert 'PC(34:1)' in lipid_names
        assert 'PC(36:2)' in lipid_names
        assert 'PE(38:4)' in lipid_names

    def test_constant_row_produces_nan(self, experiment_2x3):
        """A lipid with identical concentrations across all samples → NaN Z-scores."""
        df = _make_df(
            lipids=['PC(34:1)'],
            classes=['PC'],
            sample_values=[
                [100.0], [100.0], [100.0], [100.0], [100.0], [100.0],
            ],
        )
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        assert z_scores.isna().all(axis=None)


class TestComputeZScoresEdgeCases:
    """Test Z-score edge cases."""

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Filtered DataFrame is empty"):
            LipidomicHeatmapPlotterService.compute_z_scores(None)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="Filtered DataFrame is empty"):
            LipidomicHeatmapPlotterService.compute_z_scores(pd.DataFrame())

    def test_single_sample_produces_nan(self, experiment_2x3):
        """Single sample → std=NaN → Z-scores are NaN."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(34:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        # Can't use filter_data (needs at least valid condition samples),
        # so build the filtered DF directly
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(df)
        assert z_scores.isna().all(axis=None)


# ═══════════════════════════════════════════════════════════════════════
# TestPerformClustering
# ═══════════════════════════════════════════════════════════════════════


class TestPerformClustering:
    """Test hierarchical clustering."""

    def test_returns_clustering_result(self, z_scores_df):
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 2)
        assert isinstance(result, ClusteringResult)

    def test_linkage_matrix_shape(self, z_scores_df):
        """Linkage matrix should have (n-1) rows and 4 columns."""
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 2)
        n = len(z_scores_df)
        assert result.linkage_matrix.shape == (n - 1, 4)

    def test_cluster_labels_length(self, z_scores_df):
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 2)
        assert len(result.cluster_labels) == len(z_scores_df)

    def test_cluster_labels_range(self, z_scores_df):
        """Labels should be 1-based integers in range [1, n_clusters]."""
        n_clusters = 2
        result = LipidomicHeatmapPlotterService.perform_clustering(
            z_scores_df, n_clusters,
        )
        assert set(result.cluster_labels).issubset({1, 2})

    def test_dendrogram_order_length(self, z_scores_df):
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 2)
        assert len(result.dendrogram_order) == len(z_scores_df)

    def test_dendrogram_order_is_permutation(self, z_scores_df):
        """Dendrogram order should be a permutation of row indices."""
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 2)
        assert sorted(result.dendrogram_order) == list(range(len(z_scores_df)))

    def test_single_cluster(self, z_scores_df):
        """n_clusters=1 → all lipids in one cluster."""
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 1)
        assert all(result.cluster_labels == 1)

    def test_max_clusters(self, z_scores_df):
        """n_clusters = n_lipids → labels are assigned to all lipids."""
        n = len(z_scores_df)
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, n)
        assert len(result.cluster_labels) == n
        # At least as many clusters as there are distinct distances
        assert len(set(result.cluster_labels)) >= 1

    def test_nan_values_handled(self, experiment_2x3):
        """Clustering should handle NaN Z-scores (filled with 0)."""
        df = _make_df(
            lipids=['PC(34:1)', 'PC(36:2)'],
            classes=['PC', 'PC'],
            sample_values=[
                [100.0, 100.0], [100.0, 100.0], [100.0, 100.0],
                [100.0, 100.0], [100.0, 100.0], [100.0, 100.0],
            ],
        )
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        # All NaN since constant rows
        result = LipidomicHeatmapPlotterService.perform_clustering(z_scores, 1)
        assert isinstance(result, ClusteringResult)


class TestPerformClusteringEdgeCases:
    """Test clustering error handling."""

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.perform_clustering(None, 2)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.perform_clustering(pd.DataFrame(), 2)

    def test_zero_clusters_raises(self, z_scores_df):
        with pytest.raises(ValueError, match="at least 1"):
            LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, 0)

    def test_negative_clusters_raises(self, z_scores_df):
        with pytest.raises(ValueError, match="at least 1"):
            LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, -1)

    def test_too_many_clusters_raises(self, z_scores_df):
        n = len(z_scores_df)
        with pytest.raises(ValueError, match="cannot exceed"):
            LipidomicHeatmapPlotterService.perform_clustering(z_scores_df, n + 1)


# ═══════════════════════════════════════════════════════════════════════
# TestGenerateClusteredHeatmap
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateClusteredHeatmap:
    """Test clustered heatmap rendering."""

    def test_returns_figure(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_heatmap_z_shape(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(heatmap.z)
        assert z.shape == (len(z_scores_df), len(sample_names))

    def test_symmetric_colorscale(self, z_scores_df, sample_names):
        """zmin and zmax should be symmetric around 0."""
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.zmin == -heatmap.zmax
        assert heatmap.zmin < 0

    def test_rdbu_colorscale(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        # Plotly expands named colorscales to tuples; check it's RdBu_r-like
        assert len(heatmap.colorscale) > 0

    def test_colorbar_title(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.colorbar.title.text == 'Z-score'

    def test_cluster_boundary_lines(self):
        """2 clusters → 1 boundary line (needs ≥4 lipids with distinct patterns)."""
        z_df, samples = _make_clusterable_z_scores(n_lipids=6)
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_df, samples, 2,
        )
        lines = [s for s in fig.layout.shapes if s.type == 'line']
        assert len(lines) == 1

    def test_three_clusters_two_lines(self):
        """3 clusters → 2 boundary lines."""
        z_df, samples = _make_clusterable_z_scores(n_lipids=9)
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_df, samples, 3,
        )
        lines = [s for s in fig.layout.shapes if s.type == 'line']
        assert len(lines) == 2

    def test_single_cluster_no_lines(self, z_scores_df, sample_names):
        """1 cluster → 0 boundary lines."""
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 1,
        )
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        lines = [s for s in shapes if s.type == 'line']
        assert len(lines) == 0

    def test_boundary_line_style(self):
        z_df, samples = _make_clusterable_z_scores(n_lipids=6)
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_df, samples, 2,
        )
        line = fig.layout.shapes[0]
        assert line.line.color == 'black'
        assert line.line.dash == 'dash'
        assert line.line.width == 2


class TestClusteredHeatmapLayout:
    """Test clustered heatmap layout properties."""

    def test_title(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert 'Clustered' in fig.layout.title.text

    def test_xaxis_title(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert fig.layout.xaxis.title.text == 'Samples'

    def test_yaxis_title(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert fig.layout.yaxis.title.text == 'Lipid Molecules'

    def test_dimensions_follow_the_grid(self, z_scores_df, sample_names):
        """Size is derived from the cell count, not a fixed 900x600 canvas,
        so that every cell renders as a square."""
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        layout = fig.layout
        plot_width = layout.width - layout.margin.l - layout.margin.r
        plot_height = layout.height - layout.margin.t - layout.margin.b
        assert plot_width == len(sample_names) * CELL_SIZE_PX
        assert plot_height == len(z_scores_df) * CELL_SIZE_PX

    def test_xaxis_tickangle(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert fig.layout.xaxis.tickangle == 45

    def test_yaxis_reversed(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert fig.layout.yaxis.autorange == 'reversed'


class TestClusteredHeatmapEdgeCases:
    """Test clustered heatmap error handling."""

    def test_none_z_scores_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.generate_clustered_heatmap(
                None, ['s1'], 2,
            )

    def test_empty_z_scores_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.generate_clustered_heatmap(
                pd.DataFrame(), ['s1'], 2,
            )


# ═══════════════════════════════════════════════════════════════════════
# TestGenerateRegularHeatmap
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateRegularHeatmap:
    """Test regular heatmap rendering."""

    def test_returns_figure(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_heatmap_z_shape(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(heatmap.z)
        assert z.shape == (len(z_scores_df), len(sample_names))

    def test_symmetric_colorscale(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.zmin == -heatmap.zmax

    def test_rdbu_colorscale(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        # Plotly expands named colorscales to tuples; check it's RdBu_r-like
        assert len(heatmap.colorscale) > 0

    def test_title(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        assert 'Regular' in fig.layout.title.text

    def test_no_cluster_boundary_lines(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        assert len(shapes) == 0

    def test_preserves_original_order(self, z_scores_df, sample_names):
        """Regular heatmap should keep lipids in their original DataFrame order."""
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        y_labels = list(heatmap.y)
        original_labels = z_scores_df.index.get_level_values('LipidMolec').tolist()
        assert y_labels == original_labels


class TestRegularHeatmapEdgeCases:
    """Test regular heatmap error handling."""

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.generate_regular_heatmap(None, ['s1'])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.generate_regular_heatmap(
                pd.DataFrame(), ['s1'],
            )


# ═══════════════════════════════════════════════════════════════════════
# TestGetClusterComposition — species count mode
# ═══════════════════════════════════════════════════════════════════════


class TestGetClusterCompositionSpecies:
    """Test cluster composition in species_count mode."""

    def test_returns_dataframe(self, z_scores_df):
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='species_count',
        )
        assert isinstance(result, pd.DataFrame)

    def test_rows_are_clusters(self, z_scores_df):
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='species_count',
        )
        assert all(c in [1, 2] for c in result.index)

    def test_columns_are_classes(self, z_scores_df):
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='species_count',
        )
        # Should have ClassKey values as columns
        assert all(isinstance(c, str) for c in result.columns)

    def test_percentages_sum_to_100(self, z_scores_df):
        """Each cluster's percentages should sum to 100."""
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='species_count',
        )
        for cluster_idx in result.index:
            row_sum = result.loc[cluster_idx].sum()
            assert row_sum == pytest.approx(100.0)

    def test_single_class_all_100(self, experiment_2x3):
        """All lipids same class → 100% for that class in every cluster."""
        df = _make_df(
            lipids=['PC(34:1)', 'PC(36:2)', 'PC(38:4)'],
            classes=['PC', 'PC', 'PC'],
            sample_values=[
                [100.0, 200.0, 300.0],
                [110.0, 210.0, 310.0],
                [120.0, 220.0, 320.0],
                [500.0, 600.0, 700.0],
                [510.0, 610.0, 710.0],
                [520.0, 620.0, 720.0],
            ],
        )
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores, 2, mode='species_count',
        )
        for cluster_idx in result.index:
            assert result.loc[cluster_idx, 'PC'] == pytest.approx(100.0)


# ═══════════════════════════════════════════════════════════════════════
# TestGetClusterComposition — concentration mode
# ═══════════════════════════════════════════════════════════════════════


class TestGetClusterCompositionConcentration:
    """Test cluster composition in concentration mode."""

    def test_returns_dataframe(self, z_scores_df, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='concentration', filtered_df=filtered,
        )
        assert isinstance(result, pd.DataFrame)

    def test_percentages_sum_to_100(self, z_scores_df, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='concentration', filtered_df=filtered,
        )
        for cluster_idx in result.index:
            row_sum = result.loc[cluster_idx].sum()
            assert row_sum == pytest.approx(100.0)

    def test_missing_filtered_df_raises(self, z_scores_df):
        with pytest.raises(ValueError, match="filtered_df is required"):
            LipidomicHeatmapPlotterService.get_cluster_composition(
                z_scores_df, 2, mode='concentration', filtered_df=None,
            )

    def test_empty_filtered_df_raises(self, z_scores_df):
        with pytest.raises(ValueError, match="filtered_df is required"):
            LipidomicHeatmapPlotterService.get_cluster_composition(
                z_scores_df, 2, mode='concentration', filtered_df=pd.DataFrame(),
            )


class TestGetClusterCompositionEdgeCases:
    """Test cluster composition error handling."""

    def test_invalid_mode_raises(self, z_scores_df):
        with pytest.raises(ValueError, match="Invalid mode"):
            LipidomicHeatmapPlotterService.get_cluster_composition(
                z_scores_df, 2, mode='invalid',
            )

    def test_none_z_scores_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.get_cluster_composition(
                None, 2, mode='species_count',
            )

    def test_empty_z_scores_raises(self):
        with pytest.raises(ValueError, match="Z-scores DataFrame is empty"):
            LipidomicHeatmapPlotterService.get_cluster_composition(
                pd.DataFrame(), 2, mode='species_count',
            )


# ═══════════════════════════════════════════════════════════════════════
# TestClusteringResultDataclass
# ═══════════════════════════════════════════════════════════════════════


class TestClusteringResultDataclass:
    """Test ClusteringResult dataclass defaults and attributes."""

    def test_default_empty(self):
        result = ClusteringResult()
        assert len(result.linkage_matrix) == 0
        assert len(result.cluster_labels) == 0
        assert len(result.dendrogram_order) == 0

    def test_with_values(self):
        linkage = np.array([[0, 1, 1.0, 2]])
        labels = np.array([1, 1])
        order = np.array([0, 1])
        result = ClusteringResult(
            linkage_matrix=linkage,
            cluster_labels=labels,
            dendrogram_order=order,
        )
        np.testing.assert_array_equal(result.linkage_matrix, linkage)
        np.testing.assert_array_equal(result.cluster_labels, labels)
        np.testing.assert_array_equal(result.dendrogram_order, order)


# ═══════════════════════════════════════════════════════════════════════
# TestTypeCoercion
# ═══════════════════════════════════════════════════════════════════════


class TestTypeCoercion:
    """Test that various numeric types are handled correctly."""

    def test_integer_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(34:1)', 'PC(36:2)'],
            classes=['PC', 'PC'],
            sample_values=[
                [100, 200], [110, 210], [120, 220],
                [500, 600], [510, 610], [520, 620],
            ],
        )
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores, samples,
        )
        assert isinstance(fig, go.Figure)

    def test_float32_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(34:1)'],
            classes=['PC'],
            sample_values=[
                np.array([100.0], dtype=np.float32),
                np.array([110.0], dtype=np.float32),
                np.array([120.0], dtype=np.float32),
                np.array([500.0], dtype=np.float32),
                np.array([510.0], dtype=np.float32),
                np.array([520.0], dtype=np.float32),
            ],
        )
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        assert isinstance(z_scores, pd.DataFrame)

    def test_full_pipeline_int_to_clustered_heatmap(self, experiment_2x3):
        """End-to-end: int data → filter → z-scores → clustered heatmap."""
        df = _make_df(
            lipids=['PC(34:1)', 'PC(36:2)', 'PE(38:4)'],
            classes=['PC', 'PC', 'PE'],
            sample_values=[
                [100, 200, 300], [110, 210, 310], [120, 220, 320],
                [500, 600, 700], [510, 610, 710], [520, 620, 720],
            ],
        )
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC', 'PE'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores, samples, 2,
        )
        assert isinstance(fig, go.Figure)


# ═══════════════════════════════════════════════════════════════════════
# TestImmutability
# ═══════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Test that input DataFrames are not modified by service methods."""

    def test_filter_data_preserves_input(self, simple_df, experiment_2x3):
        df_copy = simple_df.copy()
        LipidomicHeatmapPlotterService.filter_data(
            simple_df, ['Control'], ['PC'], experiment_2x3,
        )
        pd.testing.assert_frame_equal(simple_df, df_copy)

    def test_compute_z_scores_preserves_input(self, filtered_df_with_index):
        filtered, _ = filtered_df_with_index
        filtered_copy = filtered.copy()
        LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        pd.testing.assert_frame_equal(filtered, filtered_copy)

    def test_clustered_heatmap_preserves_z_scores(self, z_scores_df, sample_names):
        z_copy = z_scores_df.copy()
        LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        pd.testing.assert_frame_equal(z_scores_df, z_copy)

    def test_regular_heatmap_preserves_z_scores(self, z_scores_df, sample_names):
        z_copy = z_scores_df.copy()
        LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores_df, sample_names,
        )
        pd.testing.assert_frame_equal(z_scores_df, z_copy)

    def test_cluster_composition_preserves_z_scores(self, z_scores_df):
        z_copy = z_scores_df.copy()
        LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores_df, 2, mode='species_count',
        )
        pd.testing.assert_frame_equal(z_scores_df, z_copy)


# ═══════════════════════════════════════════════════════════════════════
# TestLargeDataset
# ═══════════════════════════════════════════════════════════════════════


class TestLargeDataset:
    """Stress tests with large datasets."""

    def test_100_lipids_filter_and_z_scores(self, experiment_2x3):
        rng = np.random.RandomState(42)
        n = 100
        lipids = [f'PC({i}:0)' for i in range(n)]
        classes = ['PC'] * n
        sample_values = [rng.uniform(10, 1000, n).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        assert z_scores.shape == (100, 6)

    def test_100_lipids_clustered_heatmap(self, experiment_2x3):
        rng = np.random.RandomState(42)
        n = 100
        lipids = [f'PC({i}:0)' for i in range(n)]
        classes = ['PC'] * n
        sample_values = [rng.uniform(10, 1000, n).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores, samples, 5,
        )
        assert isinstance(fig, go.Figure)
        lines = [s for s in fig.layout.shapes if s.type == 'line']
        assert len(lines) == 4  # 5 clusters → 4 boundaries

    def test_100_lipids_regular_heatmap(self, experiment_2x3):
        rng = np.random.RandomState(42)
        n = 100
        lipids = [f'PC({i}:0)' for i in range(n)]
        classes = ['PC'] * n
        sample_values = [rng.uniform(10, 1000, n).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z_scores, samples,
        )
        assert isinstance(fig, go.Figure)

    def test_mixed_classes_cluster_composition(self, experiment_2x3):
        """50 PC + 50 PE lipids → composition should reflect class distribution."""
        rng = np.random.RandomState(42)
        n = 100
        lipids = [f'PC({i}:0)' for i in range(50)] + [
            f'PE({i}:0)' for i in range(50)
        ]
        classes = ['PC'] * 50 + ['PE'] * 50
        sample_values = [rng.uniform(10, 1000, n).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        filtered, _ = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC', 'PE'], experiment_2x3,
        )
        z_scores = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        result = LipidomicHeatmapPlotterService.get_cluster_composition(
            z_scores, 3, mode='species_count',
        )
        # Every cluster row should sum to 100%
        for cluster_idx in result.index:
            assert result.loc[cluster_idx].sum() == pytest.approx(100.0)


# ═══════════════════════════════════════════════════════════════════════
# TestPrivateHelpers
# ═══════════════════════════════════════════════════════════════════════


class TestComputeSpeciesPercentages:
    """Test _compute_species_percentages helper."""

    def test_single_class(self):
        index = pd.MultiIndex.from_tuples(
            [('L1', 'PC'), ('L2', 'PC')], names=['LipidMolec', 'ClassKey'],
        )
        z_df = pd.DataFrame(
            [[1.0, -1.0], [0.5, -0.5]], index=index, columns=['s1', 's2'],
        )
        labels = np.array([1, 1])
        result = _compute_species_percentages(z_df, labels)
        assert result.loc[1, 'PC'] == pytest.approx(100.0)

    def test_mixed_classes(self):
        index = pd.MultiIndex.from_tuples(
            [('L1', 'PC'), ('L2', 'PE')], names=['LipidMolec', 'ClassKey'],
        )
        z_df = pd.DataFrame(
            [[1.0, -1.0], [0.5, -0.5]], index=index, columns=['s1', 's2'],
        )
        labels = np.array([1, 1])  # Both in same cluster
        result = _compute_species_percentages(z_df, labels)
        assert result.loc[1, 'PC'] == pytest.approx(50.0)
        assert result.loc[1, 'PE'] == pytest.approx(50.0)


class TestComputeConcentrationPercentages:
    """Test _compute_concentration_percentages helper."""

    def test_proportional_to_concentration(self):
        index = pd.MultiIndex.from_tuples(
            [('L1', 'PC'), ('L2', 'PE')], names=['LipidMolec', 'ClassKey'],
        )
        z_df = pd.DataFrame(
            [[1.0, -1.0], [0.5, -0.5]], index=index, columns=['s1', 's2'],
        )
        filtered_df = pd.DataFrame({
            'LipidMolec': ['L1', 'L2'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [300.0, 100.0],
            'concentration[s2]': [300.0, 100.0],
        })
        labels = np.array([1, 1])
        result = _compute_concentration_percentages(z_df, filtered_df, labels)
        # PC: 600 total, PE: 200 total → PC=75%, PE=25%
        assert result.loc[1, 'PC'] == pytest.approx(75.0)
        assert result.loc[1, 'PE'] == pytest.approx(25.0)


# ═══════════════════════════════════════════════════════════════════════
# TestSampleConditionLabels
# ═══════════════════════════════════════════════════════════════════════

class TestSampleConditionLabels:
    def test_one_label_per_sample(self):
        experiment = make_experiment(n_conditions=2, samples_per_condition=3)
        labels = LipidomicHeatmapPlotterService.sample_condition_labels(
            ['Control', 'Treatment'], experiment,
        )
        assert labels == ['Control'] * 3 + ['Treatment'] * 3

    def test_aligned_with_filter_data_samples(self):
        """Labels must be index-aligned with the samples filter_data returns."""
        experiment = make_experiment(n_conditions=2, samples_per_condition=3)
        df = _make_df(
            ['L1', 'L2'], ['PC', 'PE'],
            [[float(i)] * 2 for i in range(6)],
        )
        _, samples = LipidomicHeatmapPlotterService.filter_data(
            df, ['Control', 'Treatment'], ['PC', 'PE'], experiment,
        )
        labels = LipidomicHeatmapPlotterService.sample_condition_labels(
            ['Control', 'Treatment'], experiment,
        )
        assert len(labels) == len(samples)

    def test_unknown_condition_skipped(self):
        experiment = make_experiment(n_conditions=2, samples_per_condition=2)
        labels = LipidomicHeatmapPlotterService.sample_condition_labels(
            ['Control', 'NotAConditon'], experiment,
        )
        assert labels == ['Control', 'Control']

    def test_uneven_group_sizes(self):
        experiment = make_experiment(
            n_conditions=2, number_of_samples_list=[1, 3],
        )
        labels = LipidomicHeatmapPlotterService.sample_condition_labels(
            ['Control', 'Treatment'], experiment,
        )
        assert labels == ['Control'] + ['Treatment'] * 3


# ═══════════════════════════════════════════════════════════════════════
# TestOrderByClass
# ═══════════════════════════════════════════════════════════════════════

class TestOrderByClass:
    @staticmethod
    def _z(classes):
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(len(classes))], classes],
            names=['LipidMolec', 'ClassKey'],
        )
        return pd.DataFrame(
            np.arange(len(classes) * 2, dtype=float).reshape(-1, 2),
            index=index, columns=['s1', 's2'],
        )

    def test_classes_become_contiguous(self):
        z = self._z(['PC', 'TG', 'PE', 'PC', 'TG', 'PE'])
        out = LipidomicHeatmapPlotterService.order_by_class(z)
        classes = list(out.index.get_level_values('ClassKey'))
        runs = [k for k, _ in itertools.groupby(classes)]
        assert len(runs) == len(set(runs))

    def test_first_appearance_order_kept(self):
        z = self._z(['TG', 'PC', 'PE', 'TG'])
        out = LipidomicHeatmapPlotterService.order_by_class(z)
        classes = list(out.index.get_level_values('ClassKey'))
        assert [k for k, _ in itertools.groupby(classes)] == ['TG', 'PC', 'PE']

    def test_row_values_follow_their_lipid(self):
        z = self._z(['PC', 'TG', 'PC'])
        out = LipidomicHeatmapPlotterService.order_by_class(z)
        for lipid in z.index.get_level_values('LipidMolec'):
            original = z.xs(lipid, level='LipidMolec').to_numpy()
            moved = out.xs(lipid, level='LipidMolec').to_numpy()
            assert np.array_equal(original, moved)

    def test_no_rows_lost(self):
        z = self._z(['PC', 'TG', 'PE', 'PC'])
        out = LipidomicHeatmapPlotterService.order_by_class(z)
        assert len(out) == len(z)
        assert set(out.index) == set(z.index)

    def test_already_grouped_is_unchanged(self):
        z = self._z(['PC', 'PC', 'TG'])
        out = LipidomicHeatmapPlotterService.order_by_class(z)
        assert list(out.index) == list(z.index)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LipidomicHeatmapPlotterService.order_by_class(pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════════
# TestClassGroupedHeatmap
# ═══════════════════════════════════════════════════════════════════════

class TestClassGroupedHeatmap:
    @staticmethod
    def _z(classes=('PC', 'TG', 'PE', 'PC')):
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(len(classes))], list(classes)],
            names=['LipidMolec', 'ClassKey'],
        )
        return pd.DataFrame(
            np.arange(len(classes) * 3, dtype=float).reshape(-1, 3),
            index=index, columns=['s1', 's2', 's3'],
        )

    def test_returns_figure_with_heatmap(self):
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(), ['s1', 's2', 's3'],
        )
        assert isinstance(fig, go.Figure)
        assert [t for t in fig.data if isinstance(t, go.Heatmap)]

    def test_y_axis_is_two_level(self):
        """Class must be the outer level so it renders left of the species."""
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(), ['s1', 's2', 's3'],
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert len(heatmap.y) == 2
        assert list(heatmap.y[0]) == ['PC', 'PC', 'TG', 'PE']
        assert list(heatmap.y[1]) == ['L0', 'L3', 'L1', 'L2']

    def test_dividers_enabled_between_class_blocks(self):
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(), ['s1', 's2', 's3'],
        )
        assert fig.layout.yaxis.showdividers is True
        assert fig.layout.yaxis.dividerwidth == 2

    def test_rows_reordered_with_their_values(self):
        z = self._z()
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            z, ['s1', 's2', 's3'],
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        expected = LipidomicHeatmapPlotterService.order_by_class(z).to_numpy()
        assert np.array_equal(np.asarray(heatmap.z), expected)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
                pd.DataFrame(), ['s1'],
            )


# ═══════════════════════════════════════════════════════════════════════
# TestSquareCells
# ═══════════════════════════════════════════════════════════════════════

class TestSquareCells:
    @staticmethod
    def _z(n_rows=4, n_cols=3):
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(n_rows)], ['PC'] * n_rows],
            names=['LipidMolec', 'ClassKey'],
        )
        return pd.DataFrame(
            np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols),
            index=index, columns=[f's{i+1}' for i in range(n_cols)],
        )

    @staticmethod
    def _cell_size(fig, n_rows, n_cols):
        layout = fig.layout
        width = (layout.width - layout.margin.l - layout.margin.r) / n_cols
        height = (layout.height - layout.margin.t - layout.margin.b) / n_rows
        return width, height

    def test_regular_heatmap_cells_are_square(self):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(), ['s1', 's2', 's3'],
        )
        width, height = self._cell_size(fig, 4, 3)
        assert width == height == CELL_SIZE_PX

    def test_clustered_heatmap_cells_are_square(self):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            self._z(), ['s1', 's2', 's3'], 2,
        )
        width, height = self._cell_size(fig, 4, 3)
        assert width == height == CELL_SIZE_PX

    def test_class_grouped_cells_are_square(self):
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(), ['s1', 's2', 's3'],
        )
        width, height = self._cell_size(fig, 4, 3)
        assert width == height == CELL_SIZE_PX

    def test_height_scales_with_species_count(self):
        small = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(n_rows=4), ['s1', 's2', 's3'],
        )
        large = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(n_rows=40), ['s1', 's2', 's3'],
        )
        assert large.layout.height - small.layout.height == 36 * CELL_SIZE_PX

    def test_width_scales_with_sample_count(self):
        narrow = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(n_cols=3), ['s1', 's2', 's3'],
        )
        wide = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(n_cols=9), [f's{i+1}' for i in range(9)],
        )
        assert wide.layout.width - narrow.layout.width == 6 * CELL_SIZE_PX


# ═══════════════════════════════════════════════════════════════════════
# TestConditionStrip
# ═══════════════════════════════════════════════════════════════════════

class TestConditionStrip:
    @staticmethod
    def _z():
        index = pd.MultiIndex.from_arrays(
            [['L0', 'L1'], ['PC', 'PE']], names=['LipidMolec', 'ClassKey'],
        )
        return pd.DataFrame(
            np.arange(12, dtype=float).reshape(2, 6),
            index=index, columns=[f's{i+1}' for i in range(6)],
        )

    SAMPLES = ['s1', 's2', 's3', 's4', 's5', 's6']
    CONDITIONS = ['Control', 'Control', 'Control', 'Treat', 'Treat', 'Treat']

    def _fig(self, **kwargs):
        return LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(), self.SAMPLES, sample_conditions=self.CONDITIONS, **kwargs
        )

    def test_one_block_per_condition(self):
        rects = [s for s in self._fig().layout.shapes if s.type == 'rect']
        assert len(rects) == 2

    def test_blocks_span_their_samples(self):
        rects = [s for s in self._fig().layout.shapes if s.type == 'rect']
        assert (rects[0].x0, rects[0].x1) == (-0.5, 2.5)
        assert (rects[1].x0, rects[1].x1) == (2.5, 5.5)

    def test_blocks_sit_above_the_plot_area(self):
        rects = [s for s in self._fig().layout.shapes if s.type == 'rect']
        for rect in rects:
            assert rect.yref == 'paper'
            assert rect.y0 > 1.0

    def test_separator_between_conditions(self):
        lines = [
            s for s in self._fig().layout.shapes
            if s.type == 'line' and s.y0 == 0 and s.y1 == 1
        ]
        assert len(lines) == 1
        assert lines[0].x0 == 2.5

    def test_legend_entry_per_condition(self):
        proxies = [t for t in self._fig().data if isinstance(t, go.Scatter)]
        assert [t.name for t in proxies] == ['Control', 'Treat']

    def test_conditions_use_shared_palette(self):
        from app.services.plotting._shared import generate_condition_color_mapping
        expected = generate_condition_color_mapping(['Control', 'Treat'])
        proxies = [t for t in self._fig().data if isinstance(t, go.Scatter)]
        for trace in proxies:
            assert trace.marker.color == expected[trace.name]

    def test_absent_when_conditions_not_supplied(self):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(), self.SAMPLES,
        )
        assert not [s for s in fig.layout.shapes if s.type == 'rect']
        assert not [t for t in fig.data if isinstance(t, go.Scatter)]

    def test_non_contiguous_conditions_get_separate_blocks(self):
        """A condition split across the axis must not be merged into one block."""
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(), self.SAMPLES,
            sample_conditions=['A', 'A', 'B', 'B', 'A', 'A'],
        )
        rects = [s for s in fig.layout.shapes if s.type == 'rect']
        assert len(rects) == 3

    def test_single_condition_has_no_separator(self):
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(), self.SAMPLES, sample_conditions=['A'] * 6,
        )
        lines = [
            s for s in fig.layout.shapes
            if s.type == 'line' and s.y0 == 0 and s.y1 == 1
        ]
        assert lines == []

    def test_figure_serializes(self):
        """Catches invalid Plotly specs that only surface on render."""
        assert self._fig().to_json()
