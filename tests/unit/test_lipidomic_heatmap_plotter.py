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
    GROUPED_PAGE_SIZE,
    HEATMAP_HEIGHT,
    MARGIN_TOP,
    MAX_CELL_SIZE_PX,
    STRIP_GAP_PX,
    STRIP_HEIGHT_PX,
    STRIP_LABEL_GAP_PX,
    STRIP_LABEL_PX_PER_CHAR,
    STRIP_LABEL_ROW_PX,
    STRETCHED_PLOT_WIDTH_PX,
    ClusteringResult,
    LipidomicHeatmapPlotterService,
    _compute_concentration_percentages,
    _compute_species_percentages,
    _condition_blocks,
    _label_rows,
    cell_size,
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

    def test_dimensions(self, z_scores_df, sample_names):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_scores_df, sample_names, 2,
        )
        assert fig.layout.width == 900
        assert fig.layout.height == 600

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
# TestOrderByClusters
# ═══════════════════════════════════════════════════════════════════════

class TestOrderByClusters:
    """Rows in the clustered heatmap's order, with their cluster, for the CSV."""

    def test_rows_follow_dendrogram_order(self):
        z_df, _ = _make_clusterable_z_scores(n_lipids=8)
        clustering = LipidomicHeatmapPlotterService.perform_clustering(z_df, 2)
        out = LipidomicHeatmapPlotterService.order_by_clusters(z_df, 2)
        assert list(out.index) == list(z_df.index[clustering.dendrogram_order])

    def test_rows_match_the_clustered_figure(self):
        z_df, samples = _make_clusterable_z_scores(n_lipids=8)
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            z_df, samples, 2,
        )
        y = fig.data[0].y
        # Tolerate a two-level (class, species) axis as well as a flat one.
        species = list(y[-1]) if isinstance(y[0], (list, tuple, np.ndarray)) else list(y)
        out = LipidomicHeatmapPlotterService.order_by_clusters(z_df, 2)
        assert list(out.index.get_level_values('LipidMolec')) == species

    def test_leading_cluster_column(self):
        z_df, _ = _make_clusterable_z_scores(n_lipids=8)
        clustering = LipidomicHeatmapPlotterService.perform_clustering(z_df, 2)
        out = LipidomicHeatmapPlotterService.order_by_clusters(z_df, 2)
        assert out.columns[0] == 'Cluster'
        assert list(out.columns[1:]) == list(z_df.columns)
        assert out['Cluster'].tolist() == list(
            clustering.cluster_labels[clustering.dendrogram_order]
        )

    def test_row_values_follow_their_lipid(self):
        z_df, _ = _make_clusterable_z_scores(n_lipids=8)
        out = LipidomicHeatmapPlotterService.order_by_clusters(z_df, 2)
        pd.testing.assert_frame_equal(
            out.drop(columns='Cluster').loc[z_df.index], z_df,
        )

    def test_input_not_mutated(self):
        z_df, _ = _make_clusterable_z_scores(n_lipids=8)
        before = z_df.copy()
        LipidomicHeatmapPlotterService.order_by_clusters(z_df, 2)
        pd.testing.assert_frame_equal(z_df, before)


# ═══════════════════════════════════════════════════════════════════════
# TestOrderByClassSorted
# ═══════════════════════════════════════════════════════════════════════

class TestOrderByClassSorted:
    """Species ranked by value inside each class block."""

    @staticmethod
    def _z(classes, values):
        """One row per class/value pair, the same value in both columns."""
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(len(classes))], list(classes)],
            names=['LipidMolec', 'ClassKey'],
        )
        return pd.DataFrame(
            {'s1': list(values), 's2': list(values)},
            index=index, dtype=float,
        )

    @staticmethod
    def _lipids(frame):
        return list(frame.index.get_level_values('LipidMolec'))

    def test_descending_is_the_default(self):
        z = self._z(['PC', 'PC', 'PC'], [0.0, 2.0, -2.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        assert self._lipids(out) == ['L1', 'L0', 'L2']

    def test_ascending_reverses_the_ranking(self):
        z = self._z(['PC', 'PC', 'PC'], [0.0, 2.0, -2.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'], ascending=True,
        )
        assert self._lipids(out) == ['L2', 'L0', 'L1']

    def test_sorting_stays_inside_the_class_block(self):
        """A large PE value must not outrank the PC species it follows."""
        z = self._z(['PC', 'PE', 'PC', 'PE'], [1.0, 9.0, 5.0, -9.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        classes = list(out.index.get_level_values('ClassKey'))
        assert classes == ['PC', 'PC', 'PE', 'PE']
        assert self._lipids(out) == ['L2', 'L0', 'L1', 'L3']

    def test_class_block_order_is_unaffected(self):
        """Blocks keep first-appearance order however the species rank."""
        z = self._z(['TG', 'PC', 'TG'], [-5.0, 5.0, 5.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        classes = list(out.index.get_level_values('ClassKey'))
        assert [k for k, _ in itertools.groupby(classes)] == ['TG', 'PC']

    def test_row_values_follow_their_lipid(self):
        z = self._z(['PC', 'PC', 'PE'], [1.0, 3.0, 2.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        for lipid in self._lipids(z):
            original = z.xs(lipid, level='LipidMolec').to_numpy()
            moved = out.xs(lipid, level='LipidMolec').to_numpy()
            assert np.array_equal(original, moved)

    def test_ties_keep_input_order(self):
        z = self._z(['PC', 'PC', 'PC'], [1.0, 1.0, 1.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        assert self._lipids(out) == ['L0', 'L1', 'L2']

    def test_unrankable_species_go_last(self):
        """An all-NaN row has no rank; it belongs at the foot of its block."""
        z = self._z(['PC', 'PC', 'PC'], [1.0, np.nan, 3.0])
        for ascending in (False, True):
            out = LipidomicHeatmapPlotterService.order_by_class(
                z, sort_columns=['s1', 's2'], ascending=ascending,
            )
            assert self._lipids(out)[-1] == 'L1'

    def test_partial_nan_row_ranks_on_what_it_has(self):
        z = self._z(['PC', 'PC'], [1.0, 0.0])
        z.iloc[1, 0] = np.nan  # L1 keeps only s2 = 0.0
        z.iloc[1, 1] = 9.0     # ...raised to 9.0, so L1 should outrank L0
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        assert self._lipids(out) == ['L1', 'L0']

    def test_no_sort_columns_keeps_input_order(self):
        z = self._z(['PC', 'PC', 'PC'], [1.0, 3.0, 2.0])
        out = LipidomicHeatmapPlotterService.order_by_class(z)
        assert self._lipids(out) == ['L0', 'L1', 'L2']

    def test_unknown_sort_columns_raise(self):
        z = self._z(['PC', 'PC'], [1.0, 2.0])
        with pytest.raises(ValueError, match="sort columns"):
            LipidomicHeatmapPlotterService.order_by_class(
                z, sort_columns=['nope'],
            )

    def test_no_rows_lost(self):
        z = self._z(['PC', 'TG', 'PE', 'PC'], [1.0, 2.0, 3.0, 4.0])
        out = LipidomicHeatmapPlotterService.order_by_class(
            z, sort_columns=['s1', 's2'],
        )
        assert set(out.index) == set(z.index)
        assert len(out) == len(z)


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
# ═══════════════════════════════════════════════════════════════════════
# TestClassAggregatedZScores
# ═══════════════════════════════════════════════════════════════════════

class TestClassAggregatedZScores:
    @staticmethod
    def _filtered():
        """Two PC species and one PE species across four samples."""
        return pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(18:1)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'concentration[s1]': [10.0, 20.0, 5.0],
            'concentration[s2]': [20.0, 40.0, 5.0],
            'concentration[s3]': [30.0, 60.0, 5.0],
            'concentration[s4]': [40.0, 80.0, 5.0],
        })

    def test_one_row_per_class(self):
        result = LipidomicHeatmapPlotterService.compute_class_z_scores(
            self._filtered(),
        )
        assert list(result.index) == ['PC', 'PE']

    def test_species_are_summed_within_class(self):
        """PC row must standardise 30/60/90/120, i.e. the summed species."""
        result = LipidomicHeatmapPlotterService.compute_class_z_scores(
            self._filtered(),
        )
        totals = pd.Series([30.0, 60.0, 90.0, 120.0])
        expected = (totals - totals.mean()) / totals.std()
        assert result.loc['PC'].to_numpy() == pytest.approx(expected.to_numpy())

    def test_rows_have_zero_mean(self):
        result = LipidomicHeatmapPlotterService.compute_class_z_scores(
            self._filtered(),
        )
        assert result.loc['PC'].mean() == pytest.approx(0.0)

    def test_rows_have_unit_std(self):
        result = LipidomicHeatmapPlotterService.compute_class_z_scores(
            self._filtered(),
        )
        assert result.loc['PC'].std() == pytest.approx(1.0)

    def test_constant_class_produces_nan(self):
        """PE is flat across samples, so its Z-scores are undefined."""
        result = LipidomicHeatmapPlotterService.compute_class_z_scores(
            self._filtered(),
        )
        assert result.loc['PE'].isna().all()

    def test_differs_from_species_level_z_scores(self):
        """Aggregating then standardising is not the same as standardising
        each species, which is the whole point of this mode."""
        filtered = self._filtered()
        class_z = LipidomicHeatmapPlotterService.compute_class_z_scores(filtered)
        species_z = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        assert len(class_z) == 2
        assert len(species_z) == 3

    def test_columns_preserved(self):
        result = LipidomicHeatmapPlotterService.compute_class_z_scores(
            self._filtered(),
        )
        assert list(result.columns) == [
            'concentration[s1]', 'concentration[s2]',
            'concentration[s3]', 'concentration[s4]',
        ]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LipidomicHeatmapPlotterService.compute_class_z_scores(pd.DataFrame())

    def test_no_concentration_columns_raises(self):
        df = pd.DataFrame({'LipidMolec': ['A'], 'ClassKey': ['PC']})
        with pytest.raises(ValueError, match="No concentration columns"):
            LipidomicHeatmapPlotterService.compute_class_z_scores(df)


# ═══════════════════════════════════════════════════════════════════════
# TestClassAggregatedHeatmap
# ═══════════════════════════════════════════════════════════════════════

class TestClassAggregatedHeatmap:
    @staticmethod
    def _class_z(classes=('PC', 'PE', 'TG')):
        return pd.DataFrame(
            np.arange(len(classes) * 3, dtype=float).reshape(-1, 3),
            index=pd.Index(list(classes), name='ClassKey'),
            columns=['s1', 's2', 's3'],
        )

    def test_returns_figure_with_heatmap(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
        )
        assert isinstance(fig, go.Figure)
        assert [t for t in fig.data if isinstance(t, go.Heatmap)]

    def test_one_row_per_class(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert list(heatmap.y) == ['PC', 'PE', 'TG']
        assert np.asarray(heatmap.z).shape == (3, 3)

    def test_y_axis_titled_for_classes(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
        )
        assert fig.layout.yaxis.title.text == 'Lipid Classes'

    def test_cells_are_square(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
        )
        layout = fig.layout
        width = (layout.width - layout.margin.l - layout.margin.r) / 3
        height = (layout.height - layout.margin.t - layout.margin.b) / 3
        assert width == height == cell_size(3)

    def test_condition_strip_drawn(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
            sample_conditions=['A', 'A', 'B'],
        )
        assert len([s for s in fig.layout.shapes if s.type == 'rect']) == 2

    def test_symmetric_colorscale(self):
        z = self._class_z()
        z.iloc[0, 0] = -9.0
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            z, ['s1', 's2', 's3'],
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.zmin == -heatmap.zmax

    def test_figure_serializes(self):
        assert LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
        ).to_json()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
                pd.DataFrame(), ['s1'],
            )


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# TestCellSize
# ═══════════════════════════════════════════════════════════════════════

class TestCellSize:
    def test_species_sized_heatmaps_use_the_base_cell(self):
        assert cell_size(40) == CELL_SIZE_PX
        assert cell_size(200) == CELL_SIZE_PX

    def test_short_heatmaps_get_larger_cells(self):
        """A 6-class aggregated map would be a sliver at 18px."""
        assert cell_size(6) > CELL_SIZE_PX

    def test_growth_is_capped(self):
        assert cell_size(1) == MAX_CELL_SIZE_PX
        assert cell_size(2) == MAX_CELL_SIZE_PX

    def test_never_below_the_base_cell(self):
        assert all(cell_size(n) >= CELL_SIZE_PX for n in range(1, 500))

    def test_monotonically_non_increasing(self):
        sizes = [cell_size(n) for n in range(1, 100)]
        assert all(a >= b for a, b in zip(sizes, sizes[1:]))

    def test_zero_rows_is_safe(self):
        assert cell_size(0) == CELL_SIZE_PX


# ═══════════════════════════════════════════════════════════════════════
# TestSquareCells
#
# Only the two class-oriented modes are square-celled. Clustered and Regular
# keep their original fixed-canvas layout.
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
    def _plot_area(fig):
        layout = fig.layout
        return (
            layout.width - layout.margin.l - layout.margin.r,
            layout.height - layout.margin.t - layout.margin.b,
        )

    @pytest.mark.parametrize('n_rows', [2, 6, 20, 60])
    def test_class_grouped_cells_are_square(self, n_rows):
        samples = ['s1', 's2', 's3']
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(n_rows=n_rows), samples,
        )
        width, height = self._plot_area(fig)
        assert width / len(samples) == height / n_rows == cell_size(n_rows)

    @pytest.mark.parametrize('n_rows', [2, 6, 20])
    def test_class_aggregated_cells_are_square(self, n_rows):
        classes = [f'C{i}' for i in range(n_rows)]
        class_z = pd.DataFrame(
            np.arange(n_rows * 3, dtype=float).reshape(n_rows, 3),
            index=pd.Index(classes, name='ClassKey'),
            columns=['s1', 's2', 's3'],
        )
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            class_z, ['s1', 's2', 's3'],
        )
        width, height = self._plot_area(fig)
        assert width / 3 == height / n_rows == cell_size(n_rows)

    def test_height_scales_with_species_count(self):
        """At a fixed cell size, twice the rows is twice the plot height."""
        small = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(n_rows=40), ['s1', 's2', 's3'],
        )
        large = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(n_rows=80), ['s1', 's2', 's3'],
        )
        assert self._plot_area(large)[1] == 2 * self._plot_area(small)[1]

    def test_width_scales_with_sample_count(self):
        narrow = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(n_cols=3), ['s1', 's2', 's3'],
        )
        wide = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(n_cols=9), [f's{i+1}' for i in range(9)],
        )
        assert wide.layout.width - narrow.layout.width == 6 * cell_size(4)

    def test_regular_keeps_its_stretched_layout(self):
        """Regular mode still stretches to the container width; it only
        gained a fixed height so the condition strip has a known canvas."""
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            self._z(), ['s1', 's2', 's3'],
        )
        assert fig.layout.width is None
        assert fig.layout.height == HEATMAP_HEIGHT
        assert fig.layout.margin.l == 10

    def test_clustered_keeps_original_fixed_canvas(self):
        fig = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
            self._z(), ['s1', 's2', 's3'], 2,
        )
        assert fig.layout.width == 900
        assert fig.layout.height == 600


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

    def _fig(self, conditions=None):
        return LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(), self.SAMPLES,
            sample_conditions=self.CONDITIONS if conditions is None else conditions,
        )

    def test_one_block_per_condition(self):
        rects = [s for s in self._fig().layout.shapes if s.type == 'rect']
        assert len(rects) == 2

    def test_blocks_span_their_samples(self):
        rects = [s for s in self._fig().layout.shapes if s.type == 'rect']
        assert (rects[0].x0, rects[0].x1) == (-0.5, 2.5)
        assert (rects[1].x0, rects[1].x1) == (2.5, 5.5)

    def test_blocks_sit_above_the_plot_area(self):
        for rect in [s for s in self._fig().layout.shapes if s.type == 'rect']:
            assert rect.yref == 'paper'
            assert rect.yanchor == 1
            assert rect.y0 > 0

    def test_separator_between_conditions(self):
        lines = [
            s for s in self._fig().layout.shapes
            if s.type == 'line' and s.y0 == 0 and s.y1 == 1
        ]
        assert len(lines) == 1
        assert lines[0].x0 == 2.5

    def test_each_block_is_labelled_directly(self):
        """Blocks are named in place rather than through a legend, so the
        figure stays readable however few rows it has."""
        fig = self._fig()
        labels = [a.text for a in fig.layout.annotations]
        assert 'Control' in labels
        assert 'Treat' in labels
        assert fig.layout.showlegend is False

    def test_labels_stay_black_on_the_fixed_white_background(self):
        fig = self._fig()
        assert fig.layout.paper_bgcolor == 'white'
        assert all(a.font.color == 'black' for a in fig.layout.annotations)

    def test_blocks_use_shared_condition_palette(self):
        from app.services.plotting._shared import generate_condition_color_mapping
        expected = generate_condition_color_mapping(['Control', 'Treat'])
        rects = [s for s in self._fig().layout.shapes if s.type == 'rect']
        assert rects[0].fillcolor == expected['Control']
        assert rects[1].fillcolor == expected['Treat']

    def test_absent_when_conditions_not_supplied(self):
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            self._z(), self.SAMPLES,
        )
        assert not [s for s in fig.layout.shapes if s.type == 'rect']

    def test_non_contiguous_conditions_get_separate_blocks(self):
        """A condition split across the axis must not be merged into one block."""
        fig = self._fig(conditions=['A', 'A', 'B', 'B', 'A', 'A'])
        assert len([s for s in fig.layout.shapes if s.type == 'rect']) == 3

    def test_single_condition_has_no_separator(self):
        fig = self._fig(conditions=['A'] * 6)
        assert not [
            s for s in fig.layout.shapes
            if s.type == 'line' and s.y0 == 0 and s.y1 == 1
        ]

    def test_clustered_and_regular_absent_when_conditions_not_supplied(self):
        for fig in (
            LipidomicHeatmapPlotterService.generate_regular_heatmap(
                self._z(), self.SAMPLES,
            ),
            LipidomicHeatmapPlotterService.generate_clustered_heatmap(
                self._z(), self.SAMPLES, 2,
            ),
        ):
            assert not [s for s in fig.layout.shapes if s.type == 'rect']

    def test_figure_serializes(self):
        """Catches invalid Plotly specs that only surface on render."""
        assert self._fig().to_json()


# ═══════════════════════════════════════════════════════════════════════
# TestSpeciesPaging
# ═══════════════════════════════════════════════════════════════════════

class TestCountSpecies:
    @staticmethod
    def _df():
        return pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c', 'd'],
            'ClassKey': ['PC', 'PC', 'PE', 'TG'],
        })

    def test_counts_selected_classes(self):
        assert LipidomicHeatmapPlotterService.count_species(
            self._df(), ['PC'],
        ) == 2

    def test_counts_across_several_classes(self):
        assert LipidomicHeatmapPlotterService.count_species(
            self._df(), ['PC', 'TG'],
        ) == 3

    def test_unknown_class_counts_zero(self):
        assert LipidomicHeatmapPlotterService.count_species(
            self._df(), ['NOPE'],
        ) == 0

    def test_empty_selection_counts_zero(self):
        assert LipidomicHeatmapPlotterService.count_species(self._df(), []) == 0

    def test_empty_frame_is_safe(self):
        assert LipidomicHeatmapPlotterService.count_species(
            pd.DataFrame(), ['PC'],
        ) == 0

    def test_missing_classkey_column_is_safe(self):
        df = pd.DataFrame({'LipidMolec': ['a']})
        assert LipidomicHeatmapPlotterService.count_species(df, ['PC']) == 0


class TestPageBounds:
    def test_first_page(self):
        assert LipidomicHeatmapPlotterService.page_bounds(
            GROUPED_PAGE_SIZE * 3, 0,
        ) == (0, GROUPED_PAGE_SIZE)

    def test_middle_page(self):
        assert LipidomicHeatmapPlotterService.page_bounds(
            GROUPED_PAGE_SIZE * 3, 1,
        ) == (GROUPED_PAGE_SIZE, GROUPED_PAGE_SIZE * 2)

    def test_last_page_is_truncated_to_the_total(self):
        total = GROUPED_PAGE_SIZE + 10
        assert LipidomicHeatmapPlotterService.page_bounds(total, 1) == (
            GROUPED_PAGE_SIZE, total,
        )

    def test_page_past_the_end_clamps_to_the_last(self):
        total = GROUPED_PAGE_SIZE + 10
        assert LipidomicHeatmapPlotterService.page_bounds(total, 50) == (
            GROUPED_PAGE_SIZE, total,
        )

    def test_negative_page_clamps_to_the_first(self):
        assert LipidomicHeatmapPlotterService.page_bounds(
            GROUPED_PAGE_SIZE * 2, -3,
        ) == (0, GROUPED_PAGE_SIZE)

    def test_total_smaller_than_a_page(self):
        assert LipidomicHeatmapPlotterService.page_bounds(20, 0) == (0, 20)
        assert LipidomicHeatmapPlotterService.page_bounds(20, 9) == (0, 20)

    def test_smaller_page_size(self):
        assert LipidomicHeatmapPlotterService.page_bounds(190, 2, 50) == (100, 150)
        assert LipidomicHeatmapPlotterService.page_bounds(190, 3, 50) == (150, 190)
        assert LipidomicHeatmapPlotterService.page_bounds(190, 9, 50) == (150, 190)

    def test_exact_multiple_has_no_trailing_empty_page(self):
        total = GROUPED_PAGE_SIZE * 2
        assert LipidomicHeatmapPlotterService.page_bounds(total, 2) == (
            GROUPED_PAGE_SIZE, total,
        )

    def test_zero_total(self):
        assert LipidomicHeatmapPlotterService.page_bounds(0, 0) == (0, 0)

    def test_bounds_never_exceed_the_total(self):
        for total in (1, 7, 150, 151, 999):
            for page in range(0, 12):
                start, end = LipidomicHeatmapPlotterService.page_bounds(total, page)
                assert 0 <= start <= end <= total
                assert end - start <= GROUPED_PAGE_SIZE


class TestStripGeometry:
    """The strip is positioned in pixels, not as a fraction of plot height.

    As a fixed paper fraction it thickened with the plot and pushed its own
    labels out of the top margin, so a tall heatmap showed condition colours
    with no condition names. Converting pixels to a paper fraction per figure
    fixed the natural-size render but not a resized one: the PDF report
    re-exports the heatmap on a taller canvas, where the strip grew over the
    title again. Plotly's pixel size mode is immune to the resize.
    """

    @staticmethod
    def _fig(n_rows):
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(n_rows)], ['PC'] * n_rows],
            names=['LipidMolec', 'ClassKey'],
        )
        z = pd.DataFrame(
            np.zeros((n_rows, 4)), index=index,
            columns=['s1', 's2', 's3', 's4'],
        )
        return LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            z, ['s1', 's2', 's3', 's4'],
            sample_conditions=['A', 'A', 'B', 'B'],
        )

    @pytest.mark.parametrize('n_rows', [2, 10, 60, 150])
    def test_strip_thickness_is_constant_in_pixels(self, n_rows):
        fig = self._fig(n_rows)
        rect = [s for s in fig.layout.shapes if s.type == 'rect'][0]
        assert rect.ysizemode == 'pixel'
        assert (rect.y0, rect.y1) == (
            STRIP_GAP_PX, STRIP_GAP_PX + STRIP_HEIGHT_PX,
        )

    @pytest.mark.parametrize('n_rows', [2, 10, 60, 150])
    def test_labels_stay_inside_the_top_margin(self, n_rows):
        fig = self._fig(n_rows)
        for annotation in fig.layout.annotations:
            assert annotation.y == 1
            assert annotation.yshift + STRIP_LABEL_ROW_PX < MARGIN_TOP

    def test_tall_heatmap_still_labels_every_block(self):
        labels = [a.text for a in self._fig(150).layout.annotations]
        assert 'A' in labels and 'B' in labels

    def test_second_label_row_keeps_cells_square(self):
        """A colliding name adds to the top margin, not to the plot area."""
        class_z = pd.DataFrame(
            np.zeros((3, 4)),
            index=pd.Index(['PC', 'PE', 'TG'], name='ClassKey'),
            columns=['s1', 's2', 's3', 's4'],
        )
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            class_z, ['s1', 's2', 's3', 's4'],
            sample_conditions=['CLEANUP_BLANK', 'KNOCKOUT_24H', 'QC', 'QC'],
        )
        layout = fig.layout
        assert layout.margin.t == MARGIN_TOP + 2 * STRIP_LABEL_ROW_PX
        assert layout.height - layout.margin.t - layout.margin.b == 3 * cell_size(3)


# ═══════════════════════════════════════════════════════════════════════
# TestStretchedModeConditionStrip
#
# Clustered and Regular stretch to the container rather than drawing square
# cells, but carry the same condition strip as the class modes.
# ═══════════════════════════════════════════════════════════════════════

class TestStretchedModeConditionStrip:
    SAMPLES = [f's{i+1}' for i in range(7)]
    CONDITIONS = ['WT'] * 3 + ['KO'] * 2 + ['Rescue'] * 2

    @staticmethod
    def _z(n_rows=12):
        rng = np.random.default_rng(0)
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(n_rows)], ['PC'] * n_rows],
            names=['LipidMolec', 'ClassKey'],
        )
        return pd.DataFrame(
            rng.normal(size=(n_rows, 7)), index=index,
            columns=TestStretchedModeConditionStrip.SAMPLES,
        )

    def _figs(self, conditions=None):
        conditions = self.CONDITIONS if conditions is None else conditions
        return {
            'clustered': LipidomicHeatmapPlotterService.generate_clustered_heatmap(
                self._z(), self.SAMPLES, 3, sample_conditions=conditions,
            ),
            'regular': LipidomicHeatmapPlotterService.generate_regular_heatmap(
                self._z(), self.SAMPLES, sample_conditions=conditions,
            ),
        }

    @staticmethod
    def _separators(fig):
        return [
            s for s in fig.layout.shapes
            if s.type == 'line' and s.yref == 'paper'
        ]

    def test_one_labelled_block_per_condition(self):
        for mode, fig in self._figs().items():
            rects = [s for s in fig.layout.shapes if s.type == 'rect']
            assert [(r.x0, r.x1) for r in rects] == [
                (-0.5, 2.5), (2.5, 4.5), (4.5, 6.5),
            ], mode
            assert [a.text for a in fig.layout.annotations] == [
                'WT', 'KO', 'Rescue',
            ], mode

    def test_separators_run_the_full_plot_height(self):
        for mode, fig in self._figs().items():
            separators = self._separators(fig)
            assert [s.x0 for s in separators] == [2.5, 4.5], mode
            for s in separators:
                assert (s.y0, s.y1) == (0, 1), mode
                assert s.line.dash is None, mode

    def test_clustered_keeps_its_dashed_cluster_boundaries(self):
        fig = self._figs()['clustered']
        dashed = [
            s for s in fig.layout.shapes
            if s.type == 'line' and s.line.dash == 'dash'
        ]
        assert len(dashed) == 2
        assert all(s.yref != 'paper' for s in dashed)

    def test_strip_is_pixel_sized_above_the_plot(self):
        for mode, fig in self._figs().items():
            for rect in [s for s in fig.layout.shapes if s.type == 'rect']:
                assert rect.yref == 'paper' and rect.yanchor == 1, mode
                assert rect.ysizemode == 'pixel', mode
                assert (rect.y0, rect.y1) == (
                    STRIP_GAP_PX, STRIP_GAP_PX + STRIP_HEIGHT_PX,
                ), mode

    def test_labels_clear_the_title(self):
        """The title sits at the vertical centre of the top margin, so every
        condition name must top out below that line."""
        long_names = ['CLEANUP_BLANK', 'KNOCKOUT_24H'] + ['QC'] * 5
        for conditions in (self.CONDITIONS, long_names):
            for mode, fig in self._figs(conditions).items():
                top = fig.layout.margin.t
                for a in fig.layout.annotations:
                    assert a.yshift + STRIP_LABEL_ROW_PX < top / 2, mode

    def test_colliding_names_get_a_second_row_and_more_margin(self):
        long_names = ['CLEANUP_BLANK', 'KNOCKOUT_24H'] + ['QC'] * 5
        for mode, fig in self._figs(long_names).items():
            shifts = [a.yshift for a in fig.layout.annotations]
            assert shifts[1] == shifts[0] + STRIP_LABEL_ROW_PX, mode
            assert fig.layout.margin.t == MARGIN_TOP + 2 * STRIP_LABEL_ROW_PX, mode

    def test_short_names_stay_on_one_row(self):
        for mode, fig in self._figs().items():
            assert len({a.yshift for a in fig.layout.annotations}) == 1, mode
            assert fig.layout.margin.t == MARGIN_TOP, mode

    def test_x_axis_pinned_to_the_columns(self):
        """Autorange would widen the axis to fit a name overhanging the last
        block, leaving an empty sliver beside the heatmap."""
        for mode, fig in self._figs().items():
            assert tuple(fig.layout.xaxis.range) == (-0.5, 6.5), mode

    def test_canvas_size_unchanged(self):
        figs = self._figs()
        assert (figs['clustered'].layout.width,
                figs['clustered'].layout.height) == (900, HEATMAP_HEIGHT)
        assert figs['regular'].layout.width is None

    @pytest.mark.parametrize('conditions', [None, []])
    def test_no_strip_without_conditions(self, conditions):
        for fig in (
            LipidomicHeatmapPlotterService.generate_clustered_heatmap(
                self._z(), self.SAMPLES, 3, sample_conditions=conditions,
            ),
            LipidomicHeatmapPlotterService.generate_regular_heatmap(
                self._z(), self.SAMPLES, sample_conditions=conditions,
            ),
        ):
            assert not [s for s in fig.layout.shapes if s.type == 'rect']
            assert not self._separators(fig)
            assert not fig.layout.annotations

    def test_labels_follow_the_theme_text_colour(self):
        """These figures leave their background to the Streamlit theme, so
        black names would vanish on its dark background; they inherit the
        figure's font colour instead, as the title does."""
        for mode, fig in self._figs().items():
            assert fig.layout.paper_bgcolor is None, mode
            for a in fig.layout.annotations:
                assert a.font.color is None, mode

    def test_figures_serialize(self):
        for fig in self._figs().values():
            assert fig.to_json()


class TestLabelRows:
    def test_names_that_fit_share_one_row(self):
        blocks = _condition_blocks(['A'] * 3 + ['B'] * 3)
        assert _label_rows(blocks, column_px=20) == [0, 0]

    def test_a_colliding_name_moves_up(self):
        blocks = _condition_blocks(['CLEANUP_BLANK', 'KNOCKOUT_24H'])
        assert _label_rows(blocks, column_px=20) == [0, 1]

    def test_rows_alternate_through_a_run_of_collisions(self):
        blocks = _condition_blocks(
            ['CLEANUP_BLANK', 'KNOCKOUT_24H', 'VEHICLE_ONLY', 'RESCUE_48H'],
        )
        assert _label_rows(blocks, column_px=60) == [0, 1, 0, 1]

    @pytest.mark.parametrize('conditions', [
        ['WT_CONTROL'] * 24 + ['CLEANUP_BLANK'] * 2
        + ['KNOCKOUT_24H'] * 2 + ['VEHICLE_ONLY'] * 2,
        ['CONTROL'] * 6 + ['TREATED'] * 6 + ['VEHICLE'] * 6
        + ['KNOCKOUT_24H'] * 3 + ['KNOCKOUT_48H'] * 3
        + ['RESCUE_24H'] * 3 + ['RESCUE_48H'] * 3,
    ])
    def test_names_sharing_a_row_never_overlap(self, conditions):
        """Narrow neighbouring blocks with long names used to all land on
        the second row, printed over one another."""
        column_px = STRETCHED_PLOT_WIDTH_PX / len(conditions)
        blocks = _condition_blocks(conditions)
        extents = {}
        for (name, start, end), row in zip(blocks, _label_rows(blocks, column_px)):
            centre = (start + end + 1) / 2 * column_px
            half = len(name) * STRIP_LABEL_PX_PER_CHAR / 2
            extents.setdefault(row, []).append((centre - half, centre + half))
        for spans in extents.values():
            for (_, right), (left, _) in zip(spans, spans[1:]):
                assert left >= right + STRIP_LABEL_GAP_PX

    def test_every_extra_row_buys_top_margin(self):
        conditions = (
            ['WT_CONTROL'] * 24 + ['CLEANUP_BLANK'] * 2
            + ['KNOCKOUT_24H'] * 2 + ['VEHICLE_ONLY'] * 2
        )
        samples = [f's{i+1}' for i in range(30)]
        index = pd.MultiIndex.from_arrays(
            [['L0', 'L1'], ['PC', 'PC']], names=['LipidMolec', 'ClassKey'],
        )
        z = pd.DataFrame(
            np.arange(60, dtype=float).reshape(2, 30),
            index=index, columns=samples,
        )
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(
            z, samples, sample_conditions=conditions,
        )
        top_row = max(a.yshift for a in fig.layout.annotations)
        assert top_row + STRIP_LABEL_ROW_PX < fig.layout.margin.t / 2

    def test_wider_columns_need_no_second_row(self):
        blocks = _condition_blocks(['CLEANUP_BLANK', 'KNOCKOUT_24H'])
        assert _label_rows(blocks, column_px=200) == [0, 0]

    @pytest.mark.parametrize('mode', ['class_grouped', 'class_aggregated'])
    def test_class_modes_allow_for_a_narrowed_figure(self, mode):
        """Regression: the class modes judged collisions at their full 18px
        cell, but Streamlit narrows a figure wider than the page, so on the
        JYK design SS48 and SS48_JW overlapped. They are now judged at the
        same width as the stretched modes, putting SS48_JW on a second row."""
        sizes = {'ID': 6, 'Blank': 4, 'QC': 5, '30Perc': 3, 'CM': 3,
                 'SS24': 3, 'SS48': 3, 'SS48_JW': 3}
        conditions = [c for c, n in sizes.items() for _ in range(n)]
        samples = [f's{i+1}' for i in range(len(conditions))]
        index = pd.MultiIndex.from_arrays(
            [['L0', 'L1'], ['PC', 'PE']], names=['LipidMolec', 'ClassKey'],
        )
        z = pd.DataFrame(
            np.arange(2.0 * len(samples)).reshape(2, -1),
            index=index, columns=samples,
        )
        if mode == 'class_grouped':
            fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
                z, samples, sample_conditions=conditions,
            )
        else:
            fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
                z.droplevel('LipidMolec'), samples, sample_conditions=conditions,
            )
        shift = {a.text: a.yshift for a in fig.layout.annotations}
        assert shift['SS48_JW'] == shift['SS48'] + STRIP_LABEL_ROW_PX
        assert shift['ID'] == shift['SS48']


# ═══════════════════════════════════════════════════════════════════════
# TestMissingSampleColumns
#
# filter_data used to drop concentration columns it could not find while
# still returning every selected sample. The heatmap then received more x
# labels than data columns, and Plotly silently shifted them: with s2's
# column absent, column 1 was labelled "s2" but held s3's data, and so on
# down the axis. A desync between the data and the experiment config must
# fail loudly instead.
# ═══════════════════════════════════════════════════════════════════════

class TestMissingSampleColumns:
    @staticmethod
    def _experiment():
        return make_experiment(n_conditions=1, samples_per_condition=4)

    @staticmethod
    def _df_without(sample):
        experiment = make_experiment(n_conditions=1, samples_per_condition=4)
        data = {'LipidMolec': ['PC(16:0)', 'PE(18:0)'], 'ClassKey': ['PC', 'PE']}
        for s in experiment.full_samples_list:
            if s != sample:
                data[f'concentration[{s}]'] = [10.0, 20.0]
        return pd.DataFrame(data)

    def test_missing_middle_sample_raises(self):
        with pytest.raises(ValueError, match="out of sync"):
            LipidomicHeatmapPlotterService.filter_data(
                self._df_without('s2'), ['Control'], ['PC', 'PE'],
                self._experiment(),
            )

    def test_error_names_the_missing_sample(self):
        with pytest.raises(ValueError, match="s3"):
            LipidomicHeatmapPlotterService.filter_data(
                self._df_without('s3'), ['Control'], ['PC', 'PE'],
                self._experiment(),
            )

    def test_missing_last_sample_raises(self):
        """The truncating case: without the check the column count merely
        shrinks, so nothing looks wrong at a glance."""
        with pytest.raises(ValueError, match="out of sync"):
            LipidomicHeatmapPlotterService.filter_data(
                self._df_without('s4'), ['Control'], ['PC', 'PE'],
                self._experiment(),
            )

    def test_all_columns_missing_still_raises(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'], 'ClassKey': ['PC'],
            'concentration[other]': [1.0],
        })
        with pytest.raises(ValueError, match="No concentration columns"):
            LipidomicHeatmapPlotterService.filter_data(
                df, ['Control'], ['PC'], self._experiment(),
            )

    def test_complete_data_is_unaffected(self):
        """The happy path must not change: all four samples, in order."""
        experiment = self._experiment()
        data = {'LipidMolec': ['PC(16:0)'], 'ClassKey': ['PC']}
        for s in experiment.full_samples_list:
            data[f'concentration[{s}]'] = [1.0]

        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            pd.DataFrame(data), ['Control'], ['PC'], experiment,
        )
        assert samples == experiment.full_samples_list
        assert [c for c in filtered.columns if c.startswith('concentration[')] == [
            f'concentration[{s}]' for s in experiment.full_samples_list
        ]

    def test_labels_can_no_longer_outnumber_the_data(self):
        """The invariant the bug broke: one x label per data column."""
        experiment = self._experiment()
        data = {'LipidMolec': ['PC(16:0)', 'PE(18:0)'], 'ClassKey': ['PC', 'PE']}
        for s in experiment.full_samples_list:
            data[f'concentration[{s}]'] = [10.0, 20.0]

        filtered, samples = LipidomicHeatmapPlotterService.filter_data(
            pd.DataFrame(data), ['Control'], ['PC', 'PE'], experiment,
        )
        z = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        fig = LipidomicHeatmapPlotterService.generate_regular_heatmap(z, samples)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert np.asarray(heatmap.z).shape[1] == len(heatmap.x)


# ═══════════════════════════════════════════════════════════════════════
# TestLog2FoldChange
#
# Z-scores place each sample within its own row's spread, so a class that
# is merely higher than its row mean reads red even when it is unchanged —
# the confusion this mode exists to remove. Fold change is taken against
# the control condition, so control reads zero and colour is direction of
# change versus control.
# ═══════════════════════════════════════════════════════════════════════

class TestLog2FoldChange:
    CONTROL = ['s1', 's2']

    @staticmethod
    def _filtered():
        """CL halves, PC doubles, PE is unchanged, across 2 control + 2 test."""
        return pd.DataFrame({
            'LipidMolec': ['CL 70:6', 'PC 34:1', 'PE 34:1'],
            'ClassKey': ['CL', 'PC', 'PE'],
            'concentration[s1]': [100.0, 10.0, 50.0],
            'concentration[s2]': [100.0, 10.0, 50.0],
            'concentration[s3]': [50.0, 20.0, 50.0],
            'concentration[s4]': [50.0, 20.0, 50.0],
        })

    def test_control_columns_are_zero(self):
        """The whole point: control must read as no change, not as red."""
        fc = LipidomicHeatmapPlotterService.compute_log2fc(
            self._filtered(), self.CONTROL,
        )
        control_cols = ['concentration[s1]', 'concentration[s2]']
        assert fc[control_cols].to_numpy() == pytest.approx(0.0)

    def test_halving_is_minus_one(self):
        fc = LipidomicHeatmapPlotterService.compute_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert fc.loc[('CL 70:6', 'CL'), 'concentration[s3]'] == pytest.approx(-1.0)

    def test_doubling_is_plus_one(self):
        fc = LipidomicHeatmapPlotterService.compute_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert fc.loc[('PC 34:1', 'PC'), 'concentration[s4]'] == pytest.approx(1.0)

    def test_unchanged_is_zero(self):
        fc = LipidomicHeatmapPlotterService.compute_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert fc.loc[('PE 34:1', 'PE'), 'concentration[s3]'] == pytest.approx(0.0)

    def test_differs_from_z_scores_on_the_confusing_case(self):
        """A row that is flat in control but lower in test is positive under
        Z-scores and zero under fold change."""
        filtered = self._filtered()
        z = LipidomicHeatmapPlotterService.compute_z_scores(filtered)
        fc = LipidomicHeatmapPlotterService.compute_log2fc(filtered, self.CONTROL)
        assert z.loc[('CL 70:6', 'CL'), 'concentration[s1]'] > 0.5
        assert fc.loc[('CL 70:6', 'CL'), 'concentration[s1]'] == pytest.approx(0.0)

    def test_zeros_do_not_produce_infinity(self):
        """A species dropping to zero must stay plottable."""
        df = pd.DataFrame({
            'LipidMolec': ['X 1:0'], 'ClassKey': ['X'],
            'concentration[s1]': [100.0], 'concentration[s2]': [100.0],
            'concentration[s3]': [0.0], 'concentration[s4]': [0.0],
        })
        fc = LipidomicHeatmapPlotterService.compute_log2fc(df, self.CONTROL)
        assert np.isfinite(fc.to_numpy()).all()
        assert fc['concentration[s3]'].iloc[0] < 0

    def test_all_zero_row_is_finite(self):
        df = pd.DataFrame({
            'LipidMolec': ['X 1:0'], 'ClassKey': ['X'],
            'concentration[s1]': [0.0], 'concentration[s2]': [0.0],
            'concentration[s3]': [0.0], 'concentration[s4]': [0.0],
        })
        fc = LipidomicHeatmapPlotterService.compute_log2fc(df, self.CONTROL)
        assert np.isfinite(fc.to_numpy()).all()

    def test_shape_and_index_preserved(self):
        fc = LipidomicHeatmapPlotterService.compute_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert fc.shape == (3, 4)
        assert list(fc.index.get_level_values('LipidMolec')) == [
            'CL 70:6', 'PC 34:1', 'PE 34:1',
        ]

    def test_unknown_control_samples_raise(self):
        with pytest.raises(ValueError, match="control condition"):
            LipidomicHeatmapPlotterService.compute_log2fc(
                self._filtered(), ['nope'],
            )

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError, match="empty"):
            LipidomicHeatmapPlotterService.compute_log2fc(
                pd.DataFrame(), self.CONTROL,
            )

    def test_each_row_uses_its_own_control_mean(self):
        """Rows are independent: one row's scale must not affect another's."""
        df = pd.DataFrame({
            'LipidMolec': ['big', 'small'], 'ClassKey': ['A', 'B'],
            'concentration[s1]': [1000.0, 2.0],
            'concentration[s2]': [1000.0, 2.0],
            'concentration[s3]': [2000.0, 4.0],
            'concentration[s4]': [2000.0, 4.0],
        })
        fc = LipidomicHeatmapPlotterService.compute_log2fc(df, self.CONTROL)
        assert fc['concentration[s3]'].to_numpy() == pytest.approx([1.0, 1.0])


class TestClassLog2FoldChange:
    CONTROL = ['s1', 's2']

    @staticmethod
    def _filtered():
        """Two PC species that both double, one PE species that halves."""
        return pd.DataFrame({
            'LipidMolec': ['PC a', 'PC b', 'PE a'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'concentration[s1]': [10.0, 30.0, 80.0],
            'concentration[s2]': [10.0, 30.0, 80.0],
            'concentration[s3]': [20.0, 60.0, 40.0],
            'concentration[s4]': [20.0, 60.0, 40.0],
        })

    def test_one_row_per_class(self):
        fc = LipidomicHeatmapPlotterService.compute_class_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert list(fc.index) == ['PC', 'PE']

    def test_control_is_zero(self):
        fc = LipidomicHeatmapPlotterService.compute_class_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert fc[['concentration[s1]', 'concentration[s2]']].to_numpy() == (
            pytest.approx(0.0)
        )

    def test_class_totals_drive_the_fold_change(self):
        """PC totals 40 -> 80, so the class row is +1 even though the two
        member species differ in size."""
        fc = LipidomicHeatmapPlotterService.compute_class_log2fc(
            self._filtered(), self.CONTROL,
        )
        assert fc.loc['PC', 'concentration[s3]'] == pytest.approx(1.0)
        assert fc.loc['PE', 'concentration[s3]'] == pytest.approx(-1.0)

    def test_matches_summing_then_folding(self):
        """Aggregation must happen on concentrations, before the ratio."""
        filtered = self._filtered()
        fc = LipidomicHeatmapPlotterService.compute_class_log2fc(
            filtered, self.CONTROL,
        )
        totals = filtered.groupby('ClassKey')[
            [c for c in filtered.columns if c.startswith('concentration[')]
        ].sum()
        expected = np.log2(
            totals['concentration[s3]']
            / totals[['concentration[s1]', 'concentration[s2]']].mean(axis=1)
        )
        assert fc['concentration[s3]'].to_numpy() == pytest.approx(
            expected.to_numpy()
        )

    def test_unknown_control_samples_raise(self):
        with pytest.raises(ValueError, match="control condition"):
            LipidomicHeatmapPlotterService.compute_class_log2fc(
                self._filtered(), ['nope'],
            )


class TestValueLabel:
    @staticmethod
    def _class_z():
        return pd.DataFrame(
            np.arange(9, dtype=float).reshape(3, 3),
            index=pd.Index(['PC', 'PE', 'TG'], name='ClassKey'),
            columns=['s1', 's2', 's3'],
        )

    def test_colorbar_defaults_to_z_score(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'],
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.colorbar.title.text == 'Z-score'

    def test_colorbar_follows_the_value_label(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'], value_label='log2FC',
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.colorbar.title.text == 'log2FC'

    def test_title_names_the_quantity(self):
        fig = LipidomicHeatmapPlotterService.generate_class_aggregated_heatmap(
            self._class_z(), ['s1', 's2', 's3'], value_label='log2FC',
        )
        assert 'log2FC' in fig.layout.title.text

    def test_grouped_mode_takes_the_label_too(self):
        index = pd.MultiIndex.from_arrays(
            [['L0', 'L1'], ['PC', 'PE']], names=['LipidMolec', 'ClassKey'],
        )
        z = pd.DataFrame(
            np.zeros((2, 3)), index=index, columns=['s1', 's2', 's3'],
        )
        fig = LipidomicHeatmapPlotterService.generate_class_grouped_heatmap(
            z, ['s1', 's2', 's3'], value_label='log2FC',
        )
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.colorbar.title.text == 'log2FC'
        assert 'log2FC' in fig.layout.title.text


# ═══════════════════════════════════════════════════════════════════════
# TestLongSampleLabels
#
# With "Show samples as" on original names, the sample axis carries names
# such as 'CLEANUP_BLANK_01' rather than s1, s2, ... At 45 degrees they ran
# into each other, and the fixed bottom margins of the Clustered (50px) and
# Regular (20px) modes clipped them wherever Plotly does not grow the margin
# itself, as in the PDF report's export.
# ═══════════════════════════════════════════════════════════════════════

class TestLongSampleLabels:
    SHORT = ['s1', 's2', 's3', 's10']
    LONG = ['ID_01', 'CLEANUP_BLANK_01', 'QC (s3)', 'SS48_JW_03']
    CONDITIONS = ['A', 'A', 'B', 'B']

    @staticmethod
    def _z():
        index = pd.MultiIndex.from_arrays(
            [[f'L{i}' for i in range(6)], ['PC'] * 3 + ['PE'] * 3],
            names=['LipidMolec', 'ClassKey'],
        )
        rng = np.random.default_rng(1)
        return pd.DataFrame(rng.normal(size=(6, 4)), index=index)

    def _figs(self, samples):
        z = self._z()
        class_z = z.groupby(level='ClassKey').sum()
        S = LipidomicHeatmapPlotterService
        return {
            'clustered': S.generate_clustered_heatmap(
                z, samples, 2, sample_conditions=self.CONDITIONS),
            'regular': S.generate_regular_heatmap(
                z, samples, sample_conditions=self.CONDITIONS),
            'class_grouped': S.generate_class_grouped_heatmap(
                z, samples, sample_conditions=self.CONDITIONS),
            'class_aggregated': S.generate_class_aggregated_heatmap(
                class_z, samples, sample_conditions=self.CONDITIONS),
        }

    def test_short_labels_keep_the_original_layout(self):
        figs = self._figs(self.SHORT)
        for mode, fig in figs.items():
            assert fig.layout.xaxis.tickangle == 45, mode
            assert fig.layout.xaxis.automargin is None, mode
        assert figs['clustered'].layout.margin.b == 50
        assert figs['regular'].layout.margin.b == 20

    def test_long_labels_stand_upright_with_room_below(self):
        for mode, fig in self._figs(self.LONG).items():
            assert list(fig.data[0].x) == self.LONG, mode
            assert fig.layout.xaxis.tickangle == 90, mode
            assert fig.layout.xaxis.automargin is True, mode
            # 16 characters at PX_PER_CHAR, plus room for the axis title.
            assert fig.layout.margin.b >= 16 * 7 + 40, mode

    def test_square_cells_survive_long_labels(self):
        """The class modes size the figure from the margins, so growing the
        bottom margin must grow the figure, not shrink the cells."""
        fig = self._figs(self.LONG)['class_aggregated']
        layout = fig.layout
        assert (
            layout.height - layout.margin.t - layout.margin.b
        ) == 2 * cell_size(2)
        assert (
            layout.width - layout.margin.l - layout.margin.r
        ) == 4 * cell_size(2)
