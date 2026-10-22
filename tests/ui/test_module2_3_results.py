"""Pins: Quality Check and every analysis page give the numbers the methods
document (docs/supplementary_methods.md) defines.

Each test drives the real app on a sample dataset with normalization off (so
the analysed values are the uploaded intensities), reads what the user
downloads, and recomputes it here from the data the page received, with the
formula the methods give: not with the service that produced it.
"""

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from streamlit.testing.v1 import AppTest

from app.constants import (
    FORMAT_LIPIDSEARCH, MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS, PAGE_APP,
)
from app.ui.main_content.analysis._entry import ANALYSIS_OPTIONS

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')
NO_NORMALIZATION = 'None (pre-normalized data)'

# The Generic sample: WT s1-s4, ADGAT-DKO s5-s8, BQC s9-s12.
SAMPLES = {
    'WT': ['s1', 's2', 's3', 's4'],
    'ADGAT-DKO': ['s5', 's6', 's7', 's8'],
    'BQC': ['s9', 's10', 's11', 's12'],
}
BAR, PIE, SATURATION, CHAIN_LENGTH, FACH, PATHWAY, VOLCANO, HEATMAP = ANALYSIS_OPTIONS


def _ok(at):
    assert not at.exception, at.exception


def _load(data_format=None, method=NO_NORMALIZATION):
    """Sample data → confirm → normalization ``method``, in Module 1."""
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    if data_format is not None:
        at.sidebar.selectbox[0].set_value(data_format).run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    at.radio(key='norm_method_selection').set_value(method).run()
    _ok(at)
    return at


def _next(at):
    [b for b in at.button if b.label.startswith('Next')][0].click().run()
    _ok(at)


def _back(at):
    """Return to Module 1. Clicking Back trips an AppTest artifact (after
    its st.rerun() the element tree still holds Module 2's widgets, whose
    popped keys the next run cannot read); Back only clears the qc stage,
    which the following Next clears again."""
    at.session_state['module'] = MODULE_DATA_PROCESSING
    at.run()
    _ok(at)


def _enter_qc(at):
    """Enter Quality Check without the Next click, whose st.rerun() leaves
    Module 1 widgets in AppTest's tree (LipidSearch's standards picker
    then breaks the next run). Next only clears the qc stage, which is
    empty on first entry."""
    at.session_state['module'] = MODULE_QC_ANALYSIS
    at.run()
    _ok(at)


def _to_analysis():
    at = _load()
    _next(at)
    return at


def _csv(captured, key):
    assert key in captured, f"no download rendered for {key}"
    return pd.read_csv(io.BytesIO(captured[key]))


def _show(at, analysis):
    at.radio(key='analysis_radio').set_value(analysis).run()
    _ok(at)


def _columns(samples):
    return [f'concentration[{s}]' for s in samples]


def _analysed(at):
    """The data the analysis pages received, by lipid."""
    return at.session_state['qc_continuation_df'].set_index('LipidMolec')


def _class_totals(at):
    df = at.session_state['qc_continuation_df']
    return df.groupby('ClassKey')[_columns(sum(SAMPLES.values(), []))].sum()


def _chains(name):
    """Fatty acyl chains as (carbons, double bonds); a sphingoid base
    (d18:1, t18:0, m18:0) is not a fatty acid."""
    return [
        (int(c), int(db))
        for c, db in re.findall(r'(?<![dtm])(\d+):(\d+)', name.split(' ', 1)[1])
    ]


def _coefficients_of_variation(values):
    mean = values.mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(mean > 0, values.std(axis=1, ddof=1) / mean * 100, np.nan)


# =============================================================================
# Module 2: Quality Check
# =============================================================================

class TestQualityCheck:

    def test_missing_values_are_percent_zeros_per_sample(self, captured_downloads):
        at = _to_analysis()
        df = _analysed(at)
        missing = _csv(captured_downloads, 'qc_missing_values_csv')
        all_samples = sum(SAMPLES.values(), [])
        assert missing['Sample'].tolist() == all_samples
        expected = (df[_columns(all_samples)] == 0).mean() * 100
        np.testing.assert_allclose(missing['Percentage Missing'], expected.to_numpy())

    def test_correlation_is_pearson_without_zeros(self, captured_downloads):
        """S4.3: zeros excluded, Pearson between replicates."""
        at = _to_analysis()
        at.selectbox(key='corr_condition').set_value('ADGAT-DKO').run()
        _ok(at)
        matrix = _csv(captured_downloads, 'corr_csv_download').set_index('Sample')
        values = _analysed(at)[_columns(SAMPLES['ADGAT-DKO'])].replace(0, np.nan)
        assert matrix.index.tolist() == SAMPLES['ADGAT-DKO']
        np.testing.assert_allclose(matrix.to_numpy(), values.corr().to_numpy())

    def test_pca_standardizes_each_lipid(self, captured_downloads):
        """S4.4: features centred and scaled to unit variance, two PCs (sign
        is arbitrary)."""
        at = _to_analysis()
        all_samples = sum(SAMPLES.values(), [])
        pca = _csv(captured_downloads, 'pca_csv_download')
        values = _analysed(at)[_columns(all_samples)].T
        expected = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(values))
        assert pca['Sample'].tolist() == all_samples
        np.testing.assert_allclose(
            np.abs(pca[['PC1', 'PC2']].to_numpy()), np.abs(expected), atol=1e-9,
        )

    def test_cov_filter_removes_lipids_at_or_above_threshold(self, captured_downloads):
        """S4.2: CoV = SD / mean × 100 over the BQC samples; lipids at or
        above the threshold leave every analysis, except those kept."""
        at = _to_analysis()
        uploaded = _analysed(at)
        cov = _coefficients_of_variation(
            uploaded[_columns(SAMPLES['BQC'])].to_numpy(float)
        )
        high = set(uploaded.index[cov >= 10])
        assert 0 < len(high) < len(uploaded)

        at.number_input(key='bqc_cov_threshold').set_value(10).run()
        at.radio(key='bqc_filter_choice').set_value('Yes').run()
        kept = sorted(high)[:2]
        at.multiselect(key='bqc_lipids_to_keep').set_value(kept).run()
        _ok(at)

        analysed = set(_analysed(at).index)
        assert set(uploaded.index) - analysed == high - set(kept)
        bar = _csv(captured_downloads, 'analysis_csv_bar').set_index('ClassKey')
        df = at.session_state['qc_continuation_df']
        wt_totals = df.groupby('ClassKey')[_columns(SAMPLES['WT'])].sum()
        np.testing.assert_allclose(
            bar['mean_AUC_WT'], wt_totals.mean(axis=1).loc[bar.index],
        )

    def test_pca_exclusion_reaches_the_analyses(self, captured_downloads):
        at = _to_analysis()
        totals = _class_totals(at)
        at.multiselect(key='pca_samples_remove').set_value(['s2']).run()
        _ok(at)
        bar = _csv(captured_downloads, 'analysis_csv_bar').set_index('ClassKey')
        kept_wt = _columns(['s1', 's3', 's4'])
        np.testing.assert_allclose(
            bar['mean_AUC_WT'], totals[kept_wt].mean(axis=1).loc[bar.index],
        )
        np.testing.assert_allclose(
            bar['mean_AUC_ADGAT-DKO'],
            totals[_columns(SAMPLES['ADGAT-DKO'])].mean(axis=1).loc[bar.index],
        )

    def test_retention_time_plots_each_lipids_mass_and_rt(self, captured_downloads):
        at = _load(FORMAT_LIPIDSEARCH)
        _enter_qc(at)
        df = _analysed(at)
        selected = at.multiselect(key='rt_class_selection').value
        comparison = _csv(captured_downloads, 'rt_csv_comparison').set_index('LipidMolec')
        assert sorted(comparison.index) == sorted(df.index[df['ClassKey'].isin(selected)])
        np.testing.assert_allclose(comparison['Mass'], df.loc[comparison.index, 'CalcMass'])
        np.testing.assert_allclose(comparison['Retention'], df.loc[comparison.index, 'BaseRt'])

        at.radio(key='rt_viewing_mode').set_value('Individual Mode').run()
        _ok(at)
        n_plots = len([k for k in captured_downloads if k and k.startswith('rt_csv_individual_')])
        assert n_plots == df['ClassKey'].nunique()


# =============================================================================
# Module 3: Class-level pages
# =============================================================================

class TestClassLevel:

    def test_bar_chart_mean_and_sd_on_each_scale(self, captured_downloads):
        """S5.3: linear mean/SD of class totals; log10 mean/SD of the
        log10 class totals."""
        at = _to_analysis()
        totals = _class_totals(at)
        bar = _csv(captured_downloads, 'analysis_csv_bar').set_index('ClassKey')
        for condition, samples in SAMPLES.items():
            t = totals[_columns(samples)].loc[bar.index]
            np.testing.assert_allclose(bar[f'mean_AUC_{condition}'], t.mean(axis=1))
            np.testing.assert_allclose(bar[f'std_AUC_{condition}'], t.std(axis=1, ddof=1))
            np.testing.assert_allclose(
                bar[f'log10_mean_AUC_{condition}'], np.log10(t).mean(axis=1),
            )
            np.testing.assert_allclose(
                bar[f'log10_std_AUC_{condition}'], np.log10(t).std(axis=1, ddof=1),
            )

    def test_pie_charts_share_of_class_totals(self, captured_downloads):
        at = _to_analysis()
        _show(at, PIE)
        totals = _class_totals(at)
        for condition, samples in SAMPLES.items():
            pie = _csv(captured_downloads, f'analysis_csv_pie_{condition}').set_index('ClassKey')
            summed = totals[_columns(samples)].sum(axis=1).loc[pie.index]
            np.testing.assert_allclose(pie['Total Abundance'], summed)
            np.testing.assert_allclose(pie['Percentage'], summed / summed.sum() * 100)

    def test_saturation_weights_each_lipid_by_its_chains(self, captured_downloads):
        """S6.6: each lipid's value split by the share of its chains that
        are SFA (0 DB), MUFA (1) and PUFA (≥2), summed per class; mean and
        SD over the condition's samples."""
        at = _to_analysis()
        _show(at, SATURATION)
        df = _analysed(at)
        kinds = {'SFA': lambda db: db == 0, 'MUFA': lambda db: db == 1, 'PUFA': lambda db: db >= 2}
        for lipid_class in ('PC', 'TG', 'Cer', 'LPC'):
            table = _csv(captured_downloads, f'analysis_csv_sat_concentration_{lipid_class}')
            table['Condition'] = table['Condition'].str.replace(r' \(n=\d+\)$', '', regex=True)
            table = table.set_index('Condition')
            species = df[df['ClassKey'] == lipid_class]
            for kind, is_kind in kinds.items():
                share = np.array([
                    np.mean([is_kind(db) for _, db in _chains(name)])
                    for name in species.index
                ])
                for condition, samples in SAMPLES.items():
                    per_sample = species[_columns(samples)].mul(share, axis=0).sum()
                    assert table.loc[condition, f'{kind}_AUC'] == pytest.approx(
                        per_sample.mean()
                    ), (lipid_class, kind, condition)
                    assert table.loc[condition, f'{kind}_std'] == pytest.approx(
                        per_sample.std(ddof=1)
                    ), (lipid_class, kind, condition)

    def test_chain_length_sums_species_of_one_composition(self, captured_downloads):
        at = _to_analysis()
        _show(at, CHAIN_LENGTH)
        df = _analysed(at)
        table = _csv(captured_downloads, 'analysis_csv_chain_length')
        pc = df[df['ClassKey'] == 'PC']
        composition = pd.DataFrame(
            [
                (sum(c for c, _ in _chains(n)), sum(db for _, db in _chains(n)))
                for n in pc.index
            ],
            index=pc.index, columns=['TotalCarbons', 'TotalDoubleBonds'],
        )
        for condition, samples in SAMPLES.items():
            expected = (
                pc[_columns(samples)].sum(axis=1).div(len(samples))
                .groupby([composition['TotalCarbons'], composition['TotalDoubleBonds']]).sum()
            )
            got = table[(table['ClassKey'] == 'PC') & (table['Condition'] == condition)]
            got = got.set_index(['TotalCarbons', 'TotalDoubleBonds'])['MeanConcentration']
            pd.testing.assert_series_equal(
                got.sort_index(), expected.loc[got.index].sort_index(),
                check_names=False,
            )
            assert set(got.index) == set(expected[expected > 0].index)

    def test_fach_is_percent_of_class_total(self, captured_downloads):
        """S6.5: each carbon/double-bond cell is its species' share of the
        class total, in percent."""
        at = _to_analysis()
        _show(at, FACH)
        at.selectbox(key='fach_class').set_value('PC').run()
        _ok(at)
        df = _analysed(at)
        pc = df[df['ClassKey'] == 'PC']
        table = _csv(captured_downloads, 'analysis_csv_fach')
        for condition in at.multiselect(key='fach_conditions').value:
            means = pc[_columns(SAMPLES[condition])].mean(axis=1)
            carbons = [sum(c for c, _ in _chains(n)) for n in pc.index]
            bonds = [sum(db for _, db in _chains(n)) for n in pc.index]
            expected = (means.groupby([carbons, bonds]).sum() / means.sum() * 100)
            got = table[table['Condition'] == condition].set_index(['Carbon', 'DB'])['Proportion']
            np.testing.assert_allclose(got.sort_index(), expected.loc[got.index].sort_index())
            assert got.sum() == pytest.approx(100)

    def test_pathway_fold_change_of_class_totals(self, captured_downloads):
        """S6.7: mean experimental / mean control class total."""
        at = _to_analysis()
        _show(at, PATHWAY)
        control = at.selectbox(key='pathway_control').value
        experimental = at.selectbox(key='pathway_experimental').value
        totals = _class_totals(at)
        pathway = _csv(captured_downloads, 'analysis_csv_pathway').set_index('Lipid Class')
        detected = pathway.index.intersection(totals.index)
        expected = (
            totals[_columns(SAMPLES[experimental])].mean(axis=1)
            / totals[_columns(SAMPLES[control])].mean(axis=1)
        )
        np.testing.assert_allclose(pathway.loc[detected, 'Fold Change'], expected[detected])


# =============================================================================
# Module 3: Species-level pages
# =============================================================================

class TestSpeciesLevel:

    def test_volcano_fold_change_and_welch_on_log10(self, captured_downloads):
        """S5.4: log2 of the ratio of means, each lipid's zeros floored at
        its smallest positive value in the two groups / 10 (as S6.4 says of
        these fold changes); Welch's t-test on log10 values, zeros floored
        at the smallest non-zero value in the data / 10 (S2); BH adjustment
        when chosen."""
        at = _to_analysis()
        _show(at, VOLCANO)
        at.selectbox(key='volcano_correction').set_value('fdr_bh').run()
        _ok(at)
        assert at.selectbox(key='volcano_control').value == 'WT'
        assert at.selectbox(key='volcano_experimental').value == 'ADGAT-DKO'
        df = _analysed(at)
        volcano = _csv(captured_downloads, 'analysis_csv_volcano').set_index('LipidMolec')
        control = df.loc[volcano.index, _columns(SAMPLES['WT'])].astype(float)
        treated = df.loc[volcano.index, _columns(SAMPLES['ADGAT-DKO'])].astype(float)

        both = pd.concat([control, treated], axis=1)
        row_floor = both.where(both > 0).min(axis=1) / 10
        np.testing.assert_allclose(
            volcano['FoldChange'],
            np.log2(
                treated.clip(lower=row_floor, axis=0).mean(axis=1)
                / control.clip(lower=row_floor, axis=0).mean(axis=1)
            ),
        )
        everything = df[_columns(sum(SAMPLES.values(), []))].to_numpy(float)
        data_floor = everything[everything > 0].min() / 10
        p = stats.ttest_ind(
            np.log10(treated.clip(lower=data_floor)),
            np.log10(control.clip(lower=data_floor)),
            axis=1, equal_var=False,
        ).pvalue
        np.testing.assert_allclose(volcano['pValue'], p)
        np.testing.assert_allclose(
            volcano['adjusted_pValue'], multipletests(p, method='fdr_bh')[1],
        )

        excluded = _csv(captured_downloads, 'analysis_csv_excluded').set_index('LipidMolec')
        assert (df.loc[excluded.index, _columns(SAMPLES['ADGAT-DKO'])] == 0).all(axis=None)
        assert len(excluded) + len(volcano) == len(df)

    def test_volcano_detail_lists_each_samples_value(self, captured_downloads):
        at = _to_analysis()
        _show(at, VOLCANO)
        df = _analysed(at)
        detail = _csv(captured_downloads, 'analysis_csv_dist')
        assert set(detail['LipidMolec']) == set(at.multiselect(key='volcano_detail_lipids').value)
        for row in detail.itertuples():
            assert row.Sample in SAMPLES[row.Condition]
            assert row.Concentration == df.loc[row.LipidMolec, f'concentration[{row.Sample}]']

    def test_heatmap_z_scores_per_lipid(self, captured_downloads):
        """S6.2: row-wise (x - mean) / SD."""
        at = _to_analysis()
        _show(at, HEATMAP)
        heatmap = _csv(captured_downloads, 'analysis_csv_heatmap').set_index('LipidMolec')
        columns = _columns(sum(SAMPLES.values(), []))
        raw = _analysed(at).loc[heatmap.index, columns].astype(float)
        z = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1), axis=0)
        np.testing.assert_allclose(heatmap[columns].to_numpy(), z.to_numpy(), atol=1e-12)

    def test_heatmap_class_rows_standardize_class_totals(self, captured_downloads):
        """S6.3: sum species per class and sample, then z-score."""
        at = _to_analysis()
        _show(at, HEATMAP)
        at.radio(key='heatmap_type').set_value('Aggregated by Class').run()
        _ok(at)
        heatmap = _csv(captured_downloads, 'analysis_csv_heatmap').set_index('ClassKey')
        columns = _columns(sum(SAMPLES.values(), []))
        totals = _class_totals(at).loc[heatmap.index, columns]
        z = totals.sub(totals.mean(axis=1), axis=0).div(totals.std(axis=1), axis=0)
        np.testing.assert_allclose(heatmap[columns].to_numpy(), z.to_numpy(), atol=1e-12)

    def test_heatmap_fold_change_against_control_mean(self, captured_downloads):
        """S6.4: each value, floored at the row's smallest positive value /
        10, as log2 of its ratio to the control mean."""
        at = _to_analysis()
        _show(at, HEATMAP)
        at.radio(key='heatmap_type').set_value('Regular').run()
        at.radio(key='heatmap_color_scale').set_value('Log2 Fold Change').run()
        _ok(at)
        assert at.selectbox(key='heatmap_control_condition').value == 'WT'
        heatmap = _csv(captured_downloads, 'analysis_csv_heatmap').set_index('LipidMolec')
        columns = _columns(sum(SAMPLES.values(), []))
        raw = _analysed(at).loc[heatmap.index, columns].astype(float)
        floored = raw.clip(lower=raw.where(raw > 0).min(axis=1) / 10, axis=0)
        control_mean = floored[_columns(SAMPLES['WT'])].mean(axis=1)
        expected = np.log2(floored.div(control_mean, axis=0))
        np.testing.assert_allclose(heatmap[columns].to_numpy(), expected.to_numpy(), atol=1e-12)


# =============================================================================
# Across modules
# =============================================================================

def test_back_and_next_reruns_the_analyses_on_the_new_data(captured_downloads):
    """Back, a different normalization, Next: the analyses use the new
    data, and figures, tests and exclusions from before are gone."""
    at = _to_analysis()
    _show(at, VOLCANO)
    at.multiselect(key='pca_samples_remove').set_value(['s3']).run()
    _show(at, BAR)
    _ok(at)
    _back(at)
    at.radio(key='norm_method_selection').set_value('Total Intensity').run()
    _ok(at)
    captured_downloads.clear()
    _next(at)

    state = at.session_state
    assert sorted(state['analysis_all_plots']) == ['bar_chart']
    assert sorted(state['analysis_statistics']) == ['Abundance bar chart']
    assert at.multiselect(key='pca_samples_remove').value == []
    totals = state['normalized_df'].groupby('ClassKey')[_columns(SAMPLES['WT'])].sum()
    bar = _csv(captured_downloads, 'analysis_csv_bar').set_index('ClassKey')
    np.testing.assert_allclose(bar['mean_AUC_WT'], totals.mean(axis=1).loc[bar.index])
