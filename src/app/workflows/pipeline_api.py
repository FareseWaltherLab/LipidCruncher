"""
The ``lc`` namespace exposed to AI-generated sandbox code.

Gives AI-generated code direct access to LipidCruncher's own pipeline
services, so questions the main pipeline can already answer are answered
*by the pipeline* instead of being re-derived with looser ad-hoc
statistics. The AI is instructed (see ``prompt_templates``) to check this
catalog first and only write from-scratch code when nothing here fits.

Lives in the workflows layer because it delegates to ``AnalysisWorkflow``
for the plots; a service may not import a workflow. The sandbox therefore
receives a constructed instance rather than building one itself.

Pure — no Streamlit. Bound to one DataFrame + ExperimentConfig and
injected into the sandbox globals as ``lc``.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models.experiment import ExperimentConfig
from ..models.statistics import StatisticalTestConfig
from ..services.statistical_testing import StatisticalTestingService


def _sample_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns
            if c.startswith(f'{prefix}[') and c.endswith(']')]


def _as_concentration_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame whose sample columns are named ``concentration[sN]``.

    The statistics services address sample columns as ``concentration[sN]``.
    Data normalized by intensity carries ``intensity[sN]`` instead, so
    rename a copy rather than duplicating the service logic.
    """
    if _sample_columns(df, 'concentration'):
        return df
    intensity_cols = _sample_columns(df, 'intensity')
    if not intensity_cols:
        return df
    rename = {
        c: f"concentration[{c[len('intensity['):-1]}]"
        for c in intensity_cols
    }
    return df.rename(columns=rename)


@dataclass(frozen=True)
class LCResult:
    """Uniform return shape for every ``lc`` plot helper.

    ``figure`` is LipidCruncher's own Plotly figure — assign it to ``fig``
    to display it. ``data`` is the frame behind it. ``raw`` is the
    workflow's native result object for anything else it carries.
    """
    figure: Any = None
    data: Any = None
    raw: Any = None


class PipelineAPI:
    """LipidCruncher pipeline capabilities, callable from sandbox code.

    Exposed to AI-generated code as ``lc``. Every method returns a plain
    pandas DataFrame so the generated code only has to present the result.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        experiment: Optional[ExperimentConfig] = None,
    ):
        self._df = df
        self._experiment = experiment

    def with_dataframe(self, df: pd.DataFrame) -> "PipelineAPI":
        """Return a copy bound to ``df``.

        The sandbox coerces sample columns to numeric before exposing the
        frame; this rebinds ``lc`` to that same prepared copy so both see
        identical data.
        """
        return PipelineAPI(df, self._experiment)

    # -- introspection ------------------------------------------------

    def conditions(self) -> List[str]:
        """Return the experiment's condition names."""
        self._require_experiment()
        return list(self._experiment.conditions_list)

    def samples_for(self, condition: str) -> List[str]:
        """Return the sample names belonging to ``condition``."""
        self._require_experiment()
        names = list(self._experiment.conditions_list)
        if condition not in names:
            raise ValueError(
                f"Unknown condition '{condition}'. "
                f"Available conditions: {names}"
            )
        idx = names.index(condition)
        return list(self._experiment.individual_samples_list[idx])

    # -- statistics ---------------------------------------------------

    def compare_species(
        self,
        control: str,
        experimental: str,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Per-lipid statistical comparison between two conditions.

        Runs LipidCruncher's documented species-level method — Welch's
        t-test on zero-replaced, log10-transformed concentrations with
        Benjamini-Hochberg FDR correction across all lipids. This is the
        same engine behind the volcano plot.

        Args:
            control: Control condition name (e.g. 'WT').
            experimental: Experimental condition name (e.g. 'ADGAT-DKO').
            alpha: Significance threshold.

        Returns:
            DataFrame with one row per tested lipid and columns
            ``LipidMolec``, ``ClassKey``, ``p_value``, ``adj_p_value``,
            ``log2FC`` (experimental / control), ``significant``,
            sorted by ``adj_p_value`` ascending. Lipids the pipeline skips
            (fewer than 2 valid replicates, or all-zero in a condition)
            are absent, exactly as in the main pipeline.
        """
        self._require_experiment()
        frame = _as_concentration_frame(self._df)
        config = StatisticalTestConfig(alpha=alpha)

        summary = StatisticalTestingService.run_species_level_tests(
            df=frame,
            control_samples=self.samples_for(control),
            experimental_samples=self.samples_for(experimental),
            config=config,
        )

        class_by_lipid = {}
        if 'LipidMolec' in frame.columns and 'ClassKey' in frame.columns:
            class_by_lipid = dict(
                zip(frame['LipidMolec'], frame['ClassKey'])
            )

        rows = [{
            'LipidMolec': lipid,
            'ClassKey': class_by_lipid.get(lipid),
            'p_value': r.p_value,
            'adj_p_value': r.adjusted_p_value,
            'log2FC': r.effect_size,
            'significant': r.significant,
        } for lipid, r in summary.results.items()]

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        # Stable sort so tied adjusted p-values keep a deterministic order
        # (matches the eval reference implementation).
        return out.sort_values(
            'adj_p_value', kind='stable'
        ).reset_index(drop=True)

    def compare_classes(
        self,
        conditions: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Class-level statistical comparison across conditions.

        Sums species concentrations per class per sample, then compares
        across the given conditions using LipidCruncher's class-level
        method (the engine behind the abundance bar chart).

        Args:
            conditions: Condition names to compare. Defaults to all.
            classes: Lipid classes to test. Defaults to all in the data.
            alpha: Significance threshold.

        Returns:
            DataFrame with ``ClassKey``, ``p_value``, ``adj_p_value``,
            ``significant``, sorted by ``adj_p_value`` ascending.
        """
        self._require_experiment()
        frame = _as_concentration_frame(self._df)

        if conditions is None:
            conditions = list(self._experiment.conditions_list)
        if classes is None:
            classes = sorted(frame['ClassKey'].dropna().unique().tolist())

        summary = StatisticalTestingService.run_class_level_tests(
            df=frame,
            experiment=self._experiment,
            selected_conditions=conditions,
            selected_classes=classes,
            config=StatisticalTestConfig(alpha=alpha),
        )

        rows = [{
            'ClassKey': key,
            'p_value': r.p_value,
            'adj_p_value': r.adjusted_p_value,
            'significant': r.significant,
        } for key, r in summary.results.items()]

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(
            'adj_p_value', kind='stable'
        ).reset_index(drop=True)

    # -- descriptive --------------------------------------------------

    def class_summary(self) -> pd.DataFrame:
        """Per-class species counts and mean abundance.

        Returns:
            DataFrame with ``ClassKey``, ``n_species``, and
            ``mean_abundance`` (mean across all sample columns of the
            summed class concentration), sorted by ``n_species``
            descending.
        """
        frame = _as_concentration_frame(self._df)
        sample_cols = _sample_columns(frame, 'concentration')
        if 'ClassKey' not in frame.columns:
            raise ValueError(
                "This dataset has no 'ClassKey' column, so per-class "
                "summaries are unavailable."
            )

        grouped = frame.groupby('ClassKey')
        out = pd.DataFrame({
            'ClassKey': list(grouped.groups.keys()),
            'n_species': grouped.size().values,
        })
        if sample_cols:
            out['mean_abundance'] = (
                grouped[sample_cols].sum().mean(axis=1).values
            )
        return out.sort_values(
            'n_species', ascending=False
        ).reset_index(drop=True)

    # -- internals ----------------------------------------------------

    def _require_experiment(self) -> None:
        if self._experiment is None:
            raise ValueError(
                "No experiment is defined for this dataset, so conditions "
                "cannot be compared. Define the conditions and sample "
                "groups in the sidebar first."
            )

    # -- analysis plots -----------------------------------------------
    #
    # Each delegates to AnalysisWorkflow, i.e. the exact code path the main
    # app's own analysis pages use, so a figure produced here matches the
    # built-in one rather than approximating it.

    def _analysis(self):
        from .analysis import AnalysisWorkflow
        return AnalysisWorkflow

    def _conditions_or_all(self, conditions: Optional[List[str]]) -> List[str]:
        self._require_experiment()
        if conditions:
            return list(conditions)
        return self._analysis().get_all_conditions(self._experiment)

    def _classes_or_all(self, classes: Optional[List[str]]) -> List[str]:
        if classes:
            return list(classes)
        return self._analysis().get_available_classes(
            _as_concentration_frame(self._df)
        )

    @staticmethod
    def _check(result, what: str):
        """Surface a workflow's validation errors as a readable exception."""
        errors = getattr(result, 'validation_errors', None)
        if getattr(result, 'success', True) is False:
            detail = '; '.join(errors) if errors else 'unknown error'
            raise ValueError(f"{what} could not be produced: {detail}")
        return result

    def bar_chart(
        self,
        conditions: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        scale: str = 'linear',
        with_stats: bool = True,
    ) -> LCResult:
        """Class-level abundance bar chart (the app's Abundance Bar Chart).

        Args:
            conditions: Conditions to include. Defaults to all.
            classes: Lipid classes to include. Defaults to all.
            scale: 'linear' or 'log10'.
            with_stats: Run the class-level tests and annotate significance.

        Returns:
            LCResult with ``figure`` and ``data`` (mean/std per class).
        """
        res = self._analysis().run_bar_chart(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            selected_conditions=self._conditions_or_all(conditions),
            selected_classes=self._classes_or_all(classes),
            stat_config=StatisticalTestConfig() if with_stats else None,
            scale=scale,
        )
        self._check(res, "Bar chart")
        return LCResult(res.figure, res.abundance_df, res)

    def pie_charts(
        self,
        conditions: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, LCResult]:
        """Class-composition pie charts, one per condition.

        Returns:
            Dict of condition name -> LCResult.
        """
        out = self._analysis().run_pie_charts(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            selected_conditions=self._conditions_or_all(conditions),
            selected_classes=self._classes_or_all(classes),
        )
        return {
            cond: LCResult(r.figure, r.data_df, r) for cond, r in out.items()
        }

    def saturation(
        self,
        conditions: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        plot_type: str = 'concentration',
        show_significance: bool = True,
    ) -> Dict[str, Any]:
        """Saturation profile (SFA/MUFA/PUFA) per lipid class.

        Returns:
            Dict of lipid class -> Plotly figure, as the app renders them.
        """
        res = self._analysis().run_saturation(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            selected_conditions=self._conditions_or_all(conditions),
            selected_classes=self._classes_or_all(classes),
            stat_config=StatisticalTestConfig(),
            plot_type=plot_type,
            show_significance=show_significance,
        )
        self._check(res, "Saturation plot")
        return res.plots

    def chain_length(
        self,
        conditions: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
    ) -> LCResult:
        """Fatty-acid chain-length distribution (the app's Chain Length plot).

        Uses LipidCruncher's own chain-parsing rules, which decide how mixed
        species and sphingoid/ether chains are counted — do not re-derive
        these by parsing lipid names yourself.
        """
        res = self._analysis().run_chain_length(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            selected_conditions=self._conditions_or_all(conditions),
            selected_classes=self._classes_or_all(classes),
        )
        self._check(res, "Chain length plot")
        return LCResult(res.figure, res.per_condition_data, res)

    def fatty_acid_composition(
        self,
        lipid_class: str,
        conditions: Optional[List[str]] = None,
    ) -> LCResult:
        """Fatty-acid composition heatmap (FACH) for one lipid class."""
        res = self._analysis().run_fach(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            selected_class=lipid_class,
            selected_conditions=self._conditions_or_all(conditions),
        )
        self._check(res, "Fatty-acid composition plot")
        return LCResult(res.figure, res.data_dict, res)

    def pathway(self, control: str, experimental: str) -> LCResult:
        """Lipid pathway visualization comparing two conditions."""
        self._require_experiment()
        res = self._analysis().run_pathway(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            control=control,
            experimental=experimental,
        )
        self._check(res, "Pathway plot")
        return LCResult(res.figure, res.fold_change_df, res)

    def volcano(
        self,
        control: str,
        experimental: str,
        classes: Optional[List[str]] = None,
        p_threshold: float = 0.05,
        fc_threshold: float = 1.0,
    ) -> LCResult:
        """The app's volcano plot for two conditions.

        Same statistics as ``compare_species``; use this when the user wants
        the plot, and ``compare_species`` when they want the numbers.
        """
        self._require_experiment()
        res = self._analysis().run_volcano(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            control=control,
            experimental=experimental,
            selected_classes=self._classes_or_all(classes),
            stat_config=StatisticalTestConfig(),
            p_threshold=p_threshold,
            fc_threshold=fc_threshold,
        )
        self._check(res, "Volcano plot")
        return LCResult(res.figure, res.volcano_data, res)

    def heatmap(
        self,
        conditions: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        heatmap_type: str = 'regular',
        n_clusters: int = 3,
    ) -> LCResult:
        """Lipidomic heatmap of z-scores.

        Args:
            heatmap_type: 'regular' or 'clustered'.
            n_clusters: Cluster count when clustered.
        """
        res = self._analysis().run_heatmap(
            df=_as_concentration_frame(self._df),
            experiment=self._experiment,
            selected_conditions=self._conditions_or_all(conditions),
            selected_classes=self._classes_or_all(classes),
            heatmap_type=heatmap_type,
            n_clusters=n_clusters,
        )
        self._check(res, "Heatmap")
        return LCResult(res.figure, res.z_scores_df, res)

    def available_classes(self) -> List[str]:
        """Lipid classes present in the data."""
        return self._analysis().get_available_classes(
            _as_concentration_frame(self._df)
        )
