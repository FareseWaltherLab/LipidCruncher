"""
The ``lc`` namespace exposed to AI-generated sandbox code.

Gives AI-generated code direct access to LipidCruncher's own pipeline
services, so questions the main pipeline can already answer are answered
*by the pipeline* instead of being re-derived with looser ad-hoc
statistics. The AI is instructed (see ``prompt_templates``) to check this
catalog first and only write from-scratch code when nothing here fits.

Pure — no Streamlit. Bound to one DataFrame + ExperimentConfig and
injected into the sandbox globals as ``lc``.
"""
from typing import List, Optional

import pandas as pd

from ...models.experiment import ExperimentConfig
from ...models.statistics import StatisticalTestConfig
from ..statistical_testing import StatisticalTestingService


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
