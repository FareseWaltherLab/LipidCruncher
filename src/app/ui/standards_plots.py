"""
Internal standards consistency plots UI component.
"""
import pandas as pd
import streamlit as st

from ..models.experiment import ExperimentConfig
from ..services.plotting.standards_plotter import StandardsPlotterService


def display_standards_consistency_plots(
    intsta_df: pd.DataFrame,
    experiment: ExperimentConfig
) -> None:
    """
    Display internal standards consistency bar charts.

    Shows bar plots for each internal standard class with a condition multiselect
    to filter which samples to display. Consistent bar heights across samples
    indicate good sample preparation and instrument performance.

    Args:
        intsta_df: Internal standards DataFrame with LipidMolec, ClassKey, and intensity columns.
        experiment: Experiment configuration with conditions and sample mappings.
    """
    if intsta_df is None or intsta_df.empty:
        return

    st.markdown("---")
    st.markdown("##### 📊 Internal Standards Consistency")
    st.markdown(
        "*Consistent bar heights across samples indicate good sample preparation "
        "and instrument performance.*"
    )

    # Condition multiselect to filter samples
    conditions = experiment.conditions_list
    selected_conditions = st.multiselect(
        'Select conditions:',
        conditions,
        default=conditions,
        key='standards_conditions_select'
    )

    if not selected_conditions:
        st.info("Select at least one condition to view plots.")
        return

    # Get samples for selected conditions
    selected_samples = []
    for cond in selected_conditions:
        idx = conditions.index(cond)
        selected_samples.extend(experiment.individual_samples_list[idx])

    # Maintain original sample order
    full_samples = experiment.full_samples_list
    selected_samples_ordered = [s for s in full_samples if s in selected_samples]

    if not selected_samples_ordered:
        st.warning("No samples available for selected conditions.")
        return

    # Generate and display plots
    plots = StandardsPlotterService.create_consistency_plots(
        intsta_df,
        selected_samples_ordered
    )

    if plots:
        for fig in plots:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No standards data available for plotting.")
