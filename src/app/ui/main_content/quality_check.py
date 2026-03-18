"""
Quality Check module UI (Module 2).

Displays 5 QC sections in expanders:
1. Box Plots — missing values + concentration distributions
2. BQC Assessment — CoV analysis with filtering (if BQC samples exist)
3. Retention Time — mass vs retention time (LipidSearch/MS-DIAL only)
4. Pairwise Correlation — correlation heatmaps per condition
5. PCA Analysis — PCA plot with sample removal

Data flow:
    continuation_df → box_plots (read-only) → bqc (may filter lipids) → qc_df
    qc_df → rt (read-only) → correlation (read-only)
          → pca (may remove samples) → final_df, updated_experiment
"""
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from app.constants import get_format_display_to_enum
from app.workflows.quality_check import QualityCheckWorkflow, QualityCheckConfig
from app.services.format_detection import DataFormat
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.download_utils import (
    plotly_svg_download_button,
    matplotlib_svg_download_button,
    csv_download_button,
)


# =============================================================================
# Entry Point
# =============================================================================

def display_quality_check_module(
    continuation_df: pd.DataFrame,
    experiment: 'ExperimentConfig',
    bqc_label: Optional[str],
    format_type,
) -> Tuple[pd.DataFrame, 'ExperimentConfig']:
    """Display the full Quality Check module.

    Args:
        continuation_df: DataFrame with concentration[sample] columns.
        experiment: ExperimentConfig.
        bqc_label: BQC condition label or None.
        format_type: DataFormat enum or format string.

    Returns:
        Tuple of (qc_df, updated_experiment).
    """
    st.subheader("Quality Check and Anomaly Detection Module")

    # Resolve format_type to DataFormat enum
    if isinstance(format_type, str):
        format_enum = get_format_display_to_enum().get(format_type, DataFormat.GENERIC)
    else:
        format_enum = format_type

    # Validate inputs
    errors = QualityCheckWorkflow.validate_inputs(continuation_df, experiment)
    if errors:
        for error in errors:
            st.error(error)
        return continuation_df, experiment

    # Working copy — start fresh from continuation_df each rerun
    qc_df = continuation_df.copy()

    # Section 1: Box Plots (read-only)
    _display_box_plots(qc_df, experiment)

    # Section 2: BQC Assessment (may filter lipids)
    qc_df = _display_bqc_assessment(qc_df, experiment, bqc_label)

    # Section 3: Retention Time (read-only)
    config = QualityCheckConfig(
        bqc_label=bqc_label,
        format_type=format_enum,
    )
    _display_retention_time_plots(qc_df, config)

    # Section 4: Correlation (read-only)
    _display_correlation_analysis(qc_df, experiment, bqc_label)

    # Section 5: PCA (may remove samples)
    qc_df, experiment = _display_pca_analysis(qc_df, experiment)

    # Store final state
    st.session_state.qc_continuation_df = qc_df

    return qc_df, experiment


# =============================================================================
# Section 1: Box Plots
# =============================================================================

def _display_box_plots(df: pd.DataFrame, experiment: 'ExperimentConfig') -> None:
    """Display missing values distribution and concentration box plots."""
    with st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns'):
        st.markdown(
            "Assess data quality and identify anomalies. "
            "Two diagnostic visualizations help detect technical issues or outliers."
        )
        st.markdown(
            "**What to look for:** Similar patterns across replicates indicate "
            "good reproducibility. Large differences may signal quality issues."
        )

        # Compute box plots (cached)
        fig1, fig2, mean_area_df, zero_values_percent_list, current_samples = (
            StreamlitAdapter.run_box_plots(df, experiment)
        )

        # Store for PDF report
        st.session_state.qc_box_plot_fig1 = fig1
        st.session_state.qc_box_plot_fig2 = fig2

        if not current_samples:
            st.warning("No concentration columns found for the current samples.")
            return

        # --- Results ---
        st.markdown("---")
        st.markdown("##### 📈 Results")

        # Missing Values Distribution
        st.markdown("###### Missing Values Distribution")
        st.markdown(
            "Percentage of zero values per sample. "
            "High percentages may indicate lower sensitivity or technical issues."
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Download buttons
        missing_values_df = pd.DataFrame({
            "Sample": current_samples,
            "Percentage Missing": zero_values_percent_list,
        })
        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(fig1, "missing_values_distribution.svg",
                                       key="qc_missing_values_svg")
        with col2:
            csv_download_button(missing_values_df, "missing_values_data.csv",
                               key="qc_missing_values_csv")

        st.markdown("---")

        # Box Plot of Non-Zero Concentrations
        st.markdown("###### Concentration Distribution")
        st.markdown(
            "Log10-transformed non-zero concentrations. "
            "Box = IQR (25th-75th percentile), line = median, points = outliers."
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(fig2, "box_plot.svg", key="qc_box_plot_svg")
        with col2:
            csv_download_button(mean_area_df, "box_plot_data.csv",
                               key="qc_box_plot_csv")


# =============================================================================
# Section 2: BQC Quality Assessment
# =============================================================================

def _display_bqc_assessment(
    df: pd.DataFrame,
    experiment: 'ExperimentConfig',
    bqc_label: Optional[str],
) -> pd.DataFrame:
    """Display BQC quality assessment and optional filtering.

    Returns the (potentially filtered) DataFrame.
    """
    if bqc_label is None or bqc_label not in experiment.conditions_list:
        return df

    with st.expander("Quality Check Using BQC Samples"):
        st.markdown(
            "Evaluate measurement reliability using Batch Quality Control (BQC) samples. "
            "Lower CoV = more consistent measurements."
        )
        st.markdown("**Coefficient of Variation (CoV):**")
        st.code("CoV = (Standard_Deviation / Mean) × 100%", language=None)
        st.markdown("Blue points are below threshold (reliable), red points are above (variable).")

        cov_threshold, scatter_plot, prepared_df, reliable_data_percent = (
            _render_bqc_scatter(df, experiment, bqc_label)
        )

        filtered_df = _render_bqc_filtering(df, prepared_df, cov_threshold)
        if filtered_df is not None:
            return filtered_df

    return df


def _render_bqc_scatter(
    df: pd.DataFrame,
    experiment: 'ExperimentConfig',
    bqc_label: str,
) -> tuple:
    """Render BQC settings, scatter plot, reliability assessment, and download buttons.

    Returns (cov_threshold, scatter_plot, prepared_df, reliable_data_percent).
    """
    # --- Settings ---
    st.markdown("---")
    st.markdown("##### ⚙️ Settings")

    cov_threshold = st.number_input(
        'CoV Threshold (%)',
        min_value=10,
        max_value=1000,
        value=st.session_state.get('qc_cov_threshold', 30),
        step=1,
        help="Points above threshold highlighted in red.",
        key='bqc_cov_threshold'
    )
    st.session_state.qc_cov_threshold = cov_threshold

    # --- Results ---
    st.markdown("---")
    st.markdown("##### 📈 Results")

    bqc_sample_index = experiment.conditions_list.index(bqc_label)
    scatter_plot, prepared_df, reliable_data_percent, _ = (
        StreamlitAdapter.run_bqc_scatter(
            df, experiment, bqc_sample_index, cov_threshold,
        )
    )

    st.plotly_chart(scatter_plot, use_container_width=True)

    # Store for PDF report
    st.session_state.qc_bqc_plot = scatter_plot

    # Reliability assessment
    if reliable_data_percent >= 80:
        st.success(f"{reliable_data_percent:.1f}% of datapoints are reliable (CoV < {cov_threshold}%).")
    elif reliable_data_percent >= 50:
        st.warning(f"{reliable_data_percent:.1f}% of datapoints are reliable (CoV < {cov_threshold}%).")
    else:
        st.error(f"Less than 50% of datapoints are reliable (CoV < {cov_threshold}%).")

    # Download buttons
    cov_data = prepared_df[['LipidMolec', 'cov', 'mean']].dropna()
    col1, col2 = st.columns(2)
    with col1:
        plotly_svg_download_button(scatter_plot, "bqc_quality_check.svg",
                                   key="qc_bqc_svg")
    with col2:
        csv_download_button(cov_data, "cov_plot_data.csv", key="bqc_csv_download")

    return cov_threshold, scatter_plot, prepared_df, reliable_data_percent


def _render_bqc_filtering(
    df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    cov_threshold: int,
) -> Optional[pd.DataFrame]:
    """Render BQC filtering UI: filter choice, high-CoV table, apply filter, results.

    Returns filtered DataFrame if filtering was applied, None otherwise.
    """
    if prepared_df is None or prepared_df.empty:
        return None

    st.markdown("---")
    st.markdown("##### 🔧 Data Filtering")

    filter_options = ("No", "Yes")
    restored_value = StreamlitAdapter.restore_widget_value(
        'bqc_filter_choice', '_preserved_bqc_filter_choice', 'No'
    )
    filter_cov = st.radio(
        f"Filter lipids with CoV ≥ {cov_threshold}%?",
        filter_options,
        index=filter_options.index(restored_value) if restored_value in filter_options else 0,
        horizontal=True,
        key='bqc_filter_choice'
    )
    StreamlitAdapter.save_widget_value('bqc_filter_choice', '_preserved_bqc_filter_choice')

    # Identify high-CoV lipids
    high_cov_mask = prepared_df['cov'] >= cov_threshold
    high_cov_lipids = prepared_df.loc[high_cov_mask, 'LipidMolec'].tolist()

    lipids_to_keep = []
    if filter_cov == "Yes" and high_cov_lipids:
        # Show high-CoV lipids table
        cov_filtered_df = prepared_df.loc[
            high_cov_mask, ['LipidMolec', 'ClassKey', 'cov', 'mean']
        ].copy()
        cov_filtered_df['cov'] = cov_filtered_df['cov'].round(2)
        cov_filtered_df['mean'] = cov_filtered_df['mean'].round(4)

        st.markdown("###### Lipids Above Threshold")
        st.dataframe(cov_filtered_df, use_container_width=True)

        lipids_to_keep = st.multiselect(
            "Keep despite high CoV:",
            options=cov_filtered_df['LipidMolec'].tolist(),
            format_func=lambda x: (
                f"{x} (CoV: "
                f"{cov_filtered_df[cov_filtered_df['LipidMolec'] == x]['cov'].iloc[0]}%)"
            ),
            help="Select lipids to retain in the dataset.",
            key='bqc_lipids_to_keep'
        )
    elif filter_cov == "No":
        # Keep all high-CoV lipids (no filtering)
        lipids_to_keep = high_cov_lipids

    # Apply BQC filter via workflow
    result = QualityCheckWorkflow.apply_bqc_filter(
        df=df,
        high_cov_lipids=high_cov_lipids,
        lipids_to_keep=lipids_to_keep,
    )

    # Summary
    if result.removed_count > 0:
        st.warning(
            f"Removed {result.removed_count} lipids "
            f"({result.removed_percentage:.1f}% of dataset)."
        )
    else:
        st.success("No lipids removed.")

    if filter_cov == "Yes" and lipids_to_keep:
        st.info(f"Retained {len(lipids_to_keep)} lipids despite high CoV.")

    # Show filtered dataset
    st.markdown("###### Filtered Dataset")
    st.dataframe(result.filtered_df, use_container_width=True)

    csv_download_button(result.filtered_df, "filtered_data.csv", key="bqc_filtered_download")

    return result.filtered_df


# =============================================================================
# Section 3: Retention Time Plots
# =============================================================================

def _display_retention_time_plots(df: pd.DataFrame, config: QualityCheckConfig) -> None:
    """Display retention time plots (LipidSearch/MS-DIAL only)."""
    rt_result = QualityCheckWorkflow.check_retention_time_availability(df, config)
    if not rt_result.available:
        return

    with st.expander('Retention Time Analysis'):
        st.markdown(
            "Verify lipid identification quality by plotting retention time vs. "
            "calculated mass for each lipid class."
        )
        st.markdown(
            "**What to look for:** Lipids within each class should cluster together, "
            "and classes should follow expected elution order (e.g., TGs elute last "
            "due to high hydrophobicity). Outliers may indicate misidentifications."
        )

        # --- Settings ---
        st.markdown("---")
        st.markdown("##### ⚙️ Settings")

        rt_options = ['Comparison Mode', 'Individual Mode']
        restored_mode = StreamlitAdapter.restore_widget_value(
            'rt_viewing_mode', '_preserved_rt_viewing_mode', 'Comparison Mode'
        )
        mode = st.radio(
            'Viewing Mode',
            rt_options,
            index=rt_options.index(restored_mode) if restored_mode in rt_options else 0,
            horizontal=True,
            key='rt_viewing_mode'
        )
        StreamlitAdapter.save_widget_value('rt_viewing_mode', '_preserved_rt_viewing_mode')

        if mode == 'Individual Mode':
            st.markdown("---")
            st.markdown("##### 📈 Results")

            plots = StreamlitAdapter.run_retention_time_single(df)
            for idx, (plot, retention_df) in enumerate(plots, 1):
                # Store first individual plot for PDF report (fallback)
                if idx == 1:
                    st.session_state.qc_retention_time_plot = plot
                st.plotly_chart(plot, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    plotly_svg_download_button(
                        plot, f"retention_time_plot_{idx}.svg",
                        key=f'qc_rt_svg_individual_{idx}'
                    )
                with col2:
                    csv_download_button(retention_df, f"retention_plot_{idx}.csv",
                                       key=f"rt_csv_individual_{idx}")
                st.markdown("---")

        elif mode == 'Comparison Mode':
            # --- Data Selection ---
            st.markdown("---")
            st.markdown("##### 🎯 Data Selection")

            all_classes = rt_result.lipid_classes
            selected_classes = st.multiselect(
                'Lipid Classes',
                all_classes,
                default=all_classes[:min(5, len(all_classes))],
                key='rt_class_selection'
            )

            if not selected_classes:
                st.warning("Please select at least one lipid class.")
                return

            # --- Results ---
            st.markdown("---")
            st.markdown("##### 📈 Results")

            plot, retention_df = StreamlitAdapter.run_retention_time_multi(df, selected_classes)
            if plot:
                # Store comparison plot for PDF report (preferred over individual)
                st.session_state.qc_retention_time_plot = plot
                st.plotly_chart(plot, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    plotly_svg_download_button(
                        plot, "retention_time_comparison.svg",
                        key='qc_rt_svg_comparison'
                    )
                with col2:
                    csv_download_button(retention_df, "retention_time_comparison.csv",
                                       key="rt_csv_comparison")


# =============================================================================
# Section 4: Pairwise Correlation
# =============================================================================

def _display_correlation_analysis(
    df: pd.DataFrame,
    experiment: 'ExperimentConfig',
    bqc_label: Optional[str],
) -> None:
    """Display pairwise correlation heatmaps."""
    with st.expander('Pairwise Correlation Analysis'):
        st.markdown(
            "Assess reproducibility by calculating Pearson correlation "
            "coefficients between sample replicates."
        )
        st.markdown(
            "**Interpretation:** Values close to 1 = similar patterns (good). "
            "Blue = higher correlation, Red = lower correlation."
        )

        # Get eligible conditions (>1 replicate)
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(experiment)
        if not eligible:
            st.error(
                "No conditions with multiple replicates found. "
                "Correlation analysis requires at least two replicates."
            )
            return

        # --- Data Selection ---
        st.markdown("---")
        st.markdown("##### 🎯 Data Selection")

        selected_condition = st.selectbox(
            'Condition',
            eligible,
            key='corr_condition'
        )

        # Auto-detect sample type based on BQC presence
        if bqc_label is not None:
            sample_type = 'technical replicates'
            st.info("**Sample type:** Technical replicates (BQC samples detected)")
        else:
            sample_type = 'biological replicates'
            st.info("**Sample type:** Biological replicates (no BQC samples)")

        # --- Results ---
        st.markdown("---")
        st.markdown("##### 📈 Results")

        condition_index = experiment.conditions_list.index(selected_condition)
        fig, correlation_df = StreamlitAdapter.run_correlation(
            df, experiment, condition_index, sample_type,
        )

        st.pyplot(fig)

        # Store for potential PDF generation
        st.session_state.qc_correlation_plots[selected_condition] = fig

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            matplotlib_svg_download_button(
                fig, f"correlation_plot_{selected_condition}.svg",
                key='qc_corr_svg'
            )
        with col2:
            csv_download_button(correlation_df,
                               f"correlation_matrix_{selected_condition}.csv",
                               key="corr_csv_download")

        # Correlation matrix table
        st.markdown("###### Correlation Coefficients")
        st.dataframe(correlation_df, use_container_width=True)


# =============================================================================
# Section 5: PCA Analysis
# =============================================================================

def _display_pca_analysis(
    df: pd.DataFrame,
    experiment: 'ExperimentConfig',
) -> Tuple[pd.DataFrame, 'ExperimentConfig']:
    """Display PCA analysis with optional sample removal.

    Returns (updated_df, updated_experiment).
    """
    with st.expander("Principal Component Analysis (PCA)"):
        st.markdown(
            "Visualize sample clustering based on lipid profiles. "
            "Each point = one sample, ellipses = 95% confidence intervals."
        )
        st.markdown(
            "**Interpretation:** Clustered points = similar profiles. "
            "Separated clusters = distinct conditions. "
            "Outliers fall outside ellipses."
        )

        # --- Settings ---
        st.markdown("---")
        st.markdown("##### ⚙️ Settings")

        restored_pca = StreamlitAdapter.restore_widget_value(
            'pca_samples_remove', '_preserved_pca_samples_remove', []
        )
        # Filter preserved values to only include currently valid samples
        valid_restored = [s for s in restored_pca if s in experiment.full_samples_list]
        samples_to_remove = st.multiselect(
            'Exclude Samples (optional)',
            experiment.full_samples_list,
            default=valid_restored if valid_restored else [],
            help="Exclude suspected outliers from analysis.",
            key='pca_samples_remove'
        )
        StreamlitAdapter.save_widget_value('pca_samples_remove', '_preserved_pca_samples_remove')

        if samples_to_remove:
            remaining_count = len(experiment.full_samples_list) - len(samples_to_remove)
            if remaining_count < 2:
                st.error('At least two samples required for PCA.')
                return df, experiment

            # Apply removal via workflow
            removal_result = QualityCheckWorkflow.remove_samples(
                df, experiment, samples_to_remove
            )
            df = removal_result.updated_df
            experiment = removal_result.updated_experiment
            st.session_state.qc_samples_removed = samples_to_remove

            st.warning(
                f"⚠️ {len(samples_to_remove)} sample(s) excluded from "
                f"**all downstream analyses**, not just PCA."
            )
            st.success(f"Proceeding with {remaining_count} samples.")

        # --- Results ---
        st.markdown("---")
        st.markdown("##### 📈 Results")

        pca_plot, pca_df = StreamlitAdapter.run_pca(df, experiment)
        st.plotly_chart(pca_plot, use_container_width=True)
        st.session_state.qc_pca_plot = pca_plot

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(pca_plot, "pca_plot.svg", key="qc_pca_svg")
        with col2:
            csv_download_button(pca_df, "pca_data.csv", key="pca_csv_download")

    return df, experiment