import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import streamlit as st
import itertools
from typing import Dict, List, Tuple, Optional, Any

class VolcanoPlot:
    """
    Volcano plot class with rigorous statistical testing methodology
    following the same framework as the bar chart analysis.
    """

    @staticmethod
    def _get_samples_for_conditions(experiment, control_condition, experimental_condition):
        """
        Retrieve sample lists for control and experimental conditions from an experiment object.
        
        Args:
            experiment: An object containing experiment details, including conditions and sample lists.
            control_condition: The control condition for the volcano plot.
            experimental_condition: The experimental condition for comparison in the volcano plot.
        
        Returns:
            A tuple of two lists: control samples list and experimental samples list.
        """
        control_index = experiment.conditions_list.index(control_condition)
        experimental_index = experiment.conditions_list.index(experimental_condition)
        return experiment.individual_samples_list[control_index], experiment.individual_samples_list[experimental_index]

    @staticmethod
    @st.cache_data(ttl=3600)
    def _prepare_data(df, control_samples, experimental_samples):
        """
        Prepare data by selecting necessary columns based on control and experimental samples.
        
        Args:
            df: DataFrame containing lipidomics data.
            control_samples: List of sample names for the control condition.
            experimental_samples: List of sample names for the experimental condition.
        
        Returns:
            A tuple containing the processed DataFrame and lists of column names for control and experimental samples.
        """
        control_cols = ['concentration[' + sample + ']' for sample in control_samples]
        experimental_cols = ['concentration[' + sample + ']' for sample in experimental_samples]
        selected_cols = control_cols + experimental_cols + ['LipidMolec', 'ClassKey']
        return df[selected_cols], control_cols, experimental_cols

    @staticmethod
    def perform_statistical_tests(df: pd.DataFrame,
                                control_cols: List[str],
                                experimental_cols: List[str],
                                test_type: str = "parametric",
                                correction_method: str = "uncorrected",
                                alpha: float = 0.05,
                                auto_transform: bool = True) -> Dict:
        """
        Perform rigorous statistical testing for volcano plot data following the bar chart methodology.
        
        Args:
            df: DataFrame containing the lipidomics data
            control_cols: List of control sample column names
            experimental_cols: List of experimental sample column names
            test_type: "parametric", "non_parametric", or "auto"
            correction_method: "uncorrected", "fdr_bh", or "bonferroni"
            alpha: Significance threshold
            auto_transform: Whether to apply log10 transformation automatically
            
        Returns:
            Dictionary containing statistical test results for each lipid
        """
        statistical_results = {}
        all_p_values = []
        all_lipids = []
        
        for idx, row in df.iterrows():
            lipid_name = row['LipidMolec']
            
            try:
                # Extract concentration values for control and experimental groups
                control_values = row[control_cols].values
                experimental_values = row[experimental_cols].values
                
                # Filter out any NaN values
                control_values = control_values[~pd.isna(control_values)]
                experimental_values = experimental_values[~pd.isna(experimental_values)]
                
                # Skip if either group has no valid values or if either group has all zeros
                if (len(control_values) == 0 or len(experimental_values) == 0 or
                    np.all(control_values == 0) or np.all(experimental_values == 0)):
                    continue
                
                # Handle zeros and apply small value replacement for transformation
                all_values = np.concatenate([control_values, experimental_values])
                min_positive = min([x for x in all_values if x > 0]) if any(x > 0 for x in all_values) else 1
                small_value = min_positive / 10
                
                # Replace zeros with small values to enable log transformation
                control_values_adj = np.array([max(x, small_value) for x in control_values])
                experimental_values_adj = np.array([max(x, small_value) for x in experimental_values])
                
                # Store original data for fold change calculation
                original_control = control_values_adj.copy()
                original_experimental = experimental_values_adj.copy()
                
                # Apply transformation if auto_transform is enabled
                transformed_control = original_control.copy()
                transformed_experimental = original_experimental.copy()
                transformation_applied = "none"
                
                if auto_transform:
                    # Apply log10 transformation automatically (standard practice)
                    transformed_control = np.log10(control_values_adj)
                    transformed_experimental = np.log10(experimental_values_adj)
                    transformation_applied = "log10"
                
                # Choose and perform statistical test
                if test_type == "non_parametric":
                    statistic, p_value = mannwhitneyu(transformed_control, transformed_experimental, 
                                                    alternative='two-sided')
                    test_chosen = "Mann-Whitney U"
                elif test_type in ["parametric", "auto"]:
                    # Use Welch's t-test (doesn't assume equal variances)
                    statistic, p_value = stats.ttest_ind(transformed_control, transformed_experimental, 
                                                       equal_var=False)
                    test_chosen = "Welch's t-test"
                
                # Calculate fold change using original (adjusted) values for biological interpretation
                mean_control = np.mean(original_control)
                mean_experimental = np.mean(original_experimental)
                fold_change = mean_experimental / mean_control
                log2_fold_change = np.log2(fold_change)
                
                # Store results
                statistical_results[lipid_name] = {
                    'test': test_chosen,
                    'statistic': statistic,
                    'p-value': p_value,
                    'transformation': transformation_applied,
                    'mean_control': mean_control,
                    'mean_experimental': mean_experimental,
                    'fold_change': fold_change,
                    'log2_fold_change': log2_fold_change,
                    'class_key': row['ClassKey']
                }
                
                all_p_values.append(p_value)
                all_lipids.append(lipid_name)
                
            except Exception as e:
                # Skip problematic lipids
                continue
        
        # Apply multiple testing correction if requested
        if all_p_values and correction_method != "uncorrected":
            if correction_method == "fdr_bh":
                adjusted = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
                method_name = "Benjamini-Hochberg FDR"
            elif correction_method == "bonferroni":
                adjusted = multipletests(all_p_values, alpha=alpha, method='bonferroni')
                method_name = "Bonferroni"
            
            adjusted_p_values = adjusted[1]
            significance_flags = adjusted[0]
            
            # Update results with adjusted p-values
            for i, lipid_name in enumerate(all_lipids):
                if lipid_name in statistical_results:
                    statistical_results[lipid_name]['adjusted_p_value'] = adjusted_p_values[i]
                    statistical_results[lipid_name]['significant'] = significance_flags[i]
                    statistical_results[lipid_name]['correction_method'] = method_name
        else:
            # No correction applied
            for lipid_name in all_lipids:
                if lipid_name in statistical_results:
                    statistical_results[lipid_name]['adjusted_p_value'] = statistical_results[lipid_name]['p-value']
                    statistical_results[lipid_name]['significant'] = statistical_results[lipid_name]['p-value'] <= alpha
                    statistical_results[lipid_name]['correction_method'] = "Uncorrected"
        
        # Add metadata
        statistical_results['_parameters'] = {
            'test_type': test_type,
            'correction_method': correction_method,
            'alpha': alpha,
            'auto_transform': auto_transform,
            'n_tests_performed': len(all_p_values)
        }
        
        return statistical_results

    @staticmethod
    @st.cache_data(ttl=3600)
    def _calculate_stats_enhanced(df, control_cols, experimental_cols, 
                                test_type="parametric", correction_method="uncorrected", 
                                alpha=0.05, auto_transform=True):
        """
        Enhanced statistics calculation with rigorous statistical testing.
        
        Args:
            df: DataFrame containing relevant data for control and experimental conditions.
            control_cols: List of column names for control samples.
            experimental_cols: List of column names for experimental samples.
            test_type: Type of statistical test to perform
            correction_method: Multiple testing correction method
            alpha: Significance threshold
            auto_transform: Whether to apply automatic log transformation
        
        Returns:
            A tuple containing the DataFrame and statistical results dictionary.
        """
        # Perform enhanced statistical testing
        statistical_results = VolcanoPlot.perform_statistical_tests(
            df, control_cols, experimental_cols, 
            test_type, correction_method, alpha, auto_transform
        )
        
        return df, statistical_results
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def _format_results_enhanced(df, statistical_results, control_cols, experimental_cols):
        """
        Format the results using the enhanced statistical testing framework.
        
        Args:
            df: The DataFrame containing lipidomics data
            statistical_results: Dictionary containing statistical test results
            control_cols: List of control sample column names
            experimental_cols: List of experimental sample column names
        
        Returns:
            tuple: 
                - A DataFrame (volcano_df) containing the calculated metrics
                - A DataFrame (removed_lipids_df) listing excluded lipids
        """
        volcano_data = []
        removed_lipids = []
        
        for idx, row in df.iterrows():
            lipid_name = row['LipidMolec']
            
            if lipid_name in statistical_results:
                # Lipid was successfully processed
                result = statistical_results[lipid_name]
                
                volcano_data.append({
                    'LipidMolec': lipid_name,
                    'ClassKey': result['class_key'],
                    'FoldChange': result['log2_fold_change'],
                    'pValue': result['p-value'],
                    'adjusted_pValue': result['adjusted_p_value'],
                    '-log10(pValue)': -np.log10(result['p-value']),
                    '-log10(adjusted_pValue)': -np.log10(result['adjusted_p_value']),
                    'Log10MeanControl': np.log10(result['mean_control']),
                    'mean_control': result['mean_control'],
                    'mean_experimental': result['mean_experimental'],
                    'test_method': result['test'],
                    'transformation': result['transformation'],
                    'significant': result['significant'],
                    'correction_method': result['correction_method']
                })
            else:
                # Lipid was excluded - determine why
                control_values = row[control_cols].values
                experimental_values = row[experimental_cols].values
                
                # Filter out NaN values
                control_values = control_values[~pd.isna(control_values)]
                experimental_values = experimental_values[~pd.isna(experimental_values)]
                
                if len(control_values) == 0:
                    reason = "No valid control values"
                elif len(experimental_values) == 0:
                    reason = "No valid experimental values"
                elif np.all(control_values == 0):
                    reason = "All control values are zero"
                elif np.all(experimental_values == 0):
                    reason = "All experimental values are zero"
                else:
                    reason = "Statistical test failed"
                
                removed_lipids.append({
                    'LipidMolec': lipid_name,
                    'ClassKey': row['ClassKey'],
                    'Reason': reason
                })
        
        volcano_df = pd.DataFrame(volcano_data)
        removed_lipids_df = pd.DataFrame(removed_lipids)
        
        return volcano_df, removed_lipids_df

    @staticmethod
    def _generate_color_mapping(merged_df):
        """
        Generate a color mapping for different lipid classes in the plot using Plotly's color sequences.
        
        Args:
            merged_df: DataFrame that contains merged data for plotting, including lipid classes.
        
        Returns:
            A dictionary mapping lipid classes to colors.
        """
        unique_classes = list(merged_df['ClassKey'].unique())
        colors = itertools.cycle(px.colors.qualitative.Plotly)
        return {class_name: next(colors) for class_name in unique_classes}

    @staticmethod
    def _create_plot_enhanced(volcano_df, color_mapping, q_value_threshold, 
                            hide_non_significant, use_adjusted_p=True, 
                            fold_change_threshold=1.0, top_n_labels=0):
        """
        Create a Plotly figure for the volcano plot visualization with improved statistical framework.
        
        Args:
            volcano_df: DataFrame containing volcano plot data
            color_mapping: Dictionary mapping lipid classes to colors
            q_value_threshold: The threshold for significance (-log10 of p-value)
            hide_non_significant: Boolean indicating whether to hide non-significant data points
            use_adjusted_p: Whether to use adjusted p-values for significance determination
            fold_change_threshold: Fold change threshold for biological significance
            top_n_labels: Number of top lipids to label (ranked by p-value)
        
        Returns:
            A Plotly figure object representing the volcano plot.
        """
        # Choose which p-value column to use
        p_col = '-log10(adjusted_pValue)' if use_adjusted_p else '-log10(pValue)'
        p_raw_col = 'adjusted_pValue' if use_adjusted_p else 'pValue'
        
        if hide_non_significant:
            significant_df = volcano_df[
                ((volcano_df['FoldChange'] < -fold_change_threshold) | 
                 (volcano_df['FoldChange'] > fold_change_threshold)) & 
                (volcano_df[p_col] >= q_value_threshold)
            ]
        else:
            significant_df = volcano_df
        
        fig = go.Figure()

        # Add traces for each lipid class
        for class_name, color in color_mapping.items():
            class_df = significant_df[significant_df['ClassKey'] == class_name]
            if not class_df.empty:
                fig.add_trace(go.Scatter(
                    x=class_df['FoldChange'],
                    y=class_df[p_col],
                    mode='markers',
                    name=class_name,
                    marker=dict(color=color, size=5),
                    text=class_df['LipidMolec'],
                    customdata=np.column_stack((
                        class_df[p_raw_col], 
                        class_df['test_method'],
                        class_df['transformation'],
                        class_df['correction_method']
                    )),
                    hovertemplate='<b>Lipid:</b> %{text}<br>' +
                                  '<b>Log2 Fold Change:</b> %{x:.3f}<br>' +
                                  f'<b>{"-log10(adj. p-value)" if use_adjusted_p else "-log10(p-value)"}:</b> %{{y:.3f}}<br>' +
                                  f'<b>{"Adj. p-value" if use_adjusted_p else "p-value"}:</b> %{{customdata[0]:.2e}}<br>' +
                                  '<b>Test:</b> %{customdata[1]}<br>' +
                                  '<b>Transform:</b> %{customdata[2]}<br>' +
                                  '<b>Correction:</b> %{customdata[3]}<extra></extra>'
                ))

        # Add significance threshold lines
        if not significant_df.empty:
            # Horizontal line for p-value threshold
            fig.add_shape(
                type="line", 
                x0=significant_df['FoldChange'].min(), 
                x1=significant_df['FoldChange'].max(),
                y0=q_value_threshold, 
                y1=q_value_threshold, 
                line=dict(dash="dash", color="red", width=2)
            )
            
            # Vertical lines for fold change thresholds
            fig.add_shape(
                type="line", 
                x0=-fold_change_threshold, 
                x1=-fold_change_threshold, 
                y0=0, 
                y1=significant_df[p_col].max(), 
                line=dict(dash="dash", color="red", width=2)
            )
            fig.add_shape(
                type="line", 
                x0=fold_change_threshold, 
                x1=fold_change_threshold, 
                y0=0, 
                y1=significant_df[p_col].max(), 
                line=dict(dash="dash", color="red", width=2)
            )
        
        # Add labels for top N most significant points using annotations with arrows
        if top_n_labels > 0:
            # Sort by significance (descending -log10(p))
            sorted_df = volcano_df.sort_values(p_col, ascending=False).head(top_n_labels)
            
            # Parameters for bounding box estimation
            char_width = 0.05  # approximate width per character in plot units
            label_height = 0.3  # approximate height of label
            buffer = 0.2  # buffer for arrows
            
            placed_boxes = []  # list of {'left':, 'right':, 'bottom':, 'top':}
            
            # Define candidate positions relative to point: (dx, dy, align)
            candidates = [
                (0.1, 0.3, 'left'),   # above right
                (-0.1, 0.3, 'right'), # above left
                (0.1, -0.3, 'left'),  # below right
                (-0.1, -0.3, 'right'),# below left
                (0.3, 0.1, 'left'),   # right upper
                (0.3, -0.1, 'left'),  # right lower
                (-0.3, 0.1, 'right'), # left upper
                (-0.3, -0.1, 'right') # left lower
            ]
            
            for i, row in sorted_df.iterrows():
                color = color_mapping[row['ClassKey']]
                text = row['LipidMolec']
                point_x = row['FoldChange']
                point_y = row[p_col]
                
                # Estimate width
                w = len(text) * char_width
                
                placed = False
                for cand_i, (dx, dy, align) in enumerate(candidates):
                    # Scale offsets with try number for more spread
                    scale = 1 + (cand_i // len(candidates)) * 0.5
                    label_x = point_x + dx * scale
                    label_y = point_y + dy * scale
                    
                    # Calculate box edges based on align
                    if align == 'left':
                        left = label_x
                        right = label_x + w
                    else:
                        left = label_x - w
                        right = label_x
                    
                    bottom = label_y - label_height / 2
                    top = label_y + label_height / 2
                    
                    # Check for overlap with placed boxes (including buffer for arrows)
                    overlap = False
                    for box in placed_boxes:
                        if not (right < box['left'] - buffer or left > box['right'] + buffer or 
                                top < box['bottom'] - buffer or bottom > box['top'] + buffer):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Place the text annotation
                        fig.add_annotation(
                            x=label_x,
                            y=label_y,
                            text=text,
                            showarrow=False,
                            font=dict(color=color, size=12),
                            align=align
                        )
                        
                        # Add the arrow annotation (tail at label, head at point)
                        fig.add_annotation(
                            x=point_x,
                            y=point_y,
                            ax=label_x,
                            ay=label_y,
                            axref='x',
                            ayref='y',
                            text='',
                            showarrow=True,
                            arrowhead=1,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='black'
                        )
                        
                        # Add to placed boxes with buffer
                        placed_boxes.append({
                            'left': left - buffer,
                            'right': right + buffer,
                            'bottom': bottom - buffer,
                            'top': top + buffer
                        })
                        placed = True
                        # print(f"Placed {text} at ({label_x:.2f}, {label_y:.2f}) with align {align}")  # Debug
                        break
                
                if not placed:
                    # Fallback: place above with left/right align
                    align_fallback = 'left' if point_x > 0 else 'right'
                    label_x_fallback = point_x + (0.1 if align_fallback == 'left' else -0.1)
                    label_y_fallback = point_y + 0.2
                    
                    fig.add_annotation(
                        x=label_x_fallback,
                        y=label_y_fallback,
                        text=text,
                        showarrow=False,
                        font=dict(color=color, size=12),
                        align=align_fallback
                    )
                    
                    # Fallback arrow
                    fig.add_annotation(
                        x=point_x,
                        y=point_y,
                        ax=label_x_fallback,
                        ay=label_y_fallback,
                        axref='x',
                        ayref='y',
                        text='',
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor='black'
                    )
                    # print(f"Fallback placed {text} at ({label_x_fallback:.2f}, {label_y_fallback:.2f})")  # Debug

        # Update layout
        p_label = "Adjusted p-value" if use_adjusted_p else "p-value"
        fig.update_layout(
            title=dict(
                text="Volcano Plot", 
                font=dict(size=20, color='black')
            ),
            xaxis_title=dict(text="Log2(Fold Change)", font=dict(size=16, color='black')),
            yaxis_title=dict(text=f"-log10({p_label})", font=dict(size=16, color='black')),
            xaxis=dict(tickfont=dict(size=12, color='black'), showgrid=True, gridcolor='lightgray'),
            yaxis=dict(tickfont=dict(size=12, color='black'), showgrid=True, gridcolor='lightgray'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=10, color='black')),
            height=600,
            margin=dict(t=80, r=50, b=50, l=50)
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig

    @staticmethod
    def create_and_display_volcano_plot_enhanced(experiment, df, control_condition, experimental_condition, 
                                               selected_classes, q_value_threshold, hide_non_significant,
                                               test_type="parametric", correction_method="uncorrected", 
                                               alpha=0.05, auto_transform=True, 
                                               use_adjusted_p=True, fold_change_threshold=1.0,
                                               top_n_labels=0):
        """
        Volcano plot generation with rigorous statistical testing framework.
        
        Args:
            experiment: Experiment object containing experimental setup information
            df: DataFrame containing lipidomics data
            control_condition: Name of the control condition
            experimental_condition: Name of the experimental condition
            selected_classes: List of lipid classes to include
            q_value_threshold: Threshold for statistical significance (-log10 scale)
            hide_non_significant: Whether to hide non-significant points
            test_type: "parametric", "non_parametric", or "auto"
            correction_method: "uncorrected", "fdr_bh", or "bonferroni"
            alpha: Significance threshold
            auto_transform: Whether to apply log10 transformation
            use_adjusted_p: Whether to use adjusted p-values for plotting
            fold_change_threshold: Biological significance threshold for fold change
            top_n_labels: Number of top significant lipids to label on the plot
        
        Returns:
            tuple: (plot figure, volcano_df, removed_lipids_df, statistical_summary)
        """
        # Get samples for conditions
        control_samples, experimental_samples = VolcanoPlot._get_samples_for_conditions(
            experiment, control_condition, experimental_condition
        )
        
        # Prepare data
        df_processed, control_cols, experimental_cols = VolcanoPlot._prepare_data(
            df, control_samples, experimental_samples
        )
        
        # Filter by selected classes early
        df_processed = df_processed[df_processed['ClassKey'].isin(selected_classes)]
        
        # Enhanced statistical analysis
        df_processed, statistical_results = VolcanoPlot._calculate_stats_enhanced(
            df_processed, control_cols, experimental_cols, 
            test_type, correction_method, alpha, auto_transform
        )
        
        # Format results with enhanced framework
        volcano_df, removed_lipids_df = VolcanoPlot._format_results_enhanced(
            df_processed, statistical_results, control_cols, experimental_cols
        )
        
        if volcano_df.empty:
            return None, volcano_df, removed_lipids_df, statistical_results
        
        # Generate color mapping
        color_mapping = VolcanoPlot._generate_color_mapping(volcano_df)
        
        # Create enhanced plot
        plot = VolcanoPlot._create_plot_enhanced(
            volcano_df, color_mapping, q_value_threshold, hide_non_significant,
            use_adjusted_p, fold_change_threshold, top_n_labels=top_n_labels
        )
        
        return plot, volcano_df, removed_lipids_df, statistical_results

    @staticmethod
    def _create_concentration_vs_fold_change_plot(merged_df, color_mapping, q_value_threshold, 
                                                hide_non_significant, use_adjusted_p=True):
        """
        Create a Plotly figure for Log10(Mean Control Concentration) vs. Log2(Fold Change) with enhanced features.
        """
        p_col = '-log10(adjusted_pValue)' if use_adjusted_p else '-log10(pValue)'
        
        if hide_non_significant:
            significant_df = merged_df[
                ((merged_df['FoldChange'] < -1) | (merged_df['FoldChange'] > 1)) & 
                (merged_df[p_col] >= q_value_threshold)
            ]
        else:
            significant_df = merged_df
        
        fig = go.Figure()

        for class_name, color in color_mapping.items():
            class_df = significant_df[significant_df['ClassKey'] == class_name]
            if not class_df.empty:
                fig.add_trace(go.Scatter(
                    x=class_df['FoldChange'],
                    y=class_df['Log10MeanControl'],
                    mode='markers',
                    name=class_name,
                    marker=dict(color=color, size=5),
                    text=class_df['LipidMolec'],
                    hovertemplate='<b>Lipid:</b> %{text}<br>' +
                                  '<b>Log2 Fold Change:</b> %{x:.3f}<br>' +
                                  '<b>Log10(Mean Control):</b> %{y:.3f}<extra></extra>'
                ))

        fig.update_layout(
            title=dict(text="Fold Change vs. Mean Control Concentration", font=dict(size=20, color='black')),
            xaxis_title=dict(text="Log2(Fold Change)", font=dict(size=16, color='black')),
            yaxis_title=dict(text="Log10(Mean Control Concentration)", font=dict(size=16, color='black')),
            xaxis=dict(tickfont=dict(size=12, color='black'), showgrid=True, gridcolor='lightgray'),
            yaxis=dict(tickfont=dict(size=12, color='black'), showgrid=True, gridcolor='lightgray'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=10, color='black')),
            height=600,
            margin=dict(t=50, r=50, b=50, l=50)
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig, significant_df[['LipidMolec', 'Log10MeanControl', 'FoldChange', 'ClassKey']]

    # Keep the existing helper methods for backward compatibility
    @staticmethod
    def get_most_abundant_lipid(df, selected_class):
        """Get the most abundant lipid in the selected class."""
        class_df = df[df['ClassKey'] == selected_class]
        most_abundant_lipid = class_df.set_index('LipidMolec').sum(axis=1).idxmax()
        return most_abundant_lipid
    
    @staticmethod
    def create_concentration_distribution_data(volcano_df, selected_lipids, selected_conditions, experiment):
        """Prepares data for concentration distribution plot of selected lipids across conditions."""
        plot_data = []
        for lipid in selected_lipids:
            for condition in selected_conditions:
                samples = experiment.individual_samples_list[experiment.conditions_list.index(condition)]
                for sample in samples:
                    concentration = volcano_df.loc[volcano_df['LipidMolec'] == lipid, f'concentration[{sample}]'].values[0]
                    plot_data.append({'Lipid': lipid, 'Condition': condition, 'Concentration': concentration})
        
        return pd.DataFrame(plot_data)

    @staticmethod
    def create_concentration_distribution_plot(plot_df, selected_lipids, selected_conditions):
        """Creates a seaborn box plot for the concentration distribution of selected lipids."""
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        ax = sns.boxplot(x="Lipid", y="Concentration", hue="Condition", data=plot_df, palette="Set2")
        ax.grid(False)
        plt.title("Concentration Distribution for Selected Lipids", fontsize=20)
        plt.xlabel("Lipid", fontsize=20)
        plt.ylabel("Concentration", fontsize=20)
        plt.xticks(fontsize=14, rotation=45, ha='right')
        plt.yticks(fontsize=14)
        plt.legend(title='Condition', loc='upper right')
        plt.tight_layout()
        return plt.gcf()

    # Backward compatibility wrapper
    @staticmethod
    def create_and_display_volcano_plot(experiment, df, control_condition, experimental_condition, 
                                      selected_classes, q_value_threshold, hide_non_significant):
        """
        Backward compatibility wrapper that calls the enhanced version with default parameters.
        """
        return VolcanoPlot.create_and_display_volcano_plot_enhanced(
            experiment, df, control_condition, experimental_condition, 
            selected_classes, q_value_threshold, hide_non_significant,
            test_type="parametric", correction_method="uncorrected", 
            alpha=0.05, auto_transform=True, use_adjusted_p=False, fold_change_threshold=1.0
        )