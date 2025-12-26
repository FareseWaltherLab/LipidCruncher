import numpy as np
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, List, Tuple, Optional, Any

class AbundanceBarChart:
    """
    A class for generating abundance bar charts from lipidomics data.

    This class processes lipidomics data to calculate and visualize the total 
    abundance of lipid classes under selected experimental conditions. It supports 
    different modes of data representation (linear or logarithmic scale) and provides 
    functionalities for grouping data, calculating mean and standard deviation, 
    handling log transformations, and filtering based on selected classes.
    """

    @staticmethod
    @st.cache_data(ttl=3600)
    def create_mean_std_columns(df: pd.DataFrame, 
                              full_samples_list: List[str], 
                              individual_samples_list: List[List[str]], 
                              conditions_list: List[str], 
                              selected_conditions: List[str], 
                              selected_classes: List[str]) -> pd.DataFrame:
        """Create mean and standard deviation columns for each condition with mathematically rigorous approach."""
        try:
            available_samples = [sample for sample in full_samples_list 
                               if f"concentration[{sample}]" in df.columns]
            grouped_df = pd.DataFrame()
            
            for condition in selected_conditions:
                if condition not in conditions_list:
                    continue
                    
                condition_index = conditions_list.index(condition)
                individual_samples = [sample for sample in individual_samples_list[condition_index] 
                                   if sample in available_samples]
                
                if not individual_samples:
                    continue
                
                mean_cols = [f"concentration[{sample}]" for sample in individual_samples]
                
                # Calculate statistics for each class
                class_stats = []
                for class_name in selected_classes:
                    class_df = df[df['ClassKey'] == class_name]
                    
                    if not class_df.empty:
                        # Get total abundance per sample (sum across all species in the class)
                        total_abundance_per_sample = class_df[mean_cols].sum(axis=0)
                        
                        # LINEAR SCALE STATISTICS
                        linear_mean = total_abundance_per_sample.mean()
                        linear_std = total_abundance_per_sample.std(ddof=1)  # Use sample std (N-1)
                        
                        # LOG10 SCALE STATISTICS - Transform individual values first
                        # Use consistent approach: replace zeros with small value to maintain sample size
                        # This ensures log scale statistics are based on same samples as linear scale
                        
                        # Replace zeros with a small value (e.g., minimum positive value / 10)
                        min_positive = total_abundance_per_sample[total_abundance_per_sample > 0].min()
                        if pd.isna(min_positive):
                            # All values are zero - handle this case
                            log10_mean = np.nan
                            log10_std = np.nan
                        else:
                            # Replace zeros with small value to maintain sample consistency
                            small_value = min_positive / 10
                            adjusted_values = total_abundance_per_sample.replace(0, small_value)
                            
                            log10_values = np.log10(adjusted_values)
                            log10_mean = log10_values.mean()
                            log10_std = log10_values.std(ddof=1) if len(log10_values) > 1 else 0.0
                        
                        class_stats.append({
                            'ClassKey': class_name,
                            f'mean_AUC_{condition}': linear_mean,
                            f'std_AUC_{condition}': linear_std,
                            f'log10_mean_AUC_{condition}': log10_mean,
                            f'log10_std_AUC_{condition}': log10_std
                        })
                
                if not grouped_df.empty:
                    temp_df = pd.DataFrame(class_stats)
                    grouped_df = grouped_df.merge(temp_df, on='ClassKey', how='outer')
                else:
                    grouped_df = pd.DataFrame(class_stats)
            
            grouped_df = AbundanceBarChart.filter_by_selected_classes(grouped_df, selected_classes)
            
            return grouped_df
            
        except Exception as e:
            st.error(f"Error in create_mean_std_columns: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def perform_statistical_tests(continuation_df: pd.DataFrame, 
                                experiment: Any, 
                                selected_conditions: List[str], 
                                selected_classes: List[str],
                                test_type: str = "parametric",
                                correction_method: str = "uncorrected",
                                posthoc_correction: str = "uncorrected",
                                alpha: float = 0.05,
                                auto_transform: bool = True) -> Dict:
        """
        Perform statistical tests (t-test or ANOVA) on the data with user-controlled post-hoc correction.
        
        Args:
            continuation_df: DataFrame containing the lipidomics data
            experiment: Experiment object containing experimental setup information
            selected_conditions: List of conditions to compare
            selected_classes: List of lipid classes to test
            test_type: "parametric", "non_parametric", or "auto"
            correction_method: "uncorrected", "fdr_bh", or "bonferroni" (Level 1 correction)
            posthoc_correction: "uncorrected", "standard", or "bonferroni" (Level 2 correction)
            alpha: Significance threshold
            auto_transform: Whether to apply log10 transformation automatically
            
        Returns:
            Dictionary containing statistical test results
        """
        from scipy.stats import mannwhitneyu, kruskal
        from statsmodels.stats.multitest import multipletests
        from itertools import combinations
        
        def perform_bonferroni_posthoc_internal(continuation_df: pd.DataFrame, 
                                              lipid_class: str, 
                                              selected_conditions: List[str], 
                                              experiment: Any, 
                                              transformation_applied: str,
                                              use_parametric: bool = False) -> Dict:
            """
            Helper function to perform Bonferroni-corrected pairwise comparisons.
            
            Args:
                use_parametric: If True, use Welch's t-test; if False, use Mann-Whitney U
            """
            pairs = list(combinations(selected_conditions, 2))
            pairwise_p_values = []
            
            class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
            condition_data = {}
            
            # Collect data for each condition
            for condition in selected_conditions:
                condition_idx = experiment.conditions_list.index(condition)
                condition_samples = experiment.individual_samples_list[condition_idx]
                sample_columns = [f"concentration[{sample}]" for sample in condition_samples 
                                if f"concentration[{sample}]" in class_df.columns]
                if sample_columns:
                    sample_sums = class_df[sample_columns].sum()
                    
                    # Apply transformation if needed
                    if transformation_applied == "log10":
                        min_positive = min([x for x in sample_sums if x > 0]) if any(x > 0 for x in sample_sums) else 1
                        small_value = min_positive / 10
                        sample_sums = np.log10(np.maximum(sample_sums, small_value))
                    
                    condition_data[condition] = sample_sums.values
            
            # Perform pairwise tests using appropriate test type
            for cond1, cond2 in pairs:
                if cond1 in condition_data and cond2 in condition_data:
                    if use_parametric:
                        # Parametric: Welch's t-test (unequal variances)
                        _, p_val = stats.ttest_ind(condition_data[cond1], condition_data[cond2], 
                                                   equal_var=False)
                    else:
                        # Non-parametric: Mann-Whitney U
                        _, p_val = mannwhitneyu(condition_data[cond1], condition_data[cond2], 
                                              alternative='two-sided')
                    pairwise_p_values.append(p_val)
            
            # Apply Bonferroni correction to pairwise comparisons
            bonf_corrected = multipletests(pairwise_p_values, method='bonferroni')[1]
            
            # Set method name based on test type used
            method_name = "Welch's t-test + Bonferroni" if use_parametric else "Mann-Whitney U + Bonferroni"
            
            tukey_results = {
                'group1': [pair[0] for pair in pairs],
                'group2': [pair[1] for pair in pairs],
                'p_values': bonf_corrected,
                'method': method_name
            }
            
            return tukey_results
        
        statistical_results = {}
        all_p_values = []
        test_info = {
            'transformation_applied': {},
            'test_chosen': {}
        }
    
        for lipid_class in selected_classes:
            try:
                class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                
                # Collect data for each condition
                condition_groups = {}
                for condition in selected_conditions:
                    condition_idx = experiment.conditions_list.index(condition)
                    condition_samples = experiment.individual_samples_list[condition_idx]
                    sample_columns = [f"concentration[{sample}]" for sample in condition_samples 
                                    if f"concentration[{sample}]" in class_df.columns]
                    
                    if sample_columns:
                        sample_sums = class_df[sample_columns].sum()
                        condition_groups[condition] = sample_sums.values
    
                if len(condition_groups) < 2:
                    continue
                    
                # Check for zeros and apply small value replacement if needed
                all_values = np.concatenate(list(condition_groups.values()))
                min_positive = min([x for x in all_values if x > 0]) if any(x > 0 for x in all_values) else 1
                small_value = min_positive / 10
                
                # Replace zeros in condition groups
                for condition in condition_groups:
                    condition_groups[condition] = np.array([max(x, small_value) for x in condition_groups[condition]])
    
                # Apply transformation if auto_transform is enabled
                original_data = list(condition_groups.values())
                transformed_data = original_data.copy()
                transformation_applied = "none"
                
                if auto_transform:
                    # Apply log10 transformation automatically (no normality testing)
                    log_data = [np.log10(group) for group in condition_groups.values()]
                    transformed_data = log_data
                    transformation_applied = "log10"
                
                test_info['transformation_applied'][lipid_class] = transformation_applied
    
                # Choose and perform statistical test
                if len(selected_conditions) == 2:
                    # Two-group comparison
                    group1, group2 = transformed_data[0], transformed_data[1]
                    
                    if test_type == "non_parametric":
                        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                        test_chosen = "Mann-Whitney U"
                    elif test_type in ["parametric", "auto"]:
                        # For both parametric and auto mode, use parametric tests
                        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                        test_chosen = "Welch's t-test"
                    
                    statistical_results[lipid_class] = {
                        'test': test_chosen,
                        'statistic': statistic,
                        'p-value': p_value,
                        'transformation': transformation_applied
                    }
                    all_p_values.append(p_value)
                    
                else:
                    # Multiple group comparison
                    if test_type == "non_parametric":
                        statistic, p_value = kruskal(*transformed_data)
                        test_chosen = "Kruskal-Wallis"
                    elif test_type in ["parametric", "auto"]:
                        # For both parametric and auto mode, use parametric tests
                        from scipy.stats import alexandergovern
                        result = alexandergovern(*transformed_data)
                        statistic = result.statistic
                        p_value = result.pvalue
                        test_chosen = "Welch's ANOVA"
                    
                    statistical_results[lipid_class] = {
                        'test': test_chosen,
                        'statistic': statistic,
                        'p-value': p_value,
                        'transformation': transformation_applied,
                        'tukey_results': None
                    }
                    all_p_values.append(p_value)
                
                test_info['test_chosen'][lipid_class] = test_chosen
                        
            except Exception as e:
                continue
    
        # Multiple testing correction (Level 1 - Between-Class)
        if all_p_values and correction_method != "uncorrected":
            if correction_method == "fdr_bh":
                adjusted = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
                method_name = "Benjamini-Hochberg FDR"
            elif correction_method == "bonferroni":
                adjusted = multipletests(all_p_values, alpha=alpha, method='bonferroni')
                method_name = "Bonferroni"
            else:
                adjusted = (np.array(all_p_values) <= alpha, all_p_values)
                method_name = "Uncorrected"
            
            adjusted_p_values = adjusted[1] if correction_method != "uncorrected" else all_p_values
            significance_flags = adjusted[0]
    
            # Update results with adjusted p-values
            for i, lipid_class in enumerate(statistical_results):
                if not lipid_class.startswith('_'):  # Skip metadata
                    if correction_method != "uncorrected":
                        statistical_results[lipid_class]['adjusted p-value'] = adjusted_p_values[i]
                        statistical_results[lipid_class]['significant'] = significance_flags[i]
                        statistical_results[lipid_class]['correction_method'] = method_name
                    else:
                        statistical_results[lipid_class]['significant'] = all_p_values[i] <= alpha
                        statistical_results[lipid_class]['correction_method'] = "Uncorrected"
    
        # Identify significant omnibus tests for post-hoc analysis
        significant_omnibus_tests = []
        for lipid_class, result in statistical_results.items():
            if not lipid_class.startswith('_'):
                # Track significant omnibus tests for post-hoc analysis
                if (len(selected_conditions) > 2 and 
                    result['test'] in ["Welch's ANOVA", "Kruskal-Wallis"] and
                    result['p-value'] <= alpha):  # Use uncorrected p-value for omnibus qualification
                    significant_omnibus_tests.append(lipid_class)
    
        # Perform post-hoc tests (Level 2 - Within-Class)
        if len(selected_conditions) > 2 and posthoc_correction != "uncorrected" and significant_omnibus_tests:
            for lipid_class in significant_omnibus_tests:
                if lipid_class in statistical_results:
                    try:
                        # Choose post-hoc method based on user selection and original test type
                        if posthoc_correction in ["standard", "tukey"]:
                            # Use standard approach: Tukey for parametric, Bonferroni for non-parametric
                            if ("ANOVA" in statistical_results[lipid_class]['test']):
                                # Parametric post-hoc: Tukey's HSD
                                class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                                data_for_posthoc = []
                                conditions_for_posthoc = []
                                
                                for condition in selected_conditions:
                                    condition_idx = experiment.conditions_list.index(condition)
                                    condition_samples = experiment.individual_samples_list[condition_idx]
                                    sample_columns = [f"concentration[{sample}]" for sample in condition_samples 
                                                    if f"concentration[{sample}]" in class_df.columns]
    
                                    if sample_columns:
                                        sample_sums = class_df[sample_columns].sum()
                                        # Apply same transformation as main test
                                        transformation_applied = statistical_results[lipid_class]['transformation']
                                        if transformation_applied == "log10":
                                            min_positive = min([x for x in sample_sums if x > 0]) if any(x > 0 for x in sample_sums) else 1
                                            small_value = min_positive / 10
                                            sample_sums = np.log10(np.maximum(sample_sums, small_value))
                                        data_for_posthoc.extend(sample_sums.values)
                                        conditions_for_posthoc.extend([condition] * len(sample_sums))
    
                                tukey = pairwise_tukeyhsd(data_for_posthoc, conditions_for_posthoc)
                                tukey_results = {
                                    'group1': [str(res[0]) for res in tukey._results_table[1:]],
                                    'group2': [str(res[1]) for res in tukey._results_table[1:]],
                                    'p_values': tukey.pvalues,
                                    'method': "Tukey's HSD"
                                }
                                statistical_results[lipid_class]['tukey_results'] = tukey_results
                            else:
                                # Non-parametric: use Bonferroni approach with Mann-Whitney U
                                tukey_results = perform_bonferroni_posthoc_internal(
                                    continuation_df, lipid_class, selected_conditions, experiment, 
                                    statistical_results[lipid_class]['transformation'],
                                    use_parametric=False
                                )
                                statistical_results[lipid_class]['tukey_results'] = tukey_results
                                
                        elif posthoc_correction == "bonferroni":
                            # Use Bonferroni correction with test type matching the omnibus test
                            is_parametric = "ANOVA" in statistical_results[lipid_class]['test']
                            tukey_results = perform_bonferroni_posthoc_internal(
                                continuation_df, lipid_class, selected_conditions, experiment, 
                                statistical_results[lipid_class]['transformation'],
                                use_parametric=is_parametric
                            )
                            statistical_results[lipid_class]['tukey_results'] = tukey_results
                            
                    except Exception as e:
                        continue
    
        # Add test information to results
        statistical_results['_test_info'] = test_info
        statistical_results['_parameters'] = {
            'test_type': test_type,
            'correction_method': correction_method,
            'posthoc_correction': posthoc_correction,
            'alpha': alpha,
            'auto_transform': auto_transform
        }
    
        return statistical_results

    @staticmethod
    def create_abundance_bar_chart(df: pd.DataFrame, 
                                 full_samples_list: List[str],
                                 individual_samples_list: List[List[str]], 
                                 conditions_list: List[str],
                                 selected_conditions: List[str], 
                                 selected_classes: List[str],
                                 mode: str,
                                 anova_results: Optional[Dict] = None) -> Tuple[go.Figure, Optional[pd.DataFrame]]:
        """Create abundance bar chart using Plotly with corrected error bar calculations."""
        # Color scheme for conditions
        colors = {
            condition: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, condition in enumerate(selected_conditions)
        }
        
        try:
            # Data validation and preparation
            expected_columns = ['ClassKey'] + [f"concentration[{sample}]" for sample in full_samples_list]
            valid_columns = ['ClassKey'] + [col for col in expected_columns if col in df.columns and col != 'ClassKey']
            df = df[valid_columns]
    
            if df.empty or len(valid_columns) <= 1:
                raise ValueError("No valid data available to create the abundance bar chart.")
    
            if 'ClassKey' not in df.columns:
                raise ValueError("ClassKey column is missing from the dataset.")
            
            selected_conditions = [cond for cond in selected_conditions if cond in conditions_list]
    
            # Create mean and std columns using the corrected method
            abundance_df = AbundanceBarChart.create_mean_std_columns(
                df, full_samples_list, individual_samples_list, 
                conditions_list, selected_conditions, selected_classes
            )
        
            if abundance_df.empty:
                raise ValueError("No data available after processing.")
    
            # Create Plotly figure
            fig = go.Figure()
            
            # Add bars for each condition
            bar_width = 0.8 / len(selected_conditions)
            for i, condition in enumerate(selected_conditions):
                mean, std = AbundanceBarChart.get_mode_specific_values(abundance_df, condition, mode)
                if mean.isnull().all() or std.isnull().all() or mean.empty or std.empty:
                    continue
                
                # Calculate y-positions for bars
                y_positions = np.arange(len(abundance_df)) + (i - len(selected_conditions)/2 + 0.5) * bar_width
                
                # Add error bars (using standard deviation)
                error_x = dict(
                    type='data',
                    array=std,
                    visible=True,
                    color='black',
                    thickness=1,
                    width=0  # This removes the perpendicular line at the end
                )
                
                # Add bars
                fig.add_trace(go.Bar(
                    y=y_positions,
                    x=mean,
                    name=condition,
                    orientation='h',
                    error_x=error_x,
                    width=bar_width,
                    showlegend=True,
                    marker_color=colors[condition]
                ))
            
            # Add significance markers if available
            if anova_results:
                max_x = max(max(trace.x) for trace in fig.data)
                for i, lipid_class in enumerate(abundance_df['ClassKey']):
                    if lipid_class in anova_results:
                        result = anova_results[lipid_class]
                        p_value = result.get('adjusted p-value', result['p-value'])
                        
                        significance = ''
                        if p_value < 0.001:
                            significance = '***'
                        elif p_value < 0.01:
                            significance = '**'
                        elif p_value < 0.05:
                            significance = '*'
                        
                        if significance:
                            fig.add_annotation(
                                x=max_x * 1.1,
                                y=i,
                                text=significance,
                                showarrow=False,
                                font=dict(size=12, color='black')
                            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=AbundanceBarChart._generate_plot_title(mode, selected_conditions),
                    font=dict(color='black')
                ),
                xaxis_title=dict(
                    text='Mean Concentration',
                    font=dict(color='black')
                ),
                yaxis_title=dict(
                    text='Lipid Class',
                    font=dict(color='black')
                ),
                xaxis=dict(
                    tickfont=dict(color='black'),
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    ticktext=abundance_df['ClassKey'].values,
                    tickvals=np.arange(len(abundance_df)),
                    autorange="reversed",
                    tickfont=dict(color='black'),
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(color='black'),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                ),
                margin=dict(r=180),  # Increased right margin for legend outside plot
                height=max(400, len(abundance_df) * 30),  # Dynamic height based on number of classes
                width=800,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig, abundance_df
    
        except Exception as e:
            st.error(f"Error in create_abundance_bar_chart: {str(e)}")
            return None, None

    @staticmethod
    def get_mode_specific_values(abundance_df: pd.DataFrame, 
                               condition: str, 
                               mode: str) -> Tuple[pd.Series, pd.Series]:
        """Get mode-specific values (linear or log10 scale) - returns standard deviation."""
        if mode == 'linear scale':
            mean = abundance_df.get(f"mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            std = abundance_df.get(f"std_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        elif mode in ['log2 scale', 'log10 scale']:  # Accept both for backward compatibility
            mean = abundance_df.get(f"log10_mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            std = abundance_df.get(f"log10_std_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'linear scale', 'log2 scale', or 'log10 scale'")
        return mean, std

    @staticmethod
    def filter_by_selected_classes(grouped_df: pd.DataFrame, 
                                 selected_classes: List[str]) -> pd.DataFrame:
        """Filter DataFrame to include only selected classes."""
        return grouped_df[grouped_df['ClassKey'].isin(selected_classes)]

    @staticmethod
    def _generate_plot_title(mode: str, selected_conditions: List[str]) -> str:
        """Generate an appropriate title for the plot."""
        if mode == 'linear scale':
            scale_type = "Linear"
        elif mode in ['log2 scale', 'log10 scale']:
            scale_type = "Log10"
        else:
            scale_type = "Unknown"
        
        conditions_str = " vs ".join(selected_conditions)
        return f"Class Concentration Bar Chart ({scale_type} Scale)<br>{conditions_str}"