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
                                correction_method: str = "none",
                                alpha: float = 0.05,
                                auto_transform: bool = True) -> Dict:
        """
        Perform statistical tests (t-test or ANOVA) on the data with enhanced options.
        """
        from scipy.stats import shapiro, levene, mannwhitneyu, kruskal
        from statsmodels.stats.multitest import multipletests
        
        statistical_results = {}
        all_p_values = []
        test_info = {
            'transformation_applied': {},
            'normality_tests': {},
            'variance_tests': {},
            'test_chosen': {}
        }
    
        for lipid_class in selected_classes:
            try:
                class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                data_for_stats = []
                conditions_for_stats = []
                
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
                        data_for_stats.extend(sample_sums.values)
                        conditions_for_stats.extend([condition] * len(sample_sums))
                
                if len(condition_groups) < 2:
                    continue
                    
                # Check for zeros and apply small value replacement if needed
                min_positive = min([x for x in data_for_stats if x > 0]) if any(x > 0 for x in data_for_stats) else 1
                small_value = min_positive / 10  # Changed from 1000 to 10
                
                # Replace zeros in condition groups
                for condition in condition_groups:
                    condition_groups[condition] = np.array([max(x, small_value) for x in condition_groups[condition]])
                
                # Determine if transformation is needed
                original_data = list(condition_groups.values())
                transformed_data = original_data.copy()
                transformation_applied = "none"
                
                if auto_transform and test_type in ["auto", "parametric"]:
                    # Test normality on original data
                    combined_data = np.concatenate(original_data)
                    if len(combined_data) >= 3:  # Need at least 3 samples for Shapiro-Wilk
                        try:
                            _, normality_p = shapiro(combined_data)
                            test_info['normality_tests'][lipid_class] = {
                                'original_p_value': normality_p,
                                'is_normal': normality_p > 0.05
                            }
                            
                            # If not normal, try log10 transformation
                            if normality_p <= 0.05:
                                log_data = [np.log10(group) for group in condition_groups.values()]
                                combined_log = np.concatenate(log_data)
                                _, log_normality_p = shapiro(combined_log)
                                
                                test_info['normality_tests'][lipid_class]['log_p_value'] = log_normality_p
                                test_info['normality_tests'][lipid_class]['log_is_normal'] = log_normality_p > 0.05
                                
                                # Use log-transformed data if it's more normal
                                if log_normality_p > normality_p:
                                    transformed_data = log_data
                                    transformation_applied = "log10"
                        except:
                            # If normality test fails, assume non-normal
                            test_info['normality_tests'][lipid_class] = {'test_failed': True}
                
                test_info['transformation_applied'][lipid_class] = transformation_applied
                
                # Test homogeneity of variance only for multi-group comparisons
                equal_variances = True
                if test_type in ["auto", "parametric"] and len(transformed_data) > 2:
                    try:
                        _, variance_p = levene(*transformed_data)
                        equal_variances = variance_p > 0.05
                        test_info['variance_tests'][lipid_class] = {
                            'p_value': variance_p,
                            'equal_variances': equal_variances
                        }
                    except:
                        test_info['variance_tests'][lipid_class] = {'test_failed': True}
                
                # Choose and perform statistical test
                if len(selected_conditions) == 2:
                    # Two-group comparison
                    group1, group2 = transformed_data[0], transformed_data[1]
                    
                    if test_type == "non_parametric":
                        # Mann-Whitney U test
                        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                        test_chosen = "Mann-Whitney U"
                    elif test_type == "parametric":
                        # Welch's t-test (more robust)
                        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                        test_chosen = "Welch's t-test"
                    else:  # auto
                        # Choose based on normality and sample size with more nuanced rules
                        is_normal = test_info['normality_tests'].get(lipid_class, {}).get('is_normal', False)
                        # More nuanced sample size rule: ≥20 reliable, 10-19 if normal, <10 non-parametric
                        min_size = min(len(group1), len(group2))
                        sample_size_ok = min_size >= 20 or (min_size >= 10 and is_normal)
                        
                        if is_normal or sample_size_ok:
                            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                            test_chosen = "Welch's t-test"
                        else:
                            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                            test_chosen = "Mann-Whitney U"
                    
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
                        # Kruskal-Wallis test
                        statistic, p_value = kruskal(*transformed_data)
                        test_chosen = "Kruskal-Wallis"
                    elif test_type == "parametric":
                        # Choose between regular ANOVA and Welch's ANOVA based on variance test
                        if equal_variances:
                            statistic, p_value = stats.f_oneway(*transformed_data)
                            test_chosen = "One-way ANOVA"
                        else:
                            # Use Welch's ANOVA for unequal variances
                            from scipy.stats import alexandergovern
                            try:
                                statistic, p_value = alexandergovern(*transformed_data)
                                test_chosen = "Welch's ANOVA"
                            except:
                                # Fallback to regular ANOVA if Welch's fails
                                statistic, p_value = stats.f_oneway(*transformed_data)
                                test_chosen = "One-way ANOVA (fallback)"
                    else:  # auto
                        # Choose based on normality and sample sizes with CLT rules
                        is_normal = test_info['normality_tests'].get(lipid_class, {}).get('is_normal', False)
                        min_sample_size = min(len(group) for group in transformed_data)
                        # CLT rule: ≥30 reliable, 15-29 if normal, <15 non-parametric
                        sample_size_ok = min_sample_size >= 30 or (min_sample_size >= 15 and is_normal)
                        
                        if is_normal or sample_size_ok:
                            if equal_variances:
                                statistic, p_value = stats.f_oneway(*transformed_data)
                                test_chosen = "One-way ANOVA"
                            else:
                                from scipy.stats import alexandergovern
                                try:
                                    statistic, p_value = alexandergovern(*transformed_data)
                                    test_chosen = "Welch's ANOVA"
                                except:
                                    statistic, p_value = stats.f_oneway(*transformed_data)
                                    test_chosen = "One-way ANOVA (fallback)"
                        else:
                            statistic, p_value = kruskal(*transformed_data)
                            test_chosen = "Kruskal-Wallis"
                    
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
                st.warning(f"Could not perform statistical test for {lipid_class}: {str(e)}")
                continue
    
        # Multiple testing correction
        if all_p_values and correction_method != "none":
            if correction_method == "fdr_bh":
                adjusted = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
                method_name = "Benjamini-Hochberg FDR"
            elif correction_method == "bonferroni":
                adjusted = multipletests(all_p_values, alpha=alpha, method='bonferroni')
                method_name = "Bonferroni"
            else:
                # No correction
                adjusted = (np.array(all_p_values) <= alpha, all_p_values)
                method_name = "None"
            
            adjusted_p_values = adjusted[1] if correction_method != "none" else all_p_values
            significance_flags = adjusted[0]
    
            # Update results with adjusted p-values
            for i, lipid_class in enumerate(statistical_results):
                if correction_method != "none":
                    statistical_results[lipid_class]['adjusted p-value'] = adjusted_p_values[i]
                    statistical_results[lipid_class]['significant'] = significance_flags[i]
                    statistical_results[lipid_class]['correction_method'] = method_name
                else:
                    statistical_results[lipid_class]['significant'] = all_p_values[i] <= alpha
                    statistical_results[lipid_class]['correction_method'] = "None"
    
                # Perform post-hoc tests for significant multi-group results
                if (len(selected_conditions) > 2 and 
                    statistical_results[lipid_class]['test'] in ["One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis"] and
                    statistical_results[lipid_class]['p-value'] <= alpha):  # Use uncorrected p-value for post-hoc decision
                    
                    try:
                        if "ANOVA" in statistical_results[lipid_class]['test']:
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
                                    if transformation_applied == "log10":
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
                            # Non-parametric post-hoc: Pairwise Mann-Whitney with Bonferroni correction
                            from itertools import combinations
                            pairs = list(combinations(selected_conditions, 2))
                            pairwise_p_values = []
                            
                            class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                            condition_data = {}
                            for condition in selected_conditions:
                                condition_idx = experiment.conditions_list.index(condition)
                                condition_samples = experiment.individual_samples_list[condition_idx]
                                sample_columns = [f"concentration[{sample}]" for sample in condition_samples 
                                                if f"concentration[{sample}]" in class_df.columns]
                                if sample_columns:
                                    sample_sums = class_df[sample_columns].sum()
                                    condition_data[condition] = sample_sums.values
                            
                            for cond1, cond2 in pairs:
                                if cond1 in condition_data and cond2 in condition_data:
                                    _, p_val = mannwhitneyu(condition_data[cond1], condition_data[cond2], 
                                                          alternative='two-sided')
                                    pairwise_p_values.append(p_val)
                            
                            # Apply Bonferroni correction to pairwise comparisons
                            bonf_corrected = multipletests(pairwise_p_values, method='bonferroni')[1]
                            
                            tukey_results = {
                                'group1': [pair[0] for pair in pairs],
                                'group2': [pair[1] for pair in pairs],
                                'p_values': bonf_corrected,
                                'method': "Mann-Whitney U + Bonferroni"
                            }
                            statistical_results[lipid_class]['tukey_results'] = tukey_results
                    except Exception as e:
                        st.warning(f"Post-hoc test failed for {lipid_class}: {str(e)}")
        
        # Add test information to results
        statistical_results['_test_info'] = test_info
        statistical_results['_parameters'] = {
            'test_type': test_type,
            'correction_method': correction_method,
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
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=1.15,
                    font=dict(color='black')
                ),
                margin=dict(r=150),  # Add right margin for significance markers
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
        elif mode == 'log2 scale':  # Keep backward compatibility 
            mean = abundance_df.get(f"log10_mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            std = abundance_df.get(f"log10_std_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'linear scale' or 'log2 scale'")
        return mean, std

    @staticmethod
    def filter_by_selected_classes(grouped_df: pd.DataFrame, 
                                 selected_classes: List[str]) -> pd.DataFrame:
        """Filter DataFrame to include only selected classes."""
        return grouped_df[grouped_df['ClassKey'].isin(selected_classes)]

    @staticmethod
    def _generate_plot_title(mode: str, selected_conditions: List[str]) -> str:
        """Generate an appropriate title for the plot."""
        scale_type = "Linear" if mode == 'linear scale' else "Log10"
        conditions_str = " vs ".join(selected_conditions)
        return f"Class Concentration Bar Chart ({scale_type} Scale)<br>{conditions_str}"