import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
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
    handling log transformations, and filtering based on selected classes. The class 
    also includes methods for customizing and rendering the final plot.
    """

    @staticmethod
    @st.cache_data(ttl=3600)
    def create_mean_std_columns(df: pd.DataFrame, 
                              full_samples_list: List[str], 
                              individual_samples_list: List[List[str]], 
                              conditions_list: List[str], 
                              selected_conditions: List[str], 
                              selected_classes: List[str]) -> pd.DataFrame:
        """Create mean and standard error columns for each condition."""
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
                        total_abundance_per_sample = class_df[mean_cols].sum(axis=0)
                        
                        class_mean = total_abundance_per_sample.mean()
                        class_sem = total_abundance_per_sample.sem()
                        
                        class_stats.append({
                            'ClassKey': class_name,
                            f'mean_AUC_{condition}': class_mean,
                            f'sem_AUC_{condition}': class_sem
                        })
                
                if not grouped_df.empty:
                    temp_df = pd.DataFrame(class_stats)
                    grouped_df = grouped_df.merge(temp_df, on='ClassKey', how='outer')
                else:
                    grouped_df = pd.DataFrame(class_stats)
            
            grouped_df = AbundanceBarChart.filter_by_selected_classes(grouped_df, selected_classes)
            grouped_df = AbundanceBarChart.calculate_log2_values(grouped_df, selected_conditions)
            
            return grouped_df
            
        except Exception as e:
            st.error(f"Error in create_mean_std_columns: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
    def group_and_sum(df: pd.DataFrame, full_samples_list: List[str]) -> pd.DataFrame:
        """
        Groups and sums the mean area values for each lipid class based on the full sample list.
        """
        return df.groupby('ClassKey')[[f"concentration[{sample}]" for sample in full_samples_list]].sum().reset_index()

    @staticmethod
    def calculate_mean_std_for_conditions(grouped_df: pd.DataFrame, 
                                        individual_samples_list: List[List[str]], 
                                        conditions_list: List[str], 
                                        selected_conditions: List[str]):
        """
        Calculates mean and standard deviation for specific conditions.
        """
        for condition in selected_conditions:
            condition_index = conditions_list.index(condition)
            individual_samples = individual_samples_list[condition_index]
            mean_cols = [f"concentration[{sample}]" for sample in individual_samples]
            grouped_df[f"mean_AUC_{condition}"] = grouped_df[mean_cols].mean(axis=1)
            grouped_df[f"sem_AUC_{condition}"] = grouped_df[mean_cols].sem(axis=1)

    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_log2_values(grouped_df: pd.DataFrame, 
                            selected_conditions: List[str]) -> pd.DataFrame:
        """Calculate log2 values for mean and SEM."""
        for condition in selected_conditions:
            mean_col = f"mean_AUC_{condition}"
            sem_col = f"sem_AUC_{condition}"
            
            if mean_col in grouped_df.columns and sem_col in grouped_df.columns:
                log2_mean_col = f"log2_mean_AUC_{condition}"
                log2_sem_col = f"log2_sem_AUC_{condition}"
                
                grouped_df[log2_mean_col] = np.log2(grouped_df[mean_col].replace(0, np.nan))
                grouped_df[log2_sem_col] = grouped_df[sem_col] / (grouped_df[mean_col] * np.log(2))
        
        return grouped_df

    @staticmethod
    def filter_by_selected_classes(grouped_df: pd.DataFrame, 
                                 selected_classes: List[str]) -> pd.DataFrame:
        """Filter DataFrame to include only selected classes."""
        return grouped_df[grouped_df['ClassKey'].isin(selected_classes)]
    
    @staticmethod
    def perform_statistical_tests(continuation_df: pd.DataFrame, 
                                experiment: Any, 
                                selected_conditions: List[str], 
                                selected_classes: List[str]) -> Dict:
        """Perform statistical tests (t-test or ANOVA) on the data."""
        statistical_results = {}
        all_p_values = []
    
        for lipid_class in selected_classes:
            try:
                class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                data_for_stats = []
                conditions_for_stats = []
                
                for condition in selected_conditions:
                    condition_idx = experiment.conditions_list.index(condition)
                    condition_samples = experiment.individual_samples_list[condition_idx]
                    sample_columns = [f"concentration[{sample}]" for sample in condition_samples 
                                    if f"concentration[{sample}]" in class_df.columns]
                    
                    if sample_columns:
                        sample_sums = class_df[sample_columns].sum()
                        data_for_stats.extend(sample_sums.values)
                        conditions_for_stats.extend([condition] * len(sample_sums))
                
                if len(selected_conditions) == 2:
                    # T-test for two conditions
                    group1 = [x for x, c in zip(data_for_stats, conditions_for_stats) 
                            if c == selected_conditions[0]]
                    group2 = [x for x, c in zip(data_for_stats, conditions_for_stats) 
                            if c == selected_conditions[1]]
                    
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
                        statistical_results[lipid_class] = {
                            'test': 't-test',
                            'statistic': t_stat,
                            'p-value': p_value
                        }
                        all_p_values.append(p_value)
                else:
                    # ANOVA for more than two conditions
                    groups = [
                        [x for x, c in zip(data_for_stats, conditions_for_stats) if c == condition]
                        for condition in selected_conditions
                    ]
                    groups = [g for g in groups if g]  # Remove empty groups
                    
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        statistical_results[lipid_class] = {
                            'test': 'ANOVA',
                            'statistic': f_stat,
                            'p-value': p_value,
                            'tukey_results': None
                        }
                        all_p_values.append(p_value)
                        
            except Exception as e:
                st.warning(f"Could not perform statistical test for {lipid_class}: {str(e)}")
                continue
    
        # Multiple testing correction
        if all_p_values:
            from statsmodels.stats.multitest import multipletests
            adjusted = multipletests(all_p_values, alpha=0.05, method='fdr_bh')
            adjusted_p_values = adjusted[1]
            significance_flags = adjusted[0]
    
            # Update results with adjusted p-values
            for i, lipid_class in enumerate(statistical_results):
                statistical_results[lipid_class]['adjusted p-value'] = adjusted_p_values[i]
                statistical_results[lipid_class]['significant'] = significance_flags[i]
    
                # Perform Tukey's test for significant ANOVA results
                if (statistical_results[lipid_class]['test'] == 'ANOVA' and 
                    significance_flags[i]):
                    class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                    data_for_stats = []
                    conditions_for_stats = []
                    
                    for condition in selected_conditions:
                        condition_idx = experiment.conditions_list.index(condition)
                        condition_samples = experiment.individual_samples_list[condition_idx]
                        sample_columns = [f"concentration[{sample}]" for sample in condition_samples 
                                        if f"concentration[{sample}]" in class_df.columns]
    
                        if sample_columns:
                            sample_sums = class_df[sample_columns].sum()
                            data_for_stats.extend(sample_sums.values)
                            conditions_for_stats.extend([condition] * len(sample_sums))
    
                    tukey = pairwise_tukeyhsd(data_for_stats, conditions_for_stats)
                    tukey_results = {
                        'group1': [str(res[0]) for res in tukey._results_table[1:]],
                        'group2': [str(res[1]) for res in tukey._results_table[1:]],
                        'p_values': tukey.pvalues
                    }
                    statistical_results[lipid_class]['tukey_results'] = tukey_results
    
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
        """Create abundance bar chart using Plotly."""
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
    
            # Create mean and std columns
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
                mean, sem = AbundanceBarChart.get_mode_specific_values(abundance_df, condition, mode)
                if mean.isnull().all() or sem.isnull().all() or mean.empty or sem.empty:
                    continue
                
                # Calculate y-positions for bars
                y_positions = np.arange(len(abundance_df)) + (i - len(selected_conditions)/2 + 0.5) * bar_width
                
                # Add error bars
                error_x = dict(
                    type='data',
                    array=sem,
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
                    showlegend=True
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
            logging.error(f"Error in create_abundance_bar_chart: {str(e)}")
            return None, None

    @staticmethod
    def get_mode_specific_values(abundance_df: pd.DataFrame, 
                               condition: str, 
                               mode: str) -> Tuple[pd.Series, pd.Series]:
        """Get mode-specific values (linear or log2 scale)."""
        if mode == 'linear scale':
            mean = abundance_df.get(f"mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            sem = abundance_df.get(f"sem_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        elif mode == 'log2 scale':
            mean = abundance_df.get(f"log2_mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            sem = abundance_df.get(f"log2_sem_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'linear scale' or 'log2 scale'")
        return mean, sem

    @staticmethod
    def _generate_plot_title(mode: str, selected_conditions: List[str]) -> str:
        """Generate an appropriate title for the plot."""
        scale_type = "Linear" if mode == 'linear scale' else "Log2"
        conditions_str = " vs ".join(selected_conditions)
        return f"Class Concentration Bar Chart ({scale_type} Scale)<br>{conditions_str}"