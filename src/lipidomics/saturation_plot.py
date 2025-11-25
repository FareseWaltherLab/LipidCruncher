import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import streamlit as st

class SaturationPlot:
    """
    Enhanced SaturationPlot class with rigorous statistical methodology.
    Core logic only - UI components handled in main app.
    """

    @staticmethod
    def _calculate_fa_ratios(mol_structure):
        """
        Calculate the ratios of SFA, MUFA, and PUFA in a molecular structure.
        Args:
            mol_structure (str): Molecular structure string, e.g., 'PA(16:0_20:4)'.
        Returns:
            tuple: Ratios of SFA, MUFA, and PUFA.
        """
        try:
            parts = mol_structure.split('(')[1][:-1].split('_')
            # Extract double bond count, stripping ;XO hydroxyl notation (e.g., ;2O, ;3O)
            fatty_acids = [fa.split(':')[-1].split(';')[0] for fa in parts]
            ratios = [fatty_acids.count(x) for x in ['0', '1']]
            pufa_ratio = len(fatty_acids) - sum(ratios)
            ratios.append(pufa_ratio)
            total = sum(ratios)
            return tuple(ratio / total for ratio in ratios) if total > 0 else (0, 0, 0)
        except (IndexError, ValueError):
            return (0, 0, 0)

    @staticmethod
    @st.cache_data(ttl=3600)
    def _calculate_sfa_mufa_pufa_with_stats(df, condition, samples, lipid_class):
        """
        Calculates SFA, MUFA, and PUFA values for a specific lipid class with proper statistics.
        Returns both individual sample values and statistics for error bar calculation.
        """
        filtered_df = SaturationPlot._filter_df_for_lipid_class(df, lipid_class)
        available_samples = [sample for sample in samples if f'concentration[{sample}]' in filtered_df.columns]
        
        if not available_samples:
            return {
                'SFA': {'values': [], 'mean': 0, 'std': 0},
                'MUFA': {'values': [], 'mean': 0, 'std': 0},
                'PUFA': {'values': [], 'mean': 0, 'std': 0}
            }
        
        # Calculate values for each sample
        fa_data = {'SFA': [], 'MUFA': [], 'PUFA': []}
        
        for sample in available_samples:
            sample_df = filtered_df[['LipidMolec', f'concentration[{sample}]']].copy()
            sample_df['fa_ratios'] = sample_df['LipidMolec'].apply(SaturationPlot._calculate_fa_ratios)
            
            # Calculate weighted contributions for each fatty acid type
            sfa_value = sample_df.apply(lambda x: x[f'concentration[{sample}]'] * x['fa_ratios'][0], axis=1).sum()
            mufa_value = sample_df.apply(lambda x: x[f'concentration[{sample}]'] * x['fa_ratios'][1], axis=1).sum()
            pufa_value = sample_df.apply(lambda x: x[f'concentration[{sample}]'] * x['fa_ratios'][2], axis=1).sum()
            
            fa_data['SFA'].append(sfa_value)
            fa_data['MUFA'].append(mufa_value)
            fa_data['PUFA'].append(pufa_value)
        
        # Calculate statistics
        results = {}
        for fa_type in ['SFA', 'MUFA', 'PUFA']:
            values = np.array(fa_data[fa_type])
            results[fa_type] = {
                'values': values,
                'mean': values.mean(),
                'std': values.std(ddof=1) if len(values) > 1 else 0.0
            }
        
        return results

    @staticmethod
    def _filter_df_for_lipid_class(df, lipid_class):
        """Filters the DataFrame for a specific lipid class."""
        return df[df['ClassKey'] == lipid_class]

    @staticmethod
    def perform_statistical_tests(continuation_df, experiment, selected_conditions, selected_classes,
                                test_type="parametric", correction_method="uncorrected", 
                                posthoc_correction="uncorrected", alpha=0.05, auto_transform=True):
        """
        Perform statistical tests using rigorous methodology with two-level corrections.
        """
        from scipy.stats import mannwhitneyu, kruskal
        
        def perform_bonferroni_posthoc_internal(continuation_df, lipid_class, fa_type, 
                                              selected_conditions, experiment, transformation_applied):
            """Helper function to perform Bonferroni-corrected pairwise comparisons."""
            pairs = list(combinations(selected_conditions, 2))
            pairwise_p_values = []
            condition_data = {}
            
            # Collect data for each condition
            for condition in selected_conditions:
                condition_idx = experiment.conditions_list.index(condition)
                condition_samples = experiment.individual_samples_list[condition_idx]
                fa_stats = SaturationPlot._calculate_sfa_mufa_pufa_with_stats(
                    continuation_df, condition, condition_samples, lipid_class
                )
                
                if fa_stats[fa_type]['values'] is not None and len(fa_stats[fa_type]['values']) > 0:
                    values = fa_stats[fa_type]['values']
                    
                    # Apply transformation with zero handling
                    if transformation_applied == "log10":
                        min_positive = min([x for x in values if x > 0]) if any(x > 0 for x in values) else 1
                        small_value = min_positive / 10
                        values = np.log10(np.maximum(values, small_value))
                    
                    condition_data[condition] = values
            
            # Perform pairwise tests
            for cond1, cond2 in pairs:
                if cond1 in condition_data and cond2 in condition_data:
                    _, p_val = mannwhitneyu(condition_data[cond1], condition_data[cond2], 
                                          alternative='two-sided')
                    pairwise_p_values.append(p_val)
            
            # Apply Bonferroni correction to pairwise comparisons
            bonf_corrected = multipletests(pairwise_p_values, method='bonferroni')[1]
            
            return {
                'group1': [pair[0] for pair in pairs],
                'group2': [pair[1] for pair in pairs],
                'p_values': bonf_corrected,
                'method': "Mann-Whitney U + Bonferroni"
            }
        
        statistical_results = {}
        all_p_values = []
        test_info = {
            'transformation_applied': {},
            'test_chosen': {}
        }
        
        # Test each combination of lipid class and fatty acid type
        for lipid_class in selected_classes:
            statistical_results[lipid_class] = {}
            
            for fa_type in ['SFA', 'MUFA', 'PUFA']:
                try:
                    # Collect data for each condition
                    condition_groups = {}
                    for condition in selected_conditions:
                        condition_idx = experiment.conditions_list.index(condition)
                        condition_samples = experiment.individual_samples_list[condition_idx]
                        
                        fa_stats = SaturationPlot._calculate_sfa_mufa_pufa_with_stats(
                            continuation_df, condition, condition_samples, lipid_class
                        )
                        
                        if (fa_stats[fa_type]['values'] is not None and 
                            len(fa_stats[fa_type]['values']) > 0):
                            condition_groups[condition] = fa_stats[fa_type]['values']
    
                    if len(condition_groups) < 2:
                        continue
                        
                    # Check for zeros and apply small value replacement
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
                        log_data = [np.log10(group) for group in condition_groups.values()]
                        transformed_data = log_data
                        transformation_applied = "log10"
                    
                    test_key = f"{lipid_class}_{fa_type}"
                    test_info['transformation_applied'][test_key] = transformation_applied
    
                    # Choose and perform statistical test
                    if len(selected_conditions) == 2:
                        # Two-group comparison
                        group1, group2 = transformed_data[0], transformed_data[1]
                        
                        if test_type == "non_parametric":
                            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                            test_chosen = "Mann-Whitney U"
                        elif test_type in ["parametric", "auto"]:
                            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                            test_chosen = "Welch's t-test"
                        
                        # Check for NaN p-values
                        if np.isnan(p_value) or np.isnan(statistic):
                            continue
                        
                        statistical_results[lipid_class][fa_type] = {
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
                            try:
                                from scipy.stats import alexandergovern
                                result = alexandergovern(*transformed_data)
                                statistic = result.statistic
                                p_value = result.pvalue
                                test_chosen = "Welch's ANOVA"
                            except (ImportError, AttributeError):
                                statistic, p_value = stats.f_oneway(*transformed_data)
                                test_chosen = "One-way ANOVA (Welch's unavailable)"
                            except Exception:
                                statistic, p_value = stats.f_oneway(*transformed_data)
                                test_chosen = "One-way ANOVA (fallback)"
                        
                        # Check for NaN p-values
                        if np.isnan(p_value) or np.isnan(statistic):
                            continue
                        
                        statistical_results[lipid_class][fa_type] = {
                            'test': test_chosen,
                            'statistic': statistic,
                            'p-value': p_value,
                            'transformation': transformation_applied,
                            'tukey_results': None
                        }
                        all_p_values.append(p_value)
                    
                    test_info['test_chosen'][test_key] = test_chosen
                            
                except Exception:
                    continue
        
        # Multiple testing correction (Level 1 - Between Class/FA combinations)
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
            p_idx = 0
            for lipid_class in statistical_results:
                for fa_type in statistical_results[lipid_class]:
                    if correction_method != "uncorrected":
                        statistical_results[lipid_class][fa_type]['adjusted p-value'] = adjusted_p_values[p_idx]
                        statistical_results[lipid_class][fa_type]['significant'] = significance_flags[p_idx]
                        statistical_results[lipid_class][fa_type]['correction_method'] = method_name
                    else:
                        statistical_results[lipid_class][fa_type]['significant'] = all_p_values[p_idx] <= alpha
                        statistical_results[lipid_class][fa_type]['correction_method'] = "Uncorrected"
                    p_idx += 1
        else:
            # No correction applied
            for lipid_class in statistical_results:
                for fa_type in statistical_results[lipid_class]:
                    p_val = statistical_results[lipid_class][fa_type]['p-value']
                    statistical_results[lipid_class][fa_type]['significant'] = p_val <= alpha
                    statistical_results[lipid_class][fa_type]['correction_method'] = "Uncorrected"
    
        # Identify significant omnibus tests for post-hoc analysis using adjusted p-values
        significant_omnibus_tests = []
        for lipid_class in statistical_results:
            for fa_type in statistical_results[lipid_class]:
                result = statistical_results[lipid_class][fa_type]
                
                # Use adjusted p-value if available, otherwise use raw p-value
                p_value_to_check = result.get('adjusted p-value', result['p-value'])
                
                if (len(selected_conditions) > 2 and 
                    result['test'] in ["One-way ANOVA", "Welch's ANOVA", "One-way ANOVA (fallback)", 
                                     "One-way ANOVA (Welch's unavailable)", "Kruskal-Wallis"] and
                    p_value_to_check <= alpha):
                    significant_omnibus_tests.append((lipid_class, fa_type))
    
        # Perform post-hoc tests (Level 2 - Within-Class/FA combination)
        if len(selected_conditions) > 2 and posthoc_correction != "uncorrected" and significant_omnibus_tests:
            for lipid_class, fa_type in significant_omnibus_tests:
                try:
                    if posthoc_correction == "standard":
                        # Use standard approach: Tukey for parametric, Bonferroni for non-parametric
                        if ("ANOVA" in statistical_results[lipid_class][fa_type]['test']):
                            # Parametric post-hoc: Tukey's HSD
                            data_for_posthoc = []
                            conditions_for_posthoc = []
                            
                            for condition in selected_conditions:
                                condition_idx = experiment.conditions_list.index(condition)
                                condition_samples = experiment.individual_samples_list[condition_idx]
                                fa_stats = SaturationPlot._calculate_sfa_mufa_pufa_with_stats(
                                    continuation_df, condition, condition_samples, lipid_class
                                )
    
                                if (fa_stats[fa_type]['values'] is not None and 
                                    len(fa_stats[fa_type]['values']) > 0):
                                    values = fa_stats[fa_type]['values']
                                    # Apply same transformation and zero handling as main test
                                    transformation_applied = statistical_results[lipid_class][fa_type]['transformation']
                                    if transformation_applied == "log10":
                                        min_positive = min([x for x in values if x > 0]) if any(x > 0 for x in values) else 1
                                        small_value = min_positive / 10
                                        values = np.log10(np.maximum(values, small_value))
                                    data_for_posthoc.extend(values)
                                    conditions_for_posthoc.extend([condition] * len(values))
    
                            tukey = pairwise_tukeyhsd(data_for_posthoc, conditions_for_posthoc)
                            tukey_results = {
                                'group1': [str(res[0]) for res in tukey._results_table[1:]],
                                'group2': [str(res[1]) for res in tukey._results_table[1:]],
                                'p_values': tukey.pvalues,
                                'method': "Tukey's HSD"
                            }
                            statistical_results[lipid_class][fa_type]['tukey_results'] = tukey_results
                        else:
                            # Non-parametric: use Bonferroni approach
                            tukey_results = perform_bonferroni_posthoc_internal(
                                continuation_df, lipid_class, fa_type, selected_conditions, experiment, 
                                statistical_results[lipid_class][fa_type]['transformation']
                            )
                            statistical_results[lipid_class][fa_type]['tukey_results'] = tukey_results
                            
                    elif posthoc_correction == "bonferroni_all":
                        # Always use Bonferroni regardless of test type
                        tukey_results = perform_bonferroni_posthoc_internal(
                            continuation_df, lipid_class, fa_type, selected_conditions, experiment, 
                            statistical_results[lipid_class][fa_type]['transformation']
                        )
                        statistical_results[lipid_class][fa_type]['tukey_results'] = tukey_results
                        
                except Exception:
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
    def create_plots(df, experiment, selected_conditions, statistical_results=None, show_significance=False):
        """Generates saturation plots for all lipid classes with optional statistical annotations."""
        plots = {}
        single_sample_conditions = []
        valid_conditions = []
    
        # Check sample counts for each condition
        for condition in selected_conditions:
            samples = experiment.individual_samples_list[experiment.conditions_list.index(condition)]
            if len(samples) <= 1:
                single_sample_conditions.append(condition)
            else:
                valid_conditions.append(condition)
        
        # Alert user about excluded conditions
        if single_sample_conditions:
            import streamlit as st
            st.warning(f"The following conditions were excluded due to having only one sample: {', '.join(single_sample_conditions)}")
        
        if not valid_conditions:
            import streamlit as st
            st.error("No conditions with multiple samples available for statistical analysis.")
            return plots
        
        # Create plots for valid conditions
        for lipid_class in df['ClassKey'].unique():
            plot_data = SaturationPlot._prepare_plot_data(df, valid_conditions, lipid_class, experiment)
            if not plot_data.empty:
                class_stats = statistical_results.get(lipid_class, {}) if statistical_results else {}
                
                # Pass show_significance parameter to the plot creation function
                main_plot = SaturationPlot._create_auc_plot(plot_data, lipid_class, class_stats, show_significance)
                percentage_df = SaturationPlot._calculate_percentage_df(plot_data)
                percentage_plot = SaturationPlot._create_percentage_plot(percentage_df, lipid_class)
                plots[lipid_class] = (main_plot, percentage_plot, plot_data)
        
        return plots

    @staticmethod
    def _prepare_plot_data(df, valid_conditions, lipid_class, experiment):
        """Prepares plot data for a given lipid class across valid conditions with proper statistics."""
        plot_data = []
        for condition in valid_conditions:
            samples = experiment.individual_samples_list[experiment.conditions_list.index(condition)]
            if len(samples) > 1:
                fa_stats = SaturationPlot._calculate_sfa_mufa_pufa_with_stats(
                    df, condition, samples, lipid_class
                )
                
                # Check if we have meaningful data
                has_data = any(
                    fa_stats[fa_type]['values'] is not None and 
                    len(fa_stats[fa_type]['values']) > 0 and 
                    fa_stats[fa_type]['mean'] > 0
                    for fa_type in ['SFA', 'MUFA', 'PUFA']
                )
                
                if has_data:
                    plot_data.append({
                        'Condition': f"{condition} (n={len(samples)})",
                        'SFA_AUC': fa_stats['SFA']['mean'],
                        'MUFA_AUC': fa_stats['MUFA']['mean'],
                        'PUFA_AUC': fa_stats['PUFA']['mean'],
                        'SFA_std': fa_stats['SFA']['std'],
                        'MUFA_std': fa_stats['MUFA']['std'],
                        'PUFA_std': fa_stats['PUFA']['std'],
                        'SFA_values': fa_stats['SFA']['values'],
                        'MUFA_values': fa_stats['MUFA']['values'],
                        'PUFA_values': fa_stats['PUFA']['values']
                    })
        
        return pd.DataFrame(plot_data)

    @staticmethod
    @st.cache_data(ttl=3600)
    def _calculate_percentage_df(main_df):
        """Calculates the percentage distribution of SFA, MUFA, and PUFA in the dataset."""
        if main_df.empty:
            return pd.DataFrame()
            
        total_auc = main_df[['SFA_AUC', 'MUFA_AUC', 'PUFA_AUC']].sum(axis=1)
        percentage_df = 100 * main_df[['SFA_AUC', 'MUFA_AUC', 'PUFA_AUC']].div(total_auc, axis=0)
        percentage_df['Condition'] = main_df['Condition']
        return percentage_df

    @staticmethod
    def _create_auc_plot(df, lipid_class, statistical_results, show_significance=False):
        """Creates an AUC plot with standard deviation error bars and optional statistical annotations."""
        if df.empty:
            return None
        
        fig = go.Figure()
        colors = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
        bar_width = 0.25
        offsets = [-bar_width, 0, bar_width]
        
        # Add bars for each fatty acid type
        for idx, fa in enumerate(['SFA', 'MUFA', 'PUFA']):
            fig.add_trace(go.Bar(
                x=[c + offsets[idx] for c in range(len(df['Condition']))],
                y=df[f'{fa}_AUC'],
                name=fa,
                marker_color=colors[fa],
                error_y=dict(
                    type='data',
                    array=df[f'{fa}_std'],
                    visible=True,
                    color='black',
                    thickness=1,
                    width=0
                ),
                width=bar_width,
                offset=0
            ))
    
        # Set y-axis to start from 0 to prevent negative error bars
        y_max = max(
            df[f'{fa}_AUC'].max() + df[f'{fa}_std'].max() 
            for fa in ['SFA', 'MUFA', 'PUFA'] 
            if f'{fa}_AUC' in df.columns and f'{fa}_std' in df.columns
        )
        
        fig.update_layout(
            title=dict(text=f"Concentration Profile of Fatty Acids in {lipid_class}", 
                      font=dict(size=24, color='black')),
            xaxis_title=dict(text="Condition", font=dict(size=18, color='black')),
            yaxis_title=dict(text="Concentration", font=dict(size=18, color='black')),
            xaxis=dict(
                tickfont=dict(size=14, color='black'),
                tickmode='array',
                tickvals=list(range(len(df['Condition']))),
                ticktext=df['Condition']
            ),
            yaxis=dict(
                tickfont=dict(size=14, color='black'),
                range=[0, y_max * 1.1]  # Start from 0, add 10% padding at top
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            barmode='group',
            bargap=0.15,
            bargroupgap=0,
            height=500,
            margin=dict(t=50, r=150, b=50, l=50)
        )
    
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
        # Add statistical annotations only if requested and available
        if show_significance and statistical_results:
            SaturationPlot._add_enhanced_statistical_annotations(fig, df, statistical_results, lipid_class)
    
        return fig

    @staticmethod
    def _create_percentage_plot(df, lipid_class):
        """Creates a percentage distribution plot for a given lipid class using Plotly."""
        if df.empty:
            return None
        
        fig = go.Figure()
        colors = ['#c9d9d3', '#718dbf', '#e84d60']
        
        for idx, fa in enumerate(['SFA', 'MUFA', 'PUFA']):
            fig.add_trace(go.Bar(
                x=df['Condition'],
                y=df[f'{fa}_AUC'],
                name=fa,
                marker_color=colors[idx]
            ))

        fig.update_layout(
            title=dict(text=f"Percentage Distribution of Fatty Acids in {lipid_class}", 
                      font=dict(size=24, color='black')),
            xaxis_title=dict(text="Condition", font=dict(size=18, color='black')),
            yaxis_title=dict(text="Percentage", font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            barmode='stack',
            height=500,
            margin=dict(t=50, r=50, b=50, l=50)
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True, range=[0, 100])

        return fig

    @staticmethod
    def _add_enhanced_statistical_annotations(fig, df, statistical_results, lipid_class):
        """Enhanced statistical annotations with proper significance symbols and layout."""
        # Calculate plot dimensions
        y_max = max(
            df[f'{fa}_AUC'].max() + df[f'{fa}_std'].max() 
            for fa in ['SFA', 'MUFA', 'PUFA'] 
            if f'{fa}_AUC' in df.columns and f'{fa}_std' in df.columns
        )
        y_min = min(
            df[f'{fa}_AUC'].min() - df[f'{fa}_std'].max()
            for fa in ['SFA', 'MUFA', 'PUFA'] 
            if f'{fa}_AUC' in df.columns and f'{fa}_std' in df.columns
        )
        
        y_range = y_max - y_min
        y_increment = y_range * 0.15
        
        # Colors and offsets for each fatty acid type
        colors = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
        bar_width = 0.25
        offsets = {'SFA': -bar_width, 'MUFA': 0, 'PUFA': bar_width}
        
        annotation_count = 0
        
        # Add significance annotations for each fatty acid type
        for fa in ['SFA', 'MUFA', 'PUFA']:
            if fa in statistical_results:
                result = statistical_results[fa]
                
                # Get adjusted p-value if available, otherwise use raw p-value
                p_value = result.get('adjusted p-value', result.get('p-value', 1.0))
                
                if len(df) == 2:  # Two conditions
                    if p_value < 0.05:
                        x0 = 0 + offsets[fa]
                        x1 = 1 + offsets[fa]
                        y = y_max + y_increment * (annotation_count + 1)
                        
                        # Add significance line
                        fig.add_shape(
                            type="line",
                            x0=x0, y0=y, x1=x1, y1=y,
                            line=dict(color=colors[fa], width=2)
                        )
                        
                        # Add significance symbol
                        symbol = SaturationPlot._get_significance_symbol(p_value)
                        fig.add_annotation(
                            x=(x0+x1)/2, y=y+y_increment*0.1,
                            text=symbol,
                            showarrow=False,
                            font=dict(size=20, color=colors[fa])
                        )
                        
                        annotation_count += 1
                
                elif len(df) > 2:  # Multiple conditions with post-hoc
                    if (result.get('p-value', 1.0) < 0.05 and 
                        result.get('tukey_results') is not None):
                        # Add post-hoc pairwise comparisons
                        tukey_results = result['tukey_results']
                        
                        for i, (group1, group2, p_val) in enumerate(zip(
                            tukey_results['group1'], 
                            tukey_results['group2'], 
                            tukey_results['p_values']
                        )):
                            if p_val < 0.05:
                                # Find x positions for the groups
                                try:
                                    # Remove sample size info for matching
                                    clean_conditions = [cond.split(' (n=')[0] for cond in df['Condition']]
                                    x0_idx = clean_conditions.index(group1)
                                    x1_idx = clean_conditions.index(group2)
                                    
                                    x0 = x0_idx + offsets[fa]
                                    x1 = x1_idx + offsets[fa]
                                    y = y_max + y_increment * (annotation_count + 1)
                                    
                                    # Add significance line
                                    fig.add_shape(
                                        type="line",
                                        x0=x0, y0=y, x1=x1, y1=y,
                                        line=dict(color=colors[fa], width=2)
                                    )
                                    
                                    # Add significance symbol
                                    symbol = SaturationPlot._get_significance_symbol(p_val)
                                    fig.add_annotation(
                                        x=(x0+x1)/2, y=y+y_increment*0.1,
                                        text=symbol,
                                        showarrow=False,
                                        font=dict(size=20, color=colors[fa])
                                    )
                                    
                                    annotation_count += 1
                                except (ValueError, IndexError):
                                    continue
        
        # Adjust y-axis range to accommodate annotations
        if annotation_count > 0:
            y_axis_max = y_max + y_increment * (annotation_count + 2)
            fig.update_layout(yaxis_range=[max(0, y_min - y_range * 0.05), y_axis_max])

    @staticmethod
    def _get_significance_symbol(p_value):
        """Returns a symbol representing the level of significance."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""

    @staticmethod
    def apply_auto_mode_logic(selected_classes, selected_conditions):
        """Auto mode logic for intelligent statistical defaults."""
        # For saturation plots, we test 3 fatty acid types per class
        total_tests = len(selected_classes) * 3  # SFA, MUFA, PUFA for each class
        
        # Auto Level 1 (Between Class/FA combination) Correction
        if total_tests <= 3:  # Single class analysis
            auto_correction_method = "uncorrected"
            auto_rationale = "Single class (≤3 tests) → no correction needed"
        elif total_tests <= 15:  # Up to 5 classes
            auto_correction_method = "fdr_bh"
            auto_rationale = f"Moderate testing burden ({total_tests} tests) → FDR recommended"
        else:
            auto_correction_method = "fdr_bh"
            auto_rationale = f"High testing burden ({total_tests} tests) → FDR for discovery balance"
        
        # Auto Level 2 (Within-Class/FA combination) Correction
        if len(selected_conditions) <= 2:
            auto_posthoc_correction = "uncorrected"
            auto_posthoc_rationale = "≤2 conditions → no post-hoc needed"
        elif len(selected_conditions) <= 4:
            auto_posthoc_correction = "standard"
            auto_posthoc_rationale = "Few conditions → standard post-hoc approach"
        else:
            auto_posthoc_correction = "bonferroni_all"
            auto_posthoc_rationale = "Many conditions → conservative Bonferroni"
        
        return {
            'correction_method': auto_correction_method,
            'correction_rationale': auto_rationale,
            'posthoc_correction': auto_posthoc_correction,
            'posthoc_rationale': auto_posthoc_rationale,
            'total_tests': total_tests
        }
    
    @staticmethod
    def identify_consolidated_lipids(continuation_df, selected_classes):
        """
        Identify lipids that may be in consolidated format (e.g., PC(34:1) instead of PC(16:0_18:1)).
        Excludes lipid classes that are always single-chain.
        
        Args:
            continuation_df: DataFrame containing lipid data
            selected_classes: List of selected lipid classes to analyze
            
        Returns:
            dict: Dictionary mapping lipid class to list of potentially consolidated lipids
        """
        # Lipid classes that typically have only one fatty acid chain
        SINGLE_CHAIN_CLASSES = {
            'CE', 'Cer', 'CerG1', 'CerG2', 'CerG3', 'SM', 'LPC', 'LPE', 'LPG', 
            'LPI', 'LPS', 'LPA', 'MAG', 'ChE', 'FFA'
        }
        
        consolidated_lipids = {}
        
        for lipid_class in selected_classes:
            # Skip classes that are always single-chain
            if lipid_class in SINGLE_CHAIN_CLASSES:
                continue
                
            # Get lipids for this class
            class_df = continuation_df[continuation_df['ClassKey'] == lipid_class]
            
            # Find lipids without underscore (consolidated format)
            consolidated = []
            for lipid in class_df['LipidMolec']:
                lipid_str = str(lipid)
                # Check if lipid has parentheses and no underscore
                if '(' in lipid_str and ')' in lipid_str:
                    content = lipid_str.split('(')[1].split(')')[0]
                    if '_' not in content and ':' in content:
                        consolidated.append(lipid)
            
            if consolidated:
                consolidated_lipids[lipid_class] = consolidated
        
        return consolidated_lipids