import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st

class SaturationPlot:

    @staticmethod
    def _calculate_fa_ratios(mol_structure):
        """
        Calculate the ratios of SFA, MUFA, and PUFA in a molecular structure.
        Args:
            mol_structure (str): Molecular structure string, e.g., 'PA(16:0_20:4)'.
        Returns:
            tuple: Ratios of SFA, MUFA, and PUFA.
        """
        parts = mol_structure.split('(')[1][:-1].split('_')
        fatty_acids = [fa.split(':')[-1] for fa in parts]
        ratios = [fatty_acids.count(x) for x in ['0', '1']]
        pufa_ratio = len(fatty_acids) - sum(ratios)
        ratios.append(pufa_ratio)
        total = sum(ratios)
        return tuple(ratio / total for ratio in ratios)

    @staticmethod
    def _calculate_sfa_mufa_pufa(df, condition, samples, lipid_class):
        """
        Calculates SFA, MUFA, and PUFA values for a specific lipid class under given conditions.
        """
        filtered_df = SaturationPlot._filter_df_for_lipid_class(df, lipid_class)
        available_samples = [sample for sample in samples if f'MeanArea[{sample}]' in filtered_df.columns]
        
        if not available_samples:
            st.warning(f"No data available for {lipid_class} in condition {condition}")
            return ([], [], [], 0, 0, 0)  # Return empty lists and zeros if no data is available
        
        sfa_values, mufa_values, pufa_values = [], [], []
        
        for sample in available_samples:
            sample_df = filtered_df[['LipidMolec', f'MeanArea[{sample}]']].copy()
            sample_df['fa_ratios'] = sample_df['LipidMolec'].apply(SaturationPlot._calculate_fa_ratios)
            
            sfa_value = sample_df.apply(lambda x: x[f'MeanArea[{sample}]'] * x['fa_ratios'][0], axis=1).sum()
            mufa_value = sample_df.apply(lambda x: x[f'MeanArea[{sample}]'] * x['fa_ratios'][1], axis=1).sum()
            pufa_value = sample_df.apply(lambda x: x[f'MeanArea[{sample}]'] * x['fa_ratios'][2], axis=1).sum()
            
            sfa_values.append(sfa_value)
            mufa_values.append(mufa_value)
            pufa_values.append(pufa_value)
        
        # Calculate means
        sfa_mean = np.mean(sfa_values) if sfa_values else 0
        mufa_mean = np.mean(mufa_values) if mufa_values else 0
        pufa_mean = np.mean(pufa_values) if pufa_values else 0
        
        return (sfa_values, mufa_values, pufa_values, sfa_mean, mufa_mean, pufa_mean)

    @staticmethod
    def _filter_df_for_lipid_class(df, lipid_class):
        """
        Filters the DataFrame for a specific lipid class.
        Args:
            df (pd.DataFrame): The DataFrame containing lipidomics data.
            lipid_class (str): The lipid class to filter for.
        Returns:
            pd.DataFrame: A filtered DataFrame containing only data for the specified lipid class.
        """
        return df[df['ClassKey'] == lipid_class]

    @staticmethod
    def _compute_mean_variance_auc(df, samples):
        """
        Computes the mean and variance of the area under the curve (AUC) for each lipid across given samples.
        """
        mean_cols = [f'MeanArea[{sample}]' for sample in samples if f'MeanArea[{sample}]' in df.columns]
        if mean_cols:
            df['mean_AUC'] = df[mean_cols].mean(axis=1)
            df['var_AUC'] = df[mean_cols].var(axis=1)
        else:
            df['mean_AUC'] = 0
            df['var_AUC'] = 0

    @staticmethod
    def _calculate_fatty_acid_auc_variance(df):
        """
        Calculates the AUC and variance for each type of fatty acid (SFA, MUFA, PUFA) in the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame with lipidomics data, which should include 'mean_AUC' and 'var_AUC' values.
        Effects:
            Modifies the DataFrame to include AUC and variance for SFA, MUFA, and PUFA.
        """
        df['fa_ratios'] = df['LipidMolec'].apply(SaturationPlot._calculate_fa_ratios)
        for fatty_acid in ['SFA', 'MUFA', 'PUFA']:
            idx = ['SFA', 'MUFA', 'PUFA'].index(fatty_acid)
            df[f'{fatty_acid}_AUC'] = df.apply(lambda x: x['mean_AUC'] * x['fa_ratios'][idx], axis=1)
            df[f'{fatty_acid}_var'] = df.apply(lambda x: x['var_AUC'] * x['fa_ratios'][idx] ** 2, axis=1)

    @staticmethod
    def _aggregate_fatty_acid_values(df):
        """
        Aggregates the AUC values for SFA, MUFA, and PUFA across all entries in the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing AUC values for fatty acids.
        Returns:
            tuple: Total AUC values for SFA, MUFA, PUFA, and their variances.
        """
        totals = []
        var_totals = []
        
        for fatty_acid in ['SFA', 'MUFA', 'PUFA']:
            auc_col = f'{fatty_acid}_AUC'
            var_col = f'{fatty_acid}_var'
            
            if auc_col not in df.columns:
                totals.append(0)
            else:
                total = df[auc_col].sum()
                totals.append(total)
            
            if var_col not in df.columns:
                var_totals.append(0)
            else:
                var_total = df[var_col].sum()
                var_totals.append(var_total)
         
        return (*totals, *var_totals)

    @staticmethod
    def create_plots(df, experiment, selected_conditions):
        """
        Generates saturation plots for all lipid classes found in the DataFrame, for the given conditions.
        """
        plots = {}
        for lipid_class in df['ClassKey'].unique():
            plot_data = SaturationPlot._prepare_plot_data(df, selected_conditions, lipid_class, experiment)
            if not plot_data.empty:
                statistical_results = SaturationPlot._perform_statistical_tests(plot_data, lipid_class)
                main_plot = SaturationPlot._create_auc_plot(plot_data, lipid_class, statistical_results)
                percentage_df = SaturationPlot._calculate_percentage_df(plot_data)
                percentage_plot = SaturationPlot._create_percentage_plot(percentage_df, lipid_class)
                plots[lipid_class] = (main_plot, percentage_plot, plot_data)
        return plots

    @staticmethod
    def _prepare_plot_data(df, selected_conditions, lipid_class, experiment):
        """
        Prepares plot data for a given lipid class across selected conditions.
        """
        plot_data = []
        for condition in selected_conditions:
            samples = experiment.individual_samples_list[experiment.conditions_list.index(condition)]
            if samples:  # Only process if there are samples for this condition
                sfa_values, mufa_values, pufa_values, sfa_mean, mufa_mean, pufa_mean = SaturationPlot._calculate_sfa_mufa_pufa(df, condition, samples, lipid_class)
                if any(sfa_values + mufa_values + pufa_values):  # Only add data if there are non-zero values
                    plot_data.append({
                        'Condition': condition,
                        'SFA_AUC': sfa_mean,
                        'MUFA_AUC': mufa_mean,
                        'PUFA_AUC': pufa_mean,
                        'SFA_values': sfa_values,
                        'MUFA_values': mufa_values,
                        'PUFA_values': pufa_values
                    })
        return pd.DataFrame(plot_data)

    @staticmethod
    def _display_saturation_plot(plot_data, lipid_class, experiment):
        """
        Generates and displays saturation plots for a given lipid class.
        Args:
            plot_data (pd.DataFrame): The DataFrame with data specific to a lipid class.
            lipid_class (str): The lipid class for which plots are being generated.
            experiment: The experiment object containing necessary information.
        Effects:
            Creates and displays both the main AUC plot and the percentage distribution plot, along with download buttons for each.
        """
        main_df, percentage_df = SaturationPlot._prepare_df_for_plots(plot_data)
        main_plot = SaturationPlot._create_auc_plot(main_df, lipid_class, experiment)
        percentage_plot = SaturationPlot._create_percentage_plot(percentage_df, lipid_class)

    @staticmethod
    def _prepare_df_for_plots(plot_data):
        """
        Prepares data for both main AUC and percentage distribution plots.
        Args:
            plot_data (pd.DataFrame): The DataFrame with plot data.
        Returns:
            tuple: Two DataFrames, one for the main AUC plot and one for the percentage distribution plot.
        """
        # Prepare data for main and percentage plot
        main_df = plot_data.copy()
        main_df = SaturationPlot._calculate_std_dev(main_df)
        percentage_df = SaturationPlot._calculate_percentage_df(main_df)
        return main_df, percentage_df

    @staticmethod
    def _calculate_std_dev(df):
        """
        Calculates standard deviations for AUC values in the DataFrame.
        """
        for fatty_acid in ['SFA', 'MUFA', 'PUFA']:
            df[f'{fatty_acid}_stdv'] = df[f'{fatty_acid}_values'].apply(np.std, ddof=1)
        return df

    @staticmethod
    @st.cache_data(ttl=3600)
    def _calculate_percentage_df(main_df):
        """
        Calculates the percentage distribution of SFA, MUFA, and PUFA in the dataset.
        Args:
            main_df (pd.DataFrame): The DataFrame containing AUC values for fatty acids.
        Returns:
            pd.DataFrame: A DataFrame with percentage values of each fatty acid type.
        """
        total_auc = main_df[['SFA_AUC', 'MUFA_AUC', 'PUFA_AUC']].sum(axis=1)
        percentage_df = 100 * main_df[['SFA_AUC', 'MUFA_AUC', 'PUFA_AUC']].div(total_auc, axis=0)
        percentage_df['Condition'] = main_df['Condition']
        return percentage_df

    @staticmethod
    def _create_auc_plot(df, lipid_class, statistical_results):
        """
        Creates an AUC plot for a given lipid class using Plotly with non-overlapping bars and statistical indicators.
        """
        if df.empty:
            return None
        
        df = SaturationPlot._calculate_std_dev(df)
        
        fig = go.Figure()
        
        colors = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
        
        bar_width = 0.25
        offsets = [-bar_width, 0, bar_width]
        
        for idx, fa in enumerate(['SFA', 'MUFA', 'PUFA']):
            fig.add_trace(go.Bar(
                x=[c + offsets[idx] for c in range(len(df['Condition']))],
                y=df[f'{fa}_AUC'],
                name=fa,
                marker_color=colors[fa],
                error_y=dict(
                    type='data',
                    array=df[f'{fa}_stdv'],
                    visible=True
                ),
                width=bar_width,
                offset=0
            ))
    
        fig.update_layout(
            title=dict(text=f"Concentration Profile of Fatty Acids in {lipid_class}", font=dict(size=24, color='black')),
            xaxis_title=dict(text="Condition", font=dict(size=18, color='black')),
            yaxis_title=dict(text="AUC", font=dict(size=18, color='black')),
            xaxis=dict(
                tickfont=dict(size=14, color='black'),
                tickmode='array',
                tickvals=list(range(len(df['Condition']))),
                ticktext=df['Condition']
            ),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            barmode='group',
            bargap=0.15,
            bargroupgap=0,
            height=500,
            margin=dict(t=50, r=50, b=50, l=50)
        )
    
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
        SaturationPlot._add_statistical_annotations(fig, df, statistical_results, lipid_class)
    
        return fig
        
        @staticmethod
        def _add_error_bounds(df):
            """
            Adds error bounds to the DataFrame for AUC values.
            Args:
                df (pd.DataFrame): The DataFrame with AUC and standard deviation data.
            Effects:
                Modifies the DataFrame to include upper and lower bounds for error bars.
            """
            for fa in ['SFA', 'MUFA', 'PUFA']:
                df[f'{fa}_upper'] = df[f'{fa}_AUC'] + df[f'{fa}_stdv']
                df[f'{fa}_lower'] = df[f'{fa}_AUC'] - df[f'{fa}_stdv']
            return df
    
    @staticmethod
    def _get_max_y_value(df):
        """
        Calculates the maximum y-value for the AUC plot.
        Args:
            df (pd.DataFrame): The DataFrame with upper bound values.
        Returns:
            float: A value to be used as the maximum y-value in the plot.
        """
        max_upper_bounds = [df[f'{fa}_upper'].max() for fa in ['SFA', 'MUFA', 'PUFA']]
        return max(max_upper_bounds) * 1.1

    @staticmethod
    def _create_percentage_plot(df, lipid_class):
        """
        Creates a percentage distribution plot for a given lipid class using Plotly.
        """
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
            title=dict(text=f"Percentage Distribution of Fatty Acids in {lipid_class}", font=dict(size=24, color='black')),
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
    def _move_legend_outside_plot(plot):
        """
        Moves the legend of the percentage plot to an external position.
        Args:
            plot: The Bokeh plot object whose legend needs repositioning.
        Effects:
            Changes the layout of the plot to position the legend to the right side, outside the plot area.
        """
        plot.add_layout(plot.legend[0], 'right')
                    
    @staticmethod
    def _perform_statistical_tests(plot_data, lipid_class):
        """
        Performs statistical tests on the plot data and displays debugging information using Streamlit.
        Now includes the lipid class in the debug output.
        """
        results = {}
        for fa in ['SFA', 'MUFA', 'PUFA']:
            st.write(f"### Debugging statistical tests for {fa} in {lipid_class}:")
            if f'{fa}_values' in plot_data.columns and len(plot_data['Condition']) >= 2:
                if len(plot_data['Condition']) == 2:
                    results[fa] = SaturationPlot._perform_t_test(plot_data, fa, lipid_class)
                    st.write(f"T-test result for {lipid_class}: p-value = {results[fa]['p_value']:.4f}")
                    if results[fa]['p_value'] < 0.05:
                        conditions = plot_data['Condition'].tolist()
                        st.write(f"Significant difference in {lipid_class} between {conditions[0]} and {conditions[1]}")
                elif len(plot_data['Condition']) > 2:
                    results[fa] = SaturationPlot._perform_anova(plot_data, fa, lipid_class)
                    st.write(f"ANOVA result for {lipid_class}: p-value = {results[fa]['p_value']:.4f}")
                    if results[fa]['p_value'] < 0.05 and results[fa]['posthoc'] is not None:
                        st.write(f"Significant pairwise comparisons for {lipid_class}:")
                        posthoc_results = results[fa]['posthoc']
                        for row in posthoc_results.summary().data[1:]:
                            group1, group2, _, p_value = row[:4]
                            if float(p_value) < 0.05:
                                st.write(f"  {group1} vs {group2}: p-value = {p_value:.4f}")
            else:
                results[fa] = {'test': 'insufficient_data'}
                st.write(f"Insufficient data for statistical testing of {fa} in {lipid_class}")
        return results

    @staticmethod
    def _perform_t_test(plot_data, fa):
        """
        Performs a t-test for a given fatty acid type between two conditions.
        """
        conditions = plot_data['Condition'].unique()
        group1 = plot_data[plot_data['Condition'] == conditions[0]][f'{fa}_values'].iloc[0]
        group2 = plot_data[plot_data['Condition'] == conditions[1]][f'{fa}_values'].iloc[0]
        
        t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        st.write(f"T-test details: t-statistic = {t_statistic:.4f}, p-value = {p_value:.4f}")
        return {'test': 't-test', 't_statistic': t_statistic, 'p_value': p_value}

    @staticmethod
    def _perform_anova(plot_data, fa, lipid_class):
        """
        Performs ANOVA for a given fatty acid type across multiple conditions.
        Now includes the lipid class in the debug output.
        """
        groups = [group[f'{fa}_values'].iloc[0] for _, group in plot_data.groupby('Condition')]
        f_statistic, p_value = stats.f_oneway(*groups)
        st.write(f"ANOVA details for {lipid_class}: f-statistic = {f_statistic:.4f}, p-value = {p_value:.4f}")
        
        posthoc = None
        if p_value < 0.05:
            posthoc = SaturationPlot._perform_tukey_test(plot_data, fa, lipid_class)
        
        return {'test': 'ANOVA', 'f_statistic': f_statistic, 'p_value': p_value, 'posthoc': posthoc}

    @staticmethod
    def _perform_t_test(plot_data, fa, lipid_class):
        """
        Performs a t-test for a given fatty acid type between two conditions.
        Now includes the lipid class in the debug output.
        """
        conditions = plot_data['Condition'].unique()
        group1 = plot_data[plot_data['Condition'] == conditions[0]][f'{fa}_values'].iloc[0]
        group2 = plot_data[plot_data['Condition'] == conditions[1]][f'{fa}_values'].iloc[0]
        
        t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        st.write(f"T-test details for {lipid_class}: t-statistic = {t_statistic:.4f}, p-value = {p_value:.4f}")
        return {'test': 't-test', 't_statistic': t_statistic, 'p_value': p_value}
    
    @staticmethod
    def _perform_tukey_test(plot_data, fa, lipid_class):
        """
        Performs Tukey's HSD test for pairwise comparisons.
        Now includes the lipid class in the debug output.
        """
        values = [val for group in plot_data[f'{fa}_values'] for val in group]
        conditions = [cond for cond, group in zip(plot_data['Condition'], plot_data[f'{fa}_values']) for _ in range(len(group))]
        
        tukey_result = pairwise_tukeyhsd(values, conditions)
        st.write(f"Tukey's HSD test results for {lipid_class}:")
        st.write(tukey_result)
        return tukey_result
                    
    @staticmethod
    def _add_statistical_annotations(fig, df, statistical_results, lipid_class):
        """
        Adds statistical significance annotations to the plot with improved y-axis range adjustment
        and better placement of significance indicators.
        """
        # Calculate the maximum and minimum y-values including error bars
        y_max = max(
            df[f'{fa}_AUC'].max() + df[f'{fa}_stdv'].max() 
            for fa in ['SFA', 'MUFA', 'PUFA'] 
            if f'{fa}_AUC' in df.columns and f'{fa}_stdv' in df.columns
        )
        y_min = min(
            df[f'{fa}_AUC'].min() - df[f'{fa}_stdv'].max()
            for fa in ['SFA', 'MUFA', 'PUFA'] 
            if f'{fa}_AUC' in df.columns and f'{fa}_stdv' in df.columns
        )
        
        y_range = y_max - y_min
        
        # Calculate the number of significant comparisons
        num_significant = sum(
            len([p for p in result.get('posthoc', []) if p < 0.05])
            for result in statistical_results.values()
            if isinstance(result, dict) and 'p_value' in result and result['p_value'] < 0.05
        )
        
        # Adjust y_increment based on the number of significant comparisons
        y_increment = y_range * 0.15 * max(num_significant, 1)
        
        # Add annotations
        annotation_count = 0
        for fa in ['SFA', 'MUFA', 'PUFA']:
            if fa in statistical_results:
                result = statistical_results[fa]
                if result['test'] == 't-test':
                    annotation_count = SaturationPlot._add_t_test_annotation(fig, df, fa, result, y_increment, y_max, annotation_count)
                elif result['test'] == 'ANOVA':
                    annotation_count = SaturationPlot._add_anova_annotation(fig, df, fa, result, y_increment, y_max, annotation_count)
    
        # Adjust y-axis range to accommodate all annotations and ensure full visibility
        y_axis_min = max(0, y_min - y_range * 0.05)  # 5% padding below, but not less than 0
        y_axis_max = y_max + y_increment * (annotation_count + 1)  # Extra space for annotations
        
        # Ensure y_axis_max is at least 30% higher than y_max
        y_axis_max = max(y_axis_max, y_max * 1.3)
        
        fig.update_layout(yaxis_range=[y_axis_min, y_axis_max])
    
        # Add debug information
        st.write(f"Lipid Class: {lipid_class}")
        st.write(f"Y-axis range: {y_axis_min:.2f} to {y_axis_max:.2f}")
        st.write(f"Number of significant comparisons: {num_significant}")
        st.write(f"Number of annotations added: {annotation_count}")
        
    @staticmethod
    def _add_t_test_annotation(fig, df, fa, result, y_increment, y_max, annotation_count):
        """
        Adds t-test result annotation to the plot for a specific fatty acid type.
        """
        colors = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
        bar_width = 0.25
        offsets = {'SFA': -bar_width, 'MUFA': 0, 'PUFA': bar_width}
        
        x0 = 0 + offsets[fa]
        x1 = 1 + offsets[fa]
        y = y_max + y_increment * (annotation_count + 1)
        p_value = result['p_value']
        
        if p_value < 0.05:  # Only add annotation if the result is significant
            fig.add_shape(
                type="line",
                x0=x0, y0=y, x1=x1, y1=y,
                line=dict(color=colors[fa], width=2)
            )
            
            fig.add_annotation(
                x=(x0+x1)/2, y=y+y_increment*0.1,
                text=SaturationPlot._get_significance_symbol(p_value),
                showarrow=False,
                font=dict(size=24, color=colors[fa])
            )
            
            # Debug information
            st.write(f"Added t-test annotation for {fa} at y={y:.2f}")
            annotation_count += 1
    
        return annotation_count
    
    @staticmethod
    def _add_anova_annotation(fig, df, fa, result, y_increment, y_max, annotation_count):
        """
        Adds ANOVA result and post-hoc test annotations to the plot.
        """
        colors = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
        bar_width = 0.25
        offsets = {'SFA': -bar_width, 'MUFA': 0, 'PUFA': bar_width}
    
        if result['p_value'] < 0.05 and result['posthoc'] is not None:
            posthoc_results = result['posthoc']
            for idx, row in enumerate(posthoc_results.summary().data[1:]):
                group1, group2, _, p_value = row[:4]
                x0 = df[df['Condition'] == group1].index[0] + offsets[fa]
                x1 = df[df['Condition'] == group2].index[0] + offsets[fa]
                y = y_max + y_increment * (annotation_count + 1)
                
                if float(p_value) < 0.05:  # Only add annotation if the pairwise comparison is significant
                    fig.add_shape(type="line", x0=x0, y0=y, x1=x1, y1=y, line=dict(color=colors[fa], width=2))
                    fig.add_annotation(x=(x0+x1)/2, y=y+y_increment*0.1, 
                                       text=SaturationPlot._get_significance_symbol(float(p_value)), 
                                       showarrow=False,
                                       font=dict(size=24, color=colors[fa]))
                    
                    # Debug information
                    st.write(f"Added ANOVA annotation for {fa} between {group1} and {group2} at y={y:.2f}")
                    annotation_count += 1
    
        return annotation_count
    
    @staticmethod
    def _get_significance_symbol(p_value):
        """
        Returns a symbol representing the level of significance.
        """
        if p_value < 0.001:
            return "★★★"
        elif p_value < 0.01:
            return "★★"
        elif p_value < 0.05:
            return "★"
        else:
            return ""