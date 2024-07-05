import pandas as pd
import numpy as np
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Whisker, BasicTickFormatter, FactorRange
from bokeh.transform import dodge, factor_cmap

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
    @st.cache_data
    def _calculate_sfa_mufa_pufa(df, condition, samples, lipid_class):
        """
        Calculates SFA, MUFA, and PUFA values for a specific lipid class under given conditions.
        """
        filtered_df = SaturationPlot._filter_df_for_lipid_class(df, lipid_class)
        available_samples = [sample for sample in samples if f'MeanArea[{sample}]' in filtered_df.columns]
        
        if not available_samples:
            st.warning(f"No data available for {lipid_class} in condition {condition}")
            return (0, 0, 0, 0, 0, 0)  # Return zeros if no data is available
        
        SaturationPlot._compute_mean_variance_auc(filtered_df, available_samples)
        SaturationPlot._calculate_fatty_acid_auc_variance(filtered_df)
        return SaturationPlot._aggregate_fatty_acid_values(filtered_df)

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
    @st.cache_data
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
        totals = [df[f'{fatty_acid}_AUC'].sum() for fatty_acid in ['SFA', 'MUFA', 'PUFA']]
        var_totals = [df[f'{fatty_acid}_var'].sum() for fatty_acid in ['SFA', 'MUFA', 'PUFA']]
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
                main_plot = SaturationPlot._create_auc_plot(plot_data, lipid_class)
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
                values = SaturationPlot._calculate_sfa_mufa_pufa(df, condition, samples, lipid_class)
                if any(values):  # Only add data if there are non-zero values
                    plot_data.append(dict(zip(['Condition', 'SFA_AUC', 'MUFA_AUC', 'PUFA_AUC', 'SFA_var', 'MUFA_var', 'PUFA_var'], [condition] + list(values))))
        return pd.DataFrame(plot_data)

    @staticmethod
    def _display_saturation_plot(plot_data, lipid_class):
        """
        Generates and displays saturation plots for a given lipid class.
        Args:
            plot_data (pd.DataFrame): The DataFrame with data specific to a lipid class.
            lipid_class (str): The lipid class for which plots are being generated.
        Effects:
            Creates and displays both the main AUC plot and the percentage distribution plot, along with download buttons for each.
        """
        main_df, percentage_df = SaturationPlot._prepare_df_for_plots(plot_data)
        main_plot = SaturationPlot._create_auc_plot(main_df, lipid_class)
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
        Args:
            df (pd.DataFrame): The DataFrame with 'var_AUC' values for fatty acids.
        Effects:
            Adds standard deviation columns to the DataFrame for each fatty acid type.
        """
        # Calculate standard deviation
        for fatty_acid in ['SFA', 'MUFA', 'PUFA']:
            df[f'{fatty_acid}_stdv'] = np.sqrt(df[f'{fatty_acid}_var'])
        return df

    @staticmethod
    @st.cache_data
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
    def _create_auc_plot(df, lipid_class):
        """
        Creates an AUC plot for a given lipid class.
        """
        if df.empty:
            return None
        
        df = SaturationPlot._calculate_std_dev(df)
        df = SaturationPlot._add_error_bounds(df)
        max_y_value = SaturationPlot._get_max_y_value(df)
        plot = SaturationPlot._initialize_auc_plot(df, lipid_class, max_y_value)
        SaturationPlot._add_vbars_and_whiskers_to_plot(plot, df)
        SaturationPlot._style_auc_plot(plot)
        SaturationPlot._move_legend_outside_plot(plot)
        return plot
    
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
    def _initialize_auc_plot(df, lipid_class, max_y_value):
        """
        Initializes a Bokeh plot for AUC values.
        Args:
            df (pd.DataFrame): The DataFrame containing the plot data.
            lipid_class (str): The lipid class being plotted.
            max_y_value (float): The maximum y-value for the plot.
        Returns:
            Bokeh plot object: An initialized plot for AUC values.
        """
        source = ColumnDataSource(df)
        return figure(x_range=df['Condition'].tolist(), plot_height=350, 
                      title=f"Concentration Profile of Fatty Acids in {lipid_class}", 
                      toolbar_location=None, tools="", y_range=(0, max_y_value))
    
    @staticmethod
    def _add_vbars_and_whiskers_to_plot(plot, df):
        """
        Adds vertical bars and whiskers to the AUC plot.
        Args:
            plot: The Bokeh plot object to modify.
            df (pd.DataFrame): The DataFrame containing the data for plotting.
        Effects:
            Modifies the provided plot by adding vertical bars and whiskers.
        """
        colors = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
        for idx, fa in enumerate(['SFA', 'MUFA', 'PUFA']):
            plot.vbar(x=dodge('Condition', -0.25 + idx * 0.25, range=plot.x_range), 
                      top=f'{fa}_AUC', width=0.2, source=ColumnDataSource(df), 
                      color=colors[fa], legend_label=fa)
            plot.add_layout(Whisker(source=ColumnDataSource(df), 
                                    base=dodge('Condition', -0.25 + idx * 0.25, range=plot.x_range), 
                                    upper=f'{fa}_upper', lower=f'{fa}_lower', 
                                    level='overlay', line_color='black'))
    
    @staticmethod
    def _style_auc_plot(plot):
        """
        Applies styling to the AUC plot.
        Args:
            plot: The Bokeh plot object to be styled.
        Effects:
            Modifies the plot with specific visual styles, including font sizes and grid line colors.
        """
        plot.xgrid.grid_line_color = None
        plot.legend.location = "top_right"
        plot.legend.orientation = "vertical"
        plot.title.text_font_size = "15pt"
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
    
    @staticmethod
    def _move_legend_outside_plot(plot):
        """
        Moves the legend of the plot to an external position.
        Args:
            plot: The Bokeh plot object whose legend needs repositioning.
        Effects:
            Adjusts the layout of the plot to move the legend to the right side, outside the plot area.
        """
        plot.add_layout(plot.legend[0], 'right')

    @staticmethod   
    def _create_percentage_plot(df, lipid_class):
        """
        Creates a percentage distribution plot for a given lipid class.
        """
        if df.empty:
            return None
        
        source = ColumnDataSource(df)
        plot = SaturationPlot._initialize_percentage_plot(df, lipid_class)
        SaturationPlot._add_stacked_vbars_to_plot(plot, source)
        SaturationPlot._style_percentage_plot(plot)
        SaturationPlot._move_legend_outside_plot(plot)
        return plot
    
    @staticmethod
    def _initialize_percentage_plot(df, lipid_class):
        """
        Initializes a Bokeh plot for percentage distribution values.
        Args:
            df (pd.DataFrame): The DataFrame containing the data for the plot.
            lipid_class (str): The lipid class being visualized.
        Returns:
            Bokeh plot object: An initialized plot for percentage distribution values.
        """
        return figure(x_range=df['Condition'].tolist(), plot_height=350, 
                      title=f"Percentage Distribution of Fatty Acids in {lipid_class}",
                      toolbar_location=None, tools="")
    
    @staticmethod
    def _add_stacked_vbars_to_plot(plot, source):
        """
        Adds stacked vertical bars to the percentage plot.
        Args:
            plot: The Bokeh plot object to be modified.
            source (ColumnDataSource): The data source for the plot.
        Effects:
            Modifies the provided plot by adding stacked vertical bars representing different fatty acids.
        """
        colors = ['#c9d9d3', '#718dbf', '#e84d60']
        plot.vbar_stack(['SFA_AUC', 'MUFA_AUC', 'PUFA_AUC'], x='Condition', 
                        width=0.2, color=colors, source=source, 
                        legend_label=['SFA', 'MUFA', 'PUFA'])
    
    @staticmethod
    def _style_percentage_plot(plot):
        """
        Applies styling to the percentage plot.
        Args:
            plot: The Bokeh plot object to be styled.
        Effects:
            Adjusts the visual style of the plot, including the y-range, grid line color, and font sizes.
        """
        plot.y_range.start = 0
        plot.y_range.end = 100
        plot.x_range.range_padding = 0.1
        plot.xgrid.grid_line_color = None
        plot.title.text_font_size = "15pt"
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
    
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

