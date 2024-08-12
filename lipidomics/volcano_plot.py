import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
import itertools

class VolcanoPlot:
    """
    A class dedicated to creating volcano plots for lipidomics data analysis.
    This class handles data processing and plot generation for comparing two conditions
    to identify significantly altered lipids.
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
    @st.cache_data
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
        control_cols = ['MeanArea[' + sample + ']' for sample in control_samples]
        experimental_cols = ['MeanArea[' + sample + ']' for sample in experimental_samples]
        selected_cols = control_cols + experimental_cols + ['LipidMolec']
        return df[selected_cols], control_cols, experimental_cols

    @staticmethod
    @st.cache_data
    def _calculate_stats(df, control_cols, experimental_cols):
        """
        Calculate mean values and identify valid rows where neither mean control nor mean experimental values are zero.
        
        Args:
            df: DataFrame containing relevant data for control and experimental conditions.
            control_cols: List of column names for control samples.
            experimental_cols: List of column names for experimental samples.
        
        Returns:
            A tuple containing the DataFrame, mean control values, mean experimental values, and a mask of valid rows.
        """
        mean_control = df[control_cols].mean(axis=1)
        mean_experimental = df[experimental_cols].mean(axis=1)
        valid_rows = (mean_control != 0) & (mean_experimental != 0)
        return df, mean_control, mean_experimental, valid_rows
    
    @staticmethod
    @st.cache_data
    def _format_results(df, mean_control, mean_experimental, valid_rows, control_cols, experimental_cols):
        """
        Format the results of volcano plot data preparation by calculating fold changes, p-values, and logging mean control concentrations.
    
        This method processes the provided DataFrame to compute statistical metrics used for volcano plot generation. 
        It calculates the log2 fold change between experimental and control conditions, performs a t-test to 
        determine p-values, and logs the mean concentration of control samples. Rows where the mean concentration
        in control or experimental conditions is zero are deemed invalid and excluded from these calculations.
    
        Args:
            df (pd.DataFrame): The DataFrame containing lipidomics data filtered to include relevant columns only.
            mean_control (pd.Series): A Series containing mean values of control samples across lipids.
            mean_experimental (pd.Series): A Series containing mean values of experimental samples across lipids.
            valid_rows (pd.Series): A boolean Series indicating rows where neither mean control nor mean experimental values are zero.
            control_cols (list): A list of column names corresponding to control samples.
            experimental_cols (list): A list of column names corresponding to experimental samples.
    
        Returns:
            tuple: 
                - A DataFrame (volcano_df) containing the calculated fold change, p-values, log10 of mean control concentration,
                  and lipid identifiers. Additionally includes -log10 transformed p-values for easier visualization.
                - A DataFrame (removed_lipids_df) listing lipids excluded from the analysis due to having zero concentration in
                  either control or experimental groups, useful for troubleshooting and quality control.
    
        Raises:
            ValueError: If `valid_rows` contains no true values, indicating no valid data points to process.
        """
        # Identify valid rows where neither control nor experimental mean values are zero
        valid_rows = (mean_control != 0) & (mean_experimental != 0)

        # Calculate fold changes and p-values for valid rows
        if valid_rows.any():
            fold_change = np.log2(mean_experimental[valid_rows] / mean_control[valid_rows])
            p_values = stats.ttest_ind(df.loc[valid_rows, experimental_cols], df.loc[valid_rows, control_cols], axis=1).pvalue
            log10_mean_control = np.log10(mean_control[valid_rows] + 1)  # Adding 1 to avoid log10(0)
        else:
            fold_change, p_values, log10_mean_control = pd.Series([]), pd.Series([]), pd.Series([])

        volcano_df = pd.DataFrame({
            'FoldChange': fold_change,
            'pValue': p_values,
            'Log10MeanControl': log10_mean_control,
            'Lipid': df.loc[valid_rows, 'LipidMolec']
        })
        volcano_df['-log10(pValue)'] = -np.log10(volcano_df['pValue'])

        # Determine the reason for exclusion: whether due to zeros in control or experimental
        zero_control = mean_control == 0
        zero_experimental = mean_experimental == 0

        conditions_with_zeros = zero_control & zero_experimental
        condition_labels = np.where(zero_control & ~zero_experimental, 'Control', 
                                    np.where(zero_experimental & ~zero_control, 'Experimental', 'Both'))

        # Create DataFrame for excluded entries, retaining the original index
        removed_lipids_df = df[~valid_rows][['LipidMolec']].reset_index()
        removed_lipids_df['ConditionWithZero'] = condition_labels[~valid_rows]
        removed_lipids_df.columns = ['Lipid Index', 'LipidMolec', 'ConditionWithZero']

        return volcano_df, removed_lipids_df

    @staticmethod
    @st.cache_data
    def _merge_and_filter_df(df, volcano_df, selected_classes):
        """
        Merge the original DataFrame with volcano plot data and filter based on selected classes.
        
        Args:
            df: Original DataFrame with lipidomics data.
            volcano_df: DataFrame containing volcano plot data such as fold change and p-values.
            selected_classes: List of lipid classes to be included in the plot.
        
        Returns:
            A merged and filtered DataFrame based on selected lipid classes.
        """
        volcano_df['Lipid'] = volcano_df.index
        filtered_df = df[df['ClassKey'].isin(selected_classes)]
        filtered_df['Lipid'] = filtered_df.index
        return filtered_df.merge(volcano_df, on='Lipid')

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
    def _create_plot(merged_df, color_mapping, q_value_threshold, hide_non_significant):
        """
        Create a Plotly figure for the volcano plot visualization.
        
        Args:
            merged_df: DataFrame containing data to be plotted.
            color_mapping: Dictionary mapping lipid classes to colors.
            q_value_threshold: The threshold for significance in the plot (-log10 of p-value).
            hide_non_significant: Boolean indicating whether to hide non-significant data points.
        
        Returns:
            A Plotly figure object representing the volcano plot.
        """
        if hide_non_significant:
            significant_df = merged_df[((merged_df['FoldChange'] < -1) | (merged_df['FoldChange'] > 1)) & (merged_df['-log10(pValue)'] >= q_value_threshold)]
        else:
            significant_df = merged_df
        
        fig = go.Figure()

        for class_name, color in color_mapping.items():
            class_df = significant_df[significant_df['ClassKey'] == class_name]
            fig.add_trace(go.Scatter(
                x=class_df['FoldChange'],
                y=class_df['-log10(pValue)'],
                mode='markers',
                name=class_name,
                marker=dict(color=color, size=5),
                text=class_df['LipidMolec'],
                hovertemplate='<b>Lipid:</b> %{text}<br>' +
                              '<b>Fold Change:</b> %{x:.2f}<br>' +
                              '<b>-log10(p-value):</b> %{y:.2f}<extra></extra>'
            ))

        fig.add_shape(type="line", x0=merged_df['FoldChange'].min(), x1=merged_df['FoldChange'].max(),
                      y0=q_value_threshold, y1=q_value_threshold, line=dict(dash="dash", color="black"))
        fig.add_shape(type="line", x0=-1, x1=-1, y0=0, y1=merged_df['-log10(pValue)'].max(), line=dict(dash="dash", color="black"))
        fig.add_shape(type="line", x0=1, x1=1, y0=0, y1=merged_df['-log10(pValue)'].max(), line=dict(dash="dash", color="black"))

        fig.update_layout(
            title=dict(text="Volcano Plot", font=dict(size=24, color='black')),
            xaxis_title=dict(text="Log2(Fold Change)", font=dict(size=18, color='black')),
            yaxis_title=dict(text="-log10(p-value)", font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            height=600,
            margin=dict(t=50, r=50, b=50, l=50)
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig

    @staticmethod
    def _add_scatter_plots(plot, merged_df, color_mapping):
        """
        Add scatter plots to the Bokeh plot for each lipid class.
        
        Args:
            plot: Bokeh plot figure to which scatter plots will be added.
            merged_df: DataFrame containing data to be plotted.
            color_mapping: Dictionary mapping lipid classes to colors.
        """
        for class_name, color in color_mapping.items():
            class_df = merged_df[merged_df['ClassKey'] == class_name]
            class_source = ColumnDataSource(class_df)
            plot.scatter(x='FoldChange', y='-log10(pValue)', color=color, legend_label=class_name, source=class_source)

    @staticmethod
    def _configure_hover_tool(plot, plot_type="volcano"):
        """
        Configure and add a hover tool to the Bokeh plot.
        
        Args:
            plot: Bokeh plot figure to which the hover tool will be added.
            plot_type: Type of plot ("volcano" or "fold_change_vs_control") to customize hover tools accordingly.
        
        Returns:
            The Bokeh plot with the hover tool configured.
        """
        if plot_type == "volcano":
            hover = HoverTool(tooltips=[
                ("Lipid", "@LipidMolec"),
                ("Fold Change", "@FoldChange"),
                ("p-value", "@pValue")
            ])
        elif plot_type == "fold_change_vs_control":
            hover = HoverTool(tooltips=[
                ("Lipid", "@LipidMolec"),
                ("Fold Change", "@FoldChange"),
                ("Mean Control Concentration", "@Log10MeanControl")
            ])
        
        plot.add_tools(hover)
        plot.title.text_font_size = "15pt"
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"


    @staticmethod
    def create_and_display_volcano_plot(experiment, df, control_condition, experimental_condition, selected_classes, q_value_threshold, hide_non_significant):
        """
        Generates a volcano plot for comparing two conditions in lipidomics data to identify significantly altered lipids.
        
        This method orchestrates the creation of a volcano plot by preparing the data, calculating necessary statistics,
        merging and filtering based on selected lipid classes, and then generating a color-coded plot to visually represent
        the data. It is intended to provide insights into which lipids are significantly altered between two experimental
        conditions based on log2 fold changes and statistical significance (p-values).
    
        Args:
            experiment (Experiment): An object containing detailed information about the experiment setup, including conditions and samples.
            df (pd.DataFrame): The DataFrame containing lipidomics data.
            control_condition (str): The name of the control condition.
            experimental_condition (str): The name of the experimental condition to be compared against the control.
            selected_classes (list): A list of lipid classes to include in the plot. Only lipids from these classes will be displayed.
            q_value_threshold (float): The threshold for statistical significance, represented as -log10(p-value), used to draw a threshold line on the plot.
            hide_non_significant (bool): Whether to hide non-significant data points in the plot.
    
        Returns:
            plot (figure): A Bokeh figure object representing the volcano plot.
            merged_df (pd.DataFrame): A DataFrame containing the data used in the plot, including identifiers and computed metrics such as fold changes and p-values.
            removed_lipids_df (pd.DataFrame): A DataFrame listing lipids excluded from the analysis due to zero concentration in either control or experimental groups.
        """
        control_samples, experimental_samples = VolcanoPlot._get_samples_for_conditions(experiment, control_condition, experimental_condition)
        df_processed, control_cols, experimental_cols = VolcanoPlot._prepare_data(df, control_samples, experimental_samples)
        df_processed, mean_control, mean_experimental, valid_rows = VolcanoPlot._calculate_stats(df_processed, control_cols, experimental_cols)
        volcano_df, removed_lipids_df = VolcanoPlot._format_results(df_processed, mean_control, mean_experimental, valid_rows, control_cols, experimental_cols)
        merged_df = VolcanoPlot._merge_and_filter_df(df, volcano_df, selected_classes)
        color_mapping = VolcanoPlot._generate_color_mapping(merged_df)
        plot = VolcanoPlot._create_plot(merged_df, color_mapping, q_value_threshold, hide_non_significant)
        
        return plot, merged_df, removed_lipids_df

    
    @staticmethod
    @st.cache_data
    def get_most_abundant_lipid(df, selected_class):
        """
        Get the most abundant lipid in the selected class.
        
        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            selected_class (str): The selected lipid class.
        
        Returns:
            str: The most abundant lipid in the selected class.
        """
        class_df = df[df['ClassKey'] == selected_class]
        most_abundant_lipid = class_df.set_index('LipidMolec').sum(axis=1).idxmax()
        return most_abundant_lipid

    
    @staticmethod
    def create_concentration_distribution_data(volcano_df, selected_lipids, selected_conditions, experiment):
        """
        Prepares data for concentration distribution plot of selected lipids across conditions.
    
        Args:
            volcano_df (pd.DataFrame): DataFrame containing volcano plot data.
            selected_lipids (list): List of selected lipid molecules for the plot.
            selected_conditions (list): Conditions selected for the plot.
            experiment (Experiment): Object containing experiment setup details.
    
        Returns:
            pd.DataFrame: DataFrame containing concentration data for the selected lipids.
        """
        plot_data = []
        for lipid in selected_lipids:
            for condition in selected_conditions:
                samples = experiment.individual_samples_list[experiment.conditions_list.index(condition)]
                for sample in samples:
                    concentration = volcano_df.loc[volcano_df['LipidMolec'] == lipid, f'MeanArea[{sample}]'].values[0]
                    plot_data.append({'Lipid': lipid, 'Condition': condition, 'Concentration': concentration})
    
        return pd.DataFrame(plot_data)

    @staticmethod
    def _create_concentration_vs_fold_change_plot(merged_df, color_mapping, q_value_threshold, hide_non_significant):
        """
        Create a Plotly figure for Log10(Mean Control Concentration) vs. Log2(Fold Change).
    
        Args:
            merged_df: DataFrame containing merged data for plotting, including calculated metrics.
            color_mapping: Dictionary mapping lipid classes to colors.
            q_value_threshold: The threshold for significance in the plot (-log10 of p-value).
            hide_non_significant: Boolean indicating whether to hide non-significant data points.
    
        Returns:
            Plotly figure object representing the new plot.
        """
        if hide_non_significant:
            significant_df = merged_df[((merged_df['FoldChange'] < -1) | (merged_df['FoldChange'] > 1)) & (merged_df['-log10(pValue)'] >= q_value_threshold)]
        else:
            significant_df = merged_df
        
        fig = go.Figure()

        for class_name, color in color_mapping.items():
            class_df = significant_df[significant_df['ClassKey'] == class_name]
            fig.add_trace(go.Scatter(
                x=class_df['FoldChange'],
                y=class_df['Log10MeanControl'],
                mode='markers',
                name=class_name,
                marker=dict(color=color, size=5),
                text=class_df['LipidMolec'],
                hovertemplate='<b>Lipid:</b> %{text}<br>' +
                              '<b>Fold Change:</b> %{x:.2f}<br>' +
                              '<b>Log10(Mean Control):</b> %{y:.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text="Fold Change vs. Mean Control Concentration", font=dict(size=24, color='black')),
            xaxis_title=dict(text="Log2(Fold Change)", font=dict(size=18, color='black')),
            yaxis_title=dict(text="Log10(Mean Control Concentration)", font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            height=600,
            margin=dict(t=50, r=50, b=50, l=50)
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig, significant_df[['LipidMolec', 'Log10MeanControl', 'FoldChange', 'ClassKey']]

    @staticmethod
    def create_concentration_distribution_plot(plot_df, selected_lipids, selected_conditions):
        """
        Creates a seaborn box plot for the concentration distribution of selected lipids.
    
        Args:
            plot_df (pd.DataFrame): DataFrame containing concentration data for the selected lipids.
            selected_lipids (list): List of selected lipid molecules.
            selected_conditions (list): List of selected conditions.
    
        Returns:
            plt.Figure: Matplotlib figure object with the box plot.
        """
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

