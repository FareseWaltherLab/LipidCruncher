import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import itertools

class AbundancePieChart:
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_total_abundance(df, full_samples_list):
        """
        Calculates the total abundance of each lipid class across all available samples.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            full_samples_list (list): List of all sample names in the experiment.

        Returns:
            pd.DataFrame: Aggregated DataFrame with total abundance per lipid class.
        """
        available_samples = [sample for sample in full_samples_list if f"MeanArea[{sample}]" in df.columns]
        if not available_samples:
            st.warning("No sample columns found in the dataset.")
            return pd.DataFrame()
        
        grouped_df = df.groupby('ClassKey')[[f"MeanArea[{sample}]" for sample in available_samples]].sum()
        return grouped_df

    @staticmethod
    def get_all_classes(df, full_samples_list):
        """
        Retrieves a list of all lipid classes present in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            full_samples_list (list): List of all sample names in the experiment.

        Returns:
            list: List of lipid classes.
        """
        total_abundance_df = AbundancePieChart.calculate_total_abundance(df, full_samples_list)
        if total_abundance_df.empty:
            return []
        return list(total_abundance_df.index)

    @staticmethod
    @st.cache_data(ttl=3600)
    def filter_df_for_selected_classes(df, full_samples_list, selected_classes):
        """
        Filters the DataFrame to include only the selected lipid classes.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            full_samples_list (list): List of all sample names in the experiment.
            selected_classes (list): List of selected lipid classes.

        Returns:
            pd.DataFrame: Filtered DataFrame with selected lipid classes.
        """
        total_abundance_df = AbundancePieChart.calculate_total_abundance(df, full_samples_list)
        return total_abundance_df[total_abundance_df.index.isin(selected_classes)]

    @staticmethod
    @st.cache_data(ttl=3600)
    def create_pie_chart(df, full_samples_list, condition, samples, color_mapping):
        """
        Creates a pie chart for the total abundance of lipid classes under a specific condition.
    
        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            full_samples_list (list): List of all sample names in the experiment.
            condition (str): The condition to create the pie chart for.
            samples (list): List of sample names under the given condition.
            color_mapping (dict): Dictionary mapping lipid class labels to colors.
    
        Returns:
            tuple: A tuple containing the Plotly figure object and the DataFrame used.
        """
        available_samples = [sample for sample in samples if f"MeanArea[{sample}]" in df.columns]
        if not available_samples:
            st.warning(f"No sample columns found for condition: {condition}")
            return None, df
        
        condition_abundance = df[[f"MeanArea[{sample}]" for sample in available_samples]].sum(axis=1)
        sorted_sizes, sorted_labels = AbundancePieChart._sort_pie_chart_data(condition_abundance, df.index)
    
        custom_labels = [f'{label} - {pct:.1f}%' for label, pct in zip(sorted_labels, 100 * sorted_sizes / sorted_sizes.sum())]
        fig = AbundancePieChart._configure_pie_chart(sorted_sizes, custom_labels, condition, sorted_labels, color_mapping)
    
        return fig, df

    @staticmethod
    def _sort_pie_chart_data(sizes, labels):
        """
        Sorts pie chart data by size in descending order.

        Args:
            sizes (np.ndarray): Array of sizes (abundances) for the pie chart.
            labels (np.ndarray): Array of labels (lipid classes) for the pie chart.

        Returns:
            tuple: Sorted sizes and labels arrays.
        """
        sorted_indices = np.argsort(sizes)[::-1]
        return sizes[sorted_indices], labels[sorted_indices]

    @staticmethod
    def _configure_pie_chart(sizes, labels, condition, sorted_labels, color_mapping):
        """
        Configures the visual aspects of the pie chart.
    
        Args:
            sizes (np.ndarray): Array of sizes for the pie chart segments.
            labels (np.ndarray): Array of labels for the pie chart segments.
            condition (str): The condition the pie chart represents.
            sorted_labels (list): Sorted list of labels.
            color_mapping (dict): Dictionary mapping lipid class labels to colors.
    
        Returns:
            plotly.graph_objects.Figure: Configured Plotly figure object for the pie chart.
        """
        colors = [color_mapping[label] for label in sorted_labels]
    
        fig = px.pie(values=sizes, names=labels, title=f'Total Abundance Pie Chart - {condition}',
                     color_discrete_sequence=colors)
        hovertemplate = '%{label}<extra>%{percent:.1%}</extra>'
        fig.update_traces(hovertemplate=hovertemplate, textinfo='none')
        fig.update_layout(legend_title="Lipid Classes", margin=dict(l=10, r=100, t=40, b=10), width=450, height=300)
        return fig
    
    @staticmethod
    def _generate_color_mapping(labels):
        """
        Generate a color mapping for different lipid classes to ensure consistent colors across conditions.
        
        Args:
            labels (list): List of lipid class labels.
        
        Returns:
            dict: Dictionary mapping lipid class labels to colors.
        """
        color_palette = px.colors.qualitative.Plotly
        color_cycle = itertools.cycle(color_palette)
        return {label: next(color_cycle) for label in labels}