import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.colors as pc

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
        available_samples = [sample for sample in full_samples_list if f"concentration[{sample}]" in df.columns]
        if not available_samples:
            st.warning("No sample columns found in the dataset.")
            return pd.DataFrame()
        
        grouped_df = df.groupby('ClassKey')[[f"concentration[{sample}]" for sample in available_samples]].sum()
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
        available_samples = [sample for sample in samples if f"concentration[{sample}]" in df.columns]
        if not available_samples:
            st.warning(f"No sample columns found for condition: {condition}")
            return None, df
        
        condition_abundance = df[[f"concentration[{sample}]" for sample in available_samples]].sum(axis=1)
        sorted_sizes, sorted_labels = AbundancePieChart._sort_pie_chart_data(condition_abundance, df.index)
    
        # Calculate percentages with meaningful decimal precision
        percentages = 100 * sorted_sizes / sorted_sizes.sum()
        custom_labels = [f'{label} - {AbundancePieChart._format_percentage(pct)}%' 
                        for label, pct in zip(sorted_labels, percentages)]
        
        fig = AbundancePieChart._configure_pie_chart(sorted_sizes, custom_labels, condition, sorted_labels, color_mapping)
    
        return fig, df

    @staticmethod
    def _format_percentage(percentage):
        """
        Format percentage to show the first meaningful decimal digit.
        
        Args:
            percentage (float): The percentage value to format.
            
        Returns:
            str: Formatted percentage string.
        """
        if percentage == 0:
            return "0.0"
        
        if percentage >= 10:
            # For percentages >= 10%, show 1 decimal place
            return f"{percentage:.1f}"
        elif percentage >= 1:
            # For percentages >= 1%, show 1 decimal place
            return f"{percentage:.1f}"
        elif percentage >= 0.1:
            # For percentages >= 0.1%, show 2 decimal places
            return f"{percentage:.2f}"
        elif percentage >= 0.01:
            # For percentages >= 0.01%, show 3 decimal places
            return f"{percentage:.3f}"
        elif percentage >= 0.001:
            # For percentages >= 0.001%, show 4 decimal places
            return f"{percentage:.4f}"
        else:
            # For very small percentages, use scientific notation
            return f"{percentage:.2e}"

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
        
        # Update hover template to use meaningful decimal precision
        percentages = 100 * sizes / sizes.sum()
        hover_labels = [f'{label}<br>{AbundancePieChart._format_percentage(pct)}%' 
                       for label, pct in zip(sorted_labels, percentages)]
        
        fig.update_traces(
            hovertemplate='%{label}<extra></extra>',
            textinfo='none'
        )
        fig.update_layout(legend_title="Lipid Classes", margin=dict(l=10, r=100, t=40, b=10), width=450, height=300)
        return fig
    
    @staticmethod
    def _generate_color_mapping(labels):
        """
        Generate a unique color mapping for different lipid classes to ensure each class gets a distinct color.
        
        Args:
            labels (list): List of lipid class labels.
        
        Returns:
            dict: Dictionary mapping lipid class labels to unique colors.
        """
        # Combine multiple color palettes to ensure we have enough unique colors
        all_colors = (
            pc.qualitative.Plotly +
            pc.qualitative.Set1 + 
            pc.qualitative.Set2 + 
            pc.qualitative.Set3 +
            pc.qualitative.Pastel1 +
            pc.qualitative.Pastel2 +
            pc.qualitative.Dark2 +
            pc.qualitative.Alphabet
        )
        
        # Remove duplicates while preserving order
        unique_colors = []
        seen = set()
        for color in all_colors:
            if color not in seen:
                unique_colors.append(color)
                seen.add(color)
        
        # If we still don't have enough colors, generate additional ones
        num_labels = len(labels)
        if num_labels > len(unique_colors):
            # Generate additional colors using HSV color space
            import colorsys
            additional_colors = []
            for i in range(num_labels - len(unique_colors)):
                hue = (i * 0.618033988749895) % 1  # Golden ratio for better distribution
                saturation = 0.7 + (i % 3) * 0.1  # Vary saturation
                value = 0.8 + (i % 2) * 0.1  # Vary brightness
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
                additional_colors.append(hex_color)
            unique_colors.extend(additional_colors)
        
        # Create the mapping, ensuring each label gets a unique color
        color_mapping = {}
        for i, label in enumerate(labels):
            color_mapping[label] = unique_colors[i % len(unique_colors)]
        
        return color_mapping