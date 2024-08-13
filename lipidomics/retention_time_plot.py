import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import colorsys
import itertools
import streamlit as st

class RetentionTime:
    """
    A class dedicated to creating and handling retention time plots for lipidomic data.
    """

    @staticmethod
    @st.cache_data(ttl=3600)
    def prep_single_plot_inputs(df, lipid_class):
        """
        Prepare the input data for a single retention time plot.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            lipid_class (str): The lipid class for which the plot will be generated.

        Returns:
            tuple: Arrays of base retention time, calculated mass, and lipid molecule names.
        """
        class_df = df[df['ClassKey'] == lipid_class]
        return class_df['BaseRt'].values, class_df['CalcMass'].values, class_df['LipidMolec'].values

    @staticmethod
    def render_single_plot(retention_df, lipid_class):
        """
        Create a single retention time plot for a given lipid class using Plotly.

        Args:
            retention_df (pd.DataFrame): DataFrame with data for the plot.
            lipid_class (str): The lipid class for which the plot will be generated.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure object.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=retention_df["Mass"],
            y=retention_df["Retention"],
            mode='markers',
            marker=dict(size=6),
            text=retention_df["Species"],
            hovertemplate='<b>Mass</b>: %{x:.4f}<br>' +
                          '<b>Retention time</b>: %{y:.2f}<br>' +
                          '<b>Species</b>: %{text}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=lipid_class, font=dict(size=24, color='black')),
            xaxis_title=dict(text='Calculated Mass', font=dict(size=18, color='black')),
            yaxis_title=dict(text='Retention Time (mins)', font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=50, r=50, b=50, l=50)
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig

    @staticmethod
    @st.cache_data(ttl=3600)
    def prep_multi_plot_input(df, selected_classes_list, unique_color_list):
        """
        Prepare the input data for a multi-class retention time comparison plot.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            selected_classes_list (list): List of lipid classes to be included in the plot.
            unique_color_list (list): List of colors assigned to each lipid class.

        Returns:
            pd.DataFrame: DataFrame with prepared data for plotting.
        """
        plot_data = []
        for lipid_class in selected_classes_list:
            current_class_df = df[df['ClassKey'] == lipid_class]
            plot_data.append(pd.DataFrame({
                "Mass": current_class_df['CalcMass'],
                "Retention": current_class_df['BaseRt'],
                "LipidMolec": current_class_df['LipidMolec'],
                "Class": lipid_class,
                "Color": unique_color_list[selected_classes_list.index(lipid_class)]
            }))
        return pd.concat(plot_data, ignore_index=True)

    def get_distinct_colors(n):
        """
        Generate n visually distinct colors.
        
        Args:
            n (int): Number of colors to generate.
        
        Returns:
            list: List of RGB color tuples.
        """
        hue_partition = 1.0 / (n + 1)
        return [colorsys.hsv_to_rgb(hue_partition * value, 1.0, 1.0) for value in range(n)]
    
    @staticmethod
    def render_multi_plot(retention_df):
        """
        Create a multi-class retention time comparison plot using Plotly.
    
        Args:
            retention_df (pd.DataFrame): DataFrame with data for the plot.
    
        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure object for comparing multiple classes.
        """
        fig = go.Figure()
    
        # Get the number of unique classes
        num_classes = len(retention_df['Class'].unique())
    
        # Generate distinct colors for each class
        colors = RetentionTime.get_distinct_colors(num_classes)  # Changed this line
        color_palette = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
    
        for i, lipid_class in enumerate(retention_df['Class'].unique()):
            class_df = retention_df[retention_df['Class'] == lipid_class]
            fig.add_trace(go.Scatter(
                x=class_df["Mass"],
                y=class_df["Retention"],
                mode='markers',
                marker=dict(size=6, color=color_palette[i]),
                name=lipid_class,
                text=class_df["LipidMolec"],
                hovertemplate='<b>Mass</b>: %{x:.4f}<br>' +
                              '<b>Retention time</b>: %{y:.2f}<br>' +
                              '<b>Lipid Molecule</b>: %{text}<extra></extra>'
            ))
    
        fig.update_layout(
            title=dict(text='Retention Time vs. Mass - Comparison Mode', font=dict(size=24, color='black')),
            xaxis_title=dict(text='Calculated Mass', font=dict(size=18, color='black')),
            yaxis_title=dict(text='Retention Time (mins)', font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            margin=dict(t=50, r=50, b=50, l=50)
        )
    
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
        return fig

    @staticmethod
    def plot_single_retention(df):
        """
        Generate individual retention time plots for each lipid class in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.

        Returns:
            list of tuples: Each tuple contains a plot, its DataFrame, filename, and svg string.
        """
        plots = []
        for lipid_class in df['ClassKey'].value_counts().index:
            retention, mass, species = RetentionTime.prep_single_plot_inputs(df, lipid_class)
            retention_df = pd.DataFrame({"Mass": mass, "Retention": retention, "Species": species})
            plot = RetentionTime.render_single_plot(retention_df, lipid_class)
            plots.append((plot, retention_df))
        return plots

    @staticmethod
    def plot_multi_retention(df, selected_classes_list):
        """
        Generate a retention time plot comparing multiple lipid classes.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            selected_classes_list (list): List of lipid classes to include in the plot.

        Returns:
            tuple: A plot (plotly.graph_objs._figure.Figure) and its DataFrame.
        """
        all_lipid_classes_lst = df['ClassKey'].value_counts().index.tolist()
        unique_color_lst = RetentionTime.get_unique_colors(len(all_lipid_classes_lst))
    
        retention_df = RetentionTime.prep_multi_plot_input(df, selected_classes_list, unique_color_lst)
        plot = RetentionTime.render_multi_plot(retention_df)
        return plot, retention_df
    
    @staticmethod
    def get_unique_colors(n_classes):
        """
        Generate a list of unique color hex codes for a given number of classes using Plotly Express color sequences.
        If the number of classes exceeds the palette size, colors will cycle through the palette.

        Args:
            n_classes (int): Number of unique colors required.

        Returns:
            list: List of color hex codes from Plotly Express color sequences.
        """
        colors = itertools.cycle(px.colors.qualitative.Plotly)
        return [next(colors) for _ in range(n_classes)]
