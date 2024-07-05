import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import itertools
from bokeh.palettes import Category20 as palette
import streamlit as st

class PCAAnalysis:
    """
    A class for performing Principal Component Analysis (PCA) on lipidomics data.
    Methods:
    - run_pca: Runs PCA on the provided DataFrame.
    - _generate_color_mapping: Generates a color mapping for different conditions.
    - _create_scatter_plot: Creates a scatter plot based on PCA results.
    - plot_pca: Coordinates the entire process of generating a PCA plot.
    """
    @staticmethod
    @st.cache_data
    def run_pca(_df, full_samples_list):
        """
        Perform PCA on the provided DataFrame.
        Args:
            _df (pd.DataFrame or tuple): DataFrame containing lipidomics data or a tuple where the first element is the DataFrame.
            full_samples_list (list): List of all sample names in the experiment.
        Returns:
            tuple: A tuple of a DataFrame with PCA scores and a list of PCA component names with explained variance.
        """
        if isinstance(_df, tuple):
            df = _df[0]  # Assume the DataFrame is the first element of the tuple
        else:
            df = _df
        mean_area_df = df[['MeanArea[' + sample + ']' for sample in full_samples_list if f'MeanArea[{sample}]' in df.columns]].T
        scaled_data = StandardScaler().fit_transform(mean_area_df)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        return pc_df, [f'PC{i+1} ({var:.0%})' for i, var in enumerate(explained_variance)]

    @staticmethod
    @st.cache_data
    def _generate_color_mapping(conditions):
        """
        Generate a mapping of conditions to unique colors.
        Args:
            conditions (list): A list of condition names.
        Returns:
            dict: A dictionary mapping conditions to colors.
        """
        unique_conditions = sorted(set(conditions))
        colors = itertools.cycle(palette[20])
        return {condition: next(colors) for condition in unique_conditions}

    @staticmethod
    def _create_scatter_plot(df, pc_names, color_mapping):
        """
        Creates a scatter plot based on PCA results and color mapping.
        Args:
            df (pd.DataFrame): DataFrame containing PCA scores and sample information.
            pc_names (list): Names of the principal components.
            color_mapping (dict): A dictionary mapping conditions to colors.
        Returns:
            bokeh.plotting.figure: A Bokeh plot figure.
        """
        plot = figure(title="PCA Plot", x_axis_label=pc_names[0], y_axis_label=pc_names[1])
        for condition, color in color_mapping.items():
            cond_df = df[df['Condition'] == condition]
            cond_source = ColumnDataSource(cond_df)
            plot.scatter(x='PC1', y='PC2', source=cond_source, legend_label=condition, color=color)
        hover = HoverTool(tooltips=[('Sample', '@Sample'), ('PC1', '@PC1'), ('PC2', '@PC2')])
        plot.add_tools(hover)
        plot.legend.click_policy = 'hide'
        plot.title.text_font_size = "15pt"
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
        return plot

    @staticmethod
    def plot_pca(df, full_samples_list, extensive_conditions_list):
        pc_df, pc_names = PCAAnalysis.run_pca(df, full_samples_list)
        pc_df['Sample'] = full_samples_list[:len(pc_df)]  # Ensure lengths match
        pc_df['Condition'] = extensive_conditions_list[:len(pc_df)]  # Ensure lengths match
        color_mapping = PCAAnalysis._generate_color_mapping(pc_df['Condition'])
        plot = PCAAnalysis._create_scatter_plot(pc_df, pc_names, color_mapping)
        return plot, pc_df  # Returning both the plot and the DataFrame