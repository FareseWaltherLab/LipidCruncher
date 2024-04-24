import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import streamlit as st

class LipidomicHeatmap:
    
    @staticmethod
    def filter_data(df, selected_conditions, selected_classes, conditions_list, individual_samples_list):
        """
        Filters the lipidomic data based on the selected conditions and lipid classes.
    
        Args:
            df (pd.DataFrame): The original DataFrame containing lipidomics data.
            selected_conditions (list): List of conditions chosen by the user.
            selected_classes (list): List of lipid classes chosen by the user.
            conditions_list (list): List of all conditions in the experiment.
            individual_samples_list (list of lists): Nested list containing sample names for each condition.
    
        Returns:
            tuple: A tuple containing the filtered DataFrame and a list of selected sample names.
        """
        selected_samples = [sample for condition in selected_conditions 
                            for sample in individual_samples_list[conditions_list.index(condition)]]
        abundance_cols = ['MeanArea[' + sample + ']' for sample in selected_samples]
        filtered_df = df[df['ClassKey'].isin(selected_classes)][['LipidMolec', 'ClassKey'] + abundance_cols]
        return filtered_df, selected_samples

    @staticmethod
    @st.cache_data
    def compute_z_scores(filtered_df):
        """
        Calculates Z-scores for each lipid molecule in the filtered DataFrame.
    
        Z-scores are computed for lipid abundances in each row, providing a standardized measure 
        of abundance relative to the mean and standard deviation of the row.
    
        Args:
            filtered_df (pd.DataFrame): Filtered DataFrame with lipid abundances.
    
        Returns:
            pd.DataFrame: DataFrame with Z-scores for each lipid molecule.
        """
        filtered_df.set_index(['LipidMolec', 'ClassKey'], inplace=True)
        abundance_cols = filtered_df.columns
        z_scores_df = filtered_df[abundance_cols].apply(lambda x: (x - x.mean(skipna=True)) / x.std(skipna=True), axis=1)
        return z_scores_df

    @staticmethod
    @st.cache_data
    def perform_clustering(z_scores_df):
        """
        Performs hierarchical clustering on the lipidomic data based on Z-scores.
    
        This method uses Ward's method for hierarchical clustering to organize lipid molecules 
        in a way that groups together molecules with similar abundance patterns.
    
        Args:
            z_scores_df (pd.DataFrame): DataFrame containing Z-scores for lipid molecules.
    
        Returns:
            pd.DataFrame: Clustered DataFrame with lipid molecules reordered according to the clustering result.
        """
        linkage_matrix = linkage(pdist(z_scores_df, 'euclidean'), method='ward')
        dendrogram_order = leaves_list(linkage_matrix)
        clustered_df = z_scores_df.iloc[dendrogram_order]
        return clustered_df

    @staticmethod
    def generate_clustered_heatmap(z_scores_df, selected_samples):
        """
        Generates a clustered heatmap based on the hierarchical clustering of Z-scores.
    
        The heatmap visualizes the abundance patterns of lipid molecules across different samples, 
        with the molecules reordered according to the clustering results to highlight patterns 
        and relationships.
    
        Args:
            z_scores_df (pd.DataFrame): DataFrame containing clustered Z-scores.
            selected_samples (list): List of sample names to be included in the heatmap.
            save_svg_path (str, optional): Path to save the heatmap as an SVG file.
    
        Returns:
            plotly.graph_objects.Figure: Plotly figure object representing the clustered heatmap.
        """
        z_scores_array = z_scores_df.to_numpy()
        fig = px.imshow(
            z_scores_array, 
            labels=dict(color="Z-score"),
            x=selected_samples,
            y=z_scores_df.index.get_level_values('LipidMolec'),
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )

        fig.update_layout(
            title='Lipidomic Heatmap',
            coloraxis_colorbar=dict(title="Z-score"),
            xaxis_title="Samples",
            yaxis_title="Lipid Molecules",
            margin=dict(l=100, r=100, t=50, b=50)  # Increased margins
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickmode='array')

        return fig
    
    @staticmethod
    def generate_regular_heatmap(z_scores_df, selected_samples):
        """
        Creates a regular heatmap (without clustering) from the Z-scores DataFrame.
    
        This heatmap provides a visualization of lipid molecule abundances across various samples, 
        maintaining the original order of molecules in the dataset.
    
        Args:
            z_scores_df (pd.DataFrame): DataFrame containing Z-scores for lipid molecules.
            selected_samples (list): List of sample names for the heatmap's columns.
    
        Returns:
            plotly.graph_objects.Figure: Plotly figure object representing the regular heatmap.
        """
        z_scores_array = z_scores_df.to_numpy()

        fig = px.imshow(
            z_scores_array, 
            labels=dict(color="Z-score"),
            x=selected_samples,
            y=z_scores_df.index.get_level_values('LipidMolec'),
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )

        fig.update_layout(
            title='Regular Lipidomic Heatmap',
            coloraxis_colorbar=dict(title="Z-score"),
            xaxis_title="Samples",
            yaxis_title="Lipid Molecules",
            margin=dict(l=10, r=10, t=25, b=20)
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickmode='array')

        return fig
    
