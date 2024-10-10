import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
import plotly.graph_objects as go
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
    @st.cache_data(ttl=3600)
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
    @st.cache_data(ttl=3600)
    def perform_clustering(z_scores_df, n_clusters):
        linkage_matrix = linkage(pdist(z_scores_df, 'euclidean'), method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        dendrogram_order = leaves_list(linkage_matrix)
        return linkage_matrix, cluster_labels, dendrogram_order
    
    @staticmethod
    def identify_clusters_and_percentages(z_scores_df, n_clusters):
        _, cluster_labels, _ = LipidomicHeatmap.perform_clustering(z_scores_df, n_clusters)
        
        clustered_df = z_scores_df.copy()
        clustered_df['Cluster'] = cluster_labels
        
        class_percentages = clustered_df.groupby('Cluster').apply(
            lambda x: x.index.get_level_values('ClassKey').value_counts(normalize=True)
        ).unstack(fill_value=0) * 100
        
        return n_clusters, class_percentages
    
    @staticmethod
    def generate_clustered_heatmap(z_scores_df, selected_samples, n_clusters):
        linkage_matrix, cluster_labels, dendrogram_order = LipidomicHeatmap.perform_clustering(z_scores_df, n_clusters)
        clustered_df = z_scores_df.iloc[dendrogram_order]
        clustered_df['Cluster'] = cluster_labels

        z_scores_array = clustered_df.drop('Cluster', axis=1).to_numpy()

        # Ensure z_scores_array is a 2D numpy array
        if z_scores_array.ndim == 1:
            z_scores_array = z_scores_array.reshape(-1, 1)

        # Calculate the color scale range
        vmin = np.nanmin(z_scores_array)
        vmax = np.nanmax(z_scores_array)
        abs_max = max(abs(vmin), abs(vmax))

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_scores_array,
            x=selected_samples,
            y=clustered_df.index.get_level_values('LipidMolec'),
            colorscale='RdBu_r',
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title="Z-score")
        ))

        # Add cluster boundaries
        cluster_sizes = clustered_df['Cluster'].value_counts().sort_index()
        cumulative_sizes = np.cumsum(cluster_sizes.values[:-1])  # We don't need a line after the last cluster
        
        for size in cumulative_sizes:
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=size - 0.5,
                x1=len(selected_samples) - 0.5,
                y1=size - 0.5,
                line=dict(color="black", width=2, dash="dash")
            )

        fig.update_layout(
            title='Clustered Lipidomic Heatmap',
            xaxis_title="Samples",
            yaxis_title="Lipid Molecules",
            margin=dict(l=100, r=100, t=50, b=50)
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickmode='array', autorange="reversed")

        class_percentages = clustered_df.groupby('Cluster').apply(
            lambda x: x.index.get_level_values('ClassKey').value_counts(normalize=True)
        ).unstack(fill_value=0) * 100

        return fig, class_percentages
    
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
    
