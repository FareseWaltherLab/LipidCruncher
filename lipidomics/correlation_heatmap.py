import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

class Correlation:
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def prepare_data_for_correlation(df, individual_samples_list, condition_index):
        """
        Prepares data for correlation analysis by selecting mean area columns and renaming them.

        Args:
            df (pd.DataFrame): The DataFrame containing the experiment data.
            experiment (Experiment): The experiment object with setup details.
            condition_index (int): Index of the selected condition.

        Returns:
            pd.DataFrame: A DataFrame with selected mean area columns.
        """
        mean_area_df = df[['concentration[' + sample + ']' for sample in individual_samples_list[condition_index]]]
        mean_area_df.columns = individual_samples_list[condition_index]
        return mean_area_df

    @staticmethod
    @st.cache_data(ttl=3600)
    def compute_correlation(df, sample_type):
        """
        Computes the correlation matrix for the given DataFrame and sample type.

        Args:
            df (pd.DataFrame): DataFrame to calculate correlations for.
            sample_type (str): Type of samples ('biological replicates' or 'Technical replicates').

        Returns:
            tuple: A tuple containing the correlation DataFrame, vmin, and threshold values for heatmap plotting.
        """
        #v_min = 0.5 if sample_type == 'biological replicates' else 0.75
        v_min = 0.5
        thresh = 0.7 if sample_type == 'biological replicates' else 0.8
        correlation_df = df.corr()
        return correlation_df, v_min, thresh

    @staticmethod
    def render_correlation_plot(correlation_df, v_min, thresh, condition):
        """
        Renders a correlation heatmap plot.
    
        Args:
            correlation_df (pd.DataFrame): DataFrame containing correlation data.
            v_min (float): Minimum value for the heatmap color scale.
            thresh (float): Center value for the heatmap color scale.
            condition (str): The condition label for which the heatmap is being rendered.
    
        Returns:
            matplotlib.figure.Figure: The figure object containing the heatmap.
        """
        fig = plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        sns.set(font_scale=3)
        heatmap = sns.heatmap(correlation_df, mask=mask, vmin=v_min, vmax=1, center=thresh, annot=False, cmap='RdBu', square=False, cbar=True)
        heatmap.set_title('Triangle Correlation Heatmap - ' + condition, fontdict={'fontsize': 40})
        plt.close(fig)  # Close the plot to prevent it from displaying twice
        return fig