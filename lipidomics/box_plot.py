import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

class BoxPlot:
    """
    Class for generating and displaying box plots for lipidomics data analysis.
    """
    
    def __init__(self):
        pass

    @staticmethod
    @st.cache_data(ttl=3600)
    def create_mean_area_df(df, full_samples_list):
        """
        Creates a DataFrame containing only the 'concentration' columns from the provided DataFrame.
        Args:
            df (pd.DataFrame): The dataset to be processed.
            full_samples_list (list[str]): List of sample names.
        Returns:
            pd.DataFrame: A DataFrame with the 'concentration' columns.
        """
        concentration_cols = [f'concentration[{sample}]' for sample in full_samples_list]
        return df[concentration_cols]

    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_missing_values_percentage(mean_area_df):
        """
        Calculates the percentage of missing values for each sample in the experiment.

        Args:
            mean_area_df (pd.DataFrame): DataFrame containing mean area data for the samples.

        Returns:
            list: A list containing the percentage of missing values for each sample.
        """
        return [len(mean_area_df[mean_area_df[col] == 0]) / len(mean_area_df) * 100 for col in mean_area_df.columns]
    
    @staticmethod
    def plot_missing_values(full_samples_list, zero_values_percent_list):
        """
        Plots a bar chart of missing values percentage for each sample.

        Args:
            full_samples_list (list[str]): List of sample names.
            zero_values_percent_list (list[float]): List of missing value percentages for each sample.
        """
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.barh(full_samples_list, zero_values_percent_list)
        ax.set_xlabel('Percentage of Missing Values')
        ax.set_ylabel('Sample')
        ax.set_title('Missing Values Distribution')
        return fig

    @staticmethod
    def plot_box_plot(mean_area_df, full_samples_list):
        """
        Plots a box plot for the non-zero values in the mean area DataFrame.

        Args:
            mean_area_df (pd.DataFrame): DataFrame containing mean area data for the samples.
            full_samples_list (list[str]): List of sample names.
        """
        log_transformed_data = [list(np.log10(mean_area_df[col][mean_area_df[col] > 0])) for col in mean_area_df.columns]
        
        plt.rcdefaults()
        fig, ax = plt.subplots()
        plt.boxplot(log_transformed_data)
        ax.set_xlabel('Sample')
        ax.set_ylabel('log10(Concentration)')
        ax.set_title('Box Plot of Non-Zero Concentrations')
        ax.set_xticklabels(full_samples_list)
        return fig
