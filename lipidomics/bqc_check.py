import streamlit as st 
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter

class BQCQualityCheck:
    """
    A class dedicated to conducting Batch Quality Control (BQC) assessments on lipidomics data.
    It provides functionalities to calculate the coefficient of variation, mean excluding zeros,
    log transformation, and to generate and display a scatter plot for CoV analysis.
    """
    
    @staticmethod
    def calculate_coefficient_of_variation(numbers):
        """
        Calculate the coefficient of variation (CoV) for a given array of numbers.

        Args:
            numbers (list): A list or array of numerical values.

        Returns:
            float or None: The coefficient of variation as a percentage, or None if calculation is not applicable.
        """
        non_zero_numbers = np.array(numbers)[np.array(numbers) > 0]
        if non_zero_numbers.size > 1:
            return np.std(non_zero_numbers, ddof=1) / np.mean(non_zero_numbers) * 100
        return None
    
    @staticmethod
    def calculate_mean_excluding_zeros(numbers):
        """
        Calculate the mean of a list of numbers, excluding zeros.

        Args:
            numbers (list): A list or array of numerical values.

        Returns:
            float or None: The mean of the non-zero numbers, or None if no non-zero numbers are present.
        """
        non_zero_numbers = np.array(numbers)[np.array(numbers) > 0]
        return np.mean(non_zero_numbers) if non_zero_numbers.size > 1 else None

    @staticmethod
    def apply_log_transformation(series):
        """
        Apply log10 transformation to a Pandas series, excluding zero values.

        Args:
            series (pd.Series): A Pandas series with numerical values.

        Returns:
            pd.Series: A series with the log10 transformation applied to non-zero values.
        """
        return np.log10(series[series > 0])

    @staticmethod
    @st.cache_data
    def prepare_dataframe_for_plot(dataframe, area_under_curve_columns):
        """
        Prepare a DataFrame for plotting by calculating CoV and mean values for specified columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame to process.
            area_under_curve_columns (list): A list of columns representing areas under the curve.

        Returns:
            pd.DataFrame: The processed DataFrame with additional 'cov' and 'mean' columns.
        """
        dataframe['cov'] = dataframe[area_under_curve_columns].apply(
            BQCQualityCheck.calculate_coefficient_of_variation, axis=1)
        dataframe['mean'] = dataframe[area_under_curve_columns].apply(
            BQCQualityCheck.calculate_mean_excluding_zeros, axis=1)
        dataframe.loc[dataframe['mean'].notnull(), 'mean'] = BQCQualityCheck.apply_log_transformation(dataframe['mean'])
        return dataframe

    @staticmethod
    def prepare_plot_inputs(dataframe):
        """
        Prepare the inputs needed for creating a scatter plot from the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame with the necessary data.

        Returns:
            tuple: Three arrays representing mean concentrations, CoV values, and lipid species.
        """
        return dataframe['mean'].values, dataframe['cov'].values, dataframe['LipidMolec'].values

    @staticmethod
    def create_cov_scatter_plot(mean_concentrations, coefficients_of_variation, lipid_species):
        """
        Create a scatter plot showing the coefficient of variation for lipid species.

        Args:
            mean_concentrations (array): Array of mean concentrations.
            coefficients_of_variation (array): Array of CoV values.
            lipid_species (array): Array of lipid species names.

        Returns:
            tuple: A Bokeh figure object and a DataFrame used for the plot.
        """
        scatter_plot = figure(
            title='CoV - All lipid Species', 
            x_axis_label='Log10 of Mean BQC Concentration', 
            y_axis_label='CoV(%)'
        )
        
        cov_df = pd.DataFrame({
            "Mean_concentration": mean_concentrations, 
            "CoV": coefficients_of_variation, 
            'Species': lipid_species
        })

        source = ColumnDataSource(cov_df)
        scatter_plot.scatter(
            x="Mean_concentration", 
            y="CoV", 
            source=source, 
            name='cov'
        )

        hover_tool = HoverTool(
            tooltips=[
                ('Mean_concentration', '@Mean_concentration'), 
                ('CoV', "@CoV"), 
                ('Species', "@Species")
            ], 
            names=['cov']
        )

        scatter_plot.add_tools(hover_tool)

        # Set font sizes for axes labels
        scatter_plot.xaxis.axis_label_text_font_size = "15pt"
        scatter_plot.yaxis.axis_label_text_font_size = "15pt"

        # Set font sizes for axis ticks
        scatter_plot.xaxis.major_label_text_font_size = "12pt"
        scatter_plot.yaxis.major_label_text_font_size = "12pt"

        scatter_plot.title.text_font_size = "15pt"

        return scatter_plot, cov_df

    @staticmethod
    @st.cache_data
    def generate_cov_plot_data(dataframe, individual_samples_list, bqc_sample_index):
        """
        Prepare data for generating a CoV scatter plot.
        
        Args:
            dataframe (pd.DataFrame): The DataFrame containing lipidomics data.
            _experiment_details (Experiment): An object containing details about the experiment setup. Prefixed with _ to avoid hashing.
            bqc_sample_index (int): The index of BQC samples in the experiment's sample list.
        
        Returns:
            pd.DataFrame: The DataFrame prepared for plotting.
        """
        #bqc_samples_list = _experiment_details.individual_samples_list[bqc_sample_index]
        bqc_samples_list = individual_samples_list[bqc_sample_index]
        auc = ['MeanArea[' + sample + ']' for sample in bqc_samples_list]
        return BQCQualityCheck.prepare_dataframe_for_plot(dataframe.copy(), auc)
    
    @staticmethod
    def generate_and_display_cov_plot(dataframe, experiment_details, bqc_sample_index):
        """
        Generate and display a CoV scatter plot based on BQC samples from a given DataFrame.
    
        Args:
            dataframe (pd.DataFrame): The DataFrame containing lipidomics data.
            experiment_details (Experiment): An object containing details about the experiment setup.
            bqc_sample_index (int): The index of BQC samples in the experiment's sample list.
    
        Returns:
            tuple: A Bokeh figure object for the scatter plot, the DataFrame used for plotting, and the reliable data percentage.
        """
        prepared_df = BQCQualityCheck.generate_cov_plot_data(dataframe, experiment_details.individual_samples_list, bqc_sample_index)
        mean_concentrations, coefficients_of_variation, lipid_species = BQCQualityCheck.prepare_plot_inputs(prepared_df)
        scatter_plot, cov_df = BQCQualityCheck.create_cov_scatter_plot(mean_concentrations, coefficients_of_variation, lipid_species)
        
        reliable_data_percent = round(len(prepared_df[prepared_df['cov'] < 30]) / len(prepared_df) * 100, 1)
    
        return scatter_plot, prepared_df, reliable_data_percent

    @staticmethod
    @st.cache_data
    def filter_dataframe_by_cov_threshold(threshold, prepared_df):
        """
        Filter a DataFrame based on a specified CoV threshold.

        Args:
            threshold (float): The CoV threshold for filtering.
            prepared_df (pd.DataFrame): The DataFrame to be filtered.

        Returns:
            pd.DataFrame: The filtered DataFrame with irrelevant columns removed and index reset.
        """
        filtered_df = prepared_df[prepared_df['cov'] <= threshold]
        return filtered_df.drop(['mean', 'cov'], axis=1).reset_index(drop=True)