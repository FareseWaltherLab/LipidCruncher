import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

class AbundanceBarChart:
    """
    A class for generating abundance bar charts from lipidomics data.

    This class processes lipidomics data to calculate and visualize the total 
    abundance of lipid classes under selected experimental conditions. It supports 
    different modes of data representation (linear or logarithmic scale) and provides 
    functionalities for grouping data, calculating mean and standard deviation, 
    handling log transformations, and filtering based on selected classes. The class 
    also includes methods for customizing and rendering the final plot.
    """

    @staticmethod
    @st.cache_data
    def create_mean_std_columns(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes):
        """
        Aggregates and computes mean and standard deviation for selected lipid classes 
        across specific conditions using provided sample lists.

        This method groups the data by lipid classes and calculates the mean and standard 
        deviation of the mean areas for each condition, then computes log2 values for 
        improved visualization and analysis. The results are filtered for the selected classes.

        Parameters:
        df (pd.DataFrame): The DataFrame containing lipidomics data.
        full_samples_list (list): List of all sample names in the dataset.
        individual_samples_list (list of list): List of sample names categorized by condition.
        conditions_list (list): List of experimental conditions.
        selected_conditions (list of str): Conditions selected for analysis.
        selected_classes (list of str): Lipid classes selected for analysis.

        Returns:
        pd.DataFrame: DataFrame with aggregated mean, standard deviation, and log2 values 
                      for the selected classes and conditions.
        """
        grouped_df = AbundanceBarChart.group_and_sum(df, full_samples_list)
        AbundanceBarChart.calculate_mean_std_for_conditions(grouped_df, individual_samples_list, conditions_list, selected_conditions)
        grouped_df = AbundanceBarChart.filter_by_selected_classes(grouped_df, selected_classes)
        AbundanceBarChart.calculate_log2_values(grouped_df, selected_conditions)
        return grouped_df

    @staticmethod
    @st.cache_data
    def group_and_sum(df, full_samples_list):
        """
        Groups and sums the mean area values for each lipid class based on the full sample list.

        Parameters:
        df (pd.DataFrame): The DataFrame containing lipidomics data.
        full_samples_list (list): List of all sample names in the dataset.

        Returns:
        pd.DataFrame: DataFrame grouped by 'ClassKey' with summed mean areas.
        """
        return df.groupby('ClassKey')[[f"MeanArea[{sample}]" for sample in full_samples_list]].sum().reset_index()

    @staticmethod
    def calculate_mean_std_for_conditions(grouped_df, individual_samples_list, conditions_list, selected_conditions):
        """
        Calculates mean and standard deviation for specific conditions in the grouped DataFrame.

        Parameters:
        grouped_df (pd.DataFrame): DataFrame grouped by lipid classes.
        individual_samples_list (list of list): List of sample names categorized by condition.
        conditions_list (list): List of experimental conditions.
        selected_conditions (list of str): Conditions selected for analysis.
        """
        for condition in selected_conditions:
            condition_index = conditions_list.index(condition)
            individual_samples = individual_samples_list[condition_index]
            mean_cols = [f"MeanArea[{sample}]" for sample in individual_samples]
            grouped_df[f"mean_AUC_{condition}"] = grouped_df[mean_cols].mean(axis=1)
            grouped_df[f"std_AUC_{condition}"] = grouped_df[mean_cols].std(axis=1)

    @staticmethod
    @st.cache_data
    def calculate_log2_values(grouped_df, selected_conditions):
        """
        Computes log2 transformed mean and standard deviation values for selected conditions.

        Parameters:
        grouped_df (pd.DataFrame): DataFrame with mean and standard deviation values.
        selected_conditions (list of str): Conditions selected for analysis.

        Returns:
        tuple: A tuple containing the filtered DataFrame and a list of removed classes.
        """
        for condition in selected_conditions:
            mean_col = f"mean_AUC_{condition}"
            std_col = f"std_AUC_{condition}"
            if mean_col in grouped_df.columns and std_col in grouped_df.columns:
                log2_mean_col = f"log2_mean_AUC_{condition}"
                log2_std_col = f"log2_std_AUC_{condition}"
                
                grouped_df[log2_mean_col] = np.where(grouped_df[mean_col] > grouped_df[std_col],
                                                     np.log2(grouped_df[mean_col]),
                                                     np.nan)
                grouped_df[log2_std_col] = np.where(grouped_df[mean_col] > grouped_df[std_col],
                                                    (np.log2(grouped_df[mean_col] + grouped_df[std_col]) - 
                                                     np.log2(grouped_df[mean_col] - grouped_df[std_col])) / 2,
                                                    np.nan)
        
        # Count valid conditions for each class
        valid_condition_count = grouped_df[[f"log2_mean_AUC_{condition}" for condition in selected_conditions]].notna().sum(axis=1)
        
        # Filter classes with at least two valid conditions
        filtered_df = grouped_df[valid_condition_count >= 2]
        
        # Identify removed classes
        removed_classes = grouped_df[valid_condition_count < 2]['ClassKey'].tolist()
        
        return filtered_df, removed_classes

    @staticmethod
    def filter_by_selected_classes(grouped_df, selected_classes):
        """
        Filters the DataFrame to include only the selected lipid classes.

        Parameters:
        grouped_df (pd.DataFrame): The DataFrame containing aggregated data.
        selected_classes (list of str): Lipid classes selected for analysis.

        Returns:
        pd.DataFrame: Filtered DataFrame containing only data for selected lipid classes.
        """
        return grouped_df[grouped_df['ClassKey'].isin(selected_classes)]

    @staticmethod
    def create_abundance_bar_chart(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes, mode):
        """
        Creates an abundance bar chart for the selected lipid classes and conditions.

        Parameters:
        df (pd.DataFrame): The DataFrame containing lipidomics data.
        full_samples_list (list): List of all sample names in the dataset.
        individual_samples_list (list of list): List of sample names categorized by condition.
        conditions_list (list): List of experimental conditions.
        selected_conditions (list of str): Conditions selected for analysis.
        selected_classes (list of str): Lipid classes selected for analysis.
        mode (str): The mode for value calculation ('linear scale' or 'log2 scale').

        Returns:
        tuple: A tuple containing the matplotlib figure and the DataFrame used for plotting.
        """
        abundance_df = AbundanceBarChart.create_mean_std_columns(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes)
        
        if mode == 'log2 scale':
            abundance_df, removed_classes = AbundanceBarChart.calculate_log2_values(abundance_df, selected_conditions)
        else:
            removed_classes = []
        
        fig, ax = AbundanceBarChart.initialize_plot(len(abundance_df))
        AbundanceBarChart.add_bars_to_plot(ax, abundance_df, selected_conditions, mode)
        AbundanceBarChart.style_plot(ax, abundance_df)
        
        # Add note about removed classes if any
        if removed_classes:
            note = (
                f"Note: The following classes were removed from the log2 scale plot due to insufficient valid data: {', '.join(removed_classes)}. "
                "Insufficient valid data means that for these classes, fewer than two of the selected conditions had a mean value greater than its standard deviation. "
                "This occurs when the data for these classes shows high variability relative to its average, "
                "which can lead to unreliable or undefined results in log2 scale. "
                "These classes are still included in the linear scale plot if available."
            )
            st.write(note)
        
        return fig, abundance_df

    @staticmethod
    def initialize_plot(num_classes):
        """
        Initializes a matplotlib plot with a white background and dynamic figure size.

        Parameters:
        num_classes (int): The number of lipid classes to be plotted.

        Returns:
        tuple: A tuple containing a figure and axis object of the plot.
        """
        fig_width = 10
        fig_height = max(5, num_classes * 0.5)  # Adjust height based on number of classes
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        return fig, ax

    @staticmethod
    def add_bars_to_plot(ax, abundance_df, selected_conditions, mode):
        """
        Adds horizontal bars to the plot for selected conditions and mode.

        Parameters:
        ax (matplotlib.axes.Axes): The axes object to add bars to.
        abundance_df (pd.DataFrame): DataFrame with calculated mean and std values.
        selected_conditions (list of str): Conditions selected for analysis.
        mode (str): The mode for value calculation ('linear scale' or 'log2 scale').
        """
        y = np.arange(len(abundance_df))
        width = 1 / (len(selected_conditions) + 1)
        bar_height = 0.8 / len(selected_conditions)  # Adjust bar height to add spacing
        multiplier = 0
        for condition in selected_conditions:
            mean, std = AbundanceBarChart.get_mode_specific_values(abundance_df, condition, mode)
            if mean.isnull().all() or std.isnull().all() or mean.empty or std.empty:
                continue  # Skip if mean or std are all NaN or empty
            offset = width * multiplier
            ax.barh(y + offset, mean, bar_height, xerr=std, label=condition, align='center')
            multiplier += 1
    
        ax.set_yticks(y + width * (len(selected_conditions) - 1) / 2)
        ax.set_yticklabels(abundance_df['ClassKey'].values, rotation=45, ha='right', fontsize=10)

    @staticmethod
    def get_mode_specific_values(abundance_df, condition, mode):
        """
        Retrieves mean and standard deviation values based on the selected mode.

        Parameters:
        abundance_df (pd.DataFrame): DataFrame with calculated mean and std values.
        condition (str): Specific condition for which to retrieve values.
        mode (str): The mode for value calculation ('linear scale' or 'log2 scale').

        Returns:
        tuple: A tuple containing mean and standard deviation values.
        """
        if mode == 'linear scale':
            mean = abundance_df.get(f"mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            std = abundance_df.get(f"std_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        elif mode == 'log2 scale':
            mean = abundance_df.get(f"log2_mean_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
            std = abundance_df.get(f"log2_std_AUC_{condition}", pd.Series(dtype=float)).fillna(0)
        return mean, std

    @staticmethod
    def style_plot(ax, abundance_df):
        """
        Styles the plot with labels, title, legend, and a frame.

        Parameters:
        ax (matplotlib.axes.Axes): The axes object to style.
        abundance_df (pd.DataFrame): DataFrame used for the plot.
        """
        ax.set_xlabel('Mean Concentration', fontsize=15)
        ax.set_ylabel('Lipid Class', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='lower right', fontsize=12)
        ax.set_title('Class Concentration Bar Chart', fontsize=15)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        plt.tight_layout(pad=2.0)