import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

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
    @st.cache_data(ttl=3600)
    def create_mean_std_columns(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes):
        try:
            available_samples = [sample for sample in full_samples_list if f"MeanArea[{sample}]" in df.columns]
            
            grouped_df = AbundanceBarChart.group_and_sum(df, available_samples)
            
            for condition in selected_conditions:
                if condition not in conditions_list:
                    continue
                condition_index = conditions_list.index(condition)
                individual_samples = [sample for sample in individual_samples_list[condition_index] if sample in available_samples]
                
                if not individual_samples:
                    continue
                
                mean_cols = [f"MeanArea[{sample}]" for sample in individual_samples]
                grouped_df[f"mean_AUC_{condition}"] = grouped_df[mean_cols].mean(axis=1)
                grouped_df[f"std_AUC_{condition}"] = grouped_df[mean_cols].std(axis=1)
    
            grouped_df = AbundanceBarChart.filter_by_selected_classes(grouped_df, selected_classes)
            grouped_df = AbundanceBarChart.calculate_log2_values(grouped_df, selected_conditions)
            
            return grouped_df
    
        except Exception as e:
            st.error(f"Error in create_mean_std_columns: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=3600)
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
    @st.cache_data(ttl=3600)
    def calculate_log2_values(grouped_df, selected_conditions):
        for condition in selected_conditions:
            mean_col = f"mean_AUC_{condition}"
            std_col = f"std_AUC_{condition}"
            if mean_col in grouped_df.columns and std_col in grouped_df.columns:
                log2_mean_col = f"log2_mean_AUC_{condition}"
                log2_std_col = f"log2_std_AUC_{condition}"
                
                grouped_df[log2_mean_col] = np.log2(grouped_df[mean_col].replace(0, np.nan))
                grouped_df[log2_std_col] = grouped_df[std_col] / (grouped_df[mean_col] * np.log(2))
        
        return grouped_df

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
    def perform_two_way_anova(continuation_df, experiment, selected_conditions, selected_classes):
        anova_results = {}
        
        for lipid_class in selected_classes:
            class_df = continuation_df[continuation_df['ClassKey'] == lipid_class]
            
            data = []
            species = []
            conditions = []
            
            for condition in selected_conditions:
                condition_samples = experiment.individual_samples_list[experiment.conditions_list.index(condition)]
                for sample in condition_samples:
                    col_name = f"MeanArea[{sample}]"
                    if col_name in class_df.columns:
                        data.extend(class_df[col_name])
                        species.extend(class_df['LipidMolec'])
                        conditions.extend([condition] * len(class_df))
            
            if len(data) > 0:
                f_value, p_value = stats.f_oneway(*[group for name, group in pd.DataFrame({'data': data, 'condition': conditions}).groupby('condition')['data']])
                anova_results[lipid_class] = {'F-value': f_value, 'p-value': p_value}
        
        return anova_results

    @staticmethod
    def create_abundance_bar_chart(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes, mode, anova_results=None):
        full_samples_list = st.session_state.full_samples_list if 'full_samples_list' in st.session_state else full_samples_list
    
        try:
            expected_columns = ['ClassKey'] + [f"MeanArea[{sample}]" for sample in full_samples_list]
            valid_columns = ['ClassKey'] + [col for col in expected_columns if col in df.columns and col != 'ClassKey']
            df = df[valid_columns]
    
            if df.empty or len(valid_columns) <= 1:
                st.error("No valid data available to create the abundance bar chart.")
                return None, None
    
            if 'ClassKey' not in df.columns:
                st.error("ClassKey column is missing from the dataset.")
                return None, None
            
            if df['ClassKey'].dtype == 'object' and df['ClassKey'].str.contains(',').any():
                st.error("ClassKey column contains multiple values. Please ensure it's a single value per row.")
                return None, None
    
            selected_conditions = [cond for cond in selected_conditions if cond in conditions_list]
    
            abundance_df = AbundanceBarChart.create_mean_std_columns(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes)
        
            if abundance_df.empty:
                st.error("No data available after processing for the selected conditions and classes.")
                return None, None
    
            fig, ax = AbundanceBarChart.initialize_plot(len(abundance_df))
            
            AbundanceBarChart.add_bars_to_plot(ax, abundance_df, selected_conditions, mode)
            
            AbundanceBarChart.style_plot(ax, abundance_df)
            
            # Add significance indicators directly to the plot
            y_positions = np.arange(len(abundance_df))
            for i, lipid_class in enumerate(abundance_df['ClassKey']):
                if lipid_class in anova_results:
                    p_value = anova_results[lipid_class]['p-value']
                    significance = ''
                    if p_value < 0.001:
                        significance = '***'
                    elif p_value < 0.01:
                        significance = '**'
                    elif p_value < 0.05:
                        significance = '*'
                    
                    if significance:
                        ax.text(ax.get_xlim()[1], y_positions[i], significance, ha='left', va='center')
            
            # Add a legend for significance
            ax.text(1.05, -0.05, '* p < 0.05\n** p < 0.01\n*** p < 0.001', transform=ax.transAxes, ha='left', va='top')
            
            return fig, abundance_df
    
        except Exception as e:
            st.error(f"An unexpected error occurred while creating the abundance bar chart: {str(e)}")
            return None, None

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