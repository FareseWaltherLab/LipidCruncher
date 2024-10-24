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
    def perform_statistical_tests(continuation_df, experiment, selected_conditions, selected_classes):
        """
        Performs appropriate statistical tests based on number of conditions.
        """
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        statistical_results = {}
        
        # Statistical testing guidance
        if len(selected_conditions) > 2:
            st.info("""
            📊 **Statistical Testing Note:**
            - Multiple conditions detected: Using ANOVA + Tukey's test
            - This is the correct approach when interested in multiple comparisons
            - Do NOT run separate t-tests by unselecting conditions - this would inflate the false positive rate
            - The current analysis automatically adjusts significance thresholds to maintain a 5% false positive rate across all comparisons
            """)
        elif len(selected_conditions) == 2:
            st.info("""
            📊 **Statistical Testing Note:**
            - Two conditions detected: Using t-test
            - This is appropriate ONLY for a single pre-planned comparison
            - If you plan to compare multiple conditions, please select all relevant conditions to use ANOVA + Tukey's test
            """)
        
        for lipid_class in selected_classes:
            try:
                # Filter for current lipid class
                class_df = continuation_df[continuation_df['ClassKey'] == lipid_class].copy()
                
                # Prepare data for statistical testing
                data_for_stats = []
                conditions_for_stats = []
                
                for condition in selected_conditions:
                    condition_idx = experiment.conditions_list.index(condition)
                    condition_samples = experiment.individual_samples_list[condition_idx]
                    
                    # Get data columns for this condition
                    sample_columns = [f"MeanArea[{sample}]" for sample in condition_samples 
                                    if f"MeanArea[{sample}]" in class_df.columns]
                    
                    if sample_columns:
                        condition_data = class_df[sample_columns].sum(axis=1).values
                        data_for_stats.extend(condition_data)
                        conditions_for_stats.extend([condition] * len(condition_data))
                
                if len(selected_conditions) == 2:
                    # Perform t-test
                    group1 = [x for x, c in zip(data_for_stats, conditions_for_stats) 
                             if c == selected_conditions[0]]
                    group2 = [x for x, c in zip(data_for_stats, conditions_for_stats) 
                             if c == selected_conditions[1]]
                    
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                        statistical_results[lipid_class] = {
                            'test': 't-test',
                            'statistic': t_stat,
                            'p-value': p_value
                        }
                else:
                    # Perform ANOVA
                    groups = []
                    for condition in selected_conditions:
                        group = [x for x, c in zip(data_for_stats, conditions_for_stats) 
                                if c == condition]
                        if group:
                            groups.append(group)
                    
                    if len(groups) > 1:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        # Perform Tukey's test if ANOVA is significant
                        tukey_results = None
                        if p_value < 0.05:
                            tukey = pairwise_tukeyhsd(data_for_stats, conditions_for_stats)
                            tukey_results = {
                                'group1': [],
                                'group2': [],
                                'p_values': []
                            }
                            
                            for row in tukey._results_table[1:]:
                                tukey_results['group1'].append(str(row[0]))
                                tukey_results['group2'].append(str(row[1]))
                                # Handle p-value carefully
                                try:
                                    if isinstance(row[3], (float, int)):
                                        p_val = float(row[3])
                                    elif isinstance(row[3], str):
                                        if row[3].lower() == 'cell':
                                            p_val = 1.0  # Use 1.0 for non-significant results
                                        else:
                                            p_val = float(row[3])
                                    else:
                                        p_val = 1.0
                                    tukey_results['p_values'].append(p_val)
                                except (ValueError, TypeError):
                                    tukey_results['p_values'].append(1.0)
                        
                        statistical_results[lipid_class] = {
                            'test': 'ANOVA',
                            'statistic': f_stat,
                            'p-value': p_value,
                            'tukey_results': tukey_results
                        }
                
            except Exception as e:
                st.warning(f"Could not perform statistical test for {lipid_class}: {str(e)}")
                continue
        
        return statistical_results

    @staticmethod
    def display_statistical_details(statistical_results, selected_conditions):
        """
        Displays detailed statistical results in a formatted way.
        """
        st.write("### Detailed Statistical Analysis")
        
        if len(selected_conditions) == 2:
            st.write(f"Comparing conditions: {selected_conditions[0]} vs {selected_conditions[1]}")
            st.write("Method: Independent t-test (Welch's t-test)")
        else:
            st.write(f"Comparing {len(selected_conditions)} conditions using ANOVA + Tukey's test")
            st.write("Note: P-values are adjusted for multiple comparisons")
        
        for lipid_class, results in statistical_results.items():
            st.write(f"\n#### {lipid_class}")
            p_value = results['p-value']
            
            if results['test'] == 't-test':
                st.write(f"t-statistic: {results['statistic']:.3f}")
                st.write(f"p-value: {p_value:.3f}")
            else:  # ANOVA
                st.write(f"F-statistic: {results['statistic']:.3f}")
                st.write(f"p-value: {p_value:.3f}")
                
                if results.get('tukey_results') and p_value < 0.05:
                    st.write("\nSignificant pairwise comparisons (Tukey's test):")
                    tukey = results['tukey_results']
                    for g1, g2, p in zip(tukey['group1'], tukey['group2'], tukey['p_values']):
                        if p < 0.05:
                            st.write(f"- {g1} vs {g2}: p = {p:.3f}")

    @staticmethod
    def create_abundance_bar_chart(df, full_samples_list, individual_samples_list, conditions_list, selected_conditions, selected_classes, mode, anova_results=None):
        """
        Creates an abundance bar chart with statistical significance indicators.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame containing lipid data
        full_samples_list (list): List of all samples
        individual_samples_list (list): List of samples for each condition
        conditions_list (list): List of all conditions
        selected_conditions (list): List of conditions selected for analysis
        selected_classes (list): List of lipid classes selected for analysis
        mode (str): Display mode ('linear scale' or 'log2 scale')
        anova_results (dict): Results from statistical testing
        """
        full_samples_list = st.session_state.full_samples_list if 'full_samples_list' in st.session_state else full_samples_list
    
        try:
            # Validate and prepare data
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
    
            # Create mean and std columns for selected conditions
            abundance_df = AbundanceBarChart.create_mean_std_columns(
                df, full_samples_list, individual_samples_list, 
                conditions_list, selected_conditions, selected_classes
            )
        
            if abundance_df.empty:
                st.error("No data available after processing for the selected conditions and classes.")
                return None, None
    
            # Create plot
            fig, ax = AbundanceBarChart.initialize_plot(len(abundance_df))
            
            # Add bars to plot
            y = np.arange(len(abundance_df))
            width = 1 / (len(selected_conditions) + 1)
            bar_height = 0.8 / len(selected_conditions)
            multiplier = 0
            
            for condition in selected_conditions:
                mean, std = AbundanceBarChart.get_mode_specific_values(abundance_df, condition, mode)
                if mean.isnull().all() or std.isnull().all() or mean.empty or std.empty:
                    continue
                offset = width * multiplier
                ax.barh(y + offset, mean, bar_height, xerr=std, label=condition, align='center')
                multiplier += 1
    
            ax.set_yticks(y + width * (len(selected_conditions) - 1) / 2)
            ax.set_yticklabels(abundance_df['ClassKey'].values, rotation=45, ha='right', fontsize=20)
    
            # Style the plot
            ax.set_xlabel('Mean Concentration', fontsize=15)
            ax.set_ylabel('Lipid Class', fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.legend(loc='lower right', fontsize=12)
            ax.set_title('Class Concentration Bar Chart', fontsize=15)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
    
            # Add statistical significance annotations
            if anova_results:
                y_positions = np.arange(len(abundance_df))
                max_x = ax.get_xlim()[1]
                
                # Add a legend for significance levels
                if any(result['p-value'] < 0.05 for result in anova_results.values()):
                    significance_legend = [
                        "Statistical Significance:",
                        "* p < 0.05",
                        "** p < 0.01",
                        "*** p < 0.001"
                    ]
                    
                    # Add testing method used
                    if len(selected_conditions) == 2:
                        significance_legend.insert(0, "Method: t-test")
                    else:
                        significance_legend.insert(0, "Method: ANOVA + Tukey test")
                    
                    # Add legend text to plot
                    legend_y = -0.2  # Adjust based on your plot layout
                    for i, text in enumerate(significance_legend):
                        plt.figtext(0.7, legend_y - i*0.03, text, fontsize=8)
                
                for i, lipid_class in enumerate(abundance_df['ClassKey']):
                    if lipid_class in anova_results:
                        result = anova_results[lipid_class]
                        p_value = result['p-value']
                        
                        # Get significance level
                        significance = ''
                        if p_value < 0.001:
                            significance = '***'
                        elif p_value < 0.01:
                            significance = '**'
                        elif p_value < 0.05:
                            significance = '*'
                        
                        if significance:
                            # Add overall significance
                            ax.text(max_x, y_positions[i], significance, 
                                   ha='left', va='center')
    
            plt.tight_layout(pad=2.0)
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
        ax.set_yticklabels(abundance_df['ClassKey'].values, rotation=45, ha='right', fontsize=20)

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