import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import streamlit as st

class PathwayViz:

    @staticmethod
    def calculate_number_of_saturated_and_unsaturated_chains_for_single_species(mol_structure):
        """
        Calculates the number of saturated and unsaturated fatty acid chains in a single lipid molecule.

        Args:
            mol_structure (str): The molecular structure of a lipid, represented as a string.

        Returns:
            tuple: A tuple (num_saturated, num_unsaturated) representing the number of saturated and unsaturated chains.
        """
        chains = mol_structure.split('(')[1][:-1].split('_')
        sat_unsat_count = [0, 0]
    
        for chain in chains:
            chain_saturation = chain.split(':')[-1]
            if chain_saturation == '0' or '0' in chain_saturation and not chain_saturation.isnumeric():
                sat_unsat_count[0] += 1  # Increase count of saturated chains
            else:
                sat_unsat_count[1] += 1  # Increase count of unsaturated chains
    
        return tuple(sat_unsat_count)

    @staticmethod
    @st.cache_data
    def calculate_total_number_of_saturated_and_unsaturated_chains_for_each_class(df):
        """
        Calculates the total number of saturated and unsaturated chains for each lipid class in a DataFrame.

        Args:
            df (DataFrame): DataFrame containing lipid molecules and their classes.

        Returns:
            DataFrame: DataFrame with added columns for total saturated and unsaturated chain counts per class.
        """
        df['number_of_sat_unsat_chains'] = df['LipidMolec'].apply(
            PathwayViz.calculate_number_of_saturated_and_unsaturated_chains_for_single_species)
        df[['number_of_sat_chains', 'number_of_unsat_chains']] = pd.DataFrame(
            df['number_of_sat_unsat_chains'].tolist(), index=df.index)
        return df

    @staticmethod
    @st.cache_data
    def add_saturation_ratio_column(df):
        """
        Adds a saturation ratio column to a DataFrame grouped by lipid class.

        Args:
            df (DataFrame): DataFrame to add saturation ratio to.

        Returns:
            DataFrame: Updated DataFrame with saturation ratio column.
        """
        saturation_ratio_df = df.groupby('ClassKey')[['number_of_sat_chains', 'number_of_unsat_chains']].sum()
        saturation_ratio_df['saturation_ratio'] = saturation_ratio_df[['number_of_sat_chains', 'number_of_unsat_chains']].apply(lambda x: x[0]/(x[0]+x[1]), axis = 1)
        return saturation_ratio_df

    @staticmethod
    @st.cache_data
    def calculate_class_saturation_ratio(df):
        """
       Calculates the saturation ratio for each lipid class in a DataFrame.

       Args:
           df (DataFrame): DataFrame containing lipid molecules and their classes.

       Returns:
           DataFrame: DataFrame with saturation ratio for each class.
       """
        df_copy = df.copy()
        df_copy = PathwayViz.calculate_total_number_of_saturated_and_unsaturated_chains_for_each_class(df_copy)
        saturation_ratio_df = df_copy.groupby('ClassKey').apply(
            lambda x: x['number_of_sat_chains'].sum() / max(x['number_of_sat_chains'].sum() + x['number_of_unsat_chains'].sum(), 1))
        return saturation_ratio_df.reset_index(name='saturation_ratio')

    @staticmethod
    @st.cache_data
    def calculate_total_class_abundance(df, full_samples_list):
        """
       Calculates the total abundance of each lipid class based on given sample names.

       Args:
           df (DataFrame): DataFrame with lipid data.
           full_samples_list (list): List of sample names.

       Returns:
           DataFrame: DataFrame with total abundance per class.
       """
        mean_area_cols = ["MeanArea[" + sample + ']' for sample in full_samples_list]
        return df.groupby('ClassKey')[mean_area_cols].sum()

    @staticmethod
    @st.cache_data
    def get_fold_change(abundance_df, control_samples, experimental_samples):
        """
       Calculates the fold change between control and experimental samples for each lipid class.

       Args:
           abundance_df (DataFrame): DataFrame with lipid abundances.
           control_samples (list): List of control sample names.
           experimental_samples (list): List of experimental sample names.

       Returns:
           Series: Pandas Series with fold change values for each class.
       """
        return (abundance_df[[f'MeanArea[{sample}]' for sample in experimental_samples]].mean(axis=1) /
                abundance_df[[f'MeanArea[{sample}]' for sample in control_samples]].mean(axis=1))

    @staticmethod
    @st.cache_data
    def add_fold_change_column(df, control, experimental, control_samples, experimental_samples):
        """
        Adds a column for fold change between control and experimental samples to a DataFrame.

        Args:
            df (DataFrame): DataFrame to add fold change to.
            control (str): Control condition name.
            experimental (str): Experimental condition name.
            control_samples (list): List of control sample names.
            experimental_samples (list): List of experimental sample names.

        Returns:
            DataFrame: Updated DataFrame with fold change column.
        """
        fc_column = f'fc_{experimental}_{control}'
        df[fc_column] = PathwayViz.get_fold_change(df, control_samples, experimental_samples)
        return df[[fc_column]]

    @staticmethod
    def calculate_class_fold_change(df, experiment, control, experimental):
        """
        Calculates the fold change for each lipid class between control and experimental conditions.

        Args:
            df (DataFrame): DataFrame with lipid data.
            experiment: Object containing experimental setup information.
            control (str): Control condition name.
            experimental (str): Experimental condition name.

        Returns:
            DataFrame: DataFrame with fold change values for each class.
        """
        full_samples_list = experiment.full_samples_list
        control_idx, experimental_idx = map(experiment.conditions_list.index, [control, experimental])
        control_samples, experimental_samples = map(experiment.individual_samples_list.__getitem__, [control_idx, experimental_idx])

        class_abundance_df = PathwayViz.calculate_total_class_abundance(df, full_samples_list)
        return PathwayViz.add_fold_change_column(class_abundance_df, control, experimental, control_samples, experimental_samples)
    
    @staticmethod
    @st.cache_data
    def create_pathway_viz(class_fold_change_df, class_saturation_ratio_df, control, experimental):
        """
        Constructs and returns a complete lipid pathway visualization plot along with its corresponding data dictionary. 
        This method integrates various components of the visualization such as circles representing lipid classes, 
        connecting lines, and textual annotations. It utilizes data on fold changes and saturation ratios to determine
        the visual characteristics (like color and size) of elements in the plot.
    
        Args:
            class_fold_change_df (DataFrame): DataFrame containing fold change data for lipid classes between 
                                              control and experimental conditions.
            class_saturation_ratio_df (DataFrame): DataFrame containing saturation ratio data for lipid classes.
            control (str): Name of the control condition.
            experimental (str): Name of the experimental condition.
    
        Returns:
            tuple: A tuple containing the matplotlib figure object of the plot and a dictionary with pathway data. 
                   The dictionary includes classes, their abundance ratios (fold change), and saturation ratios.
        """
        fig, ax = PathwayViz.initiate_plot()
        PathwayViz.draw_all_circles(ax)
        PathwayViz.draw_connecting_lines(ax)
        PathwayViz.add_text(ax)
        pathway_dict = PathwayViz.create_pathway_dictionary(class_fold_change_df, class_saturation_ratio_df, control, experimental)
        color_contour, size = PathwayViz.prep_plot_inputs(pathway_dict)
        PathwayViz.render_plot(ax, color_contour, size, fig)
        return fig, pathway_dict

    @staticmethod
    def initiate_plot():
        """
        Initiates the lipid pathway visualization plot.

        Returns:
           tuple: A tuple containing the initialized matplotlib figure and axis.
        """
        plt.rcParams["figure.figsize"] = [10, 10]
        fig, ax = plt.subplots()
        
        # Set plot background color
        fig.set_facecolor('white')
        ax.set_facecolor('white')
    
        # Set title and limits
        ax.set_title('Lipid Pathway Visualization', fontsize=20)
        ax.set_xlim([-25, 25])
        ax.set_ylim([-20, 30])
    
        # Hide axis
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    
        # Set aspect and add a black box (frame) around the plot area
        plt.gca().set_aspect('equal', adjustable='box')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
    
        return fig, ax
    
    @staticmethod
    def draw_one_circle(ax, radius, x0, y0, color):
        """
        Draws a single circle on a given axis.

        Args:
            ax: Matplotlib axis to draw on.
            radius (float): Radius of the circle.
            x0 (float): X-coordinate of the circle center.
            y0 (float): Y-coordinate of the circle center.
            color (str): Color of the circle.
        """
        ax.add_patch(plt.Circle((x0, y0), radius, color=color, fill=False))
        
    @staticmethod
    def draw_all_circles(ax):
        """
        Draws all circles representing lipid classes on a given axis.

        Args:
            ax: Matplotlib axis to draw on.
        """
        # Define coordinates and radius for each circle
        circle_data = [
            (5, 0, 0, 'b'),
            (2.5, -7.5 * math.cos(math.pi/6), -7.5 * math.cos(math.pi/3), 'b'),
            (2.5, 7.5 * math.cos(math.pi/6), -7.5 * math.cos(math.pi/3), 'b'),
            (2.5, 10 + 2.5 * math.cos(math.pi/4), 15 + 2.5 * math.sin(math.pi/4), 'b'),
            (2.5, 12.5, 10, 'b'),
            (2.5, 10 + 2.5 * math.cos(math.pi/4), 5 - 2.5 * math.sin(math.pi/4), 'b'),
            # Continue with small black circles
            (0.5, 0, 0, 'black'),
            (0.5, 0, 5, 'black'),
            (0.5, 0, 10, 'black'),
            (0.5, -10, 10, 'black'),
            (0.5, -10, 5, 'black'),
            (0.5, 5 * math.cos(math.pi/6), -5 * math.sin(math.pi/6), 'black'),
            (0.5, 10 * math.cos(math.pi/6), -10 * math.sin(math.pi/6), 'black'),
            (0.5, -5 * math.cos(math.pi/6), -5 * math.sin(math.pi/6), 'black'),
            (0.5, -10 * math.cos(math.pi/6), -10 * math.sin(math.pi/6), 'black'),
            (0.5, 10, 15, 'black'),
            (0.5, 10 + 5 * math.cos(math.pi/4), 15 + 5 * math.sin(math.pi/4), 'black'),
            (0.5, 10, 10, 'black'),
            (0.5, 15, 10, 'black'),
            (0.5, 10, 5, 'black'),
            (0.5, 10 + 5 * math.cos(math.pi/4), 5 - 5 * math.sin(math.pi/4), 'black')
            # Ensure to cover all necessary circles...
        ]
    
        # Draw each circle on the axis
        for radius, x0, y0, color in circle_data:
            PathwayViz.draw_one_circle(ax, radius, x0, y0, color)
            
    @staticmethod
    def draw_connecting_lines(ax):
        """
        Draws connecting lines between lipid classes on a given axis.

        Args:
            ax: Matplotlib axis to draw on.
        """
        # Define start and end points for each line as per the old code
        line_data = [
            ([0, 0], [0, 20], 'b'),  # Line from G3P
            ([0, -5], [15, 20], 'b'),  # Line towards Fatty Acids
            ([-5, -10], [20, 15], 'b'),  # Line from Fatty Acids to LPA
            ([-10, -10], [15, 5], 'b'),  # Line from LPA to DAG
            ([0, 5], [10, 10], 'b'),  # Line from PA to CDP-DAG
            ([5, 10], [10, 15], 'b'),  # Line from CDP-DAG to PI
            ([5, 10], [10, 10], 'b'),  # Line from CDP-DAG to PG
            ([5, 10], [10, 5], 'b'),  # Line from CDP-DAG to PS
            ([5 * math.cos(math.pi/6), 10], [-5 * math.sin(math.pi/6), 5], 'b'),  # Line from PA to PE
        ]
    
        # Draw each line on the axis
        for x, y, color in line_data:
            ax.plot(x, y, c=color)
            
    @staticmethod
    def add_text(ax):
        """
        Adds text annotations to the lipid pathway visualization.

        Args:
            ax: Matplotlib axis to annotate.
        """
        ax.annotate('G3P', xy=(0, 20), xytext=(0, 23),arrowprops=dict(facecolor='black'), fontsize=15)
        ax.annotate('Fatty Acids', xy=(-5, 20), xytext=(-5, 25),arrowprops=dict(facecolor='black'), fontsize=15)
        ax.text(-3.5, 14, 'LPA', fontsize=15)
        ax.text(-2.5, 9.5, 'PA', fontsize=15)
        ax.text(-4, 5.5, 'DAG', fontsize=15)
        ax.text(-4, 0.5, 'TAG', fontsize=15)
        ax.text(-4, -2, 'PC', fontsize=15)
        ax.text(-12, -6.5, 'LPC', fontsize=15)
        ax.text(2.5, -2, 'PE', fontsize=15)
        ax.text(9, -6, 'LPE', fontsize=15)
        ax.text(-14, 15, 'LCBs', fontsize=15)
        ax.text(-13.5, 10, 'Cer', fontsize=15)
        ax.text(-13, 5, 'SM', fontsize=15)
        ax.text(2, 11, 'CDP-DAG', fontsize=15)
        ax.text(10.5, 15.5, 'PI', fontsize=15)
        ax.text(14, 19, 'LPI', fontsize=15)
        ax.text(10.5, 9.5, 'PG', fontsize=15)
        ax.text(15.5, 9.5, 'LPG', fontsize=15)
        ax.text(10.5, 4, 'PS', fontsize=15)
        ax.text(14.5, 1, 'LPS', fontsize=15)
        return 
        
    @staticmethod
    @st.cache_data
    def create_pathway_dictionary(class_fold_change_df, class_saturation_ratio_df, control, experimental):
        """
        Creates a dictionary for lipid pathway visualization with classes, abundance ratios, and saturation ratios.

        Args:
            class_fold_change_df (DataFrame): DataFrame with fold change data.
            class_saturation_ratio_df (DataFrame): DataFrame with saturation ratio data.
            control (str): Control condition name.
            experimental (str): Experimental condition name.

        Returns:
            dict: Dictionary with lipid classes and their corresponding abundance and saturation ratios.
        """
        pathway_classes_list = ['TG', 'DG', 'PA', 'LPA', 'LCB', 'Cer', 'SM', 'PE', 'LPE', 'PC', 'LPC', 'PI', 'LPI', 'CDP-DAG', 'PG', 'LPG', 'PS', 'LPS']
        fc_column = f'fc_{experimental}_{control}'
    
        # Ensure dataframes are indexed by ClassKey
        fc_values = class_fold_change_df.set_index('ClassKey')[fc_column] if 'ClassKey' in class_fold_change_df.columns else class_fold_change_df[fc_column]
        saturation_ratios = class_saturation_ratio_df.set_index('ClassKey')['saturation_ratio'] if 'ClassKey' in class_saturation_ratio_df.columns else class_saturation_ratio_df['saturation_ratio']
    
        # Use dictionary comprehension for streamlined construction
        pathway_dict = {
            'class': pathway_classes_list,
            'abundance ratio': [fc_values.get(lipid_class, 0) for lipid_class in pathway_classes_list],
            'saturated fatty acids ratio': [saturation_ratios.get(lipid_class, 0) for lipid_class in pathway_classes_list]
        }
    
        return pathway_dict

    @staticmethod
    def prep_plot_inputs(pathway_dict):
        """
        Prepares color and size data for plotting the lipid pathway visualization.

        Args:
            pathway_dict (dict): Dictionary with pathway data.

        Returns:
            tuple: Color and size data for the plot.
        """
        color_contour = pathway_dict['saturated fatty acids ratio']
        size = [50 * ele**2 for ele in pathway_dict['abundance ratio']]
        return color_contour, size

    @staticmethod
    def render_plot(ax, color_contour, size, fig):
        """
        Renders the lipid pathway visualization plot.

        Args:
            ax: Matplotlib axis to render on.
            color_contour (list): List of colors for each data point.
            size (list): List of sizes for each data point.
            fig: Matplotlib figure object.
        """
        # Define the points' coordinates for the plot
        points_x = [0, 0, 0, 0, -10, -10, -10, 5*math.cos(math.pi/6), 10*math.cos(math.pi/6), \
                        -5*math.cos(math.pi/6), -10*math.cos(math.pi/6), 10, 10+5*math.cos(math.pi/4), \
                        5, 10, 15, 10, 10+5*math.cos(math.pi/4)]
        points_y = [0, 5, 10, 15, 15, 10, 5, -5*math.sin(math.pi/6), -10*math.sin(math.pi/6), \
                        -5*math.sin(math.pi/6), -10*math.sin(math.pi/6), 15, 15+5*math.sin(math.pi/4), \
                        10, 10, 10, 5, 5-5*math.sin(math.pi/4)]

        # Scatter plot with color and size based on calculated values
        points = ax.scatter(points_x, points_y, c=color_contour, s=size, cmap="plasma")

        # Add a color bar to the plot
        cbar = fig.colorbar(points)
        cbar.set_label(label='Saturation Ratio', size=15)
        cbar.ax.tick_params(labelsize=15)
