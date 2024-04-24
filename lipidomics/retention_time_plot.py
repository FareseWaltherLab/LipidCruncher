import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
import itertools
from bokeh.palettes import Category20 as palette
import streamlit as st

class RetentionTime:
    """
    A class dedicated to creating and handling retention time plots for lipidomic data.
    """

    @staticmethod
    @st.cache_data
    def prep_single_plot_inputs(df, lipid_class):
        """
        Prepare the input data for a single retention time plot.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            lipid_class (str): The lipid class for which the plot will be generated.

        Returns:
            tuple: Arrays of base retention time, calculated mass, and lipid molecule names.
        """
        class_df = df[df['ClassKey'] == lipid_class]
        return class_df['BaseRt'].values, class_df['CalcMass'].values, class_df['LipidMolec'].values

    @staticmethod
    def render_single_plot(retention_df, lipid_class):
        """
        Create a single retention time plot for a given lipid class.

        Args:
            retention_df (pd.DataFrame): DataFrame with data for the plot.
            lipid_class (str): The lipid class for which the plot will be generated.

        Returns:
            bokeh.plotting.figure: A Bokeh plot figure.
        """
        src = ColumnDataSource(retention_df)
        plot = figure(title=lipid_class, x_axis_label='Calculated Mass', y_axis_label='Retention Time (mins)')
        plot.scatter(x="Mass", y="Retention", source=src)
        hover = HoverTool(tooltips=[('Mass', '@Mass'), ('Retention_time', '@Retention'), ('Species', '@Species')])
        plot.add_tools(hover)
        plot.title.text_font_size = '15pt'
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
        return plot

    @staticmethod
    @st.cache_data
    def prep_multi_plot_input(df, selected_classes_list, unique_color_list):
        """
        Prepare the input data for a multi-class retention time comparison plot.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            selected_classes_list (list): List of lipid classes to be included in the plot.
            unique_color_list (list): List of colors assigned to each lipid class.

        Returns:
            tuple: Arrays of retention times, masses, class labels, and corresponding colors.
        """
        retention, mass, class_lst, color_lst = [], [], [], []
        for lipid_class in selected_classes_list:
            current_class_df = df[df['ClassKey'] == lipid_class]
            retention.extend(current_class_df['BaseRt'].values.tolist())
            mass.extend(current_class_df['CalcMass'].values.tolist())
            class_lst.extend(current_class_df['ClassKey'].values.tolist())
            color_lst.extend([unique_color_list[selected_classes_list.index(lipid_class)] for _ in range(len(current_class_df))])
        return retention, mass, class_lst, color_lst

    @staticmethod
    def render_multi_plot(retention_df):
        """
        Create a multi-class retention time comparison plot.

        Args:
            retention_df (pd.DataFrame): DataFrame with data for the plot.

        Returns:
            bokeh.plotting.figure: A Bokeh plot figure for comparing multiple classes.
        """
        src = ColumnDataSource(retention_df)
        plot = figure(title='Retention Time vs. Mass - Comparison Mode', x_axis_label='Calculated Mass', y_axis_label='Retention Time (mins)')
        plot.scatter(x="Mass", y="Retention", legend_group='Class', color='Color', source=src)
        plot.title.text_font_size = '15pt'
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
        return plot

    @staticmethod
    def plot_single_retention(df):
        """
        Generate individual retention time plots for each lipid class in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.

        Returns:
            list of tuples: Each tuple contains a plot, its DataFrame, filename, and svg string.
        """
        plots = []
        for lipid_class in df['ClassKey'].value_counts().index:
            retention, mass, species = RetentionTime.prep_single_plot_inputs(df, lipid_class)
            retention_df = pd.DataFrame({"Mass": mass, "Retention": retention, "Species": species})
            plot = RetentionTime.render_single_plot(retention_df, lipid_class)
            plots.append((plot, retention_df))
        return plots

    @staticmethod
    def plot_multi_retention(df, selected_classes_list):
        """
        Generate a retention time plot comparing multiple lipid classes.

        Args:
            df (pd.DataFrame): DataFrame containing lipidomics data.
            selected_classes_list (list): List of lipid classes to include in the plot.

        Returns:
            tuple: A plot, its DataFrame, filename, and svg string.
        """
        all_lipid_classes_lst = df['ClassKey'].value_counts().index.tolist()
        unique_color_lst = RetentionTime.get_unique_colors(len(all_lipid_classes_lst))
    
        retention, mass, class_lst, color_lst = RetentionTime.prep_multi_plot_input(df, selected_classes_list, unique_color_lst)
        retention_df = pd.DataFrame({"Mass": mass, "Retention": retention, "Class": class_lst, "Color": color_lst})
        plot = RetentionTime.render_multi_plot(retention_df)
        return plot, retention_df
    
    @staticmethod
    def get_unique_colors(n_classes):
        """
        Generate a list of unique color hex codes for a given number of classes using a built-in Bokeh palette.
        If the number of classes exceeds the palette size, colors will cycle through the palette.
    
        Args:
            n_classes (int): Number of unique colors required.
    
        Returns:
            list: List of color hex codes from the selected Bokeh palette.
        """
        colors = itertools.cycle(palette[20])
        return [next(colors) for _ in range(n_classes)]
