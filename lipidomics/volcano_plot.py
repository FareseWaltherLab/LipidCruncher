import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

class VolcanoPlot:
    
    def __init__(self, an_experiment, a_df, a_control_condition, an_experimental_condition, a_p_val, a_q_val):
        self.experiment = an_experiment
        self.df = a_df
        self.control = a_control_condition
        self.experimental = an_experimental_condition
        self.p_value_threshold = a_p_val
        self.q_value_threshold = a_q_val
        
    @st.cache_data
    def convert_df(_self, a_dataframe):
         return a_dataframe.to_csv().encode('utf-8')
        
    def add_fold_change_and_p_value_columns(self):
        
        @st.cache_data
        def fold_change_calculator(nums_1, nums_2): 
            if np.mean(nums_1) > 0 and np.mean(nums_2) > 0: 
                return np.log2(np.mean(nums_1)/np.mean(nums_2))
            else:
                return 0
        
        @st.cache_data
        def p_value_calculator(nums_1, nums_2):
            t_value, p_value = stats.ttest_ind(nums_1,nums_2)
            return p_value
        
        def get_index():
            return self.experiment.conditions_list.index(self.control),\
                   self.experiment.conditions_list.index(self.experimental)
            
        control_idx, experimental_idx = get_index()
        
        def get_individual_samples_list():
            return self.experiment.individual_samples_list[control_idx], \
                   self.experiment.individual_samples_list[experimental_idx]
        
        control_samples_list, experimental_samples_list = get_individual_samples_list()
        
        @st.cache_data
        def add_fold_change_column(a_control_samples_list, a_control, an_experimental, an_experimental_samples_list, a_dataframe):
            a_dataframe['fc_' + an_experimental + '_' + a_control] = \
            a_dataframe[['MeanArea[' + sample + ']'  for sample in a_control_samples_list + an_experimental_samples_list]]\
            .apply(lambda x: fold_change_calculator(x[['MeanArea[' + sample + ']'  for sample in an_experimental_samples_list]], \
                                                    x[['MeanArea[' + sample + ']'  for sample in a_control_samples_list]]), axis=1)
            return a_dataframe
        
        @st.cache_data
        def add_p_val_column(a_control_samples_list, a_control, an_experimental, an_experimental_samples_list, a_dataframe):
            a_dataframe['p_val_' + an_experimental + '_' + a_control] = \
            a_dataframe[['MeanArea[' + sample + ']'  for sample in a_control_samples_list + an_experimental_samples_list]]\
            .apply(lambda x: p_value_calculator(x[['MeanArea[' + sample + ']'  for sample in an_experimental_samples_list]], \
                                              x[['MeanArea[' + sample + ']'  for sample in a_control_samples_list]]), axis=1)
            return a_dataframe
        
        a_volcano_df = add_fold_change_column(control_samples_list, self.control, self.experimental, experimental_samples_list, self.df.copy(deep=True))
        a_volcano_df = add_p_val_column(control_samples_list, self.control, self.experimental, experimental_samples_list, a_volcano_df.copy(deep=True))
        
        return a_volcano_df
    
    def create_volcano_plot(self, a_volcano_df, a_selected_classes_list):
        
        a_volcano_df['fc_' + self.experimental + '_' + self.control].fillna(value=0, inplace=True)
        a_volcano_df['p_val_' + self.experimental + '_' + self.control].fillna(value=0, inplace=True)
        
        fold_change_list = a_volcano_df['fc_' + self.experimental + '_' + self.control].values
        
        @st.cache_data
        def find_fold_change_axis_limit(a_fold_change_list):
            if len([x for x in a_fold_change_list if x != 0]) > 0:
                max_fold_change = np.max([x for x in a_fold_change_list if x != 0])
                min_fold_change = np.min([x for x in a_fold_change_list if x != 0])
                return np.ceil(max(abs(max_fold_change), abs(min_fold_change)))
            else:
                return 1
        
        fold_change_axis_limit = find_fold_change_axis_limit(fold_change_list)
        
        q_values_list = a_volcano_df['p_val_' + self.experimental + '_' + self.control].values
        
        @st.cache_data
        def find_q_value_axis_limit(a_q_values_list):
            if len([x for x in a_q_values_list if x != 0]) > 0:
                return -np.log10(np.min([x for x in a_q_values_list if x != 0]))
            else:
                return 1
        
        q_value_axis_limit = find_q_value_axis_limit(q_values_list)
        
        @st.cache_data
        def build_plot_df(a_volcano_dataframe, a_selected_classes_lst, an_experimental, a_control):
            unique_color_lst =[ 'red', 'blue', 'green', 'magenta', 'cyan', 'orange', 'black', 'pink', 'brown', 'yellow', 'purple', \
                                'gray', 'olive', 'chocolate', 'silver', 'darkred', 'khaki', 'skyblue', 'navy', 'orchid', 'deepskyblue', \
                                'tan', 'lime', 'tomato', 'deeppink', 'lavender', 'salmon', 'darkkhaki', 'lavenderblush', 'indigo']
            fc = []
            pval = []
            class_lst = []
            color_lst = []
            species = []
            for lipid_class in a_selected_classes_lst:
                fc = fc + a_volcano_dataframe[a_volcano_dataframe['ClassKey'] == lipid_class]['fc_' + an_experimental + '_' + a_control].values.tolist()
                pval = pval + a_volcano_dataframe[a_volcano_dataframe['ClassKey'] == lipid_class]['p_val_' + an_experimental + '_' + a_control].values.tolist()
                color_lst = color_lst + \
                    [unique_color_lst[a_selected_classes_lst.index(lipid_class)] for i in range(len(a_volcano_dataframe[a_volcano_dataframe['ClassKey'] == lipid_class]))]
                class_lst = class_lst + [lipid_class for i in range(len(a_volcano_dataframe[a_volcano_dataframe['ClassKey'] == lipid_class]))]
                species = species + a_volcano_dataframe[a_volcano_dataframe['ClassKey'] == lipid_class]['LipidMolec'].values.tolist()
            a_plot_df = pd.DataFrame({"FC": fc, "qvalue": -np.log10(pval), "Species": species, "Class": class_lst, "Color": color_lst})
            return a_plot_df
        
        plot_df = build_plot_df(a_volcano_df, a_selected_classes_list, self.experimental, self.control)
        
        def render_plot(a_plot_df):
            a_plot = figure(title='Volcano Plot', x_axis_label='Fold Change (' + self.experimental + '/' + self.control + ')', y_axis_label='q-value')
            src = ColumnDataSource(a_plot_df)
            a_plot.scatter(x="FC", y="qvalue", legend_group='Class', color='Color', name='volcano', size = 4, source=src)
            a_plot.line(x=[i for i in range(int(-fold_change_axis_limit)-1, int(fold_change_axis_limit)+1)], \
                        y=-np.log10(self.p_value_threshold), line_dash = 'dashed', color='black') # horizontal black line
            a_plot.line(x=-1, y=[i for i in range(0, int(q_value_axis_limit)+1)], line_dash = 'dashed', color='black') # vertical black line
            a_plot.line(x=1, y=[i for i in range(0, int(q_value_axis_limit)+1)], line_dash = 'dashed', color='black') # vertical black line
            hover = HoverTool(tooltips = [('FC', '@FC'), ('q-value', '@qvalue'), ('Species', '@Species')], names=['volcano'])
            a_plot.add_tools(hover)
            a_plot.title.text_font_size = "15pt"
            a_plot.xaxis.axis_label_text_font_size = "15pt"
            a_plot.yaxis.axis_label_text_font_size = "15pt"
            a_plot.xaxis.major_label_text_font_size = "15pt"
            a_plot.yaxis.major_label_text_font_size = "15pt"
            return a_plot
        
        plot = render_plot(plot_df)
        
        def display_plot(a_plot, a_plot_df):
            st.bokeh_chart(a_plot)
            csv_download = self.convert_df(a_plot_df)
            st.download_button(
                                label="Download Data", 
                                data=csv_download,
                                file_name='volcano_plot.csv',
                                mime='text/csv')
            return 
        
        display_plot(plot, plot_df)
        return 