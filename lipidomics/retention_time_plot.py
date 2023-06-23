import streamlit as st 
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

class RetentionTime:
    
    def __init__(self, a_df, an_experiment):
        self.df = a_df
        self.experiment = an_experiment
        
   # function for downloading data
    @st.cache_data
    def convert_df(_self, some_df):
        return some_df.to_csv().encode('utf-8')
        
    @st.cache_data
    def prep_single_plot_inputs(_self, a_lipid_class):
        return _self.df[_self.df['ClassKey'] == a_lipid_class]['BaseRt'].values, _self.df[_self.df['ClassKey'] == a_lipid_class]['CalcMass'].values, \
               _self.df[_self.df['ClassKey'] == a_lipid_class]['LipidMolec'].values
               
    def render_single_plot(self, a_retention_df, a_lipid_class):
        src = ColumnDataSource(a_retention_df)
        plot = figure(title=a_lipid_class, x_axis_label='Calculated Mass', y_axis_label='Retention Time (mins)')
        plot.scatter(x="Mass", y="Retention", source=src)
        hover = HoverTool(tooltips = [('Mass', '@Mass'), ('Retention_time', '@Retention'), ('Species', '@Species')]) # hover tool
        plot.add_tools(hover)
        plot.title.text_font_size = '15pt'
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
        return plot 
    
    def display_plot(self, a_plot, a_retention_df, a_file_name):
        st.bokeh_chart(a_plot)
        csv_download = self.convert_df(a_retention_df)
        st.download_button(
            label="Download Data",
            data=csv_download,
            file_name=a_file_name+'.csv',
            mime='text/csv')
        st.write("----------------------------------------------------------------------------------------------------")
        return 
    
    @st.cache_data
    def prep_multi_plot_input(_self, a_selected_classes_list, a_unique_color_list):
        retention = []
        mass = []
        class_lst = []
        color_lst = []
        for lipid_class in a_selected_classes_list:
            retention = retention + _self.df[_self.df['ClassKey'] == lipid_class]['BaseRt'].values.tolist()
            mass = mass + _self.df[_self.df['ClassKey'] == lipid_class]['CalcMass'].values.tolist()
            class_lst = class_lst + _self.df[_self.df['ClassKey'] == lipid_class]['ClassKey'].values.tolist()
            color_lst = color_lst + [a_unique_color_list[a_selected_classes_list.index(lipid_class)] for i in range(len(_self.df[_self.df['ClassKey'] == lipid_class]))]
        return retention, mass, class_lst, color_lst
    
    def render_multi_plot(self, a_retention_df):
        src = ColumnDataSource(a_retention_df)
        plot = figure(title='Retention Time vs. Mass - Comparison Mode', x_axis_label='Calculated Mass', y_axis_label='Retention Time (mins)')
        plot.scatter(x="Mass", y="Retention", legend_group='Class', color='Color', source=src)
        plot.title.text_font_size = '15pt'
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
        return plot 
        
    def plot_single_retention(self):
        for lipid_class in self.df['ClassKey'].value_counts().index:
            retention, mass, species = self.prep_single_plot_inputs(lipid_class)
            retention_df = pd.DataFrame({"Mass": mass, "Retention": retention, "Species": species})
            plot = self.render_single_plot(retention_df, lipid_class)
            self.display_plot(plot, retention_df, 'retention_plot_' + lipid_class)
        return
    
    def plot_multi_retention(self):
        unique_color_lst =[ 'red', 'blue', 'green', 'magenta', 'cyan', 'orange', 'black', 'pink', 'brown', 'yellow', 'purple', \
                            'gray', 'olive', 'chocolate', 'silver', 'darkred', 'khaki', 'skyblue', 'navy', 'orchid']
        all_lipid_classes_lst = self.df['ClassKey'].value_counts().index.tolist()
        selected_classes_list = st.multiselect('Add or remove classes (up to 20 classes):', all_lipid_classes_lst, all_lipid_classes_lst[:1])
        
        if len(selected_classes_list) > 20:
            st.error('You can only compare up to 20 lipid classes at a time!')
            return None
        else:
            retention, mass, class_lst, color_lst = self.prep_multi_plot_input(selected_classes_list, unique_color_lst)
            retention_df = pd.DataFrame({"Mass": mass, "Retention": retention, "Class": class_lst, "Color": color_lst})
            plot = self.render_multi_plot(retention_df)
            self.display_plot(plot, retention_df, 'Retention_Time_Comparison_Mode.csv')
        return