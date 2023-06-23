import streamlit as st 
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

class FilteredData:
    
    def __init__(self, a_df, an_experiment, a_condition, an_index):
        self.df = a_df
        self.experiment = an_experiment 
        self.condition = a_condition
        self.index = an_index
        
    @st.cache_data
    def cov_calculator(_self, numbers): 
         non_zero_lst = [number for number in numbers if (number>0)]
         if len(non_zero_lst) > 1:
             cov = (np.std(non_zero_lst)/np.mean(non_zero_lst))*100
         else:
             cov = None
         return cov
     
    @st.cache_data
    def mean_calculator(_self, numbers): 
         non_zero_lst = [number for number in numbers if (number>0)]
         if len(non_zero_lst) > 1:
             mean = np.mean(non_zero_lst)
         else:
             mean = None
         return mean
     
    @st.cache_data
    def log_transform(_self, number): 
         return np.log10(number)
     
     # function for downloading data
    @st.cache_data
    def convert_df(_self, a_dataframe):
         return a_dataframe.to_csv().encode('utf-8')
     
    @st.cache_data
    def prep_plot_inputs(_self, a_dataframe):
        x = a_dataframe['mean'].values
        y = a_dataframe['cov'].values
        species = a_dataframe['LipidMolec'].values
        return x, y, species
    
    def render_plot(self, x, y, species):  
        plot = figure(title='CoV - All lipid Species', x_axis_label='Mean of log10(AUC) Over All BQC Samples', y_axis_label='CoV(%)')
        cov_df = pd.DataFrame({"Mean_AUC": x, "CoV": y, 'Species': species})
        src = ColumnDataSource(cov_df)
        plot.scatter(x="Mean_AUC", y="CoV", name='cov', source=src)
        hover = HoverTool(tooltips = [('Mean_AUC', '@Mean_AUC'), ('CoV', "@CoV"), ('Species', "@Species")], names=['cov'])
        plot.add_tools(hover)
        plot.title.text_font_size = "15pt"
        plot.xaxis.axis_label_text_font_size = "15pt"
        plot.yaxis.axis_label_text_font_size = "15pt"
        plot.xaxis.major_label_text_font_size = "15pt"
        plot.yaxis.major_label_text_font_size = "15pt"
        return plot
    
    def display_plot(self, plot):
            st.bokeh_chart(plot)
            csv_download = self.convert_df(self.df)
            st.download_button(
                label="Download Data",
                data=csv_download,
                file_name='cov.csv',
                mime='text/csv')
            return 
        
    def prep_df(self, a_dataframe, an_auc_list):
        a_dataframe['cov'] = a_dataframe[an_auc_list].apply(lambda x: self.cov_calculator(x), axis=1)
        a_dataframe['mean'] = a_dataframe[an_auc_list].apply(lambda x: self.mean_calculator(x), axis=1)
        a_dataframe['mean'] = a_dataframe['mean'].apply(lambda x: self.log_transform(x))
        return a_dataframe 
    
    def plot_cov(self):
        BQC_samples_list = self.experiment.individual_samples_list[self.index]
        auc = ['MeanArea[' + sample + ']' for sample in BQC_samples_list]
        prepared_df = self.prep_df(self.df.copy(deep=True), auc)
        x, y, species = self.prep_plot_inputs(prepared_df)
        plot = self.render_plot(x, y, species)
        self.display_plot(plot)
        reliable_data_percent = round(len(prepared_df[prepared_df['cov'] < 30]) / len(prepared_df) * 100, 1)
        if reliable_data_percent >= 80:
            st.info(str(reliable_data_percent) + "% of the datapoints are highly reliable (CoV < 30%).")
        elif reliable_data_percent < 80 and reliable_data_percent >= 50:
            st.warning(str(reliable_data_percent) + "% of the datapoints are highly reliable (CoV < 30%).")
        else:
            st.error(str(reliable_data_percent) + "% of the datapoints are highly reliable (CoV < 30%).")
            
        return prepared_df
    
    @st.cache_data
    def filter_df(_self, a_thresh, a_prepared_df):
        a_prepared_df = a_prepared_df.loc[a_prepared_df['cov'] <= a_thresh] # removes datapoints with CoV > thresh
        a_prepared_df.drop(['mean', 'cov'], axis=1, inplace=True)
        return a_prepared_df
    
    