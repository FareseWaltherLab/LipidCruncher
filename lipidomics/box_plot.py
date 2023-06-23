import streamlit as st
import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import numpy as np 

class BoxPlot:
    
    def __init__(self, an_experiment, a_df):
        self.experiment = an_experiment 
        self.df = a_df 
        
    @st.cache_data
    def create_mean_area_df(_self, a_df, a_full_samples_list):
        return a_df[['MeanArea[' + sample + ']' for sample in a_full_samples_list]] # picks the 'MeanArea[s1]', ..., 'MeanArea[sN]' columns only
    
    @st.cache_data
    def calculate_missing_values_percentage(_self, a_mean_area_df, a_full_samples_list):
        return [len(a_mean_area_df[a_mean_area_df['MeanArea[' + sample + ']'] == 0]) / len(a_mean_area_df) * 100 for sample in a_full_samples_list]
        
    @st.cache_data
    def plot_missing_values(_self, a_full_samples_list, a_zero_values_percent_list):
        plt.rcdefaults()
        fig, ax = plt.subplots()
        y = a_full_samples_list
        ax.barh(y, a_zero_values_percent_list)
        ax.set_xlabel('Percentage of Missing Values')
        ax.set_ylabel('Sample')
        ax.set_title('Missing Values Distribution')
        st.pyplot(fig)
        return 
    
    @st.cache_data
    def plot_box_plot(_self, a_mean_area_df, a_full_samples_list):
        x = []
        for sample in a_full_samples_list:
            x.append(list(np.log10(a_mean_area_df[a_mean_area_df['MeanArea[' + sample + ']'] > 0]['MeanArea[' + sample + ']'].values)))
        
        plt.rcdefaults()
        fig, ax = plt.subplots()
        x_ticks = np.arange(len(a_full_samples_list))
        plt.boxplot(x)
        ax.set_xlabel('Sample')
        ax.set_ylabel('log10(AUC)')
        ax.set_title('Box Plot of Non-Zero Values')
        ax.set_xticklabels(a_full_samples_list, rotation=90)
        st.pyplot(fig)
        return 
    
        