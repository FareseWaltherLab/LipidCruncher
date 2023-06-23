import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

class Correlation:
    
    def __init__(self, a_df, an_experiment, a_condition, a_sample_type, an_index):
        self.df = a_df
        self.experiment = an_experiment
        self.condition = a_condition
        self.sample_type = a_sample_type
        self.index = an_index
        
    # function for downloading data
    @st.cache_data
    def convert_df(_self, some_df):
         return some_df.to_csv().encode('utf-8')
        
    def correlation_plot(self):
        mean_area_df = self.df[['MeanArea[' + sample + ']' for sample in self.experiment.individual_samples_list[self.index]]]
        
        # re-naming the columns from MainArea[s1] to s1 etc
        @st.cache_data
        def rename_columns(a_mean_area_df):
            counter = 0
            for column in a_mean_area_df.columns:
                a_mean_area_df.rename(columns={column: self.experiment.individual_samples_list[self.index][counter]}, inplace=True)
                counter = counter + 1
            return a_mean_area_df
        
        mean_area_df = rename_columns(mean_area_df)
        
        # set the min and the center of the color bar
        def set_color_bar():
            if self.sample_type == 'biological replicates':
                v_min = 0.5
                thresh = 0.8
            elif self.sample_type == 'Technical replicates':
                v_min = 0.75
                thresh = 0.9  
            return v_min, thresh
        
        v_min, thresh = set_color_bar()
        
        @st.cache_data
        def compute_correlation_df(a_mean_area_df):
            return a_mean_area_df.corr()
        
        def render_and_display_plot(a_mean_area_df, a_v_min, a_thresh):
            fig = plt.figure(figsize=(20, 16))
            mask = np.triu(np.ones_like(compute_correlation_df(a_mean_area_df), dtype=np.bool))
            sns.set(font_scale=3)
            heatmap = sns.heatmap(compute_correlation_df(a_mean_area_df), mask=mask, vmin=a_v_min, vmax=1, center=a_thresh, annot=False, cmap='RdBu', square=False, cbar=True)
            heatmap.set_title('Triangle Correlation Heatmap - ' + self.condition, fontdict={'fontsize':40})
            return st.pyplot(fig)
        
        render_and_display_plot(mean_area_df, v_min, thresh)
        
        def display_correlation_table(a_mean_area_df):
            st.write('---------------------------------------------------------------------------------------------------')
            st.write('Find the exact correlation coefficients in the table below:')
            st.write(compute_correlation_df(a_mean_area_df))
            csv_download = self.convert_df(compute_correlation_df(a_mean_area_df))
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='Correlation_Matrix_'+str(self.condition)+'.csv',
                            mime='text/csv')
            return 
        
        display_correlation_table(mean_area_df)
        
        return 