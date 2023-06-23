import streamlit as st
import pandas as pd
from sklearn.preprocessing import scale # Data scaling
from sklearn import decomposition # PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from numpy import inf

class PCA:
    
    def __init__(self, a_df, an_experiment):
        self.df = a_df
        self.experiment = an_experiment 
        
    @st.cache_data
    def convert_df(_self, a_dataframe):
         return a_dataframe.to_csv().encode('utf-8')
     
    @st.cache_data
    def run_pca(_self, a_mean_area_df):
        Y = a_mean_area_df.T
        #Y = _self.df[['MeanArea[' + sample + ']' for sample in _self.experiment.full_samples_list]].T
        Y = scale(Y)
        pca = decomposition.PCA(n_components=2)
        pca.fit(Y)
        scores = pca.transform(Y)
        explained_variance = pca.explained_variance_ratio_
        return scores, ['PC'+str(i+1)+' ('+str("{:.0f}".format(explained_variance[i]*100))+'%)' for i in range(2)] # e.g. [PC1 (80%), PC2 (15%)]
    
    def plot_pca(self):
        
        mean_area_df = self.df[['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]]
        scores, PC_list = self.run_pca(mean_area_df)
        PCx = scores[:, 0].tolist()
        PCy = scores[:, 1].tolist()
        
        def create_plot_inputs():
            color_lst =[ 'red', 'blue', 'green', 'magenta', 'cyan', 'orange', 'black', 'pink', 'brown', 'yellow', 'purple', \
                        'gray', 'olive', 'chocolate', 'silver', 'darkred', 'khaki', 'skyblue', 'navy', 'orchid']
            a_legend = []
            a_color = []
            for index, condition in enumerate(self.experiment.conditions_list):
                a_legend = a_legend + [condition for i in range(self.experiment.number_of_samples_list[index])]
                a_color = a_color + [color_lst[index] for i in range(self.experiment.number_of_samples_list[index])]
            return a_legend, a_color
        
        legend, color = create_plot_inputs()
        
        def render_plot(a_PCx, a_PCy, a_PC_list):
            a_pca_df = pd.DataFrame({"PC1": a_PCx, "PC2": a_PCy, "sample": self.experiment.full_samples_list, "legend": legend, "color": color})
            src = ColumnDataSource(a_pca_df)
            a_plot = figure(title='PCA', x_axis_label=a_PC_list[0], y_axis_label=a_PC_list[1])
            a_plot.scatter(x="PC1", y="PC2", legend_group='legend', color='color', source=src)
            hover = HoverTool(tooltips = [('PC1', '@PC1'), ('PC2', '@PC2'), ('sample', '@sample')])
            a_plot.add_tools(hover)
            a_plot.title.text_font_size = "15pt"
            a_plot.xaxis.axis_label_text_font_size = "15pt"
            a_plot.yaxis.axis_label_text_font_size = "15pt"
            a_plot.xaxis.major_label_text_font_size = "15pt"
            a_plot.yaxis.major_label_text_font_size = "15pt"
            return a_pca_df, a_plot 
        
        pca_df, plot = render_plot(PCx, PCy, PC_list)
        
        def display_plot(a_plot, a_pca_df):
            st.bokeh_chart(a_plot)
            csv_download = self.convert_df(a_pca_df[['PC1', 'PC2', 'sample', 'legend']])
            st.download_button(
                        label="Download Data",
                        data=csv_download,
                        file_name='PCA.csv',
                        mime='text/csv')
            return
        
        display_plot(plot, pca_df)
        
        return