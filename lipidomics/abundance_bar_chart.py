import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt

class AbundanceBarChart:
    
    def __init__(self, an_experiment, a_df):
        self.experiment = an_experiment
        self.df = a_df
        
    @st.cache_data
    def convert_df(_self, a_dataframe):
         return a_dataframe.to_csv().encode('utf-8')
     
    @st.cache_data
    def create_mean_std_columns(_self, a_df, a_full_samples_list, a_conditions_list, an_individual_samples_list):
        grouped_df = a_df.groupby('ClassKey')[["MeanArea[" + sample + ']' for sample in a_full_samples_list]].sum()
        for index, condition in enumerate(a_conditions_list):
            grouped_df["mean_AUC_" + condition] = grouped_df[["MeanArea[" + sample + "]" for sample in an_individual_samples_list[index]]]\
                .apply(lambda x: np.mean(x), axis = 1)
            grouped_df["std_AUC_" + condition] = grouped_df[["MeanArea[" + sample + "]" for sample in an_individual_samples_list[index]]]\
                .apply(lambda x: np.std(x), axis = 1)
            grouped_df["log2_mean_AUC_" + condition] = np.log2(grouped_df["mean_AUC_" + condition].values)
            grouped_df["log2_std_AUC_" + condition] = (np.log2(grouped_df["mean_AUC_" + condition].values + grouped_df["std_AUC_" + condition].values) - \
                                                       np.log2(grouped_df["mean_AUC_" + condition].values - grouped_df["std_AUC_" + condition].values)) / 2
        return grouped_df
    
    def create_abundance_bar_chart(self, a_selected_conditions_list, a_selected_classes_list, a_mode):
        
        def prep_plot_df(a_df, a_full_samples_list, a_conditions_list, an_individual_samples_list):
            a_temp = self.create_mean_std_columns(a_df, a_full_samples_list, a_conditions_list, an_individual_samples_list).copy(deep=True)
            a_temp.reset_index(inplace=True)
            a_temp = a_temp.loc[a_temp['ClassKey'].isin(a_selected_classes_list)]
            return a_temp 
        
        df = self.df
        conditions_list = self.experiment.conditions_list
        full_samples_list = self.experiment.full_samples_list
        individual_samples_list = self.experiment.individual_samples_list
        abundance_df = prep_plot_df(df, full_samples_list, conditions_list, individual_samples_list)
        
        def render_plot():
            plt.rcdefaults()
            fig, ax = plt.subplots()
            y = np.arange(len(a_selected_classes_list))
            width = 1/(len(a_selected_conditions_list)+1)
            multiplier = 0
            for condition in a_selected_conditions_list:
                if a_mode == 'linear scale':
                    mean = abundance_df["mean_AUC_" + condition]
                    std = abundance_df["std_AUC_" + condition]
                elif a_mode == 'log2 scale':
                    mean = abundance_df["log2_mean_AUC_" + condition]
                    std = abundance_df["log2_std_AUC_" + condition]
                offset = width * multiplier
                ax.barh(y + offset, mean, width, xerr = std, label=condition, align='center')
                multiplier += 1
            ax.set_yticks(y)
            ax.set_yticklabels(abundance_df['ClassKey'].values)
            ax.set_xlabel('Mean AUC')
            ax.set_ylabel('Lipid Class')
            ax.set_title('Total Abundance Bar Chart')
            plt.legend()
            return fig
        
        fig = render_plot()
        
        def display_plot(a_fig):
            abundance_df.drop(['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list], axis=1, inplace=True)
            st.pyplot(a_fig)
            csv_download = self.convert_df(abundance_df)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='abundance_bar_chart.csv',
                            mime='text/csv')
            
            return
        
        display_plot(fig)
        return 
        
    