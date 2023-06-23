import streamlit as st
import matplotlib.pyplot as plt
import numpy as np 

class Histogram:
    
    def __init__(self, an_experiment, a_df):
        self.experiment = an_experiment 
        self.df = a_df 
        
    def plot_hist(self):  
        
        main_area_df = self.df[['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]] # picks the 'MeanArea[s1]', ..., 'MeanArea[sN]' columns only

        @st.cache_data
        def prep_df(a_main_area_df):
            a_prepped_df = a_main_area_df.mask(a_main_area_df<1).fillna(1)
            return a_prepped_df
        
        prepped_df = prep_df(main_area_df)
        total_reps = len(self.experiment.full_samples_list)
        
        # function that prepares the histogram plot
        def prep_hist(a_dataframe, index):
            figure, ax = plt.subplots(figsize=(10, 10))
            lst = np.log10(a_dataframe['MeanArea[' + self.experiment.full_samples_list[index] + ']'].values)
            ax.grid(False)
            ax.hist(lst, bins = 75, range=(0, 12))
            ax.set_title('Histogram of AUC - '+ self.experiment.full_samples_list[index], fontsize=50)
            ax.set_xlabel('log10(AUC)', fontsize=50)
            ax.set_ylabel('Count', fontsize=50)
            plt.xticks(fontsize=50)
            plt.yticks(fontsize=50)
            ax.set_xlim([-0.5, 12])
            return figure
            
        def arrange_hist(a_dataframe, total_replicates):
            for i in range(int(round(total_replicates/3, 1))):
                    # put 3 histagrams in each row 
                    col1, col2, col3 = st.columns(3)
                    col1.pyplot(prep_hist(a_dataframe, 3*i))
                    col2.pyplot(prep_hist(a_dataframe, 3*i+1))
                    col3.pyplot(prep_hist(a_dataframe, 3*i+2))
        
            # if the last row includes only 2 histograms 
            if total_replicates % 3 == 2:
                    col1, col2, col3 = st.columns(3)
                    col1.pyplot(prep_hist(a_dataframe, -2))
                    col2.pyplot(prep_hist(a_dataframe, -1))
                    
            # if the last row includes only 1 histogram                     
            elif total_replicates %3 == 1:
                    col1, col2, col3 = st.columns(3)
                    col1.pyplot(prep_hist(a_dataframe, -1))
            return
        
        return arrange_hist(prepped_df, total_reps)
    
    
        