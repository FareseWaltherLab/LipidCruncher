import streamlit as st 
import pandas as pd
import numpy as np

class NormalizeData:
    
    def __init__(self, a_df, an_intsta_df, an_experiment):
        self.df = a_df
        self.intsta_df = an_intsta_df
        self.experiment = an_experiment 
        
    def collect_user_input(self):
        all_class_lst = self.df['ClassKey'].unique() # list of lipid classes corresponding to all detected lipid species   
        intsta_species_lst = self.intsta_df['LipidMolec'].tolist() # list of internal standards species 
        
        st.write('The following are the detected internal standards species listed above:')
        for lipid in intsta_species_lst:
            st.write(lipid)
        
        # list of classes selected by the user that can be normalized using the detected internal standards species 
        selected_class_list = st.multiselect('Remove classes that cannot be normalized with any of the detected internal standards:', all_class_lst, all_class_lst)
        
        # function that receives the user input on which internal standards to use for specified lipid classes
        def pick_intsta(a_selected_class_lst, an_intsta_species_lst): 
            selected_intsta_lst = [] # list of internal standards species selected by the user to normalize classes in a_selected_class_lst 
            for lipid_class in a_selected_class_lst:
                selected_intsta_lst.append(build_intsta_selectbox(lipid_class, intsta_species_lst)) # function defined below 
            return selected_intsta_lst
        
        # function that creates a select box that includes the list of the detected internal standards 
        def build_intsta_selectbox(a_lipid_class, an_intsta_species_lst):  
            added_intsta_species = st.selectbox('Pick a internal standard for ' + a_lipid_class + ' species', an_intsta_species_lst)
            return added_intsta_species
        
        st.write('For each lipid class, pick one internal standard that most closely matches it.')
        added_intsta_species_lst = pick_intsta(selected_class_list, intsta_species_lst)
        for index, lipid_class in enumerate(selected_class_list):
            st.write('- ' + lipid_class + ' data will be normalized using ' + added_intsta_species_lst[index])
        
        # function for letting the user input the concentration of IS 
        def build_intsta_concentration_dict(an_added_intsta_species_lst):
            intsta_concentration_dict = {}
            for lipid in an_added_intsta_species_lst:
                if lipid not in intsta_concentration_dict.keys(): 
                    intsta_concentration_dict[lipid] = build_concentration_input_box(lipid)
            return intsta_concentration_dict
        
        def build_concentration_input_box(a_lipid):
            concentration = st.number_input('Enter the concentration of ' + a_lipid + ' in micromole', 
                                      min_value = 0, max_value = 100000, value = 1, step = 1)
            return concentration 
        
        st.write("Now, enter the concentration of each internal standard species:")
        intsta_concentration_dict = build_intsta_concentration_dict(added_intsta_species_lst)
        
        return selected_class_list, added_intsta_species_lst, intsta_concentration_dict
    
    def normalization(self, a_selected_class_list, an_added_intsta_species_lst, an_intsta_concentration_dict):
        
        @st.cache_data
        def prep_df(a_dataframe, a_selected_class_list):
            return a_dataframe[a_dataframe['ClassKey'].isin(a_selected_class_list)]
        
        selected_df = prep_df(self.df, a_selected_class_list) # normalized dataframe
        
        full_samples_list = self.experiment.full_samples_list
        
        @st.cache_data
        def compute_normalized_auc(a_selected_df, a_full_samples_list, a_lipid_class, an_intsta_auc, a_concentration):
            a_temp_df = a_selected_df[['MeanArea[' + sample + ']' for sample in a_full_samples_list]][a_selected_df['ClassKey'] == a_lipid_class]\
                .apply(lambda auc: auc / an_intsta_auc * a_concentration, axis=1)
            a_temp_df[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt']] = \
                a_selected_df[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt']][a_selected_df['ClassKey'] == a_lipid_class]
            return a_temp_df 
            
        def create_normalized_df(a_norm_df):
            for intsta_species, lipid_class in zip(an_added_intsta_species_lst, a_selected_class_list):
                concentration = an_intsta_concentration_dict[intsta_species]
                intsta_auc = self.intsta_df[['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]][self.intsta_df['LipidMolec'] == intsta_species]\
                    .values.reshape(len(self.experiment.full_samples_list),)
                temp = compute_normalized_auc(selected_df.copy(deep=True), full_samples_list, lipid_class, intsta_auc, concentration)
                temp = temp[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt'] + ['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]]
                a_norm_df = pd.concat([a_norm_df, temp], axis=0)
            return a_norm_df 
        
        norm_df = pd.DataFrame(columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt'] + \
                               ['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list])
        norm_df = create_normalized_df(norm_df)
        norm_df.reset_index(inplace=True)
        norm_df.drop('index', axis=1, inplace=True)
        norm_df = norm_df.fillna(0)
        norm_df.replace([np.inf, -np.inf], 0, inplace=True)
        st.write(norm_df)
        return norm_df 