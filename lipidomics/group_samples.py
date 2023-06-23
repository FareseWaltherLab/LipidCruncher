import streamlit as st 
import pandas as pd

class GroupSamples:
    
    def __init__(self, a_df, an_experiment):
        self.df = a_df
        self.experiment = an_experiment
        
    def check_dataset_validity(self):
        cols = self.df.columns    
        if ('LipidMolec' not in cols) or ('ClassKey' not in cols) or ('BaseRt' not in cols):
            raise TypeError ("This is not a valid LipidSearch 5.0 dataset!")
            
    def check_input_validity(self):
        if len(self.build_mean_area_col_lst()) != len(self.experiment.full_samples_list):
            raise ValueError ("Invalid total number of samples!")
        
    # give user the necessary info about how the samples should be grouped together
    def give_info(self):
        for index, cond in enumerate(self.experiment.conditions_list):
            if index == 0:
                st.sidebar.write('- Samples indexed from 0 to ' + str(self.experiment.aggregate_number_of_samples_list[0]-1) + ' must belong to ' + cond)
            else: 
                st.sidebar.write('- Samples indexed from ' + str(self.experiment.aggregate_number_of_samples_list[index-1]) + ' to ' 
                                 + str(self.experiment.aggregate_number_of_samples_list[index]-1) + ' must belong to ' + cond)
                
    def build_mean_area_col_lst(self):
        mean_area_col_lst = [int(col[10:-1]) for col in self.df.columns if ('MeanArea[s' in col and 'Org' not in col)]
        mean_area_col_lst.sort()
        return mean_area_col_lst
    
    
    def build_group_df(self):
        group_dict = {'sample name' : [], 'condition' : []}
        group_dict['sample name'] = ['s' + str(ele) for ele in self.build_mean_area_col_lst()]
        group_dict['condition'] = self.experiment.extensive_conditions_list
        group_df = pd.DataFrame.from_dict(group_dict) # df including sample_name and corresponding condition 
        return group_df 
    
    # groups the samples together the correct way, if they aren't grouped together properly 
    def group_samples(self, group_df):
        ordered_col_lst = []
        col_lst = group_df['sample name']
        for cond in self.experiment.conditions_list:
            temp = st.sidebar.multiselect('Pick the samples that belong to condition ' + cond, col_lst)
            ordered_col_lst += temp
            col_lst = [ele for ele in col_lst if ele not in temp]
        if len(ordered_col_lst) == sum(self.experiment.number_of_samples_list):
            group_df['sample name'] = ordered_col_lst
        return group_df
    
    # function that updates the sample names, so, they are consistent with the following format: s1, s2, s3, ..., sN. 
    def update_sample_names(self, group_df):  
        """
        Fixed the problem of missing samples. For example, if you have 4 samples with the following names: s1, s2, s3, s5,
        then the updated names are s1, s2, s3 and s4.  
        """
        name_dict = {"old name" : [] , "updated name" : [], "condition": []}
        name_dict['old name'] = group_df['sample name']
        name_dict['updated name'] = ['s'+str(i+1) for i in range(sum(self.experiment.number_of_samples_list))]
        name_dict['condition'] = group_df['condition']
        name_df = pd.DataFrame.from_dict(name_dict)
        return name_df
    
    # pairs the replicates with their corresponding condition
    def build_replicate_condition_pair(self, condition):  
        '''
        For example, if conditions_lst = [WT, KO] and number_of_samples_list = [2, 2], the function returns:
            s1-s2 correspond to WT
            s3-s4 correspond to KO
        '''
        index = self.experiment.conditions_list.index(condition)
        # list of samples corresponding to the current condition 
        current_sample_lst = self.experiment.individual_samples_list[index]
        if len(current_sample_lst) == 1:
            return st.sidebar.write("- "+current_sample_lst[0]+" corresponds to "+condition)
        else:
            return st.sidebar.write("- "+current_sample_lst[0]+"-"+current_sample_lst[-1]+" correspond to "+condition)
    
    def confirm_inputs(self, group_df):
        st.sidebar.write("There are a total of "+str(sum(self.experiment.number_of_samples_list))+" samples.")
        for condition in self.experiment.conditions_list:
            self.build_replicate_condition_pair(condition) 
        confirm = st.sidebar.checkbox("Confirm the inputs by checking this box")
        return confirm
    
    
        