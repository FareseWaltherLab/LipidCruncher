import streamlit as st

class Experiment:
    
    def __init__(self):
        self.n_conditions = None
        self.conditions_list = []
        self.number_of_samples_list = []
        self.aggregate_number_of_samples_list = []
        self.extensive_conditions_list = []
        self.individual_samples_list = []
        self.full_samples_list = None
        
    def create_aggregate_number_of_samples_list(self, a_number_of_samples_list):
        '''
        For example, if number_of_samples_list = [2, 4, 5], the aggregate_number_of_samples_list = [2, 6, 11]
        '''
        return [sum(a_number_of_samples_list[0:i+1]) for i in range(len(a_number_of_samples_list))]
    
    # List of the list of samples corresponding to each condition 
    def create_individual_samples_list(self):
        self.individual_samples_list = []
        temporary_list = self.full_samples_list.copy() # Make a copy of full_sample_list to prevent affecting it 
        for condition, replicate in zip(self.conditions_list, self.number_of_samples_list):
            self.individual_samples_list.append(temporary_list[: replicate])
            del temporary_list[: replicate]
        return self.individual_samples_list
        
    def create_attributes(self):
        
        # Number of experimental groups or conditions
        def get_n_conditions():
            return st.sidebar.number_input('Enter the number of conditions',min_value = 1, max_value= 20, value = 1, step = 1)
        self.n_conditions = get_n_conditions()

        # List of conditions
        def get_conditions_list():
            return [st.sidebar.text_input('Create a label for condition #'+str(i+1)+' (e.g. WT or KO)') for i in range(self.n_conditions)]
        self.conditions_list = get_conditions_list()
        
        # print an error message if a condition has no label
        if "" in self.conditions_list:
            raise NameError ("Condition's label must be at least one character long! ")

        # List of the number of samples corresponding to each condition
        def get_number_of_samples_list():
            return [st.sidebar.number_input('Enter the number of samples for condition #'+str(i+1), min_value = 1, max_value = 1000, value = 1, step = 1)\
                    for i in range(self.n_conditions)]
        self.number_of_samples_list = get_number_of_samples_list()
        self.aggregate_number_of_samples_list = self.create_aggregate_number_of_samples_list(self.number_of_samples_list)
        
        # list includes the condition correponding to each replicae - length equal to the total number of replicates
        def create_extensive_conditions_list():
            extensive_conditions_list = []  
            for index, cond in enumerate(self.conditions_list):
                extensive_conditions_list += [cond for i in range(self.number_of_samples_list[index])]
            return extensive_conditions_list 
        self.extensive_conditions_list = create_extensive_conditions_list()
        
        # List of all samples 
        def create_full_samples_list():
            return ['s' + str(i+1) for i in range(sum(self.number_of_samples_list))]
        self.full_samples_list = create_full_samples_list()
        self.individual_samples_list = self.create_individual_samples_list()
        
        return
    
    def remove_bad_samples(self, a_list_of_bad_samples, a_dataframe):
        a_dataframe.drop(['MeanArea[' + sample + ']' for sample in a_list_of_bad_samples], axis=1, inplace=True) # updating dataframe
        bad_samples_index_list = [index for index, sample in enumerate(self.full_samples_list) if sample in a_list_of_bad_samples]
        
        def reconstruct_number_of_samples_list():
            new_number_of_samples_list = []
            for condition_1 in self.conditions_list:
                counter = 0
                for condition_2 in self.extensive_conditions_list:
                    if condition_2 == condition_1:
                        counter += 1
                new_number_of_samples_list.append(counter)
            return new_number_of_samples_list
        
        def reconstruct_object():
            self.full_samples_list = [sample for sample in self.full_samples_list if sample not in a_list_of_bad_samples]
            self.extensive_conditions_list = [condition for index, condition in enumerate(self.extensive_conditions_list) if index not in bad_samples_index_list]
            self.conditions_list = [condition for condition in self.conditions_list if condition in self.extensive_conditions_list]
            self.n_conditions = len(self.conditions_list)
            self.number_of_samples_list = reconstruct_number_of_samples_list()
            self.aggregate_number_of_samples_list = self.create_aggregate_number_of_samples_list(self.number_of_samples_list)
            self.individual_samples_list = self.create_individual_samples_list()
            return 
        
        reconstruct_object()
        
        return a_dataframe
            
    def __repr__(self):
        return " conditions_list:{},\n number_of_samples_list:{},\n aggregate_number_of_samples_list:{},\n extensive_conditions_list:{},\n \
                individual_samples_list:{},\n full_samples_list:{},\n "\
                .format(self.conditions_list, self.number_of_samples_list, self.aggregate_number_of_samples_list, self.extensive_conditions_list, \
                self.individual_samples_list, self.full_samples_list)
    
    def __str__(self):
        return self.__repr__()
    

                
                
