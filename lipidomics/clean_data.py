import streamlit as st 
import pandas as pd
import numpy as np

class CleanData:
    
    def __init__(self, a_df, an_experiment, a_name_df):
        self.df = a_df
        self.experiment = an_experiment
        self.name_df = a_name_df
        
    def data_cleaner(self):
        
        # Extract relevant columns for analysis 
        def extract_relevant_columns():
            return self.df[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalGrade', 'TotalSmpIDRate(%)'] + \
                           ['MeanArea[' + sample + ']' for sample in self.name_df['old name']]]
        
        df = extract_relevant_columns()
        
        def update_column_names(a_df):
            total_reps = sum(self.experiment.number_of_samples_list) # total number of all replicates
            for (sample_1, sample_2) in zip(self.name_df['old name'], self.name_df['updated name']):
                a_df.rename(columns={'MeanArea[' + sample_1 + ']' : 'Area[' + sample_2 + ']'}, inplace = True)
            for i in range(total_reps):
                a_df.rename(columns={'Area[s' + str(i+1) + ']' : 'MeanArea[s' + str(i+1) + ']'}, inplace = True)
            return a_df
        
        df = update_column_names(df)
        df = df.fillna(0)
        auc = ['MeanArea[' + sample +']' for sample in self.experiment.full_samples_list]
        for col in df[auc].columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # removes the datapoint if 'TotalGrade' = C or D
        @st.cache_data
        def apply_filter(a_df):
            return a_df[a_df['TotalGrade'].isin(['A', 'B', 'C'])]
        
        filtered_df = apply_filter(df)
        
        # extracts the list of fatty acids corresponding to the given lipid 
        def extract_fatty_acids(a_lipid):
            fatty_acids_str = a_lipid.split(')')[0].split('(')
            fatty_acids_list = fatty_acids_str[1].split('_')
            fatty_acids_list.sort()
            return fatty_acids_list
        
        @st.cache_data
        def add_FAKey_column(a_df):
            a_df['FAKey'] = a_df['LipidMolec'].apply(lambda x: extract_fatty_acids(x))
            return a_df
        
        df_with_FAKey = add_FAKey_column(filtered_df.copy(deep=True))
        
        def correct_single_lipid_molec(attributes_list):
            """
            Sometimes two lipids are identical but they appear with slightly different names in the LipidMolec column.
            For example, PE(18:0_20:2) and PE(20:2_18:0). This function makes the naming consistent. After passing through 
            this function, both names appear as PE(18:0_20:2).
            
            the input is in the form of a list including some attributes of the selected lipid. 
            For example, for PE(18:0_20:2), the input (attributes list) is as following:
            ['PE', ['18:0', '20:2'], 'PE(18:0_20:2)']
            """
            corrected_lipid_molec = attributes_list[0] + '(' 
            for i in range(len(attributes_list[1])):
                corrected_lipid_molec += attributes_list[1][i] + '_'
            corrected_lipid_molec = corrected_lipid_molec[0 : len(corrected_lipid_molec) - 1]
            corrected_lipid_molec += ')'
            if len(attributes_list[2].split('+')) > 1:
                corrected_lipid_molec += '+' + attributes_list[2].split('+')[1]
            return corrected_lipid_molec
        
        @st.cache_data
        def correct_lipid_molec_column(a_df):
            a_df['LipidMolec_modified'] = a_df[['ClassKey', 'FAKey', 'LipidMolec']].apply(lambda x: correct_single_lipid_molec(x), axis=1)
            a_df.drop(columns='LipidMolec', inplace=True)
            a_df.rename(columns={'LipidMolec_modified': 'LipidMolec'}, inplace=True)
            return a_df 
        
        corrected_df = correct_lipid_molec_column(df_with_FAKey.copy(deep=True))
        
        @st.cache_data
        # function that picks the lipid species with highest quality peak (highest TotalSmpIDRate for each unique lipid) 
        def select_AUC(a_df):
            clean_df = pd.DataFrame(columns = ['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                                    ['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list])
            temp_df = a_df[a_df['TotalGrade'].isin(['A', 'B'])]
            for lipid in temp_df['LipidMolec'].unique():
                isolated_df = temp_df[temp_df['LipidMolec'] == lipid] # isolates all datapoints corresponding to the current lipid 
                max_peak_quality = max(isolated_df['TotalSmpIDRate(%)'].values)
                isolated_df = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality]
                isolated_df = isolated_df.iloc[:1]
                isolated_df = isolated_df[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                                          ['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]]
                clean_df = pd.concat([clean_df, isolated_df], axis=0)
             
            # consider 'C' species only for SM and LPC
            temp_df = a_df[a_df['TotalGrade'].isin(['C'])]   
            for lipid in temp_df[temp_df['ClassKey'].isin(['LPC', 'SM'])]['LipidMolec'].unique():
                if lipid not in clean_df['LipidMolec'].unique():
                    isolated_df = temp_df[temp_df['LipidMolec'] == lipid] # isolates all datapoints corresponding to the current lipid 
                    max_peak_quality = max(isolated_df['TotalSmpIDRate(%)'].values)
                    isolated_df = isolated_df[isolated_df['TotalSmpIDRate(%)'] == max_peak_quality]
                    isolated_df = isolated_df.iloc[:1]
                    isolated_df = isolated_df[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'TotalSmpIDRate(%)'] + \
                                              ['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]]
                    clean_df = pd.concat([clean_df, isolated_df], axis=0)
                    
            clean_df = clean_df[['LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt'] + \
                           ['MeanArea[' + sample + ']' for sample in self.experiment.full_samples_list]]
            clean_df.reset_index(drop = True, inplace = True)
            return clean_df 
        
        cleaned_df = select_AUC(corrected_df)
        return cleaned_df
    
        # function to extract the internal standards dataframe from the cleaned dataset  
    @st.cache_data
    def extract_internal_standards(_self, a_clean_df):
            intsta_index_lst = [index for index, lipid in enumerate(a_clean_df['LipidMolec'].values.tolist()) if '(s)' in lipid]
            intsta_df = a_clean_df.iloc[intsta_index_lst, :]
            a_clean_df.drop(intsta_index_lst,0,inplace=True) # removes the internal standard rows from the dataset
            return a_clean_df, intsta_df
    
        # function to log transform the abundance columns of the df
    @st.cache_data
    def log_transform_df(_self, a_clean_df):  
            auc = ['MeanArea[' + sample +']' for sample in _self.experiment.full_samples_list]
            a_clean_df[auc] = a_clean_df[auc].mask(a_clean_df[auc]<1).fillna(1) # filling zero values with 1's to avoid infinity
            a_clean_df[auc] = a_clean_df[auc].apply(lambda x: np.log10(x), axis=0)
            return a_clean_df
    
