import pandas as pd
import numpy as np
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Whisker, BasicTickFormatter
from bokeh.transform import dodge

class SaturationPlot:
    
    def __init__(self, an_experiment, a_df):
        self.experiment = an_experiment
        self.df = a_df
        
    @st.cache_data
    def convert_df(_self, a_dataframe):
         return a_dataframe.to_csv().encode('utf-8')
        
    @st.cache_data
    def calculate_FA_ratio(_self, a_mol_structure):
        """
        The following is an example of a_mol_structrure: PA(16:0_20:4)
        """
        a = a_mol_structure.split('(')
        b = a[1][:-1]
        c = b.split('_')
        sfa_ratio = 0
        mufa_ratio = 0
        pufa_ratio = 0
        for item in c:
            d = item.split(':')[-1]
            if str(d).strip() == '0':
                sfa_ratio += 1
            elif str(d).strip() == '1': 
                mufa_ratio += 1
            else: 
                pufa_ratio += 1
        total = sfa_ratio + mufa_ratio + pufa_ratio
        sfa_ratio = sfa_ratio / total
        mufa_ratio = mufa_ratio / total
        pufa_ratio = pufa_ratio / total
        return sfa_ratio, mufa_ratio, pufa_ratio
    
    @st.cache_data
    def calculate_SFA_MUFA_PUFA(_self, a_df, a_conditions_list, an_individual_samples_list, a_lipid_class, a_condition):
        index = a_conditions_list.index(a_condition)
        temp_df = a_df[a_df['ClassKey'] == a_lipid_class]
        temp_df[a_condition + '_mean_AUC'] = temp_df[['MeanArea[' + sample + ']' for sample in an_individual_samples_list[index]]]\
            .apply(lambda x: np.mean(x), axis = 1)
        temp_df[a_condition + '_var_AUC'] = temp_df[['MeanArea[' + sample + ']' for sample in an_individual_samples_list[index]]]\
            .apply(lambda x: np.var(x), axis = 1)
        temp_df = temp_df[['LipidMolec', a_condition + '_mean_AUC', a_condition + '_var_AUC']]
        temp_df['FA_ratio'] = temp_df['LipidMolec'].apply(lambda x: _self.calculate_FA_ratio(x))
        temp_df['SFA_ratio'] = temp_df['FA_ratio'].apply(lambda x: x[0])
        temp_df['MUFA_ratio'] = temp_df['FA_ratio'].apply(lambda x: x[1])
        temp_df['PUFA_ratio'] = temp_df['FA_ratio'].apply(lambda x: x[2])
        temp_df['SFA_AUC'] = temp_df[[a_condition + '_mean_AUC', 'SFA_ratio']].apply(lambda x: x[0]*x[1], axis = 1)
        temp_df['MUFA_AUC'] = temp_df[[a_condition + '_mean_AUC', 'MUFA_ratio']].apply(lambda x: x[0]*x[1], axis = 1)
        temp_df['PUFA_AUC'] = temp_df[[a_condition + '_mean_AUC', 'PUFA_ratio']].apply(lambda x: x[0]*x[1], axis = 1)
        temp_df['SFA_var'] = temp_df[[a_condition + '_var_AUC', 'SFA_ratio']].apply(lambda x: x[0]*x[1]**2, axis = 1)
        temp_df['MUFA_var'] = temp_df[[a_condition + '_var_AUC', 'MUFA_ratio']].apply(lambda x: x[0]*x[1]**2, axis = 1)
        temp_df['PUFA_var'] = temp_df[[a_condition + '_var_AUC', 'PUFA_ratio']].apply(lambda x: x[0]*x[1]**2, axis = 1)
        sfa = temp_df['SFA_AUC'].sum()
        sfa_var = temp_df['SFA_var'].sum()
        mufa = temp_df['MUFA_AUC'].sum()
        mufa_var = temp_df['MUFA_var'].sum()
        pufa = temp_df['PUFA_AUC'].sum()
        pufa_var = temp_df['PUFA_var'].sum()
        return sfa, mufa, pufa, sfa_var, mufa_var, pufa_var
        
    def create_saturation_plot(self, a_selected_conditions_list):
        
        def create_sfa_mufa_pufa_lists():
            a_sfa_lst = []
            a_mufa_lst = []
            a_pufa_lst = []
            a_sfa_var_lst = []
            a_mufa_var_lst = []
            a_pufa_var_lst = []
            for condition in a_selected_conditions_list:
                sfa, mufa, pufa, sfa_var, mufa_var, pufa_var = self.calculate_SFA_MUFA_PUFA(self.df.copy(deep=True), self.experiment.conditions_list, \
                                                                                            self.experiment.individual_samples_list, lipid_class, condition)
                a_sfa_lst.append(sfa)
                a_mufa_lst.append(mufa)
                a_pufa_lst.append(pufa)
                a_sfa_var_lst.append(np.sqrt(sfa_var))
                a_mufa_var_lst.append(np.sqrt(mufa_var))
                a_pufa_var_lst.append(np.sqrt(pufa_var))
            return a_sfa_lst, a_mufa_lst, a_pufa_lst, a_sfa_var_lst, a_mufa_var_lst, a_pufa_var_lst
        
        def calculate_saturation_level_percentage(a_sfa_lst, a_mufa_lst, a_pufa_lst):
            return [a_sfa_lst[i]/(a_sfa_lst[i]+a_mufa_lst[i]+a_pufa_lst[i])*100 for i in range(len(a_selected_conditions_list))],\
                   [a_mufa_lst[i]/(a_sfa_lst[i]+a_mufa_lst[i]+a_pufa_lst[i])*100 for i in range(len(a_selected_conditions_list))],\
                   [a_pufa_lst[i]/(a_sfa_lst[i]+a_mufa_lst[i]+a_pufa_lst[i])*100 for i in range(len(a_selected_conditions_list))]
        
        def create_main_plot_df(a_sfa_lst, a_mufa_lst, a_pufa_lst, a_sfa_var_lst, a_mufa_var_lst, a_pufa_var_lst):
            a_main_data_df = {'conditions' : a_selected_conditions_list,
                    'SFA'   : a_sfa_lst,
                    'SFA_upper' : [x+e for x,e in zip(a_sfa_lst, a_sfa_var_lst)],
                    'SFA_lower' : [x-e for x,e in zip(a_sfa_lst, a_sfa_var_lst)],
                    'MUFA'   : a_mufa_lst,
                    'MUFA_upper' : [x+e for x,e in zip(a_mufa_lst, a_mufa_var_lst)],
                    'MUFA_lower' : [x-e for x,e in zip(a_mufa_lst, a_mufa_var_lst)],
                    'PUFA'   : a_pufa_lst,
                    'PUFA_upper' : [x+e for x,e in zip(a_pufa_lst, a_pufa_var_lst)],
                    'PUFA_lower' : [x-e for x,e in zip(a_pufa_lst, a_pufa_var_lst)]}
            return a_main_data_df
        
        def create_percentage_plot_df(a_sfa_percentage_lst, a_mufa_percentage_lst, a_pufa_percentage_lst):
            a_percentage_data_df = {'conditions' : a_selected_conditions_list,
                    'SFA'   : a_sfa_percentage_lst,
                    'MUFA'   : a_mufa_percentage_lst,
                    'PUFA'   : a_pufa_percentage_lst}
            return a_percentage_data_df 
        
        def render_main_plot(a_lipid_class, a_plot_df, a_sfa_lst, a_mufa_lst, a_pufa_lst):
            a_plot_df = a_plot_df.fillna(0)
            max_height = max([max(a_sfa_lst), max(a_mufa_lst), max(a_pufa_lst)])
            source = ColumnDataSource(data=a_plot_df)
            p = figure(x_range = a_selected_conditions_list, y_range = (0, max_height * 2), height=250, title="Saturation Level Plot - " + a_lipid_class,
                       x_axis_label= 'Conditions', y_axis_label= 'Total AUC',toolbar_location='right')
            p.vbar(x=dodge('conditions', -0.25, range=p.x_range), top='SFA', width=0.2, source=source,
                       color="#c9d9d3", legend_label="SFA")
            p.add_layout(Whisker(source=source, base=dodge('conditions', -0.25, range=p.x_range), \
                                 upper="SFA_upper", lower="SFA_lower", level='overlay'))
            p.vbar(x=dodge('conditions',  0.0,  range=p.x_range), top='MUFA', width=0.2, source=source,
                       color="#718dbf", legend_label="MUFA")
            p.add_layout(Whisker(source=source, base=dodge('conditions', 0.0, range=p.x_range), \
                                 upper="MUFA_upper", lower="MUFA_lower", level='overlay'))
            p.vbar(x=dodge('conditions',  0.25, range=p.x_range), top='PUFA', width=0.2, source=source,
                       color="#e84d60", legend_label="PUFA")
            p.add_layout(Whisker(source=source, base=dodge('conditions', 0.25, range=p.x_range), \
                                 upper="PUFA_upper", lower="PUFA_lower", level='overlay'))
            p.x_range.range_padding = 0.1
            p.xgrid.grid_line_color = None
            p.legend.location = "top_center"
            p.legend.orientation = "horizontal"
            p.legend.label_text_font_size = "10pt"
            p.title.text_font_size = "15pt"
            p.xaxis.axis_label_text_font_size = "15pt"
            p.yaxis.axis_label_text_font_size = "15pt"
            p.xaxis.major_label_text_font_size = "15pt"
            p.yaxis.major_label_text_font_size = "15pt"
            p.yaxis.formatter = BasicTickFormatter(precision=1)
            return p 
        
        def render_percentage_plot(a_lipid_class, a_plot_df):
            a_plot_df = a_plot_df.fillna(0)
            colors = ["#c9d9d3", "#718dbf", "#e84d60"]
            p = figure(x_range = a_selected_conditions_list, y_range = (0, 100), height=250, title="Saturation Level Plot - " + lipid_class,
                       x_axis_label= 'Conditions', y_axis_label= 'Percentage',toolbar_location='right')
            p.vbar_stack(['SFA', 'MUFA', 'PUFA'], x='conditions', color = colors, width=0.2, source=a_plot_df,
                         legend_label=['SFA', 'MUFA', 'PUFA'])
            p.x_range.range_padding = 0.1
            p.xgrid.grid_line_color = None
            p.legend.location = "center_right"
            p.legend.orientation = "vertical"
            p.legend.label_text_font_size = "6.5pt"
            p.title.text_font_size = "15pt"
            p.xaxis.axis_label_text_font_size = "15pt"
            p.yaxis.axis_label_text_font_size = "15pt"
            p.xaxis.major_label_text_font_size = "15pt"
            p.yaxis.major_label_text_font_size = "15pt"
            p.yaxis.formatter = BasicTickFormatter(precision=1)
            return p
        
        def display_plot(a_main_plot, a_sat_df, a_percentage_plot, a_percentage_df):
            st.bokeh_chart(a_main_plot)
            csv_download = self.convert_df(a_sat_df)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='saturation_level_plot_main.csv',
                            mime='text/csv',
                            key=lipid_class)
            
            st.bokeh_chart(a_percentage_plot)
            csv_download = self.convert_df(a_percentage_df)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='saturation_level_plot_percentage.csv',
                            mime='text/csv',
                            key=lipid_class+'2')
            st.write('------------------------------------------------------------------------------------------------')
            return 
        
        for lipid_class in self.df['ClassKey'].unique():
            sfa_lst, mufa_lst, pufa_lst, sfa_var_lst, mufa_var_lst, pufa_var_lst = create_sfa_mufa_pufa_lists()
            sfa_percentage_lst, mufa_percentage_lst, pufa_percentage_lst = calculate_saturation_level_percentage(sfa_lst, mufa_lst, pufa_lst)
            main_data = create_main_plot_df(sfa_lst, mufa_lst, pufa_lst, sfa_var_lst, mufa_var_lst, pufa_var_lst)
            main_df = pd.DataFrame.from_dict(main_data)
            percentage_data = create_percentage_plot_df(sfa_percentage_lst, mufa_percentage_lst, pufa_percentage_lst)
            percentage_df = pd.DataFrame.from_dict(percentage_data)
            main_plot = render_main_plot(lipid_class, main_df, sfa_lst, mufa_lst, pufa_lst)
            percentage_plot = render_percentage_plot(lipid_class, percentage_df)
            sat_df_main = pd.DataFrame({"Conditions": a_selected_conditions_list, "SFA": sfa_lst, "SFA_STDV": sfa_var_lst, "MUFA": mufa_lst, \
                           "MUFA_STDV": mufa_var_lst, "PUFA": pufa_lst, "PUFA_STDV": pufa_lst})
            sat_df_percentage = pd.DataFrame({"Conditions": a_selected_conditions_list, "SFA": sfa_percentage_lst, "MUFA": mufa_percentage_lst, \
                           "PUFA": pufa_percentage_lst})
            display_plot(main_plot, sat_df_main, percentage_plot, sat_df_percentage)
        return 
    
    
    
    