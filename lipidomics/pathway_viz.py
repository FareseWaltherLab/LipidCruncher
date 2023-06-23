import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class PathwayViz:
    
    def __init__(self, an_experiment, a_df, a_control_condition, an_experimental_condition):
        self.experiment = an_experiment
        self.df = a_df
        self.control = a_control_condition
        self.experimental = an_experimental_condition
        
    @st.cache_data
    def convert_df(_self, a_dataframe):
         return a_dataframe.to_csv().encode('utf-8')
    
    def calculate_class_saturation_ratio(self):
        
        @st.cache_data
        def calculate_number_of_saturated_and_unsaturated_chains_for_single_species(mol_structure):
            a = mol_structure.split('(')
            b = a[1][:-1]
            c = b.split('_')
            number_of_sat_chains = 0
            number_of_unsat_chains = 0
            for item in c:
                d = item.split(':')[-1]
                if d.isnumeric():
                    if int(d) == 0:
                        number_of_sat_chains += 1 
                    else:
                        number_of_unsat_chains += 1
                else:
                    if '0' in d:   
                        number_of_sat_chains += 1
                    else:
                        number_of_unsat_chains += 1
            return number_of_sat_chains, number_of_unsat_chains
        
        df = self.df.copy(deep=True)
        
        @st.cache_data
        def calculate_total_number_of_saturated_and_unsaturated_chains_for_each_class(a_df):
            a_df['number_of_sat_unsat_chains'] = a_df['LipidMolec'].apply(lambda x: calculate_number_of_saturated_and_unsaturated_chains_for_single_species(x))
            a_df['number_of_sat_chains'] = a_df['number_of_sat_unsat_chains'].apply(lambda x: x[0])
            a_df['number_of_unsat_chains'] = a_df['number_of_sat_unsat_chains'].apply(lambda x: x[1])
            return a_df
        df = calculate_total_number_of_saturated_and_unsaturated_chains_for_each_class(df)
        
        @st.cache_data
        def add_saturation_ratio_column(a_df):
            saturation_ratio_df = a_df.groupby('ClassKey')[['number_of_sat_chains', 'number_of_unsat_chains']].sum()
            saturation_ratio_df['saturation_ratio'] = saturation_ratio_df[['number_of_sat_chains', 'number_of_unsat_chains']].apply(lambda x: x[0]/(x[0]+x[1]), axis = 1)
            return saturation_ratio_df
        saturation_ratio_df = add_saturation_ratio_column(df)
        saturation_ratio_df.drop(['number_of_sat_chains', 'number_of_unsat_chains'], axis=1, inplace=True)
        
        return saturation_ratio_df
    
    
    def calculate_class_fold_change(self):
        
        df = self.df
        full_samples_list = self.experiment.full_samples_list
        control = self.control
        experimental = self.experimental
        
        @st.cache_data
        def calculate_total_class_abundance(a_df, a_full_samples_list):
            return a_df.groupby('ClassKey')[["MeanArea[" + sample + ']' for sample in a_full_samples_list]].sum()
        
        def get_index():
            return self.experiment.conditions_list.index(self.control),\
                   self.experiment.conditions_list.index(self.experimental)
        control_idx, experimental_idx = get_index()
        
        def get_individual_samples_list():
            return self.experiment.individual_samples_list[control_idx], \
                   self.experiment.individual_samples_list[experimental_idx]
        control_samples_list, experimental_samples_list = get_individual_samples_list()
        
        @st.cache_data
        def add_fold_change_column(a_dataframe, a_control, an_experimental, a_full_samples_list):
            a_dataframe['fc_' + an_experimental + '_' + a_control] = \
            a_dataframe[['MeanArea[' + sample + ']'  for sample in control_samples_list + experimental_samples_list]]\
            .apply(lambda x: np.mean(x[['MeanArea[' + sample + ']'  for sample in experimental_samples_list]])\
                   / np.mean(x[['MeanArea[' + sample + ']'  for sample in control_samples_list]]), axis = 1)
            a_dataframe.drop(["MeanArea[" + sample + ']' for sample in a_full_samples_list], axis=1, inplace=True)
            return a_dataframe
        class_fold_change_df = add_fold_change_column(calculate_total_class_abundance(df, full_samples_list).copy(deep=True), control, experimental, full_samples_list)
        
        return class_fold_change_df
    
    def create_pathway_viz(self, a_class_fold_change_df, a_saturation_ratio_df):
        
        def initiate_plot():
            plt.rcParams["figure.figsize"] = [10, 10]
            fig, ax = plt.subplots()
            ax.set_title('Lipid Pathway Visualization', fontsize=20)
            ax.set_xlim([-25, 25])
            ax.set_ylim([-20, 30])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.gca().set_aspect('equal', adjustable='box')
            return fig, ax
        
        fig, ax = initiate_plot()
        
        def draw_one_circle(radius, x0, y0, color):
            return plt.Circle((x0, y0), radius, color = color, fill = False)
        
        def draw_all_circles():
            ax.add_patch(draw_one_circle(5, 0, 0, 'b'))
            ax.add_patch(draw_one_circle(2.5, -7.5 * math.cos(math.pi/6), -7.5 * math.cos(math.pi/3), 'b'))
            ax.add_patch(draw_one_circle(2.5, 7.5 * math.cos(math.pi/6), -7.5 * math.cos(math.pi/3), 'b'))
            ax.add_patch(draw_one_circle(2.5, 10 + 2.5 * math.cos(math.pi/4), 15 + 2.5 * math.sin(math.pi/4), 'b'))
            ax.add_patch(draw_one_circle(2.5, 12.5, 10, 'b'))
            ax.add_patch(draw_one_circle(2.5, 10 + 2.5 * math.cos(math.pi/4), 5 - 2.5 * math.sin(math.pi/4), 'b'))
            ax.add_patch(draw_one_circle(0.5, 0, 0, 'black'))
            ax.add_patch(draw_one_circle(0.5, 0, 5, 'black'))
            ax.add_patch(draw_one_circle(0.5, 0, 10, 'black'))
            ax.add_patch(draw_one_circle(0.5, -10, 10, 'black'))
            ax.add_patch(draw_one_circle(0.5, -10, 5, 'black'))
            ax.add_patch(draw_one_circle(0.5, 5*math.cos(math.pi/6), -5*math.sin(math.pi/6), 'black'))
            ax.add_patch(draw_one_circle(0.5, 10*math.cos(math.pi/6), -10*math.sin(math.pi/6), 'black'))
            ax.add_patch(draw_one_circle(0.5, -5*math.cos(math.pi/6), -5*math.sin(math.pi/6), 'black'))
            ax.add_patch(draw_one_circle(0.5, -10*math.cos(math.pi/6), -10*math.sin(math.pi/6), 'black'))
            ax.add_patch(draw_one_circle(0.5, 10, 15, 'black'))
            ax.add_patch(draw_one_circle(0.5, 10+5*math.cos(math.pi/4), 15+5*math.sin(math.pi/4), 'black'))
            ax.add_patch(draw_one_circle(0.5, 10, 10, 'black'))
            ax.add_patch(draw_one_circle(0.5, 15, 10, 'black'))
            ax.add_patch(draw_one_circle(0.5, 10, 5, 'black'))
            ax.add_patch(draw_one_circle(0.5, 10+5*math.cos(math.pi/4), 5-5*math.sin(math.pi/4), 'black'))
            return 
            
        draw_all_circles()
        
        def draw_connecting_lines():
            ax.plot([0, 0], [0, 20], c='b')
            ax.plot([0, -5], [15, 20], c='b')
            ax.plot([-5, -10], [20, 15], c='b')
            ax.plot([-10, -10], [15, 5], c='b')
            ax.plot([0, 5], [10, 10], c='b')
            ax.plot([5, 10], [10, 15], c='b')
            ax.plot([5, 10], [10, 10], c='b')
            ax.plot([5, 10], [10, 5], c='b')
            ax.plot([5*math.cos(math.pi/6), 10], [-5*math.sin(math.pi/6), 5], c='b')
            return 
        
        draw_connecting_lines()
        
        def add_text():
            ax.annotate('G3P', xy=(0, 20), xytext=(0, 23),arrowprops=dict(facecolor='black'), fontsize=15)
            ax.annotate('Fatty Acids', xy=(-5, 20), xytext=(-5, 25),arrowprops=dict(facecolor='black'), fontsize=15)
            ax.text(-3.5, 14, 'LPA', fontsize=15)
            ax.text(-2.5, 9.5, 'PA', fontsize=15)
            ax.text(-4, 5.5, 'DAG', fontsize=15)
            ax.text(-4, 0.5, 'TAG', fontsize=15)
            ax.text(-4, -2, 'PC', fontsize=15)
            ax.text(-12, -6.5, 'LPC', fontsize=15)
            ax.text(2.5, -2, 'PE', fontsize=15)
            ax.text(9, -6, 'LPE', fontsize=15)
            ax.text(-14, 15, 'LCBs', fontsize=15)
            ax.text(-13.5, 10, 'Cer', fontsize=15)
            ax.text(-13, 5, 'SM', fontsize=15)
            ax.text(2, 11, 'CDP-DAG', fontsize=15)
            ax.text(10.5, 15.5, 'PI', fontsize=15)
            ax.text(14, 19, 'LPI', fontsize=15)
            ax.text(10.5, 9.5, 'PG', fontsize=15)
            ax.text(15.5, 9.5, 'LPG', fontsize=15)
            ax.text(10.5, 4, 'PS', fontsize=15)
            ax.text(14.5, 1, 'LPS', fontsize=15)
            return 
        
        add_text()
        
        def create_pathway_dictionary(a_class_fold_change_df, a_saturation_ratio_df):
            pathway_classes_list = ['TG', 'DG', 'PA', 'LPA', 'LCB', 'Cer', 'SM', 'PE', 'LPE', 'PC', 'LPC', 'PI', 'LPI', 'CDP-DAG', 'PG', 'LPG', 'PS', 'LPS'] 
            pathway_class_fold_change_list = [0] * len(pathway_classes_list)
            pathway_class_saturation_ratio_list = [0] * len(pathway_classes_list)
            i = 0
            for lipid_class in pathway_classes_list:
                if lipid_class in a_class_fold_change_df.index:
                    pathway_class_fold_change_list[i] = float(a_class_fold_change_df.loc[lipid_class].values)
                    pathway_class_saturation_ratio_list[i] = float(a_saturation_ratio_df.loc[lipid_class].values)
                i += 1
            pathway_dict = {'class': pathway_classes_list, 'abundance ratio': pathway_class_fold_change_list, \
                            'saturated fatty acids ratio': pathway_class_saturation_ratio_list}
            return pathway_dict
        
        pathway_dict = create_pathway_dictionary(a_class_fold_change_df, a_saturation_ratio_df)
        
        @st.cache_data
        def prep_plot_inputs(a_pathway_dict):
            color_contour = a_pathway_dict['saturated fatty acids ratio']
            size = [50*ele**2 for ele in a_pathway_dict['abundance ratio']]
            return color_contour, size
        
        color_contour, size = prep_plot_inputs(pathway_dict) 
        
        def render_plot():
            points_x = [0, 0, 0, 0, -10, -10, -10, 5*math.cos(math.pi/6), 10*math.cos(math.pi/6), \
                        -5*math.cos(math.pi/6), -10*math.cos(math.pi/6), 10, 10+5*math.cos(math.pi/4), \
                        5, 10, 15, 10, 10+5*math.cos(math.pi/4)]
            points_y = [0, 5, 10, 15, 15, 10, 5, -5*math.sin(math.pi/6), -10*math.sin(math.pi/6), \
                        -5*math.sin(math.pi/6), -10*math.sin(math.pi/6), 15, 15+5*math.sin(math.pi/4), \
                        10, 10, 10, 5, 5-5*math.sin(math.pi/4)]
            points = ax.scatter(points_x, points_y, c=color_contour, s=size, cmap="plasma")
            cbar = fig.colorbar(points)
            cbar.set_label(label='Unsaturated <--- Saturatted Fatty Acids Ratio ---> Saturated', size=15)
            cbar.ax.tick_params(labelsize=15)
            return 
        
        render_plot()
        
        def display_plot():
            st.pyplot(fig)
            pathway_df = pd.DataFrame.from_dict(pathway_dict)
            pathway_df.set_index('class', inplace=True)
            st.write(pathway_df)
            csv_download = self.convert_df(pathway_df)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='pathway_df.csv',
                            mime='text/csv')
            return 
        
        display_plot()
        
        return 
    
   
    
    
    
    
    
    
    