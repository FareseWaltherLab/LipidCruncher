import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Color palette for conditions (colorblind-friendly)
CONDITION_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

class BoxPlot:
    """
    Class for generating and displaying box plots for lipidomics data analysis using Plotly.
    """
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def create_mean_area_df(df, full_samples_list):
        """
        Creates a DataFrame containing only the 'concentration' columns from the provided DataFrame.
        """
        concentration_cols = [f'concentration[{sample}]' for sample in full_samples_list]
        return df[concentration_cols]

    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_missing_values_percentage(mean_area_df):
        """
        Calculates the percentage of missing values for each sample in the experiment.
        """
        return [len(mean_area_df[mean_area_df[col] == 0]) / len(mean_area_df) * 100 
                for col in mean_area_df.columns]

    @staticmethod
    def _get_sample_colors(full_samples_list, conditions_list, individual_samples_list):
        """
        Creates a list of colors for each sample based on its condition.
        
        Args:
            full_samples_list: List of all sample names
            conditions_list: List of condition names
            individual_samples_list: List of lists, samples grouped by condition
            
        Returns:
            tuple: (colors list, sample_to_condition dict, condition_to_color dict)
        """
        # Build sample to condition mapping
        sample_to_condition = {}
        for cond_idx, samples in enumerate(individual_samples_list):
            for sample in samples:
                sample_to_condition[sample] = conditions_list[cond_idx]
        
        # Assign colors
        condition_to_color = {cond: CONDITION_COLORS[i % len(CONDITION_COLORS)] 
                            for i, cond in enumerate(conditions_list)}
        
        colors = []
        for sample in full_samples_list:
            condition = sample_to_condition.get(sample, conditions_list[0])
            colors.append(condition_to_color[condition])
        
        return colors, sample_to_condition, condition_to_color

    @staticmethod
    def plot_missing_values(full_samples_list, zero_values_percent_list, 
                           conditions_list=None, individual_samples_list=None):
        """
        Creates an interactive horizontal bar chart of missing values percentage using Plotly.
        Bars are colored by condition if condition information is provided.
        """
        # Calculate dynamic height based on number of samples
        dynamic_height = max(400, len(full_samples_list) * 25)
        total_height = dynamic_height + 60

        fig = go.Figure()
        
        # Get colors if condition info provided
        if conditions_list and individual_samples_list:
            colors, sample_to_condition, condition_to_color = BoxPlot._get_sample_colors(
                full_samples_list, conditions_list, individual_samples_list
            )
            
            # Add bars (no legend on individual bars)
            for i, sample in enumerate(full_samples_list):
                fig.add_trace(go.Bar(
                    y=[sample],
                    x=[zero_values_percent_list[i]],
                    orientation='h',
                    text=[f'{zero_values_percent_list[i]:.1f}%'],
                    textposition='outside',
                    textfont=dict(size=12, color='black'),
                    marker_color=colors[i],
                    showlegend=False
                ))
            
            # Add invisible traces for legend (one per condition)
            for condition in conditions_list:
                fig.add_trace(go.Bar(
                    y=[None],
                    x=[None],
                    orientation='h',
                    marker_color=condition_to_color[condition],
                    name=condition,
                    showlegend=True
                ))
        else:
            # Original behavior - single color
            fig.add_trace(go.Bar(
                y=full_samples_list,
                x=zero_values_percent_list,
                orientation='h',
                text=[f'{val:.1f}%' for val in zero_values_percent_list],
                textposition='outside',
                textfont=dict(size=12, color='black'),
                marker_color='lightblue',
                name='',
                showlegend=False
            ))
        
        fig.update_layout(
            title={
                'text': 'Missing Values Distribution',
                'font': {'size': 20, 'color': 'black'},
                'y': 0.99,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Percentage of Missing Values',
                'font': {'size': 14, 'color': 'black'},
                'standoff': 15
            },
            yaxis_title={
                'text': 'Sample',
                'font': {'size': 14, 'color': 'black'},
                'standoff': 15
            },
            yaxis={'tickfont': {'size': 12, 'color': 'black'}},
            xaxis={'tickfont': {'size': 12, 'color': 'black'}},
            margin={'l': 100, 'r': 100, 't': 60, 'b': 120},
            showlegend=bool(conditions_list),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.18,
                xanchor='center',
                x=0.5
            ),
            height=total_height + 80,
            width=900,
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig

    @staticmethod
    def plot_box_plot(mean_area_df, full_samples_list,
                     conditions_list=None, individual_samples_list=None):
        """
        Creates an interactive box plot using Plotly.
        Boxes are colored by condition if condition information is provided.
        """
        # Prepare data
        log_transformed_data = [list(np.log10(mean_area_df[col][mean_area_df[col] > 0])) 
                              for col in mean_area_df.columns]
        
        fig = go.Figure()
        
        # Get colors if condition info provided
        if conditions_list and individual_samples_list:
            colors, sample_to_condition, condition_to_color = BoxPlot._get_sample_colors(
                full_samples_list, conditions_list, individual_samples_list
            )
            
            # Add box plots (no legend on individual boxes)
            for i, data in enumerate(log_transformed_data):
                sample = full_samples_list[i]
                
                fig.add_trace(go.Box(
                    y=data,
                    name=sample,
                    boxpoints='outliers',
                    marker_color=colors[i],
                    line_color=colors[i],
                    showlegend=False
                ))
            
            # Add invisible traces for legend (one per condition)
            for condition in conditions_list:
                fig.add_trace(go.Box(
                    y=[None],
                    name=condition,
                    marker_color=condition_to_color[condition],
                    line_color=condition_to_color[condition],
                    showlegend=True
                ))
        else:
            # Original behavior - single color
            for i, data in enumerate(log_transformed_data):
                fig.add_trace(go.Box(
                    y=data,
                    name=full_samples_list[i],
                    boxpoints='outliers',
                    marker_color='lightblue',
                    line_color='darkblue',
                    showlegend=False
                ))
        
        fig.update_layout(
            title={
                'text': 'Box Plot of Non-Zero Concentrations',
                'font': {'size': 20, 'color': 'black'},
                'y': 0.99,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Sample',
                'font': {'size': 14, 'color': 'black'},
                'standoff': 15
            },
            yaxis_title={
                'text': 'log10(Concentration)',
                'font': {'size': 14, 'color': 'black'},
                'standoff': 15
            },
            xaxis={
                'tickangle': 45,
                'tickfont': {'size': 12, 'color': 'black'},
                'title_standoff': 30
            },
            yaxis={'tickfont': {'size': 12, 'color': 'black'}},
            margin={'l': 100, 'r': 100, 't': 60, 'b': 120},
            showlegend=bool(conditions_list),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.25,
                xanchor='center',
                x=0.5
            ),
            height=700,
            width=900,
            plot_bgcolor='white'
        )
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig