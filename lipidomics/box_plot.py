import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

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
    def plot_missing_values(full_samples_list, zero_values_percent_list):
        """
        Creates an interactive horizontal bar chart of missing values percentage using Plotly.
        """
        # Calculate dynamic height based on number of samples
        dynamic_height = max(400, len(full_samples_list) * 25)
        # Add extra padding for title
        total_height = dynamic_height + 60  # Added 60px for title space
        
        fig = go.Figure()
        
        # Add horizontal bar chart
        fig.add_trace(go.Bar(
            y=full_samples_list,
            x=zero_values_percent_list,
            orientation='h',
            text=[f'{val:.1f}%' for val in zero_values_percent_list],
            textposition='outside',
            textfont=dict(size=12, color='black'),  # Set text color to black
            marker_color='lightblue',
            name=''  # Empty name to avoid legend
        ))
        
        # Update layout with improved spacing and black text
        fig.update_layout(
            title={
                'text': 'Missing Values Distribution',
                'font': {'size': 20, 'color': 'black'},  # Set title color to black
                'y': 0.98,  # Moved title up
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Percentage of Missing Values',
                'font': {'size': 14, 'color': 'black'},  # Set axis title color to black
                'standoff': 15  # Add small space between axis title and tick labels
            },
            yaxis_title={
                'text': 'Sample',
                'font': {'size': 14, 'color': 'black'},  # Set axis title color to black
                'standoff': 15  # Add small space between axis title and tick labels
            },
            yaxis={'tickfont': {'size': 12, 'color': 'black'}},  # Set y-axis tick labels to black
            xaxis={'tickfont': {'size': 12, 'color': 'black'}},  # Set x-axis tick labels to black
            margin={'l': 100, 'r': 100, 't': 60, 'b': 60},  # Moderately increased margins
            showlegend=False,
            height=total_height,  # Use new total height
            width=900,  # Slightly increased width to prevent text cutoff
            plot_bgcolor='white'  # White background
        )
        
        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig

    @staticmethod
    def plot_box_plot(mean_area_df, full_samples_list):
        """
        Creates an interactive box plot using Plotly.
        """
        # Prepare data
        log_transformed_data = [list(np.log10(mean_area_df[col][mean_area_df[col] > 0])) 
                              for col in mean_area_df.columns]
        
        fig = go.Figure()
        
        # Add box plots
        for i, data in enumerate(log_transformed_data):
            fig.add_trace(go.Box(
                y=data,
                name=full_samples_list[i],
                boxpoints='outliers',  # Show outliers
                marker_color='lightblue',
                line_color='darkblue',
                showlegend=False
            ))
        
        # Update layout with improved spacing and black text
        fig.update_layout(
            title={
                'text': 'Box Plot of Non-Zero Concentrations',
                'font': {'size': 20, 'color': 'black'},  # Set title color to black
                'y': 0.98,  # Moved title up
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Sample',
                'font': {'size': 14, 'color': 'black'},  # Set axis title color to black
                'standoff': 15  # Add small space between axis title and tick labels
            },
            yaxis_title={
                'text': 'log10(Concentration)',
                'font': {'size': 14, 'color': 'black'},  # Set axis title color to black
                'standoff': 15  # Add small space between axis title and tick labels
            },
            xaxis={
                'tickangle': 45,
                'tickfont': {'size': 12, 'color': 'black'},  # Set x-axis tick labels to black
                'title_standoff': 30  # Add moderate space for rotated labels
            },
            yaxis={'tickfont': {'size': 12, 'color': 'black'}},  # Set y-axis tick labels to black
            margin={'l': 100, 'r': 100, 't': 60, 'b': 100},  # Moderately increased margins
            height=600,
            width=900,  # Slightly increased width
            plot_bgcolor='white'  # White background
        )
        
        # Add gridlines
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig