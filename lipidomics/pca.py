import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import numpy as np 
from scipy.stats import chi2

class PCAAnalysis:
    @staticmethod
    def run_pca(_df, full_samples_list):
        if isinstance(_df, tuple):
            df = _df[0]
        else:
            df = _df
        available_samples = [sample for sample in full_samples_list if f'MeanArea[{sample}]' in df.columns]
        mean_area_df = df[['MeanArea[' + sample + ']' for sample in available_samples]].T
        scaled_data = StandardScaler().fit_transform(mean_area_df)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        return pc_df, [f'PC{i+1} ({var:.0%})' for i, var in enumerate(explained_variance)], available_samples

    @staticmethod
    @st.cache_data(ttl=3600)
    def generate_color_mapping(conditions):
        unique_conditions = sorted(set(conditions))
        return {condition: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                for i, condition in enumerate(unique_conditions)}

    @staticmethod
    def create_scatter_plot(df, pc_names, color_mapping):
        fig = go.Figure()
        
        for condition in df['Condition'].unique():
            cond_df = df[df['Condition'] == condition]
            fig.add_trace(go.Scatter(
                x=cond_df['PC1'],
                y=cond_df['PC2'],
                mode='markers',
                name=condition,
                marker=dict(color=color_mapping[condition], size=5),
                text=cond_df['Sample'],
                hovertemplate='<b>Sample:</b> %{text}<br>' +
                              '<b>PC1:</b> %{x:.4f}<br>' +
                              '<b>PC2:</b> %{y:.4f}<br>' +
                              '<extra></extra>'
            ))
            
            # Add confidence ellipse
            x = cond_df['PC1']
            y = cond_df['PC2']
            PCAAnalysis.add_confidence_ellipse(fig, x, y, color_mapping[condition], condition)

        fig.update_layout(
            title=dict(text="PCA Plot", font=dict(size=24, color='black')),
            xaxis_title=dict(text=pc_names[0], font=dict(size=18, color='black')),
            yaxis_title=dict(text=pc_names[1], font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black')),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            margin=dict(t=50, r=50, b=50, l=50)
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        return fig

    @staticmethod
    def add_confidence_ellipse(fig, x, y, color, name, n_std=2.0):
        if len(x) < 2:  # We need at least two points to define an ellipse
            return

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        
        # Eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(cov)
        
        # Sort eigenvalues in descending order
        order = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]
        
        # Calculate angles
        angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
        
        # Width and height of ellipse
        width, height = 2 * n_std * np.sqrt(eig_vals)
        
        # Generate points on a circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle = np.array([np.cos(theta), np.sin(theta)])
        
        # Transform circle to ellipse
        ellipse = np.dot(eig_vecs, circle * np.array([[width/2], [height/2]]))
        
        # Shift to center
        center = np.mean(list(zip(x, y)), axis=0)
        ellipse[0] += center[0]
        ellipse[1] += center[1]
        
        fig.add_trace(go.Scatter(
            x=ellipse[0],
            y=ellipse[1],
            mode='lines',
            line=dict(color=color, width=1),
            name=f'{name} Confidence Ellipse',
            showlegend=False
        ))

    @staticmethod
    def plot_pca(df, full_samples_list, extensive_conditions_list):
        pc_df, pc_names, available_samples = PCAAnalysis.run_pca(df, full_samples_list)
        pc_df['Sample'] = available_samples
        pc_df['Condition'] = [extensive_conditions_list[full_samples_list.index(sample)] for sample in available_samples]
        color_mapping = PCAAnalysis.generate_color_mapping(pc_df['Condition'])
        plot = PCAAnalysis.create_scatter_plot(pc_df, pc_names, color_mapping)
        return plot, pc_df