import re
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class FACH:
    """
    A class for generating Fatty Acid Composition Heatmaps (FACH) from lipidomics data.
    This class processes lipidomics data to create heatmaps showing the relative abundance
    of lipids within a selected class, grouped by total carbon chain length and number of
    double bonds.
    """
    @staticmethod
    def parse_carbon_db(lipid_name):
        """
        Parse total carbon atoms and double bonds from lipid name.
        Handles formats like 'PC(34:1)' or 'PC(16:0_18:1)'.
        Returns (total_carbons, total_db) or (None, None) if unparsable.
        """
        # Extract the part inside parentheses
        match = re.search(r'\((.*?)\)', lipid_name)
        if not match:
            return None, None
        composition = match.group(1)
        
        total_c = 0
        total_db = 0
        
        # Split by '_' for detailed or use single for total
        chains = composition.split('_') if '_' in composition else [composition]
        
        for chain in chains:
            chain_match = re.match(r'(\d+):(\d+)', chain)
            if chain_match:
                total_c += int(chain_match.group(1))
                total_db += int(chain_match.group(2))
        
        return total_c, total_db if total_c > 0 else (None, None)

    @staticmethod
    def prepare_fach_data(df, experiment, selected_class, selected_conditions):
        """
        Prepare data for FACH: aggregate proportions per (C, DB) for each condition.
        Returns dict of {condition: pd.DataFrame with 'Carbon', 'DB', 'Proportion'}
        """
        fach_data = {}
        
        # Filter to selected class
        class_df = df[df['ClassKey'] == selected_class].copy()
        
        if class_df.empty:
            return fach_data
        
        # Parse C and DB for each lipid
        class_df['Carbon'], class_df['DB'] = zip(*class_df['LipidMolec'].apply(FACH.parse_carbon_db))
        class_df = class_df.dropna(subset=['Carbon', 'DB'])  # Drop unparsable
        
        for condition in selected_conditions:
            cond_idx = experiment.conditions_list.index(condition)
            samples = experiment.individual_samples_list[cond_idx]
            conc_cols = [f'concentration[{s}]' for s in samples if f'concentration[{s}]' in class_df.columns]
            
            if not conc_cols:
                continue
            
            # Compute mean concentration per lipid
            class_df['Mean_Conc'] = class_df[conc_cols].mean(axis=1)
            
            # Aggregate by (Carbon, DB)
            agg_df = class_df.groupby(['Carbon', 'DB'])['Mean_Conc'].sum().reset_index()
            
            # Compute proportions
            total_conc = agg_df['Mean_Conc'].sum()
            if total_conc > 0:
                agg_df['Proportion'] = (agg_df['Mean_Conc'] / total_conc) * 100  # As percentage
            else:
                agg_df['Proportion'] = 0
            
            fach_data[condition] = agg_df[['Carbon', 'DB', 'Proportion']]
        
        return fach_data

    @staticmethod
    def create_fach_heatmap(data_dict):
        """
        Create side-by-side heatmaps for each condition with average DB and carbon chain length lines.
        Includes annotations for both lines positioned outside the heatmap grid to avoid overlap,
        with a semi-transparent background for clarity, and uses black text for all plot elements,
        including subplot titles.
        Returns Plotly figure.
        """
        if not data_dict:
            return None
        
        # Find global min/max for consistent color scale
        all_proportions = pd.concat([df['Proportion'] for df in data_dict.values()])
        vmin, vmax = all_proportions.min(), all_proportions.max()
        
        # Find global max for DB and Carbon to set consistent axes
        all_db = pd.concat([df['DB'] for df in data_dict.values()])
        all_carbon = pd.concat([df['Carbon'] for df in data_dict.values()])
        max_db = int(all_db.max()) if not all_db.empty else 9
        max_carbon = int(all_carbon.max()) if not all_carbon.empty else 50
        
        # Create subplots: one per condition
        n_conditions = len(data_dict)
        fig = make_subplots(
            rows=1,
            cols=n_conditions,
            subplot_titles=list(data_dict.keys()),
            shared_yaxes=True
        )
        
        col_idx = 1
        for condition, df in data_dict.items():
            # Calculate average DB and Carbon, weighted by Proportion
            if not df.empty and df['Proportion'].sum() > 0:
                avg_db = np.average(df['DB'], weights=df['Proportion'])
                avg_carbon = np.average(df['Carbon'], weights=df['Proportion'])
            else:
                avg_db = 0
                avg_carbon = 0
            
            heatmap = go.Heatmap(
                x=df['DB'],
                y=df['Carbon'],
                z=df['Proportion'],
                colorscale='Plasma',
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title='Proportion (%)', titlefont=dict(color='black'), tickfont=dict(color='black'))
            )
            fig.add_trace(heatmap, row=1, col=col_idx)
            
            # Add vertical dashed line for average DB
            fig.add_vline(
                x=avg_db,
                line=dict(color='black', dash='dash', width=2),
                row=1,
                col=col_idx
            )
            # Add annotation for average DB, positioned above the heatmap
            fig.add_annotation(
                x=avg_db,
                y=max_carbon + 2,
                text=f'Avg DB: {avg_db:.1f}',
                showarrow=False,
                font=dict(size=14, color='black'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                xref=f'x{col_idx}',
                yref=f'y{col_idx}',
                row=1,
                col=col_idx
            )
            
            # Add horizontal dashed line for average Carbon
            fig.add_hline(
                y=avg_carbon,
                line=dict(color='black', dash='dash', width=2),
                row=1,
                col=col_idx
            )
            # Add annotation for average Carbon, positioned to the right of the heatmap
            fig.add_annotation(
                x=max_db + 1,
                y=avg_carbon,
                text=f'Avg C: {avg_carbon:.1f}',
                showarrow=False,
                font=dict(size=14, color='black'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                xref=f'x{col_idx}',
                yref=f'y{col_idx}',
                row=1,
                col=col_idx
            )
            
            # Update x-axis to show all integers from 0 to max_db
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(max_db + 1)),
                ticktext=list(range(max_db + 1)),
                title_text='Double Bonds',
                titlefont=dict(color='black'),
                tickfont=dict(color='black'),
                range=[-0.5, max_db + 1.5],
                row=1,
                col=col_idx
            )
            
            # Update y-axis for the first subplot only to avoid redundancy
            if col_idx == 1:
                fig.update_yaxes(
                    title_text='Carbon Chain Length',
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black'),
                    range=[min(df['Carbon'].min() - 0.5, 0) if not df.empty else 0, max_carbon + 2.5],
                    row=1,
                    col=col_idx
                )
            else:
                fig.update_yaxes(
                    tickfont=dict(color='black'),
                    range=[min(df['Carbon'].min() - 0.5, 0) if not df.empty else 0, max_carbon + 2.5],
                    row=1,
                    col=col_idx
                )
            
            col_idx += 1
        
        # Update subplot titles to black
        for annotation in fig.layout.annotations:
            if annotation.text in data_dict.keys():  # Check if annotation is a subplot title
                annotation.font = dict(color='black', size=12)
        
        fig.update_layout(
            title=f'Fatty Acid Composition Heatmaps',
            titlefont=dict(color='black'),
            height=600,
            width=300 * n_conditions,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig