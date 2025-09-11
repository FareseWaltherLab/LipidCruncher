import re
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

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
        match = re.search(r'\((.*?)\)', lipid_name)
        if not match:
            return None, None
        composition = match.group(1)
       
        total_c = 0
        total_db = 0
       
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
       
        class_df = df[df['ClassKey'] == selected_class].copy()
       
        if class_df.empty:
            return fach_data
       
        # Parse C and DB for each lipid
        class_df['Carbon'], class_df['DB'] = zip(*class_df['LipidMolec'].apply(FACH.parse_carbon_db))
        class_df = class_df.dropna(subset=['Carbon', 'DB']) # Drop unparsable
       
        for condition in selected_conditions:
            cond_idx = experiment.conditions_list.index(condition)
            samples = experiment.individual_samples_list[cond_idx]
            conc_cols = [f'concentration[{s}]' for s in samples if f'concentration[{s}]' in class_df.columns]
           
            if not conc_cols:
                continue
           
            class_df['Mean_Conc'] = class_df[conc_cols].mean(axis=1)
           
            agg_df = class_df.groupby(['Carbon', 'DB'])['Mean_Conc'].sum().reset_index()
           
            total_conc = agg_df['Mean_Conc'].sum()
            if total_conc > 0:
                agg_df['Proportion'] = (agg_df['Mean_Conc'] / total_conc) * 100 # As percentage
            else:
                agg_df['Proportion'] = 0
           
            fach_data[condition] = agg_df[['Carbon', 'DB', 'Proportion']]
       
        return fach_data

    @staticmethod
    def create_fach_heatmap(data_dict):
        """
        Create side-by-side heatmaps for each condition, using a pivot table to avoid interpolation.
        Returns Plotly figure.
        """
        if not data_dict:
            return None
       
        # Find global min/max for consistent color scale
        all_proportions = pd.concat([df['Proportion'] for df in data_dict.values()])
        vmin, vmax = all_proportions.min(), all_proportions.max()
       
        # Create subplots: one per condition
        n_conditions = len(data_dict)
        fig = make_subplots(rows=1, cols=n_conditions, subplot_titles=list(data_dict.keys()), shared_yaxes=True)
       
        col_idx = 1
        for condition, df in data_dict.items():
            if df.empty:
                continue
            # Pivot the data to create a matrix for the heatmap
            pivot_df = df.pivot(index='Carbon', columns='DB', values='Proportion').fillna(np.nan)
            # Get all unique carbons and sort them, convert to strings
            all_carbons = [str(c) for c in sorted(pivot_df.index.unique())]
            # Get all unique DBs and sort them, convert to strings
            all_dbs = [str(d) for d in sorted(pivot_df.columns.unique())]
            # Create a full matrix with zeros instead of NaNs for hover consistency
            z = np.zeros((len(all_carbons), len(all_dbs)))
            for i, carbon in enumerate(all_carbons):
                for j, db in enumerate(all_dbs):
                    if int(carbon) in pivot_df.index and int(db) in pivot_df.columns:
                        z[i, j] = pivot_df.loc[int(carbon), int(db)]
            # Add heatmap trace with custom hover template
            heatmap = go.Heatmap(
                x=all_dbs,
                y=all_carbons,
                z=z,
                colorscale='YlOrRd',
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title='Proportion (%)', titlefont=dict(color='black'), tickfont=dict(color='black')),
                zauto=False, # Disable auto-scaling to respect zeros
                hoverongaps=False, # Prevent hovering over gaps
                hovertemplate='Carbon: %{y}<br>Double Bonds: %{x}<br>Proportion: %{z:.2f}%<extra></extra>'
            )
            fig.add_trace(heatmap, row=1, col=col_idx)
            # Update axes for each subplot
            fig.update_xaxes(
                title_text='Double Bonds',
                title_font=dict(color='black'),
                tickfont=dict(color='black'),
                tickvals=all_dbs,
                ticktext=all_dbs,
                row=1,
                col=col_idx
            )
            fig.update_yaxes(
                title_text='Carbon Chain Length',
                type='category',
                title_font=dict(color='black'),
                tickfont=dict(color='black'),
                tickvals=all_carbons,
                ticktext=all_carbons,
                row=1,
                col=col_idx
            )
            col_idx += 1
       
        # Update layout
        fig.update_layout(
            title=f'Fatty Acid Composition Heatmaps',
            title_font=dict(color='black'),
            height=600,
            width=300 * n_conditions,
            annotations=[dict(
                font=dict(color='black')
            ) for annotation in fig.layout.annotations]
        )
        return fig