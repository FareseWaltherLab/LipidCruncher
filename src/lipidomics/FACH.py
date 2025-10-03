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
        Assumes lipid names have been standardized to Class(chain_details) format
        where chain_details uses '_' as separator.
        
        Handles various formats:
        - Standard: 'PC(34:1)' or 'PC(16:0_18:1)'
        - Ether lipids: 'PC(O-38:4)', 'PE(P-36:2)'
        - Sphingoid bases: 'Cer(d18:1_24:0)', 'SM(d18:1_16:0)'
        - Oxidized: 'PC(16:0_18:1+O)', 'PE(18:0_20:4+2O)'
        - With modifications: 'LPC(18:1)(d7)'
        
        Returns (total_carbons, total_db) or (None, None) if unparsable.
        """
        # Extract the part inside the first set of parentheses (ignoring modifications in second parentheses)
        match = re.search(r'\(([^)]+)\)', lipid_name)
        if not match:
            return None, None
        composition = match.group(1)
        
        total_c = 0
        total_db = 0
        
        # Split by underscore (standardized separator)
        chains = composition.split('_')
        
        for chain in chains:
            # Remove common prefixes that don't contribute to carbon count:
            # - Ether lipids: O-, P-
            # - Sphingoid bases: d, t, m (e.g., d18:1, t18:0, m18:1)
            # Note: Keep the numbers after these prefixes
            chain_cleaned = re.sub(r'^[OPdtm]-?', '', chain)
            
            # Remove oxidation and hydroxyl suffixes that don't affect carbon:db count
            # Examples: +O, +2O, +3O
            chain_cleaned = re.sub(r'\+\d*O', '', chain_cleaned)
            
            # Remove chain identifiers like C in C24:0
            chain_cleaned = re.sub(r'^C', '', chain_cleaned)
            
            # Now extract carbon:db pattern
            chain_match = re.match(r'(\d+):(\d+)', chain_cleaned)
            if chain_match:
                total_c += int(chain_match.group(1))
                total_db += int(chain_match.group(2))
        
        return (total_c, total_db) if total_c > 0 else (None, None)

    @staticmethod
    def prepare_fach_data(df, experiment, selected_class, selected_conditions):
        """
        Prepare data for FACH: aggregate proportions per (C, DB) for each condition.
        Returns dict of {condition: pd.DataFrame with 'Carbon', 'DB', 'Proportion'}
        """
        import streamlit as st
        
        fach_data = {}
        
        # Filter to selected class
        class_df = df[df['ClassKey'] == selected_class].copy()
        
        if class_df.empty:
            return fach_data
        
        # Parse C and DB for each lipid
        parsed_results = []
        unparsable_lipids = []
        
        for lipid_name in class_df['LipidMolec']:
            result = FACH.parse_carbon_db(lipid_name)
            parsed_results.append(result)
            if result == (None, None):
                unparsable_lipids.append(lipid_name)
        
        # Report unparsable lipids if any
        if unparsable_lipids:
            st.warning(f"Could not parse {len(unparsable_lipids)} lipids in class '{selected_class}':")
            for lipid in unparsable_lipids[:10]:
                st.write(f"  - {lipid}")
            if len(unparsable_lipids) > 10:
                st.write(f"  ... and {len(unparsable_lipids) - 10} more")
        
        # Unpack results
        class_df['Carbon'], class_df['DB'] = zip(*parsed_results)
        
        # Convert None values and any remaining tuples to np.nan
        class_df['Carbon'] = class_df['Carbon'].apply(lambda x: np.nan if x is None else x)
        class_df['DB'] = class_df['DB'].apply(lambda x: np.nan if isinstance(x, tuple) or x is None else x)
        
        # Drop unparsable rows
        class_df = class_df.dropna(subset=['Carbon', 'DB'])
        
        if class_df.empty:
            st.error(f"No parsable lipids remaining for class '{selected_class}'")
            return fach_data
        
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
                agg_df['Proportion'] = (agg_df['Mean_Conc'] / total_conc) * 100
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
        including subplot titles. Ensures x-axis numbers are not rotated (tickangle=0).
        Uses a custom YlOrRd colorscale starting with yellowish at 0.
        Y-axis starts slightly below the minimum carbon chain length.
        Returns Plotly figure.
        """
        if not data_dict:
            return None
    
        # Find global min/max for consistent color scale
        all_proportions = pd.concat([df['Proportion'] for df in data_dict.values()])
        vmin, vmax = all_proportions.min(), all_proportions.max()
    
        # Find global max for DB and min/max for Carbon to set consistent axes
        all_db = pd.concat([df['DB'] for df in data_dict.values()])
        all_carbon = pd.concat([df['Carbon'] for df in data_dict.values()])
        max_db = int(all_db.max()) if not all_db.empty else 9
        min_carbon = int(all_carbon.min()) if not all_carbon.empty else 0
        max_carbon = int(all_carbon.max()) if not all_carbon.empty else 50
    
        # Create subplots: one per condition
        n_conditions = len(data_dict)
        fig = make_subplots(
            rows=1,
            cols=n_conditions,
            subplot_titles=list(data_dict.keys()),
            shared_yaxes=True
        )
    
        # Define custom YlOrRd colorscale (yellow at 0, progressing to red)
        custom_colorscale = [
            [0.0, 'rgb(255, 255, 204)'],  # Light yellow for 0%
            [0.2, 'rgb(255, 204, 153)'],  # Light orange
            [0.4, 'rgb(255, 153, 102)'],  # Orange
            [0.6, 'rgb(255, 102, 51)'],   # Darker orange
            [0.8, 'rgb(204, 51, 0)'],     # Red-orange
            [1.0, 'rgb(153, 0, 0)']       # Dark red for max
        ]
    
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
                colorscale=custom_colorscale,
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
    
            # Update x-axis to show all integers from 0 to max_db with no rotation
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(max_db + 1)),
                ticktext=list(range(max_db + 1)),
                tickangle=0,
                title_text='Double Bonds',
                titlefont=dict(color='black'),
                tickfont=dict(color='black'),
                range=[-0.5, max_db + 1.5],
                row=1,
                col=col_idx
            )
    
            # Update y-axis - start slightly below minimum carbon chain length
            y_range_min = min_carbon - 1 if min_carbon > 0 else 0
            if col_idx == 1:
                fig.update_yaxes(
                    title_text='Carbon Chain Length',
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black'),
                    range=[y_range_min, max_carbon + 2.5],
                    row=1,
                    col=col_idx
                )
            else:
                fig.update_yaxes(
                    tickfont=dict(color='black'),
                    range=[y_range_min, max_carbon + 2.5],
                    row=1,
                    col=col_idx
                )
    
            col_idx += 1
    
        # Update subplot titles to black
        for annotation in fig.layout.annotations:
            if annotation.text in data_dict.keys():
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