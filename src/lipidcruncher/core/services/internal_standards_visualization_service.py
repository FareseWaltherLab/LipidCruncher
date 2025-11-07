"""
Internal Standards Visualization Service

Service for creating internal standards consistency visualizations.
Generates stacked bar plots showing raw intensity values across samples.

This service has no UI dependencies and can be used in any context.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional


class InternalStandardsVisualizationService:
    """
    Service for creating internal standards consistency plots.
    
    These plots help users verify sample preparation consistency,
    detect instrument performance issues, and validate internal
    standards quality before normalization.
    """
    
    @staticmethod
    def create_consistency_plots(
        intsta_df: pd.DataFrame,
        samples: List[str]
    ) -> List[go.Figure]:
        """
        Create separate stacked bar plots for each internal standard class.
        Stacks by individual standards (LipidMolec) if multiple in a class.
        
        Args:
            intsta_df: Internal standards DataFrame with columns:
                       - LipidMolec: Name of the internal standard
                       - ClassKey: Lipid class
                       - intensity[s1], intensity[s2], etc.: Sample intensities
            samples: List of sample names to display (in desired order)
            
        Returns:
            List of plotly figures, one per class. Empty list if no plots generated.
            
        Example:
            >>> service = InternalStandardsVisualizationService()
            >>> figs = service.create_consistency_plots(intsta_df, ['s1', 's2', 's3'])
            >>> if figs:
            >>>     for fig in figs:
            >>>         fig.show()
        
        Notes:
            - Returns empty list if intsta_df is empty
            - Returns empty list if no valid sample columns found
            - Each figure represents one lipid class
            - Bars are stacked if multiple standards exist per class
        """
        # Validation: Check for empty DataFrame
        if intsta_df is None or intsta_df.empty:
            return []
        
        # Validation: Check for required columns
        if 'LipidMolec' not in intsta_df.columns or 'ClassKey' not in intsta_df.columns:
            return []
        
        # Validation: Check for samples
        if not samples:
            return []
        
        # Get all intensity columns
        intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
        
        # Filter to only columns for selected samples
        valid_intensity_cols = [f'intensity[{s}]' for s in samples if f'intensity[{s}]' in intensity_cols]
        
        if not valid_intensity_cols:
            return []
        
        # Generate plots for each class
        figs = []
        classes = sorted(intsta_df['ClassKey'].unique())
        
        for class_key in classes:
            fig = InternalStandardsVisualizationService._create_class_plot(
                intsta_df,
                class_key,
                samples,
                valid_intensity_cols
            )
            if fig is not None:
                figs.append(fig)
        
        return figs
    
    @staticmethod
    def _create_class_plot(
        intsta_df: pd.DataFrame,
        class_key: str,
        samples: List[str],
        valid_intensity_cols: List[str]
    ) -> Optional[go.Figure]:
        """
        Create a single plot for one lipid class.
        
        Args:
            intsta_df: Internal standards DataFrame
            class_key: Lipid class to plot
            samples: Sample names in desired order
            valid_intensity_cols: Pre-filtered intensity columns
            
        Returns:
            Plotly figure or None if no data for this class
        """
        # Filter to this class
        class_df = intsta_df[intsta_df['ClassKey'] == class_key]
        
        if class_df.empty:
            return None
        
        # Melt the dataframe for plotting (only selected columns)
        melted_df = pd.melt(
            class_df,
            id_vars=['LipidMolec'],
            value_vars=valid_intensity_cols,
            var_name='Sample_Column',
            value_name='Intensity'
        )
        
        # Extract sample name from column (remove 'intensity[' and ']')
        melted_df['Sample'] = (
            melted_df['Sample_Column']
            .str.replace('intensity[', '', regex=False)
            .str.replace(']', '', regex=False)
        )
        
        # Create stacked bar plot (stacks if multiple LipidMolec)
        fig = px.bar(
            melted_df,
            x='Sample',
            y='Intensity',
            color='LipidMolec',
            title=f'Internal Standards Intensity for {class_key}',
            labels={'Intensity': 'Raw Intensity', 'Sample': 'Sample'},
            category_orders={'Sample': samples},
            height=500
        )
        
        # Configure layout for stacking and styling
        fig.update_layout(
            barmode='stack',
            xaxis_title='Samples',
            yaxis_title='Raw Intensity',
            legend_title='Internal Standard',
            bargap=0.2,
            xaxis={'categoryorder': 'array', 'categoryarray': samples},
            font=dict(color='black'),
            legend_title_font=dict(color='black')
        )
        
        # Explicitly set colors for all text elements (ensures visibility)
        fig.update_xaxes(
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        )
        fig.update_yaxes(
            tickfont=dict(color='black'),
            title_font=dict(color='black')
        )
        fig.update_layout(
            legend=dict(font=dict(color='black')),
            title_font=dict(color='black')
        )
        
        return fig
    
    @staticmethod
    def prepare_csv_data(
        intsta_df: pd.DataFrame,
        class_key: str,
        samples: List[str]
    ) -> pd.DataFrame:
        """
        Prepare CSV export data for a specific class.
        
        Converts wide-format intensity data to long format suitable
        for CSV export and external analysis.
        
        Args:
            intsta_df: Internal standards DataFrame
            class_key: Lipid class to filter
            samples: List of samples to include
            
        Returns:
            Melted DataFrame with columns:
            - LipidMolec: Standard name
            - ClassKey: Lipid class
            - Sample: Sample name
            - Intensity: Raw intensity value
            
        Example:
            >>> csv_df = InternalStandardsVisualizationService.prepare_csv_data(
            ...     intsta_df, 'PC', ['s1', 's2']
            ... )
            >>> csv_df.to_csv('pc_standards.csv', index=False)
        """
        # Filter to the requested class
        class_df = intsta_df[intsta_df['ClassKey'] == class_key].copy()
        
        # Get intensity columns for requested samples
        intensity_cols = [
            f'intensity[{s}]' for s in samples 
            if f'intensity[{s}]' in class_df.columns
        ]
        
        # Melt to long format
        melted_df = pd.melt(
            class_df,
            id_vars=['LipidMolec', 'ClassKey'],
            value_vars=intensity_cols,
            var_name='Sample_Column',
            value_name='Intensity'
        )
        
        # Extract clean sample names
        melted_df['Sample'] = (
            melted_df['Sample_Column']
            .str.replace('intensity[', '', regex=False)
            .str.replace(']', '', regex=False)
        )
        
        # Return only the relevant columns in logical order
        return melted_df[['LipidMolec', 'ClassKey', 'Sample', 'Intensity']]