import pandas as pd
import plotly.express as px

class InternalStandardsPlotter:
    @staticmethod
    def create_consistency_plot(intsta_df, samples):
        """
        Create a stacked bar plot showing internal standards intensity by sample.
        
        Args:
            intsta_df (pd.DataFrame): Internal standards DataFrame
            samples (list): List of sample names
            
        Returns:
            plotly.graph_objs.Figure or None: The plot figure if successful
        """
        if intsta_df.empty:
            return None
            
        intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
        
        # Map intensity columns to sample names
        col_to_sample = {f'intensity[{sample}]': sample for sample in samples}
        valid_intensity_cols = [col for col in intensity_cols if col in col_to_sample]
        
        if not valid_intensity_cols:
            return None
            
        # Melt the dataframe for plotting
        melted_df = pd.melt(
            intsta_df, 
            id_vars=['ClassKey', 'LipidMolec'], 
            value_vars=valid_intensity_cols,
            var_name='Sample_Column', 
            value_name='Intensity'
        )
        
        # Map column names to sample names
        melted_df['Sample'] = melted_df['Sample_Column'].map(col_to_sample)
        
        # Group by Sample and ClassKey, sum intensities
        grouped_df = melted_df.groupby(['Sample', 'ClassKey'])['Intensity'].sum().reset_index()
        
        # Create stacked bar plot
        fig = px.bar(
            grouped_df, 
            x='Sample', 
            y='Intensity', 
            color='ClassKey',
            title='Internal Standards Intensity by Sample (Stacked by Class)',
            labels={'Intensity': 'Raw Intensity', 'Sample': 'Sample'},
            category_orders={'Sample': samples},  # Preserve sample order
            height=500
        )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Samples',
            yaxis_title='Total Raw Intensity (Stacked by Class)',
            legend_title='Internal Standard Class',
            bargap=0.2,
            xaxis={'categoryorder': 'array', 'categoryarray': samples}
        )
        
        return fig