import pandas as pd
import plotly.express as px

class InternalStandardsPlotter:
    @staticmethod
    def create_consistency_plots(intsta_df, samples):
        """
        Create separate stacked bar plots for each internal standard class.
        Stacks by individual standards (LipidMolec) if multiple in a class.
        
        Args:
            intsta_df (pd.DataFrame): Internal standards DataFrame
            samples (list): List of sample names to display (in desired order)
            
        Returns:
            list: List of plotly figures, one per class
        """
        if intsta_df.empty:
            return []
            
        intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
        
        # Filter to only columns for selected samples
        valid_intensity_cols = [f'intensity[{s}]' for s in samples if f'intensity[{s}]' in intensity_cols]
        
        if not valid_intensity_cols:
            return []
            
        figs = []
        
        # Get unique classes sorted
        classes = sorted(intsta_df['ClassKey'].unique())
        
        for class_key in classes:
            class_df = intsta_df[intsta_df['ClassKey'] == class_key]
            
            if class_df.empty:
                continue
            
            # Melt the dataframe for plotting (only selected columns)
            melted_df = pd.melt(
                class_df,
                id_vars=['LipidMolec'],
                value_vars=valid_intensity_cols,
                var_name='Sample_Column',
                value_name='Intensity'
            )
            
            # Extract sample name from column
            melted_df['Sample'] = melted_df['Sample_Column'].str.replace('intensity[', '', regex=False).str.replace(']', '', regex=False)
            
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
            
            # Set stacked mode and layout
            fig.update_layout(
                barmode='stack',
                xaxis_title='Samples',
                yaxis_title='Raw Intensity',
                legend_title='Internal Standard',
                bargap=0.2,
                xaxis={'categoryorder': 'array', 'categoryarray': samples},
                font=dict(color='black'),  # Global font color
                legend_title_font=dict(color='black')  # Explicitly set legend title font color
            )
            
            # Explicitly set colors for all text elements
            fig.update_xaxes(
                tickfont=dict(color='black'),
                title_font=dict(color='black')
            )
            fig.update_yaxes(
                tickfont=dict(color='black'),
                title_font=dict(color='black')
            )
            fig.update_layout(
                legend=dict(
                    font=dict(color='black')
                ),
                title_font=dict(color='black')
            )
            
            figs.append(fig)
        
        return figs