import pandas as pd
import plotly.express as px

class InternalStandardsPlotter:
    @staticmethod
    def create_consistency_plots(intsta_df, samples):
        """
        Create bar plots for each internal standard class.
        Multiple standards in the same class get separate subplot rows with different colors.
        
        Args:
            intsta_df (pd.DataFrame): Internal standards DataFrame
            samples (list): List of sample names to display (in desired order)
            
        Returns:
            list: List of plotly figures, one per class
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        if intsta_df.empty:
            return []
            
        intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
        
        # Filter to only columns for selected samples
        valid_intensity_cols = [f'intensity[{s}]' for s in samples if f'intensity[{s}]' in intensity_cols]
        
        if not valid_intensity_cols:
            return []
            
        figs = []
        
        # Color palette for multiple standards in same class
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
        
        # Get unique classes sorted
        classes = sorted(intsta_df['ClassKey'].unique())
        
        for class_key in classes:
            class_df = intsta_df[intsta_df['ClassKey'] == class_key]
            
            if class_df.empty:
                continue
            
            standards = class_df['LipidMolec'].unique()
            n_standards = len(standards)
            
            if n_standards == 1:
                # Single standard: simple bar plot
                standard_name = standards[0]
                intensities = []
                for col in valid_intensity_cols:
                    val = class_df[col].values[0]
                    intensities.append(val)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=samples,
                        y=intensities,
                        marker_color=colors[0],
                        name=standard_name
                    )
                ])
                
                fig.update_layout(
                    title=f'Internal Standards Intensity for {class_key}',
                    xaxis_title='Samples',
                    yaxis_title='Raw Intensity',
                    height=400,
                    bargap=0.2,
                    showlegend=True,
                    legend_title='Internal Standard',
                    font=dict(color='black'),
                    title_font=dict(color='black')
                )
            else:
                # Multiple standards: separate subplot for each
                fig = make_subplots(
                    rows=n_standards, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.08,
                    subplot_titles=[f'{std}' for std in standards]
                )
                
                for i, standard_name in enumerate(standards):
                    std_row = class_df[class_df['LipidMolec'] == standard_name]
                    intensities = []
                    for col in valid_intensity_cols:
                        val = std_row[col].values[0] if not std_row.empty else 0
                        intensities.append(val)
                    
                    fig.add_trace(
                        go.Bar(
                            x=samples,
                            y=intensities,
                            marker_color=colors[i % len(colors)],
                            name=standard_name,
                            showlegend=True
                        ),
                        row=i+1, col=1
                    )
                    
                    # Update y-axis label for each subplot
                    fig.update_yaxes(title_text='Intensity', row=i+1, col=1)
                
                fig.update_layout(
                    title=f'Internal Standards Intensity for {class_key}',
                    height=300 * n_standards,
                    bargap=0.2,
                    showlegend=True,
                    legend_title='Internal Standard',
                    font=dict(color='black'),
                    title_font=dict(color='black')
                )
                
                # Add x-axis title only to bottom subplot
                fig.update_xaxes(title_text='Samples', row=n_standards, col=1)
            
            # Style all axes
            fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
            fig.update_layout(legend=dict(font=dict(color='black')))
            
            figs.append(fig)
        
        return figs