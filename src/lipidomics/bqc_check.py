import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class BQCQualityCheck:
    """
    A class dedicated to conducting Batch Quality Control (BQC) assessments on lipidomics data.
    It provides functionalities to calculate the coefficient of variation, mean excluding zeros,
    log transformation, and to generate and display a scatter plot for CoV analysis.
    """
    
    @staticmethod
    def impute_zeros_by_class(dataframe, value_columns, class_column='ClassKey'):
        """
        Impute zeros or filter rows based on the number of zeros in specified columns.
        - If exactly one zero, impute with the smallest non-zero value in the same class divided by 10.
        - If more than one zero, exclude the row and include it in filtered lipids.
        
        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            value_columns (list): List of columns where zeros are checked/imputed.
            class_column (str): The column name containing class labels (default: 'ClassKey').
        
        Returns:
            tuple: (processed DataFrame with imputed zeros and filtered rows removed,
                    DataFrame of filtered lipids with multiple zeros)
        """
        df = dataframe.copy()
        
        if class_column not in df.columns:
            raise ValueError(f"Class column '{class_column}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        # Count zeros per row in value_columns
        zero_counts = (df[value_columns] == 0).sum(axis=1)
        
        # Initialize filtered lipids DataFrame
        filtered_lipids = pd.DataFrame()
        
        # Process rows with exactly one zero
        one_zero_mask = zero_counts == 1
        if one_zero_mask.any():
            one_zero_df = df[one_zero_mask].copy()
            for class_label in one_zero_df[class_column].unique():
                class_mask = one_zero_df[class_column] == class_label
                class_data = one_zero_df.loc[class_mask, value_columns]
                
                # Find smallest non-zero value in class
                all_values = class_data.values.flatten()
                non_zero_values = all_values[all_values > 0]
                if len(non_zero_values) == 0:
                    min_value = 1e-10  # Fallback
                else:
                    min_value = non_zero_values.min() / 10.0
                
                # Impute zeros
                for col in value_columns:
                    one_zero_df.loc[class_mask & (one_zero_df[col] == 0), col] = min_value
            
            df.loc[one_zero_mask] = one_zero_df
        
        # Collect rows with more than one zero
        multi_zero_mask = zero_counts > 1
        if multi_zero_mask.any():
            filtered_lipids = df[multi_zero_mask].copy()
            filtered_lipids['Reason'] = 'Multiple zeros in BQC concentration columns'
            # Keep only relevant columns (adjust as needed)
            filtered_lipids = filtered_lipids[['LipidMolec', class_column, *value_columns, 'Reason']]
        
        # Exclude rows with multiple zeros from main DataFrame
        df = df[~multi_zero_mask].reset_index(drop=True)
        
        return df, filtered_lipids
    
    @staticmethod
    def calculate_coefficient_of_variation(numbers):
        """
        Calculate the coefficient of variation (CoV) for a given array of numbers.
        Now includes zero values in the calculation.
    
        Args:
            numbers (list): A list or array of numerical values.
    
        Returns:
            float or None: The coefficient of variation as a percentage, or None if calculation is not applicable.
        """
        # Convert to numpy array for easier handling
        numbers = np.array(numbers)
        
        # We now include zeros in the calculation
        # Only requirement is having at least 2 values to calculate std
        if len(numbers) >= 2:
            mean_val = np.mean(numbers)
            if mean_val == 0:
                # If mean is zero, CoV is undefined (division by zero)
                return None
            return np.std(numbers, ddof=1) / mean_val * 100
        return None
    
    @staticmethod
    def calculate_mean_including_zeros(numbers):
        """
        Calculate the mean of a list of numbers, including zeros.
        
        Args:
            numbers (list): A list or array of numerical values.
    
        Returns:
            float or None: The mean of the numbers, or None if fewer than 2 values are present.
        """
        numbers = np.array(numbers)
        if len(numbers) >= 2:
            return np.mean(numbers)
        return None

    @staticmethod
    def apply_log_transformation(series):
        """
        Apply log10 transformation to a Pandas series.
        """
        return np.log10(series)

    @staticmethod
    @st.cache_data(ttl=3600)
    def prepare_dataframe_for_plot(dataframe, area_under_curve_columns):
        """
        Prepare a DataFrame for plotting by imputing zeros, calculating CoV, and mean values for specified columns.
        
        Args:
            dataframe (pd.DataFrame): The DataFrame to process.
            area_under_curve_columns (list): A list of columns representing areas under the curve.
        
        Returns:
            tuple: (processed DataFrame with additional 'cov' and 'mean' columns,
                    DataFrame of filtered lipids with multiple zeros)
        """
        # Impute zeros and get filtered lipids
        dataframe, filtered_lipids = BQCQualityCheck.impute_zeros_by_class(
            dataframe, 
            value_columns=area_under_curve_columns, 
            class_column='ClassKey'
        )
        
        # Calculate CoV and mean
        dataframe['cov'] = dataframe[area_under_curve_columns].apply(
            BQCQualityCheck.calculate_coefficient_of_variation, axis=1)
        dataframe['mean'] = dataframe[area_under_curve_columns].apply(
            BQCQualityCheck.calculate_mean_including_zeros, axis=1)
        
        # Apply log transformation where mean > 0
        valid_mean_mask = (dataframe['mean'].notnull()) & (dataframe['mean'] > 0)
        dataframe.loc[valid_mean_mask, 'mean'] = np.log10(dataframe.loc[valid_mean_mask, 'mean'])
        
        return dataframe, filtered_lipids

    @staticmethod
    def prepare_plot_inputs(dataframe):
        """
        Prepare the inputs needed for creating a scatter plot from the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame with the necessary data.

        Returns:
            tuple: Three arrays representing mean concentrations, CoV values, and lipid species.
        """
        return dataframe['mean'].values, dataframe['cov'].values, dataframe['LipidMolec'].values

    @staticmethod
    def create_cov_scatter_plot(mean_concentrations, coefficients_of_variation, lipid_species):
        """
        Create a scatter plot showing the coefficient of variation for lipid species using Plotly.

        Args:
            mean_concentrations (array): Array of mean concentrations.
            coefficients_of_variation (array): Array of CoV values.
            lipid_species (array): Array of lipid species names.

        Returns:
            tuple: A Plotly Figure object and a DataFrame used for the plot.
        """
        cov_df = pd.DataFrame({
            "Mean_concentration": mean_concentrations, 
            "CoV": coefficients_of_variation, 
            'Species': lipid_species
        })

        fig = go.Figure(data=go.Scatter(
            x=cov_df["Mean_concentration"],
            y=cov_df["CoV"],
            mode='markers',
            marker=dict(size=5, color='blue'),  # Reduced marker size
            text=cov_df['Species'],
            hovertemplate=
            '<b>Species:</b> %{text}<br>' +
            '<b>Mean concentration:</b> %{x:.4f}<br>' +
            '<b>CoV:</b> %{y:.2f}%<br>' +
            '<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': 'CoV - All lipid Species',
                'font': {'size': 24, 'color': 'black'}  # Larger, black title
            },
            xaxis_title='Log10 of Mean BQC Concentration',
            yaxis_title='CoV(%)',
            xaxis=dict(
                title_font=dict(size=18, color='black'),  # Larger, black axis title
                tickfont=dict(size=14, color='black'),  # Larger, black tick labels
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title_font=dict(size=18, color='black'),  # Larger, black axis title
                tickfont=dict(size=14, color='black'),  # Larger, black tick labels
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True
            ),
            plot_bgcolor='white',  # White background
            paper_bgcolor='white',  # White surrounding area
            showlegend=False,
            margin=dict(t=50, r=50, b=50, l=50)  # Add some margin
        )

        # Add a frame around the plot
        fig.update_xaxes(zeroline=False)
        fig.update_yaxes(zeroline=False)

        return fig, cov_df

    @staticmethod
    def create_cov_scatter_plot_with_threshold(mean_concentrations, coefficients_of_variation, lipid_species, cov_threshold):
        """
        Create a scatter plot showing the coefficient of variation for lipid species using Plotly with threshold coloring.
    
        Args:
            mean_concentrations (array): Array of mean concentrations.
            coefficients_of_variation (array): Array of CoV values.
            lipid_species (array): Array of lipid species names.
            cov_threshold (float): The CoV threshold for coloring data points.
    
        Returns:
            tuple: A Plotly Figure object and a DataFrame used for the plot.
        """
        # Convert to numpy arrays for easier handling
        mean_concentrations = np.array(mean_concentrations)
        coefficients_of_variation = np.array(coefficients_of_variation)
        lipid_species = np.array(lipid_species)
        
        # Filter out NaN values
        valid_indices = ~(np.isnan(mean_concentrations) | np.isnan(coefficients_of_variation))
        mean_concentrations = mean_concentrations[valid_indices]
        coefficients_of_variation = coefficients_of_variation[valid_indices]
        lipid_species = lipid_species[valid_indices]
        
        cov_df = pd.DataFrame({
            "Mean_concentration": mean_concentrations, 
            "CoV": coefficients_of_variation, 
            'Species': lipid_species
        })
    
        fig = go.Figure()
        
        # Separate data by threshold
        below_threshold = cov_df[cov_df['CoV'] <= cov_threshold]
        above_threshold = cov_df[cov_df['CoV'] > cov_threshold]
        
        # Add points below threshold (blue)
        if len(below_threshold) > 0:
            fig.add_trace(go.Scatter(
                x=below_threshold["Mean_concentration"],
                y=below_threshold["CoV"],
                mode='markers',
                marker=dict(size=5, color='blue'),  # Reduced marker size
                text=below_threshold['Species'],
                hovertemplate=
                '<b>Species:</b> %{text}<br>' +
                '<b>Mean concentration:</b> %{x:.4f}<br>' +
                '<b>CoV:</b> %{y:.2f}%<br>' +
                '<extra></extra>'
            ))
        
        # Add points above threshold (red)
        if len(above_threshold) > 0:
            fig.add_trace(go.Scatter(
                x=above_threshold["Mean_concentration"],
                y=above_threshold["CoV"],
                mode='markers',
                marker=dict(size=5, color='red'),  # Reduced marker size  
                text=above_threshold['Species'],
                hovertemplate=
                '<b>Species:</b> %{text}<br>' +
                '<b>Mean concentration:</b> %{x:.4f}<br>' +
                '<b>CoV:</b> %{y:.2f}%<br>' +
                '<extra></extra>'
            ))
        
        # Add horizontal threshold line
        fig.add_hline(
            y=cov_threshold,
            line_dash="solid",
            line_color="black",
            line_width=2,
            annotation_text=f"Threshold: {cov_threshold}%",
            annotation_position="top right",
            annotation_font=dict(size=14, color="black")
        )
    
        fig.update_layout(
            title={
                'text': 'CoV - All lipid Species',
                'font': {'size': 24, 'color': 'black'}  # Larger, black title
            },
            xaxis_title='Log10 of Mean BQC Concentration',
            yaxis_title='CoV(%)',
            xaxis=dict(
                title_font=dict(size=18, color='black'),  # Larger, black axis title
                tickfont=dict(size=14, color='black'),  # Larger, black tick labels
                tickcolor='black',  # Black tick marks
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,
                ticks='outside'
            ),
            yaxis=dict(
                title_font=dict(size=18, color='black'),  # Larger, black axis title
                tickfont=dict(size=14, color='black'),  # Larger, black tick labels
                tickcolor='black',  # Black tick marks  
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,
                ticks='outside'
            ),
            plot_bgcolor='white',  # White background
            paper_bgcolor='white',  # White surrounding area
            showlegend=False,
            margin=dict(t=50, r=50, b=50, l=50),  # Add some margin
            font=dict(color='black')  # Set global font color to black
        )
    
        # Add a frame around the plot
        fig.update_xaxes(zeroline=False)
        fig.update_yaxes(zeroline=False)
    
        return fig, cov_df

    @staticmethod
    @st.cache_data(ttl=3600)
    def generate_cov_plot_data(dataframe, individual_samples_list, bqc_sample_index):
        """
        Prepare data for generating a CoV scatter plot.
        
        Args:
            dataframe (pd.DataFrame): The DataFrame containing lipidomics data.
            individual_samples_list (list): A list of individual samples.
            bqc_sample_index (int): The index of BQC samples in the experiment's sample list.
        
        Returns:
            tuple: (DataFrame prepared for plotting, DataFrame of filtered lipids)
        """
        bqc_samples_list = individual_samples_list[bqc_sample_index]
        auc = ['concentration[' + sample + ']' for sample in bqc_samples_list]
        return BQCQualityCheck.prepare_dataframe_for_plot(dataframe.copy(), auc)
    
    @staticmethod
    def generate_and_display_cov_plot(dataframe, experiment_details, bqc_sample_index, cov_threshold=30):
        """
        Generate and display a CoV scatter plot based on BQC samples from a given DataFrame.
        
        Args:
            dataframe (pd.DataFrame): The DataFrame containing lipidomics data.
            experiment_details (Experiment): An object containing details about the experiment setup.
            bqc_sample_index (int): The index of BQC samples in the experiment's sample list.
            cov_threshold (float): The CoV threshold for coloring data points and calculating reliability.
        
        Returns:
            tuple: (Plotly Figure object, prepared DataFrame, reliable data percentage, filtered lipids DataFrame)
        """
        prepared_df, filtered_lipids = BQCQualityCheck.generate_cov_plot_data(
            dataframe, experiment_details.individual_samples_list, bqc_sample_index
        )
        mean_concentrations, coefficients_of_variation, lipid_species = BQCQualityCheck.prepare_plot_inputs(prepared_df)
        scatter_plot, cov_df = BQCQualityCheck.create_cov_scatter_plot_with_threshold(
            mean_concentrations, coefficients_of_variation, lipid_species, cov_threshold
        )
        
        reliable_data_percent = round(len(prepared_df[prepared_df['cov'] < cov_threshold]) / len(prepared_df) * 100, 1)
        
        return scatter_plot, prepared_df, reliable_data_percent, filtered_lipids

    @staticmethod
    def display_filtered_lipids(filtered_lipids, filter_choice, cov_filtered_df=None):
        """
        Display a table of filtered lipids if the user selects 'yes' for filtering.
        
        Args:
            filtered_lipids (pd.DataFrame): DataFrame of lipids filtered due to multiple zeros.
            filter_choice (str): User's filtering choice ('yes' or 'no').
            cov_filtered_df (pd.DataFrame, optional): DataFrame of lipids filtered by CoV threshold.
        
        Returns:
            pd.DataFrame: Combined DataFrame of filtered lipids (if any).
        """
        if filter_choice.lower() != 'yes':
            return pd.DataFrame()  # Return empty DataFrame if no filtering
        
        # Combine filtered lipids (multiple zeros) and CoV-filtered lipids (if provided)
        combined_filtered = filtered_lipids.copy()
        if cov_filtered_df is not None and not cov_filtered_df.empty:
            cov_filtered_df['Reason'] = f'CoV above threshold ({cov_threshold}%)'
            combined_filtered = pd.concat([combined_filtered, cov_filtered_df], ignore_index=True)
        
        if not combined_filtered.empty:
            st.subheader("Filtered Lipids")
            st.dataframe(combined_filtered)
        
        return combined_filtered

    @staticmethod
    @st.cache_data(ttl=3600)
    def filter_dataframe_by_cov_threshold(threshold, prepared_df):
        """
        Filter a DataFrame based on a specified CoV threshold.
        
        Args:
            threshold (float): The CoV threshold for filtering.
            prepared_df (pd.DataFrame): The DataFrame to be filtered.
        
        Returns:
            tuple: (filtered DataFrame with irrelevant columns removed and index reset,
                    DataFrame of lipids filtered by CoV)
        """
        filtered_df = prepared_df[prepared_df['cov'] <= threshold]
        cov_filtered_df = prepared_df[prepared_df['cov'] > threshold].copy()
        cov_filtered_df['Reason'] = f'CoV above {threshold}%'
        cov_filtered_df = cov_filtered_df[['LipidMolec', 'ClassKey', 'cov', 'mean', 'Reason']]  # Adjust columns as needed
        
        return filtered_df.drop(['mean', 'cov'], axis=1).reset_index(drop=True), cov_filtered_df
    
    