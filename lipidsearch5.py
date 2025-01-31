# Existing imports
import base64
import copy
import hashlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# Local imports
import lipidomics as lp

def main():
    """
    Main function for the LipidSearch 5.0 Module Streamlit application.
    """

    st.header("LipidSearch 5.0 Module")
    st.markdown("Process, visualize and analyze LipidSearch 5.0 data.")
    
    # Initialize session state for cache clearing
    if 'clear_cache' not in st.session_state:
        st.session_state.clear_cache = False
    
    st.info("""
    **Dataset Requirements for LipidSearch 5.0 Module**
    
    Ensure your dataset includes the following mandatory columns:
    * `LipidMolec`: The molecule identifier for the lipid.
    * `ClassKey`: The classification key for the lipid type.
    * `CalcMass`: The calculated mass of the lipid molecule.
    * `BaseRt`: The base retention time.
    * `TotalGrade`: The overall quality grade of the lipid data.
    * `TotalSmpIDRate(%)`: The total sample identification rate as a percentage.
    * `FAKey`: The fatty acid key associated with the lipid.
    
    Additionally, each sample in your dataset must have a corresponding `MeanArea` column to represent intensity values. 
    For instance, if your dataset comprises 10 samples, you should have the following columns: 
    `MeanArea[s1]`, `MeanArea[s2]`, ..., `MeanArea[s10]` for each respective sample intensity.
    """)
    
    initialize_session_state()
    uploaded_file = st.sidebar.file_uploader('Upload your LipidSearch 5.0 dataset', type=['csv', 'txt'])
    if uploaded_file:
        df = load_data(uploaded_file)
        confirmed, name_df, experiment, bqc_label, valid_samples = process_experiment(df)
        if confirmed and valid_samples:
            update_session_state(name_df, experiment, bqc_label)
            if st.session_state.module == "Data Cleaning, Filtering, & Normalization":
                cleaned_df, intsta_df = data_cleaning_module(df, st.session_state.experiment, st.session_state.name_df)
                if cleaned_df is not None and intsta_df is not None:
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.intsta_df = intsta_df
                    st.session_state.continuation_df = cleaned_df
                
                _, right_column = st.columns([3, 1])
                with right_column:
                    if st.button("Next: Quality Check & Analysis", key="next_to_qc_analysis"):
                        st.session_state.module = "Quality Check & Analysis"
                        st.experimental_rerun()
            elif st.session_state.module == "Quality Check & Analysis":
                if st.session_state.cleaned_df is not None:
                    quality_check_and_analysis_module(
                        st.session_state.continuation_df, 
                        st.session_state.intsta_df, 
                        st.session_state.experiment,
                        st.session_state.bqc_label
                    )
                
                if st.button("Back to Data Cleaning, Filtering, & Normalization", key="back_to_cleaning"):
                    st.session_state.module = "Data Cleaning, Filtering, & Normalization"
                    st.experimental_rerun()
    
    # Place the cache clearing button in the sidebar
    if st.sidebar.button("End Session and Clear Cache"):
        st.session_state.clear_cache = True
        st.experimental_rerun()

    # Check if cache should be cleared
    if st.session_state.clear_cache:
        clear_streamlit_cache()
        st.sidebar.success("Cache cleared successfully!")
        st.session_state.clear_cache = False  # Reset the flag
    
    st.info("""
            **Tip:** When you're done, please click 'End Session and Clear Cache' in the sidebar. This ensures each session starts fresh and can help maintain optimal app performance. Additionally, if the app crashes and you are forced to restart the session, clearing the cache can help prevent further issues.
            """)

def clear_streamlit_cache():
    """
    Clear all Streamlit caches.
    """
    st.cache_data.clear()
    st.cache_resource.clear()

def initialize_session_state():
    """
    Initialize the Streamlit session state with default values.

    This function sets up the initial state for various session variables
    used throughout the application. It ensures that all necessary state
    variables are present and set to their default values if they haven't
    been initialized yet.

    The following session state variables are initialized:
    - module: Current module of the application
    - cleaned_df: Cleaned dataframe
    - intsta_df: Internal standard dataframe
    - continuation_df: Continuation dataframe
    - experiment: Experiment object
    - bqc_label: Batch Quality Control label
    - normalization_inputs: Dictionary for normalization inputs
    - normalization_method: Selected normalization method

    Note: This function should be called at the beginning of the Streamlit app
    to ensure proper state management.
    """
    if 'module' not in st.session_state:
        st.session_state.module = "Data Cleaning, Filtering, & Normalization"
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'intsta_df' not in st.session_state:
        st.session_state.intsta_df = None
    if 'continuation_df' not in st.session_state:
        st.session_state.continuation_df = None
    if 'experiment' not in st.session_state:
        st.session_state.experiment = None
    if 'bqc_label' not in st.session_state:
        st.session_state.bqc_label = None
    if 'normalization_inputs' not in st.session_state:
        st.session_state.normalization_inputs = {}
    if 'normalization_method' not in st.session_state:
        st.session_state.normalization_method = 'None'

def update_session_state(name_df, experiment, bqc_label):
    """
    Update the Streamlit session state with experiment-related information.

    This function updates several session state variables with information
    derived from the experiment setup and naming dataframe.

    Args:
        name_df (pd.DataFrame): DataFrame containing naming information.
        experiment (Experiment): Experiment object containing experimental setup details.
        bqc_label (str): Label for Batch Quality Control samples.

    The following session state variables are updated:
    - name_df: DataFrame with naming information
    - experiment: Experiment object
    - bqc_label: Batch Quality Control label
    - full_samples_list: List of all samples
    - individual_samples_list: List of individual samples for each condition
    - conditions_list: List of experimental conditions
    - extensive_conditions_list: Detailed list of conditions
    - number_of_samples_list: Number of samples for each condition
    - aggregate_number_of_samples_list: Aggregated number of samples
    """
    st.session_state.name_df = name_df
    st.session_state.experiment = experiment
    st.session_state.bqc_label = bqc_label
    st.session_state.full_samples_list = experiment.full_samples_list
    st.session_state.individual_samples_list = experiment.individual_samples_list
    st.session_state.conditions_list = experiment.conditions_list
    st.session_state.extensive_conditions_list = experiment.extensive_conditions_list
    st.session_state.number_of_samples_list = experiment.number_of_samples_list
    st.session_state.aggregate_number_of_samples_list = experiment.aggregate_number_of_samples_list

def data_cleaning_module(df, experiment, name_df):
    """
    Perform data cleaning, filtering, and normalization on the input dataframe.

    This function orchestrates the data cleaning process, including displaying
    raw data, cleaning and filtering the data, and applying normalization.

    Args:
        df (pd.DataFrame): The raw input dataframe.
        experiment (Experiment): Experiment object containing experimental setup details.
        name_df (pd.DataFrame): DataFrame containing naming information.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - normalized_df: The cleaned, filtered, and normalized dataframe.
            - intsta_df: The internal standard dataframe.

    Note: This function uses Streamlit to display various UI elements and
    intermediate results during the data cleaning process.
    """
    st.subheader("1) Clean, Filter, & Normalize Data")
    display_raw_data(df)
    cleaned_df, intsta_df = display_cleaned_data(df, experiment, name_df)
    normalized_df = display_normalization_options(cleaned_df, intsta_df, experiment)
    
    return normalized_df, intsta_df

def quality_check_and_analysis_module(continuation_df, intsta_df, experiment, bqc_label):
    st.subheader("2) Quality Check & Anomaly Detection")
    
    # Initialize variables
    box_plot_fig1 = None
    box_plot_fig2 = None
    bqc_plot = None
    retention_time_plot = None
    pca_plot = None

    # Initialize session state for plots
    if 'heatmap_generated' not in st.session_state:
        st.session_state.heatmap_generated = False
    if 'heatmap_fig' not in st.session_state:
        st.session_state.heatmap_fig = {}
    if 'correlation_plots' not in st.session_state:
        st.session_state.correlation_plots = {}
    if 'abundance_bar_charts' not in st.session_state:
        st.session_state.abundance_bar_charts = {'linear': None, 'log2': None}
    if 'abundance_pie_charts' not in st.session_state:
        st.session_state.abundance_pie_charts = {}
    if 'saturation_plots' not in st.session_state:
        st.session_state.saturation_plots = {}
    if 'volcano_plots' not in st.session_state:
        st.session_state.volcano_plots = {}
    if 'pathway_visualization' not in st.session_state:
        st.session_state.pathway_visualization = None

    # Quality Check
    box_plot_fig1, box_plot_fig2 = display_box_plots(continuation_df, experiment)
    continuation_df, bqc_plot = conduct_bqc_quality_assessment(bqc_label, continuation_df, experiment)
    retention_time_plot = display_retention_time_plots(continuation_df)
    
    # Pairwise Correlation Analysis
    selected_condition, corr_fig = analyze_pairwise_correlation(continuation_df, experiment)
    if selected_condition and corr_fig:
        st.session_state.correlation_plots[selected_condition] = corr_fig
    
    # PCA Analysis
    continuation_df, pca_plot = display_pca_analysis(continuation_df, experiment)
    
    st.subheader("3) Data Visualization, Interpretation, and Analysis ")
    # Analysis
    analysis_option = st.radio(
        "Select an analysis feature:",
        (
            "Class Level Breakdown - Bar Chart", 
            "Class Level Breakdown - Pie Charts", 
            "Class Level Breakdown - Saturation Plots", 
            "Class Level Breakdown - Pathway Visualization",
            "Species Level Breakdown - Volcano Plot",
            "Species Level Breakdown - Lipidomic Heatmap"
        )
    )

    if analysis_option == "Class Level Breakdown - Bar Chart":
        linear_chart, log2_chart = display_abundance_bar_charts(experiment, continuation_df)
        if linear_chart:
            st.session_state.abundance_bar_charts['linear'] = linear_chart
        if log2_chart:
            st.session_state.abundance_bar_charts['log2'] = log2_chart
    elif analysis_option == "Class Level Breakdown - Pie Charts":
        pie_charts = display_abundance_pie_charts(experiment, continuation_df)
        st.session_state.abundance_pie_charts.update(pie_charts)
    elif analysis_option == "Class Level Breakdown - Saturation Plots":
        saturation_plots = display_saturation_plots(experiment, continuation_df)
        st.session_state.saturation_plots.update(saturation_plots)
    elif analysis_option == "Class Level Breakdown - Pathway Visualization":
        pathway_fig = display_pathway_visualization(experiment, continuation_df)
        if pathway_fig:
            st.session_state.pathway_visualization = pathway_fig
    elif analysis_option == "Species Level Breakdown - Volcano Plot":
        volcano_plots = display_volcano_plot(experiment, continuation_df)
        st.session_state.volcano_plots.update(volcano_plots)
    elif analysis_option == "Species Level Breakdown - Lipidomic Heatmap":
        heatmap_result = display_lipidomic_heatmap(experiment, continuation_df)
        if isinstance(heatmap_result, tuple) and len(heatmap_result) == 2:
            regular_heatmap, clustered_heatmap = heatmap_result
        else:
            regular_heatmap = heatmap_result
            clustered_heatmap = None
        
        if regular_heatmap:
            st.session_state.heatmap_generated = True
            st.session_state.heatmap_fig['Regular Heatmap'] = regular_heatmap
        if clustered_heatmap:
            st.session_state.heatmap_fig['Clustered Heatmap'] = clustered_heatmap

    # PDF Generation Section
    st.subheader("Generate PDF Report")
    st.warning(
        "⚠️ Important: PDF Report Generation Guidelines\n\n"
        "1. Generate the PDF only after completing all desired analyses.\n"
        "2. Ensure all analyses you want in the report have been viewed at least once.\n"
        "3. Use this feature instead of downloading plots individually - it's more efficient for multiple downloads.\n"
        "4. Generate the PDF before clearing the cache.\n"
        "5. Avoid interacting with the app during PDF generation.\n"
        "6. Select 'No' if you're not ready to end your session."
    )

    generate_pdf = st.radio("Are you ready to generate the PDF report?", ('No', 'Yes'), index=0)

    if generate_pdf == 'Yes':
        st.info("Please confirm that you have completed all analyses and are ready to end your session.")
        confirm_generate = st.checkbox("I confirm that I'm ready to generate the PDF and end my session.")
        
        if confirm_generate:
            if box_plot_fig1 and box_plot_fig2:
                with st.spinner('Generating PDF report... Please do not interact with the app.'):
                    pdf_buffer = generate_pdf_report(
                        box_plot_fig1, box_plot_fig2, bqc_plot, retention_time_plot, pca_plot, 
                        st.session_state.heatmap_fig, st.session_state.correlation_plots, 
                        st.session_state.abundance_bar_charts, st.session_state.abundance_pie_charts, 
                        st.session_state.saturation_plots, st.session_state.volcano_plots,
                        st.session_state.pathway_visualization
                    )
                if pdf_buffer:
                    st.success("PDF report generated successfully!")
                    st.download_button(
                        label="Download Quality Check & Analysis Report (PDF)",
                        data=pdf_buffer,
                        file_name="quality_check_and_analysis_report.pdf",
                        mime="application/pdf",
                    )
                    st.info(
                        "You can now download your PDF report. After downloading, please use the "
                        "'End Session and Clear Cache' button in the sidebar to conclude your session."
                    )
                    
                    # Close the figures to free up memory after PDF generation
                    plt.close(box_plot_fig1)
                    plt.close(box_plot_fig2)
                    plt.close('all')  # This closes all remaining matplotlib figures
                else:
                    st.error("Failed to generate PDF report. Please check the logs for details.")
            else:
                st.warning("Some plots are missing. Unable to generate PDF report.")
        else:
            st.info("Please confirm when you're ready to generate the PDF report.")

def generate_pdf_report(box_plot_fig1, box_plot_fig2, bqc_plot, retention_time_plot, pca_plot, heatmap_figs, correlation_plots, abundance_bar_charts, abundance_pie_charts, saturation_plots, volcano_plots, pathway_visualization):
    pdf_buffer = io.BytesIO()
    
    try:
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Page 1: Box Plots
        img_buffer1 = io.BytesIO()
        box_plot_fig1.savefig(img_buffer1, format='png', dpi=300, bbox_inches='tight')
        img_buffer1.seek(0)
        img1 = ImageReader(img_buffer1)
        pdf.drawImage(img1, 50, 400, width=500, height=300, preserveAspectRatio=True)
        
        img_buffer2 = io.BytesIO()
        box_plot_fig2.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
        img_buffer2.seek(0)
        img2 = ImageReader(img_buffer2)
        pdf.drawImage(img2, 50, 50, width=500, height=300, preserveAspectRatio=True)
        
        # Page 2: BQC Plot
        pdf.showPage()
        if bqc_plot is not None:
            bqc_bytes = pio.to_image(bqc_plot, format='png', width=800, height=600, scale=2)
            bqc_img = ImageReader(io.BytesIO(bqc_bytes))
            pdf.drawImage(bqc_img, 50, 100, width=500, height=400, preserveAspectRatio=True)
        
        # Page 3: Retention Time Plot
        pdf.showPage()
        pdf.setPageSize(landscape(letter))
        if retention_time_plot is not None:
            svg_bytes = pio.to_image(retention_time_plot, format='svg')
            drawing = svg2rlg(io.BytesIO(svg_bytes))
            renderPDF.draw(drawing, pdf, 50, 50)
        
        # Page 4: PCA Plot
        pdf.showPage()
        pdf.setPageSize(landscape(letter))
        if pca_plot is not None:
            svg_bytes = pio.to_image(pca_plot, format='svg')
            drawing = svg2rlg(io.BytesIO(svg_bytes))
            renderPDF.draw(drawing, pdf, 50, 50)
        
        # Pages for Correlation Plots
        for condition, corr_fig in correlation_plots.items():
            pdf.showPage()
            pdf.setPageSize(letter)
            img_buffer = io.BytesIO()
            corr_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            pdf.drawImage(img, 50, 100, width=500, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 50, f"Correlation Plot for {condition}")
        
        # Add Abundance Bar Charts
        for scale, chart in abundance_bar_charts.items():
            if chart is not None:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                img_buffer = io.BytesIO()
                chart.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img = ImageReader(img_buffer)
                pdf.drawImage(img, 50, 50, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 30, f"Abundance Bar Chart ({scale} scale)")
        
        # Add Abundance Pie Charts
        for condition, pie_chart in abundance_pie_charts.items():
            pdf.showPage()
            pdf.setPageSize(letter)
            pie_bytes = pio.to_image(pie_chart, format='png', width=800, height=600, scale=2)
            pie_img = ImageReader(io.BytesIO(pie_bytes))
            pdf.drawImage(pie_img, 50, 100, width=500, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 50, f"Abundance Pie Chart for {condition}")
        
        # Add Saturation Plots
        for lipid_class, plots in saturation_plots.items():
            if isinstance(plots, dict) and 'main' in plots and 'percentage' in plots:
                # Main plot
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                main_bytes = pio.to_image(plots['main'], format='png', width=1000, height=700, scale=2)
                main_img = ImageReader(io.BytesIO(main_bytes))
                pdf.drawImage(main_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, f"Saturation Plot (Main) for {lipid_class}")
                
                # Percentage plot
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                percentage_bytes = pio.to_image(plots['percentage'], format='png', width=1000, height=700, scale=2)
                percentage_img = ImageReader(io.BytesIO(percentage_bytes))
                pdf.drawImage(percentage_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, f"Saturation Plot (Percentage) for {lipid_class}")
        
        # Add Volcano Plots
        if volcano_plots:
            # Main Volcano Plot
            if 'main' in volcano_plots:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                main_bytes = pio.to_image(volcano_plots['main'], format='png', width=1000, height=700, scale=2)
                main_img = ImageReader(io.BytesIO(main_bytes))
                pdf.drawImage(main_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, "Volcano Plot")

            # Concentration vs Fold Change Plot
            if 'concentration_vs_fold_change' in volcano_plots:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                conc_bytes = pio.to_image(volcano_plots['concentration_vs_fold_change'], format='png', width=1000, height=700, scale=2)
                conc_img = ImageReader(io.BytesIO(conc_bytes))
                pdf.drawImage(conc_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, "Concentration vs Fold Change Plot")

            # Concentration Distribution Plot
            if 'concentration_distribution' in volcano_plots:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                dist_buffer = io.BytesIO()
                volcano_plots['concentration_distribution'].savefig(dist_buffer, format='png', dpi=300, bbox_inches='tight')
                dist_buffer.seek(0)
                dist_img = ImageReader(dist_buffer)
                pdf.drawImage(dist_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, "Concentration Distribution Plot")
        
        # Add Pathway Visualization
        if pathway_visualization is not None:
            pdf.showPage()
            pdf.setPageSize(landscape(letter))
            img_buffer = io.BytesIO()
            pathway_visualization.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            pdf.drawImage(img, 50, 50, width=700, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 30, "Lipid Pathway Visualization")
        
        # Add Lipidomic Heatmaps
        if 'Regular Heatmap' in heatmap_figs:
            add_heatmap_to_pdf(pdf, heatmap_figs['Regular Heatmap'], "Regular Lipidomic Heatmap")
        
        if 'Clustered Heatmap' in heatmap_figs:
            add_heatmap_to_pdf(pdf, heatmap_figs['Clustered Heatmap'], "Clustered Lipidomic Heatmap")
        
        pdf.save()
        pdf_buffer.seek(0)
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return pdf_buffer

def add_heatmap_to_pdf(pdf, heatmap_fig, title):
    pdf.showPage()
    pdf.setPageSize(landscape(letter))
    heatmap_bytes = pio.to_image(heatmap_fig, format='png', width=900, height=600, scale=2)
    heatmap_img = ImageReader(io.BytesIO(heatmap_bytes))
    page_width, page_height = landscape(letter)
    img_width = 700
    img_height = (img_width / 900) * 600
    x_position = (page_width - img_width) / 2
    y_position = (page_height - img_height) / 2
    pdf.drawImage(heatmap_img, x_position, y_position, width=img_width, height=img_height, preserveAspectRatio=True)
    pdf.drawString(x_position, y_position - 20, title)

@st.cache_data(ttl=3600)
def convert_df(df):
    """
    Convert a DataFrame to CSV format and encode it for downloading.

    Parameters:
    df (pd.DataFrame): The DataFrame to be converted.

    Returns:
    bytes: A CSV-formatted byte string of the DataFrame.

    Raises:
    Exception: If the DataFrame conversion to CSV fails.
    """
    if df is None or df.empty:
        raise ValueError("The DataFrame provided is empty or None.")
    try:
        return df.to_csv().encode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to convert DataFrame to CSV: {e}")
        
def matplotlib_svg_download_button(fig, filename):
    """
    Creates a Streamlit download button for the SVG format of a matplotlib figure.
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object to be converted into SVG.
        filename (str): The desired name of the downloadable SVG file.
    """
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    
    # Read the SVG data and create a download button
    svg_string = buf.getvalue().decode('utf-8')
    st.download_button(
        label="Download SVG",
        data=svg_string,
        file_name=filename,
        mime="image/svg+xml"
    )

def plotly_svg_download_button(fig, filename):
    """
    Creates a Streamlit download button for the SVG format of a plotly figure.
    Args:
        fig (plotly.graph_objs.Figure): Plotly figure object to be converted into SVG.
        filename (str): The desired name of the downloadable SVG file.
    """
    # Convert the figure to SVG format
    svg_bytes = fig.to_image(format="svg")
    
    # Decode the bytes to a string
    svg_string = svg_bytes.decode('utf-8')
    
    # Ensure the SVG string starts with the correct XML declaration
    if not svg_string.startswith('<?xml'):
        svg_string = '<?xml version="1.0" encoding="utf-8"?>\n' + svg_string
    
    # Create a download button
    st.download_button(
        label="Download SVG",
        data=svg_string,
        file_name=filename,
        mime="image/svg+xml"
    )

@st.cache_data(ttl=3600)
def load_data(file_object):
    """
    Load data from a file object into a DataFrame.

    Parameters:
    file_object (File-like object): The file object to load data from.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.

    Raises:
    Exception: If loading the data into a DataFrame fails.
    """
    try:
        return pd.read_csv(file_object)
    except Exception as e:
        raise Exception(f"Failed to load data from file: {e}")

def process_experiment(df):
    """
    Process the experiment setup and sample grouping based on user input.

    Returns:
    confirmed (bool): Whether the user has confirmed the inputs.
    name_df (DataFrame): DataFrame containing name mappings.
    experiment (Experiment): The configured Experiment object.
    bqc_label (str or None): Label used for BQC samples, if any.
    valid_samples (bool): Indicates whether sample validation was successful.
    """
    st.sidebar.subheader("Define Experiment")
    n_conditions = st.sidebar.number_input('Enter the number of conditions', min_value=1, max_value=20, value=1, step=1)
    conditions_list = [st.sidebar.text_input(f'Create a label for condition #{i + 1}') for i in range(n_conditions)]
    number_of_samples_list = [st.sidebar.number_input(f'Number of samples for condition #{i + 1}', min_value=1, max_value=1000, value=1, step=1) for i in range(n_conditions)]

    experiment = lp.Experiment()
    if not experiment.setup_experiment(n_conditions, conditions_list, number_of_samples_list):
        st.sidebar.error("All condition labels must be non-empty.")
        return None, None, None, None, False  # Ensure five values are returned

    name_df, group_df, valid_samples = process_group_samples(df, experiment)
    if not valid_samples:
        return None, None, None, None, False  # Consistently return five values

    bqc_label = specify_bqc_samples(experiment)
    confirmed = confirm_user_inputs(group_df, experiment)
    if not confirmed:
        return None, None, None, None, False  # Ensure five values are returned

    return confirmed, name_df, experiment, bqc_label, valid_samples

def specify_bqc_samples(experiment):
    """
    Queries the user through the Streamlit sidebar to specify if Batch Quality Control (BQC) samples exist
    within their dataset and, if so, to identify the label associated with these samples.

    BQC samples are technical replicates used to assess data quality. If the user indicates BQC samples are present,
    this function further prompts the user to select the label that corresponds to these BQC samples from a list of conditions
    with more than one sample. This is because BQC samples are typically derived from pooling multiple samples.

    Args:
        experiment (Experiment): The experiment object containing setup details, including conditions and the number of samples per condition.

    Returns:
        str or None: The label selected by the user that corresponds to BQC samples, or None if the user indicates there are no BQC samples.
    """
    st.sidebar.subheader("Specify Label of BQC Samples")
    bqc_ans = st.sidebar.radio('Do you have Batch Quality Control (BQC) samples?', ['Yes', 'No'], 1)
    bqc_label = None
    if bqc_ans == 'Yes':
        # Generate a list of condition labels where the number of samples per condition is greater than one.
        conditions_with_two_plus_samples = [
            condition for condition, number_of_samples in zip(experiment.conditions_list, experiment.number_of_samples_list)
            if number_of_samples > 1
        ]
        # Prompt the user to select the condition label that corresponds to the BQC samples.
        bqc_label = st.sidebar.radio('Which label corresponds to BQC samples?', conditions_with_two_plus_samples, 0)
    return bqc_label

def process_group_samples(df, experiment):
    """
    Group the samples based on the experiment setup.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the experiment data.
    experiment (Experiment): The experiment object with setup details.

    Returns:
    tuple: A tuple containing the confirmation status, name DataFrame, and the grouped DataFrame.

    Note:
    Errors in grouping samples are communicated through user feedback in the Streamlit interface.
    """
    grouped_samples = lp.GroupSamples(experiment)
    if not grouped_samples.check_dataset_validity(df):
        st.sidebar.error("This is not a valid LipidSearch 5.0 dataset!")
        return None, None, False  # Returning a failure status

    valid_samples = validate_total_samples(grouped_samples, experiment, df)
    if not valid_samples:
        st.sidebar.error("Invalid total number of samples!")
        return None, None, False  # Early exit with a failure status

    st.sidebar.subheader('Group Samples')
    group_df = grouped_samples.build_group_df(df)
    st.sidebar.write(group_df)

    # Proceed with allowing the user to interact with grouped samples
    st.sidebar.write('Are your samples properly grouped together?')
    ans = st.sidebar.radio('', ['Yes', 'No'])
    if ans == 'No':
        selections = {}
        remaining_samples = group_df['sample name'].tolist()
        for condition in experiment.conditions_list:
            selected_samples = st.sidebar.multiselect(f'Pick the samples that belong to condition {condition}', remaining_samples)
            selections[condition] = selected_samples
            remaining_samples = [s for s in remaining_samples if s not in selected_samples]
        group_df = grouped_samples.group_samples(group_df, selections)
        st.sidebar.write('Check the updated table below to make sure the samples are properly grouped together:')
        st.sidebar.write(group_df)

    name_df = grouped_samples.update_sample_names(group_df)

    # Return the confirmation status, name DataFrame, and the group DataFrame with a success status
    return name_df, group_df, True

def validate_total_samples(grouped_samples, experiment, df):
    """
    Validates if the total number of samples matches with the experimental setup.

    Parameters:
    grouped_samples (GroupSamples): The GroupSamples object for handling sample grouping.
    experiment (Experiment): The Experiment object with the setup details.
    df (pd.DataFrame): The DataFrame containing the experiment data.

    Returns:
    bool: True if the total number of samples matches, False otherwise.
    """
    mean_area_cols = grouped_samples.build_mean_area_col_list(df)
    return len(mean_area_cols) == len(experiment.full_samples_list)

def display_updated_names(name_df):
    """
    Display updated sample names if there are any changes.

    Parameters:
    name_df (pd.DataFrame): DataFrame containing the old and updated sample names.

    Raises:
    Exception: If the DataFrame operations fail.
    """
    st.sidebar.subheader("Updated Sample Names")
    try:
        if not name_df['old name'].equals(name_df['updated name']):
            st.sidebar.markdown("The table below shows the updated sample names:")
            st.sidebar.write(name_df)
        else:
            st.sidebar.markdown("No changes in sample names were necessary.")
    except Exception as e:
        raise Exception(f"Error in displaying updated names: {e}")

def confirm_user_inputs(group_df, experiment):
    """
    Confirm user inputs before proceeding to the next step in the app.

    Parameters:
    group_df (pd.DataFrame): DataFrame containing grouped sample information.
    experiment (Experiment): Experiment object with setup details.

    Returns:
    bool: Confirmation status from the user.

    Raises:
    Exception: If there is an error in displaying confirmation options.
    """
    try:
        st.sidebar.subheader("Confirm Inputs")

        # Display total number of samples
        st.sidebar.write("There are a total of " + str(sum(experiment.number_of_samples_list)) + " samples.")

        # Display information about sample and condition pairings
        for condition in experiment.conditions_list:
            build_replicate_condition_pair(condition, experiment)

        # Display a checkbox for the user to confirm the inputs
        return st.sidebar.checkbox("Confirm the inputs by checking this box")
    except Exception as e:
        raise Exception(f"Error in confirming user inputs: {e}")

def build_replicate_condition_pair(condition, experiment):
    """
    Display information about sample and condition pairings in the Streamlit sidebar.

    Parameters:
    condition (str): The condition label.
    experiment (Experiment): The experiment object containing sample information.

    Note:
    This function is used for displaying sample-condition pairings and does not perform data processing.
    """
    index = experiment.conditions_list.index(condition)
    samples = experiment.individual_samples_list[index]

    display_text = f"- {' to '.join([samples[0], samples[-1]])} (total {len(samples)}) correspond to {condition}" if len(samples) > 5 else f"- {'-'.join(samples)} correspond to {condition}"
    st.sidebar.write(display_text)

def display_raw_data(df):
    """
    Display the raw data within an expandable section in the Streamlit interface.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the raw data.

    Note:
    This function is intended to simply display the raw data for review in the Streamlit app.
    """
    with st.expander("Raw Data"):
        st.write(df)


def display_data(df, title, filename):
    """
    Display a DataFrame and provide a download button for the data.

    Parameters:
    df (pd.DataFrame): The DataFrame to be displayed.
    title (str): Title for the data section.
    filename (str): The filename for the downloadable data.

    Note:
    Used for presenting both raw and processed data in the Streamlit interface.
    """
    st.write('------------------------------------------------------------------------------------------------')
    st.write(f'View the {title}:')
    st.write(df)
    csv_download = convert_df(df)
    st.download_button(label="Download CSV", data=csv_download, file_name=filename, mime='text/csv')

def display_cleaned_data(df, experiment, name_df):
    """
    Display the cleaned and transformed data within the Streamlit interface.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the raw data.
    experiment (Experiment): The experiment object with setup details.
    name_df (pd.DataFrame): DataFrame containing old and new sample names for renaming.

    Note:
    This function orchestrates the display of cleaned data, including log-transformed and internal standards data.
    """
    with st.expander("Cleaned Data"):
        clean_data_object = lp.CleanData()

        cleaned_df = clean_data_object.data_cleaner(df, name_df, experiment)
        cleaned_df_without_intsta, intsta_df = clean_data_object.extract_internal_standards(cleaned_df)

        display_data(cleaned_df_without_intsta, 'Cleaned Data', 'cleaned_data.csv')

        total_samples = sum(experiment.number_of_samples_list)
        cleaned_df_log = clean_data_object.log_transform_df(cleaned_df_without_intsta.copy(deep=True), total_samples)
        display_data(cleaned_df_log, 'Log-transformed Cleaned Data', 'log_transformed_cleaned_data.csv')

        display_data(intsta_df, 'Internal Standards Data', 'internal_standards.csv')
        intsta_df_log = clean_data_object.log_transform_df(intsta_df.copy(deep=True), total_samples)
        display_data(intsta_df_log, 'Log-transformed Internal Standards Data', 'log_transformed_intsta_df.csv')
        return cleaned_df_without_intsta, intsta_df
        
def display_normalization_options(cleaned_df, intsta_df, experiment):
    """
    Display options for data normalization and process user inputs.

    Parameters:
        cleaned_df (pd.DataFrame): The cleaned DataFrame containing the experiment data.
        intsta_df (pd.DataFrame): DataFrame containing the internal standards data.
        experiment (Experiment): The experiment object with setup details.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    normalized_data_object = lp.NormalizeData()

    # Get the full list of classes
    all_class_lst = list(cleaned_df['ClassKey'].unique())

    # Update all_classes in session state
    st.session_state.all_classes = all_class_lst

    # Initialize selected_classes if it's empty or contains invalid classes
    if 'selected_classes' not in st.session_state or not st.session_state.selected_classes or not all(cls in all_class_lst for cls in st.session_state.selected_classes):
        st.session_state.selected_classes = all_class_lst.copy()

    # Callback function for multiselect
    def update_selected_classes():
        st.session_state.selected_classes = st.session_state.temp_selected_classes

    # Class selection with callback
    selected_classes = st.multiselect(
        'Select lipid classes you would like to analyze:',
        all_class_lst, 
        default=st.session_state.selected_classes,
        key='temp_selected_classes',
        on_change=update_selected_classes
    )

    # Use selected classes for filtering, defaulting to all if none selected
    selected_class_list = st.session_state.selected_classes if st.session_state.selected_classes else all_class_lst

    # Callback function for normalization method
    def update_normalization_method():
        st.session_state.normalization_method = st.session_state.temp_normalization_method

    # Normalization method selection with callback
    normalization_method = st.radio(
        "Select how you would like to normalize your data:",
        ['None', 'Internal Standards', 'BCA Assay', 'Both'],
        index=['None', 'Internal Standards', 'BCA Assay', 'Both'].index(st.session_state.get('normalization_method', 'None')),
        key='temp_normalization_method',
        on_change=update_normalization_method
    )

    # Filter DataFrame based on selected classes
    normalized_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_class_list)].copy()

    if st.session_state.normalization_method != 'None':
        if st.session_state.normalization_method in ['BCA Assay', 'Both']:
            with st.expander("Enter Inputs for BCA Assay Data Normalization"):
                protein_df = collect_protein_concentrations(experiment)
                if protein_df is not None:
                    normalized_df = normalized_data_object.normalize_using_bca(normalized_df, protein_df)
                    st.session_state.normalization_inputs['BCA'] = protein_df

        if st.session_state.normalization_method in ['Internal Standards', 'Both']:
            with st.expander("Enter Inputs For Data Normalization Using Internal Standards"):
                added_intsta_species_lst = st.session_state.normalization_inputs.get('Internal_Standards', {}).get('added_intsta_species_lst', [])
                intsta_concentration_dict = st.session_state.normalization_inputs.get('Internal_Standards', {}).get('intsta_concentration_dict', {})

                new_added_intsta_species_lst, new_intsta_concentration_dict = collect_user_input_for_normalization(
                    normalized_df, intsta_df, selected_class_list, added_intsta_species_lst, intsta_concentration_dict
                )

                normalized_df = normalized_data_object.normalize_data(
                    selected_class_list, new_added_intsta_species_lst, new_intsta_concentration_dict, normalized_df, intsta_df, experiment
                )

                st.session_state.normalization_inputs.setdefault('Internal_Standards', {}).update({
                    'added_intsta_species_lst': new_added_intsta_species_lst,
                    'intsta_concentration_dict': new_intsta_concentration_dict
                })

        # View and download normalized dataset only if normalization method is not 'None'
        view_download_checkbox = st.checkbox(
            "View and Download Normalized Dataset",
            value=st.session_state.get('create_norm_dataset', False),
            key="view_download_normalized_dataset"
        )

        if view_download_checkbox:
            display_data(normalized_df, 'Normalized Data', 'normalized_data.csv')

        # Update create_norm_dataset in session state
        st.session_state.create_norm_dataset = view_download_checkbox

    return normalized_df

def collect_user_input_for_normalization(normalized_df, intsta_df, selected_class_list, added_intsta_species_lst, intsta_concentration_dict):
    """
    Collects user input for normalization process using Streamlit UI.

    Parameters:
        normalized_df (pd.DataFrame): Main dataset.
        intsta_df (pd.DataFrame): Dataset containing internal standards.
        selected_class_list (list): Preselected classes for normalization.
        added_intsta_species_lst (list): Preselected internal standard species.
        intsta_concentration_dict (dict): Preselected concentrations of internal standards.

    Returns:
        tuple: Contains selected internal standards and concentrations.
    """
    intsta_species_lst = intsta_df['LipidMolec'].tolist()

    added_intsta_species_lst = [
        st.selectbox(f'Pick an internal standard for {lipid_class} species', intsta_species_lst, index=intsta_species_lst.index(added_intsta_species_lst[i]) if i < len(added_intsta_species_lst) else 0)
        for i, lipid_class in enumerate(selected_class_list)
    ]

    st.write('Enter the concentration of each internal standard species:')
    intsta_concentration_dict = {
        lipid: st.number_input(
            f'Enter concentration of {lipid} in micromole',
            min_value=0.0, max_value=100000.0, value=intsta_concentration_dict.get(lipid, 1.0), step=0.1
        )
        for lipid in set(added_intsta_species_lst)
    }

    return added_intsta_species_lst, intsta_concentration_dict

def collect_protein_concentrations(experiment):
    """
    Collects protein concentrations for each sample using Streamlit's UI.
    Args:
        experiment (Experiment): The experiment object containing the list of sample names.
    Returns:
        pd.DataFrame: A DataFrame with 'Sample' as a column and 'Concentration' containing the protein concentrations.
    """
    method = st.radio(
        "Select the method for providing protein concentrations:",
        ["Manual Input", "Upload Excel File"],
        index=1
    )
    
    if method == "Manual Input":
        protein_concentrations = {}
        for sample in experiment.full_samples_list:
            concentration = st.number_input(f'Enter protein concentration for {sample} (mg/mL):',
                                            min_value=0.0, max_value=100000.0, value=1.0, step=0.1,
                                            key=sample)  # Unique key for each input
            protein_concentrations[sample] = concentration

        protein_df = pd.DataFrame(list(protein_concentrations.items()), columns=['Sample', 'Concentration'])
    
    elif method == "Upload Excel File":
        st.info("Upload an Excel file with a single column named 'Concentration'. Each row should correspond to the protein concentration for each sample in the experiment, in order.")
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
        
        if uploaded_file is not None:
            protein_df = pd.read_excel(uploaded_file, engine='openpyxl')  # Ensure the correct engine is used
            if len(protein_df) != len(experiment.full_samples_list):
                st.error(f"The number of concentrations in the file ({len(protein_df)}) does not match the number of samples ({len(experiment.full_samples_list)}). Please upload a valid file.")
                return None
            protein_df['Sample'] = experiment.full_samples_list
        else:
            st.warning("Please upload an Excel file to proceed.")
            return None

    return protein_df

def display_box_plots(continuation_df, experiment):
    # Initialize a counter in session state if it doesn't exist
    if 'box_plot_counter' not in st.session_state:
        st.session_state.box_plot_counter = 0
    
    # Increment the counter
    st.session_state.box_plot_counter += 1
    
    # Generate a unique identifier based on the current data and counter
    unique_id = hashlib.md5(f"{str(continuation_df.index.tolist())}_{st.session_state.box_plot_counter}".encode()).hexdigest()
    
    expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
    with expand_box_plot:
        # Creating a deep copy for visualization to keep the original continuation_df intact
        visualization_df = continuation_df.copy(deep=True)
        
        # Ensure the columns reflect the current state of the DataFrame
        current_samples = [sample for sample in experiment.full_samples_list if f'MeanArea[{sample}]' in visualization_df.columns]
        
        mean_area_df = lp.BoxPlot.create_mean_area_df(visualization_df, current_samples)
        zero_values_percent_list = lp.BoxPlot.calculate_missing_values_percentage(mean_area_df)
        
        # Generate and display the first plot (Missing Values Distribution)
        fig1 = lp.BoxPlot.plot_missing_values(current_samples, zero_values_percent_list)
        st.pyplot(fig1)
        matplotlib_svg_download_button(fig1, "missing_values_distribution.svg")

        # Data for download (Missing Values)
        missing_values_data = np.vstack((current_samples, zero_values_percent_list)).T
        missing_values_csv = convert_df(pd.DataFrame(missing_values_data, columns=["Sample", "Percentage Missing"]))
        st.download_button(
            label="Download CSV",
            data=missing_values_csv,
            file_name="missing_values_data.csv",
            mime='text/csv',
            key=f"download_missing_values_data_{unique_id}"
        )
        
        st.write('--------------------------------------------------------------------------------')
        
        # Generate and display the second plot (Box Plot)
        fig2 = lp.BoxPlot.plot_box_plot(mean_area_df, current_samples)
        st.pyplot(fig2)
        matplotlib_svg_download_button(fig2, "box_plot.svg")
        
        # Provide option to download the raw data for Box Plot
        csv_data = convert_df(mean_area_df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="box_plot_data.csv",
            mime='text/csv',
            key=f"download_box_plot_data_{unique_id}"
        )
    
    # Return the figure objects for later use in PDF generation
    return fig1, fig2

def conduct_bqc_quality_assessment(bqc_label, data_df, experiment):
    scatter_plot = None  # Initialize scatter_plot to None
    
    if bqc_label is not None:
        with st.expander("Quality Check Using BQC Samples"):
            bqc_sample_index = experiment.conditions_list.index(bqc_label)
            scatter_plot, prepared_df, reliable_data_percent = lp.BQCQualityCheck.generate_and_display_cov_plot(data_df, experiment, bqc_sample_index)
            
            st.plotly_chart(scatter_plot, use_container_width=True)
            plotly_svg_download_button(scatter_plot, "bqc_quality_check.svg")
            csv_data = convert_df(prepared_df[['LipidMolec', 'cov', 'mean']].dropna())
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="CoV_Plot_Data.csv",
                mime='text/csv'
            )
            
            if reliable_data_percent >= 80:
                st.info(f"{reliable_data_percent}% of the datapoints are confidently reliable (CoV < 30%).")
            elif reliable_data_percent >= 50:
                st.warning(f"{reliable_data_percent}% of the datapoints are confidently reliable.")
            else:
                st.error(f"Less than 50% of the datapoints are confidently reliable (CoV < 30%).")
            
            if prepared_df is not None and not prepared_df.empty:
                filter_option = st.radio("Would you like to filter your data using BQC samples?", ("No", "Yes"), index=0)
                if filter_option == "Yes":
                    cov_threshold = st.number_input('Enter the maximum acceptable CoV in %', min_value=10, max_value=1000, value=30, step=1)
                    filtered_df = lp.BQCQualityCheck.filter_dataframe_by_cov_threshold(cov_threshold, prepared_df)
                    
                    if filtered_df.empty:
                        st.error("The filtered dataset is empty. Please try a higher CoV threshold.")
                        st.warning("Returning the original dataset without filtering.")
                    else:
                        st.write('Filtered dataset:')
                        st.write(filtered_df)
                        csv_download = convert_df(filtered_df)
                        st.download_button(
                            label="Download Filtered Data",
                            data=csv_download,
                            file_name='Filtered_Data.csv',
                            mime='text/csv'
                        )
                        data_df = filtered_df  # Update data_df with the filtered dataset

    return data_df, scatter_plot  # Always return a tuple

def integrate_retention_time_plots(continuation_df):
    """
    Integrates retention time plots into the Streamlit app using Plotly.
    Based on the user's choice, this function either plots individual retention 
    times for each lipid class or allows for comparison across different classes. 
    It also provides options for downloading the plot data in CSV format.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing lipidomic data post-cleaning and normalization.

    Returns:
        plotly.graph_objs._figure.Figure or None: The multi-class retention time comparison plot if in Comparison Mode, else None.
    """
    mode = st.radio('Pick a mode', ['Comparison Mode', 'Individual Mode'])
    if mode == 'Individual Mode':
        # Handling individual retention time plots
        plots = lp.RetentionTime.plot_single_retention(continuation_df)
        for plot, retention_df in plots:
            st.plotly_chart(plot, use_container_width=True)
            plotly_svg_download_button(plot, f"retention_time_plot_{i+1}.svg")
            csv_download = convert_df(retention_df)
            st.download_button(label="Download CSV", data=csv_download, file_name='retention_plot.csv', mime='text/csv')
        return None
    elif mode == 'Comparison Mode':
        # Handling comparison mode for retention time plots
        all_lipid_classes_lst = continuation_df['ClassKey'].value_counts().index.tolist()
        selected_classes_list = st.multiselect('Add/Remove classes:', all_lipid_classes_lst, all_lipid_classes_lst)
        if selected_classes_list:  # Ensuring that selected_classes_list is not empty
            plot, retention_df = lp.RetentionTime.plot_multi_retention(continuation_df, selected_classes_list)
            if plot:
                st.plotly_chart(plot, use_container_width=True)
                plotly_svg_download_button(plot, "retention_time_comparison.svg")
                csv_download = convert_df(retention_df)
                st.download_button(label="Download CSV", data=csv_download, file_name='Retention_Time_Comparison.csv', mime='text/csv')
                return plot
    return None

def display_retention_time_plots(continuation_df):
    """
    Displays retention time plots for lipid species within the Streamlit interface using Plotly.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing lipidomic data after any necessary transformations.

    Returns:
        plotly.graph_objs._figure.Figure or None: The multi-class retention time comparison plot if generated, else None.
    """
    expand_retention = st.expander('View Retention Time Plots: Check Sanity of Data')
    with expand_retention:
        return integrate_retention_time_plots(continuation_df)
        
def analyze_pairwise_correlation(continuation_df, experiment):
    """
    Analyzes pairwise correlations for given conditions in the experiment data.
    This function creates a Streamlit expander for displaying pairwise correlation analysis.
    It allows the user to select a condition from those with multiple replicates and a sample type.
    The function then computes and displays a correlation heatmap for the selected condition.
    
    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.
    
    Returns:
        tuple: A tuple containing the selected condition and the matplotlib figure, or (None, None) if no plot was generated.
    """
    expand_corr = st.expander('Pairwise Correlation Analysis')
    with expand_corr:
        st.info("LipidCruncher removes the missing values before performing the correlation test.")
        # Filter out conditions with only one replicate
        multi_replicate_conditions = [condition for condition, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1]
        # Ensure there are multi-replicate conditions before proceeding
        if multi_replicate_conditions:
            selected_condition = st.selectbox('Select a condition', multi_replicate_conditions)
            condition_index = experiment.conditions_list.index(selected_condition)
            sample_type = st.selectbox('Select the type of your samples', ['biological replicates', 'Technical replicates'])
            mean_area_df = lp.Correlation.prepare_data_for_correlation(continuation_df, experiment.individual_samples_list, condition_index)
            correlation_df, v_min, thresh = lp.Correlation.compute_correlation(mean_area_df, sample_type)
            fig = lp.Correlation.render_correlation_plot(correlation_df, v_min, thresh, experiment.conditions_list[condition_index])
            st.pyplot(fig)
            matplotlib_svg_download_button(fig, f"correlation_plot_{experiment.conditions_list[condition_index]}.svg")
            
            st.write('Find the exact correlation coefficients in the table below:')
            st.write(correlation_df)
            csv_download = convert_df(correlation_df)
            st.download_button(
                label="Download CSV",
                data=csv_download,
                file_name='Correlation_Matrix_' + experiment.conditions_list[condition_index] + '.csv',
                mime='text/csv'
            )
            return selected_condition, fig
        else:
            st.error("No conditions with multiple replicates found.")
            return None, None

def display_pca_analysis(continuation_df, experiment):
    """
    Displays the PCA analysis interface in the Streamlit app and generates a PCA plot using Plotly.
    
    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.
    
    Returns:
        tuple: A tuple containing the updated continuation_df and the PCA plot.
    """
    pca_plot = None
    with st.expander("Principal Component Analysis (PCA)"):
        samples_to_remove = st.multiselect('Select samples to remove from the analysis (optional):', experiment.full_samples_list)
        
        if samples_to_remove:
            if (len(experiment.full_samples_list) - len(samples_to_remove)) >= 2:
                continuation_df = experiment.remove_bad_samples(samples_to_remove, continuation_df)
            else:
                st.error('At least two samples are required for a meaningful analysis!')
        
        pca_plot, pca_df = lp.PCAAnalysis.plot_pca(continuation_df, experiment.full_samples_list, experiment.extensive_conditions_list)
        st.plotly_chart(pca_plot, use_container_width=True)
        plotly_svg_download_button(pca_plot, "pca_plot.svg")
        
        csv_data = convert_df(pca_df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="PCA_data.csv",
            mime="text/csv"
        )
    return continuation_df, pca_plot

def display_volcano_plot(experiment, continuation_df):
    """
    Display a user interface for creating and interacting with volcano plots in lipidomics data using Plotly.
    Returns a dictionary containing the generated plots.
    """
    volcano_plots = {}
    with st.expander("Volcano Plots - Test Hypothesis"):
        conditions_with_replicates = [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1]
        if len(conditions_with_replicates) <= 1:
            st.error('You need at least two conditions with more than one replicate to create a volcano plot.')
            return volcano_plots

        p_value_threshold = st.number_input('Enter the significance level for Volcano Plot', min_value=0.001, max_value=0.1, value=0.05, step=0.001, key="volcano_plot_p_value_threshold")
        q_value_threshold = -np.log10(p_value_threshold)
        control_condition = st.selectbox('Pick the control condition', conditions_with_replicates)
        default_experimental = conditions_with_replicates[1] if len(conditions_with_replicates) > 1 else conditions_with_replicates[0]
        experimental_condition = st.selectbox('Pick the experimental condition', conditions_with_replicates, index=conditions_with_replicates.index(default_experimental))
        selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), list(continuation_df['ClassKey'].value_counts().index))
        
        hide_non_significant = st.checkbox('Hide non-significant data points', value=False)

        plot, merged_df, removed_lipids_df = lp.VolcanoPlot.create_and_display_volcano_plot(experiment, continuation_df, control_condition, experimental_condition, selected_classes_list, q_value_threshold, hide_non_significant)
        st.plotly_chart(plot, use_container_width=True)
        volcano_plots['main'] = plot
        
        # Add SVG download button for the main volcano plot
        plotly_svg_download_button(plot, "volcano_plot.svg")

        # Download options
        csv_data = convert_df(merged_df[['LipidMolec', 'FoldChange', '-log10(pValue)', 'ClassKey']])
        st.download_button("Download CSV", csv_data, file_name="volcano_data.csv", mime="text/csv")
        st.write('------------------------------------------------------------------------------------')
        
        # Generate and display the concentration vs. fold change plot
        color_mapping = lp.VolcanoPlot._generate_color_mapping(merged_df)
        concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(merged_df, color_mapping, q_value_threshold, hide_non_significant)
        st.plotly_chart(concentration_vs_fold_change_plot, use_container_width=True)
        volcano_plots['concentration_vs_fold_change'] = concentration_vs_fold_change_plot
        
        # Add SVG download button for the concentration vs fold change plot
        plotly_svg_download_button(concentration_vs_fold_change_plot, "concentration_vs_fold_change_plot.svg")

        # CSV download option for concentration vs. fold change plot
        csv_data_for_concentration_plot = convert_df(download_df)
        st.download_button("Download CSV", csv_data_for_concentration_plot, file_name="concentration_vs_fold_change_data.csv", mime="text/csv")
        st.write('------------------------------------------------------------------------------------')
        
        # Additional functionality for lipid concentration distribution
        all_classes = list(merged_df['ClassKey'].unique())
        selected_class = st.selectbox('Select Lipid Class:', all_classes)
        if selected_class:
            class_lipids = merged_df[merged_df['ClassKey'] == selected_class]['LipidMolec'].unique().tolist()
            selected_lipids = st.multiselect('Select Lipids:', class_lipids, default=class_lipids[:1])
            if selected_lipids:
                selected_conditions = [control_condition, experimental_condition]
                plot_df = lp.VolcanoPlot.create_concentration_distribution_data(
                    merged_df, selected_lipids, selected_conditions, experiment
                )
                fig = lp.VolcanoPlot.create_concentration_distribution_plot(plot_df, selected_lipids, selected_conditions)
                st.pyplot(fig)
                volcano_plots['concentration_distribution'] = fig
                
                # Add SVG download button for the concentration distribution plot (matplotlib)
                matplotlib_svg_download_button(fig, "concentration_distribution_plot.svg")
                
                csv_data = convert_df(plot_df)
                st.download_button("Download CSV", csv_data, file_name=f"{'_'.join(selected_lipids)}_concentration.csv", mime="text/csv")
        st.write('------------------------------------------------------------------------------------')

        # Displaying the table of invalid lipids
        if not removed_lipids_df.empty:
            st.write("Lipids excluded from the plot (fold change zero or infinity):")
            st.dataframe(removed_lipids_df)
        else:
            st.write("No invalid lipids found.")

    return volcano_plots
        
def display_saturation_plots(experiment, continuation_df):
    saturation_plots = {}
    with st.expander("Saturation Plots"):
        full_samples_list = experiment.full_samples_list
        
        # Get all unique classes from the DataFrame
        all_classes = continuation_df['ClassKey'].unique().tolist()
        
        selected_classes_list = st.multiselect('Select classes for the saturation plot:', all_classes, all_classes)
        
        if selected_classes_list:
            # Filter the DataFrame for selected classes
            filtered_df = continuation_df[continuation_df['ClassKey'].isin(selected_classes_list)]
            
            selected_conditions = st.multiselect('Select conditions for the saturation plot:', experiment.conditions_list, experiment.conditions_list)
            
            if selected_conditions:
                # Add statistical testing guidance based on number of conditions
                if len(selected_conditions) > 2:
                    st.info("""
                    📊 **Statistical Testing for Fatty Acid Comparisons:**
                    - Multiple conditions detected: Using ANOVA + Tukey's test
                    - Each fatty acid type (SFA, MUFA, PUFA) is tested separately
                    - ANOVA determines if there are any differences between conditions
                    - If ANOVA is significant (p < 0.05), Tukey's test shows which specific conditions differ
                    - The analysis automatically adjusts significance thresholds to maintain a 5% false positive rate
                    - Stars (★) indicate significance levels:
                        * ★ p < 0.05
                        * ★★ p < 0.01
                        * ★★★ p < 0.001
                    """)
                elif len(selected_conditions) == 2:
                    st.info("""
                    📊 **Statistical Testing for Fatty Acid Comparisons:**
                    - Two conditions detected: Using Welch's t-test
                    - Each fatty acid type (SFA, MUFA, PUFA) is tested separately
                    - The t-test determines if there are significant differences between conditions
                    - Stars (★) indicate significance levels:
                        * ★ p < 0.05
                        * ★★ p < 0.01
                        * ★★★ p < 0.001
                    - Note: If you plan to compare multiple conditions, select all relevant conditions to use ANOVA
                    """)

                plots = lp.SaturationPlot.create_plots(filtered_df, experiment, selected_conditions)
                
                for lipid_class, (main_plot, percentage_plot, plot_data) in plots.items():
                    st.subheader(f"Saturation Plot for {lipid_class}")
                    
                    st.plotly_chart(main_plot)
                    plotly_svg_download_button(main_plot, f"saturation_plot_main_{lipid_class}.svg")
                    
                    main_csv_download = convert_df(plot_data)
                    st.download_button(
                        label="Download CSV",
                        data=main_csv_download,
                        file_name=f'saturation_plot_main_data_{lipid_class}.csv',
                        mime='text/csv'
                    )
                    
                    st.plotly_chart(percentage_plot)
                    plotly_svg_download_button(percentage_plot, f"saturation_plot_percentage_{lipid_class}.svg")
                    
                    percentage_data = lp.SaturationPlot._calculate_percentage_df(plot_data)
                    percentage_csv_download = convert_df(percentage_data)
                    st.download_button(
                        label="Download CSV",
                        data=percentage_csv_download,
                        file_name=f'saturation_plot_percentage_data_{lipid_class}.csv',
                        mime='text/csv'
                    )
                    
                    saturation_plots[lipid_class] = {'main': main_plot, 'percentage': percentage_plot}
                    
                    st.markdown("---")
    return saturation_plots

def display_abundance_bar_charts(experiment, continuation_df):
    with st.expander("Class Concentration Bar Chart"):
        experiment = st.session_state.experiment
        
        # Filter out conditions with only one sample
        valid_conditions = [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1]
        
        selected_conditions_list = st.multiselect(
            'Add or remove conditions', 
            valid_conditions, 
            valid_conditions,
            key='conditions_select'
        )
        selected_classes_list = st.multiselect(
            'Add or remove classes:',
            list(continuation_df['ClassKey'].value_counts().index), 
            list(continuation_df['ClassKey'].value_counts().index),
            key='classes_select'
        )
        
        linear_fig, log2_fig = None, None
        if selected_conditions_list and selected_classes_list:
            # Statistical testing guidance
            if len(selected_conditions_list) > 2:
                st.info("""
                📊 **Statistical Testing Note:**
                - Multiple conditions detected: Using ANOVA + Tukey's test
                - This is the correct approach when interested in multiple comparisons
                - Do NOT run separate t-tests by unselecting conditions - this would inflate the false positive rate
                - The current analysis automatically adjusts significance thresholds to maintain a 5% false positive rate across all comparisons
                """)
            elif len(selected_conditions_list) == 2:
                st.info("""
                📊 **Statistical Testing Note:**
                - Two conditions detected: Using t-test
                - This is appropriate ONLY for a single pre-planned comparison
                - If you plan to compare multiple conditions, please select all relevant conditions to use ANOVA + Tukey's test
                """)
            
            # Perform statistical tests
            statistical_results = lp.AbundanceBarChart.perform_statistical_tests(
                continuation_df, 
                experiment, 
                selected_conditions_list, 
                selected_classes_list
            )
            
            # Display statistical testing information
            if statistical_results:
                if len(selected_conditions_list) == 2:
                    st.info("""
                    📊 **Statistical Testing Information**
                    - Method: Student t-test
                    - Significance levels:
                         * p < 0.05
                        - ** p < 0.01
                        - *** p < 0.001
                    """)
                else:
                    st.info("""
                    📊 **Statistical Testing Information**
                    - Method: ANOVA + Tukey's test
                    - Multiple comparison correction applied
                    - Significance levels:
                         * p < 0.05
                        - ** p < 0.01
                        - *** p < 0.001
                    """)
            
            # Optional detailed statistical results
            if st.checkbox("Show detailed statistical analysis"):
                display_statistical_details(
                    statistical_results, 
                    selected_conditions_list
                )

            # Generate linear scale chart
            linear_fig, abundance_df = lp.AbundanceBarChart.create_abundance_bar_chart(
                continuation_df, 
                experiment.full_samples_list, 
                experiment.individual_samples_list, 
                experiment.conditions_list, 
                selected_conditions_list, 
                selected_classes_list, 
                'linear scale',
                statistical_results
            )
            if linear_fig is not None and abundance_df is not None and not abundance_df.empty:
                st.pyplot(linear_fig)
                st.write("Linear Scale")
                
                matplotlib_svg_download_button(linear_fig, "abundance_bar_chart_linear.svg")
                
                try:
                    csv_data = convert_df(abundance_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name='abundance_bar_chart_linear.csv',
                        mime='text/csv',
                        key='abundance_chart_download_linear'
                    )
                except Exception as e:
                    st.error(f"Failed to convert DataFrame to CSV: {str(e)}")
            
            # Generate log2 scale chart
            log2_fig, abundance_df_log2 = lp.AbundanceBarChart.create_abundance_bar_chart(
                continuation_df, 
                experiment.full_samples_list, 
                experiment.individual_samples_list, 
                experiment.conditions_list, 
                selected_conditions_list, 
                selected_classes_list, 
                'log2 scale',
                statistical_results
            )
            if log2_fig is not None and abundance_df_log2 is not None and not abundance_df_log2.empty:
                st.pyplot(log2_fig)
                st.write("Log2 Scale")
                
                matplotlib_svg_download_button(log2_fig, "abundance_bar_chart_log2.svg")
                
                try:
                    csv_data_log2 = convert_df(abundance_df_log2)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data_log2,
                        file_name='abundance_bar_chart_log2.csv',
                        mime='text/csv',
                        key='abundance_chart_download_log2'
                    )
                except Exception as e:
                    st.error(f"Failed to convert DataFrame to CSV: {str(e)}")
            
            # Check if any conditions were removed due to having only one sample
            removed_conditions = set(experiment.conditions_list) - set(valid_conditions)
            if removed_conditions:
                st.warning(f"The following conditions were excluded due to having only one sample: {', '.join(removed_conditions)}")
                
        else:
            st.warning("Please select at least one condition and one class to create the charts.")
        
        return linear_fig, log2_fig

def display_statistical_details(statistical_results, selected_conditions):
    """Display detailed statistical analysis"""
    st.write("### Detailed Statistical Analysis")
    
    if len(selected_conditions) == 2:
        st.write(f"Comparing conditions: {selected_conditions[0]} vs {selected_conditions[1]}")
        st.write("Method: Independent t-test (Welch's t-test)")
    else:
        st.write(f"Comparing {len(selected_conditions)} conditions using ANOVA + Tukey's test")
        st.write("Note: P-values are adjusted for multiple comparisons")
    
    for lipid_class, results in statistical_results.items():
        st.write(f"\n#### {lipid_class}")
        p_value = results['p-value']
        
        if results['test'] == 't-test':
            st.write(f"t-statistic: {results['statistic']:.3f}")
            st.write(f"p-value: {p_value:.3f}")
        else:  # ANOVA
            st.write(f"F-statistic: {results['statistic']:.3f}")
            st.write(f"p-value: {p_value:.3f}")
            
            if results.get('tukey_results'):
                st.write("\nTukey's test pairwise comparisons:")
                tukey = results['tukey_results']
                for g1, g2, p in zip(tukey['group1'], tukey['group2'], tukey['p_values']):
                    st.write(f"- {g1} vs {g2}: p = {p:.3f}")
            else:
                st.write("No pairwise comparisons available (ANOVA p-value > 0.05)")

def display_pathway_visualization(experiment, continuation_df):
    """
    Displays an interactive lipid pathway visualization within a Streamlit application, 
    based on lipidomics data from a specified experiment. The function allows users 
    to select control and experimental conditions from the experiment setup. It then 
    calculates the saturation ratio and fold change for each lipid class and generates 
    a comprehensive pathway visualization, highlighting the abundance and saturation 
    levels of different lipid classes.

    The visualization is displayed within an expandable section in the Streamlit interface. 
    Users can also download the visualization as an SVG file for high-quality printing or 
    further use, as well as the underlying data as a CSV file.

    Args:
        experiment: An object containing detailed information about the experimental setup, 
                    including conditions and samples. This object is used to derive 
                    control and experimental conditions for the analysis.
        continuation_df (DataFrame): A DataFrame containing processed lipidomics data, 
                                    which includes information necessary for generating 
                                    the pathway visualization, such as lipid classes, 
                                    molecular structures, and abundances.

    Raises:
        ValueError: If there are not at least two conditions with more than one replicate 
                    in the experiment, which is a prerequisite for creating a meaningful 
                    pathway visualization.
    """

    with st.expander("Lipid Pathway Visualization"):
        # UI for selecting control and experimental conditions
        if len([x for x in experiment.number_of_samples_list if x > 1]) > 1:
            control_condition = st.selectbox('Select Control Condition',
                                             [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1], 
                                             index=0)
            experimental_condition = st.selectbox('Select Experimental Condition',
                                                  [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1 and cond != control_condition], 
                                                  index=1)
    
            # Check if both conditions are selected
            if control_condition and experimental_condition:
                class_saturation_ratio_df = lp.PathwayViz.calculate_class_saturation_ratio(continuation_df)
                class_fold_change_df = lp.PathwayViz.calculate_class_fold_change(continuation_df, experiment, control_condition, experimental_condition)
                
                fig, pathway_dict = lp.PathwayViz.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df, control_condition, experimental_condition)
                
                if fig is not None and pathway_dict:
                    st.pyplot(fig)
                    
                    # Add SVG download button for the pathway visualization
                    matplotlib_svg_download_button(fig, f"pathway_visualization_{control_condition}_vs_{experimental_condition}.svg")
                    
                    pathway_df = pd.DataFrame.from_dict(pathway_dict)
                    pathway_df.set_index('class', inplace=True)
                    st.write(pathway_df)
                    csv_download = convert_df(pathway_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_download,
                        file_name='pathway_df.csv',
                        mime='text/csv')
                    
                    return fig
                    
                else:
                    st.warning("Unable to generate pathway visualization due to insufficient data.")
        else:
            st.warning("At least two conditions with more than one replicate are required for pathway visualization.")
            
        return None

def display_abundance_pie_charts(experiment, continuation_df):
    pie_charts = {}
    with st.expander("Class Concentration Pie Chart"):
        full_samples_list = experiment.full_samples_list
        all_classes = lp.AbundancePieChart.get_all_classes(continuation_df, full_samples_list)
        selected_classes_list = st.multiselect('Select classes for the chart:', all_classes, all_classes)
        if selected_classes_list:
            filtered_df = lp.AbundancePieChart.filter_df_for_selected_classes(continuation_df, full_samples_list, selected_classes_list)
            color_mapping = lp.AbundancePieChart._generate_color_mapping(selected_classes_list)
            for condition, samples in zip(experiment.conditions_list, experiment.individual_samples_list):
                if len(samples) > 1:  # Skip conditions with only one sample
                    fig, df = lp.AbundancePieChart.create_pie_chart(filtered_df, full_samples_list, condition, samples, color_mapping)
                    st.subheader(f"Abundance Pie Chart for {condition}")
                    st.plotly_chart(fig)
                    
                    # Add SVG download button for the pie chart
                    plotly_svg_download_button(fig, f"abundance_pie_chart_{condition}.svg")
                    
                    # Add CSV download button for the pie chart data
                    csv_download = convert_df(df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_download,
                        file_name=f'abundance_pie_chart_{condition}.csv',
                        mime='text/csv'
                    )
                    
                    pie_charts[condition] = fig
                    
                    st.markdown("---")  # Add a separator between charts
    return pie_charts

def display_lipidomic_heatmap(experiment, continuation_df):
    """
    Displays a lipidomic heatmap in the Streamlit app, offering an interactive interface for users to 
    select specific conditions and lipid classes for visualization. This function facilitates the exploration 
    of lipidomic data by generating both clustered and regular heatmaps based on user input.
    Args:
        experiment (Experiment): An object containing detailed information about the experiment setup.
        continuation_df (pd.DataFrame): A DataFrame containing processed lipidomics data.
    Returns:
        tuple: A tuple containing the regular heatmap figure and the clustered heatmap figure (if generated).
    """
    regular_heatmap = None
    clustered_heatmap = None
    with st.expander("Lipidomic Heatmap"):
        # UI for selecting conditions and classes
        all_conditions = experiment.conditions_list
        selected_conditions = st.multiselect("Select conditions:", all_conditions, default=all_conditions)
        selected_conditions = [condition for condition in selected_conditions if len(experiment.individual_samples_list[experiment.conditions_list.index(condition)]) > 1]
        all_classes = continuation_df['ClassKey'].unique()
        selected_classes = st.multiselect("Select lipid classes:", all_classes, default=all_classes)
        
        # UI for selecting number of clusters
        n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=5)
        
        if selected_conditions and selected_classes:
            # Extract sample names based on selected conditions
            selected_samples = [sample for condition in selected_conditions 
                                for sample in experiment.individual_samples_list[experiment.conditions_list.index(condition)]]
            # Process data for heatmap generation
            filtered_df, _ = lp.LipidomicHeatmap.filter_data(continuation_df, selected_conditions, selected_classes, experiment.conditions_list, experiment.individual_samples_list)
            z_scores_df = lp.LipidomicHeatmap.compute_z_scores(filtered_df)
            
            # Generate both regular and clustered heatmaps
            regular_heatmap = lp.LipidomicHeatmap.generate_regular_heatmap(z_scores_df, selected_samples)
            clustered_heatmap, class_percentages = lp.LipidomicHeatmap.generate_clustered_heatmap(z_scores_df, selected_samples, n_clusters)
            
            # Apply layout updates to both heatmaps
            for heatmap, heatmap_type in [(regular_heatmap, "Regular"), (clustered_heatmap, "Clustered")]:
                heatmap.update_layout(
                    width=900,
                    height=600,
                    margin=dict(l=150, r=50, t=50, b=100),
                    xaxis_tickangle=-45,
                    yaxis_title="Lipid Molecules",
                    xaxis_title="Samples",
                    title=f"Lipidomic Heatmap ({heatmap_type})",
                    font=dict(size=10)
                )
                heatmap.update_traces(colorbar=dict(len=0.9, thickness=15))
            
            # Allow users to choose which heatmap to display
            heatmap_type = st.radio("Select Heatmap Type", ["Clustered", "Regular"], index=0)
            current_heatmap = clustered_heatmap if heatmap_type == "Clustered" else regular_heatmap
            
            # Display the selected heatmap
            st.plotly_chart(current_heatmap, use_container_width=True)
            
            # Display cluster information
            if heatmap_type == "Clustered":
                st.subheader(f"Cluster Information (Number of clusters: {n_clusters})")
                st.dataframe(class_percentages.round(2))
                
                # Create a downloadable CSV for cluster information
                csv_cluster_info = convert_df(class_percentages.reset_index())
                st.download_button("Download Cluster Information CSV", csv_cluster_info, 'cluster_information.csv', 'text/csv')
            
            # Download buttons
            plotly_svg_download_button(current_heatmap, f"Lipidomic_{heatmap_type}_Heatmap.svg")
            csv_download = convert_df(z_scores_df.reset_index())
            st.download_button("Download CSV", csv_download, f'z_scores_{heatmap_type}_heatmap.csv', 'text/csv')
    return regular_heatmap, clustered_heatmap   

if __name__ == "__main__":
    main()