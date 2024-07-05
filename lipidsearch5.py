import pandas as pd
import numpy as np
import streamlit as st
import lipidomics as lp
import matplotlib.pyplot as plt
import tempfile
from bokeh.io import export_svg
from bokeh.io.export import export_svgs
from bokeh.plotting import figure
from io import BytesIO
import base64
import plotly.io as pio

# Initialize session state variables
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'intsta_df' not in st.session_state:
    st.session_state.intsta_df = None
if 'continuation_df' not in st.session_state:
    st.session_state.continuation_df = None
if 'normalization_method' not in st.session_state:
    st.session_state.normalization_method = 'None'
if 'normalization_inputs' not in st.session_state:
    st.session_state.normalization_inputs = {}
if 'create_norm_dataset' not in st.session_state:
    st.session_state.create_norm_dataset = False
if 'module' not in st.session_state:
    st.session_state.module = "Data Cleaning, Filtering, & Normalization"
if 'proceed_with_analysis' not in st.session_state:
    st.session_state.proceed_with_analysis = False
if 'remove_samples' not in st.session_state:
    st.session_state.remove_samples = False
if 'samples_to_remove' not in st.session_state:
    st.session_state.samples_to_remove = []
if 'filter_data' not in st.session_state:
    st.session_state.filter_data = False
if 'cov_threshold' not in st.session_state:
    st.session_state.cov_threshold = 30
if 'experiment' not in st.session_state:
    st.session_state.experiment = None
if 'full_samples_list' not in st.session_state:
    st.session_state.full_samples_list = []
if 'individual_samples_list' not in st.session_state:
    st.session_state.individual_samples_list = []
if 'conditions_list' not in st.session_state:
    st.session_state.conditions_list = []
if 'proceed_with_analysis_qc' not in st.session_state:
    st.session_state.proceed_with_analysis_qc = False

def main():
    st.header("LipidSearch 5.0 Module")
    st.markdown("Process, visualize and analyze LipidSearch 5.0 data.")

    module_options = [
        "Data Cleaning, Filtering, & Normalization",
        "Quality Check & Anomaly Detection",
        "Data Visualization, Interpretation, & Analysis"
    ]

    # Initialize session state for module if not already set
    if 'module' not in st.session_state or st.session_state.module not in module_options:
        st.session_state.module = module_options[0]

    # Move the module selection dropdown to the main page
    selected_module = st.selectbox("Choose a module to proceed", module_options, index=module_options.index(st.session_state.module))

    # Update the selected module in the session state immediately if different
    if selected_module != st.session_state.module:
        st.session_state.module = selected_module

    uploaded_file = st.sidebar.file_uploader('Upload your LipidSearch 5.0 dataset', type=['csv', 'txt'])
    if uploaded_file:
        df = load_data(uploaded_file)
        confirmed, name_df, experiment, bqc_label, valid_samples = process_experiment(df)

        if confirmed and valid_samples:
            st.session_state.name_df = name_df
            st.session_state.experiment = experiment
            st.session_state.bqc_label = bqc_label
            st.session_state.full_samples_list = experiment.full_samples_list
            st.session_state.individual_samples_list = experiment.individual_samples_list
            st.session_state.conditions_list = experiment.conditions_list

            # Display the selected module
            if st.session_state.module == "Data Cleaning, Filtering, & Normalization":
                cleaned_df, intsta_df = data_cleaning_module(df, st.session_state.experiment, st.session_state.name_df)
                if cleaned_df is not None and intsta_df is not None:
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.intsta_df = intsta_df
                    st.session_state.continuation_df = cleaned_df  # Ensure continuation_df is initially the cleaned_df

            elif st.session_state.module == "Quality Check & Anomaly Detection" and st.session_state.cleaned_df is not None:
                if st.session_state.proceed_with_analysis:
                    continuation_df, updated_experiment = quality_check_and_anomaly_detection_module(
                        st.session_state.cleaned_df, 
                        st.session_state.intsta_df, 
                        st.session_state.experiment
                    )
                    st.session_state.continuation_df = continuation_df
                    st.session_state.experiment = updated_experiment
                    # Update experiment-related session state variables
                    st.session_state.full_samples_list = updated_experiment.full_samples_list
                    st.session_state.individual_samples_list = updated_experiment.individual_samples_list
                    st.session_state.conditions_list = updated_experiment.conditions_list
                else:
                    st.warning("Please confirm your normalization choices in the previous module before proceeding.")

            elif st.session_state.module == "Data Visualization, Interpretation, & Analysis" and st.session_state.continuation_df is not None:
                if st.session_state.proceed_with_analysis_qc:
                    analysis_module(st.session_state.continuation_df, st.session_state.experiment)
                else:
                    st.warning("Please confirm your quality check choices in the previous module before proceeding.")

def data_cleaning_module(df, experiment, name_df):
    st.subheader("1) Clean, Filter, & Normalize Data")
    display_raw_data(df)
    cleaned_df, intsta_df = display_cleaned_data(df, experiment, name_df)
    proceed_with_analysis, normalized_df = display_normalization_options(cleaned_df, intsta_df, experiment)
    if proceed_with_analysis:
        st.session_state.continuation_df = normalized_df  # Store the normalized_df in session state
        return normalized_df, intsta_df
    return None, None

def quality_check_and_anomaly_detection_module(cleaned_df, intsta_df, experiment):
    st.subheader("2) Quality Check & Anomaly Detection")
    
    # Perform PCA analysis and get updated DataFrame and experiment object
    continuation_df, experiment = display_pca_analysis(cleaned_df, experiment, key_prefix="initial_pca")
    
    display_box_plots(continuation_df, experiment)
    continuation_df = conduct_bqc_quality_assessment(st.session_state.bqc_label, continuation_df, experiment)
    display_retention_time_plots(continuation_df)
    
    st.subheader("3) Detect & Remove Anomalies")
    analyze_pairwise_correlation(continuation_df, experiment)
    continuation_df, experiment = display_pca_analysis(continuation_df, experiment, key_prefix="final_pca")
    
    st.session_state.continuation_df = continuation_df
    st.session_state.experiment = experiment

    # Add a button to confirm quality check choices
    if st.button("Confirm Quality Check Choices", key="confirm_quality_check"):
        st.session_state.proceed_with_analysis_qc = True
        st.info("Quality check choices confirmed. You can now proceed with further analysis.")
    else:
        st.session_state.proceed_with_analysis_qc = False

    return continuation_df, experiment

def analysis_module(continuation_df, experiment):
    st.subheader("4) Data Visualization, Interpretation, & Analysis")
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
        display_abundance_bar_chart(experiment, continuation_df)
    elif analysis_option == "Class Level Breakdown - Pie Charts":
        display_abundance_pie_charts(experiment, continuation_df)
    elif analysis_option == "Class Level Breakdown - Saturation Plots":
        display_saturation_plots(experiment, continuation_df)
    elif analysis_option == "Class Level Breakdown - Pathway Visualization":
        display_pathway_visualization(experiment, continuation_df)
    elif analysis_option == "Species Level Breakdown - Volcano Plot":
        display_volcano_plot(experiment, continuation_df)
    elif analysis_option == "Species Level Breakdown - Lipidomic Heatmap":
        display_lipidomic_heatmap(experiment, continuation_df)

@st.cache_data
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
        
def plt_plot_to_svg(fig):
    """
    Converts a matplotlib figure to an SVG string.

    This function takes a matplotlib figure object and converts it into an SVG string. 
    It uses a BytesIO stream as an intermediary buffer. After the figure is saved 
    into this buffer in SVG format, the buffer's content is decoded from bytes to a 
    UTF-8 string, which is the standard encoding for SVG content. The figure is closed 
    after saving to free up resources.

    Parameters:
    fig (matplotlib.figure.Figure): A matplotlib figure object to be converted into SVG.

    Returns:
    str: The SVG representation of the input figure as a string. This string can be 
         directly used for displaying in web interfaces or saved to an SVG file.

    Note:
    This function is especially useful when working with web frameworks or applications 
    like Streamlit, where it is often required to convert plots to SVG format for better 
    rendering in web browsers.
    """
    output = BytesIO()
    fig.savefig(output, format='svg')
    plt.close(fig)
    output.seek(0)
    svg_data = output.getvalue().decode('utf-8')  # Decoding bytes to string
    return svg_data  # Returning SVG data as string

def bokeh_plot_as_svg(plot):
    """
    Save a Bokeh plot as an SVG string and return the SVG data.

    Args:
        plot (Bokeh.plotting.figure): Bokeh plot to be saved as SVG.
    
    Returns:
        str: A string containing the SVG representation of the plot.
    """
    # Ensure the plot has a title, otherwise export_svgs will not work
    plot.title.text = plot.title.text if plot.title.text else "Plot Title"
    
    # Configure plot for SVG export
    plot.output_backend = "svg"
    
    # Export the SVG to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_svg_file:
        export_svgs(plot, filename=temp_svg_file.name)
        temp_svg_file.seek(0)
        svg_data = temp_svg_file.read().decode('utf-8')
        
    return svg_data

def svg_download_button(fig, filename):
    """
    Creates a Streamlit download button for the SVG format of a plotly figure.

    Args:
        fig (plotly.graph_objs.Figure): Plotly figure object to be converted into SVG.
        filename (str): The desired name of the downloadable SVG file.
    """
    svg = fig.to_image(format="svg")
    b64 = base64.b64encode(svg).decode()
    st.download_button(
        label="Download SVG",
        data=b64,
        file_name=filename,
        mime="image/svg+xml"
    )

@st.cache_data
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
    st.download_button(label="Download Data", data=csv_download, file_name=filename, mime='text/csv')

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
        tuple: A tuple containing a boolean indicating if normalization dataset is to be created and the normalized DataFrame.
    """
    normalized_data_object = lp.NormalizeData()  # Assume NormalizeData class is properly defined and imported
    proceed_with_analysis = False

    # Use local variables to track changes within the module
    local_normalization_method = st.radio(
        "Select how you would like to normalize your data:",
        ['None', 'Internal Standards', 'BCA Assay', 'Both'],
        index=['None', 'Internal Standards', 'BCA Assay', 'Both'].index(st.session_state.normalization_method)
    )

    local_normalization_inputs = st.session_state.normalization_inputs.get('Internal_Standards', {})
    local_create_norm_dataset = st.session_state.create_norm_dataset

    if local_normalization_method == 'None':
        normalized_df = cleaned_df
    else:
        normalized_df = cleaned_df.copy()  # Start with a copy of the cleaned data frame

        if local_normalization_method in ['BCA Assay', 'Both']:
            with st.expander("Enter Inputs for BCA Assay Data Normalization"):
                protein_df = collect_protein_concentrations(experiment)
                if protein_df is not None:
                    normalized_df = normalized_data_object.normalize_using_bca(normalized_df, protein_df)
                    local_normalization_inputs['BCA'] = protein_df  # Store inputs for BCA Assay

        if local_normalization_method in ['Internal Standards', 'Both']:
            with st.expander("Enter Inputs For Data Normalization Using Internal Standards"):
                # Retrieve stored values or set defaults
                all_class_lst = cleaned_df['ClassKey'].unique()
                selected_class_list = local_normalization_inputs.get('selected_class_list', list(all_class_lst))
                added_intsta_species_lst = local_normalization_inputs.get('added_intsta_species_lst', [])
                intsta_concentration_dict = local_normalization_inputs.get('intsta_concentration_dict', {})

                new_selected_class_list, new_added_intsta_species_lst, new_intsta_concentration_dict = collect_user_input_for_normalization(
                    normalized_df, intsta_df, selected_class_list, added_intsta_species_lst, intsta_concentration_dict
                )

                normalized_df = normalized_data_object.normalize_data(
                    new_selected_class_list, new_added_intsta_species_lst, new_intsta_concentration_dict, normalized_df, intsta_df, experiment
                )

                # Store new values only if there is a change
                local_normalization_inputs = {
                    'selected_class_list': new_selected_class_list,
                    'added_intsta_species_lst': new_added_intsta_species_lst,
                    'intsta_concentration_dict': new_intsta_concentration_dict
                }

        local_create_norm_dataset = st.checkbox("View and Download Normalized Dataset", value=local_create_norm_dataset)

        if local_create_norm_dataset:
            st.info('The normalized dataset is created! Please confirm your normalization choices to proceed.')
            display_data(normalized_df, 'Normalized Data', 'normalized_data.csv')

    # Add a button to confirm changes and update session state
    if st.button("Confirm Normalization Choices"):
        st.session_state.normalization_method = local_normalization_method
        st.session_state.normalization_inputs['Internal_Standards'] = local_normalization_inputs
        st.session_state.create_norm_dataset = local_create_norm_dataset
        st.session_state.proceed_with_analysis = True
        st.session_state.proceed_with_analysis_qc = False  # Reset QC state
        st.info("Normalization choices confirmed. You can now proceed with further analysis.")
    else:
        st.session_state.proceed_with_analysis = False

    return st.session_state.proceed_with_analysis, normalized_df

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
        tuple: Contains selected classes, internal standards, and concentrations.
    """
    all_class_lst = normalized_df['ClassKey'].unique()
    intsta_species_lst = intsta_df['LipidMolec'].tolist()

    selected_class_list = st.multiselect(
        'Select lipid classes you would like to analyze. For each selected class, you will choose an internal standard for normalization. '
        'If a matching internal standard lipid species is available, it should be used for normalization. If no direct match is available, '
        'select an internal standard that most closely matches the structure of the lipid class in question from the available options.',
        all_class_lst, selected_class_list
    )

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

    return selected_class_list, added_intsta_species_lst, intsta_concentration_dict

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
    expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
    with expand_box_plot:
        # Creating a deep copy for visualization to keep the original continuation_df intact
        visualization_df = continuation_df.copy(deep=True)
        
        # Ensure the columns reflect the current state of the DataFrame
        current_samples = [sample for sample in experiment.full_samples_list if f'MeanArea[{sample}]' in visualization_df.columns]
        
        mean_area_df = lp.BoxPlot.create_mean_area_df(visualization_df, current_samples)
        zero_values_percent_list = lp.BoxPlot.calculate_missing_values_percentage(mean_area_df)
        fig = lp.BoxPlot.plot_missing_values(current_samples, zero_values_percent_list)
        
        st.write('--------------------------------------------------------------------------------')

        fig = lp.BoxPlot.plot_box_plot(mean_area_df, current_samples)

def conduct_bqc_quality_assessment(bqc_label, data_df, experiment):
    if bqc_label is not None:
        expand_quality_check = st.expander("Quality Check Using BQC Samples")
        with expand_quality_check:
            bqc_sample_index = experiment.conditions_list.index(bqc_label)
            scatter_plot, prepared_df, reliable_data_percent = lp.BQCQualityCheck.generate_and_display_cov_plot(data_df, experiment, bqc_sample_index)
            
            st.bokeh_chart(scatter_plot)
            csv_data = convert_df(prepared_df[['LipidMolec', 'cov', 'mean']].dropna())
            st.download_button(
                "Download Data",
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
                st.session_state.filter_data = st.radio('Would you like to filter the data using BQC samples?', ['Yes', 'No'], 1) == 'Yes'
                if st.session_state.filter_data:
                    st.session_state.cov_threshold = st.number_input('Enter the maximum acceptable CoV in %', min_value=10, max_value=1000, value=st.session_state.cov_threshold, step=1)
                    filtered_df = lp.BQCQualityCheck.filter_dataframe_by_cov_threshold(st.session_state.cov_threshold, prepared_df)
                    st.write('View and download the filtered dataset:')
                    st.write(filtered_df)
                    csv_download = convert_df(filtered_df)
                    st.download_button(
                        label="Download Data",
                        data=csv_download,
                        file_name='Filtered_Data.csv',
                        mime='text/csv'
                    )
                    return filtered_df
    return data_df

def integrate_retention_time_plots(continuation_df):
    """
    Integrates retention time plots into the Streamlit app.

    Based on the user's choice, this function either plots individual retention 
    times for each lipid class or allows for comparison across different classes. 
    It also provides options for downloading the plot data in CSV and SVG formats.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing lipidomic data post-cleaning and normalization.
    """
    mode = st.radio('Pick a mode', ['Comparison Mode', 'Individual Mode'])
    if mode == 'Individual Mode':
        # Handling individual retention time plots
        plots = lp.RetentionTime.plot_single_retention(continuation_df)
        for plot, retention_df in plots:
            st.bokeh_chart(plot)
            csv_download = convert_df(retention_df)
            #svg = bokeh_plot_as_svg(plot)  # Using the refactored method to get SVG
            st.download_button(label="Download Data", data=csv_download, file_name='retention_plot.csv', mime='text/csv')
            #st.download_button(label="Download SVG", data=svg, file_name='retention_plot.svg', mime='image/svg+xml')

    elif mode == 'Comparison Mode':
        # Handling comparison mode for retention time plots
        all_lipid_classes_lst = continuation_df['ClassKey'].value_counts().index.tolist()
        selected_classes_list = st.multiselect('Add/Remove classes:', all_lipid_classes_lst, all_lipid_classes_lst)
        if selected_classes_list:  # Ensuring that selected_classes_list is not empty
            plot, retention_df = lp.RetentionTime.plot_multi_retention(continuation_df, selected_classes_list)
            if plot:
                st.bokeh_chart(plot)
                csv_download = convert_df(retention_df)
                #svg = bokeh_plot_as_svg(plot)  # Using the refactored method to get SVG
                st.download_button(label="Download Data", data=csv_download, file_name='Retention_Time_Comparison.csv', mime='text/csv')
                #st.download_button(label="Download SVG", data=svg, file_name='Retention_Time_Comparison.svg', mime='image/svg+xml')

def display_retention_time_plots(continuation_df):
    """
    Displays retention time plots for lipid species within the Streamlit interface.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing lipidomic data after any necessary transformations.
    """
    expand_retention = st.expander('View Retention Time Plots: Check Sanity of Data')
    with expand_retention:
        integrate_retention_time_plots(continuation_df)
        
def analyze_pairwise_correlation(continuation_df, experiment):
    """
    Analyzes pairwise correlations for given conditions in the experiment data.

    This function creates a Streamlit expander for displaying pairwise correlation analysis.
    It allows the user to select a condition from those with multiple replicates and a sample type.
    The function then computes and displays a correlation heatmap for the selected condition.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.

    """
    expand_corr = st.expander('Pairwise Correlation Analysis')
    with expand_corr:
        st.info("LipidCruncher removes the missing values before performing the correlation test.")

        # Filter out conditions with only one replicate
        multi_replicate_conditions = [condition for condition, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1]

        # Ensure there are multi-replicate conditions before proceeding
        if multi_replicate_conditions:
            condition_index = experiment.conditions_list.index(st.selectbox('Select a condition', multi_replicate_conditions))
            sample_type = st.selectbox('Select the type of your samples', ['biological replicates', 'Technical replicates'])

            mean_area_df = lp.Correlation.prepare_data_for_correlation(continuation_df, experiment.individual_samples_list, condition_index)
            correlation_df, v_min, thresh = lp.Correlation.compute_correlation(mean_area_df, sample_type)
            fig = lp.Correlation.render_correlation_plot(correlation_df, v_min, thresh, experiment.conditions_list[condition_index])
            st.pyplot(fig)
            #svg = plt_plot_to_svg(fig)  # Convert matplotlib figure to SVG using the existing function
            #st.download_button(label="Download SVG", data=svg, file_name=f'Correlation_Matrix_{experiment.conditions_list[condition_index]}.svg', mime="image/svg+xml")
            
            st.write('Find the exact correlation coefficients in the table below:')
            st.write(correlation_df)
            csv_download = convert_df(correlation_df)
            st.download_button(
                label="Download Data",
                data=csv_download,
                file_name='Correlation_Matrix_' + experiment.conditions_list[condition_index] + '.csv',
                mime='text/csv'
            )
        else:
            st.error("No conditions with multiple replicates found.")

def display_pca_analysis(continuation_df, experiment, key_prefix=""):
    with st.expander("Principal Component Analysis (PCA)"):
        # Generate a unique key for this instance of the function
        unique_key = f"{key_prefix}_{id(continuation_df)}"
        
        st.session_state.remove_samples = st.radio(
            "Would you like to remove any samples from the analysis?",
            ['No', 'Yes'],
            index=0 if not st.session_state.get(f'remove_samples_{unique_key}', False) else 1,
            key=f"remove_samples_radio_{unique_key}"
        ) == 'Yes'
        
        # Store the state in session state with a unique key
        st.session_state[f'remove_samples_{unique_key}'] = st.session_state.remove_samples
        
        if st.session_state.remove_samples:
            st.warning('The samples you remove now will be removed for the rest of the analysis.')
            st.session_state.samples_to_remove = st.multiselect(
                'Pick the sample(s) that you want to remove from the analysis',
                experiment.full_samples_list,
                default=st.session_state.get(f'samples_to_remove_{unique_key}', []),
                key=f"samples_to_remove_multiselect_{unique_key}"
            )
            
            # Store the selected samples in session state with a unique key
            st.session_state[f'samples_to_remove_{unique_key}'] = st.session_state.samples_to_remove
            
            if (len(experiment.full_samples_list) - len(st.session_state.samples_to_remove)) >= 2 and len(st.session_state.samples_to_remove) > 0:
                continuation_df = experiment.remove_bad_samples(st.session_state.samples_to_remove, continuation_df)
                # Update session state
                st.session_state.continuation_df = continuation_df
                st.session_state.experiment = experiment
                st.session_state.full_samples_list = experiment.full_samples_list
                st.session_state.individual_samples_list = experiment.individual_samples_list
                st.session_state.conditions_list = experiment.conditions_list
            elif (len(experiment.full_samples_list) - len(st.session_state.samples_to_remove)) < 2:
                st.error('At least two samples are required for a meaningful analysis!')
        
        # Generate and display the PCA plot
        pca_plot, pca_df = lp.PCAAnalysis.plot_pca(continuation_df, experiment.full_samples_list, experiment.extensive_conditions_list)
        st.bokeh_chart(pca_plot)
        
        csv_data = convert_df(pca_df)
        st.download_button(
            label="Download Data",
            data=csv_data,
            file_name="PCA_data.csv",
            mime="text/csv",
            key=f"pca_data_download_{unique_key}"
        )
    
    return continuation_df, experiment

def display_volcano_plot(experiment, continuation_df):
    """
    Display a user interface for creating and interacting with volcano plots in lipidomics data.

    Args:
        experiment: An object containing experiment details such as conditions and sample lists.
        continuation_df: DataFrame containing continuation data for volcano plot creation.

    This function creates a user interface section for volcano plots. It allows users to select control and experimental 
    conditions, set significance levels, choose lipid classes for the plot, and view the resulting plot. Users can also 
    download the plot data in CSV and SVG formats.
    """
    with st.expander("Volcano Plots - Test Hypothesis"):
        conditions_with_replicates = [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1]
        if len(conditions_with_replicates) <= 1:
            st.error('You need at least two conditions with more than one replicate to create a volcano plot.')
            return
        
        p_value_threshold = st.number_input('Enter the significance level for Volcano Plot', min_value=0.001, max_value=0.1, value=0.05, step=0.001, key="volcano_plot_p_value_threshold")
        q_value_threshold = -np.log10(p_value_threshold)
        control_condition = st.selectbox('Pick the control condition', conditions_with_replicates)
        default_experimental = conditions_with_replicates[1] if len(conditions_with_replicates) > 1 else conditions_with_replicates[0]
        experimental_condition = st.selectbox('Pick the experimental condition', conditions_with_replicates, index=conditions_with_replicates.index(default_experimental))
        selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), list(continuation_df['ClassKey'].value_counts().index))
        
        hide_non_significant = st.checkbox('Hide non-significant data points', value=False)

        plot, merged_df, removed_lipids_df = lp.VolcanoPlot.create_and_display_volcano_plot(experiment, continuation_df, control_condition, experimental_condition, selected_classes_list, q_value_threshold, hide_non_significant)
        st.bokeh_chart(plot)

        # Download options
        csv_data = convert_df(merged_df[['LipidMolec', 'FoldChange', '-log10(pValue)', 'ClassKey']])
        st.download_button("Download CSV", csv_data, file_name="volcano_data.csv", mime="text/csv")
        #svg_data = bokeh_plot_as_svg(plot)
        #st.download_button("Download SVG", svg_data, file_name="volcano_plot.svg", mime="image/svg+xml")
        st.write('------------------------------------------------------------------------------------')
        
        # Generate and display the concentration vs. fold change plot
        color_mapping = lp.VolcanoPlot._generate_color_mapping(merged_df)
        concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(merged_df, color_mapping, q_value_threshold, hide_non_significant)
        st.bokeh_chart(concentration_vs_fold_change_plot)

        # CSV and SVG download options for concentration vs. fold change plot
        csv_data_for_concentration_plot = convert_df(download_df)
        st.download_button("Download CSV", csv_data_for_concentration_plot, file_name="concentration_vs_fold_change_data.csv", mime="text/csv")
        #svg_data_for_concentration_plot = bokeh_plot_as_svg(concentration_vs_fold_change_plot)
        #st.download_button("Download SVG", svg_data_for_concentration_plot, file_name="concentration_vs_fold_change_plot.svg", mime="image/svg+xml")
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
                csv_data = convert_df(plot_df)
                st.download_button("Download Data", csv_data, file_name=f"{'_'.join(selected_lipids)}_concentration.csv", mime="text/csv")
                #svg_data = plt_plot_to_svg(fig)
                #st.download_button("Download SVG", svg_data, file_name=f"{'_'.join(selected_lipids)}_concentration.svg", mime="image/svg+xml")
        st.write('------------------------------------------------------------------------------------')

        # Displaying the table of invalid lipids
        if not removed_lipids_df.empty:
            st.write("Lipids excluded from the plot (fold change zero or infinity):")
            st.dataframe(removed_lipids_df)
        else:
            st.write("No invalid lipids found.")
        
def display_saturation_plots(experiment, df):
    """
    Create and display saturation plots in the Streamlit app.
    Args:
        experiment: Experiment object containing conditions and samples.
        df: DataFrame with the lipidomics data.
    """
    with st.expander("Investigate the Saturation Profile of Different Lipid Classes"):
        selected_conditions = st.multiselect('Select conditions for analysis:', experiment.conditions_list, experiment.conditions_list)
        if selected_conditions:
            plots = lp.SaturationPlot.create_plots(df, experiment, selected_conditions)
            for lipid_class, (main_plot, percentage_plot, plot_data) in plots.items():
                # Display plots and create download buttons in Streamlit script
                st.bokeh_chart(main_plot)
                st.download_button("Download Data", convert_df(plot_data), f'{lipid_class}_saturation_level_plot_main.csv', 'text/csv', key=f'download-main-data-{lipid_class}')
                #st.download_button("Download Main SVG", bokeh_plot_as_svg(main_plot), f'{lipid_class}_saturation_level_plot_main.svg', 'image/svg+xml', key=f'download-main-svg-{lipid_class}')

                st.bokeh_chart(percentage_plot)
                st.download_button("Download Data", convert_df(plot_data), f'{lipid_class}_saturation_level_plot_percentage.csv', 'text/csv', key=f'download-percentage-data-{lipid_class}')
                #st.download_button("Download Percentage SVG", bokeh_plot_as_svg(percentage_plot), f'{lipid_class}_saturation_level_plot_percentage.svg', 'image/svg+xml', key=f'download-percentage-svg-{lipid_class}')
                
                st.write('---------------------------------------------------------')

def display_abundance_bar_chart(experiment, continuation_df):
    with st.expander("Class Concentration Bar Chart"):
        st.write(experiment)
        full_samples_list = st.session_state.full_samples_list
        individual_samples_list = st.session_state.individual_samples_list
        conditions_list = st.session_state.conditions_list

        # Filter out conditions with no samples
        valid_conditions = [cond for cond, samples in zip(conditions_list, individual_samples_list) if samples]

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

        mode = st.radio('Select a mode', ('linear scale', 'log2 scale'), 0, key='scale_mode')

        if selected_conditions_list and selected_classes_list:
            fig, abundance_df = lp.AbundanceBarChart.create_abundance_bar_chart(
                continuation_df, 
                full_samples_list, 
                individual_samples_list, 
                conditions_list, 
                selected_conditions_list, 
                selected_classes_list, 
                mode
            )

            if fig is not None and abundance_df is not None and not abundance_df.empty:
                st.pyplot(fig)
                try:
                    csv_data = convert_df(abundance_df)
                    st.download_button(
                        label="Download Data",
                        data=csv_data,
                        file_name='abundance_bar_chart.csv',
                        mime='text/csv',
                        key='abundance_chart_download'
                    )
                except Exception as e:
                    st.error(f"Failed to convert DataFrame to CSV: {str(e)}")
            else:
                st.error("Unable to create the abundance bar chart due to insufficient data.")
        else:
            st.warning("Please select at least one condition and one class to create the chart.")

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
                # Calculate saturation ratio and fold change
                class_saturation_ratio_df = lp.PathwayViz.calculate_class_saturation_ratio(continuation_df)
                class_fold_change_df = lp.PathwayViz.calculate_class_fold_change(continuation_df, experiment, control_condition, experimental_condition)
    
                # Generate and display the pathway visualization
                fig, pathway_dict = lp.PathwayViz.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df, control_condition, experimental_condition)
    
                st.pyplot(fig)
                
                #svg_data = plt_plot_to_svg(fig)
                #st.download_button(
                    #label="Download SVG",
                    #data=svg_data,
                    #file_name='abundance_bar_chart.svg',
                    #mime='image/svg+xml')
    
                # Display data and download button
                pathway_df = pd.DataFrame.from_dict(pathway_dict)
                pathway_df.set_index('class', inplace=True)
                st.write(pathway_df)
                csv_download = convert_df(pathway_df)
                st.download_button(
                    label="Download Data",
                    data=csv_download,
                    file_name='pathway_df.csv',
                    mime='text/csv')
                
        else:
            st.error('You need at least two conditions with more than one replicate to create a pathway visualization!')

def display_abundance_pie_charts(experiment, continuation_df):
    """
    Displays abundance pie charts in the Streamlit app interface for selected lipid classes.

    This function provides an interactive component in the Streamlit application where users can
    select specific lipid classes to visualize their abundance distribution across different conditions.
    Pie charts are generated for each condition, provided that the condition has more than one sample.
    Additionally, the function allows for downloading the visualized data in both SVG (for the pie charts)
    and CSV formats.

    Args:
        experiment (Experiment): Object containing details about the experiment setup.
        continuation_df (pd.DataFrame): DataFrame containing the lipidomics data after processing.

    The function uses the AbundancePieChart class to perform data manipulation and visualization.
    """

    # Expanding the Streamlit interface section to display pie charts
    with st.expander("Class Concentration Pie Chart"):
        # Extract full_samples_list from the experiment object
        full_samples_list = experiment.full_samples_list

        all_classes = lp.AbundancePieChart.get_all_classes(continuation_df, full_samples_list)
        selected_classes_list = st.multiselect('Select classes for the chart:', all_classes, all_classes)

        if selected_classes_list:
            filtered_df = lp.AbundancePieChart.filter_df_for_selected_classes(continuation_df, full_samples_list, selected_classes_list)
            color_mapping = lp.AbundancePieChart._generate_color_mapping(selected_classes_list)
            for condition, samples in zip(experiment.conditions_list, experiment.individual_samples_list):
                if len(samples) > 1:  # Skip conditions with only one sample
                    fig, df = lp.AbundancePieChart.create_pie_chart(filtered_df, full_samples_list, condition, samples, color_mapping)
                    st.plotly_chart(fig)

                    # Save plot to SVG
                    #svg_data = pio.to_image(fig, format='svg')

                    # Provide option to download SVG
                    #st.download_button(label="Download SVG", data=svg_data, file_name=f'abundance_pie_chart_{condition}.svg', mime='image/svg+xml')

                    # Provide option to download CSV
                    csv_download = convert_df(df)
                    st.download_button("Download Data", csv_download, f'abundance_pie_chart_{condition}.csv', 'text/csv')

def display_lipidomic_heatmap(experiment, continuation_df):
    """
    Displays a lipidomic heatmap in the Streamlit app, offering an interactive interface for users to 
    select specific conditions and lipid classes for visualization. This function facilitates the exploration 
    of lipidomic data by generating either clustered or regular heatmaps based on user input. The clustered heatmap 
    organizes lipid molecules based on hierarchical clustering, while the regular heatmap presents data in its 
    original order. Users can also download the heatmap as an SVG file and the data used for the heatmap as a CSV file.

    Args:
        experiment (Experiment): An object containing detailed information about the experiment setup. This includes 
                                 information about conditions, samples, and other experimental parameters.
        continuation_df (pd.DataFrame): A DataFrame containing processed lipidomics data, which includes information 
                                        about various lipid molecules, their classes, and abundance across different samples.
    """
    with st.expander("Lipidomic Heatmap"):
        # UI for selecting conditions and classes
        all_conditions = experiment.conditions_list
        selected_conditions = st.multiselect("Select conditions:", all_conditions, default=all_conditions)
        selected_conditions = [condition for condition in selected_conditions if len(experiment.individual_samples_list[experiment.conditions_list.index(condition)]) > 1]

        all_classes = continuation_df['ClassKey'].unique()
        selected_classes = st.multiselect("Select lipid classes:", all_classes, default=all_classes)
        
        if selected_conditions and selected_classes:
            # Extract sample names based on selected conditions
            selected_samples = [sample for condition in selected_conditions 
                                for sample in experiment.individual_samples_list[experiment.conditions_list.index(condition)]]

            # Process data for heatmap generation
            filtered_df, _ = lp.LipidomicHeatmap.filter_data(continuation_df, selected_conditions, selected_classes, experiment.conditions_list, experiment.individual_samples_list)
            z_scores_df = lp.LipidomicHeatmap.compute_z_scores(filtered_df)

            # Allow users to choose between clustered and regular heatmap
            heatmap_type = st.radio("Select Heatmap Type", ["Clustered", "Regular"])

            if heatmap_type == "Clustered":
                clustered_df = lp.LipidomicHeatmap.perform_clustering(z_scores_df)
                heatmap_fig = lp.LipidomicHeatmap.generate_clustered_heatmap(clustered_df, selected_samples)
            else:  # Regular heatmap
                heatmap_fig = lp.LipidomicHeatmap.generate_regular_heatmap(z_scores_df, selected_samples)

            # Display the heatmap
            st.plotly_chart(heatmap_fig, use_container_width=True)

            # Download buttons
            #svg_download_button(heatmap_fig, f"Lipidomic_{heatmap_type}_Heatmap.svg")
            csv_download = convert_df(z_scores_df.reset_index())
            st.download_button("Download Data", csv_download, f'z_scores_{heatmap_type}_heatmap.csv', 'text/csv')
        

if __name__ == "__main__":
    main()