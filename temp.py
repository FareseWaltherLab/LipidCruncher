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
        
        plot, merged_df, removed_lipids_df = lp.VolcanoPlot.create_and_display_volcano_plot(experiment, continuation_df, control_condition, experimental_condition, selected_classes_list, q_value_threshold)
        st.bokeh_chart(plot)

        # Download options
        csv_data = convert_df(merged_df[['LipidMolec', 'FoldChange', '-log10(pValue)', 'ClassKey']])
        st.download_button("Download CSV", csv_data, file_name="volcano_data.csv", mime="text/csv")
        #svg_data = bokeh_plot_as_svg(plot)
        #st.download_button("Download SVG", svg_data, file_name="volcano_plot.svg", mime="image/svg+xml")
        st.write('------------------------------------------------------------------------------------')
        
        # Generate and display the concentration vs. fold change plot
        color_mapping = lp.VolcanoPlot._generate_color_mapping(merged_df)
        concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(merged_df, color_mapping)
        st.bokeh_chart(concentration_vs_fold_change_plot)

        # CSV and SVG download options for concentration vs. fold change plot
        csv_data_for_concentration_plot = convert_df(download_df)
        st.download_button("Download CSV", csv_data_for_concentration_plot, file_name="concentration_vs_fold_change_data.csv", mime="text/csv")
        #svg_data_for_concentration_plot = bokeh_plot_as_svg(concentration_vs_fold_change_plot)
        #st.download_button("Download SVG", svg_data_for_concentration_vs_fold_change_plot, file_name="concentration_vs_fold_change_plot.svg", mime="image/svg+xml")
        st.write('------------------------------------------------------------------------------------')
        
        # Additional functionality for multiple lipid concentration distribution
        class_list = list(merged_df['ClassKey'].unique())
        selected_class = st.selectbox('Select a Lipid Class:', class_list)
        
        most_abundant_lipid = lp.VolcanoPlot.get_most_abundant_lipid(merged_df, selected_class)
        selected_lipids = st.multiselect('Select Lipids:', list(merged_df[merged_df['ClassKey'] == selected_class]['LipidMolec'].unique()), default=[most_abundant_lipid])

        if selected_lipids:
            selected_conditions = [control_condition, experimental_condition]
            plot_df = lp.VolcanoPlot.create_concentration_distribution_data(merged_df, selected_lipids, selected_conditions, experiment)
            fig = lp.VolcanoPlot.create_concentration_distribution_plot(plot_df, selected_lipids)
            st.pyplot(fig)
            csv_data = convert_df(plot_df)
            st.download_button("Download Data", csv_data, file_name=f"{'_'.join(selected_lipids)}_concentration.csv", mime="text/csv")
            #svg_data = plt_plot_to_svg(fig)
            #st.download_button("Download SVG", svg_data, f"{'_'.join(selected_lipids)}_concentration.svg", "image/svg+xml")
        st.write('------------------------------------------------------------------------------------')

        # Displaying the table of invalid lipids
        if not removed_lipids_df.empty:
            st.write("Lipids excluded from the plot (fold change zero or infinity):")
            st.dataframe(removed_lipids_df)
        else:
            st.write("No invalid lipids found.")