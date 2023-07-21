import lipidomics as lp
import streamlit as st
import pandas as pd
import numpy as np

# function for downloading data
@st.cache_data
def convert_df(a_df):
    return a_df.to_csv().encode('utf-8')

st.header("LipidSearch 5.0 Module")

st.markdown("""
                The following module allows the user to process, visualize and analyze LipidSearch 5.0 data. 
                Start by uploading your dataset and completing the next steps on the side bar.
            """)

st.sidebar.subheader("Upload Data")

@st.cache_data
def load_data(a_lipid_search_object):
    return pd.read_csv(a_lipid_search_object)
        
uploaded_f = st.sidebar.file_uploader(label='Upload your LipidSearch 5.0 dataset', type=['csv', 'txt'])
        
if uploaded_f is not None:
        
    # build the side bar
    try:
        confirm = None # initializing variable 'confirm' used later for confirming user inputs
        
        df = load_data(uploaded_f)
        
        st.sidebar.subheader("Define Experiment")
        
        experiment = lp.Experiment() 
        experiment.create_attributes()
        
        st.sidebar.subheader('Group Samples')
        
        grouped_samples_object = lp.GroupSamples(df, experiment)
        grouped_samples_object.check_dataset_validity()
        grouped_samples_object.check_input_validity()
        grouped_samples_object.give_info()
        group_df = grouped_samples_object.build_group_df()
        st.sidebar.write(group_df)
        
        st.sidebar.write('Are your samples properly grouped together?')
        ans = st.sidebar.radio('', ['Yes', 'No'])
        if ans == 'Yes':
            st.sidebar.write('Go to the next section!')
        else:
            group_df = grouped_samples_object.group_samples(group_df)
            st.sidebar.write('Check the updated table below to make sure the samples are properly grouped together:') 
            st.sidebar.write(group_df)
            
        st.sidebar.subheader("Specify Label of BQC Samples")
        bqc_ans = st.sidebar.radio('Do you have Batch Quality Control (BQC) samples?', ['Yes', 'No'], 1)
        bqc_label = None
        if bqc_ans == 'Yes':
            conditions_with_two_plus_samples = \
                            [condition for condition, number_of_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) \
                             if number_of_samples > 1]
            bqc_label = st.sidebar.radio('Which label corresponds to BQC samples?', conditions_with_two_plus_samples, 0)
       
        
        st.sidebar.subheader("Confirm Inputs")
        st.sidebar.markdown("""
                            LipidCruncher uses the following protocol for naming the samples: s1, s2, ..., sN.
                            The table below shows the updated sample names (there is usually no difference between
                            the old names and the updated names unless there is a missing sample or the samples were 
                            not originally grouped together properly):
                            """)
        name_df = grouped_samples_object.update_sample_names(group_df)
        st.sidebar.write(name_df)
        st.sidebar.write("Now, confirm your inputs:")
        confirm = grouped_samples_object.confirm_inputs(group_df)
            
    except TypeError:
        st.sidebar.error('This is not a valid LipidSearch 5.0 dataset!')
    
    except ValueError:
        st.sidebar.error('Invalid total number of samples!')
        
    except NameError:
        st.sidebar.error("Condition's label must be at least one character long!")
        
    except:
        st.sidebar.error("Something went wrong!")
        
    # build the main page
    if confirm:
                    
        st.subheader("1) Clean & Normalize Data")
        
        expand_raw_data = st.expander("Raw Data")
        with expand_raw_data:
            st.write(df)
            csv_download = convert_df(df)
            st.download_button(
                label="Download Data",
                data=csv_download,
                file_name='raw_data.csv',
                mime='text/csv')
            
        expand_cleaned_data = st.expander("Cleaned Data")
        with expand_cleaned_data:
            clean_data_object = lp.CleanData(df, experiment, name_df)
            X_intsta_df = clean_data_object.data_cleaner()
            X, intsta_df = clean_data_object.extract_internal_standards(X_intsta_df.copy(deep=True))
            X_log = clean_data_object.log_transform_df(X.copy(deep=True))
            intsta_df_log = clean_data_object.log_transform_df(intsta_df.copy(deep=True))
            
            st.write('------------------------------------------------------------------------------------------------')
            st.write('View the cleaned data in the conventional format:')
            st.write(X)
            csv_download = convert_df(X)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='cleaned_data.csv',
                            mime='text/csv')
            st.write('------------------------------------------------------------------------------------------------')
            
            st.write('View the cleaned data in the log-transformed format:')
            st.write(X_log)
            csv_download = convert_df(X_log)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='log_transformed_cleaned_data.csv',
                            mime='text/csv')
            st.write('------------------------------------------------------------------------------------------------')
            
            st.write('View the internal standards dataset in the conventional format:')
            st.write(intsta_df)
            csv_download = convert_df(intsta_df)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='internal_standards.csv',
                            mime='text/csv')
            st.write('------------------------------------------------------------------------------------------------')
        
            st.write('View the internal standards dataset in the log-transformed format:')
            st.write(intsta_df_log)
            csv_download = convert_df(intsta_df_log)
            st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='log_transformed_intsta_df.csv',
                            mime='text/csv')
                
        
        continuation_df = X.copy(deep=True)
        X_norm = None # initializing the normalized dataset
        if len(intsta_df) > 0:
            dataset_normality_status = st.radio('Would you like to normalize your data?', ['Yes', 'No'], 1)
            if dataset_normality_status == 'Yes':
                    expand_input_norm_data = st.expander('Enter Inputs For Data Normalization')
                    with expand_input_norm_data:
                         st.info('LipidCruncher normalizes the data using internal standards.')
                         st.write('The following formula is used for data normalization:')
                         latext = r'''
                                   $$ 
                                   Concentration (analyte) = \frac{AUC(analyte)}{AUC(IS)} \times Concentration(IS) 
                                   $$  
                                   '''
                         st.write(latext)
                           
                         normalized_data_object = lp.NormalizeData(X.copy(deep=True), intsta_df.copy(deep=True), experiment)
                         selected_class_list, added_intsta_species_lst, intsta_concentration_dict = normalized_data_object.collect_user_input()
                         
                    st.warning("""  
                               To experience the optimal performance of the app, first verify the the inputs you entered in the section 
                               "Enter Inputs for Data Normalization". Only check this box after you ensure the inputs are accurate. 
                               In case you need to change any of the inputs, first uncheck this box.
                               """)    
                    if st.checkbox("Create, View and Download the Normalized Dataset"):
                        st.info('The normalized dataset is created!')
                        expand_create_norm_data = st.expander('View & Download Normalized Data')
                        with expand_create_norm_data:  
                            X_norm = normalized_data_object.normalization(selected_class_list, added_intsta_species_lst, intsta_concentration_dict)
                            continuation_df = X_norm.copy(deep=True)
                            csv_download = convert_df(X_norm)
                            st.download_button(
                                label="Download Data",
                                data=csv_download,
                                file_name='normalized_data.csv',
                                mime='text/csv')
                            
                        st.subheader("2) Scan Data & Run Quality Checks")
                        
                        expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
                        with expand_box_plot:
                            box_plot_object = lp.BoxPlot(experiment, continuation_df.copy(deep=True))
                            mean_area_df = box_plot_object.create_mean_area_df(box_plot_object.df, experiment.full_samples_list)
                            zero_values_percent_list = box_plot_object.calculate_missing_values_percentage(mean_area_df, experiment.full_samples_list)
                            box_plot_object.plot_missing_values(experiment.full_samples_list, zero_values_percent_list)
                            st.write('--------------------------------------------------------------------------------')
                            box_plot_object.plot_box_plot(mean_area_df, experiment.full_samples_list)
                            
                        if (bqc_label is not None): 
                            expand_filtered_data = st.expander("Check Consistency of Readngs Using BQC Samples")
                            with expand_filtered_data:
                                index = experiment.conditions_list.index(bqc_label)
                                filtered_data_object = lp.FilteredData(continuation_df.copy(deep=True), experiment, bqc_label, index)
                                prepared_df = filtered_data_object.plot_cov()
                                filter_ans = st.radio('Would you like to filter the data using BQC samples?', ['Yes', 'No'], 1)
                                if filter_ans == 'Yes':
                                    thresh = st.number_input('Enter the maximum acceptable CoV in %', min_value = 10, max_value = 1000, value = 30, step = 1)
                                    continuation_df = filtered_data_object.filter_df(thresh, prepared_df.copy(deep=True))
                                    st.write('View and download the filtered dataset:')
                                    st.write(continuation_df)
                                    csv_download = convert_df(continuation_df)
                                    st.download_button(
                                        label="Download Data",
                                        data=csv_download,
                                        file_name='filtered_data.csv',
                                        mime='text/csv')
                                
                        expand_retention = st.expander('View Retention Time Plots: Check Sanity of Data')
                        with expand_retention:
                            st.markdown("""
                                The retention time of a lipid species is a function of its degree of hydrophobicity. 
                                The more hydrophobic the lipid species, the longer the retention time. When retention time is 
                                plotted versus molecular mass, lipid species tend to form separate clusters based upon which lipid class they belong to.
                                
                                Inspect the retention time of lipid species within any lipid class and compare with other lipid classes. 
                                Does everything make sense?
                                """) 
                                
                            retention_time_object = lp.RetentionTime(continuation_df.copy(deep=True), experiment)
                            mode = st.radio('Pick a mode', ['Comparison Mode', 'Individual Mode']) 
                            if mode == 'Individual Mode':
                                retention_time_object.plot_single_retention()
                            elif mode == 'Comparison Mode':
                                retention_time_object.plot_multi_retention()
                                
                        st.subheader("3) Detect & Remove Anomalies")
                    
                        expand_corr = st.expander('Pairwise Correlation Analysis') 
                        with expand_corr:
                            st.markdown("""
                                Typically, the AUC's of any sample is highly linearly correlated to those of its biological replicate
                                (i.e. correlation coefficient > 0.8). This linear correlation is expected to be even stronger for technical replicates 
                                (i.e. correlation coefficient > 0.9).
                                A sample that has a weak correlation with its biological replicates is an outlier. That sample might be an outlier because 
                                of the natural biological variance or an error during sample preparation.
                                
                                Run a correlation test to inspect the degree of correlation between any two biological or technical replicates:
                                """)
                            st.info("LipidCruncher removes the missing values before preforming the correlation test.")
                            condition = st.selectbox('Select a condition', [condition for condition, number_of_samples in \
                                                                            zip(experiment.conditions_list, experiment.number_of_samples_list) if number_of_samples > 1])
                            sample_type = st.selectbox('Select the type of your samples',['biological replicates', 'Technical replicates'])
                            index = experiment.conditions_list.index(condition)
                            correlation_plot_object = lp.Correlation(continuation_df, experiment, condition, sample_type, index)
                            correlation_plot_object.correlation_plot()
                            
                        expand_pca = st.expander('Principal Component Analysis (PCA)')
                        with expand_pca:
                            st.markdown("""
                                Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the 
                                dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains 
                                most of the information in the large set.
                                
                                Typically, biological replicates are expected to cluster together with the exception of rare outliers that fall 
                                further away from the rest of the replicates. 
                                
                                A plot of the top two principal components (PC1 and PC2) against each other is the best way to inspect the clustering 
                                of different samples. This is because the PC's are ordered based on how much of the variability in the data 
                                they can explain (i.e. PC1 is the PC with the highest variance explained ratio, PC2 is the PC with the second highest 
                                variance expalined ratio and so on).
                                
                                Run PCA to inspect the clustering of different samples:
                                """)
                            st.info("LipidCruncher does NOT remove missing values before performng PCA analysis.")
                            
                            remove_ans = st.radio("Would you like to remove any samples from the analysis?", ['Yes', 'No'], 1)
                            if remove_ans == 'Yes':
                                st.warning('The samples you remove now, will be removed for the rest of the analysis.')
                                list_of_bad_samples = st.multiselect('Pick the sample(s) that you want to remove from the analysis', experiment.full_samples_list)
                                if (len(experiment.full_samples_list) - len(list_of_bad_samples)) >= 2 and len(list_of_bad_samples) > 0:
                                    continuation_df = experiment.remove_bad_samples(list_of_bad_samples, continuation_df.copy(deep=True))
                                elif (len(experiment.full_samples_list) - len(list_of_bad_samples)) < 2:
                                    st.error('At least two samples are required for a meanigful analysis!')
                                    
                            pca_object = lp.PCA(continuation_df, experiment)
                            pca_object.plot_pca()
                            
                        st.subheader("4) Analyze Data & Test Hypothesis")
                        
                        expand_vol_plot = st.expander("Volcano Plots - Test Hypothesis")
                        with expand_vol_plot:
                            st.markdown("""
                                        In statistics, a volcano plot is a type of scatter-plot that is used to quickly identify changes in \
                                        large data sets composed of replicate data. It plots significance versus fold-change on the y and x axes, respectively. \
                                        A volcano plot combines a measure of statistical significance from a statistical test (e.g., a p value from a T-test) \
                                        with the magnitude of the change, enabling quick visual identification of those data-points that display large magnitude\
                                        changes that are also statistically significant (datapoints at the top left and top right quadrant).
                                        
                                        Below, q-value (i.e. -log10(p-value)) of each lipid species is plotted versus the fold change of that species. \
                                        The p-value is computed from a two-sample T-test and the fold change is computed from the following formula:
                                        """)
                            latext = r'''
                            $$ 
                            Fold Change = log2(\frac{Mean AUC(Condition 1)}{Mean AUC(Condition 2)})
                            $$  
                            '''
                            st.write(latext)
                            
                            if len([x for x in experiment.number_of_samples_list if x>1]) > 1:
                                control_condition = st.selectbox('pick the control condition', \
                                                                 [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 0)
                                experimental_condition = st.selectbox('Pick the experimental condition', \
                                                                      [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 1)
                                p_value_threshold = st.number_input('Enter the significance level', min_value = 0.001, max_value= 0.1, value = 0.05, step = 0.001)
                                q_value_threshold = -np.log10(p_value_threshold)
                                
                                volcano_object = lp.VolcanoPlot(experiment, continuation_df.copy(deep=True), control_condition, experimental_condition, p_value_threshold, q_value_threshold)
                                volcano_df = volcano_object.add_fold_change_and_p_value_columns()
                                
                                selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), \
                                                                       list(continuation_df['ClassKey'].value_counts().index)[: 3])
                                if len(selected_classes_list) > 30:
                                    st.error('You can only compare up to 30 lipid classes at a time!')
                                else:
                                    volcano_object.create_volcano_plot(volcano_df, selected_classes_list)
                                
                            else:
                                st.error('You need at least two conditions with more than one replicate to create a volcano plot!')
                                
                        expand_sat_plot = st.expander("Saturation Level Plots - Investigate Saturation Profile of Different Lipid Classes")
                        with expand_sat_plot:  
                             st.markdown("""
                                         Saturation level plots show the saturation profile of each lipid class. 
                                         First, for each lipid species, the ratio of Saturated Fatty Acids (SFA), Mono Unsaturated \
                                         Fatty Acids (MUFA) and Poly Unsaturated Fatty Acids (PUFA) is calculated as following:
                                             
                                         SFA ratio = total number of saturated fatty acids / total number of fatty acids
                                         
                                         MUFA ratio = total number of mono unsaturated fatty acids / total number of fatty acids
                                         
                                         PUFA ratio = total number of poly unsaturated fatty acids / total number of fatty acids
                                         
                                         Then, for each lipid species, the abundance of SFA, MUFA and PUFA is calculated as following:
                                             
                                         AUC(SFA) = (AUC averaged over all replicates).(SFA ratio)
                                         
                                         AUC(MUFA) = (AUC averaged over all replicates).(MUFA ratio)
                                         
                                         AUC(PUFA) = (AUC averaged over all replicates).(PUFA ratio)
                                         
                                         Finally, total AUC(SFA), AUC(MUFA) and AUC(PUFA) for each lipid class is calculated by taking the sum of \
                                         AUC(SFA), AUC(MUFA) and AUC(PUFA) over all lipid species that belong to that class. 
                                         """)
                                         
                             selected_conditions_list = st.multiselect('Add or remove conditions', experiment.conditions_list, experiment.conditions_list)
                             sat_plot_object = lp.SaturationPlot(experiment, continuation_df.copy(deep=True))
                             sat_plot_object.create_saturation_plot(selected_conditions_list)
                             
                        expand_abundance_bar_chart = st.expander("Class Abundance Bar Chart")
                        with expand_abundance_bar_chart:
                         st.markdown("""
                                     The total abundance of a class is computed by summing the abundances of all the lipid species belonging 
                                     to that class.  
                                     """)
                                     
                         selected_conditions_list = st.multiselect('Add or remove conditions  ', experiment.conditions_list, experiment.conditions_list)
                         selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), \
                                                                list(continuation_df['ClassKey'].value_counts().index))
                         mode = st.radio('Select a mode', ('linear scale', 'log2 scale'), 1)
                         abundance_chart_object = lp.AbundanceBarChart(experiment, continuation_df.copy(deep=True))
                         abundance_chart_object.create_abundance_bar_chart(selected_conditions_list, selected_classes_list, mode)
                         
                        expand_pathway_plot = st.expander("Lipid Pathway Visualization")
                        with expand_pathway_plot:
                            if len([x for x in experiment.number_of_samples_list if x>1]) > 1:
                                control_condition = st.selectbox('Pick the control condition', \
                                                    [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 0)
                                experimental_condition = st.selectbox('Pick the experimental condition ', \
                                                    [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 1)
                                
                                pathway_viz_object = lp.PathwayViz(experiment, continuation_df.copy(deep=True), control_condition, experimental_condition)
                                class_saturation_ratio_df = pathway_viz_object.calculate_class_saturation_ratio()
                                class_fold_change_df = pathway_viz_object.calculate_class_fold_change()
                                pathway_viz_object.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df)
                                    
                            else:
                                st.error('You need at least two conditions with more than one replicate to create a pathway visualization!')
                            
            if dataset_normality_status == 'No':
                
                st.subheader("2) Scan Data & Run Quality Checks")
                
                expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
                with expand_box_plot:
                    box_plot_object = lp.BoxPlot(experiment, continuation_df.copy(deep=True))
                    mean_area_df = box_plot_object.create_mean_area_df(box_plot_object.df, experiment.full_samples_list)
                    zero_values_percent_list = box_plot_object.calculate_missing_values_percentage(mean_area_df, experiment.full_samples_list)
                    box_plot_object.plot_missing_values(experiment.full_samples_list, zero_values_percent_list)
                    st.write('--------------------------------------------------------------------------------')
                    box_plot_object.plot_box_plot(mean_area_df, experiment.full_samples_list)
                
                if (bqc_label is not None): 
                    expand_filtered_data = st.expander("Check Consistency of Readngs Using BQC Samples")
                    with expand_filtered_data:
                        index = experiment.conditions_list.index(bqc_label)
                        filtered_data_object = lp.FilteredData(continuation_df.copy(deep=True), experiment, bqc_label, index)
                        prepared_df = filtered_data_object.plot_cov()
                        filter_ans = st.radio('Would you like to filter the data using BQC samples?', ['Yes', 'No'], 1)
                        if filter_ans == 'Yes':
                            thresh = st.number_input('Enter the maximum acceptable CoV in %', min_value = 10, max_value = 1000, value = 30, step = 1)
                            continuation_df = filtered_data_object.filter_df(thresh, prepared_df.copy(deep=True))
                            st.write('View and download the filtered dataset:')
                            st.write(continuation_df)
                            csv_download = convert_df(continuation_df)
                            st.download_button(
                                label="Download Data",
                                data=csv_download,
                                file_name='filtered_data.csv',
                                mime='text/csv')
                        
                expand_retention = st.expander('View Retention Time Plots: Check Sanity of Data')
                with expand_retention:
                    st.markdown("""
                        The retention time of a lipid species is a function of its degree of hydrophobicity. 
                        The more hydrophobic the lipid species, the longer the retention time. When retention time is 
                        plotted versus molecular mass, lipid species tend to form separate clusters based upon which lipid class they belong to.
                        
                        Inspect the retention time of lipid species within any lipid class and compare with other lipid classes. 
                        Does everything make sense?
                        """) 
                        
                    retention_time_object = lp.RetentionTime(continuation_df.copy(deep=True), experiment)
                    mode = st.radio('Pick a mode', ['Comparison Mode', 'Individual Mode']) 
                    if mode == 'Individual Mode':
                        retention_time_object.plot_single_retention()
                    elif mode == 'Comparison Mode':
                        retention_time_object.plot_multi_retention()
                        
                st.subheader("3) Detect & Remove Anomalies")
            
                expand_corr = st.expander('Pairwise Correlation Analysis') 
                with expand_corr:
                    st.markdown("""
                        Typically, the AUC's of any sample is highly linearly correlated to those of its biological replicate
                        (i.e. correlation coefficient > 0.8). This linear correlation is expected to be even stronger for technical replicates 
                        (i.e. correlation coefficient > 0.9).
                        A sample that has a weak correlation with its biological replicates is an outlier. That sample might be an outlier because 
                        of the natural biological variance or an error during sample preparation.
                        
                        Run a correlation test to inspect the degree of correlation between any two biological or technical replicates:
                        """)
                    st.info("LipidCruncher removes the missing values before preforming the correlation test.")
                    condition = st.selectbox('Select a condition', [condition for condition, number_of_samples in \
                                                                    zip(experiment.conditions_list, experiment.number_of_samples_list) if number_of_samples > 1])
                    sample_type = st.selectbox('Select the type of your samples',['biological replicates', 'Technical replicates'])
                    index = experiment.conditions_list.index(condition)
                    correlation_plot_object = lp.Correlation(continuation_df, experiment, condition, sample_type, index)
                    correlation_plot_object.correlation_plot()
                    
                expand_pca = st.expander('Principal Component Analysis (PCA)')
                with expand_pca:
                    st.markdown("""
                        Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the 
                        dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains 
                        most of the information in the large set.
                        
                        Typically, biological replicates are expected to cluster together with the exception of rare outliers that fall 
                        further away from the rest of the replicates. 
                        
                        A plot of the top two principal components (PC1 and PC2) against each other is the best way to inspect the clustering 
                        of different samples. This is because the PC's are ordered based on how much of the variability in the data 
                        they can explain (i.e. PC1 is the PC with the highest variance explained ratio, PC2 is the PC with the second highest 
                        variance expalined ratio and so on).
                        
                        Run PCA to inspect the clustering of different samples:
                        """)
                    st.info("LipidCruncher does NOT remove missing values before performng PCA analysis.")
                    
                    remove_ans = st.radio("Would you like to remove any samples from the analysis?", ['Yes', 'No'], 1)
                    if remove_ans == 'Yes':
                        st.warning('The samples you remove now, will be removed for the rest of the analysis.')
                        list_of_bad_samples = st.multiselect('Pick the sample(s) that you want to remove from the analysis', experiment.full_samples_list)
                        if (len(experiment.full_samples_list) - len(list_of_bad_samples)) >= 2 and len(list_of_bad_samples) > 0:
                            continuation_df = experiment.remove_bad_samples(list_of_bad_samples, continuation_df.copy(deep=True))
                        elif (len(experiment.full_samples_list) - len(list_of_bad_samples)) < 2:
                            st.error('At least two samples are required for a meanigful analysis!')
                            
                    pca_object = lp.PCA(continuation_df, experiment)
                    pca_object.plot_pca()
                    
                st.subheader("4) Analyze Data & Test Hypothesis")
                
                expand_vol_plot = st.expander("Volcano Plots - Test Hypothesis")
                with expand_vol_plot:
                    st.markdown("""
                                In statistics, a volcano plot is a type of scatter-plot that is used to quickly identify changes in \
                                large data sets composed of replicate data. It plots significance versus fold-change on the y and x axes, respectively. \
                                A volcano plot combines a measure of statistical significance from a statistical test (e.g., a p value from a T-test) \
                                with the magnitude of the change, enabling quick visual identification of those data-points that display large magnitude\
                                changes that are also statistically significant (datapoints at the top left and top right quadrant).
                                
                                Below, q-value (i.e. -log10(p-value)) of each lipid species is plotted versus the fold change of that species. \
                                The p-value is computed from a two-sample T-test and the fold change is computed from the following formula:
                                """)
                    latext = r'''
                    $$ 
                    Fold Change = log2(\frac{Mean AUC(Condition 1)}{Mean AUC(Condition 2)})
                    $$  
                    '''
                    st.write(latext)
                    
                    if len([x for x in experiment.number_of_samples_list if x>1]) > 1:
                        control_condition = st.selectbox('pick the control condition', \
                                                         [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 0)
                        experimental_condition = st.selectbox('Pick the experimental condition', \
                                                              [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 1)
                        p_value_threshold = st.number_input('Enter the significance level', min_value = 0.001, max_value= 0.1, value = 0.05, step = 0.001)
                        q_value_threshold = -np.log10(p_value_threshold)
                        
                        volcano_object = lp.VolcanoPlot(experiment, continuation_df.copy(deep=True), control_condition, experimental_condition, p_value_threshold, q_value_threshold)
                        volcano_df = volcano_object.add_fold_change_and_p_value_columns()
                        
                        selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), \
                                                               list(continuation_df['ClassKey'].value_counts().index)[: 3])
                        if len(selected_classes_list) > 30:
                            st.error('You can only compare up to 30 lipid classes at a time!')
                        else:
                            volcano_object.create_volcano_plot(volcano_df, selected_classes_list)
                        
                    else:
                        st.error('You need at least two conditions with more than one replicate to create a volcano plot!')
                        
                expand_sat_plot = st.expander("Saturation Level Plots - Investigate Saturation Profile of Different Lipid Classes")
                with expand_sat_plot:  
                    st.markdown("""
                                Saturation level plots show the saturation profile of each lipid class. 
                                First, for each lipid species, the ratio of Saturated Fatty Acids (SFA), Mono Unsaturated \
                                Fatty Acids (MUFA) and Poly Unsaturated Fatty Acids (PUFA) is calculated as following:
                                    
                                SFA ratio = total number of saturated fatty acids / total number of fatty acids
                                
                                MUFA ratio = total number of mono unsaturated fatty acids / total number of fatty acids
                                
                                PUFA ratio = total number of poly unsaturated fatty acids / total number of fatty acids
                                
                                Then, for each lipid species, the abundance of SFA, MUFA and PUFA is calculated as following:
                                    
                                AUC(SFA) = (AUC averaged over all replicates).(SFA ratio)
                                
                                AUC(MUFA) = (AUC averaged over all replicates).(MUFA ratio)
                                
                                AUC(PUFA) = (AUC averaged over all replicates).(PUFA ratio)
                                
                                Finally, total AUC(SFA), AUC(MUFA) and AUC(PUFA) for each lipid class is calculated by taking the sum of \
                                AUC(SFA), AUC(MUFA) and AUC(PUFA) over all lipid species that belong to that class. 
                                """)
                                
                    selected_conditions_list = st.multiselect('Add or remove conditions', experiment.conditions_list, experiment.conditions_list)
                    sat_plot_object = lp.SaturationPlot(experiment, continuation_df.copy(deep=True))
                    sat_plot_object.create_saturation_plot(selected_conditions_list)
                    
                expand_abundance_bar_chart = st.expander("Class Abundance Bar Chart")
                with expand_abundance_bar_chart:
                     st.markdown("""
                                 The total abundance of a class is computed by summing the abundances of all the lipid species belonging 
                                 to that class.  
                                 """)
                                 
                     selected_conditions_list = st.multiselect('Add or remove conditions ', experiment.conditions_list, experiment.conditions_list)
                     selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), \
                                                            list(continuation_df['ClassKey'].value_counts().index))
                     mode = st.radio('Select a mode', ('linear scale', 'log2 scale'), 1)
                     abundance_chart_object = lp.AbundanceBarChart(experiment, continuation_df.copy(deep=True))
                     abundance_chart_object.create_abundance_bar_chart(selected_conditions_list, selected_classes_list, mode)
                     
                expand_pathway_plot = st.expander("Lipid Pathway Visualization")
                with expand_pathway_plot:
                    if len([x for x in experiment.number_of_samples_list if x>1]) > 1:
                        control_condition = st.selectbox('Pick the control condition', \
                                            [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 0)
                        experimental_condition = st.selectbox('Pick the experimental condition ', \
                                            [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 1)
                        
                        pathway_viz_object = lp.PathwayViz(experiment, continuation_df.copy(deep=True), control_condition, experimental_condition)
                        class_saturation_ratio_df = pathway_viz_object.calculate_class_saturation_ratio()
                        class_fold_change_df = pathway_viz_object.calculate_class_fold_change()
                        pathway_viz_object.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df)
                            
                    else:
                        st.error('You need at least two conditions with more than one replicate to create a pathway visualization!')
                 
                

        else:
            
            st.warning('You do not have any internal standards. LipidCruncher cannot normalize your data!')
            
            st.subheader("2) Scan Data & Run Quality Checks")
            
            expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
            with expand_box_plot:
                box_plot_object = lp.BoxPlot(experiment, continuation_df.copy(deep=True))
                mean_area_df = box_plot_object.create_mean_area_df(box_plot_object.df, experiment.full_samples_list)
                zero_values_percent_list = box_plot_object.calculate_missing_values_percentage(mean_area_df, experiment.full_samples_list)
                box_plot_object.plot_missing_values(experiment.full_samples_list, zero_values_percent_list)
                st.write('--------------------------------------------------------------------------------')
                box_plot_object.plot_box_plot(mean_area_df, experiment.full_samples_list)
            
            if (bqc_label is not None): 
                expand_filtered_data = st.expander("Check Consistency of Readngs Using BQC Samples")
                with expand_filtered_data:
                    index = experiment.conditions_list.index(bqc_label)
                    filtered_data_object = lp.FilteredData(continuation_df.copy(deep=True), experiment, bqc_label, index)
                    prepared_df = filtered_data_object.plot_cov()
                    filter_ans = st.radio('Would you like to filter the data using BQC samples?', ['Yes', 'No'], 1)
                    if filter_ans == 'Yes':
                        thresh = st.number_input('Enter the maximum acceptable CoV in %', min_value = 10, max_value = 1000, value = 30, step = 1)
                        continuation_df = filtered_data_object.filter_df(thresh, prepared_df.copy(deep=True))
                        st.write(continuation_df)
                        csv_download = convert_df(continuation_df)
                        st.download_button(
                            label="Download Data",
                            data=csv_download,
                            file_name='filtered_data.csv',
                            mime='text/csv')
                    
            expand_retention = st.expander('View Retention Time Plots: Check Sanity of Data')
            with expand_retention:
                st.markdown("""
                    The retention time of a lipid species is a function of its degree of hydrophobicity. 
                    The more hydrophobic the lipid species, the longer the retention time. When retention time is 
                    plotted versus molecular mass, lipid species tend to form separate clusters based upon which lipid class they belong to.
                    
                    Inspect the retention time of lipid species within any lipid class and compare with other lipid classes. 
                    Does everything make sense?
                    """) 
                    
                retention_time_object = lp.RetentionTime(continuation_df.copy(deep=True), experiment)
                mode = st.radio('Pick a mode', ['Comparison Mode', 'Individual Mode']) 
                if mode == 'Individual Mode':
                    retention_time_object.plot_single_retention()
                elif mode == 'Comparison Mode':
                    retention_time_object.plot_multi_retention()
                    
            st.subheader("3) Detect & Remove Anomalies")
        
            expand_corr = st.expander('Pairwise Correlation Analysis') 
            with expand_corr:
                st.markdown("""
                    Typically, the AUC's of any sample is highly linearly correlated to those of its biological replicate
                    (i.e. correlation coefficient > 0.8). This linear correlation is expected to be even stronger for technical replicates 
                    (i.e. correlation coefficient > 0.9).
                    A sample that has a weak correlation with its biological replicates is an outlier. That sample might be an outlier because 
                    of the natural biological variance or an error during sample preparation.
                    
                    Run a correlation test to inspect the degree of correlation between any two biological or technical replicates:
                    """)
                st.info("LipidCruncher removes the missing values before preforming the correlation test.")
                condition = st.selectbox('Select a condition', [condition for condition, number_of_samples in \
                                                                zip(experiment.conditions_list, experiment.number_of_samples_list) if number_of_samples > 1])
                sample_type = st.selectbox('Select the type of your samples',['biological replicates', 'Technical replicates'])
                index = experiment.conditions_list.index(condition)
                correlation_plot_object = lp.Correlation(continuation_df, experiment, condition, sample_type, index)
                correlation_plot_object.correlation_plot()
                
            expand_pca = st.expander('Principal Component Analysis (PCA)')
            with expand_pca:
                st.markdown("""
                    Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the 
                    dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains 
                    most of the information in the large set.
                    
                    Typically, biological replicates are expected to cluster together with the exception of rare outliers that fall 
                    further away from the rest of the replicates. 
                    
                    A plot of the top two principal components (PC1 and PC2) against each other is the best way to inspect the clustering 
                    of different samples. This is because the PC's are ordered based on how much of the variability in the data 
                    they can explain (i.e. PC1 is the PC with the highest variance explained ratio, PC2 is the PC with the second highest 
                    variance expalined ratio and so on).
                    
                    Run PCA to inspect the clustering of different samples:
                    """)
                st.info("LipidCruncher does NOT remove missing values before performng PCA analysis.")
                
                remove_ans = st.radio("Would you like to remove any samples from the analysis?", ['Yes', 'No'], 1)
                if remove_ans == 'Yes':
                    st.warning('The samples you remove now, will be removed for the rest of the analysis.')
                    list_of_bad_samples = st.multiselect('Pick the sample(s) that you want to remove from the analysis', experiment.full_samples_list)
                    if (len(experiment.full_samples_list) - len(list_of_bad_samples)) >= 2 and len(list_of_bad_samples) > 0:
                        continuation_df = experiment.remove_bad_samples(list_of_bad_samples, continuation_df.copy(deep=True))
                    elif (len(experiment.full_samples_list) - len(list_of_bad_samples)) < 2:
                        st.error('At least two samples are required for a meanigful analysis!')
                        
                pca_object = lp.PCA(continuation_df, experiment)
                pca_object.plot_pca()
                
            st.subheader("4) Analyze Data & Test Hypothesis")
            st.markdown("""
                        The Data Analysis & Hypothesis Testing submodule allows the user to run statistical analysis on the data and test their hypothesis. 
                        """)
            
            expand_vol_plot = st.expander("Volcano Plots - Test Hypothesis")
            with expand_vol_plot:
                st.markdown("""
                            In statistics, a volcano plot is a type of scatter-plot that is used to quickly identify changes in \
                            large data sets composed of replicate data. It plots significance versus fold-change on the y and x axes, respectively. \
                            A volcano plot combines a measure of statistical significance from a statistical test (e.g., a p value from a T-test) \
                            with the magnitude of the change, enabling quick visual identification of those data-points that display large magnitude\
                            changes that are also statistically significant (datapoints at the top left and top right quadrant).
                            
                            Below, q-value (i.e. -log10(p-value)) of each lipid species is plotted versus the fold change of that species. \
                            The p-value is computed from a two-sample T-test and the fold change is computed from the following formula:
                            """)
                latext = r'''
                $$ 
                Fold Change = log2(\frac{Mean AUC(Condition 1)}{Mean AUC(Condition 2)})
                $$  
                '''
                st.write(latext)
                
                if len([x for x in experiment.number_of_samples_list if x>1]) > 1:
                    control_condition = st.selectbox('pick the control condition', \
                                                     [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 0)
                    experimental_condition = st.selectbox('Pick the experimental condition', \
                                                          [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 1)
                    p_value_threshold = st.number_input('Enter the significance level', min_value = 0.001, max_value= 0.1, value = 0.05, step = 0.001)
                    q_value_threshold = -np.log10(p_value_threshold)
                    
                    volcano_object = lp.VolcanoPlot(experiment, continuation_df.copy(deep=True), control_condition, experimental_condition, p_value_threshold, q_value_threshold)
                    volcano_df = volcano_object.add_fold_change_and_p_value_columns()
                    
                    selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), \
                                                           list(continuation_df['ClassKey'].value_counts().index)[: 3])
                    if len(selected_classes_list) > 30:
                        st.error('You can only compare up to 30 lipid classes at a time!')
                    else:
                        volcano_object.create_volcano_plot(volcano_df, selected_classes_list)
                    
                else:
                    st.error('You need at least two conditions with more than one replicate to create a volcano plot!')
                    
            expand_sat_plot = st.expander("Saturation Level Plots - Investigate Saturation Profile of Different Lipid Classes")
            with expand_sat_plot:    
                 st.markdown("""
                             Saturation level plots show the saturation profile of each lipid class. 
                             First, for each lipid species, the ratio of Saturated Fatty Acids (SFA), Mono Unsaturated \
                             Fatty Acids (MUFA) and Poly Unsaturated Fatty Acids (PUFA) is calculated as following:
                                 
                             SFA ratio = total number of saturated fatty acids / total number of fatty acids
                             
                             MUFA ratio = total number of mono unsaturated fatty acids / total number of fatty acids
                             
                             PUFA ratio = total number of poly unsaturated fatty acids / total number of fatty acids
                             
                             Then, for each lipid species, the abundance of SFA, MUFA and PUFA is calculated as following:
                                 
                             AUC(SFA) = (AUC averaged over all replicates).(SFA ratio)
                             
                             AUC(MUFA) = (AUC averaged over all replicates).(MUFA ratio)
                             
                             AUC(PUFA) = (AUC averaged over all replicates).(PUFA ratio)
                             
                             Finally, total AUC(SFA), AUC(MUFA) and AUC(PUFA) for each lipid class is calculated by taking the sum of \
                             AUC(SFA), AUC(MUFA) and AUC(PUFA) over all lipid species that belong to that class. 
                             """)
                             
                 selected_conditions_list = st.multiselect('Add or remove conditions', experiment.conditions_list, experiment.conditions_list)
                 sat_plot_object = lp.SaturationPlot(experiment, continuation_df.copy(deep=True))
                 sat_plot_object.create_saturation_plot(selected_conditions_list)
                 
            expand_abundance_bar_chart = st.expander("Class Abundance Bar Chart")
            with expand_abundance_bar_chart:
             st.markdown("""
                         The total abundance of a class is computed by summing the abundances of all the lipid species belonging 
                         to that class.  
                         """)
                         
             selected_conditions_list = st.multiselect('Add or remove conditions', experiment.conditions_list, experiment.conditions_list)
             selected_classes_list = st.multiselect('Add or remove classes: ', list(continuation_df['ClassKey'].value_counts().index), \
                                                    list(continuation_df['ClassKey'].value_counts().index))
             mode = st.radio('Select a mode', ('linear scale', 'log2 scale'), 1)
             abundance_chart_object = lp.AbundanceBarChart(experiment, continuation_df.copy(deep=True))
             abundance_chart_object.create_abundance_bar_chart(selected_conditions_list, selected_classes_list, mode)
             
            expand_pathway_plot = st.expander("Lipid Pathway Visualization")
            with expand_pathway_plot:
                if len([x for x in experiment.number_of_samples_list if x>1]) > 1:
                    control_condition = st.selectbox('Pick the control condition', \
                                        [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 0)
                    experimental_condition = st.selectbox('Pick the experimental condition ', \
                                        [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1], 1)
                    
                    pathway_viz_object = lp.PathwayViz(experiment, continuation_df.copy(deep=True), control_condition, experimental_condition)
                    class_saturation_ratio_df = pathway_viz_object.calculate_class_saturation_ratio()
                    class_fold_change_df = pathway_viz_object.calculate_class_fold_change()
                    pathway_viz_object.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df)
                        
                else:
                    st.error('You need at least two conditions with more than one replicate to create a pathway visualization!')
                
