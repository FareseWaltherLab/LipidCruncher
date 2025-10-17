"""
Test page for refactored data cleaning and normalization services.
This is a temporary page to test the new architecture before full integration.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lipidcruncher.adapters.streamlit_adapter import StreamlitDataAdapter
from lipidcruncher.core.models.experiment import ExperimentConfig
from lipidcruncher.core.models.normalization import NormalizationConfig

st.set_page_config(page_title="Test Refactored Services", page_icon="🧪")

st.title("🧪 Test Refactored Services")
st.markdown("---")

st.info("""
This is a **test page** for the refactored data cleaning and normalization services.
Use this to verify the new architecture works with your real data before full integration.
""")

# Initialize adapter
if 'adapter' not in st.session_state:
    st.session_state.adapter = StreamlitDataAdapter()

adapter = st.session_state.adapter

# Step 1: Upload Data
st.header("1️⃣ Upload Data")

data_format = st.selectbox(
    "Select data format:",
    ['LipidSearch 5.0', 'Generic Format']
)

uploaded_file = st.file_uploader(
    "Upload your lipidomics data",
    type=['csv', 'txt', 'tsv'],
    help="Upload CSV, TXT, or TSV file"
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.write(f"Shape: {df.shape}")
        
        with st.expander("Preview uploaded data"):
            st.dataframe(df.head())
        
        # Step 2: Experiment Setup
        st.header("2️⃣ Experiment Setup")
        
        n_conditions = st.number_input("Number of conditions:", min_value=1, max_value=10, value=2)
        
        conditions = []
        samples_per_condition = []
        
        for i in range(n_conditions):
            col1, col2 = st.columns(2)
            with col1:
                cond_name = st.text_input(f"Condition {i+1} name:", value=f"Condition{i+1}", key=f"cond_{i}")
                conditions.append(cond_name)
            with col2:
                n_samples = st.number_input(f"Samples in {cond_name}:", min_value=1, value=3, key=f"samples_{i}")
                samples_per_condition.append(n_samples)
        
        if st.button("Create Experiment Config"):
            experiment_config = ExperimentConfig(
                n_conditions=n_conditions,
                conditions_list=conditions,
                number_of_samples_list=samples_per_condition
            )
            st.session_state.experiment_config = experiment_config
            st.success(f"✅ Experiment configured: {experiment_config.samples_list}")
        
        # Step 3: Data Cleaning
        if 'experiment_config' in st.session_state:
            st.header("3️⃣ Data Cleaning")
            
            grade_config = None
            if data_format == 'LipidSearch 5.0':
                with st.expander("Grade Filtering (Optional)"):
                    st.info("Leave empty to use defaults (A/B/C for all)")
                    # Simplified - you can expand this later
            
            if st.button("Clean Data"):
                with st.spinner("Cleaning data..."):
                    try:
                        cleaned_df, standards_df = adapter.clean_data(
                            df,
                            st.session_state.experiment_config,
                            data_format,
                            grade_config
                        )
                        
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.standards_df = standards_df
                        
                        st.write("### Cleaned Data")
                        st.dataframe(cleaned_df)
                        
                        st.write("### Internal Standards")
                        st.dataframe(standards_df)
                        
                    except Exception as e:
                        st.error(f"Error during cleaning: {str(e)}")
                        st.exception(e)
        
        # Step 4: Normalization
        if 'cleaned_df' in st.session_state:
            st.header("4️⃣ Normalization")
            
            norm_method = st.selectbox(
                "Normalization method:",
                ['none', 'internal_standard', 'protein', 'both']
            )
            
            if norm_method != 'none':
                # Get available classes
                available_classes = sorted(st.session_state.cleaned_df['ClassKey'].unique())
                selected_classes = st.multiselect(
                    "Select lipid classes to normalize:",
                    available_classes,
                    default=available_classes[:3] if len(available_classes) >= 3 else available_classes
                )
                
                internal_standards = None
                intsta_concentrations = None
                protein_concentrations = None
                
                if norm_method in ['internal_standard', 'both']:
                    st.subheader("Internal Standards Setup")
                    
                    if st.session_state.standards_df.empty:
                        st.warning("No internal standards found in data!")
                    else:
                        available_standards = st.session_state.standards_df['LipidMolec'].tolist()
                        
                        internal_standards = {}
                        intsta_concentrations = {}
                        
                        for lipid_class in selected_classes:
                            col1, col2 = st.columns(2)
                            with col1:
                                standard = st.selectbox(
                                    f"Standard for {lipid_class}:",
                                    available_standards,
                                    key=f"std_{lipid_class}"
                                )
                                internal_standards[lipid_class] = standard
                            with col2:
                                conc = st.number_input(
                                    f"Concentration:",
                                    value=1.0,
                                    min_value=0.0,
                                    key=f"conc_{lipid_class}"
                                )
                                intsta_concentrations[standard] = conc
                
                if norm_method in ['protein', 'both']:
                    st.subheader("Protein Concentrations")
                    
                    protein_concentrations = {}
                    for sample in st.session_state.experiment_config.samples_list:
                        protein_concentrations[sample] = st.number_input(
                            f"Protein concentration for {sample}:",
                            value=2.0,
                            min_value=0.01,
                            key=f"protein_{sample}"
                        )
                    
                    # Create protein DataFrame
                    protein_df = pd.DataFrame({
                        'Sample': list(protein_concentrations.keys()),
                        'Concentration': list(protein_concentrations.values())
                    })
                    st.session_state.protein_df = protein_df
                
                if st.button("Normalize Data"):
                    with st.spinner("Normalizing..."):
                        try:
                            norm_config = NormalizationConfig(
                                method=norm_method,
                                selected_classes=selected_classes,
                                internal_standards=internal_standards,
                                intsta_concentrations=intsta_concentrations,
                                protein_concentrations=protein_concentrations
                            )
                            
                            normalized_df = adapter.normalize_data(
                                st.session_state.cleaned_df,
                                norm_config,
                                st.session_state.experiment_config,
                                intsta_df=st.session_state.standards_df if not st.session_state.standards_df.empty else None,
                                protein_df=st.session_state.get('protein_df')
                            )
                            
                            st.session_state.normalized_df = normalized_df
                            
                            st.write("### Normalized Data")
                            st.dataframe(normalized_df)
                            
                            # Download button
                            csv = normalized_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Normalized Data",
                                csv,
                                "normalized_data.csv",
                                "text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error during normalization: {str(e)}")
                            st.exception(e)
            else:
                st.info("No normalization selected. Data will be used as-is.")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Upload a file to begin testing")
