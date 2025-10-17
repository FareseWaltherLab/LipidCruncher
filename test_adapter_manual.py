"""
Manual test script for the adapter.
Run this to verify the adapter works before modifying main_app.py
"""
import pandas as pd
from src.lipidcruncher.adapters.streamlit_adapter import StreamlitDataAdapter
from src.lipidcruncher.core.models.experiment import ExperimentConfig

# Create sample data
data = pd.DataFrame({
    'LipidMolec': ['PC 16:0_18:1', 'PC 16:0_18:1+D7', 'PE 18:0_20:4'],
    'ClassKey': ['PC', 'PC', 'PE'],
    'FAKey': ['16:0_18:1', '16:0_18:1+D7', '18:0_20:4'],
    'TotalGrade': ['A', 'A', 'B'],
    'TotalSmpIDRate(%)': [95.0, 98.0, 90.0],
    'CalcMass': [760.5, 767.5, 768.5],
    'BaseRt': [12.5, 12.3, 13.2],
    'intensity[s1]': [1000.0, 100.0, 1200.0],
    'intensity[s2]': [1100.0, 110.0, 1300.0]
})

# Create experiment config
experiment = ExperimentConfig(
    n_conditions=1,
    conditions_list=['Control'],
    number_of_samples_list=[2]
)

# Test the adapter
adapter = StreamlitDataAdapter()

print("Testing data cleaning...")
cleaned_df, standards_df = adapter.clean_data(
    data,
    experiment,
    'LipidSearch 5.0'
)

print(f"\nCleaned data: {len(cleaned_df)} lipids")
print(cleaned_df)
print(f"\nStandards: {len(standards_df)} standards")
print(standards_df)

print("\n✅ Adapter test successful!")
