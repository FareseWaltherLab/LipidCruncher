"""
Service for processing internal standards.
Handles standards file upload and validation.
"""
import pandas as pd
from typing import Optional


class StandardsService:
    """Service for managing internal standards."""
    
    @staticmethod
    def process_standards_file(
        standards_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        existing_standards: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Process uploaded standards file and validate against cleaned data.
        
        Args:
            standards_df: DataFrame from uploaded standards file
            cleaned_df: Cleaned dataset
            existing_standards: Existing internal standards DataFrame
            
        Returns:
            Validated standards DataFrame or None if validation fails
            
        Raises:
            ValueError: If standards file is invalid
        """
        # Validate required column
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must contain 'LipidMolec' column")
        
        # FIX: Remove duplicates from uploaded standards file
        original_count = len(standards_df)
        standards_df = standards_df.drop_duplicates(subset=['LipidMolec'], keep='first')
        duplicates_removed = original_count - len(standards_df)
        
        if duplicates_removed > 0:
            import streamlit as st
            st.warning(f"⚠️ Removed {duplicates_removed} duplicate standard(s) from uploaded file. "
                      f"Only the first occurrence of each standard was kept.")
        
        # Get list of lipids in cleaned data
        if cleaned_df is None or 'LipidMolec' not in cleaned_df.columns:
            raise ValueError("Cleaned data must contain 'LipidMolec' column")
        
        available_lipids = set(cleaned_df['LipidMolec'].unique())
        
        # Validate that standards exist in dataset
        uploaded_standards = set(standards_df['LipidMolec'].unique())
        missing_standards = uploaded_standards - available_lipids
        
        if missing_standards:
            raise ValueError(
                f"The following standards are not found in your dataset: "
                f"{', '.join(list(missing_standards)[:5])}"
                + (f" and {len(missing_standards) - 5} more..." if len(missing_standards) > 5 else "")
            )
        
        # Create new standards DataFrame with required columns
        new_standards = pd.DataFrame()
        new_standards['LipidMolec'] = standards_df['LipidMolec']
        
        # Add ClassKey if available from cleaned data
        if 'ClassKey' in cleaned_df.columns:
            # Map standards to their classes
            lipid_to_class = dict(zip(cleaned_df['LipidMolec'], cleaned_df['ClassKey']))
            new_standards['ClassKey'] = new_standards['LipidMolec'].map(lipid_to_class)
        
        # Add intensity columns from cleaned data (handle both MeanArea and intensity[...] formats)
        intensity_cols = [col for col in cleaned_df.columns if 
                         col.startswith('MeanArea') or col.startswith('intensity[')]
        
        if not intensity_cols:
            raise ValueError("Cleaned data does not contain intensity columns (MeanArea or intensity[...])")
        
        for col in intensity_cols:
            # Map standards to their intensities
            lipid_to_intensity = dict(zip(cleaned_df['LipidMolec'], cleaned_df[col]))
            new_standards[col] = new_standards['LipidMolec'].map(lipid_to_intensity)
        
        # Validate that we have data
        if new_standards.empty:
            raise ValueError("No valid standards found in the dataset")
        
        # Ensure we have intensity data
        intensity_cols_in_standards = [col for col in new_standards.columns if 
                                       col.startswith('MeanArea') or col.startswith('intensity[')]
        if not intensity_cols_in_standards:
            raise ValueError("Internal standards data does not contain properly formatted intensity columns")
        
        return new_standards