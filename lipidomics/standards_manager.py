import streamlit as st
import pandas as pd

class StandardsManager:
    """
    Manages addition and removal of internal standards in lipidomics data.
    """
    
    @staticmethod
    def add_standard(cleaned_df, intsta_df, class_key, lipid_molec):
        """
        Adds a new internal standard for a lipid class.
        """
        try:
            st.write("\nDEBUG - StandardsManager.add_standard inputs:")
            st.write(f"- class_key: {class_key}")
            st.write(f"- lipid_molec: {lipid_molec}")
            st.write(f"- cleaned_df shape: {cleaned_df.shape}")
            st.write(f"- intsta_df shape: {intsta_df.shape if not intsta_df.empty else '(empty)'}")
            
            # Create copies of the DataFrames to avoid modifying the original
            cleaned_df = cleaned_df.copy()
            intsta_df = intsta_df.copy() if not intsta_df.empty else pd.DataFrame(columns=cleaned_df.columns)
            
            # Check if class already has a standard
            if not intsta_df.empty and class_key in intsta_df['ClassKey'].values:
                st.write(f"DEBUG - Class {class_key} already has a standard!")
                return cleaned_df, intsta_df
            
            # Get the row to move
            standard_row = cleaned_df[cleaned_df['LipidMolec'] == lipid_molec].copy()
            st.write("\nDEBUG - Standard row:")
            st.write(f"- Found matching row: {not standard_row.empty}")
            if not standard_row.empty:
                st.write("- Row content:")
                st.write(standard_row)
            
            if standard_row.empty:
                st.write("DEBUG - No matching row found for standard!")
                return cleaned_df, intsta_df
            
            # Add (s) marker to the lipid molecule if it doesn't already have one
            if not standard_row['LipidMolec'].iloc[0].endswith(':(s)'):
                standard_row['LipidMolec'] = standard_row['LipidMolec'] + ":(s)"
                st.write("\nDEBUG - Added (s) marker to standard:")
                st.write(f"- Modified LipidMolec: {standard_row['LipidMolec'].iloc[0]}")
            
            # Remove from cleaned_df and add to intsta_df
            st.write("\nDEBUG - Before DataFrame updates:")
            st.write(f"- cleaned_df shape: {cleaned_df.shape}")
            st.write(f"- intsta_df shape: {intsta_df.shape}")
            st.write(f"- Current standards: {intsta_df['LipidMolec'].tolist()}")
            
            cleaned_df = cleaned_df[cleaned_df['LipidMolec'] != lipid_molec]
            intsta_df = pd.concat([intsta_df, standard_row], ignore_index=True)
            
            # Sort both DataFrames
            cleaned_df = cleaned_df.sort_values(['ClassKey', 'LipidMolec']).reset_index(drop=True)
            intsta_df = intsta_df.sort_values('ClassKey').reset_index(drop=True)
            
            st.write("\nDEBUG - After DataFrame updates:")
            st.write(f"- cleaned_df shape: {cleaned_df.shape}")
            st.write(f"- intsta_df shape: {intsta_df.shape}")
            st.write(f"- Updated standards list: {intsta_df['LipidMolec'].tolist()}")
            
            return cleaned_df, intsta_df
            
        except Exception as e:
            st.write(f"Error adding standard: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
            return cleaned_df, intsta_df

    @staticmethod
    def remove_standards(cleaned_df, intsta_df, standard_molecs):
        """
        Removes multiple internal standards and returns them to the main dataset.
        """
        try:
            st.write("\nDEBUG - remove_standards inputs:")
            st.write(f"- cleaned_df shape: {cleaned_df.shape}")
            st.write(f"- intsta_df shape: {intsta_df.shape}")
            st.write(f"- standards to remove: {standard_molecs}")
            
            # Create copies of the DataFrames
            cleaned_df = cleaned_df.copy()
            intsta_df = intsta_df.copy()
            
            # Convert single standard to list if necessary
            if isinstance(standard_molecs, str):
                standard_molecs = [standard_molecs]
                
            st.write("\nDEBUG - Current standards before removal:")
            st.write(intsta_df['LipidMolec'].tolist())
            
            # Get all rows to move back
            standards_mask = intsta_df['LipidMolec'].isin(standard_molecs)
            standards_to_move = intsta_df[standards_mask].copy()
            
            st.write(f"\nDEBUG - Found {len(standards_to_move)} standards to move")
            
            if not standards_to_move.empty:
                # Remove (s) marker from the lipid molecules
                standards_to_move['LipidMolec'] = standards_to_move['LipidMolec'].str.replace(':(s)', '')
                
                # Remove from intsta_df and add to cleaned_df
                intsta_df = intsta_df[~standards_mask].reset_index(drop=True)
                cleaned_df = pd.concat([cleaned_df, standards_to_move], ignore_index=True)
                
                # Sort both DataFrames
                cleaned_df = cleaned_df.sort_values(['ClassKey', 'LipidMolec']).reset_index(drop=True)
                intsta_df = intsta_df.sort_values('ClassKey').reset_index(drop=True)
                
                st.write("\nDEBUG - After removal:")
                st.write(f"- new cleaned_df shape: {cleaned_df.shape}")
                st.write(f"- new intsta_df shape: {intsta_df.shape}")
                st.write("- remaining standards:", intsta_df['LipidMolec'].tolist())
            
            return cleaned_df, intsta_df
            
        except Exception as e:
            st.write(f"Error removing standards: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
            return cleaned_df, intsta_df

    @staticmethod
    def get_current_standards(intsta_df):
        """
        Get a mapping of current standards by class.
        
        Args:
            intsta_df (pd.DataFrame): Internal standards dataset
            
        Returns:
            dict: Mapping of ClassKey to LipidMolec
        """
        if intsta_df.empty:
            return {}
            
        return dict(zip(intsta_df['ClassKey'], intsta_df['LipidMolec']))