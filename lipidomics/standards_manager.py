import pandas as pd

class StandardsManager:
   """
   Manages addition and removal of internal standards in lipidomics data.
   """
   
   @staticmethod
   def add_standard(cleaned_df, intsta_df, class_key, lipid_molec):
       """
       Adds a new internal standard for a lipid class.
       
       Args:
           cleaned_df (pd.DataFrame): Main dataset
           intsta_df (pd.DataFrame): Internal standards dataset
           class_key (str): Lipid class to add standard for
           lipid_molec (str): Molecule to use as standard
           
       Returns:
           tuple: (cleaned_df, intsta_df) with updated data
       """
       try:
           # Create copies of the DataFrames to avoid modifying the original
           cleaned_df = cleaned_df.copy()
           intsta_df = intsta_df.copy() if not intsta_df.empty else pd.DataFrame(columns=cleaned_df.columns)
           
           # Check if class already has a standard
           if not intsta_df.empty and class_key in intsta_df['ClassKey'].values:
               return cleaned_df, intsta_df
           
           # Get the row to move
           standard_row = cleaned_df[cleaned_df['LipidMolec'] == lipid_molec].copy()
           
           if standard_row.empty:
               return cleaned_df, intsta_df
           
           # Add (s) marker to the lipid molecule if it doesn't already have one
           if not standard_row['LipidMolec'].iloc[0].endswith(':(s)'):
               standard_row['LipidMolec'] = standard_row['LipidMolec'] + ":(s)"
           
           # Remove from cleaned_df and add to intsta_df
           cleaned_df = cleaned_df[cleaned_df['LipidMolec'] != lipid_molec]
           intsta_df = pd.concat([intsta_df, standard_row], ignore_index=True)
           
           # Sort both DataFrames
           cleaned_df = cleaned_df.sort_values(['ClassKey', 'LipidMolec']).reset_index(drop=True)
           intsta_df = intsta_df.sort_values('ClassKey').reset_index(drop=True)
           
           return cleaned_df, intsta_df
           
       except Exception as e:
           st.error(f"Error adding standard: {str(e)}")
           return cleaned_df, intsta_df

   @staticmethod
   def remove_standards(cleaned_df, intsta_df, standard_molecs):
       """
       Removes multiple internal standards and returns them to the main dataset.
       
       Args:
           cleaned_df (pd.DataFrame): Main dataset
           intsta_df (pd.DataFrame): Internal standards dataset
           standard_molecs (str or list): Standard(s) to remove
           
       Returns:
           tuple: (cleaned_df, intsta_df) with updated data
       """
       try:
           # Create copies of the DataFrames
           cleaned_df = cleaned_df.copy()
           intsta_df = intsta_df.copy()
           
           # Convert single standard to list if necessary
           if isinstance(standard_molecs, str):
               standard_molecs = [standard_molecs]
           
           # Get all rows to move back
           standards_mask = intsta_df['LipidMolec'].isin(standard_molecs)
           standards_to_move = intsta_df[standards_mask].copy()
           
           if not standards_to_move.empty:
               # Remove (s) marker from the lipid molecules 
               standards_to_move['LipidMolec'] = standards_to_move['LipidMolec'].str.replace(':(s)', '')
               
               # Remove from intsta_df and add to cleaned_df
               intsta_df = intsta_df[~standards_mask].reset_index(drop=True)
               cleaned_df = pd.concat([cleaned_df, standards_to_move], ignore_index=True)
               
               # Sort both DataFrames
               cleaned_df = cleaned_df.sort_values(['ClassKey', 'LipidMolec']).reset_index(drop=True)
               intsta_df = intsta_df.sort_values('ClassKey').reset_index(drop=True)
           
           return cleaned_df, intsta_df
           
       except Exception as e:
           st.error(f"Error removing standards: {str(e)}")
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