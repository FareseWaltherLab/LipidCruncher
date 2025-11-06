"""
Sample Grouping Service

Handles sample-to-condition mapping and DataFrame reordering.
Pure business logic - no UI dependencies.
"""

from typing import Dict, List, Tuple
import pandas as pd


class SampleGroupingService:
    """
    Service for handling sample-to-condition mapping and reordering.
    
    This service provides functionality to:
    - Build initial sample-to-condition mappings
    - Validate user selections for manual grouping
    - Reorder DataFrame columns based on new sample order
    - Create name mappings showing old->new sample names
    
    All methods are pure business logic with no Streamlit dependencies.
    """
    
    def build_group_dataframe(
        self,
        sample_list: List[str],
        condition_list: List[str]
    ) -> pd.DataFrame:
        """
        Build initial sample-to-condition mapping DataFrame.
        
        Creates a DataFrame showing which samples map to which conditions
        based on the experiment configuration.
        
        Args:
            sample_list: List of sample names (e.g., ['s1', 's2', 's3', ...])
            condition_list: List of conditions matching samples 
                          (e.g., ['WT', 'WT', 'KO', 'KO', ...])
        
        Returns:
            DataFrame with columns: ['sample_name', 'condition']
        
        Example:
            >>> service = SampleGroupingService()
            >>> samples = ['s1', 's2', 's3', 's4']
            >>> conditions = ['WT', 'WT', 'KO', 'KO']
            >>> df = service.build_group_dataframe(samples, conditions)
            >>> print(df)
               sample_name condition
            0           s1        WT
            1           s2        WT
            2           s3        KO
            3           s4        KO
        """
        if len(sample_list) != len(condition_list):
            raise ValueError(
                f"Sample list length ({len(sample_list)}) must match "
                f"condition list length ({len(condition_list)})"
            )
        
        return pd.DataFrame({
            'sample_name': sample_list,
            'condition': condition_list
        })
    
    def validate_selections(
        self,
        selections: Dict[str, List[str]],
        conditions_list: List[str],
        number_of_samples_list: List[int]
    ) -> Tuple[bool, str]:
        """
        Validate that user selections have correct number of samples per condition.
        
        Checks:
        1. Each condition has the correct number of samples
        2. No duplicate samples across conditions
        3. All expected conditions are present
        
        Args:
            selections: Dict mapping condition name to list of selected sample names
                       e.g., {'WT': ['s1', 's2'], 'KO': ['s3', 's4']}
            conditions_list: List of condition names
            number_of_samples_list: List of expected sample counts per condition
        
        Returns:
            Tuple of (is_valid, error_message)
            - If valid: (True, "")
            - If invalid: (False, "descriptive error message")
        
        Example:
            >>> service = SampleGroupingService()
            >>> selections = {'WT': ['s1', 's2'], 'KO': ['s3', 's4']}
            >>> valid, msg = service.validate_selections(
            ...     selections, ['WT', 'KO'], [2, 2]
            ... )
            >>> print(valid)
            True
        """
        # Check all conditions are present
        if set(selections.keys()) != set(conditions_list):
            missing = set(conditions_list) - set(selections.keys())
            extra = set(selections.keys()) - set(conditions_list)
            msg = []
            if missing:
                msg.append(f"Missing conditions: {', '.join(missing)}")
            if extra:
                msg.append(f"Extra conditions: {', '.join(extra)}")
            return False, "; ".join(msg)
        
        # Check each condition has correct number of samples
        for condition, expected_count in zip(conditions_list, number_of_samples_list):
            actual_count = len(selections.get(condition, []))
            if actual_count != expected_count:
                return False, (
                    f"Condition '{condition}' needs {expected_count} samples, "
                    f"but got {actual_count}"
                )
        
        # Check no duplicate samples across conditions
        all_selected = []
        for samples in selections.values():
            all_selected.extend(samples)
        
        if len(all_selected) != len(set(all_selected)):
            duplicates = [s for s in all_selected if all_selected.count(s) > 1]
            return False, f"Duplicate samples found: {', '.join(set(duplicates))}"
        
        return True, ""
    
    def reorder_dataframe(
        self,
        df: pd.DataFrame,
        selections: Dict[str, List[str]],
        conditions_list: List[str]
    ) -> pd.DataFrame:
        """
        Reorder DataFrame columns based on manual sample selections.
        
        Creates new column order where samples are grouped by condition
        and renamed to standardized names (intensity[s1], intensity[s2], etc.)
        
        Args:
            df: DataFrame with intensity columns to reorder
            selections: Dict mapping condition to list of sample names
                       e.g., {'WT': ['s3', 's1'], 'KO': ['s2', 's4']}
            conditions_list: Ordered list of conditions (determines final order)
        
        Returns:
            Reordered DataFrame with standardized column names
        
        Raises:
            ValueError: If a referenced sample column doesn't exist in DataFrame
        
        Example:
            >>> # Original DF has: intensity[s1], intensity[s2], intensity[s3], intensity[s4]
            >>> # User wants: s3, s1 for WT; s2, s4 for KO
            >>> selections = {'WT': ['s3', 's1'], 'KO': ['s2', 's4']}
            >>> conditions = ['WT', 'KO']
            >>> reordered_df = service.reorder_dataframe(df, selections, conditions)
            >>> # Result has: intensity[s1] (was s3), intensity[s2] (was s1), 
            >>> #              intensity[s3] (was s2), intensity[s4] (was s4)
        """
        # Create ordered sample list following condition order
        ordered_samples = []
        for condition in conditions_list:
            if condition not in selections:
                raise ValueError(f"Condition '{condition}' not found in selections")
            ordered_samples.extend(selections[condition])
        
        # Separate static and intensity columns
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        static_cols = [col for col in df.columns if col not in intensity_cols]
        
        # Build reordered DataFrame - start with static columns
        reordered_df = df[static_cols].copy()
        
        # Add intensity columns in new order with standardized names
        for i, old_sample in enumerate(ordered_samples):
            old_col = f'intensity[{old_sample}]'
            new_col = f'intensity[s{i+1}]'
            
            if old_col not in df.columns:
                raise ValueError(
                    f"Sample column '{old_col}' not found in DataFrame. "
                    f"Available intensity columns: {intensity_cols}"
                )
            
            reordered_df[new_col] = df[old_col].values
        
        return reordered_df
    
    def create_name_mapping(
        self,
        selections: Dict[str, List[str]],
        conditions_list: List[str]
    ) -> pd.DataFrame:
        """
        Create mapping DataFrame showing old sample names -> new names -> conditions.
        
        Useful for displaying to users what happened during reordering.
        
        Args:
            selections: Dict mapping condition to list of sample names
            conditions_list: Ordered list of conditions
        
        Returns:
            DataFrame with columns: ['old_name', 'new_name', 'condition']
        
        Example:
            >>> selections = {'WT': ['s3', 's1'], 'KO': ['s2', 's4']}
            >>> conditions = ['WT', 'KO']
            >>> mapping = service.create_name_mapping(selections, conditions)
            >>> print(mapping)
              old_name new_name condition
            0       s3       s1        WT
            1       s1       s2        WT
            2       s2       s3        KO
            3       s4       s4        KO
        """
        ordered_samples = []
        ordered_conditions = []
        
        for condition in conditions_list:
            samples = selections[condition]
            ordered_samples.extend(samples)
            ordered_conditions.extend([condition] * len(samples))
        
        return pd.DataFrame({
            'old_name': ordered_samples,
            'new_name': [f's{i+1}' for i in range(len(ordered_samples))],
            'condition': ordered_conditions
        })
    
    def extract_sample_names_from_dataframe(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        """
        Extract sample names from intensity columns in DataFrame.
        
        Args:
            df: DataFrame with intensity[...] columns
        
        Returns:
            List of sample names extracted from column names
        
        Example:
            >>> df = pd.DataFrame({
            ...     'LipidMolec': ['PC(16:0_18:1)'],
            ...     'intensity[s1]': [100],
            ...     'intensity[s2]': [200]
            ... })
            >>> samples = service.extract_sample_names_from_dataframe(df)
            >>> print(samples)
            ['s1', 's2']
        """
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        sample_names = [
            col.replace('intensity[', '').replace(']', '')
            for col in intensity_cols
        ]
        return sample_names