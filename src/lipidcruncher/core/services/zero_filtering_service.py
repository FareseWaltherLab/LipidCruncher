"""
Service for filtering lipid species based on zero/low intensity values.
"""
from typing import List, Optional
import pandas as pd
from ..models.experiment import ExperimentConfig


class ZeroFilteringService:
    """
    Filters lipid species based on zero or low intensity values across conditions.
    
    Filtering logic:
    - If BQC condition exists: Remove if ≥50% of BQC replicates are zeros
    - For non-BQC conditions: Remove if ALL conditions have ≥75% zeros
    """
    
    def filter_by_zeros(
        self,
        df: pd.DataFrame,
        experiment_config: ExperimentConfig,
        threshold: float = 0.0,
        bqc_label: Optional[str] = None
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Filter lipid species based on zero/low intensity values.
        
        Args:
            df: DataFrame with intensity columns
            experiment_config: Experiment configuration
            threshold: Intensity threshold (values <= threshold are considered zeros)
            bqc_label: Label for BQC condition (if present)
        
        Returns:
            Tuple of (filtered_df, removed_species_list)
        """
        if df.empty:
            return df, []
        
        # Get all lipid species before filtering
        all_species = df['LipidMolec'].tolist()
        
        # Check if BQC label is valid
        valid_bqc = bqc_label is not None and bqc_label in experiment_config.conditions_list
        
        # List to keep track of rows to keep
        to_keep = []
        
        for idx, row in df.iterrows():
            should_keep = self._evaluate_lipid_for_filtering(
                row,
                experiment_config,
                threshold,
                bqc_label if valid_bqc else None
            )
            
            if should_keep:
                to_keep.append(idx)
        
        # Filter the dataframe
        filtered_df = df.loc[to_keep].reset_index(drop=True)
        
        # Compute removed species
        removed_species = [
            species for species in all_species 
            if species not in filtered_df['LipidMolec'].tolist()
        ]
        
        return filtered_df, removed_species
    
    def _evaluate_lipid_for_filtering(
        self,
        row: pd.Series,
        experiment_config: ExperimentConfig,
        threshold: float,
        bqc_label: Optional[str]
    ) -> bool:
        """
        Evaluate whether a single lipid should be kept.
        
        Returns:
            True if lipid should be kept, False if it should be removed
        """
        non_bqc_all_fail = True
        bqc_fail = False if bqc_label is None else True  # Default based on BQC presence
        
        for cond_idx, cond_samples in enumerate(experiment_config.individual_samples_list):
            if not cond_samples:  # Skip empty conditions
                continue
            
            # Count zeros in this condition
            zero_count = self._count_zeros_in_condition(row, cond_samples, threshold)
            n_samples = len(cond_samples)
            
            # Calculate zero proportion
            zero_proportion = zero_count / n_samples if n_samples > 0 else 1.0
            
            # Determine threshold based on condition type
            is_bqc_condition = experiment_config.conditions_list[cond_idx] == bqc_label
            zero_threshold = 0.5 if is_bqc_condition else 0.75
            
            # Check if condition passes its threshold
            if zero_proportion < zero_threshold:
                if is_bqc_condition:
                    bqc_fail = False
                else:
                    non_bqc_all_fail = False
        
        # Retain lipid if BQC does not fail AND not all non-BQC conditions fail
        return not bqc_fail and not non_bqc_all_fail
    
    def _count_zeros_in_condition(
        self,
        row: pd.Series,
        samples: List[str],
        threshold: float
    ) -> int:
        """
        Count how many samples in a condition have values <= threshold.
        
        Args:
            row: DataFrame row
            samples: List of sample names in this condition
            threshold: Threshold value
        
        Returns:
            Count of zeros/low values
        """
        zero_count = 0
        
        for sample in samples:
            col = f'intensity[{sample}]'
            if col in row.index:
                value = row[col]
                if value <= threshold:
                    zero_count += 1
        
        return zero_count
