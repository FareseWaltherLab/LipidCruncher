"""
Normalization service for lipid data.
Pure business logic - no UI dependencies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from ..models.normalization import NormalizationConfig
from ..models.experiment import ExperimentConfig


class NormalizationService:
    """
    Service for normalizing lipidomic data.
    Handles internal standard normalization, protein normalization, and combined normalization.
    """
    
    def normalize(
        self,
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        intsta_df: pd.DataFrame = None,
        protein_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Main normalization method that routes to appropriate normalization type.
        
        Args:
            df: DataFrame with lipid data
            config: Normalization configuration
            experiment: Experiment configuration
            intsta_df: DataFrame with internal standards (required for internal_standard/both methods)
            protein_df: DataFrame with protein concentrations (required for protein/both methods)
            
        Returns:
            Normalized DataFrame
            
        Raises:
            ValueError: If required data is missing for the chosen method
        """
        # Validate configuration is complete
        config.validate_complete()
        
        # Route to appropriate method
        if config.method == 'none':
            return df.copy()
        
        elif config.method == 'internal_standard':
            if intsta_df is None or intsta_df.empty:
                raise ValueError("intsta_df required for internal_standard normalization")
            return self.normalize_by_internal_standards(df, config, experiment, intsta_df)
        
        elif config.method == 'protein':
            if protein_df is None or protein_df.empty:
                raise ValueError("protein_df required for protein normalization")
            return self.normalize_by_protein(df, config, protein_df)
        
        elif config.method == 'both':
            if intsta_df is None or intsta_df.empty:
                raise ValueError("intsta_df required for combined normalization")
            if protein_df is None or protein_df.empty:
                raise ValueError("protein_df required for combined normalization")
            
            # First apply internal standards
            temp_df = self.normalize_by_internal_standards(df, config, experiment, intsta_df)
            # Then apply protein normalization (preserve prefix since we're chaining)
            return self.normalize_by_protein(temp_df, config, protein_df, preserve_prefix=True)
        
        else:
            raise ValueError(f"Unknown normalization method: {config.method}")
    
    def normalize_by_internal_standards(
        self,
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        intsta_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize using internal standards.
        
        Args:
            df: DataFrame with lipid data
            config: Normalization configuration
            experiment: Experiment configuration
            intsta_df: DataFrame with internal standards
            
        Returns:
            Normalized DataFrame with concentration columns
        """
        # Filter to selected classes
        selected_df = df[df['ClassKey'].isin(config.selected_classes)].copy()
        
        # Remove standards from the dataset
        standards_to_remove = set(config.internal_standards.values())
        selected_df = selected_df[~selected_df['LipidMolec'].isin(standards_to_remove)]
        
        # Process each lipid class
        normalized_dfs = []
        for lipid_class in config.selected_classes:
            standard_name = config.internal_standards[lipid_class]
            concentration = config.intsta_concentrations[standard_name]
            
            # Get AUC for this standard
            intsta_auc = self._compute_intsta_auc(
                intsta_df, 
                standard_name, 
                experiment.samples_list  # FIXED: Changed from full_samples_list
            )
            
            # Normalize this class
            class_df = self._compute_normalized_auc(
                selected_df,
                experiment.samples_list,  # FIXED: Changed from full_samples_list
                lipid_class,
                intsta_auc,
                concentration
            )
            
            normalized_dfs.append(class_df)
        
        # Combine all classes
        result = pd.concat(normalized_dfs, axis=0, ignore_index=True)
        
        # Clean up the data
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)
        
        # Rename columns unless preserving prefix
        if not config.preserve_column_prefix:
            result = self._rename_intensity_to_concentration(result)
        
        return result
    
    def normalize_by_protein(
        self,
        df: pd.DataFrame,
        config: NormalizationConfig,
        protein_df: pd.DataFrame,
        preserve_prefix: bool = False
    ) -> pd.DataFrame:
        """
        Normalize using protein concentrations (BCA assay).
        
        Args:
            df: DataFrame with lipid data
            config: Normalization configuration
            protein_df: DataFrame with protein concentrations
            preserve_prefix: If True, keep 'intensity[' prefix
            
        Returns:
            Normalized DataFrame
        """
        normalized_df = df.copy()
        
        # Set up protein concentrations
        if 'Sample' in protein_df.columns:
            protein_df = protein_df.set_index('Sample')
        
        # Find and normalize intensity columns
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        
        for col in intensity_cols:
            sample_name = col[col.find('[')+1:col.find(']')]
            
            if sample_name in protein_df.index:
                protein_conc = protein_df.loc[sample_name, 'Concentration']
                
                if protein_conc <= 0:
                    # Skip this sample - don't normalize with zero/negative
                    continue
                
                normalized_df[col] = df[col] / protein_conc
        
        # Rename columns unless preserving prefix or explicitly configured
        use_preserve = preserve_prefix or config.preserve_column_prefix
        if not use_preserve:
            normalized_df = self._rename_intensity_to_concentration(normalized_df)
        
        return normalized_df
    
    def _compute_intsta_auc(
        self,
        intsta_df: pd.DataFrame,
        intsta_species: str,
        full_samples_list: List[str]
    ) -> np.ndarray:
        """
        Compute AUC values for the selected internal standard.
        
        Args:
            intsta_df: DataFrame with internal standards
            intsta_species: Name of the standard to use
            full_samples_list: List of sample IDs
            
        Returns:
            Array of AUC values for each sample
        """
        cols = [f"intensity[{sample}]" for sample in full_samples_list]
        filtered_intsta = intsta_df[intsta_df['LipidMolec'] == intsta_species]
        
        if filtered_intsta.empty:
            raise ValueError(f"Standard '{intsta_species}' not found in intsta_df")
        
        return filtered_intsta[cols].values.reshape(len(full_samples_list),)
    
    def _compute_normalized_auc(
        self,
        selected_df: pd.DataFrame,
        full_samples_list: List[str],
        lipid_class: str,
        intsta_auc: np.ndarray,
        concentration: float
    ) -> pd.DataFrame:
        """
        Normalize AUC values for a specific lipid class using internal standard.
        
        Args:
            selected_df: DataFrame with lipid data (already filtered to selected classes)
            full_samples_list: List of sample IDs
            lipid_class: Lipid class to normalize
            intsta_auc: Array of standard AUC values
            concentration: Standard concentration
            
        Returns:
            DataFrame with normalized values for this class
        """
        class_df = selected_df[selected_df['ClassKey'] == lipid_class].copy()
        intensity_cols = [f"intensity[{sample}]" for sample in full_samples_list]
        intensity_df = class_df[intensity_cols]
        
        # Normalize: (lipid_intensity / standard_intensity) * standard_concentration
        normalized_intensity_df = (intensity_df.divide(intsta_auc, axis='columns') * concentration)
        
        # Keep essential columns
        essential_cols = ['LipidMolec', 'ClassKey']
        result_df = pd.concat([
            class_df[essential_cols].reset_index(drop=True),
            normalized_intensity_df.reset_index(drop=True)
        ], axis=1)
        
        return result_df
    
    def _rename_intensity_to_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename intensity columns to concentration columns.
        
        Args:
            df: DataFrame with intensity columns
            
        Returns:
            DataFrame with renamed columns
        """
        renamed_cols = {
            col: col.replace('intensity[', 'concentration[')
            for col in df.columns if 'intensity[' in col
        }
        return df.rename(columns=renamed_cols)
