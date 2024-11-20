class DataFormatHandler:
    """
    Handles initial data validation for different data formats.
    """
    
    @staticmethod
    def validate_and_preprocess(df, data_format):
        """
        Validates data based on format
        
        Args:
            df (pd.DataFrame): Input dataframe
            data_format (str): Either 'lipidsearch' or 'generic'
            
        Returns:
            tuple: (df, success, error_message)
        """
        if data_format == 'lipidsearch':
            return DataFormatHandler._validate_lipidsearch(df)
        else:
            return DataFormatHandler._validate_generic(df)
    
    @staticmethod
    def _validate_lipidsearch(df):
        """Validates LipidSearch format data"""
        required_cols = [
            'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
            'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, False, f"Missing required columns for LipidSearch format: {', '.join(missing_cols)}"
            
        # Check for MeanArea columns
        if not any(col.startswith('MeanArea[') for col in df.columns):
            return None, False, "No MeanArea columns found"
            
        return df, True, "Valid LipidSearch format"
    
    @staticmethod
    def _validate_generic(df):
        """Validates generic format data"""
        # Check minimum requirements
        if 'LipidMolec' not in df.columns:
            return None, False, "Missing required column: LipidMolec"
            
        if not any(col.startswith('MeanArea[') for col in df.columns):
            return None, False, "No MeanArea columns found"
            
        return df, True, "Valid generic format"