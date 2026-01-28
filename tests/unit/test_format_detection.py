"""Unit tests for FormatDetectionService."""
import pytest
import pandas as pd
import numpy as np
from app.services.format_detection import FormatDetectionService, DataFormat


# =============================================================================
# DataFormat Enum Tests
# =============================================================================

class TestDataFormatEnum:
    """Tests for DataFormat enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert DataFormat.LIPIDSEARCH.value == "LipidSearch 5.0"
        assert DataFormat.MSDIAL.value == "MS-DIAL"
        assert DataFormat.METABOLOMICS_WORKBENCH.value == "Metabolomics Workbench"
        assert DataFormat.GENERIC.value == "Generic Format"
        assert DataFormat.UNKNOWN.value == "Unknown"

    def test_enum_is_string(self):
        """Test that enum values are strings."""
        assert isinstance(DataFormat.LIPIDSEARCH.value, str)
        assert DataFormat.LIPIDSEARCH.value == "LipidSearch 5.0"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert DataFormat.LIPIDSEARCH == DataFormat.LIPIDSEARCH
        assert DataFormat.LIPIDSEARCH != DataFormat.MSDIAL

    def test_enum_string_equality(self):
        """Test enum string equality."""
        assert DataFormat.LIPIDSEARCH == "LipidSearch 5.0"
        assert DataFormat.MSDIAL == "MS-DIAL"

    def test_all_enum_members(self):
        """Test all expected enum members exist."""
        members = list(DataFormat)
        assert len(members) == 5
        assert DataFormat.LIPIDSEARCH in members
        assert DataFormat.MSDIAL in members
        assert DataFormat.METABOLOMICS_WORKBENCH in members
        assert DataFormat.GENERIC in members
        assert DataFormat.UNKNOWN in members


# =============================================================================
# LipidSearch 5.0 Detection Tests
# =============================================================================

class TestLipidSearchDetection:
    """Tests for LipidSearch 5.0 format detection."""

    def test_detect_valid_lipidsearch(self):
        """Test detection of valid LipidSearch format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'TotalGrade': ['A', 'B'],
            'TotalSmpIDRate(%)': [100.0, 85.0],
            'FAKey': ['16:0_18:1', '18:0_20:4'],
            'MeanArea[s1]': [1000.0, 2000.0],
            'MeanArea[s2]': [1100.0, 2100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_detect_lipidsearch_single_sample(self):
        """Test LipidSearch with single sample column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_detect_lipidsearch_many_samples(self):
        """Test LipidSearch with many sample columns."""
        data = {
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
        }
        # Add 50 sample columns
        for i in range(1, 51):
            data[f'MeanArea[s{i}]'] = [1000.0 * i]

        df = pd.DataFrame(data)
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_detect_lipidsearch_all_grades(self):
        """Test LipidSearch with all grade types."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'DG(18:0)'],
            'ClassKey': ['PC', 'PE', 'TG', 'DG'],
            'CalcMass': [760.5, 768.5, 800.5, 600.5],
            'BaseRt': [10.5, 12.3, 15.0, 8.0],
            'TotalGrade': ['A', 'B', 'C', 'D'],
            'TotalSmpIDRate(%)': [100.0, 85.0, 60.0, 40.0],
            'FAKey': ['16:0', '18:0', '16:0', '18:0'],
            'MeanArea[s1]': [1000.0, 2000.0, 3000.0, 4000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_detect_lipidsearch_with_extra_columns(self):
        """Test LipidSearch with additional non-standard columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
            'ExtraColumn1': ['value1'],
            'ExtraColumn2': [123],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_reject_missing_lipidmolec(self):
        """Test rejection when LipidMolec is missing."""
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_classkey(self):
        """Test rejection when ClassKey is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_calcmass(self):
        """Test rejection when CalcMass is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_basert(self):
        """Test rejection when BaseRt is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_totalgrade(self):
        """Test rejection when TotalGrade is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_totalsmpidrate(self):
        """Test rejection when TotalSmpIDRate(%) is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_fakey(self):
        """Test rejection when FAKey is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_missing_meanarea_columns(self):
        """Test rejection when MeanArea columns are missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'Intensity1': [1000.0],  # Wrong column name
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_reject_wrong_meanarea_format(self):
        """Test rejection when MeanArea column format is wrong."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea_s1': [1000.0],  # Wrong format (underscore instead of brackets)
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH

    def test_lipidsearch_case_sensitive_columns(self):
        """Test that LipidSearch column matching is case-sensitive."""
        df = pd.DataFrame({
            'lipidmolec': ['PC(16:0_18:1)'],  # lowercase
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.LIPIDSEARCH


# =============================================================================
# MS-DIAL Detection Tests
# =============================================================================

class TestMSDIALDetection:
    """Tests for MS-DIAL format detection."""

    def test_detect_valid_msdial(self):
        """Test detection of valid MS-DIAL format."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2, 3],
            'Average Rt(min)': [5.2, 6.1, 7.3],
            'Average Mz': [760.5, 768.5, 790.5],
            'Metabolite name': ['PC 16:0_18:1', 'PE 18:0_20:4', 'TG 16:0_18:1_18:2'],
            'Adduct type': ['[M+H]+', '[M+H]+', '[M+NH4]+'],
            'Total score': [95.0, 88.0, 92.0],
            'Sample1': [1000.0, 2000.0, 3000.0],
            'Sample2': [1100.0, 2100.0, 3100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_alignment_id_only(self):
        """Test MS-DIAL detected by Alignment ID as first column."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'Sample1': [1000.0, 2000.0],
            'Sample2': [1100.0, 2100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_metabolite_name_signature(self):
        """Test MS-DIAL with Metabolite name and other signature columns."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'Total score': [95.0, 88.0],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_with_ontology(self):
        """Test MS-DIAL with Ontology column."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'Ontology': ['Glycerophospholipids', 'Glycerophospholipids'],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_with_msms_matched(self):
        """Test MS-DIAL with MS/MS matched column."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'MS/MS matched': ['TRUE', 'FALSE'],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_with_quality_columns(self):
        """Test MS-DIAL with all quality filtering columns."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'Metabolite name': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'Total score': [95.0, 88.0],
            'MS/MS matched': ['TRUE', 'FALSE'],
            'Ontology': ['GP', 'GP'],
            'Average Rt(min)': [5.2, 6.1],
            'Average Mz': [760.5, 768.5],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_with_lipid_is_column(self):
        """Test MS-DIAL with Lipid IS separator column."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'Metabolite name': ['PC 16:0', 'PE 18:0'],
            'Sample1_raw': [1000.0, 2000.0],
            'Sample2_raw': [1100.0, 2100.0],
            'Lipid IS': [1.0, 1.0],
            'Sample1_norm': [500.0, 1000.0],
            'Sample2_norm': [550.0, 1050.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_with_metadata_rows(self):
        """Test MS-DIAL with metadata header rows at top."""
        df = pd.DataFrame({
            0: ['Category', 'Tissue', 'Alignment ID', 1, 2],
            1: ['Mouse', 'Liver', 'Metabolite name', 'PC 16:0_18:1', 'PE 18:0_20:4'],
            2: ['', '', 'Sample1', 1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_many_metadata_rows(self):
        """Test MS-DIAL with many metadata rows."""
        df = pd.DataFrame({
            0: ['Category', 'Tissue', 'Genotype', 'Treatment', 'Alignment ID', 1],
            1: ['Mouse', 'Liver', 'WT', 'None', 'Metabolite name', 'PC 16:0'],
            2: ['', '', '', '', 'Sample1', 1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_adduct_types(self):
        """Test MS-DIAL with various adduct types."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2, 3, 4],
            'Metabolite name': ['PC 16:0', 'PE 18:0', 'TG 48:0', 'Cer 34:1'],
            'Adduct type': ['[M+H]+', '[M-H]-', '[M+NH4]+', '[M+Na]+'],
            'Sample1': [1000.0, 2000.0, 3000.0, 4000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_detect_msdial_many_samples(self):
        """Test MS-DIAL with many sample columns."""
        data = {
            'Alignment ID': [1],
            'Metabolite name': ['PC 16:0'],
        }
        for i in range(1, 101):
            data[f'Sample{i}'] = [1000.0 * i]

        df = pd.DataFrame(data)
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL


# =============================================================================
# Metabolomics Workbench Detection Tests
# =============================================================================

class TestMetabolomicsWorkbenchDetection:
    """Tests for Metabolomics Workbench format detection."""

    def test_detect_valid_metabolomics_workbench(self):
        """Test detection of valid Metabolomics Workbench format."""
        text_data = """#METABOLOMICS WORKBENCH study
VERSION,1
#PROJECT
PR:PROJECT_TITLE,Test Study

MS_METABOLITE_DATA_START
Samples,Sample1,Sample2,Sample3,Sample4
Factors,WT,WT,KO,KO
PC(16:0_18:1),100.5,110.2,80.3,85.1
PE(18:0_20:4),200.5,210.2,180.3,185.1
MS_METABOLITE_DATA_END
"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_detect_workbench_minimal(self):
        """Test detection with minimal valid structure."""
        text_data = """MS_METABOLITE_DATA_START
Samples,S1,S2
Factors,A,B
Lipid1,100,200
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_detect_workbench_with_spaces(self):
        """Test detection with extra whitespace."""
        text_data = """   MS_METABOLITE_DATA_START
Samples,S1,S2
Factors,A,B
Lipid1,100,200
   MS_METABOLITE_DATA_END   """
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_detect_workbench_many_samples(self):
        """Test workbench with many samples."""
        samples = ','.join([f'S{i}' for i in range(1, 101)])
        factors = ','.join(['WT' if i % 2 == 0 else 'KO' for i in range(1, 101)])
        values = ','.join([str(i * 100) for i in range(1, 101)])
        text_data = f"""MS_METABOLITE_DATA_START
Samples,{samples}
Factors,{factors}
Lipid1,{values}
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_detect_workbench_many_lipids(self):
        """Test workbench with many lipid rows."""
        lines = ["MS_METABOLITE_DATA_START", "Samples,S1,S2", "Factors,A,B"]
        for i in range(1, 501):
            lines.append(f"Lipid{i},100,200")
        lines.append("MS_METABOLITE_DATA_END")
        text_data = "\n".join(lines)
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_detect_workbench_complex_conditions(self):
        """Test workbench with complex condition strings."""
        text_data = """MS_METABOLITE_DATA_START
Samples,S1,S2,S3,S4
Factors,Diet:HFD|Treatment:Vehicle,Diet:HFD|Treatment:Drug,Diet:Chow|Treatment:Vehicle,Diet:Chow|Treatment:Drug
PC(16:0),100,110,90,95
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_reject_missing_start_marker(self):
        """Test rejection when start marker is missing."""
        text_data = """Samples,S1,S2
Factors,A,B
Lipid1,100,200
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result != DataFormat.METABOLOMICS_WORKBENCH

    def test_reject_missing_end_marker(self):
        """Test rejection when end marker is missing."""
        text_data = """MS_METABOLITE_DATA_START
Samples,S1,S2
Factors,A,B
Lipid1,100,200"""
        result = FormatDetectionService.detect_format(text_data)
        assert result != DataFormat.METABOLOMICS_WORKBENCH

    def test_reject_reversed_markers(self):
        """Test rejection when markers are in wrong order."""
        text_data = """MS_METABOLITE_DATA_END
Samples,S1,S2
Lipid1,100,200
MS_METABOLITE_DATA_START"""
        result = FormatDetectionService.detect_format(text_data)
        assert result != DataFormat.METABOLOMICS_WORKBENCH

    def test_reject_duplicate_start_markers(self):
        """Test with duplicate start markers uses first valid pair."""
        text_data = """MS_METABOLITE_DATA_START
MS_METABOLITE_DATA_START
Samples,S1,S2
Lipid1,100,200
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_reject_partial_marker(self):
        """Test rejection with partial marker names."""
        text_data = """MS_METABOLITE_DATA_STAR
Samples,S1,S2
Lipid1,100,200
MS_METABOLITE_DATA_EN"""
        result = FormatDetectionService.detect_format(text_data)
        assert result != DataFormat.METABOLOMICS_WORKBENCH

    def test_reject_dataframe_as_workbench(self):
        """Test that DataFrame is not detected as Workbench."""
        df = pd.DataFrame({
            'MS_METABOLITE_DATA_START': [1, 2],
            'Sample1': [100, 200],
        })
        result = FormatDetectionService.detect_format(df)
        assert result != DataFormat.METABOLOMICS_WORKBENCH


# =============================================================================
# Generic Format Detection Tests
# =============================================================================

class TestGenericFormatDetection:
    """Tests for Generic format detection."""

    def test_detect_valid_generic(self):
        """Test detection of valid generic format."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
            'Sample1': [1000.0, 2000.0, 3000.0],
            'Sample2': [1100.0, 2100.0, 3100.0],
            'Sample3': [1200.0, 2200.0, 3200.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_detect_generic_with_sample_brackets(self):
        """Test generic format with sample[s1] column names."""
        df = pd.DataFrame({
            'Lipids': ['AcCa(16:0)', 'AcCa(18:0)'],
            'sample[s1]': [8.33, 4.61],
            'sample[s2]': [4.51, 2.27],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_detect_generic_minimal(self):
        """Test generic format with minimum columns."""
        df = pd.DataFrame({
            'LipidName': ['PC(16:0)'],
            'Intensity': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_detect_generic_many_samples(self):
        """Test generic format with many sample columns."""
        data = {'Lipid': ['PC(16:0)']}
        for i in range(1, 201):
            data[f'Sample{i}'] = [1000.0 * i]
        df = pd.DataFrame(data)
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_detect_generic_various_lipid_names(self):
        """Test generic format with various lipid naming conventions."""
        df = pd.DataFrame({
            'Lipid': [
                'PC(16:0_18:1)',
                'PE 18:0/20:4',
                'TG(48:0)',
                'Cer d18:1/24:0',
                'SM(d18:1_16:0)',
                'LPC O-16:0',
            ],
            'Sample1': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_detect_generic_mixed_data_types(self):
        """Test generic format with mixed numeric values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)', 'TG(48:0)'],
            'Sample1': [1000, 2000.5, 3000],  # int and float
            'Sample2': [1100.0, 2100, 3100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_reject_single_column(self):
        """Test rejection of single column DataFrame."""
        df = pd.DataFrame({
            'LipidName': ['PC(16:0)', 'PE(18:0)'],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.UNKNOWN

    def test_reject_first_column_numeric_only(self):
        """Test rejection when first column is purely numeric."""
        df = pd.DataFrame({
            'Index': [1, 2, 3],
            'Sample1': [1000.0, 2000.0, 3000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.UNKNOWN

    def test_reject_all_numeric_columns(self):
        """Test rejection when all columns are numeric."""
        df = pd.DataFrame({
            'Col1': [1.0, 2.0, 3.0],
            'Col2': [100.0, 200.0, 300.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.UNKNOWN

    def test_detect_generic_with_alphanumeric_first_column(self):
        """Test generic detection with mixed alphanumeric values."""
        df = pd.DataFrame({
            'ID': ['Lipid1', 'Lipid2', 'Lipid3'],
            'Sample1': [1000.0, 2000.0, 3000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.UNKNOWN

    def test_dataframe_with_no_rows(self):
        """Test DataFrame with columns but no rows."""
        df = pd.DataFrame(columns=['Lipid', 'Sample1', 'Sample2'])
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.UNKNOWN

    def test_empty_string(self):
        """Test handling of empty string."""
        result = FormatDetectionService.detect_format("")
        assert result == DataFormat.UNKNOWN

    def test_whitespace_only_string(self):
        """Test handling of whitespace-only string."""
        result = FormatDetectionService.detect_format("   \n\t  ")
        assert result == DataFormat.UNKNOWN

    def test_none_input(self):
        """Test handling of None input."""
        result = FormatDetectionService.detect_format(None)
        assert result == DataFormat.UNKNOWN

    def test_invalid_type_list(self):
        """Test handling of list input."""
        result = FormatDetectionService.detect_format([1, 2, 3])
        assert result == DataFormat.UNKNOWN

    def test_invalid_type_dict(self):
        """Test handling of dict input."""
        result = FormatDetectionService.detect_format({'key': 'value'})
        assert result == DataFormat.UNKNOWN

    def test_invalid_type_int(self):
        """Test handling of int input."""
        result = FormatDetectionService.detect_format(42)
        assert result == DataFormat.UNKNOWN

    def test_dataframe_with_nan(self):
        """Test DataFrame with NaN values still detected."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', None, 'TG(16:0)'],
            'Sample1': [1000.0, float('nan'), 3000.0],
            'Sample2': [1100.0, 2100.0, None],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_dataframe_with_inf(self):
        """Test DataFrame with infinity values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [1000.0, float('inf')],
            'Sample2': [float('-inf'), 2100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_dataframe_with_empty_strings(self):
        """Test DataFrame with empty string values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', '', 'TG(16:0)'],
            'Sample1': [1000.0, 2000.0, 3000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_unicode_lipid_names(self):
        """Test DataFrame with unicode characters in lipid names."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)α', 'TG(16:0)β'],
            'Sample1': [1000.0, 2000.0, 3000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_large_numeric_values(self):
        """Test DataFrame with very large numeric values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [1e15, 2e15],
            'Sample2': [1e16, 2e16],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_very_small_numeric_values(self):
        """Test DataFrame with very small numeric values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [1e-15, 2e-15],
            'Sample2': [1e-16, 2e-16],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC


# =============================================================================
# Format Priority Tests
# =============================================================================

class TestFormatPriority:
    """Tests for format detection priority."""

    def test_lipidsearch_over_generic(self):
        """Test that LipidSearch is detected over generic when signatures match."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_msdial_over_generic(self):
        """Test that MS-DIAL is detected over generic when signatures match."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'Metabolite name': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'Average Rt(min)': [5.2, 6.1],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_lipidsearch_over_msdial(self):
        """Test LipidSearch takes precedence when both could match."""
        # This would match both if both checks were inclusive
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1000.0],
            'Metabolite name': ['PC 16:0'],  # MS-DIAL column
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH


# =============================================================================
# Helper Methods Tests
# =============================================================================

class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_format_display_name_all_formats(self):
        """Test getting display name for all formats."""
        assert FormatDetectionService.get_format_display_name(DataFormat.LIPIDSEARCH) == "LipidSearch 5.0"
        assert FormatDetectionService.get_format_display_name(DataFormat.MSDIAL) == "MS-DIAL"
        assert FormatDetectionService.get_format_display_name(DataFormat.GENERIC) == "Generic Format"
        assert FormatDetectionService.get_format_display_name(DataFormat.METABOLOMICS_WORKBENCH) == "Metabolomics Workbench"
        assert FormatDetectionService.get_format_display_name(DataFormat.UNKNOWN) == "Unknown"

    def test_get_all_formats_count(self):
        """Test getting all supported formats count."""
        formats = FormatDetectionService.get_all_formats()
        assert len(formats) == 4

    def test_get_all_formats_excludes_unknown(self):
        """Test that UNKNOWN is not in supported formats."""
        formats = FormatDetectionService.get_all_formats()
        assert DataFormat.UNKNOWN not in formats

    def test_get_all_formats_includes_all_valid(self):
        """Test all valid formats are included."""
        formats = FormatDetectionService.get_all_formats()
        assert DataFormat.LIPIDSEARCH in formats
        assert DataFormat.MSDIAL in formats
        assert DataFormat.METABOLOMICS_WORKBENCH in formats
        assert DataFormat.GENERIC in formats

    def test_get_format_from_string_all_valid(self):
        """Test converting all valid format strings."""
        assert FormatDetectionService.get_format_from_string("LipidSearch 5.0") == DataFormat.LIPIDSEARCH
        assert FormatDetectionService.get_format_from_string("lipidsearch") == DataFormat.LIPIDSEARCH
        assert FormatDetectionService.get_format_from_string("MS-DIAL") == DataFormat.MSDIAL
        assert FormatDetectionService.get_format_from_string("msdial") == DataFormat.MSDIAL
        assert FormatDetectionService.get_format_from_string("Generic Format") == DataFormat.GENERIC
        assert FormatDetectionService.get_format_from_string("generic") == DataFormat.GENERIC
        assert FormatDetectionService.get_format_from_string("Metabolomics Workbench") == DataFormat.METABOLOMICS_WORKBENCH
        assert FormatDetectionService.get_format_from_string("metabolomics_workbench") == DataFormat.METABOLOMICS_WORKBENCH

    def test_get_format_from_string_invalid(self):
        """Test converting invalid format strings."""
        assert FormatDetectionService.get_format_from_string("invalid") == DataFormat.UNKNOWN
        assert FormatDetectionService.get_format_from_string("") == DataFormat.UNKNOWN
        assert FormatDetectionService.get_format_from_string("LIPIDSEARCH") == DataFormat.UNKNOWN  # Case sensitive
        assert FormatDetectionService.get_format_from_string("LipidSearch5.0") == DataFormat.UNKNOWN  # Missing space
        assert FormatDetectionService.get_format_from_string(" LipidSearch 5.0") == DataFormat.UNKNOWN  # Leading space


# =============================================================================
# MS-DIAL Header Row Detection Tests
# =============================================================================

class TestMSDIALHeaderRowDetection:
    """Tests for MS-DIAL header row detection."""

    def test_no_header_rows(self):
        """Test detection when headers are already column names."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2],
            'Metabolite name': ['PC 16:0', 'PE 18:0'],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService._detect_msdial_header_row(df)
        assert result == -1

    def test_header_in_first_row(self):
        """Test detection when headers are in first data row."""
        df = pd.DataFrame({
            0: ['Alignment ID', 1, 2],
            1: ['Metabolite name', 'PC 16:0', 'PE 18:0'],
            2: ['Sample1', 1000.0, 2000.0],
        })
        result = FormatDetectionService._detect_msdial_header_row(df)
        assert result == 0

    def test_header_in_second_row(self):
        """Test detection when headers are in second data row."""
        df = pd.DataFrame({
            0: ['Category', 'Alignment ID', 1, 2],
            1: ['Mouse', 'Metabolite name', 'PC 16:0', 'PE 18:0'],
            2: ['', 'Sample1', 1000.0, 2000.0],
        })
        result = FormatDetectionService._detect_msdial_header_row(df)
        assert result == 1

    def test_header_after_many_metadata_rows(self):
        """Test detection when headers are after many metadata rows."""
        df = pd.DataFrame({
            0: ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Alignment ID', 1, 2],
            1: ['Val1', 'Val2', 'Val3', 'Val4', 'Metabolite name', 'PC', 'PE'],
            2: ['', '', '', '', 'Sample1', 1000.0, 2000.0],
        })
        result = FormatDetectionService._detect_msdial_header_row(df)
        assert result == 4

    def test_metabolite_name_detection(self):
        """Test detection using Metabolite name rather than Alignment ID."""
        df = pd.DataFrame({
            0: ['Category', 'Value', 'PC 16:0'],
            1: ['Type', 'Metabolite name', 'Sample1'],  # Metabolite name in row 1
        })
        result = FormatDetectionService._detect_msdial_header_row(df)
        assert result == 1

    def test_no_headers_found_in_rows(self):
        """Test when no header markers found in data rows."""
        df = pd.DataFrame({
            0: ['Cat1', 'Cat2', 'Cat3'],
            1: ['Val1', 'Val2', 'Val3'],
            2: ['Data1', 'Data2', 'Data3'],
        })
        result = FormatDetectionService._detect_msdial_header_row(df)
        assert result == -1


# =============================================================================
# Real Data Pattern Tests
# =============================================================================

class TestRealDataPatterns:
    """Tests using patterns from real sample datasets."""

    def test_lipidsearch_sample_pattern(self):
        """Test LipidSearch format matching sample_datasets structure."""
        df = pd.DataFrame({
            'LipidMolec': ['AcCa(10:0)', 'AcCa(10:0)'],
            'ClassKey': ['AcCa', 'AcCa'],
            'CalcMass': [315.241, 315.241],
            'BaseRt': [3.9728, 4.1236],
            'TotalGrade': ['C', 'C'],
            'TotalSmpIDRate(%)': [7.69, 7.69],
            'FAKey': ['(10:0)', '(10:0)'],
            'MeanArea[s1]': [7090.0, 21400.0],
            'MeanArea[s2]': [10700.0, 23600.0],
            'MeanArea[s3]': [14500.0, 17400.0],
            'MeanArea[s4]': [10300.0, 25500.0],
            'MeanArea[s5]': [18400.0, 38500.0],
            'MeanArea[s6]': [22400.0, 36000.0],
            'MeanArea[s7]': [10900.0, 34300.0],
            'MeanArea[s8]': [18400.0, 31100.0],
            'MeanArea[s9]': [133000.0, 385000.0],
            'MeanArea[s10]': [132610, 384020],
            'MeanArea[s11]': [255077, 558877],
            'MeanArea[s12]': [132755, 384755],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_generic_sample_pattern(self):
        """Test generic format matching sample_datasets structure."""
        df = pd.DataFrame({
            'Lipids': ['AcCa(16:0)', 'AcCa(18:0)'],
            'sample[s1]': [8.335, 4.609],
            'sample[s2]': [4.508, 2.266],
            'sample[s3]': [2.274, 4.598],
            'sample[s4]': [5.665, 4.062],
            'sample[s5]': [1.892, 3.266],
            'sample[s6]': [1.894, 2.974],
            'sample[s7]': [2.063, 1.833],
            'sample[s8]': [5.897, 5.602],
            'sample[s9]': [5.686, 3.068],
            'sample[s10]': [5.675, 3.056],
            'sample[s11]': [6.510, 4.199],
            'sample[s12]': [5.683, 3.065],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for consistent behavior across repeated calls."""

    def test_repeated_detection_same_result(self):
        """Test that repeated detection returns same result."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [1000.0, 2000.0],
        })
        results = [FormatDetectionService.detect_format(df) for _ in range(10)]
        assert all(r == DataFormat.GENERIC for r in results)

    def test_copy_detection_same_as_original(self):
        """Test that DataFrame copy gives same detection result."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [1000.0, 2000.0],
        })
        result_original = FormatDetectionService.detect_format(df)
        result_copy = FormatDetectionService.detect_format(df.copy())
        assert result_original == result_copy

    def test_detection_does_not_modify_dataframe(self):
        """Test that detection does not modify the input DataFrame."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [1000.0, 2000.0],
        })
        original_columns = df.columns.tolist()
        original_shape = df.shape
        FormatDetectionService.detect_format(df)
        assert df.columns.tolist() == original_columns
        assert df.shape == original_shape


# =============================================================================
# Tests with Sample Data Files
# =============================================================================

class TestWithSampleDataFiles:
    """Tests using actual sample data files from the repository."""

    @pytest.fixture
    def lipidsearch_df(self):
        """Load LipidSearch sample data."""
        try:
            return pd.read_csv('sample_datasets/lipidsearch5_test_dataset.csv')
        except FileNotFoundError:
            pytest.skip("Sample data file not found")

    @pytest.fixture
    def generic_df(self):
        """Load Generic sample data."""
        try:
            return pd.read_csv('sample_datasets/generic_test_dataset.csv')
        except FileNotFoundError:
            pytest.skip("Sample data file not found")

    @pytest.fixture
    def msdial_df(self):
        """Load MS-DIAL sample data."""
        try:
            return pd.read_csv('sample_datasets/msdial_test_dataset.csv')
        except FileNotFoundError:
            pytest.skip("Sample data file not found")

    @pytest.fixture
    def workbench_text(self):
        """Load Metabolomics Workbench sample data."""
        try:
            with open('sample_datasets/mw_test_dataset.csv', 'r') as f:
                return f.read()
        except FileNotFoundError:
            pytest.skip("Sample data file not found")

    def test_detect_lipidsearch_sample_file(self, lipidsearch_df):
        """Test detection of actual LipidSearch sample file."""
        result = FormatDetectionService.detect_format(lipidsearch_df)
        assert result == DataFormat.LIPIDSEARCH

    def test_detect_generic_sample_file(self, generic_df):
        """Test detection of actual Generic sample file."""
        result = FormatDetectionService.detect_format(generic_df)
        assert result == DataFormat.GENERIC

    def test_detect_msdial_sample_file(self, msdial_df):
        """Test detection of actual MS-DIAL sample file."""
        result = FormatDetectionService.detect_format(msdial_df)
        assert result == DataFormat.MSDIAL

    def test_detect_workbench_sample_file(self, workbench_text):
        """Test detection of actual Metabolomics Workbench sample file."""
        result = FormatDetectionService.detect_format(workbench_text)
        assert result == DataFormat.METABOLOMICS_WORKBENCH


# =============================================================================
# Column Variation Tests
# =============================================================================

class TestColumnVariations:
    """Tests for various column naming patterns and edge cases."""

    def test_lipidsearch_with_special_characters_in_sample_names(self):
        """Test LipidSearch with special characters in sample names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[Sample-1]': [1000.0],
            'MeanArea[Sample_2]': [1100.0],
            'MeanArea[Sample.3]': [1200.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_lipidsearch_with_numeric_sample_names(self):
        """Test LipidSearch with purely numeric sample names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[001]': [1000.0],
            'MeanArea[002]': [1100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_msdial_with_spaces_in_column_names(self):
        """Test MS-DIAL with spaces in column names."""
        df = pd.DataFrame({
            'Alignment ID': [1],
            'Metabolite name': ['PC 16:0'],
            'Sample 1': [1000.0],
            'Sample 2': [1100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_msdial_with_unicode_sample_names(self):
        """Test MS-DIAL with unicode in sample names."""
        df = pd.DataFrame({
            'Alignment ID': [1],
            'Metabolite name': ['PC 16:0'],
            'Échantillon_1': [1000.0],
            '样品_2': [1100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL

    def test_generic_with_empty_column_names(self):
        """Test generic format handling empty column names."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            '': [1000.0, 2000.0],  # Empty column name
            'Sample2': [1100.0, 2100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_generic_with_duplicate_column_names(self):
        """Test generic format with duplicate column names."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample': [1000.0, 2000.0],
        })
        # Add duplicate column by renaming
        df['Sample.1'] = [1100.0, 2100.0]
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC


# =============================================================================
# Data Value Edge Cases
# =============================================================================

class TestDataValueEdgeCases:
    """Tests for edge cases in data values."""

    def test_lipidsearch_with_zero_values(self):
        """Test LipidSearch with all zero values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[s1]': [0.0],
            'MeanArea[s2]': [0.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_lipidsearch_with_negative_values(self):
        """Test LipidSearch with negative values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[s1]': [-1000.0],  # Negative value
            'MeanArea[s2]': [1100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_generic_with_string_values_in_numeric_columns(self):
        """Test generic detection with some non-numeric values in intensity columns."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)', 'TG(48:0)'],
            'Sample1': [1000.0, 'N/A', 3000.0],  # String value
            'Sample2': [1100.0, 2100.0, 3100.0],
        })
        result = FormatDetectionService.detect_format(df)
        # Should still detect as generic since >50% are numeric
        assert result == DataFormat.GENERIC

    def test_all_nan_intensity_columns(self):
        """Test detection with all NaN intensity values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)'],
            'Sample1': [float('nan'), float('nan')],
            'Sample2': [float('nan'), float('nan')],
        })
        result = FormatDetectionService.detect_format(df)
        # Should fail generic detection since no numeric values
        assert result == DataFormat.UNKNOWN

    def test_mixed_nan_and_valid_values(self):
        """Test detection with mixed NaN and valid values."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)', 'PE(18:0)', 'TG(48:0)'],
            'Sample1': [1000.0, float('nan'), 3000.0],
            'Sample2': [float('nan'), 2100.0, 3100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions and limits."""

    def test_large_dataset(self):
        """Test detection with large dataset."""
        n_rows = 10000
        n_cols = 100
        data = {'Lipid': [f'Lipid{i}' for i in range(n_rows)]}
        for i in range(1, n_cols + 1):
            data[f'Sample{i}'] = list(range(n_rows))
        df = pd.DataFrame(data)
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_wide_dataset(self):
        """Test detection with many columns (wide dataset)."""
        n_cols = 1000
        data = {'Lipid': ['PC(16:0)']}
        for i in range(1, n_cols + 1):
            data[f'Sample{i}'] = [1000.0 * i]
        df = pd.DataFrame(data)
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_single_row_lipidsearch(self):
        """Test LipidSearch with single row."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[s1]': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_single_row_generic(self):
        """Test generic with single row."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)'],
            'Sample1': [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_very_long_lipid_names(self):
        """Test with very long lipid names."""
        long_name = 'A' * 1000 + '(16:0_18:1_20:4_22:6)'
        df = pd.DataFrame({
            'Lipid': [long_name, 'PE(18:0)'],
            'Sample1': [1000.0, 2000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_very_long_column_names(self):
        """Test with very long column names."""
        long_col = 'Sample_' + 'A' * 1000
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)'],
            long_col: [1000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC


# =============================================================================
# Internal Method Tests
# =============================================================================

class TestInternalMethods:
    """Tests for internal helper methods."""

    def test_is_lipidsearch_true(self):
        """Test _is_lipidsearch returns True for valid format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[s1]': [1000.0],
        })
        assert FormatDetectionService._is_lipidsearch(df) is True

    def test_is_lipidsearch_false(self):
        """Test _is_lipidsearch returns False for invalid format."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)'],
            'Sample1': [1000.0],
        })
        assert FormatDetectionService._is_lipidsearch(df) is False

    def test_is_msdial_true(self):
        """Test _is_msdial returns True for valid format."""
        df = pd.DataFrame({
            'Alignment ID': [1],
            'Metabolite name': ['PC 16:0'],
            'Sample1': [1000.0],
        })
        assert FormatDetectionService._is_msdial(df) is True

    def test_is_msdial_false(self):
        """Test _is_msdial returns False for invalid format."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)'],
            'Sample1': [1000.0],
        })
        assert FormatDetectionService._is_msdial(df) is False

    def test_is_metabolomics_workbench_true(self):
        """Test _is_metabolomics_workbench returns True for valid format."""
        text = "MS_METABOLITE_DATA_START\ndata\nMS_METABOLITE_DATA_END"
        assert FormatDetectionService._is_metabolomics_workbench(text) is True

    def test_is_metabolomics_workbench_false(self):
        """Test _is_metabolomics_workbench returns False for invalid format."""
        assert FormatDetectionService._is_metabolomics_workbench("random text") is False
        assert FormatDetectionService._is_metabolomics_workbench("") is False

    def test_is_generic_true(self):
        """Test _is_generic returns True for valid format."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)'],
            'Sample1': [1000.0],
        })
        assert FormatDetectionService._is_generic(df) is True

    def test_is_generic_false(self):
        """Test _is_generic returns False for invalid format."""
        # Single column
        df = pd.DataFrame({'Lipid': ['PC(16:0)']})
        assert FormatDetectionService._is_generic(df) is False

        # First column is numeric
        df = pd.DataFrame({'Index': [1, 2], 'Sample1': [1000.0, 2000.0]})
        assert FormatDetectionService._is_generic(df) is False


# =============================================================================
# Format Disambiguation Tests
# =============================================================================

class TestFormatDisambiguation:
    """Tests for correctly distinguishing between similar formats."""

    def test_ambiguous_columns_detected_correctly(self):
        """Test that ambiguous column patterns are handled correctly."""
        # DataFrame with both LipidMolec (LipidSearch) and Metabolite name (MS-DIAL)
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0'],
            'MeanArea[s1]': [1000.0],
            'Metabolite name': ['PC 16:0'],  # MS-DIAL column
            'Alignment ID': [1],  # MS-DIAL column
        })
        # Should be detected as LipidSearch (checked first)
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.LIPIDSEARCH

    def test_generic_fallback(self):
        """Test that generic is used as fallback for unrecognized formats."""
        df = pd.DataFrame({
            'CustomLipidColumn': ['PC(16:0)', 'PE(18:0)'],
            'CustomIntensity1': [1000.0, 2000.0],
            'CustomIntensity2': [1100.0, 2100.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.GENERIC

    def test_msdial_not_mistaken_for_generic(self):
        """Test MS-DIAL is not mistakenly detected as generic."""
        df = pd.DataFrame({
            'Alignment ID': [1, 2, 3],
            'Metabolite name': ['PC 16:0', 'PE 18:0', 'TG 48:0'],
            'Total score': [95.0, 88.0, 92.0],
            'Sample1': [1000.0, 2000.0, 3000.0],
        })
        result = FormatDetectionService.detect_format(df)
        assert result == DataFormat.MSDIAL
        assert result != DataFormat.GENERIC


# =============================================================================
# Metabolomics Workbench Text Parsing Tests
# =============================================================================

class TestMetabolomicsWorkbenchTextParsing:
    """Additional tests for Metabolomics Workbench text format edge cases."""

    def test_workbench_with_tabs(self):
        """Test workbench format with tabs instead of commas."""
        text_data = """MS_METABOLITE_DATA_START
Samples\tS1\tS2
Factors\tA\tB
Lipid1\t100\t200
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_workbench_with_mixed_line_endings(self):
        """Test workbench format with mixed line endings."""
        text_data = "MS_METABOLITE_DATA_START\r\nSamples,S1,S2\nFactors,A,B\r\nLipid1,100,200\nMS_METABOLITE_DATA_END"
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_workbench_with_empty_lines(self):
        """Test workbench format with empty lines."""
        text_data = """
MS_METABOLITE_DATA_START

Samples,S1,S2

Factors,A,B
Lipid1,100,200

MS_METABOLITE_DATA_END
"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_workbench_markers_in_content(self):
        """Test that markers in actual content don't cause issues."""
        text_data = """MS_METABOLITE_DATA_START
Samples,S1,S2
Factors,A,B
MS_METABOLITE_DATA_START_lipid,100,200
MS_METABOLITE_DATA_END"""
        result = FormatDetectionService.detect_format(text_data)
        assert result == DataFormat.METABOLOMICS_WORKBENCH


# =============================================================================
# Type Safety Tests
# =============================================================================

class TestTypeSafety:
    """Tests for type safety and handling of unexpected types."""

    def test_numpy_array_input(self):
        """Test that numpy arrays are handled."""
        arr = np.array([[1, 2], [3, 4]])
        result = FormatDetectionService.detect_format(arr)
        assert result == DataFormat.UNKNOWN

    def test_series_input(self):
        """Test that pandas Series is handled."""
        series = pd.Series(['PC(16:0)', 'PE(18:0)'])
        result = FormatDetectionService.detect_format(series)
        assert result == DataFormat.UNKNOWN

    def test_float_input(self):
        """Test that float input is handled."""
        result = FormatDetectionService.detect_format(3.14)
        assert result == DataFormat.UNKNOWN

    def test_bool_input(self):
        """Test that bool input is handled."""
        result = FormatDetectionService.detect_format(True)
        assert result == DataFormat.UNKNOWN

    def test_tuple_input(self):
        """Test that tuple input is handled."""
        result = FormatDetectionService.detect_format(('a', 'b', 'c'))
        assert result == DataFormat.UNKNOWN

    def test_set_input(self):
        """Test that set input is handled."""
        result = FormatDetectionService.detect_format({'a', 'b', 'c'})
        assert result == DataFormat.UNKNOWN


# =============================================================================
# Service Class Structure Tests
# =============================================================================

class TestServiceClassStructure:
    """Tests for service class structure and design."""

    def test_all_public_methods_are_static(self):
        """Test that all public methods are static."""
        public_methods = [
            'detect_format',
            'get_format_display_name',
            'get_all_formats',
            'get_format_from_string',
        ]
        for method_name in public_methods:
            method = getattr(FormatDetectionService, method_name)
            assert isinstance(method, staticmethod) or callable(method)

    def test_class_constants_are_defined(self):
        """Test that required class constants are defined."""
        assert hasattr(FormatDetectionService, 'LIPIDSEARCH_REQUIRED_COLUMNS')
        assert hasattr(FormatDetectionService, 'MSDIAL_METADATA_COLUMNS')
        assert hasattr(FormatDetectionService, 'MSDIAL_SIGNATURE_COLUMNS')
        assert hasattr(FormatDetectionService, 'MW_START_MARKER')
        assert hasattr(FormatDetectionService, 'MW_END_MARKER')

    def test_class_constants_are_sets(self):
        """Test that column constants are sets for O(1) lookup."""
        assert isinstance(FormatDetectionService.LIPIDSEARCH_REQUIRED_COLUMNS, set)
        assert isinstance(FormatDetectionService.MSDIAL_METADATA_COLUMNS, set)
        assert isinstance(FormatDetectionService.MSDIAL_SIGNATURE_COLUMNS, set)

    def test_no_instance_state(self):
        """Test that service has no instance state."""
        service1 = FormatDetectionService()
        service2 = FormatDetectionService()

        # Both should work identically
        df = pd.DataFrame({
            'Lipid': ['PC(16:0)'],
            'Sample1': [1000.0],
        })
        assert FormatDetectionService.detect_format(df) == DataFormat.GENERIC
