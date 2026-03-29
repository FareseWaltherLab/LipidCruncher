"""Unit tests for sample dataset metadata and auto-populate configuration.

Validates the structure, completeness, and correctness of the experiment
metadata in SAMPLE_DATA_INFO used to auto-populate experiment config
when loading sample datasets.
"""
import os

import pytest

from app.ui.content.sample_data import SAMPLE_DATA_INFO, get_sample_data_info


# =============================================================================
# Metadata structure validation
# =============================================================================

# Formats that should have experiment metadata
FORMATS_WITH_EXPERIMENT = ['Generic Format', 'LipidSearch 5.0', 'MS-DIAL']

# Metabolomics Workbench uses auto-detection, not metadata
FORMATS_WITHOUT_EXPERIMENT = ['Metabolomics Workbench']


class TestSampleDataInfoStructure:
    """Validate SAMPLE_DATA_INFO structure."""

    def test_all_formats_present(self):
        """All expected formats are in the info dict."""
        expected = ['Generic Format', 'LipidSearch 5.0', 'MS-DIAL', 'Metabolomics Workbench']
        for fmt in expected:
            assert fmt in SAMPLE_DATA_INFO, f"Missing format: {fmt}"

    def test_each_format_has_file(self):
        """Every format has a 'file' key."""
        for fmt, info in SAMPLE_DATA_INFO.items():
            assert 'file' in info, f"{fmt} missing 'file' key"

    def test_each_format_has_description(self):
        """Every format has a 'description' key."""
        for fmt, info in SAMPLE_DATA_INFO.items():
            assert 'description' in info, f"{fmt} missing 'description' key"

    def test_sample_files_exist(self):
        """Referenced sample dataset CSV files exist on disk."""
        sample_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'sample_datasets')
        for fmt, info in SAMPLE_DATA_INFO.items():
            filepath = os.path.join(sample_dir, info['file'])
            assert os.path.isfile(filepath), f"Sample file missing for {fmt}: {info['file']}"


class TestExperimentMetadataPresence:
    """Validate experiment metadata is present for the correct formats."""

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_format_has_experiment_metadata(self, fmt):
        """Formats that should have metadata do have it."""
        info = SAMPLE_DATA_INFO[fmt]
        assert 'experiment' in info, f"{fmt} should have 'experiment' metadata"

    def test_metabolomics_workbench_no_experiment(self):
        """Metabolomics Workbench intentionally has no experiment metadata."""
        info = SAMPLE_DATA_INFO['Metabolomics Workbench']
        assert 'experiment' not in info


class TestExperimentMetadataStructure:
    """Validate experiment metadata dict structure for each format."""

    REQUIRED_KEYS = ['n_conditions', 'conditions', 'samples_per_condition', 'bqc_label']

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_has_all_required_keys(self, fmt):
        """Each experiment metadata has all required keys."""
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        for key in self.REQUIRED_KEYS:
            assert key in exp, f"{fmt} experiment missing key: {key}"

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_n_conditions_is_positive_int(self, fmt):
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        assert isinstance(exp['n_conditions'], int)
        assert exp['n_conditions'] > 0

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_conditions_list_length_matches(self, fmt):
        """conditions list length equals n_conditions."""
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        assert len(exp['conditions']) == exp['n_conditions']

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_samples_per_condition_length_matches(self, fmt):
        """samples_per_condition list length equals n_conditions."""
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        assert len(exp['samples_per_condition']) == exp['n_conditions']

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_samples_per_condition_all_positive(self, fmt):
        """All sample counts are positive integers."""
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        for count in exp['samples_per_condition']:
            assert isinstance(count, int)
            assert count > 0

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_conditions_are_strings(self, fmt):
        """Condition labels are non-empty strings."""
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        for cond in exp['conditions']:
            assert isinstance(cond, str)
            assert len(cond) > 0

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_bqc_label_is_string_or_none(self, fmt):
        """bqc_label is either a string or None."""
        exp = SAMPLE_DATA_INFO[fmt]['experiment']
        assert exp['bqc_label'] is None or isinstance(exp['bqc_label'], str)


# =============================================================================
# Format-specific metadata correctness
# =============================================================================


class TestGenericFormatMetadata:
    """Validate Generic Format experiment metadata."""

    @pytest.fixture
    def exp(self):
        return SAMPLE_DATA_INFO['Generic Format']['experiment']

    def test_n_conditions(self, exp):
        assert exp['n_conditions'] == 3

    def test_conditions(self, exp):
        assert exp['conditions'] == ['WT', 'ADGAT-DKO', 'BQC']

    def test_samples_per_condition(self, exp):
        assert exp['samples_per_condition'] == [4, 4, 4]

    def test_bqc_label(self, exp):
        assert exp['bqc_label'] == 'BQC'

    def test_total_samples(self, exp):
        assert sum(exp['samples_per_condition']) == 12

    def test_bqc_label_in_conditions(self, exp):
        """BQC label should match one of the condition names."""
        assert exp['bqc_label'] in exp['conditions']


class TestLipidSearchMetadata:
    """Validate LipidSearch 5.0 experiment metadata."""

    @pytest.fixture
    def exp(self):
        return SAMPLE_DATA_INFO['LipidSearch 5.0']['experiment']

    def test_n_conditions(self, exp):
        assert exp['n_conditions'] == 3

    def test_conditions(self, exp):
        assert exp['conditions'] == ['WT', 'ADGAT-DKO', 'BQC']

    def test_samples_per_condition(self, exp):
        assert exp['samples_per_condition'] == [4, 4, 4]

    def test_bqc_label(self, exp):
        assert exp['bqc_label'] == 'BQC'

    def test_matches_generic(self, exp):
        """LipidSearch and Generic come from the same study — same experiment config."""
        generic_exp = SAMPLE_DATA_INFO['Generic Format']['experiment']
        assert exp['n_conditions'] == generic_exp['n_conditions']
        assert exp['conditions'] == generic_exp['conditions']
        assert exp['samples_per_condition'] == generic_exp['samples_per_condition']


class TestMSDIALMetadata:
    """Validate MS-DIAL experiment metadata."""

    @pytest.fixture
    def exp(self):
        return SAMPLE_DATA_INFO['MS-DIAL']['experiment']

    def test_n_conditions(self, exp):
        assert exp['n_conditions'] == 3

    def test_conditions(self, exp):
        assert exp['conditions'] == ['Blank', 'fads2 KO', 'Wild-type']

    def test_samples_per_condition(self, exp):
        assert exp['samples_per_condition'] == [1, 3, 3]

    def test_bqc_label_none(self, exp):
        """MS-DIAL sample data has no BQC samples."""
        assert exp['bqc_label'] is None

    def test_total_samples(self, exp):
        assert sum(exp['samples_per_condition']) == 7


# =============================================================================
# get_sample_data_info function tests
# =============================================================================


class TestGetSampleDataInfo:
    """Tests for get_sample_data_info()."""

    def test_returns_dict_for_known_format(self):
        result = get_sample_data_info('Generic Format')
        assert isinstance(result, dict)

    def test_returns_none_for_unknown_format(self):
        result = get_sample_data_info('Nonexistent Format')
        assert result is None

    def test_returns_correct_info(self):
        result = get_sample_data_info('MS-DIAL')
        assert result['file'] == 'msdial_test_dataset.csv'

    @pytest.mark.parametrize('fmt', FORMATS_WITH_EXPERIMENT)
    def test_returns_experiment_for_supported(self, fmt):
        result = get_sample_data_info(fmt)
        assert 'experiment' in result

    def test_metabolomics_workbench_no_experiment(self):
        result = get_sample_data_info('Metabolomics Workbench')
        assert 'experiment' not in result
