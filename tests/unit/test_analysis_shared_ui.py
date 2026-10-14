"""Tests for pure helpers in app.ui.main_content.analysis._shared."""

from app.ui.main_content.analysis._shared import _skipped_tests_message


class TestSkippedTestsMessage:

    def test_species_level_counts_lipids(self):
        params = {
            'n_lipids_total': 120, 'n_lipids_skipped': 3,
            'n_skipped_insufficient_data': 2, 'n_skipped_all_zero': 1,
            'n_skipped_test_error': 0,
        }
        assert _skipped_tests_message(params) == (
            "3 of 120 lipids were skipped during testing (2 had insufficient "
            "data (<2 values per group); 1 had all-zero values in one or both groups)"
        )

    def test_saturation_counts_tests_not_lipids(self):
        """Regression: saturation results report class x fatty-acid-type
        tests, which the message called lipids, with "?" as the total."""
        params = {'n_tests': 7, 'n_skipped': 2}
        assert _skipped_tests_message(params) == (
            "2 of 9 tests (lipid class × fatty-acid type) were skipped during testing"
        )

    def test_nothing_skipped(self):
        assert _skipped_tests_message({'n_tests': 7, 'n_skipped': 0}) is None
        assert _skipped_tests_message({}) is None
