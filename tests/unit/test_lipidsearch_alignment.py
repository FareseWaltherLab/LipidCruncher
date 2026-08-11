"""Tests for LipidSearch 5.2 dual-polarity alignment parsing and merging."""
import numpy as np
import pandas as pd
import pytest

from app.services.data_standardization import DataStandardizationService
from app.services.lipidsearch_alignment import (
    AlignmentError,
    AlignmentMap,
    parse_alignment_file,
    merge_dual_polarity,
    has_grouped_per_file_columns,
    _sample_base,
)


def _alignment_text(rows):
    """Build a minimal Alignment Setting file. `rows` = list of
    (filename, dataid, condition)."""
    head = (
        "*Parameters setting\n"
        "Job Name\tTestJob\n"
        "NormalizeType\tNONE\n"
        "\n"
        "*Target search job\n"
    )
    body = "".join(
        f"TestJob\t{fname}\t{dataid}\t{cond}\tuuid-{i}\n"
        for i, (fname, dataid, cond) in enumerate(rows)
    )
    return head + body


# Two conditions: Control has 2 mice (01, 02), Treated has 1 mouse (03).
_ROWS = [
    ("sample_01n.raw", "s1-1", "*Control"),
    ("sample_01p.raw", "s1-2", "*Control"),
    ("sample_02n.raw", "s1-3", "*Control"),
    ("sample_02p.raw", "s1-4", "*Control"),
    ("sample_03n.raw", "s2-1", "Treated"),
    ("sample_03p.raw", "s2-2", "Treated"),
]


class TestSampleBase:
    """Filename base extraction strips extension + polarity marker."""

    @pytest.mark.parametrize("fname,expected", [
        ("140509_FT_01n.raw", "140509_FT_01"),
        ("140509_FT_01p.raw", "140509_FT_01"),
        ("sample_02_neg.raw", "sample_02"),
        ("sample_02_pos.raw", "sample_02"),
        ("mouseA-negative.mzML", "mouseA"),
        ("mouseA-positive.mzML", "mouseA"),
        ("run5P.raw", "run5"),
    ])
    def test_strips_polarity_and_extension(self, fname, expected):
        assert _sample_base(fname) == expected

    def test_pos_and_neg_of_same_sample_share_base(self):
        assert _sample_base("x01n.raw") == _sample_base("x01p.raw")


class TestParseAlignment:
    """Parsing the *Target search job section into biological samples."""

    def test_groups_into_samples(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        assert isinstance(m, AlignmentMap)
        assert len(m.samples) == 3
        # First sample pairs the two Control files of mouse 01.
        assert m.samples[0].dataids == ["s1-1", "s1-2"]
        assert m.samples[0].base == "sample_01"

    def test_condition_asterisk_stripped(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        assert m.conditions == ["Control", "Treated"]

    def test_samples_per_condition(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        assert m.samples_per_condition == [2, 1]

    def test_ignores_rows_before_target_section(self):
        # The header lines (Job Name, NormalizeType) must not become samples.
        m = parse_alignment_file(_alignment_text(_ROWS))
        assert sum(len(s.dataids) for s in m.samples) == 6

    def test_no_job_rows_raises(self):
        with pytest.raises(AlignmentError, match="Target search job"):
            parse_alignment_file("*Parameters setting\nJob Name\tX\n")

    def test_mixed_cardinality_is_allowed(self):
        # A run may mix a dual-polarity sample (a01) with a single-polarity one
        # (a02); sample counts need not be uniform across a condition.
        rows = [
            ("a01n.raw", "s1-1", "A"),
            ("a01p.raw", "s1-2", "A"),
            ("a02n.raw", "s1-3", "A"),   # lone single-polarity sample
        ]
        m = parse_alignment_file(_alignment_text(rows))
        assert [len(s.dataids) for s in m.samples] == [2, 1]
        assert m.samples_per_condition == [2]

    def test_duplicate_polarity_marker_raises(self):
        rows = [
            ("s01n.raw", "s1-1", "A"),
            ("s01_n.raw", "s1-2", "A"),   # same base, differing name, same marker
        ]
        with pytest.raises(AlignmentError, match="not.*distinct"):
            parse_alignment_file(_alignment_text(rows))

    def test_comma_delimited_alignment_parses(self):
        # LipidSearch also exports the Alignment Setting file comma-delimited
        # (single-polarity condition-grouped runs). Regression for the
        # "No '*Target search job' rows found" error on such files.
        head = (
            "*Parameters setting,,,\n"
            "Job Name,TestJob,,\n"
            "NormalizeType,NONE,,\n"
            ",,,\n"
            "*Target search job,,,\n"
        )
        body = "".join(
            f"TestJob,{fname},{dataid},{cond},uuid-{i}\n"
            for i, (fname, dataid, cond) in enumerate([
                ("A_01.raw", "s1-1", "Ctrl"),
                ("A_02.raw", "s1-2", "Ctrl"),
                ("B_01.raw", "s2-1", "Trt"),
            ])
        )
        # Trailing padded-empty rows, as real exports contain.
        tail = ",,,\n,,,\n"
        m = parse_alignment_file(head + body + tail)
        assert m.conditions == ["Ctrl", "Trt"]
        assert m.samples_per_condition == [2, 1]
        assert [len(s.dataids) for s in m.samples] == [1, 1, 1]

    def test_same_raw_file_multiple_jobs_group_into_one_sample(self):
        # One raw file searched by two jobs (two complementary polarity product
        # searches) is stored under the same filename with two per-file tokens;
        # they must merge into a single biological sample.
        rows = [
            ("ID_01.raw", "s1-1", "ID"),   # main polarity job
            ("ID_01.raw", "s1-2", "ID"),   # other polarity job, same raw file
            ("Blank_01.raw", "s2-1", "Blank"),
        ]
        m = parse_alignment_file(_alignment_text(rows))
        assert [len(s.dataids) for s in m.samples] == [2, 1]
        assert m.samples[0].dataids == ["s1-1", "s1-2"]
        assert m.samples_per_condition == [1, 1]


def _lipidmol_df():
    """Synthetic LipidMol frame with per-file OriginalArea columns matching _ROWS."""
    data = {'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'FA(15:0)']}
    # s1-1 (neg) detects rows 0,2; s1-2 (pos) detects row 1 — complementary.
    vals = {
        'OriginalArea[s1-1]': [100.0, np.nan, 300.0],
        'OriginalArea[s1-2]': [np.nan, 200.0, np.nan],
        'OriginalArea[s1-3]': [110.0, np.nan, 330.0],
        'OriginalArea[s1-4]': [np.nan, 220.0, np.nan],
        'OriginalArea[s2-1]': [np.nan, np.nan, np.nan],   # all-missing pair member
        'OriginalArea[s2-2]': [np.nan, np.nan, np.nan],
    }
    data.update(vals)
    return pd.DataFrame(data)


class TestMergeDualPolarity:
    """Merging paired per-file columns into flat intensity[s1..sN]."""

    def test_produces_flat_intensity_columns(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        out = merge_dual_polarity(_lipidmol_df(), m)
        icols = [c for c in out.columns if c.startswith('intensity[')]
        assert icols == ['intensity[s1]', 'intensity[s2]', 'intensity[s3]']

    def test_sum_equals_coalesce_for_complementary_pair(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        out = merge_dual_polarity(_lipidmol_df(), m)
        # s1 = OriginalArea[s1-1] + [s1-2]; each lipid seen in one polarity only.
        assert out['intensity[s1]'].tolist() == [100.0, 200.0, 300.0]
        assert out['intensity[s2]'].tolist() == [110.0, 220.0, 330.0]

    def test_all_missing_pair_stays_nan(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        out = merge_dual_polarity(_lipidmol_df(), m)
        # s3 pair (s2-1, s2-2) is entirely NaN -> merged stays NaN, not 0.
        assert out['intensity[s3]'].isna().all()

    def test_missing_column_raises(self):
        m = parse_alignment_file(_alignment_text(_ROWS))
        df = _lipidmol_df().drop(columns=['OriginalArea[s2-2]'])
        with pytest.raises(AlignmentError, match="not present in the data"):
            merge_dual_polarity(df, m)

    def test_mixed_single_and_dual_polarity_merge(self):
        # A run mixing a same-raw dual-job sample (ID) with a single-polarity
        # sample (Blank): the ID pair coalesces, the single column passes
        # through unchanged (NaN preserved).
        rows = [
            ("ID_01.raw", "s1-1", "ID"),
            ("ID_01.raw", "s1-2", "ID"),
            ("Blank_01.raw", "s2-1", "Blank"),
        ]
        m = parse_alignment_file(_alignment_text(rows))
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'FA(15:0)'],
            'OriginalArea[s1-1]': [100.0, np.nan, 300.0],   # main polarity
            'OriginalArea[s1-2]': [np.nan, 200.0, np.nan],  # other polarity
            'OriginalArea[s2-1]': [10.0, np.nan, 30.0],     # single polarity
        })
        out = merge_dual_polarity(df, m)
        assert out['intensity[s1]'].tolist() == [100.0, 200.0, 300.0]
        # Single-polarity sample passes through, NaN preserved (not zero-filled).
        assert out['intensity[s2]'].fillna(-1).tolist() == [10.0, -1.0, 30.0]


class TestHasGroupedPerFileColumns:
    """Detecting the condition-grouped dual-polarity layout."""

    def test_true_for_per_file_columns(self):
        assert has_grouped_per_file_columns(_lipidmol_df()) is True

    def test_false_for_flat_layout(self):
        flat = pd.DataFrame({'LipidMolec': ['PC'], 'MeanArea[s1]': [1.0]})
        assert has_grouped_per_file_columns(flat) is False


def _full_lipidmol_df():
    """LipidMol frame with the required metadata columns + per-file areas."""
    base = _lipidmol_df()
    base.insert(1, 'ClassKey', ['PC', 'PE', 'FA'])
    base.insert(2, 'CalcMass', [757.5, 767.5, 242.2])
    base.insert(3, 'BaseRt', [10.0, 12.0, 3.0])
    base.insert(4, 'TotalGrade', ['A', 'B', 'A'])
    base.insert(5, 'TotalSmpIDRate(%)', [100.0, 90.0, 100.0])
    base.insert(6, 'FAKey', ['(16:0_18:1)', '(18:0_20:4)', '(15:0)'])
    return base


class TestStandardizeWithAlignment:
    """DataStandardizationService.standardize_lipidsearch_with_alignment."""

    def test_merges_and_reports_experiment_layout(self):
        result = DataStandardizationService.standardize_lipidsearch_with_alignment(
            _full_lipidmol_df(), _alignment_text(_ROWS),
        )
        assert result.success is True
        assert result.n_intensity_cols == 3
        assert result.lipidsearch_conditions == ['Control', 'Treated']
        assert result.lipidsearch_samples_per_condition == [2, 1]
        icols = [c for c in result.standardized_df.columns if c.startswith('intensity[')]
        assert icols == ['intensity[s1]', 'intensity[s2]', 'intensity[s3]']

    def test_missing_required_columns_fails(self):
        df = _full_lipidmol_df().drop(columns=['ClassKey'])
        result = DataStandardizationService.standardize_lipidsearch_with_alignment(
            df, _alignment_text(_ROWS),
        )
        assert result.success is False
        assert 'ClassKey' in result.message

    def test_bad_alignment_fails_gracefully(self):
        result = DataStandardizationService.standardize_lipidsearch_with_alignment(
            _full_lipidmol_df(), "not an alignment file",
        )
        assert result.success is False
        assert 'Target search job' in result.message


class TestDualPolarityEndToEnd:
    """Merge -> clean: the standardized dual-polarity frame flows through the
    real LipidSearch cleaner and yields the expected per-sample layout."""

    def test_merge_then_clean(self):
        from app.services.data_cleaning import DataCleaningService
        from app.services.format_detection import DataFormat
        from app.models.experiment import ExperimentConfig

        result = DataStandardizationService.standardize_lipidsearch_with_alignment(
            _full_lipidmol_df(), _alignment_text(_ROWS),
        )
        exp = ExperimentConfig(
            n_conditions=len(result.lipidsearch_conditions),
            conditions_list=result.lipidsearch_conditions,
            number_of_samples_list=result.lipidsearch_samples_per_condition,
        )
        cleaned = DataCleaningService.clean_data(
            result.standardized_df, exp, DataFormat.LIPIDSEARCH,
        ).cleaned_df

        # 3 merged samples (Control x2, Treated x1); cleaner projects to the
        # canonical metadata + intensity schema and drops per-file columns.
        assert [c for c in cleaned.columns if c.startswith('intensity[')] == [
            'intensity[s1]', 'intensity[s2]', 'intensity[s3]'
        ]
        assert not any(c.startswith('OriginalArea[') for c in cleaned.columns)
        assert {'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt', 'FAKey',
                'TotalGrade'}.issubset(cleaned.columns)
        assert len(cleaned) > 0
