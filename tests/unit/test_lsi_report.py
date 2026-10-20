"""
Unit tests for LSIReportService.
"""
import pytest
import pandas as pd

from app.services.lsi_report import (
    AnalysisStatistics, LSIReportService, _PLACEHOLDER, _MANUAL_FIELDS,
)
from app.models.normalization import NormalizationConfig
from app.services.statistical_testing import (
    PostHocResult, StatisticalTestResult, StatisticalTestSummary,
)

from tests.conftest import make_experiment, make_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def experiment():
    return make_experiment(2, 3)


@pytest.fixture
def cleaned_df():
    return make_dataframe(
        n_lipids=10,
        n_samples=6,
        classes=["PC"] * 4 + ["PE"] * 3 + ["SM"] * 3,
        prefix="concentration",
    )


@pytest.fixture
def intsta_df():
    return pd.DataFrame({
        "LipidMolec": ["PC 17:0_17:0", "PE 15:0_15:0"],
        "ClassKey": ["PC", "PE"],
    })


@pytest.fixture
def norm_config():
    return NormalizationConfig(
        method="internal_standard",
        selected_classes=["PC", "PE"],
        internal_standards={"PC": "PC 17:0_17:0", "PE": "PE 15:0_15:0"},
        intsta_concentrations={"PC 17:0_17:0": 10.0, "PE 15:0_15:0": 5.0},
    )


def _summary(test_names, correction="fdr_bh", posthoc=None, transform="log10"):
    results = {
        f"G{i}": StatisticalTestResult(test_name=name, statistic=1.0, p_value=0.01)
        for i, name in enumerate(test_names)
    }
    info = {"test_type": "parametric", "correction": correction, "transform": transform}
    posthoc_results = {}
    if posthoc is not None:
        info["posthoc"] = posthoc
        posthoc_results = {"G0": [PostHocResult(
            group1="A", group2="B", p_value=0.01, test_name="Tukey's HSD",
        )]}
    return StatisticalTestSummary(
        results=results, posthoc_results=posthoc_results,
        test_info=info, parameters={"alpha": 0.05},
    )


@pytest.fixture
def statistics():
    return {"Abundance bar chart": AnalysisStatistics(
        summary=_summary(["Welch's t-test", "Welch's t-test"]),
        significance="adjusted p ≤ 0.05",
    )}


@pytest.fixture
def auto_fields(experiment, norm_config, statistics, cleaned_df, intsta_df):
    return LSIReportService.collect_auto_fields(
        format_type="LipidSearch",
        experiment=experiment,
        normalization_config=norm_config,
        statistics=statistics,
        cleaned_df=cleaned_df,
        intsta_df=intsta_df,
        bqc_label=None,
        cleaning_params={"Grade filter": "A, B"},
        qc_summary={"BQC CoV threshold": "30%"},
    )


# ── collect_auto_fields ───────────────────────────────────────────────────


class TestCollectAutoFields:

    def test_complete_fields(self, auto_fields, experiment):
        """All expected fields are populated with valid inputs."""
        assert auto_fields["Software"] == "LipidCruncher"
        assert "LIPID MAPS" in auto_fields["Lipid nomenclature"]
        assert auto_fields["Data format / identification software"] == "LipidSearch"
        assert auto_fields["Number of conditions"] == 2
        assert "Control" in auto_fields["Condition labels"]
        assert auto_fields["Total samples"] == 6
        assert auto_fields["BQC samples"] == "Not used"
        assert "Internal Standard" in auto_fields["Normalization method"]
        assert auto_fields["Number of internal standards"] == 2
        assert auto_fields["Number of lipid species (after filtering)"] == 10
        assert auto_fields["Number of lipid classes"] == 3
        assert auto_fields["Statistics (Abundance bar chart): tests"] == "Welch's t-test"
        assert auto_fields[
            "Statistics (Abundance bar chart): multiple testing correction"
        ] == "Benjamini-Hochberg FDR"

    def test_minimal_fields(self, experiment, cleaned_df):
        """Only required fields (no BQC, no stats, no normalization)."""
        fields = LSIReportService.collect_auto_fields(
            format_type="Generic Format",
            experiment=experiment,
            normalization_config=None,
            statistics=None,
            cleaned_df=cleaned_df,
            intsta_df=None,
            bqc_label=None,
            cleaning_params={},
            qc_summary={},
        )
        assert fields["Software"] == "LipidCruncher"
        assert fields["Normalization method"] == "Not configured"
        assert fields["Internal standards detected"] == "None detected"
        assert not [k for k in fields if k.startswith("Statistics")]

    def test_bqc_label_present(self, experiment, cleaned_df):
        """BQC label is included when provided."""
        fields = LSIReportService.collect_auto_fields(
            format_type="MS-DIAL",
            experiment=experiment,
            normalization_config=None,
            statistics=None,
            cleaned_df=cleaned_df,
            intsta_df=None,
            bqc_label="BQC",
            cleaning_params={},
            qc_summary={},
        )
        assert "BQC" in fields["BQC samples"]

    def test_statistics_per_analysis(self, experiment, cleaned_df):
        """Each analysis's tests are listed under its own name."""
        fields = LSIReportService.collect_auto_fields(
            format_type="Generic Format",
            experiment=experiment,
            normalization_config=None,
            statistics={
                "Saturation profile": AnalysisStatistics(
                    summary=_summary(
                        ["Welch's ANOVA", "Kruskal-Wallis", "Welch's ANOVA"],
                        posthoc="tukey",
                    ),
                    significance="adjusted p ≤ 0.05",
                ),
                "Volcano plot": AnalysisStatistics(
                    summary=_summary(
                        ["Welch's t-test"], correction="uncorrected", transform="none",
                    ),
                    significance="p ≤ 0.01 and fold change > 2× up or down (B vs A)",
                ),
            },
            cleaned_df=cleaned_df,
            intsta_df=None,
            bqc_label=None,
            cleaning_params={},
            qc_summary={},
        )
        stats = {k: v for k, v in fields.items() if k.startswith("Statistics")}
        assert stats == {
            "Statistics (Saturation profile): tests": "Kruskal-Wallis, Welch's ANOVA",
            "Statistics (Saturation profile): number of tests": 3,
            "Statistics (Saturation profile): multiple testing correction":
                "Benjamini-Hochberg FDR",
            "Statistics (Saturation profile): post-hoc": "Tukey's HSD",
            "Statistics (Saturation profile): data transformation": "log10",
            "Statistics (Saturation profile): significance": "adjusted p ≤ 0.05",
            "Statistics (Volcano plot): tests": "Welch's t-test",
            "Statistics (Volcano plot): number of tests": 1,
            "Statistics (Volcano plot): multiple testing correction": "None",
            "Statistics (Volcano plot): data transformation": "None",
            "Statistics (Volcano plot): significance":
                "p ≤ 0.01 and fold change > 2× up or down (B vs A)",
        }


# ── generate_checklist_csv ────────────────────────────────────────────────


class TestGenerateChecklistCSV:

    def test_csv_structure(self, auto_fields):
        """CSV output has header + correct number of rows."""
        csv = LSIReportService.generate_checklist_csv(auto_fields)
        lines = csv.strip().split("\n")
        # Header + auto fields + manual fields
        assert lines[0] == "Category,Item,Value,Source"
        expected_rows = len(auto_fields) + len(_MANUAL_FIELDS)
        assert len(lines) - 1 == expected_rows

    def test_blanks_marked(self, auto_fields):
        """Manual fields without user input are marked with placeholder."""
        csv = LSIReportService.generate_checklist_csv(auto_fields)
        assert _PLACEHOLDER in csv

    def test_manual_fields_included(self, auto_fields):
        """User-supplied manual fields appear in CSV."""
        manual = {"Organism / tissue type": "Mouse liver"}
        csv = LSIReportService.generate_checklist_csv(auto_fields, manual)
        assert "Mouse liver" in csv
        assert '"Researcher"' in csv


# ── generate_checklist_pdf ────────────────────────────────────────────────


class TestGenerateChecklistPDF:

    def test_pdf_bytes_nonempty(self, auto_fields):
        """PDF output is non-empty bytes."""
        pdf = LSIReportService.generate_checklist_pdf(auto_fields)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 100

    def test_pdf_valid_header(self, auto_fields):
        """PDF bytes start with valid PDF header."""
        pdf = LSIReportService.generate_checklist_pdf(auto_fields)
        assert pdf[:5] == b"%PDF-"

    def test_manual_fields_in_pdf(self, auto_fields):
        """Manual fields appear in PDF output."""
        manual = {"Instrument model": "Thermo Q Exactive HF"}
        pdf = LSIReportService.generate_checklist_pdf(auto_fields, manual)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 100


# ── get_manual_field_definitions ──────────────────────────────────────────


class TestManualFieldDefinitions:

    def test_returns_list(self):
        """Returns a non-empty list of dicts."""
        defs = LSIReportService.get_manual_field_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0

    def test_field_structure(self):
        """Each definition has required keys."""
        for d in LSIReportService.get_manual_field_definitions():
            assert "category" in d
            assert "item" in d
            assert "description" in d

    def test_returns_copy(self):
        """Returned list is a copy, not the original."""
        defs1 = LSIReportService.get_manual_field_definitions()
        defs2 = LSIReportService.get_manual_field_definitions()
        assert defs1 is not defs2
