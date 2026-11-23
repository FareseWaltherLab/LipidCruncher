"""Generating IsoCorrectoR's Molecule and Element files from the Measurement
file (app.services.isotope_correction.input_builder)."""
from pathlib import Path

import pandas as pd
import pytest

from app.services.isotope_correction.input_builder import (
    build_input_files,
    default_tracer,
    lipid_formula,
    molecule_table,
    parse_formula,
    untagged_ids,
    untagged_message,
)

SAMPLE_DIR = Path(__file__).parents[2] / "sample_datasets" / "isotope_tracing"


def _measurement(*ids):
    return pd.DataFrame({"Measurements/Samples": list(ids), "s1": [1.0] * len(ids)})


def _table(**overrides):
    row = {"Molecule": "BMP_22:6_22:6", "Formula": "C50H75O10P",
           "Ion": "[M-H]-", "Labelled atoms": 6}
    row.update(overrides)
    return pd.DataFrame([row])


class TestLipidFormula:
    """Neutral formulas, each checked against the lipid's known formula."""

    @pytest.mark.parametrize("name, formula", [
        ("BMP_22:6_22:6", "C50H75O10P"),
        ("BMP_18:1_18:1", "C42H79O10P"),
        ("PC_34:1", "C42H82NO8P"),
        ("PC(16:0_18:1)", "C42H82NO8P"),
        ("PC 16:0/18:1", "C42H82NO8P"),
        ("PE_34:1", "C39H76NO8P"),
        ("PS_34:1", "C40H76NO10P"),
        ("PG_34:1", "C40H77O10P"),
        ("PI_34:1", "C43H81O13P"),
        ("PA_34:1", "C37H71O8P"),
        ("LPC_16:0", "C24H50NO7P"),
        ("LPE_18:1", "C23H46NO7P"),
        ("CL_72:8", "C81H142O17P2"),
        ("TG_16:0_18:1_18:1", "C55H102O6"),
        ("DG_34:1", "C37H70O5"),
        ("CE_18:1", "C45H78O2"),
        ("ChE_18:1", "C45H78O2"),
        ("FA_16:0", "C16H32O2"),
        ("Cer_d18:1_16:0", "C34H67NO3"),
        ("SM_d34:1", "C39H79N2O6P"),
        ("Hex1Cer_d18:1_24:1", "C48H91NO8"),
        ("PC_O-34:1", "C42H84NO7P"),
        ("PE_P-18:0_22:6", "C45H78NO7P"),
    ])
    def test_known_lipids(self, name, formula):
        assert lipid_formula(name) == formula

    @pytest.mark.parametrize("name", [
        "XYZ_16:0",            # unknown class
        "PC_16:0_18:1_18:1",   # three chains on a two-chain class
        "PC_34:1;O",           # modification
        "PG_d34:1",            # sphingoid prefix on a glycerolipid
        "PC",                  # no chains
    ])
    def test_unknown_forms_are_left_for_the_user(self, name):
        assert lipid_formula(name) is None


def test_parse_formula():
    assert parse_formula("C42H82NO8P") == {"C": 42, "H": 82, "N": 1, "O": 8, "P": 1}
    assert parse_formula("C42H82 N O8P") == {"C": 42, "H": 82, "N": 1, "O": 8, "P": 1}
    assert parse_formula("C42-H82") is None
    assert parse_formula("") is None


class TestMoleculeTable:

    def test_one_row_per_lipid_with_suggestions(self):
        table = molecule_table(pd.read_csv(SAMPLE_DIR / "MeasurementFile.csv"))
        assert table["Molecule"].tolist() == ["BMP_18:1_18:1", "PC_34:1"]
        assert table["Formula"].tolist() == ["C42H79O10P", "C42H82NO8P"]
        assert table["Ion"].tolist() == ["[M-H]-", "[M-H]-"]
        # The heaviest isotopologue measured is _C6.
        assert table["Labelled atoms"].tolist() == [6, 6]

    def test_standard_resolution_ids(self):
        table = molecule_table(_measurement(*(f"BMP_22:6_22:6_{i}" for i in range(4))))
        assert table["Molecule"].tolist() == ["BMP_22:6_22:6"]
        assert table["Labelled atoms"].tolist() == [3]

    def test_unknown_class_has_a_blank_formula(self):
        table = molecule_table(_measurement("Mystery_1:0_0", "Mystery_1:0_1"))
        assert table["Formula"].tolist() == [""]

    def test_ids_without_an_isotopologue_tag_are_flagged(self):
        # A bare lipid name was once read as BMP_22:6 with 226 labelled atoms.
        assert untagged_ids(_measurement("BMP_22:6_22:6", "BMP_22:6_22:6")) == [
            "BMP_22:6_22:6", "BMP_22:6_22:6"]
        assert untagged_ids(_measurement("PC_34:1 M+1", "PC34:1")) == ["PC_34:1 M+1", "PC34:1"]

    def test_message_lists_each_id_once(self):
        message = untagged_message(["BMP_22:6_22:6"] * 7)
        assert message.startswith("7 rows of the Measurement file have IDs that do not end")
        assert message.count("`BMP_22:6_22:6`") == 1
        assert untagged_message(["PC34:1"]).startswith("1 row of the Measurement file has an ID that does")

    def test_tagged_ids_pass(self):
        assert untagged_ids(_measurement("BMP_22:6_22:6_0", "PC_34:1_C6", "PC_34:1_N12")) == []
        assert untagged_ids(pd.read_csv(SAMPLE_DIR / "MeasurementFile.csv")) == []

    def test_tracer_comes_from_uhr_ids(self):
        assert default_tracer(_measurement("PC_34:1_N0", "PC_34:1_N1")) == "N"
        assert default_tracer(_measurement("PC_34:1_0", "PC_34:1_1")) == "C"


class TestBuildInputFiles:

    def test_full_ion_formula_with_label(self):
        """Regression: a hand-written Molecule file gave BMP 22:6_22:6 as C44
        (the 6 labelled carbons subtracted); IsoCorrectoR needs all 50."""
        molecule_file, _, problems = build_input_files(_table(), "C", 0.99)
        assert problems == []
        assert molecule_file.to_dict("records") == [{
            "Molecule": "BMP_22:6_22:6",
            "MS ion or MS/MS product ion": "C50H74O10P1LabC6",
            "MS/MS neutral loss": "",
        }]

    def test_element_file_matches_isocorrectors_example(self):
        table = molecule_table(pd.read_csv(SAMPLE_DIR / "MeasurementFile.csv"))
        _, element_file, _ = build_input_files(table, "C", 0.99)
        sample = pd.read_csv(SAMPLE_DIR / "ElementFile.csv", encoding="utf-8-sig")
        by_element = lambda df: df.set_index("Element").sort_index()  # noqa: E731
        pd.testing.assert_frame_equal(
            by_element(element_file), by_element(sample), check_dtype=False,
        )

    def test_ion_forms_change_the_formula_and_elements(self):
        molecule_file, element_file, _ = build_input_files(
            _table(Ion="[M+Na]+"), "C", 0.99,
        )
        assert molecule_file.iloc[0, 1] == "C50H75Na1O10P1LabC6"
        assert "Na" in element_file["Element"].tolist()
        molecule_file, _, _ = build_input_files(_table(Ion="[M+HCOO]-"), "C", 0.99)
        assert molecule_file.iloc[0, 1] == "C51H76O12P1LabC6"

    def test_tracer_row_carries_mass_shift_and_purity(self):
        table = _table(Molecule="PC_34:1", Formula="C42H82NO8P", **{"Labelled atoms": 1})
        _, element_file, _ = build_input_files(table, "N", 0.98)
        rows = element_file.set_index("Element")
        assert rows.loc["N", "Tracer isotope mass shift"] == 1
        assert rows.loc["N", "Tracer purity"] == 0.98
        assert pd.isna(rows.loc["C", "Tracer purity"])
        assert pd.isna(rows.loc["C", "Tracer isotope mass shift"])

    @pytest.mark.parametrize("overrides, message", [
        ({"Formula": ""}, "enter its formula"),
        ({"Formula": "not a formula"}, "is not a formula"),
        ({"Formula": "C50H75O10PXe"}, "no natural abundances for Xe"),
        ({"Ion": None}, "choose the ion form"),
        ({"Labelled atoms": None}, "can carry the label"),
        ({"Labelled atoms": 51}, "only 50"),
    ])
    def test_incomplete_rows_produce_no_files(self, overrides, message):
        molecule_file, element_file, problems = build_input_files(
            _table(**overrides), "C", 0.99,
        )
        assert molecule_file is None and element_file is None
        assert len(problems) == 1 and message in problems[0]
