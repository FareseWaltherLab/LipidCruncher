"""Tests for StandardMixService: premade internal standard mix amounts."""

import pytest

from app.services.standard_mixes import (
    STANDARD_MIXES, StandardMixService, _match_key,
)

SPLASH = 'SPLASH LIPIDOMIX (330707)'
EQUI = 'EquiSPLASH (330731)'
SPHINGO = 'SphingoSPLASH I (330734)'
ULTIMATE = 'UltimateSPLASH ONE (330820)'

# SPLASH PC 15:0-18:1(d7): 160 µg/mL, MW 753.11.
SPLASH_PC_PMOL_10UL = 160 * 10 / 753.11 * 1000


def _pmol(standards, volumes):
    return {k: v.pmol for k, v in StandardMixService.amounts(standards, volumes).items()}


class TestPmolPerSample:

    def test_ug_per_ml_times_ul_over_mw(self):
        """µg/mL × µL = ng; ng ÷ g/mol = nmol; × 1000 = pmol."""
        pc = STANDARD_MIXES[SPLASH][0]
        assert StandardMixService.pmol_per_sample(pc, 10) == pytest.approx(2124.5, abs=0.1)

    def test_scales_with_volume(self):
        pc = STANDARD_MIXES[SPLASH][0]
        assert StandardMixService.pmol_per_sample(pc, 2) == pytest.approx(
            StandardMixService.pmol_per_sample(pc, 10) / 5
        )


class TestMixData:

    @pytest.mark.parametrize('mix,count', [
        (SPLASH, 14), (SPHINGO, 6), (ULTIMATE, 69), (EQUI, 13),
    ])
    def test_component_counts(self, mix, count):
        assert len(STANDARD_MIXES[mix]) == count

    @pytest.mark.parametrize('mix', list(STANDARD_MIXES))
    def test_components_have_distinct_match_keys(self, mix):
        """A dataset standard can match at most one component of a mix."""
        keys = [_match_key(c.name, c.lipid_class) for c in STANDARD_MIXES[mix]]
        assert None not in keys
        assert len(set(keys)) == len(keys)

    def test_equisplash_is_equimolar_by_mass(self):
        assert {c.ug_per_ml for c in STANDARD_MIXES[EQUI]} == {100}


class TestMatching:

    @pytest.mark.parametrize('name,cls', [
        ('PC(15:0_18:1)+D7:(s)', 'PC'),     # LipidSearch
        ('PC(18:1_15:0)+D7:(s)', 'PC'),     # LipidSearch, chains swapped
        ('PC(33:1)+D7:(s)', 'PC'),          # LipidSearch sum composition
        ('PC 15:0_18:1(d7)', 'PC'),         # MS-DIAL / Generic
        ('PC 15:0-18:1-d7', 'PC'),
    ])
    def test_format_spellings_match_the_same_component(self, name, cls):
        assert _pmol({name: cls}, {SPLASH: 10}) == {
            name: pytest.approx(SPLASH_PC_PMOL_10UL)
        }

    @pytest.mark.parametrize('name,cls,component', [
        ('SM(d36:2)+D9:(s)', 'SM', 'SM d18:1/18:1(d9)'),
        ('SPLASH 18:1_SM 18:1;O2(d9)', 'SM', 'SM d18:1/18:1(d9)'),
        ('ChE(18:1)+D7:(s)', 'ChE', 'CE 18:1(d7)'),
        ('Cholesterol(d7)', 'Cholesterol', 'Cholesterol(d7)'),
        ('SPLASH 18:1_PI 15:0(d7)', 'PI', 'PI 15:0_18:1(d7)'),
        ('TG(15:0_15:0_18:1)+D7:(s)', 'TG', 'TG 15:0_15:0_18:1(d7)'),
    ])
    def test_class_aliases_and_sphingoid_bases(self, name, cls, component):
        """A sphingoid base like d18:1 or d36:2 is not read as a d-label."""
        amounts = StandardMixService.amounts({name: cls}, {SPLASH: 10})
        assert [c.name for _, c in amounts[name].sources] == [component]

    def test_label_must_match(self):
        """UltimateSPLASH's PC 17:0_16:1(d5) is also PC 33:1, but d5."""
        assert _pmol({'PC 15:0_18:1(d7)': 'PC'}, {ULTIMATE: 10}) == {}
        assert _pmol({'PC 17:0_16:1(d5)': 'PC'}, {ULTIMATE: 10}) == {
            'PC 17:0_16:1(d5)': pytest.approx(100 * 10 / 751.09 * 1000)
        }

    def test_unlabelled_name_not_matched(self):
        assert _pmol({'PC 15:0_18:1': 'PC'}, {SPLASH: 10}) == {}

    def test_standard_in_no_mix_not_matched(self):
        assert _pmol({'SPLASH FA 20:4(d11)': 'FA'}, {SPLASH: 10}) == {}

    def test_hexcer_matches_glccer_by_composition(self):
        amounts = _pmol({'HexCer d18:1/15:0(d7)': 'HexCer'}, {SPHINGO: 5})
        assert amounts == {'HexCer d18:1/15:0(d7)': pytest.approx(6.93 * 5 / 693.06 * 1000)}

    def test_standard_in_two_mixes_gets_the_sum(self):
        """Spiking SPLASH and EquiSPLASH puts both mixes' PC in the sample."""
        amounts = _pmol({'PC 15:0_18:1(d7)': 'PC'}, {SPLASH: 10, EQUI: 5})
        assert amounts['PC 15:0_18:1(d7)'] == pytest.approx(
            SPLASH_PC_PMOL_10UL + 100 * 5 / 753.11 * 1000
        )

    def test_no_mixes_no_amounts(self):
        assert _pmol({'PC 15:0_18:1(d7)': 'PC'}, {}) == {}
