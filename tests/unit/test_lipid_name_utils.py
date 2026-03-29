"""Unit tests for LIPID MAPS nomenclature utility functions in constants.py.

Tests cover: parse_lipid_name, format_lipid_name, sort_chains_lipid_maps,
normalize_hydroxyl, remove_phantom_chains, get_chain_separator, and the
related frozenset constants (SPHINGOLIPID_CLASSES, SINGLE_CHAIN_CLASSES,
LYSO_CLASSES).
"""
import pytest
from app.constants import (
    HYDROXYL_PATTERN,
    LYSO_CLASSES,
    SINGLE_CHAIN_CLASSES,
    SPHINGOLIPID_CLASSES,
    format_lipid_name,
    get_chain_separator,
    normalize_hydroxyl,
    parse_lipid_name,
    remove_phantom_chains,
    sort_chains_lipid_maps,
)


# =============================================================================
# Constants validation
# =============================================================================


class TestSphingolipidClasses:
    """Validate SPHINGOLIPID_CLASSES constant."""

    def test_contains_core_sphingolipids(self):
        for cls in ['Cer', 'SM', 'HexCer', 'dhCer', 'LCB', 'SPB']:
            assert cls in SPHINGOLIPID_CLASSES

    def test_contains_glycosphingolipids(self):
        for cls in ['GlcCer', 'GalCer', 'CerG1', 'CerG2', 'CerG3']:
            assert cls in SPHINGOLIPID_CLASSES

    def test_contains_hexcer_variants(self):
        for cls in ['Hex2Cer', 'Hex3Cer']:
            assert cls in SPHINGOLIPID_CLASSES

    def test_contains_other_sphingolipids(self):
        for cls in ['CerP', 'SL']:
            assert cls in SPHINGOLIPID_CLASSES

    def test_does_not_contain_phospholipids(self):
        for cls in ['PC', 'PE', 'PS', 'PI', 'PG', 'PA']:
            assert cls not in SPHINGOLIPID_CLASSES

    def test_does_not_contain_glycerolipids(self):
        for cls in ['TG', 'DG', 'MG', 'MAG', 'DAG', 'TAG']:
            assert cls not in SPHINGOLIPID_CLASSES

    def test_is_frozenset(self):
        assert isinstance(SPHINGOLIPID_CLASSES, frozenset)


class TestSingleChainClasses:
    """Validate SINGLE_CHAIN_CLASSES constant."""

    def test_contains_lysophospholipids(self):
        for cls in ['LPC', 'LPE', 'LPG', 'LPI', 'LPS', 'LPA']:
            assert cls in SINGLE_CHAIN_CLASSES

    def test_contains_cholesteryl_esters(self):
        for cls in ['CE', 'ChE']:
            assert cls in SINGLE_CHAIN_CLASSES

    def test_contains_monoacylglycerols(self):
        for cls in ['MAG', 'MG']:
            assert cls in SINGLE_CHAIN_CLASSES

    def test_contains_free_fatty_acids(self):
        for cls in ['FFA', 'FA']:
            assert cls in SINGLE_CHAIN_CLASSES

    def test_contains_sphingoid_bases(self):
        for cls in ['LCB', 'SPB']:
            assert cls in SINGLE_CHAIN_CLASSES

    def test_does_not_contain_multichain_classes(self):
        for cls in ['PC', 'PE', 'TG', 'DG', 'Cer', 'SM']:
            assert cls not in SINGLE_CHAIN_CLASSES

    def test_is_frozenset(self):
        assert isinstance(SINGLE_CHAIN_CLASSES, frozenset)


class TestLysoClasses:
    """Validate LYSO_CLASSES constant."""

    def test_contains_all_lyso_species(self):
        for cls in ['LPC', 'LPE', 'LPG', 'LPI', 'LPS', 'LPA']:
            assert cls in LYSO_CLASSES

    def test_is_subset_of_single_chain(self):
        assert LYSO_CLASSES.issubset(SINGLE_CHAIN_CLASSES)

    def test_does_not_contain_non_lyso(self):
        for cls in ['PC', 'PE', 'CE', 'MAG', 'FA']:
            assert cls not in LYSO_CLASSES

    def test_is_frozenset(self):
        assert isinstance(LYSO_CLASSES, frozenset)


# =============================================================================
# normalize_hydroxyl
# =============================================================================


class TestNormalizeHydroxyl:
    """Tests for normalize_hydroxyl()."""

    # --- Standard conversions ---

    def test_2O_to_O2(self):
        assert normalize_hydroxyl(';2O') == ';O2'

    def test_3O_to_O3(self):
        assert normalize_hydroxyl(';3O') == ';O3'

    def test_1O_to_O(self):
        assert normalize_hydroxyl(';1O') == ';O'

    def test_4O_to_O4(self):
        assert normalize_hydroxyl(';4O') == ';O4'

    # --- Already-correct format (no change) ---

    def test_O2_unchanged(self):
        assert normalize_hydroxyl(';O2') == ';O2'

    def test_O3_unchanged(self):
        assert normalize_hydroxyl(';O3') == ';O3'

    def test_O_unchanged(self):
        assert normalize_hydroxyl(';O') == ';O'

    # --- In context of full chain strings ---

    def test_in_sphingolipid_chain(self):
        assert normalize_hydroxyl('18:1;2O') == '18:1;O2'

    def test_in_full_lipid_name(self):
        assert normalize_hydroxyl('Cer 18:1;2O/24:0') == 'Cer 18:1;O2/24:0'

    def test_multiple_hydroxyl_groups_in_name(self):
        result = normalize_hydroxyl('Cer 18:1;2O/24:0;1O')
        assert result == 'Cer 18:1;O2/24:0;O'

    def test_mixed_old_and_new_in_name(self):
        result = normalize_hydroxyl('Cer 18:1;O2/24:0;3O')
        assert result == 'Cer 18:1;O2/24:0;O3'

    # --- No hydroxyl present (passthrough) ---

    def test_no_hydroxyl_unchanged(self):
        assert normalize_hydroxyl('PC 16:0_18:1') == 'PC 16:0_18:1'

    def test_empty_string(self):
        assert normalize_hydroxyl('') == ''

    def test_plain_chain(self):
        assert normalize_hydroxyl('18:1') == '18:1'

    # --- Double-digit hydroxyl counts ---

    def test_10O_to_O10(self):
        assert normalize_hydroxyl(';10O') == ';O10'

    def test_12O_to_O12(self):
        assert normalize_hydroxyl(';12O') == ';O12'


class TestNormalizeHydroxylEdgeCases:
    """Edge cases for normalize_hydroxyl()."""

    def test_semicolon_only(self):
        # No digits or O after semicolon — not a hydroxyl pattern
        assert normalize_hydroxyl(';') == ';'

    def test_chain_with_ether_prefix(self):
        assert normalize_hydroxyl('O-18:1;2O') == 'O-18:1;O2'

    def test_chain_with_plasmalogen_prefix(self):
        assert normalize_hydroxyl('P-18:0;3O') == 'P-18:0;O3'

    def test_preserves_surrounding_text(self):
        assert normalize_hydroxyl('prefix;2Osuffix') == 'prefix;O2suffix'


# =============================================================================
# parse_lipid_name
# =============================================================================


class TestParseLipidName:
    """Tests for parse_lipid_name()."""

    # --- Standard phospholipids ---

    def test_two_chain_phospholipid(self):
        assert parse_lipid_name('PC 16:0_18:1') == ('PC', '16:0_18:1', '')

    def test_pe_two_chain(self):
        assert parse_lipid_name('PE 18:0_20:4') == ('PE', '18:0_20:4', '')

    def test_ps_two_chain(self):
        assert parse_lipid_name('PS 18:0_22:6') == ('PS', '18:0_22:6', '')

    # --- Sphingolipids with slash separator ---

    def test_ceramide_slash(self):
        assert parse_lipid_name('Cer 18:1;O2/24:0') == ('Cer', '18:1;O2/24:0', '')

    def test_sm_slash(self):
        assert parse_lipid_name('SM 18:1;O2/16:0') == ('SM', '18:1;O2/16:0', '')

    def test_hexcer(self):
        assert parse_lipid_name('HexCer 18:1;O2/24:0') == ('HexCer', '18:1;O2/24:0', '')

    # --- Triacylglycerols ---

    def test_three_chain_tg(self):
        assert parse_lipid_name('TG 16:0_18:1_18:2') == ('TG', '16:0_18:1_18:2', '')

    # --- Lysophospholipids (single chain) ---

    def test_lpc_single_chain(self):
        assert parse_lipid_name('LPC 18:1') == ('LPC', '18:1', '')

    def test_lpe_single_chain(self):
        assert parse_lipid_name('LPE 20:4') == ('LPE', '20:4', '')

    # --- Internal standards with modification ---

    def test_modification_d7(self):
        assert parse_lipid_name('LPC 18:1(d7)') == ('LPC', '18:1', '(d7)')

    def test_modification_d5(self):
        assert parse_lipid_name('PC 16:0_18:1(d5)') == ('PC', '16:0_18:1', '(d5)')

    def test_modification_d9(self):
        assert parse_lipid_name('Cer 18:1;O2/24:0(d9)') == ('Cer', '18:1;O2/24:0', '(d9)')

    # --- Class-only names (no chains) ---

    def test_class_only_cholesterol(self):
        assert parse_lipid_name('Ch') == ('Ch', '', '')

    def test_class_only_cholesterol_full(self):
        assert parse_lipid_name('Cholesterol') == ('Cholesterol', '', '')

    # --- Ether lipids ---

    def test_ether_pc(self):
        assert parse_lipid_name('PC O-16:0_18:1') == ('PC', 'O-16:0_18:1', '')

    def test_plasmalogen_pe(self):
        assert parse_lipid_name('PE P-18:0_20:4') == ('PE', 'P-18:0_20:4', '')

    # --- Consolidated format ---

    def test_consolidated_single_chain_info(self):
        assert parse_lipid_name('PC 34:1') == ('PC', '34:1', '')

    # --- Cardiolipin (four chains) ---

    def test_four_chain_cardiolipin(self):
        assert parse_lipid_name('CL 16:0_18:1_18:1_18:2') == (
            'CL', '16:0_18:1_18:1_18:2', ''
        )


class TestParseLipidNameEdgeCases:
    """Edge cases and error handling for parse_lipid_name()."""

    def test_empty_string(self):
        assert parse_lipid_name('') == ('', '', '')

    def test_none_input(self):
        assert parse_lipid_name(None) == ('', '', '')

    def test_integer_input(self):
        assert parse_lipid_name(123) == ('', '', '')

    def test_float_input(self):
        assert parse_lipid_name(3.14) == ('', '', '')

    def test_bool_input(self):
        assert parse_lipid_name(True) == ('', '', '')

    def test_list_input(self):
        assert parse_lipid_name(['PC']) == ('', '', '')

    def test_whitespace_only(self):
        # ' ' — space_idx is 0, so it goes to else branch
        result = parse_lipid_name(' ')
        assert result[0] == ' '  # entire string is treated as class

    def test_leading_space(self):
        # ' PC 16:0' — first space at index 0
        result = parse_lipid_name(' PC 16:0')
        assert result[0] == ' PC 16:0'  # space_idx == 0 → no split

    def test_name_without_chains_or_space(self):
        assert parse_lipid_name('Cholesterol') == ('Cholesterol', '', '')

    def test_multiple_modifications_only_last(self):
        # Only trailing (dN) is extracted
        result = parse_lipid_name('PC 16:0(d5)_18:1(d7)')
        assert result[2] == '(d7)'
        assert '(d5)' in result[1]

    def test_non_deuterium_parenthetical(self):
        # (iso) is NOT a (dN) pattern, so no modification extracted
        result = parse_lipid_name('PC 16:0_18:1(iso)')
        assert result[2] == ''
        assert result[1] == '16:0_18:1(iso)'


# =============================================================================
# get_chain_separator
# =============================================================================


class TestGetChainSeparator:
    """Tests for get_chain_separator()."""

    def test_sphingolipid_uses_slash(self):
        for cls in ['Cer', 'SM', 'HexCer', 'dhCer', 'LCB', 'SPB',
                     'GlcCer', 'GalCer', 'CerG1', 'CerG2', 'CerG3',
                     'Hex2Cer', 'Hex3Cer', 'CerP', 'SL']:
            assert get_chain_separator(cls) == '/'

    def test_phospholipid_uses_underscore(self):
        for cls in ['PC', 'PE', 'PS', 'PI', 'PG', 'PA', 'CL']:
            assert get_chain_separator(cls) == '_'

    def test_glycerolipid_uses_underscore(self):
        for cls in ['TG', 'DG', 'MG', 'MAG', 'DAG', 'TAG']:
            assert get_chain_separator(cls) == '_'

    def test_lyso_uses_underscore(self):
        for cls in ['LPC', 'LPE', 'LPG', 'LPI', 'LPS', 'LPA']:
            assert get_chain_separator(cls) == '_'

    def test_cholesteryl_ester_uses_underscore(self):
        for cls in ['CE', 'ChE']:
            assert get_chain_separator(cls) == '_'

    def test_unknown_class_uses_underscore(self):
        assert get_chain_separator('UNKNOWN') == '_'

    def test_empty_class_uses_underscore(self):
        assert get_chain_separator('') == '_'


# =============================================================================
# remove_phantom_chains
# =============================================================================


class TestRemovePhantomChains:
    """Tests for remove_phantom_chains()."""

    def test_single_phantom_removed(self):
        assert remove_phantom_chains(['18:1', '0:0']) == ['18:1']

    def test_phantom_first_position(self):
        assert remove_phantom_chains(['0:0', '18:1']) == ['18:1']

    def test_multiple_phantoms_removed(self):
        assert remove_phantom_chains(['20:0', '0:0', '18:1', '0:0']) == ['20:0', '18:1']

    def test_all_phantoms(self):
        assert remove_phantom_chains(['0:0', '0:0']) == []

    def test_single_phantom_only(self):
        assert remove_phantom_chains(['0:0']) == []

    def test_no_phantoms(self):
        assert remove_phantom_chains(['18:1', '20:4']) == ['18:1', '20:4']

    def test_empty_list(self):
        assert remove_phantom_chains([]) == []

    def test_single_real_chain(self):
        assert remove_phantom_chains(['16:0']) == ['16:0']

    def test_three_chains_one_phantom(self):
        assert remove_phantom_chains(['16:0', '18:1', '0:0']) == ['16:0', '18:1']

    def test_phantom_like_but_not_exact(self):
        # '0:1' and '1:0' are not phantoms
        assert remove_phantom_chains(['0:1', '1:0']) == ['0:1', '1:0']

    def test_chain_with_hydroxyl_not_phantom(self):
        assert remove_phantom_chains(['0:0;O2', '18:1']) == ['0:0;O2', '18:1']

    def test_preserves_order(self):
        chains = ['20:4', '16:0', '18:1']
        assert remove_phantom_chains(chains) == ['20:4', '16:0', '18:1']


# =============================================================================
# sort_chains_lipid_maps
# =============================================================================


class TestSortChainsLipidMaps:
    """Tests for sort_chains_lipid_maps()."""

    # --- Sphingolipids: no reordering ---

    def test_sphingolipid_not_reordered(self):
        chains = ['18:1;O2', '24:0']
        assert sort_chains_lipid_maps(chains, 'Cer') == ['18:1;O2', '24:0']

    def test_sphingolipid_reverse_order_preserved(self):
        chains = ['24:0', '18:1;O2']
        assert sort_chains_lipid_maps(chains, 'Cer') == ['24:0', '18:1;O2']

    def test_sm_not_reordered(self):
        chains = ['18:1;O2', '16:0']
        assert sort_chains_lipid_maps(chains, 'SM') == ['18:1;O2', '16:0']

    def test_hexcer_not_reordered(self):
        chains = ['18:1;O2', '24:1']
        assert sort_chains_lipid_maps(chains, 'HexCer') == ['18:1;O2', '24:1']

    def test_all_sphingolipid_classes_preserve_order(self):
        chains = ['24:0', '18:1']
        for cls in SPHINGOLIPID_CLASSES:
            assert sort_chains_lipid_maps(chains, cls) == ['24:0', '18:1']

    # --- Phospholipids: sort by carbon count, then double bonds ---

    def test_sort_by_carbon_count(self):
        chains = ['18:1', '16:0']
        assert sort_chains_lipid_maps(chains, 'PC') == ['16:0', '18:1']

    def test_sort_by_double_bonds_same_carbon(self):
        chains = ['18:2', '18:0']
        assert sort_chains_lipid_maps(chains, 'PE') == ['18:0', '18:2']

    def test_already_sorted(self):
        chains = ['16:0', '18:1']
        assert sort_chains_lipid_maps(chains, 'PC') == ['16:0', '18:1']

    def test_three_chains_sorted(self):
        chains = ['20:4', '16:0', '18:1']
        assert sort_chains_lipid_maps(chains, 'TG') == ['16:0', '18:1', '20:4']

    def test_four_chains_sorted(self):
        chains = ['18:2', '18:1', '16:0', '18:1']
        assert sort_chains_lipid_maps(chains, 'CL') == ['16:0', '18:1', '18:1', '18:2']

    def test_identical_chains_stable(self):
        chains = ['18:1', '18:1']
        assert sort_chains_lipid_maps(chains, 'PC') == ['18:1', '18:1']

    # --- Ether lipid prefixes ---

    def test_ether_prefix_O(self):
        # O-18:0 → strip O- → 18:0 → (18, 0)
        chains = ['20:4', 'O-18:0']
        assert sort_chains_lipid_maps(chains, 'PE') == ['O-18:0', '20:4']

    def test_plasmalogen_prefix_P(self):
        chains = ['20:4', 'P-18:0']
        assert sort_chains_lipid_maps(chains, 'PE') == ['P-18:0', '20:4']

    # --- Chains with hydroxyl suffixes ---

    def test_hydroxyl_stripped_for_sorting(self):
        # 18:1;O2 → strip ;O2 → 18:1 → (18, 1)
        chains = ['20:4', '18:1;O2']
        assert sort_chains_lipid_maps(chains, 'PC') == ['18:1;O2', '20:4']

    def test_oxidation_suffix_stripped(self):
        chains = ['20:4+O', '16:0']
        assert sort_chains_lipid_maps(chains, 'PC') == ['16:0', '20:4+O']

    # --- Edge cases ---

    def test_empty_list(self):
        assert sort_chains_lipid_maps([], 'PC') == []

    def test_single_chain(self):
        assert sort_chains_lipid_maps(['18:1'], 'LPC') == ['18:1']

    def test_malformed_chain_goes_last(self):
        chains = ['16:0', 'notachain']
        result = sort_chains_lipid_maps(chains, 'PC')
        assert result == ['16:0', 'notachain']

    def test_all_malformed_stable(self):
        chains = ['abc', 'xyz']
        result = sort_chains_lipid_maps(chains, 'PC')
        # Both get (9999, 9999), so stable sort preserves order
        assert result == ['abc', 'xyz']

    def test_unknown_class_still_sorts(self):
        chains = ['20:4', '16:0']
        assert sort_chains_lipid_maps(chains, 'UNKNOWN') == ['16:0', '20:4']

    def test_d_prefix_stripped(self):
        # d18:1 → strip d → 18:1 → (18, 1)
        chains = ['24:0', 'd18:1']
        assert sort_chains_lipid_maps(chains, 'PC') == ['d18:1', '24:0']

    def test_C_prefix_stripped(self):
        chains = ['C24:0', '16:0']
        assert sort_chains_lipid_maps(chains, 'PC') == ['16:0', 'C24:0']


# =============================================================================
# format_lipid_name
# =============================================================================


class TestFormatLipidName:
    """Tests for format_lipid_name()."""

    # --- Phospholipids (underscore separator) ---

    def test_two_chain_phospholipid(self):
        assert format_lipid_name('PC', ['16:0', '18:1']) == 'PC 16:0_18:1'

    def test_pe_two_chain(self):
        assert format_lipid_name('PE', ['18:0', '20:4']) == 'PE 18:0_20:4'

    def test_three_chain_glycerolipid(self):
        assert format_lipid_name('TG', ['16:0', '18:1', '18:2']) == 'TG 16:0_18:1_18:2'

    def test_four_chain_cardiolipin(self):
        result = format_lipid_name('CL', ['16:0', '18:1', '18:1', '18:2'])
        assert result == 'CL 16:0_18:1_18:1_18:2'

    # --- Sphingolipids (slash separator) ---

    def test_ceramide(self):
        assert format_lipid_name('Cer', ['18:1;O2', '24:0']) == 'Cer 18:1;O2/24:0'

    def test_sphingomyelin(self):
        assert format_lipid_name('SM', ['18:1;O2', '16:0']) == 'SM 18:1;O2/16:0'

    def test_hexcer(self):
        assert format_lipid_name('HexCer', ['18:1;O2', '24:0']) == 'HexCer 18:1;O2/24:0'

    def test_dhcer(self):
        assert format_lipid_name('dhCer', ['18:0;O2', '24:0']) == 'dhCer 18:0;O2/24:0'

    # --- Single chain ---

    def test_single_chain_lpc(self):
        assert format_lipid_name('LPC', ['18:1']) == 'LPC 18:1'

    def test_single_chain_ce(self):
        assert format_lipid_name('CE', ['18:2']) == 'CE 18:2'

    def test_single_chain_fa(self):
        assert format_lipid_name('FA', ['16:0']) == 'FA 16:0'

    # --- With modification ---

    def test_with_d7_modification(self):
        assert format_lipid_name('LPC', ['18:1'], '(d7)') == 'LPC 18:1(d7)'

    def test_with_d5_modification(self):
        assert format_lipid_name('PC', ['16:0', '18:1'], '(d5)') == 'PC 16:0_18:1(d5)'

    def test_sphingolipid_with_modification(self):
        result = format_lipid_name('Cer', ['18:1;O2', '24:0'], '(d9)')
        assert result == 'Cer 18:1;O2/24:0(d9)'

    # --- Class-only (no chains) ---

    def test_class_only(self):
        assert format_lipid_name('Ch') == 'Ch'

    def test_class_only_with_modification(self):
        assert format_lipid_name('Ch', modification='(d7)') == 'Ch(d7)'

    def test_class_only_none_chains(self):
        assert format_lipid_name('Cholesterol', None) == 'Cholesterol'

    def test_class_only_empty_list(self):
        assert format_lipid_name('Cholesterol', []) == 'Cholesterol'

    def test_class_only_all_empty_strings(self):
        assert format_lipid_name('PC', ['', '']) == 'PC'


class TestFormatLipidNameEdgeCases:
    """Edge cases for format_lipid_name()."""

    def test_ether_chain(self):
        assert format_lipid_name('PC', ['O-16:0', '18:1']) == 'PC O-16:0_18:1'

    def test_plasmalogen_chain(self):
        assert format_lipid_name('PE', ['P-18:0', '20:4']) == 'PE P-18:0_20:4'

    def test_empty_modification_string(self):
        assert format_lipid_name('PC', ['16:0', '18:1'], '') == 'PC 16:0_18:1'

    def test_single_empty_chain_in_list(self):
        # Single empty string chain → treated as all empty
        assert format_lipid_name('PC', ['']) == 'PC'


# =============================================================================
# Round-trip: parse → format
# =============================================================================


class TestParseFormatRoundTrip:
    """Verify parse_lipid_name → format_lipid_name round-trips correctly."""

    @pytest.mark.parametrize('name', [
        'PC 16:0_18:1',
        'PE 18:0_20:4',
        'TG 16:0_18:1_18:2',
        'LPC 18:1',
        'CE 18:2',
        'Ch',
    ])
    def test_phospholipid_roundtrip(self, name):
        cls, chains, mod = parse_lipid_name(name)
        chain_list = chains.split('_') if chains else None
        assert format_lipid_name(cls, chain_list, mod) == name

    @pytest.mark.parametrize('name', [
        'Cer 18:1;O2/24:0',
        'SM 18:1;O2/16:0',
        'HexCer 18:1;O2/24:0',
    ])
    def test_sphingolipid_roundtrip(self, name):
        cls, chains, mod = parse_lipid_name(name)
        chain_list = chains.split('/') if chains else None
        assert format_lipid_name(cls, chain_list, mod) == name

    @pytest.mark.parametrize('name', [
        'LPC 18:1(d7)',
        'PC 16:0_18:1(d5)',
    ])
    def test_modification_roundtrip(self, name):
        cls, chains, mod = parse_lipid_name(name)
        sep = get_chain_separator(cls)
        chain_list = chains.split(sep) if chains else None
        assert format_lipid_name(cls, chain_list, mod) == name


# =============================================================================
# HYDROXYL_PATTERN constant
# =============================================================================


class TestHydroxylPattern:
    """Tests for the HYDROXYL_PATTERN regex."""

    @pytest.mark.parametrize('text,expected', [
        ('18:1;O2', ';O2'),
        ('18:0;O3', ';O3'),
        ('18:1;O', ';O'),
        ('18:1;2O', ';2O'),
        ('18:1;3O', ';3O'),
    ])
    def test_matches_hydroxyl(self, text, expected):
        match = HYDROXYL_PATTERN.search(text)
        assert match is not None
        assert match.group() == expected

    def test_no_match_plain_chain(self):
        assert HYDROXYL_PATTERN.search('18:1') is None

    def test_no_match_no_semicolon(self):
        assert HYDROXYL_PATTERN.search('18:1O2') is None
