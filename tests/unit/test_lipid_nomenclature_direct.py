"""Unit tests for direct imports from app/lipid_nomenclature.py.

The canonical location for nomenclature functions is lipid_nomenclature.py.
These tests verify the module works standalone (not just via constants.py
re-exports). Existing tests in test_lipid_name_utils.py cover the full
behavioral surface; these focus on import paths and module-level constants.
"""
import pytest

from app.lipid_nomenclature import (
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
# Direct Import Smoke Tests
# =============================================================================


class TestDirectImports:
    """Verify all public names are importable from lipid_nomenclature.py."""

    def test_sphingolipid_classes_is_frozenset(self):
        assert isinstance(SPHINGOLIPID_CLASSES, frozenset)
        assert 'Cer' in SPHINGOLIPID_CLASSES
        assert 'SM' in SPHINGOLIPID_CLASSES

    def test_single_chain_classes_is_frozenset(self):
        assert isinstance(SINGLE_CHAIN_CLASSES, frozenset)
        assert 'LPC' in SINGLE_CHAIN_CLASSES

    def test_lyso_classes_is_frozenset(self):
        assert isinstance(LYSO_CLASSES, frozenset)
        assert 'LPC' in LYSO_CLASSES

    def test_hydroxyl_pattern_is_compiled_regex(self):
        import re
        assert isinstance(HYDROXYL_PATTERN, re.Pattern)


class TestConsistencyWithConstants:
    """Verify direct imports match constants.py re-exports."""

    def test_parse_lipid_name_matches(self):
        from app.constants import parse_lipid_name as from_constants
        assert parse_lipid_name is from_constants

    def test_format_lipid_name_matches(self):
        from app.constants import format_lipid_name as from_constants
        assert format_lipid_name is from_constants

    def test_normalize_hydroxyl_matches(self):
        from app.constants import normalize_hydroxyl as from_constants
        assert normalize_hydroxyl is from_constants

    def test_sort_chains_matches(self):
        from app.constants import sort_chains_lipid_maps as from_constants
        assert sort_chains_lipid_maps is from_constants

    def test_remove_phantom_chains_matches(self):
        from app.constants import remove_phantom_chains as from_constants
        assert remove_phantom_chains is from_constants

    def test_get_chain_separator_matches(self):
        from app.constants import get_chain_separator as from_constants
        assert get_chain_separator is from_constants

    def test_sphingolipid_classes_matches(self):
        from app.constants import SPHINGOLIPID_CLASSES as from_constants
        assert SPHINGOLIPID_CLASSES is from_constants


# =============================================================================
# Core Function Smoke Tests (direct import path)
# =============================================================================


class TestDirectFunctionCalls:
    """Verify functions work when called via the direct import path."""

    def test_parse_standard_lipid(self):
        cls, chains, mod = parse_lipid_name('PC 16:0_18:1')
        assert cls == 'PC'
        assert chains == '16:0_18:1'
        assert mod == ''

    def test_parse_sphingolipid(self):
        cls, chains, mod = parse_lipid_name('Cer 18:1;O2/24:0')
        assert cls == 'Cer'
        assert chains == '18:1;O2/24:0'

    def test_format_standard(self):
        assert format_lipid_name('PC', ['16:0', '18:1']) == 'PC 16:0_18:1'

    def test_format_sphingolipid(self):
        assert format_lipid_name('Cer', ['18:1;O2', '24:0']) == 'Cer 18:1;O2/24:0'

    def test_normalize_hydroxyl_old_to_new(self):
        assert normalize_hydroxyl(';2O') == ';O2'

    def test_sort_chains_ascending(self):
        result = sort_chains_lipid_maps(['18:1', '16:0'], 'PC')
        assert result == ['16:0', '18:1']

    def test_sort_chains_sphingolipid_unchanged(self):
        result = sort_chains_lipid_maps(['18:1;O2', '24:0'], 'Cer')
        assert result == ['18:1;O2', '24:0']

    def test_remove_phantom(self):
        assert remove_phantom_chains(['18:1', '0:0']) == ['18:1']

    def test_get_separator_phospholipid(self):
        assert get_chain_separator('PC') == '_'

    def test_get_separator_sphingolipid(self):
        assert get_chain_separator('Cer') == '/'