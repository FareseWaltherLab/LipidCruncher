"""
Premade internal standard mixes.

The commercial mixes users spike into every sample, and the amount of each
of their standards that ends up in a sample: vendor stock (µg/mL) × volume
spiked (µL) ÷ molecular weight (g/mol) = nmol. The amounts are matched to
the dataset's standards by class, total chain composition and isotope label,
since each format writes a standard's name its own way
(``PC(15:0_18:1)+D7:(s)``, ``PC 15:0_18:1(d7)``, ``PC(33:1)+D7:(s)``).

Pure logic — no Streamlit dependencies.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class MixComponent:
    """One standard of a mix: stock concentration and molecular weight
    (of the labelled compound, counter-ion included for salts)."""
    name: str
    lipid_class: str
    ug_per_ml: float
    mw: float


@dataclass(frozen=True)
class MixAmount:
    """A dataset standard's amount per sample, summed over the mixes that
    contain it, and the (mix, component) pairs it came from."""
    nmol: float
    sources: Tuple[Tuple[str, MixComponent], ...]


DEFAULT_SPIKE_VOLUME_UL = 10.0


def _c(name: str, lipid_class: str, ug_per_ml: float, mw: float) -> MixComponent:
    return MixComponent(name, lipid_class, ug_per_ml, mw)


# Avanti Research catalog values (product pages, checked 2026-10). Lots vary
# from these targets; a lot's certificate of analysis has the measured values.
STANDARD_MIXES: Dict[str, Tuple[MixComponent, ...]] = {
    'SPLASH LIPIDOMIX (330707)': (
        _c('PC 15:0_18:1(d7)', 'PC', 160, 753.11),
        _c('PE 15:0_18:1(d7)', 'PE', 5, 711.03),
        _c('PS 15:0_18:1(d7)', 'PS', 5, 777.02),
        _c('PG 15:0_18:1(d7)', 'PG', 30, 764.02),
        _c('PI 15:0_18:1(d7)', 'PI', 10, 847.13),
        _c('PA 15:0_18:1(d7)', 'PA', 7, 689.94),
        _c('LPC 18:1(d7)', 'LPC', 25, 528.72),
        _c('LPE 18:1(d7)', 'LPE', 5, 486.64),
        _c('CE 18:1(d7)', 'CE', 350, 658.16),
        _c('MG 18:1(d7)', 'MG', 2, 363.59),
        _c('DG 15:0_18:1(d7)', 'DG', 10, 587.98),
        _c('TG 15:0_15:0_18:1(d7)', 'TG', 55, 812.37),
        _c('SM d18:1/18:1(d9)', 'SM', 30, 738.12),
        _c('Cholesterol(d7)', 'Chol', 100, 393.71),
    ),
    'SphingoSPLASH I (330734)': (
        _c('Cer d18:1/15:0(d7)', 'Cer', 5.31, 530.917),
        _c('GlcCer d18:1/15:0(d7)', 'GlcCer', 6.93, 693.06),
        _c('GalCer d18:1/13:0(d7)', 'GalCer', 6.65, 665.02),
        _c('LacCer d18:1/15:0(d7)', 'LacCer', 8.55, 855.21),
        _c('Cer d18:0/13:0(d7)', 'Cer', 5.05, 504.88),
        _c('SM d18:1/15:0(d9)', 'SM', 6.98, 698.06),
    ),
    'UltimateSPLASH ONE (330820)': (
        _c('TG 14:0_13:0_14:0(d5)', 'TG', 25, 714.18),
        _c('TG 14:0_15:1_14:0(d5)', 'TG', 50, 740.22),
        _c('TG 14:0_17:1_14:0(d5)', 'TG', 75, 768.27),
        _c('TG 16:0_15:1_16:0(d5)', 'TG', 100, 796.33),
        _c('TG 16:0_17:1_16:0(d5)', 'TG', 125, 824.38),
        _c('TG 16:0_19:2_16:0(d5)', 'TG', 100, 850.42),
        _c('TG 18:1_17:1_18:1(d5)', 'TG', 75, 876.46),
        _c('TG 18:1_19:2_18:1(d5)', 'TG', 50, 902.47),
        _c('TG 18:1_21:2_18:1(d5)', 'TG', 25, 930.53),
        _c('CE 14:1(d7)', 'CE', 25, 602.04),
        _c('CE 16:1(d7)', 'CE', 50, 630.09),
        _c('CE 18:1(d7)', 'CE', 75, 658.143),
        _c('CE 20:3(d7)', 'CE', 50, 682.18),
        _c('CE 22:4(d7)', 'CE', 25, 708.2),
        _c('Cer d18:1/16:1(d7)', 'Cer', 75, 542.93),
        _c('Cer d18:1/18:1(d7)', 'Cer', 50, 570.98),
        _c('Cer d18:1/20:1(d7)', 'Cer', 25, 599.03),
        _c('Cer d18:1/22:1(d7)', 'Cer', 50, 627.09),
        _c('Cer d18:1/24:1(d7)', 'Cer', 75, 655.14),
        _c('SM d18:1/16:1(d9)', 'SM', 75, 710.08),
        _c('SM d18:1/18:1(d9)', 'SM', 50, 738.14),
        _c('SM d18:1/20:1(d9)', 'SM', 25, 766.19),
        _c('SM d18:1/22:1(d9)', 'SM', 50, 794.24),
        _c('SM d18:1/24:1(d9)', 'SM', 75, 822.28),
        _c('PC 17:0_14:1(d5)', 'PC', 50, 723.04),
        _c('PC 17:0_16:1(d5)', 'PC', 100, 751.09),
        _c('PC 17:0_18:1(d5)', 'PC', 150, 779.15),
        _c('PC 17:0_20:3(d5)', 'PC', 100, 803.17),
        _c('PC 17:0_22:4(d5)', 'PC', 50, 829.21),
        _c('PE 17:0_14:1(d5)', 'PE', 25, 680.96),
        _c('PE 17:0_16:1(d5)', 'PE', 50, 709.01),
        _c('PE 17:0_18:1(d5)', 'PE', 75, 737.07),
        _c('PE 17:0_20:3(d5)', 'PE', 50, 761.09),
        _c('PE 17:0_22:4(d5)', 'PE', 25, 787.13),
        _c('PG 17:0_14:1(d5)', 'PG', 25, 733.95),
        _c('PG 17:0_16:1(d5)', 'PG', 50, 762.01),
        _c('PG 17:0_18:1(d5)', 'PG', 75, 790.06),
        _c('PG 17:0_20:3(d5)', 'PG', 50, 814.08),
        _c('PG 17:0_22:4(d5)', 'PG', 25, 840.12),
        _c('PS 17:0_14:1(d5)', 'PS', 25, 746.95),
        _c('PS 17:0_16:1(d5)', 'PS', 50, 775.0),
        _c('PS 17:0_18:1(d5)', 'PS', 75, 803.06),
        _c('PS 17:0_20:3(d5)', 'PS', 50, 827.08),
        _c('PS 17:0_22:4(d5)', 'PS', 25, 853.12),
        _c('DG 17:0_14:1(d5)', 'DG', 25, 557.91),
        _c('DG 17:0_16:1(d5)', 'DG', 50, 585.97),
        _c('DG 17:0_18:1(d5)', 'DG', 75, 614.02),
        _c('DG 17:0_20:3(d5)', 'DG', 50, 638.04),
        _c('DG 17:0_22:4(d5)', 'DG', 25, 664.08),
        _c('PI 17:0_14:1(d5)', 'PI', 25, 817.06),
        _c('PI 17:0_16:1(d5)', 'PI', 50, 845.12),
        _c('PI 17:0_18:1(d5)', 'PI', 75, 873.17),
        _c('PI 17:0_20:3(d5)', 'PI', 50, 897.19),
        _c('PI 17:0_22:4(d5)', 'PI', 25, 923.23),
        _c('LPI 15:0(d5)', 'LPI', 25, 580.66),
        _c('LPI 17:0(d5)', 'LPI', 50, 608.72),
        _c('LPI 19:0(d5)', 'LPI', 25, 636.77),
        _c('LPS 15:0(d5)', 'LPS', 25, 510.55),
        _c('LPS 17:0(d5)', 'LPS', 50, 538.61),
        _c('LPS 19:0(d5)', 'LPS', 25, 566.66),
        _c('LPG 15:0(d5)', 'LPG', 25, 497.55),
        _c('LPG 17:0(d5)', 'LPG', 50, 525.61),
        _c('LPG 19:0(d5)', 'LPG', 25, 553.66),
        _c('LPC 15:0(d5)', 'LPC', 25, 486.63),
        _c('LPC 17:0(d5)', 'LPC', 50, 514.7),
        _c('LPC 19:0(d5)', 'LPC', 25, 542.75),
        _c('LPE 15:0(d5)', 'LPE', 25, 444.55),
        _c('LPE 17:0(d5)', 'LPE', 50, 472.61),
        _c('LPE 19:0(d5)', 'LPE', 25, 500.67),
    ),
    'EquiSPLASH (330731)': (
        _c('PC 15:0_18:1(d7)', 'PC', 100, 753.11),
        _c('LPC 18:1(d7)', 'LPC', 100, 528.72),
        _c('PE 15:0_18:1(d7)', 'PE', 100, 711.03),
        _c('LPE 18:1(d7)', 'LPE', 100, 486.64),
        _c('PG 15:0_18:1(d7)', 'PG', 100, 764.02),
        _c('PI 15:0_18:1(d7)', 'PI', 100, 847.13),
        _c('PS 15:0_18:1(d7)', 'PS', 100, 777.02),
        _c('TG 15:0_15:0_18:1(d7)', 'TG', 100, 812.37),
        _c('DG 15:0_18:1(d7)', 'DG', 100, 587.98),
        _c('MG 18:1(d7)', 'MG', 100, 363.59),
        _c('CE 18:1(d7)', 'CE', 100, 658.16),
        _c('SM d18:1/18:1(d9)', 'SM', 100, 738.12),
        _c('Cer d18:1/15:0(d7)', 'Cer', 100, 530.92),
    ),
}

# Class names that differ between formats and the mixes' vendor names.
_CLASS_ALIASES = {
    'CHE': 'CE',
    'CHOLESTEROL': 'CHOL',
    'GLCCER': 'HEXCER', 'GALCER': 'HEXCER', 'HEX1CER': 'HEXCER',
    'HEX2CER': 'LACCER',
    'DHCER': 'CER',
}

# An isotope label: d7 / D7 after "(", "+", "-" or "[", not a sphingoid base
# like (d18:1 or a sum composition like (d36:2.
_LABEL = re.compile(r'[(+\-\[]\s*[dD](\d+)(?!\d|:\d)')
_CHAIN = re.compile(r'(\d+):(\d+)')


def _match_key(name: str, lipid_class: str) -> Optional[tuple]:
    """(class, label, carbons, double bonds); None for an unlabelled name."""
    label = _LABEL.search(name)
    if label is None:
        return None
    chains = _CHAIN.findall(name[:label.start()] + name[label.end():])
    cls = str(lipid_class).upper()
    return (
        _CLASS_ALIASES.get(cls, cls), int(label.group(1)),
        sum(int(c) for c, _ in chains), sum(int(d) for _, d in chains),
    )


class StandardMixService:
    """Amounts of premade-mix standards in each sample."""

    @staticmethod
    def nmol_per_sample(component: MixComponent, volume_ul: float) -> float:
        """µg/mL × µL = ng; ng ÷ (g/mol) = nmol."""
        return component.ug_per_ml * volume_ul / component.mw

    @staticmethod
    def amounts(
        standards: Dict[str, str], volumes: Dict[str, float],
    ) -> Dict[str, MixAmount]:
        """Each dataset standard's amount per sample from the spiked mixes.

        Args:
            standards: Dataset standard name -> its ClassKey.
            volumes: Mix name (a STANDARD_MIXES key) -> µL spiked per sample.

        Returns:
            Standard name -> MixAmount, for the standards found in a mix. A
            standard in several mixes gets the sum: the same compound from
            each was spiked into the sample.
        """
        amounts: Dict[str, MixAmount] = {}
        for name, lipid_class in standards.items():
            key = _match_key(name, lipid_class)
            if key is None:
                continue
            nmol, sources = 0.0, []
            for mix, volume in volumes.items():
                for component in STANDARD_MIXES[mix]:
                    if _match_key(component.name, component.lipid_class) == key:
                        nmol += StandardMixService.nmol_per_sample(component, volume)
                        sources.append((mix, component))
                        break
            if sources:
                amounts[name] = MixAmount(nmol, tuple(sources))
        return amounts
