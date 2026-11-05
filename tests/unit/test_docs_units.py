"""The docs' worked examples use the units the app now reports."""
from pathlib import Path

from app.ui.content import analysis_docs

ROOT = Path(__file__).resolve().parents[2]


def test_saturation_examples_are_not_in_micromolar():
    """Since 2026-10-30 internal standards give pmol per sample; the
    saturation examples in the supplementary methods and the in-app help
    still spoke of µM, which the AI Studio also reads."""
    methods = (ROOT / "docs" / "supplementary_methods.md").read_text()
    assert "100 µM" not in methods
    assert "PC 16:0_18:1 at 100 pmol" in methods
    assert "of known concentration" not in methods
    assert not any("µM" in v for v in vars(analysis_docs).values()
                   if isinstance(v, str))
