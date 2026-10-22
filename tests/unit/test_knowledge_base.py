"""Tests for the knowledge base text embedded in the AI prompts."""

from app.services.ai_chat.knowledge_base import get_app_knowledge


def test_wrapped_lines_reach_the_prompt_joined():
    """Regression: the text is a raw string, so the backslashes that wrap its
    long lines reached the prompt literally ("Walther \\" + newline)."""
    text = get_app_knowledge()
    assert '\\\n' not in text
    assert 'Farese & Walther Lab at Memorial Sloan Kettering' in text
    assert text.startswith('## About LipidCruncher')


def test_regex_examples_keep_their_backslashes():
    text = get_app_knowledge()
    assert r'`\[d\d+\]`' in text
    assert r'`\(IS\)`' in text


def test_lipidsearch_adduct_rule_is_described():
    """M10 (2026-10-22): the app keeps one row per lipid, in its class's
    internal-standard adduct; the LipidCruncher Questions tab must say so,
    not the old best-grade-only rule."""
    knowledge = get_app_knowledge()
    assert "quantified in the **standard's adduct**" in knowledge
    assert "never sums them" in knowledge
    assert "Configure Adducts" in knowledge
