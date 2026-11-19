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


def test_normalization_inputs_and_units_are_current():
    """2026-10-30: standards are entered as pmol spiked per sample (or from a
    premade mix) and protein as µg per sample. The AI still described the old
    stock-concentration and BCA-concentration inputs."""
    from app.services.ai_chat.prompt_examples import get_lc_questions_examples
    knowledge = get_app_knowledge()
    assert "spiked into each sample, in pmol" in knowledge
    assert "SPLASH LIPIDOMIX" in knowledge
    assert "`Protein_ug`" in knowledge
    assert "intensity per µg protein" in knowledge
    examples = get_lc_questions_examples()
    assert "IS_concentration" not in examples
    assert "protein concentration (BCA assay)" not in examples
    assert "IS_pmol, giving pmol per sample" in examples


def test_answer_writer_does_not_call_every_value_a_concentration():
    """Power-user feedback 2026-10-27: don't say concentration unless the data
    were properly normalized."""
    from app.services.ai_chat.prompt_templates import (
        answer_from_results_system_prompt,
    )
    assert "call the values abundance or response" in (
        answer_from_results_system_prompt()
    )


def test_figure_labels_follow_the_values_units():
    """2026-11-10: the answer said raw units but the code titled the y-axis
    'Mean concentration'; the units rule was only in the answer writer."""
    from app.services.ai_chat.prompt_templates import data_questions_system_prompt
    assert "titles, axis labels and legends" in data_questions_system_prompt()


def test_bqc_samples_are_left_out_of_summaries():
    """2026-11-10: a top-abundance mean silently included the BQC samples."""
    from app.services.ai_chat.prompt_templates import (
        answer_from_results_system_prompt, data_questions_system_prompt,
    )
    assert "Leave BQC samples out" in data_questions_system_prompt()
    assert "BQC samples out of a mean" in answer_from_results_system_prompt()


def test_suspect_annotations_are_framed_calmly():
    """Power-user feedback 2026-10-27: misannotation answers would alarm a
    general user; much of it is LipidSearch's annotation."""
    from app.services.ai_chat.prompt_templates import (
        answer_from_results_system_prompt, data_questions_system_prompt,
    )
    for prompt in (data_questions_system_prompt(),
                   answer_from_results_system_prompt()):
        assert "identification software's annotation" in prompt
