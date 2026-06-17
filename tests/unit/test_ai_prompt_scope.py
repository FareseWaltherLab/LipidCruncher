"""
Unit tests for AI prompt scope enforcement.

Verifies that each tab's system prompt contains a SCOPE section
that instructs the AI to refuse out-of-scope requests and redirect
users to the correct tab.
"""
import pytest

from app.services.ai_chat.prompt_templates import (
    data_questions_system_prompt,
    data_transform_system_prompt,
    lipidcruncher_questions_system_prompt,
)
from app.services.ai_chat.viz_generator import _viz_system_prompt


class TestPromptScopeEnforcement:

    def test_transform_prompt_has_scope_section(self):
        # The Transform prompt intentionally states its scope without the
        # "STRICTLY ENFORCED" wording (it allows info-gathering turns), so
        # assert the section header exists rather than a specific phrase.
        prompt = data_transform_system_prompt()
        assert "## SCOPE" in prompt

    def test_transform_prompt_redirects_to_viz(self):
        prompt = data_transform_system_prompt()
        assert "Custom Visualization" in prompt

    def test_transform_prompt_redirects_to_data_questions(self):
        prompt = data_transform_system_prompt()
        assert "Data Questions" in prompt

    def test_data_questions_prompt_has_scope_section(self):
        prompt = data_questions_system_prompt()
        assert "SCOPE" in prompt
        assert "STRICTLY ENFORCED" in prompt

    def test_data_questions_prompt_redirects_to_transform(self):
        prompt = data_questions_system_prompt()
        assert "Transform to Generic" in prompt

    def test_data_questions_prompt_redirects_to_viz(self):
        prompt = data_questions_system_prompt()
        assert "Custom Visualization" in prompt

    def test_lc_questions_prompt_has_scope_section(self):
        prompt = lipidcruncher_questions_system_prompt()
        assert "SCOPE" in prompt
        assert "STRICTLY ENFORCED" in prompt

    def test_lc_questions_prompt_redirects_to_other_tabs(self):
        prompt = lipidcruncher_questions_system_prompt()
        assert "Data Questions" in prompt
        assert "Transform to Generic" in prompt
        assert "Custom Visualization" in prompt

    def test_viz_prompt_has_scope_section(self):
        prompt = _viz_system_prompt()
        assert "SCOPE" in prompt
        assert "STRICTLY ENFORCED" in prompt

    def test_viz_prompt_redirects_to_data_questions(self):
        prompt = _viz_system_prompt()
        assert "Data Questions" in prompt

    def test_viz_prompt_redirects_to_transform(self):
        prompt = _viz_system_prompt()
        assert "Data Transform" in prompt


class TestPromptExamplesIncluded:

    def test_transform_prompt_has_examples(self):
        prompt = data_transform_system_prompt()
        assert "Few-Shot Examples" in prompt
        assert "Example 1" in prompt
        assert "Example 8" in prompt

    def test_data_questions_prompt_has_examples(self):
        prompt = data_questions_system_prompt()
        assert "Few-Shot Examples" in prompt
        assert "Example 1" in prompt
        assert "Example 8" in prompt

    def test_lc_questions_prompt_has_examples(self):
        prompt = lipidcruncher_questions_system_prompt()
        assert "Few-Shot Examples" in prompt
        assert "Example 1" in prompt
        assert "Example 8" in prompt

    def test_viz_prompt_has_examples(self):
        prompt = _viz_system_prompt()
        assert "Few-Shot Examples" in prompt
        assert "Example 1" in prompt
        assert "Example 8" in prompt
