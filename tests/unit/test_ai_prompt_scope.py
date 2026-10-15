"""
Unit tests for AI prompt scope enforcement.

Verifies that each tab's system prompt contains a SCOPE section
that instructs the AI to refuse out-of-scope requests and redirect
users to the correct tab.
"""
import re

import pytest

from app.models.chat import AIConfig
from app.models.experiment import ExperimentConfig
from app.services.ai_chat import data_transformer as dt
from app.services.ai_chat import viz_generator as viz
from app.services.ai_chat.ai_client import AIClientService
from app.services.ai_chat.prompt_templates import (
    build_transform_context,
    data_questions_system_prompt,
    data_transform_system_prompt,
    lipidcruncher_questions_system_prompt,
)
from app.services.ai_chat.viz_generator import _viz_system_prompt
from tests.ai_turns import canned_stream


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
        assert "Transform to Generic" in prompt


ALL_PROMPTS = [
    data_transform_system_prompt, data_questions_system_prompt,
    lipidcruncher_questions_system_prompt, _viz_system_prompt,
]
AI_STUDIO_TABS = {  # the labels ai_studio_page passes to st.tabs
    "LipidCruncher Questions", "Data Questions", "Transform to Generic",
    "Custom Visualization",
}


@pytest.mark.parametrize("prompt_fn", ALL_PROMPTS)
def test_prompts_name_only_real_tabs(prompt_fn):
    """AQ14: two prompts sent users to a "Data Transform" tab, which does
    not exist (it is "Transform to Generic")."""
    named = set(re.findall(r"\*\*([^*]+)\*\* tab", prompt_fn()))
    assert named and named <= AI_STUDIO_TABS


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


class TestPromptExamplesXmlWrapped:
    """Few-shot examples must be wrapped in <examples> tags so the model
    can distinguish them from instructions (Anthropic prompt guidance)."""

    @pytest.mark.parametrize("prompt_fn", [
        data_transform_system_prompt,
        data_questions_system_prompt,
        lipidcruncher_questions_system_prompt,
        _viz_system_prompt,
    ])
    def test_examples_block_is_xml_tagged(self, prompt_fn):
        prompt = prompt_fn()
        assert "<examples>" in prompt
        assert "</examples>" in prompt


class TestDataContextStructure:
    """The data schema appended to each user message is XML-tagged and placed
    above the user's question (Anthropic long-context guidance)."""

    def test_transform_context_is_xml_tagged(self):
        ctx = build_transform_context("SRC_MARKER", "TGT_MARKER")
        assert "<source_data_schema>" in ctx and "</source_data_schema>" in ctx
        assert "<target_format>" in ctx and "</target_format>" in ctx
        assert "SRC_MARKER" in ctx and "TGT_MARKER" in ctx

    def test_transform_message_puts_schema_before_question(self, monkeypatch):
        stream = canned_stream("ok\n```python\nresult_df = df\n```")
        monkeypatch.setattr(
            AIClientService, "stream_tool_loop", staticmethod(stream)
        )
        list(dt.DataTransformerService.stream_transformation_code(
            source_schema="SRC_SCHEMA_MARKER",
            user_message="USER_QUESTION_MARKER",
            history=[], config=AIConfig(), holder={"full": ""},
        ))
        content = stream.calls[-1]["messages"][-1]["content"]
        assert "<source_data_schema>" in content
        assert content.index("SRC_SCHEMA_MARKER") < content.index("USER_QUESTION_MARKER")

    def test_viz_message_puts_schema_before_question(self, monkeypatch):
        stream = canned_stream("ok\n```python\nfig = None\n```")
        monkeypatch.setattr(
            AIClientService, "stream_tool_loop", staticmethod(stream)
        )
        experiment = ExperimentConfig(
            conditions_list=["Control", "Treatment"],
            number_of_samples_list=[2, 2],
        )
        list(viz.VizGeneratorService.stream_viz_code(
            user_request="USER_REQUEST_MARKER",
            schema_context="SCHEMA_MARKER",
            experiment=experiment,
            history=[], config=AIConfig(), holder={"full": ""},
        ))
        content = stream.calls[-1]["messages"][-1]["content"]
        assert "<dataframe_schema>" in content
        assert content.index("SCHEMA_MARKER") < content.index("USER_REQUEST_MARKER")


class TestSystemPromptCaching:
    """The static system prefix is sent as a cache-enabled content block."""

    def test_system_blocks_marks_ephemeral_cache(self):
        blocks = AIClientService._system_blocks("SYSTEM_TEXT")
        assert blocks == [{
            "type": "text",
            "text": "SYSTEM_TEXT",
            "cache_control": {"type": "ephemeral"},
        }]
