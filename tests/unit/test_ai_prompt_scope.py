"""
Unit tests for AI prompt scope enforcement.

Verifies that each tab's system prompt contains a SCOPE section
that instructs the AI to refuse out-of-scope requests and redirect
users to the correct tab.
"""
import re

import pytest

import pandas as pd

from app.models.chat import AIConfig
from app.services.ai_chat import data_transformer as dt
from app.services.ai_chat.ai_client import AIClientService
from app.services.ai_chat.prompt_templates import (
    build_transform_context,
    data_questions_system_prompt,
    data_transform_system_prompt,
    lipidcruncher_questions_system_prompt,
)
from app.workflows.ai_chat import AIChatWorkflow
from tests.ai_turns import canned_stream


class TestPromptScopeEnforcement:

    def test_transform_prompt_has_scope_section(self):
        # The Transform prompt intentionally states its scope without the
        # "STRICTLY ENFORCED" wording (it allows info-gathering turns), so
        # assert the section header exists rather than a specific phrase.
        prompt = data_transform_system_prompt()
        assert "## SCOPE" in prompt

    def test_transform_prompt_redirects_to_ask_your_data(self):
        prompt = data_transform_system_prompt()
        assert "Ask Your Data" in prompt

    def test_data_questions_prompt_has_scope_section(self):
        prompt = data_questions_system_prompt()
        assert "SCOPE" in prompt
        assert "STRICTLY ENFORCED" in prompt

    def test_data_questions_prompt_redirects_to_transform(self):
        prompt = data_questions_system_prompt()
        assert "Transform to Generic" in prompt

    def test_ask_your_data_draws_figures_itself(self):
        """Custom Visualization merged in (2026-11-08): plots are in scope,
        not redirected."""
        prompt = data_questions_system_prompt()
        assert "## Figures" in prompt
        assert "produces visualizations" not in prompt

    def test_lc_questions_prompt_has_scope_section(self):
        prompt = lipidcruncher_questions_system_prompt()
        assert "SCOPE" in prompt
        assert "STRICTLY ENFORCED" in prompt

    def test_lc_questions_prompt_redirects_to_other_tabs(self):
        prompt = lipidcruncher_questions_system_prompt()
        assert "Ask Your Data" in prompt
        assert "Transform to Generic" in prompt


ALL_PROMPTS = [
    data_transform_system_prompt, data_questions_system_prompt,
    lipidcruncher_questions_system_prompt,
]
AI_STUDIO_TABS = {  # the labels ui/ai_studio/tabs.py passes to st.tabs
    "LipidCruncher Questions", "Ask Your Data", "Transform to Generic",
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
        assert "Figure Examples" in prompt
        assert "Example 1" in prompt
        assert "Example 8" in prompt

    def test_lc_questions_prompt_has_examples(self):
        prompt = lipidcruncher_questions_system_prompt()
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

    def test_data_questions_message_puts_schema_before_question(self, monkeypatch):
        stream = canned_stream("ok")
        monkeypatch.setattr(
            AIClientService, "stream_tool_loop", staticmethod(stream)
        )
        df = pd.DataFrame({"SCHEMA_MARKER": [1]})
        generator, _ = AIChatWorkflow.stream_data_questions_message(
            "USER_REQUEST_MARKER", AIChatWorkflow.start_data_questions_session(df), df,
        )
        list(generator)
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


def test_every_variable_the_prompt_advertises_exists_in_the_sandbox():
    """Regression (AQ6, from Custom Visualization): the prompt listed `plt`
    and `sns`, which the sandbox neither binds nor imports, so a matplotlib
    request was a guaranteed failed execution."""
    from app.services.ai_chat.sandbox import build_sandbox_globals

    section = data_questions_system_prompt().split(
        "The execution environment provides:", 1)[1].split("\n\n", 1)[0]
    advertised = set()
    for line in re.findall(r"^- (.+)$", section, flags=re.M):
        advertised.update(re.findall(r"`(\w+)`", line.split(" — ")[0]))
    available = build_sandbox_globals(pd.DataFrame({"a": [1]}))
    assert {"px", "go", "lc"} <= advertised
    assert advertised <= set(available), advertised - set(available)
