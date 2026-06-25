"""
Unit tests for the streaming tool loop (AIClientService.stream_tool_loop).

Uses a fake Anthropic client so no real API calls are made. Verifies that
text chunks are yielded live, that tool calls between turns are executed and
their results fed back, and that the system prompt is sent as a cached block.
"""
import types

import pandas as pd

from app.models.chat import AIConfig, ChatMode, ChatSession
from app.models.experiment import ExperimentConfig
from app.services.ai_chat.ai_client import AIClientService
from app.workflows.ai_chat import AIChatWorkflow


class _FakeStreamCtx:
    """Mimics the SDK's client.messages.stream(...) context manager."""

    def __init__(self, chunks, final_message):
        self._chunks = chunks
        self._final = final_message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for chunk in self._chunks:
            yield chunk

    def get_final_message(self):
        return self._final


class _FakeMessages:
    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        chunks, final = self._turns.pop(0)
        return _FakeStreamCtx(chunks, final)


class _FakeClient:
    def __init__(self, turns):
        self.messages = _FakeMessages(turns)


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_block(tool_id, name, tool_input):
    return types.SimpleNamespace(
        type="tool_use", id=tool_id, name=name, input=tool_input
    )


def _final(blocks):
    return types.SimpleNamespace(content=blocks)


class TestStreamToolLoop:

    def test_single_turn_yields_chunks(self, monkeypatch):
        turns = [(["Hello ", "world"], _final([_text_block("Hello world")]))]
        fake = _FakeClient(turns)
        monkeypatch.setattr(
            AIClientService, "create_client", staticmethod(lambda: fake)
        )

        gen = AIClientService.stream_tool_loop(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="SYS",
            config=AIConfig(),
            tools=[],
            tool_executor=lambda name, inp: "",
        )

        assert list(gen) == ["Hello ", "world"]
        # System prompt sent as a cache-enabled block.
        sys_block = fake.messages.calls[0]["system"][0]
        assert sys_block["text"] == "SYS"
        assert sys_block["cache_control"] == {"type": "ephemeral"}

    def test_runs_tool_then_streams_answer(self, monkeypatch):
        turns = [
            # Turn 1: a tool call, no user-facing text.
            ([], _final([_tool_block("t1", "search_codebase", {"query": "x"})])),
            # Turn 2: the streamed answer.
            (["The ", "answer"], _final([_text_block("The answer")])),
        ]
        fake = _FakeClient(turns)
        monkeypatch.setattr(
            AIClientService, "create_client", staticmethod(lambda: fake)
        )

        executed = []
        notified = []

        def executor(name, inp):
            executed.append((name, inp))
            return "tool result text"

        gen = AIClientService.stream_tool_loop(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="SYS",
            config=AIConfig(),
            tools=[{"name": "search_codebase"}],
            tool_executor=executor,
            on_tool_call=lambda name, inp: notified.append(name),
        )

        assert list(gen) == ["The ", "answer"]
        assert executed == [("search_codebase", {"query": "x"})]
        assert notified == ["search_codebase"]

        # The second API call must carry the tool_result back to the model.
        second_call_messages = fake.messages.calls[1]["messages"]
        assert any(
            m["role"] == "user"
            and isinstance(m["content"], list)
            and m["content"][0].get("type") == "tool_result"
            and m["content"][0]["content"] == "tool result text"
            for m in second_call_messages
        )


class TestFilterToExplanation:
    """The explanation filter yields only prose before the code fence while
    accumulating the complete raw text for later code extraction."""

    def test_suppresses_code_keeps_full(self):
        holder = {"full": ""}
        chunks = ["Here is the answer.\n\n", "```python\n", "x = 1\n", "```"]
        out = "".join(AIClientService._filter_to_explanation(chunks, holder))
        assert out == "Here is the answer.\n\n"
        assert holder["full"] == (
            "Here is the answer.\n\n```python\nx = 1\n```"
        )

    def test_handles_fence_split_across_chunks(self):
        holder = {"full": ""}
        chunks = ["text ``", "`python\ncode\n```"]
        out = "".join(AIClientService._filter_to_explanation(chunks, holder))
        assert out == "text "
        assert "```python" in holder["full"]

    def test_no_fence_yields_everything(self):
        holder = {"full": ""}
        chunks = ["Hello ", "world"]
        out = "".join(AIClientService._filter_to_explanation(chunks, holder))
        assert out == "Hello world"
        assert holder["full"] == "Hello world"


class TestStreamingWorkflowEndToEnd:
    """Stream the explanation, then finalize: code is extracted from the
    accumulated raw text and executed in the real sandbox."""

    def test_data_questions_stream_then_execute(self, monkeypatch):
        full = (
            "Here is the head.\n\n"
            "```python\nresult_df = df.head(2)\n```"
        )
        chunks = [
            "Here is the head.\n\n",
            "```python\n",
            "result_df = df.head(2)\n",
            "```",
        ]
        fake = _FakeClient([(chunks, _final([_text_block(full)]))])
        monkeypatch.setattr(
            AIClientService, "create_client", staticmethod(lambda: fake)
        )

        session = ChatSession(
            mode=ChatMode.DATA_QUESTIONS, schema_context="cols: a, b"
        )
        source_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        gen, finalize = AIChatWorkflow.stream_data_questions_message(
            user_message="Show the first two rows",
            session=session,
            source_df=source_df,
        )

        streamed = "".join(gen)
        # Code fence is suppressed from the live stream.
        assert streamed == "Here is the head.\n\n"
        assert "```" not in streamed

        result = finalize()
        # Code was extracted from the accumulated raw text and executed.
        assert result.code == "result_df = df.head(2)"
        assert result.result_df is not None
        assert len(result.result_df) == 2
        # Session now holds the user + assistant turns.
        assert len(result.session.messages) == 2
        assert result.session.messages[-1].role == "assistant"

    def test_transform_stream_then_execute(self, monkeypatch):
        full = "Converting now.\n\n```python\nresult_df = df\n```"
        chunks = ["Converting now.\n\n", "```python\nresult_df = df\n```"]
        fake = _FakeClient([(chunks, _final([_text_block(full)]))])
        monkeypatch.setattr(
            AIClientService, "create_client", staticmethod(lambda: fake)
        )

        session = ChatSession(
            mode=ChatMode.DATA_TRANSFORM, schema_context="Lipid, s1"
        )
        source_df = pd.DataFrame({"Lipid": ["PC 16:0", "PE 18:0"], "s1": [1.0, 2.0]})

        gen, finalize = AIChatWorkflow.stream_transform_message(
            user_message="Convert to Generic",
            session=session,
            source_df=source_df,
        )

        streamed = "".join(gen)
        assert streamed == "Converting now.\n\n"
        assert "```" not in streamed

        result = finalize()
        assert result.transformation is not None
        assert result.transformation.success
        assert result.transformation.code == "result_df = df"
        assert len(result.session.messages) == 2

    def test_viz_stream_then_execute(self, monkeypatch):
        full = (
            "Plotting.\n\n"
            "```python\nfig = px.bar(df, x='ClassKey', y='intensity[s1]')\n```"
        )
        chunks = [
            "Plotting.\n\n",
            "```python\nfig = px.bar(df, x='ClassKey', y='intensity[s1]')\n```",
        ]
        fake = _FakeClient([(chunks, _final([_text_block(full)]))])
        monkeypatch.setattr(
            AIClientService, "create_client", staticmethod(lambda: fake)
        )

        session = ChatSession(
            mode=ChatMode.VISUALIZATION, schema_context="ClassKey, intensity[s1]"
        )
        analysis_df = pd.DataFrame(
            {"ClassKey": ["PC", "PE"], "intensity[s1]": [1.0, 2.0]}
        )
        experiment = ExperimentConfig(
            conditions_list=["A"], number_of_samples_list=[1]
        )

        gen, finalize = AIChatWorkflow.stream_viz_message(
            user_message="Bar chart by class",
            session=session,
            analysis_df=analysis_df,
            experiment=experiment,
        )

        streamed = "".join(gen)
        assert streamed == "Plotting.\n\n"
        assert "```" not in streamed

        result = finalize()
        assert result.sandbox_result is not None
        assert result.sandbox_result.success
        assert result.sandbox_result.figure is not None
        assert len(result.session.messages) == 2


class TestStreamToolLoopExhaustion:
    """When tools never stop, the loop forces one final streamed answer."""

    def test_forced_final_answer_after_iteration_cap(self, monkeypatch):
        turns = [
            # Iteration 1 (the cap): a tool call, no answer yet.
            ([], _final([_tool_block("t1", "search_codebase", {"query": "x"})])),
            # Forced final turn: streamed answer.
            (["final answer"], _final([_text_block("final answer")])),
        ]
        fake = _FakeClient(turns)
        monkeypatch.setattr(
            AIClientService, "create_client", staticmethod(lambda: fake)
        )

        gen = AIClientService.stream_tool_loop(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="SYS",
            config=AIConfig(),
            tools=[{"name": "search_codebase"}],
            tool_executor=lambda name, inp: "result",
            max_iterations=1,
        )

        assert "".join(gen) == "final answer"
        # The forced final call carries the "give your final answer" nudge.
        final_messages = fake.messages.calls[-1]["messages"]
        assert any(
            m["role"] == "user"
            and isinstance(m["content"], str)
            and "final answer" in m["content"].lower()
            for m in final_messages
        )
