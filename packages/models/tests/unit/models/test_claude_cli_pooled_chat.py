"""The subscription chat model on a pooled CLI: prompt shape, launch options, and fallback.

A running CLI's system prompt cannot change, so a session can only serve turn after turn if the
part of the prompt that changes travels in the query instead. Callers that cache prompts already
mark that boundary with ``cache_control``; these pin that the model honours it, that the old
``str(list)`` repr is gone, and that a call is never refused for want of a pooled session.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from threetears.models import DEFAULT_CHAT_MODEL, claude_cli_pool
from threetears.models.claude_cli_pool import ClaudeCliPoolExhausted, ClaudeCliSessionError
from threetears.models.providers import claude_cli
from threetears.models.providers.claude_cli import create_subscription_chat

from .claude_cli_recorder import pooled_launch, sent_to_cli

TOKEN = "sk-ant-oat01-faketokenfortest"


def _structured_system(stable: str, variable: str) -> SystemMessage:
    """The shape a prompt-caching caller sends: the stable block marked, the variable block after."""
    return SystemMessage(
        content=[
            {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": variable},
        ]
    )


class TestThePromptShape:
    """What the CLI receives: its system prompt and its query, for each shape of round."""

    def test_the_stable_part_is_the_system_prompt_and_the_variable_part_travels(self) -> None:
        call = sent_to_cli(
            [_structured_system("## Persona\nBe terse.", "## Memory\nLikes tea."), HumanMessage(content="hi")]
        )
        assert call.system_prompt == "## Persona\nBe terse."
        assert "<prompt-context>\n## Memory\nLikes tea.\n</prompt-context>" in call.query, (
            "the changing context did not reach the query"
        )
        assert call.query.endswith("<prompt-current-message>\nhi\n</prompt-current-message>")

    def test_the_system_prompt_is_text_not_a_python_repr(self) -> None:
        """Found live: the CLI received ``[{'type': 'text', ...}]`` as the agent's persona."""
        system = sent_to_cli([_structured_system("## Persona\nBe terse.", "## Memory\nLikes tea.")]).system_prompt
        assert system is not None
        assert "cache_control" not in system
        assert "{'type'" not in system
        assert "\\n" not in system, "newlines arrived as literal backslash-n"

    def test_a_turn_whose_context_changed_keeps_the_same_system_prompt(self) -> None:
        """That is the whole condition for one CLI serving the next turn."""
        first = sent_to_cli([_structured_system("persona", "memory one")]).options.system_prompt
        second = sent_to_cli([_structured_system("persona", "memory two")]).options.system_prompt
        assert first == second

    def test_a_string_system_message_is_all_stable(self) -> None:
        call = sent_to_cli([SystemMessage(content="plain persona"), HumanMessage(content="hi")])
        assert call.system_prompt == "plain persona"
        assert call.query == "The person's current message:\n<prompt-current-message>\nhi\n</prompt-current-message>"

    def test_list_content_everywhere_is_read_as_text(self) -> None:
        query = sent_to_cli(
            [
                HumanMessage(
                    content=[{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}}]
                ),
                AIMessage(content=[{"type": "text", "text": "seen"}]),
                ToolMessage(content=[{"type": "text", "text": "42"}], tool_call_id="c", name="calc"),
            ]
        ).query
        assert "<prompt-current-message>\nlook\n\n[image_url content omitted]\n</prompt-current-message>" in query
        assert '<prompt-turn role="assistant">\nseen\n</prompt-turn>' in query
        assert '<prompt-turn role="tool" name="calc">\n42\n</prompt-turn>' in query
        assert "{'type'" not in query


class TestPooledLaunchOptions:
    """The launch a call asks the pool for, read where the pool receives it."""

    _ALLOWED = ["mcp__langchain-tools__calc", "WebSearch"]

    def _launch(self) -> Any:
        return pooled_launch([HumanMessage(content="hi")], allowed_tools=self._ALLOWED)

    def test_tools_are_approved_for_the_whole_server_so_a_swapped_set_needs_no_relaunch(self) -> None:
        launch = self._launch().options
        assert "mcp__langchain-tools" in launch.allowed_tools
        assert "mcp__langchain-tools__calc" not in launch.allowed_tools
        assert "WebSearch" in launch.allowed_tools, "a caller's own non-MCP approval was dropped"

    def test_the_session_launches_with_a_placeholder_server_and_partial_messages(self) -> None:
        launch = self._launch().options
        assert set(launch.mcp_servers) == {"langchain-tools"}
        assert launch.include_partial_messages is True

    def test_isolation_survives_into_the_pooled_launch(self) -> None:
        launch = self._launch().options
        assert launch.env.get("CLAUDE_CONFIG_DIR")
        assert launch.env.get("ENABLE_CLAUDEAI_MCP_SERVERS") == "false"
        assert "strict-mcp-config" in launch.extra_args
        assert "no-session-persistence" in launch.extra_args

    def test_the_calls_own_options_are_not_mutated(self) -> None:
        """A call the pool refuses runs on a CLI of its own with the options it was built with; the
        pooled launch derived from them first must not have changed them."""
        unpooled = sent_to_cli([HumanMessage(content="hi")], allowed_tools=self._ALLOWED).options
        fallback = self._launch().fallback
        assert fallback.allowed_tools == unpooled.allowed_tools
        assert "mcp__langchain-tools__calc" in fallback.allowed_tools
        assert fallback.mcp_servers == unpooled.mcp_servers


# parity-exempt: answers a one-off call with one fixed reply and counts how many were opened; nothing else is driven
class _FakeClient:
    """Stands in for a one-off ``ClaudeSDKClient``: counts openings and answers with one reply."""

    opened = 0

    def __init__(self, options: Any) -> None:
        self.options = options

    async def __aenter__(self) -> _FakeClient:
        _FakeClient.opened += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def query(self, prompt: str, session_id: str = "default") -> None:
        del prompt, session_id

    async def receive_response(self) -> Any:
        async for message in _ScriptedClient("own answer").receive_response():
            yield message


class TestACallIsNeverRefused:
    @pytest.fixture(autouse=True)
    def _own_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import claude_agent_sdk

        _FakeClient.opened = 0
        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)

    async def _answer_with(self, pool: Any, monkeypatch: pytest.MonkeyPatch, **call_kwargs: Any) -> str:
        """one call through the model's public entry point, with ``pool`` as the process-wide pool.

        :return: the answer's text, which names the CLI that gave it
        :rtype: str
        """
        monkeypatch.setattr(claude_cli, "claude_cli_pool", lambda: pool)
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        result = await model.ainvoke([HumanMessage(content="hi")], **call_kwargs)
        return str(result.content)

    @pytest.mark.parametrize("failure", [ClaudeCliPoolExhausted("busy"), ClaudeCliSessionError("would not start")])
    async def test_an_unavailable_pool_falls_back_to_a_cli_of_the_calls_own(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Failing:
            @asynccontextmanager
            async def checkout(self, *args: Any, **kwargs: Any) -> Any:
                raise failure
                yield  # pragma: no cover

        assert await self._answer_with(_Failing(), monkeypatch) == "own answer"
        assert _FakeClient.opened == 1

    async def test_a_resumed_session_never_touches_the_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Unused:
            def checkout(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("a call resuming a stored session was pooled")

        assert await self._answer_with(_Unused(), monkeypatch, session_id="stored-session") == "own answer"
        assert _FakeClient.opened == 1

    async def test_a_host_that_turned_pooling_off_gets_a_cli_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert await self._answer_with(None, monkeypatch) == "own answer"
        assert _FakeClient.opened == 1

    async def test_a_pooled_session_is_used_when_one_is_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class _Serving:
            @asynccontextmanager
            async def checkout(self, options: Any, *, token: Any, tool_server: Any, call_context: Any = None) -> Any:
                seen["token"] = token
                seen["call_context"] = call_context
                seen["system_prompt"] = options.system_prompt
                yield _ScriptedClient("pooled answer")

        assert await self._answer_with(_Serving(), monkeypatch) == "pooled answer"
        assert _FakeClient.opened == 0, "a CLI of its own was started although a pooled one was free"
        assert seen["token"] == TOKEN
        assert seen["call_context"] is not None, "tool calls on a reused CLI would run in its first caller's context"


def test_the_process_wide_pool_can_be_turned_off() -> None:
    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    try:
        assert claude_cli_pool.claude_cli_pool() is None
    finally:
        claude_cli_pool.configure_claude_cli_pool(enabled=True)


# parity-exempt: a scripted reply stream standing in for a connected ClaudeSDKClient; only query and receive_response are driven
class _ScriptedClient:
    """A connected client that answers every query with one scripted reply."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.queries: list[str] = []

    async def query(self, prompt: str, session_id: str = "default") -> None:
        del session_id
        self.queries.append(prompt)

    async def receive_response(self) -> Any:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        yield AssistantMessage(content=[TextBlock(text=self.text)], model=DEFAULT_CHAT_MODEL)
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s"
        )


class TestTheModelIsWiredToThePool:
    """Pinned through the model's public entry points, not ``_cli_client`` directly: reverting either
    generation path to ``ClaudeSDKClient(options)`` must fail here (found by review)."""

    @pytest.fixture
    def serving_pool(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        state: dict[str, Any] = {"checkouts": 0, "client": _ScriptedClient("pooled answer")}

        class _Serving:
            @asynccontextmanager
            async def checkout(self, options: Any, *, token: Any, tool_server: Any, call_context: Any = None) -> Any:
                state["checkouts"] += 1
                state["system_prompt"] = options.system_prompt
                yield state["client"]

        monkeypatch.setattr(claude_cli, "claude_cli_pool", lambda: _Serving())

        def _never(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a CLI of the call's own was started although the pool served the call")

        import claude_agent_sdk

        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _never)
        return state

    async def test_a_non_streaming_call_runs_on_a_pooled_cli(self, serving_pool: dict[str, Any]) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        result = await model.ainvoke([_structured_system("persona", "memory"), HumanMessage(content="hi")])
        assert serving_pool["checkouts"] == 1
        assert "pooled answer" in str(result.content)
        # A blank line first: the CLI's own identity line comes before it (see test_claude_cli_api_parity).
        assert serving_pool["system_prompt"] == "\n\npersona"
        assert "<prompt-context>\nmemory\n</prompt-context>" in serving_pool["client"].queries[0]

    async def test_a_streaming_call_runs_on_a_pooled_cli(self, serving_pool: dict[str, Any]) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        chunks = [
            chunk
            async for chunk in model.astream([_structured_system("persona", "memory"), HumanMessage(content="hi")])
        ]
        assert serving_pool["checkouts"] == 1
        assert "pooled answer" in "".join(str(c.content) for c in chunks)


class TestRealOptionsArePoolable:
    """Pinned on REAL ``ClaudeAgentOptions`` from the real model, not a stand-in. A stand-in let a
    check ship that refused every real call: ``debug_stderr`` defaults to ``sys.stderr``, which is
    truthy, and was treated as a callable -- so pooling was silently off for every call."""

    def _launch(self) -> Any:
        return pooled_launch([HumanMessage(content="hi")]).options

    def test_an_ordinary_call_may_share_a_cli(self) -> None:
        assert claude_cli_pool.poolable(self._launch()), "every real call would run on a CLI of its own"

    def test_two_identical_calls_get_the_same_key(self) -> None:
        a, b = self._launch(), self._launch()
        assert claude_cli_pool.launch_key(a, TOKEN) == claude_cli_pool.launch_key(b, TOKEN), (
            "an option renders unstably, so no session would ever be shared"
        )
