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
from threetears.models.providers import _claude_cli
from threetears.models.providers._claude_cli import create_subscription_chat

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
    def test_the_stable_part_is_the_system_prompt_and_the_variable_part_travels(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        query, system = model._convert_messages(  # noqa: SLF001
            [_structured_system("## Persona\nBe terse.", "## Memory\nLikes tea."), HumanMessage(content="hi")]
        )
        assert system == "## Persona\nBe terse."
        assert query.startswith("## Memory\nLikes tea."), "the changing context did not reach the query"
        assert query.endswith("Human: hi")

    def test_the_system_prompt_is_text_not_a_python_repr(self) -> None:
        """Found live: the CLI received ``[{'type': 'text', ...}]`` as the agent's persona."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        _query, system = model._convert_messages(  # noqa: SLF001
            [_structured_system("## Persona\nBe terse.", "## Memory\nLikes tea.")]
        )
        assert system is not None
        assert "cache_control" not in system
        assert "{'type'" not in system
        assert "\\n" not in system, "newlines arrived as literal backslash-n"

    def test_a_turn_whose_context_changed_keeps_the_same_system_prompt(self) -> None:
        """That is the whole condition for one CLI serving the next turn."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        _q1, first = model._convert_messages([_structured_system("persona", "memory one")])  # noqa: SLF001
        _q2, second = model._convert_messages([_structured_system("persona", "memory two")])  # noqa: SLF001
        assert first == second

    def test_a_string_system_message_is_all_stable(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        query, system = model._convert_messages([SystemMessage(content="plain persona"), HumanMessage(content="hi")])  # noqa: SLF001
        assert system == "plain persona"
        assert query == "Human: hi"

    def test_list_content_everywhere_is_read_as_text(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        query, _system = model._convert_messages(  # noqa: SLF001
            [
                HumanMessage(
                    content=[{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "x"}}]
                ),
                AIMessage(content=[{"type": "text", "text": "seen"}]),
                ToolMessage(content=[{"type": "text", "text": "42"}], tool_call_id="c", name="calc"),
            ]
        )
        assert "Human: look" in query
        assert "[image_url content omitted]" in query
        assert "Assistant: seen" in query
        assert "Tool (calc): 42" in query
        assert "{'type'" not in query


class TestPooledLaunchOptions:
    def _call_options(self) -> Any:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        options = model._build_options(allowed_tools=["mcp__langchain-tools__calc", "WebSearch"])  # noqa: SLF001
        return options

    def test_tools_are_approved_for_the_whole_server_so_a_swapped_set_needs_no_relaunch(self) -> None:
        launch = _claude_cli._pooled_launch_options(self._call_options())  # noqa: SLF001
        assert "mcp__langchain-tools" in launch.allowed_tools
        assert "mcp__langchain-tools__calc" not in launch.allowed_tools
        assert "WebSearch" in launch.allowed_tools, "a caller's own non-MCP approval was dropped"

    def test_the_session_launches_with_a_placeholder_server_and_partial_messages(self) -> None:
        launch = _claude_cli._pooled_launch_options(self._call_options())  # noqa: SLF001
        assert set(launch.mcp_servers) == {"langchain-tools"}
        assert launch.include_partial_messages is True

    def test_isolation_survives_into_the_pooled_launch(self) -> None:
        launch = _claude_cli._pooled_launch_options(self._call_options())  # noqa: SLF001
        assert launch.env.get("CLAUDE_CONFIG_DIR")
        assert launch.env.get("ENABLE_CLAUDEAI_MCP_SERVERS") == "false"
        assert "strict-mcp-config" in launch.extra_args
        assert "no-session-persistence" in launch.extra_args

    def test_the_calls_own_options_are_not_mutated(self) -> None:
        options = self._call_options()
        before = list(options.allowed_tools)
        _claude_cli._pooled_launch_options(options)  # noqa: SLF001
        assert options.allowed_tools == before


# the test checks which client a call got and never drives query/receive_response, so method parity would test nothing
# parity-exempt: stands in for ClaudeSDKClient only as an async context manager, never as a driven client
class _FakeClient:
    """Stands in for a one-off ``ClaudeSDKClient``: an async context manager."""

    opened = 0

    def __init__(self, options: Any) -> None:
        self.options = options

    async def __aenter__(self) -> _FakeClient:
        _FakeClient.opened += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class TestACallIsNeverRefused:
    @pytest.fixture(autouse=True)
    def _own_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import claude_agent_sdk

        _FakeClient.opened = 0
        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)

    async def _client_for(self, pool: Any, *, pooled: bool = True) -> Any:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        options = model._build_options()  # noqa: SLF001
        original = _claude_cli.claude_cli_pool
        _claude_cli.claude_cli_pool = lambda: pool  # type: ignore[assignment]
        try:
            async with model._cli_client(options, pooled=pooled) as client:  # noqa: SLF001
                return client
        finally:
            _claude_cli.claude_cli_pool = original  # type: ignore[assignment]

    @pytest.mark.parametrize("failure", [ClaudeCliPoolExhausted("busy"), ClaudeCliSessionError("would not start")])
    async def test_an_unavailable_pool_falls_back_to_a_cli_of_the_calls_own(self, failure: Exception) -> None:
        class _Failing:
            @asynccontextmanager
            async def checkout(self, *args: Any, **kwargs: Any) -> Any:
                raise failure
                yield  # pragma: no cover

        client = await self._client_for(_Failing())
        assert isinstance(client, _FakeClient)
        assert _FakeClient.opened == 1

    async def test_a_resumed_session_never_touches_the_pool(self) -> None:
        class _Unused:
            def checkout(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("a call resuming a stored session was pooled")

        client = await self._client_for(_Unused(), pooled=False)
        assert isinstance(client, _FakeClient)

    async def test_a_host_that_turned_pooling_off_gets_a_cli_per_call(self) -> None:
        client = await self._client_for(None)
        assert isinstance(client, _FakeClient)

    async def test_a_pooled_session_is_used_when_one_is_free(self) -> None:
        pooled_client = object()
        seen: dict[str, Any] = {}

        class _Serving:
            @asynccontextmanager
            async def checkout(self, options: Any, *, token: Any, tool_server: Any, call_context: Any = None) -> Any:
                seen["token"] = token
                seen["call_context"] = call_context
                seen["system_prompt"] = options.system_prompt
                yield pooled_client

        client = await self._client_for(_Serving())
        assert client is pooled_client
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

        monkeypatch.setattr(_claude_cli, "claude_cli_pool", lambda: _Serving())

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
        assert serving_pool["system_prompt"] == "persona"
        assert serving_pool["client"].queries[0].startswith("memory")

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
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        return _claude_cli._pooled_launch_options(model._build_options())  # noqa: SLF001

    def test_an_ordinary_call_may_share_a_cli(self) -> None:
        assert claude_cli_pool.poolable(self._launch()), "every real call would run on a CLI of its own"

    def test_two_identical_calls_get_the_same_key(self) -> None:
        a, b = self._launch(), self._launch()
        assert claude_cli_pool.launch_key(a, TOKEN) == claude_cli_pool.launch_key(b, TOKEN), (
            "an option renders unstably, so no session would ever be shared"
        )
