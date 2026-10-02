"""The same messages put the same input before the model on the API route and the subscription route.

metallm measured the two routes behaving differently on one rewrite task (median copy-similarity
0.32 on the subscription against 0.18 on the API). The subscription route differed in what 3tears
itself sends:

- the CLI puts its own identity line before the caller's system prompt, in a system block of its
  own, and the caller's prompt was glued to it with no separator;
- thinking and effort were left unset, so Claude Code's defaults applied -- adaptive thinking on,
  and each model's own launch effort -- where the API route sends neither.

Both are pinned here against the API route's actual request, captured by a scripted Messages API on
loopback,
and the subscription route's actual CLI options and query, captured at the Agent SDK client. What
the CLI adds on its own and no option removes (the identity line, a ``currentDate`` reminder, the
attribution block) is listed in ``claude_cli``'s module docstring and is outside what these see.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL, claude_cli_pool  # noqa: E402
from threetears.models.factory import create_chat_model  # noqa: E402

from ._provider_wire import serve_http_handler  # noqa: E402

API_KEY = "sk-ant-api03-faketestkey"
TOKEN = "sk-ant-oat01-faketokenfortest"

_PERSONA = "You rewrite text in your own words. Never copy a sentence verbatim."
_REQUEST = "Rewrite this: The quick brown fox jumps over the lazy dog."
_MESSAGES = [SystemMessage(content=_PERSONA), HumanMessage(content=_REQUEST)]


@pytest.fixture(autouse=True)
def _one_off_cli_per_call() -> Iterator[None]:
    """drive the one-off-client path, which the fake client stands in for."""
    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    yield
    claude_cli_pool.configure_claude_cli_pool(enabled=True)


async def _api_request(**provider_kwargs: Any) -> dict[str, Any]:
    """the request body the API route sends for :data:`_MESSAGES`.

    :param provider_kwargs: what the caller passes to ``create_chat_model``
    :ptype provider_kwargs: Any
    :return: the JSON body the Messages API received
    :rtype: dict[str, Any]
    """
    received: list[dict[str, Any]] = []
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_CHAT_MODEL,
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    with serve_http_handler(handler) as base_url:
        model = create_chat_model(
            DEFAULT_CHAT_MODEL, api_key=API_KEY, max_retries=0, base_url=base_url, **provider_kwargs
        )
        await model.ainvoke(_MESSAGES)
    [request] = received
    return request


# parity-exempt: stands in for ClaudeSDKClient's query/receive_response only, recording options and prompt
class _FakeSDKClient:
    """Stands in for ``claude_agent_sdk.ClaudeSDKClient``: records what the CLI would be launched with."""

    options: list[Any] = []
    prompts: list[str] = []

    def __init__(self, options: Any) -> None:
        _FakeSDKClient.options.append(options)

    async def __aenter__(self) -> _FakeSDKClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        _FakeSDKClient.prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[Any]:
        yield AssistantMessage(content=[TextBlock(text="done")], model=DEFAULT_CHAT_MODEL)
        yield ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=1, session_id="s"
        )


@contextmanager
def _fake_cli() -> Iterator[None]:
    """every binding of the SDK client patched to record, for as long as the block runs."""
    _FakeSDKClient.options = []
    _FakeSDKClient.prompts = []
    with (
        patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
        patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
    ):
        yield


async def _cli_launch(**provider_kwargs: Any) -> tuple[Any, str]:
    """the options and query the subscription route sends the CLI for :data:`_MESSAGES`.

    :param provider_kwargs: what the caller passes to ``create_chat_model``
    :ptype provider_kwargs: Any
    :return: ``(options, query)``
    :rtype: tuple[Any, str]
    """
    with _fake_cli():
        model = create_chat_model(DEFAULT_CHAT_MODEL, api_key=TOKEN, **provider_kwargs)
        await model.ainvoke(_MESSAGES)
    [options] = _FakeSDKClient.options
    [query] = _FakeSDKClient.prompts
    return options, query


class TestWhatTheModelIsGiven:
    async def test_the_system_prompt_is_the_callers_own_after_a_blank_line(self) -> None:
        api = await _api_request()
        options, _query = await _cli_launch()

        assert api["system"] == _PERSONA
        assert options.system_prompt == f"\n\n{_PERSONA}", (
            "the CLI's identity line precedes the prompt in a block of its own; the caller's prompt "
            "must start a paragraph of its own, and carry nothing else"
        )

    async def test_the_whole_prompt_replaces_claude_codes_rather_than_appending_to_it(self) -> None:
        """a preset with ``append`` would send Claude Code's own system prompt first."""
        options, _query = await _cli_launch()

        assert isinstance(options.system_prompt, str)

    async def test_the_query_is_the_persons_message_and_nothing_else(self) -> None:
        api = await _api_request()
        _options, query = await _cli_launch()

        assert api["messages"] == [{"role": "user", "content": _REQUEST}]
        assert (
            query == f"The person's current message:\n<prompt-current-message>\n{_REQUEST}\n</prompt-current-message>"
        )


class TestThinkingAndEffort:
    async def test_left_unset_there_is_no_extended_thinking_on_either_route(self) -> None:
        api = await _api_request()
        options, _query = await _cli_launch()

        assert "thinking" not in api, "the API route sends no thinking: the model does not think"
        assert options.thinking == {"type": "disabled"}, "the CLI defaults to adaptive thinking unless told"
        assert options.max_thinking_tokens is None

    async def test_left_unset_the_effort_is_the_apis_own_default(self) -> None:
        """the API's default effort is ``high``; the CLI would send each model's launch effort instead.

        ``CLAUDE_CODE_EFFORT_LEVEL`` outranks every other effort source in the CLI, so the pin lives
        there as well as in the flag.
        """
        api = await _api_request()
        options, _query = await _cli_launch()

        assert "effort" not in api.get("output_config", {})
        assert options.effort == "high"
        assert options.env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"

    async def test_what_a_caller_sets_once_reaches_both_routes(self) -> None:
        thinking = {"type": "enabled", "budget_tokens": 2048}
        api = await _api_request(thinking=thinking, effort="low")
        options, _query = await _cli_launch(thinking=thinking, effort="low")

        assert api["thinking"] == thinking
        assert api["output_config"]["effort"] == "low"
        assert options.thinking == thinking
        assert options.effort == "low"
        assert options.env["CLAUDE_CODE_EFFORT_LEVEL"] == "low"

    async def test_values_bound_for_one_call_reach_the_cli_too(self) -> None:
        with _fake_cli():
            model = create_chat_model(DEFAULT_CHAT_MODEL, api_key=TOKEN)
            await model.bind(thinking={"type": "adaptive"}, effort="medium").ainvoke(_MESSAGES)
        [options] = _FakeSDKClient.options

        assert options.thinking == {"type": "adaptive"}
        assert options.effort == "medium"
        assert options.env["CLAUDE_CODE_EFFORT_LEVEL"] == "medium"
