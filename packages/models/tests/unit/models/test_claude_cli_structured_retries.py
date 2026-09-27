"""A structured call under a subscription gets the turns the CLI's schema retries need.

Found live in 0.55.0: about a third of one consumer's structured calls failed with
``ModelProviderError: Claude subscription call failed (error_max_turns): Reached maximum number of
turns (1)``. The CLI does not constrain the model's output to the schema. It checks the model's
``StructuredOutput`` call, answers a mismatch with what did not match, and lets the model try again
in a turn of its own. Measured against the bundled CLI (2.1.207, ``claude-sonnet-5``, a schema whose
``sentences`` is an array of objects with enum fields), the model's first call regularly fills the
tool with a placeholder, ``{"$PARAMETER_VALUE": "<the answer, as a string>"}``, or wraps the answer
in one key too many. At ``--max-turns 1`` there was no next turn: the call ended on the rejected
attempt with no ``structured_output``. Before 0.55.0 such a call came back as empty content with no
error; since 0.55.0 it raises, which is what made it visible.

The message sequences below are the ones the CLI sent in those measurements, trimmed to the fields
that matter. Only the Claude Agent SDK subprocess boundary is faked.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from threetears.models import DEFAULT_CHAT_MODEL, claude_cli_pool  # noqa: E402
from threetears.models.errors import ModelProviderError  # noqa: E402
from threetears.models.providers._claude_cli import create_subscription_chat  # noqa: E402
from threetears.models.providers.structured_output import structured_output_kwargs  # noqa: E402

TOKEN = "sk-ant-oat01-faketokenfortest"

#: The shape of the consumer's ``check_draft`` schema: an array of objects with enum fields.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["keep", "revise", "cut"]},
                },
                "required": ["index", "verdict"],
            },
        },
        "overall": {"type": "string", "enum": ["ready", "needs_work", "rewrite"]},
    },
    "required": ["sentences", "overall"],
}

_ANSWER: dict[str, Any] = {"sentences": [{"index": 1, "verdict": "keep"}], "overall": "ready"}

#: What the CLI answers a ``StructuredOutput`` call that does not match the schema.
_MISMATCH = (
    "Output does not match required schema: root: must have required property 'sentences', "
    "root: must have required property 'overall', root: must NOT have additional properties"
)


@pytest.fixture(autouse=True)
def _one_off_cli_per_call() -> Iterator[None]:
    """drive the one-off-client path, which the fake client stands in for."""
    claude_cli_pool.configure_claude_cli_pool(enabled=False)
    yield
    claude_cli_pool.configure_claude_cli_pool(enabled=True)


# parity-exempt: stands in for ClaudeSDKClient's query/receive_response only, recording options, driven by a per-test script
class _FakeSDKClient:
    """Stands in for ``claude_agent_sdk.ClaudeSDKClient``: records its options, answers from ``script``."""

    options: list[Any] = []
    script: Callable[[], list[Any]] | None = None

    def __init__(self, options: Any) -> None:
        _FakeSDKClient.options.append(options)

    async def __aenter__(self) -> _FakeSDKClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self) -> AsyncIterator[Any]:
        assert _FakeSDKClient.script is not None, "test must set _FakeSDKClient.script"
        for msg in _FakeSDKClient.script():
            yield msg


def _tripwire_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("the REAL ClaudeSDKClient ran; a patch binding was missed")


@contextmanager
def _fake_cli(*messages: Any) -> Iterator[None]:
    """every binding of the SDK client patched to record and answer with ``messages``.

    :param messages: what the CLI sends for each call
    :ptype messages: Any
    :return: a context in which no real CLI can start
    :rtype: Iterator[None]
    """
    _FakeSDKClient.options = []
    _FakeSDKClient.script = lambda: list(messages)
    with (
        patch("claude_agent_sdk.ClaudeSDKClient", _FakeSDKClient),
        patch("langchain_claude_code.claude_chat_model.ClaudeSDKClient", _FakeSDKClient),
        patch("claude_agent_sdk.client.ClaudeSDKClient.__init__", _tripwire_init),
    ):
        yield
    _FakeSDKClient.script = None


def _structured_call(tool_use_id: str, arguments: dict[str, Any]) -> AssistantMessage:
    """the model's ``StructuredOutput`` call, as the CLI relays it.

    :param tool_use_id: the tool-use id
    :ptype tool_use_id: str
    :param arguments: what the model filled the tool with
    :ptype arguments: dict[str, Any]
    :return: the assistant message
    :rtype: AssistantMessage
    """
    return AssistantMessage(
        content=[ToolUseBlock(id=tool_use_id, name="StructuredOutput", input=arguments)], model=DEFAULT_CHAT_MODEL
    )


def _tool_result(tool_use_id: str, content: str, *, is_error: bool | None) -> UserMessage:
    """the CLI's answer to a ``StructuredOutput`` call.

    :param tool_use_id: the call it answers
    :ptype tool_use_id: str
    :param content: what the CLI said
    :ptype content: str
    :param is_error: whether the call was rejected
    :ptype is_error: bool | None
    :return: the user message carrying the tool result
    :rtype: UserMessage
    """
    return UserMessage(content=[ToolResultBlock(tool_use_id=tool_use_id, content=content, is_error=is_error)])


#: A placeholder-wrapped attempt whose JSON misses the schema (no ``overall``), so it stays a rejection.
_WRAPPED_MISS = {"$PARAMETER_VALUE": json.dumps({"sentences": _ANSWER["sentences"]})}


def _placeholder_call() -> AssistantMessage:
    """the malformed first attempt: an answer as a string under a placeholder parameter name, one
    that misses the schema, so nothing can be recovered from it.

    :return: the assistant message
    :rtype: AssistantMessage
    """
    return _structured_call("toolu_1", _WRAPPED_MISS)


def _recovered() -> tuple[Any, ...]:
    """a call whose first attempt was rejected and whose second was accepted.

    :return: the messages the CLI sends
    :rtype: tuple[Any, ...]
    """
    return (
        _placeholder_call(),
        _tool_result("toolu_1", _MISMATCH, is_error=True),
        _structured_call("toolu_2", _ANSWER),
        _tool_result("toolu_2", "Structured output provided successfully", is_error=None),
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=3,
            session_id="s",
            stop_reason="tool_use",
            structured_output=_ANSWER,
        ),
    )


def _cut_off_after_one_attempt() -> tuple[Any, ...]:
    """what ``--max-turns 1`` made of the same call: the rejected attempt, then the turn limit.

    :return: the messages the CLI sends
    :rtype: tuple[Any, ...]
    """
    return (
        _placeholder_call(),
        _tool_result("toolu_1", _MISMATCH, is_error=True),
        ResultMessage(
            subtype="error_max_turns",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=2,
            session_id="s",
            stop_reason="tool_use",
            errors=["Reached maximum number of turns (1)"],
        ),
    )


def _out_of_attempts() -> tuple[Any, ...]:
    """a call whose every attempt was rejected: the CLI's attempt cap ends it.

    :return: the messages the CLI sends
    :rtype: tuple[Any, ...]
    """
    return (
        _placeholder_call(),
        _tool_result("toolu_1", _MISMATCH, is_error=True),
        ResultMessage(
            subtype="error_max_structured_output_retries",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=2,
            session_id="s",
            errors=["Failed to provide valid structured output after 5 attempts"],
        ),
    )


@tool
def lookup(word: str) -> str:
    """Look a word up.

    :param word: the word
    :ptype word: str
    :return: the word
    :rtype: str
    """
    return word


def _structured(model: Any) -> Any:
    """``model`` bound to answer in :data:`_SCHEMA`, as a caller asks on either Anthropic route.

    :param model: the model, or a tool binding of it
    :ptype model: Any
    :return: the bound runnable
    :rtype: Any
    """
    return model.bind(**structured_output_kwargs("anthropic", _SCHEMA))


async def test_a_call_that_can_only_answer_in_its_schema_gets_the_turns_the_retries_need() -> None:
    with _fake_cli(*_recovered()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        # A bound kwarg reaches the option builder as an override; the factory drops max_turns.
        message = await _structured(model.bind(max_turns=12)).ainvoke([HumanMessage(content="check the draft")])

    [options] = _FakeSDKClient.options
    assert options.max_turns == 6, "five attempts, and one turn more, whatever the caller asked for"
    assert options.env["MAX_STRUCTURED_OUTPUT_RETRIES"] == "5", "the CLI's attempt cap is pinned, not inherited"
    assert isinstance(message, AIMessage)
    assert json.loads(message.content) == _ANSWER, "the answer is the accepted attempt"
    assert message.tool_calls == [], "a StructuredOutput call is never handed back as a tool call"
    assert message.response_metadata["is_error"] is False


async def test_a_streamed_call_that_can_only_answer_in_its_schema_gets_the_same_turns() -> None:
    with _fake_cli(*_recovered()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        merged: Any = None
        async for chunk in _structured(model).astream([HumanMessage(content="check the draft")]):
            merged = chunk if merged is None else merged + chunk

    [options] = _FakeSDKClient.options
    assert options.max_turns == 6
    assert options.env["MAX_STRUCTURED_OUTPUT_RETRIES"] == "5"
    assert json.loads(merged.content) == _ANSWER


async def test_a_call_with_a_schema_and_bound_tools_stays_at_one_turn() -> None:
    with _fake_cli(*_recovered()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        await _structured(model.bind_tools([lookup])).ainvoke([HumanMessage(content="look it up")])

    [options] = _FakeSDKClient.options
    assert options.max_turns == 1, "a second turn could run the caller's tools in the caller's place"
    assert "MAX_STRUCTURED_OUTPUT_RETRIES" not in options.env


async def test_a_call_with_a_schema_and_built_in_tools_stays_at_one_turn() -> None:
    with _fake_cli(*_recovered()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN, tools=["WebSearch"])
        await _structured(model).ainvoke([HumanMessage(content="search for it")])

    [options] = _FakeSDKClient.options
    assert options.max_turns == 1, "a second turn could run a built-in tool the caller never saw asked for"


async def test_a_call_with_the_full_built_in_preset_stays_at_one_turn() -> None:
    with _fake_cli(*_recovered()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN, tools=None)
        await _structured(model).ainvoke([HumanMessage(content="do it")])

    [options] = _FakeSDKClient.options
    assert options.max_turns == 1, "no tools list means Claude Code's whole built-in belt"


async def test_a_call_with_a_schema_and_another_mcp_server_stays_at_one_turn() -> None:
    other_server = {"other": {"type": "stdio", "command": "never-started"}}
    with _fake_cli(*_recovered()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        await _structured(model.bind(mcp_servers=other_server)).ainvoke([HumanMessage(content="use it")])

    [options] = _FakeSDKClient.options
    assert options.mcp_servers == other_server, "the caller's server reached the options"
    assert options.max_turns == 1, "a second turn could run that server's tools in the caller's place"
    assert "MAX_STRUCTURED_OUTPUT_RETRIES" not in options.env


async def test_a_call_without_a_schema_stays_at_one_turn() -> None:
    with _fake_cli(AssistantMessage(content=[TextBlock(text="hi")], model=DEFAULT_CHAT_MODEL), _recovered()[-1]):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        await model.ainvoke([HumanMessage(content="hi")])

    [options] = _FakeSDKClient.options
    assert options.max_turns == 1
    assert "MAX_STRUCTURED_OUTPUT_RETRIES" not in options.env


async def test_a_call_that_runs_out_of_attempts_raises() -> None:
    with _fake_cli(*_out_of_attempts()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        with pytest.raises(ModelProviderError) as raised:
            await _structured(model).ainvoke([HumanMessage(content="check the draft")])

    assert raised.value.reason == "error_max_structured_output_retries"
    assert "Failed to provide valid structured output after 5 attempts" in str(raised.value)


async def test_a_call_that_fails_on_its_schema_carries_what_the_schema_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _fake_cli(*_out_of_attempts()), caplog.at_level("WARNING"):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        with pytest.raises(ModelProviderError) as raised:
            await _structured(model).ainvoke([HumanMessage(content="check the draft")])

    assert raised.value.rejected_output == _WRAPPED_MISS, "exactly what the model gave"
    assert raised.value.rejection == _MISMATCH, "the CLI's own reason, unparsed"
    assert _MISMATCH in str(raised.value), "an operator reading the error sees which field was missed"
    [logged] = [r for r in caplog.records if r.getMessage() == "A subscription model call failed"]
    assert logged.__dict__["extra_data"]["rejection"] == _MISMATCH
    assert logged.__dict__["extra_data"]["rejected_keys"] == ["$PARAMETER_VALUE"], "keys, never the model words"


async def test_a_streamed_call_that_fails_on_its_schema_carries_what_the_schema_rejected() -> None:
    with _fake_cli(*_cut_off_after_one_attempt()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        with pytest.raises(ModelProviderError) as raised:
            async for _chunk in _structured(model).astream([HumanMessage(content="check the draft")]):
                pass

    assert raised.value.rejected_output == _WRAPPED_MISS
    assert raised.value.rejection == _MISMATCH


async def test_an_attempt_the_cli_accepted_is_never_reported_as_rejected() -> None:
    accepted_then_failed = (
        _structured_call("toolu_1", _ANSWER),
        _tool_result("toolu_1", "Structured output provided successfully", is_error=None),
        _out_of_attempts()[-1],
    )
    with _fake_cli(*accepted_then_failed):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        with pytest.raises(ModelProviderError) as raised:
            await _structured(model).ainvoke([HumanMessage(content="check the draft")])

    assert raised.value.rejected_output is None
    assert raised.value.rejection is None


async def test_a_call_cut_off_after_a_rejected_attempt_raises_and_is_never_answered_with_it() -> None:
    with _fake_cli(*_cut_off_after_one_attempt()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        with pytest.raises(ModelProviderError) as raised:
            await _structured(model).ainvoke([HumanMessage(content="check the draft")])

    assert raised.value.reason == "error_max_turns", "a rejected attempt is not an answer"


async def test_a_streamed_call_cut_off_after_a_rejected_attempt_raises() -> None:
    with _fake_cli(*_cut_off_after_one_attempt()):
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
        with pytest.raises(ModelProviderError) as raised:
            async for _chunk in _structured(model).astream([HumanMessage(content="check the draft")]):
                pass

    assert raised.value.reason == "error_max_turns"


def _exhausted_on(*attempts: dict[str, Any]) -> tuple[Any, ...]:
    """a call whose every attempt the CLI rejected, ending on its attempt cap, as measured live.

    :param attempts: the ``StructuredOutput`` inputs the model sent, in order
    :ptype attempts: dict[str, Any]
    :return: the messages the CLI sends
    :rtype: tuple[Any, ...]
    """
    messages: list[Any] = []
    for number, attempt in enumerate(attempts, start=1):
        messages.append(_structured_call(f"toolu_{number}", attempt))
        messages.append(_tool_result(f"toolu_{number}", _MISMATCH, is_error=True))
    messages.append(_out_of_attempts()[-1])
    return tuple(messages)


def _cut_off_on(attempt: dict[str, Any]) -> tuple[Any, ...]:
    """a one-turn call whose only attempt was rejected: the turn limit ends it.

    :param attempt: the ``StructuredOutput`` input the model sent
    :ptype attempt: dict[str, Any]
    :return: the messages the CLI sends
    :rtype: tuple[Any, ...]
    """
    return (
        _structured_call("toolu_1", attempt),
        _tool_result("toolu_1", _MISMATCH, is_error=True),
        _cut_off_after_one_attempt()[-1],
    )


_WRAPPED = {"$PARAMETER_VALUE": json.dumps(_ANSWER)}


class TestAPlaceholderWrappedAnswerIsUnwrapped:
    """Found live (0.56.0): about one structured call in forty spent all five attempts sending the
    whole, correct answer as a JSON string under ``$PARAMETER_VALUE``, which the CLI rejects every
    time. That exact shape, when its JSON matches the call's schema, is the answer."""

    async def test_an_exhausted_call_whose_attempts_were_a_valid_wrapped_answer_answers_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with _fake_cli(*_exhausted_on(_WRAPPED, _WRAPPED, _WRAPPED, _WRAPPED, _WRAPPED)), caplog.at_level("WARNING"):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            message = await _structured(model).ainvoke([HumanMessage(content="check the draft")])

        assert json.loads(message.content) == _ANSWER
        assert message.response_metadata["structured_output_unwrapped"] == 1, "the recovery must be visible"
        assert message.response_metadata["is_error"] is False
        [logged] = [r for r in caplog.records if r.levelname == "WARNING"]
        assert "placeholder" in logged.getMessage(), "one warning, for the unwrap, and no failure logged"
        assert json.dumps(_ANSWER) not in json.dumps(logged.__dict__["extra_data"]), "the answer was logged"

    async def test_a_streamed_call_is_unwrapped_the_same_way(self) -> None:
        with _fake_cli(*_exhausted_on(_WRAPPED)):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            merged: Any = None
            async for chunk in _structured(model).astream([HumanMessage(content="check the draft")]):
                merged = chunk if merged is None else merged + chunk

        assert json.loads(merged.content) == _ANSWER
        assert merged.response_metadata["structured_output_unwrapped"] == 1

    async def test_a_one_turn_call_cut_off_on_a_valid_wrapped_answer_answers_it(self) -> None:
        with _fake_cli(*_cut_off_on(_WRAPPED)):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            message = await _structured(model.bind_tools([lookup])).ainvoke([HumanMessage(content="look it up")])

        assert json.loads(message.content) == _ANSWER

    @pytest.mark.parametrize("key", ["$PARAMETER_VALUE", "$PARAMETER_NAME", "$FUNCTION_NAME"])
    async def test_every_leaked_template_placeholder_is_unwrapped(self, key: str) -> None:
        """``$PARAMETER_NAME`` was the shape of the one call still failing after the first unwrap."""
        with _fake_cli(*_exhausted_on({key: json.dumps(_ANSWER)})):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            message = await _structured(model).ainvoke([HumanMessage(content="check the draft")])

        assert json.loads(message.content) == _ANSWER
        assert message.response_metadata["structured_output_unwrapped"] == 1

    async def test_the_latest_valid_wrapped_attempt_is_the_answer(self) -> None:
        earlier = {"sentences": [{"index": 1, "verdict": "cut"}], "overall": "rewrite"}
        with _fake_cli(*_exhausted_on({"$PARAMETER_VALUE": json.dumps(earlier)}, {"answer": 1}, _WRAPPED)):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            message = await _structured(model).ainvoke([HumanMessage(content="check the draft")])

        assert json.loads(message.content) == _ANSWER

    @pytest.mark.parametrize(
        "attempt",
        [
            pytest.param({"$PARAMETER_VALUE": json.dumps({"sentences": "none"})}, id="json-that-misses-the-schema"),
            pytest.param({"$PARAMETER_VALUE": "{'sentences': []"}, id="text-that-is-not-json"),
            pytest.param({"$PARAMETER_VALUE": _ANSWER}, id="a-value-that-is-not-a-string"),
            pytest.param({"$TOOL_INPUT": json.dumps(_ANSWER)}, id="a-key-outside-the-placeholder-family"),
            pytest.param(
                {"$PARAMETER_NAME": json.dumps({"sentences": "none"})}, id="parameter-name-missing-the-schema"
            ),
            pytest.param({"$FUNCTION_NAME": "not json"}, id="function-name-not-json"),
            pytest.param({"$PARAMETER_NAME": _ANSWER}, id="parameter-name-not-a-string"),
            pytest.param({"$PARAMETER_NAME": json.dumps(_ANSWER), "$FUNCTION_NAME": "x"}, id="two-placeholder-keys"),
            pytest.param({**_WRAPPED, "overall": "ready"}, id="extra-keys"),
            pytest.param({"sentences": {"sentences": []}}, id="an-extra-wrapping-key"),
        ],
    )
    async def test_anything_else_is_still_a_rejection(self, attempt: dict[str, Any]) -> None:
        with _fake_cli(*_exhausted_on(attempt)):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelProviderError) as raised:
                await _structured(model).ainvoke([HumanMessage(content="check the draft")])

        assert raised.value.reason == "error_max_structured_output_retries"
        assert raised.value.rejected_output == attempt

    async def test_a_call_with_no_schema_is_never_unwrapped(self) -> None:
        with _fake_cli(*_exhausted_on(_WRAPPED)):
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, TOKEN)
            with pytest.raises(ModelProviderError):
                await model.ainvoke([HumanMessage(content="hi")])
