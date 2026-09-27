"""Real structured calls through the real Claude CLI, on a subscription, and not one of them fails.

Every other test of the subscription route fakes the CLI's messages, so it can only replay a shape
somebody already saw. 0.55.0 shipped with about a third of one consumer's structured calls failing
with ``error_max_turns``: the model's first ``StructuredOutput`` call regularly misses the schema,
the CLI answers the mismatch and expects a retry in a second turn, and the call allowed one turn.
No faked sequence contained that first miss, because nobody knew the real model made it. Only a
real model shows what a real model does, so this batch runs one.

Opt-in, like the repo's other live tests: set ``THREETEARS_LIVE_CLAUDE_CLI=1``, and give it the
credential a deployment gives it, ``CLAUDE_CODE_OAUTH_TOKEN`` -- a subscription token from
``claude setup-token``. A logged-in ``claude`` on the host is not enough: 3tears runs the CLI in an
isolated configuration directory (``threetears.models.claude_cli_isolation``), where a stored login
is never read. Once opted in, a missing token or a missing ``claude-cli`` extra fails the run rather
than skipping it: a skip reads as a pass.

The batch is the three schema shapes the consumer reported -- an array of objects with enum fields,
an object of several fields, one string field -- 20 calls, six at a time, streamed and not.
``THREETEARS_LIVE_CLAUDE_CLI_MODEL`` picks the model (default
:data:`threetears.models.DEFAULT_CHAT_MODEL`, the sonnet model the consumer measured).

Run it with ``./scripts/test-live-claude-cli.sh``, which turns it on, cannot skip it, and records
the result for the release PR (``docs/releasing.md``, "Cutting a release").
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import jsonschema
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from threetears.models import DEFAULT_CHAT_MODEL

_ENABLED = os.environ.get("THREETEARS_LIVE_CLAUDE_CLI") == "1"
_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
_MODEL = os.environ.get("THREETEARS_LIVE_CLAUDE_CLI_MODEL", DEFAULT_CHAT_MODEL)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _ENABLED,
        reason="live Claude CLI calls on a subscription -- set THREETEARS_LIVE_CLAUDE_CLI=1 and "
        "CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`) to run",
    ),
]

#: Calls in flight at once.
_CONCURRENCY = 6

_SOURCE = (
    "The city council voted on Tuesday to extend the bike lane on Harbor Street by two miles. "
    "Supporters said the lane would reduce accidents near the school. "
    "Opponents argued that parking would be lost for nearby shops. "
    "The project is expected to cost 1.2 million dollars and finish next spring. "
    "Construction will happen at night to limit traffic delays."
)
_DRAFT = (
    "The council voted Tuesday to make the Harbor Street bike lane two miles longer. "
    "Supporters said the lane would reduce accidents near the school. "
    "Some shop owners are worried about losing parking. "
    "It will cost about a million dollars. "
    "Work is planned for nights so traffic is not blocked, and it should be done by next spring."
)

#: An array of objects with enum fields: the shape that failed most.
_CHECK_DRAFT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["keep", "revise", "cut"]},
                    "issue": {"type": "string", "enum": ["none", "copied", "unclear", "wrong", "tone"]},
                    "note": {"type": "string"},
                },
                "required": ["index", "verdict", "issue", "note"],
            },
        },
        "overall": {"type": "string", "enum": ["ready", "needs_work", "rewrite"]},
    },
    "required": ["sentences", "overall"],
}

#: An object of several fields.
_BRIEF: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "audience": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string", "enum": ["formal", "friendly", "neutral"]},
        "length_words": {"type": "integer"},
    },
    "required": ["title", "audience", "key_points", "tone", "length_words"],
}

#: One string field.
_OWN_WORDS: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

#: ``(name, schema, system prompt, request, calls)``. The array-of-objects shape gets most of the
#: batch: at 0.55.0 it failed about one call in three, so twelve calls without a failure is not luck.
_TASKS: list[tuple[str, dict[str, Any], str, str, int]] = [
    (
        "check_draft",
        _CHECK_DRAFT,
        "You are an editor. You check a draft against its source, sentence by sentence.",
        f"Source:\n{_SOURCE}\n\nDraft:\n{_DRAFT}\n\nCheck every sentence of the draft against the source. "
        "For each, give its index (from 1), a verdict, the issue, and a short note. Then an overall verdict.",
        12,
    ),
    (
        "brief",
        _BRIEF,
        "You are a writing assistant who plans short articles.",
        f"Plan a short article for local residents from this source:\n{_SOURCE}",
        4,
    ),
    (
        "own_words",
        _OWN_WORDS,
        "You are a writing assistant.",
        f"Restate this source in your own words, in two sentences, copying no phrase of it:\n{_SOURCE}",
        4,
    ),
]


async def _one_call(
    number: int, name: str, schema: dict[str, Any], system: str, request: str, gate: asyncio.Semaphore
) -> str | None:
    """one structured call, as a consumer makes it; what went wrong, or ``None``.

    Even-numbered calls are invoked and odd-numbered ones streamed, so both of the backend's query
    paths are measured.

    :param number: the call's place in the batch
    :ptype number: int
    :param name: the schema's name, for the report
    :ptype name: str
    :param schema: the JSON schema the answer must satisfy
    :ptype schema: dict[str, Any]
    :param system: the system prompt
    :ptype system: str
    :param request: the person's message
    :ptype request: str
    :param gate: bounds the calls in flight
    :ptype gate: asyncio.Semaphore
    :return: a description of the failure, or ``None`` for a valid answer
    :rtype: str | None
    """
    from threetears.models.factory import create_chat_model  # noqa: PLC0415
    from threetears.models.providers.structured_output import structured_output_kwargs  # noqa: PLC0415

    messages = [SystemMessage(content=system), HumanMessage(content=request)]
    failure: str | None = None
    async with gate:
        model = create_chat_model(_MODEL, api_key=_TOKEN, provider="anthropic", tools=[])
        bound = model.bind(**structured_output_kwargs("anthropic", schema))
        try:
            if number % 2 == 0:
                content = (await bound.ainvoke(messages)).content
            else:
                merged: Any = None
                async for chunk in bound.astream(messages):
                    merged = chunk if merged is None else merged + chunk
                content = merged.content
            jsonschema.validate(json.loads(content), schema)
        except Exception as exc:  # noqa: BLE001 -- every failure is collected and reported together
            failure = f"call {number} ({name}): {type(exc).__name__}: {exc}"
    return failure


async def test_a_batch_of_structured_calls_on_the_real_cli_all_answer_in_their_schema() -> None:
    if not _TOKEN:
        pytest.fail(
            "THREETEARS_LIVE_CLAUDE_CLI=1 but CLAUDE_CODE_OAUTH_TOKEN is not set: give it a subscription "
            "token from `claude setup-token`; a logged-in CLI's stored login is never read"
        )
    try:
        import claude_agent_sdk  # noqa: F401, PLC0415
        import langchain_claude_code  # noqa: F401, PLC0415
    except ImportError as exc:
        pytest.fail(f"THREETEARS_LIVE_CLAUDE_CLI=1 but the claude-cli extra is not installed: {exc}")

    gate = asyncio.Semaphore(_CONCURRENCY)
    calls = [(name, schema, system, request) for name, schema, system, request, count in _TASKS for _ in range(count)]
    outcomes = await asyncio.gather(*(_one_call(number, *call, gate) for number, call in enumerate(calls)))

    failures = [outcome for outcome in outcomes if outcome is not None]
    assert len(outcomes) == 20, "the batch ran every call"
    assert failures == [], f"{len(failures)} of {len(outcomes)} structured calls failed:\n" + "\n".join(failures)
