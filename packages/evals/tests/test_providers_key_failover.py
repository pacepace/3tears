"""A provider key whose credit is spent fails over to the next key ``<VARIABLE>_FALLBACKS`` names, over scripted HTTP.

Every test sends through ``httpx.MockTransport``, answering by the key each request carries, so which key a call went
out on is read off the request itself. Each test's keys are fresh values: a key spent in one test is spent for the
rest of the process, as it is for every model in a run, and a fresh value is a key no other test has spent.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import types
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from threetears.evals.providers import Catalogue, DeclinedRetry, Model, ProviderError, describe_failure
from threetears.evals.providers.failures import error_from_response
from threetears.evals.schema import TurnMessage, TurnToolSpec

PRIMARY: Final = "ANTHROPIC_API_KEY"
LUMENS: Final = "ANTHROPIC_API_KEY_LUMENS"
THIRD: Final = "ANTHROPIC_API_KEY_3"
FALLBACKS: Final = "ANTHROPIC_API_KEY_FALLBACKS"

#: Anthropic's answer when an account's prepaid credit is spent: a 400, not a 402.
CREDIT_SPENT: Final = (
    400,
    {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing.",
        },
    },
)

#: The one tool a scripted turn offers.
TOOLS: Final = (TurnToolSpec("evals_help", "Explain an action.", {"type": "object", "properties": {}}),)

#: The scripted model's row as the catalogue lists it, so a call is priced without a network read.
CATALOGUE: Final[dict[str, Any]] = {
    "data": [{"id": "anthropic/scripted-model", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}]
}


def reply(text: str = "hello") -> tuple[int, dict[str, Any]]:
    """An Anthropic Messages reply."""
    return 200, {
        "model": "scripted-model",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }


class KeyedServer:
    """A provider that answers by the key a request carries (Anthropic's ``x-api-key``, or the bearer token the OpenAI
    shape sends), and keeps which key each was sent on.

    :ivar sent: Each request's key and JSON body, in the order they arrived.
    """

    def __init__(self, answers: Mapping[str, tuple[int, Mapping[str, Any]]], *, hold: str | None = None) -> None:
        self._answers = dict(answers)
        #: A key whose requests are each held until two have arrived, so two calls meet its answer at once.
        self._hold = hold
        self._two_held = asyncio.Event()
        self._held = 0
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        key = request.headers.get("x-api-key") or request.headers["authorization"].removeprefix("Bearer ")
        self.sent.append((key, json.loads(request.content)))
        if key == self._hold:
            self._held += 1
            if self._held == 2:
                self._two_held.set()
            await self._two_held.wait()
        status, payload = self._answers[key]
        return httpx.Response(status, json=payload)

    @property
    def keys(self) -> list[str]:
        """The key each request was sent on, in order."""
        return [key for key, _ in self.sent]


def keys(monkeypatch: pytest.MonkeyPatch, *variables: str) -> dict[str, str]:
    """Set each variable to a fresh key, and the primary's ``_FALLBACKS`` to the others, in order; the keys by name."""
    values = {variable: f"sk-{variable.lower()}-{uuid.uuid4().hex}" for variable in variables}
    for variable, value in values.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv(FALLBACKS, ",".join(variables[1:]))
    return values


def opus(server: KeyedServer, /, **options: Any) -> Model:
    """A scripted model on Anthropic, sending through ``server``, its key read from the environment unless one is passed."""
    options.setdefault(
        "catalogue",
        Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=CATALOGUE)))),
    )
    return Model(
        "scripted-model",
        provider="anthropic",
        http=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        **options,
    )


def switches(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The WARNING lines that say a key failed over."""
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "failing over" in r.getMessage()]


async def test_a_spent_key_fails_over_to_the_next_and_the_same_request_is_resent_on_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    server = KeyedServer({key[PRIMARY]: CREDIT_SPENT, key[LUMENS]: reply("served on the fallback")})

    with caplog.at_level(logging.WARNING, logger="threetears.evals"):
        completion = await opus(server).generate(system="Be brief.", user="Say hello.")

    assert completion.content == "served on the fallback"
    assert server.keys == [key[PRIMARY], key[LUMENS]]
    assert server.sent[0][1] == server.sent[1][1], "the refused request is resent unchanged"
    [switch] = switches(caplog)
    assert PRIMARY in switch and LUMENS in switch
    assert all(value not in caplog.text for value in key.values()), "a log line names variables, never a key"


async def test_a_spent_key_is_not_sent_again_by_any_model_in_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    server = KeyedServer({key[PRIMARY]: CREDIT_SPENT, key[LUMENS]: reply()})
    await opus(server).generate(system="Be brief.", user="Say hello.")

    await opus(server).generate(system="Be brief.", user="Say hello again.")

    assert server.keys == [key[PRIMARY], key[LUMENS], key[LUMENS]]


async def test_two_calls_meeting_a_spent_key_at_once_advance_the_pool_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS, THIRD)
    server = KeyedServer({key[PRIMARY]: CREDIT_SPENT, key[LUMENS]: reply(), key[THIRD]: reply()}, hold=key[PRIMARY])
    first, second = opus(server), opus(server)

    with caplog.at_level(logging.WARNING, logger="threetears.evals"):
        await asyncio.gather(
            first.generate(system="Be brief.", user="One."), second.generate(system="Be brief.", user="Two.")
        )

    assert sorted(server.keys) == sorted([key[PRIMARY], key[PRIMARY], key[LUMENS], key[LUMENS]])
    assert key[THIRD] not in server.keys, "the second refusal of the same key moves nothing further"
    assert len(switches(caplog)) == 1


async def test_every_key_spent_is_the_account_failure_naming_every_variable_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS, THIRD)
    server = KeyedServer({value: CREDIT_SPENT for value in key.values()})

    with pytest.raises(ProviderError) as raised:
        await opus(server).generate(system="Be brief.", user="Say hello.")

    assert server.keys == [key[PRIMARY], key[LUMENS], key[THIRD]]
    assert raised.value.owner == "account"
    assert describe_failure(raised.value).owner == "account"
    said = str(raised.value)
    assert all(variable in said for variable in (PRIMARY, LUMENS, THIRD))
    assert all(value not in said for value in key.values())


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: too large"}}),
        (401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}),
        (403, {"type": "error", "error": {"type": "permission_error", "message": "not allowed"}}),
        (500, {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}),
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    ],
)
async def test_a_failure_that_is_no_spent_credit_never_fails_over(
    monkeypatch: pytest.MonkeyPatch, status: int, body: dict[str, Any]
) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    server = KeyedServer({key[PRIMARY]: (status, body), key[LUMENS]: reply()})

    with pytest.raises(ProviderError) as raised:
        await opus(server, retry=DeclinedRetry(attempts=1)).generate(system="Be brief.", user="Say hello.")

    assert raised.value.status == status
    assert set(server.keys) == {key[PRIMARY]}


async def test_a_timeout_never_fails_over(monkeypatch: pytest.MonkeyPatch) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    sent: list[str] = []

    def timing_out(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers["x-api-key"])
        raise httpx.ReadTimeout("no answer", request=request)

    model = Model(
        "scripted-model",
        provider="anthropic",
        http=httpx.AsyncClient(transport=httpx.MockTransport(timing_out)),
        catalogue=Catalogue(
            http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=CATALOGUE)))
        ),
    )

    with pytest.raises(httpx.ReadTimeout):
        await model.generate(system="Be brief.", user="Say hello.")

    assert sent == [key[PRIMARY]]


async def test_a_key_passed_as_api_key_never_fails_over(monkeypatch: pytest.MonkeyPatch) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    passed = f"sk-passed-{uuid.uuid4().hex}"
    server = KeyedServer({passed: CREDIT_SPENT, key[LUMENS]: reply()})

    with pytest.raises(ProviderError) as raised:
        await opus(server, api_key=passed).generate(system="Be brief.", user="Say hello.")

    assert server.keys == [passed]
    assert raised.value.owner == "account"
    assert LUMENS not in str(raised.value)


def test_a_fallback_naming_an_unset_variable_is_refused_when_the_model_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys(monkeypatch, PRIMARY, LUMENS)
    monkeypatch.setenv(FALLBACKS, f"{LUMENS},ANTHROPIC_API_KEY_UNSET")
    monkeypatch.delenv("ANTHROPIC_API_KEY_UNSET", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY_UNSET"):
        opus(KeyedServer({}))


async def test_a_tool_turn_fails_over_as_a_completion_does(monkeypatch: pytest.MonkeyPatch) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    server = KeyedServer({key[PRIMARY]: CREDIT_SPENT, key[LUMENS]: reply("turned")})

    turned = await opus(server).turn(
        system="Read the snapshot.", messages=[TurnMessage("user", "Begin.")], tools=TOOLS, max_tokens=1024
    )

    assert server.keys == [key[PRIMARY], key[LUMENS]]
    assert server.sent[0][1] == server.sent[1][1]
    assert turned.stop_reason == "end_turn"


async def test_no_key_and_no_variable_name_enters_a_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    monkeypatch.delenv("RECORD", raising=False)
    server = KeyedServer({key[PRIMARY]: CREDIT_SPENT, key[LUMENS]: reply("recorded")})
    completions, turns = tmp_path / "completions.jsonl", tmp_path / "turns.jsonl"
    model = opus(server, recording=completions, turn_recording=turns)

    completion = await model.generate(system="Be brief.", user="Say hello.")
    session = model.turn_session("analyst", snapshot="snap-0001")
    turned = await session.turn(
        system="Read the snapshot.", messages=[TurnMessage("user", "Begin.")], tools=TOOLS, max_tokens=1024
    )
    await model.aclose()

    assert server.keys == [key[PRIMARY], key[LUMENS], key[LUMENS]]
    written = completions.read_text(encoding="utf-8") + turns.read_text(encoding="utf-8")
    kept = written + repr(completion) + repr(turned)
    assert "recorded" in written
    for secret in (*key.values(), PRIMARY, LUMENS, FALLBACKS, "API_KEY"):
        assert secret not in kept


#: The rows the OpenAI-shaped tests' models are priced from.
ROUTER_CATALOGUE: Final[dict[str, Any]] = {
    "data": [
        {"id": "deepseek/deepseek-v4-flash", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
        {"id": "openai/gpt-5-nano", "pricing": {"prompt": "0.00000005", "completion": "0.0000004"}},
    ]
}


def chat(text: str = "hello") -> tuple[int, dict[str, Any]]:
    """An OpenAI-shaped reply, as OpenRouter and OpenAI answer."""
    return 200, {
        "model": "deepseek/deepseek-v4-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }


def routed(model_id: str, server: KeyedServer, /, **options: Any) -> Model:
    """A model on the OpenAI shape, sending through ``server``, its key read from the environment."""
    return Model(
        model_id,
        http=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        catalogue=Catalogue(
            http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ROUTER_CATALOGUE)))
        ),
        **options,
    )


def chain(monkeypatch: pytest.MonkeyPatch, primary: str, fallback: str) -> dict[str, str]:
    """Set ``primary`` and ``fallback`` to fresh keys, ``primary``'s ``_FALLBACKS`` naming ``fallback``."""
    values = {name: f"sk-{name.lower()}-{uuid.uuid4().hex}" for name in (primary, fallback)}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(f"{primary}_FALLBACKS", fallback)
    return values


async def test_openrouters_402_out_of_credit_fails_over(monkeypatch: pytest.MonkeyPatch) -> None:
    key = chain(monkeypatch, "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_SPARE")
    spent = (402, {"error": {"code": 402, "message": "Insufficient credits"}})
    server = KeyedServer({key["OPENROUTER_API_KEY"]: spent, key["OPENROUTER_API_KEY_SPARE"]: chat("on the spare")})

    completion = await routed("deepseek/deepseek-v4-flash", server).generate(system="Be brief.", user="Hi.")

    assert completion.content == "on the spare"
    assert server.keys == [key["OPENROUTER_API_KEY"], key["OPENROUTER_API_KEY_SPARE"]]


async def test_openais_insufficient_quota_never_fails_over(monkeypatch: pytest.MonkeyPatch) -> None:
    key = chain(monkeypatch, "OPENAI_API_KEY", "OPENAI_API_KEY_SPARE")
    quota = (429, {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota."}})
    server = KeyedServer({key["OPENAI_API_KEY"]: quota, key["OPENAI_API_KEY_SPARE"]: chat()})

    with pytest.raises(ProviderError) as raised:
        await routed("gpt-5-nano", server, provider="openai").generate(system="Be brief.", user="Hi.")

    assert raised.value.owner == "account"
    assert server.keys == [key["OPENAI_API_KEY"]]


@pytest.mark.parametrize(
    ("provider", "status", "body", "spent"),
    [
        ("anthropic", *CREDIT_SPENT, True),
        ("openrouter", 402, {"error": {"code": 402, "message": "Insufficient credits"}}, True),
        ("openai", 402, {"error": {"message": "Insufficient Balance"}}, True),
        ("anthropic", 400, {"error": {"type": "invalid_request_error", "message": "max_tokens: too large"}}, False),
        ("openrouter", 400, {"error": {"message": "Your credit balance is too low"}}, False),
        ("anthropic", 401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}}, False),
        ("anthropic", 403, {"error": {"type": "permission_error", "message": "not allowed"}}, False),
        ("openai", 429, {"error": {"code": "insufficient_quota", "message": "You exceeded your quota."}}, False),
        ("openrouter", 429, {"error": {"message": "Rate limit exceeded"}}, False),
        ("anthropic", 529, {"error": {"type": "overloaded_error", "message": "Overloaded"}}, False),
        ("anthropic", 500, {"error": {"type": "api_error", "message": "Internal server error"}}, False),
    ],
)
def test_credit_spent_is_the_credit_notice_and_a_402_and_nothing_else(
    provider: Any, status: int, body: dict[str, Any], spent: bool
) -> None:
    assert error_from_response(provider, status, json.dumps(body)).credit_spent is spent


@pytest.mark.parametrize(
    "listed", [PRIMARY, f"{LUMENS},{LUMENS}", f"{LUMENS}, {PRIMARY}"], ids=["itself", "twice", "itself-after"]
)
def test_a_fallback_list_naming_the_primary_or_one_variable_twice_is_refused_when_the_model_is_built(
    monkeypatch: pytest.MonkeyPatch, listed: str
) -> None:
    keys(monkeypatch, PRIMARY, LUMENS)
    monkeypatch.setenv(FALLBACKS, listed)

    with pytest.raises(ValueError, match=f"{FALLBACKS} names {PRIMARY} itself or one variable twice"):
        opus(KeyedServer({}))


def test_no_repr_or_str_of_a_model_with_fallbacks_or_of_anything_it_holds_prints_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS, THIRD)
    model = opus(KeyedServer({}))

    # Every engine object the model reaches, walked from the model itself: whatever holds the keys is among them.
    seen: set[int] = set()
    pending: list[object] = [model]
    printed: list[str] = []
    while pending:
        held = pending.pop()
        if id(held) in seen or isinstance(held, str | bytes | type | types.ModuleType | types.FunctionType):
            continue
        seen.add(id(held))
        if type(held).__module__.startswith("threetears.evals"):
            printed += [repr(held), str(held)]
        pending.extend(gc.get_referents(held))

    assert len(printed) > 2, "the walk reached what the model holds"
    assert all(value not in text for value in key.values() for text in printed)


async def test_a_turns_resend_on_the_next_key_is_recorded_as_a_resend_naming_why_never_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = keys(monkeypatch, PRIMARY, LUMENS)
    server = KeyedServer({key[PRIMARY]: CREDIT_SPENT, key[LUMENS]: reply("turned")})

    turned = await opus(server).turn(
        system="Read the snapshot.", messages=[TurnMessage("user", "Begin.")], tools=TOOLS, max_tokens=1024
    )

    [resent] = turned.resent_after
    assert "credit" in resent and "next key" in resent
    assert all(name not in resent for name in (*key, *key.values(), FALLBACKS))
