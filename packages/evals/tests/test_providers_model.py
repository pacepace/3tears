"""The completion client over scripted HTTP: routing, the three wires, stop reasons, pricing, refusals and replay.

Every test sends through ``httpx.MockTransport``: the request bodies a model POSTs are kept and read, and each answer
is the provider's own shape. The live behaviour against OpenRouter is ``test_providers_live.py``, replayed from its
recording.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from threetears.evals import Model as RootModel
from threetears.evals.providers import (
    Catalogue,
    Clients,
    Completion,
    Model,
    ProviderError,
    describe_failure,
)
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import (
    RecordedCompletions,
    RecordingRefused,
    check_completion_conformance,
    check_refusal_negotiation,
)

#: The catalogue every scripted model reads: three priced rows, a variant row and an alias, as OpenRouter lists them.
CATALOGUE: dict[str, Any] = {
    "data": [
        {
            "id": "deepseek/deepseek-v4-flash",
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "supported_parameters": ["max_tokens", "temperature", "reasoning", "response_format", "structured_outputs"],
            "top_provider": {"max_completion_tokens": 100000},
        },
        {
            "id": "z-ai/glm-5.3-flashx",
            "pricing": {"prompt": "0.0000004", "completion": "0.0000012"},
            "supported_parameters": ["max_tokens", "temperature", "reasoning", "response_format"],
        },
        {
            "id": "anthropic/claude-haiku-5.5",
            "pricing": {"prompt": "0.000001", "completion": "0.000005"},
            "supported_parameters": ["max_tokens", "reasoning", "response_format", "structured_outputs"],
        },
        {"id": "anthropic/claude-haiku-5.5:batch", "pricing": {"prompt": "0.0000005", "completion": "0.0000025"}},
        {"id": "openai/gpt-5-nano", "pricing": {"prompt": "0.00000005", "completion": "0.0000004"}},
        {"id": "~deepseek/deepseek-flash-latest", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
    ]
}

Answer = tuple[int, Mapping[str, Any]]


class Scripted:
    """A provider that answers each request in turn and keeps what it was sent.

    Attributes:
        sent: Each request: its URL, its headers and its JSON body.
    """

    def __init__(self, *answers: Answer | Callable[[Mapping[str, Any]], Answer]) -> None:
        self._answers = list(answers)
        self.sent: list[tuple[str, httpx.Headers, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.sent.append((str(request.url), request.headers, body))
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        status, payload = answer(body) if callable(answer) else answer
        return httpx.Response(status, json=payload)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        """The request bodies, in order."""
        return [body for _, _, body in self.sent]


def catalogue(rows: Mapping[str, Any] | None = None) -> Catalogue:
    """A catalogue read from ``rows`` (:data:`CATALOGUE` by default) over a scripted fetch."""
    payload = CATALOGUE if rows is None else rows
    return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))))


def model(model_id: str, server: Scripted, /, **options: Any) -> Model:
    """A model with a key, sending through ``server``, reading :data:`CATALOGUE` unless told otherwise."""
    options.setdefault("catalogue", catalogue())
    options.setdefault("api_key", "test-key")
    return Model(model_id, http=httpx.AsyncClient(transport=httpx.MockTransport(server)), **options)


def chat(
    content: str = "hello",
    *,
    finish: str | None = "stop",
    usage: Mapping[str, Any] | None = None,
    served: str = "deepseek/deepseek-v4-flash",
    **extra: Any,
) -> Answer:
    """An OpenAI-shaped reply (OpenRouter's, with ``extra`` at the top level)."""
    choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": content}}
    if finish is not None:
        choice["finish_reason"] = finish
    return 200, {
        "model": served,
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, **(usage or {})},
        **extra,
    }


def message(text: str = "hello", *, stop: str | None = "end_turn", usage: Mapping[str, Any] | None = None) -> Answer:
    """An Anthropic Messages reply."""
    return 200, {
        "model": "claude-haiku-5-5-20260901",
        "content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": text}],
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 4, **(usage or {})},
    }


class Triage(BaseModel):
    """What the json helper is asked for."""

    category: Literal["billing", "bug"]
    urgent: bool


# --- routing and refusals at construction ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "options", "url", "auth"),
    [
        ("deepseek/deepseek-v4-flash", {}, "https://openrouter.ai/api/v1/chat/completions", "authorization"),
        ("claude-haiku-5-5", {"provider": "anthropic"}, "https://api.anthropic.com/v1/messages", "x-api-key"),
        ("gpt-5-nano", {"provider": "openai"}, "https://api.openai.com/v1/chat/completions", "authorization"),
        (
            "deepseek-chat",
            {"base_url": "https://api.deepseek.com/v1/"},
            "https://api.deepseek.com/v1/chat/completions",
            "authorization",
        ),
    ],
)
async def test_an_id_goes_where_its_shape_or_provider_says(
    model_id: str, options: dict[str, Any], url: str, auth: str
) -> None:
    answer = message() if url.endswith("/messages") else chat()
    provider = Scripted(answer)
    await model(model_id, provider, **options).generate(system="", user="hi")
    [(sent_to, headers, _)] = provider.sent
    assert sent_to == url
    assert "test-key" in headers[auth]


def test_a_bare_id_names_its_provider_or_is_refused() -> None:
    with pytest.raises(ValueError, match="names no provider: pass provider='anthropic'"):
        Model("claude-haiku-5-5", api_key="k")


def test_a_missing_key_is_refused_at_construction_naming_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="set ANTHROPIC_API_KEY"):
        Model("claude-haiku-5-5", provider="anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert Model("claude-haiku-5-5", provider="anthropic").provider == "anthropic"


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        ({"effort": "extreme"}, "is not one of none, minimal"),
        ({"effort": 4096, "max_tokens": 4096}, "not below max_tokens"),
        ({"effort": 512, "provider": "anthropic"}, "below 1024"),
        ({"max_tokens": 0}, "not a positive number"),
        ({"provider": "bedrock"}, "is not one of openrouter"),
    ],
)
def test_the_callers_own_errors_are_refused_by_name(options: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Model("vendor/m", api_key="k", **options)


def test_the_package_root_exports_the_same_model() -> None:
    assert RootModel is Model


# --- what each wire carries --------------------------------------------------------------------------------------

STRICT = {
    "type": "json_schema",
    "json_schema": {"name": "Triage", "strict": True, "schema": {"type": "object", "properties": {}}},
}


async def test_openrouter_sends_a_strict_schema_enforced_with_reasoning_and_usage() -> None:
    provider = Scripted(chat("{}"))
    await model("deepseek/deepseek-v4-flash", provider, effort="low", temperature=0.2).generate(
        system="Be brief.", user="hi", response_format=STRICT
    )
    [body] = provider.bodies
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    assert body["response_format"] == {"type": "json_schema", "json_schema": STRICT["json_schema"]}
    assert body["provider"] == {"require_parameters": True}
    assert body["usage"] == {"include": True}
    assert body["reasoning"] == {"effort": "low"}
    assert (body["max_tokens"], body["temperature"]) == (8192, 0.2)


async def test_a_router_model_no_endpoint_of_which_takes_a_schema_is_sent_json_mode_unenforced() -> None:
    provider = Scripted(chat("{}"))
    completion = await model("z-ai/glm-5.3-flashx", provider).generate(system="", user="hi", response_format=STRICT)
    [body] = provider.bodies
    assert body["response_format"] == {"type": "json_object"}
    assert body["provider"] == {"require_parameters": False}
    assert completion.sent_settings is not None and completion.sent_settings.strict_output is False


async def test_openai_sends_its_own_cap_and_effort_and_a_compatible_endpoint_the_common_cap() -> None:
    direct, compatible = Scripted(chat()), Scripted(chat())
    await model("gpt-5-nano", direct, provider="openai", effort="minimal", max_tokens=500).generate(system="", user="x")
    await model("deepseek-chat", compatible, base_url="https://api.deepseek.com/v1").generate(system="", user="x")
    assert direct.bodies[0]["max_completion_tokens"] == 500 and direct.bodies[0]["reasoning_effort"] == "minimal"
    assert compatible.bodies[0]["max_tokens"] == 8192 and "max_completion_tokens" not in compatible.bodies[0]


async def test_a_claude_5_5_model_is_sent_no_temperature_and_its_effort_over_adaptive_thinking() -> None:
    provider = Scripted(message('{"category": "bug", "urgent": true}'))
    claude = model("claude-haiku-5-5", provider, provider="anthropic", effort="low", temperature=0.3)
    served = await claude.generate(system="Triage.", user="email", response_format=STRICT)
    [body] = provider.bodies
    assert "temperature" not in body and served.temperature is None
    assert body["system"] == "Triage." and body["max_tokens"] == 8192
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": {"type": "object", "properties": {}}},
    }
    assert served.content == '{"category": "bug", "urgent": true}'  # the text block only, never the thinking


async def test_an_anthropic_budget_goes_as_thinking_enabled() -> None:
    provider = Scripted(message())
    await model("claude-sonnet-4-6", provider, provider="anthropic", effort=2048, max_tokens=4096).generate(
        system="", user="x"
    )
    assert provider.bodies[0]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "system" not in provider.bodies[0]


async def test_a_reasoning_budget_goes_as_a_budget_where_the_provider_takes_one() -> None:
    """OpenRouter's ``reasoning.max_tokens`` (never beside ``effort``), Anthropic's thinking budget; OpenAI's chat
    completions take none, so the word goes and the completion names the budget unsent."""
    router, claude, direct = Scripted(chat()), Scripted(message()), Scripted(chat())
    routed = await model("deepseek/deepseek-v4-flash", router, effort=8192, max_tokens=10240).generate(
        system="", user="x"
    )
    assert router.bodies[0]["reasoning"] == {"max_tokens": 8192}
    assert routed.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192)
    thought = await model("claude-sonnet-4-6", claude, provider="anthropic", effort=8192, max_tokens=10240).generate(
        system="", user="x"
    )
    assert claude.bodies[0]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert claude.bodies[0]["max_tokens"] == 10240 and "output_config" not in claude.bodies[0]
    assert thought.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192)
    worded = await model("gpt-5-nano", direct, provider="openai", effort=8192, max_tokens=10240).generate(
        system="", user="x"
    )
    assert direct.bodies[0]["reasoning_effort"] == "high" and "reasoning" not in direct.bodies[0]
    assert worded.sent_settings == ClientRequestSettings(
        max_tokens=10240, reasoning_effort="high", unsent_reasoning_max_tokens=8192
    )


async def test_a_router_model_catalogued_without_reasoning_is_sent_no_budget_and_says_so() -> None:
    """The catalogue's refusal path still applies to a budget: no ``reasoning`` listed, none is sent, and named."""
    rows = {"data": [{"id": "vendor/plain", "supported_parameters": ["max_tokens", "temperature"]}]}
    provider = Scripted(chat())
    completion = await model(
        "vendor/plain", provider, effort=8192, max_tokens=10240, catalogue=catalogue(rows)
    ).generate(system="", user="x")
    assert "reasoning" not in provider.bodies[0]
    assert completion.sent_settings == ClientRequestSettings(max_tokens=10240, unsent_reasoning_max_tokens=8192)


def test_a_body_no_provider_could_take_as_built_is_refused_never_clamped_or_dropped() -> None:
    """An Anthropic budget its ``max_tokens`` does not exceed, named with both; a budget on OpenAI's wire."""
    from threetears.evals.negotiation import SettingFacts, WireSettings
    from threetears.evals.providers.wire import Directive, request_body

    def body(provider: Any, sent: WireSettings) -> dict[str, Any]:
        return request_body(
            provider,
            "m",
            system="",
            user="x",
            sent=sent,
            facts=SettingFacts(),
            directive=Directive(None),
            direct_openai=True,
            json_mode=False,
        )

    with pytest.raises(ValueError, match=r"max_tokens \(8192\) must exceed the thinking budget \(8192\)"):
        body("anthropic", WireSettings(max_tokens=8192, reasoning_max_tokens=8192))
    with pytest.raises(ValueError, match="take no reasoning budget"):
        body("openai", WireSettings(max_tokens=10240, reasoning_max_tokens=8192))
    both = body("openrouter", WireSettings(max_tokens=10240, reasoning_effort="low", reasoning_max_tokens=8192))
    assert both["reasoning"] == {"max_tokens": 8192}, "OpenRouter takes one: the budget wins"


async def test_a_directive_the_client_does_not_send_is_refused() -> None:
    with pytest.raises(ValueError, match="is not one the client sends"):
        await model("vendor/m", Scripted(chat())).generate(system="", user="x", response_format={"type": "xml"})


# --- what a reply says -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("finish", "native", "stop", "raw"),
    [
        ("stop", None, "end_turn", "stop"),
        ("length", None, "max_tokens", "length"),
        ("content_filter", None, "content_filter", "content_filter"),
        ("error", None, "error", "error"),
        ("stop", "end_turn", "end_turn", "end_turn"),
        (None, None, "truncated", None),
        ("something_new", None, "truncated", "something_new"),
    ],
)
async def test_an_openai_shaped_stop_maps_onto_the_engines_words(
    finish: str | None, native: str | None, stop: str, raw: str | None
) -> None:
    answer = chat(finish=finish)
    if native is not None:
        answer[1]["choices"][0]["native_finish_reason"] = native
    completion = await model("vendor/m", Scripted(answer)).generate(system="", user="x")
    assert (completion.stop_reason, completion.provider_stop_reason) == (stop, raw)


async def test_a_refusal_beside_empty_content_is_the_content_filter() -> None:
    answer = chat(finish="stop")
    answer[1]["choices"][0]["message"] = {"role": "assistant", "content": None, "refusal": "I can't."}
    completion = await model("vendor/m", Scripted(answer)).generate(system="", user="x")
    assert (completion.stop_reason, completion.content) == ("content_filter", "")


@pytest.mark.parametrize(
    ("raw", "stop"),
    [
        ("end_turn", "end_turn"),
        ("stop_sequence", "end_turn"),
        ("max_tokens", "max_tokens"),
        ("refusal", "content_filter"),
        ("pause_turn", "truncated"),
        ("model_context_window_exceeded", "truncated"),
        (None, "truncated"),
    ],
)
async def test_an_anthropic_stop_maps_onto_the_engines_words(raw: str | None, stop: str) -> None:
    completion = await model("claude-x", Scripted(message(stop=raw)), provider="anthropic").generate(
        system="", user="x"
    )
    assert (completion.stop_reason, completion.provider_stop_reason) == (stop, raw)


async def test_counts_are_none_when_unreported_and_cache_reads_count_as_input() -> None:
    bare = chat()
    bare[1]["usage"] = {}
    silent = await model("vendor/m", Scripted(bare)).generate(system="", user="x")
    assert (silent.input_tokens, silent.output_tokens, silent.reasoning_tokens) == (None, None, None)
    split = await model(
        "vendor/m", Scripted(chat(usage={"completion_tokens_details": {"reasoning_tokens": 3}}))
    ).generate(system="", user="x")
    assert split.reasoning_tokens == 3
    cached = message(usage={"cache_read_input_tokens": 100, "output_tokens_details": {"thinking_tokens": 2}})
    claude = await model("claude-haiku-5-5", Scripted(cached), provider="anthropic").generate(system="", user="x")
    assert (claude.input_tokens, claude.reasoning_tokens) == (110, 2)


async def test_an_alias_echoed_back_names_no_served_model_and_the_upstream_is_kept() -> None:
    alias = "~deepseek/deepseek-flash-latest"
    echoed = await model(alias, Scripted(chat(served=alias, provider="DeepInfra"))).generate(system="", user="x")
    named = await model("deepseek/deepseek-v4-flash", Scripted(chat(provider="DeepInfra"))).generate(
        system="", user="x"
    )
    assert (echoed.served_model, echoed.served_provider) == (None, "DeepInfra")
    assert named.served_model == "deepseek/deepseek-v4-flash"


async def test_a_completion_meets_the_port_and_reports_what_was_sent() -> None:
    completion = await model("vendor/m", Scripted(chat())).generate(system="", user="x")
    assert isinstance(completion, Completion)
    check_completion_conformance(completion, require_sent_settings=True)
    assert completion.sent_settings == ClientRequestSettings(max_tokens=8192, strict_output=False)


# --- pricing -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("usage", "cost", "source", "why"),
    [
        ({"cost": 0.0003}, 0.0003, "openrouter:reported", None),
        ({"cost": 0.0, "is_byok": False}, 0.0, "openrouter:reported", None),
        (
            {"cost": 0.0001, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.002}},
            0.0021,
            "openrouter:reported+upstream",
            None,
        ),
        ({"cost": 0.0001, "is_byok": True}, None, None, "no cost_details.upstream_inference_cost"),
        ({}, None, None, "carried no cost"),
    ],
)
async def test_openrouter_is_priced_by_its_own_figure_and_absent_is_unpriced_never_zero(
    usage: dict[str, Any], cost: float | None, source: str | None, why: str | None
) -> None:
    completion = await model("deepseek/deepseek-v4-flash", Scripted(chat(usage=usage))).generate(system="", user="x")
    assert completion.cost_usd == pytest.approx(cost) if cost is not None else completion.cost_usd is None
    assert completion.price_source == source
    assert (completion.pricing_failure is None) if why is None else why in (completion.pricing_failure or "")


async def test_a_direct_call_is_priced_by_the_matching_catalogue_row() -> None:
    plain = await model("claude-haiku-5-5", Scripted(message()), provider="anthropic").generate(system="", user="x")
    assert plain.cost_usd == pytest.approx(10 * 0.000001 + 4 * 0.000005)
    assert plain.price_source == "openrouter-catalogue:anthropic/claude-haiku-5.5"
    cached = message(usage={"cache_creation_input_tokens": 5})
    approx = await model("claude-haiku-5-5", Scripted(cached), provider="anthropic").generate(system="", user="x")
    assert approx.price_source == "openrouter-catalogue:anthropic/claude-haiku-5.5;cache-approximate"


async def test_a_direct_call_the_catalogue_cannot_price_is_returned_unpriced_saying_why() -> None:
    unlisted = await model("claude-opus-9", Scripted(message()), provider="anthropic").generate(system="", user="x")
    assert unlisted.cost_usd is None and "matches 'claude-opus-9'" in (unlisted.pricing_failure or "")

    def down(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    unread = Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(down)))
    answered = await model("claude-haiku-5-5", Scripted(message()), provider="anthropic", catalogue=unread).generate(
        system="", user="x"
    )
    assert answered.content == "hello" and answered.cost_usd is None
    assert "could not be read (HTTPStatusError" in (answered.pricing_failure or "")


class CatalogueServer:
    """A catalogue endpoint answering each fetch in turn (the last answer repeating), counting the fetches."""

    def __init__(self, *answers: tuple[int, Mapping[str, Any]]) -> None:
        self._answers = list(answers)
        self.fetches = 0

    def __call__(self, _: httpx.Request) -> httpx.Response:
        self.fetches += 1
        status, payload = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return httpx.Response(status, json=payload)

    def catalogue(self, refresh_after: timedelta) -> Catalogue:
        return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(self)), refresh_after=refresh_after)


def _repriced(rate: str) -> dict[str, Any]:
    """:data:`CATALOGUE` with the Haiku row's input rate moved to ``rate``."""
    rows = [
        {**row, "pricing": {**row["pricing"], "prompt": rate}} if row["id"] == "anthropic/claude-haiku-5.5" else row
        for row in CATALOGUE["data"]
    ]
    return {"data": rows}


async def test_the_catalogue_refreshes_on_the_first_call_after_it_expires_and_not_before() -> None:
    kept = CatalogueServer((200, CATALOGUE), (200, _repriced("0.000002")))
    haiku = model(
        "claude-haiku-5-5", Scripted(message()), provider="anthropic", catalogue=kept.catalogue(timedelta(hours=6))
    )
    first = await haiku.generate(system="", user="x")
    second = await haiku.generate(system="", user="x")
    assert kept.fetches == 1 and first.cost_usd == second.cost_usd == pytest.approx(10 * 0.000001 + 4 * 0.000005)

    expiring = CatalogueServer((200, CATALOGUE), (200, _repriced("0.000002")))
    shared = expiring.catalogue(timedelta(0))
    haiku = model("claude-haiku-5-5", Scripted(message()), provider="anthropic", catalogue=shared)
    before = await haiku.generate(system="", user="x")
    after = await haiku.generate(system="", user="x")
    assert expiring.fetches == 2, "an expired catalogue is fetched again on the next call"
    assert before.cost_usd == pytest.approx(10 * 0.000001 + 4 * 0.000005)
    assert after.cost_usd == pytest.approx(10 * 0.000002 + 4 * 0.000005), "the refreshed rates price the call"
    assert before.pricing_stale_since is None and after.pricing_stale_since is None and shared.stale_since is None


async def test_a_failed_refresh_keeps_the_last_rates_and_marks_the_pricing_stale_never_failing_the_call() -> None:
    down = (503, {"error": "down"})
    server = CatalogueServer((200, CATALOGUE), down, down, (200, _repriced("0.000002")))
    shared = server.catalogue(timedelta(0))
    haiku = model("claude-haiku-5-5", Scripted(message()), provider="anthropic", catalogue=shared)
    fresh = await haiku.generate(system="", user="x")
    read_by = datetime.now(UTC)

    stale = await haiku.generate(system="", user="x")

    assert stale.content == "hello" and stale.cost_usd == fresh.cost_usd, "the last rates price it"
    assert stale.pricing_stale_since is not None and stale.pricing_stale_since <= read_by
    assert stale.pricing_stale_since == shared.stale_since
    assert "could not be refreshed (HTTPStatusError" in (shared.refresh_failure or "")
    router = await model(
        "deepseek/deepseek-v4-flash", Scripted(chat(usage={"cost": 0.001})), catalogue=shared
    ).generate(system="", user="x")
    assert shared.stale_since is not None, "the second refresh failed too"
    assert router.pricing_stale_since is None, "OpenRouter's own figure priced it, not the stale rates"

    recovered = await haiku.generate(system="", user="x")
    assert server.fetches == 4 and recovered.pricing_stale_since is None and shared.stale_since is None
    assert shared.refresh_failure is None
    assert recovered.cost_usd == pytest.approx(10 * 0.000002 + 4 * 0.000005)


def test_a_negative_refresh_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="refresh_after .* is negative"):
        Catalogue(refresh_after=timedelta(seconds=-1))


async def test_the_price_ceiling_answers_once_the_catalogue_is_loaded() -> None:
    shared = catalogue()
    deepseek = model("deepseek/deepseek-v4-flash", Scripted(chat()), catalogue=shared, max_tokens=1000)
    assert deepseek.price_ceiling(system="", user="hi") is None
    await shared.load()
    ceiling = deepseek.price_ceiling(system="", user="hi")
    assert ceiling == pytest.approx((2 + 2048) * 0.0000001 + 1000 * 0.0000002)


# --- refusals and failures ---------------------------------------------------------------------------------------

NO_ENDPOINT: Answer = (
    404,
    {"error": {"code": 404, "message": "No endpoints found that can handle the requested parameters."}},
)


async def test_a_router_refusal_is_resent_without_the_temperature_and_learned_across_the_catalogue() -> None:
    shared = catalogue({"data": []})

    def router(body: Mapping[str, Any]) -> Answer:
        return NO_ENDPOINT if "temperature" in body else chat("{}")

    provider = Scripted(router)
    first = await model("openai/gpt-5-nano", provider, temperature=0.2, catalogue=shared).generate(
        system="", user="x", response_format=STRICT
    )
    assert ["temperature" in body for body in provider.bodies] == [True, False]
    assert first.temperature is None and first.sent_settings is not None and first.sent_settings.strict_output
    second = await model("openai/gpt-5-nano", provider, temperature=0.2, catalogue=shared).generate(
        system="", user="x", response_format=STRICT
    )
    assert len(provider.bodies) == 3 and "temperature" not in provider.bodies[2] and second.temperature is None


async def test_the_client_negotiates_a_routers_refusals_as_the_conformance_kit_scripts_them() -> None:
    def build(transport: Callable[[Mapping[str, object]], Any]) -> Model:
        async def handler(request: httpx.Request) -> httpx.Response:
            status, payload = await transport(json.loads(request.content))
            return httpx.Response(status, json=dict(payload))

        return Model(
            "vendor/model",
            api_key="k",
            max_tokens=10240,
            effort="high",
            temperature=0.2,
            catalogue=catalogue({"data": []}),
            http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    await check_refusal_negotiation(build)


@pytest.mark.parametrize(
    ("provider_name", "model_id", "refused"),
    [
        (
            "openai",
            "gpt-5-nano",
            (
                400,
                {
                    "error": {
                        "message": "Unsupported value: 'temperature'",
                        "param": "temperature",
                        "code": "unsupported_value",
                    }
                },
            ),
        ),
        (
            "anthropic",
            "claude-sonnet-4-6",
            (
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "`temperature` is deprecated for this model.",
                    },
                },
            ),
        ),
    ],
)
async def test_a_direct_providers_refusal_of_the_temperature_is_resent_without_it(
    provider_name: str, model_id: str, refused: Answer
) -> None:
    served = message() if provider_name == "anthropic" else chat()
    provider = Scripted(lambda body: refused if "temperature" in body else served)
    completion = await model(model_id, provider, provider=provider_name, temperature=0.5).generate(system="", user="x")
    assert len(provider.bodies) == 2 and completion.temperature is None


async def test_any_other_failure_is_raised_naming_the_provider_and_never_its_envelope() -> None:
    envelope = {"error": {"code": 402, "message": "Insufficient credits", "metadata": {"account_id": "acct-SECRET"}}}
    with pytest.raises(ProviderError) as raised:
        await model("vendor/m", Scripted((402, envelope))).generate(system="", user="x")
    assert str(raised.value) == "openrouter 402: Insufficient credits"
    assert raised.value.account and "acct-SECRET" not in str(raised.value)
    failure = describe_failure(raised.value)
    assert (failure.description, failure.payload_withheld, failure.account_refused) == (str(raised.value), False, True)


async def test_an_error_in_a_200_body_is_a_failure_not_a_reply() -> None:
    with pytest.raises(ProviderError, match="openrouter 502: upstream died"):
        await model("vendor/m", Scripted((200, {"error": {"code": 502, "message": "upstream died"}}))).generate(
            system="", user="x"
        )


@pytest.mark.parametrize(
    ("provider_name", "status", "body", "account"),
    [
        ("openrouter", 401, {"error": {"message": "bad key"}}, True),
        ("openrouter", 403, {"error": {"message": "flagged by moderation"}}, False),
        (
            "anthropic",
            400,
            {"error": {"type": "invalid_request_error", "message": "Your credit balance is too low"}},
            True,
        ),
        ("openai", 429, {"error": {"message": "quota", "code": "insufficient_quota"}}, True),
        ("openai", 429, {"error": {"message": "slow down", "code": "rate_limit_exceeded"}}, False),
        ("openai", 500, "<html>oops</html>", False),
    ],
)
def test_the_account_flag_is_set_where_every_model_behind_the_key_would_be_refused(
    provider_name: Any, status: int, body: object, account: bool
) -> None:
    from threetears.evals.providers.failures import error_from_response

    raw = body if isinstance(body, str) else json.dumps(body)
    error = error_from_response(provider_name, status, raw)
    assert error.account is account
    assert describe_failure(error).account_refused is account


def test_describe_failure_is_total_and_withholds_what_it_cannot_vouch_for() -> None:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    assert describe_failure(httpx.ConnectTimeout("slow", request=request)).description == (
        "the request to openrouter.ai timed out (ConnectTimeout)"
    )
    try:
        Triage.model_validate_json('{"category": "sales", "urgent": true}')
    except ValidationError as invalid:
        validation = describe_failure(invalid)
    assert validation.payload_withheld and "sales" not in validation.description
    assert describe_failure(RuntimeError("secret body")).description.startswith("RuntimeError (its text is withheld")


# --- the runner helpers ------------------------------------------------------------------------------------------


async def test_json_sends_the_schema_strict_and_validates_the_reply() -> None:
    provider = Scripted(chat('{"category": "bug", "urgent": false}'))
    answer = await model("deepseek/deepseek-v4-flash", provider).json("Triage this.", Triage)
    assert answer == Triage(category="bug", urgent=False)
    sent = provider.bodies[0]["response_format"]["json_schema"]
    assert sent["strict"] is True and sent["schema"]["additionalProperties"] is False
    assert sent["schema"]["required"] == ["category", "urgent"]


async def test_json_refuses_a_reply_that_is_not_the_schema_with_pydantics_message() -> None:
    with pytest.raises(ValidationError, match="category"):
        await model("vendor/m", Scripted(chat('{"category": "sales", "urgent": true}'))).json("x", Triage)


async def test_a_reply_that_did_not_finish_is_refused_naming_its_stop() -> None:
    with pytest.raises(ValueError, match=r"stopped at max_tokens \(length\)"):
        await model("vendor/m", Scripted(chat("half", finish="length"))).text("x")


async def test_prompt_fills_the_case_and_refuses_a_field_it_lacks() -> None:
    provider = Scripted(chat("an answer"))
    runner = model("vendor/m", provider).prompt("Triage this email:\n\n{email} {{literal}}")
    assert runner.__doc__ == "Triage this email:"
    assert await runner({"email": "my card was charged twice"}) == "an answer"
    assert provider.bodies[0]["messages"][-1]["content"] == "Triage this email:\n\nmy card was charged twice {literal}"
    with pytest.raises(KeyError, match="names email, which the case lacks"):
        await runner({"subject": "x"})
    with pytest.raises(ValueError, match="not a plain field name"):
        model("vendor/m", provider).prompt("{email.subject}")


# --- the role factory --------------------------------------------------------------------------------------------


async def test_clients_builds_each_role_once_sharing_one_catalogue_and_asks_a_judge_as_the_run_records() -> None:
    provider = Scripted(chat())
    template = model("deepseek/deepseek-v4-flash", provider)
    clients = Clients(default=template, judge="z-ai/glm-5.3-flashx", catalogue=catalogue())
    analysis = clients("analysis", None)
    assert analysis.model_name == "deepseek/deepseek-v4-flash"
    await analysis.generate(system="", user="x")
    await analysis.aclose()  # releases nothing
    await clients("analysis", None).generate(system="", user="y")
    assert len(provider.bodies) == 2
    with pytest.raises(ValueError, match="is not a role"):
        Clients(critic="vendor/m")
    with pytest.raises(ValueError, match="named no model for 'judge'"):
        Clients()("judge", None)
    unpriced = await clients.load_prices()
    assert unpriced == {}
    await clients.aclose()


async def test_a_judge_named_by_id_is_built_from_the_judges_request_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    provider = Scripted(chat())
    http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    clients = Clients(judge="vendor/judge", catalogue=catalogue({"data": []}), http=http)
    judge = clients("judge", None, temperature=0.0)
    assert judge.price_ceiling(system="", user="x") is None  # the catalogue is not loaded yet
    completion = await judge.generate(system="", user="x")
    [body] = provider.bodies
    # JUDGE_REQUEST_SETTINGS: a 10240 cap and an 8192 budget, which a router takes as a budget.
    assert (body["max_tokens"], body["reasoning"], body["temperature"]) == (10240, {"max_tokens": 8192}, 0.0)
    assert completion.sent_settings == ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192)
    unpriced = await Clients(judge="vendor/unlisted", catalogue=catalogue()).load_prices()
    assert "vendor/unlisted" in unpriced


# --- recording ---------------------------------------------------------------------------------------------------


async def test_a_recording_is_made_live_once_then_replayed_with_no_key_and_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECORD", raising=False)
    path = tmp_path / "recordings" / "triage.jsonl"
    live = model("deepseek/deepseek-v4-flash", Scripted(chat('{"category": "bug", "urgent": true}')), recording=path)
    assert await live.json("Triage.", Triage) == Triage(category="bug", urgent=True)
    assert path.exists()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    replayed = Model("deepseek/deepseek-v4-flash", recording=path)
    assert await replayed.json("Triage.", Triage) == Triage(category="bug", urgent=True)
    assert replayed.price_ceiling(system="", user="x") == 0.0
    with pytest.raises(RecordingRefused, match="no call is recorded"):
        await replayed.text("something else")
    with pytest.raises(RecordingRefused):
        await Model("deepseek/deepseek-v4-flash", recording=path, effort="high").json("Triage.", Triage)
    with pytest.raises(RecordingRefused, match="holds calls of model 'deepseek/deepseek-v4-flash'"):
        Model("vendor/other", recording=path)  # a recording holds one model's calls


async def test_record_starts_a_stale_recording_afresh_and_adds_to_one_written_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "triage.jsonl"
    monkeypatch.delenv("RECORD", raising=False)
    await model("vendor/m", Scripted(chat("old")), recording=path).text("x")
    os.utime(path, (time.time() - 3600, time.time() - 3600))
    monkeypatch.setenv("RECORD", "1")
    await model("vendor/m", Scripted(chat("new")), recording=path).text("x")
    await model("vendor/m", Scripted(chat("newer")), recording=path).text("x")
    monkeypatch.delenv("RECORD")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["result"]["content"] for line in lines] == ["new", "newer"]
    replay = RecordedCompletions.replay(path, settings=ClientRequestSettings(max_tokens=8192))
    assert (await replay.generate(system="", user="x")).content == "new"
