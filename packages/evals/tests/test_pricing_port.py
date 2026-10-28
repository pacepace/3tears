"""The pricing port: one injected pricer bounds every call before it is sent and prices it after it returns.

The host owns prices and the engine owns the arithmetic. A :class:`~threetears.evals.providers.Pricer` answers the
rates a call is billed at before it is sent (``rates_for``), from which the engine computes the bound, and prices the
reply after it returns (``price``), from the raw usage block the wire returned. A pricer that answers no rates leaves
the call to the default, end to end: Anthropic's published rates for a model handed neither a pricer nor a catalogue,
then OpenRouter's reported cost for an OpenRouter call, then the catalogue.

**The default is pinned byte for byte first.** Every recorded completion and turn under ``fixtures/providers/`` is sent
through ``httpx.MockTransport`` on each default route, and its ``cost_usd``, ``price_source`` and bound are compared
with ``==`` against what the client answered before the port existed. A turn fixture's reply is a Messages API body,
so it is a completion's reply too; a recorded OpenRouter completion holds its result, so its reply is rebuilt from it
(the tokens, the served upstream, and the cost OpenRouter reported). Nothing here reaches a live API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from threetears.evals.providers import Catalogue, Model, Provider
from threetears.evals.schema import TurnMessage, TurnToolSpec

FIXTURES: Final = Path(__file__).parent / "fixtures" / "providers"

#: The recorded tool turns, by case: each one's request body and the Messages API reply it was answered with.
TURNS: Final[dict[str, dict[str, Any]]] = {
    case["case"]: case for case in map(json.loads, (FIXTURES / "turns.jsonl").read_text(encoding="utf-8").splitlines())
}

#: The recorded OpenRouter completions, by fixture: each one's stored result and what it carried beside it.
COMPLETIONS: Final[dict[str, dict[str, Any]]] = {
    name: json.loads((FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for name in ("plain", "priced", "json_schema", "negotiation")
}

MODEL: Final = "claude-opus-5-5"
SYSTEM: Final = "You are the analyst. Read the snapshot through the tool, then submit the report."
OPENING: Final = "Write the report for snapshot snap-0001."
TOOLS: Final = (
    TurnToolSpec(
        "evals_analyst",
        "Read the frozen snapshot: one action per call.",
        {
            "type": "object",
            "properties": {"action": {"type": "string"}, "snapshot": {"type": "string"}},
            "required": ["action"],
        },
    ),
    TurnToolSpec("evals_help", "Explain an action.", {"type": "object", "properties": {"topic": {"type": "string"}}}),
)

#: A reply writing to both cache lifetimes, split by TTL as Anthropic reports it.
SPLIT_USAGE: Final[dict[str, Any]] = {
    "input_tokens": 1_200,
    "output_tokens": 300,
    "cache_read_input_tokens": 40_000,
    "cache_creation_input_tokens": 6_000,
    "cache_creation": {"ephemeral_5m_input_tokens": 5_000, "ephemeral_1h_input_tokens": 1_000},
    "output_tokens_details": {"thinking_tokens": 120},
}

#: The rows a host's catalogue lists for every model the pins send to: as OpenRouter lists them.
ROWS: Final[dict[str, Any]] = {
    "data": [
        {
            "id": "anthropic/claude-opus-5.5",
            "pricing": {
                "prompt": "0.000004",
                "completion": "0.00002",
                "input_cache_read": "0.0000004",
                "input_cache_write": "0.000005",
            },
        },
        {"id": "deepseek/deepseek-v4-flash", "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}},
        {"id": "openai/gpt-5-nano", "pricing": {"prompt": "0.00000005", "completion": "0.0000004"}},
    ]
}


def catalogue() -> Catalogue:
    """A host's catalogue, read from :data:`ROWS` over a scripted fetch."""
    return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=ROWS))))


def answering(reply: Mapping[str, Any]) -> httpx.AsyncClient:
    """A provider answering every request with ``reply``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=dict(reply))))


def anthropic_body(usage: Mapping[str, Any]) -> dict[str, Any]:
    """A Messages API reply carrying ``usage``."""
    return {
        "model": "claude-opus-5-5-20261001",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": dict(usage),
    }


def router_body(recorded: Mapping[str, Any], *, byok: bool) -> dict[str, Any]:
    """The OpenRouter reply a recorded completion was read from: its tokens, its upstream and its reported cost.

    Under bring-your-own-key the recorded cost is split into OpenRouter's fee and the upstream's charge, which the
    reply states apart, in two figures that each read exactly as written.
    """
    result = recorded["result"]
    usage: dict[str, Any] = {
        "prompt_tokens": result["input_tokens"],
        "completion_tokens": result["output_tokens"],
        "completion_tokens_details": {"reasoning_tokens": result["reasoning_tokens"]},
        "cost": result["cost_usd"],
        "is_byok": byok,
    }
    if byok:
        usage["cost"] = 0.00001
        usage["cost_details"] = {"upstream_inference_cost": 0.00002}
    return {
        "model": result["served_model"],
        "provider": recorded["extra"]["served_provider"],
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


#: Each default route: what the model is handed. ``published`` is the report writer's command line (no pricer, no
#: catalogue); ``catalogue`` is a host's ``Clients`` shape (a catalogue, no pricer).
ROUTES: Final = ("published", "catalogue")


def opus(route: str, reply: Mapping[str, Any]) -> Model:
    """Opus 5.5 on Anthropic at high effort, on ``route``, answering every request with ``reply``."""
    options: dict[str, Any] = {"catalogue": catalogue()} if route == "catalogue" else {}
    return Model(MODEL, provider="anthropic", api_key="test-key", effort="high", http=answering(reply), **options)


#: What the published rates and the catalogue row name a call they price as.
PUBLISHED: Final = "anthropic-published:2026-10-28"
CATALOGUED: Final = "openrouter-catalogue:anthropic/claude-opus-5.5"

#: What each recorded turn's ceiling, cost and source were, on each default route, before the port existed.
TURN_PINS: Final[dict[tuple[str, str], tuple[float | None, float | None, str | None]]] = {
    ("cut_short", "published"): (0.653005, 0.6458, PUBLISHED),
    ("cut_short", "catalogue"): (0.653005, 0.6468, CATALOGUED),
    ("opening", "published"): (0.653005, 0.0358, PUBLISHED),
    ("opening", "catalogue"): (0.653005, 0.0358, CATALOGUED),
    ("tool_results", "published"): (0.653005, 0.0132, PUBLISHED),
    ("tool_results", "catalogue"): (0.653005, 0.0142, CATALOGUED),
}

#: What each recorded reply's completion ceiling, cost and source were, on each default route, before the port.
COMPLETION_PINS: Final[dict[tuple[str, str], tuple[float | None, float | None, str | None]]] = {
    ("cut_short", "published"): (0.17409, 0.6458, PUBLISHED),
    ("cut_short", "catalogue"): (0.17409, 0.6468, CATALOGUED),
    ("opening", "published"): (0.17409, 0.0358, PUBLISHED),
    ("opening", "catalogue"): (0.17409, 0.0358, CATALOGUED),
    ("tool_results", "published"): (0.17409, 0.0132, PUBLISHED),
    ("tool_results", "catalogue"): (0.17409, 0.0142, CATALOGUED),
    ("split", "published"): (0.17409, 0.0518, PUBLISHED),
    ("split", "catalogue"): (0.17409, 0.0558, CATALOGUED + ";cache-approximate"),
}

#: What each recorded OpenRouter completion was priced at, plain and under bring-your-own-key.
ROUTER_PINS: Final[dict[tuple[str, bool], tuple[float | None, float | None, str | None]]] = {
    ("json_schema", False): (0.000405, 3.24471e-05, "openrouter:reported"),
    ("json_schema", True): (0.000405, 3.0000000000000004e-05, "openrouter:reported+upstream"),
    ("negotiation", False): (0.0009025, 1.63e-05, "openrouter:reported"),
    ("negotiation", True): (0.0009025, 3.0000000000000004e-05, "openrouter:reported+upstream"),
    ("plain", False): (0.000405, 1.176e-05, "openrouter:reported"),
    ("plain", True): (0.000405, 3.0000000000000004e-05, "openrouter:reported+upstream"),
    ("priced", False): (0.000405, 1.204e-05, "openrouter:reported"),
    ("priced", True): (0.000405, 3.0000000000000004e-05, "openrouter:reported+upstream"),
}

#: A direct OpenAI completion, priced from the catalogue.
OPENAI_PIN: Final[tuple[float | None, float | None, str | None]] = (
    0.0033793,
    1.63e-05,
    "openrouter-catalogue:openai/gpt-5-nano",
)


def ceiling_usd(ceiling: object) -> float | None:
    """A ceiling as dollars: the bound as the client answered it."""
    assert ceiling is None or isinstance(ceiling, float), ceiling
    return ceiling


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", sorted(TURNS))
async def test_the_default_prices_and_bounds_every_recorded_turn_as_before(route: str, case: str) -> None:
    model = opus(route, TURNS[case]["reply"])
    await model.refresh_prices()
    asked = [TurnMessage("user", OPENING)]

    ceiling = model.turn_price_ceiling(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000)
    reply = await model.turn(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000)

    assert (ceiling_usd(ceiling), reply.cost_usd, reply.price_source) == TURN_PINS[(case, route)]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", [*sorted(TURNS), "split"])
async def test_the_default_prices_and_bounds_every_recorded_anthropic_reply_as_a_completion_as_before(
    route: str, case: str
) -> None:
    body = anthropic_body(SPLIT_USAGE) if case == "split" else TURNS[case]["reply"]
    model = opus(route, body)
    await model.refresh_prices()

    ceiling = model.price_ceiling(system="s", user="u")
    completion = await model.generate(system="s", user="u")

    assert (ceiling, completion.cost_usd, completion.price_source) == COMPLETION_PINS[(case, route)]


@pytest.mark.parametrize("byok", [False, True])
@pytest.mark.parametrize("name", sorted(COMPLETIONS))
async def test_the_default_prices_every_recorded_openrouter_completion_as_recorded(name: str, byok: bool) -> None:
    recorded = COMPLETIONS[name]
    model = Model(
        recorded["result"]["model"],
        api_key="test-key",
        catalogue=catalogue(),
        max_tokens=recorded["settings"]["max_tokens"],
        http=answering(router_body(recorded, byok=byok)),
    )
    await model.refresh_prices()

    ceiling = model.price_ceiling(system="s", user="u")
    completion = await model.generate(system="s", user="u")

    assert (ceiling, completion.cost_usd, completion.price_source) == ROUTER_PINS[(name, byok)]
    if not byok:
        assert (completion.cost_usd, completion.price_source) == (
            recorded["result"]["cost_usd"],
            recorded["result"]["price_source"],
        ), "the recorded figure, exactly"


async def test_the_default_prices_a_direct_openai_completion_from_the_catalogue_as_before() -> None:
    provider: Provider = "openai"
    body = {
        "model": "gpt-5-nano",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 78, "completion_tokens": 31},
    }
    model = Model("gpt-5-nano", provider=provider, api_key="test-key", catalogue=catalogue(), http=answering(body))
    await model.refresh_prices()

    ceiling = model.price_ceiling(system="s", user="u")
    completion = await model.generate(system="s", user="u")

    assert (ceiling, completion.cost_usd, completion.price_source) == OPENAI_PIN
