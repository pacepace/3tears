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
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_doc_support import snapshot_of
from packages.evals.tests.report_writer_support import DOC
from packages.evals.tests.stored_models import stored_sample
from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.analysis.report_doc_store import generation_record, made_line
from threetears.evals.analysis.summary import EvalSummary, RoleSpend, dollars_text
from threetears.evals.kernel import OutOfRunBudget, PlannedCall, RoleUsageLedger, ValidationFailedError
from threetears.evals.kernel.capped_completion import (
    call_ceiling,
    ceiling_is_estimate,
    prompt_token_bound,
    turn_prompt_token_bound,
)
from threetears.evals.ops.lenses import ArmEstimate, LaunchEstimate, estimate_text
from threetears.evals.kernel.report_doc import EvalReportDoc
from threetears.evals.ops.report_writer.session import ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally, TurnCost
from threetears.evals.providers import (
    AnthropicPublishedRates,
    CallPrice,
    Catalogue,
    Clients,
    HostRates,
    Model,
    ModelRates,
    PriceSource,
    Provider,
)
from threetears.evals.providers.model import DEFAULT_MAX_TOKENS
from threetears.evals.providers.wire import usage_counts
from threetears.evals.run import CallCeiling
from threetears.evals.schema import EvalRun, OutOfRunSpend, RoleUsage, TurnCeiling, TurnMessage, TurnToolSpec

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


def ceiling_usd(ceiling: TurnCeiling | None, route: str) -> float | None:
    """A turn's ceiling as dollars: the bound as the client answered it. Anthropic's published rates are what it bills
    a direct call, so their bound is no estimate; a catalogue row is a third party's listing, so its bound is one."""
    assert ceiling is None or ceiling.estimate is (route == "catalogue"), (route, ceiling)
    return ceiling.usd if ceiling is not None else None


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("case", sorted(TURNS))
async def test_the_default_prices_and_bounds_every_recorded_turn_as_before(route: str, case: str) -> None:
    model = opus(route, TURNS[case]["reply"])
    await model.refresh_prices()
    asked = [TurnMessage("user", OPENING)]

    ceiling = model.turn_price_ceiling(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000)
    reply = await model.turn(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000)

    assert (ceiling_usd(ceiling, route), reply.cost_usd, reply.price_source) == TURN_PINS[(case, route)]
    assert model.ceiling_is_estimate() is (route == "catalogue")


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
    assert model.ceiling_is_estimate() is (route == "catalogue"), "the catalogue's bound is an estimate"


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
    assert model.ceiling_is_estimate(), "OpenRouter may route the call to an endpoint priced otherwise"
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


# --- a host's pricer: the engine bounds the call on the host's rates, and the host prices the reply ------------------


#: A host's deal on Opus 5.5: half the published price.
DEAL: Final = ModelRates(
    base_input=Decimal(2),
    cache_write_5m=Decimal("2.5"),
    cache_write_1h=Decimal(4),
    cache_read=Decimal("0.1"),
    output=Decimal(10),
)


@dataclass(frozen=True)
class Priced:
    """One call a host pricer was asked to price: what it was handed."""

    model: str
    provider: str
    usage: dict[str, Any]
    served_provider: str | None


# parity-with: threetears.evals.providers.Pricer
class FakeHostPricer:
    """A host's port: its rates for the models it lists, guaranteed or not, and its own price for each reply, which it
    keeps a record of being asked for. A model it does not list is left to the default."""

    def __init__(self, rates: Mapping[str, ModelRates], *, guaranteed: bool = True, said: CallPrice | None = None):
        self.rates = dict(rates)
        self.guaranteed = guaranteed
        self.said = said
        self.asked: list[tuple[str, str]] = []
        self.priced: list[Priced] = []

    def rates_for(self, model: str, *, provider: Provider) -> HostRates | None:
        self.asked.append((model, provider))
        rates = self.rates.get(model)
        return HostRates(rates, guaranteed=self.guaranteed) if rates is not None else None

    def price(
        self, model: str, *, provider: Provider, usage: Mapping[str, Any], served_provider: str | None
    ) -> CallPrice:
        self.priced.append(Priced(model, provider, json.loads(json.dumps(usage)), served_provider))
        if self.said is not None:
            return self.said
        counts = usage_counts(provider, usage)
        assert counts.input_tokens is not None and counts.output_tokens is not None
        dollars, why = self.rates[model].cost(
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            cache_read_tokens=counts.cache_read_tokens,
            cache_write_tokens=counts.cache_write_tokens,
            cache_write_1h_tokens=counts.cache_write_1h_tokens,
        )
        assert dollars is not None, why
        return CallPrice(usd=float(dollars), source="host-deal:2026")


def never_read() -> Catalogue:
    """OpenRouter's catalogue, every fetch of which fails the test: a call the host prices reads none."""

    def fetched(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a call the host's pricer prices read the catalogue at {request.url}")

    return Catalogue(http=httpx.AsyncClient(transport=httpx.MockTransport(fetched)))


def hosted(pricer: FakeHostPricer, reply: Mapping[str, Any], *, model: str = MODEL, **options: Any) -> Model:
    """``model`` on Anthropic, handed the host's pricer and a catalogue that must not be read."""
    options.setdefault("catalogue", never_read())
    options.setdefault("effort", "high")
    return Model(model, provider="anthropic", api_key="test-key", pricer=pricer, http=answering(reply), **options)


async def test_host_rates_give_the_bound_price_ceiling_gives_on_the_same_rates_completion_and_turn() -> None:
    published = AnthropicPublishedRates().rates_for(MODEL, provider="anthropic")
    assert published is not None
    host = FakeHostPricer({MODEL: published.rates})
    model = hosted(host, TURNS["opening"]["reply"])
    asked = [TurnMessage("user", OPENING)]

    completion = model.price_ceiling(system="s", user="u")
    turn = model.turn_price_ceiling(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000)

    # The host handed the rates the default holds for Opus 5.5, so the engine's bound is the default's, exactly.
    assert completion == COMPLETION_PINS[("opening", "published")][0]
    assert turn == TurnCeiling(usd=TURN_PINS[("opening", "published")][0], estimate=False)
    # And it is the engine's arithmetic on whatever rates the host hands: the host computes no bound.
    dealt = hosted(FakeHostPricer({MODEL: DEAL}), TURNS["opening"]["reply"])
    assert dealt.price_ceiling(system="s", user="u") == float(
        DEAL.ceiling(prompt_tokens=prompt_token_bound("s", "u", None), output_tokens=DEFAULT_MAX_TOKENS)
    )
    assert dealt.turn_price_ceiling(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000) == TurnCeiling(
        usd=float(DEAL.ceiling(prompt_tokens=turn_prompt_token_bound(SYSTEM, asked, TOOLS), output_tokens=32000))
    )


async def test_the_host_pricer_receives_the_raw_usage_the_served_provider_and_the_ttl_split() -> None:
    host = FakeHostPricer({MODEL: DEAL})
    model = hosted(host, anthropic_body(SPLIT_USAGE))

    completion = await model.generate(system="s", user="u")

    (seen,) = host.priced
    assert seen == Priced(MODEL, "anthropic", SPLIT_USAGE, None), "the usage block exactly as the wire returned it"
    assert seen.usage["cache_creation"] == {"ephemeral_5m_input_tokens": 5_000, "ephemeral_1h_input_tokens": 1_000}
    # 1,200 fresh x $2 + 40,000 read x $0.10 + 5,000 five-minute writes x $2.50 + 1,000 one-hour writes x $4
    # + 300 output x $10, a million tokens.
    assert completion.cost_usd == pytest.approx(0.0259)  # 2,400 + 4,000 + 12,500 + 4,000 + 3,000
    assert completion.price_source == "host-deal:2026" and completion.pricing_failure is None


async def test_an_openrouter_host_pricer_receives_the_byok_upstream_cost_and_the_served_upstream() -> None:
    recorded = COMPLETIONS["json_schema"]
    flash = recorded["result"]["model"]
    said = CallPrice(usd=0.000031, source="host:openrouter+upstream")
    host = FakeHostPricer({flash: DEAL}, guaranteed=False, said=said)
    model = Model(
        flash, api_key="test-key", pricer=host, catalogue=never_read(), http=answering(router_body(recorded, byok=True))
    )

    completion = await model.generate(system="s", user="u")

    (seen,) = host.priced
    assert (seen.provider, seen.served_provider) == ("openrouter", "Venice")
    assert seen.usage["is_byok"] is True
    assert seen.usage["cost_details"] == {"upstream_inference_cost": 0.00002}
    assert (completion.cost_usd, completion.price_source) == (0.000031, "host:openrouter+upstream")


async def test_a_turn_is_priced_by_the_host_from_its_raw_usage_too() -> None:
    host = FakeHostPricer({MODEL: DEAL})
    model = hosted(host, TURNS["tool_results"]["reply"])

    reply = await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000)

    (seen,) = host.priced
    assert seen.usage == TURNS["tool_results"]["reply"]["usage"]
    # 50 fresh x $2 + 5,000 read x $0.10 + 800 five-minute writes x $2.50 + 400 output x $10, a million tokens.
    assert reply.cost_usd == pytest.approx(0.0066) and reply.price_source == "host-deal:2026"


async def test_a_model_the_host_does_not_list_is_left_to_the_default_end_to_end() -> None:
    host = FakeHostPricer({})
    model = Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        pricer=host,
        catalogue=catalogue(),
        effort="high",
        http=answering(TURNS["opening"]["reply"]),
    )
    await model.refresh_prices()

    ceiling = model.price_ceiling(system="s", user="u")
    completion = await model.generate(system="s", user="u")

    assert host.priced == [], "a pricer that answered no rates is not asked for a price"
    assert (ceiling, completion.cost_usd, completion.price_source) == COMPLETION_PINS[("opening", "catalogue")]


# --- a bound before every send -------------------------------------------------------------------------------------


class Counting:
    """A provider that counts what it was sent, and answers with ``reply``."""

    def __init__(self, reply: Mapping[str, Any]) -> None:
        self.reply = dict(reply)
        self.sent = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.sent += 1
        return httpx.Response(200, json=self.reply)


def unboundable(server: Counting) -> Model:
    """A model the host's pricer answers no rates for, whose catalogue cannot be read: nothing bounds its calls."""
    down = Catalogue(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503, text="unavailable")))
    )
    return Model(
        MODEL,
        provider="anthropic",
        api_key="test-key",
        effort="high",
        pricer=FakeHostPricer({}),
        catalogue=down,
        http=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )


async def test_a_completion_nothing_can_bound_is_refused_with_nothing_sent() -> None:
    server = Counting(TURNS["opening"]["reply"])
    model = unboundable(server)
    budget = OutOfRunBudget(RecordingLedger(), scope_id="s", cap_usd=1.0, blocking_executor=None)

    await model.refresh_prices()
    with pytest.raises(ValidationFailedError, match="price_ceiling returned None"):
        budget.admit(model, "variation", [PlannedCall(system="s", user="u")])

    assert server.sent == 0


async def test_a_turn_nothing_can_bound_is_refused_with_nothing_sent() -> None:
    server = Counting(TURNS["opening"]["reply"])
    model = unboundable(server)
    session = ToolSession(
        name="analyst",
        role="analyst",
        model=MODEL,
        client=model.turn_session("analyst", snapshot="snap-0001"),
        mount=FakeToolMount(),
        system=SkillText(text="ANALYST SKILL TEXT", files=("SKILL.md",)),
        tally=Tally(),
        budget=ReportBudget(usd=20.0),
    )
    session.open_round(1, calls=5)

    stop = await session.send(OPENING)

    assert stop.kind == "over_budget" and "cannot bound" in stop.reason
    assert server.sent == 0
    assert await model.turn_pricing_refusal() is not None, "and the report's preflight refuses it before it starts"


# --- an estimate is disclosed -------------------------------------------------------------------------------------


async def test_a_bound_on_rates_the_host_does_not_guarantee_admits_the_turn_and_is_disclosed() -> None:
    host = FakeHostPricer({MODEL: DEAL}, guaranteed=False)
    answers = {**TURNS["opening"]["reply"], "content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn"}
    model = hosted(host, answers)
    asked = [TurnMessage("user", OPENING)]
    ceiling = model.turn_price_ceiling(system=SYSTEM, messages=asked, tools=TOOLS, max_tokens=32000)
    assert ceiling is not None and ceiling.estimate, "a bound on rates the host does not guarantee is an estimate"
    assert model.price_ceiling(system="s", user="u") is not None, "and a completion is still bounded on it"
    tally = Tally()
    session = ToolSession(
        name="analyst",
        role="analyst",
        model=MODEL,
        client=model.turn_session("analyst", snapshot="snap-0001"),
        mount=FakeToolMount(),
        system=SkillText(text="ANALYST SKILL TEXT", files=("SKILL.md",)),
        tally=tally,
        budget=ReportBudget(usd=20.0),
    )
    session.open_round(1, calls=5)

    stop = await session.send(OPENING)

    assert stop.kind == "answered", "an estimate admits the turn"
    (turn,) = tally.turns
    assert turn.admitted_on_estimate
    total = tally.to_json()["total"]
    assert total["admitted_on_estimate"] == 1
    line = made_line(record(generation(tally)))
    # 1,200 fresh x $2 + 5,000 five-minute writes x $2.50 + 300 output x $10, a million tokens: $0.0179.
    assert "for $0.0179 (its budget checked against estimated prices on 1 of 1 turns)" in line, line


def test_a_guaranteed_bound_and_a_token_budget_are_never_disclosed_as_estimates() -> None:
    estimate = TurnCeiling(usd=0.5, estimate=True)
    assert ReportBudget(usd=1.0).admitted_on_estimate(estimate)
    assert not ReportBudget(usd=1.0).admitted_on_estimate(TurnCeiling(usd=0.5))
    assert not ReportBudget(tokens=1_000).admitted_on_estimate(estimate), "a token budget checks no price"


# --- unknown is None, never 0 --------------------------------------------------------------------------------------


async def test_a_reason_the_pricer_gave_is_kept_through_role_usage_and_the_tally_never_as_zero() -> None:
    why = "the host holds no rate for 'claude-opus-5-5' on this route"
    host = FakeHostPricer({MODEL: DEAL}, said=CallPrice(usd=None, source=None, unpriced_reason=why))
    model = hosted(host, TURNS["opening"]["reply"])

    completion = await model.generate(system="s", user="u")
    reply = await model.turn(system=SYSTEM, messages=[TurnMessage("user", OPENING)], tools=TOOLS, max_tokens=32000)

    assert (completion.cost_usd, completion.price_source, completion.pricing_failure) == (None, None, why)
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(completion)
    ledger.add_llm_result(completion)
    (row,) = ledger.rows()
    assert row.cost_usd is None and row.unpriced_reasons == [why], "each distinct reason once, the cost unknown"
    assert row.model_dump()["unpriced_reasons"] == [why]

    tally = Tally()
    tally.record_turn(TurnCost.of(reply, session="analyst", role="analyst", round=1, turn=0, wall_s=1.0))
    total = tally.to_json()["total"]
    assert (total["cost_usd"], total["unpriced_turns"], total["unpriced_reasons"]) == (None, 1, [why])
    assert tally.spent_usd() is None


def test_a_priced_row_stores_no_unpriced_reasons_and_refuses_one() -> None:
    ledger = RoleUsageLedger(role="judge")
    ledger.add(model=MODEL, prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=0.01)
    (row,) = ledger.rows()
    assert row.unpriced_reasons is None and "unpriced_reasons" not in row.model_dump(), "left out when unset"
    with pytest.raises(ValueError, match="this row's cost_usd is known"):
        RoleUsage(role="judge", cost_usd=0.01, unpriced_reasons=["no rate"])
    with pytest.raises(ValueError, match="at least one reason"):
        RoleUsage(role="judge", unpriced_reasons=[])


def test_a_host_s_price_that_contradicts_itself_is_refused() -> None:
    with pytest.raises(ValueError, match="says why"):
        CallPrice(usd=None, source=None)
    with pytest.raises(ValueError, match="names who priced it"):
        CallPrice(usd=0.1, source=None)
    with pytest.raises(ValueError, match="no price source"):
        CallPrice(usd=None, source="host", unpriced_reason="no rate")
    with pytest.raises(ValueError, match="finite number of dollars"):
        CallPrice(usd=-0.1, source="host")


# --- PriceSource keeps the words Discodon stores -------------------------------------------------------------------


def test_price_source_keeps_openrouter_s_two_reported_words_unchanged() -> None:
    assert PriceSource.OPENROUTER_REPORTED == "openrouter:reported"
    assert PriceSource.OPENROUTER_REPORTED_UPSTREAM == "openrouter:reported+upstream"
    assert PriceSource.OPENROUTER_CATALOGUE == "openrouter-catalogue"


# --- support ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    text: str
    is_error: bool = False


# parity-with: threetears.evals.ops.report_writer.session.ToolMount
class FakeToolMount:
    """The analyst tool over a frozen snapshot: the overview, read."""

    @property
    def name(self) -> str:
        return "evals_analyst"

    @property
    def description(self) -> str:
        return "Read the frozen snapshot: one action per call."

    @property
    def rederivable(self) -> bool:
        return True

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}

    async def call(self, arguments: Mapping[str, Any]) -> Outcome:
        return Outcome("OVERVIEW: arms September, control")


# parity-with: threetears.evals.schema.out_of_run_spend.OutOfRunSpendStore
class RecordingLedger:
    """Keeps every out-of-run row written."""

    def __init__(self) -> None:
        self.rows: list[OutOfRunSpend] = []

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        self.rows.append(spend)


def generation(tally: Tally) -> dict[str, Any]:
    """A generation record over ``tally``, as ``report_write`` stores one."""
    return generation_record(
        model=MODEL,
        tally=tally.to_json(),
        skill={"analyst": "a" * 64, "editor": "e" * 64, "package": "0123456789abcdef" * 4},
        reviews=[],
        sessions={},
    )


def record(generated: dict[str, Any]) -> EvalReportDoc:
    """A stored report document carrying ``generated``."""
    snapshot = snapshot_of("prototype3/kairo")
    return EvalReportDoc(
        scope_id=CALLER.scope_id,
        snapshot_id=snapshot.id,
        campaign_id=snapshot.campaign_id,
        doc=DOC,
        ending="stored",
        generation=generated,
        created_by=CALLER.identity,
        created_at="2026-10-28T09:30:00+00:00",
    )


# --- a host client that is not a Model prices its own calls -------------------------------------------------------


@dataclass
class Reported:
    """A completion as a host's own client reports it."""

    cost_usd: float | None
    price_source: str | None
    content: str = "{}"
    model: str = "host-model"
    served_model: str | None = None
    stop_reason: str = "end_turn"
    temperature: float | None = None
    input_tokens: int | None = 10
    output_tokens: int | None = 5
    reasoning_tokens: int | None = None


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
class FakeBoundCompletionClient:
    """A host's own client: it bounds its calls itself and reports what each cost, or nothing."""

    model_name = "host-model"

    def __init__(self, *, ceiling: float | None, reported: Reported) -> None:
        self.ceiling = ceiling
        self.reported = reported
        self.calls = 0

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        return self.ceiling

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Reported:
        self.calls += 1
        return self.reported

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> FakeBoundCompletionClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


async def test_a_host_client_s_ceiling_is_its_bound_and_its_cost_is_recorded_as_reported() -> None:
    client = FakeBoundCompletionClient(ceiling=0.02, reported=Reported(cost_usd=0.0013, price_source="host:rate-card"))
    ledger = RecordingLedger()
    budget = OutOfRunBudget(ledger, scope_id="s", cap_usd=1.0, blocking_executor=None)

    assert call_ceiling(client, system="s", user="u") == 0.02, "its own bound, no pricer asked"
    assert not ceiling_is_estimate(client), "a client that says nothing of estimates bounds on its own word"
    [admitted] = budget.admit(client, "variation", [PlannedCall(system="s", user="u")])
    made = await budget.generate(client, admitted)

    assert (made.spend.priced_ceiling_usd, made.spend.cost_usd, made.spend.price_source) == (
        0.02,
        0.0013,
        "host:rate-card",
    )


async def test_a_host_client_that_reports_no_cost_is_unpriced_never_zero() -> None:
    client = FakeBoundCompletionClient(ceiling=0.02, reported=Reported(cost_usd=None, price_source=None))
    budget = OutOfRunBudget(RecordingLedger(), scope_id="s", cap_usd=1.0, blocking_executor=None)
    [admitted] = budget.admit(client, "variation", [PlannedCall(system="s", user="u")])

    made = await budget.generate(client, admitted)
    ledger = RoleUsageLedger(role="judge")
    ledger.add_llm_result(await client.generate(system="s", user="u"))
    (row,) = ledger.rows()

    assert made.spend.cost_usd is None and row.cost_usd is None and row.price_source is None
    assert row.unpriced_reasons is None, "it gave no reason, and none is invented"
    assert dollars_text(row.cost_usd) == "unpriced"


# --- a run's cap checked against estimates ------------------------------------------------------------------------


async def test_a_model_says_its_bound_is_an_estimate_through_a_factory_s_lease_and_the_kernel_reader() -> None:
    estimated = hosted(FakeHostPricer({MODEL: DEAL}, guaranteed=False), TURNS["opening"]["reply"])
    guaranteed = hosted(FakeHostPricer({MODEL: DEAL}), TURNS["opening"]["reply"])

    assert estimated.ceiling_is_estimate() and not guaranteed.ceiling_is_estimate()
    assert ceiling_is_estimate(estimated) and not ceiling_is_estimate(guaranteed)
    leased = Clients(default=estimated, catalogue=never_read())("judge", None)
    assert ceiling_is_estimate(leased), "the lease a Clients role hands out forwards it"
    assert not opus("published", TURNS["opening"]["reply"]).ceiling_is_estimate(), "the default's rates are Anthropic's"


def test_a_call_ceiling_is_exact_unless_said_to_be_an_estimate() -> None:
    assert not CallCeiling(usd=0.1, call="the judge call", lowered_by="a cheaper judge").estimate
    assert CallCeiling(usd=0.1, call="the judge call", lowered_by="a cheaper judge", estimate=True).estimate


def test_a_run_records_a_cap_checked_against_estimates_and_leaves_it_out_when_unset() -> None:
    sample = stored_sample(EvalRun)
    assert "cap_checked_on_estimates" not in sample
    unset = EvalRun.model_validate(sample)
    checked = EvalRun.model_validate({**sample, "cap_checked_on_estimates": True})

    assert "cap_checked_on_estimates" not in unset.to_dict(), "unset, the stored run is as before"
    assert EvalRun.model_validate(checked.to_dict()).cap_checked_on_estimates is True


def test_the_spend_cap_line_says_it_was_checked_against_estimated_rates() -> None:
    fields: dict[str, Any] = {
        "run_id": "r",
        "scope_id": "s",
        "template_id": None,
        "candidate_model": "m",
        "status": "completed",
        "k_runs": 2,
        "n_cases": 3,
        "n_results": 6,
        "n_scored": 6,
        "n_candidate_failed": 0,
        "n_excluded": 0,
        "measures": [],
        "confusion": [],
        "labels": [],
        "errors": [],
        "max_cost_usd": 2.5,
        "spend": [RoleSpend(role="candidate", calls=3, cost_usd=0.01)],
    }

    assert EvalSummary(**fields).spend_cap_text == "spend cap: $2.50 for this run"
    assert "cap_checked_on_estimates" not in EvalSummary(**fields).model_dump()
    assert EvalSummary(**fields, cap_checked_on_estimates=True).spend_cap_text == (
        "spend cap: $2.50 for this run, checked against estimated rates"
    )


def test_the_launch_estimate_says_an_arm_s_cap_is_checked_against_estimated_rates() -> None:
    arm: dict[str, Any] = {
        "described": "arm 'm' on template 't'",
        "candidate_model": "m",
        "case_count": 3,
        "case_source": "stored",
        "n_observations": 6,
        "predicted_usd": 0.5,
        "central_usd": None,
        "low_usd": None,
        "method_id": None,
        "basis": "the largest call's ceiling",
        "outcome": "admitted",
        "refusal": None,
        "call_ceiling_usd": 0.1,
    }
    estimate: dict[str, Any] = {
        "template_id": "t",
        "subject_id": "s",
        "cassette_mode": "live",
        "k_runs": 2,
        "n_variations": 0,
        "hypothetical_case_count": None,
        "cap_usd": 1.0,
        "cap_origin": "chosen",
        "total_predicted_usd": 0.5,
        "would_launch": True,
        "computed_at": "2026-10-28T09:30:00+00:00",
    }
    said = "its cap is checked against estimated rates"

    assert said not in estimate_text(LaunchEstimate(**estimate, arms=[ArmEstimate(**arm)]))
    assert "call_ceiling_estimate" not in ArmEstimate(**arm).model_dump()
    flagged = estimate_text(LaunchEstimate(**estimate, arms=[ArmEstimate(**arm, call_ceiling_estimate=True)]))
    assert f"  {said}: its largest call is bounded on rates the host's pricer does not guarantee" in flagged
