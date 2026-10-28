"""The report writer's turns priced end to end: the shipped client, recording, under the report's budget.

The first live paid run ended ``over_budget`` before its first turn: a model recording its turns
(``turn_recording=``) handed the session a recording wrapper that had no ``refresh_prices``, the engine's
``getattr``-based refresh skipped it in silence, OpenRouter's catalogue was never read, no ceiling could be given, and
the budget refused the turn. Every test before then drove the session through fakes with fixed prices, so none saw
it. These drive the real :class:`~threetears.evals.providers.Model`, recording to a temporary file, through a real
:class:`~threetears.evals.ops.report_writer.session.ToolSession`, over ``httpx.MockTransport`` with canned Anthropic
Messages replies. The model is built as the report writer's command line builds it, with no pricer and no catalogue,
so the default pricer prices it (Anthropic's published rates as the package ships them); every client opened after
the model's own fails the test, so no OpenRouter catalogue is fetched, nor anything else.

They also hold the report's preflight: a model whose turns cannot be bounded is refused before anything is frozen,
sent or spent, naming the model and what to do.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host
from packages.evals.tests.report_writer_support import FakeTurnModel
from threetears.evals.actions.report_writing import write_report
from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.ops.report_writer.session import ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.providers import Catalogue, Model
from threetears.evals.schema.turns import TurnMessage

MODEL: Final = "claude-opus-5-5"
SNAPSHOT: Final = "snap-0001"
KEY: Final = "the-host-s-own-anthropic-key"

#: The analyst reads the overview: a tool call, its prompt half written to the cache (five-minute writes, the only
#: lifetime the client asks for).
CALLS: Final[dict[str, Any]] = {
    "model": "claude-opus-5-5-20261001",
    "content": [
        {"type": "text", "text": "Reading the overview first."},
        {
            "type": "tool_use",
            "id": "toolu_01A",
            "name": "evals_analyst",
            "input": {"action": "campaign_overview", "snapshot_id": SNAPSHOT},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_creation_input_tokens": 5000,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 5000, "ephemeral_1h_input_tokens": 0},
    },
}

#: Then answers, reading the conversation before it from the cache.
ANSWERS: Final[dict[str, Any]] = {
    "model": "claude-opus-5-5-20261001",
    "content": [{"type": "text", "text": "The overview reads as expected; here is the report."}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 50,
        "output_tokens": 400,
        "cache_creation_input_tokens": 800,
        "cache_read_input_tokens": 5000,
        "cache_creation": {"ephemeral_5m_input_tokens": 800, "ephemeral_1h_input_tokens": 0},
    },
}

#: What each turn cost at Anthropic's published Opus 5.5 rates ($4 input, $5 five-minute cache write, $0.20 cache
#: read, $20 output, a million tokens), by hand.
CALLS_USD: Final = (1200 * 4 + 5000 * 5 + 300 * 20) / 1e6  # 0.0358
ANSWERS_USD: Final = (50 * 4 + 5000 * 0.20 + 800 * 5 + 400 * 20) / 1e6  # 0.0132


class Messages:
    """Anthropic's Messages API: each request answered with the next canned reply, every request kept."""

    def __init__(self, *replies: Mapping[str, Any]) -> None:
        self._replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"a request the script holds no reply for: {request.url}")
        return httpx.Response(200, json=self._replies.pop(0))


def dev_model(messages: Messages, monkeypatch: pytest.MonkeyPatch, **options: Any) -> Model:
    """``claude-opus-5-5`` built as the report writer's command line builds it: the host's key, no pricer and no
    catalogue, so the default pricer. Every ``httpx`` client opened after the model's own fails the test: a catalogue
    fetch opens one."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(messages))

    def opened(**_: Any) -> httpx.AsyncClient:
        raise AssertionError("a client was opened past the model's own: something read the network (a catalogue)")

    monkeypatch.setattr(httpx, "AsyncClient", opened)
    return Model(MODEL, provider="anthropic", api_key=KEY, effort="high", http=http, **options)


@dataclass(frozen=True)
class Outcome:
    text: str
    is_error: bool = False


# parity-with: threetears.evals.ops.report_writer.session.ToolMount
class FakeToolMount:
    """The analyst tool over a frozen snapshot: the overview, read."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(dict(arguments))
        return Outcome("OVERVIEW: arms September, control; aspects late")


async def test_a_recorded_session_on_the_shipped_client_is_priced_admitted_and_tallied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = Messages(CALLS, ANSWERS)
    recording = tmp_path / "turns.jsonl"
    model = dev_model(messages, monkeypatch, turn_recording=recording)
    assert await model.turn_pricing_refusal() is None, "the default prices Opus 5.5 before the run starts"
    client = model.turn_session("analyst", snapshot=SNAPSHOT)
    mount, tally, budget = FakeToolMount(), Tally(), ReportBudget(usd=20.0)
    session = ToolSession(
        name="analyst",
        role="analyst",
        model=MODEL,
        client=client,
        mount=mount,
        system=SkillText(text="ANALYST SKILL TEXT", files=("SKILL.md",)),
        tally=tally,
        budget=budget,
    )
    session.open_round(1, calls=5)

    stop = await session.send(f"Write the report for snapshot {SNAPSHOT}.")

    assert stop.kind == "answered", stop.reason
    assert mount.calls == [{"action": "campaign_overview", "snapshot_id": SNAPSHOT}]

    # Admitted: each turn had a ceiling, recorded beside it, at least what the turn cost.
    lines = [json.loads(line) for line in recording.read_text().splitlines()]
    ceilings = [line["ceiling"] for line in lines if line["kind"] == "ceiling"]
    assert len(ceilings) == 2 and all(ceiling is not None for ceiling in ceilings)
    assert ceilings[0] >= 32_000 * 20 / 1e6, "the whole output cap at the output rate, at least"
    assert ceilings[0] >= CALLS_USD and ceilings[1] >= ANSWERS_USD

    # Tallied at a price, and charged to the budget at it.
    total = tally.to_json()["total"]
    assert total["turns"] == 2 and total["unpriced_turns"] == 0
    assert total["cost_usd"] == pytest.approx(CALLS_USD + ANSWERS_USD)
    assert budget.spent == pytest.approx(CALLS_USD + ANSWERS_USD)
    replies = [line["reply"] for line in lines if line["kind"] == "reply"]
    assert [reply["price_source"] for reply in replies] == ["anthropic-published:2026-10-28"] * 2
    assert [reply["cost_usd"] for reply in replies] == [pytest.approx(CALLS_USD), pytest.approx(ANSWERS_USD)]

    # Sent to Anthropic on the host's own key, and nothing else read (dev_model fails any other client).
    assert [str(request.url) for request in messages.requests] == ["https://api.anthropic.com/v1/messages"] * 2
    assert {request.headers["x-api-key"] for request in messages.requests} == {KEY}


async def test_a_recording_session_forwards_every_pricing_method_to_the_live_client(tmp_path: Path) -> None:
    rows = {"data": [{"id": "anthropic/claude-opus-5.5", "pricing": {"prompt": "0.000004", "completion": "0.00002"}}]}
    catalogue = Catalogue(
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows)))
    )
    model = Model(
        MODEL,
        provider="anthropic",
        api_key=KEY,
        turn_recording=tmp_path / "turns.jsonl",
        catalogue=catalogue,  # a host's catalogue, and no pricer: priced from the catalogue, read on refresh
        http=httpx.AsyncClient(transport=httpx.MockTransport(Messages())),
    )
    session = model.turn_session("analyst", snapshot=SNAPSHOT)
    asked = [TurnMessage("user", "x")]

    assert session.turn_price_ceiling(system="s", messages=asked, tools=(), max_tokens=10) is None, "not read yet"
    found = await session.refresh_prices()

    assert getattr(found, "refreshed", None) is True, "the live client read the catalogue: the wrapper dropped nothing"
    assert catalogue.loaded
    assert session.turn_price_ceiling(system="s", messages=asked, tools=(), max_tokens=10) is not None


async def test_a_replaying_session_is_free_and_reads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = tmp_path / "turns.jsonl"
    recorder = dev_model(Messages(ANSWERS), monkeypatch, turn_recording=recording)
    live = recorder.turn_session("analyst", snapshot=SNAPSHOT)
    await live.refresh_prices()
    asked = [TurnMessage("user", "Write the report.")]
    recorded_ceiling = live.turn_price_ceiling(system="s", messages=asked, tools=(), max_tokens=1000)
    await live.turn(system="s", messages=asked, tools=(), max_tokens=1000)

    replayer = Model(MODEL, provider="anthropic", turn_recording=recording, api_key="unused")
    assert await replayer.turn_pricing_refusal() is None
    replay = replayer.turn_session("analyst", snapshot=SNAPSHOT)

    assert await replay.refresh_prices() is None, "a replay reads nothing"
    assert replay.turn_price_ceiling(system="other", messages=asked, tools=(), max_tokens=1) == recorded_ceiling
    replayed = await replay.turn(system="other", messages=asked, tools=(), max_tokens=1)
    assert replayed.cost_usd == pytest.approx(ANSWERS_USD), "the recorded cost, the historical fact"


async def test_a_replay_of_a_model_the_rates_do_not_list_is_still_admitted_since_it_costs_nothing(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "turns.jsonl"
    recording.write_text("")
    replayer = Model("claude-opus-9", provider="anthropic", turn_recording=recording, api_key="unused")
    assert await replayer.turn_pricing_refusal() is None


async def test_a_route_other_than_anthropic_s_is_raised_as_itself_not_as_a_pricing_refusal() -> None:
    model = Model("anthropic/claude-opus-5.5", api_key="openrouter-key")
    with pytest.raises(ValueError, match="goes to openrouter") as refused:
        await model.turn_pricing_refusal()
    assert "provider='anthropic'" in str(refused.value)


async def test_the_report_raises_a_route_refusal_as_the_model_words_it_without_the_budget_s_lead_in() -> None:
    model = Model("anthropic/claude-opus-5.5", api_key="openrouter-key")

    with pytest.raises(ValueError) as refused:
        await write_report(
            off_loop(seeded_host),
            CALLER,
            model=model,
            budget=ReportBudget(usd=20.0),
            campaign_id="no-such-campaign",
        )

    said = str(refused.value)
    assert said.startswith("tool turns go to Anthropic's Messages API alone") and "budget" not in said


async def test_the_report_is_refused_before_anything_is_frozen_sent_or_spent() -> None:
    why = "no turn on 'claude-opus-9' can be bounded before it is sent (the price catalogue has not been read)"
    model = FakeTurnModel(unpriced=why)

    with pytest.raises(ValueError) as refused:
        # A campaign that does not exist: had the freeze run first, it would have refused on that instead.
        await write_report(
            off_loop(seeded_host),
            CALLER,
            model=model,
            budget=ReportBudget(usd=20.0),
            campaign_id="no-such-campaign",
        )

    said = str(refused.value)
    assert said.startswith("the report was not started, and nothing was frozen, sent or spent")
    assert "$20.00 budget" in said and why in said
    assert model.asked == [], "no session was opened"
