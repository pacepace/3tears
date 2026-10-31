"""A report budget in tokens and turns: a model nothing prices still writes its report, tallied unpriced, never $0.

A host whose model has no per-token price (a local model, a subscription) declares its report's cap as tokens and
turns (``ReportBudget(tokens=, turns=)``) rather than dollars. Such a budget admits each turn by a bound on the tokens
it can use and by the turns already sent, and never asks for a price, so ``report_write`` asks for none before it
starts either. Exactly one form is declared, never both and never neither; a turn cap may also ride on dollars. Every turn here is scripted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import pytest

from packages.evals.tests.fixtures.report_plans.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import (
    MODEL,
    FakeToolTurnClient,
    FakeTurnModel,
    call,
    kairo_id,
    reply,
    review,
    submitted,
)
from threetears.evals.actions.report_writing import write_report
from threetears.evals.analysis.analyst.skill import SkillText
from threetears.evals.kernel.capped_completion import turn_prompt_token_bound
from threetears.evals.ops.report_writer.session import SessionLimits, ToolSession
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.schema import TurnMessage, TurnReply, TurnText, TurnToolSpec

SYSTEM: Final = SkillText(text="ANALYST SKILL TEXT", files=("SKILL.md",))
OPENING: Final = "Write the report for snapshot snap-0001."

#: Why every scripted turn here is unpriced, as a pricer that holds no rate for the model says.
UNPRICED: Final = "no rate for the model"


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


def session(client: FakeToolTurnClient, budget: ReportBudget, tally: Tally | None = None) -> ToolSession:
    opened = ToolSession(
        name="analyst",
        role="analyst",
        model=MODEL,
        client=client,
        mount=FakeToolMount(),
        system=SYSTEM,
        tally=tally or Tally(),
        budget=budget,
    )
    opened.open_round(1, calls=10)
    return opened


def first_turn_bound(tools: tuple[TurnToolSpec, ...]) -> int:
    """The tokens the session's first turn is admitted on: its prompt's bound and its output cap."""
    prompt = turn_prompt_token_bound(SYSTEM.text, [TurnMessage("user", OPENING)], tools)
    return prompt + SessionLimits().max_tokens


def unpriced(*content: Any) -> TurnReply:
    """A scripted reply nothing priced: 1,000 input and 50 output tokens, ``cost_usd`` ``None``."""
    return reply(*content, cost=None)


# --- the caps ------------------------------------------------------------------------------------------------------


async def test_a_turn_cap_admits_its_turns_then_refuses_with_nothing_sent() -> None:
    client = FakeToolTurnClient(
        unpriced(call(1, "campaign_overview", "snap-0001")), unpriced(TurnText("Done.")), ceiling=None
    )
    budget = ReportBudget(turns=1)

    stop = await session(client, budget).send(OPENING)

    assert stop.kind == "over_budget" and stop.reason.endswith("the report's 1-turn budget is spent")
    assert len(client.requests) == 1, "the refused turn sent nothing"


async def test_a_token_cap_admits_on_the_turn_s_token_bound_then_refuses_with_nothing_sent() -> None:
    client = FakeToolTurnClient(
        unpriced(call(1, "campaign_overview", "snap-0001")), unpriced(TurnText("Done.")), ceiling=None
    )
    opened = session(client, ReportBudget(usd=1.0))  # only to read the tools it mounts
    bound = first_turn_bound(opened.tools)
    budget = ReportBudget(tokens=bound)

    stop = await session(client, budget).send(OPENING)

    assert stop.kind == "over_budget", "the first turn fits its bound exactly; the second, after 1,050 used, cannot"
    assert "tokens, and 1,050 of the report's" in stop.reason and f"{bound:,}-token budget are used" in stop.reason
    assert len(client.requests) == 1, "the refused turn sent nothing"


async def test_a_token_and_turn_budget_admits_a_turn_nothing_can_price() -> None:
    tally = Tally()
    client = FakeToolTurnClient(unpriced(TurnText("Done.")), ceiling=None)

    stop = await session(client, ReportBudget(tokens=10_000_000, turns=5), tally).send(OPENING)

    assert stop.kind == "answered", "no price is asked for: the turn was admitted on its tokens"
    total = tally.to_json()["total"]
    assert (total["cost_usd"], total["unpriced_turns"], total["unpriced_reasons"]) == (None, 1, [UNPRICED])
    assert total["admitted_on_estimate"] == 0


async def test_a_turn_that_reported_no_tokens_is_counted_at_its_bound() -> None:
    silent = TurnReply(
        content=(call(1, "campaign_overview", "snap-0001"),),
        stop_reason="tool_use",
        provider_stop_reason=None,
        model=MODEL,
        served_model=None,
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=None,
        price_source=None,
        pricing_failure=UNPRICED,
    )
    client = FakeToolTurnClient(silent, unpriced(TurnText("Done.")), ceiling=None)
    bound = first_turn_bound(session(client, ReportBudget(usd=1.0)).tools)

    stop = await session(client, ReportBudget(tokens=bound * 2)).send(OPENING)

    assert stop.kind == "over_budget", "the silent turn counts its whole bound, so the next cannot fit beside it"
    assert f"{bound:,} of the report's" in stop.reason


# --- one form ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "caps",
    [{"usd": 5.0, "tokens": 1_000}, {"usd": 5.0, "tokens": 1_000, "turns": 10}],
    ids=["usd-and-tokens", "all-three"],
)
def test_both_forms_at_once_are_refused(caps: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="never both"):
        ReportBudget(**caps)


async def test_a_turn_cap_rides_on_a_dollar_budget_and_refuses_with_nothing_sent() -> None:
    client = FakeToolTurnClient(reply(call(1, "campaign_overview", "snap-0001")), reply(TurnText("Done.")))
    budget = ReportBudget(usd=5.0, turns=1)

    stop = await session(client, budget).send(OPENING)

    assert budget.spelled == "$5.00, 1-turn"
    assert stop.kind == "over_budget" and stop.reason.endswith("the report's 1-turn budget is spent")
    assert len(client.requests) == 1, "the refused turn sent nothing"


def test_neither_form_is_refused() -> None:
    with pytest.raises(ValueError, match="declares a cap"):
        ReportBudget()


@pytest.mark.parametrize(
    ("caps", "match"),
    [
        ({"tokens": 0}, "tokens budget is a whole number above 0"),
        ({"turns": -1}, "turns budget is a whole number above 0"),
        ({"tokens": 1.5}, "tokens budget is a whole number above 0"),
        ({"turns": True}, "turns budget is a whole number above 0"),
        ({"usd": 0}, "finite number of dollars above 0"),
    ],
)
def test_a_cap_that_is_no_count_is_refused(caps: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ReportBudget(**caps)


def test_each_form_is_spelled_as_declared() -> None:
    assert ReportBudget(usd=20).spelled == "$20.00"
    assert ReportBudget(tokens=2_000_000, turns=60).spelled == "2,000,000-token, 60-turn"
    assert ReportBudget(turns=60).spelled == "60-turn"


# --- a report on a model nothing prices ----------------------------------------------------------------------------


async def test_an_unpriced_model_writes_its_report_under_a_token_and_turn_budget_and_the_page_says_unpriced() -> None:
    snap = kairo_id()
    model = FakeTurnModel(
        # A dollar budget's preflight would refuse it: nothing bounds its turns.
        unpriced="no turn on the model can be bounded before it is sent (no rates for it)",
        analyst=FakeToolTurnClient(unpriced(call(1, "options_dominance", snap)), unpriced(submitted()), ceiling=None),
        editor_1=FakeToolTurnClient(unpriced(call(9, "options_dominance", snap)), unpriced(review()), ceiling=None),
    )

    result = await write_report(
        off_loop(seeded_host),
        CALLER,
        model=model,
        budget=ReportBudget(tokens=20_000_000, turns=40),
        snapshot_id=snap,
    )

    assert result.ending == "stored", result.outcome.reason
    total = result.generation["tally"]["total"]
    assert (total["turns"], total["cost_usd"], total["unpriced_turns"]) == (4, None, 4), "unknown, never $0"
    assert total["unpriced_reasons"] == [UNPRICED]
    assert result.page is not None
    assert ", unpriced (4 turns, none of them priced), " in result.page.made, result.page.made
    assert "$0" not in result.page.made


async def test_under_a_dollar_budget_the_same_model_is_refused_before_anything_is_sent() -> None:
    why = "no turn on the model can be bounded before it is sent (no rates for it)"
    model = FakeTurnModel(unpriced=why)

    with pytest.raises(ValueError) as refused:
        await write_report(
            off_loop(seeded_host), CALLER, model=model, budget=ReportBudget(usd=20.0), snapshot_id=kairo_id()
        )

    said = str(refused.value)
    assert "its $20.00 budget admits each turn on its price ceiling" in said and why in said
    assert "a model nothing prices is written under a token and turn budget (ReportBudget(tokens=, turns=))" in said
    assert model.asked == [], "no session was opened"
