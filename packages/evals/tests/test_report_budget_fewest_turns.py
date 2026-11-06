"""A report budget that cannot admit a finished report is refused before any turn is bought (backlog W6).

A report takes at least ``MIN_TURNS_FOR_A_REPORT`` turns (the analyst's submission and the editor's review), and a
dollar budget admits each turn on its ceiling, so a budget under that many of the first turn's ceiling is refused at
the report's first turn, naming the budget asked, the ceiling, the fewest turns and the budget that admits them, with
nothing sent. The 2026-11-06 acceptance run is the incident: a $0.75 budget, a Sonnet turn whose ceiling is about $0.67,
and 36 of 50 reports that ended over budget with no page after 1 to 5 turns at about $0.10 each. A $1.50 budget, under
which every report of the second run finished, is admitted. A report that still runs out says how many turns it was
admitted and how many its budget admits whole at that ceiling, and a writer eval says, for an arm whose reports ran out
with no page, what the reports of its run that stored one spent. The launch refusal a writer eval makes before
anything runs, priced at the rates Anthropic publishes, is held in ``test_writer_acceptance_dry.py``.

Mutations that turn this file red: the first turn admitted on its ceiling alone; the refusal not naming the budget, the
ceiling, the fewest turns or the budget that admits them; a budget at two ceilings refused; the rule applied to a turn
after the first; a reserve at an average of what turns cost; the ending's words dropping the turns admitted or the turns
the budget admits; the writer eval's disclosure missing, or counting a report that stored a page as one with none.

Every turn is scripted, and nothing reaches a model.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.report_writer_support import (
    FakeToolTurnClient,
    FakeTurnModel,
    call,
    reply,
    review,
    submitted,
    written,
)
from packages.evals.tests.writer_case_support import SEED
from packages.evals.tests.writer_eval_support import FakeScoringJudge, kairo_case
from threetears.evals.harness import Writer, WriterCaseFile, WriterEval
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.kernel import EvalStorage
from threetears.evals.ops.report_writer.tally import MIN_TURNS_FOR_A_REPORT, ReportBudget
from threetears.evals.providers.wire import KEY_VARIABLES
from threetears.evals.schema.turns import TurnCeiling, TurnReply
from threetears.evals.storage import SqliteDocumentStore
from threetears.evals.testing import RecordedTurns

#: The incident's numbers: the report's budget, and a Sonnet turn's ceiling at its 32,000-token output cap.
INCIDENT_BUDGET: Final = 0.75
SONNET_CEILING: Final = 0.67

#: The second run's budget, under which every report finished.
ADMITTED_BUDGET: Final = 1.5

#: The writer every arm of the writer eval names: one no published rate lists, so the eval's launch leaves it to the
#: reports' first turns.
WRITER: Final = "report-writer-candidate"

SCOPE: Final = "fewest-turns"

REFUSED_AT_LAUNCH: Final = (
    "this first turn can cost up to $0.67, and a report takes at least 2 turns, so the report's $0.75 budget cannot "
    "admit one at that ceiling; a budget of $1.34 or more can"
)


def test_a_report_takes_at_least_the_submission_and_the_review() -> None:
    assert MIN_TURNS_FOR_A_REPORT == 2


def test_the_incident_s_budget_is_refused_naming_what_admits_a_report() -> None:
    budget = ReportBudget(usd=INCIDENT_BUDGET)

    assert budget.admit(TurnCeiling(usd=SONNET_CEILING), tokens=1) == REFUSED_AT_LAUNCH
    # It refuses a budget a single turn fits in: $0.67 is under $0.75, and two of it are not.
    assert SONNET_CEILING < INCIDENT_BUDGET < SONNET_CEILING * MIN_TURNS_FOR_A_REPORT


@pytest.mark.parametrize("usd", [ADMITTED_BUDGET, SONNET_CEILING * MIN_TURNS_FOR_A_REPORT])
def test_a_budget_at_or_above_the_fewest_turns_is_admitted(usd: float) -> None:
    assert ReportBudget(usd=usd).admit(TurnCeiling(usd=SONNET_CEILING), tokens=1) is None


def test_a_budget_in_tokens_and_turns_is_never_refused_for_dollars() -> None:
    budget = ReportBudget(tokens=100_000, turns=40)

    assert budget.fewest_turns_refusal(TurnCeiling(usd=SONNET_CEILING), priced="this first turn") is None
    assert budget.admit(TurnCeiling(usd=SONNET_CEILING), tokens=1) is None


async def test_the_report_ends_over_budget_before_its_first_turn_with_nothing_sent() -> None:
    analyst = FakeToolTurnClient(reply(submitted()), ceiling=SONNET_CEILING)
    model = FakeTurnModel(analyst=analyst)

    result = await written(model, budget_usd=INCIDENT_BUDGET)

    assert result.ending == "over_budget" and result.stored is None
    assert result.outcome.reason == f"the report's budget refused the next turn: {REFUSED_AT_LAUNCH}"
    assert analyst.requests == [] and model.asked == ["analyst"]
    assert result.generation["tally"]["total"]["turns"] == 0


async def test_the_admitted_budget_writes_the_report() -> None:
    model = FakeTurnModel(
        analyst=FakeToolTurnClient(reply(submitted(), cost=0.3), ceiling=SONNET_CEILING),
        editor_1=FakeToolTurnClient(reply(review(), cost=0.1), ceiling=0.4),
    )

    result = await written(model, budget_usd=ADMITTED_BUDGET)

    assert result.ending == "stored" and result.stored is not None


async def test_a_report_that_runs_out_later_says_what_its_budget_admitted() -> None:
    # The incident's shape: a cheap first turn admitted, then a turn whose ceiling the rest of the budget cannot hold.
    analyst = FakeToolTurnClient(
        reply(call(1, "campaign_overview", "snap"), cost=0.11), reply(submitted()), ceiling=0.3
    )
    analyst.ceilings[1] = SONNET_CEILING
    model = FakeTurnModel(analyst=analyst)

    result = await written(model, budget_usd=INCIDENT_BUDGET)

    assert result.ending == "over_budget" and result.stored is None
    assert result.outcome.reason == (
        "the report's budget refused the next turn: the next turn can cost up to $0.67, and $0.11 of the report's $0.75 "
        "budget is spent; it admitted 1 turn, and at that ceiling the whole budget admits 1, where a report takes at "
        "least 2"
    )
    assert len(analyst.requests) == 1


# --- the writer eval's account of the reports that ran out --------------------------------------------------------


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """No key in the environment, and every socket connect refused and counted."""
    for variable in KEY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    attempts: list[object] = []

    def connect(_self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise AssertionError(f"a replay-only run connected to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def _lines(path: Path, sessions: dict[str, Sequence[tuple[float, TurnReply | None]]]) -> None:
    """A turn recording of ``sessions``: each turn's ceiling, then its reply; a last turn with no reply is refused."""
    lines: list[dict[str, Any]] = []
    for session, turns in sessions.items():
        for index, (ceiling, said) in enumerate(turns):
            stamp = {
                "recorder_version": RecordedTurns.FORMAT_VERSION,
                "recorded_at": "2026-11-06T00:00:00+00:00",
                "session": session,
                "index": index,
            }
            lines.append({**stamp, "kind": "ceiling", "ceiling": ceiling})
            if said is not None:
                lines.append({**stamp, "kind": "reply", "reply": said.to_json()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _laid(tmp_path: Path, arms: dict[str, Writer]) -> None:
    """Kairo's report, which stores a page, and r1's, which runs out after one turn, laid for each arm."""
    kairo = kairo_case(tmp_path / "cases")
    group, withhold = SEED["r1"]
    r1 = WriterCaseFile(
        name="r1", group=group, withhold=list(withhold), snapshot=fixture_snapshots("frozen-elsewhere")["r1"]
    )
    r1.write(tmp_path / "cases" / "r1.json.gz")
    for (arm, writer), cost in zip(arms.items(), (0.1, 0.15), strict=True):
        routed = writer_turns(
            tmp_path / "turns",
            arm=arm,
            skill=writer.analyst_skill,
            revisions=writer.revisions,
            cases={kairo.snapshot.id: "kairo", r1.snapshot.id: "r1"},
        )
        _lines(
            routed.path(WRITER, None, kairo.snapshot.id),
            {
                "analyst": [
                    (0.05, reply(call(1, "options_dominance", "{snapshot}"), cost=cost, model=WRITER)),
                    (0.05, reply(submitted(), cost=cost, model=WRITER)),
                ],
                "editor-1": [(0.05, reply(review(), cost=cost, model=WRITER))],
            },
        )
        _lines(
            routed.path(WRITER, None, r1.snapshot.id),
            {
                "analyst": [
                    (0.3, reply(call(1, "campaign_overview", "{snapshot}"), cost=0.11, model=WRITER)),
                    (SONNET_CEILING, None),
                ]
            },
        )


async def test_the_writer_eval_says_how_many_reports_ran_out_and_what_the_paged_ones_spent(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = {"two revisions": Writer(WRITER), "one revision": Writer(WRITER, revisions=1)}
    _laid(tmp_path, arms)

    result = await WriterEval(
        tmp_path / "cases",
        arms=arms,
        judge=FakeScoringJudge(),
        recordings=tmp_path / "turns",
        budget_usd=INCIDENT_BUDGET,
        max_cost_usd=10.0,
        held_out_fraction=0.99,
        name=SCOPE,
        store=tmp_path / "evals.sqlite",
    ).run()

    assert offline == []
    assert (
        "two revisions: 1 of 2 report(s) ended over their $0.75, 40-turn budget with no page; the 1 that stored a page under "
        "it spent $0.30" in result.disclosures
    )
    assert (
        "one revision: 1 of 2 report(s) ended over their $0.75, 40-turn budget with no page; the 1 that stored a page under "
        "it spent $0.45" in result.disclosures
    )
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    results = await asyncio.to_thread(storage.query_eval_results_by_run, result.arms["two revisions"].run_id, SCOPE)
    reasons = [
        r.kind_payload["reason"] for r in results if r.kind_payload and r.kind_payload["ending"] == "over_budget"
    ]
    assert reasons == [
        "the report's budget refused the next turn: the next turn can cost up to $0.67, and $0.11 of the report's "
        "$0.75 budget is spent; it admitted 1 turn, and at that ceiling the whole budget admits 1, where a report "
        "takes at least 2"
    ]
