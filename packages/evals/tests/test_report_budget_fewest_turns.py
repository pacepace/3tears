"""A report budget that cannot admit even one turn is refused before any turn is bought, and one under two is warned
(backlog W6).

A report takes at least ``MIN_TURNS_FOR_A_REPORT`` turns (the analyst's submission and the editor's review). The cap
holds back the next turn's ceiling against what was actually spent, so a two-turn report fits when the budget covers
the first turn's ceiling, and then the first turn's real cost plus the second's ceiling. The arithmetic proves a budget
too small only below the first turn's ceiling: that alone is refused, at $0, saying it cannot admit even one turn and
naming the budget that admits it. The 2026-11-06 acceptance run is the incident: a $0.75 budget and a Sonnet turn whose
ceiling is about $0.67. 36 of its 50 reports ended over budget with no page, and 16 stored one, so $0.75 is admitted.
A writer eval warns once when its budget is under two of the analyst's first-turn ceiling, citing the recorded runs.
A report that still runs out says how many turns it was admitted and how many its budget admits whole at that ceiling.
For an arm whose reports ran out with no page, the writer eval says what the reports that stored one needed: what they
spent, plus one turn's ceiling.

Mutations that turn this file red: a budget above the first turn's ceiling refused; a budget below it admitted, or
refused without naming the budget, the ceiling, the fewest turns and the budget that admits the turn; the rule applied
to a turn after the first; the launch warning missing, or given at a budget two ceilings fit; the launch refusal made
after the store opens; the ending's words dropping the turns admitted or the turns the budget admits; the writer eval's
disclosure missing, giving spend alone, or counting a report that stored a page as one with none.

Every turn is scripted, and nothing reaches a model. The writer eval's arms run on the incident's model, whose
published rates price its launch; each expected figure is read back from the line that states it and checked against
the line's own arithmetic, so no figure here moves with the skill's size.
"""

from __future__ import annotations

import asyncio
import json
import re
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

#: The writer every arm of the writer eval names: the incident's, whose published rates price the launch.
WRITER: Final = "claude-sonnet-5-5"

SCOPE: Final = "fewest-turns"

#: A budget under the incident's first-turn ceiling, which cannot admit even one turn.
TOO_SMALL: Final = 0.5

REFUSED: Final = (
    "this first turn can cost up to $0.67, above the report's $0.50 budget, so it cannot admit even one turn of the 2 "
    "a report takes; a budget of $0.67 or more admits it"
)


def test_a_report_takes_at_least_the_submission_and_the_review() -> None:
    assert MIN_TURNS_FOR_A_REPORT == 2


def test_a_budget_under_the_first_turn_s_ceiling_is_refused_naming_what_admits_the_turn() -> None:
    assert ReportBudget(usd=TOO_SMALL).admit(TurnCeiling(usd=SONNET_CEILING), tokens=1) == REFUSED


@pytest.mark.parametrize("usd", [INCIDENT_BUDGET, ADMITTED_BUDGET, SONNET_CEILING])
def test_a_budget_the_first_turn_fits_is_admitted(usd: float) -> None:
    # $0.75 is under two of $0.67, and is admitted: whether a second turn fits turns on what the first really costs.
    assert ReportBudget(usd=usd).admit(TurnCeiling(usd=SONNET_CEILING), tokens=1) is None


def test_a_budget_in_tokens_and_turns_is_never_refused_for_dollars() -> None:
    budget = ReportBudget(tokens=100_000, turns=40)

    assert budget.first_turn_refusal(TurnCeiling(usd=SONNET_CEILING), priced="this first turn") is None
    assert budget.admit(TurnCeiling(usd=SONNET_CEILING), tokens=1) is None


async def test_the_report_ends_over_budget_before_its_first_turn_with_nothing_sent() -> None:
    analyst = FakeToolTurnClient(reply(submitted()), ceiling=SONNET_CEILING)
    model = FakeTurnModel(analyst=analyst)

    result = await written(model, budget_usd=TOO_SMALL)

    assert result.ending == "over_budget" and result.stored is None
    assert result.outcome.reason == f"the report's budget refused the next turn: {REFUSED}"
    assert analyst.requests == [] and model.asked == ["analyst"]
    assert result.generation["tally"]["total"]["turns"] == 0


async def test_the_incident_s_budget_writes_a_two_turn_report_whose_first_turn_is_cheap() -> None:
    # Under two of the ceiling, and still a report: $0.67 fits against nothing spent, then $0.10 spent and the review's
    # $0.60 ceiling fit in $0.75.
    model = FakeTurnModel(
        analyst=FakeToolTurnClient(reply(submitted(), cost=0.1), ceiling=SONNET_CEILING),
        editor_1=FakeToolTurnClient(reply(review(), cost=0.05), ceiling=0.6),
    )

    result = await written(model, budget_usd=INCIDENT_BUDGET)

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


def _eval(tmp_path: Path, arms: dict[str, Writer], budget_usd: float) -> WriterEval:
    return WriterEval(
        tmp_path / "cases",
        arms=arms,
        judge=FakeScoringJudge(),
        recordings=tmp_path / "turns",
        budget_usd=budget_usd,
        max_cost_usd=10.0,
        held_out_fraction=0.99,
        name=SCOPE,
        store=tmp_path / "evals.sqlite",
    )


def _dollars(text: str) -> float:
    return float(text.removeprefix("$"))


async def test_the_writer_eval_warns_once_and_says_what_the_paged_reports_needed(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = {"two revisions": Writer(WRITER), "one revision": Writer(WRITER, revisions=1)}
    _laid(tmp_path, arms)

    result = await _eval(tmp_path, arms, INCIDENT_BUDGET).run()

    assert offline == []
    # One warning for the launch, citing the records, at the figure two of its own first-turn ceiling come to.
    (warning,) = [line for line in result.disclosures if "recorded Sonnet 5.5" in line]
    twice = re.fullmatch(
        r"the \$0\.75, 40-turn report budget is under 2 of a first turn's ceiling on claude-sonnet-5-5 \((\$[0-9.]+)\), "
        r"so the cap may stop a report before its review; in the recorded Sonnet 5\.5 acceptance runs, at \$0\.75 16 of "
        r"50 reports stored a page, at \$1\.00 17 to 22 of 22, and at \$1\.50 50 of 50, the budget at which no "
        r"recorded report ran out",
        warning,
    )
    assert twice is not None and _dollars(twice[1]) > INCIDENT_BUDGET
    for arm, spent in (("two revisions", 0.3), ("one revision", 0.45)):
        (line,) = [line for line in result.disclosures if line.startswith(f"{arm}: 1 of 2 report(s)")]
        needed = re.fullmatch(
            rf"{arm}: 1 of 2 report\(s\) ended over their \$0\.75, 40-turn budget with no page; the 1 that stored a "
            rf"page under it spent \${spent:.2f}, so under the cap the costliest needed about (\$[0-9.]+), that plus "
            r"one turn's ceiling of at least (\$[0-9.]+)",
            line,
        )
        assert needed is not None, line
        # What it spent plus a turn's ceiling, the ceiling half the warning's figure.
        assert _dollars(needed[1]) == pytest.approx(spent + _dollars(needed[2]), abs=0.011)
        assert _dollars(needed[2]) == pytest.approx(_dollars(twice[1]) / 2, abs=0.011)
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


async def test_the_writer_eval_gives_no_warning_where_two_ceilings_fit(tmp_path: Path, offline: list[object]) -> None:
    arms = {"two revisions": Writer(WRITER), "one revision": Writer(WRITER, revisions=1)}
    _laid(tmp_path, arms)

    result = await _eval(tmp_path, arms, ADMITTED_BUDGET).run()

    assert not [line for line in result.disclosures if "recorded Sonnet 5.5" in line]


async def test_a_writer_eval_whose_budget_cannot_admit_one_turn_is_refused_before_the_store_opens(
    tmp_path: Path, offline: list[object]
) -> None:
    arms = {"two revisions": Writer(WRITER), "one revision": Writer(WRITER, revisions=1)}
    _laid(tmp_path, arms)

    with pytest.raises(ValueError) as refused:
        await _eval(tmp_path, arms, 0.25).run()

    said = re.fullmatch(
        r"the writer eval was not launched, and nothing was sent or spent: the analyst's first turn of arm 'two "
        r"revisions' on claude-sonnet-5-5, priced at its 32,000-token output cap and its text before any snapshot is "
        r"read, can cost up to (\$[0-9.]+), above the report's \$0\.25 budget, so it cannot admit even one turn of the "
        r"2 a report takes; a budget of (\$[0-9.]+) or more admits it: raise budget_usd",
        str(refused.value),
    )
    assert said is not None, str(refused.value)
    assert _dollars(said[1]) > 0.25 and _dollars(said[2]) >= _dollars(said[1])
    assert not (tmp_path / "evals.sqlite").exists() and offline == []
