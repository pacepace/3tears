"""``WriterEval`` under a rule to stop: its cases launch in batches, in the bet's order, into one campaign.

The seven fixture cases prototype 5 wrote over, every one on the held-out side, each arm's writer a scripted recording:
each arm submits the document prototype 5 accepted for the case and its editor clears it, the candidate's method
appendix carrying one more sentence. A scripted judge reads each pair from its prompt alone: one prefers the page with that
sentence, or the one without, wherever it sits, and one gives a mixed preference that reads the same in any store. Every socket
connect fails: nothing here reaches a provider.

Mutations that turn this file red: every case launched before the first look; a later batch's pass asking a pair an
earlier one answered; the batches launched out of the bet's order, so the reading at the cap differs from the one-batch
reading; a stop with its guardrails not shown held left standing, or gone on without saying so; the batch size or the
stop left off the record; a side under the eval's floor refused, or run without saying so.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.fixtures.writer_cases.writer_acceptance import (
    accepted_documents,
    default_cases,
    dry_sessions,
)
from packages.evals.tests.writer_eval_support import JUDGE, FakeScoringJudge, ScoredReply, lay_recording
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.stats import bet_order
from threetears.evals.harness import Writer, WriterEval, WriterResult, read_writer_cases
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.ops import OpsHost, job_poll
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport

#: The writer every arm names.
WRITER: Final = "report-writer-candidate"

#: The control and the candidate.
CONTROL: Final = "shipped"
CANDIDATE: Final = "variant"

#: The cases: prototype 5's seven, all on the held-out side at a 0.99 share.
CASES: Final = 7

#: Where an outright landslide first separates on a bet fixed for seven cases.
LANDSLIDE_AT: Final = 6

#: The batch these tests launch: three, so a landslide stops after two batches, one case short of the seven.
BATCH: Final = 3

SCOPE: Final = "batched"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """No writer key in the environment, and every socket connect refused and counted."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    attempts: list[object] = []

    def connect(_self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise AssertionError(f"a replay-only run connected to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def _variant() -> AnalystSkill:
    """The shipped skill with one blank line added to ``SKILL.md``: another digest, the same instructions."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}\n"
    return AnalystSkill(files=files)


def _arms() -> dict[str, Writer]:
    return {CONTROL: Writer(WRITER, skill=load_skill()), CANDIDATE: Writer(WRITER, skill=_variant())}


#: The sentence the candidate's method appendix adds, by which the marking judges tell its page.
MARK: Final = "The candidate signed this page."


def _lay(tmp_path: Path, arms: dict[str, Writer]) -> Path:
    """Prototype 5's seven cases as case files, and each arm's recording of each: the accepted page, the candidate's
    last method item carrying :data:`MARK`. The cases' directory."""
    cases = default_cases(tmp_path / "cases")
    named = {case.snapshot.id: case.name for case in read_writer_cases(cases)}
    documents = accepted_documents()
    for arm, writer in arms.items():
        routed = writer_turns(tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases=named)
        for snapshot in named:
            page = documents[snapshot]
            if arm == CANDIDATE:
                *kept, last = page["method"]
                page = {**page, "method": [*kept, {**last, "text": f"{last['text']} {MARK}"}]}
            lay_recording(routed.path(WRITER, None, snapshot), dry_sessions(page, model=WRITER))
    return cases


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeMarkedJudge(FakeScoringJudge):
    """The scripted judge, preferring outright, on every side-by-side call, the page carrying :data:`MARK` (or, with
    ``marked=False``, the page without it), wherever it sits; every other dimension it scores as its parent does."""

    def __init__(self, *, marked: bool) -> None:
        super().__init__(JUDGE)
        self.marked = marked

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> ScoredReply:
        if f'"{preference_rubric().name}"' not in system:
            return await super().generate(system=system, user=user, response_format=response_format)
        self.calls.append(preference_rubric().name)
        in_b = user.find(MARK) > user.index("=== Report B ===")
        score = 5 if in_b == self.marked else 1
        answer = json.dumps({"reasoning": "read the pair", "criteria_scores": {preference_rubric().name: score}})
        return ScoredReply(answer, model=self.model_name, served_model=self.model_name)


#: What a stored id looks like inside a prompt: the judge's reading never turns on one.
_IDS: Final = re.compile(r"[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeSplitJudge(FakeScoringJudge):
    """The scripted judge, scoring each call from its prompt with every stored id taken out: a close, mixed
    preference that reads the same in any store."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> ScoredReply:
        match = re.search(r'single key "([^"]+)"', system)
        assert match is not None, "the judge's system prompt names no dimension"
        self.calls.append(match[1])
        score = 1 + int(hashlib.sha256(_IDS.sub("", user).encode()).hexdigest(), 16) % 5
        answer = json.dumps({"reasoning": f"read {match[1]}", "criteria_scores": {match[1]: score}})
        return ScoredReply(answer, model=self.model_name, served_model=self.model_name)


def _eval(tmp_path: Path, cases: Path, arms: dict[str, Writer], **overrides: Any) -> WriterEval:
    arguments: dict[str, Any] = {
        "arms": arms,
        "judge": FakeScoringJudge(JUDGE),
        "recordings": tmp_path / "turns",
        "budget_usd": 0.75,
        "max_cost_usd": 30.0,
        "held_out_fraction": 0.99,
        "name": SCOPE,
        "store": tmp_path / "evals.sqlite",
        "stop_when": "separated",
        "batch_size": BATCH,
    }
    return WriterEval(cases, **{**arguments, **overrides})


async def _report(result: WriterResult) -> WriterPairwiseReport:
    status = await job_poll(OpsHost.reading(result.host), result.pairwise[CANDIDATE].job_id, result.scope_id)
    return WriterPairwiseReport.model_validate(status.result)


async def _runs(result: WriterResult) -> list[str]:
    """The runs the result's campaign holds."""
    campaign = await asyncio.to_thread(result.host.storage.load_campaign, result.campaign_id, SCOPE)
    assert campaign is not None
    return list(campaign.run_ids)


async def test_a_landslide_stops_launching_at_the_first_batch_whose_preference_separates(
    tmp_path: Path, offline: list[object]
) -> None:
    """The judge prefers the control's page on every case: the preference separates toward keep at case 6, in the
    second batch, so the arms launch 6 of the 7 cases and the run says it read its other readings on those 6."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    judge = FakeMarkedJudge(marked=False)

    result = await _eval(tmp_path, cases, arms, judge=judge).run()

    assert result.pairwise_unread == {}
    report = await _report(result)
    stopping = report.stopping
    assert stopping is not None and stopping.rule is not None
    assert (stopping.rule.batch_size, stopping.rule.max_cases) == (BATCH, CASES)
    assert stopping.stopped_at == stopping.first_separated_at == LANDSLIDE_AT
    assert stopping.cases_found == 2 * BATCH < CASES and not stopping.past_the_stop
    # Fewer launches than N: two batches of three, each arm a run per batch, all in the one campaign.
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: 6, CANDIDATE: 6}
    assert len(await _runs(result)) == 2 * len(arms)
    # The cases launched are the bet's first eight, in its order.
    planned = stopping.rule.case_ids
    bet = [planned[index] for index in bet_order(CASES)]
    assert [look.case_id for look in stopping.looks] == bet[:LANDSLIDE_AT]
    assert {pair.case_id for pair in report.pairs} == set(bet[: 2 * BATCH])
    assert result.pairwise[CANDIDATE].primary.outcome == "regressed"
    assert result.decisions == {CANDIDATE: "keep"}
    said = (
        f"Read on 6 cases: the arms ran 6 of the {CASES} held-out cases, {BATCH} at a time in the bet's order, and "
        "stopped once every candidate's preference had separated, so the gated readings and guardrails were read once, "
        "on those 6."
    )
    assert said in result.disclosures
    # The pass asked the six cases up to the stop, each in both orders, once.
    assert report.calls_made == 2 * LANDSLIDE_AT == judge.calls.count(preference_rubric().name)
    assert "across 6 held-out cases" in result.page.model_dump_json()
    assert offline == []
