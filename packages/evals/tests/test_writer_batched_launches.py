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
import dataclasses
import hashlib
import json
import re
import socket
from collections.abc import Collection
from pathlib import Path
from typing import Any, Final

import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.fixtures.writer_cases.writer_acceptance import (
    accepted_documents,
    default_cases,
    dry_sessions,
)
from packages.evals.tests.writer_case_support import SEED
from packages.evals.tests.writer_eval_support import JUDGE, FakeScoringJudge, ScoredReply, lay_recording
from threetears.evals.analysis import EvalSummary, summarize_run, summarize_runs
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.stats import bet_order
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.harness import Writer, WriterCaseFile, WriterEval, WriterResult, read_writer_cases
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops import OpsHost, job_poll
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_contract import writer_scoring_rubric
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport, writer_pairwise
from threetears.evals.ops.report_writer.writer_promotion import writer_promotion
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.schema.models import CaseSetRef
from threetears.evals.storage import SqliteDocumentStore

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


def _variant(lines: int = 1) -> AnalystSkill:
    """The shipped skill with ``lines`` blank lines added to ``SKILL.md``: another digest, the same instructions."""
    files = dict(load_skill().files)
    files["SKILL.md"] = f"{files['SKILL.md']}{chr(10) * lines}"
    return AnalystSkill(files=files)


def _arms() -> dict[str, Writer]:
    return {CONTROL: Writer(WRITER, skill=load_skill()), CANDIDATE: Writer(WRITER, skill=_variant())}


#: The sentence the candidate's method appendix adds, by which the marking judges tell its page.
MARK: Final = "The candidate signed this page."


def _lay(tmp_path: Path, arms: dict[str, Writer], *, marked: Collection[str] = (CANDIDATE,)) -> Path:
    """Prototype 5's seven cases as case files, and each arm's recording of each: the accepted page, the last method
    item of each arm in ``marked`` carrying :data:`MARK`. The cases' directory."""
    cases = default_cases(tmp_path / "cases")
    named = {case.snapshot.id: case.name for case in read_writer_cases(cases)}
    documents = accepted_documents()
    for arm, writer in arms.items():
        routed = writer_turns(tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases=named)
        for snapshot in named:
            page = documents[snapshot]
            if arm in marked:
                *kept, last = page["method"]
                page = {**page, "method": [*kept, {**last, "text": f"{last['text']} {MARK}"}]}
            lay_recording(routed.path(WRITER, None, snapshot), dry_sessions(page, model=WRITER))
    return cases


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


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeMarkedJudge(FakeSplitJudge):
    """The scripted judge, preferring outright, on every side-by-side call, the page carrying :data:`MARK` (or, with
    ``marked=False``, the page without it), wherever it sits; every other dimension it scores as its parent does, from
    its prompt."""

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


def _guarded(store: Path) -> None:
    """Make the first judged dimension of the scope's writer template a guardrail (the boundary axis). Blocking."""
    storage = EvalStorage(SqliteDocumentStore(store))
    template = storage.load_template_by_name("report writer", SCOPE)
    assert template is not None
    first, *rest = template.rubric
    storage.save_template(
        template.model_copy(update={"rubric": [first.model_copy(update={"axis": "boundary"}), *rest]})
    )


async def _report(result: WriterResult, candidate: str = CANDIDATE) -> WriterPairwiseReport:
    status = await job_poll(OpsHost.reading(result.host), result.pairwise[candidate].job_id, result.scope_id)
    return WriterPairwiseReport.model_validate(status.result)


async def _runs(result: WriterResult) -> list[str]:
    """The runs the result's campaign holds."""
    campaign = await asyncio.to_thread(result.host.storage.load_campaign, result.campaign_id, result.scope_id)
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
    # A cap that admits every batch: the run stopped on the preference, not on the cap.
    assert stopping.stopped_on == "separated"
    # Fewer launches than N: two batches of three, each arm a run per batch, all in the one campaign.
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: 6, CANDIDATE: 6}
    assert len(await _runs(result)) == 2 * len(arms)
    # Each arm ran under its part of the $30 cap across its batches: its ceiling is that part, never above the cap.
    assert {summary.max_cost_usd for summary in result.arms.values()} == {30.0 * 0.8 / len(arms)}
    # The cases launched are the bet's first six, in its order.
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


async def test_a_close_run_goes_to_the_cap_and_reads_what_one_batch_of_every_case_reads(
    tmp_path: Path, offline: list[object]
) -> None:
    """A mixed preference never separates, so the arms launch all seven cases, three at a time; the reading at the cap
    is the one the same cases read when every case launches at once, bit for bit, and no pair is asked twice."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    batched_judge, whole_judge = FakeSplitJudge(JUDGE), FakeSplitJudge(JUDGE)

    batched = await _eval(tmp_path, cases, arms, judge=batched_judge).run()
    whole = await _eval(
        tmp_path, cases, arms, judge=whole_judge, batch_size=CASES, store=tmp_path / "whole.sqlite", scope="whole"
    ).run()

    stepped, at_once = await _report(batched), await _report(whole)
    assert stepped.stopping is not None and stepped.stopping.rule is not None
    assert stepped.stopping.stopped_at is None and stepped.stopping.cases_found == CASES
    assert stepped.whole.verdict.outcome == "not_separated"
    assert len(await _runs(batched)) == 3 * len(arms) and len(await _runs(whole)) == len(arms)
    assert stepped.whole.model_dump() == at_once.whole.model_dump()
    assert stepped.finished_only.model_dump() == at_once.finished_only.model_dump()
    # Each store mints its own case ids, in the same order, so the cases and looks are compared by everything else.
    assert at_once.stopping is not None
    assert [case.model_dump(exclude={"case_id"}) for case in stepped.cases] == [
        case.model_dump(exclude={"case_id"}) for case in at_once.cases
    ]
    assert [look.model_dump(exclude={"case_id"}) for look in stepped.stopping.looks] == [
        look.model_dump(exclude={"case_id"}) for look in at_once.stopping.looks
    ]
    # Each pair was asked once in each order, across the three batches' passes.
    asked = preference_rubric().name
    assert batched_judge.calls.count(asked) == whole_judge.calls.count(asked) == 2 * CASES == stepped.calls_made
    assert {arm: summary.n_cases for arm, summary in batched.arms.items()} == {CONTROL: CASES, CANDIDATE: CASES}
    said = (
        f"The arms ran every held-out case, {BATCH} at a time in the bet's order, so the gated readings and "
        "guardrails were read once, on every case."
    )
    assert said in batched.disclosures
    assert batched.decisions == whole.decisions
    assert offline == []


async def test_an_improved_stop_whose_guardrails_are_not_shown_held_goes_on_to_the_cap_and_says_so(
    tmp_path: Path, offline: list[object]
) -> None:
    """The judge prefers the candidate's page on every case, so the preference separates improved at case 6; on the six
    cases run by then its guardrail, a judged dimension the scope's template reads as one, is not shown held, so the
    run launches the last case and reads the gated readings and guardrails a second time, and says so.

    The writer template declares no guardrail ("None is a guardrail", writer_contract), so no run on today's template
    takes this path: the test gives the scope's template one, as a host's own template may declare.
    """
    arms = _arms()
    cases = _lay(tmp_path, arms)
    # The scope's template, made by a first run, then given a guardrail as a host may: its first judged dimension
    # read on the boundary axis, so it is decided on its own, held only when its interval clears no change.
    await _eval(tmp_path, cases, arms, stop_when="never").run()
    await asyncio.to_thread(_guarded, tmp_path / "evals.sqlite")

    result = await _eval(tmp_path, cases, arms, judge=FakeMarkedJudge(marked=True)).run()

    assert "1 of 1 guardrails are not shown held" in result.promotion_why[CANDIDATE]

    report = await _report(result)
    stopping = report.stopping
    assert stopping is not None
    assert stopping.first_separated_at == LANDSLIDE_AT and stopping.stopped_at is None, result.promotion_why
    assert stopping.past_the_stop and stopping.cases_found == CASES
    assert report.whole.cases == CASES and report.whole.verdict.outcome == "improved"
    assert "had a second look" in stopping.line
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: CASES, CANDIDATE: CASES}
    said = (
        f"The arms ran every held-out case, {BATCH} at a time in the bet's order, so the gated readings and "
        "guardrails were read on every case."
    )
    assert said in result.disclosures
    assert f"pairwise pass of {CANDIDATE}: {stopping.line}" in result.render()
    assert offline == []


async def test_a_side_under_the_eval_s_floor_runs_and_its_resolution_line_says_so(
    tmp_path: Path, offline: list[object]
) -> None:
    """Fewer side cases than the eval asks for is a disclosure, never a refusal: the run goes ahead, and the floor's
    line, first among the launch's disclosures after any budget warning, says it is under them."""
    arms = _arms()
    cases = _lay(tmp_path, arms)

    result = await _eval(tmp_path, cases, arms, stop_when="never", fewest_cases=10).run()

    line = "At 7 cases, under the 10 this eval asks for, this run can separate only a preference of 89% or more, held on every case."
    assert result.resolution == line and line in result.disclosures
    assert all(summary.status == "completed" for summary in result.arms.values())
    assert result.pairwise_unread == {}
    assert offline == []


async def test_a_second_batched_run_in_the_scope_reuses_its_batch_sets(tmp_path: Path, offline: list[object]) -> None:
    """The held-out set and the bet are the same on a second run, so its batches list the same cases: each batch's set
    is the one the first run minted, not refused as unchanged and not minted again."""
    arms = _arms()
    cases = _lay(tmp_path, arms)

    first = await _eval(tmp_path, cases, arms, judge=FakeMarkedJudge(marked=False)).run()
    second = await _eval(tmp_path, cases, arms, judge=FakeMarkedJudge(marked=False)).run()

    assert first.decisions == second.decisions == {CANDIDATE: "keep"}
    storage = second.host.storage
    sets = await asyncio.to_thread(storage.query_case_sets, SCOPE)
    batches = sorted((listed.name, listed.version) for listed in sets if not listed.tracked)
    assert [name.rsplit(" batch ", 1)[1] for name, _ in batches] == ["1 of 3", "2 of 3"]
    assert all(version == 1 for _, version in batches)
    assert len(await _runs(second)) == 2 * len(arms)
    assert offline == []


def _rename_batch(storage: EvalStorage, runs: Collection[str], batch: str, ref: CaseSetRef) -> None:
    """Point every run of ``runs`` that ran batch ``batch`` (``1 of 3``) at ``ref`` instead. Blocking."""
    for run_id in runs:
        run = storage.load_eval_run(run_id, SCOPE)
        assert run is not None and run.case_set is not None
        if run.case_set.name.endswith(f" batch {batch}"):
            storage.save_eval_run(run.model_copy(update={"case_set": ref}))


@pytest.mark.parametrize("claimed", ["hand-picked", "other split"])
async def test_a_one_launch_set_that_is_no_batch_of_the_held_out_set_is_still_refused(
    tmp_path: Path, offline: list[object], claimed: str
) -> None:
    """The promotion read accepts a case run from a set made for one launch only as a batch of the held-out set version
    the set's name gives, which lists it. A hand-picked one-launch subset of the held-out cases, or a set named a batch
    of another split's held-out set, is refused as a set no run names, as before batches."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    result = await _eval(tmp_path, cases, arms, judge=FakeMarkedJudge(marked=False)).run()
    storage = result.host.storage
    runs = await _runs(result)
    first = await asyncio.to_thread(storage.load_eval_run, runs[0], SCOPE)
    assert first is not None and first.case_set is not None and first.template_id is not None
    ids = list(first.test_case_ids)
    if claimed == "hand-picked":
        name = "hand-picked held-out cases"
    else:
        other = await asyncio.to_thread(
            mint_case_set,
            storage,
            scope_id=SCOPE,
            name="other-held-out",
            template_id=first.template_id,
            test_case_ids=ids,
        )
        name = f"{other.ref.label} batch 1 of 3"
    foreign = await asyncio.to_thread(
        mint_case_set,
        storage,
        scope_id=SCOPE,
        name=name,
        template_id=first.template_id,
        test_case_ids=ids,
        tracked=False,
    )
    await asyncio.to_thread(_rename_batch, storage, runs, "1 of 3", foreign.ref)
    status = await job_poll(OpsHost.reading(result.host), result.pairwise[CANDIDATE].job_id, SCOPE)

    with pytest.raises(ValidationFailedError, match="which no run names"):
        await asyncio.to_thread(
            writer_promotion, result.host, result.campaign_id, SCOPE, candidate=CANDIDATE, pairwise_job=status
        )
    assert offline == []


async def test_three_arms_each_candidate_stops_at_its_own_stop_and_its_readings_are_read_there_once(
    tmp_path: Path, offline: list[object]
) -> None:
    """One candidate's page loses on every case and separates at case 6; the other's matches the control's, so its
    preference never separates. The first stops launching after the second batch, its pass and its promotion read
    held as they were read there; the control and the second go on to the last case, and the run says each
    candidate's readings were read on the cases launched by its own stop."""
    arms = {**_arms(), "twin": Writer(WRITER, skill=_variant(2))}
    cases = _lay(tmp_path, arms)
    judge = FakeMarkedJudge(marked=False)

    result = await _eval(tmp_path, cases, arms, judge=judge, max_cost_usd=45.0).run()

    loser, twin = await _report(result), await _report(result, "twin")
    assert loser.stopping is not None and twin.stopping is not None
    assert (loser.stopping.stopped_at, loser.stopping.cases_found) == (LANDSLIDE_AT, 2 * BATCH)
    assert (twin.stopping.stopped_at, twin.stopping.cases_found) == (None, CASES)
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: 7, CANDIDATE: 6, "twin": 7}
    # The stopped candidate's read is the one made at its stop, on its own pass there, and was not made again.
    assert result.pairwise[CANDIDATE].read_on == result.pairwise[CANDIDATE].job_id
    assert result.decisions[CANDIDATE] == "keep"
    said = (
        f"The arms ran every held-out case, {BATCH} at a time in the bet's order, and each candidate stopped "
        f"launching at its own stop, so its gated readings and guardrails were read once, on the cases launched by "
        f"then: 6 for {CANDIDATE}, 7 for twin."
    )
    assert said in result.disclosures
    # Every pair was asked once: six of the stopped candidate's, seven of the twin's, each in both orders.
    assert judge.calls.count(preference_rubric().name) == 2 * (LANDSLIDE_AT + CASES)
    assert offline == []


async def test_an_arm_s_batches_summarise_as_one_and_runs_of_two_arms_are_refused(
    tmp_path: Path, offline: list[object]
) -> None:
    """``summarize_runs`` over one arm's batches: its cases, results and spend are the batches' together, its ceiling
    their ceilings summed, its status completed only when each is; one run reads as ``summarize_run`` reads it; and
    two arms' runs, which differ in their overlays, are refused rather than read as one."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    result = await _eval(tmp_path, cases, arms, judge=FakeMarkedJudge(marked=False)).run()
    host = result.host
    runs = await _runs(result)
    by_arm: dict[str, list[str]] = {}
    for run_id in runs:
        run = await asyncio.to_thread(host.storage.load_eval_run, run_id, SCOPE)
        assert run is not None
        by_arm.setdefault(run.overlays["skill"], []).append(run_id)
    control = next(ids for ids in by_arm.values() if len(ids) == 2)
    each = [await asyncio.to_thread(summarize_run, host, run_id, SCOPE) for run_id in control]
    merged = await asyncio.to_thread(summarize_runs, host, control, SCOPE)

    assert merged.n_cases == sum(summary.n_cases for summary in each) == 2 * BATCH
    assert merged.n_results == sum(summary.n_results for summary in each)
    assert merged.total_cost_usd == pytest.approx(sum(summary.total_cost_usd or 0.0 for summary in each))
    assert merged.max_cost_usd == pytest.approx(sum(summary.max_cost_usd or 0.0 for summary in each))
    assert merged.status == "completed" and merged.run_id == control[0]
    alone = await asyncio.to_thread(summarize_runs, host, control[:1], SCOPE)
    assert alone == each[0]
    with pytest.raises(ValueError, match="differ in overlays"):
        await asyncio.to_thread(summarize_runs, host, runs, SCOPE)
    with pytest.raises(ValueError, match="at least one run"):
        await asyncio.to_thread(summarize_runs, host, [], SCOPE)
    assert offline == []


# parity-with: threetears.evals.harness.writer_host.WriterJudge
class FakeCostlyJudge(FakeSplitJudge):
    """The split judge, billing every call ``cost`` and pricing a call that carries a page at that, while it prices a
    call with no page in it, the floor a launch reserves judging at, at ``floor``: a call priced above its floor, as
    every real call with a page is."""

    def __init__(self, *, cost: float, floor: float) -> None:
        super().__init__(JUDGE)
        self.cost = cost
        self.floor = floor

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The floor for a call with nothing in it; ``cost`` for any other."""
        return self.floor if not system and not user else self.cost

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> ScoredReply:
        said = await super().generate(system=system, user=user, response_format=response_format)
        return dataclasses.replace(said, cost_usd=self.cost)


async def _spent_by_arm(result: WriterResult) -> dict[str, list[EvalSummary]]:
    """Each arm's runs' summaries, by the arm's skill digest, in launch order."""
    by_arm: dict[str, list[EvalSummary]] = {}
    for run_id in await _runs(result):
        run = await asyncio.to_thread(result.host.storage.load_eval_run, run_id, SCOPE)
        assert run is not None
        summary = await asyncio.to_thread(summarize_run, result.host, run_id, SCOPE)
        by_arm.setdefault(run.overlays["skill"], []).append(summary)
    return by_arm


async def test_a_spend_cap_that_admits_two_batches_stops_the_run_there_and_says_so_once(
    tmp_path: Path, offline: list[object]
) -> None:
    """Every judge call costs $0.05, its floor too. A batch of two reserves $2.90 of an arm's part (two reports at $0.75
    and 14 judge calls each), and spends about $0.70; a $10 spend cap gives each arm $4, so the second batch fits what
    the first left and a third does not. A preference that never separates stops at the spend cap after four of the
    seven cases, and its line says so once, with what was spent and what the next batch would reserve."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    judge = FakeCostlyJudge(cost=0.05, floor=0.05)

    result = await _eval(tmp_path, cases, arms, judge=judge, batch_size=2, max_cost_usd=10.0).run()

    report = await _report(result)
    stopping = report.stopping
    assert stopping is not None and stopping.stopped_on == "spend_cap"
    assert stopping.cases_found == 4 and stopping.stopped_at is None
    stop = re.search(
        r"Stopped at the spend cap after 4 of 7 cases, \$([0-9.]+) spent: the next batch of 2 would reserve \$2\.90 "
        r"against the \$([0-9.]+) left of (shipped|variant)'s part\.$",
        stopping.line,
    )
    assert stop is not None, stopping.line
    by_arm = await _spent_by_arm(result)
    arms_spent = sum(summary.total_cost_usd or 0.0 for summaries in by_arm.values() for summary in summaries)
    assert float(stop[1]) == pytest.approx(arms_spent + (report.cost_usd or 0.0), abs=0.005)
    assert float(stop[2]) < 2.90
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: 4, CANDIDATE: 4}
    # The other lines refer to the stop, never state it again.
    launch = [line for line in result.disclosures if line.startswith("Read on 4 cases")]
    assert launch == [
        "Read on 4 cases: the arms ran 4 of the 7 held-out cases, 2 at a time in the bet's order, until the spend cap "
        "stopped it (the pass's line says where), so the gated readings and guardrails were read once, on those 4."
    ]
    assert result.pairwise_unread == {}
    assert offline == []


async def test_a_batch_whose_judging_costs_past_its_reservation_leaves_the_next_only_what_was_not_spent(
    tmp_path: Path, offline: list[object]
) -> None:
    """The judge reserves $0.01 a call before any page is read and bills $0.50 for a call with a page, so each batch
    spends past its reservation. Each batch's runs run under the batch's own reservation, never the whole part, and
    the engine stops each run at it (between cells: a cell running when it is crossed finishes). The next batch
    launches only when what the recorded spend leaves of the arm's $12 part holds its reservation, so the third,
    which it does not, is never launched, and the run stops at the spend cap after six cases."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    judge = FakeCostlyJudge(cost=0.50, floor=0.01)

    result = await _eval(tmp_path, cases, arms, judge=judge, max_cost_usd=30.0).run()

    part = 30.0 * 0.8 / len(arms)
    per_case = 0.75 + len(writer_scoring_rubric()) * JUDGE_CALL_ATTEMPTS * 0.01
    by_arm = await _spent_by_arm(result)
    assert len(by_arm) == len(arms)
    for summaries in by_arm.values():
        assert [summary.n_cases for summary in summaries] == [BATCH, BATCH]
        before = 0.0
        for summary in summaries:
            # Held to its batch's reservation, never the whole part, and launched only into what was left.
            assert summary.max_cost_usd == pytest.approx(BATCH * per_case)
            assert before + BATCH * per_case <= part + 1e-9
            assert summary.status == "budget_stopped"
            before += summary.total_cost_usd or 0.0
        # What the two batches spent leaves less than the last batch's reservation, so it was not launched.
        assert part - before < 1 * per_case
    report = await _report(result)
    assert report.stopping is not None and report.stopping.stopped_on == "spend_cap"
    # Six launched; the cells each run's cap left unstarted wrote no pair, and the line says so beside the stop.
    assert "Stopped at the spend cap after 6 of 7 cases" in report.stopping.line
    assert "the rule planned were not found" in report.stopping.line
    assert offline == []


async def test_a_cap_that_admits_no_batch_is_refused_naming_the_smallest_that_would(
    tmp_path: Path, offline: list[object]
) -> None:
    """The one cap refused: a $3 cap gives each arm $1.20, under the first batch's two reports at $0.75. The refusal
    names the smallest cap that admits it, two arms' $1.50 over the four fifths the arms share: $3.75."""
    arms = _arms()
    cases = _lay(tmp_path, arms)

    with pytest.raises(ValueError) as refused:
        await _eval(tmp_path, cases, arms, batch_size=2, max_cost_usd=3.0).run()

    assert str(refused.value) == (
        "the first batch's 2 report(s) at $0.75 each could pass an arm's $1.20 part of max_cost_usd $3.00, after the "
        "pairwise pass's $0.60 (pairwise_share 0.2); a max_cost_usd of $3.75 or more admits it, or lower budget_usd"
    )
    storage = EvalStorage(SqliteDocumentStore(tmp_path / "evals.sqlite"))
    assert await asyncio.to_thread(storage.query_eval_runs, SCOPE) == [], "no arm was launched"
    assert offline == []


#: Seeded fixtures prototype 5 wrote no page over: each arm writes nothing on them, so nothing of theirs is judged.
UNJUDGED: Final = ("r1", "r2", "r3", "r4")


def _lay_unjudged(tmp_path: Path, cases: Path, arms: dict[str, Writer]) -> None:
    """Add :data:`UNJUDGED` as case files, and each arm's recording of each: no page."""
    snapshots = fixture_snapshots("frozen-elsewhere")
    for name in UNJUDGED:
        group, withhold = SEED[name]
        WriterCaseFile(name=name, group=group, withhold=list(withhold), snapshot=snapshots[name]).write(
            cases / f"{name}.json.gz"
        )
    named = {snapshots[name].id: name for name in UNJUDGED}
    for arm, writer in arms.items():
        routed = writer_turns(tmp_path / "turns", arm=arm, skill=writer.analyst_skill, cases=named)
        for snapshot in named:
            lay_recording(routed.path(WRITER, None, snapshot), dry_sessions(None, model=WRITER))


def _witnessed(storage: EvalStorage, run_id: str) -> None:
    """Record one run as witnessed rather than commissioned: another rig. Blocking."""
    run = storage.load_eval_run(run_id, SCOPE)
    assert run is not None
    storage.save_eval_run(run.model_copy(update={"apparatus_provenance": "witnessed"}))


async def _verdicts(result: WriterResult) -> list[tuple[str, str, float | None]]:
    """The candidate's gated readings, as its promotion read reads them: each one's kind, name and level."""
    status = await job_poll(OpsHost.reading(result.host), result.pairwise[CANDIDATE].job_id, SCOPE)
    read = await asyncio.to_thread(
        writer_promotion, result.host, result.campaign_id, SCOPE, candidate=CANDIDATE, pairwise_job=status
    )
    return sorted((verdict.kind, verdict.name, verdict.interval_level) for verdict in read.verdicts)


async def test_a_batch_that_stored_no_page_on_either_arm_adds_no_second_set_of_gated_readings(
    tmp_path: Path, offline: list[object]
) -> None:
    """One case a batch, four of them unjudged: a batch that stored no page on either arm judged nothing, so its runs
    record no judge. They stay out of the campaign's readings, said once, and the pass reads their pairs as ties: the
    gated readings, and each one's level, are those of the same run without the four cases."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    alone = await _eval(tmp_path, cases, arms, judge=FakeSplitJudge(JUDGE), batch_size=1, max_cost_usd=60.0).run()
    _lay_unjudged(tmp_path, cases, arms)

    result = await _eval(
        tmp_path,
        cases,
        arms,
        judge=FakeSplitJudge(JUDGE),
        batch_size=1,
        max_cost_usd=60.0,
        store=tmp_path / "with.sqlite",
    ).run()

    assert result.pairwise_unread == {}
    report = await _report(result)
    assert report.stopping is not None and report.stopping.stopped_on == "all_cases"
    assert sum(1 for pair in report.pairs if pair.missing == "both") == len(UNJUDGED)
    assert len(report.outside_run_ids) == len(UNJUDGED) * len(arms)
    assert set(report.outside_run_ids).isdisjoint(await _runs(result))
    said = (
        "4 cases, in batches that stored no page on either arm, are read side by side as ties and left out of the "
        "gated readings, which no judge was asked of on them."
    )
    assert result.disclosures.count(said) == 1
    assert (
        "The arms ran every held-out case, 1 at a time in the bet's order, so the gated readings and guardrails were "
        "read once, on every case but the 4 that stored no page."
    ) in result.disclosures
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: 11, CANDIDATE: 11}
    with_them, without = await _verdicts(result), await _verdicts(alone)
    assert with_them == without and len(with_them) == len({name for _, name, _ in with_them})
    assert offline == []


async def test_a_pair_s_two_runs_must_share_a_rig_and_an_unjudged_batch_makes_no_second_one(
    tmp_path: Path, offline: list[object]
) -> None:
    """One case a batch, four of them unjudged and so left out of the campaign's readings: each arm sits under one rig.
    Given a second rig on one batch's two runs, each arm sits under two and each pair's two runs still share one, so
    the pass reads them; moved crosswise, a pair's two runs under two rigs, the pass refuses, as it refuses two rigs in
    one run."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    _lay_unjudged(tmp_path, cases, arms)

    result = await _eval(tmp_path, cases, arms, judge=FakeSplitJudge(JUDGE), batch_size=1, max_cost_usd=60.0).run()

    assert result.pairwise_unread == {}

    async def rigs() -> list[int]:
        bundle = await asyncio.to_thread(assemble_campaign_bundle, result.host, result.campaign_id, SCOPE)
        under: dict[str, set[str]] = {}
        for aspect in bundle.aspects:
            for cell in aspect.cells:
                under.setdefault(cell.variant_key, set()).add(cell.apparatus_class_id)
        return [len(each) for each in under.values()]

    assert await rigs() == [1, 1]
    report = await _report(result)
    judged = [pair for pair in report.pairs if pair.missing is None][:2]
    storage = result.host.storage

    async def run_of(result_id: str) -> str:
        stored = await asyncio.to_thread(storage.load_eval_result, result_id, SCOPE)
        assert stored is not None
        return stored.eval_run_id

    # One batch's two runs witnessed: a second rig for each arm, every pair still within one.
    for result_id in (judged[0].control_result_id, judged[0].candidate_result_id):
        await asyncio.to_thread(_witnessed, storage, await run_of(result_id))
    assert await rigs() == [2, 2]
    reading = OpsHost.reading(result.host)
    await writer_pairwise(reading, result.campaign_id, SCOPE, candidate=CANDIDATE)
    # Crosswise: another batch's control run witnessed alone, so that pair's two runs read two rigs.
    await asyncio.to_thread(_witnessed, storage, await run_of(judged[1].control_result_id))
    with pytest.raises(ValidationFailedError, match="a pair compares two pages written under one rig"):
        await writer_pairwise(reading, result.campaign_id, SCOPE, candidate=CANDIDATE)
    assert offline == []
