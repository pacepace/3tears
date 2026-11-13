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
from packages.evals.tests.offline_support import Connect, refuse_connects
from packages.evals.tests.writer_case_support import SEED
from packages.evals.tests.writer_eval_support import JUDGE, FakeScoringJudge, ScoredReply, lay_recording
from threetears.evals.analysis import EvalSummary, summarize_run, summarize_runs
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.stats import bet_order
from threetears.evals.analysis.report.blocks import Verdict
from threetears.evals.analysis.service import assemble_campaign_bundle, campaign_verdicts
from threetears.evals.harness import Writer, WriterCaseFile, WriterEval, WriterResult, read_writer_cases
from threetears.evals.harness.writer_turns import writer_turns
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.errors import StorageError, ValidationFailedError
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.ops import OpsHost, job_poll
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_contract import (
    BOUNCES_MEASURE,
    FINISHED_MEASURE,
    MUST_FIX_OPEN_MEASURE,
    writer_scoring_rubric,
)
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport, writer_pairwise
from threetears.evals.ops.report_writer.writer_promotion import writer_promotion
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.schema import EvalRun
from threetears.evals.schema.models import CaseSetRef, EvalResult, EvalTrace
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
def offline(monkeypatch: pytest.MonkeyPatch) -> list[Connect]:
    """No writer key in the environment, and every socket connect refused and counted."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return refuse_connects(monkeypatch)


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
    tmp_path: Path, offline: list[Connect]
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
    tmp_path: Path, offline: list[Connect]
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
    tmp_path: Path, offline: list[Connect]
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
    tmp_path: Path, offline: list[Connect]
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


async def test_a_second_batched_run_in_the_scope_reuses_its_batch_sets(tmp_path: Path, offline: list[Connect]) -> None:
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
    tmp_path: Path, offline: list[Connect], claimed: str
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
    tmp_path: Path, offline: list[Connect]
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
    tmp_path: Path, offline: list[Connect]
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
    tmp_path: Path, offline: list[Connect]
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


async def _runs_by_arm(result: WriterResult) -> dict[str, list[EvalRun]]:
    """Each arm's runs, by the arm's skill digest, in launch order."""
    by_arm: dict[str, list[EvalRun]] = {}
    for run_id in await _runs(result):
        run = await asyncio.to_thread(result.host.storage.load_eval_run, run_id, SCOPE)
        assert run is not None
        by_arm.setdefault(run.overlays["skill"], []).append(run)
    return by_arm


def _scoring_calls(judge: FakeSplitJudge) -> int:
    """The judge's calls on the writer's scoring rubric, the pass's preference calls left out."""
    return len([dim for dim in judge.calls if dim != preference_rubric().name])


async def _scored_by_run(result: WriterResult, runs: list[EvalRun]) -> list[int]:
    """How many cases each run judged, in launch order, each judged whole; every case it set aside asked nothing and
    names the cap."""
    scored: list[int] = []
    for run in runs:
        rows = await asyncio.to_thread(result.host.storage.query_eval_results_by_run, run.id, SCOPE)
        for row in rows:
            if row.judge_error is not None:
                assert row.rubric_scores == [], "a case set aside asked the judge nothing"
                assert "the run's cost cap had too little left" in row.judge_error
            else:
                assert len(row.rubric_scores) == len(writer_scoring_rubric()), "a judged case was judged whole"
        scored.append(len([row for row in rows if row.judge_error is None]))
    return scored


async def _held_within_the_part(result: WriterResult, part: float, per_case: float) -> dict[str, list[EvalSummary]]:
    """Each arm's runs, after holding each one to what its arm's part had left when it launched, which its cap is, and
    which its spend stays within; each batch launched only when its reservation fit what was left; the arm within its
    part."""
    by_arm = await _spent_by_arm(result)
    for summaries in by_arm.values():
        before = 0.0
        for summary, batch in zip(summaries, (BATCH, BATCH, CASES - 2 * BATCH), strict=True):
            assert before + batch * per_case <= part + 1e-9, "launched only when its reservation fit what was left"
            assert summary.max_cost_usd == pytest.approx(part - before), "run under all its arm's part had left"
            assert summary.total_cost_usd is not None and summary.total_cost_usd <= summary.max_cost_usd
            before += summary.total_cost_usd
        assert before <= part
    return by_arm


async def test_a_batch_whose_judging_costs_past_its_reservation_leaves_the_next_only_what_was_not_spent(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """The judge reserves $0.01 a call before any page is read and bills $0.50 for a call with a page, so a case's
    judging is priced at $7 (seven dimensions, two attempts each) and billed $3.50, far past the $2.67 a batch of three
    reserves at the floor. The pass has room for every pair. The reservation decides only whether a batch launches;
    each arm's run runs under all its $12 part has left, and the engine admits a case's judging against that before paying for any of it. The first
    batch judges two cases whole and sets the third aside; the second and third launch into what is left, which no
    case's judging fits, so they set theirs aside. No run spends past its cap, and the arm stays within its part."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    judge = FakeCostlyJudge(cost=0.50, floor=0.01)

    # The pass's part is wide enough for every pair at the page's price, so only the arms' parts decide the batches.
    result = await _eval(tmp_path, cases, arms, judge=judge, max_cost_usd=40.0, pairwise_share=0.4).run()

    part = 40.0 * 0.6 / len(arms)
    per_case = 0.75 + len(writer_scoring_rubric()) * JUDGE_CALL_ATTEMPTS * 0.01
    by_arm = await _held_within_the_part(result, part, per_case)
    runs = await _runs_by_arm(result)
    judged = 0
    for skill, summaries in by_arm.items():
        assert [summary.status for summary in summaries] == ["budget_stopped"] * 3
        for run in runs[skill]:
            (stop,) = run.budget_stops or ()
            assert stop.currency == "cost_usd" and stop.landed_final <= stop.ceiling and stop.cells_not_judged >= 1
        scored = await _scored_by_run(result, runs[skill])
        assert scored == [2, 0, 0]
        judged += sum(scored)
    assert _scoring_calls(judge) == judged * len(writer_scoring_rubric())
    # Every batch run joins its arm's one test setup. A case the cap set aside wrote its report: Report finished counts
    # it as finished, and only the judged readings leave it out, under the cap's own cause.
    await _finished_reads_every_case(result, judged_per_arm=2)
    assert offline == []


async def _rows_by_arm(result: WriterResult) -> dict[str, list[EvalResult]]:
    """Every case each arm ran, as its stored rows, by the arm's variant key."""
    rows: dict[str, list[EvalResult]] = {}
    for run_id in await _runs(result):
        for row in await asyncio.to_thread(result.host.storage.query_eval_results_by_run, run_id, SCOPE):
            rows.setdefault(row.variant_key, []).append(row)
    return rows


async def _finished_reads_every_case(result: WriterResult, *, judged_per_arm: int | None = None) -> None:
    """Each arm is one test setup whose Report finished reads every case it ran, as the case went (a report written and
    cleared, or not), the cases the cap set aside among them; and whose judged readings leave out only the set-aside
    cases, under ``cost_cap``. ``judged_per_arm``, when given, is how many cases each arm had judged."""
    bundle = await asyncio.to_thread(assemble_campaign_bundle, result.host, result.campaign_id, SCOPE)
    (aspect,) = bundle.aspects
    cells = aspect.cell_measures
    assert len(cells) == len(result.arms) == len({cell.variant_key for cell in cells}), "one test setup an arm"
    trials = {arm.variant_key: arm for arm in aspect.exclusions}
    rows = await _rows_by_arm(result)
    for cell in cells:
        (finished,) = [summary for summary in cell.measures.measures if summary.name == FINISHED_MEASURE]
        ran = rows[cell.variant_key]
        assert finished.n == len(ran) == trials[cell.variant_key].n_trials, "every case it ran, set aside or not"
        did = [row.host_measures.get(FINISHED_MEASURE) is True for row in ran]
        assert finished.n_true == sum(did), "what the cases did"
        assert finished.rate == pytest.approx(sum(did) / len(did))
        set_aside = [row for row in ran if row.judging_set_aside]
        assert all(row.host_measures.get(FINISHED_MEASURE) is True for row in set_aside), "each wrote its report"
        failed = sum(1 for row in ran if classify_result(row) is ResultOutcome.CANDIDATE_FAIL)
        assert cell.n_candidate_failed == failed, "no case set aside counts as a candidate failure"
        assert not any(classify_result(row) is ResultOutcome.CANDIDATE_FAIL for row in set_aside)
        assert trials[cell.variant_key].by_cause == ({"cost_cap": len(set_aside)} if set_aside else {})
        if judged_per_arm is not None:
            assert trials[cell.variant_key].n_usable == judged_per_arm


async def test_an_unwritten_report_and_cases_the_cap_set_aside_share_a_run_and_report_finished_reads_both(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """The judge bills $0.50 a call, so each arm has two cases judged and the cap sets the rest aside, and the candidate
    writes no report on three of the seven cases. Every batch run joins its arm's setup, and Report finished reads every
    case each arm ran as it went: an unwritten report as a failure, a set-aside case as finished, since it wrote and
    cleared its report. Only the judged readings leave the set-aside cases out."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    named = {case.snapshot.id: case.name for case in read_writer_cases(cases)}
    routed = writer_turns(tmp_path / "turns", arm=CANDIDATE, skill=arms[CANDIDATE].analyst_skill, cases=named)
    for silent in sorted(named)[:3]:
        lay_recording(routed.path(WRITER, None, silent), dry_sessions(None, model=WRITER))
    judge = FakeCostlyJudge(cost=0.50, floor=0.01)

    result = await _eval(tmp_path, cases, arms, judge=judge, max_cost_usd=40.0, pairwise_share=0.4).run()

    rows = [row for ran in (await _rows_by_arm(result)).values() for row in ran]
    # Seven cases, at most two not started an arm: at least one of the three unwritten reports ran.
    assert any(row.host_measures.get(FINISHED_MEASURE) is False for row in rows), "an unwritten report ran"
    assert any(row.judging_set_aside for row in rows), "and the cap set cases aside"
    await _finished_reads_every_case(result)
    assert offline == []


async def test_judging_billed_far_above_its_floor_with_room_in_the_arm_is_judged_in_every_batch(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """The judge reserves $0.01 a call before any page is read and bills $0.15 for a call with a page, fifteen times its
    floor: a case's judging is priced at $2.10 and billed $1.05, while a batch of three reserves $2.67 in all. Each
    arm's $12 part holds every case's judging, so every batch's run, under what its arm's part has left, judges every
    case it runs: nothing is set aside, no run is stopped, and the arm stays within its part."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    judge = FakeCostlyJudge(cost=0.15, floor=0.01)

    result = await _eval(tmp_path, cases, arms, judge=judge, max_cost_usd=30.0).run()

    part = 30.0 * 0.8 / len(arms)
    per_case = 0.75 + len(writer_scoring_rubric()) * JUDGE_CALL_ATTEMPTS * 0.01
    by_arm = await _held_within_the_part(result, part, per_case)
    runs = await _runs_by_arm(result)
    for skill, summaries in by_arm.items():
        assert [summary.status for summary in summaries] == ["completed"] * 3
        assert all(not run.budget_stops for run in runs[skill])
        assert await _scored_by_run(result, runs[skill]) == [BATCH, BATCH, CASES - 2 * BATCH]
    assert _scoring_calls(judge) == len(arms) * CASES * len(writer_scoring_rubric())
    assert offline == []


async def test_judging_billed_far_above_its_floor_ends_each_batch_within_one_call_of_its_cap(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """The same judge, fifteen times its floor, under a $15 cap: each arm's part is $6. The first batch's run, under the
    whole $6, judges its three cases (about $3.21). The second launches into the $2.79 left, which holds its $2.67
    reservation: its first case's $2.10 judging fits, the second's does not fit beside what the first spent, so it is
    set aside and the run stops. The last batch launches into what is left, which no case's judging fits. Each case's
    judging is admitted against what its run has left before any call is paid for, so every run ends within its cap
    (and so within one call of it), and the arm within its part."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    judge = FakeCostlyJudge(cost=0.15, floor=0.01)

    result = await _eval(tmp_path, cases, arms, judge=judge, max_cost_usd=15.0).run()

    part = 15.0 * 0.8 / len(arms)
    per_case = 0.75 + len(writer_scoring_rubric()) * JUDGE_CALL_ATTEMPTS * 0.01
    by_arm = await _held_within_the_part(result, part, per_case)
    runs = await _runs_by_arm(result)
    judged = 0
    for skill, summaries in by_arm.items():
        assert [summary.status for summary in summaries] == ["completed", "budget_stopped", "budget_stopped"]
        for summary, run in zip(summaries[1:], runs[skill][1:], strict=True):
            (stop,) = run.budget_stops or ()
            assert stop.landed_final == pytest.approx(summary.total_cost_usd)
            assert stop.landed_final <= stop.ceiling + 0.15, "within one call of the cap"
            assert stop.cells_not_judged >= 1
        scored = await _scored_by_run(result, runs[skill])
        assert scored == [BATCH, 1, 0]
        judged += sum(scored)
    assert _scoring_calls(judge) == judged * len(writer_scoring_rubric())
    assert offline == []


async def test_a_cap_that_admits_no_batch_is_refused_naming_the_smallest_that_would(
    tmp_path: Path, offline: list[Connect]
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


async def _finished(result: WriterResult, name: str = FINISHED_MEASURE) -> Verdict:
    """The candidate's one contrast against the control on ``name`` (Report finished unless named), as the campaign's
    verdicts read it."""
    verdicts = await asyncio.to_thread(campaign_verdicts, result.host, result.campaign_id, SCOPE)
    (found,) = [
        verdict
        for verdict in verdicts
        if verdict.name == name and verdict.kind == "contrast" and verdict.arm == CANDIDATE
    ]
    return found


async def test_a_batch_that_stored_no_page_on_either_arm_joins_its_arms_setup_and_adds_no_second_set_of_readings(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """One case a batch, four of them unjudged: a batch that stored no page on either arm judged nothing, yet its runs
    join the campaign and their arms' test setups, said once, and the pass reads their pairs as ties. The gated
    readings are the same ones, each read once, as on the same run without the four cases, and Report finished counts
    the four unwritten reports against each arm."""
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
    assert report.outside_run_ids == []
    assert len(await _runs(result)) == (CASES + len(UNJUDGED)) * len(arms)
    said = (
        "4 cases stored no page on at least one arm. A batch run that stored none, 4 of shipped's and 4 of variant's, "
        "is left out of the judged readings, since no judge was asked of it; its unwritten reports count against its "
        "arm in Report finished and side by side, where a missing page loses its pair, or ties it when both are."
    )
    assert result.disclosures.count(said) == 1
    assert (
        "The arms ran every held-out case, 1 at a time in the bet's order, so the gated readings and guardrails were "
        "read once, on every case."
    ) in result.disclosures
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: 11, CANDIDATE: 11}
    with_them, without = await _verdicts(result), await _verdicts(alone)
    assert [(kind, name) for kind, name, _ in with_them] == [(kind, name) for kind, name, _ in without]
    assert len(with_them) == len({name for _, name, _ in with_them})
    # Only the readings an unwritten report carries a value on move: whether the report was finished, and the
    # submissions sent back on the way to writing none. Every other reading, each judged one among them, reads at the
    # level it reads without the four cases.
    moved = {FINISHED_MEASURE, BOUNCES_MEASURE}
    assert [verdict for verdict in with_them if verdict[1] not in moved] == [
        verdict for verdict in without if verdict[1] not in moved
    ]
    assert moved <= {name for _, name, _ in with_them}
    finished = await _finished(result)
    assert finished.delta == pytest.approx(0.0), "both arms left the same four reports unwritten"
    bounced = await _finished(result, BOUNCES_MEASURE)
    assert bounced.delta == pytest.approx(0.0), "both arms sent back the same submissions on the four cases"
    assert offline == []


async def test_a_pair_s_two_runs_must_share_a_rig_and_an_unjudged_batch_makes_no_second_one(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """One case a batch, four of them unjudged: their runs join the campaign, and each arm sits under one rig. Given a
    second rig on one batch's two runs, each arm sits under two and each pair's two runs still share one, so the pass
    reads them; moved crosswise, a pair's two runs under two rigs, the pass refuses, as it refuses two rigs in one
    run."""
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


async def test_a_batch_run_that_stored_no_page_on_one_arm_lowers_that_arm_s_report_finished_and_is_named(
    tmp_path: Path, offline: list[Connect]
) -> None:
    """One case a batch, and the candidate writes no page on one of them while the control does: the candidate's run
    of that batch judged nothing, yet it joins the campaign and the candidate's one test setup. Its unwritten report
    counts as a failure in Report finished, so the candidate finishes one report fewer than the control there; the
    pass reads the pair as the control's; and the run names the arm, said once with its count."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    named = {case.snapshot.id: case.name for case in read_writer_cases(cases)}
    silent = sorted(named)[0]
    routed = writer_turns(tmp_path / "turns", arm=CANDIDATE, skill=arms[CANDIDATE].analyst_skill, cases=named)
    lay_recording(routed.path(WRITER, None, silent), dry_sessions(None, model=WRITER))

    result = await _eval(tmp_path, cases, arms, judge=FakeSplitJudge(JUDGE), batch_size=1, max_cost_usd=60.0).run()

    assert result.pairwise_unread == {}
    report = await _report(result)
    (lost,) = [pair for pair in report.pairs if pair.missing is not None]
    assert lost.missing == "candidate"
    assert report.outside_run_ids == []
    assert len(await _runs(result)) == CASES * len(arms)
    finished = await _finished(result)
    assert finished.delta == pytest.approx(-1 / CASES), "the candidate's unwritten report is a failure to finish"
    said = (
        "1 case stored no page on at least one arm. A batch run that stored none, 1 of variant's, is left out of the "
        "judged readings, since no judge was asked of it; its unwritten reports count against its arm in Report "
        "finished and side by side, where a missing page loses its pair, or ties it when both are."
    )
    assert result.disclosures.count(said) == 1
    assert said in result.render()
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {CONTROL: CASES, CANDIDATE: CASES}
    assert offline == []


async def test_a_batch_run_that_stored_no_result_at_all_is_named_by_its_arm_and_count(
    tmp_path: Path, offline: list[Connect], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One case a batch, the candidate writes no page on one of them, and the store refuses that cell's write, so the
    candidate's run of that batch stores no result at all: the arm is still named, by the case its run launched, and
    the case is counted, never left out of the sentence."""
    arms = _arms()
    cases = _lay(tmp_path, arms)
    named = {case.snapshot.id: case.name for case in read_writer_cases(cases)}
    silent = sorted(named)[0]
    routed = writer_turns(tmp_path / "turns", arm=CANDIDATE, skill=arms[CANDIDATE].analyst_skill, cases=named)
    lay_recording(routed.path(WRITER, None, silent), dry_sessions(None, model=WRITER))
    save_run_cell = EvalStorage.save_run_cell

    def refusing(storage: EvalStorage, result: EvalResult, trace: EvalTrace | None = None) -> None:
        if result.host_measures.get(FINISHED_MEASURE) is False and MUST_FIX_OPEN_MEASURE not in result.host_measures:
            raise StorageError(f"refused the write of cell {result.id}")
        save_run_cell(storage, result, trace)

    monkeypatch.setattr(EvalStorage, "save_run_cell", refusing)

    result = await _eval(tmp_path, cases, arms, judge=FakeSplitJudge(JUDGE), batch_size=1, max_cost_usd=60.0).run()

    said = (
        "1 case stored no page on at least one arm. A batch run that stored none, 1 of variant's, is left out of the "
        "judged readings, since no judge was asked of it; its unwritten reports count against its arm in Report "
        "finished and side by side, where a missing page loses its pair, or ties it when both are."
    )
    storage = result.host.storage
    stored = {
        run_id: await asyncio.to_thread(storage.query_eval_results_by_run, run_id, SCOPE)
        for run_id in await _runs(result)
    }
    assert sum(1 for results in stored.values() if not results) == 1, "the precondition: one run stored no result"
    assert [line for line in result.disclosures if "stored no page on at least one arm" in line] == [said]
    assert offline == []
