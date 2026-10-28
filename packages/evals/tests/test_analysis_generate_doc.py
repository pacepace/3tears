"""``analysis_generate`` writes the campaign's report with the report writer, on the host's ``analyst`` model and budget.

Contracts held here (``threetears.evals.ops.analysis_generate``, ``analysis_estimate``,
``threetears.evals.analysis.doc_analysis``, ``threetears.evals.kernel.inner_ceilings.writer_ceiling_s``):

- **a stored report is linked**: the job mints an analysis naming the stored document (``report_doc_id``, ``basis``
  ``doc``), its provenance the analyst skill's digest, and records the attempt ``stored``; its priced spend is ledgered
  under purpose ``analysis``, an unpriced share as unpriced, never as $0;
- **the recommended variant** is the arm's variant key when ``recommends`` names exactly one arm, and ``None`` for none
  or two or more;
- **insights** come only from claims marked ``durable``, each citing the runs behind its figure keys;
- **an ending that stores nothing** (``refused``, ``failed``, a store that refuses the document) fails the job with
  its ending and reason, the turns' tally kept on the attempt;
- **a host with no report budget** is refused in the call, saying where to set it, with nothing recorded or sent;
- **an ``analyst`` client that is not a turn model** ends the job refused before anything is frozen or sent, naming
  the two methods a wrapper forwards, the client released;
- **``writer_ceiling_s`` bounds the worst case**: every round, revision, bounce and review used sends exactly the turns
  it counts, so one fewer would not bound it;
- **no planner call** is reachable from this path;
- **readers**: the listing, the campaign's page and a later snapshot's ``campaign_overview`` read a ``doc`` analysis.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

import threetears.evals.analysis.planned_analysis as planned_analysis
import threetears.evals.analysis.service as analysis_service
import threetears.evals.ops.analysis as analysis_ops
from packages.evals.tests.doc_writer_support import (
    ANALYST_MODEL,
    TOY_DOC,
    FakeAnalystClient,
    passing_review,
    scripted_reply,
    submission,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import (
    CALLER,
    REPORT_BUDGET_USD,
    TOYHOST_SCOPE,
    WRITER,
    OpsFixture,
    generate,
    ops_fixture,
    settled,
)
from threetears.evals.actions.report_writing import write_report
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.out_of_run_spend import (
    OutOfRunSpendTotals,
    out_of_run_spend_text,
    scope_out_of_run_spend,
)
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend
from threetears.evals.analysis.doc_analysis import doc_analysis
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.campaign import EvalAnalysis, EvalAnalysisAttempt
from threetears.evals.kernel.inner_ceilings import writer_ceiling_s, writer_turns
from threetears.evals.kernel.report_doc import EvalReportDoc
from threetears.evals.kernel.storage import StorageError
from threetears.evals.ops import (
    AnalysisGeneration,
    JobStatus,
    OpsHost,
    ReportBudget,
    WriterLimits,
    analyses_list,
    analysis_estimate,
    analysis_generate,
    report_read,
)
from threetears.evals.ops.analyst import report_snapshot_freeze, snapshot_input
from threetears.evals.ops.analyst_facts import campaign_overview
from threetears.evals.ops.report_writer.analyst import AnalystLimits
from threetears.evals.ops.report_writer.editor import EditorLimits
from threetears.evals.ops.report_writer.tally import Tally, TurnCost
from threetears.evals.run import Ceilings
from threetears.evals.schema.turns import TurnReply, TurnText, TurnToolUse

# -- the job, scripted -------------------------------------------------------------------------------------------------


def _scripted(fixture: OpsFixture, script: Mapping[str, Sequence[TurnReply | Exception]] | None = None) -> None:
    """Make every ``analyst`` client the fixture's host builds run ``script`` (the default: submitted, cleared)."""
    built = fixture.analysts
    factory = fixture.host.eval_host.clients
    assert factory is not None

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> Any:
        if role != "analyst":
            raise AssertionError(f"the doc path built a {role!r} client; it runs on the analyst role alone")
        analyst = FakeAnalystClient(script)
        built.append(analyst)
        return analyst

    object.__setattr__(fixture.host.eval_host, "clients", clients)


async def _generated(fixture: OpsFixture) -> JobStatus:
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    return await settled(fixture.host, job.job_id)


def _attempt(fixture: OpsFixture, status: JobStatus) -> EvalAnalysisAttempt:
    attempt_id = status.job_id.rsplit(":", 1)[1]
    stored, _ = off_loop(fixture.host.eval_host.storage.load_job_record, EvalAnalysisAttempt, attempt_id, TOYHOST_SCOPE)
    assert stored is not None
    return stored


def _analysis(fixture: OpsFixture, analysis_id: str | None) -> EvalAnalysis:
    assert analysis_id is not None
    stored = off_loop(fixture.host.eval_host.storage.load_analysis, analysis_id, TOYHOST_SCOPE)
    assert stored is not None
    return stored


def _ledger(fixture: OpsFixture) -> list[Any]:
    return list(off_loop(fixture.host.eval_host.storage.query_out_of_run_spend, TOYHOST_SCOPE))


async def test_a_stored_report_is_linked_by_a_doc_analysis_with_the_skill_as_its_provenance() -> None:
    fixture = off_loop(ops_fixture)
    status = await _generated(fixture)
    assert (status.state, status.status) == ("completed", "stored"), status
    analysis = _analysis(fixture, status.analysis_id)
    assert analysis.basis == "doc"
    assert analysis.document is None and analysis.report_plan_id is None
    stored = off_loop(fixture.host.eval_host.storage.load_report_doc, analysis.report_doc_id, TOYHOST_SCOPE)
    assert isinstance(stored, EvalReportDoc) and stored.ending == "stored"
    assert stored.campaign_id == fixture.campaign.id == analysis.campaign_id
    skill = off_loop(load_skill).digest
    assert analysis.generation.prompt_id == skill
    assert analysis.generation.generator_model == ANALYST_MODEL
    assert analysis.generation.token_cost == pytest.approx(0.02), "two priced turns: the analyst's and the editor's"
    attempt = _attempt(fixture, status)
    assert (attempt.outcome, attempt.analysis_id, attempt.prompt_id) == ("stored", analysis.id, skill)
    assert (attempt.calls, attempt.unpriced_calls) == (2, 0)
    assert attempt.bundle_fingerprint == analysis.generation.bundle_fingerprint
    (row,) = _ledger(fixture)
    assert (row.purpose, row.campaign_id, row.cost_usd) == ("analysis", fixture.campaign.id, pytest.approx(0.02))
    assert all(analyst.released for analyst in fixture.analysts), "the analyst client is released"


async def test_a_report_with_an_unpriced_turn_is_ledgered_once_unpriced_and_its_cost_is_unknown_not_zero() -> None:
    fixture = off_loop(ops_fixture)
    _scripted(
        fixture,
        {
            "analyst": [scripted_reply(submission(), cost=None)],
            "editor-1": [scripted_reply(passing_review())],
        },
    )
    status = await _generated(fixture)
    analysis = _analysis(fixture, status.analysis_id)
    assert analysis.generation.token_cost is None
    attempt = _attempt(fixture, status)
    assert (attempt.calls, attempt.unpriced_calls) == (2, 1)
    (row,) = _ledger(fixture)
    assert row.cost_usd is None, "one row for the report, unpriced: never $0, never the priced share alone"
    assert (row.calls, row.unpriced_calls, row.priced_usd) == (2, 1, pytest.approx(0.01))
    text = out_of_run_spend_text(off_loop(scope_out_of_run_spend, fixture.host.eval_host, TOYHOST_SCOPE))
    assert "purpose analysis: 2 call(s): $0.0100 reported, 1 unpriced (so at least)" in text, text
    assert "2 calls: $0.0100 reported, 1 unpriced (so at least)" in text, text
    assert attempt.token_cost == pytest.approx(0.01), "the attempt keeps the priced floor beside its unpriced count"


async def test_a_durable_claim_mints_an_insight_and_others_do_not() -> None:
    fixture = off_loop(ops_fixture)
    status = await _generated(fixture)
    analysis = _analysis(fixture, status.analysis_id)
    insights = off_loop(fixture.host.eval_host.storage.query_insights, TOYHOST_SCOPE, subject_id=analysis.subject_id)
    (insight,) = [insight for insight in insights if insight.source_analysis_id == analysis.id]
    assert insight.statement == TOY_DOC["claims"][0]["text"]
    assert insight.confidence == "low", "a low-confidence document mints its durable claims low"


async def test_the_reports_ledger_row_states_its_turns_its_shares_and_the_budget_that_admitted_them() -> None:
    fixture = off_loop(ops_fixture)
    await _generated(fixture)
    (row,) = _ledger(fixture)
    assert (row.calls, row.unpriced_calls, row.priced_usd, row.cost_usd) == (
        2,
        0,
        pytest.approx(0.02),
        pytest.approx(0.02),
    )
    assert row.admitted_by == "the report budget of $10.00"
    report = off_loop(scope_out_of_run_spend, fixture.host.eval_host, TOYHOST_SCOPE)
    assert (report.totals.n_calls, report.totals.n_unpriced, report.totals.n_unbounded, report.totals.n_budgeted) == (
        2,
        0,
        0,
        2,
    )
    text = out_of_run_spend_text(report)
    assert "purpose analysis: 2 call(s): $0.0200 reported; 2 admitted by a report budget" in text, text
    assert "admitted at up to" not in text, "a budget-admitted call has no ceiling in the ledger, never read as $0"
    assert "unbounded" not in text and "(admitted by the report budget of $10.00)" in text, text


def _priced_turn(cost: float | None, n: int) -> TurnCost:
    return TurnCost(
        session="analyst",
        role="analyst",
        round=1,
        turn=n,
        model=ANALYST_MODEL,
        served_model=None,
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=None,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=cost,
        price_source=None if cost is None else "catalogue",
        pricing_failure=None if cost is not None else "no rate",
        stop_reason="end_turn",
        wall_s=0.1,
    )


def _tally_of(costs: Sequence[float | None], *, unreturned: int = 0) -> Tally:
    tally = Tally()
    for n, cost in enumerate(costs):
        tally.record_sent()
        tally.record_turn(_priced_turn(cost, n))
    for _ in range(unreturned):
        tally.record_sent()
    return tally


def _row_like_the_jobs(tally: Tally) -> OutOfRunSpend:
    """The row the job writes from a tally: its total, its shares and its raised turns, as the tally gives them."""
    return OutOfRunSpend(
        scope_id=TOYHOST_SCOPE,
        purpose="analysis",
        model=ANALYST_MODEL,
        outcome="completed",
        cost_usd=tally.spent_usd(),
        calls=tally.sent,
        unpriced_calls=tally.unpriced_turns(),
        raised_calls=tally.unreturned_turns(),
        priced_usd=tally.priced_usd(),
        admitted_by="the report budget of $10.00",
    )


async def test_a_report_whose_turn_costs_sum_apart_two_ways_is_still_ledgered_and_linked() -> None:
    """0.1 + 0.2 + 0.3 sums to 0.6000000000000001 one way and 0.6 the other: the row holds one figure for both."""
    fixture = off_loop(ops_fixture)
    _scripted(
        fixture,
        {
            "analyst": [
                scripted_reply(TurnToolUse("toolu_1", "nope", {}), cost=0.1),
                scripted_reply(submission(), cost=0.2),
            ],
            "editor-1": [scripted_reply(passing_review(), cost=0.3)],
        },
    )
    status = await _generated(fixture)
    assert status.status == "stored", status
    (row,) = _ledger(fixture)
    assert row.cost_usd == row.priced_usd == pytest.approx(0.6)
    assert (row.calls, row.unpriced_calls, row.raised_calls) == (3, 0, 0)


def test_every_fully_priced_tally_gives_one_figure_for_its_total_and_its_share() -> None:
    chance = random.Random(20261028)
    for _ in range(1000):
        costs = [round(chance.uniform(0.001, 0.2), chance.choice((3, 6, 9))) for _ in range(chance.randint(2, 40))]
        tally = _tally_of(costs)
        assert tally.spent_usd() == tally.priced_usd() == pytest.approx(math.fsum(costs))
        row = _row_like_the_jobs(tally)
        assert row.cost_usd == row.priced_usd


def test_a_rows_raised_turns_are_counted_raised_and_unpriced() -> None:
    row = _row_like_the_jobs(_tally_of([0.1, None], unreturned=2))
    assert (row.calls, row.unpriced_calls, row.raised_calls, row.cost_usd) == (4, 3, 2, None)
    assert row.priced_usd == pytest.approx(0.1)
    totals = OutOfRunSpendTotals.of([row])
    assert (totals.n_calls, totals.n_raised, totals.n_unpriced) == (4, 2, 3)


def test_a_host_row_whose_total_and_share_differ_in_the_last_place_reads_and_one_a_cent_apart_is_refused() -> None:
    fields: dict[str, Any] = {
        "scope_id": "s",
        "purpose": "analysis",
        "model": "m",
        "outcome": "completed",
        "calls": 3,
        "unpriced_calls": 0,
        "raised_calls": 0,
    }
    OutOfRunSpend(**fields, cost_usd=0.1 + 0.2 + 0.3, priced_usd=math.fsum([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="costs its priced share"):
        OutOfRunSpend(**fields, cost_usd=0.61, priced_usd=0.6)


async def test_a_failure_after_the_document_is_stored_still_writes_the_ending(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)

    def breaks(*_: Any, **__: Any) -> Any:
        raise RuntimeError("a defect minting the analysis")

    monkeypatch.setattr(analysis_ops, "doc_analysis", breaks)
    status = await _generated(fixture)
    attempt = _attempt(fixture, status)
    assert (status.state, attempt.outcome) == ("failed", "failed")
    assert attempt.error is not None and "is stored, and its analysis could not be built or ledgered" in attempt.error
    assert "RuntimeError: a defect minting the analysis" in attempt.error
    assert attempt.calls == 2
    (row,) = _ledger(fixture)
    assert row.calls == 2, "the report's spend is ledgered whatever became of its analysis"


async def test_a_ledger_write_that_breaks_after_the_document_is_stored_still_writes_the_ending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = off_loop(ops_fixture)

    def breaks(row: Any) -> None:
        raise RuntimeError("a defect in the ledger write")

    monkeypatch.setattr(fixture.host.eval_host.storage, "save_out_of_run_spend", breaks)
    status = await _generated(fixture)
    attempt = _attempt(fixture, status)
    assert attempt.outcome == "failed"
    assert attempt.error is not None and "RuntimeError: a defect in the ledger write" in attempt.error


# -- the minting, over the toy snapshot -------------------------------------------------------------------------------


async def _toy_snapshot(fixture: OpsFixture) -> tuple[CampaignBundle, Any, str, str]:
    frozen = await report_snapshot_freeze(
        fixture.host, TOYHOST_SCOPE, campaign_id=fixture.campaign.id, created_by=CALLER.identity
    )
    snapshot = off_loop(fixture.host.eval_host.storage.load_report_snapshot, frozen.snapshot_id, TOYHOST_SCOPE)
    assert snapshot is not None and snapshot.bundle is not None and snapshot.bundle_assembled_at is not None
    report_input = await snapshot_input(fixture.host, TOYHOST_SCOPE, frozen.snapshot_id)
    return (
        CampaignBundle.from_dict(snapshot.bundle),
        report_input,
        frozen.snapshot_id,
        snapshot.bundle_assembled_at,
    )


def _minted(
    bundle: CampaignBundle,
    report_input: Any,
    snapshot_id: str,
    assembled_at: str,
    doc: dict[str, Any],
    bounces: Mapping[str, int] | None = None,
) -> Any:
    stored = EvalReportDoc(
        scope_id=TOYHOST_SCOPE,
        snapshot_id=snapshot_id,
        campaign_id=bundle.campaign_id,
        doc=doc,
        ending="stored",
        findings=[],
        generation={"format": 1},
        created_by=CALLER.identity,
    )
    return doc_analysis(
        bundle,
        report_input,
        stored,
        input_digest="0" * 64,
        generator_model=ANALYST_MODEL,
        skill_digest="skill",
        bundle_assembled_at=assembled_at,
        token_cost=None,
        bounces=bounces or {},
    )


def _caveat(caveat_id: str, *, checked: bool) -> dict[str, Any]:
    if checked:
        return {"id": caveat_id, "label": None, "material_if": None, "checked": "immaterial"}
    return {"id": caveat_id, "label": None, "material_if": "only if it moves the call", "checked": None}


@pytest.mark.parametrize(
    ("form", "claim_caveat", "expected"),
    [
        ("high", None, "high"),
        ("high_if_immaterial", None, "medium"),
        ("low", None, "low"),
        ("high", "open", "medium"),
        ("high", "checked", "high"),
        ("high_if_immaterial", "open", "medium"),
        ("low", "open", "low"),
    ],
    ids=[
        "high",
        "high-if-immaterial",
        "low",
        "high-claim-open-caveat",
        "high-claim-checked-caveat",
        "conditional-claim-open-caveat",
        "low-claim-open-caveat",
    ],
)
async def test_a_durable_claim_is_minted_at_its_documents_form_held_to_medium_by_an_open_caveat_of_its_own(
    form: str, claim_caveat: str | None, expected: str
) -> None:
    fixture = off_loop(ops_fixture)
    bundle, report_input, snapshot_id, assembled_at = await _toy_snapshot(fixture)
    doc = copy.deepcopy(TOY_DOC)
    doc["confidence"]["form"] = form
    if claim_caveat is not None:
        doc["confidence"]["caveats"][0] = _caveat("judge_uncalibrated", checked=claim_caveat == "checked")
        doc["claims"][0]["caveats"] = ["judge_uncalibrated"]
    (insight,) = _minted(bundle, report_input, snapshot_id, assembled_at, doc).insights
    assert insight.confidence == expected


async def test_bounces_are_counted_as_repairs_and_summarised_by_kind() -> None:
    fixture = off_loop(ops_fixture)
    bundle, report_input, snapshot_id, assembled_at = await _toy_snapshot(fixture)
    minted = _minted(bundle, report_input, snapshot_id, assembled_at, TOY_DOC, {"shape": 1, "invariant": 2})
    generation = minted.analysis.generation
    assert generation.repair_attempts == 3
    assert generation.repaired_refusal == "submissions sent back, by kind: invariant x2, shape x1"
    clean = _minted(bundle, report_input, snapshot_id, assembled_at, TOY_DOC).analysis.generation
    assert (clean.repair_attempts, clean.repaired_refusal) == (0, None)


@pytest.mark.parametrize(
    ("recommends", "expected"),
    [([], None), (["256tok"], "256tok"), (["256tok", "1024tok"], None)],
    ids=["none", "one", "two"],
)
async def test_the_recommended_variant_is_set_by_exactly_one_recommended_arm(
    recommends: list[str], expected: str | None
) -> None:
    fixture = off_loop(ops_fixture)
    bundle, report_input, snapshot_id, assembled_at = await _toy_snapshot(fixture)
    doc = copy.deepcopy(TOY_DOC) | {"recommends": recommends}
    analysis = _minted(bundle, report_input, snapshot_id, assembled_at, doc).analysis
    keys = {arm.id: arm.variant_key for arm in report_input.arms}
    assert analysis.recommended_variant_key == (keys[expected] if expected is not None else None)


async def test_insights_come_from_durable_claims_only_each_citing_the_runs_its_keys_name() -> None:
    fixture = off_loop(ops_fixture)
    bundle, report_input, snapshot_id, assembled_at = await _toy_snapshot(fixture)
    doc = copy.deepcopy(TOY_DOC)
    durable = copy.deepcopy(doc["claims"][0]) | {"id": "c1", "text": "{arm:256tok} holds its own.", "durable": True}
    passing = copy.deepcopy(doc["claims"][0]) | {
        "id": "c2",
        "text": "{arm:1024tok} is the other arm.",
        "durable": False,
    }
    doc["claims"] = [durable, passing]
    minted = _minted(bundle, report_input, snapshot_id, assembled_at, doc)
    (insight,) = minted.insights
    assert insight.source_analysis_id == minted.analysis.id
    variant = next(arm.variant_key for arm in report_input.arms if arm.id == "256tok")
    runs = {
        run
        for aspect in bundle.aspects
        for cell in aspect.cell_measures
        if cell.variant_key == variant
        for run in cell.run_ids
    }
    assert runs and set(insight.evidence_run_ids) == runs, "the runs of the one arm the claim's key names"


async def test_the_hosts_writer_limits_reach_the_writer_unchanged_and_bound_the_estimate() -> None:
    fixture = off_loop(ops_fixture)
    limits = WriterLimits(analyst=AnalystLimits(draft_calls=7, revision_calls=3, bounces=2), revisions=1)
    assert fixture.host.generation is not None
    generation = dataclasses.replace(fixture.host.generation, limits=limits)
    host = dataclasses.replace(fixture.host, generation=generation)
    handed: list[WriterLimits] = []

    async def writer(host: OpsHost, **given: Any) -> Any:
        handed.append(given["limits"])
        return await WRITER(host, **given)

    (job,) = (await analysis_generate(host, fixture.campaign.id, TOYHOST_SCOPE, writer=writer)).jobs
    assert (await settled(host, job.job_id)).status == "stored"
    assert len(handed) == 1 and handed[0] is limits
    assert (await analysis_estimate(host, fixture.campaign.id, TOYHOST_SCOPE)).max_calls == writer_turns(limits)


# -- endings that store nothing -----------------------------------------------------------------------------------------


async def test_a_refused_report_fails_the_job_with_its_ending_and_keeps_the_tally() -> None:
    fixture = off_loop(ops_fixture)
    unfenced = [scripted_reply(TurnText("no document here")) for _ in range(4)]
    _scripted(fixture, {"analyst": unfenced})
    status = await _generated(fixture)
    assert status.state == "failed", status
    attempt = _attempt(fixture, status)
    assert attempt.outcome == "refused"
    assert attempt.error is not None and attempt.error.startswith("the report writer ended refused:")
    assert (attempt.calls, attempt.token_cost) == (4, pytest.approx(0.04)), "the draft and its three bounces"
    assert attempt.refusals, "the faults the last submission was refused for"
    assert attempt.analysis_id is None
    (row,) = _ledger(fixture)
    assert row.cost_usd == pytest.approx(0.04), "the report's spend, ledgered once"


async def test_a_failed_report_fails_the_job_with_its_ending_and_reason() -> None:
    fixture = off_loop(ops_fixture)
    _scripted(fixture, {"analyst": [scripted_reply(TurnToolUse("toolu_1", "nope", {})), RuntimeError("provider down")]})
    status = await _generated(fixture)
    attempt = _attempt(fixture, status)
    assert (status.state, attempt.outcome) == ("failed", "failed")
    assert attempt.error is not None and attempt.error.startswith("the report writer ended failed: RuntimeError")
    assert "provider down" not in attempt.error, "the client's text is kept only as the host's describer allows"
    assert (attempt.calls, attempt.unpriced_calls) == (2, 1), "the turn that came back, and the one that raised"
    assert attempt.token_cost == pytest.approx(0.01), "the priced share, a floor beside the unpriced turn"
    (row,) = _ledger(fixture)
    assert (row.calls, row.unpriced_calls, row.priced_usd, row.cost_usd) == (2, 1, pytest.approx(0.01), None)
    assert row.raised_calls == 1, "the turn that raised is counted raised"
    assert row.prompt_tokens is None, "nothing reported the raised turn's tokens"


async def test_a_report_cut_off_by_its_time_limit_counts_the_turn_in_flight_as_sent_and_unpriced() -> None:
    fixture = off_loop(ops_fixture, gate=asyncio.Event())  # never set: the first turn is in flight when time runs out
    assert fixture.host.generation is not None
    host = dataclasses.replace(fixture.host, generation=dataclasses.replace(fixture.host.generation, budget_s=0.5))
    (job,) = (await generate(host, fixture.campaign.id)).jobs
    status = await settled(host, job.job_id)
    attempt = _attempt(fixture, status)
    assert attempt.outcome == "timed_out"
    assert (attempt.calls, attempt.unpriced_calls, attempt.token_cost) == (1, 1, 0.0)
    (row,) = _ledger(fixture)
    assert (row.calls, row.unpriced_calls, row.cost_usd) == (1, 1, None), "unpriced, never $0"


async def test_a_value_error_after_turns_were_sent_is_no_refusal_of_the_start() -> None:
    fixture = off_loop(ops_fixture)

    async def breaks(host: OpsHost, *, tally: Tally, **_: Any) -> Any:
        tally.record_sent()
        raise ValueError("a defect after the first turn")

    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, writer=breaks)).jobs
    status = await settled(fixture.host, job.job_id)
    attempt = _attempt(fixture, status)
    assert attempt.error == "the report writer failed after 1 turn(s) (ValueError)"
    assert attempt.calls == 1 and attempt.unpriced_calls == 1


async def test_a_value_error_before_any_turn_is_a_refusal_before_any_call() -> None:
    fixture = off_loop(ops_fixture)

    async def refuses(host: OpsHost, **_: Any) -> Any:
        raise ValueError("the model sends no turn")

    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE, writer=refuses)).jobs
    status = await settled(fixture.host, job.job_id)
    attempt = _attempt(fixture, status)
    assert attempt.error is not None and attempt.error.startswith("refused before any call: the report was not written")
    assert attempt.calls == 0 and _ledger(fixture) == []


async def test_a_report_the_store_refuses_fails_the_job_with_its_record_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = off_loop(ops_fixture)

    def refuse(report: EvalReportDoc, **_: Any) -> None:
        raise StorageError("the store is read-only")

    monkeypatch.setattr(fixture.host.eval_host.storage, "save_report_doc", refuse)
    status = await _generated(fixture)
    attempt = _attempt(fixture, status)
    assert (status.state, attempt.outcome) == ("failed", "failed")
    assert attempt.error is not None and "ended stored" in attempt.error and "was not stored" in attempt.error
    assert "StorageError: the store is read-only" in attempt.error, "it failed for the store's reason"
    assert "under report document" in attempt.error and "delete_report_doc" in attempt.error
    assert attempt.calls == 2
    assert (
        off_loop(analysis_service.list_analyses, fixture.host.eval_host.storage, fixture.campaign.id, TOYHOST_SCOPE)
        == []
    )


# -- refusals before anything is sent ---------------------------------------------------------------------------------


async def test_a_host_with_no_report_budget_is_refused_saying_where_to_set_it_with_nothing_sent() -> None:
    fixture = off_loop(ops_fixture)
    host = OpsHost(launch=fixture.host.launch, generation=AnalysisGeneration(budget_s=60.0))
    with pytest.raises(ValidationFailedError) as refused:
        await generate(host, fixture.campaign.id)
    assert "no budget for a report" in refused.value.message
    assert "AnalysisGeneration.report_budget" in (refused.value.remedy or "")
    assert fixture.analysts == [] and fixture.writers == [], "no client was built"
    with pytest.raises(ValidationFailedError, match="no budget for a report"):
        await analysis_estimate(host, fixture.campaign.id, TOYHOST_SCOPE)


async def test_an_analyst_client_that_is_not_a_turn_model_is_refused_before_anything_is_frozen() -> None:
    fixture = off_loop(ops_fixture)
    released: list[bool] = []

    class _PlainClient:
        model_name = "plain-model"

        async def __aenter__(self) -> _PlainClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            released.append(True)

    object.__setattr__(fixture.host.eval_host, "clients", lambda role, model, *, temperature=None: _PlainClient())
    status = await _generated(fixture)
    attempt = _attempt(fixture, status)
    assert (status.state, attempt.outcome, attempt.calls) == ("failed", "failed", 0)
    assert attempt.error is not None and attempt.error.startswith("refused before any call:")
    assert "turn_session" in attempt.error and "turn_pricing_refusal" in attempt.error
    assert released == [True], "the client is released"
    frozen = await report_snapshot_freeze(
        fixture.host, TOYHOST_SCOPE, campaign_id=fixture.campaign.id, created_by=CALLER.identity
    )
    assert not frozen.reused, "nothing was frozen before"
    estimate = await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    assert not estimate.would_start and "turn_session" in (estimate.refusal or "")


async def test_the_estimate_states_the_report_budget_and_the_most_turns_making_no_call() -> None:
    fixture = off_loop(ops_fixture)
    estimate = await analysis_estimate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)
    assert (estimate.cap_usd, estimate.max_calls, estimate.would_start) == (
        REPORT_BUDGET_USD,
        writer_turns(WriterLimits()),
        True,
    )
    assert estimate.first_call_ceiling_usd is None
    assert all(not analyst.sessions for analyst in fixture.analysts), "no session was opened"
    assert estimate.bound == "the report budget of $10.00, which admits every turn before it is sent"
    assert estimate.unchecked is not None and "checked when the job freezes it" in estimate.unchecked, (
        "would_start is the start's checks, and says what it did not check"
    )


async def test_a_token_and_turn_budget_reads_as_no_dollar_cap_and_writes_a_report_nothing_prices() -> None:
    fixture = off_loop(ops_fixture, analyst_unpriced="no rate for toy-analyst-model")
    assert fixture.host.generation is not None
    budget = dataclasses.replace(
        fixture.host.generation, report_budget=lambda: ReportBudget(tokens=2_000_000, turns=60)
    )
    host = dataclasses.replace(fixture.host, generation=budget)

    estimate = await analysis_estimate(host, fixture.campaign.id, TOYHOST_SCOPE)
    assert (estimate.cap_usd, estimate.would_start) == (None, True), "no dollar cap, never $0, and no price asked"
    assert estimate.bound == (
        "the 2,000,000-token, 60-turn report budget, with no dollar cap, which admits every turn before it is sent"
    )

    (job,) = (await generate(host, fixture.campaign.id)).jobs
    assert (await settled(host, job.job_id)).status == "stored"
    (row,) = _ledger(fixture)
    assert row.admitted_by == "the 2,000,000-token, 60-turn report budget, with no dollar cap"


# -- the writer's ceiling ---------------------------------------------------------------------------------------------


def _one_second(max_tokens: int) -> float:
    return 1.0


async def test_writer_ceiling_s_bounds_a_worst_case_that_uses_every_round_revision_bounce_and_review() -> None:
    limits = WriterLimits(
        analyst=AnalystLimits(draft_calls=2, revision_calls=1, bounces=1),
        editor=EditorLimits(calls=1),
        revisions=2,
    )

    def tool(n: int) -> TurnReply:
        return scripted_reply(TurnToolUse(f"toolu_{n:03d}", "nope", {}))

    unread = scripted_reply(TurnText("no review here"))
    failing = scripted_reply(passing_review(scores=(0, 2, 2, 2, 2, 2, 2)))
    bad, good = scripted_reply(TurnText("no document here")), scripted_reply(submission())
    script: dict[str, list[TurnReply | Exception]] = {
        # Each round: its calls in tool turns, a submission bounced, then one that passes.
        "analyst": [tool(1), tool(2), bad, good, tool(3), bad, good, tool(4), bad, good],
        # Each review: its call, an answer that cannot be read, then a review that does not clear the report.
        **{f"editor-{round}": [tool(10 + round), unread, failing] for round in (1, 2, 3)},
    }
    fixture = off_loop(ops_fixture)
    tally = Tally()
    written = await write_report(
        fixture.host,
        CALLER,
        model=FakeAnalystClient(script),
        budget=ReportBudget(usd=REPORT_BUDGET_USD),
        campaign_id=fixture.campaign.id,
        limits=limits,
        tally=tally,
    )
    assert written.ending == "stored_with_findings", written.outcome.reason
    sent = len(tally.turns)
    ceiling = Ceilings().writer_ceiling_s(limits, request_s=_one_second)
    assert sent <= ceiling, "the ceiling bounds every turn the loop sent"
    assert sent > ceiling - 1, "the worst case reaches the count: one fewer would not bound it"
    assert writer_ceiling_s(limits, request_s=_one_second) == writer_turns(limits) == sent


def test_writer_ceiling_s_prices_each_session_at_its_own_output_cap() -> None:
    limits = WriterLimits()
    asked: set[int] = set()

    def request_s(max_tokens: int) -> float:
        asked.add(max_tokens)
        return 2.0

    assert writer_ceiling_s(limits, request_s=request_s) == 2.0 * writer_turns(limits)
    assert asked == {limits.analyst.session.max_tokens, limits.editor.session.max_tokens}


# -- no planner ---------------------------------------------------------------------------------------------------------


async def test_no_planner_call_is_reachable_from_this_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def planner(*_: Any, **__: Any) -> Any:
        raise AssertionError("analysis_generate reached the report planner")

    monkeypatch.setattr(planned_analysis, "generate_report_plan", planner)
    monkeypatch.setattr(analysis_service, "generate_planned_analysis", planner)
    monkeypatch.setattr(analysis_service, "prepare_analysis_generation", planner)
    fixture = off_loop(ops_fixture)
    _scripted(fixture)
    status = await _generated(fixture)
    assert status.status == "stored"
    assert fixture.writers == [], "no client of the planner's role was built"


# -- readers --------------------------------------------------------------------------------------------------------------


async def test_a_doc_analysis_is_listed_read_as_the_campaign_page_and_its_insight_is_a_later_prior() -> None:
    fixture = off_loop(ops_fixture)
    status = await _generated(fixture)
    eval_host = fixture.host.eval_host
    (line,) = off_loop(analyses_list, eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    assert (line.basis, line.report_doc_id, line.findings, line.recommendations) == (
        "doc",
        _analysis(fixture, status.analysis_id).report_doc_id,
        1,
        0,
    )
    assert line.needs_regeneration is None
    page = off_loop(report_read, eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="json")
    assert page.analysis_id == status.analysis_id and page.analysis_basis == "doc"
    markdown = off_loop(report_read, eval_host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown").body
    assert f"is stored as report document {line.report_doc_id}" in markdown, (
        "the interim page says a report was written"
    )
    assert "no analysis has been generated" not in markdown
    frozen = await report_snapshot_freeze(
        fixture.host, TOYHOST_SCOPE, campaign_id=fixture.campaign.id, created_by=CALLER.identity
    )
    overview = await campaign_overview(fixture.host, TOYHOST_SCOPE, snapshot_id=frozen.snapshot_id)
    assert TOY_DOC["claims"][0]["text"] in [prior.statement for prior in overview.prior_insights]
