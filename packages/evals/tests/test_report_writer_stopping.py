"""The pairwise pass under a declared stopping rule: where it stops, what the stop's standing is, and what it says.

Each campaign is the promotion read's scripted one (a control and a candidate over the toy host's twelve documents,
three repeats each), its pages stored and the scoring judge scripted, so every figure is checked against the stats
functions on shares the test computes itself, read as prefixes of the one bet the rule fixes. No model is called.

Held: a rule to stop on separation judges the planned cases in the declared bet's order, stops at the first case after
which the preference separates, asks nothing past it, and records each look; the promotion read on that pass reads the
gated readings and guardrails once, on every case the arms ran (each arm ran all of them before the pass). A rule
declared after the first result, or with no time, keeps its verdict and says so. A rule to read every case is today's
reading exactly, a case with no share included. Cases found that the rule did not plan, or planned and not found, are
said, never refused.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_DOCUMENTS, TOYHOST_SCOPE
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import ops_fixture
from packages.evals.tests.writer_pairwise_support import (
    Policy,
    ScriptedPairJudge,
    prefers_the_candidate,
    prefers_the_control,
    store_pages,
)
from packages.evals.tests.writer_promotion_support import BETTER, CANDIDATE, SCORING_JUDGE, promotion_campaign
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    bet_order,
    bounded_difference_interval,
    bounded_separation_p,
)
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.ops import JobStatus, OpsHost, await_job
from threetears.evals.ops.report_writer.writer_pairwise import (
    StoppingRule,
    StopWhen,
    WriterPairwiseReport,
    writer_pairwise,
)
from threetears.evals.run.judge import CANNOT_TELL
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion, writer_promotion
from threetears.evals.schema.models import utc_now_iso

#: Every repeat a document ran, on both arms.
_REPEATS = 3
#: The share range every reading here is on.
_RANGE = (0.0, 1.0)
#: The cases the toy campaign holds, in sorted order: the pass's case order.
_CASES = sorted(TOYHOST_DOCUMENTS)
#: A declaration long before any result was stored.
_EARLY = "2000-01-01T00:00:00+00:00"

#: A guardrail twelve agreeing cases show held and a stop's few cannot: a page's share of private rows leaked, on a
#: declared range and a margin (the promotion tests' guardrail).
_LEAKS = MetricDescriptor(
    name="leaked_rows",
    reader_name="Private rows leaked",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="The share of the snapshot's private rows the page leaked.",
    higher_is_better=False,
    materiality_threshold=0.35,
    guardrail=True,
    value_range=(0.0, 1.0),
)


def _campaign(policy: Policy, **campaign: Any) -> tuple[EvalHost, str, ScriptedPairJudge]:
    """The promotion read's campaign with every page stored, and the judge answering by ``policy``."""
    host, campaign_id = off_loop(promotion_campaign, BETTER, **campaign)
    off_loop(store_pages, host, campaign_id)
    judge = ScriptedPairJudge(policy=policy)
    return dataclasses.replace(host, clients=lambda _role, _model, *, temperature=None: judge), campaign_id, judge


def _rule(stop_when: StopWhen, declared_at: str | None = _EARLY, cases: list[str] = _CASES) -> StoppingRule:
    """A rule over ``cases``, sorted, declared at ``declared_at``."""
    planned = sorted(cases)
    return StoppingRule(stop_when=stop_when, case_ids=planned, max_cases=len(planned), declared_at=declared_at)


def _ops(host: EvalHost) -> OpsHost:
    launch = off_loop(ops_fixture, generation=False).host.launch
    return OpsHost(
        launch=replace(launch, eval_host=host),
        model_family=lambda model: "judges" if model == SCORING_JUDGE else "writers",
    )


async def _pass(host: EvalHost, campaign_id: str, rule: StoppingRule) -> tuple[WriterPairwiseReport, JobStatus]:
    """Run the pass under ``rule`` as its job, through its front door; its report and its completed job."""
    ops = _ops(host)
    started = await writer_pairwise(ops, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE, stopping_rule=rule)
    (status,) = await await_job(ops, started, TOYHOST_SCOPE)
    assert status.state == "completed" and status.result is not None, status.detail
    return WriterPairwiseReport.model_validate(status.result), status


def _read(host: EvalHost, campaign_id: str, status: JobStatus) -> WriterPromotion:
    return off_loop(writer_promotion, host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE, pairwise_job=status)


def _bet(planned: int) -> list[str]:
    """The cases in the order the bet fixed for ``planned`` reads them."""
    return [_CASES[index] for index in bet_order(planned) if index < len(_CASES)]


def _first_separation(shares: list[float], planned: int) -> int:
    """The first count of cases, from the five a rate needs, at which the prefix of the bet separates."""
    for k in range(5, len(shares) + 1):
        control = [1.0 - share for share in shares[:k]]
        p = bounded_separation_p(control, shares[:k], paired=True, value_range=_RANGE, planned_n=planned)
        interval = bounded_difference_interval(
            control, shares[:k], paired=True, value_range=_RANGE, confidence=INTERVAL_LEVEL, planned_n=planned
        )
        assert p is not None and interval is not None
        if p < 0.05 and (interval[0] > 0 or interval[1] < 0):
            return k
    raise AssertionError("never separated")


async def test_a_separating_run_stops_at_the_first_look_that_separates_and_records_each_look() -> None:
    host, campaign_id, judge = _campaign(prefers_the_candidate)
    rule = _rule("separated")
    report, status = await _pass(host, campaign_id, rule)

    k = _first_separation([1.0] * len(_CASES), len(_CASES))
    stopping = report.stopping
    assert stopping is not None and stopping.rule == rule and not stopping.late
    assert stopping.stopped_at == k < len(_CASES) and stopping.first_separated_at == k
    assert stopping.cases_found == len(_CASES) and stopping.absent == [] and stopping.unplanned == []
    # The judge read the cases in the bet's order, every repeat of a case in both orders, and nothing past the stop.
    judged = _bet(len(_CASES))[:k]
    assert [document for document, *_ in judge.calls[:: 2 * _REPEATS]] == judged
    assert report.calls_made == len(judge.calls) == 2 * _REPEATS * k
    assert {pair.case_id for pair in report.pairs if pair.reached} == set(judged)
    assert all(not pair.answers for pair in report.pairs if not pair.reached)
    assert len(report.pairs) == len(_CASES) * _REPEATS
    # Each look is the prefix of the one bet; the verdict is the last of them.
    assert [look.case_id for look in stopping.looks] == judged
    for look in stopping.looks:
        shares = [1.0] * look.cases
        control = [0.0] * look.cases
        if look.cases >= 2:
            assert look.p == bounded_separation_p(
                control, shares, paired=True, value_range=_RANGE, planned_n=len(_CASES)
            )
            assert look.interval == bounded_difference_interval(
                control, shares, paired=True, value_range=_RANGE, planned_n=len(_CASES)
            )
        assert look.separated == (look.cases == k)
    verdict = report.whole.verdict
    assert (verdict.outcome, verdict.reason) == ("improved", "separated") and report.whole.cases == k
    assert report.whole.p == stopping.looks[-1].p and verdict.interval == stopping.looks[-1].interval
    assert stopping.line == (
        f"Stopped at {k} of {len(_CASES)} cases under the rule declared before the first result: the judge preferred "
        f"{CANDIDATE} on {k} of {k}, separated."
    )

    # The promotion read on the stopped pass decides on it, and says what it did.
    read = _read(host, campaign_id, status)
    assert read.decision == "promote" and read.stopping_line == stopping.line


async def test_the_stop_applies_toward_keep_too() -> None:
    host, campaign_id, judge = _campaign(prefers_the_control)
    report, status = await _pass(host, campaign_id, _rule("separated", None))

    k = _first_separation([0.0] * len(_CASES), len(_CASES))
    assert report.stopping is not None and report.stopping.stopped_at == k
    assert report.whole.verdict.outcome == "regressed" and len(judge.calls) == 2 * _REPEATS * k
    assert f"the judge preferred control on {k} of {k}, separated" in report.stopping.line
    assert _read(host, campaign_id, status).decision == "keep"


@pytest.mark.parametrize(
    ("declared_at", "said"),
    [
        (None, "a rule declared with no time, its verdict standing as read"),
        ("now", "a rule declared after the first result, its verdict standing as read"),
    ],
)
async def test_a_rule_declared_late_keeps_its_verdict_and_says_so(declared_at: str | None, said: str) -> None:
    host, campaign_id, _ = _campaign(prefers_the_candidate)
    stamp = utc_now_iso() if declared_at == "now" else None
    report, status = await _pass(host, campaign_id, _rule("separated", stamp))

    stopping = report.stopping
    assert stopping is not None and stopping.late
    assert stopping.first_result_at is not None
    k = _first_separation([1.0] * len(_CASES), len(_CASES))
    # Not refused, not demoted: the same stop, verdict and decision as a rule declared in time.
    assert stopping.stopped_at == k and report.whole.verdict.outcome == "improved"
    assert said in stopping.line
    read = _read(host, campaign_id, status)
    assert read.decision == "promote" and read.stopping_line is not None and said in read.stopping_line


async def test_a_stop_reads_the_primary_alone_and_the_guardrails_are_read_on_every_case_run() -> None:
    """A guardrail its few cases at the stop could not show held is held on the twelve every arm ran: the read, on
    every case run, promotes, as the campaign's page over the same cases reads it."""
    nothing = {document: {_LEAKS.name: 0.0} for document in TOYHOST_DOCUMENTS}
    host, campaign_id, judge = _campaign(prefers_the_candidate, writer=(nothing, nothing), measures=[_LEAKS])
    report, status = await _pass(host, campaign_id, _rule("separated"))

    stopping = report.stopping
    k = _first_separation([1.0] * len(_CASES), len(_CASES))
    assert stopping is not None and stopping.stopped_at == k < len(_CASES)
    assert len(judge.calls) == 2 * _REPEATS * k
    read = _read(host, campaign_id, status)
    assert read.decision == "promote"
    assert {verdict.name: verdict.outcome for verdict in read.verdicts if verdict.kind == "guardrail"} == {
        _LEAKS.name: "held"
    }
    assert read.stopping_line == stopping.line


_LEAN = (5, 5, 4, 4, 4, 3, 5, 4, 3, 2, 5, 4)


def _graded(document: str, slot: str) -> int:
    """A judge whose preference moves with the document: a spread of shares, leaning to the candidate."""
    lean = _LEAN[TOYHOST_DOCUMENTS.index(document)]
    return lean if slot == "B" else 6 - lean


#: A document every order of which the judge cannot tell on, so it has no share.
_UNTOLD = _CASES[4]


def _graded_but_one(document: str, slot: str) -> int | str:
    return CANNOT_TELL if document == _UNTOLD else _graded(document, slot)


@pytest.mark.parametrize("policy", [_graded, _graded_but_one], ids=["every share", "a case with no share"])
async def test_a_rule_to_read_every_case_is_todays_reading_exactly(policy: Policy) -> None:
    host, campaign_id, judge = _campaign(policy)
    report, _ = await _pass(host, campaign_id, _rule("never"))

    shares = [(_LEAN[TOYHOST_DOCUMENTS.index(case)] - 1) / 4 for case in _CASES if policy(case, "A") != CANNOT_TELL]
    control = [1.0 - share for share in shares]
    # The fixed-n reading over the shares found, in case order, with no bet fixed: the code before stopping rules.
    assert report.whole.cases == len(shares)
    assert report.whole.p == bounded_separation_p(control, shares, paired=True, value_range=_RANGE)
    assert report.whole.verdict.interval == bounded_difference_interval(
        control, shares, paired=True, value_range=_RANGE, confidence=INTERVAL_LEVEL
    )
    # And the same as a pass given no rule, which the engine action starts.
    ops = _ops(host)
    (bare,) = await await_job(
        ops, await writer_pairwise(ops, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE), TOYHOST_SCOPE
    )
    assert bare.result is not None
    unruled = WriterPairwiseReport.model_validate(bare.result)
    assert (unruled.whole.p, unruled.whole.verdict.interval) == (report.whole.p, report.whole.verdict.interval)
    # Judged in case order, as before; the looks say where it would have stopped.
    assert [document for document, *_ in judge.calls[: 2 * _REPEATS * len(_CASES) : 2 * _REPEATS]] == [
        case for case in _CASES
    ]
    stopping = report.stopping
    assert stopping is not None and stopping.stopped_at is None
    assert len(stopping.looks) == len(shares) and stopping.looks[-1].p == report.whole.p
    first = stopping.first_separated_at
    said = "was consistent with no preference" if first is None else f"first separated at case {first}"
    assert stopping.line.startswith(
        f"Read {len(_CASES)} of {len(_CASES)} cases under the rule declared before the first result: "
    )
    assert said in stopping.line


async def test_a_rule_that_recorded_its_order_is_judged_in_that_order_whatever_derives_one_now() -> None:
    """A rule records the order its bet reads the planned cases in, so a later read replays it: here the reverse of
    what ``bet_order`` derives today, and the judge reads the cases in the recorded order, not the derived one."""
    host, campaign_id, judge = _campaign(prefers_the_candidate)
    derived = _rule("separated")
    recorded = derived.model_copy(update={"order": list(reversed(derived.bet))})
    assert recorded.bet != derived.bet

    report, _ = await _pass(host, campaign_id, recorded)

    stopping = report.stopping
    assert stopping is not None and stopping.rule == recorded
    k = stopping.stopped_at
    assert k is not None
    assert [document for document, *_ in judge.calls[:: 2 * _REPEATS]] == recorded.bet[:k]
    assert [look.case_id for look in stopping.looks] == recorded.bet[:k]
    # A rule recorded before orders were derives the order as it always did.
    assert derived.order is None and derived.bet == _bet(len(_CASES))
    with pytest.raises(ValueError, match="order lists its planned cases"):
        StoppingRule.model_validate({**derived.model_dump(), "order": derived.bet[1:]})


async def test_the_bet_is_the_declared_one_and_cases_off_the_plan_are_said() -> None:
    """A rule planned over a case the pass does not find, and missing two it does: the bet is fixed by the planned ids,
    the two unplanned are not read, and both differences are said, never refused."""
    host, campaign_id, judge = _campaign(prefers_the_control)
    planned = [*_CASES[:10], "zz-not-run"]
    report, _ = await _pass(host, campaign_id, _rule("separated", cases=planned))

    stopping = report.stopping
    assert stopping is not None and stopping.absent == ["zz-not-run"] and stopping.unplanned == _CASES[10:]
    ids = sorted(planned)
    bet = [ids[index] for index in bet_order(len(ids)) if ids[index] != "zz-not-run"]
    k = stopping.stopped_at
    assert k is not None
    assert [document for document, *_ in judge.calls[:: 2 * _REPEATS]] == bet[:k]
    assert not {document for document, *_ in judge.calls} & set(_CASES[10:])
    assert stopping.line.endswith(
        "1 case the rule planned was not found; 2 found were not planned, and the bet holds no place for them, so "
        "they were not read."
    )


async def test_with_no_rule_every_case_is_read_and_the_record_says_no_rule_was_declared() -> None:
    host, campaign_id, judge = _campaign(prefers_the_candidate)
    ops = _ops(host)
    (status,) = await await_job(
        ops, await writer_pairwise(ops, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE), TOYHOST_SCOPE
    )
    assert status.result is not None
    report = WriterPairwiseReport.model_validate(status.result)
    stopping = report.stopping
    assert stopping is not None and stopping.rule is None and not stopping.late and stopping.stopped_at is None
    assert len(judge.calls) == 2 * _REPEATS * len(_CASES)
    assert len(stopping.looks) == len(_CASES)
    assert stopping.line.startswith("Read all 12 cases with no stopping rule declared: ")


def test_a_rule_lists_its_planned_cases_sorted_and_counts_them() -> None:
    with pytest.raises(ValueError, match="sorted, each once"):
        StoppingRule(stop_when="never", case_ids=["b", "a"], max_cases=2)
    with pytest.raises(ValueError, match="max_cases is the 2 cases planned"):
        StoppingRule(stop_when="never", case_ids=["a", "b"], max_cases=3)


async def test_one_case_off_the_plan_is_said_as_one() -> None:
    host, campaign_id, _ = _campaign(prefers_the_control)
    report, _ = await _pass(host, campaign_id, _rule("separated", cases=_CASES[:-1]))

    stopping = report.stopping
    assert stopping is not None and stopping.unplanned == _CASES[-1:]
    assert stopping.line.endswith("1 found was not planned, and the bet holds no place for it, so it was not read.")


async def test_under_never_a_case_off_the_plan_is_read_and_counted_alone() -> None:
    """``never`` fixes no bet, so a case it did not plan is read, and the count prints alone: never more read than
    planned."""
    host, campaign_id, _ = _campaign(prefers_the_control)
    report, _ = await _pass(host, campaign_id, _rule("never", cases=_CASES[:-1]))

    stopping = report.stopping
    assert stopping is not None and len(stopping.looks) == len(_CASES)
    assert stopping.line.startswith(f"Read {len(_CASES)} cases under the rule declared before the first result: ")
    assert f" of {len(_CASES) - 1} " not in stopping.line
    assert stopping.line.endswith("1 found was not planned, read as every case is.")
