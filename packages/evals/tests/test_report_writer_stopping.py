"""The pairwise pass under a declared stopping rule: where it stops, what the stop's standing is, and what it says.

Each campaign is the promotion read's scripted one (a control and a candidate over the toy host's twelve documents,
three repeats each), its pages stored and the scoring judge scripted, so every figure is checked against the stats
functions on shares the test computes itself, read as prefixes of the one bet the rule fixes. No model is called.

Held: a rule to stop on separation judges the cases in the bet's order, stops at the first case after which the
preference separates and the decision read there stands, asks nothing past it, and records each look; the promotion
read on that pass reads the gated readings and guardrails once, on the cases judged. A guardrail not shown held at the
stop sends the pass on to the cap, saying it looked twice. A rule declared after the first result, or with no time,
keeps its verdict and says so. A rule to read every case equals the fixed-n reading at N, judged in the bet's order.
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
from threetears.evals.ops.report_writer.writer_pairwise import StoppingRule, WriterPairwiseReport, writer_pairwise
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
    rule = StoppingRule(stop_when="separated", max_cases=len(_CASES), declared_at=_EARLY)
    report, status = await _pass(host, campaign_id, rule)

    k = _first_separation([1.0] * len(_CASES), len(_CASES))
    stopping = report.stopping
    assert stopping is not None and stopping.rule == rule and not stopping.late
    assert stopping.stopped_at == k < len(_CASES) and stopping.first_separated_at == k
    assert stopping.second_look is None and stopping.cases_found == len(_CASES) and stopping.beyond_cap == []
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

    # The promotion read on the stopped pass reads the gated readings once, on the cases judged, and says so.
    read = _read(host, campaign_id, status)
    assert read.decision == "promote" and read.gated_on == k
    assert read.stopping_line == (
        f"{stopping.line} The gated readings and guardrails were read once, on the {k} cases judged."
    )


async def test_the_stop_applies_toward_keep_too() -> None:
    host, campaign_id, judge = _campaign(prefers_the_control)
    report, status = await _pass(host, campaign_id, StoppingRule(stop_when="separated", max_cases=len(_CASES)))

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
    report, status = await _pass(
        host, campaign_id, StoppingRule(stop_when="separated", max_cases=len(_CASES), declared_at=stamp)
    )

    stopping = report.stopping
    assert stopping is not None and stopping.late
    assert stopping.first_result_at is not None
    k = _first_separation([1.0] * len(_CASES), len(_CASES))
    # Not refused, not demoted: the same stop, verdict and decision as a rule declared in time.
    assert stopping.stopped_at == k and report.whole.verdict.outcome == "improved"
    assert said in stopping.line
    read = _read(host, campaign_id, status)
    assert read.decision == "promote" and read.stopping_line is not None and said in read.stopping_line


async def test_a_guardrail_not_held_at_the_stop_reads_on_to_the_cap_and_says_it_looked_twice() -> None:
    nothing = {document: {_LEAKS.name: 0.0} for document in TOYHOST_DOCUMENTS}
    host, campaign_id, judge = _campaign(prefers_the_candidate, writer=(nothing, nothing), measures=[_LEAKS])
    report, status = await _pass(host, campaign_id, StoppingRule(stop_when="separated", max_cases=len(_CASES)))

    stopping = report.stopping
    k = _first_separation([1.0] * len(_CASES), len(_CASES))
    assert stopping is not None and stopping.first_separated_at == k
    # The guardrail is not shown held on k cases, so the decision there did not stand: the judge read every case.
    assert stopping.stopped_at is None
    assert stopping.second_look is not None and "1 of 1 guardrails are not shown held" in stopping.second_look
    assert len(judge.calls) == 2 * _REPEATS * len(_CASES) and all(pair.reached for pair in report.pairs)
    assert f"separated at case {k}, but the decision there did not stand" in stopping.line
    assert "looked at a second time, on every case" in stopping.line
    # On every case the guardrail is held, and the read decides on all of them.
    read = _read(host, campaign_id, status)
    assert read.gated_on is None and read.decision == "promote"
    assert {verdict.name: verdict.outcome for verdict in read.verdicts if verdict.kind == "guardrail"} == {
        _LEAKS.name: "held"
    }
    assert read.stopping_line == stopping.line


def _graded(document: str, slot: str) -> int:
    """A judge whose preference moves with the document: a spread of shares, leaning to the candidate."""
    lean = [5, 5, 4, 4, 4, 3, 5, 4, 3, 2, 5, 4][TOYHOST_DOCUMENTS.index(document)]
    return lean if slot == "B" else 6 - lean


async def test_a_rule_to_read_every_case_equals_the_fixed_n_reading_and_says_where_it_would_have_stopped() -> None:
    host, campaign_id, judge = _campaign(_graded)
    report, _ = await _pass(
        host, campaign_id, StoppingRule(stop_when="never", max_cases=len(_CASES), declared_at=_EARLY)
    )

    shares = [(([5, 5, 4, 4, 4, 3, 5, 4, 3, 2, 5, 4][TOYHOST_DOCUMENTS.index(case)]) - 1) / 4 for case in _CASES]
    control = [1.0 - share for share in shares]
    # The fixed-n reading over the cases in case order, as with no rule: the same numbers.
    assert report.whole.p == bounded_separation_p(control, shares, paired=True, value_range=_RANGE)
    assert report.whole.verdict.interval == bounded_difference_interval(
        control, shares, paired=True, value_range=_RANGE, confidence=INTERVAL_LEVEL
    )
    stopping = report.stopping
    assert stopping is not None and stopping.stopped_at is None and stopping.second_look is None
    assert [document for document, *_ in judge.calls[:: 2 * _REPEATS]] == _bet(len(_CASES))
    assert len(stopping.looks) == len(_CASES) and stopping.looks[-1].p == report.whole.p
    first = stopping.first_separated_at
    assert first is None or stopping.looks[first - 1].separated
    said = "no look up to 12 excluded no preference" if first is None else f"first separated at case {first}"
    assert stopping.line.startswith("Read 12 of 12 cases under the rule declared before the first result: ")
    assert said in stopping.line


async def test_cases_past_the_rules_cap_are_named_and_never_read() -> None:
    host, campaign_id, judge = _campaign(prefers_the_candidate)
    report, _ = await _pass(host, campaign_id, StoppingRule(stop_when="never", max_cases=10, declared_at=_EARLY))

    stopping = report.stopping
    assert stopping is not None and stopping.beyond_cap == _CASES[10:]
    assert {document for document, *_ in judge.calls} == set(_CASES[:10])
    assert report.whole.cases == 10
    assert all(not pair.reached for pair in report.pairs if pair.case_id in _CASES[10:])
    assert stopping.line.startswith("Read 10 of 10 cases")


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
