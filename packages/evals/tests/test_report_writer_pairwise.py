"""The pairwise pass: a writer campaign's candidate and control pages shown to the scoring judge side by side.

Each campaign is the promotion read's scripted one (a control and a candidate over the toy host's twelve documents,
three repeats each, scored by one judge), with every result's page stored as its judge evidence and the scoring judge
scripted: it reads which report is whose off the stored pages it is shown and answers by a policy the test names. So
what the pass asks, what it reads and the verdict it comes to are each checked against what the scripted judge was
asked and against the stats functions on shares the test computes itself. No model is called.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_DOCUMENTS, TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.writer_pairwise_support import (
    GARBLED,
    PAIR_CEILING,
    REFUSED,
    Pages,
    Policy,
    ScriptedPairJudge,
    describe_account_refusal,
    every_page,
    pair_page,
    prefers_report_a,
    prefers_the_candidate,
    prefers_the_control,
    store_pages,
    ties,
)
from packages.evals.tests.writer_promotion_support import BETTER, CANDIDATE, SCORING_JUDGE, promotion_campaign
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis.stats import INTERVAL_LEVEL, bounded_difference_interval, bounded_separation_p
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.ops import JobsStarted, OpsHost, await_job
from threetears.evals.ops.report_writer.writer_contract import FINISHED_MEASURE
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport, writer_pairwise
from threetears.evals.run.judge import CANNOT_TELL, JUDGE_CALL_ATTEMPTS

#: Every repeat a document ran, on both arms.
_REPEATS = (1, 2, 3)
#: The share range every bounded reading here is on.
_RANGE = (0.0, 1.0)


def _campaign(
    judge: ScriptedPairJudge, *, pages: Pages = every_page, unfinished_control: int = 0
) -> tuple[EvalHost, str]:
    """The promotion read's campaign, each result's page stored as ``pages`` says, the judge on its clients.

    The control's last ``unfinished_control`` documents are stored unfinished; every other page finished.
    """
    finished = {
        "control": {
            document: {FINISHED_MEASURE: index < len(TOYHOST_DOCUMENTS) - unfinished_control}
            for index, document in enumerate(TOYHOST_DOCUMENTS)
        },
        "candidate": {document: {FINISHED_MEASURE: True} for document in TOYHOST_DOCUMENTS},
    }
    host, campaign_id = off_loop(promotion_campaign, BETTER, writer=(finished["control"], finished["candidate"]))
    off_loop(store_pages, host, campaign_id, pages)
    return dataclasses.replace(host, clients=lambda _role, _model, *, temperature=None: judge), campaign_id


def _apart(model: str) -> str | None:
    """A host's reading of a model's family: the scoring judge in one, every writer in another."""
    return "judges" if model == SCORING_JUDGE else "writers"


def _ops(host: EvalHost, *, family: Callable[[str], str | None] | None = None, cap: float | None = None) -> OpsHost:
    """The ops host over ``host``'s store, reading families by ``family`` and enforcing ``cap`` when one is given."""
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(
        update={
            "enforcement_enabled": cap is not None,
            "max_out_of_run_cost_usd": TOYHOST_LAUNCH_SETTINGS.max_out_of_run_cost_usd if cap is None else cap,
        }
    )
    launch = off_loop(ops_fixture, generation=False, settings=settings).host.launch
    return OpsHost(launch=replace(launch, eval_host=host), model_family=family)


async def _pass(
    host: EvalHost, campaign_id: str, *, family: Callable[[str], str | None] | None = None, cap: float | None = None
) -> WriterPairwiseReport:
    """Start the pass as a job through its front door, wait for it to end, and read its report off its record."""
    ops = _ops(host, family=family, cap=cap)
    started = await writer_pairwise(ops, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE)
    (status,) = await await_job(ops, started, TOYHOST_SCOPE)
    assert status.state == "completed" and status.result is not None, status.detail
    return WriterPairwiseReport.model_validate(status.result)


def _share(score: int, slot: str) -> Fraction:
    """The candidate's share of one answer, by the instrument's formula: (s-1)/4 as B, (5-s)/4 as A."""
    return Fraction(5 - score, 4) if slot == "A" else Fraction(score - 1, 4)


def _expected(control: list[float], candidate: list[float]) -> tuple[float | None, tuple[float, float] | None]:
    """The stats functions on the shares, as every rate contrast calls them."""
    p = bounded_separation_p(control, candidate, paired=True, value_range=_RANGE)
    interval = bounded_difference_interval(
        control, candidate, paired=True, value_range=_RANGE, confidence=INTERVAL_LEVEL
    )
    return p, interval


async def test_every_pair_is_asked_in_both_orders_from_its_stored_pages() -> None:
    judge = ScriptedPairJudge()
    host, campaign_id = _campaign(judge)
    report = await _pass(host, campaign_id)

    pairs = len(TOYHOST_DOCUMENTS) * len(_REPEATS)
    assert len(report.pairs) == pairs and not report.unpaired
    assert report.calls_made == len(judge.calls) == 2 * pairs
    asked = sorted((document, slot, first, second) for document, slot, first, second in judge.calls)
    for document in TOYHOST_DOCUMENTS:
        for repeat in _REPEATS:
            mine, theirs = pair_page("candidate", document, repeat), pair_page("control", document, repeat)
            assert (document, "A", mine, theirs) in asked
            assert (document, "B", theirs, mine) in asked
    for pair in report.pairs:
        assert pair.missing is None
        assert [answer.candidate for answer in pair.answers] == ["B", "A"]
        assert all(answer.answer == "scored" for answer in pair.answers)
    assert report.judge_model == SCORING_JUDGE
    assert report.candidate == CANDIDATE and report.control == "control"


async def test_a_side_with_no_page_loses_its_pair_and_both_missing_is_a_tie() -> None:
    gone = {("candidate", TOYHOST_DOCUMENTS[0], 1), ("control", TOYHOST_DOCUMENTS[1], 1)}
    gone |= {(arm, TOYHOST_DOCUMENTS[2], 1) for arm in ("candidate", "control")}
    judge = ScriptedPairJudge(policy=ties)
    host, campaign_id = _campaign(judge, pages=lambda arm, document, repeat: (arm, document, repeat) not in gone)
    report = await _pass(host, campaign_id)

    missing = {(pair.case_id, pair.repeat): pair.missing for pair in report.pairs if pair.missing is not None}
    assert missing == {
        (TOYHOST_DOCUMENTS[0], 1): "candidate",
        (TOYHOST_DOCUMENTS[1], 1): "control",
        (TOYHOST_DOCUMENTS[2], 1): "both",
    }
    assert all(not pair.answers for pair in report.pairs if pair.missing is not None)
    assert len(judge.calls) == 2 * (len(report.pairs) - 3)
    shares = {case.case_id: case.share for case in report.cases}
    # One forced pair beside two tied ones: the candidate's share is the mean of 0 (or 1, or 1/2) and two halves.
    assert shares[TOYHOST_DOCUMENTS[0]] == pytest.approx(float((0 + Fraction(1, 2) * 2) / 3))
    assert shares[TOYHOST_DOCUMENTS[1]] == pytest.approx(float((1 + Fraction(1, 2) * 2) / 3))
    assert shares[TOYHOST_DOCUMENTS[2]] == pytest.approx(0.5)
    assert all(shares[document] == 0.5 for document in TOYHOST_DOCUMENTS[3:])


async def test_the_cap_refuses_the_pass_before_its_first_call() -> None:
    judge = ScriptedPairJudge()
    host, campaign_id = _campaign(judge)
    pairs = len(TOYHOST_DOCUMENTS) * len(_REPEATS)
    planned = 2 * pairs * JUDGE_CALL_ATTEMPTS * PAIR_CEILING
    with pytest.raises(ValidationFailedError, match="cap"):
        await writer_pairwise(_ops(host, cap=planned / 2), campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE)
    assert judge.calls == []
    report = await _pass(host, campaign_id, cap=planned)
    assert report.cap_usd == planned and report.calls_made == 2 * pairs


async def test_a_judge_of_a_writers_family_is_refused_before_any_call() -> None:
    judge = ScriptedPairJudge()
    host, campaign_id = _campaign(judge)
    with pytest.raises(ValidationFailedError, match="is of the writer .* family"):
        await writer_pairwise(
            _ops(host, family=lambda _model: "one-vendor"), campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE
        )
    assert judge.calls == []
    placed = await _pass(host, campaign_id, family=_apart)
    assert placed.judge_family == ["writer writers, judge judges"]
    unplaced = await _pass(host, campaign_id)
    assert all(check.startswith("unchecked") for check in unplaced.judge_family)


def _graded(document: str, slot: str) -> int:
    """A judge whose preference moves with the document: a spread of shares, leaning to the candidate."""
    lean = [5, 5, 4, 4, 4, 3, 5, 4, 3, 2, 5, 4][TOYHOST_DOCUMENTS.index(document)]
    return lean if slot == "B" else 6 - lean


@pytest.mark.parametrize(
    ("policy", "outcome"),
    [
        (prefers_the_candidate, "improved"),
        (prefers_the_control, "regressed"),
        (ties, "not_separated"),
        (_graded, None),
    ],
)
async def test_the_verdict_is_the_bounded_test_on_the_candidates_shares(policy: Policy, outcome: str | None) -> None:
    host, campaign_id = _campaign(ScriptedPairJudge(policy=policy))
    report = await _pass(host, campaign_id)

    candidate = [
        float(sum((_share(int(policy(document, slot)), slot) for slot in ("A", "B")), Fraction(0)) / 2)
        for document in sorted(TOYHOST_DOCUMENTS)
    ]
    control = [1.0 - share for share in candidate]
    p, interval = _expected(control, candidate)
    verdict = report.whole.verdict
    assert [case.share for case in report.cases] == pytest.approx(candidate)
    assert report.whole.p == p and verdict.p_adjusted == p
    assert verdict.interval == interval and verdict.interval_level == INTERVAL_LEVEL
    assert verdict.delta == pytest.approx(sum(candidate) / len(candidate) - sum(control) / len(control))
    assert p is not None and interval is not None
    separated = p < 0.05 and (interval[0] > 0 or interval[1] < 0)
    expected = ("improved" if interval[0] > 0 else "regressed") if separated else "not_separated"
    assert verdict.outcome == expected == (outcome or expected)
    assert (verdict.kind, verdict.reading, verdict.name) == ("contrast", "judged", "report_writer.preferred")
    assert verdict.arm == CANDIDATE and verdict.control == "control"
    assert verdict.reason == ("separated" if separated else "no_margin")


async def test_order_consistency_first_slot_bias_and_cannot_tell_are_reported() -> None:
    """A judge that always prefers the page it reads first contradicts itself across orders; one that cannot tell on
    one order of a document leaves that document scored one way only, counted apart from the rate."""
    undecided = TOYHOST_DOCUMENTS[0]

    def policy(document: str, slot: str) -> int | str:
        if document == undecided and slot == "A":
            return CANNOT_TELL
        return prefers_report_a(document, slot)

    host, campaign_id = _campaign(ScriptedPairJudge(policy=policy))
    report = await _pass(host, campaign_id)

    reliability = report.reliability
    assert reliability.one_order == 1
    assert reliability.both_orders == len(TOYHOST_DOCUMENTS) - 1
    assert reliability.order_consistent == 0 and reliability.order_consistency == 0.0
    assert reliability.first_slot_bias == pytest.approx(0.5)
    assert reliability.cannot_tell == len(_REPEATS)
    cases = {case.case_id: case for case in report.cases}
    assert cases[undecided].order_consistent is None and cases[undecided].cannot_tell == len(_REPEATS)
    assert cases[undecided].share == 0.0  # read second every time it was scored, so the control won those
    assert all(cases[document].first_slot_bias == pytest.approx(0.5) for document in TOYHOST_DOCUMENTS[1:])
    assert all(cases[document].share == 0.5 for document in TOYHOST_DOCUMENTS[1:])


async def test_the_reading_over_pairs_whose_pages_both_finished_is_reported_beside_the_whole() -> None:
    unfinished = 3
    host, campaign_id = _campaign(ScriptedPairJudge(policy=_graded), unfinished_control=unfinished)
    report = await _pass(host, campaign_id)

    kept = len(TOYHOST_DOCUMENTS) - unfinished
    assert report.whole.pairs == len(TOYHOST_DOCUMENTS) * len(_REPEATS) and report.whole.cases == len(TOYHOST_DOCUMENTS)
    assert report.finished_only.pairs == kept * len(_REPEATS) and report.finished_only.cases == kept
    finished_cases = set(TOYHOST_DOCUMENTS[:kept])
    candidate = [case.share for case in report.cases if case.case_id in finished_cases]
    assert None not in candidate
    shares = [share for share in candidate if share is not None]
    p, interval = _expected([1.0 - share for share in shares], shares)
    assert report.finished_only.p == p and report.finished_only.verdict.interval == interval
    assert all(pair.both_finished == (pair.case_id in finished_cases) for pair in report.pairs)


async def test_the_action_starts_the_pass_as_a_job_that_completes_with_its_report() -> None:
    judge = ScriptedPairJudge()
    eval_host, campaign_id = _campaign(judge)
    host = _ops(eval_host, family=_apart)
    tool = {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]
    started = await tool.call(
        {"action": "writer_pairwise", "campaign_id": campaign_id, "candidate": CANDIDATE}, host=host, caller=CALLER
    )
    assert not started.is_error, started.text
    assert started.structured is not None and len(started.structured["jobs"]) == 1
    (status,) = await await_job(host, JobsStarted.model_validate(started.structured), CALLER.scope_id)
    assert status.state == "completed", status.detail
    assert status.result is not None
    report = WriterPairwiseReport.model_validate(status.result)
    assert report.whole.verdict.outcome == "improved"
    assert report.judge_family == ["writer writers, judge judges"]
    assert len(judge.calls) == 2 * len(report.pairs)


@pytest.mark.parametrize("kept", [3, 1])
async def test_a_reading_on_too_few_cases_is_never_read_as_tested(kept: int) -> None:
    """Under the five cases a rate needs the verdict says it is not tested; under the two the test needs, it is
    untested, with no p and no interval."""
    scored = set(sorted(TOYHOST_DOCUMENTS)[:kept])

    def policy(document: str, slot: str) -> int | str:
        return prefers_the_candidate(document, slot) if document in scored else CANNOT_TELL

    host, campaign_id = _campaign(ScriptedPairJudge(policy=policy))
    report = await _pass(host, campaign_id)

    verdict = report.whole.verdict
    assert report.whole.cases == kept
    shares = [1.0] * kept
    p, interval = _expected([0.0] * kept, shares)
    assert report.whole.p == p and verdict.interval == interval
    if kept >= 2:
        assert verdict.reason == "too_few_cases" and verdict.words == f"not tested (n={kept})"
    else:
        assert p is None and interval is None
        assert (verdict.outcome, verdict.reason) == ("untested", "untestable")


async def test_a_failed_call_is_counted_and_scores_nothing() -> None:
    broken = sorted(TOYHOST_DOCUMENTS)[0]

    def policy(document: str, slot: str) -> int | str:
        return GARBLED if document == broken else prefers_the_candidate(document, slot)

    host, campaign_id = _campaign(ScriptedPairJudge(policy=policy))
    report = await _pass(host, campaign_id)

    assert report.reliability.failed == 2 * len(_REPEATS)
    cases = {case.case_id: case for case in report.cases}
    assert cases[broken].share is None and cases[broken].scored == 0
    assert report.whole.cases == len(TOYHOST_DOCUMENTS) - 1
    assert all(answer.answer == "failed" for pair in report.pairs if pair.case_id == broken for answer in pair.answers)


async def test_an_account_refusal_stops_the_pass_and_leaves_its_verdicts_untested() -> None:
    refused = sorted(TOYHOST_DOCUMENTS)[5]

    def policy(document: str, slot: str) -> int | str:
        return REFUSED if document == refused else prefers_the_candidate(document, slot)

    judge = ScriptedPairJudge(policy=policy)
    host, campaign_id = _campaign(judge)
    host = dataclasses.replace(host, failure_describer=describe_account_refusal)
    report = await _pass(host, campaign_id)

    assert report.stopped is not None and refused in report.stopped
    asked = [pair for pair in report.pairs if pair.answers]
    assert asked[-1].case_id == refused and len(asked[-1].answers) == 1
    assert all(not pair.answers for pair in report.pairs[len(asked) :])
    for reading in (report.whole, report.finished_only):
        assert (reading.verdict.outcome, reading.verdict.reason) == ("untested", "untestable")
        assert reading.p is None and reading.verdict.interval is None
