"""The pairwise pass: a writer campaign's candidate and control pages shown to the scoring judge side by side.

Each campaign is the promotion read's scripted one (a control and a candidate over the toy host's twelve documents,
three repeats each, scored by one judge), with every result's page stored as its judge evidence and the scoring judge
scripted: it reads which report is whose off the stored pages it is shown and answers by a policy the test names. So
what the pass asks, what it reads and the verdict it comes to are each checked against what the scripted judge was
asked and against the stats functions on shares the test computes itself. No model is called.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_DOCUMENTS, TOYHOST_SCOPE
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.writer_promotion_support import BETTER, CANDIDATE, SCORING_JUDGE, promotion_campaign
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis.stats import INTERVAL_LEVEL, bounded_difference_interval, bounded_separation_p
from threetears.evals.kernel import JobLeases
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.ops import JobsStarted, OpsHost, await_job
from threetears.evals.ops.report_writer.writer_contract import FINISHED_MEASURE, WRITER_KIND
from threetears.evals.ops.report_writer.writer_pairwise import (
    WRITER_PAIRWISE_ACTION,
    WriterPairwiseReport,
    writer_pairwise_work,
)
from threetears.evals.run import run_job_work
from threetears.evals.run.judge import CANNOT_TELL, JUDGE_CALL_ATTEMPTS
from threetears.evals.schema.models import (
    EvalTemplate,
    EvalTrace,
    JudgedArtifact,
    JudgeEvidence,
    eval_trace_doc_id,
)

#: The template every run is pointed at, so the judge's prompt carries its intent.
_TEMPLATE = "writer-template"
#: Every repeat a document ran, on both arms.
_REPEATS = (1, 2, 3)
#: What one judge call is priced at.
_CEILING = 0.01
#: The share range every bounded reading here is on.
_RANGE = (0.0, 1.0)

#: A page by arm, document and repeat: whether the cell stored one.
Pages = Callable[[str, str, int], bool]
#: The judge's answer, given the document and which slot (A or B) holds the candidate's page: a 1-5 score, or
#: CANNOT_TELL.
Policy = Callable[[str, str], int | str]


def _every_page(_arm: str, _document: str, _repeat: int) -> bool:
    return True


def _prefers_the_candidate(_document: str, slot: str) -> int:
    """Report A clearly better when the candidate is A, Report B when it is B."""
    return 1 if slot == "A" else 5


def _prefers_the_control(_document: str, slot: str) -> int:
    return 5 if slot == "A" else 1


def _ties(_document: str, _slot: str) -> int:
    return 3


def _prefers_report_a(_document: str, _slot: str) -> int:
    """A judge that always prefers the page it reads first, whoever wrote it."""
    return 1


def _page(arm: str, document: str, repeat: int) -> str:
    return f"# {arm} page on {document}, repeat {repeat}\n\nThe {arm} arm's findings."


@dataclass(frozen=True)
class _Completion:
    content: str
    stop_reason: str = "end_turn"
    temperature: float | None = None
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = SCORING_JUDGE
    served_model: str | None = SCORING_JUDGE


@dataclass
class _ScriptedJudge:
    """The scoring judge: reads which report is the candidate's off the two pages it is shown, answers by ``policy``."""

    policy: Policy = _prefers_the_candidate
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)
    model_name: str = SCORING_JUDGE

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        return _CEILING

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Completion:
        """Record which pages it read, in which slots, and answer."""
        dim = re.search(r'the single key "(.+?)"', system).group(1)  # type: ignore[union-attr]
        pages = re.search(r"=== Report A ===\n(.*?)\n\n=== Report B ===\n(.*)$", user, re.S)
        assert pages is not None, user
        first, second = pages.group(1), pages.group(2)
        document = re.search(r" page on (\S+),", first).group(1)  # type: ignore[union-attr]
        slot = "A" if first.startswith("# candidate") else "B"
        self.calls.append((document, slot, first, second))
        return _Completion(
            content=json.dumps({"reasoning": "read both", "criteria_scores": {dim: self.policy(document, slot)}})
        )

    async def aclose(self) -> None:
        """Nothing to release."""


def _campaign(
    judge: _ScriptedJudge, *, pages: Pages = _every_page, unfinished_control: int = 0
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
    off_loop(_store_pages, host, campaign_id, pages)
    return dataclasses.replace(host, clients=lambda _role, _model, *, temperature=None: judge), campaign_id


def _store_pages(host: EvalHost, campaign_id: str, pages: Pages) -> None:
    storage = host.storage
    storage.save_template(
        EvalTemplate(
            id=_TEMPLATE,
            scope_id=TOYHOST_SCOPE,
            name="writer",
            intent="Write the campaign's report.",
            candidate_kind=WRITER_KIND,
        )
    )
    campaign = storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert campaign is not None
    for arm, run_id in zip(("control", "candidate"), campaign.run_ids, strict=True):
        run = storage.load_eval_run(run_id, TOYHOST_SCOPE)
        assert run is not None
        storage.save_eval_run(run.model_copy(update={"template_id": _TEMPLATE}))
        for result in storage.query_eval_results(TOYHOST_SCOPE, run_id=run_id):
            if not pages(arm, result.test_case_id, result.k_iteration):
                continue
            evidence = JudgeEvidence(
                case_material=f"Computed facts of {result.test_case_id}.",
                artifact=_page(arm, result.test_case_id, result.k_iteration),
            )
            trace = EvalTrace(
                id=eval_trace_doc_id(result.id),
                scope_id=TOYHOST_SCOPE,
                result_id=result.id,
                eval_run_id=run_id,
                judge_evidence=evidence,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            storage.save_eval_result(result, trace)


async def _pass(
    host: EvalHost,
    campaign_id: str,
    *,
    family: Callable[[str], str | None] = lambda _model: None,
    cap: float | None = None,
) -> WriterPairwiseReport:
    """Run the pass in the foreground, as a script does: admitted, run to its end, its report read off its record."""
    leases = JobLeases(host.storage)
    try:
        work = await writer_pairwise_work(
            host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE, model_family=family, out_of_run_cap_usd=cap
        )
        ended = await run_job_work(work, store=host.storage, executor=host.blocking_executor, leases=leases)
    finally:
        await leases.shut_down()
    assert ended.action == WRITER_PAIRWISE_ACTION
    assert ended.outcome == "completed" and ended.result is not None, ended.detail
    return WriterPairwiseReport.model_validate(ended.result)


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
    judge = _ScriptedJudge()
    host, campaign_id = _campaign(judge)
    report = await _pass(host, campaign_id)

    pairs = len(TOYHOST_DOCUMENTS) * len(_REPEATS)
    assert len(report.pairs) == pairs and not report.unpaired
    assert report.calls_made == len(judge.calls) == 2 * pairs
    asked = sorted((document, slot, first, second) for document, slot, first, second in judge.calls)
    for document in TOYHOST_DOCUMENTS:
        for repeat in _REPEATS:
            mine, theirs = _page("candidate", document, repeat), _page("control", document, repeat)
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
    judge = _ScriptedJudge(policy=_ties)
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
    judge = _ScriptedJudge()
    host, campaign_id = _campaign(judge)
    pairs = len(TOYHOST_DOCUMENTS) * len(_REPEATS)
    planned = 2 * pairs * JUDGE_CALL_ATTEMPTS * _CEILING
    with pytest.raises(ValidationFailedError):
        await writer_pairwise_work(
            host,
            campaign_id,
            TOYHOST_SCOPE,
            candidate=CANDIDATE,
            model_family=lambda _model: None,
            out_of_run_cap_usd=planned / 2,
        )
    assert judge.calls == []
    report = await _pass(host, campaign_id, cap=planned)
    assert report.cap_usd == planned and report.calls_made == 2 * pairs


async def test_a_judge_of_a_writers_family_is_refused_before_any_call() -> None:
    judge = _ScriptedJudge()
    host, campaign_id = _campaign(judge)
    with pytest.raises(ValidationFailedError, match="is of the writer .* family"):
        await writer_pairwise_work(
            host,
            campaign_id,
            TOYHOST_SCOPE,
            candidate=CANDIDATE,
            model_family=lambda _model: "one-vendor",
            out_of_run_cap_usd=None,
        )
    assert judge.calls == []
    placed = await _pass(host, campaign_id, family=lambda model: "judges" if model == SCORING_JUDGE else "writers")
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
        (_prefers_the_candidate, "improved"),
        (_prefers_the_control, "regressed"),
        (_ties, "not_separated"),
        (_graded, None),
    ],
)
async def test_the_verdict_is_the_bounded_test_on_the_candidates_shares(policy: Policy, outcome: str | None) -> None:
    host, campaign_id = _campaign(_ScriptedJudge(policy=policy))
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
        return _prefers_report_a(document, slot)

    host, campaign_id = _campaign(_ScriptedJudge(policy=policy))
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
    host, campaign_id = _campaign(_ScriptedJudge(policy=_graded), unfinished_control=unfinished)
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
    judge = _ScriptedJudge()
    eval_host, campaign_id = _campaign(judge)
    launch = off_loop(ops_fixture, generation=False).host.launch
    host = OpsHost(
        launch=replace(launch, eval_host=eval_host),
        model_family=lambda model: "judges" if model == SCORING_JUDGE else "writers",
    )
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
