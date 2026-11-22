"""What a claims-graded eval's print and pages say about how its readings came to be, and nothing they do not hold.

A claims-only eval grades no trial pass or fail: claim readings never enter a trial's pass, so every answered trial
"passes" and a pass rate would count nothing. No surface states one there, and each says no grader decides a pass.
The claims judge has no calibration yet, and the print and the page both say its readings are directional, in the
words a rubric dimension's uncalibrated judge is described in. A runner that returns an ``Answer`` only to carry what
it read made no model call, and nothing reads one into it. A failed web check and a replayed one are said on the
quiet print as on the page. A reading the host derives from a criterion is carried by the run's summary, named for its
criterion on the page, and its interval stays inside the bound it declares. A runner whose own code raises a
programming error fails its trial as the host's bug, never as a model to change.

Every grader is scripted: it answers each call by the reply schema it is sent and reads its answer off the prompt, as a
model would, so nothing is spent. The text is invented: a town's bridges.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    make_template,
    minimal_aspect_design,
)
from packages.evals.tests.report_shapes_support import ArmTrials, AspectTrials, shape_of
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.off_loop import off_loop
from threetears.evals import Answer, Eval, Judge
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.report.build import build_code_only_report
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown, report_markdown
from threetears.evals.analysis.summary import summarize_run
from threetears.evals.kernel import CampaignArm
from threetears.evals.kernel.host import ClaimView, HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.quick import Claims, Document
from threetears.evals.run import get_result_trace, list_results
from threetears.evals.schema import EvalResult, EvalRun, WebBudget, WebPage, WebSearchResult
from threetears.evals.schema.models import ClaimCriterion, ClaimTally, PageBudget, RepairBudget

MODEL = "scripted/claims"
MEMORY = ":memory:"

_SEGMENT = re.compile(r"Segment (\d+):\n(.*?)(?=\n\nSegment \d+:|\n\n# |\Z)", re.S)
_PASSAGE = re.compile(r"^([SW]\d+)\. From [^\n]*:\n(.*?)(?=\n\n[SKW]\d+\. |\n\n# |\Z)", re.S | re.M)
_CLAIM = re.compile(r"^Claim (\d+), [^:]*: (.*)$", re.M)
_WEB_CLAIM = re.compile(r"^The claim: (.*)$", re.M)
_SENTENCE = re.compile(r"[^.\s][^.]*\.")


@dataclass(frozen=True)
class _Reply:
    """One completion, in the attribute names the engine reads."""

    content: str
    stop_reason: str = "end_turn"
    model: str = MODEL
    served_model: str | None = MODEL
    input_tokens: int | None = 100
    output_tokens: int | None = 20
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    price_source: str | None = "scripted"
    temperature: float | None = None


def _cited(claim: str, passages: list[tuple[str, str]], *, contradicts: bool) -> dict[str, Any] | None:
    """The first shown sentence on the claim's subject (its first three words): a support where it says the same, a
    contradiction (when asked for) where it says otherwise, quoting its subject and its last word as the value."""
    subject = " ".join(claim.split()[:3])
    for ref, text in passages:
        for sentence in _SENTENCE.findall(text):
            same = sentence.rstrip(".") == claim.rstrip(".")
            if sentence.startswith(subject) and (same or contradicts):
                words = sentence.rstrip(".").split()
                stance = "supports" if same else "contradicts"
                return {"ref": ref, "stance": stance, "quote": sentence, "subject": subject, "value": words[-1]}
    return None


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeClaimsGrader:
    """Answers each claims call from what its prompt shows, and a rubric dimension a pass."""

    model_name: str = MODEL
    asked: list[str] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        if name == "judge_reply":
            (key,) = re.findall(r'single key "([^"]+)"', system)
            return _Reply(json.dumps({"reasoning": "reads well", "criteria_scores": {key: "pass"}}))
        if name == "claim_listing":
            listed = [
                {
                    "segment": int(index),
                    "claim": text.strip(),
                    "excerpt": text.strip().rstrip("."),
                    "specific": True,
                    "hedged": False,
                    "attributed": False,
                    "scope": "world",
                    "restates": None,
                }
                for index, text in _SEGMENT.findall(user)
            ]
            return _Reply(json.dumps({"claims": listed}))
        passages = _PASSAGE.findall(user)
        if name == "claim_truth":
            truth = []
            for index, claim in _CLAIM.findall(user):
                cited = _cited(claim, passages, contradicts=True)
                truth.append({"claim": int(index), "citations": [cited] if cited else [], "suspect": cited is None})
            return _Reply(json.dumps({"claims": truth}))
        if name == "claim_ground":
            ground = []
            for index, claim in _CLAIM.findall(user):
                cited = _cited(claim, passages, contradicts=False)
                if cited is not None:
                    del cited["stance"]
                ground.append({"claim": int(index), "citations": [cited] if cited else []})
            return _Reply(json.dumps({"claims": ground}))
        if name == "claim_web_queries":
            (claim,) = _WEB_CLAIM.findall(user)
            return _Reply(json.dumps({"queries": [claim]}))
        if name == "claim_web_verdict":
            (claim,) = _WEB_CLAIM.findall(user)
            cited = _cited(claim, passages, contradicts=True)
            if cited is None:
                return _Reply(
                    json.dumps({"outcome": "inconclusive", "ref": None, "quote": None, "subject": None, "value": None})
                )
            outcome = "true" if cited.pop("stance") == "supports" else "false"
            return _Reply(json.dumps({"outcome": outcome, **cited}))
        raise AssertionError(f"the scripted grader was asked {name!r}, which no test here scripts")

    async def aclose(self) -> None:
        """Nothing to release."""


# parity-with: threetears.evals.schema.WebSearch
@dataclass
class _FakeSearch:
    """Answers every query with its one page, from a recording where ``replayed``; raises where ``down``."""

    page: WebPage
    replayed: bool = False
    down: bool = False
    identity: str = "scripted-search"

    async def search(self, query: str) -> WebSearchResult:
        if self.down:
            raise ConnectionError("the search is down")
        return WebSearchResult(pages=(self.page,), usd=0.0, replayed=self.replayed)

    def price_ceiling(self) -> float | None:
        return 0.0


#: Twelve towns, each with its bridge's year: enough cases for two arms' readings to be compared.
CASES = [{"town": f"Town{at}", "source": f"The Town{at} bridge opened in {1900 + at}. It is old."} for at in range(12)]


async def recalls(case: dict[str, Any]) -> str:
    """Say when the town's bridge opened."""
    return str(case["source"].split(" It ")[0])


async def guesses(case: dict[str, Any]) -> str:
    """Say when the town's bridge opened."""
    at = int(case["town"].removeprefix("Town"))
    return f"The {case['town']} bridge opened in {1900 + at + at % 3 * 7}."


def _claims() -> Claims:
    return Claims(_FakeClaimsGrader(), case_documents="source")


def _lines(text: str, pattern: str) -> list[str]:
    return [line for line in text.splitlines() if re.search(pattern, line)]


# =============================================================================
# An Answer that only carries documents made no model call
# =============================================================================


async def test_an_answer_carrying_only_documents_reports_no_model_call() -> None:
    async def read(case: dict[str, Any]) -> Answer:
        return Answer(await recalls(case), documents=[Document(case["source"], title="archive")])

    result = await Eval(CASES[:3], read, [Claims(_FakeClaimsGrader(), grounded_in="run")], k=1).run(store=MEMORY)

    printed = result.render()
    assert "unpriced" not in printed and "Model identity unconfirmed" not in printed
    assert "pools with no other run" not in printed
    assert "candidate spend" not in result.render(full=True), "only the judge spent"
    stored = off_loop(list_results, result.host.storage, result.summary.run_id, result.scope_id)
    assert [[row.role for row in cell.usage] for cell in stored] == [["judge"]] * 3, "no candidate row"
    first = off_loop(get_result_trace, result.host.storage, stored[0]).claim_evidence
    assert first is not None and [document.title for document in first.documents or ()] == ["archive"]


@pytest.mark.parametrize(
    "carried",
    [
        pytest.param({"session": {"ordered": ["tea"]}}, id="session"),
        pytest.param({"session": {"ordered": ["tea"]}, "documents": ["The archive."]}, id="session-and-documents"),
    ],
)
async def test_an_answer_carrying_only_its_session_or_documents_reports_no_model_call(carried: dict[str, Any]) -> None:
    """The run's own record beside the answer is no model call either: no candidate usage row, no unpriced call."""

    async def kept(case: dict[str, Any]) -> Answer:
        return Answer(await recalls(case), **carried)

    result = await Eval(CASES[:3], kept, [_claims()], k=1).run(store=MEMORY)

    stored = off_loop(list_results, result.host.storage, result.summary.run_id, result.scope_id)
    assert [[row.role for row in cell.usage] for cell in stored] == [["judge"]] * 3, "no candidate row"
    assert "unpriced" not in result.render() and "Model identity unconfirmed" not in result.render()


async def test_an_answer_naming_its_model_still_reports_its_call() -> None:
    async def called(case: dict[str, Any]) -> Answer:
        return Answer(await recalls(case), model="m-1", documents=())

    result = await Eval(CASES[:3], called, [Claims(_FakeClaimsGrader(), grounded_in="run")], k=1).run(store=MEMORY)

    stored = off_loop(list_results, result.host.storage, result.summary.run_id, result.scope_id)
    assert all(row.model == "m-1" for cell in stored for row in cell.usage if row.role == "candidate")
    assert "candidate spend: unpriced over 3 call(s)" in result.render(full=True), "a model named is a call made"


# =============================================================================
# No pass rate where nothing can fail a trial; the claims judge said uncalibrated
# =============================================================================

#: What the print and the page say of a claims judge no person has checked.
UNCALIBRATED = (
    f"Judge {MODEL} is uncalibrated on answer.claims: no person has checked its claim verdicts yet, so its readings "
    "are directional."
)


async def test_one_arm_graded_by_claims_alone_states_no_pass_rate_and_its_judge_uncalibrated() -> None:
    result = await Eval(CASES, recalls, [_claims()], k=1, name="bridges").run(store=MEMORY)

    printed, page, aspect = result.render(), campaign_report_markdown(result.page), report_markdown(result.report)
    assert printed.splitlines()[0] == "bridges: recalls ran 12 of 12 cases; no grader here says pass or fail."
    assert UNCALIBRATED in printed
    assert "**recalls ran 12 of 12 cases; no grader here says pass or fail.**" in page
    assert "Pass rate" not in page and "passed 12 of 12" not in page
    assert "Judged scores on bridges are directional until people's labels calibrate the judge." in page
    assert UNCALIBRATED in page and UNCALIBRATED in aspect
    (header,) = _lines(aspect, r"^\| Arm \| Ran \|")
    assert header == "| Arm | Ran | Usable | Cost per 1,000 replies |", "no count of passes or grader failures"
    for shown in (page, aspect):
        assert "✓" not in shown and "Every one passed" not in shown and "pass^k" not in shown, "no mark says passed"
        assert "12 trials ran. Every one was answered." in shown
        terms = {line.split("**")[1] for line in _lines(shown, r"^- \*\*")}
        assert not terms & {"passed", "failed by check", "failed by judge", "marks", "flag", "outcome"}, terms


async def test_an_arm_failing_to_answer_where_nothing_can_fail_a_trial_is_compared_on_how_often_it_answered() -> None:
    """A runner crashing on 3 of 12 cases of an eval graded by numbers alone: no pass rate, but the share answered is
    read, compared and said, so a crashing arm is never read on its survivors alone."""

    def length(case: Mapping[str, Any], answer: str) -> float:
        return float(len(answer))

    result = await Eval(CASES, {"recalls": recalls, "buggy": buggy}, [length], k=1, name="lengths").run(store=MEMORY)

    printed, page, aspect = result.render(), campaign_report_markdown(result.page), report_markdown(result.report)
    assert "buggy answered 9 of 12 cases." in printed.splitlines()[0]
    (lead,) = _lines(page, r"^\*\*")[:1]
    assert "buggy answered 9 of 12 cases." in lead, lead
    assert _lines(page, r"^\| Aspect \| Cases \| ▸ recalls · answered \| buggy · answered vs recalls \|$")
    assert (
        "lengths has no pass rate: no grader there says pass or fail, so what its row compares is how often each"
        in page
    )
    assert "Pass rate" not in page
    (header,) = _lines(aspect, r"^\| Arm \| Ran \|")
    assert "| Usable | Answered | Code raised |" in header
    assert _lines(aspect, r"^\| case 1 \| regression \| ● \| ✗ \(failed: code raised ×1\) \|")
    assert "24 trials ran. Of them 21 were answered and 3 failed to answer." in aspect


async def test_one_arm_failing_to_answer_some_says_how_many_it_answered() -> None:
    def length(case: Mapping[str, Any], answer: str) -> float:
        return float(len(answer))

    result = await Eval(CASES, buggy, [length], k=1, name="lengths").run(store=MEMORY)

    said = "buggy answered 9 of 12 cases; no grader here says pass or fail."
    assert result.render().splitlines()[0].startswith(f"lengths: {said}")
    assert f"**{said}**" in campaign_report_markdown(result.page), "never 'ran 12 of 12'"


async def test_two_arms_graded_by_claims_alone_are_compared_on_their_claim_readings_and_no_pass_rate() -> None:
    result = await Eval(CASES, {"recalls": recalls, "guesses": guesses}, [_claims()], k=1, name="bridges").run(
        store=MEMORY
    )

    page = campaign_report_markdown(result.page)
    assert not _lines(page, r"^\*\*Pass rate|^\| bridges \| .*\| Pass rate|· pass rate"), "no pass-rate lead or row"
    assert "bridges has no pass rate: no grader there says pass or fail, so its readings are read on their own." in page
    assert _lines(page, r"^\| False claims per answer \(answer\.claims, judged\) \|"), "the claim readings compared"
    assert UNCALIBRATED in result.render()


async def test_a_claims_eval_with_a_judged_dimension_keeps_its_pass_rate() -> None:
    """The control: a rubric dimension can fail a trial, so the pass rate counts something and stands."""
    client = _FakeClaimsGrader()
    judge = Judge(client, {"clear": "The answer is clear."})
    result = await Eval(CASES, recalls, [judge, Claims(client, case_documents="source")], k=1, name="bridges").run(
        store=MEMORY
    )

    page = campaign_report_markdown(result.page)
    assert "| bridges | Pass rate | 100% (12 of 12 cases) |" in page
    (header,) = _lines(report_markdown(result.report), r"^\| Arm \| Ran \|")
    assert "| Passed | Failed by check | Failed by judge |" in header
    # The quiet print names each judge not yet calibrated, the rubric dimension's as the claim criterion's, so none
    # left unsaid reads as calibrated.
    printed = result.render()
    assert (
        f"Judge {MODEL} is uncalibrated on answer clear: no human labels yet, so its scores are directional. Rate about "
        "20 of its outputs to calibrate it."
    ) in printed
    assert UNCALIBRATED in printed


async def test_an_eval_graded_by_numbers_alone_states_no_pass_rate_either() -> None:
    """Never by the eval's kind: a number decides no pass, as a claim reading does not."""

    def length(case: Mapping[str, Any], answer: str) -> float:
        return float(len(answer))

    result = await Eval(CASES, recalls, [length], k=1, name="lengths").run(store=MEMORY)

    page = campaign_report_markdown(result.page)
    assert "Pass rate" not in page and "passed 12 of 12" not in page
    assert "**recalls ran 12 of 12 cases; no grader here says pass or fail.**" in page


# =============================================================================
# A failed or replayed web check is said on the quiet print as on the page
# =============================================================================

_PAGE = WebPage(url="https://example.org/town0", title="Town0", text="The Town0 bridge opened in 1900.")
_BUDGET = WebBudget(
    max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=400, window_chars=200
)


@pytest.mark.parametrize(
    ("search", "said"),
    [
        pytest.param(
            _FakeSearch(_PAGE, down=True),
            "The web check of answer.claims failed in 2 cells of bridges: its undecided claims read unverifiable, "
            "never false, so its false claims there may be undercounted.",
            id="failed",
        ),
        pytest.param(
            _FakeSearch(_PAGE, replayed=True),
            "The web checks of answer.claims read a recording in 2 cells of bridges, never a live search: what the web "
            "decided there is what the recorded pages said.",
            id="replayed",
        ),
    ],
)
async def test_a_failed_or_replayed_web_check_is_said_on_the_quiet_print_and_the_page(
    search: _FakeSearch, said: str
) -> None:
    async def bridges(case: dict[str, Any]) -> str:
        return "The Town0 bridge opened in 1951."

    cases = [{"id": f"town0-{at}"} for at in range(2)]
    result = await Eval(cases, bridges, [Claims(_FakeClaimsGrader(), search=search, web=_BUDGET)], k=1).run(
        store=MEMORY
    )

    assert said in result.render()
    assert said in report_markdown(result.report) and said in campaign_report_markdown(result.page)


async def test_a_live_web_check_says_nothing_of_a_recording() -> None:
    async def bridges(case: dict[str, Any]) -> str:
        return "The Town0 bridge opened in 1951."

    cases = [{"id": f"town0-{at}"} for at in range(2)]
    result = await Eval(cases, bridges, [Claims(_FakeClaimsGrader(), search=_FakeSearch(_PAGE), web=_BUDGET)], k=1).run(
        store=MEMORY
    )

    assert "read a recording" not in result.render() + report_markdown(result.report)
    for cell in off_loop(list_results, result.host.storage, result.summary.run_id, result.scope_id):
        assert cell.claim_tallies["answer.claims"].web_replayed is False


# =============================================================================
# A reading derived from a criterion: carried by the summary, named for its criterion, inside its bound
# =============================================================================

_CRITERION = "show.claims"
_COUNT = f"{_CRITERION}_departed_named"


def _count_reader(view: ClaimView) -> Mapping[str, float | bool]:
    """How many of the cell's claims name someone the session record says has left."""
    gone: list[str] = list((view.session or {}).get("gone", ()))
    return {_COUNT: float(sum(any(name in claim.text for name in gone) for claim in view.claims))}


def _profile() -> HostProfile:
    base = toyhost_profile()
    derived = MetricDescriptor(
        name=_COUNT,
        reader_name="Departed people named per draft",
        data_type="numeric",
        family="claims",
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        description="How many of a draft's claims name someone the session record says has left.",
        derived_from=_CRITERION,
        higher_is_better=False,
        nonnegative=True,
    )
    return replace(
        base,
        measures=MeasureRegistry([*TOYHOST_MEASURES, derived], families=base.measures.families),
        claim_readers={_CRITERION: _count_reader},
    )


def _criterion() -> ClaimCriterion:
    return ClaimCriterion(
        name=_CRITERION,
        pages=PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
        repairs=RepairBudget(rounds=2, max_items_per_cell=4),
        judge="builtin",
    )


def _tally(count: float) -> ClaimTally:
    return ClaimTally(
        criterion_fingerprint="c" * 64,
        documents_declared=False,
        units=2,
        units_with_false=0,
        session_claims=0,
        derived={_COUNT: count},
    )


def _stored(counts: Mapping[str, list[float]]) -> tuple[list[EvalRun], dict[str, list[EvalResult]]]:
    """One run per arm, a cell per count, each cell's tally carrying the derived count."""
    runs: list[EvalRun] = []
    results: dict[str, list[EvalResult]] = {}
    for model, values in counts.items():
        run = make_eval_run(status="completed", candidate_model=model, template_id="tpl-show")
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{at:02d}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{at:02d}",
                goal_state_outcomes=[],
                rubric_scores=[],
                cost_usd=None,
                claim_tallies={_CRITERION: _tally(value)},
            )
            for at, value in enumerate(values)
        ]
    return runs, results


def test_the_run_summary_carries_a_reading_derived_from_a_criterion() -> None:
    runs, results = _stored({"draft-model": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]})
    (run,) = runs
    host = toyhost_host(profile=_profile())
    off_loop(host.storage.save_eval_run, run)
    for result in results[run.id]:
        off_loop(host.storage.save_eval_result, result, None)

    summary = summarize_run(host, run.id, run.scope_id)

    (line,) = [measure for measure in summary.measure_lines if measure.name == _COUNT]
    assert line.state == "measured" and line.text.startswith(f"{_COUNT}: mean 0.33")
    assert summary.uncarried_text is None or _COUNT not in summary.uncarried_text


def test_a_derived_count_is_named_for_its_criterion_and_its_interval_stays_above_zero() -> None:
    runs, results = _stored(
        {"memory-model": [0.0] * 12, "lookup-model": [1.0 if at in (0, 5) else 0.0 for at in range(12)]}
    )
    asked = min(run.created_at for run in runs)
    campaign = make_aspect_campaign(
        label="show",
        template_id="tpl-show",
        run_ids=[run.id for run in runs],
        design=minimal_aspect_design(merit_axes=["quality"]),
        question="does the new writer name departed people less often?",
        asked_at=asked,
        control=CampaignArm(
            label="memory", role="control", variant_key=fixture_variant_key("memory-model"), named_at=asked
        ),
        scope_id=runs[0].scope_id,
        name="departed people",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
    )
    template = make_template(id="tpl-show", rubric=[], claim_criteria=[_criterion()])
    bundle = assemble_context_bundle(
        campaign, storage=ToyhostStorage(runs, results, templates=[template]), profile=_profile()
    )

    (measure,) = [held for held in bundle.judged_measures if held.name == _COUNT]
    assert (measure.reader_name, measure.nonnegative) == (f"Departed people named per draft ({_CRITERION})", True)
    page = report_markdown(
        build_code_only_report(bundle, assembled_at="2026-11-22T00:00:00+00:00", campaign_name="departed people")
    )
    (row,) = _lines(page, rf"^\| Departed people named per draft \({re.escape(_CRITERION)}, judged\) \|")
    # Two of twelve: the t interval on the mean reaches below zero, and the count declares it never is.
    assert "| 0.17 (0 to " in row, row


# =============================================================================
# A runner's own bug is the host's, never a model's
# =============================================================================

#: The cases a runner's own code raises on.
_BROKEN = {"Town0", "Town1", "Town2"}


async def buggy(case: dict[str, Any]) -> str:
    """Say when the town's bridge opened, reading a field only some cases carry."""
    if case["town"] in _BROKEN:
        return str(case["opened"])
    return await recalls(case)


async def test_a_runner_s_own_programming_error_is_named_as_the_host_s_code() -> None:
    result = await Eval(CASES, {"recalls": recalls, "buggy": buggy}, [_claims()], k=1, name="bridges").run(store=MEMORY)

    stored = off_loop(list_results, result.host.storage, result.arms["buggy"].run_id, result.scope_id)
    raised = [cell for cell in stored if cell.candidate_error is not None]
    assert len(raised) == 3 and all(cell.candidate_code_raised for cell in raised)
    assert all(cell.candidate_code_raised is None for cell in stored if cell.candidate_error is None)
    aspect = report_markdown(result.report)
    assert "model failed" not in aspect.lower()
    (header,) = _lines(aspect, r"^\| Arm \| Ran \|")
    assert "| Code raised |" in header
    assert len(_lines(aspect, r"\| buggy \| failed \| code raised \|")) == 3
    # Named as the code's, with the reply's handling to check, since an empty or malformed reply can set it off too.
    assert "check the runner's handling of the reply" in aspect
    assert "no other model" not in aspect


async def test_a_runner_whose_own_code_raises_on_every_case_is_said_so() -> None:
    async def broken(case: dict[str, Any]) -> str:
        return str(case["opened"])

    result = await Eval(CASES, {"recalls": recalls, "broken": broken}, [_claims()], k=1, name="bridges").run(
        store=MEMORY
    )

    aspect = report_markdown(result.report)
    assert "Every trial of broken failed: its own code raised while handling the reply on every case." in aspect
    assert "the model refused or errored" not in aspect
    assert result.report.bundle.all_failed is not None
    assert "the candidate's own code raised while handling the reply on every call" in result.report.bundle.all_failed
    assert "model refused" not in result.report.bundle.all_failed, "the stored sentence names the cause too"


async def test_a_runner_raising_what_a_client_raises_is_still_read_as_its_model_failing() -> None:
    """The control: a raise no programming error names (a client's own failure) keeps the model-failed reading."""

    async def refused(case: dict[str, Any]) -> str:
        if case["town"] in _BROKEN:
            raise RuntimeError("the model's client gave up")
        return await recalls(case)

    result = await Eval(CASES, {"recalls": recalls, "refused": refused}, [_claims()], k=1, name="bridges").run(
        store=MEMORY
    )

    stored = off_loop(list_results, result.host.storage, result.arms["refused"].run_id, result.scope_id)
    assert all(cell.candidate_code_raised is None for cell in stored)
    assert len(_lines(report_markdown(result.report), r"\| refused \| failed \| model failed \|")) == 3


# =============================================================================
# Across aspects: the aggregate and the sizing table name the rate as each aspect reads it
# =============================================================================


def _across(*, facts_checked: bool) -> str:
    """A change on two aspects, its candidate failing to answer some cases on each: nothing graded can fail a trial on
    ``tone``, and on ``facts`` only where ``facts_checked``. The campaign's page, as Markdown."""
    facts = (
        [ArmTrials(label="v1", passing=20), ArmTrials(label="v2", passing=18, unanswered=2)]
        if facts_checked
        else [ArmTrials(label="v1"), ArmTrials(label="v2", unanswered=2)]
    )
    aspects = [
        AspectTrials(label="tone", arms=[ArmTrials(label="v1"), ArmTrials(label="v2", unanswered=3)], checks=False),
        AspectTrials(label="facts", arms=facts, checks=facts_checked),
    ]
    return campaign_report_markdown(shape_of("answering", aspects, control=True).page())


def test_across_aspects_nothing_can_fail_the_aggregate_and_the_sizing_table_read_answered() -> None:
    page = _across(facts_checked=False)

    assert "pass rate" not in page.replace("has no pass rate", "").lower().replace("never on a pass rate", "")
    assert _lines(page, r"^\| Aspect \| Cases \| ▸ v1 · answered \| v2 · answered vs v1 \|$")
    assert "**Answered and cost per 1,000 replies, each arm, 95% intervals, all aspects (equal weight)**" in page
    assert _lines(page, r"^\| Arm \| Cost per 1,000 replies \| Interval \| Answered \(%\) \| Interval \(%\) \|$")
    sized = _lines(page, r"^\| v2 \| .* \| (tone|facts) \| no worse than the 10-point margin \|")
    assert len(sized) == 2 and all("| Answered |" in row for row in sized), sized


def test_across_aspects_one_with_a_check_the_rate_is_named_as_both_and_each_row_as_its_own() -> None:
    page = _across(facts_checked=True)

    assert _lines(page, r"^\| Aspect \| Cases \| ▸ v1 · pass or answered rate \| v2 · pass or answered rate vs v1 \|$")
    assert (
        "**Pass or answered rate and cost per 1,000 replies, each arm, 95% intervals, all aspects (equal weight)**"
        in page
    )
    (tone,) = _lines(page, r"^\| v2 \| .* \| tone \| no worse than the 10-point margin \|")
    (facts,) = _lines(page, r"^\| v2 \| .* \| facts \| no worse than the 10-point margin \|")
    assert "| Answered |" in tone and "| Pass rate |" in facts
