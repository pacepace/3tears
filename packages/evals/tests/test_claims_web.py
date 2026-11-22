"""The web check of claims grading: a nominated claim checked on a host's search, inside the truth step.

Every behaviour is driven through the real runner over a scripted judge, which answers each call by the reply schema
it is sent, and a scripted search, so nothing is spent and each page is chosen. The web decides only a claim the judge
itself nominated (no case evidence bears on it, and the judge doubts it); case evidence is never searched past; the
candidate's own documents are never shown; a web verdict quotes a window it was shown; what was not checked reads
``skipped_cap`` and ``unverifiable``, never false; a port's failure loses the criterion once, never retried; and its
searches are billed as the cell's judge spend in rows of their own. The text is invented: a reading room's almanac.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.claims.grading import criterion_ceiling, criterion_fingerprint, criterion_steps
from threetears.evals.kernel.claims.readings import claim_readings
from threetears.evals.kernel.host import ReplayMiss
from threetears.evals.kernel.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.kernel.usage_capture import ExternalRateTable
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.runner import RunnerOptions, execute_run, run_one_result
from threetears.evals.kernel.claims.readings import claim_tally
from threetears.evals.run.runner import judge_witnessed_output
from threetears.evals.schema import Segment, WebPage, WebSearchResult
from threetears.evals.schema.models import (
    ClaimCall,
    ClaimCriterion,
    ClaimEvidence,
    ClaimGrade,
    EvalTemplate,
    EvalTestCase,
    EvidenceDocument,
    GradedClaim,
    JudgedArtifact,
    PageBudget,
    RepairBudget,
    WebBudget,
    WebCheckRecord,
    WebWindow,
)
from threetears.evals.schema.values import SweepableValue
from threetears.evals.testing import RecordedWebSearch, RecordingRefused

_KIND = "claims-web-single-shot"
_JUDGE = "scripted/claims-judge"
_CRITERION = "almanac.claims"
_PORT = "scripted-search"

_FIRST = "The Lantern Almanac was first printed in 1921."
_SECOND = "Its pages were bound in green cloth."

#: The shelf notes the candidate itself was given: never shown to the web step.
_NOTES = EvidenceDocument(
    id="shelf-notes", provenance="run", title="Shelf notes", text="Shelf notes: the almanac's catalogue number is 77."
)

_SEGMENTS = (Segment(text=_FIRST, unit=0), Segment(text=_SECOND, unit=1), Segment(text="Again: " + _FIRST, unit=2))

_EVIDENCE = ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=(_NOTES,))

#: Case evidence that bears on no claim the cell states.
_ROOM_RULES = "Readers of the Lantern Almanac must wear gloves, and no volume leaves the reading room."

#: A page that states the first claim.
_PAGE = WebPage(
    url="https://example.org/almanac",
    title="The Lantern Almanac",
    text="The Lantern Almanac was first printed in 1921 by Harrow Press. It ran to forty pages.",
)


def _listed(segment: int, text: str, *, restates: int | None = None) -> dict[str, Any]:
    return {
        "segment": segment,
        "claim": text,
        "excerpt": text.rstrip("."),
        "specific": True,
        "hedged": False,
        "attributed": False,
        "scope": "world",
        "restates": restates,
    }


def _listing(*claims: dict[str, Any]) -> str:
    return json.dumps({"claims": list(claims)})


def _truth(claim: int, *, suspect: bool = True, cited: list[dict[str, Any]] | None = None) -> str:
    return json.dumps({"claims": [{"claim": claim, "citations": cited or [], "suspect": suspect}]})


def _verdict(outcome: str = "true", **overrides: Any) -> str:
    answer: dict[str, Any] = {"outcome": outcome, "ref": None, "quote": None, "subject": None, "value": None}
    if outcome != "inconclusive":
        answer.update(
            ref="W1",
            quote="The Lantern Almanac was first printed in 1921 by Harrow Press.",
            subject="The Lantern Almanac",
            value="1921",
        )
    answer.update(overrides)
    return json.dumps(answer)


#: What each step answers by default: one claim, nominated, and found true on its page.
_ANSWERS: dict[str, str] = {
    "claim_listing": _listing(_listed(0, _FIRST)),
    "claim_truth": _truth(0),
    "claim_web_queries": json.dumps({"queries": ["Lantern Almanac first printed"]}),
    "claim_web_verdict": _verdict(),
    "claim_web_repair": _verdict(),
}


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _JUDGE
    served_model: str | None = _JUDGE
    input_tokens: int | None = 400
    output_tokens: int | None = 80
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.002
    price_source: str | None = "scripted"
    temperature: float | None = None
    sent_settings: Any = None


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeWebJudge:
    """Answers each call by the reply schema it is sent, from its script (the last answer repeats), or hangs.

    Records each call's schema name and the prompt it was sent.
    """

    script: dict[str, list[Any]] = field(default_factory=dict)
    usd_per_call: float | None = None
    model_name: str = _JUDGE
    asked: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return self.usd_per_call

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        self.prompts.append(system + "\n" + user)
        answers = self.script.get(name) or [_ANSWERS[name]]
        answer = answers[min(self.asked.count(name), len(answers)) - 1]
        if answer == "hang":
            await asyncio.Event().wait()
        if isinstance(answer, BaseException):
            raise answer
        return _Reply(content=answer)

    async def aclose(self) -> None:
        """Nothing to release."""


# parity-with: threetears.evals.schema.web_search.WebSearch
@dataclass
class _FakeWebSearch:
    """Answers every query with its pages (``by_query`` first), or raises its failure; records every query asked."""

    pages: tuple[WebPage, ...] = (_PAGE,)
    usd_per_query: float | None = 0.01
    reported_usd: float | None = None
    by_query: dict[str, tuple[WebPage, ...]] = field(default_factory=dict)
    failure: BaseException | None = None
    identity: str = _PORT
    queries: list[str] = field(default_factory=list)

    async def search(self, query: str) -> WebSearchResult:
        self.queries.append(query)
        if self.failure is not None:
            raise self.failure
        return WebSearchResult(pages=self.by_query.get(query, self.pages), usd=self.reported_usd)

    def price_ceiling(self) -> float | None:
        return self.usd_per_query


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeWebKind:
    """A single-shot kind graded by code that hands back one canned output and its claim evidence."""

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, evidence: ClaimEvidence = _EVIDENCE) -> None:
        self.output = CandidateOutput(output=[{"answer": _FIRST}], claim_evidence=evidence)

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> object:
        return object()

    async def invoke(self, instance: object, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        return self.output


_WEB = WebBudget(max_queries_per_claim=2, max_checked_per_cell=2, results_per_query=2, page_chars=400, window_chars=200)


def _criterion(**overrides: Any) -> ClaimCriterion:
    fields: dict[str, Any] = {
        "name": _CRITERION,
        "pages": PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
        "repairs": RepairBudget(rounds=2, max_items_per_cell=4),
        "web": _WEB,
        "judge": "builtin",
    }
    fields.update(overrides)
    return ClaimCriterion(**fields)


def _template(criterion: ClaimCriterion | None = None) -> EvalTemplate:
    return EvalTemplate(
        scope_id="u",
        name="almanac answers",
        intent="answer questions about the reading room's almanac",
        candidate_kind=_KIND,
        claim_criteria=[criterion or _criterion()],
    )


def _service(judge: _FakeWebJudge) -> JudgeService:
    return JudgeService(
        client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
    )


_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}

_RATES = ExternalRateTable(rates={(_PORT, "query"): 0.01})


async def _cell(
    judge: _FakeWebJudge,
    port: Any,
    *,
    template: EvalTemplate | None = None,
    rates: ExternalRateTable | None = _RATES,
    evidence: ClaimEvidence = _EVIDENCE,
) -> Any:
    template = template or _template()
    kind = _FakeWebKind(evidence)
    return await run_one_result(
        toyhost_host(),
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id="run-1",
        scope_id="u",
        judge_service=_service(judge),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}, web_search=port, external_rates=rates),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )


async def test_a_nominated_claim_the_web_confirms_reads_true_ungrounded_by_a_quote_of_a_window_it_read() -> None:
    judge = _FakeWebJudge()
    port = _FakeWebSearch()

    result, trace = await _cell(judge, port)

    grade = trace.claim_grades[_CRITERION]
    (claim,) = grade.claims
    assert (claim.verdict, claim.graded_against, claim.nominated_by, claim.web) == (
        "true_ungrounded",
        "web",
        "judge",
        "checked",
    )
    assert judge.asked == ["claim_listing", "claim_truth", "claim_web_queries", "claim_web_verdict"]
    assert claim.decided_by == 3 and grade.calls[3].role == "web", "the reading call fixed the verdict"
    (record,) = grade.web
    assert record.queries == ("Lantern Almanac first printed",) and record.urls == (_PAGE.url,)
    assert [window.text for window in record.windows] == [_PAGE.text], "the windows read are stored on the trace"
    assert record.quote == "The Lantern Almanac was first printed in 1921 by Harrow Press."
    assert record.value_span is not None and record.quote[record.value_span.start : record.value_span.end] == "1921"
    assert port.queries == ["Lantern Almanac first printed"]
    (counts,) = result.claim_tallies[_CRITERION].by_judge
    assert (counts.true_ungrounded, counts.web_checked, counts.web_decided, counts.asserted_contradicted) == (
        1,
        1,
        1,
        0,
    )
    assert not grade.web_replayed


async def test_a_nominated_claim_a_window_contradicts_reads_contradicted_against_the_web() -> None:
    page = WebPage(url="https://example.org/almanac", title="", text="The Lantern Almanac was first printed in 1931.")
    contradiction = _verdict("false", quote="The Lantern Almanac was first printed in 1931.", value="1931")
    judge = _FakeWebJudge(script={"claim_web_verdict": [contradiction]})

    result, trace = await _cell(judge, _FakeWebSearch(pages=(page,)))

    (claim,) = trace.claim_grades[_CRITERION].claims
    assert (claim.verdict, claim.graded_against) == ("contradicted", "web")
    assert claim_readings(_CRITERION, result.claim_tallies[_CRITERION], grounded=False)[f"{_CRITERION}_false"] == 1


@pytest.mark.parametrize("stance", ["supports", "contradicts"])
async def test_case_evidence_decides_a_claim_the_web_never_searches_past(stance: str) -> None:
    case = EvidenceDocument(id="catalogue", provenance="case", title="Catalogue", text=_FIRST)
    evidence = ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=(case,))
    cited = {"ref": "S1", "stance": stance, "quote": _FIRST, "subject": "The Lantern Almanac", "value": "1921"}
    judge = _FakeWebJudge(script={"claim_truth": [_truth(0, suspect=False, cited=[cited])]})
    port = _FakeWebSearch()
    template = _template()
    kind = _FakeWebKind(evidence)

    _, trace = await run_one_result(
        toyhost_host(),
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id="run-1",
        scope_id="u",
        judge_service=_service(judge),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}, web_search=port, external_rates=_RATES),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )

    (claim,) = trace.claim_grades[_CRITERION].claims
    assert claim.graded_against == "case" and claim.web == "none" and claim.nominated_by is None
    assert claim.verdict == ("supported" if stance == "supports" else "contradicted")
    assert port.queries == [] and "claim_web_queries" not in judge.asked, "the web never overrules case evidence"


async def test_the_web_step_is_shown_neither_the_candidates_documents_nor_any_case_evidence() -> None:
    judge = _FakeWebJudge()
    case = EvidenceDocument(id="rules", provenance="case", title="Reading room rules", text=_ROOM_RULES)
    evidence = ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=(_NOTES, case))

    await _cell(judge, _FakeWebSearch(), evidence=evidence)

    truth = [prompt for name, prompt in zip(judge.asked, judge.prompts, strict=True) if name == "claim_truth"]
    assert truth and _ROOM_RULES in truth[0], "the control: the truth step was shown the case document"
    web_prompts = [prompt for name, prompt in zip(judge.asked, judge.prompts, strict=True) if "web" in name]
    assert web_prompts and not any(_NOTES.text in prompt or _ROOM_RULES in prompt for prompt in web_prompts)


async def test_searches_are_billed_to_the_judge_role_in_a_row_of_their_own_per_query() -> None:
    queries = json.dumps({"queries": ["Lantern Almanac first printed", "Lantern Almanac 1921"]})
    judge = _FakeWebJudge(script={"claim_web_queries": [queries]})

    result, _ = await _cell(judge, _FakeWebSearch())

    (row,) = [row for row in result.usage if row.role == "judge" and row.provider == _PORT]
    assert (row.provider_unit, row.provider_units, row.call_count) == ("query", 2, 2)
    assert (row.model, row.prompt_tokens, row.completion_tokens) == (None, None, None), "a search spends no tokens"
    assert row.cost_usd == pytest.approx(0.02) and row.price_source == f"{_PORT}:configured_rate"
    assert not [row for row in result.usage if row.role == "external"], "a search is judging, never production spend"
    (calls,) = [row for row in result.usage if row.role == "judge" and row.provider is None]
    assert calls.call_count == 4


async def test_a_port_with_no_declared_rate_bills_each_query_at_its_own_ceiling_named_as_such() -> None:
    result, _ = await _cell(_FakeWebJudge(), _FakeWebSearch(usd_per_query=0.005), rates=None)

    (row,) = [row for row in result.usage if row.provider == _PORT]
    assert (row.cost_usd, row.price_source) == (0.005, f"{_PORT}:price_ceiling")


async def test_a_decisive_answer_whose_quote_is_in_no_window_is_asked_again_and_the_fixed_answer_stands() -> None:
    invented = _verdict(quote="The almanac first appeared in 1921, the press says.")
    judge = _FakeWebJudge(script={"claim_web_verdict": [invented]})

    _, trace = await _cell(judge, _FakeWebSearch())

    assert judge.asked[-2:] == ["claim_web_verdict", "claim_web_repair"]
    grade = trace.claim_grades[_CRITERION]
    (claim,) = grade.claims
    assert claim.verdict == "true_ungrounded" and grade.calls[claim.decided_by or 0].role == "repair"


#: A case document that supports the second claim, so the cell holds a verdict case evidence decided.
_BINDING = EvidenceDocument(id="binding", provenance="case", title="Binding notes", text=_SECOND)

#: Two claims: the first nominated for the web, the second supported by case evidence.
_MIXED = {
    "claim_listing": [_listing(_listed(0, _FIRST), _listed(1, _SECOND))],
    "claim_truth": [
        _truth(0),
        _truth(
            1,
            suspect=False,
            cited=[
                {"ref": "S1", "stance": "supports", "quote": _SECOND, "subject": "Its pages", "value": "green cloth"}
            ],
        ),
    ],
}

_UNQUOTED = _verdict("false", quote="It was first printed in 1931.", value="1931")


@pytest.mark.parametrize(
    ("script", "port", "loss"),
    [
        pytest.param(
            {"claim_web_verdict": [_UNQUOTED], "claim_web_repair": [_UNQUOTED]},
            None,
            "repair_budget_spent",
            id="unquoted",
        ),
        pytest.param({"claim_web_queries": ["not json"]}, None, "unreadable", id="unreadable"),
        pytest.param(
            {"claim_web_verdict": [ConnectionError("the provider is unavailable")]},
            None,
            "provider_error",
            id="provider",
        ),
        pytest.param({}, ConnectionError("the search is unavailable"), "port_error", id="port"),
    ],
)
async def test_a_failed_web_step_is_its_own_loss_and_every_verdict_case_evidence_decided_stands(
    script: dict[str, list[Any]], port: BaseException | None, loss: str
) -> None:
    # A failure here must never cost a cell its case-decided truth: how many claims the judge doubts differs by arm.
    judge = _FakeWebJudge(script={**_MIXED, **script})
    evidence = ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=(_BINDING,))

    result, trace = await _cell(judge, _FakeWebSearch(failure=port), evidence=evidence)

    assert result.claim_failures == {}, "a web step's failure is never the criterion's loss"
    tally = result.claim_tallies[_CRITERION]
    assert tally.web_loss == loss
    grade = trace.claim_grades[_CRITERION]
    assert grade.web_loss == loss and grade.web == ()
    doubted, supported = grade.claims
    assert (doubted.web, doubted.verdict, doubted.graded_against) == ("lost", "unverifiable", None)
    assert (supported.verdict, supported.graded_against) == ("supported", "case"), "case evidence's verdict stands"
    (counts,) = tally.by_judge
    assert (counts.asserted_contradicted, counts.unverifiable, counts.web_skipped) == (0, 1, 0), "never read as false"
    assert claim_tally(grade, evidence, grounded_in=None).web_loss == loss, "recomputed from the stored grade, it stays"


async def test_claims_past_the_cells_web_budget_read_skipped_cap_and_unverifiable_never_false() -> None:
    listing = _listing(_listed(0, _FIRST), _listed(1, _SECOND))
    judge = _FakeWebJudge(script={"claim_listing": [listing], "claim_truth": [_truth(0), _truth(1)]})
    template = _template(_criterion(web=_WEB.model_copy(update={"max_checked_per_cell": 1})))

    result, trace = await _cell(judge, _FakeWebSearch(), template=template)

    first, second = trace.claim_grades[_CRITERION].claims
    assert (first.web, first.verdict) == ("checked", "true_ungrounded")
    assert (second.web, second.verdict, second.graded_against) == ("skipped_cap", "unverifiable", None)
    readings = claim_readings(_CRITERION, result.claim_tallies[_CRITERION], grounded=False)
    assert readings[f"{_CRITERION}_web_skipped"] == 1 and readings[f"{_CRITERION}_false"] == 0
    assert judge.asked.count("claim_web_queries") == 1


async def test_a_restatement_takes_its_originals_web_outcome_from_the_one_check_made() -> None:
    listing = _listing(_listed(0, _FIRST), {**_listed(2, _FIRST), "excerpt": _FIRST.rstrip("."), "restates": 0})
    contradiction = _verdict("false", quote="The Lantern Almanac was first printed in 1931.", value="1931")
    page = WebPage(url="https://example.org/almanac", title="", text="The Lantern Almanac was first printed in 1931.")
    judge = _FakeWebJudge(script={"claim_listing": [listing], "claim_web_verdict": [contradiction]})
    port = _FakeWebSearch(pages=(page,))

    result, trace = await _cell(judge, port)

    grade = trace.claim_grades[_CRITERION]
    original, again = grade.claims
    assert (again.web, again.verdict, again.graded_against) == ("checked", "contradicted", "web")
    assert again.decided_by == original.decided_by
    assert [record.claim for record in grade.web] == [0], "one check, the original's, which the restatement shares"
    assert len(port.queries) == 1
    (counts,) = result.claim_tallies[_CRITERION].by_judge
    assert counts.asserted_contradicted == 2, "a restatement is counted as heard, with its original's verdict"


async def test_a_port_failing_a_search_is_never_retried_and_keeps_the_checks_already_decided() -> None:
    listing = _listing(_listed(0, _FIRST), _listed(1, _SECOND))
    queries = [json.dumps({"queries": ["first"]}), json.dumps({"queries": ["second"]})]
    judge = _FakeWebJudge(
        script={"claim_listing": [listing], "claim_truth": [_truth(0), _truth(1)], "claim_web_queries": queries}
    )

    @dataclass
    class _FailsSecond(_FakeWebSearch):  # parity-exempt: a scripted search whose second query fails, for one test
        async def search(self, query: str) -> WebSearchResult:
            if query == "second":
                self.queries.append(query)
                raise ConnectionError("the search is unavailable")
            return await super().search(query)

    port = _FailsSecond()

    result, trace = await _cell(judge, port)

    assert port.queries == ["first", "second"], "the engine never retries a port's search"
    first, second = trace.claim_grades[_CRITERION].claims
    assert (first.web, first.verdict) == ("checked", "true_ungrounded"), "the check already decided stands"
    assert (second.web, second.verdict) == ("lost", "unverifiable")
    assert result.claim_tallies[_CRITERION].web_loss == "port_error" and result.claim_failures == {}
    (row,) = [row for row in result.usage if row.provider == _PORT]
    assert row.call_count == 2, "a search sent is billed, answered or not"


async def test_a_replay_miss_from_the_port_fails_the_run_naming_the_search(tmp_path: Path) -> None:
    recording = tmp_path / "search.jsonl"
    recorder = RecordedWebSearch.record(_FakeWebSearch(), recording)
    await recorder.search("something else entirely")

    with pytest.raises(ReplayMiss, match="Lantern Almanac first printed"):
        await _cell(_FakeWebJudge(), RecordedWebSearch.replay(recording))


async def test_a_recorded_search_replays_by_identity_and_query_and_its_grades_say_they_were_replayed(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "search.jsonl"
    live = _FakeWebSearch()
    _, recorded = await _cell(_FakeWebJudge(), RecordedWebSearch.record(live, recording))
    replaying = RecordedWebSearch.replay(recording)

    result, replayed = await _cell(_FakeWebJudge(), replaying)

    assert replaying.identity == _PORT and replaying.price_ceiling() == 0.0
    assert replayed.claim_grades[_CRITERION].web == recorded.claim_grades[_CRITERION].web
    assert replayed.claim_grades[_CRITERION].web_replayed and not recorded.claim_grades[_CRITERION].web_replayed
    assert len(live.queries) == 1, "the replay searched nothing"
    (row,) = [row for row in result.usage if row.provider == _PORT]
    assert row.cost_usd == 0.0, "a replay spends nothing, whatever rate the run declares"


async def test_recording_a_query_the_file_holds_answers_from_the_file(tmp_path: Path) -> None:
    live = _FakeWebSearch()
    recorder = RecordedWebSearch.record(live, tmp_path / "search.jsonl")

    first = await recorder.search("q")
    again = await recorder.search("q")

    assert live.queries == ["q"] and again.pages == first.pages and again.replayed and not first.replayed


async def test_a_recording_takes_one_port_and_refuses_a_second_onto_its_file(tmp_path: Path) -> None:
    recording = tmp_path / "search.jsonl"
    await RecordedWebSearch.record(_FakeWebSearch(), recording).search("q")

    with pytest.raises(RecordingRefused, match="a file of its own"):
        RecordedWebSearch.record(_FakeWebSearch(identity="another-search"), recording)
    with pytest.raises(RecordingRefused, match="does not exist"):
        RecordedWebSearch.replay(tmp_path / "missing.jsonl")


async def test_a_criterion_with_a_web_budget_and_no_port_is_refused_before_any_call() -> None:
    judge = _FakeWebJudge()

    with pytest.raises(ValueError, match="has no web search"):
        await _cell(judge, None)
    assert judge.asked == []


async def _run(port: Any, *, rates: ExternalRateTable | None, cap_usd: float | None) -> Any:
    judge = _FakeWebJudge()
    template = _template()
    host = toyhost_host()
    case = EvalTestCase(template_id=template.id, scope_id="u")
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)
    cap = EvalRunCostCap(run.id, cap_usd, enabled=True) if cap_usd is not None else None
    kind = _FakeWebKind()
    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=_service(judge),
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}, web_search=port, external_rates=rates),
            **({"budget_gate": cap.check, "on_cost": cap.record} if cap is not None else {}),
        )
    except BudgetStoppedError:
        pass
    return judge, off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)


async def test_an_unpriced_port_under_an_enforcing_cap_is_refused_at_launch_unless_a_rate_prices_it() -> None:
    unpriced = _FakeWebSearch(usd_per_query=None)

    with pytest.raises(ValueError, match="states no price_ceiling"):
        await _run(unpriced, rates=None, cap_usd=5.0)
    judge, (result,) = await _run(unpriced, rates=_RATES, cap_usd=5.0)
    assert "claim_web_verdict" in judge.asked and result.claim_tallies
    _, (uncapped,) = await _run(unpriced, rates=None, cap_usd=None)
    (row,) = [row for row in uncapped.usage if row.provider == _PORT]
    assert row.cost_usd is None, "with no cap, an unpriced port's searches are counted and unpriced"


async def test_a_cost_cap_refusing_the_web_step_leaves_its_claims_unchecked_and_truth_standing() -> None:
    # The judge's calls are unpriced, so each is sent only while the run is within its $0.003 cap; each costs $0.002.
    # The listing and the truth call go out; the call writing the searches is refused.
    judge, (result,) = await _run(_FakeWebSearch(usd_per_query=0.0), rates=None, cap_usd=0.003)

    assert judge.asked == ["claim_listing", "claim_truth"]
    assert result.claim_failures == {}, "a web step the cap refused is no loss of the criterion"
    (counts,) = result.claim_tallies[_CRITERION].by_judge
    assert (counts.web_skipped, counts.unverifiable, counts.asserted_contradicted) == (1, 1, 0)


async def test_a_deadline_in_the_web_step_keeps_truth_and_reads_the_unchecked_claims_skipped() -> None:
    judge = _FakeWebJudge(script={"claim_web_queries": ["hang"]})
    template = _template()
    host = toyhost_host()
    case = EvalTestCase(template_id=template.id, scope_id="u")
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)
    kind = _FakeWebKind()

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=[case],
        judge_service=_service(judge),
        options=RunnerOptions(
            candidate_kinds={_KIND: lambda _cell: kind},
            web_search=_FakeWebSearch(),
            external_rates=_RATES,
            cell_timeout_s=0.5,
        ),
    )

    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    assert result.termination == "cell_timeout" and result.claim_failures == {}
    (counts,) = result.claim_tallies[_CRITERION].by_judge
    assert (counts.web_skipped, counts.unverifiable) == (1, 1)


def test_the_web_step_is_priced_per_claim_checked_with_its_own_repair_budget() -> None:
    judge = _FakeWebJudge(usd_per_call=1.0)
    criterion = _criterion(web=_WEB.model_copy(update={"max_checked_per_cell": 1}))
    # Three units, three attempts each, four repairs per step: the extraction 3 + 4 and the truth step 3 x 3 + 4.
    # The web step: per claim checked (one), 3 query calls + 2 searches at $0.5 + 3 reading calls; then 4 repairs.
    assert criterion_ceiling(criterion, _EVIDENCE, key=None, client=judge, query_usd=0.5) == 7 + 13 + 7 + 4
    assert criterion_ceiling(criterion, _EVIDENCE, key=None, client=judge) is None, "an unpriced search bounds nothing"


def test_a_criterions_fingerprint_moves_with_its_web_budget() -> None:
    def fingerprint(criterion: ClaimCriterion) -> str:
        return criterion_fingerprint(criterion, *criterion_steps(criterion, _FakeWebJudge()))

    base = fingerprint(_criterion())
    assert fingerprint(_criterion(web=None)) != base
    assert fingerprint(_criterion(web=_WEB.model_copy(update={"results_per_query": 1}))) != base


def _graded(**changes: Any) -> GradedClaim:
    fields: dict[str, Any] = {
        "segment": 0,
        "unit": 0,
        "claim": _FIRST,
        "excerpt": _FIRST,
        "start": 0,
        "end": len(_FIRST),
        "scope": "world",
        "specific": True,
        "hedged": False,
        "attributed": False,
        "suspect": True,
        "nominated_by": "judge",
        "verdict": "contradicted",
        "graded_against": "web",
        "web": "checked",
        "decided_by": 0,
    }
    fields.update(changes)
    return GradedClaim(**fields)


def _grade(claims: tuple[GradedClaim, ...], web: tuple[WebCheckRecord, ...]) -> ClaimGrade:
    return ClaimGrade(
        criterion_fingerprint="a" * 64,
        claim_evidence_digest="b" * 64,
        calls=(ClaimCall(role="web", served_model=_JUDGE, judge_temperature=0.0),),
        claims=claims,
        web=web,
    )


_RECORD = WebCheckRecord(
    claim=0,
    queries=("q",),
    windows=(WebWindow(url="https://example.org/a", text="It was first printed in 1931."),),
    outcome="false",
    deciding_url="https://example.org/a",
    quote="It was first printed in 1931.",
    value_span={"start": 24, "end": 28},
)


def test_a_grade_holds_one_web_record_per_check_which_each_restatement_shares() -> None:
    again = _graded(segment=0, restates=0)

    assert _grade((_graded(), again), (_RECORD,)).claims[1].web == "checked"
    with pytest.raises(ValidationError, match="takes its original's web check"):
        _grade((_graded(), _graded(restates=0, web="none", verdict="unverifiable", graded_against=None)), (_RECORD,))
    with pytest.raises(ValidationError, match="its claims read checked"):
        _grade((_graded(), again), (_RECORD, _RECORD.model_copy(update={"claim": 1})))
    with pytest.raises(ValidationError, match="web check found 'false'"):
        _grade((_graded(), _graded(restates=0, verdict="true_ungrounded")), (_RECORD,))


async def test_a_search_whose_result_reports_its_cost_is_billed_at_that_over_any_declared_rate() -> None:
    result, _ = await _cell(_FakeWebJudge(), _FakeWebSearch(reported_usd=0.003))

    (row,) = [row for row in result.usage if row.provider == _PORT]
    assert (row.cost_usd, row.price_source) == (0.003, f"{_PORT}:reported")


async def test_only_the_first_results_per_query_of_a_search_are_read() -> None:
    pages = tuple(
        WebPage(url=f"https://example.org/{n}", title="", text=f"The Lantern Almanac, page {n}.") for n in range(3)
    )
    template = _template(_criterion(web=_WEB.model_copy(update={"results_per_query": 2})))

    _, trace = await _cell(
        _FakeWebJudge(script={"claim_web_verdict": [_verdict("inconclusive")]}),
        _FakeWebSearch(pages=pages),
        template=template,
    )

    (record,) = trace.claim_grades[_CRITERION].web
    assert record.urls == ("https://example.org/0", "https://example.org/1")
    assert {window.url for window in record.windows} == {"https://example.org/0", "https://example.org/1"}


async def test_a_page_two_searches_return_is_read_once() -> None:
    queries = json.dumps({"queries": ["Lantern Almanac first printed", "Lantern Almanac 1921"]})

    _, trace = await _cell(_FakeWebJudge(script={"claim_web_queries": [queries]}), _FakeWebSearch())

    (record,) = trace.claim_grades[_CRITERION].web
    assert record.urls == (_PAGE.url, _PAGE.url), "every result is recorded, in order"
    assert [window.url for window in record.windows] == [_PAGE.url], "and the page is read once"


async def test_at_most_page_chars_of_a_page_is_read_in_windows_cut_around_the_claim() -> None:
    filler = "The reading room opens at nine and closes at five. " * 40
    page = WebPage(url="https://example.org/long", title="", text=filler + _PAGE.text + " " + filler)
    budget = _WEB.model_copy(update={"page_chars": 300, "window_chars": 100})

    _, trace = await _cell(
        _FakeWebJudge(script={"claim_web_verdict": [_verdict("inconclusive")]}),
        _FakeWebSearch(pages=(page,)),
        template=_template(_criterion(web=budget)),
    )

    (record,) = trace.claim_grades[_CRITERION].web
    assert len(page.text) > 3 * 300 and sum(len(window.text) for window in record.windows) <= 300
    assert any("1921" in window.text for window in record.windows), "the windows read are those the claim names"


async def test_a_reply_writing_more_searches_than_the_budget_is_asked_again() -> None:
    too_many = json.dumps({"queries": ["a", "b", "c"]})
    judge = _FakeWebJudge(script={"claim_web_queries": [too_many, _ANSWERS["claim_web_queries"]]})
    port = _FakeWebSearch()

    await _cell(judge, port)

    assert judge.asked.count("claim_web_queries") == 2 and port.queries == ["Lantern Almanac first printed"]


async def test_a_witnessed_cell_refuses_a_web_budget_saying_it_is_not_supported_there() -> None:
    template = _template()

    with pytest.raises(ValueError, match="not supported on witnessed cells yet, so declare web None"):
        await judge_witnessed_output(
            template=template,
            test_case=EvalTestCase(template_id=template.id, scope_id="u"),
            output=CandidateOutput(output=[{"answer": _FIRST}], claim_evidence=_EVIDENCE),
            goal_outcomes=[],
            judged_artifact=JudgedArtifact.UNJUDGED,
            judge_service=_service(_FakeWebJudge()),
            concurrency=1,
            eval_run_id="run-1",
        )


async def test_a_port_whose_ceiling_is_no_amount_is_refused_at_launch_with_no_cap() -> None:
    with pytest.raises(ValueError, match="a ceiling is a finite amount"):
        await _run(_FakeWebSearch(usd_per_query=-1.0), rates=None, cap_usd=None)
