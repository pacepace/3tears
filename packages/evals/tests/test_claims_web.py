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
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.claims.grading import (
    claim_payload_bytes,
    criterion_ceiling,
    criterion_fingerprint,
    criterion_steps,
)
from threetears.evals.kernel.claims.readings import claim_readings
from threetears.evals.kernel.host import ReplayMiss
from threetears.evals.kernel.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.kernel.usage_capture import ExternalRateTable
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.runner import RunnerOptions, execute_run, run_one_result
from threetears.evals.kernel.claims.readings import claim_tally
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.lifecycle import rejudge_result
from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.ops import OpsHost
from threetears.evals.run import regrade
from threetears.evals.run.runner import StoredCell, StoredCells, judge_witnessed_output
from threetears.evals.run.witnessed import WitnessedJudging, record_witnessed_cell, stamp_witnessed_judge
from threetears.evals.schema.completion import CompletionProtocolDefect
from threetears.evals.schema.job_lease import JobLease
from threetears.evals.schema import Segment, WebPage, WebSearchResult
from threetears.evals.schema.models import (
    CLAIM_TRACE_BYTES,
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


def _listed(segment: int, text: str, *, restates: str | None = None) -> dict[str, Any]:
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

    ``"hang"`` never answers; ``"cut"`` answers cut short at the output cap; ``"defective"`` answers with its
    ``sent_settings`` the wrong type, the host's protocol defect. ``reply_usd`` prices one schema's replies otherwise
    than the rest. Records each call's schema name and the prompt it was sent.
    """

    script: dict[str, list[Any]] = field(default_factory=dict)
    usd_per_call: float | None = None
    reply_usd: dict[str, float] = field(default_factory=dict)
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
        cost = self.reply_usd.get(name, _Reply.cost_usd)
        if answer == "cut":
            return _Reply(content='{"outcome": "tr', stop_reason="max_tokens", cost_usd=cost)
        if answer == "defective":
            return _Reply(content=_ANSWERS[name], sent_settings="fast", cost_usd=cost)
        return _Reply(content=answer, cost_usd=cost)

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
        pytest.param({"claim_web_verdict": ["cut"]}, None, "cut_short", id="cut-short"),
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


def _no_window(ref: str) -> str:
    """What the web step names to the model when an answer cites a window it was not shown."""
    return f"it cites {ref}, which is no window shown here; cite the refs listed under Windows"


async def test_a_web_step_its_repairs_lose_keeps_its_calls_and_the_failure_each_repair_named() -> None:
    # Each repair's answer fails anew: the record keeps the fault each repair was asked to fix, and the last
    # answer's own fault as the failure that spent the budget.
    judge = _FakeWebJudge(
        script={
            "claim_web_verdict": [_verdict(ref="W9")],
            "claim_web_repair": [_verdict(ref="W8"), _verdict(ref="W7")],
        }
    )

    result, trace = await _cell(judge, _FakeWebSearch())

    assert result.claim_failures == {} and result.claim_tallies[_CRITERION].web_loss == "repair_budget_spent"
    record = trace.claim_losses[_CRITERION]
    assert [(call.step, call.role, call.reason) for call in record.calls] == [
        ("extract", "extract", None),
        ("truth", "truth", None),
        ("web", "web", None),
        ("web", "web", None),
        ("web", "repair", _no_window("W9")),
        ("web", "repair", _no_window("W8")),
    ], "each repair names the failure it was asked to fix"
    assert record.failure == (
        "web step: the web check of claim 0 could not be repaired within the step's repair budget: " + _no_window("W7")
    ), "the failure kept is the last answer's own, which no call named"
    assert [ClaimCall.model_validate(call.model_dump(exclude={"step", "reason"})) for call in record.calls] == list(
        trace.claim_grades[_CRITERION].calls
    ), "the calls kept are the grade's own, in its order"


async def test_a_search_port_failing_keeps_the_web_steps_calls_and_never_the_ports_message() -> None:
    judge = _FakeWebJudge()

    result, trace = await _cell(judge, _FakeWebSearch(failure=ConnectionError("the search is unavailable")))

    assert result.claim_tallies[_CRITERION].web_loss == "port_error"
    record = trace.claim_losses[_CRITERION]
    assert [call.step for call in record.calls][-1] == "web", "the call that wrote the search is kept"
    assert record.failure is None and "unavailable" not in record.model_dump_json(), "the port's message is its own"


async def test_web_replies_asked_again_after_they_could_not_be_read_keep_why_they_could_not() -> None:
    judge = _FakeWebJudge(
        script={
            "claim_web_queries": ["not json", _ANSWERS["claim_web_queries"]],
            "claim_web_verdict": ["not json", _verdict(ref="W9")],
            "claim_web_repair": [_verdict(ref="W9")],
        }
    )

    result, trace = await _cell(judge, _FakeWebSearch())

    assert result.claim_tallies[_CRITERION].web_loss == "repair_budget_spent"
    web = [call for call in trace.claim_losses[_CRITERION].calls if call.role == "web"]
    assert len(web) == 4 and web[0].reason is None and web[2].reason is None, "each first ask names nothing"
    assert web[1].reason and web[3].reason, "the searches and the reading asked again each name why"


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
    listing = _listing(_listed(0, _FIRST), {**_listed(2, _FIRST), "restates": _FIRST.rstrip(".")})
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
    recording = tmp_path / "search.jsonl"
    recorder = RecordedWebSearch.record(live, recording)

    first = await recorder.search("q")
    again = await recorder.search("q")
    later = await RecordedWebSearch.record(live, recording).search("q")

    assert live.queries == ["q"], "a query the recording holds is never searched live again"
    assert again.pages == first.pages == later.pages and (again.usd, later.usd) == (0.0, 0.0)
    assert not first.replayed and not again.replayed, "a repeat of this port's own live search is no replay"
    assert later.replayed, "an answer from a recording made earlier is"


async def test_a_recording_run_that_repeats_a_query_never_marks_its_own_grades_replayed(tmp_path: Path) -> None:
    # Two claims whose checks write the same search: the second is answered from the recording this run just made.
    listing = _listing(_listed(0, _FIRST), _listed(1, _SECOND))
    judge = _FakeWebJudge(script={"claim_listing": [listing], "claim_truth": [_truth(0), _truth(1)]})
    live = _FakeWebSearch()

    _, trace = await _cell(judge, RecordedWebSearch.record(live, tmp_path / "search.jsonl"))

    grade = trace.claim_grades[_CRITERION]
    assert len(grade.web) == 2 and live.queries == ["Lantern Almanac first printed"], "the control: one live search"
    assert not grade.web_replayed, "the run searched live, and its grades say so"


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


async def _run(
    port: Any, *, rates: ExternalRateTable | None, cap_usd: float | None, judge: _FakeWebJudge | None = None
) -> Any:
    judge = judge or _FakeWebJudge()
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


async def _witnessed_judged(judge: _FakeWebJudge, port: Any, **options: Any) -> Any:
    """``judge_witnessed_output`` over one witnessed cell of the almanac answer, on ``port``."""
    template = _template()
    return await judge_witnessed_output(
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        output=CandidateOutput(output=[{"answer": _FIRST}], claim_evidence=_EVIDENCE),
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.UNJUDGED,
        judge_service=_service(judge),
        concurrency=1,
        eval_run_id="run-1",
        web_search=port,
        **options,
    )


async def test_a_witnessed_cell_checks_its_nominated_claims_on_the_web_search_it_is_handed() -> None:
    judge = _FakeWebJudge()
    port = _FakeWebSearch()

    judged, _ = await _witnessed_judged(judge, port, external_rates=_RATES)

    assert judged.claims is not None and judged.claims.failures == {}
    (counts,) = judged.claims.tallies[_CRITERION].by_judge
    assert (counts.true_ungrounded, counts.web_checked) == (1, 1) and port.queries == ["Lantern Almanac first printed"]
    (row,) = [row for row in judged.usage if row.provider == _PORT]
    assert (row.role, row.provider_unit, row.call_count, row.cost_usd) == ("judge", "query", 1, 0.01)


@pytest.mark.parametrize(
    ("port", "options", "refusal"),
    [
        pytest.param(None, {}, "has no web search", id="no-port"),
        pytest.param(
            _FakeWebSearch(usd_per_query=None), {"enforced": True}, "states no price_ceiling", id="unpriced-enforced"
        ),
        pytest.param(_FakeWebSearch(usd_per_query=-1.0), {}, "a ceiling is a finite amount", id="ceiling-no-amount"),
    ],
)
async def test_a_witnessed_cell_is_refused_by_the_runs_own_web_rule_before_any_call(
    port: Any, options: dict[str, Any], refusal: str
) -> None:
    judge = _FakeWebJudge()

    with pytest.raises(ValueError, match=refusal):
        await _witnessed_judged(judge, port, **options)

    assert judge.asked == []


async def test_an_unpriced_port_on_a_witnessed_cell_is_admitted_when_a_rate_prices_it_or_nothing_is_enforced() -> None:
    priced, _ = await _witnessed_judged(
        _FakeWebJudge(), _FakeWebSearch(usd_per_query=None), external_rates=_RATES, enforced=True
    )
    unenforced, _ = await _witnessed_judged(_FakeWebJudge(), _FakeWebSearch(usd_per_query=None))

    assert [row.cost_usd for row in priced.usage if row.provider == _PORT] == [0.01]
    assert [row.cost_usd for row in unenforced.usage if row.provider == _PORT] == [None], "counted, unpriced"


async def test_a_port_whose_ceiling_is_no_amount_is_refused_at_launch_with_no_cap() -> None:
    with pytest.raises(ValueError, match="a ceiling is a finite amount"):
        await _run(_FakeWebSearch(usd_per_query=-1.0), rates=None, cap_usd=None)


async def test_a_hold_overflow_inside_the_web_step_reads_skipped_cap_and_truth_stands() -> None:
    # The port says its search cost far more than its ceiling: the hold the criterion was priced at has no room left
    # for the reading call, which is the criterion's own hold refusing it, never a loss of the web step.
    judge = _FakeWebJudge(usd_per_call=0.001)

    judge, (result,) = await _run(_FakeWebSearch(reported_usd=5.0), rates=None, cap_usd=1000.0, judge=judge)

    assert "claim_web_queries" in judge.asked and "claim_web_verdict" not in judge.asked
    assert result.claim_failures == {}
    tally = result.claim_tallies[_CRITERION]
    (counts,) = tally.by_judge
    assert tally.web_loss is None and (counts.web_skipped, counts.unverifiable) == (1, 1), "a budget stop, no loss"


async def _stored_web_cell(
    first: _FakeWebJudge, again: _FakeWebJudge, port: Any, *, template: EvalTemplate | None = None
) -> tuple[Any, Any]:
    """One cell graded by ``first`` on ``port`` and stored under a finished run, on a host whose judge is ``again``."""
    host = toyhost_host(clients=lambda role, model, *, temperature=None: again)
    template = template or _template()
    case = EvalTestCase(template_id=template.id, scope_id="u")
    off_loop(host.storage.save_template, template)
    off_loop(host.storage.save_test_case, case)
    run = make_eval_run(
        scope_id="u",
        template_id=template.id,
        candidate_kind=_KIND,
        test_case_ids=[case.id],
        status="completed",
        judge_model=_JUDGE,
        effective_judges={},
        judge_asked_settings=JUDGE_REQUEST_SETTINGS,
    )
    off_loop(host.storage.save_eval_run, run)
    kind = _FakeWebKind()
    outcome = await run_one_result(
        host,
        template=template,
        test_case=case,
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id=run.id,
        scope_id="u",
        judge_service=_service(first),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}, web_search=port, external_rates=_RATES),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )
    off_loop(host.storage.save_eval_result, outcome.result, outcome.trace)
    return host, outcome.result


_LOST_FIRST = {"claim_listing": [ConnectionError("the provider is unavailable")]}


async def test_a_rejudge_grades_a_lost_web_criterion_again_whole_searching_through_the_port_it_is_handed() -> None:
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), _FakeWebJudge(), _FakeWebSearch())
    assert result.claim_failures == {_CRITERION: "provider_error"}
    port = _FakeWebSearch()

    rejudged = await rejudge_result(host, result.id, "u", web_search=port)

    assert port.queries == ["Lantern Almanac first printed"], "the web step searched live through the port"
    assert rejudged.claim_failures == {} and rejudged.claim_tallies[_CRITERION].by_judge[0].true_ungrounded == 1
    stored = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert stored is not None and stored.claim_grades[_CRITERION].web[0].urls == (_PAGE.url,)
    (rescore,) = rejudged.claim_rescores
    (searched,) = [row for row in rescore.usage if row.provider == _PORT]
    assert (searched.provider_unit, searched.call_count) == ("query", 1), "the re-judge billed its search"


async def test_a_rejudge_of_a_lost_web_criterion_with_no_port_is_refused_by_name_before_any_call() -> None:
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), again, _FakeWebSearch())

    with pytest.raises(ValidationFailedError, match=f"claim criteria '{_CRITERION}'.*declare a web budget"):
        await rejudge_result(host, result.id, "u")

    assert again.asked == [], "refused before any call"
    stored = off_loop(host.storage.load_eval_result, result.id, "u")
    assert stored is not None and stored.claim_failures == {_CRITERION: "provider_error"}, "never read as a new loss"


async def test_a_regrade_reports_a_lost_web_criterion_and_applied_grades_it_again_on_the_port_it_is_handed() -> None:
    """The re-grade door names the claim criteria a result lost and, applied, grades them again through the re-judge,
    its web step searching live on the port the caller hands in, billed at the run's rates."""
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), again, _FakeWebSearch())

    reported = await regrade(host, result.id, "u", checks=False)
    assert (reported.claim_criteria, reported.applied, again.asked) == ([_CRITERION], False, [])

    port = _FakeWebSearch()
    applied = await regrade(host, result.id, "u", checks=False, apply=True, web_search=port, external_rates=_RATES)

    assert applied.claim_criteria == [_CRITERION] and applied.claim_cost_usd is not None
    assert port.queries == ["Lantern Almanac first printed"], "the web step searched live through the port"
    stored = off_loop(host.storage.load_eval_result, result.id, "u")
    assert stored is not None and stored.claim_failures == {} and stored.claim_tallies[_CRITERION].by_judge


async def test_a_regrade_of_a_lost_web_criterion_with_no_port_is_refused_by_name_before_anything_is_written() -> None:
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), again, _FakeWebSearch())
    before, etag = off_loop(host.storage.load_eval_result_with_etag, result.id, "u")

    with pytest.raises(ValidationFailedError, match=f"claim criteria '{_CRITERION}'.*declare a web budget"):
        await regrade(host, result.id, "u", apply=True)

    assert again.asked == [], "refused before any call"
    after, etag_after = off_loop(host.storage.load_eval_result_with_etag, result.id, "u")
    assert (after, etag_after) == (before, etag), "nothing was written, the re-graded checks included"


async def test_the_regrade_action_names_a_lost_claim_criterion_and_refuses_to_grade_it_with_no_model_call() -> None:
    """The action makes no model call: it reports the lost criterion, refuses ``apply`` naming it, and with
    ``judges=false`` stores the re-graded checks alone, asking and searching nothing."""
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), again, _FakeWebSearch())
    before = off_loop(host.storage.load_eval_result, result.id, "u")
    (tool, *_) = eval_catalogue().mount_all(standard_tools())
    caller = Caller(scope_id="u", identity="agent:regrade")

    async def call(**params: Any) -> Any:
        return await tool.call(
            {"action": "result_regrade", "result_id": result.id, **params}, host=OpsHost.reading(host), caller=caller
        )

    reported, refused, checks_only = await call(), await call(apply=True), await call(apply=True, judges=False)

    assert not reported.is_error and reported.structured is not None
    assert reported.structured["claim_criteria"] == [_CRITERION]
    assert refused.is_error and _CRITERION in refused.text and "asking its judge again spends" in refused.text
    assert not checks_only.is_error and checks_only.structured is not None and checks_only.structured["applied"]
    assert again.asked == []
    after = off_loop(host.storage.load_eval_result, result.id, "u")
    assert after is not None and before is not None and after.claim_failures == before.claim_failures


async def _pending_web_cell() -> tuple[Any, Any, _FakeWebJudge]:
    """A web criterion graded and its grade stored, its row then left pending, as a row written before its trace is."""
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(_FakeWebJudge(), again, _FakeWebSearch())
    assert result.claim_tallies and not result.claim_failures
    pending = result.model_copy(update={"claim_tallies": {}, "claim_failures": {_CRITERION: "trace_pending"}})
    _, etag = off_loop(host.storage.load_eval_result_with_etag, result.id, "u")
    off_loop(partial(host.storage.replace_eval_result, pending, if_match=etag))
    return host, result, again


async def test_a_pending_criterion_is_never_due_a_paid_regrade_and_is_settled_free_with_no_port() -> None:
    host, result, again = await _pending_web_cell()

    reported = await regrade(host, result.id, "u", checks=False)
    assert (reported.claim_criteria, reported.claims_pending) == ([], [_CRITERION]), "pending is no loss to pay for"

    applied = await regrade(host, result.id, "u", checks=False, apply=True)

    assert applied.claim_criteria == [] and again.asked == [], "settled with no call and no port"
    stored = off_loop(host.storage.load_eval_result, result.id, "u")
    assert stored is not None and stored.claim_failures == {}
    assert stored.claim_tallies[_CRITERION] == result.claim_tallies[_CRITERION], "recomputed from the stored grade"


async def test_the_regrade_action_applies_over_a_pending_criterion_without_refusing_it() -> None:
    host, result, again = await _pending_web_cell()
    (tool, *_) = eval_catalogue().mount_all(standard_tools())

    applied = await tool.call(
        {"action": "result_regrade", "result_id": result.id, "apply": True},
        host=OpsHost.reading(host),
        caller=Caller(scope_id="u", identity="agent:regrade"),
    )

    assert not applied.is_error and applied.structured is not None and applied.structured["claim_criteria"] == []
    assert again.asked == []
    stored = off_loop(host.storage.load_eval_result, result.id, "u")
    assert stored is not None and stored.claim_failures == {}


async def test_a_take_up_grades_the_cells_it_runs_on_the_runs_own_web_search() -> None:
    host, earlier = await _stored_web_cell(_FakeWebJudge(), _FakeWebJudge(), _FakeWebSearch())
    template = off_loop(host.storage.load_template, _template().id, "u") or _template()
    second = EvalTestCase(template_id=template.id, scope_id="u")
    off_loop(host.storage.save_test_case, second)
    stored_run = off_loop(host.storage.load_eval_run, earlier.eval_run_id, "u")
    assert stored_run is not None
    taken_up = stored_run.model_copy(
        update={
            "status": "running",
            "test_case_ids": [earlier.test_case_id, second.id],
            "lease": JobLease(holder="this-process", expires_at="2099-01-01T00:00:00+00:00", epoch=1),
        }
    )
    off_loop(host.storage.save_eval_run, taken_up)
    (first_case,) = [c for c in off_loop(host.storage.query_test_cases, "u") if c.id == earlier.test_case_id]
    row = off_loop(host.storage.load_eval_result, earlier.id, "u")
    assert row is not None
    port = _FakeWebSearch()
    kind = _FakeWebKind()

    await execute_run(
        host,
        run=taken_up,
        template=template,
        test_cases=[first_case, second],
        judge_service=_service(_FakeWebJudge()),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}, web_search=port, external_rates=_RATES),
        stored=StoredCells(done=(StoredCell.from_result(row),)),
    )

    assert port.queries == ["Lantern Almanac first printed"], "only the cell the take-up ran searched, on its port"
    rows = off_loop(host.storage.query_eval_results_by_run, taken_up.id, "u")
    (ran,) = [r for r in rows if r.test_case_id == second.id]
    assert ran.claim_failures == {} and ran.claim_tallies[_CRITERION].by_judge[0].web_checked == 1


async def test_a_rejudge_bills_an_unreported_search_in_the_runs_order_its_declared_rate_before_the_ceiling() -> None:
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), _FakeWebJudge(), _FakeWebSearch())

    rejudged = await rejudge_result(
        host, result.id, "u", web_search=_FakeWebSearch(usd_per_query=0.5), external_rates=_RATES
    )

    (searched,) = [row for row in rejudged.claim_rescores[0].usage if row.provider == _PORT]
    assert (searched.cost_usd, searched.price_source) == (0.01, f"{_PORT}:configured_rate")


# =============================================================================
# A witnessed cell's web step, recorded through record_witnessed_cell
# =============================================================================


async def _witnessed_cell(
    judge: _FakeWebJudge,
    port: Any,
    *,
    template: EvalTemplate | None = None,
    evidence: ClaimEvidence = _EVIDENCE,
    judging: WitnessedJudging | None = None,
) -> tuple[Any, Any]:
    """One witnessed cell of the almanac answer recorded on a judged witnessed run, its claims checked on ``port``."""
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    template = template or _template()
    off_loop(host.storage.save_template, template)
    case = EvalTestCase(id="visit-1", scope_id="u", template_id=None)
    off_loop(host.storage.save_test_case, case)
    run = stamp_witnessed_judge(
        host,
        make_eval_run(
            id="run-witnessed",
            scope_id="u",
            template_id=None,
            candidate_kind=_KIND,
            test_case_ids=[case.id],
            apparatus_provenance="witnessed",
        ),
        template,
        judge_model=_JUDGE,
        judged_artifact=JudgedArtifact.UNJUDGED,
        configured_max_cost_usd=5.0,
        enforcement_enabled=True,
    )
    return await record_witnessed_cell(
        host,
        run,
        case,
        CandidateOutput(output=[{"answer": _FIRST}], claim_evidence=evidence),
        k_iteration=1,
        result_id="witnessed-1",
        scored_at="2026-11-22T12:00:00+00:00",
        judged_artifact=JudgedArtifact.UNJUDGED,
        judging=judging or WitnessedJudging(run.id),
        web_search=port,
        external_rates=_RATES,
    )


async def test_a_witnessed_cell_replays_its_searches_from_a_recording_and_its_grade_says_so(tmp_path: Path) -> None:
    recording = tmp_path / "search.jsonl"
    live = _FakeWebSearch()
    await _witnessed_cell(_FakeWebJudge(), RecordedWebSearch.record(live, recording))

    result, trace = await _witnessed_cell(_FakeWebJudge(), RecordedWebSearch.replay(recording))

    assert len(live.queries) == 1, "the replay searched nothing"
    grade = trace.claim_grades[_CRITERION]
    assert grade.web_replayed and result.claim_tallies[_CRITERION].by_judge[0].true_ungrounded == 1
    (row,) = [row for row in result.usage if row.provider == _PORT]
    assert (row.role, row.cost_usd) == ("judge", 0.0), "a replay spends nothing, whatever rate the run declares"


async def test_a_witnessed_cells_web_step_keeps_the_cells_budget_and_loses_only_the_webs_decisions() -> None:
    listing = _listing(_listed(0, _FIRST), _listed(1, _SECOND))
    script = {"claim_listing": [listing], "claim_truth": [_truth(0), _truth(1)]}
    one_check = _template(_criterion(web=_WEB.model_copy(update={"max_checked_per_cell": 1})))

    budgeted, _ = await _witnessed_cell(_FakeWebJudge(script=script), _FakeWebSearch(), template=one_check)
    failed, trace = await _witnessed_cell(
        _FakeWebJudge(script={**_MIXED}),
        _FakeWebSearch(failure=ConnectionError("the search is unavailable")),
        evidence=ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=(_BINDING,)),
    )

    (counts,) = budgeted.claim_tallies[_CRITERION].by_judge
    assert (counts.web_checked, counts.web_skipped, counts.unverifiable) == (1, 1, 1), "past its budget: skipped"
    assert failed.claim_failures == {} and failed.claim_tallies[_CRITERION].web_loss == "port_error"
    doubted, supported = trace.claim_grades[_CRITERION].claims
    assert (doubted.web, doubted.verdict) == ("lost", "unverifiable")
    assert (supported.verdict, supported.graded_against) == ("supported", "case"), "case evidence's verdict stands"


async def test_a_witnessed_web_calls_defect_counts_its_searches_against_the_runs_ceiling() -> None:
    judging = WitnessedJudging("run-witnessed")

    with pytest.raises(CompletionProtocolDefect):
        await _witnessed_cell(
            _FakeWebJudge(script={"claim_web_verdict": ["defective"]}), _FakeWebSearch(), judging=judging
        )

    # The listing, the truth call, the searches' writing and the defective reading at $0.002 each, and one search at
    # the declared $0.01.
    assert judging.spent([]) == [pytest.approx(4 * 0.002 + 0.01)]


# =============================================================================
# Budgets: what the hold admits, and what the trace holds
# =============================================================================


@pytest.mark.parametrize(
    ("judge", "cap_usd"),
    [
        pytest.param(
            _FakeWebJudge(usd_per_call=0.002, reply_usd={"claim_web_queries": 5.0}), 1000.0, id="criterions-hold"
        ),
        pytest.param(_FakeWebJudge(), 0.005, id="runs-cap"),
    ],
)
async def test_a_search_refused_at_admission_is_never_sent_and_its_claim_reads_skipped(
    judge: _FakeWebJudge, cap_usd: float
) -> None:
    # criterions-hold: the reply writing the searches cost far more than its ceiling, so the hold has no room for even a
    # free search. runs-cap: the judge's calls are unpriced, each sent only within the run's $0.005 cap; the listing,
    # the truth call and the searches' writing go out at $0.002 each, and the search would be sent past it.
    port = _FakeWebSearch(usd_per_query=0.0)

    judge, (result,) = await _run(port, rates=None, cap_usd=cap_usd, judge=judge)

    assert "claim_web_queries" in judge.asked and port.queries == [], "the search was refused before it was sent"
    assert "claim_web_verdict" not in judge.asked and result.claim_failures == {}
    (counts,) = result.claim_tallies[_CRITERION].by_judge
    assert (counts.web_skipped, counts.unverifiable, counts.asserted_contradicted) == (1, 1, 0)
    assert not [row for row in result.usage if row.provider == _PORT], "a search never sent is never billed"


async def test_a_search_reporting_more_than_its_price_overshoots_the_hold_by_that_one_search_alone() -> None:
    queries = json.dumps({"queries": ["Lantern Almanac first printed", "Lantern Almanac 1921"]})
    judge = _FakeWebJudge(script={"claim_web_queries": [queries]}, usd_per_call=0.002)
    port = _FakeWebSearch(reported_usd=5.0)

    judge, (result,) = await _run(port, rates=None, cap_usd=1000.0, judge=judge)

    assert port.queries == ["Lantern Almanac first printed"], "the search after it is refused at admission"
    # The criterion was held at its ceiling, each search priced at the port's own $0.01 ceiling.
    hold = criterion_ceiling(
        _criterion(), _EVIDENCE, key=None, client=_FakeWebJudge(usd_per_call=0.002), query_usd=0.01
    )
    assert hold is not None
    billed = sum(row.cost_usd or 0.0 for row in result.usage if row.role == "judge")
    assert hold < billed <= hold + (5.0 - 0.01), "past the hold by at most the one search's excess over its price"
    (counts,) = result.claim_tallies[_CRITERION].by_judge
    assert result.claim_failures == {} and counts.web_skipped == 1


#: A web budget whose one check reads up to 200,000 characters of one page.
_WIDE = WebBudget(
    max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=200_000, window_chars=200_000
)

#: A page that states the first claim, then runs on past the wide budget's window.
_LONG_PAGE = WebPage(url=_PAGE.url, title="", text=_PAGE.text + " " + "The reading room opens at nine. " * 7000)


def _filled(chars: int) -> ClaimEvidence:
    """The mixed cell's evidence with a segment of about ``chars`` characters that states nothing."""
    filler = Segment(text="Shelf mark. " * (chars // 12), unit=3)
    return ClaimEvidence(segmentation="sentences-v1", segments=(*_SEGMENTS, filler), documents=(_BINDING,))


async def test_web_windows_that_would_outgrow_the_trace_are_the_web_steps_loss_alone_and_truth_stands() -> None:
    # The evidence leaves the trace about 100,000 bytes, and the one check reads a 200,000-character window.
    evidence = _filled(CLAIM_TRACE_BYTES - 100_000)
    template = _template(_criterion(web=_WIDE))

    kept, kept_trace = await _cell(
        _FakeWebJudge(script=_MIXED), _FakeWebSearch(pages=(_LONG_PAGE,)), template=template, evidence=_filled(12)
    )
    result, trace = await _cell(
        _FakeWebJudge(script=_MIXED), _FakeWebSearch(pages=(_LONG_PAGE,)), template=template, evidence=evidence
    )

    assert len(kept_trace.claim_grades[_CRITERION].web) == 1 and kept.claim_tallies[_CRITERION].web_loss is None
    assert result.claim_failures == {}, "windows past the trace's cap are never the criterion's loss"
    grade = trace.claim_grades[_CRITERION]
    assert grade.web == () and grade.web_loss == result.claim_tallies[_CRITERION].web_loss == "evidence_over_cap"
    doubted, supported = grade.claims
    assert (doubted.web, doubted.verdict, doubted.graded_against) == ("lost", "unverifiable", None)
    assert (supported.verdict, supported.graded_against) == ("supported", "case"), "case evidence's verdict stands"
    assert claim_payload_bytes(evidence, trace.claim_grades) <= CLAIM_TRACE_BYTES


# =============================================================================
# A re-judge finishes a lost web step, and only that step
# =============================================================================


@dataclass
class _FailingSearch(
    _FakeWebSearch
):  # parity-exempt: a scripted search that fails the queries it names, for one family
    failing: frozenset[str] = frozenset()

    async def search(self, query: str) -> WebSearchResult:
        if query in self.failing:
            self.queries.append(query)
            raise ConnectionError("the search is unavailable")
        return await super().search(query)


def _queries(query: str) -> str:
    return json.dumps({"queries": [query]})


#: Three doubted claims, each nominated for the web.
_THREE = {
    "claim_listing": [_listing(_listed(0, _FIRST), _listed(1, _SECOND), _listed(2, _FIRST))],
    "claim_truth": [_truth(0), _truth(1), _truth(2)],
}


async def test_a_rejudge_finishes_a_lost_web_step_alone_and_never_searches_a_check_already_made() -> None:
    # The first grade checked claim 0, then the port failed claim 1's search: claims 1 and 2 read lost.
    first = _FakeWebJudge(script={**_THREE, "claim_web_queries": [_queries("first"), _queries("second")]})
    again = _FakeWebJudge(script={"claim_web_queries": [_queries("second")]})
    host, result = await _stored_web_cell(first, again, _FailingSearch(failing=frozenset({"second"})))
    assert result.claim_failures == {} and result.claim_tallies[_CRITERION].web_loss == "port_error"
    stored = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert stored is not None
    before = stored.claim_grades[_CRITERION]
    port = _FakeWebSearch()

    rejudged = await rejudge_result(host, result.id, "u", web_search=port)

    assert again.asked == ["claim_web_queries", "claim_web_verdict"], "the extraction and truth step are never asked"
    assert port.queries == ["second"], "the check already made is never searched again"
    finished = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert finished is not None
    grade = finished.claim_grades[_CRITERION]
    assert grade.calls[: len(before.calls)] == before.calls and grade.web[0] == before.web[0]
    assert grade.claims[0] == before.claims[0], "the claim checked before stands as stored"
    assert [claim.truth_citations for claim in grade.claims] == [claim.truth_citations for claim in before.claims]
    checked, skipped = grade.claims[1], grade.claims[2]
    assert (checked.web, checked.verdict, checked.decided_by) == ("checked", "true_ungrounded", len(before.calls) + 1)
    assert (skipped.web, skipped.verdict) == ("skipped_cap", "unverifiable"), "the cell's budget counts claim 0's check"
    assert grade.web_loss is None and rejudged.claim_failures == {}
    (counts,) = rejudged.claim_tallies[_CRITERION].by_judge
    assert (counts.web_checked, counts.web_skipped, counts.true_ungrounded) == (2, 1, 2)
    (rescore,) = rejudged.claim_rescores
    assert (rescore.criteria, rescore.web_steps, rescore.prior, rescore.lost) == (
        [_CRITERION],
        [_CRITERION],
        {_CRITERION: "port_error"},
        {},
    )
    (searched,) = [row for row in rescore.usage if row.provider == _PORT]
    assert searched.call_count == 1, "the re-judge billed its search"


async def test_a_web_step_finished_again_keeps_the_repairs_it_spent_and_loses_only_the_web_again() -> None:
    # The cell's one web repair went on claim 0's first answer; claim 1's search then failed.
    unquoted = _verdict(quote="The almanac first appeared in 1921, the press says.")
    listing = _listing(_listed(0, _FIRST), _listed(1, _SECOND))
    first = _FakeWebJudge(
        script={
            "claim_listing": [listing],
            "claim_truth": [_truth(0), _truth(1)],
            "claim_web_queries": [_queries("first"), _queries("second")],
            "claim_web_verdict": [unquoted],
        }
    )
    again = _FakeWebJudge(script={"claim_web_queries": [_queries("second")], "claim_web_verdict": [unquoted]})
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1)))
    host, result = await _stored_web_cell(
        first, again, _FailingSearch(failing=frozenset({"second"})), template=template
    )
    assert "claim_web_repair" in first.asked, "the control: the first grade spent the step's one repair"

    rejudged = await rejudge_result(host, result.id, "u", web_search=_FakeWebSearch())

    assert "claim_web_repair" not in again.asked, "no repair is left for the step it finishes"
    assert rejudged.claim_failures == {} and rejudged.claim_tallies[_CRITERION].web_loss == "repair_budget_spent"
    finished = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert finished is not None
    grade = finished.claim_grades[_CRITERION]
    assert (grade.claims[0].web, grade.claims[0].verdict) == ("checked", "true_ungrounded"), "truth stands"
    assert (grade.claims[1].web, grade.claims[1].verdict) == ("lost", "unverifiable")
    (rescore,) = rejudged.claim_rescores
    assert (rescore.prior, rescore.lost) == ({_CRITERION: "port_error"}, {_CRITERION: "repair_budget_spent"})


async def test_a_rejudge_of_a_lost_web_step_with_no_port_is_refused_by_name_before_any_call() -> None:
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(
        _FakeWebJudge(), again, _FakeWebSearch(failure=ConnectionError("the search is unavailable"))
    )

    with pytest.raises(ValidationFailedError, match=f"claim criteria '{_CRITERION}'.*declare a web budget"):
        await rejudge_result(host, result.id, "u")

    assert again.asked == []
    stored = off_loop(host.storage.load_eval_result, result.id, "u")
    assert (
        stored is not None and stored.claim_tallies[_CRITERION].web_loss == "port_error" and not stored.claim_rescores
    )


@pytest.mark.parametrize("lost", ["criterion", "web-step"])
async def test_a_loss_to_the_traces_cap_is_never_graded_again_and_a_rejudge_refuses_it_by_name(lost: str) -> None:
    # The evidence a re-judge reads is the evidence the cell graded, so the grade would outgrow the trace again.
    again = _FakeWebJudge()
    failing = _FakeWebSearch(failure=ConnectionError("the search is unavailable"))
    first = _FakeWebJudge(script=_LOST_FIRST if lost == "criterion" else {})
    host, result = await _stored_web_cell(first, again, failing)
    trace = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert trace is not None
    if lost == "criterion":
        result = result.model_copy(update={"claim_failures": {_CRITERION: "evidence_over_cap"}})
    else:
        tally = result.claim_tallies[_CRITERION].model_copy(update={"web_loss": "evidence_over_cap"})
        grade = trace.claim_grades[_CRITERION].model_copy(update={"web_loss": "evidence_over_cap"})
        result = result.model_copy(update={"claim_tallies": {_CRITERION: tally}})
        trace = trace.model_copy(update={"claim_grades": {_CRITERION: grade}})
    off_loop(host.storage.save_eval_result, result, trace)
    port = _FakeWebSearch()

    with pytest.raises(ValidationFailedError, match=f"'{_CRITERION}' to the trace's cap \\(evidence_over_cap\\)"):
        await rejudge_result(host, result.id, "u", web_search=port)

    assert again.asked == [] and port.queries == [], "refused before any call or search"


async def test_a_rejudge_grades_the_other_lost_criterion_and_names_the_one_lost_to_the_traces_cap_as_skipped() -> None:
    # Two criteria, both lost on the first grade; the cell's store then says the second outgrew the trace.
    other = "almanac.dates"
    template = _template(_criterion())
    template = template.model_copy(update={"claim_criteria": [_criterion(), _criterion(name=other, web=None)]})
    again = _FakeWebJudge()
    host, result = await _stored_web_cell(_FakeWebJudge(script=_LOST_FIRST), again, _FakeWebSearch(), template=template)
    assert result.claim_failures == {_CRITERION: "provider_error", other: "provider_error"}
    trace = off_loop(host.storage.load_eval_trace, result.id, "u")
    over = result.model_copy(update={"claim_failures": {_CRITERION: "provider_error", other: "evidence_over_cap"}})
    off_loop(host.storage.save_eval_result, over, trace)

    rejudged = await rejudge_result(host, result.id, "u", web_search=_FakeWebSearch())

    assert again.asked.count("claim_listing") == 1, "only the criterion not lost to the cap is graded again"
    assert rejudged.claim_failures == {other: "evidence_over_cap"}, "the one lost to the cap is left as stored"
    assert set(rejudged.claim_tallies) == {_CRITERION}
    (rescore,) = rejudged.claim_rescores
    assert (rescore.criteria, rescore.skipped) == ([_CRITERION], {other: "evidence_over_cap"})
