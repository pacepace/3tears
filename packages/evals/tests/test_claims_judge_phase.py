"""Claim criteria graded in the judge phase: after the rubric dimensions, on the same judge, each under its own hold.

Every behaviour is driven through the real runner over a scripted judge that answers each call by the reply schema it
is sent, so nothing is spent and each reply is chosen. A cell carrying claim evidence is claim-graded whatever its kind
declares a judge reads. Every way a criterion fails is its own loss on the cell, never a judge error: the trial keeps
its pass and its rubric scores, and every other criterion stands. A ground call's failure, however it fails, costs
groundedness alone. A deadline inside a criterion loses the criteria not yet graded. Each criterion is priced before
any call, per step, and never spends past what the cap set aside for it. A trace that is not stored takes the claims
it would have backed with it. A claims call refused for the calling account stops the run, as a dimension's does.

A finished cell's lost criteria are re-judged from the claim evidence its trace stores: a lost one is graded again
whole, under the fingerprint the cell recorded for it or not at all, a completed grade is never asked again, and a
criterion whose tally waited on its trace is recomputed from the stored grade with no call. The grade records its
ground step's loss, so a recomputed tally loses groundedness alone, never truth. Claims calls replay by their prompt.
The text is invented: a reading room's almanac and its shelf notes.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.evals.tests.factories import make_eval_run, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.claims.readings import claim_readings
from threetears.evals.kernel.claims.grading import (
    criterion_ceiling,
    criterion_fingerprint,
    criterion_steps,
    grade_tally,
)
from threetears.evals.kernel.errors import ConflictError, ValidationFailedError
from threetears.evals.kernel.host.apparatus import ReplayMiss
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.run.budget import AccountExhaustedError, BudgetStoppedError, EvalRunCostCap
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.lifecycle import rejudge_result
from threetears.evals.run.runner import (
    RunnerOptions,
    StoredCell,
    StoredCells,
    execute_run,
    judge_witnessed_output,
    run_one_result,
)
from threetears.evals.run.witnessed import WitnessedJudging, record_witnessed_cell, stamp_witnessed_judge
from threetears.evals.schema import Segment
from threetears.evals.schema.call_inputs import current_call_inputs
from threetears.evals.schema.completion import CompletionProtocolDefect, ProviderFailure, ProviderFailureDescriber
from threetears.evals.schema.models import (
    CLAIM_LOSS_REASON_CHARS,
    ClaimCall,
    ClaimCriterion,
    ClaimEvidence,
    EvalTemplate,
    EvalTestCase,
    EvidenceDocument,
    JudgedArtifact,
    JudgeEvidence,
    PageBudget,
    RepairBudget,
    RubricDim,
    WebBudget,
)
from threetears.evals.schema.job_lease import JobLease
from threetears.evals.schema.reference import ReferenceEntity, ReferenceFact, ReferenceKey
from threetears.evals.schema.values import SweepableValue
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.testing.recorded_completions import RecordedCompletions

_KIND = "claims-single-shot"
_JUDGE = "scripted/claims-judge"
_CRITERION = "almanac.claims"

_SENTENCE = "The Lantern Almanac was first printed in 1921."

_SEGMENTS = (Segment(text="Welcome back to the reading room."), Segment(text=_SENTENCE, unit=0))

#: The shelf notes the candidate itself was given: what grounds its claim.
_NOTES = EvidenceDocument(
    id="shelf-notes",
    provenance="run",
    title="Shelf notes",
    text="The Lantern Almanac was first printed in 1921, by a small press.",
)

_EVIDENCE = ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=(_NOTES,))

_KEY = ReferenceKey(
    core=(
        ReferenceEntity(
            name="The Lantern Almanac",
            facts=(ReferenceFact(claim=_SENTENCE, source_url="https://example.org/almanac"),),
        ),
    )
)

_LISTED = {
    "segment": 1,
    "claim": _SENTENCE,
    "excerpt": _SENTENCE.rstrip("."),
    "specific": True,
    "hedged": False,
    "attributed": False,
    "scope": "world",
    "restates": None,
}


def _criterion(name: str = _CRITERION, **overrides: Any) -> ClaimCriterion:
    fields: dict[str, Any] = {
        "name": name,
        "grounded_in": ("run",),
        "pages": PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
        "repairs": RepairBudget(rounds=2, max_items_per_cell=4),
        "judge": "builtin",
    }
    fields.update(overrides)
    return ClaimCriterion(**fields)


def _template(*criteria: ClaimCriterion, rubric: tuple[RubricDim, ...] = ()) -> EvalTemplate:
    return EvalTemplate(
        scope_id="u",
        name="almanac answers",
        intent="answer questions about the reading room's almanac",
        candidate_kind=_KIND,
        rubric=list(rubric),
        claim_criteria=list(criteria or (_criterion(),)),
    )


def _case(template: EvalTemplate) -> EvalTestCase:
    return EvalTestCase(template_id=template.id, scope_id="u", reference=_KEY)


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
    #: What the request was sent with, as the client reports it; a string is the wrong type, a defect.
    sent_settings: Any = None


#: A completion a real client returns with its ``sent_settings`` the wrong type: the host's defect, found on read.
_DEFECTIVE = "defective"


def _listing(*claims: dict[str, Any]) -> str:
    return json.dumps({"claims": list(claims)})


def _truth_supported() -> str:
    cited = {"ref": "K1", "stance": "supports", "quote": _SENTENCE, "subject": "The Lantern Almanac", "value": "1921"}
    return json.dumps({"claims": [{"claim": 0, "citations": [cited], "suspect": False}]})


def _ground_supported() -> str:
    cited = {"ref": "S1", "quote": _NOTES.text, "subject": "The Lantern Almanac", "value": "1921"}
    return json.dumps({"claims": [{"claim": 0, "citations": [cited]}]})


#: What each step answers by default: the claim listed, supported by the key, and grounded in the shelf notes.
_ANSWERS: dict[str, str] = {
    "claim_listing": _listing(_LISTED),
    "claim_truth": _truth_supported(),
    "claim_ground": _ground_supported(),
}


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeClaimsJudge:
    """Answers each call by the reply schema it is sent: a rubric dimension a 4, each claims step from its script.

    ``script`` maps a reply schema's name to what each call of it answers in turn (the last answers every later
    call), an exception it raises, or ``hang`` for a call that never answers. Records every call's schema name, and
    the inputs bound around it.
    """

    script: dict[str, list[Any]] = field(default_factory=dict)
    usd_per_call: float | None = None
    model_name: str = _JUDGE
    asked: list[str] = field(default_factory=list)
    bound: list[tuple[str, Any]] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return self.usd_per_call

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        del system, user
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        self.bound.append((name, current_call_inputs()))
        if name == "judge_reply":
            return _Reply(content=json.dumps({"reasoning": "reads well", "criteria_scores": {"doc.clarity": 4}}))
        answers = self.script.get(name) or [_ANSWERS[name]]
        answer = answers[min(self.asked.count(name), len(answers)) - 1]
        if answer == "hang":
            await asyncio.Event().wait()
        if isinstance(answer, BaseException):
            raise answer
        if answer == _DEFECTIVE:
            return _Reply(content=_ANSWERS[name], sent_settings="fast")
        return _Reply(content=answer)

    async def aclose(self) -> None:
        """Nothing to release."""


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeClaimsKind:
    """A single-shot kind that hands back one canned output, claim evidence and all."""

    def __init__(self, output: CandidateOutput, *, judged_artifact: JudgedArtifact = JudgedArtifact.UNJUDGED) -> None:
        self.output = output
        self.judged_artifact = judged_artifact

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


def _output(*, judged: bool = False, evidence: ClaimEvidence | None = _EVIDENCE, inputs: Any = None) -> CandidateOutput:
    return CandidateOutput(
        output=[{"answer": _SENTENCE}],
        judge_evidence=(
            JudgeEvidence(case_material="the almanac's catalogue entry", artifact=_SENTENCE, inputs=inputs)
            if judged
            else None
        ),
        claim_evidence=evidence,
    )


def _service(judge: Any, describer: ProviderFailureDescriber = withhold_failure_detail) -> JudgeService:
    return JudgeService(client_factory=lambda model, temperature: judge, configs={}, failure_describer=describer)


_RUBRIC = (RubricDim(name="doc.clarity", description="reads clearly", scale="ordinal"),)

_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}


async def _cell(judge: Any, template: EvalTemplate, kind: _FakeClaimsKind) -> Any:
    return await run_one_result(
        toyhost_host(),
        template=template,
        test_case=_case(template),
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id="run-1",
        scope_id="u",
        judge_service=_service(judge),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )


async def test_a_cell_of_a_kind_graded_by_code_is_graded_by_its_templates_claim_criteria() -> None:
    judge = _FakeClaimsJudge()

    result, trace = await _cell(judge, _template(), _FakeClaimsKind(_output()))

    tally = result.claim_tallies[_CRITERION]
    (counts,) = tally.by_judge
    assert (tally.units, counts.asserted, counts.asserted_specific, counts.grounded_specific) == (1, 1, 1, 1)
    assert result.claim_failures == {} and result.judge_error is None and result.rubric_scores == []
    assert judge.asked == ["claim_listing", "claim_truth", "claim_ground"]
    grade = trace.claim_grades[_CRITERION]
    assert grade.criterion_fingerprint == tally.criterion_fingerprint
    assert [call.role for call in grade.calls] == ["extract", "truth", "ground"]
    assert trace.claim_evidence == _EVIDENCE
    (row,) = [row for row in result.usage if row.role == "judge"]
    assert row.call_count == 3, "every claims call is one of the cell's judge calls, billed in its judge rows"


async def test_a_lost_criterion_keeps_the_rubric_scores_and_never_reads_as_a_judge_error() -> None:
    judge = _FakeClaimsJudge(script={"claim_listing": ["not json"]})
    kind = _FakeClaimsKind(_output(judged=True), judged_artifact=JudgedArtifact.DOCUMENT)

    result, trace = await _cell(judge, _template(rubric=_RUBRIC), kind)

    assert result.claim_failures == {_CRITERION: "unreadable"} and result.claim_tallies == {}
    assert [score.dim for score in result.rubric_scores] == ["doc.clarity"]
    assert result.judge_error is None, "a criterion's loss is never a judge error, which would exclude the trial"
    assert judge.asked == ["judge_reply", "claim_listing", "claim_listing", "claim_listing"], "dims first"
    assert trace.claim_grades == {} and trace.claim_evidence == _EVIDENCE
    first, *again = trace.claim_losses[_CRITERION].calls
    assert first.reason is None and len(again) == 2, "each listing asked again keeps why the last could not be read"
    assert all(call.step == "extract" and call.reason for call in again)


async def test_a_provider_failing_a_call_loses_that_criterion_and_every_other_is_graded() -> None:
    judge = _FakeClaimsJudge(
        script={"claim_listing": [ConnectionError("the provider is unavailable"), _ANSWERS["claim_listing"]]}
    )
    template = _template(_criterion("almanac.first"), _criterion("almanac.second"))

    result, _ = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {"almanac.first": "provider_error"}
    assert set(result.claim_tallies) == {"almanac.second"}


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("the provider is unavailable"),
        _DEFECTIVE,
        "not json",
    ],
    ids=["provider", "protocol_defect", "unreadable"],
)
async def test_a_ground_call_failing_costs_groundedness_alone_and_truth_stands(failure: Any) -> None:
    judge = _FakeClaimsJudge(script={"claim_ground": [failure]})

    result, trace = await _cell(judge, _template(), _FakeClaimsKind(_output()))

    tally = result.claim_tallies[_CRITERION]
    (counts,) = tally.by_judge
    expected = {"not json": "unreadable", _DEFECTIVE: "protocol_defect"}.get(failure, "provider_error")
    assert tally.ground_loss == expected, "a host's broken client is named as its defect, never a provider's failure"
    assert (counts.asserted, counts.grounded_specific) == (1, 0), "the truth counts stand; nothing is grounded"
    assert trace.claim_grades[_CRITERION].claims[0].verdict == "supported"
    if isinstance(failure, ConnectionError):
        # The provider's message is the provider's: the record keeps the calls, and no failure text.
        record = trace.claim_losses[_CRITERION]
        assert record.failure is None and "unavailable" not in record.model_dump_json()
    if failure == _DEFECTIVE:
        (row,) = [row for row in result.usage if row.role == "judge"]
        assert row.call_count == 3, "the defective completion is billed, once"


async def test_a_completion_breaking_the_protocol_in_a_truth_call_fails_the_run() -> None:
    judge = _FakeClaimsJudge(script={"claim_truth": [_DEFECTIVE]})

    with pytest.raises(CompletionProtocolDefect):
        await _cell(judge, _template(), _FakeClaimsKind(_output()))


async def test_a_run_a_claims_completion_breaks_records_its_cell_with_what_it_spent_and_fails() -> None:
    judge = _FakeClaimsJudge(script={"claim_listing": [_DEFECTIVE]})
    template = _template()
    host = toyhost_host()
    case = _case(template)
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)
    kind = _FakeClaimsKind(_output())

    with pytest.raises(CompletionProtocolDefect):
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=_service(judge),
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        )

    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    (row,) = [row for row in result.usage if row.role == "judge"]
    assert (row.call_count, row.cost_usd) == (1, 0.002), "the defective call is billed once, and counted"
    assert result.cost_usd == 0.002
    assert result.claim_failures == {_CRITERION: "provider_error"}
    assert result.infra_error is not None and "CompletionProtocolDefect" in result.infra_error


async def test_the_extractions_repairs_count_against_the_criterions_repair_budget() -> None:
    misquoted = {**_LISTED, "excerpt": "The Lantern Almanac was first bound in 1921"}
    judge = _FakeClaimsJudge(script={"claim_listing": [_listing(misquoted)]})
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=0)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {_CRITERION: "repair_budget_spent"}
    assert judge.asked == ["claim_listing"], "a budget of no items asks none again"
    record = trace.claim_losses[_CRITERION]
    assert [call.reason for call in record.calls] == [None], "the step stopped before it asked anything again"
    assert record.failure is not None and record.failure.startswith(
        "the claim extractor would ask 1 claim(s) again with 0 of the step's 0 repairs spent: claim 0: "
    ), "the failure that spent the budget is kept, naming the claim, though no call carried it"


#: A truth answer citing a key fact the step never showed: a fault every repair is asked to fix.
_MISCITED = {"ref": "K9", "stance": "supports", "quote": _SENTENCE, "subject": "The Lantern Almanac", "value": "1921"}

#: What the truth step names to the model when an answer cites ``K9``.
_MISCITED_FAULT = "it cites K9, which is no evidence shown here; cite the refs listed under Evidence"


async def test_a_cell_its_truth_repairs_lose_keeps_every_call_it_made_with_the_failure_each_repair_named() -> None:
    truth = json.dumps({"claims": [{"claim": 0, "citations": [_MISCITED], "suspect": False}]})
    repaired = json.dumps({"citations": [_MISCITED], "suspect": False})
    judge = _FakeClaimsJudge(script={"claim_truth": [truth], "claim_truth_repair": [repaired]})
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {_CRITERION: "repair_budget_spent"} and trace.claim_grades == {}
    calls = trace.claim_losses[_CRITERION].calls
    assert [(call.step, call.role, call.reason) for call in calls] == [
        ("extract", "extract", None),
        ("truth", "truth", None),
        ("truth", "repair", _MISCITED_FAULT),
    ], "the lost cell keeps each call its budget paid for, the repair naming the failure it was asked to fix"
    assert {call.served_model for call in calls} == {_JUDGE}
    assert trace.claim_losses[_CRITERION].failure == (
        f"the claim verifier would ask 1 truth claim(s) again with 1 of the step's 1 repairs spent: truth claim 0: "
        f"{_MISCITED_FAULT}"
    ), "the step stopped before its next round, and the failure that stopped it is kept"


async def test_a_cell_lost_after_its_last_round_keeps_the_failure_of_the_last_answer_not_the_one_repaired() -> None:
    other = {**_MISCITED, "ref": "K8"}
    truth = json.dumps({"claims": [{"claim": 0, "citations": [_MISCITED], "suspect": False}]})
    repaired = json.dumps({"citations": [other], "suspect": False})
    judge = _FakeClaimsJudge(script={"claim_truth": [truth], "claim_truth_repair": [repaired]})
    template = _template(_criterion(repairs=RepairBudget(rounds=1, max_items_per_cell=4)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {_CRITERION: "repair_budget_spent"}
    record = trace.claim_losses[_CRITERION]
    assert record.calls[-1].reason == _MISCITED_FAULT, "the repair was asked to fix the first fault"
    assert record.failure == (
        "the claim verifier could not repair 1 claim(s) in 1 round(s): truth claim 0: it cites K8, which is no "
        "evidence shown here; cite the refs listed under Evidence"
    ), "the failure kept is the repaired answer's own, which no call named"


async def test_a_truth_reply_asked_again_after_it_could_not_be_read_keeps_why_it_could_not() -> None:
    truth = json.dumps({"claims": [{"claim": 0, "citations": [_MISCITED], "suspect": False}]})
    repaired = json.dumps({"citations": [_MISCITED], "suspect": False})
    judge = _FakeClaimsJudge(script={"claim_truth": ["not json", truth], "claim_truth_repair": [repaired]})
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {_CRITERION: "repair_budget_spent"}
    first, again = trace.claim_losses[_CRITERION].calls[1:3]
    assert (first.step, first.role, first.reason) == ("truth", "truth", None)
    assert (again.step, again.role) == ("truth", "truth") and again.reason, "the reply asked again names why"


async def test_a_fault_past_the_records_cap_is_cut_and_the_criterion_keeps_its_own_loss() -> None:
    # Recording must never change a grade's outcome: a fault longer than the cap is cut, never refused by the record.
    long = {**_MISCITED, "ref": "K" + "9" * (CLAIM_LOSS_REASON_CHARS + 100)}
    truth = json.dumps({"claims": [{"claim": 0, "citations": [long], "suspect": False}]})
    repaired = json.dumps({"citations": [long], "suspect": False})
    judge = _FakeClaimsJudge(script={"claim_truth": [truth], "claim_truth_repair": [repaired]})
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {_CRITERION: "repair_budget_spent"}, "its own loss, never provider_error"
    record = trace.claim_losses[_CRITERION]
    reason = record.calls[-1].reason
    assert reason is not None and len(reason) == CLAIM_LOSS_REASON_CHARS and reason.startswith("it cites K999")
    assert record.failure is not None and len(record.failure) == CLAIM_LOSS_REASON_CHARS


async def test_a_cell_its_re_ask_loses_names_the_re_ask_as_the_step_that_spent_the_budget() -> None:
    # A suspect answer citing nothing about a key entity is asked again alone; that re-ask's repairs spend the budget.
    doubted = json.dumps({"claims": [{"claim": 0, "citations": [], "suspect": True}]})
    reask = json.dumps({"claims": [{"claim": 0, "citations": [_MISCITED]}]})
    repaired = json.dumps({"citations": [_MISCITED], "suspect": False})
    judge = _FakeClaimsJudge(
        script={"claim_truth": [doubted], "claim_reask": [reask], "claim_truth_repair": [repaired]}
    )
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {_CRITERION: "repair_budget_spent"}
    assert [(call.step, call.role, call.reason) for call in trace.claim_losses[_CRITERION].calls] == [
        ("extract", "extract", None),
        ("truth", "truth", None),
        ("reask", "truth", None),
        ("reask", "repair", _MISCITED_FAULT),
    ]


async def test_a_ground_step_its_repairs_lose_keeps_its_calls_and_the_failure_each_repair_named() -> None:
    miscited = {"ref": "S9", "quote": _NOTES.text, "subject": "The Lantern Almanac", "value": "1921"}
    ground = json.dumps({"claims": [{"claim": 0, "citations": [miscited]}]})
    judge = _FakeClaimsJudge(
        script={"claim_ground": [ground], "claim_ground_repair": [json.dumps({"citations": [miscited]})]}
    )
    template = _template(_criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1)))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output()))

    assert result.claim_failures == {} and result.claim_tallies[_CRITERION].ground_loss == "repair_budget_spent"
    calls = trace.claim_losses[_CRITERION].calls
    assert [(call.step, call.role) for call in calls] == [
        ("extract", "extract"),
        ("truth", "truth"),
        ("ground", "ground"),
        ("ground", "repair"),
    ]
    assert calls[-1].reason == "it cites S9, which is no evidence shown here; cite the refs listed under Evidence"
    assert trace.claim_losses[_CRITERION].failure == (
        "ground step: the claim verifier would ask 1 ground claim(s) again with 1 of the step's 1 repairs spent: "
        "ground claim 0: it cites S9, which is no evidence shown here; cite the refs listed under Evidence"
    )
    grade = trace.claim_grades[_CRITERION]
    assert [ClaimCall.model_validate(call.model_dump(exclude={"step", "reason"})) for call in calls] == list(
        grade.calls
    ), "the calls kept are the grade's own, in its order"


async def test_a_cell_that_loses_nothing_stores_no_loss_record() -> None:
    judge = _FakeClaimsJudge()

    _, trace = await _cell(judge, _template(), _FakeClaimsKind(_output()))

    assert trace.claim_grades and trace.claim_losses == {}
    assert "claim_losses" not in trace.model_dump(mode="json"), "a trace with no loss is stored as before"


async def test_no_input_a_rubric_call_binds_reaches_a_claims_call() -> None:
    judge = _FakeClaimsJudge()
    kind = _FakeClaimsKind(
        _output(judged=True, inputs={"entry": "almanac-1921"}), judged_artifact=JudgedArtifact.DOCUMENT
    )

    await _cell(judge, _template(rubric=_RUBRIC), kind)

    rubric = [inputs for name, inputs in judge.bound if name == "judge_reply"]
    assert rubric and None not in rubric, "the control: the dim's call is bound to the inputs its evidence names"
    assert {inputs for name, inputs in judge.bound if name.startswith("claim_")} == {None}


async def test_a_template_with_claim_criteria_needs_a_judge_and_a_web_budget_needs_a_web_search() -> None:
    template = _template()
    run = make_eval_run(scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=["tc-1"])
    kind = _FakeClaimsKind(_output())

    with pytest.raises(ValueError, match="this run has no judge"):
        await execute_run(
            toyhost_host(),
            run=run,
            template=template,
            test_cases=[_case(template)],
            judge_service=None,
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        )
    webbed = _template(
        _criterion(
            web=WebBudget(
                max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=100, window_chars=100
            )
        )
    )
    with pytest.raises(ValueError, match="this run has no web search"):
        await _cell(_FakeClaimsJudge(), webbed, kind)


async def _judged_run(
    judge: _FakeClaimsJudge,
    template: EvalTemplate,
    kind: _FakeClaimsKind,
    *,
    cap_usd: float | None = None,
    cell_timeout_s: float | None = None,
) -> tuple[Any, Any, BudgetStoppedError | None]:
    """One judged cell through the run loop: its stored result, its stored trace, and the run's stop, if one."""
    host = toyhost_host()
    case = _case(template)
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)
    cap = EvalRunCostCap(run.id, cap_usd, enabled=True) if cap_usd is not None else None
    stopped: BudgetStoppedError | None = None
    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=_service(judge),
            options=RunnerOptions(
                candidate_kinds={_KIND: lambda _cell: kind},
                **({"cell_timeout_s": cell_timeout_s} if cell_timeout_s is not None else {}),
            ),
            **({"budget_gate": cap.check, "on_cost": cap.record} if cap is not None else {}),
        )
    except BudgetStoppedError as error:
        stopped = error
    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    trace = off_loop(host.storage.load_eval_trace, result.id, result.scope_id)
    return result, trace, stopped


async def test_a_deadline_inside_a_criterion_loses_the_unfinished_criteria_and_the_rest_stands() -> None:
    judge = _FakeClaimsJudge(script={"claim_listing": [_ANSWERS["claim_listing"], "hang"]})
    kind = _FakeClaimsKind(_output(judged=True), judged_artifact=JudgedArtifact.DOCUMENT)
    template = _template(_criterion("almanac.first"), _criterion("almanac.second"), rubric=_RUBRIC)

    result, trace, _ = await _judged_run(judge, template, kind, cell_timeout_s=0.5)

    assert result.termination == "cell_timeout"
    assert result.claim_failures == {"almanac.second": "deadline"}
    assert set(result.claim_tallies) == {"almanac.first"} and set(trace.claim_grades) == {"almanac.first"}
    assert [score.dim for score in result.rubric_scores] == ["doc.clarity"], "the dims already settled stand"
    assert result.judge_error is None and result.infra_error is None, "the cell keeps its trial"


async def test_a_deadline_inside_the_claims_of_a_kind_graded_by_code_is_the_criterions_alone() -> None:
    judge = _FakeClaimsJudge(script={"claim_truth": ["hang"]})

    result, trace, _ = await _judged_run(judge, _template(), _FakeClaimsKind(_output()), cell_timeout_s=0.5)

    assert result.claim_failures == {_CRITERION: "deadline"}
    assert result.infra_error is None, "a deadline in the claims phase is the criterion's, never the rig's"
    assert trace.claim_evidence == _EVIDENCE
    assert [(call.step, call.role) for call in trace.claim_losses[_CRITERION].calls] == [("extract", "extract")], (
        "a criterion the deadline cut keeps the calls that returned before it"
    )


async def test_a_deadline_in_the_ground_step_costs_groundedness_alone_and_truth_stands() -> None:
    judge = _FakeClaimsJudge(script={"claim_ground": ["hang"]})

    result, trace, _ = await _judged_run(judge, _template(), _FakeClaimsKind(_output()), cell_timeout_s=0.5)

    assert result.termination == "cell_timeout" and result.claim_failures == {}
    tally = result.claim_tallies[_CRITERION]
    (counts,) = tally.by_judge
    assert tally.ground_loss == "deadline"
    assert (counts.asserted, counts.grounded_specific) == (1, 0), "truth stands; nothing is grounded"
    assert trace.claim_grades[_CRITERION].claims[0].verdict == "supported"


async def test_a_criterion_the_cap_has_no_room_for_is_lost_to_it_and_the_rubric_dims_are_judged() -> None:
    # Each call is priced at $0.30: the dim's two attempts fit a $1 cap, and the criterion's 21 calls do not.
    judge = _FakeClaimsJudge(usd_per_call=0.3)
    kind = _FakeClaimsKind(_output(judged=True), judged_artifact=JudgedArtifact.DOCUMENT)

    result, _, stopped = await _judged_run(judge, _template(rubric=_RUBRIC), kind, cap_usd=1.0)

    assert result.claim_failures == {_CRITERION: "cost_cap"}
    assert [score.dim for score in result.rubric_scores] == ["doc.clarity"] and not result.judging_set_aside
    assert not any(name.startswith("claim_") for name in judge.asked), "nothing of the criterion was asked"
    assert stopped is not None, "the cap had too little left, so the run starts no further cell"
    assert stopped.stops[0].cells_not_judged == 0, "the cell's dims were judged: it is no cell the cap left unjudged"


async def test_a_cap_refusing_an_unpriced_criterions_call_asks_no_later_criterion_and_counts_no_cell() -> None:
    # Unpriced, so each criterion's calls are sent one at a time while the run is within its $0.003 cap. Each call
    # costs $0.002: the extraction and the truth call go out, the ground call is refused, and the second criterion
    # is never asked.
    judge = _FakeClaimsJudge()
    template = _template(_criterion("almanac.first"), _criterion("almanac.second"))

    result, _, stopped = await _judged_run(judge, template, _FakeClaimsKind(_output()), cap_usd=0.003)

    assert judge.asked == ["claim_listing", "claim_truth"]
    assert result.claim_tallies["almanac.first"].ground_loss == "cost_cap", "truth stands; the cap cost groundedness"
    assert result.claim_failures == {"almanac.second": "cost_cap"}
    assert stopped is not None and stopped.stops[0].cells_not_judged == 0


@dataclass
class _BytePricedJudge(_FakeClaimsJudge):
    """Prices a call by the bytes it sends, so a call whose prompt outgrows what it was admitted at costs more."""

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del response_format
        return (len(system.encode()) + len(user.encode())) * 1e-6


async def test_a_criterion_never_spends_past_the_ceiling_it_was_admitted_at() -> None:
    # The extractor states its claim at a length no text of this size gives: the truth call that would echo it is
    # priced past what the criterion's hold set aside for it, so it is not sent.
    padded = {**_LISTED, "claim": _SENTENCE + " " + "And so on. " * 100_000}
    judge = _BytePricedJudge(script={"claim_listing": [_listing(padded)]})

    result, _, stopped = await _judged_run(judge, _template(), _FakeClaimsKind(_output()), cap_usd=50.0)

    assert result.claim_failures == {_CRITERION: "hold_exceeded"}, "the price's miss, never the cap's"
    assert judge.asked == ["claim_listing"], "the call past the hold was refused before it was sent"
    assert stopped is None, "the run's cap had room: no cell after it is kept from running"


def test_a_criterion_is_priced_per_step_each_with_its_own_repair_budget() -> None:
    judge = _FakeClaimsJudge(usd_per_call=1.0)
    # Two batches (one unit, and the segment outside any unit), three attempts each, and four repairs per step:
    # the extraction 3 + 4, the truth step 2 x 3 + 4 and, since the key states a fact, its one re-ask of three
    # attempts, the ground step 2 x 3 + 4.
    assert criterion_ceiling(_criterion(), _EVIDENCE, key=_KEY, client=judge) == 7 + 13 + 10
    assert criterion_ceiling(_criterion(grounded_in=None), _EVIDENCE, key=_KEY, client=judge) == 7 + 13
    assert criterion_ceiling(_criterion(grounded_in=None), _EVIDENCE, key=None, client=judge) == 7 + 10
    assert criterion_ceiling(_criterion(), _EVIDENCE, key=_KEY, client=_FakeClaimsJudge()) is None


def test_a_criterions_fingerprint_moves_with_its_definition_and_never_with_its_model() -> None:
    def fingerprint(criterion: ClaimCriterion, model: str) -> str:
        return criterion_fingerprint(criterion, *criterion_steps(criterion, _FakeClaimsJudge(model_name=model)))

    base = fingerprint(_criterion(), _JUDGE)
    assert fingerprint(_criterion(), "scripted/another-judge") == base
    assert fingerprint(_criterion(guidance="An almanac's edition is a claim."), _JUDGE) != base
    assert fingerprint(_criterion(repairs=RepairBudget(rounds=1, max_items_per_cell=4)), _JUDGE) != base


async def test_a_witnessed_cell_is_claim_graded_after_its_dims() -> None:
    judge = _FakeClaimsJudge()
    template = _template(rubric=_RUBRIC)

    judged, judge_ms = await judge_witnessed_output(
        template=template,
        test_case=_case(template),
        output=_output(judged=True),
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.DOCUMENT,
        judge_service=_service(judge),
        concurrency=1,
        eval_run_id="run-1",
    )

    assert judged is not None and judge_ms is not None
    assert judged.claims is not None and set(judged.claims.tallies) == {_CRITERION}
    assert judge.asked == ["judge_reply", "claim_listing", "claim_truth", "claim_ground"]


class _TraceRefusingStore(InMemoryDocumentStore):
    """An in-memory store that refuses every trace write, as a store over its row limit would."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        if document.get("doc_type") == "eval_trace":
            raise RuntimeError("the row is over the store's size limit")
        super().upsert(document, if_match=if_match, create_only=create_only)


class _ReadFailsAfterTraceStore(InMemoryDocumentStore):
    """An in-memory store whose reads fail once a trace has landed, as a store whose reads went down would."""

    def __init__(self) -> None:
        super().__init__()
        self.traced = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        super().upsert(document, if_match=if_match, create_only=create_only)
        self.traced = self.traced or document.get("doc_type") == "eval_trace"

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.traced:
            raise RuntimeError("the store's reads are down")
        return super().get_with_etag(doc_id, scope_id)


async def test_a_read_failing_after_the_row_and_trace_landed_leaves_its_claims_pending_never_lost() -> None:
    result, trace = await _graded_cell()
    store = _ReadFailsAfterTraceStore()

    off_loop(EvalStorage(store).save_run_cell, result, trace)

    store.traced = False
    stored = off_loop(EvalStorage(store).load_eval_result, result.id, result.scope_id)
    assert stored is not None, "the write did not fail: the row and its trace landed"
    assert stored.claim_failures == {_CRITERION: "trace_pending"} and stored.claim_tallies == {}
    loaded = off_loop(EvalStorage(store).load_eval_trace, result.id, result.scope_id)
    assert loaded is not None and set(loaded.claim_grades) == {_CRITERION}, "its grade is stored: recoverable"
    assert claim_readings(_CRITERION, "trace_pending", grounded=True) == {}, "not yet known reads no reading"
    assert claim_readings(_CRITERION, "trace_lost", grounded=True) == {f"{_CRITERION}_lost": 1.0}


class _TraceRefusingThenReadlessStore(_TraceRefusingStore):
    """Refuses every trace write, and every read after one: a store that went down as the trace was written."""

    def __init__(self) -> None:
        super().__init__()
        self.refused = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        self.refused = self.refused or document.get("doc_type") == "eval_trace"
        super().upsert(document, if_match=if_match, create_only=create_only)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.refused:
            raise RuntimeError("the store's reads are down")
        return super().get_with_etag(doc_id, scope_id)


async def test_a_failed_trace_whose_loss_could_not_be_written_leaves_its_claims_pending() -> None:
    result, trace = await _graded_cell()
    store = _TraceRefusingThenReadlessStore()

    off_loop(EvalStorage(store).save_run_cell, result, trace)

    store.refused = False
    stored = off_loop(EvalStorage(store).load_eval_result, result.id, result.scope_id)
    assert stored is not None and stored.claim_failures == {_CRITERION: "trace_pending"}


async def _graded_cell() -> tuple[Any, Any]:
    return await _cell(_FakeClaimsJudge(), _template(), _FakeClaimsKind(_output()))


@pytest.mark.parametrize("write", ["save_run_cell", "save_eval_result"])
async def test_a_trace_that_is_not_stored_takes_the_claims_it_backs_with_it(write: str) -> None:
    result, trace = await _graded_cell()
    refusing = EvalStorage(_TraceRefusingStore())

    off_loop(getattr(refusing, write), result, trace)

    stored = off_loop(refusing.load_eval_result, result.id, result.scope_id)
    assert stored is not None
    assert stored.claim_tallies == {} and stored.claim_failures == {_CRITERION: "trace_lost"}
    assert stored.has_trace, "the cell produced a trace, which reads missing"


@pytest.mark.parametrize("write", ["save_run_cell", "save_eval_result"])
async def test_a_stored_trace_keeps_the_claims_it_backs(write: str) -> None:
    result, trace = await _graded_cell()
    storage, _ = memory_storage()

    off_loop(getattr(storage, write), result, trace)

    stored = off_loop(storage.load_eval_result, result.id, result.scope_id)
    assert stored is not None and stored.claim_tallies == result.claim_tallies and stored.claim_failures == {}
    loaded = off_loop(storage.load_eval_trace, result.id, result.scope_id)
    assert loaded is not None and loaded.claim_grades == trace.claim_grades


# =============================================================================
# The grade records its ground step's loss; the cell records every due criterion's fingerprint
# =============================================================================


async def test_a_grade_records_its_lost_ground_step_so_its_tally_is_recomputed_from_the_grade_alone() -> None:
    template = _template()
    lost, lost_trace = await _cell(
        _FakeClaimsJudge(script={"claim_ground": ["not json"]}), template, _FakeClaimsKind(_output())
    )
    whole, whole_trace = await _cell(_FakeClaimsJudge(), template, _FakeClaimsKind(_output()))
    (criterion,) = template.claim_criteria

    grade = lost_trace.claim_grades[_CRITERION]
    assert grade.ground_loss == "unreadable"
    assert grade_tally(grade, lost_trace.claim_evidence, criterion, derivation=None) == lost.claim_tallies[_CRITERION]
    assert whole_trace.claim_grades[_CRITERION].ground_loss is None
    assert (
        grade_tally(whole_trace.claim_grades[_CRITERION], whole_trace.claim_evidence, criterion, derivation=None)
        == (whole.claim_tallies[_CRITERION])
    )
    with pytest.raises(ValueError, match="grounded"):
        # The control: a grade that forgot its ground step's loss could not be read back at all.
        grade_tally(
            grade.model_copy(update={"ground_loss": None}), lost_trace.claim_evidence, criterion, derivation=None
        )


#: A candidate that said nothing in an output holding other parties' lines: claim evidence with no segment.
_SILENT = ClaimEvidence(segmentation="sentences-v1", segments=(), documents=None)


async def test_a_cell_whose_candidate_said_nothing_asks_nothing_and_is_tallied_silent_never_lost() -> None:
    """No segment: no claims call is made for any criterion, nothing is billed, the grade lists no claim, and the
    tally is silent with no unit and no claim, so the cell carries no claim reading and no loss. Undeclared documents
    under grounded_in cost a silent cell nothing either: there is no claim to ground."""
    judge = _FakeClaimsJudge()
    template = _template(_criterion("almanac.first"), _criterion("almanac.second", grounded_in=None))

    result, trace = await _cell(judge, template, _FakeClaimsKind(_output(evidence=_SILENT)))

    assert judge.asked == [], "a cell with nothing said makes no judge call"
    assert result.claim_failures == {} and result.judge_error is None
    for name, criterion in zip(("almanac.first", "almanac.second"), template.claim_criteria, strict=True):
        tally = result.claim_tallies[name]
        assert (tally.silent, tally.units, tally.by_judge, tally.ground_loss) == (True, 0, (), None)
        assert claim_readings(name, tally) == {}, "no reading, never a count of 0 false claims"
        grade = trace.claim_grades[name]
        assert (grade.calls, grade.claims) == ((), ())
        assert grade_tally(grade, trace.claim_evidence, criterion, derivation=None) == tally
    assert not [row for row in result.usage if row.role == "judge" and row.call_count]


async def test_a_silent_cell_is_neither_admitted_against_the_cap_nor_lost_to_it() -> None:
    # Each call is priced at $0.30, so a criterion that asked anything would not fit what the dims leave of $1.
    judge = _FakeClaimsJudge(usd_per_call=0.3)
    kind = _FakeClaimsKind(_output(judged=True, evidence=_SILENT), judged_artifact=JudgedArtifact.DOCUMENT)

    result, _, _ = await _judged_run(judge, _template(rubric=_RUBRIC), kind, cap_usd=1.0)

    assert result.claim_failures == {} and result.claim_tallies[_CRITERION].silent
    assert judge.asked == ["judge_reply"]


async def test_output_with_no_claim_evidence_is_refused_and_evidence_with_no_segment_is_not() -> None:
    """None is a kind that forgot, and no reader could tell it from silence: it is refused before the judge is paid.
    No segment is the candidate's silence, which is graded."""
    with pytest.raises(ValueError, match="rendered no claim_evidence"):
        await _cell(_FakeClaimsJudge(), _template(), _FakeClaimsKind(_output(evidence=None)))
    result, _ = await _cell(_FakeClaimsJudge(), _template(), _FakeClaimsKind(_output(evidence=_SILENT)))
    assert result.claim_tallies[_CRITERION].silent


async def test_undeclared_documents_under_grounded_in_lose_the_ground_step_alone_and_truth_stands() -> None:
    """Evidence with documents None under grounded_in is graded: no ground call is made, the grade and its tally
    record the ground step lost as documents_undeclared, and the truth counts are those of the declared cell."""
    judge = _FakeClaimsJudge()
    undeclared = ClaimEvidence(segmentation="sentences-v1", segments=_SEGMENTS, documents=None)

    result, trace = await _cell(judge, _template(), _FakeClaimsKind(_output(evidence=undeclared)))
    declared, _ = await _cell(_FakeClaimsJudge(), _template(), _FakeClaimsKind(_output()))

    assert judge.asked == ["claim_listing", "claim_truth"], "no ground call"
    tally = result.claim_tallies[_CRITERION]
    assert (tally.ground_loss, tally.documents_declared, tally.silent) == ("documents_undeclared", False, False)
    assert result.claim_failures == {}, "a ground loss is the tally's, never the criterion's"
    (counts,), (whole,) = tally.by_judge, declared.claim_tallies[_CRITERION].by_judge
    assert (counts.asserted, counts.asserted_contradicted) == (whole.asserted, whole.asserted_contradicted)
    assert counts.grounded_specific == 0
    readings = claim_readings(_CRITERION, tally)
    assert f"{_CRITERION}_grounded_share" not in readings and readings[f"{_CRITERION}_ground_lost"] == 1.0
    grade = trace.claim_grades[_CRITERION]
    assert grade.ground_loss == "documents_undeclared" and grade.claims[0].verdict == "supported"
    (criterion,) = _template().claim_criteria
    assert grade_tally(grade, trace.claim_evidence, criterion, derivation=None) == tally
    assert trace.claim_losses == {}, "a ground step never asked failed at nothing, so no call is kept for it"


async def test_a_cell_records_each_due_criterions_fingerprint_whether_it_was_graded_or_lost() -> None:
    judge = _FakeClaimsJudge(
        script={"claim_listing": [ConnectionError("the provider is unavailable"), _ANSWERS["claim_listing"]]}
    )
    template = _template(_criterion("almanac.first"), _criterion("almanac.second", guidance="Editions are claims."))

    result, _ = await _cell(judge, template, _FakeClaimsKind(_output()))

    expected = {c.name: criterion_fingerprint(c, *criterion_steps(c, judge)) for c in template.claim_criteria}
    assert result.claim_failures == {"almanac.first": "provider_error"}
    assert result.claim_fingerprints == expected and len(set(expected.values())) == 2
    assert result.claim_tallies["almanac.second"].criterion_fingerprint == expected["almanac.second"]


async def test_a_cell_cut_before_its_criteria_finished_still_records_their_fingerprints() -> None:
    judge = _FakeClaimsJudge(script={"claim_listing": ["hang"]})
    template = _template(_criterion("almanac.first"), _criterion("almanac.second"))

    result, _, _ = await _judged_run(judge, template, _FakeClaimsKind(_output()), cell_timeout_s=0.5)

    assert result.claim_failures == {"almanac.first": "deadline", "almanac.second": "deadline"}
    assert set(result.claim_fingerprints) == {"almanac.first", "almanac.second"}


# =============================================================================
# A claims call refused for the calling account stops the run
# =============================================================================


class _AccountRefused(Exception):
    """What a provider raises when it refuses the calling account: out of credit."""


def _account_describer(exc: BaseException) -> ProviderFailure:
    return ProviderFailure(
        "HTTP 402: no credit left",
        payload_withheld=False,
        owner="account" if isinstance(exc, _AccountRefused) else "provider",
    )


async def _run_refused(judge: _FakeClaimsJudge, template: EvalTemplate) -> tuple[Any, BaseException | None]:
    """One cell through the run loop under a describer that reads :class:`_AccountRefused` as the account's."""
    host = toyhost_host()
    case = _case(template)
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)
    kind = _FakeClaimsKind(_output())
    stopped: BaseException | None = None
    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=_service(judge, _account_describer),
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        )
    except AccountExhaustedError as error:
        stopped = error
    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    return result, stopped


async def test_a_claims_call_refused_for_the_account_loses_its_criterion_and_stops_the_run() -> None:
    judge = _FakeClaimsJudge(script={"claim_listing": [_AccountRefused("out of credit")]})

    result, stopped = await _run_refused(judge, _template())

    assert isinstance(stopped, AccountExhaustedError), "every later call would meet the same key"
    assert result.claim_failures == {_CRITERION: "provider_error"} and result.account_refused
    assert result.judge_error is None, "the cell keeps its trial: the loss is the criterion's"


async def test_a_ground_call_refused_for_the_account_stops_the_run_and_truth_stands() -> None:
    judge = _FakeClaimsJudge(script={"claim_ground": [_AccountRefused("out of credit")]})

    result, stopped = await _run_refused(judge, _template())

    assert isinstance(stopped, AccountExhaustedError)
    assert result.claim_tallies[_CRITERION].ground_loss == "provider_error"
    assert result.claim_tallies[_CRITERION].by_judge[0].asserted == 1, "truth stands"


async def test_a_claims_call_the_provider_declined_for_the_moment_stops_no_run() -> None:
    # The control: the same failure the describer charges to the provider, not the account.
    judge = _FakeClaimsJudge(script={"claim_listing": [ConnectionError("overloaded")]})

    result, stopped = await _run_refused(judge, _template())

    assert stopped is None and not result.account_refused
    assert result.claim_failures == {_CRITERION: "provider_error"}


# =============================================================================
# A witnessed cell of a kind graded by code is claim-graded; a defect's spend is counted
# =============================================================================


def _witnessed(judge: _FakeClaimsJudge) -> tuple[EvalHost, EvalTemplate, Any, EvalTestCase]:
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    template = _template()
    off_loop(host.storage.save_template, template)
    case = EvalTestCase(id="visit-3", scope_id="u", template_id=None, reference=_KEY)
    off_loop(host.storage.save_test_case, case)
    run = make_eval_run(
        id="run-witnessed",
        scope_id="u",
        template_id=None,
        candidate_kind=_KIND,
        test_case_ids=[case.id],
        apparatus_provenance="witnessed",
    )
    stamped = stamp_witnessed_judge(
        host,
        run,
        template,
        judge_model=_JUDGE,
        judged_artifact=JudgedArtifact.UNJUDGED,
        configured_max_cost_usd=5.0,
        enforcement_enabled=True,
    )
    return host, template, stamped, case


async def test_a_witnessed_cell_of_a_kind_graded_by_code_is_graded_by_its_templates_claim_criteria() -> None:
    judge = _FakeClaimsJudge()
    host, _, run, case = _witnessed(judge)

    result, trace = await record_witnessed_cell(
        host,
        run,
        case,
        _output(),
        k_iteration=1,
        result_id="witnessed-1",
        scored_at="2026-11-21T12:00:00+00:00",
        judged_artifact=JudgedArtifact.UNJUDGED,
        judging=WitnessedJudging(run.id),
    )

    assert set(result.claim_tallies) == {_CRITERION} and result.judge_error is None
    assert judge.asked == ["claim_listing", "claim_truth", "claim_ground"]
    assert trace.claim_grades[_CRITERION].criterion_fingerprint == result.claim_fingerprints[_CRITERION]


async def test_a_witnessed_claims_defect_counts_what_its_calls_cost_against_the_runs_ceiling() -> None:
    judge = _FakeClaimsJudge(script={"claim_truth": [_DEFECTIVE]})
    host, _, run, case = _witnessed(judge)
    judging = WitnessedJudging(run.id)

    with pytest.raises(CompletionProtocolDefect):
        await record_witnessed_cell(
            host,
            run,
            case,
            _output(),
            k_iteration=1,
            result_id="witnessed-1",
            scored_at="2026-11-21T12:00:00+00:00",
            judged_artifact=JudgedArtifact.UNJUDGED,
            judging=judging,
        )

    assert judging.spent([]) == [pytest.approx(0.004)], "the extraction and the defective truth call were paid for"


# =============================================================================
# Re-judging a finished cell's lost criteria
# =============================================================================


class _NoCallJudge(_FakeClaimsJudge):
    """A judge that answers no call: a re-judge that recovers a tally from its stored grade sends none."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        raise AssertionError("no claims call is made")


async def _stored_cell(
    first: _FakeClaimsJudge,
    template: EvalTemplate,
    again: _FakeClaimsJudge,
    *,
    trace_stored: bool = True,
    storage: EvalStorage | None = None,
    **run_fields: Any,
) -> tuple[EvalHost, Any, Any]:
    """One claims-only cell graded by ``first`` and stored under a finished run, on a host whose judge is ``again``;
    its trace stored too unless ``trace_stored`` is False, as a cell whose trace write never landed."""
    host = toyhost_host(clients=lambda role, model, *, temperature=None: again, storage=storage)
    case = _case(template)
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
        **{"judge_asked_settings": JUDGE_REQUEST_SETTINGS, **run_fields},
    )
    off_loop(host.storage.save_eval_run, run)
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
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _FakeClaimsKind(_output())}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )
    off_loop(host.storage.save_eval_result, outcome.result, outcome.trace if trace_stored else None)
    return host, outcome.result, outcome.trace


def _pending(result: Any) -> Any:
    """``result`` as a row written before its trace and never rewritten: each tallied criterion pending its trace."""
    return result.model_copy(
        update={"claim_tallies": {}, "claim_failures": dict.fromkeys(result.claim_tallies, "trace_pending")}
    )


@pytest.mark.parametrize("ground", ["kept", "lost"])
async def test_a_pending_criterion_is_settled_from_its_stored_grade_with_no_call(ground: str) -> None:
    first = _FakeClaimsJudge(script={"claim_ground": ["not json"]} if ground == "lost" else {})
    again = _NoCallJudge()
    host, result, trace = await _stored_cell(first, _template(), again)
    off_loop(host.storage.save_eval_result, _pending(result), trace)

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_tallies == result.claim_tallies and rejudged.claim_failures == {}
    expected_loss = "unreadable" if ground == "lost" else None
    assert rejudged.claim_tallies[_CRITERION].ground_loss == expected_loss, "a lost groundedness stays exactly that"
    assert rejudged.claim_tallies[_CRITERION].by_judge[0].asserted == 1, "and truth stands"
    assert rejudged.claim_rescores == [], "settling completes the cell's own write; nothing was re-judged"
    assert again.asked == []
    assert off_loop(host.storage.load_eval_result, result.id, "u") == rejudged


@pytest.mark.parametrize("moved", ["template_edited", "request_settings"])
async def test_settling_a_pending_criterion_passes_no_apparatus_check_since_it_asks_nothing(moved: str) -> None:
    template = _template()
    host, result, trace = await _stored_cell(
        _FakeClaimsJudge(),
        template,
        _NoCallJudge(),
        **({"judge_asked_settings": None} if moved == "request_settings" else {}),
    )
    if moved == "template_edited":
        off_loop(host.storage.save_template, template.model_copy(update={"updated_at": "2099-01-01T00:00:00+00:00"}))
    off_loop(host.storage.save_eval_result, _pending(result), trace)

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_tallies == result.claim_tallies and rejudged.claim_failures == {}


async def test_a_pending_criterion_whose_trace_was_never_stored_is_lost_once_its_run_has_ended() -> None:
    host, result, _ = await _stored_cell(_FakeClaimsJudge(), _template(), _NoCallJudge(), trace_stored=False)
    off_loop(host.storage.save_eval_result, _pending(result), None)
    assert off_loop(host.storage.load_eval_trace, result.id, "u") is None

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_failures == {_CRITERION: "trace_lost"} and rejudged.claim_tallies == {}
    assert rejudged.claim_rescores == []


def _spends_its_truth_repairs(**script: list[Any]) -> _FakeClaimsJudge:
    """A judge whose truth answers cite a fact never shown, under a budget of one repair: the criterion is lost."""
    truth = json.dumps({"claims": [{"claim": 0, "citations": [_MISCITED], "suspect": False}]})
    repaired = json.dumps({"citations": [_MISCITED], "suspect": False})
    return _FakeClaimsJudge(script={"claim_truth": [truth], "claim_truth_repair": [repaired], **script})


_ONE_REPAIR = _criterion(repairs=RepairBudget(rounds=2, max_items_per_cell=1))


async def test_a_rejudge_that_grades_a_lost_criterion_drops_its_stale_loss_record() -> None:
    template = _template(_ONE_REPAIR)
    host, result, trace = await _stored_cell(_spends_its_truth_repairs(), template, _FakeClaimsJudge())
    assert result.claim_failures == {_CRITERION: "repair_budget_spent"} and _CRITERION in trace.claim_losses

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_failures == {} and _CRITERION in rejudged.claim_tallies
    stored = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert _CRITERION in stored.claim_grades
    assert stored.claim_losses == {}, "the record described the loss the new grade replaces, so it goes with it"


async def test_a_rejudge_that_loses_a_criterion_again_records_what_spent_its_own_budget() -> None:
    template = _template(_ONE_REPAIR)
    first = _FakeClaimsJudge(script={"claim_listing": ["not json"]})
    host, result, trace = await _stored_cell(first, template, _spends_its_truth_repairs())
    assert result.claim_failures == {_CRITERION: "unreadable"}
    assert {call.step for call in trace.claim_losses[_CRITERION].calls} == {"extract"}

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_failures == {_CRITERION: "repair_budget_spent"}
    record = off_loop(host.storage.load_eval_trace, result.id, "u").claim_losses[_CRITERION]
    assert [(call.step, call.role, call.reason) for call in record.calls] == [
        ("extract", "extract", None),
        ("truth", "truth", None),
        ("truth", "repair", _MISCITED_FAULT),
    ], "the re-judge's own calls replace the first grade's"
    assert record.failure is not None and record.failure.startswith("the claim verifier would ask 1 truth claim(s)")


async def test_a_lost_criterion_is_graded_again_whole_and_a_completed_grade_is_never_asked_again() -> None:
    first = _FakeClaimsJudge(
        script={"claim_listing": [ConnectionError("the provider is unavailable"), _ANSWERS["claim_listing"]]}
    )
    again = _FakeClaimsJudge()
    template = _template(_criterion("almanac.first"), _criterion("almanac.second"))
    host, result, trace = await _stored_cell(first, template, again)
    assert result.claim_failures == {"almanac.first": "provider_error"}

    rejudged = await rejudge_result(host, result.id, "u")

    assert again.asked == ["claim_listing", "claim_truth", "claim_ground"], "one criterion, whole"
    assert set(rejudged.claim_tallies) == {"almanac.first", "almanac.second"} and rejudged.claim_failures == {}
    assert rejudged.claim_tallies["almanac.second"] == result.claim_tallies["almanac.second"]
    stored = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert stored is not None and set(stored.claim_grades) == {"almanac.first", "almanac.second"}
    assert stored.claim_grades["almanac.second"] == trace.claim_grades["almanac.second"], "never asked again"
    assert stored.claim_grades["almanac.first"].criterion_fingerprint == result.claim_fingerprints["almanac.first"]
    (rescore,) = rejudged.claim_rescores
    assert (rescore.criteria, rescore.prior, rescore.lost) == (
        ["almanac.first"],
        {"almanac.first": "provider_error"},
        {},
    )
    (row,) = rescore.usage
    assert (row.call_count, rescore.cost_usd) == (3, pytest.approx(0.006))
    assert rejudged.cost_usd == result.cost_usd, "a re-judge's spend is its own, never the cell's"


async def test_a_criterion_lost_again_keeps_its_new_loss() -> None:
    again = _FakeClaimsJudge(script={"claim_listing": ["not json"]})
    host, result, _ = await _stored_cell(
        _FakeClaimsJudge(script={"claim_listing": [ConnectionError("unavailable")]}), _template(), again
    )

    rejudged = await rejudge_result(host, result.id, "u")

    assert rejudged.claim_failures == {_CRITERION: "unreadable"}
    assert rejudged.claim_rescores[0].lost == {_CRITERION: "unreadable"}


@pytest.mark.parametrize("recorded", ["another", "none"])
async def test_a_criterion_whose_definition_moved_since_its_cell_is_refused_by_name_before_any_call(
    recorded: str,
) -> None:
    again = _FakeClaimsJudge()
    host, result, trace = await _stored_cell(
        _FakeClaimsJudge(script={"claim_listing": [ConnectionError("unavailable")]}), _template(), again
    )
    fingerprints = {_CRITERION: "0" * 64} if recorded == "another" else {}
    off_loop(host.storage.save_eval_result, result.model_copy(update={"claim_fingerprints": fingerprints}), trace)

    with pytest.raises(ValidationFailedError, match=repr(_CRITERION)) as refused:
        await rejudge_result(host, result.id, "u")

    assert "fingerprint" in refused.value.message
    assert again.asked == []


async def test_a_completed_grade_whose_ground_step_was_lost_is_never_re_judged() -> None:
    again = _FakeClaimsJudge()
    host, result, _ = await _stored_cell(_FakeClaimsJudge(script={"claim_ground": ["not json"]}), _template(), again)
    assert result.claim_tallies[_CRITERION].ground_loss == "unreadable"

    with pytest.raises(ValidationFailedError, match="lost no claim criterion"):
        await rejudge_result(host, result.id, "u")

    assert again.asked == []


# =============================================================================
# Claims calls replay by their prompt
# =============================================================================


async def test_a_replayed_claims_call_is_keyed_by_its_prompt_so_a_changed_document_is_another_call(
    tmp_path: Any,
) -> None:
    path = tmp_path / "claims.jsonl"
    template = _template()
    recorded, _ = await _cell(
        RecordedCompletions.record(_FakeClaimsJudge(), path, settings=None), template, _FakeClaimsKind(_output())
    )

    replayed, _ = await _cell(RecordedCompletions.replay(path, settings=None), template, _FakeClaimsKind(_output()))

    assert replayed.claim_tallies == recorded.claim_tallies
    reprinted = _NOTES.model_copy(update={"text": _NOTES.text + " It was reprinted in 1950."})
    moved = _output(evidence=_EVIDENCE.model_copy(update={"documents": (reprinted,)}))
    with pytest.raises(ReplayMiss):
        await _cell(RecordedCompletions.replay(path, settings=None), template, _FakeClaimsKind(moved))


# =============================================================================
# A run settles its own pending rows: when a holding ends, and when one takes it up
# =============================================================================


class _RewriteReadFailsOnceStore(InMemoryDocumentStore):
    """Fails the first read made once a trace has landed (or been refused), as a store whose reads blinked: the
    cell's row is left pending its trace, and every later read succeeds."""

    def __init__(self, *, refuse_traces: bool = False) -> None:
        super().__init__()
        self.refuse_traces = refuse_traces
        self.armed = False
        self.blinked = 0

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        if document.get("doc_type") == "eval_trace":
            self.armed = not self.blinked
            if self.refuse_traces:
                raise RuntimeError("the row is over the store's size limit")
        super().upsert(document, if_match=if_match, create_only=create_only)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        if self.armed:
            self.armed, self.blinked = False, self.blinked + 1
            raise RuntimeError("the store's reads blinked")
        return super().get_with_etag(doc_id, scope_id)


@pytest.mark.parametrize("trace", ["landed", "refused"])
async def test_a_row_left_pending_is_settled_when_its_holding_ends(trace: str) -> None:
    store = _RewriteReadFailsOnceStore(refuse_traces=trace == "refused")
    host = dataclasses.replace(toyhost_host(), storage=EvalStorage(store))
    template = _template()
    case = _case(template)
    run = make_eval_run(
        scope_id="u", template_id=template.id, candidate_kind=_KIND, test_case_ids=[case.id], judge_model=_JUDGE
    )
    off_loop(host.storage.save_eval_run, run)
    kind = _FakeClaimsKind(_output())

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=[case],
        judge_service=_service(_FakeClaimsJudge()),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
    )

    assert store.blinked == 1, "the control: the cell's own rewrite met the blink and left its row pending"
    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    if trace == "landed":
        assert result.claim_failures == {} and set(result.claim_tallies) == {_CRITERION}
    else:
        assert result.claim_failures == {_CRITERION: "trace_lost"} and result.claim_tallies == {}


class _ReadsAtInvokeKind(_FakeClaimsKind):
    """Reads one stored row as the cell it runs starts: what a holding has settled before it runs a cell. Given a
    trace, it then writes it, as an earlier holder's trace write that lands late, after the take-up has started."""

    def __init__(self, output: CandidateOutput, storage: EvalStorage, row_id: str, *, landing: Any = None) -> None:
        super().__init__(output)
        self.storage = storage
        self.row_id = row_id
        self.landing = landing
        self.seen: list[dict[str, str]] = []

    async def invoke(self, instance: object, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        row = await asyncio.to_thread(self.storage.load_eval_result, self.row_id, "u")
        assert row is not None
        self.seen.append(dict(row.claim_failures))
        if self.landing is not None:
            await asyncio.to_thread(self.storage.replace_eval_trace, self.landing, if_match=None)
        return self.output


@pytest.mark.parametrize("trace", ["stored", "lands_late"])
async def test_a_take_up_recovers_an_earlier_holdings_pending_rows_and_never_loses_one_whose_trace_may_land(
    trace: str,
) -> None:
    template = _template()
    host, earlier, earlier_trace = await _stored_cell(
        _FakeClaimsJudge(), template, _FakeClaimsJudge(), trace_stored=trace == "stored"
    )
    off_loop(
        host.storage.save_eval_result,
        _pending(earlier),
        off_loop(host.storage.load_eval_trace, earlier.id, "u") if trace == "stored" else None,
    )
    second = EvalTestCase(template_id=template.id, scope_id="u", reference=_KEY)
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
    kind = _ReadsAtInvokeKind(
        _output(), host.storage, earlier.id, landing=earlier_trace if trace == "lands_late" else None
    )
    (first_case,) = [c for c in off_loop(host.storage.query_test_cases, "u") if c.id == earlier.test_case_id]
    row = off_loop(host.storage.load_eval_result, earlier.id, "u")
    assert row is not None

    await execute_run(
        host,
        run=taken_up,
        template=template,
        test_cases=[first_case, second],
        judge_service=_service(_FakeClaimsJudge()),
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        stored=StoredCells(done=(StoredCell.from_result(row),)),
    )

    expected_at_start = {} if trace == "stored" else {_CRITERION: "trace_pending"}
    assert kind.seen == [expected_at_start], "recovered before the first cell, and never lost while its trace may land"
    settled = off_loop(host.storage.load_eval_result, earlier.id, "u")
    assert settled is not None and settled.claim_failures == {}, "scored once its trace landed, at the holding's end"
    assert settled.claim_tallies == earlier.claim_tallies


# =============================================================================
# A claims re-judge's writes are conditional, and say what they stored
# =============================================================================


class _TraceRacedStorage(EvalStorage):
    """The real store, except another re-judge rewrites the trace between this re-judge's read and its write."""

    def replace_eval_trace(self, trace: Any, *, if_match: str | None) -> None:
        super().replace_eval_trace(trace.model_copy(update={"claim_grades": {}}), if_match=None)
        super().replace_eval_trace(trace, if_match=if_match)


class _ResultRacedStorage(EvalStorage):
    """The real store, except another writer changes the result after this re-judge's trace has landed."""

    def replace_eval_result(self, result: Any, *, if_match: str | None) -> None:
        raise ConflictError(f"eval_result {result.id} was written since it was read", remedy="try again")


async def _lost_on_a_raced_store(raced: type[EvalStorage]) -> tuple[EvalHost, Any]:
    """A stored cell whose one criterion was lost, on a store that races the re-judge's write."""
    host, result, _ = await _stored_cell(
        _FakeClaimsJudge(script={"claim_listing": [ConnectionError("unavailable")]}),
        _template(),
        _FakeClaimsJudge(),
        storage=raced(InMemoryDocumentStore()),
    )
    return host, result


async def test_a_rejudge_whose_trace_another_rewrote_meanwhile_is_refused_and_stores_nothing() -> None:
    host, result = await _lost_on_a_raced_store(_TraceRacedStorage)
    before = off_loop(host.storage.load_eval_result, result.id, "u")

    with pytest.raises(ConflictError, match="trace changed while it was being re-judged; nothing was stored"):
        await rejudge_result(host, result.id, "u")

    stored_trace = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert stored_trace is not None and stored_trace.claim_grades == {}, "the other writer's trace stands"
    assert off_loop(host.storage.load_eval_result, result.id, "u") == before


async def test_a_rejudge_whose_result_write_loses_after_its_trace_landed_says_so() -> None:
    host, result = await _lost_on_a_raced_store(_ResultRacedStorage)
    before = off_loop(host.storage.load_eval_result, result.id, "u")

    with pytest.raises(ConflictError, match="its trace holds the new grades and its result was not rewritten"):
        await rejudge_result(host, result.id, "u")

    stored_trace = off_loop(host.storage.load_eval_trace, result.id, "u")
    assert stored_trace is not None and _CRITERION in stored_trace.claim_grades
    assert off_loop(host.storage.load_eval_result, result.id, "u") == before
