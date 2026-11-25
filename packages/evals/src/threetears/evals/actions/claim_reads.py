"""The claim read's actions: draw one, serve its sheet, take a person's answer, close it, have a judge grade its
claims again, and read it back.

Each is the catalogue's door onto one operation of :mod:`threetears.evals.ops.claim_reads`, which holds the rules
(who may answer, what a sheet may show, when figures are read, what a repeat may spend). The engine's catalogue
includes them (:func:`~threetears.evals.actions.engine.eval_catalogue`).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from functools import partial
from typing import Annotated, Any, get_args

from pydantic import Field

from threetears.evals.actions import render
from threetears.evals.actions.catalogue import Action, Caller, ConformanceStage
from threetears.evals.actions.params import CampaignId
from threetears.evals.kernel.claims.grading import claim_evidence_digest, criterion_fingerprint, criterion_steps
from threetears.evals.kernel.claims.grading import grade_tally
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.claim_reads import (
    ClaimReadLine,
    ClaimReadListing,
    ClaimSheet,
    claim_read_answer,
    claim_read_close,
    claim_read_create,
    claim_read_get,
    claim_read_repeat,
    claim_read_sheet,
    claim_reads_list,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobsStarted
from threetears.evals.ops.runs import PERSON_DESCRIPTION
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema import Segment
from threetears.evals.schema.claim_read import ClaimQuestion, ClaimStratifier, ClaimStratum
from threetears.evals.schema.models import (
    ClaimCall,
    ClaimCriterion,
    ClaimEvidence,
    ClaimGrade,
    EvalTrace,
    GradedClaim,
    PageBudget,
    RepairBudget,
    eval_trace_doc_id,
)

#: How an action names a read.
ClaimReadId = Annotated[str, Field(min_length=1, description="The claim read, by the id claim_read_create returned.")]


class ClaimReadCreateParams(EvalBaseModel):
    """``claim_read_create``."""

    claim_criterion: Annotated[
        str, Field(min_length=1, description="A claim criterion, by name, as the runs' template declares it.")
    ]
    campaign_id: Annotated[str | None, get_args(CampaignId)[1]] = None
    run_ids: Annotated[
        list[str] | None,
        Field(description="Runs, by id, as runs_list names them; every one in the caller's scope."),
    ] = None
    questions: Annotated[
        list[ClaimQuestion] | None,
        Field(
            description="What each item asks: true, grounded, is_claim. Omitted asks all three, grounded only where "
            "the grades read groundedness."
        ),
    ] = None
    stratify_by: Annotated[
        ClaimStratifier,
        Field(description="The grader's answer the draw is stratified by: truth (its verdict) or grounded."),
    ] = "truth"
    claim_items: Annotated[
        int | None,
        Field(ge=1, description="The claims planned in all, spread over the strata; omitted is the default plan."),
    ] = None
    per_stratum: Annotated[
        dict[ClaimStratum, int] | None,
        Field(description="The claims planned per stratum, fixed now; a stratum left out plans none."),
    ] = None
    read_name: Annotated[str | None, Field(min_length=1, description="What the read is called.")] = None


class ClaimReadSheetParams(EvalBaseModel):
    """``claim_read_sheet``."""

    claim_read_id: ClaimReadId
    rater: Annotated[
        str | None,
        Field(min_length=1, description="Whose progress to show (answered, next), as the host names them."),
    ] = None


class ClaimReadAnswerParams(EvalBaseModel):
    """``claim_read_answer``."""

    claim_read_id: ClaimReadId
    item: Annotated[str, Field(min_length=1, description="The item answered, as the sheet names it.")]
    person: Annotated[str, Field(min_length=1, description=PERSON_DESCRIPTION)]
    claim_answers: Annotated[
        dict[str, str],
        Field(description="Each question the sheet asks, answered pass or fail (e.g. {'true': 'fail'})."),
    ] = Field(default_factory=dict)
    skip: Annotated[
        bool, Field(description="The person passes on the item: no verdict or preference. A skip is an answer.")
    ] = False
    answer_reason: Annotated[str, Field(description="The person's own words for the answer, where given.")] = ""


class ClaimReadParams(EvalBaseModel):
    """``claim_read_close`` and ``claim_read_get``: the read."""

    claim_read_id: ClaimReadId


class ClaimReadRepeatParams(EvalBaseModel):
    """``claim_read_repeat``."""

    claim_read_id: ClaimReadId
    repeat_judge: Annotated[
        str | None,
        Field(min_length=1, description="The judge's model, by id; omitted asks each run's own judge again."),
    ] = None


class ClaimReadsListParams(EvalBaseModel):
    """``claim_reads_list``."""

    campaign_id: Annotated[str | None, get_args(CampaignId)[1]] = None


async def _create(host: OpsHost, caller: Caller, params: ClaimReadCreateParams) -> ClaimReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            claim_read_create,
            created_by=caller.identity,
            criterion=params.claim_criterion,
            campaign_id=params.campaign_id,
            run_ids=params.run_ids,
            questions=params.questions,
            stratify_by=params.stratify_by,
            planned_items=params.claim_items,
            planned=params.per_stratum,
            name=params.read_name,
        ),
        eval_host,
        caller.scope_id,
    )


async def _sheet(host: OpsHost, caller: Caller, params: ClaimReadSheetParams) -> ClaimSheet:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(claim_read_sheet, rater=params.rater),
        eval_host,
        params.claim_read_id,
        caller.scope_id,
    )


async def _answer(host: OpsHost, caller: Caller, params: ClaimReadAnswerParams) -> ClaimReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            claim_read_answer,
            collected_by=caller.identity,
            person=params.person,
            item=params.item,
            answers=params.claim_answers,
            skipped=params.skip,
            reason=params.answer_reason,
        ),
        eval_host,
        params.claim_read_id,
        caller.scope_id,
    )


async def _close(host: OpsHost, caller: Caller, params: ClaimReadParams) -> ClaimReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(claim_read_close, closed_by=caller.identity),
        eval_host,
        params.claim_read_id,
        caller.scope_id,
    )


async def _repeat(host: OpsHost, caller: Caller, params: ClaimReadRepeatParams) -> JobsStarted:
    return await claim_read_repeat(host, params.claim_read_id, caller.scope_id, judge_model=params.repeat_judge)


async def _get(host: OpsHost, caller: Caller, params: ClaimReadParams) -> ClaimReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor, claim_read_get, eval_host, params.claim_read_id, caller.scope_id
    )


async def _list(host: OpsHost, caller: Caller, params: ClaimReadsListParams) -> ClaimReadListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(claim_reads_list, campaign_id=params.campaign_id),
        eval_host,
        caller.scope_id,
    )


#: The claim criterion a conformance read is drawn under, and the one sentence every graded cell says.
_CONFORMANCE_CRITERION = ClaimCriterion(
    name="conformance.claims",
    judge="builtin",
    pages=PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
    repairs=RepairBudget(rounds=1, max_items_per_cell=2),
)
_CONFORMANCE_SENTENCE = "The reading room opened in 1921."


async def _repeat_path(stage: ConformanceStage) -> Mapping[str, Any]:
    """A claim read over a copy of the sample's finished run graded under a claim criterion, each cell's one claim
    stored as unverifiable: a repeat of it asks the judge about every claim, whatever the host's kind renders.
    Arranged in the stage host's own store."""
    run_id = await stage.finished_run()
    eval_host = stage.host.eval_host
    run = await run_blocking(
        eval_host.blocking_executor, eval_host.storage.load_eval_run, run_id, stage.caller.scope_id
    )
    assert run is not None and run.judge_model is not None  # the stage's run finished judged
    client = eval_host.completion_clients("a conformance claim read")("judge", run.judge_model)
    try:
        fingerprint = criterion_fingerprint(_CONFORMANCE_CRITERION, *criterion_steps(_CONFORMANCE_CRITERION, client))
    finally:
        await client.aclose()
    graded = await run_blocking(
        eval_host.blocking_executor, _graded_copy, eval_host, run_id, stage.caller.scope_id, fingerprint
    )
    line = await run_blocking(
        eval_host.blocking_executor,
        partial(
            claim_read_create,
            created_by=stage.caller.identity,
            criterion=_CONFORMANCE_CRITERION.name,
            run_ids=[graded],
            questions=["true"],
        ),
        eval_host,
        stage.caller.scope_id,
    )
    return {"claim_read_id": line.read_id}


def _graded_copy(eval_host: EvalHost, run_id: str, scope_id: str, fingerprint: str) -> str:
    """Store a copy of the run, its template declaring the conformance criterion and each result's trace holding a
    grade of one claim under it; the copy's id. Blocking."""
    storage = eval_host.storage
    run = storage.load_eval_run(run_id, scope_id)
    assert run is not None and run.template_id is not None
    template = storage.load_template(run.template_id, scope_id)
    assert template is not None
    copied = template.model_copy(
        update={"id": str(uuid.uuid7()), "claim_criteria": [*template.claim_criteria, _CONFORMANCE_CRITERION]}
    )
    storage.save_template(copied)
    copy = run.model_copy(update={"id": str(uuid.uuid7()), "template_id": copied.id})
    storage.save_eval_run(copy)
    text = _CONFORMANCE_SENTENCE
    evidence = ClaimEvidence(segmentation="conformance", segments=(Segment(text=text, unit=0),), documents=())
    claim = GradedClaim.model_validate(
        {
            "segment": 0,
            "unit": 0,
            "claim": text,
            "excerpt": text,
            "start": 0,
            "end": len(text),
            "scope": "world",
            "specific": True,
            "hedged": False,
            "attributed": False,
            "restates": None,
            "verdict": "unverifiable",
            "decided_by": 0,
        }
    )
    grade = ClaimGrade(
        criterion_fingerprint=fingerprint,
        claim_evidence_digest=claim_evidence_digest(evidence),
        calls=(ClaimCall(role="truth", served_model=run.judge_model, judge_temperature=0.0),),
        evidence=(),
        claims=(claim,),
    )
    tally = grade_tally(grade, evidence, _CONFORMANCE_CRITERION, derivation=None)
    name = _CONFORMANCE_CRITERION.name
    for result in storage.query_eval_results(scope_id, run_id=run_id):
        stored = result.model_copy(
            update={
                "id": str(uuid.uuid7()),
                "eval_run_id": copy.id,
                "claim_tallies": {**result.claim_tallies, name: tally},
                "claim_fingerprints": {**result.claim_fingerprints, name: fingerprint},
                "has_trace": True,
            }
        )
        trace = EvalTrace(
            id=eval_trace_doc_id(stored.id),
            scope_id=scope_id,
            result_id=stored.id,
            eval_run_id=copy.id,
            claim_evidence=evidence,
            claim_grades={name: grade},
        )
        storage.save_eval_result(stored, trace)
    return copy.id


def claim_read_actions(workflow: str) -> tuple[Action, ...]:
    """The claim read's seven actions, in the order help lists them.

    :param workflow: The help index's group they are listed under.
    :return: The actions.
    """
    read_id = "claimread-0193a1b2c3d4"
    return (
        Action(
            name="claim_read_create",
            summary="Draw a claim read: graded claims drawn by the grader's verdict, for people to label blind.",
            workflow=workflow,
            permission="write",
            params=ClaimReadCreateParams,
            result=ClaimReadLine,
            handler=_create,
            render=render.render_model,
            example={"claim_criterion": "answers.claims", "campaign_id": "0193a1b2-campaign", "claim_items": 200},
            detail=(
                "Over a campaign's runs or named runs, one arm per variant key. Each stratum of the grader's answer "
                "is drawn at its planned rate, at most two claims a cell. There is no seed: it is derived from the "
                "claims, the stratifier, the questions and how many reads were drawn over them before, which "
                "earlier_reads states."
            ),
        ),
        Action(
            name="claim_read_sheet",
            summary="Read a claim read's sheet: each claim, its passage and the evidence shown, for a rater to label.",
            workflow=workflow,
            permission="read",
            params=ClaimReadSheetParams,
            result=ClaimSheet,
            handler=_sheet,
            render=render.render_model,
            example={"claim_read_id": read_id, "rater": "ana@example.com"},
            detail=(
                "Show nothing but what it carries, items in order and next first; a skip is an answer. It names no "
                "arm, stratum, run, result or case and carries no verdict and no count of answers."
            ),
        ),
        Action(
            name="claim_read_answer",
            summary="Record a person's answer to one claim of a claim read, relayed by a caller the host attests.",
            workflow=workflow,
            permission="write",
            params=ClaimReadAnswerParams,
            result=ClaimReadLine,
            handler=_answer,
            render=render.render_model,
            example={
                "claim_read_id": read_id,
                "item": "3f9a0c1d2e4b",
                "person": "ana@example.com",
                "claim_answers": {"true": "fail", "is_claim": "pass"},
            },
            detail=(
                "The host decides whether this caller relays this person's answers; without its word the call is "
                "refused (refused, 403) and nothing is written. The person's earlier answer to the item is replaced. "
                "A closed read takes no answer."
            ),
        ),
        Action(
            name="claim_read_close",
            summary="Close a claim read: its figures and checks are read, and later answers are refused.",
            workflow=workflow,
            permission="write",
            params=ClaimReadParams,
            result=ClaimReadLine,
            handler=_close,
            render=render.render_model,
            example={"claim_read_id": read_id},
            detail=(
                "Once, at the look the plan fixed: kappa against the calibration bar, recall, precision and rate "
                "bias per arm, and the lean test across arms, each weighted to the claims drawn from. A read closed "
                "with claims nobody answered says so."
            ),
        ),
        Action(
            name="claim_read_repeat",
            summary="Have a judge grade a claim read's claims again over their stored evidence, priced and capped.",
            workflow=workflow,
            permission="spend",
            params=ClaimReadRepeatParams,
            result=JobsStarted,
            handler=_repeat,
            render=render.render_jobs_started,
            long_running=True,
            happy_path=_repeat_path,
            example={"claim_read_id": read_id},
            detail=(
                "The run's own judge (the grader against itself) or another (a second judge). Claims are kept as "
                "listed and the web read again over the stored windows; nothing is searched. Priced at the worst "
                "case its repair budget allows and refused over the out-of-run cap before any call."
            ),
        ),
        Action(
            name="claim_read_get",
            summary="Read one claim read: what was drawn, who answered how far, its repeats, and once closed its figures.",
            workflow=workflow,
            permission="read",
            params=ClaimReadParams,
            result=ClaimReadLine,
            handler=_get,
            render=render.render_model,
            example={"claim_read_id": read_id},
        ),
        Action(
            name="claim_reads_list",
            summary="List the scope's claim reads, oldest first, or those drawn over one campaign.",
            workflow=workflow,
            permission="read",
            params=ClaimReadsListParams,
            result=ClaimReadListing,
            handler=_list,
            render=render.render_model,
            example={"campaign_id": "0193a1b2-campaign"},
        ),
    )


__all__ = ["claim_read_actions"]
