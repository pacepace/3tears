"""The blind read's actions: draw one, serve its sheet, take a person's answer, have its model read it, close it,
and read it back.

Each is the catalogue's door onto one operation of :mod:`threetears.evals.ops.blind_reads`, which holds the rules
(who may answer, what a sheet may show, what close writes). The engine's catalogue includes them
(:func:`~threetears.evals.actions.engine.eval_catalogue`).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Annotated, Any, Literal, get_args

from pydantic import Field

from threetears.evals.actions import render
from threetears.evals.actions.catalogue import Action, Caller, ConformanceStage
from threetears.evals.actions.params import CampaignId
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.ops.blind_reads import (
    BlindReadLine,
    BlindReadListing,
    blind_read_answer,
    blind_read_close,
    blind_read_create,
    blind_read_get,
    blind_read_review,
    blind_read_sheet,
    blind_reads_list,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobsStarted
from threetears.evals.ops.runs import PERSON_DESCRIPTION
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.blind_read import BlindSheet

#: How an action names a read.
ReadId = Annotated[str, Field(min_length=1, description="The blind read, by the id blind_read_create returned.")]


class BlindReadCreateParams(EvalBaseModel):
    """``blind_read_create``."""

    campaign_id: Annotated[str | None, get_args(CampaignId)[1]] = None
    run_ids: Annotated[
        list[str] | None,
        Field(description="Runs, by id, as runs_list names them; every one in the caller's scope."),
    ] = None
    candidate: Annotated[
        str | None,
        Field(min_length=1, description="The candidate arm, by its label or its variant key; never the control."),
    ] = None
    criteria: Annotated[
        list[str] | None,
        Field(description="Criteria, by dimension name; omitted takes every dimension a result was judged on."),
    ] = None
    planned_items: Annotated[
        int | None,
        Field(ge=1, description="The items the preference is read over, fixed now; omitted is all, up to 30."),
    ] = None
    read_name: Annotated[str | None, Field(min_length=1, description="What the read is called.")] = None
    reviewer: Annotated[
        str | None,
        Field(
            min_length=1,
            description="A model to read the sheet too, by id, declared now or never; not the judge's own model.",
        ),
    ] = None


class BlindReadSheetParams(EvalBaseModel):
    """``blind_read_sheet``."""

    read_id: ReadId
    rater: Annotated[
        str | None,
        Field(min_length=1, description="Whose progress to show (answered, next), as the host names them."),
    ] = None


class BlindReadAnswerParams(EvalBaseModel):
    """``blind_read_answer``."""

    read_id: ReadId
    item: Annotated[str, Field(min_length=1, description="The item answered, as the sheet names it.")]
    person: Annotated[str, Field(min_length=1, description=PERSON_DESCRIPTION)]
    verdicts: Annotated[
        dict[str, str],
        Field(
            description=(
                "A verdict per output and criterion the item shows, keyed '<slot>/<criterion>' (e.g. "
                "'A/answer.grounded'), each answered as the sheet's levels word it: pass or fail, or 1 to 5."
            )
        ),
    ] = Field(default_factory=dict)
    prefer: Annotated[
        Literal["A", "B", "tie"] | None,
        Field(description="Of an item showing two outputs: the better slot, or tie."),
    ] = None
    skip: Annotated[
        bool, Field(description="The person passes on the item: no verdict or preference. A skip is an answer.")
    ] = False
    answer_reason: Annotated[str, Field(description="The person's own words for the answer, where given.")] = ""


class BlindReadParams(EvalBaseModel):
    """``blind_read_review``, ``blind_read_close`` and ``blind_read_get``: the read."""

    read_id: ReadId


class BlindReadsListParams(EvalBaseModel):
    """``blind_reads_list``."""

    campaign_id: Annotated[str | None, get_args(CampaignId)[1]] = None


async def _create(host: OpsHost, caller: Caller, params: BlindReadCreateParams) -> BlindReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            blind_read_create,
            created_by=caller.identity,
            campaign_id=params.campaign_id,
            run_ids=params.run_ids,
            candidate=params.candidate,
            criteria=params.criteria,
            planned_items=params.planned_items,
            name=params.read_name,
            reviewer=params.reviewer,
        ),
        eval_host,
        caller.scope_id,
    )


async def _sheet(host: OpsHost, caller: Caller, params: BlindReadSheetParams) -> BlindSheet:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(blind_read_sheet, rater=params.rater),
        eval_host,
        params.read_id,
        caller.scope_id,
    )


async def _answer(host: OpsHost, caller: Caller, params: BlindReadAnswerParams) -> BlindReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(
            blind_read_answer,
            collected_by=caller.identity,
            person=params.person,
            item=params.item,
            verdicts=params.verdicts,
            prefer=params.prefer,
            skipped=params.skip,
            reason=params.answer_reason,
        ),
        eval_host,
        params.read_id,
        caller.scope_id,
    )


async def _review(host: OpsHost, caller: Caller, params: BlindReadParams) -> JobsStarted:
    return await blind_read_review(host, params.read_id, caller.scope_id)


async def _close(host: OpsHost, caller: Caller, params: BlindReadParams) -> BlindReadLine:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(blind_read_close, closed_by=caller.identity),
        eval_host,
        params.read_id,
        caller.scope_id,
    )


async def _get(host: OpsHost, caller: Caller, params: BlindReadParams) -> BlindReadLine:
    eval_host = host.eval_host
    return await run_blocking(eval_host.blocking_executor, blind_read_get, eval_host, params.read_id, caller.scope_id)


async def _list(host: OpsHost, caller: Caller, params: BlindReadsListParams) -> BlindReadListing:
    eval_host = host.eval_host
    return await run_blocking(
        eval_host.blocking_executor,
        partial(blind_reads_list, campaign_id=params.campaign_id),
        eval_host,
        caller.scope_id,
    )


async def _review_path(stage: ConformanceStage) -> Mapping[str, Any]:
    """A read over the sample's finished run, its sheet to be read by the sample's second judge."""
    run_id = await stage.finished_run()
    eval_host = stage.host.eval_host
    line = await run_blocking(
        eval_host.blocking_executor,
        partial(
            blind_read_create,
            created_by=stage.caller.identity,
            run_ids=[run_id],
            reviewer=stage.sample.second_judge_model,
        ),
        eval_host,
        stage.caller.scope_id,
    )
    return {"read_id": line.read_id}


def blind_read_actions(workflow: str) -> tuple[Action, ...]:
    """The blind read's seven actions, in the order help lists them.

    :param workflow: The help index's group they are listed under.
    :return: The actions.
    """
    read_id = "blind-0193a1b2c3d4"
    return (
        Action(
            name="blind_read_create",
            summary="Draw a blind read: a campaign's judged outputs in a seeded random order, for people to answer.",
            workflow=workflow,
            permission="write",
            params=BlindReadCreateParams,
            result=BlindReadLine,
            handler=_create,
            render=render.render_model,
            example={"campaign_id": "0193a1b2-campaign", "candidate": "v2", "planned_items": 20},
            detail=(
                "Over a campaign's control and one candidate (a pair read: each item shows the two outputs of one "
                "case in random slots, and asks which is better), or over runs of one arm (each item shows one "
                "output). There is no seed: it is derived from the outputs, the arms and how many reads were drawn "
                "over them before, which earlier_reads states. A reviewer model is declared here or never."
            ),
        ),
        Action(
            name="blind_read_sheet",
            summary="Read a blind read's sheet: the items in order, content and criteria only, for a rater to answer.",
            workflow=workflow,
            permission="read",
            params=BlindReadSheetParams,
            result=BlindSheet,
            handler=_sheet,
            render=render.render_model,
            example={"read_id": read_id, "rater": "ana@example.com"},
            detail=(
                "Show nothing but what it carries, items in order and next first, slots as labelled; a skip is an "
                "answer. It names no arm, run, result or case and carries no judge score and no count of answers."
            ),
        ),
        Action(
            name="blind_read_answer",
            summary="Record a person's answer to one item of a blind read, relayed by a caller the host attests.",
            workflow=workflow,
            permission="write",
            params=BlindReadAnswerParams,
            result=BlindReadLine,
            handler=_answer,
            render=render.render_model,
            example={
                "read_id": read_id,
                "item": "3f9a0c1d2e4b",
                "person": "ana@example.com",
                "verdicts": {"A/chat.tone": "pass", "B/chat.tone": "fail"},
                "prefer": "A",
            },
            detail=(
                "The host decides whether this caller relays this person's answers; without its word the call is "
                "refused (refused, 403) and nothing is written. The person's earlier answer to the item is replaced. "
                "A model's answers come only from the read's own review (blind_read_review)."
            ),
        ),
        Action(
            name="blind_read_review",
            summary="Have a blind read's declared reviewer model answer its sheet, priced and capped, as a job.",
            workflow=workflow,
            permission="spend",
            params=BlindReadParams,
            result=JobsStarted,
            handler=_review,
            render=render.render_jobs_started,
            long_running=True,
            happy_path=_review_path,
            example={"read_id": read_id},
            detail=(
                "One call per item the model has not answered, each in a fresh context with the sheet's content "
                "alone; admitted against the out-of-run cap first, ledgered as reviewer spend. Its answers are a "
                "model's, read apart from people's and never as theirs."
            ),
        ),
        Action(
            name="blind_read_close",
            summary="Close a blind read: its answers become ratings, and what they say of the judge is read.",
            workflow=workflow,
            permission="write",
            params=BlindReadParams,
            result=BlindReadLine,
            handler=_close,
            render=render.render_model,
            example={"read_id": read_id},
            detail=(
                "A person's answers become their ratings (naming the read and who relayed them), a model's an "
                "agent's. Later answers are refused. Closed before anyone answered every planned item, its "
                "preference is read by the anytime test, valid wherever it stopped. A close whose ratings were not "
                "all written (the store refused one) is completed by closing again, which writes the ones missing."
            ),
        ),
        Action(
            name="blind_read_get",
            summary="Read one blind read: what was drawn, who has answered how far, and once closed what it found.",
            workflow=workflow,
            permission="read",
            params=BlindReadParams,
            result=BlindReadLine,
            handler=_get,
            render=render.render_model,
            example={"read_id": read_id},
        ),
        Action(
            name="blind_reads_list",
            summary="List the scope's blind reads, oldest first, or those drawn over one campaign.",
            workflow=workflow,
            permission="read",
            params=BlindReadsListParams,
            result=BlindReadListing,
            handler=_list,
            render=render.render_model,
            example={"campaign_id": "0193a1b2-campaign"},
        ),
    )


__all__ = ["blind_read_actions"]
