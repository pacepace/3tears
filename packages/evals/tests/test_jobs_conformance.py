"""The jobs conformance kit over the package's whole catalogue, and the kit catching what it exists to catch.

No transport call waits on a model: every action's call makes no completion request, and every spend action's
happy path returns ``JobsStarted`` and its job makes at least one. The kit (:mod:`threetears.evals.testing`) is run
over every engine action here, on a judged host whose client factory it guards; and against host actions built to
break it — one that calls its model in the call, one whose happy path never reaches its work — to show it is not
green by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.actions import Action, Caller, ConformanceStage, eval_catalogue
from threetears.evals.ops import JobsStarted, OpsHost, await_job, job_poll, start_action_job
from pydantic import Field

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.storage import DocTypeRoute, InMemoryDocumentStore, RoutedDocumentStore
from threetears.evals.testing import (
    CompletionGuard,
    JobsConformanceCase,
    JobsConformanceHost,
    JobsConformanceFailure,
    jobs_conformance_cases,
)
from packages.evals.tests.jobs_conformance_support import CONFORMANCE_SCOPE, conformance_host


class _DraftParams(EvalBaseModel):
    """A host action's parameters."""

    subject_area: str = Field(default="tone", description="What the rubric is to measure.")


class _Draft(EvalBaseModel):
    """What a drafting host action produces."""

    text: str = Field(description="The draft.")


async def _draft_in_job(host: OpsHost, caller: Caller, params: _DraftParams) -> JobsStarted:
    """A conforming host action: refuses and prices nothing here, and drafts with a model in its job."""
    clients = host.eval_host.completion_clients("a draft")

    async def draft() -> _Draft:
        client = clients("proposer", None)
        completion = await client.generate(system="Draft a rubric.", user=params.subject_area)
        await client.aclose()
        return _Draft(text=str(completion.content))

    return await start_action_job(host, caller.scope_id, action="rubric_propose", run=draft, label=params.subject_area)


async def _draft_in_call(host: OpsHost, caller: Caller, params: _DraftParams) -> JobsStarted:
    """A non-conforming host action: it drafts with its model in the call, then starts a job to look like one."""
    client = host.eval_host.completion_clients("a draft")("proposer", None)
    completion = await client.generate(system="Draft a rubric.", user=params.subject_area)

    async def keep() -> _Draft:
        return _Draft(text=str(completion.content))

    return await start_action_job(host, caller.scope_id, action="rubric_propose", run=keep)


async def _waits_on_its_job(host: OpsHost, caller: Caller, params: _DraftParams) -> JobsStarted:
    """A non-conforming host action: it drafts in a job, then waits for the job before returning."""
    started = await _draft_in_job(host, caller, params)
    await await_job(host, started, caller.scope_id)
    return started


async def _nothing_in_job(host: OpsHost, caller: Caller, params: _DraftParams) -> JobsStarted:
    """A host action whose job makes no request on the parameters its happy path names."""

    async def nothing() -> _Draft:
        return _Draft(text="")

    return await start_action_job(host, caller.scope_id, action="rubric_propose", run=nothing)


async def _topic(_stage: ConformanceStage) -> Mapping[str, Any]:
    return {"subject_area": "tone"}


def _host_action(handler: Any) -> Action:
    return Action(
        name="rubric_propose",
        summary="Draft a rubric for a person to review, as a job.",
        workflow="Curate",
        permission="spend",
        params=_DraftParams,
        result=JobsStarted,
        handler=handler,
        render=lambda started: str(started),
        example={"subject_area": "tone"},
        long_running=True,
        happy_path=_topic,
    )


def _cases(handler: Any, *, call_return_timeout_s: float = 10.0) -> dict[str, JobsConformanceCase]:
    catalogue = eval_catalogue((_host_action(handler),))
    cases = jobs_conformance_cases(catalogue, job_timeout_s=10.0, call_return_timeout_s=call_return_timeout_s)
    return {case.name: case for case in cases}


@pytest.mark.parametrize(
    "case", jobs_conformance_cases(eval_catalogue(), job_timeout_s=30.0), ids=lambda case: case.name
)
async def test_every_engine_action_keeps_the_job_rule(case: JobsConformanceCase) -> None:
    await case.check(conformance_host)


def test_every_engine_spend_action_has_a_happy_path_case() -> None:
    names = {case.name for case in jobs_conformance_cases(eval_catalogue())}
    spend = {action.name for action in eval_catalogue().actions if action.permission == "spend"}
    assert spend == {
        "run_launch",
        "sweep_launch",
        "analysis_generate",
        "judge_second",
        "judge_drift_check",
        "judge_temperature",
    }
    assert {f"{name}.happy_path" for name in spend} <= names
    assert all(action.long_running for action in eval_catalogue().actions if action.permission == "spend")


async def test_a_conforming_host_action_passes_and_its_job_returns_its_draft() -> None:
    cases = _cases(_draft_in_job)
    await cases["rubric_propose.call"].check(conformance_host)
    await cases["rubric_propose.happy_path"].check(conformance_host)

    built = conformance_host(CompletionGuard())
    started = await _draft_in_job(built.host, built.caller, _DraftParams())
    (job,) = started.jobs
    assert job.job_id.startswith("action:") and job.kind == "action"
    while not (status := await job_poll(built.host, job.job_id, CONFORMANCE_SCOPE)).done:
        pass
    assert status.state == "completed" and status.result == {"text": "{}"}


async def test_the_kit_catches_a_spend_action_that_calls_its_model_in_the_call() -> None:
    cases = _cases(_draft_in_call)

    with pytest.raises(JobsConformanceFailure, match="its call made a completion request"):
        await cases["rubric_propose.happy_path"].check(conformance_host)
    with pytest.raises(JobsConformanceFailure, match="its example call made a proposer request"):
        await cases["rubric_propose.call"].check(conformance_host)


async def test_the_kit_catches_a_handler_that_waits_on_its_own_job() -> None:
    """It makes no request in its call and returns JobsStarted — and still holds the call until its model answers."""
    cases = _cases(_waits_on_its_job, call_return_timeout_s=0.2)

    with pytest.raises(JobsConformanceFailure, match="its call waited on its own job's model work"):
        await cases["rubric_propose.happy_path"].check(conformance_host)
    with pytest.raises(JobsConformanceFailure, match="its call waited on its own job's model work"):
        await cases["rubric_propose.call"].check(conformance_host)


async def test_the_kit_catches_a_vacuous_happy_path() -> None:
    cases = _cases(_nothing_in_job)

    await cases["rubric_propose.call"].check(conformance_host)
    with pytest.raises(JobsConformanceFailure, match="vacuous"):
        await cases["rubric_propose.happy_path"].check(conformance_host)


async def test_the_kit_catches_a_call_that_does_not_return_jobs_started() -> None:
    async def answers_in_call(host: OpsHost, caller: Caller, params: _DraftParams) -> JobsStarted:
        return "a draft"  # type: ignore[return-value]

    with pytest.raises(JobsConformanceFailure, match="returned str, not JobsStarted"):
        await _cases(answers_in_call)["rubric_propose.happy_path"].check(conformance_host)


def test_a_spend_action_without_a_happy_path_is_refused_where_it_is_declared() -> None:
    with pytest.raises(ValueError, match="a spend action declares a happy_path"):
        Action(
            name="rubric_propose",
            summary="Draft a rubric.",
            workflow="Curate",
            permission="spend",
            params=_DraftParams,
            result=JobsStarted,
            handler=_draft_in_job,
            render=str,
            example={},
            long_running=True,
        )


async def test_the_kit_passes_the_whole_catalogue_over_a_two_store_router() -> None:
    """Every case again, each host over a router keeping what runs read in one store and what happened in another.

    A job's record, its runs and its results land in the record store, its template and cases in the other, and an
    analysis wherever derived documents go: job_poll, a lease's renewal, a cancel's merge and a settle's conditional
    write each reach the record they name through the route.
    """
    route = DocTypeRoute(
        by="origin", to={"input": "given", "record": "happened", "derived": "given"}, unregistered="given"
    )

    def routed(guard: CompletionGuard) -> JobsConformanceHost:
        stores = {"given": InMemoryDocumentStore(), "happened": InMemoryDocumentStore()}
        RoutedDocumentStore.relocate(stores, route, scope_ids=[], routing_scope="evals-routing")
        built = conformance_host(guard, store=RoutedDocumentStore(stores, route, routing_scope="evals-routing"))
        held.append(stores)
        return built

    held: list[dict[str, InMemoryDocumentStore]] = []
    for case in jobs_conformance_cases(eval_catalogue((_host_action(_draft_in_job),)), job_timeout_s=30.0):
        await case.check(routed)
    job_records = {"eval_launch", "eval_sweep", "eval_second_judge_pass", "eval_judge_temperature_check"}
    written = {
        doc_type
        for stores in held
        for doc_type in job_records | {"eval_action_job"}
        if stores["happened"].by_doc_type(doc_type, CONFORMANCE_SCOPE)
    }
    assert written == job_records | {"eval_action_job"}, "every job record was written to, and read from, its store"
    assert not any(stores["given"].by_doc_type("eval_launch", CONFORMANCE_SCOPE) for stores in held)
