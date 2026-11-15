"""The courier host's uses of names, each the only door to something a host must do (issue #1199).

Offline throughout: the courier's planner model is scripted, so nothing here spends.

- ``failed_call_output``: a round planned while the planner's endpoint is overloaded is excluded as the provider's,
  and the run goes on; the courier's describer is what says whose failure it was.
- ``start_action_job``: the courier's own action proposes cases as a job, read back through ``job_poll``.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from threetears.evals.actions import Caller, MountedTool, standard_tools
from threetears.evals.kernel import withhold_failure_detail
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import OpsHost
from threetears.evals.run import RunnerOptions, execute_run
from threetears.evals.schema import EvalResult, EvalTestCase
from packages.evals.tests.fixtures.courierhost import (
    COURIER_KIND,
    COURIER_SCOPE,
    COURIER_TEMPLATE_ID,
    RoutePlannerKind,
    courier_cases,
    courier_catalogue,
    courier_host,
    courier_ops_host,
    courier_run,
    courier_template,
)
from packages.evals.tests.off_loop import off_loop

CALLER = Caller(scope_id=COURIER_SCOPE, identity="dispatcher:north")


def _outage_case() -> EvalTestCase:
    """A round of six stops planned while the planner's endpoint is overloaded."""
    return EvalTestCase(
        id=f"{COURIER_TEMPLATE_ID}:stops-6:outage",
        scope_id=COURIER_SCOPE,
        template_id=COURIER_TEMPLATE_ID,
        variation_params={"stops": "6", "outage": "true"},
    )


async def _planned(host: EvalHost, model: str, cases: list[EvalTestCase]) -> list[EvalResult]:
    """One arm of the courier's planner over ``cases``, run through the real trial loop: its results."""
    world = host.profile.world
    assert world is not None
    kind = RoutePlannerKind(world, describe=host.failure_describer)
    run = courier_run(model, cases=cases, world=world)
    off_loop(host.storage.save_template, courier_template())
    off_loop(host.storage.save_eval_run, run)
    await execute_run(
        host,
        run=run,
        template=courier_template(),
        test_cases=cases,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={COURIER_KIND: lambda _cell: kind}),
    )
    return off_loop(host.storage.query_eval_results_by_run, run.id, COURIER_SCOPE)


async def test_an_overloaded_planner_call_is_excluded_as_the_provider_s_and_the_run_goes_on() -> None:
    outage_case = _outage_case()
    results = await _planned(courier_host(), "planner-lite", [*courier_cases(), outage_case])

    outage = [result for result in results if result.test_case_id == outage_case.id]
    others = [result for result in results if result.test_case_id != outage_case.id]
    assert outage and all(result.candidate_error is None for result in outage)
    assert all(
        (result.infra_error or "").startswith("apparatus: planner: the endpoint declined the call: planner-lite: 529")
        for result in outage
    ), [result.infra_error for result in outage]
    assert len(others) == 2 * len(courier_cases()), "the run stopped at the declined call"
    assert all(result.infra_error is None and result.candidate_error is None for result in others)


async def test_with_no_describer_of_its_own_the_same_failure_is_charged_to_the_planner() -> None:
    """The describer decides whose failure it was: without the courier's, the overload is the candidate's."""
    host = replace(courier_host(), failure_describer=withhold_failure_detail)

    results = await _planned(host, "planner-lite", [_outage_case()])

    assert results and all(result.infra_error is None for result in results)
    assert all((result.candidate_error or "").startswith("planner: the call failed: ") for result in results)


async def _call(tool: MountedTool, host: OpsHost, arguments: dict[str, Any]) -> dict[str, Any]:
    outcome = await tool.call(arguments, host=host, caller=CALLER)
    assert outcome.refusal is None and outcome.structured is not None, outcome.text
    return outcome.structured


async def _settled(tool: MountedTool, host: OpsHost, job_id: str) -> dict[str, Any]:
    """The job's status through ``job_poll`` once it has ended."""
    async with asyncio.timeout(30):
        while not (status := await _call(tool, host, {"action": "job_poll", "job_id": job_id}))["done"]:
            await asyncio.sleep(0.01)
    return status


async def test_the_courier_s_own_action_proposes_cases_as_a_job_read_back_through_job_poll() -> None:
    host = off_loop(courier_ops_host)  # the factory stores the courier's template, a blocking call
    (tool, _admin) = courier_catalogue().mount_all(standard_tools())

    started = await _call(tool, host, {"action": "courier_cases_propose", "max_rounds": 3})
    (job,) = started["jobs"]
    status = await _settled(tool, host, job["job_id"])

    assert (job["job_id"].split(":", 1)[0], job["target_id"]) == ("action", "courier_cases_propose")
    assert status["state"] == "completed", status
    # Eleven-stop rounds were dispatched three times and fourteen once; four, six and eight are already cases.
    proposed = [case["variation_params"]["stops"] for case in status["result"]["cases"]]
    assert proposed == ["11", "14"]
