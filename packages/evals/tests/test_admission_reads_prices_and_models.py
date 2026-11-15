"""Admission reads the price catalogue and the provider's model list before it prices or admits an arm.

Two failures met in Discodon's production, both at launch admission:

- **A largest-call check skipped without notice.** A launch, or a resume right after a restart, priced each arm's
  largest call before any price catalogue had been read in the process. The ceiling came back unknown, and the
  "largest call above the cap" refusal was skipped with only a WARNING logged.
- **A run on a model no longer served.** A sweep arm on a model OpenRouter had retired was estimated from history,
  admitted, and judged every empty reply as a fail.

So admission brings each client's prices up to date before it prices a call, and asks the host's client factory
what the provider's own model list, read now, says of each model the arm names. Pinned here, on the shipped
``Clients`` over a fake of OpenRouter's models endpoint (nothing reaches the network):

- **Unread catalogue, fresh process:** the estimate and the launch read it, then price; the largest call is priced.
  A resume taken up by a fresh process does the same.
- **Unreadable catalogue, under a cap:** the arm is refused, naming the call and why, by the estimate, the launch
  and the resume alike. Never a silent pass.
- **No cap:** nothing to check a ceiling against, so the arm is admitted, and the estimate and the launch say the
  served-model check was skipped and why.
- **A model the list does not name** is refused whatever the cap, naming it and the provider.
- **One read serves both checks.**
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import timedelta
from typing import Any

import httpx
import pytest

from threetears.evals.kernel import (
    JUDGE_KIND,
    JUDGE_KIND_CONTRACT,
    JudgeCase,
    JudgeCaseSource,
    ValidationFailedError,
    judge_case_payload,
)
from threetears.evals.ops import (
    LaunchArguments,
    OpsHost,
    SweepArguments,
    SweepArm,
    job_cancel,
    job_resume,
    launch_estimate,
    run_job_id,
    run_launch,
    sweep_launch,
)
from threetears.evals.ops.lenses import estimate_text
from threetears.evals.providers import Catalogue, Clients, Model
from threetears.evals.run import (
    ArmPrice,
    ArmQuote,
    LaunchHost,
    ServedCheck,
    admit_launch,
    launchable_judge_kind,
    quote_launch,
    resume_run,
    start_run,
)
from threetears.evals.schema import EvalTemplate, EvalTestCase, JudgedArtifact, JudgeEvidence
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import with_registry
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import settled

#: The judge under test: the judge kind's candidate, called on OpenRouter.
_FLASH = "google/gemini-3.8-flash"

#: The template the judge cases are frozen into.
_TEMPLATE_ID = "judge_calibration"

#: OpenRouter's row for the judge under test: cheap enough that its largest call is well under a dollar.
_FLASH_ROW = {"id": _FLASH, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}

#: A row for some other model, so a list without the judge's row is still a list that was read.
_OTHER_ROW = {"id": "vendor/other", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}

#: A model the judge role sends to Anthropic directly, shaped like a router id so only the role keeps it off the list.
_DIRECT_MODEL = "vendor/direct-model"

#: What OpenRouter reports each judge call cost: enough that one call passes the small cap a run is stopped by.
_REPORTED_CALL_USD = 0.3


# parity-exempt: stands in for OpenRouter's HTTP endpoints behind httpx.MockTransport, not a Python protocol
class _FakeOpenRouter:
    """OpenRouter's models list (``GET``) and chat completions (``POST``), scripted, counting the list's reads."""

    def __init__(self, rows: list[dict[str, Any]], *, list_fails: bool = False) -> None:
        self.rows = rows
        self.list_fails = list_fails
        self.list_reads = 0
        self.completions = 0
        #: When set, every completion waits here until it is set, so a test can hold a run part-way.
        self.held: asyncio.Event | None = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.list_reads += 1
            if self.list_fails:
                return httpx.Response(503, json={"error": {"message": "unavailable"}})
            return httpx.Response(200, json={"data": self.rows})
        self.completions += 1
        if self.held is not None:
            await self.held.wait()
        content = json.dumps({"reasoning": "read it", "criteria_scores": {TRANSCRIPT_DIM_ID: 3}})
        return httpx.Response(
            200,
            json={
                "model": _FLASH,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 40,
                    "cost": _REPORTED_CALL_USD,
                    "is_byok": False,
                },
            },
        )


def _judge_case(index: int) -> EvalTestCase:
    case = JudgeCase(
        dim=TRANSCRIPT_DIM_ID,
        scale="ordinal",
        intent="Answer the member's question as the bot would.",
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        evidence=JudgeEvidence(case_material="The member asked about the raid schedule.", artifact="a short reply"),
        source=JudgeCaseSource(run_id="source-run", result_id=f"result-{index}", test_case_id=f"source-case-{index}"),
    )
    return EvalTestCase(
        id=f"judge-case-{index}",
        scope_id=TOYHOST_SCOPE,
        template_id=_TEMPLATE_ID,
        variation_params={},
        host_payload=judge_case_payload(case),
    )


class _Process:
    """One process of a host launching judge campaigns on the shipped ``Clients``: its own catalogue, read only when
    something asks, and its own job manager, over a store a test may share between processes."""

    def __init__(self, provider: _FakeOpenRouter, *, enforced: bool = True, storage: Any = None) -> None:
        self.provider = provider
        http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        self.clients = Clients(_FLASH, catalogue=Catalogue(http=http), http=http)
        settings = TOYHOST_LAUNCH_SETTINGS.model_copy(
            update={"enforcement_enabled": enforced, "max_concurrent_cells": 1}
        )
        profile = toyhost_profile()
        toy, _ = toyhost_launch_host(
            profile=dataclasses.replace(
                profile, kinds=(*profile.kinds, dataclasses.replace(JUDGE_KIND_CONTRACT, seats=None))
            ),
            storage=storage,
            clients=self.clients,
            settings=lambda: settings,
        )

        def priced(_quote: ArmQuote) -> ArmPrice:
            return ArmPrice(predicted_usd=0.01, basis="from earlier runs of the template")

        holder: list[LaunchHost] = []
        self.host: LaunchHost = dataclasses.replace(
            toy,
            eval_host=with_registry(toy.eval_host, {JUDGE_KIND: launchable_judge_kind(lambda: holder[0])}),
            launch_pricer=priced,
        )
        holder.append(self.host)
        if storage is None:
            self.host.eval_host.storage.save_template(
                EvalTemplate(
                    id=_TEMPLATE_ID,
                    scope_id=TOYHOST_SCOPE,
                    name="judge calibration",
                    intent="Score as a person would.",
                    candidate_kind=JUDGE_KIND,
                )
            )
            for index in range(3):
                self.host.eval_host.storage.save_test_case(_judge_case(index))

    def arguments(self, max_cost_usd: float | None) -> dict[str, Any]:
        return {
            "template_id": _TEMPLATE_ID,
            "subject_id": "the flash judge",
            "models": [_FLASH],
            "k_runs": 1,
            "max_cost_usd": max_cost_usd,
        }

    async def estimate(self, max_cost_usd: float | None) -> Any:
        [arm] = (await quote_launch(self.host, **self.arguments(max_cost_usd), scope_id=TOYHOST_SCOPE)).arms
        return arm

    async def ops_estimate(self, max_cost_usd: float | None) -> Any:
        arguments = LaunchArguments(**self.arguments(max_cost_usd))
        [arm] = (await launch_estimate(OpsHost(launch=self.host), arguments, TOYHOST_SCOPE)).arms
        return arm

    def stored_runs(self) -> list[Any]:
        return off_loop(self.host.eval_host.storage.query_eval_runs, TOYHOST_SCOPE)


@pytest.fixture(autouse=True)
def _openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


# =============================================================================
# The price catalogue, unread when admission begins
# =============================================================================


async def test_a_fresh_process_reads_the_catalogue_once_and_prices_the_largest_call_and_checks_the_model() -> None:
    process = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW]))
    assert not process.clients.catalogue.loaded, "nothing has read the catalogue in this process yet"

    arm = await process.estimate(1.0)

    assert (arm.outcome, arm.refusal) == ("admitted", None)
    assert arm.plan.call_ceiling is not None and arm.plan.call_ceiling.usd > 0, "priced after the refresh"
    assert arm.plan.call_unpriced is None
    [check] = arm.plan.served_checks
    assert (check.role, check.model, check.provider, check.listed) == ("candidate", _FLASH, "openrouter", True)
    assert process.provider.list_reads == 1, "one read of the list served the price and the served-model check"


async def test_an_unreadable_catalogue_refuses_a_capped_arm_by_the_estimate_and_the_launch_naming_why() -> None:
    process = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW], list_fails=True))

    arm = await process.estimate(1.0)
    assert arm.outcome == "refused"
    assert arm.plan.call_ceiling is None
    assert arm.plan.call_unpriced is not None
    assert "cannot be held to its $1.00 cap: the judge call on case 'judge-case-0'" in arm.refusal
    assert "cannot be priced before it is sent (the price catalogue could not be read (HTTPStatusError" in arm.refusal
    assert "unknown, not $0" in arm.refusal
    assert arm.remedy.startswith("launch again once the host's client can price the call")

    reported = await process.ops_estimate(1.0)
    assert (reported.outcome, reported.refusal, reported.remedy) == ("refused", arm.refusal, arm.remedy)

    with pytest.raises(ValidationFailedError) as refused:
        await start_run(process.host, **process.arguments(1.0), scope_id=TOYHOST_SCOPE)
    assert refused.value.message == arm.refusal, "the launch refuses word for word what the estimate reports"
    assert process.provider.completions == 0, "no call was made"
    assert process.stored_runs() == []


async def test_with_no_cap_an_unreadable_list_admits_the_arm_and_the_estimate_and_launch_say_the_check_was_skipped() -> (
    None
):
    process = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW], list_fails=True), enforced=False)

    arm = await process.estimate(None)
    assert (arm.outcome, arm.refusal) == ("uncapped", None)
    [check] = arm.plan.served_checks
    assert check.listed is None
    notice = check.skipped_notice()
    assert notice is not None
    assert f"served-model check of candidate model {_FLASH!r} was skipped" in notice
    assert "the price catalogue could not be read (HTTPStatusError" in notice

    estimate = await launch_estimate(
        OpsHost(launch=process.host), LaunchArguments(**process.arguments(None)), TOYHOST_SCOPE
    )
    [reported] = estimate.arms
    assert reported.served_check_skipped == [notice]
    assert "served_check_skipped" in reported.model_dump(mode="json")
    assert f"  {notice}" in estimate_text(estimate), "the estimate's text says it too"

    admitted = await admit_launch(process.host, **process.arguments(None), scope_id=TOYHOST_SCOPE)
    assert admitted.notices == (notice,)
    await admitted.discard()

    host = OpsHost(launch=process.host)
    started = await run_launch(host, LaunchArguments(**process.arguments(None)), TOYHOST_SCOPE)
    assert started.notices == [notice], "the launch's caller is told, not only the log"
    (job,) = started.jobs
    assert (await settled(host, job.job_id)).state == "completed"


async def test_a_listed_and_priced_launch_carries_no_notice() -> None:
    process = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW]))

    reported = await process.ops_estimate(1.0)
    assert reported.served_check_skipped == []
    assert "served_check_skipped" not in reported.model_dump(mode="json")
    admitted = await admit_launch(process.host, **process.arguments(1.0), scope_id=TOYHOST_SCOPE)
    assert admitted.notices == ()
    await admitted.discard()


# =============================================================================
# A model the provider no longer lists
# =============================================================================


@pytest.mark.parametrize("enforced", [True, False], ids=["capped", "uncapped"])
async def test_a_model_the_provider_no_longer_lists_is_refused_by_the_estimate_and_the_launch_naming_it(
    enforced: bool,
) -> None:
    process = off_loop(_Process, _FakeOpenRouter([_OTHER_ROW]), enforced=enforced)
    cap = 1.0 if enforced else None

    arm = await process.estimate(cap)
    assert arm.outcome == "refused"
    assert (
        f"names a model its provider no longer serves: candidate model {_FLASH!r} is not on openrouter's list of "
        "the models it serves (the provider's model list, read at "
    ) in arm.refusal
    assert "Every call to it would fail" in arm.refusal
    assert "name a model the provider lists" in arm.remedy

    with pytest.raises(ValidationFailedError) as refused:
        await start_run(process.host, **process.arguments(cap), scope_id=TOYHOST_SCOPE)
    assert refused.value.message == arm.refusal
    assert process.provider.completions == 0
    assert process.stored_runs() == []


def _host_routed(process: _Process) -> None:
    """``process``'s judge kind as a kind whose candidate the host calls through its own route, not the factory."""
    (name, kind) = next(iter(process.host.kinds.items()))
    process.host = dataclasses.replace(
        process.host,
        eval_host=with_registry(process.host.eval_host, {name: dataclasses.replace(kind, candidate_role=None)}),
    )


async def test_a_host_routed_candidate_off_the_list_is_admitted_and_its_check_reported_skipped() -> None:
    # The host names its own route's model with a vendor/model id OpenRouter does not list: the package does not
    # route that candidate, so OpenRouter's list is no evidence it is unserved.
    process = off_loop(_Process, _FakeOpenRouter([_OTHER_ROW]), enforced=False)
    _host_routed(process)

    arm = await process.estimate(None)

    assert (arm.outcome, arm.refusal) == ("uncapped", None), "a launch the host would serve is not refused on a guess"
    [check] = arm.plan.served_checks
    assert (check.role, check.listed) == ("candidate", None)
    notice = check.skipped_notice()
    assert notice is not None
    assert f"served-model check of candidate model {_FLASH!r} was skipped" in notice
    assert "calls this candidate through its own route" in notice
    assert "set candidate_role to the factory role" in notice and "model_listing(model, role=None)" in notice
    reported = await process.ops_estimate(None)
    assert reported.served_check_skipped == [notice]
    admitted = await admit_launch(process.host, **process.arguments(None), scope_id=TOYHOST_SCOPE)
    assert admitted.notices == (notice,)
    await admitted.discard()


async def test_a_package_routed_candidate_off_the_list_is_refused() -> None:
    # The judge kind calls its candidate as a judge-role client of the host's factory: the package knows the route.
    process = off_loop(_Process, _FakeOpenRouter([_OTHER_ROW]), enforced=False)
    assert next(iter(process.host.kinds.values())).candidate_role == "judge"

    arm = await process.estimate(None)

    assert arm.outcome == "refused"
    assert f"candidate model {_FLASH!r} is not on openrouter's list" in arm.refusal


# =============================================================================
# A resume taken up by a process that has read nothing yet
# =============================================================================


async def _budget_stopped_run() -> tuple[_Process, str]:
    """A judge run whose first reported call passed its cap, so it stopped ``budget_stopped`` with cases left."""
    first = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW]))
    [run] = await start_run(first.host, **first.arguments(0.1), scope_id=TOYHOST_SCOPE)
    async with asyncio.timeout(10):
        while first.host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    [stored] = first.stored_runs()
    assert stored.status == "budget_stopped"
    return first, run.id


async def test_a_resume_in_a_fresh_process_whose_catalogue_cannot_be_read_is_refused_naming_why() -> None:
    first, run_id = await _budget_stopped_run()
    restarted = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW], list_fails=True), storage=first.host.eval_host.storage)

    with pytest.raises(ValidationFailedError) as refused:
        await resume_run(restarted.host, run_id, TOYHOST_SCOPE, max_cost_usd=3.0)

    assert "cannot be held to its $3.00 cap: the judge call on case" in refused.value.message
    assert "the price catalogue could not be read" in refused.value.message
    assert restarted.provider.completions == 0
    [stored] = restarted.stored_runs()
    assert stored.status == "budget_stopped", "a refused take-up writes nothing"


async def test_a_resume_in_a_fresh_process_reads_the_catalogue_and_is_priced_and_taken_up() -> None:
    first, run_id = await _budget_stopped_run()
    restarted = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW]), storage=first.host.eval_host.storage)
    assert not restarted.clients.catalogue.loaded

    taken_up = await resume_run(restarted.host, run_id, TOYHOST_SCOPE, max_cost_usd=3.0)
    async with asyncio.timeout(10):
        while restarted.host.job_manager.is_active(taken_up.id):
            await asyncio.sleep(0.01)

    assert restarted.provider.list_reads >= 1, "the take-up read the catalogue before it priced"
    [stored] = restarted.stored_runs()
    assert stored.status == "completed"


async def _notice_of(process: _Process) -> str:
    """What an uncapped estimate on ``process`` says of the skipped check: the sentence every result carries."""
    arm = await process.estimate(None)
    [check] = arm.plan.served_checks
    notice = check.skipped_notice()
    assert notice is not None
    return notice


async def test_a_resume_whose_list_cannot_be_read_is_taken_up_and_job_resume_says_the_check_was_skipped() -> None:
    first, run_id = await _budget_stopped_run()
    restarted = off_loop(
        _Process, _FakeOpenRouter([_FLASH_ROW], list_fails=True), enforced=False, storage=first.host.eval_host.storage
    )
    notice = await _notice_of(restarted)
    host = OpsHost(launch=restarted.host)

    started = await job_resume(host, run_job_id(run_id), TOYHOST_SCOPE, max_cost_usd=3.0)

    assert started.notices == [notice], "the resume's caller is told, not only the log"
    (job,) = started.jobs
    assert (await settled(host, job.job_id)).state == "completed"


def _sweep_arguments() -> SweepArguments:
    return SweepArguments(
        template_id=_TEMPLATE_ID,
        subject_id="the flash judge",
        campaign_name="flash judge",
        campaign_behavior="judge agreement",
        k_runs=1,
        arms=[SweepArm(model=_FLASH, label="flash")],
    )


async def test_a_sweep_whose_list_cannot_be_read_is_launched_and_its_result_says_the_check_was_skipped() -> None:
    process = off_loop(_Process, _FakeOpenRouter([_FLASH_ROW], list_fails=True), enforced=False)
    notice = await _notice_of(process)
    host = OpsHost(launch=process.host)

    started = await sweep_launch(host, _sweep_arguments(), TOYHOST_SCOPE, created_by="agent:test")

    assert started.notices == [notice], "the sweep's caller is told, not only the log"
    (job,) = started.jobs
    assert (await settled(host, job.job_id)).state == "completed"


async def test_a_sweep_taken_up_where_its_list_cannot_be_read_says_the_check_was_skipped() -> None:
    held = _FakeOpenRouter([_FLASH_ROW])
    held.held = asyncio.Event()
    first = off_loop(_Process, held, enforced=False)
    holder = OpsHost(launch=first.host)
    (job,) = (await sweep_launch(holder, _sweep_arguments(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    async with asyncio.timeout(10):
        while held.completions == 0:
            await asyncio.sleep(0.01)
    await job_cancel(holder, job.job_id, TOYHOST_SCOPE, reason="restarting the host")
    assert (await settled(holder, job.job_id)).state == "cancelled"
    held.held.set()
    manager = first.host.job_manager
    async with asyncio.timeout(10):
        while manager.is_task_active(job.job_id.removeprefix("sweep:")) or any(
            manager.is_active(run.id) for run in first.stored_runs()
        ):
            await asyncio.sleep(0.01)

    restarted = off_loop(
        _Process, _FakeOpenRouter([_FLASH_ROW], list_fails=True), enforced=False, storage=first.host.eval_host.storage
    )
    notice = await _notice_of(restarted)
    resumer = OpsHost(launch=restarted.host)
    started = await job_resume(resumer, job.job_id, TOYHOST_SCOPE)

    assert started.notices == [notice], "the sweep's take-up tells its caller, not only the log"
    (resumed,) = started.jobs
    assert (await settled(resumer, resumed.job_id)).state == "completed"


# =============================================================================
# What the shipped client factory reads, and for which models
# =============================================================================


async def test_clients_read_openrouters_list_only_for_a_model_that_goes_to_openrouter() -> None:
    provider = _FakeOpenRouter([_FLASH_ROW])
    http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    clients = Clients(
        _FLASH,
        judge=Model(_DIRECT_MODEL, provider="anthropic", api_key="sk-ant-test", http=http),
        catalogue=Catalogue(http=http),
        http=http,
    )

    assert await clients.model_listing(_DIRECT_MODEL, role="judge") is None, "Anthropic's list is not read"
    assert await clients.model_listing("gpt-5-nano", role="simulator") is None, "a bare id names no OpenRouter model"
    assert await clients.model_listing(_FLASH) is None, "a candidate the host routes itself is not this factory's"
    assert provider.list_reads == 0

    listed = await clients.model_listing(_FLASH, role="simulator")
    assert listed is not None and (listed.provider, listed.listed) == ("openrouter", True)
    assert listed.detail.startswith("the provider's model list, read at ")
    routed = await clients.model_listing(f"{_FLASH}:nitro", role="simulator")
    assert routed is not None and routed.listed is True, "a routing suffix is no row of its own"
    gone = await clients.model_listing("inception/mercury-3", role="simulator")
    assert gone is not None and gone.listed is False
    assert provider.list_reads == 1, "rows younger than refresh_after are kept, not fetched again"


async def test_a_list_whose_refresh_failed_is_not_current_and_the_check_is_skipped_saying_so() -> None:
    provider = _FakeOpenRouter([_FLASH_ROW])
    http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    clients = Clients(_FLASH, catalogue=Catalogue(http=http, refresh_after=timedelta(0)), http=http)
    assert (await clients.model_listing(_FLASH, role="judge")).listed is True  # type: ignore[union-attr]

    provider.list_fails = True
    stale = await clients.model_listing("inception/mercury-3", role="judge")

    assert stale is not None and stale.listed is None, "a stale list neither refuses nor vouches for a model"
    assert "reading it again failed" in stale.detail and "not current" in stale.detail


def test_a_served_check_says_nothing_once_the_list_was_read() -> None:
    read = ServedCheck(role="judge", model=_FLASH, provider="openrouter", listed=False, detail="the list")
    assert read.skipped_notice() is None
