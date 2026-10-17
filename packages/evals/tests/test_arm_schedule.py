"""How a launch's arms share time: interleaved by default, so a provider's bad minute lands on every arm alike (#841).

A declared two-arm comparison launched with one call used to start both arms side by side, each at its own width:
the provider rationed one model's calls, and that arm's sample shrank while the other's did not. These drive the real
launch and the real runner over the toy host, whose extractor is wrapped in a provider that rations every call made
inside a window of its own logical clock (the calls it has been asked, so no wall-clock sleep decides anything):

* a two-model launch naming no schedule runs interleaved, and its runs and its launch record say so;
* under it, the window's exclusions land on both arms alike, where arms one after another put them all on the first;
* the arms play their cases in one order, the same case on every arm at once;
* the schedule is declared once, for the launch and the sweep alike, and says what it is rather than naming a call;
* the scheduler stops waiting for a run that ended without handing over its cells, and one run's failure is its own.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.kernel import EvalStorage, ValidationFailedError
from threetears.evals.kernel.host import ApparatusError
from threetears.evals.ops import CampaignDefinition, LaunchArguments, OpsHost, campaign_create, run_get, run_launch
from threetears.evals.ops.sweeps import SweepSettings
from threetears.evals.run.arm_schedule import CellInterleaver
from threetears.evals.run.launch_job import EvalLaunch
from threetears.evals.schema import EvalRun
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ExtractionResult, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.product import ExtractionRequest
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import launched_runs

#: Repeats per case: the toy template has three cases, so twelve cells an arm.
_K = 4
#: The provider's bad minute, in its own calls: the first eight it is asked, of the launch's twenty-four.
_WINDOW = range(0, 8)
_ARMS = list(RUN_MODELS[:2])


class _RationingProvider(ScriptedExtractionClient):
    """The toy extractor's provider, rationing every call asked inside a window of its own call count."""

    def __init__(self, window: range) -> None:
        super().__init__(tuple(replace(script, latency_s=0.0) for script in TOY_SCRIPTS))
        self._window = window
        self._asked = 0
        self.rationed: Counter[str] = Counter()

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        asked, self._asked = self._asked, self._asked + 1
        if asked in self._window:
            self.rationed[request.model] += 1
            raise ApparatusError("the persona's model failed: API error (429): Provider returned error")
        return await super().extract(request)


def _host(client: ScriptedExtractionClient, **settings: Any) -> tuple[OpsHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    launch_settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update=settings)
    launch, _client = toyhost_launch_host(storage=storage, client=client, settings=lambda: launch_settings)
    return OpsHost(launch=launch), storage


async def _launch(host: OpsHost, **arguments: Any) -> list[EvalRun]:
    launch = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=_ARMS, k_runs=_K, **arguments
    )
    run_ids = await launched_runs(host, await run_launch(host, launch, TOYHOST_SCOPE))
    return off_loop(host.eval_host.storage.load_eval_runs, run_ids, TOYHOST_SCOPE)


def _excluded_by_arm(storage: EvalStorage, runs: list[EvalRun]) -> dict[str, int]:
    def excluded(run: EvalRun) -> int:
        results = storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)
        return sum(1 for result in results if result.termination == "apparatus_failed")

    return {run.candidate_model: off_loop(excluded, run) for run in runs}


# =============================================================================
# The default: interleaved, and recorded
# =============================================================================


async def test_a_two_model_launch_naming_no_schedule_runs_interleaved_and_its_records_say_so() -> None:
    host, storage = off_loop(_host, ScriptedExtractionClient())
    launch = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=_ARMS, k_runs=1
    )
    started = await run_launch(host, launch, TOYHOST_SCOPE)
    run_ids = await launched_runs(host, started)
    runs = off_loop(storage.load_eval_runs, run_ids, TOYHOST_SCOPE)

    assert [run.arm_schedule for run in runs] == ["interleaved", "interleaved"]
    assert {run.status for run in runs} == {"completed"}
    (job,) = started.jobs
    record, _ = off_loop(storage.load_job_record, EvalLaunch, job.job_id.removeprefix("launch:"), TOYHOST_SCOPE)
    assert record is not None and record.arm_schedule == "interleaved", "the launch's own record says so too"
    summary = off_loop(run_get, host.eval_host, run_ids[0], TOYHOST_SCOPE)
    assert summary.arm_schedule == "interleaved", "and so does the run's summary"


async def test_a_launch_measuring_latency_runs_its_arms_one_after_another_and_refuses_them_at_once() -> None:
    host, _storage = off_loop(_host, ScriptedExtractionClient())

    runs = await _launch(host, measure_latency=True)
    assert [run.arm_schedule for run in runs] == ["sequential", "sequential"]
    with pytest.raises(ValidationFailedError, match="one after another"):
        await _launch(host, measure_latency=True, arm_schedule="interleaved")


# =============================================================================
# A provider's bad minute
# =============================================================================


async def test_under_the_default_a_bad_minute_lands_on_both_arms_alike() -> None:
    provider = _RationingProvider(_WINDOW)
    # One cell at a time across the launch, so which cell meets which call is fixed: the arms strictly alternate.
    host, storage = off_loop(_host, provider, max_concurrent_cells=1)

    runs = await _launch(host)

    assert _excluded_by_arm(storage, runs) == {_ARMS[0]: 4, _ARMS[1]: 4}
    assert provider.rationed == Counter({_ARMS[0]: 4, _ARMS[1]: 4})


async def test_at_a_shared_width_the_bad_minute_is_still_split_between_the_arms() -> None:
    provider = _RationingProvider(_WINDOW)
    host, storage = off_loop(_host, provider)

    runs = await _launch(host)

    # The window's eight calls come from the first eight-plus-width cells started, which the turns split between
    # the arms within one cell of each other: neither arm can take more than six of the eight.
    excluded = _excluded_by_arm(storage, runs)
    assert sum(excluded.values()) == len(_WINDOW)
    assert max(excluded.values()) <= 6 and min(excluded.values()) >= 2, excluded


async def test_arms_one_after_another_put_the_whole_bad_minute_on_the_arm_launched_first() -> None:
    provider = _RationingProvider(_WINDOW)
    host, storage = off_loop(_host, provider)

    runs = await _launch(host, arm_schedule="sequential")

    assert [run.arm_schedule for run in runs] == ["sequential", "sequential"]
    assert _excluded_by_arm(storage, runs) == {_ARMS[0]: len(_WINDOW), _ARMS[1]: 0}, (
        "launch order alone decided which arm took the provider's rationing"
    )


async def test_interleaved_arms_play_the_same_case_at_the_same_time() -> None:
    provider = _RationingProvider(range(0))
    host, _storage = off_loop(_host, provider, max_concurrent_cells=1)

    await _launch(host)

    calls = provider.calls
    assert len(calls) == 2 * _K * len(toyhost_test_cases(toyhost_template()))
    assert [model for model, _ in calls] == _ARMS * (len(calls) // 2), "the arms take their turns cell by cell"
    pairs = list(zip(calls[::2], calls[1::2], strict=True))
    assert all(first[1] == second[1] for first, second in pairs), "each turn runs one case on both arms"


def _disclosure(host: OpsHost, storage: EvalStorage, runs: list[EvalRun], name: str) -> str | None:
    line = campaign_create(
        host.eval_host,
        CampaignDefinition(
            name=name,
            subject_id=TOYHOST_SUBJECT.subject_id,
            behavior="extraction",
            run_ids=[run.id for run in runs],
            exploratory=True,
        ),
        TOYHOST_SCOPE,
        created_by="agent:test",
    )
    campaign = storage.load_campaign(line.id, TOYHOST_SCOPE)
    assert campaign is not None
    return assemble_context_bundle(campaign, storage=storage, profile=host.eval_host.profile).launch_disclosure


async def test_a_campaign_says_when_its_arms_were_co_loaded_or_run_one_after_another() -> None:
    host, storage = off_loop(_host, ScriptedExtractionClient())

    for schedule, says in (("concurrent", "loaded their providers at once"), ("sequential", "one arm after another")):
        disclosure = off_loop(_disclosure, host, storage, await _launch(host, arm_schedule=schedule), schedule)
        assert disclosure is not None and says in disclosure
    interleaved = off_loop(_disclosure, host, storage, await _launch(host), "interleaved")
    assert interleaved is None, "arms taking their turns case by case need no timing disclosure"


# =============================================================================
# One definition
# =============================================================================


def test_the_launch_and_the_sweep_share_one_definition_of_the_schedule() -> None:
    launch, sweep = LaunchArguments.model_fields["arm_schedule"], SweepSettings.model_fields["arm_schedule"]

    assert (sweep.annotation, sweep.default, sweep.description) == (
        launch.annotation,
        launch.default,
        launch.description,
    )
    assert "max_concurrent_arms" not in SweepSettings.model_fields, "the sweep takes no second scheduling argument"
    catalogue = eval_catalogue()
    for action in ("run_launch", "sweep_launch"):
        offered = catalogue.get(action)
        assert offered is not None
        assert offered.params.model_fields["arm_schedule"].description == launch.description


def test_the_schedule_says_what_it_is_and_names_no_action() -> None:
    description = LaunchArguments.model_fields["arm_schedule"].description
    assert description is not None
    named = [action.name for action in eval_catalogue().actions if action.name in description]
    assert not named, f"a host may share the parameter's name, so its description names no action: {named}"


# =============================================================================
# The scheduler over N runs
# =============================================================================


async def test_the_turns_start_once_every_run_is_in_and_stop_waiting_for_one_that_ended() -> None:
    interleaver = CellInterleaver()
    for run_id in ("a", "b", "c"):
        interleaver.expect(run_id)
    interleaver.seal()
    ran: list[str] = []

    def cell(name: str) -> Any:
        async def work() -> None:
            ran.append(name)

        return work

    a = asyncio.create_task(interleaver.lane("a", None).execute([cell("a1"), cell("a2")], width=1))
    b = asyncio.create_task(interleaver.lane("b", None).execute([cell("b1")], width=1))
    await asyncio.sleep(0)
    assert ran == [], "no cell starts while a run of the launch has not handed its cells over"

    interleaver.withdraw("c")  # its job ended before its loop began
    await asyncio.gather(a, b)
    assert ran == ["a1", "b1", "a2"]


async def test_one_runs_failure_is_its_own_and_each_cell_runs_in_its_own_runs_context() -> None:
    interleaver = CellInterleaver()
    interleaver.expect("broken")
    interleaver.expect("fine")
    interleaver.seal()
    run_of: contextvars.ContextVar[str] = contextvars.ContextVar("run_of")
    seen: list[tuple[str, str]] = []

    def cell(name: str, *, fails: bool = False) -> Any:
        async def work() -> None:
            seen.append((name, run_of.get()))
            if fails:
                raise RuntimeError(f"{name} broke")

        return work

    async def execute(run_id: str, cells: list[Any]) -> None:
        run_of.set(run_id)
        await interleaver.lane(run_id, None).execute(cells, width=2)

    broken = asyncio.create_task(execute("broken", [cell("x1", fails=True), cell("x2"), cell("x3")]))
    fine = asyncio.create_task(execute("fine", [cell("y1"), cell("y2"), cell("y3")]))
    results = await asyncio.gather(broken, fine, return_exceptions=True)

    assert isinstance(results[0], RuntimeError) and results[1] is None
    assert [name for name, _ in seen if name.startswith("y")] == ["y1", "y2", "y3"], "its sibling ran every cell"
    assert "x3" not in [name for name, _ in seen], "no cell of the failed run started after its failure"
    assert all(run == ("broken" if name.startswith("x") else "fine") for name, run in seen)
