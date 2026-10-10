"""The per-cell deadline is set at launch, bounded by the host, and recorded on the run (#649).

A toy-host launch whose kind wires a short deadline loses a slow cell to ``cell_timeout``; the same launch
naming a longer ``cell_timeout_s`` lets it finish, and the run records the deadline it ran under with its
origin. A value above the host's ceiling is refused before anything is built, and a host that declares no
ceiling may only have its kind's deadline lowered.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.evals.contracts import EvalRun, EvalStorage, ValidationFailedError
from threetears.evals.ops import LaunchArguments, launch_estimate, run_launch
from threetears.evals.ops.host import OpsHost
from threetears.evals.run import LaunchHost, start_run
from threetears.evals.run.runner import DEFAULT_CELL_TIMEOUT_S
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template

#: The kind's own deadline in these tests, and how long the slow extractor takes over each cell.
KIND_DEADLINE_S = 0.05
SLOW_CELL_S = 0.2


def _launching(
    *, max_cell_timeout_s: float | None = None, kind_deadline_s: float | None = KIND_DEADLINE_S
) -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    slow = ScriptedExtractionClient(
        tuple(type(script)(**{**vars(script), "latency_s": SLOW_CELL_S}) for script in TOY_SCRIPTS)
    )
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_cell_timeout_s": max_cell_timeout_s})
    host, _client = toyhost_launch_host(
        storage=storage, settings=lambda: settings, extraction_client=slow, kind_cell_timeout_s=kind_deadline_s
    )
    return host, storage


async def _launch(host: LaunchHost, storage: EvalStorage, **arguments: Any) -> EvalRun:
    (run,) = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=1,
        **arguments,
    )
    async with asyncio.timeout(20):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None
    return stored


def _terminations(storage: EvalStorage, run: EvalRun) -> set[str]:
    return {result.termination for result in storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)}


async def test_a_launch_deadline_above_the_kinds_lets_a_slow_cell_finish_and_is_recorded():
    host, storage = _launching(max_cell_timeout_s=5.0)

    short = await _launch(host, storage)
    assert _terminations(storage, short) == {"cell_timeout"}, "the kind's deadline cuts every slow cell"
    assert (short.cell_timeout_s, short.cell_timeout_s_origin) == (KIND_DEADLINE_S, "kind")

    longer = await _launch(host, storage, cell_timeout_s=2.0)
    assert _terminations(storage, longer) == {"completed"}, "the launch's deadline lets each cell finish"
    assert (longer.cell_timeout_s, longer.cell_timeout_s_origin) == (2.0, "launch")


async def test_a_launch_without_a_deadline_on_a_kind_with_none_records_the_engine_default():
    host, storage = _launching(kind_deadline_s=None)
    run = await _launch(host, storage)
    assert (run.cell_timeout_s, run.cell_timeout_s_origin) == (DEFAULT_CELL_TIMEOUT_S, "default")


async def test_a_deadline_above_the_hosts_ceiling_is_refused_before_anything_is_built():
    host, storage = _launching(max_cell_timeout_s=5.0)

    with pytest.raises(ValidationFailedError, match=r"cell_timeout_s=10 is above the host's ceiling of 5s"):
        await _launch(host, storage, cell_timeout_s=10.0)
    assert storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_a_host_declaring_no_ceiling_lets_a_launch_only_lower_its_kinds_deadline():
    host, storage = _launching(max_cell_timeout_s=None)

    with pytest.raises(ValidationFailedError, match=r"above the kind's ceiling of 0.05s"):
        await _launch(host, storage, cell_timeout_s=1.0)
    lowered = await _launch(host, storage, cell_timeout_s=0.01)
    assert (lowered.cell_timeout_s, lowered.cell_timeout_s_origin) == (0.01, "launch")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
async def test_a_deadline_no_cell_could_run_under_is_refused(bad: float):
    host, storage = _launching(max_cell_timeout_s=5.0)
    with pytest.raises(ValidationFailedError, match="cell_timeout_s must be a finite number"):
        await _launch(host, storage, cell_timeout_s=bad)


async def test_the_operation_and_its_estimate_carry_the_deadline():
    host, storage = _launching(max_cell_timeout_s=5.0)
    ops = OpsHost(launch=host)
    arguments = LaunchArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=1,
        cell_timeout_s=9.0,
    )
    with pytest.raises(ValidationFailedError, match="above the host's ceiling"):
        await launch_estimate(ops, arguments, TOYHOST_SCOPE)
    with pytest.raises(ValidationFailedError, match="above the host's ceiling"):
        await run_launch(ops, arguments, TOYHOST_SCOPE)
    started = await run_launch(ops, arguments.model_copy(update={"cell_timeout_s": 2.0}), TOYHOST_SCOPE)
    (job,) = started.jobs
    async with asyncio.timeout(20):
        while host.job_manager.is_active(job.target_id):
            await asyncio.sleep(0.01)
    stored = storage.load_eval_run(job.target_id, TOYHOST_SCOPE)
    assert stored is not None and (stored.cell_timeout_s, stored.cell_timeout_s_origin) == (2.0, "launch")


def test_a_deadline_and_its_origin_are_recorded_together():
    with pytest.raises(ValueError, match="recorded together or not at all"):
        make_eval_run(cell_timeout_s=3.0)
    run = make_eval_run()
    assert run.cell_timeout_s is None and run.cell_timeout_s_origin is None, "a run stored before #649 reads unknown"
