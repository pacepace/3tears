"""A cell's identity is readable from everything the runner does for it, and from nothing outside it (#809).

``current_cell_identity`` is bound by the run loop around a whole cell, so a host's logging filter can stamp a
record with the cell that wrote it. These drive the real runner over the toy host at width 2 with a judge whose
calls interleave, and read what each call and each log record saw. ``bind_cell_identity`` is the same binder,
public, so a host carries the identity across a task boundary of its own (#824).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import TOY_JUDGE_MODEL, ToyJudgeCompletion, toyhost_judged_template
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_run, toyhost_test_cases
from packages.evals.tests.off_loop import off_loop
from threetears.evals.run import RunnerOptions, execute_run
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.launch import build_judge_service
from threetears.evals.schema import CellIdentity, bind_cell_identity, current_cell_identity
from threetears.evals.schema.models import DEFAULT_JUDGE_TEMPERATURE, JudgedArtifact


class _SeeingJudge:
    """A judge client that records the identity current at each call, sleeping so cells interleave."""

    def __init__(self) -> None:
        self.seen: list[CellIdentity | None] = []

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return None

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        del user, response_format
        before = current_cell_identity()
        await asyncio.sleep(0.01)
        # Read again after other cells have run through the sleep: a call keeps its own cell throughout.
        assert current_cell_identity() == before
        self.seen.append(before)
        match = re.search(r'the single key "(.+?)"', system)
        assert match is not None
        return ToyJudgeCompletion(
            content=json.dumps({"reasoning": "fine", "criteria_scores": {match.group(1): 4}}),
            input_tokens=10,
            output_tokens=4,
            cost_usd=0.0001,
            model=TOY_JUDGE_MODEL,
            price_source="toyhost_script",
        )

    async def aclose(self) -> None:
        """Nothing to release."""


class _IdentityStamp(logging.Filter):
    """A host's logging filter: stamps each record with the cell current when it was written."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[str, CellIdentity | None]] = []

    def filter(self, record: logging.LogRecord) -> bool:
        self.records.append((record.getMessage(), current_cell_identity()))
        return True


async def _run_matrix(judge: _SeeingJudge, *, cell_timeout_s: float | None = None) -> list[CellIdentity | None]:
    host = toyhost_host(clients=lambda _role, _model, *, temperature=None: judge)
    world = host.profile.world
    assert world is not None
    template = toyhost_judged_template()
    kind = ToyExtractorKind(
        client=ScriptedExtractionClient(), world=world, judged=True, goal_checks=tuple(template.goal_state_checks)
    )
    cases = toyhost_test_cases(template)
    off_loop(host.storage.save_template, template)
    for case in cases:
        off_loop(host.storage.save_test_case, case)
    run_judge = off_loop(build_judge_service, host, template, TOY_JUDGE_MODEL, judged_artifact=JudgedArtifact.DOCUMENT)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={
            "k_runs": 1,
            "test_case_ids": [case.id for case in cases],
            "judge_model": TOY_JUDGE_MODEL,
            "effective_judges": run_judge.effective_judges,
            "judge_config_ids": {dim: config.id for dim, config in run_judge.configs.items()},
            "judge_asked_settings": JUDGE_REQUEST_SETTINGS,
            "judge_temperature": DEFAULT_JUDGE_TEMPERATURE,
        }
    )
    off_loop(host.storage.save_eval_run, run)
    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=cases,
        judge_service=run_judge.service,
        options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, max_concurrent_cells=2),
    )
    return judge.seen


async def test_judge_calls_of_interleaving_cells_each_read_their_own_cell_and_nothing_outside() -> None:
    judge = _SeeingJudge()
    stamp = _IdentityStamp()
    logging.getLogger("threetears.evals").addFilter(stamp)
    try:
        seen = await _run_matrix(judge)
    finally:
        logging.getLogger("threetears.evals").removeFilter(stamp)

    assert seen and all(identity is not None for identity in seen), "every judge call ran inside a cell"
    cells = {identity.cell_id for identity in seen if identity is not None}
    assert len(cells) >= 2, "the cells ran side by side, each judged under its own identity"
    assert current_cell_identity() is None, "unbound once the run is over"


async def test_a_log_record_written_in_a_cell_reads_that_cells_identity_and_one_outside_reads_none() -> None:
    judge = _SeeingJudge()
    stamp = _IdentityStamp()
    logger = logging.getLogger("threetears.evals.run.runner")
    logger.addFilter(stamp)
    try:
        logger.warning("outside any cell")
        await _run_matrix(judge)
        logger.warning("after the run")
    finally:
        logger.removeFilter(stamp)

    by_message = dict(stamp.records)
    assert by_message["outside any cell"] is None
    assert by_message["after the run"] is None
    assert all(isinstance(identity, CellIdentity) or identity is None for identity in by_message.values())


async def test_the_identity_is_reset_when_a_cell_is_cancelled_or_raises() -> None:
    outer = CellIdentity(run_id="r", test_case_id="t0", subject_id="s", cell_id="c0")
    cell = CellIdentity(run_id="r", test_case_id="t", subject_id="s", cell_id="c")

    with bind_cell_identity(outer):
        with pytest.raises(RuntimeError):
            with bind_cell_identity(cell):
                assert current_cell_identity() is cell
                raise RuntimeError("boom")
        assert current_cell_identity() is outer, "the previous binding is restored, not cleared"
    assert current_cell_identity() is None

    after_cancel: list[CellIdentity | None] = []

    async def _bound() -> None:
        try:
            with bind_cell_identity(cell):
                await asyncio.sleep(10)
        finally:
            # Read in the cancelled task's own context, where the binding was made.
            after_cancel.append(current_cell_identity())

    task = asyncio.ensure_future(_bound())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert after_cancel == [None], "a cancelled block unbinds in its own task"
    assert current_cell_identity() is None


class _DetachingJudge(_SeeingJudge):
    """A host's judge that hands work to a detached job started in a fresh context, as a host's own job runner would.

    The job reads the identity once unbound, then under the host's binding of the identity the cell handed it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[asyncio.Task[tuple[CellIdentity, CellIdentity | None, CellIdentity | None]]] = []

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        handed_off = current_cell_identity()
        assert handed_off is not None

        async def _job() -> tuple[CellIdentity, CellIdentity | None, CellIdentity | None]:
            unbound = current_cell_identity()
            with bind_cell_identity(handed_off):
                await asyncio.sleep(0.01)
                bound = current_cell_identity()
            return handed_off, unbound, bound

        self.jobs.append(asyncio.get_running_loop().create_task(_job(), context=contextvars.Context()))
        return await super().generate(system=system, user=user, response_format=response_format)


async def test_a_job_started_in_a_fresh_context_reads_the_cells_identity_under_the_hosts_binding() -> None:
    judge = _DetachingJudge()
    await _run_matrix(judge)
    results = await asyncio.gather(*judge.jobs)

    assert results, "the judge started a job for every call"
    for handed_off, unbound, bound in results:
        assert unbound is None, "a fresh context does not carry the cell, which is why the host binds it"
        assert bound is handed_off, "under the host's binding the job reads the identity the engine bound"
    assert len({handed_off.cell_id for handed_off, _, _ in results}) >= 2, "each job read its own cell"
    assert current_cell_identity() is None


async def test_concurrent_tasks_each_see_their_own_identity_and_a_child_task_inherits_it() -> None:
    async def _read(name: str) -> tuple[CellIdentity | None, CellIdentity | None]:
        mine = CellIdentity(run_id="r", test_case_id=name, subject_id="s", cell_id=name)
        with bind_cell_identity(mine):
            await asyncio.sleep(0.01)
            child = await asyncio.ensure_future(_current())
            assert current_cell_identity() is mine
            return mine, child

    async def _current() -> CellIdentity | None:
        return current_cell_identity()

    results = await asyncio.gather(*(_read(f"c{i}") for i in range(4)))
    for mine, child in results:
        assert child is mine
    assert current_cell_identity() is None
