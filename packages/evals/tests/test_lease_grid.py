"""A launch keeps its job leases across a grid of runs, cell width and store latency.

The lease tests beside this one drive :class:`~threetears.evals.kernel.JobLeases` directly, or one launch at a time.
This drives the whole path — launches through the operations layer, runs on the job manager, cells writing their
results and progress, every record renewed on the leases' own pool — over a store that answers every call late, and
asserts that no record anywhere in the grid lapsed: none is written lost, none carries a lease fault, and every run
completes.

The lease is short so a run outlives it twice over, and the store's latency is a far larger share of that lease than
a production store's is of a production lease (milliseconds against a minute), so a renewal that would lose its
lease at some width or latency loses it here first. A lapse names its grid cell.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from threetears.evals.kernel import EvalStorage
from threetears.evals.ops import LaunchArguments, OpsHost, await_job, job_poll, run_job_id, run_launch
from threetears.evals.schema import EvalRun
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.off_loop import off_loop

#: The lease every grid cell runs under, renewed every third of it. Long enough that a healthy renewal clears a
#: scheduling stall on a machine running the whole suite (the margin a renewal leaves before a lapse is two thirds of
#: it), short enough that every run outlives two of them.
LEASE_S = 1.8

#: How long one extraction takes. Each run executes three waves of cells, and its store calls, late, take longer
#: still; the test holds every run to outliving two leases, or it would prove nothing about keeping them.
EXTRACT_S = 0.6

#: The toy corpus's cases, each run's cells per repeat.
CASES = 3


@dataclass(frozen=True)
class GridCell:
    """One point of the grid: how many runs in flight, how many of each run's cells at once, and the store's latency."""

    runs: int
    width: int
    latency_s: float

    @property
    def repeats(self) -> int:
        """Each run's ``k_runs``: enough cells for three waves at its width."""
        return self.width

    def __str__(self) -> str:
        return f"{self.runs} runs x width {self.width} at {self.latency_s * 1000:g} ms per store call"


#: The production shape (8 runs of 4 cells in flight each, 32 at once) and a wider one (16 of 8, 128 at once).
GRID = (GridCell(runs=8, width=4, latency_s=0.02), GridCell(runs=16, width=8, latency_s=0.02))


class LatentStore(InMemoryDocumentStore):
    """An in-memory store answering every call ``latency_s`` late, as a database across a network does.

    The wait is before the call and outside the store's lock: a round-trip's latency, not one connection serialising
    every caller.
    """

    def __init__(self, latency_s: float) -> None:
        """Answer every call ``latency_s`` late."""
        super().__init__()
        self.latency_s = latency_s

    def get(self, doc_id: str, scope_id: str) -> dict[str, Any] | None:
        """Read one document, late."""
        time.sleep(self.latency_s)
        return super().get(doc_id, scope_id)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Read one document and its etag, late."""
        time.sleep(self.latency_s)
        return super().get_with_etag(doc_id, scope_id)

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Read several documents, late."""
        time.sleep(self.latency_s)
        return super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        """Query by document type, late."""
        time.sleep(self.latency_s)
        return super().by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )

    def iter_by_doc_type(self, doc_type: str, scope_id: str) -> Iterator[str]:
        """List ids by document type, late."""
        time.sleep(self.latency_s)
        return super().iter_by_doc_type(doc_type, scope_id)

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Write one document, late."""
        time.sleep(self.latency_s)
        super().upsert(document, if_match=if_match, create_only=create_only)

    def merge_fields(self, doc_id: str, scope_id: str, fields: dict[str, Any]) -> bool:
        """Merge fields into one document, late."""
        time.sleep(self.latency_s)
        return super().merge_fields(doc_id, scope_id, fields)

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Delete one document, late."""
        time.sleep(self.latency_s)
        return super().delete(doc_id, scope_id)


def _grid_host(cell: GridCell) -> OpsHost:
    """The toy host over a late store: every run of ``cell``'s launches admitted at once, at ``cell``'s width."""
    storage = EvalStorage(LatentStore(cell.latency_s))
    storage.save_template(toyhost_template())
    client = ScriptedExtractionClient(tuple(type(s)(**{**vars(s), "latency_s": EXTRACT_S}) for s in TOY_SCRIPTS))
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(
        update={"max_admitted_runs": cell.runs, "max_concurrent_cells": cell.width}
    )
    launch, _client = toyhost_launch_host(
        storage=storage,
        client=client,
        settings=lambda: settings,
        job_lease_s=LEASE_S,
        kind_max_concurrent_cells=cell.width,
    )
    return OpsHost(launch=launch)


async def _settled(host: OpsHost, job_id: str, cell: GridCell) -> Any:
    """Poll a run's job until it has ended; a run still going long after it should have ended fails, naming the cell."""
    try:
        async with asyncio.timeout(60):
            while not (status := await job_poll(host, job_id, TOYHOST_SCOPE)).done:
                await asyncio.sleep(0.05)
    except TimeoutError:
        pytest.fail(f"[{cell}] {job_id} had not ended a minute after its launch")
    return status


def _runs(host: OpsHost) -> list[EvalRun]:
    return host.eval_host.storage.query_eval_runs(TOYHOST_SCOPE)


@pytest.mark.integration
@pytest.mark.parametrize("cell", GRID, ids=str)
async def test_a_launch_keeps_every_job_lease_across_the_grid(cell: GridCell) -> None:
    host = off_loop(_grid_host, cell)
    arguments = LaunchArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=list(RUN_MODELS),
        k_runs=cell.repeats,
    )
    started = await asyncio.gather(
        *(run_launch(host, arguments, TOYHOST_SCOPE) for _ in range(cell.runs // len(RUN_MODELS)))
    )
    launches = [status for one in started for status in await await_job(host, one, TOYHOST_SCOPE)]
    run_ids = [run_id for launch in launches for run_id in launch.run_ids]
    runs = [await _settled(host, run_job_id(run_id), cell) for run_id in run_ids]

    assert len(run_ids) == cell.runs, f"[{cell}] every run was launched: {[launch.detail for launch in launches]}"
    for job in [*launches, *runs]:
        assert job.state != "lost" and not job.faults, (
            f"[{cell}] {job.job_id} lost its lease: state {job.state}, {job.detail}, faults {job.faults}"
        )
    for run in off_loop(_runs, host):
        assert run.lost is None and not run.faults, f"[{cell}] run {run.id} lapsed: {run.lost}, {run.faults}"
        assert run.status == "completed", f"[{cell}] run {run.id} ended {run.status}: {run.cancellation_reason}"
        assert run.completed_at is not None
        lasted = (datetime.fromisoformat(run.completed_at) - datetime.fromisoformat(run.created_at)).total_seconds()
        assert lasted >= 2 * LEASE_S, f"[{cell}] run {run.id} lasted {lasted:.1f}s, not two leases: it proves nothing"
