"""A launch keeps its job leases across a grid of runs, cell width and store latency, and the grid can tell when not.

The lease tests beside this one drive :class:`~threetears.evals.kernel.JobLeases` directly, or one launch at a time.
This drives the whole path — launches through the operations layer, runs on the job manager, cells writing their
results and progress, every record renewed on the leases' own pool — over a store that answers every call late, and
asserts that no record anywhere in the grid lapsed: none is written lost, none carries a lease fault, and every run
completes.

The lease is short so a run outlives it by a quarter or more, and the store's latency is a far larger share of
that lease than a production store's is of a production lease (milliseconds against a minute): along the latency axis
a contended renewal spends most of its interval on the store, so a renewal that would lose its lease at some width or
latency loses it here first. A lapse names its grid cell. The control cell starves the renewals (the leases' pool cut
to one thread at the higher latency) and must report a lapse, so a green grid is a grid that could have gone red.

The grid is kept to about half a minute. It is marked with an xdist group of its own (:data:`pytestmark`); where the
suite's own grouping puts it among the package's other tests instead, that half minute is what it adds.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from threetears.evals.kernel import EvalStorage, job_leases
from threetears.evals.ops import LaunchArguments, OpsHost, await_job, job_poll, run_job_id, run_launch
from threetears.evals.schema import EvalRun
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.off_loop import off_loop

#: The lease every grid cell runs under, renewed every third of it (0.8 s). Long enough that a healthy renewal clears
#: a scheduling stall on a loaded machine running the whole suite (it has two thirds of the lease, 1.6 s, before a
#: lapse), short enough that the grid stays quick.
LEASE_S = 2.4

#: How long one extraction takes. Each run executes two waves of cells, so it outlives :data:`OUTLIVES_LEASES` leases
#: (the test holds every run to that, or it would prove nothing about keeping them), while its cells stay few.
EXTRACT_S = 1.6

#: How many leases every run must outlive for its cell to count: past one lease, through at least three renewals.
OUTLIVES_LEASES = 1.25

#: The xdist group the grid is marked with.
pytestmark = pytest.mark.xdist_group("evals-lease-grid")


@dataclass(frozen=True)
class GridCell:
    """One point of the grid: runs in flight, each run's cells at once, the store's latency, and each run's repeats.

    ``pool_workers``, when set, cuts the leases' own pool to that many threads: the control cell, starved on purpose.
    """

    runs: int
    width: int
    latency_s: float
    repeats: int
    pool_workers: int | None = None

    def __str__(self) -> str:
        starved = f", lease pool cut to {self.pool_workers}" if self.pool_workers is not None else ""
        return f"{self.runs} runs x width {self.width} at {self.latency_s * 1000:g} ms per store call{starved}"


#: The production shape (8 runs of 4 cells in flight each, 32 at once), the same shape at a latency where a renewal
#: that loses its three attempts to the run's own writes spends most of its interval on the store (six round-trips,
#: 0.48 s of 0.8 s), and a wider shape (16 runs of 8, 128 at once).
GRID = (
    GridCell(runs=8, width=4, latency_s=0.02, repeats=2),
    GridCell(runs=8, width=4, latency_s=0.08, repeats=2),
    GridCell(runs=16, width=8, latency_s=0.02, repeats=3),
)

#: The control: the higher-latency cell of :data:`GRID`, which keeps every lease, with the leases' pool cut to one
#: thread. Its twelve leases (eight runs, four launches) then renew one after another, at about 0.16 s each uncontended
#: and up to 0.48 s contended, a round about as long as the lease or longer, so a lease lapses: the grid must say so, naming
#: this cell.
CONTROL = GridCell(runs=8, width=4, latency_s=0.08, repeats=2, pool_workers=1)


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
    """The toy host over a late store: every run of ``cell``'s launches admitted and executing at once, at its width.

    The job manager runs as many jobs at once as the cell has runs (``LaunchHost.max_concurrent_jobs``), so every run is
    in flight together, as at the production shape; at the engine's default of two, most would wait their turn.
    """
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
    return OpsHost(launch=dataclasses.replace(launch, max_concurrent_jobs=cell.runs))


async def _settled(host: OpsHost, job_id: str, cell: GridCell) -> Any:
    """Poll a job until it has ended; one still going long after it should have ended fails, naming the cell."""
    try:
        async with asyncio.timeout(90):
            while not (status := await job_poll(host, job_id, TOYHOST_SCOPE)).done:
                await asyncio.sleep(0.05)
    except TimeoutError:
        pytest.fail(f"[{cell}] {job_id} had not ended long after its launch")
    return status


def _runs(host: OpsHost) -> list[EvalRun]:
    return host.eval_host.storage.query_eval_runs(TOYHOST_SCOPE)


async def _drive(cell: GridCell) -> tuple[list[str], list[EvalRun]]:
    """Launch ``cell``'s runs and wait for every one to end: each lapse found, naming the cell, and the runs."""
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
    jobs = [*launches, *[await _settled(host, run_job_id(run_id), cell) for run_id in run_ids]]
    runs = off_loop(_runs, host)
    lapses = [
        f"[{cell}] {job.job_id} lost its lease: state {job.state}, {job.detail}, faults {job.faults}"
        for job in jobs
        if job.state == "lost" or job.faults
    ] + [f"[{cell}] run {run.id} lapsed: {run.lost}, {run.faults}" for run in runs if run.lost or run.faults]
    return lapses, runs


@pytest.mark.parametrize("cell", GRID, ids=str)
async def test_a_launch_keeps_every_job_lease_across_the_grid(cell: GridCell) -> None:
    lapses, runs = await _drive(cell)

    assert lapses == [], "\n".join(lapses)
    assert len(runs) == cell.runs, f"[{cell}] {len(runs)} of {cell.runs} runs were launched"
    for run in runs:
        assert run.status == "completed", f"[{cell}] run {run.id} ended {run.status}: {run.cancellation_reason}"
        assert run.completed_at is not None
        lasted = (datetime.fromisoformat(run.completed_at) - datetime.fromisoformat(run.created_at)).total_seconds()
        assert lasted >= OUTLIVES_LEASES * LEASE_S, (
            f"[{cell}] run {run.id} lasted {lasted:.1f}s, under {OUTLIVES_LEASES} leases: it proves nothing"
        )


async def test_the_grid_reports_a_lapse_naming_its_cell_when_renewals_are_starved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert CONTROL.pool_workers is not None
    monkeypatch.setattr(job_leases, "LEASE_POOL_WORKERS", CONTROL.pool_workers)

    lapses, _ = await _drive(CONTROL)

    assert lapses, f"[{CONTROL}] starved renewals lapsed no lease: the grid cannot see a lapse"
    assert all(line.startswith(f"[{CONTROL}] ") for line in lapses), lapses
