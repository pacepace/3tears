"""A healthy process under load keeps its job leases: renewal never waits on job work, and never gives up on contention.

The production defect (a launch of 8 runs at width 4, so about 32 cells in flight): every lease lapsed at its first
expiry and the runs were written lost while the process was alive. Renewals ran one after another on the host's
blocking executor, behind the cells' store work, and a renewal that lost three conditional writes to the holder's own
progress writes waited a whole interval. Pinned, each against the shape that failed:

- **Starved host pool**: leases renew though every worker of the loop's default executor is busy for longer than a lease.
- **One slow record** delays no other record's renewal.
- **Contended renewal**: tried again within the interval, after a short wait, not a whole interval later.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any


from threetears.evals.kernel import RUN_JOB, EvalStorage, JobLeases
from threetears.evals.schema import EvalRun
from threetears.evals.schema.store_port import StoreConflict
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.off_loop import off_loop

SCOPE = "scope-leases"

#: A lease short enough to keep these tests fast; renewed every third of it (0.6 s). Twice the 0.9 s these first
#: used: the margin each test leaves between a healthy renewal and a lapse (a lease less the renewal interval, 1.2 s)
#: must outlast a scheduling stall on a machine running the whole suite, which a 0.6 s margin did not.
LEASE_S = 1.8


def _storage(store: InMemoryDocumentStore | None = None) -> EvalStorage:
    return EvalStorage(store or InMemoryDocumentStore())


def _hold(leases: JobLeases, storage: EvalStorage, run_id: str) -> None:
    """Store a running run carrying a fresh lease and start renewing it (as a job manager starts a job)."""
    run = make_eval_run(id=run_id, scope_id=SCOPE, status="running", lease=off_loop(leases.grant))
    off_loop(storage.save_job_record, run.to_dict(), if_match=None, create_only=True)
    leases.hold(RUN_JOB, run_id, SCOPE, cancel=lambda _reason: None, stop=lambda: None)


def _stored(storage: EvalStorage, run_id: str) -> EvalRun:
    record, _ = off_loop(storage.load_job_record, EvalRun, run_id, SCOPE)
    assert record is not None
    return record


def _live(storage: EvalStorage, run_id: str) -> bool:
    """Whether the stored run is unended, unlost and its lease unexpired by the store's time."""
    run = _stored(storage, run_id)
    assert run.lease is not None
    return run.lost is None and run.status == "running" and run.lease.expires() > datetime.now(UTC)


class _SpyLeases(JobLeases):
    """Leases counting how often a holder was stopped for a whole unrenewed lease."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.expired: list[str] = []

    async def _expire(self, held: Any) -> None:
        self.expired.append(held.record_id)
        await super()._expire(held)


async def test_leases_renew_though_every_worker_of_the_hosts_pool_is_busy_for_longer_than_a_lease() -> None:
    storage = _storage()
    host_pool = ThreadPoolExecutor(max_workers=2)
    asyncio.get_running_loop().set_default_executor(host_pool)
    leases = _SpyLeases(storage, lease_s=LEASE_S)
    try:
        # the host's cells: every worker occupied for well over two leases
        for _ in range(2):
            host_pool.submit(time.sleep, 5.2)
        ids = [f"run-{n}" for n in range(9)]
        for run_id in ids:
            _hold(leases, storage, run_id)

        await asyncio.sleep(3.8)  # past two lease durations: a lease not renewed has lapsed

        assert all(_live(storage, run_id) for run_id in ids), "every lease was renewed while the host's pool was busy"
        assert leases.expired == [] and not any(leases.fenced(run_id) for run_id in ids)
    finally:
        await leases.shut_down()
        host_pool.shutdown(wait=True)


class _SlowRecordStore(InMemoryDocumentStore):
    """A store whose writes of one record take a long time, as a stalled round-trip does."""

    def __init__(self, slow_id: str, delay_s: float) -> None:
        super().__init__()
        self.slow_id = slow_id
        self.delay_s = delay_s

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Write, after the delay when the record is the slow one and the write is not its creation."""
        if document.get("id") == self.slow_id and not create_only:
            time.sleep(self.delay_s)
        super().upsert(document, if_match=if_match, create_only=create_only)


async def test_one_slow_record_delays_no_other_records_renewal() -> None:
    store = _SlowRecordStore("run-slow", 3.0)  # a stall longer than a lease
    storage = _storage(store)
    leases = _SpyLeases(storage, lease_s=LEASE_S)
    try:
        for run_id in ("run-slow", "run-b", "run-c"):  # the slow one is renewed first
            _hold(leases, storage, run_id)

        await asyncio.sleep(2.8)  # past one lease: a record waiting on the slow one would have lapsed

        assert _live(storage, "run-b") and _live(storage, "run-c")
        assert "run-b" not in leases.expired and "run-c" not in leases.expired
    finally:
        await leases.shut_down()


class _ContendedStore(InMemoryDocumentStore):
    """A store refusing the first ``refusals`` conditional writes of a record, as the holder's own writes make it."""

    def __init__(self) -> None:
        super().__init__()
        self.refusals = 0
        self.landed_at: float | None = None

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Refuse a conditional write while refusals remain; note when the first one lands."""
        if if_match is not None:
            if self.refusals > 0:
                self.refusals -= 1
                raise StoreConflict("another writer moved the record")
            if self.landed_at is None:
                self.landed_at = time.monotonic()
        super().upsert(document, if_match=if_match, create_only=create_only)


async def test_a_renewal_that_loses_its_races_is_tried_again_within_the_interval() -> None:
    store = _ContendedStore()
    storage = _storage(store)
    leases = _SpyLeases(storage, lease_s=3.0)  # renewed every second
    try:
        _hold(leases, storage, "run-a")
        store.refusals = 6  # two whole renewals' attempts lost
        started = time.monotonic()

        await asyncio.sleep(2.4)

        assert store.landed_at is not None, "the renewal landed"
        assert store.landed_at - started < 1.9, "after the first interval and a short backoff, not another interval"
        assert leases.expired == [] and not leases.fenced("run-a")
    finally:
        await leases.shut_down()

