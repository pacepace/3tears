"""Job liveness lives on the job's record, as a lease: two processes over one store answer every job alike.

Two "processes" here are two launching hosts over one store — each with its own job manager and its own
:class:`~threetears.evals.kernel.JobLeases`, so its own holder id — which is what two processes sharing a database
are to the engine. Pinned:

- **A job another process runs reads running here**, and a cancel asked here reaches it: the request is written on
  the record and the holder ends the job ``cancelled`` at its next renewal.
- **A holder that stops renewing is lost once its lease lapses**: the first reader writes it lost, once, and every
  later read in either process answers from that write.
- **Fencing**: a holder whose record was written lost writes nothing more — its paid analysis is not stored, its
  run's progress write stops the run, and its ending does not overwrite the loss; a holder that finds its own lease
  lapsed at renewal writes itself lost and stops.
- **A graceful shutdown** writes every job its process holds lost at once, its cause the shutdown, lease released.
- **Store time**: a store offering a clock stamps and reads every lease; a clock with no zone is refused.
- **Old records**: a record that has not ended and carries no lease reads lost.
- **Scope**: another scope's job reads absent, on poll and on cancel.
- **Sweeps** are cancelled from another process, and a lapsed one launches no further arm.
- **One generation of a campaign at a time**: two processes racing a start both reach the claim of the campaign's
  generation slot; exactly one wins, and the other is refused naming the live attempt. A slot whose attempt has
  ended or lapsed is taken over.
- **The reclaim** writes lapsed runs lost and leaves any process's live one alone.
- **A holder's own writes keep its lease**: a run whose renewals all lose to its own progress writes keeps its lease
  and still hears a cancel from another process; an own write extends only a lease this process holds, never one the
  store's clock reads lapsed, and one whose clock read fails still lands.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.evals import kernel, schema
from threetears.evals.kernel import (
    ANALYSIS_JOB,
    ConflictError,
    RUN_JOB,
    SHUTDOWN_CAUSE,
    SWEEP_JOB,
    EvalAnalysisAttempt,
    EvalStorage,
    EvalSweep,
    JobFenced,
    JobLeases,
    NotFoundError,
    StorageError,
    SweepArmRecord,
    ValidationFailedError,
    request_job_cancel,
    require_held,
    settle_job_record,
)
from threetears.evals.kernel import job_leases
from threetears.evals.ops import OpsHost, analyses_list, analysis_generate, job_cancel, job_poll, run_job_id, run_launch
from threetears.evals.ops import LaunchArguments, SweepArguments, SweepArm, generation_key, sweep_launch
from threetears.evals.ops import sweeps
from threetears.evals.run import sweep_abandoned_runs
from threetears.evals.schema import EvalRun, JobLease, StoreConflict
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.kind import TOY_SCRIPTS, ScriptedExtractionClient
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture, settled
from packages.evals.tests.off_loop import off_loop

#: The sweep's arms, by their prompt style.
STYLES = ("terse", "standard", "verbose")

#: When the clocked store's clock starts.
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class ClockedStore(InMemoryDocumentStore):
    """An in-memory store keeping a clock of its own, which the test sets — a database server's clock, in effect."""

    def __init__(self, at: datetime = T0) -> None:
        """Start the clock at ``at``."""
        super().__init__()
        self.at = at

    def now(self) -> datetime:
        """The store's time, as the test set it."""
        return self.at


class ZonelessClockStore(InMemoryDocumentStore):
    """A store whose clock answers a time with no zone, which names no instant."""

    def now(self) -> datetime:
        """A naive time."""
        return datetime(2026, 10, 10, 12, 0)


def _other_process(host: OpsHost) -> OpsHost:
    """Another process over the same store: its own job manager and leases, so its own holder id."""
    return dataclasses.replace(host, launch=dataclasses.replace(host.launch))


def _attempt(host: OpsHost, job_id: str) -> EvalAnalysisAttempt:
    attempt, _ = host.eval_host.storage.load_job_record(EvalAnalysisAttempt, job_id.rsplit(":", 1)[1], TOYHOST_SCOPE)
    assert attempt is not None
    return attempt


def _run(host: OpsHost, run_id: str) -> EvalRun:
    run, _ = host.eval_host.storage.load_job_record(EvalRun, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run


async def _until(condition: Any, *, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def _held_extraction(monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Hold every extraction until the returned event is set, so a launched run stays running."""
    gate = asyncio.Event()
    extract = ScriptedExtractionClient.extract

    async def held(self: ScriptedExtractionClient, request: Any) -> Any:
        await gate.wait()
        return await extract(self, request)

    monkeypatch.setattr(ScriptedExtractionClient, "extract", held)
    return gate


async def _running_launch(host: OpsHost) -> tuple[str, str]:
    """Launch one arm on ``host`` and wait until its run reads running; the launch's job id and the run's id."""
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[RUN_MODELS[0]]
    )
    (job,) = (await run_launch(host, arguments, TOYHOST_SCOPE)).jobs
    async with asyncio.timeout(10):
        while not (run_ids := (await job_poll(host, job.job_id, TOYHOST_SCOPE)).run_ids):
            await asyncio.sleep(0.01)
        while (await job_poll(host, run_job_id(run_ids[0]), TOYHOST_SCOPE)).status != "running":
            await asyncio.sleep(0.01)
    return job.job_id, run_ids[0]


async def _running_run(host: OpsHost) -> str:
    """Launch one arm on ``host`` and wait until its run reads running; the run's id."""
    return (await _running_launch(host))[1]


class _Heard:
    """What a holder's process told anyone about its runs: each run-end hook call, and each broadcast."""

    def __init__(self) -> None:
        """Start with nothing heard."""
        self.ends: list[tuple[str, str]] = []
        self.broadcasts: list[tuple[str, dict[str, Any]]] = []

    async def run_end(self, run_id: str, scope_id: str, status: str) -> None:
        """The run-end hook: record the run and the status."""
        self.ends.append((run_id, status))

    def broadcast(self, run_id: str, payload: dict[str, Any]) -> None:
        """The progress broadcast: record it."""
        self.broadcasts.append((run_id, payload))


def _heard(host: OpsHost) -> tuple[OpsHost, _Heard]:
    """``host`` with a run-end hook and a broadcast that record what its process says about its runs."""
    heard = _Heard()
    launch = dataclasses.replace(host.launch, on_run_end=heard.run_end, on_job_progress=heard.broadcast)
    return dataclasses.replace(host, launch=launch), heard


# =============================================================================
# Another process's job: running there, cancelled from there
# =============================================================================


async def test_a_generation_another_process_runs_reads_running_and_is_cancelled_from_there() -> None:
    gate = asyncio.Event()
    holder = off_loop(ops_fixture, gate=gate, job_lease_s=0.3)
    elsewhere = _other_process(holder.host)
    (job,) = (await analysis_generate(holder.host, holder.campaign.id, TOYHOST_SCOPE)).jobs
    await asyncio.sleep(0)

    polled = await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)
    assert (polled.state, polled.done) == ("running", False), "the other process reads the holder's live job"

    asked = await job_cancel(elsewhere, job.job_id, TOYHOST_SCOPE, reason="stop it from over here")
    # request-then-converge: the holder ends it at its next renewal, which with a 0.3 s lease may come before
    # job_cancel's own read-back on a loaded machine; what the other process may never do is end it itself
    assert asked.state in ("running", "cancelled"), f"the other process's cancel reads {asked.state!r}"
    ended = await settled(elsewhere, job.job_id)

    assert (ended.state, ended.status) == ("cancelled", "cancelled")
    attempt = off_loop(_attempt, holder.host, job.job_id)
    assert attempt.cancel_request_reason == "stop it from over here" and attempt.lost is None
    assert not gate.is_set(), "the holder stopped it while it was held, not after it finished"


async def test_a_run_another_process_runs_is_cancelled_from_there_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _held_extraction(monkeypatch)
    holder = off_loop(ops_fixture, job_lease_s=0.3)
    elsewhere = _other_process(holder.host)
    run_id = await _running_run(holder.host)

    assert (await job_poll(elsewhere, run_job_id(run_id), TOYHOST_SCOPE)).state == "running"
    await job_cancel(elsewhere, run_job_id(run_id), TOYHOST_SCOPE, reason="wrong template")
    ended = await settled(elsewhere, run_job_id(run_id))

    assert (ended.state, ended.status) == ("cancelled", "cancelled")
    assert ended.detail is not None and "wrong template" in ended.detail
    assert off_loop(_run, holder.host, run_id).lost is None


# =============================================================================
# A holder that stops renewing: lost after its lease, written once, and fenced
# =============================================================================


async def test_a_holder_that_stops_renewing_reads_lost_after_its_lease_and_the_loss_is_written_once() -> None:
    gate = asyncio.Event()
    store = ClockedStore()
    holder = off_loop(ops_fixture, gate=gate, store=store)  # a 60s lease: no renewal comes due while the test runs
    elsewhere = _other_process(holder.host)
    (job,) = (await analysis_generate(holder.host, holder.campaign.id, TOYHOST_SCOPE)).jobs
    await asyncio.sleep(0)
    assert (await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)).state == "running"

    store.at = T0 + timedelta(seconds=59)
    assert (await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)).state == "running", "inside its lease"

    store.at = T0 + timedelta(seconds=61)
    lost = await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)
    assert (lost.state, lost.status, lost.done) == ("lost", "cancelled", True)
    assert lost.detail is not None and holder.host.launch.job_leases.holder_id in lost.detail
    written = off_loop(_attempt, holder.host, job.job_id).lost
    assert written is not None and written.at == store.at.isoformat()

    store.at = T0 + timedelta(seconds=90)
    again = await job_poll(holder.host, job.job_id, TOYHOST_SCOPE)
    assert again.state == "lost", "the holder's own process reads what the other one wrote"
    assert off_loop(_attempt, holder.host, job.job_id).lost == written, "the loss is written once, never recomputed"
    attempt_id = job.job_id.rsplit(":", 1)[1]
    record, wrote = off_loop(settle_job_record, holder.host.eval_host.storage, ANALYSIS_JOB, attempt_id, TOYHOST_SCOPE)
    assert record is not None and wrote is False

    # Fenced: the holder, released, finds its record lost and stores nothing — not its analysis, not its ending.
    gate.set()
    await _until(lambda: not holder.host.launch.job_manager.is_task_active(attempt_id))
    assert off_loop(analyses_list, holder.host.eval_host, holder.campaign.id, TOYHOST_SCOPE).analyses == []
    attempt = off_loop(_attempt, holder.host, job.job_id)
    assert (attempt.outcome, attempt.lost) == ("cancelled", written)


class LostOnStoreStore(ClockedStore):
    """Writes a running generation's attempt lost — as another process's reader would — once its analysis is stored.

    The race a check before storing cannot close: the holder found its attempt held, stored its paid analysis, and
    only then did the lease lapse and a reader write the loss.
    """

    def __init__(self) -> None:
        """Lose nothing until told to."""
        super().__init__()
        self.lose_next = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Write; then, for an analysis while told to, lapse the clock past the lease and settle every attempt."""
        super().upsert(document, if_match=if_match, create_only=create_only)
        if self.lose_next and document.get("doc_type") == "eval_analysis":
            self.lose_next = False
            self.at = self.at + timedelta(seconds=61)
            storage = EvalStorage(self)
            scope_id = str(document["scope_id"])
            for attempt in storage.list_analysis_attempts_by_campaign(str(document["campaign_id"]), scope_id):
                settle_job_record(storage, ANALYSIS_JOB, attempt.id, scope_id)


async def test_a_generation_whose_attempt_is_lost_while_it_stores_withdraws_what_it_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import packages.evals.tests.ops_support as support

    memo = support.memo_payload

    def durable_memo(bundle: Any) -> dict[str, Any]:
        payload = memo(bundle)
        payload["findings"][0]["durable"] = "the wide chunk is slower per document than the narrow one"
        return payload

    monkeypatch.setattr(support, "memo_payload", durable_memo)
    store = LostOnStoreStore()
    fixture = off_loop(ops_fixture, store=store)
    storage = fixture.host.eval_host.storage
    (first,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, first.job_id)).state == "completed"
    (kept,) = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    ledger = off_loop(storage.query_insights, TOYHOST_SCOPE)
    assert ledger and all(insight.source_analysis_id == kept.id for insight in ledger)

    store.lose_next = True
    (second,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    attempt_id = second.job_id.rsplit(":", 1)[1]
    await _until(lambda: not fixture.host.launch.job_manager.is_task_active(attempt_id))

    assert (await job_poll(fixture.host, second.job_id, TOYHOST_SCOPE)).state == "lost"
    analyses = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    assert [analysis.id for analysis in analyses] == [kept.id], "the lost attempt keeps no analysis"
    assert off_loop(storage.query_insights, TOYHOST_SCOPE) == ledger, (
        "the insights it restated are put back as they stood"
    )


class UnwithdrawableStore(LostOnStoreStore):
    """Loses the attempt as :class:`LostOnStoreStore` does, and then refuses to delete any analysis."""

    def delete(self, doc_id: str, scope_id: str) -> bool:
        """Refuse an analysis's delete, as a store fault would; delete anything else."""
        stored = self.get(doc_id, scope_id)
        if stored is not None and stored.get("doc_type") == "eval_analysis":
            raise RuntimeError("the store refused the delete")
        return super().delete(doc_id, scope_id)


async def test_a_withdrawal_that_fails_is_not_reported_as_done(caplog: pytest.LogCaptureFixture) -> None:
    """The fence said the analysis "and its insights were withdrawn" when the withdrawal had failed part-way."""
    store = UnwithdrawableStore()
    fixture = off_loop(ops_fixture, store=store)
    store.lose_next = True
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    attempt_id = job.job_id.rsplit(":", 1)[1]
    with caplog.at_level("WARNING"):
        await _until(lambda: not fixture.host.launch.job_manager.is_task_active(attempt_id))

    assert (await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)).state == "lost"
    (left,) = off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses
    stopped = [record.getMessage() for record in caplog.records if "stopped:" in record.getMessage()]
    assert stopped and "were withdrawn" not in stopped[0], stopped
    assert "withdrawing what it stored failed part-way (RuntimeError: the store refused the delete)" in stopped[0]
    assert f"analysis '{left.id}' may still be stored" in stopped[0]


async def test_a_holder_whose_own_lease_lapsed_writes_itself_lost_at_renewal_and_stops() -> None:
    gate = asyncio.Event()
    store = ClockedStore()
    holder = off_loop(ops_fixture, gate=gate, store=store, job_lease_s=0.3)
    (job,) = (await analysis_generate(holder.host, holder.campaign.id, TOYHOST_SCOPE)).jobs
    attempt_id = job.job_id.rsplit(":", 1)[1]
    await asyncio.sleep(0)

    store.at = T0 + timedelta(seconds=30)  # the process stalled past its lease
    await _until(lambda: not holder.host.launch.job_manager.is_task_active(attempt_id))

    attempt = off_loop(_attempt, holder.host, job.job_id)
    assert attempt.outcome == "cancelled" and attempt.lost is not None
    assert "lapsed" in attempt.lost.cause and attempt.error == f"lost: {attempt.lost.cause}", (
        "the cancelled generation's own ending did not overwrite the loss"
    )
    assert not gate.is_set()


async def test_a_run_whose_record_was_written_lost_elsewhere_stops_at_its_next_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _held_extraction(monkeypatch)
    store = ClockedStore()
    holder, heard = _heard(off_loop(ops_fixture, store=store).host)
    elsewhere = _other_process(holder)
    run_id = await _running_run(holder)
    assert (await job_poll(elsewhere, run_job_id(run_id), TOYHOST_SCOPE)).state == "running"

    store.at = T0 + timedelta(seconds=61)
    lost = await job_poll(elsewhere, run_job_id(run_id), TOYHOST_SCOPE)
    assert (lost.state, lost.status) == ("lost", "cancelled")
    written = off_loop(_run, holder, run_id)
    assert written.completeness is not None, "the loss reconstructed what the run delivered"
    heard.broadcasts.clear()

    store.at = T0 + timedelta(seconds=75)  # the holder's own ending would be stamped later than the loss
    gate.set()
    await _until(lambda: not holder.launch.job_manager.is_active(run_id))
    after = off_loop(_run, holder, run_id)
    assert (after.status, after.lost, after.completeness) == ("cancelled", written.lost, written.completeness), (
        "the fenced run wrote nothing more: not its progress, its completeness or its ending"
    )
    assert (after.completed_at, after.cancellation_reason) == (written.completed_at, written.cancellation_reason), (
        "the loss's own ending stands: its time and its reason"
    )
    assert heard.ends == [] and heard.broadcasts == [], "the process that lost the run told nobody it ended"


async def test_a_lost_run_cancelled_by_its_old_holder_keeps_the_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """The holder's operator cancels a run another process already wrote lost: the cancel's ending is refused."""
    _held_extraction(monkeypatch)
    store = ClockedStore()
    holder, heard = _heard(off_loop(ops_fixture, store=store).host)
    elsewhere = _other_process(holder)
    run_id = await _running_run(holder)
    store.at = T0 + timedelta(seconds=61)
    assert (await job_poll(elsewhere, run_job_id(run_id), TOYHOST_SCOPE)).state == "lost"
    written = off_loop(_run, holder, run_id)
    heard.broadcasts.clear()

    assert holder.launch.job_manager.cancel_job(run_id, reason="operator said stop")
    await _until(lambda: not holder.launch.job_manager.is_active(run_id))

    after = off_loop(_run, holder, run_id)
    assert (after.completed_at, after.cancellation_reason, after.lost) == (
        written.completed_at,
        written.cancellation_reason,
        written.lost,
    )
    assert after.cancellation_reason is not None and after.cancellation_reason.startswith("lost: ")
    assert heard.ends == [] and heard.broadcasts == []


class RenewalRefusingStore(ClockedStore):
    """A clocked store that refuses, once told to, every write moving a record's lease and no other.

    The writes moving a lease are its renewals and the holder's own writes of its record, each of which extends it. A
    store a holder cannot renew through while the store's clock has not reached the lease's expiry: no reader
    writes the loss yet, so only the holder's own count can tell it its lease is gone.
    """

    def __init__(self) -> None:
        """Start accepting every write."""
        super().__init__()
        self.refusing = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Refuse a write that moves a stored record's lease while refusing; write anything else."""
        if self.refusing and document.get("lease") is not None:
            stored = self.get(str(document["id"]), str(document["scope_id"]))
            if stored is not None and stored.get("lease") != document["lease"]:
                raise RuntimeError("the store refused the renewal")
        super().upsert(document, if_match=if_match, create_only=create_only)


async def test_a_run_whose_renewals_fail_for_a_whole_lease_is_stopped_and_writes_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _held_extraction(monkeypatch)
    store = RenewalRefusingStore()
    holder, heard = _heard(off_loop(ops_fixture, store=store, job_lease_s=0.3).host)
    elsewhere = _other_process(holder)
    run_id = await _running_run(holder)
    store.at = T0 + timedelta(seconds=0.1)  # inside the lease by the store's clock, for the whole test
    store.refusing = True
    heard.broadcasts.clear()

    await _until(lambda: not holder.launch.job_manager.is_active(run_id), timeout=5)

    assert not gate.is_set(), "the holder stopped spending, rather than running on past its lease"
    run = off_loop(_run, holder, run_id)
    assert (run.status, run.lost) == ("running", None), "it wrote no ending of its own: it no longer held the run"
    assert heard.ends == [] and heard.broadcasts == []
    store.refusing = False
    store.at = T0 + timedelta(seconds=1)
    lost = await job_poll(elsewhere, run_job_id(run_id), TOYHOST_SCOPE)
    assert (lost.state, lost.status) == ("lost", "cancelled"), "the next reader writes the loss, as for any lapse"


async def test_a_generation_whose_renewals_fail_for_a_whole_lease_is_stopped_before_its_paid_call() -> None:
    gate = asyncio.Event()
    store = RenewalRefusingStore()
    fixture = off_loop(ops_fixture, gate=gate, store=store, job_lease_s=0.3)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    attempt_id = job.job_id.rsplit(":", 1)[1]
    await asyncio.sleep(0)
    store.at = T0 + timedelta(seconds=0.1)
    store.refusing = True

    await _until(lambda: not fixture.host.launch.job_manager.is_task_active(attempt_id), timeout=5)

    assert not gate.is_set() and all(not writer.calls for writer in fixture.writers), "no call was paid for"
    attempt = off_loop(_attempt, fixture.host, job.job_id)
    assert (attempt.outcome, attempt.lost) == ("running", None), (
        "it wrote no ending of its own, though the store took every write but one moving its lease: it no longer "
        "held the attempt"
    )
    assert off_loop(analyses_list, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).analyses == []
    store.refusing = False
    store.at = T0 + timedelta(seconds=1)
    lost = await job_poll(_other_process(fixture.host), job.job_id, TOYHOST_SCOPE)
    assert (lost.state, lost.status) == ("lost", "cancelled"), "the next reader writes the loss, as for any lapse"


class LeasedWriteRefusingStore(ClockedStore):
    """A clocked store that fails, once told to, every write of a stored record carrying a lease, and no other.

    A store a job's process cannot reach for its own records for a whole lease: its renewals fail, and so do the
    job's own writes of its record, which would otherwise extend the lease as they landed.
    """

    def __init__(self) -> None:
        """Start accepting every write."""
        super().__init__()
        self.refusing = False

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Fail a write over a stored record carrying a lease while refusing; write anything else."""
        if self.refusing and not create_only:
            stored = self.get(str(document["id"]), str(document["scope_id"]))
            if stored is not None and stored.get("lease") is not None:
                raise RuntimeError("the store failed the write")
        super().upsert(document, if_match=if_match, create_only=create_only)


async def test_a_sweep_whose_store_fails_its_writes_for_a_whole_lease_writes_nothing_more_and_launches_nothing_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _held_extraction(monkeypatch)  # its first arm runs until the sweep has let go
    store = LeasedWriteRefusingStore()
    holder = off_loop(
        _sweeping_host, store, job_lease_s=0.9
    )  # not 0.3: its first arm must be running before a loaded loop can lapse the lease
    (job,) = (await sweep_launch(holder, _three_arm_sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    sweep_id = job.job_id.removeprefix("sweep:")
    await _first_arm_running(holder, job.job_id)
    store.at = T0 + timedelta(seconds=0.1)
    store.refusing = True

    await _until(lambda: not holder.launch.job_manager.is_task_active(sweep_id), timeout=5)

    sweep, _ = off_loop(holder.eval_host.storage.load_job_record, EvalSweep, sweep_id, TOYHOST_SCOPE)
    assert sweep is not None and (sweep.outcome, sweep.lost) == ("running", None), (
        "it wrote no cancelled ending of its own once its lease lapsed by its own count"
    )
    assert len(off_loop(holder.eval_host.storage.query_eval_runs, TOYHOST_SCOPE)) == 1, "it launched no further arm"


class SlotClaimFailingStore(InMemoryDocumentStore):
    """Fails the claim's read of a campaign's generation slot — its second read; the first is the early refusal."""

    def __init__(self) -> None:
        """Count the slot's reads."""
        super().__init__()
        self.slot_reads = 0

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Read, failing the second read of a generation slot."""
        if doc_id.startswith("generation-slot:"):
            self.slot_reads += 1
            if self.slot_reads == 2:
                raise RuntimeError("the store dropped the connection")
        return super().get_with_etag(doc_id, scope_id)


async def test_a_start_whose_slot_claim_fails_ends_its_attempt_and_releases_its_client() -> None:
    fixture = off_loop(ops_fixture, store=SlotClaimFailingStore())

    with pytest.raises(RuntimeError, match="dropped the connection"):
        await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)

    (attempt,) = off_loop(
        fixture.host.eval_host.storage.list_analysis_attempts_by_campaign, fixture.campaign.id, TOYHOST_SCOPE
    )
    assert attempt.outcome == "failed" and attempt.error is not None
    assert "generation slot could not be claimed" in attempt.error, "nothing is left reading running"
    assert fixture.writers == [], "the start built no client: that is its job's"


class AttemptRefusingStore(InMemoryDocumentStore):
    """Refuses every write of a generation's attempt record."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Refuse an attempt record; write anything else."""
        if document.get("doc_type") == "eval_analysis_attempt":
            raise RuntimeError("the store refused the attempt")
        super().upsert(document, if_match=if_match, create_only=create_only)


async def test_a_start_whose_attempt_cannot_be_written_starts_nothing() -> None:
    fixture = off_loop(ops_fixture, store=AttemptRefusingStore())

    with pytest.raises(StorageError, match="refused the attempt"):
        await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)

    assert fixture.writers == [], "no client was built"
    assert fixture.host.launch.job_manager.active_task_ids(generation_key(fixture.campaign.id, TOYHOST_SCOPE)) == []


# =============================================================================
# Graceful shutdown
# =============================================================================


async def test_a_graceful_shutdown_writes_every_held_job_lost_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _held_extraction(monkeypatch)
    gate = asyncio.Event()
    store = ClockedStore()
    fixture = off_loop(ops_fixture, gate=gate, store=store)
    holder, heard = _heard(fixture.host)
    elsewhere = _other_process(holder)
    (generation,) = (await analysis_generate(holder, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    launch_id, run_id = await _running_launch(holder)
    heard.broadcasts.clear()

    store.at = T0 + timedelta(seconds=5)
    await holder.launch.job_manager.shutdown()

    for job_id in (generation.job_id, run_job_id(run_id), launch_id):
        status = await job_poll(elsewhere, job_id, TOYHOST_SCOPE)
        assert (status.state, status.detail) == ("lost", SHUTDOWN_CAUSE), job_id
    assert off_loop(_attempt, holder, generation.job_id).lease is None, "the lease was released"
    run = off_loop(_run, holder, run_id)
    assert run.lease is None
    # What a shutdown writes is the loss, stamped in the store's time as it shut down; the runs' own
    # cancelled endings, written as they unwind, are refused, and nobody is told they ended.
    assert (run.status, run.completed_at, run.cancellation_reason) == (
        "cancelled",
        store.at.isoformat(),
        f"lost: {SHUTDOWN_CAUSE}",
    )
    assert heard.ends == [] and heard.broadcasts == []


# =============================================================================
# Store time
# =============================================================================


async def test_a_store_offering_a_clock_stamps_every_lease_in_its_time() -> None:
    gate = asyncio.Event()
    holder = off_loop(ops_fixture, gate=gate, store=ClockedStore())
    (job,) = (await analysis_generate(holder.host, holder.campaign.id, TOYHOST_SCOPE)).jobs

    lease = off_loop(_attempt, holder.host, job.job_id).lease
    assert lease == JobLease(
        holder=holder.host.launch.job_leases.holder_id, expires_at=(T0 + timedelta(seconds=60)).isoformat()
    )
    gate.set()
    await settled(holder.host, job.job_id)


def test_a_store_without_a_clock_is_read_on_this_process_clock_and_a_zoneless_clock_is_refused() -> None:
    before = datetime.now(UTC)
    assert before <= EvalStorage(InMemoryDocumentStore()).store_time() <= datetime.now(UTC)
    assert EvalStorage(ClockedStore()).store_time() == T0
    with pytest.raises(StorageError, match="no zone"):
        EvalStorage(ZonelessClockStore()).store_time()


def test_a_lease_must_last_some_time() -> None:
    with pytest.raises(ValueError, match="lease_s must be positive"):
        JobLeases(EvalStorage(InMemoryDocumentStore()), lease_s=0)
    fixture = ops_fixture()
    with pytest.raises(ValueError, match="lease_s must be positive"):
        dataclasses.replace(fixture.host.launch, job_lease_s=-1.0)


# =============================================================================
# Records written before leases
# =============================================================================


async def test_a_run_not_ended_with_no_lease_reads_lost_and_cannot_be_cancelled() -> None:
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    run = off_loop(storage.load_eval_run, fixture.campaign.run_ids[0], TOYHOST_SCOPE)
    assert run is not None
    off_loop(
        storage.save_eval_run, run.model_copy(update={"status": "running", "completed_at": None, "completeness": None})
    )

    lost = await job_poll(fixture.host, run_job_id(run.id), TOYHOST_SCOPE)
    assert (lost.state, lost.status, lost.done) == ("lost", "cancelled", True)
    assert lost.detail is not None and "carries no lease" in lost.detail
    assert "ended before" in lost.detail, "the lost cause names why nothing is renewing it"
    stored = off_loop(_run, fixture.host, run.id)
    assert stored.completeness is not None and stored.completeness.counted_from == "stored_results"
    assert stored.cancellation_reason == f"lost: {lost.detail}"

    with pytest.raises(ValidationFailedError, match="it was lost: it carries no lease"):
        await job_cancel(fixture.host, run_job_id(run.id), TOYHOST_SCOPE)


async def test_a_sweep_reading_running_with_no_lease_reads_lost() -> None:
    fixture = off_loop(ops_fixture)
    sweep = EvalSweep(
        scope_id=TOYHOST_SCOPE,
        campaign_id=fixture.campaign.id,
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        arms=[SweepArmRecord(label="a", model=RUN_MODELS[0])],
        max_concurrent_arms=1,
    )
    off_loop(fixture.host.eval_host.storage.save_sweep, sweep)

    status = await job_poll(fixture.host, f"sweep:{sweep.id}", TOYHOST_SCOPE)
    assert (status.state, status.status) == ("lost", "cancelled")
    assert status.detail is not None and "no further arm will be launched" in status.detail


# =============================================================================
# Scope
# =============================================================================


async def test_another_scopes_job_reads_absent_and_its_cancel_is_refused() -> None:
    gate = asyncio.Event()
    fixture = off_loop(ops_fixture, gate=gate)
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await asyncio.sleep(0)

    with pytest.raises(NotFoundError):
        await job_poll(fixture.host, job.job_id, "some-other-scope")
    with pytest.raises(NotFoundError):
        await job_cancel(fixture.host, job.job_id, "some-other-scope")

    assert (await job_poll(fixture.host, job.job_id, TOYHOST_SCOPE)).state == "running"
    assert off_loop(_attempt, fixture.host, job.job_id).cancel_requested_at is None, "the refused cancel asked nothing"
    gate.set()
    assert (await settled(fixture.host, job.job_id)).state == "completed"


# =============================================================================
# The reclaim
# =============================================================================


def test_the_reclaim_writes_lapsed_runs_lost_and_leaves_any_process_live_run_alone() -> None:
    fixture = ops_fixture(store=ClockedStore())
    storage = fixture.host.eval_host.storage
    stored = storage.load_eval_run(fixture.campaign.run_ids[0], TOYHOST_SCOPE)
    assert stored is not None
    live, lapsed, unleased = (
        stored.model_copy(update={"id": f"run-{name}"}) for name in ("live", "lapsed", "unleased")
    )
    running: dict[str, Any] = {"status": "running", "completed_at": None}
    storage.save_eval_run(
        live.model_copy(
            update={**running, "lease": JobLease(holder="another-process", expires_at="2026-10-10T12:01:00+00:00")}
        )
    )
    storage.save_eval_run(
        lapsed.model_copy(
            update={**running, "lease": JobLease(holder="another-process", expires_at="2026-10-10T11:59:00+00:00")}
        )
    )
    storage.save_eval_run(unleased.model_copy(update=running))

    report = sweep_abandoned_runs(storage, [TOYHOST_SCOPE])

    assert sorted(report.cancelled_run_ids) == sorted([lapsed.id, unleased.id])
    assert (report.scanned, report.skipped_live, report.already_settled, report.unreadable) == (3, 1, 0, 0)
    assert _run(fixture.host, live.id).status == "running"
    again = sweep_abandoned_runs(storage, [TOYHOST_SCOPE])
    assert (again.cancelled_run_ids, again.skipped_live) == ([], 1), "a second reclaim writes nothing twice"
    assert settle_job_record(storage, RUN_JOB, lapsed.id, TOYHOST_SCOPE)[1] is False


# =============================================================================
# Sweeps
# =============================================================================


def _sweeping_host(store: InMemoryDocumentStore, *, job_lease_s: float = 60.0) -> OpsHost:
    """A toy host whose extractions each take a moment, so a sweep's arms are seen running."""
    storage = EvalStorage(store)
    storage.save_template(toyhost_template())
    client = ScriptedExtractionClient(
        tuple(type(script)(**{**vars(script), "latency_s": 0.05}) for script in TOY_SCRIPTS)
    )
    launch, _client = toyhost_launch_host(storage=storage, client=client, job_lease_s=job_lease_s)
    return OpsHost(launch=launch)


def _three_arm_sweep() -> SweepArguments:
    return SweepArguments(
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        campaign_name="prompt styles",
        campaign_behavior="extraction accuracy",
        k_runs=1,
        arms=[SweepArm(model=RUN_MODELS[0], label=style, overlays={"prompt_style": style}) for style in STYLES],
        # Arm after arm: what these read is the fence checked before each further arm is launched.
        arm_schedule="sequential",
    )


async def _first_arm_running(host: OpsHost, job_id: str) -> None:
    async with asyncio.timeout(10):
        while (await job_poll(host, job_id, TOYHOST_SCOPE)).progress.get("arms", {}).get(STYLES[0]) != "running":
            await asyncio.sleep(0.005)


def test_a_job_asks_whether_it_still_holds_its_record_before_it_spends_again() -> None:
    """``require_held``: a live record passes, a lapsed one is written lost and refused, an ended one is refused."""
    fixture = ops_fixture(store=ClockedStore())
    storage = fixture.host.eval_host.storage
    sweep = EvalSweep(
        scope_id=TOYHOST_SCOPE,
        campaign_id=fixture.campaign.id,
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        arms=[SweepArmRecord(label="a", model=RUN_MODELS[0])],
        max_concurrent_arms=1,
        lease=JobLease(holder="this-process", expires_at=(T0 + timedelta(seconds=1)).isoformat()),
    )
    storage.save_sweep(sweep)

    require_held(storage, SWEEP_JOB, sweep.id, TOYHOST_SCOPE)  # live: nothing raised, nothing written
    assert storage.load_job_record(EvalSweep, sweep.id, TOYHOST_SCOPE)[0] == sweep

    storage.save_sweep(sweep.model_copy(update={"lease": JobLease(holder="x", expires_at=T0.isoformat())}))
    with pytest.raises(JobFenced, match="was lost; it starts nothing more"):
        require_held(storage, SWEEP_JOB, sweep.id, TOYHOST_SCOPE)
    lost, _ = storage.load_job_record(EvalSweep, sweep.id, TOYHOST_SCOPE)
    assert lost is not None and lost.lost is not None, "the lapse was written, as any reader writes it"
    with pytest.raises(JobFenced):
        require_held(storage, SWEEP_JOB, sweep.id, TOYHOST_SCOPE)
    require_held(storage, SWEEP_JOB, "never-stored", TOYHOST_SCOPE)


async def test_a_sweep_another_process_runs_is_cancelled_from_there() -> None:
    holder = off_loop(_sweeping_host, InMemoryDocumentStore(), job_lease_s=0.3)
    elsewhere = _other_process(holder)
    (job,) = (await sweep_launch(holder, _three_arm_sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    await _first_arm_running(elsewhere, job.job_id)

    await job_cancel(elsewhere, job.job_id, TOYHOST_SCOPE)
    ended = await settled(elsewhere, job.job_id)

    assert ended.state == "cancelled"
    arms = ended.progress["arms"]
    assert arms[STYLES[2]] == "not_launched", "the holder stopped the sweep before its last arm"
    assert set(arms.values()) <= {"completed", "cancelled", "not_launched"}


async def test_a_sweep_whose_lease_lapsed_reads_lost_and_launches_nothing_more() -> None:
    store = ClockedStore()
    holder = off_loop(_sweeping_host, store)
    elsewhere = _other_process(holder)
    (job,) = (await sweep_launch(holder, _three_arm_sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    sweep_id = job.job_id.removeprefix("sweep:")
    await _first_arm_running(elsewhere, job.job_id)

    store.at = T0 + timedelta(seconds=61)
    lost = await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)
    assert (lost.state, lost.status) == ("lost", "cancelled")

    await _until(lambda: not holder.launch.job_manager.is_task_active(sweep_id))
    after = await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)
    assert after.state == "lost"
    # Counted in the store, not off the sweep's record: the fence makes that record unwritable, so an arm launched
    # after the loss is exactly the one it could never name.
    runs = off_loop(holder.eval_host.storage.query_eval_runs, TOYHOST_SCOPE)
    assert len(runs) == 1, f"the fenced sweep launched no further arm; the store holds {len(runs)} run(s)"
    assert runs[0].overlays.get("prompt_style") == STYLES[0]
    sweep, _ = off_loop(holder.eval_host.storage.load_job_record, EvalSweep, sweep_id, TOYHOST_SCOPE)
    assert sweep is not None and sweep.outcome == "cancelled" and sweep.lost is not None


def _interleaved_sweep() -> SweepArguments:
    return _three_arm_sweep().model_copy(update={"arm_schedule": None})


async def _every_arm_running(host: OpsHost, job_id: str) -> None:
    async with asyncio.timeout(10):
        while set((await job_poll(host, job_id, TOYHOST_SCOPE)).progress.get("arms", {}).values()) != {"running"}:
            await asyncio.sleep(0.005)


async def test_an_interleaved_sweep_cancelled_part_way_stops_every_arm_it_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _held_extraction(monkeypatch)  # every arm's cells wait until the cancel stops them
    holder = off_loop(_sweeping_host, InMemoryDocumentStore())
    (job,) = (await sweep_launch(holder, _interleaved_sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    await _every_arm_running(holder, job.job_id)

    await job_cancel(holder, job.job_id, TOYHOST_SCOPE)
    ended = await settled(holder, job.job_id)

    assert ended.state == "cancelled"
    assert set(ended.progress["arms"].values()) == {"cancelled"}, "one cancel stopped every arm of the one launch"
    sweep, _ = off_loop(
        holder.eval_host.storage.load_job_record, EvalSweep, job.job_id.removeprefix("sweep:"), TOYHOST_SCOPE
    )
    assert sweep is not None and sweep.arm_schedule == "interleaved"
    campaign = off_loop(holder.eval_host.storage.load_campaign, sweep.campaign_id, TOYHOST_SCOPE)
    assert campaign is not None and len(campaign.run_ids) == 3, "every arm it started stays a member"


async def test_an_interleaved_sweep_whose_record_reads_lost_while_its_arms_prepare_starts_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ClockedStore()
    holder = off_loop(_sweeping_host, store)
    prepared: list[str] = []
    start_run = sweeps.start_run

    async def prepare_then_lapse(*args: Any, **kwargs: Any) -> Any:
        runs = await start_run(*args, **kwargs)
        prepared.extend(run.id for run in runs)
        if len(prepared) == 3:
            # The last arm is prepared, and the sweep's lease lapses before any arm starts: a long preparation.
            store.at = T0 + timedelta(seconds=61)
        return runs

    monkeypatch.setattr(sweeps, "start_run", prepare_then_lapse)
    (job,) = (await sweep_launch(holder, _interleaved_sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    sweep_id = job.job_id.removeprefix("sweep:")
    await _until(lambda: not holder.launch.job_manager.is_task_active(sweep_id))

    assert len(prepared) == 3, "every arm was prepared"
    assert off_loop(holder.eval_host.storage.query_eval_runs, TOYHOST_SCOPE) == [], (
        "the sweep found its record lost before starting them, so it started none it could no longer name"
    )
    sweep, _ = off_loop(holder.eval_host.storage.load_job_record, EvalSweep, sweep_id, TOYHOST_SCOPE)
    assert sweep is not None and sweep.lost is not None and all(arm.run_id is None for arm in sweep.arms)


async def test_an_interleaved_sweep_whose_lease_lapsed_while_its_arms_ran_reads_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _held_extraction(monkeypatch)  # its arms run on until the holder lets go
    store = ClockedStore()
    # A short lease, so the holder's own renewal finds the loss: its arms are all launched, and it awaits them.
    holder = off_loop(_sweeping_host, store, job_lease_s=0.9)
    elsewhere = _other_process(holder)
    (job,) = (await sweep_launch(holder, _interleaved_sweep(), TOYHOST_SCOPE, created_by="agent:test")).jobs
    sweep_id = job.job_id.removeprefix("sweep:")
    await _every_arm_running(elsewhere, job.job_id)

    store.at = T0 + timedelta(seconds=61)
    lost = await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)
    assert (lost.state, lost.status) == ("lost", "cancelled")

    await _until(lambda: not holder.launch.job_manager.is_task_active(sweep_id))
    sweep, _ = off_loop(holder.eval_host.storage.load_job_record, EvalSweep, sweep_id, TOYHOST_SCOPE)
    assert sweep is not None and sweep.outcome == "cancelled" and sweep.lost is not None, (
        "the holder wrote no ending over the loss"
    )
    assert len(off_loop(holder.eval_host.storage.query_eval_runs, TOYHOST_SCOPE)) == 3


# =============================================================================
# One generation of a campaign at a time, across processes
# =============================================================================


class RacingStore(InMemoryDocumentStore):
    """Holds the two claims of a campaign's generation slot until both have read it, so both race the write.

    Each start reads the slot twice: once before it prepares (a cheap early refusal) and once in its claim. The
    third and fourth reads are the two claims; each waits for the other, so both find the slot empty and both
    try to create it.
    """

    def __init__(self) -> None:
        """Arm the barrier the two claims meet at."""
        super().__init__()
        self.slot_reads = 0
        self._claims = threading.Barrier(2, timeout=5)

    def get_with_etag(self, doc_id: str, scope_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Read, holding a claim's read of the slot until the other claim has read it too."""
        found = super().get_with_etag(doc_id, scope_id)
        if doc_id.startswith("generation-slot:"):
            self.slot_reads += 1
            if self.slot_reads in (3, 4):
                self._claims.wait()
        return found


async def test_two_processes_racing_a_generation_start_one_wins_and_the_other_is_refused_naming_it() -> None:
    gate = asyncio.Event()
    store = RacingStore()
    first = off_loop(ops_fixture, gate=gate, store=store)
    second = _other_process(first.host)

    outcomes = await asyncio.gather(
        analysis_generate(first.host, first.campaign.id, TOYHOST_SCOPE),
        analysis_generate(second, first.campaign.id, TOYHOST_SCOPE),
        return_exceptions=True,
    )

    started = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    refused = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(started) == 1 and len(refused) == 1, outcomes
    assert store.slot_reads >= 4, "both starts reached the claim"
    (job,) = started[0].jobs
    assert isinstance(refused[0], ConflictError)
    assert f"(job {job.job_id})" in str(refused[0]), "the refusal names the live attempt"

    attempts = {
        attempt.id: attempt
        for attempt in off_loop(
            first.host.eval_host.storage.list_analysis_attempts_by_campaign, first.campaign.id, TOYHOST_SCOPE
        )
    }
    loser = next(attempt for attempt_id, attempt in attempts.items() if attempt_id != job.job_id.rsplit(":", 1)[1])
    assert loser.outcome == "failed" and loser.calls == 0
    assert loser.error is not None and loser.error.startswith("refused before any call") and job.job_id in loser.error

    gate.set()
    assert (await settled(first.host, job.job_id)).state == "completed"


async def test_a_generation_slot_is_taken_over_once_its_attempt_has_ended_or_lapsed() -> None:
    gate = asyncio.Event()
    store = ClockedStore()
    holder = off_loop(ops_fixture, gate=gate, store=store)
    elsewhere = _other_process(holder.host)
    (job,) = (await analysis_generate(holder.host, holder.campaign.id, TOYHOST_SCOPE)).jobs

    with pytest.raises(ConflictError, match=f"job {job.job_id}"):
        await analysis_generate(elsewhere, holder.campaign.id, TOYHOST_SCOPE)

    store.at = T0 + timedelta(seconds=61)  # the holder's process went away
    (taken,) = (await analysis_generate(elsewhere, holder.campaign.id, TOYHOST_SCOPE)).jobs
    assert taken.job_id != job.job_id
    assert (await job_poll(elsewhere, job.job_id, TOYHOST_SCOPE)).state == "lost"
    slot, _ = off_loop(elsewhere.eval_host.storage.load_generation_slot_with_etag, holder.campaign.id, TOYHOST_SCOPE)
    assert slot is not None and slot.attempt_id == taken.job_id.rsplit(":", 1)[1]
    gate.set()


# =============================================================================
# A holder's own writes keep its lease
# =============================================================================


class RunRenewalLosingStore(InMemoryDocumentStore):
    """Every renewal of a run loses its race to the run's own writes, as a run ticking fast over a slow store sees.

    A conditional write of a run that changes nothing but its lease (a renewal) is refused as a lost race; every other
    write lands, the run's own progress and status writes included.
    """

    def __init__(self) -> None:
        """Count the refused renewal attempts."""
        super().__init__()
        self.renewal_attempts_refused = 0

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Refuse a run's write moving only its lease; write anything else."""
        if if_match is not None and document.get("doc_type") == "eval_run":
            stored = self.get(str(document["id"]), str(document["scope_id"]))
            if stored is not None and {**stored, "lease": None} == {**document, "lease": None}:
                self.renewal_attempts_refused += 1
                raise StoreConflict("the run's own write reached the record first")
        super().upsert(document, if_match=if_match, create_only=create_only)


def _paced_extraction(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Make every extraction take ``seconds``, so a run's cells, and its progress writes, come at that pace."""
    extract = ScriptedExtractionClient.extract

    async def paced(self: ScriptedExtractionClient, request: Any) -> Any:
        await asyncio.sleep(seconds)
        return await extract(self, request)

    monkeypatch.setattr(ScriptedExtractionClient, "extract", paced)


async def _launched_run(host: OpsHost, k_runs: int) -> str:
    """Launch one arm of ``k_runs`` repeats and wait until its run reads running; the run's id."""
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[RUN_MODELS[0]], k_runs=k_runs
    )
    (job,) = (await run_launch(host, arguments, TOYHOST_SCOPE)).jobs
    async with asyncio.timeout(10):
        while not (run_ids := (await job_poll(host, job.job_id, TOYHOST_SCOPE)).run_ids):
            await asyncio.sleep(0.01)
        while (await job_poll(host, run_job_id(run_ids[0]), TOYHOST_SCOPE)).status != "running":
            await asyncio.sleep(0.01)
    return run_ids[0]


async def test_a_run_whose_renewals_lose_to_its_own_writes_for_two_leases_keeps_its_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _paced_extraction(monkeypatch, 0.6)  # thirty cells, four at a time: progress writes in bursts for about 5 s
    store = RunRenewalLosingStore()
    host = off_loop(ops_fixture, store=store, job_lease_s=1.8)
    run_id = await _launched_run(host.host, k_runs=10)

    ended = await settled(host.host, run_job_id(run_id))

    run = off_loop(_run, host.host, run_id)
    assert (ended.state, run.status, run.lost, run.faults) == ("completed", "completed", None, []), run.faults
    assert run.completed_at is not None
    lasted = (datetime.fromisoformat(run.completed_at) - datetime.fromisoformat(run.created_at)).total_seconds()
    assert lasted >= 2 * 1.8, f"the run lasted {lasted:.1f}s, not two leases: its lease was never at risk"
    assert store.renewal_attempts_refused >= 6, "the run's renewals lost their races to its own writes throughout"


async def test_a_run_whose_renewals_lose_to_its_own_writes_still_hears_a_cancel_from_another_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _paced_extraction(monkeypatch, 0.6)
    store = RunRenewalLosingStore()
    holder = off_loop(ops_fixture, store=store, job_lease_s=3.0)  # renewed every second
    run_id = await _launched_run(holder.host, k_runs=10)

    asked = time.monotonic()
    await job_cancel(_other_process(holder.host), run_job_id(run_id), TOYHOST_SCOPE, reason="stop it from over here")
    ended = await settled(holder.host, run_job_id(run_id))

    assert (ended.state, ended.status) == ("cancelled", "cancelled"), "the cancel reached a run no renewal reached"
    assert time.monotonic() - asked < 2.0, "within two renewal intervals, from the run's own next write"
    assert ended.detail is not None and "stop it from over here" in ended.detail
    assert off_loop(_run, holder.host, run_id).lost is None


def _stored_run(storage: EvalStorage, run_id: str) -> tuple[EvalRun, str | None]:
    run, etag = storage.load_job_record(EvalRun, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run, etag


async def _own_write(
    leases: JobLeases, storage: EvalStorage, run_id: str, *, tick: int = 1, status: str = "running"
) -> EvalRun:
    """The run's own write, made as the holder makes it: read, then written through the leases."""
    stored, etag = off_loop(_stored_run, storage, run_id)
    ticked = stored.model_copy(update={"progress": {"cells": tick}, "status": status})
    return await asyncio.to_thread(
        leases.write_own, stored, ticked, lambda written: storage.save_eval_run(written, if_match=etag)
    )


def _seeded_run(storage: EvalStorage, run_id: str, lease: JobLease) -> EvalRun:
    run = make_eval_run(id=run_id, scope_id=TOYHOST_SCOPE, status="running", lease=lease)
    off_loop(storage.save_job_record, run.to_dict(), if_match=None, create_only=True)
    return run


#: The lease the own-write tests hold, renewed every second: an own write extends it only once its last extension is
#: half an interval old, so each waits :data:`DUE_S` after the hold, and is made before the first renewal at 1 s.
OWN_LEASE_S = 3.0

#: How long after a hold an own write waits so that it extends the lease, inside the first renewal interval.
DUE_S = 0.6


async def test_an_own_write_never_revives_a_lease_the_stores_clock_reads_lapsed() -> None:
    store = ClockedStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=OWN_LEASE_S)
    stopped: list[str] = []
    try:
        seeded = _seeded_run(storage, "run-own", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-own", TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: stopped.append("x"))
        await asyncio.sleep(DUE_S)
        store.at = T0 + timedelta(seconds=4)  # past the expiry by the store's clock; no reader has looked

        with pytest.raises(JobFenced, match="lapsed"):
            await _own_write(leases, storage, "run-own")
        await asyncio.sleep(0.05)

        after, _ = off_loop(_stored_run, storage, "run-own")
        assert after.lease == seeded.lease, "the lapsed lease was carried as it was, never extended"
        assert after.progress != {"cells": 1}, "the write itself was not made"
        assert after.lost is not None, "the lapse was written lost, as a renewal finding it writes it"
        assert leases.fenced("run-own") and stopped == ["x"], "the holder let it go and stopped its job"
    finally:
        await leases.shut_down()


@pytest.mark.parametrize(
    ("whose", "extended"),
    [("this process's, held here", True), ("another process's", False), ("this process's, not held here", False)],
)
async def test_an_own_write_extends_only_a_lease_this_process_holds(whose: str, extended: bool) -> None:
    store = ClockedStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=OWN_LEASE_S)
    try:
        lease = (
            JobLease(holder="another-process", expires_at=(T0 + timedelta(seconds=OWN_LEASE_S)).isoformat())
            if whose == "another process's"
            else off_loop(leases.grant)
        )
        _seeded_run(storage, "run-own", lease)
        if whose != "this process's, not held here":
            leases.hold(RUN_JOB, "run-own", TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: None)
        await asyncio.sleep(DUE_S)
        store.at = T0 + timedelta(seconds=1)

        written = await _own_write(leases, storage, "run-own")

        after, _ = off_loop(_stored_run, storage, "run-own")
        assert after.progress == {"cells": 1} and after.lease == written.lease, "the write landed"
        expected = (
            JobLease(holder=leases.holder_id, expires_at=(T0 + timedelta(seconds=1 + OWN_LEASE_S)).isoformat())
            if extended
            else lease
        )
        assert after.lease == expected, f"{whose}: extended {extended}"
    finally:
        await leases.shut_down()


class ClockFailingStore(ClockedStore):
    """A clocked store whose clock, once told to, cannot be read: the store is reachable, its time is not."""

    def __init__(self) -> None:
        """Start with a readable clock."""
        super().__init__()
        self.clock_down = False
        self.clock_reads = 0

    def now(self) -> datetime:
        """The store's time, or a failure while the clock is down."""
        self.clock_reads += 1
        if self.clock_down:
            raise RuntimeError("the store's clock query failed")
        return self.at


async def test_an_own_write_whose_clock_read_fails_still_lands_with_its_lease_as_stored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = ClockFailingStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=OWN_LEASE_S)
    try:
        seeded = _seeded_run(storage, "run-own", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-own", TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: None)
        await asyncio.sleep(DUE_S)
        store.at, store.clock_down = T0 + timedelta(seconds=1), True

        await _own_write(leases, storage, "run-own")

        after, _ = off_loop(_stored_run, storage, "run-own")
        assert after.progress == {"cells": 1} and after.lease == seeded.lease, "it landed, carrying the lease as stored"
        assert any("could not be read to extend the lease" in record.getMessage() for record in caplog.records)
    finally:
        await leases.shut_down()


async def test_an_own_write_soon_after_the_lease_was_extended_reads_no_clock_and_carries_it() -> None:
    store = ClockFailingStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=OWN_LEASE_S)
    try:
        seeded = _seeded_run(storage, "run-own", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-own", TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: None)
        reads = store.clock_reads

        await _own_write(leases, storage, "run-own")  # under half an interval after the hold extended it

        after, _ = off_loop(_stored_run, storage, "run-own")
        assert after.progress == {"cells": 1} and after.lease == seeded.lease, "it landed, its lease as stored"
        assert store.clock_reads == reads, "no renewal was due, so the store's clock was not read"
    finally:
        await leases.shut_down()


async def test_a_write_that_ends_the_job_does_not_extend_its_lease_or_deliver_a_cancel() -> None:
    store = ClockedStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=OWN_LEASE_S)
    cancelled: list[str | None] = []
    try:
        seeded = _seeded_run(storage, "run-own", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-own", TOYHOST_SCOPE, cancel=cancelled.append, stop=lambda: None)
        off_loop(request_job_cancel, storage, "run-own", TOYHOST_SCOPE, reason="too late")
        await asyncio.sleep(DUE_S)
        store.at = T0 + timedelta(seconds=1)

        await _own_write(leases, storage, "run-own", status="completed")
        await asyncio.sleep(0.05)

        after, _ = off_loop(_stored_run, storage, "run-own")
        assert (after.status, after.lease) == ("completed", seeded.lease), "its ending carried the lease as stored"
        assert cancelled == [], "a cancel does not reach a job its own write just ended"
    finally:
        await leases.shut_down()


async def test_a_cancel_reaches_the_job_once_though_an_own_write_and_a_renewal_both_find_it() -> None:
    store = ClockedStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=OWN_LEASE_S)
    cancelled: list[str | None] = []
    try:
        _seeded_run(storage, "run-own", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-own", TOYHOST_SCOPE, cancel=cancelled.append, stop=lambda: None)
        off_loop(request_job_cancel, storage, "run-own", TOYHOST_SCOPE, reason="stop it")
        await asyncio.sleep(DUE_S)

        await _own_write(leases, storage, "run-own", tick=1)  # finds the cancel and delivers it
        await asyncio.sleep(1.0)  # the first renewal, at one interval, finds it too
        await _own_write(leases, storage, "run-own", tick=2)
        await asyncio.sleep(DUE_S)
        await _own_write(leases, storage, "run-own", tick=3)
        await asyncio.sleep(0.05)

        assert cancelled == ["stop it"], "handed to the job once"
    finally:
        await leases.shut_down()


class RenewalHangingStore(InMemoryDocumentStore):
    """A store where a hanging run's renewal (a conditional write moving only its lease) hangs until released, and
    everything else lands: a renewal-path hang, a dead connection only the lease pool's threads hold.

    Every run hangs by default; ``hanging`` names the runs that do when it is set.
    """

    def __init__(self, hanging: set[str] | None = None) -> None:
        """Hang the renewals of ``hanging`` (every run's when None) until released; count each run's renewals sent."""
        super().__init__()
        self.hanging = hanging
        self.release = threading.Event()
        self.renewals_sent: dict[str, int] = {}

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Hang a write moving only a hanging run's lease; write anything else."""
        run_id = str(document["id"])
        if if_match is not None and document.get("doc_type") == "eval_run":
            stored = self.get(run_id, str(document["scope_id"]))
            if stored is not None and {**stored, "lease": None} == {**document, "lease": None}:
                self.renewals_sent[run_id] = self.renewals_sent.get(run_id, 0) + 1
                if self.hanging is None or run_id in self.hanging:
                    self.release.wait()
        super().upsert(document, if_match=if_match, create_only=create_only)


def _live_lease(storage: EvalStorage, run_id: str) -> bool:
    lease = off_loop(_stored_run, storage, run_id)[0].lease
    return lease is not None and lease.expires() > datetime.now(UTC)


async def test_a_stuck_renewal_is_never_resent_its_own_writes_keep_the_lease_and_renewal_resumes_once_it_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = RenewalHangingStore()
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=1.2)  # renewed every 0.4 s; the first renewal, at 0.4 s, hangs
    stopped: list[str] = []
    tick = 0

    async def writing_until(seconds: float) -> None:
        nonlocal tick
        until = time.monotonic() + seconds
        while time.monotonic() < until and not stopped:
            tick += 1
            await _own_write(leases, storage, "run-hung", tick=tick)
            await asyncio.sleep(0.25)  # past half an interval: each write extends the lease

    try:
        _seeded_run(storage, "run-hung", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-hung", TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: stopped.append("x"))

        await writing_until(3.6)  # stuck from 0.4 s, hung by about 1.6 s, and due again every interval since

        assert stopped == [] and leases.holds("run-hung") and not leases.fenced("run-hung"), (
            "its own writes kept the lease the whole time its renewal was stuck, three leases"
        )
        assert store.renewals_sent == {"run-hung": 1}, "no second renewal was sent while the first was stuck"
        assert any("renewal stuck — kept by the job's own writes" in r.getMessage() for r in caplog.records)

        store.release.set()  # the stuck call returns; the job stops writing, so only renewals can keep its lease
        await asyncio.sleep(2.0)  # past a whole lease

        assert store.renewals_sent["run-hung"] >= 2, "renewal resumed once the stuck call returned"
        assert stopped == [] and not leases.fenced("run-hung") and _live_lease(storage, "run-hung"), (
            "the resumed renewals keep the lease"
        )
    finally:
        store.release.set()
        await leases.shut_down()


async def test_hangs_retire_the_lease_pool_so_they_never_take_renewal_capacity_from_other_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_leases, "LEASE_POOL_WORKERS", 2)  # two hangs in one pool would leave it no thread
    store = RenewalHangingStore(hanging={"run-a1", "run-a2", "run-a3"})
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=1.2)
    stopped: list[str] = []
    try:
        _seeded_run(storage, "run-b", off_loop(leases.grant))
        leases.hold(RUN_JOB, "run-b", TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: stopped.append("b"))
        for hanging in ("run-a1", "run-a2", "run-a3"):  # one after another, each hanging its first renewal
            _seeded_run(storage, hanging, off_loop(leases.grant))
            leases.hold(RUN_JOB, hanging, TOYHOST_SCOPE, cancel=lambda _reason: None, stop=lambda: None)
            async with asyncio.timeout(5):
                while not leases.fenced(hanging):  # stuck, found hung, and expired: nothing else keeps it
                    await asyncio.sleep(0.05)

        assert all(store.renewals_sent[hanging] == 1 for hanging in ("run-a1", "run-a2", "run-a3"))
        assert stopped == [] and not leases.fenced("run-b") and _live_lease(storage, "run-b"), (
            "the healthy record renewed on a fresh pool after every hang"
        )
    finally:
        store.release.set()
        await leases.shut_down()


def test_the_kernel_keeps_its_leases_by_the_numbers_the_schema_defines() -> None:
    leases = kernel.JobLeases(kernel.EvalStorage(InMemoryDocumentStore()))

    assert leases.lease_s == schema.DEFAULT_JOB_LEASE_S
    assert leases.renew_every_s == schema.DEFAULT_JOB_LEASE_S * schema.JOB_LEASE_RENEWAL_FRACTION
    assert job_leases.LEASE_POOL_WORKERS is schema.LEASE_POOL_WORKERS, "the pool's width is the schema's"
    assert not {"DEFAULT_JOB_LEASE_S", "LEASE_POOL_WORKERS"} & set(kernel.__all__), "defined once, exported from schema"
