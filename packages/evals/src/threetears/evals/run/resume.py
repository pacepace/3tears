"""Taking a run up again: a run whose holding ended before its matrix did, continued as the same run.

A run lost to a reboot, an OOM kill, a deploy or a provider outage longer than its lease, a run an operator cancelled
to ship a fix, and a run its budget or its account stopped each end with cells of their matrix never measured. A
new run over the cells left is a different run: its case set is another, so its pass^k is another cell, and a case
the first one reached fewer than ``k`` times drops out of both. :func:`resume_run` continues the run itself:

- **Only what is missing runs.** A cell with no stored row runs, and so does a cell whose row was cut short by a
  cancel (termination ``cancelled``: the attempt a shutdown or a cancel abandoned). Every other row is what the
  run would have stored uninterrupted, so its cell is done (:func:`planned_cells`). The cells run in the run's own
  order, its done ones left out, and a re-run cell is written over its row.
- **The same measurement, or none.** The run is launched again from what it recorded, through its kind's launcher,
  and the run that launch assembles is compared with the stored one: every context component, the variant key, the
  template, the simulator and judge pins and the apparatus. Any difference refuses the take-up, naming every one
  (:func:`resume_drift`). The host's labels — a build, a version — are not the measurement: they are recorded on
  the holding that began under them (``EvalRun.lifetimes``).
- **One holding at a time.** The run is reopened under its lease's next epoch, conditionally on the record as read,
  so two take-ups of one run cannot both win, and a holding the run has passed from writes nothing more into it.
- **Its spend counts.** What its stored rows spent seeds its cost cap; a run its budget stopped is taken up only
  under a cap above that spend.

A take-up is always asked for, never made at boot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.identity import resolve_variant_identity
from threetears.evals.kernel.job_leases import RUN_JOB, JobLeases, settle_job_record
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.remedies import FIND_RUN
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.budget import EvalRunCostCap
from threetears.evals.run.launch import LaunchHost, admit_resumed_run
from threetears.evals.run.runner import StoredCells
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import (
    TERMINAL_RUN_STATUSES,
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    RunLifetime,
    ended_lifetimes,
    utc_now_iso,
)

log = get_logger(__name__)

#: The fields of a run's judge and simulator, and of its apparatus, a take-up holds equal: what scored its cells and
#: what its cells were set up with. Most also reach the context key; each is named on its own so a refusal says
#: which moved.
PINNED_FIELDS: tuple[str, ...] = (
    "judge_model",
    "judge_config_ids",
    "effective_judges",
    "judge_asked_settings",
    "judge_temperature",
    "simulator_model",
    "simulator_asked_settings",
    "apparatus_settings",
    "overlays",
    "kind_spec",
    "cassette_mode",
    "cassette_corpus_id",
    "resolved_world_seed",
    "resolved_ambient_perturbation_turns",
    "resolved_tools_allowed",
)


def planned_cells(rows: Sequence[EvalResult]) -> StoredCells:
    """What a run taken up again holds of its matrix: the rows that count as done, and the rows it runs again.

    A row whose cell was cut short by a cancel (``cancelled``) is an attempt a shutdown or a cancel abandoned, which
    an uninterrupted run never stores, so its cell runs again, written over the row. Every other row — a timeout, a
    rig fault, a failed seed included — is what the uninterrupted run would also have stored, so its cell is done.
    A cell with no row is neither, and runs.

    :param rows: Every row stored for the run.
    :ptype rows: Sequence[EvalResult]
    :return: The done rows, and the cut-short rows by cell.
    :rtype: StoredCells
    """
    return StoredCells(
        done=tuple(row for row in rows if row.termination != "cancelled"),
        replaced={(row.test_case_id, row.k_iteration): row.id for row in rows if row.termination == "cancelled"},
    )


def _shown(value: Any) -> str:
    """A value as a refusal names it: a long digest cut to its head."""
    text = repr(value)
    return text if len(text) <= 40 else f"{text[:37]}..."


def resume_drift(stored: EvalRun, fresh: EvalRun, template: EvalTemplate, profile: HostProfile) -> list[str]:
    """Every way the run a launch of ``stored`` assembles now would measure otherwise than ``stored`` did.

    Each context component by name, then the context key when no component accounts for its move (a run stamped
    under another identity version), each variant lever, the template when it was edited after the run was created,
    and each of :data:`PINNED_FIELDS`. The subject's labels and the run's own are none of these: they identify, and
    determine nothing.

    :param stored: The stored run.
    :ptype stored: EvalRun
    :param fresh: What a launch of it assembles now, stamped.
    :ptype fresh: EvalRun
    :param template: The template that launch read.
    :ptype template: EvalTemplate
    :param profile: The host, whose lever reader derives the variant of a run that recorded no lever map.
    :ptype profile: HostProfile
    :return: One phrase per difference; empty when the two measure alike.
    :rtype: list[str]
    """
    differences: list[str] = []
    was, now = stored.context_components, fresh.context_components
    if was is not None and now is not None:
        for name in type(was).model_fields:
            before, after = getattr(was, name), getattr(now, name)
            if before == after:
                continue
            if isinstance(before, Mapping) and isinstance(after, Mapping):
                moved = sorted(key for key in {*before, *after} if before.get(key) != after.get(key))
                differences.append(f"context component {name} ({', '.join(moved)})")
            else:
                differences.append(f"context component {name} ({_shown(before)} -> {_shown(after)})")
    if stored.context_key != fresh.context_key and not differences:
        differences.append(
            f"context key ({_shown(stored.context_key)} at identity version {stored.identity_version} -> "
            f"{_shown(fresh.context_key)} at {fresh.identity_version})"
        )
    levers_was = resolve_variant_identity(run=stored, profile=profile).levers
    levers_now = resolve_variant_identity(run=fresh, profile=profile).levers
    for name in sorted({*levers_was, *levers_now}):
        before_lever, after_lever = levers_was.get(name), levers_now.get(name)
        if (before_lever.content_hash if before_lever else None) != (after_lever.content_hash if after_lever else None):
            differences.append(f"variant lever {name}")
    if template.updated_at > stored.created_at:
        differences.append(
            f"template {template.id!r} (edited at {template.updated_at}, after the run was created at "
            f"{stored.created_at})"
        )
    for name in PINNED_FIELDS:
        before, after = getattr(stored, name), getattr(fresh, name)
        if before != after:
            differences.append(f"{name} ({_shown(before)} -> {_shown(after)})")
    return differences


def _why_it_ended(run: EvalRun) -> str | None:
    """Why a run's holding ended, in the words its record gives."""
    if run.lost is not None:
        return run.lost.cause
    if (reason := run.cancellation_reason or run.budget_stop_reason) is not None:
        return reason
    return run.error_details[-1] if run.status == "exhausted" and run.error_details else None


@dataclass(frozen=True)
class RunResumption:
    """A stored run taken up again, as its launch replays it: carried on the replayed arm's request
    (:attr:`~threetears.evals.run.launch.LaunchRequest.resuming`).

    :ivar run: The run as stored, settled.
    :ivar cases: Its frozen cases, in its own order: the launcher freezes these.
    :ivar stored: What it holds of its matrix (:func:`planned_cells`).
    :ivar spend: The spend of every row stored for it, done or run again, which its cost cap counts.
    :ivar replaced_spend_usd: The priced spend of the rows it runs again, which no row will carry once they are
        written over.
    :ivar holder: The process taking it up.
    :ivar epoch: The lease epoch it is taken up under.
    :ivar cell_order_seed: What its cells were ordered by when that was its launch's group, shared with its sibling
        arms; ``None`` when they were ordered by its own id.
    """

    run: EvalRun
    cases: tuple[EvalTestCase, ...]
    stored: StoredCells
    spend: tuple[float | None, ...]
    replaced_spend_usd: float
    holder: str | None
    epoch: int
    cell_order_seed: str | None

    def take_up(
        self, fresh: EvalRun, template: EvalTemplate, profile: HostProfile
    ) -> tuple[EvalRun, list[EvalTestCase]]:
        """Compare what its launch assembled with the stored run, and reopen the stored run to go on in its place.

        The reopened run reads ``running``. What it carried of its ending — its loss, its reasons, its budget stops,
        its completeness and when it ended — moves to the holding that ended (its first holding's entry is written
        now, for a run never taken up before), and a new holding opens: this process, at :attr:`epoch`, under the
        labels the host captured now and the cost cap the launch resolved, which the run records as its own.

        :param fresh: The run the launch assembled and stamped.
        :ptype fresh: EvalRun
        :param template: The template the launch read.
        :ptype template: EvalTemplate
        :param profile: The host's profile.
        :ptype profile: HostProfile
        :return: The reopened run, its lease still the stored one (the take-up's write grants the next), and its
            cases in its own order.
        :rtype: tuple[EvalRun, list[EvalTestCase]]
        :raises ValidationFailedError: The two would not measure alike; every difference is named.
        """
        stored = self.run
        if differences := resume_drift(stored, fresh, template, profile):
            raise ValidationFailedError(
                f"run '{stored.id}' is not taken up again: launched now, it would measure otherwise than it did — "
                f"{'; '.join(differences)}",
                remedy=(
                    "launch the template again to measure as things stand now, or restore what moved and take the run "
                    "up again"
                ),
            )
        first = RunLifetime(
            holder=stored.lease.holder if stored.lease is not None else None,
            epoch=self.epoch - 1,
            started_at=stored.created_at,
            labels=dict(stored.subject_snapshot.labels),
            run_labels=list(stored.labels),
            max_cost_usd=stored.max_cost_usd,
        )
        ended = ended_lifetimes(
            stored.lifetimes or [first],
            status=stored.status,
            at=stored.completed_at or utc_now_iso(),
            reason=_why_it_ended(stored),
            lost=stored.lost,
            budget_stops=stored.budget_stops,
            completeness=stored.completeness,
            # One spend per stored row: the rows the run held when its holding ended.
            cells_stored=len(self.spend),
        )
        holding = RunLifetime(
            holder=self.holder,
            epoch=self.epoch,
            started_at=utc_now_iso(),
            labels=dict(fresh.subject_snapshot.labels),
            run_labels=list(stored.labels),
            max_cost_usd=fresh.max_cost_usd,
            replaced_cells=len(self.stored.replaced),
            replaced_spend_usd=self.replaced_spend_usd,
        )
        reopened = stored.model_copy(
            update={
                "status": "running",
                "lost": None,
                "cancellation_reason": None,
                "cancel_requested_at": None,
                "cancel_request_reason": None,
                "completed_at": None,
                "budget_stop_reason": None,
                "budget_stops": None,
                "failed_because": None,
                "completeness": None,
                "max_cost_usd": fresh.max_cost_usd,
                "max_cost_usd_origin": fresh.max_cost_usd_origin,
                "lifetimes": [*ended, holding],
            }
        )
        return reopened, list(self.cases)

    def seed(self, cap: EvalRunCostCap) -> None:
        """Count what the run's stored rows spent toward its cost cap, an unpriced one as unpriced.

        :param cap: The cap the launch built for this holding.
        :ptype cap: EvalRunCostCap
        """
        for spend in self.spend:
            cap.record(spend)


@dataclass(frozen=True)
class _Read:
    """What a take-up reads before it decides: the run settled, its etag, its rows, its cases and its launch's size."""

    run: EvalRun
    etag: str | None
    rows: list[EvalResult]
    cases: tuple[EvalTestCase, ...]
    missing_cases: list[str]
    group_size: int


def _read(storage: EvalStorage, run_id: str, scope_id: str) -> _Read:
    """Settle the run as every reader does, then read it with its etag, its rows, its cases and its launch's runs.

    :raises NotFoundError: There is no such run in the scope.
    """
    settle_job_record(storage, RUN_JOB, run_id, scope_id)
    run, etag = storage.load_job_record(EvalRun, run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id, remedy=FIND_RUN)
    rows = storage.query_eval_results_by_run(run_id, scope_id)
    loaded = {case_id: storage.load_test_case(case_id, scope_id) for case_id in dict.fromkeys(run.test_case_ids)}
    group_size = (
        len(storage.launch_group_run_ids(run.launch_group_id, scope_id)) if run.launch_group_id is not None else 1
    )
    return _Read(
        run=run,
        etag=etag,
        rows=rows,
        cases=tuple(case for case in loaded.values() if case is not None),
        missing_cases=[case_id for case_id, case in loaded.items() if case is None],
        group_size=group_size,
    )


def _refuse_what_cannot_be_taken_up(read: _Read, *, cap: float | None, live_here: bool) -> None:
    """Refuse a run that cannot be taken up, with how to proceed.

    :raises ValidationFailedError: The run is live, completed, failed, stopped by its budget under a cap its spend has
        reached, or its cases are gone.
    """
    run = read.run
    if run.status not in TERMINAL_RUN_STATUSES:
        holder = f" by process {run.lease.holder} until {run.lease.expires_at}" if run.lease is not None else ""
        raise ValidationFailedError(
            f"run '{run.id}' is still {run.status}{holder}; a live run is never taken up",
            remedy="cancel it (job_cancel) or wait for it to end, then take it up again",
        )
    if live_here:
        raise ValidationFailedError(
            f"run '{run.id}' ended, and its job in this process is still unwinding",
            remedy="take it up again once its job has stopped",
        )
    if run.status == "completed":
        raise ValidationFailedError(
            f"run '{run.id}' completed; there is nothing left of it to run",
            remedy="launch its template again to measure again",
        )
    if run.status == "failed":
        raise ValidationFailedError(
            f"run '{run.id}' failed ({run.failed_because or 'see its error_details'}); taking it up again would run "
            "what failed it again",
            remedy="fix what failed it, which its error_details name, and launch its template again",
        )
    spent = sum(row.cost_usd for row in read.rows if row.cost_usd is not None)
    if run.status == "budget_stopped" and cap is not None and spent >= cap:
        raise ValidationFailedError(
            f"run '{run.id}' was stopped by its budget ({run.budget_stop_reason}), and its stored rows have spent "
            f"${spent:.4f} — at least that, since a cell in flight when it stopped may have spent more — against a "
            f"cap of ${cap:.4f}",
            remedy=f"take it up with max_cost_usd above ${spent:.4f}",
        )
    if read.missing_cases:
        raise ValidationFailedError(
            f"run '{run.id}' froze case(s) {', '.join(read.missing_cases)}, which scope {run.scope_id!r} no longer "
            "holds, so its cells cannot run",
            remedy="launch its template again over the cases the scope holds",
        )


async def resume_run(host: LaunchHost, run_id: str, scope_id: str, *, max_cost_usd: float | None = None) -> EvalRun:
    """Take a run up again: the cells it has not measured run, in its own order, and it ends as the same run.

    Resumable: a run that was lost (its holder gone, or shut down), one an operator cancelled, one its budget stopped
    — under a cap above what its stored rows spent, so a cost-capped stop needs a raised ``max_cost_usd`` — and one
    its provider account stopped (``exhausted``), which, like a budget, is resumable once the account is topped up.
    A run that completed or failed is not, nor one still live.

    In order: the run is settled as every reader settles it (written lost when its lease lapsed); refused when it
    cannot be taken up; its cells planned (:func:`planned_cells`); its launch replayed from what it recorded, through
    its kind's launcher (:func:`~threetears.evals.run.launch.admit_resumed_run`), and refused when what that launch
    assembles would measure otherwise (:func:`resume_drift`); reopened under its lease's next epoch, conditionally on
    the record as read; and started in a group of its own, ordered as it was — by its launch's group when that
    interleaved several arms, else by its own id.

    :param host: The host, whose launcher wires the run and whose job manager holds it.
    :ptype host: LaunchHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap to take it up under; ``None`` keeps the one its launch named, or inherits the
        host's. Not part of the measurement: the holding records the cap it ran under.
    :ptype max_cost_usd: float | None
    :return: The run, reopened and started.
    :rtype: EvalRun
    :raises NotFoundError: There is no such run, or its template is gone.
    :raises ValidationFailedError: It cannot be taken up (see above), it would measure otherwise now (every
        difference named), or a launch of it would be refused now.
    :raises ConflictError: Another take-up of it won, or another writer moved it, while this one prepared; nothing was
        started.
    :raises AdmissionRefusedError: The host has no room for another run.
    """
    eval_host = host.eval_host
    storage = eval_host.storage
    leases = host.job_manager.leases
    read = await run_blocking(eval_host.blocking_executor, _read, storage, run_id, scope_id)
    run = read.run
    cap = max_cost_usd if max_cost_usd is not None else run.max_cost_usd
    _refuse_what_cannot_be_taken_up(read, cap=cap, live_here=host.job_manager.is_active(run_id))
    previous = run.lease.epoch if run.lease is not None else (run.lifetimes[-1].epoch if run.lifetimes else 0)
    stored = planned_cells(read.rows)
    replaced = set(stored.replaced.values())
    resuming = RunResumption(
        run=run,
        cases=read.cases,
        stored=stored,
        spend=tuple(row.cost_usd for row in read.rows),
        replaced_spend_usd=sum(row.cost_usd for row in read.rows if row.id in replaced and row.cost_usd is not None),
        holder=leases.holder_id if leases is not None else None,
        epoch=previous + 1,
        # One arm of an interleaved launch played its cases in the order its group's id gave every arm; a group of one
        # never interleaved, and played them in its own run's order.
        cell_order_seed=run.launch_group_id if run.arm_schedule == "interleaved" and read.group_size > 1 else None,
    )

    async def reopen(runs: list[EvalRun]) -> None:
        (taken_up,) = runs
        await run_blocking(
            eval_host.blocking_executor, _write_reopened, storage, leases, taken_up, read.etag, resuming.epoch
        )

    admitted = await admit_resumed_run(
        host,
        resuming,
        max_cost_usd=max_cost_usd
        if max_cost_usd is not None
        else (run.max_cost_usd if run.max_cost_usd_origin == "chosen" else None),
        attach=reopen,
    )
    (started,) = await admitted.start()
    log.info(
        "eval.resume_run run=%s scope=%s taken up at epoch %d: %d cell(s) done, %d to run again over their rows",
        run_id,
        scope_id,
        resuming.epoch,
        len(stored.done),
        len(stored.replaced),
    )
    return started


def _write_reopened(storage: EvalStorage, leases: JobLeases | None, run: EvalRun, etag: str | None, epoch: int) -> None:
    """Write the reopened run under a lease at its next epoch, conditionally on the record the take-up read.

    :raises ConflictError: The record moved since it was read: another take-up won, or another writer wrote it.
    """
    run.lease = leases.grant(epoch) if leases is not None else None
    try:
        storage.save_eval_run(run, if_match=etag)
    except ConflictError as moved:
        raise ConflictError(
            f"run '{run.id}' changed while it was being taken up — another take-up of it won, or another writer moved "
            "it — so this one started nothing",
            remedy="read it with job_poll; take it up again if it is still not running",
        ) from moved


__all__ = [
    "PINNED_FIELDS",
    "RunResumption",
    "planned_cells",
    "resume_drift",
    "resume_run",
]
