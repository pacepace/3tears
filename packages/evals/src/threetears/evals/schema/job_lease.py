"""What a job's record says about who is running it: a lease, a cancel request, and how it was lost.

A job's work runs in one process, and its record — a run, a sweep, an analysis attempt — lives in a store
several processes share. Which process (if any) is still running the work is therefore a fact the record
must carry, or every process answers it from its own memory and two of them answer differently: a poll in
the process that did not start a job would read a live job as lost, and a cancel there would be refused.

So the record carries a **lease** (:class:`JobLease`): the holding process's id and an expiry, renewed by the
holder with a conditional write every third of its duration. A record that has not ended by its own account
and whose lease has expired — or that carries none — is **lost**, and every process reads it so. The first
reader to see that writes it down (:class:`JobLoss`, with the record's own ending beside it), so the answer
is durable and the same everywhere rather than recomputed.

A **cancel** is a field on the record too (``cancel_requested_at`` and ``cancel_request_reason``), written by whichever
process was asked and read by the holder at its next renewal, so a cancel does not need the job to be
live in the process that hears it.

A record also keeps its **faults** (:data:`FaultsField`): what went wrong at the lease's boundary while
the job had not yet ended — chiefly a holder whose renewals failed for a whole lease, which stops the job. A
record's own ending (a run's status and reason, a sweep's ``detail``, an attempt's ``error``) is one value, and the
loss a reader writes later replaces it; a fault written there would be overwritten by the very loss it explains.
So faults are a list beside the ending, which no ending writes over, the same on every job record.

The fields are declared on each job record's model through the aliases here (:data:`LeaseField`,
:data:`CancelRequestedAtField`, :data:`CancelRequestReasonField`, :data:`LostField`, :data:`FaultsField`) rather than
a base class, so each record keeps its own field order and one more record type takes the same five lines;
:class:`LeasedRecord` is the surface the lease machinery reads off any of them. The four single values are declared
with ``= None`` (``lease: LeaseField = None``), and the list with ``= Field(default_factory=list)``: every one is
optional, and a record none of them has been written on reads it as None, or as no faults.

Times are ISO-8601 in UTC, stamped in the store's time when the store keeps a clock
(:class:`~threetears.evals.schema.store_port.StoreClock`) and in the writer's own otherwise.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Final, Protocol

from pydantic import Field, StringConstraints

from threetears.evals.schema.base import CoreShapeModel, EvalBaseModel


class JobLease(EvalBaseModel, CoreShapeModel):
    """Which process holds a job, and until when: renewed while the work runs, and lost once it lapses."""

    holder: str = Field(
        min_length=1,
        description=(
            "The id of the process holding the job (a uuid7 its job manager minted once), for a reader to say "
            "whose lease lapsed; only the holder renews it."
        ),
    )
    expires_at: str = Field(
        min_length=1,
        description=(
            "When the lease lapses unless renewed (ISO-8601, UTC), in the store's time when the store keeps one. "
            "Past it, a record that has not ended reads lost."
        ),
    )

    def expires(self) -> datetime:
        """The expiry as an aware datetime.

        Returns:
            :attr:`expires_at`, parsed.
        """
        return datetime.fromisoformat(self.expires_at)


class JobLoss(EvalBaseModel, CoreShapeModel):
    """How a job was found lost: when, and why — written once, by whoever found it."""

    at: str = Field(min_length=1, description="When the loss was written (ISO-8601, UTC).")
    cause: str = Field(min_length=1, description="Why the job is lost, in words a poll reports.")


#: A job record's lease: who is running its work and until when; None once released, and on a record no process
#: has leased — a run a launch prepared and never started (which reads lost while it has not ended — nothing can
#: be renewing it).
LeaseField = Annotated[
    JobLease | None,
    Field(
        description=(
            "The lease of the process running this job's work; None on a record no process has leased (a run "
            "a launch prepared and never started), or released at a shutdown. A record that has not ended reads lost once this lapses or is None."
        ),
    ),
]

#: When a cancel was asked of a job, by any process; the holder reads it at its next renewal.
CancelRequestedAtField = Annotated[
    str | None,
    Field(
        description=(
            "When a cancel was asked of this job (ISO-8601). Written by whichever process was asked, and read by "
            "the process running the work at its next lease renewal, which ends it cancelled. None: never asked."
        ),
    ),
]

#: Why a cancel was asked, when the asker said.
CancelRequestReasonField = Annotated[
    str | None,
    Field(description="Why the cancel was asked, when the asker said; None otherwise."),
]

#: How a job was lost, when it was: the record's own ending is written beside it.
LostField = Annotated[
    JobLoss | None,
    Field(
        description=(
            "Set when the job was found lost — its lease lapsed, or its process shut down — by whoever found it. "
            "The record's own status says how it reads beside that; None on a job that was not lost."
        ),
    ),
]


#: How many faults a job record keeps: a record whose lease machinery kept failing says so without growing without
#: bound. Once full, the newest fault takes the last place (:func:`with_fault`), so the first ones — usually the
#: cause — and the latest are both kept.
JOB_FAULTS_MAX: Final[int] = 32

#: How long one fault may be, in characters; a longer one is cut to it (:func:`with_fault`), marked as cut.
JOB_FAULT_MAX_CHARS: Final[int] = 2000

#: A job record's faults: what went wrong at its lease's boundary before it ended.
FaultsField = Annotated[
    list[Annotated[str, StringConstraints(min_length=1, max_length=JOB_FAULT_MAX_CHARS)]],
    Field(
        max_length=JOB_FAULTS_MAX,
        description=(
            "Faults at this job's lease boundary, written on the record before it ended, oldest first: each names "
            "where it happened, then the error (its type and message) — a holder whose renewals failed for a "
            "whole lease, which stopped the job, writes the last renewal's error here. Beside the record's own "
            "ending, which a loss writes over, so the loss never erases what explains it. Empty on a job with "
            "none. At most 32, each at most 2000 characters."
        ),
    ),
]


def with_fault(faults: list[str], entry: str) -> list[str]:
    """``faults`` with ``entry`` appended, kept inside :data:`FaultsField`'s caps.

    An entry longer than :data:`JOB_FAULT_MAX_CHARS` is cut to it, ending in ``…``; a list already holding
    :data:`JOB_FAULTS_MAX` entries gives its last place to the new one, keeping the earliest.

    Args:
        faults: The record's faults as stored.
        entry: The fault to add; blank is refused.

    Returns:
        The new list; ``faults`` is not changed.

    Raises:
        ValueError: ``entry`` is blank.
    """
    if not entry.strip():
        raise ValueError("a job fault names what went wrong; it cannot be blank")
    if len(entry) > JOB_FAULT_MAX_CHARS:
        entry = entry[: JOB_FAULT_MAX_CHARS - 1] + "…"
    return [*faults[: JOB_FAULTS_MAX - 1], entry]


class LeasedRecord(Protocol):
    """What the lease machinery reads off a job record: its identity and the five lease fields.

    Every job record (:class:`~threetears.evals.schema.models.EvalRun`,
    :class:`~threetears.evals.kernel.campaign.EvalSweep`,
    :class:`~threetears.evals.kernel.campaign.EvalAnalysisAttempt`) declares them with this module's aliases.
    """

    @property
    def id(self) -> str:
        """The record's id."""
        ...

    @property
    def scope_id(self) -> str:
        """The scope it lives in."""
        ...

    @property
    def lease(self) -> JobLease | None:
        """The lease of the process running it, or None."""
        ...

    @property
    def cancel_requested_at(self) -> str | None:
        """When a cancel was asked of it, or None."""
        ...

    @property
    def cancel_request_reason(self) -> str | None:
        """Why, when the asker said."""
        ...

    @property
    def lost(self) -> JobLoss | None:
        """How it was lost, or None."""
        ...

    @property
    def faults(self) -> list[str]:
        """Faults at its lease boundary before it ended, oldest first."""
        ...


__all__ = [
    "JOB_FAULTS_MAX",
    "JOB_FAULT_MAX_CHARS",
    "CancelRequestReasonField",
    "CancelRequestedAtField",
    "FaultsField",
    "JobLease",
    "JobLoss",
    "LeaseField",
    "LeasedRecord",
    "LostField",
    "with_fault",
]
