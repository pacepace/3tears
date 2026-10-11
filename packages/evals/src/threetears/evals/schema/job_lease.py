"""What a job's record says about who is running it: a lease, a cancel request, and how it was lost.

A job's work runs in one process, and its record — a run, a sweep, an analysis attempt — lives in a store
several processes share. Which process (if any) is still running the work is therefore a fact the record
must carry, or every process answers it from its own memory and two of them answer differently: a poll in
the process that did not start a job used to read a live job as lost, and a cancel there was refused.

So the record carries a **lease** (:class:`JobLease`): the holding process's id and an expiry, renewed by the
holder with a conditional write every third of its duration. A record that has not ended by its own account
and whose lease has expired — or that carries none — is **lost**, and every process reads it so. The first
reader to see that writes it down (:class:`JobLoss`, with the record's own ending beside it), so the answer
is durable and the same everywhere rather than recomputed.

A **cancel** is a field on the record too (``cancel_requested_at`` and ``cancel_request_reason``), written by whichever
process was asked and read by the holder at its next renewal, so a cancel no longer needs the job to be
live in the process that hears it.

The three fields are declared on each job record's model through the aliases here (:data:`LeaseField`,
:data:`CancelRequestedAtField`, :data:`CancelRequestReasonField`, :data:`LostField`) rather than a base class, so
each record keeps its own field order and one more record type takes the same four lines;
:class:`LeasedRecord` is the surface the lease machinery reads off any of them. Each is declared with ``= None``
(``lease: LeaseField = None``): every one is optional, and a record written before it reads it as None.

Times are ISO-8601 in UTC, stamped in the store's time when the store keeps a clock
(:class:`~threetears.evals.schema.store_port.StoreClock`) and in the writer's own otherwise.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Protocol

from pydantic import Field

from threetears.evals.schema.base import EvalBaseModel


class JobLease(EvalBaseModel):
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


class JobLoss(EvalBaseModel):
    """How a job was found lost: when, and why — written once, by whoever found it."""

    at: str = Field(min_length=1, description="When the loss was written (ISO-8601, UTC).")
    cause: str = Field(min_length=1, description="Why the job is lost, in words a poll reports.")


#: A job record's lease: who is running its work and until when; None once released, and on a record written
#: before leases (which reads lost while it has not ended — nothing can be renewing it).
LeaseField = Annotated[
    JobLease | None,
    Field(
        description=(
            "The lease of the process running this job's work; None on a record written before leases, or "
            "released at a shutdown. A record that has not ended reads lost once this lapses or is None."
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


class LeasedRecord(Protocol):
    """What the lease machinery reads off a job record: its identity and the four lease fields.

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


__all__ = [
    "CancelRequestReasonField",
    "CancelRequestedAtField",
    "JobLease",
    "JobLoss",
    "LeaseField",
    "LeasedRecord",
    "LostField",
]
