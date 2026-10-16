"""Structured error types raised across the eval layer.

These live in a leaf module — importing nothing from ``threetears.evals`` — so any
eval module can raise them without depending on the service. They were first defined
beside the eval service, which worked while the service was the
only thing that raised them; the moment a sibling module does, that arrangement is
a cycle, because the service has to import the sibling to delegate to it.

A host's service may re-export every name here, as a stable import path for its own
callers. Import from there or from
here — they are the same class objects, so ``except NotFoundError`` catches
identically either way.

``status_code`` is carried on the error rather than mapped by each adapter, so the
REST routes and the MCP tool translate a failure the same way without either one
owning the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AdmissionRefusedError",
    "ConflictError",
    "EvalServiceError",
    "JobFenced",
    "NotFoundError",
    "ProviderRefusedError",
    "RefusedInput",
    "StorageError",
    "ValidationFailedError",
]


@dataclass
class EvalServiceError(Exception):
    """Structured error from the eval service layer.

    **A refusal names what its caller can do next** (:attr:`remedy`), apart from what went wrong
    (:attr:`message`), so a surface states both and a reader is never left to retry the same call. Every refusal
    carries one: each class's constructor requires it, except :class:`StorageError` and
    :class:`ProviderRefusedError`, whose remedy is their class's own, and :class:`AdmissionRefusedError`, which
    derives it from the numbers it was refused on. ``tests/test_refusal_remedies.py`` holds every refusal the engine
    raises to that, and every action a remedy names to the engine's catalogue.
    """

    message: str
    status_code: int = 400
    details: dict[str, Any] = field(default_factory=dict)
    remedy: str = field(kw_only=True)

    def __post_init__(self) -> None:
        """Refuse a blank remedy — a refusal that names nothing to do is the defect this field exists against.

        :raises ValueError: ``remedy`` is blank.
        """
        if not self.remedy.strip():
            raise ValueError(f"a refusal names its remedy; this one ({self.message!r}) names a blank one")

    def __str__(self) -> str:
        """The message, then the remedy, whatever arguments the subclass was constructed with.

        ``Exception.__str__`` renders the constructor's positional arguments, which a dataclass
        exception does not route through: an error built from keywords (an admission refusal) read
        as an empty string in every log line and traceback, and one built from two positionals (a
        not-found) as their tuple.
        """
        return f"{self.message}; {self.remedy}"


class NotFoundError(EvalServiceError):
    """Resource not found (status 404).

    :ivar resource: What kind of record was asked for (``template``, ``run``).
    :ivar resource_id: The id it was asked by.
    """

    def __init__(self, resource: str, resource_id: str, *, remedy: str):
        """Build a 404 with a ``"<resource> '<id>' not found"`` message.

        :param resource: What kind of record was asked for.
        :ptype resource: str
        :param resource_id: The id it was asked by.
        :ptype resource_id: str
        :param remedy: How the caller finds the id of one that exists — required, because a bare "not found" leaves
            its reader nothing to do but ask again (:mod:`threetears.evals.kernel.remedies` holds the engine's).
        :ptype remedy: str
        """
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(message=f"{resource} '{resource_id}' not found", status_code=404, remedy=remedy)


class ConflictError(EvalServiceError):
    """State conflict (status 409)."""

    def __init__(self, message: str, *, remedy: str):
        """Build a 409 with the given message.

        :param message: What the conflict is.
        :ptype message: str
        :param remedy: What the caller does about it — required: a conflict is a state the caller can always do
            something about, if only read the record again.
        :ptype remedy: str
        """
        super().__init__(message=message, status_code=409, remedy=remedy)


class ValidationFailedError(EvalServiceError):
    """Validation failed (status 422)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None, *, remedy: str):
        """Build a 422.

        :param message: What was wrong with the request.
        :ptype message: str
        :param details: An optional structured payload.
        :ptype details: dict[str, Any] | None
        :param remedy: What the caller sends instead, or does first — required, as on every refusal.
        :ptype remedy: str
        """
        super().__init__(message=message, status_code=422, details=details or {}, remedy=remedy)


#: What a caller does about a store that did not answer: the same for every storage refusal.
STORAGE_REMEDY = "the store did not complete the call; retry once it answers"


class StorageError(EvalServiceError):
    """Storage operation failed (status 503). Its remedy is the class's own (:data:`STORAGE_REMEDY`)."""

    def __init__(self, message: str):
        """Build a 503 — signal that the storage round-trip failed."""
        super().__init__(message=message, status_code=503, remedy=STORAGE_REMEDY)


#: What a caller does about a provider that refused a call: the same for every provider refusal.
PROVIDER_REMEDY = "the provider refused the call; retry later, or with another model"


class ProviderRefusedError(EvalServiceError):
    """A paid provider call made on the caller's behalf raised (status 502).

    Raised for a failure whose payload the host's
    :class:`~threetears.evals.schema.completion.ProviderFailureDescriber` withheld — a provider
    status error in its chain, or no describer able to say otherwise, since that fallback withholds all;
    a defect or a transport failure propagates as itself, so this status and code never point
    an operator at a provider for a fault that is not the provider's. The message is that
    description and never the exception's own text, because a provider status error
    stringifies as its response envelope, account id included; the error is raised
    ``from None`` for the same reason, since a chained cause would print that envelope in
    any traceback of this one.

    Its MCP code is ``PROVIDER_REFUSED`` (the class name), the code the eval tool's
    provider-error translator already returns for the same failure on every other
    long-running action, so raising it here moves no action onto a code of its own.
    """

    def __init__(self, message: str):
        """Build a 502 with the described failure as its message, and the class's remedy (:data:`PROVIDER_REMEDY`)."""
        super().__init__(message=message, status_code=502, remedy=PROVIDER_REMEDY)


class AdmissionRefusedError(EvalServiceError):
    """A launch refused because it would admit more unfinished runs than the process may hold (status 429).

    Raised before the launch prepares anything, so a refused launch has read, snapshotted,
    generated and saved nothing, and started no run of its own — never part of one. The
    numbers ride on the error as well as in its message, so a surface that wants to render
    them differently does not have to parse prose.

    :ivar requested: How many runs the refused launch would have admitted.
    :ivar admitted: How many runs were admitted and unfinished (pending or running) when it asked.
    :ivar limit: The ceiling it was measured against.
    """

    def __init__(self, *, requested: int, admitted: int, limit: int, limit_name: str):
        """Build a 429 naming the ceiling, the current count and the knob that raises it.

        :param requested: Runs the launch would admit.
        :ptype requested: int
        :param admitted: Runs already admitted and unfinished.
        :ptype admitted: int
        :param limit: The ceiling.
        :ptype limit: int
        :param limit_name: What an operator sets to change the ceiling — supplied by the host,
            because the engine does not know its host's configuration names.
        :ptype limit_name: str
        """
        self.requested = requested
        self.admitted = admitted
        self.limit = limit
        # Two different facts, and they want opposite actions: a launch larger than the limit on its
        # own will be refused by an idle process too, so "wait" is advice that never helps it.
        if requested > limit:
            message = (
                f"launch refused: it would admit {requested} run(s) at once, more than this process's limit of "
                f"{limit} ({limit_name}) allows even when idle. Nothing was prepared or started"
            )
            remedy = (
                f"launch fewer arms (or, for a battery, fewer models or universal templates), or raise {limit_name}"
            )
        else:
            message = (
                f"launch refused: it would admit {requested} run(s) while {admitted} are already admitted and "
                f"unfinished (pending or running), past this process's limit of {limit} ({limit_name}). Nothing "
                "was prepared or started"
            )
            remedy = f"wait for runs to finish, cancel some, launch fewer arms, or raise {limit_name}"
        super().__init__(
            message=message,
            status_code=429,
            details={"requested": requested, "admitted": admitted, "limit": limit, "limit_name": limit_name},
            remedy=remedy,
        )


class RefusedInput(ValueError):
    """A lower layer's refusal of what a caller asked for, held as its fault and its remedy apart.

    Raised below the service layer — a lens refusing an axis, the declaration gate refusing a bar, a launch refusing
    a raised ceiling — where an :class:`EvalServiceError` would be the wrong layer's type, and translated by the
    surface that called it into a :class:`ValidationFailedError` carrying the same two halves, so the remedy a lower
    layer knows reaches the caller as a remedy rather than as prose inside a message. A :class:`ValueError`, so a
    caller that needs only the words reads ``str()``: the fault, then the remedy.

    :ivar fault: What was wrong with what the caller asked.
    :ivar remedy: What the caller asks instead, or does first.
    """

    def __init__(self, fault: str, *, remedy: str) -> None:
        """Hold the fault and the remedy apart.

        :param fault: What was wrong.
        :ptype fault: str
        :param remedy: What the caller does instead.
        :ptype remedy: str
        :raises ValueError: ``remedy`` is blank.
        """
        if not remedy.strip():
            raise ValueError(f"a refusal names its remedy; this one ({fault!r}) names a blank one")
        super().__init__(f"{fault}; {remedy}")
        self.fault = fault
        self.remedy = remedy


class JobFenced(Exception):
    """A job's own write found its record ended by someone else — written lost — so the job writes nothing more.

    Not an :class:`EvalServiceError`: no caller is told it. It is how a job's work, holding a lease that another
    process found lapsed (or that this process released at shutdown), learns that its record now says it was
    lost, and stops rather than overwrite that with an ending of its own — the record's answer is the one every
    process has already read. Raised by the job's guarded writes
    (:func:`~threetears.evals.kernel.job_leases.write_held_record`) and caught at the job's own boundary.
    """
