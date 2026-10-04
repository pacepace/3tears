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
    "NotFoundError",
    "ProviderRefusedError",
    "StorageError",
    "ValidationFailedError",
]


@dataclass
class EvalServiceError(Exception):
    """Structured error from the eval service layer."""

    message: str
    status_code: int = 400
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """The message, whatever arguments the subclass was constructed with.

        ``Exception.__str__`` renders the constructor's positional arguments, which a dataclass
        exception does not route through: an error built from keywords (an admission refusal) read
        as an empty string in every log line and traceback, and one built from two positionals (a
        not-found) as their tuple.
        """
        return self.message


class NotFoundError(EvalServiceError):
    """Resource not found (status 404)."""

    def __init__(self, resource: str, resource_id: str):
        """Build a 404 with a ``"<resource> '<id>' not found"`` message."""
        super().__init__(message=f"{resource} '{resource_id}' not found", status_code=404)


class ConflictError(EvalServiceError):
    """State conflict (status 409)."""

    def __init__(self, message: str):
        """Build a 409 with the given message."""
        super().__init__(message=message, status_code=409)


class ValidationFailedError(EvalServiceError):
    """Validation failed (status 422)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """Build a 422 with the given message and optional details payload."""
        super().__init__(message=message, status_code=422, details=details or {})


class StorageError(EvalServiceError):
    """Storage operation failed (status 503)."""

    def __init__(self, message: str):
        """Build a 503 — signal that the storage round-trip failed."""
        super().__init__(message=message, status_code=503)


class ProviderRefusedError(EvalServiceError):
    """A paid provider call made on the caller's behalf raised (status 502).

    Raised for a failure whose payload the host's
    :class:`~threetears.evals.contracts.provider.ProviderFailureDescriber` withheld — a provider
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
        """Build a 502 with the described failure as its message."""
        super().__init__(message=message, status_code=502)


class AdmissionRefusedError(EvalServiceError):
    """A launch refused because it would admit more unfinished runs than the process may hold (status 429).

    Raised before the launch prepares anything, so a refused launch has read, snapshotted,
    generated and saved nothing, and started no run of its own — never part of one. The
    numbers ride on the error as well as in its message, so a surface that wants to render
    them differently does not have to parse prose.

    Attributes:
        requested: How many runs the refused launch would have admitted.
        admitted: How many runs were admitted and unfinished (pending or running) when it asked.
        limit: The ceiling it was measured against.
    """

    def __init__(self, *, requested: int, admitted: int, limit: int, limit_name: str):
        """Build a 429 naming the ceiling, the current count and the knob that raises it.

        Args:
            requested: Runs the launch would admit.
            admitted: Runs already admitted and unfinished.
            limit: The ceiling.
            limit_name: What an operator sets to change the ceiling — supplied by the host,
                because the engine does not know its host's configuration names.
        """
        self.requested = requested
        self.admitted = admitted
        self.limit = limit
        # Two different facts, and they want opposite actions: a launch larger than the limit on its
        # own will be refused by an idle process too, so "wait" is advice that never helps it.
        if requested > limit:
            message = (
                f"launch refused: it would admit {requested} run(s) at once, more than this process's limit of "
                f"{limit} ({limit_name}) allows even when idle. Nothing was prepared or started. Launch fewer arms "
                f"(or, for a battery, fewer models or universal templates), or raise {limit_name}."
            )
        else:
            message = (
                f"launch refused: it would admit {requested} run(s) while {admitted} are already admitted and "
                f"unfinished (pending or running), past this process's limit of {limit} ({limit_name}). Nothing "
                f"was prepared or started. Wait for runs to finish, cancel some, launch fewer arms, or raise "
                f"{limit_name}."
            )
        super().__init__(
            message=message,
            status_code=429,
            details={"requested": requested, "admitted": admitted, "limit": limit, "limit_name": limit_name},
        )
