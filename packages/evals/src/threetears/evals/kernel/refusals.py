"""What kind of refusal an engine error is: the one mapping a refused call and a refused job are both read through.

A refused call's :class:`~threetears.evals.actions.Refusal` and a refused job's record
(:class:`~threetears.evals.schema.JobRefusal`) carry the same two facts, the error's kind and its status, so both are
derived here, from the error's class, and a host branches on one set of kinds whichever way the refusal reached it.
"""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.kernel.errors import (
    AdmissionRefusedError,
    ConflictError,
    EvalServiceError,
    NotFoundError,
    ProviderRefusedError,
    StorageError,
    ValidationFailedError,
)
from threetears.evals.schema.refusal import JobRefusal, RefusalKind


#: The kind each public error class refuses as. ``tests/test_action_refusals.py`` holds every
#: :class:`EvalServiceError` class :mod:`threetears.evals.kernel.errors` exports to a kind of its own, together covering
#: :data:`~threetears.evals.schema.REFUSAL_KINDS`.
_KIND_OF_ERROR: Mapping[type[EvalServiceError], RefusalKind] = {
    EvalServiceError: "refused",
    NotFoundError: "not_found",
    ConflictError: "conflict",
    ValidationFailedError: "validation_failed",
    StorageError: "storage_failed",
    ProviderRefusedError: "provider_refused",
    AdmissionRefusedError: "admission_refused",
}


def refusal_kind(error: EvalServiceError) -> RefusalKind:
    """The kind an engine error refuses as: its nearest public class's.

    Args:
        error: The error.

    Returns:
        The kind; a host's own subclass of a public error takes that error's kind.
    """
    return next(_KIND_OF_ERROR[base] for base in type(error).__mro__ if base in _KIND_OF_ERROR)


def job_refusal(error: EvalServiceError) -> JobRefusal:
    """The refusal a job's record keeps when ``error`` ended its work: its kind and its own status.

    Args:
        error: The engine error that refused the work.

    Returns:
        The refusal, as a poll of the job serves it.
    """
    return JobRefusal(kind=refusal_kind(error), status=error.status_code)


__all__ = ["job_refusal", "refusal_kind"]
