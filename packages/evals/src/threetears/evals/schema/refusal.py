"""Why a call or a job was refused, as data: the kind of engine error it was, and the status that error carries.

A refused call reaches a transport's client as a :class:`~threetears.evals.actions.Refusal` beside its text, so a host
branches on the kind and never parses the words. A job's work can be refused too — after its call returned the job's
id, by a provider refusing the account it calls on, a store failing, a budget refusing a call — and that refusal ends
the job ``failed``. Its record keeps the same two facts (:class:`JobRefusal`) beside the words it ended with, so a
poll of the job serves them as data the way the call would have: a host retrying a job its provider throttled reads
``provider_refused`` on the job, the same kind it reads on a refused call.

The kinds are closed (:data:`RefusalKind`), one per public engine error class, and live here, below every layer that
reads them: the action catalogue's refusal, the job records the kernel and the runner write, and the poll that serves
them.
"""

from __future__ import annotations

from typing import Annotated, Literal, get_args

from pydantic import ConfigDict, Field

from threetears.evals.schema.base import CoreShapeModel, EvalBaseModel

#: What refused a call or ended a job: the public engine error it was, by kind. One kind per
#: :class:`~threetears.evals.kernel.EvalServiceError` class (``JobFenced``, which no caller is told, is none) —
#: ``refused`` for :class:`~threetears.evals.kernel.EvalServiceError` itself, raised with a status of its own — so the
#: set is closed: an error class a host derives maps to the kind of its nearest public ancestor, and adding a kind
#: means adding a public error class.
RefusalKind = Literal[
    "refused",
    "not_found",
    "conflict",
    "validation_failed",
    "storage_failed",
    "provider_refused",
    "admission_refused",
]

#: Every refusal kind.
REFUSAL_KINDS: tuple[RefusalKind, ...] = get_args(RefusalKind)


class JobRefusal(EvalBaseModel, CoreShapeModel):
    """The engine refusal a job ended on, as data: its kind, and the status that error carries.

    Set on a job's record exactly when the job ended ``failed`` because an engine error refused its work, and served
    by a poll of the job (``JobStatus.refusal``). The words it ended with stay in the record's ``detail`` (an analysis
    attempt's ``error``), for a person; this is for a host's code.
    """

    model_config = ConfigDict(frozen=True)

    kind: RefusalKind = Field(description="Which public engine error refused the job's work.")
    status: int = Field(ge=100, le=599, description="The HTTP-style status that error carries.")


#: A job record's ``refusal`` field (:class:`JobRefusal`): declared on each record that can end on a refusal with
#: ``= None``, so a record that ended any other way reads None.
RefusalField = Annotated[
    JobRefusal | None,
    Field(
        description=(
            "The engine refusal the job ended on, as data — its kind and status, as a refused call's are — when it "
            "ended `failed` because an engine error refused its work; None while it runs, and when it ended any other "
            "way."
        )
    ),
]


__all__ = ["REFUSAL_KINDS", "JobRefusal", "RefusalField", "RefusalKind"]
