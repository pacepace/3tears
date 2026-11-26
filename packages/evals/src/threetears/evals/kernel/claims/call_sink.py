"""Where a claim criterion's steps write each call as it is made, for the caller that owns the grade.

A grade's calls are built from what its steps return, and a step that fails raises past that return: a criterion
lost to its repair budget would keep no record of the calls that spent it. So each step also writes every call that
returns here, as it returns, with the step that asked it and the failure it named to the model; the grade writes the
failure that lost the criterion or its step; and the caller, who owns the sink, reads it however the grade ends:
returned, raised or cut by the deadline.

Engine-internal: the claims steps and the grade write to it, and the judge phase and a re-judge read it onto the
cell's trace.
"""

from __future__ import annotations

from threetears.evals.schema.models import (
    CLAIM_LOSS_REASON_CHARS,
    ClaimCall,
    ClaimGrade,
    ClaimLossCall,
    ClaimLossRecord,
    ClaimStep,
)


class ClaimCallSink:
    """Every call one criterion's grade made that returned, in the order made, each with its step and reason, and
    the failure that lost the criterion or its step.

    :ivar calls: The calls written so far.
    :ivar failure: The failure that lost it, once one did and the engine named it; None otherwise.
    """

    def __init__(self) -> None:
        """Start with no call and no failure."""
        self.calls: list[ClaimLossCall] = []
        self.failure: str | None = None

    def write(self, call: ClaimCall, *, step: ClaimStep, reason: str | None) -> None:
        """Keep one call as it returned.

        :param call: The call, as the grade records it.
        :param step: The step that asked it.
        :param reason: The failure it named to the model, for a call that asked again; None for a first ask. Kept
            to :data:`~threetears.evals.schema.models.CLAIM_LOSS_REASON_CHARS` characters.
        """
        self.calls.append(ClaimLossCall(**call.model_dump(), step=step, reason=_capped(reason)))

    def failed(self, failure: str) -> None:
        """Keep the failure that lost the criterion or its step, as the engine named it.

        :param failure: The failure. Kept to :data:`~threetears.evals.schema.models.CLAIM_LOSS_REASON_CHARS`
            characters.
        """
        self.failure = _capped(failure)

    def kept(self, *, lost: bool, grade: ClaimGrade | None) -> ClaimLossRecord | None:
        """What the trace keeps of this sink, or None when it keeps nothing: a criterion that made no call, or that
        lost neither the cell nor a step while it ran.

        :param lost: Whether the criterion lost the cell.
        :param grade: The grade the cell keeps for it; None when it keeps none.
        :return: The record, or None.
        """
        if not self.calls or not (lost or (grade is not None and lost_a_step(grade))):
            return None
        return ClaimLossRecord(failure=self.failure, calls=tuple(self.calls))


def lost_a_step(grade: ClaimGrade) -> bool:
    """Whether a grade's ground or web step failed while it ran: a ground step never asked, since the evidence
    declares no documents (``documents_undeclared``), made no call to keep.

    :param grade: The grade.
    :return: Whether it did.
    """
    return grade.web_loss is not None or grade.ground_loss not in (None, "documents_undeclared")


def _capped(text: str | None) -> str | None:
    return None if text is None else text[:CLAIM_LOSS_REASON_CHARS]


__all__: list[str] = []
