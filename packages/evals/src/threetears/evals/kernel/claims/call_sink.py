"""Where a claim criterion's steps write each call as it is made, for the caller that owns the grade.

A grade's calls are built from what its steps return, and a step that fails raises past that return: a criterion
lost to its repair budget would keep no record of the calls that spent it. So each step also writes every call that
returns here, as it returns, with the step that asked it and the failure it named to the model, and the caller, who
owns the sink, reads it however the grade ends: returned, raised or cut by the deadline.

Engine-internal: the claims steps write to it, and the judge phase reads it onto the cell's trace.
"""

from __future__ import annotations

from typing import Literal

from threetears.evals.schema.models import CLAIM_LOSS_REASON_CHARS, ClaimCall, ClaimLossCall

#: The steps a call is asked by, as :attr:`~threetears.evals.schema.models.ClaimLossCall.step` names them.
type ClaimStep = Literal["extract", "truth", "reask", "ground", "web"]


class ClaimCallSink:
    """Every call one criterion's grade made that returned, in the order made, each with its step and reason.

    :ivar calls: The calls written so far.
    """

    def __init__(self) -> None:
        """Start with no call."""
        self.calls: list[ClaimLossCall] = []

    def write(self, call: ClaimCall, *, step: ClaimStep, reason: str | None) -> None:
        """Keep one call as it returned.

        :param call: The call, as the grade records it.
        :param step: The step that asked it.
        :param reason: The failure it named to the model, for a call that asked again; None for a first ask. Kept
            to :data:`~threetears.evals.schema.models.CLAIM_LOSS_REASON_CHARS` characters.
        """
        kept = None if reason is None else reason[:CLAIM_LOSS_REASON_CHARS]
        self.calls.append(ClaimLossCall(**call.model_dump(), step=step, reason=kept))


__all__: list[str] = []
