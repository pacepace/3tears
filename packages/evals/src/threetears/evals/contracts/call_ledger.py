"""The call ledger: every action a candidate took that succeeded, in the order it took them.

A goal check reads two things about a finished cell: the world the candidate left behind, and what
it did to get there. They are different facts with different owners. The world is the host's —
each dimension read back through its declared ``read`` handle and named as the registry names it.
The calls are the kind's — only the kind sees what its candidate asked its tools to do, and what
those tools answered. Holding both in one container let a check read a call record as though it
were world state, and let a world read pass on a seed nobody changed; keeping them apart is what
lets the goal language say which it is reading (``state.<dimension>`` against ``calls(...)``).

**Any kind fills one.** A kind records each action its candidate took that succeeded, from what the
tool actually returned — never what a simulation reported — and hands the ledger back on
:attr:`~threetears.evals.contracts.candidate_kind.CandidateOutput.call_ledger`. The runner stores it
beside the cell's output, which is what lets a stored result be re-graded later under today's rule
without re-running it (:mod:`threetears.evals.run.recheck`).

**A refused call is not recorded.** A call the tool refused changed nothing, and a check that
counted it would grade an attempt as an effect.

Pure data: no I/O, no clients, JSON-serializable by construction.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import Field

from threetears.evals.contracts.base import EvalDocumentModel


class RecordedCall(EvalDocumentModel):
    """One call a candidate made that succeeded — or, in a control end state, one it is stated to have made."""

    tool: str = Field(min_length=1, description="The tool the call went to, as the host's tools name it.")
    action: str = Field(min_length=1, description="The action called.")
    params: dict[str, Any] = Field(default_factory=dict, description="The parameters the call passed.")


class CallLedger(EvalDocumentModel):
    """The calls one cell's candidate made that succeeded, across every tool, in recorded order.

    One ledger across tools rather than one per tool, because the goal language's ordering
    predicates (``called_before``, ``called_after``, ``last_call_was``) ask about order ACROSS tools,
    which per-tool lists cannot answer.
    """

    calls: list[RecordedCall] = Field(
        default_factory=list, description="Every recorded call, in the order the candidate made them."
    )

    def record(self, tool: str, action: str, params: dict[str, Any] | None = None) -> None:
        """Append one call that succeeded.

        The parameters are deep-copied, so a caller that goes on mutating the dict it passed cannot
        rewrite what the candidate is recorded as having asked for.

        Args:
            tool: The tool the call went to.
            action: The action called.
            params: The parameters it passed; ``None`` records a call that passed none.
        """
        self.calls.append(RecordedCall(tool=tool, action=action, params=copy.deepcopy(params or {})))


__all__ = ["CallLedger", "RecordedCall"]
