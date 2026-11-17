"""Reading what one role spent over a run in a test, its line checked present first."""

from __future__ import annotations

from threetears.evals.analysis.summary import EvalSummary, RoleSpend


def spent(summary: EvalSummary, role: str) -> RoleSpend:
    """``role``'s spend line on ``summary``; a summary with no line for it fails here, naming the role.

    :param summary: The run's or the arm's summary.
    :ptype summary: EvalSummary
    :param role: The usage role (``candidate``, ``judge``, ...).
    :ptype role: str
    :return: Its spend.
    :rtype: RoleSpend
    """
    line = summary.role_spend(role)
    assert line is not None, f"no {role} spend on the summary"
    return line
