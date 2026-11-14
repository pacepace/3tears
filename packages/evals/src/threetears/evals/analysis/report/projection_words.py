"""A projection of the cases a floor needs, as a page words it in its own forms.

Reading a floor is numbers, and lives in :mod:`threetears.evals.analysis.floors`: the typed decision
(:class:`~threetears.evals.analysis.floors.FloorDecision`), the status it gives a candidate, the rates it reads and the
cases that would settle an unclear one (:class:`~threetears.evals.analysis.floors.SampleProjection`). What stays here
is the wording a page gives a projection, each count and cost printed by the page's forms: :func:`projected_cell` and
:func:`projection_note`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from threetears.evals.analysis.floors import CasesNeeded, SampleProjection
from threetears.evals.analysis.report.unit_once import interval_beside
from threetears.evals.kernel.numbers import MoneyForm

if TYPE_CHECKING:
    # A page's forms decide over figures a floor's readings hold, so the module that decides them reads those: a page
    # hands its forms in to state a projection, and the type is named for the checker alone.
    from threetears.evals.analysis.report.forms import PageForms


def projected_cell(
    projection: SampleProjection, need: CasesNeeded, form: MoneyForm | None = None, *, forms: PageForms
) -> str:
    """One side of a projection as a table cell says it, with its cost: ``107 cases, $2.7``, ``this run's 12 cases``,
    ``re-run the lost trials``, ``more than 10000 cases`` or ``none would``. Handed the form every projected cost the
    table states reads in (#993), the cost, to two figures, is written in it (``107 cases, $2.70`` beside ``$27.00``).
    The page's ``forms`` state the count and the cost
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.projected_money`).
    """
    if need.kind == "never":
        return "none would"
    if need.kind == "lost":
        return "re-run the lost trials"
    if need.kind == "enough":
        return f"this run's {projection.cases} cases"
    if need.kind == "beyond":
        return f"more than {forms.count_alone(need.cases).text} cases"
    cost = projection.cost(need)
    if cost is None:
        return f"{need.cases} cases"
    return f"{need.cases} cases, {forms.projected_money(cost, form).text}"


def projection_note(
    projections: Sequence[SampleProjection], *, run_cases: int, cost_clause: bool = False, forms: PageForms
) -> str:
    """The sentences under the projected cases: how each count was found and that it is about, what repeats do, what a
    cost floor's blank means, and, where holding needs more cases than the run used on every side, that it does.

    :param projections: Each arm's projection over its floors (:func:`floors_projection`), not each floor's: what
        showing that arm's floors hold needs.
    :ptype projections: Sequence[SampleProjection]
    :param run_cases: The cases the run used: the most any arm ran.
    :ptype run_cases: int
    :param cost_clause: Whether the table shows a cost floor, whose no-difference cell is blank.
    :ptype cost_clause: bool
    :param forms: The page's forms, which state each count.
    :ptype forms: PageForms
    :return: The sentences.
    :rtype: str
    """
    said = [
        "Each count is about: it reads the interval again over this run's cases repeated to that many, so it assumes "
        "new cases vary as these did.",
        "Repeats of a case narrow an interval far less than new cases.",
    ]
    if cost_clause:
        said.append(
            "A cost floor has no margin, so arms that do not differ are never shown cheaper; its count scales its "
            "interval by the square root of the cases."
        )
    needs = [need for projection in projections for need in (projection.if_none, projection.if_observed)]
    counted = sorted(need.cases for need in needs if need.kind == "more")
    if counted and all(need.kind in ("more", "beyond", "never") for need in needs):
        beyond = any(need.kind == "beyond" for need in needs)
        caps = sorted({need.cases for need in needs if need.kind == "beyond"})
        if not beyond:
            span = (
                f"about {counted[0]}"
                if counted[0] == counted[-1]
                else f"about {interval_beside(str(counted[0]), str(counted[-1]))}"
            )
        elif caps[0] >= counted[-1]:
            span = f"about {counted[0]} to more than {forms.count_alone(caps[-1]).text}"
        else:
            # A search stopped at a cap below a count another row reached: that row needs more than its cap, which
            # may be fewer or more than the counts, so the range is the counts' and the cap is said apart.
            low, high = counted[0], counted[-1]
            counts = f"about {low}" if low == high else f"about {interval_beside(str(low), str(high))}"
            span = f"{counts}, or more than {forms.count_alone(caps[0]).text} where a search stopped there"
        said.append(f"This run used {run_cases} cases; showing the floors hold needs {span}, more than it ran.")
    return " ".join(said)


__all__ = [
    "projected_cell",
    "projection_note",
]
