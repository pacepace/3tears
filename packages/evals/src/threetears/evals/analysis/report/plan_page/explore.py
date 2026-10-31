"""The explore part (GENERATOR §2.10), drawn static: where each reading flips, for a plan whose control moves the
margin.

A plan's explore part names the business choices a reader may move. This page draws one of them, without a script:
where a control ``binds: "margin"``, a table of each reading on an aspect that declares a margin
(:func:`~threetears.evals.analysis.report_decision.margin_sensitivity`), its arm's chip, the margin above which it
holds (its interval's worst drop) and whether it is within the margin declared, as a pill in words; then one line in
code's words on whether the call holds at every margin. The part is headed by the plan's title for it.

A control binding anything else (a floor, a percentile, a lever, the price of a point) draws nothing yet, and a plan
none of whose controls binds the margin, or an input no aspect of which declares one, has no explore part. The plan's
``recomputes`` is not printed: what it states is the controls' to settle, and a static page has none.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.analysis.quantities import POINTS, with_unit
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.plan_page.look import PageLook
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION, PageExplore
from threetears.evals.analysis.report.plan_page.words import (
    EXPLORE_HEADERS,
    EXPLORE_HOLDS,
    EXPLORE_TITLE,
    MARGIN_STATUS_WORDS,
)
from threetears.evals.analysis.report_decision import CandidateMargin, MarginSensitivity, margin_sensitivity
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import Explore, ExploreControl, ReportPlan

__all__ = ["explore_part"]

#: The criterion a control moves for this page to draw it: the asked aspect's margin.
_MARGIN = "margin"

#: Plan prose, its keys written out.
_Written = Callable[[str], str]


def _moves_the_margin(control: ExploreControl) -> bool:
    """Whether a control moves the asked aspect's margin, the one criterion this page draws a static table for."""
    return control.binds == _MARGIN


def _binds_margin(explore: Explore) -> bool:
    return any(_moves_the_margin(control) for control in explore.controls)


def _row(candidate: CandidateMargin, look: PageLook, *, by_aspect: bool) -> dict[str, TableCell]:
    arm = look.arm(candidate.arm) if candidate.arm in look.arms else None
    cells = {
        "arm": TableCell(value=arm.name, tag=arm.tag, color=arm.token) if arm else TableCell(value=candidate.arm),
        "flip": TableCell(
            value=with_unit(candidate.flip_margin, POINTS) if candidate.flip_margin is not None else None
        ),
        "declared": TableCell(
            value=MARGIN_STATUS_WORDS[candidate.within], status="good" if candidate.within else "bad"
        ),
    }
    if by_aspect:
        cells["aspect"] = TableCell(value=look.reader("aspect", candidate.aspect))
    return cells


def explore_part(plan: ReportPlan, report_input: ReportInput, look: PageLook, written: _Written) -> PageExplore | None:
    """The explore part, where the plan's control moves the margin (this module's account).

    :param plan: The stored plan.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param look: The page's look.
    :ptype look: PageLook
    :param written: Plan prose, its keys written out.
    :ptype written: Callable[[str], str]
    :return: The part; None where the plan has none this page draws.
    :rtype: PageExplore | None
    """
    explore = next(iter(plan.explore), None)
    if explore is None or not _binds_margin(explore):
        return None
    sensitivity: MarginSensitivity = margin_sensitivity(report_input)
    if not sensitivity.candidates:
        return None
    by_aspect = len({candidate.aspect for candidate in sensitivity.candidates}) > 1
    keys = ["arm", *(["aspect"] if by_aspect else []), "flip", "declared"]
    flips = TableBlock(
        section=PLANNED_SECTION,
        name="explore_flips",
        title=EXPLORE_TITLE,
        columns=[TableColumn(key=key, header=EXPLORE_HEADERS[key]) for key in keys],
        rows=[_row(candidate, look, by_aspect=by_aspect) for candidate in sensitivity.candidates],
        order="in the input's order of aspects and readings",
        total_rows=len(sensitivity.candidates),
    )
    return PageExplore(title=written(explore.title), flips=flips, holds=EXPLORE_HOLDS[sensitivity.holds])
