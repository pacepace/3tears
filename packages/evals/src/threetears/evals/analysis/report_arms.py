"""Each arm of a report input as a page draws it: its letter tag, its printed name, its role and its colour slot.

The analyst's document page, its blocks and the editor's gaps read the arms one way (:func:`fact_arms`): the control
takes the ink-gray slot (-1) and the tag ``P``; each candidate takes the next series colour and letter in the input's
order (GENERATOR §5.2, §6.1); and each is named as the page prints it
(:func:`~threetears.evals.analysis.report.figure_keys.page_arm_names`: a cut level by its lever and tag).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.report.figure_keys import page_arm_names
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_input import ReportInput

__all__ = ["FactArm", "fact_arms"]


class FactArm(BaseModel):
    """One arm as a page draws it: its id, how the page names and tags it, its role and its colour."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="The arm's id, as the report input names it.")
    tag: str = Field(description="Its letter tag: P for the control, then A, B, C in the input's order.")
    name: str = Field(description="What the page calls it: its declared short name, else its alias, else its id.")
    role: str = Field(description="control or challenger.")
    color_slot: int = Field(description="-1 for the control (ink gray); 0.. for series colours in fixed order.")


def fact_arms(report_input: ReportInput) -> tuple[FactArm, ...]:
    """Each arm as a page draws it: its tag, name, role and colour slot (-1 for the control, then series colours in
    the input's order).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The arms, in the input's order.
    :rtype: tuple[FactArm, ...]
    """
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    tags = arm_tags([arm.id for arm in report_input.arms], control)
    names = page_arm_names(report_input)
    slots = iter(range(len(report_input.arms)))
    return tuple(
        FactArm(
            id=arm.id,
            tag=tags[arm.id],
            name=names[arm.id],
            role=arm.role,
            color_slot=-1 if arm.id == control else next(slots),
        )
        for arm in report_input.arms
    )
