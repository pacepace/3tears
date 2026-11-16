"""The parameters more than one action module declares, each declared once.

A parameter name means one thing across a mounted tool (:meth:`~threetears.evals.actions.ActionCatalogue.mount`
refuses two meanings), so a parameter two modules' actions share is declared here, as an annotated type, and every
action takes it from here: the engine's (:mod:`threetears.evals.actions.engine`) and the analyst's
(:mod:`threetears.evals.actions.analyst`). A module of its own because the analyst's actions are added to the
engine's catalogue by :func:`~threetears.evals.actions.eval_catalogue`, so neither module can import the other's
parameters without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BeforeValidator, Field


@dataclass(frozen=True, slots=True)
class NullIsOmitted:
    """An optional parameter's declaration, in its ``Annotated`` metadata, that ``null`` sent for it is the parameter
    not given: the call reads exactly as one that left it out (its default, the fields it set, its next page's call).

    Declared parameter by parameter, never assumed: without it ``null`` is a value, which the action receives as sent
    where the parameter admits it (``campaign_update`` tells ``control_from_run_id: null``, which clears the control,
    from leaving it out) and refuses as ``validation_failed`` where it does not. Only a parameter with a default may
    declare it (:class:`~threetears.evals.actions.Action` refuses a required one).
    """


#: The most rows one page of a listing may ask for: a page is read into an agent's context.
_PAGE_MAX = 200

CampaignId = Annotated[str, Field(min_length=1, description="A campaign's id, as campaigns_list names it.")]
AnalysisId = Annotated[str, Field(min_length=1, description="A stored analysis's id, as analyses_list names it.")]
Offset = Annotated[int, Field(ge=0, description="Skip this many rows; the previous page's next_offset.")]
Limit = Annotated[int, Field(ge=1, le=_PAGE_MAX, description=f"The most rows to return, up to {_PAGE_MAX}.")]

#: A page's size as :data:`Limit` declares it, which a caller may leave out to have the page sized to one result
#: (:func:`~threetears.evals.actions.render.page_sizing`).
SizedLimit = Annotated[
    int | None, Field(ge=1, le=_PAGE_MAX, description=f"The most rows to return, up to {_PAGE_MAX}.")
]

SnapshotId = Annotated[
    str,
    Field(
        min_length=1,
        description="The snapshot every figure is read from, by the id its freeze returned.",
    ),
]
#: The forms of an arm a result prints, any of which a call may name an arm by: its figure key (``{arm:<id>}``), the
#: cited form (``arm:<id>``), or its bare id.
_ARM_PREFIX = "arm:"


def written_arm(written: Any) -> Any:
    """An arm as a call names it, read to its id by the one rule every action's arm parameters share: an arm written
    the way any result prints it (``<id>``, ``arm:<id>``, or the figure key ``{arm:<id>}``) is the arm of that id.
    Anything else is passed on as given, for the parameter's own type to accept or refuse.

    An arm id that itself begins with ``arm:`` would lose that prefix: no snapshot's arm is named so, since every
    result cites an arm as ``arm:<id>``.

    :param written: The value the call gave.
    :ptype written: Any
    :return: The arm's id, where ``written`` is a string; else ``written`` unchanged.
    :rtype: Any
    """
    if not isinstance(written, str):
        return written
    text = written.strip()
    if text.startswith("{" + _ARM_PREFIX) and text.endswith("}"):
        text = text[1:-1].strip()
    return text.removeprefix(_ARM_PREFIX).strip() if text.startswith(_ARM_PREFIX) else written


#: How every action's arm parameter describes the forms it takes.
ARM_FORMS = "by its id in the snapshot, or as results print it (arm:<id>, or the key {arm:<id>}): all name the same arm"

#: One arm, as any action takes it: read to its id by :func:`written_arm`.
ArmId = Annotated[str, BeforeValidator(written_arm), Field(min_length=1, description=f"An arm, {ARM_FORMS}.")]

#: An arm a call may leave out: read to its id by :func:`written_arm` where one is given, and ``null`` declared the
#: arm not given (:class:`NullIsOmitted`), which reads every arm. The arm's own constraints sit on the string, inside
#: the optional, so they are never applied to ``null``: on the optional as a whole, pydantic measured ``null``'s
#: length and raised TypeError where a refusal was due (#1249).
ArmFilter = Annotated[
    Annotated[str, BeforeValidator(written_arm), Field(min_length=1)] | None,
    Field(description=f"Read only this arm, {ARM_FORMS}; omitted reads every arm."),
    NullIsOmitted(),
]
AspectFilter = Annotated[
    str | None,
    Field(min_length=1, description="Read only this aspect, by its id in the snapshot; omitted reads every aspect."),
]
MeasureKey = Annotated[
    str, Field(min_length=1, description="A measure or judged dimension, by the key campaign_overview lists.")
]

__all__ = [
    "ARM_FORMS",
    "AnalysisId",
    "ArmFilter",
    "ArmId",
    "AspectFilter",
    "CampaignId",
    "Limit",
    "MeasureKey",
    "NullIsOmitted",
    "Offset",
    "SizedLimit",
    "SnapshotId",
    "written_arm",
]
