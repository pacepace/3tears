"""The parameters more than one action module declares, each declared once.

A parameter name means one thing across a mounted tool (:meth:`~threetears.evals.actions.ActionCatalogue.mount`
refuses two meanings), so a parameter two modules' actions share is declared here, as an annotated type, and every
action takes it from here: the engine's (:mod:`threetears.evals.actions.engine`) and the analyst's
(:mod:`threetears.evals.actions.analyst`). A module of its own because the analyst's actions are added to the
engine's catalogue by :func:`~threetears.evals.actions.eval_catalogue`, so neither module can import the other's
parameters without a cycle.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

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
ArmFilter = Annotated[
    str | None,
    Field(min_length=1, description="Read only this arm, by its id in the snapshot; omitted reads every arm."),
]
AspectFilter = Annotated[
    str | None,
    Field(min_length=1, description="Read only this aspect, by its id in the snapshot; omitted reads every aspect."),
]
MeasureKey = Annotated[
    str, Field(min_length=1, description="A measure or judged dimension, by the key campaign_overview lists.")
]

__all__ = [
    "AnalysisId",
    "ArmFilter",
    "AspectFilter",
    "CampaignId",
    "Limit",
    "MeasureKey",
    "Offset",
    "SizedLimit",
    "SnapshotId",
]
