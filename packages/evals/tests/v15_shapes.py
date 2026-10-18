"""The v15 campaign declaration, as a stored v15 campaign holds it: what the tests write a stored campaign from.

0.71 removed ``CampaignDesign``, ``Question`` and ``DecisionRule``: a campaign is a campaign of aspects, and a stored
v15 campaign reads as one aspect through ``upgrade_campaign_v15``, which works on the stored dict. Nothing in the
package builds this shape any more, but a test of that upgrade, or one written against a campaign declared the old way,
still needs to write the dict a v15 campaign stored. These models write it: each field is the v15 field of the same
name, its parts are the kernel's own (``SweptAxis``, ``BarOverride``, ``ControlDeclaration``, ``DecisionClause`` ...),
and ``model_dump(mode="json")`` is the stored ``declared_design``. They check nothing the upgrade does not need: a
dict a v15 build would have refused is the upgrade's to refuse.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.kernel.declaration import (
    BarOverride,
    ControlDeclaration,
    DecisionClause,
    GuardrailMargin,
    SweptAxis,
)
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.kernel.metrics import MeritAxis
from threetears.evals.schema.models import utc_now_iso
from threetears.evals.schema.values import SweepableValue


class _V15(BaseModel):
    model_config = ConfigDict(extra="forbid")


class V15Rule(_V15):
    """A v15 question's typed decision rule: the clauses a stored campaign's floors upgrade from."""

    clauses: tuple[DecisionClause, ...] = Field(min_length=1)


class V15Question(_V15):
    """A v15 declared question: its words, its rule, its axes, and when it was asked, retired or superseded."""

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    text: str = Field(min_length=1)
    decision: V15Rule | None = None
    merit_axes: list[MeritAxis] = Field(default_factory=list)
    asked_at: str = Field(default_factory=utc_now_iso)
    retired_at: str | None = None
    supersedes: str | None = None


class V15Design(_V15):
    """A v15 campaign's ``declared_design``: one design, its questions and its control beside it."""

    axes: list[SweptAxis] = Field(min_length=1)
    questions: list[V15Question] = Field(default_factory=list)
    bars: list[BarOverride] = Field(default_factory=list)
    guardrail_margins: list[GuardrailMargin] = Field(default_factory=list)
    control: str | None = None
    control_levers: dict[str, SweepableValue] | None = None
    control_identity_version: int | None = None
    intended_repetitions: int | None = None
    held_fixed: ControlDeclaration
    merit_priority: list[MeritAxis] = Field(default_factory=list)
    crossing: Literal["full", "star"] | None = None
    skipped_cells: list[dict[str, str]] = Field(default_factory=list)
    measure_latency: bool = False
    declared_at: str = Field(default_factory=utc_now_iso)
    declared_by: str = ""

    @model_validator(mode="before")
    @classmethod
    def _a_typed_control_is_dated(cls, data: Any) -> Any:
        """Date a control typed without a predicate with this build's, as a v15 build dated it on write."""
        if isinstance(data, dict) and data.get("control") and data.get("control_identity_version") is None:
            return {**data, "control_identity_version": IDENTITY_VERSION}
        return data

    def live_questions(self) -> list[V15Question]:
        """The questions neither retired nor superseded, as a v15 build read them."""
        superseded = {question.supersedes for question in self.questions if question.supersedes}
        return [q for q in self.questions if q.retired_at is None and q.id not in superseded]


__all__ = ["V15Design", "V15Question", "V15Rule"]
