"""rollups: a named group of units, and the label it is obliged to emit.

A rollup names a group of units. It emits a flag column in the wide
projection AND a label on the long and provenance projections -- the
prototype stamps that label per row, and the rollup is the grain two of
the four client deliverables are sold at. So :attr:`Rollup.emit` admits
``wide_flag``, ``long_label``, and ``provenance_label``, and a rollup
layer that only ever emitted a wide flag would not deliver either of
those two.

The corpus evidence is that rollups were never the tool's:

- ``rollup_unit`` has 20 authored sites in ``amazon_tech_audience`` and
  NO template support at all. The committed long DDL carries a
  ``rollup_unit varchar`` column and stamps a label at 19 ``INSERT``
  sites, and no file under the prototype's templates mentions the key.
  The audience's own YAML comment says the tool "does not currently
  support these types of things".
- the UHG rollups are hand-written SQL entirely outside the tool, and
  they are the single strongest argument that the tool never owned the
  deliverable.

Two properties are load-bearing rather than incidental:

- :attr:`Rollup.members` is an ORDERED list and first match wins. A set
  silently changes which label a multi-member entity gets, and entities
  ARE in several rollups: the two UHG rollups cover 22 of 23 units and 83
  entities are in both.
- membership is NOT exclusive and rollups are NOT a partition, so
  overlap across rollups is legal and only a repeat WITHIN one rollup is
  refused.

:class:`LabelArm` is here rather than in ``setexpr`` because it is the
same shape a rollup's ordered first-match-wins categorisation takes, and
``ranked_precedence``'s delivered category reuses it. Its arms are
predicates rather than term names, because the one real category's first
arm tests an expansion flag rather than a dataset term.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.source import ArtifactRef
from threetears.datasources.definition.expression import Predicate

__all__ = ["LabelArm", "Rollup", "RollupEmit"]


class RollupEmit(StrEnum):
    """projection a rollup lands in.

    :cvar WIDE_FLAG: one flag column in the wide projection
    :cvar LONG_LABEL: a label stamped per long row
    :cvar PROVENANCE_LABEL: a label stamped per provenance row
    """

    WIDE_FLAG = "wide_flag"
    LONG_LABEL = "long_label"
    PROVENANCE_LABEL = "provenance_label"


class LabelArm(BaseModel):
    """one arm of an ordered, first-match-wins categorisation.

    The arm is a PREDICATE rather than a term name, because the one real
    delivered category's first arm tests an expansion flag
    (``householders = 1``) rather than a dataset term. Modelling arms as
    a list of term names would make that category inexpressible.

    :ivar when: condition selecting this arm
    :ivar label: label delivered when the arm matches
    """

    model_config = ConfigDict(extra="forbid")

    when: Predicate
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def _label_is_not_blank(self) -> Self:
        """reject a label that is only whitespace.

        :returns: validated arm
        :rtype: LabelArm
        :raises ValueError: label carries no text
        """
        if not self.label.strip():
            raise ValueError("label carries no text")
        return self


class Rollup(BaseModel):
    """a named group of units, and where its flag and label are projected.

    :ivar name: rollup name; also the label value it delivers
    :ivar members: units (or rollups) this group covers, ORDERED, first
        match wins; a member may carry the same name as the rollup, which
        ``amazon_tech_audience`` does
    :ivar otherwise: label for an entity in none of :attr:`members`, or
        ``None`` for an implicit NULL bucket
    :ivar over: artifact this rollup is computed against; one real rollup
        is computed over the PROVENANCE artifact rather than long or wide
    :ivar emit: projections this rollup lands in, at least one
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    members: list[str] = Field(min_length=1)
    otherwise: str | None = Field(default=None, min_length=1)
    over: ArtifactRef | None = None
    emit: list[RollupEmit] = Field(min_length=1)

    @model_validator(mode="after")
    def _members_and_emit_are_distinct(self) -> Self:
        """reject a repeated member or emit kind, and a self-referential bucket.

        Overlap ACROSS rollups is legal and observed -- 83 entities are
        in both UHG rollups -- so only a repeat WITHIN one rollup is
        refused, where first-match-wins makes the second copy dead.

        :returns: validated rollup
        :rtype: Rollup
        :raises ValueError: a member or emit kind repeats, a member is
            blank, or ``otherwise`` equals the rollup name
        """
        if any(not member.strip() for member in self.members):
            raise ValueError(f"{self.name}: members carries a blank name")
        if len(self.members) != len(set(self.members)):
            raise ValueError(
                f"{self.name}: members names the same member twice, and first match wins, so the "
                "second copy is dead. overlap ACROSS rollups is legal and is declared there"
            )
        if len(self.emit) != len(set(self.emit)):
            raise ValueError(f"{self.name}: emit names the same projection twice")
        if self.otherwise == self.name:
            raise ValueError(
                f"{self.name}: otherwise equals the rollup name, so members and non-members deliver "
                "the same label and the rollup distinguishes nothing"
            )
        return self

    @property
    def emits_wide_flag(self) -> bool:
        """whether this rollup projects a flag column in the wide artifact.

        :returns: True when ``wide_flag`` is emitted
        :rtype: bool
        """
        return RollupEmit.WIDE_FLAG in self.emit

    @property
    def emits_label(self) -> bool:
        """whether this rollup stamps a label on long or provenance rows.

        :returns: True when either label kind is emitted
        :rtype: bool
        """
        return bool({RollupEmit.LONG_LABEL, RollupEmit.PROVENANCE_LABEL} & set(self.emit))
