"""expansion: walking an edge out of the qualified set, and recording who by.

An expansion adds members reached from an already-qualified member --
householders, spouses -- and it is a first-class part of the definition
because the prototype's version of it is a dead field. ``include_householders``
is declared 92 times across two shipped audiences and is read by exactly
one Python property that is never called, so those audiences shipped with
no expansion members at all while their settings said otherwise.

Four properties, each from committed evidence:

- :attr:`Expansion.edge` is a
  :class:`~threetears.datasources.definition.relation.RelationRef` and may
  branch from a NON-GRAIN key. One committed edge joins on ``list_id``
  rather than on the entity, which is legal and changes what the walk
  produces.
- :attr:`Expansion.member_column` names which side of the edge carries
  the ADDED member. The join pins the other side -- ``household`` is
  joined on ``influencer_voterbase_id`` -- and the walked relation
  carries no column of the entity's own name at all, so an emitter that
  reads the entity column off the edge alias emits a column the relation
  does not have. It is declared rather than inferred from the rationale
  body, because an expansion may legitimately emit no rationale.
- :attr:`Expansion.applies_to` scopes the walk to a unit SET. The dead
  generation's own committed SQL does exactly that, selecting the
  eligible entities with ``WHERE unit IN (...)``.
- :attr:`Expansion.exclude_existing` anti-joins the QUALIFIED set, which
  is a different stage from
  :class:`~threetears.datasources.definition.exclusion.ExclusionSpec`'s
  subtrahend. Both stages are real and neither is a default for the
  other.
- :attr:`Expansion.provenance` is the ONLY place an expansion member's
  per-person rationale exists. Its ``fact`` column carries the influencer
  who brought each member in and its ``fact_type`` the literal
  ``'<relationship> of'``; it lands in the ``relationship_union``
  artifact, which is the fifth artifact kind.

Expansion members carry no unit flag and no quality measurement at all,
which is why composition runs BEFORE expansion (D17) and why the skew
detector needs an explicit unmeasured band (D16).
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import Predicate
from threetears.datasources.definition.namespace import Namespace, Reference
from threetears.datasources.definition.provenance import ProvenanceSpec
from threetears.datasources.definition.relation import RelationRef

__all__ = ["Expansion"]


class Expansion(BaseModel):
    """one edge walked out of the qualified set.

    :ivar name: relationship name; stamped as a literal on every expansion
        row and read back as the delivered relationship column
    :ivar edge: the join walked, which may branch from a non-grain key
    :ivar member_column: column of the walked relation carrying the
        entity this walk ADDS, which is never the column the edge joins
        on and is not the entity's own name on any committed edge
    :ivar applies_to: unit names this walk starts from; empty means every
        unit
    :ivar predicate: per-edge filter
    :ivar exclude_existing: whether members already in the QUALIFIED set
        are dropped
    :ivar provenance: which member brought each one in
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    edge: RelationRef
    member_column: str = Field(min_length=1)
    applies_to: list[str] = Field(default_factory=list)
    predicate: Predicate | None = None
    exclude_existing: bool = True
    provenance: ProvenanceSpec | None = None

    @model_validator(mode="after")
    def _scope_is_coherent(self) -> Self:
        """reject a blank name and a repeating scope.

        :returns: validated expansion
        :rtype: Expansion
        :raises ValueError: the name is blank, the member column is blank
            or is the column the edge joins on, or applies_to repeats a
            unit
        """
        if not self.name.strip():
            raise ValueError("expansion name carries no text")
        if not self.member_column.strip():
            raise ValueError(f"{self.name}: member_column carries no text")
        joined = {
            reference.name
            for reference in self.edge.references
            if reference.alias == self.edge.alias and reference.namespace is Namespace.REL
        }
        if self.member_column in joined:
            raise ValueError(
                f"{self.name}: member_column {self.member_column!r} is the column the edge joins on, "
                "so the walk would add back the member it started from"
            )
        if len(self.applies_to) != len(set(self.applies_to)):
            raise ValueError(f"{self.name}: applies_to names a unit twice")
        if any(not unit.strip() for unit in self.applies_to):
            raise ValueError(f"{self.name}: applies_to carries a blank unit name")
        return self

    def covers(self, unit_name: str) -> bool:
        """whether this walk starts from one unit.

        :param unit_name: declared unit name
        :ptype unit_name: str
        :returns: True when the walk names it, or names no unit at all
        :rtype: bool
        """
        return not self.applies_to or unit_name in self.applies_to

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from the edge, the filter, and the rationale.

        :returns: references in edge-then-filter-then-provenance order
        :rtype: tuple[Reference, ...]
        """
        collected = self.edge.references
        if self.predicate is not None:
            collected = collected + self.predicate.references
        if self.provenance is not None:
            collected = collected + self.provenance.references
        return collected
