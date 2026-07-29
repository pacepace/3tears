"""set algebra over dataset terms, with per-term projection.

``composition`` is a :class:`SetExpr` over DATASET TERMS -- a unit, a
rollup, or an upstream dataset -- combined by ``union``, ``intersect``,
``difference``, or ``ranked_precedence``, and defaulting to the union of
every unit. That default is what the prototype does by having every unit
``INSERT`` into one table with nothing filtering the union.

Four things this module gets right that a naive set algebra does not:

- **each term carries its own projection.** One shipped deliverable
  unions two audiences where one branch projects three rollup flags and
  the other projects ``NULL`` for all three and computes its category by
  a different rule. A single delivery spec over a composition cannot
  express that, so the projection is per-term. A second deliverable does
  the same at the pivot: one branch projects unit flags and ``0 AS
  householders``, the other projects ``0`` for every unit flag and
  computes ``householders``.
- **``intersect`` carries payload, not just membership.** Its one real
  use delivers aggregated columns drawn from BOTH sides -- a list of
  source records from each, under names that differ while the source
  column name does not. So the intersect declares, per output column,
  which side it comes from.
- **``intersect`` occurs in two positions, and they are two plans.** At
  composition it filters the composed set; inside a resolution it applies
  before that resolution's own aggregation, via a reference to another
  unit's resolved rows. :class:`ResolutionIntersect` is a structurally
  distinct node rather than a flag on :class:`SetExpr`, because a flag
  invites the compiler to guess. It attaches at
  :attr:`~threetears.datasources.definition.unit.Resolution.intersect`,
  which is the position it was built for: a declared and documented type
  reachable from no field invites the next author to assume a capability
  the model does not have.
- **``ranked_precedence`` carries TWO independent orders.** Its
  motivating audience deduplicates core-before-expansion while labelling
  householders-before-core-before-expansion, computes that label BEFORE
  the dedup, and its first arm tests an expansion flag rather than a
  dataset term. Deriving one order from the other reproduces neither, so
  :attr:`SetExpr.dedup_order` and :attr:`SetExpr.category_order` are
  named distinctly and declared separately.

**D17: composition applies after qualification and BEFORE expansion, and
it filters the long and qualified artifacts as well as the wide one.**
The model deliberately admits no stage, phase, or placement field --
:class:`CompositionPlacement` carries exactly one member, so "after
expansion" is not spellable. The failure it prevents is invisible without
a long-artifact diff: expansion members carry no unit flag, so composing
after expansion tests them against a membership rule they cannot satisfy,
and under the default composition that silently deletes every householder
from a householder-heavy audience while the statistics stay internally
consistent and wrong.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.source import ArtifactRef
from threetears.datasources.definition.expression import Expression
from threetears.datasources.definition.namespace import BindingStage, reject_unbindable
from threetears.datasources.definition.rollup import LabelArm

__all__ = [
    "COMPOSITION_FILTERED_ARTIFACTS",
    "CategoryPosition",
    "CompositionPlacement",
    "IntersectColumn",
    "ResolutionIntersect",
    "ResolutionIntersectColumn",
    "SetExpr",
    "SetOperator",
    "SetTerm",
    "TermColumn",
]

COMPOSITION_FILTERED_ARTIFACTS: Final[tuple[str, ...]] = ("long", "qualified", "wide")
"""artifacts composition filters, per D17.

Filtering the wide artifact alone leaves an entity excluded by an
``intersect`` with long rows, and per-unit counts computed from long then
count non-members. Recorded here so ``dsc-task-03`` reads the decision
rather than re-deriving it.
"""


class CompositionPlacement(StrEnum):
    """the one placement composition has, per D17.

    One member, deliberately. Composition applies after qualification and
    before expansion, and no second member exists for a plan to choose.
    Expansion members carry no unit flag, so composing after expansion
    would test them against a membership rule they cannot satisfy.

    :cvar AFTER_QUALIFICATION_BEFORE_EXPANSION: the only placement
    """

    AFTER_QUALIFICATION_BEFORE_EXPANSION = "after_qualification_before_expansion"


class SetOperator(StrEnum):
    """how composition terms combine.

    :cvar UNION: every term's members, the default composition
    :cvar INTERSECT: members of every term, carrying payload from both sides
    :cvar DIFFERENCE: the first term less every later one, ordered
    :cvar RANKED_PRECEDENCE: deduplicated by a declared order, with a
        separately declared delivered category
    """

    UNION = "union"
    INTERSECT = "intersect"
    DIFFERENCE = "difference"
    RANKED_PRECEDENCE = "ranked_precedence"


class CategoryPosition(StrEnum):
    """when ``ranked_precedence``'s delivered category is computed.

    Load-bearing rather than cosmetic. The one real audience computes the
    label BEFORE the dedup, and its first arm reads a flag carried by the
    losing term -- so an entity present in both terms is labelled
    ``householder`` when the label runs first and ``core`` when it runs
    after the surviving row is chosen.

    :cvar BEFORE_DEDUP: computed on every row, then deduplicated
    :cvar AFTER_DEDUP: computed on the surviving row only
    """

    BEFORE_DEDUP = "before_dedup"
    AFTER_DEDUP = "after_dedup"


class TermColumn(BaseModel):
    """one column of one composition term's own projection.

    Distinct from ``dsm-task-01d``'s ``DerivedColumn``, which is the
    delivery-level shape over a whole artifact. A term column binds
    positionally inside a union arm, which is what lets one branch
    project ``NULL`` where another projects a rollup flag.

    :ivar name: output column name, shared across the terms of one
        composition
    :ivar value: what THIS term projects into it; ``{"literal": null}``
        is the ``NULL`` the corpus's opinion-elite branch projects
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: Expression

    @model_validator(mode="after")
    def _name_is_not_blank(self) -> Self:
        """reject a column name that is only whitespace.

        :returns: validated column
        :rtype: TermColumn
        :raises ValueError: name carries no text
        """
        if not self.name.strip():
            raise ValueError("term column name carries no text")
        return self


class IntersectColumn(BaseModel):
    """one output column of a composition intersect, and which term supplies it.

    Required because the one real intersection delivers a list of source
    records from EACH side under different output names while the source
    column name is the same on both, so membership alone loses the
    payload and a name alone cannot say which side it came from.

    :ivar name: output column name
    :ivar term: name of the :class:`SetTerm` supplying it
    :ivar column: that term's own column
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    term: str = Field(min_length=1)
    column: str = Field(min_length=1)


class ResolutionIntersectColumn(BaseModel):
    """one output column drawn from a resolution-stage intersect's other side.

    Carries no ``term``: a resolution intersect has exactly two sides,
    the resolution itself and the set it intersects, and the resolution's
    own columns are its ordinary projection rather than this node's
    business.

    :ivar name: output column name
    :ivar column: the intersected side's own column
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    column: str = Field(min_length=1)


class ResolutionIntersect(BaseModel):
    """intersection applied INSIDE a resolution, before its own aggregation.

    Structurally distinct from :class:`SetExpr`'s ``intersect`` operator
    rather than the same node with a stage flag, because the two compile
    to different plans: the composition operator filters the composed
    set, while this one narrows a resolution's rows before that
    resolution groups them. A flag would invite the compiler to guess
    which plan was meant.

    :ivar alias: name this intersected side is known by inside the
        resolution
    :ivar against: the set intersected, typically another unit's resolved
        rows
    :ivar key_columns: columns matching the two sides
    :ivar payload: columns drawn out of the intersected side
    """

    model_config = ConfigDict(extra="forbid")

    alias: str = Field(min_length=1)
    against: ArtifactRef
    key_columns: list[str] = Field(min_length=1)
    payload: list[ResolutionIntersectColumn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _alias_is_addressable_and_columns_are_distinct(self) -> Self:
        """reject an unaddressable alias, or a repeated key or output column.

        the alias must be an identifier for the same reason
        :class:`~threetears.datasources.definition.relation.RelationRef`'s
        must: it is the ``rel.<alias>`` half of a namespaced reference, so
        a non-identifier alias leaves the intersected side addressable by
        no predicate at all. that failure surfaces as an unresolved name
        against a definition the model itself accepted, which reads as a
        compiler defect rather than an authoring one.

        :returns: validated node
        :rtype: ResolutionIntersect
        :raises ValueError: the alias is not an identifier, the key
            columns or a payload output name repeat, or a key column is
            blank
        """
        if not self.alias.isidentifier():
            raise ValueError(f"intersect alias {self.alias!r} is not an identifier")
        if any(not column.strip() for column in self.key_columns):
            raise ValueError(f"{self.alias}: key_columns carries a blank column name")
        if len(self.key_columns) != len(set(self.key_columns)):
            raise ValueError(f"{self.alias}: key_columns names a column twice")
        names = [column.name for column in self.payload]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.alias}: payload names an output column twice")
        return self


class SetTerm(BaseModel):
    """one dataset term of a composition, with its own projection.

    A term is a unit, a rollup, an upstream dataset, or a nested
    :class:`SetExpr`. There is deliberately NO expansion kind: expansion
    members carry no unit flag and composition runs before expansion, so
    a term naming one could only be tested against a rule it cannot
    satisfy.

    :ivar name: term identity, referenced by ``dedup_order`` and by
        :class:`IntersectColumn`; the corpus's per-term ``'core'`` /
        ``'expansion'`` literal is this identity made into data
    :ivar unit: unit of this definition
    :ivar rollup: rollup of this definition
    :ivar upstream: another dataset, by name
    :ivar expression: nested composition
    :ivar projection: what THIS term projects, which is where one branch
        projects three rollup flags and another projects ``NULL``
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    unit: str | None = Field(default=None, min_length=1)
    rollup: str | None = Field(default=None, min_length=1)
    upstream: str | None = Field(default=None, min_length=1)
    expression: SetExpr | None = None
    projection: list[TermColumn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names_exactly_one_kind(self) -> Self:
        """require exactly one kind, and a projection with distinct names.

        :returns: validated term
        :rtype: SetTerm
        :raises ValueError: no kind or more than one kind is named, or a
            projected column name repeats
        """
        named = [kind for kind in ("unit", "rollup", "upstream", "expression") if getattr(self, kind) is not None]
        if len(named) != 1:
            raise ValueError(
                f"{self.name}: a composition term names exactly one of unit / rollup / upstream / "
                f"expression; got {named or 'none'}"
            )
        columns = [column.name for column in self.projection]
        if len(columns) != len(set(columns)):
            raise ValueError(f"{self.name}: projection names a column twice")
        return self


_RANKED_FIELDS: tuple[str, ...] = (
    "dedup_key_columns",
    "dedup_order",
    "tiebreak",
    "category_order",
    "category_position",
)


class SetExpr(BaseModel):
    """composition over dataset terms, at the one placement composition has.

    Carries no stage, phase, or placement field, and never will: per D17
    composition runs after qualification and before expansion, and
    admitting a second placement would let a plan silently delete every
    householder from a householder-heavy audience while the statistics
    stayed internally consistent.

    :ivar op: how the terms combine
    :ivar terms: dataset terms, in authored order; empty means the
        default composition, the union of every unit
    :ivar payload: for ``intersect`` only -- per output column, which
        term supplies it
    :ivar dedup_key_columns: for ``ranked_precedence`` only -- columns
        one surviving row is chosen per; the design spells this
        ``dedup_key`` and
        :mod:`~threetears.datasources.definition.exclusion` records why
        the short spelling is unavailable
    :ivar dedup_order: for ``ranked_precedence`` only -- term names in
        precedence order, covering every declared term exactly once
    :ivar tiebreak: for ``ranked_precedence`` only -- columns ordering
        rows WITHIN one term, because a term can carry an entity twice
        and an undeclared tiebreak makes the survivor arbitrary
    :ivar category_order: for ``ranked_precedence`` only -- the DELIVERED
        category, a separate declaration from :attr:`dedup_order` and in
        the one real audience a different order with an extra arm
    :ivar category_otherwise: label for a row matching no arm, or
        ``None`` for an implicit NULL bucket
    :ivar category_position: whether the category is computed before or
        after the dedup, which changes the delivered label of any entity
        present in more than one term
    """

    model_config = ConfigDict(extra="forbid")

    op: SetOperator = SetOperator.UNION
    terms: list[SetTerm] = Field(default_factory=list)
    payload: list[IntersectColumn] = Field(default_factory=list)
    dedup_key_columns: list[str] | None = None
    dedup_order: list[str] | None = None
    tiebreak: list[str] | None = None
    category_order: list[LabelArm] | None = None
    category_otherwise: str | None = Field(default=None, min_length=1)
    category_position: CategoryPosition | None = None

    @model_validator(mode="after")
    def _operator_carries_its_own_declarations(self) -> Self:
        """reject declarations belonging to a different operator, and missing ones.

        :returns: validated expression
        :rtype: SetExpr
        :raises ValueError: a term name repeats, an operator carries too
            few terms, a payload or ranked declaration is present for the
            wrong operator or absent for the right one, ``dedup_order``
            does not cover the terms, or a category arm binds a
            pre-aggregate namespace
        """
        self._terms_are_distinct()
        self._arity_matches_the_operator()
        self._payload_belongs_to_intersect()
        self._ranked_declarations_are_complete()
        return self

    def _terms_are_distinct(self) -> None:
        """reject two terms claiming one name.

        :returns: None
        :rtype: None
        :raises ValueError: a term name repeats
        """
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError(
                "a composition term name is declared twice; dedup_order and intersect payload "
                "address terms by name, so a duplicate makes both ambiguous"
            )

    def _arity_matches_the_operator(self) -> None:
        """reject an operator carrying too few terms.

        :returns: None
        :rtype: None
        :raises ValueError: a non-union operator carries fewer than two terms
        """
        if self.op is not SetOperator.UNION and len(self.terms) < 2:
            raise ValueError(
                f"{self.op.value!r} combines two or more terms; got {len(self.terms)}. only union "
                "may carry no terms, where it means the default composition of every unit"
            )

    def _payload_belongs_to_intersect(self) -> None:
        """require a payload for ``intersect`` and forbid one elsewhere.

        :returns: None
        :rtype: None
        :raises ValueError: payload is absent on an intersect, present on
            another operator, repeats an output name, or names an
            undeclared term
        """
        if self.op is not SetOperator.INTERSECT:
            if self.payload:
                raise ValueError(
                    f"payload declares which side each output column comes from, so {self.op.value!r} carries none"
                )
            return
        if not self.payload:
            raise ValueError(
                "intersect carries payload, not just membership: declare per output column which "
                "term supplies it. its one real use delivers aggregated columns from both sides"
            )
        names = [column.name for column in self.payload]
        if len(names) != len(set(names)):
            raise ValueError("payload names an output column twice")
        declared = {term.name for term in self.terms}
        unknown = sorted({column.term for column in self.payload} - declared)
        if unknown:
            raise ValueError(f"payload draws from undeclared terms {unknown}")

    def _ranked_declarations_are_complete(self) -> None:
        """require every ranked declaration for ``ranked_precedence`` and none elsewhere.

        :returns: None
        :rtype: None
        :raises ValueError: a ranked declaration is present for another
            operator or absent for this one, ``dedup_order`` does not
            cover the terms exactly once, a category label repeats, or a
            category arm binds a namespace composition cannot bind
        """
        present = [name for name in _RANKED_FIELDS if getattr(self, name) is not None]
        if self.op is not SetOperator.RANKED_PRECEDENCE:
            if present or self.category_otherwise is not None:
                raise ValueError(
                    f"{present or ['category_otherwise']} belong to ranked_precedence; "
                    f"{self.op.value!r} declares no precedence"
                )
            return
        missing = [name for name in _RANKED_FIELDS if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"ranked_precedence carries no {missing}. the dedup order, the tiebreak, and the "
                "delivered category are separate declarations, and an undeclared one leaves the "
                "surviving row or the delivered label arbitrary"
            )
        self._ranked_lists_are_well_formed()

    def _ranked_lists_are_well_formed(self) -> None:
        """check the ranked declarations against the declared terms.

        :returns: None
        :rtype: None
        :raises ValueError: a key or tiebreak column repeats,
            ``dedup_order`` does not cover the terms exactly once, a
            category label repeats, or a category arm binds a
            pre-aggregate namespace
        """
        for field_name in ("dedup_key_columns", "tiebreak"):
            columns: list[str] = getattr(self, field_name)
            if len(columns) != len(set(columns)):
                raise ValueError(f"{field_name} names a column twice")
        order: list[str] = self.dedup_order or []
        declared = [term.name for term in self.terms]
        if sorted(order) != sorted(declared):
            raise ValueError(
                f"dedup_order {order} does not cover the declared terms {declared} exactly once; "
                "an unranked term has no defined precedence against the others"
            )
        arms: list[LabelArm] = self.category_order or []
        labels = [arm.label for arm in arms]
        if len(labels) != len(set(labels)):
            raise ValueError("category_order delivers the same label from two arms, and first match wins")
        for index, arm in enumerate(arms):
            reject_unbindable(arm.when.references, BindingStage.QUALIFICATION, f"SetExpr.category_order[{index}].when")

    @property
    def is_default_union(self) -> bool:
        """whether this is the default composition, the union of every unit.

        :returns: True for a union declaring no terms
        :rtype: bool
        """
        return self.op is SetOperator.UNION and not self.terms

    @property
    def placement(self) -> CompositionPlacement:
        """where composition runs, per D17.

        Constant by construction: there is one placement and no field
        selects it.

        :returns: after qualification and before expansion
        :rtype: CompositionPlacement
        """
        return CompositionPlacement.AFTER_QUALIFICATION_BEFORE_EXPANSION


SetTerm.model_rebuild()
