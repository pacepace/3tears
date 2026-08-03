"""per-stage FROM extensions: relation references and derived-table bodies.

A :class:`RelationRef` is one join a resolution, qualification, or
composition term adds to its own FROM clause. Four of its six fields
exist because the prototype gets them wrong in a way no projection check
can see:

- **the join kind is AUTHORED.** ``1_generate_audience_units_table.sql.jinja2:37``
  sets ``join_type = 'INNER' if "industries" in linkedin_query_dict else
  'LEFT'``, so declaring an industry filter changes the audience twice --
  once by the predicate and once by dropping null-extended rows -- and
  nothing in the authored YAML says so. :attr:`RelationRef.join` is never
  inferred from another field's presence.
- **:attr:`RelationRef.optional` states the null semantics.**
  ``null_industry_execs`` is non-empty ONLY because omitting ``industries``
  flips two joins to ``LEFT``, and two units partition the LinkedIn
  universe on that. The same declaration is what makes the corpus's
  three-valued-logic inversion visible: ``sa.source_archetype NOT IN (…)``
  over a ``LEFT`` join drops every unmatched source, silently converting
  the outer join into an inner one and reversing the authored intent.
- **:attr:`RelationRef.when` omits the join ENTIRELY.** ``tsmart_comm``
  injects a join at four stages across seven sites, and at the pivot it
  applies to the expansion branch only, never the influencer branch,
  because that branch reads an already-restricted table. So the gate is
  authored per relation per stage, not once per definition, and no scalar
  substitution expresses it.
- **:attr:`RelationRef.on` is the join CONDITION, distinct from the
  body.** The corpus authors ``table`` and ``on`` as separate keys, and
  ``on`` may reference an alias declared by an earlier join in the same
  list.

``when`` and ``optional`` are NOT the same mechanism and are not
collapsible. ``when`` false means the join is not in the query at all;
``optional`` true means it is present as a ``LEFT``. A definition that
omits a join and one that outer-joins it produce different audiences.

Two derived-table TYPES, not one type with a mode flag.
:class:`TypedDerivedTable` is a first-class field: a nested ``LEFT JOIN``,
a ``GROUP BY``, a ``HAVING``, and a nine-element ``IN`` list are all real
in the corpus, used by five of eight audiences.
:class:`RawDerivedTable` holds a SQL string and IS an escape hatch, which
``parity-task-03`` scores as a ``raw:`` use. Making that a type rather
than a flag lets ``dsh-task-15``'s gate be structural instead of a
runtime string check.

**A typed body stacks row sources** (:attr:`TypedDerivedTable.union`).
``coworkers_of_linkedin_execs.sql.jinja2:1-40`` builds its company
allowlist as ``SELECT DISTINCT company_id FROM (A UNION ALL B UNION ALL
C)``, each arm a five-table join with its own filters. Without the field
the only authoring is three ``LEFT JOIN``s plus ``any_of(IS NOT NULL)`` --
set-equivalent, and a rewrite rather than a transcription, which
``parity-task-01`` diffs. :attr:`TypedDerivedTable.union_all` is REQUIRED
alongside it and defaults to nothing: ``UNION`` de-duplicates and ``UNION
ALL`` does not, the difference changes every aggregate over the result,
and this module already refuses to infer the join kind for the same
reason. The stack is FLAT -- an arm declaring its own union is refused,
because a mixed stack would make the emitted parenthesisation decide the
row count.

``dsm-task-01d`` closed the seam ``dsm-task-01c`` left here:
:attr:`RelationRef.relation` now admits an
:class:`~threetears.datasources.definition.source.ArtifactRef`, so a
relation can be re-pointed at another artifact of this definition, at
another dataset's published output, or at an external governed relation
carrying a per-run schema. The reference is a FORWARD one, because
``source`` imports this module for :class:`Projection` and
:class:`RelationRef`; ``source`` closes it, and importing the
``definition`` package always imports ``source``. That is also why this
module no longer calls ``model_rebuild()`` itself -- at the point it used
to, ``ArtifactRef`` does not exist yet, and pydantic completes both
models on first use anyway. The relation LAYER's four
gaps -- a payload field for a quality measure, ``from_schema``, a
per-edge predicate slot, and non-chain edges -- are ``dsh-task-24``'s
work on ``JoinPath`` and ``datasource_relations``. This module is shaped
to CONSUME the extended layer: quality rides on
:class:`~threetears.datasources.definition.bridge.BridgeRef`, the
per-edge predicate is :attr:`RelationRef.on`, and because ``on`` carries
an arbitrary predicate rather than a key pair, an edge that branches from
a non-grain key is authorable today and only its resolution waits.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import Expression, LiteralExpression, Predicate
from threetears.datasources.definition.namespace import Namespace, Reference

if TYPE_CHECKING:
    from threetears.datasources.definition.source import ArtifactRef

__all__ = [
    "DerivedTable",
    "DuplicateRelationAlias",
    "JoinKind",
    "Projection",
    "RawDerivedTable",
    "RelationBody",
    "RelationRef",
    "TypedDerivedTable",
    "UndeclaredRelationAlias",
    "validate_relation_aliases",
]


class DuplicateRelationAlias(ValueError):
    """raised when two relations in one list claim one alias.

    ``rel.<alias>.*`` is a namespace key, so a repeated alias makes every
    predicate that names it ambiguous. The prototype's alias drift --
    ``knowwho.`` authored, ``facts.`` emitted -- is the defect the
    namespace exists to close, and a duplicate reopens it.
    """


class UndeclaredRelationAlias(ValueError):
    """raised when a join condition names an alias not yet in scope.

    Aliases scope left to right: ``cmte.cmte_id = ext.cmte_id`` is legal
    because ``ext`` is declared by an earlier entry in the same list.
    Naming a LATER alias is a forward reference SQL cannot satisfy.
    """


class JoinKind(StrEnum):
    """join kind emitted for one relation, authored rather than inferred.

    :cvar INNER: ``INNER JOIN``
    :cvar LEFT: ``LEFT JOIN``; null-extends the left side
    :cvar RIGHT: ``RIGHT JOIN``
    :cvar FULL: ``FULL OUTER JOIN``; null-extends both sides
    :cvar CROSS: ``CROSS JOIN``; carries no condition
    """

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"


_NULL_EXTENDING_JOINS = frozenset({JoinKind.LEFT, JoinKind.FULL})


def _canonicalise_join(value: object) -> object:
    """lower-case an authored join keyword before validation.

    The corpus authors ``INNER`` and ``LEFT`` upper-case in ``how:`` and
    ``cat_join_type:``; the member values are lower-case.

    :param value: raw authored join keyword
    :ptype value: object
    :returns: canonical spelling when recognisable, else value unchanged
    :rtype: object
    """
    canonical: object = value
    if isinstance(value, str):
        collapsed = " ".join(value.split()).lower()
        if collapsed in {member.value for member in JoinKind}:
            canonical = collapsed
    return canonical


class Projection(BaseModel):
    """one output column of a derived-table body.

    :ivar expression: value this column carries
    :ivar alias: output name, or ``None`` when the expression names itself
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    expression: Expression
    alias: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _alias_is_a_name(self) -> Self:
        """reject an alias that is not an identifier.

        :returns: validated projection
        :rtype: Projection
        :raises ValueError: alias is not an identifier
        """
        if self.alias is not None and not self.alias.isidentifier():
            raise ValueError(f"projection alias {self.alias!r} is not an identifier")
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this projection.

        :returns: references in appearance order
        :rtype: tuple[Reference, ...]
        """
        return (self.expression,) if isinstance(self.expression, Reference) else self.expression.references


class RawDerivedTable(BaseModel):
    """derived-table body held as a SQL string.

    This IS an escape hatch and counts as a ``raw:`` use for
    ``parity-task-03``'s gate. It is a separate TYPE from
    :class:`TypedDerivedTable` rather than a flag on one type, so the gate
    is a structural check rather than a runtime inspection of a mode
    field.

    :ivar raw_sql: body text, re-emitted byte-exactly
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_sql: str

    @model_validator(mode="after")
    def _body_is_not_blank(self) -> Self:
        """reject empty body text.

        :returns: validated body
        :rtype: RawDerivedTable
        :raises ValueError: text is blank
        """
        if not self.raw_sql.strip():
            raise ValueError("raw_sql carries no text")
        return self

    @property
    def is_escape_hatch(self) -> bool:
        """whether this body escapes the typed model.

        :returns: True; a SQL string is always an escape hatch
        :rtype: bool
        """
        return True


class TypedDerivedTable(BaseModel):
    """derived-table body expressed in the model rather than in SQL text.

    A first-class field, NOT an escape hatch. The shape is drawn from
    ``amazon_audience_l2_2025/standard_audience_units.yaml:11-23``, used
    by five of eight audiences: a parenthesised ``SELECT`` over a nested
    ``LEFT JOIN`` to an inner aggregate, a ``GROUP BY``, and a
    nine-element membership test. ``uhg_opinion_elites`` adds a body
    carrying its own ``HAVING``.

    :ivar projections: output columns, at least one, in authored order
    :ivar distinct: whether the body de-duplicates its own rows
    :ivar source: base relation name, or a nested body
    :ivar source_alias: alias the base relation is addressed by
    :ivar relations: joins this body adds, in authored order
    :ivar where: filter over this body's own scope
    :ivar group_by: group key; an integer literal is a positional ordinal
        into :attr:`projections`, which is how the corpus authors it
    :ivar having: post-aggregate filter over this body's own scope
    :ivar union: further row sources stacked onto this one, each its own
        naming scope; empty means this body is one ``SELECT``
    :ivar union_all: whether the stack keeps duplicate rows; required
        with a union and forbidden without one, because multiplicity
        changes every aggregate downstream and is never inferred
    """

    model_config = ConfigDict(extra="forbid")

    projections: list[Projection] = Field(min_length=1)
    distinct: bool = False
    source: RelationBody
    source_alias: str
    relations: list[RelationRef] = Field(default_factory=list)
    where: Predicate | None = None
    group_by: list[Expression] = Field(default_factory=list)
    having: Predicate | None = None
    union: list[TypedDerivedTable] = Field(default_factory=list)
    union_all: bool | None = None

    @model_validator(mode="after")
    def _body_is_coherent(self) -> Self:
        """reject a non-identifier alias and an out-of-range group ordinal.

        A positional ``GROUP BY`` that names no projection is a runtime
        failure in Redshift and a silent regrouping if the projection list
        later grows, so it is checked at authoring.

        :returns: validated body
        :rtype: TypedDerivedTable
        :raises ValueError: source alias is not an identifier, a group key
            is a literal that is not a positional ordinal in range, or the
            union declarations are incoherent
        """
        if not self.source_alias.isidentifier():
            raise ValueError(f"source_alias {self.source_alias!r} is not an identifier")
        for term in self.group_by:
            if not isinstance(term, LiteralExpression):
                continue
            ordinal = term.literal
            if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise ValueError(
                    f"group_by literal {ordinal!r} is not a positional ordinal; a group key is a "
                    "reference, an arithmetic expression, or an integer position into projections"
                )
            if not 1 <= ordinal <= len(self.projections):
                raise ValueError(
                    f"group_by ordinal {ordinal} names no projection; this body projects "
                    f"{len(self.projections)} columns"
                )
        validate_relation_aliases(self.relations, bound_aliases=(self.source_alias,))
        self._union_is_coherent()
        return self

    def _union_is_coherent(self) -> None:
        """check the stacked row sources against this body and each other.

        :returns: None
        :rtype: None
        :raises ValueError: multiplicity is undeclared with a union or
            declared without one, an arm projects a different number of
            columns, or an arm declares a union of its own
        """
        if not self.union:
            if self.union_all is not None:
                raise ValueError(
                    "union_all declares whether the stack keeps duplicate rows, and this body "
                    "declares no union; drop union_all, or name the row sources it stacks"
                )
            return
        if self.union_all is None:
            raise ValueError(
                "a union declares union_all: UNION de-duplicates and UNION ALL does not, and the "
                "difference in row multiplicity changes every aggregate computed over the result. "
                "the join kind is authored here for the same reason"
            )
        for index, arm in enumerate(self.union):
            if len(arm.projections) != len(self.projections):
                raise ValueError(
                    f"union arm {index} projects {len(arm.projections)} column(s) against this "
                    f"body's {len(self.projections)}; stacked row sources agree on arity, and the "
                    "emitted column names come from the first arm"
                )
            if arm.union:
                raise ValueError(
                    f"union arm {index} declares a union of its own; a stack is FLAT, with one "
                    "union_all governing every arm. a mixed UNION / UNION ALL stack would make the "
                    "emitted parenthesisation decide the row count, so it is refused rather than guessed"
                )

    @property
    def is_escape_hatch(self) -> bool:
        """whether any SQL string is reachable from this body.

        A typed body is a first-class field, but nesting a
        :class:`RawDerivedTable` anywhere inside it -- including inside a
        union arm -- reintroduces the escape hatch, and
        ``parity-task-03`` scores the whole body by what it actually
        contains.

        :returns: True when a raw body is reachable
        :rtype: bool
        """
        nested = [self.source] + [relation.relation for relation in self.relations] + list(self.union)
        return any(body.is_escape_hatch for body in nested if isinstance(body, RawDerivedTable | TypedDerivedTable))

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this body's own scope.

        Deliberately NOT bubbled up to the enclosing
        :class:`RelationRef`: a derived body is its own naming scope, and
        folding its aliases into the outer one would make the outer
        left-to-right alias check reject legal SQL. A :attr:`union` arm is
        a further scope of its own, one level down, and is excluded here
        for the same reason -- an arm's aliases are not in this body's
        ``FROM`` clause. Reach them through the arm's own
        :attr:`references`.

        :returns: references in authored order
        :rtype: tuple[Reference, ...]
        """
        collected: tuple[Reference, ...] = ()
        for projection in self.projections:
            collected = collected + projection.references
        for relation in self.relations:
            collected = collected + relation.references
        for term in self.group_by:
            collected = collected + ((term,) if isinstance(term, Reference) else term.references)
        for predicate in (self.where, self.having):
            if predicate is not None:
                collected = collected + predicate.references
        return collected


DerivedTable: TypeAlias = TypedDerivedTable | RawDerivedTable
"""derived-table body, typed or held as a SQL string."""

RelationBody: TypeAlias = "str | TypedDerivedTable | RawDerivedTable | ArtifactRef"
"""what a relation reference points at.

A ``str`` is the authored name of a governed relation; an
:class:`~threetears.datasources.definition.source.ArtifactRef` re-points
it at an artifact of this definition, another dataset's published output,
or an external relation carrying a per-run schema.
"""


class RelationRef(BaseModel):
    """one join a stage adds to its own FROM clause.

    :ivar relation: governed relation name, or a derived-table body
    :ivar alias: name predicates address this relation by; ``rel.<alias>.*``
    :ivar join: join kind, authored and never inferred
    :ivar on: join condition, distinct from the body; ``None`` uses the
        governed relation's declared join path
    :ivar optional: whether rows with no match survive, null-extended
    :ivar when: gate that omits the join entirely when false; binds run
        parameters only, because structure is decided before any row exists
    """

    model_config = ConfigDict(extra="forbid")

    relation: RelationBody
    alias: str
    join: Annotated[JoinKind, BeforeValidator(_canonicalise_join)]
    on: Predicate | None = None
    optional: bool = False
    when: Predicate | None = None

    @model_validator(mode="after")
    def _declaration_is_coherent(self) -> Self:
        """reject an alias, join kind, and null semantics that disagree.

        :returns: validated relation
        :rtype: RelationRef
        :raises ValueError: alias is not an identifier, a cross join
            carries a condition, ``optional`` is declared over a join that
            cannot null-extend, or ``when`` binds anything but ``param.*``
        """
        if not self.alias.isidentifier():
            raise ValueError(f"relation alias {self.alias!r} is not an identifier")
        if self.join is JoinKind.CROSS and self.on is not None:
            raise ValueError("a cross join carries no condition; move the predicate to the enclosing filter")
        if self.optional and self.join not in _NULL_EXTENDING_JOINS:
            raise ValueError(
                f"{self.alias}: optional=True declares that unmatched rows survive null-extended, "
                f"which a {self.join.value!r} join cannot do. author join=left, or optional=False"
            )
        if self.when is not None:
            offending = next(
                (reference for reference in self.when.references if reference.namespace is not Namespace.PARAM),
                None,
            )
            if offending is not None:
                raise ValueError(
                    f"{self.alias}: when references {offending.ref!r}, but a structural gate binds "
                    "param.* only -- whether a join is in the query is decided once per run, not "
                    "once per row. a row-level condition belongs in on or in the stage predicate"
                )
        return self

    @property
    def is_conditional(self) -> bool:
        """whether this join is omitted for some parameter values.

        Distinct from :attr:`optional`: a conditional join may be absent
        from the query altogether, while an optional one is present and
        null-extends.

        :returns: True when a gate is declared
        :rtype: bool
        """
        return self.when is not None

    @property
    def references(self) -> tuple[Reference, ...]:
        """references this relation binds in the ENCLOSING scope.

        The condition and the gate only. A derived body's own references
        are reachable through :attr:`TypedDerivedTable.references`, which
        is a different naming scope.

        :returns: references in condition-then-gate order
        :rtype: tuple[Reference, ...]
        """
        condition = () if self.on is None else self.on.references
        gate = () if self.when is None else self.when.references
        return condition + gate


def validate_relation_aliases(
    relations: Sequence[RelationRef], bound_aliases: Sequence[str] = ()
) -> Sequence[RelationRef]:
    """reject a duplicate alias or a forward alias reference in one list.

    Both failures are silent in the prototype: a duplicate alias resolves
    by accident of FROM order, and a forward reference is a render-time
    error that names the alias rather than the definition.

    :param relations: relations of one stage, in authored order
    :ptype relations: collections.abc.Sequence[RelationRef]
    :param bound_aliases: aliases the enclosing FROM already binds, e.g. a
        derived body's own :attr:`TypedDerivedTable.source_alias`
    :ptype bound_aliases: collections.abc.Sequence[str]
    :returns: the same sequence, unchanged
    :rtype: collections.abc.Sequence[RelationRef]
    :raises DuplicateRelationAlias: two relations claim one alias
    :raises UndeclaredRelationAlias: a condition names an alias declared
        later in the list, or not at all
    """
    in_scope: set[str] = set(bound_aliases)
    for relation in relations:
        if relation.alias in in_scope:
            raise DuplicateRelationAlias(
                f"relation alias {relation.alias!r} is declared twice; rel.<alias>.* is a namespace "
                "key, so a repeated alias makes every predicate that names it ambiguous"
            )
        in_scope.add(relation.alias)
        for reference in relation.references:
            if reference.namespace is not Namespace.REL:
                continue
            if reference.alias not in in_scope:
                raise UndeclaredRelationAlias(
                    f"{relation.alias}: condition references {reference.ref!r}, but alias "
                    f"{reference.alias!r} is not declared at or before this join. aliases scope "
                    "left to right, as the emitted FROM clause does"
                )
    return relations
