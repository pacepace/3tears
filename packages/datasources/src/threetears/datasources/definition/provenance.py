"""provenance: a projection with its own column contract, authored not derived.

Per D9a the contract is PINNED once in the Ripple binding and verified
against every emitted projection, and per D9b the pinned form is the
template DDL's 15 columns. Neither half is derivable from a unit body:
the corpus carries three mutually inconsistent versions of the contract
and nine hand-written bodies that are not recoverable from the unit query
they belong to -- one projects an aggregate that appears nowhere in it.

Five fields, and four of them exist because the corpus proves the naive
version wrong:

- :attr:`ProvenanceSpec.anchor` -- every body inner-joins the long table
  on ``voterbase_id AND unit = '<name>'``, so the anchor is a reference
  rather than an implicit "whatever produced this unit".
- :attr:`ProvenanceSpec.relations` are **NOT inherited from the
  resolution**. The ``source_archetype`` join is GATED at stage 1 and
  UNCONDITIONAL in the provenance body, so inheriting would silently add
  a join to nine bodies that never had one, or drop one from those that
  do.
- :attr:`ProvenanceSpec.grain` is required because every body is an
  aggregate query, and its ``GROUP BY`` is computed by positional
  arithmetic over conditional branches -- ``1,2,3`` then ``,4,5,6,7``
  then ``,8``. There is nothing to infer.
- :attr:`ProvenanceSpec.measures` exists because a body projects a
  measure the unit never had.

Order is part of the contract, not merely membership. Every body is
emitted as ``INSERT INTO ... ( <body> )`` with **no column list**, so the
load binds POSITIONALLY against the ``CREATE TABLE``. That is the
mechanism that arbitrates D9b, and it is also why
:meth:`ProvenanceSpec.verify_columns_against` compares positions rather
than sets.

A trap worth stating before someone hits it: the committed template
contradicts itself and gets away with it. Its DDL declares
``linkedin_industries`` (plural) and its body aliases
``linkedin_industry`` (singular), which works only because the INSERT is
positional. The check below fails on that, CORRECTLY, and the first
person to see it will conclude the compiler is broken.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import Expression
from threetears.datasources.definition.measure import Measure
from threetears.datasources.definition.namespace import Reference
from threetears.datasources.definition.relation import RelationRef, validate_relation_aliases

if TYPE_CHECKING:
    from threetears.datasources.definition.source import ArtifactRef

__all__ = [
    "ProvenanceColumn",
    "ProvenanceContractViolation",
    "ProvenanceSpec",
]


class ProvenanceContractViolation(ValueError):
    """raised when a provenance projection disagrees with the pinned contract.

    Names the exact column and its position, because the load binds
    positionally and a missing column therefore shifts every later value
    into the wrong destination rather than failing on the one that moved.
    """


class ProvenanceColumn(BaseModel):
    """one column of a provenance projection.

    :ivar name: output column name, matched against the pinned contract
    :ivar expression: value this column carries; a per-resolution
        expression, because the linkedin arm lower-cases three columns the
        standard arm does not, and one arm projects a literal where the
        other projects the source's own value
    :ivar sql_type: declared width, or ``None`` to inherit the contract's
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    expression: Expression
    sql_type: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _name_is_not_blank(self) -> Self:
        """reject a name that is only whitespace.

        :returns: validated column
        :rtype: ProvenanceColumn
        :raises ValueError: name carries no text
        """
        if not self.name.strip():
            raise ValueError("provenance column name carries no text")
        return self

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this column's expression.

        :returns: references in appearance order
        :rtype: tuple[Reference, ...]
        """
        return (self.expression,) if isinstance(self.expression, Reference) else self.expression.references


class ProvenanceSpec(BaseModel):
    """the per-person rationale one unit, resolution, or expansion emits.

    :ivar anchor: artifact this body joins back to; ``None`` for a body
        that stands alone
    :ivar relations: joins this body adds, NEVER inherited from the
        resolution
    :ivar grain: group key, required because every body is an aggregate
    :ivar measures: aggregates this body computes, including ones the unit
        never had
    :ivar columns: the projection, in contract order; required
    """

    model_config = ConfigDict(extra="forbid")

    anchor: ArtifactRef | None = None
    relations: list[RelationRef] = Field(default_factory=list)
    grain: list[str] = Field(min_length=1)
    measures: list[Measure] = Field(default_factory=list)
    columns: list[ProvenanceColumn] = Field(min_length=1)

    @model_validator(mode="after")
    def _projection_is_coherent(self) -> Self:
        """reject a repeated column name, grain key, or relation alias.

        :returns: validated spec
        :rtype: ProvenanceSpec
        :raises ValueError: a column name or grain key repeats, or a grain
            key is blank
        """
        names = [column.name for column in self.columns]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"provenance projects {duplicates} twice; the load binds positionally against the "
                "pinned contract, so a repeated name means two values racing for one destination"
            )
        if any(not key.strip() for key in self.grain):
            raise ValueError("grain carries a blank key")
        if len(self.grain) != len(set(self.grain)):
            raise ValueError(f"grain repeats a key: {self.grain!r}")
        validate_relation_aliases(self.relations)
        return self

    @property
    def column_names(self) -> tuple[str, ...]:
        """projected column names, in authored order.

        :returns: column names
        :rtype: tuple[str, ...]
        """
        return tuple(column.name for column in self.columns)

    @property
    def references(self) -> tuple[Reference, ...]:
        """references reachable from this body's own scope.

        :returns: references in column-then-relation order
        :rtype: tuple[Reference, ...]
        """
        collected: tuple[Reference, ...] = ()
        for column in self.columns:
            collected = collected + column.references
        for relation in self.relations:
            collected = collected + relation.references
        return collected

    def verify_columns_against(self, contract: Sequence[str]) -> None:
        """reject a projection that disagrees with the pinned contract.

        Positional, because every provenance body is emitted with no
        column list and the load therefore binds by position. A set
        comparison would pass a body whose columns are all present in the
        wrong order, and that body loads job titles into the employer
        column with no error at all.

        :param contract: pinned contract columns, in contract order
        :ptype contract: collections.abc.Sequence[str]
        :returns: None
        :rtype: None
        :raises ProvenanceContractViolation: a column is missing, extra,
            or in the wrong position
        """
        projected = self.column_names
        pinned = tuple(contract)
        missing = [name for name in pinned if name not in projected]
        extra = [name for name in projected if name not in pinned]
        if missing or extra:
            raise ProvenanceContractViolation(
                f"provenance projection does not match the pinned contract: missing {missing}, "
                f"extra {extra}. the contract is pinned once in the binding (D9a) and every body "
                "is verified against it; the template's own linkedin_industries / linkedin_industry "
                "divergence fails here correctly, and is recorded rather than excused"
            )
        offending = next(
            (
                (index, left, right)
                for index, (left, right) in enumerate(zip(projected, pinned, strict=True))
                if left != right
            ),
            None,
        )
        if offending is not None:
            index, projected_name, pinned_name = offending
            raise ProvenanceContractViolation(
                f"provenance projects {projected_name!r} at position {index + 1} where the pinned "
                f"contract carries {pinned_name!r}. the INSERT carries no column list, so the load "
                "binds positionally and a transposition writes each value into the wrong column"
            )
