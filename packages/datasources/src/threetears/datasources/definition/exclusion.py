"""the residual operator: what a unit subtracts, at which stage, keyed how.

Six production units anti-join their own audience's partially-built
table, so their result depends on YAML ordering AND on directory-listing
order for custom units. This module converts that genuine nondeterminism
into a declared dependency.

Four things decide a residual unit's membership, and per D7a all four are
authored with NO defaults. Each default silently reproduces one of the
prototype's holes and makes it invisible in the diff:

- **stage.** The subtracted set is the RESOLVED rows or the QUALIFIED
  ones. The prototype subtracts the pre-qualification set, so a person
  excluded from a later unit can then be dropped from the earlier one by
  qualification and land in NEITHER. Both stages are exercised in the
  committed corpus -- the residual units anti-join the resolved long
  table, while ``exclude_existing`` in the expansion template anti-joins
  the qualified set -- so there is no stage a default could pick that is
  right for both.
- **key.** The prototype anti-joins on the entity alone while the long
  grain is ``(unit, source record, entity)``, so an entity qualifying
  under a later unit via a DIFFERENT source record is still excluded.
  Entity-keyed and record-keyed exclusion give different audiences.
- **level.** The anti-join sits in ``WHERE``, before the group-by that
  computes ``MIN(quality)`` and any measure, so pre- and post-aggregate
  exclusion differ.
- **extent.** "Not already in the audience" is :attr:`ExclusionSpec.all_prior`
  rather than a hand-enumerated list of forty-odd unit names re-enumerated
  whenever a unit is added.

``all_prior`` EXPANDS TO EXPLICIT EDGES AT PARSE TIME
(:meth:`ExclusionSpec.expanded_against`, :func:`expand_all_prior`). Left
as a runtime notion, "prior" is ambiguous between authored list order and
a topological order free to reorder, and the two give different audiences
with no error -- which is the nondeterminism this module exists to
remove. The corpus's residual extent is order-dependent and the order is
unsorted ``os.listdir()``; expansion at parse is what puts the chosen
order in the definition and in the content hash. D7b records the chosen
direction: in ``amazon_audience_2025``,
``omnibus_other_sources_non_overlap`` is authored FIRST, so it does not
subtract ``omnibus_other_sources_overlap`` while ``overlap`` does subtract
it. This module does not choose that direction; it expresses whichever
the authored unit order carries.

A NAMING NOTE, because the field diverges from the design's spelling.
Design section 7 and D7a write the second authored property as ``key``.
This package spells it :attr:`ExclusionSpec.key_columns`, because
``tests/enforcement/test_secrets_typed.py`` reserves a bare ``key`` (and
any ``*_key``) as a credential name that must be typed ``SecretStr``.
The anti-join key is a LIST OF COLUMN NAMES, so the enforcement rule and
this field cannot both keep the short spelling; the enforcement rule is
the live guard and wins. ``dsm-task-01b``'s ``SetExpr.dedup_key`` is
renamed for the same reason.

:class:`ArtifactHandle` is a deliberately NARROW seam.
``dsm-task-01d``'s ``ArtifactRef`` replaces it and adds ``scope``,
``datasource``, ``run`` (:class:`UpstreamPin`), ``artifact``,
``schema_expr``, and ``projection``. Only the three discriminating cases
the corpus's exclusion and rollup rows need are carried here, so nothing
in this shard pre-empts that model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ArtifactHandle",
    "ArtifactStage",
    "ExclusionLevel",
    "ExclusionSpec",
    "UnexpandedExclusion",
    "expand_all_prior",
    "reject_unexpanded_exclusions",
]


class UnexpandedExclusion(ValueError):
    """raised when an ``all_prior`` exclusion reaches the compiler unexpanded.

    An unexpanded exclusion carries no edges, so the compiler would have
    to decide what "prior" means -- and authored order and topological
    order give different audiences with no error. Refused at the
    boundary instead.
    """


class ArtifactStage(StrEnum):
    """stage of a set of rows an exclusion or a reference names.

    :cvar RESOLVED: materialized long rows, before qualification
    :cvar QUALIFIED: rows surviving the qualification stage
    """

    RESOLVED = "resolved"
    QUALIFIED = "qualified"


class ExclusionLevel(StrEnum):
    """where the anti-join sits relative to the group-by.

    :cvar PRE_AGGREGATE: in ``WHERE``, before the group-by computing
        ``MIN(quality)``; a row removed here never reaches the aggregate
    :cvar POST_AGGREGATE: after the aggregate, over already-grouped rows
    """

    PRE_AGGREGATE = "pre_aggregate"
    POST_AGGREGATE = "post_aggregate"


class ArtifactHandle(BaseModel):
    """narrow reference to one set of rows, by exactly one of three kinds.

    Replaced by ``dsm-task-01d``'s ``ArtifactRef``, which adds ``scope``,
    ``datasource``, ``run``, ``artifact``, ``schema_expr``, and
    ``projection``. The three kinds here are the ones the corpus's
    exclusion and rollup rows exercise: a unit of THIS definition at a
    named stage, another definition's published output, and an external
    governed relation.

    :ivar unit: unit of this definition
    :ivar stage: which stage of that unit; required with :attr:`unit` and
        forbidden without it, because the resolved and qualified sets of
        one unit are different sets
    :ivar dataset: another definition, by name
    :ivar table: external relation, by its authored name
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: str | None = Field(default=None, min_length=1)
    stage: ArtifactStage | None = None
    dataset: str | None = Field(default=None, min_length=1)
    table: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _names_exactly_one_kind(self) -> Self:
        """require exactly one kind, and a stage for a unit alone.

        :returns: validated handle
        :rtype: ArtifactHandle
        :raises ValueError: no kind or more than one kind is named, a
            unit carries no stage, or a non-unit carries one
        """
        named = [name for name in ("unit", "dataset", "table") if getattr(self, name) is not None]
        if len(named) != 1:
            raise ValueError(f"an artifact handle names exactly one of unit / dataset / table; got {named or 'none'}")
        if self.unit is not None and self.stage is None:
            raise ValueError(
                f"unit {self.unit!r} carries no stage; the resolved and qualified sets of one unit "
                "are different sets, and subtracting the wrong one changes the audience silently"
            )
        if self.unit is None and self.stage is not None:
            raise ValueError(f"stage {self.stage.value!r} applies to a unit of this definition only")
        return self


_REQUIRED_TRIPLE: dict[str, str] = {
    "stage": (
        "subtracting the pre-qualification set lets a person be excluded from a later unit and "
        "then dropped from the earlier one by qualification, landing in neither audience"
    ),
    "key_columns": (
        "entity-keyed and record-keyed exclusion give different audiences: the long grain is "
        "(unit, source record, entity), so an entity qualifying under a later unit via a "
        "different source record is still excluded"
    ),
    "level": (
        "pre- and post-aggregate exclusion differ because the anti-join sits before the group-by "
        "that computes MIN(quality), so the audience differs by which rows reached the aggregate"
    ),
}


class ExclusionSpec(BaseModel):
    """what a unit subtracts, at which stage, keyed how, at which level.

    :attr:`stage`, :attr:`key_columns`, and :attr:`level` are REQUIRED
    with no defaults. This is unhelpful when authoring, and that is the
    point: each one silently changes the audience and the prototype
    picked one of each by accident.

    :ivar subtrahends: sets subtracted, in authored order; the explicit
        edges ``all_prior`` expands to land here
    :ivar all_prior: "not already in the audience"; expands to explicit
        edges at parse time and is kept on the model for round-tripping
    :ivar key_columns: columns identifying a match between minuend and
        subtrahend; the design spells this ``key`` and the module
        docstring records why the short spelling is unavailable
    :ivar level: whether the anti-join runs before or after the aggregate
    :ivar stage: which stage of the subtrahends is subtracted
    """

    model_config = ConfigDict(extra="forbid")

    subtrahends: list[ArtifactHandle] = Field(default_factory=list)
    all_prior: bool = False
    key_columns: list[str] = Field(min_length=1)
    level: ExclusionLevel
    stage: ArtifactStage

    @model_validator(mode="before")
    @classmethod
    def _residual_triple_is_authored(cls, value: object) -> object:
        """reject an omitted ``stage``, ``key_columns``, or ``level``, naming the consequence.

        Pydantic's own "field required" message states the omission and
        not what it costs, and the cost is a different audience with no
        diff, so the message is authored here instead.

        :param value: raw authored payload
        :ptype value: object
        :returns: value unchanged
        :rtype: object
        :raises ValueError: one of the three authored properties is absent
        """
        if isinstance(value, dict):
            for name, consequence in _REQUIRED_TRIPLE.items():
                if name not in value:
                    raise ValueError(f"exclusion carries no {name}, and there is no safe default: {consequence}")
        return value

    @model_validator(mode="after")
    def _extent_and_key_are_coherent(self) -> Self:
        """reject an exclusion that subtracts nothing or repeats itself.

        :returns: validated exclusion
        :rtype: ExclusionSpec
        :raises ValueError: no extent is declared, or the key columns or
            the subtrahend list repeat
        """
        if not self.all_prior and not self.subtrahends:
            raise ValueError(
                "exclusion subtracts nothing: declare all_prior, or name at least one subtrahend. "
                "an exclusion that subtracts nothing reads as a residual and behaves as a no-op"
            )
        if any(not column.strip() for column in self.key_columns):
            raise ValueError("key_columns carries a blank column name")
        if len(self.key_columns) != len(set(self.key_columns)):
            raise ValueError(f"key_columns names a column twice: {self.key_columns!r}")
        if len(self.subtrahends) != len(set(self.subtrahends)):
            raise ValueError("subtrahends names the same set twice")
        return self

    @property
    def is_expanded(self) -> bool:
        """whether every declared edge is explicit.

        :returns: False only when ``all_prior`` is set and no edge was
            expanded, which is the state the compiler refuses
        :rtype: bool
        """
        return not (self.all_prior and not self.subtrahends)

    def expanded_against(self, prior: Sequence[str]) -> ExclusionSpec:
        """expand ``all_prior`` into one explicit edge per prior unit.

        Authored subtrahends keep their authored positions and an already
        authored edge is not duplicated, so the expansion is additive and
        the diff shows exactly which edges "prior" resolved to.

        :param prior: unit names authored BEFORE the owning unit, in
            authored order
        :ptype prior: collections.abc.Sequence[str]
        :returns: a copy carrying explicit subtrahends, or this spec
            unchanged when ``all_prior`` is not set
        :rtype: ExclusionSpec
        """
        expanded = self
        if self.all_prior:
            already = {handle.unit for handle in self.subtrahends if handle.unit is not None}
            edges = [ArtifactHandle(unit=name, stage=self.stage) for name in prior if name not in already]
            expanded = self.model_copy(update={"subtrahends": [*self.subtrahends, *edges]})
        return expanded


def expand_all_prior(unit_names: Sequence[str], exclusions: Mapping[str, ExclusionSpec]) -> dict[str, ExclusionSpec]:
    """expand every ``all_prior`` exclusion against the authored unit order.

    The authored order IS what "prior" means. D7b's chosen direction for
    the two mutually-subtracting omnibus units is expressed by their
    positions in ``unit_names`` and by nothing else, which is why the
    expansion has to happen here rather than at plan time.

    :param unit_names: every declared unit name, in authored order
    :ptype unit_names: collections.abc.Sequence[str]
    :param exclusions: exclusion declared by each owning unit
    :ptype exclusions: collections.abc.Mapping[str, ExclusionSpec]
    :returns: expanded exclusions, keyed by owning unit
    :rtype: dict[str, ExclusionSpec]
    :raises ValueError: a unit name repeats, or an exclusion is owned by
        a unit the definition does not declare
    """
    if len(unit_names) != len(set(unit_names)):
        raise ValueError("unit_names declares a name twice; authored order cannot resolve a duplicate")
    position = {name: index for index, name in enumerate(unit_names)}
    undeclared = sorted(owner for owner in exclusions if owner not in position)
    if undeclared:
        raise ValueError(f"exclusions are owned by undeclared units {undeclared}")
    return {owner: spec.expanded_against(list(unit_names[: position[owner]])) for owner, spec in exclusions.items()}


def reject_unexpanded_exclusions(exclusions: Mapping[str, ExclusionSpec]) -> None:
    """reject any exclusion still carrying an unexpanded ``all_prior``.

    The guard the compiler calls. An unexpanded exclusion would make the
    compiler decide what "prior" means, and the two plausible answers --
    authored order and topological order -- give different audiences with
    no error.

    :param exclusions: exclusions, keyed by owning unit
    :ptype exclusions: collections.abc.Mapping[str, ExclusionSpec]
    :returns: None
    :rtype: None
    :raises UnexpandedExclusion: an exclusion carries ``all_prior`` and
        no edges
    """
    unexpanded = sorted(owner for owner, spec in exclusions.items() if not spec.is_expanded)
    if unexpanded:
        raise UnexpandedExclusion(
            f"units {unexpanded} carry all_prior with no expanded edges. call expand_all_prior "
            "against the authored unit order first: leaving it to the compiler makes 'prior' "
            "ambiguous between authored order and topological order, and the two give different "
            "audiences with no error"
        )
