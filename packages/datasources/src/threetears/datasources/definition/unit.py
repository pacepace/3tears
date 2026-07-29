"""units, resolutions, and qualification.

A unit is a SET of resolutions, not one. ``uhg_opinion_elites`` declares
``journalists_health_policy`` twice with different sources, joins, and
predicates, and the emitted SQL renders BOTH bodies as separate
``INSERT``s under one label -- while the prototype's metadata dict keeps
only the last. So "last wins" is true for metadata and false for data, and
which one an operator observes depends on whether they read an xtab or the
table. Its xtabs already under-report those units.

Two consequences, and they are not alternatives:

- :attr:`Unit.resolutions` is a LIST. There is deliberately NO
  merge-duplicates convenience: the corpus's duplicate declarations
  become two entries, ``parity-task-01`` will show the count difference,
  and that difference is a fix rather than a regression.
- a duplicate authored ``Unit.name`` is a PARSE ERROR
  (:func:`validate_unique_unit_names`), not a silent overwrite. One name
  meaning two different things depending on which layer you read is not
  something a migration should resolve silently.

``Unit.name`` is logical and free-form. Twenty-five authored names carry a
space or a ``>``; ``fec_contribution>2000`` ships today as the Redshift
identifier ``"fec_contribution>2000"``. The physical identifier is
DERIVED from the name by a pure function (D8a) and recorded on the run,
so this model carries no identifier field at all -- a stored copy is a
second source of truth that drifts.

Qualification is authored per unit but declarable over a unit SET. One
audience applies six candidate-count tiers and a two-branch age window,
each scoped to a group of units; flattening that to per-unit values
reproduces the result and destroys the authored intent, so the grouping is
preserved and may carry the authored label the group was known by.

The stage guards are the load-bearing part. ``source.*`` and ``bridge.*``
are refused in a qualification predicate because they name pre-aggregate
rows that no longer exist there, and ``resolved.*`` and ``measure.*`` are
refused in a resolution predicate because they name rows that do not
exist yet. Both mis-binds are otherwise silent and both change the
audience.

``dsm-task-01d`` closed the seams ``dsm-task-01a`` left here.
:attr:`Resolution.source` is a
:data:`~threetears.datasources.definition.source.SourceRef` rather than a
name, so a resolution can read a fact table, a raw body, a literal id
set, or another artifact; and :attr:`Unit.provenance` /
:attr:`Resolution.provenance` carry the per-person rationale, which is
authored per RESOLUTION because the corpus's nine hand-written bodies are
not recoverable from the unit query they belong to.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.bridge import BridgeRef
from threetears.datasources.definition.exclusion import ExclusionSpec
from threetears.datasources.definition.expression import Predicate
from threetears.datasources.definition.measure import (
    Measure,
    validate_having_measures,
    validate_unique_measure_names,
)
from threetears.datasources.definition.namespace import BindingStage, reject_unbindable
from threetears.datasources.definition.provenance import ProvenanceSpec
from threetears.datasources.definition.relation import RelationRef, validate_relation_aliases
from threetears.datasources.definition.source import SourceRef

__all__ = [
    "DuplicateUnitName",
    "Qualification",
    "Resolution",
    "Unit",
    "UnqualifiedUnits",
    "units_without_qualification",
    "validate_qualification_coverage",
    "validate_unique_unit_names",
]


class DuplicateUnitName(ValueError):
    """raised when two units, or a unit and a label, claim one name.

    The prototype resolves this three different ways at three layers --
    last-wins in the metadata dict, union in the emitted SQL, and a
    duplicate-column failure in the wide pivot -- so the name genuinely
    means two things at once. Refused at parse instead.
    """


class UnqualifiedUnits(ValueError):
    """raised when a declared unit is named by no qualification arm.

    ``department_of_commerce`` is emitted as a unit and appears in no arm
    of its audience's qualification chain and in neither rollup list, so
    every one of its rows is dropped. Not filtered, not errored. The safe
    default is a validation error.
    """


class Resolution(BaseModel):
    """one way a unit resolves entities, out of the set that unit carries.

    :ivar source: where this resolution's rows come from, or ``None`` for
        a bridge-only resolution -- which is legal and shipped, since a
        unit declared with no ``facts_table`` and no filters resolves
        through the bridge alone
    :ivar bridge: bridge this resolution matches through, or ``None`` for
        a literal source
    :ivar relations: per-resolution FROM extensions, by authored alias
    :ivar measures: aggregates computed inside this resolution
    :ivar predicate: filter over ``source.*`` / ``bridge.*`` / ``entity.*``
        / ``rel.<alias>.*`` / ``param.*``
    :ivar having: post-aggregate filter, the only place ``measure.<name>``
        binds
    :ivar provenance: per-person rationale this resolution emits;
        authored here rather than on the unit because the corpus's bodies
        are per-resolution and not recoverable from the unit query
    """

    model_config = ConfigDict(extra="forbid")

    source: SourceRef | None = None
    bridge: BridgeRef | None = None
    relations: list[RelationRef] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    predicate: Predicate | None = None
    having: Predicate | None = None
    provenance: ProvenanceSpec | None = None

    @model_validator(mode="after")
    def _predicates_bind_legal_namespaces(self) -> Self:
        """refuse ``resolved.*`` and ``measure.*`` outside where they exist.

        also refuses a duplicate relation alias, a predicate naming an
        alias no relation declares, and a ``having`` naming a measure this
        resolution does not compute. all three are authoring errors that
        the warehouse would otherwise report as an unresolved identifier
        at build time, after the schema is already half built.

        :returns: validated resolution
        :rtype: Resolution
        :raises ValueError: a predicate binds a namespace the stage does
            not carry, an alias is duplicated or undeclared, or a having
            names an undeclared measure
        """
        if self.predicate is not None:
            reject_unbindable(self.predicate.references, BindingStage.RESOLUTION, "Resolution.predicate")
        if self.having is not None:
            reject_unbindable(self.having.references, BindingStage.HAVING, "Resolution.having")
        validate_relation_aliases(self.relations)
        validate_unique_measure_names(self.measures)
        validate_having_measures(self.having, self.measures)
        return self


class Qualification(BaseModel):
    """filter applied to the materialized long row, scoped to a unit set.

    :ivar name: authored label for the unit group this arm was known by,
        or ``None`` when the group is anonymous, as every corpus group is
    :ivar relations: joins this stage adds, including the unconditional
        entity join that defines the qualified set
    :ivar predicate: filter over ``resolved.*`` / ``entity.*`` /
        ``rel.<alias>.*`` / ``param.*``
    :ivar applies_to: unit names this arm covers; ``None`` means the
        owning unit only
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    relations: list[RelationRef] = Field(default_factory=list)
    predicate: Predicate | None = None
    applies_to: list[str] | None = None

    @model_validator(mode="after")
    def _scope_and_namespaces_are_legal(self) -> Self:
        """refuse an empty or repeating scope, and pre-aggregate namespaces.

        :returns: validated qualification
        :rtype: Qualification
        :raises ValueError: applies_to is empty or repeats a unit, or the
            predicate binds ``source.*``, ``bridge.*``, or ``measure.*``
        """
        if self.applies_to is not None:
            if not self.applies_to:
                raise ValueError("applies_to is empty; omit it to scope this arm to the owning unit")
            if len(self.applies_to) != len(set(self.applies_to)):
                raise ValueError("applies_to names a unit twice")
        if self.predicate is not None:
            reject_unbindable(self.predicate.references, BindingStage.QUALIFICATION, "Qualification.predicate")
        validate_relation_aliases(self.relations)
        return self


class Unit(BaseModel):
    """one named member of a dataset definition, resolved several ways.

    :ivar name: logical name; free-form, and the physical identifier is
        derived from it rather than authored beside it
    :ivar emits: labels this unit projects, or ``None`` to project
        :attr:`name` alone
    :ivar resolutions: ways this unit resolves entities, at least one; a
        unit declared twice becomes two entries rather than one
    :ivar qualify: per-unit qualification; definition-level arms scoped by
        ``applies_to`` sit beside it on the definition
    :ivar exclude: residual subtraction this unit applies, or ``None``.
        Its stage, key, and level are required with no defaults, because
        each default silently reproduces a distinct hole: subtracting the
        pre-qualification set strands a person in neither unit,
        entity-keyed and record-keyed exclusion give different audiences,
        and pre- and post-aggregate exclusion differ because the anti-join
        sits before the group-by that computes the quality measure
    :ivar provenance: unit-level rationale, where every resolution shares
        one body; a resolution declaring its own overrides nothing and is
        emitted beside this
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    emits: list[str] | None = None
    resolutions: list[Resolution] = Field(min_length=1)
    exclude: ExclusionSpec | None = None
    qualify: Qualification | None = None
    provenance: ProvenanceSpec | None = None

    @model_validator(mode="after")
    def _labels_are_distinct(self) -> Self:
        """reject an empty or repeating label list.

        :returns: validated unit
        :rtype: Unit
        :raises ValueError: emits is empty or repeats a label
        """
        if self.emits is not None:
            if not self.emits:
                raise ValueError(f"{self.name}: emits is empty; omit it to project the unit name alone")
            if len(self.emits) != len(set(self.emits)):
                raise ValueError(f"{self.name}: emits repeats a label")
        return self

    @property
    def emitted_labels(self) -> tuple[str, ...]:
        """labels this unit projects.

        :returns: authored labels, or the unit name alone
        :rtype: tuple[str, ...]
        """
        return tuple(self.emits) if self.emits is not None else (self.name,)


def validate_unique_unit_names(units: list[Unit]) -> list[Unit]:
    """reject a definition where two units claim one label.

    Checked over EMITTED labels rather than over :attr:`Unit.name` alone,
    because a unit that emits two labels can collide with another unit's
    name, and the wide pivot then emits one quoted column twice and the
    CTAS fails at execution instead of at authoring.

    :param units: declared units, in authored order
    :ptype units: list[Unit]
    :returns: the same list, unchanged
    :rtype: list[Unit]
    :raises DuplicateUnitName: a label is claimed twice
    """
    seen: set[str] = set()
    for unit in units:
        for label in unit.emitted_labels:
            if label in seen:
                raise DuplicateUnitName(
                    f"unit label {label!r} is declared twice; a duplicate name means the union "
                    "of two in SQL and the second one in every reporting path, so it is refused "
                    "at parse. author one unit with several entries in resolutions instead"
                )
            seen.add(label)
    return units


def units_without_qualification(unit_names: Sequence[str], qualifications: Sequence[Qualification]) -> tuple[str, ...]:
    """unit names no qualification arm covers.

    An arm with ``applies_to=None`` at definition level covers every unit,
    so a non-empty result means those units are silently dropped by the
    qualification stage rather than filtered by it.

    :param unit_names: every declared unit name
    :ptype unit_names: collections.abc.Sequence[str]
    :param qualifications: definition-level qualification arms
    :ptype qualifications: collections.abc.Sequence[Qualification]
    :returns: uncovered unit names, in declaration order
    :rtype: tuple[str, ...]
    """
    uncovered: tuple[str, ...] = ()
    if qualifications and not any(arm.applies_to is None for arm in qualifications):
        covered = {name for arm in qualifications for name in arm.applies_to or ()}
        uncovered = tuple(name for name in unit_names if name not in covered)
    return uncovered


def validate_qualification_coverage(unit_names: Sequence[str], qualifications: Sequence[Qualification]) -> None:
    """reject a definition whose qualification silently annihilates a unit.

    A definition carrying no qualification at all is legal --
    ``amazon_tech_audience`` has no settings file and every unit survives.
    What is refused is a qualification chain that names some units and not
    others, because the unnamed ones lose every row with no error.

    :param unit_names: every declared unit name
    :ptype unit_names: collections.abc.Sequence[str]
    :param qualifications: definition-level qualification arms
    :ptype qualifications: collections.abc.Sequence[Qualification]
    :returns: None
    :rtype: None
    :raises UnqualifiedUnits: a declared unit is named by no arm
    """
    uncovered = units_without_qualification(unit_names, qualifications)
    if uncovered:
        raise UnqualifiedUnits(
            f"units {list(uncovered)!r} are named by no qualification arm, so the qualified set "
            "would drop every one of their rows with no error. name them in an arm -- an arm "
            "with no predicate is how the corpus admits a unit unconditionally"
        )
