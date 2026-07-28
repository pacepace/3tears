"""bridge references, and the PLURAL quality measures a bridge declares.

D16: **there is no ``long.quality`` column.** The long artifact carries
one column per declared quality measure, named ``quality_<measure_name>``,
and :class:`BridgeRef` declares each measure explicitly.

Plurality is not a generalisation, it is what the corpus already needs.
``uhg_healthcare_providers`` gates on ``mat.source_match_tier <= 12``
while the Amazon audiences gate on ``candidate_count``, and
``uhg_policymakers`` gates on ``candidate_count`` through a six-tier
threshold ladder that pairs a ceiling with a named unit set. The two
columns are different scales, and ``candidate_count`` is match quality
where LOWER is better. A single column cannot hold both, and a
``match_quality_skew`` detector keyed on ``candidate_count`` alone
silently mis-reports a tier-gated audience.

:attr:`QualityMeasure.direction` and
:attr:`QualityMeasure.threshold_semantics` stay separate on purpose.
``lower_is_better`` with ``at_most`` is ``candidate_count <= N``; a tier
could plausibly be ``lower_is_better`` with ``at_least``. Deriving one
from the other bakes in an assumption the corpus does not support.

:attr:`QualityMeasure.unmeasured_is_null` defaults to ``True`` because
the expansion artifact carries no quality at all: ``candidate_count`` is
``NULL::int`` and ``source_name`` is ``NULL`` in ``_relationship_union_``,
and the wide table then computes ``MIN(candidate_count)`` over a column
entirely NULL for expansion rows. ``dsh-task-06b`` renders an explicit
unmeasured band off that flag, and without it a band table reports only
the direct members' distribution -- which on an expansion-heavy audience
is a small minority of it.

:func:`union_quality_measures` is what an ``intersect`` across two
differently-bridged units emits: BOTH measure sets, with ``NULL`` on the
side that supplied none. It does not coerce, rescale, or pick one, and it
refuses two same-named measures whose semantics disagree rather than
silently choosing.

Seams named so they are not mistaken for omissions:
:attr:`BridgeRef.relation` carries the authored relation NAME as a
``str``, which ``dsm-task-01d`` extends with ``ArtifactRef``. The
relation layer's own gaps -- a payload field for a quality measure, and
``from_schema`` so a bridge can be re-pointed at a per-run schema -- are
``dsh-task-24``'s work on ``JoinPath`` and ``datasource_relations``. The
payload this module declares is what that layer must carry.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import Predicate

__all__ = [
    "BridgeRef",
    "ConflictingQualityMeasure",
    "QualityDirection",
    "QualityMeasure",
    "ThresholdSemantics",
    "union_quality_measures",
]

_LONG_COLUMN_PREFIX = "quality_"


class ConflictingQualityMeasure(ValueError):
    """raised when one measure name carries two different semantics.

    Two bridges may declare ``candidate_count`` off two different
    physical columns and still mean the same thing. They may NOT declare
    it with opposite polarity or opposite threshold semantics: the union
    would then emit one ``quality_candidate_count`` column holding two
    incomparable scales, which is exactly the collapse D16 forbids.
    """


class QualityDirection(StrEnum):
    """polarity of one quality measure.

    :cvar LOWER_IS_BETTER: smaller values are better matches, e.g.
        ``candidate_count``
    :cvar HIGHER_IS_BETTER: larger values are better matches
    """

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class ThresholdSemantics(StrEnum):
    """how a threshold over one quality measure reads.

    Separate from :class:`QualityDirection` because the corpus does not
    support deriving either from the other.

    :cvar AT_MOST: threshold is a ceiling, e.g. ``candidate_count <= 4``
    :cvar AT_LEAST: threshold is a floor
    """

    AT_MOST = "at_most"
    AT_LEAST = "at_least"


class QualityMeasure(BaseModel):
    """one match-quality measure a bridge supplies.

    :ivar name: measure name; the long artifact projects it as
        ``quality_<name>``
    :ivar column: bridge column this measure reads
    :ivar direction: whether lower or higher values are better matches
    :ivar threshold_semantics: whether a threshold over it is a ceiling
        or a floor
    :ivar unmeasured_is_null: whether rows that supply no measurement
        carry ``NULL`` rather than a value, which is what an explicit
        unmeasured band keys on
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    column: str = Field(min_length=1)
    direction: QualityDirection
    threshold_semantics: ThresholdSemantics
    unmeasured_is_null: bool = True

    @model_validator(mode="after")
    def _name_is_a_column_name(self) -> Self:
        """reject a name that cannot become a column identifier.

        :returns: validated measure
        :rtype: QualityMeasure
        :raises ValueError: name is not an identifier
        """
        if not self.name.isidentifier():
            raise ValueError(
                f"quality measure name {self.name!r} is not an identifier; it becomes the long "
                f"artifact's {_LONG_COLUMN_PREFIX}<name> column"
            )
        return self

    @property
    def long_column(self) -> str:
        """column this measure occupies in the long artifact.

        :returns: ``quality_<name>``
        :rtype: str
        """
        return f"{_LONG_COLUMN_PREFIX}{self.name}"

    @property
    def semantics(self) -> tuple[QualityDirection, ThresholdSemantics, bool]:
        """what two declarations of one name must agree on.

        The physical column deliberately does NOT participate: two
        bridges reading two differently-named columns can still supply one
        comparable measure, and the corpus does exactly that.

        :returns: direction, threshold semantics, and unmeasured handling
        :rtype: tuple[QualityDirection, ThresholdSemantics, bool]
        """
        return (self.direction, self.threshold_semantics, self.unmeasured_is_null)


class BridgeRef(BaseModel):
    """bridge relation a resolution matches entities through.

    :ivar relation: governed bridge relation name
    :ivar alias: name predicates address the bridge by, or ``None`` to
        use the ``bridge.*`` namespace alone
    :ivar quality_measures: match-quality measures this bridge supplies;
        empty for a bridge that measures nothing, which the expansion
        edge and every literal-entity source genuinely are
    :ivar on: join condition, or ``None`` to use the relation layer's
        declared join path
    """

    model_config = ConfigDict(extra="forbid")

    relation: str = Field(min_length=1)
    alias: str | None = Field(default=None, min_length=1)
    quality_measures: list[QualityMeasure] = Field(default_factory=list)
    on: Predicate | None = None

    @model_validator(mode="after")
    def _measures_are_distinct(self) -> Self:
        """reject a repeated measure name and a non-identifier alias.

        Two measures of one name on one bridge would collide in a single
        ``quality_<name>`` column, which is the collapse D16 exists to
        prevent.

        :returns: validated bridge
        :rtype: BridgeRef
        :raises ValueError: alias is not an identifier, or a measure name
            repeats
        """
        if self.alias is not None and not self.alias.isidentifier():
            raise ValueError(f"bridge alias {self.alias!r} is not an identifier")
        names = [measure.name for measure in self.quality_measures]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.relation}: declares the same quality measure name twice")
        return self

    @property
    def long_columns(self) -> tuple[str, ...]:
        """columns this bridge contributes to the long artifact.

        :returns: ``quality_<name>`` per declared measure, in authored order
        :rtype: tuple[str, ...]
        """
        return tuple(measure.long_column for measure in self.quality_measures)


def union_quality_measures(bridges: Sequence[BridgeRef]) -> tuple[QualityMeasure, ...]:
    """measures an artifact emits across several differently-bridged units.

    Both sets, in first-appearance order, with the side that supplied no
    measure contributing nothing and therefore carrying ``NULL`` in the
    other side's columns. Nothing is coerced, rescaled, or dropped.

    :param bridges: bridges of the units being combined, in authored order
    :ptype bridges: collections.abc.Sequence[BridgeRef]
    :returns: distinct measures, in first-appearance order
    :rtype: tuple[QualityMeasure, ...]
    :raises ConflictingQualityMeasure: one name is declared with two
        different sets of semantics
    """
    emitted: dict[str, QualityMeasure] = {}
    for bridge in bridges:
        for measure in bridge.quality_measures:
            existing = emitted.get(measure.name)
            if existing is None:
                emitted[measure.name] = measure
            elif existing.semantics != measure.semantics:
                raise ConflictingQualityMeasure(
                    f"quality measure {measure.name!r} is declared with two different semantics "
                    f"({existing.semantics} and {measure.semantics}); an intersect emits both "
                    "measure sets with NULL on the side that supplied none, and never coerces or "
                    "rescales one onto the other. rename one measure"
                )
    return tuple(emitted.values())
