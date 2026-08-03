"""declared measures: an aggregate, the grain it is computed at, and where
the filter over it lands.

The grain decides the answer, and it is per ``(entity, source record)``
rather than per entity. A donor whose contributions arrive under two
source records is summed ONCE PER RECORD, so a threshold the combined
total would clear can fail in every group. Eleven production units are
defined by that, and the failure is silent: getting the grain wrong
changes which entities qualify without changing anything visible in the
definition. So :attr:`Measure.grain` is required and has no default. It
is never inherited from whatever the surrounding plan happens to group
by, and the same named measure genuinely carries two grains in the corpus
-- ``(voterbase_id, list_id)`` for standard units and
``(unit, voterbase_id, record_year, list_id)`` for LinkedIn ones.

:attr:`Measure.filter_position` is the second half of the same trap. The
prototype renders ``custom_aggregate_filters`` as an outer ``WHERE`` over
the aggregated subquery, NOT a ``HAVING``, and the two differ whenever
the measure's grain differs from the unit's output grain. The
qualification stage is the worked example: it re-aggregates
``MIN(candidate_count)`` at ``(voterbase_id, list_id, unit)`` while the
threshold is evaluated in the ``WHERE`` against the pre-re-aggregate
value, so the qualified artifact's value is a MIN of a MIN at a different
grain than the one that was tested.

``measure.<name>`` binds in ``having`` and nowhere else, which
:mod:`~threetears.datasources.definition.namespace` already enforces by
stage. :func:`validate_having_measures` closes the other half: a
``having`` naming a measure nobody declared.

The grain is a list of plain column names rather than namespaced
references because it is the emitted aggregate's GROUP BY key, expressed
in the resolution's own output columns -- the names the long artifact
carries -- not a reference resolved against a FROM clause.

:attr:`Measure.result_type` is the third trap and the one
``parity-task-03`` found. An aggregate's type is not the type its operands
share: ``COUNT`` over anything is an integer, a ``1``/``0`` ``CASE``
classifier over ``job_title`` is an integer, and a ``SUM`` over a cast is
the cast's type. The corpus's healthcare classifier is
``MAX(CASE WHEN job_title LIKE ... THEN 1 ELSE 0 END)`` filtered by
``= 1``, and inferring varchar from its references rejects a correct
definition outright.

It is OPTIONAL, and that is a decision rather than a shortcut.
:attr:`expression` is an
:class:`~threetears.datasources.definition.expression.ArithmeticExpression`
whose text this package deliberately does not parse, so the model cannot
tell a family-changing aggregate from a family-preserving one and cannot
require the field only where it matters. Requiring it everywhere would put
a restated type on some two hundred committed measures whose operand types
already give the right answer, and a copy-pasted declaration is WORSE than
an inference, because a declaration overrides and nothing cross-checks it.
Declared, it is authoritative; omitted, the consumer infers and reports
the comparison as unchecked when it cannot. The five families come from
:class:`~threetears.datasources.definition.parameters.ParameterType`
rather than a second enum, so the type checker keeps one family map.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Self, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from threetears.datasources.definition.expression import ArithmeticExpression, Predicate
from threetears.datasources.definition.namespace import Namespace
from threetears.datasources.definition.parameters import ParameterType

__all__ = [
    "DuplicateMeasureName",
    "FilterPosition",
    "Measure",
    "MeasureExpression",
    "MeasureScope",
    "UndeclaredMeasure",
    "validate_having_measures",
    "validate_unique_measure_names",
]


class DuplicateMeasureName(ValueError):
    """raised when two measures of one resolution claim one name.

    ``measure.<name>`` is a namespace key, so a repeat makes every
    ``having`` that names it ambiguous. The corpus already carries the
    inverse defect -- one computation authored under two names,
    ``contribution_sum`` and ``sum_of_contributions`` -- which is legal
    and stays authored.
    """


class UndeclaredMeasure(ValueError):
    """raised when a ``having`` names a measure nobody declared.

    In the prototype this renders a reference to a select-list alias that
    may or may not exist, and Redshift's permissiveness decides whether it
    is an error or a wrong answer.
    """


class MeasureScope(StrEnum):
    """stage a measure is computed at.

    :cvar RESOLUTION: inner aggregate of one resolution, before the long
        artifact exists
    :cvar DELIVERY: aggregate over already-materialized rows, e.g. the
        qualified stage's re-aggregation and the wide pivot's
    """

    RESOLUTION = "resolution"
    DELIVERY = "delivery"


class FilterPosition(StrEnum):
    """where a filter over one measure is emitted.

    The two differ whenever the measure's grain differs from the stage's
    output grain, which in the corpus it routinely does.

    :cvar HAVING: post-aggregate filter inside the aggregating query
    :cvar OUTER_WHERE: filter in a query wrapping the aggregate, which is
        what the prototype emits for ``custom_aggregate_filters``
    """

    HAVING = "having"
    OUTER_WHERE = "outer_where"


def _coerce_measure_expression(value: object) -> object:
    """map authored aggregate text onto the arithmetic expression shape.

    A measure expression is always an aggregate over other columns, so a
    bare string is arithmetic text rather than the namespaced reference a
    bare string means everywhere else in the model.

    :param value: raw authored expression
    :ptype value: object
    :returns: value in a shape :class:`ArithmeticExpression` can validate
    :rtype: object
    """
    return {"arith": value} if isinstance(value, str) else value


MeasureExpression: TypeAlias = Annotated[ArithmeticExpression, BeforeValidator(_coerce_measure_expression)]
"""aggregate expression field that also accepts the bare authored string."""


class Measure(BaseModel):
    """one named aggregate a stage computes.

    :ivar name: measure name; ``measure.<name>`` binds it in ``having``
    :ivar expression: aggregate text, carried verbatim and scanned for
        namespaced references
    :ivar grain: group key this aggregate is computed at, in authored
        order; required and NEVER inherited
    :ivar scope: stage the measure is computed at
    :ivar filter_position: where a filter over it is emitted
    :ivar result_type: family this aggregate YIELDS, which is not the
        family its operands share; ``None`` leaves the consumer to infer
        one from the operands and report the comparison as unchecked when
        it cannot
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    expression: MeasureExpression
    grain: list[str] = Field(min_length=1)
    scope: MeasureScope
    filter_position: FilterPosition = FilterPosition.HAVING
    result_type: ParameterType | None = None

    @model_validator(mode="after")
    def _declaration_is_coherent(self) -> Self:
        """reject a non-identifier name and a malformed grain.

        :returns: validated measure
        :rtype: Measure
        :raises ValueError: name is not an identifier, or a grain key is
            not an identifier or repeats
        """
        if not self.name.isidentifier():
            raise ValueError(f"measure name {self.name!r} is not an identifier; measure.<name> binds it")
        for key in self.grain:
            if not key.isidentifier():
                raise ValueError(
                    f"{self.name}: grain key {key!r} is not an identifier. the grain is the emitted "
                    "GROUP BY over this stage's own output columns, not a namespaced reference"
                )
        if len(self.grain) != len(set(self.grain)):
            raise ValueError(f"{self.name}: grain repeats a key")
        return self


def validate_unique_measure_names(measures: Sequence[Measure]) -> Sequence[Measure]:
    """reject two measures of one stage claiming one name.

    :param measures: declared measures, in authored order
    :ptype measures: collections.abc.Sequence[Measure]
    :returns: the same sequence, unchanged
    :rtype: collections.abc.Sequence[Measure]
    :raises DuplicateMeasureName: a name is claimed twice
    """
    seen: set[str] = set()
    for measure in measures:
        if measure.name in seen:
            raise DuplicateMeasureName(
                f"measure name {measure.name!r} is declared twice; measure.<name> is a namespace "
                "key, so a repeat makes every having that names it ambiguous"
            )
        seen.add(measure.name)
    return measures


def validate_having_measures(having: Predicate | None, measures: Sequence[Measure]) -> None:
    """reject a post-aggregate filter naming an undeclared measure.

    :param having: post-aggregate filter, or ``None`` when the stage has
        none, which is legal -- one corpus unit computes a measure and
        declares no filter over it
    :ptype having: Predicate | None
    :param measures: measures declared by the same stage
    :ptype measures: collections.abc.Sequence[Measure]
    :returns: None
    :rtype: None
    :raises UndeclaredMeasure: a ``measure.<name>`` reference names no
        declared measure
    """
    if having is not None:
        declared = {measure.name for measure in measures}
        offending = next(
            (
                reference
                for reference in having.references
                if reference.namespace is Namespace.MEASURE and reference.name not in declared
            ),
            None,
        )
        if offending is not None:
            raise UndeclaredMeasure(
                f"having references {offending.ref!r}, which names no declared measure "
                f"({sorted(declared) or 'none declared'}). measure.<name> binds a measure of this "
                "stage, and the grain it is computed at decides which entities qualify"
            )
