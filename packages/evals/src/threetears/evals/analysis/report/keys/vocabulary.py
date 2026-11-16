"""The vocabulary a key's path is matched against: the slots (a fixed word, ids, a statistic, a number), a
family's shape, and the readers every family shares (an arm, an aspect, a reading, a contrast's scale, a criterion,
a task, a stratum).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

from threetears.evals.analysis.checks import Looks
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.keys.figure import Figure, PERCENTILE_NAMES, Scale, _SCALE_COUNTS
from threetears.evals.analysis.report.number_forms.kinds import Role
from threetears.evals.analysis.report.parts.words import (
    PLAN_LEVEL_TEMPLATE,
    PLAN_PAST_CLAUSE,
    PLAN_SIZED_CLAUSE,
    PLAN_UNCOUNTED_CLAUSE,
)
from threetears.evals.analysis.report.words import stratum_name, stratum_names
from threetears.evals.analysis.report_input import (
    SHARE_UNIT,
    Arm,
    Aspect,
    Contrast,
    ContrastFamily,
    Criterion,
    Judged,
    Measure,
    Reading,
    ReportInput,
    Stratum,
    Task,
)
from threetears.evals.analysis.report_margins import lead_aspect
from threetears.evals.analysis.significance import SeparationTest, plan_kept


# --- the vocabulary a path is matched against -------------------------------------------------------------------


@dataclass(frozen=True)
class _Word:
    """A fixed word in a path: a family keyword or a sub-field, as the path spells it."""

    spelling: str


@dataclass(frozen=True)
class _Ids:
    """A slot an id fills: its name in the grammar, and the ids the input offers given the slots matched before it."""

    name: str
    options: Callable[[ReportInput, tuple[str, ...]], Iterable[str]]


@dataclass(frozen=True)
class _Statistic:
    """A slot a statistic (or a field) from a fixed set fills, and its name in the grammar."""

    options: tuple[str, ...]
    name: str = "<statistic>"


@dataclass(frozen=True)
class _Number:
    """A slot a number the caller chose fills (a hypothetical line), written as digits with an optional sign and
    decimals; its name in the grammar."""

    name: str


_Slot = _Word | _Ids | _Statistic | _Number

#: A number as a path spells one in a :class:`_Number` slot.
_NUMBER_AT = re.compile(r"-?\d+(?:\.\d+)?")


def _number_options(path: str, at: int) -> list[str]:
    """The numbers a path can be read to spell at ``at``: its whole part, and with its decimals where it has them
    (``97.5`` before a dot may be ``97`` then ``5``, or ``97.5``)."""
    match = _NUMBER_AT.match(path, at)
    if match is None:
        return []
    whole, _dot, decimals = match.group(0).partition(".")
    return [whole, match.group(0)] if decimals else [whole]


#: Where a page's tables hold a family's figure, for its ledger (:attr:`Figure.ledger_as`): the measure, its kind and
#: the role, read from the input, the ids the path matched and the figure as resolved (its statistic, its set); None
#: where its set says it.
type _LedgerAs = Callable[[ReportInput, tuple[str, ...], Figure], tuple[str, str, Role] | None]


def _a_percentile_is_a_level(
    _report_input: ReportInput, _values: tuple[str, ...], figure: Figure
) -> tuple[str, str, Role] | None:
    """A percentile of a measure (a turn time's ``p95``) is a level of it, and money's its money, as a table that states
    the measure by its percentiles prints it; every family that reads a measure's statistics reads its percentiles so."""
    if figure.set_of is None or figure.statistic not in PERCENTILE_NAMES:
        return None
    return (*figure.set_of, "money" if figure.scale in _SCALE_COUNTS else "level")


#: The rate a family's figures in points are differences of, with its kind, from the input and the ids its path
#: matched (:attr:`_Family.rate`); None where the input names no measure it holds.
type _RateOf = Callable[[ReportInput, tuple[str, ...]], tuple[str, str] | None]


@dataclass(frozen=True)
class _Family:
    """One path family: its slots in order, what it reads, and how its matched ids and statistic become a figure.

    ``derived`` marks a family whose every figure is computed here; ``derived_statistics`` the statistics of an
    otherwise copied family that are (a slice's ``vs_control``, a floor's ``held_above``). ``example`` is a key on the
    grammar's made-up input, which the grammar renders. ``derived_note`` replaces the grammar's "descriptive, no
    interval" line for a derived family that is not descriptive: one whose figures are read off stated spreads.
    ``compares`` gives, for a family whose money figures are changes, the amounts a change compares (each arm's cost per
    1,000 for a cost gap, the measure's means for a contrast), whose form the change reads in; ``None`` from it, or no
    ``compares``, reads the family's own amounts (:func:`_money_amounts`). ``ledger_as`` says where the page's tables
    hold a figure of the family whose path names no measure they hold it under, or whose statistic names no role they
    hold it in (:attr:`Figure.ledger_as`); every family reads a percentile of its measure as a level of it. ``rate`` is
    the rate a family's figures in points are differences of where its path names no measure (an aspect's pass rate, a
    floor's own rate), whose set they read at; None for a family that names its measure, or reads none.
    """

    slots: tuple[_Slot, ...]
    what: str
    resolve: Callable[[ReportInput, tuple[str, ...], str], Figure]
    derived: bool = False
    derived_statistics: tuple[str, ...] = ()
    example: str = ""
    derived_note: str = ""
    compares: Callable[[ReportInput, tuple[str, ...]], Iterable[float] | None] | None = None
    ledger_as: _LedgerAs = _a_percentile_is_a_level
    rate: _RateOf | None = None

    def form(self) -> str:
        return ".".join(slot.spelling if isinstance(slot, _Word) else slot.name for slot in self.slots)

    def statistic(self) -> _Statistic | None:
        return next((slot for slot in self.slots if isinstance(slot, _Statistic)), None)


def _refuse(path: str, why: str) -> UnresolvableReference:
    return UnresolvableReference(f"figure key {{{path}}} {why}")


def _unstated(path: str, why: str) -> UnstatedFigure:
    """The refusal for a well-formed key whose figure the input leaves unstated, typed so a listing reads it as absent
    (:class:`~threetears.evals.analysis.errors.UnstatedFigure`)."""
    return UnstatedFigure(f"figure key {{{path}}} {why}")


def _arm(report_input: ReportInput, arm_id: str) -> Arm:
    return next(arm for arm in report_input.arms if arm.id == arm_id)


def _control(report_input: ReportInput) -> Arm | None:
    return next((arm for arm in report_input.arms if arm.role == "control"), None)


def _candidates(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.role == "challenger"]


def _asked(report_input: ReportInput) -> Aspect | None:
    """The one aspect the campaign asked about: the only aspect marked asked, else the only aspect; None otherwise."""
    asked = [aspect for aspect in report_input.aspects if aspect.asked]
    if len(asked) == 1:
        return asked[0]
    if not asked and len(report_input.aspects) == 1:
        return report_input.aspects[0]
    return None


def _asked_or_refuse(report_input: ReportInput, path: str) -> Aspect:
    aspect = _asked(report_input)
    if aspect is None:
        ids = ", ".join(f"`aspects.{a.id}.`" for a in report_input.aspects) or "none: the input holds no aspect"
        raise _refuse(path, f"reads the one asked aspect, and this input has no single one; name it: {ids}")
    return aspect


def on_lead_aspect(report_input: ReportInput, contrast: Contrast) -> bool:
    """Whether a contrast is read on the lead aspect, the one an arm's overall figures (the by-arm maps) belong to: it
    names that aspect, or none. A contrast on another aspect pairs another aspect's figures, never the overall ones.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param contrast: One of its contrasts.
    :ptype contrast: Contrast
    :return: Whether the contrast pairs the overall figures.
    :rtype: bool
    """
    return contrast.aspect is None or contrast.aspect == lead_aspect(report_input)


def _held_apart(
    path: str, what: str, aspects: Sequence[str | None], qualified: Callable[[str], str]
) -> UnresolvableReference:
    """The refusal of an unqualified key whose figure the input holds more than once: it names the key per aspect.

    :param path: The key refused.
    :param what: What it names, in words.
    :param aspects: The aspect each held copy sits on, None where one names none.
    :param qualified: The qualified key for one aspect.
    """
    named = [f"`{qualified(aspect)}`" for aspect in dict.fromkeys(aspects) if aspect is not None]
    choose = f"name its aspect: {', '.join(named)}" if named else "and no aspect tells them apart"
    return _refuse(path, f"names {what}, which the input holds {len(aspects)} times; {choose}")


def _reading(aspect: Aspect, arm_id: str) -> Reading:
    return next(reading for reading in aspect.readings if reading.arm == arm_id)


def _rate_scale(values: Iterable[float | None]) -> Scale:
    """A rate's scale read from its figures: shares when every one lies in 0 to 1, percents otherwise.

    The input copies each rate as the pipeline gave it, and a judged pass rate carries no unit, so its scale is read
    from the values themselves.
    """
    present = [value for value in values if value is not None]
    return "share" if all(0 <= value <= 1 for value in present) else "percent"


def _measure_scale(measure: Measure) -> tuple[Scale, str]:
    unit = measure.unit or ""
    if unit == "pct":
        return "percent", ""
    if unit == SHARE_UNIT:
        return "share", ""
    if unit == "usd":
        return "usd", ""
    if unit == "usd_per_1k":
        return "usd_per_1k", ""
    if unit == "usd_per_1m":
        return "usd_per_1m", ""
    if not unit and measure.grade in ("pass_fail", "class_label"):
        return "share", ""
    return "plain", unit


def _judged_scale(judged: Judged) -> Scale:
    if judged.scale == "pass_fail":
        return _rate_scale(stat.mean for stat in judged.per_arm.values())
    return "plain"


def _contrast_scale(report_input: ReportInput, measure: str, kind: str | None) -> tuple[Scale, str]:
    if measure in report_input.measures:
        return _measure_scale(report_input.measures[measure])
    if measure in report_input.judged:
        return _judged_scale(report_input.judged[measure]), ""
    if kind == "trial":
        return "share", ""
    return "plain", ""


def _text(path: str, value: str | None, what: str) -> Figure:
    if value is None or not value.strip():
        raise _unstated(path, f"names {what}, which the input does not state")
    return Figure(path=path, kind="text", value=value)


def _number(
    path: str,
    value: float | None,
    what: str,
    scale: Scale,
    *,
    unit: str = "",
    change: bool = False,
    whole: bool = False,
) -> Figure:
    if value is None:
        raise _unstated(path, f"names {what}, which the input does not state")
    return Figure(path=path, kind="number", value=float(value), scale=scale, unit=unit, change=change, whole=whole)


def _level(path: str, value: float | None, what: str) -> Figure:
    """An interval's confidence level, a share, read in percent as stated (:class:`Figure`'s ``level``)."""
    return replace(_number(path, value, what, "share"), level=True)


def joint_guarantee(alpha: float | None, size: int, *, corrected: bool = True) -> tuple[float, int] | None:
    """What a correcting family's intervals guarantee together, the one computation of it for every page: one minus
    its alpha (a share, ``0.95``) over the family's size, which the written page's keys and headings
    (:func:`joint_level`), the aspect page's headings and the code page's notes and scale lines state. None for a
    family that corrects nothing, one that states no alpha, and one of fewer than two comparisons, which adjusts
    nothing: its interval reads ``95%``, never "adjusted for 1 comparisons".

    :param alpha: The family-wise error rate its comparisons hold at; None for none stated.
    :ptype alpha: float | None
    :param size: How many comparisons it corrects together.
    :ptype size: int
    :param corrected: Whether it corrects them at all.
    :ptype corrected: bool
    :return: The joint level and the family's size, or None.
    :rtype: tuple[float, int] | None
    """
    if alpha is None or not corrected or size < 2:
        return None
    return (1.0 - float(alpha), size)


class UnkeptPlan(NamedTuple):
    """A plan of looks not shown to hold the level it is for (:func:`unkept_plan`).

    :ivar planned: The looks the plan allows.
    :ivar taken: The looks taken; None where they were not recorded.
    :ivar why: Why it is not shown: within the plan, a test a reading uses is not shown to keep it where a look's cases
        were sized from an earlier reading (``sized``); more looks were taken than planned (``past``); or the count of
        looks is a lower one, or was not recorded, so more may have been (``uncounted``).
    :ivar tests: The tests it is not shown on: for ``sized`` those
        :func:`~threetears.evals.analysis.significance.plan_kept` does not show, otherwise every test given.
    """

    planned: int
    taken: int | None
    why: Literal["sized", "past", "uncounted"]
    tests: tuple[SeparationTest, ...]


def unkept_plan(
    looks: Looks | None, tests: Iterable[SeparationTest] | None = None, *, planned: int | None = None
) -> UnkeptPlan | None:
    """Whether the level a plan of looks is for is shown to hold over the looks an aspect took, for every test its
    readings use, the one decision of it for every page that prints that level: shown where the looks taken are at
    most the plan and exactly counted (or, not recorded, the stored evidence changed once), and
    :func:`~threetears.evals.analysis.significance.plan_kept` holds for each test (each look's cases fixed before
    its data by the union bound, planner-sized extends only where the simulation shows it). It decides what a page
    prints, never how a reading was read: the statistical method is unchanged.

    No plan in force (none, or a plan of one look) is no plan's level, so nothing is returned; nor where ``looks`` is
    None, which an aspect carries only where its contrasts were read once or not at all.

    :param looks: The aspect's looks (``reading_scope.looks``, or the report input's ``Aspect.looks``).
    :ptype looks: Looks | None
    :param tests: The tests the readings whose level is printed use; None for every test the aspect's readings use
        (``Looks.tests``).
    :ptype tests: Iterable[SeparationTest] | None
    :param planned: The plan the readings were read under where it is not the aspect's own (a campaign's readings read
        under the largest plan among its aspects); None for the aspect's.
    :ptype planned: int | None
    :return: The plan and why its level is not shown; None where it is shown, or no plan is in force.
    :rtype: UnkeptPlan | None
    """
    plan = planned if planned is not None else looks.planned if looks is not None else None
    if looks is None or plan is None or plan <= 1:
        return None
    used = tuple(looks.tests if tests is None else sorted(set(tests)))
    taken = looks.taken
    if taken is not None and taken > plan:
        return UnkeptPlan(plan, taken, "past", used)
    if taken is None:
        # Looks not recorded: a single change of the stored evidence is one look; more could be any number.
        return UnkeptPlan(plan, taken, "uncounted", used) if (looks.evidence_changes or 0) > 1 else None
    if looks.lower_bound:
        return UnkeptPlan(plan, taken, "uncounted", used)
    unshown = tuple(test for test in used if not plan_kept(test, plan, taken))
    return UnkeptPlan(plan, taken, "sized", unshown) if unshown else None


def plan_condition(level: str, unkept: UnkeptPlan, readings: Sequence[str]) -> str:
    """The sentence a page states in place of a plan's level it cannot show held (:func:`unkept_plan`): the level, the
    plan it is for, the readings it is not shown for, and why, every word from
    :mod:`~threetears.evals.analysis.report.parts.words`. True of every reading it names: it says where the level holds
    and where it is not shown, never that it holds.

    :param level: The level as the page prints it (``95% (adjusted for 16 comparisons)``).
    :ptype level: str
    :param unkept: The plan, and why its level is not shown.
    :ptype unkept: UnkeptPlan
    :param readings: What the page calls the readings it is not shown for, at least one.
    :ptype readings: Sequence[str]
    :return: The sentence.
    :rtype: str
    """
    named = readings[0] if len(readings) == 1 else f"{', '.join(readings[:-1])} and {readings[-1]}"
    one = len(readings) == 1
    clause = {
        "sized": PLAN_SIZED_CLAUSE,
        "past": PLAN_PAST_CLAUSE,
        "uncounted": PLAN_UNCOUNTED_CLAUSE,
    }[unkept.why].format(
        readings=named, were="was" if one else "were", them="it" if one else "them", taken=unkept.taken
    )
    return PLAN_LEVEL_TEMPLATE.format(level=level, planned=unkept.planned, clause=clause)


def joint_level(family: ContrastFamily | None, looks: Looks | None) -> tuple[float, int] | None:
    """What a written page's contrast family's intervals guarantee together (:func:`joint_guarantee`), which a figure
    in the package's words holds (:attr:`Figure.joint`) and every heading states
    (:func:`~threetears.evals.analysis.report.number_forms.intervals.interval_level`). None for no family, one that
    corrects nothing (no correction, or ``none``), every family :func:`joint_guarantee` reads none for, and one whose
    aspect's plan of looks is not shown to hold that level (:func:`unkept_plan`, on every test the aspect's readings
    use, the input carrying no test per contrast): its heading then states each interval's own level, and the caveat
    on looks says why.

    :param family: The contrast family, or None.
    :ptype family: ContrastFamily | None
    :param looks: The looks of the family's aspect (``Aspect.looks``, :func:`looks_of`); None for none.
    :ptype looks: Looks | None
    :return: The joint level and the family's size, or None.
    :rtype: tuple[float, int] | None
    """
    if family is None or unkept_plan(looks) is not None:
        return None
    corrected = (family.correction or "").strip().lower() not in ("", "none")
    return joint_guarantee(family.alpha, family.family_size or 0, corrected=corrected)


def looks_of(report_input: ReportInput, aspect: str | None) -> Looks | None:
    """The looks of the aspect a family or a contrast is read on, as the input carries them; None for none.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param aspect: The aspect's id; None for none.
    :ptype aspect: str | None
    :return: Its looks, or None.
    :rtype: Looks | None
    """
    return next((held.looks for held in report_input.aspects if held.id == aspect), None)


def _end(path: str, interval: tuple[float, float] | None, statistic: str, what: str) -> float:
    """One end of ``interval`` (``low`` or ``high``), refused by name where the input states no interval."""
    if interval is None:
        raise _unstated(path, f"names {what}, which the input does not state")
    return interval[0] if statistic == "low" else interval[1]


def _interval(
    path: str,
    value: tuple[float, float] | None,
    what: str,
    scale: Scale,
    *,
    unit: str = "",
    change: bool = False,
    whole: bool = False,
) -> Figure:
    if value is None:
        raise _unstated(path, f"names {what}, which the input does not state")
    low, high = value
    return Figure(
        path=path, kind="interval", value=(float(low), float(high)), scale=scale, unit=unit, change=change, whole=whole
    )


def _task(report_input: ReportInput, task_id: str) -> Task:
    return next(task for task in report_input.tasks if task.id == task_id)


def _criterion(report_input: ReportInput, criterion_id: str) -> Criterion:
    return next(criterion for criterion in report_input.criteria if criterion.id == criterion_id)


def _stratum(report_input: ReportInput, stratum_id: str, aspect_id: str | None = None) -> Stratum:
    """The stratum of that id (on that aspect, where one is named): the first the input holds.

    :raises KeyError: The input holds no such stratum.
    """
    held = (s for s in _scoped_strata(report_input, aspect_id) if s.id == stratum_id)
    found = next(held, None)
    if found is None:
        on = "" if aspect_id is None else f" on aspect {aspect_id}"
        raise KeyError(f"the input holds no stratum {stratum_id}{on}")
    return found


def _scoped_strata(report_input: ReportInput, aspect_id: str | None) -> list[Stratum]:
    """The strata a key reads from: every one, or, in an aspect's own strata keys, the ones slicing its cases."""
    return [stratum for stratum in report_input.strata if aspect_id is None or stratum.aspect == aspect_id]


def _printed_stratum(report_input: ReportInput, stratum: Stratum) -> str:
    """A stratum's printed name among its own aspect's strata (:func:`~threetears.evals.analysis.report.words.
    stratum_names`), so two of them that would print alike each print as their id. A stratum is read by its id on its
    aspect, and named by its own declared name; another aspect's stratum of the same id is another stratum."""
    held = [other for other in report_input.strata if other.aspect == stratum.aspect]
    declared = {other.id: other.name for other in held}
    declared[stratum.id] = stratum.name
    return stratum_names([other.id for other in held], declared).get(stratum.id) or stratum_name(
        stratum.id, stratum.name
    )


__all__ = [
    "UnkeptPlan",
    "looks_of",
    "joint_guarantee",
    "joint_level",
    "on_lead_aspect",
    "plan_condition",
    "unkept_plan",
]
