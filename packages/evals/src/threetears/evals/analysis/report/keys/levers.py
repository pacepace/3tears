"""Lever effects: one setting moved between two arms that differ in it alone, and two settings' interaction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from threetears.evals.analysis.report.keys.figure import Figure
from threetears.evals.analysis.report.keys.reads import _cell_spelling, _scale_of
from threetears.evals.analysis.report.keys.separation import _Difference, _OVERALL, _mean_difference, _read_difference
from threetears.evals.analysis.report.keys.vocabulary import _Ids, _Word, _control, _level, _refuse, _unstated
from threetears.evals.analysis.report_input import ReportInput, SettingValue, Stat
from threetears.evals.analysis.separation import NoSpread


# --- lever effects: one setting moved between two arms that differ in it alone ------------------------------------

#: A lever effect's fields: the second arm on the measure minus the first (``effect``), the interval on it as the compare
#: family reads the same two arms (its ends, its level), and whether that interval is paired or independent.
_LEVER_FIELDS = ("effect", "interval", "low", "high", "level", "basis")

#: An interaction's fields: the second move's effect minus the first's (``difference``), the interval on it, its level.
_INTERACTION_FIELDS = ("difference", "interval", "low", "high", "level")


def arm_settings(report_input: ReportInput) -> dict[str, dict[str, SettingValue]]:
    """Every arm's settings, each setting any arm states: a setting an arm leaves out has the control's value, as the
    input's own rule says; with no control it is unstated (None), which agrees only with another unstated one.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Each arm's settings, by arm id then setting, in the input's orders.
    :rtype: dict[str, dict[str, SettingValue]]
    """
    control = _control(report_input)
    base: Mapping[str, SettingValue] = control.settings if control is not None else {}
    names = list(dict.fromkeys(name for arm in report_input.arms for name in arm.settings))
    return {arm.id: {name: arm.settings.get(name, base.get(name)) for name in names} for arm in report_input.arms}


def lever_of(settings: Mapping[str, Mapping[str, SettingValue]], first: str, second: str) -> str | None:
    """The one setting two arms differ in, by :func:`arm_settings`; None where they differ in none or in several.

    :param settings: Every arm's settings, as :func:`arm_settings` reads them.
    :ptype settings: Mapping[str, Mapping[str, SettingValue]]
    :param first: One arm's id.
    :ptype first: str
    :param second: The other's.
    :ptype second: str
    :return: The setting, or None.
    :rtype: str | None
    """
    differ = [name for name, value in settings[first].items() if settings[second].get(name) != value]
    return differ[0] if len(differ) == 1 else None


def _lever_settings(report_input: ReportInput, _: tuple[str, ...] = ()) -> list[str]:
    """Every setting some two arms differ in alone, in the order the arms state settings."""
    settings = arm_settings(report_input)
    arms = list(settings)
    held = {lever_of(settings, first, second) for first in arms for second in arms if first != second}
    return [name for name in dict.fromkeys(n for values in settings.values() for n in values) if name in held]


def _lever_partners(settings: Mapping[str, Mapping[str, SettingValue]], setting: str, arm: str) -> list[str]:
    """The arms whose settings differ from one arm's in ``setting`` alone."""
    return [other for other in settings if other != arm and lever_of(settings, arm, other) == setting]


def _lever_froms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    settings = arm_settings(report_input)
    return [arm for arm in settings if _lever_partners(settings, prior[0], arm)]


def _lever_tos(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return _lever_partners(arm_settings(report_input), prior[0], prior[1])


def _lever_measures(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys([*report_input.measures, *report_input.judged]))


def _lever_places(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    """Every place a lever's effect can be read within: a task, a stratum the input holds on one aspect, a cell."""
    strata = [stratum.id for stratum in report_input.strata]
    return list(
        dict.fromkeys(
            [
                *(task.id for task in report_input.tasks),
                *(held for held in strata if strata.count(held) == 1),
                *(_cell_spelling(cell) for cell in report_input.cells.values()),
            ]
        )
    )


def _place_figures(report_input: ReportInput, place: str, path: str) -> Mapping[str, Mapping[str, Stat]]:
    """Each arm's figures within one place, by arm then measure."""
    held = [
        *(task.per_arm for task in report_input.tasks if task.id == place),
        *(stratum.per_arm for stratum in report_input.strata if stratum.id == place),
        *(cell.per_arm for cell in report_input.cells.values() if _cell_spelling(cell) == place),
    ]
    if len(held) > 1:
        raise _refuse(path, f"names place {place!r}, which the input holds {len(held)} times (a task, stratum or cell)")
    return held[0]


def _lever_where(report_input: ReportInput, place: str, path: str) -> str:
    """A lever key's ``<place>`` as a separation key's ``<where>`` spells it: ``overall`` for none, else the task, the
    stratum or the cell it names."""
    if not place:
        return _OVERALL
    _place_figures(report_input, place, path)
    if any(task.id == place for task in report_input.tasks):
        return f"task.{place}"
    if any(stratum.id == place for stratum in report_input.strata):
        return f"strata.{place}"
    return f"cells.{place}"


@dataclass(frozen=True)
class _Effect:
    """A move's effect and, where one reads, the interval on it (:class:`_Difference`), or why none does."""

    change: float
    read: _Difference | None
    why: str


def _lever_effect(report_input: ReportInput, values: tuple[str, ...], path: str) -> _Effect:
    """A move's effect, read as the compare family reads the arm it ends at against the arm it starts from (one spread
    rule for both): ``values`` = (from, to, measure, place)."""
    first, second, key, place = values
    where = _lever_where(report_input, place, path)
    try:
        read = _read_difference(report_input, second, first, key, where)
    except NoSpread as none:
        change = _mean_difference(report_input, second, first, key, where)
        if change is None:
            words = "" if where == _OVERALL else f" in {place}"
            raise _unstated(path, f"names {first}'s and {second}'s means on {key}{words}, which one lacks") from none
        return _Effect(change, None, str(none))
    return _Effect(read.difference, read, "")


def _recovered(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float]:
    """The interval on ``second``'s estimate minus ``first``'s, each ``(estimate, low, high)``, from their own
    intervals by the method of variance estimates recovery, the two read as independent."""
    (one, one_low, one_high), (two, two_low, two_high) = first, second
    change = two - one
    return change - math.hypot(two - two_low, one_high - one), change + math.hypot(two_high - two, one - one_low)


def _lever_figure(report_input: ReportInput, key: str, effect: _Effect, field: str, path: str) -> Figure:
    scale, unit = _scale_of(report_input, key)
    if field in ("effect", "difference"):
        return Figure(path=path, kind="number", value=effect.change, scale=scale, unit=unit, change=True)
    read = effect.read
    if read is None:
        raise _unstated(path, f"names an interval, which none reads: {effect.why}")
    if field == "basis":
        return Figure(path=path, kind="text", value=read.basis)
    if field == "level":
        if read.level is None:
            raise _unstated(path, "names the level of the contrast's interval, which the input does not state")
        return _level(path, read.level, "the interval's level")
    if field == "interval":
        return Figure(path=path, kind="interval", value=(read.low, read.high), scale=scale, unit=unit, change=True)
    end = read.low if field == "low" else read.high
    return Figure(path=path, kind="number", value=end, scale=scale, unit=unit, change=True)


def _lever(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """``lever.<setting>.<from>.<to>.<measure>[.<place>].<field>``."""
    _setting_id, first, second, key, *place, field = values
    effect = _lever_effect(report_input, (first, second, key, place[0] if place else ""), path)
    return _lever_figure(report_input, key, effect, field, path)


def _interaction_froms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a second move can start from: the first's setting value, in a context one other setting away."""
    setting, first, second = prior[:3]
    settings = arm_settings(report_input)
    return [
        arm
        for arm in settings
        if arm not in (first, second)
        and settings[arm][setting] == settings[first][setting]
        and lever_of(settings, first, arm) not in (None, setting)
        and _lever_partners(settings, setting, arm)
    ]


def _interaction_tos(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    """The arms a second move can end at: the first move's value, its context the second start's."""
    setting, first, second = prior[:3]
    start = prior[-1]
    settings = arm_settings(report_input)
    across = lever_of(settings, first, start)
    return [
        arm
        for arm in _lever_partners(settings, setting, start)
        if settings[arm][setting] == settings[second][setting] and lever_of(settings, second, arm) == across
    ]


def _interaction(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """``lever.<setting>.<from>.<to>.<measure>[.<place>].vs.<from>.<to>.<field>``: the second move's effect minus the
    first's, and the interval on it from theirs."""
    _setting_id, first, second, key, *place, start, end, field = values
    where = place[0] if place else ""
    one = _lever_effect(report_input, (first, second, key, where), path)
    two = _lever_effect(report_input, (start, end, key, where), path)
    change = two.change - one.change
    if one.read is None or two.read is None:
        return _lever_figure(report_input, key, _Effect(change, None, one.why or two.why), field, path)
    if one.read.level != two.read.level:
        why = "its two effects' intervals are at different levels, which do not combine"
        return _lever_figure(report_input, key, _Effect(change, None, why), field, path)
    first_read, second_read = one.read, two.read
    low, high = _recovered(
        (first_read.difference, first_read.low, first_read.high),
        (second_read.difference, second_read.low, second_read.high),
    )
    read = _Difference(change, low, high, first_read.level, "")
    return _lever_figure(report_input, key, _Effect(change, read, ""), field, path)


#: The slots every lever key begins with: the setting moved, the arm it moves from and the one it moves to (differing
#: in it alone), and the measure read.
_LEVER = (
    _Word("lever"),
    _Ids("<setting>", _lever_settings),
    _Ids("<from>", _lever_froms),
    _Ids("<to>", _lever_tos),
    _Ids("<measure>", _lever_measures),
)
#: The slots an interaction's second move is named by.
_SECOND_MOVE = (_Word("vs"), _Ids("<from>", _interaction_froms), _Ids("<to>", _interaction_tos))
#: Where a lever's effect is read within.
_PLACE = _Ids("<place>", _lever_places)

#: What the grammar says of a lever figure, in place of "descriptive, no interval".
_LEVER_NOTE = "Derived: computed here from two arms' stated figures and their spreads, never a test the pipeline ran."

#: What every lever key's figures are: computed here, from two arms' figures the input states.
_LEVER_BASIS = (
    "Computed here, as `compare.<to>.<from>.<measure>.<where>` reads the same two arms: an effect's `interval` (its "
    "ends `low`, `high`, its `level`) is a contrast's own where one pairs the two arms overall, else the two arms' own "
    "spreads combined as independent (a stated interval, half-width or standard error, or a pass/fail rate's binomial "
    "spread from its count), and its `basis` says which; an interaction's is recovered from its two effects' intervals "
    "as if independent (MOVER), at their common level; with no spread to read there is no interval. levers_effects "
    "lists them"
)


#: What the grammar says of a priced move, in place of "descriptive, no interval".
_PRICE_NOTE = (
    "Derived: computed here from the options' stated costs and gains; an interval carried through is labelled paired "
    "(a contrast's) or unpaired (two options' own, as if independent), never a test the pipeline ran."
)
#: What the grammar says of a pooled figure, in place of "descriptive, no interval".
_POOLED_NOTE = (
    "Derived: pooled here from the slices' stated figures with the weights `weights` names; its interval combines "
    "theirs as if independent, never a test the pipeline ran."
)

#: What the grammar says of an overall interval derived from its cells, in place of "descriptive, no interval".
_OVERALL_INTERVAL_NOTE = (
    "Derived: combined here from the cells' stated spreads, the cells read as independent, never a test the pipeline "
    "ran."
)

__all__ = [
    "arm_settings",
    "lever_of",
]
