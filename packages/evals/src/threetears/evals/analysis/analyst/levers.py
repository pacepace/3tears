"""Lever effects: what moving one arm setting does to a measure, read between arms that differ in that setting alone.

A campaign whose arms vary several settings (a model, a prompt, a reasoning level, a retry) asks what each one buys,
and whether what it buys depends on another. :func:`lever_effects` answers both from the arms' own figures, and only
where the arms isolate the setting.

**Settings.** Each arm's settings as the input states them
(:func:`~threetears.evals.analysis.report.figure_keys.arm_settings`): with a control, a setting an arm leaves out has
the control's value, the input's own rule; with none, it is unstated, and agrees only with another unstated one. A
setting *varies* when the arms hold more than one value of it. Two arms are a *lever pair* for a setting when their
settings differ in it and agree on every other (:func:`~threetears.evals.analysis.report.figure_keys.lever_of`).

**An effect** is the second arm minus the first on a measure or judged dimension: overall (the measure's by-arm
figure, the lead aspect's), or within one task, stratum or target-matrix cell. It is signed as the measure reads;
``higher_is_better`` says which way is better. Every lever pair is listed once, ordered from the arm holding the
control's value where one does, else in the input's arm order; any lever pair is a key, either way round
(``lever.<setting>.<from>.<to>.<measure>[.<place>].effect``).

**Its interval** is the one the compare family reads for the same two arms at the same place
(``compare.<to>.<from>.<measure>.<where>``; :mod:`~threetears.evals.analysis.analyst.compare`), so a lever and a
comparison of the same arms never disagree on whether an interval exists or what it is: one spread rule for both
(:mod:`~threetears.evals.analysis.separation`). Overall, where a contrast states one of the two arms against the other
(the control) on the lead aspect, the effect is that paired estimate, its difference and its interval at its family's
level. Otherwise the effect is the two means' difference, and its interval combines the two arms' own spreads as
independent samples: each arm's standard error is read from its stated interval or half-width at the stated level,
its standard error, or, for a pass/fail rate with a count, its binomial spread. **That assumes the two figures are
independent.** Where the arms ran the same cases their figures move together and the paired interval would be
narrower, so this one is the conservative, wider one. With no spread to read for either arm the effect is a point
difference: descriptive, with no interval and no test. Every effect's ``basis`` says which it is, and its ``level``
the interval's level. Nothing here is a significance test.

**An interaction** compares the same move (the same setting, between the same two values) made in two contexts that
differ in exactly one other setting: the second move's effect minus the first's (``difference``). Two moves between
the same two values are the same move whichever way each was listed; the second is read in the first's direction
(from the first's starting value), so an interaction is found however the arms are ordered. Four arms crossing two
settings are one interaction, listed once: one setting's move across the other, never that and the other's move across
the first too, which is the same difference of differences. Its interval is recovered
from the two effects' intervals by the method of variance estimates recovery (MOVER; Zou and Donner, 2008), four
distinct arms read as independent: for effects ``a`` (interval ``l_a`` to ``u_a``) and ``b`` (``l_b`` to ``u_b``),
the difference ``d = b - a`` runs from ``d - sqrt((b - l_b)^2 + (u_a - a)^2)`` to ``d + sqrt((u_b - b)^2 +
(a - l_a)^2)``, at the two effects' common level (two intervals at different levels do not combine, and give none).
Its verdict: ``reverses`` (both effects' intervals clear zero, on opposite sides), ``differs`` (the difference's
interval clears zero: the effect's size changes beyond its intervals), ``within`` (the difference's interval holds
zero), or ``no_interval`` (an effect has none, so only the point estimates can be compared, and are). A setting whose
effect reverses or differs across another is not one effect: report it per level of the other setting.

**Order.** Effects come first, then interactions; within each, the rows with an interval lead (they say the most), in
the order above, and the rows with none follow.

**What it refuses**, each with the call that answers instead: a measure, place or setting the snapshot does not hold;
a snapshot whose arms vary no setting; and one where no two arms differ in exactly one setting, since every pair that
differs differs in two or more and no difference between two arms can be put on one setting (the remedy reads each
arm's figure with ``measure_read`` and each arm's settings with ``campaign_overview``); and a measure no lever pair
holds a mean on at the place asked.

**Conditions are on what the snapshot holds**: the arms' settings, which arms hold a figure, which figures state an
interval. Nothing here reads what kind of eval the campaign is or how it was designed: a factorial sweep and two models
compared on one prompt are read the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.report.figure_keys import arm_settings, lever_of
from threetears.evals.analysis.report_input import Judged, Measure, ReportInput, SettingValue
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["LEVERS_LIMIT", "LeverEffects", "lever_effects"]

#: A page's default rows: effects, then interactions, each a handful of figures.
LEVERS_LIMIT = 40

#: How two effects of one move compare: see this module's An interaction.
LeverVerdict = Literal["reverses", "differs", "within", "no_interval"]

#: What every effect and interaction rests on, as the result states it.
_ASSUMPTION = (
    "Each effect is the second arm minus the first, between two arms whose settings differ in one setting alone, read "
    "as arms_compare reads the same two arms: paired where a contrast pairs them overall, else the two means' "
    "difference with the two arms' own spreads (a stated interval, half-width or standard error, or a pass/fail rate's "
    "binomial spread from its count) combined as if independent, which is conservative where the arms ran the same "
    "cases; each effect's basis and level say which. An interaction's interval is recovered from its two effects' as "
    "if independent (MOVER), at their common level. With no spread to read there is no interval: a point difference, "
    "descriptive, with no test. Rows with an interval come first."
)

#: What reads each arm's settings and figures, for a refusal's remedy.
_READ_ARMS = (
    "campaign_overview lists each arm's settings; measure_read reads each arm's figure on the measure and its interval, "
    "to compare arms whose settings differ in more than one"
)


class _Lever(BaseModel):
    """A lever result or row: frozen, and no field it does not declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SettingRow(_Lever):
    """One setting the arms vary: its values, and whether two arms isolate it."""

    setting: str = Field(description="The setting, as `settings.<arm>.<setting>` names it.")
    values: list[CitedFigure] = Field(
        description="Each value the arms hold, by the key of the first arm stating it (the control's, for a value an "
        "arm takes from the control)."
    )
    isolated: bool = Field(description="Whether some two arms differ in this setting alone.")
    note: str | None = Field(description="Why no effect of it is read, where none is.")


class EffectRow(_Lever):
    """One lever pair: the setting moved from one arm's value to another's, and what the move does to the measure."""

    setting: str = Field(description="The setting moved.")
    from_arm: CitedFigure = Field(description="The arm the move starts from.")
    to_arm: CitedFigure = Field(description="The arm it ends at, whose settings differ in this one alone.")
    from_value: CitedFigure | None = Field(description="The setting's value where the move starts.")
    to_value: CitedFigure | None = Field(description="Its value where it ends.")
    figures: list[CitedFigure] = Field(
        description="The effect (`effect`: to minus from), then, where a spread reads, its interval, its level and its "
        "basis (paired or independent)."
    )


class InteractionRow(_Lever):
    """The same move made in two contexts that differ in one other setting, and how its effect changes."""

    setting: str = Field(description="The setting moved.")
    across: str = Field(description="The other setting the two contexts differ in.")
    first: CitedFigure = Field(description="The first move's effect.")
    second: CitedFigure = Field(description="The second move's effect, in the other context.")
    figures: list[CitedFigure] = Field(
        description="The second effect minus the first (`difference`), then its interval and level, where both effects "
        "hold one at one level."
    )
    verdict: LeverVerdict = Field(
        description="reverses: both effects clear zero on opposite sides; differs: the difference clears zero; "
        "within: it holds zero; no_interval: an effect states none, so only the point estimates compare."
    )


class LeverEffects(_Lever):
    """What each varying setting does to one measure, between arms that differ in it alone, and how that changes."""

    measure: str = Field(description="The measure or judged dimension read, by key.")
    place: str | None = Field(description="The task, stratum or cell it is read within; None for overall.")
    level: str | None = Field(
        description="The level the measure states its own intervals at; each effect's interval states its own level "
        "beside it (a pairing contrast's family level, else this, else 95%)."
    )
    higher_is_better: bool | None = Field(description="Which way the measure is better, where it says.")
    assumption: str = Field(description="What every effect and interval here rests on.")
    settings: list[SettingRow] = Field(description="Every setting the arms vary, on every page.")
    effects: list[EffectRow] = Field(description="This page's effects, by setting, each lever pair once.")
    interactions: list[InteractionRow] = Field(description="This page's interactions, after every effect.")
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("effect or interaction", "effects and interactions")
    total: int = Field(description="Effects and interactions in every page.")
    offset: int = Field(description="The rows skipped before this page.")
    next_offset: int | None = Field(description="The next page's offset; none on the last page.")


def _table(report_input: ReportInput, measure_key: str) -> Measure | Judged:
    """The measure or judged dimension a key names.

    :raises RefusedInput: The input holds neither of that key.
    """
    table = report_input.measures.get(measure_key) or report_input.judged.get(measure_key)
    if table is None:
        raise RefusedInput(
            f"measure_key {measure_key!r} names no measure or judged dimension this snapshot holds",
            remedy="campaign_overview lists every measure key, under measures",
            parameters=("measure_key",),
            valid_values=[*report_input.measures, *report_input.judged],
        )
    return table


def _places(report_input: ReportInput) -> list[str]:
    """Every place an effect can be read within: a task, a stratum the input holds on one aspect, a cell."""
    strata = [stratum.id for stratum in report_input.strata]
    return list(
        dict.fromkeys(
            [
                *(task.id for task in report_input.tasks),
                *(held for held in strata if strata.count(held) == 1),
                *(f"{cell.row}.{cell.column}" for cell in report_input.cells.values()),
            ]
        )
    )


def _value_figure(
    report_input: ReportInput, settings: Mapping[str, Mapping[str, SettingValue]], arm: str, setting: str
) -> CitedFigure | None:
    """An arm's value of a setting, by the key that states it: its own, else the control's it takes."""
    stated = next(held for held in report_input.arms if held.id == arm)
    if setting in stated.settings:
        return cite_if(f"settings.{arm}.{setting}", report_input)
    control = next((held.id for held in report_input.arms if held.role == "control"), None)
    if control is None or settings[control].get(setting) != settings[arm].get(setting):
        return None
    return cite_if(f"settings.{control}.{setting}", report_input)


def _varying(settings: Mapping[str, Mapping[str, SettingValue]]) -> list[str]:
    names = list(dict.fromkeys(name for values in settings.values() for name in values))
    return [name for name in names if len({_hashed(values.get(name)) for values in settings.values()}) > 1]


def _hashed(value: SettingValue) -> tuple[str, SettingValue]:
    """A setting's value as a set holds it: a number and a text of the same digits stay apart."""
    return ("text" if isinstance(value, str) else "number", value)


def _pairs(
    report_input: ReportInput, settings: Mapping[str, Mapping[str, SettingValue]], setting: str
) -> list[tuple[str, str]]:
    """Every lever pair for a setting, once each: from the arm holding the control's value where one does, else in the
    input's arm order."""
    arms = list(settings)
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    held = settings[control].get(setting) if control is not None else None
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(arms):
        for second in arms[index + 1 :]:
            if lever_of(settings, first, second) != setting:
                continue
            flipped = control is not None and settings[second][setting] == held and settings[first][setting] != held
            pairs.append((second, first) if flipped else (first, second))
    return pairs


def _tangled(settings: Mapping[str, Mapping[str, SettingValue]], setting: str) -> str:
    """Why no two arms isolate a setting: a pair differing in it, and every other setting the pair differs in."""
    arms = list(settings)
    for index, first in enumerate(arms):
        for second in arms[index + 1 :]:
            differ = [name for name, value in settings[first].items() if settings[second].get(name) != value]
            if setting in differ:
                others = ", ".join(name for name in differ if name != setting)
                return f"every two arms differing in it differ in another setting too ({first} and {second}: {others})"
    return "no two arms differ in it"


def _stem(setting: str, first: str, second: str, measure: str, place: str | None) -> str:
    return f"lever.{setting}.{first}.{second}.{measure}" + (f".{place}" if place is not None else "")


#: One lever pair, structurally: the setting moved, the arm it moves from, the arm it moves to.
_Move = tuple[str, str, str]


def _move_stem(move: _Move, measure: str, place: str | None) -> str:
    setting, first, second = move
    return _stem(setting, first, second, measure, place)


def _crossing_stem(crossed: tuple[_Move, _Move, str], measure: str, place: str | None) -> str:
    """An interaction's key stem: the first move's, then ``vs.<from>.<to>`` of the second."""
    one, two, _across = crossed
    return f"{_move_stem(one, measure, place)}.vs.{two[1]}.{two[2]}"


def _holds_mean(report_input: ReportInput, measure: str, arm: str, place: str | None) -> bool:
    """Whether an arm holds a mean on the measure, overall or within the place."""
    if place is None:
        table = report_input.measures.get(measure) or report_input.judged.get(measure)
        stat = table.per_arm.get(arm) if table is not None else None
    else:
        held = [
            *(task.per_arm for task in report_input.tasks if task.id == place),
            *(stratum.per_arm for stratum in report_input.strata if stratum.id == place),
            *(cell.per_arm for cell in report_input.cells.values() if f"{cell.row}.{cell.column}" == place),
        ]
        stat = held[0].get(arm, {}).get(measure) if len(held) == 1 else None
    return stat is not None and stat.mean is not None


def _effect_row(
    report_input: ReportInput,
    settings: Mapping[str, Mapping[str, SettingValue]],
    move: _Move,
    measure: str,
    place: str | None,
) -> EffectRow:
    """One lever pair's effect, and its interval where both arms state one."""
    setting, first, second = move
    stem = _stem(setting, first, second, measure, place)
    return EffectRow(
        setting=setting,
        from_arm=cite(f"arm:{first}", report_input),
        to_arm=cite(f"arm:{second}", report_input),
        from_value=_value_figure(report_input, settings, first, setting),
        to_value=_value_figure(report_input, settings, second, setting),
        figures=_with_interval(report_input, stem, "effect", ("level", "basis")),
    )


def _with_interval(report_input: ReportInput, stem: str, field: str, beside: Sequence[str]) -> list[CitedFigure]:
    """A key's point figure, then its interval and the figures beside it, where an interval reads."""
    point = cite(f"{stem}.{field}", report_input)
    interval = cite_if(f"{stem}.interval", report_input)
    if interval is None:
        return [point]
    found = [cite_if(f"{stem}.{other}", report_input) for other in beside]
    return [point, interval, *(figure for figure in found if figure is not None)]


def _has_interval(report_input: ReportInput, stem: str) -> bool:
    """Whether a lever or interaction key reads an interval."""
    return cite_if(f"{stem}.interval", report_input) is not None


def _clears(interval: CitedFigure | None) -> int:
    """Which side of zero an interval figure sits wholly on: 1 above, -1 below, 0 holding it (or none)."""
    if interval is None or not isinstance(interval.value, tuple):
        return 0
    low, high = interval.value
    return 1 if low > 0 else -1 if high < 0 else 0


def _verdict(first: EffectRow, second: EffectRow, difference: CitedFigure | None) -> LeverVerdict:
    if len(first.figures) < 2 or len(second.figures) < 2 or difference is None:
        return "no_interval"
    if _clears(first.figures[1]) * _clears(second.figures[1]) < 0:
        return "reverses"
    return "differs" if _clears(difference) else "within"


def _crossed(
    settings: Mapping[str, Mapping[str, SettingValue]], moves: Sequence[_Move]
) -> list[tuple[_Move, _Move, str]]:
    """Every two moves of one setting between the same two values whose contexts differ in exactly one other setting,
    with that setting: in the moves' order, the second read in the first's direction (flipped where it was listed the
    other way round), so the arms' order never hides one.

    One interaction per four arms: the same four, read as the other setting's move across this one (``retry`` across
    ``reasoning`` beside ``reasoning`` across ``retry``), are one difference of differences, so the first found stands
    for them and the other is not listed again."""
    found: list[tuple[_Move, _Move, str]] = []
    squares: set[frozenset[str]] = set()
    for index, one in enumerate(moves):
        setting, first, second = one
        for two in moves[index + 1 :]:
            if two[0] != setting:
                continue
            start, end = two[1], two[2]
            if (
                settings[first][setting] == settings[end][setting]
                and settings[second][setting] == settings[start][setting]
            ):
                start, end = end, start
            same = settings[first][setting] == settings[start][setting] and (
                settings[second][setting] == settings[end][setting]
            )
            across = lever_of(settings, first, start)
            square = frozenset({first, second, start, end})
            if same and across not in (None, setting) and lever_of(settings, second, end) == across:
                if square not in squares:
                    squares.add(square)
                    found.append((one, (setting, start, end), str(across)))
    return found


def _interaction_row(
    report_input: ReportInput,
    settings: Mapping[str, Mapping[str, SettingValue]],
    crossed: tuple[_Move, _Move, str],
    measure: str,
    place: str | None,
) -> InteractionRow:
    """Two effects of one move in contexts one other setting apart: the second minus the first, and the verdict."""
    one, two, across = crossed
    first, second = (_effect_row(report_input, settings, move, measure, place) for move in (one, two))
    stem = _crossing_stem(crossed, measure, place)
    figures = _with_interval(report_input, stem, "difference", ("level",))
    return InteractionRow(
        setting=one[0],
        across=across,
        first=first.figures[0],
        second=second.figures[0],
        figures=figures,
        verdict=_verdict(first, second, figures[1] if len(figures) > 1 else None),
    )


def lever_effects(
    report_input: ReportInput,
    measure_key: str,
    *,
    setting_filter: str | None = None,
    place_filter: str | None = None,
    limit: int = LEVERS_LIMIT,
    offset: int = 0,
) -> LeverEffects:
    """What each setting the arms vary does to one measure, between arms that differ in it alone, and whether that
    changes with another setting (this module's account).

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param measure_key: The measure or judged dimension, by key.
    :ptype measure_key: str
    :param setting_filter: Only this setting's effects and interactions.
    :ptype setting_filter: str | None
    :param place_filter: The task, stratum or cell (``<row>.<column>``) to read the measure within; None for overall.
    :ptype place_filter: str | None
    :param limit: The most rows (effects, then interactions) on the page.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :return: The settings, and a page of effects and interactions.
    :rtype: LeverEffects
    :raises RefusedInput: The measure, place or setting names nothing the snapshot holds; no setting varies; no two
        arms differ in exactly one setting (or in the one named); or no such pair holds the measure there.
    """
    table = _table(report_input, measure_key)
    places = _places(report_input)
    if place_filter is not None and place_filter not in places:
        raise RefusedInput(
            f"place_filter {place_filter!r} names no task, stratum or cell this snapshot holds",
            remedy="campaign_overview lists every task, stratum and cell; omit place_filter to read overall",
            parameters=("place_filter",),
            valid_values=places,
        )
    settings = arm_settings(report_input)
    varying = _varying(settings)
    if not varying:
        raise RefusedInput(
            "the arms vary no setting: every arm's settings agree, so no effect is read",
            remedy=_READ_ARMS,
            parameters=("measure_key",),
        )
    if setting_filter is not None and setting_filter not in varying:
        raise RefusedInput(
            f"setting_filter {setting_filter!r} names no setting the arms vary",
            remedy="name one of the settings the arms vary, or omit setting_filter",
            parameters=("setting_filter",),
            valid_values=varying,
        )
    named = varying if setting_filter is None else [setting_filter]
    pairs = {setting: _pairs(report_input, settings, setting) for setting in varying}
    if not any(pairs[setting] for setting in named):
        which = "any one setting" if setting_filter is None else f"{setting_filter} alone"
        why = "; ".join(f"{setting}: {_tangled(settings, setting)}" for setting in named)
        raise RefusedInput(
            f"no two arms differ in {which}, so no difference between two arms can be put on one setting ({why})",
            remedy=_READ_ARMS,
            parameters=("setting_filter",) if setting_filter is not None else ("measure_key",),
            valid_values=[setting for setting in varying if pairs[setting]],
        )
    moves = [
        (setting, first, second)
        for setting in named
        for first, second in pairs[setting]
        if _holds_mean(report_input, measure_key, first, place_filter)
        and _holds_mean(report_input, measure_key, second, place_filter)
    ]
    if not moves:
        where = "overall" if place_filter is None else f"in {place_filter}"
        raise RefusedInput(
            f"no two arms that differ in one setting both hold a mean on {measure_key!r} {where}",
            remedy="measure_read lists which arms hold it, by arm, task, stratum or cell; name place_filter for a "
            "measure held within slices",
            parameters=("place_filter",) if place_filter is not None else ("measure_key",),
        )
    crossed = _crossed(settings, moves)
    moves.sort(key=lambda move: not _has_interval(report_input, _move_stem(move, measure_key, place_filter)))
    crossed.sort(key=lambda pair: not _has_interval(report_input, _crossing_stem(pair, measure_key, place_filter)))
    total = len(moves) + len(crossed)
    end = offset + limit
    shown_moves = moves[offset:end]
    shown_crossed = crossed[max(offset - len(moves), 0) : max(end - len(moves), 0)]
    return LeverEffects(
        measure=measure_key,
        place=place_filter,
        level=table.level,
        higher_is_better=table.higher_is_better if isinstance(table, Measure) else True,
        assumption=_ASSUMPTION,
        settings=[
            SettingRow(
                setting=setting,
                values=_values(report_input, settings, setting),
                isolated=bool(pairs[setting]),
                note=None if pairs[setting] else _tangled(settings, setting),
            )
            for setting in varying
        ],
        effects=[_effect_row(report_input, settings, move, measure_key, place_filter) for move in shown_moves],
        interactions=[
            _interaction_row(report_input, settings, pair, measure_key, place_filter) for pair in shown_crossed
        ],
        total=total,
        offset=offset,
        next_offset=end if end < total else None,
    )


def _values(
    report_input: ReportInput, settings: Mapping[str, Mapping[str, SettingValue]], setting: str
) -> list[CitedFigure]:
    """Each value the arms hold of a setting, by the key of the first arm stating it."""
    seen: set[tuple[str, SettingValue]] = set()
    figures: list[CitedFigure] = []
    for arm, values in settings.items():
        held = _hashed(values.get(setting))
        if held in seen:
            continue
        seen.add(held)
        figure = _value_figure(report_input, settings, arm, setting)
        if figure is not None:
            figures.append(figure)
    return figures
