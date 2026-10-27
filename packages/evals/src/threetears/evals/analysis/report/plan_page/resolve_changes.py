"""The charts of change: a forest (each change against the control with its interval) and diverging bars (deltas
about a centre).

**forest** reads the pipeline's own tests:

- *readings*: a row per candidate, its change and the interval the reading was read on, bold where separated
  (the interval clear of zero), its note in GENERATOR §3's words (worse, better, not separated);
- *contrasts*: a row per arm and measure, grouped under the measure's reader name where the chart reads several (and
  under the aspect's too where two aspects' contrasts share a measure, a query naming no aspect selecting both; in a
  panel per aspect where the plan asks for panels by aspect and the contrasts sit on several), a rate's delta in
  points, bold where the contrast regressed or improved. A comparison the query asks for that the
  input holds no contrast of is named in a caption, never filled in.

**diverging_bars** reads a change about its centre:

- *readings* and *contrasts*: as the forest's rows, a bar each, bold where separated and muted where not;
- *measures*, *judged*, *strata* and *tasks*: each arm less the control (a diverging chart has a centre, so a plan
  storing no transform is drawn so too), per slice for strata and tasks (the query's ``limit`` keeps the largest
  strata), in panels by arm, or by slice, where the plan asks for them, else grouped by arm; bold or muted by the
  descriptive 95% rule, uncorrected, which a caption says; a slice whose figure states only its change from the
  control is drawn at that change, with no bold; the bars share one axis, so the measures read are in one unit
  (:func:`~threetears.evals.analysis.report.plan_page.charting.unit_refusal`);
- *human_pref*: the share of blind ratings preferring the control, per slice, about the no-preference point (50%),
  bold where its interval excludes it;
- *configs*: lever effects, each level of a factor against the factor's first level on the query's measure,
  averaged over every combination of the other factors that holds both: descriptive, uncorrected;
- *compare*: the query's first arm less its second at each place ``arms_compare`` reads, with its interval, bold
  where the interval clears zero; a caption names the two arms and how the interval was read (paired or independent);
- *levers*: each lever effect, grouped by its setting, then each interaction, grouped by its setting and the setting
  it crosses, as ``lever_effects`` reads them: bold where an effect's interval clears zero, or an interaction reverses
  or differs. An effect is labelled by the arm it starts from and the value it moves to, as the input states it (a
  declared level's name, else the value, a whole number whole: :func:`setting_value`); an interaction by its two
  moves, the second's effect less the first's, so no two rows read alike (:func:`lever_label`). A row's short label
  names its arms by their tags. The caption speaks of interactions only where the chart draws one. A lever's effect,
  which its key states as a share where the measure is a rate, is drawn in points, as a comparison is.

Every interval is drawn only where the chart states a level; every note is written by the package's number rule. A
chart of what an analysis computes (compare, levers) spells its values table at its quantity's precision, a point to
one decimal, as the rows' keys read (:func:`~threetears.evals.analysis.report.plan_page.charting.chart_block`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.analyst.compare import ComparisonRow
from threetears.evals.analysis.analyst.levers import EffectRow, InteractionRow
from threetears.evals.analysis.report.charts import BarRow, BarsChart, BarsPanel, ChartReference, ForestChart, ForestRow
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    chart_block,
    contrast_groups,
    contrast_unit_of,
    contrast_words,
    figure_stat,
    figure_unit,
    interval_words,
    measure_words,
    missing_note,
    reference_line,
    separated,
    slice_keys,
    slice_name,
    spread,
    stat_value,
    unit_refusal,
)
from threetears.evals.analysis.report.plan_page.look import Quantity, centred_domain, figure_text, quantity
from threetears.evals.analysis.report.plan_page.words import (
    BASIS_NOTE,
    COMPARE_NARROW_NOTE,
    COMPARE_NESTED_TEMPLATE,
    COMPARE_STATED_COUNT,
    COMPARE_UNSTATED_COUNT,
    COMPARE_NOTE,
    HUMAN_PREF_NOTE,
    LEVER_CROSS_GROUP,
    LEVER_CROSS_TEMPLATE,
    LEVER_EFFECT_GROUP,
    LEVER_MOVE_TEMPLATE,
    LEVER_NOTE,
    LEVERS_INTERACTION_NOTE,
    LEVERS_NOTE,
    LEVERS_READ_NOTE,
    OVERALL_WORD,
    PAIRED_BASIS_TEMPLATE,
    UNCORRECTED_NOTE,
    change_words,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.quantities import PERCENT, POINTS
from threetears.evals.analysis.report.figure_keys import arm_settings
from threetears.evals.analysis.report_input import Contrast, PreferenceRow, Reading, ReportInput, Stat, Stratum, Task
from threetears.evals.analysis.report_rows import lever_arms
from threetears.evals.analysis.style import humanize

__all__ = ["RESOLVERS"]

#: The sources whose rows an analysis computes, unrounded: a values table spells them at their quantity's precision.
_COMPUTED = frozenset({"compare", "levers"})

#: The contrast verdicts that separate an arm from the control in a direction.
_DIRECTED = frozenset({"improved", "regressed"})

#: The no-preference point a human preference is read about, in percent.
_NO_PREFERENCE = 50.0


@dataclass(frozen=True)
class _Change:
    """One drawn change: its row's id, label, colour, group and panel, its value and interval, and how it reads."""

    id: str
    label: str
    color: str | None
    value: float
    interval: tuple[float, float] | None
    strong: bool | None
    note: str
    tip: list[str]
    group: str | None = None
    panel: str | None = None
    short_label: str | None = None


@dataclass(frozen=True)
class _Panel:
    """One small multiple: its title (an arm's, with its chip, or a slice's) and its changes."""

    title: str
    tag: str | None
    color: str | None
    rows: list[_Change]


def _arm_panel(context: ChartContext, arm: str, rows: list[_Change]) -> _Panel:
    look = context.look.arm(arm)
    return _Panel(title=look.name, tag=look.tag, color=look.token, rows=rows)


# --- readings and contrasts: the pipeline's own tests -----------------------------------------------------------


def _reading_changes(context: ChartContext) -> tuple[list[_Change], Quantity]:
    drawn = quantity("points")
    readings: dict[str, Reading] = {}
    for row in context.rows():
        assert isinstance(row.item, Reading)
        readings[row.arm] = row.item
    changes = []
    for arm in context.arm_order(readings):
        reading = readings[arm]
        look = context.look.arm(arm)
        interval = reading.within_interval
        low, high = (float(interval[0]), float(interval[1])) if interval is not None else (None, None)
        direction = "worse" if high is not None and high < 0 else "better" if low is not None and low > 0 else ""
        tip = [look.name, figure_text(float(reading.change), drawn)]
        if low is not None and high is not None:
            tip[1] += f", {interval_words(low, high, drawn)}"
        changes.append(
            _Change(
                id=arm,
                label=look.label,
                color=look.token,
                value=float(reading.change),
                interval=(low, high) if low is not None and high is not None else None,
                strong=bool(direction),
                note=change_words(direction),
                tip=tip,
            )
        )
    return changes, drawn


def _contrast_changes(
    context: ChartContext, *, by_aspect: bool = False
) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """Each selected contrast as a change, a rate's delta in points; and the caption naming each asked pair the input
    holds no contrast of.

    Where ``by_aspect`` (the chart panels, and its query asks for panels by aspect) and the contrasts sit on more than
    one aspect, each change carries its aspect's name as its panel, grouped within it by measure where there are
    several. Otherwise the changes are grouped by measure where there are several groups, a measure whose contrasts
    sit on more than one aspect a group per aspect, each naming it."""
    report_input = context.report_input
    contrasts = [row.item for row in context.rows() if isinstance(row.item, Contrast)]
    groups = contrast_groups(context, contrasts, context.query.measures)
    panelled = by_aspect and len({group.aspect for group in groups}) > 1
    grouped = len({group.key for group in groups}) > 1 if panelled else len(groups) > 1
    drawn: Quantity | None = None
    changes: list[_Change] = []
    for group in groups:
        named = group.aspect if group.split or panelled else None
        for arm in context.arm_order(group.by_arm):
            contrast = group.by_arm[arm]
            if contrast.delta is None:
                continue
            unit = figure_unit(report_input, group.key, kind=contrast.kind)
            here = quantity(unit, change=True)
            drawn = drawn or here
            look = context.look.arm(arm)
            value = float(contrast.delta) * here.factor
            interval = (
                (float(contrast.interval[0]) * here.factor, float(contrast.interval[1]) * here.factor)
                if contrast.interval is not None
                else None
            )
            words = contrast_words(report_input, contrast, named)
            direction = {"regressed": "worse", "improved": "better"}.get(contrast.verdict, "")
            tip = [look.name, words, figure_text(value, here)]
            if interval is not None:
                tip[2] += f", {interval_words(*interval, here)}"
            shown = contrast_words(report_input, contrast, None) if panelled else words
            changes.append(
                _Change(
                    id=arm,
                    label=look.label,
                    color=look.token,
                    value=value,
                    interval=interval,
                    strong=contrast.verdict in _DIRECTED,
                    note=" ".join(part for part in (figure_text(value, here), change_words(direction)) if part),
                    tip=tip,
                    group=shown if grouped else None,
                    panel=named if panelled else None,
                )
            )
    return changes, drawn or quantity("points"), _missing_contrasts(context, contrasts)


def _missing_contrasts(context: ChartContext, contrasts: Sequence[Contrast]) -> tuple[str, ...]:
    """Each (arm, measure) the query asks for by name that no contrast holds, in words (GENERATOR §2: say so in one
    caption line; never fill it in)."""
    control = context.look.control
    if not context.query.measures or control is None:
        return ()
    asked_arms = context.query.arms or [arm for arm in context.look.arms if arm != control.id]
    held = {(contrast.arm, contrast.measure) for contrast in contrasts}
    kinds = {contrast.measure: contrast.kind for contrast in context.report_input.contrasts}
    missing = [
        f"{context.look.arm(arm).name}, {measure_words(context.report_input, key, kind=kinds.get(key))}"
        for key in context.query.measures
        for arm in context.arm_order(arm for arm in asked_arms if arm != control.id)
        if (arm, key) not in held
    ]
    return missing_note(missing)


# --- each arm less the control, computed here from point figures -------------------------------------------------


def _control_or_refuse(context: ChartContext) -> str:
    control = context.look.control
    if control is None:
        raise context.refuse("it draws each arm less the control, and the input names no control")
    return control.id


def _descriptive(
    context: ChartContext, key: str, arm: str, mine: Stat | None, theirs: Stat | None, drawn: Quantity, where: str = ""
) -> _Change | None:
    """One arm less the control on one figure: the pipeline's own change where the arm's figure states only that,
    else the difference of the two values, bold or muted by the descriptive rule; None where neither is held."""
    if mine is None:
        return None
    look = context.look.arm(arm)
    head = [where] if where else []
    if mine.change is not None and (theirs is None or stat_value(context.report_input, key, mine) is None):
        change = float(mine.change) * drawn.factor
        return _Change(
            id=arm,
            label=look.label,
            color=look.token,
            value=change,
            interval=None,
            strong=None,
            note=figure_text(change, drawn),
            tip=[*head, look.name, figure_text(change, drawn)],
        )
    mine_value = stat_value(context.report_input, key, mine)
    theirs_value = stat_value(context.report_input, key, theirs) if theirs is not None else None
    if mine_value is None or theirs_value is None or theirs is None:
        return None
    change = (mine_value[0] - theirs_value[0]) * drawn.factor
    spreads = [spread(mine), spread(theirs)]
    read = separated(change, [one * drawn.factor if one is not None else None for one in spreads])
    control = context.look.arm(_control_or_refuse(context))
    level = quantity("percent") if drawn.unit == "pts" else drawn
    tip = [
        *head,
        f"{control.name} {_level_text(theirs_value[0] * drawn.factor, level)} · "
        f"{look.name} {_level_text(mine_value[0] * drawn.factor, level)}",
        figure_text(change, drawn),
    ]
    return _Change(
        id=arm,
        label=look.label,
        color=look.token,
        value=change,
        interval=None,
        strong=read,
        note=figure_text(change, drawn),
        tip=tip,
    )


def _level_text(value: float, drawn: Quantity) -> str:
    """A level (not a change) in the quantity's unit, unsigned."""
    return figure_text(value, Quantity(drawn.factor, drawn.fmt, drawn.unit))


def _figure_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """measures or judged: each candidate's figure less the control's, per measure."""
    control = _control_or_refuse(context)
    report_input = context.report_input
    keys = list(dict.fromkeys(row.key for row in context.rows()))
    arms = list(dict.fromkeys(row.arm for row in context.rows() if row.arm != control))
    grouped = len(keys) > 1
    changes: list[_Change] = []
    drawn = quantity("points")
    for key in keys:
        drawn = quantity(figure_unit(report_input, key), change=True)
        theirs = figure_stat(report_input, key, control, context.query.aspect)
        for arm in context.arm_order(arms):
            mine = figure_stat(report_input, key, arm, context.query.aspect)
            change = _descriptive(context, key, arm, mine, theirs, drawn)
            if change is not None:
                group = measure_words(report_input, key) if grouped else None
                changes.append(_with_group(change, group))
    return changes, drawn, (UNCORRECTED_NOTE,) if any(c.strong is not None for c in changes) else ()


def _with_group(change: _Change, group: str | None, label: str | None = None) -> _Change:
    return dataclasses.replace(change, label=label if label is not None else change.label, group=group)


def _slice_changes(context: ChartContext) -> tuple[dict[str, list[_Change]], Quantity, tuple[str, ...]]:
    """strata or tasks: per candidate, each slice's figure less the control's there, the slices in the query's
    selection (the largest strata first, to the limit)."""
    control = _control_or_refuse(context)
    slices = [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)]
    keys = slice_keys(context, slices)
    arms = [arm for arm in context.query.arms or context.look.arms if arm != control]
    by_arm: dict[str, list[_Change]] = {}
    drawn = quantity(figure_unit(context.report_input, keys[0]), change=True)
    for key in keys:
        drawn = quantity(figure_unit(context.report_input, key), change=True)
        for arm in context.arm_order(arms):
            for item in slices:
                name = slice_name(context, item)
                where = name + (f" · {measure_words(context.report_input, key)}" if len(keys) > 1 else "")
                change = _descriptive(
                    context,
                    key,
                    arm,
                    item.per_arm.get(arm, {}).get(key),
                    item.per_arm.get(control, {}).get(key),
                    drawn,
                    where=where,
                )
                if change is not None:
                    by_arm.setdefault(arm, []).append(_with_group(change, None, label=where))
    notes = (UNCORRECTED_NOTE,) if any(c.strong is not None for rows in by_arm.values() for c in rows) else ()
    return by_arm, drawn, notes


def _preference_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    preference = context.report_input.human_pref
    if preference is None:
        raise context.refuse("the input holds no human preference")
    drawn = quantity("percent")
    changes = []
    for row in context.rows():
        assert isinstance(row.item, PreferenceRow)
        item = row.item
        value = float(item.prefers_control)
        width = float(item.half_width) if item.half_width is not None else None
        tip = [item.name or item.id, figure_text(value, drawn)]
        changes.append(
            _Change(
                id=item.id,
                label=item.name or item.id,
                color=context.look.arm(preference.arm).token,
                value=value,
                interval=(value - width, value + width) if width is not None else None,
                strong=abs(value - _NO_PREFERENCE) > width if width is not None else None,
                note=figure_text(value, drawn),
                tip=tip,
            )
        )
    words = HUMAN_PREF_NOTE.format(
        control=context.look.arm(preference.control).name, arm=context.look.arm(preference.arm).name, n=preference.n
    )
    return changes, drawn, (words,)


def _lever_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """configs: each level of each factor against the factor's first level, averaged over every combination of the
    other factors that ran both."""
    report_input = context.report_input
    keys = list(dict.fromkeys(row.key for row in context.rows()))
    if len(keys) != 1:
        raise context.refuse(f"lever effects read one measure, and the query selects {keys}")
    (key,) = keys
    values: dict[str, float] = {}
    for row in context.rows():
        assert isinstance(row.item, Stat)
        read = stat_value(report_input, key, row.item)
        if read is not None:
            values[row.arm] = read[0]
    drawn = quantity(figure_unit(report_input, key), change=True)
    by_levels = {tuple(sorted(config.levels.items())): config.arm for config in report_input.configs}
    changes: list[_Change] = []
    for factor in report_input.factors:
        first = factor.levels[0].id
        for level in factor.levels[1:]:
            pairs = []
            for config in report_input.configs:
                if config.levels.get(factor.id) != level.id or config.arm not in values:
                    continue
                base = by_levels.get(tuple(sorted({**config.levels, factor.id: first}.items())))
                if base is not None and base in values:
                    pairs.append(values[config.arm] - values[base])
            if not pairs:
                continue
            change = sum(pairs) / len(pairs) * drawn.factor
            name = level.name or level.id
            changes.append(
                _Change(
                    id=f"{factor.id}={level.id}",
                    label=name,
                    color=None,
                    value=change,
                    interval=None,
                    strong=None,
                    note=figure_text(change, drawn),
                    tip=[name, figure_text(change, drawn)],
                    group=factor.name or factor.id,
                )
            )
    return changes, drawn, (LEVER_NOTE,)


# --- rows an analysis computes: two arms' difference, and the lever effects ----------------------------------------


def _difference_quantity(context: ChartContext, key: str) -> Quantity:
    """How a difference between two arms on one measure is drawn: a rate's in points, another's in its own unit."""
    return quantity(contrast_unit_of(context.report_input, key), change=True)


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _arm_tag(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).tag if arm in context.look.arms else arm


def _factor(figure: CitedFigure, drawn: Quantity) -> float:
    """What a computed figure's number is multiplied by to be drawn: a rate's change by the scale the figure states
    it on (a share's, 0 to 1, a hundred times, to the points the chart draws; one in points or percent as it is), so
    a lever's effect, which its key states as a share, draws in points as a comparison's does; any other by the
    chart's own factor."""
    if drawn.unit in (POINTS, PERCENT):
        if figure.scale == "share":
            return 100.0
        if figure.scale in ("points", "percent"):
            return 1.0
    return drawn.factor


def _figure_value(figure: CitedFigure | None, drawn: Quantity) -> float | None:
    if figure is None or not isinstance(figure.value, float):
        return None
    return float(figure.value) * _factor(figure, drawn)


def _figure_ends(figure: CitedFigure | None, drawn: Quantity) -> tuple[float, float] | None:
    if figure is None or not isinstance(figure.value, tuple):
        return None
    factor = _factor(figure, drawn)
    return float(figure.value[0]) * factor, float(figure.value[1]) * factor


def _analysed_change(
    *,
    row_id: str,
    label: str,
    short: str,
    value: float,
    interval: tuple[float, float] | None,
    drawn: Quantity,
    head: Sequence[str],
    strong: bool | None,
    group: str | None = None,
) -> _Change:
    """One row an analysis computed, as a change: bold where its interval clears zero, muted where it holds it."""
    tip = [*head, figure_text(value, drawn)]
    if interval is not None:
        tip[-1] += f", {interval_words(*interval, drawn)}"
    return _Change(
        id=row_id,
        label=label,
        color=None,
        value=value,
        interval=interval,
        strong=strong,
        note=figure_text(value, drawn),
        tip=tip,
        group=group,
        short_label=short,
    )


def _compare_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """compare: the query's first arm less its second at each place ``arms_compare`` reads, a row each."""
    rows = context.rows()
    if not rows:
        raise context.refuse("it selects no difference of two arms: name the two arms and one measure")
    key = rows[0].key
    first, second = context.query.arms
    drawn = _difference_quantity(context, key)
    changes: list[_Change] = []
    bases: list[str] = []
    for row in rows:
        item = row.item
        assert isinstance(item, ComparisonRow)
        value = _figure_value(item.difference, drawn)
        if value is None:
            continue
        place = str(item.name.value) if item.name is not None else OVERALL_WORD
        interval = _figure_ends(item.interval, drawn)
        if item.basis is not None:
            bases.append(compare_basis(context, item, key))
        changes.append(
            _analysed_change(
                row_id=row.label,
                label=place,
                short=place,
                value=value,
                interval=interval,
                drawn=drawn,
                head=[place],
                strong=item.excludes_zero,
            )
        )
    named = COMPARE_NOTE.format(
        first=_arm_name(context, first),
        second=_arm_name(context, second),
        measure=measure_words(context.report_input, key),
    )
    basis = tuple(dict.fromkeys(BASIS_NOTE.format(basis=basis) for basis in bases))
    return changes, drawn, (named, *basis, *narrow_notes(context))


def narrow_notes(context: ChartContext) -> tuple[str, ...]:
    """What a compare chart or table says, once, where a row's interval may be too narrow (``may_be_narrow``): each
    arm that ran more trials than cases, by its reader name and the trials and cases its key reads, said as a may,
    since a marked row's interval may count cases; then the unit only where the input states it (an unpaired row's
    ``counts``: trials or observations), and that a paired row's unit is unstated. Empty where no row is marked.

    :param context: The chart being resolved, over the compare source.
    :ptype context: ChartContext
    :return: The note, or none.
    :rtype: tuple[str, ...]
    """
    nested: dict[str, str] = {}
    stated: dict[str, None] = {}
    unstated = False
    for row in context.rows():
        item = row.item
        if not isinstance(item, ComparisonRow) or not item.may_be_narrow:
            continue
        unstated = unstated or item.counted_unstated
        stated.update(dict.fromkeys(str(figure.value) for figure in item.counts if figure.value != "cases"))
        by_arm: dict[str, list[str]] = {}
        for figure in item.nested:
            arm = figure.key.removeprefix(_SCOREBOARD).rpartition(".")[0]
            by_arm.setdefault(arm, []).append(figure.reads)
        for arm, (trials, cases, *_rest) in ((arm, reads) for arm, reads in by_arm.items() if len(reads) >= 2):
            nested.setdefault(
                arm, COMPARE_NESTED_TEMPLATE.format(arm=_arm_name(context, arm), trials=trials, cases=cases)
            )
    if not nested:
        return ()
    counted = COMPARE_STATED_COUNT.format(units=" or ".join(stated)) if stated else ""
    counted += COMPARE_UNSTATED_COUNT if unstated else ""
    return (COMPARE_NARROW_NOTE.format(nested="; ".join(nested.values()), counted=counted),)


#: How a scoreboard figure's key opens, before its arm: ``scoreboard.<arm>.<field>``.
_SCOREBOARD = "scoreboard."


def compare_basis(context: ChartContext, row: ComparisonRow, key: str) -> str:
    """How a compare row's interval was read, in reader words: the paired estimate (``paired_with``) names the
    candidate a contrast reads against the control and the measure by their reader names, never by id or key; the
    independent reading is the analysis's own words, which name neither.

    :param context: The chart being resolved, over the compare source.
    :ptype context: ChartContext
    :param row: One ``arms_compare`` row.
    :ptype row: ComparisonRow
    :param key: The measure it reads.
    :ptype key: str
    :return: The words; empty where the row reads no interval.
    :rtype: str
    """
    if row.basis is None:
        return ""
    if row.paired_with is None:
        return str(row.basis.value)
    candidate = row.paired_with.removeprefix("arm:")
    return PAIRED_BASIS_TEMPLATE.format(
        arm=_arm_name(context, candidate), measure=measure_words(context.report_input, key)
    )


def setting_value(report_input: ReportInput, setting: str, arm: str) -> str | None:
    """One arm's value of a setting as the input states it: the name its declared factor gives that level (a level
    matched by its id or name, as a lattice matches one), else the value itself, a number by the package's number rule
    (``1``, never ``1.0``); None where the arm states none.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param setting: The setting, by id.
    :ptype setting: str
    :param arm: The arm, by id.
    :ptype arm: str
    :return: The value as a reader reads it, or None.
    :rtype: str | None
    """
    value = arm_settings(report_input).get(arm, {}).get(setting)
    if value is None:
        return None
    spelled = value if isinstance(value, str) else format_number(float(value))
    factor = next((factor for factor in report_input.factors if factor.id == setting), None)
    for level in factor.levels if factor is not None else []:
        if spelled.casefold() in {level.id.casefold(), (level.name or level.id).casefold()}:
            return level.name or level.id
    return spelled


def lever_label(context: ChartContext, row: EffectRow | InteractionRow) -> tuple[str, str, str]:
    """A lever row's label, short label and group: an effect by the arm it moves from and the value it moves to (as
    :func:`setting_value` names it), under its setting; an interaction as its second move's effect less its first's,
    each move named as an effect is, under its setting and the other setting it crosses. Every row's label is its
    own: two effects differ in where they start or where they end, and two interactions in a move.

    :param context: The chart being resolved, over the levers source.
    :ptype context: ChartContext
    :param row: One ``lever_effects`` row.
    :ptype row: EffectRow | InteractionRow
    :return: ``(label, short label, group)``.
    :rtype: tuple[str, str, str]
    """
    arms = lever_arms(row, context.report_input)

    def moved(start: str, end: str, *, short: bool) -> str:
        value = setting_value(context.report_input, row.setting, end)
        first = _arm_tag(context, start) if short else _arm_name(context, start)
        return LEVER_MOVE_TEMPLATE.format(start=first, value=value if value is not None else _arm_name(context, end))

    if isinstance(row, EffectRow):
        start, end = arms
        label = moved(start, end, short=False)
        short = LEVER_MOVE_TEMPLATE.format(start=_arm_tag(context, start), value=_arm_tag(context, end))
        return label, short, LEVER_EFFECT_GROUP.format(setting=humanize("dimension", row.setting))
    first_from, first_to, second_from, second_to = (arms + ["", "", "", ""])[:4]
    label = LEVER_CROSS_TEMPLATE.format(
        second=moved(second_from, second_to, short=False), first=moved(first_from, first_to, short=False)
    )
    short = LEVER_CROSS_TEMPLATE.format(
        second=moved(second_from, second_to, short=True), first=moved(first_from, first_to, short=True)
    )
    group = LEVER_CROSS_GROUP.format(
        setting=humanize("dimension", row.setting), across=humanize("dimension", row.across)
    )
    return label, short, group


def _lever_rows(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """levers: each effect, then each interaction, ``lever_effects`` reads on the query's one measure and place."""
    rows = context.rows()
    if not rows:
        raise context.refuse("it selects no lever effect: name one measure")
    key = rows[0].key
    drawn = _difference_quantity(context, key)
    changes: list[_Change] = []
    for row in rows:
        item = row.item
        assert isinstance(item, EffectRow | InteractionRow)
        value = _figure_value(item.figures[0], drawn)
        if value is None:
            continue
        interval = _figure_ends(item.figures[1], drawn) if len(item.figures) > 1 else None
        label, short, group = lever_label(context, item)
        if isinstance(item, InteractionRow):
            strong: bool | None = item.verdict in ("reverses", "differs") if item.verdict != "no_interval" else None
        else:
            strong = None if interval is None else interval[0] > 0 or interval[1] < 0
        changes.append(
            _analysed_change(
                row_id=row.label,
                label=label,
                short=short,
                value=value,
                interval=interval,
                drawn=drawn,
                head=[group, label],
                strong=strong,
                group=group,
            )
        )
    if len({change.group for change in changes}) == 1:
        changes = [_with_group(change, None) for change in changes]
    crossed = LEVERS_INTERACTION_NOTE if any(isinstance(row.item, InteractionRow) for row in rows) else ""
    note = LEVERS_NOTE.format(measure=measure_words(context.report_input, key), interactions=crossed)
    return changes, drawn, (f"{note} {LEVERS_READ_NOTE}",)


# --- the charts ----------------------------------------------------------------------------------------------------


def _forest_row(change: _Change, context: ChartContext) -> ForestRow:
    if change.interval is None:
        raise context.refuse(f"row {change.label!r} carries no interval, and a forest draws intervals")
    return ForestRow(
        id=change.id,
        label=change.label,
        short_label=change.short_label,
        value=change.value,
        lo=change.interval[0],
        hi=change.interval[1],
        color=change.color,
        strong=change.strong or None,
        note=change.note,
        tip=change.tip,
        group=change.group,
    )


def _ends(changes: Sequence[_Change], reference: ChartReference | None, *, intervals: bool) -> Iterator[float]:
    for change in changes:
        yield change.value
        if intervals and change.interval is not None:
            yield from change.interval
    if reference is not None:
        yield reference.value


def _forest(context: ChartContext) -> Drawn:
    source = context.query.source
    refusal = unit_refusal(context.spec.kind, context.query, context.report_input)
    if refusal:
        raise context.refuse(refusal)
    notes: tuple[str, ...] = ()
    if source == "readings":
        changes, drawn = _reading_changes(context)
    elif source == "contrasts":
        changes, drawn, notes = _contrast_changes(context)
    else:
        raise context.refuse("a forest draws readings or contrasts")
    if not context.level:
        raise context.refuse("a forest draws intervals, and the chart states no level for them")
    reference = reference_line(context, drawn)
    domain = context.spec.domain or centred_domain(_ends(changes, reference, intervals=True), drawn)
    data = ForestChart(
        rows=[_forest_row(change, context) for change in changes],
        domain=(domain[0], domain[1]),
        fmt=drawn.fmt,
        level=context.level,
        reference=reference,
        emphasis=context.emphasis,
    )
    return Drawn(chart_block(context, data), notes)


def _bar_row(change: _Change, level: str) -> BarRow:
    interval = change.interval if level else None
    return BarRow(
        id=change.id,
        label=change.label,
        short_label=change.short_label,
        value=change.value,
        lo=interval[0] if interval is not None else None,
        hi=interval[1] if interval is not None else None,
        color=change.color,
        strong=True if change.strong else None,
        muted=True if change.strong is False else None,
        note=change.note,
        tip=change.tip,
        group=change.group,
    )


def _diverging(context: ChartContext) -> Drawn:
    source = context.query.source
    refusal = unit_refusal(context.spec.kind, context.query, context.report_input)
    if refusal:
        raise context.refuse(refusal)
    notes: tuple[str, ...] = ()
    centre = 0.0
    panels: list[_Panel] | None = None
    if source == "readings":
        changes, drawn = _reading_changes(context)
        changes = [_with_group(change, None) for change in changes]
    elif source == "contrasts":
        changes, drawn, notes = _contrast_changes(context, by_aspect=context.query.small_multiples_by == "aspect")
        if any(change.panel is not None for change in changes):
            by_aspect: dict[str, list[_Change]] = {}
            for change in changes:
                by_aspect.setdefault(change.panel or "", []).append(change)
            panels = [_Panel(title=name, tag=None, color=None, rows=rows) for name, rows in by_aspect.items()]
    elif source in ("measures", "judged"):
        changes, drawn, notes = _figure_changes(context)
    elif source in ("strata", "tasks"):
        by_arm, drawn, notes = _slice_changes(context)
        changes = [change for rows in by_arm.values() for change in rows]
        by = context.query.small_multiples_by
        if by == "arm" and len(by_arm) > 1:
            panels = [_arm_panel(context, arm, rows) for arm, rows in by_arm.items()]
        elif by in ("task", "aspect") and len({change.label for change in changes}) > 1:
            by_slice: dict[str, list[_Change]] = {}
            for change in changes:
                arm_label = context.look.arm(change.id).label
                by_slice.setdefault(change.label, []).append(_with_group(change, None, label=arm_label))
            panels = [_Panel(title=name, tag=None, color=None, rows=rows) for name, rows in by_slice.items()]
        else:
            grouped = len(by_arm) > 1
            changes = [
                _with_group(change, context.look.arm(arm).label if grouped else None, label=change.label)
                for arm, rows in by_arm.items()
                for change in rows
            ]
    elif source == "human_pref":
        changes, drawn, notes = _preference_changes(context)
        centre = _NO_PREFERENCE
    elif source == "configs":
        changes, drawn, notes = _lever_changes(context)
    elif source == "compare":
        changes, drawn, notes = _compare_changes(context)
    elif source == "levers":
        changes, drawn, notes = _lever_rows(context)
    else:
        raise context.refuse(
            "diverging bars draw readings, contrasts, measures, judged, strata, tasks, configs, human preference, "
            "two arms compared or lever effects"
        )
    if not changes:
        raise context.refuse("it selects no change to draw")
    level = context.level
    if centre == _NO_PREFERENCE:
        reference: ChartReference | None = ChartReference(value=centre, kind="no_preference")
    else:
        reference = reference_line(context, drawn)
    ends = list(_ends(changes, reference, intervals=bool(level)))
    domain = context.spec.domain or centred_domain(ends, drawn, centre=centre)
    read_at = drawn.unit if source in _COMPUTED else ""
    if panels is not None:
        data = BarsChart(
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            level=level or None,
            reference=reference,
            emphasis=context.emphasis,
            panels=[
                BarsPanel(
                    title=panel.title,
                    tag=panel.tag,
                    color=panel.color,
                    rows=[_bar_row(change, level) for change in panel.rows],
                )
                for panel in panels
            ],
        )
    else:
        data = BarsChart(
            rows=[_bar_row(change, level) for change in changes],
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            level=level or None,
            reference=reference,
            emphasis=context.emphasis,
        )
    return Drawn(chart_block(context, data, read_at=read_at), notes)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"forest": _forest, "diverging_bars": _diverging}
