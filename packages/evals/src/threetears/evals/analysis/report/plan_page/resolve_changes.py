"""The charts of change: a forest (each change against the control with its interval) and diverging bars (deltas
about a centre).

**forest** reads the pipeline's own tests:

- *readings*: a row per candidate, its change and the interval the reading was read on, bold where separated
  (the interval clear of zero), its note in GENERATOR §3's words (worse, better, not separated);
- *contrasts*: a row per arm and measure, grouped under the measure's reader name where the chart reads several, a
  rate's delta in points, bold where the contrast regressed or improved. A comparison the query asks for that the
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
  averaged over every combination of the other factors that holds both: descriptive, uncorrected.

Every interval is drawn only where the chart states a level; every note is written by the package's number rule.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from threetears.evals.analysis.report.charts import BarRow, BarsChart, BarsPanel, ChartReference, ForestChart, ForestRow
from threetears.evals.analysis.report.plan_page.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    chart_block,
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
    HUMAN_PREF_NOTE,
    LEVER_NOTE,
    UNCORRECTED_NOTE,
    change_words,
)
from threetears.evals.analysis.report_input import Contrast, PreferenceRow, Reading, Stat, Stratum, Task

__all__ = ["RESOLVERS"]

#: The contrast verdicts that separate an arm from the control in a direction.
_DIRECTED = frozenset({"improved", "regressed"})

#: The no-preference point a human preference is read about, in percent.
_NO_PREFERENCE = 50.0


@dataclass(frozen=True)
class _Change:
    """One drawn change: its row's id, label, colour and group, its value and interval, and how it reads."""

    id: str
    label: str
    color: str | None
    value: float
    interval: tuple[float, float] | None
    strong: bool | None
    note: str
    tip: list[str]
    group: str | None = None


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


def _contrast_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """Each selected contrast as a change, a rate's delta in points, grouped by measure where there are several; and
    the caption naming each asked pair the input holds no contrast of."""
    report_input = context.report_input
    contrasts = [row.item for row in context.rows() if isinstance(row.item, Contrast)]
    keys = list(dict.fromkeys([*context.query.measures, *(contrast.measure for contrast in contrasts)]))
    grouped = len({contrast.measure for contrast in contrasts}) > 1
    drawn: Quantity | None = None
    changes: list[_Change] = []
    for key in keys:
        on_key = {contrast.arm: contrast for contrast in contrasts if contrast.measure == key}
        for arm in context.arm_order(on_key):
            contrast = on_key[arm]
            if contrast.delta is None:
                continue
            unit = figure_unit(report_input, key, kind=contrast.kind)
            here = quantity(unit, change=True)
            drawn = drawn or here
            look = context.look.arm(arm)
            value = float(contrast.delta) * here.factor
            interval = (
                (float(contrast.interval[0]) * here.factor, float(contrast.interval[1]) * here.factor)
                if contrast.interval is not None
                else None
            )
            words = measure_words(report_input, key, kind=contrast.kind)
            direction = {"regressed": "worse", "improved": "better"}.get(contrast.verdict, "")
            tip = [look.name, words, figure_text(value, here)]
            if interval is not None:
                tip[2] += f", {interval_words(*interval, here)}"
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
                    group=words if grouped else None,
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
    return _Change(
        id=change.id,
        label=label if label is not None else change.label,
        color=change.color,
        value=change.value,
        interval=change.interval,
        strong=change.strong,
        note=change.note,
        tip=change.tip,
        group=group,
    )


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


# --- the charts ----------------------------------------------------------------------------------------------------


def _forest_row(change: _Change, context: ChartContext) -> ForestRow:
    if change.interval is None:
        raise context.refuse(f"row {change.label!r} carries no interval, and a forest draws intervals")
    return ForestRow(
        id=change.id,
        label=change.label,
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
        changes, drawn, notes = _contrast_changes(context)
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
    else:
        raise context.refuse(
            "diverging bars draw readings, contrasts, measures, judged, strata, tasks, configs or human preference"
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
    return Drawn(chart_block(context, data), notes)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"forest": _forest, "diverging_bars": _diverging}
