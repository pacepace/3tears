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
  (:func:`~threetears.evals.analysis.report.parts.charting.unit_refusal`);
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

Asked as diverging bars, changes per slice (*strata*, *tasks*), per place (*compare*) and per lever (*configs*,
*levers*) are drawn as dots with intervals (``dots``, GENERATOR §5.1), most of their rows carrying no interval: a
row with none is an open mark, which a caption says, faded where the descriptive rule cannot tell it from zero, and a
row with an interval is never faded, so no row reads surer than the one that has an interval; groups and panels are the
bars'.

Every change chart names its zero: no change against the control as a chart row names it, against the second of two
arms compared, against each factor's first level, or from each lever move's start (and, where interactions are drawn,
no difference between the moves); a no-preference point is named by its own line. The level each change is from, as
the input states it (a pass rate's passes of its trials, a figure's mean over its cases; money on the chart's one
ruler), is on the figure: in the zero label where every row shares it, in each group's head where each measure's rows
do, else in each row's note (``+1.7 pts from 88%``). Under its axis a chart over measures that are better one way
says which (``← worse · better →``). A forest's margin line says its band is no worse than the margin, and a forest
of contrasts corrected together says what its level holds as a family.

Every interval is drawn only where the chart states a level; every note is written by the package's number rule. A
chart of what an analysis computes (compare, levers) spells its values table at its quantity's precision, a point to
one decimal, as the rows' keys read (:func:`~threetears.evals.analysis.report.parts.charting.chart_block`).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from threetears.evals.analysis.analyst.cited import CitedFigure
from threetears.evals.analysis.analyst.compare import ComparisonRow
from threetears.evals.analysis.analyst.levers import EffectRow, InteractionRow
from threetears.evals.analysis.report.charts import BarRow, BarsChart, BarsPanel, ChartReference, ForestChart, ForestRow
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.parts.charting import (
    ChartContext,
    Drawn,
    KindResolver,
    money_drawn,
    contrast_groups,
    contrast_unit_of,
    contrast_words,
    figure_kind,
    figure_stat,
    measure_words,
    money_words,
    missing_note,
    reference_line,
    separated,
    slice_keys,
    slice_name,
    spread,
    stat_value,
    unit_refusal,
)
from threetears.evals.analysis.report.parts.look import Quantity, centred_domain
from threetears.evals.analysis.report.parts.words import (
    BASIS_NOTE,
    BETTER_WORDS,
    CASES_MEAN_TEMPLATE,
    COMPARE_NARROW_NOTE,
    COMPARE_NESTED_TEMPLATE,
    COMPARE_STATED_COUNT,
    COMPARE_UNSTATED_COUNT,
    COMPARE_NOTE,
    FROM_LEVEL_TEMPLATE,
    GROUP_LEVEL_TEMPLATE,
    HUMAN_PREF_NOTE,
    LEVER_CROSS_GROUP,
    LEVER_CROSS_TEMPLATE,
    LEVER_EFFECT_GROUP,
    LEVER_MOVE_TEMPLATE,
    LEVER_NOTE,
    LEVERS_INTERACTION_NOTE,
    LEVERS_NOTE,
    LEVERS_READ_NOTE,
    LEVEL_COUNT_TEMPLATE,
    MARGIN_BAND_TEMPLATE,
    OPEN_MARKS_NOTE,
    OVERALL_WORD,
    PAIRED_BASIS_TEMPLATE,
    PAIRED_CASES,
    TRIALS_COUNT_TEMPLATE,
    UNCORRECTED_NOTE,
    ZERO_FIRST_LEVELS,
    ZERO_LEVEL_TEMPLATE,
    ZERO_STARTS,
    ZERO_STARTS_CROSS,
    ZERO_TEMPLATE,
    change_words,
)
from threetears.evals.analysis.quantities import MONEY, PERCENT, POINTS
from threetears.evals.analysis.report.figure_keys import arm_settings, correction_words, pairing_contrast, figure_unit
from threetears.evals.analysis.report_input import Contrast, PreferenceRow, Reading, ReportInput, Stat, Stratum, Task
from threetears.evals.analysis.report_margins import aspect_board
from threetears.evals.analysis.report.rows import lever_arms, query_aspect
from threetears.evals.analysis.style import lever_reader_name

__all__ = ["RESOLVERS", "higher_is_better"]

#: The sources whose rows an analysis computes, unrounded: a values table spells them at their quantity's precision.
_COMPUTED = frozenset({"compare", "levers"})

#: The contrast verdicts that separate an arm from the control in a direction.
_DIRECTED = frozenset({"improved", "regressed"})

#: The no-preference point a human preference is read about, in percent.
_NO_PREFERENCE = 50.0

#: The sources whose changes the diverging chart draws as dots with intervals, never as bars (GENERATOR §5.1): per slice,
#: per place two arms are compared at, and per lever, where most rows carry no interval, so a bar's solid length would
#: read surer than the one row that has one. Bars stay for the pipeline's own changes and each arm's figure less the
#: control's.
_AS_DOTS = frozenset({"strata", "tasks", "compare", "configs", "levers"})


@dataclass(frozen=True)
class _Change:
    """One drawn change: its row's id, label, colour, group and panel, its value and interval, and how it reads; the
    level of what it is compared with (``base``, spelled, and what that level counts, ``count``) where the input
    holds it; and the measure it is a change of (``measure`` and ``kind``; both empty for a preference's, of none).
    Where the page's forms decide the measure's digits, its value's and its interval's ends' texts as they print them
    (``texts``), which its row carries, so its values table and its labels read as every other surface of the page."""

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
    base: str | None = None
    count: str = ""
    measure: str = ""
    kind: str | None = ""
    texts: tuple[str, str | None, str | None] | None = None


class _HasEnds(Protocol):
    """An interval as the page's forms print it, by its two ends' digits."""

    @property
    def low_digits(self) -> str: ...

    @property
    def high_digits(self) -> str: ...


def _end_digits(ends: _HasEnds | None) -> tuple[str, str] | None:
    """An interval's two ends' digits as the page's forms printed them; None for no interval."""
    return None if ends is None else (ends.low_digits, ends.high_digits)


def _decided(
    context: ChartContext,
    drawn: Quantity,
    measure: str,
    kind: str | None,
    said: str,
    ends: tuple[str, str] | None,
) -> tuple[str, str | None, str | None] | None:
    """A row's value and interval ends as the page's forms print them, where they decide its measure's digits; None
    for money, which the chart settles on one ruler after its rows are drawn, and for a measure the forms hold no
    digits of, whose row its chart's format spells."""
    if drawn.unit == MONEY or not measure:
        return None
    try:
        form = context.forms.measure(measure, kind=kind)
    except KeyError:
        # NOSILENT: a measure the page states no form of (a chart of a measure the input names only in a slice) keeps
        # its chart's format.
        return None
    if form.places is None:
        return None
    return (said, None if ends is None else ends[0], None if ends is None else ends[1])


def _in_one_form(
    context: ChartContext, keys: Sequence[str], changes: list[_Change], drawn: Quantity
) -> tuple[list[_Change], Quantity]:
    """A chart's money changes on one ruler (#993): the ruler of the amounts they compare (every mean the input holds
    of the chart's measures, by the one owner of money's scale), so a change chart scales as its source set does and
    never because an interval end sits near zero; in the changes' own form, capped by their widest interval. Each
    change is redrawn at that ruler's factor, its note and tip respelled in it, and its measure's words name what its
    money is drawn per. Changes of any other unit are returned as they are.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param keys: The measures the changes are on.
    :ptype keys: Sequence[str]
    :param changes: The changes, drawn at ``drawn``.
    :ptype changes: list[_Change]
    :param drawn: Their unit's change quantity, as :func:`quantity` makes it.
    :ptype drawn: Quantity
    :return: The changes and their quantity.
    :rtype: tuple[list[_Change], Quantity]
    """
    report_input, forms = context.report_input, context.forms
    money = _money_keys(context, keys)
    if drawn.unit != MONEY or not money or not changes:
        return changes, drawn
    amounts = _amounts_drawn(context, keys)
    if amounts is None:
        return changes, drawn
    scale = amounts.factor / drawn.factor
    settled = forms.chart_changes(
        amounts,
        [change.value * scale for change in changes],
        [abs(change.interval[1] - change.interval[0]) * scale for change in changes if change.interval is not None],
        measure=money[0],
        kind=changes[0].kind,
    )
    names = {measure_words(report_input, key): money_words(report_input, key, settled) for key in money}

    def respelled(text: str, numbers: Sequence[float], change: _Change) -> str:
        for number in numbers:
            old = forms.chart_note(number, drawn, measure=change.measure, kind=change.kind).text
            new = forms.chart_note(number * scale, settled, measure=change.measure, kind=change.kind).text
            text = text.replace(old, new)
        return relabelled(text)

    moved = {old: new for old, new in names.items() if old != new}
    # One pass, the longest name first, so a measure named inside another's name (``cost`` in ``production replicating
    # cost``) is never replaced twice.
    pattern = re.compile("|".join(re.escape(old) for old in sorted(moved, key=len, reverse=True))) if moved else None

    def relabelled(text: str | None) -> str:
        if not text or pattern is None:
            return text or ""
        return pattern.sub(lambda found: moved[found.group(0)], text)

    redrawn = []
    for change in changes:
        numbers = (change.value, *(change.interval or ()))
        redrawn.append(
            dataclasses.replace(
                change,
                value=change.value * scale,
                interval=(change.interval[0] * scale, change.interval[1] * scale) if change.interval else None,
                note=respelled(change.note, numbers, change),
                tip=[respelled(line, numbers, change) for line in change.tip],
                label=relabelled(change.label),
                group=relabelled(change.group) if change.group is not None else None,
            )
        )
    return redrawn, settled


def _drawn_money(context: ChartContext, drawn: Quantity) -> list[tuple[str, Quantity]]:
    """Each measure a change chart reads, with the one quantity its changes are drawn as (the one-unit rule)."""
    keys = dict.fromkeys([*context.query.measures, *(row.key for row in context.rows() if row.key)])
    return [(key, drawn) for key in keys]


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


# --- what a change is compared with: its level, and which way is better -------------------------------------------


def higher_is_better(report_input: ReportInput, key: str, *, kind: str | None = None) -> bool | None:
    """Which way a measure or judged dimension is better: the pass rate (a contrast of kind ``trial``) and a judged
    dimension up, a measure as it declares, else as the one direction its criteria give; None where nothing says.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :param kind: A contrast's kind, where the key is a contrast's; None otherwise.
    :ptype kind: str | None
    :return: True where higher is better, False where lower is, None where the input does not say.
    :rtype: bool | None
    """
    if kind == "trial":
        return True
    measure = report_input.measures.get(key)
    if measure is not None and kind != "judged" and measure.higher_is_better is not None:
        return measure.higher_is_better
    if key in report_input.judged and (measure is None or kind == "judged"):
        return True
    directions = {criterion.direction for criterion in report_input.criteria if criterion.measure == key}
    return directions.pop() == "higher_is_better" if len(directions) == 1 else None


def _better(context: ChartContext) -> tuple[str, str] | None:
    """The words under a change chart's two ends: the one direction every measure it draws is better in, a reading's
    the pass rate's; None where they differ, or one does not say, or the chart is about a no-preference point."""
    source, report_input = context.query.source, context.report_input
    if source == "human_pref":
        return None
    if source == "readings":
        return BETTER_WORDS[True]
    kinds = {contrast.measure: contrast.kind for contrast in report_input.contrasts} if source == "contrasts" else {}
    if source in ("strata", "tasks"):
        keys = slice_keys(context, [row.item for row in context.rows() if isinstance(row.item, Stratum | Task)])
    else:
        keys = list(dict.fromkeys([*context.query.measures, *(row.key for row in context.rows() if row.key)]))
    ups = {higher_is_better(report_input, key, kind=kinds.get(key)) for key in keys}
    if len(ups) != 1:
        return None
    up = ups.pop()
    return None if up is None else BETTER_WORDS[up]


def _level_quantity(context: ChartContext, key: str, keys: Sequence[str]) -> Quantity:
    """How a level (not a change) of one measure is spelled on a change chart: money on the one ruler the chart's
    money is drawn on (:func:`_amounts_drawn`), so a level and the changes from it read on one scale; any other unit
    as its figure reads (a share as a percent)."""
    unit, kind = figure_unit(context.report_input, key), figure_kind(context, key)
    if context.forms.chart_quantity(unit, measure=key, kind=kind).unit == MONEY:
        changes = context.forms.chart_quantity(unit, change=True, measure=key, kind=kind)
        return _levels_drawn(context, keys or [key], changes)
    return context.forms.chart_quantity(unit, measure=key, kind=kind)


def _passes(context: ChartContext, aspect: str | None) -> tuple[str | None, str]:
    """The control's pass rate on an aspect (the lead's where None) as its scoreboard counts it, its passes over its
    usable trials beside it: what a reading or a pass-rate contrast is compared with."""
    control = context.look.control
    try:
        board = aspect_board(context.report_input, aspect)
    except KeyError:
        # NOSILENT: an aspect the input does not hold has no scoreboard; the chart names the control without a level.
        return None, ""
    row = board.get(control.id) if control is not None else None
    if row is None or row.passed is None or not row.usable:
        return None, ""
    # The control's pass rate, its passes and trials counts of no measure.
    forms, rate = context.forms, TRIAL_PASS_NAME
    drawn = forms.chart_quantity("share", measure=rate, kind="trial")
    return (
        forms.chart_note(row.passed / row.usable * drawn.factor, drawn, measure=rate, kind="trial").text,
        TRIALS_COUNT_TEMPLATE.format(
            passed=context.forms.count_alone(row.passed).text, trials=context.forms.count_alone(row.usable).text
        ),
    )


def _figure_base(
    context: ChartContext, key: str, arm: str, aspect: str, keys: Sequence[str]
) -> tuple[str | None, str]:
    """An arm's level on one measure as the input states it, and what it counts (a mean over its cases): what each
    candidate's change on that measure is compared with where the arm is the control."""
    stat = figure_stat(context.report_input, key, arm, aspect)
    read = stat_value(context.report_input, key, stat) if stat is not None else None
    if stat is None or read is None:
        return None, ""
    level = _level_quantity(context, key, keys)
    count = (
        CASES_MEAN_TEMPLATE.format(cases=context.forms.count_alone(stat.n_cases).text)
        if stat.n_cases and read[1] == "mean"
        else ""
    )
    said = context.forms.chart_level(read[0] * level.factor, level, measure=key, kind=figure_kind(context, key)).text
    return said, count


def _reference_arm(context: ChartContext) -> tuple[str, str] | None:
    """What a change chart's rows are compared with, as a chart row names it and by its name: the query's second arm
    for two arms compared, else the control; None for lever effects and a preference, whose zero is no one arm."""
    source = context.query.source
    if source in ("configs", "levers", "human_pref"):
        return None
    if source == "compare":
        second = context.query.arms[1]
        return _arm_label(context, second), _arm_name(context, second)
    control = context.look.control
    return (control.label, control.name) if control is not None else None


def _one_base(changes: Sequence[_Change]) -> tuple[str, str] | None:
    """The one level every change is compared with, and what it counts (where they all count it alike); None where
    the changes are compared with several, or with one the input does not state."""
    bases = {change.base for change in changes}
    if len(bases) != 1 or None in bases or not changes:
        return None
    counts = {change.count for change in changes}
    return str(bases.pop()), counts.pop() if len(counts) == 1 else ""


def _zero_label(context: ChartContext, changes: Sequence[_Change]) -> str | None:
    """What a change chart's zero line is labelled with, so a reader cold to the page reads zero as what the rows are
    compared with: no change against the control as a chart row names it (``P · September``), or against the second
    of two arms compared, with its level where every row shares one; against each factor's first level, or the arm
    each lever's move starts at; None about a no-preference point, whose own line names it, or with no control."""
    source = context.query.source
    if source == "configs":
        return ZERO_FIRST_LEVELS
    if source == "levers":
        crossed = any(isinstance(row.item, InteractionRow) for row in context.rows())
        return ZERO_STARTS_CROSS if crossed else ZERO_STARTS
    reference = _reference_arm(context)
    if reference is None:
        return None
    shared = _one_base(changes)
    if shared is None:
        return ZERO_TEMPLATE.format(reference=reference[0])
    base, count = shared
    level = LEVEL_COUNT_TEMPLATE.format(level=base, count=count) if count else base
    return ZERO_LEVEL_TEMPLATE.format(reference=reference[0], level=level)


def _baselined(context: ChartContext, changes: list[_Change]) -> list[_Change]:
    """The changes, each with what it is compared with in sight: where every row shares one level the zero line says
    it (:func:`_zero_label`) and the rows are as they are; where each group of rows shares one (a measure's, on a chart
    of several) the group's head carries it; else each change's note carries its own (``+1.7 pts from 88%``)."""
    reference = _reference_arm(context)
    if reference is None or _one_base(changes) is not None or all(change.base is None for change in changes):
        return changes
    groups = list(dict.fromkeys(change.group for change in changes))
    if None not in groups and all(_one_base([c for c in changes if c.group == group]) for group in groups):
        return [
            dataclasses.replace(
                change, group=GROUP_LEVEL_TEMPLATE.format(group=change.group, reference=reference[1], level=change.base)
            )
            for change in changes
        ]
    return [
        dataclasses.replace(change, note=FROM_LEVEL_TEMPLATE.format(change=change.note, level=change.base))
        if change.base is not None
        else change
        for change in changes
    ]


# --- readings and contrasts: the pipeline's own tests -----------------------------------------------------------


def _reading_changes(context: ChartContext) -> tuple[list[_Change], Quantity]:
    # A reading is the pipeline's change in the pass rate.
    forms, rate = context.forms, TRIAL_PASS_NAME
    drawn = forms.chart_quantity("points", change=True, measure=rate, kind="trial")
    aspect = query_aspect(context.query.aspect, context.report_input)
    base, count = _passes(context, aspect.id if aspect is not None else None)
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
        printed = forms.chart_note(float(reading.change), drawn, measure=rate, kind="trial")
        tip = [look.name, printed.text]
        ends = None
        if low is not None and high is not None:
            ends = forms.chart_interval(low, high, drawn, measure=rate, kind="trial")
            tip[1] += f", {ends.beside}"
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
                base=base,
                count=count,
                measure=rate,
                kind="trial",
                texts=_decided(context, drawn, rate, "trial", printed.digits, _end_digits(ends)),
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
            here = context.forms.chart_quantity(unit, change=True, measure=group.key, kind=contrast.kind)
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
            printed = context.forms.chart_note(value, here, measure=group.key, kind=contrast.kind)
            said = printed.text
            tip = [look.name, words, said]
            ends = None
            if interval is not None:
                ends = context.forms.chart_interval(*interval, here, measure=group.key, kind=contrast.kind)
                tip[2] += f", {ends.beside}"
            shown = contrast_words(report_input, contrast, None) if panelled else words
            if contrast.kind == "trial":
                base, count = _passes(context, contrast.aspect)
            else:
                control = contrast.control or (context.look.control.id if context.look.control else "")
                base, count = _figure_base(
                    context, group.key, control, contrast.aspect or "", [one.key for one in groups]
                )
            changes.append(
                _Change(
                    id=arm,
                    label=look.label,
                    color=look.token,
                    value=value,
                    interval=interval,
                    strong=contrast.verdict in _DIRECTED,
                    note=" ".join(part for part in (said, change_words(direction)) if part),
                    tip=tip,
                    group=shown if grouped else None,
                    panel=named if panelled else None,
                    base=base,
                    count=count,
                    measure=group.key,
                    kind=contrast.kind,
                    texts=_decided(context, here, group.key, contrast.kind, printed.digits, _end_digits(ends)),
                )
            )
    keys = [group.key for group in groups]
    # A chart that drew no contrast has an axis of no measure.
    drawn = drawn or context.forms.chart_quantity("points")
    return (*_in_one_form(context, keys, changes, drawn), _missing_contrasts(context, contrasts))


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
        f"{context.look.arm(arm).name}, {measure_words(context.report_input, key, kind=kinds.get(key), prose=True)}"
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
    context: ChartContext,
    key: str,
    arm: str,
    mine: Stat | None,
    theirs: Stat | None,
    drawn: Quantity,
    where: str = "",
    *,
    keys: Sequence[str] = (),
) -> _Change | None:
    """One arm less the control on one figure: the pipeline's own change where the arm's figure states only that,
    else the difference of the two values, bold or muted by the descriptive rule; None where neither is held. The two
    levels its tip compares are spelled on the ruler the chart settles every measure it draws (``keys``) on
    (:func:`_levels_drawn`), as the chart's words
    state it, never at the change's factor before the chart settles its ruler."""
    if mine is None:
        return None
    look = context.look.arm(arm)
    head = [where] if where else []
    forms, kind = context.forms, figure_kind(context, key)
    if mine.change is not None and (theirs is None or stat_value(context.report_input, key, mine) is None):
        change = float(mine.change) * drawn.factor
        printed = forms.chart_note(change, drawn, measure=key, kind=kind)
        said = printed.text
        return _Change(
            id=arm,
            label=look.label,
            color=look.token,
            value=change,
            interval=None,
            strong=None,
            note=said,
            tip=[*head, look.name, said],
            measure=key,
            kind=kind,
            texts=_decided(context, drawn, key, kind, printed.digits, None),
        )
    mine_value = stat_value(context.report_input, key, mine)
    theirs_value = stat_value(context.report_input, key, theirs) if theirs is not None else None
    if mine_value is None or theirs_value is None or theirs is None:
        return None
    change = (mine_value[0] - theirs_value[0]) * drawn.factor
    spreads = [spread(mine), spread(theirs)]
    read = separated(change, [one * drawn.factor if one is not None else None for one in spreads])
    control = context.look.arm(_control_or_refuse(context))
    # A level reads in its own unit (a share as a percent), money on the chart's one ruler, never at the change's.
    level = _level_quantity(context, key, keys)
    base = forms.chart_level(theirs_value[0] * level.factor, level, measure=key, kind=kind).text
    mine_level = forms.chart_level(mine_value[0] * level.factor, level, measure=key, kind=kind).text
    # The bar's value as its note prints it, at its measure's precision, never at the float's four figures.
    printed = forms.chart_note(change, drawn, measure=key, kind=kind)
    said = printed.text
    tip = [*head, f"{control.name} {base} · {look.name} {mine_level}", said]
    return _Change(
        id=arm,
        label=look.label,
        color=look.token,
        value=change,
        interval=None,
        strong=read,
        note=said,
        tip=tip,
        base=base,
        count=(
            CASES_MEAN_TEMPLATE.format(cases=forms.count_alone(theirs.n_cases).text)
            if theirs.n_cases and theirs_value[1] == "mean"
            else ""
        ),
        measure=key,
        kind=kind,
        texts=_decided(context, drawn, key, kind, printed.digits, None),
    )


def _levels_drawn(context: ChartContext, keys: Sequence[str], drawn: Quantity) -> Quantity:
    """How the levels a change compares are drawn: money on the one ruler the chart settles its changes on
    (:func:`_amounts_drawn` over every money measure it draws), so a tip's two levels differ by its bar; any other unit
    as the change is drawn."""
    if drawn.unit != MONEY:
        return drawn
    amounts = _amounts_drawn(context, keys)
    return amounts if amounts is not None else drawn


def _money_keys(context: ChartContext, keys: Sequence[str]) -> list[str]:
    """The measures among a chart's that are money."""
    return [key for key in keys if _level_unit(context, key) == MONEY]


def _level_unit(context: ChartContext, key: str) -> str:
    """The unit a chart draws one measure's levels in."""
    unit = figure_unit(context.report_input, key)
    return context.forms.chart_quantity(unit, measure=key, kind=figure_kind(context, key)).unit


def _amounts_drawn(context: ChartContext, keys: Sequence[str]) -> Quantity | None:
    """The one ruler a change chart's money is drawn on: the one owner of money's scale over every mean the input
    holds of every money measure the chart draws; None where it draws no money."""
    money = _money_keys(context, keys)
    if not money:
        return None
    return context.forms.chart_amounts(context.report_input, money)


def _figure_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    """measures or judged: each candidate's figure less the control's, per measure."""
    control = _control_or_refuse(context)
    report_input = context.report_input
    keys = list(dict.fromkeys(row.key for row in context.rows()))
    arms = list(dict.fromkeys(row.arm for row in context.rows() if row.arm != control))
    grouped = len(keys) > 1
    changes: list[_Change] = []
    # A chart that selects no measure has an axis of none.
    drawn = context.forms.chart_quantity("points")
    for key in keys:
        unit = figure_unit(report_input, key)
        drawn = context.forms.chart_quantity(unit, change=True, measure=key, kind=figure_kind(context, key))
        theirs = figure_stat(report_input, key, control, context.query.aspect)
        for arm in context.arm_order(arms):
            mine = figure_stat(report_input, key, arm, context.query.aspect)
            change = _descriptive(context, key, arm, mine, theirs, drawn, keys=keys)
            if change is not None:
                group = measure_words(report_input, key) if grouped else None
                changes.append(_with_group(change, group))
    notes = (UNCORRECTED_NOTE,) if any(c.strong is not None for c in changes) else ()
    return (*_in_one_form(context, keys, changes, drawn), notes)


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
    drawn = _change_quantity(context, keys[0])
    for key in keys:
        drawn = _change_quantity(context, key)
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
                    keys=keys,
                )
                if change is not None:
                    by_arm.setdefault(arm, []).append(_with_group(change, None, label=where))
    notes = (UNCORRECTED_NOTE,) if any(c.strong is not None for rows in by_arm.values() for c in rows) else ()
    # One form across every arm's changes (#993), each arm's rows kept in order, each with what it is compared with.
    flat, settled = _in_one_form(context, keys, [change for rows in by_arm.values() for change in rows], drawn)
    redrawn = iter(_baselined(context, flat))
    return {arm: [next(redrawn) for _ in rows] for arm, rows in by_arm.items()}, settled, notes


def _change_quantity(context: ChartContext, key: str) -> Quantity:
    """How a chart draws the changes of one measure, in its figures' unit."""
    unit = figure_unit(context.report_input, key)
    return context.forms.chart_quantity(unit, change=True, measure=key, kind=figure_kind(context, key))


def _preference_changes(context: ChartContext) -> tuple[list[_Change], Quantity, tuple[str, ...]]:
    preference = context.report_input.human_pref
    if preference is None:
        raise context.refuse("the input holds no human preference")
    # A human preference is a share of raters, no measure's figure: its format names the digits its shares read at.
    items: list[PreferenceRow] = []
    for row in context.rows():
        assert isinstance(row.item, PreferenceRow)
        items.append(row.item)
    drawn = context.forms.chart_quantity("percent", [float(item.prefers_control) for item in items])
    changes = []
    for item in items:
        value = float(item.prefers_control)
        width = float(item.half_width) if item.half_width is not None else None
        said = context.forms.chart_note(value, drawn).text
        tip = [item.name or item.id, said]
        changes.append(
            _Change(
                id=item.id,
                label=item.name or item.id,
                color=context.look.arm(preference.arm).token,
                value=value,
                interval=(value - width, value + width) if width is not None else None,
                strong=abs(value - _NO_PREFERENCE) > width if width is not None else None,
                note=said,
                tip=tip,
                measure="",
                kind="",
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
    drawn = _change_quantity(context, key)
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
            kind = figure_kind(context, key)
            said = context.forms.chart_note(change, drawn, measure=key, kind=kind).text
            changes.append(
                _Change(
                    id=f"{factor.id}={level.id}",
                    label=name,
                    color=None,
                    value=change,
                    interval=None,
                    strong=None,
                    note=said,
                    tip=[name, said],
                    group=factor.name or factor.id,
                    measure=key,
                    kind=kind,
                )
            )
    return (*_in_one_form(context, [key], changes, drawn), (LEVER_NOTE,))


# --- rows an analysis computes: two arms' difference, and the lever effects ----------------------------------------


def _difference_quantity(context: ChartContext, key: str) -> Quantity:
    """How a difference between two arms on one measure is drawn: a rate's in points, another's in its own unit."""
    unit = contrast_unit_of(context.report_input, key)
    return context.forms.chart_quantity(unit, change=True, measure=key, kind=figure_kind(context, key))


def _arm_name(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).name if arm in context.look.arms else arm


def _arm_label(context: ChartContext, arm: str) -> str:
    return context.look.arm(arm).label if arm in context.look.arms else arm


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
    context: ChartContext,
    *,
    measure: str,
    row_id: str,
    label: str,
    short: str,
    value: float,
    interval: tuple[float, float] | None,
    drawn: Quantity,
    head: Sequence[str],
    strong: bool | None,
    group: str | None = None,
    base: str | None = None,
) -> _Change:
    """One row an analysis computed, as a change: bold where its interval clears zero, muted where it holds it; what
    it is compared with (``base``) where the input states it; on ``measure``, the measure the analysis compared."""
    forms, kind = context.forms, figure_kind(context, measure)
    printed = forms.chart_note(value, drawn, measure=measure, kind=kind)
    said = printed.text
    tip = [*head, said]
    ends = None
    if interval is not None:
        ends = forms.chart_interval(*interval, drawn, measure=measure, kind=kind)
        tip[-1] += f", {ends.beside}"
    return _Change(
        id=row_id,
        label=label,
        color=None,
        value=value,
        interval=interval,
        strong=strong,
        note=said,
        tip=tip,
        group=group,
        short_label=short,
        base=base,
        measure=measure,
        kind=kind,
        texts=_decided(context, drawn, measure, kind, printed.digits, _end_digits(ends)),
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
        level = _level_quantity(context, key, [key])
        theirs = _figure_value(item.second, level)
        if item.basis is not None:
            bases.append(compare_basis(context, item, key))
        changes.append(
            _analysed_change(
                context,
                measure=key,
                row_id=row.label,
                label=place,
                short=place,
                value=value,
                interval=interval,
                drawn=drawn,
                head=[place],
                strong=item.excludes_zero,
                base=(
                    context.forms.chart_level(theirs, level, measure=key, kind=figure_kind(context, key)).text
                    if theirs is not None
                    else None
                ),
            )
        )
    changes, drawn = _in_one_form(context, [key], changes, drawn)
    named = COMPARE_NOTE.format(
        first=_arm_name(context, first),
        second=_arm_name(context, second),
        measure=money_words(context.report_input, key, drawn, prose=True),
    )
    basis = tuple(dict.fromkeys(BASIS_NOTE.format(basis=basis) for basis in bases))
    return changes, drawn, (named, *basis, *narrow_notes(context))


def narrow_notes(context: ChartContext) -> tuple[str, ...]:
    """What a compare chart or table says, once, where a row's interval may be too narrow (``may_be_narrow``): each
    arm that ran more trials than cases, by its reader name and the trials and cases its key reads, said as a may,
    since a marked row's interval may count cases; then the unit only where the input states it (an unpaired row's
    ``counts``: trials or observations), and that a paired row's unit is unstated where its contrast states no count.
    A paired row whose contrast states its cases counts cases, is never marked, and says its cases in its basis
    (:func:`compare_basis`). Empty where no row is marked.

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
    candidate a contrast reads against the control and the measure by their reader names, never by id or key, and
    whether it is corrected as the contrast's own family says (:func:`correction_words`: corrected only where the
    family names a method); the independent reading is the analysis's own words, which name neither.

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
    first, second = context.query.arms
    contrast = pairing_contrast(context.report_input, first, second, key, row.where)
    if contrast is None:
        return str(row.basis.value)
    return PAIRED_BASIS_TEMPLATE.format(
        paired=_paired_cases(context, row, candidate),
        arm=_arm_name(context, candidate),
        measure=measure_words(context.report_input, key, prose=True),
        corrected=correction_words(context.report_input, contrast),
    )


def _paired_cases(context: ChartContext, row: ComparisonRow, candidate: str) -> str:
    """How a paired row's contrast read its cases, from the counts the row cites (``cases``): paired over them,
    unpaired over each side's, or paired alone where the contrast states none."""
    said = {figure.key.rpartition(".")[2]: figure for figure in row.cases}
    cases = said.get("n_cases")
    if cases is None:
        return PAIRED_CASES["unstated"]
    paired = said.get("paired")
    if paired is not None and paired.value == "unpaired":
        control = said.get("control_n_cases", cases)
        return PAIRED_CASES["unpaired"].format(
            cases=cases.reads, arm=_arm_name(context, candidate), control_cases=control.reads
        )
    return PAIRED_CASES["paired"].format(cases=cases.reads)


def setting_value(context: ChartContext, setting: str, arm: str) -> str | None:
    """One arm's value of a setting as the input states it: the name its declared factor gives that level (a level
    matched by its id or name, as a lattice matches one), else the value itself, a number by the package's number rule
    (``1``, never ``1.0``); None where the arm states none.

    :param context: The chart being resolved.
    :ptype context: ChartContext
    :param setting: The setting, by id.
    :ptype setting: str
    :param arm: The arm, by id.
    :ptype arm: str
    :return: The value as a reader reads it, or None.
    :rtype: str | None
    """
    report_input = context.report_input
    value = arm_settings(report_input).get(arm, {}).get(setting)
    if value is None:
        return None
    spelled = value if isinstance(value, str) else context.forms.declared(float(value)).text
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
    names = context.report_input.lever_names

    def moved(start: str, end: str, *, short: bool) -> str:
        value = setting_value(context, row.setting, end)
        first = _arm_tag(context, start) if short else _arm_name(context, start)
        return LEVER_MOVE_TEMPLATE.format(start=first, value=value if value is not None else _arm_name(context, end))

    if isinstance(row, EffectRow):
        start, end = arms
        label = moved(start, end, short=False)
        short = LEVER_MOVE_TEMPLATE.format(start=_arm_tag(context, start), value=_arm_tag(context, end))
        return label, short, LEVER_EFFECT_GROUP.format(setting=lever_reader_name(row.setting, names))
    first_from, first_to, second_from, second_to = (arms + ["", "", "", ""])[:4]
    label = LEVER_CROSS_TEMPLATE.format(
        second=moved(second_from, second_to, short=False), first=moved(first_from, first_to, short=False)
    )
    short = LEVER_CROSS_TEMPLATE.format(
        second=moved(second_from, second_to, short=True), first=moved(first_from, first_to, short=True)
    )
    group = LEVER_CROSS_GROUP.format(
        setting=lever_reader_name(row.setting, names), across=lever_reader_name(row.across, names)
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
                context,
                measure=key,
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
    changes, drawn = _in_one_form(context, [key], changes, drawn)
    note = LEVERS_NOTE.format(measure=money_words(context.report_input, key, drawn, prose=True), interactions=crossed)
    return changes, drawn, (f"{note} {LEVERS_READ_NOTE}",)


# --- the charts ----------------------------------------------------------------------------------------------------


def _forest_row(change: _Change, context: ChartContext) -> ForestRow:
    if change.interval is None:
        raise context.refuse(f"row {change.label!r} carries no interval, and a forest draws intervals")
    texts = _row_texts(change)
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
        value_text=texts[0],
        lo_text=texts[1],
        hi_text=texts[2],
    )


def _ends(changes: Sequence[_Change], reference: ChartReference | None, *, intervals: bool) -> Iterator[float]:
    for change in changes:
        yield change.value
        if intervals and change.interval is not None:
            yield from change.interval
    if reference is not None:
        yield reference.value


def _family_level(context: ChartContext) -> str:
    """A forest's level as its head prints it (:meth:`~threetears.evals.analysis.report.forms.PageForms.confidence`,
    which reads :func:`~threetears.evals.analysis.report.forms.interval_level`, the form a table's cell prints it in):
    where its contrasts are one family's, corrected together at the chart's level, the joint guarantee the family
    holds at, ``95% (adjusted for 16 comparisons)`` (the kit prints ``95% intervals (adjusted for 16 comparisons)``)."""
    aspects = {row.item.aspect for row in context.rows() if isinstance(row.item, Contrast)}
    if context.query.source == "compare":
        # Two arms compared read a paired row on its contrast's own interval, corrected within its family: the head
        # states that family's joint guarantee, as a forest of the contrast does, never its per-interval level.
        first, second = context.query.arms
        key = context.rows()[0].key if context.rows() else ""
        paired = (
            pairing_contrast(context.report_input, first, second, key, row.item.where)
            for row in context.rows()
            if isinstance(row.item, ComparisonRow) and row.item.paired_with is not None
        )
        aspects = {contrast.aspect for contrast in paired if contrast is not None}
    families = [family for family in context.report_input.contrast_families if family.aspect in aspects]
    return context.forms.confidence(context.level, families[0] if len(families) == 1 else None).text


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
    zero = _zero_label(context, changes)
    changes = _baselined(context, changes)
    reference = reference_line(context, drawn)
    if reference is not None and reference.kind == "margin" and reference.label:
        # The band the margin shades is what "no worse than the margin" means; the line's label says so.
        reference = reference.model_copy(update={"label": MARGIN_BAND_TEMPLATE.format(line=reference.label)})
    domain = context.spec.domain or centred_domain(_ends(changes, reference, intervals=True), drawn)
    data = ForestChart(
        rows=[_forest_row(change, context) for change in changes],
        domain=(domain[0], domain[1]),
        fmt=drawn.fmt,
        level=_family_level(context) if source == "contrasts" else context.level,
        reference=reference,
        zero_label=zero,
        better=_better(context),
        emphasis=context.emphasis,
    )
    # Its values table reads each value and interval end at the unit's precision (a point to one decimal), as its
    # end labels and the keys citing the rows read them: one precision, never the float's four figures.
    return money_drawn(context, data, notes, money=_drawn_money(context, drawn), read_at=drawn.unit)


def _bar_row(change: _Change, level: str, *, dots: bool = False) -> BarRow:
    """A change as a bar, or as a dot (``dots``): a dot's row with no interval is an open mark, faded where the
    descriptive rule cannot tell it from zero; a dot with an interval is never faded, its interval saying how sure it
    is, so no row reads surer than one whose interval holds zero."""
    interval = change.interval if level else None
    texts = _row_texts(change, with_interval=interval is not None)
    return BarRow(
        id=change.id,
        label=change.label,
        short_label=change.short_label,
        value=change.value,
        lo=interval[0] if interval is not None else None,
        hi=interval[1] if interval is not None else None,
        color=change.color,
        strong=True if change.strong else None,
        muted=True if change.strong is False and (not dots or interval is None) else None,
        note=change.note,
        tip=change.tip,
        group=change.group,
        value_text=texts[0],
        lo_text=texts[1],
        hi_text=texts[2],
    )


def _row_texts(change: _Change, *, with_interval: bool = True) -> tuple[str | None, str | None, str | None]:
    """A row's decided texts (:attr:`_Change.texts`): its value's, and its interval's two ends' where the row draws
    its interval; each None where the forms decide none, which the chart's format spells."""
    if change.texts is None:
        return None, None, None
    value, low, high = change.texts
    return (value, low, high) if with_interval else (value, None, None)


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
    elif source in ("measures", "judged"):
        changes, drawn, notes = _figure_changes(context)
    elif source in ("strata", "tasks"):
        by_arm, drawn, notes = _slice_changes(context)
        changes = [change for rows in by_arm.values() for change in rows]
        by = context.query.small_multiples_by
        if by in ("task", "aspect") and len({change.label for change in changes}) > 1:
            by_slice: dict[str, list[_Change]] = {}
            for change in changes:
                arm_label = context.look.arm(change.id).label
                by_slice.setdefault(change.label, []).append(_with_group(change, None, label=arm_label))
            panels = [_Panel(title=name, tag=None, color=None, rows=rows) for name, rows in by_slice.items()]
        elif by == "arm" and len(by_arm) > 1:
            panels = [_arm_panel(context, arm, rows) for arm, rows in by_arm.items()]
        elif len(by_arm) > 1:
            changes = [
                _with_group(change, context.look.arm(arm).label, label=change.label)
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
    zero = _zero_label(context, changes)
    if source not in ("strata", "tasks"):
        # Slices are set beside what they are compared with as their changes are read (:func:`_slice_changes`).
        changes = _baselined(context, changes)
    dots = source in _AS_DOTS
    if source == "contrasts" and any(change.panel is not None for change in changes):
        by_aspect: dict[str, list[_Change]] = {}
        for change in changes:
            by_aspect.setdefault(change.panel or "", []).append(change)
        panels = [_Panel(title=name, tag=None, color=None, rows=rows) for name, rows in by_aspect.items()]
    # Contrasts corrected together print the family's level in the head, as a forest of them does.
    level = _family_level(context) if source in ("contrasts", "compare") else context.level
    if dots:
        if not any(change.interval is not None for change in changes):
            level = ""
        if any(change.interval is None or not level for change in changes):
            notes = (*notes, OPEN_MARKS_NOTE)
    if centre == _NO_PREFERENCE:
        reference: ChartReference | None = ChartReference(value=centre, kind="no_preference")
    else:
        reference = reference_line(context, drawn)
    ends = list(_ends(changes, reference, intervals=bool(level)))
    domain = context.spec.domain or centred_domain(ends, drawn, centre=centre)
    read_at = drawn.unit if source in _COMPUTED else ""
    better = _better(context)
    if panels is not None:
        data = BarsChart(
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            level=level or None,
            reference=reference,
            zero_label=zero,
            better=better,
            dots=dots or None,
            emphasis=context.emphasis,
            panels=[
                BarsPanel(
                    title=panel.title,
                    tag=panel.tag,
                    color=panel.color,
                    rows=[_bar_row(change, level, dots=dots) for change in panel.rows],
                )
                for panel in panels
            ],
        )
    else:
        data = BarsChart(
            rows=[_bar_row(change, level, dots=dots) for change in changes],
            domain=(domain[0], domain[1]),
            fmt=drawn.fmt,
            level=level or None,
            reference=reference,
            zero_label=zero,
            better=better,
            dots=dots or None,
            emphasis=context.emphasis,
        )
    # Its values table names the unit every change is in (``-23 pts``), whatever source the changes come from.
    return money_drawn(context, data, notes, money=_drawn_money(context, drawn), read_at=read_at, table_unit=drawn.unit)


#: The kinds this module resolves.
RESOLVERS: dict[str, KindResolver] = {"forest": _forest, "diverging_bars": _diverging}
