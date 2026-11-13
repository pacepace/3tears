"""What each family reads from the report input, the values derived here from its point figures, and where a
page's ledger holds a family's figure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.report.cost_unit import REPLIES, CostBasis, MoneyRuler, money_ruler, per_count, per_noun
from threetears.evals.analysis.report.keys.figure import (
    DIRECTION_WORDS,
    Figure,
    Scale,
    _PER_THOUSAND_FACTORS,
    _SCALE_COUNTS,
    _count,
    _listed,
)
from threetears.evals.analysis.report.keys.names import _measure_reader_name, reader_name
from threetears.evals.analysis.report.keys.vocabulary import (
    _LedgerAs,
    _arm,
    _asked,
    _asked_or_refuse,
    _candidates,
    _contrast_scale,
    _control,
    _criterion,
    _end,
    _held_apart,
    _interval,
    _judged_scale,
    _level,
    _measure_scale,
    _number,
    _printed_stratum,
    _rate_scale,
    _reading,
    _refuse,
    _scoped_strata,
    _task,
    _text,
    _unstated,
)
from threetears.evals.analysis.report.words import given_name
from threetears.evals.analysis.report_input import (
    Arm,
    Aspect,
    Cell,
    Contrast,
    ContrastFamily,
    Criterion,
    Depth,
    FloorRead,
    Judged,
    LateCriterion,
    Reading,
    ReportInput,
    ScoreboardRow,
    Stat,
    Stratum,
    Task,
)
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    aspect_board,
    aspect_of,
    criterion_aspect,
    criterion_figures,
    flip_margin,
    lead_aspect,
    on_aspect,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.style import lever_reader_name
from threetears.evals.kernel.metrics import goal_check_of

if TYPE_CHECKING:
    # A page's forms decide over the figures this module reads, so the module that decides them reads this one: every
    # caller hands its page's forms in, and the type is named for the checker alone.
    from threetears.evals.analysis.report.forms import Role


def _of_the_pass_rate(roles: Mapping[str, Role]) -> _LedgerAs:
    """A family whose figures, by statistic, are the pass rate's (:func:`pass_rate_measure`), each in the role given:
    a reading's change, its ends, two arms' pass rates and the difference between them, which a path names by its arms
    and aspect alone."""

    def held(report_input: ReportInput, _values: tuple[str, ...], figure: Figure) -> tuple[str, str, Role] | None:
        role = roles.get(figure.statistic or "")
        return None if role is None else (*pass_rate_measure(report_input), role)

    return held


def _the_page_s_cost(*statistics: str) -> _LedgerAs:
    """A family whose figures, by statistic, are the page's cost (:func:`cost_measure`), held as the money its cost per
    1,000 and its table's cost cell are: an arm's cost per reply, and its cost with unreported trials counted free or
    at its 95th percentile; nothing where the page reads its cost from no measure."""

    def held(report_input: ReportInput, _values: tuple[str, ...], figure: Figure) -> tuple[str, str, Role] | None:
        measure = cost_measure(report_input)
        if measure is None or figure.statistic not in statistics:
            return None
        return (*measure, "money")

    return held


#: An arm's cost per reply, on its scoreboard: the page's cost.
_A_REPLY_COST = _the_page_s_cost("reply_cost_usd")


def _a_bar_s_level(
    _report_input: ReportInput, _values: tuple[str, ...], figure: Figure
) -> tuple[str, str, Role] | None:
    """An arm's figure against a bar, and its interval's ends, are its level on the bar's measure (money's its money),
    as the measure's table prints it; the bar itself is a declared line, no figure of the measure."""
    if figure.set_of is None or figure.statistic not in ("value", "low", "high"):
        return None
    return (*figure.set_of, "money" if figure.scale in _SCALE_COUNTS else "level")


#: A reading's figures that are the pass rate's: its change and that change's ends.
_READING_LEDGER = _of_the_pass_rate({"change": "change", "interval": "change", "low": "change", "high": "change"})

#: A reading's change against its margin, as a settle reads it: the pass rate's.
_MARGIN_LEDGER = _of_the_pass_rate({"change": "change"})


def _a_floor_s_change(
    report_input: ReportInput, values: tuple[str, ...], figure: Figure
) -> tuple[str, str, Role] | None:
    """A floor's change, as a settle reads it, where the floor speaks for the aspect's pass rate: the reading's change,
    the pass rate's (:func:`_floor_spread`); a floor on its own measure reads a difference of two means no table holds,
    so nothing."""
    aspect_id, floor, arm_id, _field = values
    read = next(
        held for held in report_input.floors if (held.aspect, held.floor, held.arm) == (aspect_id, floor, arm_id)
    )
    return _MARGIN_LEDGER(report_input, values, figure) if read.reading is None else None


def _the_pass_rate(report_input: ReportInput, _values: tuple[str, ...]) -> tuple[str, str] | None:
    """The rate of a family that reads an aspect's pass rate (a reading, a margin, two arms apart): the pass rate's."""
    return pass_rate_measure(report_input)


def _floor_rate(report_input: ReportInput, read: FloorRead) -> tuple[str, str] | None:
    """The rate a floor read is read on, with its kind: the rate it names (``reading``), a judged dimension or the one
    measure of that key, or the aspect's pass rate where it names none; None where it names a rate the input holds no
    one measure of, which is no measure's, never the pass rate's."""
    if read.reading is None:
        return pass_rate_measure(report_input)
    if read.reading in report_input.judged:
        return (read.reading, "judged")
    named = [key for key in report_input.measures if key == read.reading or goal_check_of(key) == read.reading]
    return (named[0], "measure") if len(named) == 1 else None


def _a_floor_s_rate(report_input: ReportInput, values: tuple[str, ...]) -> tuple[str, str] | None:
    """The rate of a floor family's figures, from the floor read its path names (its aspect, floor and arm)."""
    aspect_id, floor, arm_id = values[:3]
    read = next(
        held for held in report_input.floors if (held.aspect, held.floor, held.arm) == (aspect_id, floor, arm_id)
    )
    return _floor_rate(report_input, read)


# --- what each family reads ------------------------------------------------------------------------------------

_META_FIELDS = (
    "campaign",
    "question",
    "behavior",
    "kind",
    "kind_means",
    "design_shape",
    "run_date",
    "provenance",
    "volume_per_month",
)


def _meta(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    if field == "volume_per_month":
        return _number(path, report_input.meta.volume_per_month, "the monthly volume", "count")
    value = getattr(report_input.meta, field)
    return _text(path, value, f"the campaign's {field}")


_ARM_FIELDS = ("id", "name", "served_model", "description")


def _arm_field(arm: Arm, field: str, path: str) -> Figure:
    if field == "name":
        return _text(path, arm.name or arm.id, "the arm's name")
    return _text(path, getattr(arm, field), f"the arm's {field}")


def _control_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    control = _control(report_input)
    if control is None:
        raise _refuse(path, "names the control, and this campaign has none")
    return _arm_field(control, values[0], path)


def _n_candidates(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return Figure(path=path, kind="number", value=float(len(_candidates(report_input))), scale="count")


_ASPECT_FIELDS = ("id", "name", "margin", "description", "graded_by", "pass_rule", "pass_threshold", "no_criterion")


def _aspect_field(aspect: Aspect, field: str, path: str) -> Figure:
    if field == "name":
        return _text(path, given_name(aspect.id, aspect.name), "the aspect's name")
    if field == "margin":
        return _number(path, aspect.margin, "the aspect's margin", "points")
    rule = aspect.pass_rule
    if field == "pass_rule":
        return _text(
            path,
            rule.label if rule is not None else None,
            "the aspect's pass rule: every goal check, each other check with the value it must hold to pass, and the "
            "level each counted judged dimension must reach",
        )
    if field == "pass_threshold":
        return _number(path, rule.threshold if rule is not None else None, "the aspect's pass threshold", "plain")
    if field == "no_criterion":
        if rule is None:
            raise _unstated(
                path, "names the trials that passed with nothing to fail them, which the input does not state"
            )
        return _count(path, rule.no_criterion)
    return _text(path, getattr(aspect, field), f"the aspect's {field}")


def _asked_aspect_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _aspect_field(_asked_or_refuse(report_input, path), values[0], path)


def _named_aspect_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, field = values
    return _aspect_field(aspect_of(report_input, aspect_id), field, path)


_READING_STATISTICS = (
    "change",
    "interval",
    "low",
    "high",
    "margin",
    "level",
    "verdict",
    "control_passed",
    "arm_passed",
    "why",
)


def _reading_figure(reading: Reading, statistic: str, path: str) -> Figure:
    if statistic == "change":
        return _number(path, reading.change, "the change", "points", change=True)
    if statistic == "interval":
        return _interval(path, reading.within_interval, "the interval", "points", change=True)
    if statistic in ("low", "high"):
        end = _end(path, reading.within_interval, statistic, "the interval")
        return _number(path, end, f"the interval's {statistic} end", "points", change=True)
    if statistic == "margin":
        return _number(path, reading.margin, "the reading's margin", "points")
    if statistic == "level":
        # The level of the interval `interval`, `low` and `high` read (within_interval): the nominal one, with no
        # correction. gain_confidence is the gain interval's, which no reading key reads.
        stated = INTERVAL_LEVEL if reading.within_interval is not None else None
        return _level(path, stated, "the interval's level")
    if statistic == "verdict":
        return _text(path, reading.reading, "the reading")
    return _text(path, getattr(reading, statistic), f"the reading's {statistic}")


def _reading_arms(aspect: Aspect | None) -> list[str]:
    return [reading.arm for reading in aspect.readings] if aspect else []


def _asked_reading(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, statistic = values
    return _reading_figure(_reading(_asked_or_refuse(report_input, path), arm_id), statistic, path)


def _named_reading(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, statistic = values
    return _reading_figure(_reading(aspect_of(report_input, aspect_id), arm_id), statistic, path)


_CONTRAST_STATISTICS = (
    "delta",
    "interval",
    "low",
    "high",
    "verdict",
    "p_adjusted",
    "level",
    "materiality",
    "reason",
    "n_cases",
    "control_n_cases",
    "paired",
    "pairing_r",
)


def _contrasted(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(c.measure for c in report_input.contrasts if c.arm == prior[0]))


def _contrast_figure(report_input: ReportInput, contrast: Contrast, statistic: str, path: str) -> Figure:
    scale, unit = _contrast_scale(report_input, contrast.measure, contrast.kind)
    if statistic == "delta":
        return _number(path, contrast.delta, "the delta", scale, unit=unit, change=True)
    if statistic == "interval":
        return _interval(path, contrast.interval, "the interval", scale, unit=unit, change=True)
    if statistic in ("low", "high"):
        end = _end(path, contrast.interval, statistic, "the interval")
        return _number(path, end, f"the interval's {statistic} end", scale, unit=unit, change=True)
    if statistic == "verdict":
        return _text(path, contrast.verdict, "the verdict")
    if statistic == "p_adjusted":
        return _number(path, contrast.p_adjusted, "the adjusted p", "plain")
    if statistic == "level":
        return _text(path, contrast.level, "the interval's level")
    if statistic in ("n_cases", "control_n_cases"):
        cases = getattr(contrast, statistic)
        if cases is None:
            raise _unstated(path, "names the cases the contrast's test read, which the input does not state")
        return _count(path, cases)
    if statistic == "paired":
        # A word, as a reader writes it: "paired over 60 cases".
        paired = None if contrast.paired is None else "paired" if contrast.paired else "unpaired"
        return _text(path, paired, "whether the contrast's test paired the arms")
    if statistic == "pairing_r":
        return _number(path, contrast.pairing_r, "the arms' correlation across the paired cases", "plain")
    return _text(path, getattr(contrast, statistic), f"the contrast's {statistic}")


def _contrast(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, measure, statistic = values
    held = [c for c in report_input.contrasts if c.arm == arm_id and c.measure == measure]
    if len(held) > 1:
        raise _held_apart(
            path,
            f"{arm_id}'s contrast on {measure}",
            [c.aspect for c in held],
            lambda aspect: f"aspects.{aspect}.contrast.{arm_id}.{measure}.{statistic}",
        )
    return _contrast_figure(report_input, held[0], statistic, path)


def _aspect_contrast_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(c.arm for c in report_input.contrasts if c.aspect == prior[0]))


def _aspect_contrasted(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, arm_id = prior
    return list(dict.fromkeys(c.measure for c in report_input.contrasts if c.aspect == aspect_id and c.arm == arm_id))


def _aspect_contrast(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, measure, statistic = values
    held = [c for c in report_input.contrasts if (c.aspect, c.arm, c.measure) == (aspect_id, arm_id, measure)]
    if len(held) > 1:
        raise _refuse(path, f"names {arm_id}'s contrast on {measure}, which aspect {aspect_id} holds {len(held)} times")
    return _contrast_figure(report_input, held[0], statistic, path)


def _judged_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.judged[prior[0]].per_arm)


def _judged(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, arm_id = values
    judged = report_input.judged[dimension]
    return _number(path, judged.per_arm[arm_id].mean, "the dimension's mean", _judged_scale(judged))


def _judged_stat(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, arm_id, statistic = values
    stat = report_input.judged[dimension].per_arm[arm_id]
    scale, unit = _scale_of(report_input, dimension)
    return _stat_figure(path, stat, statistic, scale, unit, f"the dimension's {statistic}")


def _aspect_judged_dimensions(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [key for key, judged in report_input.judged.items() if on_aspect(report_input, judged, prior[0])]


def _aspect_judged_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, dimension = prior
    return list(on_aspect(report_input, report_input.judged[dimension], aspect_id))


def _aspect_judged(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, dimension, arm_id, statistic = values
    stat = on_aspect(report_input, report_input.judged[dimension], aspect_id)[arm_id]
    scale, unit = _scale_of(report_input, dimension)
    return _stat_figure(path, stat, statistic, scale, unit, f"the dimension's {statistic} on the aspect")


_JUDGE_FIELDS = ("model", "tier", "calibration", "calibration_ratings", "agreement")
_JUDGED_ATTRIBUTES = {"model": "judge_model"}


def _judge_value(judged: Judged, field: str, path: str) -> Figure:
    value = getattr(judged, _JUDGED_ATTRIBUTES.get(field, field))
    if field == "calibration_ratings":
        return _number(path, value, "the judge's calibration ratings", "count")
    if field == "agreement":
        return _number(path, value, "the judge's agreement", "share", whole=True)
    return _text(path, value, f"the judge's {field}")


def _judged_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    dimension, field = values
    judged = report_input.judged[dimension]
    if field == "label":
        return _text(path, reader_name("judged", dimension, report_input), "the dimension's label")
    return _judge_value(judged, field, path)


def _judge(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    attribute = _JUDGED_ATTRIBUTES.get(field, field)
    stated = {dimension: getattr(judged, attribute) for dimension, judged in report_input.judged.items()}
    distinct = {value for value in stated.values() if value is not None}
    if not distinct:
        raise _refuse(path, f"names the judge's {field}, which no judged dimension states")
    if len(distinct) > 1:
        per = ", ".join(f"`judged.{dimension}.{field}`" for dimension, value in stated.items() if value is not None)
        raise _refuse(path, f"names one judge {field}, and the judged dimensions differ; name one: {per}")
    dimension = next(dimension for dimension, value in stated.items() if value is not None)
    return _judge_value(report_input.judged[dimension], field, path)


def _passing(report_input: ReportInput) -> dict[str, float]:
    """Each arm's pass rate: passed over usable from the scoreboard, else the input's ``pass_rate`` measure."""
    board = {
        arm: row.passed / row.usable
        for arm, row in report_input.scoreboard.items()
        if row.passed is not None and row.usable
    }
    if board:
        return board
    measure = report_input.measures.get("pass_rate")
    if measure is None:
        return {}
    return {arm: float(stat.mean) for arm, stat in measure.per_arm.items() if stat.mean is not None}


def _counts_passes(report_input: ReportInput) -> bool:
    """Whether the scoreboard counts some arm's passes, so a pass rate is passed over usable."""
    return any(row.passed is not None and row.usable for row in report_input.scoreboard.values())


def _pass_rate_scale(report_input: ReportInput) -> Scale:
    if _counts_passes(report_input):
        return "share"
    return _measure_scale(report_input.measures["pass_rate"])[0]


def pass_rate_measure(report_input: ReportInput) -> tuple[str, str]:
    """The measure an arm's pass rate is a figure of, with its kind, as every table and key holds it: how each trial
    ended (``trial``) where the scoreboard counts passes, else the input's ``pass_rate`` measure, which the figure is
    read from (:func:`_passing`).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The measure's key and kind.
    :rtype: tuple[str, str]
    """
    return (TRIAL_PASS_NAME, "trial") if _counts_passes(report_input) else ("pass_rate", "measure")


def cost_measure(report_input: ReportInput) -> tuple[str, str] | None:
    """The measure an arm's cost per 1,000 is a figure of, with its kind, as every table and key holds it: the measure
    the page's cost is read from (:func:`cost_basis`); None where it is read from the scoreboard, or none states it.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The measure's key and kind, or None.
    :rtype: tuple[str, str] | None
    """
    source = cost_basis(report_input).source
    return (source, "measure") if source in report_input.measures else None


def _pass_rate(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    read = _number(path, _passing(report_input)[values[0]], "the pass rate", _pass_rate_scale(report_input))
    return replace(read, set_of=pass_rate_measure(report_input))


def _cost_measures(report_input: ReportInput) -> tuple[list[str], list[str]]:
    """The measures an arm's cost may be read from, in the order they are tried: each one alone of its kind.

    First the measures read before the scoreboard: the one stated per 1,000 (``usd_per_1k``), per 1,000,000,
    the one in dollars on the cost merit axis. Then the one dollar measure, in any of those units, where lower is
    better, read after it.
    """
    measures = report_input.measures

    def alone(keys: list[str]) -> list[str]:
        return keys if len(keys) == 1 else []

    first = [
        *alone([key for key, m in measures.items() if m.unit == "usd_per_1k"]),
        *alone([key for key, m in measures.items() if m.unit == "usd_per_1m"]),
        *alone([key for key, m in measures.items() if m.unit == "usd" and m.merit_axis == "cost"]),
    ]
    lower = alone(
        [key for key, m in measures.items() if m.unit in _PER_THOUSAND_FACTORS and m.higher_is_better is False]
    )
    return first, lower


def _overall_cost(report_input: ReportInput, key: str) -> dict[str, float]:
    measure = report_input.measures[key]
    factor = _PER_THOUSAND_FACTORS[measure.unit or ""]
    return {arm: stat.mean * factor for arm, stat in measure.per_arm.items() if stat.mean is not None}


def _blended_cost(report_input: ReportInput, key: str) -> dict[str, float]:
    """Each arm's cost on one measure blended over the tasks: its task figures weighted by each task's traffic share.

    An arm is blended over the tasks that carry traffic and hold its figure, each weighted by its share of their
    traffic, and left out where any of those figures is unstated.
    """
    factor = _PER_THOUSAND_FACTORS[report_input.measures[key].unit or ""]
    blended: dict[str, float] = {}
    for arm in report_input.arms:
        parts = [
            (float(task.share), task.per_arm[arm.id][key].mean)
            for task in report_input.tasks
            if task.share and arm.id in task.per_arm and key in task.per_arm[arm.id]
        ]
        means = [mean for _, mean in parts if mean is not None]
        if parts and len(means) == len(parts):
            total = sum(share for share, _ in parts)
            blended[arm.id] = factor * sum(share * mean for (share, _), mean in zip(parts, means, strict=True)) / total
    return blended


#: The cost source that is the scoreboard's cost per reply, as :class:`CostBasis` names it.
_SCOREBOARD_COST = "scoreboard"


def _cost_source(report_input: ReportInput) -> tuple[str, dict[str, float]]:
    """The first cost source that holds a figure, by name (a measure's key, or :data:`_SCOREBOARD_COST`), and each
    arm's cost per 1,000 on it; ``("", {})`` where none does (:func:`_costs_per_thousand`'s order)."""
    first, lower = _cost_measures(report_input)
    sources = [(key, _overall_cost(report_input, key)) for key in first]
    sources.append(
        (
            _SCOREBOARD_COST,
            {
                a: row.reply_cost_usd * 1000
                for a, row in report_input.scoreboard.items()
                if row.reply_cost_usd is not None
            },
        )
    )
    sources += [(key, _overall_cost(report_input, key)) for key in lower]
    sources += [(key, _blended_cost(report_input, key)) for key in dict.fromkeys([*first, *lower])]
    return next(((name, source) for name, source in sources if source), ("", {}))


def _costs_per_thousand(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000 of what it counts, from the first source that holds one: the one rule a page and the
    decision read cost by (``cost_per_1k.<arm>``).

    The arms' overall figures on a measure stated per 1,000; else on one stated per 1,000,000, divided by 1,000; else
    on the one dollar measure on the cost merit axis, times 1,000; else the scoreboard's cost per reply, times 1,000;
    else the one dollar measure where lower is better. Where none of those states a figure, each measure in the same
    order blended over the tasks by traffic share (:func:`_blended_cost`). What the count counts is
    :func:`cost_basis`'s.
    """
    source, costs = _cost_source(report_input)
    return _with_reply_costs(report_input, source, costs, report_input.scoreboard)


def _with_reply_costs(
    report_input: ReportInput, source: str, costs: dict[str, float], board: Mapping[str, ScoreboardRow]
) -> dict[str, float]:
    """``costs`` (a source's figures, per 1,000) with each arm the source lacks and a scoreboard's cost per reply
    states, in that figure times 1,000.

    A source can leave an arm out that the scoreboard states: the cost per reply is the same cost over the same count
    where the source counts replies, so the arm is not left without a cost beside arms that carry one. Every table and
    key reads an arm's cost here, so one arm states one cost. A source counting anything else takes no fallback.
    """
    if source == _SCOREBOARD_COST:
        return costs
    measure = report_input.measures.get(source)
    if measure is None or per_noun(measure) != REPLIES:
        return costs
    return {
        **{arm: row.reply_cost_usd * 1000 for arm, row in board.items() if row.reply_cost_usd is not None},
        **costs,
    }


def cost_basis(report_input: ReportInput) -> CostBasis:
    """What the page's cost (``cost_per_1k.<arm>``) is per, read from the source it is read from.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The noun (``replies`` from the scoreboard or a dollar-per-reply measure; a per-count measure's label's
        noun; empty where unstated, or where the input states no cost) and the source.
    :rtype: CostBasis
    """
    source, _costs = _cost_source(report_input)
    if source == _SCOREBOARD_COST:
        return CostBasis(noun=REPLIES, source=source)
    measure = report_input.measures.get(source)
    return CostBasis(noun=per_noun(measure) if measure is not None else "", source=source)


def cost_ruler(report_input: ReportInput) -> MoneyRuler:
    """How a column of the page's cost (``cost_per_1k.<arm>`` for every arm) is stated, by the one owner of money's
    scale: per 1,000 of what the cost counts, or per 1,000,000 where an arm's cost is under a cent per 1,000, a table's
    head naming the count (:meth:`~threetears.evals.analysis.report.cost_unit.MoneyRuler.words`).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The ruler.
    :rtype: MoneyRuler
    """
    return money_ruler(
        _costs_per_thousand(report_input).values(),
        per=1000,
        start=1000,
        reconciles=report_input.meta.volume_per_month is not None,
    )


def criterion_ruler(report_input: ReportInput, criterion: Criterion, line: Figure) -> tuple[MoneyRuler, int]:
    """The ruler a money criterion's figures are stated on, where its count is named once beside them (a gate's
    reason, a gate row's line): the one owner of money's scale over the measure's every arm's figure and the line, per
    1,000 of what it counts at least, never a per-1,000 mark.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion: The criterion, on a money measure.
    :ptype criterion: Criterion
    :param line: Its threshold's figure (``criteria.<id>.threshold``).
    :ptype line: Figure
    :return: The ruler, and the count its measure's own unit is per.
    :rtype: tuple[MoneyRuler, int]
    """
    own = _SCALE_COUNTS[line.scale]
    measure = report_input.measures.get(criterion.measure)
    amounts = [
        float(stat.mean) for stat in (measure.per_arm.values() if measure is not None else ()) if stat.mean is not None
    ]
    return money_ruler([*amounts, float(criterion.threshold)], per=own, start=max(own, 1000)), own


def aspect_costs_per_thousand(report_input: ReportInput, aspect_id: str | None) -> dict[str, float]:
    """Each arm's cost per 1,000 of what it counts on one aspect, by the one cost rule's source read on that aspect.

    On the lead aspect it is the page's cost (``cost_per_1k.<arm>``). On another, it is the same source's figures
    there: the cost measure's ``by_aspect`` means, in the same units, or the aspect's own scoreboard's cost per reply,
    so every aspect's cost counts what :func:`cost_basis` names. A source blended over the tasks holds no aspect's
    figures, and neither does an aspect that does not carry the source: none, which a table prints as a dash.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param aspect_id: The aspect, by id; None for the lead aspect.
    :ptype aspect_id: str | None
    :return: Each arm's cost per 1,000, by arm; empty where the aspect holds none.
    :rtype: dict[str, float]
    """
    if aspect_id is None or aspect_id == lead_aspect(report_input):
        return _costs_per_thousand(report_input)
    source = _cost_source(report_input)[0]
    if source == _SCOREBOARD_COST:
        return {
            arm: row.reply_cost_usd * 1000
            for arm, row in aspect_board(report_input, aspect_id).items()
            if row.reply_cost_usd is not None
        }
    measure = report_input.measures.get(source)
    if measure is None:
        return {}
    factor = _PER_THOUSAND_FACTORS[measure.unit or ""]
    held = {
        arm: stat.mean * factor for arm, stat in measure.by_aspect.get(aspect_id, {}).items() if stat.mean is not None
    }
    return _with_reply_costs(report_input, source, held, aspect_board(report_input, aspect_id))


def cost_in_measure_unit(report_input: ReportInput, measure_key: str, per_thousand: float) -> float:
    """A cost per 1,000 (the page's cost rule's figure) in the unit a dollar measure states its means in, so an arm the
    measure holds no figure for can be read in the measure's own column.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param measure_key: The dollar measure, by key.
    :ptype measure_key: str
    :param per_thousand: A cost per 1,000.
    :ptype per_thousand: float
    :return: The cost as a mean of that measure.
    :rtype: float
    :raises KeyError: The key names no dollar measure.
    """
    return per_thousand / _PER_THOUSAND_FACTORS[report_input.measures[measure_key].unit or ""]


def _money_per(report_input: ReportInput, values: tuple[str, ...]) -> str:
    """What a money figure read by a path counts: the money measure the path names, else the page's cost basis."""
    for value in values:
        measure = report_input.measures.get(value)
        if measure is not None and measure.unit in _PER_THOUSAND_FACTORS:
            return per_noun(measure)
    return cost_basis(report_input).noun


def _input_unit_phrases(report_input: ReportInput) -> set[str]:
    """The phrases this input's costs are counted in, which prose may write whole: ``per 1,000 conversations`` and
    ``per 1,000,000 conversations`` for each noun a money measure or the cost basis states."""
    nouns = {per_noun(measure) for measure in report_input.measures.values()} | {cost_basis(report_input).noun}
    return {per_count(noun, count) for noun in nouns if noun for count in (1_000, 1_000_000)}


def _cost_per_1k(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    read = _number(path, _costs_per_thousand(report_input)[values[0]], "the cost", "usd_per_1k")
    return replace(read, set_of=cost_measure(report_input))


_MEASURE_STATISTICS = ("mean", "ci", "half_width", "sem", "n", "n_cases", "p50", "p95", "change")


def _measure_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(report_input.measures[prior[0]].per_arm)


def _measure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    key, arm_id, statistic = values
    measure = report_input.measures[key]
    stat = measure.per_arm[arm_id]
    scale, unit = _measure_scale(measure)
    if statistic in ("n", "n_cases"):
        return _number(path, getattr(stat, statistic), f"the measure's {statistic}", "count")
    if statistic == "ci":
        return _interval(path, stat.ci, "the measure's interval", scale, unit=unit)
    return _number(
        path, getattr(stat, statistic), f"the measure's {statistic}", scale, unit=unit, change=statistic == "change"
    )


def _stat_figure(path: str, stat: Stat, statistic: str, scale: Scale, unit: str, what: str) -> Figure:
    """One statistic of one arm's figure (:data:`_MEASURE_STATISTICS`), on the scale its measure is on."""
    if statistic in ("n", "n_cases"):
        return _number(path, getattr(stat, statistic), what, "count")
    if statistic == "ci":
        return _interval(path, stat.ci, what, scale, unit=unit)
    return _number(path, getattr(stat, statistic), what, scale, unit=unit, change=statistic == "change")


def _aspect_measures(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return [key for key, measure in report_input.measures.items() if on_aspect(report_input, measure, prior[0])]


def _aspect_measure_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    aspect_id, key = prior
    return list(on_aspect(report_input, report_input.measures[key], aspect_id))


def _aspect_measure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, key, arm_id, statistic = values
    measure = report_input.measures[key]
    scale, unit = _measure_scale(measure)
    stat = on_aspect(report_input, measure, aspect_id)[arm_id]
    return _stat_figure(path, stat, statistic, scale, unit, f"the measure's {statistic} on the aspect")


#: A slice's statistics (a task's, a stratum's, a cell's): a figure's, and the arm's mean against the control's there.
_SLICE_STATISTICS = (*_MEASURE_STATISTICS, "vs_control")
#: The slice statistics computed here rather than copied.
_SLICE_DERIVED = ("vs_control",)


def _slice_figure(
    report_input: ReportInput,
    per_arm: Mapping[str, Mapping[str, Stat]],
    values: tuple[str, str, str],
    path: str,
    where: str,
) -> Figure:
    """One arm's statistic on a measure or judged dimension within a slice (a task, a stratum, a cell), or its mean
    minus the control's there (``vs_control``: derived, descriptive)."""
    arm_id, key, statistic = values
    scale, unit = _scale_of(report_input, key)
    what = f"the {statistic} on {key} in {where}"
    if statistic != "vs_control":
        return _stat_figure(path, per_arm[arm_id][key], statistic, scale, unit, what)
    control = _control(report_input)
    if control is None:
        raise _refuse(path, "compares an arm with the control, and this campaign has none")
    if arm_id == control.id:
        raise _refuse(path, "compares an arm with the control, and names the control itself")
    mine, theirs = per_arm[arm_id][key].mean, per_arm.get(control.id, {}).get(key)
    if mine is None or theirs is None or theirs.mean is None:
        raise _unstated(
            path, f"names the arm's and the control's mean on {key} in {where}, which the input does not state"
        )
    return _number(path, mine - theirs.mean, f"the change from the control in {where}", scale, unit=unit, change=True)


_SCOREBOARD_STATISTICS = ("ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd")

#: What ``count.failures`` counts, and what the examples beside it are: a failure is a trial its pass rule failed.
_FAILURES_RULE = (
    "A failure is a trial its pass rule failed, for any cause: a check, a judged dimension or the candidate, as "
    "`scoreboard.<arm>.failed` counts them on the aspect. The examples are not a sample of them where the snapshot "
    "names a primary judged dimension: they are trials scoring a nought on it, so a trial a check failed is never one"
)

#: What the scoreboard's ``failed`` counts, beside ``count.failures``.
_SCOREBOARD_FAILED = (
    "`failed` counts every trial its pass rule failed, for any cause, as `count.failures.<aspect>.<arm>` does on the "
    "aspect"
)


def _scoreboard(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    arm_id, statistic = values
    return _board_figure(report_input.scoreboard[arm_id], statistic, path)


def _aspect_board_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(aspect_board(report_input, prior[0]))


def _aspect_scoreboard(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id, statistic = values
    return _board_figure(aspect_board(report_input, aspect_id)[arm_id], statistic, path)


def _board_figure(row: ScoreboardRow, statistic: str, path: str) -> Figure:
    if statistic in ("lost", "failed"):
        return _number(path, sum(getattr(row, statistic).values()), f"trials {statistic}", "count")
    if statistic.endswith("_usd"):
        read = replace(_number(path, getattr(row, statistic), f"the {statistic}", "usd"), per=REPLIES)
        # A run's test cost counts nothing: a total, read to cents whatever its size.
        return replace(read, total=True) if statistic == "test_cost_usd" else read
    return _number(path, getattr(row, statistic), f"trials {statistic}", "count")


def _reasoning_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.reasoning_ratio is not None]


def _reasoning_ratio(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    ratio = _arm(report_input, values[0]).reasoning_ratio
    return _number(path, ratio, "the share of output spent reasoning", "share", whole=True)


def _depth_figure(report_input: ReportInput, path: str, read: Callable[[ReportInput, Aspect], float | None]) -> Figure:
    """The asked aspect's depth as one figure; refused as a figure the input does not state where it has no single
    asked aspect, or the aspect's arms ran it to different depths: the input's shape, never a key spelled wrong."""
    aspect = _asked(report_input)
    if aspect is None:
        ids = ", ".join(f"`aspects.{a.id}.`" for a in report_input.aspects) or "none: the input holds no aspect"
        raise _unstated(
            path,
            f"reads the one asked aspect's depth, which the input does not state: it has no single one; name it: {ids}",
        )
    value = read(report_input, aspect)
    if value is None:
        raise _unstated(
            path,
            "reads the asked aspect's depth as one figure, which the input does not state: its arms ran it to "
            "different depths or none is stated",
        )
    return Figure(path=path, kind="number", value=float(value), scale="count")


def _mean_repeats(depth: Depth) -> float | None:
    """The mean repeats a case got (``repeats_per_case`` is fewest, mean, most), where the input states it."""
    return float(depth.repeats_per_case[1]) if len(depth.repeats_per_case) > 1 else None


def _depth_of(aspect: Aspect, report_input: ReportInput) -> list[tuple[int, float]]:
    """``(cases, mean repeats)`` for the control where it ran the aspect, else for every arm that did."""
    depth = aspect.depth or {}
    control = _control(report_input)
    if control is not None and control.id in depth:
        return [(depth[control.id].cases, float(depth[control.id].repeats_per_case[1]))]
    return [(d.cases, mean) for d in depth.values() if (mean := _mean_repeats(d)) is not None]


def _agreed(values: Iterable[float]) -> float | None:
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


def _cases(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _depth_figure(report_input, path, lambda ri, a: _agreed(c for c, _r in _depth_of(a, ri)))


def _repeats(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _depth_figure(report_input, path, lambda ri, a: _agreed(r for _c, r in _depth_of(a, ri)))


def _aspect_depth(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """The cases (or mean repeats) one aspect ran as one figure: the control's where it ran the aspect, else the one
    every arm agrees on."""
    aspect_id, field = values
    depth = _depth_of(aspect_of(report_input, aspect_id), report_input)
    value = _agreed(cases if field == "cases" else repeats for cases, repeats in depth)
    if value is None:
        raise _unstated(
            path,
            f"names the {field} aspect {aspect_id} ran as one figure, which the input does not state: its arms ran "
            f"it to different depths, or no depth is stated; read each arm's `aspects.{aspect_id}.depth.<arm>.{field}`",
        )
    return Figure(path=path, kind="number", value=float(value), scale="count")


_DEPTH_FIELDS = ("cases", "trials", "repeats")


def _depth_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(aspect_of(report_input, prior[0]).depth or {})


def _arm_depth(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """How deeply one arm ran one aspect: its distinct cases, its trials, the mean repeats a case got."""
    aspect_id, arm_id, field = values
    depth = (aspect_of(report_input, aspect_id).depth or {})[arm_id]
    if field == "repeats":
        return _number(path, _mean_repeats(depth), "the mean repeats a case got", "count")
    return _number(path, getattr(depth, field), f"the arm's {field} on the aspect", "count")


def _late_set_at(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    when = list(dict.fromkeys(late.set_at for late in report_input.caveats.late_criteria))
    if not when:
        raise _refuse(path, "names when the late criteria were set, and no criterion was set late")
    if len(when) > 1:
        per = ", ".join(f"`late_criteria.{late}.set_at`" for late in _late_names(report_input, ()))
        raise _unstated(
            path,
            f"names one date, which the input does not state: the late criteria were set at {len(when)} different "
            f"times; name one: {per}",
        )
    return Figure(path=path, kind="text", value=when[0])


def _late_name(late: LateCriterion) -> str:
    """How a key names one late criterion: what was set, then which one where it names one (``floor:<floor>``), as
    its caveat's id names it after ``late:``."""
    return f"{late.criterion}:{late.name}" if late.name else late.criterion


def _late_names(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(_late_name(late) for late in report_input.caveats.late_criteria))


def _late_criterion_set_at(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (named,) = values
    held = [late for late in report_input.caveats.late_criteria if _late_name(late) == named]
    when = list(dict.fromkeys(late.set_at for late in held))
    if len(when) > 1:
        raise _refuse(path, f"names when {named} was set, which the input records {len(when)} different times")
    return _text(path, when[0], f"when {named} was set")


def _setting_arms(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [arm.id for arm in report_input.arms if arm.settings]


def _setting_names(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_arm(report_input, prior[0]).settings)


def _setting(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    """One of an arm's settings as the input holds it: a number where it holds a number, else its text."""
    arm_id, name = values
    value = _arm(report_input, arm_id).settings[name]
    # The lever by its stored reader name, as every page names it, never its id.
    what = f"the arm's {lever_reader_name(name, report_input.lever_names)} setting"
    if isinstance(value, str) or value is None:
        return _text(path, value, what)
    return _number(path, value, what, "plain")


_FAMILY_FIELDS = ("disclosure", "correction", "alpha", "family_size", "level")


def _family_key(report_input: ReportInput, index: int, field: str) -> str:
    """The key that reads one field of the family at ``index``: by its aspect where it alone sits on one, else by its
    place in the input's families."""
    aspect = report_input.contrast_families[index].aspect
    alone = aspect is not None and sum(family.aspect == aspect for family in report_input.contrast_families) == 1
    return f"aspects.{aspect}.family.{field}" if alone else f"families.{index}.{field}"


def _family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    (field,) = values
    families = report_input.contrast_families
    if len(families) > 1:
        keys = ", ".join(f"`{_family_key(report_input, index, field)}`" for index in range(len(families)))
        raise _refuse(
            path, f"names the one contrast family, which the input holds {len(families)} times; name one: {keys}"
        )
    if not families:
        raise _refuse(path, "reads the one contrast family, and the input holds 0")
    return _family_figure(families[0], field, path)


def _aspect_family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, field = values
    held = [index for index, family in enumerate(report_input.contrast_families) if family.aspect == aspect_id]
    if not held:
        raise _unstated(path, f"names aspect {aspect_id}'s contrast family, which the input does not state")
    if len(held) > 1:
        keys = ", ".join(f"`families.{index}.{field}`" for index in held)
        raise _refuse(
            path, f"names aspect {aspect_id}'s contrast family, and the input holds {len(held)} on it; name one: {keys}"
        )
    return _family_figure(report_input.contrast_families[held[0]], field, path)


def _family_places(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [str(index) for index in range(len(report_input.contrast_families))]


def _indexed_family(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    index, field = values
    return _family_figure(report_input.contrast_families[int(index)], field, path)


def _family_figure(family: ContrastFamily, field: str, path: str) -> Figure:
    if field == "alpha":
        return _number(path, family.alpha, "the family's alpha", "plain")
    if field == "family_size":
        return _number(path, family.family_size, "the family's size", "count")
    return _text(path, getattr(family, field), f"the family's {field}")


def _latency(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    return _text(path, report_input.caveats.latency, "why latency is not compared")


def held_means(report_input: ReportInput, key: str) -> list[float]:
    """Every mean the input holds for a measure or judged dimension: overall, per aspect, per task, stratum and cell.
    A money ruler over the amounts a chart's changes compare, a rate's scale and a rubric's bounds all read this set.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension.
    :ptype key: str
    :return: The means, as stored.
    :rtype: list[float]
    """
    return [float(stat.mean) for stat in held_stats(report_input, key) if stat.mean is not None]


def held_stats(report_input: ReportInput, key: str) -> list[Stat]:
    """Every row of figures the input holds for a measure or judged dimension: overall, per aspect, per task, stratum
    and cell. The one walk of them (:func:`held_means` reads their means).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension.
    :ptype key: str
    :return: The rows, as stored.
    :rtype: list[Stat]
    """
    stats: list[Stat] = []
    for table in (report_input.measures, report_input.judged):
        if key in table:
            stats.extend(table[key].per_arm.values())
            stats.extend(stat for arms in table[key].by_aspect.values() for stat in arms.values())
    slices: list[Task | Stratum | Cell] = [*report_input.tasks, *report_input.strata, *report_input.cells.values()]
    for held in slices:
        stats.extend(figures[key] for figures in held.per_arm.values() if key in figures)
    return stats


def _scale_of(report_input: ReportInput, key: str, extra: Iterable[float] = ()) -> tuple[Scale, str]:
    """The scale a measure or judged dimension's figures are on wherever they sit: a measure's unit, a pass-fail
    dimension's rate scale read from every figure it holds (and ``extra``, a threshold on it), else plain."""
    if key in report_input.measures:
        return _measure_scale(report_input.measures[key])
    judged = report_input.judged.get(key)
    if judged is not None and judged.scale == "pass_fail":
        return _rate_scale([*held_means(report_input, key), *extra]), ""
    return "plain", ""


_TASK_FIELDS = ("name", "n_cases", "share")


def _tasks(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return [task.id for task in report_input.tasks]


def _task_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(_task(report_input, prior[0]).per_arm)


def _task_measures(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    task_id, arm_id = prior
    return list(_task(report_input, task_id).per_arm[arm_id])


def _task_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    task_id, field = values
    task = _task(report_input, task_id)
    if field == "name":
        return _text(path, given_name(task.id, task.name), "the task's name")
    if field == "n_cases":
        return _number(path, task.n_cases, "the task's cases", "count")
    return _number(path, task.share, "the task's share of traffic", "share", whole=True)


def _task_figure(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    task_id, arm_id, key, statistic = values
    per_arm = _task(report_input, task_id).per_arm
    return _slice_figure(report_input, per_arm, (arm_id, key, statistic), path, f"task {task_id}")


_CRITERION_FIELDS = ("threshold", "measure", "statistic", "direction", "aspect")


def _criterion_field(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, field = values
    criterion = _criterion(report_input, criterion_id)
    if field == "threshold":
        threshold = float(criterion.threshold)
        scale, unit = _scale_of(report_input, criterion.measure, (threshold,))
        return _number(path, threshold, "the criterion's threshold", scale, unit=unit)
    if field == "measure":
        return _text(path, _measure_reader_name(criterion.measure, report_input), "the criterion's measure")
    if field == "statistic":
        return _text(path, criterion.statistic, "the criterion's statistic")
    if field == "aspect":
        aspect = criterion_aspect(report_input, criterion)
        if aspect is None or aspect not in {held.id for held in report_input.aspects}:
            raise _unstated(path, "names the aspect the criterion reads, and the input holds no such aspect")
        return _text(path, reader_name("aspect", aspect, report_input), "the aspect whose figures the criterion reads")
    return _text(path, DIRECTION_WORDS[criterion.direction], "the criterion's direction")


_CRITERION_FIGURE_FIELDS = ("value", "low", "high", "gap")
#: The criterion figure's field computed here: how far the figure clears the threshold.
_CRITERION_DERIVED = ("gap",)


def _cell_spelling(cell: Cell) -> str:
    """How a key names a target-matrix cell: its row and column level ids joined by a dot, since a cell's id joins
    them with ``|``, which a key cannot hold."""
    return f"{cell.row}.{cell.column}"


def _place(report_input: ReportInput, figure: CriterionFigure) -> str | None:
    """Where a criterion's figure sits, as a key names it: a task's id, a cell's spelling, None for the arm's overall
    value."""
    if figure.cell is not None:
        return _cell_spelling(report_input.cells[figure.cell])
    return figure.task


def _criterion_arms(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion = _criterion(report_input, prior[0])
    return [arm.id for arm in report_input.arms if criterion_figures(report_input, criterion, arm.id)]


def _criterion_places(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    criterion_id, arm_id = prior
    figures = criterion_figures(report_input, _criterion(report_input, criterion_id), arm_id)
    return [place for figure in figures if (place := _place(report_input, figure)) is not None]


def _gap_figure(report_input: ReportInput, path: str, measure: str, line: float, gap: float, what: str) -> Figure:
    """A gap from a threshold on a measure, signed toward passing: in points for a rate, else in the measure's units.
    It is a change on the measure (#1127 item 4), read at the precision of the measure's changes, and a distance, which
    prints no ``+`` (:attr:`Figure.distance`)."""
    scale, unit = _scale_of(report_input, measure, (line,))
    if scale in ("share", "percent"):
        read = _number(path, gap * 100 if scale == "share" else gap, what, "points", change=True)
    else:
        read = _number(path, gap, what, scale, unit=unit, change=True)
    return replace(read, distance=True)


def _criterion_figure_field(
    report_input: ReportInput, criterion: Criterion, figure: CriterionFigure, field: str, path: str
) -> Figure:
    threshold = float(criterion.threshold)
    scale, unit = _scale_of(report_input, criterion.measure, (threshold,))
    if field == "value":
        # The criterion's figure is a level of the measure it reads, so it carries that measure: the path names it by
        # the criterion's id alone.
        read = _number(path, figure.value, "the criterion's figure", scale, unit=unit)
        return replace(read, measure=criterion.measure)
    if field in ("low", "high"):
        end = _end(path, figure.ci, field, "the figure's interval")
        read = replace(
            _number(path, end, f"the figure's interval's {field} end", scale, unit=unit), measure=criterion.measure
        )
        return replace(read, derived=True) if figure.derived else read
    gap = figure.value - threshold if criterion.direction == "higher_is_better" else threshold - figure.value
    # A gap is a change on the measure the criterion reads, so it reads at that measure's changes' set; as a distance it
    # prints no ``+`` (``99 ms``), the prose saying which side of the threshold it is on.
    read = _gap_figure(report_input, path, criterion.measure, threshold, gap, "the figure's gap from the threshold")
    return replace(read, measure=criterion.measure)


def _criterion_arm(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, field = values
    criterion = _criterion(report_input, criterion_id)
    figures = criterion_figures(report_input, criterion, arm_id)
    if len(figures) > 1:
        places = [f"criteria.{criterion_id}.{arm_id}.{_place(report_input, figure)}.{field}" for figure in figures]
        raise _refuse(
            path, f"reads one figure, and the criterion reads {len(figures)}, one per place: {_listed(places)}"
        )
    return _criterion_figure_field(report_input, criterion, figures[0], field, path)


def _criterion_arm_at(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    criterion_id, arm_id, place, field = values
    criterion = _criterion(report_input, criterion_id)
    figure = next(
        figure for figure in criterion_figures(report_input, criterion, arm_id) if _place(report_input, figure) == place
    )
    return _criterion_figure_field(report_input, criterion, figure, field, path)


# --- derived values: descriptive, computed here from the input's point figures ----------------------------------


def _control_cost(report_input: ReportInput, path: str) -> tuple[dict[str, float], float]:
    costs = _costs_per_thousand(report_input)
    control = _control(report_input)
    if control is None or not costs.get(control.id):
        raise _refuse(path, "compares a cost with the control's, and the input holds no control cost")
    return costs, costs[control.id]


def _costed_candidates(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    costs = _costs_per_thousand(report_input)
    return [arm for arm in _candidates(report_input) if arm in costs]


def _uplift(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    costs, control = _control_cost(report_input, path)
    return Figure(path=path, kind="number", value=costs[values[0]] / control - 1, scale="share", whole=True)


def _uplift_range(report_input: ReportInput, _: tuple[str, ...], path: str) -> Figure:
    costs, control = _control_cost(report_input, path)
    uplifts = [costs[arm] / control - 1 for arm in _candidates(report_input) if arm in costs]
    if not uplifts:
        raise _refuse(path, "spans the candidates' cost uplifts, and no candidate has a cost")
    return Figure(path=path, kind="interval", value=(min(uplifts), max(uplifts)), scale="share", whole=True)


_STRATUM_FIELDS = ("name", "n_cases", "share")

#: A strata family's resolver: what one stratum, and the slots after its id, read.
_StratumRead = Callable[[ReportInput, Stratum, tuple[str, ...], str], Figure]


def _strata(report_input: ReportInput, _: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(stratum.id for stratum in report_input.strata))


def _aspect_strata(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(stratum.id for stratum in _scoped_strata(report_input, prior[0])))


def _named_strata(
    report_input: ReportInput, prior: tuple[str, ...], qualified: bool
) -> tuple[list[Stratum], tuple[str, ...]]:
    """The strata a key's stratum id names (in its aspect, where the key names one), and the slots after the id."""
    aspect_id, rest = (prior[0], prior[1:]) if qualified else (None, prior)
    return [stratum for stratum in _scoped_strata(report_input, aspect_id) if stratum.id == rest[0]], rest[1:]


def _one_stratum(report_input: ReportInput, aspect_id: str | None, stratum_id: str, path: str) -> Stratum:
    """The one stratum a key names; a stratum id held on more than one aspect is refused naming each aspect's key."""
    held = [stratum for stratum in _scoped_strata(report_input, aspect_id) if stratum.id == stratum_id]
    if len(held) == 1:
        return held[0]
    if aspect_id is not None:
        raise _refuse(path, f"names stratum {stratum_id}, which aspect {aspect_id} holds {len(held)} times")
    tail = path.removeprefix(f"strata.{stratum_id}.")
    raise _held_apart(
        path,
        f"stratum {stratum_id}",
        [stratum.aspect for stratum in held],
        lambda aspect: f"aspects.{aspect}.strata.{stratum_id}.{tail}",
    )


def _strata_resolver(read: _StratumRead, qualified: bool) -> Callable[[ReportInput, tuple[str, ...], str], Figure]:
    """A strata family's resolver, reading the one stratum its key names, in the key's aspect when it names one."""

    def resolve(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
        aspect_id, rest = (values[0], values[1:]) if qualified else (None, values)
        return read(report_input, _one_stratum(report_input, aspect_id, rest[0], path), rest[1:], path)

    return resolve


def _stratum_arms(qualified: bool) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    def options(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        held, _rest = _named_strata(report_input, prior, qualified)
        return list(dict.fromkeys(arm for stratum in held for arm in stratum.per_arm))

    return options


def _stratum_keys(qualified: bool) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    def options(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        held, (arm_id,) = _named_strata(report_input, prior, qualified)
        return list(dict.fromkeys(key for stratum in held for key in stratum.per_arm.get(arm_id, {})))

    return options


def _loss_keys(qualified: bool) -> Callable[[ReportInput, tuple[str, ...]], list[str]]:
    """The measures and dimensions a candidate's loss may be read on in a stratum: those it and the control hold."""

    def options(report_input: ReportInput, prior: tuple[str, ...]) -> list[str]:
        held, (arm_id,) = _named_strata(report_input, prior, qualified)
        control = _control(report_input)
        if control is None:
            return []
        return list(
            dict.fromkeys(
                key
                for stratum in held
                for key in stratum.per_arm.get(arm_id, {})
                if key in stratum.per_arm.get(control.id, {})
            )
        )

    return options


def _stratum_field(report_input: ReportInput, stratum: Stratum, rest: tuple[str, ...], path: str) -> Figure:
    (field,) = rest
    if field == "name":
        return _text(path, _printed_stratum(report_input, stratum), "the stratum's name")
    if field == "n_cases":
        return _number(path, stratum.n_cases, "the stratum's cases", "count")
    return _number(path, stratum.share, "the stratum's share of traffic", "share", whole=True)


def _stratum_figure(report_input: ReportInput, stratum: Stratum, rest: tuple[str, ...], path: str) -> Figure:
    arm_id, key, statistic = rest
    return _slice_figure(report_input, stratum.per_arm, (arm_id, key, statistic), path, f"stratum {stratum.id}")


def _better_sign(report_input: ReportInput, key: str, path: str) -> float:
    """Which way a loss runs on a measure or judged dimension: 1 where higher is better (the control's figure minus the
    arm's), -1 where lower is. A judged dimension reads higher as better."""
    if key in report_input.judged:
        return 1.0
    higher = report_input.measures[key].higher_is_better if key in report_input.measures else None
    if higher is None:
        raise _refuse(path, f"reads a loss on {key}, whose better direction the input does not state")
    return 1.0 if higher else -1.0


def _loss_share_of(report_input: ReportInput, target: Stratum, arm_id: str, key: str | None, path: str) -> float:
    """The share of ``arm_id``'s loss against the control, on ``key`` (else the primary judged dimension), that one
    stratum carries.

    Loss in a stratum is the control's figure minus the arm's where higher is better (the arm's minus the control's
    where lower is), weighted by the stratum's cases (by its traffic share where cases are not stated), over the strata
    of the same stratification on the same aspect.
    """
    dimension = key if key is not None else report_input.primary_judged
    control = _control(report_input)
    if dimension is None or control is None:
        raise _refuse(path, "reads loss on the primary judged dimension against the control; the input lacks one")
    sign = 1.0 if key is None else _better_sign(report_input, key, path)
    group = [s for s in report_input.strata if (s.aspect, s.dimension) == (target.aspect, target.dimension)]
    weights: list[float | None] = [s.n_cases for s in group]
    if any(weight is None for weight in weights):
        weights = [s.share for s in group]
    if any(weight is None for weight in weights):
        raise _refuse(path, "weights each stratum by its cases or its share, and a stratum states neither")
    losses: list[float] = []
    for stratum, weight in zip(group, weights, strict=True):
        figures = stratum.per_arm
        held = figures.get(control.id, {}).get(dimension), figures.get(arm_id, {}).get(dimension)
        if held[0] is None or held[1] is None or held[0].mean is None or held[1].mean is None:
            raise _refuse(path, f"needs both arms' {dimension} in stratum {stratum.id}, which the input lacks")
        losses.append(float(weight or 0) * sign * (held[0].mean - held[1].mean))
    total = sum(losses)
    if total <= 0:
        raise _refuse(path, f"divides by {arm_id}'s loss on {dimension}, and it lost nothing overall")
    return losses[next(at for at, stratum in enumerate(group) if stratum is target)] / total


def _loss_share(report_input: ReportInput, stratum: Stratum, rest: tuple[str, ...], path: str) -> Figure:
    arms = rest[:1] or tuple(_candidates(report_input))
    key = rest[1] if len(rest) > 1 else None
    if not arms:
        raise _refuse(path, "reads each candidate's loss, and the campaign has no candidate")
    share = min(_loss_share_of(report_input, stratum, arm, key, path) for arm in arms)
    return Figure(path=path, kind="number", value=share, scale="share", whole=True)


def _flip_margin(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    return _flip_figure(_reading(_asked_or_refuse(report_input, path), values[0]), path)


def _aspect_flip_margin(report_input: ReportInput, values: tuple[str, ...], path: str) -> Figure:
    aspect_id, arm_id = values
    return _flip_figure(_reading(aspect_of(report_input, aspect_id), arm_id), path)


def _flip_figure(reading: Reading, path: str) -> Figure:
    flip = flip_margin(reading)
    if flip is None:
        raise _unstated(path, "names the interval a flip is read on, which the input does not state")
    return Figure(path=path, kind="number", value=flip, scale="points")


__all__ = [
    "aspect_costs_per_thousand",
    "cost_in_measure_unit",
    "cost_ruler",
    "cost_basis",
    "criterion_ruler",
]
