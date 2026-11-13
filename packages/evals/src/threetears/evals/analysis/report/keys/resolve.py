"""Reading keys: a path matched against the families, each input's reads cached once, and the public surface that
parses, resolves, spells and renders a key.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.quantities import PERCENT
from threetears.evals.analysis.report.cost_unit import priced_in_part
from threetears.evals.analysis.report.keys.families import _FAMILIES, _grouped
from threetears.evals.analysis.report.keys.figure import (
    Figure,
    FigureKey,
    PERCENTILE_NAMES,
    _ARM_PREFIX,
    _COUNT_WORDS,
    _DIGITS,
    _FILTERS,
    _KEY,
    _PER_THOUSAND_FACTORS,
    _SCALE_COUNTS,
    _SPELLINGS,
    _UNIT_PHRASES,
    _WHOLE_DOLLARS_FROM,
    _count,
    _listed,
    written_arm,
)
from threetears.evals.analysis.report.keys.names import page_arm_names
from threetears.evals.analysis.report.keys.prices import _KEPT_UNITS
from threetears.evals.analysis.report.keys.reads import (
    _SLICE_DERIVED,
    _input_unit_phrases,
    _money_per,
    cost_basis,
    held_means,
)
from threetears.evals.analysis.report.keys.separation import _arms
from threetears.evals.analysis.report.keys.vocabulary import (
    _Family,
    _Ids,
    _Number,
    _Slot,
    _Statistic,
    _Word,
    _arm,
    _asked,
    _control,
    _number_options,
    _refuse,
)
from threetears.evals.analysis.report.keys.weakest import _COMPARISONS_SLOT
from threetears.evals.analysis.report.per_input import kept_for
from threetears.evals.analysis.report.words import arm_tags
from threetears.evals.analysis.report_input import SHARE_UNIT, ReportInput
from threetears.evals.kernel.numbers import CENT

if TYPE_CHECKING:
    # A page's forms decide over the figures this module reads, so the module that decides them reads this one: every
    # caller hands its page's forms in, and the type is named for the checker alone.
    from threetears.evals.analysis.report.forms import PageForms


@dataclass(frozen=True)
class _Reading:
    """One complete reading of a path: the family and the ids and statistic it matched, in order."""

    family: _Family
    values: tuple[str, ...]


@dataclass
class _Stop:
    """Where the furthest partial reading stopped, and what could have come next there."""

    at: int = -1
    expected: tuple[str, ...] = ()


#: The ids each id slot offered on one input, after the slots matched before it (:attr:`_InputCache.offered`).
type _Offered = dict[tuple[Callable[[ReportInput, tuple[str, ...]], Iterable[str]], tuple[str, ...]], list[str]]


@dataclass
class _InputCache:
    """What one report input has read as (:func:`_input_cache`): each key's figure or the refusal it raised (kept
    with no frame, :func:`_frameless`), and the ids each slot offered after the slots matched before it."""

    figures: dict[str, Figure | UnresolvableReference]
    offered: _Offered
    amounts: dict[tuple[int, tuple[str, ...], str, bool], tuple[float, ...]]


def _empty_input_cache() -> _InputCache:
    """A fresh cache for one input (:func:`_input_cache`)."""
    return _InputCache(figures={}, offered={}, amounts={})


def _input_cache(report_input: ReportInput) -> _InputCache:
    """What one report input has read as, made empty on its first read.

    A key's figure and a slot's ids are pure in the input, and a report input is frozen and never changed in place (an
    input with a moved line is a copy, :meth:`~pydantic.BaseModel.model_copy`, with its own cache). So a key read twice
    on one input, which a cited figure's text, a page sized by halving and a listing ordered by its figures all do,
    walks the grammar once, and each family sharing a slot reads the slot's ids once (four lever families read a
    setting's arm pairs, an O(arms squared) scan, alike).
    """
    return kept_for(report_input).get(_InputCache, _empty_input_cache)


def _offered(slot: _Ids, report_input: ReportInput, values: tuple[str, ...], offered: _Offered) -> list[str]:
    """The ids an id slot offers on an input after the slots matched before it, read once per input into ``offered``,
    the input's own (:func:`_input_cache`)."""
    key = (slot.options, values)
    found = offered.get(key)
    if found is None:
        found = offered[key] = list(slot.options(report_input, values))
    return found


def _walk(
    path: str,
    at: int,
    family: _Family,
    slots: tuple[_Slot, ...],
    values: tuple[str, ...],
    report_input: ReportInput,
    offered: _Offered,
    found: list[_Reading],
    stop: _Stop,
) -> None:
    slot, rest = slots[0], slots[1:]
    if isinstance(slot, _Word):
        options: Sequence[str] = (slot.spelling,)
    elif isinstance(slot, _Ids):
        options = _offered(slot, report_input, values, offered)
    elif isinstance(slot, _Number):
        options = _number_options(path, at)
    else:
        options = slot.options
    advanced = False
    for option in options:
        end = at + len(option)
        if not option or not path.startswith(option, at):
            continue
        captured = values if isinstance(slot, _Word) else (*values, option)
        if not rest and end == len(path):
            found.append(_Reading(family, captured))
            advanced = True
        elif rest and end < len(path) and path[end] == ".":
            advanced = True
            _walk(path, end + 1, family, rest, captured, report_input, offered, found, stop)
    if not advanced:
        empty = "a number" if isinstance(slot, _Number) else "none in this input"
        expected = tuple(options) if options else (f"{_slot_name(slot)} ({empty})",)
        if at > stop.at:
            stop.at, stop.expected = at, expected
        elif at == stop.at:
            stop.expected = tuple(dict.fromkeys((*stop.expected, *expected)))


def _slot_name(slot: _Slot) -> str:
    return slot.spelling if isinstance(slot, _Word) else slot.name


def _readings(path: str, report_input: ReportInput) -> tuple[list[_Reading], _Stop]:
    found: list[_Reading] = []
    stop = _Stop()
    offered = _input_cache(report_input).offered
    for family in _FAMILIES:
        first = family.slots[0]
        if isinstance(first, _Word) and not path.startswith(first.spelling):
            # A family whose first word the path does not begin with stops at its start, where no refusal reads what
            # it expected (:func:`_one_reading` names every beginning there), so it is passed over unwalked.
            stop.at = max(stop.at, 0)
            continue
        _walk(path, 0, family, family.slots, (), report_input, offered, found, stop)
    return found, stop


def _named(reading: _Reading) -> str:
    """One reading of a path, for an ambiguity refusal: each slot's name and what filled it."""
    parts: list[str] = []
    values = iter(reading.values)
    for slot in reading.family.slots:
        parts.append(f"`{slot.spelling}`" if isinstance(slot, _Word) else f"{_slot_name(slot)} `{next(values)}`")
    return " · ".join(parts)


# --- the public surface ---------------------------------------------------------------------------------------------


def parse_key(inner: str, span: tuple[int, int] | None = None) -> FigureKey:
    """Split one key's inside into its path and filters, refusing a filter the grammar does not define.

    :param inner: The text between the braces, or a bare path (a durable claim's key, a chart reference's key).
    :ptype inner: str
    :param span: Where the key sits in the text it came from, braces included; None for a key parsed alone.
    :ptype span: tuple[int, int] | None
    :return: The key.
    :rtype: FigureKey
    :raises UnresolvableReference: The path is empty, a filter is unknown or repeated, or two filters each choose
        the spelling.
    """
    path, *filters = (part.strip() for part in inner.split("|"))
    if not path:
        raise UnresolvableReference(f"figure key {{{inner}}} has no path")
    for name in filters:
        if name not in _FILTERS:
            raise _refuse(inner, f"names filter {name!r}; the filters are {_listed(_FILTERS)}")
    if len(set(filters)) != len(filters):
        raise _refuse(inner, "repeats a filter")
    spellings = [name for name in filters if name in _SPELLINGS]
    if len(spellings) > 1:
        raise _refuse(inner, f"names two spellings, {' and '.join(spellings)}; keep one")
    if "abs" in filters and "signed" in filters:
        raise _refuse(inner, "names `abs` and `signed`, which read its sign two ways; keep one")
    if path.startswith(_ARM_PREFIX) and filters:
        raise _refuse(inner, "is an arm, which takes no filter")
    return FigureKey(path=path, filters=tuple(filters), span=span)


def keys_in(text: str) -> list[FigureKey]:
    """Every figure key in a piece of report prose, in order.

    :param text: Report prose.
    :ptype text: str
    :return: Each key with its span.
    :rtype: list[FigureKey]
    :raises UnresolvableReference: The text holds a double brace (the retired figure reference's grammar, never a
        report document's), a brace that opens or closes no key, or a key :func:`parse_key` refuses.
    """
    if "{{" in text or "}}" in text:
        raise UnresolvableReference(
            "report prose holds a double brace: a figure key is `{path|filter}` in single braces, and `{{…}}` is the "
            "retired figure reference, which a report document never writes"
        )
    keys = [parse_key(match.group(1), match.span()) for match in _KEY.finditer(text)]
    left = _KEY.sub("", text)
    if "{" in left or "}" in left:
        raise UnresolvableReference(f"plan prose holds a brace that opens or closes no figure key: {text!r}")
    return keys


def is_figure_key(text: str) -> bool:
    """Whether ``text`` is written as a figure key, read against no input: a braced key or a bare path whose filters
    the grammar defines and whose path starts with a family's word (``cost.gap.``), or names an arm (``arm:<id>``).

    It says nothing of whether the key resolves on any snapshot: only that the grammar reads it as a key, so a word
    that is no family's (an action's name, a model id, a version) is not one.

    :param text: A braced key (``{reading.a.change|abs}``) or a bare path.
    :ptype text: str
    :return: Whether it is a figure key's form.
    :rtype: bool
    """
    inner = text[1:-1] if text.startswith("{") and text.endswith("}") else text
    try:
        key = parse_key(inner)
    except UnresolvableReference:
        return False
    path = key.path
    if path.startswith(_ARM_PREFIX):
        return len(path) > len(_ARM_PREFIX)
    return any(
        path.startswith(f"{family.slots[0].spelling}.") or (path == family.slots[0].spelling and len(family.slots) == 1)
        for family in _FAMILIES
        if isinstance(family.slots[0], _Word)
    )


def resolve(key: str, report_input: ReportInput) -> Figure:
    """Read one key's path from the report input.

    :param key: The path, without braces or filters (``reading.qwen3.8-flash.change``, ``arm:1017``).
    :ptype key: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :return: The figure the path names.
    :rtype: Figure
    :raises UnresolvableReference: The path names nothing the input holds (the refusal lists what could come next
        where it stopped matching), reads two ways (naming both), or names a figure the input leaves unstated.
    """
    figures = _input_cache(report_input).figures
    held = figures.get(key)
    if isinstance(held, UnresolvableReference):
        raise _frameless(held)
    if held is not None:
        return held
    try:
        figure = _resolved(key, report_input)
    except UnresolvableReference as refused:
        figures[key] = _frameless(refused)
        raise
    figures[key] = figure
    return figure


def held_figure(key: str, report_input: ReportInput) -> Figure | None:
    """The figure a key built by code names, or None where the input holds none for it.

    For a table's cell, whose key code builds from an id the input holds and a family it reads, so a key that does
    not resolve means the input holds no such figure there, which the cell says with a dash. Never for a key an analyst
    wrote, which is refused by name (:func:`resolve`).

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The figure, or None.
    :rtype: Figure | None
    """
    try:
        return resolve(key, report_input)
    except UnresolvableReference:
        # NOSILENT: the caller's key is code's, so a refusal is the input holding no such figure, which the caller
        # states as a dash; a refusal of a written key reaches its writer through resolve.
        return None


def _frameless(refused: UnresolvableReference) -> UnresolvableReference:
    """A copy of a refusal that holds no frame: its type and message, its cause copied the same way, and whether a
    traceback shows its context, as a first raise has it.

    A raised refusal's traceback, and its cause's, hold the frames it unwound through, each with the report input among
    its locals, so the cache keeps this copy and raises a fresh one on each read: a refused key never keeps its input
    alive, and no two reads share an exception.
    """
    twin = type(refused)(*refused.args)
    cause = refused.__cause__
    if cause is not None:
        twin.__cause__ = type(cause)(*cause.args)
    twin.__suppress_context__ = refused.__suppress_context__
    return twin


def _resolved(key: str, report_input: ReportInput) -> Figure:
    """One key's figure, read from the input (:func:`resolve`, which keeps it)."""
    if key.startswith(_ARM_PREFIX):
        arm_id = key.removeprefix(_ARM_PREFIX)
        if arm_id not in {arm.id for arm in report_input.arms}:
            raise _refuse(key, f"names no arm; the arms are {_listed(_arms(report_input, ()))}")
        return Figure(path=key, kind="arm", value=arm_id)
    reading = _one_reading(key, report_input)
    figure = reading.family.resolve(report_input, reading.values, key)
    if figure.scale in _PER_THOUSAND_FACTORS and not figure.per:
        figure = replace(figure, per=_money_per(report_input, reading.values))
    if figure.scale in _PER_THOUSAND_FACTORS and figure.kind in ("number", "interval"):
        figure = replace(
            figure,
            amounts=_money_amounts(reading, figure, report_input),
            spread=_change_spread(reading, figure, report_input),
            reconciles=_beside_its_month(reading, figure, report_input),
        )
    if figure.kind in ("number", "interval") and (
        figure.scale in _PER_THOUSAND_FACTORS or _family_word(reading) in _COST_RATIOS
    ):
        figure = replace(figure, priced_in_part=_priced_in_part(reading, report_input))
    if figure.kind in ("number", "interval"):
        # Every figure names the measure it is of, so the page's forms record it there whatever its scale; which of them
        # read at a set's decimals is the forms' to decide. A family whose path names an arm alone (an arm's pass rate,
        # its cost) says which measure it read, from the source every table reads that figure from.
        figure = replace(
            figure,
            statistic=_statistic_of(reading),
            set_of=figure.set_of or _set_of(reading, figure, report_input),
        )
        # Where the tables hold it is the ledger's alone (Figure.ledger_as): set_of above still picks its digits.
        figure = replace(figure, ledger_as=reading.family.ledger_as(report_input, reading.values, figure))
    return replace(figure, derived=figure.derived or _is_derived(reading))


def _statistic_of(reading: _Reading) -> str | None:
    """The statistic a path names last (``mean``, ``delta``, ``threshold``), the one a set's decision reads a figure's
    role by; None for a path that names none, which reads its family's own figure."""
    statistics = [
        value for slot, value in zip(_value_slots(reading), reading.values, strict=True) if isinstance(slot, _Statistic)
    ]
    return statistics[-1] if statistics else None


def _set_of(reading: _Reading, figure: Figure, report_input: ReportInput) -> tuple[str, str] | None:
    """The measure a figure is of, as the page's tables print it, with its kind: the measure or judged dimension a key
    names (or its figure carries, :attr:`Figure.measure`), a judged score on the 1–5 rubric as ``judged``, a measure as
    ``measure``, how each trial ended as ``trial`` (the pass rate's comparisons); and a figure in points whose path
    names no measure as the rate its family reads (:attr:`_Family.rate`: the pass rate for a reading, an aspect's
    margin or two arms apart, a floor's own rate), so a key spelling it in points reads at that rate's precision as its
    table does. None for a key naming none. Which of its figures read at a set's decimals is the page's forms' to decide
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.key_number`)."""
    for value in (figure.measure,) if figure.measure else _named_measures(reading):
        if rubric_score(report_input, value):
            return (value, "judged")
        if value in report_input.measures:
            return (value, "measure")
        if value in report_input.judged:
            return (value, "judged")
        if value == TRIAL_PASS_NAME:
            return (value, "trial")
    rate = reading.family.rate
    return rate(report_input, reading.values) if rate is not None and figure.scale == "points" else None


#: The id slots that name a measure or a judged dimension, as a path spells them.
_MEASURE_SLOTS = frozenset({"<measure>", "<dimension>"})


def _named_measures(reading: _Reading) -> tuple[str, ...]:
    """The values a path names a measure by: its measure's or dimension's slot, never an aspect, an arm or a place
    whose id happens to be a measure's too; every value where the family has no such slot."""
    named = tuple(
        value
        for slot, value in zip(_value_slots(reading), reading.values, strict=True)
        if isinstance(slot, _Ids) and slot.name in _MEASURE_SLOTS
    )
    return named or reading.values


#: A judged score's rubric scale (``ordinal``): the two decimals and the ``(1–5)`` a page prints it with.
RUBRIC_SCALE = (1.0, 5.0)


def rubric_score(report_input: ReportInput, key: str) -> bool:
    """Whether a judged dimension is a score on the 1–5 rubric: not pass/fail, and every mean the input holds of it on
    the scale (an input that states a dimension ``ordinal`` but holds 92 is read by the number rule, never as a 1–5
    score at two decimals).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The judged dimension's key; any other key is no score.
    :ptype key: str
    :return: Whether it is a 1–5 score.
    :rtype: bool
    """
    judged = report_input.judged.get(key)
    if judged is None or judged.scale == "pass_fail":
        return False
    low, high = RUBRIC_SCALE
    return all(low <= mean <= high for mean in held_means(report_input, key))


def _rate_unit(values: Sequence[float]) -> str:
    return "share" if all(0 <= value <= 1 for value in values) else "percent"


#: The scoreboard's figures that are money, each to how it is stated: a reply's cost, and what the arm's trials cost
#: in all. Every other scoreboard figure is a count.
_SCOREBOARD_UNITS = {"reply_cost_usd": "usd", "test_cost_usd": "dollars"}


def figure_unit(report_input: ReportInput, key: str, *, kind: str | None = None, source: str | None = None) -> str:
    """A measure's or judged dimension's unit as the page reads it (the chart rules' Units,
    :mod:`~threetears.evals.analysis.report.parts.charting`), wherever its figures sit; a scoreboard figure's, where
    the chart reads the scoreboard.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key; a scoreboard figure's name over the scoreboard.
    :ptype key: str
    :param kind: A contrast's kind (``measure``, ``judged``, ``trial``), where the key is a contrast's; None otherwise.
    :ptype kind: str | None
    :param source: The chart's source, where ``scoreboard`` makes the key a scoreboard figure; None otherwise.
    :ptype source: str | None
    :return: ``share``, ``percent``, ``score``, a money unit, ``ms``, ``s``, any other declared unit, or empty (a
        bare number, as every scoreboard count is).
    :rtype: str
    """
    if source == "scoreboard":
        return _SCOREBOARD_UNITS.get(key, "")
    if kind == "trial":
        return "share"
    measure = report_input.measures.get(key)
    if measure is not None and kind != "judged":
        unit = measure.unit or ""
        if unit == "pct":
            return "percent"
        if unit == SHARE_UNIT:
            return "share"
        if not unit and measure.grade in ("pass_fail", "class_label"):
            return _rate_unit(held_means(report_input, key))
        return unit
    judged = report_input.judged.get(key)
    if judged is not None:
        return _rate_unit(held_means(report_input, key)) if judged.scale == "pass_fail" else "score"
    return ""


def held_kind(report_input: ReportInput, key: str) -> str:
    """The kind a figure on ``key`` is keyed under in the page's forms (one of the kinds
    :meth:`~threetears.evals.analysis.report.forms.PageForms.of` keys), by what the input holds it as: ``judged`` where
    it holds the key only as a judged dimension; ``trial`` for the pass rate (how each trial ended) where no measure
    takes its key; ``measure`` otherwise. A scoreboard figure (a reply's cost, a count) and an aspect a criterion reads
    are measured figures of an arm, neither judged nor a contrast's trial, so they key as a measure's. The one rule for
    a criterion's kind, whether its threshold is a gate's line or a chart's.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure, judged dimension, scoreboard figure or aspect.
    :ptype key: str
    :return: ``judged``, ``trial`` or ``measure``.
    :rtype: str
    """
    if key in report_input.judged and key not in report_input.measures:
        return "judged"
    return "trial" if key == TRIAL_PASS_NAME and key not in report_input.measures else "measure"


def is_rubric_score(report_input: ReportInput, key: str, *, kind: str | None = None) -> bool:
    """Whether a figure is a judged score on the 1–5 rubric (:func:`rubric_score`), not a share, nor a figure an input
    calls ordinal but holds off the scale.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param key: The measure's or judged dimension's key.
    :ptype key: str
    :param kind: A contrast's kind where the key is one's; None for a measure's.
    :ptype kind: str | None
    :return: Whether it is a 1–5 score.
    :rtype: bool
    """
    return (
        kind != "measure" and figure_unit(report_input, key, kind=kind) == "score" and rubric_score(report_input, key)
    )


def _beside_its_month(reading: _Reading, figure: Figure, report_input: ReportInput) -> bool:
    """Whether a money figure is the page's cost per 1,000 (``cost_per_1k.<arm>``, or the cost basis measure it is read
    from) where a monthly volume is declared, so its month (``cost_per_month.<arm>``) is printed from it."""
    if report_input.meta.volume_per_month is None or figure.change or figure.total:
        return False
    return _family_word(reading) == "cost_per_1k" or cost_basis(report_input).source in reading.values


#: The families whose figure is a ratio of two arms' costs, not money, which a partly priced cost leaves a ratio to a
#: smaller total.
_COST_RATIOS = frozenset({"cost.uplift", "cost.gap", "cost.uplift_range"})


def _family_word(reading: _Reading) -> str:
    first = reading.family.slots[0]
    return first.spelling if isinstance(first, _Word) else ""


def _priced_in_part(reading: _Reading, report_input: ReportInput) -> str:
    """The words beside a cost whose arms' costs count only part of their calls (:class:`Figure`'s ``priced_in_part``):
    the arm's priced share where the key names one arm, each such arm's by name where it names more (an uplift names
    the control it is on too), and every arm's where a ratio names none (a range over them all); empty where every cost
    counts every call, or the input does not say."""
    arms = _cost_arms_read(reading, report_input)
    parts = [(arm, share) for arm in arms if (share := priced_in_part(report_input, arm)) is not None]
    if len(arms) == 1:
        return parts[0][1] if parts else ""
    return "; ".join(f"{_arm(report_input, arm).name or arm}: {share}" for arm, share in parts)


def _cost_arms_read(reading: _Reading, report_input: ReportInput) -> list[str]:
    """The arms whose cost a money or cost-ratio reading reads: the arms it names, the control besides for a ratio of
    costs (an uplift is on it), and every arm for a ratio naming none (a range over them all)."""
    named = [
        value for slot, value in zip(_value_slots(reading), reading.values, strict=True) if slot.name in _ARM_SLOTS
    ]
    ratio = _family_word(reading) in _COST_RATIOS
    control = _control(report_input)
    arms = list(dict.fromkeys([*named, *([control.id] if ratio and control is not None and named else [])]))
    return arms or ([arm.id for arm in report_input.arms] if ratio else [])


def cost_arms(key: str, report_input: ReportInput) -> list[str]:
    """The arms whose cost a figure key reads: a money figure's arms, a ratio of costs' with the control it is on (an
    uplift, a gap), every arm for a range over them all; empty for a key that reads no cost, or no arm's.

    The one reading of which costs a key compares, shared with the words a partly priced cost carries
    (:class:`Figure`'s ``priced_in_part``), so a surface naming who served each cost reads the arms that note does.

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The arms by id, the ones the key names first, in its order.
    :rtype: list[str]
    :raises UnresolvableReference: The path names nothing the input holds, reads two ways, or names a figure the
        input leaves unstated.
    """
    figure = resolve(key, report_input)
    if figure.kind not in ("number", "interval"):
        return []
    reading = _one_reading(key, report_input)
    if figure.scale not in _PER_THOUSAND_FACTORS and _family_word(reading) not in _COST_RATIOS:
        return []
    return _cost_arms_read(reading, report_input)


def _value_slots(reading: _Reading) -> list[_Ids | _Statistic | _Number]:
    """A reading's slots that took a value, in order: every slot but its words."""
    return [slot for slot in reading.family.slots if not isinstance(slot, _Word)]


def _sibling_values(reading: _Reading, report_input: ReportInput) -> list[tuple[str, ...]]:
    """The readings of a reading's family that share its words, its statistic and its ids but its arms: every arm its
    arm slots offer (each slot's ids as offered after the ones before it). A statistic is a figure of its own type (a
    scoreboard's cost per reply is not its trials' total), so it is never varied."""
    combos: list[tuple[str, ...]] = [()]
    values = iter(reading.values)
    offered = _input_cache(report_input).offered
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            continue
        value = next(values)
        if isinstance(slot, _Ids) and slot.name in _ARM_SLOTS:
            combos = [
                (*prefix, option) for prefix in combos for option in _offered(slot, report_input, prefix, offered)
            ]
        else:
            combos = [(*prefix, value) for prefix in combos]
    return combos


def _sibling_figures(reading: _Reading, report_input: ReportInput) -> list[Figure]:
    """The figures of :func:`_sibling_values`' readings that the input states; one it leaves unstated is no member."""
    figures: list[Figure] = []
    for values in _sibling_values(reading, report_input):
        try:
            figures.append(reading.family.resolve(report_input, values, ".".join(values)))
        except UnresolvableReference:
            # NOSILENT: a sibling the input leaves unstated, or one its resolver refuses, has no amount to add to the
            # set; the key being read raises its own refusal where it is the one unstated.
            continue
    return figures


def _fixed_values(reading: _Reading) -> tuple[str, ...]:
    """A reading's values with its arm slots left blank: what every member of its set shares."""
    fixed: list[str] = []
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            continue
        value = next(values)
        fixed.append("" if isinstance(slot, _Ids) and slot.name in _ARM_SLOTS else value)
    return tuple(fixed)


def _numbers_of(figures: Iterable[Figure]) -> list[float]:
    """Every number the figures hold: a number's value, an interval's two ends."""
    held: list[float] = []
    for figure in figures:
        if isinstance(figure.value, float):
            held.append(figure.value)
        elif isinstance(figure.value, tuple):
            held.extend(float(end) for end in figure.value)
    return held


def _money_amounts(reading: _Reading, figure: Figure, report_input: ReportInput) -> tuple[float, ...]:
    """The set a money figure is spelled with (:class:`Figure`'s ``amounts``): a change's compared amounts where its
    family says what a change compares; else every amount its family states of the same statistic on its scale across
    the arms.

    The set is the family's, never the figure's: every key of ``cost_per_1k.*`` reads the same set, so each is spelled
    in one form. Read once per family and fixed ids on an input."""
    cached = _input_cache(report_input).amounts
    key = (id(reading.family), _fixed_values(reading), figure.scale, figure.change)
    held = cached.get(key)
    if held is not None:
        return held
    compared = reading.family.compares(report_input, reading.values) if reading.family.compares is not None else None
    if figure.change and compared is not None:
        found = tuple(compared)
    else:
        siblings = [one for one in _sibling_figures(reading, report_input) if one.scale == figure.scale]
        amounts = _numbers_of(one for one in siblings if not one.change)
        found = tuple(amounts or _numbers_of(siblings))
    cached[key] = found
    return found


def _change_spread(reading: _Reading, figure: Figure, report_input: ReportInput) -> float | None:
    """The width of the interval a money change is read with: its own, for an interval; for a number, its family's
    interval at the same arms (a contrast's delta beside its ``interval``), where it states one."""
    if not figure.change:
        return None
    if isinstance(figure.value, tuple):
        return abs(figure.value[1] - figure.value[0])
    statistic = reading.family.statistic()
    if statistic is None:
        return None
    at = [slot for slot in reading.family.slots if not isinstance(slot, _Word)].index(statistic)
    for option in ("interval", "ci", "gain_interval"):
        if option in statistic.options:
            values = list(reading.values)
            values[at] = option
            try:
                held = reading.family.resolve(report_input, tuple(values), ".".join(values))
            except UnresolvableReference:
                # NOSILENT: a change whose family states no interval at these arms reads its set's own decimals.
                return None
            return abs(held.value[1] - held.value[0]) if isinstance(held.value, tuple) else None
    return None


def _one_reading(key: str, report_input: ReportInput) -> _Reading:
    """The one reading of a path; a path reading two ways, or none, is refused naming the readings or what could
    come next where it stopped matching."""
    found, stop = _readings(key, report_input)
    if len(found) > 1:
        readings = "; ".join(_named(reading) for reading in found)
        raise _refuse(key, f"reads {len(found)} ways: {readings}. Choose ids that leave one reading.")
    if not found:
        if stop.at <= 0:
            raise _refuse(key, f"names no figure; a key begins with {_listed(_beginnings())}")
        raise _refuse(key, f"names no figure: after `{key[: stop.at]}` comes {_listed(stop.expected)}")
    return found[0]


def _is_derived(reading: _Reading) -> bool:
    """Whether a reading's figure is computed here: its family is derived, or the statistic it names is. A resolver
    may also mark one figure derived where its family copies the rest (a criterion figure's ends, read off an interval
    derived from the cells)."""
    if reading.family.derived:
        return True
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            continue
        value = next(values)
        if isinstance(slot, _Statistic) and value in reading.family.derived_statistics:
            return True
    return False


@dataclass(frozen=True)
class KeySubjects:
    """What one key is about: the arms and aspects its path names, and whether it reads an arm against the control.

    ``arms`` are the ids the path's arm or candidate slots matched (or the arm of ``{arm:<id>}``); ``aspects`` the
    aspect ids it named, or the asked aspect for a family that reads it without naming it (``reading``, ``aspect``,
    ``cases``, ``repeats``, ``flip_margin``). ``against_control`` is True for a figure that is a difference from the control (a
    reading, a contrast, a cost uplift, a flip margin, a stratum's loss share, a slice's ``vs_control``), whose
    evidence is the control's too.
    """

    arms: tuple[str, ...]
    aspects: tuple[str, ...]
    against_control: bool


#: The slots an arm id fills.
_ARM_SLOTS = frozenset({"<arm>", "<candidate>", "<from>", "<to>"})

#: The families that read the asked aspect without naming it.
_ASKED_FAMILIES = frozenset({"aspect", "reading", "cases", "repeats", "flip_margin"})

#: The words that make a figure a difference from the control.
_AGAINST_CONTROL = frozenset(
    {"reading", "contrast", "cost.uplift", "cost.uplift_range", "flip_margin", "loss_share", *_SLICE_DERIVED}
)

#: The statistics that make a figure a difference from the control.
_AGAINST_CONTROL_STATISTICS = frozenset(_SLICE_DERIVED)


def key_subjects(key: str, report_input: ReportInput) -> KeySubjects:
    """The arms and aspects one key's path names, read off its one reading of the input.

    What a durable claim rests on is read from its keys: the runs behind each arm the key names on each aspect it
    names (each insight an analysis mints from a durable claim).

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :return: What the key is about.
    :rtype: KeySubjects
    :raises UnresolvableReference: The path does not resolve (:func:`resolve`).
    """
    resolve(key, report_input)
    if key.startswith(_ARM_PREFIX):
        return KeySubjects(arms=(key.removeprefix(_ARM_PREFIX),), aspects=(), against_control=False)
    (reading,), _ = _readings(key, report_input)
    arms: list[str] = []
    aspects: list[str] = []
    words: list[str] = []
    values = iter(reading.values)
    for slot in reading.family.slots:
        if isinstance(slot, _Word):
            words.append(slot.spelling)
            continue
        value = next(values)
        if isinstance(slot, _Statistic) and value in _AGAINST_CONTROL_STATISTICS:
            words.append(value)
        elif isinstance(slot, _Ids) and slot.name in _ARM_SLOTS:
            arms.append(value)
        elif isinstance(slot, _Ids) and slot.name == "<aspect>":
            aspects.append(value)
    if not aspects and words[0] in _ASKED_FAMILIES and (asked := _asked(report_input)) is not None:
        aspects.append(asked.id)
    return KeySubjects(
        arms=tuple(arms), aspects=tuple(aspects), against_control=any(word in _AGAINST_CONTROL for word in words)
    )


@dataclass(frozen=True)
class KeyForm:
    """What one key's path is, as the grammar teaches it: the form of the family it reads, what that family reads,
    and the help page the form is listed on.

    :ivar form: The family's form, its ids as slot names (``price.<from>.<to>.<measure>.<field>``); ``arm:<id>`` for
        an arm.
    :ivar what: What the family reads, as the grammar says it.
    :ivar group: The group of the grammar's paths the form is listed in (``figure_keys.<group>``).
    """

    form: str
    what: str
    group: str


#: Each family's group of the grammar's paths (:func:`~threetears.evals.analysis.report.keys.families._grouped`), read
#: once at import: the families and their groups are constants, and a family in no group fails the import.
_GROUP_OF: Final[Mapping[_Family, str]] = {
    family: group for group, families in _grouped().items() for family in families
}

#: An arm key's form, as a key check names it.
_ARM_FORM = KeyForm(form="arm:<id>", what="an arm, printed by its short name and letter tag", group="campaign")


def key_form(key: str, report_input: ReportInput) -> KeyForm:
    """The form one key's path reads by: its family, what the family reads and the grammar's page listing it.

    A path is matched as :func:`resolve` matches it, so a path that names nothing, or reads two ways, is refused
    alike; one naming a figure the input leaves unstated still has a form.

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The report input the key is read against.
    :ptype report_input: ReportInput
    :return: The form.
    :rtype: KeyForm
    :raises UnresolvableReference: The path names nothing the input holds, or reads two ways.
    """
    if key.startswith(_ARM_PREFIX):
        return _ARM_FORM
    family = _one_reading(key, report_input).family
    group = _GROUP_OF[family]
    return KeyForm(form=family.form(), what=family.what, group=group)


def figure_filters(figure: Figure, *, forms: PageForms) -> tuple[str, ...]:
    """The filters a figure takes: each filter :func:`format_figure` spells it with, alone, in the grammar's order.

    :param figure: What a key resolved to.
    :ptype figure: Figure
    :param forms: The forms of the page the figure would be printed on (:func:`format_figure`).
    :ptype forms: PageForms
    :return: The filters; none for an arm or a piece of text.
    :rtype: tuple[str, ...]
    """
    if figure.kind in ("arm", "text"):
        return ()
    taken: list[str] = []
    for name in _FILTERS:
        try:
            format_figure(figure, (name,), forms=forms)
        except UnresolvableReference:
            # NOSILENT: a filter the figure does not suit is refused by format_figure; that refusal is the answer
            # sought here, and the filter is left off the list it returns.
            continue
        taken.append(name)
    return tuple(taken)


def _beginnings() -> list[str]:
    return list(dict.fromkeys(_slot_name(family.slots[0]) for family in _FAMILIES))


@dataclass(frozen=True)
class _Spelled:
    """A number as it reads, and the suffix unit it carries (``%``), which a range writes once."""

    number: str
    suffix: str = ""


def plus_signed(text: str, value: float) -> str:
    """``text`` with the ``+`` a positive change carries, as :func:`~threetears.evals.kernel.numbers.format_signed`
    places it: never on a bound, never on a zero."""
    shown = re.search(r"[1-9]", text) is not None
    return f"+{text}" if value > 0 and not text.startswith(("<", "+")) and shown else text


def _counted_under_a_cent(figure: Figure) -> bool:
    """Whether a money figure counts something (its ``per``) and its family holds an amount under a cent: one its
    ``usd`` spelling states per 1,000 of what it counts, with the count, rather than as a bare amount."""
    if not figure.per:
        return False
    own = (
        figure.value if isinstance(figure.value, tuple) else (figure.value,) if isinstance(figure.value, float) else ()
    )
    stated = [abs(amount) for amount in (figure.amounts or own) if isinstance(amount, float) and amount]
    return bool(stated) and min(stated) < CENT


#: The spelling a figure on the ``usd_per_1m`` scale reads in without a filter: dollars with the words of what it
#: counts, ``$180 per 1,000,000 turns``. No filter names it: a key's own spelling, never chosen.
_PER_MILLION = "per_million"


def _spelling_for(figure: Figure, filters: tuple[str, ...]) -> str:
    chosen = next((name for name in filters if name in _SPELLINGS), None)
    if chosen is not None:
        return chosen
    # A distance reads unsigned unless a filter signs it (Figure.distance).
    signs = figure.change and not figure.distance
    defaults: dict[str, str] = {
        "share": "pts" if figure.change else "pct",
        "percent": "pts" if figure.change else "pct",
        "points": "pts" if signs else "points",
        "usd": "usd",
        "usd_per_1k": "usd",
        "usd_per_1m": _PER_MILLION,
    }
    return defaults.get(figure.scale, "plain")


def _spell(
    value: float, figure: Figure, spelling: str, absolute: bool, forms: PageForms, *, finer: bool = False
) -> _Spelled:
    """A figure's number by the spelling its filters chose, or the filter refused: which spelling suits which scale is
    the grammar's, and every number's text is the page's forms' (each ``PageForms.key_*``)."""
    if absolute:
        value = abs(value)
    signed = figure.change and not figure.distance and not absolute
    rate = figure.scale in ("share", "percent")
    hundred = value * 100 if figure.scale == "share" else value
    if spelling in ("pct", "pct10"):
        if not rate or figure.change:
            why = "a change in a rate reads in points (pts)" if rate else f"it is on the {figure.scale} scale"
            raise _refuse(figure.path, f"cannot read as a percent: {why}")
        return _Spelled(forms.key_percent(figure, hundred, tens=spelling == "pct10"), PERCENT)
    if spelling in ("pts", "points"):
        if figure.scale not in ("share", "percent", "points"):
            raise _refuse(figure.path, f"cannot read in points: it is on the {figure.scale} scale")
        text = forms.key_points(figure, hundred)
        return _Spelled(plus_signed(text, hundred) if spelling == "pts" and not absolute else text)
    if spelling in ("usd", "usd_per_1k"):
        if figure.scale not in _PER_THOUSAND_FACTORS:
            raise _refuse(figure.path, f"cannot read as dollars: it is on the {figure.scale} scale")
        if spelling == "usd" and not _counted_under_a_cent(figure):
            return _Spelled(forms.key_money(figure, value, per=0, finer=finer, signed=signed)[0])
        counts = _SCALE_COUNTS[figure.scale]
        amount, words = forms.key_money(figure, value, per=counts, start=1_000, finer=finer, signed=signed)
        if spelling == "usd":
            # The count said once after an interval's two ends (``+$0.17 to +$0.81 per 1,000 replies``).
            return _Spelled(amount, f" {words}")
        # Per 1,000 of what the cost counts (a reply's cost under a cent reads ``$0.49 per 1,000 replies``, never a
        # bare ``$0.00049`` a reader takes for a total), or per 1,000,000 where the family is under a cent even so: the
        # ruler moves (the one owner's rule), and the count is said with the amount, since an amount whose count is
        # left unsaid reads as one per reply, a thousand or a million times what it is.
        return _Spelled(f"{amount} {words}")
    if spelling == "usd_whole":
        if figure.scale not in _PER_THOUSAND_FACTORS:
            raise _refuse(figure.path, f"cannot read as dollars: it is on the {figure.scale} scale")
        if abs(value) < _WHOLE_DOLLARS_FROM:
            amount, floor = forms.key_refused_whole(figure, value, _WHOLE_DOLLARS_FROM, finer=finer, signed=signed)
            raise _refuse(
                figure.path, f"is {amount}, under ${floor}, which whole dollars would round away: write it with `usd`"
            )
        dollars = "$" + forms.key_whole(figure, abs(value))
        return _Spelled(f"-{dollars}" if value < 0 else plus_signed(dollars, value) if signed else dollars)
    if spelling == _PER_MILLION:
        # In its family's one form, as every other money key: a price per million of anything is whole dollars or near
        # it, so the form's cents are its precision.
        amount, words = forms.key_money(figure, value, per=1_000_000, finer=finer, signed=signed)
        return _Spelled(f"{amount} {words}")
    if spelling == "words":
        whole = forms.key_whole_count(value)
        if figure.scale != "count" or whole is None:
            why = "it is not a whole count" if figure.scale == "count" else f"it is on the {figure.scale} scale"
            raise _refuse(figure.path, f"cannot read as a word: {why}")
        if whole < len(_COUNT_WORDS):
            return _Spelled(_COUNT_WORDS[whole])
        return _Spelled(forms.key_whole(figure, value))
    if spelling == "int":
        # A price per 1,000,000 (of what it counts) also reads as a whole number, as stored plans spell it.
        if figure.scale not in ("count", "plain", "points", "usd_per_1m"):
            raise _refuse(figure.path, f"cannot read as a whole number: it is on the {figure.scale} scale")
        text = forms.key_whole(figure, value)
        text = plus_signed(text, value) if signed else text
        return _Spelled(f"{text} {figure.unit}" if figure.unit in _KEPT_UNITS else text)
    if figure.unit in _KEPT_UNITS:
        return _Spelled(f"{forms.key_multiple(figure, value)} {figure.unit}")
    # A plain figure the page's tables print at a set's decimals reads at them here too (a score's ``-0.21``), and a
    # count in a sentence groups its thousands (4,000,000 a month), as ``words`` and ``int`` spell it.
    text = forms.key_number(figure, value)
    text = plus_signed(text, value) if signed else text
    if not figure.unit:
        return _Spelled(text)
    return _Spelled(f"{text} {figure.unit}")


def format_figure(figure: Figure, filters: Sequence[str] = (), *, forms: PageForms) -> str:
    """Spell a figure for a sentence, by its filters or, without one, by its scale.

    A rate reads as a percent (one decimal; a descriptive ratio in whole percents), a change in points, money in
    dollars, a count whole. An interval reads ``low to high``, its unit once after the high end, signed or not
    (``-13 to -2 pts``, ``27 to 28%``; number rule 3). A plain figure of a measure the page's tables print, and an interval's two ends,
    read at the precision the page's forms decide for it. Every number it prints, and how an interval's ends read
    together and are joined, is the forms' (each ``PageForms.key_*``), each recorded in their ledger: this chooses the
    spelling the filters name and refuses one that does not suit. A figure said in words prints its number between
    them (:attr:`Figure.said_between`), and a text figure of counts prints each count after its word
    (:attr:`Figure.counts`), so a number in words reads as the page's forms print it too.

    :param figure: What a key resolved to; not an arm, which :func:`render` names.
    :ptype figure: Figure
    :param filters: The key's filters, as :func:`parse_key` checked them.
    :ptype filters: Sequence[str]
    :param forms: The forms of the page the figure is printed on (an off-page reader's own, built from the input it
        reads), which decide a measure's figure's precision.
    :ptype forms: PageForms
    :return: The figure's text.
    :rtype: str
    :raises UnresolvableReference: A filter does not suit the figure: a percent of money, points of a count, an
        ``interval`` on a single number, any filter on text.
    :raises ValueError: The figure is an arm.
    """
    before, after = figure.said_between
    if figure.joint is not None and _COMPARISONS_SLOT in after:
        # The family's size reads as a count anywhere on the page does, by the page's forms.
        size = _formatted_figure(_count(figure.path, figure.joint[1]), (), forms)
        after = after.replace(_COMPARISONS_SLOT, size)
    text = before + _formatted_figure(figure, tuple(filters), forms) + after
    return f"{text} ({figure.priced_in_part})" if figure.priced_in_part else text


def _formatted_figure(figure: Figure, filters: tuple[str, ...], forms: PageForms) -> str:
    """A figure's text by its filters (:func:`format_figure`), without what it says beside the amount."""
    chosen = tuple(filters)
    if figure.kind == "arm":
        raise ValueError(f"{figure.path} is an arm, which render names with its tag")
    if figure.kind == "text":
        if chosen:
            raise _refuse(figure.path, "is text, which takes no filter")
        if figure.counts:
            # Each count reads as a count anywhere on the page does, by the page's forms.
            return ", ".join(
                f"{word} {_formatted_figure(_count(figure.path, count), (), forms)}" for word, count in figure.counts
            )
        return str(figure.value)
    if "signed" in chosen:
        if not figure.change:
            raise _refuse(figure.path, "is no change, and `signed` signs a change")
        # A distance signed on request reads as the change it is (Figure.distance).
        figure = replace(figure, distance=False)
    spelling = _spelling_for(figure, chosen)
    absolute = "abs" in chosen
    if figure.kind == "number":
        if "interval" in chosen:
            raise _refuse(figure.path, "is one number, and `interval` reads an interval")
        if not isinstance(figure.value, float):
            raise TypeError(f"{figure.path} is a number figure holding {figure.value!r}, not a float")
        spelled = _spell(figure.value, figure, spelling, absolute, forms)
        return spelled.number + spelled.suffix
    if not isinstance(figure.value, tuple):
        raise TypeError(f"{figure.path} is an interval figure holding {figure.value!r}, not a pair")
    low, high = figure.value
    if absolute:
        low, high = sorted((abs(low), abs(high)))

    def one_end(end: float, finer: bool) -> tuple[str, str]:
        one = _spell(end, figure, spelling, absolute, forms, finer=finer)
        return one.number, one.suffix

    # How the two ends read together, and are joined, is the page's forms' (:meth:`PageForms.key_interval`).
    return forms.key_interval(
        figure, (low, high), one_end, spelling=spelling, absolute=absolute, joined="interval" in chosen
    )


def render(text: str, arm_names: Mapping[str, str], *, forms: PageForms) -> str:
    """Replace every figure key in report prose with its value, and every ``{arm:<id>}`` with the arm's name and tag.

    :param text: Report prose.
    :ptype text: str
    :param arm_names: Each arm's computed short name, by id (GENERATOR §6.1). An arm declaring a short name
        (``Arm.short_name``) prints that instead; one missing from both is named by its alias, else its id.
    :ptype arm_names: Mapping[str, str]
    :param forms: The forms of the page the prose is printed on (an off-page reader's own), which decide each figure's
        precision (:func:`format_figure`). Each key resolves on the report input they were built from, which they carry,
        so prose never reads one input in another's forms.
    :ptype forms: PageForms
    :return: The prose with every key written out.
    :rtype: str
    :raises UnresolvableReference: A key does not parse, resolve or suit its filters.
    :raises ValueError: The forms are a code page's or an off-page surface's, which carry no report input.
    """
    report_input = forms.report_input
    control = _control(report_input)
    tags = arm_tags([arm.id for arm in report_input.arms], control.id if control else None)
    page: dict[str, str] | None = None
    pieces: list[str] = []
    at = 0
    for key in keys_in(text):
        if key.span is None:
            raise TypeError(f"{key.path} was found in the text with no span")
        start, end = key.span
        figure = resolve(key.path, report_input)
        if figure.kind == "arm":
            arm = _arm(report_input, str(figure.value))
            if page is None:
                page = page_arm_names(report_input)
            name = arm.short_name or arm_names.get(arm.id) or page[arm.id]
            written = written_arm(name, tags[arm.id])
        else:
            written = format_figure(figure, key.filters, forms=forms)
            if "words" in key.filters and _opens_sentence(text[:start]) and text[end:].strip():
                # A count spelled as a word that opens a sentence reads as one does, with a capital: the writer
                # cannot capitalise a word code spells for it ("Ten conversations", never "ten conversations"). A key
                # alone is a figure, not a sentence, and reads as its filter spells it.
                written = written[:1].upper() + written[1:]
        pieces.extend((text[at:start], written))
        at = end
    pieces.append(text[at:])
    return "".join(pieces)


def _opens_sentence(before: str) -> bool:
    """Whether a key after ``before`` opens a sentence: nothing but space before it, or a sentence's end."""
    said = before.rstrip()
    return not said or said.endswith((".", "!", "?")) or before.endswith("\n")


def _allowed_names(report_input: ReportInput) -> set[str]:
    """Every name the input holds that prose may write with its digits: never one made only of digits."""
    names: set[str] = {report_input.meta.campaign}
    for arm in report_input.arms:
        names.update((arm.id, arm.name or "", arm.short_name or "", arm.served_model or ""))
        names.update(value for value in arm.settings.values() if isinstance(value, str))
    for aspect in report_input.aspects:
        names.update((aspect.id, aspect.name or ""))
    for key, measure in report_input.measures.items():
        names.update((key, measure.label or ""))
    for key, judged in report_input.judged.items():
        names.update((key, judged.label or ""))
    for stratum in report_input.strata:
        names.update((stratum.id, stratum.name or ""))
    for task in report_input.tasks:
        names.update((task.id, task.name or ""))
    for factor in report_input.factors:
        names.update((factor.id, factor.name or ""))
        names.update(name for level in factor.levels for name in (level.id, level.name or ""))
    for example in report_input.examples:
        names.update((example.trial or "", example.case or ""))
    return {name for name in names if name.strip() and not name.strip().isdecimal()}


def literal_numbers(text: str, report_input: ReportInput) -> list[tuple[int, int]]:
    """Every number report prose typed itself (GENERATOR §6.2's D3 rule), as ``(start, end)`` offsets into ``text``.

    Keys are set aside first; then every name the input holds that carries digits (arm ids and names, served models,
    settings as displayed, aspect, measure, dimension, stratum, task and factor ids and names, the campaign's name,
    trial and case ids), every percentile name (``p50``, ``p90``, ``p95``, ``p99``) and every one of the package's
    unit phrases (``per 1,000 replies``, matched whole) is set aside where a non-word character, or the text's edge,
    bounds it on both sides. Any Unicode decimal digit left is an offence (``95 percent``, ``1,000 cases``). An id
    made only of digits (an arm ``1017``) is never set aside: prose writes ``{arm:1017}``.

    :param text: Report prose.
    :ptype text: str
    :param report_input: The report input the plan was written against.
    :ptype report_input: ReportInput
    :return: Each run of typed digits, in order; empty when the prose types none.
    :rtype: list[tuple[int, int]]
    """
    masked = list(text)

    def mask(start: int, end: int) -> None:
        # A word character, so a masked name never makes its neighbour look bounded.
        masked[start:end] = "_" * (end - start)

    for match in _KEY.finditer(text):
        mask(*match.span())
    allowed = (
        _allowed_names(report_input) | set(PERCENTILE_NAMES) | set(_UNIT_PHRASES) | _input_unit_phrases(report_input)
    )
    for name in sorted(allowed, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer("".join(masked)):
            mask(*match.span())
    return [match.span() for match in _DIGITS.finditer("".join(masked))]


__all__ = [
    "held_figure",
    "cost_arms",
    "KeySubjects",
    "figure_filters",
    "figure_unit",
    "format_figure",
    "held_kind",
    "is_figure_key",
    "is_rubric_score",
    "key_form",
    "key_subjects",
    "keys_in",
    "literal_numbers",
    "parse_key",
    "plus_signed",
    "render",
    "resolve",
]
