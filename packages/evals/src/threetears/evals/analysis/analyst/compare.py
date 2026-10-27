"""Two arms compared on one figure, with an interval on their difference: the comparison a campaign with no control
leaves the analyst to improvise.

Design note. ``arms_compare`` reads the first arm minus the second on one measure or judged dimension, overall and in
every slice (task, target-matrix cell, stratum) where both arms state a mean, each as the ``compare.*`` figure keys
state it (:mod:`~threetears.evals.analysis.report.figure_keys`; the arithmetic and its own design note are
:mod:`~threetears.evals.analysis.separation`). The W2 prototype's editors ran their own two-sample z-tests on
half-widths; this states the same reading as keys, so a report cites it and the editor checks it with the same tool.

**Paired or independent, from what the snapshot holds.** Overall, where a contrast states one of the two arms against
the other (the control) on the measure, with its difference and interval, the comparison is that paired estimate:
its interval is the pipeline's own, at its family's level, corrected or not as the family's own ``correction`` says
(never asserted: unstated where the family names none). Everywhere else (in a slice,
between two candidates, in a campaign with no control) it combines the two arms' own spreads as independent samples
(:func:`~threetears.evals.analysis.separation.independent_difference`), from each arm's stated interval or half-width
at the stated level, its standard error, or, for a pass/fail rate with a count, its binomial spread. Every row says
which (``basis``). Two arms that ran the same cases are read as if they had not, which reads a wider interval than the
paired one: a separation it shows holds, one it misses may still be there *(inferred: a paired interval needs each
case's difference, which the snapshot does not hold)*.

**An assumed count** (``assumed``, a what-if the caller states): a slice where an arm's figure states no count has no
binomial spread to read, so no interval. Read as if each such slice held that many cases, or (``share``) the arm's own
distinct cases times the slice's traffic share (and, on a repeated design, its trials times the share); every figure read under it is keyed ``assumed.<cases>.compare.``, so it reads
as an assumption wherever it is cited, and a place whose figures state their count is read as they state it.

**What a row says.** The two means, the difference (in points for a rate), its interval and level, whether the
interval excludes zero, and, where it does and the measure states which way is better, the arm ahead. A slice where
either arm states no spread is listed with why, and no interval.

**What the interval counts, and when it may be too narrow.** An independent row says what each arm's count counts
(``counts``: cases, trials, or observations), as the ``spread.`` keys read it; a paired row says its contrast states
no count, so the input does not say whether its interval counts cases or trials. Where what the interval counts is
trials, observations or unstated, the spread was not read on distinct cases, and the arm ran more trials than cases
(its scoreboard's usable or ran trials above its cases: replies of one conversation, or repeats of one case), the
interval may be too narrow, since trials of one case are not independent: the row says so (``may_be_narrow``, a
paired row's unstated unit ``counted_unstated``), cites each such arm's trials and cases (``nested``), and the page's
notes say why, once. It never widens the interval itself *(inferred: the snapshot holds no
per-case outcomes to read a clustered interval from)*.

**What it refuses**, teaching the next call: anything but two distinct arms of the snapshot; a measure the snapshot
holds no figure of for both; and a measure on which neither arm states a spread anywhere both hold it (no interval,
half-width or standard error, and no pass/fail rate with a count), where no interval on a difference can be read: the
remedy names ``measure_read`` for the two means, descriptively, and the measures that do carry a spread.

**No branch on the kind of eval.** Every condition is on what the snapshot holds: a contrast between the two arms, a
stated spread, a count, a slice holding both arms.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import ASSUMED_SHARE, assumable, pairing_contrast, separation_wheres
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["COMPARE_PAGE", "ArmsComparison", "arms_compare"]

#: The places one page holds unless the caller asks for more: overall, then each slice.
COMPARE_PAGE = 12

#: How many valid choices a refusal lists before it stops.
_SHOWN = 24

#: How a refusal says how many arms a call reads.
_COUNT_WORDS = {1: "one", 2: "two"}

#: What every comparison says it is.
_READS_AS = (
    "code's arithmetic on the frozen input: the first arm minus the second, with an interval read off what the "
    "snapshot states (a paired contrast where one holds, else each arm's own spread combined as independent); a "
    "normal approximation, never a test the pipeline ran"
)

#: The resolver's own preamble on a refused key, which a row's reason leaves out: the key, then its words.
_KEY_PREAMBLE = re.compile(r"^figure key \{[^{}]*\} ")

_MODEL = ConfigDict(frozen=True, extra="forbid")


def _arm(arm_id: str) -> str:
    return f"arm:{arm_id}"


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(repr(name) for name in names[:_SHOWN])
    return shown + (f", and {len(names) - _SHOWN} more" if len(names) > _SHOWN else "") if names else "none"


def cite_or_why(key: str, report_input: ReportInput) -> tuple[CitedFigure | None, str]:
    """A figure a key names, or None and the resolver's reason it is unstated (its key left out).

    :param key: The path, without braces or filters.
    :ptype key: str
    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :return: The figure and an empty reason, or None and why the input leaves it unstated.
    :rtype: tuple[CitedFigure | None, str]
    :raises UnresolvableReference: The key names nothing the input holds, or reads two ways.
    """
    try:
        return cite(key, report_input), ""
    except UnstatedFigure as unstated:
        # NOSILENT: a figure the input leaves unstated (no spread to read, an estimate on its line) is the input's
        # shape, not a fault: the row shows it absent and carries the resolver's own words, so the caller reads why.
        return None, _KEY_PREAMBLE.sub("", str(unstated))


def mean_key(report_input: ReportInput, arm: str, key: str, where: str) -> str:
    """The key of an arm's own mean on a measure or judged dimension at a separation key's ``<where>``.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :param arm: The arm, by id.
    :ptype arm: str
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :param where: ``overall``, ``task.<task>``, ``cells.<row>.<column>`` or ``strata.<stratum>``.
    :ptype where: str
    :return: The key.
    :rtype: str
    """
    if where != "overall":
        return f"{where}.{arm}.{key}.mean"
    return f"measure.{key}.{arm}.mean" if key in report_input.measures else f"judged.{key}.{arm}"


def place_name(report_input: ReportInput, where: str) -> CitedFigure | None:
    """What a reader calls a separation key's ``<where>``: a task's, cell's or stratum's name; None overall.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :param where: The place, as the key spells it.
    :ptype where: str
    :return: The name, as its key reads it; None overall.
    :rtype: CitedFigure | None
    """
    return None if where == "overall" else cite(f"{where}.name", report_input)


def better_direction(report_input: ReportInput, key: str) -> Literal["higher_is_better", "lower_is_better", "unstated"]:
    """Which way a measure or judged dimension is better: a judged dimension higher; a measure as it states.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :param key: The measure or judged dimension, by key.
    :ptype key: str
    :return: The direction, or ``unstated``.
    :rtype: Literal["higher_is_better", "lower_is_better", "unstated"]
    """
    higher = True if key in report_input.judged else report_input.measures[key].higher_is_better
    return "unstated" if higher is None else "higher_is_better" if higher else "lower_is_better"


def refuse_arms(report_input: ReportInput, arms: Sequence[str], count: Sequence[int], action: str) -> None:
    """Refuse ``arm_ids`` naming the wrong number of arms, an arm twice, or an arm the snapshot does not hold.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :param arms: The arms named.
    :ptype arms: Sequence[str]
    :param count: The numbers of arms the call may name.
    :ptype count: Sequence[int]
    :param action: The action, as a refusal names it.
    :ptype action: str
    :raises RefusedInput: Any of those.
    """
    held = [arm.id for arm in report_input.arms]
    unknown = [arm for arm in arms if arm not in held]
    if unknown:
        raise RefusedInput(
            f"arm_ids names {_listed(unknown)}, which the snapshot holds as no arm",
            remedy=f"name arms by id: {_listed(held)}",
            parameters=("arm_ids",),
            valid_values=held,
        )
    if len(arms) not in count or len(set(arms)) != len(arms):
        wanted = " or ".join(_COUNT_WORDS[number] for number in count)
        fault = "names one arm twice" if len(set(arms)) != len(arms) else "names another number of arms"
        raise RefusedInput(
            f"{action} reads {wanted} distinct arms, and arm_ids {fault}",
            remedy=f"name {wanted} distinct arms by id: {_listed(held)}",
            parameters=("arm_ids",),
            valid_values=held,
        )


class ComparisonRow(BaseModel):
    """The two arms at one place: their means, the difference, and its interval where their spreads give one."""

    model_config = _MODEL

    where: str = Field(description="overall, or the slice: task.<task>, cells.<row>.<column> or strata.<stratum>.")
    name: CitedFigure | None = Field(description="What a reader calls the slice; None overall.")
    first: CitedFigure = Field(description="The first arm's mean there.")
    second: CitedFigure = Field(description="The second arm's mean there.")
    difference: CitedFigure = Field(description="The first minus the second, in points for a rate.")
    interval: CitedFigure | None = Field(description="The interval on the difference; None where none is read.")
    level: CitedFigure | None = Field(description="The interval's level; None where the input states none.")
    basis: CitedFigure | None = Field(description="Paired (a contrast's own interval) or independent, and from what.")
    paired_with: str | None = Field(
        default=None,
        description="Where the interval is the paired estimate a contrast states, the candidate that contrast reads "
        "against the control, as arm:<id>; None where the two arms' own spreads are read as independent.",
    )
    counts: list[CitedFigure] = Field(
        default_factory=list,
        description="What each arm's count counts on an independent row (cases, trials or observations), as its "
        "spread.<arm>.<measure>.<where>.counts key reads it; empty on a paired row, whose contrast states no count.",
    )
    nested: list[CitedFigure] = Field(
        default_factory=list,
        description="Where the interval may be too narrow, each such arm's trials and its cases: it ran more trials "
        "than cases, and the interval counts trials, observations or an unstated unit; empty otherwise.",
    )
    counted_unstated: bool = Field(
        default=False,
        description="True on a paired row: its contrast states no count, so the input does not say whether its "
        "interval counts cases or trials (notes says so once).",
    )
    may_be_narrow: bool = Field(
        default=False,
        description="True where the interval may be too narrow: the arms in nested ran more trials than cases, and "
        "the interval counts trials, observations or an unstated unit (notes says why once).",
    )
    excludes_zero: bool | None = Field(description="Whether the interval excludes zero; None with no interval.")
    ahead: str | None = Field(
        description="The arm ahead, as arm:<id>, where the interval excludes zero and the measure states which way "
        "is better; None otherwise."
    )
    unread: str = Field(description="Why no interval is read here; empty where one is.")


class ArmsComparison(BaseModel):
    """Two arms compared on one measure, overall and in each slice holding both."""

    model_config = _MODEL

    reads_as: str = Field(description="What this is: code's arithmetic, never a test the pipeline ran.")
    first: str = Field(description="The first arm, as arm:<id>: every difference is it minus the second.")
    second: str = Field(description="The second arm, as arm:<id>.")
    measure: str = Field(description="The measure or judged dimension, by key.")
    better: Literal["higher_is_better", "lower_is_better", "unstated"] = Field(
        description="Which way the measure is better; unstated: no arm is named ahead."
    )
    assumed: str = Field(
        description="The count the caller assumed for each slice stating none, and the key prefix its figures carry; "
        "empty where none is assumed."
    )
    rows: list[ComparisonRow] = Field(description="The places on this page: overall first, then each slice.")
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("place", "places")
    total: int = Field(description="The places both arms hold the measure at.")
    offset: int = Field(description="The first place this page holds.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")
    notes: list[str] = Field(description="What could not be read, and why.")


#: What a paired row's interval counts, in words: the contrast says nothing of it.
_PAIRED_COUNT = (
    "a paired row's contrast states no count (counted_unstated), so the input does not say whether its interval "
    "counts cases or trials"
)

#: Why an interval may be too narrow, in words.
_MAY_BE_NARROW = (
    "a row marked may_be_narrow may have too narrow an interval: it counts trials, observations or an unstated unit, "
    "and the arms its nested figures name ran more trials than cases (replies of one conversation, or repeats of one "
    "case), which are not independent; read on the cases it would be wider. Say so wherever it is cited"
)


def _trials_over_cases(report_input: ReportInput, arm: str) -> list[CitedFigure]:
    """An arm's trials and its cases, as its scoreboard states them, where it ran more trials than cases; else empty."""
    board = report_input.scoreboard.get(arm)
    if board is None or board.cases is None:
        return []
    trials, field = (board.usable, "usable") if board.usable is not None else (board.ran, "ran")
    if trials is None or trials <= board.cases:
        return []
    return [cite(f"scoreboard.{arm}.{field}", report_input), cite(f"scoreboard.{arm}.cases", report_input)]


def _what_it_counts(
    report_input: ReportInput, arms: tuple[str, str], key: str, where: str, head: str, paired: bool
) -> tuple[list[CitedFigure], list[CitedFigure]]:
    """What a row's interval counts (each arm's count, on an independent row), and each arm whose trials may make it
    too narrow, by its trials and cases (this module's What the interval counts)."""
    counts: list[CitedFigure] = []
    loose: list[str] = list(arms) if paired else []
    if not paired:
        for arm in arms:
            counted, _why = cite_or_why(f"{head}spread.{arm}.{key}.{where}.counts", report_input)
            if counted is None:
                continue
            counts.append(counted)
            cases, _why = cite_or_why(f"{head}spread.{arm}.{key}.{where}.cases", report_input)
            if counted.value in ("trials", "observations") and cases is None:
                loose.append(arm)
    return counts, [figure for arm in loose for figure in _trials_over_cases(report_input, arm)]


def _row(
    report_input: ReportInput, first: str, second: str, key: str, where: str, assumed: int | str | None = None
) -> ComparisonRow:
    head = f"assumed.{assumed}." if assumed is not None and assumable(report_input, (first, second), key, where) else ""
    prefix = f"{head}compare.{first}.{second}.{key}.{where}"
    interval, why = cite_or_why(f"{prefix}.interval", report_input)
    level, _level_why = cite_or_why(f"{prefix}.level", report_input) if interval is not None else (None, "")
    paired = pairing_contrast(report_input, first, second, key, where)
    excludes: bool | None = None
    ahead: str | None = None
    counts: list[CitedFigure] = []
    nested: list[CitedFigure] = []
    if interval is not None:
        counts, nested = _what_it_counts(report_input, (first, second), key, where, head, paired is not None)
    if interval is not None and isinstance(interval.value, tuple):
        low, high = interval.value
        excludes = low > 0 or high < 0
        direction = better_direction(report_input, key)
        if excludes and direction != "unstated":
            first_ahead = (low > 0) == (direction == "higher_is_better")
            ahead = _arm(first if first_ahead else second)
    return ComparisonRow(
        where=where,
        name=place_name(report_input, where),
        first=cite(mean_key(report_input, first, key, where), report_input),
        second=cite(mean_key(report_input, second, key, where), report_input),
        difference=cite(f"{prefix}.difference", report_input),
        interval=interval,
        level=level,
        basis=cite(f"{prefix}.basis", report_input) if interval is not None else None,
        paired_with=_arm(paired.arm) if interval is not None and paired is not None else None,
        counts=counts,
        nested=nested,
        counted_unstated=interval is not None and paired is not None,
        may_be_narrow=bool(nested),
        excludes_zero=excludes,
        ahead=ahead,
        unread=why,
    )


def _spread_keys(report_input: ReportInput, first: str, second: str) -> list[str]:
    """The measures and judged dimensions on which the two arms' difference has an interval somewhere."""
    found: list[str] = []
    for key in dict.fromkeys([*report_input.measures, *report_input.judged]):
        for where in separation_wheres(report_input, (first, second), key):
            if cite_or_why(f"compare.{first}.{second}.{key}.{where}.interval", report_input)[0] is not None:
                found.append(key)
                break
    return found


def arms_compare(
    report_input: ReportInput,
    arms: Sequence[str],
    measure: str,
    *,
    assumed: int | str | None = None,
    limit: int = COMPARE_PAGE,
    offset: int = 0,
) -> ArmsComparison:
    """The first arm minus the second on one measure, overall and in each slice holding both, with an interval.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param arms: The two arms, by id: every difference is the first minus the second.
    :ptype arms: Sequence[str]
    :param measure: The measure or judged dimension, by key.
    :ptype measure: str
    :param assumed: A what-if: the cases each slice stating no count of its own is read as holding, or ``share``
        (each arm's own distinct cases times the slice's traffic share); its figures are keyed ``assumed.<cases>.``.
    :ptype assumed: int | str | None
    :param limit: The most places one page holds.
    :ptype limit: int
    :param offset: The first place the page holds.
    :ptype offset: int
    :return: Each place's comparison, overall first.
    :rtype: ArmsComparison
    :raises RefusedInput: Not two distinct arms of the snapshot; a measure both do not hold; or no spread to read an
        interval from anywhere both hold it.
    """
    refuse_arms(report_input, arms, (2,), "arms_compare")
    first, second = arms
    known = list(dict.fromkeys([*report_input.measures, *report_input.judged]))
    wheres = separation_wheres(report_input, (first, second), measure) if measure in known else []
    if not wheres:
        shared = [key for key in known if separation_wheres(report_input, (first, second), key)]
        held = (
            "names no measure or judged dimension the snapshot holds"
            if measure not in known
            else ("is a measure the two arms do not both state a mean on anywhere")
        )
        raise RefusedInput(
            f"measure_key {measure!r} {held}",
            remedy=f"name a measure both arms hold: {_listed(shared)}",
            parameters=("measure_key",),
            valid_values=shared,
        )
    if assumed is not None and not any(assumable(report_input, (first, second), measure, where) for where in wheres):
        raise RefusedInput(
            f"assumed is {assumed!r}, and every place both arms hold {measure!r} at states its count: there is "
            "nothing to assume",
            remedy="omit assumed; it fills a slice whose figure states no count (measure_read with by=task, stratum "
            "or cell shows which state n or n_cases)",
            parameters=("assumed",),
        )
    rows = [_row(report_input, first, second, measure, where, assumed) for where in wheres]
    if all(row.interval is None for row in rows):
        spread = _spread_keys(report_input, first, second)
        raise RefusedInput(
            f"no interval on the difference can be read on {measure!r}: neither arm states a spread there "
            f"({rows[0].unread})",
            remedy=(
                f"read the two means with measure_read (measure_key={measure!r}), descriptively, or compare on a "
                f"measure that states its spread: {_listed(spread)}"
            ),
            parameters=("measure_key",),
            valid_values=spread,
        )
    page = rows[offset : offset + limit]
    end = offset + len(page)
    return ArmsComparison(
        reads_as=_READS_AS,
        first=_arm(first),
        second=_arm(second),
        measure=measure,
        better=better_direction(report_input, measure),
        assumed=assumed_words(assumed),
        rows=page,
        total=len(rows),
        offset=offset,
        next_offset=end if end < len(rows) else None,
        notes=[
            *([_PAIRED_COUNT] if any(row.counted_unstated for row in page) else []),
            *([_MAY_BE_NARROW] if any(row.may_be_narrow for row in page) else []),
            *unkeyed_strata(report_input),
        ],
    )


def assumed_words(assumed: int | str | None) -> str:
    """What an assumed count is, as a result says it; empty for none.

    :param assumed: The cases each slice stating no count is read as holding, ``share``, or None.
    :ptype assumed: int | str | None
    :return: The words.
    :rtype: str
    """
    if assumed is None:
        return ""
    how = (
        "the arm's own distinct cases times the slice's traffic share"
        if assumed == ASSUMED_SHARE
        else f"{assumed} cases"
    )
    return (
        f"a what-if: each slice whose figure states no count read as holding {how}; its figures are keyed "
        f"assumed.{assumed}., and a place whose figures state their count is read as they state it"
    )


def unkeyed_strata(report_input: ReportInput) -> list[str]:
    """Each stratum a separation key cannot name, since the input holds its id more than once, as a note.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :return: One note per such stratum id; empty for none.
    :rtype: list[str]
    """
    ids = [stratum.id for stratum in report_input.strata]
    repeated = [stratum for stratum in dict.fromkeys(ids) if ids.count(stratum) > 1]
    return [f"stratum {stratum} is held more than once, so no key names it alone; left out" for stratum in repeated]
