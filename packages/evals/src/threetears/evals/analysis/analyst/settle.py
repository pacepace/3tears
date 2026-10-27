"""The cases that would settle a reading: an arm against the line it is judged by, or two arms apart.

Design note. ``settle_cases`` answers the question a next step owes its reader (ANALYST §4, "what would settle an open
question?"), which the W2 prototype's analysts could only name in words and its editors computed with their own power
scripts. Every number is a ``settle.*`` figure key (:mod:`~threetears.evals.analysis.report.figure_keys`); the
arithmetic and its design note are :mod:`~threetears.evals.analysis.separation`.

**What it settles.**

- *One arm against a line*: each declared criterion (gate, guardrail, floor or bar) reading the arm, at each place it
  reads it, or the one criterion named (``criterion_id``), or a **hypothetical line** the caller gives on one measure
  (``hypothetical_line`` with ``measure_key``), which the snapshot does not declare. The question is one-sided (which
  side of the line is the figure on?), at the snapshot's stated level, else 95%. Every figure read against a
  hypothetical line is keyed ``settle.hypothetical.<line>.``, so it reads as hypothetical wherever it is cited, and
  the result says so.
- *A candidate's reading against its aspect's margin, or a floor read on it* (``against``: ``margin:<aspect>`` or
  ``floor:<aspect>/<floor>``, as ``decision_sensitivity`` names them): the change from the control, in points, against
  minus the margin, one-sided at the reading's own level, its spread read off the reading's own interval
  (:func:`~threetears.evals.analysis.separation.difference_spread`), each arm running the size; a floor on another
  rate than the aspect's reads the candidate's rate on that measure minus the control's, as independent arms. With
  one arm and nothing named, every line the arm is judged by: each declared criterion, each margin a reading of it is
  read against, each floor read on it. Every figure is keyed ``settle.margin.`` or ``settle.floor.``.
- *Two arms apart* (two ``arm_ids`` and a ``measure_key``): overall and in each slice holding both, two-sided at the
  same level, each arm running the cases named. Read as independent arms: a paired design over the same cases needs
  fewer where the arms agree case by case *(inferred: the snapshot holds no case-level differences to size a paired
  design by)*.
- *An assumed count* (``assumed``, a what-if the caller states): a slice whose figure states no count has no spread to
  read; read as if each such slice held that many cases, or (``share``) the arm's own count times the slice's traffic
  share. Every figure read under it is keyed ``assumed.<cases>.`` or ``assumed.share.``, so it reads as an assumption
  wherever it is cited; a place whose figures state their count is read as they state it.

**What a row says.** Today's estimate and the line (or the two arms' difference), its distance toward passing, the
cases a design needs for an interval at the level, centred on today's estimate, to sit wholly on today's side of the
line (``cases_needed``: if the truth equals today's estimate, half such designs do, so it is the size at which
settling is a coin toss; ``cases_needed_80``: four in five do), how often a fresh run of today's size lands on the
other side, and the count and spread basis it rests on. The variance is the binomial one for a pass/fail rate with a
count, else the arm's own stated spread times the root of its count.

- *Which way it settles* (``settles``): the sizes are read on the distance's size, so they settle whichever side
  today's estimate is on. A row whose estimate passes says ``clears`` (the size would show the line cleared); one whose
  estimate misses says ``misses`` (the size would confirm the miss, never that the arm clears).
- *Sidedness*: a line row's sizes are for a one-sided bound, and its ``level`` reads ``95% one-sided``; since a page
  prints two-sided intervals, a line row gives beside them the sizes at which the two-sided interval at the level
  sits on today's side (``cases_needed_two_sided``, ``cases_needed_two_sided_80``). A two-arm row is two-sided, and
  its ``level`` says so.
- *What a case is*: a fresh run adds cases, so a size counts cases, each run as often as today's design runs one;
  ``trials_needed`` (and its siblings) is the trials that takes at today's repeats (``repeats``). On a repeated design
  the count a spread rests on is trials (``count_of``); where the figure states its distinct cases the size is read
  on one case's spread over its repeats. A binomial spread read from trials treats each repeat as independent, so
  where a case's repeats agree the cases needed are more, at most the trials figure. Where a figure states no
  distinct cases its size counts what its count counts.

**What it refuses**, teaching: the wrong number of arms; a criterion the snapshot does not declare, or one reading no
figure for the arm; a hypothetical line with no measure, or a measure that states no direction; a line written only
in exponent notation; and a call where no row can be read (every figure a p50 or p95, no spread, no count, or each
estimate exactly on its line), naming why. A row that cannot be read among others that can is listed with why
(``unsettled``), never dropped.

**No branch on the kind of eval.** Every condition is on what the snapshot holds: a declared criterion, a stated
spread, a count, a slice holding the arms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite
from threetears.evals.analysis.analyst.compare import (
    assumed_words,
    cite_or_why,
    mean_key,
    place_name,
    refuse_arms,
    unkeyed_strata,
)
from threetears.evals.analysis.report.figure_keys import (
    assumable,
    criterion_wheres,
    separation_wheres,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["SETTLE_PAGE", "Assumed", "Settle", "settle_cases"]

#: The rows one page holds unless the caller asks for more.
SETTLE_PAGE = 10

#: How many valid choices a refusal lists before it stops.
_SHOWN = 24

#: What every result says it is.
_READS_AS = (
    "code's arithmetic on the frozen input: the cases a design needs, sized on today's estimate and its spread "
    "(a normal approximation), and how often today's design would read the other way if the truth equals today's "
    "estimate; it plans a run, it tests nothing"
)

#: What a row against a line reads, and what one between two arms reads.
_ONE_SIDED = (
    "one-sided at the snapshot's level: a bound centred on today's figure sits wholly on today's side of the line, "
    "which clears the line where today's figure passes (settles: clears) and confirms the miss where it misses "
    "(settles: misses); cases_needed is the size at which half such designs do if the truth equals today's figure, "
    "cases_needed_80 the size at which four in five do. A page prints two-sided intervals, which need more: "
    "cases_needed_two_sided and cases_needed_two_sided_80 are the same sizes for the two-sided interval at the level. "
    "Each size counts cases, each run as often as today (repeats), and trials_needed and its siblings count the trials "
    "that takes; where a figure states no distinct cases its sizes count what count_of says its count counts"
)
_TWO_SIDED = (
    "two-sided at the snapshot's level, each arm running cases_needed: an interval on the difference, centred on "
    "today's, excludes zero; half such designs do at cases_needed if the truth equals today's difference, four in five "
    "at cases_needed_80. Read as independent arms: a paired design over the same cases needs fewer. Each size counts "
    "cases per arm, each run as often as today (repeats), and trials_needed counts the trials that takes"
)

#: What a row against a margin or a floor reads.
_AGAINST_MARGIN = (
    "one-sided at the reading's own level: the candidate's change from the control, in points, against minus the "
    "margin, its spread read off the reading's own interval (a floor on another rate: the candidate's rate on that "
    "measure minus the control's, as independent arms); each arm running cases_needed. Settles says which side today's "
    "change is on"
)

#: The fields every row against a line reads, in the key's spelling.
_LINE_FIELDS = (
    "distance",
    "cases_needed",
    "cases_needed_80",
    "cases_needed_two_sided",
    "cases_needed_two_sided_80",
    "trials_needed",
    "trials_needed_80",
    "trials_needed_two_sided",
    "trials_needed_two_sided_80",
    "wrong_side",
    "level",
)

#: The fields a row between two arms reads.
_PAIR_FIELDS = (
    "difference",
    "cases_needed",
    "cases_needed_80",
    "trials_needed",
    "trials_needed_80",
    "wrong_side",
    "level",
)

#: The fields a row is read without, where its figure states no count of trials over fewer cases.
_UNREPEATED = frozenset(
    {"trials_needed", "trials_needed_80", "trials_needed_two_sided", "trials_needed_two_sided_80", "level"}
)

#: What an assumed count is, as the caller gives it: a number of cases, or ``share`` (the arm's own count times the
#: slice's traffic share).
Assumed = int | Literal["share"]

#: Which side of its line today's estimate is on, so which way a row's sizes settle it.
Settles = Literal["clears", "misses"]

_MODEL = ConfigDict(frozen=True, extra="forbid")


def _arm(arm_id: str) -> str:
    return f"arm:{arm_id}"


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(repr(name) for name in names[:_SHOWN])
    return shown + (f", and {len(names) - _SHOWN} more" if len(names) > _SHOWN else "") if names else "none"


class SettleRow(BaseModel):
    """One reading to settle, at one place: today's estimate against its line or the other arm, and the cases."""

    model_config = _MODEL

    where: str = Field(description="overall, or the slice: task.<task>, cells.<row>.<column> or strata.<stratum>.")
    name: CitedFigure | None = Field(description="What a reader calls the slice; None overall.")
    against: str = Field(
        description="What the reading is settled against: criterion <id>, the hypothetical line, or arm:<id>."
    )
    estimate: list[CitedFigure] = Field(description="Today's mean: the arm's, or each of the two arms'.")
    line: CitedFigure | None = Field(description="The line: the criterion's threshold or the hypothetical line.")
    distance: CitedFigure | None = Field(
        description="Against a line, the estimate's distance toward passing (negative where it misses); between arms, "
        "the first minus the second."
    )
    settles: Settles | None = Field(
        description="Against a line, which side today's estimate is on, so what the sizes would settle: clears (the "
        "line cleared) or misses (the miss confirmed); None between arms or with no distance."
    )
    cases_needed: CitedFigure | None = Field(
        description="The size at which half such designs settle it, sized at the level's sidedness."
    )
    cases_needed_80: CitedFigure | None = Field(description="The size at which four in five such designs do.")
    cases_needed_two_sided: CitedFigure | None = Field(
        description="Against a line, the size at which half such designs' two-sided interval, as a page prints it, "
        "sits on today's side; None between arms, whose sizes are two-sided already."
    )
    cases_needed_two_sided_80: CitedFigure | None = Field(
        description="Against a line, the two-sided size at which four in five do."
    )
    trials_needed: CitedFigure | None = Field(
        description="The trials cases_needed takes, each case run as often as today; None where the figure states no "
        "count of trials over fewer cases."
    )
    trials_needed_80: CitedFigure | None = Field(description="The trials cases_needed_80 takes.")
    trials_needed_two_sided: CitedFigure | None = Field(
        description="Against a line, the trials cases_needed_two_sided takes."
    )
    trials_needed_two_sided_80: CitedFigure | None = Field(
        description="Against a line, the trials cases_needed_two_sided_80 takes."
    )
    wrong_side: CitedFigure | None = Field(
        description="How often a fresh run of today's size reads the other way, if the truth equals today's estimate."
    )
    level: CitedFigure | None = Field(description="The level the design is sized at, with its sidedness in words.")
    counts: list[CitedFigure] = Field(description="The count each arm's spread rests on, today.")
    count_of: list[CitedFigure] = Field(
        description="What each count counts: cases, trials (a repeated case counts again), or observations."
    )
    repeats: list[CitedFigure] = Field(
        description="The trials each case ran today, for each arm whose count is trials over its stated cases."
    )
    basis: list[CitedFigure] = Field(description="What each arm's spread was read from.")
    unsettled: str = Field(description="Why no design size is read here; empty where one is.")


class Settle(BaseModel):
    """The cases that would settle each reading named, a page of them."""

    model_config = _MODEL

    reads_as: str = Field(description="What this is: code's arithmetic, which plans a run and tests nothing.")
    sided: str = Field(
        description="How the question is read: one-sided against a line (with the two-sided sizes beside), two-sided "
        "between arms; which way each line row settles, and what its sizes count."
    )
    hypothetical: bool = Field(
        description="True where the line is the caller's own: every figure read against it is hypothetical."
    )
    assumed: str = Field(
        description="The count the caller assumed for each slice stating none, and the key prefix its figures carry; "
        "empty where none is assumed."
    )
    arms: list[str] = Field(description="The arms, as arm:<id>.")
    rows: list[SettleRow] = Field(description="The readings on this page, each criterion's places in order.")
    total: int = Field(description="The readings in all.")
    offset: int = Field(description="The first reading this page holds.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")
    notes: list[str] = Field(description="What could not be read, and why.")


def _spread_figures(
    report_input: ReportInput, arms: Sequence[str], key: str, where: str, head: str
) -> dict[str, list[CitedFigure]]:
    """Each arm's count, what it counts, its repeats and its spread basis at a place, where its spread reads; ``head``
    the assumption's key prefix, or empty."""
    found: dict[str, list[CitedFigure]] = {"n": [], "counts": [], "repeats": [], "basis": []}
    for arm in arms:
        if where not in separation_wheres(report_input, (arm,), key):
            continue
        for field, figures in found.items():
            figure, _why = cite_or_why(f"{head}spread.{arm}.{key}.{where}.{field}", report_input)
            figures += [figure] if figure is not None else []
    return found


def _row(
    report_input: ReportInput,
    prefix: str,
    *,
    where: str,
    against: str,
    arms: Sequence[str],
    key: str,
    estimate: Sequence[CitedFigure],
    line: CitedFigure | None,
    fields: Sequence[str],
    head: str = "",
) -> SettleRow:
    """One row: every field the prefix's key family reads, the first unstated one's reason as ``unsettled``; ``head``
    the assumption's key prefix where one applies here, else empty. A row with no measure key (a margin's or floor's)
    names no spread figures."""
    read: dict[str, CitedFigure | None] = {}
    why = ""
    for field in fields:
        figure, reason = cite_or_why(f"{head}{prefix}.{field}", report_input)
        read[field] = figure
        if figure is None and not why and field not in _UNREPEATED:
            why = reason
    spread = _spread_figures(report_input, arms, key, where, head) if key else {}
    distance = read.get("distance")
    settles: Settles | None = None
    if distance is not None and isinstance(distance.value, int | float) and distance.value != 0:
        settles = "clears" if distance.value > 0 else "misses"
    return SettleRow(
        where=where,
        name=place_name(report_input, where),
        against=against,
        estimate=list(estimate),
        line=line,
        distance=distance or read.get("difference"),
        settles=settles,
        cases_needed=read["cases_needed"],
        cases_needed_80=read["cases_needed_80"],
        cases_needed_two_sided=read.get("cases_needed_two_sided"),
        cases_needed_two_sided_80=read.get("cases_needed_two_sided_80"),
        trials_needed=read.get("trials_needed"),
        trials_needed_80=read.get("trials_needed_80"),
        trials_needed_two_sided=read.get("trials_needed_two_sided"),
        trials_needed_two_sided_80=read.get("trials_needed_two_sided_80"),
        wrong_side=read["wrong_side"],
        level=read["level"],
        counts=spread.get("n", []),
        count_of=spread.get("counts", []),
        repeats=spread.get("repeats", []),
        basis=spread.get("basis", []),
        unsettled=why if read["cases_needed"] is None or read["wrong_side"] is None else "",
    )


def _means(report_input: ReportInput, arms: Sequence[str], key: str, where: str) -> list[CitedFigure]:
    """Each arm's own mean at a place, by its key."""
    return [cite(mean_key(report_input, arm, key, where), report_input) for arm in arms]


def _line_text(line: float) -> str:
    """A hypothetical line as a key spells it: digits with an optional sign and decimals, no trailing ``.0``.

    :raises RefusedInput: The line is not finite, or only exponent notation writes it.
    """
    text = repr(float(line))
    if not math.isfinite(line) or "e" in text:
        raise RefusedInput(
            "hypothetical_line is written in exponent notation or is not a number, which a figure key cannot spell",
            remedy="give the line in the figure's own units as a plain decimal (a rate on its own scale, as "
            "measure_read prints its mean)",
            parameters=("hypothetical_line",),
        )
    return text.removesuffix(".0")


def _head(report_input: ReportInput, assumed: Assumed | None, arms: Sequence[str], key: str, where: str) -> str:
    """The key prefix an assumed count gives a row's figures where it fills something there; empty elsewhere."""
    return f"assumed.{assumed}." if assumed is not None and assumable(report_input, arms, key, where) else ""


def _criteria_rows(
    report_input: ReportInput, arm: str, criteria: Sequence[str], assumed: Assumed | None = None
) -> list[SettleRow]:
    rows: list[SettleRow] = []
    for criterion_id in criteria:
        criterion = next(held for held in report_input.criteria if held.id == criterion_id)
        line = cite(f"criteria.{criterion_id}.threshold", report_input)
        wheres = criterion_wheres(report_input, criterion_id, arm)
        for where in wheres:
            place = "" if len(wheres) == 1 else "." + where.removeprefix("task.").removeprefix("cells.")
            value = cite(f"criteria.{criterion_id}.{arm}{place}.value", report_input)
            rows.append(
                _row(
                    report_input,
                    f"settle.criteria.{criterion_id}.{arm}.{where}",
                    where=where,
                    against=f"criterion {criterion_id}",
                    arms=(arm,),
                    key=criterion.measure,
                    estimate=[value],
                    line=line,
                    fields=_LINE_FIELDS,
                    head=_head(report_input, assumed, (arm,), criterion.measure, where),
                )
            )
    return rows


def _margin_lines(report_input: ReportInput, arm: str) -> dict[str, str]:
    """Every margin and floor a candidate's reading is read against, as ``decision_sensitivity`` names them
    (``margin:<aspect>``, ``floor:<aspect>/<floor>``), each to the prefix of its ``settle.`` keys."""
    lines: dict[str, str] = {}
    for aspect in report_input.aspects:
        for reading in aspect.readings:
            margin = reading.margin if reading.margin is not None else aspect.margin
            if reading.arm == arm and margin is not None:
                lines[f"margin:{aspect.id}"] = f"settle.margin.{aspect.id}.{arm}"
    for read in report_input.floors:
        if read.arm == arm and read.margin is not None:
            lines[f"floor:{read.aspect}/{read.floor}"] = f"settle.floor.{read.aspect}.{read.floor}.{arm}"
    return lines


def _margin_rows(report_input: ReportInput, arm: str, named: Sequence[str]) -> list[SettleRow]:
    """A row per margin or floor named, read on one candidate's change from the control."""
    lines = _margin_lines(report_input, arm)
    rows: list[SettleRow] = []
    for against in named:
        prefix = lines[against]
        change, _why = cite_or_why(f"{prefix}.change", report_input)
        rows.append(
            _row(
                report_input,
                prefix,
                where="overall",
                against=against,
                arms=(arm,),
                key="",
                estimate=[change] if change is not None else [],
                line=cite(f"{prefix}.line", report_input),
                fields=_LINE_FIELDS,
            )
        )
    return rows


def _refuse_measure(
    report_input: ReportInput, measure: str | None, arms: Sequence[str], need: str
) -> tuple[str, list[str]]:
    """The measure and the places the arms hold it at, refusing one that is absent or held nowhere by all of them."""
    known = list(dict.fromkeys([*report_input.measures, *report_input.judged]))
    shared = [key for key in known if separation_wheres(report_input, arms, key)]
    wheres = separation_wheres(report_input, arms, measure) if measure in known else []
    if measure is None or not wheres:
        fault = "is not given" if measure is None else f"names {measure!r}, which the arms do not all state a mean on"
        raise RefusedInput(
            f"{need}, and measure_key {fault}",
            remedy=f"name a measure the arms hold: {_listed(shared)}",
            parameters=("measure_key",),
            valid_values=shared,
        )
    return measure, wheres


def _rows(
    report_input: ReportInput,
    arms: Sequence[str],
    measure: str | None,
    criterion: str | None,
    line: float | None,
    against: str | None = None,
    assumed: Assumed | None = None,
) -> tuple[list[SettleRow], str, bool]:
    """Every row the call names, how they are read (one- or two-sided), and whether the line is hypothetical."""
    if len(arms) == 2:
        if criterion is not None or line is not None or against is not None:
            raise RefusedInput(
                "two arms are settled against each other, not against a line, and criterion_id, hypothetical_line or "
                "against is given",
                remedy="omit criterion_id, hypothetical_line and against to settle the two arms apart, or name one arm",
                parameters=("criterion_id", "hypothetical_line", "against"),
            )
        first, second = arms
        key, wheres = _refuse_measure(report_input, measure, arms, "two arms are settled apart on one measure")
        rows = [
            _row(
                report_input,
                f"settle.{first}.{second}.{key}.{where}",
                where=where,
                against=_arm(second),
                arms=arms,
                key=key,
                estimate=_means(report_input, arms, key, where),
                line=None,
                fields=_PAIR_FIELDS,
                head=_head(report_input, assumed, arms, key, where),
            )
            for where in wheres
        ]
        return rows, _TWO_SIDED, False
    (arm,) = arms
    given = [
        name
        for name, value in (("criterion_id", criterion), ("hypothetical_line", line), ("against", against))
        if value is not None
    ]
    if len(given) > 1:
        raise RefusedInput(
            f"{' and '.join(given)} each name a line; settle reads one, or every line the arm is judged by with none "
            "named",
            remedy="name a declared criterion, a margin or floor (against), or a hypothetical line with its "
            "measure_key; one, not both",
            parameters=tuple(given),
        )
    if line is not None:
        key, wheres = _refuse_measure(report_input, measure, arms, "a hypothetical line is read on one measure")
        spelled = _line_text(line)
        cited_line = cite(f"settle.hypothetical.{spelled}.{arm}.{key}.{wheres[0]}.line", report_input)
        rows = [
            _row(
                report_input,
                f"settle.hypothetical.{spelled}.{arm}.{key}.{where}",
                where=where,
                against="the hypothetical line",
                arms=arms,
                key=key,
                estimate=_means(report_input, arms, key, where),
                line=cited_line,
                fields=_LINE_FIELDS,
                head=_head(report_input, assumed, arms, key, where),
            )
            for where in wheres
        ]
        return rows, _ONE_SIDED, True
    margins = list(_margin_lines(report_input, arm))
    if against is not None:
        if against not in margins:
            raise RefusedInput(
                f"against {against!r} names no margin or floor a reading of {_arm(arm)} is read against",
                remedy=f"name one of {_listed(margins)}; or a criterion_id, or a hypothetical_line and measure_key",
                parameters=("against",),
                valid_values=margins,
            )
        return _margin_rows(report_input, arm, [against]), _AGAINST_MARGIN, False
    reading = [held.id for held in report_input.criteria if criterion_wheres(report_input, held.id, arm)]
    if criterion is not None and criterion not in reading:
        declared = criterion in {held.id for held in report_input.criteria}
        fault = "reads no figure for the arm" if declared else "names no criterion the snapshot declares"
        raise RefusedInput(
            f"criterion_id {criterion!r} {fault}",
            remedy=f"name a criterion reading {_arm(arm)}: {_listed(reading)}; or a margin or floor (against: "
            f"{_listed(margins)}); or give a hypothetical_line and measure_key",
            parameters=("criterion_id",),
            valid_values=reading,
        )
    if criterion is not None:
        return _criteria_rows(report_input, arm, [criterion], assumed), _ONE_SIDED, False
    if not reading and not margins:
        raise RefusedInput(
            f"no declared criterion reads a figure for {_arm(arm)}, and no margin or floor is read on it, so there is "
            "no line to settle it against",
            remedy="give a hypothetical_line with its measure_key (the line in the figure's own units), or name a "
            "second arm to settle the two apart",
            parameters=("hypothetical_line", "measure_key"),
        )
    rows = _criteria_rows(report_input, arm, reading, assumed) + _margin_rows(report_input, arm, margins)
    sided = " ".join(text for text, held in ((_ONE_SIDED, reading), (_AGAINST_MARGIN, margins)) if held)
    return rows, sided, False


def settle_cases(
    report_input: ReportInput,
    arms: Sequence[str],
    *,
    measure: str | None = None,
    criterion: str | None = None,
    line: float | None = None,
    against: str | None = None,
    assumed: Assumed | None = None,
    limit: int = SETTLE_PAGE,
    offset: int = 0,
) -> Settle:
    """The cases that would settle each named reading: one arm against every line it is judged by (its criteria, the
    margins its readings are read against, the floors read on it), one criterion, margin or floor, or a hypothetical
    line; or two arms apart on one measure.

    :param report_input: The snapshot's report input.
    :ptype report_input: ReportInput
    :param arms: One arm (against a line) or two (apart), by id.
    :ptype arms: Sequence[str]
    :param measure: The measure or judged dimension, by key: for two arms, or a hypothetical line.
    :ptype measure: str | None
    :param criterion: For one arm, the one declared criterion to settle it against; None for every one reading it.
    :ptype criterion: str | None
    :param line: For one arm, a hypothetical line on ``measure``, in the figure's own units.
    :ptype line: float | None
    :param against: For one arm, the one margin or floor to settle its reading against: ``margin:<aspect>`` or
        ``floor:<aspect>/<floor>``.
    :ptype against: str | None
    :param assumed: A what-if: the cases each slice stating no count of its own is read as holding, or ``share``
        (the arm's own count times the slice's traffic share); its figures are keyed ``assumed.<cases>.``.
    :ptype assumed: Assumed | None
    :param limit: The most rows one page holds.
    :ptype limit: int
    :param offset: The first row the page holds.
    :ptype offset: int
    :return: A page of rows.
    :rtype: Settle
    :raises RefusedInput: The arms, criterion, line or measure name nothing that can be settled, or no row reads.
    """
    refuse_arms(report_input, arms, (1, 2), "settle_cases")
    rows, sided, hypothetical = _rows(report_input, arms, measure, criterion, line, against, assumed)
    if assumed is not None and not any(row.basis and row.basis[0].key.startswith("assumed.") for row in rows):
        raise RefusedInput(
            f"assumed is {assumed!r}, and every row named reads a count its figures state, or is read against a "
            "margin or floor: there is nothing to assume",
            remedy="omit assumed; it fills a slice whose figure states no count (measure_read with by=task, stratum "
            "or cell shows which state n or n_cases)",
            parameters=("assumed",),
        )
    if all(row.cases_needed is None and row.wrong_side is None for row in rows):
        raise RefusedInput(
            f"nothing named can be settled: {rows[0].unsettled}",
            remedy="settle a mean that states its spread and count (measure_read lists each arm's statistics), or "
            "name another criterion or measure",
            parameters=("measure_key", "criterion_id") if len(arms) == 1 else ("measure_key",),
        )
    page = rows[offset : offset + limit]
    end = offset + len(page)
    return Settle(
        reads_as=_READS_AS,
        sided=sided,
        hypothetical=hypothetical,
        assumed=assumed_words(assumed),
        arms=[_arm(arm) for arm in arms],
        rows=page,
        total=len(rows),
        offset=offset,
        next_offset=end if end < len(rows) else None,
        notes=unkeyed_strata(report_input) if measure is not None else [],
    )
