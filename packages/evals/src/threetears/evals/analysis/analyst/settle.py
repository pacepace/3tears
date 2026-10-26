"""The cases that would settle a reading: an arm against the line it is judged by, or two arms apart.

Design note. ``settle_cases`` answers the question a next step owes its reader (ANALYST §4, "what would settle an open
question?"), which the W2 prototype's analysts could only name in words and its editors computed with their own power
scripts. Every number is a ``settle.*`` figure key (:mod:`~threetears.evals.analysis.report.figure_keys`); the
arithmetic and its design note are :mod:`~threetears.evals.analysis.separation`.

**What it settles.**

- *One arm against a line*: each declared criterion (gate, guardrail, floor or bar) reading the arm, at each place it
  reads it, or the one criterion named (``criterion_id``), or a **hypothetical line** the caller gives on one measure
  (``hypothetical_line`` with ``measure_key``), which the snapshot does not declare. The question is one-sided (does
  the figure clear the line on the side the criterion asks?), at the snapshot's stated level, else 95%. Every figure
  read against a hypothetical line is keyed ``settle.hypothetical.<line>.``, so it reads as hypothetical wherever it
  is cited, and the result says so.
- *Two arms apart* (two ``arm_ids`` and a ``measure_key``): overall and in each slice holding both, two-sided at the
  same level, each arm running the cases named. Read as independent arms: a paired design over the same cases needs
  fewer where the arms agree case by case *(inferred: the snapshot holds no case-level differences to size a paired
  design by)*.

**What a row says.** Today's estimate and the line (or the two arms' difference), its distance toward passing, the
cases a design needs for an interval at the level, centred on today's estimate, to clear the line (``cases_needed``:
if the truth equals today's estimate, half such designs clear, so it is the size at which clearing is a coin toss;
``cases_needed_80``: four in five clear), how often a fresh run of today's size lands on the other side, and the count
and spread basis it rests on. The variance is the binomial one for a pass/fail rate with a count, else the arm's own
stated spread times the root of its count.

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

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite
from threetears.evals.analysis.analyst.compare import (
    cite_or_why,
    mean_key,
    place_name,
    refuse_arms,
    unkeyed_strata,
)
from threetears.evals.analysis.report.figure_keys import criterion_wheres, separation_wheres
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["SETTLE_PAGE", "Settle", "settle_cases"]

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
    "one-sided at the snapshot's level: an interval centred on today's figure clears the line on the side the line "
    "asks; cases_needed is the size at which half such designs clear if the truth equals today's figure, "
    "cases_needed_80 the size at which four in five do"
)
_TWO_SIDED = (
    "two-sided at the snapshot's level, each arm running cases_needed: an interval on the difference, centred on "
    "today's, excludes zero; half such designs do at cases_needed if the truth equals today's difference, four in five "
    "at cases_needed_80. Read as independent arms: a paired design over the same cases needs fewer"
)

#: The fields every row reads, in the key's spelling.
_LINE_FIELDS = ("distance", "cases_needed", "cases_needed_80", "wrong_side", "level")

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
        description="Against a line, the estimate's distance toward passing; between arms, the first minus the second."
    )
    cases_needed: CitedFigure | None = Field(description="The cases at which half such designs settle it.")
    cases_needed_80: CitedFigure | None = Field(description="The cases at which four in five such designs do.")
    wrong_side: CitedFigure | None = Field(
        description="How often a fresh run of today's size reads the other way, if the truth equals today's estimate."
    )
    level: CitedFigure | None = Field(description="The level the design is sized at.")
    counts: list[CitedFigure] = Field(description="The count each arm's spread rests on, today.")
    basis: list[CitedFigure] = Field(description="What each arm's spread was read from.")
    unsettled: str = Field(description="Why no design size is read here; empty where one is.")


class Settle(BaseModel):
    """The cases that would settle each reading named, a page of them."""

    model_config = _MODEL

    reads_as: str = Field(description="What this is: code's arithmetic, which plans a run and tests nothing.")
    sided: str = Field(description="How the question is read: one-sided against a line, two-sided between arms.")
    hypothetical: bool = Field(
        description="True where the line is the caller's own: every figure read against it is hypothetical."
    )
    arms: list[str] = Field(description="The arms, as arm:<id>.")
    rows: list[SettleRow] = Field(description="The readings on this page, each criterion's places in order.")
    total: int = Field(description="The readings in all.")
    offset: int = Field(description="The first reading this page holds.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")
    notes: list[str] = Field(description="What could not be read, and why.")


def _spread_figures(
    report_input: ReportInput, arms: Sequence[str], key: str, where: str
) -> tuple[list[CitedFigure], list[CitedFigure]]:
    """Each arm's count and spread basis at a place, where its spread reads."""
    counts: list[CitedFigure] = []
    bases: list[CitedFigure] = []
    for arm in arms:
        if where not in separation_wheres(report_input, (arm,), key):
            continue
        count, _why = cite_or_why(f"spread.{arm}.{key}.{where}.n", report_input)
        basis, _basis_why = cite_or_why(f"spread.{arm}.{key}.{where}.basis", report_input)
        counts += [count] if count is not None else []
        bases += [basis] if basis is not None else []
    return counts, bases


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
) -> SettleRow:
    """One row: every field the prefix's key family reads, the first unstated one's reason as ``unsettled``."""
    read: dict[str, CitedFigure | None] = {}
    why = ""
    for field in fields:
        figure, reason = cite_or_why(f"{prefix}.{field}", report_input)
        read[field] = figure
        if figure is None and not why and field != "level":
            why = reason
    counts, bases = _spread_figures(report_input, arms, key, where)
    return SettleRow(
        where=where,
        name=place_name(report_input, where),
        against=against,
        estimate=list(estimate),
        line=line,
        distance=read.get("distance") or read.get("difference"),
        cases_needed=read["cases_needed"],
        cases_needed_80=read["cases_needed_80"],
        wrong_side=read["wrong_side"],
        level=read["level"],
        counts=counts,
        basis=bases,
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


def _criteria_rows(report_input: ReportInput, arm: str, criteria: Sequence[str]) -> list[SettleRow]:
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
) -> tuple[list[SettleRow], str, bool]:
    """Every row the call names, how they are read (one- or two-sided), and whether the line is hypothetical."""
    if len(arms) == 2:
        if criterion is not None or line is not None:
            raise RefusedInput(
                "two arms are settled against each other, not against a line, and criterion_id or hypothetical_line "
                "is given",
                remedy="omit criterion_id and hypothetical_line to settle the two arms apart, or name one arm",
                parameters=("criterion_id", "hypothetical_line"),
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
                fields=("difference", *_LINE_FIELDS[1:]),
            )
            for where in wheres
        ]
        return rows, _TWO_SIDED, False
    (arm,) = arms
    if line is not None:
        if criterion is not None:
            raise RefusedInput(
                "criterion_id and hypothetical_line both name a line; settle reads one",
                remedy="name a declared criterion, or give a hypothetical line with its measure_key, not both",
                parameters=("criterion_id", "hypothetical_line"),
            )
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
            )
            for where in wheres
        ]
        return rows, _ONE_SIDED, True
    reading = [held.id for held in report_input.criteria if criterion_wheres(report_input, held.id, arm)]
    if criterion is not None and criterion not in reading:
        declared = criterion in {held.id for held in report_input.criteria}
        fault = "reads no figure for the arm" if declared else "names no criterion the snapshot declares"
        raise RefusedInput(
            f"criterion_id {criterion!r} {fault}",
            remedy=f"name a criterion reading {_arm(arm)}: {_listed(reading)}; or give a hypothetical_line and "
            "measure_key",
            parameters=("criterion_id",),
            valid_values=reading,
        )
    if not reading:
        raise RefusedInput(
            f"no declared criterion reads a figure for {_arm(arm)}, so there is no line to settle it against",
            remedy="give a hypothetical_line with its measure_key (the line in the figure's own units), or name a "
            "second arm to settle the two apart",
            parameters=("hypothetical_line", "measure_key"),
        )
    return _criteria_rows(report_input, arm, [criterion] if criterion is not None else reading), _ONE_SIDED, False


def settle_cases(
    report_input: ReportInput,
    arms: Sequence[str],
    *,
    measure: str | None = None,
    criterion: str | None = None,
    line: float | None = None,
    limit: int = SETTLE_PAGE,
    offset: int = 0,
) -> Settle:
    """The cases that would settle each named reading: one arm against its criteria, one criterion or a hypothetical
    line, or two arms apart on one measure.

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
    :param limit: The most rows one page holds.
    :ptype limit: int
    :param offset: The first row the page holds.
    :ptype offset: int
    :return: A page of rows.
    :rtype: Settle
    :raises RefusedInput: The arms, criterion, line or measure name nothing that can be settled, or no row reads.
    """
    refuse_arms(report_input, arms, (1, 2), "settle_cases")
    rows, sided, hypothetical = _rows(report_input, arms, measure, criterion, line)
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
        arms=[_arm(arm) for arm in arms],
        rows=page,
        total=len(rows),
        offset=offset,
        next_offset=end if end < len(rows) else None,
        notes=unkeyed_strata(report_input) if measure is not None else [],
    )
