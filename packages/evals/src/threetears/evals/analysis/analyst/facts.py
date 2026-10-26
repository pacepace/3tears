"""The analyst's fact tools: what a frozen report input holds, each figure by the key that names it.

Six computations, one per ``Investigate: facts`` tool, each a pure function of the report input and the tool's
parameters:

- :func:`campaign_overview`: what the campaign is, its arms, aspects, measures, criteria, caveats and prior insights,
  and the names a reader is shown for each id: what an analyst reads first, and the keys every other tool takes;
- :func:`readings_list`: each aspect's readings against the control, a choice's pair readings, floors, bars and the
  contrast family correcting it;
- :func:`contrasts_list`: every corrected contrast, its confounds named by their caveat ids, and the families;
- :func:`measure_read`: one measure's or judged dimension's statistics, by arm or within each stratum, task or
  target-matrix cell;
- :func:`examples_list`: the failed trials frozen with the snapshot, with their stored results where the bundle names
  them;
- :func:`caveats_list`: every caveat, where it may land, whether code proved it immaterial, and what it touches.

**Every number is a figure key's.** A tool builds the key that names a figure and :func:`~.cited.cite` (or
:func:`~.cited.cite_if`, where the input may leave it unstated) resolves it; no number here is computed. A figure the
input holds on more than one aspect is named by its aspect (``aspects.<aspect>.``), as the grammar requires: a
contrast whose arm and measure the input holds more than once, a stratum id held more than once, a measure held on
more than one aspect, and the contrast families of an input holding more than one (a family naming no aspect, or
sharing its aspect with another, by its place in the input's families). A floor or bar is keyed by the aspect its read
names, whether or not the input holds a record for that aspect.

**Conditions are on data**, never on what kind of eval the input is: whether a control exists, which slices hold a
measure, which aspect a figure sits on. A filter naming nothing the input holds is refused with
:class:`~threetears.evals.kernel.errors.RefusedInput`, its remedy naming ``campaign_overview``, which lists every id.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.report.figure_keys import reader_names
from threetears.evals.analysis.report_decision import (
    Caveat,
    CaveatReach,
    caveat_materiality,
    report_caveats,
    unsettled_arms,
)
from threetears.evals.analysis.report_input import (
    Aspect,
    BarRead,
    Contrast,
    Example,
    FloorRead,
    Judged,
    LateCriterion,
    Measure,
    PairRead,
    PriorInsight,
    Reading,
    ReportInput,
    Stat,
)
from threetears.evals.analysis.report_input_build import EXAMPLE_EXCERPT_CHARS
from threetears.evals.kernel.errors import RefusedInput

__all__ = [
    "CONTRASTS_LIMIT",
    "EXAMPLES_LIMIT",
    "MEASURE_LIMIT",
    "READINGS_LIMIT",
    "CampaignOverview",
    "CaveatListing",
    "ContrastListing",
    "ExampleListing",
    "MeasureBy",
    "MeasureFigures",
    "MeasureStatistic",
    "ReadingsListing",
    "campaign_overview",
    "caveats_list",
    "contrasts_list",
    "examples_list",
    "measure_read",
    "readings_list",
]

#: What every unknown-id refusal tells the analyst to do: read the ids the snapshot holds.
_OVERVIEW_REMEDY = "campaign_overview lists every arm, aspect, measure key, stratum, task and cell this snapshot holds"

#: Each listing's default page: as many rows as keep its text within what one tool result may take of an analyst's
#: context. A slice's figure keys run long (``strata.<stratum>.<arm>.<measure>.<statistic>``), so a measure's page is
#: shorter than a listing of readings or contrasts.
READINGS_LIMIT = 50
CONTRASTS_LIMIT = 50
MEASURE_LIMIT = 40
EXAMPLES_LIMIT = 10

#: The meta fields the grammar keys (``meta.<field>``), in its order: the campaign's own words.
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

#: How deeply an arm ran an aspect, as the depth keys name each figure.
_DEPTH = ("cases", "trials", "repeats")

#: How an arm's trials came out, as the scoreboard keys name each count and cost.
_SCOREBOARD = ("ran", "usable", "passed", "cases", "lost", "failed", "reply_cost_usd", "test_cost_usd")

#: A reading's figures: the change, its interval, the margin it is read against, the interval's level, the verdict,
#: each arm's passes and why. An interval's ends are its ``low`` and ``high`` keys, which the interval carries.
_READING = ("change", "interval", "margin", "level", "verdict", "control_passed", "arm_passed", "why")

#: A pair reading's figures.
_PAIR = ("change", "interval", "confidence", "p_holm", "ahead", "why")

#: A floor read's figures, its derived held-above and broken-at margins among them.
_FLOOR = (
    "status",
    "reading",
    "speaks_for",
    "interval",
    "worst_drop",
    "margin",
    "short_by",
    "reason",
    "held_above",
    "broken_at",
)

#: A bar read's figures.
_BAR = ("status", "value", "bar", "interval", "why")

#: A contrast's figures.
_CONTRAST = ("delta", "interval", "verdict", "p_adjusted", "level", "materiality", "reason")

#: A contrast family's figures.
_FAMILY = ("disclosure", "correction", "alpha", "family_size", "level")

#: The statistics :func:`measure_read` reads: an arm's figure's, and, within a slice, its mean minus the control's.
MeasureStatistic = Literal["mean", "ci", "half_width", "sem", "n", "n_cases", "p50", "p95", "change", "vs_control"]

#: What :func:`measure_read` reads one measure by: each arm, or each arm within each stratum, task or cell.
MeasureBy = Literal["arm", "stratum", "task", "cell"]

#: Each slice kind's plural, as a refusal names it.
_SLICE_WORDS: dict[MeasureBy, str] = {"arm": "arms", "stratum": "strata", "task": "tasks", "cell": "cells"}

#: The statistic that reads an arm against the control within a slice, which an arm's overall figure has no key for.
_VS_CONTROL = "vs_control"

#: What the examples a snapshot holds are: the frozen pool an analyst pages.
_POOL = (
    "the failed trials frozen with the snapshot: at most one per aspect, arm and stratum, each reply's end and the "
    "judge's or the check's reason"
)


class _Fact(BaseModel):
    """A fact tool's result or row: frozen, and no field it does not declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- shared ---------------------------------------------------------------------------------------------------------


def _figures(keys: Iterable[str], report_input: ReportInput) -> list[CitedFigure]:
    """The figures the keys name that the input states, in the keys' order."""
    return [figure for key in keys if (figure := cite_if(key, report_input)) is not None]


def _page[T](rows: Sequence[T], limit: int, offset: int) -> tuple[list[T], int | None]:
    """One page of rows, and the offset the next page starts at (``None`` on the last)."""
    end = offset + limit
    return list(rows[offset:end]), end if end < len(rows) else None


def _arm_ids(report_input: ReportInput) -> list[str]:
    return [arm.id for arm in report_input.arms]


def _control(report_input: ReportInput) -> str | None:
    return next((arm.id for arm in report_input.arms if arm.role == "control"), None)


def _lead(report_input: ReportInput) -> str | None:
    """The lead aspect, whose figures the input's by-arm maps hold: the first asked about, else the first."""
    asked = [aspect.id for aspect in report_input.aspects if aspect.asked]
    return asked[0] if asked else next((aspect.id for aspect in report_input.aspects), None)


def _arm_figure(report_input: ReportInput, arm_id: str | None) -> CitedFigure | None:
    """``arm:<id>``, for an arm the input holds; ``None`` for none or an arm it does not."""
    return cite(f"arm:{arm_id}", report_input) if arm_id in _arm_ids(report_input) else None


def _known(value: str | None, held: Iterable[str], what: str) -> None:
    """Refuse a filter naming nothing the input holds.

    :raises RefusedInput: ``value`` is given and is none of ``held``.
    """
    known = list(dict.fromkeys(held))
    if value is not None and value not in known:
        raise RefusedInput(
            f"{what} {value!r} names nothing this snapshot holds",
            remedy=_OVERVIEW_REMEDY,
            parameters=(what,),
            valid_values=known,
        )


# --- campaign_overview ----------------------------------------------------------------------------------------------


class ArmRow(_Fact):
    """One arm: who it is, how it is set up, and how its trials came out."""

    arm: CitedFigure = Field(description="The arm, as `arm:<id>`: its short name and letter tag.")
    role: str = Field(description="control, or challenger: a candidate measured against the control.")
    short_name: str | None = Field(description="The arm's short name, where the input gives one.")
    served_model: str | None = Field(description="The model that served its replies.")
    settings: list[CitedFigure] = Field(
        description="Its settings (`settings.<arm>.<setting>`): a number where the input holds one, else its text."
    )
    scoreboard: list[CitedFigure] = Field(description="How its trials came out (`scoreboard.<arm>.<count>`).")


class AspectRow(_Fact):
    """One aspect: whether it is asked, how it is graded, its margin, and which rows it holds."""

    aspect: str = Field(description="The aspect's id, as every `aspects.<aspect>.` key names it.")
    asked: bool | None = Field(description="Whether the campaign's question asks about it.")
    graded_by: str | None = Field(description="judged or checked.")
    margin: CitedFigure | None = Field(description="Its non-inferiority margin, in points, where one is declared.")
    cases: CitedFigure | None = Field(
        description="The cases it ran: the control's, else the count every arm agrees on; none where they differ."
    )
    repeats: CitedFigure | None = Field(
        description="The mean repeats each case ran: the control's, else the figure every arm agrees on."
    )
    depth: list[CitedFigure] = Field(
        description="Each arm's cases, trials and mean repeats on it (`aspects.<aspect>.depth.<arm>.<field>`), where "
        "its arms ran it to different depths."
    )
    holds: list[str] = Field(
        description="The rows the input holds on it: readings, pairs, floors, bars, contrasts, strata, scoreboard."
    )


class MeasureRow(_Fact):
    """One measure or judged dimension: what it is, where it is read, and the key form that reads it."""

    key: str = Field(description="Its key, which measure_read takes as measure_key.")
    source: Literal["measure", "judged"] = Field(description="A measure, or a judged dimension.")
    unit: str | None = Field(description="Its unit, for a measure.")
    higher_is_better: bool | None = Field(description="Which way is better, where the input says.")
    merit_axis: str | None = Field(description="The merit axis it counts toward.")
    grade: str | None = Field(description="A measure's grade, or a judged dimension's scale.")
    level: str | None = Field(description="The confidence level of its intervals.")
    arms_without: list[str] = Field(
        description="The arms holding no figure on it, by id; empty where every arm holds one."
    )
    aspects: list[str] = Field(description="The aspects it is read on, where it is read on more than the lead one.")
    slices: list[str] = Field(description="Where measure_read can read it: arm, stratum, task, cell.")
    key_form: str = Field(description="The figure key that reads one arm's statistic on it.")


class CriterionRow(_Fact):
    """One declared criterion: a threshold on one measure."""

    id: str = Field(description="The criterion's id, as `criteria.<criterion>.` keys name it.")
    kind: str = Field(description="gate, floor, bar or guardrail.")
    measure: str = Field(description="The measure, judged dimension or aspect it reads, by key.")
    statistic: str = Field(description="Which statistic it reads.")
    direction: str = Field(description="Which way clearing it runs.")
    scope: str = Field(description="overall, each_task or each_cell.")
    task: str | None = Field(description="The one task it applies within, where it names one.")
    threshold: CitedFigure = Field(description="Its threshold, in its measure's units.")


class LateRow(_Fact):
    """One criterion set after the results were seen, and when."""

    criterion: str = Field(description="What was set: aspect, control, margin, floor, gate.")
    name: str | None = Field(description="Which one, where several could be.")
    set_at: CitedFigure = Field(description="When it was set (`late_criteria.<late>.set_at`).")


class CaveatRef(_Fact):
    """One caveat by its id, and where it may land; caveats_list says what it touches."""

    id: str = Field(description="The caveat's id.")
    kind: str = Field(
        description="late, judge_uncalibrated, confound, apparatus, short_run, unsettled, latency or unreported."
    )
    reach: CaveatReach = Field(description="report (the confidence rests on it), scoped (beside a claim) or method.")


class CampaignOverview(_Fact):
    """What the snapshot holds: the campaign, its arms, aspects, measures, criteria, caveats and prior insights."""

    snapshot: str = Field(description="The snapshot every figure here is read from.")
    meta: list[CitedFigure] = Field(description="The campaign's own words (`meta.<field>`), each it states.")
    n_candidates: CitedFigure = Field(description="How many candidates ran.")
    cases: CitedFigure | None = Field(description="The cases the asked aspect ran, where its arms agree on one count.")
    repeats: CitedFigure | None = Field(description="The repeats each case ran, where the arms agree.")
    arms: list[ArmRow] = Field(description="Every arm, in the input's order.")
    aspects: list[AspectRow] = Field(description="Every aspect, in the input's order.")
    measures: list[MeasureRow] = Field(description="Every measure, then every judged dimension.")
    criteria: list[CriterionRow] = Field(description="Every declared criterion.")
    cells: list[CitedFigure] = Field(description="Each target-matrix cell's name, as `cells.<row>.<column>.name`.")
    caveats: list[CaveatRef] = Field(description="Every caveat, by id, with its reach.")
    late_criteria: list[LateRow] = Field(description="The criteria set after the results were seen, each when.")
    late_set_at: CitedFigure | None = Field(description="When they were set, where they share one date.")
    prior_insights: list[PriorInsight] = Field(description="What earlier campaigns concluded.")
    names: dict[str, dict[str, str]] = Field(
        description="The one name a reader is shown for each id, by kind (aspect, measure, judged, stratum, task, "
        "criterion), then id."
    )


def _late_name(late: LateCriterion) -> str:
    """How a key names a late criterion, as its caveat's id names it after ``late:``: what was set, then which one."""
    return f"{late.criterion}:{late.name}" if late.name else late.criterion


def _depths(report_input: ReportInput, aspect: Aspect) -> list[CitedFigure]:
    """Each arm's depth figures on an aspect, where its arms ran it to different depths; none where they agree."""
    depth = aspect.depth or {}
    if len({(d.cases, d.trials, tuple(d.repeats_per_case)) for d in depth.values()}) <= 1:
        return []
    keys = (f"aspects.{aspect.id}.depth.{arm}.{field}" for arm in depth for field in _DEPTH)
    return _figures(keys, report_input)


def _aspect_holds(report_input: ReportInput, aspect: Aspect) -> list[str]:
    held = {
        "readings": bool(aspect.readings),
        "pairs": bool(aspect.pairs),
        "floors": any(read.aspect == aspect.id for read in report_input.floors),
        "bars": any(read.aspect == aspect.id for read in report_input.bars),
        "contrasts": any(contrast.aspect == aspect.id for contrast in report_input.contrasts),
        "strata": any(stratum.aspect == aspect.id for stratum in report_input.strata),
        "scoreboard": bool(aspect.scoreboard),
    }
    return [name for name, holds in held.items() if holds]


def _tables(report_input: ReportInput) -> list[tuple[str, Literal["measure", "judged"], Measure | Judged]]:
    """Every measure, then every judged dimension, by key."""
    measures: list[tuple[str, Literal["measure", "judged"], Measure | Judged]] = [
        (key, "measure", measure) for key, measure in report_input.measures.items()
    ]
    return measures + [(key, "judged", judged) for key, judged in report_input.judged.items()]


def _on_aspects(report_input: ReportInput, table: Measure | Judged) -> list[str]:
    """The aspects a measure is read on, where it is read on more than the lead one: each the input holds."""
    if not table.by_aspect:
        return []
    lead = _lead(report_input)
    return [
        aspect.id
        for aspect in report_input.aspects
        if (aspect.id == lead and table.per_arm) or table.by_aspect.get(aspect.id)
    ]


def _slices_of(report_input: ReportInput, key: str, table: Measure | Judged) -> list[str]:
    held = {
        "arm": bool(table.per_arm) or any(table.by_aspect.values()),
        "stratum": any(key in figures for s in report_input.strata for figures in s.per_arm.values()),
        "task": any(key in figures for t in report_input.tasks for figures in t.per_arm.values()),
        "cell": any(key in figures for c in report_input.cells.values() for figures in c.per_arm.values()),
    }
    return [by for by, holds in held.items() if holds]


def _measure_row(
    report_input: ReportInput, key: str, source: Literal["measure", "judged"], table: Measure | Judged
) -> MeasureRow:
    aspects = _on_aspects(report_input, table)
    form = f"aspects.<aspect>.{source}.{key}.<arm>.<statistic>" if aspects else f"{source}.{key}.<arm>.<statistic>"
    held = set(table.per_arm) | {arm for figures in table.by_aspect.values() for arm in figures}
    if isinstance(table, Measure):
        unit, higher, axis, grade = table.unit, table.higher_is_better, table.merit_axis, table.grade
    else:
        unit, higher, axis, grade = None, None, None, table.scale
    return MeasureRow(
        key=key,
        source=source,
        unit=unit,
        higher_is_better=higher,
        merit_axis=axis,
        grade=grade,
        level=table.level,
        arms_without=[arm for arm in _arm_ids(report_input) if arm not in held],
        aspects=aspects,
        slices=_slices_of(report_input, key, table),
        key_form=form,
    )


def campaign_overview(report_input: ReportInput, snapshot_id: str) -> CampaignOverview:
    """What a snapshot holds, read first: every id the other tools take, and the campaign's own figures.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param snapshot_id: The snapshot's id.
    :ptype snapshot_id: str
    :return: The overview.
    :rtype: CampaignOverview
    """
    reaches = report_caveats(report_input)
    late = report_input.caveats.late_criteria
    arms = [
        ArmRow(
            arm=cite(f"arm:{arm.id}", report_input),
            role=arm.role,
            short_name=arm.short_name,
            served_model=arm.served_model,
            settings=_figures((f"settings.{arm.id}.{name}" for name in arm.settings), report_input),
            scoreboard=(
                _figures((f"scoreboard.{arm.id}.{count}" for count in _SCOREBOARD), report_input)
                if arm.id in report_input.scoreboard
                else []
            ),
        )
        for arm in report_input.arms
    ]
    aspects = [
        AspectRow(
            aspect=aspect.id,
            asked=aspect.asked,
            graded_by=aspect.graded_by,
            margin=cite_if(f"aspects.{aspect.id}.margin", report_input),
            cases=cite_if(f"aspects.{aspect.id}.cases", report_input),
            repeats=cite_if(f"aspects.{aspect.id}.repeats", report_input),
            depth=_depths(report_input, aspect),
            holds=_aspect_holds(report_input, aspect),
        )
        for aspect in report_input.aspects
    ]
    criteria = [
        CriterionRow(
            id=criterion.id,
            kind=criterion.kind,
            measure=criterion.measure,
            statistic=criterion.statistic,
            direction=criterion.direction,
            scope=criterion.scope,
            task=criterion.task,
            threshold=cite(f"criteria.{criterion.id}.threshold", report_input),
        )
        for criterion in report_input.criteria
    ]
    return CampaignOverview(
        snapshot=snapshot_id,
        meta=_figures((f"meta.{field}" for field in _META_FIELDS), report_input),
        n_candidates=cite("n_candidates", report_input),
        cases=cite_if("cases", report_input),
        repeats=cite_if("repeats", report_input),
        arms=arms,
        aspects=aspects,
        measures=[_measure_row(report_input, *table) for table in _tables(report_input)],
        criteria=criteria,
        cells=[cite(f"cells.{cell.row}.{cell.column}.name", report_input) for cell in report_input.cells.values()],
        caveats=[CaveatRef(id=caveat.id, kind=caveat.kind, reach=caveat.reach) for caveat in reaches],
        late_criteria=[
            LateRow(
                criterion=criterion.criterion,
                name=criterion.name,
                set_at=cite(f"late_criteria.{_late_name(criterion)}.set_at", report_input),
            )
            for criterion in late
        ],
        late_set_at=cite_if("late_criteria.set_at", report_input) if late else None,
        prior_insights=list(report_input.prior_insights),
        names={kind: dict(names) for kind, names in reader_names(report_input).items()},
    )


# --- readings_list --------------------------------------------------------------------------------------------------


class ReadingRow(_Fact):
    """One arm's reading against the control on one aspect."""

    arm: CitedFigure = Field(description="The arm read.")
    figures: list[CitedFigure] = Field(description="Its change, interval, margin, level, verdict, passes and why.")
    flip_margin: CitedFigure | None = Field(
        description="The margin above which it holds (derived), where its interval is stated."
    )


class PairRow(_Fact):
    """A choice's pair reading on one aspect: the first arm minus the second."""

    arms: list[CitedFigure] = Field(description="The first arm, then the second.")
    figures: list[CitedFigure] = Field(description="Its change, interval, confidence, Holm p, the arm ahead, why.")


class FloorRow(_Fact):
    """A floor read on one candidate."""

    floor: str = Field(description="The floor's key.")
    arm: str = Field(description="The candidate, by id.")
    status: str = Field(description="broken, held or unclear.")
    figures: list[CitedFigure] = Field(description="Its figures, keyed by the aspect the read names.")


class BarRow(_Fact):
    """An arm against an absolute bar."""

    measure: str = Field(description="What the bar is read on, by key.")
    arm: str = Field(description="The arm, by id.")
    direction: str = Field(description="Which way clearing it runs.")
    status: str = Field(description="met, missed or unclear.")
    figures: list[CitedFigure] = Field(description="Its figures, keyed by the aspect the read names.")


class AspectReadings(_Fact):
    """One aspect's rows on this page."""

    aspect: str = Field(description="The aspect's id.")
    margin: CitedFigure | None = Field(description="Its declared margin, where one is.")
    family: list[CitedFigure] = Field(description="The contrast family correcting its contrasts, where one names it.")
    readings: list[ReadingRow] = Field(description="Each arm's reading against the control.")
    pairs: list[PairRow] = Field(description="A choice's pair readings.")
    floors: list[FloorRow] = Field(description="Each floor read on each candidate.")
    bars: list[BarRow] = Field(description="Each arm against each absolute bar.")
    note: str | None = Field(
        description="Where the input holds no record for the aspect its floors and bars name: why it has no margin or "
        "family."
    )


class ReadingsListing(_Fact):
    """A page of every aspect's readings, pair readings, floors and bars."""

    aspects: list[AspectReadings] = Field(description="The page's rows, by aspect, in the input's order.")
    total: int = Field(description="Rows in every page: readings, pairs, floors and bars.")
    offset: int = Field(description="The rows skipped before this page.")
    next_offset: int | None = Field(description="The next page's offset; none on the last page.")


#: A row of :func:`readings_list` before it is grouped: its aspect, and the row.
_Placed = tuple[str, ReadingRow | PairRow | FloorRow | BarRow]


def _reading_row(report_input: ReportInput, aspect: str, reading: Reading) -> ReadingRow:
    prefix = f"aspects.{aspect}.reading.{reading.arm}"
    return ReadingRow(
        arm=cite(f"arm:{reading.arm}", report_input),
        figures=_figures((f"{prefix}.{statistic}" for statistic in _READING), report_input),
        flip_margin=cite_if(f"aspects.{aspect}.flip_margin.{reading.arm}", report_input),
    )


def _pair_row(report_input: ReportInput, aspect: str, pair: PairRead) -> PairRow:
    first, second = pair.arms
    prefix = f"aspects.{aspect}.pair.{first}.{second}"
    return PairRow(
        arms=[cite(f"arm:{first}", report_input), cite(f"arm:{second}", report_input)],
        figures=_figures((f"{prefix}.{statistic}" for statistic in _PAIR), report_input),
    )


def _floor_row(report_input: ReportInput, read: FloorRead) -> FloorRow:
    prefix = f"aspects.{read.aspect}.floor.{read.floor}.{read.arm}"
    figures = _figures((f"{prefix}.{field}" for field in _FLOOR), report_input)
    return FloorRow(floor=read.floor, arm=read.arm, status=read.status, figures=figures)


def _bar_row(report_input: ReportInput, read: BarRead) -> BarRow:
    prefix = f"aspects.{read.aspect}.bar.{read.measure}.{read.arm}"
    figures = _figures((f"{prefix}.{field}" for field in _BAR), report_input)
    return BarRow(measure=read.measure, arm=read.arm, direction=read.direction, status=read.status, figures=figures)


def _family_keys(report_input: ReportInput, index: int) -> list[str]:
    """The keys reading the contrast family at ``index``: ``family.*`` for an input's one family, else
    ``aspects.<a>.family.*`` where it alone sits on its aspect, else ``families.<index>.*``."""
    families = report_input.contrast_families
    if len(families) == 1:
        return [f"family.{field}" for field in _FAMILY]
    aspect = families[index].aspect
    if aspect is not None and sum(family.aspect == aspect for family in families) == 1:
        return [f"aspects.{aspect}.family.{field}" for field in _FAMILY]
    return [f"families.{index}.{field}" for field in _FAMILY]


def _aspect_family(report_input: ReportInput, aspect: str) -> list[CitedFigure]:
    """The figures of each contrast family correcting one aspect's contrasts."""
    held = [index for index, family in enumerate(report_input.contrast_families) if family.aspect == aspect]
    return [figure for index in held for figure in _figures(_family_keys(report_input, index), report_input)]


def _reading_rows(report_input: ReportInput, aspect_filter: str | None, arm_filter: str | None) -> list[_Placed]:
    """Every row :func:`readings_list` pages, by aspect: each held aspect's readings, pairs, floors and bars, then the
    floors and bars of aspects the input holds no record for."""

    def kept(*arms: str) -> bool:
        return arm_filter is None or arm_filter in arms

    held = [aspect.id for aspect in report_input.aspects]
    rows: list[_Placed] = []
    for aspect in report_input.aspects:
        if aspect_filter is not None and aspect.id != aspect_filter:
            continue
        rows += [(aspect.id, _reading_row(report_input, aspect.id, r)) for r in aspect.readings if kept(r.arm)]
        rows += [(aspect.id, _pair_row(report_input, aspect.id, pair)) for pair in aspect.pairs if kept(*pair.arms)]
        floors = [read for read in report_input.floors if read.aspect == aspect.id and kept(read.arm)]
        rows += [(aspect.id, _floor_row(report_input, read)) for read in floors]
        bars = [read for read in report_input.bars if read.aspect == aspect.id and kept(read.arm)]
        rows += [(aspect.id, _bar_row(report_input, read)) for read in bars]
    for floor in report_input.floors:
        if floor.aspect not in held and aspect_filter in (None, floor.aspect) and kept(floor.arm):
            rows.append((floor.aspect, _floor_row(report_input, floor)))
    for bar in report_input.bars:
        if bar.aspect not in held and aspect_filter in (None, bar.aspect) and kept(bar.arm):
            rows.append((bar.aspect, _bar_row(report_input, bar)))
    return rows


def _grouped(report_input: ReportInput, page: Sequence[_Placed]) -> list[AspectReadings]:
    held = {aspect.id for aspect in report_input.aspects}
    order = list(dict.fromkeys(aspect for aspect, _row in page))
    groups: list[AspectReadings] = []
    for aspect in order:
        rows = [row for at, row in page if at == aspect]
        keyed = aspect in held
        groups.append(
            AspectReadings(
                aspect=aspect,
                margin=cite_if(f"aspects.{aspect}.margin", report_input) if keyed else None,
                family=_aspect_family(report_input, aspect) if keyed else [],
                readings=[row for row in rows if isinstance(row, ReadingRow)],
                pairs=[row for row in rows if isinstance(row, PairRow)],
                floors=[row for row in rows if isinstance(row, FloorRow)],
                bars=[row for row in rows if isinstance(row, BarRow)],
                note=None
                if keyed
                else f"the input holds no record for aspect {aspect!r}: its reads are keyed by the aspect they name, "
                "and it declares no margin or family",
            )
        )
    return groups


def readings_list(
    report_input: ReportInput,
    *,
    aspect_filter: str | None = None,
    arm_filter: str | None = None,
    limit: int = READINGS_LIMIT,
    offset: int = 0,
) -> ReadingsListing:
    """A page of every aspect's readings against the control, pair readings, floors and bars, grouped by aspect.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param aspect_filter: Only this aspect's rows.
    :ptype aspect_filter: str | None
    :param arm_filter: Only the rows of this arm (a pair reading of either of its arms).
    :ptype arm_filter: str | None
    :param limit: The most rows on the page.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :return: The page.
    :rtype: ReadingsListing
    :raises RefusedInput: A filter names an aspect or arm the snapshot does not hold.
    """
    aspects = [aspect.id for aspect in report_input.aspects]
    _known(
        aspect_filter,
        [*aspects, *(f.aspect for f in report_input.floors), *(b.aspect for b in report_input.bars)],
        "aspect_filter",
    )
    _known(arm_filter, _arm_ids(report_input), "arm_filter")
    rows = _reading_rows(report_input, aspect_filter, arm_filter)
    page, next_offset = _page(rows, limit, offset)
    return ReadingsListing(
        aspects=_grouped(report_input, page), total=len(rows), offset=offset, next_offset=next_offset
    )


# --- contrasts_list -------------------------------------------------------------------------------------------------


class ContrastRow(_Fact):
    """One corrected contrast of an arm against the control."""

    arm: CitedFigure = Field(description="The arm contrasted.")
    measure: str = Field(description="The measure or judged dimension, by key.")
    aspect: str | None = Field(description="The aspect it is read on, where it names one.")
    confounds: list[str] = Field(description="Its confounds, by caveat id (caveats_list says what each touches).")
    figures: list[CitedFigure] = Field(description="Its delta, interval, verdict, adjusted p, level, materiality, why.")


class FamilyRow(_Fact):
    """One contrast family: the correction its contrasts were read under."""

    aspect: str | None = Field(description="The aspect whose contrasts it corrects, where it names one.")
    merit_axes: list[str] = Field(description="The merit axes it ranks, in priority order.")
    figures: list[CitedFigure] = Field(description="Its disclosure, correction, alpha, size and level.")


class ContrastListing(_Fact):
    """A page of the contrasts, and every contrast family."""

    contrasts: list[ContrastRow] = Field(description="The page's contrasts, in the input's order.")
    families: list[FamilyRow] = Field(description="The contrast families, on the filtered aspect where one is.")
    total: int = Field(description="Contrasts in every page.")
    offset: int = Field(description="The contrasts skipped before this page.")
    next_offset: int | None = Field(description="The next page's offset; none on the last page.")


def _contrast_prefix(report_input: ReportInput, contrast: Contrast) -> str:
    """The key a contrast's figures are read under: unqualified, or by its aspect where the input holds its arm and
    measure more than once."""
    held = [c for c in report_input.contrasts if (c.arm, c.measure) == (contrast.arm, contrast.measure)]
    if len(held) > 1 and contrast.aspect is not None:
        return f"aspects.{contrast.aspect}.contrast.{contrast.arm}.{contrast.measure}"
    return f"contrast.{contrast.arm}.{contrast.measure}"


def _caveat_id(caveats: Iterable[Caveat], dimension: str) -> str:
    """A confound's caveat id: its apparatus or confound caveat where the input lists one, else the dimension."""
    ids = {caveat.id for caveat in caveats}
    return next((held for held in (f"apparatus:{dimension}", f"confound:{dimension}") if held in ids), dimension)


def contrasts_list(
    report_input: ReportInput,
    *,
    arm_filter: str | None = None,
    measure_filter: str | None = None,
    aspect_filter: str | None = None,
    limit: int = CONTRASTS_LIMIT,
    offset: int = 0,
) -> ContrastListing:
    """A page of the corrected contrasts, each by its figure keys, and the contrast families correcting them.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param arm_filter: Only this arm's contrasts.
    :ptype arm_filter: str | None
    :param measure_filter: Only the contrasts on this measure or judged dimension.
    :ptype measure_filter: str | None
    :param aspect_filter: Only the contrasts read on this aspect, and its family.
    :ptype aspect_filter: str | None
    :param limit: The most contrasts on the page.
    :ptype limit: int
    :param offset: The contrasts to skip.
    :ptype offset: int
    :return: The page.
    :rtype: ContrastListing
    :raises RefusedInput: A filter names an arm, measure or aspect the snapshot does not hold.
    """
    contrasts = report_input.contrasts
    _known(arm_filter, _arm_ids(report_input), "arm_filter")
    measures = [*report_input.measures, *report_input.judged, *(c.measure for c in contrasts)]
    _known(measure_filter, measures, "measure_filter")
    aspects = [*(a.id for a in report_input.aspects), *(c.aspect for c in contrasts if c.aspect is not None)]
    _known(aspect_filter, aspects, "aspect_filter")
    kept = [
        contrast
        for contrast in contrasts
        if arm_filter in (None, contrast.arm)
        and measure_filter in (None, contrast.measure)
        and aspect_filter in (None, contrast.aspect)
    ]
    page, next_offset = _page(kept, limit, offset)
    caveats = report_caveats(report_input)
    rows = [
        ContrastRow(
            arm=cite(f"arm:{contrast.arm}", report_input),
            measure=contrast.measure,
            aspect=contrast.aspect,
            confounds=[_caveat_id(caveats, dimension) for dimension in contrast.confounds],
            figures=_figures(
                (f"{_contrast_prefix(report_input, contrast)}.{statistic}" for statistic in _CONTRAST), report_input
            ),
        )
        for contrast in page
    ]
    families = [
        FamilyRow(
            aspect=family.aspect,
            merit_axes=list(family.merit_axes),
            figures=_figures(_family_keys(report_input, index), report_input),
        )
        for index, family in enumerate(report_input.contrast_families)
        if aspect_filter in (None, family.aspect)
    ]
    return ContrastListing(contrasts=rows, families=families, total=len(kept), offset=offset, next_offset=next_offset)


# --- measure_read ---------------------------------------------------------------------------------------------------


class FigureRow(_Fact):
    """One arm's statistics on the measure: overall, on one aspect, or within one slice."""

    arm: CitedFigure = Field(description="The arm.")
    aspect: str | None = Field(description="The aspect the figure is read on, where the key names one.")
    slice: str | None = Field(description="The stratum or task id, or the cell as `<row>.<column>`; none by arm.")
    figures: list[CitedFigure] = Field(description="The statistics asked for, each the input states.")


class SliceRow(_Fact):
    """One slice a page's rows are read within: its name and its cases."""

    slice: str = Field(description="The stratum or task id, or the cell as `<row>.<column>`.")
    aspect: str | None = Field(description="The aspect whose cases it slices, where its keys name one.")
    name: CitedFigure | None = Field(description="Its name.")
    n_cases: CitedFigure | None = Field(description="Its cases, where the input states them.")


class MeasureFigures(_Fact):
    """A page of one measure's figures, by arm or by slice."""

    measure: str = Field(description="The measure or judged dimension read, by key.")
    by: MeasureBy = Field(description="What it is read by.")
    slices: list[SliceRow] = Field(description="The slices this page's rows are read within; none by arm.")
    rows: list[FigureRow] = Field(description="The page's rows: by slice, then arm, in the input's order.")
    total: int = Field(description="Rows in every page.")
    offset: int = Field(description="The rows skipped before this page.")
    next_offset: int | None = Field(description="The next page's offset; none on the last page.")


#: One row :func:`measure_read` pages, before its figures are cited: the arm, the aspect, the slice, the key prefix
#: its statistics are read under, and the prefix the slice's own fields are.
_Spot = tuple[str, str | None, str | None, str, str | None]


def _table(report_input: ReportInput, measure_key: str) -> tuple[Literal["measure", "judged"], Measure | Judged]:
    """The measure or judged dimension a key names.

    :raises RefusedInput: The input holds neither of that key.
    """
    if measure_key in report_input.measures:
        return "measure", report_input.measures[measure_key]
    if measure_key in report_input.judged:
        return "judged", report_input.judged[measure_key]
    raise RefusedInput(
        f"measure_key {measure_key!r} names no measure or judged dimension this snapshot holds",
        remedy="campaign_overview lists every measure key, under measures",
        parameters=("measure_key",),
        valid_values=[*report_input.measures, *report_input.judged],
    )


def _on_arm(
    report_input: ReportInput,
    measure_key: str,
    source: Literal["measure", "judged"],
    table: Measure | Judged,
    aspect_filter: str | None,
) -> list[_Spot]:
    """The arm rows of one measure: overall where it is read on one aspect, else on each aspect it is read on."""
    lead = _lead(report_input)

    def on(aspect: str) -> Mapping[str, Stat]:
        return table.per_arm if aspect == lead else table.by_aspect.get(aspect, {})

    if aspect_filter is not None:
        places: list[tuple[str | None, Mapping[str, Stat]]] = [(aspect_filter, on(aspect_filter))]
    elif table.by_aspect:
        places = [(aspect, on(aspect)) for aspect in _on_aspects(report_input, table)]
    else:
        places = [(None, table.per_arm)]
    spots: list[_Spot] = []
    for aspect, figures in places:
        stem = f"{source}.{measure_key}" if aspect is None else f"aspects.{aspect}.{source}.{measure_key}"
        spots += [(arm, aspect, None, f"{stem}.{arm}", None) for arm in _arm_ids(report_input) if arm in figures]
    return spots


def _on_strata(report_input: ReportInput, measure_key: str, aspect_filter: str | None) -> list[_Spot]:
    ids = [stratum.id for stratum in report_input.strata]
    spots: list[_Spot] = []
    for stratum in report_input.strata:
        if aspect_filter is not None and stratum.aspect != aspect_filter:
            continue
        qualified = stratum.aspect is not None and (aspect_filter is not None or ids.count(stratum.id) > 1)
        stem = f"aspects.{stratum.aspect}.strata.{stratum.id}" if qualified else f"strata.{stratum.id}"
        spots += [
            (arm, stratum.aspect if qualified else None, stratum.id, f"{stem}.{arm}.{measure_key}", stem)
            for arm in _arm_ids(report_input)
            if measure_key in stratum.per_arm.get(arm, {})
        ]
    return spots


def _on_tasks(report_input: ReportInput, measure_key: str) -> list[_Spot]:
    return [
        (arm, None, task.id, f"task.{task.id}.{arm}.{measure_key}", f"task.{task.id}")
        for task in report_input.tasks
        for arm in _arm_ids(report_input)
        if measure_key in task.per_arm.get(arm, {})
    ]


def _on_cells(report_input: ReportInput, measure_key: str) -> list[_Spot]:
    spots: list[_Spot] = []
    for cell in report_input.cells.values():
        spelled = f"{cell.row}.{cell.column}"
        spots += [
            (arm, None, spelled, f"cells.{spelled}.{arm}.{measure_key}", f"cells.{spelled}")
            for arm in _arm_ids(report_input)
            if measure_key in cell.per_arm.get(arm, {})
        ]
    return spots


def _slices_held(report_input: ReportInput, by: MeasureBy) -> bool:
    """Whether the input holds any slice of the kind a read is by (every input holds arms)."""
    held: dict[MeasureBy, bool] = {
        "arm": True,
        "stratum": bool(report_input.strata),
        "task": bool(report_input.tasks),
        "cell": bool(report_input.cells),
    }
    return held[by]


def measure_read(
    report_input: ReportInput,
    measure_key: str,
    *,
    by: MeasureBy = "arm",
    statistics: Sequence[MeasureStatistic] = ("mean", "ci", "n"),
    aspect_filter: str | None = None,
    arm_filter: str | None = None,
    limit: int = MEASURE_LIMIT,
    offset: int = 0,
) -> MeasureFigures:
    """A page of one measure's or judged dimension's statistics, by arm or within each stratum, task or cell.

    By arm, a measure read on more than the lead aspect is read on each, by its aspect's key. Within a slice,
    ``vs_control`` is each candidate's mean minus the control's there (derived, descriptive); the control has none.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param measure_key: The measure or judged dimension, by key.
    :ptype measure_key: str
    :param by: What to read it by.
    :ptype by: MeasureBy
    :param statistics: The statistics to read, each where the input states it.
    :ptype statistics: Sequence[MeasureStatistic]
    :param aspect_filter: Only this aspect: its figures by arm, or its strata.
    :ptype aspect_filter: str | None
    :param arm_filter: Only this arm.
    :ptype arm_filter: str | None
    :param limit: The most rows on the page.
    :ptype limit: int
    :param offset: The rows to skip.
    :ptype offset: int
    :return: The page.
    :rtype: MeasureFigures
    :raises RefusedInput: The key names no measure; a filter names an arm or aspect the snapshot does not hold; the
        snapshot holds no slice of that kind; ``vs_control`` is asked by arm, or with no control; an aspect is named
        for tasks or cells, which no aspect holds.
    """
    source, table = _table(report_input, measure_key)
    _known(arm_filter, _arm_ids(report_input), "arm_filter")
    _known(aspect_filter, [aspect.id for aspect in report_input.aspects], "aspect_filter")
    if not _slices_held(report_input, by):
        raise RefusedInput(
            f"this snapshot holds no {_SLICE_WORDS[by]} to read {measure_key!r} by",
            remedy="campaign_overview lists, under each measure's slices, what it can be read by",
            parameters=("by",),
            valid_values=[held for held in _SLICE_WORDS if _slices_held(report_input, held)],
        )
    if _VS_CONTROL in statistics and (by == "arm" or _control(report_input) is None):
        raise RefusedInput(
            "vs_control reads a candidate's mean against the control's within a slice, "
            + ("and this snapshot has no control" if by != "arm" else "and by arm there is no slice"),
            remedy="contrasts_list holds each candidate against the control; drop vs_control, or read by a slice",
            parameters=("statistics",),
        )
    if aspect_filter is not None and by in ("task", "cell"):
        raise RefusedInput(
            f"{by}s are not held per aspect, so aspect_filter cannot narrow them",
            remedy="omit aspect_filter",
            parameters=("aspect_filter",),
        )
    if by == "arm":
        spots = _on_arm(report_input, measure_key, source, table, aspect_filter)
    elif by == "stratum":
        spots = _on_strata(report_input, measure_key, aspect_filter)
    elif by == "task":
        spots = _on_tasks(report_input, measure_key)
    else:
        spots = _on_cells(report_input, measure_key)
    kept = [spot for spot in spots if arm_filter in (None, spot[0])]
    page, next_offset = _page(kept, limit, offset)
    control = _control(report_input)
    rows = [
        FigureRow(
            arm=cite(f"arm:{arm}", report_input),
            aspect=aspect,
            slice=slice_id,
            figures=_figures(
                (
                    f"{prefix}.{statistic}"
                    for statistic in dict.fromkeys(statistics)
                    if statistic != _VS_CONTROL or arm != control
                ),
                report_input,
            ),
        )
        for arm, aspect, slice_id, prefix, _stem in page
    ]
    stems = {(slice_id, aspect): stem for _arm, aspect, slice_id, _prefix, stem in page if slice_id and stem}
    slices = [
        SliceRow(
            slice=slice_id,
            aspect=aspect,
            name=cite_if(f"{stem}.name", report_input),
            n_cases=cite_if(f"{stem}.n_cases", report_input),
        )
        for (slice_id, aspect), stem in stems.items()
    ]
    return MeasureFigures(
        measure=measure_key, by=by, slices=slices, rows=rows, total=len(kept), offset=offset, next_offset=next_offset
    )


# --- examples_list --------------------------------------------------------------------------------------------------


class ExampleRow(_Fact):
    """One failed trial, as the snapshot froze it."""

    trial: str | None = Field(description="The trial's id: how a report names this example.")
    case: str | None = Field(description="The case's id.")
    stratum: str | None = Field(description="The stratum or task it falls in.")
    aspect: str | None = Field(description="The aspect it ran on.")
    arm: CitedFigure | None = Field(description="The arm that ran it.")
    repeat: str | None = Field(description="Which repeat of the case it was, as a label.")
    outcome: str | None = Field(description="The failure class.")
    reason: str | None = Field(description="Why it failed: the judge's reason or a note.")
    excerpt: str | None = Field(description="The end of the reply, or what the arm produced.")
    input: str | None = Field(description="The input the case gave.")
    expected: str | None = Field(description="What the case expected.")
    result_id: str | None = Field(description="The stored result the trial records, where the bundle named it.")


class ExampleListing(_Fact):
    """A page of the failed trials the snapshot froze."""

    examples: list[ExampleRow] = Field(description="The page's examples, in the input's order.")
    pool: str = Field(description="What the examples are.")
    total: int = Field(description="Examples in every page.")
    offset: int = Field(description="The examples skipped before this page.")
    next_offset: int | None = Field(description="The next page's offset; none on the last page.")


def _clipped(text: str | None) -> str | None:
    """A long text cut to the excerpt's length, its cut marked."""
    if text is None or len(text) <= EXAMPLE_EXCERPT_CHARS:
        return text
    return text[: EXAMPLE_EXCERPT_CHARS - 1] + "…"


def _example_row(report_input: ReportInput, example: Example, results: Mapping[str, str]) -> ExampleRow:
    return ExampleRow(
        trial=example.trial,
        case=example.case,
        stratum=example.stratum,
        aspect=example.aspect,
        arm=_arm_figure(report_input, example.arm),
        repeat=str(example.repeat) if example.repeat is not None else None,
        outcome=example.outcome,
        reason=_clipped(example.reason),
        excerpt=_clipped(example.excerpt),
        input=_clipped(example.input),
        expected=_clipped(example.expected),
        result_id=results.get(example.trial) if example.trial is not None else None,
    )


def examples_list(
    report_input: ReportInput,
    example_result_ids: Mapping[str, str],
    *,
    arm_filter: str | None = None,
    stratum_filter: str | None = None,
    aspect_filter: str | None = None,
    outcome_filter: str | None = None,
    limit: int = EXAMPLES_LIMIT,
    offset: int = 0,
) -> ExampleListing:
    """A page of the failed trials the snapshot froze, with the stored result each records where the bundle named it.

    An arm or aspect filter keeps an example naming none, as an examples chart does.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param example_result_ids: Each shown trial's stored result, by trial id, as the snapshot froze them.
    :ptype example_result_ids: Mapping[str, str]
    :param arm_filter: Only this arm's examples.
    :ptype arm_filter: str | None
    :param stratum_filter: Only the examples in this stratum or task.
    :ptype stratum_filter: str | None
    :param aspect_filter: Only this aspect's examples.
    :ptype aspect_filter: str | None
    :param outcome_filter: Only the examples of this failure class.
    :ptype outcome_filter: str | None
    :param limit: The most examples on the page.
    :ptype limit: int
    :param offset: The examples to skip.
    :ptype offset: int
    :return: The page.
    :rtype: ExampleListing
    :raises RefusedInput: A filter names an arm, stratum, aspect or failure class the snapshot does not hold.
    """
    examples = report_input.examples
    _known(arm_filter, _arm_ids(report_input), "arm_filter")
    slices = [*(s.id for s in report_input.strata), *(t.id for t in report_input.tasks)]
    _known(stratum_filter, [*slices, *(e.stratum for e in examples if e.stratum is not None)], "stratum_filter")
    aspects = [*(a.id for a in report_input.aspects), *(e.aspect for e in examples if e.aspect is not None)]
    _known(aspect_filter, aspects, "aspect_filter")
    outcomes = list(dict.fromkeys(e.outcome for e in examples if e.outcome is not None))
    if outcome_filter is not None and outcome_filter not in outcomes:
        raise RefusedInput(
            f"outcome_filter {outcome_filter!r} is no failure class this snapshot's examples hold",
            remedy=f"name one of {', '.join(outcomes)}" if outcomes else "omit outcome_filter: no example names one",
            parameters=("outcome_filter",),
            valid_values=outcomes,
        )
    kept = [
        example
        for example in examples
        if (arm_filter is None or example.arm in (None, arm_filter))
        and stratum_filter in (None, example.stratum)
        and (aspect_filter is None or example.aspect in (None, aspect_filter))
        and outcome_filter in (None, example.outcome)
    ]
    page, next_offset = _page(kept, limit, offset)
    return ExampleListing(
        examples=[_example_row(report_input, example, example_result_ids) for example in page],
        pool=_POOL,
        total=len(kept),
        offset=offset,
        next_offset=next_offset,
    )


# --- caveats_list ---------------------------------------------------------------------------------------------------


class CaveatRow(_Fact):
    """One caveat: where it may land, whether code proved it immaterial, and what it touches."""

    id: str = Field(description="The caveat's id, as a report's flag and confidence clauses name it.")
    kind: str = Field(
        description="late, judge_uncalibrated, confound, apparatus, short_run, unsettled, latency or unreported."
    )
    reach: CaveatReach = Field(description="report (the confidence rests on it), scoped (beside a claim) or method.")
    immaterial: bool = Field(description="Whether code proved the decision holds across its whole range.")
    checked: str | None = Field(
        description="The line code renders for an immaterial caveat, which a report never restates."
    )
    touches_arms: list[str] = Field(description="The arms it touches, by id.")
    touches_aspects: list[str] = Field(description="The aspects it touches, by id.")
    touches_measures: list[str] = Field(description="The measures and judged dimensions it touches, by key.")
    says: str | None = Field(description="What the input says of it, in its own words.")
    figures: list[CitedFigure] = Field(description="The figures it rests on.")


class CaveatListing(_Fact):
    """Every caveat the snapshot carries, in the order a report lands them."""

    caveats: list[CaveatRow] = Field(description="Each caveat.")


class _Touch(BaseModel):
    """What one caveat touches and says, before its row is assembled."""

    model_config = ConfigDict(frozen=True)

    arms: list[str] = []
    aspects: list[str] = []
    measures: list[str] = []
    says: str | None = None
    keys: list[str] = []


def _distinct(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value is not None))


def _late_touch(report_input: ReportInput, late: LateCriterion) -> _Touch:
    """What a late criterion touches (the asked aspects, the control, the margined aspects, its floor's or bar's reads,
    or the measure it names), and when it was set."""
    criterion, name = late.criterion, late.name
    said = f"the {criterion}{' ' + name if name else ''} was set after the results"
    keys = [f"late_criteria.{_late_name(late)}.set_at"]
    if criterion == "aspect":
        return _Touch(aspects=[a.id for a in report_input.aspects if a.asked], says=said, keys=keys)
    if criterion == "control":
        return _Touch(arms=_distinct([_control(report_input)]), says=said, keys=keys)
    if criterion == "margin":
        margined = [
            a.id for a in report_input.aspects if a.margin is not None or any(r.margin is not None for r in a.readings)
        ]
        return _Touch(aspects=margined, says=said, keys=keys)
    if criterion == "floor":
        reads = [read for read in report_input.floors if read.floor == name]
        return _Touch(
            arms=_distinct(r.arm for r in reads), aspects=_distinct(r.aspect for r in reads), says=said, keys=keys
        )
    if criterion == "bar":
        bars = [read for read in report_input.bars if read.measure == name]
        return _Touch(
            arms=_distinct(r.arm for r in bars),
            aspects=_distinct(r.aspect for r in bars),
            measures=_distinct([name]),
            says=said,
            keys=keys,
        )
    named = next((c for c in report_input.criteria if c.id == name), None)
    return _Touch(measures=_distinct([named.measure if named is not None else name]), says=said, keys=keys)


def _confound_touch(report_input: ReportInput, dimension: str, says: str | None) -> _Touch:
    """What a confound touches: the contrasts naming it, their arms, aspects and measures."""
    naming = [contrast for contrast in report_input.contrasts if dimension in contrast.confounds]
    return _Touch(
        arms=_distinct(c.arm for c in naming),
        aspects=_distinct(c.aspect for c in naming),
        measures=_distinct(c.measure for c in naming),
        says=says,
    )


def _touch(report_input: ReportInput, caveat: Caveat, unsettled: Mapping[str, tuple[str, ...]]) -> _Touch:
    caveats = report_input.caveats
    lates = {f"late:{_late_name(late)}": late for late in caveats.late_criteria}
    if caveat.id in lates:
        return _late_touch(report_input, lates[caveat.id])
    head, _, rest = caveat.id.partition(":")
    if caveat.id == "judge_uncalibrated":
        return _Touch(measures=_distinct([*report_input.judged, *caveats.judge_tiers]))
    if head == "confound":
        return _confound_touch(report_input, rest, caveats.confounds.get(rest))
    if head == "apparatus":
        apparatus = next(c for c in caveats.apparatus_confounds if c.dimension == rest)
        return _confound_touch(report_input, rest, caveats.confounds.get(rest) or apparatus.status)
    if head == "short_run":
        run = next(r for r in caveats.short_runs if r.run == rest)
        return _Touch(arms=[run.arm] if run.arm is not None else _arm_ids(report_input), says=run.note)
    if head == "unsettled":
        arms = list(unsettled.get(caveat.id, ()))
        keys = [f"aspects.{rest}.margin"]
        for arm in arms:
            keys += [f"aspects.{rest}.reading.{arm}.interval", f"aspects.{rest}.flip_margin.{arm}"]
        return _Touch(arms=arms, aspects=[rest], keys=keys)
    if caveat.id == "latency":
        return _Touch(keys=["caveats.latency"])
    return _Touch()


def caveats_list(report_input: ReportInput) -> CaveatListing:
    """Every caveat the snapshot carries: where it may land, whether code proved it immaterial, and what it touches,
    each read from the input (a confound: the contrasts naming it; a short run: its arm; a late criterion: what it
    set; an unsettled aspect: its candidates).

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :return: The caveats.
    :rtype: CaveatListing
    """
    materiality = caveat_materiality(report_input)
    unsettled = unsettled_arms(report_input)
    rows: list[CaveatRow] = []
    for caveat in report_caveats(report_input):
        touch = _touch(report_input, caveat, unsettled)
        figures = _figures(touch.keys, report_input)
        held = materiality[caveat.id]
        rows.append(
            CaveatRow(
                id=caveat.id,
                kind=caveat.kind,
                reach=caveat.reach,
                immaterial=held.immaterial,
                checked=held.checked,
                touches_arms=touch.arms,
                touches_aspects=touch.aspects,
                touches_measures=touch.measures,
                says=touch.says,
                figures=figures,
            )
        )
    return CaveatListing(caveats=rows)
