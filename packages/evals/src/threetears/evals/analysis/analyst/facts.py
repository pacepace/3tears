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
- :func:`caveats_list`: every caveat, where it may land, whether code proved it immaterial, and what it touches: the
  ones the input states, then the ones what it holds implies
  (:func:`~threetears.evals.analysis.report_decision.implied_caveats`).

**Intervals are read wherever they are held.** A measure read with no statistics named reads every statistic the
input holds on it read that way, its interval (``ci`` or ``half_width``) among them, and names what it holds and the
intervals' level; the overview says, for each measure, which statistics each way of reading it holds, so a measure
held only per cell or stratum is held per slice, not missing. **One interval is derived, structurally.** Where an
overall figure holds no interval and is, by the numbers the input states, the plain mean of one measure's
target-matrix cells, each stating its own spread (a balanced accuracy beside its cells' accuracies), code derives its
interval from theirs, the cells read as independent because the target matrix partitions the cases
(:mod:`~threetears.evals.analysis.cell_mean`): ``overall_interval.<measure>.<arm>.*`` names it, marked derived, a
measure read by arm lists it beside the figure, and a gate reads it, so no ``point_estimate:<criterion>`` caveat is
implied there. Any other overall with no interval is not assumed to be a mean of its slices: it reads as a point
estimate (``point_estimate:<criterion>`` where a gate reads it), its slices' intervals stay citable, and the missing
overall interval is the input's to carry (an upstream gap).

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
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import CitedFigure, cite, cite_if
from threetears.evals.analysis.cell_mean import NoCellMean, cell_mean_interval
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import NamedKind, reader_names
from threetears.evals.analysis.report_decision import (
    Caveat,
    CaveatReach,
    ImpliedCaveat,
    Materiality,
    caveat_materiality,
    every_caveat,
    implied_caveats,
    report_caveats,
    unsettled_arms,
)
from threetears.evals.analysis.report_input import (
    Aspect,
    BarRead,
    Contrast,
    Disclosure,
    Example,
    FloorRead,
    Judged,
    LateCriterion,
    Measure,
    PairRead,
    PriorInsight,
    Reading,
    ReportInput,
    RequestSettings,
    RunRig,
    Stat,
)
from threetears.evals.analysis.report_input_build import EXAMPLE_EXCERPT_CHARS
from threetears.evals.analysis.report_margins import criterion_aspect
from threetears.evals.kernel.errors import RefusedInput

__all__ = [
    "CAVEATS_LIMIT",
    "CONTRASTS_LIMIT",
    "EXAMPLES_LIMIT",
    "MEASURE_LIMIT",
    "OVERVIEW_LIMIT",
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
    "confound_caveat_id",
    "contrast_key",
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
OVERVIEW_LIMIT = 200
CAVEATS_LIMIT = 50

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

#: An arm's unanswered calls beside its cost, as the unanswered keys name each figure after the bare count.
_UNANSWERED = ("retried", "lost", "share", "reasons", "uncounted")

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

#: The statistics a figure may hold, as the input stores them, in the grammar's order: what a read with no statistics
#: named reads, each where the input states it.
_STORED: tuple[MeasureStatistic, ...] = ("mean", "ci", "half_width", "sem", "n", "n_cases", "p50", "p95", "change")

#: What :func:`measure_read` reads one measure by: each arm, or each arm within each stratum, task or cell.
MeasureBy = Literal["arm", "stratum", "task", "cell"]

#: Each slice kind's plural, as a refusal names it.
_SLICE_WORDS: dict[MeasureBy, str] = {"arm": "arms", "stratum": "strata", "task": "tasks", "cell": "cells"}

#: The statistic that reads an arm against the control within a slice, which an arm's overall figure has no key for.
_VS_CONTROL = "vs_control"

#: What the examples a snapshot holds are: the frozen pool an analyst pages (up to three per slice, as
#: :func:`~threetears.evals.analysis.analyst.snapshot.analyst_input` holds them).
_POOL = (
    "the failed trials frozen with the snapshot: up to three per aspect, arm and stratum, every slice's first before "
    "any slice's second and its other cases before a repeat; each excerpt cut around the failure (centred on the "
    "passage the judge's reason quotes, else the reply's end), with the judge's or the check's reason. They are a "
    "sample: kept_from says, per aspect and arm, how many were kept of how many failed, so a pattern read in them "
    "names the share of the failures it rests on"
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
    # Left out when empty, so an arm whose calls all came back reads as it did before the field existed.
    unanswered: list[CitedFigure] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="Beside its cost, where any went unanswered: its calls unanswered (`calls.unanswered.<arm>`), "
        "retried and lost, their share of the calls sent, the reasons, and the trials that counted none.",
    )


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
    same_cases: list[CitedFigure] = Field(
        description="Whether it ran the same cases as each later aspect (`overlap.<aspect>.<other>.same`): the same, "
        "partly, different, or not known; `shared`, `only_first` and `only_second` count them where known."
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
        description="The arms holding no figure on it overall or in any slice, by id; empty where every arm holds one "
        "somewhere. A measure held only within slices is held per slice, not missing: statistics says where."
    )
    aspects: list[str] = Field(description="The aspects it is read on, where it is read on more than the lead one.")
    slices: list[str] = Field(description="Where measure_read can read it: arm, stratum, task, cell.")
    statistics: list[str] = Field(
        description="The statistics it holds, per way it can be read (`cell: mean, half_width`): ci or half_width is "
        "an interval, at its level. measure_read with no statistics named reads every one."
    )
    key_form: str = Field(description="The figure key that reads one arm's statistic on it.")
    judge: list[CitedFigure] = Field(
        description="For a judged dimension, its judge as the input states it: its evidence tier, model, calibration "
        "state, the human ratings it was calibrated on and its agreement with them (`judged.<dimension>.<field>`); "
        "empty for a measure, and a field the input leaves unstated is left out."
    )


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
    aspect: str | None = Field(
        default=None,
        description="The aspect whose figures it reads: the one it is declared on, else the lead aspect; None where "
        "the snapshot holds no aspect.",
    )


class LateRow(_Fact):
    """One criterion set after the results were seen, and when."""

    criterion: str = Field(description="What was set: aspect, control, margin, floor, gate.")
    name: str | None = Field(description="Which one, where several could be.")
    set_at: CitedFigure = Field(description="When it was set (`late_criteria.<late>.set_at`).")


#: What a caveat's kind is, as a row's field describes it: the kinds the input states, then the kinds its figures imply.
_CAVEAT_KINDS = (
    "stated: late, judge_uncalibrated, confound, apparatus, short_run, unsettled, latency or unreported; implied: "
    "judge_unstated, judge_partly_uncalibrated, no_control, nothing_declared, point_estimate, borderline or best_of_n."
)

#: Where a caveat comes from, as a row's field describes it.
_CAVEAT_SOURCE = (
    "stated: the input states it as a caveat; implied: what the input holds implies it (a judge it does not describe, "
    "no control, nothing declared, a gate cleared on a point estimate or on a figure whose interval reaches past it, a "
    "pick among many)."
)


class CaveatRef(_Fact):
    """One caveat by its id, and where it may land; caveats_list says what it touches."""

    id: str = Field(description="The caveat's id.")
    kind: str = Field(description=_CAVEAT_KINDS)
    reach: CaveatReach = Field(description="report (the confidence rests on it), scoped (beside a claim) or method.")
    source: Literal["stated", "implied"] = Field(description=_CAVEAT_SOURCE)


class RigRow(_Fact):
    """The runs measured with one rig: every field of their apparatus but the execution modes alike, which are summed."""

    runs: dict[str, str | None] = Field(
        description="The runs measured with it, in the input's order, each by id to the arm it ran; none for a run the "
        "design placed in no arm."
    )
    aspects: list[str] = Field(description="The aspects those runs ran, by id.")
    judges: dict[str, str] | None = Field(
        description="The model each judged dimension was asked of; none: not recorded."
    )
    judge_asked: RequestSettings | None = Field(description="What every judge call was asked for; none: not recorded.")
    judge_temperature: float | None = Field(
        description="The temperature judge calls were asked at; none: not recorded."
    )
    simulator: str | None = Field(description="The simulated user's model; none: none was pinned.")
    measure_latency: bool | None = Field(description="Whether latency was declared under test; none: nothing declared.")
    arm_schedule: str | None = Field(description="How the arms shared time; none: not recorded.")
    cell_concurrency: int = Field(description="How many cells executed at once; 1 is serial.")
    rubric_scales: dict[str, str] = Field(description="The scale each rubric dimension was answered on.")
    execution_modes: dict[str, int] = Field(
        description="The runs' results by the execution mode each recorded, summed."
    )
    # Left out when empty, so a rig whose runs' hosts disclosed nothing reads as it did before the field existed.
    disclosed: dict[str, Disclosure] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="What the launching host disclosed about each of the runs (its machine, load, how it measures), "
        "by run id, for each run it disclosed something for. A disclosure, never part of the rig: runs that disclose "
        "different values share one row.",
    )


def _rigs(report_input: ReportInput) -> list[RigRow]:
    """The input's run rigs, one row per rig in first-seen order: runs alike in every field but their execution modes
    and their hosts' disclosures share one row, which names each with the arm it ran, sums the modes and keeps each
    run's disclosure by its id: a disclosure is no coordinate of the rig, so it splits no row."""
    grouped: dict[str, tuple[RunRig, list[RunRig]]] = {}
    for rig in report_input.apparatus or ():
        shape = rig.model_dump_json(exclude={"run", "arm", "aspect", "execution_modes", "disclosure"})
        grouped.setdefault(shape, (rig, []))[1].append(rig)
    rows: list[RigRow] = []
    for first, members in grouped.values():
        modes: dict[str, int] = {}
        for member in members:
            for mode, count in member.execution_modes.items():
                modes[mode] = modes.get(mode, 0) + count
        rows.append(
            RigRow(
                runs={member.run: member.arm for member in members},
                aspects=list(dict.fromkeys(member.aspect for member in members)),
                judges=first.judges,
                judge_asked=first.judge_asked,
                judge_temperature=first.judge_temperature,
                simulator=first.simulator,
                measure_latency=first.measure_latency,
                arm_schedule=first.arm_schedule,
                cell_concurrency=first.cell_concurrency,
                rubric_scales=dict(first.rubric_scales),
                execution_modes=dict(sorted(modes.items())),
                disclosed={member.run: member.disclosure for member in members if member.disclosure is not None},
            )
        )
    return rows


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
    # Left out when empty, so an input whose runs recorded no apparatus reads as it did before the field existed.
    apparatus: list[RigRow] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="The rigs the runs were measured with, one row per rig, naming the runs measured with it: the judge "
        "each dimension was asked of and what it was asked for, the simulated user, whether latency was under test, the "
        "arms' schedule, the cells run at once, each rubric dimension's scale and the runs' results by execution mode. "
        "Paged as rows, after the prior insights.",
    )
    names: dict[str, dict[str, str]] = Field(
        description="The one name a reader is shown for each id, by kind (aspect, measure, judged, stratum, task, "
        "criterion), then id."
    )
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("row", "rows")
    total: int = Field(
        description="The rows the overview holds: its arms, aspects, measures, criteria, cells, caveats, late "
        "criteria, prior insights and names, paged in that order; the campaign's own figures are on every page."
    )
    offset: int = Field(description="The first row this page holds.")
    next_offset: int | None = Field(description="The offset of the next page of rows; None when this is the last.")


def _unanswered(report_input: ReportInput, arm_id: str) -> list[CitedFigure]:
    """An arm's unanswered calls, cited beside its cost where any went unanswered; none where none did or no role
    counted them."""
    row = report_input.scoreboard.get(arm_id)
    if row is None or not row.unanswered:
        return []
    if not any(held.retried or held.lost for held in row.unanswered.values()):
        return []
    # The bare count is not counted where a role counted one kind only, so it is cited where it reads.
    keys = [f"calls.unanswered.{arm_id}", *(f"calls.unanswered.{arm_id}.{field}" for field in _UNANSWERED)]
    return _figures(keys, report_input)


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


def _stats_by_slice(report_input: ReportInput, key: str, table: Measure | Judged) -> dict[MeasureBy, list[Stat]]:
    """Every figure the input holds on one measure, by the way it is read: by arm (overall and on each aspect), and
    within each stratum, task and cell; a way holding none is left out."""
    held: dict[MeasureBy, list[Stat]] = {
        "arm": [*table.per_arm.values(), *(stat for arms in table.by_aspect.values() for stat in arms.values())],
        "stratum": [f[key] for s in report_input.strata for f in s.per_arm.values() if key in f],
        "task": [f[key] for t in report_input.tasks for f in t.per_arm.values() if key in f],
        "cell": [f[key] for c in report_input.cells.values() for f in c.per_arm.values() if key in f],
    }
    return {by: stats for by, stats in held.items() if stats}


def _stated(stats: Iterable[Stat]) -> list[MeasureStatistic]:
    """The statistics any of the figures states, in the grammar's order."""
    held = list(stats)
    return [statistic for statistic in _STORED if any(getattr(stat, statistic) is not None for stat in held)]


def _slices_of(report_input: ReportInput, key: str, table: Measure | Judged) -> list[str]:
    return list(_stats_by_slice(report_input, key, table))


def _arms_holding(report_input: ReportInput, key: str, table: Measure | Judged) -> set[str]:
    """The arms holding a figure on one measure anywhere: overall, on an aspect, or within any slice."""
    slices = [
        *(s.per_arm for s in report_input.strata),
        *(t.per_arm for t in report_input.tasks),
        *(c.per_arm for c in report_input.cells.values()),
    ]
    held = set(table.per_arm) | {arm for figures in table.by_aspect.values() for arm in figures}
    return held | {arm for per_arm in slices for arm, figures in per_arm.items() if key in figures}


def _measure_row(
    report_input: ReportInput, key: str, source: Literal["measure", "judged"], table: Measure | Judged
) -> MeasureRow:
    aspects = _on_aspects(report_input, table)
    form = f"aspects.<aspect>.{source}.{key}.<arm>.<statistic>" if aspects else f"{source}.{key}.<arm>.<statistic>"
    held = _arms_holding(report_input, key, table)
    by_slice = _stats_by_slice(report_input, key, table)
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
        slices=list(by_slice),
        statistics=[f"{by}: {', '.join(_stated(stats))}" for by, stats in by_slice.items()],
        key_form=form,
        judge=[] if source == "measure" else _figures(_judge_keys(key), report_input),
    )


#: What a judged dimension's judge is cited by, in the order a reader weighs it: how far its scores can be trusted,
#: whose they are, and what it was calibrated on.
_JUDGE_FIELDS = ("tier", "model", "calibration", "calibration_ratings", "agreement")


def _judge_keys(dimension: str) -> list[str]:
    """The keys of one judged dimension's judge figures."""
    return [f"judged.{dimension}.{field}" for field in _JUDGE_FIELDS]


def campaign_overview(
    report_input: ReportInput, snapshot_id: str, *, limit: int | None = None, offset: int = 0
) -> CampaignOverview:
    """What a snapshot holds, read first: every id the other tools take, and the campaign's own figures.

    Its rows (arms, aspects, measures, criteria, cells, caveats, late criteria, prior insights, the runs' rigs, then each
    reader name) are paged in that order, ``limit`` from ``offset`` (every one where ``limit`` is None); the campaign's own figures
    (its meta, candidates, cases, repeats, when the late criteria were set) are on every page.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param snapshot_id: The snapshot's id.
    :ptype snapshot_id: str
    :param limit: The most rows one page holds; None for every one.
    :ptype limit: int | None
    :param offset: The first row the page holds.
    :ptype offset: int
    :return: The overview, or one page of it.
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
            unanswered=_unanswered(report_input, arm.id),
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
            same_cases=[
                cite(f"overlap.{aspect.id}.{other.id}.same", report_input)
                for other in report_input.aspects[index + 1 :]
            ],
        )
        for index, aspect in enumerate(report_input.aspects)
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
            aspect=criterion_aspect(report_input, criterion),
        )
        for criterion in report_input.criteria
    ]
    names = reader_names(report_input)
    named = [(kind, held, name) for kind, by_id in names.items() for held, name in by_id.items()]
    measures = [_measure_row(report_input, *table) for table in _tables(report_input)]
    cells = [cite(f"cells.{cell.row}.{cell.column}.name", report_input) for cell in report_input.cells.values()]
    caveats = [CaveatRef(id=caveat.id, kind=caveat.kind, reach=caveat.reach, source="stated") for caveat in reaches]
    caveats += [
        CaveatRef(id=caveat.id, kind=caveat.kind, reach=caveat.reach, source="implied")
        for caveat in implied_caveats(report_input)
    ]
    late_rows = [
        LateRow(
            criterion=criterion.criterion,
            name=criterion.name,
            set_at=cite(f"late_criteria.{_late_name(criterion)}.set_at", report_input),
        )
        for criterion in late
    ]
    insights = list(report_input.prior_insights)
    rigs = _rigs(report_input)
    sections: list[Sequence[object]] = [
        arms,
        aspects,
        measures,
        criteria,
        cells,
        caveats,
        late_rows,
        insights,
        rigs,
        named,
    ]
    total = sum(len(section) for section in sections)
    end = total if limit is None else min(offset + limit, total)
    paged = _windowed(sections, offset, end)
    return CampaignOverview(
        snapshot=snapshot_id,
        meta=_figures((f"meta.{field}" for field in _META_FIELDS), report_input),
        n_candidates=cite("n_candidates", report_input),
        cases=cite_if("cases", report_input),
        repeats=cite_if("repeats", report_input),
        arms=[arms[index] for index in paged[0]],
        aspects=[aspects[index] for index in paged[1]],
        measures=[measures[index] for index in paged[2]],
        criteria=[criteria[index] for index in paged[3]],
        cells=[cells[index] for index in paged[4]],
        caveats=[caveats[index] for index in paged[5]],
        late_criteria=[late_rows[index] for index in paged[6]],
        late_set_at=cite_if("late_criteria.set_at", report_input) if late else None,
        prior_insights=[insights[index] for index in paged[7]],
        apparatus=[rigs[index] for index in paged[8]],
        names=_names_of(names, [named[index] for index in paged[9]], whole=limit is None),
        total=total,
        offset=offset,
        next_offset=end if end < total else None,
    )


def _windowed(sections: Sequence[Sequence[object]], start: int, end: int) -> list[range]:
    """For each section of rows paged one after another, the indices of its rows from ``start`` up to ``end``."""
    windows: list[range] = []
    first = 0
    for section in sections:
        windows.append(range(max(start - first, 0), max(min(end - first, len(section)), 0)))
        first += len(section)
    return windows


def _names_of(
    names: Mapping[NamedKind, Mapping[str, str]], shown: Sequence[tuple[str, str, str]], *, whole: bool
) -> dict[str, dict[str, str]]:
    """The reader names a page holds, by kind then id: every kind where the overview is whole, else the kinds the
    page holds a name of."""
    paged: dict[str, dict[str, str]] = {kind: {} for kind in names} if whole else {}
    for kind, held, name in shown:
        paged.setdefault(kind, {})[held] = name
    return paged


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
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("row", "rows")
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
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("contrast", "contrasts")
    total: int = Field(description="Contrasts in every page.")
    offset: int = Field(description="The contrasts skipped before this page.")
    next_offset: int | None = Field(description="The next page's offset; none on the last page.")


def contrast_key(report_input: ReportInput, contrast: Contrast) -> str:
    """The key a contrast's figures are read under: unqualified, or by its aspect where the input holds its arm and
    measure more than once. ``contrasts_list`` cites a contrast's figures under it, and the editor's
    ``caveat_not_carried`` fact (``analysis/report_doc_facts.py``) names them alike.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param contrast: One of its contrasts.
    :ptype contrast: Contrast
    :return: The key's path, before its statistic (``contrast.<arm>.<measure>``).
    :rtype: str
    """
    held = [c for c in report_input.contrasts if (c.arm, c.measure) == (contrast.arm, contrast.measure)]
    if len(held) > 1 and contrast.aspect is not None:
        return f"aspects.{contrast.aspect}.contrast.{contrast.arm}.{contrast.measure}"
    return f"contrast.{contrast.arm}.{contrast.measure}"


def confound_caveat_id(caveats: Iterable[Caveat], dimension: str) -> str:
    """A confound's caveat id: its apparatus or confound caveat where the input lists one, else the dimension.

    ``contrasts_list`` names each contrast's confounds by it, and the editor's ``caveat_not_carried`` fact reads them
    alike.

    :param caveats: The caveats code assigns on the snapshot (``report_caveats``).
    :ptype caveats: Iterable[Caveat]
    :param dimension: The confound's dimension, as a contrast names it.
    :ptype dimension: str
    :return: ``apparatus:<dimension>`` or ``confound:<dimension>``, whichever the caveats hold, else the dimension.
    :rtype: str
    """
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
            confounds=[confound_caveat_id(caveats, dimension) for dimension in contrast.confounds],
            figures=_figures(
                (f"{contrast_key(report_input, contrast)}.{statistic}" for statistic in _CONTRAST), report_input
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
    derived: list[CitedFigure] = Field(
        description="The interval code derived for an overall figure that states none and is the plain mean of its "
        "target-matrix cells (overall_interval.<measure>.<arm>.*), each marked derived; empty where none is."
    )


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
    level: str | None = Field(description="The confidence level of every interval (ci, half_width) it holds.")
    held: list[str] = Field(
        description="Every statistic the snapshot holds on it read this way, whichever were asked: ci or half_width "
        "is an interval on the mean."
    )
    not_shown: list[str] = Field(
        description="The statistics it holds that this read did not ask for; omit statistics to read every one."
    )
    slices: list[SliceRow] = Field(description="The slices this page's rows are read within; none by arm.")
    rows: list[FigureRow] = Field(description="The page's rows: by slice, then arm, in the input's order.")
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("row", "rows")
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


#: The figures of an overall interval derived from the cells that a measure read by arm lists beside the figure.
_DERIVED_OVERALL = ("interval", "half_width", "level")


def _derived_overall(report_input: ReportInput, measure_key: str, arm: str) -> list[CitedFigure]:
    """The interval derived for an arm's overall figure from its cells, by its keys
    (``overall_interval.<measure>.<arm>.*``, each marked derived); none where the figure does not qualify."""
    try:
        cell_mean_interval(report_input, measure_key, arm)
    except NoCellMean:
        # NOSILENT: an overall that is no plain mean of its cells is the input's shape, not a fault: the row lists the
        # figures the input states and no derived interval, which is what an empty list says.
        return []
    return [cite(f"overall_interval.{measure_key}.{arm}.{field}", report_input) for field in _DERIVED_OVERALL]


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
    statistics: Sequence[MeasureStatistic] | None = None,
    aspect_filter: str | None = None,
    arm_filter: str | None = None,
    limit: int = MEASURE_LIMIT,
    offset: int = 0,
) -> MeasureFigures:
    """A page of one measure's or judged dimension's statistics, by arm or within each stratum, task or cell.

    With no statistics named, every statistic the snapshot holds on the measure read this way is read: the mean, its
    interval (``ci`` or ``half_width``) wherever one is held, the counts, the percentiles. The result names every
    statistic held (``held``), the ones a narrower read left out (``not_shown``) and the intervals' level, so an
    interval is never hidden behind a read that did not ask for it.

    By arm, a measure read on more than the lead aspect is read on each, by its aspect's key. An arm's overall figure
    that states no interval and is the plain mean of its target-matrix cells lists, beside its statistics
    (``derived``), the interval derived from theirs (``overall_interval.<measure>.<arm>.*``: the interval, its half-width and level, each marked
    derived; its ``basis`` key says how it was built) wherever the read asks for every statistic or for an interval.
    Within a slice,
    ``vs_control`` is each candidate's mean minus the control's there (derived, descriptive); the control has none.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param measure_key: The measure or judged dimension, by key.
    :ptype measure_key: str
    :param by: What to read it by.
    :ptype by: MeasureBy
    :param statistics: The statistics to read, each where the input states it; None for every one it holds.
    :ptype statistics: Sequence[MeasureStatistic] | None
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
        snapshot holds no slice of that kind, or holds the measure in none; ``vs_control`` is asked by arm, or with no
        control; an aspect is named for tasks or cells, which no aspect holds.
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
    by_slice = _stats_by_slice(report_input, measure_key, table)
    if by not in by_slice:
        where = ", ".join(_SLICE_WORDS[held] for held in by_slice) or "nowhere"
        raise RefusedInput(
            f"{measure_key!r} holds no figure read by {by}: it is held per {where}",
            remedy=f"read it by {' or '.join(by_slice)}" if by_slice else "campaign_overview lists every measure key",
            parameters=("by",),
            valid_values=list(by_slice),
        )
    held = _stated(by_slice[by])
    asked = held if statistics is None else list(dict.fromkeys(statistics))
    if _VS_CONTROL in asked and (by == "arm" or _control(report_input) is None):
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
    overall: set[str | None] = {None, _lead(report_input)} if by == "arm" else set()
    interval_asked = statistics is None or bool({"ci", "half_width"} & set(asked))
    rows = [
        FigureRow(
            arm=cite(f"arm:{arm}", report_input),
            aspect=aspect,
            slice=slice_id,
            figures=_figures(
                (f"{prefix}.{statistic}" for statistic in asked if statistic != _VS_CONTROL or arm != control),
                report_input,
            ),
            derived=_derived_overall(report_input, measure_key, arm) if interval_asked and aspect in overall else [],
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
        measure=measure_key,
        by=by,
        level=table.level,
        held=list(held),
        not_shown=[statistic for statistic in held if statistic not in asked],
        slices=slices,
        rows=rows,
        total=len(kept),
        offset=offset,
        next_offset=next_offset,
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
    reason: str | None = Field(description="Why it failed: the judge's reason or a note, whole.")
    excerpt: str | None = Field(
        description="What the judge read, cut around the failure where the snapshot was frozen so; each cut end marked."
    )
    input: str | None = Field(description="The input the case gave.")
    expected: str | None = Field(description="What the case expected.")
    # Left out when empty, so an example frozen without every dimension's reason reads as it did.
    reasons: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Each judged dimension's reason, by dimension, the primary's among them, cut to the excerpt's "
        "length as the input and the excerpt are; empty where the snapshot froze none.",
    )
    result_id: str | None = Field(description="The stored result the trial records, where the bundle named it.")


class ExampleSlice(_Fact):
    """One aspect and arm's failed trials (one stratum's, where the call names one): how many the snapshot froze as
    examples, and how many the run had."""

    aspect: str = Field(description="The aspect.")
    arm: CitedFigure = Field(description="The arm.")
    stratum: str | None = Field(description="The stratum or task, where the call names one; none for all of them.")
    kept: CitedFigure = Field(description="The failed trials frozen as examples (`count.examples.<aspect>.<arm>`).")
    failed: CitedFigure | None = Field(
        description="The failed trials the run had, as the examples count a failure (`count.failures.<aspect>.<arm>`); "
        "none where the snapshot does not count them."
    )
    unknown: str = Field(description="Why the failed trials are not known, where they are not; empty where they are.")


class ExampleListing(_Fact):
    """A page of the failed trials the snapshot froze."""

    examples: list[ExampleRow] = Field(description="The page's examples, in the input's order.")
    pool: str = Field(description="What the examples are.")
    kept_from: list[ExampleSlice] = Field(
        description="Per aspect and arm the filters keep (and stratum, where stratum_filter names one), the examples "
        "frozen against the failed trials the run had, whatever the outcome filter: what the examples are a sample of."
    )
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("example", "examples")
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
        reason=example.reason,
        excerpt=_clipped(example.excerpt),
        input=_clipped(example.input),
        expected=_clipped(example.expected),
        reasons={dim: _clipped(reason) or "" for dim, reason in (example.reasons or {}).items()},
        result_id=results.get(example.trial) if example.trial is not None else None,
    )


def _kept_from(
    report_input: ReportInput, arm_filter: str | None, aspect_filter: str | None, stratum_filter: str | None
) -> list[ExampleSlice]:
    """Each aspect and arm an example or a failed-trial count names, as the filters keep them: the examples kept, and
    the failed trials they were kept from where the snapshot counts them."""
    named = [(example.aspect, example.arm, example.stratum) for example in report_input.examples]
    named += [(row.aspect, row.arm, row.stratum) for row in report_input.failed_trials or ()]
    order = {arm: index for index, arm in enumerate(_arm_ids(report_input))}
    aspects = {aspect.id: index for index, aspect in enumerate(report_input.aspects)}
    pairs = {
        (aspect, arm)
        for aspect, arm, stratum in named
        if aspect is not None
        and arm in order
        and arm_filter in (None, arm)
        and aspect_filter in (None, aspect)
        and stratum_filter in (None, stratum)
    }
    rows: list[ExampleSlice] = []
    for aspect, arm in sorted(pairs, key=lambda pair: (aspects.get(pair[0], len(aspects)), pair[0], order[pair[1]])):
        tail = f".{stratum_filter}" if stratum_filter is not None else ""
        failed: CitedFigure | None = None
        unknown = ""
        try:
            failed = cite(f"count.failures.{aspect}.{arm}{tail}", report_input)
        except UnstatedFigure as untold:
            # NOSILENT: an uncounted pool is the snapshot's shape, not a fault: the row says why in `unknown`.
            unknown = str(untold)
        rows.append(
            ExampleSlice(
                aspect=aspect,
                arm=cite(f"arm:{arm}", report_input),
                stratum=stratum_filter,
                kept=cite(f"count.examples.{aspect}.{arm}{tail}", report_input),
                failed=failed,
                unknown=unknown,
            )
        )
    return rows


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
        kept_from=_kept_from(report_input, arm_filter, aspect_filter, stratum_filter),
        total=len(kept),
        offset=offset,
        next_offset=next_offset,
    )


# --- caveats_list ---------------------------------------------------------------------------------------------------


class CaveatRow(_Fact):
    """One caveat: where it may land, whether code proved it immaterial, and what it touches."""

    id: str = Field(description="The caveat's id, as a report's flag and confidence clauses name it.")
    kind: str = Field(description=_CAVEAT_KINDS)
    reach: CaveatReach = Field(description="report (the confidence rests on it), scoped (beside a claim) or method.")
    source: Literal["stated", "implied"] = Field(description=_CAVEAT_SOURCE)
    matters_if: str | None = Field(
        description="For an implied caveat, when it matters: what a report's material_if clause weighs. None for a "
        "stated one."
    )
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

    caveats: list[CaveatRow] = Field(description="Each caveat on this page: the stated ones, then the implied.")
    #: What its total counts, one and many: its page says how much of the whole it holds in it.
    page_unit: ClassVar[tuple[str, str]] = ("caveat", "caveats")
    total: int = Field(description="The caveats the snapshot carries, stated and implied.")
    offset: int = Field(description="The first caveat this page holds.")
    next_offset: int | None = Field(description="The offset of the next page of caveats; None when this is the last.")


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
        keys = [key for dimension in report_input.judged for key in _judge_keys(dimension)]
        return _Touch(measures=_distinct([*report_input.judged, *caveats.judge_tiers]), keys=keys)
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


def _implied_touch(report_input: ReportInput, caveat: ImpliedCaveat) -> _Touch:
    """What an implied caveat touches, and the keys of the figures it rests on: a judge's stated figures, the
    candidates counted, a gate's threshold and each surviving option's clearance with no interval, or with one that
    reaches past the line."""
    keys: list[str] = []
    if caveat.kind == "judge_partly_uncalibrated":
        keys = [key for dimension in caveat.measures for key in _judge_keys(dimension)]
    elif caveat.kind in ("no_control", "best_of_n"):
        keys = ["n_candidates"]
    elif caveat.kind in ("point_estimate", "borderline"):
        criterion = caveat.id.partition(":")[2]
        keys = [f"criteria.{criterion}.threshold"]
        for arm, task, cell in caveat.at:
            place = task if cell is None else f"{report_input.cells[cell].row}.{report_input.cells[cell].column}"
            single = sum(held == arm for held, _task, _cell in caveat.at) == 1 and place is None
            keys.append(f"criteria.{criterion}.{arm}.gap" if single else f"criteria.{criterion}.{arm}.{place}.gap")
    return _Touch(arms=list(caveat.arms), measures=list(caveat.measures), keys=keys)


def caveats_list(report_input: ReportInput, *, limit: int | None = None, offset: int = 0) -> CaveatListing:
    """Every caveat the snapshot carries: where it may land, whether code proved it immaterial, and what it touches,
    each read from the input (a confound: the contrasts naming it; a short run: its arm; a late criterion: what it
    set; an unsettled aspect: its candidates). The caveats the input states come first, then the ones what it holds
    implies (:func:`~threetears.evals.analysis.report_decision.implied_caveats`), each with when it matters.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param limit: The most caveats one page holds; None for every one.
    :ptype limit: int | None
    :param offset: The first caveat the page holds.
    :ptype offset: int
    :return: The caveats, or one page of them.
    :rtype: CaveatListing
    """
    materiality = caveat_materiality(report_input)
    unsettled = unsettled_arms(report_input)
    every = every_caveat(report_input)
    shown, next_offset = _page(every, len(every) if limit is None else limit, offset)
    rows: list[CaveatRow] = []
    for caveat in shown:
        if isinstance(caveat, ImpliedCaveat):
            touch, held = _implied_touch(report_input, caveat), Materiality(caveat.id, False, None)
        else:
            touch, held = _touch(report_input, caveat, unsettled), materiality[caveat.id]
        implied = isinstance(caveat, ImpliedCaveat)
        rows.append(
            CaveatRow(
                id=caveat.id,
                kind=caveat.kind,
                reach=caveat.reach,
                source="implied" if implied else "stated",
                matters_if=caveat.matters_if if isinstance(caveat, ImpliedCaveat) else None,
                immaterial=held.immaterial,
                checked=held.checked,
                touches_arms=touch.arms,
                touches_aspects=touch.aspects,
                touches_measures=touch.measures,
                says=touch.says,
                figures=_figures(touch.keys, report_input),
            )
        )
    return CaveatListing(caveats=rows, total=len(every), offset=offset, next_offset=next_offset)
