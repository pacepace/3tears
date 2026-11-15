"""The one caveat list: every qualification of a report's answer raised once, typed, and the first screen read off it.

A caveat is a :class:`~threetears.evals.analysis.report_input.CaveatEntry`: a qualification of the answer with its
class, reach, state and reason, so the first screen is ranked from values and never from a sentence
(:func:`~threetears.evals.analysis.first_screen.first_screen`). Each source of caveats is one raiser here, a pure
function of the report input (and, where the builder holds it, the campaign's bundle); a part of the package that has
something to qualify adds a raiser and one row of :data:`CLASS_OF_KIND`, and renders no line of its own.

**The raisers, in the list's order.**

- *Stated* (:func:`from_legacy`): what the input's ``caveats`` part states, by the ids a report document names them
  by (``late:<criterion>``, ``judge_uncalibrated``, ``confound:<id>``, ``apparatus:<dimension>``, ``short_run:<run>``,
  ``unsettled:<aspect>``, ``latency``, ``unreported:<observation>``). How a stored input with no list is given one.
- *Implied* (:func:`implied`): what the input holds implies (``judge_unstated``, ``judge_partly_uncalibrated``,
  ``no_control``, ``nothing_declared``, ``point_estimate:<criterion>``, ``borderline:<criterion>``, ``best_of_n``).
- *Typed* (:func:`from_states`): every reading, contrast, pair and figure the input holds in a state other than
  counted, every arm whose identity is not one cell, and every check of a judge that does not speak for it.
- *The bundle's* (:func:`from_bundle`): what only the bundle holds: an arm read per answering model, an arm holding
  several cells of one rig, results with no ordered time, and request settings not confirmed as sent.

**Reach**, which decides where a caveat may land. *Report-wide*: every late criterion, the uncalibrated judge, and a
confound, apparatus confound or short run that touches every comparison the report makes (each candidate against the
control in a change, every pair in a choice, the one arm in a measurement; a short run touches the comparisons its arm
is in, every one when it names no arm; a confound touches the comparisons whose contrasts name it), and an unsettled
aspect whose unsettled candidates include the option code's decision names. *Scoped*: a confound or short run touching
only some comparisons, and an unsettled aspect on no option the decision names. *Method-level*: latency, an unreported
observation, and an apparatus confound that is not report-wide.

**Classes and remedies are data.** A kind's class (:data:`CLASS_OF_KIND`) says what it does to the answer; the rows
marked default are this module's, for kinds that existed before classes did. What would close a caveat is its
reason's remedy (:data:`REMEDY_OF_REASON`), else its kind's, and the step that takes it (:func:`next_step`) is read
off :data:`STEP_OF_REMEDY`: never stored, so a change to a remedy re-keys nothing.

Nothing here reads an eval's kind, a campaign's shape or a measure's name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, NamedTuple

from threetears.evals.analysis.answer import (
    asked_aspects,
    comparisons,
    control_arm,
    decision_at,
    judge_described,
    judged_dimensions,
    late_immaterial,
    stated_uncalibrated,
    surviving_options,
    unsettled_arms,
)
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.cells import NextExperiment, answer_split_variants
from threetears.evals.analysis.completeness import gate_rule
from threetears.evals.analysis.criteria import criterion_reads
from threetears.evals.analysis.first_screen import first_screen
from threetears.evals.analysis.report_input import (
    CallRead,
    CaveatEntry,
    FirstScreen,
    Gate,
    GateRun,
    LateCriterion,
    RemedyStep,
    Remedy,
    ReportInput,
)
from threetears.evals.analysis.report_margins import criterion_figures
from threetears.evals.analysis.stats import SMALL_N_BAND_FLOOR
from threetears.evals.kernel.declaration import CampaignCall
from threetears.evals.schema.reading_states import CaveatClass, CaveatReach, Coordinate, ReadingState, Reason

#: Nothing is declared yet: the list's readers are the report input's builder and the page's thin reads of it.
__all__: list[str] = []

#: The class of each caveat kind; None for a kind that only ever lands in the method section. A kind absent here (an
#: analyst's own) has no class and ranks after every class.
CLASS_OF_KIND: Final[Mapping[str, CaveatClass | None]] = MappingProxyType(
    {
        # A: the answer was changed. An entry of this class carries the state that changed it.
        "set_aside": "A",
        "withheld": "A",
        # B: the evidence is not what it appears.
        "late": "B",
        "confound": "B",
        "apparatus": "B",
        "short_run": "B",  # default
        "no_control": "B",  # default
        "nothing_declared": "B",  # default
        "untimed": "B",  # default: the code page's "no ordered time"
        "request_settings": "B",  # default: the code page's "request settings not confirmed"
        # C: an instrument on an asked reading is unchecked, or checked against a model only.
        "judge_uncalibrated": "C",
        "judge_partly_uncalibrated": "C",
        "judge_unstated": "C",
        "check": "C",
        # D: the test's level is loosened.
        "looks": "D",
        "best_of_n": "D",  # default: a pick among several is the same loosening
        # E: precision is missing on the deciding measure.
        "unsettled": "E",  # default
        "point_estimate": "E",  # default
        "borderline": "E",  # default
        # Method reach, never first-screen.
        "latency": None,
        "unreported": None,
        # A figure that cannot be computed is stated in its table cell, one step down.
        "not_available": None,
    }
)

#: The kinds the input's ``caveats`` part states (:func:`from_legacy`): a stored plan's flag names only these.
STATED_KINDS: Final = frozenset(
    {"late", "judge_uncalibrated", "confound", "apparatus", "short_run", "unsettled", "latency", "unreported"}
)

#: The kinds what the input holds implies (:func:`implied`).
IMPLIED_KINDS: Final = frozenset(
    {
        "judge_unstated",
        "judge_partly_uncalibrated",
        "no_control",
        "nothing_declared",
        "point_estimate",
        "borderline",
        "best_of_n",
    }
)

#: What would close a caveat, by its reason: every reason of
#: :data:`~threetears.evals.schema.reading_states.REASONS_OF` has exactly one. ``nothing_to_do`` is the answer itself:
#: every case read the same, so more of the same reads the same.
REMEDY_OF_REASON: Final[Mapping[str, Remedy]] = MappingProxyType(
    {
        "instrument_unchecked": "check_against_people",
        "instrument_below_bar": "check_against_people",
        "instrument_leans": "replace_instrument",
        "people_disagree": "replace_instrument",
        "identical_arms_separated": "repair_rig",
        "unfinished": "resume_or_rerun",
        "split": "rerun_on_one_cell",
        "identity_unrecorded": "remeasure_or_attest",
        "measured_under_concurrency": "rerun_serially",
        "trials_lost": "repair_rig",
        "too_few": "add_cases",
        "no_variation": "nothing_to_do",
        "not_recorded": "record_and_rerun",
        "replayed": "run_live",
        "not_drawn_at_random": "draw_at_random",
        "skipped_past_limit": "finish_read",
    }
)

#: What would close a caveat that carries no reason, by its kind.
_REMEDY_OF_KIND: Final[Mapping[str, Remedy]] = MappingProxyType(
    {
        "looks": "declare_looks",
        "judge_uncalibrated": "check_against_people",
        "judge_partly_uncalibrated": "check_against_people",
        "check": "check_against_people",
        "short_run": "resume_or_rerun",
        "unsettled": "add_cases",
        "point_estimate": "add_cases",
        "borderline": "add_cases",
        "best_of_n": "add_cases",
        "request_settings": "record_and_rerun",
    }
)


class StepRule(NamedTuple):
    """How a remedy is taken, as data.

    :ivar action: The catalogue action that takes it; None where a host or a person acts outside the engine.
    :ivar adds_evidence: Whether it is more evidence: it then states a planned size before it runs, and taking it is
        one more declared look at the readings it names.
    :ivar spends: Whether taking it costs money; it then names the action that prices it.
    """

    action: str | None
    adds_evidence: bool = False
    spends: bool = False


#: The action that prices every step that spends, before it is taken.
PRICED_BY: Final = "launch_estimate"

#: How each remedy is taken. A remedy whose action depends on the coordinate it is about takes it from
#: :data:`_ACTION_BY_COORDINATE`.
STEP_OF_REMEDY: Final[Mapping[Remedy, StepRule]] = MappingProxyType(
    {
        "resume_or_rerun": StepRule("job_resume", spends=True),
        "rerun_on_one_cell": StepRule("run_launch", spends=True),
        "remeasure_or_attest": StepRule(None),
        "rerun_serially": StepRule("run_launch", spends=True),
        "check_against_people": StepRule("result_rate_person"),
        "replace_instrument": StepRule(None),
        "repair_rig": StepRule(None),
        "add_cases": StepRule("run_launch", adds_evidence=True, spends=True),
        "record_and_rerun": StepRule("run_launch", spends=True),
        "run_live": StepRule("run_launch", spends=True),
        "draw_at_random": StepRule(None),
        "finish_read": StepRule(None),
        "declare_looks": StepRule("campaign_update"),
        "nothing_to_do": StepRule(None),
    }
)

#: The action that re-measures or attests a coordinate, where the catalogue offers one for it.
_ACTION_BY_COORDINATE: Final[Mapping[str, str]] = MappingProxyType({"measure_definition": "run_remeasure"})

#: What a gate's rule is, as a reason: what lifts the gate is that reason's remedy.
REASON_OF_GATE: Final[Mapping[str, Reason]] = MappingProxyType(
    {"setup": "trials_lost", "unfinished": "unfinished", "too_small": "too_few"}
)


# --- the stated caveats ----------------------------------------------------------------------------------------------


def late_id(late: LateCriterion) -> str:
    """A late criterion's caveat id: ``late:<criterion>``, or ``late:<criterion>:<name>`` for a named one.

    :param late: The late criterion.
    :ptype late: LateCriterion
    :return: Its id.
    :rtype: str
    """
    return f"late:{late.criterion}" if late.name is None else f"late:{late.criterion}:{late.name}"


def late_subject(report_input: ReportInput, late: LateCriterion) -> tuple[str | None, str | None]:
    """What a late criterion applies to: the aspect a floor is read on and the reading it names, or the measure a bar
    or a guardrail names.

    A floor carries its aspect and reading beside its key, so nothing is parsed out of the key, whose parts may hold a
    slash. An input built before a floor carried them names the floor's aspect by the aspect its reads are on, and no
    reading.

    :param report_input: The input the criterion is in, whose floors' reads name an older floor's aspect.
    :ptype report_input: ReportInput
    :param late: The criterion.
    :ptype late: LateCriterion
    :return: The aspect and the measure it names, each None where it names none; both None for an unnamed criterion.
    :rtype: tuple[str | None, str | None]
    """
    if not late.name:
        return None, None
    if late.criterion == "floor":
        if late.aspect is not None:
            return late.aspect, late.reading
        return next((read.aspect for read in report_input.floors if read.floor == late.name), None), None
    if late.criterion in ("bar", "guardrail"):
        return None, late.name
    return None, None


def _touches_every_comparison(report_input: ReportInput, arms: set[str] | None) -> bool:
    """Whether a caveat on ``arms`` (None: on every arm) touches every comparison the report makes."""
    if arms is None:
        return True
    made = comparisons(report_input)
    return bool(made) and all(comparison & arms for comparison in made)


def _confound_arms(report_input: ReportInput, dimension: str) -> set[str]:
    return {contrast.arm for contrast in report_input.contrasts if dimension in contrast.confounds}


def _asked(report_input: ReportInput, readings: Sequence[str]) -> bool:
    """Whether a caveat on ``readings`` touches an asked aspect: one of them is on one, or it names none (the whole
    report) and the report asks about one."""
    asked = {aspect.id for aspect in asked_aspects(report_input)}
    if not readings:
        return bool(asked)
    return any(reading.partition("/")[0] in asked for reading in readings)


def _entry(report_input: ReportInput, id: str, kind: str, reach: CaveatReach, **facts: Any) -> CaveatEntry:
    """One entry, its class its kind's and whether it touches an asked aspect read off its readings."""
    cls = CLASS_OF_KIND.get(kind)
    readings: list[str] = list(facts.pop("readings", []))
    deciding = facts.pop("deciding", cls == "E" and reach == "report")
    return CaveatEntry.model_validate(
        {
            "id": id,
            "kind": kind,
            "class": cls,
            "reach": reach,
            "readings": readings,
            "asked": _asked(report_input, readings),
            "deciding": deciding,
            **facts,
        }
    )


def _judge_uncalibrated(report_input: ReportInput) -> bool:
    """Whether every judged dimension's judge is stated uncalibrated, from the input's stated tiers: the stored
    caveat, read the way a stored input states it until a judge's checks are on the input."""
    dimensions = judged_dimensions(report_input)
    return bool(dimensions) and all(stated_uncalibrated(report_input, dimension) for dimension in dimensions)


def from_legacy(report_input: ReportInput) -> list[CaveatEntry]:
    """The caveats the input's ``caveats`` part states, each once, by the id a report document names it by, with its
    kind's class and where it may land (this module's Reach).

    What a stored input with no list is given on read, and the first part of every list: nothing is written back.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Late criteria, the judge, confounds, apparatus confounds, short runs, unsettled aspects, latency, then
        unreported observations, each in the input's order.
    :rtype: list[CaveatEntry]
    """
    caveats = report_input.caveats
    found: dict[str, CaveatEntry] = {}

    def add(entry: CaveatEntry) -> None:
        found.setdefault(entry.id, entry)

    for late in caveats.late_criteria:
        aspect, measure = late_subject(report_input, late)
        readings = [f"{aspect}/{measure}"] if aspect is not None and measure is not None else []
        immaterial = True if late_immaterial(report_input, late) else None
        add(_entry(report_input, late_id(late), "late", "report", readings=readings, immaterial=immaterial))
    if _judge_uncalibrated(report_input):
        add(_entry(report_input, "judge_uncalibrated", "judge_uncalibrated", "report"))
    apparatus = [confound.dimension for confound in caveats.apparatus_confounds]
    for dimension in caveats.confounds:
        if dimension not in apparatus:
            arms = _confound_arms(report_input, dimension)
            reach: CaveatReach = "report" if _touches_every_comparison(report_input, arms) else "scoped"
            add(_entry(report_input, f"confound:{dimension}", "confound", reach, arms=sorted(arms)))
    for dimension in apparatus:
        arms = _confound_arms(report_input, dimension)
        reach = "report" if _touches_every_comparison(report_input, arms) else "method"
        add(_entry(report_input, f"apparatus:{dimension}", "apparatus", reach, arms=sorted(arms)))
    for run in caveats.short_runs:
        wide = _touches_every_comparison(report_input, None if run.arm is None else {run.arm})
        arms_of = [] if run.arm is None else [run.arm]
        add(
            _entry(
                report_input,
                f"short_run:{run.run}",
                "short_run",
                "report" if wide else "scoped",
                arms=arms_of,
                runs=[run.run],
            )
        )
    unsettled = unsettled_arms(report_input)
    if unsettled:
        called = set(decision_at(report_input).dominant)
        for caveat_id, arms in unsettled.items():
            aspect_id = caveat_id.partition(":")[2]
            add(
                _entry(
                    report_input,
                    caveat_id,
                    "unsettled",
                    "report" if called & set(arms) else "scoped",
                    arms=list(arms),
                    readings=[f"{aspect_id}/passed"],
                )
            )
    if caveats.latency:
        add(
            _entry(
                report_input,
                "latency",
                "latency",
                "method",
                state="withheld",
                reason="measured_under_concurrency",
            )
        )
    for observation in caveats.unreported:
        add(_entry(report_input, f"unreported:{observation}", "unreported", "method"))
    return list(found.values())


# --- the implied caveats ---------------------------------------------------------------------------------------------

#: How many candidates make a pick the best of many (``best_of_n``): from three, the best of the candidates is a
#: selection among several, whose winner reads high by chance.
_MANY_OPTIONS = 3

#: The criterion kinds that decide which options survive or are shown no worse: a bar only warns.
_DECIDING = frozenset({"gate", "guardrail", "floor"})

#: The criterion kinds that gate an option.
_GATING = frozenset({"gate", "guardrail"})

#: A place a criterion is read on an arm: the arm, and the task or cell id where the criterion reads one there.
Place = tuple[str, str | None, str | None]


def _deciding_reach(report_input: ReportInput, measures: Sequence[str]) -> CaveatReach:
    """``report`` where a deciding criterion or the primary judged dimension reads one of the measures, else
    ``scoped``."""
    read = {criterion.measure for criterion in report_input.criteria if criterion.kind in _DECIDING}
    read.add(report_input.primary_judged or "")
    return "report" if read & set(measures) else "scoped"


def _clears(threshold: float, direction: str, value: float) -> bool:
    return value >= threshold if direction == "higher_is_better" else value <= threshold


def unstated_judges(report_input: ReportInput) -> tuple[str, ...]:
    """The judged dimensions whose judge the input does not describe: no model, no tier, no calibration state.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The dimensions, by key, in the input's order.
    :rtype: tuple[str, ...]
    """
    return tuple(
        dimension for dimension in judged_dimensions(report_input) if not judge_described(report_input, dimension)
    )


def uncalibrated_judges(report_input: ReportInput) -> tuple[str, ...]:
    """The judged dimensions whose judge the input states uncalibrated.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The dimensions, by key, in the input's order.
    :rtype: tuple[str, ...]
    """
    return tuple(
        dimension for dimension in judged_dimensions(report_input) if stated_uncalibrated(report_input, dimension)
    )


def cleared_on_point_estimate(report_input: ReportInput, criterion_id: str) -> tuple[Place, ...]:
    """Each place a surviving option clears a gate or guardrail on a figure that holds no interval there: none
    stated, and none derived from its cells.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion_id: The gate or guardrail, by id.
    :ptype criterion_id: str
    :return: Each ``(arm, task, cell)``, in the survivors' and the criterion's places' order.
    :rtype: tuple[Place, ...]
    """
    criterion = next(held for held in report_input.criteria if held.id == criterion_id)
    return tuple(
        (arm, figure.task, figure.cell)
        for arm in surviving_options(report_input)
        for figure in criterion_figures(report_input, criterion, arm)
        if figure.ci is None
        and figure.half_width is None
        and _clears(float(criterion.threshold), criterion.direction, figure.value)
    )


def cleared_borderline(report_input: ReportInput, criterion_id: str) -> tuple[Place, ...]:
    """Each place a surviving option clears a gate or guardrail on its point estimate while the figure's interval
    there reaches the wrong side of the line: the gate's borderline, which the decision counts as passing.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param criterion_id: The gate or guardrail, by id.
    :ptype criterion_id: str
    :return: Each ``(arm, task, cell)``, in the survivors' and the criterion's places' order.
    :rtype: tuple[Place, ...]
    """
    return tuple(
        (arm, read.task, read.cell)
        for arm in surviving_options(report_input)
        for read in criterion_reads(report_input, arm, criterion=criterion_id)
        if read.status == "borderline"
    )


def _cleared(
    report_input: ReportInput, kind: str, places: Callable[[ReportInput, str], tuple[Place, ...]]
) -> Iterator[CaveatEntry]:
    for criterion in report_input.criteria:
        if criterion.kind not in _GATING:
            continue
        at = places(report_input, criterion.id)
        if at:
            arms = list(dict.fromkeys(arm for arm, _task, _cell in at))
            yield _entry(report_input, f"{kind}:{criterion.id}", kind, "report", arms=arms, deciding=True)


def implied(report_input: ReportInput, stated: Iterable[CaveatEntry] = ()) -> list[CaveatEntry]:
    """The caveats what the report input holds implies, beyond the ones it states.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param stated: The caveats it states (:func:`from_legacy`), which one implied caveat defers to.
    :ptype stated: Iterable[CaveatEntry]
    :return: The judge's, then no control, nothing declared, each gate cleared on a point estimate, each gate cleared
        on a borderline figure, then a pick among many; each once, in that order.
    :rtype: list[CaveatEntry]
    """
    found: list[CaveatEntry] = []
    unstated = unstated_judges(report_input)
    if unstated:
        reach = _deciding_reach(report_input, unstated)
        found.append(_entry(report_input, "judge_unstated", "judge_unstated", reach))
    uncalibrated = uncalibrated_judges(report_input)
    if uncalibrated and "judge_uncalibrated" not in {caveat.id for caveat in stated}:
        reach = _deciding_reach(report_input, uncalibrated)
        found.append(_entry(report_input, "judge_partly_uncalibrated", "judge_partly_uncalibrated", reach))
    if len(report_input.arms) > 1 and control_arm(report_input) is None:
        found.append(_entry(report_input, "no_control", "no_control", "report"))
    margins = any(
        aspect.margin is not None or any(reading.margin is not None for reading in aspect.readings)
        for aspect in report_input.aspects
    )
    deciding = any(criterion.kind in _DECIDING for criterion in report_input.criteria)
    # A question of a control is a declaration: asked with no margin it is a superiority question, shown better than
    # the control its bar. A choice has no control to be shown better than, so its question declares no bar.
    superiority = bool(asked_aspects(report_input)) and control_arm(report_input) is not None
    if not deciding and not margins and not report_input.floors and not superiority:
        found.append(_entry(report_input, "nothing_declared", "nothing_declared", "report"))
    found += _cleared(report_input, "point_estimate", cleared_on_point_estimate)
    found += _cleared(report_input, "borderline", cleared_borderline)
    candidates = [arm.id for arm in report_input.arms if arm.role == "challenger"]
    if len(candidates) >= _MANY_OPTIONS:
        found.append(_entry(report_input, "best_of_n", "best_of_n", "report", arms=candidates))
    return found


# --- the typed states ------------------------------------------------------------------------------------------------


class _Fact(NamedTuple):
    """One holder the input or the bundle holds in a state other than counted."""

    state: ReadingState
    reason: Reason | None
    coordinate: Coordinate | None
    reading: str
    arm: str | None
    cases: int | None = None


#: What the trial reading of an aspect is addressed by: ``<aspect>/passed``.
_PASSED = "passed"


def _input_facts(report_input: ReportInput) -> Iterator[_Fact]:
    """Every reading, pair, contrast and figure the input holds in a state, and every arm not one cell."""
    lead = report_input.aspects[0].id if report_input.aspects else ""
    for aspect in report_input.aspects:
        for reading in aspect.readings:
            if reading.state is not None and reading.state.state != "counted":
                held = reading.state
                yield _Fact(held.state, held.reason, held.coordinate, f"{aspect.id}/{_PASSED}", reading.arm)
        for pair in aspect.pairs:
            if pair.state is not None and pair.state.state != "counted":
                for arm in pair.arms:
                    held = pair.state
                    yield _Fact(held.state, held.reason, held.coordinate, f"{aspect.id}/{_PASSED}", arm)
    for contrast in report_input.contrasts:
        if contrast.state is not None and contrast.state.state != "counted":
            held = contrast.state
            at = f"{contrast.aspect or lead}/{contrast.measure}"
            yield _Fact(held.state, held.reason, held.coordinate, at, contrast.arm, contrast.n_cases)
    for key, measure in report_input.measures.items():
        figures = [(lead, measure.per_arm), *measure.by_aspect.items()]
        for aspect_id, per_arm in figures:
            for arm, stat in per_arm.items():
                if stat.state is not None and stat.state.state != "counted":
                    held = stat.state
                    yield _Fact(held.state, held.reason, held.coordinate, f"{aspect_id}/{key}", arm, stat.n_cases)
    for arm in report_input.arms:
        identity = arm.identity
        if identity is None:
            continue
        coordinates: list[tuple[Coordinate, str | None, str]] = [
            ("served_model", identity.served_model, f"{lead}/{_PASSED}"),
            ("template", identity.template, f"{lead}/{_PASSED}"),
            ("rig", identity.rig, f"{lead}/{_PASSED}"),
            *(
                ("measure_definition", held, f"{aspect_id}/{key}")
                for key, held in identity.measure_definition.items()
                for aspect_id in _measured_on(report_input, key, arm.id, lead)
            ),
        ]
        for coordinate, state, reading in coordinates:
            if state == "split":
                yield _Fact("withheld", "split", coordinate, reading, arm.id)
            elif state == "unrecorded":
                yield _Fact("withheld", "identity_unrecorded", coordinate, reading, arm.id)


def _measured_on(report_input: ReportInput, key: str, arm: str, lead: str) -> list[str]:
    """The aspects one measure holds an arm's figure on, by id: every reading of it the arm has; the lead aspect where
    the input holds none."""
    measure = report_input.measures.get(key)
    if measure is None:
        return [lead]
    held = [lead] if arm in measure.per_arm else []
    held += [aspect_id for aspect_id, arms in measure.by_aspect.items() if arm in arms]
    return held or [lead]


def _from_facts(report_input: ReportInput, facts: Iterable[_Fact]) -> list[CaveatEntry]:
    """One caveat per state and reading, its arms every arm the fact holds for: ``<state>:<reading>``, and
    ``<state>:<reading>:<reason>`` for each further reason one reading is in that state for."""
    grouped: dict[tuple[ReadingState, str, Reason | None, Coordinate | None], list[_Fact]] = {}
    for fact in facts:
        grouped.setdefault((fact.state, fact.reading, fact.reason, fact.coordinate), []).append(fact)
    asked = {aspect.id for aspect in asked_aspects(report_input)}
    seen: set[str] = set()
    found: list[CaveatEntry] = []
    for (state, reading, reason, coordinate), held in grouped.items():
        caveat_id = f"{state}:{reading}"
        if caveat_id in seen:
            caveat_id = f"{caveat_id}:{reason}"
        seen.add(caveat_id)
        on_asked = not asked or reading.partition("/")[0] in asked
        counts = {fact.cases for fact in held if fact.cases is not None}
        found.append(
            _entry(
                report_input,
                caveat_id,
                state,
                "report" if on_asked and state != "not_available" else "scoped",
                state=state,
                reason=reason,
                coordinate=coordinate,
                arms=sorted({fact.arm for fact in held if fact.arm is not None}),
                readings=[reading],
                cases=min(counts) if counts else None,
            )
        )
    return found


def _checks(report_input: ReportInput) -> Iterator[CaveatEntry]:
    """A caveat per check of a judge that does not speak for it: one against a model, or one whose result is not
    met. A check against people that is met is said one step down, where it can still silence a check against a
    model on its criterion."""
    for dimension, judged in report_input.judged.items():
        for check in judged.checks:
            if check.against not in ("people", "model"):
                continue
            met = check.result == "met"
            if check.against == "people" and met:
                reach: CaveatReach = "method"
            else:
                reach = _deciding_reach(report_input, [dimension])
            yield _entry(
                report_input,
                f"check:{dimension}:{check.criterion or check.instrument}:{check.against}",
                "check",
                reach,
                against=check.against,
                criterion=check.criterion or check.instrument,
                check=check.result,
                same_family=bool(check.same_family_as),
                reason=check.reason,
                coordinate=check.coordinate if check.reason == "identity_unrecorded" else None,
            )


def _looks(report_input: ReportInput) -> Iterator[CaveatEntry]:
    """A caveat per aspect whose test was looked at more often than it planned: more than once with no plan, or more
    often than the plan said. Looks within the plan qualify nothing and raise none."""
    for aspect in report_input.aspects:
        looks = aspect.looks
        if looks is None or looks.taken <= (looks.planned or 1):
            continue
        reach: CaveatReach = "report" if aspect.asked or not asked_aspects(report_input) else "scoped"
        yield _entry(report_input, f"looks:{aspect.id}", "looks", reach, readings=[f"{aspect.id}/{_PASSED}"])


def from_states(report_input: ReportInput) -> list[CaveatEntry]:
    """A caveat per reading, contrast, pair and figure in a state other than counted, per arm not one cell, per check
    of a judge that does not speak for it, and per aspect looked at more often than it planned.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The caveats, each holder's in the input's order, then the checks', then the looks'.
    :rtype: list[CaveatEntry]
    """
    return [*_from_facts(report_input, _input_facts(report_input)), *_checks(report_input), *_looks(report_input)]


def _bundle_facts(bundle: CampaignBundle) -> Iterator[_Fact]:
    """What only the bundle holds about an arm not being one cell: an arm read per answering model on an aspect, and
    one holding several cells of one rig there, the second unrecorded where recording a dimension would pool them."""
    for aspect in bundle.aspects:
        answered = answer_split_variants(aspect.cells)
        for variant in sorted(answered):
            yield _Fact("withheld", "split", "served_model", f"{aspect.aspect}/{_PASSED}", bundle.label_of(variant))
        rigs: dict[str, set[str]] = {}
        for cell in aspect.cells:
            if cell.variant_key not in answered:
                rigs.setdefault(cell.variant_key, set()).add(cell.apparatus_class_id)
        unrecorded = {experiment.variant_key for experiment in aspect.next_experiments}
        for variant, held in rigs.items():
            if len(held) > 1:
                reason: Reason = "identity_unrecorded" if variant in unrecorded else "split"
                yield _Fact("withheld", reason, "rig", f"{aspect.aspect}/{_PASSED}", bundle.label_of(variant))


def from_bundle(bundle: CampaignBundle, report_input: ReportInput) -> list[CaveatEntry]:
    """The caveats only the campaign's bundle holds: an arm read per answering model, an arm holding several cells of
    one rig, results with no ordered time, and request settings not confirmed as sent.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :param report_input: The input built from it, which the caveats are scoped by.
    :ptype report_input: ReportInput
    :return: The caveats.
    :rtype: list[CaveatEntry]
    """
    found = _from_facts(report_input, _bundle_facts(bundle))
    if untimed := [aspect.aspect for aspect in bundle.aspects if aspect.reading_scope.untimed_results]:
        readings = [f"{aspect}/{_PASSED}" for aspect in untimed]
        found.append(_entry(report_input, "untimed", "untimed", "report", readings=readings))
    if unconfirmed := [aspect.aspect for aspect in bundle.aspects if aspect.request_settings_disclosure]:
        readings = [f"{aspect}/{_PASSED}" for aspect in unconfirmed]
        found.append(_entry(report_input, "request_settings", "request_settings", "report", readings=readings))
    return found


def raise_caveats(report_input: ReportInput, bundle: CampaignBundle | None = None) -> list[CaveatEntry]:
    """Every caveat on a report input, each once: stated, implied, typed, then the bundle's (this module's raisers).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param bundle: The bundle it was built from, where the builder holds it; None for a stored input.
    :ptype bundle: CampaignBundle | None
    :return: The list.
    :rtype: list[CaveatEntry]
    """
    stated = from_legacy(report_input)
    raised = [
        *stated,
        *implied(report_input, stated),
        *from_states(report_input),
        *(from_bundle(bundle, report_input) if bundle is not None else ()),
    ]
    found: dict[str, CaveatEntry] = {}
    for entry in raised:
        found.setdefault(entry.id, entry)
    return list(found.values())


def caveat_list(report_input: ReportInput) -> list[CaveatEntry]:
    """The input's one caveat list: the one it carries, else the one its stored parts give on read.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The list.
    :rtype: list[CaveatEntry]
    """
    if report_input.caveat_list is not None:
        return list(report_input.caveat_list)
    return raise_caveats(report_input)


# --- the gate and the call -------------------------------------------------------------------------------------------


def gate_of(bundle: CampaignBundle) -> Gate | None:
    """What stops the campaign being read at all: the first rule of the verdict chain that holds on an asked aspect
    (every aspect where none is asked), with the runs it names; None when the campaign can be read.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :return: The gate, or None.
    :rtype: Gate | None
    """
    asked = [aspect for aspect in bundle.aspects if aspect.question_declared] or list(bundle.aspects)
    rules = [(aspect, rule) for aspect in asked if (rule := gate_rule(aspect)) is not None]
    if not rules:
        return None
    order = list(REASON_OF_GATE)
    rule = min((held for _aspect, held in rules), key=order.index)
    runs: dict[str, GateRun] = {}
    if rule == "unfinished":
        for aspect, held in rules:
            if held != rule:
                continue
            ended = {summary.run_id: summary.ended_because for summary in aspect.run_summaries}
            for run_id, status in aspect.incomplete_runs.items():
                runs.setdefault(run_id, GateRun(run=run_id, status=status, ended_because=ended.get(run_id)))
    return Gate(rule=rule, runs=list(runs.values()))


def call_read(call: CampaignCall, bundle: CampaignBundle) -> CallRead:
    """A recorded call as the first screen reads it: the arm, who made it, and what has moved since.

    :param call: The call.
    :ptype call: CampaignCall
    :param bundle: The campaign's bundle as it stands now.
    :ptype bundle: CampaignBundle
    :return: The call's read.
    :rtype: CallRead
    """
    joined = len(set(bundle.run_ids) - set(call.run_ids)) if call.run_ids else None
    changed = None if call.evidence_fingerprint is None else call.evidence_fingerprint != bundle.fingerprint()
    return CallRead(
        arm=bundle.label_of(call.variant_key),
        made_by=call.made_by,
        model=call.model,
        on_behalf_of=call.on_behalf_of,
        runs_joined_since=joined,
        identity_changed=changed,
    )


# --- what would change it --------------------------------------------------------------------------------------------


def remedy(caveat: CaveatEntry) -> Remedy | None:
    """The one act that would close ``caveat``: its reason's, else its kind's.

    :param caveat: The caveat.
    :ptype caveat: CaveatEntry
    :return: The act, or None where neither its reason nor its kind names one.
    :rtype: Remedy | None
    """
    if caveat.reason is not None:
        return REMEDY_OF_REASON[caveat.reason]
    return _REMEDY_OF_KIND.get(caveat.kind)


def planned_cases(cases: int | None) -> int:
    """The size of a step that adds cases, planned before it runs: as many cases again as the reading rests on, and at
    least enough to reach the floor where an interval is first stated. Never "until it separates": the size is fixed
    here, and taking the step is one more declared look.

    :param cases: The cases the reading rests on now; None where no count says it.
    :ptype cases: int | None
    :return: The cases to add.
    :rtype: int
    """
    held = cases or 0
    return max(held, SMALL_N_BAND_FLOOR - held, 1)


def next_step(
    remedied: Remedy,
    *,
    closes: str | None = None,
    arms: Sequence[str] = (),
    runs: Sequence[str] = (),
    coordinate: Coordinate | None = None,
    cases: int | None = None,
    results_needed: int | None = None,
    recording: Sequence[NextExperiment] = (),
) -> RemedyStep:
    """The step that takes one remedy, filled from what the caveat (or the gate) names (:data:`STEP_OF_REMEDY`).

    :param remedied: The remedy.
    :ptype remedied: Remedy
    :param closes: The caveat it closes, by id; None for the gate's.
    :ptype closes: str | None
    :param arms: The arms the caveat touches.
    :ptype arms: Sequence[str]
    :param runs: The runs it names.
    :ptype runs: Sequence[str]
    :param coordinate: The coordinate it is about.
    :ptype coordinate: Coordinate | None
    :param cases: The cases its readings rest on.
    :ptype cases: int | None
    :param results_needed: For a check: the results it needs before it decides.
    :ptype results_needed: int | None
    :param recording: What recording an unrecorded coordinate would pool.
    :ptype recording: Sequence[NextExperiment]
    :return: The step.
    :rtype: RemedyStep
    """
    rule = STEP_OF_REMEDY[remedied]
    action = _ACTION_BY_COORDINATE.get(coordinate or "") if remedied == "remeasure_or_attest" else rule.action
    arguments: dict[str, str | list[str]] = {}
    if action is not None:
        if arms:
            arguments["arms"] = list(arms)
        if runs:
            arguments["runs"] = list(runs)
        if coordinate is not None:
            arguments["coordinate"] = coordinate
    return RemedyStep(
        remedy=remedied,
        closes=closes,
        action=action,
        arguments=arguments,
        planned_cases=planned_cases(cases) if rule.adds_evidence else None,
        declared_look=rule.adds_evidence,
        results_needed=results_needed if remedied == "check_against_people" else None,
        recording=list(recording),
        spends=rule.spends,
        priced_by=PRICED_BY if rule.spends else None,
    )


def caveat_step(
    caveat: CaveatEntry,
    *,
    results_needed: int | None = None,
    recording: Sequence[NextExperiment] = (),
) -> RemedyStep | None:
    """The step that would close one caveat; None where nothing names one.

    :param caveat: The caveat.
    :ptype caveat: CaveatEntry
    :param results_needed: For a check against people: the results it needs, where the bundle says.
    :ptype results_needed: int | None
    :param recording: What recording its coordinate would pool, where the bundle says.
    :ptype recording: Sequence[NextExperiment]
    :return: The step, or None.
    :rtype: RemedyStep | None
    """
    remedied = remedy(caveat)
    if remedied is None:
        return None
    return next_step(
        remedied,
        closes=caveat.id,
        arms=caveat.arms,
        runs=caveat.runs,
        coordinate=caveat.coordinate,
        cases=caveat.cases,
        results_needed=results_needed,
        recording=recording if caveat.reason == "identity_unrecorded" else (),
    )


def gate_step(gate: Gate) -> RemedyStep:
    """The step that would lift a gate: its rule's reason's remedy, on the runs it names.

    :param gate: The gate.
    :ptype gate: Gate
    :return: The step.
    :rtype: RemedyStep
    """
    return next_step(REMEDY_OF_REASON[REASON_OF_GATE[gate.rule]], runs=[run.run for run in gate.runs])


class Evidence(NamedTuple):
    """What the bundle says about how far a step would go, where the builder holds the bundle.

    :ivar results_needed: The most results any judge's check against people needs before it decides.
    :ivar recording: What recording an unrecorded coordinate would pool, by arm.
    """

    results_needed: int | None = None
    recording: Mapping[str, tuple[NextExperiment, ...]] = MappingProxyType({})


def evidence_of(bundle: CampaignBundle) -> Evidence:
    """What a campaign's bundle says about how far each step would go.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :return: The results its judges' checks need, and what recording would pool, by arm.
    :rtype: Evidence
    """
    needed = [
        held
        for aspect in bundle.aspects
        for tier in aspect.judge_evidence_tiers
        if (held := tier.calibration.results_needed) is not None
    ]
    recording: dict[str, list[NextExperiment]] = {}
    for aspect in bundle.aspects:
        for experiment in aspect.next_experiments:
            recording.setdefault(bundle.label_of(experiment.variant_key), []).append(experiment)
    return Evidence(max(needed) if needed else None, {arm: tuple(held) for arm, held in recording.items()})


def screen(
    report_input: ReportInput,
    caveats: Sequence[CaveatEntry],
    *,
    gate: Gate | None = None,
    call: CallRead | None = None,
    evidence: Evidence = Evidence(),
) -> FirstScreen:
    """The first screen of one input: its caveats ranked and capped (:func:`~threetears.evals.analysis.first_screen.
    first_screen`), the arms code's own read calls, and the step that would change each reason shown (under a gate,
    the step that lifts it).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param caveats: Its caveat list.
    :ptype caveats: Sequence[CaveatEntry]
    :param gate: What stops the campaign being read, or None.
    :ptype gate: Gate | None
    :param call: The call, where one was made.
    :ptype call: CallRead | None
    :param evidence: What the bundle says about how far a step would go; none for a stored input.
    :ptype evidence: Evidence
    :return: The first screen's value.
    :rtype: FirstScreen
    """
    value = first_screen(caveats, gate=gate, called_arms=decision_at(report_input).dominant, call=call)
    if value.gate is not None:
        return value.model_copy(update={"steps": [gate_step(value.gate)]})
    by_id = {caveat.id: caveat for caveat in caveats}
    steps = []
    for reason in value.reasons:
        caveat = by_id[reason]
        recording = [held for arm in caveat.arms for held in evidence.recording.get(arm, ())]
        step = caveat_step(caveat, results_needed=evidence.results_needed, recording=recording)
        if step is not None:
            steps.append(step)
    return value.model_copy(update={"steps": steps})


def first_screen_of(report_input: ReportInput) -> FirstScreen:
    """The first screen an input shows: the one it carries, else the one its caveat list gives on read.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The value.
    :rtype: FirstScreen
    """
    if report_input.first_screen is not None:
        return report_input.first_screen
    return screen(report_input, caveat_list(report_input))
