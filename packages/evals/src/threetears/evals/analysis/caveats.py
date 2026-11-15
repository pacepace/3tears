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
  counted, every arm whose identity is not one cell, and, where the builder holds the bundle, what only the bundle
  holds of an arm not being one cell (an arm read per answering model, an arm holding several cells of one rig), all
  grouped once; then every check of a judge that does not speak for it, and looks past their plan.
- *The bundle's* (:func:`from_bundle`): results with no ordered time, and request settings not confirmed as sent.

**Ids are unique by construction**, and :func:`raise_caveats` refuses two different entries under one id rather than
keep either: a typed caveat is ``<state>:<reading>`` where it is the only one on that reading in that state, else it
adds its reason, and its coordinate where two share a reason; a check that repeats its dimension, criterion and what
it was checked against adds its ordinal.

**Reach**, which decides where a caveat may land. *Report-wide*: every late criterion, the uncalibrated judge, and a
confound, apparatus confound or short run that touches every comparison the report makes (each candidate against the
control in a change, every pair in a choice, the one arm in a measurement; a short run touches the comparisons its arm
is in, every one when it names no arm; a confound touches the comparisons whose contrasts name it), and an unsettled
aspect whose unsettled candidates include the option code's decision names. *Scoped*: a confound or short run touching
only some comparisons, and an unsettled aspect on no option the decision names. *Method-level*: latency, an unreported
observation, and an apparatus confound that is not report-wide.

**Classes and remedies are data.** A kind's class (:data:`CLASS_OF_KIND`) says what it does to the answer; the rows
marked default are this module's, for kinds that existed before classes did. What would close a caveat is its
reason's remedy (:data:`REMEDY_OF_REASON`), else its kind's, chosen by what addresses it and never by what reads the
same evidence again; a caveat nothing closes takes ``no_action`` with why. The step that takes it
(:func:`next_step`) is read off :data:`STEP_OF_REMEDY`: never stored in a digest, so a change to a remedy re-keys
nothing. Its calls are named and typed as the catalogue action's own parameters, and what it cannot fill it names.
A step that launches runs is the whole act the catalogue's actions take, in order: a launch repeating the set-up of
the arms it runs (what the bundle holds of it filled, every other setting that defines an arm named, to be set as the
arms' runs were launched), then its runs joined to the campaign that reads them, since a launched run joins none, and
for a re-run the runs it replaces taken out; a confirmation on unseen cases creates its campaign first, with its
criteria, launches, joins the runs and names the control from them as soon as they start, before any result is
scored. A step ships a sequence only where following it through the real actions on a fixture campaign has been shown
to do what it says (``test_caveat_steps_driven.py``): more cases on arms that differ only in their model, a re-run that
holds an apparatus confound's dimension alike, and a confirmation. What following one cannot guarantee is stated,
typed (``RemedyStep.limits``): an arm launched in a call of its own pairs with no new case of the control's, so a
step whose caveat reads a contrast against the control launches no such arm, and makes no call where no arm it names
is left beside the control, while a confirmation launches it and names it (a gate or a figure reads each arm's own
cases, and names none); the reading holds no template's axes, so the cases are new only where the launch generated
them; a confirmation leaves this report's caveat raised and is read in the campaign it creates; and its control reads
late if named after the first result. A re-run with no sequence driven to lift it makes no call: one whose fault no
launch parameter sets (an answering model the provider split, a rig, recording, a replay, a confound no setting holds
alike), and the serial re-run of cases run at once, which a launch can ask for (``measure_latency``) but no sequence
here has followed through.

**A step that is more evidence is planned** (:func:`plan_of`) against the rule that lifts its caveat, on the
interval that rule reads: a gate's borderline is lifted when the figure's two-sided interval, at its table's stated
level, clears the line (:mod:`~threetears.evals.analysis.criteria`); an unsettled reading when its nominal interval
(``within_interval``, two-sided at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`) clears the margin's line
(:func:`~threetears.evals.analysis.answer.unsettled_arms`). Its size is the cases a design needs for that two-sided
interval, centred on today's estimate, to clear that line four times in five
(:func:`~threetears.evals.analysis.separation.cases_to_separate`, ``sided=2``, at
:data:`~threetears.evals.analysis.separation.POWER`), less the cases it rests on; on fresh cases, the whole design. A
rate's nominal interval is the bounded one, wider than its cases' own spread at a small count, so a size read off it
clears more often than four in five: a spread read ``k`` times too wide asks for about ``k**2`` times the cases, and
clears about the normal share below ``z * k`` of designs (``z`` the quantile of ``POWER``) rather than ``POWER``; the
seeded simulation in ``test_caveat_list.py`` holds both bounds. A figure that reads too few cases for an interval is
sized to the floor where one is first stated. A step that adds cases passes its size as the count of new cases its
launch generates. Where no size can be read, the step says why and makes no
call, so it is never taken open-ended; where no added case can close the caveat (a percentile's point estimate),
the step says that instead of offering an act that cannot work. A criterion set or a pick made on these results is
closed by no case added to this campaign, so its step is the confirmation in a new one.
Taking a step that adds cases is one more look at the readings it names, and says so; nothing counts looks yet, so a
reading after it is read at its full level as if it were the first.

Nothing here reads an eval's kind, a campaign's shape or a measure's name.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, Literal, NamedTuple

from threetears.evals.analysis.answer import (
    asked_aspects,
    comparisons,
    control_arm,
    decision_at,
    dominant_options,
    judge_described,
    judged_dimensions,
    late_immaterial,
    stated_uncalibrated,
    surviving_options,
    unsettled_arms,
)
from threetears.evals.analysis.checks import JudgeCheck
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.cells import NextExperiment, answer_split_variants
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.completeness import GateRule, gate_rule
from threetears.evals.analysis.criteria import criterion_reads
from threetears.evals.analysis.first_screen import first_screen
from threetears.evals.analysis.report_input import (
    Aspect,
    CallRead,
    CaveatEntry,
    Criterion,
    FirstScreen,
    Gate,
    GateRun,
    LateCriterion,
    NoActionReason,
    Reading,
    RemedyStep,
    Remedy,
    ReportInput,
    StepArgument,
    UnsizedReason,
)
from threetears.evals.analysis.report_margins import (
    CriterionFigure,
    aspect_board,
    criterion_figures,
    criterion_other_aspect,
)
from threetears.evals.analysis.separation import (
    POWER,
    NoSpread,
    Spread,
    cases_to_separate,
    difference_spread,
    level_share,
    stat_spread,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL, SMALL_N_BAND_FLOOR
from threetears.evals.kernel.declaration import CampaignCall
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER
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

#: What would close a caveat, by its reason: every reason of
#: :data:`~threetears.evals.schema.reading_states.REASONS_OF` has exactly one, the act that addresses the reason.
#: An instrument below its bar was checked and missed: checking it again is one more look at the same instrument, so
#: it is fixed or replaced. People who disagree with each other say the criterion is unclear, which a new judge would
#: read just as unclearly. Every case reading alike is no proof that none differs (twenty passes of twenty still allow
#: a failure rate near fourteen in a hundred), and only more cases narrow that; at today's estimate no added case is
#: expected to differ, so that step states no size and makes no call.
REMEDY_OF_REASON: Final[Mapping[str, Remedy]] = MappingProxyType(
    {
        "instrument_unchecked": "check_against_people",
        "instrument_below_bar": "replace_instrument",
        "instrument_leans": "replace_instrument",
        "people_disagree": "sharpen_criterion",
        "identical_arms_separated": "repair_rig",
        "unfinished": "resume_or_rerun",
        "split": "rerun_on_one_cell",
        "identity_unrecorded": "remeasure_or_attest",
        "measured_under_concurrency": "rerun_serially",
        "trials_lost": "repair_rig",
        "too_few": "add_cases",
        "no_variation": "add_cases",
        "not_recorded": "record_and_rerun",
        "replayed": "run_live",
        "not_drawn_at_random": "draw_at_random",
        "skipped_past_limit": "finish_read",
    }
)

#: What would close a caveat that carries no reason, by its kind. A criterion set after the results, or a pick among
#: many, is confirmed on fresh cases in a new campaign, never by pooling more cases with the ones it was chosen on (no
#: run added to this campaign lifts either); a confound is re-run with the arms differing only in what is compared.
_REMEDY_OF_KIND: Final[Mapping[str, Remedy]] = MappingProxyType(
    {
        "late": "confirm_on_fresh_cases",
        "confound": "rerun_balanced",
        "apparatus": "rerun_balanced",
        "short_run": "resume_or_rerun",
        "no_control": "name_control",
        "nothing_declared": "declare_criteria",
        "untimed": "record_and_rerun",
        "request_settings": "record_and_rerun",
        "judge_uncalibrated": "check_against_people",
        "judge_partly_uncalibrated": "check_against_people",
        "judge_unstated": "record_and_rerun",
        "check": "check_against_people",
        "looks": "declare_looks",
        "best_of_n": "confirm_on_fresh_cases",
        "unsettled": "add_cases",
        "point_estimate": "add_cases",
        "borderline": "add_cases",
        "latency": "rerun_serially",
    }
)


class StepRule(NamedTuple):
    """How a remedy is taken, as data.

    :ivar action: The catalogue action that takes it first; None where a host or a person acts outside the engine.
    :ivar adds_evidence: Whether it is more evidence: it then states a planned size (or why it has none) before it
        runs.
    :ivar spends: Whether taking it costs money; it then names the action that prices it.
    :ivar priced_by: The catalogue action that prices it, for one that spends.
    :ivar needs: The first action's parameters the step leaves to the taker, by the action's own names.
    :ivar fixed: The arguments the remedy itself sets on every call of its launch, or of its one action.
    :ivar per_run: The parameter one call per named run fills, and how the run's id is spelled into it.
    :ivar sized_by: The parameter its planned size fills on every call of its launch: the count of new cases it runs.
    :ivar looks_again: Whether taking it reads the same readings again with what it adds: one more look at them.
    :ivar launches: Whether it launches runs: the launch repeats the set-up of the arms it runs, and its runs join the
        campaign that reads them, since a run joins none by being launched.
    :ivar replaces: Whether the runs it launches stand in for runs the campaign holds, which then leave it.
    :ivar new_campaign: Whether its runs join a campaign it creates first, declared before their results, rather than
        the one the report reads.
    :ivar balances: Whether it re-runs a confound with its dimension held alike: it runs every arm the confound
        touches beside the control, and makes no call where no launch setting holds the dimension.
    :ivar limits: What following it as written cannot guarantee (``RemedyStep.limits``).
    :ivar no_action: For a remedy no action takes, or none has been shown to finish: why.
    """

    action: str | None
    adds_evidence: bool = False
    spends: bool = False
    priced_by: str | None = None
    needs: tuple[str, ...] = ()
    fixed: Mapping[str, StepArgument] = MappingProxyType({})
    per_run: tuple[str, str] | None = None
    sized_by: str | None = None
    looks_again: bool = False
    launches: bool = False
    replaces: bool = False
    new_campaign: bool = False
    balances: bool = False
    limits: tuple[str, ...] = ()
    no_action: NoActionReason | None = None


#: What a launch leaves to its taker where the reading holds none of it: the template and subject, and the models the
#: arms it names ran.
_LAUNCH_NEEDS: Final = ("template_id", "subject_id", "models")

#: What sets an arm up beyond its template, subject and model: the launch parameters a campaign reads as part of the
#: arm (its repeats, whether latency was under test and how the arms shared time, its judge and simulated user, the
#: knobs turned on its kind, its rig and its deadline). A new run left at a default the arm's runs did not run at is
#: not more of that arm, so each is set as the arm's runs were launched.
_ARM_SETUP: Final = (
    "k_runs",
    "measure_latency",
    "arm_schedule",
    "judge_model",
    "simulator_model",
    "overlays",
    "apparatus_settings",
    "cell_timeout_s",
)

#: What a re-run repeats beside: the case set the arm's runs ran, where they ran one (a re-run naming none runs every
#: case the template holds when it starts).
_RERUN_SETUP: Final = ("case_set_name", "case_set_version")

#: What a launch that generates repeats beside: the model that wrote the arm's cases, where its template has an axis a
#: model writes (none where it has not, which refuses one).
_GENERATING_SETUP: Final = ("variation_model",)

#: A step that re-runs what the caveat names: a launch, priced by the estimate that takes the launch's own arguments,
#: whose runs join the campaign in place of the ones they replace.
_LAUNCH: Final = StepRule(
    "run_launch", spends=True, priced_by="launch_estimate", needs=_LAUNCH_NEEDS, launches=True, replaces=True
)

#: A step no sequence of the catalogue's actions has been driven to lift, so it names its remedy and makes no call:
#: a re-run whose fault is no parameter a launch sets, the serial re-run (a launch's ``measure_latency`` runs cases one
#: at a time, but no sequence has followed it through), and more cases on a contrast whose arms the control's new cases
#: would not pair.
_NOT_SHOWN: Final = StepRule(None, no_action="lift_not_shown")

#: The reading holds no template's axes, so a launch's new cases are new only where its template generates them.
_GENERATES: Final = ("axis_may_not_generate",)

#: A step that launches runs on new cases as more evidence: its planned size is the count of new cases the template
#: generates (``n_variations``; 0 would re-run the stored cases, which adds none), joined to the campaign beside the
#: ones it rests on.
_MORE: Final = _LAUNCH._replace(
    adds_evidence=True, sized_by="n_variations", looks_again=True, replaces=False, limits=_GENERATES
)

#: A confirmation on cases the criterion or the pick never saw, in a new campaign declared before its results: the
#: campaign created first with the criteria (``campaign_create``, which takes a design and its floors before any run),
#: then the launch of that many new cases, its runs joined to that campaign as soon as they start, and its control
#: named from them before any result is scored. Priced as the launch it holds. It closes nothing on this report: the
#: caveat is read off this report's own results. ``sweep_launch`` creates a campaign and joins its runs, but declares
#: no criteria, generates no new cases and joins each run as it is created, so nothing it does is declared before its
#: results.
_CONFIRM: Final = StepRule(
    "campaign_create",
    adds_evidence=True,
    spends=True,
    priced_by="launch_estimate",
    needs=("name", "subject_id", "behavior", "question", "aspects", "floors"),
    sized_by="n_variations",
    launches=True,
    new_campaign=True,
    limits=("this_report_stays_flagged", "control_may_read_late", *_GENERATES),
)

#: How each remedy is taken. A remedy whose action depends on the coordinate it is about takes it from
#: :data:`_STEP_BY_COORDINATE`. A run resumed is priced by the listing that states what it has spent and its cap; a
#: declaration once a campaign holds runs is a new campaign, as ``campaign_update`` refuses one.
STEP_OF_REMEDY: Final[Mapping[Remedy, StepRule]] = MappingProxyType(
    {
        "resume_or_rerun": StepRule("job_resume", spends=True, priced_by="jobs_resumable", per_run=("job_id", "run:")),
        "rerun_on_one_cell": _NOT_SHOWN,
        "remeasure_or_attest": StepRule(None),
        "rerun_serially": _NOT_SHOWN,
        "rerun_balanced": _LAUNCH._replace(balances=True),
        "check_against_people": StepRule(
            "result_rate_person",
            needs=("result_id", "rubric_dim", "score", "rating_reason", "person", "rating_source"),
        ),
        "replace_instrument": StepRule(None),
        "sharpen_criterion": StepRule(None),
        "repair_rig": StepRule(None),
        "add_cases": _MORE,
        "confirm_on_fresh_cases": _CONFIRM,
        "record_and_rerun": _NOT_SHOWN,
        "run_live": _NOT_SHOWN,
        "draw_at_random": StepRule(None),
        "finish_read": StepRule(None),
        "declare_looks": StepRule("campaign_create", needs=("name", "subject_id", "behavior", "aspects")),
        "declare_criteria": StepRule(
            "campaign_create", needs=("name", "subject_id", "behavior", "question", "aspects")
        ),
        "name_control": StepRule("campaign_update", needs=("campaign_id", "control_from_run_id")),
        "no_action": StepRule(None),
    }
)

#: The step that re-measures or attests a coordinate, where the catalogue offers one for it: each named run read
#: again under the current definitions, and stored.
_STEP_BY_COORDINATE: Final[Mapping[str, StepRule]] = MappingProxyType(
    {"measure_definition": StepRule("run_remeasure", fixed=MappingProxyType({"store": True}), per_run=("run_id", ""))}
)

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

    Where the part states one id more than once (a late floor under one key on two aspects, a short run on two arms),
    the id is one caveat holding every part: each reading, arm and run it names, its reach the widest, and immaterial
    only where every part is. Nothing a part names is dropped.

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
        _keep_once(found, entry)

    lates: dict[str, list[LateCriterion]] = {}
    for late in caveats.late_criteria:
        lates.setdefault(late_id(late), []).append(late)
    for caveat_id, held_lates in lates.items():
        subjects = [late_subject(report_input, late) for late in held_lates]
        readings = [f"{aspect}/{measure}" for aspect, measure in subjects if aspect is not None and measure is not None]
        whole = len(readings) < len(subjects)
        immaterial = True if all(late_immaterial(report_input, late) for late in held_lates) else None
        add(
            _entry(
                report_input,
                caveat_id,
                "late",
                "report",
                readings=[] if whole else list(dict.fromkeys(readings)),
                immaterial=immaterial,
            )
        )
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
    short: dict[str, list[str | None]] = {}
    for run in caveats.short_runs:
        short.setdefault(run.run, []).append(run.arm)
    for run_id, run_arms in short.items():
        named = None if None in run_arms else [arm for arm in dict.fromkeys(run_arms) if arm is not None]
        wide = _touches_every_comparison(report_input, None if named is None else set(named))
        add(
            _entry(
                report_input,
                f"short_run:{run_id}",
                "short_run",
                "report" if wide else "scoped",
                arms=named or [],
                runs=[run_id],
            )
        )
    unsettled = unsettled_arms(report_input)
    if unsettled:
        called = set(decision_at(report_input).dominant)
        for caveat_id, left in unsettled.items():
            aspect_id = caveat_id.partition(":")[2]
            add(
                _entry(
                    report_input,
                    caveat_id,
                    "unsettled",
                    "report" if called & set(left) else "scoped",
                    arms=list(left),
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


def _keep_once(found: dict[str, CaveatEntry], entry: CaveatEntry) -> None:
    """Add ``entry`` under its id: the same caveat raised twice is one, and two different caveats under one id are
    refused, never one kept and the other dropped.

    :raises ValueError: ``found`` holds a different entry under the same id.
    """
    held = found.get(entry.id)
    if held is None:
        found[entry.id] = entry
    elif held != entry:
        raise ValueError(f"two different caveats were raised under one id, {entry.id!r}: a raiser's ids are not unique")


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
            yield _entry(
                report_input, f"{kind}:{criterion.id}", kind, "report", arms=arms, deciding=True, criterion=criterion.id
            )


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
    for held_arm in report_input.arms:
        identity = held_arm.identity
        if identity is None:
            continue
        coordinates: list[tuple[Coordinate, str | None, str]] = [
            ("served_model", identity.served_model, f"{lead}/{_PASSED}"),
            ("template", identity.template, f"{lead}/{_PASSED}"),
            ("rig", identity.rig, f"{lead}/{_PASSED}"),
            *(
                ("measure_definition", held, f"{aspect_id}/{key}")
                for key, held in identity.measure_definition.items()
                for aspect_id in _measured_on(report_input, key, held_arm.id, lead)
            ),
        ]
        for coordinate, state, address in coordinates:
            if state == "split":
                yield _Fact("withheld", "split", coordinate, address, held_arm.id)
            elif state == "unrecorded":
                yield _Fact("withheld", "identity_unrecorded", coordinate, address, held_arm.id)


def _measured_on(report_input: ReportInput, key: str, arm: str, lead: str) -> list[str]:
    """The aspects one measure holds an arm's figure on, by id: every reading of it the arm has; the lead aspect where
    the input holds none."""
    measure = report_input.measures.get(key)
    if measure is None:
        return [lead]
    held = [lead] if arm in measure.per_arm else []
    held += [aspect_id for aspect_id, arms in measure.by_aspect.items() if arm in arms]
    return held or [lead]


def _fact_id(
    state: ReadingState,
    reading: str,
    reason: Reason | None,
    coordinate: Coordinate | None,
    shared: Mapping[tuple[str, ...], int],
) -> str:
    """A typed caveat's id, unique among the groups ``shared`` counts: ``<state>:<reading>`` where it is the only
    group on that reading in that state, else with its reason, and with its coordinate where two share the reason."""
    if shared[(state, reading)] == 1:
        return f"{state}:{reading}"
    if shared[(state, reading, str(reason))] == 1:
        return f"{state}:{reading}:{reason}"
    return f"{state}:{reading}:{reason}:{coordinate}"


def _from_facts(report_input: ReportInput, facts: Iterable[_Fact]) -> list[CaveatEntry]:
    """One caveat per state, reading, reason and coordinate, its arms every arm the fact holds for, under an id
    unique among them (:func:`_fact_id`)."""
    grouped: dict[tuple[ReadingState, str, Reason | None, Coordinate | None], list[_Fact]] = {}
    for fact in facts:
        grouped.setdefault((fact.state, fact.reading, fact.reason, fact.coordinate), []).append(fact)
    shared: dict[tuple[str, ...], int] = {}
    for state, reading, reason, _coordinate in grouped:
        for key in ((state, reading), (state, reading, str(reason))):
            shared[key] = shared.get(key, 0) + 1
    asked = {aspect.id for aspect in asked_aspects(report_input)}
    found: list[CaveatEntry] = []
    for (state, reading, reason, coordinate), held in grouped.items():
        caveat_id = _fact_id(state, reading, reason, coordinate, shared)
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
    """A caveat per check of a judge against people or a model. A check against people that is met qualifies
    nothing, so it is said one step down, where it still silences a check against a model on its criterion."""
    repeats: dict[str, int] = {}
    for dimension, judged in report_input.judged.items():
        for check in judged.checks:
            if not isinstance(check, JudgeCheck) or check.against not in ("people", "model"):
                continue
            check_id = f"check:{dimension}:{check.criterion or check.instrument}:{check.against}"
            repeats[check_id] = repeats.get(check_id, 0) + 1
            if repeats[check_id] > 1:
                check_id = f"{check_id}:{repeats[check_id]}"
            met = check.result == "met"
            if check.against == "people" and met:
                reach: CaveatReach = "method"
            else:
                reach = _deciding_reach(report_input, [dimension])
            yield _entry(
                report_input,
                check_id,
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


def from_states(report_input: ReportInput, bundle: CampaignBundle | None = None) -> list[CaveatEntry]:
    """A caveat per reading, contrast, pair and figure in a state other than counted, per arm not one cell (on the
    input, and where ``bundle`` is given on the bundle too, grouped once), per check of a judge that does not speak
    for it, and per aspect looked at more often than it planned.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param bundle: The bundle it was built from, where the builder holds it; None for a stored input.
    :ptype bundle: CampaignBundle | None
    :return: The caveats, each holder's in the input's order then the bundle's, then the checks', then the looks'.
    :rtype: list[CaveatEntry]
    """
    facts = [*_input_facts(report_input), *(_bundle_facts(bundle) if bundle is not None else ())]
    return [*_from_facts(report_input, facts), *_checks(report_input), *_looks(report_input)]


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
    """The caveats only the campaign's bundle holds beyond its arms' states (:func:`from_states` reads those): results
    with no ordered time, and request settings not confirmed as sent.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :param report_input: The input built from it, which the caveats are scoped by.
    :ptype report_input: ReportInput
    :return: The caveats.
    :rtype: list[CaveatEntry]
    """
    found: list[CaveatEntry] = []
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
    :raises ValueError: Two raisers raise different caveats under one id.
    """
    stated = from_legacy(report_input)
    raised = [
        *stated,
        *implied(report_input, stated),
        *from_states(report_input, bundle),
        *(from_bundle(bundle, report_input) if bundle is not None else ()),
    ]
    found: dict[str, CaveatEntry] = {}
    for entry in raised:
        _keep_once(found, entry)
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
    rules: list[tuple[AnalysisContextBundle, GateRule]] = []
    for aspect in asked:
        found = gate_rule(aspect)
        if found is not None:
            rules.append((aspect, found))
    if not rules:
        return None
    order: list[str] = list(REASON_OF_GATE)
    rule = min((held for _aspect, held in rules), key=order.index)
    runs: dict[str, GateRun] = {}
    if rule == "too_small":
        counts = [board.usable_cases for aspect, found in rules if found == rule for board in aspect.scoreboard]
        return Gate(rule=rule, fewest_cases=min(counts) if counts else None)
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


def remedy(caveat: CaveatEntry) -> Remedy:
    """The one act that would close ``caveat``: its reason's, else its kind's, else ``no_action``.

    :param caveat: The caveat.
    :ptype caveat: CaveatEntry
    :return: The act.
    :rtype: Remedy
    """
    if caveat.reason is not None:
        return REMEDY_OF_REASON[caveat.reason]
    return _REMEDY_OF_KIND.get(caveat.kind, "no_action")


def _no_action(caveat: CaveatEntry) -> NoActionReason:
    """Why no act closes a caveat: its kind only ever lands in the method section, or the package does not class it."""
    return "method_only" if caveat.kind in CLASS_OF_KIND and CLASS_OF_KIND[caveat.kind] is None else "unclassed_kind"


class Plan(NamedTuple):
    """The size of a step that is more evidence, read before it runs, or why none can be read, or why no added case
    can close the caveat.

    :ivar cases: The cases to add per arm it names (on fresh cases, the cases to run); None where none can be read.
    :ivar unsized: Why none can be read; None where one was.
    :ivar no_action: Why no added case closes the caveat on this input; the step then takes no action.
    :ivar size_bias: Why the size is known to be too small, where it is.
    :ivar arms: The arms the planned design runs, where they are not the caveat's own (a paired reading's control).
    """

    cases: int | None = None
    unsized: UnsizedReason | None = None
    no_action: NoActionReason | None = None
    size_bias: Literal["picked_best_of_n"] | None = None
    arms: tuple[str, ...] | None = None


#: How near its line, relative to the figure's size, an estimate reads as on it: a distance of rounding, at which no
#: finite design settles it.
_ON_THE_LINE: Final = 1e-9

#: A step that is more evidence with nothing named to size it by.
_NOTHING_TO_SETTLE: Final = Plan(unsized="nothing_to_settle")


def _settling(spread: Spread, distance: float, level: float, *, fresh: bool) -> Plan:
    """The cases that settle a figure ``distance`` from its line on a two-sided interval at ``level`` (the interval the
    rule that lifts its caveat reads), four in five designs clearing if the truth is today's estimate
    (:func:`~threetears.evals.analysis.separation.cases_to_separate`, ``sided=2``, at
    :data:`~threetears.evals.analysis.separation.POWER`): the whole design on fresh cases, else the cases to add to
    the ones it rests on."""
    if spread.per_case is None:
        return Plan(unsized="no_count")
    if abs(distance) <= _ON_THE_LINE * max(1.0, abs(spread.mean)):
        return Plan(unsized="on_the_line")
    if spread.per_case <= 0:
        return Plan(unsized="no_spread")
    try:
        needed = cases_to_separate(spread.per_case, distance, level, sided=2, power=POWER)
    except NoSpread:
        return Plan(unsized="no_spread")
    if fresh:
        return Plan(needed)
    held = spread.cases if spread.cases is not None else spread.n
    return Plan(max(needed - math.ceil(held or 0), 1))


def _widest(plans: Iterable[Plan]) -> Plan:
    """The largest size the plans read, so every figure named is settled; where one figure no added case closes, none
    closes them all, and where one reads no size, no size settles them all: the first such reason is the plan's."""
    found = list(plans)
    closed = next((plan for plan in found if plan.no_action is not None), None)
    if closed is not None:
        return closed
    unsized = next((plan for plan in found if plan.cases is None), None)
    if unsized is not None:
        return unsized
    return Plan(max(plan.cases for plan in found if plan.cases is not None)) if found else _NOTHING_TO_SETTLE


def _figure_counts(
    report_input: ReportInput, criterion: Criterion, arm: str, figure: CriterionFigure
) -> tuple[int | None, int | None]:
    """The count a figure's spread rests on and the distinct cases behind it: its own, else its place's cases (a
    task's or a cell's), else, for an overall figure, the arm's usable trials over its cases on the criterion's aspect
    (the page's settle reads the same)."""
    if figure.n is not None or figure.n_cases is not None:
        return (figure.n if figure.n is not None else figure.n_cases), figure.n_cases
    if figure.cell is not None:
        return report_input.cells[figure.cell].n_cases, None
    if figure.task is not None:
        return next((task.n_cases for task in report_input.tasks if task.id == figure.task), None), None
    other = criterion_other_aspect(report_input, criterion)
    if other is not None and other not in {aspect.id for aspect in report_input.aspects}:
        return None, None
    board = aspect_board(report_input, other).get(arm)
    if board is None:
        return None, None
    if board.usable is not None:
        return board.usable, board.cases
    return board.cases, None


def _figure_plan(
    report_input: ReportInput, criterion: Criterion, arm: str, figure: CriterionFigure, *, fresh: bool
) -> Plan:
    """The plan that settles one arm's figure against a gate's, a guardrail's or a bar's line."""
    if criterion.statistic != "mean":
        return Plan(unsized="no_interval")
    table = report_input.measures.get(criterion.measure) or report_input.judged.get(criterion.measure)
    n, cases = _figure_counts(report_input, criterion, arm, figure)
    try:
        spread = stat_spread(
            figure.value,
            ci=figure.ci,
            half_width=figure.half_width,
            sem=figure.sem,
            n=n,
            stated_level=None if table is None else level_share(table.level),
            cases=cases,
        )
    except NoSpread:
        return Plan(unsized="no_interval")
    line = float(criterion.threshold)
    distance = figure.value - line if criterion.direction == "higher_is_better" else line - figure.value
    # A figure's status reads its interval two-sided at the level its table states it at (criteria's borderline).
    return _settling(spread, distance, spread.level, fresh=fresh)


def _criteria_plan(
    report_input: ReportInput, criteria: Iterable[Criterion], arms: Iterable[str], *, fresh: bool
) -> Plan:
    held = list(arms)
    return _widest(
        _figure_plan(report_input, criterion, arm, figure, fresh=fresh)
        for criterion in criteria
        for arm in held
        for figure in criterion_figures(report_input, criterion, arm)
    )


def _reading_plan(aspect: Aspect, reading: Reading, *, fresh: bool) -> Plan:
    """The plan that settles a candidate's reading against its margin's line (no margin: against no change) on its
    nominal interval, two-sided at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`: what an unsettled reading
    is lifted by, and what one candidate against its control reads a gain on. Its spread is read from that interval
    and the cases each arm ran; a reading with no margin holds no nominal interval, so its gain interval gives the
    spread, read at that interval's own level."""
    level: float | None
    if reading.within_interval is not None:
        interval, level = reading.within_interval, INTERVAL_LEVEL
    elif reading.gain_interval is not None:
        interval, level = reading.gain_interval, level_share(reading.gain_confidence)
    else:
        return Plan(unsized="no_interval")
    counts = tuple(float(count) for count in reading.cases)
    try:
        spread = difference_spread(float(reading.change), (float(interval[0]), float(interval[1])), level, counts)
    except NoSpread:
        return Plan(unsized="no_interval")
    margin = reading.margin if reading.margin is not None else aspect.margin
    line = 0.0 if margin is None else -float(margin)
    return _settling(spread, float(reading.change) - line, INTERVAL_LEVEL, fresh=fresh)


def _readings_plan(aspects: Iterable[Aspect], arms: Iterable[str], *, fresh: bool) -> Plan:
    named = set(arms)
    return _widest(
        _reading_plan(aspect, reading, fresh=fresh)
        for aspect in aspects
        for reading in aspect.readings
        if reading.arm in named
    )


def _to_the_floor(cases: int | None) -> Plan:
    """A figure too few cases read for an interval: the cases that reach the floor where one is first stated."""
    if cases is None:
        return Plan(unsized="no_count")
    if cases >= SMALL_N_BAND_FLOOR:
        return _NOTHING_TO_SETTLE
    return Plan(SMALL_N_BAND_FLOOR - cases)


def _point_estimate_plan(report_input: ReportInput, criterion: Criterion) -> Plan:
    """A gate cleared on a figure with no interval, at each place it is (the caveat's own places,
    :func:`cleared_on_point_estimate`): a mean read from fewer cases than the floor where its interval is first stated
    is sized to that floor; a percentile, or a mean read from that many cases that still states none, gains no
    interval from more cases, so no added case closes it."""
    plans: list[Plan] = []
    for arm, task, cell in cleared_on_point_estimate(report_input, criterion.id):
        if criterion.statistic != "mean":
            plans.append(Plan(no_action="no_interval_to_gain"))
            continue
        figure = next(
            held for held in criterion_figures(report_input, criterion, arm) if (held.task, held.cell) == (task, cell)
        )
        n, cases = _figure_counts(report_input, criterion, arm, figure)
        count = cases if cases is not None else n
        if count is not None and count >= SMALL_N_BAND_FLOOR:
            plans.append(Plan(no_action="no_interval_to_gain"))
        else:
            plans.append(_to_the_floor(count))
    return _widest(plans)


def _with_control(report_input: ReportInput, arms: Iterable[str]) -> tuple[str, ...]:
    """The arms a design that re-reads a change against the control runs: the control's new cases pair with theirs."""
    named = list(dict.fromkeys(arms))
    control = control_arm(report_input)
    return tuple(named if control is None or control in named else [*named, control])


def _late_plan(report_input: ReportInput, caveat: CaveatEntry) -> Plan:
    """A criterion set after the results, confirmed on fresh cases: the design that settles what it reads, on every
    surviving option: a declared gate's, guardrail's or bar's figures, else the readings of the aspect it is on, else
    of every asked aspect."""
    late = next((held for held in report_input.caveats.late_criteria if late_id(held) == caveat.id), None)
    if late is None:
        return _NOTHING_TO_SETTLE
    survivors = surviving_options(report_input)
    criteria = [held for held in report_input.criteria if held.kind == late.criterion and held.measure == late.name]
    if criteria:
        return _criteria_plan(report_input, criteria, survivors, fresh=True)
    aspect_id, _measure = late_subject(report_input, late)
    on = aspect_id or late.aspect
    aspects = [aspect for aspect in report_input.aspects if aspect.id == on] or asked_aspects(report_input)
    return _readings_plan(aspects, survivors, fresh=True)


def plan_of(report_input: ReportInput, caveat: CaveatEntry) -> Plan | None:
    """The size of the step that would close ``caveat``, where that step is more evidence (this module's A step that
    is more evidence is planned); None where it is not.

    :param report_input: The input the caveat was raised on, which holds the figures it names.
    :ptype report_input: ReportInput
    :param caveat: The caveat.
    :ptype caveat: CaveatEntry
    :return: The plan, or None.
    :rtype: Plan | None
    """
    remedied = remedy(caveat)
    if not STEP_OF_REMEDY[remedied].adds_evidence:
        return None
    if caveat.reason == "too_few":
        return _to_the_floor(caveat.cases)
    if caveat.reason == "no_variation":
        return Plan(unsized="no_spread")
    criteria = [held for held in report_input.criteria if held.id == caveat.criterion]
    if caveat.kind == "point_estimate" and criteria:
        return _point_estimate_plan(report_input, criteria[0])
    if caveat.kind == "borderline" and criteria:
        return _criteria_plan(report_input, criteria, caveat.arms, fresh=False)
    aspects = [aspect for aspect in report_input.aspects if f"{aspect.id}/{_PASSED}" in caveat.readings]
    if caveat.kind == "unsettled":
        plan = _readings_plan(aspects, caveat.arms, fresh=False)
        return plan._replace(arms=_with_control(report_input, caveat.arms))
    if caveat.kind == "best_of_n":
        # No estimate the engine reads corrects a pick for having been the best of several, so the confirmation is
        # sized from the pick's own and says it is too small.
        picks = [arm for arm in dominant_options(report_input) if arm in caveat.arms] or caveat.arms
        plan = _readings_plan(asked_aspects(report_input) or report_input.aspects, picks, fresh=True)
        bias: Literal["picked_best_of_n"] | None = "picked_best_of_n" if plan.cases is not None else None
        return plan._replace(size_bias=bias, arms=_with_control(report_input, picks))
    if caveat.kind == "late":
        # The confirmation runs every surviving option, and the control its readings are read against.
        return _late_plan(report_input, caveat)._replace(
            arms=_with_control(report_input, surviving_options(report_input))
        )
    return _NOTHING_TO_SETTLE


class ArmRuns(NamedTuple):
    """What an arm's runs on one aspect were launched with, where the bundle says.

    :ivar runs: The arm's runs there, by id.
    :ivar model: The candidate model every one of them was launched with; None where they differ.
    :ivar k_runs: The repeats per case every one of them was launched with; None where they differ.
    :ivar beyond_model: Whether the arm's levers differ from the control's in more than the model: one launch runs
        arms that differ only in their model, so such an arm is launched in a call of its own.
    :ivar measure_latency: Whether every one of them declared latency under test; None where they differ or one
        recorded no rig.
    :ivar arm_schedule: How every one of them shared time with the other arms; None where they differ or one
        recorded none.
    """

    runs: tuple[str, ...]
    model: str | None
    k_runs: int | None
    beyond_model: bool
    measure_latency: bool | None = None
    arm_schedule: str | None = None


class Setup(NamedTuple):
    """What a step that launches can fill from the reading: the campaign's subject, each aspect's template and each
    arm's runs on it, the control and the aspects.

    :ivar subject_id: The subject the campaign measures; None where the reading does not hold it.
    :ivar templates: Each aspect's template, by the aspect's id.
    :ivar arms: Each aspect's arms' runs, by the aspect's id and then the arm's.
    :ivar control: The control arm, by id; None where the campaign names none.
    :ivar aspects: The input's aspects, by id.
    """

    subject_id: str | None = None
    templates: Mapping[str, str] = MappingProxyType({})
    arms: Mapping[str, Mapping[str, ArmRuns]] = MappingProxyType({})
    control: str | None = None
    aspects: tuple[str, ...] = ()


def _aspect_of(readings: Sequence[str], setup: Setup) -> str | None:
    """The one aspect a step's runs are of: the one the caveat's readings are on, else the campaign's only one."""
    named = {reading.partition("/")[0] for reading in readings} & set(setup.aspects)
    if len(named) == 1:
        return next(iter(named))
    return setup.aspects[0] if not named and len(setup.aspects) == 1 else None


#: The settings a launch call fills per arm where every arm it runs holds one, beside the model.
_PER_ARM: Final = ("k_runs", "measure_latency", "arm_schedule")


class _Calls(NamedTuple):
    """A step's launch calls: each call's arguments, the arms each runs, what every call fills of the arms' set-up,
    and the arms whose call is not the control's."""

    calls: list[dict[str, StepArgument]]
    arms: list[list[str]]
    filled: set[str]
    unpaired: list[str]


def _launch_calls(
    base: Mapping[str, StepArgument], arms: Sequence[str], held: Mapping[str, ArmRuns], control: str | None
) -> _Calls:
    """The launch calls that run ``arms`` as their runs ran, which arms each runs, and what every one of them fills.

    Arms whose levers differ from the control's only in the model, and that ran at the same repeats, latency and
    schedule, share one call, so the new cases it generates are every one of theirs; an arm that differs in more takes
    a call of its own, whose new cases the control does not run. Where an arm's runs are not held, or disagree on their
    model, one call names no model and the taker supplies them. A setting of :data:`_PER_ARM` is filled where every
    arm's runs agree on one.
    """
    found = [held.get(arm) for arm in arms]
    runs = [arm_runs for arm_runs in found if arm_runs is not None and arm_runs.model is not None]
    if not arms or len(runs) != len(arms):
        return _Calls([dict(base)], [], set(), [])
    every = {name for name in _PER_ARM if all(getattr(arm_runs, name) is not None for arm_runs in runs)}
    groups: dict[tuple[Any, ...], tuple[list[str], list[str]]] = {}
    for arm, arm_runs in zip(arms, runs, strict=True):
        key = (arm if arm_runs.beyond_model else None, *(getattr(arm_runs, name) for name in _PER_ARM if name in every))
        models, named = groups.setdefault(key, ([], []))
        named.append(arm)
        if arm_runs.model is not None and arm_runs.model not in models:
            models.append(arm_runs.model)
    calls: list[dict[str, StepArgument]] = []
    call_arms: list[list[str]] = []
    for key, (models, named) in groups.items():
        call: dict[str, StepArgument] = {**base, "models": models}
        for name, value in zip([name for name in _PER_ARM if name in every], key[1:], strict=True):
            call[name] = value
        calls.append(call)
        call_arms.append(named)
    with_control = next((named for named in call_arms if control in named), None)
    unpaired = [arm for named in call_arms if named is not with_control for arm in named] if with_control else []
    return _Calls(calls, call_arms, {"models", *every}, unpaired)


def _follow(
    remedied: Remedy, closes: str | None, action: str, call: dict[str, StepArgument], **fields: Any
) -> RemedyStep:
    """One call that follows a step's first, of the step's remedy and the caveat it closes."""
    return RemedyStep(remedy=remedied, closes=closes, action=action, calls=[call], **fields)


class _Launching(NamedTuple):
    """A launching step's first call, the calls that follow it, and what the head step states of the launch: the arms
    it cannot pair, and whether no arm the caveat names but the control is left to launch, so no call can lift it."""

    first: RemedyStep
    then: list[RemedyStep]
    unpaired: list[str]
    stranded: bool = False


def _launching(
    remedied: Remedy,
    rule: StepRule,
    *,
    closes: str | None,
    arms: Sequence[str],
    runs: Sequence[str],
    readings: Sequence[str],
    cases: int | None,
    setup: Setup,
    alike: Sequence[str] = (),
    contrast: bool = False,
) -> _Launching:
    """A launch that repeats the arms' set-up, and the calls that finish it: its runs joined to the campaign it is
    read in, the runs it replaces taken out, or, for a confirmation, the campaign created before it and its control
    named from its runs. A caveat that names no arm is on every arm the aspect holds.

    Only a reading of a contrast against the control pairs cases, so only a step whose caveat reads one
    (``contrast``), and a confirmation, whose new campaign reads its contrasts, state the arms a generating launch
    cannot pair; a gate or a figure reads each arm's own cases. On this report's contrast, a call the control is not
    in adds no pair, so it is not made, and where that leaves no arm the caveat names but the control, the step is
    stranded: no call it could make lifts the caveat."""
    aspect = _aspect_of(readings, setup)
    held = setup.arms.get(aspect, {}) if aspect is not None else {}
    arms = list(arms) or list(held)
    base: dict[str, StepArgument] = dict(rule.fixed)
    template = setup.templates.get(aspect) if aspect is not None else None
    if template is not None:
        base["template_id"] = template
    if setup.subject_id is not None:
        base["subject_id"] = setup.subject_id
    generates = rule.sized_by is not None and cases is not None
    if rule.sized_by is not None and cases is not None:
        base[rule.sized_by] = cases
    built = _launch_calls(base, arms, held, setup.control)
    # A re-run on the template's stored cases runs one set in every call; only cases a call generates are its own.
    unpaired = built.unpaired if generates and (contrast or rule.new_campaign) else []
    calls, call_arms, stranded = built.calls, built.arms, False
    if unpaired and not rule.new_campaign:
        # The contrast pairs only the cases both sides ran: a call the control is not in adds no pair to it.
        kept = [index for index, named in enumerate(built.arms) if setup.control in named]
        calls, call_arms = [built.calls[index] for index in kept], [built.arms[index] for index in kept]
        stranded = not [arm for named in call_arms for arm in named if arm != setup.control and arm in arms]
    setting = [*_ARM_SETUP, *(_GENERATING_SETUP if generates else ()), *(_RERUN_SETUP if rule.replaces else ())]
    launch = RemedyStep(
        remedy=remedied,
        closes=closes,
        action="run_launch",
        calls=calls,
        call_arms=call_arms,
        needs=[name for name in _LAUNCH_NEEDS if name not in base and name not in built.filled],
        as_launched=[name for name in setting if name not in base and name not in built.filled],
        alike=list(alike),
        spends=True,
        priced_by="launch_estimate",
    )
    if rule.new_campaign:
        # The new campaign's aspect is the one its taker declares: each run joins the aspect on the template it ran.
        created: dict[str, Any] = {"campaign_id": "created_campaign"}
        then = [
            launch,
            _follow(remedied, closes, "campaign_runs_add", {}, after={**created, "run_ids": "launched_runs"}),
        ]
        if setup.control is not None and setup.control in arms:
            control = {**created, "control_from_run_id": "launched_control_run"}
            then.append(_follow(remedied, closes, "campaign_update", {}, after=control))
        first: dict[str, StepArgument] = {"subject_id": setup.subject_id} if setup.subject_id is not None else {}
        create = RemedyStep(
            remedy=remedied,
            closes=closes,
            action="campaign_create",
            calls=[first],
            needs=[name for name in rule.needs if name not in first],
        )
        return _Launching(create, then, unpaired)
    joined: dict[str, StepArgument] = {"aspect": aspect} if aspect is not None else {}
    then = [
        _follow(
            remedied, closes, "campaign_runs_add", joined, needs=["campaign_id"], after={"run_ids": "launched_runs"}
        )
    ]
    if rule.replaces:
        replaced = list(runs) or [run for arm in arms if arm in held for run in held[arm].runs]
        if runs or (arms and all(arm in held for arm in arms) and replaced):
            then.append(_follow(remedied, closes, "campaign_runs_remove", {"run_ids": replaced}, needs=["campaign_id"]))
        else:
            then.append(_follow(remedied, closes, "campaign_runs_remove", {}, needs=["campaign_id", "run_ids"]))
    return _Launching(launch, then, unpaired, stranded)


def next_step(
    remedied: Remedy,
    *,
    closes: str | None = None,
    arms: Sequence[str] = (),
    runs: Sequence[str] = (),
    readings: Sequence[str] = (),
    coordinate: Coordinate | None = None,
    plan: Plan | None = None,
    results_needed: int | None = None,
    recording: Sequence[NextExperiment] = (),
    no_action: NoActionReason | None = None,
    setup: Setup = Setup(),
    alike: Sequence[str] = (),
    contrast: bool = False,
) -> RemedyStep:
    """The step that takes one remedy, filled from what the caveat (or the gate) names (:data:`STEP_OF_REMEDY`): one
    call of its action per run it names where the action takes one run, else one call, each with the arguments the
    remedy sets, and the parameters the caveat cannot fill named for the taker.

    A step that launches runs is the whole act the existing actions take, in order: the launch, repeating the set-up
    of the arms it runs (what ``setup`` holds of it filled, the rest named in ``as_launched``), its runs joined to the
    campaign that reads them, and, for a re-run, the runs it replaces taken out; a confirmation creates its campaign
    first, joins the runs to it and names its control from them. Each call that follows names what only an earlier
    call's result holds (``after``), and the step states, typed, what following it cannot guarantee (``limits``). A
    re-run that balances a confound runs every arm it touches beside the control, with the confound's dimensions held
    at one value on every call (``alike``); one with no dimension a launch sets makes no call. A step whose caveat
    reads a contrast against the control (``contrast``) launches only the arms whose new cases the control runs too,
    since the contrast pairs only those, and names the rest (``unpaired_arms``); where no arm the caveat names but the
    control is left, it makes no call (``lift_not_shown``) and states no size. A gate or a figure reads each arm's own
    cases, so pairing does not apply to its step.

    :param remedied: The remedy.
    :ptype remedied: Remedy
    :param closes: The caveat it closes, by id; None for the gate's.
    :ptype closes: str | None
    :param arms: The arms the caveat touches.
    :ptype arms: Sequence[str]
    :param runs: The runs it names.
    :ptype runs: Sequence[str]
    :param readings: The readings it names, whose aspect a launch's runs are of.
    :ptype readings: Sequence[str]
    :param coordinate: The coordinate it is about.
    :ptype coordinate: Coordinate | None
    :param plan: For more evidence, its size or why it has none (:func:`plan_of`); None reads nothing to settle. A
        step that adds cases passes the size as its count of new cases; one with no size makes no call.
    :ptype plan: Plan | None
    :param results_needed: For a check: the results it needs before it decides.
    :ptype results_needed: int | None
    :param recording: What recording an unrecorded coordinate would pool.
    :ptype recording: Sequence[NextExperiment]
    :param no_action: For ``no_action``: why nothing closes it.
    :ptype no_action: NoActionReason | None
    :param setup: What a launch can fill from the reading; none for a step read off nothing but the caveat.
    :ptype setup: Setup
    :param alike: For a re-run that balances a confound: the apparatus dimensions it holds at one value.
    :ptype alike: Sequence[str]
    :param contrast: Whether the rule that lifts the caveat reads a contrast against the control on this report,
        which pairs only the cases both sides ran.
    :ptype contrast: bool
    :return: The step.
    :rtype: RemedyStep
    """
    rule = STEP_OF_REMEDY[remedied]
    if rule.balances:
        if not alike:
            # No launch setting holds the confound's dimension alike, so no re-run is shown to remove it.
            rule = _NOT_SHOWN
        elif setup.control is not None and arms and setup.control not in arms:
            arms = [*arms, setup.control]
    if remedied == "remeasure_or_attest":
        rule = _STEP_BY_COORDINATE.get(coordinate or "", rule)
    sized = (plan or _NOTHING_TO_SETTLE) if rule.adds_evidence else Plan()
    if rule.adds_evidence and sized.cases is None:
        # No size, no call: a step that adds cases with no planned count would run open-ended, or not at all.
        rule = StepRule(None)
    head: dict[str, Any] = {}
    then: list[RemedyStep] = []
    if rule.launches:
        launching = _launching(
            remedied,
            rule,
            closes=closes,
            arms=arms,
            runs=runs,
            readings=readings,
            cases=sized.cases,
            setup=setup,
            alike=alike if rule.balances else (),
            contrast=contrast,
        )
        if launching.stranded:
            # Every arm the caveat names but the control would run on new cases the control never runs, which its
            # contrast cannot pair: no call lifts it, so the step makes none, states no size and names the arms.
            rule, sized = _NOT_SHOWN, Plan()
            head = {"limits": ["unpaired_arms"], "unpaired_arms": launching.unpaired}
        else:
            first, then = launching.first, launching.then
            head = {
                "action": first.action,
                "calls": first.calls,
                "call_arms": first.call_arms,
                "needs": first.needs,
                "as_launched": first.as_launched,
                "alike": first.alike,
                "limits": [*rule.limits, *(["unpaired_arms"] if launching.unpaired else [])],
                "unpaired_arms": launching.unpaired,
            }
    elif rule.action is not None:
        needs = list(rule.needs)
        if rule.per_run is not None and runs:
            parameter, prefix = rule.per_run
            calls = [{**rule.fixed, parameter: f"{prefix}{run}"} for run in runs]
        else:
            calls = [dict(rule.fixed)]
            if rule.per_run is not None:
                needs.append(rule.per_run[0])
        head = {"action": rule.action, "calls": calls, "needs": needs}
    return RemedyStep(
        remedy=remedied,
        closes=closes,
        **head,
        then=then,
        arms=list(arms),
        runs=list(runs),
        planned_cases=sized.cases,
        unsized=sized.unsized,
        size_bias=sized.size_bias,
        declared_look=rule.looks_again and sized.cases is not None,
        results_needed=results_needed if remedied == "check_against_people" else None,
        recording=list(recording),
        spends=rule.spends,
        priced_by=rule.priced_by,
        no_action=(no_action or "unclassed_kind") if remedied == "no_action" else rule.no_action,
    )


def caveat_step(
    caveat: CaveatEntry,
    *,
    plan: Plan | None = None,
    results_needed: int | None = None,
    recording: Sequence[NextExperiment] = (),
    setup: Setup = Setup(),
) -> RemedyStep:
    """The step that would close one caveat; ``no_action``, with why, where nothing does, or where its plan reads that
    no added case closes it on this input.

    :param caveat: The caveat.
    :ptype caveat: CaveatEntry
    :param plan: For more evidence, its size read off the input (:func:`plan_of`).
    :ptype plan: Plan | None
    :param results_needed: For a check against people: the results it needs, where the bundle says.
    :ptype results_needed: int | None
    :param recording: What recording its coordinate would pool, where the bundle says.
    :ptype recording: Sequence[NextExperiment]
    :param setup: What a launch can fill from the reading.
    :ptype setup: Setup
    :return: The step.
    :rtype: RemedyStep
    """
    remedied = remedy(caveat)
    if plan is not None and plan.no_action is not None:
        return next_step("no_action", closes=caveat.id, arms=caveat.arms, runs=caveat.runs, no_action=plan.no_action)
    return next_step(
        remedied,
        closes=caveat.id,
        arms=list(plan.arms) if plan is not None and plan.arms is not None else caveat.arms,
        runs=caveat.runs,
        readings=caveat.readings,
        coordinate=caveat.coordinate,
        plan=plan,
        results_needed=results_needed,
        recording=recording if caveat.reason == "identity_unrecorded" else (),
        no_action=_no_action(caveat) if remedied == "no_action" else None,
        setup=setup,
        alike=[caveat.id.partition(":")[2]] if caveat.kind == "apparatus" else (),
        contrast=caveat.kind == "unsettled",
    )


def gate_step(gate: Gate, setup: Setup = Setup()) -> RemedyStep:
    """The step that would lift a gate: its rule's reason's remedy, on the runs it names; too few cases, sized to the
    floor where an interval is first stated from the fewest an arm holds.

    :param gate: The gate.
    :ptype gate: Gate
    :param setup: What a launch can fill from the reading.
    :ptype setup: Setup
    :return: The step.
    :rtype: RemedyStep
    """
    reason = REASON_OF_GATE[gate.rule]
    plan = _to_the_floor(gate.fewest_cases) if reason == "too_few" else None
    return next_step(REMEDY_OF_REASON[reason], runs=[run.run for run in gate.runs], plan=plan, setup=setup)


class Evidence(NamedTuple):
    """What the bundle says about how far a step would go, and what a launch repeats, where the builder holds it.

    :ivar results_needed: The most results any judge's check against people needs before it decides.
    :ivar recording: What recording an unrecorded coordinate would pool, by arm.
    :ivar subject_id: The subject the campaign measures.
    :ivar templates: Each aspect's template, by the aspect's id.
    :ivar arms: Each aspect's arms' runs and what they were launched with, by the aspect's id and then the arm's.
    """

    results_needed: int | None = None
    recording: Mapping[str, tuple[NextExperiment, ...]] = MappingProxyType({})
    subject_id: str | None = None
    templates: Mapping[str, str] = MappingProxyType({})
    arms: Mapping[str, Mapping[str, ArmRuns]] = MappingProxyType({})


def _one(values: Iterable[Any]) -> Any:
    """The one value every item holds, else None."""
    held = set(values)
    return next(iter(held)) if len(held) == 1 else None


def evidence_of(bundle: CampaignBundle) -> Evidence:
    """What a campaign's bundle says about how far each step would go, and what each arm's runs were launched with.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :return: The results its judges' checks need, what recording would pool, by arm, and the subject, each aspect's
        template and each arm's runs there.
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
    arms: dict[str, dict[str, ArmRuns]] = {}
    for aspect in bundle.aspects:
        summaries = {summary.run_id: summary for summary in aspect.run_summaries}
        control = [aspect.design.control_arm] if aspect.design.control_arm is not None else []
        for arm in [*control, *aspect.design.contrasts]:
            ran = [summaries[run] for run in arm.run_ids if run in summaries]
            rigs = [summary.apparatus for summary in ran]
            recorded = [rig for rig in rigs if rig is not None] if all(rig is not None for rig in rigs) else []
            arms.setdefault(aspect.aspect, {})[bundle.label_of(arm.variant_key)] = ArmRuns(
                runs=tuple(arm.run_ids),
                model=_one(summary.candidate_model for summary in ran),
                k_runs=_one(summary.k_runs for summary in ran),
                beyond_model=bool(set(arm.moved) - {CANDIDATE_MODEL_LEVER}),
                measure_latency=_one(rig.measure_latency for rig in recorded),
                arm_schedule=_one(rig.arm_schedule for rig in recorded),
            )
    return Evidence(
        max(needed) if needed else None,
        {arm: tuple(held) for arm, held in recording.items()},
        bundle.subject_id,
        {aspect.aspect: aspect.template_id for aspect in bundle.aspects if aspect.template_id},
        arms,
    )


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
    :param evidence: What the bundle says about how far a step would go and what a launch repeats; none for a stored
        input.
    :ptype evidence: Evidence
    :return: The first screen's value.
    :rtype: FirstScreen
    """
    value = first_screen(caveats, gate=gate, called_arms=decision_at(report_input).dominant, call=call)
    setup = Setup(
        subject_id=evidence.subject_id,
        templates=evidence.templates,
        arms=evidence.arms,
        control=control_arm(report_input),
        aspects=tuple(aspect.id for aspect in report_input.aspects),
    )
    if value.gate is not None:
        return value.model_copy(update={"steps": [gate_step(value.gate, setup)]})
    by_id = {caveat.id: caveat for caveat in caveats}
    steps = []
    for reason in value.reasons:
        caveat = by_id[reason]
        recording = [held for arm in caveat.arms for held in evidence.recording.get(arm, ())]
        plan = plan_of(report_input, caveat)
        steps.append(
            caveat_step(caveat, plan=plan, results_needed=evidence.results_needed, recording=recording, setup=setup)
        )
    return value.model_copy(update={"steps": steps})
