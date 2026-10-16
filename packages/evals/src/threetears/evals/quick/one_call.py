"""The engine under an :class:`~threetears.evals.quick.contract.Eval`: its arms launched as one, each cell a callable.

An eval hands over its cases, its runner (one arm, or a mapping of arms) and its graders; everything else a host would
build is built here, from the public roots, on the terms the engine already sets:

- **The kind** is :class:`CallableKind`: ``invoke`` calls the runner on the case and each function grader on what it
  returned, and reports the grades as the host's measures. It is unjudged — the graders are the grade — and seeds no
  world. A runner that calls :class:`~threetears.evals.providers.Model` reports what each call spent, and one that
  returns an :class:`~threetears.evals.quick.answer.Answer` reports its own spend beside its answer; both land as the
  cell's ``candidate`` usage row.
- **A classifier** is a runner whose answer carries a label, declared by its eval's first field grader. Each cell then
  lands the core ``match`` and ``confusion_cell`` as a classifier kind does, so the summary carries the confusion
  matrix and each label's precision, recall and F1, and the analysis derives ``accuracy``. An answer that has no
  usable label lands under :data:`~threetears.evals.quick.graders.UNUSABLE_ANSWER`, a predicted label of its own —
  and so does a runner that raised, a failure counted as a miss rather than left out of every rate.
- **A judge** is a model grading each answer on a rubric (:class:`~threetears.evals.quick.judged.Judge`). The run is
  then of :data:`JUDGED_CALLABLE_KIND`, a document kind: the template carries the rubric, each cell renders the answer
  and its case as the judge's evidence, and the engine's own judge service scores every dimension and records the
  judge's spend on the result's ``judge`` usage row, as it does for any judged run. The judge also reads the
  template's intent: the first line of the runner's docstring, else a generic sentence; the summary names which.
- **A world** (:class:`~threetears.evals.quick.world.World`) is either state the agent acts on through tools — each
  case's starting state seeded before the agent's first turn, read back after its last, and graded by goal checks
  through the engine's own world session and goal-state evaluation — or a set of lookups the agent calls, recorded
  to a file and replayed from it (:class:`~threetears.evals.quick.tools.ToolRecording`).
- **The host**, when none is given, is :func:`callable_host`: the shared sweepable core, one measure per function
  grader, the world when it has state, and the store the eval runs in. Given one, its storage and vocabulary are used:
  every function grader must already be a measure it declares, and it must declare a contract for the callable kind
  that seats no judge, simulator or spend ceiling — and, for a judged eval, a contract for the judged kind that seats
  the judge and nothing else of the engine's (:func:`~threetears.evals.quick.contract.as_host` builds one).
- **The launch** goes through :func:`~threetears.evals.run.start_run`, the path a product serving launches takes, so
  the run is assembled, stamped and identified exactly as any other run is; the call then waits for its job and reads
  the stored run back.

**The case set is content-addressed.** The template's id is a digest of the cases, and each case's id is its position
under it, so two evals over the same cases in one store share a template and their runs are runs of one scenario —
comparable in a campaign — while a different case list is a different template.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, cast

from pydantic import BaseModel

from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.metrics import describe_reported_measure
from threetears.evals.kernel import default_job_timeout
from threetears.evals.kernel import (
    CandidateKind,
    ACCURACY_MEASURE,
    CONFUSION_CELL_MEASURE,
    JUDGE_KIND,
    MATCH_MEASURE,
    METRIC_DESCRIPTORS,
    CandidateOutput,
    CandidateTelemetry,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    CostCapReached,
    EvalStorage,
    MetricDescriptor,
    VariantConfig,
    WorldSession,
    confusion_cell,
    withhold_failure_detail,
)
from threetears.evals.schema import (
    DEFAULT_LAUNCH_K_RUNS,
    DocumentStore,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    KnownUsageRole,
    WorldSeed,
    canonical_digest,
)
from threetears.evals.schema.models import stored_variation
from threetears.evals.kernel.host import (
    SHARED_CORE,
    ApparatusError,
    CompletionClients,
    EvalHost,
    HostProfile,
    KindContract,
    MeasureRegistry,
    Sweepable,
    SweepableRegistry,
    WorldRegistry,
    default_cell_timeout,
)
from threetears.evals.schema import NominalScale, SubjectSnapshot, SweepableValue, TraceSink
from threetears.evals.schema import WorldPlacement
from threetears.evals.kernel.host.sweepables import CORE_ROLES, CORE_SWEEPABLES
from threetears.evals.analysis.summary import CaseResult, EvalSummary, summarize_run
from threetears.evals.run import (
    CellContext,
    KindFactory,
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    LaunchArgument,
    LaunchGroup,
    LaunchSettings,
    RunJudge,
    TemplateField,
    UnknownSubject,
    build_judge_service,
    freeze_judge_cases,
    get_result_trace,
    launch_as_group,
    launch_run,
    list_results,
    start_run,
)
from threetears.evals.run.authoring import active_judge_config_in, refuse_undeclared_judges, refuse_unsupplied_world
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.runner_spend import cell_spend, spent
from threetears.evals.kernel.usage_capture import blended_cost
from threetears.evals.quick.answer import Answer, unwrap_answer
from threetears.evals.quick.guardrails import Guardrail
from threetears.evals.quick.judged import Judge, judge_evidence
from threetears.evals.quick.levers import CallableLevers, levers_model
from threetears.evals.quick.logged import Log
from threetears.evals.quick.measures import Measure
from threetears.evals.quick.graders import UNUSABLE_ANSWER, predicted_label
from threetears.evals.quick.tools import CellTools, Tool, ToolRecording, ToolUsingCandidate
from threetears.evals.quick.world import World, WorldAgent, WorldCellKind, WorldOutcome, world_case_payload
from threetears.evals.storage import InMemoryDocumentStore

#: The runner under test: a callable taking one case and returning its answer — usually ``async def``; a plain
#: ``def`` is called in a worker thread, and a callable returning an awaitable has it awaited.
Candidate = Callable[[Mapping[str, Any]], Awaitable[Any] | Any]

#: One grade: takes the case and the candidate's answer, returns a number (``True``/``False`` count
#: as 1 and 0). Its ``__name__`` is the measure's name, and higher is better. No scorer may take the name
#: of an engine core measure (``score``, ``f1``, ``cost_usd`` and the rest of ``METRIC_DESCRIPTORS``).
Scorer = Callable[[Mapping[str, Any], Any], float | bool]

#: A classifier's expected label for one case: takes the case, returns the label a correct answer gives.
ExpectedLabel = Callable[[Mapping[str, Any]], str]

#: The names a classifier lands (``match``, ``confusion_cell``) and the one the engine derives from them
#: (``accuracy``). No scorer may take one: it would collide with a classifier's own measure, or be read
#: under the core descriptor of a measure it is not.
_CLASSIFIER_NAMES = frozenset({MATCH_MEASURE, CONFUSION_CELL_MEASURE, ACCURACY_MEASURE})

#: The kind :class:`~threetears.evals.quick.contract.Eval` launches, as its template names it.
CALLABLE_KIND = "callable"

#: The kind :class:`~threetears.evals.quick.contract.Eval` launches when it is handed a judge: the callable, its answers judged as documents.
JUDGED_CALLABLE_KIND = "callable-judged"

#: The host :func:`callable_host` builds, by the id the engine prints in its logs and errors.
CALLABLE_HOST_ID = "eval"

#: The lever a comparison (an eval whose runner is a mapping of arms) names its arms on, declared by
#: ``callable_host(arms=True)``: each arm's run states its name as its level, so the report calls the arm
#: ``candidate=<name>`` rather than calling the name a model.
ARM_LEVER = "candidate"

#: The candidate model every arm of a comparison (an eval whose runner is a mapping of arms) shares when its arms
#: are named on :data:`ARM_LEVER`: the arms' names are not models, and a lever every arm shares splits none of them.
SHARED_ARM_MODEL = "callable"

#: Where a named arm's run carries its name, on its ``host_payload``, for the host's lever reader.
_ARM_PAYLOAD_KEY = "arm"

#: Where a case rides on its stored test case: verbatim, for the candidate to be handed back.
_CASE_KEY = "case"

#: Where a classifier's case carries its expected label on its stored test case.
_EXPECTED_KEY = "expected"

#: The reserved case key a case's stratum is read from: the kind of case it is, never shown to the runner.
STRATUM_KEY = "stratum"

#: The reserved case key a case's starting state is read from, for a world with state; never shown to the runner.
WORLD_KEY = "world"

#: The reserved case keys: what names a case, what kind it is, and where its world starts. Each is metadata about the
#: case rather than stimulus, so none reaches the runner, a grader or the judge.
RESERVED_CASE_KEYS: frozenset[str] = frozenset({"id", STRATUM_KEY, WORLD_KEY})

#: The callable kind's contract: no overlays, no spec, and no rig seat — nothing in a eval's run is
#: graded by a model or talks to a simulated user. Without the empty seats each blank judge and simulator
#: dimension would read as an unrecoverable judge or simulator and confound every comparison of two such
#: runs. A host of the caller's own declares a contract for the callable kind on its profile too —
#: this one, or one seating apparatus of its own the runs do read — and :class:`~threetears.evals.quick.contract.Eval` refuses a host
#: that does not (:data:`CALLABLE_UNSEATED`).
CALLABLE_KIND_CONTRACT = KindContract(CALLABLE_KIND, seats=frozenset())

#: What a eval's run never has as a level of its rig, whatever host it runs in, so a callable-kind contract
#: may seat none of it: the engine's judge and simulator (by role or by any pinned dimension) — the callable kind
#: is unjudged and simulates nobody — and the spend ceiling, which is off unless a call caps it
#: (``max_cost_usd=``) and is then a condition stated beside the run: a run it stopped says so by its status, and
#: two runs under different caps measured the same candidate the same way until one stopped. A seat here would
#: make every blank such dimension an unrecoverable level, and every comparison of two eval's runs
#: ``undecided``.
CALLABLE_UNSEATED: frozenset[str] = frozenset(
    {
        *(name for role in SHARED_CORE.roles for name in (role.name, *role.pins)),
        "max_cost_usd",
    }
)


#: The judged callable kind's contract: no overlays, no spec, and the engine's judge seated — its runs are
#: graded by a model, so who judged them is part of what two of them are compared on, and a run judged by
#: another model or under other judge configs is a confound rather than a blank. Still no simulator and no
#: spend ceiling (:data:`JUDGED_CALLABLE_UNSEATED`). Declared beside :data:`CALLABLE_KIND_CONTRACT` because a
#: contract is a kind's, and the unjudged kind's empty seats are what keep its runs comparable.
JUDGED_CALLABLE_KIND_CONTRACT = KindContract(JUDGED_CALLABLE_KIND, seats=frozenset({"judge"}))

#: What a judged eval's run never has as a level of its rig: the simulator (by role or by any pinned
#: dimension) and the spend ceiling, a condition beside the run as it is for the unjudged kind. The judge is not
#: here — it is the one seat the judged kind fills.
JUDGED_CALLABLE_UNSEATED: frozenset[str] = frozenset(
    {
        *(name for role in SHARED_CORE.roles if role.name != "judge" for name in (role.name, *role.pins)),
        "max_cost_usd",
    }
)

#: The judge pins a judged callable kind's contract must seat, by role or one by one.
_JUDGE_SEATS = next(role for role in SHARED_CORE.roles if role.name == "judge")


def _launch_settings(arms: int = 1, max_cost_usd: float | None = None) -> LaunchSettings:
    """The launch settings of the one-call path, for a launch of ``arms`` arms, capped at ``max_cost_usd`` a run.

    Every arm of the launch admitted and started together (:class:`~threetears.evals.quick.contract.Eval` launches one,
    a comparison one per arm). With no cap, ceiling enforcement is off and every run
    records that it ran uncapped (``max_cost_usd_origin == "uncapped"``), which its summary says: the candidate
    is an opaque callable whose spend the engine sees only after the fact, and only when it reports it (an
    :class:`~threetears.evals.quick.answer.Answer`), so no ceiling is imposed that nobody asked for. With one,
    enforcement is on and each run's own cost cap (:class:`~threetears.evals.run.EvalRunCostCap`) counts every
    result's reported spend — the candidate's and the judge's — and stops the run ``budget_stopped`` between
    cells once it is over, or at the first result whose spend went unpriced. The cap is also the configured
    ceiling, so the launch, which names it (:func:`~threetears.evals.run.start_run`'s ``max_cost_usd``), may
    not exceed it, and the run records it as ``chosen``. The out-of-run ceiling binds nothing, since the
    callable kind declines ``n_variations`` and so never generates; no metered tools are declared
    (``max_metered_calls=None``). A judge scores one dimension at a time.
    """
    return LaunchSettings(
        max_launch_arms=arms,
        max_admitted_runs=arms,
        judge_concurrency=1,
        enforcement_enabled=max_cost_usd is not None,
        max_cost_usd=1.0 if max_cost_usd is None else max_cost_usd,
        max_metered_calls=None,
        max_out_of_run_cost_usd=1.0 if max_cost_usd is None else max_cost_usd,
    )


def refuse_an_unusable_cap(max_cost_usd: float | None) -> None:
    """Refuse a spend ceiling that is not a positive finite number of US dollars.

    :raises ValueError: A ``max_cost_usd`` that is a bool, not a number, not finite, or not above 0.
    """
    if max_cost_usd is not None and (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, int | float)
        or not math.isfinite(max_cost_usd)
        or max_cost_usd <= 0
    ):
        raise ValueError(f"max_cost_usd= is a spend ceiling in US dollars: a positive number, not {max_cost_usd!r}")


def _scorer_reader_name(name: str) -> str:
    """What a reader calls a scorer's measure: its def's name in words, as a score — ``exact_match`` is "Exact match
    score".

    :param name: The scorer's ``__name__``.
    :ptype name: str
    :return: The reader-facing name its descriptor carries.
    :rtype: str
    """
    words = " ".join(name.replace("_", " ").split())
    return f"{words[:1].upper()}{words[1:]} score"


def scorer_measure(
    scorer: Scorer,
    *,
    margin: float | None = None,
    value_range: tuple[float, float] | None = None,
    guardrail: Guardrail | None = None,
) -> MetricDescriptor:
    """The measure one scorer function reports: a quality score, higher is better, over scored results.

    A scorer whose return is annotated ``bool`` is a pass/fail — a check, ``pass_fail``, whose 0 fails the trial it is
    on — and is declared on the range 0 to 1, margin or not: its intervals stay inside what a pass rate can be, and
    with a margin its equivalence test is the bounded one that holds its error rate at any n. Any other scorer is on
    the range its caller declares (``grader(range=...)``), or on none, since nothing else says what its values can
    be: its intervals are then the unclipped t intervals, and it takes no margin (:func:`refuse_unusable_margins`).

    Declared a guardrail (``guardrail=``), it is instead a measure no arm may get worse on: on no merit axis, so it
    joins no contrast and no composite, with the guardrail's direction and its margin as the measure's
    ``materiality_threshold``.

    :param scorer: The scorer. Its ``__name__`` names the measure and the first line of its docstring,
        when it has one, describes it.
    :ptype scorer: Scorer
    :param margin: The measure's declared margin (``materiality_threshold``): the most a difference in it may be and
        still be too small to act on, so a contrast on it can read ``equivalent``. ``None`` declares none,
        and no contrast on it can.
    :ptype margin: float | None
    :param value_range: The lowest and highest value a non-``bool`` scorer can return, or ``None``.
    :ptype value_range: tuple[float, float] | None
    :param guardrail: Declares the measure a guardrail, with its margin and direction; ``None`` for a quality score.
        A guardrail takes no ``margin`` beside it: its margin is the guardrail's own.
    :ptype guardrail: Guardrail | None
    :return: The descriptor :func:`callable_host` registers for it.
    :rtype: MetricDescriptor
    :raises ValueError: Both ``margin`` and ``guardrail`` are given.
    """
    name = _scorer_name(scorer)
    doc = inspect.getdoc(scorer)
    if guardrail is not None:
        if margin is not None:
            raise ValueError(_MARGIN_ON_A_GUARDRAIL.format(name=name))
        return MetricDescriptor(
            name=name,
            reader_name=_scorer_reader_name(name),
            data_type="numeric",
            family="mechanical",
            transferability_class="mechanical",
            attribution_scope="end_to_end",
            description=doc.splitlines()[0] if doc else f"The score the {name} function gave the candidate's answer.",
            higher_is_better=guardrail.higher_is_better,
            guardrail=True,
            population="scored",
            materiality_threshold=guardrail.margin,
            value_range=(0.0, 1.0) if _returns_bool(scorer) else value_range,
            pass_fail=_returns_bool(scorer),
        )
    return MetricDescriptor(
        name=name,
        # The def's own name in words, marked a score so it never heads a column alike with an engine measure.
        reader_name=_scorer_reader_name(name),
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=doc.splitlines()[0] if doc else f"The score the {name} function gave the candidate's answer.",
        higher_is_better=True,
        merit_axis="quality",
        population="scored",
        materiality_threshold=margin,
        value_range=(0.0, 1.0) if _returns_bool(scorer) else value_range,
        pass_fail=_returns_bool(scorer),
    )


def _as_range(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    """A declared range as two floats, or None when none is declared."""
    return None if bounds is None else (float(bounds[0]), float(bounds[1]))


def _returns_bool(scorer: Scorer) -> bool:
    """Whether ``scorer``'s return is annotated ``bool``, under postponed annotations or not."""
    try:
        returned = inspect.signature(scorer).return_annotation
    except TypeError, ValueError:
        return False
    return returned is bool or returned == "bool"


def refuse_unusable_margins(
    scorers: Sequence[Scorer],
    margins: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    """Refuse a margin or a range no scorer's measure could carry, naming what can carry one.

    A margin is what lets a contrast read ``equivalent``, and the equivalence test needs the measure's range
    (#695). A ``bool`` scorer is on 0 to 1; any other has the range ``ranges`` declares for it, or none, and a
    margin on it with none is refused here rather than accepted and never tested.

    :param scorers: The scorers whose measures the margins are declared on.
    :ptype scorers: Sequence[Scorer]
    :param margins: Each margin, by the name of the scorer it is declared on.
    :ptype margins: Mapping[str, float]
    :param ranges: Each non-``bool`` scorer's range, by its name.
    :ptype ranges: Mapping[str, tuple[float, float]] | None
    :raises ValueError: A margin or range on a name no scorer has (the classifier's accuracy and every engine core
        measure among them), a margin that is not a positive finite number, a range on a ``bool`` scorer or
        one that is not two finite numbers, low below high, or a margin on a scorer with no range.
    """
    ranges = dict(ranges or {})
    names = [_scorer_name(scorer) for scorer in scorers]
    by_name = dict(zip(names, scorers, strict=True))
    for name, bounds in ranges.items():
        if name not in names:
            raise ValueError(
                f"a range names {name!r}, which no grader reports; a range is declared on a grader "
                f"(grader(<function>, range=(1, 5))): {', '.join(map(repr, names)) or 'none were given'}"
            )
        if _returns_bool(by_name[name]):
            raise ValueError(
                f"{name!r} returns a bool, a pass/fail, which is on 0 to 1 already: declare range= only on a "
                "grader returning a number"
            )
        if (
            not isinstance(bounds, tuple | list)
            or len(bounds) != 2
            or any(
                isinstance(end, bool) or not isinstance(end, int | float) or not math.isfinite(end) for end in bounds
            )
            or not bounds[0] < bounds[1]
        ):
            raise ValueError(
                f"the range of {name!r} is the lowest and the highest score it can return, as two numbers, low "
                f"first: grader({name}, range=(1, 5)) for a 1-to-5 score, not {bounds!r}"
            )
    for name, margin in margins.items():
        if name in _CLASSIFIER_NAMES or name in METRIC_DESCRIPTORS:
            raise ValueError(
                f"a margin on {name!r} is not a scorer's: it is an engine core measure, whose descriptor the engine "
                "owns. A margin on a core rate measure (accuracy) is declared on the runs instead: "
                "grader('label', margin=0.05) on the first field grader declares it on every arm's run"
            )
        if name not in names:
            raise ValueError(
                f"a margin names {name!r}, which no grader reports; a margin is declared on a grader "
                f"(grader(<function>, margin=0.05)): {', '.join(map(repr, names)) or 'none were given'}"
            )
        if isinstance(margin, bool) or not isinstance(margin, int | float) or not math.isfinite(margin) or margin <= 0:
            raise ValueError(
                f"the margin on {name!r} is a positive number in the measure's own units (0.05 is five points on a "
                f"pass rate), not {margin!r}"
            )
        if not _returns_bool(by_name[name]) and name not in ranges:
            raise ValueError(
                f"a margin on {name!r} needs its range too: {name!r} does not return a bool, so nothing says what "
                "its scores can be, and with no range no test can show two arms inside a margin (one rare large "
                f"score can hide a move). Declare the lowest and highest score it can return beside the margin: "
                f"grader({name}, margin=..., range=(1, 5)) for a 1-to-5 score. A pass/fail grader annotated -> bool "
                "needs none"
            )


def refuse_a_store_beside_a_host(store: DocumentStore | None, host: EvalHost | None) -> None:
    """Refuse ``store=`` beside ``host=``: a host brings its own storage, and one call stores in one place.

    :raises ValueError: Both were given.
    """
    if store is not None and host is not None:
        raise ValueError(
            f"store= is where the host this call builds keeps the runs, and host {host.profile.host_id!r} brings its "
            "own storage; pass one or the other (as_host([...], store=...) builds a host over a store)"
        )


#: Why a scorer named in both ``margins=`` and ``guardrails=`` is refused, naming it.
_MARGIN_ON_A_GUARDRAIL = (
    "{name!r} is declared a guardrail and given a margin too: a guardrail is never contrasted, so a margin for "
    "equivalence has no use on it, and its own margin is its guard (grader(..., guard=0.02)); drop the margin"
)


def refuse_unusable_guardrails(
    scorers: Sequence[Scorer],
    guardrails: Mapping[str, Guardrail],
    *,
    margins: Mapping[str, float] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    """Refuse a guardrail no grader's measure could carry, naming the ones that can.

    A judged dimension's guardrail is the :class:`~threetears.evals.quick.judged.Judge`'s to declare and refuse
    (``Judge(guard=...)``); this holds a function grader's.

    :param scorers: The graders whose measures a guardrail may be declared on.
    :ptype scorers: Sequence[Scorer]
    :param guardrails: Each guardrail, by the name of the grader it is declared on.
    :ptype guardrails: Mapping[str, Guardrail]
    :param margins: The margins declared beside them, by grader.
    :ptype margins: Mapping[str, float] | None
    :param ranges: The ranges declared for graders returning a number, by grader.
    :ptype ranges: Mapping[str, tuple[float, float]] | None
    :raises ValueError: A guardrail that is not a :class:`~threetears.evals.quick.guardrails.Guardrail`; on an engine
        core measure, or a name no grader has; on a grader also given a margin; or a margin as wide as the grader's
        range (0 to 1 for a pass/fail).
    """
    names = [_scorer_name(scorer) for scorer in scorers]
    by_name = dict(zip(names, scorers, strict=True))
    for name, guardrail in guardrails.items():
        if not isinstance(guardrail, Guardrail):
            raise ValueError(f"a guard is declared as grader(<function>, guard=0.02), not {guardrail!r}")
        if name in _CLASSIFIER_NAMES or name in METRIC_DESCRIPTORS:
            raise ValueError(
                f"a guardrail on {name!r} cannot be declared: it is an engine core measure, read under the core's own "
                "description. Grade what the runner must not get worse on with a function "
                "(def no_leak(case, out) -> bool: ...) and guard that: grader(no_leak, guard=0.02)"
            )
        if name not in by_name:
            raise ValueError(
                f"a guard names {name!r}, which no grader reports; a guard is declared on a grader "
                f"(grader(<function>, guard=0.02)): {', '.join(map(repr, names)) or 'none were given'}"
            )
        if margins and name in margins:
            raise ValueError(_MARGIN_ON_A_GUARDRAIL.format(name=name))
        bounds = (0.0, 1.0) if _returns_bool(by_name[name]) else (ranges or {}).get(name)
        if bounds is not None and guardrail.margin >= bounds[1] - bounds[0]:
            raise ValueError(
                f"a margin of {guardrail.margin!r} on {name!r}, which scores {bounds[0]:g} to {bounds[1]:g}, is "
                "as wide as every value it can return, so every arm would hold whatever it did; declare a "
                "smaller one"
            )


def callable_kind_contracts(levers: Sequence[str] = ()) -> tuple[KindContract, KindContract]:
    """The contracts of both callable kinds, declaring ``levers`` as each run's levels beside its model.

    With no levers, :data:`CALLABLE_KIND_CONTRACT` and :data:`JUDGED_CALLABLE_KIND_CONTRACT` themselves, so a
    host declaring none keys its runs exactly as before. With levers, the same seats and an overlay model
    (:func:`~threetears.evals.quick.levers.levers_model`) whose every field is a lever the engine names
    ``callable.<name>`` (``callable-judged.<name>`` on a judged run), resolves into each run's variant key and
    accepts as a campaign's axis. A host of the caller's own declares these on its profile's ``kinds`` to
    take an arm's levers.

    :param levers: The levers' names.
    :ptype levers: Sequence[str]
    :return: The unjudged kind's contract, then the judged kind's.
    :rtype: tuple[KindContract, KindContract]
    :raises ValueError: A lever name no overlay field could carry, ``model``, or one given twice.
    """
    if not levers:
        return CALLABLE_KIND_CONTRACT, JUDGED_CALLABLE_KIND_CONTRACT
    overlays = levers_model(levers)
    return (
        KindContract(CALLABLE_KIND, overlays=overlays, seats=CALLABLE_KIND_CONTRACT.seats),
        KindContract(JUDGED_CALLABLE_KIND, overlays=overlays, seats=JUDGED_CALLABLE_KIND_CONTRACT.seats),
    )


def _arm_payload(arm: str | None) -> dict[str, str]:
    """The ``host_payload`` a named arm's run carries: its name, under this module's own key; nothing for an unnamed arm."""
    if arm is None:
        return {}
    return {_ARM_PAYLOAD_KEY: arm}


def _arm_of_run(run: EvalRun) -> str:
    """The arm a run is, by name: what its launch stated, else its candidate model's label."""
    named = (run.host_payload or {}).get(_ARM_PAYLOAD_KEY)
    return named if isinstance(named, str) and named else run.candidate_model


def _arm_levels(run: EvalRun) -> dict[str, SweepableValue]:
    """A run's level of :data:`ARM_LEVER`, the host's one lever of its own, keyed into its variant."""
    arm = _arm_of_run(run)
    return {ARM_LEVER: SweepableValue.of(arm, display=arm, scale=NominalScale())}


#: The arm lever as ``callable_host(arms=True)`` declares it.
_ARM_SWEEPABLE = Sweepable(
    name=ARM_LEVER,
    role="lever",
    read=lambda run, _results: _arm_of_run(run),
    reader_prose="the arm of a comparison the run is, by the name the comparison gave it",
    # Which candidate the run is, not a setting it ran at: the arm moves nothing off its own footing.
    departs_production=lambda _run, _results: False,
)


def callable_host(
    scorers: Sequence[Scorer] = (),
    *,
    levers: Sequence[str] = (),
    world: World | None = None,
    arms: bool = False,
    margins: Mapping[str, float] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    store: DocumentStore | None = None,
    clients: CompletionClients | None = None,
    trace_sink: TraceSink | None = None,
    guardrails: Mapping[str, Guardrail] | None = None,
) -> EvalHost:
    """The least host there is: the shared core, one measure per scorer, no world, an in-memory store.

    What :class:`~threetears.evals.quick.contract.Eval` builds when it is handed no host. With no ``store``, its store
    lives as long as the returned value, so a caller wanting to run several candidates into one store and compare them
    builds this once and hands it to each call. With one, the runs are wherever that store keeps them, and a later
    process reads them through ``callable_host(<the same scorers>, store=<the same store>)``. A classifier's ``match``
    and ``confusion_cell`` are core measures, so a host for a classifier with no scorers of its own declares none:
    ``callable_host()``. It declares the contracts of both callable kinds, so one host serves judged and unjudged calls
    alike.

    :param scorers: The scorer functions whose measures the host declares.
    :ptype scorers: Sequence[Scorer]
    :param levers: The levers every run in this host states a level of beside its model
        (:func:`callable_kind_contracts`), each handed to :class:`~threetears.evals.quick.contract.Eval` as ``levers=``.
    :ptype levers: Sequence[str]
    :param world: The world a world run seeds (:class:`~threetears.evals.quick.world.World`), declared on the
        profile; ``None`` declares none.
    :ptype world: World | None
    :param arms: Declare :data:`ARM_LEVER`, the lever a comparison (an eval whose runner is a mapping of arms)
        names its arms on, so its report reads ``candidate=<name>``; what a comparison builds when handed no
        host. A run that states no arm (an eval of one arm in this host) is at its candidate model's label.
    :ptype arms: bool
    :param margins: A margin declared on a scorer's measure, by the scorer's name (:func:`scorer_measure`), so a
        contrast on it can read ``equivalent``; ``None`` declares none. No margin is ever assumed.
    :ptype margins: Mapping[str, float] | None
    :param ranges: The lowest and highest score a scorer returning a number can give, by the scorer's name
        (``{"rating": (1, 5)}``): its intervals stay inside it, and a margin on it can be tested. A ``bool``
        scorer is on 0 to 1 already. A score outside its range excludes its cell, naming the scorer.
    :ptype ranges: Mapping[str, tuple[float, float]] | None
    :param store: Where every document is stored (any :class:`~threetears.evals.schema.DocumentStore`):
        :class:`~threetears.evals.storage.SqliteDocumentStore` keeps the runs in a file. ``None`` builds a fresh
        :class:`~threetears.evals.storage.InMemoryDocumentStore`, which keeps nothing past the process.
    :ptype store: DocumentStore | None
    :param clients: The completion clients the engine's own model calls use after the run (a re-judge, an
        analysis), as :class:`~threetears.evals.kernel.host.EvalHost` takes them; ``None`` supplies none,
        and a call needing them is refused. A :class:`~threetears.evals.quick.Judge` brings its own for the run.
    :ptype clients: CompletionClients | None
    :param trace_sink: Where each cell's trace is sent, as :class:`~threetears.evals.kernel.host.EvalHost` takes
        it; ``None`` traces nothing.
    :ptype trace_sink: TraceSink | None
    :param guardrails: The scorers declared guardrails, by name, each with its margin and direction
        (:class:`~threetears.evals.quick.Guardrail`): measures no arm may get worse on, decided apart from
        every contrast. A guardrail on a judged dimension is the campaign's to declare, so it is
        :class:`~threetears.evals.quick.judged.Judge`'s, not a host's. ``None`` declares none.
    :ptype guardrails: Mapping[str, Guardrail] | None
    :return: The host.
    :rtype: EvalHost
    :raises ValueError: A scorer has no usable name, two share one, or one takes a classifier measure's name or
        any other engine core measure's; a lever name is unusable or repeated; a margin names no scorer,
        is not a positive number or is on a scorer with no range; a range is unusable
        (:func:`refuse_unusable_margins`); or a guardrail names no scorer, is not a ``Guardrail``, is declared
        on a scorer given a margin too, or is as wide as its scorer's range.
    """
    _refuse_unnamed_or_repeated(scorers)
    margins = dict(margins or {})
    ranges = dict(ranges or {})
    if declared_twice := sorted(
        scorer.descriptor.name
        for scorer in scorers
        if isinstance(scorer, Measure) and scorer.descriptor.name in {*margins, *ranges, *(guardrails or {})}
    ):
        raise ValueError(
            f"{', '.join(declared_twice)} declare(s) its margin, range and guardrail on its own @measure; "
            "declare them there rather than in margins=, ranges= or guardrails="
        )
    refuse_unusable_margins(scorers, margins, ranges)
    guardrails = dict(guardrails or {})
    refuse_unusable_guardrails(scorers, guardrails, margins=margins, ranges=ranges)
    return EvalHost(
        profile=HostProfile(
            host_id=CALLABLE_HOST_ID,
            host_sweepables=SweepableRegistry((*CORE_SWEEPABLES, _ARM_SWEEPABLE), roles=CORE_ROLES)
            if arms
            else SHARED_CORE,
            measures=MeasureRegistry(
                # A Measure declares itself; any other scorer is declared here, with what this call declares on it.
                scorer.descriptor
                if isinstance(scorer, Measure)
                else scorer_measure(
                    scorer,
                    margin=margins.get(_scorer_name(scorer)),
                    value_range=_as_range(ranges.get(_scorer_name(scorer))),
                    guardrail=guardrails.get(_scorer_name(scorer)),
                )
                for scorer in scorers
            ),
            kinds=callable_kind_contracts(levers),
            world=None if world is None else world.registry,
            # The world's tools described from their own schemas, so the goal-check gate closes a parameter the
            # way the agent's calls are held to: an enum-closed one may be compared, a free string may not.
            action_parameters=None if world is None or not world.stateful else world.action_parameters,
            tool_actions=None if world is None or not world.stateful else world.tool_actions,
            variant_levers=_arm_levels if arms else None,
        ),
        storage=EvalStorage(InMemoryDocumentStore() if store is None else store),
        failure_describer=withhold_failure_detail,
        trace_sink=trace_sink,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
        clients=clients,
    )


def callable_kind(
    candidate: Candidate | ToolUsingCandidate,
    scorers: Sequence[Scorer] = (),
    *,
    classifies: bool = False,
    label_field: str | None = None,
    judge: Judge | None = None,
    tools: Mapping[str, Tool] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> CandidateKind:
    """The kind over a plain async candidate and its scorers, for a host of your own to launch.

    What :class:`~threetears.evals.quick.contract.Eval` runs each cell through, as an ordinary
    :class:`~threetears.evals.kernel.CandidateKind` a launcher wires like any other (``KindWiring(kind_factory=lambda
    _cell: kind, ...)``), declared on the profile under :data:`CALLABLE_KIND` by :func:`callable_kind_contracts`
    (:data:`JUDGED_CALLABLE_KIND` with a judge). Each cell calls the candidate with its case, read from the stored test
    case's ``host_payload["case"]``, where the quick path stores it, and each scorer with the case and the answer,
    landing each score as a host measure under the scorer's name. A candidate that raises fails its cell; a scorer that
    raises, returns no number, or returns one outside its range excludes it.

    :param candidate: The callable under test.
    :ptype candidate: Candidate | ToolUsingCandidate
    :param scorers: The grades: plain functions, or :class:`~threetears.evals.quick.Measure` objects declaring their own
        measure (``@measure(...)``), each reported under its name.
    :ptype scorers: Sequence[Scorer]
    :param classifies: Whether each case carries an expected label (``host_payload["expected"]``), landing ``match``
        and ``confusion_cell``.
    :ptype classifies: bool
    :param label_field: The field of the answer its label is read from; ``None`` reads the answer itself.
    :ptype label_field: str | None
    :param judge: The judge whose evidence each answer carries, for the judged kind; ``None`` for an unjudged one.
    :ptype judge: Judge | None
    :param tools: The tools the candidate is called with beside each case; ``None`` for a candidate called alone.
    :ptype tools: Mapping[str, Tool] | None
    :param ranges: Each scorer's range, by name, every score is held to; a ``Measure``'s own ``value_range`` when not
        named here.
    :ptype ranges: Mapping[str, tuple[float, float]] | None
    :return: The kind: one instance serves every cell of a run.
    :rtype: CandidateKind
    """
    _refuse_unnamed_or_repeated(scorers)
    held = {
        scorer.descriptor.name: scorer.descriptor.value_range
        for scorer in scorers
        if isinstance(scorer, Measure) and scorer.descriptor.value_range is not None
    }
    return CallableKind(
        candidate,
        scorers,
        classifies=classifies,
        label_field=label_field,
        judge=judge,
        tools=tools,
        ranges={**held, **(ranges or {})},
    )


#: The one role a callable candidate's own spend lands under — what its answer's cost is summed over.
_CANDIDATE_ROLE: tuple[KnownUsageRole, ...] = ("candidate",)


@dataclass(frozen=True)
class _Prepared:
    """One cell's candidate, as ``prepare`` hands it to ``invoke``."""

    model: str
    tools: CellTools | None = None


class CallableKind:
    """The candidate-kind seam over a plain callable and its scorer functions.

    ``invoke`` calls the candidate with the case it was given and each scorer with that case and the
    answer, and reports the scores as host measures, and an :class:`~threetears.evals.quick.answer.Answer`'s
    spend as the cell's ``candidate`` usage. A candidate that raises FAILS its cell (a
    candidate error lowers the score; a broken candidate must not vanish); a scorer that raises, or
    returns something that is not a number, EXCLUDES it, because the grader is the rig rather than
    the thing under test; so does a score outside the range its measure declares. A classifying kind also lands ``match`` and ``confusion_cell`` against the
    expected label its case carries, the answer counted as :data:`UNUSABLE_ANSWER` when it is no label.
    A judged kind is a document kind: every answer carries the evidence its judge reads
    (:func:`~threetears.evals.quick.judged.judge_evidence`), and the engine's judge scores it after ``invoke``.
    A kind with tools hands the candidate each cell's own :class:`~threetears.evals.quick.tools.CellTools`,
    wired to the cell's cassettes when it is handed any; a rig fault a tool call met excludes the cell even
    when the candidate caught it.
    """

    judged_artifact = JudgedArtifact.UNJUDGED
    #: No limit of the kind's own: it holds nothing between cells (each cell's tools are its own
    #: :class:`~threetears.evals.quick.tools.CellTools`), and whether the callable is safe beside itself is its
    #: author's to say — which :class:`~threetears.evals.quick.contract.Eval` asks through ``measure_latency``, its cases
    #: running several at once unless latency is under test.
    max_concurrent_cells: int | None = None

    def __init__(
        self,
        candidate: Candidate | ToolUsingCandidate,
        scorers: Sequence[Scorer],
        *,
        classifies: bool = False,
        label_field: str | None = None,
        judge: Judge | None = None,
        tools: Mapping[str, Tool] | None = None,
        recording: ToolRecording | None = None,
        ranges: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        """Bind the candidate and its scorers.

        :param candidate: The callable under test.
        :ptype candidate: Candidate | ToolUsingCandidate
        :param scorers: The grades, each reported under its own name.
        :ptype scorers: Sequence[Scorer]
        :param classifies: Whether the candidate is a classifier, whose every case carries its expected label.
        :ptype classifies: bool
        :param label_field: The field of the answer its label is read from; ``None`` reads the answer itself, which is
            a label only as a non-blank string.
        :ptype label_field: str | None
        :param judge: The judge whose evidence each answer carries, or ``None`` for an unjudged kind.
        :ptype judge: Judge | None
        :param tools: The tools the candidate is called with beside each case, or ``None`` for a candidate
            called with the case alone. A candidate handed tools answers a
            :class:`~threetears.evals.quick.world.WorldOutcome`: its answer, and the calls it made.
        :ptype tools: Mapping[str, Tool] | None
        :param recording: Where the tools' answers are recorded or replayed from, or ``None`` to call them live.
        :ptype recording: ToolRecording | None
        :param ranges: Each scorer's declared range, by its name, which every score it returns must lie in.
        :ptype ranges: Mapping[str, tuple[float, float]] | None
        """
        self._candidate = candidate
        self._ranges = dict(ranges or {})
        self._tools = dict(tools) if tools is not None else None
        self._scorers = tuple(scorers)
        self._classifies = classifies
        self._label_field = label_field
        self._recording = recording
        self._judge = judge
        self.judged_artifact = JudgedArtifact.UNJUDGED if judge is None else JudgedArtifact.DOCUMENT

    @property
    def calls_tools(self) -> bool:
        """Whether the candidate declares tools, which is what lets a run of this kind record or replay."""
        return self._tools is not None

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: WorldSession | None,
    ) -> _Prepared:
        """Records the arm's model label, and builds the cell's tools, wired to its cassettes when it has any.

        :param subject_snapshot: The run's subject, unused.
        :ptype subject_snapshot: SubjectSnapshot | None
        :param variant_config: The cell's variant; its candidate model labels the arm.
        :ptype variant_config: VariantConfig
        :param world_seed: The template's seed, which this kind has no world to write.
        :ptype world_seed: WorldSeed
        :param span_window: The cell's trace windows, unused.
        :ptype span_window: CellSpanWindow
        :param cassettes: The cell's cassettes in a capture or replay run, which only a kind with tools is
            launched with, and ``None`` with cassettes off.
        :ptype cassettes: CellCassettes | None
        :param world: Unread — a callable has no world to seed, so the cell opens none.
        :ptype world: WorldSession | None
        :return: The prepared candidate.
        :rtype: _Prepared
        """
        tools = CellTools(self._tools, recording=self._recording) if self._tools is not None else None
        if cassettes is not None and tools is not None:
            cassettes.wire(tools)
        return _Prepared(model=variant_config.candidate_model, tools=tools)

    async def invoke(self, instance: _Prepared, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Run the candidate on one case, grade its answer, and report the grades.

        :param instance: What ``prepare`` returned.
        :ptype instance: _Prepared
        :param test_case: The cell's case; the caller's case rides on its ``host_payload``.
        :ptype test_case: EvalTestCase
        :param sink: The cell's sink. The candidate's call cannot be priced before it is made, so the run's cost
            cap is asked first — a cell the cap has already been reached for calls nothing and returns
            empty, which the runner excludes for the cap — and what an
            :class:`~threetears.evals.quick.answer.Answer` reports it cost is counted the moment it returns.
        :ptype sink: CellSink
        :return: The answer as the stored trace, and each scorer's grade as a host measure, beside a
            classifier's ``match`` and ``confusion_cell``, and a judged kind's evidence.
        :rtype: CandidateOutput
        """
        case = test_case.host_payload[_CASE_KEY]
        try:
            # Unpriceable, so held to the cap by asking (``CellSink.reserve`` with no ceiling holds nothing).
            held = sink.reserve("candidate", None)
        except CostCapReached:
            return CandidateOutput()
        if instance.tools is not None:
            instance.tools.bind_case(case)
        try:
            # Every Model call the runner makes is added to this call's own ledger, and lands as its spend.
            with cell_spend() as ledger:
                answer, telemetry = unwrap_answer(await _call_candidate(self._candidate, case, instance.tools))
            telemetry = spent(ledger, telemetry)
        except ApparatusError:
            raise  # the rig's fault under the candidate — a replay miss — is the engine's to exclude the cell on
        # prawduct:ok-broad-except — the candidate is the caller's code under test: whatever it raises is its failure, recorded on the cell
        except Exception as raised:
            missed = _missed(test_case.host_payload[_EXPECTED_KEY]) if self._classifies else {}
            return CandidateOutput(
                candidate_errors=[f"the candidate raised {type(raised).__name__}: {raised}"],
                host_measures=missed,
                # One call, and it answered nothing: a refusal's round trip is no turn's time or spend.
                telemetry=CandidateTelemetry(turns_delivered=0),
            )
        else:
            # What the answer says it cost, counted for every cell's question of the cap now rather than when the
            # result lands — the judge phase still to come is paid for beside it. A plain answer reports nothing.
            held.settle(blended_cost(telemetry.usage, _CANDIDATE_ROLE) if telemetry.usage else 0.0)
        finally:
            # A call that raised, or was cancelled, settled nothing; a settled reservation ignores this.
            held.release()
        # One call, answered: the kind's whole turn.
        telemetry = telemetry.model_copy(update={"turns_delivered": 1})
        evidence: JudgeEvidence | None = None
        if self._judge is not None:
            try:
                expected = test_case.host_payload[_EXPECTED_KEY] if self._classifies else None
                judged = answer.answer if isinstance(answer, WorldOutcome) else answer
                evidence = judge_evidence(self._judge, case, judged, expected=expected)
            except ValueError as unrenderable:
                # The judge's material is the rig: the cell is excluded, and with no evidence to read it
                # stores no answer either, since a judged kind's every stored answer carries its evidence.
                return CandidateOutput(infra_errors=[str(unrenderable)], telemetry=telemetry)
        trace = [_as_stored(answer)]
        measures: dict[str, bool | float | str] = {}
        if self._classifies:
            expected = test_case.host_payload[_EXPECTED_KEY]
            predicted = predicted_label(answer, self._label_field)
            measures[MATCH_MEASURE] = predicted == expected
            measures[CONFUSION_CELL_MEASURE] = confusion_cell(expected, predicted)
        for scorer in self._scorers:
            name = _scorer_name(scorer)
            try:
                score = scorer(case, answer)
            # prawduct:ok-broad-except — a scorer is the caller's grader, the rig: a raise excludes the cell and says which scorer
            except Exception as raised:
                return CandidateOutput(
                    output=trace,
                    infra_errors=[f"the scorer {name} raised {type(raised).__name__}: {raised}"],
                    judge_evidence=evidence,
                    telemetry=telemetry,
                )
            # ``bool`` is an ``int``, so True and False pass here as 1 and 0.
            if not isinstance(score, int | float) or not math.isfinite(score):
                return CandidateOutput(
                    output=trace,
                    infra_errors=[f"the scorer {name} returned {score!r}, not a finite number or a bool"],
                    judge_evidence=evidence,
                    telemetry=telemetry,
                )
            if name in self._ranges and not self._ranges[name][0] <= score <= self._ranges[name][1]:
                low, high = self._ranges[name]
                return CandidateOutput(
                    output=trace,
                    infra_errors=[
                        f"the scorer {name} returned {score!r}, outside the range {low:g} to {high:g} declared for it"
                    ],
                    judge_evidence=evidence,
                    telemetry=telemetry,
                )
            measures[name] = float(score)
        return CandidateOutput(output=trace, host_measures=measures, judge_evidence=evidence, telemetry=telemetry)


def _missed(expected: str) -> dict[str, bool | float | str]:
    """A classifier's verdict on a case its candidate raised on: a miss, predicted as no usable label.

    A raise — a refusal, a provider error — answered nothing, so a classifier counts it as it counts a
    blank answer: ``match`` False, and the expected label's confusion cell under :data:`UNUSABLE_ANSWER`.
    Landed rather than left absent because an absent ``match`` is in no rate: an arm that refused the
    cases it would have got wrong read more accurate than one that answered them, and its refusals were in
    none of its precision or recall.

    :param expected: The case's expected label.
    :ptype expected: str
    :return: The measures to land beside the failure.
    :rtype: dict[str, bool | float | str]
    """
    return {MATCH_MEASURE: False, CONFUSION_CELL_MEASURE: confusion_cell(expected, UNUSABLE_ANSWER)}


async def _call_candidate(candidate: Candidate | ToolUsingCandidate, case: Any, tools: CellTools | None) -> Any:
    """Call the candidate on one case — with its tools, when it declares any — and surface a rig fault it met.

    A candidate handed no tools may be synchronous: it is called in a worker thread, so a blocking call does
    not stall the event loop the run's other arms and its timeouts run on, and whatever it returns is awaited
    when it is awaitable. Cells within one run are serial, so it is never called beside itself in one arm.

    :raises ApparatusError: A tool call met a rig fault, re-raised here even when the candidate caught it.
    """
    if tools is None:
        if _is_async_callable(candidate):
            return await cast(Candidate, candidate)(case)
        returned = await asyncio.to_thread(cast(Callable[[Any], Any], candidate), case)
        return await returned if inspect.isawaitable(returned) else returned
    try:
        returned = await cast(ToolUsingCandidate, candidate)(case, tools.for_candidate())
    except Exception:
        tools.raise_any_fault()
        raise
    tools.raise_any_fault()
    # What a grader sees of an agent on a world of lookups: its answer, and the calls it made.
    if isinstance(returned, Answer):
        return replace(returned, value=WorldOutcome(answer=returned.value, calls=tools.calls))
    return WorldOutcome(answer=returned, calls=tools.calls)


def _is_async_callable(candidate: object) -> bool:
    """Whether calling ``candidate`` returns a coroutine: an ``async def``, a partial of one, or an object with one as
    its ``__call__``.

    :param candidate: The callable.
    :ptype candidate: object
    :return: ``True`` for an async callable; ``False`` for a plain function, which a call runs to its end.
    :rtype: bool
    """
    if inspect.iscoroutinefunction(candidate):
        return True
    call = getattr(type(candidate), "__call__", None)
    return not inspect.isroutine(candidate) and inspect.iscoroutinefunction(call)


def _refuse_a_sync_candidate_handed_tools(arms: Sequence[CallableArm], *, why: str | None) -> None:
    """Refuse, before anything runs, a synchronous candidate that would be handed async tools it cannot await.

    :param arms: The arms.
    :ptype arms: Sequence[CallableArm]
    :param why: What the candidates are handed (``"tools"``, ``"a world's tools"``), or ``None`` when nothing is,
        and a synchronous candidate is called in a thread.
    :ptype why: str | None
    """
    if why is None:
        return
    if sync := [
        getattr(arm.candidate, "__name__", None) or repr(arm.candidate)
        for arm in arms
        if not _is_async_callable(arm.candidate)
    ]:
        raise ValueError(
            f"{', '.join(sync)} is not an async function, and a candidate handed {why} must be one: its tools are "
            "async functions it awaits. Declare it `async def candidate(case, tools): ...` and await each tool call"
        )


def _as_stored(answer: Any) -> dict[str, Any]:
    """The answer as one stored JSON document: verbatim when JSON can hold it, its ``repr`` when not.

    The two keys differ so a reader of the trace can tell a value from a rendering of one. A pydantic model is
    stored as its JSON.
    """
    if isinstance(answer, BaseModel):
        return {"value": answer.model_dump(mode="json")}
    try:
        json.dumps(answer)
    except TypeError, ValueError:
        return {"repr": repr(answer)}
    return {"value": answer}


def _scorer_name(scorer: Scorer) -> str:
    return getattr(scorer, "__name__", "")


def _refuse_unnamed_or_repeated(scorers: Sequence[Scorer]) -> None:
    names = [_scorer_name(scorer) for scorer in scorers]
    if unnamed := [repr(scorer) for scorer, name in zip(scorers, names, strict=True) if not name.isidentifier()]:
        raise ValueError(
            f"a scorer's __name__ names its measure, and {', '.join(unnamed)} has none a measure can carry; "
            "write each scorer as a def"
        )
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"scorers named {', '.join(repeated)} more than once; each name is one measure")
    if taken := sorted(set(names) & _CLASSIFIER_NAMES):
        raise ValueError(
            f"a scorer named {', '.join(taken)} takes a measure the classifier track owns; to grade a classifier, "
            "grade it with a field grader (grader=['label']), and name any other scorer something else"
        )
    if core := sorted(set(names) & set(METRIC_DESCRIPTORS)):
        raise ValueError(
            f"a scorer named {', '.join(core)} takes the name of an engine core measure, so its grades would be read "
            "under the core's meaning, direction and range and pooled into the engine's own observations of it; "
            "rename the scorer's def (for example, " + ", ".join(f"{name}_grade" for name in core) + ")"
        )


def _expected_labels(cases: list[dict[str, Any]], expected: ExpectedLabel, names: list[str]) -> list[str]:
    """Each case's expected label, refusing a case whose label no confusion matrix could hold."""
    labels: list[str] = []
    for index, case in zip(names, cases, strict=True):
        try:
            label = expected(case)
        # prawduct:ok-broad-except — expected= is the caller's code: what it raises on a case is refused with that case named
        except Exception as raised:
            raise ValueError(f"case {index}: {type(raised).__name__}: {raised}") from raised
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"case {index} expects the label {label!r}; an expected label is a non-blank string")
        if label == UNUSABLE_ANSWER:
            raise ValueError(
                f"case {index} expects {UNUSABLE_ANSWER!r}, the label an unusable answer is counted under; a case "
                "expecting it would count an answer the classifier could not give as a match"
            )
        labels.append(label)
    return labels


def _refuse_an_undeclared_callable_contract(host: EvalHost, *, judged: bool) -> None:
    """Refuse a caller's host that has not declared what a eval's run's rig holds.

    A host with no contract for the kind holds its runs to every apparatus dimension
    (:meth:`~threetears.evals.kernel.host.profile.HostProfile.kind_contract`), so the blank simulator of
    every such run — and the blank judge of every unjudged one — reads as unrecoverable and confounds every
    comparison of two of them, silently. A judged kind's contract that leaves the judge unseated is the
    other half of that: a change of judge between two runs would read as no change at all. Both are
    refused here, before anything is stored, rather than discovered in an analysis.
    """
    profile = host.profile
    # The remedy names what a host imports from the ``quick`` root: the judged contract is not exported there, and
    # ``callable_kind_contracts()`` hands it over beside the unjudged one.
    kind, model = (
        (JUDGED_CALLABLE_KIND, "the judged contract callable_kind_contracts() returns")
        if judged
        else (CALLABLE_KIND, "CALLABLE_KIND_CONTRACT")
    )
    unseated = JUDGED_CALLABLE_UNSEATED if judged else CALLABLE_UNSEATED
    contract = next((declared for declared in profile.kinds if declared.kind == kind), None)
    remedy = (
        f"declare the {kind!r} kind's contract on its profile's kinds — {model}, or a "
        f"KindContract({kind!r}, seats=...) seating only apparatus of the host's own that the runs read"
        + (", beside the judge" if judged else "")
    )
    if contract is None or contract.seats is None:
        what = "no contract" if contract is None else "a contract that declares no seats"
        raise ValueError(
            f"host {profile.host_id!r} has {what} for the {kind!r} kind, so every eval's run would be held to "
            f"the apparatus it never has and every comparison of two would read undecided; {remedy}"
        )
    if seated := sorted(contract.seats & unseated):
        never = (
            "simulates nobody, and its spend ceiling is a condition beside the run"
            if judged
            else "is unjudged, simulates nobody, and its spend ceiling is a condition beside the run"
        )
        raise ValueError(
            f"host {profile.host_id!r} seats {', '.join(seated)} for the {kind!r} kind, which a eval's run never "
            f"has — it {never}; {remedy}"
        )
    if judged and _JUDGE_SEATS.name not in contract.seats and not set(_JUDGE_SEATS.pins) <= contract.seats:
        raise ValueError(
            f"host {profile.host_id!r} does not seat the judge for the {kind!r} kind, so two runs judged by "
            f"different models would compare as judged alike; {remedy}"
        )
    # The one overlay model a callable kind takes is a levers model: its levels are what an arm's levers turn.
    if (
        contract.overlays is not None and not issubclass(contract.overlays, CallableLevers)
    ) or contract.spec is not None:
        raise ValueError(
            f"host {profile.host_id!r} declares overlays or a spec for the {kind!r} kind, which an eval "
            f"neither turns nor states (its levers are declared with callable_kind_contracts); {remedy}"
        )


def _refuse_undeclared_measures(host: EvalHost, scorers: Sequence[Scorer]) -> None:
    measures = host.profile.measures
    if undeclared := [name for scorer in scorers if measures.get(name := _scorer_name(scorer)) is None]:
        raise ValueError(
            f"host {host.profile.host_id!r} declares no measure named {', '.join(undeclared)}; a host's scorers "
            "report into measures it declares, so register them on its profile's measures"
        )


def _case_names(cases: list[dict[str, Any]]) -> list[str]:
    """What each case is called in its results and in every error line: its own ``id``, else its position.

    :raises ValueError: An ``id`` that is not a non-blank string or an int, or two cases called alike.
    """
    names: list[str] = []
    for index, case in enumerate(cases):
        if "id" not in case:
            names.append(str(index))
            continue
        given = case["id"]
        if isinstance(given, bool) or not isinstance(given, str | int) or not str(given).strip():
            raise ValueError(
                f"case {index}'s id is {given!r}; a case's id names it in its results and errors, so it is a "
                "non-blank string or an int"
            )
        names.append(str(given))
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(
            f"more than one case is called {', '.join(map(repr, repeated))}; a case is called by its id, or by its "
            "position in the list when it has none, and each name must be one case's"
        )
    return names


def _plain_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The cases as plain JSON objects."""
    if isinstance(cases, Mapping | str) or not cases:
        raise ValueError("an eval needs a non-empty list of cases, each a mapping")
    plain: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not all(isinstance(key, str) for key in case):
            raise ValueError(f"case {index} is not a mapping with string keys: {case!r}")
        plain.append(dict(case))
    return plain


def _case_strata(cases: list[dict[str, Any]], names: list[str]) -> tuple[list[dict[str, Any]], list[str | None] | None]:
    """Each case with its ``stratum`` taken out of it, and the stratum each declared; the cases as given when none does.

    The stratum is metadata about the case, never stimulus: a case that says it is a lookalike tells the runner what
    to look out for. So it leaves the case before anything is handed it — the runner, every grader and the judge —
    and before the case set is addressed, so re-labelling a case keeps the case, its template and the history of
    every run over it.

    :raises ValueError: A case whose ``stratum`` is neither absent, ``None``, nor a non-blank string.
    """
    if not any(STRATUM_KEY in case for case in cases):
        return cases, None
    stripped: list[dict[str, Any]] = []
    strata: list[str | None] = []
    for name, case in zip(names, cases, strict=True):
        stratum = case.get(STRATUM_KEY)
        if stratum is not None and (not isinstance(stratum, str) or not stratum.strip()):
            raise ValueError(
                f"case {name}'s stratum is {stratum!r}; a stratum is a non-blank string naming the kind of case it is, "
                "or absent (or None) for a case that declares none"
            )
        strata.append(stratum)
        stripped.append({key: value for key, value in case.items() if key != STRATUM_KEY})
    return stripped, strata


def _template_id(
    cases: list[dict[str, Any]],
    labels: list[str] | None,
    judge: Judge | None,
    seeds: list[dict[str, Any]] | None = None,
    goal_checks: Sequence[str] = (),
    *,
    intent: str | None = None,
) -> str:
    """The id the case set is addressed by: its cases, a classifier's expected labels and a judge's rubric with them.

    A classifier's labels are part of what its runs measure, so two classifier calls over one case list
    share a template only when they expect the same labels, and neither shares one with a scorer call.
    A rubric is the template's, so the same holds of two judged calls: one template per rubric, and none
    shared with an unjudged call. The judge's model is not part of it: that is the run's apparatus, which
    two runs of one template are compared on. A world run's starting states and goal checks are its cases'
    and its template's, so they are addressed too. A judged run's intent is what its judge reads as each
    case's task, so two judged calls that state different intents never share, or overwrite, one template.
    """
    addressed: Any = cases
    if labels is not None or judge is not None or seeds is not None:
        addressed = {"cases": cases}
        if labels is not None:
            addressed["expected"] = labels
        if judge is not None:
            # Less each dimension's judge declaration, which a Judge states for itself (``"builtin"`` unless a
            # dimension given says otherwise): it says who scores the rubric, not what the rubric is, so a case set's
            # id stays the one it was before dimensions declared their judge.
            addressed["rubric"] = [dim.model_dump(mode="json", exclude={"judge"}) for dim in judge.dims]
            addressed["intent"] = intent
        if seeds is not None:
            addressed["seeds"] = seeds
            addressed["goal_checks"] = list(goal_checks)
    try:
        digest = canonical_digest(addressed)
    except TypeError as unencodable:
        raise ValueError(f"every case must be JSON: {unencodable}") from unencodable
    return f"{CALLABLE_HOST_ID}-{digest[:16]}"


_LIFECYCLE_STAMPS: set[str] = {"created_at", "updated_at"}


def _save_unless_unchanged(host: EvalHost, template: EvalTemplate, test_cases: Sequence[EvalTestCase]) -> None:
    """Store the template and its cases, leaving each one the store already holds unchanged as it is.

    Both are addressed by their content, so a second call over the same cases finds them stored. Saving them again
    would restamp ``updated_at``, and a template stamped after a run launched reads as edited since: that run's
    summary withholds its intent and a rejudge of it is refused. So a stored definition that matches this one in
    everything but its stamps is left alone, and one that differs (an unjudged template's intent, say) is a real
    edit, saved with its original ``created_at``.

    A stored case's ``name`` is fill-only: a call fills it when the store holds none and never overwrites one, so a
    case renamed since (:func:`~threetears.evals.ops.case_rename`) keeps its name through the next call.
    """
    stored = host.storage.load_template(template.id, template.scope_id)
    if stored is None or stored.model_dump(mode="json", exclude=_LIFECYCLE_STAMPS) != template.model_dump(
        mode="json", exclude=_LIFECYCLE_STAMPS
    ):
        host.storage.save_template(
            template if stored is None else template.model_copy(update={"created_at": stored.created_at})
        )
    for test_case in test_cases:
        kept = host.storage.load_test_case(test_case.id, test_case.scope_id)
        if kept is not None and kept.name is not None:
            test_case = test_case.model_copy(update={"name": kept.name})
        if kept is None or kept.model_dump(mode="json", exclude=_LIFECYCLE_STAMPS) != test_case.model_dump(
            mode="json", exclude=_LIFECYCLE_STAMPS
        ):
            host.storage.save_test_case(
                test_case if kept is None else test_case.model_copy(update={"created_at": kept.created_at})
            )


def _seed_judge_examples(host: EvalHost, template: EvalTemplate, judge: Judge) -> None:
    """Store ``judge``'s examples as authored cases of the judge template ``<template id>-judge``, in its case set.

    The examples are asked under ``template``'s intent, as its answers are, so a judge campaign over them measures
    the judge on the question the run asks it. The judge template and the case set are both named
    ``<template id>-judge``; seeding the same examples again names the stored cases, and a changed list mints the
    set's next version, listing exactly the examples given.
    """
    examples = EvalTemplate(
        id=f"{template.id}-judge",
        scope_id=template.scope_id,
        name=f"judge examples for {template.id}",
        intent=f"Grade answers to {template.id} as its judge should, on the examples written for it.",
        candidate_kind=JUDGE_KIND,
    )
    _save_unless_unchanged(host, examples, [])
    freeze_judge_cases(
        host.storage,
        template=examples,
        scope_id=template.scope_id,
        authored=judge.authored_cases(intent=template.intent),
        case_set=examples.id,
    )


def _case_payload(case: dict[str, Any], label: str | None, seed: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``host_payload`` a case's stored test case carries: the case, verbatim, a classifier's expected label, and a world case's starting state."""
    labelled = {} if label is None else {_EXPECTED_KEY: label}
    seeded = {} if seed is None else world_case_payload(seed)
    return {_CASE_KEY: case, **labelled, **seeded}


@dataclass(frozen=True)
class CallableArm:
    """One arm a one-call launch runs: the candidate, the model its run is labelled by, and its other levers.

    :ivar candidate: The candidate under test.
    :ivar model: The model the caller named for the arm, stored as its run's candidate model and recorded as chosen;
        ``None`` when the caller named none.
    :ivar levers: The arm's level of every other lever, by name, or ``None`` for none.
    :ivar arm: The arm's name, stated on its run as its level of :data:`ARM_LEVER` (and as its subject), for a
        host that declares that lever (``callable_host(arms=True)``); ``None`` names no arm.
    :ivar default_model: What an arm that names no model is labelled by, stored as its run's candidate model and
        recorded as inherited: the candidate runs as itself, so nothing moved it off its own model. ``None``
        takes the candidate's ``__name__``.
    """

    candidate: Candidate | ToolUsingCandidate | WorldAgent | Log
    model: str | None = None
    levers: Mapping[str, str] | None = None
    arm: str | None = None
    default_model: str | None = None


@dataclass(frozen=True)
class _WiredArm:
    """One arm as its launcher wires it: the levels its run is launched at, and the factory building its cells' kind."""

    model: str
    levers: dict[str, Any]
    kind_factory: KindFactory
    arm: str | None = None
    shared_subject: str | None = None
    #: Whether the caller named ``model``. One it did not is launched as the kind's default, so the run records
    #: its model inherited rather than chosen, and no surface reads the label as a model moved off production.
    named: bool = True

    @property
    def requested_model(self) -> str | None:
        """The model the launch names for this arm: the caller's, or ``None`` to run on the kind's default."""
        return self.model if self.named else None

    @property
    def subject(self) -> str:
        """The run's subject: the one every arm of the launch shares when it names one, else the arm's name, else its model."""
        if self.shared_subject is not None:
            return self.shared_subject
        return self.arm if self.arm is not None else self.model


def _launch_host(
    host: EvalHost,
    arms: Sequence[_WiredArm],
    cases: list[EvalTestCase],
    judge: Judge | None,
    world: World | None = None,
    *,
    max_cost_usd: float | None = None,
    preparing: Sequence[_WiredArm] = (),
) -> LaunchHost:
    """``host`` as a launching host whose one kind runs each arm's kind over ``cases``, judged by ``judge`` when given.

    Every arm is launched into one group (:func:`_launch_arms`), each through its own
    :func:`~threetears.evals.run.start_run` call, so the launcher is asked once per arm and wires the arm the
    request names: the one at the request's candidate model, overlays and subject. Arms that share one subject
    and differ only on the arm lever, which rides on the run's payload rather than the request, are told apart
    by ``preparing``: the arm whose ``start_run`` call is in progress, which the caller sets before each call (a
    launch into a caller's group prepares its one run inside that call).

    A judged kind's launcher builds its judge as every judged launcher does, with
    :func:`~threetears.evals.run.build_judge_service` over the run's template, on a host whose client
    factory lends the judge's client — so the run's judge pin, its per-dimension attribution and the
    service that scores it come from one resolution.
    """
    declared: WorldRegistry | None = host.profile.world
    kind_name = CALLABLE_KIND if judge is None else JUDGED_CALLABLE_KIND

    def place(_run: EvalRun) -> dict[str, WorldPlacement]:
        assert declared is not None
        if world is None or not world.stateful:
            # A run with no world of state attaches no carrier, so every dimension a host's world declares is out of
            # play.
            return declared.place(seeded=(), carriers=())
        # Every case of a world run sets every dimension, through the one carrier the world is.
        return declared.place(seeded=world.dimension_names, carriers=(world.name,))

    def arm_of(request: LaunchRequest) -> _WiredArm:
        levels = {} if request.overlays is None else request.overlays.model_dump(mode="json")
        for arm in preparing or arms:
            if (
                arm.requested_model == request.candidate_model
                and arm.levers == levels
                and arm.subject == request.subject_id
            ):
                return arm
        raise ValueError(f"no arm of this launch runs model {request.candidate_model!r} at levers {levels!r}")

    async def launch(request: LaunchRequest) -> EvalRun:
        arm = arm_of(request)
        run_judge: RunJudge | None = None
        if judge is not None:
            # It reads the judge configs the template names from the store, so it runs off the loop.
            run_judge = await run_blocking(
                host.blocking_executor,
                partial(
                    build_judge_service,
                    replace(host, clients=judge.clients()),
                    request.template,
                    judge.model,
                    judged_artifact=JudgedArtifact.DOCUMENT,
                ),
            )
        subject = SubjectSnapshot(subject_id=arm.subject, subject_label=arm.subject, state={})
        return await launch_run(
            launch_host,
            request,
            KindWiring(
                kind_factory=arm.kind_factory,
                subject=subject,
                test_cases=cases,
                judge=run_judge,
                payload=_arm_payload(arm.arm),
                default_candidate_model=None if arm.named else arm.model,
                # What every kind the factory builds declares — the callable kind's, which a world cell adopts.
                max_concurrent_cells=CallableKind.max_concurrent_cells,
            ),
        )

    # The judge pin is the one launch argument a judged kind honours: an eval names its judge's model. No cassette
    # mode is: a world of lookups records and replays its own answers (ToolRecording), never the engine's corpus.
    unhonoured: set[LaunchArgument] = {"simulator_model", "judge_config_ids", "n_variations", "cassette_mode"}
    if judge is None:
        unhonoured.add("judge_model")

    def resolve_subject(subject_id: str, _scope_id: str) -> UnknownSubject | None:
        # A one-call run's subjects are its arms' own, named by the caller as it launched them.
        held = {arm.subject for arm in (*arms, *preparing)}
        if subject_id in held:
            return None
        return UnknownSubject(remedy=f"this call's arms measure {', '.join(sorted(held))}")

    # A one-call template states its intent, its goal checks and, judged, its rubric; a world reaches a cell through
    # each case's own seed. Every other field would be stated and never acted on.
    ignored: set[TemplateField] = {
        "tools_required",
        "tools_allowed",
        "variation_axes",
        "conversation",
        "world_seed",
        "preconditions",
        "goal_check_controls",
    }
    if judge is None:
        ignored.add("rubric")
    launch_host = LaunchHost(
        eval_host=host,
        kinds={
            kind_name: LaunchableKind(
                launch=launch,
                resolve_subject=resolve_subject,
                unhonoured_launch_arguments=frozenset(unhonoured),
                ignored_template_fields=frozenset(ignored),
            )
        },
        settings=partial(_launch_settings, len(arms), max_cost_usd),
        job_timeout_factory=default_job_timeout,
        world_placements=place if declared is not None else None,
    )
    return launch_host


def _world_seeds(
    world: World | None,
    seeds: Sequence[Any] | None,
    goal_checks: Sequence[str],
    host: EvalHost | None,
    names: list[str],
) -> list[dict[str, Any]] | None:
    """Each case's starting state for a world with state, or ``None`` for any other run, refusing an incoherent one.

    :param world: The runner's world, or ``None``.
    :ptype world: World | None
    :param seeds: Each case's ``world``, as the cases carried it (``None`` where a case carried none), or ``None`` when
        no case carried one.
    :ptype seeds: Sequence[Any] | None
    :param goal_checks: The expression graders.
    :ptype goal_checks: Sequence[str]
    :param host: The caller's own host, or ``None``.
    :ptype host: EvalHost | None
    :param names: What each case is called.
    :ptype names: list[str]
    :raises ValueError: A goal check, or a case's ``world``, with no world with state to read or seed; a case of a world
        with state that carries no ``world``, or one the state refuses; a goal check reading state or naming a
        tool the world lacks; or a given host that does not declare the world.
    """
    if world is None or not world.stateful:
        if goal_checks:
            raise ValueError(
                f"the expression graders {', '.join(map(repr, goal_checks))} read a world's state, and the runner "
                "acts on none: run the agent on a world with state (World(State, tools=[...])(agent))"
            )
        if seeds is not None:
            raise ValueError(
                "the cases carry a starting state under 'world', and the runner acts on no world with state: run the "
                "agent on one (World(State, tools=[...])(agent)), or rename the key"
            )
        return None
    if host is not None and host.profile.world is not world.registry:
        raise ValueError(
            f"host {host.profile.host_id!r} does not declare world {world.name!r}; build it with as_host over the "
            "evals that run on it, so the run's world and its host's are one"
        )
    world.refuse_unreadable(goal_checks)
    given = list(seeds) if seeds is not None else [None] * len(names)
    if missing := [name for name, seed in zip(names, given, strict=True) if seed is None]:
        raise ValueError(
            f"a world with state seeds each case's starting state from its 'world', and case(s) "
            f"{', '.join(missing)} carry none"
        )
    return [world.refuse_unseedable(seed, case=name) for name, seed in zip(names, given, strict=True)]


def _kind_factory(
    candidate: Candidate | WorldAgent,
    scorers: Sequence[Scorer],
    *,
    classifies: bool,
    label_field: str | None,
    judge: Judge | None,
    world: World | None,
    recording: ToolRecording | None,
    ranges: Mapping[str, tuple[float, float]],
) -> KindFactory:
    """The factory building each cell's kind: one :class:`CallableKind` for every cell, or a world cell's own kind."""
    if world is None or not world.stateful:
        kind = CallableKind(
            cast(Candidate | ToolUsingCandidate, candidate),
            scorers,
            classifies=classifies,
            label_field=label_field,
            judge=judge,
            tools=None if world is None else world.functions,
            recording=recording,
            ranges=ranges,
        )
        return lambda _cell: kind
    stateful = world

    def over(acting: Candidate) -> CallableKind:
        return CallableKind(acting, scorers, classifies=classifies, label_field=label_field, judge=judge, ranges=ranges)

    def cell_kind(context: CellContext) -> WorldCellKind:
        return WorldCellKind(stateful, cast(WorldAgent, candidate), over, context)

    return cell_kind


def _logged_kind_factory(
    log: Log,
    logged: Mapping[str, list[Answer]],
    case_of: Mapping[str, str],
    scorers: Sequence[Scorer],
    *,
    classifies: bool,
    label_field: str | None,
    judge: Judge | None,
    ranges: Mapping[str, tuple[float, float]],
) -> KindFactory:
    """The factory building each cell's kind for a logged runner: the cell's case and repeat served from the log."""

    def cell_kind(context: CellContext) -> CallableKind:
        served = log.runner(logged, case_of[context.test_case.id], context.k_iteration)
        return CallableKind(served, scorers, classifies=classifies, label_field=label_field, judge=judge, ranges=ranges)

    return cell_kind


def _runner_doc(candidate: object) -> str | None:
    """The docstring a runner states what each case asks in: a logged runner states none."""
    return None if isinstance(candidate, Log) else inspect.getdoc(candidate)


def _template_intent(arms: Sequence[CallableArm], graded_by: str, intent: str | None) -> tuple[str, str | None]:
    """What the one template every arm shares says its cases ask, and where that came from.

    ``intent`` when given; else the candidates' docstring, when every arm's has one and they share its
    first line; else a generic sentence.

    :return: The intent, and its source as a summary names it: ``None`` for the ``intent`` given, else a phrase
        saying which docstring it was read from or why it is the generic sentence.
    :rtype: tuple[str, str | None]
    """
    if intent is not None:
        return intent, None
    firsts = {doc.splitlines()[0] if (doc := _runner_doc(arm.candidate)) else None for arm in arms}
    first = next(iter(firsts)) if len(firsts) == 1 else None
    if len(arms) > 1:
        if first:
            return first, "from the docstring every arm's candidate shares"
        lacking = "the arms' runners share no docstring first line"
    else:
        name = getattr(arms[0].candidate, "__name__", None) or arms[0].model or "the candidate"
        if first:
            return first, f"from {name}'s docstring"
        lacking = f"{name} has no docstring"
    return f"Answer each case so that {graded_by} the answer well.", f"a generic default: {lacking}"


async def run_arms(
    cases: Sequence[Mapping[str, Any]],
    arms: Sequence[CallableArm],
    scorers: Sequence[Scorer] = (),
    *,
    scope_id: str,
    host: EvalHost,
    template_name: str,
    expected: ExpectedLabel | None = None,
    label_field: str | None = None,
    judge: Judge | None = None,
    intent: str | None = None,
    world: World | None = None,
    goal_checks: Sequence[str] = (),
    k: int = DEFAULT_LAUNCH_K_RUNS,
    max_cost_usd: float | None = None,
    subject: str | None = None,
    margins: Mapping[str, float] | None = None,
    measure_latency: bool = False,
) -> list[EvalSummary]:
    """Run every arm over every case ``k`` times as ONE launch, and summarise each arm's run, in arm order.

    The arms share the template, the case set, the host and the launch: one launch group, every arm's run prepared
    (every refusal made) before any starts, and all of them started together
    (:func:`~threetears.evals.run.launch_as_group`) — so a comparison's arms are measured side by side, and a refusal
    on the last arm leaves none run. Each arm is its own :func:`~threetears.evals.run.start_run` call into that group,
    because ``start_run`` launches every run it starts at one set of overlays and the arms of a factorial differ in
    theirs.

    :param cases: The cases, each a JSON object. Its reserved keys (:data:`RESERVED_CASE_KEYS`) are read and withheld:
        ``id`` names it in its results, its errors and on the report's page (a case with none is called by its
        position, from ``0``, and numbered on the page),
        ``stratum`` is the kind of case it is, stored as its test case's stratum so the summary reads every
        measure again per stratum, and ``world`` is its starting state on a world with state.
    :ptype cases: Sequence[Mapping[str, Any]]
    :param arms: The arms, distinct — no two at one model and one level of every lever — which the caller holds.
    :ptype arms: Sequence[CallableArm]
    :param scorers: The function graders, each reported as a measure named by its ``__name__``.
    :ptype scorers: Sequence[Scorer]
    :param scope_id: The scope the template, cases and runs are stored in.
    :ptype scope_id: str
    :param host: Where to run and store: it declares a measure for every scorer, the callable kinds' contracts and the
        world (:func:`callable_host`).
    :ptype host: EvalHost
    :param template_name: The template's name, as a listing of templates shows it.
    :ptype template_name: str
    :param expected: Declares the runner a classifier: called once per case, it returns the label a correct answer
        gives. Each cell then lands ``match`` and ``confusion_cell``.
    :ptype expected: ExpectedLabel | None
    :param label_field: The field of the answer a classifier's label is read from; ``None`` reads the answer itself.
    :ptype label_field: str | None
    :param judge: The judge every arm's answers are graded by — one, so no difference between arms is one of judging.
    :ptype judge: Judge | None
    :param intent: What every case asks, which a judge reads; ``None`` takes the first line of the runners' docstring.
    :ptype intent: str | None
    :param world: The world every arm's agent acts on, or ``None``.
    :ptype world: World | None
    :param goal_checks: Goal-state checks over a world with state, each reported as passed or failed per cell.
    :ptype goal_checks: Sequence[str]
    :param k: Repeats per case, per arm.
    :ptype k: int
    :param max_cost_usd: The most each arm's run may spend, in US dollars; ``None`` runs uncapped, which every summary
        states.
    :ptype max_cost_usd: float | None
    :param subject: The subject every arm's run shares (a comparison's: its arms are variants of one subject, so the
        frontier and history rank them together); ``None`` makes each run's subject its arm's name, else its
        model.
    :ptype subject: str | None
    :param margins: Run-scoped margins on core rate measures (``{"accuracy": 0.05}``), declared on every arm's run.
    :ptype margins: Mapping[str, float] | None
    :param measure_latency: Declare latency under test: the cases run one at a time.
    :ptype measure_latency: bool
    :return: Each arm's finished run's summary, read back from the store, in arm order.
    :rtype: list[EvalSummary]
    :raises ValueError: The cases, the arms, the graders, the cap or the world refused, each naming why; a host of the
        caller's own declaring no callable-kind contract, or no measure for a scorer.
    :raises ValidationFailedError: The launch refused.
    """
    plain_cases = _plain_cases(cases)
    names = _case_names(plain_cases)
    plain_cases, strata = _case_strata(plain_cases, names)
    given_seeds = [case.get(WORLD_KEY) for case in plain_cases] if any(WORLD_KEY in c for c in plain_cases) else None
    # What the runner, every grader and the judge are handed: the case less what names, sorts and seeds it.
    addressed = [{key: value for key, value in case.items() if key != WORLD_KEY} for case in plain_cases]
    stimulus = [{key: value for key, value in case.items() if key not in RESERVED_CASE_KEYS} for case in plain_cases]
    refuse_an_unusable_cap(max_cost_usd)
    goal_checks = list(goal_checks)
    if not scorers and expected is None and judge is None and not goal_checks:
        raise ValueError(
            "an eval needs at least one grader — a function, a field name, an expression or a Judge: a run nothing "
            "grades measures nothing"
        )
    _refuse_unnamed_or_repeated(scorers)
    if intent is not None and (not isinstance(intent, str) or not intent.strip()):
        raise ValueError(
            f"an intent is the sentence the judge reads as what each case asks: a non-blank string, not {intent!r}"
        )
    _refuse_a_sync_candidate_handed_tools(arms, why="a world's tools" if world is not None else None)
    labels = None if expected is None else _expected_labels(stimulus, expected, names)
    seeds = _world_seeds(world, given_seeds, goal_checks, host, names)
    _refuse_an_undeclared_callable_contract(host, judged=judge is not None)
    _refuse_undeclared_measures(host, scorers)
    models: list[str] = []
    for arm in arms:
        model = arm.model or arm.default_model
        if model is None:
            model = getattr(arm.candidate, "__name__", None)
            if not model:
                raise ValueError(
                    f"{arm.candidate!r} has no __name__ to label its arm by; name the arm ({{'<name>': runner}})"
                )
        models.append(model)
    graded_by = "every scorer grades" if judge is None else "the judge and every scorer grade"
    template_intent, intent_source = _template_intent(arms, graded_by, intent)
    template_id = _template_id(
        addressed, labels, judge, seeds, goal_checks, intent=template_intent if judge is not None else None
    )
    template = EvalTemplate(
        id=template_id,
        scope_id=scope_id,
        name=template_name,
        intent=template_intent,
        candidate_kind=CALLABLE_KIND if judge is None else JUDGED_CALLABLE_KIND,
        rubric=list(judge.dims) if judge is not None else [],
        goal_state_checks=goal_checks,
    )
    test_cases = [
        EvalTestCase(
            id=f"{template_id}-{index}",
            scope_id=scope_id,
            template_id=template_id,
            stratum=None if strata is None else strata[index],
            # The author's own id names the case on the page; a case called by its position gets no name there.
            name=names[index] if "id" in plain_cases[index] else None,
            variation_params=stored_variation(case),
            host_payload=_case_payload(
                case, None if labels is None else labels[index], None if seeds is None else seeds[index]
            ),
        )
        for index, case in enumerate(stimulus)
    ]
    if world is not None and world.stateful:
        # The authoring gate itself, before anything is stored or run: the grammar, the paths, a string match
        # over model prose, a text comparison over a call parameter its schema does not close, and calls or
        # firings the world cannot produce. A quick run is authored here, so it is held to what authoring holds.
        try:
            refuse_unsupplied_world(template, profile=host.profile)
        except ValidationFailedError as refused:
            raise ValueError(str(refused)) from refused
    if judge is not None:
        # The judge declaration's gate, as authoring holds a host's template to it: a dimension a Judge built is
        # declared built-in, and one handed in declaring "configured" needs its config in the store this call uses.
        try:
            await run_blocking(
                host.blocking_executor,
                refuse_undeclared_judges,
                template,
                active_judge_config_in(host.storage, scope_id),
            )
        except ValidationFailedError as refused:
            raise ValueError(str(refused)) from refused
    await run_blocking(host.blocking_executor, _save_unless_unchanged, host, template, test_cases)
    if judge is not None and judge.examples:
        await run_blocking(host.blocking_executor, _seed_judge_examples, host, template, judge)
    # Each scorer's declared range, which every score it returns is held to: the bounded tests read it as a fact.
    ranges = {
        scorer: descriptor.value_range
        for scorer in map(_scorer_name, scorers)
        if (descriptor := host.profile.measures.get(scorer)) is not None and descriptor.value_range is not None
    }
    recording = None if world is None or world.recording is None else ToolRecording(world.recording)
    if recording is not None:
        recording.begin()
    # A logged runner's outputs, read once as the run starts: every cell is served from the same reading.
    case_of = {test_case.id: name for test_case, name in zip(test_cases, names, strict=True)}
    logged = {
        id(arm.candidate): await run_blocking(host.blocking_executor, arm.candidate.outputs)
        for arm in arms
        if isinstance(arm.candidate, Log)
    }
    wired = [
        _WiredArm(
            model=model,
            levers=dict(arm.levers or {}),
            kind_factory=_logged_kind_factory(
                arm.candidate,
                logged[id(arm.candidate)],
                case_of,
                scorers,
                classifies=labels is not None,
                label_field=label_field,
                judge=judge,
                ranges=ranges,
            )
            if isinstance(arm.candidate, Log)
            else _kind_factory(
                arm.candidate,
                scorers,
                classifies=labels is not None,
                label_field=label_field,
                judge=judge,
                world=world,
                recording=recording,
                ranges=ranges,
            ),
            arm=arm.arm,
            shared_subject=subject,
            named=arm.model is not None,
        )
        for arm, model in zip(arms, models, strict=True)
    ]
    preparing: list[_WiredArm] = []
    launch_host = _launch_host(
        host,
        wired,
        test_cases,
        judge,
        world,
        max_cost_usd=max_cost_usd,
        preparing=preparing,
    )

    async def form() -> tuple[LaunchGroup, None]:
        # Every arm's model is a candidate of the launch, which is what its judge is chosen against.
        return LaunchGroup(candidate_models=list(dict.fromkeys(models))), None

    async def prepare(group: LaunchGroup, _formed: None) -> list[EvalRun]:
        prepared: list[EvalRun] = []
        for arm, wired_arm, model in zip(arms, wired, models, strict=True):
            # A launch into this group prepares its one run inside the call, so the launcher wires this arm.
            preparing[:] = [wired_arm]
            prepared += await start_run(
                launch_host,
                template_id=template_id,
                subject_id=wired_arm.subject,
                models=[] if wired_arm.requested_model is None else [wired_arm.requested_model],
                k_runs=k,
                scope_id=scope_id,
                judge_model=judge.model if judge is not None else None,
                overlays=dict(arm.levers) if arm.levers else None,
                max_cost_usd=max_cost_usd,
                launch_group=group,
                margins=margins,
                measure_latency=measure_latency,
            )
        preparing.clear()
        return prepared

    runs = await launch_as_group(
        launch_host,
        len(arms),
        settings=launch_host.settings(),
        form=form,
        prepare=prepare,
        event="eval.quick",
    )
    try:
        await launch_host.job_manager.wait_for([run.id for run in runs])
    except asyncio.CancelledError:
        # This call owns the runs it started: cancelled, it settles them as cancelled rather than
        # leaving pending ones behind in a store that outlives it.
        await launch_host.job_manager.shutdown()
        raise
    by_id = {test_case.id: name for test_case, name in zip(test_cases, names, strict=True)}

    def summarized() -> list[EvalSummary]:
        return [
            _with_case_results(
                host,
                summarize_run(host, run.id, scope_id, case_names=by_id),
                test_cases,
                names,
                behavior="classify" if expected is not None else "score",
            )
            for run in runs
        ]

    # Summarizing reads each run and its results back from the store, so it runs off the loop.
    summaries = await run_blocking(host.blocking_executor, summarized)
    # What a candidate that did nothing would pass, from each case's own starting state: a quick world's template
    # names no controls, so every check is unproven, and this is the baseline its pass rate is read against.
    idle = (
        world.did_nothing_passes(goal_checks, [(seeds[i], tc.variation_params) for i, tc in enumerate(test_cases)])
        if world is not None and seeds is not None and goal_checks
        else None
    )
    # The store keeps the intent but not where it came from; a judged run's summary carries both.
    return [
        _with_baseline(summary, idle, len(test_cases), intent_source).model_copy(update={"arm": arm.arm})
        for summary, arm in zip(summaries, arms, strict=True)
    ]


def _with_case_results(
    host: EvalHost, summary: EvalSummary, test_cases: list[EvalTestCase], names: list[str], *, behavior: str
) -> EvalSummary:
    """The summary carrying every result read for a person, so they outlive the store the run was in.

    Kept on the summary, a value, rather than by keeping the store alive behind it: the default store is the
    call's own and in memory, and a summary that held it would hold every run, trace and template of it for
    as long as anyone kept the summary, and could not be serialised. What a person reads of a result is small
    — the case, the answer, the grades and the reasons — so it is copied out here, once, as the store has it.
    """
    order = {test_case.id: index for index, test_case in enumerate(test_cases)}
    stored = sorted(
        list_results(host.storage, summary.run_id, summary.scope_id),
        key=lambda result: (order.get(result.test_case_id, len(order)), result.k_iteration),
    )
    payloads = {test_case.id: test_case.host_payload for test_case in test_cases}
    strata = {test_case.id: test_case.stratum for test_case in test_cases}
    # Every measure's direction, guardrail or not: a caller's own lower-is-better measure misses above 0, not at it.
    directions = {
        name: descriptor.higher_is_better
        for name in host.profile.measures.names
        if (descriptor := host.profile.measures.get(name)) is not None
    }
    # Each result's state is read as the page reads its trials: the measures described as the host declares them, so
    # the console's count of cases passed is the page's.
    catalog = {
        name: describe_reported_measure(name, host.profile.measures)
        for name in sorted({name for result in stored for name in result.host_measures})
    }
    threshold = host.profile.bars.pass_threshold(behavior)
    case_results = []
    for result in stored:
        payload = payloads.get(result.test_case_id, {})
        index = order.get(result.test_case_id)
        trace = get_result_trace(host.storage, result) if result.has_trace else None
        case_results.append(
            CaseResult.of(
                result,
                case=names[index] if index is not None else result.test_case_id,
                given=payload.get(_CASE_KEY),
                expected=payload.get(_EXPECTED_KEY),
                answer=_stored_answer(trace.trace if trace is not None else []),
                higher_is_better=directions,
                stratum=strata.get(result.test_case_id),
                catalog=catalog,
                rubric_threshold=threshold,
            )
        )
    return summary.model_copy(update={"case_results": case_results})


def _stored_answer(trace: list[dict[str, Any]]) -> Any:
    """The answer a cell stored (:func:`_as_stored`): its value, or the ``repr`` JSON could not hold; ``None`` for none."""
    if len(trace) != 1:
        return None
    (stored,) = trace
    return stored.get("value", stored.get("repr"))


def _with_baseline(
    summary: EvalSummary, idle: Mapping[str, int] | None, cases: int, intent_source: str | None
) -> EvalSummary:
    """The summary with the intent's source, and each goal check's do-nothing baseline where there is one."""
    update: dict[str, Any] = {}
    if summary.intent is not None:
        update["intent_source"] = intent_source
    if idle is not None:
        update["goal_checks"] = [
            goal.model_copy(update={"did_nothing_passed": idle[goal.check], "did_nothing_cases": cases})
            if goal.check in idle
            else goal
            for goal in summary.goal_checks
        ]
    return summary.model_copy(update=update) if update else summary


__all__ = [
    "ARM_LEVER",
    "CALLABLE_HOST_ID",
    "CALLABLE_KIND",
    "CALLABLE_KIND_CONTRACT",
    "CALLABLE_UNSEATED",
    "JUDGED_CALLABLE_KIND",
    "JUDGED_CALLABLE_KIND_CONTRACT",
    "JUDGED_CALLABLE_UNSEATED",
    "RESERVED_CASE_KEYS",
    "SHARED_ARM_MODEL",
    "STRATUM_KEY",
    "WORLD_KEY",
    "CallableArm",
    "CallableKind",
    "Candidate",
    "ExpectedLabel",
    "Scorer",
    "callable_host",
    "callable_kind",
    "callable_kind_contracts",
    "refuse_a_store_beside_a_host",
    "refuse_an_unusable_cap",
    "refuse_unusable_guardrails",
    "refuse_unusable_margins",
    "run_arms",
    "scorer_measure",
]
