"""``run_eval``: a case list, an async candidate and scorer functions, run through the engine in one call.

Rung 0 of adopting the engine. A product with nothing but a function to test and a way to grade its
answer hands over three values and gets a summary back; everything else a host would build is built
here, from the public roots, on the terms the engine already sets:

- **The kind** is :class:`CallableKind`: ``invoke`` calls the candidate on the case and each scorer
  on what it returned, and reports the scores as the host's measures. It is unjudged — the scorers
  are the grade — and seeds no world. A candidate that returns an
  :class:`~threetears.evals.quick.answer.Answer` reports its own spend beside its answer, as the cell's
  ``candidate`` usage row.
- **A classifier** is a candidate whose answer is a label, declared by handing ``run_eval`` the
  case's expected label (``expected=``). Each cell then lands the core ``match`` and
  ``confusion_cell`` as a classifier kind does, so the summary carries the confusion matrix and each
  label's precision, recall and F1, and the analysis derives ``accuracy``. An answer that is not a
  usable label lands under :data:`UNUSABLE_ANSWER`, a predicted label of its own — and so does a
  candidate that raised, a failure counted as a miss rather than left out of every rate.
- **Tools** are plain functions the candidate calls, declared by handing ``run_eval`` them by name
  (``tools=``); the candidate is then called with the case and its tools. Each cell adapts them to the
  engine's cassette seams (:class:`~threetears.evals.quick.tools.CellTools`), so a run launched with
  ``cassette_mode='capture'`` records what they answered and one with ``'replay'`` serves a capture's
  recording in place of calling them — every arm facing the same tool answers. A candidate that
  declares no tools runs with cassettes off, and a cassette run of one is refused.
- **A judge** is a model grading each answer on a rubric, declared by handing ``run_eval`` a
  :class:`~threetears.evals.quick.judged.Judge`. The run is then of :data:`JUDGED_CALLABLE_KIND`, a
  document kind: the template carries the rubric, each cell renders the answer and its case as the
  judge's evidence, and the engine's own judge service scores every dimension and records the judge's
  spend on the result's ``judge`` usage row, as it does for any judged run. Scorers and an expected
  label grade beside it. The judge also reads the template's intent: ``intent=`` when given, else the
  first line of the candidate's docstring, else a generic sentence; the summary names which.
- **A world** is state the candidate acts on through tools, declared by handing ``run_eval`` a
  :class:`~threetears.evals.quick.world.World`, each case's starting state (``seed=``) and goal-state
  checks over the end state and the calls made (``goal_checks=``). The candidate is then called with the
  case and the tools on its cell's world; each cell seeds the case's state before the candidate's first
  turn, the runner reads it back after the last, and the checks grade it, through the engine's own world
  session and goal-state evaluation (:mod:`threetears.evals.quick.world`).
- **The host**, when none is given, is :func:`callable_host`: the shared sweepable core, one measure
  per scorer, no world, and the in-memory reference store, or the store handed as ``store=`` (a
  :class:`~threetears.evals.storage.SqliteDocumentStore` keeps the runs past the process). Given one, its storage and vocabulary are
  used: every scorer must already be a measure it declares, and it must declare a contract for the
  callable kind that seats no judge, simulator or spend ceiling — and, for a judged call, a contract
  for the judged kind that seats the judge and nothing else of the engine's.
- **The launch** goes through :func:`~threetears.evals.run.start_run`, the path a product serving
  launches takes, so the run is assembled, stamped and identified exactly as any other run is; the
  call then waits for its job and reads the stored run back.

**The case set is content-addressed.** The template's id is a digest of the cases, and each case's
id is its position under it, so two calls over the same cases in one store share a template and
their runs are runs of one scenario — comparable in a campaign — while a different case list is a
different template.
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

from threetears.evals.contracts import (
    ACCURACY_MEASURE,
    CONFUSION_CELL_MEASURE,
    DEFAULT_LAUNCH_K_RUNS,
    MATCH_MEASURE,
    METRIC_DESCRIPTORS,
    SCALES,
    CandidateOutput,
    CandidateTelemetry,
    CassetteMode,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    DocumentStore,
    EvalRun,
    EvalStorage,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    MetricDescriptor,
    VariantConfig,
    WorldSeed,
    WorldSession,
    canonical_digest,
    confusion_cell,
    withhold_failure_detail,
)
from threetears.evals.contracts.models import stored_variation
from threetears.evals.contracts.host import (
    SHARED_CORE,
    ApparatusError,
    CompletionClients,
    EvalHost,
    HostProfile,
    KindContract,
    MeasureRegistry,
    NominalScale,
    SubjectSnapshot,
    Sweepable,
    SweepableRegistry,
    SweepableValue,
    TraceSink,
    WorldPlacement,
    WorldRegistry,
    default_cell_timeout,
)
from threetears.evals.contracts.host.sweepables import CORE_ROLES, CORE_SWEEPABLES
from threetears.evals.ops.summary import CaseResult, EvalSummary, summarize_run
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
    build_judge_service,
    default_job_timeout,
    get_result_trace,
    launch_as_group,
    launch_run,
    list_results,
    start_run,
)
from threetears.evals.run.authoring import refuse_unsupplied_world
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.quick.answer import unwrap_answer
from threetears.evals.quick.guardrails import Guardrail
from threetears.evals.quick.judged import Judge, judge_evidence
from threetears.evals.quick.levers import CallableLevers, levers_model
from threetears.evals.quick.tools import CellTools, Tool, ToolUsingCandidate, refuse_unusable_tools
from threetears.evals.quick.world import CaseSeed, World, WorldCandidate, WorldCellKind, world_case_payload
from threetears.evals.storage import InMemoryDocumentStore

#: The candidate under test: a callable taking one case and returning its answer — usually ``async def``; a
#: plain ``def`` is called in a worker thread (:func:`run_eval`), and a callable returning an awaitable has it awaited.
Candidate = Callable[[Mapping[str, Any]], Awaitable[Any] | Any]

#: One grade: takes the case and the candidate's answer, returns a number (``True``/``False`` count
#: as 1 and 0). Its ``__name__`` is the measure's name, and higher is better. No scorer may take the name
#: of an engine core measure (``score``, ``f1``, ``cost_usd`` and the rest of ``METRIC_DESCRIPTORS``).
Scorer = Callable[[Mapping[str, Any], Any], float | bool]

#: A classifier's expected label for one case: takes the case, returns the label a correct answer gives.
ExpectedLabel = Callable[[Mapping[str, Any]], str]

#: The predicted label a classifier's answer is counted under when it is not a usable label: not a
#: string, or a blank one. It is its own cell in the confusion matrix and never a match, so an answer the
#: classifier could not give is never counted as one it did. No case may expect it.
UNUSABLE_ANSWER = "(unusable answer)"

#: The names a classifier lands (``match``, ``confusion_cell``) and the one the engine derives from them
#: (``accuracy``). No scorer may take one: it would collide with a classifier's own measure, or be read
#: under the core descriptor of a measure it is not.
_CLASSIFIER_NAMES = frozenset({MATCH_MEASURE, CONFUSION_CELL_MEASURE, ACCURACY_MEASURE})

#: The kind :func:`run_eval` launches, as its template names it.
CALLABLE_KIND = "callable"

#: The kind :func:`run_eval` launches when it is handed a judge: the callable, its answers judged as documents.
JUDGED_CALLABLE_KIND = "callable-judged"

#: The host :func:`callable_host` builds, by the id the engine prints in its logs and errors.
CALLABLE_HOST_ID = "run_eval"

#: The lever a single-factor :func:`~threetears.evals.quick.compare` names its arms on, declared by
#: ``callable_host(arms=True)``: each arm's run states its name as its level, so the report calls the arm
#: ``candidate=<name>`` rather than calling the name a model.
ARM_LEVER = "candidate"

#: The candidate model every arm of a single-factor :func:`~threetears.evals.quick.compare` shares when its arms
#: are named on :data:`ARM_LEVER`: the arms' names are not models, and a lever every arm shares splits none of them.
SHARED_ARM_MODEL = "callable"

#: Where a named arm's run carries its name, on its ``host_payload``, for the host's lever reader.
_ARM_PAYLOAD_KEY = "arm"

#: Where a case rides on its stored test case: verbatim, for the candidate to be handed back.
_CASE_KEY = "case"

#: Where a classifier's case carries its expected label on its stored test case.
_EXPECTED_KEY = "expected"

#: The callable kind's contract: no overlays, no spec, and no rig seat — nothing in a ``run_eval`` run is
#: graded by a model or talks to a simulated user. Without the empty seats each blank judge and simulator
#: dimension would read as an unrecoverable judge or simulator and confound every comparison of two such
#: runs. A host of the caller's own declares a contract for the callable kind on its profile too —
#: this one, or one seating apparatus of its own the runs do read — and :func:`run_eval` refuses a host
#: that does not (:data:`CALLABLE_UNSEATED`).
CALLABLE_KIND_CONTRACT = KindContract(CALLABLE_KIND, seats=frozenset())

#: What a ``run_eval`` run never has as a level of its rig, whatever host it runs in, so a callable-kind contract
#: may seat none of it: the engine's judge and simulator (by role or by any pinned dimension) — the callable kind
#: is unjudged and simulates nobody — and the spend ceiling, which is off unless a call caps it
#: (``max_cost_usd=``) and is then a condition stated beside the run: a run it stopped says so by its status, and
#: two runs under different caps measured the same candidate the same way until one stopped. A seat here would
#: make every blank such dimension an unrecoverable level, and every comparison of two ``run_eval`` runs
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

#: What a judged ``run_eval`` run never has as a level of its rig: the simulator (by role or by any pinned
#: dimension) and the spend ceiling, a condition beside the run as :data:`CALLABLE_UNSEATED` says. The judge is
#: not here — it is the one seat the judged kind fills.
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

    Every arm of the launch admitted and started together (:func:`run_eval` launches one,
    :func:`~threetears.evals.quick.compare` one per arm). With no cap, ceiling enforcement is off and every run
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


def _refuse_an_unusable_cap(max_cost_usd: float | None) -> None:
    if max_cost_usd is not None and (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, int | float)
        or not math.isfinite(max_cost_usd)
        or max_cost_usd <= 0
    ):
        raise ValueError(f"max_cost_usd= is a spend ceiling in US dollars: a positive number, not {max_cost_usd!r}")


def _scorer_reader_name(name: str) -> str:
    """What a reader calls a scorer's measure: its def's name in words, as a score — ``exact_match`` is "Exact match score".

    Args:
        name: The scorer's ``__name__``.

    Returns:
        The reader-facing name its descriptor carries.
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

    A scorer whose return is annotated ``bool`` is a pass/fail and is declared on the range 0 to 1, margin or
    not: its intervals stay inside what a pass rate can be, and with a margin its equivalence test is the
    bounded one that holds its error rate at any n. Any other scorer is on the range its caller declares
    (``compare(ranges=...)``), or on none, since nothing else says what its values can be: its intervals are
    then the unclipped t intervals, and it takes no margin (:func:`refuse_unusable_margins`).

    Declared a guardrail (``guardrail=``), it is instead a measure no arm may get worse on: on no merit axis, so it
    joins no contrast and no composite, with the guardrail's direction and its margin as the measure's
    ``materiality_threshold``.

    Args:
        scorer: The scorer. Its ``__name__`` names the measure and the first line of its docstring,
            when it has one, describes it.
        margin: The measure's declared margin (``materiality_threshold``): the most a difference in it may be and
            still be too small to act on, so a contrast on it can read ``equivalent``. ``None`` declares none,
            and no contrast on it can.
        value_range: The lowest and highest value a non-``bool`` scorer can return, or ``None``.
        guardrail: Declares the measure a guardrail, with its margin and direction; ``None`` for a quality score.
            A guardrail takes no ``margin`` beside it: its margin is the guardrail's own.

    Returns:
        The descriptor :func:`callable_host` registers for it.

    Raises:
        ValueError: Both ``margin`` and ``guardrail`` are given.
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

    Args:
        scorers: The scorers whose measures the margins are declared on.
        margins: Each margin, by the name of the scorer it is declared on.
        ranges: Each non-``bool`` scorer's range, by its name.

    Raises:
        ValueError: A margin or range on a name no scorer has (the classifier's accuracy and every engine core
            measure among them), a margin that is not a positive finite number, a range on a ``bool`` scorer or
            one that is not two finite numbers, low below high, or a margin on a scorer with no range.
    """
    ranges = dict(ranges or {})
    names = [_scorer_name(scorer) for scorer in scorers]
    by_name = dict(zip(names, scorers, strict=True))
    for name, bounds in ranges.items():
        if name not in names:
            raise ValueError(
                f"ranges= names {name!r}, which no scorer reports; a range is declared on a scorer's measure, by "
                f"its def's name: {', '.join(map(repr, names)) or 'none were given'}"
            )
        if _returns_bool(by_name[name]):
            raise ValueError(
                f"{name!r} returns a bool, a pass/fail, which is on 0 to 1 already: declare ranges= only for a "
                "scorer returning a number"
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
                f"first: ranges={{{name!r}: (1, 5)}} for a 1-to-5 score, not {bounds!r}"
            )
    for name, margin in margins.items():
        if name in _CLASSIFIER_NAMES or name in METRIC_DESCRIPTORS:
            raise ValueError(
                f"a margin on {name!r} is not a scorer's: it is an engine core measure, whose descriptor the engine "
                "owns. A margin on a core rate measure (accuracy) is declared on the runs instead: "
                "compare(margins={'accuracy': 0.05}) declares it on every arm's run"
            )
        if name not in names:
            raise ValueError(
                f"margins= names {name!r}, which no scorer reports; a margin is declared on a scorer's measure, by "
                f"its def's name: {', '.join(map(repr, names)) or 'none were given'}"
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
                f"ranges={{{name!r}: (1, 5)}} for a 1-to-5 score. A pass/fail scorer annotated -> bool needs none"
            )


def refuse_a_store_beside_a_host(store: DocumentStore | None, host: EvalHost | None) -> None:
    """Refuse ``store=`` beside ``host=``: a host brings its own storage, and one call stores in one place.

    Raises:
        ValueError: Both were given.
    """
    if store is not None and host is not None:
        raise ValueError(
            f"store= is where the host this call builds keeps the runs, and host {host.profile.host_id!r} brings its "
            "own storage; pass one or the other (callable_host(..., store=...) builds a host over a store)"
        )


#: Why a scorer named in both ``margins=`` and ``guardrails=`` is refused, naming it.
_MARGIN_ON_A_GUARDRAIL = (
    "{name!r} is declared a guardrail and given a margin in margins= too: a guardrail is never contrasted, so a "
    "margin for equivalence has no use on it, and its own margin is Guardrail(margin=...); drop it from margins="
)


def refuse_unusable_guardrails(
    scorers: Sequence[Scorer],
    guardrails: Mapping[str, Guardrail],
    *,
    margins: Mapping[str, float] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    judge: Judge | None = None,
) -> None:
    """Refuse a guardrail no scorer's measure, or judged dimension, could carry, naming the ones that can.

    Args:
        scorers: The scorers whose measures a guardrail may be declared on.
        guardrails: Each guardrail, by the name of the scorer (or the judge's rubric dimension) it is declared on.
        margins: The margins declared beside them, by scorer.
        ranges: The ranges declared for scorers returning a number, by scorer (``compare(ranges=...)``).
        judge: The judge whose rubric dimensions a guardrail may also be declared on, or ``None``.

    Raises:
        ValueError: A guardrail that is not a :class:`~threetears.evals.quick.Guardrail` (a bare margin among
            them); on an engine core measure (the classifier's accuracy among them), or a name no scorer or
            dimension has; on a scorer also given a margin; a margin as wide as the scorer's range (0 to 1 for a
            pass/fail) or the dimension's scale; or a lower-is-better direction on a judged dimension.
    """
    names = [_scorer_name(scorer) for scorer in scorers]
    by_name = dict(zip(names, scorers, strict=True))
    dims = {dim.name: dim for dim in judge.dims} if judge is not None else {}
    for name, guardrail in guardrails.items():
        if not isinstance(guardrail, Guardrail):
            raise ValueError(
                f"guardrails= maps a name to a Guardrail, its margin and direction both declared "
                f"(guardrails={{{name!r}: Guardrail(margin=0.02, direction='higher_is_better')}}), not {guardrail!r}"
            )
        if name in _CLASSIFIER_NAMES or name in METRIC_DESCRIPTORS:
            raise ValueError(
                f"a guardrail on {name!r} cannot be declared: it is an engine core measure, read under the core's own "
                "description. Grade what the candidate must not get worse on with a scorer "
                "(def no_leak(case, answer) -> bool: ...) and declare the guardrail on that"
            )
        if name in by_name:
            if margins and name in margins:
                raise ValueError(_MARGIN_ON_A_GUARDRAIL.format(name=name))
            bounds = (0.0, 1.0) if _returns_bool(by_name[name]) else (ranges or {}).get(name)
            if bounds is not None and guardrail.margin >= bounds[1] - bounds[0]:
                raise ValueError(
                    f"a margin of {guardrail.margin!r} on {name!r}, which scores {bounds[0]:g} to {bounds[1]:g}, is "
                    "as wide as every value it can return, so every arm would hold whatever it did; declare a "
                    "smaller one"
                )
            continue
        dim = dims.get(judge.dim_name(name)) if judge is not None else None
        if dim is None:
            said = ", ".join(map(repr, [*names, *dims])) or "none were given"
            raise ValueError(
                f"guardrails= names {name!r}, which no scorer reports and no rubric dimension of the judge scores; a "
                f"guardrail is declared on a scorer's measure, by its def's name, or on a judged dimension: {said}"
            )
        if not guardrail.higher_is_better:
            raise ValueError(
                f"{dim.name!r} is judged with higher better, so its guardrail's direction is 'higher_is_better': "
                "word the dimension as what the answer must do (never_leaks), and a higher score is the safer one"
            )
        low, high = SCALES[dim.scale].value_range
        if guardrail.margin >= high - low:
            raise ValueError(
                f"a margin of {guardrail.margin!r} on {dim.name!r}, judged {low:g} to {high:g}, is as wide as its "
                "scale, so every arm would hold whatever it did; declare a smaller one"
            )


def callable_kind_contracts(levers: Sequence[str] = ()) -> tuple[KindContract, KindContract]:
    """The contracts of both callable kinds, declaring ``levers`` as each run's levels beside its model.

    With no levers, :data:`CALLABLE_KIND_CONTRACT` and :data:`JUDGED_CALLABLE_KIND_CONTRACT` themselves, so a
    host declaring none keys its runs exactly as before. With levers, the same seats and an overlay model
    (:func:`~threetears.evals.quick.levers.levers_model`) whose every field is a lever the engine names
    ``callable.<name>`` (``callable-judged.<name>`` on a judged run), resolves into each run's variant key and
    accepts as a campaign's axis. A host of the caller's own declares these on its profile's ``kinds`` to
    take :func:`run_eval`'s ``levers=``.

    Args:
        levers: The levers' names.

    Returns:
        The unjudged kind's contract, then the judged kind's.

    Raises:
        ValueError: A lever name no overlay field could carry, ``model``, or one given twice.
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

    What :func:`run_eval` builds when it is handed no host. With no ``store``, its store lives as long as the
    returned value, so a caller wanting to run several candidates into one store and compare them builds this
    once and hands it to each call. With one, the runs are wherever that store keeps them, and a later process
    reads them through ``callable_host(<the same scorers>, store=<the same store>)``. A classifier's ``match`` and ``confusion_cell`` are core measures,
    so a host for a classifier with no scorers of its own declares none: ``callable_host()``. It declares
    the contracts of both callable kinds, so one host serves judged and unjudged calls alike.

    Args:
        scorers: The scorer functions whose measures the host declares.
        levers: The levers every run in this host states a level of beside its model
            (:func:`callable_kind_contracts`), each handed to :func:`run_eval` as ``levers=``.
        world: The world a world run seeds (:class:`~threetears.evals.quick.world.World`), declared on the
            profile; ``None`` declares none.
        arms: Declare :data:`ARM_LEVER`, the lever a single-factor :func:`~threetears.evals.quick.compare`
            names its arms on, so its report reads ``candidate=<name>``; what ``compare`` builds when handed no
            host. A run that states no arm (a ``run_eval`` in this host) is at its candidate model's label.
        margins: A margin declared on a scorer's measure, by the scorer's name (:func:`scorer_measure`), so a
            contrast on it can read ``equivalent``; ``None`` declares none. No margin is ever assumed.
        ranges: The lowest and highest score a scorer returning a number can give, by the scorer's name
            (``{"rating": (1, 5)}``): its intervals stay inside it, and a margin on it can be tested. A ``bool``
            scorer is on 0 to 1 already. A score outside its range excludes its cell, naming the scorer.
        store: Where every document is stored (any :class:`~threetears.evals.contracts.DocumentStore`):
            :class:`~threetears.evals.storage.SqliteDocumentStore` keeps the runs in a file. ``None`` builds a fresh
            :class:`~threetears.evals.storage.InMemoryDocumentStore`, which keeps nothing past the process.
        clients: The completion clients the engine's own model calls use after the run (a re-judge, an
            analysis), as :class:`~threetears.evals.contracts.host.EvalHost` takes them; ``None`` supplies none,
            and a call needing them is refused. A :class:`~threetears.evals.quick.Judge` brings its own for the run.
        trace_sink: Where each cell's trace is sent, as :class:`~threetears.evals.contracts.host.EvalHost` takes
            it; ``None`` traces nothing.
        guardrails: The scorers declared guardrails, by name, each with its margin and direction
            (:class:`~threetears.evals.quick.Guardrail`): measures no arm may get worse on, decided apart from
            every contrast. A guardrail on a judged dimension is the campaign's to declare, so it is
            :func:`~threetears.evals.quick.compare`'s, not a host's. ``None`` declares none.

    Returns:
        The host.

    Raises:
        ValueError: A scorer has no usable name, two share one, or one takes a classifier measure's name or
            any other engine core measure's; a lever name is unusable or repeated; a margin names no scorer,
            is not a positive number or is on a scorer with no range; a range is unusable
            (:func:`refuse_unusable_margins`); or a guardrail names no scorer, is not a ``Guardrail``, is declared
            on a scorer given a margin too, or is as wide as its scorer's range.
    """
    _refuse_unnamed_or_repeated(scorers)
    margins = dict(margins or {})
    ranges = dict(ranges or {})
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
                scorer_measure(
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
            # way the candidate's calls are held to: an enum-closed one may be compared, a free string may not.
            action_parameters=None if world is None else world.action_parameters,
            tool_actions=None if world is None else world.tool_actions,
            variant_levers=_arm_levels if arms else None,
        ),
        storage=EvalStorage(InMemoryDocumentStore() if store is None else store),
        failure_describer=withhold_failure_detail,
        trace_sink=trace_sink,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
        clients=clients,
    )


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

    def __init__(
        self,
        candidate: Candidate | ToolUsingCandidate,
        scorers: Sequence[Scorer],
        *,
        classifies: bool = False,
        judge: Judge | None = None,
        tools: Mapping[str, Tool] | None = None,
        ranges: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        """Bind the candidate and its scorers.

        Args:
            candidate: The callable under test.
            scorers: The grades, each reported under its own name.
            classifies: Whether the candidate is a classifier, whose every case carries its expected label.
            judge: The judge whose evidence each answer carries, or ``None`` for an unjudged kind.
            tools: The tools the candidate is called with beside each case, or ``None`` for a candidate
                called with the case alone.
            ranges: Each scorer's declared range, by its name, which every score it returns must lie in.
        """
        self._candidate = candidate
        self._ranges = dict(ranges or {})
        self._tools = dict(tools) if tools is not None else None
        self._scorers = tuple(scorers)
        self._classifies = classifies
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

        Args:
            subject_snapshot: The run's subject, unused.
            variant_config: The cell's variant; its candidate model labels the arm.
            world_seed: The template's seed, which this kind has no world to write.
            span_window: The cell's trace windows, unused.
            cassettes: The cell's cassettes in a capture or replay run, which only a kind with tools is
                launched with, and ``None`` with cassettes off.
            world: Unread — a callable has no world to seed, so the cell opens none.

        Returns:
            The prepared candidate.
        """
        tools = CellTools(self._tools) if self._tools is not None else None
        if cassettes is not None and tools is not None:
            cassettes.wire(tools)
        return _Prepared(model=variant_config.candidate_model, tools=tools)

    async def invoke(self, instance: _Prepared, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Run the candidate on one case, grade its answer, and report the grades.

        Args:
            instance: What ``prepare`` returned.
            test_case: The cell's case; the caller's case rides on its ``host_payload``.
            sink: The cell's sink, unused: the scores are reported on the output.

        Returns:
            The answer as the stored trace, and each scorer's grade as a host measure, beside a
            classifier's ``match`` and ``confusion_cell``, and a judged kind's evidence.
        """
        case = test_case.host_payload[_CASE_KEY]
        try:
            answer, telemetry = unwrap_answer(await _call_candidate(self._candidate, case, instance.tools))
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
        # One call, answered: the kind's whole turn.
        telemetry = telemetry.model_copy(update={"turns_delivered": 1})
        evidence: JudgeEvidence | None = None
        if self._judge is not None:
            try:
                expected = test_case.host_payload[_EXPECTED_KEY] if self._classifies else None
                evidence = judge_evidence(self._judge, case, answer, expected=expected)
            except ValueError as unrenderable:
                # The judge's material is the rig: the cell is excluded, and with no evidence to read it
                # stores no answer either, since a judged kind's every stored answer carries its evidence.
                return CandidateOutput(infra_errors=[str(unrenderable)], telemetry=telemetry)
        trace = [_as_stored(answer)]
        measures: dict[str, bool | float | str] = {}
        if self._classifies:
            expected = test_case.host_payload[_EXPECTED_KEY]
            predicted = answer if isinstance(answer, str) and answer.strip() else UNUSABLE_ANSWER
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

    Args:
        expected: The case's expected label.

    Returns:
        The measures to land beside the failure.
    """
    return {MATCH_MEASURE: False, CONFUSION_CELL_MEASURE: confusion_cell(expected, UNUSABLE_ANSWER)}


async def _call_candidate(candidate: Candidate | ToolUsingCandidate, case: Any, tools: CellTools | None) -> Any:
    """Call the candidate on one case — with its tools, when it declares any — and surface a rig fault it met.

    A candidate handed no tools may be synchronous: it is called in a worker thread, so a blocking call does
    not stall the event loop the run's other arms and its timeouts run on, and whatever it returns is awaited
    when it is awaitable. Cells within one run are serial, so it is never called beside itself in one arm.

    Raises:
        ApparatusError: A tool call met a rig fault, re-raised here even when the candidate caught it.
    """
    if tools is None:
        if _is_async_callable(candidate):
            return await cast(Candidate, candidate)(case)
        returned = await asyncio.to_thread(cast(Callable[[Any], Any], candidate), case)
        return await returned if inspect.isawaitable(returned) else returned
    try:
        answer = await cast(ToolUsingCandidate, candidate)(case, tools.for_candidate())
    except Exception:
        tools.raise_any_fault()
        raise
    tools.raise_any_fault()
    return answer


def _is_async_callable(candidate: object) -> bool:
    """Whether calling ``candidate`` returns a coroutine: an ``async def``, a partial of one, or an object with one as its ``__call__``.

    Args:
        candidate: The callable.

    Returns:
        ``True`` for an async callable; ``False`` for a plain function, which a call runs to its end.
    """
    if inspect.iscoroutinefunction(candidate):
        return True
    call = getattr(type(candidate), "__call__", None)
    return not inspect.isroutine(candidate) and inspect.iscoroutinefunction(call)


def _refuse_a_sync_candidate_handed_tools(arms: Sequence[CallableArm], *, why: str | None) -> None:
    """Refuse, before anything runs, a synchronous candidate that would be handed async tools it cannot await.

    Args:
        arms: The arms.
        why: What the candidates are handed (``"tools"``, ``"a world's tools"``), or ``None`` when nothing is,
            and a synchronous candidate is called in a thread.
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

    The two keys differ so a reader of the trace can tell a value from a rendering of one.
    """
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
            "pass run_eval its expected label (expected=), and name any other scorer something else"
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
            raise ValueError(f"expected= raised on case {index}: {type(raised).__name__}: {raised}") from raised
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"expected= gave case {index} {label!r}; an expected label is a non-blank string")
        if label == UNUSABLE_ANSWER:
            raise ValueError(
                f"expected= gave case {index} {UNUSABLE_ANSWER!r}, the label an unusable answer is counted under; "
                "a case expecting it would count an answer the classifier could not give as a match"
            )
        labels.append(label)
    return labels


def _refuse_an_undeclared_callable_contract(host: EvalHost, *, judged: bool) -> None:
    """Refuse a caller's host that has not declared what a ``run_eval`` run's rig holds.

    A host with no contract for the kind holds its runs to every apparatus dimension
    (:meth:`~threetears.evals.contracts.host.profile.HostProfile.kind_contract`), so the blank simulator of
    every such run — and the blank judge of every unjudged one — reads as unrecoverable and confounds every
    comparison of two of them, silently. A judged kind's contract that leaves the judge unseated is the
    other half of that: a change of judge between two runs would read as no change at all. Both are
    refused here, before anything is stored, rather than discovered in an analysis.
    """
    profile = host.profile
    kind, model = (
        (JUDGED_CALLABLE_KIND, "JUDGED_CALLABLE_KIND_CONTRACT") if judged else (CALLABLE_KIND, "CALLABLE_KIND_CONTRACT")
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
            f"host {profile.host_id!r} has {what} for the {kind!r} kind, so every run_eval run would be held to "
            f"the apparatus it never has and every comparison of two would read undecided; {remedy}"
        )
    if seated := sorted(contract.seats & unseated):
        never = (
            "simulates nobody, and its spend ceiling is a condition beside the run"
            if judged
            else "is unjudged, simulates nobody, and its spend ceiling is a condition beside the run"
        )
        raise ValueError(
            f"host {profile.host_id!r} seats {', '.join(seated)} for the {kind!r} kind, which a run_eval run never "
            f"has — it {never}; {remedy}"
        )
    if judged and _JUDGE_SEATS.name not in contract.seats and not set(_JUDGE_SEATS.pins) <= contract.seats:
        raise ValueError(
            f"host {profile.host_id!r} does not seat the judge for the {kind!r} kind, so two runs judged by "
            f"different models would compare as judged alike; {remedy}"
        )
    # The one overlay model a callable kind takes is a levers model: its levels are what run_eval's levers= turns.
    if (
        contract.overlays is not None and not issubclass(contract.overlays, CallableLevers)
    ) or contract.spec is not None:
        raise ValueError(
            f"host {profile.host_id!r} declares overlays or a spec for the {kind!r} kind, which run_eval "
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

    Raises:
        ValueError: An ``id`` that is not a non-blank string or an int, or two cases called alike.
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
        raise ValueError("run_eval needs a non-empty list of cases, each a mapping")
    plain: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not all(isinstance(key, str) for key in case):
            raise ValueError(f"case {index} is not a mapping with string keys: {case!r}")
        plain.append(dict(case))
    return plain


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
            addressed["rubric"] = [dim.model_dump(mode="json") for dim in judge.dims]
            addressed["intent"] = intent
        if seeds is not None:
            addressed["seeds"] = seeds
            addressed["goal_checks"] = list(goal_checks)
    try:
        digest = canonical_digest(addressed)
    except TypeError as unencodable:
        raise ValueError(f"every case must be JSON: {unencodable}") from unencodable
    return f"{CALLABLE_HOST_ID}-{digest[:16]}"


def _case_payload(case: dict[str, Any], label: str | None, seed: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``host_payload`` a case's stored test case carries: the case, verbatim, a classifier's expected label, and a world case's starting state."""
    labelled = {} if label is None else {_EXPECTED_KEY: label}
    seeded = {} if seed is None else world_case_payload(seed)
    return {_CASE_KEY: case, **labelled, **seeded}


@dataclass(frozen=True)
class CallableArm:
    """One arm a one-call launch runs: the candidate, the model its run is labelled by, and its other levers.

    Attributes:
        candidate: The candidate under test.
        model: The arm's label, stored as its run's candidate model; ``None`` takes the candidate's ``__name__``.
        levers: The arm's level of every other lever, by name, or ``None`` for none.
        arm: The arm's name, stated on its run as its level of :data:`ARM_LEVER` (and as its subject), for a
            host that declares that lever (``callable_host(arms=True)``); ``None`` names no arm.
    """

    candidate: Candidate | ToolUsingCandidate | WorldCandidate
    model: str | None = None
    levers: Mapping[str, str] | None = None
    arm: str | None = None


@dataclass(frozen=True)
class _WiredArm:
    """One arm as its launcher wires it: the levels its run is launched at, and the factory building its cells' kind."""

    model: str
    levers: dict[str, Any]
    kind_factory: KindFactory
    arm: str | None = None

    @property
    def subject(self) -> str:
        """The run's subject: the arm's name when it has one, else its model."""
        return self.arm if self.arm is not None else self.model


def _launch_host(
    host: EvalHost,
    arms: Sequence[_WiredArm],
    cases: list[EvalTestCase],
    judge: Judge | None,
    world: World | None = None,
    *,
    calls_tools: bool = False,
    max_cost_usd: float | None = None,
) -> LaunchHost:
    """``host`` as a launching host whose one kind runs each arm's kind over ``cases``, judged by ``judge`` when given.

    Every arm is launched into one group (:func:`_launch_arms`), each through its own
    :func:`~threetears.evals.run.start_run` call, so the launcher is asked once per arm and wires the arm the
    request names: the one at the request's candidate model and overlays. Its subject is its model, as a
    one-arm launch's is.

    A judged kind's launcher builds its judge as every judged launcher does, with
    :func:`~threetears.evals.run.build_judge_service` over the run's template, on a host whose client
    factory lends the judge's client — so the run's judge pin, its per-dimension attribution and the
    service that scores it come from one resolution.
    """
    declared: WorldRegistry | None = host.profile.world
    kind_name = CALLABLE_KIND if judge is None else JUDGED_CALLABLE_KIND

    def place(_run: EvalRun) -> dict[str, WorldPlacement]:
        assert declared is not None
        if world is None:
            # A world-less call attaches no carrier, so every dimension a host's world declares is out of play.
            return declared.place(seeded=(), carriers=())
        # Every case of a world run sets every dimension, through the one carrier the world is.
        return declared.place(seeded=world.dimension_names, carriers=(world.name,))

    def arm_of(request: LaunchRequest) -> _WiredArm:
        levels = {} if request.overlays is None else request.overlays.model_dump(mode="json")
        for arm in arms:
            if arm.model == request.candidate_model and arm.levers == levels and arm.subject == request.subject_id:
                return arm
        raise ValueError(f"no arm of this launch runs model {request.candidate_model!r} at levers {levels!r}")

    async def launch(request: LaunchRequest) -> EvalRun:
        arm = arm_of(request)
        run_judge: RunJudge | None = None
        if judge is not None:
            run_judge = build_judge_service(
                replace(host, clients=judge.clients()),
                request.template,
                judge.model,
                judged_artifact=JudgedArtifact.DOCUMENT,
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
            ),
        )

    # The judge pin is the one launch argument a judged kind honours: run_eval names its judge's model. A
    # cassette mode is honoured only by a kind whose candidate declares tools: there is nothing else to record.
    unhonoured: set[LaunchArgument] = {"simulator_model", "judge_config_ids", "n_variations"}
    if judge is None:
        unhonoured.add("judge_model")
    if not calls_tools:
        unhonoured.add("cassette_mode")
    launch_host = LaunchHost(
        eval_host=host,
        kinds={kind_name: LaunchableKind(launch=launch, unhonoured_launch_arguments=frozenset(unhonoured))},
        settings=partial(_launch_settings, len(arms), max_cost_usd),
        job_timeout_factory=default_job_timeout,
        world_placements=place if declared is not None else None,
    )
    return launch_host


def _world_seeds(
    cases: list[dict[str, Any]],
    world: World | None,
    seed: CaseSeed | None,
    goal_checks: Sequence[str],
    host: EvalHost | None,
    names: list[str],
) -> list[dict[str, Any]] | None:
    """Each case's starting state for a world run, or ``None`` for a world-less one, refusing an incoherent request."""
    if world is None:
        if seed is not None or goal_checks:
            raise ValueError("seed= and goal_checks= describe a world's state; pass the world they read (world=)")
        return None
    if seed is None:
        raise ValueError("a world run seeds each case's starting state; pass seed=, from a case to its state")
    if host is not None and host.profile.world is not world.registry:
        raise ValueError(
            f"host {host.profile.host_id!r} does not declare world {world.name!r}; build it with "
            "callable_host(..., world=<the world>) so the run's world and its host's are one"
        )
    world.refuse_unreadable(goal_checks)
    seeds: list[dict[str, Any]] = []
    for index, case in zip(names, cases, strict=True):
        try:
            values = seed(case)
        # prawduct:ok-broad-except — seed= is the caller's code: what it raises on a case is refused with that case named
        except Exception as raised:
            raise ValueError(f"seed= raised on case {index}: {type(raised).__name__}: {raised}") from raised
        seeds.append(world.refuse_unseedable(values, case=index))
    return seeds


def _kind_factory(
    candidate: Candidate | ToolUsingCandidate | WorldCandidate,
    scorers: Sequence[Scorer],
    *,
    classifies: bool,
    judge: Judge | None,
    world: World | None,
    tools: Mapping[str, Tool] | None,
    ranges: Mapping[str, tuple[float, float]],
) -> KindFactory:
    """The factory building each cell's kind: one :class:`CallableKind` for every cell, or a world cell's own kind."""
    if world is None:
        kind = CallableKind(
            cast(Candidate | ToolUsingCandidate, candidate),
            scorers,
            classifies=classifies,
            judge=judge,
            tools=tools,
            ranges=ranges,
        )
        return lambda _cell: kind

    def over(acting: Candidate) -> CallableKind:
        return CallableKind(acting, scorers, classifies=classifies, judge=judge, ranges=ranges)

    def cell_kind(context: CellContext) -> WorldCellKind:
        return WorldCellKind(world, cast(WorldCandidate, candidate), over, context)

    return cell_kind


#: The scope a quick call with no host of its own stores its runs in when it is named none: the call's store is
#: in memory and lives only as long as the call, so no other run can share it by accident.
DEFAULT_QUICK_SCOPE = "quick"


def quick_scope(scope_id: str | None, host: EvalHost | None, store: DocumentStore | None = None) -> str:
    """The scope a quick call stores in: the one named, else :data:`DEFAULT_QUICK_SCOPE` when its store is the call's own.

    Args:
        scope_id: The scope the caller named, or None.
        host: The caller's own host, or None for the one the call builds.
        store: The caller's own store for the host the call builds, or None for an in-memory one.

    Returns:
        The scope.

    Raises:
        ValueError: A host or a store of the caller's own and no scope: either outlives the call, so where its runs
            land is the caller's to say.
    """
    if scope_id is not None:
        return scope_id
    if host is not None or store is not None:
        raise ValueError(
            f"scope_id= is required with a {'host' if host is not None else 'store'} of your own: its store outlives "
            "this call, and the scope is where its runs and campaigns are found again"
        )
    return DEFAULT_QUICK_SCOPE


async def run_eval(
    cases: Sequence[Mapping[str, Any]],
    candidate: Candidate | ToolUsingCandidate | WorldCandidate,
    scorers: Sequence[Scorer] = (),
    *,
    scope_id: str | None = None,
    expected: ExpectedLabel | None = None,
    judge: Judge | None = None,
    intent: str | None = None,
    world: World | None = None,
    seed: CaseSeed | None = None,
    goal_checks: Sequence[str] = (),
    host: EvalHost | None = None,
    store: DocumentStore | None = None,
    k: int = DEFAULT_LAUNCH_K_RUNS,
    model: str | None = None,
    levers: Mapping[str, str] | None = None,
    tools: Mapping[str, Tool] | None = None,
    cassette_mode: CassetteMode = "off",
    cassette_corpus_id: str | None = None,
    max_cost_usd: float | None = None,
) -> EvalSummary:
    """Run ``candidate`` on every case ``k`` times, grade each answer with every scorer and the judge, and summarise.

    Args:
        cases: The cases, each a JSON object; the candidate and the scorers receive each one as given. A case's
            ``id``, when it has one (a non-blank string or an int, each case's its own), is what its results and
            error lines call it; a case with none is called by its position in the list, from ``0``.
        candidate: The callable under test, called once per case and repeat: with the case, or with
            the case and its tools when ``tools`` is given (:data:`~threetears.evals.quick.tools.ToolUsingCandidate`).
            Returning an :class:`~threetears.evals.quick.answer.Answer` reports what producing the answer
            spent, which the summary and every result's ``cost_usd`` then carry; any other return value is
            the answer itself. Usually ``async def``; a plain ``def`` is called in a worker thread, so a
            blocking call stalls nothing else. A candidate handed tools, or a world's, must be async, since it
            awaits them. For a world run, ``candidate(case, tools)`` with the
            :class:`~threetears.evals.quick.world.WorldTools` on its cell's world.
        scorers: The grades. Each is reported as a measure named by its ``__name__``; ``True`` and
            ``False`` count as 1 and 0, and higher is better. None is needed when ``expected`` or
            ``judge`` is given.
        scope_id: The scope the template, cases and run are stored in. With no ``host`` and no ``store``, ``None``
            stores them in :data:`DEFAULT_QUICK_SCOPE`, inside the in-memory store this call builds and drops; with
            a host or a store of your own, name it, since that store outlives the call and the scope is where its
            runs are found again.
        expected: Declares the candidate a classifier: called once per case, it returns the label a
            correct answer gives. Each cell then lands ``match`` (the answer is that label) and
            ``confusion_cell`` (expected, then predicted), and the summary carries the confusion matrix
            and each label's precision, recall and F1. An answer that is not a non-blank string is
            counted as :data:`UNUSABLE_ANSWER`; any other answer is a label as written, whitespace and all.
        judge: A model grading each answer on a rubric (:class:`~threetears.evals.quick.judged.Judge`): one
            call per dimension per answer, through the engine's judge service. Each dimension's scores
            are summarised beside the measures, and the judge's spend as its client priced it. A judge
            call that fails excludes that cell, as any fault of the rig does. A "can't tell" is an answer,
            not a fault: it is recorded on its dimension, counted there (``cannot_tell`` in the summary's
            dimension line), and leaves the cell out of that dimension's mean alone — every other dimension
            and measure still counts the cell.
        intent: What every case asks of the candidate, in one sentence: the template's intent, which the
            judge reads beside each answer (as ``**Intent:**`` in its prompt) and so can move its scores.
            ``None`` takes the first line of the candidate's docstring, or, with no docstring, a generic
            sentence. A judged run's summary renders it with where it came from.
        host: Where to run and store: ``None`` builds :func:`callable_host` over the scorers and ``store``. A
            host of the caller's own must declare a measure for every scorer, and a contract for the callable
            kind (:data:`CALLABLE_KIND_CONTRACT`, or one seating only apparatus of its own — never anything in
            :data:`CALLABLE_UNSEATED`), and for a judged call one for the judged kind
            (:data:`JUDGED_CALLABLE_KIND_CONTRACT`, or one seating the judge and nothing in
            :data:`JUDGED_CALLABLE_UNSEATED`).
        store: Where the host ``run_eval`` builds stores the run (any
            :class:`~threetears.evals.contracts.DocumentStore`), so it outlives this call:
            ``store=SqliteDocumentStore("evals.sqlite")`` keeps it in a file another process reads through
            ``callable_host(<the same scorers>, store=...)``. ``None`` stores it in memory, for this call alone.
            Never with ``host``, which brings its own storage.
        k: Repeats per case.
        model: The arm's label, stored as the run's candidate model and keyed into its variant;
            ``None`` takes the candidate's ``__name__``.
        levers: The run's level of every other lever, by name (``{"prompt": "v2"}``), each a non-blank
            string naming the level. Each is frozen onto the run as an overlay and keyed into its variant
            beside the model, as the lever ``callable.<name>``. With no host, the one built declares exactly
            these levers; a host of the caller's own declares them through :func:`callable_kind_contracts`,
            and every run in it states a level of each.
        tools: The tools the candidate calls, by name: each a function of keyword arguments returning a
            JSON value, sync or async (:data:`~threetears.evals.quick.tools.Tool`). The candidate is handed
            them beside each case, as async functions, and they are what a cassette run records or replays.
        cassette_mode: ``'off'`` (the default) calls the tools live; ``'capture'`` calls them live and records
            what they answered into this run's corpus, named by its run id; ``'replay'`` serves the corpus
            ``cassette_corpus_id`` names in place of calling them. A replay asked something its capture
            never recorded is the rig's failure: that cell is excluded, never run live. Capture at ``k=1``:
            the corpus keeps one session per case, the last to run.
        cassette_corpus_id: For ``'replay'``, and only for it: the run id of a capture over the same cases
            (and expected labels or rubric) in this host and scope.
        world: The state the candidate acts on (:class:`~threetears.evals.quick.world.World`). Each cell seeds
            the case's starting state before the candidate's first turn and reads the world back after its last.
        seed: A world run's starting state for each case: takes the case, returns every dimension's value.
            Required with ``world``.
        goal_checks: Goal-state checks over the world the candidate left and the calls it made
            (``state.<dimension>``, ``calls("<world>.<tool>")``), each reported as passed or failed per cell.
            Requires ``world``. Held to the gate a template's authoring applies, before anything runs: a
            string comparison over a tool parameter is accepted only where the tool's schema closes it
            (``enum``, ``const`` or ``pattern``), since a free string is what the model wrote.
        max_cost_usd: The most the run may spend, in US dollars, counted from what each result reports: the
            candidate's :class:`~threetears.evals.quick.answer.Answer` spend and the judge's. Once the total
            passes it, or a result's spend goes unpriced, the run stops between cases with status
            ``budget_stopped``, keeps what it delivered, and the summary says why (``stopped_because``). A
            candidate that reports no spend is invisible to it, and the summary says so. ``None`` (the default)
            runs uncapped, which the summary states.

    Returns:
        The finished run's summary, read back from the store. It carries every result
        (:meth:`~threetears.evals.ops.summary.EvalSummary.results`, :meth:`~threetears.evals.ops.summary.EvalSummary.misses`),
        so the answers can be read after this call returns whatever store the run used.

    Raises:
        ValueError: No ``scope_id`` with a ``host`` or ``store`` of your own, no cases, a case that is not a JSON object with
            string keys, a case ``id`` that is not a non-blank string or an int or that two cases share, a ``max_cost_usd`` that is not a positive
            number, a synchronous candidate handed tools or a world, no scorer, ``expected``
            or ``judge``, an ``intent`` that is not a non-blank string, a scorer with no name, a repeated one or
            one named after an engine core measure (``match``, ``confusion_cell``, ``accuracy``, ``score``,
            ``cost_usd`` or any other name in ``METRIC_DESCRIPTORS``), an ``expected`` that raises or gives a case a blank, non-string or
            :data:`UNUSABLE_ANSWER` label, a given host that declares no callable-kind contract
            (or one with no seats, a seat in :data:`CALLABLE_UNSEATED`, overlays or a spec), a judged call on
            a given host whose judged-kind contract is missing or seats too much or no judge, a scorer
            the given host declares no measure for, no ``model`` for a candidate that has no ``__name__``, an
            unusable lever name, ``tools`` that are no non-empty mapping of identifier to function, or a
            capture or replay of a candidate that declares no tools; for a world run, no ``seed=``, a
            ``seed=`` that raises or gives a case a state the world refuses or leaves a dimension
            unset, a goal check reading state or naming a tool the world lacks, or a given host that
            does not declare the world; ``seed=`` or ``goal_checks=`` with no ``world=``;
            ``world=`` with ``tools=`` or a cassette mode other than ``'off'``; and ``store=`` with ``host=``.
        ValidationFailedError: The launch refused: a ``k`` outside the run's bounds, ``levers`` naming a
            lever the host does not declare, leaving out one it does, or giving one a blank or non-string
            level, a replay naming no corpus or one that is no capture of these cases in this scope, or a
            corpus named off replay.
    """
    (summary,) = await run_arms(
        cases,
        [CallableArm(candidate, model, levers)],
        scorers,
        scope_id=scope_id,
        expected=expected,
        judge=judge,
        intent=intent,
        world=world,
        seed=seed,
        goal_checks=goal_checks,
        host=host,
        store=store,
        k=k,
        tools=tools,
        cassette_mode=cassette_mode,
        cassette_corpus_id=cassette_corpus_id,
        max_cost_usd=max_cost_usd,
    )
    return summary


def _template_intent(arms: Sequence[CallableArm], graded_by: str, intent: str | None) -> tuple[str, str | None]:
    """What the one template every arm shares says its cases ask, and where that came from.

    ``intent`` when given; else the candidates' docstring, when every arm's has one and they share its
    first line; else a generic sentence.

    Returns:
        The intent, and its source as a summary names it: ``None`` for the ``intent`` given, else a phrase
        saying which docstring it was read from or why it is the generic sentence.
    """
    if intent is not None:
        return intent, None
    firsts = {doc.splitlines()[0] if (doc := inspect.getdoc(arm.candidate)) else None for arm in arms}
    first = next(iter(firsts)) if len(firsts) == 1 else None
    if len(arms) > 1:
        if first:
            return first, "from the docstring every arm's candidate shares"
        lacking = "the arms' candidates share no docstring first line"
    else:
        name = getattr(arms[0].candidate, "__name__", None) or arms[0].model or "the candidate"
        if first:
            return first, f"from {name}'s docstring"
        lacking = f"{name} has no docstring"
    return f"Answer each case so that {graded_by} the answer well.", f"a generic default: no intent=, and {lacking}"


async def run_arms(
    cases: Sequence[Mapping[str, Any]],
    arms: Sequence[CallableArm],
    scorers: Sequence[Scorer] = (),
    *,
    scope_id: str | None = None,
    expected: ExpectedLabel | None = None,
    judge: Judge | None = None,
    intent: str | None = None,
    world: World | None = None,
    seed: CaseSeed | None = None,
    goal_checks: Sequence[str] = (),
    host: EvalHost | None = None,
    store: DocumentStore | None = None,
    k: int = DEFAULT_LAUNCH_K_RUNS,
    tools: Mapping[str, Tool] | None = None,
    cassette_mode: CassetteMode = "off",
    cassette_corpus_id: str | None = None,
    max_cost_usd: float | None = None,
    margins: Mapping[str, float] | None = None,
) -> list[EvalSummary]:
    """Run every arm over every case ``k`` times as ONE launch, and summarise each arm's run, in arm order.

    :func:`run_eval` is the launch of one arm, and :func:`~threetears.evals.quick.compare` the launch of
    several. The arms share the template, the case set, the host and the launch: one launch group, every
    arm's run prepared (every refusal made) before any starts, and all of them started together
    (:func:`~threetears.evals.run.launch_as_group`) — so a comparison's arms are measured side by side,
    and a refusal on the last arm leaves none run. Each arm is its own
    :func:`~threetears.evals.run.start_run` call into that group, because ``start_run`` launches every run
    it starts at one set of overlays and the arms of a factorial differ in theirs.

    Args and Raises as :func:`run_eval`, every argument but the arms the same for every arm — one judge, so
    every arm is judged by the same model, rubric and judge configs, and one ``max_cost_usd``, each arm's
    run's own cap. The arms are distinct — no two at one model and one level of every lever — which the
    caller holds. ``margins`` are run-scoped margins on core rate measures (``{"accuracy": 0.05}``), declared
    on every arm's run alike (:func:`~threetears.evals.run.start_run`'s ``margins``).

    Returns:
        Each arm's finished run's summary, read back from the store, in arm order.
    """
    refuse_a_store_beside_a_host(store, host)
    scope_id = quick_scope(scope_id, host, store)
    plain_cases = _plain_cases(cases)
    names = _case_names(plain_cases)
    _refuse_an_unusable_cap(max_cost_usd)
    goal_checks = list(goal_checks)
    if not scorers and expected is None and judge is None and not goal_checks:
        raise ValueError(
            "run_eval needs at least one scorer, a classifier's expected labels (expected=) or a judge (judge=), "
            "or a world's goal checks (goal_checks=): a run nothing grades measures nothing"
        )
    _refuse_unnamed_or_repeated(scorers)
    if intent is not None and (not isinstance(intent, str) or not intent.strip()):
        raise ValueError(
            f"intent= is the sentence the judge reads as what each case asks: a non-blank string, not {intent!r}"
        )
    if world is not None and (tools is not None or cassette_mode != "off"):
        # A world's tools act on its state, so a replay that skipped them would grade a world nothing changed.
        raise ValueError(
            "a world run's tools are the world's own (World(tools=...)): pass no tools= beside world=, and no "
            "cassette_mode, since a replayed world tool would leave the state it should have changed untouched"
        )
    refuse_unusable_tools(tools, cassette_mode)
    _refuse_a_sync_candidate_handed_tools(
        arms, why="a world's tools" if world is not None else "tools" if tools is not None else None
    )
    labels = None if expected is None else _expected_labels(plain_cases, expected, names)
    seeds = _world_seeds(plain_cases, world, seed, goal_checks, host, names)
    if host is None:
        host = callable_host(scorers, levers=tuple(arms[0].levers or ()), world=world, store=store)
    else:
        _refuse_an_undeclared_callable_contract(host, judged=judge is not None)
        _refuse_undeclared_measures(host, scorers)
    models: list[str] = []
    for arm in arms:
        model = arm.model
        if model is None:
            model = getattr(arm.candidate, "__name__", None)
            if not model:
                raise ValueError(f"{arm.candidate!r} has no __name__ to label its arm by; pass model=")
        models.append(model)
    graded_by = "every scorer grades" if judge is None else "the judge and every scorer grade"
    template_intent, intent_source = _template_intent(arms, graded_by, intent)
    template_id = _template_id(
        plain_cases, labels, judge, seeds, goal_checks, intent=template_intent if judge is not None else None
    )
    template = EvalTemplate(
        id=template_id,
        scope_id=scope_id,
        name=f"run_eval over {len(plain_cases)} case(s)",
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
            variation_params=stored_variation(case),
            host_payload=_case_payload(
                case, None if labels is None else labels[index], None if seeds is None else seeds[index]
            ),
        )
        for index, case in enumerate(plain_cases)
    ]
    if world is not None:
        # The authoring gate itself, before anything is stored or run: the grammar, the paths, a string match
        # over model prose, a text comparison over a call parameter its schema does not close, and calls or
        # firings the world cannot produce. A quick run is authored here, so it is held to what authoring holds.
        try:
            refuse_unsupplied_world(template, profile=host.profile)
        except ValidationFailedError as refused:
            raise ValueError(refused.message) from refused
    host.storage.save_template(template)
    for test_case in test_cases:
        host.storage.save_test_case(test_case)
    # Each scorer's declared range, which every score it returns is held to: the bounded tests read it as a fact.
    ranges = {
        name: descriptor.value_range
        for name in map(_scorer_name, scorers)
        if (descriptor := host.profile.measures.get(name)) is not None and descriptor.value_range is not None
    }
    wired = [
        _WiredArm(
            model=model,
            levers=dict(arm.levers or {}),
            kind_factory=_kind_factory(
                arm.candidate,
                scorers,
                classifies=labels is not None,
                judge=judge,
                world=world,
                tools=tools,
                ranges=ranges,
            ),
            arm=arm.arm,
        )
        for arm, model in zip(arms, models, strict=True)
    ]
    launch_host = _launch_host(
        host, wired, test_cases, judge, world, calls_tools=tools is not None, max_cost_usd=max_cost_usd
    )

    async def form() -> tuple[LaunchGroup, None]:
        # Every arm's model is a candidate of the launch, which is what its judge is chosen against.
        return LaunchGroup(candidate_models=list(dict.fromkeys(models))), None

    async def prepare(group: LaunchGroup, _formed: None) -> list[EvalRun]:
        prepared: list[EvalRun] = []
        for arm, model in zip(arms, models, strict=True):
            prepared += await start_run(
                launch_host,
                template_id=template_id,
                subject_id=model if arm.arm is None else arm.arm,
                models=[model],
                k_runs=k,
                scope_id=scope_id,
                judge_model=judge.model if judge is not None else None,
                overlays=dict(arm.levers) if arm.levers else None,
                cassette_mode=cassette_mode,
                cassette_corpus_id=cassette_corpus_id,
                max_cost_usd=max_cost_usd,
                launch_group=group,
                margins=margins,
            )
        return prepared

    runs = await launch_as_group(
        launch_host,
        len(arms),
        settings=launch_host.settings(),
        form=form,
        prepare=prepare,
        event="eval.run_eval",
    )
    try:
        await launch_host.job_manager.wait_for([run.id for run in runs])
    except asyncio.CancelledError:
        # This call owns the runs it started: cancelled, it settles them as cancelled rather than
        # leaving pending ones behind in a store that outlives it.
        await launch_host.job_manager.shutdown()
        raise
    by_id = {test_case.id: name for test_case, name in zip(test_cases, names, strict=True)}
    summaries = [
        _with_case_results(host, summarize_run(host, run.id, scope_id, case_names=by_id), test_cases, names)
        for run in runs
    ]
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
    host: EvalHost, summary: EvalSummary, test_cases: list[EvalTestCase], names: list[str]
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
    breached_above_zero = frozenset(
        name
        for name in host.profile.measures.names
        if (descriptor := host.profile.measures.get(name)) is not None
        and descriptor.guardrail
        and descriptor.higher_is_better is False
    )
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
                breached_above_zero=breached_above_zero,
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
    "SHARED_ARM_MODEL",
    "UNUSABLE_ANSWER",
    "CallableKind",
    "Candidate",
    "ExpectedLabel",
    "Scorer",
    "callable_host",
    "callable_kind_contracts",
    "run_eval",
]
