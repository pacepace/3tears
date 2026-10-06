"""``run_eval``: a case list, an async candidate and scorer functions, run through the engine in one call.

Rung 0 of adopting the engine. A product with nothing but a function to test and a way to grade its
answer hands over three values and gets a summary back; everything else a host would build is built
here, from the public roots, on the terms the engine already sets:

- **The kind** is :class:`CallableKind`: ``invoke`` calls the candidate on the case and each scorer
  on what it returned, and reports the scores as the host's measures. It is unjudged — the scorers
  are the grade — and seeds no world.
- **The host**, when none is given, is :func:`callable_host`: the shared sweepable core, one measure
  per scorer, no world, and the in-memory reference store. Given one, its storage and vocabulary are
  used: every scorer must already be a measure it declares, and it must declare a contract for the
  callable kind that seats no judge, simulator or spend ceiling.
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
from dataclasses import dataclass
from typing import Any

from threetears.evals.contracts import (
    DEFAULT_LAUNCH_K_RUNS,
    CandidateOutput,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    EvalRun,
    EvalStorage,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    MetricDescriptor,
    VariantConfig,
    WorldSeed,
    WorldSession,
    canonical_digest,
    withhold_failure_detail,
)
from threetears.evals.contracts.host import (
    SHARED_CORE,
    EvalHost,
    HostProfile,
    KindContract,
    MeasureRegistry,
    SubjectSnapshot,
    WorldPlacement,
    default_cell_timeout,
)
from threetears.evals.ops.summary import EvalSummary, summarize_run
from threetears.evals.run import (
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    default_job_timeout,
    launch_run,
    start_run,
)
from threetears.evals.storage import InMemoryDocumentStore

#: The candidate under test: an async callable taking one case and returning its answer.
Candidate = Callable[[Mapping[str, Any]], Awaitable[Any]]

#: One grade: takes the case and the candidate's answer, returns a number (``True``/``False`` count
#: as 1 and 0). Its ``__name__`` is the measure's name, and higher is better.
Scorer = Callable[[Mapping[str, Any], Any], float | bool]

#: The kind :func:`run_eval` launches, as its template names it.
CALLABLE_KIND = "callable"

#: The host :func:`callable_host` builds, by the id the engine prints in its logs and errors.
CALLABLE_HOST_ID = "run_eval"

#: Where a case rides on its stored test case: verbatim, for the candidate to be handed back.
_CASE_KEY = "case"

#: The callable kind's contract: no overlays, no spec, and no rig seat — nothing in a ``run_eval`` run is
#: graded by a model or talks to a simulated user. Without the empty seats each blank judge and simulator
#: dimension would read as an unrecoverable judge or simulator and confound every comparison of two such
#: runs. A host of the caller's own declares a contract for the callable kind on its profile too —
#: this one, or one seating apparatus of its own the runs do read — and :func:`run_eval` refuses a host
#: that does not (:data:`CALLABLE_UNSEATED`).
CALLABLE_KIND_CONTRACT = KindContract(CALLABLE_KIND, seats=frozenset())

#: What a ``run_eval`` run never has, whatever host it runs in, so a callable-kind contract may seat none of
#: it: the engine's judge and simulator (by role or by any pinned dimension) — the callable kind is unjudged
#: and simulates nobody — and the spend ceiling, which the one-call launch leaves off. A seat here would make
#: every blank such dimension an unrecoverable level, and every comparison of two ``run_eval`` runs ``undecided``.
CALLABLE_UNSEATED: frozenset[str] = frozenset(
    {
        *(name for role in SHARED_CORE.roles for name in (role.name, *role.pins)),
        "max_cost_usd",
    }
)


def _launch_settings() -> LaunchSettings:
    """The launch settings of the one-call path.

    One arm, one admitted run at a time, and the cost and metered-call ceilings off: the candidate is
    an opaque callable whose spend the engine cannot see, so a ceiling would bind nothing. The
    ceiling values are required by the settings model and, with enforcement off, recorded as absent
    on the run rather than as caps; the out-of-run one binds nothing either, since the callable kind
    declines ``n_variations`` and so never generates.
    """
    return LaunchSettings(
        max_launch_arms=1,
        max_admitted_runs=1,
        judge_concurrency=1,
        enforcement_enabled=False,
        max_cost_usd=1.0,
        max_metered_calls=1,
        max_out_of_run_cost_usd=1.0,
    )


def scorer_measure(scorer: Scorer) -> MetricDescriptor:
    """The measure one scorer function reports: a quality score, higher is better, over scored results.

    Args:
        scorer: The scorer. Its ``__name__`` names the measure and the first line of its docstring,
            when it has one, describes it.

    Returns:
        The descriptor :func:`callable_host` registers for it.
    """
    name = _scorer_name(scorer)
    doc = inspect.getdoc(scorer)
    return MetricDescriptor(
        name=name,
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=doc.splitlines()[0] if doc else f"The score the {name} function gave the candidate's answer.",
        higher_is_better=True,
        merit_axis="quality",
        population="scored",
    )


def callable_host(scorers: Sequence[Scorer]) -> EvalHost:
    """The least host there is: the shared core, one measure per scorer, no world, an in-memory store.

    What :func:`run_eval` builds when it is handed no host. Its store lives as long as the returned
    value, so a caller wanting to run several candidates into one store and compare them builds this
    once and hands it to each call.

    Args:
        scorers: The scorer functions whose measures the host declares.

    Returns:
        The host.

    Raises:
        ValueError: A scorer has no usable name, or two share one.
    """
    _refuse_unnamed_or_repeated(scorers)
    return EvalHost(
        profile=HostProfile(
            host_id=CALLABLE_HOST_ID,
            host_sweepables=SHARED_CORE,
            measures=MeasureRegistry(scorer_measure(scorer) for scorer in scorers),
            kinds=(CALLABLE_KIND_CONTRACT,),
        ),
        storage=EvalStorage(InMemoryDocumentStore()),
        failure_describer=withhold_failure_detail,
        trace_sink=None,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
    )


@dataclass(frozen=True)
class _Prepared:
    """One cell's candidate, as ``prepare`` hands it to ``invoke``."""

    model: str


class CallableKind:
    """The candidate-kind seam over a plain async callable and its scorer functions.

    ``invoke`` calls the candidate with the case it was given and each scorer with that case and the
    answer, and reports the scores as host measures. A candidate that raises FAILS its cell (a
    candidate error lowers the score; a broken candidate must not vanish); a scorer that raises, or
    returns something that is not a number, EXCLUDES it, because the grader is the rig rather than
    the thing under test.
    """

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, candidate: Candidate, scorers: Sequence[Scorer]) -> None:
        """Bind the candidate and its scorers.

        Args:
            candidate: The async callable under test.
            scorers: The grades, each reported under its own name.
        """
        self._candidate = candidate
        self._scorers = tuple(scorers)

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
        """Nothing to build: the candidate is already a callable. Records the arm's model label.

        Args:
            subject_snapshot: The run's subject, unused.
            variant_config: The cell's variant; its candidate model labels the arm.
            world_seed: The template's seed, which this kind has no world to write.
            span_window: The cell's trace windows, unused.
            cassettes: Always ``None``: the launch declines cassettes for this kind.
            world: Unread — a callable has no world to seed, so the cell opens none.

        Returns:
            The prepared candidate.
        """
        return _Prepared(model=variant_config.candidate_model)

    async def invoke(self, instance: _Prepared, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Run the candidate on one case, grade its answer, and report the grades.

        Args:
            instance: What ``prepare`` returned.
            test_case: The cell's case; the caller's case rides on its ``host_payload``.
            sink: The cell's sink, unused: the scores are reported on the output.

        Returns:
            The answer as the stored trace, and each scorer's grade as a host measure.
        """
        case = test_case.host_payload[_CASE_KEY]
        try:
            answer = await self._candidate(case)
        # prawduct:ok-broad-except — the candidate is the caller's code under test: whatever it raises is its failure, recorded on the cell
        except Exception as raised:
            return CandidateOutput(candidate_errors=[f"the candidate raised {type(raised).__name__}: {raised}"])
        trace = [_as_stored(answer)]
        measures: dict[str, bool | float | str] = {}
        for scorer in self._scorers:
            name = _scorer_name(scorer)
            try:
                score = scorer(case, answer)
            # prawduct:ok-broad-except — a scorer is the caller's grader, the rig: a raise excludes the cell and says which scorer
            except Exception as raised:
                return CandidateOutput(
                    output=trace, infra_errors=[f"the scorer {name} raised {type(raised).__name__}: {raised}"]
                )
            # ``bool`` is an ``int``, so True and False pass here as 1 and 0.
            if not isinstance(score, int | float) or not math.isfinite(score):
                return CandidateOutput(
                    output=trace, infra_errors=[f"the scorer {name} returned {score!r}, not a finite number or a bool"]
                )
            measures[name] = float(score)
        return CandidateOutput(output=trace, host_measures=measures)


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
    if not scorers:
        raise ValueError("run_eval needs at least one scorer: a run nothing grades measures nothing")
    names = [_scorer_name(scorer) for scorer in scorers]
    if unnamed := [repr(scorer) for scorer, name in zip(scorers, names, strict=True) if not name.isidentifier()]:
        raise ValueError(
            f"a scorer's __name__ names its measure, and {', '.join(unnamed)} has none a measure can carry; "
            "write each scorer as a def"
        )
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"scorers named {', '.join(repeated)} more than once; each name is one measure")


def _refuse_an_undeclared_callable_contract(host: EvalHost) -> None:
    """Refuse a caller's host that has not declared what a ``run_eval`` run's rig holds.

    A host with no contract for the callable kind holds its runs to every apparatus dimension
    (:meth:`~threetears.evals.contracts.host.profile.HostProfile.kind_contract`), so the blank judge and
    simulator of every such run reads as an unrecoverable one and confounds every comparison of two of
    them, silently. Refused here, before anything is stored, rather than discovered in an analysis.
    """
    profile = host.profile
    contract = next((declared for declared in profile.kinds if declared.kind == CALLABLE_KIND), None)
    remedy = (
        f"declare the callable kind's contract on its profile's kinds — CALLABLE_KIND_CONTRACT, or a "
        f"KindContract({CALLABLE_KIND!r}, seats=...) seating only apparatus of the host's own that the runs read"
    )
    if contract is None or contract.seats is None:
        what = "no contract" if contract is None else "a contract that declares no seats"
        raise ValueError(
            f"host {profile.host_id!r} has {what} for the {CALLABLE_KIND!r} kind, so every run_eval run would be "
            f"held to the judge and simulator it never has and every comparison of two would read undecided; {remedy}"
        )
    if seated := sorted(contract.seats & CALLABLE_UNSEATED):
        raise ValueError(
            f"host {profile.host_id!r} seats {', '.join(seated)} for the {CALLABLE_KIND!r} kind, which a run_eval run "
            f"never has — it is unjudged, simulates nobody and runs uncapped; {remedy}"
        )
    if contract.overlays is not None or contract.spec is not None:
        raise ValueError(
            f"host {profile.host_id!r} declares overlays or a spec for the {CALLABLE_KIND!r} kind, which run_eval "
            f"neither turns nor states; {remedy}"
        )


def _refuse_undeclared_measures(host: EvalHost, scorers: Sequence[Scorer]) -> None:
    measures = host.profile.measures
    if undeclared := [name for scorer in scorers if measures.get(name := _scorer_name(scorer)) is None]:
        raise ValueError(
            f"host {host.profile.host_id!r} declares no measure named {', '.join(undeclared)}; a host's scorers "
            "report into measures it declares, so register them on its profile's measures"
        )


def _case_set(cases: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The template id the case list is addressed by, and the cases as plain JSON objects."""
    if isinstance(cases, Mapping | str) or not cases:
        raise ValueError("run_eval needs a non-empty list of cases, each a mapping")
    plain: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or not all(isinstance(key, str) for key in case):
            raise ValueError(f"case {index} is not a mapping with string keys: {case!r}")
        plain.append(dict(case))
    try:
        digest = canonical_digest(plain)
    except TypeError as unencodable:
        raise ValueError(f"every case must be JSON: {unencodable}") from unencodable
    return f"{CALLABLE_HOST_ID}-{digest[:16]}", plain


def _case_payload(case: dict[str, Any]) -> dict[str, Any]:
    """The ``host_payload`` a case's stored test case carries: the case, verbatim, under this module's key."""
    return {_CASE_KEY: case}


def _flat(value: Any) -> str:
    """One case field as the engine's flat-string view of what varies."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _launch_host(host: EvalHost, kind: CallableKind, subject: SubjectSnapshot, cases: list[EvalTestCase]) -> LaunchHost:
    """``host`` as a launching host whose one kind runs ``kind`` over ``cases``."""
    world = host.profile.world

    def place(_run: EvalRun) -> dict[str, WorldPlacement]:
        # This kind attaches no carrier, so every dimension a host's world declares is out of play.
        assert world is not None
        return world.place(seeded=(), carriers=())

    async def launch(request: LaunchRequest) -> EvalRun:
        return await launch_run(
            launch_host, request, KindWiring(kind_factory=lambda _cell: kind, subject=subject, test_cases=cases)
        )

    launch_host = LaunchHost(
        eval_host=host,
        kinds={
            CALLABLE_KIND: LaunchableKind(
                launch=launch,
                unhonoured_launch_arguments=frozenset(
                    {"simulator_model", "judge_model", "judge_config_ids", "cassette_mode", "n_variations"}
                ),
            )
        },
        settings=_launch_settings,
        job_timeout_factory=default_job_timeout,
        world_placements=place if world is not None else None,
    )
    return launch_host


async def run_eval(
    cases: Sequence[Mapping[str, Any]],
    candidate: Candidate,
    scorers: Sequence[Scorer],
    *,
    scope_id: str,
    host: EvalHost | None = None,
    k: int = DEFAULT_LAUNCH_K_RUNS,
    model: str | None = None,
) -> EvalSummary:
    """Run ``candidate`` on every case ``k`` times, grade each answer with every scorer, and summarise.

    Args:
        cases: The cases, each a JSON object; the candidate and the scorers receive each one as given.
        candidate: The async callable under test, called once per case and repeat.
        scorers: The grades. Each is reported as a measure named by its ``__name__``; ``True`` and
            ``False`` count as 1 and 0, and higher is better.
        scope_id: The scope the template, cases and run are stored in. The engine never defaults it.
        host: Where to run and store: ``None`` builds :func:`callable_host` over the scorers, whose
            in-memory store lives only as long as this call. A host of the caller's own must declare
            a measure for every scorer, and a contract for the callable kind (:data:`CALLABLE_KIND_CONTRACT`,
            or one seating only apparatus of its own — never anything in :data:`CALLABLE_UNSEATED`).
        k: Repeats per case.
        model: The arm's label, stored as the run's candidate model and keyed into its variant;
            ``None`` takes the candidate's ``__name__``.

    Returns:
        The finished run's summary, read back from the store.

    Raises:
        ValueError: No cases, a case that is not a JSON object with string keys, no scorers, a
            scorer with no name or a repeated one, a given host that declares no callable-kind contract
            (or one with no seats, a seat in :data:`CALLABLE_UNSEATED`, overlays or a spec), a scorer
            the given host declares no measure for, or no ``model`` for a candidate that has no ``__name__``.
        ValidationFailedError: The launch refused: a ``k`` outside the run's bounds.
    """
    template_id, plain_cases = _case_set(cases)
    _refuse_unnamed_or_repeated(scorers)
    if host is None:
        host = callable_host(scorers)
    else:
        _refuse_an_undeclared_callable_contract(host)
        _refuse_undeclared_measures(host, scorers)
    if model is None:
        model = getattr(candidate, "__name__", None)
        if not model:
            raise ValueError(f"{candidate!r} has no __name__ to label its arm by; pass model=")
    doc = inspect.getdoc(candidate)
    template = EvalTemplate(
        id=template_id,
        scope_id=scope_id,
        name=f"run_eval over {len(plain_cases)} case(s)",
        intent=doc.splitlines()[0] if doc else "Answer each case so that every scorer grades the answer well.",
        candidate_kind=CALLABLE_KIND,
    )
    test_cases = [
        EvalTestCase(
            id=f"{template_id}-{index}",
            scope_id=scope_id,
            template_id=template_id,
            variation_params={key: _flat(value) for key, value in case.items()},
            host_payload=_case_payload(case),
        )
        for index, case in enumerate(plain_cases)
    ]
    host.storage.save_template(template)
    for test_case in test_cases:
        host.storage.save_test_case(test_case)
    subject = SubjectSnapshot(subject_id=model, subject_label=model, state={})
    launch_host = _launch_host(host, CallableKind(candidate, scorers), subject, test_cases)
    runs = await start_run(
        launch_host, template_id=template_id, subject_id=model, models=[model], k_runs=k, scope_id=scope_id
    )
    try:
        await launch_host.job_manager.wait_for([run.id for run in runs])
    except asyncio.CancelledError:
        # This call owns the run it started: cancelled, it settles the run as cancelled rather than
        # leaving a pending one behind in a store that outlives it.
        await launch_host.job_manager.shutdown()
        raise
    (run,) = runs
    return summarize_run(host, run.id, scope_id)


__all__ = [
    "CALLABLE_HOST_ID",
    "CALLABLE_KIND",
    "CALLABLE_KIND_CONTRACT",
    "CALLABLE_UNSEATED",
    "CallableKind",
    "Candidate",
    "Scorer",
    "callable_host",
    "run_eval",
]
