"""A second host, as small as adopting the engine can be: a courier route planner.

The toy host (``fixtures/toyhost``) exercises every shape of the host contract. This is the other
kind of example: the least a product writes to run and analyse its own candidate through the
engine, in vocabulary that shares nothing with the toy host's but one measure NAME — so the two can
run side by side in one event loop, each bundle carrying its own words and none of the other's.

**The one shared name is deliberate.** Both hosts register ``field_accuracy`` and mean different
things by it — the toy host a share of invoice fields extracted right (higher is better, no unit),
the courier the minutes its promised arrivals missed by in the field (lower is better). Two
products naming a measure alike is ordinary, and it is the only case in which a measure description
cached by name across hosts gives a visibly wrong answer: with no name in common, every such cache
returns what the asking host would have resolved anyway.

What a product supplies, in the order it is written below:

1. **Its measures** — what it can see of a result, on the engine's four merit axes.
2. **Its levers and apparatus** — the sweepables registry, extending the engine's shared core,
   with readers over the engine-owned ``host_payload`` slot of a run.
3. **Its world** — here one dimension a scenario seeds, which the planner never perceives (the
   judge-only quadrant).
4. **Its profile**, assembling the three, plus how one observation resolves its levers.
5. **Its candidate kind** — ``prepare`` seeds the world through the engine's seed walk, ``invoke``
   asks the planner model and reports its measures, or routes the model's failed call by whose it was.
6. **Its host** — the profile, a store, and the three services with no safe default.
7. **Its launching host**, and **8. its own action**, served beside the engine's.

Everything else — the trial loop, the stored documents, the bundle — is the engine's.

**It is driven two ways.** :func:`run_courier_campaign` drives the runner directly
(:func:`~threetears.evals.run.execute_run`), the low-level path: one assembled run's matrix, with the
run built and its status set by the caller. :func:`courier_launch_host` is the way a product serving
launches adopts the engine: a :class:`~threetears.evals.run.LaunchHost` whose one kind's launcher
:func:`~threetears.evals.run.start_run` dispatches to, which the engine's command line drives
(from a checkout's root, ``python -m threetears.evals run --host
packages.evals.tests.fixtures.courierhost:courier_launch_host``; with this file copied out as ``courierhost.py``,
``--host courierhost:courier_launch_host`` from beside it) and waits on through the job manager's ``wait_for``.
:func:`courier_catalogue` serves the engine's actions with the courier's own added, over :func:`courier_ops_host`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from threetears.evals.actions import Action, ActionCatalogue, Caller, eval_catalogue
from threetears.evals.kernel import default_job_timeout
from threetears.evals.kernel import (
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    EvalCampaign,
    EvalStorage,
    ValidationFailedError,
    MetricDescriptor,
    WorldSession,
    VariantConfig,
    describe_and_log_failure,
    failed_call_output,
    withhold_failure_detail,
)
from threetears.evals.schema import (
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    RoleUsage,
    WorldSeed,
)
from threetears.evals.kernel.host import (
    SHARED_CORE,
    EvalHost,
    HostProfile,
    KindContract,
    MeasureRegistry,
    MemberActsOn,
    Ordinal,
    SeedRefused,
    Sweepable,
    TemplateChecks,
    WorldDimension,
    WorldRegistry,
    default_cell_timeout,
)
from threetears.evals.ops import JobsStarted, OpsHost, start_action_job
from threetears.evals.schema import (
    IntervalScale,
    ProviderFailure,
    ProviderFailureDescriber,
    SubjectSnapshot,
    SweepableValue,
)
from threetears.evals.run import (
    KindWiring,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    LaunchableKind,
    RunnerOptions,
    UnknownSubject,
    execute_run,
    launch_run,
    run_blocking,
)
from threetears.evals.storage import InMemoryDocumentStore

COURIER_ID = "courier"
COURIER_SCOPE = "courier-depot-north"
COURIER_KIND = "route_planner"
_PAYLOAD = "courier"

_log = logging.getLogger(__name__)

# --- 1. measures -----------------------------------------------------------------------------------

ON_TIME_RATE = "on_time_rate"
DETOUR_KM = "detour_km"
#: The name the toy host also registers, for a different measure — see the module docstring.
FIELD_ACCURACY = "field_accuracy"

COURIER_MEASURES = (
    MetricDescriptor(
        name=ON_TIME_RATE,
        reader_name="On-time rate",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Share of a round's stops reached inside their delivery window.",
        reader_prose="how many stops the route reached on time",
        higher_is_better=True,
        value_range=(0.0, 1.0),
        merit_axis="quality",
        population="scored",
    ),
    MetricDescriptor(
        name=DETOUR_KM,
        reader_name="Detour",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Kilometres driven beyond the shortest open route through the round's stops.",
        reader_prose="how far the route strayed from the shortest open path",
        higher_is_better=False,
        unit="km",
        merit_axis="cost",
        population="all_observed",
    ),
    MetricDescriptor(
        name=FIELD_ACCURACY,
        reader_name="Arrival-time miss",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="Mean minutes between each promised arrival and the arrival the driver reported from the field.",
        reader_prose="how far the promised arrival times missed the driver's",
        higher_is_better=False,
        unit="min",
        merit_axis="reliability",
        population="all_observed",
    ),
)

# --- 2. levers and apparatus -----------------------------------------------------------------------


def _read(key: str) -> Any:
    def read(run: EvalRun, _results: Sequence[EvalResult]) -> Any:
        return (run.host_payload or {}).get(_PAYLOAD, {}).get(key)

    return read


COURIER_SWEEPABLES = SHARED_CORE.extend(
    (
        Sweepable(
            name="search_depth",
            role="lever",
            reader_name="search depth",
            read=_read("search_depth"),
            reader_prose="how many alternative orderings the planner tried per stop",
        ),
        Sweepable(
            name="traffic_feed",
            role="apparatus",
            read=_read("traffic_feed"),
            reader_prose="which traffic feed the planner's travel times came from",
            confounds="a different traffic feed priced the roads, so a faster route may be a kinder feed",
        ),
    )
)

# --- 3. world ----------------------------------------------------------------------------------------


@dataclass
class Depot:
    """The world's state: what a scenario seeds before the planner starts."""

    road_closures: int = 0


def courier_world() -> WorldRegistry:
    """One dimension a run sets and the planner is never told about — the judge-only quadrant."""
    depot = Depot()
    return WorldRegistry(
        (
            WorldDimension(
                name="road_closures",
                carrier="dispatch",
                schema={"type": "integer", "minimum": 0},
                matters="A scenario probing re-routing presumes roads are closed; an open map measures nothing.",
                seed="courier.seed_closures",
                read="courier.read_closures",
            ),
        ),
        bindings={
            "courier.seed_closures": lambda value: setattr(depot, "road_closures", value),
            "courier.read_closures": lambda: depot.road_closures,
        },
    )


# --- 4. profile ------------------------------------------------------------------------------------


def courier_levers(run: EvalRun) -> dict[str, SweepableValue]:
    """A run's level of the courier's own lever, its search depth; the engine resolves the planner model.

    A run of another kind in the courier's store — an eval's run over a plain function,
    say — has no search depth, and sits at the engine's "not a run of this kind" level rather than at a
    number it never had: that level is the one a report knows not to name an arm by.
    """
    if run.candidate_kind != COURIER_KIND:
        return {"search_depth": SweepableValue.not_this_kind(COURIER_KIND)}
    depth = (run.host_payload or {}).get(_PAYLOAD, {}).get("search_depth")
    return {"search_depth": SweepableValue.of(depth, scale=IntervalScale(value=float(depth), unit=None))}


#: The zone the north depot's rounds run in.
DEPOT_ZONE = "downtown"


class CourierOverlays(BaseModel):
    """The knobs a launch of the planner may turn. Each default is how the planner has always run."""

    #: ``Ordinal()`` ranks the levels in the order written, so a report reads them as less and more search rather than
    #: as three unrelated names. ``auto`` is the planner's own choice (exhaustive on a ``-pro`` model, greedy
    #: otherwise), so on every model it searches at least as hard as ``greedy`` and at most as hard as ``exhaustive``.
    solver: Annotated[Literal["greedy", "auto", "exhaustive"], Ordinal()] = Field(
        default="auto", title="solver", description="how hard the planner searches for a round's ordering"
    )
    #: A map is an open family: each zone a launch sets is a knob of its own. ``MemberActsOn`` names the measure an
    #: entry is meant to move, so a report checks that more downtown slack moved the on-time rate, and says so.
    zone_slack_min: Annotated[dict[str, int], MemberActsOn({DEPOT_ZONE: ON_TIME_RATE})] = Field(
        default_factory=dict, title="zone slack", description="minutes of slack added to each stop's window, by zone"
    )


#: The seats the planner fills: the traffic feed it plans against, and its spend ceiling. Nothing here is
#: graded by a model or talks to a simulated user, so the judge and simulator seats are not filled. Its
#: store also receives an eval's runs, which read the same feed and run uncapped. Its overlays are the knobs above.
COURIER_CONTRACT = KindContract(
    COURIER_KIND, overlays=CourierOverlays, seats=frozenset({"traffic_feed", "max_cost_usd"})
)

#: The kind an eval's runs are of (``Eval(...).run(host=...)``), by the name its template states: the courier declares
#: it so the evals it runs into its store are held to the seats they fill, the traffic feed and nothing of the engine's.
EVAL_KIND = "callable"
CALLABLE_CONTRACT = KindContract(EVAL_KIND, seats=frozenset({"traffic_feed"}))


def courier_profile() -> HostProfile:
    """The courier's vocabulary, with a fresh world."""
    return HostProfile(
        host_id=COURIER_ID,
        host_sweepables=COURIER_SWEEPABLES,
        measures=MeasureRegistry(COURIER_MEASURES),
        world=courier_world(),
        variant_levers=courier_levers,
        kinds=(COURIER_CONTRACT, CALLABLE_CONTRACT),
    )


# --- 5. candidate kind -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Planned:
    model: str
    road_closures: int


class PlannerOverloaded(Exception):
    """The planner model's endpoint turned a call away because it is overloaded; the next call may be served."""


@dataclass(frozen=True)
class _Route:
    late: int
    detour_km: float
    eta_miss_min: float


async def ask_planner(
    model: str, *, stops: int, road_closures: int, exhaustive: bool, slack_min: int, outage: bool
) -> _Route:
    """Ask the planner model for one round's route, and how it went. Scripted: no model is called.

    :param model: The planner model asked.
    :ptype model: str
    :param stops: The round's stops.
    :ptype stops: int
    :param road_closures: The roads closed in the depot's world.
    :ptype road_closures: int
    :param exhaustive: Whether the solver searches every ordering, which routes around every closure.
    :ptype exhaustive: bool
    :param slack_min: Minutes of slack on each stop's window: every five keep one more stop on time.
    :ptype slack_min: int
    :param outage: Whether the endpoint is in an outage.
    :ptype outage: bool
    :return: The route's late stops, detour and arrival-time miss.
    :rtype: _Route
    :raises PlannerOverloaded: The endpoint is in an outage.
    """
    await asyncio.sleep(0)  # a real planner awaits its model; this yields like one
    if outage:
        raise PlannerOverloaded(f"{model}: 529 overloaded")
    late = 0 if exhaustive else min(stops, road_closures)
    return _Route(
        late=max(0, late - slack_min // 5),
        detour_km=road_closures * (0.5 if exhaustive else 1.5),
        eta_miss_min=road_closures * (2.0 if exhaustive else 6.0),
    )


def describe_courier_failure(exc: BaseException) -> ProviderFailure:
    """The courier's failure describer: whose failure a raised call was, which only the courier's own types can say.

    The planner endpoint's overload is the provider's: the cell is excluded and the run goes on. Anything else is
    described by its class alone and charged to the candidate, as :func:`withhold_failure_detail` does.

    :param exc: What the call raised.
    :ptype exc: BaseException
    :return: Its description and owner.
    :rtype: ProviderFailure
    """
    if isinstance(exc, PlannerOverloaded):
        return ProviderFailure(str(exc), payload_withheld=False, owner="provider")
    return withhold_failure_detail(exc)


class RoutePlannerKind:
    """A route planner, as the engine's candidate-kind seam sees one. Its model is scripted (:func:`ask_planner`)."""

    judged_artifact = JudgedArtifact.UNJUDGED
    #: Serial: the courier's world is one table it does not bind per cell, so two cells would plan over one map.
    max_concurrent_cells: int | None = 1
    #: The carriers this kind attaches; a seed reaches the world only through these.
    CARRIERS: tuple[str, ...] = ("dispatch",)

    def __init__(
        self,
        world: WorldRegistry,
        *,
        overlays: CourierOverlays | None = None,
        describe: ProviderFailureDescriber = describe_courier_failure,
    ) -> None:
        """Bind the world this kind seeds through, the knobs its launch turned, and the host's failure describer."""
        self._world = world
        self._overlays = overlays if overlays is not None else CourierOverlays()
        self._describe = describe

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: WorldSession | None,
    ) -> _Planned:
        """Seed the depot through the cell's world session, then read it back through the declared read."""
        assert world is not None, "the courier declares a world, so every cell is handed its session"
        try:
            await world.seed(world_seed, attached=self.CARRIERS)
        except SeedRefused as refused:
            raise CandidatePreparationFailed(f"apparatus: {refused}", termination="seed_failed") from refused
        closures = await self._world.call("courier.read_closures")
        return _Planned(model=variant_config.candidate_model, road_closures=closures)

    async def invoke(self, instance: _Planned, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Plan one round and report how it went, in the courier's own measures.

        A case whose variation sets ``outage`` stands for a round planned while the planner's endpoint is overloaded.
        """
        stops = int(test_case.variation_params["stops"])
        solver = self._overlays.solver
        try:
            route = await ask_planner(
                instance.model,
                stops=stops,
                road_closures=instance.road_closures,
                exhaustive=instance.model.endswith("-pro") if solver == "auto" else solver == "exhaustive",
                slack_min=self._overlays.zone_slack_min.get(DEPOT_ZONE, 0),
                outage=test_case.variation_params.get("outage") == "true",
            )
        except PlannerOverloaded as overloaded:
            # Whose failure the call was is the host's describer's to say; describe_and_log_failure asks it, keeps the
            # failure described if the describer itself raises, and logs the traceback only when nothing is withheld.
            # What follows is the engine's rule, which failed_call_output holds: a provider's decline excludes the cell
            # and the run goes on, an account's refusal stops the run, and anything else is charged to the planner.
            failure = describe_and_log_failure(
                self._describe,
                overloaded,
                logger=_log,
                where="Courier planner call",
                message="planner: the call for case %s raised",
                args=(test_case.id,),
                level=logging.WARNING,
            )
            return failed_call_output(
                failure.owner,
                failure.description,
                refused="planner: the call was refused for the depot's account",
                declined="planner: the endpoint declined the call",
                failed="planner: the call failed",
                log=lambda what, level: _log.log(level, "%s, case %s", what, test_case.id),
            )
        return CandidateOutput(
            output=[{"stops": stops, "late": route.late}],
            host_measures={
                ON_TIME_RATE: (stops - route.late) / stops,
                DETOUR_KM: route.detour_km,
                FIELD_ACCURACY: route.eta_miss_min,
            },
            telemetry=CandidateTelemetry(
                usage=[
                    RoleUsage(
                        role="candidate",
                        model=instance.model,
                        cost_usd=0.002,
                        price_source="courier_script",
                        call_count=1,
                    )
                ],
            ),
        )


# --- 6. host, and one drive ------------------------------------------------------------------------


def courier_host() -> EvalHost:
    """The courier as the engine receives it."""
    return EvalHost(
        profile=courier_profile(),
        storage=EvalStorage(InMemoryDocumentStore()),
        failure_describer=describe_courier_failure,
        trace_sink=None,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
    )


COURIER_MODELS: tuple[str, ...] = ("planner-lite", "planner-pro")

#: Which traffic feed each arm's travel times came from.
FEEDS: dict[str, str] = {"planner-lite": "feed-v2", "planner-pro": "feed-v3"}

#: The round-planning template's id: fixed, so a command line can name it.
COURIER_TEMPLATE_ID = "plan-delivery-round"

#: The one planner configuration the courier deploys, and so the one subject a launch may name.
COURIER_SUBJECT = SubjectSnapshot(
    subject_id="depot-north-planner",
    subject_label="North depot planner",
    state={},
    captured_at="2026-08-01T00:00:00+00:00",
)


def courier_template() -> EvalTemplate:
    """The courier's one scenario: plan a round around two closed roads."""
    return EvalTemplate(
        id=COURIER_TEMPLATE_ID,
        scope_id=COURIER_SCOPE,
        name="Plan a delivery round",
        intent="Order a round's stops so every one is reached inside its window, around closed roads.",
        candidate_kind=COURIER_KIND,
        world_seed=WorldSeed(namespaces={"dispatch": {"road_closures": 2}}),
    )


def courier_cases() -> list[EvalTestCase]:
    """Three rounds of that scenario, of four, six and eight stops."""
    return [
        EvalTestCase(
            # Named by what the case IS, so a rebuild is the same three cases and a run naming them
            # derives the same identity every time.
            id=f"{COURIER_TEMPLATE_ID}:stops-{stops}",
            scope_id=COURIER_SCOPE,
            template_id=COURIER_TEMPLATE_ID,
            variation_params={"stops": str(stops)},
        )
        for stops in (4, 6, 8)
    ]


def courier_run(model: str, *, cases: Sequence[EvalTestCase], world: WorldRegistry) -> EvalRun:
    """One arm of the bake-off: one planner model over every case, before it executes.

    :param model: The planner model the arm binds.
    :ptype model: str
    :param cases: The cases the arm runs.
    :ptype cases: Sequence[EvalTestCase]
    :param world: The world the placements are computed against — the one the kind seeds through.
    :ptype world: WorldRegistry
    :return: The run, pending.
    :rtype: EvalRun
    """
    return EvalRun(
        scope_id=COURIER_SCOPE,
        template_id=COURIER_TEMPLATE_ID,
        apparatus_provenance="commissioned",
        candidate_kind=COURIER_KIND,
        subject_snapshot=COURIER_SUBJECT,
        candidate_model=model,
        k_runs=2,
        test_case_ids=[case.id for case in cases],
        # The pro arm ran on the newer traffic feed: a confound the bundle has to name, which is
        # also what puts the courier's apparatus vocabulary in front of a reader.
        host_payload={_PAYLOAD: {"search_depth": 3, "traffic_feed": FEEDS[model]}},
        # What a launch freezes onto its run: the knobs the planner ran at, defaults included.
        overlays=CourierOverlays().model_dump(mode="json"),
        world_placements=world.place(seeded=("road_closures",), carriers=RoutePlannerKind.CARRIERS),
        # What the launch stamps: each of the host's measures' definitions.
        measure_definitions=MeasureRegistry(COURIER_MEASURES).definition_digests(),
        rubric_scales={},
    )


async def run_courier_campaign(host: EvalHost) -> EvalCampaign:
    """Run one round set per planner model and file them as a campaign, all in the host's storage.

    :param host: The courier host.
    :ptype host: EvalHost
    :return: The stored campaign naming both runs.
    :rtype: EvalCampaign
    """
    world = host.profile.world
    assert world is not None
    kind = RoutePlannerKind(world, describe=host.failure_describer)
    template = courier_template()
    cases = courier_cases()
    await run_blocking(host.blocking_executor, host.storage.save_template, template)
    subject = COURIER_SUBJECT
    runs = []
    for model in COURIER_MODELS:
        run = courier_run(model, cases=cases, world=world)
        await run_blocking(host.blocking_executor, host.storage.save_eval_run, run)
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={COURIER_KIND: lambda _cell: kind}),
        )
        runs.append(run.model_copy(update={"status": "completed"}))
        await run_blocking(host.blocking_executor, host.storage.save_eval_run, runs[-1])
    campaign = EvalCampaign.model_validate(
        {
            "scope_id": COURIER_SCOPE,
            "name": "planner model bake-off",
            "subject_id": subject.subject_id,
            "subject_kind": "route_planner_config",
            "behavior": "plan_delivery_round",
            "aspects": [
                {"label": "plan_delivery_round", "template_id": template.id, "run_ids": [run.id for run in runs]}
            ],
            "created_by": "test:fixture",
        }
    )
    await run_blocking(host.blocking_executor, host.storage.save_campaign, campaign)
    return campaign


# --- 7. launching, as a product serving launches does --------------------------------------------

#: The courier's launch settings: generous, and its spend is scripted, so no ceiling is enforced.
COURIER_LAUNCH_SETTINGS = LaunchSettings(
    max_launch_arms=len(COURIER_MODELS),
    max_admitted_runs=len(COURIER_MODELS),
    judge_concurrency=1,
    enforcement_enabled=False,
    max_cost_usd=1.0,
    max_metered_calls=1,
    max_out_of_run_cost_usd=1.0,
)


def _admit(_checked: object) -> None:
    """A template check the courier host has nothing to say about: it admits."""


def courier_launch_host() -> LaunchHost:
    """The courier as a launching host, with its scenario already authored in its store.

    A zero-argument factory, which is the shape the engine's command line names a host by. Its launcher
    captures the one subject the courier deploys, freezes the scenario's three rounds, records
    each arm's search depth and traffic feed on the run, and runs the planner at the knobs the launch turned.

    :return: The launching host.
    :rtype: LaunchHost
    """
    eval_host = courier_host()
    world = eval_host.profile.world
    assert world is not None
    eval_host.storage.save_template(courier_template())

    async def launch(request: LaunchRequest) -> EvalRun:
        cases = courier_cases()
        for case in cases:
            await run_blocking(eval_host.blocking_executor, eval_host.storage.save_test_case, case)
        feed = FEEDS.get(request.candidate_model or "")
        if feed is None:
            raise ValidationFailedError(
                f"the courier plans with {', '.join(COURIER_MODELS)}; {request.candidate_model!r} is none of them",
                remedy=f"launch one of {', '.join(COURIER_MODELS)}",
            )
        payload = {"search_depth": 3, "traffic_feed": feed}
        # The launch validated the knobs it turned against CourierOverlays; the planner runs at them.
        kind = RoutePlannerKind(
            world, overlays=request.overlays_as(CourierOverlays), describe=eval_host.failure_describer
        )
        return await launch_run(
            launch_host,
            request,
            KindWiring(
                kind_factory=lambda _cell: kind,
                subject=COURIER_SUBJECT,
                test_cases=cases,
                payload={_PAYLOAD: payload},
            ),
        )

    def resolve_subject(subject_id: str, _scope_id: str) -> UnknownSubject | None:
        if subject_id == COURIER_SUBJECT.subject_id:
            return None
        return UnknownSubject(remedy=f"the courier host holds one subject, {COURIER_SUBJECT.subject_id}")

    launch_host = LaunchHost(
        # The launch registry lives on the host's template checks, stated once; the courier host checks nothing else.
        eval_host=replace(
            eval_host,
            template_checks=TemplateChecks(
                require_known_tools_allowed=_admit,
                refuse_undeclared_world_seed=_admit,
                refuse_undeliverable_template=_admit,
                refuse_inadmissible_template_case=_admit,
                kinds={
                    COURIER_KIND: LaunchableKind(
                        launch=launch,
                        resolve_subject=resolve_subject,
                        unhonoured_launch_arguments=frozenset(
                            {"simulator_model", "judge_model", "judge_config_ids", "cassette_mode", "n_variations"}
                        ),
                    )
                },
            ),
        ),
        settings=lambda: COURIER_LAUNCH_SETTINGS,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda _run: world.place(seeded=("road_closures",), carriers=RoutePlannerKind.CARRIERS),
    )
    return launch_host


# --- 8. its own action, served beside the engine's ----------------------------------------------

#: The stops of each round the depot dispatched last season, oldest first. A product reads its dispatch log here.
DEPOT_ROUND_HISTORY: tuple[int, ...] = (4, 6, 11, 6, 8, 14, 11, 6, 11, 4)


class CasesProposeParams(BaseModel):
    """What ``courier_cases_propose`` takes."""

    max_rounds: int = Field(3, ge=1, le=20, description="the most round sizes to propose a case for")


class ProposedCases(BaseModel):
    """The cases ``courier_cases_propose`` proposes: one per round size the depot dispatched and no case covers, the
    most dispatched first. Proposed, not stored: whoever asked decides which to add."""

    cases: list[EvalTestCase]


async def _cases_propose(host: OpsHost, caller: Caller, params: CasesProposeParams) -> JobsStarted:
    """Read the depot's dispatch history for round sizes no case covers, as a job the caller polls."""

    async def propose() -> ProposedCases:
        await asyncio.sleep(0)  # a season's dispatch log is read in pages; this yields like one
        covered = {int(case.variation_params["stops"]) for case in courier_cases()}
        sizes = [stops for stops, _ in Counter(DEPOT_ROUND_HISTORY).most_common() if stops not in covered]
        return ProposedCases(
            cases=[
                EvalTestCase(
                    id=f"{COURIER_TEMPLATE_ID}:stops-{stops}",
                    scope_id=caller.scope_id,
                    template_id=COURIER_TEMPLATE_ID,
                    variation_params={"stops": str(stops)},
                )
                for stops in sizes[: params.max_rounds]
            ]
        )

    # Reading a season's log outlives any transport call, and it is not one paid model call (start_paid_call's case):
    # start_action_job runs it as a job whose record holds the typed result, which job_poll returns once it is done.
    return await start_action_job(
        host, caller.scope_id, action="courier_cases_propose", run=propose, budget_s=60.0, label="round sizes"
    )


#: The courier's one action of its own: long work, so it returns the job it started and ``job_poll`` reads it back.
COURIER_CASES_PROPOSE = Action(
    name="courier_cases_propose",
    summary="Propose a case for each round size the depot dispatched that no case covers yet.",
    workflow="Courier",
    permission="read",
    params=CasesProposeParams,
    result=JobsStarted,
    handler=_cases_propose,
    long_running=True,
    example={"max_rounds": 3},
)


def courier_ops_host() -> OpsHost:
    """The courier as the actions see it: its launching host, whose job manager runs every job.

    :return: The host every action is called with.
    :rtype: OpsHost
    """
    return OpsHost(launch=courier_launch_host())


def courier_catalogue() -> ActionCatalogue:
    """The engine's actions with the courier's own added: what the courier mounts and serves.

    :return: The catalogue.
    :rtype: ActionCatalogue
    """
    return eval_catalogue(host_actions=(COURIER_CASES_PROPOSE,))


__all__ = [
    "COURIER_CASES_PROPOSE",
    "COURIER_ID",
    "COURIER_KIND",
    "COURIER_MEASURES",
    "COURIER_MODELS",
    "COURIER_SCOPE",
    "DETOUR_KM",
    "FIELD_ACCURACY",
    "ON_TIME_RATE",
    "RoutePlannerKind",
    "COURIER_SUBJECT",
    "COURIER_TEMPLATE_ID",
    "CourierOverlays",
    "DEPOT_ROUND_HISTORY",
    "DEPOT_ZONE",
    "PlannerOverloaded",
    "ask_planner",
    "courier_cases",
    "courier_catalogue",
    "courier_host",
    "courier_launch_host",
    "courier_ops_host",
    "courier_run",
    "describe_courier_failure",
    "courier_template",
    "courier_profile",
    "courier_world",
    "run_courier_campaign",
]
