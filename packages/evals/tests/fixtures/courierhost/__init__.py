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
   runs the candidate and reports its measures.
6. **Its host** — the profile, a store, and the three services with no safe default.

Everything else — the trial loop, the stored documents, the bundle — is the engine's.

**It drives the runner directly** (:func:`~threetears.evals.run.execute_run`), the low-level path: one
assembled run's matrix, with the run built and its status set by the caller. A product serving
launches adopts the engine through :func:`~threetears.evals.run.start_run`, which the toy host's
``launch.py`` wires; it starts each run as a job, and the engine has no awaitable for a launched run's
completion yet, so a script that must wait for its results — this one — drives the runner instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.contracts import (
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellCassettes,
    CellSink,
    CellSpanWindow,
    EvalCampaign,
    EvalResult,
    EvalRun,
    EvalStorage,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    MetricDescriptor,
    RoleUsage,
    WorldSeed,
    VariantConfig,
    withhold_failure_detail,
)
from threetears.evals.contracts.host import (
    SHARED_CORE,
    Coverage,
    EvalHost,
    HostProfile,
    IntervalScale,
    MeasureRegistry,
    SeedRefused,
    SubjectSnapshot,
    Sweepable,
    SweepableValue,
    WorldDimension,
    WorldRegistry,
    check_seed,
    default_cell_timeout,
)
from threetears.evals.run import RunnerOptions, execute_run
from threetears.evals.storage import InMemoryDocumentStore

COURIER_ID = "courier"
COURIER_SCOPE = "courier-depot-north"
COURIER_KIND = "route_planner"
_PAYLOAD = "courier"

# --- 1. measures -----------------------------------------------------------------------------------

ON_TIME_RATE = "on_time_rate"
DETOUR_KM = "detour_km"
#: The name the toy host also registers, for a different measure — see the module docstring.
FIELD_ACCURACY = "field_accuracy"

COURIER_MEASURES = (
    MetricDescriptor(
        name=ON_TIME_RATE,
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
    """A run's level of the courier's own lever, its search depth; the engine resolves the planner model."""
    depth = (run.host_payload or {}).get(_PAYLOAD, {}).get("search_depth")
    return {"search_depth": SweepableValue.of(depth, scale=IntervalScale(value=float(depth), unit=None))}


#: Nothing here is graded by a model or talks to a simulated user, and the courier says so.
_NO_JUDGE_OR_SIMULATOR = {
    name: Coverage("inapplicable", "routes are graded against the depot's delivery log; no model or user is involved")
    for name in (
        "judge_model",
        "judge_dim_divergence",
        "judge_request_settings",
        "simulator_model",
        "simulator_request_settings",
    )
}


def courier_profile() -> HostProfile:
    """The courier's vocabulary, with a fresh world."""
    return HostProfile(
        host_id=COURIER_ID,
        host_sweepables=COURIER_SWEEPABLES,
        measures=MeasureRegistry(COURIER_MEASURES),
        world=courier_world(),
        variant_levers=courier_levers,
        apparatus_applicability=_NO_JUDGE_OR_SIMULATOR,
    )


# --- 5. candidate kind -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Planned:
    model: str
    road_closures: int


class RoutePlannerKind:
    """A route planner, as the engine's candidate-kind seam sees one. Scripted: no model is called."""

    judged_artifact = JudgedArtifact.UNJUDGED
    #: The carriers this kind attaches; a seed reaches the world only through these.
    CARRIERS: tuple[str, ...] = ("dispatch",)

    def __init__(self, world: WorldRegistry) -> None:
        """Bind the world this kind seeds through."""
        self._world = world

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
    ) -> _Planned:
        """Seed the depot through the engine's seed walk, then read it back through the declared read."""
        try:
            writes = check_seed(self._world, world_seed.namespaces, attached=self.CARRIERS)
        except SeedRefused as refused:
            raise CandidatePreparationFailed(f"apparatus: {refused}", termination="seed_failed") from refused
        for write in writes:
            await self._world.call(write.handle, write.value)
        closures = await self._world.call("courier.read_closures")
        return _Planned(model=variant_config.candidate_model, road_closures=closures)

    async def invoke(self, instance: _Planned, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Plan one round and report how it went, in the courier's own measures."""
        await asyncio.sleep(0)  # a real planner awaits its solver; this yields like one
        stops = int(test_case.variation_params["stops"])
        careful = instance.model.endswith("-pro")
        late = 0 if careful else min(stops, instance.road_closures)
        detour = instance.road_closures * (0.5 if careful else 1.5)
        eta_miss_min = instance.road_closures * (2.0 if careful else 6.0)
        return CandidateOutput(
            output=[{"stops": stops, "late": late}],
            host_measures={ON_TIME_RATE: (stops - late) / stops, DETOUR_KM: detour, FIELD_ACCURACY: eta_miss_min},
            telemetry=CandidateTelemetry(
                usage=[
                    RoleUsage(role="candidate", model=instance.model, cost_usd=0.002, price_source="courier_script")
                ],
            ),
        )


# --- 6. host, and one drive ------------------------------------------------------------------------


def courier_host() -> EvalHost:
    """The courier as the engine receives it."""
    return EvalHost(
        profile=courier_profile(),
        storage=EvalStorage(InMemoryDocumentStore()),
        failure_describer=withhold_failure_detail,
        trace_sink=None,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
    )


COURIER_MODELS: tuple[str, ...] = ("planner-lite", "planner-pro")

#: Which traffic feed each arm's travel times came from.
FEEDS: dict[str, str] = {"planner-lite": "feed-v2", "planner-pro": "feed-v3"}


async def run_courier_campaign(host: EvalHost) -> EvalCampaign:
    """Run one round set per planner model and file them as a campaign, all in the host's storage.

    Args:
        host: The courier host.

    Returns:
        The stored campaign naming both runs.
    """
    world = host.profile.world
    assert world is not None
    kind = RoutePlannerKind(world)
    template = EvalTemplate(
        scope_id=COURIER_SCOPE,
        name="Plan a delivery round",
        intent="Order a round's stops so every one is reached inside its window, around closed roads.",
        candidate_kind=COURIER_KIND,
        world_seed=WorldSeed(namespaces={"dispatch": {"road_closures": 2}}),
    )
    cases = [
        EvalTestCase(scope_id=COURIER_SCOPE, template_id=template.id, variation_params={"stops": str(stops)})
        for stops in (4, 6, 8)
    ]
    host.storage.save_template(template)
    subject = SubjectSnapshot(subject_id="depot-north-planner", subject_label="North depot planner", state={})
    runs = []
    for model in COURIER_MODELS:
        run = EvalRun(
            scope_id=COURIER_SCOPE,
            template_id=template.id,
            candidate_kind=COURIER_KIND,
            subject_snapshot=subject,
            candidate_model=model,
            k_runs=2,
            test_case_ids=[case.id for case in cases],
            # The pro arm ran on the newer traffic feed: a confound the bundle has to name, which is
            # also what puts the courier's apparatus vocabulary in front of a reader.
            host_payload={_PAYLOAD: {"search_depth": 3, "traffic_feed": FEEDS[model]}},
            world_placements=world.place(seeded=("road_closures",), carriers=RoutePlannerKind.CARRIERS),
            rubric_scales={},
        )
        host.storage.save_eval_run(run)
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=cases,
            judge_service=None,
            options=RunnerOptions(candidate_kinds={COURIER_KIND: lambda _cell: kind}),
        )
        runs.append(run.model_copy(update={"status": "completed"}))
        host.storage.save_eval_run(runs[-1])
    campaign = EvalCampaign(
        scope_id=COURIER_SCOPE,
        name="planner model bake-off",
        subject_id=subject.subject_id,
        subject_kind="route_planner_config",
        behavior="plan_delivery_round",
        template_id=template.id,
        run_ids=[run.id for run in runs],
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    return campaign


__all__ = [
    "COURIER_ID",
    "COURIER_KIND",
    "COURIER_MEASURES",
    "COURIER_MODELS",
    "COURIER_SCOPE",
    "DETOUR_KM",
    "FIELD_ACCURACY",
    "ON_TIME_RATE",
    "RoutePlannerKind",
    "courier_host",
    "courier_profile",
    "courier_world",
    "run_courier_campaign",
]
