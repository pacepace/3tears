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
6. **Its host** — the profile, a store, and the three services with no safe default; and, since it shows blind reads
   to its dispatchers, who it vouches for relaying a dispatcher's word (``courier_attests``) and its page for a
   sheet (``courier_sheet_page``).

Everything else — the trial loop, the stored documents, the bundle — is the engine's.

**It is driven two ways.** :func:`run_courier_campaign` drives the runner directly
(:func:`~threetears.evals.run.execute_run`), the low-level path: one assembled run's matrix, with the
run built and its status set by the caller. :func:`courier_launch_host` is the way a product serving
launches adopts the engine: a :class:`~threetears.evals.run.LaunchHost` whose one kind's launcher
:func:`~threetears.evals.run.start_run` dispatches to, which the engine's command line drives
(from a checkout's root, ``python -m threetears.evals run --host
packages.evals.tests.fixtures.courierhost:courier_launch_host``; with this file copied out as ``courierhost.py``,
``--host courierhost:courier_launch_host`` from beside it) and waits on through the job manager's ``wait_for``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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
    FIND_TEMPLATE,
    NotFoundError,
    ValidationFailedError,
    MetricDescriptor,
    WorldSession,
    VariantConfig,
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
    SeedRefused,
    Sweepable,
    TemplateChecks,
    WorldDimension,
    WorldRegistry,
    default_cell_timeout,
)
from threetears.evals.schema import IntervalScale, SubjectSnapshot, SweepableValue
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
    update_template,
)
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


#: The seats the planner fills: the traffic feed it plans against, and its spend ceiling. Nothing here is
#: graded by a model or talks to a simulated user, so the judge and simulator seats are not filled. Its
#: store also receives an eval's runs, which read the same feed and run uncapped.
COURIER_CONTRACT = KindContract(COURIER_KIND, seats=frozenset({"traffic_feed", "max_cost_usd"}))

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


class RoutePlannerKind:
    """A route planner, as the engine's candidate-kind seam sees one. Scripted: no model is called."""

    judged_artifact = JudgedArtifact.UNJUDGED
    #: Serial: the courier's world is one table it does not bind per cell, so two cells would plan over one map.
    max_concurrent_cells: int | None = 1
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


#: The one caller the courier vouches for relaying a dispatcher's own word: its dispatch desk's page.
COURIER_DISPATCH_DESK = "courier:dispatch-desk"


def courier_attests(*, scope_id: str, caller: str, person: str) -> str | None:
    """Whether the courier vouches that ``caller`` relays ``person``'s blind-read answer or campaign call.

    Only its dispatch desk does, only in its depot's scope, and the courier names a dispatcher by their lower-cased
    login, whatever spelling the desk sent: the name the engine records the answer under (``attests_person``).
    """
    if scope_id != COURIER_SCOPE or caller != COURIER_DISPATCH_DESK or not person.strip():
        return None
    return person.strip().lower()


def courier_sheet_page(sheet: Mapping[str, Any]) -> str:
    """The dispatch desk's page for a blind read: the sheet as text, from the item a dispatcher reads next.

    What a host writes when it shows a blind read on its own page. It reads the ``blind_read_sheet`` action's result as
    the page receives it (its JSON), shows nothing but what that carries, and never shows again an item a dispatcher
    already answered; the items keep the drawn order and each pair its drawn slots.
    """
    answered = set(sheet.get("answered") or [])
    lines = [str(sheet["name"]), str(sheet["instructions"]), ""]
    for criterion in sheet["criteria"]:
        answers = " / ".join(str(level["value"]) for level in criterion["levels"])
        lines.append(f"{criterion['dim']}: {criterion['question']} ({answers})")
    for item in sheet["items"]:
        if item["item"] in answered:
            continue
        lines += ["", f"[{item['item']}] {item['case_material']}"]
        lines += [f"  {output['slot']}: {output['text']}" for output in item["outputs"]]
        if item["ask"].get("preference"):
            lines.append("  Which is better: A, B, or tie?")
    return "\n".join(lines)


def courier_host() -> EvalHost:
    """The courier as the engine receives it: it vouches for its dispatch desk relaying a dispatcher's word."""
    return EvalHost(
        profile=courier_profile(),
        storage=EvalStorage(InMemoryDocumentStore()),
        failure_describer=withhold_failure_detail,
        trace_sink=None,
        blocking_executor=None,
        cell_timeout=default_cell_timeout,
        attests_person=courier_attests,
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
    kind = RoutePlannerKind(world)
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
    captures the one subject the courier deploys, freezes the scenario's three rounds, and records
    each arm's search depth and traffic feed on the run.

    :return: The launching host.
    :rtype: LaunchHost
    """
    eval_host = courier_host()
    world = eval_host.profile.world
    assert world is not None
    kind = RoutePlannerKind(world)
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


# --- 8. an operator's own edits to a stored scenario ------------------------------------------------


def rename_stratum(host: EvalHost, template_id: str, stratum: str, name: str) -> EvalTemplate:
    """Give a stratum of a stored scenario the operator's own name, over the one it already has.

    The seed and a definitions reset only fill a name a template lacks, so ``update_template`` is the one door to a
    rename. A name is presentation only: the template keeps its id, its revision and its ``updated_at``, and every
    run of it reads on, named anew.

    :param host: The courier host, with its template checks (``courier_launch_host().eval_host``).
    :ptype host: EvalHost
    :param template_id: The stored scenario.
    :ptype template_id: str
    :param stratum: The stratum id its cases declare.
    :ptype stratum: str
    :param name: What a page prints for it from now on.
    :ptype name: str
    :return: The scenario as stored after the rename.
    :rtype: EvalTemplate
    :raises NotFoundError: No such scenario in the courier's scope.
    """
    stored = host.storage.load_template(template_id, COURIER_SCOPE)
    if stored is None:
        raise NotFoundError("template", template_id, remedy=FIND_TEMPLATE)
    # update_template is the only door to renaming a name a stored template already holds.
    return update_template(host, template_id, COURIER_SCOPE, {"stratum_names": {**stored.stratum_names, stratum: name}})


def retire_scenario(host: EvalHost, template_id: str) -> EvalTemplate:
    """Retire a stored scenario: it leaves the active template listing, and every run of it still reads.

    :param host: The courier host, with its template checks (``courier_launch_host().eval_host``).
    :ptype host: EvalHost
    :param template_id: The stored scenario.
    :ptype template_id: str
    :return: The scenario as stored, archived.
    :rtype: EvalTemplate
    :raises NotFoundError: No such scenario in the courier's scope.
    """
    # update_template is the only door to archiving a template outright; a reset archives only what it supersedes.
    # It stamps the template's updated_at, so today no earlier run of it can be re-judged, its run summary drops the
    # template's intent and its judges' criteria read as unknown; rename_stratum leaves updated_at alone.
    return update_template(host, template_id, COURIER_SCOPE, {"archived": True})


__all__ = [
    "COURIER_DISPATCH_DESK",
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
    "courier_attests",
    "courier_cases",
    "courier_host",
    "courier_launch_host",
    "courier_run",
    "courier_sheet_page",
    "courier_template",
    "courier_profile",
    "courier_world",
    "rename_stratum",
    "retire_scenario",
    "run_courier_campaign",
]
