"""The toy host's launch path — a :class:`~threetears.evals.run.LaunchHost` and the launcher for its one kind.

``run.py`` drives :func:`~threetears.evals.run.execute_run` directly, the way a host with no job
manager would. This is the other way in, and the one a product serving launches takes:
:func:`~threetears.evals.run.start_run` loads the template, refuses what it cannot run, admits the
runs against the host's settings, and hands each arm to the launcher registered for the template's
kind. The launcher builds what only that kind has — here, the subject it captures, the extractor and
its case set — and hands the shared tail (:func:`~threetears.evals.run.launch_run`) a typed
:class:`~threetears.evals.run.KindWiring`; everything the request already says, the tail stamps.

The host the launcher closes over is the same value the launch was handed, so the run it assembles
is stamped through the host's own profile: its world placements, its context identity and its
variant levers. What a launch may turn on the extractor is not here at all: it is the kind's overlay
model on the profile (``contract.py``), which ``start_run`` validates before this launcher is called
and :func:`~threetears.evals.run.launch_run` stamps onto the run from the request. The job manager is
the launch host's own, built over the host's storage, so a run's status and its results share a store.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.evals.contracts import EvalRun, EvalStorage, NotFoundError, WorldSeed
from threetears.evals.contracts.host import HostProfile, TraceSink, WorldPlacement
from threetears.evals.run import (
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    default_job_timeout,
    launch_run,
)
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_COST_CEILING_USD,
    TOYHOST_GRADER_VERSION,
    TOYHOST_SUBJECT,
)
from packages.evals.tests.fixtures.toyhost.contract import ExtractorSpec
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.run import RUN_CHUNK_TOKENS, RUN_RETRIEVER_TOP_K, toyhost_test_cases

#: The toy host's launch settings. Generous, so a launch is refused only where a test says so.
TOYHOST_LAUNCH_SETTINGS = LaunchSettings(
    max_launch_arms=4,
    max_admitted_runs=8,
    judge_concurrency=1,
    enforcement_enabled=True,
    max_cost_usd=TOYHOST_COST_CEILING_USD,
    max_metered_calls=100,
    # What an operator of this product turns, so a refusal names the knob rather than the field.
    setting_names={"max_launch_arms": "toyhost.launch.arms", "max_admitted_runs": "toyhost.launch.admitted"},
)

#: The subjects this host can capture, by the id a launch names. One: the toy host deploys one
#: extractor configuration.
TOYHOST_SUBJECTS = {TOYHOST_SUBJECT.subject_id: TOYHOST_SUBJECT}


def toyhost_launch_host(
    *,
    profile: HostProfile | None = None,
    storage: EvalStorage | None = None,
    trace_sink: TraceSink | None = None,
    settings: Callable[[], LaunchSettings] = lambda: TOYHOST_LAUNCH_SETTINGS,
) -> tuple[LaunchHost, ScriptedExtractionClient]:
    """The toy host as a launching host: its :class:`~threetears.evals.contracts.host.EvalHost`, plus its launch registry.

    Args:
        profile: The vocabulary; ``None`` is the toy host's own.
        storage: Where documents live; ``None`` is a fresh in-memory store.
        trace_sink: The host's tracing, or ``None``.
        settings: Reads the launch settings; a callable because a host's settings hot-reload.

    Returns:
        The host, and the scripted client its extractor calls — so a caller can see what was asked.
    """
    eval_host = toyhost_host(profile=profile, storage=storage, trace_sink=trace_sink)
    world = eval_host.profile.world
    assert world is not None, "the toy host declares a world"
    client = ScriptedExtractionClient()
    kind = ToyExtractorKind(client=client, world=world)

    def place(run: EvalRun) -> dict[str, WorldPlacement]:
        # The algebra over what this run's kind can actually do: the dimensions its seed sets,
        # through the carriers the extractor attaches.
        seed = WorldSeed(namespaces=run.resolved_world_seed or {})
        return world.place(seeded=kind.seedable_dimensions(seed), carriers=ToyExtractorKind.CARRIERS)

    async def launch(request: LaunchRequest) -> EvalRun:
        # The subject the launch names, as this host captures it — never a constant, or every launch
        # would measure one subject whatever it asked for.
        subject = TOYHOST_SUBJECTS.get(request.subject_id)
        if subject is None:
            raise NotFoundError("subject", request.subject_id)
        cases = toyhost_test_cases(request.template)
        for case in cases:
            eval_host.storage.save_test_case(case)
        # What the template states for the kind, as the dispatch validated it: the fields this run
        # grades. The run records the same validated spec, so what it graded and what it says it
        # graded are one value.
        spec = request.kind_spec_as(ExtractorSpec)
        graded = ToyExtractorKind(
            client=client,
            world=world,
            graded_fields=tuple(spec.graded_fields),
            goal_checks=tuple(request.template.goal_state_checks),
        )
        # Everything the request already says — scope, model, repeats, overlays, the template's seed —
        # the engine stamps. What is here is what only this kind resolves. Nothing is model-scored on
        # this path, so there is no judge, and the extractor has no default model to fall back on.
        return await launch_run(
            launch_host,
            request,
            KindWiring(
                kind_factory=lambda _cell: graded,
                subject=subject,
                test_cases=cases,
                payload={
                    "toyhost": {
                        "grader_version": TOYHOST_GRADER_VERSION,
                        "chunk_tokens": RUN_CHUNK_TOKENS,
                        "retriever_top_k": RUN_RETRIEVER_TOP_K,
                        "extraction_schema": "v1",
                        "ocr_engine_version": "tess-5.3.1",
                        "reviewer_pool": "pool-a",
                    }
                },
            ),
        )

    launch_host = LaunchHost(
        eval_host=eval_host,
        # The extractor honours no simulated user, no judge and no cassette, so a launch naming one
        # is refused at the dispatch rather than handed to this launcher to ignore.
        kinds={
            TOY_EXTRACTOR_KIND: LaunchableKind(
                launch=launch,
                unhonoured_launch_arguments=frozenset(
                    {"simulator_model", "judge_model", "judge_config_ids", "cassette_mode", "n_variations"}
                ),
            )
        },
        settings=settings,
        # The toy host has no timeout layer of its own, so a run's job is bounded by the engine's.
        job_timeout_factory=default_job_timeout,
        world_placements=place,
    )
    return launch_host, client


__all__ = ["TOYHOST_LAUNCH_SETTINGS", "TOYHOST_SUBJECTS", "toyhost_launch_host"]
