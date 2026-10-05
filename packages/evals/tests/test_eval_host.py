"""The host an app hands the engine: what it refuses at construction, and a launch driven through it.

``EvalHost`` replaced the process-global profile and ``LaunchContext`` with one explicit value. These
pin the refusals that value makes, and drive the toy host's launch from ``start_run`` to stored
results, so the fold of the launch wiring into the host is proven by a run rather than by reading.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import fields, replace

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import EvalRun, EvalStorage, EvalTestCase, NotFoundError, ValidationFailedError
from threetears.evals.contracts.errors import AdmissionRefusedError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.models import VariationCounts
from threetears.evals.run import (
    ArmPlan,
    ArmPrice,
    EvalJobManager,
    KindWiring,
    LaunchableKind,
    LaunchArgument,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    default_job_timeout,
    launch_run,
    start_run,
    start_universal_battery,
)
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT, toyhost_observation
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases
from threetears.evals.storage import InMemoryDocumentStore


def _launching(**settings: object) -> tuple[LaunchHost, EvalStorage]:
    """The toy launch host over a fresh store, its template saved, its settings overridden as given."""
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    chosen = TOYHOST_LAUNCH_SETTINGS.model_copy(update=settings)
    host, _client = toyhost_launch_host(storage=storage, settings=lambda: chosen)
    return host, storage


async def _settled(host: LaunchHost, run_ids: list[str]) -> None:
    """Wait for every launched run's job to end, failing after a few seconds."""
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run_id) for run_id in run_ids):
            await asyncio.sleep(0.01)


# --- a launch, end to end --------------------------------------------------------------------------


async def test_a_launch_runs_through_the_host_it_was_handed():
    """``start_run`` reads the template, dispatches the kind and stores the cells — all through the host.

    One run per model, each carrying the matrix the toy kind produces, stamped with the host's world
    placements and identity: the three things ``LaunchContext`` used to carry beside the profile,
    now read off the one value.
    """
    host, storage = _launching()
    template = toyhost_template()

    runs = await start_run(
        host,
        template_id=template.id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=list(RUN_MODELS),
        k_runs=2,
    )
    await _settled(host, [run.id for run in runs])

    assert [run.candidate_model for run in runs] == list(RUN_MODELS)
    for run in runs:
        stored = storage.load_eval_run(run.id, run.scope_id)
        assert stored is not None and stored.status == "completed", stored
        assert len(storage.query_eval_results_by_run(run.id, run.scope_id)) == len(run.test_case_ids) * 2
        assert run.world_placements, "the host's placement must be stamped before the identity hashes it"
        assert run.context_key is not None
        assert run.variant_levers is not None, "the toy profile wires a lever reader"


async def test_an_oversized_launch_is_refused_naming_the_hosts_setting():
    """The refusal names the knob an operator turns, in the host's own vocabulary."""
    host, _ = _launching(max_launch_arms=1)

    with pytest.raises(ValidationFailedError, match=r"toyhost\.launch\.arms"):
        await start_run(
            host,
            template_id=toyhost_template().id,
            scope_id=TOYHOST_SCOPE,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=list(RUN_MODELS),
        )


async def test_admission_is_refused_against_the_hosts_ceiling_naming_its_setting():
    """Admission reads the host's ceiling from the same settings, and names the host's knob."""
    host, _ = _launching(max_admitted_runs=1)

    with pytest.raises(AdmissionRefusedError) as refused:
        await start_run(
            host,
            template_id=toyhost_template().id,
            scope_id=TOYHOST_SCOPE,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=list(RUN_MODELS),
        )

    assert "toyhost.launch.admitted" in refused.value.message


async def test_a_kind_the_host_registers_no_launcher_for_is_refused():
    """The launch registry is the host's: a host with no entry for the template's kind launches nothing."""
    host, _ = _launching()
    bare = replace(host, kinds={})

    with pytest.raises(ValidationFailedError, match="has no launcher"):
        await start_run(
            bare,
            template_id=toyhost_template().id,
            scope_id=TOYHOST_SCOPE,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=[RUN_MODELS[0]],
        )


async def test_the_battery_reads_its_templates_in_the_scope_it_is_named():
    """The battery lists the universal templates of ONE scope and launches each there.

    Both reads it makes — the listing and each launch's template load — name the scope; a scope-free
    listing is the read the port no longer has.
    """
    host, storage = _launching()
    storage.save_template(toyhost_template().model_copy(update={"universal": True}))

    async def preflight(_subject_id, _models):
        async def check(_template, _cassette_mode):
            return None

        return check

    run_ids = await start_universal_battery(
        host, TOYHOST_SUBJECT.subject_id, scope_id=TOYHOST_SCOPE, models=[RUN_MODELS[0]], preflight=preflight
    )
    await _settled(host, run_ids)

    assert len(run_ids) == 1
    assert (
        await start_universal_battery(
            host, TOYHOST_SUBJECT.subject_id, scope_id="another-scope", models=[RUN_MODELS[0]], preflight=preflight
        )
        == []
    ), "another scope holds no universal template, so its battery is empty"


# --- the host's construction refusals --------------------------------------------------------------


def test_a_host_declaring_a_world_must_say_how_a_run_is_placed_in_it():
    host, _ = _launching()

    with pytest.raises(ValueError, match="declares a world"):
        replace(host, world_placements=None)


def test_a_host_declaring_no_world_has_nothing_to_place_a_run_in():
    host, _ = _launching()
    worldless = replace(toyhost_profile(), world=None)

    with pytest.raises(ValueError, match="declares no world"):
        replace(host, eval_host=replace(host.eval_host, profile=worldless))

    placed = replace(host, eval_host=replace(host.eval_host, profile=worldless), world_placements=None)
    assert placed.place(toyhost_observation(chunk_tokens=256)) == {}, "a host with no world records it placed nothing"


def test_settings_name_only_the_settings_a_launch_has():
    with pytest.raises(ValidationError, match="max_lanuch_arms"):
        LaunchSettings.model_validate(
            {**TOYHOST_LAUNCH_SETTINGS.model_dump(), "setting_names": {"max_lanuch_arms": "a typo is never read"}}
        )

    assert TOYHOST_LAUNCH_SETTINGS.name_of("max_launch_arms") == "toyhost.launch.arms"
    assert TOYHOST_LAUNCH_SETTINGS.name_of("judge_concurrency") == "judge_concurrency"


@pytest.mark.parametrize("field", ["max_launch_arms", "max_admitted_runs", "judge_concurrency", "max_metered_calls"])
def test_a_launch_ceiling_of_zero_is_refused(field: str):
    with pytest.raises(ValidationError, match=field):
        LaunchSettings.model_validate({**TOYHOST_LAUNCH_SETTINGS.model_dump(), field: 0})


def test_a_host_with_no_clients_refuses_the_work_that_needs_one_by_name():
    host: EvalHost = toyhost_host()

    with pytest.raises(ValueError, match="a re-judge calls a model"):
        host.completion_clients("a re-judge")


def test_an_admission_refusal_reads_as_its_message_wherever_it_is_printed():
    """A log line or traceback prints ``str(error)``; a refusal built from keywords used to print nothing."""
    from threetears.evals.contracts.errors import NotFoundError

    refused = AdmissionRefusedError(requested=3, admitted=7, limit=8, limit_name="toyhost.launch.admitted")

    assert str(refused) == refused.message and "toyhost.launch.admitted" in str(refused)
    assert str(NotFoundError("template", "t-1")) == "template 't-1' not found"


# --- the launch host composes the host, and the wiring is typed -----------------------------------


def test_a_launch_host_holds_its_eval_host_whole_and_builds_its_jobs_over_that_hosts_store():
    """Composed, not copied: no field of the host can be dropped on the way, and there is one store.

    A launch host that restated the host's fields field by field is how the reference launcher came
    to drop ``clients``; and a job manager handed a store of its own could write a run's status
    somewhere its results are not. Neither shape is expressible now.
    """
    eval_host = replace(toyhost_host(), clients=lambda role, model, *, temperature=None: None)  # type: ignore[arg-type,return-value]
    launch_host = LaunchHost(
        eval_host=eval_host,
        kinds={},
        settings=lambda: TOYHOST_LAUNCH_SETTINGS,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda _run: {},
    )

    assert launch_host.eval_host is eval_host and launch_host.eval_host.clients is eval_host.clients
    restated = {field.name for field in fields(LaunchHost)} & {field.name for field in fields(EvalHost)}
    assert restated == set(), f"a launch host restating {restated} can drop or disagree with the host's own"
    with pytest.raises(TypeError, match="job_manager"):
        LaunchHost(  # type: ignore[call-arg]
            eval_host=eval_host,
            kinds={},
            settings=lambda: TOYHOST_LAUNCH_SETTINGS,
            job_timeout_factory=default_job_timeout,
            world_placements=lambda _run: {},
            job_manager=EvalJobManager(EvalStorage(InMemoryDocumentStore())),
        )


def _wired(request: LaunchRequest, **overrides: object) -> KindWiring:
    """What a well-behaved toy launcher resolves for ``request``, with ``overrides`` applied."""
    world = toyhost_profile().world
    assert world is not None
    kind = ToyExtractorKind(client=ScriptedExtractionClient(), world=world)
    fields_: dict[str, object] = {
        "kind_factory": lambda _cell: kind,
        "subject": TOYHOST_SUBJECT,
        "test_cases": toyhost_test_cases(request.template),
        **overrides,
    }
    return KindWiring(**fields_)  # type: ignore[arg-type]


def _launching_with(
    wire: Callable[[LaunchRequest], KindWiring], *, unhonoured: frozenset[LaunchArgument] = frozenset()
) -> tuple[LaunchHost, EvalStorage, list[LaunchRequest]]:
    """The toy launch host with a launcher that hands the tail whatever ``wire`` resolves.

    Returns:
        The host, its store, and every request the launcher was handed — so a refusal made before
        the launcher ran is distinguishable from one the launcher made.
    """
    host, storage = _launching()
    handed: list[LaunchRequest] = []

    async def launch(request: LaunchRequest) -> EvalRun:
        handed.append(request)
        return await launch_run(launching, request, wire(request))

    def plan(request: LaunchRequest) -> ArmPlan:
        # The toy invoices, whatever a generation would have asked for: a launcher that generates nothing.
        if request.candidate_model is None:
            raise ValidationFailedError("the toy extractor has no default candidate model; name one")
        return ArmPlan(case_count=len(toyhost_test_cases(request.template)), candidate_model=request.candidate_model)

    launching = replace(
        host,
        kinds={
            TOY_EXTRACTOR_KIND: LaunchableKind(
                launch=launch,
                unhonoured_launch_arguments=unhonoured,
                plan_arm=None if "n_variations" in unhonoured else plan,
            )
        },
        # Every arm priced at nothing, so a generating launch reaches the launcher this drives.
        launch_pricer=lambda _quote: ArmPrice(predicted_usd=0.0, basis="scripted"),
    )
    return launching, storage, handed


async def _launch_one(host: LaunchHost, **arguments: object) -> list[EvalRun]:
    launched = {"template_id": toyhost_template().id, "scope_id": TOYHOST_SCOPE, "models": [RUN_MODELS[0]]}
    return await start_run(host, subject_id=TOYHOST_SUBJECT.subject_id, **{**launched, **arguments})  # type: ignore[arg-type]


async def test_the_launch_stamps_the_requests_cassette_mode_and_corpus_with_no_launcher_restating_them():
    """A launcher that says nothing about cassettes still records the replay, and the corpus, it was asked for.

    It used to build the run from a dict the launcher filled, so a launcher that forgot the mode
    recorded ``off`` and ran a requested replay live — and paid for it. The corpus a replay serves
    is the launch's to name: the id of the capture run whose recording it replays.
    """
    host, storage, _handed = _launching_with(_wired)
    (capture,) = await _launch_one(host, cassette_mode="capture")
    await _settled(host, [capture.id])

    (replay,) = await _launch_one(host, cassette_mode="replay", cassette_corpus_id=capture.id)
    await _settled(host, [replay.id])

    assert capture.cassette_mode == "capture" and capture.cassette_corpus_id is None
    assert replay.cassette_mode == "replay" and replay.cassette_corpus_id == capture.id
    assert replay.scope_id == TOYHOST_SCOPE
    stored = storage.load_eval_run(replay.id, TOYHOST_SCOPE)
    assert stored is not None and (stored.cassette_mode, stored.cassette_corpus_id) == ("replay", capture.id)


@pytest.mark.parametrize(
    ("arguments", "said"),
    [
        ({"cassette_mode": "replay"}, "names no corpus"),
        ({"cassette_mode": "capture", "cassette_corpus_id": "some-run"}, "name a corpus only with"),
        ({"cassette_mode": "replay", "cassette_corpus_id": "no-such-run"}, "names no run in scope"),
    ],
    ids=["replay-without-corpus", "corpus-without-replay", "corpus-run-missing"],
)
async def test_a_replay_corpus_the_launch_cannot_serve_is_refused_before_anything_runs(arguments, said):
    host, storage, handed = _launching_with(_wired)

    with pytest.raises(ValidationFailedError, match=said):
        await _launch_one(host, **arguments)

    assert handed == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


async def test_a_replay_corpus_must_be_a_capture_of_this_template_in_this_scope():
    """A run in another scope, a run that captured nothing, and another template's capture are all refused."""
    host, storage, handed = _launching_with(_wired)
    (capture,) = await _launch_one(host, cassette_mode="capture")
    (live,) = await _launch_one(host)
    await _settled(host, [capture.id, live.id])
    elsewhere = capture.model_copy(update={"id": "capture-elsewhere", "scope_id": "another-scope"})
    storage.save_eval_run(elsewhere)
    other_template = capture.model_copy(update={"id": "capture-of-another", "template_id": "another-template"})
    storage.save_eval_run(other_template)
    launched_so_far = len(handed)

    for corpus, said in [
        (elsewhere.id, "names no run in scope"),
        (live.id, "recorded no corpus"),
        (other_template.id, "captured template 'another-template'"),
    ]:
        with pytest.raises(ValidationFailedError, match=said):
            await _launch_one(host, cassette_mode="replay", cassette_corpus_id=corpus)

    assert len(handed) == launched_so_far, "a refused corpus reached the launcher"


async def test_a_battery_cannot_replay_one_corpus_across_its_templates():
    host, storage = _launching()
    storage.save_template(toyhost_template().model_copy(update={"universal": True}))

    async def preflight(_subject_id, _models):
        async def check(_template, _cassette_mode):
            return None

        return check

    with pytest.raises(ValidationFailedError, match="replay each template with start_run.*Nothing was launched"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            cassette_mode="replay",
            preflight=preflight,
        )

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_a_launch_argument_the_kind_cannot_honour_is_refused_before_its_launcher_runs():
    """Refused at the dispatch, so a launcher that would have ignored it never gets the chance."""
    host, storage, handed = _launching_with(_wired, unhonoured=frozenset({"cassette_mode"}))

    with pytest.raises(ValidationFailedError, match="cannot honour cassette_mode='capture'"):
        await _launch_one(host, cassette_mode="capture")

    assert handed == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


def test_a_kind_cannot_decline_an_argument_that_is_not_a_launch_argument():
    with pytest.raises(ValueError, match="cassete_mode"):
        LaunchableKind(launch=lambda _request: None, unhonoured_launch_arguments=frozenset({"cassete_mode"}))  # type: ignore[arg-type,return-value]


async def test_the_reference_launcher_captures_the_subject_the_launch_names():
    """The toy launcher used to stamp one constant subject whatever the launch asked for."""
    host, storage = _launching()

    with pytest.raises(NotFoundError, match="another-config"):
        await start_run(
            host,
            template_id=toyhost_template().id,
            scope_id=TOYHOST_SCOPE,
            subject_id="another-config",
            models=[RUN_MODELS[0]],
        )

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_an_arm_naming_no_model_is_refused_for_a_kind_with_no_default():
    """The toy kind's plan refuses it, before the arm is priced or its launcher runs."""
    host, _storage = _launching()

    with pytest.raises(ValidationFailedError, match="no default candidate model; name one"):
        await _launch_one(host, models=[])


async def test_an_unplanned_arm_naming_no_model_is_refused_at_the_tail_for_a_kind_with_no_default():
    """A kind that plans nothing reaches its launcher under a cap the launch named, and the tail refuses it there."""
    host, storage, handed = _launching_with(_wired, unhonoured=frozenset({"n_variations"}))

    with pytest.raises(ValidationFailedError, match="no default candidate model; name one"):
        await _launch_one(host, models=[], max_cost_usd=1.0)

    assert len(handed) == 1, "the launcher ran; the tail refused"
    assert storage.query_eval_runs(TOYHOST_SCOPE) == []


_OTHER_SCOPE_CASE = EvalTestCase(scope_id="another-scope", template_id=toyhost_template().id)
_OTHER_TEMPLATE_CASE = EvalTestCase(scope_id=TOYHOST_SCOPE, template_id="another-template")


@pytest.mark.parametrize(
    ("overrides", "arguments", "said"),
    [
        (
            {"subject": TOYHOST_SUBJECT.model_copy(update={"subject_id": "a-constant"})},
            {},
            "captured subject 'a-constant'",
        ),
        ({"test_cases": [_OTHER_SCOPE_CASE]}, {}, "from outside template"),
        ({"test_cases": [_OTHER_TEMPLATE_CASE]}, {}, "from outside template"),
        ({"default_candidate_model": "a-default"}, {}, "a default is for an arm that named none"),
        ({}, {"judge_model": "judge-a"}, "a pinned judge is the judge"),
        ({"simulator_model": "sim-b"}, {"simulator_model": "sim-a"}, "a pinned simulator is the simulator"),
        ({}, {"n_variations": 2}, "asked for 2 generated case"),
        ({"variation_counts": VariationCounts(requested=3, kept=3, reused=0)}, {}, "asked for 0 generated case"),
    ],
    ids=[
        "another-subject",
        "a-case-in-another-scope",
        "a-case-of-another-template",
        "a-default-beside-a-named-model",
        "a-pinned-judge-unwired",
        "another-simulator",
        "generation-unrecorded",
        "generation-unasked",
    ],
)
async def test_a_wiring_that_contradicts_its_request_is_refused_and_creates_no_run(overrides, arguments, said):
    """Where the request and the wiring speak to one fact they must agree, or the run records one thing and runs another."""
    host, storage, _handed = _launching_with(lambda request: _wired(request, **overrides))

    with pytest.raises(ValueError, match=said):
        await _launch_one(host, **arguments)

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


async def test_a_battery_refuses_a_template_its_launch_would_refuse_before_launching_any():
    """The dispatch's own refusals are pre-flighted over the whole set: all or nothing.

    A battery whose second template names a kind this host cannot launch used to launch the first
    and then be refused — the partial battery it promises never to start.
    """
    host, storage = _launching()
    first = toyhost_template().model_copy(update={"universal": True})
    unlaunchable = toyhost_template().model_copy(
        update={"id": "unlaunchable", "universal": True, "candidate_kind": "a-kind-with-no-launcher"}
    )
    storage.save_template(first)
    storage.save_template(unlaunchable)

    async def preflight(_subject_id, _models):
        async def check(_template, _cassette_mode):
            return None

        return check

    with pytest.raises(ValidationFailedError, match="no launcher.*Nothing was launched"):
        await start_universal_battery(
            host, TOYHOST_SUBJECT.subject_id, scope_id=TOYHOST_SCOPE, models=[RUN_MODELS[0]], preflight=preflight
        )

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
