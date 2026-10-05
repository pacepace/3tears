"""The model that writes a launch's generated cases: named on the launch, asked for in its own role, recorded on the run.

A template's ``llm`` variation axes are written by a model, and before ``variation_model`` the only
model a launch could name for a kind with no simulated user was ``simulator_model`` — which would
record a simulated user the run never had. These pin the replacement end to end, on the toy host's
extractor driven by a launcher that generates the way an adopter's does (once per launch through
:meth:`~threetears.evals.run.LaunchGroup.resolve_once`, its counts handed to the tail):

- **The writer is asked for in the ``variation`` role, once per launch, and recorded** on every arm's
  ``variation_counts`` as the model the client named — never as the run's simulator.
- **It enters no identity**: the cases it wrote are hashed through ``test_case_ids``, so two runs over
  one case set share a context key whatever model wrote it.
- **Every refusal fires before any spend**: a generation that needs a writer and names none, a writer
  named for a launch that generates nothing or for a template with no ``llm`` axis, and a launcher
  whose generation recorded another model than the one named.
- **Every launch surface carries it** — ``start_run``, the battery, ``run_launch`` (ops and action),
  ``launch_estimate`` and the CLI's ``run``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import CostEstimate
from threetears.evals.contracts import EvalRun, EvalStorage, ValidationFailedError
from threetears.evals.contracts.host import CompletionRole
from threetears.evals.contracts.identity import derive_context_identity
from threetears.evals.contracts.models import EvalTemplate, VariationAxis, VariationCounts
from threetears.evals.gen import generate_variations
from threetears.evals.ops import LaunchArguments, launch_estimate, run_launch
from threetears.evals.run import (
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    launch_run,
    start_run,
    start_universal_battery,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import (
    DOCUMENT_PARAM,
    TOY_EXTRACTOR_KIND,
    ScriptedExtractionClient,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_DOCUMENTS, RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import CALLER, ops_fixture

#: The model the launches name to write their cases.
WRITER = "writer-a"


@dataclass(frozen=True)
class _Completion:
    content: str


# parity-with: threetears.evals.contracts.provider.BoundCompletionClient
@dataclass
class _FakeWriter:
    """A variation writer that answers with the toy host's document ids, naming the model it resolved to."""

    model_name: str
    answer: str = '{"values": ["' + '", "'.join(RUN_DOCUMENTS) + '"]}'
    calls: int = 0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        return _Completion(self.answer)

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeWriter:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


# parity-with: threetears.evals.contracts.host.eval_host.CompletionClients
@dataclass
class _FakeClients:
    """The host's client factory: records every role and model it is asked for; resolves through ``aliases``."""

    aliases: dict[str, str] = field(default_factory=dict)
    asked: list[tuple[CompletionRole, str | None]] = field(default_factory=list)

    def __call__(self, role: CompletionRole, model: str | None, *, temperature: float | None = None) -> _FakeWriter:
        self.asked.append((role, model))
        assert model is not None, "a variation writer is always named"
        return _FakeWriter(model_name=self.aliases.get(model, model))


def _template(*axes: VariationAxis, template_id: str | None = None, universal: bool = False) -> EvalTemplate:
    template = toyhost_template()
    return template.model_copy(
        update={"variation_axes": list(axes), "universal": universal, **({"id": template_id} if template_id else {})}
    )


#: The axis a model writes: the document each case extracts, which the toy kind reads by this name.
LLM_AXIS = VariationAxis(name=DOCUMENT_PARAM, generator="llm", description="an invoice id")
#: An axis no model writes.
ENUM_AXIS = VariationAxis(name=DOCUMENT_PARAM, generator="enum", values=list(RUN_DOCUMENTS))


def _generating_host(
    *templates: EvalTemplate, clients: _FakeClients | None = None
) -> tuple[LaunchHost, EvalStorage, _FakeClients, list[LaunchRequest]]:
    """The toy launch host whose launcher generates its cases the way an adopter's does.

    Returns:
        The host, its store, the client factory (to see what was asked for), and every request the
        launcher was handed — so a refusal made before it ran is distinguishable from one it made.
    """
    storage = EvalStorage(InMemoryDocumentStore())
    for template in templates:
        storage.save_template(template)
    clients = clients or _FakeClients()
    host, _client = toyhost_launch_host(storage=storage, clients=clients)
    world = host.eval_host.profile.world
    assert world is not None
    kind = ToyExtractorKind(client=ScriptedExtractionClient(), world=world)
    handed: list[LaunchRequest] = []

    async def launch(request: LaunchRequest) -> EvalRun:
        handed.append(request)

        async def generate() -> Any:
            factory = generating.eval_host.completion_clients("a variation generation")
            if request.variation_model is None:
                return await generate_variations(
                    request.template, request.n_variations, storage=storage, scope_id=request.scope_id
                )
            async with factory("variation", request.variation_model) as writer:
                return await generate_variations(
                    request.template, request.n_variations, storage=storage, scope_id=request.scope_id, llm=writer
                )

        generation = await request.launch_group.resolve_once(
            ("cases", request.template.id, str(request.n_variations)), generate
        )
        return await launch_run(
            generating,
            request,
            KindWiring(
                kind_factory=lambda _cell: kind,
                subject=TOYHOST_SUBJECT,
                test_cases=generation.cases,
                variation_counts=generation.counts,
            ),
        )

    generating = replace(host, kinds={TOY_EXTRACTOR_KIND: LaunchableKind(launch=launch)})
    return generating, storage, clients, handed


async def _settled(host: LaunchHost, run_ids: list[str]) -> None:
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run_id) for run_id in run_ids):
            await asyncio.sleep(0.01)


async def _launch(host: LaunchHost, template: EvalTemplate, **arguments: Any) -> list[EvalRun]:
    launched: dict[str, Any] = {"models": list(RUN_MODELS), **arguments}
    return await start_run(
        host, template_id=template.id, scope_id=TOYHOST_SCOPE, subject_id=TOYHOST_SUBJECT.subject_id, **launched
    )


# =============================================================================
# The writer: asked for in its own role, once per launch, and recorded
# =============================================================================


async def test_a_launch_generates_with_the_named_writer_and_every_arm_records_it():
    template = _template(LLM_AXIS)
    host, storage, clients, _handed = _generating_host(template)

    runs = await _launch(host, template, n_variations=2, variation_model=WRITER)
    await _settled(host, [run.id for run in runs])

    assert clients.asked == [("variation", WRITER)], "one writer per launch, in its own role — never the simulator's"
    first, second = runs
    assert first.test_case_ids == second.test_case_ids, "every arm answers the same generated cases"
    for run in runs:
        stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
        assert stored is not None and stored.status == "completed", stored
        assert stored.variation_counts == VariationCounts(requested=2, kept=2, reused=0, variation_model=WRITER)
        assert stored.simulator_model is None and stored.simulator_request_settings is None
        assert "simulator" not in (stored.model_role_provenance or {})


async def test_a_generation_no_model_writes_records_no_writer_and_asks_for_no_client():
    template = _template(ENUM_AXIS)
    host, storage, clients, _handed = _generating_host(template)

    (run,) = await _launch(host, template, models=[RUN_MODELS[0]], n_variations=2)
    await _settled(host, [run.id])

    assert clients.asked == []
    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.variation_counts is not None
    assert stored.variation_counts.variation_model is None


async def test_the_writer_enters_no_identity():
    """The cases it wrote are what the candidate faced, and they are already hashed through ``test_case_ids``."""
    template = _template(LLM_AXIS)
    host, _storage, _clients, _handed = _generating_host(template)
    (run,) = await _launch(host, template, models=[RUN_MODELS[0]], n_variations=2, variation_model=WRITER)
    await _settled(host, [run.id])
    assert run.variation_counts is not None
    other_writer = run.model_copy(
        update={"variation_counts": run.variation_counts.model_copy(update={"variation_model": "writer-b"})}
    )
    other_cases = run.model_copy(update={"test_case_ids": [*run.test_case_ids, "one-more-case"]})

    profile = host.eval_host.profile
    assert derive_context_identity(other_writer, profile).context_key == run.context_key
    assert derive_context_identity(other_cases, profile).context_key != run.context_key, "the case set IS hashed"


# =============================================================================
# The refusals, each before any spend
# =============================================================================


@pytest.mark.parametrize(
    ("axes", "arguments", "said"),
    [
        ([LLM_AXIS], {"n_variations": 2}, f"axis values written by a model.*without naming one.*{'variation_model'}"),
        ([LLM_AXIS], {"variation_model": WRITER}, "generates none \\(n_variations=0\\)"),
        ([ENUM_AXIS], {"n_variations": 2, "variation_model": WRITER}, "has no llm axis, so no model writes its cases"),
        ([], {"variation_model": WRITER}, "generates none"),
    ],
    ids=["generation-needs-a-writer", "a-writer-for-no-generation", "a-writer-for-no-llm-axis", "a-writer-alone"],
)
async def test_a_writer_the_launch_needs_or_cannot_use_is_refused_before_any_spend(axes, arguments, said):
    template = _template(*axes)
    host, storage, clients, handed = _generating_host(template)

    with pytest.raises(ValidationFailedError, match=said):
        await _launch(host, template, **arguments)

    assert handed == [] and clients.asked == [], "refused before the launcher ran or any client was built"
    assert storage.query_eval_runs(TOYHOST_SCOPE) == [] and storage.query_test_cases(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


async def test_a_generation_recording_another_writer_than_the_one_named_is_refused():
    """A host resolving the named model to another is a launch recording one writer and running another."""
    template = _template(LLM_AXIS)
    host, storage, _clients, _handed = _generating_host(template, clients=_FakeClients(aliases={WRITER: "writer-z"}))

    with pytest.raises(ValueError, match="recorded 'writer-z' as the model that wrote its cases"):
        await _launch(host, template, n_variations=2, variation_model=WRITER)

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


# =============================================================================
# The battery: the writer reaches the templates a model writes, and only those
# =============================================================================


async def _no_preflight(_subject_id: str, _models: Any) -> Any:
    async def check(_template: EvalTemplate, _cassette_mode: str) -> None:
        return None

    return check


async def test_a_battery_hands_the_writer_only_to_the_templates_with_an_llm_axis():
    written = _template(LLM_AXIS, template_id="written", universal=True)
    enumerated = _template(ENUM_AXIS, template_id="enumerated", universal=True)
    host, storage, clients, handed = _generating_host(written, enumerated)

    run_ids = await start_universal_battery(
        host,
        TOYHOST_SUBJECT.subject_id,
        scope_id=TOYHOST_SCOPE,
        models=[RUN_MODELS[0]],
        n_variations=2,
        variation_model=WRITER,
        preflight=_no_preflight,
    )
    await _settled(host, run_ids)

    assert {request.template.id: request.variation_model for request in handed} == {
        "written": WRITER,
        "enumerated": None,
    }
    assert clients.asked == [("variation", WRITER)]
    writers = {}
    for run_id in run_ids:
        stored = storage.load_eval_run(run_id, TOYHOST_SCOPE)
        assert stored is not None and stored.variation_counts is not None
        writers[stored.template_id] = stored.variation_counts.variation_model
    assert writers == {"written": WRITER, "enumerated": None}


async def test_a_battery_refuses_a_writer_none_of_its_templates_uses():
    host, storage, clients, handed = _generating_host(_template(ENUM_AXIS, universal=True))

    with pytest.raises(ValidationFailedError, match="none of the battery's templates has an llm axis.*Nothing was"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            n_variations=2,
            variation_model=WRITER,
            preflight=_no_preflight,
        )

    assert handed == [] and clients.asked == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []


async def test_a_battery_that_needs_a_writer_and_names_none_launches_nothing():
    written = _template(LLM_AXIS, template_id="written", universal=True)
    enumerated = _template(ENUM_AXIS, template_id="enumerated", universal=True)
    host, storage, clients, handed = _generating_host(enumerated, written)

    with pytest.raises(ValidationFailedError, match="without naming one.*Nothing was launched"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            n_variations=2,
            preflight=_no_preflight,
        )

    assert handed == [] and clients.asked == [] and storage.query_eval_runs(TOYHOST_SCOPE) == []


# =============================================================================
# The surfaces: each carries both arguments to the launch
# =============================================================================
#
# The toy host's extractor declines ``n_variations``, so a launch naming either argument is refused
# by the dispatch — which is the evidence the surface handed it on: a surface that dropped the
# argument would launch cleanly instead.


@pytest.mark.parametrize(
    ("arguments", "said"),
    [({"n_variations": 2}, "cannot honour n_variations=2"), ({"variation_model": WRITER}, "generates none")],
    ids=["n_variations", "variation_model"],
)
async def test_run_launch_hands_the_launch_both_generation_arguments(arguments, said):
    fixture = ops_fixture()
    launched = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=[RUN_MODELS[0]], **arguments
    )

    with pytest.raises(ValidationFailedError, match=said):
        await run_launch(fixture.host, launched, TOYHOST_SCOPE)


@pytest.mark.parametrize(
    ("arguments", "said"),
    [({"n_variations": 2}, "cannot honour n_variations=2"), ({"variation_model": WRITER}, "generates none")],
    ids=["n_variations", "variation_model"],
)
async def test_the_run_launch_action_hands_the_launch_both_generation_arguments(arguments, said):
    fixture = ops_fixture()
    evals = eval_catalogue().mount_all(standard_tools())[0]
    call = {
        "action": "run_launch",
        "template_id": toyhost_template().id,
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "models": [RUN_MODELS[0]],
        **arguments,
    }

    outcome = await evals.call(call, host=fixture.host, caller=CALLER)

    assert outcome.is_error and said.split("=")[0] in outcome.text, outcome.text


async def test_launch_estimate_prices_the_cases_a_launch_would_generate():
    fixture = ops_fixture()
    evals = eval_catalogue().mount_all(standard_tools())[0]
    arguments = {"template_id": toyhost_template().id, "models": [RUN_MODELS[0]], "n_variations": 4}

    outcome = await evals.call({"action": "launch_estimate", **arguments}, host=fixture.host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    estimate = CostEstimate.model_validate(outcome.structured)
    assert (estimate.n_test_cases, estimate.n_test_cases_source) == (4, "generated")
    direct = launch_estimate(fixture.host, TOYHOST_SCOPE, **arguments)  # type: ignore[arg-type]
    assert (direct.n_test_cases, direct.n_test_cases_source) == (4, "generated")
