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
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import CostEstimate
from threetears.evals.contracts import EvalRun, EvalStorage, OutOfRunBudget, ValidationFailedError
from threetears.evals.contracts.host import CompletionRole
from threetears.evals.contracts.identity import derive_context_identity
from threetears.evals.contracts.models import EvalTemplate, VariationAxis, VariationCounts
from threetears.evals.gen import generate_variations
from threetears.evals.ops import LaunchArguments, history_launch_pricer, launch_estimate, run_launch
from threetears.evals.run import (
    ArmPlan,
    ArmPrice,
    ArmQuote,
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchPricer,
    LaunchRequest,
    LaunchSettings,
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
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_DOCUMENTS, RUN_MODELS, toyhost_template
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.ops_support import CALLER, ops_fixture

#: The model the launches name to write their cases.
WRITER = "writer-a"

#: What one of the writer's calls is priced at — well inside the toy host's out-of-run cap.
WRITER_CEILING = 0.001


def _priced_at_nothing(_quote: ArmQuote) -> ArmPrice:
    """A launch pricer reading off a rate card: the toy extractor's calls cost nothing."""
    return ArmPrice(predicted_usd=0.0, basis="the toy extractor's scripted rate card")


@dataclass(frozen=True)
class _Completion:
    content: str


# parity-with: threetears.evals.contracts.provider.BoundCompletionClient
@dataclass
class _FakeWriter:
    """A variation writer that answers with the toy host's document ids, naming the model it resolved to."""

    model_name: str
    answer: str = '{"values": ["' + '", "'.join(RUN_DOCUMENTS) + '"]}'
    ceiling: float | None = WRITER_CEILING
    #: A call whose prompt carries this marker is priced at ``dear_ceiling`` instead — one template's
    #: prompt made dearer than another's, as a longer existing-case block or description makes it.
    dear_marker: str | None = None
    dear_ceiling: float = 1000.0
    calls: int = 0

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        if self.dear_marker is not None and self.dear_marker in user:
            return self.dear_ceiling
        return self.ceiling

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
    ceiling: float | None = WRITER_CEILING
    dear_marker: str | None = None
    asked: list[tuple[CompletionRole, str | None]] = field(default_factory=list)
    writers: list[_FakeWriter] = field(default_factory=list)

    def __call__(self, role: CompletionRole, model: str | None, *, temperature: float | None = None) -> _FakeWriter:
        self.asked.append((role, model))
        assert model is not None, "a variation writer is always named"
        writer = _FakeWriter(
            model_name=self.aliases.get(model, model), ceiling=self.ceiling, dear_marker=self.dear_marker
        )
        self.writers.append(writer)
        return writer


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
    *templates: EvalTemplate,
    clients: _FakeClients | None = None,
    pricer: LaunchPricer | None = _priced_at_nothing,
    budgeted: bool = True,
    plan_cases: int | None = None,
    plan_model: str | None = None,
    default_model: str | None = None,
    settings: Callable[[], LaunchSettings] | None = None,
) -> tuple[LaunchHost, EvalStorage, _FakeClients, list[LaunchRequest]]:
    """The toy launch host whose launcher generates its cases the way an adopter's does.

    ``pricer`` is the host's launch pricer (``None`` for a host that prices no launch); ``budgeted=False``
    makes the launcher hand its generation a budget of its own rather than the request's; ``plan_cases``
    and ``plan_model`` make the kind's arm plan state another case count or model than the launch's;
    ``default_model`` is the model the launcher runs an arm naming none on; ``settings`` reads the host's
    launch settings (the toy host's own when ``None``) — the launcher's tail reads through the same host.

    Returns:
        The host, its store, the client factory (to see what was asked for), and every request the
        launcher was handed — so a refusal made before it ran is distinguishable from one it made.
    """
    storage = EvalStorage(InMemoryDocumentStore())
    for template in templates:
        storage.save_template(template)
    clients = clients or _FakeClients()
    host, _client = (
        toyhost_launch_host(storage=storage, clients=clients)
        if settings is None
        else toyhost_launch_host(storage=storage, clients=clients, settings=settings)
    )
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
            budget = request.generation_budget
            if not budgeted:
                budget = OutOfRunBudget(storage, scope_id=request.scope_id, cap_usd=None)
            async with factory("variation", request.variation_model) as writer:
                return await generate_variations(
                    request.template,
                    request.n_variations,
                    storage=storage,
                    scope_id=request.scope_id,
                    llm=writer,
                    budget=budget,
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
                default_candidate_model=default_model if request.candidate_model is None else None,
            ),
        )

    planned: list[LaunchRequest] = []

    def plan_arm(request: LaunchRequest) -> ArmPlan:
        planned.append(request)
        return ArmPlan(
            case_count=plan_cases if plan_cases is not None else request.n_variations,
            candidate_model=plan_model or request.candidate_model or RUN_MODELS[0],
        )

    generating = replace(
        host, kinds={TOY_EXTRACTOR_KIND: LaunchableKind(launch=launch, plan_arm=plan_arm)}, launch_pricer=pricer
    )
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
    # Twice, both in the writer's own role: once by the battery to price the written template's calls
    # before anything launched — asked its prices and never called — and once by that template's launch.
    assert clients.asked == [("variation", WRITER), ("variation", WRITER)]
    assert [writer.calls for writer in clients.writers] == [0, 1]
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


# =============================================================================
# A generating launch is priced before it pays for anything
# =============================================================================
#
# The generation runs inside the kind's launcher and before any run exists, so it is outside every
# run's cost cap. Two checks stand in front of it: every ARM is planned by its kind and priced by the
# host's pricer against the cap its run will be held to — before the launcher is called — and the
# generation's own CALLS are priced on the writer's client against the host's out-of-run cap before
# the first is made, then ledgered under the launch's group.


def _nothing_was_paid_for(host: LaunchHost, storage: EvalStorage, clients: _FakeClients) -> None:
    """No run, no case, no ledgered call, no writer call, and no admission left held."""
    assert storage.query_eval_runs(TOYHOST_SCOPE) == [] and storage.query_test_cases(TOYHOST_SCOPE) == []
    assert storage.query_out_of_run_spend(TOYHOST_SCOPE) == []
    assert all(writer.calls == 0 for writer in clients.writers)
    assert host.job_manager.admitted_count == 0


async def test_a_launchs_generation_is_ledgered_once_under_its_group_at_the_out_of_run_cap():
    template = _template(LLM_AXIS)
    host, storage, clients, _handed = _generating_host(template)

    runs = await _launch(host, template, n_variations=2, variation_model=WRITER)
    await _settled(host, [run.id for run in runs])

    group = {run.launch_group_id for run in runs}
    assert len(group) == 1 and None not in group
    [spend] = storage.query_out_of_run_spend(TOYHOST_SCOPE)
    assert spend.launch_group_id == runs[0].launch_group_id, "one generation for every arm, ledgered once"
    assert (spend.purpose, spend.model, spend.template_id, spend.subject_id) == (
        "variation",
        WRITER,
        template.id,
        TOYHOST_SUBJECT.subject_id,
    )
    assert (spend.priced_ceiling_usd, spend.cap_usd) == (WRITER_CEILING, host.settings().max_out_of_run_cost_usd)
    assert [writer.calls for writer in clients.writers] == [1]


async def test_a_generation_priced_above_the_out_of_run_cap_is_refused_before_its_call():
    template = _template(LLM_AXIS)
    host, storage, clients, _handed = _generating_host(template, clients=_FakeClients(ceiling=1000.0))

    with pytest.raises(ValidationFailedError, match="above the out-of-run cap.*Nothing was called"):
        await _launch(host, template, n_variations=2, variation_model=WRITER)

    assert clients.asked == [("variation", WRITER)], "the writer was built, and priced, and never called"
    _nothing_was_paid_for(host, storage, clients)


async def test_a_generation_its_writer_cannot_price_is_refused_under_an_enforced_cap():
    template = _template(LLM_AXIS)
    host, storage, clients, _handed = _generating_host(template, clients=_FakeClients(ceiling=None))

    with pytest.raises(ValidationFailedError, match="cannot be priced before they are made"):
        await _launch(host, template, n_variations=2, variation_model=WRITER)

    _nothing_was_paid_for(host, storage, clients)


async def test_with_enforcement_off_a_generation_is_admitted_unpriced_and_still_ledgered():
    template = _template(LLM_AXIS)
    host, storage, clients, _handed = _generating_host(template, clients=_FakeClients(ceiling=None), pricer=None)
    unenforced = host.settings().model_copy(update={"enforcement_enabled": False})
    host = replace(host, settings=lambda: unenforced)

    runs = await _launch(host, template, n_variations=2, variation_model=WRITER)
    await _settled(host, [run.id for run in runs])

    [spend] = storage.query_out_of_run_spend(TOYHOST_SCOPE)
    assert (spend.priced_ceiling_usd, spend.cap_usd) == (None, None)


async def test_a_generation_ledgered_outside_the_requests_budget_is_refused_at_the_tail():
    """A launcher that builds its own budget leaves the launch's cap unenforced and its ledger without the call."""
    template = _template(LLM_AXIS)
    host, storage, _clients, _handed = _generating_host(template, budgeted=False)

    with pytest.raises(ValueError, match="generation budget ledgered no call"):
        await _launch(host, template, n_variations=2, variation_model=WRITER)

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


async def test_an_arm_predicted_above_its_cap_is_refused_before_the_launcher_or_the_generation():
    template = _template(LLM_AXIS)
    quotes: list[ArmQuote] = []

    def dear(quote: ArmQuote) -> ArmPrice:
        quotes.append(quote)
        return ArmPrice(predicted_usd=9.5, basis="from 3 past results")

    host, storage, clients, handed = _generating_host(template, pricer=dear)

    with pytest.raises(
        ValidationFailedError, match=r"predicted to cost \$9\.50 \(from 3 past results\), above its \$4\.00 cap"
    ):
        await _launch(host, template, n_variations=2, variation_model=WRITER, k_runs=3)

    assert handed == [] and clients.asked == [], "refused before the launcher ran, so before the writer was built"
    _nothing_was_paid_for(host, storage, clients)
    assert [(q.candidate_model, q.case_count, q.k_runs, q.template_id) for q in quotes][:1] == [
        (RUN_MODELS[0], 2, 3, template.id)
    ], "priced at the arm's plan: its model, its planned cases and the launch's repeats"


async def test_an_unpredicted_arm_is_refused_under_an_inherited_cap_and_runs_under_a_chosen_one():
    template = _template(LLM_AXIS)

    def unknown(_quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=None, basis="no priced result of this template on that model")

    host, storage, clients, handed = _generating_host(template, pricer=unknown)

    with pytest.raises(ValidationFailedError, match="cannot be priced: no priced result.*unknown, not \\$0.*inherited"):
        await _launch(host, template, n_variations=2, variation_model=WRITER)
    assert handed == []
    _nothing_was_paid_for(host, storage, clients)

    runs = await _launch(host, template, n_variations=2, variation_model=WRITER, max_cost_usd=1.5)
    await _settled(host, [run.id for run in runs])
    assert all(run.max_cost_usd == 1.5 and run.max_cost_usd_origin == "chosen" for run in runs)


async def test_a_generating_arm_on_a_host_that_prices_no_launch_is_unpriceable():
    """No pricer is one more way an arm is unknown, read by the one rule: refused under an inherited cap only."""
    template = _template(LLM_AXIS)
    host, storage, clients, handed = _generating_host(template, pricer=None)

    with pytest.raises(
        ValidationFailedError,
        match=(
            "cannot be priced: host 'toyhost' prices no launch \\(LaunchHost.launch_pricer\\).*inherited.*"
            "Refused before the generation was paid for"
        ),
    ):
        await _launch(host, template, n_variations=2, variation_model=WRITER)
    assert handed == []
    _nothing_was_paid_for(host, storage, clients)

    runs = await _launch(host, template, n_variations=2, variation_model=WRITER, max_cost_usd=1.5)
    await _settled(host, [run.id for run in runs])
    assert all(run.max_cost_usd_origin == "chosen" for run in runs)


async def test_a_generating_arm_of_a_kind_that_plans_nothing_is_unpriceable():
    """Nothing says what the arm will run, so nothing prices it — refused under an inherited cap, run under a chosen one."""
    template = _template(ENUM_AXIS)
    asked: list[ArmQuote] = []

    def recording(quote: ArmQuote) -> ArmPrice:
        asked.append(quote)
        return ArmPrice(predicted_usd=0.0, basis="a rate card")

    host, storage, clients, handed = _generating_host(template, pricer=recording)
    (launchable,) = host.kinds.values()
    unplanned = replace(host, kinds={TOY_EXTRACTOR_KIND: replace(launchable, plan_arm=None)})

    with pytest.raises(ValidationFailedError, match="plans no arm \\(LaunchableKind.plan_arm\\).*inherited"):
        await _launch(unplanned, template, n_variations=2)
    assert handed == []
    _nothing_was_paid_for(unplanned, storage, clients)

    runs = await _launch(unplanned, template, n_variations=2, max_cost_usd=1.5)
    await _settled(unplanned, [run.id for run in runs])
    assert asked == [], "an arm nothing planned is never quoted"
    assert all(run.max_cost_usd_origin == "chosen" for run in runs)


@pytest.mark.parametrize(
    ("plan", "said"),
    [
        ({"plan_cases": 1}, "planned this arm at most 1 case\\(s\\) and its launcher froze 2"),
        ({"plan_model": "another-model"}, "planned an arm the launch named 'extractor-v2' on 'another-model'"),
    ],
    ids=["more-cases-than-planned", "another-model-than-named"],
)
async def test_an_arm_that_runs_other_than_it_was_priced_is_refused(plan, said):
    template = _template(ENUM_AXIS)
    host, storage, _clients, _handed = _generating_host(template, **plan)

    with pytest.raises(ValueError, match=said):
        await _launch(host, template, models=[RUN_MODELS[0]], n_variations=2)

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


@pytest.mark.parametrize(
    ("bad", "said"),
    [
        ({"case_count": 0, "candidate_model": "m"}, "at least one case"),
        ({"case_count": 1, "candidate_model": " "}, "candidate_model is blank"),
    ],
)
def test_an_arm_plan_refuses_no_cases_and_no_model(bad, said):
    with pytest.raises(ValueError, match=said):
        ArmPlan(**bad)


@pytest.mark.parametrize(
    ("bad", "said"),
    [
        ({"predicted_usd": -1.0, "basis": "b"}, "0 or more"),
        ({"predicted_usd": float("nan"), "basis": "b"}, "0 or more"),
        ({"predicted_usd": 1.0, "basis": "  "}, "basis is blank"),
    ],
)
def test_an_arm_price_refuses_a_negative_or_unexplained_prediction(bad, said):
    with pytest.raises(ValueError, match=said):
        ArmPrice(**bad)


async def test_a_battery_prices_every_templates_generating_arms_before_launching_any():
    """The third template's arm over its cap used to surface only after the first two were generated and paid for."""
    cheap = _template(ENUM_AXIS, template_id="cheap", universal=True)
    dear = _template(ENUM_AXIS, template_id="dear", universal=True)

    def by_template(quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=50.0 if quote.template_id == "dear" else 0.0, basis="a rate card")

    host, storage, clients, handed = _generating_host(cheap, dear, pricer=by_template)

    with pytest.raises(ValidationFailedError, match="predicted to cost \\$50\\.00.*Nothing was launched"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            n_variations=2,
            preflight=_no_preflight,
        )

    assert handed == []
    _nothing_was_paid_for(host, storage, clients)


async def test_a_generating_battery_prices_each_arm_once():
    """The pre-flight prices each arm and the template's launch carries that plan — the launch never prices it again."""
    first = _template(ENUM_AXIS, template_id="first", universal=True)
    second = _template(ENUM_AXIS, template_id="second", universal=True)
    quotes: list[ArmQuote] = []

    def recording(quote: ArmQuote) -> ArmPrice:
        quotes.append(quote)
        return ArmPrice(predicted_usd=0.0, basis="a rate card")

    host, _storage, _clients, handed = _generating_host(first, second, pricer=recording)

    run_ids = await start_universal_battery(
        host,
        TOYHOST_SUBJECT.subject_id,
        scope_id=TOYHOST_SCOPE,
        models=list(RUN_MODELS),
        n_variations=2,
        preflight=_no_preflight,
    )
    await _settled(host, run_ids)

    assert sorted((q.template_id, q.candidate_model, q.case_source) for q in quotes) == sorted(
        (template, model, "generated") for template in ("first", "second") for model in RUN_MODELS
    )
    assert len(handed) == 4 and all(request.arm_plan is not None for request in handed)


async def test_a_battery_prices_every_templates_generation_calls_before_launching_any():
    """The second template's writer calls over the out-of-run cap used to surface inside its own launch,
    after the first template had paid for its generation and started its runs — whose ids were then lost."""
    cheap = _template(LLM_AXIS, template_id="cheap", universal=True)
    dear_axis = LLM_AXIS.model_copy(update={"description": "an invoice id, DEAR to write"})
    dear = _template(dear_axis, template_id="dear", universal=True)
    host, storage, clients, handed = _generating_host(cheap, dear, clients=_FakeClients(dear_marker="DEAR"))
    templates = host.eval_host.storage.query_templates(TOYHOST_SCOPE, universal=True, archived=False)
    assert [template.id for template in templates] == ["cheap", "dear"], "the cheap template would launch first"

    with pytest.raises(ValidationFailedError, match="above the out-of-run cap.*Nothing was launched"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            n_variations=2,
            variation_model=WRITER,
            preflight=_no_preflight,
        )

    assert handed == [], "no template's launcher ran"
    assert clients.asked == [("variation", WRITER)], "one writer, built to be priced"
    _nothing_was_paid_for(host, storage, clients)


async def test_a_battery_prices_a_generation_its_writer_cannot_price_before_launching_any():
    written = _template(LLM_AXIS, template_id="written", universal=True)
    host, storage, clients, handed = _generating_host(written, clients=_FakeClients(ceiling=None))

    with pytest.raises(ValidationFailedError, match="cannot be priced before they are made.*Nothing was launched"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            n_variations=2,
            variation_model=WRITER,
            preflight=_no_preflight,
        )

    assert handed == []
    _nothing_was_paid_for(host, storage, clients)


async def test_a_battery_names_the_cap_its_runs_are_priced_against_and_held_to():
    """Without it every run inherits the host's cap, and an arm no history prices is refused — with no way out."""
    template = _template(ENUM_AXIS, template_id="unpriced", universal=True)

    def unknown(_quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=None, basis="no priced result of this template on that model")

    host, storage, clients, handed = _generating_host(template, pricer=unknown)
    battery = {
        "scope_id": TOYHOST_SCOPE,
        "models": [RUN_MODELS[0]],
        "n_variations": 2,
        "preflight": _no_preflight,
    }

    with pytest.raises(ValidationFailedError, match="cannot be priced.*inherited.*Nothing was launched"):
        await start_universal_battery(host, TOYHOST_SUBJECT.subject_id, **battery)
    assert handed == []

    run_ids = await start_universal_battery(host, TOYHOST_SUBJECT.subject_id, max_cost_usd=1.5, **battery)
    await _settled(host, run_ids)
    stored = [storage.load_eval_run(run_id, TOYHOST_SCOPE) for run_id in run_ids]
    assert [(run.max_cost_usd, run.max_cost_usd_origin) for run in stored if run is not None] == [(1.5, "chosen")]


@pytest.mark.parametrize("cap", [0.0, -1.0])
async def test_a_battery_naming_a_cap_that_is_not_positive_is_refused_before_anything(cap):
    host, storage, clients, handed = _generating_host(_template(ENUM_AXIS, universal=True))

    with pytest.raises(ValidationFailedError, match=f"max_cost_usd must be > 0 \\(got {cap}\\).*Nothing was launched"):
        await start_universal_battery(
            host,
            TOYHOST_SUBJECT.subject_id,
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            max_cost_usd=cap,
            preflight=_no_preflight,
        )

    assert handed == [] and host.job_manager.admitted_count == 0


async def test_an_arm_its_launcher_runs_on_another_model_than_its_plan_is_refused_at_the_tail():
    """An arm naming no model is planned on one and run on the launcher's default: priced as one, run as another."""
    template = _template(ENUM_AXIS)
    host, storage, _clients, handed = _generating_host(template, plan_model="planned-model", default_model="ran-model")

    with pytest.raises(ValueError, match="planned this arm on 'planned-model' and its launcher ran it on 'ran-model'"):
        await _launch(host, template, models=[], n_variations=2)

    assert len(handed) == 1, "the launcher ran; the tail refused what it wired"
    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0


class _MovingSettings:
    """The host's settings as a hot reload moves them: every read after the first declares no metered tools."""

    def __init__(self, first: LaunchSettings) -> None:
        self.first = first
        self.reads = 0

    def __call__(self) -> LaunchSettings:
        self.reads += 1
        return self.first if self.reads == 1 else self.first.model_copy(update={"max_metered_calls": None})


async def test_a_launch_reads_the_hosts_settings_once_so_a_reload_cannot_refuse_it_after_its_generation():
    """A launch naming a metered-call ceiling was refused at its tail, generation paid, when a reload landed mid-launch."""
    template = _template(LLM_AXIS)
    moving = _MovingSettings(TOYHOST_LAUNCH_SETTINGS)
    host, storage, clients, _handed = _generating_host(template, settings=moving)

    runs = await _launch(host, template, n_variations=2, variation_model=WRITER, max_metered_calls=5)
    await _settled(host, [run.id for run in runs])

    assert moving.reads == 1
    assert all(run.max_metered_calls == 5 and run.max_metered_calls_origin == "chosen" for run in runs)


async def test_a_battery_reads_the_hosts_settings_once_for_every_template():
    first = _template(LLM_AXIS, template_id="first", universal=True)
    second = _template(LLM_AXIS, template_id="second", universal=True)
    moving = _MovingSettings(TOYHOST_LAUNCH_SETTINGS)
    host, _storage, _clients, _handed = _generating_host(first, second, settings=moving)

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

    assert moving.reads == 1 and len(run_ids) == 2


# =============================================================================
# The engine's own launch pricer: the scope's usage history of the template on the model
# =============================================================================


def _history(
    storage: EvalStorage, *, template_id: str, model: str, costs: list[float | None], **run_fields: Any
) -> None:
    run = make_eval_run(scope_id=TOYHOST_SCOPE, template_id=template_id, candidate_model=model, **run_fields)
    storage.save_eval_run(run)
    for k, cost in enumerate(costs, start=1):
        storage.save_eval_result(
            make_eval_result(scope_id=TOYHOST_SCOPE, eval_run_id=run.id, model=model, k_iteration=k, cost_usd=cost)
        )


def _quote(**overrides: Any) -> ArmQuote:
    fields: dict[str, Any] = {
        "scope_id": TOYHOST_SCOPE,
        "template_id": "tpl-priced",
        "subject_id": TOYHOST_SUBJECT.subject_id,
        "candidate_model": "m-priced",
        "k_runs": 2,
        "case_count": 5,
        "n_variations": 5,
        "cassette_mode": "off",
        "judge_model": None,
        "simulator_model": None,
        "apparatus_settings": {},
        **overrides,
    }
    return ArmQuote(**fields)


def test_the_history_pricer_bounds_an_arm_by_the_upper_band_of_its_template_and_models_past_results():
    """The launch holds this figure to the cap, so it is the band's upper end, not the centre."""
    host, storage = _priced_host()
    _history(storage, template_id="tpl-priced", model="m-priced", costs=[0.10, 0.20, 0.30])
    _history(storage, template_id="another-template", model="m-priced", costs=[9.0, 9.0, 9.0])

    price = history_launch_pricer(host)(_quote())

    centre = 0.20 * 5 * 2
    assert price.predicted_usd is not None and price.predicted_usd > centre, "above the mean x cases x repeats"
    assert f"around ${centre:.2f}" in price.basis and "upper end of the band" in price.basis
    assert "usage-history" in price.basis and "3 past result(s)" in price.basis
    assert price.predicted_usd < 9.0, "another template's history is not drawn on"


def test_the_history_pricer_prices_an_arm_over_stored_cases_by_the_same_rule():
    """One rule for both sources: the same history, cases and repeats bound a stored-case arm exactly as a generating one."""
    host, storage = _priced_host()
    _history(storage, template_id="tpl-priced", model="m-priced", costs=[0.10, 0.20, 0.30])
    pricer = history_launch_pricer(host)

    stored, generated = pricer(_quote(n_variations=0)), pricer(_quote(n_variations=5))

    assert _quote(n_variations=0).case_source == "stored" and _quote().case_source == "generated"
    assert stored.predicted_usd is not None and stored.predicted_usd == generated.predicted_usd
    assert "upper end of the band" in stored.basis


def test_the_history_pricer_predicts_nothing_for_an_arm_with_no_priced_history():
    host, storage = _priced_host()
    _history(storage, template_id="tpl-priced", model="m-priced", costs=[None])

    price = history_launch_pricer(host)(_quote())

    assert price.predicted_usd is None
    assert "no priced result of template 'tpl-priced' on 'm-priced'" in price.basis
    assert "1 past result(s) ran unpriced" in price.basis


def test_the_history_pricer_predicts_nothing_from_a_history_too_thin_to_bound():
    """Two observations give a mean and no band; a mean is not a bound, so the arm is unknown."""
    host, storage = _priced_host()
    _history(storage, template_id="tpl-priced", model="m-priced", costs=[0.10, 0.20])

    price = history_launch_pricer(host)(_quote())

    assert price.predicted_usd is None
    assert "2 priced past result(s)" in price.basis and "too few to bound the arm" in price.basis


_JUDGED_CHEAPLY = {"judge_model": "judge-cheap", "model_role_provenance": {"judge": "chosen"}}
_SIMULATED_CHEAPLY = {"simulator_model": "sim-cheap", "model_role_provenance": {"simulator": "chosen"}}


@pytest.mark.parametrize(
    ("history", "other", "same"),
    [
        (_JUDGED_CHEAPLY, {"judge_model": "judge-dear"}, {"judge_model": "judge-cheap"}),
        (_JUDGED_CHEAPLY, {}, {"judge_model": "judge-cheap"}),
        (_SIMULATED_CHEAPLY, {"simulator_model": "sim-dear"}, {"simulator_model": "sim-cheap"}),
        (
            {"apparatus_settings": {"reviewer_pool": "pool-a"}},
            {"apparatus_settings": {"reviewer_pool": "pool-b"}},
            {"apparatus_settings": {"reviewer_pool": "pool-a"}},
        ),
    ],
    ids=["another-judge-pin", "a-pinned-judge-for-an-arm-naming-none", "another-simulator-pin", "another-rig"],
)
def test_the_history_pricer_draws_only_on_runs_launched_as_the_arm_will_be(history, other, same):
    """A run judged by a cheaper model, or with its rig set up otherwise, spent differently from the arm."""
    host, storage = _priced_host()
    _history(storage, template_id="tpl-priced", model="m-priced", costs=[0.10, 0.20, 0.30], **history)
    pricer = history_launch_pricer(host)

    assert pricer(_quote(**other)).predicted_usd is None
    assert pricer(_quote(**same)).predicted_usd is not None, "the condition it was launched under prices"


def test_a_run_that_inherited_its_judge_prices_an_arm_naming_none():
    host, storage = _priced_host()
    _history(
        storage,
        template_id="tpl-priced",
        model="m-priced",
        costs=[0.10, 0.20, 0.30],
        judge_model="judge-default",
        model_role_provenance={"judge": "inherited"},
    )

    assert history_launch_pricer(host)(_quote()).predicted_usd is not None


def test_the_history_pricer_reads_only_the_matching_runs_results(monkeypatch: pytest.MonkeyPatch):
    """Not every result in the scope for every arm: a scope's history grows without bound."""
    host, storage = _priced_host()
    _history(storage, template_id="tpl-priced", model="m-priced", costs=[0.10, 0.20, 0.30])
    _history(storage, template_id="another-template", model="m-priced", costs=[9.0, 9.0, 9.0])
    read: list[str | None] = []
    query = storage.query_eval_results

    def recorded(scope_id: str, **filters: Any) -> Any:
        read.append(filters.get("run_id"))
        return query(scope_id, **filters)

    monkeypatch.setattr(storage, "query_eval_results", recorded)

    assert history_launch_pricer(host)(_quote()).predicted_usd is not None
    assert None not in read, "the pricer scanned every result in the scope"
    assert len(read) == 1, "the other template's run was never read"


def _priced_host() -> tuple[Any, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    host, _client = toyhost_launch_host(storage=storage)
    return host.eval_host, storage
